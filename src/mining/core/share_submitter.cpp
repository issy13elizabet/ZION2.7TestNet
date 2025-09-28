#include "share_submitter.h"
#include "../zion-gpu-miner.h" // (if needed later)
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstdlib>

#include "../../network/pool-connection.h"
#include "../../network/stratum_client.h"

ZionShareSubmitter::ZionShareSubmitter(PoolConnection* pool, StratumClient* stratum, ZionMiningStatsAggregator* stats) : pool_(pool), stratum_(stratum), stats_(stats) {}
ZionShareSubmitter::~ZionShareSubmitter(){ stop(); }

void ZionShareSubmitter::start(){ if(running_.exchange(true)) return; thread_ = std::thread(&ZionShareSubmitter::run, this);} 
void ZionShareSubmitter::stop(){ running_.store(false); cv_.notify_all(); if(thread_.joinable()) thread_.join(); }

void ZionShareSubmitter::enqueue(const ZionShare& share){
    // Fast path: try ring buffer (single producer assumption may be violated if multiple CPU threads call enqueue; we'll still attempt using CAS on slot)
    size_t h = head_.load(std::memory_order_relaxed);
    for(int attempt=0; attempt<2; ++attempt){
        size_t pos = h & (RING_CAP - 1);
        RingSlot& slot = ring_[pos];
        bool expected = false;
        if(slot.full.compare_exchange_strong(expected, true, std::memory_order_acq_rel, std::memory_order_relaxed)){
            slot.val = share;
            head_.store(h+1, std::memory_order_release);
            if(stats_) stats_->record_share_submitted(share.nonce, share.difficulty, share.job_id);
            cv_.notify_one();
            return;
        }
        h = head_.load(std::memory_order_relaxed); // retry
    }
    // Fallback legacy queue (contention or full ring)
    {
        std::lock_guard<std::mutex> lk(mutex_);
        queue_.push(share);
    }
    cv_.notify_one();
    if(stats_) stats_->record_share_submitted(share.nonce, share.difficulty, share.job_id);
}

void ZionShareSubmitter::run(){
    while(running_.load()){
        ZionShare s; bool have=false;
        // First attempt lock-free consume
        size_t t = tail_.load(std::memory_order_relaxed);
        size_t pos = t & (RING_CAP - 1);
        RingSlot& slot = ring_[pos];
        if(slot.full.load(std::memory_order_acquire)){
            s = slot.val;
            slot.full.store(false, std::memory_order_release);
            tail_.store(t+1, std::memory_order_release);
            have = true;
        }
        if(!have){
            // Fallback to mutex queue if empty
            std::unique_lock<std::mutex> lk(mutex_);
            if(queue_.empty()){
                cv_.wait_for(lk, std::chrono::milliseconds(50), [&]{ return !running_.load() || !queue_.empty(); });
            }
            if(!running_.load()) break;
            if(!queue_.empty()){
                s = queue_.front(); queue_.pop(); have=true;
            }
        }
        if(!have) continue;
        bool ok=false;
        if(stratum_ && stratum_->running()){
            stratum_->submit_share(s.job_id, s.nonce, s.result_hex, s.difficulty);
            ok=true;
        } else if(pool_ && pool_->is_connected()){
            ok = pool_->submit_share(s.nonce, 0);
        }
        if(ok){
            accepted_.fetch_add(1);
            if(stats_) stats_->record_accepted();
            std::cout << "✅ Share submitted to pool (job=" << s.job_id << ", nonce=" << std::hex << s.nonce << std::dec << ", diff=" << s.difficulty << ")\n";
            if(result_cb_) result_cb_(s, true);
        } else {
            rejected_.fetch_add(1);
            if(stats_) stats_->record_rejected(s.nonce, s.difficulty, s.job_id);
            std::cout << "❌ Share submission failed (job=" << s.job_id << ", nonce=" << std::hex << s.nonce << std::dec << ")\n";
            if(result_cb_) result_cb_(s, false);
        }
    }
}
