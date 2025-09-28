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
    {
        std::lock_guard<std::mutex> lk(mutex_);
        queue_.push(share);
    }
    cv_.notify_one();
    if(stats_) stats_->record_share_submitted(share.nonce, share.difficulty, share.job_id);
}

void ZionShareSubmitter::run(){
    while(running_.load()){
        ZionShare s;
        {
            std::unique_lock<std::mutex> lk(mutex_);
            cv_.wait(lk, [&]{ return !running_.load() || !queue_.empty(); });
            if(!running_.load()) break;
            if(queue_.empty()) continue;
            s = queue_.front();
            queue_.pop();
        }
        bool ok = false;
        if(stratum_ && stratum_->running()){
            stratum_->submit_share(s.job_id, s.nonce, s.result_hex, s.difficulty);
            ok = true; // real pool submission - actual response handled by Stratum client
        } else if(pool_ && pool_->is_connected()){
            ok = pool_->submit_share(s.nonce, 0 /* placeholder */);
        }
        
        // Real blockchain submission - no simulation!
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
