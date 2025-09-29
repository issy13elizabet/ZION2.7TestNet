#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <vector>
#include <memory>

class StratumClient; // forward

struct ZionShare {
    std::string job_id;
    uint32_t nonce{0};
    std::string result_hex; // truncated hash hex (or full)
    uint64_t difficulty{0};
};

class ZionShareSubmitter {
public:
    explicit ZionShareSubmitter(StratumClient* sc): stratum_(sc) {}
    ~ZionShareSubmitter(){ stop(); }

    void start(){ if(running_.exchange(true)) return; thread_ = std::thread(&ZionShareSubmitter::run,this); }
    void stop(){ running_.store(false); cv_.notify_all(); if(thread_.joinable()) thread_.join(); }
    void set_stratum(StratumClient* s){ stratum_=s; }

    void enqueue(const ZionShare& share){
        // lock-free ring attempt
        size_t h = head_.load(std::memory_order_relaxed);
        for(int attempt=0; attempt<2; ++attempt){
            size_t pos = h & (RING_CAP - 1);
            Slot& slot = ring_[pos];
            bool expected=false;
            if(slot.full.compare_exchange_strong(expected, true, std::memory_order_acq_rel, std::memory_order_relaxed)){
                slot.val = share;
                head_.store(h+1, std::memory_order_release);
                cv_.notify_one();
                return;
            }
            h = head_.load(std::memory_order_relaxed);
        }
        {
            std::lock_guard<std::mutex> lk(mutex_);
            fallback_.push_back(share);
        }
        cv_.notify_one();
    }

    uint64_t accepted() const { return accepted_.load(); }
    uint64_t rejected() const { return rejected_.load(); }
    size_t queue_depth() const { return head_.load() - tail_.load(); }
    void reset(){ accepted_.store(0); rejected_.store(0); head_.store(0); tail_.store(0); for(size_t i=0;i<RING_CAP;++i){ ring_[i].full.store(false); } std::lock_guard<std::mutex> lk(mutex_); fallback_.clear(); }

private:
    void run();
    struct Slot { std::atomic<bool> full{false}; ZionShare val; };
    static constexpr size_t RING_CAP = 2048; // power of two
    std::unique_ptr<Slot[]> ring_ = std::unique_ptr<Slot[]>(new Slot[RING_CAP]);
    std::atomic<size_t> head_{0};
    std::atomic<size_t> tail_{0};
    std::vector<ZionShare> fallback_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> accepted_{0};
    std::atomic<uint64_t> rejected_{0};
    StratumClient* stratum_{nullptr};
};
