#pragma once
#include <vector>
#include <mutex>
#include <cstdint>
#include <atomic>
#include <chrono>
#include <string>
#include <deque>
#include <memory>
#include <cmath>

struct ZionShareEventRecord {
    std::chrono::steady_clock::time_point ts;
    bool accepted; // true if accepted (or submitted OK), false if rejected
    uint32_t nonce;
    uint64_t difficulty;
    std::string job_id;
};

struct ZionMiningSnapshot {
    double total_hashrate{0.0};
    std::vector<double> per_thread_hashrate; // size = threads
    uint64_t shares_found{0}; // submitted shares
    uint64_t shares_accepted{0};
    uint64_t shares_rejected{0};
    std::vector<ZionShareEventRecord> recent_events; // last N (trimmed)
    // New benchmarking metrics
    double avg_thread_hashrate{0.0};
    double best_thread_hashrate{0.0};
    double stdev_thread_hashrate{0.0};
    double baseline_total_hashrate{0.0};
    uint64_t baseline_window_seconds{0};
};

class ZionMiningStatsAggregator {
public:
    explicit ZionMiningStatsAggregator(unsigned threads) : thread_count_(threads) { 
        per_thread_hs_ = std::make_unique<std::atomic<double>[]>(threads);
        for(unsigned i = 0; i < threads; ++i) per_thread_hs_[i].store(0.0);
    }
    void update_thread_hashrate(unsigned idx, double hs){
        if(idx >= thread_count_) return; // guard
        per_thread_hs_[idx].store(hs, std::memory_order_relaxed);
    }
    void record_share_submitted(uint32_t nonce, uint64_t diff, const std::string& job){
        shares_found_.fetch_add(1, std::memory_order_relaxed);
        push_event(true, nonce, diff, job); // tentative accepted (pool side ack may adjust later if needed)
    }
    void record_accepted(){ shares_accepted_.fetch_add(1, std::memory_order_relaxed); }
    void record_rejected(uint32_t nonce, uint64_t diff, const std::string& job){
        shares_rejected_.fetch_add(1, std::memory_order_relaxed);
        push_event(false, nonce, diff, job);
    }
    void note_hashrate_sample(){
        // accumulate sliding window stats
        auto now = std::chrono::steady_clock::now();
        if(first_sample_ts_ == std::chrono::steady_clock::time_point{}) first_sample_ts_ = now;
        last_sample_ts_ = now;
    }
    void set_baseline_if_needed(){
        if(!baseline_set_){
            auto snap = snapshot();
            baseline_total_hashrate_.store(snap.total_hashrate, std::memory_order_relaxed);
            baseline_set_ = true;
            baseline_start_ = std::chrono::steady_clock::now();
        }
    }
    ZionMiningSnapshot snapshot() const {
        ZionMiningSnapshot snap;
        double total=0.0;
        snap.per_thread_hashrate.reserve(thread_count_);
        double best=0.0; double sum=0.0; double sum2=0.0; 
        for(unsigned i = 0; i < thread_count_; ++i) {
            double v = per_thread_hs_[i].load(std::memory_order_relaxed);
            snap.per_thread_hashrate.push_back(v);
            total += v; sum += v; sum2 += v*v; if(v>best) best=v;
        }
        snap.total_hashrate = total;
        snap.avg_thread_hashrate = thread_count_? (sum / thread_count_) : 0.0;
        snap.best_thread_hashrate = best;
        if(thread_count_>1){
            double mean = snap.avg_thread_hashrate; double variance = (sum2/thread_count_) - (mean*mean); if(variance < 0) variance = 0; snap.stdev_thread_hashrate = std::sqrt(variance);
        }
        snap.shares_found = shares_found_.load(std::memory_order_relaxed);
        snap.shares_accepted = shares_accepted_.load(std::memory_order_relaxed);
        snap.shares_rejected = shares_rejected_.load(std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(events_mutex_);
            snap.recent_events.assign(events_.begin(), events_.end());
        }
        snap.baseline_total_hashrate = baseline_total_hashrate_.load(std::memory_order_relaxed);
        if(baseline_set_){
            snap.baseline_window_seconds = (uint64_t) std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - baseline_start_).count();
        }
        return snap;
    }
private:
    void push_event(bool accepted, uint32_t nonce, uint64_t diff, const std::string& job){
        std::lock_guard<std::mutex> lk(events_mutex_);
        events_.push_back({std::chrono::steady_clock::now(), accepted, nonce, diff, job});
        if(events_.size() > max_events_) events_.pop_front();
    }
    std::unique_ptr<std::atomic<double>[]> per_thread_hs_;
    unsigned thread_count_;
    std::atomic<uint64_t> shares_found_{0};
    std::atomic<uint64_t> shares_accepted_{0};
    std::atomic<uint64_t> shares_rejected_{0};
    mutable std::mutex events_mutex_;
    std::deque<ZionShareEventRecord> events_;
    size_t max_events_{64};
    std::atomic<double> baseline_total_hashrate_{0.0};
    bool baseline_set_{false};
    std::chrono::steady_clock::time_point baseline_start_{};
    std::chrono::steady_clock::time_point first_sample_ts_{};
    std::chrono::steady_clock::time_point last_sample_ts_{};
};
