#pragma once

#ifndef HASHRATE_STATS_H
#define HASHRATE_STATS_H

#include <chrono>
#include <mutex>
#include <vector>
#include <atomic>

namespace zion {

class HashrateStats {
public:
    HashrateStats();
    
    void add_hash_count(uint64_t hashes);
    void record_hash_time(double time_ms);
    
    double get_current_hashrate() const;
    double get_average_hashrate() const;
    uint64_t get_total_hashes() const;
    
    void reset();
    
private:
    mutable std::mutex stats_mutex_;
    std::atomic<uint64_t> total_hashes_{0};
    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::time_point last_update_;
    
    static constexpr size_t HASHRATE_HISTORY_SIZE = 60; // 60 samples
    std::vector<double> hashrate_history_;
    size_t history_index_{0};
    bool history_full_{false};
    
    void update_hashrate_history(double hashrate);
};

} // namespace zion

#endif // HASHRATE_STATS_H