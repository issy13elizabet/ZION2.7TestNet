#pragma once
#include <atomic>
#include <cstdint>
#include <chrono>

class ZionHashrateStats {
public:
    ZionHashrateStats();
    void add_hashes(uint64_t count);
    double current_hs(); // instantaneous (window)
    double avg_hs() const;     // exponential moving average
private:
    // These are updated only by the worker thread that owns the stats, so atomic is not strictly required.
    uint64_t window_hashes_{0};
    uint64_t total_hashes_{0};
    std::chrono::steady_clock::time_point start_;
    std::chrono::steady_clock::time_point last_window_;
    double ema_{0.0};
};
