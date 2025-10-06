#include "hashrate_stats.h"

ZionHashrateStats::ZionHashrateStats() : start_(std::chrono::steady_clock::now()), last_window_(start_) {}

void ZionHashrateStats::add_hashes(uint64_t c) {
    window_hashes_ += c;
    total_hashes_ += c;
}

double ZionHashrateStats::current_hs() {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_window_).count();
    if (elapsed < 200) return ema_; // don't sample too frequently
    uint64_t wh = window_hashes_;
    window_hashes_ = 0;
    double instant = elapsed > 0 ? (double)wh * 1000.0 / (double)elapsed : 0.0;
    last_window_ = now; // non-atomic, fine under single-writer assumption
    // simple EMA smoothing
    if (ema_ == 0.0) ema_ = instant;
    else ema_ = ema_ * 0.7 + instant * 0.3;
    return instant;
}

double ZionHashrateStats::avg_hs() const {
    auto uptime_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_).count();
    if (uptime_ms == 0) return 0.0;
    double total = (double)total_hashes_;
    return total * 1000.0 / (double)uptime_ms;
}
