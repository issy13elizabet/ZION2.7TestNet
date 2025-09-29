#pragma once

#include <string>
#include <vector>
#include <array>
#include <mutex>
#include <atomic>

// Lightweight job manager (ported concept from 1.3, adapted for 1.4)

struct ZionMiningJob {
    std::string job_id;
    std::vector<uint8_t> blob;          // Raw block header / template
    std::string seed_hash;              // Future use (RandomX/epoch)
    uint64_t    target_difficulty{0};   // Numeric diff (fallback if mask unused)
    uint32_t    nonce_offset{39};       // CryptoNote typical offset
    uint64_t    epoch_id{0};            // Incremented each new job
    std::array<uint8_t,32> target_mask{}; // Optional mask (pool specific)
    double      difficulty_display{0.0}; // computed difficulty (primary)
    double      difficulty_24{0.0};      // optional 24-bit normalized difficulty (for short targets)
};

class ZionJobManager {
public:
    bool has_job() const { return has_job_.load(std::memory_order_acquire); }

    ZionMiningJob current_job() const {
        std::lock_guard<std::mutex> lk(mutex_);
        return job_;
    }

    void update_job(const ZionMiningJob& j){
        {
            std::lock_guard<std::mutex> lk(mutex_);
            job_ = j;
            job_.epoch_id = ++epoch_counter_;
        }
        has_job_.store(true, std::memory_order_release);
    }

private:
    mutable std::mutex mutex_;
    ZionMiningJob job_{};
    std::atomic<bool> has_job_{false};
    std::atomic<uint64_t> epoch_counter_{0};
};
