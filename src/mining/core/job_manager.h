#pragma once
#include <atomic>
#include <mutex>
#include <string>
#include <vector>
#include <cstdint>
#include <chrono>
#include <array>

// Represents a current mining job fetched from pool/daemon
struct ZionMiningJob {
    std::string job_id;          // Pool job identifier
    std::vector<uint8_t> blob;   // Raw block blob (with space for nonce)
    std::string seed_hash;       // RandomX seed / key (hex)
    uint64_t target_difficulty{0}; // Numeric difficulty (pool target)
    std::array<uint8_t,32> target_mask{}; // 256-bit target (little-endian typical CryptoNote). Zero => invalid.
    uint32_t height{0};
    uint64_t epoch_id{0};        // Monotonic bump when job changes
    uint32_t nonce_offset{39};   // Typical Monero-style blob offset (to verify)
};

class ZionJobManager {
public:
    ZionJobManager();
    // Update job with new data; increments epoch
    void update_job(const ZionMiningJob& job);
    // Copy current job snapshot (epoch included)
    ZionMiningJob current_job() const;
    // Return current epoch id
    uint64_t epoch() const { return current_epoch_.load(std::memory_order_relaxed); }
    bool has_job() const { return has_job_.load(std::memory_order_acquire); }
private:
    mutable std::mutex mutex_;
    ZionMiningJob job_;
    std::atomic<uint64_t> current_epoch_{0};
    std::atomic<bool> has_job_{false};
};
