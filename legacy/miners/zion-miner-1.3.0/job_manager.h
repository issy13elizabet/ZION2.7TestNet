#pragma once

#ifndef JOB_MANAGER_H
#define JOB_MANAGER_H

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace zion {
namespace mining {

/**
 * ZION Job Manager - Handles mining jobs from pool
 */
struct MiningJob {
    std::string job_id;
    std::string blob;
    std::string target;
    uint64_t height;
    std::string seed_hash;
    std::string next_seed_hash;
    uint32_t timestamp;
    bool clean_jobs;
    
    MiningJob() : height(0), timestamp(0), clean_jobs(false) {}
};

struct ShareResult {
    std::string job_id;
    std::string nonce;
    std::string result;
    bool accepted;
    std::string error;
    
    ShareResult() : accepted(false) {}
};

class JobManager {
public:
    JobManager();
    ~JobManager();
    
    // Job management
    bool set_current_job(const MiningJob& job);
    MiningJob get_current_job() const;
    bool has_valid_job() const;
    
    // Share submission
    bool submit_share(const ShareResult& share);
    
    // Job validation
    bool is_job_valid(const std::string& job_id) const;
    bool is_target_met(const std::string& hash, const std::string& target) const;
    
    // Statistics
    uint64_t get_submitted_shares() const;
    uint64_t get_accepted_shares() const;
    uint64_t get_rejected_shares() const;
    double get_acceptance_rate() const;
    
    // Configuration
    void set_pool_difficulty(uint64_t difficulty);
    uint64_t get_pool_difficulty() const;
    
private:
    mutable std::mutex job_mutex_;
    MiningJob current_job_;
    
    std::atomic<uint64_t> submitted_shares_{0};
    std::atomic<uint64_t> accepted_shares_{0};
    std::atomic<uint64_t> rejected_shares_{0};
    std::atomic<uint64_t> pool_difficulty_{1};
    
    bool validate_job_data(const MiningJob& job) const;
    uint64_t target_to_difficulty(const std::string& target) const;
};

} // namespace mining
} // namespace zion

#endif // JOB_MANAGER_H