#pragma once
#include <thread>
#include <atomic>
#include <vector>
#include <cstdint>
#include <string>
#include <functional>
#include "job_manager.h"
#include "hashrate_stats.h"
#include <optional>
#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#endif

namespace zion { class RandomXWrapper; }

struct ZionCpuWorkerConfig {
    unsigned index{};
    unsigned total_threads{};
    bool use_randomx{true};
    bool pin_threads{false};
};

class ZionShareSubmitter; // fwd
class ZionMiningStatsAggregator; // fwd

class ZionCpuWorker {
public:
    ZionCpuWorker(ZionJobManager* jm, ZionShareSubmitter* submitter, const ZionCpuWorkerConfig& cfg, ZionMiningStatsAggregator* stats=nullptr);
    ~ZionCpuWorker();
    void start();
    void stop();
    double last_instant_hs() const { return last_instant_hs_; }
private:
    void run();
    uint32_t compute_next_nonce(uint64_t base_nonce, unsigned thread_index, unsigned total) const;
    void apply_affinity(unsigned logical_index);
    ZionJobManager* job_manager_;
    ZionShareSubmitter* submitter_;
    ZionCpuWorkerConfig cfg_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    ZionHashrateStats stats_;
    double last_instant_hs_{0.0};
    ZionMiningStatsAggregator* agg_{nullptr};
};
