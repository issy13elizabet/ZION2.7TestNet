#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <cstdint>
#include <string>
#include <functional>
#include "mining_stats.h"

struct ZionShare {
    std::string job_id;
    uint32_t nonce;
    std::string result_hex; // hash result hex
    uint64_t difficulty;    // computed difficulty
};

class PoolConnection; // forward
class StratumClient;  // forward

class ZionShareSubmitter {
public:
    explicit ZionShareSubmitter(PoolConnection* pool, StratumClient* stratum=nullptr, ZionMiningStatsAggregator* stats=nullptr);
    void set_stratum(StratumClient* s){ stratum_ = s; }
    void set_stats_aggregator(ZionMiningStatsAggregator* a){ stats_ = a; }
    void set_result_callback(std::function<void(const ZionShare&, bool)> cb){ result_cb_ = std::move(cb); }
    ~ZionShareSubmitter();
    void enqueue(const ZionShare& share);
    void start();
    void stop();
    uint64_t accepted() const { return accepted_.load(); }
    uint64_t rejected() const { return rejected_.load(); }
private:
    void run();
    PoolConnection* pool_;
    StratumClient* stratum_{nullptr};
    // Lock-free SPSC ring buffer (producer: workers via enqueue, consumer: submitter thread)
    static constexpr size_t RING_CAP = 4096; // power of two
    struct RingSlot { std::atomic<bool> full{false}; ZionShare val; };
    std::unique_ptr<RingSlot[]> ring_ = std::unique_ptr<RingSlot[]>(new RingSlot[RING_CAP]);
    std::atomic<size_t> head_{0}; // producer index
    std::atomic<size_t> tail_{0}; // consumer index
    // Fallback queue (legacy) will remain until migration complete
    std::queue<ZionShare> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> accepted_{0};
    std::atomic<uint64_t> rejected_{0};
    ZionMiningStatsAggregator* stats_{nullptr};
    std::function<void(const ZionShare&, bool)> result_cb_;
};
