#pragma once

#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include <functional>
#include <optional>
#include <cstdint>

class ZionJobManager; // fwd

struct StratumJobData {
    std::string job_id;
    std::string blob_hex;
    std::string seed_hash;
    uint64_t target_difficulty{0};
    uint32_t nonce_offset{39}; // CryptoNote style default
};

class StratumClient {
public:
    StratumClient(ZionJobManager* jm,
                  std::string host,
                  int port,
                  std::string wallet,
                  std::string worker="miner1");
    ~StratumClient();

    bool start();
    void stop();
    bool running() const { return running_.load(); }

    // Submit a share (nonce + result hash hex)
    void submit_share(const std::string& job_id, uint32_t nonce, const std::string& result_hex, uint64_t difficulty_value);

    uint64_t accepted() const { return accepted_.load(); }
    uint64_t rejected() const { return rejected_.load(); }

private:
    ZionJobManager* job_manager_{};
    std::string host_;
    int port_{};
    std::string wallet_;
    std::string worker_;

    std::atomic<bool> running_{false};
    std::thread io_thread_;
    std::mutex sock_mtx_;
    int sock_fd_{-1};

    std::atomic<uint64_t> accepted_{0};
    std::atomic<uint64_t> rejected_{0};
    std::atomic<int> id_counter_{1};

    // Internal helpers
    bool connect_socket();
    void io_loop();
    void handle_line(const std::string& line);
    void parse_job_notification(const std::string& json_text);
    void send_json(const std::string& json_line);
};
