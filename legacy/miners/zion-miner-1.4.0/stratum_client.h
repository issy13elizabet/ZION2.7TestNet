#pragma once

#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <chrono>
#include "zion-job-manager.h"

// Simplified Stratum client (initial production pass)

enum class StratumProtocol { Stratum, CryptoNote };

class StratumClient {
public:
    StratumClient(ZionJobManager* jm,
                  std::string host,
                  int port,
                  std::string wallet,
                  std::string worker="miner1",
                  StratumProtocol proto = StratumProtocol::Stratum);
    ~StratumClient();

    bool start();
    void stop();
    bool running() const { return running_.load(); }

    void submit_share(const std::string& job_id, uint32_t nonce, const std::string& result_hex, uint64_t difficulty_value);

    uint64_t accepted() const { return accepted_.load(); }
    uint64_t rejected() const { return rejected_.load(); }
    void set_verbose(bool v){ verbose_.store(v); }
    bool verbose() const { return verbose_.load(); }
    void reset_counters(){ accepted_.store(0); rejected_.store(0); id_counter_.store(1); }
    void set_protocol(StratumProtocol p){ protocol_ = p; }
    StratumProtocol protocol() const { return protocol_; }
    std::string session_id() const { return session_id_; }
    std::string extranonce() const { return extranonce_; }

private:
    ZionJobManager* job_manager_{};
    std::string host_;
    int port_{};
    std::string wallet_;
    std::string worker_;

    std::atomic<bool> running_{false};
    std::thread io_thread_;
    int sock_fd_{-1};

    std::atomic<uint64_t> accepted_{0};
    std::atomic<uint64_t> rejected_{0};
    std::atomic<int> id_counter_{1};
    std::atomic<bool> verbose_{false};
    StratumProtocol protocol_{StratumProtocol::Stratum};
    std::string session_id_;
    std::string extranonce_;

    struct PendingShare { std::string job_id; uint32_t nonce; std::chrono::steady_clock::time_point ts; };
    std::mutex pending_mutex_;
    std::unordered_map<int, PendingShare> pending_;

    bool connect_socket();
    void io_loop();
    void handle_line(const std::string& line);
    void parse_job_notification(const std::string& json_text);
    void send_json(const std::string& json_line);
    void send_login();
};
