#pragma once

#include <string>
#include <thread>
#include <atomic>
#include <functional>

namespace zion {

struct PoolConfig {
    uint16_t port = 3333;
};

class PoolServer {
public:
    struct Callbacks {
        // stratum-like callbacks
        std::function<std::string(const std::string& worker, const std::string& pass)> login; // return job JSON or error
        std::function<std::string()> get_job; // return current job JSON
        std::function<bool(const std::string& job_id, const std::string& nonce_hex, const std::string& result_hex, const std::string& worker)> submit_share;
    };

    PoolServer(const PoolConfig& cfg, Callbacks cbs);
    ~PoolServer();

    bool start();
    void stop();

private:
    PoolConfig cfg_;
    Callbacks cbs_;
    std::atomic<bool> running_{false};
    int listen_fd_ = -1;
    std::thread accept_thread_;

    void accept_loop();
    void handle_client(int fd);
};

} // namespace zion
