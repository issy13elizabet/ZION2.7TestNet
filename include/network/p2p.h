#pragma once

#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <functional>
#include <mutex>
#include <unordered_set>

namespace zion {

struct P2PConfig {
    std::string bind_host = "0.0.0.0";
    uint16_t bind_port = 18080;
    std::vector<std::string> seed_nodes; // host:port
    std::string network_id; // expected network id
    int chain_id = 0;       // expected chain id
};

class P2PNode {
public:
    using LogFn = std::function<void(const std::string&)>;

    struct Callbacks {
        // Return best local height
        std::function<uint32_t()> get_height;
        // Fill headers hex from (start_height+1 .. start_height+count) up to best height
        std::function<bool(uint32_t start_height, uint32_t max_count, std::vector<std::string>& headers_hex)> get_headers;
        // Get block by hash hex
        std::function<bool(const std::string& hash_hex, std::string& block_hex)> get_block;
        // Submit a new block from hex-encoded payload; returns true if accepted (and fills hash_hex)
        std::function<bool(const std::string& block_hex, std::string& hash_hex)> submit_block;
        // Get transaction by hash hex (from mempool)
        std::function<bool(const std::string& hash_hex, std::string& tx_hex)> get_tx;
        // Submit a new transaction from hex payload; returns true if accepted (and fills hash_hex)
        std::function<bool(const std::string& tx_hex, std::string& hash_hex)> submit_tx;
    };

    explicit P2PNode(const P2PConfig& cfg);
    ~P2PNode();

    bool start();
    void stop();

    void set_logger(LogFn fn) { logger_ = std::move(fn); }
    void set_callbacks(const Callbacks& cb) { callbacks_ = cb; }

    // Broadcast helpers
    void broadcast_inv(const std::string& inv_type, const std::string& hash_hex, int exclude_fd = -1);

private:
    P2PConfig cfg_;
    std::atomic<bool> running_{false};
    int listen_fd_ = -1;
    std::thread accept_thread_;
    std::vector<std::thread> peer_threads_;
    std::mutex peers_mutex_;
    std::unordered_set<int> peer_fds_;
    LogFn logger_;
    Callbacks callbacks_{};

    void accept_loop();
    void handle_peer(int fd);
    void connect_seeds();

    void log(const std::string& s) { if (logger_) logger_(s); }
};

} // namespace zion
