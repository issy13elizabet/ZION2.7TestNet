#pragma once

#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>

// Simple pool connection for ZION
class PoolConnection {
private:
    std::string pool_address_;
    int pool_port_;
    std::string wallet_address_;
    std::atomic<bool> connected_{false};
    std::thread connection_thread_;
    mutable std::mutex connection_mutex_;

public:
    PoolConnection(const std::string& address, int port, const std::string& wallet);
    ~PoolConnection();
    
    bool connect();
    void disconnect();
    bool is_connected() const { return connected_.load(); }
    
    bool submit_share(uint32_t nonce, uint64_t hash);
    void send_donation(const std::string& address, uint64_t amount);
    
private:
    void maintain_connection();
    bool send_stratum_message(const std::string& message);
};