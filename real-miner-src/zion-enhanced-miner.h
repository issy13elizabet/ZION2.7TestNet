#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <memory>
#include <cmath>
#include <random>
#include <mutex>
#include <algorithm>
#include <functional>

#ifdef _WIN32
    #include <windows.h>
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
#else
    #include <sys/socket.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #include <netdb.h>
#endif

// Simple device types for ZION mining
enum class DeviceType {
    CPU,
    GPU_AMD,
    GPU_NVIDIA,
    QUANTUM,
    AI_ENHANCED
};

// Share notification callback
struct ShareNotification {
    std::string device;
    uint32_t nonce;
    std::string hash;
    bool accepted;
    std::chrono::steady_clock::time_point timestamp;
};

using ShareCallback = std::function<void(const ShareNotification&)>;

// Mining statistics with better tracking
struct MiningStats {
    std::atomic<uint64_t> hashrate{0};
    std::atomic<uint64_t> shares_found{0};
    std::atomic<uint64_t> shares_accepted{0};
    std::atomic<uint64_t> shares_rejected{0};
    std::atomic<double> donation_percent{5.0};
    std::atomic<uint64_t> total_hashes{0};
    std::atomic<double> difficulty{1000.0};
    std::chrono::steady_clock::time_point start_time;
    
    MiningStats() : start_time(std::chrono::steady_clock::now()) {}
    
    // Copy constructor
    MiningStats(const MiningStats& other) : start_time(other.start_time) {
        hashrate.store(other.hashrate.load());
        shares_found.store(other.shares_found.load());
        shares_accepted.store(other.shares_accepted.load());
        shares_rejected.store(other.shares_rejected.load());
        donation_percent.store(other.donation_percent.load());
        total_hashes.store(other.total_hashes.load());
        difficulty.store(other.difficulty.load());
    }
    
    // Assignment operator
    MiningStats& operator=(const MiningStats& other) {
        if (this != &other) {
            hashrate.store(other.hashrate.load());
            shares_found.store(other.shares_found.load());
            shares_accepted.store(other.shares_accepted.load());
            shares_rejected.store(other.shares_rejected.load());
            donation_percent.store(other.donation_percent.load());
            total_hashes.store(other.total_hashes.load());
            difficulty.store(other.difficulty.load());
            start_time = other.start_time;
        }
        return *this;
    }
    
    // Calculate acceptance rate
    double get_acceptance_rate() const {
        uint64_t found = shares_found.load();
        uint64_t accepted = shares_accepted.load();
        return found > 0 ? (static_cast<double>(accepted) / found * 100.0) : 0.0;
    }
    
    // Calculate average share time
    double get_avg_share_time() const {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);
        uint64_t shares = shares_found.load();
        return shares > 0 ? (elapsed.count() / static_cast<double>(shares)) : 0.0;
    }
};

// Auto-donate configuration
struct DonationConfig {
    std::string pool_dev_address = "ZiPooldEv7mining8development9donations2025";
    std::string charity_address = "ZiCharity7humanitarian8aid9global2025relief";
    std::string forest_address = "ZiForesT7reforestation8climate9carbon2025";
    std::string ocean_address = "ZiOceaN7cleanup8marine9ecosystem2025saving";
    std::string space_address = "ZiSpacE7exploration8research9cosmos2025dev";
    double total_percent = 5.0;  // 1% each = 5% total
};

// ZION Cosmic Harmony AI Miner with XMRig-style callbacks
class ZionSimpleMiner {
private:
    DeviceType device_type_;
    std::atomic<bool> is_running_{false};
    std::vector<std::thread> worker_threads_;
    MiningStats stats_;
    DonationConfig donation_config_;
    std::string pool_address_;
    int pool_port_;
    std::string wallet_address_;
    
    // Callback for share notifications
    ShareCallback share_callback_;
    std::mutex callback_mutex_;
    
    // AI Enhancement variables
    std::atomic<double> ai_multiplier_{2.718281828}; // Euler's number for cosmic AI
    std::atomic<uint32_t> consciousness_level_{42}; // Universal answer level
    std::atomic<uint64_t> target_difficulty_{0x0000FFFFFFFFFFFF}; // Target for shares
    
    // Socket connection
    int socket_fd_ = -1;
    std::mutex socket_mutex_;
    
    // Performance tracking
    std::chrono::steady_clock::time_point last_hashrate_update_;
    std::atomic<uint64_t> last_hash_count_{0};

public:
    ZionSimpleMiner(DeviceType device_type, const std::string& pool_address, int pool_port, const std::string& wallet_address);
    ~ZionSimpleMiner();
    
    bool start(int thread_count = 0);
    void stop();
    bool is_running() const { return is_running_.load(); }
    
    // Set callback for share notifications
    void set_share_callback(ShareCallback callback) {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        share_callback_ = callback;
    }
    
    MiningStats get_stats() const { 
        MiningStats copy;
        copy.hashrate.store(stats_.hashrate.load());
        copy.shares_found.store(stats_.shares_found.load());
        copy.shares_accepted.store(stats_.shares_accepted.load());
        copy.shares_rejected.store(stats_.shares_rejected.load());
        copy.donation_percent.store(stats_.donation_percent.load());
        copy.total_hashes.store(stats_.total_hashes.load());
        copy.difficulty.store(stats_.difficulty.load());
        copy.start_time = stats_.start_time;
        return copy;
    }
    
    void print_stats() const;
    void update_difficulty(double new_difficulty) { stats_.difficulty.store(new_difficulty); }
    
    // AI Enhancement methods
    void enhance_with_cosmic_ai();
    uint64_t cosmic_harmony_hash(const std::vector<uint8_t>& data, uint32_t nonce);
    
private:
    void worker_thread(int thread_id);
    bool connect_to_pool();
    void disconnect_from_pool();
    bool send_share(uint32_t nonce, uint64_t hash_value);
    void process_donations(uint64_t reward);
    void update_hashrate();
    void notify_share_found(int thread_id, uint32_t nonce, uint64_t hash_value, bool accepted);
    
    // ZION Cosmic Harmony Algorithm (ZH-2025)
    uint64_t zh2025_hash(const std::vector<uint8_t>& input, uint32_t nonce);
    std::vector<uint8_t> prepare_mining_data();
    bool is_valid_share(uint64_t hash_value) const;
    std::string hash_to_string(uint64_t hash_value) const;
};