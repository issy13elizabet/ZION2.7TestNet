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

// Mining statistics
struct MiningStats {
    std::atomic<uint64_t> hashrate{0};
    std::atomic<uint64_t> shares_found{0};
    std::atomic<uint64_t> shares_accepted{0};
    std::atomic<double> donation_percent{5.0};
    std::chrono::steady_clock::time_point start_time;
    
    MiningStats() : start_time(std::chrono::steady_clock::now()) {}
    
    // Copy constructor
    MiningStats(const MiningStats& other) : start_time(other.start_time) {
        hashrate.store(other.hashrate.load());
        shares_found.store(other.shares_found.load());
        shares_accepted.store(other.shares_accepted.load());
        donation_percent.store(other.donation_percent.load());
    }
    
    // Assignment operator
    MiningStats& operator=(const MiningStats& other) {
        if (this != &other) {
            hashrate.store(other.hashrate.load());
            shares_found.store(other.shares_found.load());
            shares_accepted.store(other.shares_accepted.load());
            donation_percent.store(other.donation_percent.load());
            start_time = other.start_time;
        }
        return *this;
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

// ZION Cosmic Harmony AI Miner
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
    
    // AI Enhancement variables
    std::atomic<double> ai_multiplier_{1.618}; // Golden ratio for cosmic harmony
    std::atomic<uint32_t> consciousness_level_{42}; // Universal answer level
    
    // Socket connection
    int socket_fd_ = -1;
    std::mutex socket_mutex_;

public:
    ZionSimpleMiner(DeviceType device_type, const std::string& pool_address, int pool_port, const std::string& wallet_address);
    ~ZionSimpleMiner();
    
    bool start(int thread_count = 0);
    void stop();
    bool is_running() const { return is_running_.load(); }
    
    MiningStats get_stats() const { 
        MiningStats copy;
        copy.hashrate.store(stats_.hashrate.load());
        copy.shares_found.store(stats_.shares_found.load());
        copy.shares_accepted.store(stats_.shares_accepted.load());
        copy.donation_percent.store(stats_.donation_percent.load());
        copy.start_time = stats_.start_time;
        return copy;
    }
    void print_stats() const;
    
    // AI Enhancement methods
    void enhance_with_cosmic_ai();
    uint64_t cosmic_harmony_hash(const std::vector<uint8_t>& data, uint32_t nonce);
    
private:
    void worker_thread(int thread_id);
    bool connect_to_pool();
    void disconnect_from_pool();
    bool send_share(uint32_t nonce, uint64_t hash_value);
    void process_donations(uint64_t reward);
    
    // ZION Cosmic Harmony Algorithm (ZH-2025)
    uint64_t zh2025_hash(const std::vector<uint8_t>& input, uint32_t nonce);
    std::vector<uint8_t> prepare_mining_data();
};