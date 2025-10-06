/*
 * ZION AI Mining - Auto-Donate System
 * Automatic donation system for charitable and development causes
 * Copyright (c) 2025 Maitreya-ZionNet
 */

#ifndef ZION_AUTO_DONATE_H
#define ZION_AUTO_DONATE_H

#include <string>
#include <vector>
#include <chrono>
#include <atomic>
#include <memory>

namespace zion {
namespace mining {
namespace donate {

struct DonationTarget {
    std::string name;
    std::string wallet_address;
    std::string description;
    float percentage;          // Percentage of mining time/hashrate
    std::string pool_host;     // Optional: different pool for this donation
    uint16_t pool_port;        // Pool port
    bool enabled;
    
    DonationTarget(const std::string& n, const std::string& addr, const std::string& desc, 
                   float pct, const std::string& host = "", uint16_t port = 0)
        : name(n), wallet_address(addr), description(desc), percentage(pct)
        , pool_host(host), pool_port(port), enabled(true) {}
};

class AutoDonateManager {
public:
    // Default donation targets with ZION addresses
    static const std::vector<DonationTarget> DEFAULT_DONATIONS;
    
    struct DonateStats {
        uint64_t total_donated_hashes;
        uint64_t total_user_hashes;
        float current_donate_percentage;
        std::string current_donate_target;
        std::chrono::seconds time_until_next_switch;
        std::vector<std::pair<std::string, uint64_t>> per_target_donations;
    };
    
    AutoDonateManager();
    ~AutoDonateManager();
    
    // Configuration
    void set_user_wallet(const std::string& wallet_address);
    void set_user_pool(const std::string& host, uint16_t port);
    void add_donation_target(const DonationTarget& target);
    void set_donation_targets(const std::vector<DonationTarget>& targets);
    void enable_donations(bool enabled);
    void set_total_donation_percentage(float percentage); // Default 5%
    
    // Runtime control
    bool should_donate_now() const;
    DonationTarget get_current_donation_target() const;
    std::string get_current_wallet_address() const;
    std::string get_current_pool_host() const;
    uint16_t get_current_pool_port() const;
    
    // Statistics
    void record_hash(bool is_donation = false);
    void record_share_accepted(bool is_donation = false);
    DonateStats get_stats() const;
    
    // Time-based donation switching
    void update_donation_cycle();
    bool is_donation_time() const;
    
    // Transparency and reporting
    std::string get_donation_report() const;
    void log_donation_status() const;
    
private:
    std::string user_wallet_;
    std::string user_pool_host_;
    uint16_t user_pool_port_;
    
    std::vector<DonationTarget> donation_targets_;
    bool donations_enabled_;
    float total_donation_percentage_;
    
    // Current donation state
    mutable std::atomic<size_t> current_target_index_;
    mutable std::atomic<bool> is_donating_;
    std::chrono::steady_clock::time_point last_switch_time_;
    std::chrono::seconds donation_cycle_duration_;
    
    // Statistics
    mutable std::atomic<uint64_t> total_hashes_;
    mutable std::atomic<uint64_t> donated_hashes_;
    mutable std::atomic<uint64_t> user_shares_;
    mutable std::atomic<uint64_t> donated_shares_;
    
    // Per-target statistics
    mutable std::vector<std::atomic<uint64_t>> target_hashes_;
    
    void initialize_default_targets();
    void calculate_donation_timing();
    size_t select_next_donation_target() const;
};

// Global donate manager instance
extern std::unique_ptr<AutoDonateManager> g_donate_manager;

// Convenience functions
void initialize_auto_donate();
void shutdown_auto_donate();
bool is_auto_donate_enabled();
std::string get_donation_wallet();
std::string get_donation_pool();
uint16_t get_donation_pool_port();

} // namespace donate
} // namespace mining
} // namespace zion

#endif // ZION_AUTO_DONATE_H