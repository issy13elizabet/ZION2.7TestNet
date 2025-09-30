/*
 * ZION AI Mining - Auto-Donate System Implementation
 * Automatic donation system for charitable and development causes
 * Copyright (c) 2025 Maitreya-ZionNet
 */

#include "auto-donate.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <random>
#include <cmath>

namespace zion {
namespace mining {
namespace donate {

// Default donation targets with ZION ecosystem addresses
const std::vector<DonationTarget> AutoDonateManager::DEFAULT_DONATIONS = {
    DonationTarget(
        "ZION Mining Pool Development", 
        "ZION1MiningPoolDevelopmentFundAddress123456789ABCDEF", 
        "Support ZION mining pool infrastructure and development",
        1.0f, // 1%
        "pool.zion-mining.net", 
        3333
    ),
    DonationTarget(
        "Charitable Foundation", 
        "ZION2CharitableFoundationGlobalOutreachAddress789ABC", 
        "Support global humanitarian aid and community development",
        1.0f, // 1%
        "charity-pool.zion-mining.net", 
        3333
    ),
    DonationTarget(
        "Reforestation Initiative", 
        "ZION3ReforestationGlobalTreePlantingFundAddress890DEF", 
        "Plant trees and restore forests worldwide for carbon offset",
        1.0f, // 1%
        "green-pool.zion-mining.net", 
        3333
    ),
    DonationTarget(
        "Ocean Cleanup Project", 
        "ZION4OceanCleanupPlasticRemovalFundAddress901ABCDEF", 
        "Remove plastic waste and restore marine ecosystems",
        1.0f, // 1%
        "ocean-pool.zion-mining.net", 
        3333
    ),
    DonationTarget(
        "Space Exploration Fund", 
        "ZION5SpaceExplorationCosmicResearchFundAddress012345", 
        "Advance human space exploration and cosmic consciousness research",
        1.0f, // 1%
        "space-pool.zion-mining.net", 
        3333
    )
};

// Global donate manager instance
std::unique_ptr<AutoDonateManager> g_donate_manager;

AutoDonateManager::AutoDonateManager() 
    : user_pool_port_(3333)
    , donations_enabled_(true)
    , total_donation_percentage_(5.0f) // 5% total
    , current_target_index_(0)
    , is_donating_(false)
    , last_switch_time_(std::chrono::steady_clock::now())
    , donation_cycle_duration_(std::chrono::minutes(10)) // Switch every 10 minutes
    , total_hashes_(0)
    , donated_hashes_(0)
    , user_shares_(0)
    , donated_shares_(0)
{
    initialize_default_targets();
    calculate_donation_timing();
}

AutoDonateManager::~AutoDonateManager() = default;

void AutoDonateManager::initialize_default_targets() {
    donation_targets_ = DEFAULT_DONATIONS;
    target_hashes_.resize(donation_targets_.size());
    for (auto& counter : target_hashes_) {
        counter.store(0);
    }
}

void AutoDonateManager::set_user_wallet(const std::string& wallet_address) {
    user_wallet_ = wallet_address;
}

void AutoDonateManager::set_user_pool(const std::string& host, uint16_t port) {
    user_pool_host_ = host;
    user_pool_port_ = port;
}

void AutoDonateManager::add_donation_target(const DonationTarget& target) {
    donation_targets_.push_back(target);
    target_hashes_.resize(donation_targets_.size());
    target_hashes_.back().store(0);
}

void AutoDonateManager::set_donation_targets(const std::vector<DonationTarget>& targets) {
    donation_targets_ = targets;
    target_hashes_.clear();
    target_hashes_.resize(donation_targets_.size());
    for (auto& counter : target_hashes_) {
        counter.store(0);
    }
}

void AutoDonateManager::enable_donations(bool enabled) {
    donations_enabled_ = enabled;
    if (!enabled) {
        is_donating_.store(false);
    }
}

void AutoDonateManager::set_total_donation_percentage(float percentage) {
    total_donation_percentage_ = std::max(0.0f, std::min(100.0f, percentage));
    calculate_donation_timing();
}

void AutoDonateManager::calculate_donation_timing() {
    // Calculate cycle duration based on donation percentage
    // 5% donation = 5 minutes donate per 100 minutes total
    // We'll use 10-minute cycles, so donate for 30 seconds every 10 minutes for 5%
    
    float cycle_minutes = 10.0f;
    float donate_seconds_per_cycle = (total_donation_percentage_ / 100.0f) * cycle_minutes * 60.0f;
    
    donation_cycle_duration_ = std::chrono::seconds(static_cast<int64_t>(cycle_minutes * 60));
}

bool AutoDonateManager::should_donate_now() const {
    if (!donations_enabled_ || donation_targets_.empty()) {
        return false;
    }
    
    // Time-based donation cycling
    auto now = std::chrono::steady_clock::now();
    auto cycle_elapsed = now - last_switch_time_;
    
    // Calculate if we should be donating based on percentage
    auto cycle_duration_seconds = std::chrono::duration_cast<std::chrono::seconds>(donation_cycle_duration_).count();
    auto donate_duration_seconds = static_cast<int64_t>((total_donation_percentage_ / 100.0f) * cycle_duration_seconds);
    
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(cycle_elapsed).count();
    
    // Distribute donation time across the cycle
    bool should_donate = (elapsed_seconds % 120) < (donate_duration_seconds / 5); // Spread across cycle
    
    return should_donate;
}

DonationTarget AutoDonateManager::get_current_donation_target() const {
    if (donation_targets_.empty()) {
        return DonationTarget("None", "", "", 0.0f);
    }
    
    size_t index = current_target_index_.load() % donation_targets_.size();
    return donation_targets_[index];
}

std::string AutoDonateManager::get_current_wallet_address() const {
    if (should_donate_now()) {
        auto target = get_current_donation_target();
        return target.wallet_address;
    }
    return user_wallet_;
}

std::string AutoDonateManager::get_current_pool_host() const {
    if (should_donate_now()) {
        auto target = get_current_donation_target();
        if (!target.pool_host.empty()) {
            return target.pool_host;
        }
    }
    return user_pool_host_;
}

uint16_t AutoDonateManager::get_current_pool_port() const {
    if (should_donate_now()) {
        auto target = get_current_donation_target();
        if (target.pool_port > 0) {
            return target.pool_port;
        }
    }
    return user_pool_port_;
}

void AutoDonateManager::record_hash(bool is_donation) {
    total_hashes_.fetch_add(1);
    
    if (is_donation) {
        donated_hashes_.fetch_add(1);
        
        if (current_target_index_.load() < target_hashes_.size()) {
            target_hashes_[current_target_index_.load()].fetch_add(1);
        }
    }
}

void AutoDonateManager::record_share_accepted(bool is_donation) {
    if (is_donation) {
        donated_shares_.fetch_add(1);
    } else {
        user_shares_.fetch_add(1);
    }
}

AutoDonateManager::DonateStats AutoDonateManager::get_stats() const {
    DonateStats stats;
    stats.total_donated_hashes = donated_hashes_.load();
    stats.total_user_hashes = total_hashes_.load() - donated_hashes_.load();
    
    uint64_t total = stats.total_donated_hashes + stats.total_user_hashes;
    stats.current_donate_percentage = total > 0 ? 
        (static_cast<float>(stats.total_donated_hashes) / total) * 100.0f : 0.0f;
    
    auto target = get_current_donation_target();
    stats.current_donate_target = target.name;
    
    // Calculate time until next switch
    auto now = std::chrono::steady_clock::now();
    auto elapsed = now - last_switch_time_;
    auto remaining = donation_cycle_duration_ - elapsed;
    stats.time_until_next_switch = std::chrono::duration_cast<std::chrono::seconds>(remaining);
    
    // Per-target stats
    for (size_t i = 0; i < donation_targets_.size() && i < target_hashes_.size(); ++i) {
        stats.per_target_donations.emplace_back(
            donation_targets_[i].name, 
            target_hashes_[i].load()
        );
    }
    
    return stats;
}

void AutoDonateManager::update_donation_cycle() {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = now - last_switch_time_;
    
    if (elapsed >= donation_cycle_duration_) {
        // Switch to next donation target
        current_target_index_.store(select_next_donation_target());
        last_switch_time_ = now;
        
        log_donation_status();
    }
    
    // Update current donation state
    is_donating_.store(should_donate_now());
}

bool AutoDonateManager::is_donation_time() const {
    return is_donating_.load();
}

size_t AutoDonateManager::select_next_donation_target() const {
    if (donation_targets_.empty()) {
        return 0;
    }
    
    // Rotate through targets sequentially
    return (current_target_index_.load() + 1) % donation_targets_.size();
}

std::string AutoDonateManager::get_donation_report() const {
    std::ostringstream report;
    
    report << "🌟 ZION AI Miner - Auto-Donation Report 🌟\n";
    report << "═══════════════════════════════════════════\n";
    
    auto stats = get_stats();
    
    report << "📊 Overall Statistics:\n";
    report << "  Total Hashes: " << (stats.total_donated_hashes + stats.total_user_hashes) << "\n";
    report << "  User Hashes: " << stats.total_user_hashes << "\n";
    report << "  Donated Hashes: " << stats.total_donated_hashes << "\n";
    report << "  Donation Percentage: " << std::fixed << std::setprecision(2) 
           << stats.current_donate_percentage << "%\n\n";
    
    report << "🎯 Current Target: " << stats.current_donate_target << "\n";
    report << "⏰ Next Switch: " << stats.time_until_next_switch.count() << " seconds\n\n";
    
    report << "💝 Donation Breakdown:\n";
    for (const auto& [name, hashes] : stats.per_target_donations) {
        float percentage = stats.total_donated_hashes > 0 ? 
            (static_cast<float>(hashes) / stats.total_donated_hashes) * 100.0f : 0.0f;
        report << "  " << name << ": " << hashes << " hashes (" 
               << std::fixed << std::setprecision(1) << percentage << "%)\n";
    }
    
    report << "\n🌍 Impact Summary:\n";
    for (const auto& target : donation_targets_) {
        report << "  " << target.name << ":\n";
        report << "    " << target.description << "\n";
        report << "    Wallet: " << target.wallet_address.substr(0, 20) << "...\n";
    }
    
    report << "\n✨ Thank you for supporting these important causes! ✨\n";
    
    return report.str();
}

void AutoDonateManager::log_donation_status() const {
    if (is_donation_time()) {
        auto target = get_current_donation_target();
        std::cout << "💝 Donating to: " << target.name << std::endl;
        std::cout << "   " << target.description << std::endl;
    } else {
        std::cout << "⛏️  Mining for user wallet" << std::endl;
    }
}

// Global convenience functions
void initialize_auto_donate() {
    if (!g_donate_manager) {
        g_donate_manager = std::make_unique<AutoDonateManager>();
    }
}

void shutdown_auto_donate() {
    if (g_donate_manager) {
        std::cout << "\n" << g_donate_manager->get_donation_report() << std::endl;
        g_donate_manager.reset();
    }
}

bool is_auto_donate_enabled() {
    return g_donate_manager && g_donate_manager->should_donate_now();
}

std::string get_donation_wallet() {
    return g_donate_manager ? g_donate_manager->get_current_wallet_address() : "";
}

std::string get_donation_pool() {
    return g_donate_manager ? g_donate_manager->get_current_pool_host() : "";
}

uint16_t get_donation_pool_port() {
    return g_donate_manager ? g_donate_manager->get_current_pool_port() : 3333;
}

} // namespace donate
} // namespace mining
} // namespace zion