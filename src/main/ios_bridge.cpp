/*
 * ZION AI Mining - iOS C++ Bridge
 * iOS native interface for ZION AI mining
 * Copyright (c) 2025 Maitreya-ZionNet
 */

#include "zion-ai-mining.h"
#include <memory>

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#endif

using namespace zion::mining;
using namespace zion::mining::ai;

// Global instances
static std::unique_ptr<MobileAIMiningEngine> g_mobile_engine;

extern "C" {

// Logging function for iOS
void ios_log(const char* level, const char* message) {
#ifdef __OBJC__
    NSLog(@"[ZION_%s] %s", level, message);
#endif
}

// Initialize mining engine
bool ios_initialize_mining(
    const char* pool_host, int pool_port,
    const char* wallet_address, const char* worker_name,
    int cpu_threads, int ai_level
) {
    try {
        ios_log("INFO", "Initializing ZION AI Mining Engine for iOS");
        
        // Configure AI for mobile
        AIEnhancedZionMiningEngine::AIConfig ai_config;
        ai_config.ai_enhancement.neural_network_depth = std::max(1, std::min(3, ai_level));
        ai_config.ai_enhancement.consciousness_layers = ai_config.ai_enhancement.neural_network_depth;
        ai_config.ai_enhancement.enable_neural_nonce_selection = true;
        ai_config.ai_enhancement.enable_cosmic_frequency_tuning = true;
        ai_config.ai_enhancement.enable_quantum_consciousness_layer = (ai_level >= 2);
        
        // Mobile-specific configuration for iOS
        MobileAIMiningEngine::MobileConfig mobile_config;
        mobile_config.battery_threshold = 20.0f; // Stop at 20% battery
        mobile_config.thermal_threshold = 43.0f; // Lower threshold for iOS devices
        mobile_config.background_mode = true;
        mobile_config.power_save_mode = true;
        
        // Create mobile mining engine
        g_mobile_engine = std::make_unique<MobileAIMiningEngine>(ai_config, mobile_config);
        
        // Configure mining settings
        ZionMiningEngine::MiningConfig mining_config;
        mining_config.pool_host = std::string(pool_host);
        mining_config.pool_port = static_cast<uint16_t>(pool_port);
        mining_config.wallet_address = std::string(wallet_address);
        mining_config.worker_name = std::string(worker_name);
        mining_config.cpu_threads = static_cast<uint32_t>(std::max(1, cpu_threads));
        mining_config.log_level = 1; // Info level for iOS
        
        bool success = g_mobile_engine->initialize(mining_config);
        
        ios_log("INFO", success ? "Mining engine initialization: SUCCESS" : "Mining engine initialization: FAILED");
        return success;
        
    } catch (const std::exception& e) {
        ios_log("ERROR", ("Exception in ios_initialize_mining: " + std::string(e.what())).c_str());
        return false;
    }
}

// Start mining
bool ios_start_mining() {
    try {
        if (!g_mobile_engine) {
            ios_log("ERROR", "Mining engine not initialized");
            return false;
        }
        
        ios_log("INFO", "Starting mining operation");
        bool success = g_mobile_engine->start_mining();
        
        ios_log("INFO", success ? "Mining start: SUCCESS" : "Mining start: FAILED");
        return success;
        
    } catch (const std::exception& e) {
        ios_log("ERROR", ("Exception in ios_start_mining: " + std::string(e.what())).c_str());
        return false;
    }
}

// Stop mining
void ios_stop_mining() {
    try {
        if (g_mobile_engine) {
            ios_log("INFO", "Stopping mining operation");
            g_mobile_engine->stop_mining();
        }
    } catch (const std::exception& e) {
        ios_log("ERROR", ("Exception in ios_stop_mining: " + std::string(e.what())).c_str());
    }
}

// Check if mining is active
bool ios_is_mining() {
    try {
        if (!g_mobile_engine) {
            return false;
        }
        return g_mobile_engine->is_mining();
    } catch (const std::exception& e) {
        ios_log("ERROR", ("Exception in ios_is_mining: " + std::string(e.what())).c_str());
        return false;
    }
}

// Mining statistics structure for iOS
struct IOSMiningStats {
    double current_hashrate;
    uint64_t total_hashes;
    uint64_t accepted_shares;
    uint64_t rejected_shares;
    uint64_t uptime_seconds;
    float ai_performance_gain;
    float battery_level;
    float temperature;
    bool is_ai_enabled;
};

// Get mining statistics
IOSMiningStats ios_get_mining_stats() {
    IOSMiningStats result = {};
    
    try {
        if (!g_mobile_engine) {
            return result;
        }
        
        auto stats = g_mobile_engine->get_stats();
        auto ai_stats = g_mobile_engine->get_ai_stats();
        auto mobile_stats = g_mobile_engine->get_mobile_stats();
        
        result.current_hashrate = stats.current_hashrate;
        result.total_hashes = stats.total_hashes;
        result.accepted_shares = stats.accepted_shares;
        result.rejected_shares = stats.rejected_shares;
        result.uptime_seconds = stats.uptime_seconds;
        result.ai_performance_gain = ai_stats.ai_performance_gain;
        result.battery_level = mobile_stats.battery_level;
        result.temperature = mobile_stats.temperature;
        result.is_ai_enabled = g_mobile_engine->is_ai_enabled();
        
    } catch (const std::exception& e) {
        ios_log("ERROR", ("Exception in ios_get_mining_stats: " + std::string(e.what())).c_str());
    }
    
    return result;
}

// Update mobile status
void ios_update_mobile_status(
    float battery_level, float temperature,
    bool is_charging, bool is_background
) {
    try {
        if (g_mobile_engine) {
            g_mobile_engine->update_mobile_status(
                battery_level,
                temperature,
                is_charging,
                is_background
            );
        }
    } catch (const std::exception& e) {
        ios_log("ERROR", ("Exception in ios_update_mobile_status: " + std::string(e.what())).c_str());
    }
}

// Set power management mode
void ios_set_power_mode(int power_mode) {
    try {
        if (g_mobile_engine) {
            MobileAIMiningEngine::PowerMode mode;
            switch (power_mode) {
                case 0: mode = MobileAIMiningEngine::PowerMode::BATTERY_SAVER; break;
                case 1: mode = MobileAIMiningEngine::PowerMode::BALANCED; break;
                case 2: mode = MobileAIMiningEngine::PowerMode::PERFORMANCE; break;
                default: mode = MobileAIMiningEngine::PowerMode::BALANCED; break;
            }
            g_mobile_engine->set_power_mode(mode);
        }
    } catch (const std::exception& e) {
        ios_log("ERROR", ("Exception in ios_set_power_mode: " + std::string(e.what())).c_str());
    }
}

// Consciousness state structure for iOS
struct IOSConsciousnessState {
    float compassion_level;
    float wisdom_level;
    float cosmic_alignment;
    bool is_valid;
};

// Get AI consciousness state
IOSConsciousnessState ios_get_consciousness_state() {
    IOSConsciousnessState result = {};
    
    try {
        if (!g_mobile_engine || !g_mobile_engine->is_ai_enabled()) {
            return result;
        }
        
        auto consciousness = g_mobile_engine->ai_engine_->get_consciousness_state();
        
        result.compassion_level = consciousness.compassion_level;
        result.wisdom_level = consciousness.wisdom_level;
        result.cosmic_alignment = consciousness.cosmic_alignment;
        result.is_valid = true;
        
    } catch (const std::exception& e) {
        ios_log("ERROR", ("Exception in ios_get_consciousness_state: " + std::string(e.what())).c_str());
    }
    
    return result;
}

// Background task support for iOS
static bool g_background_task_active = false;

bool ios_begin_background_task() {
    try {
        if (g_mobile_engine && !g_background_task_active) {
            g_background_task_active = true;
            
            // Enable background mode
            g_mobile_engine->update_mobile_status(
                g_mobile_engine->get_mobile_stats().battery_level,
                g_mobile_engine->get_mobile_stats().temperature,
                false, // charging status (unknown in background)
                true   // background mode
            );
            
            ios_log("INFO", "Background task started");
            return true;
        }
    } catch (const std::exception& e) {
        ios_log("ERROR", ("Exception in ios_begin_background_task: " + std::string(e.what())).c_str());
    }
    
    return false;
}

void ios_end_background_task() {
    try {
        if (g_mobile_engine && g_background_task_active) {
            g_background_task_active = false;
            
            // Disable background mode
            g_mobile_engine->update_mobile_status(
                g_mobile_engine->get_mobile_stats().battery_level,
                g_mobile_engine->get_mobile_stats().temperature,
                false, // charging status
                false  // foreground mode
            );
            
            ios_log("INFO", "Background task ended");
        }
    } catch (const std::exception& e) {
        ios_log("ERROR", ("Exception in ios_end_background_task: " + std::string(e.what())).c_str());
    }
}

// Memory pressure handling for iOS
void ios_handle_memory_pressure() {
    try {
        if (g_mobile_engine) {
            // Temporarily reduce AI processing to free memory
            auto current_power_mode = g_mobile_engine->get_power_mode();
            g_mobile_engine->set_power_mode(MobileAIMiningEngine::PowerMode::BATTERY_SAVER);
            
            // Log memory pressure event
            ios_log("WARNING", "Memory pressure detected - reducing AI processing");
            
            // Could restore power mode after pressure is relieved
            // For now, stay in battery saver mode
        }
    } catch (const std::exception& e) {
        ios_log("ERROR", ("Exception in ios_handle_memory_pressure: " + std::string(e.what())).c_str());
    }
}

// Thermal state handling for iOS
void ios_handle_thermal_state(int thermal_state) {
    try {
        if (g_mobile_engine) {
            // 0=Normal, 1=Fair, 2=Serious, 3=Critical
            switch (thermal_state) {
                case 0: // Normal
                    g_mobile_engine->set_power_mode(MobileAIMiningEngine::PowerMode::BALANCED);
                    ios_log("INFO", "Thermal state: Normal");
                    break;
                case 1: // Fair
                    g_mobile_engine->set_power_mode(MobileAIMiningEngine::PowerMode::BALANCED);
                    ios_log("INFO", "Thermal state: Fair");
                    break;
                case 2: // Serious
                    g_mobile_engine->set_power_mode(MobileAIMiningEngine::PowerMode::BATTERY_SAVER);
                    ios_log("WARNING", "Thermal state: Serious - reducing performance");
                    break;
                case 3: // Critical
                    g_mobile_engine->stop_mining();
                    ios_log("ERROR", "Thermal state: Critical - stopping mining");
                    break;
            }
        }
    } catch (const std::exception& e) {
        ios_log("ERROR", ("Exception in ios_handle_thermal_state: " + std::string(e.what())).c_str());
    }
}

// Cleanup
void ios_cleanup() {
    try {
        ios_log("INFO", "Cleaning up mining engine");
        if (g_mobile_engine) {
            g_mobile_engine->stop_mining();
            g_mobile_engine.reset();
        }
        g_background_task_active = false;
    } catch (const std::exception& e) {
        ios_log("ERROR", ("Exception in ios_cleanup: " + std::string(e.what())).c_str());
    }
}

} // extern "C"