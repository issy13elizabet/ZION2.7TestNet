/*
 * ZION AI Mining - Android JNI Bridge
 * Android native interface for ZION AI mining
 * Copyright (c) 2025 Maitreya-ZionNet
 */

#include "zion-ai-mining.h"
#include <jni.h>
#include <android/log.h>
#include <memory>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "ZION_MINER", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "ZION_MINER", __VA_ARGS__)

using namespace zion::mining;
using namespace zion::mining::ai;

// Global instances
static std::unique_ptr<MobileAIMiningEngine> g_mobile_engine;
static JavaVM* g_java_vm = nullptr;

extern "C" {

// JNI OnLoad - called when library is loaded
jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    LOGI("ZION AI Miner JNI loaded");
    g_java_vm = vm;
    return JNI_VERSION_1_6;
}

// Initialize mining engine
JNIEXPORT jboolean JNICALL
Java_net_zionnet_mining_ZionAIMiner_initializeMining(
    JNIEnv* env, jobject thiz,
    jstring pool_host, jint pool_port,
    jstring wallet_address, jstring worker_name,
    jint cpu_threads, jint ai_level
) {
    try {
        LOGI("Initializing ZION AI Mining Engine for Android");
        
        // Convert Java strings to C++ strings
        const char* host_cstr = env->GetStringUTFChars(pool_host, nullptr);
        const char* wallet_cstr = env->GetStringUTFChars(wallet_address, nullptr);
        const char* worker_cstr = env->GetStringUTFChars(worker_name, nullptr);
        
        std::string host_str(host_cstr);
        std::string wallet_str(wallet_cstr);
        std::string worker_str(worker_cstr);
        
        env->ReleaseStringUTFChars(pool_host, host_cstr);
        env->ReleaseStringUTFChars(wallet_address, wallet_cstr);
        env->ReleaseStringUTFChars(worker_name, worker_cstr);
        
        // Configure AI for mobile
        AIEnhancedZionMiningEngine::AIConfig ai_config;
        ai_config.ai_enhancement.neural_network_depth = std::max(1, std::min(3, static_cast<int>(ai_level)));
        ai_config.ai_enhancement.consciousness_layers = ai_config.ai_enhancement.neural_network_depth;
        ai_config.ai_enhancement.enable_neural_nonce_selection = true;
        ai_config.ai_enhancement.enable_cosmic_frequency_tuning = true;
        ai_config.ai_enhancement.enable_quantum_consciousness_layer = (ai_level >= 2);
        
        // Mobile-specific configuration
        MobileAIMiningEngine::MobileConfig mobile_config;
        mobile_config.battery_threshold = 20.0f; // Stop at 20% battery
        mobile_config.thermal_threshold = 45.0f; // Stop at 45°C
        mobile_config.background_mode = true;
        mobile_config.power_save_mode = true;
        
        // Create mobile mining engine
        g_mobile_engine = std::make_unique<MobileAIMiningEngine>(ai_config, mobile_config);
        
        // Configure mining settings
        ZionMiningEngine::MiningConfig mining_config;
        mining_config.pool_host = host_str;
        mining_config.pool_port = static_cast<uint16_t>(pool_port);
        mining_config.wallet_address = wallet_str;
        mining_config.worker_name = worker_str;
        mining_config.cpu_threads = static_cast<uint32_t>(std::max(1, static_cast<int>(cpu_threads)));
        mining_config.log_level = 1; // Info level for Android
        
        bool success = g_mobile_engine->initialize(mining_config);
        
        LOGI("Mining engine initialization: %s", success ? "SUCCESS" : "FAILED");
        return success;
        
    } catch (const std::exception& e) {
        LOGE("Exception in initializeMining: %s", e.what());
        return false;
    }
}

// Start mining
JNIEXPORT jboolean JNICALL
Java_net_zionnet_mining_ZionAIMiner_startMining(JNIEnv* env, jobject thiz) {
    try {
        if (!g_mobile_engine) {
            LOGE("Mining engine not initialized");
            return false;
        }
        
        LOGI("Starting mining operation");
        bool success = g_mobile_engine->start_mining();
        
        LOGI("Mining start: %s", success ? "SUCCESS" : "FAILED");
        return success;
        
    } catch (const std::exception& e) {
        LOGE("Exception in startMining: %s", e.what());
        return false;
    }
}

// Stop mining
JNIEXPORT void JNICALL
Java_net_zionnet_mining_ZionAIMiner_stopMining(JNIEnv* env, jobject thiz) {
    try {
        if (g_mobile_engine) {
            LOGI("Stopping mining operation");
            g_mobile_engine->stop_mining();
        }
    } catch (const std::exception& e) {
        LOGE("Exception in stopMining: %s", e.what());
    }
}

// Check if mining is active
JNIEXPORT jboolean JNICALL
Java_net_zionnet_mining_ZionAIMiner_isMining(JNIEnv* env, jobject thiz) {
    try {
        if (!g_mobile_engine) {
            return false;
        }
        return g_mobile_engine->is_mining();
    } catch (const std::exception& e) {
        LOGE("Exception in isMining: %s", e.what());
        return false;
    }
}

// Get mining statistics
JNIEXPORT jobject JNICALL
Java_net_zionnet_mining_ZionAIMiner_getMiningStats(JNIEnv* env, jobject thiz) {
    try {
        if (!g_mobile_engine) {
            return nullptr;
        }
        
        auto stats = g_mobile_engine->get_stats();
        auto ai_stats = g_mobile_engine->get_ai_stats();
        auto mobile_stats = g_mobile_engine->get_mobile_stats();
        
        // Find the MiningStats Java class
        jclass stats_class = env->FindClass("net/zionnet/mining/MiningStats");
        if (!stats_class) {
            LOGE("Could not find MiningStats class");
            return nullptr;
        }
        
        // Get constructor
        jmethodID constructor = env->GetMethodID(stats_class, "<init>", "()V");
        if (!constructor) {
            LOGE("Could not find MiningStats constructor");
            return nullptr;
        }
        
        // Create new instance
        jobject stats_obj = env->NewObject(stats_class, constructor);
        if (!stats_obj) {
            LOGE("Could not create MiningStats object");
            return nullptr;
        }
        
        // Set fields
        jfieldID hashrate_field = env->GetFieldID(stats_class, "currentHashrate", "D");
        jfieldID total_hashes_field = env->GetFieldID(stats_class, "totalHashes", "J");
        jfieldID accepted_field = env->GetFieldID(stats_class, "acceptedShares", "J");
        jfieldID rejected_field = env->GetFieldID(stats_class, "rejectedShares", "J");
        jfieldID uptime_field = env->GetFieldID(stats_class, "uptimeSeconds", "J");
        jfieldID ai_gain_field = env->GetFieldID(stats_class, "aiPerformanceGain", "F");
        jfieldID battery_field = env->GetFieldID(stats_class, "batteryLevel", "F");
        jfieldID temperature_field = env->GetFieldID(stats_class, "temperature", "F");
        
        if (hashrate_field) env->SetDoubleField(stats_obj, hashrate_field, stats.current_hashrate);
        if (total_hashes_field) env->SetLongField(stats_obj, total_hashes_field, static_cast<jlong>(stats.total_hashes));
        if (accepted_field) env->SetLongField(stats_obj, accepted_field, static_cast<jlong>(stats.accepted_shares));
        if (rejected_field) env->SetLongField(stats_obj, rejected_field, static_cast<jlong>(stats.rejected_shares));
        if (uptime_field) env->SetLongField(stats_obj, uptime_field, static_cast<jlong>(stats.uptime_seconds));
        if (ai_gain_field) env->SetFloatField(stats_obj, ai_gain_field, ai_stats.ai_performance_gain);
        if (battery_field) env->SetFloatField(stats_obj, battery_field, mobile_stats.battery_level);
        if (temperature_field) env->SetFloatField(stats_obj, temperature_field, mobile_stats.temperature);
        
        return stats_obj;
        
    } catch (const std::exception& e) {
        LOGE("Exception in getMiningStats: %s", e.what());
        return nullptr;
    }
}

// Update mobile status (battery, temperature, etc.)
JNIEXPORT void JNICALL
Java_net_zionnet_mining_ZionAIMiner_updateMobileStatus(
    JNIEnv* env, jobject thiz,
    jfloat battery_level, jfloat temperature,
    jboolean is_charging, jboolean is_background
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
        LOGE("Exception in updateMobileStatus: %s", e.what());
    }
}

// Set power management mode
JNIEXPORT void JNICALL
Java_net_zionnet_mining_ZionAIMiner_setPowerMode(
    JNIEnv* env, jobject thiz,
    jint power_mode // 0=battery_saver, 1=balanced, 2=performance
) {
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
        LOGE("Exception in setPowerMode: %s", e.what());
    }
}

// Get AI consciousness state
JNIEXPORT jobject JNICALL
Java_net_zionnet_mining_ZionAIMiner_getConsciousnessState(JNIEnv* env, jobject thiz) {
    try {
        if (!g_mobile_engine || !g_mobile_engine->is_ai_enabled()) {
            return nullptr;
        }
        
        auto consciousness = g_mobile_engine->ai_engine_->get_consciousness_state();
        
        // Find the ConsciousnessState Java class
        jclass consciousness_class = env->FindClass("net/zionnet/mining/ConsciousnessState");
        if (!consciousness_class) {
            LOGE("Could not find ConsciousnessState class");
            return nullptr;
        }
        
        // Get constructor
        jmethodID constructor = env->GetMethodID(consciousness_class, "<init>", "(FFF)V");
        if (!constructor) {
            LOGE("Could not find ConsciousnessState constructor");
            return nullptr;
        }
        
        // Create new instance
        return env->NewObject(consciousness_class, constructor,
                             consciousness.compassion_level,
                             consciousness.wisdom_level,
                             consciousness.cosmic_alignment);
        
    } catch (const std::exception& e) {
        LOGE("Exception in getConsciousnessState: %s", e.what());
        return nullptr;
    }
}

// Cleanup
JNIEXPORT void JNICALL
Java_net_zionnet_mining_ZionAIMiner_cleanup(JNIEnv* env, jobject thiz) {
    try {
        LOGI("Cleaning up mining engine");
        if (g_mobile_engine) {
            g_mobile_engine->stop_mining();
            g_mobile_engine.reset();
        }
    } catch (const std::exception& e) {
        LOGE("Exception in cleanup: %s", e.what());
    }
}

} // extern "C"