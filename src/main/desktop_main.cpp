/*
 * ZION AI Mining - Desktop Application Main
 * Cross-platform: Windows 11, Ubuntu, macOS
 * Copyright (c) 2025 Maitreya-ZionNet
 */

#include "zion-ai-mining.h"
#include "auto-donate.h"
#include <iostream>
#include <csignal>
#include <thread>
#include <chrono>

using namespace zion::mining;
using namespace zion::mining::ai;
using namespace zion::mining::donate;

// Global mining engine instance
std::unique_ptr<AIEnhancedZionMiningEngine> g_mining_engine;
std::atomic<bool> g_shutdown_requested{false};

void signal_handler(int signal) {
    std::cout << "\n🛑 Shutdown signal received (" << signal << ")" << std::endl;
    g_shutdown_requested = true;
    
    if (g_mining_engine) {
        g_mining_engine->stop_mining();
    }
}

void print_banner() {
    std::cout << R"(
╔════════════════════════════════════════════════════════════════╗
║                    🌟 ZION AI MINER 🌟                        ║
║                                                                ║
║        AI-Enhanced Multi-Algorithm Cryptocurrency Miner       ║
║                   with Cosmic Harmony Algorithm               ║
║                                                                ║
║                      MIT Licensed                              ║
║                   Copyright © 2025                             ║
║                  Maitreya-ZionNet                              ║
╚════════════════════════════════════════════════════════════════╝
)" << std::endl;

    std::cout << "🚀 Platform: ";
#ifdef _WIN32
    std::cout << "Windows 11";
#elif defined(__APPLE__)
    std::cout << "macOS";
#elif defined(__linux__)
    std::cout << "Linux/Ubuntu";
#else
    std::cout << "Unknown";
#endif
    std::cout << std::endl;
    
    std::cout << "🧠 AI Enhancement: ENABLED" << std::endl;
    std::cout << "⚡ Cosmic Harmony Algorithm: ACTIVE" << std::endl;
    std::cout << "🌟 Universal Consciousness: ONLINE" << std::endl;
    std::cout << "💝 Auto-Donate: 5% (Pool 1% + Charity 1% + Trees 1% + Ocean 1% + Space 1%)" << std::endl;
    std::cout << std::endl;
}

void print_help() {
    std::cout << "ZION AI Miner - Usage:" << std::endl;
    std::cout << "  --pool <host:port>     Mining pool address (default: 127.0.0.1:3333)" << std::endl;
    std::cout << "  --wallet <address>     Wallet address for mining rewards" << std::endl;
    std::cout << "  --worker <name>        Worker name (default: zion-ai-miner)" << std::endl;
    std::cout << "  --threads <count>      CPU mining threads (default: auto)" << std::endl;
    std::cout << "  --gpu-devices <list>   GPU device IDs (comma-separated)" << std::endl;
    std::cout << "  --ai-level <1-3>       AI enhancement level (default: 2)" << std::endl;
    std::cout << "  --cosmic-mode <on|off> Cosmic harmony mode (default: on)" << std::endl;
    std::cout << "  --log-level <0-2>      Logging level 0=error, 1=info, 2=debug" << std::endl;
    std::cout << "  --disable-donate       Disable 5% auto-donation" << std::endl;
    std::cout << "  --donate-level <0-10>  Donation percentage (default: 5)" << std::endl;
    std::cout << "  --donation-report      Show donation report" << std::endl;
    std::cout << "  --benchmark            Run performance benchmark" << std::endl;
    std::cout << "  --help                 Show this help" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  zion-miner --pool zion-pool.com:3333 --wallet ZIONxxx..." << std::endl;
    std::cout << "  zion-miner --benchmark --ai-level 3" << std::endl;
    std::cout << "  zion-miner --threads 8 --gpu-devices 0,1 --cosmic-mode on" << std::endl;
}

void run_benchmark() {
    std::cout << "🔬 Running ZION AI Mining Benchmark..." << std::endl;
    std::cout << "════════════════════════════════════════" << std::endl;
    
    // Test different AI configurations
    std::vector<int> ai_levels = {1, 2, 3};
    
    for (int level : ai_levels) {
        std::cout << "\n🧠 Testing AI Level " << level << "..." << std::endl;
        
        AIEnhancedZionMiningEngine::AIConfig ai_config;
        ai_config.ai_enhancement.neural_network_depth = level;
        ai_config.ai_enhancement.consciousness_layers = level;
        
        AIEnhancedZionMiningEngine benchmark_engine(ai_config);
        ZionMiningEngine::MiningConfig mining_config;
        mining_config.pool_host = "benchmark";
        mining_config.pool_port = 0;
        mining_config.wallet_address = "benchmark_address";
        mining_config.cpu_threads = 2; // Limited for benchmark
        
        if (benchmark_engine.initialize(mining_config)) {
            auto start_time = std::chrono::steady_clock::now();
            
            // Simulate mining for 10 seconds
            benchmark_engine.start_mining();
            std::this_thread::sleep_for(std::chrono::seconds(10));
            benchmark_engine.stop_mining();
            
            auto end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            auto stats = benchmark_engine.get_stats();
            auto ai_stats = benchmark_engine.get_ai_stats();
            
            std::cout << "  ⏱️  Duration: " << duration.count() << "ms" << std::endl;
            std::cout << "  💎 Hashes: " << stats.total_hashes << std::endl;
            std::cout << "  ⚡ Hashrate: " << stats.current_hashrate << " H/s" << std::endl;
            std::cout << "  🧠 AI Gain: " << ai_stats.ai_performance_gain << "%" << std::endl;
        }
    }
    
    std::cout << "\n✅ Benchmark completed!" << std::endl;
}

int main(int argc, char* argv[]) {
    print_banner();
    
    // Set up signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
#ifndef _WIN32
    std::signal(SIGQUIT, signal_handler);
#endif

    // Parse command line arguments
    ZionMiningEngine::MiningConfig mining_config;
    AIEnhancedZionMiningEngine::AIConfig ai_config;
    bool run_benchmark_mode = false;
    bool show_help = false;
    bool show_donation_report = false;
    bool disable_donations = false;
    float donation_level = 5.0f;
    
    // Default configuration
    mining_config.pool_host = "127.0.0.1";
    mining_config.pool_port = 3333;
    mining_config.worker_name = "zion-ai-miner";
    mining_config.cpu_threads = 0; // Auto-detect
    mining_config.log_level = 1;
    
    ai_config.ai_enhancement.enable_neural_nonce_selection = true;
    ai_config.ai_enhancement.enable_cosmic_frequency_tuning = true;
    ai_config.ai_enhancement.enable_quantum_consciousness_layer = true;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            show_help = true;
        } else if (arg == "--benchmark") {
            run_benchmark_mode = true;
        } else if (arg == "--pool" && i + 1 < argc) {
            std::string pool_addr = argv[++i];
            size_t colon_pos = pool_addr.find(':');
            if (colon_pos != std::string::npos) {
                mining_config.pool_host = pool_addr.substr(0, colon_pos);
                mining_config.pool_port = std::stoi(pool_addr.substr(colon_pos + 1));
            } else {
                mining_config.pool_host = pool_addr;
            }
        } else if (arg == "--wallet" && i + 1 < argc) {
            mining_config.wallet_address = argv[++i];
        } else if (arg == "--worker" && i + 1 < argc) {
            mining_config.worker_name = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            mining_config.cpu_threads = std::stoi(argv[++i]);
        } else if (arg == "--ai-level" && i + 1 < argc) {
            int level = std::stoi(argv[++i]);
            ai_config.ai_enhancement.neural_network_depth = std::max(1, std::min(3, level));
            ai_config.ai_enhancement.consciousness_layers = std::max(1, std::min(3, level));
        } else if (arg == "--cosmic-mode" && i + 1 < argc) {
            std::string mode = argv[++i];
            bool enable = (mode == "on" || mode == "true" || mode == "1");
            ai_config.ai_enhancement.enable_cosmic_frequency_tuning = enable;
            ai_config.ai_enhancement.enable_quantum_consciousness_layer = enable;
        } else if (arg == "--log-level" && i + 1 < argc) {
            mining_config.log_level = std::stoi(argv[++i]);
        } else if (arg == "--disable-donate") {
            disable_donations = true;
        } else if (arg == "--donate-level" && i + 1 < argc) {
            donation_level = std::max(0.0f, std::min(10.0f, static_cast<float>(std::stod(argv[++i]))));
        } else if (arg == "--donation-report") {
            show_donation_report = true;
        }
    }
    
    if (show_help) {
        print_help();
        return 0;
    }
    
    if (run_benchmark_mode) {
        run_benchmark();
        return 0;
    }
    
    // Validate required parameters
    if (mining_config.wallet_address.empty()) {
        std::cerr << "❌ Error: Wallet address is required!" << std::endl;
        std::cerr << "Use --wallet <address> or see --help for usage." << std::endl;
        return 1;
    }
    
    // Initialize auto-donate system
    initialize_auto_donate();
    if (g_donate_manager) {
        g_donate_manager->set_user_wallet(mining_config.wallet_address);
        g_donate_manager->set_user_pool(mining_config.pool_host, mining_config.pool_port);
        g_donate_manager->enable_donations(!disable_donations);
        g_donate_manager->set_total_donation_percentage(donation_level);
    }
    
    if (show_donation_report && g_donate_manager) {
        std::cout << g_donate_manager->get_donation_report() << std::endl;
        return 0;
    }
    
    std::cout << "🔧 Configuration:" << std::endl;
    std::cout << "  Pool: " << mining_config.pool_host << ":" << mining_config.pool_port << std::endl;
    std::cout << "  Wallet: " << mining_config.wallet_address.substr(0, 20) << "..." << std::endl;
    std::cout << "  Worker: " << mining_config.worker_name << std::endl;
    std::cout << "  CPU Threads: " << (mining_config.cpu_threads == 0 ? "auto" : std::to_string(mining_config.cpu_threads)) << std::endl;
    std::cout << "  AI Level: " << ai_config.ai_enhancement.neural_network_depth << std::endl;
    std::cout << "  Cosmic Mode: " << (ai_config.ai_enhancement.enable_cosmic_frequency_tuning ? "ON" : "OFF") << std::endl;
    std::cout << "  Auto-Donate: " << (disable_donations ? "DISABLED" : (std::to_string(static_cast<int>(donation_level)) + "%")) << std::endl;
    std::cout << std::endl;
    
    // Create and initialize mining engine
    std::cout << "🚀 Initializing ZION AI Mining Engine..." << std::endl;
    g_mining_engine = std::make_unique<AIEnhancedZionMiningEngine>(ai_config);
    
    if (!g_mining_engine->initialize(mining_config)) {
        std::cerr << "❌ Failed to initialize mining engine!" << std::endl;
        return 1;
    }
    
    std::cout << "✅ Mining engine initialized successfully!" << std::endl;
    
    // Detect hardware capabilities
    auto cpu_features = g_mining_engine->detect_cpu_features();
    auto gpu_devices = g_mining_engine->detect_gpu_devices();
    
    std::cout << "\n🖥️  Detected Hardware:" << std::endl;
    std::cout << "  CPU Features: ";
    for (const auto& feature : cpu_features) {
        std::cout << feature << " ";
    }
    std::cout << std::endl;
    
    if (!gpu_devices.empty()) {
        std::cout << "  GPU Devices:" << std::endl;
        for (const auto& device : gpu_devices) {
            std::cout << "    " << device << std::endl;
        }
    }
    std::cout << std::endl;
    
    // Start mining
    std::cout << "⛏️  Starting mining operation..." << std::endl;
    if (!g_mining_engine->start_mining()) {
        std::cerr << "❌ Failed to start mining!" << std::endl;
        return 1;
    }
    
    std::cout << "✨ Mining started successfully!" << std::endl;
    std::cout << "🌟 ZION Cosmic Harmony Algorithm active!" << std::endl;
    std::cout << "🧠 AI enhancements online!" << std::endl;
    std::cout << "Press Ctrl+C to stop mining..." << std::endl;
    std::cout << std::endl;
    
    // Main mining loop with periodic stats
    auto last_stats_time = std::chrono::steady_clock::now();
    
    while (!g_shutdown_requested && g_mining_engine->is_mining()) {
        std::this_thread::sleep_for(std::chrono::seconds(30)); // Update every 30 seconds
        
        if (g_shutdown_requested) break;
        
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_stats_time);
        
        if (elapsed.count() >= 30) {
            // Update donation cycle
            if (g_donate_manager) {
                g_donate_manager->update_donation_cycle();
            }
            
            // Print mining statistics
            auto stats = g_mining_engine->get_stats();
            auto ai_stats = g_mining_engine->get_ai_stats();
            
            std::cout << "📊 Mining Stats:" << std::endl;
            std::cout << "  ⚡ Hashrate: " << static_cast<int>(stats.current_hashrate) << " H/s" << std::endl;
            std::cout << "  💎 Total Hashes: " << stats.total_hashes << std::endl;
            std::cout << "  ✅ Accepted: " << stats.accepted_shares << std::endl;
            std::cout << "  ❌ Rejected: " << stats.rejected_shares << std::endl;
            std::cout << "  🧠 AI Gain: " << ai_stats.ai_performance_gain << "%" << std::endl;
            std::cout << "  ⏱️  Uptime: " << stats.uptime_seconds << "s" << std::endl;
            
            // Show donation status
            if (g_donate_manager && !disable_donations) {
                auto donate_stats = g_donate_manager->get_stats();
                std::cout << "  💝 Donated: " << donate_stats.current_donate_percentage << "% to " << donate_stats.current_donate_target << std::endl;
            }
            
            if (g_mining_engine->is_ai_enabled()) {
                auto ai_learning = g_mining_engine->ai_engine_->get_learning_stats();
                auto consciousness = g_mining_engine->ai_engine_->get_consciousness_state();
                
                std::cout << "  🌟 Consciousness Level: " << consciousness.cosmic_alignment << std::endl;
                std::cout << "  📈 Success Rate: " << (ai_learning.success_rate * 100.0) << "%" << std::endl;
            }
            
            std::cout << std::endl;
            last_stats_time = now;
        }
    }
    
    // Shutdown
    std::cout << "🛑 Shutting down mining engine..." << std::endl;
    g_mining_engine->stop_mining();
    
    // Print final statistics
    auto final_stats = g_mining_engine->get_stats();
    auto final_ai_stats = g_mining_engine->get_ai_stats();
    
    std::cout << "\n📈 Final Statistics:" << std::endl;
    std::cout << "  Total Hashes: " << final_stats.total_hashes << std::endl;
    std::cout << "  Accepted Shares: " << final_stats.accepted_shares << std::endl;
    std::cout << "  Rejected Shares: " << final_stats.rejected_shares << std::endl;
    std::cout << "  AI Performance Gain: " << final_ai_stats.ai_performance_gain << "%" << std::endl;
    std::cout << "  Total Runtime: " << final_stats.uptime_seconds << " seconds" << std::endl;
    
    std::cout << "\n🌟 Thank you for using ZION AI Miner!" << std::endl;
    
    // Shutdown auto-donate and show final report
    shutdown_auto_donate();
    
    std::cout << "✨ May the Cosmic Harmony be with you! ✨" << std::endl;
    
    return 0;
}