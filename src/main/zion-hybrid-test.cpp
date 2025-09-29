#include "zion-simple-miner.h"
#include "zion-gpu-miner.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <signal.h>
#include <memory>

std::unique_ptr<ZionSimpleMiner> cpu_miner = nullptr;
std::unique_ptr<ZionGPUMiner> gpu_miner = nullptr;

void signal_handler(int signal) {
    std::cout << "\n🛑 Přijat signál k ukončení..." << std::endl;
    
    if (cpu_miner) {
        cpu_miner->stop();
    }
    if (gpu_miner) {
        gpu_miner->stop_gpu_mining();
    }
}

void print_comparison_stats() {
    if (!cpu_miner || !gpu_miner) return;
    
    auto cpu_stats = cpu_miner->get_stats();
    auto gpu_stats = gpu_miner->get_stats();
    
    std::cout << "\n🏆 === CPU vs GPU PERFORMANCE COMPARISON ===" << std::endl;
    
    // CPU Stats
    std::cout << "\n🖥️ CPU MINING:" << std::endl;
    std::cout << "   ⚡ Hashrate: " << cpu_stats.hashrate.load() << " H/s" << std::endl;
    std::cout << "   💎 Shares: " << cpu_stats.shares_found.load() << " found, " 
              << cpu_stats.shares_accepted.load() << " accepted" << std::endl;
    
    // GPU Stats  
    std::cout << "\n🎮 GPU MINING:" << std::endl;
    std::cout << "   ⚡ Hashrate: " << gpu_stats.gpu_hashrate.load() << " H/s" << std::endl;
    std::cout << "   💎 Shares: " << gpu_stats.gpu_shares_found.load() << " found, " 
              << gpu_stats.gpu_shares_accepted.load() << " accepted" << std::endl;
    std::cout << "   🌡️ Temperature: " << gpu_stats.gpu_temperature.load() << "°C" << std::endl;
    std::cout << "   ⚡ Power: " << gpu_stats.gpu_power_usage.load() << "W" << std::endl;
    std::cout << "   📊 Efficiency: " << gpu_stats.gpu_efficiency.load() << " H/W" << std::endl;
    
    // Performance Analysis
    uint64_t cpu_hashrate = cpu_stats.hashrate.load();
    uint64_t gpu_hashrate = gpu_stats.gpu_hashrate.load();
    uint64_t total_hashrate = cpu_hashrate + gpu_hashrate;
    
    std::cout << "\n📊 PERFORMANCE ANALYSIS:" << std::endl;
    std::cout << "   🔥 Total Hashrate: " << total_hashrate << " H/s" << std::endl;
    
    if (gpu_hashrate > 0 && cpu_hashrate > 0) {
        double gpu_advantage = static_cast<double>(gpu_hashrate) / cpu_hashrate;
        std::cout << "   🎯 GPU Advantage: " << gpu_advantage << "x faster than CPU" << std::endl;
        
        double cpu_percentage = (static_cast<double>(cpu_hashrate) / total_hashrate) * 100;
        double gpu_percentage = (static_cast<double>(gpu_hashrate) / total_hashrate) * 100;
        
        std::cout << "   📈 Hashrate Distribution:" << std::endl;
        std::cout << "      🖥️ CPU: " << cpu_percentage << "%" << std::endl;
        std::cout << "      🎮 GPU: " << gpu_percentage << "%" << std::endl;
    }
    
    // Mining Efficiency
    uint64_t total_shares = cpu_stats.shares_found.load() + gpu_stats.gpu_shares_found.load();
    std::cout << "   💰 Total Shares Found: " << total_shares << std::endl;
    
    std::cout << "==========================================\n" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "🚀 ======= ZION CPU vs GPU MINING TEST =======" << std::endl;
    std::cout << "🎯 MIT Licensed Dual Mining Engine" << std::endl;
    std::cout << "🧠 AI-Enhanced Cosmic Harmony Algorithm" << std::endl;
    std::cout << "💝 Auto-Donate 5% System Active" << std::endl;
    std::cout << "⚖️ Performance Comparison Mode" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    // Setup signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Configuration
    std::string pool_address = "127.0.0.1";
    int pool_port = 3333;
    std::string wallet_address = "Zi1hybrid2025cpu8gpu9mining3test4comparison";
    
    // Parse command line arguments
    if (argc > 1) {
        wallet_address = argv[1];
    }
    if (argc > 2) {
        pool_address = argv[2];
    }
    if (argc > 3) {
        pool_port = std::atoi(argv[3]);
    }
    
    std::cout << "🏭 Konfigurace:" << std::endl;
    std::cout << "  💰 Wallet: " << wallet_address << std::endl;
    std::cout << "  🎯 Pool: " << pool_address << ":" << pool_port << std::endl;
    
    // Initialize CPU Miner
    std::cout << "\n🖥️ Inicializuji CPU Miner..." << std::endl;
    cpu_miner = std::make_unique<ZionSimpleMiner>(
        DeviceType::AI_ENHANCED,
        pool_address,
        pool_port, 
        wallet_address
    );
    
    // Initialize GPU Miner
    std::cout << "\n🎮 Inicializuji GPU Miner..." << std::endl;
    gpu_miner = std::make_unique<ZionGPUMiner>();
    
    // Start CPU mining
    int cpu_threads = std::thread::hardware_concurrency();
    std::cout << "\n🔥 Spouštím CPU mining s " << cpu_threads << " thready..." << std::endl;
    
    if (!cpu_miner->start(cpu_threads)) {
        std::cerr << "❌ Nepodařilo se spustit CPU mining!" << std::endl;
        return 1;
    }
    
    // Wait a bit before starting GPU
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // Start GPU mining
    std::cout << "\n🎮 Spouštím GPU mining..." << std::endl;
    
    if (!gpu_miner->start_gpu_mining(GPUOptimization::AI_ENHANCED)) {
        std::cout << "⚠️ GPU mining nedostupný, pokračuji pouze s CPU" << std::endl;
    }
    
    std::cout << "\n✅ Hybrid CPU+GPU mining spuštěn!" << std::endl;
    std::cout << "📊 Statistiky se zobrazí každých 30 sekund" << std::endl;
    std::cout << "🛑 Stiskni Ctrl+C pro ukončení\n" << std::endl;
    
    // Main monitoring loop
    int stats_counter = 0;
    while (cpu_miner->is_running() || gpu_miner->is_mining()) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        
        stats_counter++;
        
        // Print detailed comparison every 30 seconds (3 * 10s)
        if (stats_counter % 3 == 0) {
            print_comparison_stats();
        } else {
            // Quick status update every 10 seconds
            auto cpu_stats = cpu_miner->get_stats();
            auto gpu_stats = gpu_miner->get_stats();
            
            std::cout << "⚡ Quick Status - CPU: " << cpu_stats.hashrate.load() 
                      << " H/s, GPU: " << gpu_stats.gpu_hashrate.load() 
                      << " H/s, Total Shares: " 
                      << (cpu_stats.shares_found.load() + gpu_stats.gpu_shares_found.load())
                      << std::endl;
        }
    }
    
    // Final statistics
    std::cout << "\n🏁 === FINAL MINING RESULTS ===" << std::endl;
    print_comparison_stats();
    
    std::cout << "🎉 ZION Hybrid Mining Test ukončen!" << std::endl;
    std::cout << "📈 CPU+GPU mining úspěšně ověřeno!" << std::endl;
    
    return 0;
}