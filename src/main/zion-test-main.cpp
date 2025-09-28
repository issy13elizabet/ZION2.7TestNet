#include "zion-simple-miner.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <signal.h>

std::unique_ptr<ZionSimpleMiner> global_miner = nullptr;

void signal_handler(int signal) {
    if (global_miner) {
        std::cout << "\n🛑 Přijat signál k ukončení..." << std::endl;
        global_miner->stop();
    }
}

int main(int argc, char* argv[]) {
    std::cout << "🚀 ======= ZION AI MINER 2025 =======" << std::endl;
    std::cout << "🎯 MIT Licensed Cross-Platform Mining Engine" << std::endl;
    std::cout << "🧠 AI-Enhanced Cosmic Harmony Algorithm" << std::endl;
    std::cout << "💝 Auto-Donate 5% System Active" << std::endl;
    std::cout << "=======================================" << std::endl;
    
    // Setup signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Configuration
    std::string pool_address = "127.0.0.1";  // Local ZION pool
    int pool_port = 3333;
    std::string wallet_address = "Zi1miner2025testing3address4local5zion";
    
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
    
    // Create and start miner
    global_miner = std::make_unique<ZionSimpleMiner>(
        DeviceType::AI_ENHANCED, 
        pool_address, 
        pool_port, 
        wallet_address
    );
    
    // Start mining
    int thread_count = std::thread::hardware_concurrency();
    std::cout << "🔥 Spouštím mining s " << thread_count << " thready..." << std::endl;
    
    if (!global_miner->start(thread_count)) {
        std::cerr << "❌ Nepodařilo se spustit mining!" << std::endl;
        return 1;
    }
    
    std::cout << "✅ Mining spuštěn! Stiskni Ctrl+C pro ukončení." << std::endl;
    
    // Main loop - print statistics every 30 seconds
    while (global_miner->is_running()) {
        std::this_thread::sleep_for(std::chrono::seconds(30));
        
        if (global_miner->is_running()) {
            global_miner->print_stats();
        }
    }
    
    std::cout << "🏁 ZION AI Miner ukončen. Děkujeme za mining!" << std::endl;
    return 0;
}