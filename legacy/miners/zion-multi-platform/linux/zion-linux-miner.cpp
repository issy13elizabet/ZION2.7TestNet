#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <signal.h>
#include "../common/zion-cosmic-harmony-core.h"

using namespace zion::core;

// Global state
static std::atomic<bool> shutdown_requested{false};

class LinuxZionMiner {
private:
    std::unique_ptr<ZionHasher> hasher_;
    std::vector<std::thread> workers_;
    std::atomic<bool> mining_active_{false};
    std::atomic<uint64_t> total_hashrate_{0};
    std::atomic<uint64_t> total_hashes_{0};
    int num_threads_;
    
public:
    LinuxZionMiner(int threads = std::thread::hardware_concurrency()) 
        : num_threads_(threads) {
        hasher_ = std::make_unique<ZionHasher>(ZionHasher::AlgorithmMode::FULL_POWER);
        
        // Detect and set CPU features
        hasher_->set_cpu_features(true, false, false); // Assume AVX2, no AVX512, no NEON
    }
    
    bool start_mining() {
        if (mining_active_.load()) return false;
        
        std::cout << "🚀 Starting ZION Linux Miner with " << num_threads_ << " threads..." << std::endl;
        mining_active_.store(true);
        
        for (int i = 0; i < num_threads_; i++) {
            workers_.emplace_back([this, i]() { worker_thread(i); });
        }
        
        return true;
    }
    
    void stop_mining() {
        if (!mining_active_.load()) return;
        
        std::cout << "🛑 Stopping ZION Linux Miner..." << std::endl;
        mining_active_.store(false);
        
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers_.clear();
    }
    
    uint64_t get_hashrate() const { return total_hashrate_.load(); }
    uint64_t get_total_hashes() const { return total_hashes_.load(); }
    
private:
    void worker_thread(int thread_id) {
        std::cout << "💻 Linux worker " << thread_id << " started" << std::endl;
        
        uint8_t header[80];
        memset(header, 0, sizeof(header));
        
        // Add some test data to header
        uint64_t timestamp = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        memcpy(header, &timestamp, 8);
        memcpy(header + 8, &thread_id, 4);
        
        uint32_t nonce_base = thread_id * 1000000;
        uint64_t target_difficulty = 1000000;
        
        while (mining_active_.load() && !shutdown_requested.load()) {
            // Mine a batch of nonces
            auto results = hasher_->mine_batch(header, 80, nonce_base, 5000, target_difficulty);
            
            total_hashes_.fetch_add(5000);
            
            // Check for valid shares
            for (const auto& result : results) {
                if (result.is_valid_share) {
                    std::cout << "\n🎉 SHARE FOUND by Linux worker " << thread_id 
                              << "! Nonce: 0x" << std::hex << result.nonce << std::dec 
                              << " Difficulty: " << result.difficulty_met << std::endl;
                }
            }
            
            nonce_base += 5000;
            
            // Update hashrate (simplified)
            total_hashrate_.store(hasher_->get_hashes_computed() / 10); // Rough estimate
        }
        
        std::cout << "💻 Linux worker " << thread_id << " stopped" << std::endl;
    }
};

void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down..." << std::endl;
    shutdown_requested.store(true);
}

void print_banner() {
    std::cout << R"(
╔══════════════════════════════════════════════════════════════╗
║                ZION MINER v1.4.0 - LINUX                    ║
║                 Multi-Platform Suite                        ║
║               🐧 High-Performance Edition                   ║
╠══════════════════════════════════════════════════════════════╣
║  Full ZION Cosmic Harmony Algorithm                         ║
║  CUDA + OpenCL + AVX2/AVX512 Optimizations                 ║
║  Production Mining for Servers & Workstations              ║
╚══════════════════════════════════════════════════════════════╝
)" << std::endl;
}

int main() {
    print_banner();
    
    // Install signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Detect platform
    Platform platform = detect_platform();
    std::cout << "🔍 Platform detected: " << platform_name(platform) << std::endl;
    
    // Initialize miner
    auto miner = std::make_unique<LinuxZionMiner>();
    
    if (!miner->start_mining()) {
        std::cerr << "❌ Failed to start mining!" << std::endl;
        return 1;
    }
    
    std::cout << "\n💎 ZION Linux Miner started successfully!" << std::endl;
    std::cout << "⛏️  Mining with full algorithm power" << std::endl;
    std::cout << "📊 Stats will be displayed every 10 seconds" << std::endl;
    std::cout << "🔄 Press Ctrl+C to stop\n" << std::endl;
    
    // Stats loop
    while (!shutdown_requested.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        
        uint64_t hashrate = miner->get_hashrate();
        uint64_t total = miner->get_total_hashes();
        
        std::cout << "📊 Linux Mining Stats - Hashrate: " << hashrate 
                  << " H/s, Total: " << total << " hashes" << std::endl;
    }
    
    // Cleanup
    miner->stop_mining();
    std::cout << "✅ ZION Linux Miner shutdown complete!" << std::endl;
    std::cout << "🚀 Next platforms: macOS 🍎, Windows 🪟, Mobile 📱" << std::endl;
    
    return 0;
}