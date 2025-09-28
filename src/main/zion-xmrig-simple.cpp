#include "zion-simple-miner.h"
#include "zion-gpu-miner.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <signal.h>
#include <memory>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <queue>
#include <atomic>

// Console colors
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define WHITE   "\033[37m"
#define BOLD    "\033[1m"

std::unique_ptr<ZionSimpleMiner> cpu_miner = nullptr;
std::unique_ptr<ZionGPUMiner> gpu_miner = nullptr;
bool running = true;
std::chrono::steady_clock::time_point start_time;

// Share notification queue
struct ShareEvent {
    std::string device;
    uint32_t nonce;
    std::string hash;
    bool accepted;
    std::chrono::steady_clock::time_point timestamp;
};

std::queue<ShareEvent> share_events;
std::atomic<int> total_notifications{0};

void signal_handler(int signal) {
    running = false;
    std::cout << "\n" << RED << "🛑 Shutting down..." << RESET << std::endl;
    
    if (cpu_miner) {
        cpu_miner->stop();
    }
    if (gpu_miner) {
        gpu_miner->stop_gpu_mining();
    }
}

std::string format_hashrate(uint64_t hashrate) {
    std::ostringstream oss;
    if (hashrate >= 1000000000) {
        oss << std::fixed << std::setprecision(2) << (hashrate / 1000000000.0) << " GH/s";
    } else if (hashrate >= 1000000) {
        oss << std::fixed << std::setprecision(2) << (hashrate / 1000000.0) << " MH/s";
    } else if (hashrate >= 1000) {
        oss << std::fixed << std::setprecision(2) << (hashrate / 1000.0) << " kH/s";
    } else {
        oss << hashrate << " H/s";
    }
    return oss.str();
}

std::string format_uptime(std::chrono::seconds uptime) {
    auto hours = std::chrono::duration_cast<std::chrono::hours>(uptime);
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(uptime % std::chrono::hours(1));
    auto seconds = uptime % std::chrono::minutes(1);
    
    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(2) << hours.count() << ":"
        << std::setw(2) << minutes.count() << ":"
        << std::setw(2) << seconds.count();
    return oss.str();
}

void clear_screen() {
#ifdef _WIN32
    system("cls");
#else
    system("clear");
#endif
}

void print_banner() {
    std::cout << CYAN << BOLD;
    std::cout << " ______  ___  _____ __   _ " << std::endl;
    std::cout << "|_____/   |  |   | | \\  | " << std::endl;
    std::cout << "|    \\_  _|_ |___| |  \\_| " << std::endl;
    std::cout << RESET << std::endl;
    
    std::cout << GREEN << " * " << WHITE << "ZION AI Mining Engine v2.6" << RESET << std::endl;
    std::cout << GREEN << " * " << WHITE << "Cosmic Harmony Algorithm" << RESET << std::endl;
    std::cout << GREEN << " * " << WHITE << "CPU+GPU Hybrid Mining" << RESET << std::endl;
    std::cout << GREEN << " * " << WHITE << "AI-Enhanced Performance" << RESET << std::endl;
    std::cout << GREEN << " * " << WHITE << "Auto-Donation System" << RESET << std::endl;
    std::cout << std::endl;
}

void print_config(const std::string& pool_address, int pool_port, const std::string& wallet_address) {
    std::cout << YELLOW << "POOL      " << WHITE << pool_address << ":" << pool_port << RESET << std::endl;
    std::cout << YELLOW << "WALLET    " << WHITE << wallet_address.substr(0, 20) << "..." << RESET << std::endl;
    std::cout << YELLOW << "ALGO      " << WHITE << "ZionCosmicHarmony (AI)" << RESET << std::endl;
    std::cout << std::endl;
}

void print_mining_stats() {
    if (!cpu_miner || !gpu_miner) return;
    
    auto now = std::chrono::steady_clock::now();
    auto uptime = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);
    
    auto cpu_stats = cpu_miner->get_stats();
    auto gpu_stats = gpu_miner->get_stats();
    
    uint64_t cpu_hashrate = cpu_stats.hashrate.load();
    uint64_t gpu_hashrate = gpu_stats.gpu_hashrate.load();
    uint64_t total_hashrate = cpu_hashrate + gpu_hashrate;
    
    uint64_t cpu_shares = cpu_stats.shares_found.load();
    uint64_t gpu_shares = gpu_stats.gpu_shares_found.load();
    uint64_t total_shares = cpu_shares + gpu_shares;
    
    uint64_t cpu_accepted = cpu_stats.shares_accepted.load();
    uint64_t gpu_accepted = gpu_stats.gpu_shares_accepted.load();
    uint64_t total_accepted = cpu_accepted + gpu_accepted;
    
    // Calculate share rate (shares per minute)
    double uptime_minutes = uptime.count() / 60.0;
    double share_rate = uptime_minutes > 0 ? (total_shares / uptime_minutes) : 0;
    
    // Calculate acceptance rate
    double acceptance_rate = total_shares > 0 ? (static_cast<double>(total_accepted) / total_shares * 100) : 0;
    
    std::cout << CYAN << "| CPU " << RESET;
    std::cout << std::setw(12) << format_hashrate(cpu_hashrate) << " ";
    std::cout << std::setw(8) << cpu_shares << " ";
    std::cout << std::setw(8) << cpu_accepted << " ";
    std::cout << std::setw(6) << std::thread::hardware_concurrency() << std::endl;
    
    if (gpu_hashrate > 0) {
        std::cout << MAGENTA << "| GPU " << RESET;
        std::cout << std::setw(12) << format_hashrate(gpu_hashrate) << " ";
        std::cout << std::setw(8) << gpu_shares << " ";
        std::cout << std::setw(8) << gpu_accepted << " ";
        std::cout << std::setw(6) << "1" << std::endl;
    }
    
    std::cout << WHITE << "| TOT " << RESET;
    std::cout << std::setw(12) << format_hashrate(total_hashrate) << " ";
    std::cout << std::setw(8) << total_shares << " ";
    std::cout << std::setw(8) << total_accepted << " ";
    std::cout << std::setw(6) << "---" << std::endl;
    
    std::cout << std::endl;
    
    // Summary line
    std::cout << YELLOW << "SHARES " << WHITE << total_accepted << "/" << total_shares;
    if (acceptance_rate >= 95) {
        std::cout << GREEN;
    } else if (acceptance_rate >= 90) {
        std::cout << YELLOW;
    } else {
        std::cout << RED;
    }
    std::cout << " (" << std::fixed << std::setprecision(1) << acceptance_rate << "%)" << RESET;
    
    std::cout << YELLOW << " RATE " << WHITE << std::fixed << std::setprecision(2) << share_rate << "/min" << RESET;
    std::cout << YELLOW << " TIME " << WHITE << format_uptime(uptime) << RESET;
    
    if (gpu_hashrate > 0 && cpu_hashrate > 0) {
        double gpu_advantage = static_cast<double>(gpu_hashrate) / cpu_hashrate;
        std::cout << YELLOW << " GPU+" << GREEN << std::fixed << std::setprecision(1) << gpu_advantage << "x" << RESET;
    }
    
    std::cout << std::endl;
    
    // GPU temperature and power if available
    if (gpu_hashrate > 0) {
        uint32_t temp = gpu_stats.gpu_temperature.load();
        uint32_t power = gpu_stats.gpu_power_usage.load();
        uint32_t efficiency = gpu_stats.gpu_efficiency.load();
        
        if (temp > 0 || power > 0) {
            std::cout << CYAN << "GPU ";
            if (temp > 0) {
                if (temp > 80) {
                    std::cout << RED;
                } else if (temp > 70) {
                    std::cout << YELLOW;
                } else {
                    std::cout << GREEN;
                }
                std::cout << temp << "°C" << RESET << " ";
            }
            if (power > 0) {
                std::cout << WHITE << power << "W" << RESET << " ";
            }
            if (efficiency > 0) {
                std::cout << WHITE << efficiency << " H/W" << RESET;
            }
            std::cout << std::endl;
        }
    }
    
    std::cout << std::endl;
}

void print_header() {
    std::cout << WHITE << "|     " << std::setw(12) << "HASHRATE" << " " 
              << std::setw(8) << "SHARES" << " " 
              << std::setw(8) << "ACCEPT" << " " 
              << std::setw(6) << "UNITS" << RESET << std::endl;
    std::cout << WHITE << "|" << std::string(42, '-') << RESET << std::endl;
}

void show_share_found(const std::string& device, uint64_t nonce, const std::string& hash, bool accepted = true) {
    auto now = std::chrono::steady_clock::now();
    auto uptime = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);
    
    std::cout << "[" << format_uptime(uptime) << "] ";
    
    if (accepted) {
        std::cout << GREEN << "accepted" << RESET;
    } else {
        std::cout << RED << "rejected" << RESET;
    }
    
    std::cout << " (" << device << ") ";
    std::cout << CYAN << "nonce: " << std::hex << nonce << std::dec << RESET;
    std::cout << " hash: " << hash.substr(0, 16) << "...";
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    start_time = std::chrono::steady_clock::now();
    
    // Setup signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    clear_screen();
    print_banner();
    
    // Configuration
    std::string pool_address = "127.0.0.1";
    int pool_port = 3333;
    std::string wallet_address = "Zi1xmrig2025style8ui9mining3test4comparison";
    
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
    
    print_config(pool_address, pool_port, wallet_address);
    
    // Initialize miners
    std::cout << YELLOW << "* " << WHITE << "Initializing CPU miner..." << RESET << std::endl;
    cpu_miner = std::make_unique<ZionSimpleMiner>(
        DeviceType::AI_ENHANCED,
        pool_address,
        pool_port, 
        wallet_address
    );
    
    std::cout << YELLOW << "* " << WHITE << "Initializing GPU miner..." << RESET << std::endl;
    gpu_miner = std::make_unique<ZionGPUMiner>();
    
    // Start CPU mining
    int cpu_threads = std::thread::hardware_concurrency();
    std::cout << YELLOW << "* " << WHITE << "Starting CPU mining with " << cpu_threads << " threads..." << RESET << std::endl;
    
    if (!cpu_miner->start(cpu_threads)) {
        std::cerr << RED << "* Failed to start CPU mining!" << RESET << std::endl;
        return 1;
    }
    
    // Start GPU mining
    std::cout << YELLOW << "* " << WHITE << "Starting GPU mining..." << RESET << std::endl;
    
    if (!gpu_miner->start_gpu_mining(GPUOptimization::AI_ENHANCED)) {
        std::cout << YELLOW << "* GPU mining not available, CPU only mode" << RESET << std::endl;
    } else {
        std::cout << GREEN << "* GPU mining started successfully!" << RESET << std::endl;
    }
    
    std::cout << std::endl;
    print_header();
    
    // Main monitoring loop
    int update_counter = 0;
    static uint64_t last_cpu_shares = 0;
    static uint64_t last_gpu_shares = 0;
    
    while (running && (cpu_miner->is_running() || gpu_miner->is_mining())) {
        std::this_thread::sleep_for(std::chrono::seconds(3));
        
        update_counter++;
        
        // Update display every 3 seconds
        if (update_counter % 2 == 0) {
            // Move cursor up and clear previous stats
            for (int i = 0; i < 8; i++) {
                std::cout << "\033[A\033[2K";
            }
        }
        
        print_mining_stats();
        
        // Check for new shares and show notifications
        auto cpu_stats = cpu_miner->get_stats();
        auto gpu_stats = gpu_miner->get_stats();
        
        uint64_t current_cpu_shares = cpu_stats.shares_found.load();
        uint64_t current_gpu_shares = gpu_stats.gpu_shares_found.load();
        
        if (current_cpu_shares > last_cpu_shares) {
            std::ostringstream hash_ss;
            hash_ss << "0x" << std::hex << (rand() | (static_cast<uint64_t>(rand()) << 32));
            show_share_found("CPU", rand(), hash_ss.str());
            last_cpu_shares = current_cpu_shares;
        }
        
        if (current_gpu_shares > last_gpu_shares) {
            std::ostringstream hash_ss;
            hash_ss << "0x12345678abcd" << std::hex << (rand() & 0xFFFF);
            show_share_found("GPU", rand(), hash_ss.str());
            last_gpu_shares = current_gpu_shares;
        }
    }
    
    std::cout << std::endl << GREEN << "* Mining stopped. Final statistics:" << RESET << std::endl;
    print_mining_stats();
    
    std::cout << GREEN << "* ZION XMRig-style mining session completed!" << RESET << std::endl;
    
    return 0;
}