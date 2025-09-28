#include <iostream>
#include <vector>
#include <thread>
#include <csignal>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <conio.h>  // Windows specific for _kbhit() and _getch()
#include "../mining/core/job_manager.h"
#include "../mining/core/cpu_worker.h"
#include "../mining/core/share_submitter.h"
#include "../mining/core/mining_stats.h"
#include "../network/pool-connection.h"
#include "../network/stratum_client.h"
#ifdef ZION_HAVE_RANDOMX
#include "../../include/randomx_wrapper.h"
#endif

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

static std::atomic<bool> g_running{true};
static std::atomic<bool> g_gpu_enabled{false};
static std::atomic<int> g_gpu_algorithm{0}; // 0=RandomX, 1=Ethash, 2=KawPow
static std::atomic<bool> g_show_stats{true};
static std::atomic<bool> g_show_hashrate_details{false};

static ZionMiningStatsAggregator* g_stats = nullptr;
static std::vector<std::unique_ptr<ZionCpuWorker>>* g_workers = nullptr;

const char* gpu_algorithms[] = {"RandomX", "Ethash", "KawPow", "Autolykos2"};
const int gpu_algorithm_count = 4;

void signal_handler(int){ g_running.store(false); }

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
    
    std::cout << GREEN << " * " << WHITE << "ZION AI Mining Engine v1.1.0 (Interactive)" << RESET << std::endl;
    std::cout << GREEN << " * " << WHITE << "RandomX Core + GPU Support" << RESET << std::endl;
    std::cout << GREEN << " * " << WHITE << "Real-time Statistics & Controls" << RESET << std::endl;
    std::cout << std::endl;
}

void print_controls() {
    std::cout << YELLOW << "OVLÁDÁNÍ:" << RESET << std::endl;
    std::cout << WHITE << " [s] " << CYAN << "Statistiky ON/OFF" << RESET << std::endl;
    std::cout << WHITE << " [h] " << CYAN << "Detailní hashrate ON/OFF" << RESET << std::endl;
    std::cout << WHITE << " [g] " << (g_gpu_enabled.load() ? GREEN : RED) << "GPU mining " << (g_gpu_enabled.load() ? "ON" : "OFF") << RESET << std::endl;
    std::cout << WHITE << " [o] " << MAGENTA << "GPU algoritmus: " << gpu_algorithms[g_gpu_algorithm.load()] << RESET << std::endl;
    std::cout << WHITE << " [q] " << RED << "Ukončit" << RESET << std::endl;
    std::cout << std::endl;
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

void print_mining_stats(const ZionMiningSnapshot& snap, std::chrono::seconds uptime) {
    if (!g_show_stats.load()) return;
    
    std::cout << WHITE << "╔══════════════════════════════════════════════════════════════╗" << RESET << std::endl;
    std::cout << WHITE << "║ " << CYAN << "UNIT " << WHITE << "│ " << CYAN << std::setw(12) << "HASHRATE" << WHITE << " │ " 
              << CYAN << std::setw(8) << "FOUND" << WHITE << " │ " << CYAN << std::setw(8) << "ACCEPT" << WHITE << " │ " 
              << CYAN << std::setw(8) << "REJECT" << WHITE << " ║" << RESET << std::endl;
    std::cout << WHITE << "╠══════════════════════════════════════════════════════════════╣" << RESET << std::endl;
    
    uint64_t total_rejects = (snap.shares_found > snap.shares_accepted) ? (snap.shares_found - snap.shares_accepted) : 0;
    
    // CPU stats
    std::cout << WHITE << "║ " << CYAN << "CPU " << WHITE << " │ " 
              << std::setw(12) << format_hashrate((uint64_t)snap.total_hashrate) << WHITE << " │ "
              << std::setw(8) << snap.shares_found << WHITE << " │ "
              << std::setw(8) << snap.shares_accepted << WHITE << " │ "
              << std::setw(8) << total_rejects << WHITE << " ║" << RESET << std::endl;
    
    // GPU stats (simulated for now)
    if (g_gpu_enabled.load()) {
        uint64_t gpu_hashrate = g_gpu_algorithm.load() == 0 ? 45000000 : // RandomX
                               g_gpu_algorithm.load() == 1 ? 85000000 :  // Ethash
                               g_gpu_algorithm.load() == 2 ? 35000000 :  // KawPow
                                                            55000000;    // Autolykos2
        std::cout << WHITE << "║ " << MAGENTA << "GPU " << WHITE << " │ " 
                  << std::setw(12) << format_hashrate(gpu_hashrate) << WHITE << " │ "
                  << std::setw(8) << "0" << WHITE << " │ "
                  << std::setw(8) << "0" << WHITE << " │ "
                  << std::setw(8) << "0" << WHITE << " ║" << RESET << std::endl;
    }
    
    std::cout << WHITE << "╚══════════════════════════════════════════════════════════════╝" << RESET << std::endl;
    
    // Summary line
    double acceptance_rate = snap.shares_found > 0 ? (static_cast<double>(snap.shares_accepted) / snap.shares_found * 100) : 0;
    std::cout << YELLOW << "UPTIME " << WHITE << format_uptime(uptime) << RESET;
    std::cout << YELLOW << " │ SHARES " << WHITE << snap.shares_accepted << "/" << snap.shares_found;
    if (acceptance_rate >= 95) {
        std::cout << GREEN;
    } else if (acceptance_rate >= 90) {
        std::cout << YELLOW;
    } else {
        std::cout << RED;
    }
    std::cout << " (" << std::fixed << std::setprecision(1) << acceptance_rate << "%)" << RESET;
    
    if (g_gpu_enabled.load()) {
        std::cout << YELLOW << " │ GPU ALG " << MAGENTA << gpu_algorithms[g_gpu_algorithm.load()] << RESET;
    }
    std::cout << std::endl << std::endl;
}

void print_hashrate_details(const ZionMiningSnapshot& snap) {
    if (!g_show_hashrate_details.load()) return;
    
    std::cout << CYAN << "Per-Thread Hashrate:" << RESET << std::endl;
    for (size_t i = 0; i < snap.per_thread_hashrate.size(); ++i) {
        std::cout << WHITE << " Thread " << std::setw(2) << i << ": " 
                  << std::setw(10) << format_hashrate((uint64_t)snap.per_thread_hashrate[i]) << RESET << std::endl;
    }
    std::cout << std::endl;
}

void print_recent_events(const ZionMiningSnapshot& snap) {
    if (snap.recent_events.empty()) return;
    
    std::cout << YELLOW << "Recent Share Events:" << RESET << std::endl;
    for (const auto& event : snap.recent_events) {
        auto now = std::chrono::steady_clock::now();
        auto age = std::chrono::duration_cast<std::chrono::seconds>(now - event.ts);
        
        std::cout << "[" << std::setw(3) << age.count() << "s] ";
        if (event.accepted) {
            std::cout << GREEN << "✓ ACCEPTED" << RESET;
        } else {
            std::cout << RED << "✗ REJECTED" << RESET;
        }
        std::cout << " nonce:" << std::hex << event.nonce << std::dec 
                  << " diff:" << event.difficulty << std::endl;
    }
    std::cout << std::endl;
}

void handle_keypress() {
    while (g_running.load()) {
#ifdef _WIN32
        if (_kbhit()) {
            char key = _getch();
#else
        // Linux/Mac version would need different approach
        char key = 0;
        // TODO: implement non-blocking keyboard input for Linux
#endif
            switch (tolower(key)) {
                case 's':
                    g_show_stats.store(!g_show_stats.load());
                    std::cout << YELLOW << "Statistiky: " << (g_show_stats.load() ? "ON" : "OFF") << RESET << std::endl;
                    break;
                case 'h':
                    g_show_hashrate_details.store(!g_show_hashrate_details.load());
                    std::cout << YELLOW << "Detailní hashrate: " << (g_show_hashrate_details.load() ? "ON" : "OFF") << RESET << std::endl;
                    break;
                case 'g':
                    g_gpu_enabled.store(!g_gpu_enabled.load());
                    std::cout << YELLOW << "GPU mining: " << (g_gpu_enabled.load() ? GREEN "ON" : RED "OFF") << RESET << std::endl;
                    // TODO: Actually start/stop GPU workers here
                    break;
                case 'o':
                    if (g_gpu_enabled.load()) {
                        int current = g_gpu_algorithm.load();
                        g_gpu_algorithm.store((current + 1) % gpu_algorithm_count);
                        std::cout << YELLOW << "GPU algoritmus změněn na: " << MAGENTA << gpu_algorithms[g_gpu_algorithm.load()] << RESET << std::endl;
                        // TODO: Actually switch GPU algorithm here
                    } else {
                        std::cout << RED << "GPU mining není zapnuto!" << RESET << std::endl;
                    }
                    break;
                case 'q':
                    std::cout << RED << "Ukončuji..." << RESET << std::endl;
                    g_running.store(false);
                    break;
            }
#ifdef _WIN32
        }
#endif
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

int main(int argc, char* argv[]) {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::string wallet = argc>1? argv[1]: "Z3ExampleWalletAddress";
    std::string pool_host = argc>2? argv[2]: "127.0.0.1";
    int pool_port = argc>3? std::atoi(argv[3]) : 3333;
    bool disable_stratum = false;
    for(int i=1;i<argc;i++){ if(std::string(argv[i]) == "--no-stratum") disable_stratum = true; }

    clear_screen();
    print_banner();

    std::cout << YELLOW << "Wallet: " << WHITE << wallet << RESET << std::endl;
    std::cout << YELLOW << "Pool:   " << WHITE << pool_host << ":" << pool_port << RESET << std::endl;
    std::cout << std::endl;

#ifdef ZION_HAVE_RANDOMX
    zion::Hash dummy_seed; dummy_seed.fill(0x11);
    if(!zion::RandomXWrapper::instance().initialize(dummy_seed, false)){
        std::cerr << RED << "Failed to init RandomX – exiting" << RESET << std::endl; 
        return 1; 
    }
#else
    std::cout << YELLOW << "RandomX not available (library missing) – running stub mode." << RESET << std::endl;
#endif

    PoolConnection pool(pool_host, pool_port, wallet);
    pool.connect();

    ZionJobManager job_manager;
    std::unique_ptr<StratumClient> stratum;
    if(!disable_stratum){
        stratum.reset(new StratumClient(&job_manager, pool_host, pool_port, wallet));
        if(!stratum->start()){
            std::cerr << RED << "Failed to start Stratum client, falling back to bootstrap dummy job" << RESET << std::endl;
        }
    }
    if(disable_stratum || !stratum || !stratum->running()){
        ZionMiningJob job; job.job_id = "bootstrap"; job.seed_hash = "00"; job.target_difficulty = 100000; job.height=1; job.blob.resize(80, 0);
        job_manager.update_job(job);
    }

    unsigned threads = std::max(1u, std::thread::hardware_concurrency());
    ZionMiningStatsAggregator stats(threads);
    g_stats = &stats;
    
    ZionShareSubmitter submitter(&pool, stratum.get(), &stats);
    submitter.set_result_callback([&](const ZionShare& sh, bool ok){
        // Live share event - trigger immediate UI notification
        std::string status = ok ? "✓ ACCEPTED" : "✗ REJECTED";
        std::string color = ok ? GREEN : RED;
        std::cout << "\r" << color << "[LIVE] " << status << RESET 
                  << " share (job=" << sh.job_id << ", nonce=" << std::hex << sh.nonce << std::dec 
                  << ", diff=" << sh.difficulty << ")" << std::endl;
        std::cout.flush();
    });
    submitter.start();

    std::vector<std::unique_ptr<ZionCpuWorker>> workers;
    g_workers = &workers;
    workers.reserve(threads);
    for(unsigned i=0;i<threads;i++){
        ZionCpuWorkerConfig cfg; cfg.index=i; cfg.total_threads=threads; cfg.use_randomx=true;
        workers.emplace_back(new ZionCpuWorker(&job_manager, &submitter, cfg, &stats));
        workers.back()->start();
    }

    // Start keyboard handler thread
    std::thread key_thread(handle_keypress);

    auto start = std::chrono::steady_clock::now();
    print_controls();
    
    while(g_running.load()){
        std::this_thread::sleep_for(std::chrono::seconds(3));
        
        auto now = std::chrono::steady_clock::now();
        auto uptime = std::chrono::duration_cast<std::chrono::seconds>(now-start);
        
        // Move cursor up to overwrite previous stats
        if (g_show_stats.load()) {
            for (int i = 0; i < 15; i++) {
                std::cout << "\033[A\033[2K";
            }
        }
        
        auto snap = stats.snapshot();
        print_mining_stats(snap, uptime);
        print_hashrate_details(snap);
        print_recent_events(snap);
        print_controls();
    }

    key_thread.join();
    
    for(auto& w: workers) w->stop();
    submitter.stop();
    if(stratum) stratum->stop();
    pool.disconnect();
#ifdef ZION_HAVE_RANDOMX
    zion::RandomXWrapper::instance().cleanup();
#endif
    
    std::cout << GREEN << "Shutdown complete." << RESET << std::endl;
    return 0;
}