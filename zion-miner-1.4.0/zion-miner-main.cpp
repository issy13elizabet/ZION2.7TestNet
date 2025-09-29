#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <memory>
#include <signal.h>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <array>
#include <openssl/evp.h>
#include <blake3.h>

// Unified 256-bit helpers
#include "include/zion-big256.h"
static bool cryptonote_hash_meets_target(const uint8_t hash[32], const uint8_t target_be[32]){ return zion_hash_meets_target(hash, target_be); }
#ifdef __linux__
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#endif

#include "zion-cosmic-harmony.h"
#include "zion-gpu-miner-unified.h"
#include "stratum_client.h"
#include "zion-job-manager.h"
#include "zion-share-submitter.h"

using namespace zion;

// Using production StratumClient + ZionJobManager

// Global state
static std::atomic<bool> shutdown_requested{false};
static std::unique_ptr<zion::UnifiedGPUMiner> gpu_miner;
static std::unique_ptr<StratumClient> stratum_client;
static ZionJobManager g_job_manager;
static std::unique_ptr<ZionShareSubmitter> g_share_submitter;
static std::atomic<bool> g_show_stats{true};
static std::atomic<bool> g_show_detailed{false};
static std::atomic<bool> g_brief_mode{false};
static std::atomic<bool> g_reset_requested{false};
static std::atomic<bool> g_clear_screen{false};
static std::atomic<uint64_t> g_global_nonce_base{0};
static std::atomic<uint32_t> g_cpu_batch_size{10000};
static std::atomic<bool> g_gpu_active{true};
static std::atomic<int> g_algo_mode{0}; // 0=Cosmic,1=BLAKE3,2=Keccak (debug)
static std::array<uint8_t,32> g_forced_target_mask{}; // if non-zero => override target
static uint64_t g_forced_target64 = 0; // fallback 64-bit
static std::atomic<bool> g_forced_target_active{false};

static const char* algo_mode_name(int m){
    switch(m){ case 0: return "cosmic"; case 1: return "blake3"; case 2: return "keccak"; default: return "?"; }
}

// RAII terminal raw mode helper (Linux)
struct TerminalRawMode {
#ifdef __linux__
    termios orig{}; bool active=false;
    TerminalRawMode(){ if(isatty(STDIN_FILENO)){ if(tcgetattr(STDIN_FILENO,&orig)==0){ termios raw=orig; raw.c_lflag &= ~(ICANON | ECHO); raw.c_cc[VMIN]=0; raw.c_cc[VTIME]=0; if(tcsetattr(STDIN_FILENO,TCSANOW,&raw)==0){ int fl=fcntl(STDIN_FILENO,F_GETFL,0); fcntl(STDIN_FILENO,F_SETFL, fl | O_NONBLOCK); active=true; } } } }
    ~TerminalRawMode(){ if(active) tcsetattr(STDIN_FILENO,TCSANOW,&orig); }
#else
    TerminalRawMode(){}
#endif
};

// CPU mining worker
class CPUMiner {
private:
    std::vector<std::thread> workers_;
    std::atomic<bool> active_{false};
    std::atomic<uint64_t> total_hashes_{0};
    std::atomic<uint64_t> hashrate_{0};
    int num_threads_;
    
    ZionJobManager* job_manager_{};
    
public:
    CPUMiner(ZionJobManager* jm, int threads = std::thread::hardware_concurrency())
        : num_threads_(threads), job_manager_(jm) {}

    void set_threads(int t){ if(!active_) num_threads_ = t; }
    void set_batch(uint32_t b){ if(b>0) g_cpu_batch_size.store(b); }
    
    ~CPUMiner() {
        stop();
    }
    
    bool start() {
        if (active_.load()) return false;
        
        std::cout << "Starting CPU mining with " << num_threads_ << " threads..." << std::endl;
        active_.store(true);
        
        workers_.reserve(num_threads_);
        for (int i = 0; i < num_threads_; i++) {
            workers_.emplace_back([this, i]() { worker_thread(i); });
        }
        
        return true;
    }
    
    void stop() {
        if (!active_.load()) return;
        
        std::cout << "Stopping CPU mining..." << std::endl;
        active_.store(false);
        
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers_.clear();
    }
    
    // set_work no longer required (jobs pulled directly)
    
    uint64_t get_hashrate() const { return hashrate_.load(); }
    uint64_t get_total_hashes() const { return total_hashes_.load(); }
    void reset(){ total_hashes_.store(0); hashrate_.store(0); }
    
private:
    void worker_thread(int thread_id) {
        std::cout << "CPU worker " << thread_id << " started" << std::endl;
        
        auto last_hashrate_update = std::chrono::steady_clock::now();
        uint64_t hashes_since_update = 0;
        
        while (active_.load() && !shutdown_requested.load()) {
            if(!job_manager_ || !job_manager_->has_job()){
                static thread_local int idle_counter=0; if(++idle_counter % 50 == 0){ std::cout << "[Miner] Waiting for job..." << std::endl; }
                std::this_thread::sleep_for(std::chrono::milliseconds(100)); continue; }
            auto job = job_manager_->current_job();
            if(job.blob.size() < job.nonce_offset + 4){ std::this_thread::sleep_for(std::chrono::milliseconds(50)); continue; }
            uint8_t header[80]; memset(header,0,80); size_t copy_sz = std::min<size_t>(80, job.blob.size()); memcpy(header, job.blob.data(), copy_sz);
            uint64_t target = job.target_difficulty? job.target_difficulty : 0xFFFFFFFFFFFFULL; // fallback target
            std::string job_id = job.job_id;
            
            // Mine a batch of nonces
            // Fetch a contiguous batch of nonces globally to reduce collisions
            uint32_t batch_size = g_cpu_batch_size.load(std::memory_order_relaxed);
            uint64_t base = g_global_nonce_base.fetch_add(batch_size, std::memory_order_relaxed);
            uint32_t start_nonce = static_cast<uint32_t>(base & 0xFFFFFFFFULL);
            
            for (uint32_t i = 0; i < batch_size && active_.load(); i++) {
                uint32_t nonce = start_nonce + i;
                
                // Update nonce in header
                memcpy(header + 76, &nonce, 4);
                
                // Compute ZION Cosmic Harmony hash (basic variant)
                uint8_t hash[32];
                int mode = g_algo_mode.load(std::memory_order_relaxed);
                if(mode==0){
                    CosmicHarmonyHasher::cosmic_hash(header, 80, nonce, hash);
                } else if(mode==1){ // BLAKE3 only (debug)
                    blake3_hasher h; blake3_hasher_init(&h); blake3_hasher_update(&h, header, 80); blake3_hasher_finalize(&h, hash, 32);
                } else if(mode==2){ // Keccak/SHA3-256 only (debug)
                    unsigned int outl=0; EVP_Digest(header, 80, hash, &outl, EVP_sha3_256(), nullptr);
                } else {
                    CosmicHarmonyHasher::cosmic_hash(header, 80, nonce, hash);
                }
                
                hashes_since_update++;
                total_hashes_.fetch_add(1);
                
                // Check if hash meets target difficulty (256-bit pokud k dispozici)
                bool meets=false;
                if(job.target_mask[0] || job.target_mask[1]){
                    if(g_forced_target_active.load() && (g_forced_target_mask[0]||g_forced_target_mask[1])){
                        meets = cryptonote_hash_meets_target(hash, g_forced_target_mask.data());
                    } else {
                        meets = cryptonote_hash_meets_target(hash, job.target_mask.data());
                    }
                } else {
                    uint64_t hash_value=0; memcpy(&hash_value, hash, 8);
                    if(g_forced_target_active.load() && g_forced_target64>0){
                        meets = (hash_value < g_forced_target64);
                    } else {
                        meets = (hash_value < target);
                    }
                }
                if (meets && mode==0) { // submit only in canonical cosmic mode
                    static const char* hexchars = "0123456789abcdef";
                    char smallhex[17];
                    for(int b=0;b<8;++b){ uint8_t v=hash[b]; smallhex[b*2]=hexchars[v>>4]; smallhex[b*2+1]=hexchars[v & 0xF]; }
                    smallhex[16]='\0';
                    std::cout << "\n✅ SHARE (candidate) CPU worker " << thread_id << " job="<< job_id <<" nonce=0x"<< std::hex << nonce << std::dec << " hash="<< smallhex << std::endl;
                    if(g_share_submitter){
                        // simple difficulty approximation (64-bit fallback only)
                        uint64_t hv64=0; memcpy(&hv64, hash, 8);
                        uint64_t diff_est = (target>0 && hv64>0)? (target / hv64) : 0;
                        g_share_submitter->enqueue(ZionShare{job_id, nonce, smallhex, diff_est});
                    } else if(stratum_client && stratum_client->running()) {
                        uint64_t hv64=0; memcpy(&hv64, hash, 8);
                        stratum_client->submit_share(job_id, nonce, smallhex, hv64);
                    }
                }
            }
            
            // Update hashrate every 5 seconds
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_hashrate_update);
            if (elapsed.count() >= 5) {
                double thread_hashrate = hashes_since_update / elapsed.count();
                
                // This is approximate - we'd need proper aggregation for exact total
                hashrate_.store((uint64_t)(thread_hashrate * num_threads_));
                
                last_hashrate_update = now;
                hashes_since_update = 0;
            }
        }
        
        std::cout << "CPU worker " << thread_id << " stopped" << std::endl;
    }
};

void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down..." << std::endl;
    shutdown_requested.store(true);
}

void print_banner() {
    std::cout << R"(
╔══════════════════════════════════════════════════════════════╗
║                    ZION MINER v1.4.0                        ║
║                  Cosmic Harmony Algorithm                    ║
║              GPU & CPU Multi-Platform Mining                 ║
╠══════════════════════════════════════════════════════════════╣
║  Blake3 + Keccak-256 + SHA3-512 + Golden Ratio Matrix      ║
║  CUDA (NVIDIA) + OpenCL (AMD/Intel) + CPU Workers          ║
╚══════════════════════════════════════════════════════════════╝
)" << std::endl;
}

void print_mining_stats(const std::unique_ptr<CPUMiner>& cpu_miner) {
    static auto start_time = std::chrono::steady_clock::now();
    static double ema_hashrate = 0.0; // simple smoothing across total hashrate
    const double alpha = 0.25; // smoothing factor
    static double avg_difficulty = 0.0; // běžící průměr hlášené difficulty z targetu
    const double diff_alpha = 0.2;

    auto format_hashrate = [](uint64_t h) -> std::string {
        const char* units[] = {"H/s","KH/s","MH/s","GH/s","TH/s"};
        double v = (double)h; int u=0; while(v>=1000.0 && u<4){ v/=1000.0; ++u; }
        std::ostringstream oss; oss.setf(std::ios::fixed); oss.precision(v<10?2:(v<100?1:0)); oss<<v<<" "<<units[u];
        return oss.str();
    };

    auto format_uptime = [](uint64_t secs) -> std::string {
        uint64_t d = secs / 86400; secs %= 86400; uint64_t h = secs/3600; secs%=3600; uint64_t m=secs/60; uint64_t s=secs%60; std::ostringstream oss; if(d) oss<<d<<"d "; if(h) oss<<h<<"h "; if(m) oss<<m<<"m "; oss<<s<<"s"; return oss.str(); };
    
    while (!shutdown_requested.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        if(!g_show_stats.load()) continue;
        
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);
        
        uint64_t cpu_hashrate = cpu_miner ? cpu_miner->get_hashrate() : 0;
        uint64_t cpu_total = cpu_miner ? cpu_miner->get_total_hashes() : 0;
        
        uint64_t gpu_hashrate = gpu_miner ? gpu_miner->get_total_hashrate() : 0;
        uint64_t gpu_total = gpu_miner ? gpu_miner->get_total_hashes() : 0;
        
        uint64_t raw_total_hashrate = cpu_hashrate + gpu_hashrate;
        double current_diff = 0.0; double current_diff24 = 0.0;
        if(g_job_manager.has_job()){
            auto job = g_job_manager.current_job();
            current_diff = job.difficulty_display; current_diff24 = job.difficulty_24;
            if(current_diff>0){ if(avg_difficulty==0) avg_difficulty=current_diff; else avg_difficulty = diff_alpha*current_diff + (1.0-diff_alpha)*avg_difficulty; }
        }
    if(ema_hashrate == 0.0) ema_hashrate = (double)raw_total_hashrate;
    else ema_hashrate = (alpha * raw_total_hashrate) + (1.0 - alpha) * ema_hashrate;
    uint64_t total_hashrate = (uint64_t)ema_hashrate;
        uint64_t total_hashes = cpu_total + gpu_total;
        
        if(g_clear_screen.exchange(false)){
            std::cout << "\033[2J\033[H"; // clear
        }
    bool forced = g_forced_target_active.load();
    std::cout << "\n📊 MINING STATS (Uptime: " << format_uptime(elapsed.count()) << ") algo="<<algo_mode_name(g_algo_mode.load())<< (g_gpu_active.load()?" GPU:on":" GPU:off") << (forced?" FORCED-TARGET":"") << std::endl;
    uint64_t accepted = 0; uint64_t rejected=0; size_t qd=0;
    if(g_share_submitter){ accepted = g_share_submitter->accepted(); rejected = g_share_submitter->rejected(); qd = g_share_submitter->queue_depth(); }
    else if(stratum_client){ accepted = stratum_client->accepted(); rejected = stratum_client->rejected(); }
    double accept_rate = (accepted+rejected)>0 ? (100.0 * accepted / (accepted+rejected)) : 0.0;
        if(!g_brief_mode.load()){
            std::cout << "   CPU: " << format_hashrate(cpu_hashrate) << " (" << cpu_total << " total)" << std::endl;
            std::cout << "   GPU: " << format_hashrate(gpu_hashrate) << " (" << gpu_total << " total)" << std::endl;
            std::cout << " TOTAL: " << format_hashrate(total_hashrate) << " (raw "<< format_hashrate(raw_total_hashrate) <<", total " << total_hashes << ")" << std::endl;
            std::cout << " SHARES: accepted="<<accepted<<" rejected="<<rejected<<" acc%="<<std::fixed<<std::setprecision(1)<<accept_rate<<" queue="<<qd << std::endl;
            if(current_diff>0){ std::cout << "  DIFF: " << std::fixed << std::setprecision(2) << current_diff; if(current_diff24>0) std::cout << " (d24="<<std::setprecision(2)<<current_diff24<<")"; std::cout << std::endl; }
            std::cout << " STATUS: " << (stratum_client && stratum_client->running()? "Connected ✅" : "Disconnected ❌") << std::endl;
            if(g_show_detailed.load()){
                std::cout << "  Detail: worker threads=" << (cpu_miner? std::thread::hardware_concurrency():0) << " active_devices=" << (gpu_miner? gpu_miner->get_active_devices():0) << std::endl;
            }
            std::cout << "  Keys: [q] quit  [s] stats  [d] details  [b] brief  [g] gpu on/off  [o] algo cycle  [r] reset  [c] clear  [h|?] help" << std::endl;
        } else {
            std::cout << "  HTOT="<< format_hashrate(total_hashrate) << " (raw="<< format_hashrate(raw_total_hashrate) << ") shares="<<accepted<<"/"<< (accepted+rejected) << " acc%="<<std::fixed<<std::setprecision(1)<<accept_rate << " q="<<qd << (stratum_client && stratum_client->running()? " ✅":" ❌") << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {
    print_banner();
    
    // Install signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Configuration
    std::string pool_host = "91.98.122.165";
    int pool_port = 3333;
    std::string wallet_address = "ZiYgACwXhrYG9iLjRfBgEdgsGsT6DqQ2brtM8j9iR3Rs7geE5kyj7oEGkw9LpjaGX9p1h7uRNJg5BkWKu8HD28EMPpJAYUdJ4";
    
    bool enable_cpu = true;
    bool enable_gpu = true;
    int opt_cpu_threads = -1;
    uint32_t opt_cpu_batch = 0;
    StratumProtocol opt_protocol = StratumProtocol::Stratum;
    std::string opt_force_low_target_hex; // user provided low target (big-endian hex)
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        auto starts_with=[&](const char* p){ return arg.rfind(p,0)==0; };
        if (starts_with("--protocol=")) {
            std::string p = arg.substr(strlen("--protocol=")); for(char &c: p) c=(char)tolower((unsigned char)c); if(p=="cryptonote"||p=="cn") opt_protocol=StratumProtocol::CryptoNote; continue;
        }
        if (strcmp(argv[i], "--pool") == 0 && i + 1 < argc) {
            std::string pool_addr = argv[++i];
            size_t colon_pos = pool_addr.find(':');
            if (colon_pos != std::string::npos) {
                pool_host = pool_addr.substr(0, colon_pos);
                pool_port = std::stoi(pool_addr.substr(colon_pos + 1));
            }
        } else if (strcmp(argv[i], "--wallet") == 0 && i + 1 < argc) {
            wallet_address = argv[++i];
        } else if (strcmp(argv[i], "--cpu-only") == 0) {
            enable_gpu = false;
        } else if (strcmp(argv[i], "--gpu-only") == 0) {
            enable_cpu = false;
        } else if (strcmp(argv[i], "--cpu-threads") == 0 && i + 1 < argc) {
            opt_cpu_threads = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--cpu-batch") == 0 && i + 1 < argc) {
            opt_cpu_batch = (uint32_t)std::stoul(argv[++i]);
        } else if (strcmp(argv[i], "--protocol") == 0 && i + 1 < argc) {
            std::string p = argv[++i]; for(char &c: p) c=(char)tolower((unsigned char)c); if(p=="cryptonote"||p=="cn") opt_protocol=StratumProtocol::CryptoNote;
        } else if (strcmp(argv[i], "--force-low-target") == 0 && i + 1 < argc) {
            opt_force_low_target_hex = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --pool <host:port>  Mining pool address (default: 91.98.122.165:3333)" << std::endl;
            std::cout << "  --wallet <address>  Wallet address for mining rewards" << std::endl;
            std::cout << "  --cpu-only         Use CPU mining only" << std::endl;
            std::cout << "  --gpu-only         Use GPU mining only" << std::endl;
            std::cout << "  --cpu-threads <n>  Number of CPU worker threads" << std::endl;
            std::cout << "  --cpu-batch <n>    Nonce batch size per allocation (default 10000)" << std::endl;
            std::cout << "  --protocol <stratum|cryptonote>  Pool protocol (default stratum)" << std::endl;
            std::cout << "  --force-low-target <hex>  Force an easier (higher) target locally for debug (big-endian 256-bit or <=16 hex for 64-bit)" << std::endl;
            std::cout << "  --help             Show this help message" << std::endl;
            return 0;
        }
    }
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Pool: " << pool_host << ":" << pool_port << std::endl;
    std::cout << "  Wallet: " << wallet_address.substr(0, 20) << "..." << std::endl;
    std::cout << "  CPU Mining: " << (enable_cpu ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  GPU Mining: " << (enable_gpu ? "Enabled" : "Disabled") << std::endl;
    if(opt_cpu_threads>0) std::cout << "  CPU Threads: " << opt_cpu_threads << std::endl;
    if(opt_cpu_batch>0) std::cout << "  CPU Batch:   " << opt_cpu_batch << std::endl;
    std::cout << "  Protocol:    " << (opt_protocol==StratumProtocol::CryptoNote?"CryptoNote":"Stratum") << std::endl;
    if(!opt_force_low_target_hex.empty()) std::cout << "  Forced Target: " << opt_force_low_target_hex << " (debug)" << std::endl;
    std::cout << std::endl;

    // Parse forced low target if provided
    if(!opt_force_low_target_hex.empty()){
        std::string t = opt_force_low_target_hex; for(char &c: t) if(c>='A'&&c<='F') c=c-'A'+'a';
        if(t.rfind("0x",0)==0) t=t.substr(2);
        if(t.size()>64) t = t.substr(t.size()-64);
        if(t.size()<=16){ // treat as 64-bit
            g_forced_target64 = 0; for(char c: t){ g_forced_target64 <<=4; if(c>='0'&&c<='9') g_forced_target64|=(c-'0'); else if(c>='a'&&c<='f') g_forced_target64|=(10+(c-'a')); }
            if(g_forced_target64==0) g_forced_target64=1; g_forced_target_active.store(true);
        } else {
            g_forced_target_mask.fill(0); // big-endian fill
            std::string pad = std::string(64 - t.size(),'0') + t;
            for(int i=0;i<32;i++){ std::string hb = pad.substr(i*2,2); uint8_t v=0; for(char c: hb){ v <<=4; if(c>='0'&&c<='9') v|=(c-'0'); else if(c>='a'&&c<='f') v|=(10+(c-'a')); } g_forced_target_mask[i]=v; }
            // sanity: ensure not zero
            bool nz=false; for(auto b: g_forced_target_mask){ if(b){ nz=true; break; } }
            if(!nz) g_forced_target_mask[31]=1; g_forced_target_active.store(true);
        }
    }
    
    // Initialize miners
    std::unique_ptr<CPUMiner> cpu_miner;
    
    if (enable_cpu) {
        cpu_miner = std::make_unique<CPUMiner>(&g_job_manager);
        if(opt_cpu_threads>0) cpu_miner->set_threads(opt_cpu_threads);
        if(opt_cpu_batch>0) cpu_miner->set_batch(opt_cpu_batch);
    }
    
    if (enable_gpu) {
        gpu_miner = std::make_unique<zion::UnifiedGPUMiner>();
        if (gpu_miner->initialize()) {
            auto devices = gpu_miner->get_available_devices();
            if (!devices.empty()) {
                // Use all available devices
                std::vector<int> device_ids;
                for (const auto& dev : devices) {
                    device_ids.push_back(dev.device_id);
                }
                gpu_miner->select_devices(device_ids);
            } else {
                std::cout << "No GPU devices found, disabling GPU mining" << std::endl;
                enable_gpu = false;
                gpu_miner.reset();
            }
        } else {
            std::cout << "GPU initialization failed, disabling GPU mining" << std::endl;
            enable_gpu = false;
            gpu_miner.reset();
        }
    }
    
    // Initialize stratum client (production)
    stratum_client = std::make_unique<StratumClient>(&g_job_manager, pool_host, pool_port, wallet_address, "worker1", opt_protocol);
    if(opt_protocol==StratumProtocol::CryptoNote) stratum_client->set_verbose(true);
    if(!stratum_client->start()){
        std::cerr << "[Stratum] Failed to connect to pool" << std::endl;
    }
    g_share_submitter = std::make_unique<ZionShareSubmitter>(stratum_client.get());
    g_share_submitter->start();
    
    // Start mining
    if (enable_cpu && !cpu_miner->start()) {
        std::cerr << "Failed to start CPU mining!" << std::endl;
    }
    
    if (enable_gpu && gpu_miner && !gpu_miner->start_mining()) {
        std::cerr << "Failed to start GPU mining!" << std::endl;
    }
    
    // Start stats thread
    std::thread stats_thread(print_mining_stats, std::ref(cpu_miner));
    
    std::cout << "\n🚀 ZION Miner 1.4.0 started successfully!" << std::endl;
    std::cout << "💎 Mining ZION with Cosmic Harmony algorithm" << std::endl;
    std::cout << "⛏️  Press Ctrl+C to stop mining\n" << std::endl;
    
    // Main mining loop
    TerminalRawMode trm; // enable raw input (Linux)
    while (!shutdown_requested.load()) {
        // keyboard polling
#ifdef __linux__
        char ch;
        while(read(STDIN_FILENO,&ch,1)==1){
            if(ch=='q' || ch=='Q'){ shutdown_requested.store(true); break; }
            else if(ch=='s' || ch=='S'){ g_show_stats.store(!g_show_stats.load()); std::cout << "\n[TGL] Stats " << (g_show_stats?"ON":"OFF") << std::endl; }
            else if(ch=='d' || ch=='D'){ g_show_detailed.store(!g_show_detailed.load()); std::cout << "\n[TGL] Detailed " << (g_show_detailed?"ON":"OFF") << std::endl; }
            else if(ch=='b' || ch=='B'){ g_brief_mode.store(!g_brief_mode.load()); std::cout << "\n[TGL] Brief mode " << (g_brief_mode?"ON":"OFF") << std::endl; }
            else if(ch=='r' || ch=='R'){ g_reset_requested.store(true); std::cout << "\n[RESET] Counters requested" << std::endl; }
            else if(ch=='c' || ch=='C'){ g_clear_screen.store(true); }
            else if(ch=='h' || ch=='H' || ch=='?'){ std::cout << "\nHelp:\n  q quit\n  s toggle stats\n  d toggle detailed\n  b toggle brief\n  g GPU on/off\n  o cycle algorithm (cosmic/blake3/keccak) [debug modes don't submit shares]\n  r reset counters\n  c clear screen\n  v verbose stratum\n  h/? help" << std::endl; }
            else if(ch=='g' || ch=='G'){ bool cur=g_gpu_active.load(); bool next=!cur; g_gpu_active.store(next); if(gpu_miner){ if(next){ gpu_miner->start_mining(); } else { gpu_miner->stop_mining(); } } std::cout << "\n[TGL] GPU "<<(next?"ENABLED":"DISABLED")<< std::endl; }
            else if(ch=='o' || ch=='O'){ int m=g_algo_mode.load(); m=(m+1)%3; g_algo_mode.store(m); std::cout << "\n[ALGO] Switched to "<<algo_mode_name(m)<< (m==0?" (submitting shares)":" (debug, no submit)") << std::endl; }
            else if(ch=='v' || ch=='V'){ if(stratum_client){ bool nv=!stratum_client->verbose(); stratum_client->set_verbose(nv); std::cout << "\n[TGL] Verbose Stratum "<<(nv?"ON":"OFF")<<std::endl; } }
        }
#endif
        if(g_reset_requested.exchange(false)){
            if(cpu_miner) cpu_miner->reset();
            if(g_share_submitter) { /* reset handled in submitter class call below */ }
            if(stratum_client) { stratum_client->reset_counters(); }
            if(g_share_submitter) g_share_submitter->reset();
            g_global_nonce_base.store(0);
            std::cout << "[RESET] Counters vynulovány" << std::endl;
        }
        if(g_job_manager.has_job()){
            auto job = g_job_manager.current_job();
            if(enable_gpu && gpu_miner && job.blob.size() >= job.nonce_offset + 4){
                uint8_t header[80]; memset(header,0,80); size_t csz = std::min<size_t>(80, job.blob.size()); memcpy(header, job.blob.data(), csz);
                uint64_t effective_target = job.target_difficulty? job.target_difficulty : 0xFFFFFFFFFFFFULL;
                if(g_forced_target_active.load() && g_forced_target64>0) effective_target = g_forced_target64; // only 64-bit path influences GPU
                gpu_miner->set_work(header, effective_target);
                auto gpu_results = gpu_miner->get_results();
                for(const auto& r : gpu_results){ if(r.found_share && stratum_client){
                        std::cout << "\n🎉 SHARE FOUND by GPU ("<< r.device.name <<") nonce=0x"<< std::hex << r.nonce << std::dec << std::endl;
                        const char* hexchars = "0123456789abcdef"; char smallhex[17]; for(int b=0;b<8;++b){ uint8_t v=r.hash[b]; smallhex[b*2]=hexchars[v>>4]; smallhex[b*2+1]=hexchars[v & 0xF]; } smallhex[16]='\0';
                        stratum_client->submit_share(job.job_id, r.nonce, smallhex, 0);
                    } }
            }
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Cleanup
    std::cout << "\nShutting down miners..." << std::endl;
    
    if (cpu_miner) {
        cpu_miner->stop();
    }
    
    if (gpu_miner) {
        gpu_miner->stop_mining();
    }
    
    if (g_share_submitter) { g_share_submitter->stop(); }
    if (stratum_client) { stratum_client->stop(); }
    
    if (stats_thread.joinable()) {
        stats_thread.join();
    }
    
    std::cout << "✅ ZION Miner shutdown complete. Thanks for mining ZION!" << std::endl;
    return 0;
}