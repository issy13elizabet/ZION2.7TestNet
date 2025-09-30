#include "zion.h"
#include "block.h"
#include "transaction.h"
#include "randomx_wrapper.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>
#include <signal.h>
#include <iomanip>
#include <cstdlib>
#include <sstream>
#include <cstring>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <mutex>
#include <condition_variable>
#include <cctype>

namespace zion {

class ZionMiner {
public:
    ZionMiner(int thread_count = std::thread::hardware_concurrency(), bool use_dataset = true)
        : thread_count_(thread_count), use_dataset_(use_dataset), running_(false), blocks_mined_(0), total_hashes_(0) {}
    
    bool initialize() {
        std::cout << "Inicializace ZION miner..." << std::endl;
        
        // Initialize RandomX
        Hash seed;
        seed.fill(0x42); // Same seed as daemon for now
        
        auto& randomx = RandomXWrapper::instance();
        // Miner může použít plný režim (dataset) nebo light mode podle nastavení
        if (!randomx.initialize(seed, use_dataset_)) {
            std::cerr << "Chyba při inicializaci RandomX!" << std::endl;
            return false;
        }
        
        // Generate miner keypair
        auto [private_key, public_key] = generate_keypair();
        miner_private_key_ = private_key;
        miner_public_key_ = public_key;
        
        std::cout << "Miner adresa: ";
        for (const auto& byte : public_key) {
            std::cout << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(byte);
        }
        std::cout << std::dec << std::endl;
        
        std::cout << "Použije " << thread_count_ << " vláken pro těžbu" << std::endl;
        return true;
    }
    
    void set_payout_public_key(const PublicKey& pk) {
        payout_public_key_ = pk;
        has_payout_address_ = true;
    }

    void start_mining() {
        std::cout << "Spouštím těžbu..." << std::endl;
        running_ = true;
        
        // Create mining threads
        for (int i = 0; i < thread_count_; ++i) {
            mining_threads_.emplace_back(&ZionMiner::mining_thread, this, i);
        }
        
        // Stats thread
        stats_thread_ = std::thread(&ZionMiner::stats_thread, this);
        
        // Wait for threads to finish
        for (auto& thread : mining_threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        
        if (stats_thread_.joinable()) {
            stats_thread_.join();
        }
        
        std::cout << "Těžba ukončena" << std::endl;
    }

    // Jednoduchý pool klient – single-threaded smyčka
    void start_pool_mining(const std::string& host, int port, const std::string& payout_address, const std::string& user = std::string(), const std::string& pass = std::string(), bool mobile = false) {
        (void)payout_address; // payout se propisuje už přes set_payout_public_key, pokud byl zadán
        std::cout << "Spouštím těžbu v režimu pool klienta proti " << host << ":" << port << std::endl;
        running_ = true;
        total_hashes_ = 0;
        blocks_mined_ = 0;
        mobile_mode_ = mobile;
        // Init per-thread stats
        thread_total_hashes_.reset(new std::atomic<uint64_t>[thread_count_]);
        thread_accepted_.reset(new std::atomic<uint64_t>[thread_count_]);
        thread_rejected_.reset(new std::atomic<uint64_t>[thread_count_]);
        for (int i=0;i<thread_count_;++i){ thread_total_hashes_[i].store(0); thread_accepted_[i].store(0); thread_rejected_[i].store(0);} 

        if (mobile) {
            // Mobilní optimalizace: light mode uživatel volí parametrem programu; zde snížíme chunk a vlákna případně mimo
        }

        // Worker identifikátor
        std::string worker_name = user;
        if (worker_name.empty()) {
            // fallback z miner_public_key
            auto tohex_byte = [](uint8_t b){ static const char* x="0123456789abcdef"; std::string s; s.push_back(x[b>>4]); s.push_back(x[b&0xF]); return s; };
            worker_name = "miner-";
            for (int i=0;i<4;i++) worker_name += tohex_byte(miner_public_key_[i]);
        }
        std::string worker_pass = pass;

        // Stav aktuální práce
        struct PoolJob { std::string job_id; Hash prev; uint32_t target_bits=1; uint32_t height=1; uint64_t version=0; std::string extranonce_hex; };
        PoolJob job;
        std::mutex job_mx;
        std::condition_variable job_cv;
        std::atomic<uint64_t> job_version{0};
        std::atomic<bool> have_job{false};

        // Thread pro načítání práce
        auto fetcher = [this, &host, port, &worker_name, &worker_pass, &job, &job_mx, &job_cv, &job_version, &have_job]() {
            bool logged=false;
            while (running_) {
                std::string job_json;
                if (!logged) {
                    if (pool_login(host, port, worker_name, worker_pass, job_json)) {
                        std::string jid; Hash prev{}; uint32_t bits=1, h=1;
                        std::string ex;
                        if (parse_job_json(job_json, jid, prev, bits, h, &ex)) {
                            std::lock_guard<std::mutex> lk(job_mx);
                            job.job_id = jid; job.prev = prev; job.target_bits = bits; job.height = h; job.extranonce_hex = ex; job.version++;
                            have_job = true; job_version.store(job.version); job_cv.notify_all();
                            logged = true;
                        }
                    }
                } else {
                    if (pool_get_job(host, port, job_json) || pool_login(host, port, worker_name, worker_pass, job_json)) {
                        std::string jid; Hash prev{}; uint32_t bits=1, h=1;
                        std::string ex;
                        if (parse_job_json(job_json, jid, prev, bits, h, &ex)) {
                            bool changed=false;
                            {
                                std::lock_guard<std::mutex> lk(job_mx);
                                if (job.job_id != jid || std::memcmp(prev.data(), job.prev.data(), job.prev.size()) != 0 || job.target_bits != bits || job.height != h || job.extranonce_hex != ex) {
                                    job.job_id = jid; job.prev = prev; job.target_bits = bits; job.height = h; job.extranonce_hex = ex; job.version++;
                                    changed = true;
                                }
                            }
                            if (changed) { job_version.store(job.version); job_cv.notify_all(); }
                        }
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        };

        // Worker vlákna – rozdělení nonce prostoru start=stride-offset
        auto worker = [this, &host, port, &job, &job_mx, &job_cv, &job_version](int tid, int stride) {
            const uint64_t CHUNK = mobile_mode_ ? 50'000ULL : 200'000ULL; // mobil menší dávky
            uint64_t last_seen = 0;
            while (running_) {
                // Počkat na dostupnou práci nebo změnu jobu
                {
                    std::unique_lock<std::mutex> lk(job_mx);
                    job_cv.wait(lk, [&]{ return !running_ ? true : (job.version != last_seen); });
                    if (!running_) break;
                    last_seen = job.version;
                }

                // Připravit blok podle aktuální práce
                uint64_t reward = calculate_block_reward();
                const PublicKey& recipient = has_payout_address_ ? payout_public_key_ : miner_public_key_;
                auto coinbase_tx = std::make_shared<Transaction>(Transaction::create_coinbase(recipient, reward));
                Block blk(job.height, job.prev);
                // Embed extranonce: append zero-amount output with recipient containing extranonce bytes
                if (!job.extranonce_hex.empty()) {
                    PublicKey tag{}; tag.fill(0);
                    // parse hex (up to 32 bytes)
                    auto hexval=[&](char c){ if(c>='0'&&c<='9')return c-'0'; if(c>='a'&&c<='f')return c-'a'+10; if(c>='A'&&c<='F')return c-'A'+10; return -1; };
                    size_t bytes = std::min<size_t>(job.extranonce_hex.size()/2, tag.size());
                    for (size_t i=0;i<bytes;i++){ int hi=hexval(job.extranonce_hex[2*i]); int lo=hexval(job.extranonce_hex[2*i+1]); if(hi<0||lo<0){ bytes=i; break; } tag[i]=(uint8_t)((hi<<4)|lo); }
                    coinbase_tx->add_output(TxOutput(0, tag));
                }
                blk.addTransaction(coinbase_tx);

                // Těžit v dávkách s rozestupem nonce
                while (running_ && job_version.load() == last_seen) {
                    uint64_t attempts = 0;
                    bool found = blk.mine((uint32_t)job.target_bits, &attempts, (uint32_t)tid, (uint32_t)stride, CHUNK);
                    total_hashes_ += attempts;
                    thread_total_hashes_[tid].fetch_add(attempts, std::memory_order_relaxed);
                    if (found) {
                        // Odeslat share
                        auto bytes = blk.serialize();
                        std::string hex = to_hex(bytes);
                        if (pool_submit_block(host, port, job.job_id, hex)) {
                            blocks_mined_++;
                            thread_accepted_[tid].fetch_add(1, std::memory_order_relaxed);
                            std::cout << "[POOL] Share odeslán a přijat (T" << tid << ")" << std::endl;
                        } else {
                            thread_rejected_[tid].fetch_add(1, std::memory_order_relaxed);
                            std::cerr << "[POOL] Share zamítnut (T" << tid << ")" << std::endl;
                        }
                        break; // po nalezení se pravděpodobně změní práce, počkáme na fetcher
                    }
                }
            }
        };

        // Stats thread
        stats_thread_ = std::thread(&ZionMiner::stats_thread, this);

        // Spustit fetcher
        std::thread fetch_thread(fetcher);

        // Spustit worker vlákna podle počtu threadů
        mining_threads_.clear();
        for (int i = 0; i < thread_count_; ++i) {
            mining_threads_.emplace_back(worker, i, thread_count_);
        }

        // Join workers
        for (auto& t : mining_threads_) if (t.joinable()) t.join();
        if (fetch_thread.joinable()) fetch_thread.join();
        if (stats_thread_.joinable()) stats_thread_.join();

        std::cout << "Pool těžba ukončena" << std::endl;
    }
    
    void stop_mining() {
        std::cout << "Zastavuji těžbu..." << std::endl;
        running_ = false;
    }
    
private:
    int thread_count_;
    bool use_dataset_;
    std::atomic<bool> running_;
    std::atomic<uint64_t> blocks_mined_;
    std::atomic<uint64_t> total_hashes_;

    // Per-thread stats (arrays to avoid vector moving atomics)
    std::unique_ptr<std::atomic<uint64_t>[]> thread_total_hashes_;
    std::unique_ptr<std::atomic<uint64_t>[]> thread_accepted_;
    std::unique_ptr<std::atomic<uint64_t>[]> thread_rejected_;

    bool mobile_mode_ = false;
    
    PrivateKey miner_private_key_;
    PublicKey miner_public_key_;
    
    // Optional payout address override
    bool has_payout_address_ = false;
    PublicKey payout_public_key_{};
    
    std::vector<std::thread> mining_threads_;
    std::thread stats_thread_;

    static std::string to_hex(const std::vector<uint8_t>& data) {
        static const char* x = "0123456789abcdef";
        std::string s; s.reserve(data.size()*2);
        for (auto b : data) { s.push_back(x[b>>4]); s.push_back(x[b&0xF]); }
        return s;
    }

    static bool hex_to_hash(const std::string& hx, Hash& out) {
        if (hx.size() < 64) return false;
        auto hexval=[&](char c){ if(c>='0'&&c<='9')return c-'0'; if(c>='a'&&c<='f')return c-'a'+10; if(c>='A'&&c<='F')return c-'A'+10; return -1; };
        for (size_t i=0;i<32;i++){ int hi=hexval(hx[2*i]); int lo=hexval(hx[2*i+1]); if(hi<0||lo<0) return false; out[i]=(uint8_t)((hi<<4)|lo); }
        return true;
    }

    // Minimal JSON helpers (duplicated from server side)
    static bool extract_json_string_field(const std::string& json, const std::string& key, std::string& out) {
        std::string pat = std::string("\"") + key + "\":\"";
        auto p = json.find(pat);
        if (p == std::string::npos) return false;
        p += pat.size();
        auto e = json.find('"', p);
        if (e == std::string::npos) return false;
        out = json.substr(p, e - p);
        return true;
    }
    static bool extract_json_number_field(const std::string& json, const std::string& key, uint32_t& out) {
        std::string pat = std::string("\"") + key + "\":";
        auto p = json.find(pat);
        if (p == std::string::npos) return false;
        p += pat.size();
        while (p < json.size() && isspace((unsigned char)json[p])) ++p;
        size_t q = p;
        while (q < json.size() && (isdigit((unsigned char)json[q]) || json[q]=='-')) ++q;
        if (q == p) return false;
        try { out = static_cast<uint32_t>(std::stoul(json.substr(p, q - p))); } catch (...) { return false; }
        return true;
    }
    static bool extract_json_object(const std::string& json, const std::string& key, std::string& out) {
        std::string pat = std::string("\"") + key + "\":";
        auto p = json.find(pat);
        if (p == std::string::npos) return false;
        p += pat.size();
        while (p < json.size() && json[p] != '{' && json[p] != 'n' && json[p] != '"') ++p;
        if (p >= json.size()) return false;
        if (json[p] != '{') { out = "null"; return true; }
        int depth = 0; size_t start = p; size_t i = p;
        for (; i < json.size(); ++i) {
            if (json[i] == '{') depth++;
            else if (json[i] == '}') { depth--; if (depth == 0) { ++i; break; } }
        }
        if (depth != 0) return false;
        out = json.substr(start, i - start);
        return true;
    }

    // Jednoduché parsování JSONu z poolu (bez knihoven)
    static bool parse_job_json(const std::string& json, std::string& job_id, Hash& prev_hash, uint32_t& target_bits, uint32_t& height, std::string* extranonce_hex=nullptr) {
        extract_json_string_field(json, "job_id", job_id);
        std::string prev_hex; extract_json_string_field(json, "prev_hash", prev_hex);
        if (!hex_to_hash(prev_hex, prev_hash)) return false;
        uint32_t bits=0, h=0; extract_json_number_field(json, "target_bits", bits); extract_json_number_field(json, "height", h);
        target_bits = bits ? bits : 1; height = h ? h : 1;
        if (extranonce_hex) {
            std::string ex;
            extract_json_string_field(json, "extranonce", ex);
            *extranonce_hex = ex;
        }
        return true;
    }

    // TCP helper
    static bool tcp_send_recv_line(const std::string& host, int port, const std::string& send_line, std::string& recv_line) {
        recv_line.clear();
        int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return false;
        sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_port = htons((uint16_t)port);
        if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) { ::close(fd); return false; }
        if (::connect(fd, (sockaddr*)&addr, sizeof(addr)) != 0) { ::close(fd); return false; }
        std::string line = send_line; if (line.empty() || line.back()!='\n') line.push_back('\n');
        if (::send(fd, line.data(), line.size(), 0) <= 0) { ::close(fd); return false; }
        // Read until newline or 4KB
        char ch; size_t cap=0; while (true) {
            ssize_t r = ::recv(fd, &ch, 1, 0); if (r <= 0) { ::close(fd); return false; }
            if (ch == '\n') break; recv_line.push_back(ch); if (++cap > 4096) { ::close(fd); return false; }
        }
        ::close(fd);
        return true;
    }

    // Stratum-like JSON-RPC client helpers
    bool pool_login(const std::string& host, int port, const std::string& worker, const std::string& pass, std::string& job_json) {
        std::ostringstream req; req << "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"login\",\"params\":{\"login\":\"" << worker << "\",\"pass\":\"" << pass << "\",\"agent\":\"zion-miner/1.0\"}}";
        std::string resp;
        if (!tcp_send_recv_line(host, port, req.str(), resp)) return false;
        // Extract result object
        std::string result;
        if (!extract_json_object(resp, "result", result)) return false;
        job_json = result;
        return true;
    }

    bool pool_get_job(const std::string& host, int port, std::string& job_json) {
        std::string resp;
        if (!tcp_send_recv_line(host, port, "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"getjob\",\"params\":{}}", resp)) return false;
        if (!extract_json_object(resp, "result", job_json)) return false;
        return true;
    }

    bool pool_submit_block(const std::string& host, int port, const std::string& job_id, const std::string& block_hex) {
        std::ostringstream req; req << "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"submit\",\"params\":{\"job_id\":\"" << job_id << "\",\"nonce\":\"00000000\",\"result\":\"" << block_hex << "\"}}";
        std::string resp;
        if (!tcp_send_recv_line(host, port, req.str(), resp)) return false;
        std::string result_str; if (!extract_json_string_field(resp, "result", result_str)) return false;
        return result_str == "OK";
    }
    
    void mining_thread(int thread_id) {
        std::cout << "Mining thread " << thread_id << " spuštěn" << std::endl;
        
        auto& randomx = RandomXWrapper::instance();
        (void)randomx;
        
        while (running_) {
            // Create a new block to mine
            Block candidate_block = create_candidate_block();
            
            // Mine the block
            uint64_t target_difficulty = calculate_current_difficulty();
            
            uint64_t attempts = 0;
            if (mine_block(candidate_block, target_difficulty, thread_id, attempts)) {
                blocks_mined_++;
                std::cout << "✓ Thread " << thread_id << " vytěžil blok! "
                         << "Nonce: " << candidate_block.getHeader().nonce 
                         << " Difficulty: " << target_difficulty << std::endl;
                
                // In a real implementation, we would broadcast this block to the network
                print_block_info(candidate_block);
            }
            total_hashes_ += attempts;
        }
        
        std::cout << "Mining thread " << thread_id << " ukončen" << std::endl;
    }
    
    static std::string format_hashrate(uint64_t hps) {
        std::ostringstream ss;
        if (hps >= 1000000000ULL) {
            ss << std::fixed << std::setprecision(2) << (hps / 1000000000.0) << " GH/s";
        } else if (hps >= 1000000ULL) {
            ss << std::fixed << std::setprecision(2) << (hps / 1000000.0) << " MH/s";
        } else if (hps >= 1000ULL) {
            ss << std::fixed << std::setprecision(2) << (hps / 1000.0) << " KH/s";
        } else {
            ss << hps << " H/s";
        }
        return ss.str();
    }

    void stats_thread() {
        uint64_t last_total = 0;
        std::vector<uint64_t> last_thread_totals;
        last_thread_totals.assign(thread_count_, 0);
        auto last_time = std::chrono::steady_clock::now();
        while (running_) {
            std::this_thread::sleep_for(std::chrono::seconds(10));
            auto now = std::chrono::steady_clock::now();
            auto secs = std::chrono::duration_cast<std::chrono::seconds>(now - last_time).count();
            if (secs <= 0) secs = 1;
            uint64_t cur_total = total_hashes_.load();
            uint64_t rate = (cur_total - last_total) / (uint64_t)secs;
            last_total = cur_total;
            if (!running_) break;
            std::cout << "\n📊 Souhrn: " << blocks_mined_.load() << " bloků, " << format_hashrate(rate) << std::endl;
            // per-thread
            for (int i = 0; i < thread_count_; ++i) {
                uint64_t tt = thread_total_hashes_[i].load(std::memory_order_relaxed);
                uint64_t tr = (tt - last_thread_totals[i]) / (uint64_t)secs;
                last_thread_totals[i] = tt;
                uint64_t acc = thread_accepted_[i].load(std::memory_order_relaxed);
                uint64_t rej = thread_rejected_[i].load(std::memory_order_relaxed);
                std::cout << "  T" << i << ": " << format_hashrate(tr) << ", A/R=" << acc << "/" << rej << std::endl;
            }
            last_time = now;
        }
    }
    
    Block create_candidate_block() {
        // Create coinbase transaction
        uint64_t reward = calculate_block_reward();
        const PublicKey& recipient = has_payout_address_ ? payout_public_key_ : miner_public_key_;
        auto coinbase_tx = std::make_shared<Transaction>(Transaction::create_coinbase(recipient, reward));
        
        // Create previous block hash (for now, use a dummy hash)
        Hash prev_hash;
        prev_hash.fill(0x00);
        
        // Use proper constructor - height 1 for now
        Block block(1, prev_hash);
        
        // Add coinbase transaction
        block.addTransaction(coinbase_tx);
        
        // In a real implementation, we would also include pending transactions from mempool
        
        return block;
    }
    
    bool mine_block(Block& block, uint64_t target_difficulty, int /*thread_id*/, uint64_t& hashes_done) {
        // Use Block::mine with attempts counter for accurate hashrate
        bool found = block.mine((uint32_t)target_difficulty, &hashes_done);
        return found;
    }
    
    uint64_t calculate_current_difficulty() {
        // Simplified difficulty calculation
        // In a real implementation, this would be based on network hashrate and target block time
        return 10000; // Fixed difficulty for now
    }
    
    uint64_t calculate_block_reward() {
        // Simplified reward calculation
        // In a real implementation, this would consider block height and halvening
        return ZION_INITIAL_REWARD;
    }
    
    void print_block_info(const Block& block) {
        auto hash = block.calculateHash();
        std::cout << "🔗 Nový blok:" << std::endl;
        std::cout << "   Hash: ";
        for (const auto& byte : hash) {
            std::cout << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(byte);
        }
        std::cout << std::dec << std::endl;
        std::cout << "   Timestamp: " << block.getHeader().timestamp << std::endl;
        std::cout << "   Transakce: " << block.getTransactions().size() << std::endl;
    }
};

} // namespace zion

// Global miner instance
zion::ZionMiner* g_miner = nullptr;

void signal_handler(int signal) {
    std::cout << "Přijat signál " << signal << ", ukončuji těžbu..." << std::endl;
    if (g_miner) {
        g_miner->stop_mining();
    }
}

int main(int argc, char* argv[]) {
    // Heuristika pro mobilní zařízení (např. iOS/Android přes termux) – nyní jen přepínač --mobile
    std::cout << "ZION Cryptocurrency Miner v"
              << static_cast<int>(zion::ZION_VERSION_MAJOR) << "."
              << static_cast<int>(zion::ZION_VERSION_MINOR) << "."
              << static_cast<int>(zion::ZION_VERSION_PATCH) << std::endl;
    
    // Parse command line arguments
    int thread_count = std::thread::hardware_concurrency();
    bool light_mode = false; // pokud true, nepoužívat dataset (rychlejší start, menší RAM)
    std::string pool_host; int pool_port = 0; std::string payout_address; std::string user; std::string pass; bool mobile=false;
    
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--threads" && i + 1 < argc) {
            thread_count = std::atoi(argv[i + 1]);
            ++i;
        } else if (std::string(argv[i]) == "--light") {
            light_mode = true;
        } else if (std::string(argv[i]) == "--pool" && i + 1 < argc) {
            std::string hp = argv[++i];
            auto pos = hp.rfind(':');
            if (pos != std::string::npos) { pool_host = hp.substr(0,pos); pool_port = std::atoi(hp.substr(pos+1).c_str()); }
        } else if (std::string(argv[i]) == "--address" && i + 1 < argc) {
            payout_address = argv[++i];
        } else if (std::string(argv[i]) == "--user" && i + 1 < argc) {
            user = argv[++i];
        } else if (std::string(argv[i]) == "--pass" && i + 1 < argc) {
            pass = argv[++i];
        } else if (std::string(argv[i]) == "--mobile") {
            mobile = true;
            if (thread_count > 2) thread_count = std::max(1, thread_count/2); // snížit zátěž
        } else if (std::string(argv[i]) == "--help") {
            std::cout << "Použití: zion_miner [--threads N] [--light] [--pool host:port] [--address ADDR] [--help]" << std::endl;
            std::cout << "  --threads N    Počet těžebních vláken (výchozí: " << thread_count << ")" << std::endl;
            std::cout << "  --light        Light mode (bez RandomX datasetu; rychlejší start, nižší výkon)" << std::endl;
            std::cout << "  --pool h:p     Připojit se k poolu" << std::endl;
            std::cout << "  --address A    Adresa pro odměny (při solo miningu ignorováno)" << std::endl;
            std::cout << "  --help         Zobrazit tuto nápovědu" << std::endl;
            return 0;
        }
    }
    
    if (thread_count <= 0 || thread_count > 256) {
        std::cerr << "Neplatný počet vláken: " << thread_count << std::endl;
        return 1;
    }
    
    // Setup signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Create and initialize miner
    zion::ZionMiner miner(thread_count, !light_mode);
    g_miner = &miner;

    // If payout address provided, parse as 64-hex public key and set
    auto hexval = [](char c)->int { if(c>='0'&&c<='9')return c-'0'; if(c>='a'&&c<='f')return c-'a'+10; if(c>='A'&&c<='F')return c-'A'+10; return -1; };
    if (!payout_address.empty()) {
        if (payout_address.size() == 64) {
            zion::PublicKey pk{}; bool ok=true; for (size_t i=0;i<32;i++){ int hi=hexval(payout_address[2*i]); int lo=hexval(payout_address[2*i+1]); if(hi<0||lo<0){ ok=false; break; } pk[i]=(uint8_t)((hi<<4)|lo);} 
            if (ok) {
                miner.set_payout_public_key(pk);
                std::cout << "Payout address nastaven (public key hex)" << std::endl;
            }
        }
    }
    
    if (!miner.initialize()) {
        std::cerr << "Chyba při inicializaci miner!" << std::endl;
        return 1;
    }
    
    // Start mining
    if (!pool_host.empty() && pool_port > 0) {
        std::cout << "Režim pool klient: " << pool_host << ":" << pool_port << std::endl;
        miner.start_pool_mining(pool_host, pool_port, payout_address, user, pass, mobile);
    } else {
        miner.start_mining();
    }
    
    // Cleanup
    zion::RandomXWrapper::instance().cleanup();
    
    return 0;
}
