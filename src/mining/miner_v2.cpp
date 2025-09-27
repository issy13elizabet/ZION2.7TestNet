#include "zion.h"
#include "block.h"
#include "transaction.h"
#include "randomx_wrapper.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>
#include <signal.h>
#include <iomanip>
#include <cstdlib>
#include <map>

namespace zion {

// Konfigurace mineru
struct MinerConfig {
    int thread_count = 4;
    bool use_full_mode = true;
    int stats_interval = 30;
    uint64_t max_iterations = 10000000;
    bool testnet = false;
    std::string daemon_host = "127.0.0.1";
    int daemon_port = 18081;
    std::string log_level = "info";
    bool dev_mode = false;
    int max_cpu_usage = 90;
    
    bool loadFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        std::string line;
        std::string current_section;
        
        while (std::getline(file, line)) {
            // Odstranit bílé znaky
            line.erase(0, line.find_first_not_of(" \t"));
            line.erase(line.find_last_not_of(" \t") + 1);
            
            // Přeskočit prázdné řádky a komentáře
            if (line.empty() || line[0] == '#') continue;
            
            // Zpracovat sekce
            if (line[0] == '[' && line.back() == ']') {
                current_section = line.substr(1, line.length() - 2);
                continue;
            }
            
            // Zpracovat páry klíč=hodnota
            size_t eq_pos = line.find('=');
            if (eq_pos != std::string::npos) {
                std::string key = line.substr(0, eq_pos);
                std::string value = line.substr(eq_pos + 1);
                
                // Odstranit bílé znaky
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);
                
                // Nastavit hodnoty podle sekce
                if (current_section == "Miner") {
                    if (key == "threads") thread_count = std::stoi(value);
                    else if (key == "mode") use_full_mode = (value == "full");
                    else if (key == "stats_interval") stats_interval = std::stoi(value);
                    else if (key == "max_iterations") max_iterations = std::stoull(value);
                    else if (key == "testnet") testnet = (value == "true");
                }
                else if (current_section == "Network") {
                    if (key == "daemon_host") daemon_host = value;
                    else if (key == "daemon_port") daemon_port = std::stoi(value);
                }
                else if (current_section == "Logging") {
                    if (key == "log_level") log_level = value;
                }
                else if (current_section == "Development") {
                    if (key == "dev_mode") dev_mode = (value == "true");
                }
                else if (current_section == "Safety") {
                    if (key == "max_cpu_usage") max_cpu_usage = std::stoi(value);
                }
            }
        }
        
        file.close();
        return true;
    }
};

class ZionMinerV2 {
public:
    ZionMinerV2(const MinerConfig& config)
        : config_(config), running_(false), blocks_mined_(0), 
          total_hashes_(0), hashrate_(0) {}
    
    bool initialize() {
        std::cout << "🚀 Inicializace ZION miner v2..." << std::endl;
        std::cout << "📋 Konfigurace:" << std::endl;
        std::cout << "  • Vlákna: " << config_.thread_count << std::endl;
        std::cout << "  • Režim: " << (config_.use_full_mode ? "Full" : "Light") << std::endl;
        std::cout << "  • Testnet: " << (config_.testnet ? "Ano" : "Ne") << std::endl;
        std::cout << "  • Dev mode: " << (config_.dev_mode ? "Ano" : "Ne") << std::endl;
        
        // Inicializace RandomX s vylepšeným error handlingem
        try {
            Hash seed;
            seed.fill(config_.testnet ? 0x42 : 0x01); // Různé seedy pro testnet a mainnet
            
            auto& randomx = RandomXWrapper::instance();
            
            // Použít správný režim podle konfigurace
            if (!randomx.initialize(seed, config_.use_full_mode)) {
                std::cerr << "❌ Chyba při inicializaci RandomX!" << std::endl;
                std::cerr << "   Zkuste použít light mode (mode = light v config)" << std::endl;
                return false;
            }
            
            std::cout << "✅ RandomX inicializován" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "❌ Výjimka při inicializaci: " << e.what() << std::endl;
            return false;
        }
        
        // Generování klíčového páru pro miner
        auto [private_key, public_key] = generate_keypair();
        miner_private_key_ = private_key;
        miner_public_key_ = public_key;
        
        std::cout << "💰 Miner adresa: 0x";
        for (size_t i = 0; i < 8 && i < public_key.size(); ++i) {
            std::cout << std::hex << std::setfill('0') << std::setw(2) 
                     << static_cast<int>(public_key[i]);
        }
        std::cout << "..." << std::dec << std::endl;
        
        // V dev módu snížit obtížnost
        if (config_.dev_mode) {
            base_difficulty_ = 100; // Velmi nízká obtížnost pro rychlé testování
            std::cout << "⚠️  Dev mode: Obtížnost snížena na " << base_difficulty_ << std::endl;
        }
        
        std::cout << "✅ Miner připraven k těžbě!" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        return true;
    }
    
    void start_mining() {
        std::cout << "⛏️  Spouštím těžbu..." << std::endl;
        running_ = true;
        start_time_ = std::chrono::steady_clock::now();
        
        // Vytvořit mining vlákna s ochranou proti chybám
        for (int i = 0; i < config_.thread_count; ++i) {
            mining_threads_.emplace_back([this, i]() {
                try {
                    this->mining_thread(i);
                }
                catch (const std::exception& e) {
                    std::cerr << "❌ Vlákno " << i << " selhalo: " << e.what() << std::endl;
                }
            });
        }
        
        // Statistiky vlákno
        stats_thread_ = std::thread([this]() {
            try {
                this->stats_thread();
            }
            catch (const std::exception& e) {
                std::cerr << "❌ Stats vlákno selhalo: " << e.what() << std::endl;
            }
        });
        
        // Počkat na ukončení vláken
        for (auto& thread : mining_threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        
        if (stats_thread_.joinable()) {
            stats_thread_.join();
        }
        
        print_final_stats();
    }
    
    void stop_mining() {
        std::cout << "\n🛑 Zastavuji těžbu..." << std::endl;
        running_ = false;
    }
    
private:
    MinerConfig config_;
    std::atomic<bool> running_;
    std::atomic<uint64_t> blocks_mined_;
    std::atomic<uint64_t> total_hashes_;
    std::atomic<uint64_t> hashrate_;
    uint64_t base_difficulty_ = 10000;
    
    PrivateKey miner_private_key_;
    PublicKey miner_public_key_;
    
    std::vector<std::thread> mining_threads_;
    std::thread stats_thread_;
    std::chrono::steady_clock::time_point start_time_;
    
    // Statistiky pro každé vlákno
    std::vector<std::atomic<uint64_t>> thread_hashes_;
    
    void mining_thread(int thread_id) {
        std::cout << "🧵 Vlákno " << thread_id << " spuštěno" << std::endl;
        
        // Inicializace lokálních proměnných
        uint64_t local_hashes = 0;
        uint64_t blocks_found = 0;
        auto last_update = std::chrono::steady_clock::now();
        
        // Náhodný offset pro nonce, aby vlákna netestovala stejné hodnoty
        uint64_t nonce_offset = thread_id * 1000000000ULL;
        
        while (running_) {
            // Vytvořit kandidátský blok
            Block candidate_block = create_candidate_block();
            
            // Nastavit počáteční nonce pro toto vlákno
            auto& header = const_cast<BlockHeader&>(candidate_block.getHeader());
            header.nonce = nonce_offset;
            
            // Získat aktuální obtížnost
            uint64_t target_difficulty = calculate_current_difficulty();
            
            // Těžit s limitem iterací
            bool found = false;
            for (uint64_t i = 0; i < config_.max_iterations && running_; ++i) {
                if (candidate_block.mine(target_difficulty)) {
                    found = true;
                    blocks_found++;
                    blocks_mined_++;
                    
                    std::cout << "\n💎 [Vlákno " << thread_id << "] Nalezen blok #" 
                             << blocks_mined_.load() << "!" << std::endl;
                    std::cout << "   Nonce: " << header.nonce << std::endl;
                    std::cout << "   Obtížnost: " << target_difficulty << std::endl;
                    
                    if (!config_.dev_mode) {
                        print_block_details(candidate_block);
                    }
                    
                    break;
                }
                
                local_hashes++;
                total_hashes_++;
                
                // Aktualizovat nonce pro další pokus
                header.nonce++;
                
                // Kontrola CPU throttlingu každých 1000 hashů
                if (i % 1000 == 0 && config_.max_cpu_usage < 100) {
                    // Krátká pauza pro snížení CPU využití
                    std::this_thread::sleep_for(
                        std::chrono::microseconds(100 * (100 - config_.max_cpu_usage))
                    );
                }
            }
            
            // Aktualizovat hashrate
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_update);
            if (elapsed.count() >= 1) {
                hashrate_ = local_hashes / elapsed.count();
                local_hashes = 0;
                last_update = now;
            }
            
            // Posunout nonce offset pro další blok
            nonce_offset += config_.max_iterations;
        }
        
        std::cout << "🏁 Vlákno " << thread_id << " ukončeno"
                 << " (nalezeno " << blocks_found << " bloků)" << std::endl;
    }
    
    void stats_thread() {
        while (running_) {
            std::this_thread::sleep_for(std::chrono::seconds(config_.stats_interval));
            
            if (!running_) break;
            
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_);
            
            std::cout << "\n📊 === STATISTIKY TĚŽBY ===" << std::endl;
            std::cout << "⏱️  Doba těžby: " << elapsed.count() << " sekund" << std::endl;
            std::cout << "💎 Nalezené bloky: " << blocks_mined_.load() << std::endl;
            std::cout << "⚡ Hashrate: " << format_hashrate(hashrate_.load() * config_.thread_count) << std::endl;
            std::cout << "🔢 Celkem hashů: " << format_number(total_hashes_.load()) << std::endl;
            
            if (blocks_mined_ > 0) {
                double avg_time = static_cast<double>(elapsed.count()) / blocks_mined_;
                std::cout << "⏰ Průměrný čas/blok: " << avg_time << " sekund" << std::endl;
            }
            
            std::cout << std::string(30, '=') << std::endl;
        }
    }
    
    void print_final_stats() {
        auto end_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time_);
        
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "📈 FINÁLNÍ STATISTIKY" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        std::cout << "⏱️  Celková doba těžby: " << elapsed.count() << " sekund" << std::endl;
        std::cout << "💎 Celkem nalezeno bloků: " << blocks_mined_.load() << std::endl;
        std::cout << "🔢 Celkem vypočteno hashů: " << format_number(total_hashes_.load()) << std::endl;
        
        if (blocks_mined_ > 0) {
            double avg_time = static_cast<double>(elapsed.count()) / blocks_mined_;
            std::cout << "⏰ Průměrný čas na blok: " << avg_time << " sekund" << std::endl;
            
            // Odhadovaná odměna
            uint64_t total_reward = blocks_mined_ * ZION_INITIAL_REWARD;
            std::cout << "💰 Odhadovaná odměna: " << (total_reward / 1e8) << " ZION" << std::endl;
        }
        
        std::cout << std::string(50, '=') << std::endl;
    }
    
    Block create_candidate_block() {
        // Vytvořit coinbase transakci
        uint64_t reward = calculate_block_reward();
        auto coinbase_tx = std::make_shared<Transaction>(
            Transaction::create_coinbase(miner_public_key_, reward)
        );
        
        // Použít správný předchozí hash
        Hash prev_hash;
        if (config_.testnet) {
            prev_hash.fill(0x00); // Genesis hash pro testnet
        } else {
            // V produkci by se načítal z blockchainu
            prev_hash.fill(0x01);
        }
        
        // Vytvořit blok
        Block block(blocks_mined_ + 1, prev_hash);
        block.addTransaction(coinbase_tx);
        
        return block;
    }
    
    uint64_t calculate_current_difficulty() {
        if (config_.dev_mode) {
            return base_difficulty_; // Fixní nízká obtížnost v dev mode
        }
        
        // Dynamická obtížnost podle počtu bloků
        uint64_t difficulty = base_difficulty_;
        
        // Zvýšit obtížnost každých 10 bloků
        if (blocks_mined_ > 0 && blocks_mined_ % 10 == 0) {
            difficulty = base_difficulty_ * (1 + blocks_mined_ / 10);
        }
        
        return difficulty;
    }
    
    uint64_t calculate_block_reward() {
        // Základní odměna s halvingem dle konsenzuální konstanty
        uint64_t halvings = blocks_mined_ / ZION_HALVENING_INTERVAL;
        uint64_t reward = ZION_INITIAL_REWARD;
        
        for (uint64_t i = 0; i < halvings && reward > 0; ++i) {
            reward /= 2;
        }
        
        return reward;
    }
    
    void print_block_details(const Block& block) {
        auto hash = block.calculateHash();
        std::cout << "📦 Detaily bloku:" << std::endl;
        std::cout << "   Hash: 0x";
        for (size_t i = 0; i < 16 && i < hash.size(); ++i) {
            std::cout << std::hex << std::setfill('0') << std::setw(2) 
                     << static_cast<int>(hash[i]);
        }
        std::cout << "..." << std::dec << std::endl;
        std::cout << "   Výška: " << block.getHeader().height << std::endl;
        std::cout << "   Timestamp: " << block.getHeader().timestamp << std::endl;
        std::cout << "   Transakcí: " << block.getTransactions().size() << std::endl;
    }
    
    std::string format_hashrate(uint64_t hps) {
        std::stringstream ss;
        if (hps >= 1000000000) {
            ss << (hps / 1000000000.0) << " GH/s";
        } else if (hps >= 1000000) {
            ss << (hps / 1000000.0) << " MH/s";
        } else if (hps >= 1000) {
            ss << (hps / 1000.0) << " KH/s";
        } else {
            ss << hps << " H/s";
        }
        return ss.str();
    }
    
    std::string format_number(uint64_t num) {
        std::stringstream ss;
        if (num >= 1000000000) {
            ss << (num / 1000000000.0) << "B";
        } else if (num >= 1000000) {
            ss << (num / 1000000.0) << "M";
        } else if (num >= 1000) {
            ss << (num / 1000.0) << "K";
        } else {
            ss << num;
        }
        return ss.str();
    }
};

} // namespace zion

// Globální instance mineru
zion::ZionMinerV2* g_miner = nullptr;

// Signal handler
void signal_handler(int signal) {
    std::cout << "\n⚠️  Přijat signál " << signal << std::endl;
    if (g_miner) {
        g_miner->stop_mining();
    }
}

int main(int argc, char* argv[]) {
    std::cout << std::string(50, '=') << std::endl;
    std::cout << "⛏️  ZION CRYPTOCURRENCY MINER V2 ⛏️" << std::endl;
    std::cout << "Version: "
              << static_cast<int>(zion::ZION_VERSION_MAJOR) << "."
              << static_cast<int>(zion::ZION_VERSION_MINOR) << "."
              << static_cast<int>(zion::ZION_VERSION_PATCH) << std::endl;
    std::cout << std::string(50, '=') << std::endl << std::endl;
    
    // Načíst konfiguraci
    zion::MinerConfig config;
    
    // Výchozí konfigurace nebo ze souboru
    std::string config_file = "config/miner.conf";
    
    // Zpracovat argumenty příkazové řádky
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--threads" && i + 1 < argc) {
            config.thread_count = std::atoi(argv[++i]);
            std::cout << "📝 Nastaveno vláken: " << config.thread_count << std::endl;
        }
        else if (arg == "--config" && i + 1 < argc) {
            config_file = argv[++i];
        }
        else if (arg == "--testnet") {
            config.testnet = true;
            std::cout << "🧪 Testnet mode aktivován" << std::endl;
        }
        else if (arg == "--dev") {
            config.dev_mode = true;
            std::cout << "🔧 Developer mode aktivován" << std::endl;
        }
        else if (arg == "--light") {
            config.use_full_mode = false;
            std::cout << "💡 Light mode aktivován" << std::endl;
        }
        else if (arg == "--help") {
            std::cout << "Použití: zion_miner [volby]" << std::endl;
            std::cout << "Volby:" << std::endl;
            std::cout << "  --threads N     Počet těžebních vláken (výchozí: 4)" << std::endl;
            std::cout << "  --config FILE   Cesta ke konfiguračnímu souboru" << std::endl;
            std::cout << "  --testnet       Použít testnet" << std::endl;
            std::cout << "  --dev           Developer mode (nízká obtížnost)" << std::endl;
            std::cout << "  --light         Použít light mode (nižší paměťové nároky)" << std::endl;
            std::cout << "  --help          Zobrazit tuto nápovědu" << std::endl;
            return 0;
        }
    }
    
    // Pokusit se načíst konfigurační soubor
    if (config.loadFromFile(config_file)) {
        std::cout << "✅ Konfigurace načtena ze souboru: " << config_file << std::endl;
    } else {
        std::cout << "ℹ️  Používám výchozí konfiguraci" << std::endl;
    }
    
    // Validace konfigurace
    if (config.thread_count <= 0 || config.thread_count > 256) {
        std::cerr << "❌ Neplatný počet vláken: " << config.thread_count << std::endl;
        std::cerr << "   Použijte hodnotu mezi 1 a 256" << std::endl;
        return 1;
    }
    
    // Nastavit signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    #ifdef SIGPIPE
    signal(SIGPIPE, SIG_IGN);
    #endif
    
    // Vytvořit a inicializovat miner
    zion::ZionMinerV2 miner(config);
    g_miner = &miner;
    
    if (!miner.initialize()) {
        std::cerr << "❌ Chyba při inicializaci mineru!" << std::endl;
        std::cerr << "   Zkontrolujte konfiguraci a systémové požadavky" << std::endl;
        return 1;
    }
    
    // Spustit těžbu
    std::cout << "\n🚀 Těžba začíná! Stiskněte Ctrl+C pro ukončení.\n" << std::endl;
    miner.start_mining();
    
    // Cleanup
    std::cout << "\n🧹 Úklid..." << std::endl;
    zion::RandomXWrapper::instance().cleanup();
    
    std::cout << "✅ Miner ukončen" << std::endl;
    return 0;
}
