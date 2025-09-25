#include <iostream>
#include <csignal>
#include <thread>
#include <atomic>
#include <chrono>
#include <iomanip>
#include "blockchain.h"
#include "block.h"
#include "transaction.h"
#include "randomx_wrapper.h"
#include "zion.h"
#include "config.h"
#include "network/p2p.h"
#include "network/pool.h"
#include <sstream>

using namespace zion;

std::atomic<bool> running(true);

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        std::cout << "\nShutting down ZION daemon..." << std::endl;
        running = false;
    }
}

void print_banner() {
    std::cout << R"(
╔══════════════════════════════════════════════╗
║           ZION CRYPTOCURRENCY v1.0.0         ║
║          Proof-of-Work with RandomX          ║
╚══════════════════════════════════════════════╝
)" << std::endl;
}

void print_info() {
    auto format_amount = [](uint64_t micro) {
        std::ostringstream oss;
        uint64_t whole = micro / 1000000ULL;
        uint64_t frac = micro % 1000000ULL;
        oss << whole << "." << std::setw(6) << std::setfill('0') << frac << " ZION";
        return oss.str();
    };

    std::cout << "Network Parameters:" << std::endl;
    std::cout << "  - Max Supply: " << (ZION_MAX_SUPPLY / 1000000ULL) << " ZION" << std::endl;
    std::cout << "  - Block Time: " << ZION_TARGET_BLOCK_TIME/60 << " minutes" << std::endl;
    std::cout << "  - Initial Reward: " << format_amount(ZION_INITIAL_REWARD) << std::endl;
    std::cout << "  - Halvening: Every " << ZION_HALVENING_INTERVAL << " blocks" << std::endl;
    std::cout << "  - P2P Port: " << ZION_P2P_DEFAULT_PORT << std::endl;
    std::cout << "  - RPC Port: " << ZION_RPC_DEFAULT_PORT << std::endl;
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    // Set up signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    
    print_banner();
    print_info();

    // Parse CLI args
    std::string config_path;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--config=", 0) == 0) {
            config_path = arg.substr(9);
        } else if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        }
    }

    std::cout << "Initializing ZION daemon..." << std::endl;
    std::cout << "[DEBUG] Starting daemon debug..." << std::endl;

    // Initialize RandomX
    auto& randomx = RandomXWrapper::instance();
    Hash init_key;
    init_key.fill(0);

    std::cout << "Initializing RandomX (this may take a moment)..." << std::endl;
    if (!randomx.initialize(init_key, false)) {
        std::cerr << "Failed to initialize RandomX!" << std::endl;
        return 1;
    }

    std::cout << "RandomX initialized successfully." << std::endl;

    // Load config (optional)
    NodeConfig cfg;
    // Pro Docker - defaultně povolit pool
    cfg.pool_enable = true;
    cfg.pool_port = 3333;
    
    if (!config_path.empty()) {
        std::string err;
        if (load_node_config(config_path, cfg, err)) {
            std::cout << "Loaded config: " << config_path << std::endl;
        } else {
            std::cerr << "Warning: could not load config: " << err << std::endl;
        }
    }
    
    std::cout << "[DEBUG] Pool enabled: " << (cfg.pool_enable ? "true" : "false") << std::endl;
    std::cout << "[DEBUG] Pool port: " << cfg.pool_port << std::endl;

    // Create blockchain instance
    Blockchain blockchain;
    if (cfg.genesis_timestamp != 0 || cfg.genesis_difficulty != 0) {
        std::cout << "Initializing blockchain with config genesis..." << std::endl;
        blockchain.initialize_with_genesis(cfg.genesis_timestamp,
                                           cfg.genesis_difficulty,
                                           cfg.genesis_coinbase_address);
    } else {
        blockchain.initialize();
    }

    std::cout << "Blockchain initialized with genesis block." << std::endl;
    std::cout << "Current height: " << blockchain.getHeight() << std::endl;
    std::cout << "Current difficulty: " << blockchain.getDifficulty() << std::endl;

    // Apply mempool fee rule
    blockchain.setMempoolMinFee(cfg.mempool_min_fee);

    // Validate expected genesis hash if provided
    if (!cfg.expected_genesis_hash.empty()) {
        auto gblk = blockchain.getBlock(0);
        if (gblk) {
            auto gh = gblk->calculateHash();
            static const char* x = "0123456789abcdef";
            std::string ghx; ghx.reserve(64);
            for (auto b : gh) { ghx.push_back(x[b>>4]); ghx.push_back(x[b&0xF]); }
            if (ghx != cfg.expected_genesis_hash) {
                std::cerr << "ERROR: Genesis hash mismatch! expected=" << cfg.expected_genesis_hash
                          << " actual=" << ghx << std::endl;
                return 1;
            } else {
                std::cout << "Genesis hash verified: " << ghx << std::endl;
            }
        }
    }

    // Start minimal P2P node
    P2PConfig p2pcfg;
    p2pcfg.bind_port = cfg.p2p_port;
    p2pcfg.seed_nodes = cfg.seed_nodes;
    p2pcfg.network_id = cfg.network_id;
    p2pcfg.chain_id = cfg.chain_id;
    P2PNode p2p(p2pcfg);
    p2p.set_logger([](const std::string& s){ std::cout << "[P2P] " << s << std::endl; });

    // Callbacky pro synchronizaci s blockchainem
    P2PNode::Callbacks cbs;
    cbs.get_height = [&blockchain]() -> uint32_t {
        return blockchain.getHeight();
    };
    cbs.get_headers = [&blockchain](uint32_t start_height, uint32_t max_count, std::vector<std::string>& headers_hex) -> bool {
        headers_hex.clear();
        uint32_t best = blockchain.getHeight();
        uint32_t from = start_height + 1;
        uint32_t to = std::min<uint32_t>(best, from + max_count - 1);
        for (uint32_t h = from; h <= to; ++h) {
            auto blk = blockchain.getBlock(h);
            if (!blk) break;
            // Serializuj pouze header a pošli jako hex
            auto hdr = blk->getHeader().serialize();
            static const char* hex = "0123456789abcdef";
            std::string hx; hx.reserve(hdr.size()*2);
            for (auto b : hdr) { hx.push_back(hex[b>>4]); hx.push_back(hex[b&0xF]); }
            headers_hex.push_back(hx);
        }
        return true;
    };
    cbs.get_block = [&blockchain](const std::string& hash_hex, std::string& block_hex) -> bool {
        // Jednoduchý převod hex->Hash (prvních 32B)
        if (hash_hex.size() < 64) return false;
        Hash h{}; auto hexval=[&](char c){ if(c>='0'&&c<='9')return c-'0'; if(c>='a'&&c<='f')return c-'a'+10; if(c>='A'&&c<='F')return c-'A'+10; return 0; };
        for (size_t i=0;i<32;i++){ h[i]= (uint8_t)((hexval(hash_hex[2*i])<<4)|hexval(hash_hex[2*i+1])); }
        auto blk = blockchain.getBlock(h);
        if (!blk) return false;
        auto data = blk->serialize();
        static const char* hex = "0123456789abcdef";
        block_hex.clear(); block_hex.reserve(data.size()*2);
        for (auto b : data) { block_hex.push_back(hex[b>>4]); block_hex.push_back(hex[b&0xF]); }
        return true;
    };
    cbs.submit_block = [&blockchain, &p2p](const std::string& block_hex, std::string& hash_hex) -> bool {
        // hex->bytes
        if (block_hex.size() % 2 != 0) return false;
        std::vector<uint8_t> data(block_hex.size()/2);
        auto hexval=[&](char c){ if(c>='0'&&c<='9')return c-'0'; if(c>='a'&&c<='f')return c-'a'+10; if(c>='A'&&c<='F')return c-'A'+10; return 0; };
        for(size_t i=0;i<data.size();++i){ data[i]=(uint8_t)((hexval(block_hex[2*i])<<4)|hexval(block_hex[2*i+1])); }
        auto up = Block::deserialize(data);
        if (!up) return false;
        std::shared_ptr<Block> blk(std::move(up));
        // Pokus o přidání do řetězce
        if (!blockchain.addBlock(blk)) return false;
        // Vypočti hash a vrať jako hex
        auto h = blk->calculateHash();
        static const char* hex = "0123456789abcdef";
        hash_hex.clear(); hash_hex.reserve(64);
        for (auto b : h) { hash_hex.push_back(hex[b>>4]); hash_hex.push_back(hex[b&0xF]); }
        return true;
    };
    // TX callbacks – implementace přes mempool
    cbs.get_tx = [&blockchain](const std::string& hash_hex, std::string& tx_hex) -> bool {
        if (hash_hex.size() < 64) return false;
        auto hexval=[&](char c){ if(c>='0'&&c<='9')return c-'0'; if(c>='a'&&c<='f')return c-'a'+10; if(c>='A'&&c<='F')return c-'A'+10; return 0; };
        Hash h{}; for(size_t i=0;i<32;i++){ h[i]=(uint8_t)((hexval(hash_hex[2*i])<<4)|hexval(hash_hex[2*i+1])); }
        auto tx = blockchain.getMempoolTransaction(h);
        if (!tx) return false;
        auto data = tx->serialize();
        static const char* hex = "0123456789abcdef";
        tx_hex.clear(); tx_hex.reserve(data.size()*2);
        for (auto b : data) { tx_hex.push_back(hex[b>>4]); tx_hex.push_back(hex[b&0xF]); }
        return true;
    };
    cbs.submit_tx = [&blockchain](const std::string& tx_hex, std::string& hash_hex) -> bool {
        if (tx_hex.size() % 2 != 0) return false;
        std::vector<uint8_t> data(tx_hex.size()/2);
        auto hexval=[&](char c){ if(c>='0'&&c<='9')return c-'0'; if(c>='a'&&c<='f')return c-'a'+10; if(c>='A'&&c<='F')return c-'A'+10; return 0; };
        for(size_t i=0;i<data.size();++i){ data[i]=(uint8_t)((hexval(tx_hex[2*i])<<4)|hexval(tx_hex[2*i+1])); }
        auto tx = Transaction::from_bytes(data);
        if (!tx) return false;
        if (!blockchain.addTransaction(tx)) return false;
        auto h = tx->get_hash();
        static const char* hex = "0123456789abcdef";
        hash_hex.clear(); hash_hex.reserve(64);
        for (auto b : h) { hash_hex.push_back(hex[b>>4]); hash_hex.push_back(hex[b&0xF]); }
        return true;
    };

    p2p.set_callbacks(cbs);

    if (!p2p.start()) {
        std::cerr << "Failed to start P2P node" << std::endl;
    }

    // Start built-in pool server if enabled
    std::unique_ptr<PoolServer> pool;
    if (cfg.pool_enable) {
        std::cout << "[POOL] Starting pool server on port " << cfg.pool_port << std::endl;
        PoolConfig pcfg; pcfg.port = cfg.pool_port;
        PoolServer::Callbacks pcb;
        // jednoduchý job state
        struct JobState { Hash prev{}; uint32_t bits=1; uint32_t height=1; std::string job_id="0"; };
        static JobState js;
        auto tohex=[&](const Hash& h){ static const char* x="0123456789abcdef"; std::string s; s.reserve(64); for(auto b:h){ s.push_back(x[b>>4]); s.push_back(x[b&0xF]); } return s; };
        auto refresh_job = [&](){
            auto latest = blockchain.getLatestBlock();
            if (latest) js.prev = latest->calculateHash(); else js.prev.fill(0);
            js.bits = blockchain.getDifficulty();
            if (js.bits == 0) js.bits = 1;
            js.height = blockchain.getHeight() + 1;
            js.job_id = std::to_string(js.height) + "-" + tohex(js.prev).substr(0,8);
        };
        refresh_job();
        pcb.login = [&](const std::string& worker, const std::string& pass) -> std::string {
            // jednoduchá autentizace (pokud vyžadována v configu): pass musí odpovídat cfg.pool_password
            if (cfg.pool_require_auth && pass != cfg.pool_password) {
                return std::string();
            }
            refresh_job();
            // vygeneruj extranonce8 per worker (jednoduchá deterministika z worker jména)
            Hash seed{}; for (size_t i=0;i<worker.size() && i<32;i++){ seed[i] = (uint8_t)worker[i]; }
            static const char* xh = "0123456789abcdef";
            std::string extranonce; extranonce.reserve(16);
            for (int i=0;i<8;i++){ uint8_t b = seed[i]^0x5a; extranonce.push_back(xh[b>>4]); extranonce.push_back(xh[b&0xF]); }
            std::ostringstream oss;
            oss << "{\"job_id\":\"" << js.job_id << "\",";
            oss << "\"prev_hash\":\"" << tohex(js.prev) << "\",";
            oss << "\"target_bits\":" << js.bits << ",";
            oss << "\"height\":" << js.height << ",";
            oss << "\"extranonce\":\"" << extranonce << "\"}";
            return oss.str();
        };
        pcb.get_job = [&]() -> std::string {
            refresh_job();
            std::ostringstream oss;
            oss << "{\"job_id\":\"" << js.job_id << "\",";
            oss << "\"prev_hash\":\"" << tohex(js.prev) << "\",";
            oss << "\"target_bits\":" << js.bits << ",";
            oss << "\"height\":" << js.height << ",";
            oss << "\"extranonce\":\"0000000000000000\"}";
            return oss.str();
        };
        pcb.submit_share = [&blockchain, &p2p](const std::string& job_id, const std::string& nonce_hex, const std::string& result_hex, const std::string& worker) -> bool {
            (void)job_id; (void)nonce_hex; (void)worker;
            // result_hex je celé block_hex
            if (result_hex.size() % 2 != 0) return false;
            std::vector<uint8_t> data(result_hex.size()/2);
            auto hexval=[&](char c){ if(c>='0'&&c<='9')return c-'0'; if(c>='a'&&c<='f')return c-'a'+10; if(c>='A'&&c<='F')return c-'A'+10; return 0; };
            for(size_t i=0;i<data.size();++i){ data[i]=(uint8_t)((hexval(result_hex[2*i])<<4)|hexval(result_hex[2*i+1])); }
            auto up = Block::deserialize(data);
            if (!up) return false;
            std::shared_ptr<Block> blk(std::move(up));
            if (!blockchain.addBlock(blk)) return false;
            auto h = blk->calculateHash(); static const char* x="0123456789abcdef"; std::string hash_hex; hash_hex.reserve(64); for(auto b:h){ hash_hex.push_back(x[b>>4]); hash_hex.push_back(x[b&0xF]); }
            p2p.broadcast_inv("BLOCK", hash_hex, -1);
            return true;
        };
        pool = std::make_unique<PoolServer>(pcfg, pcb);
        if (!pool->start()) {
            std::cerr << "[POOL] Failed to start pool server" << std::endl;
        }
    }
    
    std::cout << "\nZION daemon is running. Press Ctrl+C to stop." << std::endl;
    std::cout << "Waiting for connections on port " << p2pcfg.bind_port << "..." << std::endl;
    
    // Main daemon loop
    while (running) {
        // In a real implementation, this would:
        // - Accept P2P connections
        // - Sync blockchain with peers
        // - Process incoming transactions
        // - Broadcast blocks and transactions
        
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    
    std::cout << "Cleaning up..." << std::endl;
    randomx.cleanup();
    
    std::cout << "ZION daemon stopped." << std::endl;
    return 0;
}
