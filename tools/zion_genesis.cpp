#include "config.h"
#include "blockchain.h"
#include "block.h"
#include <iostream>
#include <iomanip>

using namespace zion;

static std::string to_hex(const Hash& h) {
    static const char* x = "0123456789abcdef";
    std::string s; s.reserve(64);
    for (auto b : h) { s.push_back(x[b>>4]); s.push_back(x[b&0xF]); }
    return s;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: zion_genesis <config_path> [--max-bits N]" << std::endl;
        return 1;
    }
    std::string cfg_path = argv[1];

    uint32_t max_bits = 8; // default quick generation
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--max-bits" && i + 1 < argc) {
            max_bits = static_cast<uint32_t>(std::stoul(argv[++i]));
        }
    }

    NodeConfig cfg; std::string err;
    if (!load_node_config(cfg_path, cfg, err)) {
        std::cerr << "Failed to load config: " << err << std::endl;
        return 1;
    }

    uint32_t bits = std::min(cfg.genesis_difficulty, max_bits);
    if (bits < cfg.genesis_difficulty) {
        std::cout << "[INFO] Limiting genesis difficulty bits from " << cfg.genesis_difficulty
                  << " to " << bits << " for quick generation." << std::endl;
    }

    Blockchain bc;
    bc.initialize_with_genesis(cfg.genesis_timestamp, bits, cfg.genesis_coinbase_address);

    auto g = bc.getBlock(0);
    if (!g) { std::cerr << "Failed to create genesis block" << std::endl; return 1; }

    auto gh = g->calculateHash();
    auto hdr = g->getHeader();

    std::cout << "ZION Genesis Info" << std::endl;
    std::cout << "==================" << std::endl;
    std::cout << "Config: " << cfg_path << std::endl;
    std::cout << "Height: " << hdr.height << std::endl;
    std::cout << "Timestamp: " << hdr.timestamp << std::endl;
    std::cout << "Difficulty bits: " << hdr.difficulty << std::endl;
    std::cout << "Genesis Hash: " << to_hex(gh) << std::endl;
    std::cout << std::endl;
    std::cout << "Add this to [blockchain] in your config:" << std::endl;
    std::cout << "genesis_hash = \"" << to_hex(gh) << "\"" << std::endl;

    return 0;
}
