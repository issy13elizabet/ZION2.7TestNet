#pragma once

#include "zion.h"
#include <string>
#include <vector>

namespace zion {

struct NodeConfig {
    // Blockchain/genesis
    uint64_t genesis_timestamp = 0;
    uint32_t genesis_difficulty = 1; // interpreted as leading zero bits (0-256)
    PublicKey genesis_coinbase_address{}; // 32 bytes
    std::string expected_genesis_hash; // optional 64-hex expected genesis hash

    // Optional network info (unused for now)
    std::string network_id;
    int chain_id = 0;
    uint16_t p2p_port = ZION_P2P_DEFAULT_PORT;
    uint16_t rpc_port = ZION_RPC_DEFAULT_PORT;

    // Seed nodes (host:port)
    std::vector<std::string> seed_nodes;

    // Pool settings
    bool pool_enable = false;
    uint16_t pool_port = 3333;
    bool pool_require_auth = false;
    std::string pool_password;

    // Mempool
    uint64_t mempool_min_fee = 0; // microzions per KB
};

// Načte minimální konfiguraci ze souboru (INI-like)
bool load_node_config(const std::string& path, NodeConfig& out, std::string& error);

// Načte JSON konfiguraci
bool load_json_config(const std::string& path, NodeConfig& out, std::string& error);

} // namespace zion
