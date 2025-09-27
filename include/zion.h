#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <array>

namespace zion {

// ZION network constants
constexpr uint64_t ZION_MAX_SUPPLY = 144000000000ULL * 1000000ULL; // 144,000,000,000 ZION with 6 decimals
constexpr uint64_t ZION_INITIAL_REWARD = 333ULL * 1000000ULL;      // 333 ZION
constexpr uint32_t ZION_TARGET_BLOCK_TIME = 120;               // 2 minutes in seconds
constexpr uint32_t ZION_DIFFICULTY_WINDOW = 720;               // 720 blocks = ~24 hours
constexpr uint32_t ZION_HALVENING_INTERVAL = 210000;           // Halvening every 210k blocks
constexpr uint16_t ZION_P2P_DEFAULT_PORT = 18080;
constexpr uint16_t ZION_RPC_DEFAULT_PORT = 18081;

// Version info
constexpr uint8_t ZION_VERSION_MAJOR = 1;
constexpr uint8_t ZION_VERSION_MINOR = 0;
constexpr uint8_t ZION_VERSION_PATCH = 0;

// Block and transaction limits
constexpr size_t ZION_MAX_BLOCK_SIZE = 1024 * 1024; // 1MB
constexpr size_t ZION_MAX_TX_SIZE = 100 * 1024;     // 100KB

// Hash definitions
using Hash = std::array<uint8_t, 32>;
using PublicKey = std::array<uint8_t, 32>;
using PrivateKey = std::array<uint8_t, 32>;
using Signature = std::array<uint8_t, 64>;

// Forward declarations
class Block;
class Transaction;
class Blockchain;
class NetworkManager;
class Miner;
class Wallet;

} // namespace zion
