#pragma once

#ifndef ZION_H
#define ZION_H

#include <string>
#include <vector>
#include <cstdint>
#include <memory>

// ZION Core definitions and constants
namespace zion {

// Version information
constexpr const char* ZION_VERSION = "2.6.0";
constexpr const char* ZION_CODENAME = "Cosmic Harmony";
constexpr uint32_t ZION_VERSION_MAJOR = 2;
constexpr uint32_t ZION_VERSION_MINOR = 6;
constexpr uint32_t ZION_VERSION_PATCH = 0;

// Network constants
constexpr uint16_t ZION_DEFAULT_PORT = 3333;
constexpr uint16_t ZION_RPC_PORT = 8888;
constexpr const char* ZION_ALGORITHM = "zion-cosmic-harmony";

// Mining constants
constexpr uint32_t ZION_BLOCK_TIME = 120; // seconds
constexpr uint32_t ZION_DIFFICULTY_ADJUSTMENT = 720; // blocks
constexpr uint64_t ZION_INITIAL_DIFFICULTY = 1000000;

// Cosmic Harmony Algorithm constants
namespace cosmic_harmony {
    constexpr double GOLDEN_RATIO = 1.618033988749895;
    constexpr double COSMIC_FREQUENCIES[] = {432.0, 528.0, 741.0, 852.0, 963.0};
    constexpr uint32_t FIBONACCI_SEQUENCE[] = {1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377};
    constexpr size_t MATRIX_SIZE = 2048 * 1024; // 2MB
    constexpr uint32_t QUANTUM_LAYERS = 3;
}

// Pool configuration
struct ZionPoolConfig {
    std::string host;
    uint16_t port;
    std::string username;
    std::string password;
    std::string algorithm;
    bool use_tls;
    
    ZionPoolConfig() : port(ZION_DEFAULT_PORT), algorithm(ZION_ALGORITHM), use_tls(false) {}
};

// Utility functions
std::string get_zion_version_string();
std::string get_zion_build_info();
bool is_zion_compatible_version(const std::string& version);

// Hash utilities
std::string bytes_to_hex(const std::vector<uint8_t>& bytes);
std::vector<uint8_t> hex_to_bytes(const std::string& hex);
uint64_t difficulty_from_hash(const std::vector<uint8_t>& hash);

// Cosmic utilities
double calculate_cosmic_enhancement(uint64_t nonce, uint32_t frequency_index);
std::vector<uint8_t> apply_golden_ratio_transform(const std::vector<uint8_t>& data);
std::vector<uint8_t> fibonacci_spiral_hash(const std::vector<uint8_t>& input);

} // namespace zion

#endif // ZION_H