#ifndef ZION_COSMIC_HARMONY_CORE_H
#define ZION_COSMIC_HARMONY_CORE_H

#include <cstdint>
#include <cstring>
#include <vector>

namespace zion {
namespace core {

// Platform-agnostic ZION Cosmic Harmony algorithm
class ZionHasher {
public:
    // Hash result structure
    struct HashResult {
        uint8_t hash[32];           // Final 256-bit hash
        uint32_t nonce;             // Nonce that produced this hash
        uint64_t difficulty_met;    // Difficulty value achieved
        bool is_valid_share;        // Whether this meets target difficulty
    };
    
    // Algorithm configuration for different platforms
    enum class AlgorithmMode {
        FULL_POWER,    // Linux/Windows/macOS - full algorithm
        MOBILE_LITE,   // Mobile - simplified algorithm
        TEST_MODE      // Testing/debugging
    };
    
    // Initialize hasher with platform-specific mode
    explicit ZionHasher(AlgorithmMode mode = AlgorithmMode::FULL_POWER);
    
    // Main hash function - platform independent
    HashResult compute_hash(const uint8_t* header, size_t header_len, 
                           uint32_t nonce, uint64_t target_difficulty);
    
    // Batch mining function for multi-threading
    std::vector<HashResult> mine_batch(const uint8_t* header, size_t header_len,
                                      uint32_t start_nonce, uint32_t batch_size,
                                      uint64_t target_difficulty);
    
    // Platform-specific optimizations
    void set_cpu_features(bool has_avx2, bool has_avx512, bool has_neon);
    void set_mobile_constraints(int max_temp_celsius, int battery_level);
    
    // Performance metrics
    uint64_t get_hashes_computed() const { return total_hashes_; }
    double get_average_hash_time_ms() const;
    
private:
    AlgorithmMode mode_;
    uint64_t total_hashes_;
    std::vector<double> timing_samples_;
    
    // CPU feature flags
    bool has_avx2_;
    bool has_avx512_;
    bool has_neon_;
    
    // Mobile constraints
    int max_temp_;
    int battery_level_;
    
    // Core algorithm implementations
    void blake3_hash(const uint8_t* input, size_t len, uint8_t* output);
    void keccak256_hash(const uint8_t* input, size_t len, uint8_t* output);
    void sha3_512_hash(const uint8_t* input, size_t len, uint8_t* output);
    void golden_ratio_transform(uint8_t* data, size_t len);
    
    // Platform-specific optimizations
    void full_power_hash(const uint8_t* input, size_t len, uint32_t nonce, uint8_t* output);
    void mobile_lite_hash(const uint8_t* input, size_t len, uint32_t nonce, uint8_t* output);
    void test_mode_hash(const uint8_t* input, size_t len, uint32_t nonce, uint8_t* output);
};

// Utility functions for all platforms
uint64_t hash_to_difficulty(const uint8_t* hash);
bool meets_target_difficulty(const uint8_t* hash, uint64_t target);
void format_hash_hex(const uint8_t* hash, char* output, size_t output_len);

// Platform detection
enum class Platform {
    LINUX_X64,
    MACOS_INTEL,
    MACOS_APPLE_SILICON,
    WINDOWS_X64,
    ANDROID_ARM64,
    IOS_ARM64,
    UNKNOWN
};

Platform detect_platform();
const char* platform_name(Platform p);

} // namespace core
} // namespace zion

#endif // ZION_COSMIC_HARMONY_CORE_H