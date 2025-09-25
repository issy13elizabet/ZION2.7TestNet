#pragma once

#include "zion.h"
#include <randomx.h>
#include <memory>
#include <mutex>

namespace zion {

class RandomXWrapper {
public:
    static RandomXWrapper& instance();
    
    // Initialize RandomX with a key (seed)
    // use_dataset = true => alokuje a inicializuje 2GiB dataset (rychlejší, ale náročné na start)
    // use_dataset = false => lehký režim (pouze cache), vhodné pro daemon/start
    bool initialize(const Hash& key, bool use_dataset = false);
    
    // Calculate RandomX hash
    Hash hash(const void* data, size_t size);
    
    // Verify hash against difficulty target
    bool verify_hash(const Hash& hash, uint64_t difficulty);
    
    // Calculate difficulty from hash
    uint64_t hash_to_difficulty(const Hash& hash);
    
    // Cleanup
    void cleanup();
    
private:
    RandomXWrapper() = default;
    ~RandomXWrapper();
    
    randomx_cache* cache_ = nullptr;
    randomx_dataset* dataset_ = nullptr;
    randomx_vm* vm_ = nullptr;
    std::mutex mutex_;
    bool initialized_ = false;
};

// Utility functions
Hash sha256(const void* data, size_t size);
Hash double_sha256(const void* data, size_t size);

// Key generation utilities
std::pair<PrivateKey, PublicKey> generate_keypair();
bool verify_signature(const Hash& message, const Signature& signature, const PublicKey& public_key);
Signature sign_message(const Hash& message, const PrivateKey& private_key);

} // namespace zion
