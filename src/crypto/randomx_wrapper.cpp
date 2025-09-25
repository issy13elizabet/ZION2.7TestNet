#include "randomx_wrapper.h"
#include <openssl/sha.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <openssl/ec.h>
#include <openssl/ecdsa.h>
#include <algorithm>
#include <iostream>
#include <vector>

namespace zion {

RandomXWrapper& RandomXWrapper::instance() {
    static RandomXWrapper instance;
    return instance;
}

RandomXWrapper::~RandomXWrapper() {
    cleanup();
}

bool RandomXWrapper::initialize(const Hash& key, bool use_dataset) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (initialized_) {
        cleanup();
    }
    
    try {
        // Create cache
        cache_ = randomx_alloc_cache(RANDOMX_FLAG_DEFAULT);
        if (!cache_) {
            std::cerr << "Failed to allocate RandomX cache" << std::endl;
            return false;
        }
        
        // Initialize cache with key
        randomx_init_cache(cache_, key.data(), key.size());
        
        // Create dataset (optional, for better performance)
        if (use_dataset) {
            dataset_ = randomx_alloc_dataset(RANDOMX_FLAG_DEFAULT);
            if (dataset_) {
                // Get dataset item count
                auto item_count = randomx_dataset_item_count();
                randomx_init_dataset(dataset_, cache_, 0, item_count);
            }
        }
        
        // Create VM
        randomx_flags flags = RANDOMX_FLAG_DEFAULT;
        if (dataset_) {
            flags |= RANDOMX_FLAG_FULL_MEM;
            vm_ = randomx_create_vm(flags, cache_, dataset_);
        } else {
            vm_ = randomx_create_vm(flags, cache_, nullptr);
        }
        
        if (!vm_) {
            std::cerr << "Failed to create RandomX VM" << std::endl;
            cleanup();
            return false;
        }
        
        initialized_ = true;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in RandomX initialization: " << e.what() << std::endl;
        cleanup();
        return false;
    }
}

Hash RandomXWrapper::hash(const void* data, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    Hash result;
    result.fill(0);
    
    if (!initialized_ || !vm_) {
        std::cerr << "RandomX not initialized" << std::endl;
        return result;
    }
    
    randomx_calculate_hash(vm_, data, size, result.data());
    return result;
}

bool RandomXWrapper::verify_hash(const Hash& hash, uint64_t difficulty) {
    if (difficulty == 0) return false;
    
    // Convert hash to difficulty value
    uint64_t hash_diff = hash_to_difficulty(hash);
    return hash_diff >= difficulty;
}

uint64_t RandomXWrapper::hash_to_difficulty(const Hash& hash) {
    // Convert the first 8 bytes of hash to uint64_t (big endian)
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value = (value << 8) | hash[i];
    }
    
    // Return the inverse to get difficulty (higher hash value = lower difficulty)
    if (value == 0) return UINT64_MAX;
    return UINT64_MAX / value;
}

void RandomXWrapper::cleanup() {
    if (vm_) {
        randomx_destroy_vm(vm_);
        vm_ = nullptr;
    }
    
    if (dataset_) {
        randomx_release_dataset(dataset_);
        dataset_ = nullptr;
    }
    
    if (cache_) {
        randomx_release_cache(cache_);
        cache_ = nullptr;
    }
    
    initialized_ = false;
}

// Utility functions implementation
Hash sha256(const void* data, size_t size) {
    Hash result;
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, data, size);
    SHA256_Final(result.data(), &sha256);
    return result;
}

Hash double_sha256(const void* data, size_t size) {
    Hash first_hash = sha256(data, size);
    return sha256(first_hash.data(), first_hash.size());
}

std::pair<PrivateKey, PublicKey> generate_keypair() {
    PrivateKey private_key;
    PublicKey public_key;
    
    // Generate random private key
    if (RAND_bytes(private_key.data(), private_key.size()) != 1) {
        // Fill with zeros on failure
        private_key.fill(0);
        public_key.fill(0);
        return {private_key, public_key};
    }
    
    // Create EC key
    EC_KEY* ec_key = EC_KEY_new_by_curve_name(NID_secp256k1);
    if (!ec_key) {
        private_key.fill(0);
        public_key.fill(0);
        return {private_key, public_key};
    }
    
    // Set private key
    BIGNUM* bn = BN_new();
    BN_bin2bn(private_key.data(), private_key.size(), bn);
    EC_KEY_set_private_key(ec_key, bn);
    
    // Generate public key
    const EC_GROUP* group = EC_KEY_get0_group(ec_key);
    EC_POINT* pub_point = EC_POINT_new(group);
    EC_POINT_mul(group, pub_point, bn, nullptr, nullptr, nullptr);
    EC_KEY_set_public_key(ec_key, pub_point);
    
    // Extract compressed public key (usually 33 bytes)
    size_t pub_len = EC_POINT_point2oct(group, pub_point, POINT_CONVERSION_COMPRESSED,
                                        nullptr, 0, nullptr);
    std::vector<unsigned char> pub_compressed(pub_len);
    EC_POINT_point2oct(group, pub_point, POINT_CONVERSION_COMPRESSED,
                       pub_compressed.data(), pub_len, nullptr);

    // Derive 32-byte address from compressed pubkey using SHA-256
    Hash pub_hash = sha256(pub_compressed.data(), pub_compressed.size());
    std::copy(pub_hash.begin(), pub_hash.end(), public_key.begin());
    
    // Cleanup
    EC_POINT_free(pub_point);
    BN_free(bn);
    EC_KEY_free(ec_key);
    
    return {private_key, public_key};
}

bool verify_signature(const Hash& message, const Signature& signature, const PublicKey& public_key) {
    // Create EC key from public key
    EC_KEY* ec_key = EC_KEY_new_by_curve_name(NID_secp256k1);
    if (!ec_key) return false;
    
    const EC_GROUP* group = EC_KEY_get0_group(ec_key);
    EC_POINT* pub_point = EC_POINT_new(group);
    
    if (EC_POINT_oct2point(group, pub_point, public_key.data(), public_key.size(), nullptr) != 1) {
        EC_POINT_free(pub_point);
        EC_KEY_free(ec_key);
        return false;
    }
    
    EC_KEY_set_public_key(ec_key, pub_point);
    
    // Verify signature
    int result = ECDSA_verify(0, message.data(), message.size(), signature.data(), signature.size(), ec_key);
    
    EC_POINT_free(pub_point);
    EC_KEY_free(ec_key);
    
    return result == 1;
}

Signature sign_message(const Hash& message, const PrivateKey& private_key) {
    Signature signature;
    signature.fill(0);
    
    // Create EC key from private key
    EC_KEY* ec_key = EC_KEY_new_by_curve_name(NID_secp256k1);
    if (!ec_key) return signature;
    
    BIGNUM* bn = BN_new();
    BN_bin2bn(private_key.data(), private_key.size(), bn);
    EC_KEY_set_private_key(ec_key, bn);
    
    // Generate public key
    const EC_GROUP* group = EC_KEY_get0_group(ec_key);
    EC_POINT* pub_point = EC_POINT_new(group);
    EC_POINT_mul(group, pub_point, bn, nullptr, nullptr, nullptr);
    EC_KEY_set_public_key(ec_key, pub_point);
    
    // Sign message
    unsigned int sig_len = signature.size();
    ECDSA_sign(0, message.data(), message.size(), signature.data(), &sig_len, ec_key);
    
    // Cleanup
    EC_POINT_free(pub_point);
    BN_free(bn);
    EC_KEY_free(ec_key);
    
    return signature;
}

} // namespace zion
