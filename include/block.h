#pragma once

#include "zion.h"
#include "transaction.h"
#include <ctime>
#include <vector>
#include <string>
#include <memory>
#include <array>
#include <chrono>

namespace zion {

class Block {
public:
    struct Header {
        uint32_t version;
        uint32_t height;
        Hash prev_hash;
        Hash merkle_root;
        uint64_t timestamp;
        uint32_t difficulty;
        uint32_t nonce;
        
        Header() : version(1), height(0), timestamp(0), difficulty(1), nonce(0) {
            prev_hash.fill(0);
            merkle_root.fill(0);
        }
        
        // Serialize header for hashing
        std::vector<uint8_t> serialize() const;
    };

    Block();
    Block(uint32_t height, const Hash& prev_hash);
    Block(uint32_t height, const Hash& prev_hash, uint64_t timestamp);
    
    // Add transaction to block
    void addTransaction(std::shared_ptr<Transaction> tx);
    
    // Mining functions
    // Optional attempts_out counts the number of nonce attempts performed
    // start_nonce: starting nonce value for this attempt range
    // step: increment step (for multi-thread stride)
    // max_attempts: if > 0, stop after this many attempts (useful for yielding)
    bool mine(uint32_t difficulty, uint64_t* attempts_out = nullptr,
              uint32_t start_nonce = 0, uint32_t step = 1, uint64_t max_attempts = 0);
    Hash calculateHash() const;
    Hash calculateMerkleRoot() const;
    
    // Validation
    bool isValid() const;
    bool validateProofOfWork() const;
    
    // Getters
    const Header& getHeader() const { return header_; }
    const std::vector<std::shared_ptr<Transaction>>& getTransactions() const { return transactions_; }
    uint32_t getHeight() const { return header_.height; }
    const Hash& getPrevHash() const { return header_.prev_hash; }
    uint64_t getTimestamp() const { return header_.timestamp; }
    
    // Serialization
    std::vector<uint8_t> serialize() const;
    static std::unique_ptr<Block> deserialize(const std::vector<uint8_t>& data);
    
private:
    Header header_;
    std::vector<std::shared_ptr<Transaction>> transactions_;
    
    // RandomX proof of work
    bool checkRandomXProof(const Hash& hash, uint32_t difficulty) const;
};

} // namespace zion
