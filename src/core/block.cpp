#include "block.h"
#include "randomx_wrapper.h"
#include <cstring>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <chrono>

namespace zion {

Block::Block() {
    header_.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}

Block::Block(uint32_t height, const Hash& prev_hash) {
    header_.height = height;
    header_.prev_hash = prev_hash;
    header_.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}

Block::Block(uint32_t height, const Hash& prev_hash, uint64_t timestamp) {
    header_.height = height;
    header_.prev_hash = prev_hash;
    header_.timestamp = timestamp;
}

void Block::addTransaction(std::shared_ptr<Transaction> tx) {
    transactions_.push_back(tx);
    // Recalculate merkle root when transactions change
    header_.merkle_root = calculateMerkleRoot();
}

Hash Block::calculateHash() const {
    auto header_data = header_.serialize();
    return sha256(header_data.data(), header_data.size());
}

Hash Block::calculateMerkleRoot() const {
    if (transactions_.empty()) {
        Hash empty;
        empty.fill(0);
        return empty;
    }
    
    std::vector<Hash> tree;
    
    // Add all transaction hashes to the tree
    for (const auto& tx : transactions_) {
        tree.push_back(tx->get_hash());
    }
    
    // Build merkle tree
    while (tree.size() > 1) {
        std::vector<Hash> new_level;
        
        for (size_t i = 0; i < tree.size(); i += 2) {
            if (i + 1 < tree.size()) {
                // Combine two hashes
                std::vector<uint8_t> combined;
                combined.insert(combined.end(), tree[i].begin(), tree[i].end());
                combined.insert(combined.end(), tree[i + 1].begin(), tree[i + 1].end());
                new_level.push_back(sha256(combined.data(), combined.size()));
            } else {
                // Odd number, duplicate last hash
                std::vector<uint8_t> combined;
                combined.insert(combined.end(), tree[i].begin(), tree[i].end());
                combined.insert(combined.end(), tree[i].begin(), tree[i].end());
                new_level.push_back(sha256(combined.data(), combined.size()));
            }
        }
        
        tree = new_level;
    }
    
    return tree[0];
}

bool Block::mine(uint32_t difficulty, uint64_t* attempts_out,
                  uint32_t start_nonce, uint32_t step, uint64_t max_attempts) {
    header_.difficulty = difficulty;
    header_.merkle_root = calculateMerkleRoot();
    
    // Initialize RandomX if not already done
    auto& randomx = RandomXWrapper::instance();
    if (!randomx.initialize(header_.prev_hash, false)) {
        return false;
    }
    
    if (attempts_out) *attempts_out = 0;
    if (step == 0) step = 1;
    
    // Mining loop
    uint64_t attempts = 0;
    uint32_t nonce = start_nonce;
    while (true) {
        header_.nonce = nonce;
        Hash block_hash = calculateHash();
        
        // Use RandomX to hash the block hash
        Hash pow_hash = randomx.hash(block_hash.data(), block_hash.size());
        
        attempts++;
        if (attempts_out) *attempts_out = attempts;
        
        if (checkRandomXProof(pow_hash, difficulty)) {
            return true;
        }
        
        // Stop if max_attempts reached
        if (max_attempts > 0 && attempts >= max_attempts) {
            break;
        }
        
        // Increment nonce with stride; stop on wrap
        uint32_t next = nonce + step;
        if (next < nonce) { // wrap detected
            break;
        }
        nonce = next;
        
        // Print mining progress occasionally (optional / debug)
        // if ((attempts % 1000000ULL) == 0ULL) {
        //     std::cout << "Mining... attempts: " << attempts << std::endl;
        // }
    }
    
    return false;
}

bool Block::checkRandomXProof(const Hash& hash, uint32_t difficulty) const {
    // Check if hash meets difficulty target
    // Difficulty is number of leading zero bits required
    uint32_t leading_zeros = 0;
    
    for (size_t i = 0; i < hash.size(); i++) {
        if (hash[i] == 0) {
            leading_zeros += 8;
        } else {
            // Count leading zeros in this byte
            uint8_t byte = hash[i];
            while ((byte & 0x80) == 0 && leading_zeros < difficulty) {
                leading_zeros++;
                byte <<= 1;
            }
            break;
        }
    }
    
    return leading_zeros >= difficulty;
}

bool Block::isValid() const {
    // Check basic block validity
    if (header_.version != 1) {
        return false;
    }
    
    // Verify merkle root
    if (header_.merkle_root != calculateMerkleRoot()) {
        return false;
    }
    
    // Verify all transactions
    for (const auto& tx : transactions_) {
        if (!tx->is_valid()) {
            return false;
        }
    }
    
    // Verify proof of work
    return validateProofOfWork();
}

bool Block::validateProofOfWork() const {
    auto& randomx = RandomXWrapper::instance();
    if (!randomx.initialize(header_.prev_hash, false)) {
        return false;
    }
    
    Hash block_hash = calculateHash();
    Hash pow_hash = randomx.hash(block_hash.data(), block_hash.size());
    
    return checkRandomXProof(pow_hash, header_.difficulty);
}

std::vector<uint8_t> Block::Header::serialize() const {
    std::vector<uint8_t> result;
    
    // Calculate size
    size_t size = sizeof(uint32_t) * 4 + sizeof(uint64_t) + Hash().size() * 2;
    result.resize(size);
    
    size_t offset = 0;
    
    // Version
    memcpy(result.data() + offset, &version, sizeof(version));
    offset += sizeof(version);
    
    // Height
    memcpy(result.data() + offset, &height, sizeof(height));
    offset += sizeof(height);
    
    // Previous hash
    memcpy(result.data() + offset, prev_hash.data(), prev_hash.size());
    offset += prev_hash.size();
    
    // Merkle root
    memcpy(result.data() + offset, merkle_root.data(), merkle_root.size());
    offset += merkle_root.size();
    
    // Timestamp
    memcpy(result.data() + offset, &timestamp, sizeof(timestamp));
    offset += sizeof(timestamp);
    
    // Difficulty
    memcpy(result.data() + offset, &difficulty, sizeof(difficulty));
    offset += sizeof(difficulty);
    
    // Nonce
    memcpy(result.data() + offset, &nonce, sizeof(nonce));
    
    return result;
}

std::vector<uint8_t> Block::serialize() const {
    std::vector<uint8_t> result;
    
    // Serialize header
    auto header_data = header_.serialize();
    result.insert(result.end(), header_data.begin(), header_data.end());
    
    // Serialize number of transactions
    uint32_t tx_count = transactions_.size();
    result.resize(result.size() + sizeof(tx_count));
    memcpy(result.data() + result.size() - sizeof(tx_count), &tx_count, sizeof(tx_count));
    
    // Serialize each transaction
    for (const auto& tx : transactions_) {
        auto tx_data = tx->serialize();
        uint32_t tx_size = tx_data.size();
        
        // Add transaction size
        result.resize(result.size() + sizeof(tx_size));
        memcpy(result.data() + result.size() - sizeof(tx_size), &tx_size, sizeof(tx_size));
        
        // Add transaction data
        result.insert(result.end(), tx_data.begin(), tx_data.end());
    }
    
    return result;
}

std::unique_ptr<Block> Block::deserialize(const std::vector<uint8_t>& data) {
    if (data.size() < sizeof(Header)) {
        return nullptr;
    }
    
    auto block = std::make_unique<Block>();
    size_t offset = 0;
    
    // Deserialize header
    memcpy(&block->header_.version, data.data() + offset, sizeof(uint32_t));
    offset += sizeof(uint32_t);
    
    memcpy(&block->header_.height, data.data() + offset, sizeof(uint32_t));
    offset += sizeof(uint32_t);
    
    memcpy(block->header_.prev_hash.data(), data.data() + offset, block->header_.prev_hash.size());
    offset += block->header_.prev_hash.size();
    
    memcpy(block->header_.merkle_root.data(), data.data() + offset, block->header_.merkle_root.size());
    offset += block->header_.merkle_root.size();
    
    memcpy(&block->header_.timestamp, data.data() + offset, sizeof(uint64_t));
    offset += sizeof(uint64_t);
    
    memcpy(&block->header_.difficulty, data.data() + offset, sizeof(uint32_t));
    offset += sizeof(uint32_t);
    
    memcpy(&block->header_.nonce, data.data() + offset, sizeof(uint32_t));
    offset += sizeof(uint32_t);
    
    // Deserialize transactions
    uint32_t tx_count;
    memcpy(&tx_count, data.data() + offset, sizeof(tx_count));
    offset += sizeof(tx_count);
    
    for (uint32_t i = 0; i < tx_count; i++) {
        uint32_t tx_size;
        memcpy(&tx_size, data.data() + offset, sizeof(tx_size));
        offset += sizeof(tx_size);
        
        std::vector<uint8_t> tx_data(data.begin() + offset, data.begin() + offset + tx_size);
        auto tx = Transaction::from_bytes(tx_data);
        if (tx) {
            block->transactions_.push_back(tx);
        }
        offset += tx_size;
    }
    
    return block;
}

} // namespace zion
