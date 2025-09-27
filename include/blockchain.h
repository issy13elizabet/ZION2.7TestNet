#pragma once

#include "zion.h"
#include "block.h"
#include "transaction.h"
#include <unordered_map>
#include <deque>
#include <mutex>
#include <memory>

namespace zion {

class Blockchain {
public:
    Blockchain();
    ~Blockchain();
    
    // Initialize blockchain with genesis block
    void initialize();
    void initialize_with_genesis(uint64_t genesis_timestamp, uint32_t difficulty_bits, const PublicKey& coinbase_addr);
    
    // Add new block
    bool addBlock(std::shared_ptr<Block> block);
    
    // Create genesis block
    std::shared_ptr<Block> createGenesisBlock();
    
    // Get blocks
    std::shared_ptr<Block> getBlock(uint32_t height) const;
    std::shared_ptr<Block> getBlock(const Hash& hash) const;
    std::shared_ptr<Block> getLatestBlock() const;
    
    // Get blockchain info
    uint32_t getHeight() const;
    uint32_t getDifficulty() const;
    uint64_t getTotalSupply() const;
    
    // Transaction pool (mempool)
    bool addTransaction(std::shared_ptr<Transaction> tx);
    std::shared_ptr<Transaction> getMempoolTransaction(const Hash& tx_hash) const;
    std::vector<std::shared_ptr<Transaction>> getPendingTransactions() const;
    void removePendingTransaction(const Hash& tx_hash);

    // Mempool fee policy
    void setMempoolMinFee(uint64_t fee_per_kb) { mempool_min_fee_per_kb_ = fee_per_kb; }
    uint64_t getMempoolMinFee() const { return mempool_min_fee_per_kb_; }
    
    // UTXO management
    struct UTXO {
        Hash tx_hash;
        uint32_t output_index;
        uint64_t amount;
        PublicKey owner;
    };
    
    std::vector<UTXO> getUTXOs(const PublicKey& address) const;
    uint64_t getBalance(const PublicKey& address) const;
    
    // Validation
    bool validateBlock(const Block& block) const;
    bool validateTransaction(const Transaction& tx) const;
    
    // Difficulty adjustment
    uint32_t calculateNextDifficulty() const;
    
    // Mining reward calculation
    uint64_t calculateBlockReward(uint32_t height) const;
    
    // Persistence
    bool saveToFile(const std::string& filename) const;
    bool loadFromFile(const std::string& filename);
    
private:
    mutable std::mutex mutex_;
    
    // Custom hasher for Hash type
    struct HashHasher {
        size_t operator()(const Hash& h) const {
            size_t result = 0;
            for (size_t i = 0; i < 8; ++i) {
                result = (result << 8) | h[i];
            }
            return result;
        }
    };
    
    // Blockchain storage
    std::vector<std::shared_ptr<Block>> blocks_;
    std::unordered_map<Hash, std::shared_ptr<Block>, HashHasher> block_index_;
    
    // Transaction pool
    std::unordered_map<Hash, std::shared_ptr<Transaction>, HashHasher> mempool_;
    
    // UTXO set
    std::unordered_map<std::string, UTXO> utxo_set_;
    
    // Current state
    uint32_t current_difficulty_;
    uint64_t total_supply_;
    uint64_t mempool_min_fee_per_kb_ = 0;
    
    // Helper functions
    void updateUTXOSet(const Block& block);
    std::string makeUTXOKey(const Hash& tx_hash, uint32_t output_index) const;
    bool verifyBlockchain() const;
};

} // namespace zion
