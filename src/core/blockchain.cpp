#include "blockchain.h"
#include "randomx_wrapper.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

// Helper for hash comparison
struct HashComparator {
    size_t operator()(const zion::Hash& h) const {
        size_t result = 0;
        for (size_t i = 0; i < 8; ++i) {
            result = (result << 8) | h[i];
        }
        return result;
    }
};

namespace zion {

Blockchain::Blockchain() 
    : current_difficulty_(1), total_supply_(0), mempool_min_fee_per_kb_(0) {
}

Blockchain::~Blockchain() {
}

void Blockchain::initialize() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (blocks_.empty()) {
        auto genesis = createGenesisBlock();
        blocks_.push_back(genesis);
        block_index_[genesis->calculateHash()] = genesis;
        updateUTXOSet(*genesis);
    }
}

void Blockchain::initialize_with_genesis(uint64_t genesis_timestamp, uint32_t difficulty_bits, const PublicKey& coinbase_addr) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!blocks_.empty()) return;

    Hash prev; prev.fill(0);
    auto genesis = std::make_shared<Block>(0, prev, genesis_timestamp ? genesis_timestamp : (
        uint64_t)std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());

    // vytvořit coinbase
    PublicKey addr = coinbase_addr;
    bool addr_empty = true; for (auto b : addr) { if (b != 0) { addr_empty = false; break; } }
    if (addr_empty) { addr.fill(0x42); }
    auto coinbase = std::make_shared<Transaction>(Transaction::create_coinbase(addr, ZION_INITIAL_REWARD));
    genesis->addTransaction(coinbase);

    // vytěžit s danou obtížností (v bitech)
    uint32_t bits = difficulty_bits == 0 ? 1 : difficulty_bits;
    genesis->mine(bits);

    blocks_.push_back(genesis);
    block_index_[genesis->calculateHash()] = genesis;
    updateUTXOSet(*genesis);
}

std::shared_ptr<Block> Blockchain::createGenesisBlock() {
    // Genesis block parameters
    Hash genesis_prev;
    genesis_prev.fill(0);
    
    auto genesis_block = std::make_shared<Block>(0, genesis_prev);
    
    // Create genesis coinbase transaction
    PublicKey genesis_miner;
    genesis_miner.fill(0x42); // Arbitrary genesis miner address
    
    auto coinbase = std::make_shared<Transaction>(
        Transaction::create_coinbase(genesis_miner, ZION_INITIAL_REWARD)
    );
    
    genesis_block->addTransaction(coinbase);
    
    // Mine genesis block with low difficulty
    genesis_block->mine(1);
    
    return genesis_block;
}

bool Blockchain::addBlock(std::shared_ptr<Block> block) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Validate block
    if (!validateBlock(*block)) {
        return false;
    }
    
    // Check if block connects to chain
    if (blocks_.empty()) {
        return false;
    }
    
    auto latest = blocks_.back();
    if (block->getPrevHash() != latest->calculateHash()) {
        // Might be a fork or orphan block
        return false;
    }
    
    // Add block
    blocks_.push_back(block);
    block_index_[block->calculateHash()] = block;
    
    // Update UTXO set
    updateUTXOSet(*block);
    
    // Update total supply
    for (const auto& tx : block->getTransactions()) {
        if (tx->is_coinbase()) {
            for (const auto& output : tx->get_outputs()) {
                total_supply_ += output.amount;
            }
        }
    }
    
    // Adjust difficulty if needed
    if (blocks_.size() % ZION_DIFFICULTY_WINDOW == 0) {
        current_difficulty_ = calculateNextDifficulty();
    }
    
    return true;
}

std::shared_ptr<Block> Blockchain::getBlock(uint32_t height) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (height < blocks_.size()) {
        return blocks_[height];
    }
    return nullptr;
}

std::shared_ptr<Block> Blockchain::getBlock(const Hash& hash) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = block_index_.find(hash);
    if (it != block_index_.end()) {
        return it->second;
    }
    return nullptr;
}

std::shared_ptr<Block> Blockchain::getLatestBlock() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!blocks_.empty()) {
        return blocks_.back();
    }
    return nullptr;
}

uint32_t Blockchain::getHeight() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return blocks_.empty() ? 0 : blocks_.size() - 1;
}

uint32_t Blockchain::getDifficulty() const {
    return current_difficulty_;
}

uint64_t Blockchain::getTotalSupply() const {
    return total_supply_;
}

bool Blockchain::addTransaction(std::shared_ptr<Transaction> tx) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Validate transaction
    if (!validateTransaction(*tx)) {
        return false;
    }
    
    // Add to mempool
    Hash tx_hash = tx->get_hash();
    mempool_[tx_hash] = tx;
    
    return true;
}

std::vector<std::shared_ptr<Transaction>> Blockchain::getPendingTransactions() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::shared_ptr<Transaction>> result;
    for (const auto& [hash, tx] : mempool_) {
        result.push_back(tx);
    }
    return result;
}

std::shared_ptr<Transaction> Blockchain::getMempoolTransaction(const Hash& tx_hash) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = mempool_.find(tx_hash);
    if (it != mempool_.end()) {
        return it->second;
    }
    return nullptr;
}

void Blockchain::removePendingTransaction(const Hash& tx_hash) {
    std::lock_guard<std::mutex> lock(mutex_);
    mempool_.erase(tx_hash);
}

std::vector<Blockchain::UTXO> Blockchain::getUTXOs(const PublicKey& address) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<UTXO> result;
    for (const auto& [key, utxo] : utxo_set_) {
        if (utxo.owner == address) {
            result.push_back(utxo);
        }
    }
    return result;
}

uint64_t Blockchain::getBalance(const PublicKey& address) const {
    auto utxos = getUTXOs(address);
    uint64_t balance = 0;
    
    for (const auto& utxo : utxos) {
        balance += utxo.amount;
    }
    
    return balance;
}

bool Blockchain::validateBlock(const Block& block) const {
    // Basic validation
    if (!block.isValid()) {
        return false;
    }
    
    // Check block size
    auto serialized = block.serialize();
    if (serialized.size() > ZION_MAX_BLOCK_SIZE) {
        return false;
    }
    
    // Validate all transactions
    for (const auto& tx : block.getTransactions()) {
        if (!validateTransaction(*tx)) {
            return false;
        }
    }
    
    return true;
}

bool Blockchain::validateTransaction(const Transaction& tx) const {
    // Basic validation
    if (!tx.is_valid()) {
        return false;
    }
    
    // Check transaction size
    auto serialized = tx.serialize();
    if (serialized.size() > ZION_MAX_TX_SIZE) {
        return false;
    }
    
    // For non-coinbase transactions, verify inputs exist in UTXO set and sums
    if (!tx.is_coinbase()) {
        uint64_t input_sum = 0;
        for (const auto& input : tx.get_inputs()) {
            std::string key = makeUTXOKey(input.prev_tx_hash, input.output_index);
            auto it = utxo_set_.find(key);
            if (it == utxo_set_.end()) {
                return false; // Input doesn't exist
            }
            input_sum += it->second.amount;
        }
        uint64_t output_sum = 0;
        for (const auto& out : tx.get_outputs()) { output_sum += out.amount; }
        if (input_sum < output_sum) {
            return false; // Not enough funds
        }
        // Enforce min fee per KB
        uint64_t fee = input_sum - output_sum;
        // compute size in KB (ceil)
        size_t tx_size = serialized.size();
        uint64_t kb = (tx_size + 1023) / 1024;
        uint64_t required_fee = mempool_min_fee_per_kb_ * (kb == 0 ? 1 : kb);
        if (fee < required_fee) {
            return false;
        }
    }
    
    return true;
}

uint32_t Blockchain::calculateNextDifficulty() const {
    if (blocks_.size() < ZION_DIFFICULTY_WINDOW + 1) {
        return current_difficulty_;
    }
    
    // Get the last window of blocks
    size_t start_idx = blocks_.size() - ZION_DIFFICULTY_WINDOW;
    auto start_block = blocks_[start_idx];
    auto end_block = blocks_.back();
    
    // Calculate actual time taken
    uint64_t actual_time = end_block->getTimestamp() - start_block->getTimestamp();
    uint64_t expected_time = ZION_TARGET_BLOCK_TIME * ZION_DIFFICULTY_WINDOW;
    
    // Adjust difficulty
    uint32_t new_difficulty = current_difficulty_;
    
    if (actual_time < expected_time / 2) {
        new_difficulty = current_difficulty_ * 2; // Double difficulty
    } else if (actual_time < expected_time) {
        new_difficulty = current_difficulty_ + 1; // Slight increase
    } else if (actual_time > expected_time * 2) {
        new_difficulty = current_difficulty_ / 2; // Half difficulty
        if (new_difficulty < 1) new_difficulty = 1;
    } else if (actual_time > expected_time) {
        new_difficulty = current_difficulty_ - 1; // Slight decrease
        if (new_difficulty < 1) new_difficulty = 1;
    }
    
    return new_difficulty;
}

uint64_t Blockchain::calculateBlockReward(uint32_t height) const {
    uint32_t halvenings = height / ZION_HALVENING_INTERVAL;
    uint64_t reward = ZION_INITIAL_REWARD;
    
    for (uint32_t i = 0; i < halvenings; ++i) {
        reward /= 2;
    }
    
    return reward;
}

void Blockchain::updateUTXOSet(const Block& block) {
    // Remove spent UTXOs
    for (const auto& tx : block.getTransactions()) {
        if (!tx->is_coinbase()) {
            for (const auto& input : tx->get_inputs()) {
                std::string key = makeUTXOKey(input.prev_tx_hash, input.output_index);
                utxo_set_.erase(key);
            }
        }
    }
    
    // Add new UTXOs
    for (const auto& tx : block.getTransactions()) {
        Hash tx_hash = tx->get_hash();
        const auto& outputs = tx->get_outputs();
        
        for (uint32_t i = 0; i < outputs.size(); ++i) {
            UTXO utxo;
            utxo.tx_hash = tx_hash;
            utxo.output_index = i;
            utxo.amount = outputs[i].amount;
            utxo.owner = outputs[i].recipient;
            
            std::string key = makeUTXOKey(tx_hash, i);
            utxo_set_[key] = utxo;
        }
    }
}

std::string Blockchain::makeUTXOKey(const Hash& tx_hash, uint32_t output_index) const {
    std::stringstream ss;
    for (const auto& byte : tx_hash) {
        ss << std::hex << std::setfill('0') << std::setw(2) 
           << static_cast<int>(byte);
    }
    ss << ":" << output_index;
    return ss.str();
}

bool Blockchain::saveToFile(const std::string& filename) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        return false;
    }
    
    // Write number of blocks
    uint32_t num_blocks = blocks_.size();
    file.write(reinterpret_cast<const char*>(&num_blocks), sizeof(num_blocks));
    
    // Write each block
    for (const auto& block : blocks_) {
        auto data = block->serialize();
        uint32_t size = data.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        file.write(reinterpret_cast<const char*>(data.data()), data.size());
    }
    
    return true;
}

bool Blockchain::loadFromFile(const std::string& filename) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        return false;
    }
    
    // Clear existing data
    blocks_.clear();
    block_index_.clear();
    utxo_set_.clear();
    mempool_.clear();
    total_supply_ = 0;
    
    // Read number of blocks
    uint32_t num_blocks;
    file.read(reinterpret_cast<char*>(&num_blocks), sizeof(num_blocks));
    
    // Read each block
    for (uint32_t i = 0; i < num_blocks; ++i) {
        uint32_t size;
        file.read(reinterpret_cast<char*>(&size), sizeof(size));
        
        std::vector<uint8_t> data(size);
        file.read(reinterpret_cast<char*>(data.data()), size);
        
        auto block = Block::deserialize(data);
        if (block) {
            // Convert unique_ptr to shared_ptr
            std::shared_ptr<Block> shared_block(std::move(block));
            blocks_.push_back(shared_block);
            block_index_[shared_block->calculateHash()] = shared_block;
            updateUTXOSet(*shared_block);
            
            // Update total supply
            for (const auto& tx : shared_block->getTransactions()) {
                if (tx->is_coinbase()) {
                    for (const auto& output : tx->get_outputs()) {
                        total_supply_ += output.amount;
                    }
                }
            }
        }
    }
    
    // Update difficulty
    if (blocks_.size() > 0) {
        current_difficulty_ = calculateNextDifficulty();
    }
    
    return true;
}

bool Blockchain::verifyBlockchain() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (blocks_.empty()) {
        return true;
    }
    
    // Verify each block
    for (size_t i = 1; i < blocks_.size(); ++i) {
        auto& prev_block = blocks_[i - 1];
        auto& curr_block = blocks_[i];
        
        // Check previous hash
        if (curr_block->getPrevHash() != prev_block->calculateHash()) {
            return false;
        }
        
        // Validate block
        if (!validateBlock(*curr_block)) {
            return false;
        }
    }
    
    return true;
}

} // namespace zion
