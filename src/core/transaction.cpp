#include "transaction.h"
#include "randomx_wrapper.h"
#include <cstring>
#include <sstream>

namespace zion {

Transaction::Transaction() : version_(1), lock_time_(0) {}

Transaction::Transaction(const std::vector<TxInput>& inputs, const std::vector<TxOutput>& outputs)
    : version_(1), inputs_(inputs), outputs_(outputs), lock_time_(0) {}

Hash Transaction::get_hash() const {
    return calculate_hash();
}

uint64_t Transaction::get_fee() const {
    uint64_t input_sum = calculate_input_sum();
    uint64_t output_sum = calculate_output_sum();
    
    if (is_coinbase()) {
        return 0;
    }
    
    return (input_sum > output_sum) ? (input_sum - output_sum) : 0;
}

void Transaction::add_input(const TxInput& input) {
    inputs_.push_back(input);
}

void Transaction::add_output(const TxOutput& output) {
    outputs_.push_back(output);
}

bool Transaction::is_valid() const {
    // Check if transaction is empty
    if (inputs_.empty() && outputs_.empty()) {
        return false;
    }
    
    // Coinbase transaction checks
    if (is_coinbase()) {
        if (inputs_.size() != 1) {
            return false;
        }
        // Coinbase must have at least one output
        if (outputs_.empty()) {
            return false;
        }
    } else {
        // Regular transaction must have inputs
        if (inputs_.empty()) {
            return false;
        }
    }
    
    // Check for negative or overflow amounts
    for (const auto& output : outputs_) {
        if (output.amount == 0 || output.amount > ZION_MAX_SUPPLY) {
            return false;
        }
    }
    
    // Check total output doesn't exceed max supply
    uint64_t total_output = calculate_output_sum();
    if (total_output > ZION_MAX_SUPPLY) {
        return false;
    }
    
    // For non-coinbase transactions, verify signatures
    if (!is_coinbase() && !verify_signatures()) {
        return false;
    }
    
    return true;
}

bool Transaction::verify_signatures() const {
    if (is_coinbase()) {
        return true;
    }
    
    Hash tx_hash = calculate_hash();
    
    for (const auto& input : inputs_) {
        // Verify each input's signature
        Signature sig;
        if (input.signature.size() == 64) {
            std::memcpy(sig.data(), input.signature.data(), 64);
            if (!verify_signature(tx_hash, sig, input.public_key)) {
                return false;
            }
        } else {
            return false;
        }
    }
    
    return true;
}

std::vector<uint8_t> Transaction::serialize() const {
    std::vector<uint8_t> data;
    
    // Version (4 bytes)
    data.insert(data.end(), reinterpret_cast<const uint8_t*>(&version_),
                reinterpret_cast<const uint8_t*>(&version_) + sizeof(version_));
    
    // Input count (4 bytes)
    uint32_t input_count = inputs_.size();
    data.insert(data.end(), reinterpret_cast<const uint8_t*>(&input_count),
                reinterpret_cast<const uint8_t*>(&input_count) + sizeof(input_count));
    
    // Inputs
    for (const auto& input : inputs_) {
        // Previous tx hash
        data.insert(data.end(), input.prev_tx_hash.begin(), input.prev_tx_hash.end());
        
        // Output index
        data.insert(data.end(), reinterpret_cast<const uint8_t*>(&input.output_index),
                    reinterpret_cast<const uint8_t*>(&input.output_index) + sizeof(input.output_index));
        
        // Signature length
        uint32_t sig_len = input.signature.size();
        data.insert(data.end(), reinterpret_cast<const uint8_t*>(&sig_len),
                    reinterpret_cast<const uint8_t*>(&sig_len) + sizeof(sig_len));
        
        // Signature
        data.insert(data.end(), input.signature.begin(), input.signature.end());
        
        // Public key
        data.insert(data.end(), input.public_key.begin(), input.public_key.end());
    }
    
    // Output count (4 bytes)
    uint32_t output_count = outputs_.size();
    data.insert(data.end(), reinterpret_cast<const uint8_t*>(&output_count),
                reinterpret_cast<const uint8_t*>(&output_count) + sizeof(output_count));
    
    // Outputs
    for (const auto& output : outputs_) {
        // Amount
        data.insert(data.end(), reinterpret_cast<const uint8_t*>(&output.amount),
                    reinterpret_cast<const uint8_t*>(&output.amount) + sizeof(output.amount));
        
        // Recipient
        data.insert(data.end(), output.recipient.begin(), output.recipient.end());
    }
    
    // Lock time
    data.insert(data.end(), reinterpret_cast<const uint8_t*>(&lock_time_),
                reinterpret_cast<const uint8_t*>(&lock_time_) + sizeof(lock_time_));
    
    return data;
}

bool Transaction::deserialize(const std::vector<uint8_t>& data) {
    if (data.size() < sizeof(version_) + sizeof(uint32_t) * 2 + sizeof(lock_time_)) {
        return false;
    }
    
    size_t offset = 0;
    
    // Version
    std::memcpy(&version_, data.data() + offset, sizeof(version_));
    offset += sizeof(version_);
    
    // Input count
    uint32_t input_count;
    std::memcpy(&input_count, data.data() + offset, sizeof(input_count));
    offset += sizeof(input_count);
    
    // Inputs
    inputs_.clear();
    inputs_.reserve(input_count);
    
    for (uint32_t i = 0; i < input_count; ++i) {
        TxInput input;
        
        // Previous tx hash
        if (offset + 32 > data.size()) return false;
        std::memcpy(input.prev_tx_hash.data(), data.data() + offset, 32);
        offset += 32;
        
        // Output index
        if (offset + sizeof(input.output_index) > data.size()) return false;
        std::memcpy(&input.output_index, data.data() + offset, sizeof(input.output_index));
        offset += sizeof(input.output_index);
        
        // Signature length
        uint32_t sig_len;
        if (offset + sizeof(sig_len) > data.size()) return false;
        std::memcpy(&sig_len, data.data() + offset, sizeof(sig_len));
        offset += sizeof(sig_len);
        
        // Signature
        if (offset + sig_len > data.size()) return false;
        input.signature.resize(sig_len);
        std::memcpy(input.signature.data(), data.data() + offset, sig_len);
        offset += sig_len;
        
        // Public key
        if (offset + 32 > data.size()) return false;
        std::memcpy(input.public_key.data(), data.data() + offset, 32);
        offset += 32;
        
        inputs_.push_back(input);
    }
    
    // Output count
    uint32_t output_count;
    if (offset + sizeof(output_count) > data.size()) return false;
    std::memcpy(&output_count, data.data() + offset, sizeof(output_count));
    offset += sizeof(output_count);
    
    // Outputs
    outputs_.clear();
    outputs_.reserve(output_count);
    
    for (uint32_t i = 0; i < output_count; ++i) {
        TxOutput output;
        
        // Amount
        if (offset + sizeof(output.amount) > data.size()) return false;
        std::memcpy(&output.amount, data.data() + offset, sizeof(output.amount));
        offset += sizeof(output.amount);
        
        // Recipient
        if (offset + 32 > data.size()) return false;
        std::memcpy(output.recipient.data(), data.data() + offset, 32);
        offset += 32;
        
        outputs_.push_back(output);
    }
    
    // Lock time
    if (offset + sizeof(lock_time_) > data.size()) return false;
    std::memcpy(&lock_time_, data.data() + offset, sizeof(lock_time_));
    offset += sizeof(lock_time_);
    
    return offset == data.size();
}

Transaction Transaction::create_coinbase(const PublicKey& miner_address, uint64_t reward) {
    Transaction tx;
    tx.version_ = 1;
    
    // Coinbase input (null hash, max index)
    TxInput coinbase_input;
    coinbase_input.prev_tx_hash.fill(0);
    coinbase_input.output_index = 0xFFFFFFFF;
    coinbase_input.public_key = miner_address;
    
    tx.inputs_.push_back(coinbase_input);
    
    // Output to miner
    TxOutput miner_output(reward, miner_address);
    tx.outputs_.push_back(miner_output);
    
    return tx;
}

bool Transaction::is_coinbase() const {
    if (inputs_.size() != 1) {
        return false;
    }
    
    const auto& input = inputs_[0];
    
    // Check if previous tx hash is all zeros
    bool all_zeros = true;
    for (uint8_t byte : input.prev_tx_hash) {
        if (byte != 0) {
            all_zeros = false;
            break;
        }
    }
    
    return all_zeros && input.output_index == 0xFFFFFFFF;
}

Hash Transaction::calculate_hash() const {
    auto tx_data = serialize();
    return double_sha256(tx_data.data(), tx_data.size());
}

uint64_t Transaction::calculate_input_sum() const {
    // For coinbase transactions, input sum is 0
    if (is_coinbase()) {
        return 0;
    }
    
    // TODO: In a real implementation, we would look up the UTXOs
    // For now, return 0
    return 0;
}

uint64_t Transaction::calculate_output_sum() const {
    uint64_t sum = 0;
    for (const auto& output : outputs_) {
        sum += output.amount;
    }
    return sum;
}

// Static method to deserialize - add this to make it compatible with Block::deserialize
std::shared_ptr<Transaction> Transaction::from_bytes(const std::vector<uint8_t>& data) {
    auto tx = std::make_shared<Transaction>();
    if (tx->deserialize(data)) {
        return tx;
    }
    return nullptr;
}

} // namespace zion
