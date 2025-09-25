#pragma once

#include "zion.h"
#include <vector>
#include <memory>

namespace zion {

struct TxInput {
    Hash prev_tx_hash;
    uint32_t output_index;
    std::vector<uint8_t> signature;
    PublicKey public_key;
    
    TxInput() : output_index(0) {
        prev_tx_hash.fill(0);
        public_key.fill(0);
    }
};

struct TxOutput {
    uint64_t amount;
    PublicKey recipient;
    
    TxOutput() : amount(0) {
        recipient.fill(0);
    }
    
    TxOutput(uint64_t amt, const PublicKey& addr) : amount(amt), recipient(addr) {}
};

class Transaction {
public:
    Transaction();
    Transaction(const std::vector<TxInput>& inputs, const std::vector<TxOutput>& outputs);
    
    // Getters
    const std::vector<TxInput>& get_inputs() const { return inputs_; }
    const std::vector<TxOutput>& get_outputs() const { return outputs_; }
    Hash get_hash() const;
    uint64_t get_fee() const;
    
    // Transaction building
    void add_input(const TxInput& input);
    void add_output(const TxOutput& output);
    
    // Validation
    bool is_valid() const;
    bool verify_signatures() const;
    
    // Serialization
    std::vector<uint8_t> serialize() const;
    bool deserialize(const std::vector<uint8_t>& data);
    static std::shared_ptr<Transaction> from_bytes(const std::vector<uint8_t>& data);
    
    // Special transaction types
    static Transaction create_coinbase(const PublicKey& miner_address, uint64_t reward);
    bool is_coinbase() const;
    
private:
    uint32_t version_;
    std::vector<TxInput> inputs_;
    std::vector<TxOutput> outputs_;
    uint64_t lock_time_;
    
    Hash calculate_hash() const;
    uint64_t calculate_input_sum() const;
    uint64_t calculate_output_sum() const;
};

} // namespace zion
