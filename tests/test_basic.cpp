#include "zion.h"
#include "transaction.h"
#include "block.h"
#include "randomx_wrapper.h"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "Spouštím základní testy ZION..." << std::endl;
    
    // Test transaction creation
    {
        std::cout << "Test: Vytvoření coinbase transakce..." << std::endl;
        
        zion::PublicKey miner_addr;
        miner_addr.fill(0x42);
        
        auto tx = zion::Transaction::create_coinbase(miner_addr, zion::ZION_INITIAL_REWARD);
        
        assert(tx.is_coinbase());
        assert(tx.get_inputs().size() == 1);
        assert(tx.get_outputs().size() == 1);
        assert(tx.is_valid());
        
        std::cout << "✓ Coinbase transakce test prošel" << std::endl;
    }
    
    // Test block creation
    {
        std::cout << "Test: Vytvoření bloku..." << std::endl;
        
        zion::PublicKey miner_addr;
        miner_addr.fill(0x42);
        
        auto coinbase = zion::Transaction::create_coinbase(miner_addr, zion::ZION_INITIAL_REWARD);
        
        zion::Hash prev_hash;
        prev_hash.fill(0);
        
        // Vytvoříme blok s výškou 0 (genesis block)
        zion::Block block(0, prev_hash);
        
        // Přidáme coinbase transakci jako shared_ptr
        auto coinbase_ptr = std::make_shared<zion::Transaction>(coinbase);
        block.addTransaction(coinbase_ptr);
        
        assert(block.getTransactions().size() == 1);
        assert(block.getTransactions()[0]->is_coinbase());
        
        std::cout << "✓ Block test prošel" << std::endl;
    }
    
    // Test key generation
    {
        std::cout << "Test: Generování klíčů..." << std::endl;
        
        auto [private_key, public_key] = zion::generate_keypair();
        
        // Check that keys are not empty
        bool private_not_empty = false;
        bool public_not_empty = false;
        
        for (const auto& byte : private_key) {
            if (byte != 0) {
                private_not_empty = true;
                break;
            }
        }
        
        for (const auto& byte : public_key) {
            if (byte != 0) {
                public_not_empty = true;
                break;
            }
        }
        
        assert(private_not_empty);
        assert(public_not_empty);
        
        std::cout << "✓ Generování klíčů test prošel" << std::endl;
    }
    
    std::cout << "✅ Všechny základní testy prošly úspěšně!" << std::endl;
    return 0;
}
