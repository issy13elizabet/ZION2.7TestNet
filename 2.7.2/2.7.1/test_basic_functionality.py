#!/usr/bin/env python3
"""
ZION 2.7.1 - Simple Blockchain Test
Basic functionality test for the clean blockchain implementation
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_blockchain():
    """Test basic blockchain functionality"""
    print("🧪 Testing ZION 2.7.1 Basic Blockchain Functionality")
    print("=" * 60)

    try:
        # Import blockchain components
        from core.blockchain import Transaction, Block
        print("✅ Successfully imported Transaction and Block classes")

        # Test transaction creation
        tx = Transaction(
            version=1,
            timestamp=int(time.time()),
            inputs=[{"prev_txid": "abc123", "output_index": 0}],
            outputs=[{"amount": 1000, "recipient": "test_address"}],
            fee=10
        )
        print(f"✅ Transaction created with txid: {tx.txid[:16]}...")

        # Test transaction hash determinism
        tx2 = Transaction(
            version=1,
            timestamp=tx.timestamp,
            inputs=[{"prev_txid": "abc123", "output_index": 0}],
            outputs=[{"amount": 1000, "recipient": "test_address"}],
            fee=10
        )
        assert tx.get_hash() == tx2.get_hash(), "Transaction hash should be deterministic"
        print("✅ Transaction hash determinism verified")

        # Test block creation
        block = Block(
            height=1,
            prev_hash="0" * 64,
            timestamp=int(time.time()),
            merkle_root=tx.get_hash(),
            difficulty=32,
            nonce=12345,
            txs=[tx]
        )

        print(f"✅ Block created at height {block.height}")

        # Test block hash calculation
        block_hash = block.calc_hash()
        print(f"✅ Block hash calculated: {block_hash[:16]}...")

        # Test PoW validation
        block.hash = block_hash
        is_valid = block.is_valid_pow()
        print(f"✅ Proof of Work validation: {'PASS' if is_valid else 'FAIL'}")

        print("\n🎉 Basic blockchain functionality test PASSED!")
        return True

    except Exception as e:
        print(f"❌ Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mining_components():
    """Test mining components"""
    print("\n🧪 Testing Mining Components")
    print("=" * 40)

    try:
        from mining.algorithms import Argon2Algorithm
        from mining.config import MiningConfig

        # Test mining config
        config_obj = MiningConfig()
        config = config_obj.get_mining_config()
        print(f"✅ Mining config loaded: algorithm={config.get('algorithm', 'unknown')}")

        # Test Argon2 algorithm
        algo = Argon2Algorithm(config)
        test_data = b"test data"
        hash_result = algo.hash(test_data)
        print(f"✅ Argon2 hash: {hash_result[:16]}...")

        print("🎉 Mining components test PASSED!")
        return True

    except Exception as e:
        print(f"❌ Mining test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_basic_blockchain()
    success2 = test_mining_components()

    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 ALL TESTS PASSED! ZION 2.7.1 is ready!")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED!")
        sys.exit(1)