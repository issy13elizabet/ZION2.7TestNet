#!/usr/bin/env python3
"""
ZION 2.7.1 - Test Suite
Comprehensive testing for the clean blockchain implementation
"""

import unittest
import time
from core.blockchain import Blockchain, Block, Transaction, create_blockchain
from mining.miner import ZionMiner, create_miner
from mining.algorithms import SHA256Algorithm, RandomXAlgorithm, GPUAlgorithm, AlgorithmFactory
from mining.config import MiningConfig, set_global_algorithm, get_global_algorithm, get_algorithm_info


class TestTransaction(unittest.TestCase):
    """Test transaction functionality"""

    def test_transaction_creation(self):
        """Test basic transaction creation"""
        tx = Transaction(
            version=1,
            timestamp=int(time.time()),
            inputs=[{"prev_txid": "abc123", "output_index": 0}],
            outputs=[{"amount": 1000, "recipient": "test_address"}],
            fee=10
        )

        self.assertEqual(tx.version, 1)
        self.assertEqual(tx.fee, 10)
        self.assertIsNotNone(tx.txid)
        self.assertEqual(len(tx.txid), 64)  # SHA256 hex length

    def test_transaction_hash_determinism(self):
        """Test that transaction hash is deterministic"""
        tx1 = Transaction(
            version=1,
            timestamp=1234567890,
            inputs=[{"prev_txid": "abc123", "output_index": 0}],
            outputs=[{"amount": 1000, "recipient": "test_address"}],
            fee=10
        )

        tx2 = Transaction(
            version=1,
            timestamp=1234567890,
            inputs=[{"prev_txid": "abc123", "output_index": 0}],
            outputs=[{"amount": 1000, "recipient": "test_address"}],
            fee=10
        )

        self.assertEqual(tx1.get_hash(), tx2.get_hash())
        self.assertEqual(tx1.txid, tx2.txid)


class TestBlock(unittest.TestCase):
    """Test block functionality"""

    def test_block_creation(self):
        """Test basic block creation"""
        tx = Transaction(
            version=1,
            timestamp=int(time.time()),
            inputs=[{"prev_txid": "0"*64, "output_index": 0xFFFFFFFF}],
            outputs=[{"amount": 5000000000, "recipient": "miner_address"}],
            fee=0
        )

        block = Block(
            height=1,
            prev_hash="0"*64,
            timestamp=int(time.time()),
            merkle_root=tx.get_hash(),
            difficulty=32,
            nonce=0,
            txs=[tx]
        )

        self.assertEqual(block.height, 1)
        self.assertEqual(block.difficulty, 32)
        self.assertEqual(len(block.txs), 1)

    def test_block_hash_calculation(self):
        """Test block hash calculation"""
        tx = Transaction(
            version=1,
            timestamp=1234567890,
            inputs=[{"prev_txid": "0"*64, "output_index": 0xFFFFFFFF}],
            outputs=[{"amount": 5000000000, "recipient": "miner_address"}],
            fee=0
        )

        block = Block(
            height=1,
            prev_hash="0"*64,
            timestamp=1234567890,
            merkle_root=tx.get_hash(),
            difficulty=32,
            nonce=12345,
            txs=[tx]
        )

        # Test SHA256 hash
        sha256_algo = SHA256Algorithm()
        hash1 = block.calc_hash(sha256_algo)
        hash2 = block.calc_hash(sha256_algo)

        self.assertEqual(hash1, hash2)  # Deterministic
        self.assertEqual(len(hash1), 64)  # SHA256 hex length

    def test_proof_of_work_validation(self):
        """Test PoW validation"""
        tx = Transaction(
            version=1,
            timestamp=int(time.time()),
            inputs=[{"prev_txid": "0"*64, "output_index": 0xFFFFFFFF}],
            outputs=[{"amount": 5000000000, "recipient": "miner_address"}],
            fee=0
        )

        block = Block(
            height=1,
            prev_hash="0"*64,
            timestamp=int(time.time()),
            merkle_root=tx.get_hash(),
            difficulty=1,  # Very easy difficulty
            nonce=0,
            txs=[tx]
        )

        # Mine until we find a valid nonce
        target = ((1 << 256) - 1) // block.difficulty
        while True:
            block.hash = block.calc_hash()
            hash_int = int(block.hash, 16)
            if hash_int <= target:
                break
            block.nonce += 1

        self.assertTrue(block.is_valid_pow())


class TestBlockchain(unittest.TestCase):
    """Test blockchain functionality"""

    def setUp(self):
        """Set up test blockchain"""
        self.blockchain = create_blockchain("testnet")

    def test_genesis_block(self):
        """Test genesis block creation"""
        self.assertEqual(len(self.blockchain.chain), 1)
        genesis = self.blockchain.get_latest_block()
        self.assertEqual(genesis.height, 0)
        self.assertEqual(genesis.prev_hash, "0"*64)
        self.assertTrue(genesis.is_valid_pow())

    def test_add_block(self):
        """Test adding blocks to blockchain"""
        # Mine a block
        miner = create_miner(self.blockchain, "test_miner")
        miner.start_mining(max_blocks=1)

        if miner.mining_thread:
            miner.mining_thread.join(timeout=30)

        # Check if block was added
        self.assertEqual(len(self.blockchain.chain), 2)
        latest = self.blockchain.get_latest_block()
        self.assertEqual(latest.height, 1)

    def test_block_validation(self):
        """Test block validation"""
        # Create invalid block (wrong height)
        tx = Transaction(
            version=1,
            timestamp=int(time.time()),
            inputs=[{"prev_txid": "0"*64, "output_index": 0xFFFFFFFF}],
            outputs=[{"amount": 5000000000, "recipient": "miner_address"}],
            fee=0
        )

        invalid_block = Block(
            height=999,  # Wrong height
            prev_hash=self.blockchain.get_latest_block().hash,
            timestamp=int(time.time()),
            merkle_root=tx.get_hash(),
            difficulty=self.blockchain.difficulty,
            nonce=0,
            txs=[tx]
        )

        self.assertFalse(self.blockchain.add_block(invalid_block))


class TestMiningAlgorithms(unittest.TestCase):
    """Test mining algorithms"""

    def test_sha256_algorithm(self):
        """Test SHA256 algorithm"""
        algo = SHA256Algorithm()
        test_data = b"test data"

        hash1 = algo.hash(test_data)
        hash2 = algo.hash(test_data)

        self.assertEqual(hash1, hash2)  # Deterministic
        self.assertEqual(len(hash1), 64)
        self.assertEqual(algo.get_name(), "SHA256")
        self.assertEqual(algo.get_target_adjustment(), 1.0)

    def test_randomx_algorithm(self):
        """Test RandomX algorithm"""
        algo = RandomXAlgorithm()
        test_data = b"test data"

        hash1 = algo.hash(test_data)
        hash2 = algo.hash(test_data)

        self.assertEqual(hash1, hash2)  # Deterministic
        self.assertEqual(len(hash1), 64)
        self.assertIn("RandomX", algo.get_name())

    def test_gpu_algorithm(self):
        """Test GPU algorithm"""
        algo = GPUAlgorithm()
        test_data = b"test data"

        hash1 = algo.hash(test_data)
        hash2 = algo.hash(test_data)

        self.assertEqual(hash1, hash2)  # Deterministic
        self.assertEqual(len(hash1), 64)
        self.assertIn("GPU", algo.get_name())

    def test_algorithm_factory(self):
        """Test algorithm factory"""
        sha256 = AlgorithmFactory.create_algorithm("sha256")
        self.assertIsInstance(sha256, SHA256Algorithm)

        randomx = AlgorithmFactory.create_algorithm("randomx")
        self.assertIsInstance(randomx, RandomXAlgorithm)

        gpu = AlgorithmFactory.create_algorithm("gpu")
        self.assertIsInstance(gpu, GPUAlgorithm)

        # Test auto selection
        best = AlgorithmFactory.auto_select_best()
        self.assertIsNotNone(best)

        # Test available algorithms
        available = AlgorithmFactory.get_available_algorithms()
        self.assertIn("sha256", available)
        self.assertIn("randomx", available)
        self.assertIn("gpu", available)


class TestMiningConfig(unittest.TestCase):
    """Test mining configuration"""

    def setUp(self):
        """Reset global config before each test"""
        global _global_config
        _global_config = MiningConfig()

    def test_config_algorithm_setting(self):
        """Test setting algorithms"""
        config = MiningConfig()

        config.set_algorithm("sha256")
        self.assertEqual(config.get_algorithm().get_name(), "SHA256")

        config.set_algorithm("randomx")
        self.assertIn("RandomX", config.get_algorithm().get_name())

        config.set_algorithm("auto")
        # Should select best available
        algo_name = config.get_algorithm().get_name()
        self.assertIn(algo_name, ["SHA256", "RandomX", "GPU-CUDA", "GPU-OpenCL", "GPU-Fallback"])

    def test_global_config_functions(self):
        """Test global config functions"""
        set_global_algorithm("sha256")
        algo = get_global_algorithm()
        self.assertEqual(algo.get_name(), "SHA256")

        info = get_algorithm_info()
        self.assertEqual(info["name"], "SHA256")
        self.assertEqual(info["configured"], "sha256")


class TestMiner(unittest.TestCase):
    """Test miner functionality"""

    def setUp(self):
        """Set up test miner"""
        self.blockchain = create_blockchain("testnet")
        self.miner = create_miner(self.blockchain, "test_miner_address")

    def test_miner_creation(self):
        """Test miner creation"""
        self.assertIsNotNone(self.miner)
        self.assertEqual(self.miner.miner_address, "test_miner_address")
        self.assertFalse(self.miner.is_mining)

    def test_miner_stats(self):
        """Test miner statistics"""
        stats = self.miner.get_stats()
        self.assertIn("is_mining", stats)
        self.assertIn("miner_address", stats)
        self.assertIn("blocks_mined", stats)
        self.assertEqual(stats["miner_address"], "test_miner_address")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)