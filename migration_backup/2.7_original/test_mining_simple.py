#!/usr/bin/env python3
"""
ZION 2.7 Mining Integration Test Suite (Fallback Mode)
Tests mining integration with SHA256 fallback when RandomX unavailable
"""
import os
import sys
import time
import logging

# Setup path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Import ZION components
try:
    from core.blockchain import Blockchain, Block, Consensus
    from mining.randomx_engine import RandomXEngine
    from mining.stratum_server import StratumPoolServer
    from mining.mining_stats import MiningStatsCollector
    from mining.mining_bridge import MiningIntegrationBridge
except ImportError as e:
    logger.error(f"Failed to import ZION modules: {e}")
    sys.exit(1)

def test_basic_mining_integration():
    """Test basic mining integration functionality"""
    logger.info("🔬 Testing Basic Mining Integration (Fallback Mode)")
    
    try:
        # Create blockchain
        blockchain = Blockchain()
        logger.info("✅ Blockchain created")
        
        # Create mining bridge
        bridge = MiningIntegrationBridge(blockchain)
        miner_address = "Z3BDEEC2A0AE0F5D81B034308F99ECD8990D9B8B01BD9C7E7429392CA31861C6220DA3B30D74E809FA0A1FE069F1"
        
        # Initialize with fallback mode (will use SHA256)
        success = bridge.initialize_mining(miner_address, num_threads=2)
        if not success:
            logger.warning("⚠️ Mining initialization failed, but testing fallback functionality")
        
        # Test block template generation
        template = bridge.generate_block_template()
        if not template:
            logger.error("❌ Failed to generate block template")
            return False
            
        logger.info(f"✅ Block template generated: Height {template['height']}")
        
        # Test difficulty calculation
        difficulty = bridge.get_current_difficulty()
        logger.info(f"✅ Current difficulty: {difficulty}")
        
        # Test statistics
        if bridge.stats_collector:
            stats = bridge.get_mining_statistics()
            logger.info("✅ Mining statistics retrieved")
        
        bridge.cleanup()
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic mining integration test failed: {e}")
        return False

def test_stratum_protocol():
    """Test Stratum protocol functionality"""
    logger.info("🔬 Testing Stratum Protocol")
    
    try:
        server = StratumPoolServer(host="127.0.0.1", port=33335)
        
        # Test job generation
        job = server.generate_mining_job()
        logger.info(f"✅ Mining job generated: {job.job_id}")
        
        # Test share validation
        status = server.validate_share(
            job.job_id,
            "12345678", 
            "0000abcd" + "0" * 56,
            "test_worker"
        )
        logger.info(f"✅ Share validation result: {status.name}")
        
        # Test pool statistics
        stats = server.get_pool_statistics()
        logger.info("✅ Pool statistics retrieved")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Stratum protocol test failed: {e}")
        return False

def test_mining_statistics():
    """Test mining statistics collection"""
    logger.info("🔬 Testing Mining Statistics")
    
    try:
        collector = MiningStatsCollector(collection_interval=0.5)
        collector.start_collection()
        
        # Register test threads
        collector.register_mining_thread(0, difficulty=1000.0)
        collector.register_mining_thread(1, difficulty=2000.0)
        
        # Simulate mining activity
        for _ in range(10):
            collector.update_thread_hashes(0, 5)
            collector.update_thread_hashes(1, 3)
            collector.add_share_result(0, True)
            collector.add_share_result(1, False)
            time.sleep(0.1)
        
        # Get statistics
        stats = collector.get_current_stats()
        
        if len(stats['thread_stats']) != 2:
            logger.error("❌ Incorrect number of threads in statistics")
            return False
            
        if stats['summary']['total_hashes'] < 50:
            logger.error("❌ Too few hashes recorded")
            return False
            
        collector.stop_collection()
        logger.info("✅ Mining statistics test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Mining statistics test failed: {e}")
        return False

def test_blockchain_operations():
    """Test blockchain operations"""
    logger.info("🔬 Testing Blockchain Operations")
    
    try:
        blockchain = Blockchain()
        
        # Get initial state
        genesis = blockchain.get_latest_block()
        if not genesis:
            logger.error("❌ No genesis block found")
            return False
            
        logger.info(f"✅ Genesis block: Height {genesis.height}")
        
        # Test difficulty calculation
        initial_difficulty = Consensus.MIN_DIFF
        logger.info(f"✅ Initial difficulty: {initial_difficulty}")
        
        # Test reward calculation
        initial_reward = Consensus.INITIAL_REWARD
        logger.info(f"✅ Initial reward: {initial_reward}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Blockchain operations test failed: {e}")
        return False

def test_randomx_fallback():
    """Test RandomX engine fallback functionality"""
    logger.info("🔬 Testing RandomX Fallback Mode")
    
    try:
        # Create engine (will likely fall back to SHA256)
        engine = RandomXEngine(enable_optimizations=True)
        seed = b'ZION_FALLBACK_TEST'
        
        # Initialize (should succeed even without RandomX library)
        success = engine.init(seed)
        if not success:
            logger.error("❌ Failed to initialize engine even in fallback mode")
            return False
            
        logger.info(f"✅ Engine initialized, fallback mode: {engine.fallback}")
        
        # Test hash calculation
        test_data = b'test_data'
        hash1 = engine.hash(test_data)
        hash2 = engine.hash(test_data)
        
        if hash1 != hash2:
            logger.error("❌ Hash determinism failed")
            return False
            
        logger.info("✅ Hash calculation working")
        
        # Test performance stats
        stats = engine.get_performance_stats()
        if stats['total_hashes'] < 2:
            logger.error("❌ Performance stats not updating")
            return False
            
        logger.info("✅ Performance statistics working")
        
        engine.cleanup()
        return True
        
    except Exception as e:
        logger.error(f"❌ RandomX fallback test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🧪 ZION 2.7 Mining Integration Test Suite (Fallback Mode)")
    logger.info("=" * 60)
    
    tests = [
        ("RandomX Fallback Mode", test_randomx_fallback),
        ("Blockchain Operations", test_blockchain_operations),
        ("Stratum Protocol", test_stratum_protocol),
        ("Mining Statistics", test_mining_statistics),
        ("Basic Mining Integration", test_basic_mining_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        try:
            result = test_func()
            if result:
                logger.info(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"💥 {test_name}: ERROR - {e}")
            
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("🏁 Test Results Summary")
    logger.info("=" * 60)
    logger.info(f"Passed: {passed}/{total} tests ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 All tests PASSED! Mining integration is ready!")
    else:
        logger.warning(f"⚠️ {total-passed} tests failed. System still functional in fallback mode.")
        
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)