#!/usr/bin/env python3
"""
ZION 2.7 Mining Integration Test Suite
Comprehensive testing of all mining features with blockchain integration
"""
import os
import sys
import time
import logging
import threading
import tempfile
import shutil
from typing import Dict, Any

# Setup path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Import ZION components
try:
    import core.blockchain as blockchain_mod
    import mining.randomx_engine as randomx_mod
    import mining.stratum_server as stratum_mod
    import mining.mining_stats as stats_mod
    import mining.mining_bridge as bridge_mod
    
    Blockchain = blockchain_mod.Blockchain
    Block = blockchain_mod.Block
    Consensus = blockchain_mod.Consensus
    RandomXEngine = randomx_mod.RandomXEngine
    MiningThreadManager = randomx_mod.MiningThreadManager
    StratumPoolServer = stratum_mod.StratumPoolServer
    MiningJob = stratum_mod.MiningJob
    MiningStatsCollector = stats_mod.MiningStatsCollector
    MiningIntegrationBridge = bridge_mod.MiningIntegrationBridge
    create_mining_system = bridge_mod.create_mining_system
except ImportError as e:
    logger.error(f"Failed to import ZION modules: {e}")
    sys.exit(1)

class MiningTestSuite:
    """Comprehensive mining integration test suite"""
    
    def __init__(self):
        self.test_results = {}
        self.temp_dir = None
        self.miner_address = "Z3BDEEC2A0AE0F5D81B034308F99ECD8990D9B8B01BD9C7E7429392CA31861C6220DA3B30D74E809FA0A1FE069F1"
        
    def setup(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp(prefix="zion_mining_test_")
        logger.info(f"🧪 Test environment created: {self.temp_dir}")
        
    def teardown(self):
        """Cleanup test environment"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info("🧹 Test environment cleaned up")
            
    def test_randomx_engine(self) -> bool:
        """Test RandomX engine functionality"""
        logger.info("🔬 Testing RandomX Engine...")
        
        try:
            # Test basic engine initialization
            engine = RandomXEngine(enable_optimizations=True)
            seed = b'ZION_2_7_TEST_SEED'
            
            if not engine.init(seed):
                logger.error("❌ Failed to initialize RandomX engine")
                return False
                
            # Test hash calculation
            test_data = b'ZION_TEST_DATA'
            hash1 = engine.hash(test_data)
            hash2 = engine.hash(test_data)
            
            if hash1 != hash2:
                logger.error("❌ RandomX hash determinism test failed")
                return False
                
            # Test performance statistics
            stats = engine.get_performance_stats()
            if stats['total_hashes'] < 2:
                logger.error("❌ Performance statistics not updating")
                return False
                
            # Test with different data
            different_data = b'DIFFERENT_DATA'
            hash3 = engine.hash(different_data)
            
            if hash3 == hash1:
                logger.error("❌ RandomX produces same hash for different inputs")
                return False
                
            engine.cleanup()
            logger.info("✅ RandomX engine test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ RandomX engine test failed: {e}")
            return False
            
    def test_mining_thread_manager(self) -> bool:
        """Test multi-threaded mining manager"""
        logger.info("🔬 Testing Mining Thread Manager...")
        
        try:
            manager = MiningThreadManager(num_threads=2)
            seed = b'ZION_THREAD_TEST_SEED'
            
            # Initialize engines
            if not manager.initialize_engines(seed):
                logger.error("❌ Failed to initialize mining threads")
                return False
                
            # Test short mining operation
            results = manager.start_mining_test(duration=2.0)
            
            # Validate results
            if results['thread_count'] != 2:
                logger.error(f"❌ Expected 2 threads, got {results['thread_count']}")
                return False
                
            if results['total_hashes'] == 0:
                logger.error("❌ No hashes calculated in mining test")
                return False
                
            if results['elapsed_time'] < 1.5:
                logger.error(f"❌ Mining test too short: {results['elapsed_time']}")
                return False
                
            manager.cleanup()
            logger.info("✅ Mining thread manager test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Mining thread manager test failed: {e}")
            return False
            
    def test_stratum_server(self) -> bool:
        """Test Stratum protocol server"""
        logger.info("🔬 Testing Stratum Protocol Server...")
        
        try:
            server = StratumPoolServer(host="127.0.0.1", port=33334)  # Use different port
            
            # Test job generation
            job = server.generate_mining_job()
            if not job.job_id:
                logger.error("❌ Failed to generate mining job")
                return False
                
            # Test share validation
            status = server.validate_share(
                job.job_id, 
                "12345678", 
                "0000abcd" + "0" * 56,  # Valid looking hash
                "test_worker"
            )
            
            if status.name not in ['ACCEPTED', 'REJECTED']:
                logger.error(f"❌ Invalid share status: {status}")
                return False
                
            # Test pool statistics
            stats = server.get_pool_statistics()
            if 'pool_name' not in stats:
                logger.error("❌ Pool statistics missing required fields")
                return False
                
            logger.info("✅ Stratum server test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Stratum server test failed: {e}")
            return False
            
    def test_mining_statistics(self) -> bool:
        """Test mining statistics collection"""
        logger.info("🔬 Testing Mining Statistics...")
        
        try:
            collector = MiningStatsCollector(collection_interval=0.5)
            collector.start_collection()
            
            # Register test threads
            stats1 = collector.register_mining_thread(0, difficulty=1000.0)
            stats2 = collector.register_mining_thread(1, difficulty=2000.0)
            
            # Simulate mining activity
            for _ in range(10):
                collector.update_thread_hashes(0, 5)
                collector.update_thread_hashes(1, 3)
                
                # Simulate some shares
                collector.add_share_result(0, True)  # Accepted
                collector.add_share_result(1, False)  # Rejected
                
                time.sleep(0.1)
                
            # Wait for collection
            time.sleep(1.0)
            
            # Get and validate statistics
            current_stats = collector.get_current_stats()
            
            if len(current_stats['thread_stats']) != 2:
                logger.error("❌ Incorrect number of threads in statistics")
                return False
                
            summary = current_stats['summary']
            if summary['total_hashes'] < 50:
                logger.error(f"❌ Too few hashes recorded: {summary['total_hashes']}")
                return False
                
            if summary['accepted_shares'] == 0:
                logger.error("❌ No accepted shares recorded")
                return False
                
            # Test historical data
            historical = collector.get_historical_data(duration_seconds=10)
            if not historical['hashrates']:
                logger.error("❌ No historical hashrate data")
                return False
                
            collector.stop_collection()
            logger.info("✅ Mining statistics test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Mining statistics test failed: {e}")
            return False
            
    def test_blockchain_integration(self) -> bool:
        """Test mining integration with blockchain"""
        logger.info("🔬 Testing Blockchain Integration...")
        
        try:
            # Create test blockchain
            blockchain = Blockchain()
            
            # Create mining bridge
            bridge = MiningIntegrationBridge(blockchain)
            if not bridge.initialize_mining(self.miner_address, num_threads=2):
                logger.error("❌ Failed to initialize mining bridge")
                return False
                
            # Test block template generation
            template = bridge.generate_block_template()
            if not template:
                logger.error("❌ Failed to generate block template")
                return False
                
            # Validate template structure
            required_fields = ['height', 'prev_hash', 'difficulty', 'target', 'block_reward']
            for field in required_fields:
                if field not in template:
                    logger.error(f"❌ Missing field in block template: {field}")
                    return False
                    
            # Test current difficulty calculation
            difficulty = bridge.get_current_difficulty()
            if difficulty <= 0:
                logger.error(f"❌ Invalid difficulty: {difficulty}")
                return False
                
            # Test share validation
            is_valid = bridge._validate_share_with_blockchain(
                template['job_id'] if 'job_id' in template else "test_job",
                "12345678",
                "0000abcd" + "0" * 56
            )
            
            # Should be valid or invalid, but function should work
            if not isinstance(is_valid, bool):
                logger.error("❌ Share validation returned non-boolean")
                return False
                
            # Test mining statistics integration
            stats = bridge.get_mining_statistics()
            if 'blockchain_info' not in stats:
                logger.error("❌ Missing blockchain info in mining stats")
                return False
                
            bridge.cleanup()
            logger.info("✅ Blockchain integration test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Blockchain integration test failed: {e}")
            return False
            
    def test_complete_mining_system(self) -> bool:
        """Test complete integrated mining system"""
        logger.info("🔬 Testing Complete Mining System...")
        
        try:
            # Create complete system
            blockchain = Blockchain()
            mining_system = create_mining_system(blockchain, self.miner_address, num_threads=2)
            
            # Track blocks found
            blocks_found = []
            
            def on_block_found(block):
                blocks_found.append(block)
                logger.info(f"🎉 Test block found: Height {block.height}")
                
            mining_system.add_block_found_callback(on_block_found)
            
            # Run mining for short period
            results = mining_system.start_mining(duration=3.0)
            
            # Validate results
            if results['thread_count'] != 2:
                logger.error(f"❌ Expected 2 threads, got {results['thread_count']}")
                return False
                
            if results['total_hashes'] == 0:
                logger.error("❌ No hashes calculated")
                return False
                
            # Get final statistics
            final_stats = mining_system.get_mining_statistics()
            
            # Validate statistics structure
            required_sections = ['pool_stats', 'system_stats', 'summary', 'blockchain_info']
            for section in required_sections:
                if section not in final_stats:
                    logger.error(f"❌ Missing statistics section: {section}")
                    return False
                    
            # Cleanup
            mining_system.cleanup()
            
            logger.info("✅ Complete mining system test passed")
            logger.info(f"   Blocks found during test: {len(blocks_found)}")
            logger.info(f"   Total hashes: {results['total_hashes']}")
            logger.info(f"   Average hashrate: {results['average_hashrate']:.2f} H/s")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Complete mining system test failed: {e}")
            return False
            
    def test_performance_benchmarks(self) -> bool:
        """Test mining performance benchmarks"""
        logger.info("🔬 Testing Performance Benchmarks...")
        
        try:
            # Single engine benchmark
            engine = RandomXEngine(enable_optimizations=True)
            engine.init(b'BENCHMARK_SEED')
            
            start_time = time.time()
            hash_count = 0
            benchmark_duration = 2.0
            
            while time.time() - start_time < benchmark_duration:
                test_data = f"benchmark_data_{hash_count}".encode()
                _ = engine.hash(test_data)
                hash_count += 1
                
            single_hashrate = hash_count / benchmark_duration
            
            # Multi-threaded benchmark
            manager = MiningThreadManager(num_threads=2)
            manager.initialize_engines(b'MULTI_BENCHMARK_SEED')
            
            multi_results = manager.start_mining_test(duration=2.0)
            multi_hashrate = multi_results['average_hashrate']
            
            # Performance validation
            if single_hashrate <= 0:
                logger.error("❌ Single thread hashrate is zero")
                return False
                
            if multi_hashrate <= single_hashrate:
                logger.warning("⚠️ Multi-threading not improving performance")
                # This is a warning, not a failure
                
            # Cleanup
            engine.cleanup() 
            manager.cleanup()
            
            logger.info("✅ Performance benchmark test passed")
            logger.info(f"   Single thread: {single_hashrate:.2f} H/s")
            logger.info(f"   Multi thread: {multi_hashrate:.2f} H/s")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Performance benchmark test failed: {e}")
            return False
            
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all mining integration tests"""
        logger.info("🧪 Starting ZION 2.7 Mining Integration Test Suite")
        logger.info("=" * 60)
        
        self.setup()
        
        tests = [
            ("RandomX Engine", self.test_randomx_engine),
            ("Mining Thread Manager", self.test_mining_thread_manager), 
            ("Stratum Server", self.test_stratum_server),
            ("Mining Statistics", self.test_mining_statistics),
            ("Blockchain Integration", self.test_blockchain_integration),
            ("Complete Mining System", self.test_complete_mining_system),
            ("Performance Benchmarks", self.test_performance_benchmarks),
        ]
        
        results = {}
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                logger.info(f"\n📋 Running: {test_name}")
                result = test_func()
                results[test_name] = result
                
                if result:
                    passed += 1
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
                    
            except Exception as e:
                logger.error(f"💥 {test_name}: ERROR - {e}")
                results[test_name] = False
                
        self.teardown()
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("🏁 Test Suite Results Summary")
        logger.info("=" * 60)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"  {test_name:.<30} {status}")
            
        logger.info("-" * 60)
        logger.info(f"  Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            logger.info("🎉 All tests PASSED! Mining integration is ready!")
        else:
            logger.warning(f"⚠️  {total-passed} tests FAILED. Review results above.")
            
        return results

def main():
    """Run mining integration tests"""
    test_suite = MiningTestSuite()
    results = test_suite.run_all_tests()
    
    # Exit with appropriate code
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)

if __name__ == '__main__':
    main()