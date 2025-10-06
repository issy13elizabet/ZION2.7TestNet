#!/usr/bin/env python3
"""
ZION 2.6.75 Integration Test Suite
Complete end-to-end testing of the entire ecosystem
"""

import asyncio
import json
import sys
import time
import pytest
import aiohttp
import websockets
from decimal import Decimal
from typing import Dict, List

# Import ZION modules
sys.path.insert(0, '/media/maitreya/ZION1/zion-2.6.75')
from zion.core.blockchain import ZionBlockchain
from zion.mining.randomx_engine import RandomXEngine
from zion.rpc.server import ZionRPCServer
from zion.wallet.wallet_core import ZionWallet
from zion.network.seed_node import ZionNetworking
from zion.pool.mining_pool import ZionMiningPool
from zion.bridge.rainbow_bridge import ZionRainbowBridge, ChainType


class ZionIntegrationTester:
    """Comprehensive integration test suite"""
    
    def __init__(self):
        self.test_results = {
            'blockchain_core': {'passed': 0, 'failed': 0, 'errors': []},
            'randomx_mining': {'passed': 0, 'failed': 0, 'errors': []},
            'rpc_server': {'passed': 0, 'failed': 0, 'errors': []},
            'wallet_system': {'passed': 0, 'failed': 0, 'errors': []},
            'p2p_networking': {'passed': 0, 'failed': 0, 'errors': []},
            'mining_pool': {'passed': 0, 'failed': 0, 'errors': []},
            'rainbow_bridge': {'passed': 0, 'failed': 0, 'errors': []},
            'performance': {'passed': 0, 'failed': 0, 'errors': []}
        }
        
        # Test configuration
        self.rpc_url = 'http://localhost:18089'
        self.pool_url = 'http://localhost:8080'
        self.bridge_url = 'http://localhost:9000'
        
        # Test data
        self.test_wallet_password = 'test_password_2675'
        self.test_bridge_key = 'test_bridge_key_abcdef123456'
    
    async def run_all_tests(self):
        """Run complete integration test suite"""
        print("🧪 Starting ZION 2.6.75 Integration Test Suite")
        print("=" * 60)
        
        start_time = time.time()
        
        # Core blockchain tests
        await self._test_blockchain_core()
        
        # RandomX mining tests
        await self._test_randomx_mining()
        
        # RPC server tests
        await self._test_rpc_server()
        
        # Wallet system tests
        await self._test_wallet_system()
        
        # P2P networking tests
        await self._test_p2p_networking()
        
        # Mining pool tests
        await self._test_mining_pool()
        
        # Rainbow bridge tests
        await self._test_rainbow_bridge()
        
        # Performance tests
        await self._test_performance()
        
        # Generate report
        total_time = time.time() - start_time
        self._generate_test_report(total_time)
    
    async def _test_blockchain_core(self):
        """Test blockchain core functionality"""
        print("\n🔗 Testing Blockchain Core...")
        
        try:
            # Test blockchain initialization
            blockchain = ZionBlockchain()
            self._assert_test('blockchain_core', blockchain is not None, "Blockchain initialization")
            
            # Test genesis block
            genesis = blockchain.get_block_by_height(0)
            self._assert_test('blockchain_core', genesis is not None, "Genesis block exists")
            self._assert_test('blockchain_core', genesis['height'] == 0, "Genesis block height")
            
            # Test block template creation
            template = blockchain.create_block_template()
            self._assert_test('blockchain_core', 'height' in template, "Block template creation")
            self._assert_test('blockchain_core', template['height'] > 0, "Block template height")
            
            # Test transaction validation
            valid_tx = {
                'from': 'ZIONTEST123456789ABCDEF',
                'to': 'ZIONTEST987654321FEDCBA',
                'amount': 1000000,
                'fee': 1000,
                'timestamp': int(time.time())
            }
            
            is_valid = blockchain.validate_transaction(valid_tx)
            self._assert_test('blockchain_core', is_valid, "Transaction validation")
            
            # Test difficulty adjustment
            difficulty = blockchain.get_current_difficulty()
            self._assert_test('blockchain_core', difficulty > 0, "Difficulty calculation")
            
            print("  ✅ Blockchain core tests completed")
            
        except Exception as e:
            self._record_error('blockchain_core', f"Blockchain core error: {e}")
    
    async def _test_randomx_mining(self):
        """Test RandomX mining engine"""
        print("\n⛏️  Testing RandomX Mining Engine...")
        
        try:
            # Test RandomX engine initialization
            engine = RandomXEngine()
            self._assert_test('randomx_mining', engine is not None, "RandomX engine creation")
            
            # Test engine initialization
            seed_key = b"ZION_2675_TEST_SEED_KEY"
            init_success = engine.init(seed_key)
            self._assert_test('randomx_mining', init_success, "RandomX engine initialization")
            
            # Test hash computation
            test_data = b"ZION_TEST_BLOCK_DATA"
            hash_result = engine.hash(test_data)
            self._assert_test('randomx_mining', hash_result is not None, "RandomX hash computation")
            self._assert_test('randomx_mining', len(hash_result) == 64, "Hash result length")
            
            # Test hash consistency
            hash_result2 = engine.hash(test_data)
            self._assert_test('randomx_mining', hash_result == hash_result2, "Hash consistency")
            
            # Test performance (should be > 10 H/s)
            start_time = time.time()
            hash_count = 100
            for i in range(hash_count):
                engine.hash(f"test_data_{i}".encode())
            
            elapsed = time.time() - start_time
            hashrate = hash_count / elapsed
            self._assert_test('randomx_mining', hashrate > 10, f"RandomX performance ({hashrate:.1f} H/s)")
            
            print(f"  ✅ RandomX mining tests completed (Performance: {hashrate:.1f} H/s)")
            
        except Exception as e:
            self._record_error('randomx_mining', f"RandomX mining error: {e}")
    
    async def _test_rpc_server(self):
        """Test RPC server functionality"""
        print("\n🔌 Testing RPC Server...")
        
        try:
            # Test health endpoint
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(f"{self.rpc_url}/health") as response:
                        health_ok = response.status == 200
                        self._assert_test('rpc_server', health_ok, "Health endpoint")
                        
                        if health_ok:
                            data = await response.json()
                            self._assert_test('rpc_server', data['status'] == 'healthy', "Health status")
                except aiohttp.ClientError:
                    self._record_error('rpc_server', "RPC server not accessible")
                    return
                
                # Test getinfo RPC call
                rpc_data = {
                    'jsonrpc': '2.0',
                    'method': 'getinfo',
                    'id': 1
                }
                
                async with session.post(f"{self.rpc_url}/json_rpc", json=rpc_data) as response:
                    info_ok = response.status == 200
                    self._assert_test('rpc_server', info_ok, "getinfo RPC call")
                    
                    if info_ok:
                        data = await response.json()
                        self._assert_test('rpc_server', 'result' in data, "getinfo response format")
                        self._assert_test('rpc_server', 'height' in data['result'], "getinfo height field")
                
                # Test getblocktemplate RPC call
                rpc_data = {
                    'jsonrpc': '2.0',
                    'method': 'getblocktemplate',
                    'id': 2
                }
                
                async with session.post(f"{self.rpc_url}/json_rpc", json=rpc_data) as response:
                    template_ok = response.status == 200
                    self._assert_test('rpc_server', template_ok, "getblocktemplate RPC call")
                    
                    if template_ok:
                        data = await response.json()
                        self._assert_test('rpc_server', 'result' in data, "getblocktemplate response format")
            
            print("  ✅ RPC server tests completed")
            
        except Exception as e:
            self._record_error('rpc_server', f"RPC server error: {e}")
    
    async def _test_wallet_system(self):
        """Test wallet system functionality"""
        print("\n💳 Testing Wallet System...")
        
        try:
            # Test wallet creation
            wallet = ZionWallet()
            create_result = wallet.create_wallet("test_wallet", self.test_wallet_password)
            self._assert_test('wallet_system', create_result['success'], "Wallet creation")
            
            if create_result['success']:
                test_address = create_result['address']
                self._assert_test('wallet_system', test_address.startswith('ZION'), "ZION address format")
                
                # Test wallet opening
                open_result = wallet.open_wallet("test_wallet", self.test_wallet_password)
                self._assert_test('wallet_system', open_result['success'], "Wallet opening")
                
                # Test balance retrieval
                balance_result = wallet.get_balance()
                self._assert_test('wallet_system', balance_result['success'], "Balance retrieval")
                
                # Test transaction creation
                tx_result = wallet.create_transaction(
                    "ZIONTEST987654321FEDCBA", 1000000, 1000
                )
                self._assert_test('wallet_system', tx_result['success'], "Transaction creation")
                
                # Test mnemonic restoration
                if 'mnemonic' in create_result:
                    restore_result = wallet.restore_from_mnemonic(
                        "restored_wallet", create_result['mnemonic'], self.test_wallet_password
                    )
                    self._assert_test('wallet_system', restore_result['success'], "Mnemonic restoration")
                
                # Test wallet locking
                lock_result = wallet.lock_wallet()
                self._assert_test('wallet_system', lock_result['success'], "Wallet locking")
            
            print("  ✅ Wallet system tests completed")
            
        except Exception as e:
            self._record_error('wallet_system', f"Wallet system error: {e}")
    
    async def _test_p2p_networking(self):
        """Test P2P networking functionality"""
        print("\n🌐 Testing P2P Networking...")
        
        try:
            # Test networking initialization
            network = ZionNetworking(listen_port=19091)  # Use different port to avoid conflicts
            self._assert_test('p2p_networking', network is not None, "Networking initialization")
            
            # Test node ID generation
            self._assert_test('p2p_networking', len(network.node_id) == 64, "Node ID format")
            
            # Test network statistics
            stats = network.get_network_stats()
            self._assert_test('p2p_networking', 'node_id' in stats, "Network statistics")
            self._assert_test('p2p_networking', stats['version'] == '2.6.75', "Network version")
            
            print("  ✅ P2P networking tests completed")
            
        except Exception as e:
            self._record_error('p2p_networking', f"P2P networking error: {e}")
    
    async def _test_mining_pool(self):
        """Test mining pool functionality"""
        print("\n🏊 Testing Mining Pool...")
        
        try:
            # Test pool initialization
            pool = ZionMiningPool(
                pool_address="ZIONPOOL123456789ABCDEF",
                stratum_port=4445,  # Use different port
                web_port=8081       # Use different port
            )
            self._assert_test('mining_pool', pool is not None, "Mining pool initialization")
            
            # Test pool configuration
            self._assert_test('mining_pool', pool.pool_address.startswith('ZION'), "Pool address format")
            self._assert_test('mining_pool', pool.pool_fee >= 0, "Pool fee configuration")
            
            # Test connection ID generation
            conn_id = pool._generate_connection_id()
            self._assert_test('mining_pool', len(conn_id) == 64, "Connection ID generation")
            
            # Test job ID generation
            job_id = pool._generate_job_id()
            self._assert_test('mining_pool', len(job_id) == 16, "Job ID generation")
            
            print("  ✅ Mining pool tests completed")
            
        except Exception as e:
            self._record_error('mining_pool', f"Mining pool error: {e}")
    
    async def _test_rainbow_bridge(self):
        """Test rainbow bridge functionality"""
        print("\n🌈 Testing Rainbow Bridge...")
        
        try:
            # Test bridge initialization
            bridge = ZionRainbowBridge(
                bridge_private_key=self.test_bridge_key,
                web_port=9002,      # Use different port
                websocket_port=9003 # Use different port
            )
            self._assert_test('rainbow_bridge', bridge is not None, "Bridge initialization")
            
            # Test chain configurations
            self._assert_test('rainbow_bridge', ChainType.ZION in bridge.chains, "ZION chain config")
            self._assert_test('rainbow_bridge', ChainType.SOLANA in bridge.chains, "Solana chain config")
            self._assert_test('rainbow_bridge', ChainType.STELLAR in bridge.chains, "Stellar chain config")
            
            # Test address validation
            zion_valid = await bridge._validate_address(ChainType.ZION, "ZIONTEST123456789ABCDEF")
            self._assert_test('rainbow_bridge', zion_valid, "ZION address validation")
            
            solana_valid = await bridge._validate_address(ChainType.SOLANA, "11111111111111111111111111111111")
            self._assert_test('rainbow_bridge', solana_valid, "Solana address validation")
            
            # Test bridge ID generation
            bridge_id = bridge._generate_bridge_id()
            self._assert_test('rainbow_bridge', len(bridge_id) == 64, "Bridge ID generation")
            
            print("  ✅ Rainbow bridge tests completed")
            
        except Exception as e:
            self._record_error('rainbow_bridge', f"Rainbow bridge error: {e}")
    
    async def _test_performance(self):
        """Test system performance benchmarks"""
        print("\n🚀 Testing Performance Benchmarks...")
        
        try:
            # Test RandomX performance
            engine = RandomXEngine()
            seed_key = b"PERFORMANCE_TEST_SEED"
            engine.init(seed_key)
            
            start_time = time.time()
            hash_count = 1000
            for i in range(hash_count):
                engine.hash(f"perf_test_{i}".encode())
            
            elapsed = time.time() - start_time
            hashrate = hash_count / elapsed
            
            self._assert_test('performance', hashrate > 50, f"RandomX performance target (50 H/s): {hashrate:.1f} H/s")
            
            # Test blockchain performance
            blockchain = ZionBlockchain()
            start_time = time.time()
            
            for i in range(100):
                template = blockchain.create_block_template()
            
            elapsed = time.time() - start_time
            templates_per_sec = 100 / elapsed
            
            self._assert_test('performance', templates_per_sec > 10, f"Block template generation (10/s): {templates_per_sec:.1f}/s")
            
            # Memory usage test
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            self._assert_test('performance', memory_mb < 500, f"Memory usage (<500MB): {memory_mb:.1f}MB")
            
            print("  ✅ Performance benchmarks completed")
            
        except Exception as e:
            self._record_error('performance', f"Performance test error: {e}")
    
    def _assert_test(self, category: str, condition: bool, description: str):
        """Assert test condition and record result"""
        if condition:
            self.test_results[category]['passed'] += 1
            print(f"    ✅ {description}")
        else:
            self.test_results[category]['failed'] += 1
            print(f"    ❌ {description}")
    
    def _record_error(self, category: str, error: str):
        """Record test error"""
        self.test_results[category]['failed'] += 1
        self.test_results[category]['errors'].append(error)
        print(f"    ❌ ERROR: {error}")
    
    def _generate_test_report(self, total_time: float):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("🧪 ZION 2.6.75 Integration Test Report")
        print("=" * 60)
        
        total_passed = sum(r['passed'] for r in self.test_results.values())
        total_failed = sum(r['failed'] for r in self.test_results.values())
        total_tests = total_passed + total_failed
        
        print(f"\n📊 Overall Results:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {total_passed} ({total_passed/total_tests*100:.1f}%)")
        print(f"   Failed: {total_failed} ({total_failed/total_tests*100:.1f}%)")
        print(f"   Execution Time: {total_time:.2f} seconds")
        
        print(f"\n📋 Category Breakdown:")
        for category, results in self.test_results.items():
            total_cat = results['passed'] + results['failed']
            if total_cat > 0:
                pass_rate = results['passed'] / total_cat * 100
                status = "✅" if results['failed'] == 0 else "⚠️" if pass_rate >= 50 else "❌"
                print(f"   {status} {category.replace('_', ' ').title()}: {results['passed']}/{total_cat} ({pass_rate:.1f}%)")
                
                if results['errors']:
                    for error in results['errors']:
                        print(f"      - {error}")
        
        # Overall assessment
        overall_pass_rate = total_passed / total_tests * 100 if total_tests > 0 else 0
        
        print(f"\n🎯 Assessment:")
        if overall_pass_rate >= 95:
            print("   🎉 EXCELLENT! ZION 2.6.75 is production ready.")
        elif overall_pass_rate >= 85:
            print("   ✅ GOOD! ZION 2.6.75 is ready with minor issues to address.")
        elif overall_pass_rate >= 70:
            print("   ⚠️  ACCEPTABLE! Several issues need attention before production.")
        else:
            print("   ❌ POOR! Major issues require resolution before deployment.")
        
        # Save report to file
        with open('/media/maitreya/ZION1/zion-2.6.75/INTEGRATION_TEST_REPORT.json', 'w') as f:
            json.dump({
                'timestamp': int(time.time()),
                'total_time': total_time,
                'overall_pass_rate': overall_pass_rate,
                'results': self.test_results
            }, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: INTEGRATION_TEST_REPORT.json")


async def main():
    """Run integration tests"""
    tester = ZionIntegrationTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    import sys
    asyncio.run(main())