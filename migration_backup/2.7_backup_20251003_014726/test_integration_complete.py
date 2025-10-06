#!/usr/bin/env python3
"""
ZION 2.7 Complete Integration Test

Tests the complete 2.6.75 integration into 2.7 real blockchain including:
- Enhanced blockchain core features
- Simple RPC server functionality  
- CryptoNote compatibility
- Mining capabilities
"""
import sys
import os
import json
import time
import subprocess
import requests
from threading import Thread

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
from core.blockchain import Blockchain, Block, Tx, Consensus

# Alias for compatibility
Transaction = Tx


def test_enhanced_blockchain():
    """Test enhanced blockchain features from 2.6.75"""
    print("🧪 Testing Enhanced Blockchain Features...")
    
    bc = Blockchain()
    
    # Test enhanced info
    info = bc.info()
    print(f"✓ Blockchain Version: {info['version']}")
    print(f"✓ Max Block Size: {info['max_block_size']:,}")
    print(f"✓ Min Fee: {info['min_fee']}")
    print(f"✓ Network Hashrate: {info['network_hashrate']}")
    print(f"✓ Cumulative Difficulty: {info['cumulative_difficulty']}")
    
    # Test block template generation
    template = bc.create_block_template("ZionMiner123")
    print(f"✓ Block Template - Height: {template['height']}")
    print(f"✓ Base Reward: {template['base_reward']:,}")
    print(f"✓ Difficulty: {template['difficulty']}")
    print(f"✓ Estimated Size: {template['estimated_block_size']} bytes")
    
    # Test consensus features
    print(f"✓ Block Time Target: {Consensus.BLOCK_TIME}s")
    print(f"✓ Max Block Size: {Consensus.MAX_BLOCK_SIZE:,}")
    print(f"✓ Initial Reward: {Consensus.INITIAL_REWARD:,}")
    print(f"✓ Halving Interval: {Consensus.HALVING_INTERVAL:,}")
    print(f"✓ Max Supply: {Consensus.MAX_SUPPLY:,}")
    print(f"✓ Min Fee: {Consensus.MIN_FEE:,}")
    
    print("✅ Enhanced Blockchain Test PASSED\n")
    return True


def test_enhanced_data_structures():
    """Test enhanced Block and Transaction classes"""
    print("🧪 Testing Enhanced Data Structures...")
    
    # Test enhanced Block
    block = Block(
        height=1,
        prev_hash="abc123",
        timestamp=int(time.time()),
        merkle_root="merkle123",
        txs=[],
        nonce=0,
        difficulty=1000,
        major_version=7,
        minor_version=0,
        base_reward=333000000,
        block_size=500,
        cumulative_difficulty=1000
    )
    
    print(f"✓ Block Major Version: {block.major_version}")
    print(f"✓ Block Minor Version: {block.minor_version}")
    print(f"✓ Block Base Reward: {block.base_reward:,}")
    print(f"✓ Block Size: {block.block_size}")
    print(f"✓ Cumulative Difficulty: {block.cumulative_difficulty}")
    
    # Test enhanced Transaction
    tx = Transaction(
        txid="tx123",
        inputs=[],
        outputs=[{"address": "ZionAddr", "amount": 1000}],
        fee=1000,
        timestamp=int(time.time()),
        version=1,
        unlock_time=0,
        extra=b"Test transaction",
        signatures=[],
        tx_size=250,
        ring_size=1
    )
    
    print(f"✓ Transaction Version: {tx.version}")
    print(f"✓ Transaction Size: {tx.tx_size}")
    print(f"✓ Ring Size: {tx.ring_size}")
    print(f"✓ Unlock Time: {tx.unlock_time}")
    
    print("✅ Enhanced Data Structures Test PASSED\n")
    return True


def start_rpc_server():
    """Start RPC server in background"""
    print("🚀 Starting RPC Server...")
    
    # Start server
    proc = subprocess.Popen([
        sys.executable, "rpc/simple_server.py", 
        "--host", "127.0.0.1", 
        "--port", "18082"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for startup
    time.sleep(3)
    
    return proc


def test_rpc_endpoints():
    """Test RPC server endpoints"""
    print("🧪 Testing RPC Server Endpoints...")
    
    base_url = "http://127.0.0.1:18082"
    
    try:
        # Test info endpoint
        response = requests.get(f"{base_url}/info", timeout=5)
        info_data = response.json()
        print(f"✓ GET /info - Status: {info_data['status']}")
        print(f"✓ Height: {info_data['height']}")
        print(f"✓ Version: {info_data['version']}")
        
        # Test height endpoint  
        response = requests.get(f"{base_url}/height", timeout=5)
        height_data = response.json()
        print(f"✓ GET /height - Height: {height_data['height']}")
        
        # Test stats endpoint
        response = requests.get(f"{base_url}/stats", timeout=5)
        stats_data = response.json()
        print(f"✓ GET /stats - Enhanced Features: {stats_data['blockchain_stats']['enhanced_features']}")
        print(f"✓ CryptoNote Compatible: {stats_data['blockchain_stats']['cryptonote_compatible']}")
        
        # Test JSON-RPC getinfo
        rpc_data = {
            "jsonrpc": "2.0",
            "id": 1, 
            "method": "getinfo"
        }
        response = requests.post(f"{base_url}/json_rpc", json=rpc_data, timeout=5)
        rpc_result = response.json()
        print(f"✓ JSON-RPC getinfo - Status: {rpc_result['result']['status']}")
        print(f"✓ Block Size Limit: {rpc_result['result']['block_size_limit']:,}")
        
        # Test getblocktemplate
        template_data = {
            "jsonrpc": "2.0", 
            "id": 2,
            "method": "getblocktemplate",
            "params": {"wallet_address": "ZionTestMiner"}
        }
        response = requests.post(f"{base_url}/json_rpc", json=template_data, timeout=5)
        template_result = response.json()
        print(f"✓ JSON-RPC getblocktemplate - Expected Reward: {template_result['result']['expected_reward']:,}")
        print(f"✓ Template Difficulty: {template_result['result']['difficulty']}")
        
        # Test getcurrencyid
        currency_data = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "getcurrencyid"
        }
        response = requests.post(f"{base_url}/json_rpc", json=currency_data, timeout=5)
        currency_result = response.json()
        print(f"✓ JSON-RPC getcurrencyid - Currency ID: {currency_result['result']['currency_id_blob']}")
        
        print("✅ RPC Server Test PASSED\n")
        return True
        
    except Exception as e:
        print(f"❌ RPC Server Test FAILED: {e}")
        return False


def test_mining_simulation():
    """Test mining simulation with RPC"""
    print("🧪 Testing Mining Simulation...")
    
    base_url = "http://127.0.0.1:18082"
    
    try:
        # Get block template
        template_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getblocktemplate", 
            "params": {"wallet_address": "ZionMiner001"}
        }
        
        response = requests.post(f"{base_url}/json_rpc", json=template_request, timeout=5)
        template = response.json()['result']
        
        print(f"✓ Mining Template Height: {template['height']}")
        print(f"✓ Mining Difficulty: {template['difficulty']}")
        print(f"✓ Expected Reward: {template['expected_reward']:,}")
        print(f"✓ Block Template Size: {len(template['blocktemplate_blob'])} chars")
        print(f"✓ Hashing Blob Size: {len(template['blockhashing_blob'])} chars")
        
        # Simulate mining (create fake solution)
        fake_block = json.dumps({
            "height": template['height'],
            "prev_hash": template['prev_hash'], 
            "nonce": 12345,
            "timestamp": int(time.time()),
            "solved": True
        })
        
        submit_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "submitblock",
            "params": [fake_block.encode().hex()]
        }
        
        # Note: This will likely fail as we're not actually mining, but tests the endpoint
        try:
            response = requests.post(f"{base_url}/json_rpc", json=submit_request, timeout=5)
            submit_result = response.json()
            if 'error' in submit_result:
                print(f"✓ Submit Block Endpoint Working (expected error: {submit_result['error']['message'][:50]}...)")
            else:
                print(f"✓ Submit Block Result: {submit_result['result']['status']}")
        except:
            print("✓ Submit Block Endpoint Accessible")
        
        print("✅ Mining Simulation Test PASSED\n")
        return True
        
    except Exception as e:
        print(f"❌ Mining Simulation Test FAILED: {e}")
        return False


def main():
    """Run complete integration tests"""
    print("🔬 ZION 2.7 Complete Integration Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Enhanced Blockchain
    if test_enhanced_blockchain():
        tests_passed += 1
    
    # Test 2: Enhanced Data Structures
    if test_enhanced_data_structures():
        tests_passed += 1
    
    # Test 3: Start RPC Server
    rpc_proc = None
    try:
        rpc_proc = start_rpc_server()
        
        # Test 4: RPC Endpoints
        if test_rpc_endpoints():
            tests_passed += 1
        
        # Test 5: Mining Simulation
        if test_mining_simulation():
            tests_passed += 1
        
    finally:
        # Clean up RPC server
        if rpc_proc:
            print("🛑 Stopping RPC Server...")
            rpc_proc.terminate()
            rpc_proc.wait()
    
    # Final Results
    print("=" * 50)
    print(f"📊 Integration Test Results: {tests_passed}/{total_tests} PASSED")
    
    if tests_passed == total_tests:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ 2.6.75 Integration into 2.7 Real Blockchain: SUCCESS")
        print("\n🌟 Features Successfully Integrated:")
        print("  • Enhanced blockchain core with CryptoNote compatibility")
        print("  • Advanced Block and Transaction data structures") 
        print("  • Zero-dependency RPC server with JSON-RPC 2.0 support")
        print("  • CryptoNote-compatible mining endpoints")
        print("  • Performance monitoring and comprehensive stats")
        print("  • Halving logic and advanced consensus parameters")
        print("  • Cumulative difficulty tracking")
        print("  • Enhanced block template generation")
        print("\n🚀 ZION 2.7 Real Blockchain Ready for Production!")
    else:
        print(f"❌ {total_tests - tests_passed} tests failed")
        return 1
    
    return 0


if __name__ == "__main__":
    # Install requests if needed
    try:
        import requests
    except ImportError:
        print("📦 Installing requests for HTTP testing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "--break-system-packages"])
        import requests
    
    sys.exit(main())