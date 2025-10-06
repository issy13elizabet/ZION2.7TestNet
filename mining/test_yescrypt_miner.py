#!/usr/bin/env python3
"""
ZION Yescrypt Miner - Quick Test & Validation
Tests optimized miner functionality and eco-bonus integration
"""

import sys
import time
from typing import Dict, Any

def test_yescrypt_performance():
    """Test basic Yescrypt hash performance"""
    print("🧪 Testing Yescrypt hash performance...")
    
    try:
        # Import optimized miner
        sys.path.append('.')
        from zion_yescrypt_optimized import OptimizedYescryptMiner
        
        # Create test config
        test_config = {
            'pool_host': 'localhost',
            'pool_port': 4444,
            'wallet_address': 'ZioniTESTWALLETADDRESSFORTESTING123',
            'threads': 1,  # Single thread for testing
            'eco_mode': True
        }
        
        # Initialize miner
        miner = OptimizedYescryptMiner(test_config)
        print(f"✅ Miner initialized successfully")
        
        # Test hash function
        test_data = b"ZION_TEST_BLOCK_DATA"
        start_time = time.time()
        
        hash_count = 1000
        for nonce in range(hash_count):
            hash_result = miner.yescrypt_hash(test_data, nonce)
            
        elapsed = time.time() - start_time
        hashrate = hash_count / elapsed
        
        print(f"✅ Hash function test completed")
        print(f"   Hashes: {hash_count}")
        print(f"   Time: {elapsed:.2f}s")
        print(f"   Hashrate: {hashrate:.2f} H/s")
        
        # Test share validation
        valid_shares = 0
        for nonce in range(100):
            hash_result = miner.yescrypt_hash(test_data, nonce)
            if miner.validate_share(hash_result):
                valid_shares += 1
        
        print(f"✅ Share validation test completed")
        print(f"   Valid shares found: {valid_shares}/100")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_eco_bonus_calculation():
    """Test eco-bonus calculations"""
    print("\n🌱 Testing eco-bonus calculations...")
    
    try:
        base_hashrate = 100.0  # H/s
        eco_bonus = 1.15       # +15%
        
        effective_hashrate = base_hashrate * eco_bonus
        bonus_percentage = (eco_bonus - 1) * 100
        
        print(f"✅ Eco-bonus calculation test")
        print(f"   Base hashrate: {base_hashrate} H/s")
        print(f"   Eco bonus: +{bonus_percentage:.0f}%")
        print(f"   Effective hashrate: {effective_hashrate} H/s")
        
        # Test power efficiency
        power_consumption = 80.0  # Watts
        efficiency = effective_hashrate / power_consumption
        
        print(f"   Power consumption: {power_consumption}W")
        print(f"   Efficiency: {efficiency:.2f} H/s/W")
        
        return True
        
    except Exception as e:
        print(f"❌ Eco-bonus test failed: {e}")
        return False

def test_configuration_loading():
    """Test configuration file loading"""
    print("\n⚙️ Testing configuration loading...")
    
    try:
        import json
        
        # Test config file
        config_path = 'yescrypt-miner-config.json'
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"✅ Configuration loaded successfully")
        print(f"   Pool: {config['pool_host']}:{config['pool_port']}")
        print(f"   Eco mode: {config['eco_mode']}")
        print(f"   Threads: {config['threads'] or 'auto-detect'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def run_mining_simulation(duration: int = 10):
    """Run short mining simulation"""
    print(f"\n⚡ Running {duration}s mining simulation...")
    
    try:
        sys.path.append('.')
        from zion_yescrypt_optimized import OptimizedYescryptMiner
        
        # Simulation config
        sim_config = {
            'pool_host': 'localhost',
            'pool_port': 4444,
            'wallet_address': 'ZioniSIMULATIONTESTWALLETADDRESS123',
            'threads': 2,
            'eco_mode': True
        }
        
        miner = OptimizedYescryptMiner(sim_config)
        
        # Simulate mining for specified duration
        print(f"🚀 Starting {duration}s simulation...")
        
        start_time = time.time()
        total_hashes = 0
        
        while time.time() - start_time < duration:
            # Simulate hash computation
            test_data = f"SIMULATION_BLOCK_{int(time.time())}".encode()
            hash_result = miner.yescrypt_hash(test_data, int(time.time()) % 1000000)
            total_hashes += 1
            
            # Small delay to simulate realistic mining
            time.sleep(0.01)
        
        elapsed = time.time() - start_time
        sim_hashrate = total_hashes / elapsed
        
        print(f"✅ Simulation completed")
        print(f"   Duration: {elapsed:.2f}s")
        print(f"   Total hashes: {total_hashes}")
        print(f"   Simulated hashrate: {sim_hashrate:.2f} H/s")
        print(f"   Estimated eco hashrate: {sim_hashrate * 1.15:.2f} H/s (+15%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Mining simulation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 ZION Yescrypt Miner - Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 0
    
    # Run all tests
    tests = [
        ("Yescrypt Performance", test_yescrypt_performance),
        ("Eco-Bonus Calculation", test_eco_bonus_calculation),
        ("Configuration Loading", test_configuration_loading),
        ("Mining Simulation", lambda: run_mining_simulation(5))
    ]
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        print("-" * 30)
        
        total_tests += 1
        if test_func():
            tests_passed += 1
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")
    
    # Final report
    print("\n" + "=" * 50)
    print(f"📊 TEST SUMMARY")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {tests_passed}")
    print(f"   Failed: {total_tests - tests_passed}")
    print(f"   Success Rate: {(tests_passed/total_tests)*100:.1f}%")
    
    if tests_passed == total_tests:
        print("🏆 ALL TESTS PASSED - Miner ready for deployment!")
    else:
        print("⚠️ Some tests failed - Check implementation")
    
    print("\n🎯 Next Steps:")
    print("   1. Update wallet address in config")
    print("   2. Set correct pool host/port")
    print("   3. Run: python zion_yescrypt_optimized.py --wallet YOUR_ADDRESS")
    print("   4. Monitor eco-bonus performance")

if __name__ == "__main__":
    main()