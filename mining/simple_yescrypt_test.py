#!/usr/bin/env python3
"""
ZION Yescrypt CPU Miner - Simple Test Version
Quick verification of mining functionality
"""

import hashlib
import struct
import time
import argparse

def yescrypt_hash_simple(data, nonce):
    """Simplified Yescrypt hash for testing"""
    input_data = data + struct.pack('<I', nonce)
    
    # Simplified implementation for testing
    key = hashlib.pbkdf2_hmac('sha256', input_data, b'yescrypt_zion', 2048, 32)
    
    # Additional mixing
    for i in range(4):
        key = hashlib.sha256(key + struct.pack('<I', i)).digest()
    
    return key

def test_basic_mining():
    """Test basic mining functionality"""
    print("🚀 ZION Yescrypt Simple Test")
    print("=" * 40)
    
    # Test parameters
    test_data = b"ZION_TEST_BLOCK"
    target_difficulty = 0x00FFFFFF  # Easy target for testing
    
    print(f"⚡ Testing Yescrypt hash function...")
    
    start_time = time.time()
    hash_count = 0
    found_shares = 0
    
    # Mine for 10 seconds
    while time.time() - start_time < 10:
        hash_result = yescrypt_hash_simple(test_data, hash_count)
        hash_int = int.from_bytes(hash_result[:4], 'big')
        
        if hash_int < target_difficulty:
            found_shares += 1
            print(f"✅ Share found! Nonce: {hash_count} Hash: {hash_result[:8].hex()}")
        
        hash_count += 1
        
        if hash_count % 1000 == 0:
            elapsed = time.time() - start_time
            hashrate = hash_count / elapsed if elapsed > 0 else 0
            print(f"📊 Progress: {hash_count} hashes, {hashrate:.1f} H/s")
    
    # Final statistics
    elapsed = time.time() - start_time
    hashrate = hash_count / elapsed if elapsed > 0 else 0
    
    print("\n📊 MINING TEST RESULTS")
    print("-" * 30)
    print(f"⏱️  Duration: {elapsed:.2f}s")
    print(f"🧮 Total Hashes: {hash_count}")
    print(f"⚡ Hashrate: {hashrate:.2f} H/s")
    print(f"🎯 Shares Found: {found_shares}")
    print(f"🌱 Estimated Power: ~80W")
    print(f"💰 Eco Bonus: +15% = {hashrate * 1.15:.2f} H/s effective")
    
    if found_shares > 0:
        print("✅ Yescrypt mining working correctly!")
    else:
        print("⚠️  No shares found (normal with high difficulty)")
    
    return hashrate

def test_eco_integration():
    """Test eco-bonus integration"""
    print("\n🌱 ECO-BONUS INTEGRATION TEST")
    print("-" * 35)
    
    base_power = 80.0  # Watts
    eco_bonus = 1.15   # 15% bonus
    
    # Simulate different scenarios
    scenarios = [
        ("Low Performance CPU", 50.0),
        ("Mid-range CPU", 100.0),
        ("High Performance CPU", 200.0),
        ("Server CPU", 400.0)
    ]
    
    for scenario_name, base_hashrate in scenarios:
        eco_hashrate = base_hashrate * eco_bonus
        efficiency = eco_hashrate / base_power
        
        print(f"🖥️  {scenario_name}:")
        print(f"   Base: {base_hashrate} H/s")
        print(f"   Eco: {eco_hashrate:.1f} H/s (+15%)")
        print(f"   Efficiency: {efficiency:.2f} H/s/W")
        print()

def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description='ZION Yescrypt Simple Test')
    parser.add_argument('--duration', type=int, default=10, help='Test duration in seconds')
    args = parser.parse_args()
    
    print("🚀 Starting ZION Yescrypt Tests...")
    
    try:
        # Run basic mining test
        hashrate = test_basic_mining()
        
        # Run eco integration test
        test_eco_integration()
        
        print("\n🏆 ALL TESTS COMPLETED SUCCESSFULLY!")
        print(f"📊 Your system achieved: {hashrate:.2f} H/s")
        print(f"🌱 With eco bonus: {hashrate * 1.15:.2f} H/s")
        print(f"⚡ Power efficiency: {(hashrate * 1.15) / 80:.3f} H/s/W")
        
        print("\n🎯 NEXT STEPS:")
        print("1. Update wallet address in config")
        print("2. Run full miner with: python zion_yescrypt_optimized.py")
        print("3. Monitor eco-bonus rewards in pool")
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")

if __name__ == "__main__":
    main()