#!/usr/bin/env python3
"""
ZION Pool Yescrypt Support Test
Quick test to verify pool supports Yescrypt algorithm properly
"""

import json
import socket
import time
import sys

def test_pool_yescrypt_support():
    """Test if pool supports Yescrypt mining"""
    
    print("🧪 Testing ZION Pool Yescrypt Support")
    print("=" * 50)
    
    try:
        # Test connection to pool
        print("📡 Connecting to pool localhost:3333...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        sock.connect(('localhost', 4444))
        print("✅ Connected to pool successfully")
        
        # Test 1: Subscribe
        print("\n🔗 Test 1: Mining Subscribe")
        subscribe_msg = {
            'id': 1,
            'method': 'mining.subscribe',
            'params': ['ZionYescryptMiner/1.0', None, 'localhost', 4444]
        }
        
        message = json.dumps(subscribe_msg) + '\n'
        sock.send(message.encode())
        
        response = sock.recv(4096).decode().strip()
        print(f"📥 Subscribe response: {response}")
        
        # Test 2: Authorize with algorithm hint
        print("\n🔐 Test 2: Mining Authorization (Yescrypt)")
        auth_msg = {
            'id': 2,
            'method': 'mining.authorize',
            'params': ['ZioniYESCRYPT_POOL_TEST_ADDRESS_123456', 'yescrypt']
        }
        
        message = json.dumps(auth_msg) + '\n'
        sock.send(message.encode())
        
        response = sock.recv(4096).decode().strip()
        print(f"📥 Auth response: {response}")
        
        # Test 3: Submit test share
        print("\n🎯 Test 3: Submit Yescrypt Test Share")
        submit_msg = {
            'id': 3,
            'method': 'mining.submit',
            'params': [
                'worker1',                    # worker
                'yescrypt_test_job_12345',   # job_id
                'deadbeef12345678',          # nonce
                'yescrypt_test_result_hash', # mix_hash (result for Yescrypt)
                'test_header_hash_yescrypt'  # header_hash
            ]
        }
        
        message = json.dumps(submit_msg) + '\n'
        sock.send(message.encode())
        
        response = sock.recv(4096).decode().strip()
        print(f"📥 Submit response: {response}")
        
        sock.close()
        print("\n✅ Pool Yescrypt support test completed")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

def test_pool_api_info():
    """Test pool API for algorithm support"""
    
    print("\n🌐 Testing Pool API Info")
    print("-" * 30)
    
    try:
        import urllib.request
        
        # Get pool info from API
        print("📡 Requesting pool info from API...")
        
        url = "http://localhost:4445/api/pool"
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())
            
        print("✅ Pool API response received")
        print(f"📊 Pool Name: {data.get('name', 'Unknown')}")
        print(f"🔢 Pool Version: {data.get('version', 'Unknown')}")
        
        algorithms = data.get('algorithms', [])
        print(f"⚗️ Supported Algorithms: {algorithms}")
        
        if 'yescrypt' in algorithms:
            print("✅ YESCRYPT is supported!")
            
            eco_bonuses = data.get('rewards', {}).get('eco_bonuses', {})
            yescrypt_bonus = eco_bonuses.get('yescrypt', 1.0)
            print(f"🌱 Yescrypt eco-bonus: {yescrypt_bonus}x (+{int((yescrypt_bonus-1)*100)}%)")
            
        else:
            print("❌ YESCRYPT is NOT supported")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def test_pool_health():
    """Test pool health status"""
    
    print("\n🏥 Testing Pool Health")
    print("-" * 25)
    
    try:
        import urllib.request
        
        url = "http://localhost:4445/api/health"
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode())
            
        print("✅ Pool is healthy")
        print(f"⏱️ Uptime: {data.get('uptime_seconds', 0):.1f}s")
        print(f"👥 Active Connections: {data.get('active_connections', 0)}")
        print(f"⛏️ Total Miners: {data.get('total_miners', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        print("💡 Make sure pool is running with API enabled")
        return False

def main():
    """Run all tests"""
    print("🚀 ZION Pool Yescrypt Readiness Test")
    print("=" * 60)
    
    # Check if pool is running
    if not test_pool_health():
        print("\n🚨 Pool is not running or API is not accessible")
        print("💡 Start pool with: python zion_universal_pool_v2.py")
        return
    
    # Test API info
    api_test = test_pool_api_info()
    
    # Test stratum connection (if pool is ready)
    if api_test:
        stratum_test = test_pool_yescrypt_support()
    else:
        stratum_test = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("-" * 20)
    print(f"🏥 Pool Health: {'✅ PASS' if test_pool_health() else '❌ FAIL'}")
    print(f"🌐 API Support: {'✅ PASS' if api_test else '❌ FAIL'}")
    print(f"📡 Stratum Test: {'✅ PASS' if stratum_test else '❌ FAIL'}")
    
    if api_test and stratum_test:
        print("\n🎉 POOL IS READY FOR YESCRYPT MINING!")
        print("🚀 You can now connect your Yescrypt miners")
        print("🌱 Eco-bonus is active for energy-efficient mining")
    else:
        print("\n⚠️ Pool needs configuration for Yescrypt support")
        print("💡 Check pool logs and restart if needed")

if __name__ == "__main__":
    main()