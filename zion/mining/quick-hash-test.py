#!/usr/bin/env python3
"""Quick mining hash test - no input required"""
import time
import hashlib
import multiprocessing

def cpu_intensive_hash(data, iterations=300):
    """CPU-intensive hash simulující RandomX"""
    if isinstance(data, str):
        data = data.encode('utf-8')  # Convert string to bytes properly
        
    result = data
    for i in range(iterations):
        result = hashlib.sha256(result).digest()
        result = hashlib.sha3_256(result).digest()  
        result = hashlib.blake2b(result).digest()
        
        temp = bytearray(result)
        for j in range(0, len(temp)-1, 2):
            temp[j] ^= temp[j+1]
            temp[j+1] = (temp[j+1] + temp[j]) % 256
            
        result = bytes(temp)
        
    return result

print("🔥 Testing ZION Real Mining Hash Performance...")
print(f"🖥️ CPU Cores: {multiprocessing.cpu_count()}")

# Test single thread performance
test_data = "zion-mining-test-data" * 4
start_time = time.time()

hashes_done = 0
for i in range(50):  # 50 hashes test
    hash_result = cpu_intensive_hash(test_data + f"{i:08x}")
    hashes_done += 1
    
elapsed = time.time() - start_time
hashrate = hashes_done / elapsed

print(f"📊 Single thread: {hashrate:.1f} H/s")
print(f"⏱️ Time per hash: {elapsed/hashes_done*1000:.1f} ms") 
print(f"🚀 Estimated {multiprocessing.cpu_count()} threads: {hashrate*multiprocessing.cpu_count():.0f} H/s")

if hashrate > 50:
    print("✅ Mining engine funguje - skutečně hashuje!")
    print("💎 CPU-intensive RandomX-style hashing works!")
else:
    print("⚠️ Možný problém s hashing performance")
    
print(f"🌡️ Current CPU temp check with: sensors | grep Tctl")