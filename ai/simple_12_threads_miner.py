import time
import threading
import hashlib
import random
from concurrent.futures import ThreadPoolExecutor

MINING_THREADS = 12
mining_active = True
total_hashes = 0
start_time = time.time()

def mining_thread(thread_id):
    global total_hashes
    local_hashes = 0
    
    while mining_active:
        # Simple hash simulation
        data = f"ZION_BLOCK_{time.time()}_{thread_id}_{random.randint(0, 1000000)}"
        hash_result = hashlib.sha256(data.encode()).hexdigest()
        local_hashes += 1
        total_hashes += 1
        
        if local_hashes % 10000 == 0:
            time.sleep(0.01)  # Small pause for system stability
    
    print(f"🔥 Mining thread {thread_id} stopped. Hashes: {local_hashes}")

def monitor_hashrate():
    global total_hashes, start_time
    last_hashes = 0
    
    while mining_active:
        time.sleep(10)  # Update every 10 seconds
        current_time = time.time()
        current_hashes = total_hashes
        
        # Current hashrate (last 10 seconds)
        current_rate = (current_hashes - last_hashes) / 10
        
        # Average hashrate
        total_time = current_time - start_time
        average_rate = current_hashes / total_time if total_time > 0 else 0
        
        print("--" * 25)
        print(f"📊 ZION 12-Threads Miner Stats:")
        print(f"   Current: {current_rate:.2f} H/s")
        print(f"   Average: {average_rate:.2f} H/s")
        print(f"   Total:   {current_hashes} hashes")
        print(f"   Threads: {MINING_THREADS} active")
        print("--" * 25)
        
        last_hashes = current_hashes

def main():
    global mining_active
    
    print("🔥 ZION Simple 12-Threads Miner")
    print(f"🧵 Threads: {MINING_THREADS}")
    print("🚀 Starting mining...")
    print("=" * 50)
    
    try:
        with ThreadPoolExecutor(max_workers=MINING_THREADS + 1) as executor:
            # Start mining threads
            for i in range(MINING_THREADS):
                executor.submit(mining_thread, i)
            
            # Start monitoring
            executor.submit(monitor_hashrate)
            
            print("⭐ 12-Threads mining started!")
            print("   Press Ctrl+C to stop")
            
            while mining_active:
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping 12-threads miner...")
        mining_active = False
    
    # Final stats
    total_time = time.time() - start_time
    final_hashrate = total_hashes / total_time if total_time > 0 else 0
    
    print("\n🏁 Final Mining Stats:")
    print(f"   Total Time: {total_time:.2f} seconds")
    print(f"   Total Hashes: {total_hashes}")
    print(f"   Average Rate: {final_hashrate:.2f} H/s")
    print("🌟 12-Threads mining complete!")

if __name__ == "__main__":
    main()
