#!/usr/bin/env python3
"""
ZION GUI Miner Hashrate Test
Test skutečného hashrate reporting v GUI mineru
"""
import sys
import os
import time
import threading

# Add mining directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

try:
    from randomx_engine import RandomXEngine
    RANDOMX_AVAILABLE = True
    print("✅ RandomX engine dostupný")
except ImportError as e:
    RANDOMX_AVAILABLE = False
    print(f"❌ RandomX engine nedostupný: {e}")

def test_gui_hashrate_integration():
    """Test hashrate calculation stejně jako v GUI"""
    
    if not RANDOMX_AVAILABLE:
        print("⚠️ Používám simulační test")
        # Simulate hashrate like GUI would
        for i in range(5):
            simulated_hashrate = 1000 * 4 + (hash(str(time.time())) % 200 - 100)
            print(f"📊 Simulovaný hashrate: {simulated_hashrate:,} H/s")
            time.sleep(1)
        return
    
    print("🚀 Testování skutečného GUI hashrate...")
    
    # Initialize engines like GUI does
    threads = 4
    engines = {}
    hash_count = 0
    start_time = time.time()
    
    print(f"⚡ Inicializuji {threads} RandomX engines...")
    
    for i in range(threads):
        try:
            engine = RandomXEngine(fallback_to_sha256=True)
            seed = f"ZION_GUI_HASHRATE_TEST_{i}".encode()
            
            if engine.init(seed):
                engines[i] = engine
                print(f"✅ Engine {i+1}/{threads} ready")
            else:
                print(f"❌ Engine {i+1} failed")
        except Exception as e:
            print(f"❌ Engine {i+1} error: {e}")
    
    if not engines:
        print("❌ Žádné engines není možné inicializovat")
        return
    
    print(f"🎯 {len(engines)} engines připraveno, spouštím mining test...")
    
    # Mining test like GUI would do
    mining_active = True
    lock = threading.Lock()
    
    def mining_worker(thread_id):
        nonlocal hash_count
        
        if thread_id not in engines:
            return
            
        engine = engines[thread_id]
        local_count = 0
        nonce_base = thread_id * 0x1000000
        
        while mining_active and local_count < 5000:  # Limit for test
            try:
                # Create test data like GUI
                block_data = f"GUI_TEST_{nonce_base + local_count}".encode()
                
                # Calculate hash
                hash_result = engine.hash(block_data)
                local_count += 1
                
                # Update global count
                with lock:
                    hash_count += 1
                
                # Brief pause every 100 hashes
                if local_count % 100 == 0:
                    time.sleep(0.001)
                    
            except Exception as e:
                print(f"❌ Mining error thread {thread_id}: {e}")
                break
    
    # Start mining threads
    threads_list = []
    for thread_id in engines.keys():
        t = threading.Thread(target=mining_worker, args=(thread_id,))
        t.start()
        threads_list.append(t)
    
    # Monitor progress for 10 seconds
    print("📊 Monitoring hashrate pro 10 sekund...")
    for second in range(10):
        time.sleep(1)
        
        current_time = time.time()
        elapsed = current_time - start_time
        
        if elapsed > 0:
            current_hashrate = hash_count / elapsed
            print(f"⏱️  {second+1}/10s - Hashrate: {current_hashrate:,.0f} H/s - Total hashes: {hash_count:,}")
    
    # Stop mining
    mining_active = False
    
    # Wait for threads to finish
    for t in threads_list:
        t.join(timeout=2.0)
    
    # Final statistics
    final_time = time.time() - start_time
    final_hashrate = hash_count / final_time if final_time > 0 else 0
    
    print("\n🏆 VÝSLEDKY GUI HASHRATE TESTU:")
    print(f"   ⏱️  Celkový čas: {final_time:.2f} sekund")
    print(f"   🔢 Celkem hashů: {hash_count:,}")
    print(f"   ⚡ Finální hashrate: {final_hashrate:,.0f} H/s")
    print(f"   🧮 Per thread: {final_hashrate/len(engines):.0f} H/s")
    
    # Cleanup
    for engine in engines.values():
        if hasattr(engine, 'cleanup'):
            engine.cleanup()
    
    # Compare with expected GUI performance
    expected_gui_rate = len(engines) * 250  # Expected per thread
    if final_hashrate > expected_gui_rate * 0.8:
        print(f"✅ GUI hashrate výkon je DOBRÝ ({final_hashrate/expected_gui_rate*100:.0f}% expected)")
    else:
        print(f"⚠️ GUI hashrate je NIŽŠÍ než očekáváno ({final_hashrate/expected_gui_rate*100:.0f}% expected)")


if __name__ == "__main__":
    print("🎮 ZION GUI Miner Hashrate Test v2.6.75")
    print("=" * 50)
    
    test_gui_hashrate_integration()
    
    print("\n✨ Test completed!")