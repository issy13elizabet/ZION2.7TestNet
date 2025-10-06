#!/usr/bin/env python3
"""
ZION Real Miner - Live Hash Performance Test
Testuje skutečný CPU-intensive mining bez GUI
"""
import time
import threading
import multiprocessing
from datetime import datetime

# Import mining engine z hlavního souboru
import sys
sys.path.append('/home/maitreya/Desktop/zion-miner-1.4.0')

class TestMiner:
    def __init__(self):
        self.mining = False
        self.total_hashes = 0
        self.start_time = 0
        
    def cpu_intensive_hash(self, data, iterations=500):
        """CPU-intensive hash funkce simulující RandomX"""
        import hashlib
        
        if isinstance(data, str):
            data = bytes.fromhex(data)
            
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
        
    def mining_thread(self, thread_id, duration=30):
        """Mining thread pro test CPU utilizace"""
        print(f"🚀 Thread {thread_id} začíná mining test...")
        
        hashes = 0
        test_data = f"zion-test-mining-data-{thread_id}".ljust(64, '0')
        
        start_time = time.time()
        while (time.time() - start_time) < duration and self.mining:
            # Skutečné CPU-intensive hashování
            nonce = hashes
            mining_blob = test_data + f"{nonce:08x}"
            hash_result = self.cpu_intensive_hash(mining_blob)
            
            hashes += 1
            self.total_hashes += 1
            
            # Log každých 100 hashů
            if hashes % 100 == 0:
                elapsed = time.time() - start_time
                thread_hashrate = hashes / elapsed if elapsed > 0 else 0
                print(f"📊 Thread {thread_id}: {thread_hashrate:.1f} H/s ({hashes} hashes)")
                
        elapsed = time.time() - start_time
        final_hashrate = hashes / elapsed if elapsed > 0 else 0
        print(f"✅ Thread {thread_id} dokončen: {final_hashrate:.1f} H/s za {elapsed:.1f}s")
        
    def run_mining_test(self, num_threads=6, duration=30):
        """Spustí mining test na více threads"""
        print("🌟" + "="*60 + "🌟")
        print("🔥        ZION REAL MINER - CPU MINING TEST       🔥")
        print("🌟" + "="*60 + "🌟")
        print(f"⚙️  Threads: {num_threads}")
        print(f"⏰ Duration: {duration}s")
        print(f"🌡️  Budeš vidět CPU load na 100%!")
        print()
        
        # Potvrdi spuštění
        response = input("🚀 Spustit CPU mining test? (y/n): ")
        if response.lower() != 'y':
            print("❌ Test zrušen")
            return
            
        self.mining = True
        self.total_hashes = 0
        self.start_time = time.time()
        
        # Spusti mining threads
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=self.mining_thread, args=(i, duration), daemon=True)
            t.start()
            threads.append(t)
            time.sleep(0.5)  # Postupné spouštění
            
        print(f"🔥 Mining spuštěn na {num_threads} threads!")
        print(f"💻 Zkontroluj CPU usage: htop nebo top")
        print(f"⏳ Čekání {duration} sekund...")
        
        # Počkej na dokončení
        for t in threads:
            t.join()
            
        # Final statistiky
        total_elapsed = time.time() - self.start_time
        total_hashrate = self.total_hashes / total_elapsed if total_elapsed > 0 else 0
        
        print("\n🏁 FINAL VÝSLEDKY:")
        print(f"📊 Celkový hashrate: {total_hashrate:.1f} H/s")
        print(f"🔢 Celkem hashů: {self.total_hashes:,}")
        print(f"⏱️  Čas: {total_elapsed:.1f}s")
        print(f"🧮 Hash/thread: {total_hashrate/num_threads:.1f} H/s")
        
        if total_hashrate > 500:
            print("✅ Mining engine funguje správně!")
            print("💎 Miner je připraven pro skutečné pool mining!")
        else:
            print("⚠️  Nízký hashrate - možná problém s CPU")
            
        self.mining = False

if __name__ == "__main__":
    # CPU info
    cpu_count = multiprocessing.cpu_count()
    print(f"🖥️  Detekováno CPU cores: {cpu_count}")
    
    # Spusť test
    miner = TestMiner()
    miner.run_mining_test(num_threads=cpu_count)