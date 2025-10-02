#!/usr/bin/env python3
"""
🔥 ZION SIMPLE 7K MINER v4.2 - No Huge Pages
==============================================

Simplified verze bez huge pages pro stabilitu:
- Základní RandomX flags  
- Jednoduché memory management
- Cache-only mode fallback
- Progresivní scaling
- Error recovery

TARGET: 7000+ H/s s běžnou pamětí! 💾
"""

import ctypes
import threading
import time
import psutil
import os
from typing import Optional, Dict

class Simple7KEngine:
    """Simplified RandomX engine pro 7K bez huge pages"""
    
    _shared_lib = None
    _shared_cache = None
    _shared_dataset = None
    _init_lock = threading.Lock()
    _use_cache_only = False
    
    def __init__(self, thread_id: int):
        self.thread_id = thread_id
        self.vm = None
        self.active = False
        
    @staticmethod
    def init_simple_shared() -> bool:
        """Jednoduchá inicializace shared resources"""
        with Simple7KEngine._init_lock:
            if Simple7KEngine._shared_lib is not None:
                return True
                
            # Load library
            lib_paths = ['/usr/local/lib/librandomx.so', '/usr/lib/librandomx.so']
            for path in lib_paths:
                try:
                    Simple7KEngine._shared_lib = ctypes.CDLL(path)
                    print(f"✅ RandomX library: {path}")
                    break
                except:
                    continue
                    
            if not Simple7KEngine._shared_lib:
                print("❌ RandomX library not found")
                return False
                
            lib = Simple7KEngine._shared_lib
            
            # Setup prototypes
            lib.randomx_alloc_cache.restype = ctypes.c_void_p
            lib.randomx_alloc_cache.argtypes = [ctypes.c_uint]
            lib.randomx_init_cache.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
            lib.randomx_create_vm.restype = ctypes.c_void_p
            lib.randomx_create_vm.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
            lib.randomx_calculate_hash.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p]
            
            try:
                # Simplified flags - NO huge pages, basic optimizations only
                flags = 0x2 | 0x8  # HARD_AES + JIT only
                seed = b'ZION_SIMPLE_7K_NO_HUGEPAGES'
                
                print("📦 Allocating cache (no huge pages)...")
                Simple7KEngine._shared_cache = lib.randomx_alloc_cache(flags)
                if not Simple7KEngine._shared_cache:
                    print("❌ Cache allocation failed")
                    return False
                    
                lib.randomx_init_cache(Simple7KEngine._shared_cache, seed, len(seed))
                print("✅ Cache ready")
                
                # Try dataset, fallback to cache-only
                try:
                    lib.randomx_alloc_dataset.restype = ctypes.c_void_p
                    lib.randomx_alloc_dataset.argtypes = [ctypes.c_uint]
                    lib.randomx_init_dataset.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong]
                    
                    print("📦 Attempting dataset allocation...")
                    dataset_flags = flags | 0x4  # Add FULL_MEM
                    Simple7KEngine._shared_dataset = lib.randomx_alloc_dataset(dataset_flags)
                    
                    if Simple7KEngine._shared_dataset:
                        print("⚙️ Initializing dataset...")
                        lib.randomx_init_dataset(Simple7KEngine._shared_dataset,
                                               Simple7KEngine._shared_cache, 0, 2097152)
                        print("✅ Dataset ready (full performance)")
                    else:
                        print("⚠️ Dataset failed, using cache-only mode")
                        Simple7KEngine._use_cache_only = True
                        
                except Exception as e:
                    print(f"⚠️ Dataset error: {e}, fallback to cache-only")
                    Simple7KEngine._shared_dataset = None
                    Simple7KEngine._use_cache_only = True
                    
                return True
                
            except Exception as e:
                print(f"❌ Initialization error: {e}")
                return False
                
    def init_thread(self) -> bool:
        """Initialize thread VM"""
        if not Simple7KEngine._shared_lib or not Simple7KEngine._shared_cache:
            return False
            
        try:
            lib = Simple7KEngine._shared_lib
            
            # Choose flags based on available resources
            if Simple7KEngine._use_cache_only:
                flags = 0x2 | 0x8  # HARD_AES + JIT, no FULL_MEM
                dataset = None
                print(f"Thread {self.thread_id}: cache-only mode")
            else:
                flags = 0x2 | 0x4 | 0x8  # HARD_AES + FULL_MEM + JIT
                dataset = Simple7KEngine._shared_dataset
                print(f"Thread {self.thread_id}: dataset mode")
                
            self.vm = lib.randomx_create_vm(flags, Simple7KEngine._shared_cache, dataset, None, 0)
            
            if self.vm:
                self.active = True
                return True
            else:
                return False
                
        except Exception as e:
            print(f"Thread {self.thread_id} VM failed: {e}")
            return False
            
    def simple_hash(self, data: bytes) -> Optional[bytes]:
        """Simple hash calculation"""
        if not self.active or not self.vm:
            return None
            
        try:
            output = (ctypes.c_char * 32)()
            Simple7KEngine._shared_lib.randomx_calculate_hash(self.vm, data, len(data), output)
            return bytes(output)
        except:
            return None
            
    def cleanup(self):
        """Simple cleanup"""
        if self.vm and Simple7KEngine._shared_lib:
            try:
                Simple7KEngine._shared_lib.randomx_destroy_vm(self.vm)
                self.vm = None
                self.active = False
            except:
                pass


class Simple7KMiner:
    """Simple miner pro 7K challenge"""
    
    def __init__(self):
        self.engines = []
        self.running = False
        self.stats = {}
        
    def init_simple(self, target_threads: int = 16) -> int:
        """Simple initialization"""
        print(f"🚀 Simple 7K Miner - {target_threads} threads target")
        print("💾 No huge pages mode for compatibility")
        print("=" * 45)
        
        # Initialize shared resources
        if not Simple7KEngine.init_simple_shared():
            print("❌ Shared resources failed")
            return 0
            
        # Create threads
        successful = 0
        for i in range(target_threads):
            print(f"⚡ Thread {i+1:2d}/{target_threads}...", end="")
            
            try:
                engine = Simple7KEngine(i)
                if engine.init_thread():
                    self.engines.append(engine)
                    successful += 1
                    print(" ✅")
                else:
                    print(" ❌")
                    
                time.sleep(0.02)  # Brief delay
                
            except Exception as e:
                print(f" ❌ ({e})")
                
        print("=" * 45)
        print(f"🎯 {successful} threads ready for 7K challenge")
        
        return successful
        
    def simple_worker(self, worker_id: int, duration: float):
        """Simple mining worker"""
        if worker_id >= len(self.engines):
            return
            
        engine = self.engines[worker_id]
        local_hashes = 0
        start_time = time.time()
        
        # CPU affinity
        try:
            os.sched_setaffinity(0, {worker_id % psutil.cpu_count()})
        except:
            pass
            
        # Simple mining loop
        while self.running and (time.time() - start_time) < duration:
            nonce = local_hashes + (worker_id * 5000) + int(time.time() * 100) % 100000
            input_data = f'SIMPLE7K_{worker_id}_{nonce}'.encode()
            
            result = engine.simple_hash(input_data)
            if result:
                local_hashes += 1
                
            # Stats update
            if local_hashes % 3000 == 0:
                elapsed = time.time() - start_time
                if elapsed > 0:
                    self.stats[worker_id] = local_hashes / elapsed
                    
        # Final rate
        elapsed = time.time() - start_time
        final_rate = local_hashes / elapsed if elapsed > 0 else 0
        self.stats[worker_id] = final_rate
        
        print(f"💎 T{worker_id+1:2d}: {local_hashes:6,} = {final_rate:6.1f} H/s")
        
    def simple_7k_challenge(self, duration: float = 35.0):
        """Simple 7K challenge"""
        print(f"🔥 SIMPLE 7K CHALLENGE - {duration} seconds")
        print("🎯 Target: 7000+ H/s without complexity")
        print("=" * 40)
        
        self.running = True
        start_time = time.time()
        
        # Start threads
        threads = []
        for i in range(len(self.engines)):
            thread = threading.Thread(target=self.simple_worker, args=(i, duration))
            thread.daemon = True
            threads.append(thread)
            thread.start()
            
        # Simple monitoring
        last_update = 0
        max_rate = 0
        
        while self.running and (time.time() - start_time) < duration:
            time.sleep(4)
            elapsed = time.time() - start_time
            
            if elapsed - last_update >= 6:
                current_rate = sum(self.stats.values())
                active = len([t for t in threads if t.is_alive()])
                
                if current_rate > max_rate:
                    max_rate = current_rate
                    
                print(f"📊 {elapsed:4.0f}s | Threads:{active:2d} | Rate:{current_rate:6.1f} H/s")
                last_update = elapsed
                
        # Stop and results
        self.running = False
        for thread in threads:
            thread.join(timeout=2)
            
        final_rate = sum(self.stats.values())
        
        print("=" * 40)
        print("🏆 SIMPLE 7K RESULTS:")
        print("=" * 40)
        print(f"🚀 Final rate: {final_rate:,.1f} H/s")
        print(f"📈 Peak rate:  {max_rate:,.1f} H/s")
        print(f"🧵 Threads:    {len(self.engines)}")
        
        # 7K check
        if final_rate >= 7000:
            print(f"🏆 7K BARRIER BROKEN! +{final_rate-7000:.0f} H/s over target!")
        elif max_rate >= 7000:
            print(f"🚀 7K PEAKED at {max_rate:.0f} H/s!")
        else:
            gap = 7000 - final_rate
            progress = (final_rate / 7000) * 100
            print(f"📈 7K Progress: {progress:.1f}% (gap: {gap:.0f} H/s)")
            
        return final_rate >= 7000 or max_rate >= 7000
        
    def cleanup(self):
        """Simple cleanup"""
        for engine in self.engines:
            engine.cleanup()


if __name__ == "__main__":
    print("🔥 ZION SIMPLE 7K CHALLENGER")
    print("============================")
    print("💾 No huge pages required")
    print("🎯 Target: 7000+ H/s")
    print()
    
    try:
        miner = Simple7KMiner()
        
        threads_ready = miner.init_simple(target_threads=18)
        
        if threads_ready >= 10:
            print(f"\\n🚀 Starting 7K challenge with {threads_ready} threads...")
            
            success = miner.simple_7k_challenge(duration=35.0)
            
            if success:
                print("\\n🏆 7K SUCCESS!")
                print("💫 Simple approach wins!")
            else:
                print("\\n📈 Close to 7K target!")
                print("🔧 More optimization possible")
                
        else:
            print(f"❌ Only {threads_ready} threads - not enough for 7K")
            
    except KeyboardInterrupt:
        print("\\n🛑 Stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'miner' in locals():
            miner.cleanup()
        print("\\n👋 Simple miner finished")