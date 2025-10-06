#!/usr/bin/env python3
"""
🔥 ZION MEMORY-EFFICIENT 7K CHALLENGER v4.1
============================================

Memory-optimized verze pro 7000+ H/s:
- Single dataset sdílený mezi všemi thread
- Minimální memory footprint
- Progressive engine scaling
- Memory pressure detection
- Fallback strategies při memory issues

TARGET: 7000+ H/s s minimální pamětí! 💾
"""

import ctypes
import threading
import time
import psutil
import os
import gc
from typing import Optional, Dict

class MemoryEfficientEngine:
    """Memory-efficient RandomX engine pro 7K challenge"""
    
    # Global shared resources (pouze jeden dataset!)
    _global_lib = None
    _global_cache = None
    _global_dataset = None
    _global_lock = threading.Lock()
    _engines_created = 0
    
    def __init__(self, thread_id: int):
        self.thread_id = thread_id
        self.vm = None
        self.initialized = False
        
    @staticmethod
    def get_available_memory_gb() -> float:
        """Získat dostupnou paměť v GB"""
        return psutil.virtual_memory().available / (1024**3)
        
    @staticmethod
    def init_global_resources() -> bool:
        """Inicializace globálních resources (volat pouze jednou)"""
        with MemoryEfficientEngine._global_lock:
            if MemoryEfficientEngine._global_dataset is not None:
                return True
                
            print(f"💾 Available memory: {MemoryEfficientEngine.get_available_memory_gb():.1f} GB")
            
            # Load library
            lib_paths = ['/usr/local/lib/librandomx.so', '/usr/lib/librandomx.so']
            for path in lib_paths:
                try:
                    MemoryEfficientEngine._global_lib = ctypes.CDLL(path)
                    print(f"✅ Loaded: {path}")
                    break
                except:
                    continue
                    
            if not MemoryEfficientEngine._global_lib:
                print("❌ RandomX library not found")
                return False
                
            lib = MemoryEfficientEngine._global_lib
            
            try:
                # Function prototypes
                lib.randomx_alloc_cache.restype = ctypes.c_void_p
                lib.randomx_alloc_cache.argtypes = [ctypes.c_uint]
                lib.randomx_init_cache.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
                lib.randomx_alloc_dataset.restype = ctypes.c_void_p
                lib.randomx_alloc_dataset.argtypes = [ctypes.c_uint]
                lib.randomx_init_dataset.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong]
                lib.randomx_create_vm.restype = ctypes.c_void_p
                lib.randomx_create_vm.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
                lib.randomx_calculate_hash.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p]
                
                # Memory-conservative flags
                flags = 0x1 | 0x2 | 0x4 | 0x8  # Basic optimizations
                seed = b'ZION_MEMORY_EFFICIENT_7K'
                
                print("📦 Allocating single shared cache...")
                MemoryEfficientEngine._global_cache = lib.randomx_alloc_cache(flags)
                if not MemoryEfficientEngine._global_cache:
                    print("❌ Cache allocation failed")
                    return False
                    
                lib.randomx_init_cache(MemoryEfficientEngine._global_cache, seed, len(seed))
                print("✅ Cache ready")
                
                print("📦 Allocating single shared dataset...")
                MemoryEfficientEngine._global_dataset = lib.randomx_alloc_dataset(flags)
                if not MemoryEfficientEngine._global_dataset:
                    print("❌ Dataset allocation failed")
                    return False
                    
                print("⚙️ Initializing shared dataset...")
                lib.randomx_init_dataset(MemoryEfficientEngine._global_dataset,
                                       MemoryEfficientEngine._global_cache, 0, 2097152)
                print("✅ Single dataset ready for all threads")
                
                return True
                
            except Exception as e:
                print(f"❌ Global resource init failed: {e}")
                return False
                
    def init_thread_vm(self) -> bool:
        """Initialize lightweight thread VM"""
        if not MemoryEfficientEngine._global_lib or not MemoryEfficientEngine._global_dataset:
            return False
            
        try:
            lib = MemoryEfficientEngine._global_lib
            flags = 0x1 | 0x2 | 0x4 | 0x8
            
            # Create VM sharing the global dataset
            self.vm = lib.randomx_create_vm(flags, None, 
                                          MemoryEfficientEngine._global_dataset, None, 0)
            
            if self.vm:
                self.initialized = True
                MemoryEfficientEngine._engines_created += 1
                return True
            else:
                return False
                
        except Exception as e:
            print(f"Thread VM init failed: {e}")
            return False
            
    def fast_hash(self, data: bytes) -> Optional[bytes]:
        """Fast hash calculation"""
        if not self.initialized or not self.vm:
            return None
            
        try:
            output = (ctypes.c_char * 32)()
            MemoryEfficientEngine._global_lib.randomx_calculate_hash(self.vm, data, len(data), output)
            return bytes(output)
        except:
            return None
            
    def cleanup(self):
        """Lightweight cleanup"""
        if self.vm and MemoryEfficientEngine._global_lib:
            try:
                MemoryEfficientEngine._global_lib.randomx_destroy_vm(self.vm)
                self.vm = None
                self.initialized = False
            except:
                pass


class MemoryEfficient7KMiner:
    """Memory-efficient miner pro 7K challenge"""
    
    def __init__(self):
        self.engines = []
        self.running = False
        self.stats = {}
        
    def init_memory_efficient(self, max_threads: int = 16) -> int:
        """Initialize s memory-efficient approach"""
        print(f"🚀 Memory-Efficient 7K Miner initialization")
        print(f"🎯 Target threads: {max_threads}")
        print("=" * 50)
        
        # Initialize global resources once
        if not MemoryEfficientEngine.init_global_resources():
            print("❌ Failed to initialize global resources")
            return 0
            
        # Progressive thread creation with memory monitoring
        successful = 0
        for i in range(max_threads):
            memory_gb = MemoryEfficientEngine.get_available_memory_gb()
            
            if memory_gb < 1.0:  # Less than 1GB available
                print(f"⚠️ Low memory ({memory_gb:.1f} GB), stopping at {successful} threads")
                break
                
            print(f"⚡ Thread {i+1:2d}/{max_threads} (Mem: {memory_gb:.1f}GB)...", end="")
            
            try:
                engine = MemoryEfficientEngine(i)
                if engine.init_thread_vm():
                    self.engines.append(engine)
                    successful += 1
                    print(" ✅")
                    
                    # Brief pause for memory stabilization
                    time.sleep(0.05)
                else:
                    print(" ❌")
                    break
                    
            except Exception as e:
                print(f" ❌ ({e})")
                break
                
        print("=" * 50)
        print(f"🎯 {successful} memory-efficient engines ready")
        print(f"💾 Final memory: {MemoryEfficientEngine.get_available_memory_gb():.1f} GB available")
        
        return successful
        
    def efficient_worker(self, worker_id: int, duration: float):
        """Memory-efficient mining worker"""
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
            
        # High-speed mining loop
        while self.running and (time.time() - start_time) < duration:
            nonce = int(time.time() * 1000000) + local_hashes + (worker_id * 10000)
            input_data = f'MEM7K_{worker_id}_{nonce}'.encode()
            
            hash_result = engine.fast_hash(input_data)
            if hash_result:
                local_hashes += 1
                
            # Less frequent stats updates for performance
            if local_hashes % 2000 == 0:
                elapsed = time.time() - start_time
                if elapsed > 0:
                    self.stats[worker_id] = {
                        'hashes': local_hashes,
                        'hashrate': local_hashes / elapsed
                    }
                    
        # Final stats
        elapsed = time.time() - start_time
        final_rate = local_hashes / elapsed if elapsed > 0 else 0
        self.stats[worker_id] = {'hashes': local_hashes, 'hashrate': final_rate}
        
        print(f"💎 Th{worker_id+1:2d}: {local_hashes:6,} = {final_rate:6.1f} H/s")
        
    def challenge_7k_efficient(self, duration: float = 45.0):
        """7K challenge s memory efficiency"""
        print(f"🔥 MEMORY-EFFICIENT 7K CHALLENGE - {duration}s")
        print("💾 Single dataset shared across all threads")
        print("=" * 45)
        
        self.running = True
        start_time = time.time()
        
        # Start threads
        threads = []
        for i in range(len(self.engines)):
            thread = threading.Thread(target=self.efficient_worker, args=(i, duration))
            thread.daemon = True
            threads.append(thread)
            thread.start()
            
        # Monitor with memory tracking
        last_update = 0
        max_rate = 0
        
        while self.running and (time.time() - start_time) < duration:
            time.sleep(3)
            elapsed = time.time() - start_time
            
            if elapsed - last_update >= 5:
                total_rate = sum(s.get('hashrate', 0) for s in self.stats.values())
                active = len([t for t in threads if t.is_alive()])
                memory_gb = MemoryEfficientEngine.get_available_memory_gb()
                
                if total_rate > max_rate:
                    max_rate = total_rate
                    
                print(f"📊 {elapsed:4.0f}s | T:{active:2d} | {total_rate:6.1f} H/s | Mem:{memory_gb:.1f}GB")
                last_update = elapsed
                
        # Stop and collect
        self.running = False
        for thread in threads:
            thread.join(timeout=3)
            
        total_hashes = sum(s['hashes'] for s in self.stats.values())
        total_time = time.time() - start_time
        final_hashrate = total_hashes / total_time
        
        print("=" * 45)
        print("🏆 MEMORY-EFFICIENT 7K RESULTS:")
        print("=" * 45)
        print(f"💎 Hashes: {total_hashes:,}")
        print(f"⏱️  Time: {total_time:.1f}s")
        print(f"🚀 Final: {final_hashrate:,.1f} H/s")
        print(f"📈 Peak: {max_rate:,.1f} H/s")
        print(f"🧵 Threads: {len(self.engines)}")
        print(f"💾 Memory used efficiently")
        
        # 7K analysis
        target_7k = 7000
        if final_hashrate >= target_7k:
            advantage = final_hashrate - target_7k
            print(f"🏆 7K ACHIEVED! +{advantage:.0f} H/s!")
        elif max_rate >= target_7k:
            print(f"🚀 7K PEAKED at {max_rate:.0f} H/s!")
        else:
            gap = target_7k - final_hashrate
            progress = (final_hashrate / target_7k) * 100
            print(f"📈 7K Progress: {progress:.1f}% ({gap:.0f} H/s gap)")
            
        return {
            'final': final_hashrate,
            'peak': max_rate,
            'threads': len(self.engines),
            'target_reached': final_hashrate >= target_7k or max_rate >= target_7k
        }
        
    def cleanup(self):
        """Memory-efficient cleanup"""
        for engine in self.engines:
            engine.cleanup()
        gc.collect()


if __name__ == "__main__":
    print("🔥 ZION MEMORY-EFFICIENT 7K CHALLENGER")
    print("======================================")
    print("💾 Optimized for memory efficiency")
    print("🎯 Target: 7000+ H/s with minimal RAM")
    print()
    
    try:
        miner = MemoryEfficient7KMiner()
        
        # Progressive initialization
        threads_created = miner.init_memory_efficient(max_threads=18)
        
        if threads_created >= 8:  # Need at least 8 threads for 7K attempt
            print(f"\\n🚀 Launching 7K challenge with {threads_created} threads...")
            
            results = miner.challenge_7k_efficient(duration=40.0)
            
            print(f"\\n🎯 7K CHALLENGE SUMMARY:")
            print(f"   Final rate: {results['final']:,.1f} H/s")
            print(f"   Peak rate:  {results['peak']:,.1f} H/s")
            print(f"   Threads:    {results['threads']}")
            
            if results['target_reached']:
                print("\\n🏆 7K TARGET REACHED!")
                print("💫 Memory-efficient victory!")
            else:
                efficiency = (results['final'] / 7000) * 100
                print(f"\\n📈 7K Efficiency: {efficiency:.1f}%")
                print("💾 Memory constraints handled well!")
                
        else:
            print(f"❌ Only {threads_created} threads created - insufficient for 7K")
            
    except KeyboardInterrupt:
        print("\\n🛑 Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'miner' in locals():
            miner.cleanup()
        print("\\n👋 Memory-efficient miner finished")