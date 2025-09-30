#!/usr/bin/env python3
"""
🔥 ZION STABLE MAXIMUM PERFORMANCE RandomX Miner
===============================================
- 12 threads with stability improvements
- Huge pages enabled
- Proper memory management
- Signal handling for clean shutdown
"""

import ctypes
import threading
import time
import psutil
import os
import signal
import sys
from typing import Optional, Dict

class StableRandomXEngine:
    """Stable high-performance RandomX Engine"""
    
    def __init__(self):
        self.lib = None
        self.dataset = None
        self.vm = None
        self.cache = None
        self.initialized = False
        self.lock = threading.RLock()  # Reentrant lock for stability
        
    def load_randomx_library(self) -> bool:
        """Load RandomX library with error checking"""
        lib_paths = [
            '/usr/local/lib/librandomx.so',
            '/usr/lib/librandomx.so',
            'librandomx.so'
        ]
        
        for lib_path in lib_paths:
            try:
                self.lib = ctypes.CDLL(lib_path)
                # Test basic function availability
                if hasattr(self.lib, 'randomx_alloc_cache'):
                    return True
            except OSError:
                continue
        return False
        
    def init_stable_performance(self, seed: bytes, thread_id: int = 0) -> bool:
        """Initialize with stability focus"""
        if not self.load_randomx_library():
            return False
            
        try:
            # Set up function prototypes with proper error checking
            self.lib.randomx_alloc_cache.restype = ctypes.c_void_p
            self.lib.randomx_alloc_cache.argtypes = [ctypes.c_uint]
            
            self.lib.randomx_init_cache.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
            self.lib.randomx_init_cache.restype = None
            
            self.lib.randomx_alloc_dataset.restype = ctypes.c_void_p
            self.lib.randomx_alloc_dataset.argtypes = [ctypes.c_uint]
            
            self.lib.randomx_init_dataset.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong]
            self.lib.randomx_init_dataset.restype = None
            
            self.lib.randomx_create_vm.restype = ctypes.c_void_p
            self.lib.randomx_create_vm.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
            
            self.lib.randomx_calculate_hash.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p]
            self.lib.randomx_calculate_hash.restype = None
            
            # Conservative flags for stability
            flags = 0
            flags |= (1 << 3)  # JIT
            flags |= (1 << 1)  # Large pages
            
            # Use full dataset only for first few threads to avoid memory issues
            if thread_id < 4:  # Only first 4 threads get full dataset
                flags |= (1 << 2)  # Full dataset
                
            # Allocate cache
            self.cache = self.lib.randomx_alloc_cache(flags)
            if not self.cache:
                return False
                
            # Initialize cache
            self.lib.randomx_init_cache(self.cache, seed, len(seed))
            
            # Allocate dataset if using full memory
            if flags & (1 << 2):  # If full dataset flag is set
                self.dataset = self.lib.randomx_alloc_dataset(flags)
                if self.dataset:
                    # Initialize dataset
                    dataset_count = 2097152
                    self.lib.randomx_init_dataset(self.dataset, self.cache, 0, dataset_count)
                else:
                    # Fallback to cache-only if dataset allocation fails
                    flags &= ~(1 << 2)
                    
            # Create VM
            self.vm = self.lib.randomx_create_vm(flags, self.cache, self.dataset)
            if not self.vm:
                return False
                
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"❌ Engine {thread_id} init failed: {e}")
            return False
            
    def hash(self, data: bytes) -> Optional[bytes]:
        """Safe hash calculation with error handling"""
        if not self.initialized or not self.vm:
            return None
            
        try:
            output = (ctypes.c_char * 32)()
            
            with self.lock:
                self.lib.randomx_calculate_hash(self.vm, data, len(data), output)
                
            return bytes(output)
            
        except Exception:
            return None
            
    def cleanup(self):
        """Safe cleanup with null checks"""
        try:
            if self.vm and self.lib:
                self.lib.randomx_destroy_vm(self.vm)
                self.vm = None
            if self.dataset and self.lib:
                self.lib.randomx_release_dataset(self.dataset)
                self.dataset = None
            if self.cache and self.lib:
                self.lib.randomx_release_cache(self.cache)
                self.cache = None
        except:
            pass
        self.initialized = False


class StableZionMiner:
    """Stable ZION Miner with proper shutdown handling"""
    
    def __init__(self, num_threads: int = 12):
        self.num_threads = num_threads
        self.engines = []
        self.running = False
        self.shutdown_requested = False
        self.stats = {'threads': {}, 'start_time': None}
        
        # Setup signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n🛑 Shutdown signal received ({signum})")
        self.shutdown_requested = True
        self.running = False
        
    def initialize_engines(self) -> bool:
        """Initialize engines with stability checks"""
        print(f"🚀 Initializing {self.num_threads} stable RandomX engines...")
        print("🔥 Optimizations: Huge Pages + JIT + Selective Full Dataset")
        print("=" * 60)
        
        seed = b'ZION_STABLE_MAXIMUM_2025'
        
        for i in range(self.num_threads):
            if self.shutdown_requested:
                break
                
            print(f"⚡ Engine {i+1:2d}/{self.num_threads}...", end="", flush=True)
            
            engine = StableRandomXEngine()
            
            if engine.init_stable_performance(seed, thread_id=i):
                self.engines.append(engine)
                dataset_mode = "dataset" if i < 4 else "cache"
                print(f" ✅ ({dataset_mode})")
            else:
                print(f" ❌")
                # Continue with other engines even if one fails
                
        success_rate = len(self.engines) / self.num_threads * 100
        print("=" * 60)
        print(f"🎯 {len(self.engines)}/{self.num_threads} engines ready ({success_rate:.1f}% success)")
        
        return len(self.engines) > 0
        
    def stable_mining_worker(self, worker_id: int, duration: float):
        """Stable mining worker with error recovery"""
        if worker_id >= len(self.engines):
            return
            
        engine = self.engines[worker_id]
        local_hashes = 0
        start_time = time.time()
        consecutive_errors = 0
        
        try:
            while self.running and not self.shutdown_requested and (time.time() - start_time) < duration:
                # Create input
                nonce = int(time.time() * 1000000) + local_hashes + (worker_id * 1000000)
                input_data = f'ZION_STABLE_{worker_id}_{nonce}'.encode()
                
                # Calculate hash with error handling
                hash_result = engine.hash(input_data)
                
                if hash_result is not None:
                    local_hashes += 1
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1
                    if consecutive_errors > 10:
                        print(f"⚠️ Worker {worker_id} has too many errors, stopping")
                        break
                    time.sleep(0.001)  # Brief pause on error
                    
                # Update stats periodically
                if local_hashes > 0 and local_hashes % 50 == 0:
                    elapsed = time.time() - start_time
                    current_hashrate = local_hashes / elapsed
                    
                    self.stats['threads'][worker_id] = {
                        'hashes': local_hashes,
                        'hashrate': current_hashrate,
                        'elapsed': elapsed,
                        'errors': consecutive_errors
                    }
                    
        except Exception as e:
            print(f"❌ Worker {worker_id} exception: {e}")
            
        # Final stats
        elapsed = time.time() - start_time
        final_hashrate = local_hashes / elapsed if elapsed > 0 else 0
        
        self.stats['threads'][worker_id] = {
            'hashes': local_hashes,
            'hashrate': final_hashrate,
            'elapsed': elapsed
        }
        
        print(f"💎 Thread {worker_id+1:2d}: {local_hashes:6,} hashes = {final_hashrate:7.1f} H/s")
        
    def start_stable_mining(self, duration: float = 45.0):
        """Start stable high-performance mining"""
        print(f"🔥 STARTING STABLE MAXIMUM MINING - {duration} seconds")
        print("=" * 60)
        print(f"🧵 Active engines: {len(self.engines)}")
        print(f"💾 Huge Pages: {'✅' if self._check_huge_pages() else '❌'}")
        print("=" * 60)
        
        self.running = True
        self.stats['start_time'] = time.time()
        
        # Start mining threads
        threads = []
        for i in range(len(self.engines)):
            thread = threading.Thread(target=self.stable_mining_worker, args=(i, duration))
            thread.daemon = True
            threads.append(thread)
            thread.start()
            
        # Monitor progress
        start_time = time.time()
        last_update = 0
        
        try:
            while self.running and not self.shutdown_requested and (time.time() - start_time) < duration:
                time.sleep(2)
                
                elapsed = time.time() - start_time
                if elapsed - last_update >= 5:  # Update every 5 seconds
                    current_total = sum(
                        stats.get('hashrate', 0) 
                        for stats in self.stats['threads'].values()
                    )
                    active_count = len([t for t in threads if t.is_alive()])
                    
                    print(f"📊 {elapsed:5.1f}s | Threads: {active_count:2d} | Total: {current_total:7.1f} H/s")
                    last_update = elapsed
                    
        except KeyboardInterrupt:
            print("\n🛑 User interrupt")
            self.shutdown_requested = True
            
        # Stop mining and wait for threads
        self.running = False
        
        print("🔄 Waiting for threads to finish...")
        for thread in threads:
            thread.join(timeout=5.0)
            
        # Calculate results
        total_hashes = sum(stats['hashes'] for stats in self.stats['threads'].values())
        total_elapsed = time.time() - self.stats['start_time']
        total_hashrate = total_hashes / total_elapsed if total_elapsed > 0 else 0
        
        print("=" * 60)
        print("🎯 STABLE MINING RESULTS:")
        print("=" * 60)
        print(f"💎 Total hashes: {total_hashes:,}")
        print(f"⏱️  Runtime: {total_elapsed:.1f} seconds")
        print(f"🚀 Total hashrate: {total_hashrate:,.1f} H/s")
        print(f"🧵 Active engines: {len(self.engines)}")
        print(f"📈 Per-engine avg: {total_hashrate/max(len(self.engines), 1):.1f} H/s")
        
        return {
            'total_hashes': total_hashes,
            'total_hashrate': total_hashrate,
            'engines': len(self.engines)
        }
        
    def _check_huge_pages(self) -> bool:
        """Check huge pages status"""
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'HugePages_Total' in line:
                        return int(line.split()[1]) > 0
        except:
            pass
        return False
        
    def cleanup(self):
        """Safe cleanup of all resources"""
        print("🧹 Cleaning up engines...")
        for i, engine in enumerate(self.engines):
            try:
                engine.cleanup()
                print(f"   ✅ Engine {i+1} cleaned")
            except Exception as e:
                print(f"   ⚠️ Engine {i+1} cleanup issue: {e}")


if __name__ == "__main__":
    print("🔥 ZION STABLE MAXIMUM PERFORMANCE MINER")
    print("=======================================")
    print("🚀 12 Threads + Huge Pages + Stability")
    
    # System info
    print(f"\n💻 System: {psutil.cpu_count()} cores, {psutil.virtual_memory().total//(1024**3)} GB RAM")
    
    try:
        miner = StableZionMiner(num_threads=12)
        
        if miner.initialize_engines():
            print("\n" + "🚀" * 15 + " STABLE LAUNCH " + "🚀" * 15)
            
            results = miner.start_stable_mining(duration=45.0)
            
            print(f"\n📊 FINAL PERFORMANCE:")
            if results['total_hashrate'] > 2500:
                print(f"   🏆 EXCEPTIONAL: {results['total_hashrate']:,.1f} H/s!")
            elif results['total_hashrate'] > 1500:
                print(f"   🚀 EXCELLENT: {results['total_hashrate']:,.1f} H/s!")
            elif results['total_hashrate'] > 800:
                print(f"   ✅ VERY GOOD: {results['total_hashrate']:,.1f} H/s!")
            else:
                print(f"   📈 GOOD: {results['total_hashrate']:,.1f} H/s")
                
            print(f"\n💎 ZION STABLE MAXIMUM: {results['total_hashrate']:,.1f} H/s")
            
        else:
            print("❌ Engine initialization failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'miner' in locals():
            miner.cleanup()
        print("\n👋 ZION Stable Maximum Miner finished!")