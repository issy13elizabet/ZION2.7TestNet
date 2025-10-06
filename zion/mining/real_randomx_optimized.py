#!/usr/bin/env python3
"""
🔥 ZION Real RandomX Mining - Optimized for Maximum Performance
===============================================================

This is 100% REAL RandomX mining, no simulation!
Optimized for:
- Full dataset mode (faster than cache-only)
- Large pages support
- JIT compilation 
- Multi-threading
"""

import ctypes
import threading
import time
import psutil
import os
from typing import Optional, Tuple

class OptimizedRandomXEngine:
    """Optimized Real RandomX Engine - No Fallbacks, Pure Performance"""
    
    def __init__(self):
        self.lib = None
        self.dataset = None
        self.vm = None
        self.cache = None
        self.initialized = False
        self.lock = threading.Lock()
        
    def load_randomx_library(self) -> bool:
        """Load RandomX library with optimizations"""
        lib_paths = [
            '/usr/local/lib/librandomx.so',
            '/usr/lib/librandomx.so',  
            'librandomx.so'
        ]
        
        for lib_path in lib_paths:
            try:
                self.lib = ctypes.CDLL(lib_path)
                print(f"✅ Loaded RandomX library: {lib_path}")
                return True
            except OSError:
                continue
                
        print("❌ RandomX library not found!")
        return False
        
    def init_optimized(self, seed: bytes, use_large_pages=True, full_dataset=True, jit=True) -> bool:
        """Initialize with maximum performance settings"""
        if not self.load_randomx_library():
            return False
            
        try:
            # Set up function prototypes
            self.lib.randomx_alloc_cache.restype = ctypes.c_void_p
            self.lib.randomx_init_cache.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
            
            if full_dataset:
                self.lib.randomx_alloc_dataset.restype = ctypes.c_void_p
                self.lib.randomx_init_dataset.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong]
                
            self.lib.randomx_create_vm.restype = ctypes.c_void_p
            self.lib.randomx_calculate_hash.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p]
            
            # Flags for optimization
            flags = 0
            if jit:
                flags |= (1 << 3)  # RANDOMX_FLAG_JIT
            if use_large_pages:
                flags |= (1 << 1)  # RANDOMX_FLAG_LARGE_PAGES
            if full_dataset:
                flags |= (1 << 2)  # RANDOMX_FLAG_FULL_MEM
                
            print(f"🚀 Initializing RandomX with flags: {flags}")
            print(f"   - JIT: {'✅' if jit else '❌'}")
            print(f"   - Large Pages: {'✅' if use_large_pages else '❌'}")
            print(f"   - Full Dataset: {'✅' if full_dataset else '❌'}")
            
            # Allocate cache
            self.cache = self.lib.randomx_alloc_cache(flags)
            if not self.cache:
                print("❌ Failed to allocate RandomX cache")
                return False
                
            # Initialize cache with seed
            self.lib.randomx_init_cache(self.cache, seed, len(seed))
            print("✅ RandomX cache initialized")
            
            if full_dataset:
                # Allocate and initialize dataset for maximum speed
                print("🔄 Allocating full dataset (this may take a moment)...")
                self.dataset = self.lib.randomx_alloc_dataset(flags)
                if not self.dataset:
                    print("❌ Failed to allocate RandomX dataset")
                    return False
                    
                # Initialize dataset from cache (this is the slow part)
                dataset_count = 2097152  # Standard RandomX dataset size
                self.lib.randomx_init_dataset(self.dataset, self.cache, 0, dataset_count)
                print("✅ RandomX full dataset initialized!")
                
                # Create VM with dataset (faster)
                self.vm = self.lib.randomx_create_vm(flags, self.cache, self.dataset)
            else:
                # Create VM with cache only (slower but less memory)
                self.vm = self.lib.randomx_create_vm(flags, self.cache, None)
                
            if not self.vm:
                print("❌ Failed to create RandomX VM")
                return False
                
            print("🎯 RandomX VM created successfully!")
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"❌ RandomX initialization failed: {e}")
            return False
            
    def hash(self, data: bytes) -> bytes:
        """Calculate real RandomX hash"""
        if not self.initialized or not self.vm:
            raise RuntimeError("RandomX not initialized")
            
        output = (ctypes.c_char * 32)()
        
        with self.lock:
            self.lib.randomx_calculate_hash(self.vm, data, len(data), output)
            
        return bytes(output)
        
    def cleanup(self):
        """Clean up RandomX resources"""
        if self.vm:
            self.lib.randomx_destroy_vm(self.vm)
        if self.dataset:
            self.lib.randomx_release_dataset(self.dataset)
        if self.cache:
            self.lib.randomx_release_cache(self.cache)


class RealRandomXMiner:
    """Real RandomX Miner - Maximum Performance"""
    
    def __init__(self, num_threads: Optional[int] = None):
        self.num_threads = num_threads or min(psutil.cpu_count(logical=False), 8)
        self.engines = []
        self.running = False
        self.stats = {
            'total_hashes': 0,
            'start_time': None,
            'thread_stats': {}
        }
        
    def initialize_engines(self, seed: bytes = b'ZION_REAL_RANDOMX_2025') -> bool:
        """Initialize RandomX engines for each thread"""
        print(f"🚀 Initializing {self.num_threads} RandomX engines...")
        
        for i in range(self.num_threads):
            engine = OptimizedRandomXEngine()
            
            # First engine uses full dataset, others use cache-only for memory efficiency
            use_full_dataset = (i == 0)  
            
            if engine.init_optimized(seed, 
                                  use_large_pages=True,
                                  full_dataset=use_full_dataset, 
                                  jit=True):
                self.engines.append(engine)
                print(f"✅ Engine {i} initialized ({'dataset' if use_full_dataset else 'cache'})")
            else:
                print(f"❌ Engine {i} failed to initialize")
                return False
                
        print(f"🎯 All {len(self.engines)} engines ready!")
        return True
        
    def mining_worker(self, worker_id: int, duration: float):
        """Mining worker thread"""
        if worker_id >= len(self.engines):
            return
            
        engine = self.engines[worker_id]
        local_hashes = 0
        start_time = time.time()
        
        try:
            while self.running and (time.time() - start_time) < duration:
                # Create unique input for each hash
                nonce = int(time.time() * 1000000) + local_hashes
                input_data = f'ZION_REAL_{worker_id}_{nonce}'.encode()
                
                # Calculate real RandomX hash
                hash_result = engine.hash(input_data)
                local_hashes += 1
                
                # Update stats periodically
                if local_hashes % 10 == 0:
                    elapsed = time.time() - start_time
                    current_hashrate = local_hashes / elapsed
                    self.stats['thread_stats'][worker_id] = {
                        'hashes': local_hashes,
                        'hashrate': current_hashrate,
                        'elapsed': elapsed
                    }
                    
        except Exception as e:
            print(f"❌ Worker {worker_id} error: {e}")
            
        # Final stats
        elapsed = time.time() - start_time
        final_hashrate = local_hashes / elapsed if elapsed > 0 else 0
        
        self.stats['thread_stats'][worker_id] = {
            'hashes': local_hashes,
            'hashrate': final_hashrate, 
            'elapsed': elapsed
        }
        
        print(f"⚡ Worker {worker_id}: {local_hashes} hashes, {final_hashrate:.2f} H/s")
        
    def start_mining(self, duration: float = 30.0) -> dict:
        """Start real RandomX mining test"""
        print(f"🔥 Starting REAL RandomX mining for {duration} seconds...")
        print("=" * 60)
        
        self.running = True
        self.stats['start_time'] = time.time()
        self.stats['total_hashes'] = 0
        self.stats['thread_stats'] = {}
        
        # Start mining threads
        threads = []
        for i in range(len(self.engines)):
            thread = threading.Thread(target=self.mining_worker, args=(i, duration))
            threads.append(thread)
            thread.start()
            
        # Wait for completion
        for thread in threads:
            thread.join()
            
        self.running = False
        
        # Calculate final statistics
        total_hashes = sum(stats['hashes'] for stats in self.stats['thread_stats'].values())
        total_elapsed = time.time() - self.stats['start_time']
        total_hashrate = total_hashes / total_elapsed
        
        results = {
            'total_hashes': total_hashes,
            'total_time': total_elapsed,
            'total_hashrate': total_hashrate,
            'threads': len(self.engines),
            'per_thread_avg': total_hashrate / len(self.engines),
            'thread_details': self.stats['thread_stats']
        }
        
        print("=" * 60)
        print(f"🎯 REAL RandomX Mining Results:")
        print(f"   Total hashes: {total_hashes:,}")
        print(f"   Total time: {total_elapsed:.1f} seconds")
        print(f"   Total hashrate: {total_hashrate:.2f} H/s")
        print(f"   Threads used: {len(self.engines)}")
        print(f"   Per-thread avg: {total_hashrate / len(self.engines):.2f} H/s")
        print("=" * 60)
        print("💎 This is 100% REAL RandomX mining - NO SIMULATION!")
        
        return results
        
    def cleanup(self):
        """Clean up all engines"""
        for engine in self.engines:
            engine.cleanup()


if __name__ == "__main__":
    print("🔥 ZION Real RandomX Miner - Ultimate Performance Test")
    print("=====================================================")
    
    # Create optimized miner
    miner = RealRandomXMiner(num_threads=4)
    
    try:
        # Initialize engines
        if miner.initialize_engines():
            # Run mining test
            results = miner.start_mining(duration=20.0)
            
            # Performance analysis
            print(f"\n📊 Performance Analysis:")
            print(f"   CPU cores available: {psutil.cpu_count(logical=False)}")
            print(f"   Engines used: {results['threads']}")
            print(f"   Efficiency: {results['per_thread_avg']:.2f} H/s per core")
            
            if results['total_hashrate'] > 50:
                print(f"🚀 EXCELLENT performance for real RandomX!")
            elif results['total_hashrate'] > 20:
                print(f"✅ GOOD performance for real RandomX!")
            else:
                print(f"⚠️  Lower performance - consider optimization")
                
        else:
            print("❌ Failed to initialize RandomX engines")
            
    finally:
        miner.cleanup()
        print("\n🧹 Cleanup completed")