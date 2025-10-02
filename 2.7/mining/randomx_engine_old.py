"""
Enhanced RandomX Engine for ZION 2.7 with 2.6.75 Integration
Based on proven ZION 2.6.75 RandomX implementation
Advanced performance monitoring, optimization flags, and proper error handling.
Falls back to SHA256 if RandomX unavailable.
"""
from __future__ import annotations
import ctypes
import hashlib
import os
import logging
import time
import threading
import psutil
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# RandomX flags (proven from 2.6.75)
RANDOMX_FLAG_DEFAULT = 0x0
RANDOMX_FLAG_LARGE_PAGES = 0x1
RANDOMX_FLAG_HARD_AES = 0x2
RANDOMX_FLAG_FULL_MEM = 0x4
RANDOMX_FLAG_JIT = 0x8
RANDOMX_FLAG_SECURE = 0x10
RANDOMX_FLAG_NTLM = 0x20
RANDOMX_FLAG_LARGE_PAGES_1GB = 0x40

# Library search paths (proven from 2.6.75)
LIB_NAMES = [
    '/media/maitreya/ZION1/ZION VOL 30.9/zion-2.6.75/librandomx.so',  # 2.6.75 build
    './librandomx.so',
    '../librandomx.so', 
    'librandomx.so',
    '/usr/local/lib/librandomx.so',
    '/usr/lib/librandomx.so',
]

class RandomXUnavailable(Exception):
    """Raised when RandomX library is not available"""
    pass

class RandomXPerformanceMonitor:
    """Monitors RandomX performance metrics (from 2.6.75)"""
    
    def __init__(self):
        self.hash_count = 0
        self.total_time = 0.0
        self.last_hash_time = 0.0
        self.init_time = 0.0
        
    def start_hash(self):
        self.last_hash_time = time.time()
        
    def end_hash(self):
        if self.last_hash_time > 0:
            duration = time.time() - self.last_hash_time
            self.total_time += duration
            self.hash_count += 1
            self.last_hash_time = 0.0
            
    def get_stats(self) -> Dict[str, Any]:
        if self.hash_count == 0:
            return {"hashrate": 0.0, "avg_time": 0.0, "total_hashes": 0}
            
        avg_time = self.total_time / self.hash_count
        hashrate = 1.0 / avg_time if avg_time > 0 else 0.0
        
        return {
            "hashrate": hashrate,
            "avg_time_ms": avg_time * 1000,
            "total_hashes": self.hash_count,
            "total_time": self.total_time,
            "init_time_ms": self.init_time * 1000,
        }

class RandomXEngine:
    """Enhanced RandomX Engine for ZION 2.7 (based on proven 2.6.75 implementation)"""
    
    def __init__(self, fallback_to_sha256: bool = True):
        self.lib = None
        self.cache = None
        self.dataset = None
        self.vm = None
        self.flags = 0
        self.initialized = False
        self.full_mem = False
        self.current_seed = None
        self.fallback_to_sha256 = fallback_to_sha256
        self.monitor = RandomXPerformanceMonitor()
        self.use_fallback = False

    def try_load_library(self) -> bool:
        """Try to load RandomX library (from 2.6.75)"""
        if self.lib:
            return True
            
        for lib_path in LIB_NAMES:
            try:
                logger.info(f"Trying to load RandomX library: {lib_path}")
                self.lib = ctypes.CDLL(lib_path)
                logger.info(f"Successfully loaded RandomX library: {lib_path}")
                return True
            except OSError as e:
                logger.debug(f"Failed to load {lib_path}: {e}")
                continue
                
        logger.warning("RandomX library not found in any standard location")
        return False
    
    def validate_library_functions(self) -> bool:
        """Validate that all required RandomX functions are available (from 2.6.75)"""
        required_functions = [
            'randomx_get_flags',
            'randomx_alloc_cache', 
            'randomx_init_cache',
            'randomx_create_vm',
            'randomx_vm_set_cache',
            'randomx_calculate_hash'
        ]
        
        missing = []
        for func_name in required_functions:
            if not hasattr(self.lib, func_name):
                missing.append(func_name)
                
        if missing:
            logger.error(f"RandomX library missing functions: {missing}")
            return False
            
        return True

    def init(self, seed: bytes, use_full_memory: bool = False) -> bool:
        """Enhanced initialization with memory optimization"""
        with self.lock:
            self.seed_key = seed
            
            if self.fallback:
                logger.info("Using SHA256 fallback mode")
                self.initialized = True
                return True
                
            # Enhanced function presence check
            required = [
                'randomx_alloc_cache', 'randomx_init_cache', 'randomx_create_vm',
                'randomx_vm_set_cache', 'randomx_calculate_hash', 'randomx_destroy_vm',
                'randomx_release_cache'
            ]
            
            for fn in required:
                if not hasattr(self.lib, fn):
                    logger.error(f"❌ Missing function {fn}, switching to fallback")
                    self.fallback = True
                    self.initialized = True
                    return True
                    
            try:
                # Setup function prototypes
                self.lib.randomx_alloc_cache.restype = ctypes.c_void_p
                self.lib.randomx_alloc_cache.argtypes = [ctypes.c_uint]
                self.lib.randomx_init_cache.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
                self.lib.randomx_create_vm.restype = ctypes.c_void_p
                self.lib.randomx_create_vm.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
                self.lib.randomx_calculate_hash.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p]
                
                # Allocate cache using ZION Virtual Memory system
                logger.info("Initializing RandomX cache with ZION Virtual Memory...")
                
                # Try RandomX cache allocation with graceful fallback
                logger.info("Initializing RandomX cache...")
                try:
                    self.cache = self.lib.randomx_alloc_cache(self.optimization_flags)
                    
                    if not self.cache:
                        logger.warning("RandomX cache allocation failed, switching to fallback")
                        raise RuntimeError("Cache allocation failed")
                    
                    self.lib.randomx_init_cache(self.cache, seed, len(seed))
                    logger.info("✅ RandomX cache initialized successfully")
                    
                except Exception as cache_error:
                    logger.warning(f"RandomX cache initialization failed: {cache_error}")
                    logger.info("Switching to SHA256 fallback mode")
                    self.fallback = True
                    self.initialized = True
                    return True
                
                # Optional: Initialize dataset for full memory mode
                if use_full_memory and hasattr(self.lib, 'randomx_alloc_dataset'):
                    logger.info("Initializing RandomX dataset (full memory mode)...")
                    self.lib.randomx_alloc_dataset.restype = ctypes.c_void_p
                    self.lib.randomx_alloc_dataset.argtypes = [ctypes.c_uint]
                    self.lib.randomx_init_dataset.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong]
                    
                    # Use standard RandomX allocation for dataset
                    self.dataset = self.lib.randomx_alloc_dataset(self.optimization_flags)
                    
                    if self.dataset:
                        # Initialize dataset in chunks for better progress tracking
                        dataset_size = 2147483648  # 2GB
                        chunk_size = 134217728     # 128MB chunks
                        chunks = dataset_size // chunk_size
                        
                        logger.info(f"Initializing dataset in {chunks} chunks...")
                        for i in range(chunks):
                            start_item = i * (chunk_size // 64)
                            item_count = chunk_size // 64
                            self.lib.randomx_init_dataset(self.dataset, self.cache, start_item, item_count)
                            if (i + 1) % 4 == 0:  # Progress every 4 chunks
                                logger.info(f"Dataset initialization: {((i+1)/chunks)*100:.1f}% complete")
                
                # Create VM with error handling
                logger.info("Creating RandomX VM...")
                try:
                    vm_flags = self.optimization_flags
                    self.vm = self.lib.randomx_create_vm(vm_flags, self.cache, self.dataset)
                    
                    if not self.vm:
                        logger.warning("RandomX VM creation failed, switching to fallback")
                        raise RuntimeError("VM creation failed")
                        
                    logger.info("✅ RandomX VM created successfully")
                    
                except Exception as vm_error:
                    logger.warning(f"RandomX VM creation failed: {vm_error}")
                    logger.info("Switching to SHA256 fallback mode")
                    self.fallback = True
                    self.cleanup()
                    self.initialized = True
                    return True
                
                self.initialized = True
                logger.info("✅ RandomX engine initialized successfully")
                return True
                
            except Exception as e:
                logger.error(f"❌ RandomX initialization failed: {e}")
                self.fallback = True
                self.cleanup()
                self.initialized = True
                return True

    def hash(self, data: bytes) -> bytes:
        """Enhanced hash calculation with performance monitoring"""
        if not self.initialized:
            raise RuntimeError("RandomXEngine not initialized")
            
        start_time = time.time()
        
        try:
            if self.fallback:
                result = hashlib.sha256(self.seed_key + data).digest()
            else:
                with self.lock:
                    output = (ctypes.c_ubyte * 32)()
                    self.lib.randomx_calculate_hash(self.vm, data, len(data), output)
                    result = bytes(output)
            
            # Update performance statistics
            hash_time = time.time() - start_time
            self.stats.update_hash_stats(hash_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Hash calculation failed: {e}")
            # Fallback on error
            if not self.fallback:
                logger.warning("Switching to SHA256 fallback due to error")
                self.fallback = True
            return hashlib.sha256(self.seed_key + data).digest()

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        current_time = time.time()
        session_duration = current_time - self.stats.session_start
        
        # Calculate average hashrate
        avg_hashrate = self.stats.total_hashes / session_duration if session_duration > 0 else 0
        
        # Get system resource usage
        try:
            process = psutil.Process()
            self.stats.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            self.stats.cpu_usage_percent = process.cpu_percent()
        except:
            pass
            
        return {
            'total_hashes': self.stats.total_hashes,
            'session_duration_seconds': session_duration,
            'average_hashrate': avg_hashrate,
            'peak_hashrate': self.stats.peak_hashrate,
            'last_hash_time_ms': self.stats.last_hash_time * 1000,
            'avg_hash_time_ms': self.stats.avg_hash_time * 1000,
            'memory_usage_mb': self.stats.memory_usage_mb,
            'cpu_usage_percent': self.stats.cpu_usage_percent,
            'randomx_available': self.stats.randomx_available,
            'fallback_mode': self.fallback,
            'optimization_flags': f"0x{self.optimization_flags:x}"
        }

    def cleanup(self):
        """Enhanced cleanup with proper resource management and ZION VM cleanup"""
        with self.lock:
            if self.vm and not self.fallback:
                try:
                    self.lib.randomx_destroy_vm(self.vm)
                    self.vm = None
                except:
                    pass
                    
            # Standard RandomX cleanup
            if self.dataset and not self.fallback:
                try:
                    self.lib.randomx_release_dataset(self.dataset)
                    self.dataset = None
                except:
                    pass
                    
            if self.cache and not self.fallback:
                try:
                    self.lib.randomx_release_cache(self.cache)
                    self.cache = None
                except:
                    pass
                    
        self.initialized = False
        logger.info("RandomX engine cleanup completed")

class MiningThreadManager:
    """Multi-threaded mining manager for optimal performance"""
    
    def __init__(self, num_threads: Optional[int] = None):
        self.num_threads = num_threads or min(psutil.cpu_count(logical=False), 8)
        self.engines = []
        self.running = False
        self.total_stats = {
            'total_hashes': 0,
            'start_time': None,
            'thread_stats': {}
        }
        
    def initialize_engines(self, seed: bytes) -> bool:
        """Initialize RandomX engines for each thread"""
        logger.info(f"Initializing {self.num_threads} mining threads...")
        
        for i in range(self.num_threads):
            engine = RandomXEngine(enable_optimizations=True)
            if engine.init(seed, use_full_memory=(i == 0)):  # Only first thread uses full memory
                self.engines.append(engine)
                logger.info(f"✅ Thread {i+1} initialized")
            else:
                logger.error(f"❌ Failed to initialize thread {i+1}")
                return False
                
        return len(self.engines) > 0
        
    def mining_worker(self, worker_id: int, target_hashes: int, results: list):
        """Mining worker thread"""
        if worker_id >= len(self.engines):
            return
            
        engine = self.engines[worker_id]
        hashes_done = 0
        
        while self.running and hashes_done < target_hashes:
            try:
                # Generate test data
                nonce = hashes_done.to_bytes(4, 'little')
                data = f"ZION_2.7_MINING_TEST_{worker_id}".encode() + nonce
                
                # Calculate hash
                hash_result = engine.hash(data)
                hashes_done += 1
                
                # Check for valid result (example: hash starts with zeros)
                if hash_result[:2] == b'\x00\x00':  # Simple difficulty check
                    results.append({
                        'worker_id': worker_id,
                        'nonce': hashes_done,
                        'hash': hash_result.hex(),
                        'data': data.hex()
                    })
                    
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                break
                
        self.total_stats['thread_stats'][worker_id] = {
            'hashes_completed': hashes_done,
            'performance': engine.get_performance_stats()
        }
        
    def start_mining_test(self, duration: float = 10.0) -> Dict[str, Any]:
        """Start multi-threaded mining test"""
        if not self.engines:
            raise RuntimeError("No engines initialized")
            
        import threading
        
        self.running = True
        self.total_stats['start_time'] = time.time()
        
        target_hashes = int(duration * 100)  # Approximate target
        results = []
        threads = []
        
        # Start worker threads
        for i in range(len(self.engines)):
            thread = threading.Thread(
                target=self.mining_worker,
                args=(i, target_hashes, results)
            )
            threads.append(thread)
            thread.start()
            
        # Wait for completion
        time.sleep(duration)
        self.running = False
        
        for thread in threads:
            thread.join(timeout=5.0)
            
        # Collect final statistics
        total_hashes = sum(
            stats.get('hashes_completed', 0) 
            for stats in self.total_stats['thread_stats'].values()
        )
        
        elapsed_time = time.time() - self.total_stats['start_time']
        avg_hashrate = total_hashes / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'total_hashes': total_hashes,
            'valid_results': len(results),
            'elapsed_time': elapsed_time,
            'average_hashrate': avg_hashrate,
            'thread_count': len(self.engines),
            'thread_stats': self.total_stats['thread_stats'],
            'results': results[:10]  # First 10 results only
        }
        
    def cleanup(self):
        """Cleanup all engines"""
        self.running = False
        for engine in self.engines:
            engine.cleanup()
        self.engines.clear()

if __name__ == '__main__':
    # Enhanced testing
    print("🚀 ZION 2.7 Enhanced RandomX Engine Test")
    print("=" * 50)
    
    # Single engine test
    rx = RandomXEngine(enable_optimizations=True)
    rx.init(b'ZION_2_7_SEED')
    h = rx.hash(b'test')
    print(f"Single hash: {h.hex()}")
    print(f"Performance: {rx.get_performance_stats()}")
    
    # Multi-threaded mining test
    print("\n⚡ Multi-threaded mining test...")
    manager = MiningThreadManager(num_threads=4)
    
    if manager.initialize_engines(b'ZION_2_7_MINING_SEED'):
        results = manager.start_mining_test(duration=5.0)
        
        print(f"Total hashes: {results['total_hashes']}")
        print(f"Average hashrate: {results['average_hashrate']:.2f} H/s")
        print(f"Valid results found: {results['valid_results']}")
        print(f"Thread count: {results['thread_count']}")
        
        manager.cleanup()
    
    rx.cleanup()
    print("✅ Test completed")
