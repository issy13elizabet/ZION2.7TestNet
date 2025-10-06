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

    def init(self, seed_key: bytes, use_large_pages: bool = False,  # Changed default to False
             full_mem: bool = False, jit: bool = True) -> bool:      # Changed default to False
        """
        Initialize RandomX engine with given seed (proven 2.6.75 implementation)
        
        Args:
            seed_key: Blockchain seed for RandomX initialization
            use_large_pages: Use large pages for performance (disabled by default)
            full_mem: Use full memory dataset (disabled by default)
            jit: Enable JIT compilation
            
        Returns:
            True if initialization successful, False otherwise
        """
        init_start = time.time()
        
        # Try to load RandomX library
        if not self.try_load_library():
            if self.fallback_to_sha256:
                logger.warning("RandomX unavailable, using SHA256 fallback")
                self.use_fallback = True
                self.initialized = True
                self.monitor.init_time = time.time() - init_start
                return True
            else:
                logger.error("RandomX required but unavailable")
                return False
        
        # Validate library functions
        if not self.validate_library_functions():
            if self.fallback_to_sha256:
                logger.warning("RandomX library invalid, using SHA256 fallback")
                self.use_fallback = True
                self.initialized = True
                self.monitor.init_time = time.time() - init_start
                return True
            else:
                return False
        
        # Build flags (conservative approach)
        flags = RANDOMX_FLAG_DEFAULT
        if jit:
            flags |= RANDOMX_FLAG_JIT
        # Only add large pages if explicitly requested and they work
        if use_large_pages:
            # Test large pages first
            try:
                test_cache = self.lib.randomx_alloc_cache(RANDOMX_FLAG_LARGE_PAGES)
                if test_cache:
                    self.lib.randomx_release_cache(test_cache)
                    flags |= RANDOMX_FLAG_LARGE_PAGES
                    logger.info("✅ Large pages enabled")
                else:
                    logger.warning("⚠️ Large pages requested but not working")
            except:
                logger.warning("⚠️ Large pages test failed")
        
        if full_mem:
            flags |= RANDOMX_FLAG_FULL_MEM  
            
        self.flags = flags
        self.full_mem = full_mem
        
        try:
            # Allocate cache
            logger.info("Allocating RandomX cache...")
            alloc_cache = self.lib.randomx_alloc_cache
            alloc_cache.restype = ctypes.c_void_p
            cache_ptr = alloc_cache(flags)
            
            if not cache_ptr:
                logger.error("Failed to allocate RandomX cache")
                if self.fallback_to_sha256:
                    logger.warning("Falling back to SHA256")
                    self.use_fallback = True
                    self.initialized = True
                    self.monitor.init_time = time.time() - init_start
                    return True
                return False
                
            self.cache = cache_ptr
            
            # Initialize cache with seed
            logger.info(f"Initializing RandomX cache with seed length {len(seed_key)}")
            init_cache = self.lib.randomx_init_cache
            init_cache.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
            seed_buffer = ctypes.create_string_buffer(seed_key)
            init_cache(self.cache, seed_buffer, ctypes.c_size_t(len(seed_key)))
            
            # Try to allocate dataset if full memory requested
            if full_mem and hasattr(self.lib, 'randomx_alloc_dataset'):
                try:
                    logger.info("Allocating RandomX dataset...")
                    alloc_dataset = self.lib.randomx_alloc_dataset
                    alloc_dataset.restype = ctypes.c_void_p
                    dataset_ptr = alloc_dataset(flags)
                    
                    if dataset_ptr:
                        self.dataset = dataset_ptr
                        logger.info("Dataset allocated, initializing...")
                        
                        # Initialize dataset 
                        if (hasattr(self.lib, 'randomx_dataset_item_count') and 
                            hasattr(self.lib, 'randomx_init_dataset')):
                            
                            item_count_fn = self.lib.randomx_dataset_item_count
                            item_count_fn.restype = ctypes.c_size_t
                            total_items = item_count_fn()
                            
                            init_dataset = self.lib.randomx_init_dataset
                            init_dataset.argtypes = [ctypes.c_void_p, ctypes.c_void_p, 
                                                   ctypes.c_void_p, ctypes.c_size_t]
                            init_dataset(self.dataset, self.cache, None, 
                                       ctypes.c_size_t(total_items))
                            logger.info(f"Dataset initialized with {total_items} items")
                        else:
                            logger.warning("Dataset functions not available")
                    else:
                        logger.warning("Failed to allocate dataset, using cache-only")
                        
                except Exception as e:
                    logger.warning(f"Dataset allocation failed: {e}, using cache-only")
                    self.dataset = None
            
            # Create VM
            logger.info("Creating RandomX VM...")
            create_vm = self.lib.randomx_create_vm
            create_vm.restype = ctypes.c_void_p
            create_vm.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
            
            vm_ptr = create_vm(flags, self.cache, 
                             self.dataset if self.dataset else None)
            
            if not vm_ptr:
                logger.error("Failed to create RandomX VM")
                if self.fallback_to_sha256:
                    logger.warning("Falling back to SHA256")
                    self.use_fallback = True
                    self.initialized = True
                    self.monitor.init_time = time.time() - init_start
                    return True
                return False
                
            self.vm = vm_ptr
            
            # Setup hash function signature
            self.lib.randomx_calculate_hash.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p
            ]
            
            self.current_seed = seed_key
            self.initialized = True
            self.use_fallback = False
            
            self.monitor.init_time = time.time() - init_start
            
            logger.info(f"RandomX initialized successfully in {self.monitor.init_time:.3f}s")
            logger.info(f"Using {'dataset' if self.dataset else 'cache-only'} mode")
            
            return True
            
        except Exception as e:
            logger.error(f"RandomX initialization failed: {e}")
            if self.fallback_to_sha256:
                logger.warning("Falling back to SHA256")
                self.use_fallback = True
                self.initialized = True
                self.monitor.init_time = time.time() - init_start
                return True
            return False

    def hash(self, data: bytes) -> bytes:
        """Calculate hash of input data (from 2.6.75)"""
        if not self.initialized:
            raise RuntimeError("RandomX engine not initialized")
        
        self.monitor.start_hash()
        
        try:
            if self.use_fallback:
                # SHA256 fallback
                if self.current_seed:
                    result = hashlib.sha256(self.current_seed + data).digest()
                else:
                    result = hashlib.sha256(data).digest()
            else:
                # Real RandomX hash
                output = (ctypes.c_ubyte * 32)()
                self.lib.randomx_calculate_hash(
                    self.vm, data, len(data), ctypes.cast(output, ctypes.c_void_p)
                )
                result = bytes(output)
                
            self.monitor.end_hash()
            return result
            
        except Exception as e:
            logger.error(f"Hash calculation failed: {e}")
            # Emergency fallback
            if self.current_seed:
                return hashlib.sha256(self.current_seed + data).digest()
            else:
                return hashlib.sha256(data).digest()

    def reinit_on_new_seed(self, seed_key: bytes) -> bool:
        """Reinitialize with new seed if changed (from 2.6.75)"""
        if seed_key == self.current_seed:
            return False
            
        if self.use_fallback:
            self.current_seed = seed_key
            return True
            
        if self.cache and self.lib:
            try:
                init_cache = self.lib.randomx_init_cache
                init_cache.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
                seed_buffer = ctypes.create_string_buffer(seed_key)
                init_cache(self.cache, seed_buffer, ctypes.c_size_t(len(seed_key)))
                
                self.current_seed = seed_key
                logger.info(f"RandomX cache reinitialized with new seed")
                return True
                
            except Exception as e:
                logger.error(f"Failed to reinitialize with new seed: {e}")
                
        return False

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance monitoring statistics (enhanced for 2.7)"""
        base_stats = self.monitor.get_stats()
        
        # Add system resource usage
        try:
            process = psutil.Process()
            base_stats.update({
                'memory_usage_mb': process.memory_info().rss / 1024 / 1024,
                'cpu_usage_percent': process.cpu_percent(),
                'randomx_available': not self.use_fallback,
                'fallback_mode': self.use_fallback,
                'current_flags': f"0x{self.flags:x}",
                'full_memory_mode': self.full_mem
            })
        except:
            base_stats.update({
                'memory_usage_mb': 0.0,
                'cpu_usage_percent': 0.0,
                'randomx_available': not self.use_fallback,
                'fallback_mode': self.use_fallback,
                'current_flags': f"0x{self.flags:x}",
                'full_memory_mode': self.full_mem
            })
            
        return base_stats

    def cleanup(self):
        """Clean up RandomX resources (from 2.6.75)"""
        if self.vm and not self.use_fallback:
            try:
                if hasattr(self.lib, 'randomx_destroy_vm'):
                    self.lib.randomx_destroy_vm(self.vm)
                self.vm = None
            except:
                pass
                
        if self.dataset and not self.use_fallback:
            try:
                if hasattr(self.lib, 'randomx_release_dataset'):
                    self.lib.randomx_release_dataset(self.dataset)
                self.dataset = None
            except:
                pass
                
        if self.cache and not self.use_fallback:
            try:
                if hasattr(self.lib, 'randomx_release_cache'):
                    self.lib.randomx_release_cache(self.cache)
                self.cache = None
            except:
                pass
                
        self.initialized = False
        logger.info("RandomX engine cleanup completed")

# Multi-threaded mining manager (simplified for 2.7)
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
            engine = RandomXEngine(fallback_to_sha256=True)
            # Use conservative settings to avoid allocation failures
            if engine.init(seed, use_large_pages=False, full_mem=False):
                self.engines.append(engine)
                logger.info(f"✅ Thread {i+1} initialized (fallback: {engine.use_fallback})")
            else:
                logger.error(f"❌ Failed to initialize thread {i+1}")
                return False
                
        return len(self.engines) > 0
        
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
        
        def mining_worker(worker_id: int, target_hashes: int, results: list):
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
        
        # Start worker threads
        for i in range(len(self.engines)):
            thread = threading.Thread(
                target=mining_worker,
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
    rx = RandomXEngine(fallback_to_sha256=True)
    rx.init(b'ZION_2_7_SEED')
    h = rx.hash(b'test')
    print(f"Single hash: {h.hex()}")
    print(f"Performance: {rx.get_performance_stats()}")
    
    # Multi-threaded mining test
    print("\n⚡ Multi-threaded mining test...")
    manager = MiningThreadManager(num_threads=2)
    
    if manager.initialize_engines(b'ZION_2_7_MINING_SEED'):
        results = manager.start_mining_test(duration=5.0)
        
        print(f"Total hashes: {results['total_hashes']}")
        print(f"Average hashrate: {results['average_hashrate']:.2f} H/s")
        print(f"Valid results found: {results['valid_results']}")
        print(f"Thread count: {results['thread_count']}")
        
        manager.cleanup()
    
    rx.cleanup()
    print("✅ Test completed")