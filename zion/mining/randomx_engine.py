"""
ZION RandomX Mining Engine v2.6.75

Enhanced RandomX wrapper with improved performance, monitoring,
and real blockchain integration. Eliminates all mockups.
"""
from __future__ import annotations
import ctypes
import hashlib
import os
import time
from typing import Optional, Dict, Any
import logging

# Configure logging
logger = logging.getLogger(__name__)

# RandomX flags (same as original but documented)
RANDOMX_FLAG_DEFAULT = 0x0
RANDOMX_FLAG_LARGE_PAGES = 0x1
RANDOMX_FLAG_HARD_AES = 0x2
RANDOMX_FLAG_FULL_MEM = 0x4
RANDOMX_FLAG_JIT = 0x8
RANDOMX_FLAG_SECURE = 0x10
RANDOMX_FLAG_NTLM = 0x20
RANDOMX_FLAG_LARGE_PAGES_1GB = 0x40

# Library search paths
LIB_NAMES = [
    '/media/maitreya/ZION1/zion-2.6.75/librandomx.so',  # Local build
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
    """Monitors RandomX performance metrics"""
    
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
    """
    Enhanced RandomX engine for ZION 2.6.75
    
    Features:
    - Performance monitoring
    - Graceful fallback to SHA256 when RandomX unavailable
    - Real seed management (no mockups)
    - Memory optimization
    - Error handling and logging
    """
    
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
        """Try to load RandomX library"""
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
        """Validate that all required RandomX functions are available"""
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
    
    def init(self, seed_key: bytes, use_large_pages: bool = True, 
             full_mem: bool = True, jit: bool = True) -> bool:
        """
        Initialize RandomX engine with given seed
        
        Args:
            seed_key: Blockchain seed for RandomX initialization
            use_large_pages: Use large pages for performance
            full_mem: Use full memory dataset (vs cache-only)
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
        
        # Build flags
        flags = RANDOMX_FLAG_DEFAULT
        if use_large_pages:
            flags |= RANDOMX_FLAG_LARGE_PAGES
        if full_mem:
            flags |= RANDOMX_FLAG_FULL_MEM  
        if jit:
            flags |= RANDOMX_FLAG_JIT
            
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
        """
        Calculate hash of input data
        
        Args:
            data: Input data to hash
            
        Returns:
            32-byte hash result
            
        Raises:
            RandomXUnavailable: If engine not initialized
        """
        if not self.initialized:
            raise RandomXUnavailable("RandomX engine not initialized")
        
        self.monitor.start_hash()
        
        try:
            if self.use_fallback:
                # SHA256 fallback for compatibility
                result = hashlib.sha256(data).digest()
            else:
                # Real RandomX hash
                output_buffer = ctypes.create_string_buffer(32)
                input_buffer = ctypes.create_string_buffer(data)
                
                self.lib.randomx_calculate_hash(
                    self.vm, input_buffer, ctypes.c_size_t(len(data)), output_buffer
                )
                
                result = output_buffer.raw
                
            self.monitor.end_hash()
            return result
            
        except Exception as e:
            logger.error(f"Hash calculation failed: {e}")
            self.monitor.end_hash()
            raise
    
    def reinit_on_new_seed(self, seed_key: bytes) -> bool:
        """
        Reinitialize engine with new seed (for new blockchain epochs)
        
        Args:
            seed_key: New seed for RandomX
            
        Returns:
            True if reinitialization successful
        """
        if not self.initialized:
            return self.init(seed_key)
            
        if seed_key == self.current_seed:
            return True  # No change needed
            
        logger.info("Reinitializing RandomX with new seed")
        
        if self.use_fallback:
            # Just update seed for fallback mode
            self.current_seed = seed_key
            return True
        
        try:
            # Reinitialize cache with new seed
            init_cache = self.lib.randomx_init_cache
            init_cache.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
            seed_buffer = ctypes.create_string_buffer(seed_key)
            init_cache(self.cache, seed_buffer, ctypes.c_size_t(len(seed_key)))
            
            # Rebuild dataset if present
            if self.dataset:
                if (hasattr(self.lib, 'randomx_init_dataset') and 
                    hasattr(self.lib, 'randomx_dataset_item_count')):
                    
                    item_count_fn = self.lib.randomx_dataset_item_count
                    item_count_fn.restype = ctypes.c_size_t
                    total_items = item_count_fn()
                    
                    init_dataset = self.lib.randomx_init_dataset
                    init_dataset.argtypes = [ctypes.c_void_p, ctypes.c_void_p, 
                                           ctypes.c_void_p, ctypes.c_size_t]
                    init_dataset(self.dataset, self.cache, None, 
                               ctypes.c_size_t(total_items))
            
            self.current_seed = seed_key
            logger.info("RandomX reinitialized with new seed")
            return True
            
        except Exception as e:
            logger.error(f"RandomX reinitialization failed: {e}")
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance monitoring statistics"""
        stats = self.monitor.get_stats()
        stats.update({
            "engine_type": "SHA256_fallback" if self.use_fallback else "RandomX",
            "full_mem": self.full_mem and not self.use_fallback,
            "large_pages": bool(self.flags & RANDOMX_FLAG_LARGE_PAGES) and not self.use_fallback,
            "jit_enabled": bool(self.flags & RANDOMX_FLAG_JIT) and not self.use_fallback,
        })
        return stats
    
    def is_hardware_accelerated(self) -> bool:
        """Check if using real RandomX hardware acceleration"""
        return self.initialized and not self.use_fallback
    
    def cleanup(self):
        """Clean up allocated resources"""
        # Note: In production, should implement proper cleanup
        # For now, let OS handle cleanup on process exit
        pass


# Test harness
if __name__ == '__main__':
    import json
    
    print("ZION RandomX Engine v2.6.75 Test")
    print("=" * 40)
    
    engine = RandomXEngine(fallback_to_sha256=True)
    
    test_seed = b'ZionTestSeed2675'
    print(f"Initializing with test seed: {test_seed}")
    
    # Try with different configurations
    if engine.init(test_seed, use_large_pages=False, full_mem=False):
        print("✅ Engine initialized successfully (cache-only mode)")
        print(f"Hardware accelerated: {engine.is_hardware_accelerated()}")
        
        # Test hashing
        test_data = b'ZION blockchain test data'
        hash_result = engine.hash(test_data)
        print(f"Test hash: {hash_result.hex()}")
        
        # Performance test
        print("\nPerformance test (100 hashes)...")
        for i in range(100):
            test_input = f"test_data_{i}".encode()
            engine.hash(test_input)
        
        stats = engine.get_performance_stats()
        print("\nPerformance Statistics:")
        print(json.dumps(stats, indent=2))
        
    else:
        print("❌ Engine initialization failed")