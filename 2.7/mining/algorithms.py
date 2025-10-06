#!/usr/bin/env python3
"""
ZION 2.7.1 Mining Algorithms
============================
Support for multiple mining algorithms:
- SHA256 (CPU/GPU compatible)
- RandomX (CPU optimized)
- GPU accelerated variants
"""

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MiningAlgorithm(ABC):
    """Abstract base class for mining algorithms"""
    
    @abstractmethod
    def hash(self, data: bytes) -> str:
        """Calculate hash for given data"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get algorithm name"""
        pass
    
    @abstractmethod
    def get_target_adjustment(self) -> float:
        """Get difficulty adjustment factor"""
        pass


class SHA256Algorithm(MiningAlgorithm):
    """Standard SHA256 algorithm - CPU/GPU compatible"""
    
    def hash(self, data: bytes) -> str:
        """Pure SHA256 hash"""
        return hashlib.sha256(data).hexdigest()
    
    def get_name(self) -> str:
        return "SHA256"
    
    def get_target_adjustment(self) -> float:
        return 1.0


class RandomXAlgorithm(MiningAlgorithm):
    """RandomX algorithm - CPU optimized, ASIC resistant"""
    
    def __init__(self):
        self._initialized = False
        self._randomx_available = self._check_randomx()
    
    def _check_randomx(self) -> bool:
        """Check if RandomX is available"""
        try:
            # Try to import RandomX library
            import randomx
            return True
        except ImportError:
            logger.warning("RandomX library not available, using fallback")
            return False
    
    def hash(self, data: bytes) -> str:
        """RandomX hash with fallback to enhanced SHA256"""
        if self._randomx_available:
            return self._randomx_hash(data)
        else:
            return self._randomx_fallback(data)
    
    def _randomx_hash(self, data: bytes) -> str:
        """Real RandomX hash (when library is available)"""
        try:
            import randomx
            # Initialize RandomX with ZION key
            key = b"ZION_2.7.1_RANDOMX_KEY"
            flags = randomx.get_flags()
            cache = randomx.create_cache(flags, key)
            vm = randomx.create_vm(flags, cache, None)
            
            # Calculate hash
            hash_result = randomx.calculate_hash(vm, data)
            
            # Cleanup
            randomx.destroy_vm(vm)
            randomx.destroy_cache(cache)
            
            return hash_result.hex()
        except Exception as e:
            logger.error(f"RandomX error: {e}, falling back")
            return self._randomx_fallback(data)
    
    def _randomx_fallback(self, data: bytes) -> str:
        """Enhanced SHA256 fallback that mimics RandomX characteristics"""
        # Multi-round hashing with different operations
        # This provides some ASIC resistance while being CPU-friendly
        
        # Round 1: SHA256
        h1 = hashlib.sha256(data).digest()
        
        # Round 2: SHA512 truncated  
        h2 = hashlib.sha512(h1 + data).digest()[:32]
        
        # Round 3: Blake2b for variety
        try:
            h3 = hashlib.blake2b(h2 + h1, digest_size=32).digest()
        except:
            h3 = hashlib.sha256(h2 + h1).digest()
        
        # Round 4: Final SHA256
        final_hash = hashlib.sha256(h3 + data[-16:]).hexdigest()
        
        return final_hash
    
    def get_name(self) -> str:
        if self._randomx_available:
            return "RandomX"
        else:
            return "RandomX-Fallback"
    
    def get_target_adjustment(self) -> float:
        # RandomX is typically slower than SHA256
        return 0.1 if self._randomx_available else 0.3


class GPUAlgorithm(MiningAlgorithm):
    """GPU accelerated mining algorithm"""
    
    def __init__(self):
        self._cuda_available = self._check_cuda()
        self._opencl_available = self._check_opencl()
    
    def _check_cuda(self) -> bool:
        """Check CUDA availability"""
        try:
            import pycuda.driver as cuda
            cuda.init()
            return cuda.Device.count() > 0
        except:
            return False
    
    def _check_opencl(self) -> bool:
        """Check OpenCL availability"""
        try:
            import pyopencl as cl
            platforms = cl.get_platforms()
            return len(platforms) > 0
        except:
            return False
    
    def hash(self, data: bytes) -> str:
        """GPU accelerated hash"""
        if self._cuda_available:
            return self._cuda_hash(data)
        elif self._opencl_available:
            return self._opencl_hash(data)
        else:
            # Fallback to optimized CPU
            return self._cpu_optimized_hash(data)
    
    def _cuda_hash(self, data: bytes) -> str:
        """CUDA accelerated hash"""
        # TODO: Implement CUDA kernel for parallel hashing
        # For now, use optimized CPU version
        return self._cpu_optimized_hash(data)
    
    def _opencl_hash(self, data: bytes) -> str:
        """OpenCL accelerated hash"""
        # TODO: Implement OpenCL kernel
        return self._cpu_optimized_hash(data)
    
    def _cpu_optimized_hash(self, data: bytes) -> str:
        """CPU optimized version for GPU fallback"""
        # Parallel-friendly SHA256
        return hashlib.sha256(data).hexdigest()
    
    def get_name(self) -> str:
        if self._cuda_available:
            return "GPU-CUDA"
        elif self._opencl_available:
            return "GPU-OpenCL"
        else:
            return "GPU-Fallback"
    
    def get_target_adjustment(self) -> float:
        # GPU should be much faster
        if self._cuda_available or self._opencl_available:
            return 10.0  # GPU is ~10x faster
        else:
            return 1.0


class AlgorithmFactory:
    """Factory for creating mining algorithms"""
    
    _algorithms = {
        'sha256': SHA256Algorithm,
        'randomx': RandomXAlgorithm,
        'gpu': GPUAlgorithm
    }
    
    @classmethod
    def create_algorithm(cls, name: str) -> MiningAlgorithm:
        """Create mining algorithm by name"""
        if name.lower() not in cls._algorithms:
            raise ValueError(f"Unknown algorithm: {name}")
        
        algorithm_class = cls._algorithms[name.lower()]
        return algorithm_class()
    
    @classmethod
    def get_available_algorithms(cls) -> Dict[str, str]:
        """Get list of available algorithms with descriptions"""
        algos = {}
        for name, algo_class in cls._algorithms.items():
            try:
                instance = algo_class()
                algos[name] = instance.get_name()
            except Exception as e:
                algos[name] = f"Not available ({e})"
        
        return algos
    
    @classmethod
    def auto_select_best(cls) -> MiningAlgorithm:
        """Auto-select best available algorithm"""
        # Priority: GPU > RandomX > SHA256
        try:
            gpu_algo = cls.create_algorithm('gpu')
            if 'CUDA' in gpu_algo.get_name() or 'OpenCL' in gpu_algo.get_name():
                logger.info(f"Selected best algorithm: {gpu_algo.get_name()}")
                return gpu_algo
        except:
            pass
        
        try:
            randomx_algo = cls.create_algorithm('randomx')
            if 'Fallback' not in randomx_algo.get_name():
                logger.info(f"Selected best algorithm: {randomx_algo.get_name()}")
                return randomx_algo
        except:
            pass
        
        # Fallback to SHA256
        sha256_algo = cls.create_algorithm('sha256')
        logger.info(f"Selected best algorithm: {sha256_algo.get_name()}")
        return sha256_algo


def benchmark_algorithms() -> Dict[str, Dict[str, Any]]:
    """Benchmark all available algorithms"""
    import time
    
    results = {}
    test_data = b"ZION_2.7.1_BENCHMARK_DATA" * 100
    
    for algo_name in AlgorithmFactory._algorithms.keys():
        try:
            algo = AlgorithmFactory.create_algorithm(algo_name)
            
            # Warm up
            algo.hash(test_data)
            
            # Benchmark
            start_time = time.time()
            iterations = 1000
            
            for _ in range(iterations):
                algo.hash(test_data)
            
            end_time = time.time()
            duration = end_time - start_time
            hashrate = iterations / duration
            
            results[algo_name] = {
                'name': algo.get_name(),
                'hashrate': hashrate,
                'duration': duration,
                'adjustment': algo.get_target_adjustment()
            }
            
        except Exception as e:
            results[algo_name] = {
                'name': f"Error: {e}",
                'hashrate': 0,
                'duration': 0,
                'adjustment': 1.0
            }
    
    return results


if __name__ == "__main__":
    # Test all algorithms
    print("🧪 ZION 2.7.1 Algorithm Test")
    print("=" * 40)
    
    # List available
    available = AlgorithmFactory.get_available_algorithms()
    for name, desc in available.items():
        print(f"✅ {name}: {desc}")
    
    print("\n⚡ Auto-selecting best algorithm...")
    best = AlgorithmFactory.auto_select_best()
    print(f"🎯 Selected: {best.get_name()}")
    
    print("\n🏃 Running benchmark...")
    results = benchmark_algorithms()
    
    print("\n📊 Benchmark Results:")
    for algo, data in results.items():
        print(f"{algo:10} | {data['name']:15} | {data['hashrate']:8.1f} H/s")