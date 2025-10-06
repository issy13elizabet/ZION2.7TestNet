#!/usr/bin/env python3
"""
ZION 2.7.1 Mining Configuration
Algorithm selection and configuration utilities
"""

import logging
from typing import Dict, Any, Optional
from mining.algorithms import AlgorithmFactory, MiningAlgorithm

logger = logging.getLogger(__name__)


class MiningConfig:
    """Global mining configuration"""
    
    def __init__(self):
        self._current_algorithm: Optional[MiningAlgorithm] = None
        self._algorithm_name: str = "auto"
    
    def set_algorithm(self, algorithm_name: str) -> None:
        """Set global mining algorithm"""
        self._algorithm_name = algorithm_name
        if algorithm_name == "auto":
            self._current_algorithm = AlgorithmFactory.auto_select_best()
        else:
            self._current_algorithm = AlgorithmFactory.create_algorithm(algorithm_name)
        
        logger.info(f"Set global algorithm: {self._current_algorithm.get_name()}")
    
    def get_algorithm(self) -> MiningAlgorithm:
        """Get current algorithm"""
        if self._current_algorithm is None:
            self.set_algorithm("auto")
        return self._current_algorithm
    
    def get_algorithm_name(self) -> str:
        """Get current algorithm name"""
        return self._algorithm_name
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get detailed algorithm information"""
        algo = self.get_algorithm()
        return {
            'name': algo.get_name(),
            'configured': self._algorithm_name,
            'adjustment': algo.get_target_adjustment()
        }


# Global instance
_global_config = MiningConfig()


def set_global_algorithm(algorithm_name: str) -> None:
    """Set global mining algorithm"""
    _global_config.set_algorithm(algorithm_name)


def get_global_algorithm() -> MiningAlgorithm:
    """Get global mining algorithm"""
    return _global_config.get_algorithm()


def get_algorithm_info() -> Dict[str, Any]:
    """Get current algorithm info"""
    return _global_config.get_algorithm_info()


def list_available_algorithms() -> Dict[str, str]:
    """List all available algorithms"""
    return AlgorithmFactory.get_available_algorithms()


def benchmark_all_algorithms() -> Dict[str, Dict[str, Any]]:
    """Benchmark all algorithms"""
    from mining.algorithms import benchmark_algorithms
    return benchmark_algorithms()


if __name__ == "__main__":
    print("🔧 ZION 2.7.1 Mining Configuration")
    print("=" * 40)
    
    print("\n📋 Available Algorithms:")
    algos = list_available_algorithms()
    for name, desc in algos.items():
        print(f"  {name:10} - {desc}")
    
    print("\n⚡ Current Configuration:")
    info = get_algorithm_info()
    print(f"  Algorithm: {info['name']}")
    print(f"  Configured: {info['configured']}")
    print(f"  Adjustment: {info['adjustment']}")
    
    print("\n🏃 Running Benchmark...")
    results = benchmark_all_algorithms()
    print("\n📊 Results:")
    for name, data in results.items():
        print(f"  {name:10} | {data['hashrate']:8.1f} H/s | {data['name']}")