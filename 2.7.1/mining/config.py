#!/usr/bin/env python3
"""
ZION 2.7.1 ASIC-Resistant Mining Configuration
Supports multiple algorithms for flexibility while maintaining ASIC resistance
"""

import os
from typing import Dict, Any, Optional
from mining.algorithms import AlgorithmFactory, verify_asic_resistance

class MiningConfig:
    """
    Mining configuration with ASIC resistance enforcement
    """

    def __init__(self):
        self._config = self._load_default_config()

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default mining configuration"""
        return {
            # Primary ASIC-resistant algorithm
            'algorithm': 'argon2',
            'time_cost': 2,
            'memory_cost': 65536,  # 64MB
            'parallelism': 1,
            'hash_len': 32,

            # Difficulty and performance
            'difficulty': 0x0000FFFF,
            'block_time': 60,  # seconds
            'max_threads': 4,  # Allow up to 4 threads for ASIC resistance

            # Pool configuration
            'pool_enabled': True,
            'pool_host': 'localhost',
            'pool_ports': {
                'argon2': 3333,
                'kawpow': 3334,
                'ethash': 3335,
                'cryptonight': 3336,
                'octopus': 3337,
                'ergo': 3338
            },

            # ASIC resistance settings
            'asic_resistance_enforced': True,
            'allowed_algorithms': ['argon2', 'cryptonight', 'ergo', 'kawpow', 'ethash', 'octopus'],
            'blocked_algorithms': ['sha256', 'scrypt'],

            # GPU mining settings
            'gpu_enabled': False,
            'gpu_miner_path': 'SRBMiner-MULTI',
            'gpu_threads': 16,
            'gpu_worksize': 8,
            'gpu_intensity': 20
        }

    def get_mining_config(self) -> Dict[str, Any]:
        """Get complete mining configuration"""
        return self._config.copy()

    def get_algorithm_config(self, algorithm: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration for specific algorithm"""
        algo = algorithm or self._config['algorithm']

        if algo == 'argon2':
            return {
                'time_cost': self._config['time_cost'],
                'memory_cost': self._config['memory_cost'],
                'parallelism': self._config['parallelism'],
                'hash_len': self._config['hash_len']
            }
        elif algo in ['cryptonight', 'ergo']:
            return {
                'iterations': 75000,
                'hash_len': 32
            }
        elif algo in ['kawpow', 'ethash', 'octopus']:
            return {
                'external_miner': True,
                'pool_port': self._config['pool_ports'].get(algo, 3333)
            }
        else:
            raise ValueError(f"Unknown algorithm: {algo}")

    def set_algorithm(self, algorithm: str) -> bool:
        """
        Set mining algorithm with ASIC resistance verification

        Args:
            algorithm: Algorithm name to set

        Returns:
            True if algorithm was set successfully
        """
        algorithm = algorithm.lower()

        if algorithm not in self._config['allowed_algorithms']:
            print(f"❌ Algorithm {algorithm} not allowed")
            return False

        if self._config['asic_resistance_enforced']:
            if not verify_asic_resistance(algorithm):
                print(f"⚠️ WARNING: {algorithm} is not fully ASIC resistant!")
                print("💡 For maximum decentralization, use: argon2, cryptonight, or ergo")
                response = input("Continue anyway? (y/N): ").lower().strip()
                if response != 'y':
                    print("Algorithm change cancelled")
                    return False

        self._config['algorithm'] = algorithm
        print(f"✅ Mining algorithm set to: {algorithm}")
        return True

    def get_available_algorithms(self) -> Dict[str, str]:
        """Get list of available algorithms"""
        return AlgorithmFactory.get_available_algorithms()

    def get_algorithm_categories(self) -> Dict[str, list]:
        """Get algorithms grouped by category"""
        return AlgorithmFactory.get_algorithm_categories()

    def is_algorithm_allowed(self, algorithm: str) -> bool:
        """Check if algorithm is allowed"""
        return algorithm.lower() in self._config['allowed_algorithms']

    def is_algorithm_blocked(self, algorithm: str) -> bool:
        """Check if algorithm is blocked"""
        return algorithm.lower() in self._config['blocked_algorithms']

    def enable_gpu_mining(self, enabled: bool = True) -> None:
        """Enable or disable GPU mining"""
        self._config['gpu_enabled'] = enabled
        status = "enabled" if enabled else "disabled"
        print(f"🎮 GPU mining {status}")

    def get_pool_port(self, algorithm: str) -> int:
        """Get pool port for algorithm"""
        return self._config['pool_ports'].get(algorithm.lower(), 3333)

    def print_config_summary(self) -> None:
        """Print configuration summary"""
        print("🔧 ZION Mining Configuration:")
        print("=" * 40)
        print(f"Algorithm: {self._config['algorithm']}")
        print(f"ASIC Resistance: {'Enforced' if self._config['asic_resistance_enforced'] else 'Disabled'}")
        print(f"GPU Mining: {'Enabled' if self._config['gpu_enabled'] else 'Disabled'}")
        print(f"Pool Host: {self._config['pool_host']}")
        print(f"Block Time: {self._config['block_time']}s")
        print(f"Max Threads: {self._config['max_threads']}")

        print("\n📊 Available Algorithms:")
        categories = self.get_algorithm_categories()
        for category, algorithms in categories.items():
            print(f"  {category}: {', '.join(algorithms)}")

        print("\n🚫 Blocked Algorithms:")
        print(f"  {', '.join(self._config['blocked_algorithms'])}")


# Global configuration instance
_config_instance = None

def get_mining_config() -> MiningConfig:
    """Get global mining configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = MiningConfig()
    return _config_instance

def create_asic_resistant_algorithm():
    """Create ASIC-resistant algorithm instance"""
    config = get_mining_config()
    algo_config = config.get_algorithm_config()
    return AlgorithmFactory.create_algorithm(config._config['algorithm'], algo_config)

def benchmark_algorithms(iterations: int = 100) -> Dict[str, Any]:
    """Benchmark all available algorithms"""
    return AlgorithmFactory.benchmark_all_algorithms(iterations)

def print_asic_warning() -> None:
    """Print ASIC resistance warning"""
    print("\n⚠️  ASIC RESISTANCE WARNING ⚠️")
    print("ZION 2.7.1 prioritizes decentralization over maximum hashrate.")
    print("SHA256 and other ASIC-friendly algorithms are blocked.")
    print("For maximum decentralization, use Argon2, CryptoNight, or Ergo.")
    print("GPU-friendly algorithms (KawPow, Ethash, Octopus) are available")
    print("but offer less ASIC resistance than memory-hard algorithms.\n")

def switch_algorithm(algorithm: str) -> bool:
    """
    Switch mining algorithm

    Args:
        algorithm: New algorithm to use

    Returns:
        True if switch was successful
    """
    config = get_mining_config()
    return config.set_algorithm(algorithm)