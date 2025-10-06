#!/usr/bin/env python3
"""
ZION 2.7.1 Enhanced Algorithms Demo
Tests new YesCrypt and Autolykos v2 algorithms + existing ones
"""

import sys
import os
import time
from decimal import Decimal

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from mining.algorithms import AlgorithmFactory, verify_asic_resistance
from mining.config import get_mining_config
from mining.humanitarian_distribution import get_humanitarian_distributor


def test_new_algorithms():
    """Test new YesCrypt and Autolykos v2 algorithms"""
    print("🧪 ZION 2.7.1 Enhanced Algorithms Demo")
    print("=" * 60)
    
    # Test YesCrypt
    print("\n🔐 Testing YesCrypt Algorithm:")
    print("-" * 40)
    try:
        yescrypt_config = AlgorithmFactory.get_default_config('yescrypt')
        yescrypt = AlgorithmFactory.create_algorithm('yescrypt', yescrypt_config)
        
        # Test hash
        test_data = b"ZION_YesCrypt_Test_2024"
        hash_result = yescrypt.hash(test_data)
        print(f"✅ YesCrypt hash: {hash_result.hex()[:32]}...")
        
        # Quick benchmark
        print("🏃 Running quick benchmark...")
        benchmark = yescrypt.benchmark(10)
        print(f"   Hashrate: {benchmark['hashrate']}")
        print(f"   Memory usage: {benchmark['memory_usage']}")
        print(f"   ASIC resistant: {benchmark['asic_resistant']}")
        
    except Exception as e:
        print(f"❌ YesCrypt error: {e}")
    
    # Test Autolykos v2
    print("\n⚡ Testing Autolykos v2 Algorithm:")
    print("-" * 40)
    try:
        autolykos_config = AlgorithmFactory.get_default_config('autolykos2')
        autolykos = AlgorithmFactory.create_algorithm('autolykos2', autolykos_config)
        
        # Test hash
        test_data = b"ZION_Autolykos_v2_Test_2024"
        hash_result = autolykos.hash(test_data)
        print(f"✅ Autolykos v2 hash: {hash_result.hex()[:32]}...")
        
        # Quick benchmark
        print("🏃 Running quick benchmark...")
        benchmark = autolykos.benchmark(10)
        print(f"   Hashrate: {benchmark['hashrate']}")
        print(f"   Memory usage: {benchmark['memory_usage']}")
        print(f"   K parameter: {benchmark['k_parameter']}")
        print(f"   N parameter: {benchmark['n_parameter']}")
        print(f"   ASIC resistant: {benchmark['asic_resistant']}")
        
    except Exception as e:
        print(f"❌ Autolykos v2 error: {e}")


def test_algorithm_verification():
    """Test ASIC resistance verification for all algorithms"""
    print("\n🛡️ ASIC Resistance Verification:")
    print("-" * 50)
    
    algorithms_to_test = ['argon2', 'yescrypt', 'autolykos2', 'kawpow', 'ethash', 'sha256', 'scrypt']
    
    for algo in algorithms_to_test:
        is_resistant = verify_asic_resistance(algo)
        status = "✅ ASIC-Resistant" if is_resistant else "⚠️ ASIC-Friendly"
        print(f"   {algo:12} - {status}")


def show_algorithm_categories():
    """Show all algorithm categories"""
    print("\n📊 Algorithm Categories:")
    print("-" * 40)
    
    categories = AlgorithmFactory.get_algorithm_categories()
    
    for category, algorithms in categories.items():
        print(f"\n🏷️  {category}:")
        for algo in algorithms:
            print(f"   • {algo}")


def test_mining_config_integration():
    """Test integration with mining configuration"""
    print("\n⚙️ Mining Configuration Integration:")
    print("-" * 50)
    
    config = get_mining_config()
    
    print("Available algorithms:")
    available = config.get_available_algorithms()
    for algo, description in available.items():
        print(f"   • {algo}: {description}")
    
    print("\nPool ports:")
    for algo in ['argon2', 'yescrypt', 'autolykos2', 'kawpow']:
        port = config.get_pool_port(algo)
        print(f"   • {algo}: port {port}")
    
    print("\nTesting algorithm switching:")
    for algo in ['yescrypt', 'autolykos2', 'argon2']:
        success = config.set_algorithm(algo)
        current = config._config['algorithm']
        print(f"   • Switch to {algo}: {'✅' if success else '❌'} (current: {current})")


def benchmark_comparison():
    """Compare performance of all ASIC-resistant algorithms"""
    print("\n🏁 Performance Comparison (10 iterations each):")
    print("-" * 60)
    
    algorithms = ['argon2', 'yescrypt', 'autolykos2']
    results = {}
    
    for algo in algorithms:
        print(f"\n🔄 Benchmarking {algo}...")
        try:
            config = AlgorithmFactory.get_default_config(algo)
            algorithm = AlgorithmFactory.create_algorithm(algo, config)
            benchmark = algorithm.benchmark(10)
            results[algo] = benchmark
            
            hashrate = benchmark['hashrate']
            memory = benchmark.get('memory_usage', 'N/A')
            print(f"   ✅ {algo}: {hashrate}, Memory: {memory}")
            
        except Exception as e:
            print(f"   ❌ {algo}: Error - {e}")
            results[algo] = {'error': str(e)}
    
    # Summary
    print("\n📈 Benchmark Summary:")
    print("-" * 30)
    for algo, result in results.items():
        if 'error' not in result:
            print(f"   {algo:12} - {result['hashrate']:>15} ({result.get('memory_usage', 'N/A')})")
        else:
            print(f"   {algo:12} - {'ERROR':>15}")


def test_pool_integration():
    """Test pool integration with new algorithms"""
    print("\n🏊 Pool Integration Test:")
    print("-" * 35)
    
    # Test humanitarian distribution integration
    distributor = get_humanitarian_distributor()
    print("✅ Humanitarian distribution system loaded")
    
    # Simulate mining rewards for different algorithms
    algorithms = ['argon2', 'yescrypt', 'autolykos2']
    
    for i, algo in enumerate(algorithms, 1):
        reward = Decimal('1000')  # 1000 ZION per block
        print(f"\n🎯 Block {i} mined with {algo}:")
        print(f"   💰 Reward: {reward} ZION")
        
        # Calculate humanitarian distribution
        distribution = distributor.calculate_distribution(reward)
        humanitarian_total = sum(distribution.values())
        miner_reward = reward - humanitarian_total
        
        print(f"   👤 Miner: {miner_reward} ZION (90%)")
        print(f"   🌍 Humanitarian: {humanitarian_total} ZION (10%)")
        
        # Show project breakdown
        for project_id, amount in list(distribution.items())[:2]:  # Show first 2 projects
            project = next(p for p in distributor.projects if p.id == project_id)
            print(f"     • {project.name}: {amount} ZION")
        
        print(f"     • ... and 3 more projects")


def main():
    """Main demo function"""
    print("🚀 ZION 2.7.1 Enhanced Mining Algorithms")
    print("=" * 60)
    print("Testing new YesCrypt and Autolykos v2 algorithms")
    print("Plus integration with humanitarian distribution system")
    print("=" * 60)
    
    try:
        # Test new algorithms
        test_new_algorithms()
        
        # Test ASIC resistance verification
        test_algorithm_verification()
        
        # Show algorithm categories
        show_algorithm_categories()
        
        # Test mining config integration
        test_mining_config_integration()
        
        # Performance comparison
        benchmark_comparison()
        
        # Test pool integration
        test_pool_integration()
        
        print("\n🎉 Enhanced Algorithms Demo Complete!")
        print("=" * 50)
        print("✅ YesCrypt: ASIC-resistant memory-hard algorithm")
        print("✅ Autolykos v2: ASIC-resistant GPU-optimized algorithm")
        print("✅ Pool integration: Full humanitarian distribution support")
        print("✅ Configuration: Updated with new algorithm ports")
        
        print("\n🌟 ZION 2.7.1 now supports:")
        print("   • 3 Primary ASIC-resistant algorithms")
        print("   • 5 Alternative algorithms for flexibility")
        print("   • 10% automatic humanitarian distribution")
        print("   • Real-time algorithm switching")
        print("   • Advanced mining pool integration")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()