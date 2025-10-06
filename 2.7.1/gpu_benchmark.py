#!/usr/bin/env python3
"""
ZION 2.7.1 - GPU Benchmark Script
Test and benchmark GPU mining performance
"""

import sys
import time
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from mining.algorithms import GPUAlgorithm, AlgorithmFactory

def benchmark_gpu(iterations: int = 1000, data_size: int = 64):
    """Benchmark GPU algorithm performance"""
    print("🎮 ZION 2.7.1 GPU Benchmark")
    print("=" * 50)

    try:
        gpu_algo = GPUAlgorithm()
        algo_name = gpu_algo.get_name()

        print(f"📊 Testing: {algo_name}")
        print(f"📏 Data size: {data_size} bytes")
        print(f"🔄 Iterations: {iterations}")
        print()

        # Prepare test data
        test_data = b"ZION_GPU_BENCHMARK_" + b"A" * (data_size - 20)

        # Warm up
        print("🔥 Warming up...")
        for _ in range(10):
            gpu_algo.hash(test_data)

        # Benchmark
        print("🏃 Running benchmark...")
        start_time = time.time()

        hashes = []
        for i in range(iterations):
            hash_result = gpu_algo.hash(test_data)
            hashes.append(hash_result)

            if (i + 1) % (iterations // 10) == 0:
                progress = (i + 1) / iterations * 100
                print(".1f")

        end_time = time.time()
        duration = end_time - start_time

        # Results
        hashrate = iterations / duration
        avg_time = duration / iterations * 1000  # ms

        print()
        print("📈 Results:")
        print(f"   Total time: {duration:.2f}s")
        print(f"   Hashrate: {hashrate:.0f} H/s")
        print(f"   Avg time per hash: {avg_time:.3f} ms")
        print(f"   Sample hash: {hashes[0][:32]}...")

        # Verify determinism
        print()
        print("🔍 Verifying determinism...")
        test_hash = gpu_algo.hash(test_data)
        deterministic = all(h == test_hash for h in hashes[:10])  # Test first 10

        if deterministic:
            print("✅ All hashes are deterministic")
        else:
            print("❌ Hashes are not deterministic!")

        return {
            'algorithm': algo_name,
            'hashrate': hashrate,
            'duration': duration,
            'iterations': iterations,
            'deterministic': deterministic
        }

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return None

def compare_algorithms(iterations: int = 100):
    """Compare all available algorithms"""
    print("🔄 Comparing All Algorithms")
    print("=" * 50)

    test_data = b"ZION_ALGORITHM_COMPARISON_" + b"X" * 44

    results = {}
    algorithms = ['sha256', 'randomx', 'gpu']

    for algo_name in algorithms:
        try:
            print(f"\n🏃 Testing {algo_name}...")
            algo = AlgorithmFactory.create_algorithm(algo_name)

            start_time = time.time()
            for _ in range(iterations):
                algo.hash(test_data)
            duration = time.time() - start_time

            hashrate = iterations / duration
            results[algo_name] = {
                'hashrate': hashrate,
                'duration': duration,
                'name': algo.get_name()
            }

            print(".0f")

        except Exception as e:
            print(f"❌ {algo_name} failed: {e}")
            results[algo_name] = {'error': str(e)}

    # Summary
    print("\n📊 Algorithm Comparison:")
    print("-" * 50)
    for algo, data in results.items():
        if 'error' in data:
            print("10")
        else:
            print("10")

    return results

def main():
    parser = argparse.ArgumentParser(description="ZION 2.7.1 GPU Benchmark")
    parser.add_argument('--iterations', type=int, default=1000,
                       help='Number of hash iterations (default: 1000)')
    parser.add_argument('--data-size', type=int, default=64,
                       help='Size of test data in bytes (default: 64)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all algorithms instead of just GPU')

    args = parser.parse_args()

    if args.compare:
        compare_algorithms(args.iterations)
    else:
        benchmark_gpu(args.iterations, args.data_size)

if __name__ == "__main__":
    main()