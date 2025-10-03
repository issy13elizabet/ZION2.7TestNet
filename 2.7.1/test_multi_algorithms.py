#!/usr/bin/env python3
"""
ZION 2.7.1 Multi-Algorithm Test Suite
Tests all supported mining algorithms
"""

import sys
import time

def test_algorithm(algorithm_name, expected_category):
    """Test single algorithm"""
    print(f"\n🧪 Testing {algorithm_name}...")

    try:
        from mining.algorithms import AlgorithmFactory
        from mining.config import get_mining_config

        config = get_mining_config()
        algo_config = config.get_algorithm_config(algorithm_name)
        algorithm = AlgorithmFactory.create_algorithm(algorithm_name, algo_config)

        # Test basic functionality
        test_data = b"zion_test_" + algorithm_name.encode()
        hash_result = algorithm.hash(test_data)
        print(f"  ✅ Hash: {hash_result.hex()[:16]}...")

        # Test verification
        target = b"\x00\x00\xff\xff" + b"\x00" * 28
        is_valid = algorithm.verify(test_data, target)
        print(f"  ✅ Verify: {is_valid}")

        # Test benchmark
        benchmark = algorithm.benchmark(10)
        print(f"  ✅ Benchmark: {benchmark['hashrate']} ({benchmark['category']})")

        # Verify category
        if benchmark['category'] == expected_category:
            print(f"  ✅ Category: {expected_category}")
            return True
        else:
            print(f"  ❌ Wrong category: {benchmark['category']} (expected {expected_category})")
            return False

    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

def main():
    """Run complete algorithm test suite"""
    print("🎯 ZION 2.7.1 Multi-Algorithm Test Suite")
    print("=" * 50)

    # Test ASIC-Resistant algorithms
    print("\n🛡️ Testing ASIC-Resistant Algorithms:")
    print("-" * 40)

    asic_resistant_tests = [
        ('argon2', 'ASIC-Resistant'),
        ('cryptonight', 'ASIC-Resistant'),
        ('ergo', 'ASIC-Resistant')
    ]

    asic_passed = 0
    for algo, category in asic_resistant_tests:
        if test_algorithm(algo, category):
            asic_passed += 1

    # Test GPU-Friendly algorithms
    print("\n🎮 Testing GPU-Friendly Algorithms:")
    print("-" * 40)

    gpu_friendly_tests = [
        ('kawpow', 'GPU-Friendly'),
        ('ethash', 'GPU-Optimized'),
        ('octopus', 'GPU-Optimized')
    ]

    gpu_passed = 0
    for algo, category in gpu_friendly_tests:
        if test_algorithm(algo, category):
            gpu_passed += 1

    # Test blocked algorithms
    print("\n🚫 Testing Blocked Algorithms:")
    print("-" * 30)

    try:
        from mining.algorithms import AlgorithmFactory
        algo = AlgorithmFactory.create_algorithm('sha256', {})
        print("  ❌ SHA256 should be blocked!")
        sha256_blocked = False
    except ValueError:
        print("  ✅ SHA256 properly blocked")
        sha256_blocked = True
    except Exception as e:
        print(f"  ⚠️ Unexpected error: {e}")
        sha256_blocked = False

    # Test algorithm switching
    print("\n🔄 Testing Algorithm Switching:")
    print("-" * 35)

    try:
        from mining.config import switch_algorithm

        # Test switching to ASIC-resistant
        if switch_algorithm('argon2'):
            print("  ✅ Switch to Argon2: SUCCESS")
            switch_test = True
        else:
            print("  ❌ Switch to Argon2: FAILED")
            switch_test = False

        # Test switching to GPU-friendly
        if switch_algorithm('kawpow'):
            print("  ✅ Switch to KawPow: SUCCESS")
        else:
            print("  ❌ Switch to KawPow: FAILED")
            switch_test = False

    except Exception as e:
        print(f"  ❌ Switch test failed: {e}")
        switch_test = False

    # Summary
    print("\n📊 Test Results Summary:")
    print("=" * 30)
    print(f"ASIC-Resistant Algorithms: {asic_passed}/{len(asic_resistant_tests)} passed")
    print(f"GPU-Friendly Algorithms: {gpu_passed}/{len(gpu_friendly_tests)} passed")
    print(f"SHA256 Blocked: {'✅' if sha256_blocked else '❌'}")
    print(f"Algorithm Switching: {'✅' if switch_test else '❌'}")

    total_passed = asic_passed + gpu_passed + (1 if sha256_blocked else 0) + (1 if switch_test else 0)
    total_tests = len(asic_resistant_tests) + len(gpu_friendly_tests) + 2

    print(f"\n🎯 Overall: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("🎉 ALL TESTS PASSED! ZION 2.7.1 multi-algorithm support is working!")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())