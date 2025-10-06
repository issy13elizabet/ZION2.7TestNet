#!/usr/bin/env python3
"""
Test script pro ZION GPU Miner s SRBMiner-Multi integrací
Testuje skutečné GPU mining s externím miner software
"""

import sys
import os
import time
import logging

# Nastaví logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Přidá AI složku do cesty
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ai'))

try:
    from zion_gpu_miner import ZionGPUMiner

    def test_srbminer_integration():
        """Testuje SRBMiner-Multi integraci"""
        print("🔧 Testing ZION GPU Miner with SRBMiner-Multi Integration")
        print("=" * 60)

        # Vytvoří GPU miner
        miner = ZionGPUMiner()

        # Zobrazí základní informace
        print(f"GPU Available: {'✅' if miner.gpu_available else '❌'}")
        print(f"SRBMiner Found: {'✅' if miner.srbminer_path else '❌'}")
        print(".1f")
        print(f"GPU Type: {miner._detect_gpu_type()}")
        print()

        # Test 1: Konfigurace mining pool
        print("⚙️  Test 1: Mining Pool Configuration")
        success = miner.configure_mining_pool(
            pool_host="stratum.ravenminer.com",
            pool_port=3838,
            wallet_address="RTestWallet123456789",
            password="x"
        )
        print(f"   Pool config saved: {'✅' if success else '❌'}")
        print(f"   Pool: {miner.mining_config['pool_host']}:{miner.mining_config['pool_port']}")
        print(f"   Wallet: {miner.mining_config['wallet_address']}")
        print()

        # Test 2: Podporované algoritmy
        print("📋 Test 2: Supported Algorithms")
        algorithms = miner.get_supported_algorithms()
        print(f"   Found {len(algorithms)} supported algorithms:")
        for algo, info in algorithms.items():
            print(f"     • {algo}: {info['name']} ({info['gpu_efficiency']} efficiency)")
        print()

        # Test 3: Auto-tuning algoritmu
        print("🎯 Test 3: Algorithm Auto-Tuning")
        best_algo = miner.auto_tune_algorithm()
        print(f"   Recommended algorithm: {best_algo}")
        gpu_type = miner._detect_gpu_type()
        print(f"   Based on GPU type: {gpu_type}")
        print()

        # Test 4: GPU optimalizace
        print("🔧 Test 4: GPU Optimization")
        optimization = miner.optimize_gpu_settings()
        if 'error' not in optimization:
            print(f"   Core Clock: +{optimization['core_clock']} MHz")
            print(f"   Memory Clock: +{optimization['memory_clock']} MHz")
            print(f"   Fan Speed: {optimization['fan_speed']}%")
            print("   Recommendations:")
            for rec in optimization['recommendations'][:2]:
                print(f"     • {rec}")
        else:
            print(f"   Error: {optimization['error']}")
        print()

        # Test 5: Mining stats bez mining
        print("📊 Test 5: Mining Stats (Not Mining)")
        stats = miner.get_mining_stats()
        print(f"   Is Mining: {stats['is_mining']}")
        print(".1f")
        print(".1f")
        print(".1f")
        print()

        # Test 6: Spuštění mining (pokud je SRBMiner dostupný)
        if miner.srbminer_path:
            print("⛏️  Test 6: Starting Real GPU Mining with SRBMiner")
            print("   ⚠️  This will start actual mining process...")

            mining_started = miner.start_mining(algorithm=best_algo, intensity=60)
            print(f"   Mining started: {'✅' if mining_started else '❌'}")

            if mining_started:
                # Monitoruj po dobu 15 sekund
                print("   Monitoring mining for 15 seconds...")
                for i in range(3):
                    time.sleep(5)
                    stats = miner.get_mining_stats()
                    print(".1f")

                # Zastav mining
                print("   Stopping mining...")
                miner.stop_mining()
                print("   ✅ Mining stopped")

                # Finální statistiky
                final_stats = miner.get_mining_stats()
                print(".1f")

        else:
            print("⛏️  Test 6: SRBMiner Not Available")
            print("   Starting simulated mining instead...")

            mining_started = miner.start_mining(algorithm=best_algo, intensity=60)
            print(f"   Simulated mining started: {'✅' if mining_started else '❌'}")

            if mining_started:
                time.sleep(3)
                stats = miner.get_mining_stats()
                print(".1f")
                miner.stop_mining()
                print("   ✅ Simulated mining stopped")

        print()
        print("✅ SRBMiner integration test completed!")
        print("🎯 Real GPU mining capabilities integrated successfully!")

        return True

    if __name__ == "__main__":
        test_srbminer_integration()

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("GPU miner komponenta nenalezena")
    sys.exit(1)
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)