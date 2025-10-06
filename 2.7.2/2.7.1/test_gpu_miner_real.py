#!/usr/bin/env python3
"""
Test script pro ZION GPU Miner s reálným hashrate
Ověří funkčnost GPU mining komponenty
"""

import sys
import os
import time
import logging

# Nastaví logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Přidá AI složku do cesty
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ai'))

try:
    from zion_gpu_miner import ZionGPUMiner

    def test_gpu_miner():
        """Testuje GPU miner funkcionality"""
        print("🧪 Testing ZION GPU Miner with Real Hashrate")
        print("=" * 50)

        # Vytvoří GPU miner
        miner = ZionGPUMiner()

        # Zobrazí základní informace
        print(f"GPU Available: {miner.gpu_available}")
        print(".1f")
        print()

        # Test 1: Získání statistik bez mining
        print("📊 Test 1: Getting mining stats (not mining)")
        stats = miner.get_mining_stats()
        print(f"  Is Mining: {stats['is_mining']}")
        print(".1f")
        print(".1f")
        print(".1f")
        print()

        # Test 2: Spuštění mining
        print("⛏️  Test 2: Starting GPU mining")
        success = miner.start_mining(algorithm="kawpow", intensity=80)
        print(f"  Mining started: {success}")

        if success:
            # Počká 3 sekundy na inicializaci
            print("  Waiting for mining to stabilize...")
            time.sleep(3)

            # Získá statistiky během mining
            stats = miner.get_mining_stats()
            print(f"  Is Mining: {stats['is_mining']}")
            print(".1f")
            print(".1f")
            print(".1f")
            print(".1f")
            print()

            # Test 3: GPU optimalizace
            print("⚙️  Test 3: GPU optimization")
            optimization = miner.optimize_gpu_settings()
            if 'error' not in optimization:
                print(f"  Core Clock: +{optimization['core_clock']} MHz")
                print(f"  Memory Clock: +{optimization['memory_clock']} MHz")
                print(f"  Fan Speed: {optimization['fan_speed']}%")
                print(f"  Power Limit: {optimization['power_limit']}%")
                print("  Recommendations:")
                for rec in optimization['recommendations']:
                    print(f"    • {rec}")
            else:
                print(f"  Error: {optimization['error']}")
            print()

            # Test 4: Zastavení mining
            print("🛑 Test 4: Stopping mining")
            miner.stop_mining()
            time.sleep(1)

            stats = miner.get_mining_stats()
            print(f"  Mining stopped: {not stats['is_mining']}")
            print(".1f")

        print()
        print("✅ GPU Miner test completed!")
        return True

    if __name__ == "__main__":
        test_gpu_miner()

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("GPU miner komponenta nenalezena nebo chybí závislosti")
    sys.exit(1)
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)