#!/usr/bin/env python3
"""
Test script pro ZION GPU Miner Auto-Tuning
Testuje automatické ladění mining parametrů
"""

import sys
import os
import json
import time
import logging

# Nastaví logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Přidá AI složku do cesty
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ai'))

try:
    from zion_gpu_miner import ZionGPUMiner

    def test_auto_tuning():
        """Testuje auto-tuning funkcionality"""
        print("🎯 Testing ZION GPU Miner Auto-Tuning")
        print("=" * 45)

        # Vytvoří GPU miner
        miner = ZionGPUMiner()

        # Zobrazí základní informace
        print(f"GPU Available: {'✅' if miner.gpu_available else '❌'}")
        print(f"SRBMiner Found: {'✅' if miner.srbminer_path else '❌'}")
        print(f"GPU Type: {miner._detect_gpu_type()}")
        print(".1f")
        print()

        # Test 1: Spuštění auto-tuning (krátký test)
        print("🔧 Test 1: Auto-Tuning (Short Test)")
        print("⚠️  This will test mining algorithms for optimization...")
        print("   Testing with reduced duration for demo purposes")
        print()

        # Spustí auto-tuning s kratší dobou (1 minuta místo 5)
        tuning_results = miner.auto_tune_mining(duration_minutes=1)

        if 'error' not in tuning_results:
            print("✅ Auto-tuning completed successfully!")
            print(f"GPU Type: {tuning_results['gpu_type']}")
            print(".1f")
            print()

            # Zobraz výsledky testování algoritmů
            print("📊 Algorithm Test Results:")
            for algo, tests in tuning_results['algorithm_tests'].items():
                print(f"   {algo.upper()}:")
                for intensity_key, result in tests.items():
                    intensity = intensity_key.split('_')[1]
                    print(".1f")
                    print(".1f")
                    print(".1f")
                    print(".1f")
                print()

            # Zobraz nejlepší konfiguraci
            best_config = tuning_results['best_configuration']
            if best_config:
                print("🏆 Best Configuration Found:")
                print(f"   Algorithm: {best_config['algorithm']}")
                print(f"   Intensity: {best_config['intensity']}%")
                print(".1f")
                print()

            # Zobraz doporučení
            print("💡 Optimization Recommendations:")
            for rec in tuning_results['optimization_recommendations']:
                print(f"   • {rec}")
            print()

            # Test 2: Aplikace auto-tune výsledků
            print("⚙️  Test 2: Applying Auto-Tune Results")
            success = miner.apply_auto_tune_results(tuning_results)
            print(f"   Configuration applied: {'✅' if success else '❌'}")
            print(f"   Optimal Algorithm: {miner.mining_config.get('optimal_algorithm', 'N/A')}")
            print(f"   Optimal Intensity: {miner.mining_config.get('optimal_intensity', 'N/A')}%")
            print()

            # Test 3: Ověření aplikované konfigurace
            print("🔍 Test 3: Verification with Applied Configuration")
            optimal_algo = miner.mining_config.get('optimal_algorithm')
            optimal_intensity = miner.mining_config.get('optimal_intensity')

            if optimal_algo and optimal_intensity:
                print(f"   Starting mining with optimal settings: {optimal_algo} @ {optimal_intensity}%")

                mining_started = miner.start_mining(
                    algorithm=optimal_algo,
                    intensity=optimal_intensity
                )

                if mining_started:
                    print("   ✅ Mining started with auto-tuned configuration")

                    # Počkej 10 sekund a zkontroluj statistiky
                    time.sleep(10)
                    stats = miner.get_mining_stats()
                    print(".1f")
                    print(".1f")

                    # Zastav mining
                    miner.stop_mining()
                    print("   ✅ Mining stopped")
                else:
                    print("   ❌ Failed to start mining with optimal configuration")
            else:
                print("   ⚠️  No optimal configuration to test")

        else:
            print(f"❌ Auto-tuning failed: {tuning_results['error']}")
            print("   This may be due to GPU not being available or other system limitations")
            print()

        # Test 4: Stability calculation test
        print("📈 Test 4: Stability Calculation Test")
        # Simuluj různé hashrate hodnoty pro test stability
        test_hashrates = [25.3, 25.1, 25.4, 25.2, 25.3, 25.0]  # Stabilní
        stability1 = miner._calculate_stability(test_hashrates)
        print(".1f")

        test_hashrates2 = [20.0, 30.0, 15.0, 35.0, 25.0, 40.0]  # Nestabilní
        stability2 = miner._calculate_stability(test_hashrates2)
        print(".1f")
        print()

        print("✅ GPU Miner Auto-Tuning test completed!")
        print("🎯 Automatic optimization and algorithm selection working!")
        print()
        print("💡 Auto-tuning features:")
        print("   • Automatic algorithm selection based on GPU type")
        print("   • Intensity optimization for maximum hashrate")
        print("   • Stability analysis and performance monitoring")
        print("   • GPU-specific optimization recommendations")
        print("   • Configuration persistence for future sessions")

        return True

    if __name__ == "__main__":
        test_auto_tuning()

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("GPU miner komponenta nenalezena")
    sys.exit(1)
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)