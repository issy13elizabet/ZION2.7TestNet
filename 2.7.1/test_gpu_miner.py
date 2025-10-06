#!/usr/bin/env python3
"""
ZION GPU Miner Test Script
Test GPU mining funkcí a AI integrace
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_gpu_miner():
    """Test GPU miner funkcí"""
    print("🎮 ZION GPU Miner Test")
    print("=" * 40)

    try:
        # Test přímý import GPU miner
        from ai.zion_gpu_miner import ZionGPUMiner
        print("✅ GPU Miner import successful")

        # Vytvoření instance
        gpu_miner = ZionGPUMiner()
        print("✅ GPU Miner instance created")

        # Test základních funkcí
        gpu_stats = gpu_miner.get_mining_stats()
        print("📊 GPU Stats:")
        print(f"   GPU Available: {gpu_stats['gpu_available']}")
        print(f"   Is Mining: {gpu_stats['is_mining']}")
        print(f"   Hashrate: {gpu_stats['hashrate']:.1f} MH/s")
        print(f"   Temperature: {gpu_stats['temperature']:.1f}°C")
        print(f"   Power Usage: {gpu_stats['power_usage']:.1f}W")
        print()

        # Test spuštění mining
        print("⛏️ Testing mining start...")
        mining_started = gpu_miner.start_mining(algorithm="octopus", intensity=75)
        print(f"   Mining started: {mining_started}")

        if mining_started:
            # Získání nových statistik
            gpu_stats = gpu_miner.get_mining_stats()
            print("📊 Updated GPU Stats:")
            print(f"   Is Mining: {gpu_stats['is_mining']}")
            print(f"   Hashrate: {gpu_stats['hashrate']:.1f} MH/s")
            print()

            # Test optimalizace
            print("⚙️ Testing GPU optimization...")
            optimization = gpu_miner.optimize_gpu_settings()
            print("   Optimization results:")
            print(f"   Core Clock: {optimization.get('core_clock', 'N/A')} MHz")
            print(f"   Memory Clock: {optimization.get('memory_clock', 'N/A')} MHz")
            print(f"   Fan Speed: {optimization.get('fan_speed', 'N/A')}%")
            if optimization.get('recommendations'):
                print("   Recommendations:")
                for rec in optimization['recommendations']:
                    print(f"     - {rec}")
            print()

            # Zastavení mining
            print("⏹️ Stopping mining...")
            gpu_miner.stop_mining()
            final_stats = gpu_miner.get_mining_stats()
            print(f"   Mining stopped: {not final_stats['is_mining']}")

        print("\n✅ GPU Miner test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ GPU Miner test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ai_orchestrator_with_gpu():
    """Test AI orchestrátor s GPU miner"""
    print("\n🤖 Testing AI Orchestrator with GPU Miner")
    print("=" * 50)

    try:
        from ai.zion_ai_master_orchestrator import ZionAIMasterOrchestrator

        # Vytvoření orchestrátoru
        orchestrator = ZionAIMasterOrchestrator()
        loaded = orchestrator.load_components()
        print(f"✅ Orchestrator loaded {loaded} components")

        # Kontrola, jestli je GPU miner načten
        status = orchestrator.get_status()
        gpu_loaded = 'ZionGPUMiner' in status['components']
        print(f"🎮 GPU Miner loaded: {gpu_loaded}")

        if gpu_loaded:
            # Test sacred mining s GPU
            print("\n⛏️ Testing Sacred Mining with GPU...")
            mining_result = orchestrator.perform_sacred_mining({
                'block_hash': 'gpu_test_block_' + str(hash('gpu_test'))[:8],
                'mining_power': 100.0,
                'difficulty': 10000
            })

            print("   Sacred Mining Results:")
            print(f"   Block Hash: {mining_result['block_hash'][:16]}...")
            print(f"   Mining Power: {mining_result['mining_power']:.1f}")
            print(f"   AI Boost: {mining_result['ai_boost']:.2f}x")
            print(f"   AI Contribution: {mining_result['ai_contribution']:.1f}%")
            print(f"   GPU Components Used: {'gpu_miner' in mining_result['ai_components_used']}")
            print(f"   Divine Validation: {'✅' if mining_result['divine_validation'] else '❌'}")

        print("\n✅ AI Orchestrator with GPU test completed!")
        return True

    except Exception as e:
        print(f"❌ AI Orchestrator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 ZION GPU Mining Test Suite")
    print("=" * 50)
    print("JAI RAM SITA HANUMAN - ON THE STAR 🌟")
    print()

    # Test GPU miner
    gpu_test = test_gpu_miner()

    # Test AI orchestrátor s GPU
    ai_test = test_ai_orchestrator_with_gpu()

    if gpu_test and ai_test:
        print("\n🎉 All GPU mining tests PASSED!")
        print("✅ GPU miner is ready for production use")
    else:
        print("\n❌ Some tests failed - check GPU availability and configuration")

    sys.exit(0 if (gpu_test and ai_test) else 1)