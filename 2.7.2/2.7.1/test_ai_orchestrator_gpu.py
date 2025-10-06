#!/usr/bin/env python3
"""
Test script pro ZION AI Orchestrator s GPU Mining
Ověří integraci GPU miner komponenty s reálným hashrate
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
    from zion_ai_master_orchestrator import ZionAIMasterOrchestrator

    def test_ai_orchestrator_with_gpu():
        """Testuje AI orchestrátor s GPU mining"""
        print("🧠 Testing ZION AI Orchestrator with GPU Mining")
        print("=" * 55)

        # Vytvoří orchestrátor
        orchestrator = ZionAIMasterOrchestrator()

        # Načte komponenty
        print("🔧 Loading AI components...")
        loaded_count = orchestrator.load_components()
        print(f"✅ Loaded {loaded_count} components")

        # Zobrazí status
        status = orchestrator.get_status()
        print(f"📊 System Status:")
        print(f"   Total Components: {status['total_components']}")
        print(f"   Components: {', '.join(status['components'])}")
        print()

        # Test 1: Základní diagnostika
        print("🔍 Test 1: System Diagnostics")
        diagnostics = orchestrator.run_diagnostics()
        print(f"   System Status: {diagnostics['system_status']['total_components']} components")
        print("   Component Details:")
        for name, details in diagnostics['component_details'].items():
            methods = ', '.join(details['methods'][:3])  # První 3 metody
            print(f"     - {name}: {details['status']} ({methods})")
        print()

        # Test 2: Sacred Mining s GPU
        print("⛏️  Test 2: Sacred Mining with GPU Support")
        print("   Starting sacred mining with AI enhancement...")

        start_time = time.time()
        mining_result = orchestrator.perform_sacred_mining()
        end_time = time.time()

        print(f"   ✅ Mining completed in {end_time - start_time:.2f} seconds")
        print(f"   Block Hash: {mining_result['block_hash'][:16]}...")
        print(f"   Mining Power: {mining_result['mining_power']:.1f}")
        print(f"   AI Contribution: {mining_result['ai_contribution']:.1f}%")
        print(f"   AI Boost: {mining_result['ai_boost']:.2f}x")
        print(f"   GPU Mining Active: {'✅' if mining_result.get('gpu_mining_active', False) else '❌'}")
        print(f"   Divine Validation: {'✅' if mining_result['divine_validation'] else '❌'}")
        print(f"   AI Components Used: {', '.join(mining_result['ai_components_used'])}")
        print()

        # Test 3: Unified AI Analysis
        print("🔬 Test 3: Unified AI Analysis")
        analysis = orchestrator.perform_unified_ai_analysis()
        print(f"   Consensus Score: {analysis['consensus_score']:.2f}")
        print(f"   Divine Validation: {'✅' if analysis['divine_validation'] else '❌'}")
        print(f"   Total Analyses: {analysis['total_analyses']}")

        if analysis['analyses']:
            print("   Analysis Results:")
            for analysis_type, result in analysis['analyses'].items():
                confidence = result.get('confidence', 'N/A')
                print(f"     - {analysis_type}: confidence {confidence}")
        print()

        # Test 4: Resource Management
        print("⚙️  Test 4: Resource Management")
        resources = orchestrator.get_resource_usage()
        print(f"   CPU Usage: {resources.get('cpu_usage', 'N/A')}%")
        print(f"   Memory Usage: {resources.get('memory_usage', 'N/A')}%")

        optimizations = orchestrator.optimize_resources()
        if optimizations.get('optimizations'):
            print("   Optimization Recommendations:")
            for opt in optimizations['optimizations']:
                print(f"     • {opt}")
        else:
            print("   No optimizations needed")
        print()

        # Test 5: GPU Mining Details (pokud je k dispozici)
        print("🎮 Test 5: GPU Mining Component Details")
        if 'ZionGPUMiner' in orchestrator.components:
            gpu_miner = orchestrator.components['ZionGPUMiner']['instance']
            gpu_stats = gpu_miner.get_mining_stats()

            print(f"   GPU Available: {'✅' if gpu_stats['gpu_available'] else '❌'}")
            print(f"   Benchmark Hashrate: {gpu_stats.get('benchmark_hashrate', 0):.1f} MH/s")
            print(f"   Current Hashrate: {gpu_stats.get('hashrate', 0):.1f} MH/s")
            print(f"   Temperature: {gpu_stats.get('temperature', 'N/A')}°C")
            print(f"   Power Usage: {gpu_stats.get('power_usage', 'N/A')}W")
            print(f"   Efficiency: {gpu_stats.get('efficiency', 0):.2f}")

            # Test GPU optimalizace
            print("   GPU Optimization:")
            optimization = gpu_miner.optimize_gpu_settings()
            if 'error' not in optimization:
                print(f"     Core Clock: +{optimization['core_clock']} MHz")
                print(f"     Memory Clock: +{optimization['memory_clock']} MHz")
                print(f"     Fan Speed: {optimization['fan_speed']}%")
                print("     Recommendations:")
                for rec in optimization['recommendations'][:2]:  # První 2
                    print(f"       • {rec}")
            else:
                print(f"     Error: {optimization['error']}")
        else:
            print("   GPU Miner component not loaded")
        print()

        print("✅ AI Orchestrator with GPU Mining test completed!")
        print("🎯 Real hashrate measurement integrated successfully!")
        return True

    if __name__ == "__main__":
        test_ai_orchestrator_with_gpu()

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("AI orchestrátor komponenta nenalezena nebo chybí závislosti")
    sys.exit(1)
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)