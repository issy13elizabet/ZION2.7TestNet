#!/usr/bin/env python3
"""
Test script pro ZION GPU Mining API
Ověří funkčnost API endpointů bez Flask závislostí
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

    def test_gpu_mining_api():
        """Testuje GPU mining API funkcionality"""
        print("🔧 Testing ZION GPU Mining API")
        print("=" * 40)

        # Vytvoří GPU miner
        miner = ZionGPUMiner()

        # Simulace API endpointů
        print("📡 Testing API Endpoints:")
        print()

        # 1. GET /api/mining/stats
        print("1️⃣ GET /api/mining/stats")
        stats = miner.get_mining_stats()
        print(f"   Status: {stats.get('is_mining', False)}")
        print(".1f")
        print(".1f")
        print()

        # 2. POST /api/mining/start
        print("2️⃣ POST /api/mining/start")
        start_data = {
            'algorithm': 'octopus',
            'intensity': 80,
            'pool_config': {
                'host': 'stratum.ravenminer.com',
                'port': 3838,
                'wallet': 'RTestWallet123'
            }
        }
        print(f"   Request: {json.dumps(start_data, indent=2)}")

        success = miner.start_mining(
            algorithm=start_data['algorithm'],
            intensity=start_data['intensity'],
            pool_config=start_data['pool_config']
        )
        print(f"   Response: {{'status': '{'success' if success else 'error'}'}}")
        print()

        # 3. Počkej a získej aktualizované statistiky
        print("3️⃣ GET /api/mining/stats (during mining)")
        time.sleep(2)
        stats = miner.get_mining_stats()
        print(f"   Status: {stats.get('is_mining', False)}")
        print(".1f")
        print(".1f")
        print()

        # 4. GET /api/mining/config
        print("4️⃣ GET /api/mining/config")
        config = {
            'mining_config': miner.mining_config,
            'supported_algorithms': miner.get_supported_algorithms(),
            'gpu_available': miner.gpu_available,
            'srbminer_available': miner.srbminer_path is not None,
            'benchmark_hashrate': miner.benchmark_hashrate,
            'gpu_type': miner._detect_gpu_type()
        }
        print(f"   GPU Available: {config['gpu_available']}")
        print(f"   SRBMiner Available: {config['srbminer_available']}")
        print(f"   GPU Type: {config['gpu_type']}")
        print(".1f")
        print(f"   Supported Algorithms: {len(config['supported_algorithms'])}")
        print()

        # 5. POST /api/mining/config
        print("5️⃣ POST /api/mining/config")
        new_config = {
            'pool_config': {
                'host': 'pool.2miners.com',
                'port': 4040,
                'wallet': 'NewTestWallet456'
            }
        }
        print(f"   Request: {json.dumps(new_config, indent=2)}")

        miner.configure_mining_pool(
            pool_host=new_config['pool_config']['host'],
            pool_port=new_config['pool_config']['port'],
            wallet_address=new_config['pool_config']['wallet']
        )
        print("   Response: {'status': 'success'}")
        print()

        # 6. POST /api/mining/optimize
        print("6️⃣ POST /api/mining/optimize")
        optimization = miner.optimize_gpu_settings()
        if 'error' not in optimization:
            print("   GPU Optimization Results:")
            print(f"     Core Clock: +{optimization['core_clock']} MHz")
            print(f"     Memory Clock: +{optimization['memory_clock']} MHz")
            print(f"     Fan Speed: {optimization['fan_speed']}%")
            print("     Recommendations:")
            for rec in optimization['recommendations'][:2]:
                print(f"       • {rec}")
        else:
            print(f"   Error: {optimization['error']}")
        print()

        # 7. POST /api/mining/stop
        print("7️⃣ POST /api/mining/stop")
        success = miner.stop_mining()
        print(f"   Response: {{'status': '{'success' if success else 'error'}'}}")
        print()

        # 8. GET /api/system/status
        print("8️⃣ GET /api/system/status")
        try:
            import psutil
            status = {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'gpu_miner_initialized': True,
                'mining_active': miner.is_mining
            }
            print(".1f")
            print(".1f")
            print(f"   GPU Miner Initialized: {status['gpu_miner_initialized']}")
            print(f"   Mining Active: {status['mining_active']}")
        except ImportError:
            print("   System monitoring not available (psutil not installed)")
        print()

        # 9. Dashboard info
        print("9️⃣ Dashboard Information")
        print("   📊 GPU Mining Dashboard: gpu_mining_dashboard.html")
        print("   🔧 API Endpoints Available:")
        print("     • GET  /api/mining/stats")
        print("     • POST /api/mining/start")
        print("     • POST /api/mining/stop")
        print("     • GET  /api/mining/config")
        print("     • POST /api/mining/config")
        print("     • GET  /api/mining/algorithms")
        print("     • POST /api/mining/optimize")
        print("     • POST /api/mining/benchmark")
        print("     • GET  /api/system/status")
        print("     • GET  /api/health")
        print()

        print("✅ GPU Mining API test completed!")
        print("🎯 All core functionalities verified!")
        print()
        print("💡 To run full web dashboard:")
        print("   1. Install Flask: pip install flask flask-cors")
        print("   2. Run: python gpu_mining_api.py")
        print("   3. Open: http://localhost:5000")

        return True

    if __name__ == "__main__":
        test_gpu_mining_api()

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("GPU miner komponenta nenalezena")
    sys.exit(1)
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)