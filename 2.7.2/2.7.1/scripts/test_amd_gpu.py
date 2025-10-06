#!/usr/bin/env python3
"""
ZION AMD GPU Test Script
Test AMD RX 5600 XT GPU detection and mining capability
"""

import subprocess
import sys
import os
import json
from datetime import datetime

def test_amd_gpu():
    """Test AMD GPU detection and capabilities"""
    print("🔍 ZION AMD GPU Test")
    print("=" * 50)

    results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {},
        "gpu_info": {},
        "recommendations": []
    }

    # Test 1: ROCm detection
    print("\n1. Testing ROCm...")
    try:
        result = subprocess.run(['rocm-smi', '--showid'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            results["tests"]["rocm"] = "PASS"
            print("✅ ROCm detected")

            # Get GPU names
            try:
                name_result = subprocess.run(['rocm-smi', '--showproductname'], capture_output=True, text=True, timeout=10)
                if name_result.returncode == 0:
                    gpu_names = [line.strip() for line in name_result.stdout.split('\n') if line.strip() and 'GPU' in line.upper()]
                    results["gpu_info"]["rocm_gpus"] = gpu_names
                    print(f"   GPUs: {', '.join(gpu_names)}")
            except:
                pass
        else:
            results["tests"]["rocm"] = "FAIL"
            print("❌ ROCm not detected")
            results["recommendations"].append("Install ROCm drivers: https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.4.3")
    except FileNotFoundError:
        results["tests"]["rocm"] = "NOT_INSTALLED"
        print("❌ ROCm not installed")
        results["recommendations"].append("Install ROCm: https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.4.3")

    # Test 2: OpenCL detection
    print("\n2. Testing OpenCL...")
    try:
        result = subprocess.run(['clinfo', '--list'], capture_output=True, text=True, timeout=10)
        if 'AMD' in result.stdout.upper():
            results["tests"]["opencl"] = "PASS"
            print("✅ OpenCL AMD GPU detected")

            # Extract platform info
            lines = result.stdout.split('\n')
            for line in lines:
                if 'AMD' in line.upper():
                    results["gpu_info"]["opencl_platform"] = line.strip()
                    print(f"   Platform: {line.strip()}")
                    break
        else:
            results["tests"]["opencl"] = "FAIL"
            print("❌ OpenCL AMD GPU not detected")
    except FileNotFoundError:
        results["tests"]["opencl"] = "NOT_INSTALLED"
        print("❌ OpenCL clinfo not installed")
        results["recommendations"].append("Install OpenCL: pip install pyopencl")

    # Test 3: Windows GPU detection
    print("\n3. Testing Windows GPU detection...")
    try:
        result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                              capture_output=True, text=True, timeout=10)
        output = result.stdout.lower()
        if 'radeon' in output or 'rx 5600' in output or 'rx5600' in output:
            results["tests"]["windows_gpu"] = "PASS"
            print("✅ AMD RX 5600 XT detected via Windows")

            # Extract GPU name
            lines = result.stdout.split('\n')
            for line in lines:
                if 'radeon' in line.lower() or 'rx 5600' in line or 'rx5600' in line:
                    results["gpu_info"]["windows_gpu"] = line.strip()
                    print(f"   GPU: {line.strip()}")
                    break
        else:
            results["tests"]["windows_gpu"] = "FAIL"
            print("❌ AMD GPU not detected via Windows")
    except:
        results["tests"]["windows_gpu"] = "ERROR"
        print("❌ Windows GPU detection failed")

    # Test 4: Python GPU libraries
    print("\n4. Testing Python GPU libraries...")
    try:
        import pyopencl as cl
        results["tests"]["pyopencl"] = "PASS"
        print("✅ PyOpenCL available")

        # Try to get platforms
        platforms = cl.get_platforms()
        amd_platforms = [p for p in platforms if 'AMD' in p.name.upper()]
        if amd_platforms:
            results["gpu_info"]["pyopencl_platforms"] = [p.name for p in amd_platforms]
            print(f"   AMD platforms: {[p.name for p in amd_platforms]}")
    except ImportError:
        results["tests"]["pyopencl"] = "NOT_INSTALLED"
        print("❌ PyOpenCL not installed")
        results["recommendations"].append("Install PyOpenCL: pip install pyopencl")

    # Test 5: SRBMiner compatibility
    print("\n5. Testing SRBMiner compatibility...")
    srbminer_path = None
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'miners', 'SRBMiner-Multi-latest', 'SRBMiner-Multi-2-9-8', 'SRBMiner-MULTI.exe'),
        os.path.join(os.path.dirname(__file__), '..', 'miners', 'SRBMiner-MULTI.exe'),
        'SRBMiner-MULTI.exe'
    ]

    for path in possible_paths:
        if os.path.exists(path):
            srbminer_path = path
            break

    if srbminer_path:
        results["tests"]["srbminer"] = "PASS"
        print(f"✅ SRBMiner found: {srbminer_path}")

        # Test SRBMiner --help
        try:
            result = subprocess.run([srbminer_path, '--help'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                results["tests"]["srbminer_help"] = "PASS"
                print("✅ SRBMiner executable working")
            else:
                results["tests"]["srbminer_help"] = "FAIL"
                print("❌ SRBMiner executable error")
        except:
            results["tests"]["srbminer_help"] = "ERROR"
            print("❌ SRBMiner test failed")
    else:
        results["tests"]["srbminer"] = "NOT_FOUND"
        print("❌ SRBMiner not found")
        results["recommendations"].append("Download SRBMiner-Multi from https://github.com/doktor83/SRBMiner-Multi")

    # Test 6: ZION GPU Miner
    print("\n6. Testing ZION GPU Miner...")
    try:
        sys.path.append(os.path.dirname(__file__))
        from zion_gpu_miner import ZionGPUMiner

        miner = ZionGPUMiner()
        results["tests"]["zion_gpu_miner"] = "PASS"
        print("✅ ZION GPU Miner initialized")

        gpu_info = {
            "gpu_available": miner.gpu_available,
            "benchmark_hashrate": miner.benchmark_hashrate,
            "srbminer_found": miner.srbminer_path is not None
        }
        results["gpu_info"]["zion_miner"] = gpu_info
        print(f"   GPU available: {miner.gpu_available}")
        print(f"   Benchmark hashrate: {miner.benchmark_hashrate:.1f} MH/s")
        print(f"   SRBMiner found: {miner.srbminer_path is not None}")

    except Exception as e:
        results["tests"]["zion_gpu_miner"] = f"ERROR: {str(e)}"
        print(f"❌ ZION GPU Miner error: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)

    passed = 0
    total = len(results["tests"])

    for test, status in results["tests"].items():
        if status == "PASS":
            print(f"✅ {test}: PASS")
            passed += 1
        else:
            print(f"❌ {test}: {status}")

    print(f"\nPassed: {passed}/{total} tests")

    if results["recommendations"]:
        print("\n🔧 RECOMMENDATIONS:")
        for rec in results["recommendations"]:
            print(f"   • {rec}")

    # Save results
    output_file = os.path.join(os.path.dirname(__file__), '..', 'logs', 'amd_gpu_test_results.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📄 Results saved to: {output_file}")

    return results

if __name__ == "__main__":
    test_amd_gpu()