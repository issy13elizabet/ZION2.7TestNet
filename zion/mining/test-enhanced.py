#!/usr/bin/env python3
"""
Quick test for ZION Real Miner Enhanced functionality
"""
import os
import subprocess

def test_temperature():
    """Test temperature reading capability"""
    print("🌡️ Testing temperature reading...")
    try:
        result = subprocess.run(['sensors'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Tctl:' in line:
                    print(f"✅ AMD CPU temperature detected: {line.strip()}")
                    return True
            print("⚠️ CPU temperature not found in sensors output")
            return False
    except Exception as e:
        print(f"❌ Error reading temperature: {e}")
        return False

def test_miner_file():
    """Test if the enhanced miner file exists and is readable"""
    miner_path = "/home/maitreya/Desktop/zion-miner-1.4.0/zion-real-miner-v2.py"
    if os.path.exists(miner_path):
        print(f"✅ Enhanced miner found: {miner_path}")
        try:
            with open(miner_path, 'r') as f:
                content = f.read()
                if "Temperature Monitor" in content and "NiceHash" in content:
                    print("✅ Enhanced features detected in file")
                    return True
                else:
                    print("⚠️ Enhanced features not found in file")
                    return False
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return False
    else:
        print(f"❌ Miner file not found: {miner_path}")
        return False

if __name__ == "__main__":
    print("🔍 Testing ZION Real Miner Enhanced functionality...\n")
    
    temp_ok = test_temperature()
    file_ok = test_miner_file()
    
    print(f"\n📊 Test Results:")
    print(f"Temperature Monitoring: {'✅ OK' if temp_ok else '❌ FAIL'}")
    print(f"Enhanced Miner File: {'✅ OK' if file_ok else '❌ FAIL'}")
    
    if temp_ok and file_ok:
        print("\n🚀 All systems ready! Enhanced miner can be started.")
    else:
        print("\n⚠️ Some issues detected. Please resolve before mining.")