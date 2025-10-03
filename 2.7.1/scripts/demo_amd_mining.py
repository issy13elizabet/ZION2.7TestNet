#!/usr/bin/env python3
"""
ZION AMD RX 5600 XT Mining Demo
Demonstrates GPU mining capabilities
"""

import time
import random
from datetime import datetime

def simulate_amd_mining():
    """Simuluje AMD RX 5600 XT mining pro demonstraci"""
    print("🚀 ZION AMD RX 5600 XT Mining Demo")
    print("=" * 50)
    print("GPU: AMD Radeon RX 5600 XT (Detected)")
    print("Algorithm: KawPow")
    print("Expected Hashrate: 28 MH/s")
    print("Pool: RavenMiner (pool.ravenminer.com:3838)")
    print()

    print("Starting simulated mining...")
    print("Note: This is simulation. For real mining, enable SRBMiner first.")
    print()

    # Simulované mining statistiky
    base_hashrate = 28.0
    start_time = time.time()

    for i in range(20):
        # Přidat variaci jako v reálném mining
        variation = random.uniform(-0.05, 0.05)  # ±5%
        current_hashrate = base_hashrate * (1 + variation)

        # Simulovat teplotu a spotřebu
        temperature = random.uniform(65, 75)
        power = current_hashrate * 0.18  # ~0.18W per MH/s

        elapsed = time.time() - start_time
        total_hashes = current_hashrate * 1_000_000 * elapsed  # Převod na H/s * sekundy

        print(f"{i+1:2d}s | "
              f"Hashrate: {current_hashrate:5.1f} MH/s | "
              f"Temp: {temperature:4.1f}°C | "
              f"Power: {power:4.1f}W | "
              f"Total: {total_hashes:10,.0f} H")

        time.sleep(1)

    print()
    print("⏹️ Mining simulation completed!")
    print("💡 To enable real mining:")
    print("   1. Run: scripts\\enable_amd_mining.bat")
    print("   2. Add SRBMiner to Windows Defender exclusions")
    print("   3. Restart and run: python zion_gpu_miner.py")

if __name__ == "__main__":
    simulate_amd_mining()