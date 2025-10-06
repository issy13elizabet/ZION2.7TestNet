#!/usr/bin/env python3
"""
ZION Professional Hybrid Miner Test
Real CPU RandomX + GPU KawPow mining (no simulation!)
"""

import sys
import os
import time
import signal
import logging

# Add ai directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ai'))

from zion_hybrid_miner import ZionHybridMiner

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\nStopping mining...")
    miner.stop_hybrid_mining()
    print("Mining stopped. Goodbye!")
    sys.exit(0)

def main():
    """Main test function"""
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Handle Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    print("ZION Professional Hybrid Miner Test")
    print("===================================")

    # Initialize miner
    global miner
    miner = ZionHybridMiner()

    print(f"✓ Xmrig CPU miner: {'Found' if miner.xmrig_path else 'Not found'}")
    print(f"✓ SRBMiner GPU miner: {'Found' if miner.gpu_miner.srbminer_path else 'Not found'}")
    print(f"✓ GPU available: {miner.gpu_miner.gpu_available}")
    print()

    if not miner.xmrig_path and not miner.gpu_miner.srbminer_path:
        print("❌ No miners found! Please ensure Xmrig and SRBMiner are properly installed.")
        return

    # Test pools (you can change these to real ZION pools when available)
    cpu_pool = {
        "url": "pool.supportxmr.com:3333",  # Monero pool for RandomX testing
        "user": "test_wallet_address",
        "pass": "x"
    }

    gpu_pool = {
        "url": "pool.ravenminer.com:3838",  # Ravencoin pool for KawPow testing
        "user": "test_wallet_address",
        "pass": "x"
    }

    # Start hybrid mining
    print("Starting real hybrid mining...")
    print("CPU: RandomX (like Xmrig)")
    print("GPU: KawPow (like SRB Miner)")
    print("Press Ctrl+C to stop\n")

    success = miner.start_hybrid_mining(
        gpu_algorithm="kawpow",
        cpu_algorithm="randomx",
        gpu_pool=gpu_pool,
        cpu_pool=cpu_pool
    )

    if success:
        print("✅ Hybrid mining started successfully!")
        print("Watch for 'Accept Share!' and 'Block Found!' messages...")

        # Keep running for 60 seconds for debug
        print("Running for 60 seconds debug...")
        time.sleep(60)
        print("\nDebug completed! Stopping mining...")
        miner.stop_hybrid_mining()
        print("✅ Real hybrid mining debug completed!")
        print("Check debug output above to see miner activity!")
    else:
        print("❌ Failed to start hybrid mining")

if __name__ == "__main__":
    main()