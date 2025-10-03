#!/usr/bin/env python3
"""
ZION Continuous Hybrid Mining - No Time Limits
"""

import sys
import os
import time
import signal
import logging

# Add ai directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ai'))

from zion_hybrid_miner import ZionHybridMiner

# Global miner instance
miner = None

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n💡 Stopping mining...")
    if miner:
        miner.stop_hybrid_mining()
    print("✅ Mining stopped!")
    sys.exit(0)

def main():
    """Main continuous mining function"""
    global miner
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Handle Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🚀 ZION Continuous Hybrid Mining")
    print("=" * 50)
    
    # Initialize miner
    miner = ZionHybridMiner()
    
    print(f"✓ CPU Miner: {'Found' if miner.xmrig_path else 'Not found'}")
    print(f"✓ GPU Miner: {'Found' if miner.gpu_miner.srbminer_path else 'Not found'}")
    print(f"✓ GPU Available: {miner.gpu_miner.gpu_available}")
    print()
    
    # Pool configuration
    gpu_pool = {
        'url': 'pool.ravenminer.com:3838',
        'wallet': 'test_wallet_address', 
        'password': 'x'
    }
    
    cpu_pool = {
        'url': 'pool.supportxmr.com:3333',
        'wallet': 'test_wallet_address',
        'password': 'x'
    }
    
    print("🎯 Starting hybrid mining...")
    print("   CPU: RandomX → pool.supportxmr.com:3333")
    print("   GPU: KawPow → pool.ravenminer.com:3838")
    print()
    
    # Start mining
    success = miner.start_hybrid_mining(
        gpu_algorithm="kawpow",
        cpu_algorithm="randomx", 
        gpu_pool=gpu_pool,
        cpu_pool=cpu_pool
    )
    
    if success:
        print("✅ Mining started successfully!")
        print("📊 Monitor your GPU temperatures!")
        print("🔥 Watch for 'Accept Share!' messages...")
        print("⏹️ Press Ctrl+C to stop")
        print()
        
        # Keep running indefinitely
        start_time = time.time()
        while True:
            elapsed = int(time.time() - start_time)
            minutes = elapsed // 60
            seconds = elapsed % 60
            print(f"⏰ Mining time: {minutes:02d}:{seconds:02d} - {time.strftime('%H:%M:%S')}")
            time.sleep(60)  # Status update every minute
            
    else:
        print("❌ Failed to start hybrid mining")

if __name__ == "__main__":
    main()