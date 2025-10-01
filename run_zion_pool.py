#!/usr/bin/env python3
"""
🚀 ZION MINING POOL SERVER 🚀
Real production server for ZION mining
"""

import asyncio
import logging
from real_mining_pool import ZionRealMiningPool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def run_pool_server():
    """Run ZION mining pool server continuously"""
    print("🚀 Starting ZION Mining Pool Server...")
    print("⛏️ Port: 3333 | Algorithm: RandomX | Fee: 2%")
    print("=" * 60)
    
    # Create and initialize pool
    pool = ZionRealMiningPool()
    
    try:
        # Initialize pool infrastructure
        await pool.initialize_mining_pool()
        
        print("✅ Pool initialized successfully!")
        print("🌐 Listening on stratum+tcp://0.0.0.0:3333")
        print("⛏️ Ready for miners! Press Ctrl+C to stop.")
        print("=" * 60)
        
        # Keep server running
        while True:
            await asyncio.sleep(10)
            # Pool statistics - simple status check
            print(f"📊 Pool běží - RandomX algoritmus aktivní")
                
    except KeyboardInterrupt:
        print("\n🛑 Shutting down ZION Mining Pool...")
        print("✅ Pool shutdown complete!")
    except Exception as e:
        print(f"❌ Pool error: {e}")

if __name__ == "__main__":
    asyncio.run(run_pool_server())