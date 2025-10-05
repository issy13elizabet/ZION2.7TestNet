#!/usr/bin/env python3
"""
Simple ZION Pool Launcher for Testing Yescrypt Support
"""

import asyncio
import sys
import signal
from zion_universal_pool_v2 import ZionUniversalPool

class PoolLauncher:
    def __init__(self):
        self.pool = None
        
    def signal_handler(self, signum, frame):
        print("\n🛑 Shutting down pool...")
        if self.pool:
            # Graceful shutdown would go here
            pass
        sys.exit(0)
    
    async def start(self):
        try:
            # Register signal handler
            signal.signal(signal.SIGINT, self.signal_handler)
            
            # Create pool
            self.pool = ZionUniversalPool(port=4444)
            
            print("🚀 ZION Universal Pool v2.7.1 - Yescrypt Ready")
            print("=" * 55)
            print("⚡ Stratum Port: 4444")
            print("🌐 API Port: 4445") 
            print("🌱 Eco Algorithms:")
            print("   • Yescrypt (CPU): +15% eco bonus")
            print("   • Autolykos v2 (GPU): +20% eco bonus")
            print("   • RandomX (CPU): Standard rate")
            print("🔧 Features: Variable difficulty, IP banning, REST API")
            print("💾 Database: Persistent storage enabled")
            print("=" * 55)
            
            # Start the server
            await self.pool.start_server()
            
        except KeyboardInterrupt:
            print("\n🛑 Pool stopped by user")
        except Exception as e:
            print(f"❌ Pool error: {e}")
            import traceback
            traceback.print_exc()

def main():
    launcher = PoolLauncher()
    try:
        asyncio.run(launcher.start())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()