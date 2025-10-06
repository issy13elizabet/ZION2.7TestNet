#!/Volumes/Zion/zion27_venv/bin/python
"""
ZION 2.7 TestNet Unified Server
FastAPI backend for ZION blockchain with frontend integration
"""
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from zion.rpc.server import ZionRPCServer
from zion.core.blockchain import ZionBlockchain
from zion.mining.randomx_engine import RandomXEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Start ZION 2.7 TestNet server"""
    
    logger.info("🚀 Starting ZION 2.7 TestNet Backend...")
    
    try:
        # Initialize blockchain
        blockchain = ZionBlockchain()
        logger.info("✅ Blockchain initialized")
        
        # Initialize RandomX mining engine
        mining_engine = RandomXEngine()
        logger.info("✅ Mining engine initialized")
        
        # Start RPC server
        rpc_server = ZionRPCServer(blockchain=blockchain, randomx_engine=mining_engine)
        
        logger.info("🌐 Starting FastAPI RPC server on http://localhost:8889")
        logger.info("🎯 Frontend can connect to: http://localhost:8889/api/v1/")
        logger.info("💡 Test API: curl http://localhost:8889/api/v1/health")
        logger.info("🔄 Server will run until manually stopped (Ctrl+C)")
        
        # Run server with signal handling
        await rpc_server.start_server(host="0.0.0.0", port=8889)
        
    except KeyboardInterrupt:
        logger.info("⛔ Shutdown requested by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        logger.info("👋 ZION 2.7 TestNet Backend stopped")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⛔ Shutdown by user")
        sys.exit(0)