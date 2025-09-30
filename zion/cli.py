"""
ZION CLI Interface v2.6.75

Command-line interface for ZION blockchain operations
"""
import asyncio
import argparse
import logging
import sys
from typing import Optional

from .core.blockchain import ZionBlockchain
from .mining.randomx_engine import RandomXEngine
from .rpc.server import ZionRPCServer


def setup_logging(debug: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


async def main_node():
    """Main entry point for ZION node"""
    parser = argparse.ArgumentParser(description="ZION Node v2.6.75")
    parser.add_argument("--host", default="0.0.0.0", help="RPC server host")
    parser.add_argument("--port", type=int, default=18089, help="RPC server port")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--genesis-address", help="Genesis address override")
    
    args = parser.parse_args()
    setup_logging(args.debug)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting ZION Node v2.6.75")
    
    # Initialize blockchain
    blockchain = ZionBlockchain(genesis_address=args.genesis_address)
    
    # Initialize RandomX engine
    randomx_engine = RandomXEngine(fallback_to_sha256=True)
    
    # Start RPC server
    server = ZionRPCServer(blockchain=blockchain, randomx_engine=randomx_engine)
    await server.start_server(args.host, args.port)


def main_miner():
    """Entry point for ZION miner (placeholder for GUI miner)"""
    print("ZION Miner v2.6.75")
    print("GUI miner implementation coming in Week 2...")
    print("For now, use existing zion-real-miner-v2.py")


def main_wallet():
    """Entry point for ZION wallet (placeholder)"""
    print("ZION Wallet v2.6.75") 
    print("Wallet implementation coming in Week 3...")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "node":
        asyncio.run(main_node())
    elif len(sys.argv) > 1 and sys.argv[1] == "miner":
        main_miner()
    elif len(sys.argv) > 1 and sys.argv[1] == "wallet":
        main_wallet()
    else:
        print("ZION v2.6.75 - Python-Native Multi-Chain Blockchain")
        print("Usage:")
        print("  python -m zion.cli node    # Start blockchain node")
        print("  python -m zion.cli miner   # Start miner") 
        print("  python -m zion.cli wallet  # Start wallet")