#!/usr/bin/env python3
"""
🕉️ ZION CLI - Command Line Interface 🕉️
Sacred Technology Liberation Platform - Core CLI
Version: 2.6.75

Usage:
  python -m zion.cli --help
  python -m zion.cli status
  python -m zion.cli wallet --create
  python -m zion.cli mining --start
"""

import argparse
import sys
import os
import asyncio
from pathlib import Path

# Add ZION path for imports
ZION_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ZION_ROOT))

class ZionCLI:
    """ZION Command Line Interface"""
    
    def __init__(self):
        self.version = "2.6.75"
        
    async def system_status(self):
        """Show ZION system status"""
        print("🚀 ZION SYSTEM STATUS 🚀")
        print("=" * 30)
        
        # Test core components
        try:
            from zion.wallet.wallet_core import ZionWallet
            print("✅ Wallet Core: Available")
        except ImportError:
            print("❌ Wallet Core: Not available")
        
        try:
            from zion_ai_miner_14_integration import ZionAIMiner14Integration
            print("✅ AI Miner: Available")
        except ImportError:
            print("❌ AI Miner: Not available")
            
        try:
            from multi_chain_bridge_manager import ZionMultiChainBridgeManager
            print("✅ Multi-Chain Bridge: Available")
        except ImportError:
            print("❌ Multi-Chain Bridge: Not available")
            
        try:
            from lightning_network_service import ZionLightningService
            print("✅ Lightning Network: Available") 
        except ImportError:
            print("❌ Lightning Network: Not available")
            
        print(f"\n🕉️ ZION Version: {self.version}")
        
    async def wallet_operations(self, args):
        """Handle wallet operations"""
        print("💼 ZION WALLET OPERATIONS 💼")
        print("=" * 35)
        
        try:
            from zion.wallet.wallet_core import ZionWallet
            wallet = ZionWallet()
            
            if args.create:
                print("🔑 Creating new wallet...")
                # Basic wallet creation without password for CLI demo
                result = wallet.create_wallet("demo_passphrase_2025", "demo_passphrase_2025")
                print(f"✅ Wallet creation result: {result}")
                
            elif args.balance:
                print("💰 Checking wallet balance...")
                balance = wallet.get_balance()
                print(f"Balance: {balance}")
                
            elif args.address:
                print("📬 Getting wallet address...")
                if hasattr(wallet, 'address'):
                    print(f"Address: {wallet.address}")
                else:
                    print("⚠️  Address not available - wallet needs to be unlocked")
                    
            else:
                print("Available wallet commands:")
                print("  --create    Create new wallet")
                print("  --balance   Check wallet balance") 
                print("  --address   Show wallet address")
                
        except ImportError as e:
            print(f"❌ Wallet not available: {e}")
            
    async def mining_operations(self, args):
        """Handle mining operations"""
        print("⛏️  ZION MINING OPERATIONS ⛏️")
        print("=" * 35)
        
        try:
            from zion_ai_miner_14_integration import ZionAIMiner14Integration as ZionAIMiner
            
            if args.start:
                print("🚀 Starting AI miner...")
                miner = ZionAIMiner()
                
                # Show miner info
                print(f"✅ AI Miner loaded: {type(miner).__name__}")
                
                # Demo mining simulation
                print("⚡ Running mining simulation...")
                result = await miner.simulate_mining_cycle()
                print(f"Mining result: {result}")
                
            elif args.status:
                print("📊 Mining status...")
                miner = ZionAIMiner()
                status = miner.get_ai_miner_status()
                print(f"Status: {status}")
                print(f"Total hashes: {miner.total_hashes}")
                print(f"Miner status: {miner.status.name}")
                
            else:
                print("Available mining commands:")
                print("  --start     Start AI mining")
                print("  --status    Check mining status")
                
        except ImportError as e:
            print(f"❌ AI Miner not available: {e}")
        except Exception as e:
            print(f"⚠️  Mining operation error: {e}")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="🕉️ ZION Sacred Technology Platform CLI 🕉️",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--version', action='version', version='ZION CLI 2.6.75')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    
    # Wallet commands
    wallet_parser = subparsers.add_parser('wallet', help='Wallet operations')
    wallet_parser.add_argument('--create', action='store_true', help='Create new wallet')
    wallet_parser.add_argument('--balance', action='store_true', help='Check balance')
    wallet_parser.add_argument('--address', action='store_true', help='Show address')
    
    # Mining commands
    mining_parser = subparsers.add_parser('mining', help='Mining operations')
    mining_parser.add_argument('--start', action='store_true', help='Start mining')
    mining_parser.add_argument('--status', action='store_true', help='Mining status')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
        
    # Initialize CLI
    cli = ZionCLI()
    
    # Run appropriate command
    try:
        if args.command == 'status':
            asyncio.run(cli.system_status())
        elif args.command == 'wallet':
            asyncio.run(cli.wallet_operations(args))
        elif args.command == 'mining':
            asyncio.run(cli.mining_operations(args))
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
    except Exception as e:
        print(f"❌ CLI Error: {e}")

if __name__ == "__main__":
    main()