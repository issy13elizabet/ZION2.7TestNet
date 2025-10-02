#!/usr/bin/env python3
"""
ZION 2.7 + 2.7.1 Integrated CLI
Combines existing 2.7 functionality with new 2.7.1 multi-algorithm system
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Add both 2.7 and integrated paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def cmd_info(args):
    """Show blockchain information"""
    try:
        from core.blockchain import Blockchain
        blockchain = Blockchain()
        
        print("🌟 ZION 2.7 + 2.7.1 Integrated Blockchain Info")
        print("=" * 50)
        print(f"Network: {args.network}")
        print(f"Height: {blockchain.get_height()}")
        print(f"Latest Hash: {blockchain.get_latest_block().hash[:32]}...")
        
        # Show algorithm info if available
        try:
            from mining.config import get_algorithm_info
            algo_info = get_algorithm_info()
            print(f"Mining Algorithm: {algo_info['name']}")
        except ImportError:
            print("Mining Algorithm: Legacy (2.7 compatibility)")
            
    except Exception as e:
        print(f"❌ Error getting blockchain info: {e}")

def cmd_mine(args):
    """Start mining with integrated system"""
    try:
        print(f"⛏️  Starting ZION 2.7 integrated mining...")
        print(f"Address: {args.address}")
        
        # Try to use 2.7.1 mining system first
        try:
            from mining.config import get_global_algorithm
            from core.blockchain import Blockchain
            
            algo = get_global_algorithm()
            print(f"🔧 Using algorithm: {algo.get_name()}")
            
            blockchain = Blockchain()
            
            # Use existing 2.7 mining bridge if available
            try:
                from mining.mining_bridge import create_mining_system
                mining_system = create_mining_system(blockchain, args.address)
                
                print("🚀 Starting 2.7 mining bridge with 2.7.1 algorithms...")
                if hasattr(args, 'blocks') and args.blocks:
                    print(f"Mining {args.blocks} blocks...")
                    # TODO: Implement limited block mining
                
                mining_system.start()
                
            except ImportError:
                print("⚠️  2.7 mining bridge not available, using legacy system")
                # Fallback to basic mining
                
        except ImportError as e:
            print(f"⚠️  2.7.1 algorithms not available: {e}")
            print("🔄 Using legacy 2.7 mining system...")
            
            # Use original 2.7 mining
            from mining.zion_miner import ZionMinerCLI
            miner = ZionMinerCLI()
            miner.start_mining(args.address)
            
    except Exception as e:
        print(f"❌ Mining error: {e}")

def cmd_algorithms(args):
    """Handle algorithm commands (2.7.1 integration)"""
    try:
        from mining.config import list_available_algorithms, get_algorithm_info, set_global_algorithm, benchmark_all_algorithms
        
        if args.algo_command == 'list':
            print("🔧 Available Mining Algorithms (2.7.1 Integration)")
            print("=" * 50)
            algos = list_available_algorithms()
            for name, desc in algos.items():
                print(f"  {name:12} - {desc}")
            
            print(f"\n⚡ Current: {get_algorithm_info()['name']}")
        
        elif args.algo_command == 'set':
            print(f"🔧 Setting algorithm to: {args.algorithm}")
            set_global_algorithm(args.algorithm)
            info = get_algorithm_info()
            print(f"✅ Algorithm set to: {info['name']}")
        
        elif args.algo_command == 'benchmark':
            print("🏃 Benchmarking All Algorithms")
            print("=" * 40)
            results = benchmark_all_algorithms()
            
            print("📊 Algorithm Performance:")
            for name, data in sorted(results.items(), key=lambda x: x[1]['hashrate'], reverse=True):
                print(f"  {name:12} | {data['hashrate']:10.1f} H/s | {data['name']}")
        
        else:
            print("❌ Unknown algorithm command")
            
    except ImportError as e:
        print(f"❌ 2.7.1 algorithm system not available: {e}")
        print("Use original ZION 2.7 mining commands instead")

def cmd_server(args):
    """Start ZION 2.7 server (existing functionality)"""
    try:
        print("🚀 Starting ZION 2.7 Integrated Server...")
        
        # Use existing 2.7 server startup
        from start_zion_27_backend import main
        import asyncio
        asyncio.run(main())
        
    except ImportError:
        print("❌ ZION 2.7 server components not available")
    except Exception as e:
        print(f"❌ Server startup error: {e}")

def cmd_test(args):
    """Run integrated test suite"""
    print("🧪 ZION 2.7 + 2.7.1 Integration Tests")
    print("=" * 40)
    
    # Test 2.7.1 components
    try:
        from mining.algorithms import AlgorithmFactory
        print("✅ 2.7.1 algorithms available")
        
        # Test algorithm creation
        sha256_algo = AlgorithmFactory.create_algorithm('sha256')
        test_data = b"test_integration_data"
        hash_result = sha256_algo.hash(test_data)
        print(f"✅ SHA256 test: {hash_result[:16]}...")
        
    except Exception as e:
        print(f"❌ 2.7.1 algorithm test failed: {e}")
    
    # Test 2.7 components
    try:
        from core.blockchain import Blockchain, Tx
        blockchain = Blockchain()
        print(f"✅ 2.7 blockchain available: height {blockchain.get_height()}")
        
        # Test transaction creation
        tx = Tx.create([], [{'amount': 100, 'recipient': 'test'}], 0)
        print(f"✅ Transaction test: {tx.txid[:16]}...")
        
        # Test txid integrity
        if tx.validate_txid_integrity():
            print("✅ Transaction integrity validated")
        else:
            print("❌ Transaction integrity failed")
        
    except Exception as e:
        print(f"❌ 2.7 blockchain test failed: {e}")
    
    print("\n🎯 Integration test completed!")

def main():
    parser = argparse.ArgumentParser(description='ZION 2.7 + 2.7.1 Integrated CLI')
    
    parser.add_argument(
        '--network', 
        choices=['testnet', 'mainnet'], 
        default='testnet',
        help='Network to use (default: testnet)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show blockchain information')
    
    # Mine command
    mine_parser = subparsers.add_parser('mine', help='Start mining')
    mine_parser.add_argument('--address', required=True, help='Mining address')
    mine_parser.add_argument('--blocks', type=int, help='Maximum blocks to mine')
    
    # Algorithm commands (2.7.1 integration)
    algo_parser = subparsers.add_parser('algorithms', help='Algorithm management (2.7.1)')
    algo_subparsers = algo_parser.add_subparsers(dest='algo_command')
    
    algo_subparsers.add_parser('list', help='List available algorithms')
    
    set_algo_parser = algo_subparsers.add_parser('set', help='Set mining algorithm')
    set_algo_parser.add_argument('algorithm', help='Algorithm name (sha256, randomx, gpu, auto)')
    
    algo_subparsers.add_parser('benchmark', help='Benchmark all algorithms')
    
    # Server command (2.7 functionality)
    server_parser = subparsers.add_parser('server', help='Start ZION 2.7 server')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run integration test suite')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    try:
        if args.command == 'info':
            cmd_info(args)
        elif args.command == 'mine':
            cmd_mine(args)
        elif args.command == 'algorithms':
            cmd_algorithms(args)
        elif args.command == 'server':
            cmd_server(args)
        elif args.command == 'test':
            cmd_test(args)
        else:
            print(f"❌ Unknown command: {args.command}")
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Command failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()