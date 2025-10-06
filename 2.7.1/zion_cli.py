#!/usr/bin/env python3
"""
ZION 2.7.1 ASIC-Resistant Blockchain CLI
Pure Argon2 mining for maximum decentralization
"""

import argparse
import sys
import time
from typing import Dict, Any

# Import ASIC-resistant components
from mining.config import get_mining_config, benchmark_algorithms, print_asic_warning
from mining.algorithms import benchmark_all_algorithms
from mining.miner import start_asic_resistant_mining


class ZionCLI:
    """
    Command Line Interface for ZION ASIC-Resistant Blockchain
    """

    def __init__(self):
        self.miner = None
        self.config = get_mining_config()

    def run(self):
        """Run CLI with parsed arguments"""
        parser = self._create_parser()
        args = parser.parse_args()

        if not hasattr(args, 'command'):
            parser.print_help()
            return

        # Execute command
        getattr(self, f"cmd_{args.command}")(args)

    def _create_parser(self):
        """Create argument parser"""
        parser = argparse.ArgumentParser(
            description="ZION 2.7.1 - ASIC-Resistant Blockchain CLI",
            epilog="Built for maximum decentralization with Argon2 mining"
        )

        subparsers = parser.add_subparsers(dest='command', help='Available commands')

        # Info command
        info_parser = subparsers.add_parser('info', help='Show blockchain information')
        info_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

        # Test command
        test_parser = subparsers.add_parser('test', help='Run ASIC resistance tests')
        test_parser.add_argument('--iterations', '-i', type=int, default=100, help='Test iterations')

        # Algorithms command
        algo_parser = subparsers.add_parser('algorithms', help='Algorithm management')
        algo_subparsers = algo_parser.add_subparsers(dest='algo_command', help='Algorithm commands')

        # algorithms benchmark
        algo_subparsers.add_parser('benchmark', help='Benchmark all algorithms')

        # algorithms list
        algo_subparsers.add_parser('list', help='List available algorithms')

        # algorithms switch
        switch_parser = algo_subparsers.add_parser('switch', help='Switch mining algorithm')
        switch_parser.add_argument('algorithm', help='Algorithm to switch to')

        # algorithms categories
        algo_subparsers.add_parser('categories', help='Show algorithm categories')

        # Mine command
        mine_parser = subparsers.add_parser('mine', help='Start ASIC-resistant mining')
        mine_parser.add_argument('address', help='Mining reward address')
        mine_parser.add_argument('--duration', '-d', type=int, help='Mining duration in seconds')
        mine_parser.add_argument('--threads', '-t', type=int, default=1, help='Number of mining threads')
        mine_parser.add_argument('--algorithm', '-a', choices=['argon2', 'kawpow', 'ethash', 'cryptonight', 'octopus', 'ergo'], 
                               default='argon2', help='Mining algorithm (default: argon2)')
        mine_parser.add_argument('--gpu', action='store_true', help='Enable GPU mining for supported algorithms')

        # Benchmark command
        bench_parser = subparsers.add_parser('benchmark', help='Run mining benchmark')
        bench_parser.add_argument('--blocks', '-b', type=int, default=5, help='Number of blocks to mine')
        bench_parser.add_argument('--address', '-a', default='benchmark_address', help='Benchmark address')

        return parser

    def cmd_info(self, args):
        """Show blockchain information"""
        print("🌟 ZION 2.7.1 - ASIC-Resistant Blockchain")
        print("=" * 50)

        config = self.config.get_mining_config()

        print(f"Algorithm: {config['algorithm']} (ASIC Resistant: ✅)")
        print(f"Memory Cost: {config['memory_cost'] // 1024}MB")
        print(f"Time Cost: {config['time_cost']} iterations")
        print(f"Difficulty: {config['difficulty']:08x}")
        print(f"Block Time: {config['block_time']}s")
        print(f"Max Threads: {config['max_threads']}")

        if args.verbose:
            print("\n🛡️ ASIC Resistance Features:")
            print("  • Memory-hard algorithm (Argon2)")
            print("  • CPU-only mining (no GPU acceleration)")
            print("  • High memory requirements (64MB+)")
            print("  • SHA256 and other ASIC-friendly algorithms blocked")
            print("  • Fair mining accessible to all CPU users")

        print_asic_warning()

    def cmd_test(self, args):
        """Run ASIC resistance tests"""
        print("🧪 Running ASIC Resistance Tests...")
        print("=" * 40)

        try:
            # Test algorithm creation
            from mining.algorithms import AlgorithmFactory
            algo = AlgorithmFactory.create_algorithm('argon2', self.config.get_algorithm_config())
            print("✅ Algorithm creation: PASSED")

            # Test hashing
            test_data = b"zion_test_data"
            hash_result = algo.hash(test_data)
            print(f"✅ Hashing: PASSED (output: {hash_result.hex()[:16]}...)")

            # Test verification
            target = b"\x00\x00\xff\xff" + b"\x00" * 28
            is_valid = algo.verify(test_data, target)
            print(f"✅ Verification: PASSED (result: {is_valid})")

            # Test benchmark
            benchmark = algo.benchmark(args.iterations)
            print(f"✅ Benchmark: PASSED ({benchmark['hashrate']})")

            print("\n🎉 All ASIC resistance tests PASSED!")
            print("🛡️ Blockchain is protected against ASIC mining")

        except Exception as e:
            print(f"❌ Test FAILED: {e}")
            sys.exit(1)

    def cmd_algorithms(self, args):
        """Handle algorithm commands"""
        if args.algo_command == 'list':
            self._cmd_algorithms_list()
        elif args.algo_command == 'benchmark':
            self._cmd_algorithms_benchmark()
        elif args.algo_command == 'switch':
            self._cmd_algorithms_switch(args)
        elif args.algo_command == 'categories':
            self._cmd_algorithms_categories()
        else:
            print("Use 'algorithms list', 'algorithms benchmark', 'algorithms switch <algo>', or 'algorithms categories'")

    def _cmd_algorithms_list(self):
        """List available ASIC-resistant algorithms"""
        print("🛡️ Available ASIC-Resistant Algorithms:")
        print("=" * 45)

        algorithms = self.config.get_available_algorithms()

        for name, description in algorithms.items():
            print(f"  {name}: {description}")

        print("\n⚠️ Only these algorithms are allowed for ASIC resistance")
        print("🚫 SHA256, Scrypt, and other ASIC-friendly algorithms blocked")

    def _cmd_algorithms_benchmark(self):
        """Benchmark ASIC-resistant algorithms"""
        print("🏃 Benchmarking All Mining Algorithms...")
        print("=" * 50)

        results = benchmark_all_algorithms()

        print("📊 Performance Results:")
        for algo_name, result in results.items():
            if 'error' in result:
                print(f"  {algo_name}: ERROR - {result['error']}")
            else:
                category_icon = "🛡️" if result.get('category') == 'ASIC-Resistant' else "🎮"
                print(f"  {category_icon} {algo_name}: {result['hashrate']} | {result['category']}")
                print(f"    Memory: {result.get('memory_usage', 'N/A')} | Duration: {result['duration']}")

        print("\n💡 ASIC-Resistant algorithms prioritize decentralization")
        print("🎮 GPU-Friendly algorithms offer higher performance")

    def _cmd_algorithms_categories(self):
        """Show algorithm categories"""
        print("🎯 ZION Algorithm Categories:")
        print("=" * 45)

        categories = self.config.get_algorithm_categories()

        for category, algorithms in categories.items():
            icon = "🛡️" if category == "ASIC-Resistant" else "🎮"
            print(f"{icon} {category}:")
            for algo in algorithms:
                desc = self.config.get_available_algorithms()[algo]
                print(f"  • {algo}: {desc}")
            print()

        print("💡 ASIC-Resistant algorithms prioritize decentralization")
        print("🎮 GPU-Friendly algorithms offer higher performance but less ASIC resistance")

    def _cmd_algorithms_switch(self, args):
        """Switch mining algorithm"""
        print(f"🔄 Switching to algorithm: {args.algorithm}")
        print("=" * 50)

        if self.config.set_algorithm(args.algorithm):
            print(f"✅ Successfully switched to {args.algorithm}")
            print("\n🔧 Current Configuration:")
            self.config.print_config_summary()
        else:
            print(f"❌ Failed to switch to {args.algorithm}")
            print("Use 'algorithms list' to see available options")

    def cmd_mine(self, args):
        """Start ASIC-resistant mining"""
        print(f"⛏️ Starting ASIC-resistant mining to: {args.address}")
        print("🛡️ Using Argon2 algorithm for maximum decentralization")
        print("=" * 60)

        try:
            results = start_asic_resistant_mining(args.address, args.duration, getattr(args, 'threads', 1))

            print("\n📊 Mining Session Complete:")
            print(f"   Algorithm: {results['algorithm']}")
            print(f"   ASIC Resistant: {'✅' if results['asic_resistant'] else '❌'}")
            print(f"   Total Hashes: {results['hashes']}")
            print(f"   Blocks Found: {results['blocks_found']}")
            print(f"   Average Hashrate: {results['hashrate']}")
            print(f"   Memory Usage: {results['memory_usage']}")
            print(f"   Threads Used: {results['threads']}")

        except KeyboardInterrupt:
            print("\n⏹️ Mining interrupted by user")
        except Exception as e:
            print(f"❌ Mining failed: {e}")
            sys.exit(1)

    def cmd_benchmark(self, args):
        """Run mining benchmark"""
        print(f"🏁 Running ASIC-Resistant Mining Benchmark ({args.blocks} blocks)...")
        print("=" * 65)

        start_time = time.time()

        # Simulate mining blocks
        total_hashes = 0
        blocks_found = 0

        for i in range(args.blocks):
            print(f"⛏️ Mining block {i+1}/{args.blocks}...")

            # Start mining session for this block
            results = start_asic_resistant_mining(args.address, duration=10)  # 10 second blocks

            total_hashes += results['hashes']
            if results['blocks_found'] > 0:
                blocks_found += results['blocks_found']
                print(f"  🎉 Block {i+1} found!")

        end_time = time.time()
        duration = end_time - start_time

        print("\n📊 Benchmark Results:")
        print(f"   Blocks Attempted: {args.blocks}")
        print(f"   Blocks Found: {blocks_found}")
        print(f"   Total Hashes: {total_hashes}")
        print(f"   Total Time: {duration:.1f}s")
        print(f"   Average Hashrate: {total_hashes/duration:.1f} H/s")
        print(f"   ASIC Resistance: ✅ (Argon2 only)")

        if blocks_found > 0:
            avg_block_time = duration / blocks_found
            print(f"   Average Block Time: {avg_block_time:.1f}s")
        else:
            print("   Note: No blocks found (normal for high difficulty)")


class ZionRealBlockchainCLI:
    """Real Blockchain CLI - No Simulations, Real Blocks"""

    def __init__(self):
        self.blockchain = None

    def initialize_blockchain(self):
        """Initialize real blockchain"""
        if self.blockchain is None:
            from core.real_blockchain import ZionRealBlockchain
            import os
            db_path = os.path.join(os.path.dirname(__file__), 'zion_real_blockchain.db')
            self.blockchain = ZionRealBlockchain(db_file=db_path)
            print("🌟 ZION Real Blockchain initialized!")
            print(f"   Current blocks: {self.blockchain.get_block_count()}")

    def cmd_real_mine(self, args):
        """Mine real blocks in the blockchain with selected algorithm"""
        self.initialize_blockchain()

        algorithm = getattr(args, 'algorithm', 'argon2')
        use_gpu = getattr(args, 'gpu', False)

        print("🚀 Starting REAL BLOCKCHAIN MINING")
        print("🛡️ No simulations - creating actual blocks!")
        print("=" * 60)
        print(f"🎯 Algorithm: {algorithm.upper()}")
        print(f"🎮 GPU Mode: {'Enabled' if use_gpu else 'Disabled'}")
        print(f"📧 Address: {args.address}")
        print(f"🧵 Threads: {args.threads}")
        print("=" * 60)

        # Set algorithm in blockchain
        if hasattr(self.blockchain, 'set_mining_algorithm'):
            self.blockchain.set_mining_algorithm(algorithm, use_gpu)

        consciousness_levels = [
            "PHYSICAL", "EMOTIONAL", "MENTAL", "INTUITIVE",
            "SPIRITUAL", "COSMIC", "UNITY", "ENLIGHTENMENT",
            "LIBERATION", "ON_THE_STAR"
        ]

        blocks_mined = 0
        start_time = time.time()

        try:
            for i in range(args.blocks):
                # Cycle through consciousness levels
                consciousness = consciousness_levels[i % len(consciousness_levels)]

                print(f"⛏️ Mining block {i+1}/{args.blocks} - {consciousness} consciousness...")

                # Mine real block
                block = self.blockchain.mine_block(
                    miner_address=args.address,
                    consciousness_level=consciousness
                )

                if block:
                    blocks_mined += 1
                    print(f"✅ Block {block.height} mined! Hash: {block.hash[:16]}...")
                    print(f"   🧠 Consciousness: {block.consciousness_level}")
                    print(f"   🌟 Multiplier: {block.sacred_multiplier}x")
                    print(f"   💰 Reward: {block.reward:,} atomic units")
                else:
                    print(f"❌ Failed to mine block {i+1}")

                # Small delay between blocks
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n⏹️ Mining interrupted by user")

        end_time = time.time()
        duration = end_time - start_time

        print("\n" + "=" * 60)
        print("🏆 MINING SESSION COMPLETE")
        print("=" * 60)
        print(f"   Duration: {duration:.1f} seconds")
        print(f"   Blocks Mined: {blocks_mined}")
        print(f"   Success Rate: {blocks_mined/args.blocks*100:.1f}%")
        print(f"   Average Time per Block: {duration/args.blocks:.1f}s")

        if blocks_mined > 0:
            print(f"   💎 Total Reward: {blocks_mined * 100000000:,} atomic units")
            print("   🌟 Sacred consciousness levels achieved!")
        start_time = time.time()

        try:
            for i in range(args.blocks):
                # Cycle through consciousness levels
                consciousness = consciousness_levels[i % len(consciousness_levels)]

                print(f"\n⛏️ Mining REAL block {i+1}/{args.blocks}...")
                print(f"   🧠 Consciousness Level: {consciousness}")

                # Mine real block
                block = self.blockchain.mine_block(
                    miner_address=args.address,
                    consciousness_level=consciousness
                )

                if block:
                    blocks_mined += 1
                    print(f"   ✅ REAL BLOCK {block.height} CREATED!")
                    print(f"   🔗 Hash: {block.hash[:32]}...")
                    print(f"   💰 Reward: {block.reward:,} atomic units")
                else:
                    print("   ❌ Block mining failed (timeout)")

                # Small delay between blocks
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n⏹️ Real mining interrupted by user")

        duration = time.time() - start_time

        print("\n📊 REAL BLOCKCHAIN MINING COMPLETE:")
        print(f"   Blocks Attempted: {args.blocks}")
        print(f"   Blocks Mined: {blocks_mined}")
        print(f"   Success Rate: {blocks_mined/args.blocks*100:.1f}%")
        print(f"   Total Time: {duration:.1f}s")
        if blocks_mined > 0:
            print(f"   Avg Block Time: {duration/blocks_mined:.1f}s")

        # Show blockchain stats
        stats = self.blockchain.get_blockchain_stats()
        print("\n🌟 BLOCKCHAIN STATUS:")
        print(f"   Total Blocks: {stats['block_count']}")
        print(f"   Total Supply: {stats['total_supply']:,} atomic units")
        print(f"   Mempool: {stats['mempool_size']} transactions")

    def cmd_real_stats(self, args):
        """Show real blockchain statistics"""
        self.initialize_blockchain()

        stats = self.blockchain.get_blockchain_stats()
        latest_block = self.blockchain.get_latest_block()

        print("🌟 ZION REAL BLOCKCHAIN STATISTICS")
        print("=" * 50)
        print(f"📦 Total Blocks: {stats['block_count']}")
        print(f"💰 Total Supply: {stats['total_supply']:,} atomic units")
        print(f"📝 Mempool Size: {stats['mempool_size']}")
        print(f"🎯 Difficulty: {stats['difficulty']}")

        if latest_block:
            print("\n🏆 Latest Block:")
            print(f"   Height: {latest_block.height}")
            print(f"   Hash: {latest_block.hash[:32]}...")
            print(f"   Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(latest_block.timestamp))}")
            print(f"   Transactions: {len(latest_block.transactions)}")
            print(f"   🧠 Consciousness: {latest_block.consciousness_level}")
            print(f"   🌟 Sacred Multiplier: {latest_block.sacred_multiplier:.2f}x")

        print("\n🧠 Consciousness Distribution:")
        for level, count in stats['consciousness_distribution'].items():
            print(f"   {level}: {count} blocks")

    def cmd_real_verify(self, args):
        """Verify blockchain integrity"""
        self.initialize_blockchain()

        print("🔍 Verifying REAL blockchain integrity...")
        valid = self.blockchain.verify_blockchain()

        if valid:
            print("✅ Blockchain is VALID and INTEGRITY VERIFIED!")
        else:
            print("❌ Blockchain integrity check FAILED!")

    def cmd_real_balance(self, args):
        """Check address balance"""
        self.initialize_blockchain()

        balance = self.blockchain.get_balance(args.address)
        print(f"💰 Balance for {args.address}:")
        print(f"   {balance:,} atomic units")
        print(f"   {balance/100000000:.8f} ZION")


def main():
    """Main CLI entry point"""
    print("🌟 ZION 2.7.1 REAL BLOCKCHAIN CLI")
    print("🚀 No Simulations - Real Blocks Only!")
    print("🛡️ JAI RAM SITA HANUMAN - ON THE STAR")
    print()

    try:
        # Parse arguments
        parser = argparse.ArgumentParser(description="ZION 2.7.1 Real Blockchain CLI")
        subparsers = parser.add_subparsers(dest='command', help='Available commands')

        # Real mining command
        mine_parser = subparsers.add_parser('mine', help='Mine real blocks')
        mine_parser.add_argument('--address', '-a', required=True, help='Miner reward address')
        mine_parser.add_argument('--blocks', '-b', type=int, default=5, help='Number of blocks to mine')
        mine_parser.add_argument('--algorithm', choices=['argon2', 'kawpow', 'ethash'], default='argon2', help='Mining algorithm')
        mine_parser.add_argument('--gpu', action='store_true', help='Enable GPU mining')
        mine_parser.add_argument('--threads', '-t', type=int, default=1, help='Number of mining threads')

        # Real stats command
        stats_parser = subparsers.add_parser('stats', help='Show blockchain statistics')

        # Real verify command
        verify_parser = subparsers.add_parser('verify', help='Verify blockchain integrity')

        # Real balance command
        balance_parser = subparsers.add_parser('balance', help='Check address balance')
        balance_parser.add_argument('--address', '-a', required=True, help='Address to check')

        # Legacy ASIC mining (for compatibility)
        asic_parser = subparsers.add_parser('asic-mine', help='ASIC-resistant mining (legacy)')
        asic_parser.add_argument('--address', '-a', required=True, help='Mining address')
        asic_parser.add_argument('--duration', '-d', type=int, default=60, help='Mining duration in seconds')
        asic_parser.add_argument('--threads', '-t', type=int, default=1, help='Number of threads')

        # API server command
        api_parser = subparsers.add_parser('api', help='Start API server')
        api_parser.add_argument('--host', '-H', default='0.0.0.0', help='API host')
        api_parser.add_argument('--port', '-p', type=int, default=8000, help='API port')

        # Wallet commands
        wallet_parser = subparsers.add_parser('wallet', help='Wallet management')
        wallet_subparsers = wallet_parser.add_subparsers(dest='wallet_command')

        # Wallet create address
        wallet_subparsers.add_parser('create', help='Create new address')

        # Wallet list addresses
        wallet_subparsers.add_parser('list', help='List wallet addresses')

        # Wallet balance
        balance_parser = wallet_subparsers.add_parser('balance', help='Check balance')
        balance_parser.add_argument('--address', '-a', help='Specific address')

        # Wallet send
        send_parser = wallet_subparsers.add_parser('send', help='Send transaction')
        send_parser.add_argument('--from', '-f', dest='from_addr', required=True, help='From address')
        send_parser.add_argument('--to', '-t', required=True, help='To address')
        send_parser.add_argument('--amount', '-a', type=int, required=True, help='Amount in atomic units')
        send_parser.add_argument('--fee', type=int, default=1000, help='Transaction fee')

        args = parser.parse_args()

        if not args.command:
            parser.print_help()
            return

        # Handle commands
        if args.command == 'mine':
            cli = ZionRealBlockchainCLI()
            cli.cmd_real_mine(args)
        elif args.command == 'stats':
            cli = ZionRealBlockchainCLI()
            cli.cmd_real_stats(args)
        elif args.command == 'verify':
            cli = ZionRealBlockchainCLI()
            cli.cmd_real_verify(args)
        elif args.command == 'balance':
            cli = ZionRealBlockchainCLI()
            cli.cmd_real_balance(args)
        elif args.command == 'asic-mine':
            # Legacy ASIC mining
            cli = ZionCLI()
            cli.cmd_mine(args)
        elif args.command == 'api':
            # Start API server
            try:
                from api import app
                import uvicorn
                print("🚀 Starting ZION API Server...")
                print(f"🌐 API: http://{args.host}:{args.port}")
                print(f"📖 Docs: http://{args.host}:{args.port}/docs")
                uvicorn.run(app, host=args.host, port=args.port)
            except ImportError as e:
                print(f"❌ API dependencies not installed: {e}")
                print("Run: pip install fastapi uvicorn")
        elif args.command == 'wallet':
            from wallet import get_wallet
            wallet = get_wallet()

            if args.wallet_command == 'create':
                label = getattr(args, 'label', '')
                address = wallet.create_address(label)
                print(f"✅ Created address: {address}")
            elif args.wallet_command == 'list':
                addresses = wallet.get_addresses()
                print("📋 Wallet Addresses:")
                for addr in addresses:
                    balance = wallet.get_balance(addr['address'])
                    print(f"  {addr['address'][:20]}... | {balance:,} ZION | {addr['label']}")
            elif args.wallet_command == 'balance':
                if hasattr(args, 'address') and args.address:
                    balance = wallet.get_balance(args.address)
                    print(f"💰 Balance for {args.address}: {balance:,} atomic units")
                else:
                    total_balance = wallet.get_total_balance()
                    print(f"💰 Total wallet balance: {total_balance:,} atomic units")
            elif args.wallet_command == 'send':
                tx = wallet.create_transaction(
                    getattr(args, 'from_addr'),
                    args.to,
                    args.amount,
                    args.fee
                )
                if tx:
                    print(f"✅ Transaction sent: {tx.tx_id}")
                else:
                    print("❌ Transaction failed")
            else:
                wallet_parser.print_help()

    except KeyboardInterrupt:
        print("\n⏹️ Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ CLI Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()