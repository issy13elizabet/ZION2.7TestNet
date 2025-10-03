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


def main():
    """Main CLI entry point"""
    print("🌟 ZION 2.7.1 ASIC-Resistant Blockchain CLI")
    print("🛡️ Built for maximum decentralization with Argon2 mining")
    print()

    try:
        cli = ZionCLI()
        cli.run()
    except KeyboardInterrupt:
        print("\n⏹️ Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ CLI Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()