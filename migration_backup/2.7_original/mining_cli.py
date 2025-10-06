#!/usr/bin/env python3
"""
ZION 2.7 Mining CLI Interface
Command-line interface for mining operations and pool management
"""
import sys
import os
import time
import argparse
import json
import threading
import signal
from typing import Optional

# Setup imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.blockchain import Blockchain
from mining.mining_bridge import create_mining_system, MiningIntegrationBridge
from mining.randomx_engine import RandomXEngine
from mining.stratum_server import StratumPoolServer
from mining.mining_stats import RealTimeMonitor

class ZionMiningCLI:
    """ZION 2.7 Mining Command Line Interface"""
    
    def __init__(self):
        self.mining_system: Optional[MiningIntegrationBridge] = None
        self.blockchain: Optional[Blockchain] = None
        self.monitor: Optional[RealTimeMonitor] = None
        self.running = False
        
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            print("\n🛑 Stopping ZION mining...")
            self.stop_mining()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    def start_solo_mining(self, args):
        """Start solo mining"""
        print("🚀 Starting ZION 2.7 Solo Mining")
        print("=" * 50)
        
        try:
            # Create blockchain
            self.blockchain = Blockchain()
            print(f"✅ Blockchain loaded (Height: {self.blockchain.height})")
            
            # Create mining system
            self.mining_system = create_mining_system(
                self.blockchain,
                args.address,
                num_threads=args.threads
            )
            print(f"✅ Mining system initialized ({args.threads} threads)")
            
            # Start real-time monitoring if requested
            if args.monitor:
                self.monitor = RealTimeMonitor(self.mining_system.stats_collector)
                self.monitor.start_monitoring()
                print("✅ Real-time monitoring started")
            
            # Add block found callback
            def on_block_found(block):
                print(f"\n🎉 BLOCK FOUND! Height: {block.height}, Hash: {block.hash[:16]}...")
                
            self.mining_system.add_block_found_callback(on_block_found)
            
            # Display initial info
            template = self.mining_system.generate_block_template()
            if template:
                print(f"📋 Mining template: Height {template['height']}, Difficulty {template['difficulty']}")
            
            # Start mining
            self.running = True
            print(f"⛏️ Mining started with address: {args.address[:20]}...")
            print("Press Ctrl+C to stop\n")
            
            # Mining loop
            while self.running:
                try:
                    # Run mining in batches
                    results = self.mining_system.start_mining(duration=60.0)
                    
                    if not self.running:
                        break
                        
                    # Display periodic stats
                    stats = self.mining_system.get_mining_statistics()
                    summary = stats.get('summary', {})
                    
                    print(f"📊 Mining Report:")
                    print(f"   Hashes: {summary.get('total_hashes', 0)}")
                    print(f"   Shares: {summary.get('accepted_shares', 0)}/{summary.get('total_shares', 0)}")
                    print(f"   Efficiency: {summary.get('efficiency_percent', 0):.1f}%")
                    print()
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ Mining error: {e}")
                    time.sleep(5)
                    
        except Exception as e:
            print(f"❌ Failed to start solo mining: {e}")
            return False
            
        return True
        
    def start_pool_server(self, args):
        """Start mining pool server"""
        print("🏊 Starting ZION 2.7 Mining Pool Server")
        print("=" * 50)
        
        try:
            # Create blockchain
            self.blockchain = Blockchain()
            print(f"✅ Blockchain loaded (Height: {self.blockchain.height})")
            
            # Create Stratum server
            server = StratumPoolServer(host=args.host, port=args.port)
            print(f"✅ Stratum server created ({args.host}:{args.port})")
            
            # Add blockchain validator
            from mining.stratum_server import create_blockchain_validator
            server.add_share_validator(create_blockchain_validator(self.blockchain))
            
            # Generate initial job
            job = server.generate_mining_job()
            print(f"📋 Initial job generated: {job.job_id}")
            
            self.running = True
            print("🎯 Pool server ready for connections")
            print("Press Ctrl+C to stop\n")
            
            # Server loop
            last_stats_time = time.time()
            
            while self.running:
                try:
                    # Generate new jobs periodically
                    time.sleep(30)  # New job every 30 seconds
                    
                    if not self.running:
                        break
                        
                    new_job = server.generate_mining_job()
                    print(f"🔄 New job generated: {new_job.job_id}")
                    
                    # Display stats periodically
                    if time.time() - last_stats_time > 60:  # Every minute
                        stats = server.get_pool_statistics()
                        print(f"📊 Pool Stats:")
                        print(f"   Miners: {stats['total_miners']}")
                        print(f"   Hashrate: {stats['total_hashrate']:.2f} H/s")
                        print(f"   Shares: {stats['shares_accepted']}/{stats['shares_rejected']}")
                        print(f"   Blocks: {stats['blocks_found']}")
                        print()
                        last_stats_time = time.time()
                        
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ Pool server error: {e}")
                    time.sleep(5)
                    
        except Exception as e:
            print(f"❌ Failed to start pool server: {e}")
            return False
            
        return True
        
    def test_randomx(self, args):
        """Test RandomX engine functionality"""
        print("🧪 Testing ZION 2.7 RandomX Engine")
        print("=" * 50)
        
        try:
            # Create and initialize engine
            engine = RandomXEngine(fallback_to_sha256=True)
            seed = b'ZION_2_7_TEST_SEED'
            
            print("🔧 Initializing RandomX engine...")
            success = engine.init(seed, full_mem=args.full_memory)
            
            if not success:
                print("❌ RandomX initialization failed")
                return False
                
            print(f"✅ RandomX initialized (fallback: {engine.use_fallback})")
            
            # Performance test
            print(f"⚡ Running performance test ({args.duration}s)...")
            
            start_time = time.time()
            hash_count = 0
            
            while time.time() - start_time < args.duration:
                test_data = f"zion_test_data_{hash_count}".encode()
                _ = engine.hash(test_data)
                hash_count += 1
                
            elapsed = time.time() - start_time
            hashrate = hash_count / elapsed
            
            # Get performance stats
            stats = engine.get_performance_stats()
            
            print(f"📊 Performance Results:")
            print(f"   Hashes: {hash_count}")
            print(f"   Time: {elapsed:.2f}s")
            print(f"   Hashrate: {hashrate:.2f} H/s")
            print(f"   Avg Hash Time: {stats.get('avg_time_ms', 0):.3f}ms")
            print(f"   Current Hashrate: {stats.get('hashrate', 0):.2f} H/s")
            print(f"   Memory Usage: {stats['memory_usage_mb']:.1f} MB")
            print(f"   RandomX Available: {stats['randomx_available']}")
            
            engine.cleanup()
            print("✅ RandomX test completed")
            
        except Exception as e:
            print(f"❌ RandomX test failed: {e}")
            return False
            
        return True
        
    def show_status(self, args):
        """Show mining system status"""
        print("📊 ZION 2.7 Mining System Status")
        print("=" * 50)
        
        try:
            # Show blockchain status
            blockchain = Blockchain()
            info = blockchain.info()
            
            print("🔗 Blockchain Status:")
            print(f"   Height: {info['height']}")
            print(f"   Network: {info['network_type']}")
            print(f"   Difficulty: {info.get('difficulty', 'N/A')}")
            print(f"   Block Time: {info.get('block_time', 'N/A')}s")
            print()
            
            # Test RandomX availability
            print("🧮 RandomX Status:")
            try:
                engine = RandomXEngine()
                if engine.init(b'ZION_2_7_TEST'):
                    stats = engine.get_performance_stats()
                    print(f"   Available: {stats['randomx_available']}")
                    print(f"   Fallback Mode: {stats['fallback_mode']}")
                    print(f"   Flags: {stats['current_flags']}")
                    print(f"   Memory: {stats['memory_usage_mb']:.1f} MB")
                    engine.cleanup()
                else:
                    print(f"   Status: Initialization failed")
            except Exception as e:
                print(f"   Error: {e}")
                
            print()
            
            # Show system resources
            try:
                import psutil
                print("💻 System Resources:")
                print(f"   CPU Cores: {psutil.cpu_count()}")
                print(f"   CPU Usage: {psutil.cpu_percent()}%")
                memory = psutil.virtual_memory()
                print(f"   Memory: {memory.used//1024//1024}MB / {memory.total//1024//1024}MB")
                print(f"   Memory Available: {memory.available//1024//1024}MB")
            except:
                print("💻 System info unavailable")
                
        except Exception as e:
            print(f"❌ Status check failed: {e}")
            return False
            
        return True
        
    def stop_mining(self):
        """Stop mining operations"""
        self.running = False
        
        if self.monitor:
            self.monitor.stop_monitoring()
            
        if self.mining_system:
            self.mining_system.cleanup()
            
        print("✅ Mining stopped")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="ZION 2.7 Mining CLI - Advanced mining operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s solo --address Z3B... --threads 4
  %(prog)s pool --host 0.0.0.0 --port 3333
  %(prog)s test --duration 10
  %(prog)s status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Solo mining command
    solo_parser = subparsers.add_parser('solo', help='Start solo mining')
    solo_parser.add_argument('--address', required=True, help='Miner wallet address')
    solo_parser.add_argument('--threads', type=int, default=4, help='Number of mining threads (default: 4)')
    solo_parser.add_argument('--monitor', action='store_true', help='Enable real-time monitoring')
    
    # Pool server command
    pool_parser = subparsers.add_parser('pool', help='Start mining pool server')
    pool_parser.add_argument('--host', default='0.0.0.0', help='Server bind address (default: 0.0.0.0)')
    pool_parser.add_argument('--port', type=int, default=3333, help='Server port (default: 3333)')
    
    # RandomX test command
    test_parser = subparsers.add_parser('test', help='Test RandomX engine')
    test_parser.add_argument('--duration', type=float, default=5.0, help='Test duration in seconds (default: 5)')
    test_parser.add_argument('--full-memory', action='store_true', help='Use full memory mode (2GB dataset)')
    
    # Status command
    subparsers.add_parser('status', help='Show system status')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
        
    # Create CLI instance
    cli = ZionMiningCLI()
    cli.setup_signal_handlers()
    
    # Execute command
    try:
        if args.command == 'solo':
            success = cli.start_solo_mining(args)
        elif args.command == 'pool':
            success = cli.start_pool_server(args)
        elif args.command == 'test':
            success = cli.test_randomx(args)
        elif args.command == 'status':
            success = cli.show_status(args)
        else:
            print(f"❌ Unknown command: {args.command}")
            return 1
            
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        return 0
    except Exception as e:
        print(f"❌ CLI error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())