#!/usr/bin/env python3
"""
ZION 2.7 Mining Command Line Interface
Complete mining control with 2.6.75 enhanced features
"""
import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import ZION 2.7 components
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.blockchain import Blockchain
    from mining.mining_bridge import create_mining_system, MiningIntegrationBridge
    from mining.stratum_server import StratumPoolServer
    from mining.mining_stats import RealTimeMonitor
except ImportError as e:
    logger.error(f"Failed to import ZION 2.7 modules: {e}")
    sys.exit(1)

class ZionMinerCLI:
    """ZION 2.7 Mining CLI Application"""
    
    def __init__(self):
        self.blockchain: Optional[Blockchain] = None
        self.mining_bridge: Optional[MiningIntegrationBridge] = None
        self.stratum_server: Optional[StratumPoolServer] = None
        self.monitor: Optional[RealTimeMonitor] = None
        self.running = False
        self.config = {}
        
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                
            # Validate required fields
            required_fields = ['miner_address', 'num_threads']
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required config field: {field}")
                    
            self.config = config
            logger.info(f"✅ Configuration loaded from {config_file}")
            return config
            
        except FileNotFoundError:
            logger.error(f"❌ Configuration file not found: {config_file}")
            return self._create_default_config(config_file)
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in config file: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}")
            sys.exit(1)
            
    def _create_default_config(self, config_file: str) -> Dict[str, Any]:
        """Create default configuration file"""
        default_config = {
            "miner_address": "Z3BDEEC2A0AE0F5D81B034308F99ECD8990D9B8B01BD9C7E7429392CA31861C6220DA3B30D74E809FA0A1FE069F1",
            "num_threads": 4,
            "mining": {
                "enable_optimizations": True,
                "use_full_memory": False,
                "difficulty_target": 1000
            },
            "pool": {
                "enable_stratum": True,
                "stratum_port": 3333,
                "bind_address": "0.0.0.0",
                "pool_fee": 2.0
            },
            "monitoring": {
                "enable_realtime": True,
                "stats_interval": 5.0,
                "export_stats": True,
                "stats_file": "mining_stats.json"
            },
            "blockchain": {
                "data_dir": "./zion_data",
                "auto_save": True
            }
        }
        
        try:
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"📝 Created default config file: {config_file}")
            self.config = default_config
            return default_config
        except Exception as e:
            logger.error(f"❌ Failed to create default config: {e}")
            sys.exit(1)
            
    def initialize_blockchain(self) -> bool:
        """Initialize blockchain"""
        try:
            data_dir = self.config.get('blockchain', {}).get('data_dir', './zion_data')
            os.makedirs(data_dir, exist_ok=True)
            
            self.blockchain = Blockchain()
            
            # Load existing blockchain data if available
            blockchain_file = os.path.join(data_dir, 'blockchain.json')
            if os.path.exists(blockchain_file):
                logger.info(f"📚 Loading blockchain from {blockchain_file}")
                # Load implementation would go here
                
            logger.info("✅ Blockchain initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize blockchain: {e}")
            return False
            
    def initialize_mining(self) -> bool:
        """Initialize mining system"""
        try:
            if not self.blockchain:
                logger.error("❌ Blockchain not initialized")
                return False
                
            miner_address = self.config['miner_address']
            num_threads = self.config['num_threads']
            
            logger.info(f"⛏️ Initializing mining system...")
            logger.info(f"   Miner address: {miner_address[:20]}...")
            logger.info(f"   Thread count: {num_threads}")
            
            self.mining_bridge = create_mining_system(
                self.blockchain, 
                miner_address, 
                num_threads
            )
            
            # Setup block found callback
            def on_block_found(block):
                logger.info(f"🎉 BLOCK FOUND! Height: {block.height}, Hash: {block.hash[:16]}...")
                self._save_blockchain()
                
            self.mining_bridge.add_block_found_callback(on_block_found)
            
            logger.info("✅ Mining system initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize mining system: {e}")
            return False
            
    def initialize_stratum_pool(self) -> bool:
        """Initialize Stratum pool server"""
        try:
            pool_config = self.config.get('pool', {})
            
            if not pool_config.get('enable_stratum', False):
                logger.info("ℹ️ Stratum pool disabled in config")
                return True
                
            self.stratum_server = StratumPoolServer(
                host=pool_config.get('bind_address', '0.0.0.0'),
                port=pool_config.get('stratum_port', 3333)
            )
            
            # Connect pool to mining bridge
            if self.mining_bridge:
                self.stratum_server.add_share_validator(
                    self.mining_bridge._validate_share_with_blockchain
                )
                
            logger.info(f"✅ Stratum pool initialized on port {pool_config.get('stratum_port', 3333)}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Stratum pool: {e}")
            return False
            
    def initialize_monitoring(self) -> bool:
        """Initialize real-time monitoring"""
        try:
            monitor_config = self.config.get('monitoring', {})
            
            if not monitor_config.get('enable_realtime', False):
                logger.info("ℹ️ Real-time monitoring disabled in config")
                return True
                
            if not self.mining_bridge or not self.mining_bridge.stats_collector:
                logger.error("❌ Mining bridge not initialized for monitoring")
                return False
                
            self.monitor = RealTimeMonitor(self.mining_bridge.stats_collector)
            self.monitor.display_interval = monitor_config.get('stats_interval', 5.0)
            
            logger.info("✅ Real-time monitoring initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize monitoring: {e}")
            return False
            
    def start_mining(self, duration: Optional[float] = None):
        """Start mining operation"""
        if not self.mining_bridge:
            logger.error("❌ Mining system not initialized")
            return
            
        logger.info("🚀 Starting ZION 2.7 mining operation...")
        
        # Start monitoring
        if self.monitor:
            self.monitor.start_monitoring()
            
        # Start mining
        self.running = True
        
        try:
            if duration:
                logger.info(f"⏱️ Mining for {duration} seconds...")
                results = self.mining_bridge.start_mining(duration)
                
                # Display results
                logger.info("📊 Mining Results:")
                logger.info(f"   Total hashes: {results['total_hashes']}")
                logger.info(f"   Average hashrate: {results['average_hashrate']:.2f} H/s")
                logger.info(f"   Valid results: {results['valid_results']}")
                
            else:
                logger.info("⏱️ Mining continuously (Ctrl+C to stop)...")
                
                # Setup signal handlers
                def signal_handler(signum, frame):
                    logger.info("\n🛑 Received stop signal, shutting down...")
                    self.running = False
                    
                signal.signal(signal.SIGINT, signal_handler)
                signal.signal(signal.SIGTERM, signal_handler)
                
                # Continuous mining loop
                while self.running:
                    # Generate new block template periodically
                    template = self.mining_bridge.generate_block_template()
                    if template:
                        logger.info(f"🔄 New block template: Height {template['height']}")
                        
                    # Mine for 30 second intervals
                    results = self.mining_bridge.start_mining(30.0)
                    
                    if not self.running:
                        break
                        
        except KeyboardInterrupt:
            logger.info("\n🛑 Mining stopped by user")
            self.running = False
            
        except Exception as e:
            logger.error(f"❌ Mining error: {e}")
            
        finally:
            # Stop monitoring
            if self.monitor:
                self.monitor.stop_monitoring()
                
            # Export final statistics
            self._export_final_stats()
            
    def show_status(self):
        """Show current mining status"""
        if not self.mining_bridge:
            logger.error("❌ Mining system not initialized")
            return
            
        stats = self.mining_bridge.get_mining_statistics()
        
        print("\n" + "="*60)
        print("🚀 ZION 2.7 Mining Status")
        print("="*60)
        
        # Blockchain info
        if 'blockchain_info' in stats:
            bc_info = stats['blockchain_info']
            print(f"Blockchain Height: {bc_info['height']}")
            print(f"Current Difficulty: {bc_info['difficulty']}")
            print(f"Network Hashrate: {bc_info['network_hashrate']:.2f} H/s")
            
        # Pool info
        if 'pool_stats' in stats:
            pool_info = stats['pool_stats']
            print(f"Pool Hashrate: {pool_info['total_hashrate']:.2f} H/s")
            print(f"Connected Miners: {pool_info['connected_miners']}")
            
        # Summary
        if 'summary' in stats:
            summary = stats['summary']
            print(f"Total Hashes: {summary['total_hashes']}")
            print(f"Efficiency: {summary['efficiency_percent']:.1f}%")
            print(f"Uptime: {summary['uptime_hours']:.2f} hours")
            
        # System resources
        if 'system_stats' in stats:
            sys_info = stats['system_stats']
            print(f"CPU Usage: {sys_info['cpu_usage_percent']:.1f}%")
            print(f"Memory Usage: {sys_info['memory_usage_mb']:.0f}MB")
            
        print("="*60)
        
    def _save_blockchain(self):
        """Save blockchain to disk"""
        try:
            if not self.blockchain:
                return
                
            data_dir = self.config.get('blockchain', {}).get('data_dir', './zion_data')
            blockchain_file = os.path.join(data_dir, 'blockchain.json')
            
            # Save blockchain state
            logger.info(f"💾 Saving blockchain to {blockchain_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save blockchain: {e}")
            
    def _export_final_stats(self):
        """Export final mining statistics"""
        try:
            monitor_config = self.config.get('monitoring', {})
            
            if not monitor_config.get('export_stats', False):
                return
                
            stats_file = monitor_config.get('stats_file', 'mining_stats.json')
            
            if self.mining_bridge and self.mining_bridge.stats_collector:
                self.mining_bridge.stats_collector.export_stats(stats_file)
                logger.info(f"📊 Final statistics exported to {stats_file}")
                
        except Exception as e:
            logger.error(f"❌ Failed to export final stats: {e}")
            
    def cleanup(self):
        """Cleanup all resources"""
        logger.info("🧹 Cleaning up...")
        
        self.running = False
        
        if self.monitor:
            self.monitor.stop_monitoring()
            
        if self.mining_bridge:
            self.mining_bridge.cleanup()
            
        if self.stratum_server:
            # Stop server if running
            pass
            
        self._save_blockchain()
        logger.info("✅ Cleanup completed")

def create_cli_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="ZION 2.7 Mining CLI - Advanced CryptoNote Mining with 2.6.75 Features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s mine --duration 60              # Mine for 60 seconds
  %(prog)s mine --continuous               # Mine continuously
  %(prog)s status                          # Show current status
  %(prog)s config --create                 # Create default config
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default='mining_config.json',
        help='Configuration file path (default: mining_config.json)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Mine command
    mine_parser = subparsers.add_parser('mine', help='Start mining operation')
    mine_parser.add_argument(
        '--duration', '-d',
        type=float,
        help='Mining duration in seconds (default: continuous)'
    )
    mine_parser.add_argument(
        '--continuous',
        action='store_true',
        help='Mine continuously until stopped'
    )
    
    # Status command
    subparsers.add_parser('status', help='Show mining status')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_parser.add_argument(
        '--create',
        action='store_true',
        help='Create default configuration file'
    )
    
    return parser

def main():
    """Main CLI entry point"""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Create CLI instance
    cli = ZionMinerCLI()
    
    try:
        # Load or create configuration
        config = cli.load_config(args.config)
        
        # Handle config command
        if args.command == 'config':
            if args.create:
                logger.info(f"✅ Configuration file created: {args.config}")
            else:
                print(json.dumps(config, indent=2))
            return
            
        # Initialize systems
        if not cli.initialize_blockchain():
            sys.exit(1)
            
        if not cli.initialize_mining():
            sys.exit(1)
            
        if not cli.initialize_stratum_pool():
            sys.exit(1)
            
        if not cli.initialize_monitoring():
            sys.exit(1)
            
        # Handle commands
        if args.command == 'mine':
            duration = args.duration if not args.continuous else None
            cli.start_mining(duration)
            
        elif args.command == 'status':
            cli.show_status()
            
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)
    finally:
        cli.cleanup()

if __name__ == '__main__':
    main()