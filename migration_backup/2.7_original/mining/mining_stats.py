"""
ZION 2.7 Mining Statistics and Performance Monitor
Comprehensive mining analytics with 2.6.75 enhanced features
"""
from __future__ import annotations
import time
import threading
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from collections import deque
import psutil

logger = logging.getLogger(__name__)

@dataclass
class HashingStats:
    """Individual mining thread statistics"""
    thread_id: int
    total_hashes: int = 0
    accepted_shares: int = 0
    rejected_shares: int = 0
    start_time: float = field(default_factory=time.time)
    last_hash_time: float = 0.0
    peak_hashrate: float = 0.0
    avg_hashrate: float = 0.0
    current_difficulty: float = 1000.0
    errors: int = 0
    
    def update_hash_count(self, count: int = 1):
        """Update hash count and calculate hashrate"""
        self.total_hashes += count
        current_time = time.time()
        self.last_hash_time = current_time
        
        # Calculate average hashrate
        elapsed = current_time - self.start_time
        if elapsed > 0:
            self.avg_hashrate = self.total_hashes / elapsed
            
    def add_share(self, accepted: bool):
        """Add share result"""
        if accepted:
            self.accepted_shares += 1
        else:
            self.rejected_shares += 1
            
    def add_error(self):
        """Add error count"""
        self.errors += 1

@dataclass
class PoolStats:
    """Mining pool statistics"""
    pool_name: str
    connected_miners: int = 0
    total_hashrate: float = 0.0
    total_shares_accepted: int = 0
    total_shares_rejected: int = 0
    blocks_found: int = 0
    difficulty: float = 1000.0
    uptime_seconds: float = 0.0
    network_hashrate: float = 0.0
    next_difficulty_adjustment: float = 0.0

@dataclass
class SystemStats:
    """System resource statistics"""
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    memory_available_mb: float = 0.0
    temperature_celsius: float = 0.0
    power_watts: float = 0.0
    network_rx_mbps: float = 0.0
    network_tx_mbps: float = 0.0

class MiningStatsCollector:
    """Comprehensive mining statistics collector"""
    
    def __init__(self, collection_interval: float = 1.0):
        self.collection_interval = collection_interval
        self.running = False
        self.stats_lock = threading.Lock()
        
        # Thread statistics
        self.thread_stats: Dict[int, HashingStats] = {}
        
        # Historical data (last 300 seconds = 5 minutes)
        self.historical_hashrates: deque = deque(maxlen=300)
        self.historical_shares: deque = deque(maxlen=300)
        self.historical_system: deque = deque(maxlen=300)
        
        # Pool statistics
        self.pool_stats = PoolStats(pool_name="ZION 2.7 Mining Pool")
        
        # System monitoring
        self.system_stats = SystemStats()
        
        # Collection thread
        self.collection_thread = None
        
    def register_mining_thread(self, thread_id: int, difficulty: float = 1000.0) -> HashingStats:
        """Register new mining thread"""
        with self.stats_lock:
            stats = HashingStats(thread_id=thread_id, current_difficulty=difficulty)
            self.thread_stats[thread_id] = stats
            logger.info(f"Registered mining thread {thread_id} with difficulty {difficulty}")
            return stats
            
    def unregister_mining_thread(self, thread_id: int):
        """Unregister mining thread"""
        with self.stats_lock:
            if thread_id in self.thread_stats:
                del self.thread_stats[thread_id]
                logger.info(f"Unregistered mining thread {thread_id}")
                
    def update_thread_hashes(self, thread_id: int, hash_count: int = 1):
        """Update hash count for specific thread"""
        with self.stats_lock:
            if thread_id in self.thread_stats:
                self.thread_stats[thread_id].update_hash_count(hash_count)
                
    def add_share_result(self, thread_id: int, accepted: bool):
        """Add share result for specific thread"""
        with self.stats_lock:
            if thread_id in self.thread_stats:
                self.thread_stats[thread_id].add_share(accepted)
                
            # Update pool stats
            if accepted:
                self.pool_stats.total_shares_accepted += 1
            else:
                self.pool_stats.total_shares_rejected += 1
                
    def add_thread_error(self, thread_id: int):
        """Add error for specific thread"""
        with self.stats_lock:
            if thread_id in self.thread_stats:
                self.thread_stats[thread_id].add_error()
                
    def collect_system_stats(self):
        """Collect system resource statistics"""
        try:
            # CPU usage
            self.system_stats.cpu_usage_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.system_stats.memory_usage_mb = (memory.total - memory.available) / 1024 / 1024
            self.system_stats.memory_available_mb = memory.available / 1024 / 1024
            
            # Network I/O
            try:
                net_io = psutil.net_io_counters()
                if hasattr(self, '_last_net_rx'):
                    rx_delta = net_io.bytes_recv - self._last_net_rx
                    tx_delta = net_io.bytes_sent - self._last_net_tx
                    self.system_stats.network_rx_mbps = (rx_delta / self.collection_interval) / 1024 / 1024
                    self.system_stats.network_tx_mbps = (tx_delta / self.collection_interval) / 1024 / 1024
                    
                self._last_net_rx = net_io.bytes_recv
                self._last_net_tx = net_io.bytes_sent
            except:
                pass
                
            # Temperature (if available)
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    # Use CPU temperature if available
                    for name, entries in temps.items():
                        if 'cpu' in name.lower() or 'core' in name.lower():
                            if entries:
                                self.system_stats.temperature_celsius = entries[0].current
                                break
            except:
                pass
                
        except Exception as e:
            logger.error(f"Error collecting system stats: {e}")
            
    def collect_pool_stats(self):
        """Collect pool-wide statistics"""
        with self.stats_lock:
            # Calculate total hashrate
            current_time = time.time()
            total_hashrate = 0.0
            
            for stats in self.thread_stats.values():
                if current_time - stats.last_hash_time < 60:  # Active in last minute
                    total_hashrate += stats.avg_hashrate
                    
            self.pool_stats.total_hashrate = total_hashrate
            self.pool_stats.connected_miners = len(self.thread_stats)
            
    def start_collection(self):
        """Start statistics collection"""
        if self.running:
            return
            
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        logger.info("Mining statistics collection started")
        
    def stop_collection(self):
        """Stop statistics collection"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
        logger.info("Mining statistics collection stopped")
        
    def _collection_loop(self):
        """Main statistics collection loop"""
        start_time = time.time()
        
        while self.running:
            try:
                # Collect system statistics
                self.collect_system_stats()
                
                # Collect pool statistics
                self.collect_pool_stats()
                
                # Store historical data
                current_time = time.time()
                self.pool_stats.uptime_seconds = current_time - start_time
                
                with self.stats_lock:
                    self.historical_hashrates.append({
                        'timestamp': current_time,
                        'total_hashrate': self.pool_stats.total_hashrate,
                        'thread_hashrates': {
                            tid: stats.avg_hashrate for tid, stats in self.thread_stats.items()
                        }
                    })
                    
                    self.historical_shares.append({
                        'timestamp': current_time,
                        'accepted': self.pool_stats.total_shares_accepted,
                        'rejected': self.pool_stats.total_shares_rejected
                    })
                    
                    self.historical_system.append({
                        'timestamp': current_time,
                        'cpu_usage': self.system_stats.cpu_usage_percent,
                        'memory_usage': self.system_stats.memory_usage_mb,
                        'temperature': self.system_stats.temperature_celsius
                    })
                    
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in statistics collection loop: {e}")
                time.sleep(1.0)
                
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current comprehensive statistics"""
        with self.stats_lock:
            return {
                'pool_stats': asdict(self.pool_stats),
                'system_stats': asdict(self.system_stats),
                'thread_stats': {tid: asdict(stats) for tid, stats in self.thread_stats.items()},
                'summary': self._generate_summary()
            }
            
    def get_historical_data(self, duration_seconds: int = 300) -> Dict[str, Any]:
        """Get historical statistics data"""
        cutoff_time = time.time() - duration_seconds
        
        return {
            'hashrates': [
                entry for entry in self.historical_hashrates
                if entry['timestamp'] > cutoff_time
            ],
            'shares': [
                entry for entry in self.historical_shares
                if entry['timestamp'] > cutoff_time
            ],
            'system': [
                entry for entry in self.historical_system
                if entry['timestamp'] > cutoff_time
            ]
        }
        
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        total_hashes = sum(stats.total_hashes for stats in self.thread_stats.values())
        total_accepted = sum(stats.accepted_shares for stats in self.thread_stats.values())
        total_rejected = sum(stats.rejected_shares for stats in self.thread_stats.values())
        total_errors = sum(stats.errors for stats in self.thread_stats.values())
        
        # Calculate efficiency
        total_shares = total_accepted + total_rejected
        efficiency = (total_accepted / total_shares * 100) if total_shares > 0 else 0
        
        # Calculate average difficulty
        avg_difficulty = sum(stats.current_difficulty for stats in self.thread_stats.values())
        avg_difficulty = avg_difficulty / len(self.thread_stats) if self.thread_stats else 0
        
        return {
            'total_threads': len(self.thread_stats),
            'total_hashes': total_hashes,
            'total_shares': total_shares,
            'accepted_shares': total_accepted,
            'rejected_shares': total_rejected,
            'efficiency_percent': efficiency,
            'total_errors': total_errors,
            'average_difficulty': avg_difficulty,
            'uptime_hours': self.pool_stats.uptime_seconds / 3600
        }
        
    def export_stats(self, filename: str):
        """Export statistics to JSON file"""
        try:
            stats_data = {
                'export_time': time.time(),
                'current_stats': self.get_current_stats(),
                'historical_data': self.get_historical_data(3600)  # Last hour
            }
            
            with open(filename, 'w') as f:
                json.dump(stats_data, f, indent=2)
                
            logger.info(f"Statistics exported to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to export statistics: {e}")

class RealTimeMonitor:
    """Real-time mining monitor with console output"""
    
    def __init__(self, stats_collector: MiningStatsCollector):
        self.stats_collector = stats_collector
        self.running = False
        self.monitor_thread = None
        self.display_interval = 5.0
        
    def start_monitoring(self):
        """Start real-time monitoring"""
        if self.running:
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Real-time monitoring started")
        
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Real-time monitoring stopped")
        
    def _monitor_loop(self):
        """Main monitoring display loop"""
        while self.running:
            try:
                self._display_stats()
                time.sleep(self.display_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)
                
    def _display_stats(self):
        """Display current statistics"""
        stats = self.stats_collector.get_current_stats()
        
        print("\n" + "="*60)
        print("🚀 ZION 2.7 Mining Statistics")
        print("="*60)
        
        # Pool summary
        pool = stats['pool_stats']
        print(f"Pool: {pool['pool_name']}")
        print(f"Hashrate: {pool['total_hashrate']:.2f} H/s")
        print(f"Miners: {pool['connected_miners']}")
        print(f"Difficulty: {pool['difficulty']}")
        print(f"Uptime: {pool['uptime_seconds']/3600:.2f} hours")
        
        # Shares
        summary = stats['summary']
        print(f"Shares: {summary['accepted_shares']}/{summary['total_shares']} "
              f"({summary['efficiency_percent']:.1f}% efficiency)")
        
        # System resources
        system = stats['system_stats']
        print(f"CPU: {system['cpu_usage_percent']:.1f}% | "
              f"Memory: {system['memory_usage_mb']:.0f}MB | "
              f"Temp: {system['temperature_celsius']:.1f}°C")
        
        # Thread details
        print(f"\nThread Details:")
        for tid, thread_stats in stats['thread_stats'].items():
            print(f"  Thread {tid}: {thread_stats['avg_hashrate']:.2f} H/s "
                  f"({thread_stats['accepted_shares']}/{thread_stats['rejected_shares']})")

if __name__ == '__main__':
    # Test the statistics system
    import random
    
    print("🧪 Testing ZION 2.7 Mining Statistics System")
    
    # Create collector
    collector = MiningStatsCollector(collection_interval=1.0)
    collector.start_collection()
    
    # Register some test threads
    for i in range(4):
        collector.register_mining_thread(i, difficulty=1000.0 + i*500)
    
    # Start monitoring
    monitor = RealTimeMonitor(collector)
    monitor.start_monitoring()
    
    # Simulate mining activity
    try:
        for _ in range(30):  # 30 seconds of simulation
            # Simulate hash calculations
            for tid in range(4):
                hashes = random.randint(10, 50)
                collector.update_thread_hashes(tid, hashes)
                
                # Occasionally simulate shares
                if random.random() < 0.1:  # 10% chance
                    accepted = random.random() < 0.95  # 95% acceptance rate
                    collector.add_share_result(tid, accepted)
                    
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\nStopping simulation...")
        
    # Export final stats
    collector.export_stats("mining_stats_test.json")
    
    # Cleanup
    monitor.stop_monitoring()
    collector.stop_collection()
    
    print("✅ Statistics system test completed")