#!/usr/bin/env python3
"""
ZION Ultra Performance Mining CLI - XMRig Killer Edition
ZION 2.6.75 Sacred Technology Stack

Advanced mining benchmark and optimization tool designed to outperform XMRig.
"""
import sys
import time
import json
import argparse
import psutil
from pathlib import Path

# Add ZION mining to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from zion_ultra_performance import ZionUltraPerformanceEngine
except ImportError:
    # Fallback to direct import
    import importlib.util
    spec = importlib.util.spec_from_file_location("zion_ultra_performance", 
                                                 Path(__file__).parent / "zion-ultra-performance.py")
    zion_ultra_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(zion_ultra_module)
    ZionUltraPerformanceEngine = zion_ultra_module.ZionUltraPerformanceEngine
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ZionMiningCLI:
    """Command-line interface for ZION Ultra Performance Mining"""
    
    def __init__(self):
        self.engine = None
        
    def display_system_info(self):
        """Display system information for optimization"""
        print("🚀 ZION Ultra Performance Mining System Analysis")
        print("=" * 60)
        
        # CPU Information
        cpu_info = {
            'cores': psutil.cpu_count(logical=False),
            'threads': psutil.cpu_count(logical=True),
            'freq': psutil.cpu_freq()
        }
        
        print(f"🖥️  CPU Information:")
        print(f"   Physical Cores: {cpu_info['cores']}")
        print(f"   Logical Threads: {cpu_info['threads']}")
        if cpu_info['freq']:
            print(f"   Base Frequency: {cpu_info['freq'].current:.0f} MHz")
            print(f"   Max Frequency: {cpu_info['freq'].max:.0f} MHz")
        
        # Memory Information
        mem = psutil.virtual_memory()
        print(f"💾 Memory Information:")
        print(f"   Total RAM: {mem.total / (1024**3):.1f} GB")
        print(f"   Available: {mem.available / (1024**3):.1f} GB")
        print(f"   Usage: {mem.percent:.1f}%")
        
        # Check for large pages
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
            if 'HugePages_Total' in meminfo:
                print(f"   Large Pages: ✅ Available")
            else:
                print(f"   Large Pages: ❌ Not configured")
        except:
            print(f"   Large Pages: ❓ Unknown")
        
        print()
    
    def run_quick_benchmark(self):
        """Run quick mining benchmark"""
        print("⚡ ZION Quick Mining Benchmark")
        print("=" * 40)
        
        self.engine = ZionUltraPerformanceEngine()
        
        # Initialize engines
        seed = b"ZION_QUICK_BENCHMARK_SEED"
        if not self.engine.init_mining_engines(seed):
            print("❌ Failed to initialize mining engines")
            return False
        
        print(f"🎯 Testing with {self.engine.threads} threads...")
        print("⏱️  Running 5-second performance test...")
        
        # Run benchmark
        import hashlib
        target = hashlib.sha256(b"ZION_QUICK_TEST").digest()
        
        start_time = time.time()
        result = self.engine.start_mining(target, difficulty=50000)
        end_time = time.time()
        
        if result['success']:
            print(f"✅ Benchmark completed!")
            print(f"   Hashrate: {result['hashrate']:.2f} H/s")
            print(f"   Total Hashes: {result['total_hashes']:,}")
            print(f"   Test Duration: {end_time - start_time:.2f} seconds")
            print(f"   Threads Used: {result['threads_used']}")
            
            # Calculate performance metrics
            hashes_per_thread = result['hashrate'] / result['threads_used']
            print(f"   Per-Thread Rate: {hashes_per_thread:.2f} H/s")
            
            return True
        else:
            print(f"❌ Benchmark failed: {result.get('error', 'Unknown error')}")
            return False
    
    def run_full_benchmark(self):
        """Run comprehensive benchmark vs XMRig"""
        print("🏆 ZION vs XMRig Full Benchmark Suite")
        print("=" * 50)
        
        self.engine = ZionUltraPerformanceEngine()
        
        print("🔧 Auto-tuning optimal thread configuration...")
        optimal_threads = self.engine.optimize_thread_count(test_duration=8.0)
        
        print(f"✅ Optimal threads determined: {optimal_threads}")
        print()
        
        print("🚀 Running comprehensive benchmark...")
        benchmark_result = self.engine.benchmark_vs_xmrig()
        
        if 'error' in benchmark_result:
            print(f"❌ Benchmark failed: {benchmark_result['error']}")
            return False
        
        # Display results
        print("🎯 BENCHMARK RESULTS:")
        print("=" * 30)
        
        for name, result in benchmark_result['benchmark_results'].items():
            efficiency_indicator = "🔥" if result['efficiency'] > 200 else "⚡" if result['efficiency'] > 100 else "💪"
            print(f"{efficiency_indicator} {name}:")
            print(f"   Threads: {result['threads']}")
            print(f"   Hashrate: {result['hashrate']:.2f} H/s")
            print(f"   Efficiency: {result['efficiency']:.2f} H/s per thread")
            print()
        
        best_config = benchmark_result['best_config']
        best_rate = benchmark_result['best_hashrate']
        
        print(f"🏆 WINNER: {best_config}")
        print(f"🚀 Best Performance: {best_rate:.2f} H/s")
        print(f"📊 Optimal Threads: {benchmark_result['optimal_threads']}")
        
        # XMRig comparison estimate
        print("\n🥊 XMRig Comparison Estimate:")
        print("=" * 35)
        
        # Rough XMRig baseline (varies by hardware)
        cpu_cores = psutil.cpu_count(logical=False)
        estimated_xmrig_rate = cpu_cores * 150  # Conservative XMRig estimate
        
        if best_rate > estimated_xmrig_rate:
            advantage = ((best_rate / estimated_xmrig_rate) - 1) * 100
            print(f"🔥 ZION WINS! {advantage:.1f}% faster than estimated XMRig")
        else:
            disadvantage = ((estimated_xmrig_rate / best_rate) - 1) * 100
            print(f"📈 XMRig estimated {disadvantage:.1f}% faster - optimizing...")
        
        print(f"   ZION Rate: {best_rate:.2f} H/s")
        print(f"   XMRig Est: {estimated_xmrig_rate:.2f} H/s")
        
        return True
    
    def run_stress_test(self, duration: int = 60):
        """Run extended stress test for stability"""
        print(f"🔥 ZION Stress Test - {duration} seconds")
        print("=" * 40)
        
        self.engine = ZionUltraPerformanceEngine()
        
        # Initialize with optimal settings
        seed = b"ZION_STRESS_TEST_SEED"
        if not self.engine.init_mining_engines(seed):
            print("❌ Failed to initialize mining engines")
            return False
        
        print(f"🎯 Running {duration}s stress test with {self.engine.threads} threads...")
        
        # Monitor system during test
        import hashlib
        target = hashlib.sha256(b"ZION_STRESS_TEST").digest()
        
        start_time = time.time()
        peak_hashrate = 0
        temperature_readings = []
        
        # Temporary mining loop with monitoring
        self.engine.mining_active = True
        
        try:
            while time.time() - start_time < duration:
                # Run mini benchmark
                result = self.engine.start_mining(target, difficulty=10000)
                
                if result['success']:
                    current_rate = result['hashrate']
                    peak_hashrate = max(peak_hashrate, current_rate)
                    
                    # Try to get CPU temperature
                    try:
                        temps = psutil.sensors_temperatures()
                        if 'coretemp' in temps:
                            cpu_temp = max([t.current for t in temps['coretemp']])
                            temperature_readings.append(cpu_temp)
                        elif 'k10temp' in temps:  # AMD
                            cpu_temp = max([t.current for t in temps['k10temp']])
                            temperature_readings.append(cpu_temp)
                    except:
                        pass
                    
                    elapsed = time.time() - start_time
                    progress = (elapsed / duration) * 100
                    
                    print(f"⏱️  {elapsed:.0f}s ({progress:.0f}%) - "
                          f"Rate: {current_rate:.2f} H/s - "
                          f"Peak: {peak_hashrate:.2f} H/s", end='\r')
                
                time.sleep(1)  # Brief pause between tests
                
        except KeyboardInterrupt:
            print("\n⚠️  Stress test interrupted by user")
        
        self.engine.mining_active = False
        total_time = time.time() - start_time
        
        print(f"\n✅ Stress Test Completed!")
        print(f"   Duration: {total_time:.1f} seconds")
        print(f"   Peak Hashrate: {peak_hashrate:.2f} H/s")
        
        if temperature_readings:
            avg_temp = sum(temperature_readings) / len(temperature_readings)
            max_temp = max(temperature_readings)
            print(f"   Average CPU Temp: {avg_temp:.1f}°C")
            print(f"   Max CPU Temp: {max_temp:.1f}°C")
            
            if max_temp > 85:
                print("🔥 WARNING: High temperatures detected!")
            elif max_temp > 75:
                print("⚠️  CAUTION: Elevated temperatures")
            else:
                print("❄️  Temperature: Optimal")
        
        return True
    
    def run_interactive_mode(self):
        """Run interactive mining mode"""
        print("🎮 ZION Interactive Mining Mode")
        print("=" * 35)
        print("Commands:")
        print("  'start' - Start mining")
        print("  'stop'  - Stop mining") 
        print("  'stats' - Show statistics")
        print("  'bench' - Run benchmark")
        print("  'quit'  - Exit")
        print()
        
        self.engine = ZionUltraPerformanceEngine()
        mining_active = False
        
        while True:
            try:
                cmd = input("ZION> ").strip().lower()
                
                if cmd == 'quit' or cmd == 'exit':
                    break
                elif cmd == 'start':
                    if not mining_active:
                        print("🚀 Starting mining...")
                        # Initialize engines if needed
                        seed = b"ZION_INTERACTIVE_MINING"
                        if self.engine.init_mining_engines(seed):
                            mining_active = True
                            print("✅ Mining started!")
                        else:
                            print("❌ Failed to start mining")
                    else:
                        print("⚠️  Mining already active")
                        
                elif cmd == 'stop':
                    if mining_active:
                        self.engine.mining_active = False
                        mining_active = False
                        print("🛑 Mining stopped")
                    else:
                        print("ℹ️  Mining not active")
                        
                elif cmd == 'stats':
                    if hasattr(self.engine, 'stats'):
                        stats = self.engine.stats
                        print(f"📊 Current Stats:")
                        print(f"   Hashrate: {stats.hashrate:.2f} H/s")
                        print(f"   Total Hashes: {stats.total_hashes:,}")
                        print(f"   Uptime: {time.time() - self.engine.start_time:.1f}s")
                    else:
                        print("ℹ️  No stats available")
                        
                elif cmd == 'bench':
                    print("🏃 Running quick benchmark...")
                    self.run_quick_benchmark()
                    
                elif cmd == 'help':
                    print("Available commands: start, stop, stats, bench, quit")
                    
                else:
                    print(f"❓ Unknown command: {cmd}")
                    
            except KeyboardInterrupt:
                print("\n👋 Exiting...")
                break
            except EOFError:
                break
        
        if mining_active:
            self.engine.mining_active = False
        
        print("🔒 ZION Interactive Mode closed")


def main():
    parser = argparse.ArgumentParser(description='ZION Ultra Performance Mining CLI')
    parser.add_argument('--mode', choices=['info', 'quick', 'benchmark', 'stress', 'interactive'], 
                       default='benchmark', help='Mining mode to run')
    parser.add_argument('--duration', type=int, default=60, 
                       help='Stress test duration in seconds')
    parser.add_argument('--threads', type=int, help='Number of mining threads')
    
    args = parser.parse_args()
    
    print("⚡ ZION Ultra Performance Mining v2.6.75")
    print("🎯 Sacred Technology Stack - XMRig Killer Edition")
    print()
    
    cli = ZionMiningCLI()
    
    # Override thread count if specified
    if args.threads:
        cli.engine = ZionUltraPerformanceEngine(target_threads=args.threads)
    
    try:
        if args.mode == 'info':
            cli.display_system_info()
        elif args.mode == 'quick':
            cli.display_system_info()
            cli.run_quick_benchmark()
        elif args.mode == 'benchmark':
            cli.display_system_info()
            cli.run_full_benchmark()
        elif args.mode == 'stress':
            cli.display_system_info()
            cli.run_stress_test(args.duration)
        elif args.mode == 'interactive':
            cli.display_system_info()
            cli.run_interactive_mode()
            
        print("\n✨ ZION Ultra Performance Mining completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("CLI error")


if __name__ == "__main__":
    main()