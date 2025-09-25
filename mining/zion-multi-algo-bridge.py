#!/usr/bin/env python3
"""
ZION Multi-Algo GPU Mining Bridge v2.5.0
Intelligent algorithm switching based on profitability and network difficulty
"""

import json
import time
import subprocess
import threading
import requests
import os
import signal
from datetime import datetime
from pathlib import Path

class ZionMultiAlgoMiner:
    def __init__(self):
        self.config_file = "multi-algo-pools.json"
        self.current_process = None
        self.current_algo = None
        self.running = True
        self.stats = {
            'session_start': datetime.now(),
            'algo_switches': 0,
            'total_runtime': 0
        }
        
        # Load configuration
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)
            
        self.algorithms = self.config['multi-algo-config']['algorithms']
        self.pools = self.config['pools']
        
        print("🚀 ZION Multi-Algo GPU Mining Bridge v2.5.0")
        print("=" * 60)
        
    def get_network_stats(self, algo):
        """Fetch network difficulty and price data for profitability calculation"""
        try:
            if algo == "randomx":
                # Simulate RandomX network stats
                return {"difficulty": 350000000000, "price_usd": 150.0, "block_reward": 0.6}
            elif algo == "kawpow": 
                return {"difficulty": 85000, "price_usd": 0.025, "block_reward": 2500}
            elif algo == "octopus":
                # Conflux (CFX) network stats 
                return {"difficulty": 450000000000000, "price_usd": 0.12, "block_reward": 7.3}
            elif algo == "ergo":
                # Ergo (ERG) network stats
                return {"difficulty": 1200000000000000, "price_usd": 1.25, "block_reward": 51.0}
            elif algo == "ethash":
                return {"difficulty": 18000000000000000, "price_usd": 18.50, "block_reward": 2.0}
            elif algo == "cryptonight":
                return {"difficulty": 250000000, "price_usd": 1.0, "block_reward": 10}
        except Exception as e:
            print(f"⚠️  Could not fetch network stats for {algo}: {e}")
            return {"difficulty": 1, "price_usd": 1.0, "block_reward": 1}
    
    def calculate_profitability(self, algo):
        """Calculate expected profitability for algorithm"""
        algo_config = self.algorithms[algo]
        network_stats = self.get_network_stats(algo)
        
        # Parse hashrate string (e.g., "1200 H/s" -> 1200)
        hashrate_str = algo_config['expected_hashrate']
        hashrate = float(hashrate_str.split()[0])
        
        # Convert units
        if "MH/s" in hashrate_str:
            hashrate *= 1000000
        elif "kH/s" in hashrate_str:
            hashrate *= 1000
            
        # Simple profitability calculation
        daily_blocks = (hashrate * 86400) / network_stats['difficulty']
        daily_coins = daily_blocks * network_stats['block_reward'] 
        daily_usd = daily_coins * network_stats['price_usd']
        
        return {
            'daily_usd': daily_usd,
            'hashrate': hashrate,
            'algo': algo
        }
    
    def select_best_algorithm(self):
        """Select most profitable algorithm"""
        profitabilities = []
        
        for algo in self.algorithms:
            if self.algorithms[algo]['enabled']:
                profit = self.calculate_profitability(algo)
                profit['priority'] = self.algorithms[algo]['priority']
                profitabilities.append(profit)
        
        # Sort by daily USD profit, then by priority
        profitabilities.sort(key=lambda x: (-x['daily_usd'], x['priority']))
        
        if profitabilities:
            best = profitabilities[0]
            print(f"📊 Profitability Analysis:")
            for p in profitabilities[:3]:  # Show top 3
                print(f"   {p['algo']:12} ${p['daily_usd']:.4f}/day")
            return best['algo']
        
        return "randomx"  # Fallback
    
    def start_mining(self, algo):
        """Start mining with specified algorithm using SRBMiner-Multi GPU"""
        if self.current_process:
            self.stop_current_mining()
            
        pool = next(p for p in self.pools if p['name'].lower().endswith(algo.lower()))
        algo_config = self.algorithms[algo]
        
        print(f"🎮 Starting {algo.upper()} mining with SRBMiner-Multi...")
        print(f"   Pool: {pool['url']}")
        print(f"   Expected: {algo_config['expected_hashrate']}")
        
        # Build SRBMiner-Multi command with algorithm-specific config
        srbminer_path = "D:\\Zion TestNet\\Zion\\mining\\SRBMiner-Multi-2-9-7\\SRBMiner-MULTI.exe"
        
        # Use algorithm-specific configuration files
        config_file = None
        if algo == "kawpow":
            config_file = "D:\\Zion TestNet\\Zion\\mining\\srb-kawpow-config.json"
        elif algo == "octopus":
            config_file = "D:\\Zion TestNet\\Zion\\mining\\srb-octopus-config.json"  
        elif algo == "ergo":
            config_file = "D:\\Zion TestNet\\Zion\\mining\\srb-ergo-config.json"
        
        if config_file and os.path.exists(config_file):
            # Use config file for optimized settings
            cmd = [srbminer_path, "--config", config_file]
        else:
            # Fallback to manual command line parameters
            cmd = [
                srbminer_path,
                "--gpu-id", "0",
                "--pool", pool['url'],
                "--wallet", pool['user'], 
                "--password", pool['pass'],
                "--gpu-boost", "100",
                "--gpu-threads", "18",
                "--gpu-worksize", "256",
                "--disable-cpu",
                "--log-file", f"srbminer-{algo}.log"
            ]
            
            # Add algorithm-specific settings for SRBMiner-Multi
            if algo == "randomx":
                cmd.extend(["--algorithm", "randomx"])
            elif algo == "kawpow":
                cmd.extend(["--algorithm", "kawpow"])
            elif algo == "octopus":
                cmd.extend(["--algorithm", "octopus"])
            elif algo == "ergo":
                cmd.extend(["--algorithm", "autolykos2"])
            elif algo == "ethash":
                cmd.extend(["--algorithm", "ethash"])
            elif algo == "cryptonight":
                cmd.extend(["--algorithm", "cryptonightv2"])
            
        try:
            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            self.current_algo = algo
            print(f"✅ {algo.upper()} mining started (PID: {self.current_process.pid})")
            
        except Exception as e:
            print(f"❌ Failed to start {algo} mining: {e}")
    
    def stop_current_mining(self):
        """Stop current mining process"""
        if self.current_process:
            print(f"⏹️  Stopping {self.current_algo} mining...")
            try:
                self.current_process.terminate()
                self.current_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.current_process.kill()
            self.current_process = None
            self.current_algo = None
    
    def monitor_and_switch(self):
        """Main monitoring loop with intelligent switching"""
        switch_interval = self.config['multi-algo-config']['switching-interval']
        
        while self.running:
            try:
                # Select best algorithm
                best_algo = self.select_best_algorithm()
                
                # Switch if needed
                if best_algo != self.current_algo:
                    print(f"🔄 Switching from {self.current_algo} to {best_algo}")
                    self.start_mining(best_algo)
                    self.stats['algo_switches'] += 1
                
                # Monitor current process
                if self.current_process and self.current_process.poll() is not None:
                    print(f"⚠️  Mining process died, restarting {self.current_algo}...")
                    self.start_mining(self.current_algo)
                
                # Print status
                uptime = int((datetime.now() - self.stats['session_start']).total_seconds())
                print(f"📈 Status: {self.current_algo.upper()} mining | "
                      f"Uptime: {uptime//3600}h {(uptime%3600)//60}m | "
                      f"Switches: {self.stats['algo_switches']}")
                
                # Wait before next check
                time.sleep(switch_interval)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error in monitoring loop: {e}")
                time.sleep(30)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\n🛑 Shutting down ZION Multi-Algo Miner...")
        self.running = False
        self.stop_current_mining()
    
    def run(self):
        """Start the multi-algo mining system"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        print("🚀 Starting intelligent multi-algo mining...")
        
        # Start with most profitable algorithm
        best_algo = self.select_best_algorithm()
        self.start_mining(best_algo)
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self.monitor_and_switch)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        print("\n" + "="*60)
        print("🎯 ZION Multi-Algo Mining Bridge Active!")
        print("   Press Ctrl+C to stop")
        print("="*60)
        
        try:
            monitor_thread.join()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_current_mining()
            print("✅ ZION Multi-Algo Miner stopped.")

if __name__ == "__main__":
    miner = ZionMultiAlgoMiner()
    miner.run()