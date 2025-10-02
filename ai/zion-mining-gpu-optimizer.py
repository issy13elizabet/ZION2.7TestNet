#!/usr/bin/env python3
"""
ZION Mining-GPU Integration System
Intelligent GPU optimization based on mining performance and conditions
"""

import os
import json
import time
import subprocess
import requests
import threading
from datetime import datetime, timedelta
import psutil

class ZionMiningGPUOptimizer:
    def __init__(self):
        self.mining_api_url = "http://localhost:8080"  # ZION mining API
        self.gpu_api_url = "http://localhost:5001/api/gpu"  # GPU Afterburner API
        
        # Mining-specific optimization profiles
        self.mining_profiles = {
            "hashrate_focused": {
                "name": "Maximum Hashrate",
                "priority": "performance",
                "target_temp": 78,
                "power_efficiency": 0.3,  # 30% efficiency, 70% performance
                "mem_overclock": 150,     # +150MHz memory
                "core_overclock": 50,     # +50MHz core
                "description": "Pure hashrate optimization"
            },
            "efficiency_focused": {
                "name": "Power Efficient Mining",
                "priority": "efficiency", 
                "target_temp": 70,
                "power_efficiency": 0.8,  # 80% efficiency focused
                "mem_overclock": 100,     # +100MHz memory
                "core_overclock": 0,      # Stock core
                "description": "Best hashrate per watt"
            },
            "temperature_safe": {
                "name": "Cool & Stable",
                "priority": "stability",
                "target_temp": 65,
                "power_efficiency": 0.6,
                "mem_overclock": 50,      # +50MHz memory
                "core_overclock": -50,    # -50MHz core (underclocked)
                "description": "Maximum stability, low temps"
            },
            "adaptive": {
                "name": "AI Adaptive",
                "priority": "adaptive",
                "target_temp": 75,
                "power_efficiency": 0.5,
                "mem_overclock": "dynamic", # AI-determined
                "core_overclock": "dynamic",
                "description": "AI learns optimal settings"
            }
        }
        
        # Performance tracking
        self.performance_history = []
        self.optimization_session = {
            "start_time": datetime.now(),
            "best_hashrate": 0,
            "best_efficiency": 0,  # H/s per Watt
            "optimal_settings": {},
            "total_optimizations": 0
        }
        
        # AI learning parameters
        self.learning_data = {
            "settings_performance": [],  # [(settings, hashrate, power, efficiency)]
            "temperature_curves": [],
            "stability_scores": []
        }
        
        self.monitoring = True
        self.auto_optimize = False
        
    def get_mining_stats(self):
        """Get current mining statistics from ZION miner"""
        try:
            # First try direct file access to mining stats
            stats_file = "/tmp/zion_mining_stats.json"
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    return json.load(f)
            
            # Fallback: try API endpoint
            response = requests.get(f"{self.mining_api_url}/stats", timeout=5)
            if response.status_code == 200:
                return response.json()
                
        except Exception as e:
            print(f"Error getting mining stats: {e}")
        
        # Return mock data for testing
        return {
            "hashrate": 6012,
            "accepted_shares": 145,
            "rejected_shares": 2,
            "uptime": 3600,
            "pool_latency": 25,
            "difficulty": 150000
        }
    
    def get_gpu_stats(self):
        """Get current GPU statistics"""
        try:
            response = requests.get(f"{self.gpu_api_url}/stats", timeout=5)
            if response.status_code == 200:
                return response.json()["current"]
        except Exception as e:
            print(f"Error getting GPU stats: {e}")
        
        return {
            "temperature": 75,
            "power_usage": 110,
            "utilization": 98,
            "core_clock": 1500,
            "memory_clock": 1750
        }
    
    def calculate_efficiency(self, hashrate, power):
        """Calculate mining efficiency (H/s per Watt)"""
        if power > 0:
            return hashrate / power
        return 0
    
    def assess_stability(self, stats_window=10):
        """Assess mining stability over recent performance"""
        if len(self.performance_history) < stats_window:
            return 0.5  # Default stability
        
        recent_stats = self.performance_history[-stats_window:]
        
        # Calculate hashrate variance
        hashrates = [stat["hashrate"] for stat in recent_stats]
        avg_hashrate = sum(hashrates) / len(hashrates)
        
        if avg_hashrate == 0:
            return 0
        
        variance = sum((hr - avg_hashrate) ** 2 for hr in hashrates) / len(hashrates)
        stability = max(0, 1 - (variance / (avg_hashrate ** 2)))
        
        return stability
    
    def optimize_for_hashrate(self):
        """Optimize GPU settings for maximum hashrate"""
        current_gpu = self.get_gpu_stats()
        
        # Progressive memory overclock for mining
        memory_steps = [0, 50, 100, 150, 200, 250]  # MHz steps
        
        best_hashrate = 0
        best_memory = current_gpu.get("memory_clock", 1750)
        
        print("🚀 Hashrate optimization starting...")
        
        for mem_offset in memory_steps:
            print(f"  Testing memory +{mem_offset}MHz...")
            
            # Apply memory overclock (simplified - real implementation uses GPU API)
            success = self._apply_memory_clock(1750 + mem_offset)
            
            if success:
                time.sleep(10)  # Let mining stabilize
                
                mining_stats = self.get_mining_stats()
                hashrate = mining_stats.get("hashrate", 0)
                
                print(f"    Result: {hashrate} H/s")
                
                if hashrate > best_hashrate:
                    best_hashrate = hashrate
                    best_memory = 1750 + mem_offset
                    print(f"    ✅ New best: {hashrate} H/s @ {best_memory}MHz")
                else:
                    print(f"    📉 No improvement")
                    break  # Stop if performance degrades
        
        # Apply best settings
        self._apply_memory_clock(best_memory)
        
        return {
            "best_hashrate": best_hashrate,
            "optimal_memory": best_memory,
            "improvement": best_hashrate - mining_stats.get("hashrate", 0)
        }
    
    def optimize_for_efficiency(self):
        """Optimize for best hashrate per watt"""
        print("⚡ Efficiency optimization starting...")
        
        # Test different power limits
        power_limits = [70, 80, 90, 100, 110, 120]  # Percentage
        
        best_efficiency = 0
        best_settings = {}
        
        for power_pct in power_limits:
            print(f"  Testing {power_pct}% power limit...")
            
            # Apply power limit
            success = self._apply_power_limit(power_pct)
            
            if success:
                time.sleep(15)  # Let mining stabilize longer for power changes
                
                mining_stats = self.get_mining_stats()
                gpu_stats = self.get_gpu_stats()
                
                hashrate = mining_stats.get("hashrate", 0)
                power = gpu_stats.get("power_usage", 100)
                efficiency = self.calculate_efficiency(hashrate, power)
                
                print(f"    Result: {hashrate} H/s @ {power}W = {efficiency:.2f} H/W")
                
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_settings = {
                        "power_limit": power_pct,
                        "hashrate": hashrate,
                        "power": power
                    }
                    print(f"    ✅ New best efficiency: {efficiency:.2f} H/W")
        
        # Apply best settings
        if best_settings:
            self._apply_power_limit(best_settings["power_limit"])
        
        return best_settings
    
    def adaptive_ai_optimization(self):
        """AI-driven adaptive optimization"""
        print("🤖 AI adaptive optimization starting...")
        
        # Collect baseline performance
        baseline_mining = self.get_mining_stats()
        baseline_gpu = self.get_gpu_stats()
        baseline_efficiency = self.calculate_efficiency(
            baseline_mining.get("hashrate", 0),
            baseline_gpu.get("power_usage", 100)
        )
        
        print(f"   Baseline: {baseline_mining.get('hashrate', 0)} H/s, {baseline_efficiency:.2f} H/W")
        
        # Learning iterations
        test_settings = [
            {"mem_offset": 100, "power_limit": 85},
            {"mem_offset": 150, "power_limit": 90},
            {"mem_offset": 125, "power_limit": 95},
            {"mem_offset": 175, "power_limit": 100},
            {"mem_offset": 200, "power_limit": 105}
        ]
        
        best_performance = baseline_efficiency
        best_config = {"mem_offset": 0, "power_limit": 100}
        
        for i, settings in enumerate(test_settings):
            print(f"   AI Test {i+1}/5: Memory +{settings['mem_offset']}MHz, Power {settings['power_limit']}%")
            
            # Apply test settings
            self._apply_memory_clock(1750 + settings["mem_offset"])
            self._apply_power_limit(settings["power_limit"])
            
            time.sleep(20)  # Longer stabilization for AI learning
            
            # Measure performance
            mining_stats = self.get_mining_stats()
            gpu_stats = self.get_gpu_stats()
            
            hashrate = mining_stats.get("hashrate", 0)
            power = gpu_stats.get("power_usage", 100)
            temp = gpu_stats.get("temperature", 70)
            efficiency = self.calculate_efficiency(hashrate, power)
            
            # AI scoring function (considers multiple factors)
            stability = self.assess_stability()
            temp_penalty = max(0, (temp - 80) * 0.01)  # Penalty for temps > 80C
            
            ai_score = (efficiency * 0.6) + (hashrate * 0.0001) + (stability * 0.3) - temp_penalty
            
            print(f"      Results: {hashrate} H/s @ {power}W ({temp}°C)")
            print(f"      AI Score: {ai_score:.4f} (eff:{efficiency:.2f}, stab:{stability:.2f})")
            
            # Store learning data
            self.learning_data["settings_performance"].append({
                "settings": settings.copy(),
                "hashrate": hashrate,
                "power": power,
                "temperature": temp,
                "efficiency": efficiency,
                "stability": stability,
                "ai_score": ai_score
            })
            
            if ai_score > best_performance * 0.8:  # Allow for AI score vs pure efficiency
                best_performance = ai_score
                best_config = settings.copy()
                print(f"      ✅ New AI best config")
        
        # Apply best AI-determined settings
        print(f"🎯 AI optimal settings: {best_config}")
        self._apply_memory_clock(1750 + best_config["mem_offset"])
        self._apply_power_limit(best_config["power_limit"])
        
        return {
            "best_config": best_config,
            "ai_score": best_performance,
            "learning_iterations": len(test_settings)
        }
    
    def auto_optimization_loop(self):
        """Continuous automatic optimization based on conditions"""
        print("🔄 Auto-optimization loop started")
        
        optimization_interval = 300  # 5 minutes between optimizations
        last_optimization = datetime.now()
        
        while self.monitoring and self.auto_optimize:
            try:
                current_time = datetime.now()
                
                # Check if it's time for optimization
                if (current_time - last_optimization).seconds < optimization_interval:
                    time.sleep(30)
                    continue
                
                print(f"\n🤖 [{current_time.strftime('%H:%M:%S')}] Auto-optimization cycle")
                
                # Get current performance
                mining_stats = self.get_mining_stats()
                gpu_stats = self.get_gpu_stats()
                
                current_hashrate = mining_stats.get("hashrate", 0)
                current_temp = gpu_stats.get("temperature", 70)
                current_power = gpu_stats.get("power_usage", 100)
                
                # Determine optimization strategy
                if current_temp > 85:
                    print("🌡️ High temperature detected - optimizing for cooling")
                    self._apply_power_limit(70)  # Reduce power for cooling
                    
                elif current_hashrate < self.optimization_session["best_hashrate"] * 0.95:
                    print("📉 Hashrate drop detected - re-optimizing")
                    result = self.optimize_for_hashrate()
                    self.optimization_session["total_optimizations"] += 1
                    
                elif self.optimization_session["total_optimizations"] % 3 == 0:
                    print("🎯 Periodic AI optimization")
                    result = self.adaptive_ai_optimization()
                    self.optimization_session["total_optimizations"] += 1
                
                # Update performance tracking
                efficiency = self.calculate_efficiency(current_hashrate, current_power)
                
                self.performance_history.append({
                    "timestamp": current_time.isoformat(),
                    "hashrate": current_hashrate,
                    "power": current_power,
                    "temperature": current_temp,
                    "efficiency": efficiency
                })
                
                # Keep only recent history
                if len(self.performance_history) > 288:  # 24 hours of 5-min intervals
                    self.performance_history.pop(0)
                
                # Update session bests
                if current_hashrate > self.optimization_session["best_hashrate"]:
                    self.optimization_session["best_hashrate"] = current_hashrate
                    
                if efficiency > self.optimization_session["best_efficiency"]:
                    self.optimization_session["best_efficiency"] = efficiency
                
                last_optimization = current_time
                
            except Exception as e:
                print(f"Auto-optimization error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _apply_memory_clock(self, freq_mhz):
        """Apply memory clock setting (mock implementation)"""
        try:
            # In real implementation, this would call GPU API
            print(f"    Applying memory clock: {freq_mhz}MHz")
            # Simulate API call delay
            time.sleep(1)
            return True
        except:
            return False
    
    def _apply_power_limit(self, percent):
        """Apply power limit setting (mock implementation)"""
        try:
            print(f"    Applying power limit: {percent}%")
            time.sleep(1)
            return True
        except:
            return False
    
    def get_optimization_report(self):
        """Generate comprehensive optimization report"""
        current_time = datetime.now()
        session_duration = current_time - self.optimization_session["start_time"]
        
        recent_avg_hashrate = 0
        recent_avg_efficiency = 0
        
        if self.performance_history:
            recent_data = self.performance_history[-12:]  # Last hour
            recent_avg_hashrate = sum(d["hashrate"] for d in recent_data) / len(recent_data)
            recent_avg_efficiency = sum(d["efficiency"] for d in recent_data) / len(recent_data)
        
        return {
            "session_info": {
                "duration_hours": session_duration.total_seconds() / 3600,
                "total_optimizations": self.optimization_session["total_optimizations"],
                "best_hashrate": self.optimization_session["best_hashrate"],
                "best_efficiency": self.optimization_session["best_efficiency"]
            },
            "current_performance": {
                "avg_hashrate_1h": recent_avg_hashrate,
                "avg_efficiency_1h": recent_avg_efficiency,
                "stability_score": self.assess_stability()
            },
            "learning_data": {
                "total_tests": len(self.learning_data["settings_performance"]),
                "best_ai_config": max(self.learning_data["settings_performance"], 
                                    key=lambda x: x.get("ai_score", 0), 
                                    default={}) if self.learning_data["settings_performance"] else {}
            }
        }
    
    def start_auto_optimization(self):
        """Start automatic optimization"""
        self.auto_optimize = True
        auto_thread = threading.Thread(target=self.auto_optimization_loop, daemon=True)
        auto_thread.start()
        return "Auto-optimization started"
    
    def stop_auto_optimization(self):
        """Stop automatic optimization"""
        self.auto_optimize = False
        return "Auto-optimization stopped"

def main():
    """Test mining-GPU optimization system"""
    print("⛏️ ZION Mining-GPU Optimizer Test")
    
    optimizer = ZionMiningGPUOptimizer()
    
    while True:
        print("\n" + "="*50)
        print("🎮 ZION Mining-GPU Optimizer Menu")
        print("="*50)
        print("1. 🚀 Optimize for Maximum Hashrate")
        print("2. ⚡ Optimize for Power Efficiency") 
        print("3. 🤖 AI Adaptive Optimization")
        print("4. 🔄 Start Auto-Optimization")
        print("5. ⏹️  Stop Auto-Optimization")
        print("6. 📊 View Optimization Report")
        print("7. 🚪 Exit")
        
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == "1":
            result = optimizer.optimize_for_hashrate()
            print(f"\n✅ Hashrate optimization complete:")
            print(f"   Best hashrate: {result['best_hashrate']} H/s")
            print(f"   Optimal memory: {result['optimal_memory']} MHz")
            
        elif choice == "2":
            result = optimizer.optimize_for_efficiency()
            print(f"\n✅ Efficiency optimization complete:")
            if result:
                print(f"   Best settings: {result}")
            
        elif choice == "3":
            result = optimizer.adaptive_ai_optimization()
            print(f"\n✅ AI optimization complete:")
            print(f"   Best config: {result['best_config']}")
            print(f"   AI score: {result['ai_score']:.4f}")
            
        elif choice == "4":
            msg = optimizer.start_auto_optimization()
            print(f"\n🔄 {msg}")
            
        elif choice == "5":
            msg = optimizer.stop_auto_optimization()
            print(f"\n⏹️ {msg}")
            
        elif choice == "6":
            report = optimizer.get_optimization_report()
            print(f"\n📊 Optimization Report:")
            print(f"   Session duration: {report['session_info']['duration_hours']:.1f} hours")
            print(f"   Total optimizations: {report['session_info']['total_optimizations']}")
            print(f"   Best hashrate: {report['session_info']['best_hashrate']} H/s")
            print(f"   Best efficiency: {report['session_info']['best_efficiency']:.2f} H/W")
            print(f"   Current stability: {report['current_performance']['stability_score']:.2f}")
            
        elif choice == "7":
            optimizer.stop_auto_optimization()
            print("\n👋 Goodbye!")
            break
            
        else:
            print("\n❌ Invalid option")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()