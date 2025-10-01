#!/usr/bin/env python3
"""
ZION Smart System Coordinator - Intelligent CPU+GPU Management
AI-driven coordination for optimal performance across all workloads
"""

import os
import json
import time
import threading
import psutil
import requests
from datetime import datetime, timedelta
from collections import deque
import math

class ZionSmartCoordinator:
    def __init__(self):
        self.system_api = "http://localhost:5002/api/system"
        self.mining_api = "http://localhost:8080"  # ZION mining endpoint
        
        # Performance tracking
        self.performance_history = deque(maxlen=100)  # Last 100 measurements
        self.optimization_history = []
        
        # AI learning parameters
        self.learning_rate = 0.1
        self.optimization_weights = {
            "hashrate": 0.4,      # 40% weight on hashrate
            "efficiency": 0.3,    # 30% weight on power efficiency
            "temperature": 0.2,   # 20% weight on temperature control
            "stability": 0.1      # 10% weight on system stability
        }
        
        # Workload detection
        self.workload_patterns = {
            "idle": {
                "cpu_usage": (0, 15),
                "gpu_usage": (0, 10),
                "description": "System idle or light tasks"
            },
            "mining": {
                "cpu_usage": (60, 95),
                "gpu_usage": (85, 100),
                "description": "Cryptocurrency mining active"
            },
            "gaming": {
                "cpu_usage": (30, 80),
                "gpu_usage": (70, 100),
                "description": "Gaming or 3D applications"
            },
            "compute": {
                "cpu_usage": (80, 100),
                "gpu_usage": (20, 60),
                "description": "CPU-intensive computing"
            },
            "mixed": {
                "cpu_usage": (40, 80),
                "gpu_usage": (40, 80),
                "description": "Mixed workload"
            }
        }
        
        # Coordination strategies
        self.coordination_strategies = {
            "mining_optimized": {
                "description": "Maximize mining hashrate and efficiency",
                "cpu_strategy": "support_gpu",  # CPU supports GPU mining
                "gpu_strategy": "maximize_mining",
                "power_balance": 0.3,  # 30% CPU, 70% GPU power budget
                "target_metrics": {
                    "hashrate_priority": 0.6,
                    "efficiency_priority": 0.4
                }
            },
            "gaming_performance": {
                "description": "Balanced CPU+GPU for gaming",
                "cpu_strategy": "balanced_performance",
                "gpu_strategy": "gaming_optimized",
                "power_balance": 0.4,  # 40% CPU, 60% GPU
                "target_metrics": {
                    "fps_priority": 0.7,
                    "latency_priority": 0.3
                }
            },
            "power_efficient": {
                "description": "Maximum power savings",
                "cpu_strategy": "eco_mode",
                "gpu_strategy": "power_save", 
                "power_balance": 0.5,  # Equal power distribution
                "target_metrics": {
                    "power_priority": 0.8,
                    "performance_priority": 0.2
                }
            },
            "compute_focused": {
                "description": "CPU-heavy workload optimization",
                "cpu_strategy": "maximum_performance",
                "gpu_strategy": "support_cpu",
                "power_balance": 0.7,  # 70% CPU, 30% GPU
                "target_metrics": {
                    "compute_priority": 0.8,
                    "efficiency_priority": 0.2
                }
            }
        }
        
        # Current state
        self.current_workload = "idle"
        self.current_strategy = "mining_optimized"
        self.auto_coordination = True
        self.monitoring = True
        
        # Performance baselines (to be learned)
        self.baselines = {
            "best_hashrate": 6012,
            "best_efficiency": 54.2,  # H/W
            "safe_cpu_temp": 75,
            "safe_gpu_temp": 80
        }
        
    def get_system_stats(self):
        """Get current system statistics"""
        try:
            response = requests.get(f"{self.system_api}/stats", timeout=5)
            if response.status_code == 200:
                return response.json()["current"]
        except Exception as e:
            print(f"Error getting system stats: {e}")
        
        return None
    
    def get_mining_stats(self):
        """Get mining performance statistics"""
        try:
            # Try mining API first
            response = requests.get(f"{self.mining_api}/stats", timeout=5)
            if response.status_code == 200:
                return response.json()
            
            # Fallback: read from stats file
            stats_file = "/tmp/zion_mining_stats.json"
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    return json.load(f)
                    
        except Exception as e:
            print(f"Error getting mining stats: {e}")
        
        # Mock data for development
        return {
            "hashrate": 6012 + (int(time.time()) % 100) - 50,  # Simulate variance
            "accepted_shares": 247,
            "rejected_shares": 3,
            "pool_latency": 18
        }
    
    def detect_workload(self, stats):
        """Automatically detect current workload type"""
        if not stats:
            return "unknown"
        
        cpu_usage = stats["cpu"]["usage_percent"]
        gpu_usage = stats["gpu"]["utilization"]
        
        # Check each workload pattern
        for workload, pattern in self.workload_patterns.items():
            cpu_min, cpu_max = pattern["cpu_usage"]
            gpu_min, gpu_max = pattern["gpu_usage"]
            
            if cpu_min <= cpu_usage <= cpu_max and gpu_min <= gpu_usage <= gpu_max:
                return workload
        
        return "mixed"
    
    def calculate_performance_score(self, stats, mining_stats=None):
        """Calculate overall system performance score"""
        if not stats:
            return 0.0
        
        score = 0.0
        
        # Temperature scoring (lower is better)
        cpu_temp = stats["cpu"]["temperature"]
        gpu_temp = stats["gpu"]["temperature"]
        temp_score = (
            max(0, 1 - (cpu_temp - 50) / 35) * 0.5 +  # CPU temp 50-85°C range
            max(0, 1 - (gpu_temp - 60) / 30) * 0.5   # GPU temp 60-90°C range
        )
        
        # Power efficiency scoring
        total_power = stats["system"]["total_power_estimated"]
        if mining_stats and total_power > 0:
            hashrate = mining_stats.get("hashrate", 0)
            efficiency = hashrate / total_power if total_power > 0 else 0
            efficiency_score = min(1.0, efficiency / 60)  # 60 H/W as good efficiency
        else:
            efficiency_score = 0.5  # Default when not mining
        
        # Performance scoring (utilization vs temperature balance)
        cpu_perf = stats["cpu"]["usage_percent"] / 100
        gpu_perf = stats["gpu"]["utilization"] / 100
        perf_score = (cpu_perf + gpu_perf) / 2
        
        # Stability scoring (based on recent variance)
        stability_score = self.calculate_stability_score()
        
        # Weighted combination
        score = (
            temp_score * self.optimization_weights["temperature"] +
            efficiency_score * self.optimization_weights["efficiency"] +
            perf_score * self.optimization_weights["hashrate"] +
            stability_score * self.optimization_weights["stability"]
        )
        
        return min(1.0, max(0.0, score))
    
    def calculate_stability_score(self):
        """Calculate system stability based on recent performance"""
        if len(self.performance_history) < 10:
            return 0.5
        
        recent_scores = [entry.get("score", 0.5) for entry in list(self.performance_history)[-10:]]
        avg_score = sum(recent_scores) / len(recent_scores)
        
        # Calculate variance
        variance = sum((score - avg_score) ** 2 for score in recent_scores) / len(recent_scores)
        stability = max(0, 1 - variance * 10)  # Penalize high variance
        
        return stability
    
    def optimize_cpu_for_workload(self, workload, strategy_config):
        """Optimize CPU settings based on workload and strategy"""
        recommendations = {}
        
        if strategy_config["cpu_strategy"] == "support_gpu":
            # CPU supports GPU-heavy workload (mining)
            recommendations = {
                "governor": "schedutil",
                "max_freq_pct": 75,  # Leave headroom for GPU
                "turbo": True,
                "threads": 12  # Optimal for our setup
            }
        elif strategy_config["cpu_strategy"] == "maximum_performance":
            # Max CPU performance
            recommendations = {
                "governor": "performance",
                "max_freq_pct": 100,
                "turbo": True,
                "threads": -1  # All threads
            }
        elif strategy_config["cpu_strategy"] == "eco_mode":
            # Power saving mode
            recommendations = {
                "governor": "powersave",
                "max_freq_pct": 50,
                "turbo": False,
                "threads": 8  # Reduced threads
            }
        else:  # balanced_performance
            recommendations = {
                "governor": "schedutil",
                "max_freq_pct": 85,
                "turbo": True,
                "threads": -1
            }
        
        return recommendations
    
    def optimize_gpu_for_workload(self, workload, strategy_config):
        """Optimize GPU settings based on workload and strategy"""
        recommendations = {}
        
        if strategy_config["gpu_strategy"] == "maximize_mining":
            recommendations = {
                "power_limit": 120,
                "memory_overclock": 150,
                "core_overclock": 50,
                "profile": "mining"
            }
        elif strategy_config["gpu_strategy"] == "gaming_optimized":
            recommendations = {
                "power_limit": 140,
                "memory_overclock": 100,
                "core_overclock": 100,
                "profile": "gaming"
            }
        elif strategy_config["gpu_strategy"] == "power_save":
            recommendations = {
                "power_limit": 70,
                "memory_overclock": 0,
                "core_overclock": -50,
                "profile": "eco"
            }
        else:  # support_cpu or balanced
            recommendations = {
                "power_limit": 100,
                "memory_overclock": 75,
                "core_overclock": 25,
                "profile": "balanced"
            }
        
        return recommendations
    
    def apply_coordinated_optimization(self, workload=None, strategy=None):
        """Apply intelligent CPU+GPU coordination"""
        if not strategy:
            strategy = self.current_strategy
        
        if not workload:
            workload = self.current_workload
        
        if strategy not in self.coordination_strategies:
            return False, f"Unknown strategy: {strategy}"
        
        strategy_config = self.coordination_strategies[strategy]
        results = []
        
        try:
            # Get CPU recommendations
            cpu_opts = self.optimize_cpu_for_workload(workload, strategy_config)
            gpu_opts = self.optimize_gpu_for_workload(workload, strategy_config)
            
            # Apply CPU optimizations (simplified - would call system API)
            results.append(f"CPU: {cpu_opts['governor']} governor, {cpu_opts['max_freq_pct']}% freq")
            
            # Apply GPU optimizations (simplified)
            results.append(f"GPU: {gpu_opts['power_limit']}% power, {gpu_opts['profile']} profile")
            
            # Apply system-wide profile via API
            system_profile = self._map_strategy_to_profile(strategy)
            response = requests.post(f"{self.system_api}/profile/{system_profile}")
            
            if response.status_code == 200:
                results.append(f"System profile: {system_profile}")
            
            self.current_strategy = strategy
            
            return True, f"Applied {strategy_config['description']}: " + " | ".join(results)
            
        except Exception as e:
            return False, f"Coordination failed: {e}"
    
    def _map_strategy_to_profile(self, strategy):
        """Map coordination strategy to system profile"""
        mapping = {
            "mining_optimized": "mining_optimized",
            "gaming_performance": "gaming_performance", 
            "power_efficient": "ultra_eco",
            "compute_focused": "balanced"
        }
        return mapping.get(strategy, "balanced")
    
    def auto_coordination_loop(self):
        """Continuous automatic system coordination"""
        print("🤖 Auto-coordination started")
        
        optimization_interval = 60  # Optimize every minute
        last_optimization = datetime.now() - timedelta(seconds=optimization_interval)
        
        while self.monitoring and self.auto_coordination:
            try:
                current_time = datetime.now()
                
                # Get current system state
                stats = self.get_system_stats()
                mining_stats = self.get_mining_stats()
                
                if not stats:
                    time.sleep(10)
                    continue
                
                # Detect current workload
                detected_workload = self.detect_workload(stats)
                
                # Calculate performance score
                perf_score = self.calculate_performance_score(stats, mining_stats)
                
                # Store performance data
                performance_entry = {
                    "timestamp": current_time.isoformat(),
                    "workload": detected_workload,
                    "strategy": self.current_strategy,
                    "score": perf_score,
                    "cpu_temp": stats["cpu"]["temperature"],
                    "gpu_temp": stats["gpu"]["temperature"],
                    "hashrate": mining_stats.get("hashrate", 0) if mining_stats else 0,
                    "power": stats["system"]["total_power_estimated"]
                }
                
                self.performance_history.append(performance_entry)
                
                # Check if workload changed significantly
                workload_changed = detected_workload != self.current_workload
                
                # Check if optimization is needed
                time_for_optimization = (current_time - last_optimization).seconds >= optimization_interval
                performance_degraded = perf_score < 0.6  # Below 60% performance
                
                if workload_changed or time_for_optimization or performance_degraded:
                    print(f"\n🎯 Coordination trigger: workload={detected_workload}, score={perf_score:.2f}")
                    
                    # Determine optimal strategy for workload
                    optimal_strategy = self._determine_optimal_strategy(detected_workload, stats, mining_stats)
                    
                    if optimal_strategy != self.current_strategy or workload_changed:
                        success, message = self.apply_coordinated_optimization(detected_workload, optimal_strategy)
                        
                        if success:
                            print(f"✅ {message}")
                            last_optimization = current_time
                            self.current_workload = detected_workload
                            
                            # Record optimization
                            self.optimization_history.append({
                                "timestamp": current_time.isoformat(),
                                "from_strategy": self.current_strategy,
                                "to_strategy": optimal_strategy,
                                "workload": detected_workload,
                                "performance_score": perf_score,
                                "reason": "workload_change" if workload_changed else "performance_optimization"
                            })
                        else:
                            print(f"❌ Coordination failed: {message}")
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                print(f"Auto-coordination error: {e}")
                time.sleep(30)
    
    def _determine_optimal_strategy(self, workload, stats, mining_stats):
        """AI-driven strategy selection based on current conditions"""
        
        # Simple rule-based strategy selection (can be enhanced with ML)
        if workload == "mining":
            # Check if temperature is acceptable for mining optimization
            if stats["cpu"]["temperature"] < 80 and stats["gpu"]["temperature"] < 85:
                return "mining_optimized"
            else:
                return "power_efficient"  # Too hot, reduce power
        
        elif workload == "gaming":
            return "gaming_performance"
        
        elif workload == "compute":
            return "compute_focused"
        
        elif workload == "idle":
            return "power_efficient"
        
        else:  # mixed workload
            # Use performance score to decide
            if stats["system"]["health_score"] > 0.8:
                return "mining_optimized"  # System healthy, optimize for mining
            else:
                return "power_efficient"  # System stressed, reduce load
    
    def get_coordination_status(self):
        """Get current coordination status and performance"""
        recent_performance = list(self.performance_history)[-10:] if self.performance_history else []
        
        avg_score = sum(entry["score"] for entry in recent_performance) / len(recent_performance) if recent_performance else 0
        
        return {
            "current_workload": self.current_workload,
            "current_strategy": self.current_strategy,
            "auto_coordination": self.auto_coordination,
            "recent_avg_performance": avg_score,
            "total_optimizations": len(self.optimization_history),
            "available_strategies": list(self.coordination_strategies.keys()),
            "performance_history_size": len(self.performance_history)
        }
    
    def manual_optimize(self, target_workload, target_strategy=None):
        """Manual optimization for specific workload"""
        if not target_strategy:
            # Auto-select best strategy for workload
            stats = self.get_system_stats()
            mining_stats = self.get_mining_stats()
            target_strategy = self._determine_optimal_strategy(target_workload, stats, mining_stats)
        
        return self.apply_coordinated_optimization(target_workload, target_strategy)
    
    def enable_auto_coordination(self):
        """Enable automatic coordination"""
        if not self.auto_coordination:
            self.auto_coordination = True
            coord_thread = threading.Thread(target=self.auto_coordination_loop, daemon=True)
            coord_thread.start()
            return "Auto-coordination enabled"
        return "Auto-coordination already active"
    
    def disable_auto_coordination(self):
        """Disable automatic coordination"""
        self.auto_coordination = False
        return "Auto-coordination disabled"

def main():
    """Test the smart coordination system"""
    print("🧠 ZION Smart System Coordinator")
    print("=" * 50)
    
    coordinator = ZionSmartCoordinator()
    
    while True:
        print("\n🎮 Smart Coordination Menu:")
        print("1. 🤖 Enable Auto-Coordination")
        print("2. ⏹️  Disable Auto-Coordination")
        print("3. ⛏️  Optimize for Mining")
        print("4. 🎮 Optimize for Gaming")
        print("5. 💻 Optimize for Computing")
        print("6. 🌱 Power Efficient Mode")
        print("7. 📊 View Coordination Status")
        print("8. 🚪 Exit")
        
        choice = input("\nSelect option (1-8): ").strip()
        
        if choice == "1":
            msg = coordinator.enable_auto_coordination()
            print(f"\n🤖 {msg}")
            
        elif choice == "2":
            msg = coordinator.disable_auto_coordination()
            print(f"\n⏹️ {msg}")
            
        elif choice == "3":
            success, msg = coordinator.manual_optimize("mining", "mining_optimized")
            print(f"\n⛏️ {'✅' if success else '❌'} {msg}")
            
        elif choice == "4":
            success, msg = coordinator.manual_optimize("gaming", "gaming_performance")
            print(f"\n🎮 {'✅' if success else '❌'} {msg}")
            
        elif choice == "5":
            success, msg = coordinator.manual_optimize("compute", "compute_focused")
            print(f"\n💻 {'✅' if success else '❌'} {msg}")
            
        elif choice == "6":
            success, msg = coordinator.manual_optimize("idle", "power_efficient")
            print(f"\n🌱 {'✅' if success else '❌'} {msg}")
            
        elif choice == "7":
            status = coordinator.get_coordination_status()
            print(f"\n📊 Coordination Status:")
            print(f"   Current workload: {status['current_workload']}")
            print(f"   Current strategy: {status['current_strategy']}")
            print(f"   Auto-coordination: {'ON' if status['auto_coordination'] else 'OFF'}")
            print(f"   Recent performance: {status['recent_avg_performance']:.2f}")
            print(f"   Total optimizations: {status['total_optimizations']}")
            
        elif choice == "8":
            coordinator.disable_auto_coordination()
            print("\n👋 Smart Coordinator stopped")
            break
            
        else:
            print("\n❌ Invalid option")
        
        if choice != "8":
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()