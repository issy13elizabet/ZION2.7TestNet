"""
ZION Advanced Mining Features v2.6.75
Auto-tuning, Temperature Monitoring, Dynamic Optimization

Advanced mining system s:
- Automatické ladění výkonu
- Monitoring teploty CPU
- Dynamická adjustace difficulty
- Adaptivní thread management
- Power efficiency optimization
"""
import psutil
import time
import threading
import json
import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from collections import deque
import statistics

logger = logging.getLogger(__name__)

@dataclass 
class MiningMetrics:
    """Mining performance metrics"""
    hashrate: float = 0.0
    temperature: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    power_efficiency: float = 0.0
    thread_count: int = 0
    shares_per_minute: float = 0.0
    error_rate: float = 0.0
    uptime_hours: float = 0.0

@dataclass
class OptimizationSettings:
    """Dynamic optimization settings"""
    target_temperature: float = 75.0  # °C
    max_temperature: float = 85.0     # °C
    target_cpu_usage: float = 90.0    # %
    min_hashrate_threshold: float = 10.0  # H/s
    auto_tuning_enabled: bool = True
    thermal_throttling: bool = True
    dynamic_threads: bool = True
    efficiency_mode: bool = False

class ZionAdvancedMiningOptimizer:
    """
    Advanced mining optimizer s AI-powered tuning
    """
    
    def __init__(self, base_threads: int = None):
        self.base_threads = base_threads or max(1, psutil.cpu_count() - 1)
        self.current_threads = self.base_threads
        self.optimization_settings = OptimizationSettings()
        
        # Performance tracking
        self.metrics_history = deque(maxlen=300)  # 5 minutes at 1s intervals
        self.current_metrics = MiningMetrics()
        
        # Auto-tuning state
        self.tuning_active = False
        self.last_tuning_time = time.time()
        self.tuning_cooldown = 30.0  # seconds
        
        # Thread management
        self.active_miners = {}
        self.miner_lock = threading.Lock()
        
        # Monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        
        logger.info(f"ZION Advanced Mining Optimizer initialized")
        logger.info(f"Base threads: {self.base_threads}")
        logger.info(f"CPU cores: {psutil.cpu_count()}")
        
    def get_cpu_temperature(self) -> float:
        """Get CPU temperature from system sensors"""
        try:
            temps = psutil.sensors_temperatures()
            
            # Try different sensor sources
            temp_sources = ['coretemp', 'k10temp', 'acpi', 'cpu_thermal']
            
            for source in temp_sources:
                if source in temps and temps[source]:
                    # Get highest temperature from this source
                    cpu_temps = [t.current for t in temps[source] 
                               if t.label and 'cpu' in t.label.lower()]
                    if cpu_temps:
                        return max(cpu_temps)
                    
                    # Fallback to any temperature from this source
                    all_temps = [t.current for t in temps[source]]
                    if all_temps:
                        return max(all_temps)
            
            # Fallback - return 0 if no temperature found
            return 0.0
            
        except Exception as e:
            logger.debug(f"Temperature reading failed: {e}")
            return 0.0
    
    def get_system_metrics(self) -> MiningMetrics:
        """Collect current system performance metrics"""
        try:
            # CPU and memory info
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Temperature
            temperature = self.get_cpu_temperature()
            
            # Mining specific metrics (to be updated by mining threads)
            metrics = MiningMetrics(
                temperature=temperature,
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                thread_count=self.current_threads,
                hashrate=self.current_metrics.hashrate,
                shares_per_minute=self.current_metrics.shares_per_minute,
                error_rate=self.current_metrics.error_rate,
                uptime_hours=self.current_metrics.uptime_hours
            )
            
            # Calculate power efficiency (hashrate per CPU usage)
            if cpu_percent > 0:
                metrics.power_efficiency = metrics.hashrate / cpu_percent
            
            return metrics
            
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
            return MiningMetrics()
    
    def should_thermal_throttle(self, metrics: MiningMetrics) -> bool:
        """Determine if thermal throttling is needed"""
        if not self.optimization_settings.thermal_throttling:
            return False
            
        if metrics.temperature == 0:  # No temperature data
            return False
            
        # Aggressive throttling near max temp
        if metrics.temperature >= self.optimization_settings.max_temperature:
            return True
            
        # Gradual throttling above target temp
        if metrics.temperature > self.optimization_settings.target_temperature:
            temp_ratio = (metrics.temperature - self.optimization_settings.target_temperature) / 10.0
            return temp_ratio > 0.5  # 50% chance to throttle at +5°C above target
            
        return False
    
    def calculate_optimal_threads(self, metrics: MiningMetrics) -> int:
        """Calculate optimal thread count based on current metrics"""
        if not self.optimization_settings.dynamic_threads:
            return self.base_threads
        
        optimal_threads = self.current_threads
        
        # Thermal considerations
        if self.should_thermal_throttle(metrics):
            # Reduce threads to cool down
            optimal_threads = max(1, self.current_threads - 1)
            logger.info(f"🌡️ Thermal throttling: reducing threads to {optimal_threads}")
            
        # Performance considerations
        elif len(self.metrics_history) >= 10:
            # Analyze recent performance trends
            recent_hashrates = [m.hashrate for m in list(self.metrics_history)[-10:]]
            recent_temps = [m.temperature for m in list(self.metrics_history)[-10:] if m.temperature > 0]
            
            if recent_hashrates and len(recent_hashrates) >= 5:
                avg_hashrate = statistics.mean(recent_hashrates)
                hashrate_trend = recent_hashrates[-1] - recent_hashrates[0]
                
                # If hashrate is declining and we have thermal headroom
                if hashrate_trend < -10 and (not recent_temps or max(recent_temps) < self.optimization_settings.target_temperature):
                    if self.current_threads > 1:
                        optimal_threads = self.current_threads - 1
                        logger.info(f"📉 Performance declining: reducing threads to {optimal_threads}")
                
                # If hashrate is good and temperature is low, try adding threads
                elif (hashrate_trend > 0 and 
                      avg_hashrate > self.optimization_settings.min_hashrate_threshold and
                      self.current_threads < psutil.cpu_count() and
                      (not recent_temps or max(recent_temps) < self.optimization_settings.target_temperature - 5)):
                    optimal_threads = min(psutil.cpu_count(), self.current_threads + 1)
                    logger.info(f"📈 Adding thread for better performance: {optimal_threads}")
        
        # Sanity limits
        optimal_threads = max(1, min(psutil.cpu_count(), optimal_threads))
        return optimal_threads
    
    def auto_tune_mining(self, current_hashrate: float) -> Dict[str, Any]:
        """
        Perform automatic mining optimization
        """
        if not self.optimization_settings.auto_tuning_enabled:
            return {'tuning_applied': False, 'reason': 'Auto-tuning disabled'}
        
        # Cooldown check
        if time.time() - self.last_tuning_time < self.tuning_cooldown:
            return {'tuning_applied': False, 'reason': 'Cooldown active'}
        
        # Get current metrics
        metrics = self.get_system_metrics()
        metrics.hashrate = current_hashrate  # Update with actual mining hashrate
        
        # Add to history
        self.metrics_history.append(metrics)
        self.current_metrics = metrics
        
        # Calculate optimizations
        optimal_threads = self.calculate_optimal_threads(metrics)
        
        tuning_result = {
            'tuning_applied': False,
            'current_metrics': asdict(metrics),
            'recommendations': {}
        }
        
        # Apply thread optimization
        if optimal_threads != self.current_threads:
            old_threads = self.current_threads
            self.current_threads = optimal_threads
            
            tuning_result.update({
                'tuning_applied': True,
                'thread_change': {
                    'from': old_threads,
                    'to': optimal_threads,
                    'reason': 'Performance/thermal optimization'
                }
            })
            
            self.last_tuning_time = time.time()
        
        # Generate recommendations
        recommendations = []
        
        if metrics.temperature > self.optimization_settings.target_temperature:
            recommendations.append(f"High temperature ({metrics.temperature:.1f}°C) - consider cooling")
            
        if metrics.cpu_usage < 80:
            recommendations.append("Low CPU usage - mining may not be utilizing full capacity")
            
        if metrics.hashrate < self.optimization_settings.min_hashrate_threshold:
            recommendations.append("Low hashrate - check mining configuration")
        
        tuning_result['recommendations'] = recommendations
        
        return tuning_result
    
    def start_monitoring(self, update_callback=None):
        """Start system monitoring thread"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        
        def monitor_loop():
            logger.info("🔍 Advanced monitoring started")
            
            while self.monitoring_active:
                try:
                    metrics = self.get_system_metrics()
                    self.current_metrics = metrics
                    
                    # Call update callback if provided
                    if update_callback:
                        update_callback(metrics)
                    
                    # Log warnings for concerning metrics
                    if metrics.temperature > self.optimization_settings.max_temperature:
                        logger.warning(f"🔥 Critical temperature: {metrics.temperature:.1f}°C")
                    elif metrics.temperature > self.optimization_settings.target_temperature:
                        logger.info(f"🌡️ Elevated temperature: {metrics.temperature:.1f}°C")
                    
                    time.sleep(1.0)  # Update every second
                    
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    time.sleep(5.0)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.metrics_history:
            return {'error': 'No metrics available'}
        
        # Calculate statistics from history
        hashrates = [m.hashrate for m in self.metrics_history if m.hashrate > 0]
        temperatures = [m.temperature for m in self.metrics_history if m.temperature > 0]
        cpu_usages = [m.cpu_usage for m in self.metrics_history]
        
        report = {
            'current_metrics': asdict(self.current_metrics),
            'optimization_settings': asdict(self.optimization_settings),
            'statistics': {
                'data_points': len(self.metrics_history),
                'monitoring_duration_minutes': len(self.metrics_history) / 60.0
            }
        }
        
        if hashrates:
            report['statistics'].update({
                'avg_hashrate': statistics.mean(hashrates),
                'max_hashrate': max(hashrates),
                'min_hashrate': min(hashrates),
                'hashrate_stability': statistics.stdev(hashrates) if len(hashrates) > 1 else 0
            })
        
        if temperatures:
            report['statistics'].update({
                'avg_temperature': statistics.mean(temperatures),
                'max_temperature': max(temperatures),
                'min_temperature': min(temperatures)
            })
        
        if cpu_usages:
            report['statistics'].update({
                'avg_cpu_usage': statistics.mean(cpu_usages),
                'max_cpu_usage': max(cpu_usages)
            })
        
        return report
    
    def save_performance_log(self, filename: str = None):
        """Save performance data to JSON file"""
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"zion_mining_performance_{timestamp}.json"
        
        data = {
            'timestamp': time.time(),
            'report': self.get_performance_report(),
            'metrics_history': [asdict(m) for m in self.metrics_history]
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Performance log saved: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to save performance log: {e}")
            return None


def create_advanced_miner_test():
    """Test function for advanced mining features"""
    
    print("🚀 ZION Advanced Mining Features Test")
    print("=" * 50)
    
    # Initialize optimizer
    optimizer = ZionAdvancedMiningOptimizer(base_threads=4)
    
    # Start monitoring
    def metrics_callback(metrics):
        if int(time.time()) % 5 == 0:  # Print every 5 seconds
            print(f"📊 Temp: {metrics.temperature:.1f}°C | "
                  f"CPU: {metrics.cpu_usage:.1f}% | "
                  f"Threads: {metrics.thread_count} | "
                  f"Hashrate: {metrics.hashrate:.0f} H/s")
    
    optimizer.start_monitoring(metrics_callback)
    
    try:
        # Simulate mining with auto-tuning
        print("🔧 Testing auto-tuning for 30 seconds...")
        
        for i in range(30):
            # Simulate varying hashrate
            simulated_hashrate = 1000 + (i * 50) + (time.time() % 100)
            
            # Run auto-tuning
            tuning_result = optimizer.auto_tune_mining(simulated_hashrate)
            
            if tuning_result['tuning_applied']:
                print(f"🎯 Auto-tuning applied: {tuning_result}")
            
            time.sleep(1)
        
        # Generate and display report
        print("\n📋 Performance Report:")
        report = optimizer.get_performance_report()
        
        if 'statistics' in report:
            stats = report['statistics']
            print(f"   Monitoring duration: {stats.get('monitoring_duration_minutes', 0):.1f} minutes")
            if 'avg_hashrate' in stats:
                print(f"   Average hashrate: {stats['avg_hashrate']:.0f} H/s")
            if 'avg_temperature' in stats:
                print(f"   Average temperature: {stats['avg_temperature']:.1f}°C")
            if 'avg_cpu_usage' in stats:
                print(f"   Average CPU usage: {stats['avg_cpu_usage']:.1f}%")
        
        # Save performance log
        log_file = optimizer.save_performance_log()
        if log_file:
            print(f"💾 Performance log saved: {log_file}")
        
    finally:
        optimizer.stop_monitoring()
        print("🏁 Advanced mining test completed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_advanced_miner_test()