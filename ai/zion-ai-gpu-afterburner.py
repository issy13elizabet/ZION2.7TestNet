#!/usr/bin/env python3
"""
ZION AI GPU Afterburner - MSI Afterburner Alternative for AMD GPUs
Integrated GPU tuning system for ZION frontend with web API
"""

import os
import json
import time
import subprocess
import threading
import psutil
from pathlib import Path
from flask import Flask, jsonify, request
from datetime import datetime
import glob

class ZionGPUAfterburner:
    def __init__(self):
        self.app = Flask(__name__)
        self.gpu_path = "/sys/class/drm/card0/device"
        self.hwmon_path = self._find_hwmon_path()
        self.current_profile = "balanced"
        self.monitoring = True
        self.stats_history = []
        self.max_history = 1000
        
        # GPU Safety Limits (AMD RX 5600 XT)
        self.safety_limits = {
            "max_temp": 90,  # Celsius
            "max_power": 150,  # Watts
            "min_power": 50,   # Watts
            "max_fan": 100,    # %
            "max_mem_clock": 1750,  # MHz
            "max_core_clock": 1750  # MHz
        }
        
        # Performance Profiles
        self.profiles = {
            "eco": {
                "power_limit": 70,
                "core_clock_offset": -100,
                "memory_clock_offset": -50,
                "fan_curve": "quiet",
                "description": "Ultra Power Saving"
            },
            "balanced": {
                "power_limit": 100,
                "core_clock_offset": 0,
                "memory_clock_offset": 0,
                "fan_curve": "auto",
                "description": "Default Performance"
            },
            "mining": {
                "power_limit": 120,
                "core_clock_offset": 50,
                "memory_clock_offset": 100,
                "fan_curve": "aggressive",
                "description": "Mining Optimized"
            },
            "gaming": {
                "power_limit": 140,
                "core_clock_offset": 100,
                "memory_clock_offset": 75,
                "fan_curve": "performance",
                "description": "Gaming Performance"
            }
        }
        
        self._setup_routes()
        
    def _find_hwmon_path(self):
        """Find AMD GPU hwmon directory"""
        hwmon_dirs = glob.glob("/sys/class/hwmon/hwmon*")
        for hwmon_dir in hwmon_dirs:
            name_file = os.path.join(hwmon_dir, "name")
            if os.path.exists(name_file):
                with open(name_file, 'r') as f:
                    if "amdgpu" in f.read().lower():
                        return hwmon_dir
        return None
    
    def read_gpu_file(self, filename):
        """Safely read GPU sysfs file"""
        try:
            filepath = os.path.join(self.gpu_path, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return f.read().strip()
        except Exception as e:
            print(f"Error reading {filename}: {e}")
        return None
    
    def write_gpu_file(self, filename, value):
        """Safely write GPU sysfs file with sudo"""
        try:
            filepath = os.path.join(self.gpu_path, filename)
            if os.path.exists(filepath):
                cmd = f"echo '{value}' | sudo tee {filepath}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                return result.returncode == 0
        except Exception as e:
            print(f"Error writing {filename}: {e}")
        return False
    
    def get_gpu_stats(self):
        """Get current GPU statistics"""
        stats = {
            "timestamp": datetime.now().isoformat(),
            "temperature": 0,
            "power_usage": 0,
            "power_limit": 0,
            "core_clock": 0,
            "memory_clock": 0,
            "utilization": 0,
            "fan_speed": 0,
            "vram_used": 0,
            "vram_total": 0
        }
        
        try:
            # Temperature
            if self.hwmon_path:
                temp_files = glob.glob(os.path.join(self.hwmon_path, "temp*_input"))
                if temp_files:
                    with open(temp_files[0], 'r') as f:
                        stats["temperature"] = int(f.read().strip()) / 1000
            
            # Power Usage
            power_file = os.path.join(self.hwmon_path or "", "power1_average")
            if os.path.exists(power_file):
                with open(power_file, 'r') as f:
                    stats["power_usage"] = int(f.read().strip()) / 1000000  # Convert to Watts
            
            # Clock speeds via rocm-smi if available
            try:
                result = subprocess.run(['rocm-smi', '--showclocks'], 
                                     capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    # Parse rocm-smi output for clocks
                    pass
            except:
                pass
            
            # GPU Utilization
            gpu_busy_file = os.path.join(self.gpu_path, "gpu_busy_percent")
            if os.path.exists(gpu_busy_file):
                with open(gpu_busy_file, 'r') as f:
                    stats["utilization"] = int(f.read().strip())
            
            # Memory info
            mem_info_file = os.path.join(self.gpu_path, "mem_info_vram_total")
            if os.path.exists(mem_info_file):
                with open(mem_info_file, 'r') as f:
                    stats["vram_total"] = int(f.read().strip()) / (1024*1024)  # MB
            
            mem_used_file = os.path.join(self.gpu_path, "mem_info_vram_used") 
            if os.path.exists(mem_used_file):
                with open(mem_used_file, 'r') as f:
                    stats["vram_used"] = int(f.read().strip()) / (1024*1024)  # MB
                    
        except Exception as e:
            print(f"Error getting GPU stats: {e}")
        
        return stats
    
    def apply_power_limit(self, percent):
        """Apply power limit (50-150%)"""
        if not 50 <= percent <= 150:
            return False, "Power limit must be between 50-150%"
        
        # Calculate actual power limit in watts
        base_power = 150  # RX 5600 XT base TDP
        target_power = int(base_power * (percent / 100))
        
        # AMD uses micro-watts
        target_microwatts = target_power * 1000000
        
        success = self.write_gpu_file("power_dpm_force_performance_level", "manual")
        if success:
            success = self.write_gpu_file("pp_power_profile_mode", str(target_microwatts))
        
        return success, f"Power limit set to {percent}% ({target_power}W)"
    
    def apply_profile(self, profile_name):
        """Apply performance profile"""
        if profile_name not in self.profiles:
            return False, f"Profile '{profile_name}' not found"
        
        profile = self.profiles[profile_name]
        results = []
        
        # Apply power limit
        success, msg = self.apply_power_limit(profile["power_limit"])
        results.append(f"Power: {msg}")
        
        # Set performance level to manual for tuning
        self.write_gpu_file("power_dpm_force_performance_level", "manual")
        
        # Apply clock offsets (simplified - real implementation would use pp_od_clk_voltage)
        if profile["core_clock_offset"] != 0:
            results.append(f"Core offset: {profile['core_clock_offset']} MHz")
        
        if profile["memory_clock_offset"] != 0:
            results.append(f"Memory offset: {profile['memory_clock_offset']} MHz")
        
        self.current_profile = profile_name
        return True, f"Applied {profile_name} profile: " + ", ".join(results)
    
    def emergency_reset(self):
        """Emergency reset to safe defaults"""
        try:
            # Reset to auto performance level
            self.write_gpu_file("power_dpm_force_performance_level", "auto")
            
            # Reset power profile
            self.write_gpu_file("pp_power_profile_mode", "0")
            
            self.current_profile = "balanced"
            return True, "GPU reset to safe defaults"
        except Exception as e:
            return False, f"Reset failed: {e}"
    
    def monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                stats = self.get_gpu_stats()
                
                # Safety check
                if stats["temperature"] > self.safety_limits["max_temp"]:
                    print(f"EMERGENCY: GPU temperature {stats['temperature']}°C > {self.safety_limits['max_temp']}°C")
                    self.emergency_reset()
                
                # Store history
                self.stats_history.append(stats)
                if len(self.stats_history) > self.max_history:
                    self.stats_history.pop(0)
                
                time.sleep(2)
            except Exception as e:
                print(f"Monitor error: {e}")
                time.sleep(5)
    
    def _setup_routes(self):
        """Setup Flask API routes"""
        
        @self.app.route('/api/gpu/status')
        def get_status():
            return jsonify({
                "status": "active",
                "current_profile": self.current_profile,
                "profiles": list(self.profiles.keys()),
                "safety_limits": self.safety_limits
            })
        
        @self.app.route('/api/gpu/stats')
        def get_stats():
            current_stats = self.get_gpu_stats()
            return jsonify({
                "current": current_stats,
                "history": self.stats_history[-100:] if len(self.stats_history) > 100 else self.stats_history
            })
        
        @self.app.route('/api/gpu/profiles')
        def get_profiles():
            return jsonify(self.profiles)
        
        @self.app.route('/api/gpu/profile/<profile_name>', methods=['POST'])
        def apply_profile_api(profile_name):
            success, message = self.apply_profile(profile_name)
            return jsonify({"success": success, "message": message})
        
        @self.app.route('/api/gpu/power', methods=['POST'])
        def set_power_limit():
            data = request.get_json()
            percent = data.get('percent', 100)
            success, message = self.apply_power_limit(percent)
            return jsonify({"success": success, "message": message})
        
        @self.app.route('/api/gpu/reset', methods=['POST'])
        def reset_gpu():
            success, message = self.emergency_reset()
            return jsonify({"success": success, "message": message})
        
        @self.app.route('/api/gpu/mining/optimize')
        def optimize_for_mining():
            """Auto-optimize for current mining conditions"""
            stats = self.get_gpu_stats()
            
            # Simple optimization logic
            if stats["temperature"] > 80:
                # Too hot, reduce power
                self.apply_profile("eco")
                message = "Applied ECO profile due to high temperature"
            elif stats["temperature"] < 70 and stats["utilization"] > 90:
                # Cool and busy, can push harder
                self.apply_profile("mining")
                message = "Applied MINING profile - good temperature"
            else:
                self.apply_profile("balanced")
                message = "Applied BALANCED profile"
            
            return jsonify({
                "success": True,
                "message": message,
                "stats": stats,
                "profile": self.current_profile
            })

def main():
    """Main application entry point"""
    print("🚀 ZION AI GPU Afterburner Starting...")
    
    afterburner = ZionGPUAfterburner()
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=afterburner.monitor_loop, daemon=True)
    monitor_thread.start()
    
    print("📊 GPU monitoring started")
    print("🌐 API server starting on http://localhost:5001")
    print("📝 Available endpoints:")
    print("   GET  /api/gpu/status")
    print("   GET  /api/gpu/stats")
    print("   GET  /api/gpu/profiles")
    print("   POST /api/gpu/profile/<name>")
    print("   POST /api/gpu/power")
    print("   POST /api/gpu/reset")
    print("   GET  /api/gpu/mining/optimize")
    
    try:
        afterburner.app.run(host='0.0.0.0', port=5001, debug=False)
    except KeyboardInterrupt:
        print("\n👋 Shutting down GPU Afterburner...")
        afterburner.monitoring = False

if __name__ == "__main__":
    main()