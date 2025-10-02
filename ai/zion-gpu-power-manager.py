#!/usr/bin/env python3
"""
ZION Advanced GPU Power Management
Sophisticated power limiting, undervolting, and thermal management
"""

import os
import json
import time
import subprocess
import threading
from pathlib import Path

class ZionGPUPowerManager:
    def __init__(self):
        self.gpu_path = "/sys/class/drm/card0/device"
        self.hwmon_path = self._find_amdgpu_hwmon()
        
        # Advanced power profiles with curve tuning
        self.advanced_profiles = {
            "ultra_eco": {
                "name": "Ultra ECO",
                "power_limit": 60,  # 60% of TDP
                "voltage_offset": -100,  # mV
                "core_clock_max": 1200,  # MHz
                "memory_clock": 1500,    # MHz
                "fan_curve": [(30, 20), (50, 30), (70, 50), (80, 70), (90, 100)],
                "temp_target": 65,
                "description": "Maximum power savings, 40W target"
            },
            "efficient_mining": {
                "name": "Efficient Mining",
                "power_limit": 85,
                "voltage_offset": -50,
                "core_clock_max": 1500,
                "memory_clock": 1750,
                "fan_curve": [(30, 25), (60, 40), (75, 60), (85, 80), (90, 100)],
                "temp_target": 75,
                "description": "Optimized mining efficiency"
            },
            "performance": {
                "name": "Max Performance", 
                "power_limit": 120,
                "voltage_offset": 0,
                "core_clock_max": 1750,
                "memory_clock": 1750,
                "fan_curve": [(30, 30), (55, 50), (70, 70), (80, 85), (90, 100)],
                "temp_target": 80,
                "description": "Maximum performance mode"
            },
            "silent": {
                "name": "Silent Operation",
                "power_limit": 70,
                "voltage_offset": -75,
                "core_clock_max": 1300,
                "memory_clock": 1600,
                "fan_curve": [(30, 15), (60, 25), (75, 35), (85, 45), (90, 60)],
                "temp_target": 70,
                "description": "Minimal noise operation"
            }
        }
        
        # Voltage/frequency curves for undervolting
        self.voltage_curves = {
            "stock": [
                (800, 1100),   # 800MHz @ 1100mV
                (1200, 1150),  # 1200MHz @ 1150mV
                (1500, 1200),  # 1500MHz @ 1200mV
                (1750, 1250)   # 1750MHz @ 1250mV
            ],
            "undervolted": [
                (800, 1050),   # 800MHz @ 1050mV (-50mV)
                (1200, 1100),  # 1200MHz @ 1100mV (-50mV)
                (1500, 1150),  # 1500MHz @ 1150mV (-50mV)
                (1750, 1200)   # 1750MHz @ 1200mV (-50mV)
            ],
            "aggressive_uv": [
                (800, 1000),   # 800MHz @ 1000mV (-100mV)
                (1200, 1050),  # 1200MHz @ 1050mV (-100mV)
                (1500, 1100),  # 1500MHz @ 1100mV (-100mV)
                (1750, 1150)   # 1750MHz @ 1150mV (-100mV)
            ]
        }
        
        # Dynamic power management settings
        self.power_states = {
            "idle": {"min_freq": 300, "max_freq": 800},
            "light": {"min_freq": 800, "max_freq": 1200},
            "medium": {"min_freq": 1200, "max_freq": 1500},
            "heavy": {"min_freq": 1500, "max_freq": 1750}
        }
        
        self.current_state = "medium"
        self.monitoring = True
        self.auto_manage = False
        
    def _find_amdgpu_hwmon(self):
        """Find AMDGPU hwmon directory"""
        try:
            result = subprocess.run(['find', '/sys/class/hwmon/', '-name', 'name', '-exec', 'grep', '-l', 'amdgpu', '{}', ';'], 
                                  capture_output=True, text=True)
            if result.stdout:
                hwmon_file = result.stdout.strip().split('\n')[0]
                return os.path.dirname(hwmon_file)
        except:
            pass
        return None
    
    def read_gpu_sysfs(self, param):
        """Read GPU parameter from sysfs"""
        try:
            path = os.path.join(self.gpu_path, param)
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return f.read().strip()
        except Exception as e:
            print(f"Error reading {param}: {e}")
        return None
    
    def write_gpu_sysfs(self, param, value):
        """Write GPU parameter to sysfs with sudo"""
        try:
            path = os.path.join(self.gpu_path, param)
            if os.path.exists(path):
                cmd = f"echo '{value}' | sudo tee {path} > /dev/null"
                result = subprocess.run(cmd, shell=True)
                return result.returncode == 0
        except Exception as e:
            print(f"Error writing {param}: {e}")
        return False
    
    def set_power_profile_mode(self, mode):
        """Set AMD power profile mode"""
        # AMD power profile modes:
        # 0: DEFAULT, 1: 3D_FULL_SCREEN, 2: POWER_SAVING, 3: VIDEO, 4: VR, 5: COMPUTE
        mode_map = {
            "default": "0",
            "gaming": "1", 
            "power_save": "2",
            "video": "3",
            "vr": "4",
            "compute": "5"  # Best for mining
        }
        
        if mode in mode_map:
            return self.write_gpu_sysfs("pp_power_profile_mode", mode_map[mode])
        return False
    
    def apply_voltage_curve(self, curve_name="undervolted"):
        """Apply voltage/frequency curve for undervolting"""
        if curve_name not in self.voltage_curves:
            return False, f"Curve {curve_name} not found"
        
        try:
            # Enable manual performance level
            self.write_gpu_sysfs("power_dpm_force_performance_level", "manual")
            
            # Build voltage curve string for pp_od_clk_voltage
            curve = self.voltage_curves[curve_name]
            curve_commands = []
            
            for i, (freq, voltage) in enumerate(curve):
                curve_commands.append(f"s {i} {freq} {voltage}")
            
            # Apply each point in the curve
            for cmd in curve_commands:
                self.write_gpu_sysfs("pp_od_clk_voltage", cmd)
            
            # Commit changes
            self.write_gpu_sysfs("pp_od_clk_voltage", "c")
            
            return True, f"Applied {curve_name} voltage curve"
            
        except Exception as e:
            return False, f"Failed to apply voltage curve: {e}"
    
    def set_power_limit_watts(self, watts):
        """Set power limit in watts"""
        try:
            # AMD uses micro-watts in power1_cap
            microwatts = watts * 1000000
            
            if self.hwmon_path:
                power_cap_file = os.path.join(self.hwmon_path, "power1_cap")
                if os.path.exists(power_cap_file):
                    cmd = f"echo {microwatts} | sudo tee {power_cap_file}"
                    result = subprocess.run(cmd, shell=True)
                    return result.returncode == 0
            
            return False
        except Exception as e:
            print(f"Error setting power limit: {e}")
            return False
    
    def set_fan_curve(self, curve_points):
        """Set custom fan curve"""
        try:
            if not self.hwmon_path:
                return False, "HWMON path not found"
            
            # Enable manual fan control
            pwm_enable_file = os.path.join(self.hwmon_path, "pwm1_enable")
            cmd = f"echo 1 | sudo tee {pwm_enable_file}"  # Manual control
            subprocess.run(cmd, shell=True)
            
            # For now, just set a static fan speed based on curve
            # Real implementation would need temperature monitoring loop
            temp = self.get_current_temperature()
            fan_speed = self._calculate_fan_speed(temp, curve_points)
            
            pwm_file = os.path.join(self.hwmon_path, "pwm1")
            pwm_value = int((fan_speed / 100.0) * 255)  # Convert % to PWM
            cmd = f"echo {pwm_value} | sudo tee {pwm_file}"
            result = subprocess.run(cmd, shell=True)
            
            return result.returncode == 0, f"Set fan to {fan_speed}% ({pwm_value} PWM)"
            
        except Exception as e:
            return False, f"Fan curve error: {e}"
    
    def _calculate_fan_speed(self, temp, curve_points):
        """Calculate fan speed from temperature and curve"""
        for i in range(len(curve_points) - 1):
            temp1, fan1 = curve_points[i]
            temp2, fan2 = curve_points[i + 1]
            
            if temp1 <= temp <= temp2:
                # Linear interpolation
                ratio = (temp - temp1) / (temp2 - temp1)
                return fan1 + ratio * (fan2 - fan1)
        
        # Temperature beyond curve, use max
        return curve_points[-1][1]
    
    def get_current_temperature(self):
        """Get current GPU temperature"""
        try:
            if self.hwmon_path:
                temp_file = os.path.join(self.hwmon_path, "temp1_input")
                if os.path.exists(temp_file):
                    with open(temp_file, 'r') as f:
                        return int(f.read().strip()) / 1000  # Convert to Celsius
        except:
            pass
        return 50  # Default fallback
    
    def get_current_power(self):
        """Get current power usage"""
        try:
            if self.hwmon_path:
                power_file = os.path.join(self.hwmon_path, "power1_average")
                if os.path.exists(power_file):
                    with open(power_file, 'r') as f:
                        return int(f.read().strip()) / 1000000  # Convert to Watts
        except:
            pass
        return 0
    
    def apply_advanced_profile(self, profile_name):
        """Apply advanced power management profile"""
        if profile_name not in self.advanced_profiles:
            return False, f"Profile {profile_name} not found"
        
        profile = self.advanced_profiles[profile_name]
        results = []
        
        try:
            # Set power profile mode
            self.set_power_profile_mode("compute")  # Best for mining/compute
            results.append("Set compute power profile")
            
            # Calculate and set power limit
            base_tdp = 150  # RX 5600 XT TDP
            target_watts = int(base_tdp * (profile["power_limit"] / 100))
            
            if self.set_power_limit_watts(target_watts):
                results.append(f"Power limit: {target_watts}W")
            
            # Apply voltage curve based on voltage offset
            if profile["voltage_offset"] <= -75:
                curve_success, curve_msg = self.apply_voltage_curve("aggressive_uv")
            elif profile["voltage_offset"] <= -25:
                curve_success, curve_msg = self.apply_voltage_curve("undervolted")
            else:
                curve_success, curve_msg = self.apply_voltage_curve("stock")
            
            if curve_success:
                results.append(curve_msg)
            
            # Set fan curve
            fan_success, fan_msg = self.set_fan_curve(profile["fan_curve"])
            if fan_success:
                results.append(fan_msg)
            
            return True, f"Applied {profile['name']}: " + ", ".join(results)
            
        except Exception as e:
            return False, f"Profile application failed: {e}"
    
    def auto_power_management(self):
        """Automatic power management based on load and temperature"""
        while self.monitoring and self.auto_manage:
            try:
                temp = self.get_current_temperature()
                power = self.get_current_power()
                utilization = self._get_gpu_utilization()
                
                # Dynamic profile switching
                if temp > 85:  # Too hot
                    self.apply_advanced_profile("ultra_eco")
                    print(f"🌡️ High temp ({temp}°C) - switched to Ultra ECO")
                elif temp > 80 and utilization > 90:  # Hot and busy
                    self.apply_advanced_profile("efficient_mining")
                    print(f"⚖️ Hot mining load - switched to Efficient Mining")
                elif utilization < 10:  # Idle
                    self.apply_advanced_profile("ultra_eco")
                    print(f"💤 Idle detected - switched to Ultra ECO")
                elif utilization > 80:  # Heavy load
                    if temp < 75:  # Cool enough for performance
                        self.apply_advanced_profile("performance")
                        print(f"🚀 Heavy load, cool temp - Performance mode")
                    else:
                        self.apply_advanced_profile("efficient_mining")
                        print(f"⛏️ Heavy load, warm - Efficient Mining")
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                print(f"Auto management error: {e}")
                time.sleep(30)
    
    def _get_gpu_utilization(self):
        """Get GPU utilization percentage"""
        try:
            gpu_busy_file = os.path.join(self.gpu_path, "gpu_busy_percent")
            if os.path.exists(gpu_busy_file):
                with open(gpu_busy_file, 'r') as f:
                    return int(f.read().strip())
        except:
            pass
        return 0
    
    def enable_auto_management(self):
        """Enable automatic power management"""
        self.auto_manage = True
        auto_thread = threading.Thread(target=self.auto_power_management, daemon=True)
        auto_thread.start()
        return True, "Auto power management enabled"
    
    def disable_auto_management(self):
        """Disable automatic power management"""
        self.auto_manage = False
        return True, "Auto power management disabled"
    
    def get_power_status(self):
        """Get comprehensive power management status"""
        return {
            "current_profile": getattr(self, 'current_profile', 'unknown'),
            "temperature": self.get_current_temperature(),
            "power_usage": self.get_current_power(),
            "utilization": self._get_gpu_utilization(),
            "auto_management": self.auto_manage,
            "available_profiles": list(self.advanced_profiles.keys()),
            "voltage_curves": list(self.voltage_curves.keys())
        }

def main():
    """Test power management system"""
    print("🔋 ZION GPU Power Manager Test")
    
    pm = ZionGPUPowerManager()
    
    # Test profile application
    print("\n📊 Current Status:")
    status = pm.get_power_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print("\n🧪 Testing Profiles:")
    for profile_name in pm.advanced_profiles.keys():
        print(f"\nTesting {profile_name}...")
        success, msg = pm.apply_advanced_profile(profile_name)
        print(f"  Result: {'✅' if success else '❌'} {msg}")
        time.sleep(2)
    
    print("\n🤖 Enable auto management? (y/n): ", end="")
    if input().lower() == 'y':
        pm.enable_auto_management()
        print("Auto management enabled. Monitoring...")
        try:
            while True:
                time.sleep(5)
                status = pm.get_power_status()
                print(f"Temp: {status['temperature']}°C, Power: {status['power_usage']}W, Usage: {status['utilization']}%")
        except KeyboardInterrupt:
            pm.disable_auto_management()
            print("\n👋 Auto management stopped")

if __name__ == "__main__":
    main()