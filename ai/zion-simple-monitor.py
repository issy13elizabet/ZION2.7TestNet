#!/usr/bin/env python3
"""
ZION System Monitor - Very Simple Version
Direct system monitoring without Flask server
"""

import os
import json
import time
import psutil
from datetime import datetime

def get_cpu_info():
    """Get CPU information"""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if 'model name' in line:
                    model = line.split(':')[1].strip()
                    break
        return {
            "model": model,
            "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True)
        }
    except:
        return {
            "model": "AMD Ryzen 5 3600",
            "cores": 6,
            "threads": 12
        }

def get_cpu_temperature():
    """Get CPU temperature from various sources"""
    temp = 45  # Default
    
    # Try different thermal zone files
    thermal_files = [
        '/sys/class/thermal/thermal_zone0/temp',
        '/sys/class/thermal/thermal_zone1/temp',
        '/sys/class/thermal/thermal_zone2/temp'
    ]
    
    for thermal_file in thermal_files:
        try:
            if os.path.exists(thermal_file):
                with open(thermal_file, 'r') as f:
                    temp_raw = int(f.read().strip())
                    # Convert millicelsius to celsius
                    temp_celsius = temp_raw / 1000
                    if 30 <= temp_celsius <= 100:  # Reasonable range
                        temp = temp_celsius
                        print(f"📊 CPU Temperature from {thermal_file}: {temp_celsius}°C")
                        break
        except Exception as e:
            continue
    
    return temp

def get_cpu_governor():
    """Get current CPU governor"""
    try:
        with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor', 'r') as f:
            return f.read().strip()
    except:
        return "unknown"

def get_cpu_frequency():
    """Get current CPU frequency"""
    try:
        freq = psutil.cpu_freq()
        if freq:
            return {
                "current": freq.current,
                "min": freq.min,
                "max": freq.max
            }
    except:
        pass
    
    return {"current": 3600, "min": 1400, "max": 4200}

def monitor_system():
    """Main monitoring function"""
    print("🚀 ZION System Monitor")
    print("=" * 50)
    
    cpu_info = get_cpu_info()
    print(f"🖥️  CPU: {cpu_info['model']}")
    print(f"🔥 Cores: {cpu_info['cores']} cores, {cpu_info['threads']} threads")
    print()
    
    try:
        while True:
            print(f"\n📊 System Stats - {datetime.now().strftime('%H:%M:%S')}")
            print("-" * 40)
            
            # CPU Stats
            cpu_temp = get_cpu_temperature()
            cpu_usage = psutil.cpu_percent(interval=1)
            cpu_freq = get_cpu_frequency()
            cpu_governor = get_cpu_governor()
            
            print(f"🌡️  CPU Temperature: {cpu_temp:.1f}°C")
            print(f"📊 CPU Usage: {cpu_usage:.1f}%")
            print(f"⚡ CPU Frequency: {cpu_freq['current']:.0f} MHz")
            print(f"🎯 CPU Governor: {cpu_governor}")
            
            # Memory Stats
            memory = psutil.virtual_memory()
            print(f"💾 Memory: {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB)")
            
            # Per-core usage
            core_usage = psutil.cpu_percent(percpu=True)
            print(f"🧵 Core Usage: {', '.join(f'{usage:.0f}%' for usage in core_usage)}")
            
            # Load averages
            load1, load5, load15 = os.getloadavg()
            print(f"📈 Load: {load1:.2f}, {load5:.2f}, {load15:.2f}")
            
            # Power estimation (simple)
            estimated_power = cpu_temp * 1.5 + cpu_usage * 0.8 + 45  # Base + thermal + load
            print(f"🔋 Estimated CPU Power: {estimated_power:.0f}W")
            
            # Mining simulation
            if cpu_usage > 70:
                hashrate = 5800 + (cpu_usage - 70) * 20  # Simulate mining hashrate
                efficiency = hashrate / estimated_power
                print(f"⛏️  Mining Simulation: {hashrate:.0f} H/s ({efficiency:.1f} H/W)")
            
            # Status
            if cpu_temp > 80:
                status = "🔥 HOT - Consider ECO mode"
            elif cpu_temp < 65:
                status = "❄️  COOL - Can push harder"
            else:
                status = "✅ OPTIMAL"
            
            print(f"🎯 Status: {status}")
            
            # Create JSON output for frontend integration
            stats_data = {
                "timestamp": datetime.now().isoformat(),
                "cpu": {
                    "model": cpu_info["model"],
                    "temperature": cpu_temp,
                    "usage": cpu_usage,
                    "frequency": cpu_freq["current"],
                    "governor": cpu_governor,
                    "cores": cpu_info["cores"],
                    "threads": cpu_info["threads"]
                },
                "memory": {
                    "percent": memory.percent,
                    "used_gb": memory.used / 1024**3,
                    "total_gb": memory.total / 1024**3
                },
                "system": {
                    "load": [load1, load5, load15],
                    "estimated_power": estimated_power,
                    "status": status
                }
            }
            
            # Save stats to file for potential frontend use
            with open('/tmp/zion_system_stats.json', 'w') as f:
                json.dump(stats_data, f, indent=2)
            
            print(f"📄 Stats saved to: /tmp/zion_system_stats.json")
            
            time.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        print(f"\n👋 Monitoring stopped")

if __name__ == "__main__":
    monitor_system()