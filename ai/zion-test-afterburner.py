#!/usr/bin/env python3
"""
ZION System Afterburner - Simplified Test Version
Basic CPU+GPU monitoring for testing
"""

import os
import json
import time
import psutil
from flask import Flask, jsonify
from datetime import datetime

app = Flask(__name__)

def get_cpu_info():
    """Get basic CPU information"""
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

def get_system_stats():
    """Get current system stats"""
    cpu_info = get_cpu_info()
    
    # Get CPU temperature (try different sources)
    cpu_temp = 45  # Default
    try:
        # Try reading from thermal zones
        thermal_files = [
            '/sys/class/thermal/thermal_zone0/temp',
            '/sys/class/thermal/thermal_zone1/temp'
        ]
        for thermal_file in thermal_files:
            if os.path.exists(thermal_file):
                with open(thermal_file, 'r') as f:
                    temp = int(f.read().strip()) / 1000
                    if 30 <= temp <= 100:  # Reasonable CPU temp range
                        cpu_temp = temp
                        break
    except:
        pass
    
    # Get memory info
    memory = psutil.virtual_memory()
    
    # CPU frequency
    try:
        freq = psutil.cpu_freq()
        current_freq = freq.current if freq else 3600
    except:
        current_freq = 3600
    
    return {
        "cpu": {
            "model": cpu_info["model"],
            "temperature": cpu_temp,
            "usage_percent": psutil.cpu_percent(interval=1),
            "usage_per_core": psutil.cpu_percent(percpu=True),
            "frequency": current_freq,
            "cores": cpu_info["cores"],
            "threads": cpu_info["threads"],
            "memory_percent": memory.percent,
            "memory_used_gb": memory.used / (1024**3),
            "memory_total_gb": memory.total / (1024**3)
        },
        "gpu": {
            "model": "AMD Radeon RX 5600 XT",
            "temperature": 72,  # Mock GPU data for now
            "power_usage": 110,
            "utilization": 85,
            "vram_used": 4200,
            "vram_total": 6144
        },
        "system": {
            "total_power_estimated": cpu_temp * 2 + 110,  # Simple estimation
            "uptime_hours": time.time() - psutil.boot_time(),
            "timestamp": datetime.now().isoformat()
        }
    }

@app.route('/api/system/status')
def system_status():
    """Get system status"""
    cpu_info = get_cpu_info()
    return jsonify({
        "status": "active",
        "message": "ZION System Afterburner Test Version",
        "cpu_info": cpu_info,
        "version": "1.0.0-test",
        "profiles": ["ultra_eco", "balanced", "mining_optimized", "gaming_performance"]
    })

@app.route('/api/system/stats')
def system_stats():
    """Get real-time system statistics"""
    stats = get_system_stats()
    return jsonify({
        "status": "success",
        "current": stats,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/mining/optimize')
def mining_optimize():
    """Simulate mining optimization"""
    stats = get_system_stats()
    
    # Simple optimization logic
    cpu_temp = stats["cpu"]["temperature"]
    
    if cpu_temp > 80:
        profile = "eco_mode"
        message = "Applied ECO mode due to high temperature"
        hashrate = "5800-6000 H/s"
    elif cpu_temp < 70:
        profile = "mining_optimized"
        message = "Applied MINING optimization - good temperature"  
        hashrate = "6000-6200 H/s"
    else:
        profile = "balanced"
        message = "Applied BALANCED profile"
        hashrate = "5900-6100 H/s"
    
    return jsonify({
        "success": True,
        "message": message,
        "profile_applied": profile,
        "estimated_hashrate": hashrate,
        "cpu_temp": cpu_temp,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/')
def home():
    """Home page with quick info"""
    return jsonify({
        "service": "ZION System Afterburner",
        "version": "1.0.0-test", 
        "endpoints": [
            "/api/system/status",
            "/api/system/stats",
            "/api/mining/optimize"
        ],
        "status": "running"
    })

if __name__ == "__main__":
    print("🚀 ZION System Afterburner (Test Version) Starting...")
    print("=" * 50)
    
    cpu_info = get_cpu_info()
    print(f"🖥️  CPU: {cpu_info['model']}")
    print(f"🔥 Cores: {cpu_info['cores']} cores, {cpu_info['threads']} threads")
    print(f"🎮 GPU: AMD Radeon RX 5600 XT (simulated)")
    print()
    print("📊 Test endpoints:")
    print("   GET  /api/system/status")
    print("   GET  /api/system/stats") 
    print("   GET  /api/mining/optimize")
    print()
    print(f"🌐 Starting on http://localhost:5003")
    print("Press Ctrl+C to stop")
    
    try:
        app.run(host='0.0.0.0', port=5003, debug=False)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")