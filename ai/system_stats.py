#!/usr/bin/env python3
"""Shared system stats collection for Afterburner components"""
import os, psutil, json, time
from datetime import datetime

THERMAL_PATHS = [
    '/sys/class/thermal/thermal_zone0/temp',
    '/sys/class/thermal/thermal_zone1/temp',
    '/sys/class/thermal/thermal_zone2/temp'
]

def read_cpu_model():
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if 'model name' in line:
                    return line.split(':',1)[1].strip()
    except:  # noqa
        pass
    return 'Unknown CPU'

def read_cpu_temp(default=45):
    for path in THERMAL_PATHS:
        try:
            if os.path.exists(path):
                raw = int(open(path).read().strip())
                c = raw/1000
                if 25 < c < 100:
                    return c
        except:  # noqa
            continue
    return default

def read_cpu_freq():
    try:
        f = psutil.cpu_freq()
        if f:
            return f.current
    except:  # noqa
        pass
    return 0

def collect_stats():
    model = read_cpu_model()
    cpu_usage = psutil.cpu_percent(interval=0.3)
    core_usage = psutil.cpu_percent(percpu=True)
    cpu_temp = read_cpu_temp()
    freq = read_cpu_freq()
    try:
        with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor') as g:
            governor = g.read().strip()
    except:  # noqa
        governor = 'unknown'
    mem = psutil.virtual_memory()
    load1, load5, load15 = os.getloadavg()
    estimated_power = cpu_temp * 1.5 + cpu_usage * 0.8 + 45
    status = '🔥 HOT - Consider ECO mode' if cpu_temp > 80 else ('❄️  COOL - Can push harder' if cpu_temp < 65 else '✅ OPTIMAL')
    data = {
        'timestamp': datetime.now().isoformat(),
        'cpu': {
            'model': model,
            'temperature': cpu_temp,
            'usage': cpu_usage,
            'frequency': freq,
            'governor': governor,
            'cores': psutil.cpu_count(logical=False),
            'threads': psutil.cpu_count(logical=True),
            'per_core': core_usage
        },
        'memory': {
            'percent': mem.percent,
            'used_gb': mem.used/1024**3,
            'total_gb': mem.total/1024**3
        },
        'system': {
            'load': [load1, load5, load15],
            'estimated_power': estimated_power,
            'status': status
        }
    }
    return data

def write_tmp(path='/tmp/zion_system_stats.json'):
    stats = collect_stats()
    with open(path,'w') as f:
        json.dump(stats,f,indent=2)
    return stats

if __name__ == '__main__':
    while True:
        s = write_tmp()
        print(f"📊 {s['timestamp']} -> {s['cpu']['temperature']}°C {s['cpu']['usage']:.1f}% {s['cpu']['frequency']:.0f}MHz")
        time.sleep(5)
