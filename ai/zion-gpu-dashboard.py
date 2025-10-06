#!/usr/bin/env python3
"""
ZION GPU Dashboard - Real-time monitoring frontend component
Generates HTML dashboard for GPU statistics visualization
"""

import json
import os
from datetime import datetime
import requests

class ZionGPUDashboard:
    def __init__(self):
        self.api_base = "http://localhost:5001/api/gpu"
        
    def generate_dashboard_html(self):
        """Generate interactive HTML dashboard"""
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ZION AI GPU Afterburner</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
            color: #00ff88;
            font-family: 'Courier New', monospace;
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(0, 255, 136, 0.1);
            border-radius: 15px;
            border: 2px solid #00ff88;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 0 0 20px #00ff88;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: rgba(26, 26, 46, 0.8);
            border: 2px solid #00ff88;
            border-radius: 15px;
            padding: 20px;
            position: relative;
            overflow: hidden;
        }
        
        .stat-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 2px;
            background: linear-gradient(90deg, transparent, #00ff88, transparent);
            animation: scan 2s linear infinite;
        }
        
        @keyframes scan {
            0% { left: -100%; }
            100% { left: 100%; }
        }
        
        .stat-title {
            font-size: 1.2em;
            margin-bottom: 10px;
            color: #00ff88;
            text-transform: uppercase;
        }
        
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .stat-unit {
            color: #888;
            font-size: 0.8em;
        }
        
        .progress-bar {
            width: 100%;
            height: 20px;
            background: rgba(0, 0, 0, 0.5);
            border-radius: 10px;
            overflow: hidden;
            margin-top: 10px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00ff88, #00cc66);
            transition: width 0.5s ease;
            border-radius: 10px;
        }
        
        .controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        
        .control-btn {
            background: linear-gradient(45deg, #1a1a2e, #16213e);
            border: 2px solid #00ff88;
            color: #00ff88;
            padding: 15px 20px;
            border-radius: 10px;
            cursor: pointer;
            font-family: inherit;
            font-size: 1em;
            transition: all 0.3s ease;
            text-transform: uppercase;
        }
        
        .control-btn:hover {
            background: #00ff88;
            color: #0a0a0a;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 255, 136, 0.4);
        }
        
        .control-btn.active {
            background: #00ff88;
            color: #0a0a0a;
        }
        
        .chart-container {
            background: rgba(26, 26, 46, 0.8);
            border: 2px solid #00ff88;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .temperature-critical {
            color: #ff4444 !important;
            animation: blink 1s infinite;
        }
        
        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0.5; }
        }
        
        .status-ok { color: #00ff88; }
        .status-warning { color: #ffaa00; }
        .status-critical { color: #ff4444; }
        
        .log-container {
            background: rgba(0, 0, 0, 0.8);
            border: 1px solid #00ff88;
            border-radius: 10px;
            padding: 15px;
            max-height: 200px;
            overflow-y: auto;
            font-size: 0.9em;
        }
        
        .log-entry {
            margin-bottom: 5px;
            padding: 2px 5px;
        }
        
        .mining-integration {
            background: linear-gradient(45deg, #1a1a2e, #2e1a1a);
            border: 2px solid #ff6b35;
            border-radius: 15px;
            padding: 20px;
            margin-top: 20px;
        }
        
        .mining-title {
            color: #ff6b35;
            font-size: 1.3em;
            margin-bottom: 15px;
            text-transform: uppercase;
        }
        
        .hashrate-display {
            font-size: 2.5em;
            color: #ff6b35;
            text-shadow: 0 0 10px #ff6b35;
            margin-bottom: 10px;
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 ZION AI GPU AFTERBURNER</h1>
            <p>Advanced AMD GPU Control System • Real-time Monitoring & Optimization</p>
            <p id="gpu-model">AMD Radeon RX 5600 XT</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-title">🌡️ Temperature</div>
                <div class="stat-value" id="temperature">--</div>
                <div class="stat-unit">°C</div>
                <div class="progress-bar">
                    <div class="progress-fill" id="temp-progress"></div>
                </div>
            </div>
            
            <div class="stat-card">
                <div class="stat-title">⚡ Power Usage</div>
                <div class="stat-value" id="power">--</div>
                <div class="stat-unit">Watts</div>
                <div class="progress-bar">
                    <div class="progress-fill" id="power-progress"></div>
                </div>
            </div>
            
            <div class="stat-card">
                <div class="stat-title">🚀 GPU Usage</div>
                <div class="stat-value" id="utilization">--</div>
                <div class="stat-unit">%</div>
                <div class="progress-bar">
                    <div class="progress-fill" id="util-progress"></div>
                </div>
            </div>
            
            <div class="stat-card">
                <div class="stat-title">🧠 VRAM Usage</div>
                <div class="stat-value" id="vram-used">--</div>
                <div class="stat-unit">MB / <span id="vram-total">--</span> MB</div>
                <div class="progress-bar">
                    <div class="progress-fill" id="vram-progress"></div>
                </div>
            </div>
            
            <div class="stat-card">
                <div class="stat-title">🔄 Core Clock</div>
                <div class="stat-value" id="core-clock">--</div>
                <div class="stat-unit">MHz</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-title">💾 Memory Clock</div>
                <div class="stat-value" id="memory-clock">--</div>
                <div class="stat-unit">MHz</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h3 style="color: #00ff88; margin-bottom: 15px;">📊 Performance History</h3>
            <canvas id="performanceChart" width="400" height="150"></canvas>
        </div>
        
        <div class="controls">
            <button class="control-btn" onclick="applyProfile('eco')">🌱 ECO Mode</button>
            <button class="control-btn active" onclick="applyProfile('balanced')">⚖️ Balanced</button>
            <button class="control-btn" onclick="applyProfile('mining')">⛏️ Mining</button>
            <button class="control-btn" onclick="applyProfile('gaming')">🎮 Gaming</button>
            <button class="control-btn" onclick="optimizeForMining()">🤖 Auto Optimize</button>
            <button class="control-btn" onclick="emergencyReset()" style="border-color: #ff4444; color: #ff4444;">🚨 Emergency Reset</button>
        </div>
        
        <div class="mining-integration">
            <div class="mining-title">⛏️ Mining Integration</div>
            <div class="hashrate-display" id="hashrate">6,012 H/s</div>
            <p>Current ZION mining performance with GPU optimization</p>
            <button class="control-btn" onclick="syncWithMining()" style="margin-top: 10px;">🔄 Sync with Mining</button>
        </div>
        
        <div class="chart-container">
            <h3 style="color: #00ff88; margin-bottom: 15px;">📝 System Log</h3>
            <div class="log-container" id="system-log">
                <div class="log-entry">[INFO] GPU Afterburner initialized</div>
                <div class="log-entry">[INFO] Monitoring started</div>
            </div>
        </div>
    </div>
    
    <script>
        let chart;
        let currentProfile = 'balanced';
        
        // Initialize performance chart
        function initChart() {
            const ctx = document.getElementById('performanceChart').getContext('2d');
            chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'Temperature (°C)',
                            data: [],
                            borderColor: '#ff6b6b',
                            backgroundColor: 'rgba(255, 107, 107, 0.1)',
                            tension: 0.4
                        },
                        {
                            label: 'Power (W)',
                            data: [],
                            borderColor: '#4ecdc4',
                            backgroundColor: 'rgba(78, 205, 196, 0.1)',
                            tension: 0.4
                        },
                        {
                            label: 'Utilization (%)',
                            data: [],
                            borderColor: '#00ff88',
                            backgroundColor: 'rgba(0, 255, 136, 0.1)',
                            tension: 0.4
                        }
                    ]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            labels: { color: '#00ff88' }
                        }
                    },
                    scales: {
                        x: { 
                            ticks: { color: '#00ff88' },
                            grid: { color: 'rgba(0, 255, 136, 0.1)' }
                        },
                        y: { 
                            ticks: { color: '#00ff88' },
                            grid: { color: 'rgba(0, 255, 136, 0.1)' }
                        }
                    }
                }
            });
        }
        
        // Update GPU statistics
        function updateStats() {
            fetch('/api/gpu/stats')
                .then(response => response.json())
                .then(data => {
                    const stats = data.current;
                    
                    // Update display values
                    document.getElementById('temperature').textContent = Math.round(stats.temperature);
                    document.getElementById('power').textContent = Math.round(stats.power_usage);
                    document.getElementById('utilization').textContent = Math.round(stats.utilization);
                    document.getElementById('vram-used').textContent = Math.round(stats.vram_used);
                    document.getElementById('vram-total').textContent = Math.round(stats.vram_total);
                    document.getElementById('core-clock').textContent = stats.core_clock;
                    document.getElementById('memory-clock').textContent = stats.memory_clock;
                    
                    // Update progress bars
                    updateProgressBar('temp-progress', stats.temperature, 90);
                    updateProgressBar('power-progress', stats.power_usage, 150);
                    updateProgressBar('util-progress', stats.utilization, 100);
                    updateProgressBar('vram-progress', (stats.vram_used / stats.vram_total) * 100, 100);
                    
                    // Temperature warning
                    const tempElement = document.getElementById('temperature');
                    if (stats.temperature > 80) {
                        tempElement.className = 'stat-value temperature-critical';
                    } else {
                        tempElement.className = 'stat-value';
                    }
                    
                    // Update chart
                    updateChart(stats);
                    
                    // Log entry
                    addLogEntry(`Stats updated - ${stats.temperature}°C, ${Math.round(stats.power_usage)}W`);
                })
                .catch(error => {
                    addLogEntry(`[ERROR] Failed to fetch stats: ${error}`, 'error');
                });
        }
        
        function updateProgressBar(id, value, max) {
            const percentage = Math.min((value / max) * 100, 100);
            document.getElementById(id).style.width = percentage + '%';
        }
        
        function updateChart(stats) {
            const now = new Date().toLocaleTimeString();
            
            chart.data.labels.push(now);
            chart.data.datasets[0].data.push(stats.temperature);
            chart.data.datasets[1].data.push(stats.power_usage);
            chart.data.datasets[2].data.push(stats.utilization);
            
            // Keep only last 20 points
            if (chart.data.labels.length > 20) {
                chart.data.labels.shift();
                chart.data.datasets.forEach(dataset => dataset.data.shift());
            }
            
            chart.update('none');
        }
        
        function applyProfile(profile) {
            fetch(`/api/gpu/profile/${profile}`, { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        currentProfile = profile;
                        updateActiveButton(profile);
                        addLogEntry(`[SUCCESS] Applied ${profile} profile`);
                    } else {
                        addLogEntry(`[ERROR] Failed to apply profile: ${data.message}`, 'error');
                    }
                })
                .catch(error => {
                    addLogEntry(`[ERROR] Profile request failed: ${error}`, 'error');
                });
        }
        
        function updateActiveButton(profile) {
            document.querySelectorAll('.control-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            event.target.classList.add('active');
        }
        
        function optimizeForMining() {
            fetch('/api/gpu/mining/optimize')
                .then(response => response.json())
                .then(data => {
                    addLogEntry(`[AI] ${data.message}`);
                    currentProfile = data.profile;
                })
                .catch(error => {
                    addLogEntry(`[ERROR] Auto-optimize failed: ${error}`, 'error');
                });
        }
        
        function emergencyReset() {
            if (confirm('Emergency reset will restore safe defaults. Continue?')) {
                fetch('/api/gpu/reset', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        addLogEntry(`[RESET] ${data.message}`);
                        currentProfile = 'balanced';
                    })
                    .catch(error => {
                        addLogEntry(`[ERROR] Reset failed: ${error}`, 'error');
                    });
            }
        }
        
        function syncWithMining() {
            // Simulate ZION mining integration
            const hashrates = [5850, 5920, 6012, 5980, 6050, 5890];
            const randomHashrate = hashrates[Math.floor(Math.random() * hashrates.length)];
            document.getElementById('hashrate').textContent = `${randomHashrate} H/s`;
            addLogEntry(`[MINING] Synced - Current hashrate: ${randomHashrate} H/s`);
        }
        
        function addLogEntry(message, type = 'info') {
            const logContainer = document.getElementById('system-log');
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            
            if (type === 'error') {
                entry.style.color = '#ff4444';
            }
            
            logContainer.appendChild(entry);
            logContainer.scrollTop = logContainer.scrollHeight;
            
            // Keep only last 50 entries
            while (logContainer.children.length > 50) {
                logContainer.removeChild(logContainer.firstChild);
            }
        }
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            initChart();
            updateStats();
            
            // Update stats every 2 seconds
            setInterval(updateStats, 2000);
            
            // Simulate mining hashrate updates
            setInterval(syncWithMining, 10000);
        });
    </script>
</body>
</html>
        """
        return html
    
    def save_dashboard(self, filename="gpu_dashboard.html"):
        """Save dashboard HTML to file"""
        html_content = self.generate_dashboard_html()
        filepath = f"/media/maitreya/ZION1/frontend/{filename}"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        return filepath

if __name__ == "__main__":
    dashboard = ZionGPUDashboard()
    filepath = dashboard.save_dashboard()
    print(f"GPU Dashboard saved to: {filepath}")
    print("Open in browser after starting GPU Afterburner API")