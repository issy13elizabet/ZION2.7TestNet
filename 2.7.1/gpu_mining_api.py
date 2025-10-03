#!/usr/bin/env python3
"""
ZION GPU Mining API
REST API pro GPU mining dashboard a kontrolu
"""

import sys
import os
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

# Přidá AI složku do cesty
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ai'))

from zion_gpu_miner import ZionGPUMiner

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for dashboard

# Global GPU miner instance
gpu_miner = ZionGPUMiner()

@app.route('/api/mining/stats', methods=['GET'])
def get_mining_stats():
    """Získá aktuální mining statistiky"""
    try:
        stats = gpu_miner.get_mining_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting mining stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/mining/start', methods=['POST'])
def start_mining():
    """Spustí GPU mining"""
    try:
        data = request.get_json() or {}

        algorithm = data.get('algorithm', 'kawpow')
        intensity = data.get('intensity', 75)
        pool_config = data.get('pool_config')

        logger.info(f"Starting mining: algorithm={algorithm}, intensity={intensity}")

        success = gpu_miner.start_mining(
            algorithm=algorithm,
            intensity=intensity,
            pool_config=pool_config
        )

        if success:
            return jsonify({'status': 'success', 'message': 'Mining started'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to start mining'}), 500

    except Exception as e:
        logger.error(f"Error starting mining: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/mining/stop', methods=['POST'])
def stop_mining():
    """Zastaví GPU mining"""
    try:
        logger.info("Stopping mining")
        success = gpu_miner.stop_mining()

        if success:
            return jsonify({'status': 'success', 'message': 'Mining stopped'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to stop mining'}), 500

    except Exception as e:
        logger.error(f"Error stopping mining: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/mining/config', methods=['GET'])
def get_mining_config():
    """Získá mining konfiguraci"""
    try:
        config = {
            'mining_config': gpu_miner.mining_config,
            'supported_algorithms': gpu_miner.get_supported_algorithms(),
            'gpu_available': gpu_miner.gpu_available,
            'srbminer_available': gpu_miner.srbminer_path is not None,
            'benchmark_hashrate': gpu_miner.benchmark_hashrate,
            'gpu_type': gpu_miner._detect_gpu_type()
        }
        return jsonify(config)
    except Exception as e:
        logger.error(f"Error getting mining config: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/mining/config', methods=['POST'])
def update_mining_config():
    """Aktualizuje mining konfiguraci"""
    try:
        data = request.get_json() or {}

        if 'pool_config' in data:
            pool_config = data['pool_config']
            gpu_miner.configure_mining_pool(
                pool_host=pool_config.get('host', 'stratum.ravenminer.com'),
                pool_port=pool_config.get('port', 3838),
                wallet_address=pool_config.get('wallet', 'test_wallet'),
                password=pool_config.get('password', 'x')
            )

        return jsonify({'status': 'success', 'message': 'Configuration updated'})

    except Exception as e:
        logger.error(f"Error updating mining config: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/mining/algorithms', methods=['GET'])
def get_algorithms():
    """Získá podporované algoritmy"""
    try:
        algorithms = gpu_miner.get_supported_algorithms()
        return jsonify(algorithms)
    except Exception as e:
        logger.error(f"Error getting algorithms: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/mining/optimize', methods=['POST'])
def optimize_gpu():
    """Optimalizuje GPU nastavení"""
    try:
        optimization = gpu_miner.optimize_gpu_settings()
        return jsonify(optimization)
    except Exception as e:
        logger.error(f"Error optimizing GPU: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/mining/benchmark', methods=['POST'])
def run_benchmark():
    """Spustí benchmark algoritmů"""
    try:
        logger.info("Running algorithm benchmark")
        results = gpu_miner.benchmark_algorithms()
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error running benchmark: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/system/status', methods=['GET'])
def get_system_status():
    """Získá systémový status"""
    try:
        import psutil

        status = {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'gpu_miner_initialized': True,
            'mining_active': gpu_miner.is_mining,
            'timestamp': gpu_miner.get_mining_stats().get('timestamp')
        }

        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def dashboard():
    """Servíruje mining dashboard"""
    try:
        dashboard_path = os.path.join(os.path.dirname(__file__), 'gpu_mining_dashboard.html')
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error serving dashboard: {e}")
        return f"Error loading dashboard: {e}", 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'gpu_miner': 'initialized',
        'mining_active': gpu_miner.is_mining,
        'timestamp': gpu_miner.get_mining_stats().get('timestamp')
    })

if __name__ == '__main__':
    print("🚀 Starting ZION GPU Mining API Server")
    print("📊 Dashboard: http://localhost:5000")
    print("🔧 API endpoints available at /api/*")
    print("Press Ctrl+C to stop")

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # Disable debug mode for production
        threaded=True
    )