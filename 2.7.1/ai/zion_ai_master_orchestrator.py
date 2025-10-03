#!/usr/bin/env python3
"""
ZION AI Master Orchestrator - Simple Version
"""

import sys
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ZionAIMasterOrchestrator:
    """Zjednodušený AI Master Orchestrator"""

    def __init__(self):
        self.components = {}
        self.active_components = set()
        logger.info("ZION AI Master Orchestrator initialized")

    def load_components(self):
        """Načte dostupné AI komponenty"""
        # Zkusit načíst základní komponenty
        component_paths = [
            ('zion_blockchain_analytics', 'ZionBlockchainAnalytics'),
            ('zion_security_monitor', 'ZionSecurityMonitor'),
            ('zion_trading_bot', 'ZionTradingBot'),
            ('zion_predictive_maintenance', 'ZionPredictiveMaintenance'),
            ('zion_gpu_miner', 'ZionGPUMiner')
        ]

        loaded_count = 0
        for module_name, class_name in component_paths:
            try:
                # Přidat cesty
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '2.7', 'ai'))
                sys.path.insert(0, os.path.dirname(__file__))

                module = __import__(module_name, fromlist=[class_name])
                component_class = getattr(module, class_name)
                instance = component_class()

                self.components[class_name] = {
                    'instance': instance,
                    'status': 'loaded',
                    'module': module_name
                }
                loaded_count += 1
                logger.info(f"Loaded component: {class_name}")

            except Exception as e:
                logger.debug(f"Failed to load {class_name}: {e}")

        logger.info(f"Total components loaded: {loaded_count}")
        return loaded_count

    def get_status(self):
        """Získá status systému"""
        return {
            'total_components': len(self.components),
            'active_components': len(self.active_components),
            'components': list(self.components.keys()),
            'timestamp': datetime.now().isoformat()
        }

    def run_diagnostics(self):
        """Spustí diagnostiku"""
        diagnostics = {
            'system_status': self.get_status(),
            'component_details': {}
        }

        for name, comp in self.components.items():
            diagnostics['component_details'][name] = {
                'status': comp['status'],
                'methods': [m for m in dir(comp['instance']) if not m.startswith('_')][:5]
            }

        return diagnostics

    def perform_sacred_mining(self, mining_data=None):
        """Provede sacred mining s AI podporou"""
        if mining_data is None:
            mining_data = {
                'block_hash': 'test_block_' + str(hash(str(datetime.now())))[:8],
                'mining_power': 50.0,
                'difficulty': 5000
            }

        # Aktivace AI komponent pro mining
        ai_contributions = []
        active_components = []
        gpu_mining_started = False

        for name, comp in self.components.items():
            try:
                if name == 'ZionBlockchainAnalytics':
                    result = comp['instance'].predict_price_trend(mining_data)
                    ai_contributions.append(result.get('confidence', 0.5) * 10)
                    active_components.append('analytics')
                elif name == 'ZionSecurityMonitor':
                    result = comp['instance'].analyze_security_threats(mining_data)
                    ai_contributions.append(5.0 if result.get('threat_level') == 'low' else 2.0)
                    active_components.append('security')
                elif name == 'ZionTradingBot':
                    result = comp['instance'].make_trading_decision(mining_data)
                    ai_contributions.append(result.get('confidence', 0.5) * 8)
                    active_components.append('trading')
                elif name == 'ZionPredictiveMaintenance':
                    # Maintenance komponenta - základní contribution
                    ai_contributions.append(3.0)
                    active_components.append('maintenance')
                elif name == 'ZionGPUMiner':
                    # GPU miner - spustit skutečné mining
                    gpu_miner = comp['instance']
                    if gpu_miner.gpu_available:
                        # Spustit GPU mining
                        mining_started = gpu_miner.start_mining(algorithm="kawpow", intensity=80)
                        if mining_started:
                            gpu_mining_started = True
                            # Počkat krátce na stabilizaci
                            import time
                            time.sleep(2)

                    # Získat GPU statistiky
                    gpu_stats = gpu_miner.get_mining_stats()
                    gpu_contribution = gpu_stats.get('hashrate', 0) * 0.1  # 10% of hashrate as contribution
                    ai_contributions.append(max(gpu_contribution, 5.0))  # Minimum 5.0
                    active_components.append('gpu_miner')

                    logger.info(f"GPU Mining: {gpu_stats.get('is_mining', False)}, Hashrate: {gpu_stats.get('hashrate', 0):.1f} MH/s")
            except Exception as e:
                logger.debug(f"AI contribution failed for {name}: {e}")
                ai_contributions.append(1.0)  # Minimální contribution

        # Výpočet AI boost
        total_ai_contribution = sum(ai_contributions) / max(len(ai_contributions), 1)
        base_power = mining_data.get('mining_power', 1.0)
        ai_boost = 1.0 + (total_ai_contribution / 100)
        enhanced_power = base_power * ai_boost

        # Sacred hash generation
        sacred_hash = self.generate_sacred_hash(mining_data, total_ai_contribution)

        result = {
            'block_hash': sacred_hash,
            'mining_power': enhanced_power,
            'ai_contribution': total_ai_contribution,
            'ai_boost': ai_boost,
            'cosmic_frequency': 432.0,  # Healing frequency
            'ai_components_used': active_components,
            'gpu_mining_active': gpu_mining_started,
            'divine_validation': self.perform_divine_validation(sacred_hash),
            'timestamp': datetime.now().isoformat()
        }

        # Zastavit GPU mining po dokončení (pokud bylo spuštěno)
        if gpu_mining_started and 'ZionGPUMiner' in self.components:
            try:
                self.components['ZionGPUMiner']['instance'].stop_mining()
                logger.info("GPU mining stopped after sacred mining completion")
            except Exception as e:
                logger.debug(f"Failed to stop GPU mining: {e}")

        logger.info(f"🌟 Sacred mining completed: AI boost {ai_boost:.2f}x, power {enhanced_power:.1f}, GPU active: {gpu_mining_started}")
        return result

    def generate_sacred_hash(self, mining_data, ai_contribution):
        """Generuje sacred hash s AI enhancement"""
        import hashlib
        base_data = f"{mining_data.get('block_hash', '')}_{ai_contribution}_{datetime.now().isoformat()}"
        return hashlib.sha256(base_data.encode()).hexdigest()

    def perform_divine_validation(self, hash_value):
        """Provede divine validation"""
        # Jednoduchá validation - kontrola zda hash končí na '0'
        return hash_value.endswith('0')

    def perform_unified_ai_analysis(self, data=None):
        """Provede sjednocenou AI analýzu pomocí všech komponent"""
        if data is None:
            data = {'market_data': 'test', 'blockchain_metrics': {'hashrate': 100}}

        analyses = {}
        total_confidence = 0
        analysis_count = 0

        for name, comp in self.components.items():
            try:
                if name == 'ZionBlockchainAnalytics':
                    result = comp['instance'].predict_price_trend(data)
                    analyses['price_analysis'] = result
                    total_confidence += result.get('confidence', 0)
                    analysis_count += 1
                elif name == 'ZionSecurityMonitor':
                    result = comp['instance'].analyze_security_threats(data)
                    analyses['security_analysis'] = result
                    threat_score = {'low': 0.8, 'medium': 0.5, 'high': 0.2}.get(result.get('threat_level', 'medium'), 0.5)
                    total_confidence += threat_score
                    analysis_count += 1
                elif name == 'ZionTradingBot':
                    result = comp['instance'].make_trading_decision(data)
                    analyses['trading_analysis'] = result
                    total_confidence += result.get('confidence', 0)
                    analysis_count += 1
                elif name == 'ZionPredictiveMaintenance':
                    # Maintenance analysis
                    analyses['maintenance_status'] = {'status': 'operational', 'confidence': 0.7}
                    total_confidence += 0.7
                    analysis_count += 1
            except Exception as e:
                logger.debug(f"Analysis failed for {name}: {e}")

        consensus_score = total_confidence / max(analysis_count, 1)

        return {
            'analyses': analyses,
            'consensus_score': consensus_score,
            'divine_validation': consensus_score > 0.6,
            'total_analyses': analysis_count,
            'timestamp': datetime.now().isoformat()
        }

    def get_resource_usage(self):
        """Získá aktuální využití zdrojů"""
        try:
            import psutil
            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory().percent
            return {
                'cpu_usage': cpu,
                'memory_usage': memory,
                'timestamp': datetime.now().isoformat()
            }
        except ImportError:
            return {
                'cpu_usage': 0,
                'memory_usage': 0,
                'error': 'psutil not available',
                'timestamp': datetime.now().isoformat()
            }

    def optimize_resources(self):
        """Optimalizuje využití zdrojů"""
        resources = self.get_resource_usage()
        optimizations = []

        if resources.get('cpu_usage', 0) > 80:
            optimizations.append("High CPU usage detected - consider reducing active components")
        if resources.get('memory_usage', 0) > 85:
            optimizations.append("High memory usage detected - consider component cleanup")

        if len(self.components) > 2:
            optimizations.append("Multiple components loaded - consider selective activation")

        return {
            'current_resources': resources,
            'optimizations': optimizations,
            'active_components': len(self.components),
            'timestamp': datetime.now().isoformat()
        }

def main():
    """Hlavní funkce"""
    print("🧠 ZION AI Master Orchestrator")
    print("=" * 40)

    orchestrator = ZionAIMasterOrchestrator()
    loaded = orchestrator.load_components()

    print(f"✅ Načteno komponent: {loaded}")

    status = orchestrator.get_status()
    print(f"📊 Status: {status['total_components']} celkem, {status['active_components']} aktivních")

    if status['components']:
        print("🔧 Komponenty:")
        for comp in status['components']:
            print(f"  - {comp}")

    # Test sacred mining
    print("\n⛏️ Test Sacred Mining:")
    mining_result = orchestrator.perform_sacred_mining()
    print(f"   Block Hash: {mining_result['block_hash'][:16]}...")
    print(f"   Mining Power: {mining_result['mining_power']:.1f}")
    print(f"   AI Contribution: {mining_result['ai_contribution']:.1f}%")
    print(f"   AI Boost: {mining_result['ai_boost']:.2f}x")
    print(f"   Divine Validation: {'✅' if mining_result['divine_validation'] else '❌'}")

    # Test unified analysis
    print("\n🔍 Unified AI Analysis:")
    analysis = orchestrator.perform_unified_ai_analysis()
    print(f"   Consensus Score: {analysis['consensus_score']:.2f}")
    print(f"   Divine Validation: {'✅' if analysis['divine_validation'] else '❌'}")
    print(f"   Total Analyses: {analysis['total_analyses']}")

    # Test resource management
    print("\n⚙️ Resource Management:")
    resources = orchestrator.get_resource_usage()
    print(f"   CPU Usage: {resources['cpu_usage']}%")
    print(f"   Memory Usage: {resources['memory_usage']}%")

    optimizations = orchestrator.optimize_resources()
    if optimizations.get('optimizations'):
        print("   Recommendations:")
        for opt in optimizations['optimizations']:
            print(f"    - {opt}")
    else:
        print("   No optimizations needed")

    print("\n✅ Orchestrator ready!")

if __name__ == "__main__":
    main()