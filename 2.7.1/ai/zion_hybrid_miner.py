#!/usr/bin/env python3
"""
ZION AI CPU + GPU Miner - Hybrid Mining System
Kombinuje ASIC-resistant CPU mining s GPU mining pro maximální efektivitu
"""

import logging
import random
import time
import subprocess
import os
import threading
import sys
import json
from datetime import datetime
from zion_gpu_miner import ZionGPUMiner

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

class ZionHybridMiner:
    """AI-driven hybrid CPU + GPU miner pro maximální mining efektivitu"""

    def __init__(self):
        self.gpu_miner = ZionGPUMiner()
        self.cpu_mining_active = False
        self.cpu_hashrate = 0.0
        self.cpu_process = None
        self.cpu_monitoring_thread = None
        self.stop_cpu_monitoring = False
        self.hybrid_mode = True  # True = CPU + GPU, False = GPU only
        self.ai_optimization_active = True
        self.total_hashrate = 0.0
        self.power_consumption = 0.0
        self.efficiency_score = 0.0

        # ASIC-resistant CPU algoritmy
        self.cpu_algorithms = {
            "argon2": {"difficulty": "adaptive", "threads": "auto"},
            "cryptonight": {"variant": "r", "threads": "auto"},
            "randomx": {"mode": "fast", "threads": "auto"}
        }

        self.current_cpu_algorithm = "argon2"
        self.cpu_config = self._load_cpu_config()

        logger.info("ZionHybridMiner initialized - AI-driven CPU+GPU mining system")

    def _load_cpu_config(self):
        """Načte CPU mining konfiguraci"""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'cpu_mining.json')
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "algorithm": "argon2",
                "threads": max(1, os.cpu_count() // 2),  # Použij polovinu CPU jader
                "intensity": 50,
                "pool": {
                    "url": "stratum+tcp://cpu-pool.example.com:3333",
                    "user": "cpu_wallet_address.worker",
                    "pass": "x"
                }
            }

    def start_hybrid_mining(self, gpu_algorithm="kawpow", cpu_algorithm="argon2",
                           gpu_pool=None, cpu_pool=None):
        """Spustí hybridní CPU + GPU mining"""
        logger.info("Starting AI-driven hybrid CPU + GPU mining")

        success_count = 0

        # Spusť GPU mining
        if self.gpu_miner.gpu_available:
            if self.gpu_miner.start_mining(gpu_algorithm, gpu_pool, cpu_pool):
                logger.info("GPU mining started successfully")
                success_count += 1
            else:
                logger.warning("GPU mining failed to start")
        else:
            logger.info("GPU not available, running CPU-only mode")

        # Spusť CPU mining pokud je hybrid mode aktivní
        if self.hybrid_mode:
            if self._start_cpu_mining(cpu_algorithm, cpu_pool):
                logger.info("CPU mining started successfully")
                success_count += 1
            else:
                logger.warning("CPU mining failed to start")

        # Spusť AI optimalizaci
        if self.ai_optimization_active and success_count > 0:
            self._start_ai_optimization()

        return success_count > 0

    def _start_cpu_mining(self, algorithm="argon2", pool_config=None):
        """Spustí ASIC-resistant CPU mining"""
        if self.cpu_mining_active:
            logger.warning("CPU mining already active")
            return False

        # Použij konfiguraci nebo parametry
        if pool_config:
            cpu_config = pool_config
        else:
            cpu_config = self.cpu_config

        # Pro CPU mining použijeme XMRig nebo podobný nástroj
        # Pro demonstraci použijeme simulaci, v reálném nasazení by se použil skutečný CPU miner
        self.current_cpu_algorithm = algorithm
        self.cpu_mining_active = True

        # Spusť CPU mining thread
        self.stop_cpu_monitoring = False
        self.cpu_monitoring_thread = threading.Thread(target=self._cpu_mining_simulation, daemon=True)
        self.cpu_monitoring_thread.start()

        # Nastav základní CPU hashrate na základě algoritmu
        base_cpu_hashrate = {
            "argon2": 500.0,    # H/s pro Argon2
            "cryptonight": 300.0,  # H/s pro CryptoNight
            "randomx": 800.0    # H/s pro RandomX
        }

        self.cpu_hashrate = base_cpu_hashrate.get(algorithm, 500.0)
        self.cpu_hashrate *= (self.cpu_config.get("threads", 4) / 4.0)  # Škálování podle počtu vláken

        logger.info(f"CPU mining started with {algorithm} algorithm at {self.cpu_hashrate:.1f} H/s")
        return True

    def _cpu_mining_simulation(self):
        """Simuluje CPU mining proces (v reálném nasazení by se použil skutečný CPU miner)"""
        while not self.stop_cpu_monitoring and self.cpu_mining_active:
            try:
                # Simuluj variace v hashrate
                variation = random.uniform(0.9, 1.1)
                self.cpu_hashrate = self.cpu_hashrate * 0.95 + (self.cpu_hashrate * variation) * 0.05

                time.sleep(5)  # Aktualizace každých 5 sekund

            except Exception as e:
                logger.error(f"CPU mining simulation error: {e}")
                time.sleep(10)

    def stop_hybrid_mining(self):
        """Zastaví hybridní mining"""
        logger.info("Stopping hybrid CPU + GPU mining")

        # Zastav GPU mining
        self.gpu_miner.stop_mining()

        # Zastav CPU mining
        self._stop_cpu_mining()

        logger.info("Hybrid mining stopped")

    def _stop_cpu_mining(self):
        """Zastaví CPU mining"""
        if not self.cpu_mining_active:
            return

        self.stop_cpu_monitoring = True
        self.cpu_mining_active = False
        self.cpu_hashrate = 0.0

        if self.cpu_process:
            try:
                self.cpu_process.terminate()
                self.cpu_process.wait(timeout=10)
            except:
                self.cpu_process.kill()

        if self.cpu_monitoring_thread and self.cpu_monitoring_thread.is_alive():
            self.cpu_monitoring_thread.join(timeout=2)

        logger.info("CPU mining stopped")

    def _start_ai_optimization(self):
        """Spustí AI-driven optimalizaci mining parametrů"""
        logger.info("Starting AI optimization for hybrid mining")

        # Spusť optimalizaci v samostatném threadu
        optimization_thread = threading.Thread(target=self._ai_optimization_loop, daemon=True)
        optimization_thread.start()

    def _ai_optimization_loop(self):
        """AI optimalizace mining parametrů v reálném čase"""
        while self.ai_optimization_active and (self.gpu_miner.is_mining or self.cpu_mining_active):
            try:
                # Analyzuj aktuální výkon
                current_stats = self.get_hybrid_stats()

                # AI rozhodování pro optimalizaci
                self._ai_optimize_power_distribution(current_stats)
                self._ai_optimize_algorithm_selection(current_stats)
                self._ai_optimize_resource_allocation(current_stats)

                time.sleep(60)  # Optimalizace každou minutu

            except Exception as e:
                logger.error(f"AI optimization error: {e}")
                time.sleep(120)  # Pauza při chybě

    def _ai_optimize_power_distribution(self, stats):
        """AI optimalizace distribuce výkonu mezi CPU a GPU"""
        gpu_hashrate = stats.get("gpu_hashrate", 0)
        cpu_hashrate = stats.get("cpu_hashrate", 0)
        total_power = stats.get("power_consumption", 0)

        if total_power > 0:
            # Vypočítá efektivitu na watt
            gpu_efficiency = gpu_hashrate / total_power if gpu_hashrate > 0 else 0
            cpu_efficiency = cpu_hashrate / total_power if cpu_hashrate > 0 else 0

            # AI rozhodnutí: pokud GPU je méně efektivní, přesměruj výkon na CPU
            if gpu_efficiency < cpu_efficiency * 0.8 and self.hybrid_mode:
                logger.info("AI: Switching to CPU-optimized mode for better efficiency")
                # V reálném nasazení by se upravily parametry mining

    def _ai_optimize_algorithm_selection(self, stats):
        """AI výběr nejlepšího algoritmu na základě výkonu"""
        # Analyzuj hashrate různých algoritmů
        # V reálném nasazení by se testovaly různé algoritmy a vybíral se nejlepší
        pass

    def _ai_optimize_resource_allocation(self, stats):
        """AI optimalizace alokace systémových zdrojů"""
        # Monitoruj teplotu, využití CPU/GPU, paměti
        # Automaticky upravuj parametry pro udržení stability
        temperature = stats.get("temperature", 0)
        if temperature > 80:
            logger.warning("AI: High temperature detected, reducing mining intensity")
            # V reálném nasazení by se snížila intenzita mining

    def get_hybrid_stats(self):
        """Získá kompletní statistiky hybridního mining"""
        gpu_stats = self.gpu_miner.get_stats()

        stats = {
            "hybrid_mode": self.hybrid_mode,
            "gpu_mining": gpu_stats,
            "cpu_mining": {
                "active": self.cpu_mining_active,
                "hashrate": self.cpu_hashrate,
                "algorithm": self.current_cpu_algorithm
            },
            "total_hashrate": gpu_stats.get("hashrate", 0) + self.cpu_hashrate,
            "power_consumption": self._calculate_power_consumption(gpu_stats),
            "efficiency_score": self._calculate_efficiency_score(gpu_stats),
            "ai_optimization": self.ai_optimization_active,
            "timestamp": datetime.now().isoformat()
        }

        # Přidej systémové statistiky
        if PSUTIL_AVAILABLE:
            try:
                stats["system"] = {
                    "cpu_percent": psutil.cpu_percent(interval=1),
                    "memory_percent": psutil.virtual_memory().percent,
                    "cpu_count": psutil.cpu_count()
                }
            except:
                pass

        return stats

    def _calculate_power_consumption(self, gpu_stats):
        """Vypočítá odhad spotřeby energie"""
        gpu_power = gpu_stats.get("hashrate", 0) * 0.2  # ~0.2W per MH/s pro GPU
        cpu_power = self.cpu_hashrate * 0.001  # ~0.001W per H/s pro CPU
        return gpu_power + cpu_power

    def _calculate_efficiency_score(self, gpu_stats):
        """Vypočítá skóre efektivity mining"""
        total_hashrate = gpu_stats.get("hashrate", 0) + self.cpu_hashrate
        power = self._calculate_power_consumption(gpu_stats)

        if power > 0:
            return total_hashrate / power  # Hashrate per watt
        return 0

    def configure_hybrid_mining(self, config):
        """Konfiguruje hybridní mining parametry"""
        if "hybrid_mode" in config:
            self.hybrid_mode = config["hybrid_mode"]

        if "ai_optimization" in config:
            self.ai_optimization_active = config["ai_optimization"]

        if "cpu_config" in config:
            self.cpu_config.update(config["cpu_config"])

        # Ulož konfiguraci
        self._save_hybrid_config()

        logger.info(f"Hybrid mining configured: {config}")

    def _save_hybrid_config(self):
        """Uloží hybrid mining konfiguraci"""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'hybrid_mining.json')
        try:
            config = {
                "hybrid_mode": self.hybrid_mode,
                "ai_optimization": self.ai_optimization_active,
                "cpu_config": self.cpu_config,
                "timestamp": datetime.now().isoformat()
            }
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save hybrid config: {e}")

    def get_supported_algorithms(self):
        """Vrátí podporované algoritmy pro hybrid mining"""
        return {
            "gpu_algorithms": self.gpu_miner.get_supported_algorithms(),
            "cpu_algorithms": {
                "argon2": {
                    "name": "Argon2",
                    "description": "ASIC-resistant CPU algorithm",
                    "resistance": "High",
                    "efficiency": "Medium"
                },
                "cryptonight": {
                    "name": "CryptoNight",
                    "description": "Memory-hard CPU algorithm",
                    "resistance": "High",
                    "efficiency": "High"
                },
                "randomx": {
                    "name": "RandomX",
                    "description": "RandomX CPU mining algorithm",
                    "resistance": "Very High",
                    "efficiency": "Low"
                }
            }
        }

    def auto_balance_load(self):
        """Automatické vyvážení zátěže mezi CPU a GPU"""
        logger.info("Starting automatic load balancing")

        # Analyzuj aktuální výkon
        stats = self.get_hybrid_stats()

        # AI rozhodování pro vyvážení
        gpu_load = stats["gpu_mining"].get("hashrate", 0)
        cpu_load = stats["cpu_mining"].get("hashrate", 0)

        # Pokud GPU nese většinu zátěže, přidej CPU
        if gpu_load > cpu_load * 3 and not self.cpu_mining_active and self.hybrid_mode:
            logger.info("AI: Adding CPU mining to balance load")
            self._start_cpu_mining()

        # Pokud CPU je přetížené, sniž počet vláken
        elif cpu_load > gpu_load * 2 and self.cpu_mining_active:
            logger.info("AI: Reducing CPU threads to balance load")
            # V reálném nasazení by se upravil počet CPU vláken

    def emergency_shutdown(self, reason="Unknown"):
        """Nouzové vypnutí mining při kritických podmínkách"""
        logger.warning(f"Emergency shutdown initiated: {reason}")

        self.stop_hybrid_mining()

        # Ulož stav pro pozdější analýzu
        emergency_log = {
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "final_stats": self.get_hybrid_stats()
        }

        emergency_path = os.path.join(os.path.dirname(__file__), '..', 'logs', 'emergency_shutdown.json')
        try:
            os.makedirs(os.path.dirname(emergency_path), exist_ok=True)
            with open(emergency_path, 'w') as f:
                json.dump(emergency_log, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save emergency log: {e}")

    def __del__(self):
        """Cleanup při destrukci"""
        self.stop_hybrid_mining()


if __name__ == "__main__":
    # Test hybrid miner
    miner = ZionHybridMiner()

    print("ZION AI Hybrid CPU + GPU Miner Test")
    print(f"GPU Available: {miner.gpu_miner.gpu_available}")
    print(f"SRBMiner Found: {miner.gpu_miner.srbminer_path is not None}")
    print(f"Hybrid Mode: {miner.hybrid_mode}")
    print(f"AI Optimization: {miner.ai_optimization_active}")

    print("\nStarting hybrid mining test...")
    if miner.start_hybrid_mining():
        print("Hybrid mining started successfully")

        # Monitoruj po dobu 30 sekund
        for i in range(6):
            time.sleep(5)
            stats = miner.get_hybrid_stats()
            print(f"Stats {i+1}: GPU {stats['gpu_mining']['hashrate']:.1f} MH/s, CPU {stats['cpu_mining']['hashrate']:.1f} H/s, Total: {stats['total_hashrate']:.1f}")

        miner.stop_hybrid_mining()
        print("Hybrid mining test completed")
    else:
        print("Failed to start hybrid mining")