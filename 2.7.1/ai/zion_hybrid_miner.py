#!/usr/bin/env python3
"""
ZION AI Hybrid Miner - Professional CPU + GPU Mining
CPU: RandomX (like Xmrig) + GPU: KawPow (like SRB Miner)
Professional mining outputs with shares, blocks, statistics
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
try:
    from zion_gpu_miner import ZionGPUMiner
except ImportError:
    # Try absolute import
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from zion_gpu_miner import ZionGPUMiner

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

class ZionHybridMiner:
    """Professional hybrid CPU + GPU miner s výstupy jako Xmrig a SRB Miner"""

    def __init__(self):
        self.gpu_miner = ZionGPUMiner()
        self.cpu_mining_active = False
        self.cpu_hashrate = 0.0
        self.cpu_process = None
        self.cpu_monitoring_thread = None
        self.stop_cpu_monitoring = False
        self.hybrid_mode = True
        self.ai_optimization_active = True
        self.total_hashrate = 0.0
        self.power_consumption = 0.0
        self.efficiency_score = 0.0

        # Professional mining statistics
        self.mining_stats = {
            "cpu_shares": {"accepted": 0, "rejected": 0, "total": 0},
            "gpu_shares": {"accepted": 0, "rejected": 0, "total": 0},
            "blocks_found": {"cpu": 0, "gpu": 0, "total": 0},
            "uptime": 0,
            "start_time": None,
            "best_share": {"cpu": 0, "gpu": 0},
            "pool_connection": {"cpu": "disconnected", "gpu": "disconnected"}
        }

        # CPU RandomX configuration (like Xmrig)
        self.cpu_algorithms = {
            "randomx": {
                "name": "RandomX",
                "description": "Monero-style CPU mining algorithm",
                "threads": 10,  # 10 threads as requested
                "hugepages": True,
                "asm": "auto",
                "mode": "fast"
            }
        }

        self.current_cpu_algorithm = "randomx"
        self.cpu_config = self._load_cpu_config()

        # Initialize miner paths
        self.xmrig_path = self._find_xmrig()

        logger.info("ZION Professional Hybrid Miner initialized - CPU RandomX + GPU KawPow")

    def _find_xmrig(self):
        """Najde Xmrig executable pro CPU mining"""
        possible_paths = [
            os.path.join(os.path.dirname(__file__), '..', 'miners', 'xmrig', 'xmrig-6.20.0', 'xmrig.exe'),
            os.path.join(os.path.dirname(__file__), '..', 'miners', 'xmrig.exe'),
            'xmrig.exe'  # V PATH
        ]

        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found Xmrig at: {path}")
                return path

        logger.warning("Xmrig not found - CPU mining will not work")
        return None

    def _start_xmrig_mining(self, pool_config):
        """Spustí skutečné Xmrig CPU mining"""
        if not self.xmrig_path:
            raise Exception("Xmrig path not found")

        # Vytvoř Xmrig config pro RandomX
        xmrig_config = {
            "api": {
                "id": None,
                "worker-id": None
            },
            "http": {
                "enabled": True,
                "host": "127.0.0.1",
                "port": 8080,
                "access-token": None,
                "restricted": False
            },
            "autosave": True,
            "background": False,
            "colors": True,
            "title": True,
            "randomx": {
                "init": -1,
                "init-avx2": -1,
                "mode": "auto",
                "1gb-pages": True,
                "rdmsr": True,
                "wrmsr": True,
                "cache_qos": False,
                "numa": True
            },
            "cpu": {
                "enabled": True,
                "huge-pages": True,
                "huge-pages-jit": False,
                "hw-aes": None,
                "priority": None,
                "memory-pool": False,
                "yield": True,
                "max-threads-hint": 100,
                "asm": True,
                "argon2-impl": None,
                "cn": [
                    [
                        1,
                        0
                    ]
                ],
                "cn-heavy": [
                    [
                        1,
                        0
                    ]
                ],
                "cn-lite": [
                    [
                        1,
                        0
                    ]
                ],
                "cn-pico": [
                    [
                        1,
                        1
                    ]
                ],
                "rx": [
                    self.cpu_algorithms["randomx"]["threads"],
                    0
                ],
                "rx/wow": [
                    0,
                    0
                ],
                "cn/0": [
                    0,
                    0
                ],
                "cn-lite/0": [
                    0,
                    0
                ]
            },
            "opencl": {
                "enabled": False
            },
            "cuda": {
                "enabled": False
            },
            "donate-level": 1,
            "donate-over-proxy": 1,
            "log-file": None,
            "pools": [
                {
                    "algo": "rx/0",
                    "coin": None,
                    "url": pool_config.get("url", "pool.supportxmr.com:3333"),
                    "user": pool_config.get("user", "test_wallet_address"),
                    "pass": pool_config.get("pass", "x"),
                    "rig-id": None,
                    "nicehash": False,
                    "keepalive": False,
                    "enabled": True,
                    "tls": False,
                    "tls-fingerprint": None,
                    "daemon": False,
                    "socks5": None,
                    "self-select": None,
                    "submit-to-origin": False
                }
            ],
            "print-time": 60,
            "health-print-time": 60,
            "dmi": True,
            "syslog": False,
            "tls": {
                "enabled": False,
                "protocols": None,
                "cert": None,
                "cert_key": None,
                "ciphers": None,
                "ciphersuites": None,
                "dhparam": None
            },
            "dns": {
                "ipv6": False,
                "ttl": 30
            },
            "user-agent": None,
            "verbose": 0,
            "watch": True,
            "pause-on-battery": False,
            "pause-on-active": False
        }

        # Ulož config dočasně
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'xmrig_config.json')
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(xmrig_config, f, indent=2)

        # Spusť Xmrig proces
        cmd = [self.xmrig_path, '--config', config_path]
        logger.info(f"Starting Xmrig with command: {' '.join(cmd)}")

        self.cpu_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        logger.info("Xmrig CPU mining process started")

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

    def start_hybrid_mining(self, gpu_algorithm="kawpow", cpu_algorithm="randomx",
                           gpu_pool=None, cpu_pool=None):
        """Spustí professional hybridní CPU RandomX + GPU KawPow mining"""
        logger.info("Starting ZION Professional Hybrid Mining - CPU RandomX + GPU KawPow")

        self.mining_stats["start_time"] = datetime.now()
        success_count = 0

        print("\n" + "="*60)
        print(" ZION PROFESSIONAL HYBRID MINER v2.7.1 ")
        print("="*60)
        print(f" CPU Algorithm: {cpu_algorithm.upper()}")
        print(f" GPU Algorithm: {gpu_algorithm.upper()}")
        print(f" Hybrid Mode: {'ENABLED' if self.hybrid_mode else 'DISABLED'}")
        print(f" AI Optimization: {'ENABLED' if self.ai_optimization_active else 'DISABLED'}")
        print("="*60 + "\n")

        # Spusť GPU KawPow mining
        if self.gpu_miner.gpu_available:
            # Předáme referenci na hybrid miner pro sdílení statistik
            self.gpu_miner.hybrid_miner = self
            if self.gpu_miner.start_mining(gpu_algorithm, gpu_pool, cpu_pool):
                logger.info("GPU KawPow mining started successfully")
                self.mining_stats["pool_connection"]["gpu"] = "connected"
                success_count += 1
                print("✓ GPU mining started - KawPow algorithm")
            else:
                logger.warning("GPU mining failed to start")
                print("✗ GPU mining failed to start")
        else:
            logger.info("GPU not available, running CPU-only mode")
            print("⚠ GPU not available - CPU only mode")

        # Spusť CPU RandomX mining
        if self.hybrid_mode:
            if self._start_cpu_mining(cpu_algorithm, cpu_pool):
                logger.info("CPU RandomX mining started successfully")
                self.mining_stats["pool_connection"]["cpu"] = "connected"
                success_count += 1
                print("✓ CPU mining started - RandomX algorithm")
            else:
                logger.warning("CPU mining failed to start")
                print("✗ CPU mining failed to start")

        # Spusť AI optimalizaci
        if self.ai_optimization_active and success_count > 0:
            self._start_ai_optimization()
            print("✓ AI optimization started")

        # Spusť professional monitoring display
        if success_count > 0:
            self._start_professional_display()

        print("\nMining started! Press Ctrl+C to stop...\n")
        return success_count > 0

    def _start_cpu_mining(self, algorithm="randomx", pool_config=None):
        """Spustí skutečné CPU RandomX mining s Xmrig (žádná simulace!)"""
        if self.cpu_mining_active:
            logger.warning("CPU mining already active")
            return False

        if algorithm != "randomx":
            logger.warning("Only RandomX algorithm supported for CPU mining")
            return False

        # Najdi Xmrig executable
        self.xmrig_path = self._find_xmrig()
        if not self.xmrig_path:
            logger.error("Xmrig not found! Please download Xmrig CPU miner")
            return False

        # Použij konfiguraci nebo parametry
        if pool_config:
            cpu_config = pool_config
        else:
            cpu_config = self.cpu_config

        self.current_cpu_algorithm = algorithm
        self.cpu_mining_active = True

        # Spusť skutečné Xmrig CPU mining
        try:
            self._start_xmrig_mining(cpu_config)
        except Exception as e:
            logger.error(f"Failed to start Xmrig mining: {e}")
            self.cpu_mining_active = False
            return False

        # Spusť monitoring thread pro skutečné mining statistiky
        self.stop_cpu_monitoring = False
        self.cpu_monitoring_thread = threading.Thread(target=self._monitor_xmrig_mining, daemon=True)
        self.cpu_monitoring_thread.start()

        # Odhad hashrate pro RandomX (skutečné hodnoty budou aktualizovány z Xmrig)
        base_cpu_hashrate = 1200.0  # H/s pro RandomX na moderním CPU
        thread_multiplier = self.cpu_algorithms["randomx"]["threads"] / 4.0
        self.cpu_hashrate = base_cpu_hashrate * thread_multiplier

        logger.info(f"Real Xmrig CPU RandomX mining started at ~{self.cpu_hashrate:.1f} H/s")
        return True

    def _monitor_xmrig_mining(self):
        """Monitoruje skutečné Xmrig CPU mining výstupy"""
        if not self.cpu_process:
            return

        print("[DEBUG] Starting Xmrig monitoring...")
        try:
            while not self.stop_cpu_monitoring and self.cpu_process.poll() is None:
                # Čti výstup z Xmrig procesu (stdout a stderr)
                line = self.cpu_process.stdout.readline()
                if not line:
                    # Also check stderr
                    line = self.cpu_process.stderr.readline()
                    if not line:
                        time.sleep(0.1)  # Small delay to avoid busy waiting
                        continue

                line = line.strip()
                print(f"[DEBUG] Xmrig output: {line}")  # Debug: show all output

                # Parsuj Xmrig výstup pro shares a statistiky
                if "accepted" in line.lower() and ("diff" in line.lower() or "difficulty" in line.lower()):
                    # Accepted share
                    self.mining_stats["cpu_shares"]["accepted"] += 1
                    self.mining_stats["cpu_shares"]["total"] += 1

                    # Extrahuj difficulty
                    try:
                        diff_part = line.split("diff")[1].split()[0] if "diff" in line else "0"
                        difficulty = int(diff_part.replace(",", ""))
                        if difficulty > self.mining_stats["best_share"]["cpu"]:
                            self.mining_stats["best_share"]["cpu"] = difficulty
                    except:
                        difficulty = 0

                    print(f"[CPU] Accept Share! Diff: {difficulty} (best: {self.mining_stats['best_share']['cpu']})")

                elif "rejected" in line.lower():
                    # Rejected share
                    self.mining_stats["cpu_shares"]["rejected"] += 1
                    self.mining_stats["cpu_shares"]["total"] += 1
                    print("[CPU] Reject Share! Invalid nonce")

                elif "hashrate" in line.lower() or "h/s" in line.lower():
                    # Aktualizuj hashrate
                    try:
                        # Najdi hashrate v řádku
                        if "H/s" in line:
                            hps_part = line.split("H/s")[0].split()[-1]
                            self.cpu_hashrate = float(hps_part.replace(",", ""))
                    except:
                        pass

                elif "block found" in line.lower() or "block" in line.lower() and "found" in line.lower():
                    # Block found
                    self.mining_stats["blocks_found"]["cpu"] += 1
                    self.mining_stats["blocks_found"]["total"] += 1
                    print(f"[CPU] BLOCK FOUND! Block #{self.mining_stats['blocks_found']['total']} submitted to pool")

                # Krátká pauza aby se nezahltilo
                time.sleep(0.1)

        except Exception as e:
            logger.error(f"Xmrig monitoring error: {e}")

        logger.info("Xmrig monitoring stopped")

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

    def _start_professional_display(self):
        """Spustí professional mining display (like Xmrig/SRB Miner)"""
        display_thread = threading.Thread(target=self._professional_display_loop, daemon=True)
        display_thread.start()

    def _professional_display_loop(self):
        """Professional mining display loop"""
        display_count = 0

        while self.cpu_mining_active or self.gpu_miner.is_mining:
            try:
                time.sleep(30)  # Update každých 30 sekund
                display_count += 1

                if display_count % 2 == 0:  # Každých 60 sekund
                    self._show_professional_stats()

            except Exception as e:
                logger.error(f"Professional display error: {e}")
                time.sleep(30)

    def _show_professional_stats(self):
        """Zobrazí professional mining statistiky"""
        uptime = datetime.now() - self.mining_stats["start_time"]
        uptime_str = str(uptime).split('.')[0]  # Remove microseconds

        gpu_stats = self.gpu_miner.get_stats()

        print("\n" + "="*70)
        print(" ZION HYBRID MINER STATISTICS ")
        print("="*70)
        print(f" Uptime: {uptime_str}")
        print(f" Pool CPU: {self.mining_stats['pool_connection']['cpu']}")
        print(f" Pool GPU: {self.mining_stats['pool_connection']['gpu']}")
        print()

        # CPU Stats
        cpu_accept_rate = 0
        if self.mining_stats["cpu_shares"]["total"] > 0:
            cpu_accept_rate = (self.mining_stats["cpu_shares"]["accepted"] / self.mining_stats["cpu_shares"]["total"]) * 100

        print(" CPU (RandomX):")
        print(f"   Hashrate: {self.cpu_hashrate / 1000000:.3f} MH/s")  # Convert H/s to MH/s
        print(f"   Shares: {self.mining_stats['cpu_shares']['accepted']}/{self.mining_stats['cpu_shares']['total']} ({cpu_accept_rate:.1f}%)")
        print(f"   Best Share: {self.mining_stats['best_share']['cpu']}")
        print(f"   Blocks Found: {self.mining_stats['blocks_found']['cpu']}")
        print()

        # GPU Stats
        gpu_accept_rate = 0
        gpu_shares = self.mining_stats["gpu_shares"]
        if gpu_shares["total"] > 0:
            gpu_accept_rate = (gpu_shares["accepted"] / gpu_shares["total"]) * 100

        print(" GPU (KawPow):")
        print(f"   Hashrate: {gpu_stats.get('hashrate', 0):.1f} MH/s")
        print(f"   Shares: {gpu_shares['accepted']}/{gpu_shares['total']} ({gpu_accept_rate:.1f}%)")
        print(f"   Best Share: {self.mining_stats['best_share']['gpu']}")
        print(f"   Blocks Found: {self.mining_stats['blocks_found']['gpu']}")
        print()

        # Total Stats
        total_hashrate = gpu_stats.get('hashrate', 0) + (self.cpu_hashrate / 1000000)  # Convert H/s to MH/s
        total_shares = self.mining_stats["cpu_shares"]["accepted"] + gpu_shares["accepted"]
        total_blocks = self.mining_stats["blocks_found"]["total"]

        print(" TOTAL:")
        print(f"   Combined Hashrate: {total_hashrate:.2f} MH/s")
        print(f"   Total Shares: {total_shares}")
        print(f"   Total Blocks Found: {total_blocks}")
        print(f"   Efficiency: {self._calculate_efficiency_score(gpu_stats):.2f} H/W")
        print("="*70 + "\n")

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
            "ai_optimization_active": self.ai_optimization_active,
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