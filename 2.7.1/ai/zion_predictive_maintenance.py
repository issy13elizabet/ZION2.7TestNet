#!/usr/bin/env python3
"""
ZION AI Predictive Maintenance - Predikce poruch hardware
AI poháněná prediktivní údržba pro mining hardware a prevence výpadků
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import psutil
import GPUtil
import requests
import json
import time
from datetime import datetime, timedelta
import threading
import logging
from typing import Dict, List, Optional, Tuple
import os
from collections import deque
import platform
import subprocess

# TensorFlow import s fallback
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow není dostupný - některé AI funkce budou omezené")

logger = logging.getLogger(__name__)

class ZionPredictiveMaintenance:
    """AI poháněná prediktivní údržba pro mining hardware"""

    def __init__(self, monitoring_interval: int = 60):
        self.monitoring_interval = monitoring_interval  # Sekundy

        # AI modely
        self.failure_predictor = None
        self.anomaly_detector = None
        self.scaler = StandardScaler()

        # Hardware monitoring
        self.system_history = deque(maxlen=1000)  # Historie systémových metrik
        self.gpu_history = deque(maxlen=1000)     # Historie GPU metrik
        self.cpu_history = deque(maxlen=1000)     # Historie CPU metrik
        self.memory_history = deque(maxlen=1000)  # Historie paměti
        self.disk_history = deque(maxlen=1000)    # Historie disku

        # Predikce a alerty
        self.predicted_failures = []
        self.active_alerts = set()
        self.maintenance_schedule = []

        # Prahové hodnoty
        self.critical_temp_cpu = 85  # °C
        self.critical_temp_gpu = 80  # °C
        self.critical_memory_usage = 95  # %
        self.critical_disk_usage = 95  # %
        self.failure_prediction_days = 7  # Kolik dní dopředu predikovat

        # Konfidenční prahy
        self.alert_confidence_threshold = 0.75
        self.failure_prediction_threshold = 0.85

        # Načtení nebo vytvoření modelů
        self.load_or_create_models()

    def load_or_create_models(self):
        """Načte existující AI modely nebo vytvoří nové"""
        model_dir = os.path.dirname(__file__)

        # Failure predictor
        failure_model_path = os.path.join(model_dir, 'zion_failure_predictor.pkl')
        if os.path.exists(failure_model_path):
            try:
                import joblib
                self.failure_predictor = joblib.load(failure_model_path)
                logger.info("✅ Načten AI model pro predikci poruch")
            except Exception as e:
                logger.warning(f"❌ Chyba při načítání failure modelu: {e}")
                self.create_failure_predictor()
        else:
            self.create_failure_predictor()

        # Anomaly detector
        anomaly_model_path = os.path.join(model_dir, 'zion_anomaly_detector.pkl')
        if os.path.exists(anomaly_model_path):
            try:
                import joblib
                self.anomaly_detector = joblib.load(anomaly_model_path)
                logger.info("✅ Načten AI model pro detekci anomálií")
            except Exception as e:
                logger.warning(f"❌ Chyba při načítání anomaly modelu: {e}")
                self.create_anomaly_detector()
        else:
            self.create_anomaly_detector()

    def create_failure_predictor(self):
        """Vytvoří model pro predikci poruch hardware"""
        logger.info("🔧 Vytváření AI modelu pro predikci poruch hardware...")

        try:
            from sklearn.ensemble import RandomForestClassifier
            self.failure_predictor = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=42
            )
            logger.info("✅ Failure predictor vytvořen")
        except ImportError:
            logger.warning("❌ sklearn není dostupný - predikce poruch bude omezená")

    def create_anomaly_detector(self):
        """Vytvoří model pro detekci anomálií"""
        logger.info("🔍 Vytváření AI modelu pro detekci hardwarových anomálií...")

        try:
            from sklearn.ensemble import IsolationForest
            self.anomaly_detector = IsolationForest(
                n_estimators=100,
                contamination=0.1,  # Očekáváme 10% anomálií
                random_state=42
            )
            logger.info("✅ Anomaly detector vytvořen")
        except ImportError:
            logger.warning("❌ sklearn není dostupný - detekce anomálií bude omezená")

    def collect_system_metrics(self) -> Dict:
        """Sbírá systémové metriky pro analýzu"""
        try:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu': {},
                'memory': {},
                'disk': {},
                'gpu': [],
                'network': {},
                'system': {}
            }

            # CPU metriky
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_freq = psutil.cpu_freq()
            cpu_temp = self.get_cpu_temperature()

            metrics['cpu'] = {
                'usage_percent': cpu_percent,
                'frequency_mhz': cpu_freq.current if cpu_freq else 0,
                'temperature_c': cpu_temp,
                'core_count': psutil.cpu_count(),
                'logical_count': psutil.cpu_count(logical=True)
            }

            # Memory metriky
            memory = psutil.virtual_memory()
            metrics['memory'] = {
                'total_gb': memory.total / (1024**3),
                'used_gb': memory.used / (1024**3),
                'available_gb': memory.available / (1024**3),
                'usage_percent': memory.percent,
                'swap_total_gb': psutil.swap_memory().total / (1024**3),
                'swap_used_gb': psutil.swap_memory().used / (1024**3)
            }

            # Disk metriky
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()

            metrics['disk'] = {
                'total_gb': disk.total / (1024**3),
                'used_gb': disk.used / (1024**3),
                'free_gb': disk.free / (1024**3),
                'usage_percent': disk.percent,
                'read_mb': disk_io.read_bytes / (1024**2) if disk_io else 0,
                'write_mb': disk_io.write_bytes / (1024**2) if disk_io else 0
            }

            # GPU metriky
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    gpu_metrics = {
                        'id': gpu.id,
                        'name': gpu.name,
                        'usage_percent': gpu.load * 100,
                        'memory_used_mb': gpu.memoryUsed,
                        'memory_total_mb': gpu.memoryTotal,
                        'memory_usage_percent': gpu.memoryUtil * 100,
                        'temperature_c': gpu.temperature
                    }
                    metrics['gpu'].append(gpu_metrics)
            except:
                # Fallback bez GPU informací
                metrics['gpu'] = []

            # Network metriky
            network = psutil.net_io_counters()
            metrics['network'] = {
                'bytes_sent_mb': network.bytes_sent / (1024**2),
                'bytes_recv_mb': network.bytes_recv / (1024**2),
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv,
                'errors_in': network.errin,
                'errors_out': network.errout
            }

            # System informace
            metrics['system'] = {
                'os': platform.system(),
                'os_version': platform.version(),
                'uptime_seconds': time.time() - psutil.boot_time(),
                'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            }

            return metrics

        except Exception as e:
            logger.error(f"❌ Chyba při sběru systémových metrik: {e}")
            return {}

    def get_cpu_temperature(self) -> float:
        """Získá teplotu CPU"""
        try:
            # Linux
            if platform.system() == "Linux":
                try:
                    result = subprocess.run(['cat', '/sys/class/thermal/thermal_zone0/temp'],
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        return float(result.stdout.strip()) / 1000
                except:
                    pass

                # Alternativa - sensors
                try:
                    result = subprocess.run(['sensors'], capture_output=True, text=True)
                    if result.returncode == 0:
                        lines = result.stdout.split('\n')
                        for line in lines:
                            if 'Core 0:' in line or 'Tctl:' in line:
                                temp_str = line.split(':')[1].split('°')[0].strip()
                                return float(temp_str)
                except:
                    pass

            # Windows
            elif platform.system() == "Windows":
                try:
                    from ctypes import windll, c_int, byref
                    dll = windll.LoadLibrary('C:\\Windows\\System32\\sensapi.dll')
                    temp = c_int()
                    dll.GetCpuTemperature(byref(temp))
                    return temp.value
                except:
                    pass

            # macOS
            elif platform.system() == "Darwin":
                try:
                    result = subprocess.run(['sysctl', '-n', 'machdep.xcpm.cpu_thermal_level'],
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        return float(result.stdout.strip())
                except:
                    pass

        except Exception as e:
            logger.debug(f"Nelze získat teplotu CPU: {e}")

        return 0.0  # Výchozí hodnota

    def extract_maintenance_features(self, metrics: Dict) -> List[float]:
        """Extrahuje features pro prediktivní údržbu"""
        features = []

        try:
            # CPU features
            cpu = metrics.get('cpu', {})
            features.extend([
                cpu.get('usage_percent', 0),
                cpu.get('frequency_mhz', 0),
                cpu.get('temperature_c', 0),
                cpu.get('core_count', 0),
            ])

            # Memory features
            memory = metrics.get('memory', {})
            features.extend([
                memory.get('usage_percent', 0),
                memory.get('swap_used_gb', 0) / max(memory.get('swap_total_gb', 1), 1),  # Normalizovaný swap
            ])

            # Disk features
            disk = metrics.get('disk', {})
            features.extend([
                disk.get('usage_percent', 0),
                disk.get('read_mb', 0),
                disk.get('write_mb', 0),
            ])

            # GPU features (průměr přes všechny GPU)
            gpus = metrics.get('gpu', [])
            if gpus:
                avg_gpu_usage = np.mean([gpu.get('usage_percent', 0) for gpu in gpus])
                avg_gpu_temp = np.mean([gpu.get('temperature_c', 0) for gpu in gpus])
                avg_gpu_mem = np.mean([gpu.get('memory_usage_percent', 0) for gpu in gpus])
                features.extend([avg_gpu_usage, avg_gpu_temp, avg_gpu_mem])
            else:
                features.extend([0, 0, 0])  # Žádné GPU

            # Network features
            network = metrics.get('network', {})
            features.extend([
                network.get('errors_in', 0),
                network.get('errors_out', 0),
            ])

            # System features
            system = metrics.get('system', {})
            uptime_hours = system.get('uptime_seconds', 0) / 3600
            load_avg = system.get('load_average', [0, 0, 0])[0]
            features.extend([uptime_hours, load_avg])

            # Časové features
            now = datetime.now()
            features.extend([
                now.hour,      # Hodina dne
                now.weekday(), # Den v týdnu
            ])

            # Trend features (pokud máme historii)
            if len(self.system_history) >= 10:
                recent_cpu = [h.get('cpu', {}).get('usage_percent', 0) for h in list(self.system_history)[-10:]]
                cpu_trend = np.polyfit(range(len(recent_cpu)), recent_cpu, 1)[0]  # Lineární trend
                features.append(cpu_trend)
            else:
                features.append(0)

        except Exception as e:
            logger.warning(f"❌ Chyba při extrakci maintenance features: {e}")
            features = [0] * 20  # Výchozí features

        return features

    def detect_anomalies(self, metrics: Dict) -> List[Dict]:
        """Detekuje anomální chování hardware"""
        anomalies = []

        if not metrics or self.anomaly_detector is None:
            return anomalies

        try:
            # Extrakce features
            features = self.extract_maintenance_features(metrics)
            features_array = np.array([features])

            # Detekce anomálií
            predictions = self.anomaly_detector.predict(features_array)
            scores = self.anomaly_detector.score_samples(features_array)

            # Analýza výsledků
            anomaly_score = scores[0]
            is_anomaly = predictions[0] == -1

            if is_anomaly or anomaly_score < -0.5:
                anomaly_type = self.classify_anomaly(metrics, anomaly_score)

                anomaly = {
                    'type': 'hardware_anomaly',
                    'component': anomaly_type['component'],
                    'severity': anomaly_type['severity'],
                    'description': anomaly_type['description'],
                    'anomaly_score': float(anomaly_score),
                    'confidence': float(abs(anomaly_score)),
                    'metrics': metrics,
                    'timestamp': datetime.now().isoformat()
                }
                anomalies.append(anomaly)

        except Exception as e:
            logger.error(f"❌ Chyba při detekci anomálií: {e}")

        return anomalies

    def classify_anomaly(self, metrics: Dict, score: float) -> Dict:
        """Klasifikuje typ anomálie"""
        try:
            # Analýza jednotlivých komponent
            cpu_temp = metrics.get('cpu', {}).get('temperature_c', 0)
            gpu_temps = [gpu.get('temperature_c', 0) for gpu in metrics.get('gpu', [])]
            memory_usage = metrics.get('memory', {}).get('usage_percent', 0)
            disk_usage = metrics.get('disk', {}).get('usage_percent', 0)

            # Kritické teploty
            if cpu_temp > self.critical_temp_cpu:
                return {
                    'component': 'cpu',
                    'severity': 'critical',
                    'description': f'🚨 KRITICKÁ TEPLOTA CPU: {cpu_temp}°C (limit: {self.critical_temp_cpu}°C)'
                }

            if gpu_temps and max(gpu_temps) > self.critical_temp_gpu:
                return {
                    'component': 'gpu',
                    'severity': 'critical',
                    'description': f'🚨 KRITICKÁ TEPLOTA GPU: {max(gpu_temps)}°C (limit: {self.critical_temp_gpu}°C)'
                }

            # Vysoké využití
            if memory_usage > self.critical_memory_usage:
                return {
                    'component': 'memory',
                    'severity': 'high',
                    'description': f'⚠️ VYSOKÉ VYUŽITÍ PAMĚTI: {memory_usage}% (limit: {self.critical_memory_usage}%)'
                }

            if disk_usage > self.critical_disk_usage:
                return {
                    'component': 'disk',
                    'severity': 'high',
                    'description': f'⚠️ VYSOKÉ VYUŽITÍ DISKU: {disk_usage}% (limit: {self.critical_disk_usage}%)'
                }

            # Mírné anomálie
            if abs(score) > 1.5:
                return {
                    'component': 'system',
                    'severity': 'medium',
                    'description': f'🟡 SYSTÉMOVÁ ANOMÁLIE - Neobvyklé chování detekováno'
                }

            return {
                'component': 'unknown',
                'severity': 'low',
                'description': f'ℹ️ MÍRNÁ ANOMÁLIE - Sledování doporučeno'
            }

        except Exception as e:
            logger.warning(f"❌ Chyba při klasifikaci anomálie: {e}")
            return {
                'component': 'unknown',
                'severity': 'unknown',
                'description': 'Neznámá anomálie'
            }

    def predict_failures(self) -> List[Dict]:
        """Predikuje budoucí poruchy hardware"""
        predictions = []

        if self.failure_predictor is None or len(self.system_history) < 50:
            return predictions

        try:
            # Příprava dat pro predikci
            recent_metrics = list(self.system_history)[-30:]  # Posledních 30 měření

            for i, metrics in enumerate(recent_metrics):
                features = self.extract_maintenance_features(metrics)

                # Predikce pravděpodobnosti poruchy
                features_array = np.array([features])

                if hasattr(self.failure_predictor, 'predict_proba'):
                    probabilities = self.failure_predictor.predict_proba(features_array)[0]
                    failure_prob = probabilities[1] if len(probabilities) > 1 else 0
                else:
                    # Fallback
                    prediction = self.failure_predictor.predict(features_array)[0]
                    failure_prob = 0.8 if prediction == 1 else 0.2

                # Pokud je vysoká pravděpodobnost poruchy
                if failure_prob > self.failure_prediction_threshold:
                    component = self.predict_failure_component(features)

                    prediction = {
                        'component': component,
                        'failure_probability': float(failure_prob),
                        'predicted_days': self.failure_prediction_days,
                        'description': f'🔮 PREDIKCE PORUCHY: {component.upper()} selže za {self.failure_prediction_days} dní',
                        'recommendations': self.get_maintenance_recommendations(component),
                        'metrics': metrics,
                        'timestamp': datetime.now().isoformat()
                    }
                    predictions.append(prediction)

        except Exception as e:
            logger.error(f"❌ Chyba při predikci poruch: {e}")

        return predictions

    def predict_failure_component(self, features: List[float]) -> str:
        """Predikuje která komponenta selže"""
        try:
            # Jednoduchá logika na základě features
            cpu_temp = features[2] if len(features) > 2 else 0
            memory_usage = features[4] if len(features) > 4 else 0
            disk_usage = features[6] if len(features) > 6 else 0
            gpu_temp = features[10] if len(features) > 10 else 0

            # Nejrizikovější komponenta
            risks = {
                'cpu': cpu_temp / 100,  # Normalizováno
                'memory': memory_usage / 100,
                'disk': disk_usage / 100,
                'gpu': gpu_temp / 100 if gpu_temp > 0 else 0
            }

            return max(risks, key=risks.get)

        except Exception as e:
            logger.warning(f"❌ Chyba při predikci komponenty: {e}")
            return 'unknown'

    def get_maintenance_recommendations(self, component: str) -> List[str]:
        """Generuje doporučení pro údržbu"""
        recommendations = {
            'cpu': [
                "🧊 Zkontrolujte chlazení CPU",
                "🧹 Vyčistěte prach z chladiče",
                "💨 Zlepšete ventilaci skříně",
                "⚡ Zkontrolujte napájení"
            ],
            'gpu': [
                "🧊 Zkontrolujte chlazení GPU",
                "🧹 Vyčistěte prach z grafické karty",
                "💨 Zlepšete ventilaci skříně",
                "🎮 Aktualizujte GPU ovladače"
            ],
            'memory': [
                "🧠 Zkontrolujte RAM moduly",
                "🔍 Spusťte memtest",
                "💾 Zvažte upgrade RAM",
                "🧹 Vyčistěte kontakty"
            ],
            'disk': [
                "💽 Zkontrolujte S.M.A.R.T. status",
                "🔍 Spusťte disk diagnostiku",
                "💾 Zálohujte důležitá data",
                "🔄 Zvažte defragmentaci"
            ],
            'unknown': [
                "🔍 Proveďte kompletní diagnostiku systému",
                "📊 Sledujte systémové logy",
                "🛠️ Zkontrolujte všechny komponenty"
            ]
        }

        return recommendations.get(component, recommendations['unknown'])

    def generate_alerts(self, anomalies: List[Dict], predictions: List[Dict]) -> List[Dict]:
        """Generuje alerty na základě anomálií a predikcí"""
        alerts = []

        # Alerty pro anomálie
        for anomaly in anomalies:
            if anomaly.get('confidence', 0) > self.alert_confidence_threshold:
                alert = {
                    'type': 'anomaly_alert',
                    'severity': anomaly.get('severity', 'low'),
                    'title': f"Hardware Anomálie - {anomaly.get('component', 'unknown').upper()}",
                    'message': anomaly.get('description', 'Neznámá anomálie'),
                    'component': anomaly.get('component'),
                    'confidence': anomaly.get('confidence'),
                    'timestamp': datetime.now().isoformat()
                }
                alerts.append(alert)

        # Alerty pro predikce poruch
        for prediction in predictions:
            if prediction.get('failure_probability', 0) > self.failure_prediction_threshold:
                alert = {
                    'type': 'failure_prediction',
                    'severity': 'high',
                    'title': f"Predikce Poruchy - {prediction.get('component', 'unknown').upper()}",
                    'message': prediction.get('description', 'Predikce poruchy'),
                    'component': prediction.get('component'),
                    'days_until_failure': prediction.get('predicted_days'),
                    'recommendations': prediction.get('recommendations', []),
                    'timestamp': datetime.now().isoformat()
                }
                alerts.append(alert)

        return alerts

    def perform_maintenance_checks(self) -> Dict:
        """Provede kompletní kontrolu stavu systému"""
        try:
            metrics = self.collect_system_metrics()

            if not metrics:
                return {'error': 'Nelze získat systémové metriky'}

            # Detekce anomálií
            anomalies = self.detect_anomalies(metrics)

            # Predikce poruch
            predictions = self.predict_failures()

            # Generování alertů
            alerts = self.generate_alerts(anomalies, predictions)

            # Aktualizace historie
            self.system_history.append(metrics)

            # Přidání do specifických historií
            self.cpu_history.append(metrics.get('cpu', {}))
            self.memory_history.append(metrics.get('memory', {}))
            self.disk_history.append(metrics.get('disk', {}))
            self.gpu_history.append(metrics.get('gpu', []))

            # Aktualizace aktivních alertů
            for alert in alerts:
                alert_key = f"{alert['type']}_{alert['component']}"
                self.active_alerts.add(alert_key)

            report = {
                'system_health': self.assess_system_health(metrics),
                'anomalies_detected': len(anomalies),
                'failure_predictions': len(predictions),
                'active_alerts': len(self.active_alerts),
                'alerts': alerts,
                'metrics': metrics,
                'recommendations': self.generate_maintenance_plan(alerts),
                'timestamp': datetime.now().isoformat()
            }

            return report

        except Exception as e:
            logger.error(f"❌ Chyba při provádění maintenance kontroly: {e}")
            return {'error': str(e)}

    def assess_system_health(self, metrics: Dict) -> Dict:
        """Vyhodnotí celkové zdraví systému"""
        try:
            health_score = 100  # Začínáme s perfektním skóre
            issues = []

            # CPU health
            cpu = metrics.get('cpu', {})
            cpu_temp = cpu.get('temperature_c', 0)
            cpu_usage = cpu.get('usage_percent', 0)

            if cpu_temp > self.critical_temp_cpu:
                health_score -= 30
                issues.append("Kritická teplota CPU")
            elif cpu_temp > 70:
                health_score -= 10
                issues.append("Vysoká teplota CPU")

            if cpu_usage > 95:
                health_score -= 15
                issues.append("Extrémní zatížení CPU")

            # Memory health
            memory = metrics.get('memory', {})
            mem_usage = memory.get('usage_percent', 0)

            if mem_usage > self.critical_memory_usage:
                health_score -= 25
                issues.append("Kritické využití paměti")
            elif mem_usage > 85:
                health_score -= 10
                issues.append("Vysoké využití paměti")

            # Disk health
            disk = metrics.get('disk', {})
            disk_usage = disk.get('usage_percent', 0)

            if disk_usage > self.critical_disk_usage:
                health_score -= 20
                issues.append("Kritické využití disku")
            elif disk_usage > 85:
                health_score -= 5
                issues.append("Vysoké využití disku")

            # GPU health
            gpus = metrics.get('gpu', [])
            for gpu in gpus:
                gpu_temp = gpu.get('temperature_c', 0)
                gpu_mem = gpu.get('memory_usage_percent', 0)

                if gpu_temp > self.critical_temp_gpu:
                    health_score -= 25
                    issues.append(f"Kritická teplota GPU {gpu.get('id', 0)}")
                elif gpu_temp > 70:
                    health_score -= 8
                    issues.append(f"Vysoká teplota GPU {gpu.get('id', 0)}")

                if gpu_mem > 95:
                    health_score -= 15
                    issues.append(f"Kritické využití GPU paměti {gpu.get('id', 0)}")

            health_score = max(0, min(100, health_score))

            health_status = "excellent" if health_score >= 90 else \
                           "good" if health_score >= 75 else \
                           "fair" if health_score >= 60 else \
                           "poor" if health_score >= 40 else "critical"

            return {
                'overall_score': health_score,
                'status': health_status,
                'issues': issues,
                'component_scores': {
                    'cpu': self.calculate_component_score('cpu', metrics),
                    'memory': self.calculate_component_score('memory', metrics),
                    'disk': self.calculate_component_score('disk', metrics),
                    'gpu': self.calculate_component_score('gpu', metrics)
                }
            }

        except Exception as e:
            logger.error(f"❌ Chyba při hodnocení systémového zdraví: {e}")
            return {
                'overall_score': 0,
                'status': 'unknown',
                'issues': ['Chyba při hodnocení'],
                'component_scores': {}
            }

    def calculate_component_score(self, component: str, metrics: Dict) -> int:
        """Vypočítá skóre zdraví pro jednotlivé komponenty"""
        try:
            if component == 'cpu':
                temp = metrics.get('cpu', {}).get('temperature_c', 0)
                usage = metrics.get('cpu', {}).get('usage_percent', 0)
                score = 100 - (temp * 0.5) - (usage * 0.3)
            elif component == 'memory':
                usage = metrics.get('memory', {}).get('usage_percent', 0)
                score = 100 - usage
            elif component == 'disk':
                usage = metrics.get('disk', {}).get('usage_percent', 0)
                score = 100 - usage
            elif component == 'gpu':
                gpus = metrics.get('gpu', [])
                if gpus:
                    avg_temp = np.mean([gpu.get('temperature_c', 0) for gpu in gpus])
                    avg_usage = np.mean([gpu.get('memory_usage_percent', 0) for gpu in gpus])
                    score = 100 - (avg_temp * 0.4) - (avg_usage * 0.3)
                else:
                    score = 100  # Žádné GPU = perfektní skóre
            else:
                score = 100

            return max(0, min(100, int(score)))

        except Exception as e:
            logger.warning(f"❌ Chyba při výpočtu skóre komponenty {component}: {e}")
            return 50  # Střední hodnota při chybě

    def generate_maintenance_plan(self, alerts: List[Dict]) -> List[Dict]:
        """Generuje plán údržby na základě alertů"""
        maintenance_tasks = []

        try:
            # Kritické úkoly
            critical_alerts = [a for a in alerts if a.get('severity') == 'critical']
            if critical_alerts:
                maintenance_tasks.append({
                    'priority': 'critical',
                    'title': 'Okamžitá údržba vyžadována',
                    'description': f"{len(critical_alerts)} kritických problémů detekováno",
                    'estimated_time': '1-2 hodiny',
                    'components': list(set([a.get('component') for a in critical_alerts]))
                })

            # Vysoká priorita
            high_alerts = [a for a in alerts if a.get('severity') == 'high']
            if high_alerts:
                maintenance_tasks.append({
                    'priority': 'high',
                    'title': 'Plánovaná údržba',
                    'description': f"{len(high_alerts)} problémů vysoké priority",
                    'estimated_time': '2-4 hodiny',
                    'components': list(set([a.get('component') for a in high_alerts]))
                })

            # Preventivní údržba
            if not alerts:  # Žádné alerty
                maintenance_tasks.append({
                    'priority': 'low',
                    'title': 'Preventivní údržba',
                    'description': 'Pravidelná kontrola a čištění systému',
                    'estimated_time': '30-60 minut',
                    'components': ['cpu', 'gpu', 'memory', 'disk']
                })

        except Exception as e:
            logger.error(f"❌ Chyba při generování plánu údržby: {e}")

        return maintenance_tasks

    def start_monitoring(self):
        """Spustí kontinuální monitoring systému"""
        def monitoring_loop():
            logger.info("🔧 Spuštěn AI prediktivní maintenance monitoring")

            while True:
                try:
                    # Provedení kontroly
                    report = self.perform_maintenance_checks()

                    if 'error' not in report:
                        health = report.get('system_health', {})
                        health_score = health.get('overall_score', 0)
                        status = health.get('status', 'unknown')

                        # Logování podle stavu
                        if status == 'critical':
                            logger.critical(f"🚨 KRITICKÝ STAV SYSTÉMU - Skóre zdraví: {health_score}")
                        elif status == 'poor':
                            logger.error(f"❌ ŠPATNÝ STAV SYSTÉMU - Skóre zdraví: {health_score}")
                        elif status == 'fair':
                            logger.warning(f"⚠️ STŘEDNÍ STAV SYSTÉMU - Skóre zdraví: {health_score}")
                        elif health_score < 80:
                            logger.info(f"ℹ️ Skóre zdraví systému: {health_score} ({status})")

                        # Logování alertů
                        alerts = report.get('alerts', [])
                        for alert in alerts:
                            severity = alert.get('severity', 'low')
                            if severity in ['critical', 'high']:
                                logger.warning(f"🚨 ALERT: {alert.get('title')} - {alert.get('message')}")

                    time.sleep(self.monitoring_interval)

                except Exception as e:
                    logger.error(f"❌ Chyba v maintenance monitoringu: {e}")
                    time.sleep(self.monitoring_interval)

        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()

def main():
    """Hlavní funkce pro testování AI prediktivní údržby"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("🔧 ZION AI Predictive Maintenance")
    print("=" * 50)

    # Vytvoření instance
    maintenance = ZionPredictiveMaintenance(monitoring_interval=30)  # Každých 30 sekund

    # Spuštění monitoringu
    maintenance.start_monitoring()

    print("✅ AI Predictive Maintenance spuštěn - monitoruje hardware")
    print("📊 Pro získání maintenance reportu zavolejte maintenance.perform_maintenance_checks()")

    # Testování kontroly
    while True:
        try:
            print("\n🔍 Provedení maintenance kontroly...")
            report = maintenance.perform_maintenance_checks()

            if 'error' not in report:
                health = report.get('system_health', {})
                print(f"💚 Skóre zdraví: {health.get('overall_score', 0)}/100 ({health.get('status', 'unknown')})")

                alerts = report.get('alerts', [])
                print(f"🚨 Aktivních alertů: {len(alerts)}")

                for alert in alerts[:3]:  # Zobraz prvních 3 alerty
                    print(f"  {alert.get('severity', 'low').upper()}: {alert.get('title')}")

                issues = health.get('issues', [])
                if issues:
                    print(f"⚠️ Problémy: {', '.join(issues[:3])}")

                maintenance_plan = report.get('recommendations', [])
                if maintenance_plan:
                    print("🛠️ Doporučená údržba:")
                    for task in maintenance_plan[:2]:  # Prvních 2 úkoly
                        print(f"  {task.get('priority', 'low').upper()}: {task.get('title')}")

            time.sleep(60)  # Aktualizace každou minutu

        except KeyboardInterrupt:
            print("\n⏹️ AI Predictive Maintenance zastaven uživatelem")

            # Závěrečná zpráva
            final_report = maintenance.perform_maintenance_checks()
            health = final_report.get('system_health', {})
            print("\n📊 KONEČNÁ ZPRÁVA O ZDRAVÍ SYSTÉMU:")
            print(f"   Skóre zdraví: {health.get('overall_score', 0)}/100")
            print(f"   Status: {health.get('status', 'unknown').upper()}")
            print(f"   Aktivních alertů: {len(final_report.get('alerts', []))}")
            break

        except Exception as e:
            print(f"❌ Chyba: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()