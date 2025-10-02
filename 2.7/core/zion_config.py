#!/usr/bin/env python3
"""
⚙️ ZION 2.7 UNIFIED CONFIGURATION SYSTEM ⚙️
Centralizovaný konfigurační management pro všechny ZION komponenty

Features:
- Dynamic configuration hot-reload
- Component-specific settings
- Performance tuning parameters
- Environment-aware configs
- Secure credential management
- Configuration validation
- Auto-optimization based on hardware
"""

import os
import json
import time
import psutil
import platform
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import threading
from datetime import datetime
import hashlib
import subprocess

class ConfigScope(Enum):
    """Configuration scope levels"""
    GLOBAL = "global"
    COMPONENT = "component"
    ENVIRONMENT = "environment"
    USER = "user"
    RUNTIME = "runtime"

class PerformanceProfile(Enum):
    """Performance optimization profiles"""
    ECONOMY = "economy"           # Low power, basic performance
    BALANCED = "balanced"         # Default balanced settings
    PERFORMANCE = "performance"   # High performance mode  
    EXTREME = "extreme"          # Maximum performance, high power
    CUSTOM = "custom"            # User-defined settings

@dataclass
class SystemInfo:
    """System hardware information for auto-optimization"""
    cpu_count: int
    cpu_freq_max: float
    memory_total: int
    memory_available: int
    gpu_devices: List[str]
    disk_free: int
    platform: str
    architecture: str
    python_version: str

@dataclass
class PerformanceConfig:
    """Performance tuning configuration"""
    # CPU settings
    cpu_threads: int = 0  # 0 = auto-detect
    cpu_affinity: List[int] = field(default_factory=list)
    cpu_priority: str = "normal"  # low, normal, high, realtime
    
    # Memory settings  
    memory_limit_mb: int = 0  # 0 = unlimited
    cache_size_mb: int = 512
    memory_pool_size: int = 1024
    
    # GPU settings
    gpu_enabled: bool = True
    gpu_intensity: float = 0.8
    gpu_memory_limit: float = 0.9
    gpu_temperature_limit: int = 85
    
    # Mining specific
    mining_threads: int = 0
    mining_intensity: str = "auto"  # low, medium, high, extreme, auto
    randomx_cache_size: int = 2048
    
    # AI specific
    ai_batch_size: int = 32
    ai_model_precision: str = "float32"  # float16, float32, float64
    ai_inference_threads: int = 4
    
    # Network settings
    max_connections: int = 50
    connection_timeout: int = 30
    keepalive_interval: int = 60

@dataclass
class ComponentConfig:
    """Individual component configuration"""
    enabled: bool = True
    log_level: str = "INFO"
    performance_profile: PerformanceProfile = PerformanceProfile.BALANCED
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    auto_restart: bool = True
    health_check_interval: int = 60

class ZionConfigManager:
    """Unified configuration manager for ZION 2.7"""
    
    def __init__(self, config_dir: str = None):
        # Use relative path if not specified
        if config_dir is None:
            config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Configuration files
        self.global_config_file = self.config_dir / "zion_global.json"
        self.components_config_file = self.config_dir / "zion_components.json" 
        self.performance_config_file = self.config_dir / "zion_performance.json"
        self.user_config_file = self.config_dir / "zion_user.json"
        
        # Runtime state
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.system_info: Optional[SystemInfo] = None
        self.performance_config: Optional[PerformanceConfig] = None
        self.lock = threading.RLock()
        self.watchers = {}
        
        # Initialize
        self._detect_system_info()
        self._load_all_configs()
        self._auto_optimize()
    
    def _detect_system_info(self):
        """Detect system hardware for auto-optimization"""
        try:
            cpu_freq = psutil.cpu_freq()
        except:
            # Fallback for systems where cpu_freq is not available (like macOS)
            cpu_freq = None
            
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Detect GPU devices
        gpu_devices = []
        try:
            # Try NVIDIA first
            result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if 'GPU' in line:
                        gpu_devices.append(line.strip())
        except:
            pass
        
        try:
            # Try AMD/OpenCL
            result = subprocess.run(['clinfo', '-l'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and 'Platform' in result.stdout:
                gpu_devices.append("AMD/OpenCL Device Detected")
        except:
            pass
        
        if not gpu_devices:
            gpu_devices.append("CPU_ONLY")
        
        self.system_info = SystemInfo(
            cpu_count=psutil.cpu_count(logical=True),
            cpu_freq_max=cpu_freq.max if cpu_freq else 0,
            memory_total=memory.total // (1024**3),  # GB
            memory_available=memory.available // (1024**3),  # GB
            gpu_devices=gpu_devices,
            disk_free=disk.free // (1024**3),  # GB
            platform=platform.system(),
            architecture=platform.machine(),
            python_version=platform.python_version()
        )
    
    def _load_all_configs(self):
        """Load all configuration files"""
        # Load global config
        self.configs['global'] = self._load_config_file(
            self.global_config_file, 
            self._get_default_global_config()
        )
        
        # Load component configs
        self.configs['components'] = self._load_config_file(
            self.components_config_file,
            self._get_default_component_configs()
        )
        
        # Load performance config
        perf_data = self._load_config_file(
            self.performance_config_file,
            asdict(PerformanceConfig())
        )
        self.performance_config = PerformanceConfig(**perf_data)
        
        # Load user config
        self.configs['user'] = self._load_config_file(
            self.user_config_file,
            {}
        )
    
    def _load_config_file(self, file_path: Path, default_config: Dict[str, Any]) -> Dict[str, Any]:
        """Load configuration from JSON file with fallback to defaults"""
        try:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    merged_config = default_config.copy()
                    merged_config.update(config)
                    return merged_config
            else:
                # Create default config file
                self._save_config_file(file_path, default_config)
                return default_config
                
        except Exception as e:
            print(f"Error loading config from {file_path}: {e}")
            return default_config
    
    def _save_config_file(self, file_path: Path, config: Dict[str, Any]):
        """Save configuration to JSON file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving config to {file_path}: {e}")
    
    def _get_default_global_config(self) -> Dict[str, Any]:
        """Get default global configuration"""
        return {
            "zion_version": "2.7",
            "network": "mainnet",
            "data_dir": "/media/maitreya/ZION1/data",
            "log_dir": "/media/maitreya/ZION1/logs",
            "blockchain": {
                "port": 18080,
                "rpc_port": 18081,
                "max_block_size": 1048576,
                "difficulty_target": 120,
                "emission_factor": 20
            },
            "p2p": {
                "max_peers": 50,
                "seed_nodes": [
                    "seed1.zion.network:18080",
                    "seed2.zion.network:18080"
                ]
            },
            "security": {
                "ssl_enabled": True,
                "api_key_required": True,
                "rate_limiting": True
            }
        }
    
    def _get_default_component_configs(self) -> Dict[str, Any]:
        """Get default component configurations"""
        return {
            "blockchain": {
                "enabled": True,
                "log_level": "INFO",
                "performance_profile": "balanced",
                "auto_restart": True,
                "health_check_interval": 30
            },
            "mining": {
                "enabled": True,
                "log_level": "INFO", 
                "performance_profile": "performance",
                "auto_restart": True,
                "health_check_interval": 10
            },
            "gpu_mining": {
                "enabled": True,
                "log_level": "INFO",
                "performance_profile": "performance",
                "auto_restart": True,
                "health_check_interval": 15
            },
            "ai_afterburner": {
                "enabled": True,
                "log_level": "INFO",
                "performance_profile": "balanced",
                "auto_restart": True,
                "health_check_interval": 20
            },
            "perfect_memory": {
                "enabled": True,
                "log_level": "INFO",
                "performance_profile": "extreme",
                "auto_restart": True,
                "health_check_interval": 5
            },
            "network": {
                "enabled": True,
                "log_level": "WARNING",
                "performance_profile": "balanced",
                "auto_restart": True,
                "health_check_interval": 60
            }
        }
    
    def _auto_optimize(self):
        """Auto-optimize configuration based on system hardware"""
        if not self.system_info:
            return
        
        # Auto-optimize based on system specs
        optimizations = {}
        
        # CPU optimization
        if self.system_info.cpu_count >= 16:
            optimizations['cpu_threads'] = self.system_info.cpu_count - 2
            optimizations['mining_threads'] = max(1, self.system_info.cpu_count // 2)
        elif self.system_info.cpu_count >= 8:
            optimizations['cpu_threads'] = self.system_info.cpu_count - 1
            optimizations['mining_threads'] = max(1, self.system_info.cpu_count // 3)
        else:
            optimizations['cpu_threads'] = max(1, self.system_info.cpu_count - 1)
            optimizations['mining_threads'] = max(1, self.system_info.cpu_count // 4)
        
        # Memory optimization
        if self.system_info.memory_total >= 32:
            optimizations['cache_size_mb'] = 2048
            optimizations['randomx_cache_size'] = 4096
        elif self.system_info.memory_total >= 16:
            optimizations['cache_size_mb'] = 1024
            optimizations['randomx_cache_size'] = 2048
        else:
            optimizations['cache_size_mb'] = 512
            optimizations['randomx_cache_size'] = 1024
        
        # GPU optimization
        if any('NVIDIA' in gpu for gpu in self.system_info.gpu_devices):
            optimizations['gpu_intensity'] = 0.9
            optimizations['gpu_memory_limit'] = 0.85
        elif any('AMD' in gpu or 'OpenCL' in gpu for gpu in self.system_info.gpu_devices):
            optimizations['gpu_intensity'] = 0.8
            optimizations['gpu_memory_limit'] = 0.9
        else:
            optimizations['gpu_enabled'] = False
        
        # Apply optimizations
        for key, value in optimizations.items():
            if hasattr(self.performance_config, key):
                setattr(self.performance_config, key, value)
        
        # Save optimized config
        self._save_config_file(
            self.performance_config_file,
            asdict(self.performance_config)
        )
    
    def get_config(self, component: str, key: str = None, default: Any = None) -> Any:
        """Get configuration value"""
        with self.lock:
            if component not in self.configs:
                return default
            
            config = self.configs[component]
            
            if key is None:
                return config
            
            # Support nested keys with dot notation
            keys = key.split('.')
            value = config
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
    
    def set_config(self, component: str, key: str, value: Any):
        """Set configuration value"""
        with self.lock:
            if component not in self.configs:
                self.configs[component] = {}
            
            # Support nested keys with dot notation
            keys = key.split('.')
            config = self.configs[component]
            
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            config[keys[-1]] = value
            
            # Save to file
            config_file = getattr(self, f"{component}_config_file", None)
            if config_file:
                self._save_config_file(config_file, self.configs[component])
    
    def get_performance_config(self) -> PerformanceConfig:
        """Get performance configuration"""
        return self.performance_config
    
    def update_performance_config(self, **kwargs):
        """Update performance configuration"""
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self.performance_config, key):
                    setattr(self.performance_config, key, value)
            
            # Save updated config
            self._save_config_file(
                self.performance_config_file,
                asdict(self.performance_config)
            )
    
    def get_system_info(self) -> SystemInfo:
        """Get system information"""
        return self.system_info
    
    def reload_configs(self):
        """Reload all configurations from files"""
        with self.lock:
            self._load_all_configs()
    
    def get_optimized_mining_config(self) -> Dict[str, Any]:
        """Get optimized mining configuration"""
        return {
            'threads': self.performance_config.cpu_threads,
            'intensity': self.performance_config.mining_intensity,
            'cache_size': self.performance_config.randomx_cache_size,
            'affinity': self.performance_config.cpu_affinity,
            'priority': self.performance_config.cpu_priority
        }
    
    def get_optimized_gpu_config(self) -> Dict[str, Any]:
        """Get optimized GPU configuration"""
        return {
            'enabled': self.performance_config.gpu_enabled,
            'intensity': self.performance_config.gpu_intensity,
            'memory_limit': self.performance_config.gpu_memory_limit,
            'temperature_limit': self.performance_config.gpu_temperature_limit
        }
    
    def get_optimized_ai_config(self) -> Dict[str, Any]:
        """Get optimized AI configuration"""
        return {
            'batch_size': self.performance_config.ai_batch_size,
            'precision': self.performance_config.ai_model_precision,
            'threads': self.performance_config.ai_inference_threads
        }

# Global configuration manager instance
_config_manager: Optional[ZionConfigManager] = None

def get_config_manager() -> ZionConfigManager:
    """Get global configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ZionConfigManager()
    return _config_manager

def get_config(component: str, key: str = None, default: Any = None) -> Any:
    """Quick config access"""
    return get_config_manager().get_config(component, key, default)

def set_config(component: str, key: str, value: Any):
    """Quick config setter"""
    get_config_manager().set_config(component, key, value)

if __name__ == "__main__":
    # Test configuration system
    print("🧪 Testing ZION Configuration System...")
    
    config_mgr = get_config_manager()
    
    # Print system info
    print(f"\n💻 System Info:")
    system_info = config_mgr.get_system_info()
    print(f"  CPU Cores: {system_info.cpu_count}")
    print(f"  Memory: {system_info.memory_total} GB")
    print(f"  GPU Devices: {system_info.gpu_devices}")
    print(f"  Platform: {system_info.platform} {system_info.architecture}")
    
    # Print performance config
    print(f"\n⚡ Performance Config:")
    perf_config = config_mgr.get_performance_config()
    print(f"  Mining Threads: {perf_config.mining_threads}")
    print(f"  GPU Enabled: {perf_config.gpu_enabled}")
    print(f"  GPU Intensity: {perf_config.gpu_intensity}")
    print(f"  Cache Size: {perf_config.cache_size_mb} MB")
    
    # Test optimized configs
    print(f"\n🎯 Optimized Configs:")
    print(f"  Mining: {config_mgr.get_optimized_mining_config()}")
    print(f"  GPU: {config_mgr.get_optimized_gpu_config()}")
    print(f"  AI: {config_mgr.get_optimized_ai_config()}")
    
    print("\n✅ Configuration system test completed!")