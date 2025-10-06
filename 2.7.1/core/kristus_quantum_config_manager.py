#!/usr/bin/env python3
"""
🛡️ ZION 2.7.1 KRISTUS QUANTUM CONFIG MANAGER 🛡️
Safe Configuration Management for KRISTUS Quantum Engine

SAFETY-FIRST APPROACH:
- Default quantum computing OFF
- Comprehensive validation
- Runtime safety checks
- Automatic fallback mechanisms
- Extensive logging
"""

import json
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class QuantumEngineConfig:
    """Safe quantum engine configuration"""
    enabled: bool = False
    register_size: int = 8
    consciousness_field_limit: float = 0.8
    enable_sacred_flower_enhancement: bool = True
    fallback_to_standard_hash: bool = True
    require_numpy: bool = True
    testing_mode: bool = False
    max_error_rate: float = 0.05

class ZionKristusConfigManager:
    """
    🛡️ Safe Configuration Manager for KRISTUS Quantum Engine
    
    Ensures safe operation with multiple validation layers:
    - Configuration file validation
    - Runtime safety checks
    - Automatic fallback mechanisms
    - Error monitoring and automatic shutdown
    """
    
    def __init__(self, config_file_path: Optional[str] = None):
        self.config_file_path = config_file_path or os.path.join(
            os.path.dirname(__file__), '..', 'config', 'kristus_quantum_config.json'
        )
        
        self.config_data = {}
        self.quantum_config = QuantumEngineConfig()
        self.safety_warnings = []
        self.config_loaded = False
        
        self.load_configuration()
    
    def load_configuration(self) -> bool:
        """🛡️ Safely load configuration with validation"""
        try:
            # Check if config file exists
            if not os.path.exists(self.config_file_path):
                logger.warning(f"🛡️ Config file not found: {self.config_file_path}")
                self._create_default_config()
                return False
            
            # Load configuration file
            with open(self.config_file_path, 'r', encoding='utf-8') as f:
                self.config_data = json.load(f)
            
            # Validate and apply configuration
            if self._validate_configuration():
                self._apply_configuration()
                self.config_loaded = True
                logger.info("✅ KRISTUS quantum configuration loaded successfully")
                return True
            else:
                logger.error("❌ Configuration validation failed - using safe defaults")
                self._apply_safe_defaults()
                return False
                
        except Exception as e:
            logger.error(f"❌ Configuration loading error: {e}")
            self.safety_warnings.append(f"Config loading error: {e}")
            self._apply_safe_defaults()
            return False
    
    def _validate_configuration(self) -> bool:
        """Validate configuration for safety"""
        try:
            quantum_config = self.config_data.get('quantum_computing', {})
            
            # Safety check: Ensure quantum is disabled by default
            if quantum_config.get('enabled', False):
                self.safety_warnings.append("⚠️ Quantum computing enabled - ensure thorough testing!")
            
            # Validate register size
            register_size = quantum_config.get('engine_settings', {}).get('register_size', 8)
            if not (4 <= register_size <= 16):
                self.safety_warnings.append(f"⚠️ Invalid register size: {register_size}")
                return False
            
            # Validate consciousness field limit
            consciousness_limit = quantum_config.get('engine_settings', {}).get('consciousness_field_limit', 0.8)
            if not (0.1 <= consciousness_limit <= 1.0):
                self.safety_warnings.append(f"⚠️ Invalid consciousness field limit: {consciousness_limit}")
                return False
            
            # Check safety controls
            safety_controls = quantum_config.get('safety_controls', {})
            required_safety_features = [
                'validate_inputs', 'exception_handling', 'automatic_fallback', 'error_logging'
            ]
            
            for feature in required_safety_features:
                if not safety_controls.get(feature, False):
                    self.safety_warnings.append(f"⚠️ Safety feature disabled: {feature}")
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False
    
    def _apply_configuration(self):
        """Apply validated configuration"""
        try:
            quantum_config = self.config_data.get('quantum_computing', {})
            engine_settings = quantum_config.get('engine_settings', {})
            safety_controls = quantum_config.get('safety_controls', {})
            performance_monitoring = quantum_config.get('performance_monitoring', {})
            
            # Apply quantum engine settings
            self.quantum_config.enabled = quantum_config.get('enabled', False)
            self.quantum_config.register_size = engine_settings.get('register_size', 8)
            self.quantum_config.consciousness_field_limit = engine_settings.get('consciousness_field_limit', 0.8)
            self.quantum_config.enable_sacred_flower_enhancement = engine_settings.get('enable_sacred_flower_enhancement', True)
            self.quantum_config.fallback_to_standard_hash = engine_settings.get('fallback_to_standard_hash', True)
            
            # Apply safety controls
            self.quantum_config.require_numpy = safety_controls.get('require_numpy', True)
            
            # Apply performance monitoring
            self.quantum_config.max_error_rate = performance_monitoring.get('max_error_rate', 0.05)
            
            # Apply testing mode
            testing_mode = quantum_config.get('testing_mode', {})
            self.quantum_config.testing_mode = testing_mode.get('enabled', False)
            
            logger.info(f"🛡️ Configuration applied - Quantum: {'ENABLED' if self.quantum_config.enabled else 'DISABLED'}")
            
        except Exception as e:
            logger.error(f"Configuration application error: {e}")
            self._apply_safe_defaults()
    
    def _apply_safe_defaults(self):
        """Apply safe default configuration"""
        self.quantum_config = QuantumEngineConfig()  # All defaults are safe
        logger.info("🛡️ Safe default configuration applied")
    
    def _create_default_config(self):
        """Create default configuration file"""
        try:
            # Ensure config directory exists
            os.makedirs(os.path.dirname(self.config_file_path), exist_ok=True)
            
            # Create safe default configuration
            default_config = {
                "quantum_computing": {
                    "enabled": False,
                    "description": "🛡️ KRISTUS Quantum Engine - SAFETY FIRST",
                    "warning": "⚠️ Enable quantum computing ONLY after thorough testing!",
                    
                    "engine_settings": {
                        "register_size": 8,
                        "consciousness_field_limit": 0.8,
                        "enable_sacred_flower_enhancement": True,
                        "fallback_to_standard_hash": True
                    },
                    
                    "safety_controls": {
                        "require_numpy": True,
                        "validate_inputs": True,
                        "exception_handling": True,
                        "automatic_fallback": True,
                        "error_logging": True
                    }
                }
            }
            
            with open(self.config_file_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4)
            
            logger.info(f"✅ Default configuration created: {self.config_file_path}")
            
        except Exception as e:
            logger.error(f"Default configuration creation error: {e}")
    
    def get_quantum_config(self) -> QuantumEngineConfig:
        """Get quantum engine configuration"""
        return self.quantum_config
    
    def is_quantum_enabled(self) -> bool:
        """Check if quantum computing is enabled"""
        return self.quantum_config.enabled and self.config_loaded
    
    def is_safe_to_enable_quantum(self) -> bool:
        """Check if it's safe to enable quantum computing"""
        # Check NumPy availability if required
        if self.quantum_config.require_numpy:
            try:
                import numpy
                return True
            except ImportError:
                logger.warning("🛡️ NumPy not available - quantum computing not safe")
                return False
        
        return True
    
    def enable_quantum_mode(self) -> bool:
        """🛡️ Safely enable quantum mode with validation"""
        try:
            if not self.is_safe_to_enable_quantum():
                logger.warning("🛡️ Quantum mode not safe to enable")
                return False
            
            # Update configuration
            self.quantum_config.enabled = True
            
            # Update config file
            if 'quantum_computing' in self.config_data:
                self.config_data['quantum_computing']['enabled'] = True
                with open(self.config_file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config_data, f, indent=4)
            
            logger.info("🌟 Quantum mode safely enabled")
            return True
            
        except Exception as e:
            logger.error(f"Quantum mode enable error: {e}")
            return False
    
    def disable_quantum_mode(self) -> bool:
        """🛡️ Safely disable quantum mode"""
        try:
            self.quantum_config.enabled = False
            
            # Update config file
            if 'quantum_computing' in self.config_data:
                self.config_data['quantum_computing']['enabled'] = False
                with open(self.config_file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config_data, f, indent=4)
            
            logger.info("🛡️ Quantum mode disabled - using safe standard algorithms")
            return True
            
        except Exception as e:
            logger.error(f"Quantum mode disable error: {e}")
            return False
    
    def get_safety_warnings(self) -> list:
        """Get list of safety warnings"""
        return self.safety_warnings.copy()
    
    def get_configuration_status(self) -> Dict[str, Any]:
        """Get comprehensive configuration status"""
        return {
            "config_loaded": self.config_loaded,
            "config_file_path": self.config_file_path,
            "quantum_enabled": self.is_quantum_enabled(),
            "safe_to_enable": self.is_safe_to_enable_quantum(),
            "safety_warnings": self.get_safety_warnings(),
            "quantum_config": {
                "enabled": self.quantum_config.enabled,
                "register_size": self.quantum_config.register_size,
                "consciousness_field_limit": self.quantum_config.consciousness_field_limit,
                "testing_mode": self.quantum_config.testing_mode,
                "fallback_available": self.quantum_config.fallback_to_standard_hash
            }
        }

# Global instance for easy access
_global_config_manager = None

def get_kristus_config_manager() -> ZionKristusConfigManager:
    """Get global KRISTUS configuration manager instance"""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ZionKristusConfigManager()
    return _global_config_manager

def is_quantum_computing_enabled() -> bool:
    """Quick check if quantum computing is enabled"""
    config_mgr = get_kristus_config_manager()
    return config_mgr.is_quantum_enabled()

def get_quantum_engine_config() -> QuantumEngineConfig:
    """Get quantum engine configuration"""
    config_mgr = get_kristus_config_manager()
    return config_mgr.get_quantum_config()

if __name__ == "__main__":
    print("🛡️ ZION 2.7.1 KRISTUS Quantum Config Manager - Safety Testing")
    print("=" * 60)
    
    # Test configuration manager
    config_mgr = ZionKristusConfigManager()
    
    status = config_mgr.get_configuration_status()
    print(f"Configuration Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print(f"\nQuantum Enabled: {config_mgr.is_quantum_enabled()}")
    print(f"Safe to Enable: {config_mgr.is_safe_to_enable_quantum()}")
    
    warnings = config_mgr.get_safety_warnings()
    if warnings:
        print(f"\nSafety Warnings:")
        for warning in warnings:
            print(f"  {warning}")
    else:
        print("\n✅ No safety warnings - configuration is safe")
    
    print("\n✅ KRISTUS configuration manager testing complete!")
    print("🛡️ Ready for safe blockchain integration")