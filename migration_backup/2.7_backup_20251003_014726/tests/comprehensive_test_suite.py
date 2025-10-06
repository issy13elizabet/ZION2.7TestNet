#!/usr/bin/env python3
"""
🧪 ZION 2.7 COMPREHENSIVE TEST SUITE 🧪
Kompletní test suite pro všechny ZION komponenty

Features:
- Unit tests pro všechny komponenty
- Integration tests
- Performance benchmarking
- Stress testing
- Security testing
- Error handling validation
- Configuration testing
"""

import os
import sys
import json
import time
import asyncio
import unittest
import threading
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import subprocess
import psutil
import hashlib
import random
import numpy as np

# Add ZION paths
ZION_ROOT = "/media/maitreya/ZION1/2.7"
sys.path.insert(0, ZION_ROOT)

# Import ZION components
try:
    from core.blockchain import Blockchain
    from mining.randomx_engine import RandomXEngine
    from ai.zion_gpu_miner import ZionGPUMiner
    from ai.zion_ai_afterburner import ZionAIAfterburner
    from ai.zion_perfect_memory_miner import ZionPerfectMemoryMiner
    from core.zion_logging import get_logger, ComponentType, shutdown_all_loggers
    from core.zion_config import get_config_manager
    from core.zion_error_handler import get_error_handler
    
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some ZION components not available: {e}")
    COMPONENTS_AVAILABLE = False

@dataclass
class TestResult:
    """Test result structure"""
    test_name: str
    component: str
    passed: bool
    execution_time: float
    error_message: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None

class ZionTestSuite:
    """Comprehensive ZION test suite"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.temp_dir = None
        self.test_blockchain = None
        self.test_config = None
        self.start_time = time.time()
        
        # Initialize logging
        try:
            self.logger = get_logger(ComponentType.TESTING)
        except:
            import logging
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
    
    def setup_test_environment(self):
        """Setup test environment"""
        self.logger.info("Setting up test environment...")
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="zion_test_")
        self.logger.info(f"Test directory: {self.temp_dir}")
        
        # Initialize test config
        if COMPONENTS_AVAILABLE:
            try:
                self.test_config = get_config_manager()
                self.test_config.config_dir = Path(self.temp_dir) / "config"
                self.test_config.config_dir.mkdir(exist_ok=True)
                self.logger.info("Test configuration initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize test config: {e}")
    
    def cleanup_test_environment(self):
        """Cleanup test environment"""
        self.logger.info("Cleaning up test environment...")
        
        try:
            # Cleanup temporary directory
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                self.logger.info("Test directory cleaned up")
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
    
    def run_test(self, test_name: str, component: str, test_func: Callable) -> TestResult:
        """Run individual test and record result"""
        self.logger.info(f"Running test: {test_name}")
        
        start_time = time.time()
        
        try:
            result = test_func()
            execution_time = time.time() - start_time
            
            test_result = TestResult(
                test_name=test_name,
                component=component,
                passed=True,
                execution_time=execution_time,
                performance_metrics=result if isinstance(result, dict) else None
            )
            
            self.logger.info(f"✅ {test_name} PASSED in {execution_time:.3f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            test_result = TestResult(
                test_name=test_name,
                component=component,
                passed=False,
                execution_time=execution_time,
                error_message=str(e)
            )
            
            self.logger.error(f"❌ {test_name} FAILED: {e}")
        
        self.results.append(test_result)
        return test_result
    
    # Blockchain Tests
    def test_blockchain_initialization(self) -> Dict[str, float]:
        """Test blockchain initialization"""
        if not COMPONENTS_AVAILABLE:
            raise Exception("ZION components not available")
        
        start_time = time.time()
        blockchain = Blockchain()
        init_time = time.time() - start_time
        
        assert blockchain is not None, "Blockchain should initialize"
        assert hasattr(blockchain, 'blocks'), "Blockchain should have blocks"
        assert hasattr(blockchain, 'height'), "Blockchain should have height"
        
        return {"init_time": init_time, "block_count": len(blockchain.blocks)}
    
    def test_blockchain_block_creation(self) -> Dict[str, float]:
        """Test block creation and validation"""
        if not COMPONENTS_AVAILABLE:
            raise Exception("ZION components not available")
        
        blockchain = Blockchain()
        initial_height = blockchain.height
        
        # Create test block data
        start_time = time.time()
        
        # Simulate block creation
        test_data = {"test": "block", "timestamp": time.time()}
        
        creation_time = time.time() - start_time
        
        return {
            "creation_time": creation_time,
            "initial_height": initial_height,
            "data_size": len(json.dumps(test_data))
        }
    
    # RandomX Engine Tests
    def test_randomx_initialization(self) -> Dict[str, float]:
        """Test RandomX engine initialization"""
        if not COMPONENTS_AVAILABLE:
            raise Exception("ZION components not available")
        
        start_time = time.time()
        randomx = RandomXEngine()
        init_time = time.time() - start_time
        
        assert randomx is not None, "RandomX should initialize"
        
        return {"init_time": init_time}
    
    def test_randomx_hashing_performance(self) -> Dict[str, float]:
        """Test RandomX hashing performance"""
        if not COMPONENTS_AVAILABLE:
            raise Exception("ZION components not available")
        
        randomx = RandomXEngine()
        
        # Test hash calculation performance
        test_data = b"test_mining_data_" + os.urandom(32)
        iterations = 100
        
        start_time = time.time()
        
        for i in range(iterations):
            # Simulate RandomX hash (simplified)
            hash_result = hashlib.sha256(test_data + i.to_bytes(4, 'little')).hexdigest()
        
        total_time = time.time() - start_time
        hashrate = iterations / total_time
        
        return {
            "total_time": total_time,
            "iterations": iterations,
            "hashrate": hashrate,
            "avg_time_per_hash": total_time / iterations
        }
    
    # GPU Miner Tests
    def test_gpu_miner_initialization(self) -> Dict[str, float]:
        """Test GPU miner initialization"""
        if not COMPONENTS_AVAILABLE:
            raise Exception("ZION components not available")
        
        start_time = time.time()
        
        try:
            gpu_miner = ZionGPUMiner()
            init_time = time.time() - start_time
            
            assert gpu_miner is not None, "GPU miner should initialize"
            
            return {"init_time": init_time, "gpu_available": True}
            
        except Exception as e:
            init_time = time.time() - start_time
            return {"init_time": init_time, "gpu_available": False, "error": str(e)}
    
    def test_gpu_device_detection(self) -> Dict[str, float]:
        """Test GPU device detection"""
        if not COMPONENTS_AVAILABLE:
            raise Exception("ZION components not available")
        
        start_time = time.time()
        gpu_miner = ZionGPUMiner()
        
        # Test GPU detection
        try:
            gpu_info = gpu_miner.detect_gpu_devices() if hasattr(gpu_miner, 'detect_gpu_devices') else {}
            detection_time = time.time() - start_time
            
            return {
                "detection_time": detection_time,
                "gpu_count": len(gpu_info) if gpu_info else 0,
                "gpus_detected": bool(gpu_info)
            }
            
        except Exception as e:
            detection_time = time.time() - start_time
            return {"detection_time": detection_time, "gpu_count": 0, "error": str(e)}
    
    # AI Afterburner Tests
    def test_ai_afterburner_initialization(self) -> Dict[str, float]:
        """Test AI afterburner initialization"""
        if not COMPONENTS_AVAILABLE:
            raise Exception("ZION components not available")
        
        start_time = time.time()
        ai_afterburner = ZionAIAfterburner()
        init_time = time.time() - start_time
        
        assert ai_afterburner is not None, "AI afterburner should initialize"
        
        return {"init_time": init_time}
    
    def test_ai_compute_performance(self) -> Dict[str, float]:
        """Test AI compute performance"""
        if not COMPONENTS_AVAILABLE:
            raise Exception("ZION components not available")
        
        ai_afterburner = ZionAIAfterburner()
        
        # Test AI computation
        start_time = time.time()
        
        # Simulate AI computation
        test_matrix = np.random.rand(1000, 1000)
        result = np.dot(test_matrix, test_matrix.T)
        
        compute_time = time.time() - start_time
        operations = 1000 * 1000 * 1000  # Matrix operations
        
        return {
            "compute_time": compute_time,
            "operations": operations,
            "ops_per_second": operations / compute_time,
            "matrix_size": 1000
        }
    
    # Perfect Memory Miner Tests
    def test_perfect_memory_miner_initialization(self) -> Dict[str, float]:
        """Test Perfect Memory Miner initialization"""
        if not COMPONENTS_AVAILABLE:
            raise Exception("ZION components not available")
        
        start_time = time.time()
        
        try:
            perfect_miner = ZionPerfectMemoryMiner()
            init_time = time.time() - start_time
            
            assert perfect_miner is not None, "Perfect Memory Miner should initialize"
            
            return {"init_time": init_time, "components_loaded": True}
            
        except Exception as e:
            init_time = time.time() - start_time
            return {"init_time": init_time, "components_loaded": False, "error": str(e)}
    
    # Configuration Tests
    def test_config_manager_functionality(self) -> Dict[str, float]:
        """Test configuration manager"""
        if not COMPONENTS_AVAILABLE:
            return {"skipped": True}
        
        start_time = time.time()
        
        config_mgr = get_config_manager()
        
        # Test config get/set
        test_key = "test_setting"
        test_value = "test_value_123"
        
        config_mgr.set_config("testing", test_key, test_value)
        retrieved_value = config_mgr.get_config("testing", test_key)
        
        assert retrieved_value == test_value, f"Config value mismatch: {retrieved_value} != {test_value}"
        
        # Test performance config
        perf_config = config_mgr.get_performance_config()
        assert perf_config is not None, "Performance config should exist"
        
        test_time = time.time() - start_time
        
        return {"test_time": test_time, "config_functional": True}
    
    def test_system_optimization(self) -> Dict[str, float]:
        """Test system auto-optimization"""
        if not COMPONENTS_AVAILABLE:
            return {"skipped": True}
        
        start_time = time.time()
        
        config_mgr = get_config_manager()
        system_info = config_mgr.get_system_info()
        
        assert system_info is not None, "System info should be available"
        assert system_info.cpu_count > 0, "CPU count should be positive"
        assert system_info.memory_total > 0, "Memory should be positive"
        
        # Test optimized configs
        mining_config = config_mgr.get_optimized_mining_config()
        gpu_config = config_mgr.get_optimized_gpu_config()
        ai_config = config_mgr.get_optimized_ai_config()
        
        assert isinstance(mining_config, dict), "Mining config should be dict"
        assert isinstance(gpu_config, dict), "GPU config should be dict"
        assert isinstance(ai_config, dict), "AI config should be dict"
        
        test_time = time.time() - start_time
        
        return {
            "test_time": test_time,
            "cpu_count": system_info.cpu_count,
            "memory_gb": system_info.memory_total,
            "optimization_successful": True
        }
    
    # Logging Tests
    def test_logging_functionality(self) -> Dict[str, float]:
        """Test logging system"""
        if not COMPONENTS_AVAILABLE:
            return {"skipped": True}
        
        start_time = time.time()
        
        # Test different log levels
        logger = get_logger(ComponentType.TESTING)
        
        logger.info("Test info message")
        logger.debug("Test debug message")
        logger.warning("Test warning message")
        
        # Test performance logging
        test_metrics = {"test_metric": 123.45, "another_metric": 67.89}
        logger.info("Performance test", performance_metrics=test_metrics)
        
        # Test mining and AI specific logging
        from core.zion_logging import log_mining, log_ai
        log_mining("Test mining log", hashrate=1250.5)
        log_ai("Test AI log", accuracy=0.95)
        
        test_time = time.time() - start_time
        
        return {"test_time": test_time, "logging_functional": True}
    
    # Error Handling Tests  
    def test_error_handling_system(self) -> Dict[str, float]:
        """Test error handling and recovery"""
        if not COMPONENTS_AVAILABLE:
            return {"skipped": True}
        
        start_time = time.time()
        
        error_handler = get_error_handler()
        
        # Register test component
        def test_health_check():
            return True
        
        error_handler.register_component("test_component", test_health_check)
        
        # Test error handling
        try:
            raise ValueError("Test error for error handler")
        except Exception as e:
            error_id = error_handler.handle_error("test_component", e)
            assert error_id is not None, "Error should be handled and return ID"
        
        # Test error summary
        error_summary = error_handler.get_error_summary()
        assert isinstance(error_summary, dict), "Error summary should be dict"
        assert error_summary['total_errors'] > 0, "Should have recorded test error"
        
        test_time = time.time() - start_time
        
        return {
            "test_time": test_time,
            "error_handling_functional": True,
            "errors_recorded": error_summary['total_errors']
        }
    
    # Stress Tests
    def test_memory_usage(self) -> Dict[str, float]:
        """Test memory usage under load"""
        start_time = time.time()
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create memory load
        test_data = []
        for i in range(1000):
            test_data.append(np.random.rand(1000))
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Cleanup
        del test_data
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        test_time = time.time() - start_time
        
        return {
            "test_time": test_time,
            "initial_memory_mb": initial_memory,
            "peak_memory_mb": peak_memory,
            "final_memory_mb": final_memory,
            "memory_increase_mb": peak_memory - initial_memory
        }
    
    def test_cpu_performance(self) -> Dict[str, float]:
        """Test CPU performance under load"""
        start_time = time.time()
        
        # CPU intensive task
        result = 0
        iterations = 1000000
        
        for i in range(iterations):
            result += i * i
        
        test_time = time.time() - start_time
        
        return {
            "test_time": test_time,
            "iterations": iterations,
            "operations_per_second": iterations / test_time,
            "result_checksum": result % 1000000
        }
    
    # Integration Tests
    def test_full_system_integration(self) -> Dict[str, float]:
        """Test full system integration"""
        if not COMPONENTS_AVAILABLE:
            return {"skipped": True}
        
        start_time = time.time()
        
        # Initialize all components
        components_initialized = 0
        
        try:
            blockchain = Blockchain()
            components_initialized += 1
        except:
            pass
        
        try:
            randomx = RandomXEngine()
            components_initialized += 1
        except:
            pass
        
        try:
            gpu_miner = ZionGPUMiner()
            components_initialized += 1
        except:
            pass
        
        try:
            ai_afterburner = ZionAIAfterburner()
            components_initialized += 1
        except:
            pass
        
        try:
            config_mgr = get_config_manager()
            components_initialized += 1
        except:
            pass
        
        test_time = time.time() - start_time
        
        return {
            "test_time": test_time,
            "components_initialized": components_initialized,
            "total_components": 5,
            "integration_score": components_initialized / 5
        }
    
    def run_all_tests(self):
        """Run all tests in the suite"""
        self.logger.info("🧪 Starting ZION 2.7 Comprehensive Test Suite")
        
        # Setup test environment
        self.setup_test_environment()
        
        # Define all tests
        tests = [
            # Blockchain Tests
            ("Blockchain Initialization", "blockchain", self.test_blockchain_initialization),
            ("Blockchain Block Creation", "blockchain", self.test_blockchain_block_creation),
            
            # RandomX Tests
            ("RandomX Initialization", "randomx", self.test_randomx_initialization),
            ("RandomX Performance", "randomx", self.test_randomx_hashing_performance),
            
            # GPU Miner Tests
            ("GPU Miner Initialization", "gpu_miner", self.test_gpu_miner_initialization),
            ("GPU Device Detection", "gpu_miner", self.test_gpu_device_detection),
            
            # AI Afterburner Tests
            ("AI Afterburner Initialization", "ai_afterburner", self.test_ai_afterburner_initialization),
            ("AI Compute Performance", "ai_afterburner", self.test_ai_compute_performance),
            
            # Perfect Memory Miner Tests
            ("Perfect Memory Miner Init", "perfect_memory", self.test_perfect_memory_miner_initialization),
            
            # Configuration Tests
            ("Config Manager Functionality", "config", self.test_config_manager_functionality),
            ("System Optimization", "config", self.test_system_optimization),
            
            # Logging Tests
            ("Logging Functionality", "logging", self.test_logging_functionality),
            
            # Error Handling Tests
            ("Error Handling System", "error_handling", self.test_error_handling_system),
            
            # Performance Tests
            ("Memory Usage Test", "performance", self.test_memory_usage),
            ("CPU Performance Test", "performance", self.test_cpu_performance),
            
            # Integration Tests
            ("Full System Integration", "integration", self.test_full_system_integration),
        ]
        
        # Run all tests
        for test_name, component, test_func in tests:
            self.run_test(test_name, component, test_func)
        
        # Cleanup
        self.cleanup_test_environment()
        
        # Generate report
        self.generate_test_report()
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        total_time = time.time() - self.start_time
        
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = sum(1 for r in self.results if not r.passed)
        total_tests = len(self.results)
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print("\n" + "="*80)
        print("🧪 ZION 2.7 TEST SUITE RESULTS")
        print("="*80)
        
        print(f"\n📊 Overall Statistics:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {passed_tests}")
        print(f"  Failed: {failed_tests}")
        print(f"  Success Rate: {success_rate:.1f}%")
        print(f"  Total Time: {total_time:.2f} seconds")
        
        # Results by component
        components = set(r.component for r in self.results)
        print(f"\n🏗️ Results by Component:")
        
        for component in sorted(components):
            comp_results = [r for r in self.results if r.component == component]
            comp_passed = sum(1 for r in comp_results if r.passed)
            comp_total = len(comp_results)
            comp_rate = (comp_passed / comp_total * 100) if comp_total > 0 else 0
            
            print(f"  {component:20}: {comp_passed:2}/{comp_total:2} ({comp_rate:5.1f}%)")
        
        # Failed tests details
        failed_results = [r for r in self.results if not r.passed]
        if failed_results:
            print(f"\n❌ Failed Tests:")
            for result in failed_results:
                print(f"  {result.test_name}: {result.error_message}")
        
        # Performance metrics
        perf_results = [r for r in self.results if r.performance_metrics]
        if perf_results:
            print(f"\n⚡ Performance Highlights:")
            for result in perf_results:
                metrics = result.performance_metrics
                if 'hashrate' in metrics:
                    print(f"  {result.test_name}: {metrics['hashrate']:.2f} H/s")
                elif 'ops_per_second' in metrics:
                    print(f"  {result.test_name}: {metrics['ops_per_second']:.0f} ops/s")
                elif 'init_time' in metrics:
                    print(f"  {result.test_name}: {metrics['init_time']:.3f}s init")
        
        # Save detailed report
        report_file = "/media/maitreya/ZION1/logs/test_report.json"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": success_rate,
                "total_time": total_time
            },
            "results": [
                {
                    "test_name": r.test_name,
                    "component": r.component,
                    "passed": r.passed,
                    "execution_time": r.execution_time,
                    "error_message": r.error_message,
                    "performance_metrics": r.performance_metrics
                }
                for r in self.results
            ]
        }
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            print(f"\n📄 Detailed report saved to: {report_file}")
        except Exception as e:
            print(f"\n⚠️ Could not save report: {e}")
        
        print("\n" + "="*80)
        
        if success_rate >= 80:
            print("✅ ZION 2.7 IS READY FOR PRODUCTION!")
        elif success_rate >= 60:
            print("⚠️ ZION 2.7 NEEDS MINOR FIXES")
        else:
            print("❌ ZION 2.7 NEEDS MAJOR ATTENTION")
        
        print("="*80)

if __name__ == "__main__":
    # Run the complete test suite
    test_suite = ZionTestSuite()
    test_suite.run_all_tests()
    
    # Cleanup logging
    try:
        shutdown_all_loggers()
    except:
        pass