#!/usr/bin/env python3
"""
🛡️ ZION 2.7 UNIFIED ERROR HANDLING SYSTEM 🛡️
Robustní error handling, recovery mechanismy a monitoring

Features:
- Automated error detection and recovery
- Circuit breaker pattern implementation
- Health monitoring and alerting
- Graceful degradation strategies
- Error analytics and reporting
- Component restart mechanisms
- Resource leak prevention
"""

import os
import sys
import json
import time
import threading
import traceback
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import queue
import signal
import subprocess
from functools import wraps
import asyncio
import logging

# Add ZION paths
ZION_ROOT = "/media/maitreya/ZION1/2.7"
sys.path.insert(0, ZION_ROOT)

try:
    from core.zion_logging import get_logger, ComponentType
    logger = get_logger(ComponentType.TESTING)  # Use testing for error handling
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"           # Minor issues, continue operation
    MEDIUM = "medium"     # Moderate issues, may affect performance
    HIGH = "high"         # Serious issues, component degradation
    CRITICAL = "critical" # Critical failures, immediate action required
    FATAL = "fatal"       # System-breaking errors, shutdown required

class RecoveryStrategy(Enum):
    """Recovery strategy types"""
    IGNORE = "ignore"                    # Log and continue
    RETRY = "retry"                     # Retry operation
    RESTART_COMPONENT = "restart_component"  # Restart failing component
    GRACEFUL_DEGRADATION = "graceful_degradation"  # Reduce functionality
    FALLBACK = "fallback"               # Use backup system
    SHUTDOWN = "shutdown"               # Graceful shutdown

class ComponentState(Enum):
    """Component health states"""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"
    OFFLINE = "offline"

@dataclass
class ErrorInfo:
    """Structured error information"""
    error_id: str
    component: str
    error_type: str
    message: str
    severity: ErrorSeverity
    timestamp: datetime
    stack_trace: str
    context: Dict[str, Any]
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class ComponentHealth:
    """Component health status"""
    component: str
    state: ComponentState
    last_heartbeat: datetime
    error_count: int
    warning_count: int
    uptime: float
    performance_score: float
    resource_usage: Dict[str, float]
    last_error: Optional[ErrorInfo] = None

class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self.lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    self.failure_count = 0
                else:
                    raise Exception(f"Circuit breaker is OPEN. Try again in {self.recovery_timeout}s")
            
            try:
                result = func(*args, **kwargs)
                
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                
                return result
                
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                
                raise e

class ZionErrorHandler:
    """Unified error handling system for ZION 2.7"""
    
    def __init__(self, error_dir: str = None):
        # Use relative path if not specified
        if error_dir is None:
            error_dir = os.path.join(os.path.dirname(__file__), '..', 'errors')
        self.error_dir = Path(error_dir)
        self.error_dir.mkdir(exist_ok=True)
        
        # Error tracking
        self.errors: Dict[str, ErrorInfo] = {}
        self.component_health: Dict[str, ComponentHealth] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Recovery mechanisms
        self.recovery_handlers: Dict[str, Callable] = {}
        self.health_monitors: Dict[str, threading.Thread] = {}
        
        # Configuration
        self.max_errors_per_component = 100
        self.health_check_interval = 30
        self.error_retention_days = 7
        
        # Threading
        self.lock = threading.RLock()
        self.running = True
        self.error_queue = queue.Queue()
        
        # Start background services
        self._start_error_processor()
        self._start_health_monitor()
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers"""
        def signal_handler(signum, frame):
            logger.warning(f"Received signal {signum}, initiating graceful shutdown")
            self.shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _start_error_processor(self):
        """Start background error processing thread"""
        def process_errors():
            while self.running:
                try:
                    error_info = self.error_queue.get(timeout=1)
                    if error_info is None:  # Shutdown signal
                        break
                    
                    self._process_error(error_info)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in error processor: {e}")
        
        error_thread = threading.Thread(target=process_errors, daemon=True)
        error_thread.start()
    
    def _start_health_monitor(self):
        """Start component health monitoring"""
        def monitor_health():
            while self.running:
                try:
                    self._check_all_components()
                    time.sleep(self.health_check_interval)
                except Exception as e:
                    logger.error(f"Error in health monitor: {e}")
        
        health_thread = threading.Thread(target=monitor_health, daemon=True)
        health_thread.start()
    
    def register_component(self, component: str, health_check: Optional[Callable] = None):
        """Register component for monitoring"""
        with self.lock:
            self.component_health[component] = ComponentHealth(
                component=component,
                state=ComponentState.HEALTHY,
                last_heartbeat=datetime.now(),
                error_count=0,
                warning_count=0,
                uptime=0,
                performance_score=1.0,
                resource_usage={}
            )
            
            # Create circuit breaker for component
            self.circuit_breakers[component] = CircuitBreaker()
            
            # Register health check function
            if health_check:
                self.recovery_handlers[f"{component}_health_check"] = health_check
    
    def handle_error(self, component: str, error: Exception, 
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    context: Dict[str, Any] = None,
                    recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY):
        """Handle error with automatic recovery"""
        
        error_id = f"{component}_{int(time.time())}_{hash(str(error)) % 10000}"
        
        error_info = ErrorInfo(
            error_id=error_id,
            component=component,
            error_type=type(error).__name__,
            message=str(error),
            severity=severity,
            timestamp=datetime.now(),
            stack_trace=traceback.format_exc(),
            context=context or {},
            recovery_strategy=recovery_strategy
        )
        
        # Queue for background processing
        try:
            self.error_queue.put_nowait(error_info)
        except queue.Full:
            # If queue is full, process immediately
            self._process_error(error_info)
        
        # Update component health
        self._update_component_health(component, error_info)
        
        # Log error
        logger.error(f"[{component}] {severity.value.upper()}: {error_info.message}")
        
        return error_id
    
    def _process_error(self, error_info: ErrorInfo):
        """Process error with recovery strategy"""
        with self.lock:
            self.errors[error_info.error_id] = error_info
            
            # Apply recovery strategy
            try:
                if error_info.recovery_strategy == RecoveryStrategy.RETRY:
                    self._attempt_retry(error_info)
                elif error_info.recovery_strategy == RecoveryStrategy.RESTART_COMPONENT:
                    self._restart_component(error_info.component)
                elif error_info.recovery_strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                    self._apply_graceful_degradation(error_info.component)
                elif error_info.recovery_strategy == RecoveryStrategy.FALLBACK:
                    self._activate_fallback(error_info.component)
                elif error_info.recovery_strategy == RecoveryStrategy.SHUTDOWN:
                    self._initiate_shutdown(error_info)
                
            except Exception as e:
                logger.error(f"Recovery strategy failed: {e}")
        
        # Save error to disk
        self._save_error_report(error_info)
    
    def _attempt_retry(self, error_info: ErrorInfo):
        """Attempt to retry failed operation"""
        error_info.recovery_attempts += 1
        
        if error_info.recovery_attempts >= error_info.max_recovery_attempts:
            logger.error(f"Max retry attempts reached for {error_info.error_id}")
            error_info.recovery_strategy = RecoveryStrategy.RESTART_COMPONENT
            self._restart_component(error_info.component)
        else:
            logger.info(f"Retrying operation for {error_info.component} (attempt {error_info.recovery_attempts})")
    
    def _restart_component(self, component: str):
        """Restart failed component"""
        logger.warning(f"Restarting component: {component}")
        
        # Update component state
        if component in self.component_health:
            self.component_health[component].state = ComponentState.RECOVERING
        
        # Call component-specific restart handler
        restart_handler = self.recovery_handlers.get(f"{component}_restart")
        if restart_handler:
            try:
                restart_handler()
                logger.info(f"Component {component} restarted successfully")
                
                # Update state to healthy
                if component in self.component_health:
                    self.component_health[component].state = ComponentState.HEALTHY
                    
            except Exception as e:
                logger.error(f"Failed to restart component {component}: {e}")
                if component in self.component_health:
                    self.component_health[component].state = ComponentState.FAILED
    
    def _apply_graceful_degradation(self, component: str):
        """Apply graceful degradation for component"""
        logger.warning(f"Applying graceful degradation for {component}")
        
        # Update component state
        if component in self.component_health:
            self.component_health[component].state = ComponentState.DEGRADED
        
        # Call degradation handler
        degrade_handler = self.recovery_handlers.get(f"{component}_degrade")
        if degrade_handler:
            try:
                degrade_handler()
                logger.info(f"Graceful degradation applied for {component}")
            except Exception as e:
                logger.error(f"Failed to apply degradation for {component}: {e}")
    
    def _activate_fallback(self, component: str):
        """Activate fallback system for component"""
        logger.warning(f"Activating fallback for {component}")
        
        fallback_handler = self.recovery_handlers.get(f"{component}_fallback")
        if fallback_handler:
            try:
                fallback_handler()
                logger.info(f"Fallback activated for {component}")
            except Exception as e:
                logger.error(f"Failed to activate fallback for {component}: {e}")
    
    def _initiate_shutdown(self, error_info: ErrorInfo):
        """Initiate graceful shutdown due to critical error"""
        logger.critical(f"Initiating shutdown due to critical error: {error_info.message}")
        
        # Save all error reports
        self._save_error_summary()
        
        # Signal shutdown
        self.running = False
    
    def _update_component_health(self, component: str, error_info: ErrorInfo):
        """Update component health status"""
        if component not in self.component_health:
            self.register_component(component)
        
        health = self.component_health[component]
        health.last_error = error_info
        
        if error_info.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]:
            health.error_count += 1
            health.state = ComponentState.FAILED
        elif error_info.severity == ErrorSeverity.HIGH:
            health.error_count += 1
            health.state = ComponentState.DEGRADED
        elif error_info.severity in [ErrorSeverity.MEDIUM, ErrorSeverity.LOW]:
            health.warning_count += 1
            if health.state == ComponentState.HEALTHY:
                health.state = ComponentState.WARNING
    
    def _check_all_components(self):
        """Check health of all registered components"""
        for component, health in self.component_health.items():
            try:
                # Check if component is responding
                health_check = self.recovery_handlers.get(f"{component}_health_check")
                if health_check:
                    is_healthy = health_check()
                    if is_healthy:
                        health.last_heartbeat = datetime.now()
                        if health.state in [ComponentState.WARNING, ComponentState.DEGRADED]:
                            # Component recovered
                            health.state = ComponentState.HEALTHY
                    else:
                        # Component not responding
                        if health.state == ComponentState.HEALTHY:
                            health.state = ComponentState.WARNING
                
                # Check resource usage
                self._check_resource_usage(component, health)
                
            except Exception as e:
                logger.error(f"Health check failed for {component}: {e}")
                health.state = ComponentState.FAILED
    
    def _check_resource_usage(self, component: str, health: ComponentHealth):
        """Check component resource usage"""
        try:
            # Get process info if available
            current_process = psutil.Process()
            
            health.resource_usage = {
                'cpu_percent': current_process.cpu_percent(),
                'memory_mb': current_process.memory_info().rss / 1024 / 1024,
                'open_files': len(current_process.open_files()),
                'connections': len(current_process.connections())
            }
            
            # Check for resource leaks
            if health.resource_usage['memory_mb'] > 1000:  # 1GB threshold
                logger.warning(f"High memory usage detected for {component}: {health.resource_usage['memory_mb']:.2f} MB")
            
            if health.resource_usage['open_files'] > 100:
                logger.warning(f"High file descriptor usage for {component}: {health.resource_usage['open_files']}")
                
        except Exception as e:
            logger.debug(f"Could not check resource usage for {component}: {e}")
    
    def _save_error_report(self, error_info: ErrorInfo):
        """Save error report to disk"""
        try:
            error_file = self.error_dir / f"error_{error_info.error_id}.json"
            with open(error_file, 'w') as f:
                json.dump(asdict(error_info), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save error report: {e}")
    
    def _save_error_summary(self):
        """Save comprehensive error summary"""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'total_errors': len(self.errors),
                'component_health': {k: asdict(v) for k, v in self.component_health.items()},
                'recent_errors': [asdict(error) for error in list(self.errors.values())[-10:]]
            }
            
            summary_file = self.error_dir / f"error_summary_{int(time.time())}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save error summary: {e}")
    
    def register_recovery_handler(self, component: str, handler_type: str, handler: Callable):
        """Register recovery handler for component"""
        key = f"{component}_{handler_type}"
        self.recovery_handlers[key] = handler
    
    def get_component_health(self, component: str = None) -> Union[ComponentHealth, Dict[str, ComponentHealth]]:
        """Get component health status"""
        if component:
            return self.component_health.get(component)
        return self.component_health.copy()
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary statistics"""
        with self.lock:
            return {
                'total_errors': len(self.errors),
                'errors_by_severity': {
                    severity.value: sum(1 for e in self.errors.values() if e.severity == severity)
                    for severity in ErrorSeverity
                },
                'errors_by_component': {
                    component: sum(1 for e in self.errors.values() if e.component == component)
                    for component in set(e.component for e in self.errors.values())
                },
                'unresolved_errors': sum(1 for e in self.errors.values() if not e.resolved)
            }
    
    def shutdown(self):
        """Graceful shutdown of error handler"""
        logger.info("Shutting down error handling system")
        self.running = False
        
        # Save final error summary
        self._save_error_summary()
        
        # Stop processing
        try:
            self.error_queue.put(None)  # Shutdown signal
        except:
            pass

# Decorator functions
def handle_errors(component: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 recovery: RecoveryStrategy = RecoveryStrategy.RETRY):
    """Decorator for automatic error handling"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler = get_error_handler()
                error_handler.handle_error(component, e, severity, recovery_strategy=recovery)
                raise
        return wrapper
    return decorator

def circuit_breaker(component: str):
    """Decorator for circuit breaker pattern"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = get_error_handler()
            breaker = error_handler.circuit_breakers.get(component)
            if not breaker:
                breaker = CircuitBreaker()
                error_handler.circuit_breakers[component] = breaker
            
            return breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator

# Global error handler instance
_error_handler: Optional[ZionErrorHandler] = None

def get_error_handler() -> ZionErrorHandler:
    """Get global error handler instance"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ZionErrorHandler()
    return _error_handler

if __name__ == "__main__":
    # Test error handling system
    print("🧪 Testing ZION Error Handling System...")
    
    error_handler = get_error_handler()
    
    # Register test component
    def test_health_check():
        return True
    
    def test_restart():
        print("Test component restarted")
    
    error_handler.register_component("test_component", test_health_check)
    error_handler.register_recovery_handler("test_component", "restart", test_restart)
    
    # Test error handling
    try:
        raise ValueError("Test error")
    except Exception as e:
        error_id = error_handler.handle_error("test_component", e, ErrorSeverity.HIGH)
        print(f"Error handled: {error_id}")
    
    # Test decorator
    @handle_errors("test_component", ErrorSeverity.MEDIUM)
    def test_function():
        raise RuntimeError("Decorator test error")
    
    try:
        test_function()
    except:
        pass
    
    # Print summary
    print("\n📊 Error Summary:")
    summary = error_handler.get_error_summary()
    print(json.dumps(summary, indent=2))
    
    # Print component health
    print("\n💚 Component Health:")
    health = error_handler.get_component_health()
    for comp, h in health.items():
        print(f"  {comp}: {h.state.value} (errors: {h.error_count})")
    
    print("\n✅ Error handling system test completed!")
    error_handler.shutdown()