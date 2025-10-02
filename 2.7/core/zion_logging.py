#!/usr/bin/env python3
"""
📊 ZION 2.7 UNIFIED LOGGING SYSTEM 📊
Centralizovaný logging systém pro všechny ZION komponenty

Features:
- Structured logging with JSON output
- Performance metrics tracking
- Error aggregation and analysis
- Real-time monitoring capabilities
- Component-specific log levels
- Automatic log rotation
- GPU mining and AI afterburner integration
"""

import os
import sys
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import queue
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
import traceback

class LogLevel(Enum):
    """ZION logging levels"""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    MINING = 60  # Special level for mining operations
    AI = 70      # Special level for AI operations

class ComponentType(Enum):
    """ZION component types for structured logging"""
    BLOCKCHAIN = "blockchain"
    MINING = "mining"
    GPU_MINING = "gpu_mining"
    AI_AFTERBURNER = "ai_afterburner"
    PERFECT_MEMORY = "perfect_memory"
    NETWORK = "network"
    WALLET = "wallet"
    DEFI = "defi"
    EXCHANGE = "exchange"
    MOBILE = "mobile"
    FRONTEND = "frontend"
    BIO_AI = "bio_ai"
    TESTING = "testing"

@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: datetime
    level: str
    component: str
    message: str
    thread_id: int
    process_id: int
    extra_data: Dict[str, Any]
    performance_metrics: Optional[Dict[str, float]] = None
    error_info: Optional[Dict[str, str]] = None

class ZionLogger:
    """Enhanced ZION logging system"""
    
    def __init__(self, component: ComponentType, log_dir: str = None):
        self.component = component
        
        # Use relative path if not specified
        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create component-specific logger
        self.logger = logging.getLogger(f"zion.{component.value}")
        self.logger.setLevel(logging.DEBUG)
        
        # Performance tracking
        self.performance_data = {}
        self.error_count = 0
        self.warning_count = 0
        self.start_time = time.time()
        
        # Thread-safe queue for async logging
        self.log_queue = queue.Queue()
        self.db_lock = threading.Lock()
        
        # Setup handlers
        self._setup_handlers()
        self._setup_database()
        
        # Start background logging thread
        self.logging_thread = threading.Thread(target=self._background_logger, daemon=True)
        self.logging_thread.start()
    
    def _setup_handlers(self):
        """Setup logging handlers"""
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
        )
        
        # Console handler with colors
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self._get_colored_formatter())
        
        # File handler with rotation
        log_file = self.log_dir / f"zion_{self.component.value}.log"
        file_handler = RotatingFileHandler(
            log_file, maxBytes=50*1024*1024, backupCount=10
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # JSON handler for structured logs
        json_file = self.log_dir / f"zion_{self.component.value}.json"
        json_handler = RotatingFileHandler(
            json_file, maxBytes=100*1024*1024, backupCount=5
        )
        json_handler.setLevel(logging.DEBUG)
        json_handler.setFormatter(self._get_json_formatter())
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(json_handler)
    
    def _get_colored_formatter(self):
        """Get colored formatter for console output"""
        class ColoredFormatter(logging.Formatter):
            COLORS = {
                'DEBUG': '\033[36m',    # Cyan
                'INFO': '\033[32m',     # Green
                'WARNING': '\033[33m',  # Yellow
                'ERROR': '\033[31m',    # Red
                'CRITICAL': '\033[35m', # Magenta
                'MINING': '\033[94m',   # Blue
                'AI': '\033[95m',       # Pink
            }
            RESET = '\033[0m'
            
            def format(self, record):
                color = self.COLORS.get(record.levelname, '')
                record.levelname = f"{color}{record.levelname}{self.RESET}"
                return super().format(record)
        
        return ColoredFormatter('%(asctime)s | %(levelname)s | %(name)-20s | %(message)s')
    
    def _get_json_formatter(self):
        """Get JSON formatter for structured logging"""
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                    'level': record.levelname,
                    'component': record.name,
                    'message': record.getMessage(),
                    'thread_id': record.thread,
                    'process_id': record.process,
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno,
                }
                
                # Add extra data if present
                if hasattr(record, 'extra_data'):
                    log_entry['extra_data'] = record.extra_data
                
                if hasattr(record, 'performance_metrics'):
                    log_entry['performance_metrics'] = record.performance_metrics
                
                if record.exc_info:
                    log_entry['exception'] = self.formatException(record.exc_info)
                
                return json.dumps(log_entry)
        
        return JsonFormatter()
    
    def _setup_database(self):
        """Setup SQLite database for log storage"""
        db_file = self.log_dir / "zion_logs.db"
        
        with sqlite3.connect(str(db_file)) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL,
                    thread_id INTEGER,
                    process_id INTEGER,
                    extra_data TEXT,
                    performance_metrics TEXT,
                    error_info TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON logs(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_component ON logs(component)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_level ON logs(level)')
    
    def _background_logger(self):
        """Background thread for database logging"""
        while True:
            try:
                log_entry = self.log_queue.get(timeout=1)
                if log_entry is None:  # Shutdown signal
                    break
                
                self._store_to_database(log_entry)
                self.log_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Background logger error: {e}")
    
    def _store_to_database(self, log_entry: LogEntry):
        """Store log entry to database"""
        db_file = self.log_dir / "zion_logs.db"
        
        with self.db_lock:
            try:
                with sqlite3.connect(str(db_file)) as conn:
                    conn.execute('''
                        INSERT INTO logs (
                            timestamp, level, component, message, thread_id, process_id,
                            extra_data, performance_metrics, error_info
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        log_entry.timestamp.isoformat(),
                        log_entry.level,
                        log_entry.component,
                        log_entry.message,
                        log_entry.thread_id,
                        log_entry.process_id,
                        json.dumps(log_entry.extra_data) if log_entry.extra_data else None,
                        json.dumps(log_entry.performance_metrics) if log_entry.performance_metrics else None,
                        json.dumps(log_entry.error_info) if log_entry.error_info else None
                    ))
            except Exception as e:
                print(f"Database logging error: {e}")
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log(logging.INFO, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log(logging.DEBUG, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.warning_count += 1
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.error_count += 1
        # Auto-capture exception info
        if 'exc_info' not in kwargs:
            kwargs['exc_info'] = True
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.error_count += 1
        if 'exc_info' not in kwargs:
            kwargs['exc_info'] = True
        self._log(logging.CRITICAL, message, **kwargs)
    
    def mining(self, message: str, hashrate: float = None, **kwargs):
        """Special mining log level"""
        if hashrate:
            kwargs['performance_metrics'] = kwargs.get('performance_metrics', {})
            kwargs['performance_metrics']['hashrate'] = hashrate
        
        # Add custom level for mining
        mining_level = 25  # Between INFO and WARNING
        logging.addLevelName(mining_level, 'MINING')
        self._log(mining_level, f"⛏️  {message}", **kwargs)
    
    def ai(self, message: str, accuracy: float = None, **kwargs):
        """Special AI log level"""
        if accuracy:
            kwargs['performance_metrics'] = kwargs.get('performance_metrics', {})
            kwargs['performance_metrics']['accuracy'] = accuracy
        
        # Add custom level for AI
        ai_level = 35  # Between WARNING and ERROR
        logging.addLevelName(ai_level, 'AI')
        self._log(ai_level, f"🧠 {message}", **kwargs)
    
    def _log(self, level: int, message: str, **kwargs):
        """Internal logging method"""
        # Extract performance metrics and extra data
        performance_metrics = kwargs.pop('performance_metrics', None)
        extra_data = {k: v for k, v in kwargs.items() if k not in ['exc_info']}
        
        # Log to standard logger
        self.logger.log(level, message, **kwargs)
        
        # Create structured log entry for database
        log_entry = LogEntry(
            timestamp=datetime.now(),
            level=logging.getLevelName(level),
            component=self.component.value,
            message=message,
            thread_id=threading.current_thread().ident,
            process_id=os.getpid(),
            extra_data=extra_data if extra_data else None,
            performance_metrics=performance_metrics
        )
        
        # Add to queue for background processing
        try:
            self.log_queue.put_nowait(log_entry)
        except queue.Full:
            pass  # Drop log if queue is full
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        uptime = time.time() - self.start_time
        return {
            'component': self.component.value,
            'uptime_seconds': uptime,
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'queue_size': self.log_queue.qsize(),
            'performance_data': self.performance_data.copy()
        }
    
    def shutdown(self):
        """Graceful shutdown"""
        self.log_queue.put(None)  # Shutdown signal
        self.logging_thread.join(timeout=5)

# Global logger instances
_loggers: Dict[str, ZionLogger] = {}

def get_logger(component: ComponentType) -> ZionLogger:
    """Get or create logger for component"""
    if component.value not in _loggers:
        _loggers[component.value] = ZionLogger(component)
    return _loggers[component.value]

def shutdown_all_loggers():
    """Shutdown all active loggers"""
    for logger in _loggers.values():
        logger.shutdown()

# Convenience functions
def log_mining(message: str, hashrate: float = None, **kwargs):
    """Quick mining log"""
    logger = get_logger(ComponentType.MINING)
    logger.mining(message, hashrate=hashrate, **kwargs)

def log_ai(message: str, accuracy: float = None, **kwargs):
    """Quick AI log"""
    logger = get_logger(ComponentType.AI_AFTERBURNER)
    logger.ai(message, accuracy=accuracy, **kwargs)

def log_performance(component: ComponentType, metrics: Dict[str, float]):
    """Log performance metrics"""
    logger = get_logger(component)
    logger.info("Performance metrics", performance_metrics=metrics)

if __name__ == "__main__":
    # Test the logging system
    print("🧪 Testing ZION Unified Logging System...")
    
    # Test different components
    blockchain_logger = get_logger(ComponentType.BLOCKCHAIN)
    mining_logger = get_logger(ComponentType.MINING)
    ai_logger = get_logger(ComponentType.AI_AFTERBURNER)
    
    # Test different log levels
    blockchain_logger.info("Blockchain initialized")
    mining_logger.mining("Mining started", hashrate=1250.5)
    ai_logger.ai("AI optimization active", accuracy=0.95)
    
    # Test performance logging
    log_performance(ComponentType.GPU_MINING, {
        'hashrate': 720.0,
        'temperature': 65.5,
        'power_usage': 180.0
    })
    
    # Test error logging
    try:
        raise ValueError("Test error for logging")
    except Exception:
        blockchain_logger.error("Test exception occurred")
    
    time.sleep(1)  # Let background logger process
    
    # Print stats
    print("\n📊 Logging Statistics:")
    for component, logger in _loggers.items():
        stats = logger.get_stats()
        print(f"  {component}: {stats}")
    
    # Cleanup
    shutdown_all_loggers()
    print("\n✅ Logging system test completed!")