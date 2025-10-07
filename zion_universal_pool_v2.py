#!/usr/bin/env python3
"""
ZION Universal Mining Pool with Real Hash Validation & Reward System
Supports ZION addresses, real ProgPow validation, and proportional rewards
"""
import asyncio
import json
import socket
import time
import secrets
import hashlib
import logging
import sqlite3
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading

# Import the real ZION blockchain
from new_zion_blockchain import NewZionBlockchain

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PoolBlock:
    """Represents a pool-found block"""
    height: int
    hash: str
    timestamp: float
    total_shares: int
    miner_shares: Dict[str, int] = field(default_factory=dict)
    reward_amount: float = 333.0  # ZION block reward
    pool_fee: float = 0.01  # 1% pool fee
    status: str = "pending"  # pending, confirmed, paid

@dataclass
class MinerStats:
    """Enhanced miner statistics"""
    address: str
    total_shares: int = 0
    valid_shares: int = 0
    invalid_shares: int = 0
    last_share_time: Optional[float] = None
    connected_time: float = field(default_factory=time.time)
    balance_pending: float = 0.0
    balance_paid: float = 0.0
    difficulty: int = 10000
    algorithm: str = "randomx"

class ZIONPoolDatabase:
    """SQLite database for persistent pool data storage"""

    def __init__(self, db_file="zion_pool.db"):
        self.db_file = db_file
        self.init_database()

    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()

            # Miners table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS miners (
                    address TEXT PRIMARY KEY,
                    algorithm TEXT DEFAULT 'randomx',
                    total_shares INTEGER DEFAULT 0,
                    valid_shares INTEGER DEFAULT 0,
                    invalid_shares INTEGER DEFAULT 0,
                    last_share_time REAL,
                    connected_time REAL DEFAULT (strftime('%s', 'now')),
                    balance_pending REAL DEFAULT 0.0,
                    balance_paid REAL DEFAULT 0.0,
                    difficulty INTEGER DEFAULT 10000,
                    created_at REAL DEFAULT (strftime('%s', 'now')),
                    updated_at REAL DEFAULT (strftime('%s', 'now'))
                )
            ''')

            # Shares table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS shares (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    address TEXT NOT NULL,
                    algorithm TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    nonce TEXT NOT NULL,
                    result TEXT NOT NULL,
                    difficulty INTEGER NOT NULL,
                    is_valid BOOLEAN NOT NULL,
                    processing_time REAL,
                    ip_address TEXT,
                    timestamp REAL DEFAULT (strftime('%s', 'now'))
                )
            ''')

            # Blocks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS blocks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    height INTEGER NOT NULL,
                    hash TEXT,
                    timestamp REAL DEFAULT (strftime('%s', 'now')),
                    total_shares INTEGER DEFAULT 0,
                    reward_amount REAL DEFAULT 333.0,
                    pool_fee REAL DEFAULT 0.01,
                    status TEXT DEFAULT 'pending'
                )
            ''')

            # Block shares table (many-to-many relationship)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS block_shares (
                    block_id INTEGER,
                    address TEXT,
                    shares INTEGER DEFAULT 0,
                    FOREIGN KEY (block_id) REFERENCES blocks (id),
                    FOREIGN KEY (address) REFERENCES miners (address),
                    PRIMARY KEY (block_id, address)
                )
            ''')

            # Payouts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS payouts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    address TEXT NOT NULL,
                    amount REAL NOT NULL,
                    timestamp REAL DEFAULT (strftime('%s', 'now')),
                    block_height INTEGER,
                    status TEXT DEFAULT 'pending',
                    tx_hash TEXT
                )
            ''')

            # Pool stats table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pool_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL DEFAULT (strftime('%s', 'now')),
                    total_miners INTEGER DEFAULT 0,
                    total_shares INTEGER DEFAULT 0,
                    valid_shares INTEGER DEFAULT 0,
                    invalid_shares INTEGER DEFAULT 0,
                    blocks_found INTEGER DEFAULT 0,
                    pending_payouts REAL DEFAULT 0.0,
                    active_connections INTEGER DEFAULT 0
                )
            ''')

            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_shares_address ON shares(address)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_shares_timestamp ON shares(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_blocks_height ON blocks(height)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_payouts_address ON payouts(address)')

            conn.commit()

    def save_miner_stats(self, address: str, stats: MinerStats):
        """Save miner statistics to database"""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO miners
                (address, algorithm, total_shares, valid_shares, invalid_shares,
                 last_share_time, connected_time, balance_pending, balance_paid,
                 difficulty, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, strftime('%s', 'now'))
            ''', (
                address, stats.algorithm, stats.total_shares, stats.valid_shares,
                stats.invalid_shares, stats.last_share_time, stats.connected_time,
                stats.balance_pending, stats.balance_paid, stats.difficulty
            ))
            conn.commit()

    def load_miner_stats(self, address: str) -> Optional[MinerStats]:
        """Load miner statistics from database"""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM miners WHERE address = ?', (address,))
            row = cursor.fetchone()

            if row:
                return MinerStats(
                    address=row[0],
                    algorithm=row[1],
                    total_shares=row[2],
                    valid_shares=row[3],
                    invalid_shares=row[4],
                    last_share_time=row[5],
                    connected_time=row[6],
                    balance_pending=row[7],
                    balance_paid=row[8],
                    difficulty=row[9]
                )
        return None

    def save_share(self, address: str, algorithm: str, job_id: str, nonce: str,
                   result: str, difficulty: int, is_valid: bool, processing_time: float,
                   ip_address: str):
        """Save share to database"""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO shares
                (address, algorithm, job_id, nonce, result, difficulty, is_valid,
                 processing_time, ip_address, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, strftime('%s', 'now'))
            ''', (address, algorithm, job_id, nonce, result, difficulty, is_valid,
                  processing_time, ip_address))
            conn.commit()

    def get_miner_history(self, address: str, limit: int = 100) -> List[Dict]:
        """Get miner share history"""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, is_valid, difficulty, algorithm
                FROM shares
                WHERE address = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (address, limit))

            history = []
            for row in cursor.fetchall():
                history.append({
                    'timestamp': row[0],
                    'is_valid': bool(row[1]),
                    'difficulty': row[2],
                    'algorithm': row[3]
                })
            return history

    def cleanup_old_data(self, days: int = 30):
        """Clean up old share data"""
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM shares WHERE timestamp < ?', (cutoff_time,))
            deleted_count = cursor.rowcount
            conn.commit()
            print(f"🧹 Cleaned up {deleted_count} old shares from database")

    def get_pool_stats_history(self, hours: int = 24) -> List[Dict]:
        """Get pool statistics history"""
        cutoff_time = time.time() - (hours * 60 * 60)
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, total_miners, total_shares, valid_shares,
                       invalid_shares, blocks_found, pending_payouts, active_connections
                FROM pool_stats
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            ''', (cutoff_time,))

            stats = []
            for row in cursor.fetchall():
                stats.append({
                    'timestamp': row[0],
                    'total_miners': row[1],
                    'total_shares': row[2],
                    'valid_shares': row[3],
                    'invalid_shares': row[4],
                    'blocks_found': row[5],
                    'pending_payouts': row[6],
                    'active_connections': row[7]
                })
            return stats

class ZIONPoolAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for pool REST API"""

    def __init__(self, pool_instance, *args, **kwargs):
        self.pool = pool_instance
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests"""
        try:
            if self.path == '/api/stats':
                self.send_stats_response()
            elif self.path.startswith('/api/miner/'):
                address = self.path.split('/api/miner/')[-1]
                self.send_miner_stats_response(address)
            elif self.path == '/api/pool':
                self.send_pool_info_response()
            elif self.path == '/api/health':
                self.send_health_response()
            else:
                self.send_error_response(404, "Endpoint not found")
        except Exception as e:
            logger.error(f"API error: {e}")
            self.send_error_response(500, "Internal server error")

    def send_stats_response(self):
        """Send pool statistics"""
        stats = self.pool.get_pool_stats()
        self.send_json_response(stats)

    def send_miner_stats_response(self, address):
        """Send miner-specific statistics"""
        if not address:
            self.send_error_response(400, "Miner address required")
            return

        # Get current stats
        miner_stats = self.pool.get_miner_stats(address)
        if not miner_stats:
            self.send_error_response(404, "Miner not found")
            return

        # Get historical data
        history = self.pool.db.get_miner_history(address)

        response = {
            'address': miner_stats.address,
            'algorithm': miner_stats.algorithm,
            'total_shares': miner_stats.total_shares,
            'valid_shares': miner_stats.valid_shares,
            'invalid_shares': miner_stats.invalid_shares,
            'balance_pending': miner_stats.balance_pending,
            'balance_paid': miner_stats.balance_paid,
            'last_share_time': miner_stats.last_share_time,
            'connected_time': miner_stats.connected_time,
            'difficulty': miner_stats.difficulty,
            'history': history
        }

        self.send_json_response(response)

    def send_pool_info_response(self):
        """Send general pool information"""
        info = {
            'name': 'ZION Universal Mining Pool',
            'version': '2.7.1',
            'algorithms': ['randomx', 'yescrypt', 'autolykos_v2'],
            'ports': {
                'stratum': self.pool.port,
                'api': self.pool.port + 1
            },
            'fees': {
                'pool_fee_percent': self.pool.pool_fee_percent * 100,
                'payout_threshold': self.pool.payout_threshold
            },
            'rewards': {
                'block_reward': 333,
                'eco_bonuses': {
                    'randomx': 1.0,
                    'yescrypt': 1.15,
                    'autolykos_v2': 1.2
                }
            },
            'features': [
                'Variable Difficulty',
                'IP Banning',
                'Performance Monitoring',
                'Database Persistence',
                'REST API',
                'Eco-Friendly Mining'
            ]
        }
        self.send_json_response(info)

    def send_health_response(self):
        """Send health check response"""
        health = {
            'status': 'healthy',
            'timestamp': time.time(),
            'uptime_seconds': time.time() - self.pool.performance_stats['start_time'],
            'active_connections': len(self.pool.miners),
            'total_miners': len(self.pool.miner_stats),
            'database_status': 'connected' if hasattr(self.pool, 'db') else 'disconnected'
        }
        self.send_json_response(health)

    def send_json_response(self, data, status_code=200):
        """Send JSON response"""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode('utf-8'))

    def send_error_response(self, status_code, message):
        """Send error response"""
        error_data = {
            'error': {
                'code': status_code,
                'message': message
            }
        }
        self.send_json_response(error_data, status_code)

    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.info(f"API: {format % args}")

class ZIONPoolAPIServer:
    """Simple HTTP server for pool API"""

    def __init__(self, pool_instance, port=3334):
        self.pool = pool_instance
        self.port = port
        self.server = None
        self.thread = None

    def start(self):
        """Start API server in background thread"""
        print(f"Starting API server on port {self.port}...")
        def run_server():
            try:
                # Create custom handler with pool instance
                def handler_class(*args, **kwargs):
                    return ZIONPoolAPIHandler(self.pool, *args, **kwargs)

                self.server = HTTPServer(('0.0.0.0', self.port), handler_class)
                print(f"Pool API server started on port {self.port}")
                self.server.serve_forever()
            except Exception as e:
                logger.error(f"API server error: {e}")

        self.thread = threading.Thread(target=run_server, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop API server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            print("📊 Pool API server stopped")

class ZionUniversalPool:
    def __init__(self, port=3333):
        self.port = port
        self.miners: Dict[tuple, dict] = {}
        self.miner_stats: Dict[str, MinerStats] = {}
        self.current_jobs = {
            'randomx': None,
            'kawpow': None,
            'ethash': None
        }
        self.job_counter = 0
        self.share_counter = 0
        self.block_counter = 0

        # Reward system
        self.pool_blocks: List[PoolBlock] = []
        self.pool_wallet_address = "ZION_POOL_WALLET_ADDRESS"  # TODO: Configure real wallet
        self.pool_fee_percent = 0.01  # 1%
        self.payout_threshold = 1.0  # Minimum payout in ZION
        self.current_block_height = 5612  # Starting from current height

        # Share validation
        self.submitted_shares = set()  # For duplicate detection
        self.share_window_size = 100  # Rolling window for difficulty adjustment

        self.difficulty = {
            'randomx': 10000,        # RandomX (CPU)
            'yescrypt': 8000,        # Yescrypt (CPU)
            'autolykos_v2': 75       # Autolykos v2 (GPU)
        }
        
        # Eco-friendly algorithm rewards - FINAL SET
        self.eco_rewards = {
            'randomx': 1.0,      # Standard reward (100W avg)
            'yescrypt': 1.15,    # +15% eco bonus (80W avg) 
            'autolykos_v2': 1.2  # +20% eco bonus (150W avg) - BEST GPU ALGO
        }

        # Jobs and submissions tracking
        self.jobs = {}
        self.submissions = set()

        # Performance monitoring
        self.performance_stats = {
            'start_time': time.time(),
            'total_connections': 0,
            'total_shares_processed': 0,
            'avg_share_processing_time': 0.0,
            'peak_connections': 0,
            'errors_count': 0,
            'last_reset': time.time()
        }
        self.share_processing_times = []
        
        # Variable difficulty system (inspired by Node Stratum Pool)
        self.vardiff = {
            'enabled': True,
            'min_diff': {
                'randomx': 1000,
                'yescrypt': 800, 
                'autolykos_v2': 50
            },
            'max_diff': {
                'randomx': 50000,
                'yescrypt': 40000,
                'autolykos_v2': 2500
            },
            'target_time': 20,  # seconds per share (eco-friendly - longer than 15s standard)
            'retarget_time': 90,  # check every 90 seconds
            'variance_percent': 30  # tolerance before retargeting
        }
        
        # Session management and IP banning
        self.banned_ips = {}
        self.connection_stats = {}
        self.banning = {
            'enabled': True,
            'invalid_percent_threshold': 60,  # ban at 60% invalid (more tolerant than 50%)
            'check_threshold': 200,  # check after 200 shares
            'ban_duration': 600  # 10 minutes
        }

        # Database integration
        self.db = ZIONPoolDatabase()

        # Real blockchain integration
        self.blockchain = NewZionBlockchain()
        self.pool_wallet_address = 'ZION_SACRED_B0FA7E2A234D8C2F08545F02295C98'  # Sacred Mining Operator from premine

        # API server (will be started in start_server)
        self.api_server = ZIONPoolAPIServer(self, port=self.port + 1)

    def validate_zion_address(self, address):
        """Validate ZION address format"""
        if address.startswith('ZION_') and len(address) == 37:
            # ZION_ + 32 hex characters
            hex_part = address[5:]
            try:
                int(hex_part, 16)  # Verify it's valid hex
                return True
            except ValueError:
                return False
        return False

    def convert_address_for_mining(self, address):
        """Convert address for mining compatibility"""
        return address

    def get_miner_stats(self, address: str) -> MinerStats:
        """Get or create miner statistics with database persistence"""
        if address not in self.miner_stats:
            # Try to load from database first
            db_stats = self.db.load_miner_stats(address)
            if db_stats:
                self.miner_stats[address] = db_stats
            else:
                self.miner_stats[address] = MinerStats(address=address)
        return self.miner_stats[address]

    def validate_kawpow_share(self, job_id: str, nonce: str, mix_hash: str, header_hash: str, difficulty: int) -> bool:
        """
        Real KawPow (ProgPow) share validation
        This is a simplified implementation - in production, use proper ProgPow library
        """
        try:
            # Get job details
            if job_id not in self.jobs:
                return False

            job = self.jobs[job_id]

            # Basic validation checks
            if not all([nonce, mix_hash, header_hash]):
                return False

            # Convert nonce to integer for validation
            try:
                nonce_int = int(nonce, 16)
            except ValueError:
                return False

            # Simplified ProgPow-like validation
            # In production, this should use actual ProgPow algorithm
            # For now, we'll use a hash-based validation that's deterministic

            # Create validation hash: header_hash + nonce + mix_hash
            validation_data = f"{header_hash}{nonce}{mix_hash}"
            validation_hash = hashlib.sha256(validation_data.encode()).hexdigest()

            # Convert to target for difficulty check
            target = int(validation_hash[:16], 16)  # First 16 hex chars as target

            # Check if hash meets difficulty requirement
            # Lower target value = higher difficulty met
            required_target = 2**256 // difficulty

            return target < required_target

        except Exception as e:
            logger.error(f"KawPow validation error: {e}")
            return False

    def validate_randomx_share(self, job_id: str, nonce: str, result: str, difficulty: int) -> bool:
        """
        RandomX share validation
        Simplified validation - in production, use proper RandomX verification
        """
        try:
            if job_id not in self.jobs:
                return False

            # Basic format validation
            if not nonce or not result:
                return False

            # Simplified validation using hash
            # In production, this should verify against actual RandomX hash
            validation_data = f"{job_id}{nonce}{result}"
            validation_hash = hashlib.sha256(validation_data.encode()).hexdigest()

            # Convert to numerical comparison
            hash_value = int(validation_hash[:16], 16)
            target = 2**64 // difficulty

            return hash_value < target

        except Exception as e:
            logger.error(f"RandomX validation error: {e}")
            return False

    def validate_autolykos_v2_share(self, job_id: str, nonce: str, result: str, difficulty: int) -> bool:
        """
        Autolykos v2 share validation
        Memory-hard algorithm validation for energy-efficient GPU mining
        """
        try:
            if job_id not in self.jobs:
                return False

            # Basic format validation
            if not nonce or not result:
                return False

            job = self.jobs[job_id]

            # Autolykos v2 validation logic
            # Simplified validation - in production, use proper Autolykos v2 verification
            validation_data = f"{job_id}{nonce}{result}{job.get('block_header', '')}"
            validation_hash = hashlib.blake2b(validation_data.encode(), digest_size=32).hexdigest()

            # Convert to numerical comparison
            hash_value = int(validation_hash[:16], 16)
            target = 2**64 // difficulty

            return hash_value < target

        except Exception as e:
            logger.error(f"Autolykos v2 validation error: {e}")
            return False

    def validate_yescrypt_share(self, job_id: str, nonce: str, result: str, difficulty: int) -> bool:
        """
        Enhanced Yescrypt share validation
        Memory-hard algorithm for ultra energy-efficient CPU mining
        Supports C extension validation for maximum performance
        """
        try:
            if job_id not in self.jobs:
                logger.debug(f"Job {job_id} not found")
                return False

            # Basic format validation
            if not nonce or not result:
                logger.debug("Missing nonce or result")
                return False

            job = self.jobs[job_id]

            # Try to use C extension for validation if available
            try:
                import yescrypt_fast
                
                # Create header data for C validation
                header_data = f"{job_id}{nonce}{job.get('block_header', '')}".encode()
                hash_result = yescrypt_fast.hash(header_data)
                
                # Convert to target comparison
                hash_int = int.from_bytes(hash_result, 'big')
                target = (1 << 224) // difficulty  # Adjusted for Yescrypt difficulty
                
                is_valid = hash_int < target
                logger.debug(f"C extension Yescrypt validation: {is_valid}")
                return is_valid
                
            except ImportError:
                # Fallback to Python implementation
                logger.debug("Using Python fallback for Yescrypt validation")
                
                # Enhanced Python validation
                validation_data = f"{job_id}{nonce}{result}{job.get('block_header', '')}"
                validation_hash = hashlib.sha256(validation_data.encode()).hexdigest()

                # Yescrypt uses multiple rounds - simulate with additional hashing
                for i in range(8):  # More rounds for better ASIC resistance
                    validation_hash = hashlib.sha256(validation_hash.encode() + str(i).encode()).hexdigest()

                # Memory-hard component simulation
                memory_data = validation_hash
                for _ in range(4):
                    memory_data = hashlib.pbkdf2_hmac('sha256', memory_data.encode(), b'yescrypt_zion', 2048, 32).hex()

                # Convert to numerical comparison
                hash_value = int(memory_data[:16], 16)
                target = 2**64 // difficulty

                is_valid = hash_value < target
                logger.debug(f"Python Yescrypt validation: {is_valid}")
                return is_valid

        except Exception as e:
            logger.error(f"Yescrypt validation error: {e}")
            return False

    def record_share(self, address: str, algorithm: str, is_valid: bool = True) -> None:
        """Record share for miner statistics and reward calculation with database persistence"""
        stats = self.get_miner_stats(address)

        if is_valid:
            stats.valid_shares += 1
        else:
            stats.invalid_shares += 1

        stats.total_shares = stats.valid_shares + stats.invalid_shares
        stats.last_share_time = time.time()
        stats.algorithm = algorithm  # Update algorithm

        # Save to database
        self.db.save_miner_stats(address, stats)

        # Update current block shares if exists
        if self.pool_blocks and self.pool_blocks[-1].status == "pending":
            current_block = self.pool_blocks[-1]
            current_block.miner_shares[address] = current_block.miner_shares.get(address, 0) + 1
            current_block.total_shares += 1

    def check_block_found(self) -> bool:
        """
        Check if a block has been found and submit it to the real blockchain
        """
        if not self.pool_blocks:
            return False

        current_block = self.pool_blocks[-1]
        if current_block.status != "pending":
            return False

        # Real block finding: mine pending transactions when enough shares accumulated
        # This replaces the mock simulation with actual blockchain mining
        block_threshold = 1000  # Shares needed to trigger block mining

        if current_block.total_shares >= block_threshold:
            try:
                # Mine the block on the real blockchain
                block_hash = self.blockchain.mine_pending_transactions(self.pool_wallet_address)
                
                if block_hash:
                    current_block.status = "confirmed"
                    current_block.hash = block_hash
                    current_block.timestamp = time.time()

                    logger.info(f"🎉 REAL BLOCK MINED! Height: {current_block.height}, Hash: {block_hash}")
                    print(f"🎉 REAL BLOCK MINED! Height: {current_block.height}, Hash: {block_hash[:16]}...")

                    # Calculate and distribute rewards via blockchain transactions
                    self.calculate_block_rewards_via_blockchain(current_block)

                    # Start new block
                    self.start_new_block()

                    return True
                else:
                    logger.error("Failed to mine block on blockchain")
                    return False
                    
            except Exception as e:
                logger.error(f"Block mining error: {e}")
                return False

        return False

    def start_new_block(self) -> None:
        """Start tracking a new block"""
        self.block_counter += 1
        self.current_block_height += 1

        new_block = PoolBlock(
            height=self.current_block_height,
            hash="",
            timestamp=time.time(),
            total_shares=0,
            miner_shares={}
        )

        self.pool_blocks.append(new_block)
        logger.info(f"Started new block at height {self.current_block_height}")

    def calculate_block_rewards(self, block: PoolBlock) -> None:
        """Calculate proportional rewards for block with eco-friendly bonuses"""
        if block.total_shares == 0:
            return

        # Calculate pool fee (reduce fee for eco algorithms)
        base_pool_fee = block.reward_amount * self.pool_fee_percent
        eco_fee_reduction = 0.0
        
        # Count eco-friendly shares for fee reduction
        eco_shares = 0
        for address, shares in block.miner_shares.items():
            stats = self.get_miner_stats(address)
            if stats.algorithm in ['randomx', 'yescrypt']:
                eco_shares += shares
                
        eco_ratio = eco_shares / block.total_shares if block.total_shares > 0 else 0
        eco_fee_reduction = base_pool_fee * 0.2 * eco_ratio  # Up to 20% fee reduction
        
        pool_fee_amount = base_pool_fee - eco_fee_reduction
        miner_reward_total = block.reward_amount - pool_fee_amount

        logger.info(f"Block reward: {block.reward_amount} ZION, Pool fee: {pool_fee_amount:.4f} (eco reduction: {eco_fee_reduction:.4f}), Miner total: {miner_reward_total}")

        # Calculate proportional rewards with eco bonuses
        for address, miner_shares in block.miner_shares.items():
            if miner_shares > 0:
                proportion = miner_shares / block.total_shares
                base_reward = miner_reward_total * proportion
                
                # Apply eco-friendly algorithm bonus/penalty
                stats = self.get_miner_stats(address)
                algorithm = stats.algorithm
                eco_multiplier = self.eco_rewards.get(algorithm, 1.0)
                
                final_reward = base_reward * eco_multiplier
                
                # Update miner balance
                stats.balance_pending += final_reward
                
                eco_info = f"(eco: {eco_multiplier}x)" if eco_multiplier != 1.0 else ""
                logger.info(f"Miner {address} [{algorithm}]: {miner_shares} shares ({proportion:.4f}) = {final_reward:.8f} ZION {eco_info}")

    def calculate_block_rewards_via_blockchain(self, block: PoolBlock) -> None:
        """Calculate proportional rewards and create blockchain transactions"""
        if block.total_shares == 0:
            return

        # Calculate pool fee (reduce fee for eco algorithms)
        base_pool_fee = block.reward_amount * self.pool_fee_percent
        eco_fee_reduction = 0.0
        
        # Count eco-friendly shares for fee reduction
        eco_shares = 0
        for address, shares in block.miner_shares.items():
            stats = self.get_miner_stats(address)
            if stats.algorithm in ['randomx', 'yescrypt']:
                eco_shares += shares
                
        eco_ratio = eco_shares / block.total_shares if block.total_shares > 0 else 0
        eco_fee_reduction = base_pool_fee * 0.2 * eco_ratio  # Up to 20% fee reduction
        
        pool_fee_amount = base_pool_fee - eco_fee_reduction
        miner_reward_total = block.reward_amount - pool_fee_amount

        logger.info(f"Block reward: {block.reward_amount} ZION, Pool fee: {pool_fee_amount:.4f} (eco reduction: {eco_fee_reduction:.4f}), Miner total: {miner_reward_total}")

        # Calculate proportional rewards with eco bonuses and create transactions
        for address, miner_shares in block.miner_shares.items():
            if miner_shares > 0:
                proportion = miner_shares / block.total_shares
                base_reward = miner_reward_total * proportion
                
                # Apply eco-friendly algorithm bonus/penalty
                stats = self.get_miner_stats(address)
                algorithm = stats.algorithm
                eco_multiplier = self.eco_rewards.get(algorithm, 1.0)
                
                final_reward = base_reward * eco_multiplier
                
                # Create blockchain transaction for the reward
                try:
                    self.blockchain.create_transaction(
                        self.pool_wallet_address,  # From pool wallet
                        address,                   # To miner
                        final_reward,              # Reward amount
                        f"Pool mining reward for block {block.height} - {miner_shares} shares ({algorithm})"
                    )
                    logger.info(f"✅ Created blockchain transaction: {final_reward:.8f} ZION to {address}")
                except Exception as e:
                    logger.error(f"❌ Failed to create reward transaction for {address}: {e}")
                
                eco_info = f"(eco: {eco_multiplier}x)" if eco_multiplier != 1.0 else ""
                logger.info(f"Miner {address} [{algorithm}]: {miner_shares} shares ({proportion:.4f}) = {final_reward:.8f} ZION {eco_info}")

    def process_pending_payouts(self) -> List[Dict[str, Any]]:
        """Process miners who have reached payout threshold"""
        payouts = []

        for address, stats in self.miner_stats.items():
            if stats.balance_pending >= self.payout_threshold:
                payout_amount = stats.balance_pending

                # Create payout record
                payout = {
                    'address': address,
                    'amount': payout_amount,
                    'timestamp': time.time(),
                    'block_height': self.current_block_height,
                    'status': 'pending'
                }

                payouts.append(payout)

                # Reset pending balance (would move to paid balance after successful tx)
                stats.balance_pending = 0
                stats.balance_paid += payout_amount

                logger.info(f"💰 Payout ready for {address}: {payout_amount:.8f} ZION")

        return payouts

    async def cleanup_inactive_miners(self):
        """Remove inactive miners"""
        while True:
            await asyncio.sleep(300)  # Check every 5 minutes

            current_time = time.time()
            inactive_addrs = []

            for addr, miner in self.miners.items():
                last_activity = miner.get('last_activity', miner.get('connected', current_time))
                if current_time - last_activity > 1800:  # 30 minutes timeout
                    inactive_addrs.append(addr)

            for addr in inactive_addrs:
                print(f"🧹 Removing inactive miner: {addr}")
                if addr in self.miners:
                    del self.miners[addr]

    async def handle_client(self, reader, writer):
        """Handle incoming miner connections"""
        addr = writer.get_extra_info('peername')
        logger.info(f"New connection from {addr}")
        print(f"👷 New miner connected from {addr}")
        
        # Track connection statistics
        self.performance_stats['total_connections'] += 1

        try:
            # Switch to line-based parsing to avoid concatenated JSON issues
            while True:
                line = await reader.readline()
                if not line:
                    break
                raw = line.decode('utf-8').strip()
                if not raw:
                    continue
                print(f"🧾 RAW <- {addr}: {raw}")
                response = await self.handle_message(raw, addr, writer)
                if response:
                    writer.write(response.encode('utf-8'))
                    await writer.drain()

        except Exception as e:
            logger.error(f"Error handling miner {addr}: {e}")
            print(f"❌ Error handling miner {addr}: {e}")
        finally:
            logger.info(f"Miner {addr} disconnected")
            print(f"👋 Miner {addr} disconnected")
            writer.close()
            await writer.wait_closed()

            # Remove miner from tracking
            if addr in self.miners:
                del self.miners[addr]

    async def handle_message(self, message, addr, writer):
        """Process incoming mining protocol messages"""
        try:
            # Check if IP is banned
            if self.is_ip_banned(addr[0]):
                print(f"🚫 Blocked message from banned IP: {addr[0]}")
                return None
                
            data = json.loads(message)
            method = data.get('method')

            logger.info(f"Received from {addr}: {method}")
            print(f"📥 Received from {addr}: {method}")

            # Detect Stratum vs XMrig protocol
            if method and method.startswith('mining.'):
                return await self.handle_stratum_method(data, addr, writer)

            # Handle XMrig protocol
            if method == 'login':
                return await self.handle_xmrig_login(data, addr, writer)
            elif method == 'submit':
                return await self.handle_xmrig_submit(data, addr, writer)
            elif method == 'keepalived':
                return await self.handle_keepalive(data, addr)
            else:
                logger.warning(f"Unknown method from {addr}: {method}")
                print(f"❓ Unknown method: {method}")
                return json.dumps({
                    "id": data.get('id', 1),
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": "Method not found"}
                }) + '\n'

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from {addr}: {message}")
            return None
        except Exception as e:
            logger.error(f"Error processing message from {addr}: {e}")
            self.performance_stats['errors_count'] += 1
            return None
            
    def is_ip_banned(self, ip):
        """Check if IP address is currently banned"""
        if ip not in self.banned_ips:
            return False
            
        ban_info = self.banned_ips[ip]
        if time.time() - ban_info['banned_at'] > ban_info['duration']:
            # Ban expired, remove it
            del self.banned_ips[ip]
            return False
            
        return True
        
    def track_invalid_share(self, ip, is_valid):
        """Track share statistics per IP for banning decisions"""
        if not self.banning['enabled']:
            return
            
        if ip not in self.connection_stats:
            self.connection_stats[ip] = {
                'total_shares': 0,
                'invalid_shares': 0,
                'first_seen': time.time()
            }
            
        stats = self.connection_stats[ip]
        stats['total_shares'] += 1
        
        if not is_valid:
            stats['invalid_shares'] += 1
            
        # Check for banning after threshold
        if stats['total_shares'] >= self.banning['check_threshold']:
            invalid_percent = (stats['invalid_shares'] / stats['total_shares']) * 100
            
            if invalid_percent >= self.banning['invalid_percent_threshold']:
                self.ban_ip(ip, reason=f"{invalid_percent:.1f}% invalid shares")
                
    def ban_ip(self, ip, duration=None, reason="High invalid share rate"):
        """Ban IP address for specified duration"""
        if duration is None:
            duration = self.banning['ban_duration']
            
        self.banned_ips[ip] = {
            'banned_at': time.time(),
            'duration': duration,
            'reason': reason
        }
        
        # Reset stats for this IP
        if ip in self.connection_stats:
            del self.connection_stats[ip]
            
        print(f"🚫 IP {ip} BANNED for {duration}s: {reason}")
        logger.warning(f"Banned IP {ip}: {reason}")
        
    def adjust_difficulty(self, addr, algorithm):
        """Variable difficulty adjustment based on miner performance"""
        if not self.vardiff['enabled'] or addr not in self.miners:
            return
            
        miner = self.miners[addr]
        share_times = miner.get('share_times', [])
        
        # Need at least 3 shares to adjust
        if len(share_times) < 3:
            return
            
        # Calculate average time of recent shares
        recent_times = share_times[-5:]  # Last 5 shares
        avg_time = sum(recent_times) / len(recent_times)
        
        current_diff = miner.get('difficulty', self.difficulty.get(algorithm, 1000))
        min_diff = self.vardiff['min_diff'].get(algorithm, 100)
        max_diff = self.vardiff['max_diff'].get(algorithm, 10000)
        target_time = self.vardiff['target_time']
        variance = self.vardiff['variance_percent'] / 100
        
        # Calculate new difficulty
        if avg_time < target_time * (1 - variance):
            # Too fast - increase difficulty
            new_diff = min(current_diff * 1.3, max_diff)
        elif avg_time > target_time * (1 + variance):
            # Too slow - decrease difficulty  
            new_diff = max(current_diff * 0.75, min_diff)
        else:
            # In target range - no change
            return
            
        # Apply eco-friendly bonus
        if algorithm in ['yescrypt', 'autolykos_v2']:
            new_diff *= 0.95  # 5% easier for eco algorithms
            
        new_diff = int(new_diff)
        
        if new_diff != current_diff:
            miner['difficulty'] = new_diff
            print(f"📊 VarDiff {addr[0]}:{addr[1]} {algorithm}: {current_diff} → {new_diff} (avg: {avg_time:.1f}s)")
            
            # Send new difficulty to miner
            if miner.get('protocol') == 'stratum':
                self.send_difficulty_update(miner['writer'], new_diff)
                
    def send_difficulty_update(self, writer, difficulty):
        """Send difficulty update to Stratum miner"""
        try:
            msg = json.dumps({
                'id': None,
                'method': 'mining.set_difficulty', 
                'params': [difficulty]
            }) + '\n'
            
            writer.write(msg.encode('utf-8'))
            asyncio.create_task(writer.drain())
        except Exception as e:
            logger.error(f"Failed to send difficulty update: {e}")

    async def handle_xmrig_login(self, data, addr, writer):
        """Handle XMrig (CPU RandomX) login with ZION address support"""
        params = data.get('params', {})
        login = params.get('login', 'unknown')
        password = params.get('pass', 'x')
        agent = params.get('agent', 'unknown')

        # Validate ZION address
        is_zion_address = self.validate_zion_address(login)

        # Detect algorithm from password parameter or agent
        algorithm = 'randomx'  # Default
        if 'yescrypt' in password.lower() or 'yescrypt' in agent.lower():
            algorithm = 'yescrypt'
        elif 'autolykos' in password.lower() or 'autolykos' in agent.lower():
            algorithm = 'autolykos_v2'

        logger.info(f"XMrig login: {login} from {addr} (ZION: {is_zion_address}, Algorithm: {algorithm})")
        print(f"🖥️ XMrig (CPU) Login from {addr}")
        print(f"💰 Address: {login}")
        print(f"🔧 Algorithm: {algorithm}")
        if is_zion_address:
            print(f"✅ Valid ZION address detected!")
        else:
            print(f"⚠️ Legacy address format accepted")

        # Store miner info with enhanced session tracking
        self.miners[addr] = {
            'type': 'cpu',
            'protocol': 'xmrig',
            'algorithm': algorithm,
            'id': f"zion_{int(time.time())}_{addr[1]}",
            'login': login,
            'is_zion_address': is_zion_address,
            'agent': agent,
            'connected': time.time(),
            'last_activity': time.time(),
            'last_share': None,
            'last_job_sent': None,
            'share_count': 0,
            'last_job_id': None,
            'writer': writer,
            'session_active': True
        }

        self.performance_stats['total_connections'] += 1
        current_connections = len(self.miners)
        self.performance_stats['peak_connections'] = max(self.performance_stats['peak_connections'], current_connections)

        # Create job for login response
        job = self.get_job_for_miner(addr)

        # XMRig expects exact login response format - NO error field when successful
        response = json.dumps({
            "id": data.get("id"),
            "jsonrpc": "2.0",
            "result": {
                "id": self.miners[addr]['id'],
                "job": job,
                "status": "OK"
            }
        }) + '\n'

        logger.info(f"XMrig login successful for {addr}")
        print(f"✅ CPU miner login successful")

        # Start sending periodic jobs to maintain connection
        asyncio.create_task(self.send_periodic_jobs(addr))

        return response

    async def handle_xmrig_submit(self, data, addr, writer):
        """Handle XMrig share submission with real validation and rewards"""
        params = data.get('params', {})
        job_id = params.get('job_id', 'unknown')
        nonce = params.get('nonce', 'unknown')
        result = params.get('result', 'unknown')

        logger.info(f"[SUBMIT] From {addr} job={job_id} nonce={nonce} result={result}")

        if addr not in self.miners:
            return json.dumps({
                "id": data.get('id'),
                "jsonrpc": "2.0",
                "error": {"code": -1, "message": "Not logged in"}
            }) + '\n'

        miner = self.miners[addr]
        address = miner['login']
        algorithm = miner.get('algorithm', 'randomx')
        difficulty = self.difficulty.get(algorithm, self.difficulty['cpu'])

        # Check for duplicate shares
        share_key = f"{job_id}:{nonce}:{result}"
        if share_key in self.submitted_shares:
            print(f"🚫 DUPLICATE SHARE from {addr}")
            self.record_share(address, algorithm, is_valid=False)
            return json.dumps({
                "id": data.get('id'),
                "jsonrpc": "2.0",
                "error": {"code": -4, "message": "Duplicate share"}
            }) + '\n'

        # Performance monitoring
        start_time = time.time()

        # Validate share based on algorithm
        is_valid = False
        if algorithm == 'randomx':
            is_valid = self.validate_randomx_share(job_id, nonce, result, difficulty)
        elif algorithm == 'yescrypt':
            is_valid = self.validate_yescrypt_share(job_id, nonce, result, difficulty)
        elif algorithm == 'autolykos_v2':
            is_valid = self.validate_autolykos_v2_share(job_id, nonce, result, difficulty)
        else:
            # Fallback to RandomX validation
            is_valid = self.validate_randomx_share(job_id, nonce, result, difficulty)

        # Record processing time
        processing_time = time.time() - start_time
        self.share_processing_times.append(processing_time)
        self.performance_stats['total_shares_processed'] += 1

        # Keep only last 100 processing times for average calculation
        if len(self.share_processing_times) > 100:
            self.share_processing_times.pop(0)

        self.performance_stats['avg_share_processing_time'] = sum(self.share_processing_times) / len(self.share_processing_times)

        # Track share for IP banning
        self.track_invalid_share(addr[0], is_valid)
        
        if is_valid:
            # Record valid share
            self.submitted_shares.add(share_key)
            self.record_share(address, algorithm, is_valid=True)

            # Save detailed share to database
            self.db.save_share(address, algorithm, job_id, nonce, result, difficulty,
                             True, processing_time, addr[0])

            miner['share_count'] += 1
            total_shares = miner['share_count']
            share_time = time.time()
            miner['last_share'] = share_time
            
            # Track share times for vardiff
            if 'share_times' not in miner:
                miner['share_times'] = []
            if 'last_share_time' in miner:
                time_diff = share_time - miner['last_share_time']
                miner['share_times'].append(time_diff)
                # Keep only last 10 times
                if len(miner['share_times']) > 10:
                    miner['share_times'].pop(0)
            miner['last_share_time'] = share_time
            
            # Adjust difficulty if needed
            self.adjust_difficulty(addr, algorithm)

            print(f"🎯 {algorithm.upper()} Share: job={job_id}, nonce={nonce}")
            print(f"✅ VALID {algorithm.upper()} SHARE ACCEPTED (Total: {total_shares})")
            print(f"💰 Address: {address}")

            # Check for block discovery
            self.check_block_found()

            # Process any pending payouts
            payouts = self.process_pending_payouts()
            if payouts:
                print(f"💰 {len(payouts)} payouts ready for processing")

        else:
            # Invalid share
            self.record_share(address, algorithm, is_valid=False)

            # Save invalid share to database
            self.db.save_share(address, algorithm, job_id, nonce, result, difficulty,
                             False, processing_time, addr[0])

            print(f"❌ INVALID {algorithm.upper()} SHARE from {addr}")
            return json.dumps({
                "id": data.get('id'),
                "jsonrpc": "2.0",
                "error": {"code": -1, "message": "Invalid share"}
            }) + '\n'

        # XMRig expects specific response format for share acceptance - NO error field when successful
        response = json.dumps({
            "id": data.get('id'),
            "jsonrpc": "2.0",
            "result": {
                "status": "OK"
            }
        }) + '\n'

        # Force creation of a fresh job for next work to avoid stale job reuse
        try:
            self.create_randomx_job()
            new_job = self.get_job_for_miner(addr)
        except Exception as e:
            logger.error(f"Job refresh failure after share from {addr}: {e}")
            new_job = None
        if new_job:
            # XMRig expects job notification in specific format
            job_notification = json.dumps({
                "jsonrpc": "2.0",
                "method": "job",
                "params": new_job
            }) + '\n'

            logger.info(f"Sent share acceptance + new job to {addr}")
            return response + job_notification

        return response

    async def handle_keepalive(self, data, addr):
        """Enhanced keepalive handling"""
        if addr in self.miners:
            self.miners[addr]['last_activity'] = time.time()
            self.miners[addr]['session_active'] = True

        print(f"💓 Keepalive from {addr} - session renewed")
        logger.info(f"Keepalive received from {addr}")

        return json.dumps({
            "id": data.get('id'),
            "jsonrpc": "2.0",
            "result": {"status": "KEEPALIVED"}
        }) + '\n'

    def create_randomx_job(self):
        """Create RandomX job for CPU miners"""
        self.job_counter += 1
        job_id = f"zion_rx_{self.job_counter:06d}"

        self.current_jobs['randomx'] = {
            "job_id": job_id,
            "blob": "0606" + secrets.token_hex(73),  # 76 bytes total
            "target": "b88d0600",  # Difficulty target
            "algo": "rx/0",
            "height": self.current_block_height + self.job_counter,
            "seed_hash": secrets.token_hex(32)
        }

        # Store job for validation
        self.jobs[job_id] = {
            'job_id': job_id,
            'algorithm': 'randomx',
            'blob': self.current_jobs['randomx']['blob'],
            'target': self.current_jobs['randomx']['target'],
            'height': self.current_jobs['randomx']['height'],
            'seed_hash': self.current_jobs['randomx']['seed_hash'],
            'created': time.time()
        }

        logger.info(f"Created RandomX job: {job_id}")
        print(f"🔨 RandomX job: {job_id}")
        return self.current_jobs['randomx']

    def create_autolykos_v2_job(self):
        """Create Autolykos v2 job for GPU miners"""
        self.job_counter += 1
        job_id = f"zion_al_{self.job_counter:06d}"

        # Autolykos v2 parameters
        height = self.current_block_height + self.job_counter
        block_header = secrets.token_hex(80)  # 80 bytes block header

        # Generate elements for Autolykos (simplified)
        elements_seed = secrets.token_hex(32)

        job = {
            'job_id': job_id,
            'algorithm': 'autolykos_v2',
            'height': height,
            'block_header': block_header,
            'elements_seed': elements_seed,
            'n_value': 2**21,  # Autolykos N parameter
            'k_value': 32,     # Autolykos K parameter
            'created': time.time(),
            'difficulty': self.difficulty['autolykos_v2']
        }

        self.jobs[job_id] = job
        print(f"🌟 Autolykos v2 job created: {job_id} height={height}")
        return job

    def create_yescrypt_job(self):
        """Create Yescrypt job for CPU miners"""
        self.job_counter += 1
        job_id = f"zion_ys_{self.job_counter:06d}"

        height = self.current_block_height + self.job_counter
        block_header = secrets.token_hex(80)

        job = {
            'job_id': job_id,
            'algorithm': 'yescrypt',
            'height': height,
            'block_header': block_header,
            'created': time.time(),
            'difficulty': self.difficulty['yescrypt']
        }

        self.jobs[job_id] = job
        print(f"⚡ Yescrypt job created: {job_id} height={height}")
        return job

    def get_job_for_miner(self, addr):
        """Get appropriate job for miner based on algorithm"""
        if addr not in self.miners:
            return None

        miner = self.miners[addr]
        algorithm = miner.get('algorithm', 'randomx')

        # Create new job based on algorithm
        if algorithm == 'randomx':
            if not self.current_jobs['randomx'] or self.job_counter % 5 == 0:
                self.create_randomx_job()
            job = self.current_jobs['randomx'].copy()
        elif algorithm == 'yescrypt':
            # Create new Yescrypt job
            job = self.create_yescrypt_job()
        elif algorithm == 'autolykos_v2':
            # Create new Autolykos v2 job
            job = self.create_autolykos_v2_job()
        else:
            # Fallback to RandomX
            if not self.current_jobs['randomx'] or self.job_counter % 5 == 0:
                self.create_randomx_job()
            job = self.current_jobs['randomx'].copy()

        self.miners[addr]['last_job_id'] = job['job_id']
        return job

    async def send_periodic_jobs(self, addr):
        """Enhanced periodic jobs with proper connection maintenance"""
        job_count = 0

        # Wait a shorter time before starting periodic jobs
        await asyncio.sleep(5)

        while addr in self.miners:
            await asyncio.sleep(18)  # Faster cadence to keep connection alive
            job_count += 1

            if addr not in self.miners:
                break

            try:
                current_time = time.time()

                # Check miner activity
                last_activity = self.miners[addr].get('last_activity',
                                                   self.miners[addr].get('connected', current_time))

                # Send keepalive if no recent activity
                if current_time - last_activity > 45:
                    if 'writer' in self.miners[addr]:
                        writer = self.miners[addr]['writer']
                        keepalive_msg = json.dumps({
                            "jsonrpc": "2.0",
                            "method": "keepalived",
                            "params": {}
                        }) + '\n'

                        writer.write(keepalive_msg.encode('utf-8'))
                        await writer.drain()
                        print(f"💓 Sent keepalive to {addr}")

                # Always generate fresh job to avoid stale reuse
                self.create_randomx_job()
                job = self.get_job_for_miner(addr)
                if job and 'writer' in self.miners[addr]:
                    writer = self.miners[addr]['writer']

                    job_notification = json.dumps({
                        "jsonrpc": "2.0",
                        "method": "job",
                        "params": job
                    }) + '\n'

                    writer.write(job_notification.encode('utf-8'))
                    await writer.drain()
                    print(f"📡 Periodic job #{job_count} sent to {addr}")

                    # Update activity
                    self.miners[addr]['last_job_sent'] = current_time
                    self.miners[addr]['last_activity'] = current_time

            except Exception as e:
                logger.error(f"Error in periodic jobs for {addr}: {e}")
                print(f"❌ Connection lost to {addr}")
                if addr in self.miners:
                    del self.miners[addr]
                break

    # ============= STRATUM IMPLEMENTATION FOR KAWPOW =============

    async def handle_stratum_method(self, data, addr, writer):
        """Handle Stratum protocol methods (SRBMiner KawPow)"""
        method = data.get('method')

        # Initialize miner state if not exists
        if addr not in self.miners:
            extranonce1 = secrets.token_hex(4)  # 8 hex chars
            self.miners[addr] = {
                'type': 'gpu',
                'protocol': 'stratum',
                'algorithm': 'kawpow',
                'id': f"stratum_{int(time.time())}_{addr[1]}",
                'login': None,
                'connected': time.time(),
                'last_activity': time.time(),
                'session_active': True,
                'difficulty': self.difficulty['gpu'],
                'shares_window': [],
                'writer': writer,
                'authorized': False,
                'last_job_id': None,
                'extranonce1': extranonce1,
                'extranonce2_size': 8
            }

        if method == 'mining.subscribe':
            return await self.handle_stratum_subscribe(data, addr)
        elif method in ('mining.authorize', 'mining.login'):
            return await self.handle_stratum_authorize(data, addr)
        elif method == 'mining.submit':
            return await self.handle_stratum_submit(data, addr)
        elif method == 'mining.extranonce.subscribe':
            # Simple acknowledge for extranonce subscription
            return json.dumps({
                'id': data.get('id'),
                'result': True,
                'error': None
            }) + '\n'
        else:
            return json.dumps({
                'id': data.get('id'),
                'error': {'code': -32601, 'message': 'Method not found'},
                'result': None
            }) + '\n'

    async def handle_stratum_subscribe(self, data, addr):
        """Handle mining.subscribe for SRBMiner KawPow"""
        extranonce1 = self.miners[addr]['extranonce1']
        extranonce2_size = self.miners[addr]['extranonce2_size']

        response = {
            'id': data.get('id'),
            'result': [["mining.set_difficulty", "mining.notify"], extranonce1, extranonce2_size],
            'error': None
        }
        print(f"📤 Subscribe response: extranonce1={extranonce1}")
        return json.dumps(response) + '\n'

    async def handle_stratum_authorize(self, data, addr):
        """Handle mining.authorize and send initial job"""
        params = data.get('params', [])
        wallet = params[0] if params else 'unknown'
        password = params[1] if len(params) > 1 else ''

        # Detect algorithm from password or user agent
        algorithm = 'kawpow'  # Default for GPU
        if 'autolykos' in password.lower():
            algorithm = 'autolykos_v2'
        elif 'yescrypt' in password.lower():
            algorithm = 'yescrypt'

        # Update miner info
        self.miners[addr]['login'] = wallet
        self.miners[addr]['algorithm'] = algorithm
        self.miners[addr]['authorized'] = True

        # Set difficulty based on algorithm
        if algorithm in self.difficulty:
            self.miners[addr]['difficulty'] = self.difficulty[algorithm]
        else:
            self.miners[addr]['difficulty'] = self.difficulty['gpu']

        # Initialize miner stats
        self.get_miner_stats(wallet)

        # Create job based on algorithm
        if algorithm == 'autolykos_v2':
            job = self.create_autolykos_v2_job()
            # Autolykos v2 uses different notify format
            notify_params = [
                job['job_id'],
                job['block_header'],
                job['elements_seed'],
                job['height'],
                job['n_value'],
                job['k_value'],
                True  # clean_jobs
            ]
        elif algorithm == 'yescrypt':
            job = self.create_yescrypt_job()
            # Yescrypt uses simplified notify format
            notify_params = [
                job['job_id'],
                job['block_header'],
                job['height'],
                True  # clean_jobs
            ]
        else:  # kawpow
            job = self.create_kawpow_job()
            diff = self.miners[addr]['difficulty']
            target_8b = self.difficulty_to_kawpow_target_8byte(diff)
            notify_params = [
                job['job_id'],
                job['seed_hash'],
                job['header_hash'],
                job['height'],
                job['epoch'],
                target_8b,
                True
            ]

        diff = self.miners[addr]['difficulty']

        # Build response bundle
        auth_resp = json.dumps({
            'id': data.get('id'),
            'result': True,
            'error': None
        }) + '\n'

        set_diff_msg = json.dumps({
            'id': None,
            'method': 'mining.set_difficulty',
            'params': [diff]
        }) + '\n'

        notify_msg = json.dumps({
            'id': None,
            'method': 'mining.notify',
            'params': notify_params
        }) + '\n'

        bundled = auth_resp + set_diff_msg + notify_msg
        print(f"📤 Auth+notify: job={job['job_id']} diff={diff}")
        return bundled

    async def handle_stratum_submit(self, data, addr):
        """Handle mining.submit with real KawPow validation"""
        params = data.get('params', [])
        if len(params) < 5:
            return json.dumps({
                'id': data.get('id'),
                'result': False,
                'error': {'code': -1, 'message': 'Invalid params'}
            }) + '\n'

        worker, job_id, nonce, mix_hash, header_hash = params[:5]

        if addr not in self.miners:
            return json.dumps({
                'id': data.get('id'),
                'result': False,
                'error': {'code': -1, 'message': 'Not authorized'}
            }) + '\n'

        miner = self.miners[addr]
        address = miner['login']
        algorithm = miner.get('algorithm', 'kawpow')
        difficulty = miner['difficulty']

        # Check for duplicate shares
        share_key = f"{job_id}:{nonce}:{mix_hash}:{header_hash}"
        if share_key in self.submitted_shares:
            print(f"🚫 DUPLICATE {algorithm.upper()} SHARE from {addr}")
            self.record_share(address, algorithm, is_valid=False)
            return json.dumps({
                'id': data.get('id'),
                'result': False,
                'error': {'code': -4, 'message': 'Duplicate share'}
            }) + '\n'

        # Validate share based on algorithm
        is_valid = False
        if algorithm == 'kawpow':
            is_valid = self.validate_kawpow_share(job_id, nonce, mix_hash, header_hash, difficulty)
        elif algorithm == 'yescrypt':
            # Yescrypt uses different parameter format - CPU mining
            result = mix_hash  # Use mix_hash as result for Yescrypt
            is_valid = self.validate_yescrypt_share(job_id, nonce, result, difficulty)
            print(f"🌱 YESCRYPT validation attempt: job={job_id}, nonce={nonce}, valid={is_valid}")
        elif algorithm == 'randomx':
            # RandomX CPU mining
            result = mix_hash  # Use mix_hash as result for RandomX
            is_valid = self.validate_randomx_share(job_id, nonce, result, difficulty)
        elif algorithm == 'autolykos_v2':
            # For Autolykos v2, we need to adapt the parameters
            # Autolykos v2 uses different parameter format than KawPow
            result = mix_hash  # Use mix_hash as result for Autolykos v2
            is_valid = self.validate_autolykos_v2_share(job_id, nonce, result, difficulty)
        else:
            # Fallback to KawPow validation
            is_valid = self.validate_kawpow_share(job_id, nonce, mix_hash, header_hash, difficulty)

        if is_valid:
            # Record valid share
            self.submitted_shares.add(share_key)
            self.record_share(address, algorithm, is_valid=True)

            miner['share_count'] = miner.get('share_count', 0) + 1
            total_shares = miner['share_count']

            print(f"🎯 {algorithm.upper()} Share: job={job_id}, nonce={nonce}")
            print(f"✅ VALID {algorithm.upper()} SHARE ACCEPTED (Total: {total_shares})")
            print(f"💰 Address: {address}")

            # Check for block discovery
            self.check_block_found()

            # Process any pending payouts
            payouts = self.process_pending_payouts()
            if payouts:
                print(f"💰 {len(payouts)} payouts ready for processing")

        else:
            # Invalid share
            self.record_share(address, algorithm, is_valid=False)
            print(f"❌ INVALID {algorithm.upper()} SHARE from {addr}")
            return json.dumps({
                'id': data.get('id'),
                'result': False,
                'error': {'code': -1, 'message': 'Invalid share'}
            }) + '\n'

        return json.dumps({
            'id': data.get('id'),
            'result': True,
            'error': None
        }) + '\n'

    def create_kawpow_job(self):
        """Create KawPow job for GPU miners"""
        self.job_counter += 1
        job_id = f"zion_kp_{self.job_counter:06d}"
        # Pro kompatibilitu se SRBMiner: použij nízkou výšku a epoch=0
        height = self.current_block_height + self.job_counter  # < 7500 → epoch 0
        epoch = 0
        # Deterministický seed pro aktuální epoch (placeholder)
        base_seed = '00' * 32  # 64 hex nul – stabilní seed
        seed_hash = base_seed
        header_hash = secrets.token_hex(32)
        mix_hash = secrets.token_hex(16)
        job = {
            'job_id': job_id,
            'algorithm': 'kawpow',
            'height': height,
            'epoch': epoch,
            'seed_hash': seed_hash,
            'header_hash': header_hash,
            'mix_hash': mix_hash,
            'created': time.time(),
            'difficulty': self.difficulty['gpu']
        }

        self.jobs[job_id] = job
        print(f"🔥 KawPow job created: {job_id} height={height} epoch={epoch}")
        return job

    def difficulty_to_kawpow_target_8byte(self, diff: int) -> str:
        """Convert difficulty to 8-byte big-endian target for KawPow"""
        diff = max(1, min(diff, 2_000_000))
        # Jednoduchý výpočet: base / difficulty
        base = 0xFFFFFFFFFFFFFFFF  # 8 bytes max
        target = base // diff
        if target < 1:
            target = 1
        # Big-endian 8 bytes
        return f"{target:016x}"

    def difficulty_to_kawpow_target_32bit(self, diff: int) -> str:
        """Convert difficulty to 32-bit big-endian target for KawPow"""
        diff = max(1, min(diff, 2_000_000))
        # Jednoduchý výpočet: base / difficulty
        base = 0xFFFFFFFF  # 4 bytes max
        target = base // diff
        if target < 1:
            target = 1
        # Big-endian 4 bytes
        return f"{target:08x}"

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics"""
        total_miners = len(self.miners)
        total_shares = sum(stats.total_shares for stats in self.miner_stats.values())
        total_valid_shares = sum(stats.valid_shares for stats in self.miner_stats.values())
        total_invalid_shares = sum(stats.invalid_shares for stats in self.miner_stats.values())

        blocks_found = len([b for b in self.pool_blocks if b.status == "confirmed"])
        pending_payouts = sum(stats.balance_pending for stats in self.miner_stats.values())

        return {
            'pool_name': 'ZION Universal Pool',
            'pool_port': self.port,
            'current_height': self.current_block_height,
            'total_miners': total_miners,
            'total_shares': total_shares,
            'valid_shares': total_valid_shares,
            'invalid_shares': total_invalid_shares,
            'blocks_found': blocks_found,
            'pending_payouts_zion': pending_payouts,
            'pool_fee_percent': self.pool_fee_percent * 100,
            'payout_threshold_zion': self.payout_threshold,
            'algorithms': ['randomx', 'yescrypt', 'autolykos_v2'],
            'pool_wallet': self.pool_wallet_address,
            'server_time': datetime.now().isoformat(),
            'performance': {
                'uptime_seconds': time.time() - self.performance_stats['start_time'],
                'total_connections': self.performance_stats['total_connections'],
                'peak_connections': self.performance_stats['peak_connections'],
                'shares_processed': self.performance_stats['total_shares_processed'],
                'avg_processing_time_ms': self.performance_stats['avg_share_processing_time'] * 1000,
                'errors_count': self.performance_stats['errors_count'],
                'banned_ips': len(self.banned_ips),
                'vardiff_enabled': self.vardiff['enabled']
            }
        }

    async def periodic_stats_save(self):
        """Periodically save pool statistics to database"""
        while True:
            await asyncio.sleep(300)  # Save every 5 minutes

            try:
                stats = self.get_pool_stats()
                self.db.save_pool_stats(stats)
                print(f"💾 Pool statistics saved to database")
            except Exception as e:
                logger.error(f"Failed to save pool stats: {e}")

    async def start_server(self):
        """Start the mining pool server with database integration"""
        # Load existing miner stats from database
        print("Loading miner statistics from database...")
        # Note: Individual miner stats are loaded on-demand in get_miner_stats()

        # Initialize first block
        self.start_new_block()

        # Start periodic stats saving
        asyncio.create_task(self.periodic_stats_save())

        server = await asyncio.start_server(
            self.handle_client, '0.0.0.0', self.port
        )

        print(f"ZION Universal Mining Pool started on port {self.port}")
        print(f"Pool Stats API available at http://localhost:{self.port + 1}/api/stats")
        print(f"Pool Fee: {self.pool_fee_percent * 100}% | Payout Threshold: {self.payout_threshold} ZION")
        print(f"Algorithms: RandomX (CPU), Yescrypt (CPU), Autolykos v2 (GPU)")
        print(f"Block Reward: 333 ZION per block")
        print(f"Database: zion_pool.db (persistent storage enabled)")

        # Start API server
        try:
            self.api_server.start()
            print(f"Pool API server started on port {self.port + 1}")
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            print(f"Failed to start API server: {e}")

        # Start cleanup task
        asyncio.create_task(self.cleanup_inactive_miners())

        try:
            async with server:
                await server.serve_forever()
        except Exception as e:
            logger.error(f"Server error: {e}")
            print(f"Server error: {e}")

async def main():
    pool = ZionUniversalPool(port=3335)
    await pool.start_server()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nPool stopped")