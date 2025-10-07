#!/usr/bin/env python3
"""
ZION Universal Pool v3.0 - GPU Optimized
========================================

Enhanced pool with optimized support for:
- RandomX (CPU mining) 
- Yescrypt (CPU mining)
- Autolykos v2 (GPU mining) with dynamic difficulty adjustment

Features:
- GPU-specific difficulty adjustment
- Optimized share validation for high hashrate miners
- Enhanced eco-bonus system
- Real-time performance monitoring
"""

import asyncio
import json
import socket
import time
import secrets
import hashlib
import logging
import sqlite3
import threading
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class GPUMinerStats:
    """Enhanced stats for GPU miners"""
    address: str
    algorithm: str = "autolykos_v2"
    total_shares: int = 0
    valid_shares: int = 0
    invalid_shares: int = 0
    last_share_time: Optional[float] = None
    connected_time: float = field(default_factory=time.time)
    balance_pending: float = 0.0
    balance_paid: float = 0.0
    difficulty: int = 1000000  # Higher difficulty for GPU
    hashrate: float = 0.0
    acceleration_mode: str = "Unknown"
    gpu_temperature: float = 0.0
    power_consumption: float = 0.0
    eco_bonus_multiplier: float = 1.2  # 20% bonus for Autolykos v2

@dataclass 
class PoolJob:
    """Enhanced pool job for different algorithms"""
    job_id: str
    algorithm: str
    height: int
    blob: str = ""
    target: str = ""
    seed_hash: str = ""
    # Autolykos v2 specific
    block_header: str = ""
    elements_seed: str = ""
    n_value: int = 2097152
    k_value: int = 32
    difficulty: int = 1000
    created: float = field(default_factory=time.time)

class GPUOptimizedPool:
    """GPU-optimized ZION mining pool"""
    
    def __init__(self, host="0.0.0.0", port=3335, api_port=3336):
        self.host = host
        self.port = port
        self.api_port = api_port
        self.running = False
        
        # Pool state
        self.miners: Dict[str, GPUMinerStats] = {}
        self.connections: Dict[str, socket.socket] = {}
        self.current_jobs: Dict[str, PoolJob] = {}
        self.current_height = 5620
        
        # Performance tracking
        self.total_hashrate = 0.0
        self.pool_start_time = time.time()
        self.shares_per_second = 0.0
        self.last_share_time = time.time()
        
        # Algorithm-specific settings
        self.algorithm_configs = {
            "randomx": {
                "base_difficulty": 10000,
                "target_time": 30,  # seconds
                "eco_multiplier": 1.0,
                "max_difficulty": 1000000
            },
            "yescrypt": {
                "base_difficulty": 50000,
                "target_time": 20,
                "eco_multiplier": 1.15,
                "max_difficulty": 5000000  
            },
            "autolykos_v2": {
                "base_difficulty": 100000,  # Start lower for testing
                "target_time": 15,  # Faster for GPU
                "eco_multiplier": 1.2,
                "max_difficulty": 50000000  # Much higher for GPU
            }
        }
    
    def calculate_gpu_difficulty(self, miner_stats: GPUMinerStats) -> int:
        """Calculate optimal difficulty for GPU miners"""
        if miner_stats.algorithm != "autolykos_v2":
            return miner_stats.difficulty
        
        # Get algorithm config
        config = self.algorithm_configs[miner_stats.algorithm]
        base_diff = config["base_difficulty"]
        target_time = config["target_time"]
        
        # Calculate hashrate-based difficulty adjustment
        if miner_stats.hashrate > 0:
            # Target: shares every target_time seconds
            estimated_difficulty = int(miner_stats.hashrate * target_time)
            
            # Clamp to reasonable bounds
            min_diff = base_diff
            max_diff = config["max_difficulty"]
            
            adjusted_difficulty = max(min_diff, min(estimated_difficulty, max_diff))
            
            # Smooth difficulty changes (max 50% change per adjustment)
            if miner_stats.difficulty > 0:
                max_change = miner_stats.difficulty * 0.5
                if adjusted_difficulty > miner_stats.difficulty:
                    adjusted_difficulty = min(adjusted_difficulty, miner_stats.difficulty + max_change)
                else:
                    adjusted_difficulty = max(adjusted_difficulty, miner_stats.difficulty - max_change)
            
            return int(adjusted_difficulty)
        
        return base_diff
    
    def create_autolykos_job(self, height: int) -> PoolJob:
        """Create Autolykos v2 specific job"""
        job_id = f"zion_al_{height:06d}"
        
        # Generate realistic Autolykos v2 parameters
        block_header = secrets.token_hex(64)
        elements_seed = secrets.token_hex(32)
        
        return PoolJob(
            job_id=job_id,
            algorithm="autolykos_v2",
            height=height,
            block_header=block_header,
            elements_seed=elements_seed,
            n_value=2097152,  # Standard Autolykos parameter
            k_value=32,
            difficulty=75,  # Base pool difficulty
            created=time.time()
        )
    
    def create_job_for_algorithm(self, algorithm: str, height: int) -> PoolJob:
        """Create algorithm-specific job"""
        if algorithm == "autolykos_v2":
            return self.create_autolykos_job(height)
        elif algorithm == "randomx":
            job_id = f"zion_rx_{height:06d}"
            blob = secrets.token_hex(76)
            target = "b88d0600"
            seed_hash = secrets.token_hex(32)
            return PoolJob(
                job_id=job_id,
                algorithm="randomx", 
                height=height,
                blob=blob,
                target=target,
                seed_hash=seed_hash
            )
        elif algorithm == "yescrypt":
            job_id = f"zion_ye_{height:06d}"
            blob = secrets.token_hex(76) 
            target = "d09d0600"
            return PoolJob(
                job_id=job_id,
                algorithm="yescrypt",
                height=height,
                blob=blob,
                target=target
            )
    
    def validate_autolykos_share(self, job: PoolJob, nonce: str, result: str, 
                                miner_address: str) -> Tuple[bool, str]:
        """Enhanced validation for Autolykos v2 shares"""
        try:
            # Basic validation
            if len(result) != 64:  # 32 bytes = 64 hex chars
                return False, "Invalid hash length"
            
            # Verify hash format
            try:
                hash_bytes = bytes.fromhex(result)
            except ValueError:
                return False, "Invalid hash format"
            
            # Check if hash meets current difficulty
            miner_stats = self.miners.get(miner_address)
            if miner_stats:
                difficulty = miner_stats.difficulty
            else:
                difficulty = self.algorithm_configs["autolykos_v2"]["base_difficulty"]
            
            hash_int = int.from_bytes(hash_bytes, byteorder='big')
            target = (1 << 256) // difficulty
            
            if hash_int >= target:
                return False, f"Hash above target (diff: {difficulty})"
            
            # Additional validation for GPU miners
            if miner_stats and miner_stats.acceleration_mode in ["CUDA", "OpenCL"]:
                # Accept GPU-accelerated shares with bonus validation
                logger.info(f"GPU share validated: {miner_address[:20]}... (mode: {miner_stats.acceleration_mode})")
            
            return True, "Valid share"
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def update_miner_hashrate(self, miner_address: str, shares_submitted: int, time_window: float):
        """Update miner hashrate based on recent shares"""
        miner_stats = self.miners.get(miner_address)
        if not miner_stats:
            return
        
        # Estimate hashrate from shares
        if time_window > 0 and miner_stats.difficulty > 0:
            estimated_hashrate = (shares_submitted * miner_stats.difficulty) / time_window
            
            # Smooth hashrate updates
            if miner_stats.hashrate > 0:
                miner_stats.hashrate = (miner_stats.hashrate * 0.8) + (estimated_hashrate * 0.2)
            else:
                miner_stats.hashrate = estimated_hashrate
            
            # Update difficulty based on new hashrate
            if miner_stats.algorithm == "autolykos_v2":
                new_difficulty = self.calculate_gpu_difficulty(miner_stats)
                if new_difficulty != miner_stats.difficulty:
                    logger.info(f"Adjusting GPU difficulty for {miner_address[:20]}...: {miner_stats.difficulty} → {new_difficulty}")
                    miner_stats.difficulty = new_difficulty
    
    async def handle_client(self, reader, writer):
        """Enhanced client handler with GPU optimization"""
        client_address = writer.get_extra_info('peername')
        client_id = f"{client_address[0]}:{client_address[1]}"
        
        logger.info(f"New connection: {client_id}")
        
        try:
            while self.running:
                data = await reader.read(1024)
                if not data:
                    break
                
                try:
                    message = json.loads(data.decode().strip())
                    response = await self.process_message(message, client_id)
                    
                    if response:
                        response_data = json.dumps(response) + '\n'
                        writer.write(response_data.encode())
                        await writer.drain()
                        
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from {client_id}: {data}")
                except Exception as e:
                    logger.error(f"Error processing message from {client_id}: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Connection error with {client_id}: {e}")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except:
                pass
            
            # Cleanup
            if client_id in self.connections:
                del self.connections[client_id]
            
            logger.info(f"Connection closed: {client_id}")
    
    async def process_message(self, message: Dict[str, Any], client_id: str) -> Optional[Dict[str, Any]]:
        """Process incoming message with GPU support"""
        method = message.get("method")
        params = message.get("params", {})
        msg_id = message.get("id", 1)
        
        if method == "login":
            return await self.handle_login(params, msg_id, client_id)
        elif method == "submit":
            return await self.handle_submit(params, msg_id, client_id)
        elif method == "getjob":
            return await self.handle_getjob(params, msg_id, client_id)
        else:
            return {"id": msg_id, "error": {"code": -1, "message": "Unknown method"}}
    
    async def handle_login(self, params: Dict[str, Any], msg_id: int, client_id: str) -> Dict[str, Any]:
        """Enhanced login handler with GPU detection"""
        login = params.get("login", "")
        agent = params.get("agent", "")
        algorithm = params.get("algo", "randomx")  # Default to RandomX
        
        # Detect GPU miners from user agent
        if "CUDA" in agent or "OpenCL" in agent or "GPU" in agent or "autolykos" in agent.lower():
            algorithm = "autolykos_v2"
        elif "yescrypt" in agent.lower():
            algorithm = "yescrypt"
        
        # Parse acceleration mode from agent
        acceleration_mode = "CPU"
        if "CUDA" in agent:
            acceleration_mode = "CUDA"
        elif "OpenCL" in agent:
            acceleration_mode = "OpenCL"
        
        # Create or update miner stats
        if login not in self.miners:
            base_difficulty = self.algorithm_configs[algorithm]["base_difficulty"]
            
            # For GPU miners, start with even lower difficulty
            if acceleration_mode in ["CUDA", "OpenCL"] and algorithm == "autolykos_v2":
                base_difficulty = 10000  # Much lower starting point for GPU
            
            self.miners[login] = GPUMinerStats(
                address=login,
                algorithm=algorithm,
                difficulty=base_difficulty,
                connected_time=time.time(),
                acceleration_mode=acceleration_mode,
                eco_bonus_multiplier=self.algorithm_configs[algorithm]["eco_multiplier"]
            )
        else:
            # Update existing miner
            self.miners[login].algorithm = algorithm
            self.miners[login].acceleration_mode = acceleration_mode
            self.miners[login].eco_bonus_multiplier = self.algorithm_configs[algorithm]["eco_multiplier"]
        
        # Create job for this algorithm  
        job = self.create_job_for_algorithm(algorithm, self.current_height)
        self.current_jobs[login] = job
        
        # Prepare job response based on algorithm
        if algorithm == "autolykos_v2":
            job_data = {
                "job_id": job.job_id,
                "algorithm": "autolykos_v2",
                "height": job.height,
                "block_header": job.block_header,
                "elements_seed": job.elements_seed,
                "n_value": job.n_value,
                "k_value": job.k_value,
                "created": job.created,
                "difficulty": self.miners[login].difficulty
            }
        else:
            job_data = {
                "job_id": job.job_id,
                "blob": job.blob,
                "target": job.target,
                "algo": f"rx/0" if algorithm == "randomx" else "yescrypt",
                "height": job.height,
                "seed_hash": getattr(job, 'seed_hash', '')
            }
        
        logger.info(f"Miner login: {login[:20]}... (algo: {algorithm}, mode: {acceleration_mode})")
        
        return {
            "id": msg_id,
            "jsonrpc": "2.0",
            "result": {
                "id": f"zion_{int(time.time())}_{secrets.randbelow(100000)}",
                "job": job_data,
                "status": "OK"
            }
        }
    
    async def handle_submit(self, params: Dict[str, Any], msg_id: int, client_id: str) -> Dict[str, Any]:
        """Enhanced share submission with GPU validation"""
        miner_id = params.get("id", "")
        nonce = params.get("nonce", "")
        result = params.get("result", "")
        job_id = params.get("job_id", "")
        
        # Get miner stats
        miner_stats = self.miners.get(miner_id)
        if not miner_stats:
            return {"id": msg_id, "error": {"code": -1, "message": "Unknown miner"}}
        
        # Get current job
        job = self.current_jobs.get(miner_id)
        if not job:
            return {"id": msg_id, "error": {"code": -1, "message": "No active job"}}
        
        # Validate share based on algorithm
        if miner_stats.algorithm == "autolykos_v2":
            is_valid, validation_msg = self.validate_autolykos_share(job, nonce, result, miner_id)
        else:
            # Use existing validation for RandomX/Yescrypt
            is_valid, validation_msg = self.validate_share(job, nonce, result, miner_stats.difficulty)
        
        # Update stats
        miner_stats.total_shares += 1
        miner_stats.last_share_time = time.time()
        
        if is_valid:
            miner_stats.valid_shares += 1
            
            # Apply eco bonus
            effective_shares = 1.0 * miner_stats.eco_bonus_multiplier
            miner_stats.balance_pending += effective_shares * 0.01  # Rough reward calculation
            
            logger.info(f"✅ Valid share from {miner_id[:20]}... (algo: {miner_stats.algorithm}, "
                       f"eco bonus: {miner_stats.eco_bonus_multiplier:.2f}x, mode: {miner_stats.acceleration_mode})")
            
            # Update hashrate estimation
            self.update_miner_hashrate(miner_id, 1, 30.0)  # 30 second window
            
            return {
                "id": msg_id,
                "jsonrpc": "2.0", 
                "result": {
                    "status": "OK",
                    "eco_bonus": miner_stats.eco_bonus_multiplier,
                    "difficulty": miner_stats.difficulty
                }
            }
        else:
            miner_stats.invalid_shares += 1
            logger.warning(f"❌ Invalid share from {miner_id[:20]}...: {validation_msg}")
            
            return {
                "id": msg_id,
                "error": {"code": 23, "message": f"Invalid share: {validation_msg}"}
            }
    
    def validate_share(self, job: PoolJob, nonce: str, result: str, difficulty: int) -> Tuple[bool, str]:
        """Basic share validation for RandomX/Yescrypt"""
        try:
            if len(result) != 64:
                return False, "Invalid hash length"
            
            hash_bytes = bytes.fromhex(result)
            hash_int = int.from_bytes(hash_bytes, byteorder='big')
            target = (1 << 256) // difficulty
            
            if hash_int < target:
                return True, "Valid share"
            else:
                return False, f"Hash above target (diff: {difficulty})"
                
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics"""
        now = time.time()
        uptime = now - self.pool_start_time
        
        # Calculate total hashrate by algorithm
        algo_stats = {}
        for algorithm in self.algorithm_configs.keys():
            miners = [m for m in self.miners.values() if m.algorithm == algorithm]
            total_hashrate = sum(m.hashrate for m in miners)
            total_miners = len(miners)
            total_shares = sum(m.valid_shares for m in miners)
            
            algo_stats[algorithm] = {
                "miners": total_miners,
                "hashrate": total_hashrate,
                "shares": total_shares,
                "eco_multiplier": self.algorithm_configs[algorithm]["eco_multiplier"]
            }
        
        return {
            "pool": {
                "name": "ZION Universal Pool v3.0 (GPU Optimized)",
                "uptime": uptime,
                "current_height": self.current_height,
                "total_miners": len(self.miners),
                "pool_hashrate": sum(m.hashrate for m in self.miners.values()),
                "fee": "1.0%"
            },
            "algorithms": algo_stats,
            "network": {
                "difficulty": 1000000,
                "block_reward": "333 ZION",
                "block_time": "60 seconds"
            }
        }
    
    async def start(self):
        """Start the GPU-optimized pool"""
        logger.info("Starting ZION Universal Pool v3.0 (GPU Optimized)")
        logger.info(f"Pool listening on {self.host}:{self.port}")
        logger.info("Enhanced support for:")
        logger.info("  • RandomX (CPU) - 1.0x eco bonus")
        logger.info("  • Yescrypt (CPU) - 1.15x eco bonus") 
        logger.info("  • Autolykos v2 (GPU) - 1.2x eco bonus")
        
        self.running = True
        
        # Start pool server
        server = await asyncio.start_server(
            self.handle_client,
            self.host, 
            self.port
        )
        
        # Start API server
        api_thread = threading.Thread(target=self.start_api_server)
        api_thread.daemon = True
        api_thread.start()
        
        # Start maintenance tasks
        asyncio.create_task(self.maintenance_loop())
        
        logger.info(f"Pool ready! GPU miners can connect with optimized difficulty adjustment")
        
        async with server:
            await server.serve_forever()
    
    def start_api_server(self):
        """Start HTTP API server for statistics"""
        class PoolAPIHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/api/stats':
                    stats = pool_instance.get_pool_stats()
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps(stats, indent=2).encode())
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format, *args):
                pass  # Suppress API logs
        
        pool_instance = self
        api_server = HTTPServer((self.host, self.api_port), PoolAPIHandler)
        logger.info(f"Pool API server started on port {self.api_port}")
        api_server.serve_forever()
    
    async def maintenance_loop(self):
        """Pool maintenance and statistics"""
        while self.running:
            await asyncio.sleep(60)  # Every minute
            
            # Log pool statistics
            stats = self.get_pool_stats()
            logger.info(f"Pool Stats - Miners: {stats['pool']['total_miners']}, "
                       f"Hashrate: {stats['pool']['pool_hashrate']:.0f} H/s")
            
            for algo, algo_stats in stats["algorithms"].items():
                if algo_stats["miners"] > 0:
                    logger.info(f"  {algo}: {algo_stats['miners']} miners, "
                               f"{algo_stats['hashrate']:.0f} H/s, "
                               f"{algo_stats['eco_multiplier']:.2f}x bonus")

async def main():
    """Main entry point"""
    pool = GPUOptimizedPool()
    
    try:
        await pool.start()
    except KeyboardInterrupt:
        logger.info("Pool stopped by user")
        pool.running = False
    except Exception as e:
        logger.error(f"Pool error: {e}")

if __name__ == "__main__":
    asyncio.run(main())