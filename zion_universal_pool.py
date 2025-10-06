#!/usr/bin/env python3
"""
ZION Universal Mining Pool with Address Support
Supports ZION addresses (ZION_xxx format) and legacy formats
CPU (XMrig, RandomX) and GPU (SRBMiner, KawPow/Ethash) miners
"""
import asyncio
import json
import socket
import time
import secrets
import hashlib
import logging
from typing import Dict, Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ZionUniversalPool:
    def __init__(self, port=3333):
        self.port = port
        self.miners = {}
        self.current_jobs = {
            'randomx': None,
            'kawpow': None,
            'ethash': None
        }
        self.job_counter = 0
        self.difficulty = {
            'cpu': 10000,    # RandomX difficulty
            'gpu': 50       # GPU difficulty (lowered for testing)
        }
        # Jobs a submissions tracking
        self.jobs = {}
        self.submissions = set()
        
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
        """Convert and validate mining address"""
        if self.validate_zion_address(address):
            return address  # Keep ZION address as-is
        else:
            # Accept legacy addresses too
            return address
        
    async def start_server(self):
        """Start the universal mining pool server"""
        print(f"🚀 Starting ZION Universal Mining Pool on port {self.port}...")
        print(f"💰 Supports ZION addresses (ZION_xxx) and legacy formats")
        logger.info(f"Universal Pool starting on port {self.port}")
        
        server = await asyncio.start_server(
            self.handle_client,
            '0.0.0.0',
            self.port
        )
        
        print(f"✅ Universal Pool listening on 0.0.0.0:{self.port}")
        print(f"🔧 Algorithms: RandomX (CPU), KawPow (GPU), Ethash (GPU)")
        
        # Create initial jobs for all algorithms
        self.create_randomx_job()
        
        # Start session cleanup task
        asyncio.create_task(self.cleanup_inactive_sessions())
        
        async with server:
            await server.serve_forever()
    
    async def cleanup_inactive_sessions(self):
        """Cleanup inactive miner sessions"""
        while True:
            await asyncio.sleep(60)  # Check every minute
            
            current_time = time.time()
            inactive_miners = []
            
            for addr, miner in self.miners.items():
                last_activity = miner.get('last_activity', miner.get('connected', current_time))
                
                # Remove miners inactive for more than 5 minutes
                if current_time - last_activity > 300:
                    inactive_miners.append(addr)
            
            # Clean up inactive miners
            for addr in inactive_miners:
                logger.info(f"Cleaning up inactive miner: {addr}")
                print(f"🧹 Removing inactive miner: {addr}")
                if addr in self.miners:
                    del self.miners[addr]
    
    async def handle_client(self, reader, writer):
        """Handle incoming miner connections"""
        addr = writer.get_extra_info('peername')
        logger.info(f"New connection from {addr}")
        print(f"👷 New miner connected from {addr}")
        
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
                    "id": data.get('id'),
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": "Method not found"}
                }) + '\n'
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from {addr}: {message}")
            return None
        except Exception as e:
            logger.error(f"Error processing message from {addr}: {e}")
            return None
    
    async def handle_xmrig_login(self, data, addr, writer):
        """Handle XMrig (CPU RandomX) login with ZION address support"""
        params = data.get('params', {})
        login = params.get('login', 'unknown')
        password = params.get('pass', 'x')
        agent = params.get('agent', 'unknown')
        
        # Validate ZION address
        is_zion_address = self.validate_zion_address(login)
        
        logger.info(f"XMrig login: {login} from {addr} (ZION: {is_zion_address})")
        print(f"🖥️ XMrig (CPU) Login from {addr}")
        print(f"💰 Address: {login}")
        if is_zion_address:
            print(f"✅ Valid ZION address detected!")
        else:
            print(f"⚠️ Legacy address format accepted")
        
        # Store miner info with enhanced session tracking
        self.miners[addr] = {
            'type': 'cpu',
            'protocol': 'xmrig',
            'algorithm': 'randomx',
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
        """Handle XMrig share submission with enhanced session handling"""
        params = data.get('params', {})
        job_id = params.get('job_id', 'unknown')
        nonce = params.get('nonce', 'unknown')
        result = params.get('result', 'unknown')
        
        logger.info(f"[SUBMIT] From {addr} job={job_id} nonce={nonce} result={result}")
        
        # Accept share and update stats
        if addr in self.miners:
            self.miners[addr]['share_count'] += 1
            self.miners[addr]['last_share'] = time.time()  # Track last activity
            total_shares = self.miners[addr]['share_count']
            is_zion = self.miners[addr].get('is_zion_address', False)
            address = self.miners[addr].get('login', 'unknown')
            
            print(f"🎯 CPU Share: job={job_id}, nonce={nonce}")
            print(f"✅ Share ACCEPTED (Total: {total_shares})")
            if is_zion:
                print(f"💰 ZION Address: {address}")
            else:
                print(f"🔄 Legacy Address: {address}")
        
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
            "height": 1000 + self.job_counter,
            "seed_hash": secrets.token_hex(32)
        }
        
        logger.info(f"Created RandomX job: {job_id}")
        print(f"🔨 RandomX job: {job_id}")
        return self.current_jobs['randomx']
    
    def get_job_for_miner(self, addr):
        """Get appropriate job for miner"""
        if addr not in self.miners:
            return None
            
        # For now, only RandomX jobs
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
        
        self.miners[addr]['login'] = wallet
        self.miners[addr]['authorized'] = True
        
        # Create job and responses
        job = self.create_kawpow_job()
        diff = self.miners[addr]['difficulty']
        
        # Canonical simplified KawPow notify (seed/hash/height/epoch/target/clean)
        # [job_id, seed_hash, header_hash, height, epoch, target, clean_jobs]
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
        """Handle mining.submit (basic validation)"""
        params = data.get('params', [])
        if len(params) < 5:
            return json.dumps({
                'id': data.get('id'),
                'result': False,
                'error': {'code': -1, 'message': 'Invalid params'}
            }) + '\n'
            
        worker, job_id, nonce, mix_hash, header_hash = params[:5]
        
        # Basic validation
        if job_id not in self.jobs:
            return json.dumps({
                'id': data.get('id'),
                'result': False,
                'error': {'code': -2, 'message': 'Unknown job'}
            }) + '\n'
            
        # Accept share (placeholder)
        if addr in self.miners:
            self.miners[addr]['share_count'] = self.miners[addr].get('share_count', 0) + 1
            print(f"✅ KawPow share accepted from {addr}")
            
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
        height = 1000 + self.job_counter  # < 7500 → epoch 0
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

async def main():
    pool = ZionUniversalPool(port=3333)
    await pool.start_server()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 ZION Universal Pool stopped")