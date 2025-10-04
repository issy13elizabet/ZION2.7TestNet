#!/usr/bin/env python3
"""
ZION Universal Mining Pool
Supports CPU (XMrig, RandomX) and GPU (SRBMiner, KawPow/Ethash) miners
Full compatibility with legacy miners before AI integration
"""
import asyncio
import json            total_shares = self.miners.get(addr, {}).get('share_count', 0)
            is_zion = self.miners.get(addr, {}).get('is_zion_address', False)
            address = self.miners.get(addr, {}).get('login', 'unknown')
            
            print(f"✅ CPU Share ACCEPTED (Total: {total_shares})")
            if is_zion:
                print(f"💰 ZION Address: {address}")
            else:
                print(f"🔄 Legacy Address: {address}")
        
        # Send responseport socket
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
            'gpu': 0.1       # GPU difficulty
        }
        
    async def start_server(self):
        """Start the universal mining pool server"""
        print(f"🚀 Starting ZION Universal Mining Pool on port {self.port}...")
        logger.info(f"Universal Pool starting on port {self.port}")
        
        server = await asyncio.start_server(
            self.handle_client,
            '0.0.0.0',
            self.port
        )
        
        print(f"✅ Universal Pool listening on 0.0.0.0:{self.port}")
        print(f"🔧 Supports: RandomX (CPU), KawPow (GPU), Ethash (GPU)")
        
        # Create initial jobs for all algorithms
        self.create_randomx_job()
        self.create_kawpow_job()
        self.create_ethash_job()
        
        async with server:
            await server.serve_forever()
    
    async def handle_client(self, reader, writer):
        """Handle incoming miner connections"""
        addr = writer.get_extra_info('peername')
        logger.info(f"New connection from {addr}")
        print(f"👷 New miner connected from {addr}")
        
        try:
            while True:
                data = await reader.read(4096)  # Larger buffer
                if not data:
                    break
                
                message = data.decode('utf-8').strip()
                if not message:
                    continue
                
                # Handle multiple JSON messages
                for line in message.split('\n'):
                    if line.strip():
                        response = await self.handle_message(line.strip(), addr, writer)
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
            
            # Detect miner type and protocol
            if method == 'login':
                return await self.handle_xmrig_login(data, addr, writer)
            elif method == 'submit':
                return await self.handle_xmrig_submit(data, addr, writer)
            elif method == 'keepalived':
                return await self.handle_keepalive(data, addr)
            elif method == 'mining.subscribe':
                return await self.handle_stratum_subscribe(data, addr, writer)
            elif method == 'mining.authorize':
                return await self.handle_stratum_authorize(data, addr)
            elif method == 'mining.submit':
                return await self.handle_stratum_submit(data, addr, writer)
            elif method == 'mining.extranonce.subscribe':
                return await self.handle_extranonce_subscribe(data, addr)
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
        """Convert ZION address to mining-compatible format"""
        if self.validate_zion_address(address):
            return address  # Keep ZION address as-is
        else:
            # If not ZION address, assume it's legacy and accept it
            return address
    
    async def handle_xmrig_login(self, data, addr, writer):
        """Handle XMrig (CPU RandomX) login"""
        params = data.get('params', {})
        login = params.get('login', 'unknown')
        password = params.get('pass', 'x')
        agent = params.get('agent', 'unknown')
        
        # Validate and convert address
        converted_address = self.convert_address_for_mining(login)
        is_zion_address = self.validate_zion_address(login)
        
        logger.info(f"XMrig login: {login} from {addr} (ZION: {is_zion_address})")
        print(f"🖥️ XMrig (CPU) Login: {login} from {addr}")
        if is_zion_address:
            print(f"✅ Valid ZION address detected!")
        else:
            print(f"⚠️ Legacy address format detected")
        
        # Store miner info
        self.miners[addr] = {
            'type': 'cpu',
            'protocol': 'xmrig',
            'algorithm': 'randomx',
            'id': f"cpu_{int(time.time())}_{addr[1]}",
            'login': login,
            'converted_address': converted_address,
            'is_zion_address': is_zion_address,
            'agent': agent,
            'connected': time.time(),
            'share_count': 0,
            'last_job_id': None,
            'writer': writer
        }
        
        # Send job with login response
        job = self.get_job_for_miner(addr)
        
        response = json.dumps({
            "id": data.get("id"),
            "jsonrpc": "2.0",
            "error": None,
            "result": {
                "id": self.miners[addr]['id'],
                "job": job,
                "status": "OK"
            }
        }) + '\n'
        
        logger.info(f"XMrig login successful for {addr}")
        print(f"✅ CPU miner login successful")
        
        # Start sending periodic jobs
        asyncio.create_task(self.send_periodic_jobs(addr))
        
        return response
    
    async def handle_xmrig_submit(self, data, addr, writer):
        """Handle XMrig share submission"""
        params = data.get('params', {})
        job_id = params.get('job_id', 'unknown')
        nonce = params.get('nonce', 'unknown')
        result = params.get('result', 'unknown')
        
        logger.info(f"XMrig share: job={job_id}, nonce={nonce} from {addr}")
        print(f"🎯 CPU Share: job={job_id}, nonce={nonce}")
        
        # Accept share
        if addr in self.miners:
            self.miners[addr]['share_count'] += 1
            
        print(f"✅ CPU Share ACCEPTED (Total: {self.miners.get(addr, {}).get('share_count', 0)})")
        
        # Send response
        response = json.dumps({
            "id": data.get('id'),
            "jsonrpc": "2.0",
            "error": None,
            "result": {"status": "OK"}
        }) + '\n'
        
        # Send new job immediately after share
        new_job = self.get_job_for_miner(addr)
        job_notification = json.dumps({
            "jsonrpc": "2.0",
            "method": "job",
            "params": new_job
        }) + '\n'
        
        return response + job_notification
    
    async def handle_stratum_subscribe(self, data, addr, writer):
        """Handle Stratum subscribe (GPU miners)"""
        logger.info(f"Stratum subscribe from {addr}")
        print(f"🎮 GPU Miner (Stratum) connected from {addr}")
        
        subscription_id = secrets.token_hex(8)
        
        self.miners[addr] = {
            'type': 'gpu',
            'protocol': 'stratum',
            'algorithm': 'kawpow',  # Default, can be changed
            'subscription_id': subscription_id,
            'authorized': False,
            'share_count': 0,
            'connected': time.time(),
            'writer': writer
        }
        
        response = json.dumps({
            "id": data.get('id'),
            "result": [
                [["mining.set_difficulty", subscription_id], ["mining.notify", subscription_id]],
                subscription_id,
                4  # extranonce1 size
            ],
            "error": None
        }) + '\n'
        
        # Send difficulty and job
        asyncio.create_task(self.send_gpu_difficulty(addr))
        asyncio.create_task(self.send_gpu_job(addr))
        
        return response
    
    async def handle_stratum_authorize(self, data, addr):
        """Handle Stratum authorize"""
        params = data.get('params', [])
        username = params[0] if params else 'unknown'
        
        # Validate ZION address for GPU miners too
        converted_address = self.convert_address_for_mining(username)
        is_zion_address = self.validate_zion_address(username)
        
        if addr in self.miners:
            self.miners[addr]['authorized'] = True
            self.miners[addr]['username'] = username
            self.miners[addr]['converted_address'] = converted_address
            self.miners[addr]['is_zion_address'] = is_zion_address
            
        logger.info(f"GPU miner authorized: {username} (ZION: {is_zion_address})")
        print(f"🔓 GPU miner authorized: {username}")
        if is_zion_address:
            print(f"✅ Valid ZION address detected!")
        else:
            print(f"⚠️ Legacy address format detected")
        
        return json.dumps({
            "id": data.get('id'),
            "result": True,
            "error": None
        }) + '\n'
    
    async def handle_stratum_submit(self, data, addr, writer):
        """Handle Stratum share submission"""
        params = data.get('params', [])
        
        if len(params) >= 5:
            username = params[0]
            job_id = params[1]
            extranonce2 = params[2]
            ntime = params[3]
            nonce = params[4]
            
            logger.info(f"GPU share: {username}, job={job_id}, nonce={nonce}")
            print(f"🎯 GPU Share: {username}, job={job_id}, nonce={nonce}")
            
            if addr in self.miners:
                self.miners[addr]['share_count'] += 1
                
            print(f"✅ GPU Share ACCEPTED (Total: {self.miners.get(addr, {}).get('share_count', 0)})")
            
            return json.dumps({
                "id": data.get('id'),
                "result": True,
                "error": None
            }) + '\n'
        
        return json.dumps({
            "id": data.get('id'),
            "result": False,
            "error": [20, "Other/Unknown", None]
        }) + '\n'
    
    async def handle_keepalive(self, data, addr):
        """Handle keepalive messages"""
        print(f"💓 Keepalive from {addr}")
        return json.dumps({
            "id": data.get('id'),
            "jsonrpc": "2.0",
            "error": None,
            "result": {"status": "KEEPALIVED"}
        }) + '\n'
    
    async def handle_extranonce_subscribe(self, data, addr):
        """Handle extranonce subscribe"""
        return json.dumps({
            "id": data.get('id'),
            "result": True,
            "error": None
        }) + '\n'
    
    def create_randomx_job(self):
        """Create RandomX job for CPU miners"""
        self.job_counter += 1
        job_id = f"rx_job_{self.job_counter:06d}"
        
        self.current_jobs['randomx'] = {
            "job_id": job_id,
            "blob": "0606" + secrets.token_hex(73),  # 76 bytes
            "target": "b88d0600",
            "algo": "rx/0",
            "height": 1000 + self.job_counter,
            "seed_hash": secrets.token_hex(32)
        }
        
        logger.info(f"Created RandomX job: {job_id}")
        print(f"🔨 RandomX job: {job_id}")
        return self.current_jobs['randomx']
    
    def create_kawpow_job(self):
        """Create KawPow job for GPU miners"""
        self.job_counter += 1
        job_id = f"kp_job_{self.job_counter:06d}"
        
        self.current_jobs['kawpow'] = {
            "job_id": job_id,
            "prevhash": "00" * 32,
            "coinb1": "01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff",
            "coinb2": "ffffffff01",
            "merkle_branch": [],
            "version": "20000000",
            "nbits": "1d00ffff",
            "ntime": hex(int(time.time()))[2:].zfill(8),
            "clean_jobs": True
        }
        
        logger.info(f"Created KawPow job: {job_id}")
        print(f"🔨 KawPow job: {job_id}")
        return self.current_jobs['kawpow']
    
    def create_ethash_job(self):
        """Create Ethash job for GPU miners"""
        self.job_counter += 1
        job_id = f"eth_job_{self.job_counter:06d}"
        
        self.current_jobs['ethash'] = {
            "job_id": job_id,
            "prevhash": "0x" + secrets.token_hex(32),
            "seedhash": "0x" + secrets.token_hex(32),
            "target": "0x0000ffff" + "00" * 28
        }
        
        logger.info(f"Created Ethash job: {job_id}")
        print(f"🔨 Ethash job: {job_id}")
        return self.current_jobs['ethash']
    
    def get_job_for_miner(self, addr):
        """Get appropriate job for miner based on type"""
        if addr not in self.miners:
            return None
            
        miner = self.miners[addr]
        algorithm = miner.get('algorithm', 'randomx')
        
        if algorithm == 'randomx':
            if not self.current_jobs['randomx'] or self.job_counter % 5 == 0:
                self.create_randomx_job()
            job = self.current_jobs['randomx'].copy()
            miner['last_job_id'] = job['job_id']
            return job
        elif algorithm == 'kawpow':
            if not self.current_jobs['kawpow'] or self.job_counter % 5 == 0:
                self.create_kawpow_job()
            return self.current_jobs['kawpow']
        elif algorithm == 'ethash':
            if not self.current_jobs['ethash'] or self.job_counter % 5 == 0:
                self.create_ethash_job()
            return self.current_jobs['ethash']
            
        return None
    
    async def send_periodic_jobs(self, addr):
        """Send periodic jobs to keep connection alive"""
        while addr in self.miners:
            await asyncio.sleep(30)  # Send new job every 30 seconds
            
            if addr not in self.miners:
                break
                
            try:
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
                    print(f"📡 Sent periodic job to {addr}")
                    
            except Exception as e:
                logger.error(f"Error sending periodic job to {addr}: {e}")
                break
    
    async def send_gpu_difficulty(self, addr):
        """Send difficulty to GPU miner"""
        if addr in self.miners and 'writer' in self.miners[addr]:
            writer = self.miners[addr]['writer']
            subscription_id = self.miners[addr]['subscription_id']
            
            diff_msg = json.dumps({
                "id": None,
                "method": "mining.set_difficulty",
                "params": [self.difficulty['gpu']]
            }) + '\n'
            
            writer.write(diff_msg.encode('utf-8'))
            await writer.drain()
            print(f"🎯 Set GPU difficulty {self.difficulty['gpu']} for {addr}")
    
    async def send_gpu_job(self, addr):
        """Send job to GPU miner"""
        if addr in self.miners and 'writer' in self.miners[addr]:
            writer = self.miners[addr]['writer']
            algorithm = self.miners[addr].get('algorithm', 'kawpow')
            
            job = self.get_job_for_miner(addr)
            if job:
                if algorithm == 'kawpow':
                    job_msg = json.dumps({
                        "id": None,
                        "method": "mining.notify",
                        "params": [
                            job["job_id"],
                            job["prevhash"],
                            job["coinb1"],
                            job["coinb2"],
                            job["merkle_branch"],
                            job["version"],
                            job["nbits"],
                            job["ntime"],
                            job["clean_jobs"]
                        ]
                    }) + '\n'
                    
                    writer.write(job_msg.encode('utf-8'))
                    await writer.drain()
                    print(f"📤 Sent KawPow job to {addr}")

async def main():
    pool = ZionUniversalPool(port=3333)
    await pool.start_server()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 ZION Universal Pool stopped")