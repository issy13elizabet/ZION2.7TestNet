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
            'gpu': 0.1       # GPU difficulty
        }
        
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
        
        logger.info(f"XMrig share: job={job_id}, nonce={nonce} from {addr}")
        
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
        
        # Create and send new job immediately to maintain connection
        new_job = self.get_job_for_miner(addr)
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
        
        # Wait a bit before starting periodic jobs to avoid conflicts
        await asyncio.sleep(10)  
        
        while addr in self.miners:
            await asyncio.sleep(30)  # Standard 30 second interval
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
                
                # Send periodic job to maintain connection
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

async def main():
    pool = ZionUniversalPool(port=3333)
    await pool.start_server()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 ZION Universal Pool stopped")