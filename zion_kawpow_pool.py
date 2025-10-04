#!/usr/bin/env python3
"""
ZION KawPow Mining Pool
Simple Stratum server for KawPow algorithm (GPU mining)
"""
import asyncio
import json
import socket
import time
import secrets
import hashlib

class ZionKawPowPool:
    def __init__(self, port=3333):
        self.port = port
        self.miners = {}
        self.current_job = None
        self.job_counter = 0
        
    async def start_server(self):
        """Start the stratum server"""
        print(f"🚀 Starting ZION KawPow Pool on port {self.port}...")
        
        server = await asyncio.start_server(
            self.handle_client,
            '0.0.0.0',
            self.port
        )
        
        print(f"✅ KawPow Pool listening on 0.0.0.0:{self.port}")
        
        # Create initial job
        self.create_new_job()
        
        async with server:
            await server.serve_forever()
    
    async def handle_client(self, reader, writer):
        """Handle incoming miner connections"""
        addr = writer.get_extra_info('peername')
        print(f"👷 New KawPow miner connected from {addr}")
        
        try:
            while True:
                data = await reader.read(1024)
                if not data:
                    break
                
                message = data.decode('utf-8').strip()
                if not message:
                    continue
                
                # Handle multiple JSON messages
                for line in message.split('\n'):
                    if line.strip():
                        response = self.handle_message(line.strip(), addr)
                        if response:
                            writer.write(response.encode('utf-8'))
                            await writer.drain()
        
        except Exception as e:
            print(f"❌ Error handling miner {addr}: {e}")
        finally:
            print(f"👋 KawPow miner {addr} disconnected")
            writer.close()
            await writer.wait_closed()
            
            # Remove miner from tracking
            if addr in self.miners:
                del self.miners[addr]
    
    def handle_message(self, message, addr):
        """Process incoming stratum messages"""
        try:
            data = json.loads(message)
            method = data.get('method')
            
            print(f"📥 Received from {addr}: {method}")
            
            if method == 'mining.subscribe':
                return self.handle_subscribe(data, addr)
            elif method == 'mining.authorize':
                return self.handle_authorize(data, addr)
            elif method == 'mining.submit':
                return self.handle_submit(data, addr)
            elif method == 'mining.extranonce.subscribe':
                return self.handle_extranonce_subscribe(data, addr)
            else:
                print(f"❓ Unknown KawPow method: {method}")
                return json.dumps({
                    "id": data.get('id'),
                    "result": None,
                    "error": [20, "Other/Unknown", None]
                }) + '\n'
                
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON from {addr}: {message}")
            return None
        except Exception as e:
            print(f"❌ Error processing message from {addr}: {e}")
            return None
    
    def handle_subscribe(self, data, addr):
        """Handle mining.subscribe for KawPow"""
        subscription_id = secrets.token_hex(8)
        self.miners[addr] = {
            'subscription_id': subscription_id,
            'authorized': False,
            'shares': 0,
            'algorithm': 'kawpow'
        }
        
        print(f"✅ KawPow miner {addr} subscribed with ID: {subscription_id}")
        
        response = json.dumps({
            "id": data.get('id'),
            "result": [
                [["mining.set_difficulty", subscription_id], ["mining.notify", subscription_id]],
                subscription_id,
                4  # extranonce1 size
            ],
            "error": None
        }) + '\n'
        
        # Send initial difficulty and job
        asyncio.create_task(self.send_difficulty(addr, 0.1))
        asyncio.create_task(self.send_job_to_miner(addr))
        
        return response
    
    def handle_authorize(self, data, addr):
        """Handle mining.authorize for KawPow"""
        params = data.get('params', [])
        username = params[0] if params else 'unknown'
        
        if addr in self.miners:
            self.miners[addr]['authorized'] = True
            self.miners[addr]['username'] = username
            
        print(f"🔓 KawPow miner {addr} authorized as {username}")
        
        return json.dumps({
            "id": data.get('id'),
            "result": True,
            "error": None
        }) + '\n'
    
    def handle_submit(self, data, addr):
        """Handle mining.submit for KawPow"""
        params = data.get('params', [])
        
        if len(params) >= 5:
            username = params[0]
            job_id = params[1]
            extranonce2 = params[2]
            ntime = params[3]
            nonce = params[4]
            
            print(f"🎯 KawPow Share from {username}: job={job_id}, nonce={nonce}")
            
            # Accept all shares for testing
            if addr in self.miners:
                self.miners[addr]['shares'] += 1
                
            print(f"✅ KawPow Share ACCEPTED from {addr} (Total: {self.miners.get(addr, {}).get('shares', 0)})")
            
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
    
    def create_new_job(self):
        """Create new KawPow job"""
        self.job_counter += 1
        job_id = f"kawpow_job_{self.job_counter:04d}"
        
        # KawPow job parameters
        self.current_job = {
            "job_id": job_id,
            "prevhash": "00" * 32,
            "coinb1": "01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff",
            "coinb2": "ffffffff01",
            "merkle_branch": [],
            "version": "20000000",
            "nbits": "1d00ffff",  # Difficulty target
            "ntime": hex(int(time.time()))[2:].zfill(8),
            "clean_jobs": True
        }
        
        print(f"🔨 Created new KawPow job: {job_id}")
        return self.current_job
    
    async def send_job_to_miner(self, addr):
        """Send KawPow job to specific miner"""
        if addr not in self.miners or not self.current_job:
            return
            
        job = self.current_job
        notification = {
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
        }
        
        print(f"📤 Sending KawPow job {job['job_id']} to {addr}")
    
    async def send_difficulty(self, addr, difficulty):
        """Send difficulty to KawPow miner"""
        if addr in self.miners:
            subscription_id = self.miners[addr]['subscription_id']
            diff_msg = {
                "id": None,
                "method": "mining.set_difficulty",
                "params": [difficulty]
            }
            print(f"🎯 Setting KawPow difficulty {difficulty} for {addr}")

    def handle_extranonce_subscribe(self, data, addr):
        """Handle mining.extranonce.subscribe for KawPow"""
        print(f"📋 Extranonce subscribe from {addr}")
        
        return json.dumps({
            "id": data.get('id'),
            "result": True,
            "error": None
        }) + '\n'

async def main():
    pool = ZionKawPowPool(port=3334)  # Different port for KawPow
    await pool.start_server()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 KawPow Pool stopped")