#!/usr/bin/env python3
"""
ZION Simple Mining Pool
Jednoduší verze bez daemon dependency pro testování
"""

import asyncio
import socket
import time
import json
import hashlib
import secrets

class ZionSimplePool:
    def __init__(self, port=3333):
        self.port = port
        self.miners = {}
        self.jobs = {}
        self.current_job_id = 0
        
    async def start_server(self):
        """Start simple stratum server"""
        print(f"🚀 Starting ZION Simple Pool on port {self.port}...")
        
        server = await asyncio.start_server(
            self.handle_miner,
            '0.0.0.0',
            self.port
        )
        
        print(f"✅ Pool listening on 0.0.0.0:{self.port}")
        
        # Create initial job
        await self.create_new_job()
        
        async with server:
            await server.serve_forever()
    
    async def handle_miner(self, reader, writer):
        """Handle individual miner connection"""
        addr = writer.get_extra_info('peername')
        print(f"👷 New miner connected from {addr}")
        
        try:
            while True:
                data = await reader.readline()
                if not data:
                    break
                    
                message = data.decode('utf-8').strip()
                if message:
                    response = await self.process_message(message, addr)
                    if response:
                        writer.write((response + '\n').encode())
                        await writer.drain()
                        
        except Exception as e:
            print(f"❌ Miner {addr} error: {e}")
        finally:
            print(f"👋 Miner {addr} disconnected")
            writer.close()
            await writer.wait_closed()
    
    async def process_message(self, message, addr):
        """Process stratum message from miner"""
        try:
            data = json.loads(message)
            method = data.get('method', '')
            
            print(f"📥 Received from {addr}: {method}")
            
            if method == 'mining.subscribe':
                return self.handle_subscribe(data, addr)
            elif method == 'mining.authorize':
                return self.handle_authorize(data, addr)
            elif method == 'mining.submit':
                return self.handle_submit(data, addr)
            elif method == 'login':
                return self.handle_login(data, addr)
            else:
                print(f"❓ Unknown method: {method}")
                return json.dumps({
                    "id": data.get('id'),
                    "result": None,
                    "error": [20, "Other/Unknown", None]
                })
                
        except Exception as e:
            print(f"❌ Message processing error: {e}")
            print(f"   Raw message: {message}")
            return None
    
    def handle_subscribe(self, data, addr):
        """Handle mining.subscribe"""
        subscription_id = secrets.token_hex(8)
        self.miners[addr] = {
            'subscription_id': subscription_id,
            'authorized': False,
            'shares': 0
        }
        
        print(f"✅ Miner {addr} subscribed with ID: {subscription_id}")
        
        response = json.dumps({
            "id": data.get('id'),
            "result": [
                [["mining.set_difficulty", subscription_id], ["mining.notify", subscription_id]],
                subscription_id,
                4  # extranonce1 size
            ],
            "error": None
        })
        
        # Send difficulty after subscribe
        self.send_difficulty(addr)
        
        return response
    
    def handle_authorize(self, data, addr):
        """Handle mining.authorize"""
        params = data.get('params', [])
        username = params[0] if params else "anonymous"
        
        print(f"⚡ Miner {addr} authorized as {username}")
        
        if addr in self.miners:
            self.miners[addr]['authorized'] = True
            self.miners[addr]['username'] = username
        
        return json.dumps({
            "id": data.get('id'),
            "result": True,
            "error": None
        })
    
    def handle_submit(self, data, addr):
        """Handle mining.submit"""
        params = data.get('params', [])
        
        if len(params) >= 5:
            username = params[0]
            job_id = params[1]
            extranonce2 = params[2]
            ntime = params[3]
            nonce = params[4]
            
            print(f"📊 Share from {username}: job={job_id}, nonce={nonce}")
            
            # Accept all shares for testing
            if addr in self.miners:
                self.miners[addr]['shares'] += 1
            
            return json.dumps({
                "id": data.get('id'),
                "result": True,
                "error": None
            })
        
        return json.dumps({
            "id": data.get('id'),
            "result": False,
            "error": [20, "Invalid params", None]
        })
    
    async def create_new_job(self):
        """Create new mining job"""
        self.current_job_id += 1
        job_id = f"job_{self.current_job_id:04d}"
        
        # Create dummy job template
        job = {
            'job_id': job_id,
            'prevhash': '0' * 64,
            'coinbase1': '01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff',
            'coinbase2': 'ffffffff0100000000000000000000000000',
            'merkle_branch': [],
            'version': '00000001',
            'nbits': '1d00ffff',
            'ntime': format(int(time.time()), '08x'),
            'clean_jobs': True
        }
        
        self.jobs[job_id] = job
        print(f"🔨 Created new job: {job_id}")
        
        # Broadcast job to all miners
        await self.broadcast_job(job)
    
    async def broadcast_job(self, job):
        """Broadcast new job to all authorized miners"""
        notification = {
            "id": None,
            "method": "mining.notify",
            "params": [
                job['job_id'],
                job['prevhash'],
                job['coinbase1'],
                job['coinbase2'],
                job['merkle_branch'],
                job['version'],
                job['nbits'],
                job['ntime'],
                job['clean_jobs']
            ]
        }
        
        print(f"📡 Broadcasting job {job['job_id']} to miners")
    
    def send_difficulty(self, addr, difficulty=1000):
        """Send difficulty to miner"""
        if addr in self.miners:
            subscription_id = self.miners[addr]['subscription_id']
            diff_msg = {
                "id": None,
                "method": "mining.set_difficulty",
                "params": [difficulty]
            }
            print(f"🎯 Setting difficulty {difficulty} for {addr}")

    def handle_login(self, data, addr):
        """Handle xmrig-style login method"""
        params = data.get('params', {})
        login = params.get('login', 'unknown')
        password = params.get('pass', 'x')
        agent = params.get('agent', 'unknown')
        
        print(f"👤 Login request from {addr}: {login}, agent: {agent}")
        
        # Store miner connection
        self.miners[addr] = {
            'wallet': login,
            'connected': time.time(),
            'share_count': 0,
            'agent': agent,
            'authorized': True
        }
        
        # Send login response with job
        job_data = {
            "blob": "0100" + "00" * 74,
            "job_id": f"job_{int(time.time())}_{addr[1]}",
            "target": "0000ffff00000000000000000000000000000000000000000000000000000000",
            "algo": "rx/0",
            "height": 1000,
            "seed_hash": "00" * 32
        }
        
        # XMrig expects specific format
        response = json.dumps({
            "id": data.get("id"),
            "jsonrpc": "2.0",
            "result": {
                "id": f"miner_{addr[1]}",
                "job": job_data,
                "status": "OK"
            },
            "error": None
        }) + '\n'
        
        print(f"📤 Sent XMrig login response to {addr}")
        return response

async def main():
    pool = ZionSimplePool(port=3333)
    await pool.start_server()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Pool stopped")