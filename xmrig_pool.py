#!/usr/bin/env python3
"""
XMRig Compatible ZION Pool
Simple stratum server fully compatible with XMRig protocol
"""
import asyncio
import json
import socket
import time
import secrets
import hashlib

class XMrigCompatiblePool:
    def __init__(self, port=3333):
        self.port = port
        self.miners = {}
        self.current_job = None
        self.job_counter = 0
        
    async def start_server(self):
        """Start the stratum server"""
        print(f"🚀 Starting XMRig Compatible ZION Pool on port {self.port}...")
        
        server = await asyncio.start_server(
            self.handle_client,
            '0.0.0.0',
            self.port
        )
        
        print(f"✅ XMRig Pool listening on 0.0.0.0:{self.port}")
        
        # Create initial job
        self.create_new_job()
        
        async with server:
            await server.serve_forever()
    
    async def handle_client(self, reader, writer):
        """Handle incoming miner connections"""
        addr = writer.get_extra_info('peername')
        print(f"👷 New XMRig miner connected from {addr}")
        
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
            print(f"👋 XMRig miner {addr} disconnected")
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
            
            if method == 'login':
                return self.handle_login(data, addr)
            elif method == 'submit':
                return self.handle_submit(data, addr)
            elif method == 'keepalived':
                return self.handle_keepalive(data, addr)
            else:
                print(f"❓ Unknown XMRig method: {method}")
                return json.dumps({
                    "id": data.get('id'),
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32601,
                        "message": "Method not found"
                    }
                }) + '\n'
                
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON from {addr}: {message}")
            return None
        except Exception as e:
            print(f"❌ Error processing message from {addr}: {e}")
            return None
    
    def handle_login(self, data, addr):
        """Handle XMRig login method"""
        params = data.get('params', {})
        login = params.get('login', 'unknown')
        password = params.get('pass', 'x')
        agent = params.get('agent', 'unknown')
        
        print(f"👤 XMRig Login: {login} from {addr} (agent: {agent})")
        
        # Store miner connection
        self.miners[addr] = {
            'id': f"miner_{int(time.time())}_{addr[1]}",
            'login': login,
            'agent': agent,
            'connected': time.time(),
            'share_count': 0,
            'last_job_id': None
        }
        
        # Create job for this miner
        job = self.create_job_for_miner(addr)
        
        # XMRig login response format
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
        
        print(f"✅ XMRig login successful for {addr}")
        return response
    
    def handle_submit(self, data, addr):
        """Handle XMRig share submission"""
        params = data.get('params', {})
        
        job_id = params.get('job_id', 'unknown')
        nonce = params.get('nonce', 'unknown')
        result = params.get('result', 'unknown')
        
        print(f"🎯 Share submitted from {addr}: job={job_id}, nonce={nonce}")
        
        # Accept all shares for testing
        if addr in self.miners:
            self.miners[addr]['share_count'] += 1
            
        print(f"✅ Share ACCEPTED from {addr} (Total shares: {self.miners.get(addr, {}).get('share_count', 0)})")
        
        # Send new job with share response
        new_job = self.create_job_for_miner(addr)
        
        response = json.dumps({
            "id": data.get('id'),
            "jsonrpc": "2.0",
            "error": None,
            "result": {
                "status": "OK"
            }
        }) + '\n'
        
        # Send job notification
        job_notification = json.dumps({
            "jsonrpc": "2.0",
            "method": "job",
            "params": new_job
        }) + '\n'
        
        return response + job_notification
    
    def handle_keepalive(self, data, addr):
        """Handle XMRig keepalive"""
        print(f"💓 Keepalive from {addr}")
        
        return json.dumps({
            "id": data.get('id'),
            "jsonrpc": "2.0",
            "error": None,
            "result": {
                "status": "KEEPALIVED"
            }
        }) + '\n'
    
    def create_new_job(self):
        """Create new mining job"""
        self.job_counter += 1
        job_id = f"zion_job_{self.job_counter:06d}"
        
        # RandomX job parameters
        self.current_job = {
            "job_id": job_id,
            "blob": "0606" + secrets.token_hex(73),  # 76 bytes total
            "target": "b88d0600",  # Difficulty target
            "algo": "rx/0",
            "height": 1000 + self.job_counter,
            "seed_hash": secrets.token_hex(32)
        }
        
        print(f"🔨 Created XMRig job: {job_id}")
        return self.current_job
    
    def create_job_for_miner(self, addr):
        """Create job specifically for a miner"""
        if not self.current_job or self.job_counter % 10 == 0:
            self.create_new_job()
            
        job = self.current_job.copy()
        
        # Store job ID for this miner
        if addr in self.miners:
            self.miners[addr]['last_job_id'] = job['job_id']
        
        return {
            "job_id": job["job_id"],
            "blob": job["blob"],
            "target": job["target"],
            "algo": job["algo"],
            "height": job["height"],
            "seed_hash": job["seed_hash"]
        }

async def main():
    pool = XMrigCompatiblePool(port=3333)
    await pool.start_server()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 XMRig Compatible Pool stopped")