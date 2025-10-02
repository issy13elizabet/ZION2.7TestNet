#!/usr/bin/env python3
"""
ZION Local Mining Pool Test v2.6.75
Test mining pool server pro demonstraci ZION mineru

Simuluje Stratum protokol pro lokální testování
"""
import socket
import json
import threading
import time
import secrets

class ZionTestPool:
    """
    Test mining pool server pro ZION miner
    """
    
    def __init__(self, port=4444):
        self.port = port
        self.clients = {}
        self.job_counter = 1
        self.running = False
        
        print(f"🏊 ZION Test Pool v2.6.75")
        print(f"🌊 Port: {port}")
    
    def create_job(self):
        """Create new mining job"""
        job_id = f"zion_job_{self.job_counter:04d}"
        self.job_counter += 1
        
        # Generate random job data
        prevhash = secrets.token_hex(32)
        coinb1 = secrets.token_hex(20)
        coinb2 = secrets.token_hex(20)
        merkle_branches = []
        version = "20000000"
        nbits = "1a44b9f2"
        ntime = f"{int(time.time()):08x}"
        clean_jobs = True
        
        job = {
            "id": job_id,
            "prevhash": prevhash,
            "coinb1": coinb1,
            "coinb2": coinb2,
            "merkle_branches": merkle_branches,
            "version": version,
            "nbits": nbits,
            "ntime": ntime,
            "clean_jobs": clean_jobs
        }
        
        return job
    
    def handle_client(self, client_socket, client_address):
        """Handle client connection"""
        client_id = f"{client_address[0]}:{client_address[1]}"
        print(f"🔗 Client connected: {client_id}")
        
        try:
            while self.running:
                # Receive message
                data = client_socket.recv(1024).decode().strip()
                if not data:
                    break
                
                try:
                    message = json.loads(data)
                    response = self.process_message(message, client_id)
                    
                    if response:
                        response_json = json.dumps(response) + "\\n"
                        client_socket.send(response_json.encode())
                        
                except json.JSONDecodeError:
                    print(f"❌ Invalid JSON from {client_id}: {data}")
                    
        except Exception as e:
            print(f"❌ Client {client_id} error: {e}")
        finally:
            if client_id in self.clients:
                del self.clients[client_id]
            client_socket.close()
            print(f"📤 Client disconnected: {client_id}")
    
    def process_message(self, message, client_id):
        """Process client message"""
        method = message.get("method")
        msg_id = message.get("id")
        params = message.get("params", [])
        
        if method == "mining.subscribe":
            # Subscribe request
            print(f"📡 Subscribe request from {client_id}")
            
            extranonce1 = secrets.token_hex(4)
            extranonce2_size = 4
            
            self.clients[client_id] = {
                "subscribed": True,
                "authorized": False,
                "extranonce1": extranonce1,
                "worker": None
            }
            
            # Send subscription response
            response = {
                "id": msg_id,
                "result": [
                    [["mining.set_difficulty", "1"], ["mining.notify", "1"]], 
                    extranonce1,
                    extranonce2_size
                ],
                "error": None
            }
            
            # Send initial difficulty
            threading.Timer(0.1, self.send_difficulty, args=[client_id, client_socket]).start()
            
            # Send initial job
            threading.Timer(0.2, self.send_job, args=[client_id, client_socket]).start()
            
            return response
            
        elif method == "mining.authorize":
            # Authorization request
            if len(params) >= 1:
                worker_name = params[0]
                print(f"🔐 Authorization request: {worker_name}")
                
                if client_id in self.clients:
                    self.clients[client_id]["authorized"] = True
                    self.clients[client_id]["worker"] = worker_name
                
                return {
                    "id": msg_id,
                    "result": True,
                    "error": None
                }
        
        elif method == "mining.submit":
            # Share submission
            if len(params) >= 5:
                worker = params[0]
                job_id = params[1] 
                extranonce2 = params[2]
                ntime = params[3]
                nonce = params[4]
                
                print(f"💎 Share submitted by {worker}: job={job_id}, nonce={nonce}")
                
                # Simulate acceptance (90% acceptance rate)
                accepted = secrets.randbelow(10) < 9
                
                if accepted:
                    print(f"✅ Share accepted from {worker}")
                else:
                    print(f"❌ Share rejected from {worker}")
                
                return {
                    "id": msg_id,
                    "result": accepted,
                    "error": None if accepted else ["stale", "Stale share"]
                }
        
        return None
    
    def send_difficulty(self, client_id, client_socket):
        """Send difficulty to client"""
        try:
            difficulty_msg = {
                "id": None,
                "method": "mining.set_difficulty",
                "params": [1.0]  # Easy difficulty for testing
            }
            
            msg_json = json.dumps(difficulty_msg) + "\\n"
            client_socket.send(msg_json.encode())
            print(f"🎯 Difficulty sent to {client_id}")
            
        except Exception as e:
            print(f"❌ Failed to send difficulty to {client_id}: {e}")
    
    def send_job(self, client_id, client_socket):
        """Send mining job to client"""
        try:
            job = self.create_job()
            
            job_msg = {
                "id": None,
                "method": "mining.notify", 
                "params": [
                    job["id"],
                    job["prevhash"],
                    job["coinb1"],
                    job["coinb2"], 
                    job["merkle_branches"],
                    job["version"],
                    job["nbits"],
                    job["ntime"],
                    job["clean_jobs"]
                ]
            }
            
            msg_json = json.dumps(job_msg) + "\\n"
            client_socket.send(msg_json.encode())
            print(f"💼 Job {job['id']} sent to {client_id}")
            
        except Exception as e:
            print(f"❌ Failed to send job to {client_id}: {e}")
    
    def broadcast_new_jobs(self):
        """Periodically broadcast new jobs"""
        while self.running:
            time.sleep(30)  # New job every 30 seconds
            
            if self.clients:
                print(f"📢 Broadcasting new job to {len(self.clients)} clients...")
                
                for client_id, client_info in list(self.clients.items()):
                    if client_info.get("authorized"):
                        try:
                            # This is simplified - in real implementation we'd need socket references
                            print(f"  📤 Job broadcast to {client_id}")
                        except:
                            pass
    
    def start(self):
        """Start pool server"""
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind(('localhost', self.port))
            server_socket.listen(5)
            
            self.running = True
            
            print(f"🚀 ZION Test Pool started on localhost:{self.port}")
            print("⛏️ Ready for miners!")
            
            # Start job broadcaster
            job_thread = threading.Thread(target=self.broadcast_new_jobs, daemon=True)
            job_thread.start()
            
            while self.running:
                try:
                    client_socket, client_address = server_socket.accept()
                    
                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, client_address),
                        daemon=True
                    )
                    client_thread.start()
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ Server error: {e}")
                    
        except Exception as e:
            print(f"❌ Failed to start pool: {e}")
        finally:
            self.running = False
            if 'server_socket' in locals():
                server_socket.close()
            print("🔒 ZION Test Pool stopped")


if __name__ == "__main__":
    pool = ZionTestPool(port=4444)
    
    try:
        pool.start()
    except KeyboardInterrupt:
        print("\\n⚠️ Shutting down pool...")
        pool.running = False