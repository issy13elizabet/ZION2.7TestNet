#!/usr/bin/env python3
"""
ZION REAL Stratum Pool Server v2.0
NO SIMULATIONS! Real TCP socket listening on port 3333
Simple but functional stratum protocol implementation
"""

import socket
import threading
import json
import time
import hashlib
import struct
from datetime import datetime
import signal
import sys

class ZionRealStratumPool:
    """REAL Stratum pool - actually listens on TCP port 3333"""
    
    def __init__(self, host='0.0.0.0', port=3333):
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False
        self.clients = []
        self.current_job_id = 0
        self.total_shares = 0
        self.accepted_shares = 0
        self.sessions = {}  # socket -> session_id
        
        print(f"🚀 ZION REAL Stratum Pool Server v2.0")
        print(f"🌐 Will bind to {host}:{port}")
        print(f"⛏️ NO SIMULATIONS - Real TCP socket!")
        
    def log_message(self, message):
        """Log with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
        
    def generate_job(self):
        """Generate new mining job"""
        self.current_job_id += 1
        job_id = f"zion_job_{self.current_job_id}"
        
        # Simple job data
        prev_hash = hashlib.sha256(f"prev_block_{int(time.time())}".encode()).hexdigest()
        coinbase1 = "01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff"
        coinbase2 = "ffffffff01"
        merkle_branch = []
        version = "00000001"  
        nbits = "1d00ffff"  # Difficulty
        ntime = f"{int(time.time()):08x}"
        
        return {
            "job_id": job_id,
            "prev_hash": prev_hash,
            "coinbase1": coinbase1,
            "coinbase2": coinbase2,
            "merkle_branch": merkle_branch,
            "version": version,
            "nbits": nbits,
            "ntime": ntime,
            "clean_jobs": True
        }

    # --- XMRig / Monero style helper (simplified) ---
    def generate_monero_style_job(self):
        """Return a minimal Monero-like job object for XMRig 'login' compatibility.
        NOTE: This is a stub – blob is NOT a valid Monero block template, only enough
        for XMRig to continue producing 'submit' calls which we accept for testing.
        """
        job = self.generate_job()
        # Fake blob: 176 bytes hex (multiple of 8). Use prev_hash + padding.
        base = job["prev_hash"][:64]
        padding = ("0" * 240)  # ensure long enough
        blob = (base + padding)[:352]  # 176 bytes -> 352 hex chars
        # Monero stratum typicky posílá 32-bit LE target (8 hex) – např. "9b0f0000".
        # Chyba code:5 naznačuje problém s délkou/formátem targetu.
        # Použijeme velmi nízkou obtížnost (max target): "ffffffff".
        target = "ffffffff"  # 8 hex = 4 bytes -> parse path case 4
        return {
            "job_id": job["job_id"],
            "blob": blob,
            "target": target,
            "algo": "rx/0",
            "height": 1,
            # extranonce / seed not implemented – for real network extend here
        }
        
    def send_to_client(self, client_socket, message):
        """Send JSON message to client"""
        try:
            msg = json.dumps(message) + '\n'
            client_socket.send(msg.encode())
            return True
        except Exception as e:
            self.log_message(f"⚠️ Send error: {e}")
            return False
            
    def handle_client_message(self, client_socket, client_addr, message):
        """Handle incoming client message"""
        try:
            data = json.loads(message)
            method = data.get('method')
            msg_id = data.get('id')
            params = data.get('params', [])
            
            if method == 'mining.subscribe':
                # Send subscribe response
                response = {
                    "id": msg_id,
                    "result": [
                        [
                            ["mining.set_difficulty", "1"],
                            ["mining.notify", "1"]
                        ],
                        "00000001",  # extranonce1
                        4  # extranonce2_size
                    ],
                    "error": None
                }
                self.send_to_client(client_socket, response)
                self.log_message(f"✅ Client {client_addr} subscribed")
                
                # Send initial job
                job = self.generate_job()
                notify_msg = {
                    "id": None,
                    "method": "mining.notify",
                    "params": [
                        job["job_id"],
                        job["prev_hash"], 
                        job["coinbase1"],
                        job["coinbase2"],
                        job["merkle_branch"],
                        job["version"],
                        job["nbits"],
                        job["ntime"],
                        job["clean_jobs"]
                    ]
                }
                self.send_to_client(client_socket, notify_msg)
                self.log_message(f"📋 Sent job {job['job_id']} to {client_addr}")
                
            elif method == 'mining.authorize':
                # Authorize worker
                response = {
                    "id": msg_id,
                    "result": True,
                    "error": None
                }
                self.send_to_client(client_socket, response)
                self.log_message(f"🔑 Worker authorized: {params[0] if params else 'unknown'}")
                
            elif method == 'mining.submit':
                # Handle share submission
                self.total_shares += 1
                
                # Simple validation - accept all shares for now
                accepted = True
                if accepted:
                    self.accepted_shares += 1
                    
                response = {
                    "id": msg_id,
                    "result": accepted,
                    "error": None if accepted else "Share rejected"
                }
                self.send_to_client(client_socket, response)
                
                status = "✅ ACCEPTED" if accepted else "❌ REJECTED"
                self.log_message(f"📤 Share {self.total_shares}: {status} from {client_addr}")

            # --- XMRig Monero style methods ---
            elif method == 'login':
                # XMRig initial login for Monero-style pools
                # params may be dict: {"login":"addr","pass":"x","agent":"..","rigid":"..","algo":["rx/0"]}
                login_obj = params if isinstance(params, dict) else {}
                user_addr = login_obj.get("login") or login_obj.get("user") or "unknown"
                job = self.generate_monero_style_job()
                # Přidej seed hash hodnoty (RandomX seed) – jednoduché derivace
                seed_hash = hashlib.sha256(job["blob"].encode()).hexdigest()[:64]
                next_seed_hash = hashlib.sha256((seed_hash + "next").encode()).hexdigest()[:64]
                job["seed_hash"] = seed_hash
                job["next_seed_hash"] = next_seed_hash
                # XMRig očekává target jako hex LE -> ponecháme full FFs pro test
                session_id = f"session_{self.current_job_id}"
                self.sessions[client_socket] = session_id
                response = {
                    "id": msg_id,
                    "jsonrpc": "2.0",
                    "result": {
                        "id": session_id,  # session id
                        "job": job,
                        "status": "OK",
                        "extensions": []
                    },
                    "error": None
                }
                self.send_to_client(client_socket, response)
                self.log_message(f"✅ XMRig login accepted for {user_addr} (job_id={job['job_id']} target_len={len(job['target'])} target_head={job['target'][:16]})")

            elif method == 'getjob':
                job = self.generate_monero_style_job()
                response = {
                    "id": msg_id,
                    "jsonrpc": "2.0",
                    "result": job,
                    "error": None
                }
                self.send_to_client(client_socket, response)
                self.log_message(f"📋 getjob served {job['job_id']} -> {client_addr}")

            elif method == 'keepalived':
                # Heartbeat from some miners
                response = {"id": msg_id, "jsonrpc": "2.0", "result": True, "error": None}
                self.send_to_client(client_socket, response)
                self.log_message(f"💓 keepalived from {client_addr}")

            elif method == 'submit':
                # Monero styl share submit (XMRig)
                self.total_shares += 1
                accepted = True  # Test režim – všechno přijímáme
                # params může být dict
                submit_obj = params if isinstance(params, dict) else {}
                jid = submit_obj.get("job_id") or "?"
                nonce = submit_obj.get("nonce") or "?"
                result_hex = submit_obj.get("result") or "?"
                # jednoduché pseudo-ověření délky result
                if not isinstance(result_hex, str) or len(result_hex) < 16:
                    accepted = False
                if accepted:
                    self.accepted_shares += 1
                response = {
                    "id": msg_id,
                    "jsonrpc": "2.0",
                    "result": {"status": "OK", "accepted": accepted},
                    "error": None if accepted else {"code": -1, "message": "invalid result"}
                }
                self.send_to_client(client_socket, response)
                status = "✅ ACCEPTED" if accepted else "❌ REJECTED"
                self.log_message(f"📤 Submit share jid={jid} nonce={nonce} len(res)={len(result_hex)} {status} total={self.total_shares}")
                
            else:
                # Unknown method
                response = {
                    "id": msg_id,
                    "result": None,
                    "error": f"Unknown method: {method}"
                }
                self.send_to_client(client_socket, response)
                
        except Exception as e:
            self.log_message(f"⚠️ Message handling error: {e}")
            
    def handle_client(self, client_socket, client_addr):
        """Handle individual client connection"""
        self.log_message(f"🔗 New client connected: {client_addr}")
        self.clients.append(client_socket)
        
        try:
            buffer = ""
            while self.running:
                data = client_socket.recv(1024).decode()
                if not data:
                    break
                    
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        self.handle_client_message(client_socket, client_addr, line.strip())
                        
        except Exception as e:
            self.log_message(f"⚠️ Client {client_addr} error: {e}")
        finally:
            self.log_message(f"🔌 Client {client_addr} disconnected")
            if client_socket in self.clients:
                self.clients.remove(client_socket)
            client_socket.close()
            
    def broadcast_new_job(self):
        """Broadcast new job to all clients"""
        if not self.clients:
            return
            
        job = self.generate_job()
        notify_msg = {
            "id": None,
            "method": "mining.notify", 
            "params": [
                job["job_id"],
                job["prev_hash"],
                job["coinbase1"], 
                job["coinbase2"],
                job["merkle_branch"],
                job["version"],
                job["nbits"],
                job["ntime"],
                job["clean_jobs"]
            ]
        }
        
        for client in self.clients[:]:  # Copy list to avoid modification during iteration
            if not self.send_to_client(client, notify_msg):
                self.clients.remove(client)
                
        self.log_message(f"📡 Broadcast job {job['job_id']} to {len(self.clients)} clients")
        
    def job_broadcaster(self):
        """Send new jobs periodically"""
        while self.running:
            time.sleep(30)  # New job every 30 seconds
            if self.running:
                self.broadcast_new_job()
                
    def stats_reporter(self):
        """Report pool statistics"""
        while self.running:
            time.sleep(60)  # Stats every minute
            if self.running:
                self.log_message(f"📊 Pool Stats: {len(self.clients)} miners, {self.total_shares} shares ({self.accepted_shares} accepted)")
                
    def start(self):
        """Start the REAL stratum server"""
        try:
            # Create and bind socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            self.log_message(f"🔗 Binding to {self.host}:{self.port}...")
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            
            self.running = True
            self.log_message(f"✅ REAL Stratum server listening on {self.host}:{self.port}")
            self.log_message("🎯 Ready for miners - NO SIMULATIONS!")
            
            # Start background threads
            job_thread = threading.Thread(target=self.job_broadcaster, daemon=True)
            job_thread.start()
            
            stats_thread = threading.Thread(target=self.stats_reporter, daemon=True) 
            stats_thread.start()
            
            # Accept client connections
            while self.running:
                try:
                    client_socket, client_addr = self.server_socket.accept()
                    client_thread = threading.Thread(
                        target=self.handle_client, 
                        args=(client_socket, client_addr),
                        daemon=True
                    )
                    client_thread.start()
                    
                except Exception as e:
                    if self.running:
                        self.log_message(f"⚠️ Accept error: {e}")
                        
        except Exception as e:
            self.log_message(f"❌ Server start error: {e}")
        finally:
            self.stop()
            
    def stop(self):
        """Stop the server"""
        self.log_message("🛑 Stopping REAL Stratum server...")
        self.running = False
        
        # Close all client connections
        for client in self.clients[:]:
            try:
                client.close()
            except:
                pass
        self.clients.clear()
        
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
                
        self.log_message("✅ REAL Stratum server stopped")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Shutting down ZION REAL Stratum Pool...")
    if 'pool' in globals():
        pool.stop()
    sys.exit(0)

def main():
    """Main function"""
    global pool
    
    print("🌟 ZION REAL Stratum Pool Server v2.0")
    print("⛏️ Port: 3333 | Protocol: Stratum | Type: REAL TCP")
    print("🚫 NO SIMULATIONS - Real network socket!")
    print("=" * 60)
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and start pool
    pool = ZionRealStratumPool()
    pool.start()

if __name__ == "__main__":
    main()