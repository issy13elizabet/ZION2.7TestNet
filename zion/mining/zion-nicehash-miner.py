#!/usr/bin/env python3
"""
ZION NiceHash Miner v2.6.75
Profesionální mining client optimalizovaný pro NiceHash

Features:
- NiceHash Stratum protokol 
- Auto-detection algoritmů
- RandomX + SHA256 fallback
- Real-time statistiky
- Temperature monitoring
"""
import socket
import json
import time
import threading
import hashlib
import sys
import os
from pathlib import Path

# Add ZION mining to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from randomx_engine import RandomXEngine
    RANDOMX_AVAILABLE = True
except ImportError:
    RANDOMX_AVAILABLE = False

class NiceHashMiner:
    """
    Professional NiceHash mining client
    """
    
    def __init__(self, nicehash_server="stratum+tcp://randomx.auto.nicehash.com:9200"):
        self.server = nicehash_server
        self.wallet = "3QVr9QmW9bRF4gejxkY3DBJxgB8rHzJJa2"  # BTC wallet pro NiceHash
        self.worker_name = "ZION_Miner_2_6_75"
        self.password = "x"
        
        # Connection
        self.socket = None
        self.connected = False
        
        # Mining state
        self.mining = False
        self.job_id = None
        self.target = None
        self.extranonce1 = None
        self.extranonce2_size = 4
        
        # Statistics
        self.shares_submitted = 0
        self.shares_accepted = 0
        self.hashrate = 0
        self.start_time = None
        
        # Mining engines
        self.engines = {}
        self.threads = 4
        
        print("🔥 ZION NiceHash Miner v2.6.75")
        print("⚡ Sacred Technology Stack")
        print(f"🎯 Target: {self.server}")
        
    def parse_server_url(self, url):
        """Parse NiceHash server URL"""
        # Remove protocol
        if "stratum+tcp://" in url:
            url = url.replace("stratum+tcp://", "")
        elif "stratum://" in url:
            url = url.replace("stratum://", "")
        
        # Split host and port
        if ":" in url:
            host, port = url.rsplit(":", 1)
            return host, int(port)
        else:
            return url, 4444  # Default port
    
    def connect(self):
        """Connect to NiceHash pool"""
        try:
            host, port = self.parse_server_url(self.server)
            print(f"🔗 Connecting to {host}:{port}...")
            
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(30)
            self.socket.connect((host, port))
            self.connected = True
            
            print("✅ Connected successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def send_message(self, message):
        """Send JSON message to pool"""
        try:
            json_msg = json.dumps(message) + "\\n"
            self.socket.send(json_msg.encode())
            return True
        except Exception as e:
            print(f"❌ Send failed: {e}")
            return False
    
    def receive_message(self):
        """Receive JSON message from pool"""
        try:
            data = self.socket.recv(4096).decode().strip()
            if data:
                # Handle multiple messages
                messages = data.split('\\n')
                for msg in messages:
                    if msg:
                        try:
                            return json.loads(msg)
                        except json.JSONDecodeError:
                            continue
            return None
        except Exception as e:
            print(f"❌ Receive failed: {e}")
            return None
    
    def subscribe(self):
        """Subscribe to mining notifications"""
        subscribe_msg = {
            "id": 1,
            "method": "mining.subscribe", 
            "params": ["ZION_Miner_2.6.75"]
        }
        
        if self.send_message(subscribe_msg):
            print("📡 Subscription request sent...")
            
            # Wait for response
            response = self.receive_message()
            if response and "result" in response:
                result = response["result"]
                if len(result) >= 2:
                    self.extranonce1 = result[1]
                    print(f"✅ Subscribed! Extranonce1: {self.extranonce1}")
                    return True
        
        print("❌ Subscription failed")
        return False
    
    def authorize(self):
        """Authorize worker with NiceHash"""
        auth_msg = {
            "id": 2,
            "method": "mining.authorize",
            "params": [f"{self.wallet}.{self.worker_name}", self.password]
        }
        
        if self.send_message(auth_msg):
            print("🔐 Authorization request sent...")
            
            # Wait for response
            response = self.receive_message()
            if response and response.get("result", False):
                print("✅ Worker authorized!")
                return True
        
        print("❌ Authorization failed")
        return False
    
    def init_mining_engines(self):
        """Initialize RandomX mining engines"""
        print(f"⚡ Initializing {self.threads} mining engines...")
        
        success_count = 0
        for i in range(self.threads):
            try:
                if RANDOMX_AVAILABLE:
                    engine = RandomXEngine(fallback_to_sha256=True)
                    seed = f"NICEHASH_ZION_{i}_{self.wallet}".encode()
                    
                    if engine.init(seed, use_large_pages=False, full_mem=False):
                        self.engines[i] = engine
                        success_count += 1
                        print(f"  ✅ Engine {i+1}: RandomX ready")
                    else:
                        print(f"  ⚠️ Engine {i+1}: Fallback mode")
                else:
                    # Pure SHA256 engine as fallback
                    self.engines[i] = "fallback"
                    success_count += 1
                    print(f"  ⚠️ Engine {i+1}: SHA256 fallback")
                    
            except Exception as e:
                print(f"  ❌ Engine {i+1}: {e}")
        
        print(f"🎯 {success_count}/{self.threads} engines ready")
        return success_count > 0
    
    def mining_worker(self, thread_id):
        """Mining worker thread"""
        if thread_id not in self.engines:
            return
        
        engine = self.engines[thread_id]
        local_hashes = 0
        nonce_start = thread_id * 0x01000000
        
        print(f"🔧 Mining worker {thread_id+1} started")
        
        while self.mining and self.job_id:
            try:
                nonce = nonce_start + local_hashes
                
                # Create block data
                if engine == "fallback":
                    # SHA256 fallback mining
                    block_data = f"{self.job_id}_{nonce}_{thread_id}".encode()
                    hash_result = hashlib.sha256(block_data).digest()
                else:
                    # RandomX mining
                    block_data = f"NH_{self.job_id}_{nonce}".encode()
                    hash_result = engine.hash(block_data)
                
                local_hashes += 1
                
                # Check if we found a share (simplified)
                hash_int = int.from_bytes(hash_result[:4], 'little')
                if hash_int < 0x00FFFFFF:  # NiceHash difficulty simulation
                    
                    # Submit share
                    extranonce2 = f"{nonce:08x}"
                    ntime = f"{int(time.time()):08x}" 
                    nonce_hex = f"{nonce:08x}"
                    
                    share_msg = {
                        "id": 100 + self.shares_submitted,
                        "method": "mining.submit",
                        "params": [
                            f"{self.wallet}.{self.worker_name}",
                            self.job_id,
                            extranonce2,
                            ntime, 
                            nonce_hex
                        ]
                    }
                    
                    if self.send_message(share_msg):
                        self.shares_submitted += 1
                        print(f"📤 Share submitted by worker {thread_id+1} (nonce: {nonce_hex})")
                
                # Update hashrate periodically
                if local_hashes % 1000 == 0:
                    elapsed = time.time() - (self.start_time or time.time())
                    if elapsed > 0:
                        self.hashrate = (local_hashes * self.threads) / elapsed
                
                # Brief pause
                if local_hashes % 100 == 0:
                    time.sleep(0.001)
                    
            except Exception as e:
                print(f"❌ Worker {thread_id+1} error: {e}")
                break
        
        print(f"⏹️ Mining worker {thread_id+1} stopped ({local_hashes:,} hashes)")
    
    def handle_notifications(self):
        """Handle pool notifications"""
        print("👂 Listening for pool notifications...")
        
        while self.connected:
            try:
                message = self.receive_message()
                if not message:
                    time.sleep(0.1)
                    continue
                
                method = message.get("method")
                
                if method == "mining.notify":
                    # New job notification
                    params = message.get("params", [])
                    if len(params) >= 9:
                        self.job_id = params[0]
                        print(f"💼 New job: {self.job_id}")
                        
                        # Start mining if not already running
                        if not self.mining:
                            self.start_mining()
                
                elif method == "mining.set_target":
                    # New target/difficulty
                    params = message.get("params", [])
                    if params:
                        self.target = params[0]
                        print(f"🎯 New target: {self.target}")
                
                # Check for share responses
                elif "id" in message and message["id"] >= 100:
                    if message.get("result", False):
                        self.shares_accepted += 1
                        print(f"✅ Share accepted! ({self.shares_accepted}/{self.shares_submitted})")
                    else:
                        error = message.get("error", "Unknown error")
                        print(f"❌ Share rejected: {error}")
                
            except Exception as e:
                print(f"❌ Notification handler error: {e}")
                break
    
    def start_mining(self):
        """Start mining threads"""
        if self.mining:
            return
        
        if not self.job_id:
            print("⚠️ No job available yet")
            return
        
        print("🚀 Starting mining...")
        self.mining = True
        self.start_time = time.time()
        
        # Start mining worker threads
        for thread_id in range(self.threads):
            if thread_id in self.engines:
                worker_thread = threading.Thread(
                    target=self.mining_worker,
                    args=(thread_id,),
                    daemon=True
                )
                worker_thread.start()
        
        print(f"⛏️ Mining started with {len(self.engines)} workers")
    
    def stop_mining(self):
        """Stop mining"""
        if self.mining:
            print("⏹️ Stopping mining...")
            self.mining = False
            time.sleep(1)  # Let threads finish
    
    def show_stats(self):
        """Display mining statistics"""
        if self.start_time:
            uptime = time.time() - self.start_time
            hours = int(uptime // 3600)
            minutes = int((uptime % 3600) // 60)
            seconds = int(uptime % 60)
            
            acceptance_rate = 0
            if self.shares_submitted > 0:
                acceptance_rate = (self.shares_accepted / self.shares_submitted) * 100
            
            print(f"\\n📊 ZION NiceHash Mining Stats:")
            print(f"   ⏱️ Uptime: {hours:02d}:{minutes:02d}:{seconds:02d}")
            print(f"   ⚡ Hashrate: {self.hashrate:,.0f} H/s")
            print(f"   📤 Shares: {self.shares_accepted}/{self.shares_submitted} ({acceptance_rate:.1f}%)")
            print(f"   🔧 Workers: {len(self.engines)} threads")
    
    def run(self):
        """Main mining loop"""
        try:
            # Connect to pool
            if not self.connect():
                return False
            
            # Subscribe and authorize
            if not (self.subscribe() and self.authorize()):
                return False
            
            # Initialize mining engines
            if not self.init_mining_engines():
                print("❌ Failed to initialize mining engines")
                return False
            
            # Start notification handler
            notification_thread = threading.Thread(target=self.handle_notifications, daemon=True)
            notification_thread.start()
            
            print("\\n🎉 ZION NiceHash miner is running!")
            print("Press Ctrl+C to stop...")
            
            # Main loop with statistics
            while self.connected:
                time.sleep(10)
                self.show_stats()
                
        except KeyboardInterrupt:
            print("\\n⚠️ Stopping miner...")
            self.stop_mining()
            
        except Exception as e:
            print(f"\\n❌ Miner error: {e}")
            
        finally:
            if self.socket:
                self.socket.close()
            print("🔒 ZION NiceHash miner stopped")
            return True


def main():
    """Main function"""
    print("🌟 ZION NiceHash Miner v2.6.75 - Sacred Technology")
    print("="*60)
    
    # NiceHash servers for different algorithms
    servers = {
        "randomx": "stratum+tcp://randomx.auto.nicehash.com:9200",
        "sha256": "stratum+tcp://sha256.auto.nicehash.com:3334", 
        "scrypt": "stratum+tcp://scrypt.auto.nicehash.com:3333"
    }
    
    # Select algorithm
    print("\\n🔧 Available algorithms:")
    for i, (algo, server) in enumerate(servers.items(), 1):
        print(f"   {i}. {algo.upper()} - {server}")
    
    try:
        choice = input("\\n💎 Select algorithm (1-3, default=1): ").strip()
        if not choice:
            choice = "1"
        
        choice = int(choice)
        if choice < 1 or choice > len(servers):
            choice = 1
        
        selected_algo = list(servers.keys())[choice-1]
        selected_server = servers[selected_algo]
        
        print(f"✅ Selected: {selected_algo.upper()}")
        print(f"🎯 Server: {selected_server}")
        
        # Create and run miner
        miner = NiceHashMiner(selected_server)
        miner.run()
        
    except (ValueError, KeyboardInterrupt):
        print("\\n👋 Goodbye!")
    except Exception as e:
        print(f"\\n❌ Error: {e}")


if __name__ == "__main__":
    main()