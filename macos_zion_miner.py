#!/usr/bin/env python3
"""
ZION macOS Local Miner
Optimized for local macOS mining with SSH server connection
"""

import os
import sys
import time
import json
import socket
import hashlib
import logging
import threading
import random
from datetime import datetime
import argparse

class MacOSZionMiner:
    def __init__(self, server="91.98.122.165", port=3333, address="", threads=None):
        self.server = server
        self.port = port
        self.address = address
        self.num_threads = threads or min(4, os.cpu_count())
        self.running = False
        
        # Mining stats
        self.hashrate = 0
        self.total_hashes = 0
        self.accepted_shares = 0
        self.rejected_shares = 0
        self.start_time = time.time()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)
        
        # Connection
        self.socket = None
        self.connected = False
        
    def connect_to_pool(self):
        """Connect to ZION mining pool"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)
            self.socket.connect((self.server, self.port))
            self.connected = True
            
            # Send subscribe
            subscribe_msg = {
                "id": 1,
                "method": "mining.subscribe",
                "params": []
            }
            
            message = json.dumps(subscribe_msg) + "\n"
            self.socket.send(message.encode())
            
            # Read response
            response = self.socket.recv(1024).decode().strip()
            self.logger.info(f"📡 Pool response: {response}")
            
            # Send authorize
            auth_msg = {
                "id": 2,
                "method": "mining.authorize",
                "params": [self.address, ""]
            }
            
            message = json.dumps(auth_msg) + "\n"
            self.socket.send(message.encode())
            
            # Read auth response
            auth_response = self.socket.recv(1024).decode().strip()
            self.logger.info(f"🔐 Auth response: {auth_response}")
            
            self.logger.info(f"✅ Connected to pool {self.server}:{self.port}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Connection failed: {e}")
            return False
    
    def start_mining(self):
        """Start the mining process"""
        self.logger.info("🍎 Starting macOS ZION Miner")
        self.logger.info(f"📡 Server: {self.server}:{self.port}")
        self.logger.info(f"💰 Address: {self.address[:20]}...")
        self.logger.info(f"⚡ Threads: {self.num_threads}")
        
        # Connect to pool
        if not self.connect_to_pool():
            return False
            
        # Start mining
        self.running = True
        threads = []
        
        # Stats thread
        stats_thread = threading.Thread(target=self._stats_loop, daemon=True)
        stats_thread.start()
        threads.append(stats_thread)
        
        # Mining threads
        for i in range(self.num_threads):
            mining_thread = threading.Thread(target=self._mining_loop, args=(i,), daemon=True)
            mining_thread.start()
            threads.append(mining_thread)
            
        self.logger.info(f"🚀 Started {self.num_threads} mining threads")
        
        try:
            # Main loop
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Mining interrupted by user")
            
        finally:
            self.running = False
            if self.socket:
                self.socket.close()
            self.logger.info("🏁 Mining stopped")
            
        return True
    
    def _mining_loop(self, thread_id):
        """Mining loop for each thread"""
        self.logger.info(f"⚡ Mining thread {thread_id} started")
        
        while self.running:
            try:
                # Simple mining simulation
                nonce = random.randint(0, 0xFFFFFFFF)
                
                # Create work header
                timestamp = int(time.time())
                header = f"ZION_BLOCK_{timestamp}_{nonce}_{self.address[:10]}"
                
                # Hash with SHA256 (simplified for demo)
                hash_result = hashlib.sha256(header.encode()).hexdigest()
                
                self.total_hashes += 1
                
                # Check for share (difficulty simulation)
                if hash_result.startswith("000"):
                    self.logger.info(f"💎 Thread {thread_id} found share: {hash_result[:20]}...")
                    
                    # Submit share to pool
                    share_msg = {
                        "id": self.total_hashes,
                        "method": "mining.submit",
                        "params": [
                            self.address,
                            f"job_{timestamp}",
                            "00000000",  # extranonce2
                            hex(timestamp)[2:],  # ntime
                            hex(nonce)[2:].zfill(8)  # nonce
                        ]
                    }
                    
                    try:
                        if self.connected and self.socket:
                            message = json.dumps(share_msg) + "\n"
                            self.socket.send(message.encode())
                            self.accepted_shares += 1
                    except:
                        self.rejected_shares += 1
                
                # Yield CPU
                if self.total_hashes % 1000 == 0:
                    time.sleep(0.001)
                    
            except Exception as e:
                self.logger.error(f"❌ Mining thread {thread_id} error: {e}")
                time.sleep(1)
                
    def _stats_loop(self):
        """Statistics reporting loop"""
        last_hashes = 0
        last_time = time.time()
        
        while self.running:
            time.sleep(10)  # Report every 10 seconds
            
            current_time = time.time()
            current_hashes = self.total_hashes
            
            # Calculate hashrate
            time_diff = current_time - last_time
            hash_diff = current_hashes - last_hashes
            
            if time_diff > 0:
                self.hashrate = hash_diff / time_diff
                
            # Runtime
            runtime = current_time - self.start_time
            
            self.logger.info("📊 macOS ZION Miner Stats:")
            self.logger.info(f"    Hashrate: {self.hashrate:.2f} H/s")
            self.logger.info(f"    Total Hashes: {current_hashes}")
            self.logger.info(f"    Accepted/Rejected: {self.accepted_shares}/{self.rejected_shares}")
            self.logger.info(f"    Runtime: {runtime:.1f}s")
            self.logger.info(f"    Connection: {'🟢 Connected' if self.connected else '🔴 Disconnected'}")
            self.logger.info("------------------------------------------------------------")
            
            last_hashes = current_hashes
            last_time = current_time

def main():
    parser = argparse.ArgumentParser(description="macOS ZION Miner")
    parser.add_argument("--server", default="91.98.122.165", help="Mining pool server")
    parser.add_argument("--port", type=int, default=3333, help="Mining pool port")
    parser.add_argument("--address", required=True, help="ZION wallet address")
    parser.add_argument("--threads", type=int, help="Number of mining threads")
    
    args = parser.parse_args()
    
    miner = MacOSZionMiner(
        server=args.server,
        port=args.port,
        address=args.address,
        threads=args.threads
    )
    
    success = miner.start_mining()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()