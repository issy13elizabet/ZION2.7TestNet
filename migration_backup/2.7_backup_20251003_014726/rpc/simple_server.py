#!/usr/bin/env python3
"""
ZION 2.7 Simple RPC Server - No Dependencies

Basic HTTP RPC server for ZION 2.7 real blockchain using only standard library.
Provides essential CryptoNote-compatible endpoints for mining and monitoring.
"""
import http.server
import socketserver
import json
import urllib.parse
import sys
import os
import threading
import time
from typing import Dict, Any

# Add blockchain to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.blockchain import Blockchain


class ZionRPCHandler(http.server.BaseHTTPRequestHandler):
    """HTTP Request handler for ZION RPC"""
    
    def __init__(self, *args, blockchain=None, **kwargs):
        self.blockchain = blockchain or Blockchain()
        self.request_count = 0
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        try:
            parsed_path = urllib.parse.urlparse(self.path)
            path = parsed_path.path
            
            if path == '/info' or path == '/':
                response = self._handle_getinfo()
            elif path == '/height':
                response = {"height": self.blockchain.height()}
            elif path == '/stats':
                response = self._handle_getstats()
            else:
                self._send_error(404, "Not Found")
                return
            
            self._send_json_response(response)
            
        except Exception as e:
            self._send_error(500, f"Internal Server Error: {str(e)}")
    
    def do_POST(self):
        """Handle POST requests (JSON-RPC)"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            if self.path == '/json_rpc':
                request_body = json.loads(post_data.decode('utf-8'))
                response = self._handle_json_rpc(request_body)
            else:
                self._send_error(404, "Not Found") 
                return
            
            self._send_json_response(response)
            
        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": 1,
                "error": {
                    "code": -32603,
                    "message": "Internal error",
                    "data": str(e)
                }
            }
            self._send_json_response(error_response, status_code=500)
    
    def _handle_json_rpc(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle JSON-RPC requests"""
        method = request.get('method')
        params = request.get('params', {})
        rpc_id = request.get('id', 1)
        
        try:
            if method == 'getinfo':
                result = self._handle_getinfo()
            elif method == 'getheight':
                result = {"height": self.blockchain.height()}
            elif method == 'getblocktemplate':
                result = self._handle_getblocktemplate(params)
            elif method == 'submitblock':
                result = self._handle_submitblock(params)
            elif method == 'getlastblockheader':
                result = self._handle_getlastblockheader()
            elif method == 'getcurrencyid':
                result = {"currency_id_blob": "5a494f4e2d323037"}  # "ZION-2.7" in hex
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": rpc_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method '{method}' not found"
                    }
                }
            
            return {
                "jsonrpc": "2.0", 
                "id": rpc_id,
                "result": result
            }
            
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "error": {
                    "code": -32603,
                    "message": "Internal error", 
                    "data": str(e)
                }
            }
    
    def _handle_getinfo(self) -> Dict[str, Any]:
        """Get blockchain information (CryptoNote-compatible)"""
        info = self.blockchain.info()
        
        return {
            "status": "OK",
            "height": info["height"],
            "difficulty": info["difficulty"],
            "tx_count": info["total_txs"],
            "tx_pool_size": info["tx_pool_size"],
            "alt_blocks_count": 0,
            "outgoing_connections_count": 0,
            "incoming_connections_count": 0,
            "rpc_connections_count": 1,
            "white_peerlist_size": 0,
            "grey_peerlist_size": 0,
            "last_known_block_index": info["height"] - 1,
            "network_height": info["height"],
            "hashrate": info["network_hashrate"],
            "top_block_hash": info["last_hash"],
            "cumulative_difficulty": info["cumulative_difficulty"],
            "block_size_limit": info["max_block_size"],
            "start_time": info["start_time"],
            "version": info["version"],
            "fee": info["min_fee"]
        }
    
    def _handle_getblocktemplate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get block template for mining"""
        wallet_address = params.get('wallet_address', self.blockchain.genesis_address)
        template = self.blockchain.create_block_template(wallet_address)
        
        return {
            "difficulty": template["difficulty"],
            "height": template["height"],
            "reserved_offset": 0,
            "expected_reward": template["base_reward"] + template["total_fees"],
            "prev_hash": template["prev_hash"],
            "blocktemplate_blob": json.dumps(template).encode().hex(),
            "blockhashing_blob": self._create_hashing_blob(template),
            "seed_hash": template["seed_hash"],
            "next_seed_hash": template["next_seed_hash"]
        }
    
    def _handle_submitblock(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Submit mined block"""
        if isinstance(params, list) and len(params) > 0:
            block_blob = params[0]
        else:
            block_blob = params.get('block_blob', '')
        
        if not block_blob:
            raise ValueError("Block blob is required")
        
        try:
            # Decode block blob and submit
            block_data = json.loads(bytes.fromhex(block_blob).decode())
            success = self.blockchain.submit_block(block_data)
            return {"status": "OK" if success else "FAILED"}
        except Exception as e:
            raise ValueError(f"Failed to submit block: {str(e)}")
    
    def _handle_getlastblockheader(self) -> Dict[str, Any]:
        """Get last block header"""
        block = self.blockchain.last_block()
        return {
            "block_header": {
                "major_version": block.major_version,
                "minor_version": block.minor_version,
                "timestamp": block.timestamp,
                "prev_hash": block.prev_hash,
                "nonce": block.nonce,
                "height": block.height,
                "depth": 0,  # Last block has depth 0
                "hash": block.hash,
                "difficulty": block.difficulty,
                "cumulative_difficulty": block.cumulative_difficulty,
                "reward": block.base_reward or 0,
                "block_size": block.block_size,
                "num_txes": len(block.txs),
                "orphan_status": False
            }
        }
    
    def _handle_getstats(self) -> Dict[str, Any]:
        """Extended statistics"""
        info = self._handle_getinfo()
        return {
            **info,
            "rpc_stats": {
                "requests": getattr(self.server, 'request_count', 0),
                "uptime": int(time.time() - getattr(self.server, 'start_time', time.time())),
                "version": "2.7.0-simple"
            },
            "blockchain_stats": {
                "reorg_supported": True,
                "enhanced_features": True,
                "cryptonote_compatible": True
            }
        }
    
    def _create_hashing_blob(self, template: Dict[str, Any]) -> str:
        """Create hashing blob for mining"""
        hash_data = {
            'height': template['height'],
            'prev_hash': template['prev_hash'],
            'timestamp': template['timestamp'],
            'merkle_root': template['merkle_root'],
            'difficulty': template['difficulty']
        }
        return json.dumps(hash_data, sort_keys=True).encode().hex()
    
    def _send_json_response(self, data: Dict[str, Any], status_code: int = 200):
        """Send JSON response"""
        response_data = json.dumps(data, indent=2).encode('utf-8')
        
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response_data)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        self.wfile.write(response_data)
    
    def _send_error(self, status_code: int, message: str):
        """Send error response"""
        error_data = {"error": message, "status_code": status_code}
        self._send_json_response(error_data, status_code)
    
    def log_message(self, format, *args):
        """Custom logging"""
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {format % args}")


class ZionRPCServer:
    """Simple ZION RPC Server"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 17750):
        self.host = host
        self.port = port
        self.blockchain = Blockchain()
        self.start_time = time.time()
        self.request_count = 0
        
        # Create server with custom handler
        def handler_factory(*args, **kwargs):
            return ZionRPCHandler(*args, blockchain=self.blockchain, **kwargs)
        
        self.server = socketserver.TCPServer((host, port), handler_factory)
        self.server.start_time = self.start_time
        self.server.request_count = 0
    
    def start(self):
        """Start the RPC server"""
        print("🚀 ZION 2.7 Simple RPC Server Starting...")
        print(f"📊 Blockchain Height: {self.blockchain.height()}")
        print(f"🔗 Genesis Hash: {self.blockchain.last_block().hash}")
        print(f"🌐 RPC Endpoint: http://{self.host}:{self.port}/json_rpc")
        print(f"📈 Info Endpoint: http://{self.host}:{self.port}/info")
        print(f"📋 Stats Endpoint: http://{self.host}:{self.port}/stats")
        print("Press Ctrl+C to stop...")
        
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Shutting down RPC server...")
            self.server.shutdown()
            self.server.server_close()


def main():
    """Start ZION RPC Server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ZION 2.7 Simple RPC Server")
    parser.add_argument('--host', default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=17750, help='Port number')
    
    args = parser.parse_args()
    
    server = ZionRPCServer(host=args.host, port=args.port)
    server.start()


if __name__ == "__main__":
    main()