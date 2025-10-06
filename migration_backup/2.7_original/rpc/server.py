"""
ZION 2.7 RPC Server - Real Blockchain Integration

FastAPI-based RPC server ported from 2.6.75 and enhanced for ZION 2.7 real blockchain.
Provides CryptoNote-compatible JSON-RPC endpoints with real blockchain integration.
"""
from __future__ import annotations
import asyncio
import json
import time
from typing import Dict, Any, Optional, List
import logging

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.blockchain import Blockchain, Consensus

logger = logging.getLogger(__name__)


class JSONRPCError(Exception):
    """JSON-RPC error with code and message"""
    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"RPC Error {code}: {message}")


class ZionRPCServer:
    """
    ZION 2.7 RPC Server providing CryptoNote-compatible API
    
    Features:
    - JSON-RPC 2.0 protocol
    - CryptoNote compatibility (getinfo, getblocktemplate, submitblock)
    - Real ZION 2.7 blockchain integration (no mockups)
    - Performance monitoring
    - Enhanced from 2.6.75 with reorg support
    """
    
    def __init__(self, blockchain: Blockchain = None, host: str = "0.0.0.0", port: int = 17750):
        self.blockchain = blockchain or Blockchain()
        self.host = host
        self.port = port
        
        # FastAPI app setup
        self.app = FastAPI(
            title="ZION 2.7 RPC Server",
            description="Real blockchain RPC API with 2.6.75 compatibility",
            version="2.7.0"
        )
        
        # CORS middleware for web interface compatibility
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Performance tracking
        self.request_count = 0
        self.start_time = time.time()
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup RPC endpoints"""
        
        @self.app.post("/json_rpc")
        async def json_rpc_handler(request: Request):
            """Main JSON-RPC handler"""
            try:
                body = await request.json()
                self.request_count += 1
                
                # Validate JSON-RPC format
                if not isinstance(body, dict) or 'method' not in body:
                    raise JSONRPCError(-32600, "Invalid Request")
                
                method = body.get('method')
                params = body.get('params', {})
                rpc_id = body.get('id', 1)
                
                # Route to appropriate handler
                if method == 'getinfo':
                    result = await self._handle_getinfo()
                elif method == 'getheight':
                    result = await self._handle_getheight()
                elif method == 'getblocktemplate':
                    result = await self._handle_getblocktemplate(params)
                elif method == 'submitblock':
                    result = await self._handle_submitblock(params)
                elif method == 'getblockheaderbyheight':
                    result = await self._handle_getblockheader(params)
                elif method == 'getlastblockheader':
                    result = await self._handle_getlastblockheader()
                elif method == 'getcurrencyid':
                    result = await self._handle_getcurrencyid()
                else:
                    raise JSONRPCError(-32601, f"Method '{method}' not found")
                
                return {
                    "jsonrpc": "2.0",
                    "id": rpc_id,
                    "result": result
                }
                
            except JSONRPCError as e:
                return {
                    "jsonrpc": "2.0", 
                    "id": body.get('id', 1) if 'body' in locals() else 1,
                    "error": {
                        "code": e.code,
                        "message": e.message,
                        "data": e.data
                    }
                }
            except Exception as e:
                logger.error(f"RPC handler error: {e}")
                return {
                    "jsonrpc": "2.0",
                    "id": body.get('id', 1) if 'body' in locals() else 1,
                    "error": {
                        "code": -32603,
                        "message": "Internal error",
                        "data": str(e)
                    }
                }
        
        @self.app.get("/info")
        async def get_info():
            """Simple REST endpoint for blockchain info"""
            return await self._handle_getinfo()
        
        @self.app.get("/height")
        async def get_height():
            """Simple REST endpoint for blockchain height"""
            return {"height": self.blockchain.height()}
        
        @self.app.get("/stats")
        async def get_stats():
            """Extended statistics endpoint"""
            info = await self._handle_getinfo()
            return {
                **info,
                "rpc_stats": {
                    "requests": self.request_count,
                    "uptime": int(time.time() - self.start_time),
                    "version": "2.7.0"
                }
            }
    
    async def _handle_getinfo(self) -> Dict[str, Any]:
        """Get blockchain information (CryptoNote-compatible)"""
        info = self.blockchain.info()
        
        # Format for CryptoNote compatibility
        return {
            "status": info["status"],
            "height": info["height"],
            "difficulty": info["difficulty"],
            "tx_count": info["total_txs"],
            "tx_pool_size": info["tx_pool_size"], 
            "alt_blocks_count": 0,  # Alternative blocks (forks)
            "outgoing_connections_count": 0,  # P2P connections (not implemented yet)
            "incoming_connections_count": 0,
            "rpc_connections_count": 1,  # This RPC connection
            "white_peerlist_size": 0,  # P2P peer lists (not implemented yet)
            "grey_peerlist_size": 0,
            "last_known_block_index": info["height"] - 1,
            "network_height": info["height"],  # In real P2P, this could differ
            "hashrate": info["network_hashrate"],
            "top_block_hash": info["last_hash"],
            "cumulative_difficulty": info["cumulative_difficulty"],
            "block_size_limit": info["max_block_size"],
            "start_time": info["start_time"],
            "version": info["version"],
            "fee": info["min_fee"]
        }
    
    async def _handle_getheight(self) -> Dict[str, Any]:
        """Get blockchain height"""
        return {
            "height": self.blockchain.height()
        }
    
    async def _handle_getblocktemplate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get block template for mining (CryptoNote-compatible)"""
        wallet_address = params.get('wallet_address', self.blockchain.genesis_address)
        reserve_size = params.get('reserve_size', 0)
        
        # Create block template
        template = self.blockchain.create_block_template(wallet_address)
        
        # Convert to CryptoNote format
        return {
            "difficulty": template["difficulty"],
            "height": template["height"],
            "reserved_offset": 0,  # Reserved bytes offset in block template
            "expected_reward": template["base_reward"] + template["total_fees"],
            "prev_hash": template["prev_hash"],
            "blocktemplate_blob": self._encode_block_template(template),
            "blockhashing_blob": self._create_hashing_blob(template),
            "seed_hash": template["seed_hash"],
            "next_seed_hash": template["next_seed_hash"]
        }
    
    async def _handle_submitblock(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Submit mined block"""
        if isinstance(params, list) and len(params) > 0:
            block_blob = params[0]
        else:
            block_blob = params.get('block_blob', '')
        
        if not block_blob:
            raise JSONRPCError(-1, "Block blob is required")
        
        try:
            # Decode block blob and submit
            block_data = self._decode_block_blob(block_blob)
            success = self.blockchain.submit_block(block_data)
            
            return {
                "status": "OK" if success else "FAILED"
            }
        except Exception as e:
            logger.error(f"Submit block error: {e}")
            raise JSONRPCError(-2, f"Failed to submit block: {str(e)}")
    
    async def _handle_getblockheader(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get block header by height"""
        height = params.get('height', 0)
        block = self.blockchain.get_block_by_height(height)
        
        if not block:
            raise JSONRPCError(-5, f"Block at height {height} not found")
        
        return {
            "block_header": {
                "major_version": block.major_version,
                "minor_version": block.minor_version,
                "timestamp": block.timestamp,
                "prev_hash": block.prev_hash,
                "nonce": block.nonce,
                "height": block.height,
                "depth": self.blockchain.height() - block.height - 1,
                "hash": block.hash,
                "difficulty": block.difficulty,
                "cumulative_difficulty": block.cumulative_difficulty,
                "reward": block.base_reward or 0,
                "block_size": block.block_size,
                "num_txes": len(block.txs),
                "orphan_status": False  # In main chain
            }
        }
    
    async def _handle_getlastblockheader(self) -> Dict[str, Any]:
        """Get last block header"""
        return await self._handle_getblockheader({"height": self.blockchain.height() - 1})
    
    async def _handle_getcurrencyid(self) -> Dict[str, Any]:
        """Get currency ID (network identifier)"""
        return {
            "currency_id_blob": "5a494f4e2d323037"  # "ZION-2.7" in hex
        }
    
    def _encode_block_template(self, template: Dict[str, Any]) -> str:
        """Encode block template to blob format (simplified)"""
        # In real implementation, this would be proper CryptoNote binary serialization
        # For now, we'll use a simplified JSON-hex encoding
        template_json = json.dumps(template, sort_keys=True)
        return template_json.encode().hex()
    
    def _create_hashing_blob(self, template: Dict[str, Any]) -> str:
        """Create hashing blob for mining (simplified)"""
        # This would normally be the block header in binary format for hashing
        # Simplified version using key fields
        hash_data = {
            'height': template['height'],
            'prev_hash': template['prev_hash'], 
            'timestamp': template['timestamp'],
            'merkle_root': template['merkle_root'],
            'difficulty': template['difficulty']
        }
        return json.dumps(hash_data, sort_keys=True).encode().hex()
    
    def _decode_block_blob(self, blob: str) -> Dict[str, Any]:
        """Decode block blob from miner (simplified)"""
        try:
            # Simplified decoding - in real implementation this would be binary deserialization
            json_data = bytes.fromhex(blob).decode()
            return json.loads(json_data)
        except Exception as e:
            raise JSONRPCError(-3, f"Failed to decode block blob: {str(e)}")
    
    async def start_server(self):
        """Start the RPC server"""
        logger.info(f"Starting ZION 2.7 RPC Server on {self.host}:{self.port}")
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()


async def main():
    """Start ZION 2.7 RPC Server"""
    # Initialize blockchain
    blockchain = Blockchain()
    
    # Create and start RPC server
    rpc_server = ZionRPCServer(blockchain=blockchain)
    
    print("🚀 ZION 2.7 RPC Server Starting...")
    print(f"📊 Blockchain Height: {blockchain.height()}")
    print(f"🔗 Genesis Hash: {blockchain.last_block().hash}")
    print(f"🌐 RPC Endpoint: http://{rpc_server.host}:{rpc_server.port}/json_rpc")
    print(f"📈 Info Endpoint: http://{rpc_server.host}:{rpc_server.port}/info")
    
    await rpc_server.start_server()


if __name__ == "__main__":
    asyncio.run(main())