#!/usr/bin/env python3
"""
ZION RPC Interface
HTTP-based RPC server for wallet and node communication
"""
import json
import asyncio
from typing import Dict, Any, Optional
import os
import secrets
import time
from collections import deque, defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import logging
from urllib.parse import urlparse, parse_qs

# Avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from new_zion_blockchain import NewZionBlockchain
from crypto_utils import generate_keypair, verify_transaction_signature, tx_hash

logger = logging.getLogger(__name__)

class ZIONRPCServer:
    """HTTP RPC server for ZION blockchain"""

    def __init__(self, blockchain: 'NewZionBlockchain', host: str = '0.0.0.0', port: int = 8332, require_auth: bool = True, auth_token: Optional[str] = None,
                 rate_limit_per_minute: int = 120, burst_limit: int = 20, burst_window_seconds: int = 5):
        self.blockchain = blockchain
        self.host = host
        self.port = port
        self.server = None
        self.running = False
        self.require_auth = require_auth
        self.rate_limit_per_minute = rate_limit_per_minute
        self.burst_limit = burst_limit
        self.burst_window_seconds = burst_window_seconds
        # Determine auth token
        if self.require_auth:
            tok = auth_token or os.environ.get('ZION_RPC_TOKEN')
            if not tok:
                tok = secrets.token_hex(16)
                print(f"[SECURITY] Generated RPC auth token: {tok} (export ZION_RPC_TOKEN to set explicitly)")
            self.auth_token = tok
        else:
            self.auth_token = None

    def start(self):
        """Start the RPC server in a separate thread"""
        if self.running:
            return

        self.running = True
        self.server = ZIONRPCHandler
        self.server.blockchain = self.blockchain
        self.server.require_auth = self.require_auth
        self.server.auth_token = self.auth_token
        # Metrics counters (class-level shared)
        self.server.auth_failures = 0
        self.server.rate_limited = 0
        self.server.rate_limit_per_minute = self.rate_limit_per_minute
        self.server.burst_limit = self.burst_limit
        self.server.burst_window_seconds = self.burst_window_seconds

        def run_server():
            try:
                self.http_server = HTTPServer((self.host, self.port), ZIONRPCHandler)
                logger.info(f"ZION RPC server started on http://{self.host}:{self.port}")
                print(f"🌐 ZION RPC server spuštěn na http://{self.host}:{self.port}")
                self.http_server.serve_forever()
            except Exception as e:
                logger.error(f"RPC server error: {e}")

        self.thread = threading.Thread(target=run_server, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the RPC server"""
        self.running = False
        if hasattr(self, 'http_server') and self.http_server:
            self.http_server.shutdown()
            self.http_server.server_close()
        logger.info("ZION RPC server stopped")

class ZIONRPCHandler(BaseHTTPRequestHandler):
    """HTTP request handler for ZION RPC"""

    blockchain = None  # Will be set by the server
    require_auth: bool = False
    auth_token: Optional[str] = None
    auth_failures: int = 0
    rate_limited: int = 0
    rate_limit_per_minute: int = 0
    burst_limit: int = 0
    burst_window_seconds: int = 0
    _request_windows = defaultdict(deque)  # key -> deque[timestamps]

    def _rate_limit_key(self) -> str:
        # Prefer auth token (masked) else client address
        supplied = self.headers.get('X-ZION-AUTH') or self.headers.get('Authorization') or ''
        if supplied:
            return 'tok:' + supplied[:16]
        return 'ip:' + (self.client_address[0] if self.client_address else 'unknown')

    def _check_rate_limit(self) -> bool:
        now = time.time()
        key = self._rate_limit_key()
        window = type(self)._request_windows[key]
        # Purge entries older than 60s
        while window and now - window[0] > 60:
            window.popleft()
        # Enforce per-minute limit
        if len(window) >= self.rate_limit_per_minute > 0:
            type(self).rate_limited += 1
            self.send_error_response(429, 'Rate limit exceeded (per-minute)')
            return False
        # Burst window
        burst_count = sum(1 for ts in window if now - ts <= self.burst_window_seconds)
        if burst_count >= self.burst_limit > 0:
            type(self).rate_limited += 1
            self.send_error_response(429, 'Rate limit exceeded (burst)')
            return False
        # Record request
        window.append(now)
        return True

    def _check_auth(self) -> bool:
        if not self.require_auth:
            return True
        supplied = self.headers.get('X-ZION-AUTH') or self.headers.get('Authorization')
        if supplied == self.auth_token:
            return True
        # Increment failure metric
        type(self).auth_failures += 1
        self.send_error_response(401, 'Unauthorized')
        return False

    def do_GET(self):
        """Handle GET requests"""
        try:
            path = urlparse(self.path).path
            query = parse_qs(urlparse(self.path).query)

            # Auth (allow homepage without auth for discovery unless locked)
            if path not in ('/','/metrics') and not self._check_auth():
                return
            if path not in ('/', '/metrics') and not self._check_rate_limit():
                return

            if path == '/':
                self.send_homepage()
            elif path == '/api/status':
                self.send_json_response(self.get_status())
            elif path == '/api/block':
                height = query.get('height', [None])[0]
                self.send_json_response(self.get_block(height))
            elif path == '/api/balance':
                address = query.get('address', [None])[0]
                self.send_json_response(self.get_balance(address))
            elif path == '/api/transaction':
                tx_id = query.get('id', [None])[0]
                self.send_json_response(self.get_transaction(tx_id))
            elif path == '/api/blocks':
                limit = int(query.get('limit', ['10'])[0])
                offset = int(query.get('offset', ['0'])[0])
                self.send_json_response(self.get_blocks(limit, offset))
            elif path == '/metrics':
                self.send_metrics()
            else:
                self.send_error_response(404, "Endpoint not found")

        except Exception as e:
            logger.error(f"GET request error: {e}")
            self.send_error_response(500, str(e))

    def do_POST(self):
        """Handle POST requests"""
        try:
            if not self._check_auth():
                return
            if not self._check_rate_limit():
                return
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request = json.loads(post_data.decode())

            method = request.get('method')
            params = request.get('params', [])

            if method == 'getbalance':
                result = self.rpc_get_balance(params)
            elif method == 'sendtransaction':
                result = self.rpc_send_transaction(params)
            elif method == 'getblock':
                result = self.rpc_get_block(params)
            elif method == 'gettransaction':
                result = self.rpc_get_transaction(params)
            elif method == 'getblockcount':
                result = self.rpc_get_block_count(params)
            elif method == 'getnetworkinfo':
                result = self.rpc_get_network_info(params)
            elif method == 'getblockhash':
                result = self.rpc_get_block_hash(params)
            elif method == 'getblockbyhash':
                result = self.rpc_get_block_by_hash(params)
            elif method == 'getmempoolinfo':
                result = self.rpc_get_mempool_info(params)
            elif method == 'getdifficulty':
                result = self.rpc_get_difficulty(params)
            elif method == 'createaddress':
                result = self.rpc_create_address(params)
            elif method == 'submitrawtransaction':
                result = self.rpc_submit_raw_transaction(params, request)
            elif method == 'getblockheader':
                result = self.rpc_get_block_header(params)
            elif method == 'getrawmempool':
                result = self.rpc_get_raw_mempool(params)
            elif method == 'getnonce':
                result = self.rpc_get_nonce(params)
            elif method == 'getmetrics':
                result = self.rpc_get_metrics(params)
            else:
                self.send_json_response({
                    'error': {'code': -32601, 'message': 'Method not found'},
                    'id': request.get('id')
                })
                return

            self.send_json_response({
                'result': result,
                'error': None,
                'id': request.get('id')
            })

        except json.JSONDecodeError:
            self.send_error_response(400, "Invalid JSON")
        except Exception as e:
            logger.error(f"POST request error: {e}")
            self.send_json_response({
                'error': {'code': -32603, 'message': 'Internal error'},
                'id': None
            })

    def send_homepage(self):
        """Send the homepage"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ZION 2.7.4 RPC Interface</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #2c3e50; }
                .endpoint { background: #f8f9fa; padding: 10px; margin: 10px 0; border-left: 4px solid #3498db; }
                code { background: #ecf0f1; padding: 2px 4px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <h1>🚀 ZION 2.7.4 RPC Interface</h1>
            <p>Real blockchain s humanitarian tithe systémem</p>

            <h2>API Endpoints</h2>
            <div class="endpoint">
                <strong>GET /api/status</strong><br>
                Získá status blockchainu
            </div>
            <div class="endpoint">
                <strong>GET /api/block?height=N</strong><br>
                Získá blok podle výšky
            </div>
            <div class="endpoint">
                <strong>GET /api/balance?address=ADDR</strong><br>
                Získá zůstatek adresy
            </div>
            <div class="endpoint">
                <strong>GET /api/blocks?limit=N&offset=M</strong><br>
                Získá seznam bloků
            </div>

            <h2>RPC Methods</h2>
            <div class="endpoint">
                <strong>getbalance</strong> [address]<br>
                Vrátí zůstatek adresy
            </div>
            <div class="endpoint">
                <strong>sendtransaction</strong> [from, to, amount, purpose]<br>
                Odešle transakci
            </div>
            <div class="endpoint">
                <strong>getblock</strong> [height]<br>
                Vrátí blok
            </div>
            <div class="endpoint">
                <strong>getblockcount</strong><br>
                Vrátí počet bloků
            </div>
        </body>
        </html>
        """

        self.wfile.write(html.encode())

    def send_json_response(self, data: Dict):
        """Send JSON response"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())

    def send_error_response(self, code: int, message: str):
        """Send error response"""
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({
            'error': {'code': code, 'message': message}
        }).encode())

    def send_metrics(self):
        """Expose Prometheus-style plaintext metrics."""
        height = len(self.blockchain.blocks)
        supply = self.blockchain.get_total_supply()
        difficulty = self.blockchain.mining_difficulty
        mempool_size = len(self.blockchain.pending_transactions)
        peers = 0
        if hasattr(self.blockchain, 'get_network_status'):
            try:
                peers = self.blockchain.get_network_status().get('connected_peers', 0)
            except Exception:
                peers = 0
        auth_fail = getattr(type(self), 'auth_failures', 0)
        rate_limited = getattr(type(self), 'rate_limited', 0)
        lines = [
            '# HELP zion_block_height Current blockchain height',
            '# TYPE zion_block_height gauge',
            f'zion_block_height {height}',
            '# HELP zion_total_supply Total token supply',
            '# TYPE zion_total_supply gauge',
            f'zion_total_supply {supply}',
            '# HELP zion_difficulty Current mining difficulty (leading zeros)',
            '# TYPE zion_difficulty gauge',
            f'zion_difficulty {difficulty}',
            '# HELP zion_mempool_size Number of transactions in mempool',
            '# TYPE zion_mempool_size gauge',
            f'zion_mempool_size {mempool_size}',
            '# HELP zion_peer_count Connected peer count',
            '# TYPE zion_peer_count gauge',
            f'zion_peer_count {peers}',
            '# HELP zion_rpc_auth_failures Count of failed RPC auth attempts',
            '# TYPE zion_rpc_auth_failures counter',
            f'zion_rpc_auth_failures {auth_fail}',
            '# HELP zion_rpc_rate_limited Count of rate-limited RPC requests',
            '# TYPE zion_rpc_rate_limited counter',
            f'zion_rpc_rate_limited {rate_limited}'
        ]
        data = "\n".join(lines) + "\n"
        self.send_response(200)
        self.send_header('Content-type', 'text/plain; version=0.0.4')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(data.encode())

    # GET API methods

    def get_status(self) -> Dict:
        """Get blockchain status"""
        return {
            'blockchain': {
                'height': len(self.blockchain.blocks),
                'total_supply': self.blockchain.get_total_supply(),
                'difficulty': self.blockchain.mining_difficulty,
                'block_reward': self.blockchain.block_reward
            },
            'network': self.blockchain.get_network_status() if hasattr(self.blockchain, 'get_network_status') else {},
            'version': '2.7.4'
        }

    def get_block(self, height: Optional[str]) -> Dict:
        """Get block by height"""
        if not height:
            return {'error': 'Height parameter required'}

        try:
            h = int(height)
            if 0 <= h < len(self.blockchain.blocks):
                block = self.blockchain.blocks[h]
                return {
                    'height': block['height'],
                    'hash': block['hash'],
                    'previous_hash': block['previous_hash'],
                    'timestamp': block['timestamp'],
                    'transactions': len(block['transactions']),
                    'nonce': block['nonce'],
                    'difficulty': block['difficulty'],
                    'miner': block.get('miner'),
                    'reward': block.get('reward')
                }
            else:
                return {'error': 'Block not found'}
        except ValueError:
            return {'error': 'Invalid height'}

    def get_balance(self, address: Optional[str]) -> Dict:
        """Get balance for address"""
        if not address:
            return {'error': 'Address parameter required'}

        balance = self.blockchain.get_balance(address)
        return {
            'address': address,
            'balance': balance,
            'confirmed': True  # Simplified
        }

    def get_transaction(self, tx_id: Optional[str]) -> Dict:
        """Get transaction by ID"""
        if not tx_id:
            return {'error': 'Transaction ID required'}

        # Simplified - in production, search through all blocks
        for block in self.blockchain.blocks:
            for tx in block['transactions']:
                if tx.get('id') == tx_id:
                    return tx

        return {'error': 'Transaction not found'}

    def get_blocks(self, limit: int, offset: int) -> Dict:
        """Get list of blocks"""
        total_blocks = len(self.blockchain.blocks)
        start = max(0, total_blocks - offset - limit)
        end = max(0, total_blocks - offset)

        blocks = []
        for i in range(start, end):
            block = self.blockchain.blocks[i]
            blocks.append({
                'height': block['height'],
                'hash': block['hash'][:16] + '...',
                'timestamp': block['timestamp'],
                'transactions': len(block['transactions']),
                'miner': block.get('miner')
            })

        return {
            'blocks': blocks,
            'total': total_blocks,
            'limit': limit,
            'offset': offset
        }

    # RPC methods

    def rpc_get_balance(self, params) -> Any:
        """RPC: getbalance [address]"""
        if len(params) < 1:
            return {'error': 'Address required'}
        address = str(params[0])
        if not address or len(address) < 5:
            return {'error': 'Invalid address format'}
        try:
            balance = self.blockchain.get_balance(address)
            return {'address': address, 'balance': balance}
        except Exception as e:
            return {'error': f'Balance lookup failed: {e}'}

    def rpc_send_transaction(self, params) -> Any:
        """RPC: sendtransaction [from, to, amount, purpose]"""
        if len(params) < 3:
            return {'error': 'from, to, amount required'}
        try:
            from_addr = str(params[0])
            to_addr = str(params[1])
            amount = float(params[2])
            if amount <= 0:
                return {'error': 'Amount must be positive'}
            if len(from_addr) < 5 or len(to_addr) < 5:
                return {'error': 'Invalid address length'}
            purpose = params[3] if len(params) > 3 else ""
            tx = self.blockchain.create_transaction(from_addr, to_addr, amount, purpose)
            return {'tx_id': tx['id'], 'status': 'pending'}
        except ValueError as ve:
            return {'error': f'Invalid value: {ve}'}
        except Exception as e:
            return {'error': f'Transaction failed: {e}'}

    def rpc_get_block(self, params) -> Any:
        """RPC: getblock [height]"""
        if len(params) < 1:
            return {'error': 'Height required'}
        try:
            return self.get_block(str(params[0]))
        except Exception as e:
            return {'error': f'Block retrieval failed: {e}'}

    def rpc_get_transaction(self, params) -> Any:
        """RPC: gettransaction [tx_id]"""
        if len(params) < 1:
            return {'error': 'Transaction ID required'}
        try:
            return self.get_transaction(params[0])
        except Exception as e:
            return {'error': f'Transaction lookup failed: {e}'}

    def rpc_get_block_count(self, params) -> Any:
        """RPC: getblockcount"""
        try:
            return len(self.blockchain.blocks)
        except Exception as e:
            return {'error': f'Block count failed: {e}'}

    def rpc_get_network_info(self, params) -> Any:
        """RPC: getnetworkinfo"""
        return {
            'version': '2.7.4',
            'connections': self.blockchain.get_network_status().get('connected_peers', 0) if hasattr(self.blockchain, 'get_network_status') else 0,
            'protocolversion': 1,
            'localservices': 'blockchain,p2p,rpc'
        }

    def rpc_get_block_hash(self, params) -> Any:
        if len(params) < 1:
            return {'error': 'Height required'}
        try:
            h = int(params[0])
            if 0 <= h < len(self.blockchain.blocks):
                return self.blockchain.blocks[h]['hash']
            return {'error': 'Out of range'}
        except Exception as e:
            return {'error': str(e)}

    def rpc_get_block_by_hash(self, params) -> Any:
        if len(params) < 1:
            return {'error': 'Hash required'}
        target = params[0]
        for b in self.blockchain.blocks:
            if b['hash'] == target:
                return b
        return {'error': 'Not found'}

    def rpc_get_block_header(self, params) -> Any:
        if len(params) < 1:
            return {'error': 'Height required'}
        try:
            h = int(params[0])
            if 0 <= h < len(self.blockchain.blocks):
                b = self.blockchain.blocks[h]
                return {
                    'height': b['height'],
                    'hash': b['hash'],
                    'previous_hash': b['previous_hash'],
                    'merkle_root': b.get('merkle_root'),
                    'timestamp': b['timestamp'],
                    'difficulty': b.get('difficulty'),
                    'tx_count': len(b['transactions'])
                }
            return {'error': 'Out of range'}
        except Exception as e:
            return {'error': str(e)}

    def rpc_get_mempool_info(self, params) -> Any:
        return {
            'size': len(self.blockchain.pending_transactions),
            'bytes': 0,  # not tracked in prototype
            'usage': len(self.blockchain.pending_transactions),
            'nonce_tracking': len(getattr(self.blockchain, 'nonces', {}))
        }

    def rpc_get_raw_mempool(self, params) -> Any:
        include_tx = False
        if len(params) > 0 and isinstance(params[0], dict):
            include_tx = params[0].get('verbose', False)
        if include_tx:
            return self.blockchain.pending_transactions
        return [tx.get('id') for tx in self.blockchain.pending_transactions if tx.get('id')]

    def rpc_get_nonce(self, params) -> Any:
        if len(params) < 1:
            return {'error': 'Address required'}
        addr = str(params[0])
        nonce = self.blockchain.nonces.get(addr, 0)
        return {'address': addr, 'nonce': nonce}

    def rpc_get_metrics(self, params) -> Any:
        """Return metrics as structured JSON (alternative to /metrics plaintext)."""
        return {
            'height': len(self.blockchain.blocks),
            'total_supply': self.blockchain.get_total_supply(),
            'difficulty': self.blockchain.mining_difficulty,
            'mempool_size': len(self.blockchain.pending_transactions),
            'peers': self.blockchain.get_network_status().get('connected_peers', 0) if hasattr(self.blockchain,'get_network_status') else 0
        }

    def rpc_get_difficulty(self, params) -> Any:
        return getattr(self.blockchain, 'mining_difficulty', 0)

    def rpc_create_address(self, params) -> Any:
        kp = generate_keypair()
        return {
            'address': kp.address(),
            'public_key': kp.public_key_hex,
            'private_key': kp.private_key_hex  # NOTE: For real deployment do NOT expose like this
        }

    def rpc_submit_raw_transaction(self, params, raw_request) -> Any:
        """Expects a full transaction dict in params[0] with signature & tx_hash."""
        if len(params) < 1:
            return {'error': 'Transaction object required'}
        try:
            tx_obj = params[0]
            # Basic structure validation
            required = {'sender','receiver','amount','timestamp','nonce','signature','tx_hash','public_key'}
            if not required.issubset(set(tx_obj.keys())):
                return {'error': 'Missing transaction fields'}
            # Verify signature
            if tx_hash(tx_obj) != tx_obj['tx_hash']:
                return {'error': 'Hash mismatch'}
            if not verify_transaction_signature(tx_obj, tx_obj['public_key']):
                return {'error': 'Invalid signature'}
            # Accept into mempool
            self.blockchain.add_signed_transaction(tx_obj)
            return {'status': 'accepted', 'tx_hash': tx_obj['tx_hash']}
        except Exception as e:
            return {'error': f'submit failed: {e}'}

    def log_message(self, format, *args):
        """Override to reduce noise"""
        if "GET /api/status" not in format and "GET /" not in format:
            super().log_message(format, *args)