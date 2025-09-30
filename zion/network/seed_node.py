"""
ZION P2P Networking - Seed Nodes & Network Discovery
Advanced peer-to-peer networking with efficient block propagation
"""

import asyncio
import json
import time
import socket
import struct
import hashlib
import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
from aiohttp import web, WSMsgType
import websockets
import threading
import ipaddress


class MessageType(Enum):
    """P2P message types"""
    HANDSHAKE = 1
    PING = 2
    PONG = 3
    GET_PEERS = 4
    PEERS = 5
    NEW_BLOCK = 6
    NEW_TRANSACTION = 7
    GET_BLOCKS = 8
    BLOCKS = 9
    GET_MEMPOOL = 10
    MEMPOOL = 11
    SYNC_REQUEST = 12
    SYNC_RESPONSE = 13


@dataclass
class PeerInfo:
    """Peer connection information"""
    peer_id: str
    ip: str
    port: int
    version: str
    height: int
    last_seen: int
    connected: bool = False
    websocket: Optional[object] = None


@dataclass
class NetworkMessage:
    """P2P network message"""
    type: MessageType
    data: Dict
    timestamp: int
    sender_id: str
    message_id: str


class ZionNetworking:
    """Advanced P2P networking for ZION blockchain"""
    
    def __init__(self, 
                 node_id: str = None,
                 listen_port: int = 19089,
                 websocket_port: int = 19090,
                 max_peers: int = 50):
        
        # Node configuration
        self.node_id = node_id or self._generate_node_id()
        self.listen_port = listen_port
        self.websocket_port = websocket_port
        self.max_peers = max_peers
        self.version = "2.6.75"
        
        # Network state
        self.peers: Dict[str, PeerInfo] = {}
        self.connected_peers: Set[str] = set()
        self.known_peers: Set[str] = set()
        self.message_cache: Set[str] = set()
        self.blockchain_height = 0
        
        # Seed nodes (hardcoded bootstrap nodes)
        self.seed_nodes = [
            ("seed1.zion-network.org", 19089),
            ("seed2.zion-network.org", 19089),
            ("seed3.zion-network.org", 19089),
            ("127.0.0.1", 19089),  # Local seed for testing
        ]
        
        # Event handlers
        self.on_new_peer = None
        self.on_peer_disconnect = None
        self.on_new_block = None
        self.on_new_transaction = None
        
        # Network services
        self.server: Optional[aiohttp.web.Application] = None
        self.websocket_server = None
        self.running = False
        
        # Setup logging
        self.logger = logging.getLogger(f'zion_network_{self.node_id[:8]}')
        self.logger.setLevel(logging.INFO)
        
    def _generate_node_id(self) -> str:
        """Generate unique node ID"""
        hostname = socket.gethostname()
        timestamp = str(time.time())
        return hashlib.sha256(f"{hostname}{timestamp}".encode()).hexdigest()
    
    async def start(self):
        """Start networking services"""
        self.running = True
        self.logger.info(f"Starting ZION node {self.node_id[:8]} on port {self.listen_port}")
        
        # Start HTTP/WebSocket server
        await self._start_server()
        
        # Connect to seed nodes
        await self._connect_to_seeds()
        
        # Start periodic tasks
        asyncio.create_task(self._peer_discovery_loop())
        asyncio.create_task(self._ping_peers_loop())
        asyncio.create_task(self._cleanup_loop())
        
        self.logger.info("ZION networking started successfully")
    
    async def stop(self):
        """Stop networking services"""
        self.running = False
        
        # Disconnect from all peers
        for peer_id in list(self.connected_peers):
            await self._disconnect_peer(peer_id)
        
        # Stop servers
        if self.server:
            await self.server.shutdown()
        
        self.logger.info("ZION networking stopped")
    
    async def _start_server(self):
        """Start HTTP and WebSocket servers"""
        app = web.Application()
        
        # HTTP endpoints
        app.router.add_get('/peers', self._handle_get_peers)
        app.router.add_post('/message', self._handle_message)
        app.router.add_get('/info', self._handle_get_info)
        
        # WebSocket endpoint
        app.router.add_get('/ws', self._handle_websocket)
        
        # Start HTTP server
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.listen_port)
        await site.start()
        
        self.server = app
        self.logger.info(f"HTTP server started on port {self.listen_port}")
        
        # Start WebSocket server
        try:
            self.websocket_server = await websockets.serve(
                self._handle_websocket_connection,
                '0.0.0.0',
                self.websocket_port
            )
            self.logger.info(f"WebSocket server started on port {self.websocket_port}")
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket server: {e}")
    
    async def _handle_get_peers(self, request):
        """HTTP endpoint: Get peer list"""
        peer_list = []
        for peer in self.peers.values():
            if peer.connected:
                peer_list.append({
                    'peer_id': peer.peer_id,
                    'ip': peer.ip,
                    'port': peer.port,
                    'version': peer.version,
                    'height': peer.height,
                    'last_seen': peer.last_seen
                })
        
        return web.json_response({
            'peers': peer_list,
            'count': len(peer_list),
            'node_id': self.node_id
        })
    
    async def _handle_message(self, request):
        """HTTP endpoint: Receive P2P message"""
        try:
            data = await request.json()
            message = NetworkMessage(**data)
            await self._process_message(message)
            
            return web.json_response({'status': 'ok'})
        except Exception as e:
            return web.json_response({'error': str(e)}, status=400)
    
    async def _handle_get_info(self, request):
        """HTTP endpoint: Get node information"""
        return web.json_response({
            'node_id': self.node_id,
            'version': self.version,
            'height': self.blockchain_height,
            'peers': len(self.connected_peers),
            'uptime': time.time() - getattr(self, 'start_time', time.time()),
            'listen_port': self.listen_port,
            'websocket_port': self.websocket_port
        })
    
    async def _handle_websocket(self, request):
        """HTTP WebSocket upgrade handler"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        # Get peer info from request
        peer_ip = request.remote
        peer_id = request.headers.get('X-Node-ID', self._generate_node_id())
        
        await self._handle_websocket_peer(ws, peer_id, peer_ip)
        return ws
    
    async def _handle_websocket_connection(self, websocket, path):
        """Direct WebSocket connection handler"""
        peer_ip = websocket.remote_address[0]
        peer_id = self._generate_node_id()  # Will be updated during handshake
        
        await self._handle_websocket_peer(websocket, peer_id, peer_ip)
    
    async def _handle_websocket_peer(self, websocket, peer_id: str, peer_ip: str):
        """Handle WebSocket peer connection"""
        try:
            self.logger.info(f"New WebSocket connection from {peer_ip}")
            
            # Send handshake
            handshake = NetworkMessage(
                type=MessageType.HANDSHAKE,
                data={
                    'node_id': self.node_id,
                    'version': self.version,
                    'height': self.blockchain_height,
                    'timestamp': int(time.time())
                },
                timestamp=int(time.time()),
                sender_id=self.node_id,
                message_id=self._generate_message_id()
            )
            
            await self._send_websocket_message(websocket, handshake)
            
            # Handle incoming messages
            async for message in websocket:
                if hasattr(message, 'type'):  # aiohttp WebSocket
                    if message.type == WSMsgType.TEXT:
                        data = json.loads(message.data)
                    elif message.type == WSMsgType.ERROR:
                        break
                    else:
                        continue
                else:  # websockets library
                    data = json.loads(message)
                
                network_message = NetworkMessage(**data)
                await self._process_websocket_message(network_message, websocket, peer_id, peer_ip)
                
        except Exception as e:
            self.logger.error(f"WebSocket error with {peer_ip}: {e}")
        finally:
            if peer_id in self.connected_peers:
                await self._disconnect_peer(peer_id)
    
    async def _connect_to_seeds(self):
        """Connect to seed nodes for network bootstrap"""
        self.logger.info("Connecting to seed nodes...")
        
        for host, port in self.seed_nodes:
            try:
                # Skip self-connection
                if host == "127.0.0.1" and port == self.listen_port:
                    continue
                
                # Try HTTP connection first
                await self._connect_to_peer_http(host, port)
                
                # Then try WebSocket connection
                await self._connect_to_peer_websocket(host, port + 1)  # WS port = HTTP + 1
                
            except Exception as e:
                self.logger.debug(f"Failed to connect to seed {host}:{port}: {e}")
    
    async def _connect_to_peer_http(self, host: str, port: int):
        """Connect to peer via HTTP"""
        try:
            url = f"http://{host}:{port}/info"
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        peer = PeerInfo(
                            peer_id=data['node_id'],
                            ip=host,
                            port=port,
                            version=data['version'],
                            height=data['height'],
                            last_seen=int(time.time()),
                            connected=True
                        )
                        
                        self.peers[peer.peer_id] = peer
                        self.connected_peers.add(peer.peer_id)
                        self.logger.info(f"Connected to peer {peer.peer_id[:8]} via HTTP")
                        
                        # Get more peers from this peer
                        await self._request_peers_from_http(host, port)
                        
        except Exception as e:
            self.logger.debug(f"HTTP connection to {host}:{port} failed: {e}")
    
    async def _connect_to_peer_websocket(self, host: str, port: int):
        """Connect to peer via WebSocket"""
        try:
            uri = f"ws://{host}:{port}/ws"
            
            async with websockets.connect(uri, timeout=5) as websocket:
                peer_id = await self._websocket_handshake(websocket, host, port)
                
                if peer_id:
                    # Keep connection alive in background
                    asyncio.create_task(self._maintain_websocket_connection(websocket, peer_id, host, port))
                    
        except Exception as e:
            self.logger.debug(f"WebSocket connection to {host}:{port} failed: {e}")
    
    async def _websocket_handshake(self, websocket, host: str, port: int) -> Optional[str]:
        """Perform WebSocket handshake with peer"""
        try:
            # Send handshake
            handshake = NetworkMessage(
                type=MessageType.HANDSHAKE,
                data={
                    'node_id': self.node_id,
                    'version': self.version,
                    'height': self.blockchain_height,
                    'timestamp': int(time.time())
                },
                timestamp=int(time.time()),
                sender_id=self.node_id,
                message_id=self._generate_message_id()
            )
            
            await self._send_websocket_message(websocket, handshake)
            
            # Wait for response
            response = await asyncio.wait_for(websocket.recv(), timeout=5)
            response_data = json.loads(response)
            response_message = NetworkMessage(**response_data)
            
            if response_message.type == MessageType.HANDSHAKE:
                peer_id = response_message.data['node_id']
                
                # Create peer info
                peer = PeerInfo(
                    peer_id=peer_id,
                    ip=host,
                    port=port,
                    version=response_message.data['version'],
                    height=response_message.data['height'],
                    last_seen=int(time.time()),
                    connected=True,
                    websocket=websocket
                )
                
                self.peers[peer_id] = peer
                self.connected_peers.add(peer_id)
                self.logger.info(f"WebSocket handshake successful with {peer_id[:8]}")
                
                return peer_id
            
        except Exception as e:
            self.logger.debug(f"WebSocket handshake failed with {host}:{port}: {e}")
        
        return None
    
    async def _maintain_websocket_connection(self, websocket, peer_id: str, host: str, port: int):
        """Maintain WebSocket connection with peer"""
        try:
            while self.running and peer_id in self.connected_peers:
                # Send ping periodically
                ping_message = NetworkMessage(
                    type=MessageType.PING,
                    data={'timestamp': int(time.time())},
                    timestamp=int(time.time()),
                    sender_id=self.node_id,
                    message_id=self._generate_message_id()
                )
                
                await self._send_websocket_message(websocket, ping_message)
                
                # Wait for messages
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=30)
                    data = json.loads(message)
                    network_message = NetworkMessage(**data)
                    await self._process_websocket_message(network_message, websocket, peer_id, host)
                    
                except asyncio.TimeoutError:
                    # No message received, continue with ping
                    continue
                    
        except Exception as e:
            self.logger.warning(f"Lost connection to peer {peer_id[:8]}: {e}")
            await self._disconnect_peer(peer_id)
    
    async def _send_websocket_message(self, websocket, message: NetworkMessage):
        """Send message via WebSocket"""
        try:
            data = json.dumps(asdict(message), default=str)
            if hasattr(websocket, 'send_str'):  # aiohttp WebSocket
                await websocket.send_str(data)
            else:  # websockets library
                await websocket.send(data)
        except Exception as e:
            self.logger.error(f"Failed to send WebSocket message: {e}")
    
    async def _process_websocket_message(self, message: NetworkMessage, websocket, peer_id: str, peer_ip: str):
        """Process incoming WebSocket message"""
        # Update peer last seen
        if peer_id in self.peers:
            self.peers[peer_id].last_seen = int(time.time())
        
        # Handle different message types
        if message.type == MessageType.PING:
            # Respond with pong
            pong = NetworkMessage(
                type=MessageType.PONG,
                data={'timestamp': int(time.time())},
                timestamp=int(time.time()),
                sender_id=self.node_id,
                message_id=self._generate_message_id()
            )
            await self._send_websocket_message(websocket, pong)
            
        elif message.type == MessageType.GET_PEERS:
            # Send peer list
            peer_list = []
            for peer in self.peers.values():
                if peer.connected and peer.peer_id != peer_id:
                    peer_list.append({
                        'peer_id': peer.peer_id,
                        'ip': peer.ip,
                        'port': peer.port,
                        'version': peer.version,
                        'height': peer.height
                    })
            
            peers_message = NetworkMessage(
                type=MessageType.PEERS,
                data={'peers': peer_list},
                timestamp=int(time.time()),
                sender_id=self.node_id,
                message_id=self._generate_message_id()
            )
            await self._send_websocket_message(websocket, peers_message)
        
        elif message.type == MessageType.PEERS:
            # Add new peers to known peers
            for peer_data in message.data.get('peers', []):
                if peer_data['peer_id'] not in self.peers:
                    await self._try_connect_to_peer(peer_data['ip'], peer_data['port'])
        
        elif message.type == MessageType.NEW_BLOCK:
            # Handle new block broadcast
            if self.on_new_block:
                await self.on_new_block(message.data)
            
            # Relay to other peers
            await self._broadcast_message(message, exclude=peer_id)
            
        elif message.type == MessageType.NEW_TRANSACTION:
            # Handle new transaction broadcast
            if self.on_new_transaction:
                await self.on_new_transaction(message.data)
            
            # Relay to other peers
            await self._broadcast_message(message, exclude=peer_id)
    
    async def _broadcast_message(self, message: NetworkMessage, exclude: str = None):
        """Broadcast message to all connected peers"""
        # Prevent message loops
        if message.message_id in self.message_cache:
            return
        
        self.message_cache.add(message.message_id)
        
        # Clean old messages from cache
        if len(self.message_cache) > 10000:
            self.message_cache = set(list(self.message_cache)[-5000:])
        
        # Send to all connected peers
        for peer_id in list(self.connected_peers):
            if peer_id == exclude:
                continue
            
            peer = self.peers.get(peer_id)
            if peer and peer.websocket:
                try:
                    await self._send_websocket_message(peer.websocket, message)
                except Exception as e:
                    self.logger.warning(f"Failed to broadcast to {peer_id[:8]}: {e}")
                    await self._disconnect_peer(peer_id)
    
    async def _disconnect_peer(self, peer_id: str):
        """Disconnect from peer"""
        if peer_id in self.connected_peers:
            self.connected_peers.remove(peer_id)
        
        if peer_id in self.peers:
            peer = self.peers[peer_id]
            peer.connected = False
            
            if peer.websocket:
                try:
                    if hasattr(peer.websocket, 'close'):
                        await peer.websocket.close()
                except Exception:
                    pass
                peer.websocket = None
            
            if self.on_peer_disconnect:
                await self.on_peer_disconnect(peer_id)
            
            self.logger.info(f"Disconnected from peer {peer_id[:8]}")
    
    async def _peer_discovery_loop(self):
        """Periodic peer discovery"""
        while self.running:
            try:
                # Request peers from connected nodes
                for peer_id in list(self.connected_peers):
                    peer = self.peers.get(peer_id)
                    if peer and peer.websocket:
                        get_peers_msg = NetworkMessage(
                            type=MessageType.GET_PEERS,
                            data={},
                            timestamp=int(time.time()),
                            sender_id=self.node_id,
                            message_id=self._generate_message_id()
                        )
                        await self._send_websocket_message(peer.websocket, get_peers_msg)
                
                # Try to connect to more peers if below threshold
                if len(self.connected_peers) < 5:
                    await self._connect_to_seeds()
                
            except Exception as e:
                self.logger.error(f"Peer discovery error: {e}")
            
            await asyncio.sleep(60)  # Run every minute
    
    async def _ping_peers_loop(self):
        """Periodic ping to check peer connectivity"""
        while self.running:
            try:
                for peer_id in list(self.connected_peers):
                    peer = self.peers.get(peer_id)
                    if peer:
                        # Check if peer is still responsive
                        if time.time() - peer.last_seen > 120:  # 2 minutes timeout
                            self.logger.warning(f"Peer {peer_id[:8]} not responding, disconnecting")
                            await self._disconnect_peer(peer_id)
                
            except Exception as e:
                self.logger.error(f"Ping peers error: {e}")
            
            await asyncio.sleep(30)  # Run every 30 seconds
    
    async def _cleanup_loop(self):
        """Periodic cleanup of old data"""
        while self.running:
            try:
                current_time = time.time()
                
                # Remove old peers
                for peer_id in list(self.peers.keys()):
                    peer = self.peers[peer_id]
                    if not peer.connected and current_time - peer.last_seen > 3600:  # 1 hour
                        del self.peers[peer_id]
                
                # Clean message cache
                if len(self.message_cache) > 5000:
                    self.message_cache = set(list(self.message_cache)[-2500:])
                
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
            
            await asyncio.sleep(300)  # Run every 5 minutes
    
    def _generate_message_id(self) -> str:
        """Generate unique message ID"""
        return hashlib.sha256(f"{self.node_id}{time.time()}{id(self)}".encode()).hexdigest()[:16]
    
    async def broadcast_new_block(self, block_data: Dict):
        """Broadcast new block to network"""
        message = NetworkMessage(
            type=MessageType.NEW_BLOCK,
            data=block_data,
            timestamp=int(time.time()),
            sender_id=self.node_id,
            message_id=self._generate_message_id()
        )
        
        await self._broadcast_message(message)
        self.logger.info(f"Broadcasted new block {block_data.get('hash', 'unknown')[:8]}")
    
    async def broadcast_new_transaction(self, tx_data: Dict):
        """Broadcast new transaction to network"""
        message = NetworkMessage(
            type=MessageType.NEW_TRANSACTION,
            data=tx_data,
            timestamp=int(time.time()),
            sender_id=self.node_id,
            message_id=self._generate_message_id()
        )
        
        await self._broadcast_message(message)
        self.logger.info(f"Broadcasted new transaction {tx_data.get('txid', 'unknown')[:8]}")
    
    def get_network_stats(self) -> Dict:
        """Get current network statistics"""
        connected_count = len(self.connected_peers)
        total_peers = len(self.peers)
        
        return {
            'node_id': self.node_id,
            'connected_peers': connected_count,
            'total_known_peers': total_peers,
            'blockchain_height': self.blockchain_height,
            'version': self.version,
            'uptime': time.time() - getattr(self, 'start_time', time.time())
        }


# Seed node implementation
class ZionSeedNode(ZionNetworking):
    """Specialized seed node for network bootstrap"""
    
    def __init__(self, seed_id: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.seed_id = seed_id
        self.node_id = f"seed{seed_id}_{self.node_id[:16]}"
        
        # Seed nodes maintain more connections
        self.max_peers = 200
        
        # Enhanced logging for seed nodes
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f'%(asctime)s [SEED{seed_id}] %(levelname)s: %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    async def start(self):
        """Start seed node with enhanced connectivity"""
        self.start_time = time.time()
        await super().start()
        
        # Seed nodes are more aggressive in peer discovery
        asyncio.create_task(self._enhanced_peer_discovery())
        asyncio.create_task(self._seed_statistics_loop())
        
        self.logger.info(f"ZION Seed Node {self.seed_id} started on port {self.listen_port}")
    
    async def _enhanced_peer_discovery(self):
        """Enhanced peer discovery for seed nodes"""
        while self.running:
            try:
                # More frequent peer discovery
                if len(self.connected_peers) < 10:
                    await self._connect_to_seeds()
                
                # Seed nodes also listen for new peer announcements
                await self._announce_seed_presence()
                
            except Exception as e:
                self.logger.error(f"Enhanced peer discovery error: {e}")
            
            await asyncio.sleep(30)  # Every 30 seconds
    
    async def _announce_seed_presence(self):
        """Announce seed node presence to network"""
        # This would typically involve DHT or other discovery mechanisms
        self.logger.debug(f"Seed {self.seed_id} announcing presence to network")
    
    async def _seed_statistics_loop(self):
        """Periodic statistics logging for seed nodes"""
        while self.running:
            try:
                stats = self.get_network_stats()
                self.logger.info(
                    f"Seed {self.seed_id} stats: "
                    f"{stats['connected_peers']} connected, "
                    f"{stats['total_known_peers']} known peers, "
                    f"height {stats['blockchain_height']}"
                )
            except Exception as e:
                self.logger.error(f"Statistics error: {e}")
            
            await asyncio.sleep(300)  # Every 5 minutes


# CLI interface for running seed node
async def run_seed_node():
    """Run a ZION seed node"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ZION Seed Node')
    parser.add_argument('--seed-id', type=int, default=1, help='Seed node ID')
    parser.add_argument('--port', type=int, default=19089, help='Listen port')
    parser.add_argument('--ws-port', type=int, default=19090, help='WebSocket port')
    parser.add_argument('--max-peers', type=int, default=200, help='Maximum peers')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and start seed node
    seed = ZionSeedNode(
        seed_id=args.seed_id,
        listen_port=args.port,
        websocket_port=args.ws_port,
        max_peers=args.max_peers
    )
    
    try:
        print(f"🌱 Starting ZION Seed Node {args.seed_id}...")
        print(f"   HTTP Port: {args.port}")
        print(f"   WebSocket Port: {args.ws_port}")
        print(f"   Max Peers: {args.max_peers}")
        print("   Press Ctrl+C to stop")
        
        await seed.start()
        
        # Keep running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down seed node...")
        await seed.stop()
        print("✅ Seed node stopped")


if __name__ == "__main__":
    asyncio.run(run_seed_node())