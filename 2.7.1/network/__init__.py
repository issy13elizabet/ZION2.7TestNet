#!/usr/bin/env python3
"""
ZION 2.7.1 - P2P Network Layer
Basic peer-to-peer networking for blockchain synchronization
"""

import asyncio
import json
import logging
from typing import List, Dict, Set, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class Peer:
    """Represents a network peer"""
    host: str
    port: int
    last_seen: datetime
    version: str = "2.7.1"

    def to_dict(self) -> Dict:
        return {
            'host': self.host,
            'port': self.port,
            'last_seen': self.last_seen.isoformat(),
            'version': self.version
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Peer':
        return cls(
            host=data['host'],
            port=data['port'],
            last_seen=datetime.fromisoformat(data['last_seen']),
            version=data.get('version', 'unknown')
        )

class P2PNetwork:
    """Advanced P2P network implementation with peer discovery and synchronization"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8333):
        self.host = host
        self.port = port
        self.peers: Dict[str, Peer] = {}  # host:port -> Peer
        self.connections: Dict[str, asyncio.StreamWriter] = {}  # active connections
        self.is_running = False
        self.server: Optional[asyncio.Server] = None
        self.blockchain = None  # Will be set later
        self.mempool = None     # Will be set later

        # Expanded seed nodes
        self.seed_nodes = [
            ("seed1.zion.net", 8333),
            ("seed2.zion.net", 8333),
            ("seed3.zion.net", 8333),
            ("zion-node-01.zionchain.org", 8333),
            ("zion-node-02.zionchain.org", 8333),
        ]

        # Network constants
        self.max_peers = 50
        self.min_peers = 3
        self.peer_discovery_interval = 300  # 5 minutes
        self.ping_interval = 60  # 1 minute
        self.connection_timeout = 10

        # Message types
        self.message_types = {
            'handshake': self._handle_handshake,
            'handshake_ack': self._handle_handshake_ack,
            'ping': self._handle_ping,
            'pong': self._handle_pong,
            'get_peers': self._handle_get_peers,
            'peers': self._handle_peers,
            'new_transaction': self._handle_new_transaction,
            'new_block': self._handle_new_block,
            'get_blocks': self._handle_get_blocks,
            'blocks': self._handle_blocks,
            'get_blockchain_info': self._handle_get_blockchain_info,
            'blockchain_info': self._handle_blockchain_info,
        }

    async def start(self):
        """Start the P2P network with full functionality"""
        logger.info(f"Starting P2P network on {self.host}:{self.port}")

        # Start server
        self.server = await asyncio.start_server(
            self._handle_connection,
            self.host,
            self.port
        )

        self.is_running = True
        logger.info("P2P network started")

        # Start background tasks
        asyncio.create_task(self._peer_discovery_loop())
        asyncio.create_task(self._ping_loop())
        asyncio.create_task(self._maintain_connections())

        # Connect to seed nodes
        await self._connect_to_seeds()

    async def _peer_discovery_loop(self):
        """Continuously discover new peers"""
        while self.is_running:
            try:
                await self._discover_peers()
                await asyncio.sleep(self.peer_discovery_interval)
            except Exception as e:
                logger.error(f"Peer discovery error: {e}")
                await asyncio.sleep(60)

    async def _ping_loop(self):
        """Send ping messages to maintain connections"""
        while self.is_running:
            try:
                await self._ping_peers()
                await asyncio.sleep(self.ping_interval)
            except Exception as e:
                logger.error(f"Ping loop error: {e}")
                await asyncio.sleep(30)

    async def _maintain_connections(self):
        """Maintain minimum number of connections"""
        while self.is_running:
            try:
                if len(self.connections) < self.min_peers:
                    await self._connect_to_random_peers()
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"Connection maintenance error: {e}")
                await asyncio.sleep(60)

    async def stop(self):
        """Stop the P2P network"""
        logger.info("Stopping P2P network")
        self.is_running = False

        if self.server:
            self.server.close()
            await self.server.wait_closed()

        logger.info("P2P network stopped")

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming peer connection"""
        peer_addr = writer.get_extra_info('peername')
        logger.info(f"New connection from {peer_addr}")

        try:
            # Basic handshake
            data = await reader.read(1024)
            if data:
                message = json.loads(data.decode())
                if message.get('type') == 'handshake':
                    # Respond to handshake
                    response = {
                        'type': 'handshake_ack',
                        'version': '2.7.1',
                        'timestamp': datetime.now().isoformat()
                    }
                    writer.write(json.dumps(response).encode())
                    await writer.drain()

                    # Add peer
                    peer_key = f"{peer_addr[0]}:{peer_addr[1]}"
                    peer = Peer(
                        host=peer_addr[0],
                        port=peer_addr[1],
                        last_seen=datetime.now()
                    )
                    self.peers[peer_key] = peer
                    logger.info(f"Added peer: {peer.host}:{peer.port}")

        except Exception as e:
            logger.error(f"Error handling connection: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def _discover_peers(self):
        """Discover new peers from connected peers"""
        if not self.connections:
            return

        # Ask random peers for their peer lists
        sample_peers = list(self.connections.keys())[:5]  # Ask up to 5 peers

        for peer_key in sample_peers:
            try:
                await self._send_message(peer_key, {'type': 'get_peers'})
            except Exception as e:
                logger.warning(f"Failed to request peers from {peer_key}: {e}")

    async def _connect_to_random_peers(self):
        """Connect to random known peers to maintain minimum connections"""
        available_peers = [k for k in self.peers.keys() if k not in self.connections]

        if not available_peers:
            return

        # Try to connect to up to 3 random peers
        import random
        peers_to_try = random.sample(available_peers, min(3, len(available_peers)))

        for peer_key in peers_to_try:
            peer = self.peers[peer_key]
            try:
                await self._connect_to_peer(peer.host, peer.port)
            except Exception as e:
                logger.warning(f"Failed to connect to {peer_key}: {e}")

    async def _connect_to_peer(self, host: str, port: int):
        """Connect to a specific peer"""
        peer_key = f"{host}:{port}"

        if peer_key in self.connections:
            return  # Already connected

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=self.connection_timeout
            )

            # Send handshake
            handshake = {
                'type': 'handshake',
                'version': '2.7.1',
                'timestamp': datetime.now().isoformat(),
                'port': self.port
            }

            writer.write(json.dumps(handshake).encode())
            await writer.drain()

            # Wait for response
            response_data = await asyncio.wait_for(
                reader.read(1024),
                timeout=self.connection_timeout
            )

            if response_data:
                response = json.loads(response_data.decode())
                if response.get('type') == 'handshake_ack':
                    # Connection successful
                    self.connections[peer_key] = writer

                    # Add/update peer
                    self.peers[peer_key] = Peer(
                        host=host,
                        port=port,
                        last_seen=datetime.now(),
                        version=response.get('version', 'unknown')
                    )

                    logger.info(f"Connected to peer: {peer_key}")

                    # Start listening for messages from this peer
                    asyncio.create_task(self._listen_to_peer(peer_key, reader))

        except Exception as e:
            logger.warning(f"Failed to connect to {peer_key}: {e}")

    async def _listen_to_peer(self, peer_key: str, reader: asyncio.StreamReader):
        """Listen for messages from a connected peer"""
        try:
            while self.is_running and peer_key in self.connections:
                data = await reader.read(4096)
                if not data:
                    break

                try:
                    message = json.loads(data.decode())
                    await self._handle_message(peer_key, message)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from {peer_key}")

        except Exception as e:
            logger.error(f"Error listening to {peer_key}: {e}")
        finally:
            # Clean up connection
            if peer_key in self.connections:
                self.connections[peer_key].close()
                del self.connections[peer_key]
            logger.info(f"Disconnected from peer: {peer_key}")

    async def _send_message(self, peer_key: str, message: Dict):
        """Send a message to a specific peer"""
        if peer_key not in self.connections:
            return

        try:
            writer = self.connections[peer_key]
            writer.write(json.dumps(message).encode())
            await writer.drain()
        except Exception as e:
            logger.warning(f"Failed to send message to {peer_key}: {e}")
            # Remove failed connection
            if peer_key in self.connections:
                self.connections[peer_key].close()
                del self.connections[peer_key]

    async def _ping_peers(self):
        """Send ping messages to all connected peers"""
        ping_msg = {
            'type': 'ping',
            'timestamp': datetime.now().isoformat()
        }

        for peer_key in list(self.connections.keys()):
            try:
                await self._send_message(peer_key, ping_msg)
            except Exception as e:
                logger.warning(f"Failed to ping {peer_key}: {e}")

    async def _handle_message(self, peer_key: str, message: Dict):
        """Handle incoming message from a peer"""
        msg_type = message.get('type')
        if msg_type in self.message_types:
            try:
                await self.message_types[msg_type](peer_key, message)
            except Exception as e:
                logger.error(f"Error handling {msg_type} from {peer_key}: {e}")
        else:
            logger.warning(f"Unknown message type {msg_type} from {peer_key}")

    # Message handlers
    async def _handle_handshake(self, peer_key: str, message: Dict):
        """Handle handshake message"""
        # Respond with handshake acknowledgment
        response = {
            'type': 'handshake_ack',
            'version': '2.7.1',
            'timestamp': datetime.now().isoformat(),
            'port': self.port
        }
        await self._send_message(peer_key, response)

    async def _handle_handshake_ack(self, peer_key: str, message: Dict):
        """Handle handshake acknowledgment"""
        # Connection is now established
        logger.debug(f"Handshake acknowledged from {peer_key}")

    async def _handle_ping(self, peer_key: str, message: Dict):
        """Handle ping message"""
        # Respond with pong
        pong = {
            'type': 'pong',
            'timestamp': datetime.now().isoformat()
        }
        await self._send_message(peer_key, pong)

    async def _handle_pong(self, peer_key: str, message: Dict):
        """Handle pong message"""
        # Update peer last seen
        if peer_key in self.peers:
            self.peers[peer_key].last_seen = datetime.now()

    async def _handle_get_peers(self, peer_key: str, message: Dict):
        """Handle request for peer list"""
        peers_list = list(self.peers.keys())[:20]  # Send up to 20 peers
        response = {
            'type': 'peers',
            'peers': peers_list,
            'timestamp': datetime.now().isoformat()
        }
        await self._send_message(peer_key, response)

    async def _handle_peers(self, peer_key: str, message: Dict):
        """Handle received peer list"""
        new_peers = message.get('peers', [])
        for peer_addr in new_peers:
            if ':' in peer_addr:
                host, port_str = peer_addr.split(':', 1)
                try:
                    port = int(port_str)
                    peer_key_new = f"{host}:{port}"
                    if peer_key_new not in self.peers and len(self.peers) < self.max_peers:
                        self.peers[peer_key_new] = Peer(
                            host=host,
                            port=port,
                            last_seen=datetime.now()
                        )
                        logger.debug(f"Added peer from {peer_key}: {peer_key_new}")
                except ValueError:
                    continue

    async def _handle_new_transaction(self, peer_key: str, message: Dict):
        """Handle new transaction broadcast"""
        if self.mempool is not None:
            tx_data = message.get('transaction')
            if tx_data:
                # Add to mempool (would validate first in real implementation)
                logger.info(f"Received transaction from {peer_key}")
                # Broadcast to other peers
                for other_peer in self.connections:
                    if other_peer != peer_key:
                        await self._send_message(other_peer, message)

    async def _handle_new_block(self, peer_key: str, message: Dict):
        """Handle new block announcement"""
        if self.blockchain is not None:
            block_data = message.get('block')
            if block_data:
                logger.info(f"Received new block from {peer_key}: height {block_data.get('height', 'unknown')}")
                # Validate and add block (simplified)
                # Broadcast to other peers
                for other_peer in self.connections:
                    if other_peer != peer_key:
                        await self._send_message(other_peer, message)

    async def _handle_get_blocks(self, peer_key: str, message: Dict):
        """Handle request for blocks"""
        if self.blockchain is not None:
            start_height = message.get('start_height', 0)
            count = min(message.get('count', 10), 50)  # Max 50 blocks

            blocks = []
            for i in range(count):
                height = start_height + i
                if height < len(self.blockchain.blocks):
                    block = self.blockchain.blocks[height]
                    blocks.append({
                        'height': block.height,
                        'hash': block.hash,
                        'previous_hash': block.previous_hash,
                        'timestamp': block.timestamp,
                        'transactions': block.transactions,
                        'reward': block.reward,
                        'miner_address': block.miner_address
                    })

            response = {
                'type': 'blocks',
                'blocks': blocks,
                'timestamp': datetime.now().isoformat()
            }
            await self._send_message(peer_key, response)

    async def _handle_blocks(self, peer_key: str, message: Dict):
        """Handle received blocks"""
        if self.blockchain is not None:
            blocks = message.get('blocks', [])
            logger.info(f"Received {len(blocks)} blocks from {peer_key}")
            # Process blocks (validation would happen here)

    async def _handle_get_blockchain_info(self, peer_key: str, message: Dict):
        """Handle request for blockchain information"""
        if self.blockchain is not None:
            info = {
                'type': 'blockchain_info',
                'height': len(self.blockchain.blocks),
                'total_supply': sum(b.reward for b in self.blockchain.blocks),
                'difficulty': self.blockchain.difficulty,
                'timestamp': datetime.now().isoformat()
            }
            await self._send_message(peer_key, info)

    async def _handle_blockchain_info(self, peer_key: str, message: Dict):
        """Handle received blockchain information"""
        peer_height = message.get('height', 0)
        our_height = len(self.blockchain.blocks) if self.blockchain else 0

        if peer_height > our_height:
            logger.info(f"Peer {peer_key} has higher chain ({peer_height} vs {our_height})")
            # Request missing blocks
            await self._send_message(peer_key, {
                'type': 'get_blocks',
                'start_height': our_height,
                'count': peer_height - our_height
            })

    def set_blockchain(self, blockchain):
        """Set blockchain reference for synchronization"""
        self.blockchain = blockchain

    def set_mempool(self, mempool):
        """Set mempool reference for transaction handling"""
        self.mempool = mempool

    def add_peer(self, host: str, port: int):
        """Manually add a peer"""
        peer_key = f"{host}:{port}"
        peer = Peer(
            host=host,
            port=port,
            last_seen=datetime.now()
        )
        self.peers[peer_key] = peer
        logger.info(f"Added peer: {host}:{port}")

    def remove_peer(self, host: str, port: int):
        """Remove a peer"""
        peer_key = f"{host}:{port}"
        if peer_key in self.peers:
            del self.peers[peer_key]
            logger.info(f"Removed peer: {host}:{port}")

    def get_peer_count(self) -> int:
        """Get number of connected peers"""
        return len(self.peers)

    def get_peer_list(self) -> List[Dict]:
        """Get list of peers"""
        return [peer.to_dict() for peer in self.peers.values()]


# Global network instance
_network_instance: Optional[P2PNetwork] = None

def get_network() -> P2PNetwork:
    """Get global network instance"""
    global _network_instance
    if _network_instance is None:
        _network_instance = P2PNetwork()
    return _network_instance

async def start_network():
    """Start the global network"""
    network = get_network()
    await network.start()

async def stop_network():
    """Stop the global network"""
    network = get_network()
    await network.stop()

if __name__ == "__main__":
    # Test network
    async def test():
        network = P2PNetwork()
        await network.start()

        # Wait a bit
        await asyncio.sleep(5)

        print(f"Peers: {network.get_peer_count()}")
        print(f"Peer list: {network.get_peer_list()}")

        await network.stop()

    asyncio.run(test())