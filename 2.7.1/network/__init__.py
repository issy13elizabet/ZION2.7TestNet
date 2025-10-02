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
    """Basic P2P network implementation"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8333):
        self.host = host
        self.port = port
        self.peers: Set[Peer] = set()
        self.is_running = False
        self.server: Optional[asyncio.Server] = None

        # Seed nodes for initial connection
        self.seed_nodes = [
            ("seed1.zion.net", 8333),
            ("seed2.zion.net", 8333),
        ]

    async def start(self):
        """Start the P2P network"""
        logger.info(f"Starting P2P network on {self.host}:{self.port}")

        # Start server
        self.server = await asyncio.start_server(
            self._handle_connection,
            self.host,
            self.port
        )

        self.is_running = True
        logger.info("P2P network started")

        # Connect to seed nodes
        await self._connect_to_seeds()

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
                    peer = Peer(
                        host=peer_addr[0],
                        port=peer_addr[1],
                        last_seen=datetime.now()
                    )
                    self.peers.add(peer)
                    logger.info(f"Added peer: {peer.host}:{peer.port}")

        except Exception as e:
            logger.error(f"Error handling connection: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def _connect_to_seeds(self):
        """Connect to seed nodes"""
        for host, port in self.seed_nodes:
            try:
                logger.info(f"Connecting to seed node {host}:{port}")
                # For now, just log - actual connection logic would go here
                # In a real implementation, we'd attempt TCP connections
                logger.info(f"Seed connection simulation: {host}:{port}")
            except Exception as e:
                logger.warning(f"Failed to connect to seed {host}:{port}: {e}")

    async def broadcast_transaction(self, tx_data: Dict):
        """Broadcast a new transaction to all peers"""
        message = {
            'type': 'new_transaction',
            'transaction': tx_data,
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"Broadcasting transaction to {len(self.peers)} peers")
        for peer in self.peers:
            logger.debug(f"Would send to {peer.host}:{peer.port}")

    async def sync_blockchain(self):
        """Synchronize blockchain with peers"""
        if not self.peers:
            logger.warning("No peers available for sync")
            return

        # Get our current height
        our_height = 0  # Would get from blockchain
        message = {
            'type': 'get_blockchain_info',
            'timestamp': datetime.now().isoformat()
        }

        logger.info("Requesting blockchain sync from peers")

    def add_peer(self, host: str, port: int):
        """Manually add a peer"""
        peer = Peer(
            host=host,
            port=port,
            last_seen=datetime.now()
        )
        self.peers.add(peer)
        logger.info(f"Added peer: {host}:{port}")

    def remove_peer(self, host: str, port: int):
        """Remove a peer"""
        self.peers = {p for p in self.peers if not (p.host == host and p.port == port)}
        logger.info(f"Removed peer: {host}:{port}")

    def get_peer_count(self) -> int:
        """Get number of connected peers"""
        return len(self.peers)

    def get_peer_list(self) -> List[Dict]:
        """Get list of peers"""
        return [peer.to_dict() for peer in self.peers]


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