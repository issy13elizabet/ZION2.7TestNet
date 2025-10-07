#!/usr/bin/env python3
"""
ZION P2P Network Module
Implements peer-to-peer networking for blockchain consensus and block broadcasting
"""
import asyncio
import json
import socket
import time
import threading
import logging
from typing import Dict, List, Optional, Set, TYPE_CHECKING
from dataclasses import dataclass, field
import hashlib

# Avoid circular import
if TYPE_CHECKING:
    from new_zion_blockchain import NewZionBlockchain

logger = logging.getLogger(__name__)

@dataclass
class Peer:
    """Represents a network peer"""
    host: str
    port: int
    last_seen: float = field(default_factory=time.time)
    connected: bool = False
    version: str = "2.7.4"

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

@dataclass
class NetworkMessage:
    """P2P network message structure"""
    type: str
    data: Dict
    timestamp: float = field(default_factory=time.time)
    sender: str = ""

    def to_json(self) -> str:
        return json.dumps({
            'type': self.type,
            'data': self.data,
            'timestamp': self.timestamp,
            'sender': self.sender
        })

    @classmethod
    def from_json(cls, json_str: str) -> 'NetworkMessage':
        data = json.loads(json_str)
        return cls(
            type=data['type'],
            data=data['data'],
            timestamp=data.get('timestamp', time.time()),
            sender=data.get('sender', '')
        )

class ZIONP2PNetwork:
    """P2P network for ZION blockchain consensus"""

    def __init__(self, blockchain: 'NewZionBlockchain', host: str = '0.0.0.0', port: int = 8333):
        self.blockchain = blockchain
        self.host = host
        self.port = port
        self.peers: Dict[str, Peer] = {}
        self.server = None
        self.running = False
        # Map peer_key -> writer for outbound messaging
        self._peer_writers: Dict[str, asyncio.StreamWriter] = {}

        # Network configuration
        self.max_peers = 10
        self.ping_interval = 30  # seconds
        self.peer_timeout = 300  # 5 minutes

        # Seed nodes for initial peer discovery
        self.seed_nodes = [
            'seed.zion.network:8333',
            'seed2.zion.network:8333',
            # Add more seed nodes as needed
        ]

        # Message handlers
        self.message_handlers = {
            'ping': self.handle_ping,
            'pong': self.handle_pong,
            'get_blocks': self.handle_get_blocks,
            'blocks': self.handle_blocks,
            'new_block': self.handle_new_block,
            'get_peers': self.handle_get_peers,
            'peers': self.handle_peers,
            'version': self.handle_version,
        }
        # Alternative branch storage: fork_start_height -> list[blocks]
        self.alt_branches: Dict[int, List[Dict]] = {}

    def _record_alternative_block(self, block_data: Dict):
        """Store alternative chain block for potential multi-block reorg."""
        h = block_data['height']
        # Identify fork start height (height-1)
        fork_start = h - 1
        if fork_start < 0:
            return
        branch = self.alt_branches.setdefault(fork_start, [])
        # Reject if already have a block at this height in branch
        if any(b['height'] == h and b['hash'] == block_data['hash'] for b in branch):
            return
        # If first block ensure previous hash matches main chain at fork start
        if not branch:
            if fork_start >= len(self.blockchain.blocks):
                return
            if block_data['previous_hash'] != self.blockchain.blocks[fork_start]['hash']:
                return
            branch.append(block_data)
        else:
            # Append only if links to last alt block
            if block_data['previous_hash'] == branch[-1]['hash']:
                branch.append(block_data)
            else:
                return
        # After adding, evaluate reorg
        self._maybe_trigger_reorg(fork_start, branch)

    def _branch_cumulative_work(self, fork_start: int, branch: List[Dict]) -> int:
        base_cw = 0
        if fork_start >= 0 and fork_start < len(self.blockchain.blocks):
            base_cw = self.blockchain.blocks[fork_start].get('cumulative_work', 0)
        work_sum = 0
        for b in branch:
            diff = b.get('difficulty', self.blockchain.mining_difficulty)
            work_sum += 2 ** diff
        return base_cw + work_sum

    def _main_chain_cw_at_height(self, height: int) -> int:
        if height < len(self.blockchain.blocks):
            return self.blockchain.blocks[height].get('cumulative_work', 0)
        return self.blockchain.blocks[-1].get('cumulative_work', 0) if self.blockchain.blocks else 0

    def _maybe_trigger_reorg(self, fork_start: int, branch: List[Dict]):
        if not branch:
            return
        last_height = branch[-1]['height']
        if (last_height - fork_start) > self.blockchain.max_reorg_depth:
            return  # too deep
        alt_cw = self._branch_cumulative_work(fork_start, branch)
        main_cw = self._main_chain_cw_at_height(min(last_height, len(self.blockchain.blocks)-1))
        # Reorg condition: alternative cumulative work strictly greater
        if alt_cw > main_cw:
            # Attempt reorg
            try:
                applied = self.blockchain.apply_alternative_branch(fork_start, branch)
                if applied:
                    logger.info(f"Reorg applied at fork {fork_start} replacing {len(branch)} blocks (alt cw {alt_cw} > main cw {main_cw})")
                    # Clear branches that overlap obsolete heights
                    to_del = [fs for fs in self.alt_branches if fs >= fork_start]
                    for fs in to_del:
                        del self.alt_branches[fs]
                else:
                    logger.warning("Failed to apply alternative branch despite higher work")
            except Exception as e:
                logger.error(f"Error during reorg attempt: {e}")

    async def start(self):
        """Start the P2P network"""
        self.running = True
        logger.info(f"Starting ZION P2P network on {self.host}:{self.port}")

        # Start server
        self.server = await asyncio.start_server(
            self.handle_connection, self.host, self.port
        )

        # Connect to seed nodes
        await self.connect_to_seeds()

        # Peer security / scoring state
        self.peer_scores: Dict[str, int] = {}
        self.banned_peers: Dict[str, float] = {}  # peer_key -> ban_until (epoch)
        self.peer_ping_times: Dict[str, list] = {}
        # Configuration thresholds
        self.score_penalties = {
            'invalid_block': 25,
            'bad_timestamp': 15,
            'malformed_message': 10,
            'spam_ping': 5,
            'unknown_message': 5
        }
        self.ban_threshold = 100
        self.ban_duration = 900  # seconds (15m)
        self.score_decay_interval = 60  # seconds
        self.score_decay_amount = 5

        # Start maintenance & watchdog tasks
        asyncio.create_task(self.peer_maintenance())
        asyncio.create_task(self.broadcast_new_blocks())
        asyncio.create_task(self._score_decay_loop())

        logger.info(f"ZION P2P network started with {len(self.peers)} initial peers")

    async def stop(self):
        """Stop the P2P network"""
        self.running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        logger.info("ZION P2P network stopped")

    async def connect_to_seeds(self):
        """Connect to seed nodes for initial peer discovery"""
        for seed in self.seed_nodes:
            try:
                host, port_str = seed.split(':')
                port = int(port_str)
                await self.connect_to_peer(host, port)
            except Exception as e:
                logger.debug(f"Failed to connect to seed {seed}: {e}")

    async def connect_to_peer(self, host: str, port: int):
        """Connect to a specific peer"""
        peer_key = f"{host}:{port}"

        if peer_key in self.peers:
            return  # Already connected

        try:
            reader, writer = await asyncio.open_connection(host, port)

            peer = Peer(host, port, connected=True)
            self.peers[peer_key] = peer

            # Store writer for messaging
            self._peer_writers[peer_key] = writer

            # Start peer handler
            asyncio.create_task(self.handle_peer(peer, reader, writer))

            logger.info(f"Connected to peer {peer_key}")

        except Exception as e:
            logger.debug(f"Failed to connect to {peer_key}: {e}")

    async def handle_connection(self, reader, writer):
        """Handle incoming connections"""
        peer_addr = writer.get_extra_info('peername')
        host, port = peer_addr[0], peer_addr[1]
        peer_key = f"{host}:{port}"

        # Reject banned peers immediately
        if self._is_banned(peer_key):
            logger.warning(f"Rejected banned peer {peer_key}")
            writer.close()
            await writer.wait_closed()
            return

        peer = Peer(host, port, connected=True)
        self.peers[peer_key] = peer
        # Save writer for outbound messages
        self._peer_writers[peer_key] = writer

        logger.info(f"New connection from {peer_key}")

        await self.handle_peer(peer, reader, writer)

    async def handle_peer(self, peer: Peer, reader, writer):
        """Handle communication with a connected peer"""
        try:
            while self.running:
                data = await reader.read(4096)
                if not data:
                    break

                message_str = data.decode().strip()
                if message_str:
                    await self.process_message(message_str, peer, writer)

        except Exception as e:
            logger.debug(f"Error handling peer {peer.address}: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
            peer.connected = False
            logger.info(f"Disconnected from peer {peer.address}")
            # Cleanup writer reference
            peer_key = f"{peer.host}:{peer.port}"
            if peer_key in self._peer_writers:
                del self._peer_writers[peer_key]

    async def process_message(self, message_str: str, peer: Peer, writer):
        """Process incoming network message"""
        try:
            message = NetworkMessage.from_json(message_str)
            peer.last_seen = time.time()

            handler = self.message_handlers.get(message.type)
            if handler:
                await handler(message, peer, writer)
            else:
                logger.warning(f"Unknown message type: {message.type}")
                self._apply_penalty(peer, 'unknown_message')

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from {peer.address}: {message_str}")
            self._apply_penalty(peer, 'malformed_message')
        except Exception as e:
            logger.error(f"Error processing message from {peer.address}: {e}")
            self._apply_penalty(peer, 'malformed_message')

    async def send_message(self, peer: Peer, message: NetworkMessage):
        """Send message to a peer"""
        if not peer.connected:
            return
        peer_key = f"{peer.host}:{peer.port}"
        writer = self._peer_writers.get(peer_key)
        if not writer:
            return
        try:
            data = message.to_json() + "\n"
            writer.write(data.encode())
            await writer.drain()
        except Exception as e:
            logger.debug(f"Failed to send message to {peer.address}: {e}")

    async def broadcast_message(self, message: NetworkMessage, exclude_peer: Optional[Peer] = None):
        """Broadcast message to all connected peers"""
        for peer in self.peers.values():
            if peer.connected and peer != exclude_peer:
                await self.send_message(peer, message)

    # Message handlers

    async def handle_ping(self, message: NetworkMessage, peer: Peer, writer):
        """Handle ping message"""
        pong_msg = NetworkMessage(
            type='pong',
            data={'timestamp': message.timestamp},
            sender=self.get_node_id()
        )
        await self.send_message(peer, pong_msg)
        # Track pings for spam detection
        pk = peer.address
        times = self.peer_ping_times.setdefault(pk, [])
        now = time.time()
        times.append(now)
        # Keep only last 10 seconds
        self.peer_ping_times[pk] = [t for t in times if now - t <= 10]
        if len(self.peer_ping_times[pk]) > 20:  # >20 pings in 10s window
            self._apply_penalty(peer, 'spam_ping')

    async def handle_pong(self, message: NetworkMessage, peer: Peer, writer):
        """Handle pong message"""
        # Update peer latency if needed
        pass

    async def handle_version(self, message: NetworkMessage, peer: Peer, writer):
        """Handle version handshake"""
        peer.version = message.data.get('version', 'unknown')

        # Send our version
        version_msg = NetworkMessage(
            type='version',
            data={
                'version': '2.7.4',
                'height': len(self.blockchain.blocks),
                'timestamp': time.time()
            },
            sender=self.get_node_id()
        )
        await self.send_message(peer, version_msg)

    async def handle_get_blocks(self, message: NetworkMessage, peer: Peer, writer):
        """Handle request for blocks"""
        start_height = message.data.get('start_height', 0)
        blocks = self.blockchain.blocks[start_height:start_height+100]  # Send up to 100 blocks

        blocks_data = []
        for block in blocks:
            blocks_data.append({
                'height': block['height'],
                'hash': block['hash'],
                'previous_hash': block['previous_hash'],
                'timestamp': block['timestamp'],
                'transactions': block['transactions'],
                'nonce': block['nonce'],
                'difficulty': block['difficulty']
            })

        blocks_msg = NetworkMessage(
            type='blocks',
            data={'blocks': blocks_data},
            sender=self.get_node_id()
        )
        await self.send_message(peer, blocks_msg)

    async def handle_blocks(self, message: NetworkMessage, peer: Peer, writer):
        """Handle received blocks"""
        blocks_data = message.data.get('blocks', [])

        for block_data in blocks_data:
            await self.process_received_block(block_data, peer)

    async def handle_new_block(self, message: NetworkMessage, peer: Peer, writer):
        """Handle notification of new block"""
        block_data = message.data.get('block')
        if block_data:
            await self.process_received_block(block_data, peer)

    async def handle_get_peers(self, message: NetworkMessage, peer: Peer, writer):
        """Handle request for peer list"""
        peers_data = []
        for p in self.peers.values():
            if p.connected:
                peers_data.append({
                    'host': p.host,
                    'port': p.port,
                    'version': p.version
                })

        peers_msg = NetworkMessage(
            type='peers',
            data={'peers': peers_data},
            sender=self.get_node_id()
        )
        await self.send_message(peer, peers_msg)

    async def handle_peers(self, message: NetworkMessage, peer: Peer, writer):
        """Handle received peer list"""
        peers_data = message.data.get('peers', [])

        for peer_data in peers_data:
            host = peer_data.get('host')
            port = peer_data.get('port')
            if host and port and len(self.peers) < self.max_peers:
                await self.connect_to_peer(host, port)

    async def process_received_block(self, block_data: Dict, peer: Peer):
        """Process a block received from network"""
        try:
            # Validate block
            if self.validate_block(block_data):
                block_height = block_data['height']
                # Case 1: extends our chain directly
                if block_height == len(self.blockchain.blocks):
                    if self.blockchain.accept_network_block(block_data):
                        logger.info(f"Added new block {block_height} from {peer.address}")
                    else:
                        logger.warning(f"Failed to accept block {block_height} from {peer.address}")
                # Case 2: fork / competing height
                elif block_height > len(self.blockchain.blocks):
                    # Collect as alternative branch candidate (simplified)
                    logger.info(f"Received future block {block_height}; ignoring (out-of-order)")
                else:
                    # Potential fork with different data
                    local_block = self.blockchain.blocks[block_height]
                    if local_block['hash'] != block_data['hash']:
                        logger.warning(f"Fork detected at height {block_height}; storing alternative block")
                        self._record_alternative_block(block_data)
            else:
                logger.warning(f"Invalid block received from {peer.address}")
                self._apply_penalty(peer, 'invalid_block')

        except Exception as e:
            logger.error(f"Error processing block from {peer.address}: {e}")
            self._apply_penalty(peer, 'invalid_block')

    def validate_block(self, block_data: Dict) -> bool:
        """Validate a block received from network"""
        try:
            # Basic validation
            required_fields = ['height', 'hash', 'previous_hash', 'transactions', 'nonce', 'difficulty']
            for field in required_fields:
                if field not in block_data:
                    return False

            # Validate hash
            # This is a simplified validation - in production, validate the full block
            # Timestamp (MTP + drift) validation
            if not self.blockchain.validate_block_timestamp_external(block_data):
                # Timestamp violation penalty
                # (Penalty applied by caller if validation returns False)
                return False
            return True

        except Exception as e:
            logger.error(f"Block validation error: {e}")
            return False

    async def add_block_to_chain(self, block_data: Dict):
        """Add a validated block to the blockchain"""
        # Deprecated by accept_network_block
        self.blockchain.accept_network_block(block_data)

    async def broadcast_new_block(self, block_data: Dict):
        """Broadcast a new block to the network"""
        message = NetworkMessage(
            type='new_block',
            data={'block': block_data},
            sender=self.get_node_id()
        )
        await self.broadcast_message(message)

    async def request_blocks_from_peer(self, peer: Peer, start_height: int):
        """Request blocks from a specific peer"""
        message = NetworkMessage(
            type='get_blocks',
            data={'start_height': start_height},
            sender=self.get_node_id()
        )
        await self.send_message(peer, message)

    async def peer_maintenance(self):
        """Maintain peer connections and remove stale peers"""
        while self.running:
            await asyncio.sleep(self.ping_interval)

            # Remove stale peers
            current_time = time.time()
            stale_peers = []
            for peer_key, peer in self.peers.items():
                if current_time - peer.last_seen > self.peer_timeout:
                    stale_peers.append(peer_key)

            for peer_key in stale_peers:
                del self.peers[peer_key]
                logger.debug(f"Removed stale peer {peer_key}")

            # Ping active peers
            for peer in self.peers.values():
                if peer.connected:
                    ping_msg = NetworkMessage(
                        type='ping',
                        data={'timestamp': time.time()},
                        sender=self.get_node_id()
                    )
                    await self.send_message(peer, ping_msg)

    async def broadcast_new_blocks(self):
        """Broadcast newly mined blocks"""
        last_broadcast_height = len(self.blockchain.blocks) - 1

        while self.running:
            await asyncio.sleep(10)  # Check every 10 seconds

            current_height = len(self.blockchain.blocks) - 1
            if current_height > last_broadcast_height:
                # Broadcast new blocks
                for height in range(last_broadcast_height + 1, current_height + 1):
                    block = self.blockchain.blocks[height]
                    block_data = {
                        'height': block['height'],
                        'hash': block['hash'],
                        'previous_hash': block['previous_hash'],
                        'timestamp': block['timestamp'],
                        'transactions': block['transactions'],
                        'nonce': block['nonce'],
                        'difficulty': block['difficulty'],
                        'miner': block.get('miner'),
                        'reward': block.get('reward', 50.0)
                    }
                    await self.broadcast_new_block(block_data)

                last_broadcast_height = current_height

    def get_node_id(self) -> str:
        """Get unique node identifier"""
        return f"{self.host}:{self.port}"

    def get_network_status(self) -> Dict:
        """Get network status information"""
        return {
            'connected_peers': len([p for p in self.peers.values() if p.connected]),
            'total_peers': len(self.peers),
            'listening_port': self.port,
            'node_id': self.get_node_id(),
            'banned_peers': len(self.banned_peers),
            'peer_scores': {k: self.peer_scores.get(k,0) for k in list(self.peer_scores)[:10]}  # sample
        }

    # ---------------- Peer Scoring / Banning ---------------- #
    def _apply_penalty(self, peer: Peer, reason: str):
        pk = peer.address
        if self._is_banned(pk):
            return
        points = self.score_penalties.get(reason, 5)
        new_score = self.peer_scores.get(pk, 0) + points
        self.peer_scores[pk] = new_score
        logger.debug(f"Penalty {points} for {pk} ({reason}); score={new_score}")
        if new_score >= self.ban_threshold:
            self.banned_peers[pk] = time.time() + self.ban_duration
            logger.warning(f"Peer {pk} banned for {self.ban_duration}s (score {new_score})")
            # Disconnect if connected
            if pk in self._peer_writers:
                try:
                    self._peer_writers[pk].close()
                except Exception:
                    pass

    def _is_banned(self, peer_key: str) -> bool:
        until = self.banned_peers.get(peer_key)
        if not until:
            return False
        if time.time() > until:
            # Expire ban
            del self.banned_peers[peer_key]
            self.peer_scores[peer_key] = 0
            return False
        return True

    async def _score_decay_loop(self):
        while self.running:
            await asyncio.sleep(self.score_decay_interval)
            to_remove = []
            for pk, score in list(self.peer_scores.items()):
                if self._is_banned(pk):
                    continue
                new_score = max(0, score - self.score_decay_amount)
                self.peer_scores[pk] = new_score
                if new_score == 0:
                    to_remove.append(pk)
            for pk in to_remove:
                del self.peer_scores[pk]