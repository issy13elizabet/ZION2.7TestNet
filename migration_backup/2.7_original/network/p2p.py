"""
ZION 2.7 P2P (Phase 2 incremental)
Protocol (line-delimited JSON, unauthenticated early testnet):
    hello      -> initial handshake {type:"hello", height, node_id, timestamp}
    announce   -> new block header notification {type:"announce", height, hash}
    inv        -> inventory of block hashes {type:"inv", hashes:[...], tip_height}
    getblocks  -> request blocks by hash list {type:"getblocks", hashes:[...]} (max 50)
    block      -> full block payload {type:"block", block:{...}}
Sync Strategy (simple):
    After hello, if peer height > ours, ask for recent window via inv request.
    On announce if behind, request the announced hash via getblocks.
Security: None yet (Phase 2). No fork resolution beyond longest chain (implicit sequential growth assumption).
"""
import socket, threading, json, time, uuid
from typing import List, Dict, Any, Optional
import hashlib

class PeerConn:
    def __init__(self, sock: socket.socket, addr):
        self.sock = sock
        self.addr = addr
        self.buffer = b''
        self.alive = True

    def send(self, obj: Dict[str, Any]):
        try:
            self.sock.send((json.dumps(obj) + '\n').encode())
        except OSError:
            self.alive = False

class P2PNode:
    def __init__(self, host='0.0.0.0', port=29876, chain=None):
        self.host = host
        self.port = port
        self.node_id = str(uuid.uuid4())[:8]
        self.chain = chain
        self.peers: List[PeerConn] = []
        self.running = False
        self.MAX_BLOCK_REQUEST = 50
        self._seen_txs = set()
        # Security limits (very lightweight testnet heuristics)
        self.MAX_MESSAGE_SIZE = 64 * 1024  # 64 KB per line
        self.MAX_TX_PER_MINUTE = 120
        self.MAX_BLOCK_ANNOUNCE_PER_MINUTE = 120
        self._tx_timestamps: List[float] = []
        self._block_announce_timestamps: List[float] = []

    def start(self):
        t = threading.Thread(target=self._listen_loop, daemon=True)
        self.running = True
        t.start()

    def connect(self, host: str, port: int):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((host, port))
            peer = PeerConn(s, (host, port))
            self.peers.append(peer)
            self._send_hello(peer)
            threading.Thread(target=self._peer_loop, args=(peer,), daemon=True).start()
        except Exception:
            pass

    def _listen_loop(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, self.port))
        s.listen(20)
        while self.running:
            try:
                client, addr = s.accept()
                peer = PeerConn(client, addr)
                self.peers.append(peer)
                threading.Thread(target=self._peer_loop, args=(peer,), daemon=True).start()
            except Exception:
                break

    def _peer_loop(self, peer: PeerConn):
        self._send_hello(peer)
        while peer.alive and self.running:
            try:
                data = peer.sock.recv(4096)
                if not data:
                    break
                peer.buffer += data
                # Hard cap buffer size to prevent memory abuse
                if len(peer.buffer) > self.MAX_MESSAGE_SIZE * 4:
                    peer.alive = False
                    break
                while b'\n' in peer.buffer:
                    line, peer.buffer = peer.buffer.split(b'\n',1)
                    if line:
                        if len(line) > self.MAX_MESSAGE_SIZE:
                            continue  # drop oversized
                        self._handle(peer, line)
            except Exception:
                break
        peer.alive = False

    def _send_hello(self, peer: PeerConn):
        peer.send({
            'type': 'hello',
            'height': self.chain.height() if self.chain else 0,
            'node_id': self.node_id,
            'timestamp': int(time.time())
        })

    def announce_block(self, height: int, block_hash: str):
        for p in list(self.peers):
            if p.alive:
                p.send({'type':'announce','height':height,'hash':block_hash})

    def broadcast_tx(self, tx: Dict[str, Any]):
        txid = tx.get('txid')
        if not txid:
            return
        if txid in self._seen_txs:
            return
        self._seen_txs.add(txid)
        for p in list(self.peers):
            if p.alive:
                p.send({'type':'tx','tx': tx})

    def _handle(self, peer: PeerConn, line: bytes):
        try:
            msg = json.loads(line.decode())
        except Exception:
            return
        t = msg.get('type')
        if not isinstance(t, str):
            return
        # Basic rate limiting bookkeeping helpers
        now = time.time()
        def _prune(ls: List[float]):
            cutoff = now - 60
            while ls and ls[0] < cutoff:
                ls.pop(0)
        if t == 'tx':
            _prune(self._tx_timestamps)
            if len(self._tx_timestamps) >= self.MAX_TX_PER_MINUTE:
                return
            self._tx_timestamps.append(now)
        elif t in ('announce','block'):
            _prune(self._block_announce_timestamps)
            if len(self._block_announce_timestamps) >= self.MAX_BLOCK_ANNOUNCE_PER_MINUTE:
                return
            self._block_announce_timestamps.append(now)
        if t == 'hello':
            their_height = msg.get('height',0)
            our_height = self.chain.height() if self.chain else 0
            if their_height > our_height and self.chain:
                # request inventory: we request hashes from (our_height-20) to tip
                start = max(0, our_height-20)
                hashes = [b.hash for b in self.chain.blocks[start:our_height]] if our_height>0 else []
                peer.send({'type':'inv_request','from_height':our_height,'have_hashes':hashes})
        elif t == 'announce':
            announced_height = msg.get('height')
            h = msg.get('hash')
            # sanity: hash must be hex 64
            if not (isinstance(h, str) and len(h)==64):
                return
            if not self.chain.block_exists(h):
                # request this block explicitly
                peer.send({'type':'getblocks','hashes':[h]})
        elif t == 'inv':
            hashes = msg.get('hashes',[])
            if not isinstance(hashes, list):
                return
            # cap list length aggressively for safety
            if len(hashes) > 500:
                hashes = hashes[:500]
            missing = [h for h in hashes if not self.chain.block_exists(h)]
            if missing:
                peer.send({'type':'getblocks','hashes':missing[:self.MAX_BLOCK_REQUEST]})
        elif t == 'getblocks':
            req_hashes = msg.get('hashes',[])
            if not isinstance(req_hashes, list):
                return
            req_hashes = req_hashes[:self.MAX_BLOCK_REQUEST]
            for h in req_hashes:
                blk = self.chain.get_block_by_hash(h)
                if blk:
                    peer.send({'type':'block','block':{
                        'height': blk.height,
                        'prev_hash': blk.prev_hash,
                        'timestamp': blk.timestamp,
                        'merkle_root': blk.merkle_root,
                        'difficulty': blk.difficulty,
                        'nonce': blk.nonce,
                        'txs': blk.txs,
                        'hash': blk.hash,
                    }})
        elif t == 'block':
            b = msg.get('block')
            if not b:
                return
            # minimal structural sanity
            if not isinstance(b, dict):
                return
            for k in ('height','prev_hash','merkle_root','difficulty','nonce','txs'):
                if k not in b:
                    return
            if not (isinstance(b.get('prev_hash'), str) and len(b['prev_hash'])==64):
                return
            # basic sequential append only for now
            if b['height'] == self.chain.height() and b['prev_hash'] == self.chain.last_block().hash:
                # reconstruct mined dict shape for submit
                mined = {
                    'height': b['height'],
                    'prev_hash': b['prev_hash'],
                    'timestamp': b['timestamp'],
                    'merkle_root': b['merkle_root'],
                    'difficulty': b['difficulty'],
                    'nonce': b['nonce'],
                    'txs': b['txs']
                }
                self.chain.submit_block(mined)
            else:
                # ignore for now (no reorg logic yet)
                pass
        elif t == 'inv_request':
            # Peer is asking for our recent hashes after a certain point
            from_height = msg.get('from_height',0)
            have_hashes = set(msg.get('have_hashes',[]))
            start = max(0, from_height)
            hashes = [blk.hash for blk in self.chain.blocks[start:]]
            # Optionally filter out what they already claim
            send_hashes = [h for h in hashes if h not in have_hashes]
            peer.send({'type':'inv','hashes':send_hashes,'tip_height':self.chain.height()})
        elif t == 'tx':
            tx = msg.get('tx')
            if not tx or 'txid' not in tx:
                return
            txid = tx['txid']
            if not (isinstance(txid, str) and len(txid)==64):
                return
            if txid in self._seen_txs:
                return
            # attempt to add to mempool
            # Construct temporary Tx object analog (reuse add_tx path expects Tx dataclass; we'll adapt quickly)
            try:
                from core.blockchain import Tx as CoreTx
                tmp = CoreTx(txid, tx.get('inputs', []), tx.get('outputs', []), tx.get('fee',0), int(time.time()))
                tmp.txid = txid  # keep original
                if self.chain.add_tx(tmp):
                    self._seen_txs.add(txid)
                    # re-broadcast
                    for p2 in list(self.peers):
                        if p2 is not peer and p2.alive:
                            p2.send({'type':'tx','tx': tx})
            except Exception:
                return

if __name__ == '__main__':
    from core.blockchain import Blockchain
    chain = Blockchain()
    node = P2PNode(chain=chain)
    node.start()
    print("P2P skeleton started on 0.0.0.0:29876")
    while True:
        time.sleep(5)
