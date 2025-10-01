"""
ZION 2.7 TestNet - Minimal Real Blockchain Core
No P2P yet. No mock balances. Deterministic genesis.
"""
from __future__ import annotations
import json, time, hashlib, os, threading
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

# ---- Data Structures ----
@dataclass
class Block:
    height: int
    prev_hash: str
    timestamp: int
    merkle_root: str
    difficulty: int
    nonce: int
    txs: List[Dict[str, Any]] = field(default_factory=list)
    hash: Optional[str] = None

    def calc_hash(self) -> str:
        # Height intentionally excluded from hash so reorg height reassignment
        # does not alter block identity (prev_hash links remain valid).
        data = {
            'p': self.prev_hash,
            't': self.timestamp,
            'm': self.merkle_root,
            'd': self.difficulty,
            'n': self.nonce,
            'txs': self.txs,
        }
        blob = json.dumps(data, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()

    def seal(self):
        if not self.hash:
            self.hash = self.calc_hash()
        return self.hash

@dataclass
class Tx:
    txid: str
    inputs: List[Dict[str, Any]]
    outputs: List[Dict[str, Any]]
    fee: int
    timestamp: int

    @staticmethod
    def create(inputs, outputs, fee: int) -> 'Tx':
        tx = Tx("", inputs, outputs, fee, int(time.time()))
        body = json.dumps({
            'i': inputs,
            'o': outputs,
            'f': fee,
            'ts': tx.timestamp
        }, sort_keys=True).encode()
        tx.txid = hashlib.sha256(body).hexdigest()
        return tx

# ---- Consensus ----
class Consensus:
    BLOCK_TIME = 120
    INITIAL_REWARD = 333_000000  # atomic units
    MIN_DIFF = 1
    WINDOW = 12  # adjustment window
    MAX_ADJUST_FACTOR = 4.0  # clamp sudden jumps
    # Simplified max target (roughly 12 leading zero bits) for early testnet experimentation
    MAX_TARGET = int('000fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff', 16)
    COINBASE_MATURITY = 10  # blocks before coinbase spendable

    @staticmethod
    def difficulty_to_target(difficulty: int) -> int:
        if difficulty < 1:
            difficulty = 1
        return Consensus.MAX_TARGET // difficulty

    @staticmethod
    def reward(height: int) -> int:
        # later halving logic
        return Consensus.INITIAL_REWARD

# ---- Core Blockchain ----
class Blockchain:
    def __init__(self, data_dir: str = None, autosave_interval: int = 15):
        self.blocks: List[Block] = []
        self.mempool: Dict[str, Tx] = {}
        self.MEMPOOL_MAX = 1000
        self._perf_last_template_ms = 0.0
        self._perf_last_block_apply_ms = 0.0
        # UTXO set: key (txid, vout_index) -> { 'address': str, 'amount': int }
        self.utxos: Dict[tuple, Dict[str, Any]] = {}
        self.current_difficulty = Consensus.MIN_DIFF
        self.genesis_address = "Z3BDEEC2A0AE0F5D81B034308F99ECD8990D9B8B01BD9C7E7429392CA31861C6220DA3B30D74E809FA0A1FE069F1"
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), '..', 'data', 'blocks')
        self.data_dir = os.path.abspath(self.data_dir)
        os.makedirs(self.data_dir, exist_ok=True)
        self._lock = threading.RLock()
        # O(1) hash -> height index for fast lookups & upcoming P2P sync
        self._hash_index: Dict[str, int] = {}
        # Reorg support structures
        self._all_blocks: Dict[str, Block] = {}
        self._children: Dict[str, List[str]] = {}
        # Genesis creation
        self._create_genesis()
        # Autosave
        self._autosave_interval = autosave_interval
        t = threading.Thread(target=self._autosave_loop, daemon=True)
        t.start()

    # ---- Reorg / Branch Handling ----
    def _create_genesis(self):
        genesis = Block(
            height=0,
            prev_hash='0'*64,
            timestamp=int(time.time()),
            merkle_root='genesis',
            difficulty=Consensus.MIN_DIFF,
            nonce=0,
            txs=[{
                'txid': 'genesis',
                'type': 'coinbase',
                'outputs': [{'address': self.genesis_address, 'amount': Consensus.reward(0)}]
            }]
        )
        genesis.seal()
        self.blocks.append(genesis)
        self._hash_index[genesis.hash] = 0
        self._all_blocks[genesis.hash] = genesis
        self._persist_block(genesis)
        self._apply_block_utxos(genesis)

    # ---------------- Internal Helpers (Persistence / UTXO) ----------------
    def _persist_block(self, blk: Block):
        path = os.path.join(self.data_dir, f"{blk.height:08d}_{blk.hash}.json")
        if not os.path.exists(path):
            with open(path, 'w') as f:
                json.dump({
                    'height': blk.height,
                    'prev_hash': blk.prev_hash,
                    'timestamp': blk.timestamp,
                    'merkle_root': blk.merkle_root,
                    'difficulty': blk.difficulty,
                    'nonce': blk.nonce,
                    'txs': blk.txs,
                    'hash': blk.hash
                }, f)

    def _apply_block_utxos(self, blk: Block):
        for tx in blk.txs:
            if tx.get('type') == 'coinbase':
                for idx, out in enumerate(tx['outputs']):
                    self.utxos[(tx['txid'], idx)] = {
                        'address': out['address'],
                        'amount': out['amount'],
                        'maturity': blk.height + Consensus.COINBASE_MATURITY
                    }
            else:
                self._apply_tx(tx)

    # ---------------- Reorg Support ----------------
    def _build_branch_to_genesis(self, tip: Block) -> Optional[List[Block]]:
        branch = []
        cursor = tip
        while True:
            branch.append(cursor)
            if cursor.prev_hash == '0'*64:
                break
            parent = self._all_blocks.get(cursor.prev_hash)
            if not parent:
                return None
            cursor = parent
        return list(reversed(branch))

    def _adopt_branch(self, branch: List[Block]):
        self.blocks = []
        self._hash_index.clear()
        self.utxos.clear()
        for idx, b in enumerate(branch):
            b.height = idx
            self.blocks.append(b)
            self._hash_index[b.hash] = idx
            self._persist_block(b)
            self._apply_block_utxos(b)
        self._recalculate_difficulty()

    def _register_block_and_maybe_reorg(self, blk: Block, extending_tip: bool):
        # Register globally
        self._all_blocks[blk.hash] = blk
        self._children.setdefault(blk.prev_hash, []).append(blk.hash)
        if extending_tip:
            blk.height = self.height()
            self.blocks.append(blk)
            self._hash_index[blk.hash] = blk.height
            self._persist_block(blk)
            self._apply_block_utxos(blk)
            self._recalculate_difficulty()
            return
        # Evaluate side branch
        branch = self._build_branch_to_genesis(blk)
        if branch is None:
            return
        # branch length counts genesis + side blocks; canonical length is self.height()
        if len(branch) <= self.height():
            return  # not strictly longer
        self._adopt_branch(branch)

    def _autosave_loop(self):
        last_height = self.height()
        while True:
            time.sleep(self._autosave_interval)
            with self._lock:
                if self.height() != last_height:
                    # persist only new blocks
                    for blk in self.blocks[last_height:]:
                        self._persist_block(blk)
                    last_height = self.height()

    # ---- Accessors ----
    def height(self) -> int:
        return len(self.blocks)

    def last_block(self) -> Block:
        return self.blocks[-1]

    # ---- Transactions ----
    def add_tx(self, tx: Tx) -> bool:
        if tx.txid in self.mempool:
            return False
        if not self._validate_tx({
            'txid': tx.txid,
            'inputs': tx.inputs,
            'outputs': tx.outputs,
            'fee': tx.fee
        }):
            return False
        self.mempool[tx.txid] = tx
        if len(self.mempool) > self.MEMPOOL_MAX:
            # evict lowest fee
            lowest = min(self.mempool.values(), key=lambda t: t.fee)
            if lowest.txid != tx.txid:
                self.mempool.pop(lowest.txid, None)
        return True

    # ---- Mining Template ----
    def create_block_template(self, miner_address: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        height = self.height()
        reward = Consensus.reward(height)
        coinbase = {
            'txid': f'cb_{height}_{int(time.time())}',
            'type': 'coinbase',
            'outputs': [{'address': miner_address, 'amount': reward}],
        }
        ordered = sorted(self.mempool.values(), key=lambda t: t.fee, reverse=True)
        txs = [coinbase] + [
            {
                'txid': tx.txid,
                'inputs': tx.inputs,
                'outputs': tx.outputs,
                'fee': tx.fee
            } for tx in ordered
        ]
        merkle_root = hashlib.sha256(json.dumps([t['txid'] for t in txs]).encode()).hexdigest()
        target_int = Consensus.difficulty_to_target(self.current_difficulty)
        self._perf_last_template_ms = (time.perf_counter() - t0) * 1000.0
        return {
            'height': height,
            'prev_hash': self.last_block().hash,
            'difficulty': self.current_difficulty,
            'merkle_root': merkle_root,
            'txs': txs,
            'target': target_int,
            'target_hex': f"{target_int:064x}"
        }

    # ---- Submit Mined Block ----
    def submit_block(self, mined: Dict[str, Any]) -> bool:
        """Submit a mined block supporting side branches and reorg adoption.

        Validation rules:
        - If extending tip: height must match current height.
        - If side branch: parent must exist in any known block map.
        - Height in side branch blocks is treated as placeholder; recomputed on adoption.
        """
        t0 = time.perf_counter()
        expected_height = self.height()
        parent_hash = mined['prev_hash']
        extending_tip = (parent_hash == self.last_block().hash and mined['height'] == expected_height)
        if extending_tip and mined['height'] != expected_height:
            return False
        if not extending_tip and parent_hash not in self._hash_index and parent_hash not in self._all_blocks:
            # Parent totally unknown: reject
            return False
        blk = Block(
            height=mined['height'],
            prev_hash=parent_hash,
            timestamp=mined.get('timestamp', int(time.time())),
            merkle_root=mined['merkle_root'],
            difficulty=mined['difficulty'],
            nonce=mined['nonce'],
            txs=mined['txs']
        )
        blk.seal()
        accepted = True
        with self._lock:
            self._register_block_and_maybe_reorg(blk, extending_tip)
            if blk.hash not in self._all_blocks:
                accepted = False
            if blk.hash in self._hash_index and self._hash_index[blk.hash] == self.height()-1:
                for t in blk.txs:
                    if t.get('type') != 'coinbase':
                        self.mempool.pop(t['txid'], None)
        self._perf_last_block_apply_ms = (time.perf_counter() - t0) * 1000.0
        return accepted

    def _apply_tx(self, tx: Dict[str, Any]):
        if not self._validate_tx(tx):
            return False
        for i in tx.get('inputs', []):
            ref = (i['txid'], i['vout'])
            self.utxos.pop(ref, None)
        for idx, o in enumerate(tx.get('outputs', [])):
            self.utxos[(tx['txid'], idx)] = {'address': o['address'], 'amount': o['amount']}
        return True

    # ---------------- Validation ----------------
    def _validate_tx(self, tx: Dict[str, Any]) -> bool:
        # Basic structural and UTXO availability check
        seen = set()
        total_in = 0
        for i in tx.get('inputs', []):
            ref = (i['txid'], i['vout'])
            if ref in seen:
                return False
            seen.add(ref)
            utxo = self.utxos.get(ref)
            if not utxo:
                return False
            # Coinbase maturity check if maturity metadata present
            maturity = utxo.get('maturity')
            if maturity is not None and self.height() <= maturity:
                return False
            total_in += utxo['amount']
        total_out = sum(o['amount'] for o in tx.get('outputs', []))
        if total_in < total_out:
            return False
        return True

    # ---- Block Lookup ----
    def get_block_by_height(self, height: int) -> Optional[Block]:
        if 0 <= height < len(self.blocks):
            return self.blocks[height]
        return None

    def get_block_by_hash(self, h: str) -> Optional[Block]:
        idx = self._hash_index.get(h)
        if idx is None:
            return None
        return self.blocks[idx]

    def block_exists(self, h: str) -> bool:
        return h in self._hash_index

    # ---- Difficulty ----
    def _recalculate_difficulty(self):
        if self.height() < 2:
            return
        window = self.blocks[-Consensus.WINDOW:]
        if len(window) < 2:
            return
        span = window[-1].timestamp - window[0].timestamp
        blocks_solved = len(window) - 1
        if blocks_solved <= 0:
            return
        actual_per_block = span / blocks_solved if span > 0 else Consensus.BLOCK_TIME
        ratio = actual_per_block / Consensus.BLOCK_TIME
        # clamp ratio
        ratio = max(1.0/Consensus.MAX_ADJUST_FACTOR, min(Consensus.MAX_ADJUST_FACTOR, ratio))
        if ratio > 1.0:
            # slower than target -> decrease difficulty
            self.current_difficulty = max(Consensus.MIN_DIFF, int(self.current_difficulty / ratio))
        else:
            # faster -> increase
            self.current_difficulty = max(Consensus.MIN_DIFF, int(self.current_difficulty * (1/ratio)))


    def info(self) -> Dict[str, Any]:
        lb = self.last_block()
        current_target = Consensus.difficulty_to_target(self.current_difficulty)
        return {
            'height': self.height(),
            'last_hash': lb.hash,
            'difficulty': self.current_difficulty,
            'target_hex': f"{current_target:064x}",
            'tx_pool': len(self.mempool),
            'utxos': len(self.utxos),
            'perf_last_template_ms': round(self._perf_last_template_ms, 3),
            'perf_last_block_apply_ms': round(self._perf_last_block_apply_ms, 3),
            'version': '2.7-testnet',
            'p2p': False,  # placeholder until node integration sets
        }

if __name__ == '__main__':
    chain = Blockchain()
    print(json.dumps(chain.info(), indent=2))
