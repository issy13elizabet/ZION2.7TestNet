#!/usr/bin/env python3
"""
ZION 2.7.4 - Nový Blockchain s Novými Premine Adresami
Implementace blockchainu s čerstvě vygenerovanými adresami
"""

import hashlib
import json
import time
import secrets
import asyncio
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import sqlite3
import math

# Add P2P network import
from zion_p2p_network import ZIONP2PNetwork
from zion_rpc_server import ZIONRPCServer
from crypto_utils import tx_hash

class NewZionBlockchain:
    """Nový ZION blockchain s novými premine adresami a persistent storage"""
    
    def __init__(self, db_file="zion_blockchain.db", enable_p2p=True, p2p_port=8333, enable_rpc=True, rpc_port=8332):
        self.db_file = db_file
        self.lock = threading.Lock()
        self._init_database()
        self.blocks = self._load_blocks_from_db()
        # Persistent mempool & nonces will be loaded from DB (fallback to empty)
        self.pending_transactions: List[Dict] = []
        self.nonces: Dict[str, int] = {}
        # Difficulty and reward parameters (could be externalized later)
        self.mining_difficulty = 4
        self.block_reward = 50  # Základní hodnota – TODO: implementovat dynamické snižování
        self.premine_addresses = self._get_new_addresses()
        self.balances = self._load_balances_from_db()
        
        if not self.blocks:
            self._create_genesis_block()
        # Load persistent nonce & mempool state
        self._load_nonces_from_db()
        self.pending_transactions = self._load_mempool_from_db()
        
        # P2P Network
        self.p2p_network = None
        if enable_p2p:
            self.p2p_network = ZIONP2PNetwork(self, port=p2p_port)
        
        # RPC Server
        self.rpc_server = None
        if enable_rpc:
            self.rpc_server = ZIONRPCServer(self, port=rpc_port)
        # Security / consensus timing parameters
        self.mtp_window = 11  # number of blocks for median-time-past
        self.max_future_drift = 7200  # seconds (2h)
        self.invalid_timestamps = 0  # counter for metrics
        self.reorg_events = 0  # count multi-block reorgs (for future metrics)
        self.max_reorg_depth = 50
        # In-memory cumulative work tracking (not persisted) – compute after genesis ensured
        self._recompute_cumulative_work(0)
        
    def _init_database(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            
            # Blocks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS blocks (
                    height INTEGER PRIMARY KEY,
                    hash TEXT UNIQUE NOT NULL,
                    previous_hash TEXT,
                    timestamp REAL NOT NULL,
                    transactions TEXT NOT NULL,
                    nonce INTEGER NOT NULL,
                    difficulty INTEGER NOT NULL,
                    miner TEXT,
                    reward REAL DEFAULT 50.0
                )
            ''')
            
            # Balances table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS balances (
                    address TEXT PRIMARY KEY,
                    balance REAL NOT NULL DEFAULT 0.0,
                    last_updated REAL DEFAULT (strftime('%s', 'now'))
                )
            ''')
            
            # Transactions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    id TEXT PRIMARY KEY,
                    sender TEXT,
                    receiver TEXT,
                    amount REAL NOT NULL,
                    fee REAL DEFAULT 0.0,
                    timestamp REAL DEFAULT (strftime('%s', 'now')),
                    block_height INTEGER,
                    signature TEXT
                )
            ''')

            # Nonces table (persistent replay protection)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS nonces (
                    address TEXT PRIMARY KEY,
                    nonce INTEGER NOT NULL DEFAULT 0,
                    updated REAL DEFAULT (strftime('%s','now'))
                )
            ''')

            # Mempool table (survives restart; orphaned tx cleaned on inclusion)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS mempool (
                    id TEXT PRIMARY KEY,
                    tx_json TEXT NOT NULL,
                    received REAL DEFAULT (strftime('%s','now'))
                )
            ''')

            # Block state journal (for rollback / reorg)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS block_journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    block_height INTEGER NOT NULL,
                    address TEXT NOT NULL,
                    delta REAL NOT NULL,
                    balance_after REAL NOT NULL,
                    rolled_back INTEGER DEFAULT 0,
                    timestamp REAL DEFAULT (strftime('%s','now'))
                )
            ''')
            
            conn.commit()

    # ---------------- Persistence Helpers: Nonces & Mempool ---------------- #
    def _load_nonces_from_db(self):
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT address, nonce FROM nonces')
            rows = cursor.fetchall()
            self.nonces = {addr: n for addr, n in rows}

    def _persist_nonce(self, address: str, nonce: int):
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO nonces (address, nonce, updated) VALUES (?, ?, strftime('%s','now'))
                ON CONFLICT(address) DO UPDATE SET nonce=excluded.nonce, updated=excluded.updated
            ''', (address, nonce))
            conn.commit()

    def _load_mempool_from_db(self) -> List[Dict]:
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT tx_json FROM mempool ORDER BY received')
            txs = []
            for (tx_json,) in cursor.fetchall():
                try:
                    tx = json.loads(tx_json)
                    txs.append(tx)
                except Exception:
                    continue
            return txs

    def _add_mempool_tx(self, tx: Dict):
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT OR REPLACE INTO mempool (id, tx_json) VALUES (?, ?)', (tx['id'], json.dumps(tx, sort_keys=True)))
            conn.commit()

    def _remove_mempool_tx(self, tx_id: str):
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM mempool WHERE id=?', (tx_id,))
            conn.commit()
    
    def _load_blocks_from_db(self) -> List[Dict]:
        """Load all blocks from database"""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM blocks ORDER BY height')
            rows = cursor.fetchall()
            
            blocks = []
            for row in rows:
                height, hash_val, prev_hash, timestamp, tx_json, nonce, difficulty, miner, reward = row
                transactions = json.loads(tx_json)
                block = {
                    'height': height,
                    'hash': hash_val,
                    'previous_hash': prev_hash,
                    'timestamp': timestamp,
                    'transactions': transactions,
                    'nonce': nonce,
                    'difficulty': difficulty,
                    'miner': miner,
                    'reward': reward,
                    'cumulative_work': None  # placeholder until recomputed
                }
                blocks.append(block)
            return blocks
    
    def _load_balances_from_db(self) -> Dict[str, float]:
        """Load all balances from database"""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT address, balance FROM balances')
            rows = cursor.fetchall()
            return {row[0]: row[1] for row in rows}
    
    def _save_block_to_db(self, block: Dict):
        """Save block to database"""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO blocks (height, hash, previous_hash, timestamp, transactions, nonce, difficulty, miner, reward)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                block['height'],
                block['hash'],
                block['previous_hash'],
                block['timestamp'],
                json.dumps(block['transactions']),
                block['nonce'],
                block['difficulty'],
                block.get('miner'),
                block.get('reward', 50.0)
            ))
            conn.commit()
    
    def _save_balance_to_db(self, address: str, balance: float):
        """Save balance to database"""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO balances (address, balance, last_updated)
                VALUES (?, ?, strftime('%s', 'now'))
            ''', (address, balance))
            conn.commit()
    
    def _save_transaction_to_db(self, tx: Dict, block_height: int):
        """Save transaction to database"""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO transactions (id, sender, receiver, amount, fee, block_height, signature)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                tx['id'],
                tx.get('sender'),
                tx.get('receiver'),
                tx['amount'],
                tx.get('fee', 0.0),
                block_height,
                tx.get('signature')
            ))
            conn.commit()
    
    def _get_new_addresses(self) -> Dict:
        """Nově vygenerované adresy pro premine"""
        return {
            'ZION_SACRED_B0FA7E2A234D8C2F08545F02295C98': {
                'purpose': 'Sacred Mining Operator',
                'amount': 2_000_000_000,
                'type': 'mining'
            },
            'ZION_QUANTUM_89D80B129682D41AD76DAE3F90C3E2': {
                'purpose': 'Quantum Mining Operator', 
                'amount': 2_000_000_000,
                'type': 'mining'
            },
            'ZION_COSMIC_397B032D6E2D3156F6F709E8179D36': {
                'purpose': 'Cosmic Mining Operator',
                'amount': 2_000_000_000,
                'type': 'mining'
            },
            'ZION_ENLIGHTENED_004A5DBD12FDCAACEDCB5384DDC035': {
                'purpose': 'Enlightened Mining Operator',
                'amount': 2_000_000_000,
                'type': 'mining'
            },
            'ZION_TRANSCENDENT_6BD30CB1835013503A8167D9CD86E0': {
                'purpose': 'Transcendent Mining Operator',
                'amount': 2_000_000_000,
                'type': 'mining'
            },
            'ZION_DEVELOPMENT_TEAM_FUND_378614887FEA27791540F45': {
                'purpose': 'Development Team Fund',
                'amount': 1_440_000_000,
                'type': 'development'
            },
            'ZION_NETWORK_INFRASTRUCTURE_SITA_B5F3BE9968A1D90': {
                'purpose': 'Network Infrastructure (SITA)',
                'amount': 999_000_000,
                'type': 'infrastructure'
            },
            'ZION_CHILDREN_FUTURE_FUND_1ECCB72BC30AADD086656A59': {
                'purpose': 'Children Future Fund',
                'amount': 999_000_000,
                'type': 'charity'
            },
            'ZION_MAITREYA_BUDDHA_D7A371ABD1FF1C5D42AB02AAE4F57': {
                'purpose': 'Network Administrator',
                'amount': 999_000_000,
                'type': 'admin'
            },
            'ZION_ON_THE_STAR_0B461AB5BCACC40D1ECE95A2D82030': {
                'purpose': 'Genesis Reward',
                'amount': 333_000_000,
                'type': 'genesis'
            }
        }
    
    def _create_genesis_block(self):
        """Create and save genesis block with premine distribution"""
        genesis_transactions = []
        
        # Distribute premine
        for address, info in self.premine_addresses.items():
            tx = {
                'id': f"genesis_{address}",
                'sender': 'GENESIS',
                'receiver': address,
                'amount': info['amount'],
                'fee': 0.0,
                'timestamp': time.time(),
                'purpose': f"Genesis premine distribution - {info['purpose']}",
                'signature': 'GENESIS_SIGNATURE'
            }
            genesis_transactions.append(tx)
            self.balances[address] = info['amount']
            self._save_balance_to_db(address, info['amount'])
            self._save_transaction_to_db(tx, 0)
        
        genesis_block = {
            'height': 0,
            'hash': self._calculate_block_hash(0, '0' * 64, time.time(), genesis_transactions, 0, 4),
            'previous_hash': '0' * 64,
            'timestamp': time.time(),
            'transactions': genesis_transactions,
            'nonce': 0,
            'difficulty': 4,
            'miner': 'GENESIS',
            'reward': 0.0
        }
        
        self.blocks.append(genesis_block)
        self._save_block_to_db(genesis_block)
    
    def _calculate_block_hash(self, height: int, previous_hash: str, timestamp: float, 
                            transactions: List[Dict], nonce: int, difficulty: int) -> str:
        """Calculate block hash for genesis block"""
        block_data = {
            'height': height,
            'previous_hash': previous_hash,
            'timestamp': timestamp,
            'transactions': transactions,
            'nonce': nonce,
            'difficulty': difficulty
        }
        block_string = json.dumps(block_data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def _calculate_hash(self, block: Dict) -> str:
        """Bez-mutační výpočet hash bloku (nonce už musí být nastaven)."""
        # Exclude non-consensus / post-mining metadata fields from hash
        block_clone = {k: block[k] for k in block if k not in ('hash', 'cumulative_work')}
        block_string = json.dumps(block_clone, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(block_string.encode()).hexdigest()

    def _mine_block(self, block: Dict) -> str:
        """Těží blok pomocí Proof of Work – mění pouze nonce dokud hash nesplní target."""
        target_prefix = "0" * self.mining_difficulty
        # Zajistíme inicializaci nonce pokud chybí
        if 'nonce' not in block or not isinstance(block['nonce'], int):
            block['nonce'] = 0
        while True:
            block['nonce'] += 1
            candidate_hash = self._calculate_hash(block)
            if candidate_hash.startswith(target_prefix):
                return candidate_hash
                
    def create_transaction(self, from_address: str, to_address: str, amount: float, purpose: str = ""):
        """Vytvoří novou transakci"""
        if from_address not in self.balances:
            self.balances[from_address] = 0
        if to_address not in self.balances:
            self.balances[to_address] = 0
            
        if self.balances[from_address] < amount:
            raise ValueError(f"Insufficient balance: {self.balances[from_address]} < {amount}")
        
        nonce = self.nonces.get(from_address, 0)
        transaction = {
            'id': f"tx_{len(self.pending_transactions)}_{int(time.time())}_{secrets.token_hex(4)}",
            'sender': from_address,
            'receiver': to_address,
            'amount': amount,
            'fee': 0.0,
            'timestamp': time.time(),
            'purpose': purpose,
            'nonce': nonce,
            'signature': 'UNSIGNED'
        }
        self.nonces[from_address] = nonce + 1
        self._persist_nonce(from_address, self.nonces[from_address])
        self.pending_transactions.append(transaction)
        self._add_mempool_tx(transaction)
        return transaction

    def add_signed_transaction(self, tx: Dict):
        """Přidá podepsanou transakci do mempoolu (základní ověření)."""
        required = {'sender','receiver','amount','timestamp','nonce','signature','tx_hash'}
        if not required.issubset(set(tx.keys())):
            raise ValueError("Incomplete signed transaction")
        # Basic replay / nonce validation with allowance for replacing unsigned mempool tx
        current_nonce = self.nonces.get(tx['sender'], 0)
        if tx['nonce'] > current_nonce:
            raise ValueError("Invalid nonce (future nonce)")
        if tx['nonce'] < current_nonce:
            # Already processed / replaced
            raise ValueError("Stale nonce")
        if self.balances.get(tx['sender'],0) < tx['amount']:
            raise ValueError("Insufficient balance")
        # Hash consistency
        if tx_hash(tx) != tx['tx_hash']:
            raise ValueError("Hash mismatch")
        self.nonces[tx['sender']] = current_nonce + 1
        self._persist_nonce(tx['sender'], self.nonces[tx['sender']])
        self.pending_transactions.append(tx)
        # Ensure ID exists (if external raw tx might not have internal id)
        if 'id' not in tx:
            tx['id'] = f"signed_{tx['sender']}_{tx['nonce']}_{tx['tx_hash'][:12]}"
        self._add_mempool_tx(tx)
        return tx['tx_hash']

    def _merkle_root(self, transactions: List[Dict]) -> str:
        if not transactions:
            return '0'*64
        layer = [tx_hash(tx) for tx in transactions]
        while len(layer) > 1:
            new_layer = []
            for i in range(0, len(layer), 2):
                a = layer[i]
                b = layer[i+1] if i+1 < len(layer) else a
                new_layer.append(hashlib.sha256((a+b).encode()).hexdigest())
            layer = new_layer
        return layer[0]
    
    def mine_pending_transactions(self, mining_reward_address: str) -> str:
        """Vytěží nový blok s čekajícími transakcemi s lepší validací a ochranou proti záporným zůstatkům."""
        with self.lock:
            block = {
                'height': len(self.blocks),
                'timestamp': time.time(),
                'transactions': self.pending_transactions.copy(),
                'previous_hash': self.blocks[-1]['hash'] if self.blocks else '0',
                'nonce': 0,
                'difficulty': self.mining_difficulty,
                'miner': mining_reward_address,
                'reward': self.block_reward,
                'merkle_root': None,
                'hash': None
            }

            # Enforce Median-Time-Past & future drift
            block['timestamp'] = self._enforce_block_timestamp(block['timestamp'])

            # Validace transakcí (základní – TODO: podpisy / double-spend / UTXO)
            for tx in block['transactions']:
                if tx.get('amount') is None or tx['amount'] <= 0:
                    raise ValueError(f"Neplatná částka v transakci {tx['id']}")
                sender = tx.get('sender')
                if sender not in ('GENESIS', 'MINING_REWARD'):
                    if self.balances.get(sender, 0) < tx['amount']:
                        raise ValueError(f"Transakce {tx['id']} má nedostatek prostředků: {sender}")

            # Přidání mining reward transakce
            if mining_reward_address:
                reward_transaction = {
                    'id': f"mining_reward_{len(self.blocks)}_{int(time.time())}",
                    'sender': 'MINING_REWARD',
                    'receiver': mining_reward_address,
                    'amount': self.block_reward,
                    'fee': 0.0,
                    'timestamp': time.time(),
                    'purpose': 'Block Mining Reward',
                    'signature': 'MINING_REWARD_SIGNATURE'
                }
                block['transactions'].append(reward_transaction)

            print(f"⛏️  Mining block {block['height']}...")
            block['merkle_root'] = self._merkle_root(block['transactions'])
            block['hash'] = self._mine_block(block)

            # Aplikace efektů bloku s journalingem
            self._apply_block_effects(block)

            self.blocks.append(block)
            # Cumulative work update
            block['cumulative_work'] = (self.blocks[-2]['cumulative_work'] if len(self.blocks) > 1 else 0) + self._block_work(block)
            # Remove mined transactions from mempool (excluding reward tx which was synthetic)
            mined_ids = {tx['id'] for tx in block['transactions'] if tx.get('id')}
            self.pending_transactions = [tx for tx in self.pending_transactions if tx.get('id') not in mined_ids]
            for tx_id in mined_ids:
                self._remove_mempool_tx(tx_id)

            print(f"✅ Block {block['height']} mined: {block['hash'][:16]}...")

            # LWMA recalculation (adaptivní obtížnost)
            self._apply_lwma_difficulty()

            # Uložení bloku a transakcí do DB (bezpečně – rollback by vyžadoval transakci)
            self._save_block_to_db(block)
            for tx in block['transactions']:
                self._save_transaction_to_db(tx, block['height'])
                if tx['sender'] not in ('GENESIS', 'MINING_REWARD'):
                    self._save_balance_to_db(tx['sender'], self.balances.get(tx['sender'], 0))
                self._save_balance_to_db(tx['receiver'], self.balances.get(tx['receiver'], 0))

            # Broadcast new block to P2P network
            if self.p2p_network:
                block_data = {
                    'height': block['height'],
                    'hash': block['hash'],
                    'previous_hash': block['previous_hash'],
                    'timestamp': block['timestamp'],
                    'transactions': block['transactions'],
                    'nonce': block['nonce'],
                    'difficulty': block['difficulty'],
                    'miner': block.get('miner'),
                    'reward': block.get('reward', 50.0),
                    'merkle_root': block.get('merkle_root')
                }
                asyncio.create_task(self.p2p_network.broadcast_new_block(block_data))
            return block['hash']

    def _apply_lwma_difficulty(self, target_time: int = 120, window: int = 30, clamp_factor: float = 2.5):
        """Linear Weighted Moving Average (LWMA) difficulty adjustment.
        Simplified integer difficulty: we treat difficulty as required leading zeros length.
        Approach: compute average solvetime with linear weights, adjust by ratio.
        We clamp adjustments to avoid oscillations.
        """
        if len(self.blocks) < window + 5:
            return
        # Take last 'window' blocks excluding the most recent (current just mined block is included) -> we use last window intervals
        recent = self.blocks[-window:]
        solvetimes = []
        for i in range(1, len(recent)):
            dt = recent[i]['timestamp'] - recent[i-1]['timestamp']
            # clamp individual interval (anti timestamp attack)
            dt = max(1, min(dt, target_time * clamp_factor))
            solvetimes.append(dt)
        if not solvetimes:
            return
        # Weighted sum: weight i for ith interval
        weighted_sum = 0
        weight_total = 0
        for i, st in enumerate(solvetimes, start=1):
            weighted_sum += st * i
            weight_total += i
        lwma = weighted_sum / max(1, weight_total)
        ratio = lwma / target_time
        # Convert ratio into discrete difficulty step adjustments
        # If blocks are faster (ratio < 1), increase difficulty; slower, decrease.
        if ratio < 0.75 and self.mining_difficulty < 32:
            self.mining_difficulty += 1
        elif ratio > 1.60 and self.mining_difficulty > 1:
            self.mining_difficulty -= 1
        # else keep same

    def audit_integrity(self) -> Dict[str, bool]:
        """Provádí základní integritní audit: validita řetězce & konzistence nabídky."""
        chain_ok = self.validate_chain()
        total_supply = self.get_total_supply()
        # Genesis supply + mined rewards (odhad – reward * (blocks-1))
        genesis_supply = sum(info['amount'] for info in self.premine_addresses.values())
        mined = 0
        for b in self.blocks:
            for tx in b['transactions']:
                if tx.get('sender') == 'MINING_REWARD':
                    mined += tx.get('amount', 0)
        expected_min = genesis_supply  # minimální (kdyby žádné bloky nenesly reward)
        supply_ok = total_supply >= expected_min and total_supply <= (genesis_supply + mined + 1)
        return {
            'chain_valid': chain_ok,
            'supply_consistent': supply_ok,
            'total_supply': total_supply,
            'genesis_supply': genesis_supply,
            'mined_reward_total': mined,
            'difficulty': self.mining_difficulty
        }
    
    def get_balance(self, address: str) -> float:
        """Vrátí zůstatek adresy"""
        return self.balances.get(address, 0)
    
    def start_p2p_network(self):
        """Spustí P2P síť"""
        if self.p2p_network:
            asyncio.create_task(self.p2p_network.start())
            print("🚀 ZION P2P síť spuštěna")
    
    def stop_p2p_network(self):
        """Zastaví P2P síť"""
        if self.p2p_network:
            asyncio.create_task(self.p2p_network.stop())
            print("🛑 ZION P2P síť zastavena")
    
    def start_rpc_server(self):
        """Spustí RPC server"""
        if self.rpc_server:
            self.rpc_server.start()
    
    def stop_rpc_server(self):
        """Zastaví RPC server"""
        if self.rpc_server:
            self.rpc_server.stop()
    
    def get_network_status(self) -> Dict:
        """Vrátí status P2P sítě"""
        if self.p2p_network:
            return self.p2p_network.get_network_status()
        return {"status": "P2P disabled"}
    
    def get_total_supply(self) -> float:
        """Vrátí celkovou nabídku ZION"""
        return sum(self.balances.values())
    
    def validate_chain(self) -> bool:
        """Validuje celý blockchain bez re-miningu – přepočítá hash a ověří vazby."""
        for i in range(1, len(self.blocks)):
            current_block = self.blocks[i]
            previous_block = self.blocks[i - 1]
            # 1. Ověření previous hash
            if current_block['previous_hash'] != previous_block['hash']:
                return False
            # 2. Ověření hash (bez změny nonce)
            recalculated = self._calculate_hash(current_block)
            if recalculated != current_block['hash']:
                return False
            # 3. Ověření obtížnosti
            if not current_block['hash'].startswith('0' * current_block.get('difficulty', self.mining_difficulty)):
                return False
            # 4. Ověření merkle root
            if 'merkle_root' in current_block:
                if self._merkle_root(current_block['transactions']) != current_block['merkle_root']:
                    return False
            # 5. Timestamp sanity (median-time-past & future drift)
            if not self._check_block_timestamp(i):
                return False
        return True

    # ---------------- Timestamp / MTP ---------------- #
    def _median_time_past(self, upto_index: Optional[int] = None) -> float:
        """Return median timestamp of last self.mtp_window blocks ending at upto_index-1 (or tip)."""
        if not self.blocks:
            return time.time()
        if upto_index is None:
            upto_index = len(self.blocks)
        start = max(0, upto_index - self.mtp_window)
        subset = self.blocks[start:upto_index]
        times = sorted(b['timestamp'] for b in subset)
        mid = len(times)//2
        if len(times) % 2 == 1:
            return times[mid]
        return 0.5*(times[mid-1]+times[mid])

    def _enforce_block_timestamp(self, ts: float) -> float:
        """Adjust a locally mined block timestamp to satisfy MTP & future drift."""
        if len(self.blocks) == 0:
            return ts
        mtp = self._median_time_past()
        if ts < mtp:
            ts = mtp
        # Clamp future drift
        now = time.time()
        if ts > now + self.max_future_drift:
            ts = now + self.max_future_drift
        return ts

    def _check_block_timestamp(self, height: int) -> bool:
        """Check timestamp rules for block at height (already in chain)."""
        if height <= 0 or height >= len(self.blocks):
            return True
        block = self.blocks[height]
        mtp = self._median_time_past(upto_index=height)
        if block['timestamp'] < mtp:
            return False
        if block['timestamp'] > time.time() + self.max_future_drift:
            return False
        return True

    def validate_block_timestamp_external(self, block_data: Dict) -> bool:
        """Validate timestamp for an externally received (not yet appended) block."""
        # Determine median using current chain tip (without the new block)
        mtp = self._median_time_past()
        ts = block_data.get('timestamp', 0)
        if ts < mtp:
            self.invalid_timestamps += 1
            return False
        if ts > time.time() + self.max_future_drift:
            self.invalid_timestamps += 1
            return False
        return True

    # ---------------- Journaling & Rollback ---------------- #
    def _journal_change(self, address: str, delta: float, balance_after: float, block_height: int):
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO block_journal (block_height, address, delta, balance_after) VALUES (?, ?, ?, ?)
            ''', (block_height, address, delta, balance_after))
            conn.commit()

    def _apply_block_effects(self, block: Dict):
        """Apply balance changes for a block with journaling (idempotent if not previously applied)."""
        for tx in block['transactions']:
            sender = tx.get('sender')
            receiver = tx.get('receiver')
            amount = tx.get('amount', 0)
            # Sender side
            if sender not in ('GENESIS', 'MINING_REWARD') and sender is not None:
                prev = self.balances.get(sender, 0)
                new_balance = prev - amount
                self.balances[sender] = new_balance
                self._journal_change(sender, -amount, new_balance, block['height'])
            # Receiver side
            if receiver is not None:
                prev_r = self.balances.get(receiver, 0)
                new_r = prev_r + amount
                self.balances[receiver] = new_r
                self._journal_change(receiver, amount, new_r, block['height'])

    def rollback_block(self, block_height: int) -> bool:
        """Rollback a single block effects (not genesis)."""
        if block_height <= 0 or block_height >= len(self.blocks):
            return False
        block = self.blocks[block_height]
        # Fetch journal entries
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, address, delta, balance_after FROM block_journal WHERE block_height=? AND rolled_back=0 ORDER BY id DESC', (block_height,))
            rows = cursor.fetchall()
            for row in rows:
                _jid, address, delta, balance_after = row
                # Revert: original_before = balance_after - delta
                original_before = balance_after - delta
                self.balances[address] = original_before
            # Mark rolled_back
            cursor.execute('UPDATE block_journal SET rolled_back=1 WHERE block_height=? AND rolled_back=0', (block_height,))
            # Remove block + its transactions from DB
            cursor.execute('DELETE FROM blocks WHERE height=?', (block_height,))
            cursor.execute('DELETE FROM transactions WHERE block_height=?', (block_height,))
            conn.commit()
        # Remove from in-memory chain (truncate or keep earlier part and later part?)
        # For single-block replacement at tip or same height fork: if it's last block simply pop
        if block_height == len(self.blocks) - 1:
            self.blocks.pop()
        else:
            # Replace block placeholder with None; caller expected to insert new
            self.blocks[block_height] = None
        return True

    def replace_block_at_height(self, block_height: int, new_block: Dict):
        """Replace a block at given height with journaling rollback + reapply.
        Re-add reverted txs (non-reward) to mempool if not included in new block."""
        with self.lock:
            if block_height <= 0 or block_height >= len(self.blocks):
                return False
            old_block = self.blocks[block_height]
            if old_block is None:
                return False
            # Collect old transactions to potentially re-add
            old_txs = [tx for tx in old_block['transactions'] if tx.get('sender') not in ('GENESIS','MINING_REWARD')]
            # Rollback
            self.rollback_block(block_height)
            # Adjust new_block metadata
            new_block['height'] = block_height
            # Apply effects
            self._apply_block_effects(new_block)
            # Insert/update DB block row
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO blocks (height, hash, previous_hash, timestamp, transactions, nonce, difficulty, miner, reward)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    new_block['height'], new_block['hash'], new_block['previous_hash'], new_block['timestamp'], json.dumps(new_block['transactions']), new_block['nonce'], new_block['difficulty'], new_block.get('miner'), new_block.get('reward', 50.0)
                ))
                # Insert new transactions
                for tx in new_block['transactions']:
                    cursor.execute('''
                        INSERT INTO transactions (id, sender, receiver, amount, fee, block_height, signature)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (tx.get('id', f"ext_{tx_hash(tx)[:12]}"), tx.get('sender'), tx.get('receiver'), tx.get('amount',0), tx.get('fee',0.0), block_height, tx.get('signature')))
                conn.commit()
            # In-memory replace (after rollback the block index may have been popped if it was tip)
            if block_height == len(self.blocks):
                # Rolled back tip -> just append
                self.blocks.append(new_block)
            else:
                self.blocks[block_height] = new_block
            # Recompute cumulative work from block_height forward
            self._recompute_cumulative_work(start=block_height)
            # Re-add old txs not present in new block
            new_ids = {tx.get('id') for tx in new_block['transactions']}
            for tx in old_txs:
                if tx.get('id') not in new_ids:
                    # Return to mempool if still valid (balance check after rollback ensured)
                    # Refresh its id to avoid collision
                    tx_id = tx.get('id') or f"readd_{tx_hash(tx)[:12]}"
                    tx['id'] = tx_id
                    self.pending_transactions.append(tx)
                    self._add_mempool_tx(tx)
            return True

    # ---------------- Multi-block Reorg Support ---------------- #
    def _block_work(self, block: Dict) -> int:
        return 2 ** block.get('difficulty', self.mining_difficulty)

    def _recompute_cumulative_work(self, start: int = 0):
        if not self.blocks:
            return
        if start <= 0:
            cumulative = 0
            for b in self.blocks:
                w = self._block_work(b)
                cumulative += w
                b['cumulative_work'] = cumulative
        else:
            # ensure previous has cumulative_work
            if self.blocks[start-1].get('cumulative_work') is None:
                self._recompute_cumulative_work(0)
                return
            cumulative = self.blocks[start-1]['cumulative_work']
            for i in range(start, len(self.blocks)):
                w = self._block_work(self.blocks[i])
                cumulative += w
                self.blocks[i]['cumulative_work'] = cumulative

    def rollback_to_height(self, target_height: int) -> bool:
        """Rollback blocks down to target_height (exclusive of target's removal)."""
        if target_height < 0:
            return False
        while len(self.blocks) - 1 > target_height:
            h = len(self.blocks) - 1
            self.rollback_block(h)
        return True

    def apply_alternative_branch(self, start_height: int, alt_blocks: List[Dict]) -> bool:
        """Apply alternative branch starting after start_height.
        alt_blocks must be contiguous and start at start_height+1."""
        if not alt_blocks:
            return False
        if alt_blocks[0]['height'] != start_height + 1:
            return False
        # Validate linkage & timestamps before mutating
        prev_hash = self.blocks[start_height]['hash'] if start_height >=0 else '0'*64
        for b in alt_blocks:
            if b['previous_hash'] != prev_hash:
                return False
            if not self.validate_block_timestamp_external(b):
                return False
            prev_hash = b['hash']
        # Rollback to start_height
        self.rollback_to_height(start_height)
        # Apply alt blocks sequentially
        for b in alt_blocks:
            # Reuse accept logic but using effects apply
            block = {
                'height': b['height'],
                'hash': b['hash'],
                'previous_hash': b['previous_hash'],
                'timestamp': b['timestamp'],
                'transactions': b['transactions'],
                'nonce': b['nonce'],
                'difficulty': b.get('difficulty', self.mining_difficulty),
                'miner': b.get('miner','NETWORK'),
                'reward': b.get('reward', self.block_reward),
                'merkle_root': b.get('merkle_root'),
                'cumulative_work': None
            }
            self._apply_block_effects(block)
            self.blocks.append(block)
            # Update cumulative work for tip
            prev_cw = self.blocks[-2]['cumulative_work'] if len(self.blocks) > 1 and self.blocks[-2].get('cumulative_work') is not None else 0
            block['cumulative_work'] = prev_cw + self._block_work(block)
            self._save_block_to_db(block)
            for tx in block['transactions']:
                self._save_transaction_to_db(tx, block['height'])
                if tx.get('sender') not in ('GENESIS','MINING_REWARD'):
                    self._save_balance_to_db(tx['sender'], self.balances.get(tx['sender'],0))
                self._save_balance_to_db(tx['receiver'], self.balances.get(tx['receiver'],0))
            # Mempool cleanup for included txs
            mined_ids = {tx.get('id') for tx in block['transactions'] if tx.get('id')}
            self.pending_transactions = [tx for tx in self.pending_transactions if tx.get('id') not in mined_ids]
            for tx_id in mined_ids:
                self._remove_mempool_tx(tx_id)
        # Recompute cumulative work
        self._recompute_cumulative_work(start=start_height+1)
        self.reorg_events += 1
        return True

    def accept_network_block(self, block_data: Dict):
        """Append a network-received block at the chain tip with journaling."""
        with self.lock:
            expected_height = len(self.blocks)
            if block_data['height'] != expected_height:
                return False
            block = {
                'height': block_data['height'],
                'hash': block_data['hash'],
                'previous_hash': block_data['previous_hash'],
                'timestamp': block_data['timestamp'],
                'transactions': block_data['transactions'],
                'nonce': block_data['nonce'],
                'difficulty': block_data.get('difficulty', self.mining_difficulty),
                'miner': block_data.get('miner','NETWORK'),
                'reward': block_data.get('reward', self.block_reward),
                'merkle_root': block_data.get('merkle_root')
            }
            self._apply_block_effects(block)
            self.blocks.append(block)
            self._save_block_to_db(block)
            for tx in block['transactions']:
                self._save_transaction_to_db(tx, block['height'])
                if tx.get('sender') not in ('GENESIS','MINING_REWARD'):
                    self._save_balance_to_db(tx['sender'], self.balances.get(tx['sender'],0))
                self._save_balance_to_db(tx['receiver'], self.balances.get(tx['receiver'],0))
            # Remove mined txs from mempool
            mined_ids = {tx.get('id') for tx in block['transactions'] if tx.get('id')}
            self.pending_transactions = [tx for tx in self.pending_transactions if tx.get('id') not in mined_ids]
            for tx_id in mined_ids:
                self._remove_mempool_tx(tx_id)
            return True

    # --- Reorg skeleton (placeholder) ---
    def assess_chain_work(self) -> int:
        """Jednoduché kumulativní 'work' = sum(2^difficulty)."""
        total = 0
        for b in self.blocks:
            total += 2 ** b.get('difficulty', self.mining_difficulty)
        return total

    def maybe_accept_alternative_chain(self, alt_blocks: List[Dict]):
        """Skeleton – zatím neimplementuje plný reorg, pouze porovná work."""
        # Placeholder: future implementation
        return False
    
    def print_status(self):
        """Zobrazí status blockchainu"""
        print("\n🚀 NOVÝ ZION BLOCKCHAIN STATUS")
        print("=" * 50)
        print(f"📊 Počet bloků: {len(self.blocks)}")
        print(f"💰 Celková nabídka: {self.get_total_supply():,.0f} ZION")
        print(f"⚖️  Validní řetězec: {'✅ ANO' if self.validate_chain() else '❌ NE'}")
        print(f"📋 Čekající transakce: {len(self.pending_transactions)}")
        
        print(f"\n🏦 PREMINE DISTRIBUCE:")
        total_premine = 0
        for address, info in self.premine_addresses.items():
            balance = self.get_balance(address)
            total_premine += balance
            print(f"   {info['purpose']}: {balance:,.0f} ZION")
            print(f"      └─ {address[:30]}...")
        
        print(f"\n💎 Total Premine: {total_premine:,.0f} ZION")
        print(f"🆔 Latest Block: {self.blocks[-1]['hash'][:32]}..." if self.blocks else "No blocks")

def main():
    """Spustí demo nového blockchainu"""
    print("🚀 Inicializuji nový ZION blockchain s novými adresami...")
    
    # Vytvoření nového blockchainu
    blockchain = NewZionBlockchain()
    
    # Zobrazení počátečního stavu
    blockchain.print_status()
    
    # Test transakce
    print(f"\n🔄 Test transakce...")
    try:
        # Transakce z Sacred Mining Operator
        sacred_address = 'ZION_SACRED_B0FA7E2A234D8C2F08545F02295C98'
        test_address = 'ZION_TEST_USER_123456789'
        
        blockchain.create_transaction(
            sacred_address,
            test_address,
            100_000,
            "Test transakce z Sacred Operator"
        )
        
        # Vytěžení bloku
        miner_address = 'ZION_MINER_TESTER'
        blockchain.mine_pending_transactions(miner_address)
        
        print(f"\n✅ Transakce dokončena!")
        print(f"💰 Test user balance: {blockchain.get_balance(test_address):,.0f} ZION")
        print(f"⛏️  Miner reward: {blockchain.get_balance(miner_address):,.0f} ZION")
        
    except Exception as e:
        print(f"❌ Chyba při transakci: {e}")
    
    # Finální status
    blockchain.print_status()

if __name__ == "__main__":
    main()