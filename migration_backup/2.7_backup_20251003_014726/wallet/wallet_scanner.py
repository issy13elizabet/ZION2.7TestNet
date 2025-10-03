"""Wallet Scanner Skeleton (Phase 2)
Scans blockchain UTXO set for outputs belonging to a set of addresses.
Future: incremental scanning with last scanned height persistence.
"""
from __future__ import annotations
from typing import List, Dict, Any, Set
import os, json
from core.blockchain import Blockchain

class WalletScanner:
    def __init__(self, chain: Blockchain, addresses: List[str], state_path: str = None):
        self.chain = chain
        self.addresses: Set[str] = set(addresses)
        self.last_height_scanned = -1
        self.state_path = state_path or os.path.join(os.path.dirname(__file__), 'scanner_state.json')
        self._load_state()

    def scan_full(self) -> Dict[str, Any]:
        owned_utxos = []
        total = 0
        for (txid, vout), data in self.chain.utxos.items():
            if data['address'] in self.addresses:
                owned_utxos.append({'txid': txid, 'vout': vout, 'amount': data['amount'], 'address': data['address']})
                total += data['amount']
        self.last_height_scanned = self.chain.height()
        return {
            'height': self.last_height_scanned,
            'utxos': owned_utxos,
            'balance': total
        }

    def add_address(self, addr: str):
        self.addresses.add(addr)

    # ---- Incremental Scan ----
    def scan_incremental(self) -> Dict[str, Any]:
        """Scan only new blocks since last_height_scanned; update state."""
        start_height = self.last_height_scanned + 1
        end_height = self.chain.height() - 1
        if start_height > end_height:
            return {'updated': False, 'height': self.last_height_scanned}
        # For simplicity, just rescan UTXO diff by reconstructing owned view (UTXO set is global already)
        owned = self._collect_owned()
        self.last_height_scanned = end_height
        self._save_state()
        return {'updated': True, 'height': self.last_height_scanned, 'balance': owned['balance'], 'utxos': owned['utxos']}

    def _collect_owned(self) -> Dict[str, Any]:
        owned_utxos = []
        total = 0
        for (txid, vout), data in self.chain.utxos.items():
            if data['address'] in self.addresses:
                owned_utxos.append({'txid': txid, 'vout': vout, 'amount': data['amount'], 'address': data['address']})
                total += data['amount']
        return {'utxos': owned_utxos, 'balance': total}

    # ---- Persistence ----
    def _load_state(self):
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, 'r') as f:
                    data = json.load(f)
                self.last_height_scanned = data.get('last_height_scanned', -1)
                saved_addrs = data.get('addresses', [])
                # merge without losing new ones
                self.addresses.update(saved_addrs)
        except Exception:
            pass

    def _save_state(self):
        try:
            tmp = self.state_path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump({
                    'last_height_scanned': self.last_height_scanned,
                    'addresses': list(self.addresses)
                }, f)
            os.replace(tmp, self.state_path)
        except Exception:
            pass

if __name__ == '__main__':
    # simple manual test
    chain = Blockchain()
    scanner = WalletScanner(chain, [chain.genesis_address])
    print(scanner.scan_full())
