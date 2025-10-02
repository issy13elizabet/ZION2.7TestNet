#!/usr/bin/env python3
"""
ZION 2.7.1 - Wallet System
Address management and transaction creation
"""

import os
import json
import hashlib
import secrets
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from core.real_blockchain import ZionRealBlockchain, RealTransaction


@dataclass
class WalletAddress:
    """Wallet address with private/public key pair"""
    address: str
    public_key: str
    private_key: str  # In production, this should be encrypted
    created_at: datetime
    label: str = ""

    def to_dict(self) -> Dict:
        return {
            'address': self.address,
            'public_key': self.public_key,
            'private_key': self.private_key,
            'created_at': self.created_at.isoformat(),
            'label': self.label
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'WalletAddress':
        return cls(
            address=data['address'],
            public_key=data['public_key'],
            private_key=data['private_key'],
            created_at=datetime.fromisoformat(data['created_at']),
            label=data.get('label', '')
        )


class ZionWallet:
    """ZION Wallet for address and transaction management"""

    def __init__(self, wallet_file: str = "zion_wallet.json"):
        self.wallet_file = wallet_file
        self.addresses: List[WalletAddress] = []
        self.blockchain = ZionRealBlockchain()

        # Load existing wallet
        self._load_wallet()

        # Create default address if none exists
        if not self.addresses:
            self.create_address("Default Address")

    def create_address(self, label: str = "") -> str:
        """Create a new wallet address"""
        # Generate address (simplified - in production use proper crypto)
        random_bytes = secrets.token_bytes(32)
        address = "ZION_" + hashlib.sha256(random_bytes).hexdigest()[:32].upper()
        public_key = hashlib.sha256(random_bytes + b"public").hexdigest()
        private_key = hashlib.sha256(random_bytes + b"private").hexdigest()

        wallet_addr = WalletAddress(
            address=address,
            public_key=public_key,
            private_key=private_key,
            created_at=datetime.now(),
            label=label
        )

        self.addresses.append(wallet_addr)
        self._save_wallet()

        print(f"✅ Created new address: {address}")
        return address

    def get_addresses(self) -> List[Dict]:
        """Get all wallet addresses"""
        return [addr.to_dict() for addr in self.addresses]

    def get_balance(self, address: str) -> int:
        """Get balance for specific address"""
        return self.blockchain.get_balance(address)

    def get_total_balance(self) -> int:
        """Get total balance across all addresses"""
        total = 0
        for addr in self.addresses:
            total += self.get_balance(addr.address)
        return total

    def create_transaction(self, from_address: str, to_address: str, amount: int,
                          fee: int = 1000) -> Optional[RealTransaction]:
        """Create a new transaction"""
        # Validate addresses
        if from_address not in [addr.address for addr in self.addresses]:
            print(f"❌ Address {from_address} not in wallet")
            return None

        # Check balance
        balance = self.get_balance(from_address)
        if balance < amount + fee:
            print(f"❌ Insufficient balance: {balance} < {amount + fee}")
            return None

        # Create transaction
        tx = RealTransaction(
            tx_id=hashlib.sha256(f"{from_address}{to_address}{amount}{fee}{datetime.now().isoformat()}".encode()).hexdigest(),
            from_address=from_address,
            to_address=to_address,
            amount=amount,
            fee=fee,
            timestamp=int(datetime.now().timestamp()),
            signature=self._sign_transaction(from_address, f"{to_address}{amount}{fee}"),
            consciousness_boost=1.0
        )

        # Add to blockchain mempool
        if self.blockchain.add_transaction(tx):
            print(f"✅ Transaction created: {tx.tx_id[:16]}...")
            print(f"   {from_address} → {to_address}")
            print(f"   Amount: {amount:,} atomic units")
            print(f"   Fee: {fee:,} atomic units")
            return tx

        return None

    def _sign_transaction(self, from_address: str, message: str) -> str:
        """Sign transaction message (simplified)"""
        # Find private key
        for addr in self.addresses:
            if addr.address == from_address:
                # Simplified signature - in production use proper crypto
                return hashlib.sha256(f"{addr.private_key}{message}".encode()).hexdigest()

        return ""

    def get_transaction_history(self, address: str) -> List[Dict]:
        """Get transaction history for address"""
        history = []

        # Check all blocks for transactions involving this address
        for block in self.blockchain.blocks:
            for tx_data in block.transactions:
                if (tx_data.get('from_address') == address or
                    tx_data.get('to_address') == address):
                    history.append({
                        'block_height': block.height,
                        'tx_data': tx_data,
                        'timestamp': block.timestamp,
                        'block_hash': block.hash[:16] + "..."
                    })

        return history

    def _load_wallet(self):
        """Load wallet from file"""
        if os.path.exists(self.wallet_file):
            try:
                with open(self.wallet_file, 'r') as f:
                    data = json.load(f)
                    self.addresses = [WalletAddress.from_dict(addr_data) for addr_data in data.get('addresses', [])]
                print(f"📂 Loaded wallet with {len(self.addresses)} addresses")
            except Exception as e:
                print(f"⚠️ Error loading wallet: {e}")

    def _save_wallet(self):
        """Save wallet to file"""
        data = {
            'version': '2.7.1',
            'addresses': [addr.to_dict() for addr in self.addresses]
        }

        with open(self.wallet_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"💾 Wallet saved with {len(self.addresses)} addresses")


# Global wallet instance
_wallet_instance: Optional[ZionWallet] = None

def get_wallet() -> ZionWallet:
    """Get global wallet instance"""
    global _wallet_instance
    if _wallet_instance is None:
        _wallet_instance = ZionWallet()
    return _wallet_instance


if __name__ == "__main__":
    # Test wallet
    wallet = ZionWallet()

    print("🌟 ZION Wallet Test")
    print(f"Addresses: {len(wallet.get_addresses())}")
    print(f"Total Balance: {wallet.get_total_balance():,} atomic units")

    # Create test transaction
    addresses = wallet.get_addresses()
    if len(addresses) >= 1:
        from_addr = addresses[0]['address']
        to_addr = wallet.create_address("Test Recipient")

        # Create transaction
        tx = wallet.create_transaction(from_addr, to_addr, 1000000)
        if tx:
            print(f"✅ Test transaction created: {tx.tx_id[:16]}...")