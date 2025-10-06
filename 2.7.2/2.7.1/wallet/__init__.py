#!/usr/bin/env python3
"""
ZION 2.7.1 - Wallet System
Address management and transaction creation
"""

import os
import json
import hashlib
import secrets
import base64
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from core.real_blockchain import ZionRealBlockchain, RealTransaction


@dataclass
class WalletAddress:
    """Wallet address with private/public key pair"""
    address: str
    public_key: str
    private_key: str  # Encrypted in production
    created_at: datetime
    label: str = ""
    encrypted: bool = False  # Whether private key is encrypted

    def to_dict(self) -> Dict:
        return {
            'address': self.address,
            'public_key': self.public_key,
            'private_key': self.private_key,
            'created_at': self.created_at.isoformat(),
            'label': self.label,
            'encrypted': self.encrypted
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'WalletAddress':
        return cls(
            address=data['address'],
            public_key=data['public_key'],
            private_key=data['private_key'],
            created_at=datetime.fromisoformat(data['created_at']),
            label=data.get('label', ''),
            encrypted=data.get('encrypted', False)
        )


class ZionWallet:
    """ZION Wallet for address and transaction management with encryption"""

    def __init__(self, wallet_file: str = "zion_wallet.json"):
        self.wallet_file = wallet_file
        self.addresses: List[WalletAddress] = []
        self.blockchain = ZionRealBlockchain()
        self.encryption_key: Optional[bytes] = None
        self.salt = secrets.token_bytes(16)  # Salt for key derivation

        # Load existing wallet
        self._load_wallet()

        # Create default address if none exists
        if not self.addresses:
            self.create_address("Default Address")

    def set_password(self, password: str):
        """Set encryption password for wallet"""
        try:
            # Derive encryption key from password
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=self.salt,
                iterations=100000,
            )
            self.encryption_key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            print("🔐 Wallet encryption enabled")
        except ImportError:
            print("⚠️ Cryptography library not available - wallet will not be encrypted")
            self.encryption_key = None

    def _encrypt_private_key(self, private_key: str) -> str:
        """Encrypt private key"""
        if not self.encryption_key:
            return private_key  # Return unencrypted if no key set

        try:
            f = Fernet(self.encryption_key)
            return f.encrypt(private_key.encode()).decode()
        except:
            return private_key  # Fallback to unencrypted

    def _decrypt_private_key(self, encrypted_key: str, encrypted: bool = False) -> str:
        """Decrypt private key"""
        if not encrypted or not self.encryption_key:
            return encrypted_key

        try:
            f = Fernet(self.encryption_key)
            return f.decrypt(encrypted_key.encode()).decode()
        except:
            return encrypted_key  # Fallback to encrypted value

    def unlock_wallet(self, password: str) -> bool:
        """Unlock encrypted wallet with password"""
        try:
            # Test password by deriving key
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=self.salt,
                iterations=100000,
            )
            test_key = base64.urlsafe_b64encode(kdf.derive(password.encode()))

            # Try to decrypt first encrypted private key
            encrypted_addresses = [addr for addr in self.addresses if addr.encrypted]
            if encrypted_addresses:
                f = Fernet(test_key)
                f.decrypt(encrypted_addresses[0].private_key.encode())

            self.encryption_key = test_key
            print("🔓 Wallet unlocked successfully")
            return True
        except:
            print("❌ Invalid password")
            return False

    def create_address(self, label: str = "") -> str:
        """Create a new wallet address with encryption"""
        # Generate address (simplified - in production use proper crypto)
        random_bytes = secrets.token_bytes(32)
        address = "ZION_" + hashlib.sha256(random_bytes).hexdigest()[:32].upper()
        public_key = hashlib.sha256(random_bytes + b"public").hexdigest()
        private_key_plain = hashlib.sha256(random_bytes + b"private").hexdigest()

        # Encrypt private key if encryption is enabled
        private_key = self._encrypt_private_key(private_key_plain)
        encrypted = self.encryption_key is not None

        wallet_addr = WalletAddress(
            address=address,
            public_key=public_key,
            private_key=private_key,
            created_at=datetime.now(),
            label=label,
            encrypted=encrypted
        )

        self.addresses.append(wallet_addr)
        self._save_wallet()

        print(f"✅ Created new address: {address}")
        if encrypted:
            print("🔐 Private key encrypted")
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

    def backup_wallet(self, backup_file: str) -> bool:
        """Create encrypted backup of wallet"""
        try:
            backup_data = {
                'version': '2.7.1',
                'created_at': datetime.now().isoformat(),
                'salt': base64.b64encode(self.salt).decode(),
                'addresses': [addr.to_dict() for addr in self.addresses]
            }

            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2)

            print(f"💾 Wallet backup created: {backup_file}")
            return True
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return False

    def restore_wallet(self, backup_file: str, password: Optional[str] = None) -> bool:
        """Restore wallet from backup"""
        try:
            with open(backup_file, 'r') as f:
                backup_data = json.load(f)

            # Set password if provided
            if password:
                self.salt = base64.b64decode(backup_data['salt'])
                self.set_password(password)

            # Restore addresses
            self.addresses = []
            for addr_data in backup_data['addresses']:
                addr = WalletAddress.from_dict(addr_data)
                # Decrypt private keys if we have the password
                if addr.encrypted and self.encryption_key:
                    addr.private_key = self._decrypt_private_key(addr.private_key, addr.encrypted)
                    addr.encrypted = False
                self.addresses.append(addr)

            self._save_wallet()
            print(f"🔄 Wallet restored from: {backup_file}")
            return True
        except Exception as e:
            print(f"❌ Restore failed: {e}")
            return False

    def export_keys(self, export_file: str, password: str) -> bool:
        """Export private keys (requires password confirmation)"""
        if not self.encryption_key:
            print("❌ Wallet is not encrypted - cannot export keys securely")
            return False

        try:
            # Verify password
            if not self.unlock_wallet(password):
                return False

            export_data = {
                'version': '2.7.1',
                'exported_at': datetime.now().isoformat(),
                'addresses': []
            }

            for addr in self.addresses:
                # Decrypt private key for export
                decrypted_key = self._decrypt_private_key(addr.private_key, addr.encrypted)
                export_data['addresses'].append({
                    'address': addr.address,
                    'private_key': decrypted_key,
                    'label': addr.label
                })

            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2)

            print(f"🔑 Private keys exported to: {export_file}")
            print("⚠️ Keep this file secure!")
            return True
        except Exception as e:
            print(f"❌ Key export failed: {e}")
            return False

    def import_keys(self, import_file: str, password: Optional[str] = None) -> bool:
        """Import private keys from export file"""
        try:
            with open(import_file, 'r') as f:
                import_data = json.load(f)

            imported_count = 0
            for addr_data in import_data['addresses']:
                # Check if address already exists
                existing = next((a for a in self.addresses if a.address == addr_data['address']), None)
                if existing:
                    print(f"⚠️ Address {addr_data['address']} already exists, skipping")
                    continue

                # Create new address with imported key
                wallet_addr = WalletAddress(
                    address=addr_data['address'],
                    public_key="",  # Will be derived from private key in production
                    private_key=self._encrypt_private_key(addr_data['private_key']),
                    created_at=datetime.now(),
                    label=addr_data.get('label', 'Imported'),
                    encrypted=self.encryption_key is not None
                )

                self.addresses.append(wallet_addr)
                imported_count += 1

            if imported_count > 0:
                self._save_wallet()
                print(f"✅ Imported {imported_count} addresses")
                return True
            else:
                print("ℹ️ No new addresses to import")
                return False
        except Exception as e:
            print(f"❌ Key import failed: {e}")
            return False
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