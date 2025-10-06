"""
ZION Wallet Core - Advanced wallet management with CryptoNote compatibility
"""

import os
import json
import hashlib
import secrets
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


@dataclass
class ZionAddress:
    """ZION address structure"""
    public_key: str
    address: str
    private_key: Optional[str] = None

@dataclass
class ZionTransaction:
    """ZION transaction structure"""
    txid: str
    from_address: str
    to_address: str
    amount: int  # in atomic units (1 ZION = 1000000 atomic units)
    fee: int
    timestamp: int
    confirmations: int
    status: str  # 'pending', 'confirmed', 'failed'
    block_height: Optional[int] = None

@dataclass
class WalletInfo:
    """Wallet information"""
    address: str
    balance: int
    unlocked_balance: int
    transaction_count: int
    created_at: int
    last_sync: Optional[int] = None


class ZionCrypto:
    """Cryptographic operations for ZION wallet"""
    
    @staticmethod
    def generate_keypair() -> Tuple[ed25519.Ed25519PrivateKey, ed25519.Ed25519PublicKey]:
        """Generate Ed25519 keypair"""
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        return private_key, public_key
    
    @staticmethod
    def derive_address(public_key: ed25519.Ed25519PublicKey) -> str:
        """Derive ZION address from public key"""
        pub_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        
        # ZION address format: ZION + base58(sha256(public_key)[:20])
        hash_digest = hashlib.sha256(pub_bytes).digest()[:20]
        address = "ZION" + base64.b32encode(hash_digest).decode('ascii').rstrip('=')
        return address
    
    @staticmethod
    def sign_transaction(private_key: ed25519.Ed25519PrivateKey, message: bytes) -> bytes:
        """Sign transaction data"""
        return private_key.sign(message)
    
    @staticmethod
    def verify_signature(public_key: ed25519.Ed25519PublicKey, signature: bytes, message: bytes) -> bool:
        """Verify transaction signature"""
        try:
            public_key.verify(signature, message)
            return True
        except Exception:
            return False
    
    @staticmethod
    def encrypt_data(data: bytes, password: str, salt: bytes = None) -> Tuple[bytes, bytes]:
        """Encrypt data with password using AES-256-GCM"""
        if salt is None:
            salt = secrets.token_bytes(16)
        
        # Derive key from password
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = kdf.derive(password.encode('utf-8'))
        
        # Generate IV
        iv = secrets.token_bytes(12)
        
        # Encrypt
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        # Combine salt + iv + tag + ciphertext
        encrypted_data = salt + iv + encryptor.tag + ciphertext
        return encrypted_data, salt
    
    @staticmethod
    def decrypt_data(encrypted_data: bytes, password: str) -> bytes:
        """Decrypt data with password"""
        # Extract components
        salt = encrypted_data[:16]
        iv = encrypted_data[16:28]
        tag = encrypted_data[28:44]
        ciphertext = encrypted_data[44:]
        
        # Derive key from password
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = kdf.derive(password.encode('utf-8'))
        
        # Decrypt
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend())
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        return plaintext


class ZionWallet:
    """Advanced ZION Wallet with CryptoNote compatibility"""
    
    def __init__(self, wallet_dir: str = "~/.zion/wallets"):
        self.wallet_dir = Path(wallet_dir).expanduser()
        self.wallet_dir.mkdir(parents=True, exist_ok=True)
        
        # Current wallet state
        self.wallet_name: Optional[str] = None
        self.address: Optional[ZionAddress] = None
        self.is_unlocked: bool = False
        self.transactions: List[ZionTransaction] = []
        self.balance: int = 0
        self.unlocked_balance: int = 0
        
        # Wallet file paths
        self.wallet_file: Optional[Path] = None
        self.keys_file: Optional[Path] = None
        
    def create_wallet(self, name: str, password: str, language: str = "english") -> Dict:
        """Create new wallet with mnemonic seed"""
        try:
            # Generate keypair
            private_key, public_key = ZionCrypto.generate_keypair()
            
            # Create ZION address
            address = ZionCrypto.derive_address(public_key)
            
            # Generate mnemonic (simplified - would use BIP39 in production)
            entropy = secrets.token_bytes(32)
            mnemonic = self._entropy_to_mnemonic(entropy)
            
            # Create wallet data structure
            wallet_data = {
                'version': '2.6.75',
                'name': name,
                'address': address,
                'created_at': int(time.time()),
                'language': language,
                'encrypted_seed': None,
                'encrypted_private_key': None,
            }
            
            # Serialize private key
            private_key_bytes = private_key.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            # Encrypt sensitive data
            encrypted_seed, salt = ZionCrypto.encrypt_data(entropy, password)
            encrypted_key, _ = ZionCrypto.encrypt_data(private_key_bytes, password, salt)
            
            wallet_data['encrypted_seed'] = base64.b64encode(encrypted_seed).decode('ascii')
            wallet_data['encrypted_private_key'] = base64.b64encode(encrypted_key).decode('ascii')
            
            # Save wallet files
            wallet_file = self.wallet_dir / f"{name}.wallet"
            keys_file = self.wallet_dir / f"{name}.keys"
            
            with open(wallet_file, 'w') as f:
                json.dump(wallet_data, f, indent=2)
            
            # Keys file contains only address and public key (not sensitive)
            keys_data = {
                'address': address,
                'public_key': base64.b64encode(
                    public_key.public_bytes(
                        encoding=serialization.Encoding.Raw,
                        format=serialization.PublicFormat.Raw
                    )
                ).decode('ascii'),
                'created_at': wallet_data['created_at']
            }
            
            with open(keys_file, 'w') as f:
                json.dump(keys_data, f, indent=2)
            
            # Load wallet
            self.wallet_name = name
            self.wallet_file = wallet_file
            self.keys_file = keys_file
            
            return {
                'success': True,
                'address': address,
                'mnemonic': mnemonic,
                'message': f'Wallet "{name}" created successfully'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to create wallet: {str(e)}'
            }
    
    def open_wallet(self, name: str, password: str) -> Dict:
        """Open and unlock existing wallet"""
        try:
            wallet_file = self.wallet_dir / f"{name}.wallet"
            keys_file = self.wallet_dir / f"{name}.keys"
            
            if not wallet_file.exists():
                return {
                    'success': False,
                    'error': f'Wallet "{name}" not found'
                }
            
            # Load wallet data
            with open(wallet_file, 'r') as f:
                wallet_data = json.load(f)
            
            with open(keys_file, 'r') as f:
                keys_data = json.load(f)
            
            # Decrypt private key to verify password
            try:
                encrypted_key = base64.b64decode(wallet_data['encrypted_private_key'])
                private_key_bytes = ZionCrypto.decrypt_data(encrypted_key, password)
                
                # Reconstruct private key
                private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_key_bytes)
                public_key = private_key.public_key()
                
                # Verify address matches
                derived_address = ZionCrypto.derive_address(public_key)
                if derived_address != wallet_data['address']:
                    return {
                        'success': False,
                        'error': 'Wallet corruption detected'
                    }
                
                # Set wallet state
                self.wallet_name = name
                self.wallet_file = wallet_file
                self.keys_file = keys_file
                self.address = ZionAddress(
                    public_key=keys_data['public_key'],
                    address=wallet_data['address'],
                    private_key=base64.b64encode(private_key_bytes).decode('ascii')
                )
                self.is_unlocked = True
                
                # Load transactions and balance (would sync with blockchain in production)
                self._load_wallet_state()
                
                return {
                    'success': True,
                    'address': self.address.address,
                    'balance': self.balance,
                    'unlocked_balance': self.unlocked_balance,
                    'message': f'Wallet "{name}" opened successfully'
                }
                
            except Exception:
                return {
                    'success': False,
                    'error': 'Invalid password'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to open wallet: {str(e)}'
            }
    
    def lock_wallet(self) -> Dict:
        """Lock wallet (clear sensitive data from memory)"""
        if self.address:
            self.address.private_key = None
        self.is_unlocked = False
        
        return {
            'success': True,
            'message': 'Wallet locked successfully'
        }
    
    def get_balance(self) -> Dict:
        """Get wallet balance"""
        if not self.is_unlocked:
            return {
                'success': False,
                'error': 'Wallet is locked'
            }
        
        return {
            'success': True,
            'balance': self.balance,
            'unlocked_balance': self.unlocked_balance,
            'address': self.address.address
        }
    
    def create_transaction(self, to_address: str, amount: int, fee: int = 1000) -> Dict:
        """Create and sign transaction"""
        if not self.is_unlocked:
            return {
                'success': False,
                'error': 'Wallet is locked'
            }
        
        if amount + fee > self.unlocked_balance:
            return {
                'success': False,
                'error': 'Insufficient balance'
            }
        
        try:
            # Create transaction data
            tx_data = {
                'from': self.address.address,
                'to': to_address,
                'amount': amount,
                'fee': fee,
                'timestamp': int(time.time()),
                'nonce': secrets.randbits(64)
            }
            
            # Serialize transaction for signing
            tx_json = json.dumps(tx_data, sort_keys=True).encode('utf-8')
            
            # Sign transaction
            private_key_bytes = base64.b64decode(self.address.private_key)
            private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_key_bytes)
            signature = ZionCrypto.sign_transaction(private_key, tx_json)
            
            # Create transaction ID
            tx_hash = hashlib.sha256(tx_json + signature).hexdigest()
            
            # Create transaction object
            transaction = ZionTransaction(
                txid=tx_hash,
                from_address=self.address.address,
                to_address=to_address,
                amount=amount,
                fee=fee,
                timestamp=tx_data['timestamp'],
                confirmations=0,
                status='pending'
            )
            
            # Add to pending transactions
            self.transactions.insert(0, transaction)
            
            return {
                'success': True,
                'txid': tx_hash,
                'transaction': asdict(transaction),
                'message': 'Transaction created successfully'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to create transaction: {str(e)}'
            }
    
    def get_transactions(self, limit: int = 50) -> Dict:
        """Get transaction history"""
        if not self.is_unlocked:
            return {
                'success': False,
                'error': 'Wallet is locked'
            }
        
        transactions = [asdict(tx) for tx in self.transactions[:limit]]
        
        return {
            'success': True,
            'transactions': transactions,
            'count': len(transactions)
        }
    
    def restore_from_mnemonic(self, name: str, mnemonic: str, password: str) -> Dict:
        """Restore wallet from mnemonic seed"""
        try:
            # Convert mnemonic to entropy (simplified)
            entropy = self._mnemonic_to_entropy(mnemonic)
            
            # Generate keypair from entropy
            private_key_bytes = hashlib.pbkdf2_hmac('sha256', entropy, b'zion_seed', 4096, 32)
            private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_key_bytes)
            public_key = private_key.public_key()
            
            # Derive address
            address = ZionCrypto.derive_address(public_key)
            
            # Create wallet (similar to create_wallet but with restored keys)
            wallet_data = {
                'version': '2.6.75',
                'name': name,
                'address': address,
                'created_at': int(time.time()),
                'language': 'english',
                'restored': True,
                'encrypted_seed': None,
                'encrypted_private_key': None,
            }
            
            # Encrypt sensitive data
            encrypted_seed, salt = ZionCrypto.encrypt_data(entropy, password)
            encrypted_key, _ = ZionCrypto.encrypt_data(private_key_bytes, password, salt)
            
            wallet_data['encrypted_seed'] = base64.b64encode(encrypted_seed).decode('ascii')
            wallet_data['encrypted_private_key'] = base64.b64encode(encrypted_key).decode('ascii')
            
            # Save wallet
            wallet_file = self.wallet_dir / f"{name}.wallet"
            keys_file = self.wallet_dir / f"{name}.keys"
            
            with open(wallet_file, 'w') as f:
                json.dump(wallet_data, f, indent=2)
            
            keys_data = {
                'address': address,
                'public_key': base64.b64encode(
                    public_key.public_bytes(
                        encoding=serialization.Encoding.Raw,
                        format=serialization.PublicFormat.Raw
                    )
                ).decode('ascii'),
                'created_at': wallet_data['created_at'],
                'restored': True
            }
            
            with open(keys_file, 'w') as f:
                json.dump(keys_data, f, indent=2)
            
            return {
                'success': True,
                'address': address,
                'message': f'Wallet "{name}" restored successfully'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to restore wallet: {str(e)}'
            }
    
    def list_wallets(self) -> Dict:
        """List available wallets"""
        try:
            wallets = []
            for wallet_file in self.wallet_dir.glob("*.wallet"):
                try:
                    with open(wallet_file, 'r') as f:
                        wallet_data = json.load(f)
                    
                    wallets.append({
                        'name': wallet_data['name'],
                        'address': wallet_data['address'],
                        'created_at': wallet_data['created_at'],
                        'restored': wallet_data.get('restored', False)
                    })
                except Exception:
                    continue
            
            return {
                'success': True,
                'wallets': wallets,
                'count': len(wallets)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to list wallets: {str(e)}'
            }
    
    def _load_wallet_state(self):
        """Load wallet state (transactions, balance) - mock implementation"""
        # In production, this would sync with the blockchain
        # For demo, we'll create some mock data
        
        self.balance = 50000000  # 50 ZION
        self.unlocked_balance = 45000000  # 45 ZION
        
        # Mock transactions
        if not self.transactions:
            mock_txs = [
                ZionTransaction(
                    txid="a1b2c3d4e5f6",
                    from_address="ZION_NETWORK_REWARD",
                    to_address=self.address.address,
                    amount=10000000,
                    fee=0,
                    timestamp=int(time.time()) - 86400,
                    confirmations=144,
                    status='confirmed',
                    block_height=1000
                ),
                ZionTransaction(
                    txid="f6e5d4c3b2a1",
                    from_address=self.address.address,
                    to_address="ZIONTEST123456789ABCDEF",
                    amount=5000000,
                    fee=1000,
                    timestamp=int(time.time()) - 43200,
                    confirmations=72,
                    status='confirmed',
                    block_height=1072
                )
            ]
            self.transactions = mock_txs
    
    def _entropy_to_mnemonic(self, entropy: bytes) -> str:
        """Convert entropy to mnemonic (simplified - use BIP39 in production)"""
        # This is a simplified implementation
        # In production, use proper BIP39 wordlist
        words = []
        for i in range(0, len(entropy), 2):
            word_index = int.from_bytes(entropy[i:i+2], 'big') % 2048
            words.append(f"word{word_index:04d}")
        return ' '.join(words[:24])  # 24-word mnemonic
    
    def _mnemonic_to_entropy(self, mnemonic: str) -> bytes:
        """Convert mnemonic to entropy (simplified)"""
        words = mnemonic.split()
        entropy = b''
        for word in words:
            if word.startswith('word'):
                word_index = int(word[4:])
                entropy += word_index.to_bytes(2, 'big')
        return entropy[:32]  # 32 bytes entropy


# CLI interface for wallet operations
def main():
    """Main CLI interface"""
    import sys
    
    if len(sys.argv) < 2:
        print("ZION Wallet CLI v2.6.75")
        print("Usage:")
        print("  python zion_wallet.py create <name> <password>")
        print("  python zion_wallet.py open <name> <password>")
        print("  python zion_wallet.py list")
        print("  python zion_wallet.py restore <name> <mnemonic> <password>")
        return
    
    wallet = ZionWallet()
    command = sys.argv[1]
    
    if command == "create":
        if len(sys.argv) != 4:
            print("Usage: create <name> <password>")
            return
        
        name, password = sys.argv[2], sys.argv[3]
        result = wallet.create_wallet(name, password)
        
        if result['success']:
            print(f"✅ {result['message']}")
            print(f"Address: {result['address']}")
            print(f"Mnemonic: {result['mnemonic']}")
            print("⚠️  Save your mnemonic safely!")
        else:
            print(f"❌ {result['error']}")
    
    elif command == "open":
        if len(sys.argv) != 4:
            print("Usage: open <name> <password>")
            return
        
        name, password = sys.argv[2], sys.argv[3]
        result = wallet.open_wallet(name, password)
        
        if result['success']:
            print(f"✅ {result['message']}")
            print(f"Address: {result['address']}")
            print(f"Balance: {result['balance'] / 1000000:.6f} ZION")
        else:
            print(f"❌ {result['error']}")
    
    elif command == "list":
        result = wallet.list_wallets()
        
        if result['success']:
            print(f"Found {result['count']} wallets:")
            for w in result['wallets']:
                status = " (restored)" if w.get('restored') else ""
                print(f"  {w['name']}: {w['address'][:20]}...{status}")
        else:
            print(f"❌ {result['error']}")
    
    elif command == "restore":
        if len(sys.argv) != 5:
            print("Usage: restore <name> <mnemonic> <password>")
            return
        
        name, mnemonic, password = sys.argv[2], sys.argv[3], sys.argv[4]
        result = wallet.restore_from_mnemonic(name, mnemonic, password)
        
        if result['success']:
            print(f"✅ {result['message']}")
            print(f"Address: {result['address']}")
        else:
            print(f"❌ {result['error']}")
    
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()