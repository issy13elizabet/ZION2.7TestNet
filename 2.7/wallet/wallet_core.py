"""
ZION 2.7 Wallet Core (Clean) - NO MOCK BALANCES
Only key generation & signing. Sync layer to be implemented.
"""
import os, json, time, hashlib, secrets, base64
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

@dataclass
class WalletKeys:
    address: str
    public_key: str
    encrypted_private_key: str
    created_at: int

class WalletCrypto:
    @staticmethod
    def generate_keypair() -> Tuple[ed25519.Ed25519PrivateKey, ed25519.Ed25519PublicKey]:
        sk = ed25519.Ed25519PrivateKey.generate()
        return sk, sk.public_key()

    @staticmethod
    def derive_address(pub: ed25519.Ed25519PublicKey) -> str:
        pb = pub.public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
        digest = hashlib.sha256(pb).digest()[:20]
        return 'ZION' + base64.b32encode(digest).decode().rstrip('=')

    @staticmethod
    def encrypt(data: bytes, password: str, salt: bytes=None) -> bytes:
        if not salt:
            salt = secrets.token_bytes(16)
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000, backend=default_backend())
        key = kdf.derive(password.encode())
        iv = secrets.token_bytes(12)
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
        enc = cipher.encryptor()
        ct = enc.update(data) + enc.finalize()
        return salt + iv + enc.tag + ct

    @staticmethod
    def decrypt(blob: bytes, password: str) -> bytes:
        salt, iv, tag, ct = blob[:16], blob[16:28], blob[28:44], blob[44:]
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000, backend=default_backend())
        key = kdf.derive(password.encode())
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend())
        dec = cipher.decryptor()
        return dec.update(ct) + dec.finalize()

class Wallet:
    def __init__(self, directory: str = '~/.zion2_7/wallets'):
        self.dir = Path(directory).expanduser()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.address: Optional[str] = None
        self.encrypted_private: Optional[str] = None

    def create(self, name: str, password: str) -> WalletKeys:
        sk, pk = WalletCrypto.generate_keypair()
        address = WalletCrypto.derive_address(pk)
        sk_raw = sk.private_bytes(encoding=serialization.Encoding.Raw, format=serialization.PrivateFormat.Raw, encryption_algorithm=serialization.NoEncryption())
        enc = WalletCrypto.encrypt(sk_raw, password)
        blob_b64 = base64.b64encode(enc).decode()
        keys = WalletKeys(address=address, public_key=base64.b64encode(pk.public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)).decode(), encrypted_private_key=blob_b64, created_at=int(time.time()))
        with open(self.dir / f'{name}.wallet','w') as f:
            json.dump(keys.__dict__, f, indent=2)
        self.address = address
        self.encrypted_private = blob_b64
        return keys

    def list_wallets(self):
        out = []
        for wf in self.dir.glob('*.wallet'):
            try:
                data = json.loads(wf.read_text())
                out.append({'name': wf.stem, 'address': data.get('address'), 'created_at': data.get('created_at')})
            except Exception:
                continue
        return out

if __name__ == '__main__':
    w = Wallet()
    k = w.create('testwallet','testpass')
    print('Created wallet:', k.address)
    print('Wallet list:', w.list_wallets())
