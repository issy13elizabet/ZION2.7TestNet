#!/usr/bin/env python3
"""ZION Crypto Utilities
Provides simple ECDSA (secp256k1) key management, signing, and verification.
WARNING: Educational / prototype quality – not hardened.
"""
from __future__ import annotations
import os
import hashlib
from dataclasses import dataclass
from typing import Tuple

try:
    from ecdsa import SigningKey, VerifyingKey, SECP256k1, BadSignatureError
except ImportError as e:
    raise ImportError("Missing 'ecdsa' package. Add to requirements.txt: ecdsa==0.19.0") from e


@dataclass
class KeyPair:
    private_key_hex: str
    public_key_hex: str

    def address(self) -> str:
        # Simplified address derivation: HASH160(public_key) truncated
        pub_bytes = bytes.fromhex(self.public_key_hex)
        sha = hashlib.sha256(pub_bytes).digest()
        rip = hashlib.new('ripemd160', sha).hexdigest()
        return f"ZION_{rip[:40].upper()}"


def generate_keypair() -> KeyPair:
    sk = SigningKey.generate(curve=SECP256k1)
    vk = sk.verifying_key
    return KeyPair(private_key_hex=sk.to_string().hex(), public_key_hex=vk.to_string().hex())


def sign_message(private_key_hex: str, message: bytes) -> str:
    sk = SigningKey.from_string(bytes.fromhex(private_key_hex), curve=SECP256k1)
    sig = sk.sign_deterministic(message, hashfunc=hashlib.sha256)
    return sig.hex()


def verify_signature(public_key_hex: str, message: bytes, signature_hex: str) -> bool:
    try:
        vk = VerifyingKey.from_string(bytes.fromhex(public_key_hex), curve=SECP256k1)
        vk.verify(bytes.fromhex(signature_hex), message, hashfunc=hashlib.sha256)
        return True
    except BadSignatureError:
        return False
    except Exception:
        return False


def canonical_transaction_dict(tx: dict) -> dict:
    # Fields included in the signing process (exclude signature itself)
    return {
        'sender': tx.get('sender'),
        'receiver': tx.get('receiver'),
        'amount': tx.get('amount'),
        'fee': tx.get('fee', 0.0),
        'timestamp': tx.get('timestamp'),
        'purpose': tx.get('purpose', '')
    }


def tx_hash(tx: dict) -> str:
    import json
    canonical = canonical_transaction_dict(tx)
    encoded = json.dumps(canonical, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def sign_transaction(tx: dict, private_key_hex: str) -> dict:
    h = tx_hash(tx)
    sig = sign_message(private_key_hex, h.encode())
    tx['signature'] = sig
    tx['tx_hash'] = h
    return tx


def verify_transaction_signature(tx: dict, public_key_hex: str) -> bool:
    if 'signature' not in tx or 'tx_hash' not in tx:
        return False
    return verify_signature(public_key_hex, tx['tx_hash'].encode(), tx['signature'])
