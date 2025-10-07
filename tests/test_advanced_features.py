#!/usr/bin/env python3
"""Advanced feature tests for new enhancements.
Run with: PYTHONPATH=. python3 tests/test_advanced_features.py
"""
import time
from new_zion_blockchain import NewZionBlockchain
from crypto_utils import generate_keypair, sign_transaction, tx_hash


def test_merkle_root_and_difficulty():
    bc = NewZionBlockchain(db_file='test_adv.db', enable_p2p=False, enable_rpc=False)
    sacred = list(bc.premine_addresses.keys())[0]
    bc.create_transaction(sacred, 'ZION_RCV_1', 123, 'MR test 1')
    bc.create_transaction(sacred, 'ZION_RCV_2', 321, 'MR test 2')
    h = bc.mine_pending_transactions('MINER_X')
    blk = bc.blocks[-1]
    assert 'merkle_root' in blk and len(blk['merkle_root']) == 64
    assert blk['hash'].startswith('0' * blk['difficulty'])


def test_tamper_block_detection():
    bc = NewZionBlockchain(db_file='test_adv2.db', enable_p2p=False, enable_rpc=False)
    sacred = list(bc.premine_addresses.keys())[0]
    bc.create_transaction(sacred, 'ZION_RCV_3', 50, 'Legit')
    bc.mine_pending_transactions('MINER_Y')
    # Tamper
    bc.blocks[-1]['transactions'][0]['amount'] = 999999
    assert bc.validate_chain() is False


def main():
    test_merkle_root_and_difficulty()
    test_tamper_block_detection()
    # Signed transaction pipeline (local)
    bc = NewZionBlockchain(db_file='test_signed.db', enable_p2p=False, enable_rpc=False)
    kp = generate_keypair()
    # Fund new address from premine
    sacred = list(bc.premine_addresses.keys())[0]
    bc.create_transaction(sacred, kp.address(), 1000, 'Fund new address')
    bc.mine_pending_transactions('MINER_SIGN')
    # Build raw tx (manually) with correct current nonce (should be 0 for this address after funding tx mined)
    current_nonce = bc.nonces.get(kp.address(), 0)
    raw_tx = {
        'id': f"manual_{int(time.time())}",
        'sender': kp.address(),
        'receiver': 'ZION_TARGET_X',
        'amount': 123,
        'fee': 0.0,
        'timestamp': time.time(),
        'purpose': 'Signed transfer',
        'nonce': current_nonce,
        'signature': 'UNSIGNED'
    }
    signed = sign_transaction(raw_tx, kp.private_key_hex)
    bc.add_signed_transaction(signed)
    bc.mine_pending_transactions('MINER_SIGN2')
    assert bc.get_balance('ZION_TARGET_X') >= 123
    print('Advanced feature tests passed.')


if __name__ == '__main__':
    main()