#!/usr/bin/env python3
"""Tests: Nonce persistence & single-block rollback replacement logic."""
import os
import shutil
import time
from new_zion_blockchain import NewZionBlockchain

DB_FILE = "test_reorg_chain.db"

def fresh_chain():
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    return NewZionBlockchain(db_file=DB_FILE, enable_p2p=False, enable_rpc=False)

def test_nonce_persistence():
    chain = fresh_chain()
    addr_from = list(chain.premine_addresses.keys())[0]
    addr_to = 'ZION_TEST_ADDR_X'
    # Create and record nonce
    tx1 = chain.create_transaction(addr_from, addr_to, 10_000)
    first_nonce = tx1['nonce']
    # Simulate restart
    del chain
    chain2 = NewZionBlockchain(db_file=DB_FILE, enable_p2p=False, enable_rpc=False)
    # New tx should have nonce first_nonce+1
    tx2 = chain2.create_transaction(addr_from, addr_to, 5_000)
    assert tx2['nonce'] == first_nonce + 1, f"Expected nonce {first_nonce+1}, got {tx2['nonce']}"


def test_single_block_replacement():
    chain = fresh_chain()
    miner = 'ZION_MINER_TEST'
    # Create and mine a block with one tx
    src = list(chain.premine_addresses.keys())[1]
    dst = 'ZION_USER_A'
    chain.create_transaction(src, dst, 12345)
    chain.mine_pending_transactions(miner)
    height = len(chain.blocks) - 1
    original_block = chain.blocks[height]
    # Prepare alternative block with higher difficulty (simulated by higher difficulty field + different tx)
    alt_tx = {
        'id': 'alt_tx_1',
        'sender': src,
        'receiver': 'ZION_USER_B',
        'amount': 11111,
        'fee': 0.0,
        'timestamp': time.time(),
        'purpose': 'Alt chain tx',
        'signature': 'ALT_SIG'
    }
    alt_block = {
        'hash': original_block['hash'][:-4] + 'ABCD',  # Different trailing part (not real PoW check here)
        'previous_hash': original_block['previous_hash'],
        'timestamp': original_block['timestamp'] + 1,
        'transactions': [alt_tx] + [t for t in original_block['transactions'] if t.get('sender') == 'MINING_REWARD'],
        'nonce': original_block['nonce'],
        'difficulty': original_block['difficulty'] + 1,  # simulate higher work
        'miner': original_block.get('miner', 'NETWORK'),
        'reward': original_block.get('reward', 50.0),
        'merkle_root': None
    }
    # Apply replacement
    replaced = chain.replace_block_at_height(height, alt_block)
    assert replaced, 'Block replacement failed'
    # Validate balances reflect alt tx instead of original
    bal_dst_old = chain.get_balance(dst)
    bal_new_receiver = chain.get_balance('ZION_USER_B')
    assert bal_new_receiver > 0, 'New receiver balance not applied'
    # Original receiver should have no funds from reverted tx (other than maybe from future operations)
    assert bal_dst_old == 0 or bal_dst_old < 1000, 'Original receiver still holds reverted funds'

if __name__ == '__main__':
    test_nonce_persistence()
    test_single_block_replacement()
    print('Persistence & Reorg tests passed.')
