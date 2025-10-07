#!/usr/bin/env python3
"""Lightweight integrity tests for ZION blockchain core.
Run with: python3 tests/test_chain_integrity.py
"""
import time
import os
import tempfile
from new_zion_blockchain import NewZionBlockchain


def test_mining_and_validation():
    tmp_db = os.path.join(tempfile.gettempdir(), 'zion_test_chain.db')
    if os.path.exists(tmp_db):
        os.remove(tmp_db)
    bc = NewZionBlockchain(db_file=tmp_db, enable_p2p=False, enable_rpc=False)
    start_blocks = len(bc.blocks)
    # Create dummy transaction inside premine scope
    sacred = list(bc.premine_addresses.keys())[0]
    receiver = 'ZION_TEST_ADDR_X'
    bc.create_transaction(sacred, receiver, 1000, 'Test TX')
    h = bc.mine_pending_transactions('ZION_MINER_X')
    assert h
    assert len(bc.blocks) == start_blocks + 1
    assert bc.validate_chain() is True
    audit = bc.audit_integrity()
    assert audit['chain_valid'] is True
    assert audit['supply_consistent'] is True


def test_invalid_transaction_rejected():
    tmp_db = os.path.join(tempfile.gettempdir(), 'zion_test_chain2.db')
    if os.path.exists(tmp_db):
        os.remove(tmp_db)
    bc = NewZionBlockchain(db_file=tmp_db, enable_p2p=False, enable_rpc=False)
    # Non-existing sender with zero balance
    try:
        bc.create_transaction('FAKE_ADDR', 'ZION_X', 50, 'Should fail')
        failed = False
    except Exception:
        failed = True
    assert failed, 'Expected failure for insufficient balance'


def main():
    test_mining_and_validation()
    test_invalid_transaction_rejected()
    print('All integrity tests passed.')

if __name__ == '__main__':
    main()
