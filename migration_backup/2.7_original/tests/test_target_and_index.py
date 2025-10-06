#!/usr/bin/env python3
"""Tests for hash index and target logic."""
import hashlib, random, time, os, sys

# Add project root to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.blockchain import Blockchain, Consensus, Tx

def test_hash_index_and_target():
    chain = Blockchain()
    start_height = chain.height()
    # Mine 2 blocks artificially by brute force (very easy diff early)
    for _ in range(2):
        tmpl = chain.create_block_template(chain.genesis_address)
        target = tmpl['target']
        nonce = 0
        while True:
            h = hashlib.sha256((tmpl['prev_hash'] + str(nonce)).encode()).hexdigest()
            if int(h,16) < target:
                mined = {
                    'height': tmpl['height'],
                    'prev_hash': tmpl['prev_hash'],
                    'timestamp': int(time.time()),
                    'merkle_root': tmpl['merkle_root'],
                    'difficulty': tmpl['difficulty'],
                    'nonce': nonce,
                    'txs': tmpl['txs']
                }
                assert chain.submit_block(mined)
                break
            nonce += 1
    # Validate index
    last = chain.last_block()
    assert chain.get_block_by_hash(last.hash).height == last.height
    assert chain.block_exists(last.hash)
    # Target relationship
    assert Consensus.difficulty_to_target(chain.current_difficulty) > 0

if __name__ == '__main__':
    test_hash_index_and_target()
    print('OK')
