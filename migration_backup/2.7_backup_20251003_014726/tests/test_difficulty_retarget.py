#!/usr/bin/env python3
"""Difficulty retarget tests for ZION 2.7 chain.
Fast block production should raise difficulty; slow production should reduce (but not below 1).
"""
import time, os, sys

# Add project root to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.blockchain import Blockchain, Consensus

def mine_block_with_timestamp(chain: Blockchain, ts: int):
    tmpl = chain.create_block_template(chain.genesis_address)
    mined = {
        'height': tmpl['height'],
        'prev_hash': tmpl['prev_hash'],
        'timestamp': ts,
        'merkle_root': tmpl['merkle_root'],
        'difficulty': tmpl['difficulty'],
        'nonce': 0,
        'txs': tmpl['txs']
    }
    assert chain.submit_block(mined), "Block submit failed in test"


def test_difficulty_fast_then_slow():
    chain = Blockchain()
    start_diff = chain.current_difficulty
    base_ts = int(time.time())
    # Produce 12 fast blocks (1s apart) to trigger increase
    for i in range(12):
        mine_block_with_timestamp(chain, base_ts + i)
    after_fast = chain.current_difficulty
    assert after_fast > start_diff, f"Difficulty should increase (got {after_fast} from {start_diff})"

    # Produce 12 slow blocks (600s apart) to trigger decrease (clamped by factor 4)
    last_ts = base_ts + 12  # continue timeline
    for i in range(12):
        mine_block_with_timestamp(chain, last_ts + i * 600)
    after_slow = chain.current_difficulty
    assert after_slow < after_fast, f"Difficulty should decrease after slow blocks (from {after_fast} to {after_slow})"
    assert after_slow >= Consensus.MIN_DIFF

if __name__ == '__main__':
    test_difficulty_fast_then_slow()
    print('OK difficulty retarget')
