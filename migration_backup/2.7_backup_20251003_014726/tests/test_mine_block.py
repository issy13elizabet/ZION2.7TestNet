#!/usr/bin/env python3
"""Quick test harness: simulate share finding for low-diff chain."""
import time, json, hashlib, random, os, sys

# Add project root to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.blockchain import Blockchain

chain = Blockchain()
print("Initial height:", chain.height())

# Create template
TEMPLATE_MINER = chain.genesis_address

attempts = 0
max_attempts = 50000

while attempts < max_attempts:
    tmpl = chain.create_block_template(TEMPLATE_MINER)
    # brute force fake miner using full target integer
    nonce = random.randint(0, 1_000_000)
    h = hashlib.sha256((tmpl['prev_hash'] + str(nonce)).encode()).hexdigest()
    attempts += 1
    if int(h, 16) < tmpl['target']:
        mined = {
            'height': tmpl['height'],
            'prev_hash': tmpl['prev_hash'],
            'timestamp': int(time.time()),
            'merkle_root': tmpl['merkle_root'],
            'difficulty': tmpl['difficulty'],
            'nonce': nonce,
            'txs': tmpl['txs']
        }
        ok = chain.submit_block(mined)
        print(f"Submit result={ok} nonce={nonce} hash={chain.last_block().hash[:16]} attempts={attempts}")
        if ok:
            break

print("Final height:", chain.height())
print(json.dumps(chain.info(), indent=2))
