#!/usr/bin/env python3
"""Test coinbase maturity and tx relay in mempool."""
import time, hashlib, os, sys

# Add project root to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.blockchain import Blockchain, Consensus, Tx
from network.p2p import P2PNode

COINBASE_SPEND_FEE = 10


def mine_one(chain: Blockchain):
    tmpl = chain.create_block_template(chain.genesis_address)
    mined = {
        'height': tmpl['height'],
        'prev_hash': tmpl['prev_hash'],
        'timestamp': int(time.time()),
        'merkle_root': tmpl['merkle_root'],
        'difficulty': tmpl['difficulty'],
        'nonce': 0,
        'txs': tmpl['txs']
    }
    assert chain.submit_block(mined)


def test_coinbase_maturity_and_relay():
    chain1 = Blockchain()
    chain2 = Blockchain()

    # Mine maturity-1 blocks so genesis coinbase still immature
    for _ in range(Consensus.COINBASE_MATURITY - 1):
        mine_one(chain1)
        mine_one(chain2)

    # Attempt to spend coinbase from chain1 (should fail add_tx)
    immature_spend = Tx.create(
        inputs=[{'txid': 'genesis', 'vout': 0}],
        outputs=[{'address': chain1.genesis_address, 'amount': Consensus.reward(0) - COINBASE_SPEND_FEE}],
        fee=COINBASE_SPEND_FEE
    )
    added = chain1.add_tx(immature_spend)
    assert not added, 'Immature coinbase spend should be rejected'

    # Mine one more block to reach maturity
    mine_one(chain1)
    mine_one(chain2)

    mature_spend = Tx.create(
        inputs=[{'txid': 'genesis', 'vout': 0}],
        outputs=[{'address': chain1.genesis_address, 'amount': Consensus.reward(0) - COINBASE_SPEND_FEE}],
        fee=COINBASE_SPEND_FEE
    )
    assert chain1.add_tx(mature_spend), 'Mature coinbase spend should be accepted'

    # Start P2P nodes on different ports
    p1 = P2PNode(port=29901, chain=chain1)
    p2 = P2PNode(port=29902, chain=chain2)
    p1.start(); p2.start()
    # Connect p2 -> p1
    p2.connect('127.0.0.1', 29901)

    # Broadcast tx
    p1.broadcast_tx({
        'txid': mature_spend.txid,
        'inputs': mature_spend.inputs,
        'outputs': mature_spend.outputs,
        'fee': mature_spend.fee
    })

    # Wait for relay
    deadline = time.time() + 3
    while time.time() < deadline:
        if mature_spend.txid in p2.chain.mempool:
            break
        time.sleep(0.1)

    assert mature_spend.txid in p2.chain.mempool, 'Relay failed: tx not in second mempool'

if __name__ == '__main__':
    test_coinbase_maturity_and_relay()
    print('OK maturity & relay')
