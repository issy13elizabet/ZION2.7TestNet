#!/usr/bin/env python3
"""Run ZION 2.7 node with optional pool & P2P skeleton.

Usage:
  python run_node.py --pool            # start stratum pool on 3333
  python run_node.py --p2p --peers host:port,host2:port2
  python run_node.py --interval 5      # status print interval
"""
from core.blockchain import Blockchain
import json, time, argparse, threading

def start_pool(chain):
    from pool.stratum_pool import MinimalStratumPool
    pool = MinimalStratumPool(chain=chain)
    t = threading.Thread(target=pool.start, daemon=True)
    t.start()
    return pool

def start_p2p(chain, peers):
    from network.p2p import P2PNode
    p2p = P2PNode(chain=chain)
    p2p.start()
    for peer in peers:
        try:
            host, port = peer.split(':')
            p2p.connect(host, int(port))
        except Exception:
            pass
    return p2p

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ZION 2.7 testnet node')
    parser.add_argument('--pool', action='store_true', help='Start stratum pool')
    parser.add_argument('--p2p', action='store_true', help='Start P2P skeleton')
    parser.add_argument('--peers', default='', help='Comma-separated bootstrap peers host:port')
    parser.add_argument('--interval', type=int, default=10, help='Status print interval seconds')
    args = parser.parse_args()

    chain = Blockchain()
    print("ZION 2.7 TestNet node started")
    pool = None
    p2p = None
    if args.pool:
        pool = start_pool(chain)
        print("Stratum pool enabled (port 3333)")
    if args.p2p:
        peers = [p for p in args.peers.split(',') if p.strip()] if args.peers else []
        p2p = start_p2p(chain, peers)
        print(f"P2P skeleton enabled (listening, peers={len(peers)})")

    try:
        while True:
            info = chain.info()
            if p2p:
                info['p2p'] = True
                info['peers'] = len([peer for peer in p2p.peers if peer.alive])
            if pool:
                info['pool'] = {
                    'accepted': pool.accepted,
                    'rejected': pool.rejected,
                    'active_jobs': len(pool.active_jobs)
                }
            print(json.dumps(info, indent=2))
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Shutdown")
