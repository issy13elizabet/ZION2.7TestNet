#!/usr/bin/env python3
"""Security feature tests for ZION 2.7.4 hardened components.
Run with: PYTHONPATH=. python3 tests/test_security_features.py
"""
import os, tempfile, time, json, urllib.request, urllib.error, threading
from new_zion_blockchain import NewZionBlockchain
from zion_rpc_server import ZIONRPCServer


def start_rpc(blockchain, token, rate_per_min=10, burst=5):
    rpc = ZIONRPCServer(blockchain, auth_token=token, rate_limit_per_minute=rate_per_min, burst_limit=burst, burst_window_seconds=1, port=0)
    rpc.start()
    # Wait briefly for server
    time.sleep(0.2)
    port = rpc.http_server.server_address[1]
    return rpc, port

def http_get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req) as resp:
        return resp.status, resp.read().decode()

def http_post(url, payload, headers=None):
    data = json.dumps(payload).encode()
    h = {'Content-Type':'application/json'}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=data, headers=h, method='POST')
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()

def test_rpc_auth_and_rate_limit():
    tmp_db = os.path.join(tempfile.gettempdir(),'zion_sec_test.db')
    if os.path.exists(tmp_db):
        os.remove(tmp_db)
    bc = NewZionBlockchain(db_file=tmp_db, enable_p2p=False, enable_rpc=False)
    token = 'SECURITYTOKEN'
    rpc, port = start_rpc(bc, token, rate_per_min=5, burst=3)
    base = f'http://127.0.0.1:{port}'

    # Unauthenticated POST should 401
    status, body = http_post(base, {'method':'getblockcount','params':[]})
    assert status == 401, f'Expected 401 got {status}'

    # Authenticated POST should succeed
    status, body = http_post(base, {'method':'getblockcount','params':[]}, headers={'X-ZION-AUTH':token})
    # Expect at least genesis block present
    assert status == 200 and body['result'] >= 1

    # Hit burst limit (burst=3). We already made 1 success; allow 2 more then next should 429.
    ok_extra = 0
    for i in range(2):
        s, _ = http_post(base, {'method':'getblockcount','params':[]}, headers={'X-ZION-AUTH':token})
        if s == 200:
            ok_extra += 1
    assert ok_extra == 2
    # Next should exceed burst (within 1s window)
    s, _ = http_post(base, {'method':'getblockcount','params':[]}, headers={'X-ZION-AUTH':token})
    assert s == 429, f'Expected 429 got {s}'
    rpc.stop()


def test_timestamp_rejection():
    tmp_db = os.path.join(tempfile.gettempdir(),'zion_sec_test2.db')
    if os.path.exists(tmp_db):
        os.remove(tmp_db)
    bc = NewZionBlockchain(db_file=tmp_db, enable_p2p=False, enable_rpc=False)
    # Craft external block with too far future timestamp
    future = time.time() + bc.max_future_drift + 500
    fake_block = {
        'height': len(bc.blocks),
        'hash':'0'*64,
        'previous_hash': bc.blocks[-1]['hash'],
        'timestamp': future,
        'transactions': [],
        'nonce': 0,
        'difficulty': bc.mining_difficulty
    }
    valid = bc.validate_block_timestamp_external(fake_block)
    assert not valid, 'Future timestamp should be invalid'
    assert bc.invalid_timestamps >= 1


def main():
    test_rpc_auth_and_rate_limit()
    test_timestamp_rejection()
    print('Security feature tests passed.')

if __name__ == '__main__':
    main()
