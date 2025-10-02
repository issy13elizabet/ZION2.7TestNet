#!/usr/bin/env python3
import socket, json, time, sys, os, hashlib

HOST = os.environ.get('ZION_POOL_HOST','91.98.122.165')
PORT = int(os.environ.get('ZION_POOL_PORT','3333'))
ADDRESS = os.environ.get('ZION_MINER','Z32f72f93c095d78fc8a2fe01c0f97fd4a7f6d1bcd9b251f73b18b5625be654e84')
LOGIN_ADDR = ADDRESS


def send_recv(sock, obj, timeout=5):
    data = (json.dumps(obj) + "\n").encode()
    sock.sendall(data)
    sock.settimeout(timeout)
    buff = b''
    while True:
        part = sock.recv(4096)
        if not part:
            break
        buff += part
        if b'\n' in buff or b'}' in buff:
            break
    try:
        return json.loads(buff.decode().strip())
    except Exception:
        return buff.decode(errors='ignore')


def login():
    s = socket.socket()
    s.connect((HOST, PORT))
    req = {"id":1,"jsonrpc":"2.0","method":"login","params":{"login":ADDRESS,"pass":"","agent":"Harness/0.2","rigid":"HARNESS","algo":"rx/0"}}
    resp = send_recv(s, req)
    if not isinstance(resp, dict) or 'result' not in resp:
        print("[FAIL] Login response malformed:", resp)
        sys.exit(1)
    job = resp['result']['job']
    sess = resp['result']['id']
    token = resp['result'].get('token')
    diff = job.get('difficulty')
    print(f"[OK] Login session={sess} token={token} difficulty={diff} target={job.get('target')} job_id={job.get('job_id')}")
    return s, sess, token, diff, job


def getjob_new_socket(session_id, token):
    s = socket.socket()
    s.connect((HOST, PORT))
    req = {"id":2,"jsonrpc":"2.0","method":"getjob","params":{"token": token}}
    resp = send_recv(s, req)
    if not isinstance(resp, dict) or 'result' not in resp:
        print("[FAIL] getjob malformed:", resp)
        sys.exit(1)
    diff = resp['result'].get('difficulty')
    tgt = resp['result'].get('target')
    print(f"[OK] Cross-socket getjob difficulty={diff} target={tgt} job_id={resp['result'].get('job_id')}")
    return s, diff


def fake_submit(sock, job_id):
    # Create random fake hash (will be rejected if above target)
    fake_hash = hashlib.sha256(os.urandom(16)).hexdigest()[:64]
    req = {"id":3,"jsonrpc":"2.0","method":"submit","params":{"job_id":job_id,"result":fake_hash,"nonce":"00000000"}}
    resp = send_recv(sock, req)
    if isinstance(resp, dict):
        print(f"[INFO] Submit response: {resp}")
    else:
        print(f"[WARN] Submit raw: {resp}")


def test_invalid_token():
    """Test getjob with invalid token falls back gracefully."""
    s = socket.socket()
    s.connect((HOST, PORT))
    # Use invalid token
    req = {"id":99,"jsonrpc":"2.0","method":"getjob","params":{"token":"invalid_token_12345"}}
    resp = send_recv(s, req)
    if isinstance(resp, dict) and 'result' in resp:
        diff = resp['result'].get('difficulty')
        print(f"[TEST] Invalid token → difficulty={diff} (expect default 32)")
    else:
        print(f"[FAIL] Invalid token test: {resp}")
    s.close()

def main():
    print("=== Token-Based Difficulty Persistence Test ===")
    login_sock, session_id, token, diff_login, job_login = login()
    cross_sock, diff_cross = getjob_new_socket(session_id, token)
    if diff_cross != diff_login:
        print(f"[WARN] Difficulty mismatch: login={diff_login} cross={diff_cross}")
    else:
        print("[PASS] Difficulty persisted across sockets")
    
    # Test negative case
    print("\n=== Invalid Token Test ===")
    test_invalid_token()
    
    # Submit test
    print("\n=== Submit Test ===")
    fake_submit(login_sock, job_login['job_id'])
    
    login_sock.close()
    cross_sock.close()
    print("\n[DONE] All tests completed")

if __name__ == '__main__':
    main()
