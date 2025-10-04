import socket, json, time

HOST = '127.0.0.1'
PORT = 3333

def send_recv(sock, obj):
    data = (json.dumps(obj) + '\n').encode()
    sock.sendall(data)
    # read multiple potential lines with small timeout
    sock.settimeout(0.5)
    buff = b''
    try:
        while True:
            part = sock.recv(4096)
            if not part:
                break
            buff += part
            if len(part) < 4096:
                break
    except Exception:
        pass
    return buff.decode(errors='ignore')

with socket.create_connection((HOST, PORT)) as s:
    # 1) subscribe
    sub_resp = send_recv(s, {"id":1, "method":"mining.subscribe", "params":[]})
    print("SUBSCRIBE RESP:\n", sub_resp)
    # 2) authorize (wallet placeholder)
    auth_resp = send_recv(s, {"id":2, "method":"mining.authorize", "params":["ZION_DEADBEEF0123456789ABCDEF012345"]})
    print("AUTHORIZE RESP:\n", auth_resp)
    # 3) Simulace submit (fake values)
    # Extract job id from notify if present
    job_id = None
    for line in auth_resp.splitlines():
        try:
            j = json.loads(line)
            if j.get('method') == 'mining.notify':
                job_id = j['params'][0]
        except Exception:
            pass
    if job_id:
        submit_resp = send_recv(s, {"id":3, "method":"mining.submit", "params":["worker1", job_id, "abcdef01", "ff"*16, "aa"*32]})
        print("SUBMIT RESP:\n", submit_resp)
    else:
        print("No job id extracted from authorize response.")
