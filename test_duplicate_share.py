#!/usr/bin/env python3
"""
Test duplicitního share proti ZION poolu.
Postup:
1. Odeslat login.
2. Přijmout job.
3. Vygenerovat falešný (nonce,result) - oboje hex (nonce 8+ znaků, result 32+ znaků).
4. Poslat submit poprvé -> očekává se status OK.
5. Poslat submit podruhé (stejné job_id, nonce, result) -> očekává se error code -4 Duplicate share.
"""
import socket, json, time, os, sys, secrets

HOST = os.environ.get("ZION_POOL_HOST", "127.0.0.1")
PORT = int(os.environ.get("ZION_POOL_PORT", "3333"))
ADDRESS = os.environ.get("ZION_TEST_ADDRESS", "ZION_C806923E65C3640CFFB3DA786A0BF579")


def recv_line(sock):
    buff = b""
    while True:
        ch = sock.recv(1)
        if not ch:
            break
        buff += ch
        if ch == b"\n":
            break
    return buff.decode().strip()

def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(10)
    s.connect((HOST, PORT))

    login_msg = {
        "id": 1,
        "jsonrpc": "2.0",
        "method": "login",
        "params": {"login": ADDRESS, "pass": "x", "agent": "dup-test/1.0"}
    }
    s.send((json.dumps(login_msg) + "\n").encode())
    line = recv_line(s)
    print("LOGIN RESP:", line)
    data = json.loads(line)
    job = data["result"]["job"]
    job_id = job["job_id"]
    session_id = data["result"]["id"]  # Potřebujeme session ID

    # Falešný nonce & result (hex) – délky v povoleném rozsahu
    nonce = secrets.token_hex(4)  # 8 znaků
    result = secrets.token_hex(16)  # 32 znaků

    submit = {
        "id": 2,
        "jsonrpc": "2.0",
        "method": "submit",
        "params": {
            "id": session_id,  # Session ID je povinné
            "job_id": job_id,
            "nonce": nonce,
            "result": result
        }
    }
    # První pokus
    s.send((json.dumps(submit) + "\n").encode())
    first = recv_line(s)
    print("FIRST SUBMIT:", first)

    # Druhý pokus (stejné hodnoty + stejné ID pro duplicitní test)
    submit["id"] = 3  # Nové request ID
    s.send((json.dumps(submit) + "\n").encode())
    
    # Přečti všechny zprávy po druhém submitu
    second_messages = []
    try:
        # Může přijít více zpráv, čteme všechny dostupné
        s.settimeout(2)  # Krátký timeout
        while True:
            line = recv_line(s)
            if line:
                second_messages.append(line)
                print(f"SECOND MSG: {line}")
            else:
                break
    except socket.timeout:
        pass  # Normální konec čtení
    
    print("SECOND SUBMIT MESSAGES:", second_messages)

    try:
        first_obj = json.loads(first)
        ok_first = first_obj.get("result", {}).get("status") == "OK"
        
        # Hledej error response mezi druhými zprávami
        dup_second = False
        for msg in second_messages:
            try:
                obj = json.loads(msg)
                if "error" in obj and obj["error"].get("code") == -4:
                    dup_second = True
                    break
            except:
                pass
        
        print(f"ASSERT first OK: {ok_first}, second duplicate error: {dup_second}")
        
        if ok_first and dup_second:
            print("✅ DUPLICATE TEST PASSED!")
        else:
            print("❌ DUPLICATE TEST FAILED!")
            
    except Exception as e:
        print("Parse error:", e)

    s.close()

if __name__ == "__main__":
    main()
