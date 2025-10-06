#!/usr/bin/env python3
"""
Remote verification script for login response format
"""
import json
import socket
import time

def test_remote_login():
    # Connect to remote pool
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10)
    try:
        sock.connect(('91.98.122.165', 3333))

        login_msg = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "login",
            "params": {
                "login": "ZION_C806923E65C3640CFFB3DA786A0BF579",
                "pass": "x",
                "agent": "XMRig/6.20.0"
            }
        }

        sock.send((json.dumps(login_msg) + '\n').encode())
        response = sock.recv(4096).decode().strip()

        print("Remote Login Response:")
        print(response)

        data = json.loads(response)
        print(f"\nHas error field: {'error' in data}")
        print(f"Has result field: {'result' in data}")

        if 'error' in data:
            print(f"Error code: {data['error'].get('code')}")
            print(f"Error message: {data['error'].get('message')}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        sock.close()

if __name__ == "__main__":
    test_remote_login()