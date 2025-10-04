#!/usr/bin/env python3
"""
Test script to verify XMRig login response format
"""
import json
import socket
import time

def test_login_response():
    # Simulate XMRig login message
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

    # Connect to local pool
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect(('127.0.0.1', 3333))
        sock.send((json.dumps(login_msg) + '\n').encode())

        # Read response
        response = sock.recv(4096).decode().strip()
        print("Login Response:")
        print(response)

        # Parse and check format
        data = json.loads(response)
        print("\nParsed Response:")
        print(f"ID: {data.get('id')}")
        print(f"JSON-RPC: {data.get('jsonrpc')}")
        print(f"Has error field: {'error' in data}")
        print(f"Has result field: {'result' in data}")

        if 'result' in data:
            result = data['result']
            print(f"Result ID: {result.get('id')}")
            print(f"Result Status: {result.get('status')}")
            print(f"Has job: {'job' in result}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        sock.close()

if __name__ == "__main__":
    test_login_response()