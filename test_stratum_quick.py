#!/usr/bin/env python3
import socket
import json
import time

def test_stratum_connection():
    """Rychlý test Stratum připojení se zachytáváním odpovědí"""
    host = '91.98.122.165'
    port = 3333
    
    print(f"🔗 Připojuji se k {host}:{port}")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        sock.connect((host, port))
        
        # Subscribe
        subscribe_msg = json.dumps({
            "jsonrpc": "2.0",
            "method": "mining.subscribe", 
            "id": 1,
            "params": ["SRBMiner-MULTI/2.0.0", "", "", ""]
        }) + '\n'
        
        print(f"📤 Posílám subscribe: {subscribe_msg.strip()}")
        sock.send(subscribe_msg.encode())
        
        # Čtení odpovědi
        response = sock.recv(4096).decode()
        print(f"📥 Subscribe response: {response.strip()}")
        
        # Authorize
        auth_msg = json.dumps({
            "jsonrpc": "2.0", 
            "method": "mining.authorize",
            "id": 2,
            "params": ["ZION_test", "x"]
        }) + '\n'
        
        print(f"📤 Posílám authorize: {auth_msg.strip()}")
        sock.send(auth_msg.encode())
        
        # Čtení odpovědí (může být více řádků)
        time.sleep(0.5)
        response = sock.recv(4096).decode()
        print(f"📥 Authorize+notify response: {response.strip()}")
        
        # Ponechat otevřené pro další zprávy
        print("⏳ Čekám na další zprávy (3s)...")
        sock.settimeout(3)
        try:
            more_data = sock.recv(4096).decode()
            if more_data:
                print(f"📥 Další data: {more_data.strip()}")
        except socket.timeout:
            print("⏰ Timeout - žádné další zprávy")
            
        sock.close()
        print("✅ Test dokončen")
        
    except Exception as e:
        print(f"❌ Chyba: {e}")

if __name__ == "__main__":
    test_stratum_connection()