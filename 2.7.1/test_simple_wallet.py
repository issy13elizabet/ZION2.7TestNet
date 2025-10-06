#!/usr/bin/env python3
"""
Simple test of ZION Wallet API using urllib
"""

import urllib.request
import urllib.parse
import json

def test_wallet_api():
    """Test wallet API using urllib"""
    print("🧪 Testing ZION Wallet API...")
    
    try:
        # Test 1: Basic API health
        print("\n1️⃣ Testing API connection...")
        try:
            response = urllib.request.urlopen("http://localhost:8001/api/zion-2-7-stats")
            data = json.loads(response.read())
            print("✅ API is running!")
            print(f"Status: {data.get('message', 'N/A')}")
        except Exception as e:
            print(f"❌ API connection failed: {e}")
            return
        
        # Test 2: List wallets
        print("\n2️⃣ Testing wallet list...")
        try:
            response = urllib.request.urlopen("http://localhost:8001/api/wallet/list")
            data = json.loads(response.read())
            print(f"✅ Wallet list: {data}")
        except Exception as e:
            print(f"❌ Wallet list failed: {e}")
        
        # Test 3: Create wallet
        print("\n3️⃣ Testing wallet creation...")
        try:
            create_data = {
                "name": "test_wallet_1",
                "password": "secure123",
                "label": "My Test Wallet"
            }
            
            req_data = json.dumps(create_data).encode('utf-8')
            req = urllib.request.Request(
                "http://localhost:8001/api/wallet/create",
                data=req_data,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            
            response = urllib.request.urlopen(req)
            data = json.loads(response.read())
            print(f"✅ Wallet created: {data}")
        except Exception as e:
            print(f"❌ Wallet creation failed: {e}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_wallet_api()