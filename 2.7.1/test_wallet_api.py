#!/usr/bin/env python3
"""
Test ZION Wallet API endpoints
"""

import requests
import json

API_BASE = "http://localhost:8001"

def test_wallet_endpoints():
    """Test wallet API endpoints"""
    print("🧪 Testing ZION Wallet API...")
    
    try:
        # Test 1: List wallets
        print("\n1️⃣ Testing wallet list...")
        response = requests.get(f"{API_BASE}/api/wallet/list")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
        # Test 2: Create wallet
        print("\n2️⃣ Testing wallet creation...")
        create_data = {
            "name": "test_wallet",
            "password": "secure123",
            "label": "Test Wallet"
        }
        response = requests.post(f"{API_BASE}/api/wallet/create", json=create_data)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
        # Test 3: Unlock wallet
        print("\n3️⃣ Testing wallet unlock...")
        unlock_data = {
            "name": "test_wallet",
            "password": "secure123"
        }
        response = requests.post(f"{API_BASE}/api/wallet/unlock", json=unlock_data)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
        # Test 4: Get wallet info
        print("\n4️⃣ Testing wallet info...")
        response = requests.get(f"{API_BASE}/api/wallet/test_wallet")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
        # Test 5: Create address
        print("\n5️⃣ Testing address creation...")
        address_data = {
            "wallet_name": "test_wallet",
            "label": "My Address"
        }
        response = requests.post(f"{API_BASE}/api/wallet/address/create", json=address_data)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
    except Exception as e:
        print(f"❌ Error testing wallet API: {e}")

if __name__ == "__main__":
    test_wallet_endpoints()