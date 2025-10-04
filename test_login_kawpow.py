#!/usr/bin/env python3
"""
Test script pro KawPow login do ZION Universal Pool
Ověřuje správný formát jobu a multi-algo session handling
"""

import socket
import json
import time

def test_kawpow_login():
    """Test login s KawPow algoritmem"""
    
    # Připojení k poolu
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect(('127.0.0.1', 3333))
        print("✅ Připojeno k ZION Universal Pool (lokální)")
        
        # KawPow login request
        login_data = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "login",
            "params": {
                "login": "ZION_0123456789ABCDEF0123456789ABCDEF",
                "pass": "x",
                "algo": "kawpow"
            }
        }
        
        # Odeslání
        message = json.dumps(login_data) + '\n'
        sock.send(message.encode('utf-8'))
        print(f"📤 Odesláno: {message.strip()}")
        
        # Čekání na odpověď
        response_data = sock.recv(4096).decode('utf-8')
        print(f"📥 Přijato: {response_data.strip()}")
        
        # Parsování odpovědi
        try:
            response = json.loads(response_data.strip())
            
            # Kontroly
            if "error" in response:
                print(f"❌ Login error: {response['error']}")
                return False
                
            if "result" not in response:
                print("❌ Chybí result v odpovědi")
                return False
                
            result = response["result"]
            
            # Ověření session ID
            if "id" in result:
                print(f"✅ Session ID: {result['id']}")
            else:
                print("⚠️ Chybí session ID")
                
            # Kontrola jobu
            if "job" not in result:
                print("❌ Chybí job v result")
                return False
                
            job = result["job"]
            
            # Validace KawPow job struktury
            required_fields = ["job_id", "algo", "height", "epoch", "seed_hash", "header_hash", "target"]
            missing_fields = [field for field in required_fields if field not in job]
            
            if missing_fields:
                print(f"❌ Chybí pole v KawPow jobu: {missing_fields}")
                return False
                
            # Ověření algoritmu
            if job.get("algo") != "kawpow":
                print(f"❌ Nesprávný algoritmus: {job.get('algo')} (očekáváno: kawpow)")
                return False
                
            print("✅ Správný KawPow algoritmus")
            
            # Výpis job detailů
            print("\n📊 KawPow Job Details:")
            print(f"   Job ID: {job.get('job_id')}")
            print(f"   Algorithm: {job.get('algo')}")
            print(f"   Height: {job.get('height')}")
            print(f"   Epoch: {job.get('epoch')}")
            print(f"   Target: {job.get('target')}")
            print(f"   Seed Hash: {job.get('seed_hash')[:16]}...")
            print(f"   Header Hash: {job.get('header_hash')[:16]}...")
            
            # Kontrola epoch hodnoty
            epoch = job.get('epoch')
            height = job.get('height', 0)
            expected_epoch = (height // 7500) % 1024
            if epoch == expected_epoch:
                print(f"✅ Epoch správně vypočítán: {epoch}")
            else:
                print(f"⚠️ Epoch možná chyba: {epoch} (očekáváno: {expected_epoch})")
                
            return True
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parse error: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False
    finally:
        sock.close()

def test_randomx_fallback():
    """Test že bez algo parametru se použije RandomX"""
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect(('127.0.0.1', 3333))
        print("\n✅ Připojeno k ZION Universal Pool (lokální fallback test)")
        
        # Login bez algo parametru
        login_data = {
            "id": 2,
            "jsonrpc": "2.0",
            "method": "login",
            "params": {
                "login": "ZION_FEDCBA9876543210FEDCBA9876543210",
                "pass": "x"
            }
        }
        
        message = json.dumps(login_data) + '\n'
        sock.send(message.encode('utf-8'))
        print(f"📤 Odesláno (bez algo): {message.strip()}")
        
        response_data = sock.recv(4096).decode('utf-8')
        response = json.loads(response_data.strip())
        
        if "result" in response and "job" in response["result"]:
            job = response["result"]["job"]
            algo = job.get("algo", "neznámý")
            print(f"✅ Fallback algoritmus: {algo}")
            
            if algo in ["rx/0", "randomx"]:
                print("✅ Správný RandomX fallback")
                return True
            else:
                print(f"❌ Neočekávaný fallback: {algo}")
                return False
        else:
            print("❌ Chybná odpověď při fallback testu")
            return False
            
    except Exception as e:
        print(f"❌ Fallback test error: {e}")
        return False
    finally:
        sock.close()

if __name__ == "__main__":
    print("🚀 ZION Universal Pool - KawPow Login Test")
    print("=" * 50)
    
    # Test 1: KawPow login
    kawpow_ok = test_kawpow_login()
    
    # Test 2: RandomX fallback
    randomx_ok = test_randomx_fallback()
    
    print("\n" + "=" * 50)
    print("📊 Výsledky testů:")
    print(f"   KawPow login: {'✅ OK' if kawpow_ok else '❌ FAIL'}")
    print(f"   RandomX fallback: {'✅ OK' if randomx_ok else '❌ FAIL'}")
    
    if kawpow_ok and randomx_ok:
        print("\n🎉 Všechny testy prošly! Multi-algo pool funguje správně.")
        exit(0)
    else:
        print("\n💥 Některé testy selhaly. Zkontroluj pool konfiguraci.")
        exit(1)