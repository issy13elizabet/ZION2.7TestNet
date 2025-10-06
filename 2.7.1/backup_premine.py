#!/usr/bin/env python3
"""
ZION 2.7.1 - Pre-mine Address Backup Tool
⚠️  BEZPEČNOSTNÍ UPOZORNĚNÍ ⚠️
=====================================
TENTO SOUBOR OBSAHUJE POUZE PLACEHOLDER ADRESY!
Skutečné pre-mine adresy a private keys jsou uloženy v externí zabezpečené záloze.
NIKDY NEUKLÁDEJTE SKUTEČNÉ PRIVATE KEYS DO GIT REPOZITÁŘE!

Bezpečné zálohování a export pre-mine adres s jejich private klíči
"""

import sys
import os
import json
import hashlib
import getpass
import datetime
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.real_blockchain import ZionRealBlockchain

class PremineBackupTool:
    """Nástroj pro bezpečné zálohování pre-mine adres"""
    
    def __init__(self):
        self.blockchain = ZionRealBlockchain()
        
    def get_premine_addresses(self):
        """Získej všechny pre-mine adresy a jejich účely"""
        # BEZPEČNOSTNÍ POZNÁMKA: Skutečné adresy jsou uloženy v externí zabezpečené záloze
        # Tyto hodnoty jsou pouze pro demonstraci a testování
        return {
            # Mining operators (2B ZION each) - PLACEHOLDER ADDRESSES
            '[REDACTED_SACRED_MINING_ADDRESS]': {
                'purpose': 'SACRED Mining Operator',
                'amount': 2_000_000_000,
                'type': 'mining',
                'consciousness': 'SACRED',
                'private_key': '[EXTERNAL_BACKUP_ONLY]',
                'mnemonic': '[EXTERNAL_BACKUP_ONLY]'
            },
            '[REDACTED_QUANTUM_MINING_ADDRESS]': {
                'purpose': 'QUANTUM Mining Operator', 
                'amount': 2_000_000_000,
                'type': 'mining',
                'consciousness': 'QUANTUM',
                'private_key': '[EXTERNAL_BACKUP_ONLY]',
                'mnemonic': '[EXTERNAL_BACKUP_ONLY]'
            },
            '[REDACTED_COSMIC_MINING_ADDRESS]': {
                'purpose': 'COSMIC Mining Operator',
                'amount': 2_000_000_000,
                'type': 'mining',
                'consciousness': 'COSMIC',
                'private_key': '[EXTERNAL_BACKUP_ONLY]',
                'mnemonic': '[EXTERNAL_BACKUP_ONLY]'
            },
            '[REDACTED_ENLIGHTENED_MINING_ADDRESS]': {
                'purpose': 'ENLIGHTENED Mining Operator',
                'amount': 2_000_000_000,
                'type': 'mining',
                'consciousness': 'ENLIGHTENED',
                'private_key': '[EXTERNAL_BACKUP_ONLY]',
                'mnemonic': '[EXTERNAL_BACKUP_ONLY]'
            },
            '[REDACTED_TRANSCENDENT_MINING_ADDRESS]': {
                'purpose': 'TRANSCENDENT Mining Operator',
                'amount': 2_000_000_000,
                'type': 'mining',
                'consciousness': 'TRANSCENDENT',
                'private_key': '[EXTERNAL_BACKUP_ONLY]',
                'mnemonic': '[EXTERNAL_BACKUP_ONLY]'
            },
            
            # Special funds (1B ZION each) - PLACEHOLDER ADDRESSES
            '[REDACTED_DEV_TEAM_ADDRESS]': {
                'purpose': 'Development Team Fund',
                'amount': 1_000_000_000,
                'type': 'development',
                'consciousness': 'N/A',
                'private_key': '[EXTERNAL_BACKUP_ONLY]',
                'mnemonic': '[EXTERNAL_BACKUP_ONLY]'
            },
            '[REDACTED_SITA_NETWORK_ADDRESS]': {
                'purpose': 'Network Infrastructure (SITA)',
                'amount': 1_000_000_000,
                'type': 'infrastructure',
                'consciousness': 'N/A',
                'private_key': '[EXTERNAL_BACKUP_ONLY]',
                'mnemonic': '[EXTERNAL_BACKUP_ONLY]'
            },
            '[REDACTED_CHILDREN_FUND_ADDRESS]': {
                'purpose': 'Children Future Fund',
                'amount': 1_000_000_000,
                'type': 'social',
                'consciousness': 'N/A',
                'private_key': '[EXTERNAL_BACKUP_ONLY]',
                'mnemonic': '[EXTERNAL_BACKUP_ONLY]'
            },
            
            # Network Administrator - PLACEHOLDER ADDRESS
            '[REDACTED_NETWORK_ADMIN_ADDRESS]': {
                'purpose': 'Network Administrator (Mainnet Upgrade)',
                'amount': 1_000_000_000,
                'type': 'network_admin',
                'consciousness': 'MAITREYA_BUDDHA',
                'private_key': '[EXTERNAL_BACKUP_ONLY]',
                'mnemonic': '[EXTERNAL_BACKUP_ONLY]'
            },
            
            # Genesis reward - PLACEHOLDER ADDRESS
            '[REDACTED_GENESIS_REWARD_ADDRESS]': {
                'purpose': 'Genesis Reward',
                'amount': 342_857_142,
                'type': 'genesis',
                'consciousness': 'ON_THE_STAR',
                'private_key': '[EXTERNAL_BACKUP_ONLY]',
                'mnemonic': '[EXTERNAL_BACKUP_ONLY]'
            }
        }
    
    def derive_key_from_password(self, password: str, salt: bytes) -> bytes:
        """Odvození encryption klíče z hesla"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def encrypt_data(self, data: str, password: str) -> dict:
        """Šifrování dat pomocí hesla"""
        salt = os.urandom(16)
        key = self.derive_key_from_password(password, salt)
        f = Fernet(key)
        
        encrypted_data = f.encrypt(data.encode())
        
        return {
            'encrypted_data': base64.urlsafe_b64encode(encrypted_data).decode(),
            'salt': base64.urlsafe_b64encode(salt).decode(),
            'algorithm': 'Fernet/PBKDF2HMAC-SHA256',
            'iterations': 100000
        }
    
    def decrypt_data(self, encrypted_info: dict, password: str) -> str:
        """Dešifrování dat pomocí hesla"""
        try:
            salt = base64.urlsafe_b64decode(encrypted_info['salt'])
            encrypted_data = base64.urlsafe_b64decode(encrypted_info['encrypted_data'])
            
            key = self.derive_key_from_password(password, salt)
            f = Fernet(key)
            
            decrypted_data = f.decrypt(encrypted_data)
            return decrypted_data.decode()
        except Exception as e:
            raise ValueError(f"Dešifrování selhalo: {e}")
    
    def create_backup(self, password: str = None) -> str:
        """Vytvoř šifrovanou zálohu všech pre-mine adres"""
        if not password:
            password = getpass.getpass("🔐 Zadej silné heslo pro šifrování zálohy: ")
            confirm = getpass.getpass("🔐 Potvrď heslo: ")
            if password != confirm:
                raise ValueError("Hesla se neshodují!")
        
        addresses = self.get_premine_addresses()
        
        # Přidej aktuální balances
        for address in addresses:
            balance = self.blockchain.get_balance(address)
            addresses[address]['current_balance'] = balance
            addresses[address]['current_balance_zion'] = balance / 1_000_000
        
        backup_data = {
            'version': 'ZION_2.7.1',
            'created': str(datetime.datetime.utcnow()),
            'total_addresses': len(addresses),
            'total_premine': sum(addr['amount'] for addr in addresses.values()),
            'addresses': addresses,
            'security_note': 'KEEP THIS BACKUP SECURE! Contains private keys for ZION pre-mine addresses.',
            'recovery_instructions': 'Use ZION backup tool to decrypt this file with your password.'
        }
        
        # Šifrování
        json_data = json.dumps(backup_data, indent=2)
        encrypted_backup = self.encrypt_data(json_data, password)
        
        # Přidej metadata
        final_backup = {
            'backup_type': 'ZION_PREMINE_ADDRESSES',
            'version': '2.7.1',
            'created': str(datetime.datetime.utcnow()),
            'encrypted': True,
            **encrypted_backup
        }
        
        return json.dumps(final_backup, indent=2)
    
    def verify_backup(self, backup_file: str, password: str) -> bool:
        """Ověř integritu zálohy"""
        try:
            with open(backup_file, 'r') as f:
                backup_data = json.load(f)
            
            if backup_data.get('backup_type') != 'ZION_PREMINE_ADDRESSES':
                print("❌ Neplatný typ zálohy")
                return False
            
            # Dešifrování
            decrypted_data = self.decrypt_data(backup_data, password)
            restored_data = json.loads(decrypted_data)
            
            # Ověření adres
            addresses = restored_data.get('addresses', {})
            expected_addresses = self.get_premine_addresses()
            
            if len(addresses) != len(expected_addresses):
                print(f"❌ Počet adres se neshoduje: {len(addresses)} vs {len(expected_addresses)}")
                return False
            
            for address in expected_addresses:
                if address not in addresses:
                    print(f"❌ Chybí adresa: {address}")
                    return False
            
            print("✅ Záloha je platná a kompletní")
            return True
            
        except Exception as e:
            print(f"❌ Chyba při ověřování zálohy: {e}")
            return False
    
    def export_paper_wallet(self) -> str:
        """Export do paper wallet formátu"""
        addresses = self.get_premine_addresses()
        
        paper_wallet = []
        paper_wallet.append("🏛️  ZION 2.7.1 PAPER WALLET - PRE-MINE ADDRESSES")
        paper_wallet.append("=" * 80)
        paper_wallet.append(f"📅 Created: {datetime.datetime.utcnow()}")
        paper_wallet.append(f"🎯 Total addresses: {len(addresses)}")
        paper_wallet.append(f"💰 Total ZION: {sum(addr['amount'] for addr in addresses.values()):,}")
        paper_wallet.append("")
        paper_wallet.append("⚠️  CRITICAL SECURITY WARNING:")
        paper_wallet.append("   • Store this document in a fireproof safe")
        paper_wallet.append("   • Make multiple copies in different locations")
        paper_wallet.append("   • Never share private keys or mnemonics")
        paper_wallet.append("   • Use only for emergency recovery")
        paper_wallet.append("")
        
        for i, (address, info) in enumerate(addresses.items(), 1):
            paper_wallet.append(f"📍 ADDRESS #{i}: {info['purpose']}")
            paper_wallet.append("-" * 60)
            paper_wallet.append(f"   Public Address:")
            paper_wallet.append(f"   {address}")
            paper_wallet.append(f"   ")
            paper_wallet.append(f"   Private Key:")
            paper_wallet.append(f"   {info['private_key']}")
            paper_wallet.append(f"   ")
            paper_wallet.append(f"   Mnemonic (12 words):")
            paper_wallet.append(f"   {info['mnemonic']}")
            paper_wallet.append(f"   ")
            paper_wallet.append(f"   Details:")
            paper_wallet.append(f"   • Amount: {info['amount']:,} ZION")
            paper_wallet.append(f"   • Type: {info['type']}")
            if info['consciousness'] != 'N/A':
                paper_wallet.append(f"   • Consciousness: {info['consciousness']}")
            paper_wallet.append("")
        
        paper_wallet.append("🔐 SECURITY REMINDERS:")
        paper_wallet.append("   • Test recovery process before relying on this backup")
        paper_wallet.append("   • Update backup if any keys change")
        paper_wallet.append("   • Use multi-signature for additional security")
        paper_wallet.append("   • Regular security audits recommended")
        paper_wallet.append("")
        paper_wallet.append("For recovery support: support@zion-blockchain.org")
        
        return "\n".join(paper_wallet)

def main():
    """Hlavní funkce pro backup nástroj"""    
    tool = PremineBackupTool()
    
    print("🔐 ZION 2.7.1 - Pre-mine Address Backup Tool")
    print("=" * 60)
    
    while True:
        print("\n📋 Dostupné akce:")
        print("1. 💾 Vytvořit šifrovanou zálohu")
        print("2. 📄 Export paper wallet")
        print("3. ✅ Ověřit existující zálohu")
        print("4. 📊 Zobrazit přehled adres")
        print("5. 🚪 Ukončit")
        
        choice = input("\n🎯 Vyber akci (1-5): ").strip()
        
        if choice == '1':
            try:
                backup_data = tool.create_backup()
                filename = f"zion_premine_backup_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
                
                with open(filename, 'w') as f:
                    f.write(backup_data)
                
                print(f"✅ Šifrovaná záloha vytvořena: {filename}")
                print("⚠️  Uložte soubor a heslo na bezpečná místa!")
                
            except Exception as e:
                print(f"❌ Chyba při vytváření zálohy: {e}")
        
        elif choice == '2':
            paper_wallet = tool.export_paper_wallet()
            filename = f"zion_paper_wallet_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt"
            
            with open(filename, 'w') as f:
                f.write(paper_wallet)
            
            print(f"✅ Paper wallet vytvořen: {filename}")
            print("⚠️  POZOR: Obsahuje nešifrované private klíče!")
            print("   Vytiskněte a uložte na bezpečném místě.")
        
        elif choice == '3':
            backup_file = input("📂 Cesta k záloze: ").strip()
            if os.path.exists(backup_file):
                password = getpass.getpass("🔐 Heslo pro dešifrování: ")
                tool.verify_backup(backup_file, password)
            else:
                print("❌ Soubor nenalezen")
        
        elif choice == '4':
            addresses = tool.get_premine_addresses()
            print(f"\n📊 Přehled {len(addresses)} pre-mine adres:")
            print("-" * 60)
            
            total_amount = 0
            for address, info in addresses.items():
                balance = tool.blockchain.get_balance(address)
                balance_zion = balance / 1_000_000
                total_amount += info['amount']
                
                print(f"📍 {info['purpose']}")
                print(f"   Address: {address[:30]}...")
                print(f"   Amount: {info['amount']:,} ZION")
                print(f"   Current balance: {balance_zion:,.0f} ZION")
                print(f"   Type: {info['type']}")
                print()
            
            print(f"💰 Total pre-mine: {total_amount:,} ZION")
        
        elif choice == '5':
            print("👋 Ukončuji backup tool...")
            break
        
        else:
            print("❌ Neplatná volba")

if __name__ == "__main__":
    main()