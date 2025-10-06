#!/usr/bin/env python3
"""
ZION 2.7.4 - Secure Premine Generator
Bezpečné generování premine adres pouze lokálně - ŽÁDNÉ ULOŽENÍ DO GITU!
"""

import secrets
import hashlib
import base64
import json
import os
from datetime import datetime
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class SecurePremineGenerator:
    """Bezpečný generátor premine adres"""
    
    def __init__(self):
        self.addresses = {}
        self.warning_shown = False
        
    def show_security_warning(self):
        """Zobrazí bezpečnostní upozornění"""
        if not self.warning_shown:
            print("🚨" * 20)
            print("⚠️  KRITICKÉ BEZPEČNOSTNÍ UPOZORNĚNÍ ⚠️")
            print("🚨" * 20)
            print()
            print("TENTO NÁSTROJ GENERUJE SKUTEČNÉ PRIVATE KEYS!")
            print()
            print("PRAVIDLA BEZPEČNOSTI:")
            print("1. 🚫 NIKDY neukládejte výstup do Gitu!")
            print("2. 🔒 Používejte pouze na offline počítači!")
            print("3. 💾 Uložte na hardware wallet nebo paper backup!")
            print("4. 🔥 Smažte všechny dočasné soubory!")
            print("5. 🛡️ Použijte silné šifrování pro zálohy!")
            print()
            print("Pokračovat? (yes/NO):", end=" ")
            response = input().strip().lower()
            
            if response != 'yes':
                print("❌ Operace zrušena uživatelem")
                exit(1)
                
            self.warning_shown = True
            print()
    
    def generate_address(self, purpose, consciousness=None):
        """Generuje novou zabezpečenou adresu"""
        # Generate cryptographically secure entropy
        entropy = secrets.token_bytes(32)
        
        # Create private key
        private_key = hashlib.sha256(entropy).hexdigest()
        
        # Generate public address (simplified for demo)
        address_hash = hashlib.sha256((private_key + purpose).encode()).hexdigest()
        
        if consciousness:
            address = f"ZION_{consciousness}_{address_hash[:30].upper()}"
        else:
            address = f"ZION_{purpose.upper()}_{address_hash[:30].upper()}"
            
        return {
            'address': address,
            'private_key': private_key,
            'purpose': purpose,
            'consciousness': consciousness,
            'entropy_hex': entropy.hex(),
            'generated_at': datetime.now().isoformat()
        }
    
    def generate_premine_addresses(self):
        """Generuje všechny premine adresy"""
        self.show_security_warning()
        
        print("🔐 Generuji premine adresy...")
        print()
        
        # Mining operators (2B ZION each)
        mining_operators = [
            ('SACRED Mining Operator', 'SACRED'),
            ('QUANTUM Mining Operator', 'QUANTUM'),
            ('COSMIC Mining Operator', 'COSMIC'),
            ('ENLIGHTENED Mining Operator', 'ENLIGHTENED'),
            ('TRANSCENDENT Mining Operator', 'TRANSCENDENT')
        ]
        
        for purpose, consciousness in mining_operators:
            addr = self.generate_address(purpose, consciousness)
            addr['amount'] = 2_000_000_000  # 2B ZION
            addr['type'] = 'mining'
            self.addresses[addr['address']] = addr
            print(f"✅ {purpose}: {addr['address'][:50]}...")
        
        # Special funds (1B ZION each)
        special_funds = [
            'Development Team Fund',
            'Network Infrastructure (SITA)',
            'Children Future Fund'
        ]
        
        for purpose in special_funds:
            addr = self.generate_address(purpose)
            addr['amount'] = 1_000_000_000  # 1B ZION
            addr['type'] = 'fund'
            self.addresses[addr['address']] = addr
            print(f"✅ {purpose}: {addr['address'][:50]}...")
        
        # Network Administrator
        admin_addr = self.generate_address('Network Administrator', 'MAITREYA_BUDDHA')
        admin_addr['amount'] = 1_000_000_000  # 1B ZION
        admin_addr['type'] = 'network_admin'
        self.addresses[admin_addr['address']] = admin_addr
        print(f"✅ Network Administrator: {admin_addr['address'][:50]}...")
        
        # Genesis reward
        genesis_addr = self.generate_address('Genesis Reward', 'ON_THE_STAR')
        genesis_addr['amount'] = 342_857_142  # 342.857M ZION
        genesis_addr['type'] = 'genesis'
        self.addresses[genesis_addr['address']] = genesis_addr
        print(f"✅ Genesis Reward: {genesis_addr['address'][:50]}...")
        
        print()
        print(f"🎯 Celkem vygenerováno: {len(self.addresses)} adres")
        
        total_premine = sum(addr['amount'] for addr in self.addresses.values())
        print(f"💰 Celkový premine: {total_premine:,} ZION")
        print()
    
    def create_encrypted_backup(self, password):
        """Vytvoří šifrovanou zálohu"""
        print("🔒 Vytvářím šifrovanou zálohu...")
        
        # Serialize data
        backup_data = {
            'version': '2.7.4',
            'generated_at': datetime.now().isoformat(),
            'total_addresses': len(self.addresses),
            'addresses': self.addresses
        }
        
        json_data = json.dumps(backup_data, indent=2)
        
        # Encrypt
        salt = secrets.token_bytes(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        fernet = Fernet(key)
        
        encrypted_data = fernet.encrypt(json_data.encode())
        
        # Create backup file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"zion_premine_SECURE_BACKUP_{timestamp}.enc"
        
        backup_file = {
            'version': '2.7.4',
            'encrypted_data': base64.urlsafe_b64encode(encrypted_data).decode(),
            'salt': base64.urlsafe_b64encode(salt).decode(),
            'algorithm': 'Fernet/PBKDF2HMAC-SHA256',
            'iterations': 100000,
            'created_at': datetime.now().isoformat()
        }
        
        # SAVE TO DESKTOP - NOT TO GIT!
        desktop_path = os.path.expanduser("~/Desktop")
        backup_path = os.path.join(desktop_path, filename)
        
        with open(backup_path, 'w') as f:
            json.dump(backup_file, f, indent=2)
        
        print(f"✅ Šifrovaná záloha uložena: {backup_path}")
        print("⚠️  ULOŽTE TENTO SOUBOR NA BEZPEČNÉ MÍSTO!")
        print("⚠️  NEZAPOMEŇTE HESLO!")
        print()
        
    def create_paper_wallet(self):
        """Vytvoří paper wallet"""
        print("📄 Vytvářím paper wallet...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"zion_paper_wallet_CONFIDENTIAL_{timestamp}.txt"
        
        # SAVE TO DESKTOP - NOT TO GIT!
        desktop_path = os.path.expanduser("~/Desktop")
        paper_path = os.path.join(desktop_path, filename)
        
        with open(paper_path, 'w') as f:
            f.write("🔐 ZION 2.7.4 - CONFIDENTIAL PAPER WALLET\n")
            f.write("=" * 60 + "\n\n")
            f.write("⚠️  TENTO DOKUMENT OBSAHUJE PRIVATE KEYS!\n")
            f.write("⚠️  VYTISKNĚTE A ULOŽTE NA BEZPEČNÉM MÍSTĚ!\n")
            f.write("⚠️  SMAŽTE DIGITÁLNÍ KOPIE!\n\n")
            
            for address, data in self.addresses.items():
                f.write(f"Purpose: {data['purpose']}\n")
                f.write(f"Address: {address}\n")
                f.write(f"Private Key: {data['private_key']}\n")
                f.write(f"Amount: {data['amount']:,} ZION\n")
                f.write(f"Type: {data['type']}\n")
                if data.get('consciousness'):
                    f.write(f"Consciousness: {data['consciousness']}\n")
                f.write("-" * 60 + "\n\n")
            
            f.write(f"\nGenerated: {datetime.now().isoformat()}\n")
            f.write("Version: ZION 2.7.4\n")
        
        print(f"✅ Paper wallet vytvořen: {paper_path}")
        print("📄 VYTISKNĚTE IHNED A SMAŽTE DIGITÁLNÍ KOPII!")
        print()
    
    def security_checklist(self):
        """Bezpečnostní checklist"""
        print("🔒 BEZPEČNOSTNÍ CHECKLIST:")
        print()
        print("□ Záloha je uložena na hardware wallet")
        print("□ Paper wallet je vytištěn a uložen v trezoru")
        print("□ Digitální soubory jsou smazané z počítače")
        print("□ Hesla jsou uložena samostatně")
        print("□ Nikdo další nezná private keys")
        print("□ Žádné keys nejsou v Git repozitáři")
        print("□ Žádné keys nejsou v cloud storage")
        print("□ Testovací recovery je funkční")
        print()
        print("✅ PO DOKONČENÍ CHECKLISTU MŮŽETE POKRAČOVAT NA 2.7.5!")

def main():
    """Hlavní funkce"""
    print("🚀 ZION 2.7.4 - Secure Premine Generator")
    print("=" * 50)
    print()
    
    generator = SecurePremineGenerator()
    generator.generate_premine_addresses()
    
    print("🔐 Možnosti zálohy:")
    print("1. Šifrovaná záloha (doporučeno)")
    print("2. Paper wallet") 
    print("3. Obojí")
    print("4. Přeskočit (NEBEZPEČNÉ!)")
    print()
    
    choice = input("Vyber možnost (1-4): ").strip()
    
    if choice in ['1', '3']:
        password = input("Zadej silné heslo pro šifrování: ")
        if len(password) < 12:
            print("❌ Heslo musí mít alespoň 12 znaků!")
            return
        generator.create_encrypted_backup(password)
    
    if choice in ['2', '3']:
        generator.create_paper_wallet()
    
    if choice == '4':
        print("⚠️  VAROVÁNÍ: Bez zálohy ztratíte přístup k premine adresám!")
        confirm = input("Opravdu pokračovat bez zálohy? (yes/NO): ")
        if confirm.lower() != 'yes':
            return
    
    generator.security_checklist()

if __name__ == "__main__":
    main()