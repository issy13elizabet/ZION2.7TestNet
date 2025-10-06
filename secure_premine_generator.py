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
import argparse
from datetime import datetime
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

try:
    # BIP39 mnemonic support (trezor's mnemonic)
    from mnemonic import Mnemonic
except Exception:
    Mnemonic = None

class SecurePremineGenerator:
    """Bezpečný generátor premine adres"""
    
    def __init__(self, *, auto: bool = False, backup_mode: str = "interactive", words: int = 12, passphrase: str = ""):
        self.addresses = {}
        self.warning_shown = False
        self.auto = auto
        self.backup_mode = backup_mode  # interactive|enc|paper|both
        self.words = words  # 12 nebo 24
        self.passphrase = passphrase or ""
        if self.words not in (12, 24):
            raise ValueError("words must be 12 or 24")
        
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
            if not self.auto:
                print("Pokračovat? (yes/NO):", end=" ")
                response = input().strip().lower()
                if response != 'yes':
                    print("❌ Operace zrušena uživatelem")
                    exit(1)
            else:
                print("Pokračuji v AUTO režimu…")
                
            self.warning_shown = True
            print()

    def _gen_mnemonic_and_privkey(self, purpose: str, entropy_bytes: bytes):
        """Vygeneruje BIP39 mnemonic (12/24) a private key deterministicky z BIP39 seed + purpose.
        Pokud není k dispozici knihovna mnemonic, fallback na bez-mnemonic režim.
        """
        mnemonic_words = None
        seed = None
        if Mnemonic is not None:
            lang = "english"
            mnemo = Mnemonic(lang)
            # Pro 12 slov použijeme 128 bitů entropie (16B), pro 24 slov 256 bitů (32B)
            needed = 16 if self.words == 12 else 32
            entropy = entropy_bytes if len(entropy_bytes) >= needed else entropy_bytes + secrets.token_bytes(needed - len(entropy_bytes))
            mnemonic_words = mnemo.to_mnemonic(entropy)
            seed = mnemo.to_seed(mnemonic_words, passphrase=self.passphrase)
            privkey_material = seed + purpose.encode()
            private_key = hashlib.sha256(privkey_material).hexdigest()
        else:
            # Fallback: žádný mnemonic, pouze privkey z entropie
            private_key = hashlib.sha256(entropy_bytes).hexdigest()
        return mnemonic_words, private_key
    
    def generate_address(self, purpose, consciousness=None):
        """Generuje novou zabezpečenou adresu"""
        # Generate cryptographically secure entropy
        entropy = secrets.token_bytes(32)

        # Create mnemonic + private key (deterministic from seed + purpose)
        mnemonic_words, private_key = self._gen_mnemonic_and_privkey(purpose, entropy)
        
        # Generate public address (simplified for demo)
        address_hash = hashlib.sha256((private_key + purpose).encode()).hexdigest()
        
        if consciousness:
            address = f"ZION_{consciousness}_{address_hash[:30].upper()}"
        else:
            address = f"ZION_{purpose.upper()}_{address_hash[:30].upper()}"
            
        return {
            'address': address,
            'private_key': private_key,
            'mnemonic': mnemonic_words,
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
        
        # Development Team Fund (1.44B ZION)
        dev = self.generate_address('Development Team Fund')
        dev['amount'] = 1_440_000_000
        dev['type'] = 'development'
        self.addresses[dev['address']] = dev
        print(f"✅ Development Team Fund: {dev['address'][:50]}...")

        # Network Infrastructure (SITA) (999M)
        sita = self.generate_address('Network Infrastructure (SITA)')
        sita['amount'] = 999_000_000
        sita['type'] = 'infrastructure'
        self.addresses[sita['address']] = sita
        print(f"✅ Network Infrastructure (SITA): {sita['address'][:50]}...")

        # Children Future Fund (999M)
        chf = self.generate_address('Children Future Fund')
        chf['amount'] = 999_000_000
        chf['type'] = 'charity'
        self.addresses[chf['address']] = chf
        print(f"✅ Children Future Fund: {chf['address'][:50]}...")
        
        # Network Administrator (999M)
        admin_addr = self.generate_address('Network Administrator', 'MAITREYA_BUDDHA')
        admin_addr['amount'] = 999_000_000
        admin_addr['type'] = 'network_admin'
        self.addresses[admin_addr['address']] = admin_addr
        print(f"✅ Network Administrator: {admin_addr['address'][:50]}...")
        
        # Genesis reward (333M)
        genesis_addr = self.generate_address('Genesis Reward', 'ON_THE_STAR')
        genesis_addr['amount'] = 333_000_000
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
                if data.get('mnemonic'):
                    f.write(f"BIP39 ({self.words} words): {data['mnemonic']}\n")
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
    # CLI args
    parser = argparse.ArgumentParser(description="ZION Secure Premine Generator")
    parser.add_argument("--auto", action="store_true", help="Automatický režim (žádné interaktivní dotazy)")
    parser.add_argument("--backup", choices=["interactive","enc","paper","both"], default="interactive", help="Typ zálohy")
    parser.add_argument("--words", type=int, choices=[12,24], default=12, help="Počet slov BIP39 (12/24)")
    parser.add_argument("--password", type=str, default="", help="Heslo pro šifrovanou zálohu (min 12 znaků)")
    args = parser.parse_args()

    if args.words in (12,24) and Mnemonic is None:
        print("⚠️ Varování: Knihovna 'mnemonic' není nainstalována. BIP39 seed nebude vytvořen. Použiji fallback bez seed frází.")

    generator = SecurePremineGenerator(auto=args.auto, backup_mode=args.backup, words=args.words, passphrase="")
    generator.generate_premine_addresses()

    if args.backup == "interactive":
        print("🔐 Možnosti zálohy:")
        print("1. Šifrovaná záloha (doporučeno)")
        print("2. Paper wallet") 
        print("3. Obojí")
        print("4. Přeskočit (NEBEZPEČNÉ!)")
        print()
        choice = input("Vyber možnost (1-4): ").strip()
        if choice in ['1','3']:
            password = input("Zadej silné heslo pro šifrování: ")
            if len(password) < 12:
                print("❌ Heslo musí mít alespoň 12 znaků!")
                return
            generator.create_encrypted_backup(password)
        if choice in ['2','3']:
            generator.create_paper_wallet()
        if choice == '4':
            print("⚠️  VAROVÁNÍ: Bez zálohy ztratíte přístup k premine adresám!")
            confirm = input("Opravdu pokračovat bez zálohy? (yes/NO): ")
            if confirm.lower() != 'yes':
                return
    else:
        # Non-interactive backups
        if args.backup in ("enc","both"):
            if len(args.password) < 12:
                print("❌ Pro šifrovanou zálohu je potřeba heslo (min 12 znaků) – použijte parametr --password")
                return
            generator.create_encrypted_backup(args.password)
        if args.backup in ("paper","both"):
            generator.create_paper_wallet()

    generator.security_checklist()

if __name__ == "__main__":
    main()