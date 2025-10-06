# ZION 2.7.4 – Premine Backup & Recovery Guide

DŮLEŽITÉ: Tento dokument popisuje bezpečnou zálohu a obnovu premine peněženek (12-slovné BIP39 seed fráze, privátní klíče) a jak obnovit přístup v případě ztráty.

---

## 1) Co bylo vygenerováno

- 10 premine peněženek (Mining Operators, Development, SITA, Children, Administrator, Genesis)
- Pro každou peněženku:
  - BIP39 seed (12 slov) – generováno lokálně (nikdy neukládat do Gitu)
  - Private key (odvozený ze seedu) – uveden v paper wallet
  - Adresa a účel
- Zálohy uložené na Plochu (Desktop):
  - Šifrovaný balíček: `zion_premine_SECURE_BACKUP_YYYYMMDD_HHMMSS.enc`
  - Paper wallet (TXT): `zion_paper_wallet_CONFIDENTIAL_YYYYMMDD_HHMMSS.txt`

---

## 2) Kde zálohy jsou

- macOS Desktop uživatele: `~/Desktop/`
- Příklady posledních souborů:
  - `zion_premine_SECURE_BACKUP_20251006_234618.enc`
  - `zion_paper_wallet_CONFIDENTIAL_20251006_234618.txt`

---

## 3) Jak otevřít šifrovaný balíček (.enc)

1. Heslo: použijte přesně heslo, zadané při vytváření záloh.
2. Decrypt v Pythonu (uvnitř venv):
   ```bash
   source venv/bin/activate
   python3 - <<'PY'
   import json, base64
   from cryptography.fernet import Fernet
   from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
   from cryptography.hazmat.primitives import hashes
   from pathlib import Path

   enc_path = Path('~/Desktop').expanduser() / 'zion_premine_SECURE_BACKUP_20251006_234618.enc'
   password = input('Zadej heslo: ')
   data = json.loads(enc_path.read_text())
   salt = base64.urlsafe_b64decode(data['salt'])
   kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=data['iterations'])
   key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
   f = Fernet(key)
   decrypted = f.decrypt(base64.urlsafe_b64decode(data['encrypted_data']))
   out = Path('premine_backup_decrypted.json')
   out.write_text(decrypted.decode())
   print('✅ Decrypted -> premine_backup_decrypted.json')
   PY
   ```
3. Výsledek: `premine_backup_decrypted.json` – obsahuje všechny adresy, částky, private keys a (pokud dostupné) mnemoniky.

---

## 4) Paper wallet (TXT)

- Otevřete `zion_paper_wallet_CONFIDENTIAL_*.txt`, vytiskněte a uložte do trezoru.
- Obsahuje: Purpose, Address, Private Key, (BIP39 12 word), Amount, Type.
- Doporučení: uchovávat 2–3 kopie na různých bezpečných místech.

---

## 5) Obnova peněženky (z BIP39 seedu)

- Doporučená varianta (hardware wallet):
  1) V zařízení zvolte "Restore from seed" (12 words) a zadejte přesně 12 slov.
  2) (Volitelné) Doplňte BIP39 passphrase, pokud byla použita.
  3) Importujte účet/adresu podle potřeby (viz derivation path – dle použité peněženky).

- Software peněženky (např. Electrum/others s BIP39):
  1) Nová peněženka > I already have a seed > BIP39 > 12 slov.
  2) Zadejte passphrase (pokud byla použita).
  3) Vyberte derivation path (může se lišit – tento projekt používá demo adresy; pro produkční nasazení doporučujeme standardní derivace jako m/44'/0'/0').

Poznámka: Náš generátor používá deterministickou vazbu seed+purpose -> private key, aby byla jednoznačná identita. Pro produkci zvažte standardní HD peněženku s více adresami.

---

## 6) Bezpečnostní zásady

- Nikdy neukládejte seed/keys do Gitu ani do cloudu
- Heslo uchovávejte samostatně, offline
- Vždy testujte recovery s malým zůstatkem dříve, než přesunete velké částky
- Při přesunu na 2.7.5 repo použijte výhradně čisté adresy (bez privátních klíčů)

---

## 7) Disaster recovery – rychlé kroky

1) Najděte `zion_premine_SECURE_BACKUP_*.enc` na Desktopu
2) Decryptujte pomocí hesla do `premine_backup_decrypted.json`
3) Ověřte adresy proti on-chain datům
4) Importujte cílové peněženky pomocí seedů/privátních klíčů
5) Proveďte testovací transakci

---

## 8) Meta

- Verze: ZION 2.7.4
- Datum vytvoření: 2025-10-06
- Generátor: `secure_premine_generator.py` (auto mode, 12 words, enc+paper)
