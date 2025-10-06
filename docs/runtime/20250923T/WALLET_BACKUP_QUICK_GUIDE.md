# Wallet Backup — Quick Guide

Minimal, practical steps to back up (encrypted) your wallet files from the running container. Raw files are never committed.

## Windows (PowerShell, AES-256)
```powershell
# Create encrypted backup (.zip.aes) into logs/runtime/<ts>/vault/
./scripts/secure-backup.ps1 -Container zion-walletd -Passphrase "STRONG_PASSPHRASE"
```
Decryption (PowerShell snippet):
```powershell
# Read header (Salt+IV) and decrypt to out.zip
param([string]$File, [string]$Passphrase)
$fs=[IO.File]::OpenRead($File)
$br=New-Object IO.BinaryReader($fs)
$Salt=$br.ReadBytes(16); $IV=$br.ReadBytes(16)
$Kdf=New-Object Security.Cryptography.Rfc2898DeriveBytes($Passphrase,$Salt,100000)
$Key=$Kdf.GetBytes(32)
$Aes=[Security.Cryptography.Aes]::Create(); $Aes.Key=$Key; $Aes.IV=$IV; $Aes.Mode='CBC'; $Aes.Padding='PKCS7'
$cs=New-Object Security.Cryptography.CryptoStream($fs,$Aes.CreateDecryptor(),[Security.Cryptography.CryptoStreamMode]::Read)
$out=[IO.File]::OpenWrite("out.zip"); $cs.CopyTo($out); $out.Close(); $cs.Close(); $fs.Close()
```

## Linux/macOS (GPG)
```bash
# Create encrypted backup (.tar.gz.gpg) into logs/runtime/<ts>/vault/
./scripts/secure-backup.sh zion-walletd <RECIPIENT-KEYID>

# Decrypt back to out.tar.gz
gpg --output out.tar.gz --decrypt logs/runtime/<ts>/vault/zion-wallet-<ts>.tar.gz.gpg
```

## What gets backed up
- `/home/zion/.zion/pool.wallet`
- `/home/zion/.zion/*.keys`
- `/home/zion/.zion/*.log`

Keep passphrases and private keys in a password manager; store encrypted backups offsite with versioning.
