# Scripts

## Secure Wallet Backups

Two scripts help you extract wallet files from a running container and store them encrypted under `logs/runtime/<timestamp>/vault/`.

- Bash (Linux/macOS): `scripts/secure-backup.sh`
  ```bash
  ./scripts/secure-backup.sh zion-walletd MY-RECIPIENT-KEYID
  # Produces: logs/runtime/<timestamp>/vault/zion-wallet-<timestamp>.tar.gz.gpg
  ```
  Requirements: GnuPG installed, recipient public key present in your keyring.

- PowerShell (Windows): `scripts/secure-backup.ps1`
  ```powershell
  ./scripts/secure-backup.ps1 -Container zion-walletd -Passphrase "STRONG_PASSPHRASE"
  # Produces: logs/runtime/<timestamp>/vault/zion-wallet-<timestamp>.zip.aes
  ```
  AES-256 with PBKDF2-derived key. Keep the passphrase safe (password manager).

Decryption:
- GPG: `gpg --output out.tar.gz --decrypt zion-wallet-<ts>.tar.gz.gpg`
- AES (PS): a small decrypter script can be added; for now decrypt with a compatible tool or ask to generate `secure-restore.ps1`.

## Legacy Backup

`scripts/backup-wallet.sh` keeps a plaintext tar.gz for local testing only. Prefer the secure variants above for any real keys.

## Windows Miner Launcher

`scripts/Start-Mining.ps1` spustí XMRig na Windows s automatickou detekcí peněženky a výběrem portu.

- Autodetekce peněženky: `../wallets/miner-ryzen/miner-ryzen.wallet.address`
- Autodetekce XMRig: repo binárka v `mining/platforms/windows/xmrig-6.21.3/xmrig.exe` (případně PATH)
- Probing portů a failover: 443 → 80 → 3333 (bez TLS; 443 je pro průchod restriktivní sítí)
- Generuje config do `Zion/mining/xmrig-<PCNAME>.json`

Příklady:
```powershell
# Náhled (bez spuštění)
pwsh -NoProfile -ExecutionPolicy Bypass -File ./scripts/Start-Mining.ps1 -DryRun

# Spuštění (repo XMRig se najde automaticky)
pwsh -NoProfile -ExecutionPolicy Bypass -File ./scripts/Start-Mining.ps1 -RigId Ryzen3600
```

## SSH patch portů na serveru

`scripts/ssh-patch-ports.ps1` upraví vzdáleně publikované porty pro pool tak, aby byl dostupný i na 443/80 a restartuje pouze `uzi-pool`.

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File ./scripts/ssh-patch-ports.ps1 -ServerIp 91.98.122.165 -AlsoHttp
# Vytvoří docker/compose.override-ports.yml na serveru
# Namapuje 3333 a pokud jsou volné i 443 a/nebo 80
# Restartuje jen uzi-pool s --no-deps
```
