# Z3 Rollout & Wallet Backup (2025-09-23)

This document captures the rollout of the new ZION address prefix `Z3...` and prescribes a secure backup process for pool/dev wallets.

## Summary of Code/Config Changes
- Wallet validation updated to accept `Z3` (legacy `aj` kept):
  - `adapters/wallet-adapter/server.js`: `ZION_ADDR_REGEX = /^(Z3|aj)[...]{93}$/`
  - `adapters/uzi-pool-config/coins/zion.json`: `addressPrefix: "Z3"`, updated `addressRegex`.
- Uzi-Pool now reads runtime addresses from env:
  - Env: `POOL_ADDRESS`, `DEV_ADDRESS`, `CORE_DEV_ADDRESS`
  - `docker/uzi-pool/entrypoint.sh` injects them into runtime `/app/run-config.json`.
  - `docker/compose.pool-seeds.yml` passes the envs.
- Frontend: placeholders suggest `Z3...`.
- `.env.example` added with address variables.

## Generate New Z3 Wallets (Offline)
Use the core image to generate wallets and print the address without connecting to a daemon:
```sh
# Address will start with Z3...
docker run --rm -e HOME=/tmp zion:production-minimal \
  /usr/local/bin/zion_wallet \
  --generate-new-wallet /tmp/pool.wallet \
  --password "STRONG_PASS" \
  --command address
```
Record the following for each wallet: Address (Z3...), View Key, Spend Keys (if applicable).

Recommended wallets:
- Pool Payout Wallet (receives block rewards)
- Dev Donation Wallet
- Core-Dev Donation Wallet

## Set Addresses in Runtime (no secrets in git)
Fill `.env` (never commit real values):
```
POOL_ADDRESS=Z3...
DEV_ADDRESS=Z3...
CORE_DEV_ADDRESS=Z3...
```
Restart Uzi-Pool:
```sh
docker compose -f docker/compose.pool-seeds.yml up -d --no-deps --force-recreate uzi-pool
```

## Secure Backups (Encrypted)
Use the provided scripts to extract wallet containers from the running `zion-walletd` and encrypt them.

### Bash (Linux/macOS) – GnuPG
`scripts/secure-backup.sh` will:
- `docker cp` wallet files from container
- Create tar archive
- Encrypt to recipient key (`gpg --recipient <KEYID> --encrypt`)
- Shred temporary plaintext

### PowerShell (Windows) – AES-256
`scripts/secure-backup.ps1` will:
- `docker cp` wallet files
- Create zip
- Encrypt using AES-256 with a passphrase (PBKDF2)
- Remove plaintext zip

Backups are stored under `logs/runtime/<TIMESTAMP>/vault/` (git-ignored). Only encrypted artifacts (`.gpg`/`.aes`) are retained; never commit raw wallets.

## Operational Notes
- Pool must use a valid Z3 address; otherwise daemon RPC rejects getblocktemplate with "Failed to parse wallet address".
- Seed2 may require volume reset if blockchain init fails.
- Wallet Adapter `/wallet/validate` recognizes `Z3...` addresses; UI placeholders updated.

## Checklist
- [ ] Generate Z3 pool/dev wallets offline
- [ ] Fill `.env` with addresses
- [ ] Run secure backup scripts and store encrypted files in `vault/`
- [ ] Restart pool and verify logs
- [ ] Verify Adapter `/wallet/validate` and `/wallet/address`
