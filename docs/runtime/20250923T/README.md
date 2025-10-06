# Z3 Rollout Quickstart

This folder contains the Z3 rollout notes and secure wallet backup guide.

- Overview: `Z3_ROLLOUT_AND_WALLET_BACKUP.md`
- Quick backup steps: `WALLET_BACKUP_QUICK_GUIDE.md`
- Example env file: project root `.env.example`

Quick steps:
1) Generate new Z3 wallets (offline): see the guide for `zion_wallet --generate-new-wallet` usage.
2) Put addresses to `.env` as `POOL_ADDRESS`, `DEV_ADDRESS`, `CORE_DEV_ADDRESS`.
3) Restart Uzi-Pool: `docker compose -f docker/compose.pool-seeds.yml up -d --no-deps --force-recreate uzi-pool`
4) Securely back up wallets into `vault/` using the provided scripts.

Do not commit raw wallet files. Only encrypted artifacts (`.gpg`, `.aes`) are allowed in `vault/`.
