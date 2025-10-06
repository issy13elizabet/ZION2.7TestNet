# SESSION: Nová miner peněženka a porty (2025-09-23)

## Nová miner peněženka (Ryzen)
- Vygenerováno offline v kontejneru `zion:production-minimal`, uloženo mimo repo do `D:\Zion TestNet\wallets\miner-ryzen`.
- Adresa (Z3): `Z321vzLnnRu15AF82Kwx5DVZAX8rf1jZdDThTFZERRwKV23rXUqcV5u5mvT45ZDpW71kG3zEGuM5F2n3uytvAx5G9jFSY5HcFU`
- Pozn.: View/Spend klíče neukládat do gitu. Doporučeno zálohovat šifrovaně (AES/GPG) mimo pracovní repo.

## Stav pool/dev wallets
- Pool wallet: používá se `Z321y5Ugios4VrzgpbGkvoBrmA3QeVC2xao6kY3SDSkA7J8toEUktYzBAYyqQ8SJvP4kAaz1APCBEGRuYWersbCR33tetBoXxg` (ověřeno přes wallet-adapter `/wallet/address`).
- DEV_ADDRESS: zatím nedefinováno (env prázdné)
- CORE_DEV_ADDRESS: zatím nedefinováno (env prázdné)

## Porty 3333 vs. SSH
- `3333` je vyhrazen pro Stratum (kontejner `zion-uzi-pool` publikuje `0.0.0.0:3333->3333/tcp`).
- SSH doporučeno spouštět na `2222` (úprava `C:\ProgramData\ssh\sshd_config`: `Port 2222` + restart `sshd`).
- Kontrola: `Test-NetConnection -ComputerName 91.98.122.165 -Port 2222`

## Další kroky
- Aktualizovat XMRig konfigurace na novou miner adresu (Ryzen / test) a redeploy.
- Vygenerovat DEV a CORE-DEV peněženky (Z3), adresy zapsat do `.env` a restartovat `uzi-pool`.
