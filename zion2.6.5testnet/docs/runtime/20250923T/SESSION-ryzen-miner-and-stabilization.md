# SESSION: Ryzen miner + stabilizace pool/shim (2025-09-23)

Krátké shrnutí změn a validace v této iteraci:

## Co bylo upraveno
- XMRig Ryzen miner: opraven mount konfigu a příkaz
  - Nový soubor `docker/xmrig.ryzen.json` s `rig-id: RYZEN` a validní Z3 adresou.
  - `docker/compose.ryzen-miner.yml` přepnut na `--config=/config/xmrig.config.json` a mount nového JSONu.
- rpc-shim: seed1-only override pro stabilitu vzhledem k nestabilnímu `zion-seed2`.
  - Použito `docker/compose.rpc-shim-seed1.yml` → `ZION_RPC_URLS=http://zion-seed1:18081/json_rpc`.
- Uzi-Pool: donucen použít náš coin profil a validní pool adresu
  - V `docker/compose.pool-seeds.yml` přidáno `COINS_DIR=/config/coins`.
  - Nastaven `POOL_ADDRESS` na validní Z3 adresu (98 znaků).

## Důkazy/validace
- Ryzen miner logy: accepted shares a joby z `zion-uzi-pool:3333`.
- Pool logy: „New block to mine…“, „Miner connected …“, „Block … found … submit result: {"status":"OK"}“.
- rpc-shim běží seed1-only; zmizely ENOTFOUND chyby vůči `zion-seed2`.

## Poznámky
- Dokud nebude `zion-seed2` opraven (problém se storage init), necháváme shim seed1-only.
- Po opravě seed2 vrátit multi-URL v shim (`seed1,seed2`).
