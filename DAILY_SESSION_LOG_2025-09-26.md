# DAILY SESSION LOG — 26. září 2025

Krátké shrnutí: Dnes jsme synchronizovali repo, opravili adresní prefix na Z3 (0x433F), znovu přebuildili a nasadili node na server. Vybudovali jsme minimální pool (stratum-like) stub, ověřili základní handshake, ale XMRig stále hlásí login error (code: 1). Největší blocker: wallet dál generuje adresy „ajmr…“ místo očekávaných „Z3…“ i po opravě prefixu a rebuild/deploy.

## Akce dnes

- Git/Repo
  - Re-synchronizace lokálního repozitáře na remote historii (hard reset na `origin/master`).
  - Dokumentace a logy průběžně ukládány do repa.

- CryptoNote core / Adresní prefix
  - V dokumentaci identifikován správný Z3 prefix: `0x433F` (místo `0x5a49`).
  - V `zion-cryptonote/src/CryptoNoteConfig.h` potvrzen/opraven `CRYPTONOTE_PUBLIC_ADDRESS_BASE58_PREFIX = 0x433F`.
  - Znovu přebuildován daemon a wallet (Docker build bez cache), nasazeno do image `zion:production-fixed`.

- Docker/Compose/Deploy
  - Na serveru `91.98.122.165` rebuild bez cache a redeploy node kontejneru (`zion-production`).
  - Health-checky pro node v pořádku (předtím ověřeno `/getinfo`).

- Pool stub (Node.js)
  - Stratum-like TCP JSON-RPC stub (login, getjob, keepalived, submit) běží na 3333.
  - Základní handshake přes `nc` OK, ale XMRig reportuje `login error code: 1`.

- Wallet test
  - Po přebuildu spuštěno `zion_wallet --generate-new-wallet /data/wallet_z3.wallet --password test` v kontejneru.
  - Výsledek: stále generuje adresu začínající `ajmr…` (např. `ajmqtuCt...`) místo očekávané `Z3…`.

## Zjištění & poznámky

- Prefix v kódu je aktuálně správný (`0x433F`), build proběhl bez cache, nasazení proběhlo.
- Přesto wallet generuje starší/odlišný prefix adres (`ajmr…`).
- Vytvořen samostatný log: `Z3_ADDRESS_ISSUE_LOG_2025-09-26.md` se stavem a hypotézami.

## Hypotézy „proč ne Z3“

1. Více míst s prefixy:
   - Další konstanty (např. integrated/testnet) mohou být mimo synchronizaci.
   - Zkontrolovat i `CRYPTONOTE_PUBLIC_INTEGRATED_ADDRESS_BASE58_PREFIX` a případné testnet prefixy.
2. Build/run artefakt:
   - Přestože build bez cache proběhl, prověřit, zda kontejner spouští nově zkopírovaný `zion_wallet`.
   - Ověřit verzi binárky, otisky, případně do obrazu vytisknout prefix při startu.
3. Režim sítě:
   - Zda wallet neběží v testnet módu/config (param nebo implicitně). Ověřit parametry běhu wallet.
4. Base58/varint implementace:
   - Ověřit, že používá správný tag (varint(prefix)) a že nový prefix odpovídá „Z3…“ dle dokumentace.

## Co je hotovo

- Repo sync ✅
- Oprava prefixu v kódu na `0x433F` ✅
- Rebuild bez cache + redeploy ✅
- Pool stub běží, handshake funguje ✅
- Problém s adresami `ajmr…` přetrvává ❌

## Další kroky (navrhované)

- Prohledat celý kód na `0x5a49` a další prefixy; zkontrolovat i integrated/testnet prefix konstanty.
- Přímo v `zion_wallet` při startu logovat runtime prefix (dočasně pro debugging) a ověřit, jakou hodnotu používá.
- Vypsat z `zion_wallet` síťový režim (mainnet/testnet) a parametry.
- Ověřit, že image skutečně obsahuje nově zkompilované binárky (hash/mtime), případně dočasně přibalit `strings`/`ldd` a prověřit.
- U XMRig login erroru upřesnit JSON-RPC odpověď loginu (fields: job, status, extensions) dle očekávání XMRig.

## Artefakty vytvořené dnes

- `Z3_ADDRESS_ISSUE_LOG_2025-09-26.md` — detail log k adresnímu prefixu.
- Tento souhrnný log: `DAILY_SESSION_LOG_2025-09-26.md`.

## Poznámka

- Kontejnerové názvy: compose používá službu `zion-node`, ale vzniká kontejner `zion-production` (mapování jménem služby vs. container_name v compose). Při operacích jsem používal aktuální název kontejneru.

— Konec záznamu —
