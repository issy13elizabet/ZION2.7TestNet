# Denní log – 2025-09-26

Tento log shrnuje dnes provedené změny, stav nasazení a další kroky.

## Provedené akce

- Synchronizace repozitáře s remote (preferována upstream historie, hard-reset na origin/main).
- Nalezení autoritativních adres (pool/dev) v `config/zion-mining-pool.json` a jejich zadrátování:
  - Přidány konstanty do jádra (`zion-cryptonote/src/CryptoNoteConfig.h`): `ZION_DEV_DONATION_ADDRESS`, `ZION_POOL_PRIMARY_ADDRESS`.
  - Aktualizace adresy v `zion-core/src/modules/mining-pool.ts` na mainnet pool adresu.
- Úprava Docker sestavení pro daemon:
  - `docker/Dockerfile.zion-cryptonote.minimal` nyní kopíruje `config/mainnet.conf` do `/config/zion.conf`.
  - Vytvořen `.env` se základními proměnnými (POOL_ADDRESS, DEV_ADDRESS, log level, porty).
  - `config/mainnet.conf` zjednodušen na ini-styl s podporovanými klíči (RPC/P2P/seed/logging).
- Build image `zion:production-fixed` a spuštění služby `zion-node` přes `docker-compose.prod.yml`.
- Oprava chyby „Failed to parse arguments: unrecognised option 'data-dir'“:
  - Odstraněn `--data-dir` z CMD v `Dockerfile.zion-cryptonote.minimal`. Data se ukládají do výchozího `/home/zion/.zion` (mountováno volume).
- Smoke test: daemon startuje, P2P běží na `0.0.0.0:18080`, RPC healthcheck uvnitř kontejneru je `healthy`.

## Stav

- Kontejner `zion-production` běží a je `healthy` (RPC healthcheck OK). Logy ukazují inicializaci řetězce a generování genesis bloku, protože v novém datovém adresáři zatím není blockchain.
- Síťové porty: P2P publikováno (18080), RPC je dostupné v kontejneru (18081). V compose je RPC nezveřejněno směrem ven (záměrně, pro produkci doporučen reverzní proxy / omezení přístupu).

## Další kroky

1. Bootstrap/seed nody: případně rozšířit `seed-node` v `config/mainnet.conf` o další seed adresy.
2. Zvážit publikaci RPC (18081) přes reverzní proxy s autentizací/whitelistem, pokud je třeba vzdálený dohled.
3. Pool profil: pokud chceme spustit `zion-pool`, zapnout profil `pool` v compose a ověřit konektivitu na `zion-node` RPC.
4. Parametrizace adres v aplikacích: postupně odstranit zbylé hardcodované adresy a číst z ENV/konfigurace.
5. Monitoring: využít existující `server-monitor.sh`/`prod-monitor.sh` nebo Prometheus/Uptime pro dohled nad nody.

## Poznámky

- Chování parsování konfigurace: `data-dir` je akceptován pouze jako CLI volba `--data-dir`, nikoliv jako klíč v konfiguračním souboru. Proto byl přepínač odstraněn z CMD a používá se výchozí cesta.
- Volume v compose je nastaveno na `/home/zion/.zion`, aby odpovídalo výchozímu chování daemona.
