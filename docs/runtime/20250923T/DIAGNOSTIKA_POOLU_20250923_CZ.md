# Diagnostika poolu a shim (23.9.2025)

Tento soubor shrnuje aktuální zjištění, příčiny, a doporučené kroky pro řešení problému, že horníci (miners) odesílají share, ale ty nejsou viditelné v poolu.

Shrnutí zjištění

- Konfigurace poolu (`adapters/uzi-pool-config/config.json`) ukazuje, že port pro stratum je `3333` a `shareTrust` je povolené s `min: 10` a `threshold: 10`.
- `docker/uzi-pool/entrypoint.sh` kopíruje konfig do `/app/run-config.json` a spouští `patch-rx.js`, který runtime upravuje `/app/lib/pool.js` pro RandomX (zjednodušuje `processShare`, přidává `seed_hash`, upravuje `Miner.getJob`).
- Máme poznámky v `logs/runtime/*` že `rpc-shim` byl vracející `getblocktemplate: Core is busy` (-9) a že to způsobilo EOF v klientských spojeních. Do `rpc-shim` byla nasazena cache/prefetch logika a metriky.

Možné příčiny proč nejsou shares viditelné

1) `shareTrust` chování: nastavení `min: 10` může znamenat, že pool nezobrazuje nebo nepočítá shares pro nové minery dokud nedosáhnou prahu důvěry. To by vysvětlovalo "neviditelné" shares (jsou přijímány, ale neprojevují se ve statistikách).

2) Runtime patch (`patch-rx.js`) nefunguje korektně na běžící verzi `node-cryptonote-pool`: pokud patch selže nebo injektuje chybný kód, `processShare` může vrhat chyby nebo vůbec nezapisovat share do úložiště.

3) RPC shim submit — pokud `rpc-shim` odmítá nebo vrací chyby při `submitblock` (nebo má dlouhý backoff), pool může logovat chyby nebo ztrácet block candidate flow.

4) Redis / persistence problém: pool může přijímat share, ale je-li Redis nedostupný, nejsou shares perzistentně ukládány ani viditelné v UI/API.

Doporučené kroky (rychlé ověření)

A) Ověřit runtime config a logy v běžícím kontejneru `zion-uzi-pool`:
   - `docker logs --tail 500 zion-uzi-pool` → hledejte `Miner connected`, `Accepted share`, `Rejected low difficulty`, `submit result` a chyby.
   - `docker exec zion-uzi-pool cat /app/config.json` → ověřit, že runtime config obsahuje očekávané hodnoty (pool address, shareTrust, ports).
   - `docker exec zion-uzi-pool grep -nE 'recordShareData|processShare|Accepted share|Rejected' /app/lib/pool.js || true` → zkontrolovat implementaci `processShare` a volání `recordShareData`.

B) Dočasně zkrátit/odstranit `shareTrust` aby se ověřilo, že shares budou okamžitě viditelné:
   - V `adapters/uzi-pool-config/config.json` změnit `shareTrust.min` a `shareTrust.threshold` na `1` a restartovat `uzi-pool`:
     - `docker compose -f docker/compose.pool-seeds.yml up -d --no-deps uzi-pool` (nebo `docker restart zion-uzi-pool` pokud již používáte existující image).

C) Dočasně vypnout banning pro test:
   - V `config.json` nastavit `banning.enabled` na `false` a restartovat pool.

D) Zkontrolovat `rpc-shim` metriky na hostu:
   - `curl -sS http://127.0.0.1:18089/metrics.json` → zkontrolovat `gbt_busy_retries_total`, `submit_error_total`, `gbt_cache_hits_total`.

E) Zkontrolovat Redis:
   - `docker exec -it zion-redis redis-cli -p 6379 KEYS '*'` → ověřit existenci klíčů `pool:*` a `worker:*`.

Následující kroky (po ověření výše)

- Pokud logs ukážou `Accepted share` ale front-end je stále prázdný → pravděpodobně problém v `shareTrust` nebo předávání statistik do API. Doporučuji snížit `shareTrust.min` na `1` a ověřit.
- Pokud `processShare` vyhazuje chyby (syntax/runtime) → upravit `patch-rx.js` nebo přímo upravit `lib/pool.js` v běžícím kontejneru (hotpatch) a restartovat `uzi-pool`.
- Pokud `rpc-shim` reportuje vysoký `submit_error_total` nebo `getblocktemplate` busy frekventně → ladit shim logiku (už je deployed) a případně zvýšit `GBT_CACHE_MS` a `BUSY_CACHE_FACTOR`.

Poznámka o bezpečnosti a provozu

- Všechny změny doporučuji provádět postupně a nejprve read-only kontroly logů. Úpravy configu a restart kontejnery mají vliv na těžbu, doporučuji je dělat mimo špičku.

---
Soubor vytvořen automaticky a commitnut do repozitáře (viz následující commit). Pokud chcete, mohu teď soubor commitnout a pushnout do `origin/main` (provedu commit a push). Pokud ano, potvrďte a já to udělám.
