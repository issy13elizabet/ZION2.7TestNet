# SESSION: Iterace na mining (cílově 60 bloků) – 2025-09-23

Shrnutí:
- Opraven nestabilní RPC (seed2) přepnutím shim pouze na seed1 (override compose.rpc-shim-seed1.yml).
- Do skriptu start-mining.ps1 přidán přepínač -ShimSeed1Only.
- Pool a miner běží, share se přijímají, pool hlásí nalezené bloky, výška roste (ověřeno přes /json_rpc getheight).
- Seed2 po vyčištění volume stále padá na "Failed to initialize blockchain storage" – ponecháno k dořešení.

Provedené kroky:
1) Vytvořen compose override: docker/compose.rpc-shim-seed1.yml s `ZION_RPC_URLS=http://zion-seed1:18081/json_rpc`.
2) Restart rpc-shim s override, ověřen getheight = 3 → 4.
3) Úprava skriptu `scripts/start-mining.ps1`: nový switch `-ShimSeed1Only`, který při startu použije override.
4) Kontrola logů uzi-pool: potvrzeny block candidate a nalezení bloku (submit OK), následně nový gbt na vyšší výšku.
5) Pokus o stabilizaci seed2: stop container, odstranění docker_seed2-data, re-start. seed2 nadále hlásí chybu init blockchain storage.

Jak spustit mining do 60 bloků:
```
pwsh -File .\scripts\start-mining.ps1 -ShimSeed1Only -TargetBlocks 60
```
Skript:
- Spustí base služby (seed1, redis, walletd, wallet-adapter, rpc-shim) s override pro seed1-only.
- Vytáhne adresu z wallet-adapteru (nebo použije fallback), nastaví XMRig login.
- Spustí testovací XMRig a každých 10s čte výšku z `localhost:18089/json_rpc`.
- Po dosažení cílové výšky zastaví miner.

Poznámky k seed2 (troubleshooting návrh):
- Přidat `--data-dir=/home/zion/.zion` do command u seed2 (symetrie se seed1), případně použít compose.seed.yml s mounty `docker/seed2/config`, `docker/seed2/data`, `docker/seed2/logs`.
- Po zprovoznění seed2 vrátit shim na multi-URL: `http://zion-seed1:18081/json_rpc,http://zion-seed2:18081/json_rpc`.

Kontrolní příkazy:
```
# Výška přes shim
curl -s -H "Content-Type: application/json" -d '{"jsonrpc":"2.0","id":"0","method":"getheight"}' http://localhost:18089/json_rpc

# Logy poolu
docker logs -f zion-uzi-pool
```

Bezpečnost:
- Nepushujeme žádné raw wallet soubory, jen šifrované artefakty dle předchozích zásad.
- .gitignore již ignoruje `docker/seed2/data/` a `docker/seed2/logs/`.
