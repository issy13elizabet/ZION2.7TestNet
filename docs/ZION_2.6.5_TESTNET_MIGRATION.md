# ZION 2.6 → 2.6.5 TestNet Migration Guide

**Datum:** 30. září 2025  
**Status:** Dokončená integrace reálného těžebního stacku + Go RPC Bridge diagnostika  

---
## 🎯 Cíl Release 2.6.5
- Eliminace mock mining logiky (žádné Math.random bloky, vše skutečný daemon data)
- Konsolidace RPC vrstvy → příprava přechodu z Node `zion-rpc-shim-simple.js` na Go `bridge/main.go`
- Zpřehlednění konfigurace přes env proměnné (PATH, DEBUG, LOG_FILE)
- Rozšířená diagnostika pro `getblocktemplate` a stabilizace poolu

---
## 🔄 Klíčové Změny
| Oblast | Stav před | Stav po (2.6.5) |
|--------|-----------|-----------------|
| RPC Bridge | Node shim (jednoduchý) | Go bridge + fallback Node shim |
| Mining Pool | Závislé na shim root `/` | Podpora `DAEMON_RPC_PATH` (`/api/v1/json_rpc`) |
| Debug | Omezené logy v poolu | `MINING_POOL_DEBUG` + file output |
| Bridge Debug | Nebylo | `RPC_BRIDGE_DEBUG`, `RPC_BRIDGE_LOG_FILE` |
| GBT Retry | Základní | 4 pokusy + latence logy + busy detekce |
| Env Konzistence | Částečná | Tabulky proměnných v README |
| Dokumentace | Roztroušená | README + tento MIGRATION průvodce |

---
## 🧱 Struktura Komponent
```
[ ZION Daemon ] --> [ Go RPC Bridge (/api/v1/json_rpc) ] --> [ Mining Pool (Stratum:3333) ] --> Miners
                                ^ (fallback)                        \
                                +-- Node Shim (legacy)               +--> Prometheus / Logs
```

---
## 🆕 Env Proměnné (Nové / Rozšířené)
| Název | Default | Popis |
|-------|---------|-------|
| `DAEMON_RPC_PATH` | `/` | Přepnutí root → `/api/v1/json_rpc` pro Go bridge |
| `MINING_POOL_DEBUG` | false | Loguje každý upstream RPC call + raw body |
| `MINING_POOL_DEBUG_FILE` | /tmp/mining-pool-debug.log | Cesta k debug log souboru |
| `RPC_BRIDGE_DEBUG` | false | Aktivuje detailní CALL/RAW/GBT_* logy v Go bridge |
| `RPC_BRIDGE_LOG_FILE` | /tmp/zion-rpc-bridge.log | Cesta k log souboru (append) |

---
## ⛏️ Přechod Kroky
1. (Volitelné) Ponech Node shim dokud není Go toolchain k dispozici
2. Nasad Go bridge:
   ```bash
   cd bridge
   go build -tags lnd -o zion-bridge
   ./zion-bridge &
   ```
3. Přepni pool na API path:
   ```bash
   export DAEMON_HOST=127.0.0.1
   export DAEMON_PORT=8090
   export DAEMON_RPC_PATH=/api/v1/json_rpc
   node mining/zion-real-mining-pool.js
   ```
4. Ověř `getblocktemplate` funkčnost: tail logy bridge + pool
5. Aktivuj debug při problémech:
   ```bash
   export RPC_BRIDGE_DEBUG=true
   export MINING_POOL_DEBUG=true
   ```
6. Po ověření stabilní těžby může být Node shim odstraněn z default path

---
## 🔍 GBT Diagnostika
Bridge loguje:
- `GBT_ATTEMPT n=<i>`
- `GBT_ERR n=<i> err="core is busy"`
- `GBT_OK n=<i> latency_ms=<N>`
- `GBT_RAW n=<i> bytes=<N>`
- `GBT_FINAL success=<bool> attempts=<N>`

Pokud 4 pokusy selžou → zvaž navýšení backoffu nebo health-check daemon procesů.

---
## 🧪 Rychlý Test Set
```bash
# 1. getinfo přes bridge
curl -s -X POST -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"getinfo"}' \
  http://localhost:8090/api/v1/json_rpc | jq .

# 2. getblocktemplate (wallet + reserve_size dle potřeby)
curl -s -X POST -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":2,"method":"getblocktemplate","params":{"wallet_address":"<ADDR>","reserve_size":60}}' \
  http://localhost:8090/api/v1/json_rpc | jq .
```

---
## 📊 Prometheus Metriky
| Název | Popis |
|-------|-------|
| `zion_rpc_requests_total` | Počet RPC volání dle metody & statusu |
| `zion_rpc_request_duration_seconds` | Histogram latencí |
| `zion_daemon_up` | Gauge (1 pokud poslední call uspěl) |

---
## 🔮 Další Kroky (Roadmap)
- WebSocket notifikace pro nový template
- Cache `getinfo` TTL 2s
- Konsolidace shim kódu → archivace Node varianty
- Security token pro externí API přístup

---
## 📝 Poznámky
- Node shim zůstává dočasně kvůli fallback scénářům
- Debug logy jsou append-only → nastavit rotaci při produkčním nasazení (logrotate/systemd) 
- Všechny nové env proměnné zdokumentovány v hlavním README + zde

---
## ✅ Hotovo
Tato migrace připravuje půdu pro odlehčení runtime a lepší observabilitu. Go bridge je doporučená cesta vpřed.

---
