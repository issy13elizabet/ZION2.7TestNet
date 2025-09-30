# ZION 2.6.5 Wallet Adapter Status & Integration

**Datum:** 30. září 2025  
**Scope:** Analýza wallet adapteru v kontextu nové 2.6.5 infrastruktury  

---
## 🏛️ Současný Stav Wallet Adapteru

### Architektura
```
Frontend UI ──> Wallet Adapter (18099) ──> Wallet Daemon (8070/json_rpc)
                      │                           │
                      └─────> RPC Shim/Bridge ──> ZION Daemon (18081)
```

### Implementované Funkce (adapters/wallet-adapter/server.js)
| Kategorie | Endpoint | Popis | Security |
|-----------|----------|-------|----------|
| **Wallet Core** | `GET /wallet/balance` | Zůstatek | Public |
| | `GET /wallet/address` | Hlavní adresa (multi-variant RPC) | Public |
| | `POST /wallet/create_address` | Nová adresa | API Key |
| | `POST /wallet/save` | Flush wallet | API Key |
| | `GET /wallet/keys` | Export spend/view keys | API Key |
| | `POST /wallet/send` | Transakce | API Key + Rate Limit |
| | `POST /wallet/validate` | Validace adresy (regex Z3/aj) | Public |
| **Explorer** | `GET /explorer/summary` | Tip + poslední header | Public |
| | `GET /explorer/blocks` | Seznam headerů (asc/desc) | Public |
| | `GET /explorer/block/height/:h` | Header dle výšky | Public |
| | `GET /explorer/block/hash/:hash` | Header dle hashe | Public |
| | `GET /explorer/search` | Vyhledávání (hash/height) | Public |
| | `GET /explorer/stats` | Block interval statistiky | Public |
| **Monitoring** | `GET /healthz` | Health + daemon/wallet ping | Public |
| | `GET /metrics` | Prometheus metriky | Public |
| **Historie** | `GET /wallet/history` | Transakční historie | Public |
| **Pool Support** | `GET /pool/blocks-recent` | Poslední bloky pro pool UI | Public |

---
## 🔧 Environment Proměnné
| Název | Default | Popis |
|-------|---------|-------|
| `ADAPTER_PORT` | 18099 | HTTP port adapteru |
| `ADAPTER_BIND` | 0.0.0.0 | Bind adresa |
| `WALLET_RPC` | http://walletd:8070/json_rpc | Wallet daemon endpoint |
| `SHIM_RPC` | http://zion-rpc-shim:18089/json_rpc | Chain RPC (Node shim) |
| `ADAPTER_API_KEY` | "" | API klíč pro protected endpointy |
| `REQUIRE_API_KEY` | true (prod) | Vyžadovat API key |
| `CORS_ORIGINS` | "" | CORS whitelist (comma-separated) |
| `SEND_RATE_WINDOW_MS` | 60000 | Rate limit okno |
| `SEND_RATE_MAX` | 6 | Max transakcí per okno |
| `LOG_LEVEL` | info (prod), debug (dev) | Pino log level |
| `NODE_ENV` | - | Environment detection |

---
## 🎯 Integrace se 2.6.5 (Go Bridge)
### Aktuální Stav
- Wallet adapter stále ukazuje na Node shim (`SHIM_RPC`).
- **Migrace:** Změna `SHIM_RPC=http://zion-bridge:8090/api/v1/json_rpc` → hotovo bez kódu changes.
- Go bridge už má aliasy metod kompatibilní s očekáváními adapteru.

### Test Migrace
```bash
# Spuštění s Go bridge
export SHIM_RPC=http://localhost:8090/api/v1/json_rpc
export ADAPTER_PORT=18099
node adapters/wallet-adapter/server.js

# Test connectivity
curl http://localhost:18099/healthz
curl http://localhost:18099/explorer/summary
```

---
## 🛡️ Security & Robustnost
### Silné Stránky
- API key gate pro kritické operace (`/wallet/send`, `/wallet/keys`, `/wallet/create_address`).
- Rate limiting na `/wallet/send` (default 6/min).
- CORS whitelist (konfigurační).
- Helmet security headers.
- Address validation regex (`ZION_ADDR_REGEX`): povoluje Z3 (mainnet) + aj (legacy).
- Timeout na RPC volání (5s).

### Robustní Patterns
- `walletTry()` — zkouší více variant RPC názvů pro kompatibilitu.
- `extractAddress()` — heuristika pro různé wallet RPC odpovědi.
- Fallback chování v explorer endpointech (při chybě vrací partial data).
- In-memory cache (TTL 3–5s) pro častá explorer volání.

---
## 📊 Observability
### Prometheus Metriky
```
zion_adapter_uptime_seconds
zion_adapter_requests_total
zion_adapter_route_requests_total{route="balance|send|address|..."}
zion_adapter_send_total{status="ok|error"}
```

### Logging
- Structured JSON (pino).
- Request logging (morgan) v dev módu.
- Error context v HTTP responses.

---
## 🚨 Identifikované Mezery
| Oblast | Problém | Priorita | Doporučení |
|--------|---------|----------|------------|
| **Debug Visibility** | Chybí detailní RPC request/response logy | Medium | Přidat `ADAPTER_DEBUG` + log file |
| **Chain RPC Naming** | `SHIM_RPC` matoucí po Go migraci | Low | Alias `CHAIN_RPC_URL` |
| **Metrics Depth** | Jen counters, chybí latence histogramy | Medium | Prometheus buckets |
| **Explorer Scalability** | Serial RPC dotazy (N × getheader) | Medium | Batch endpoint v Go bridge |
| **History Performance** | Skenuje 1000 bloků lineárně | Low | Incremental cache |
| **Send Audit** | Žádná audit stopa pro transakce | High | JSON audit log |
| **Fee Estimation** | Chybí fee suggestion | Low | Endpoint `/wallet/fee_estimate` |
| **Address Validation** | Hardcoded regex, duplikace logic | Low | Shared modul |

---
## 🔮 Roadmap Konsolidace
### Fáze 1: Okamžité Vylepšení (kompatibilní změny)
- [ ] `ADAPTER_DEBUG` env → RPC request/response logging
- [ ] `CHAIN_RPC_URL` alias k `SHIM_RPC` (deprecation warning)
- [ ] Extended `/healthz` (timestamp delta, adapter verze)
- [ ] Send audit log (`/var/log/zion-wallet-send.jsonl`)

### Fáze 2: Go Port Příprava
- [ ] Přesun logiky do Go modulu `pkg/wallet-gateway/`
- [ ] Shared RPC client pool (wallet + chain)
- [ ] WebSocket push pro balance change + nové bloky
- [ ] gRPC interní API (Lightning ↔ Wallet ↔ Explorer)

### Fáze 3: Unifikace
- [ ] Single binary: `zion-bridge` (RPC + Wallet + Explorer + Lightning)
- [ ] Legacy Node adapter → `legacy/wallet-adapter/`
- [ ] Unified metriky: `zion_gateway_requests_total{component="wallet"}`

---
## ✅ Kompatibilita 2.6.5
| Komponenta | Status | Poznámka |
|------------|--------|----------|
| **Go Bridge Integration** | ✅ Ready | Jen změna `SHIM_RPC` proměnné |
| **Mining Pool** | ✅ Compatible | Pool používá daemon direct, adapter jen pro UI/API |
| **Frontend** | ✅ Working | Next.js proxy routes → adapter endpoints |
| **Explorer UI** | ✅ Functional | Všechny potřebné endpointy implementovány |
| **Wallet Operace** | ✅ Production Ready | Send + balance + keys + create address |
| **Monitoring** | ✅ Basic | Prometheus metriky + healthz |

---
## 🎯 Závěr
Wallet adapter v 2.6.5:
- **Plně funkční** s aktuálním stackem (Node shim).
- **Připraven** na jednoduchou migraci na Go bridge.
- **Robustní** design pro různé wallet daemon implementace.
- **Bezpečný** pro production použití (API keys, rate limiting, validace).
- **Možné vylepšení:** debug logging, audit trail, latence metriky.

**Doporučení:** Pokračovat s aktuální implementací, postupně připravovat Go port pro budoucí konsolidaci.

---
**Status:** Wallet adapter je production-ready komponent připravený na 2.6.5 infrastrukturu.