# ZION 2.6 → 2.6.5 MIGRATION GAP AUDIT

Datum: 2025-09-29
Status: INITIAL REPORT (Phase 1)

## 1. Cíl auditu
Zhodnotit rozdíl mezi:
- Legacy plnohodnotným C++ CryptoNote jádrem (`legacy/experiments/zion-cryptonote`)
- Novým TypeScript integračním jádrem (`zion2.6.5testnet/core`)

A navrhnout kroky k propojení (bridge) a postupné produkční aktivaci.

## 2. Shrnutí
Nové TypeScript jádro je funkční integrační orchestrátor (HTTP API + Stratum + modulární skeleton), ale nenahrazuje plnou funkcionalitu blockchain enginu. Legacy C++ část obsahuje reálný konsensus, mempool, cryptography, block storage, difficulty retarget, validaci transakcí.

## 3. Legacy C++ komponenty (reálné)
- Blockchain storage (`BlockchainStorage.*`)
- Mempool (`tx_pool`)
- Consensus & retarget
- Transaction validation (`check_tx_semantic`, key image duplicates prevention, overflow checks)
- Fork handling / alternative chains
- Difficulty management
- Block template generation (miner integration)
- RPC endpoints (CoreRpcServer)
- WalletLegacy + PaymentGate
- Crypto primitives (blake, groestl, jh, skein, keccak, slow-hash, tree-hash)

## 4. TypeScript jádro – aktuální stav (mock/partial)
| Modul | Stav | Poznámka |
|-------|------|----------|
| blockchain-core.ts | Mock | Height/diff fake, žádná validace, žádná persistence |
| p2p-network.ts | Mock | Žádné skutečné sockets, jen seed list v paměti |
| mining-pool.ts | Semi | Stratum implementace + multi-algo skeleton, ale share validation mock |
| rpc-adapter.ts | Mock | Vrací syntetické hodnoty, není napojen na C++ | 
| wallet-service.ts | Mock | In-memory balance, žádné klíče ani kryptografie |
| gpu-mining.ts | Mock | Žádná integrace s real hashing
| Stratum RandomX template fetch | Částečné | Pokus o RPC call na lokální daemon, fallback na dummy

## 5. Identifikované mezery
| Kategorie | Chybí / Zjednodušeno | Navržené řešení |
|-----------|----------------------|------------------|
| Konsensus | Žádný reálný PoW proces v TS vrstvě | Delegovat na legacy daemon přes RPC |
| Difficulty adjust | Fixní hodnota | Převzít z `get_info` / block header z legacy |
| Block template | Mock generátor | Použít `get_block_template` (legacy) |
| Submit block | Inkrement height | RPC `submit_block` |
| Mempool | txPoolSize counter | `get_transaction_pool` + proxy submit |
| Persistence | Žádná | Legacy DB (LMDB/level variant) |
| Peer sync | Mock peers | Neřešit v TS – spoléhat na legacy | 
| Share validation | Kontrola délky nonce | Integrovat RandomX/slow-hash (FFI / wasm / subprocess) |
| Hashing (RandomX) | Chybí | Extract knihovnu z mineru (RandomX lib) |
| Wallet | Fake balance | Proxy na wallet RPC (nebo generátor klíčů + wasm crypto) |
| Security | Bez rate limiting / auth | Přidat limiter + API keys tímem |
| Observability | Omezené logy | Přidat metrics endpoint + Prometheus |

## 6. Návrh integrační architektury
```
+----------------------+          +------------------------+
|  TypeScript Core     | <------> |  Daemon Bridge Module  | <----> (HTTP JSON-RPC)
|  (API, Pool, Front)  |          |  (RPC wrapper + cache) |         legacy daemon (C++)
+----------+-----------+          +-----------+------------+
           |                                    |
           | mining.notify / jobs               | get_block_template
           | submit share → validate hash       | submit_block
           | stats dashboard                    | get_info / get_block
```

## 7. Bridge Module (specifikace)
Soubor: `core/src/modules/daemon-bridge.ts`
Funkce:
- `getInfo()` – cache 2s
- `getBlockTemplate(walletAddr)` – direct call, fallback při chybě
- `submitBlock(blob)` – forward
- `getBlock(height|hash)` – forward
- `isAvailable()` – health check
- Interní retry/backoff

## 8. Úpravy Mining Pool
1. Při startu: pokusit se stáhnout reálný template (logovat fallback)
2. `generateNewJob()` – pokud bridge OK → mapovat:
   - blob → extrakce ntime / prevhash / height
   - difficulty → převod na target
3. `validateShare()` – přidat (stub) hashing pipeline
4. Později: RandomX binding (C++ addon) nebo child process worker

## 9. Environment Proměnné
```
EXTERNAL_DAEMON_ENABLED=true
DAEMON_RPC_URL=http://127.0.0.1:18081
TEMPLATE_WALLET=Z3.... (pool address)
TEMPLATE_RESERVE_SIZE=8
BRIDGE_TIMEOUT_MS=4000
BRIDGE_RETRY=2
```

## 10. Postup Migrace (Fáze)
| Fáze | Cíl | Deliverable |
|------|-----|-------------|
| 1 | Bridge + env + pool fallback | daemon-bridge.ts + úprava mining-pool.ts |
| 2 | Odstranit mock z blockchain-core | Napojený height/difficulty z bridge |
| 3 | Share validation (hashing) | randomx hash stub + interface |
| 4 | Wallet proxy | wallet-rpc adapter |
| 5 | RPC unify | rpc-adapter deleguje vše na bridge |
| 6 | Observability | metrics + health layering |

## 11. Rizika & Mitigace
| Riziko | Dopad | Mitigace |
|--------|-------|----------|
| Daemon down => pool padá na dummy | Nekonzistentní data | Fallback + health flag |
| Race při update jobu | Špatné shares | Job generation s atomic pointerem |
| Latence RPC | Zpoždění jobů | Cache + prefetch |
| Hashing CPU overhead | Nižší výkon | Worker pool / C++ addon |
| Divergence configů | Chybná obtížnost | Single source (.env + consensus.json) |

## 12. Rychlé vítězství (Quick Wins)
- Přidat Bridge modul (≤200 LOC)
- Upravit mining-pool fetch část (1 funkce + target mapping)
- Přidat env proměnné + dokumentaci
- Logovat: `[bridge] template OK / fallback`

## 13. Co zatím NEDĚLAT
- Nepřepisovat konsensus do TS
- Neimitovat peer network – necháme C++
- Neimplementovat hned RandomX v JS (až po stabilní bridge vrstvě)

## 14. Akční seznam (propojený do TODO)
- [ ] daemon-bridge.ts
- [ ] Úprava mining-pool.ts (template acquisition + fallback logika označená komentářem)
- [ ] Úprava .env.example
- [ ] Napojení server.ts
- [ ] Dokumentace v `MIGRATION_GAP_AUDIT.md` (doplňovat stav)

## 15. Metodika testování
1. `daemon-bridge.isAvailable()` → GREEN
2. `curl /api/rpc/get_info` → hodnoty z legacy (ne mock)
3. Stratum připojení mineru → job target odpovídá difficulty z legacy template
4. Simulace výpadku daemon → fallback log + synthetic template

## 16. Další krok
Implementace fáze 1 (Bridge + Pool fallback + Env). 

---
Autor: Migration AI Assistant
Verze reportu: 1.0
