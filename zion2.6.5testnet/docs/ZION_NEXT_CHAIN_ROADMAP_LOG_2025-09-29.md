# ZION NEXT CHAIN ROADMAP LOG (2025-09-29)

Tento log zachycuje plán vývoje nové experimentální "Zion Next Chain" vrstvy, paralelně se strict bridge režimem legacy sítě.

## Kontext
- Legacy CryptoNote větev běží přes daemon bridge ve strict režimu (žádné mocky).
- Nový chain (TypeScript prototyp) má: genesis nástroj (blake3+keccak prefix PoW), základní strukturu bloku, placeholder PoW koncept Cosmic Dharma.
- Cíl: dosažení minimálně použitelného MVP (validace bloků, P2P sync, simple mining, transakce) během několika týdnů.

## FÁZE 0 – Základní konsolidace
- Nahradit prefix target číselným difficulty (hash(BigInt) < target) – ASERT parametry.
- Merkle root: zatím hash(coinbase); připravit vícetx variantu.
- Header rozšířit: version, flags (rezervováno), future consensus bits.
- Testy: deterministický genesis + pow hash re-verifikace.

## FÁZE 1 – Transakční & UTXO vrstva
- Tx formát: version, vin[], vout[], lockTime, extra.
- Adresy: rozhodnout (Base58Check vs bech32 derivát) – TODO DECISION.
- UTXO in-memory + abstrakce pro budoucí LevelDB.
- Coinbase maturity (např. 60 bloků).
- Validace pořadí: header → difficulty → timestamp → merkle → tx sanity → UTXO apply.

## FÁZE 2 – P2P Skeleton
- PeerManager (OUTBOUND/INBOUND sloty, handshake).
- Messages: version, inv, getdata, block, tx, ping/pong.
- Basic anti-flood (rate limit, max inflight).
- Chain sync: simple longest-chain (tie -> lowest work hash compare).

## FÁZE 3 – Persistence
- LevelDB (chainstate + blocks + indexes).
- Indexy: height->hash, txid->{height, index}.
- Reindex režim.

## FÁZE 4 – Cosmic Dharma PoW v1
- Přechod z composite na memory-hard fázi A (parametrování paměť/iterace).
- Epoch seed = hash(parentHash || epochNumber).
- Referenční JS + plán Rust/WASM modulu.
- RandomX podpora: hybrid režim (počáteční CPU onboarding) – viz `RANDOMX_INTEGRATION_PLAN.md`.

## FÁZE 5 – Mining & Pool integrace
- Template builder (mempool -> merkle -> header skeleton).
- Stratum-lite server pro test mining.
- Share difficulty < block difficulty validace.

## FÁZE 6 – Mempool
- Fee prioritizace + eviction politika.
- Orphan tx set.
- Double-spend konflikty (optimistic reject).

## FÁZE 7 – RPC / API
- get_block, get_block_header, get_block_template, submit_block.
- get_tx, send_tx, chain_info.
- Explorer light endpoints (difficulty, mempool size, supply, height).

## FÁZE 8 – Test & QA
- Genesis reproducibility test.
- Randomized block/tx simulation (200+ bloků).
- Fork simulation (dvě větve, delší chain wins). 
- Fuzz parsování bloků/tx.

## FÁZE 9 – Observabilita & DevOps
- /healthz (peerCount, height, syncing flag).
- Prometheus metrics: block_interval, peer_count, mempool_tx_count.
- Docker image build + minimal run script.

## FÁZE 10 – Hardening & Security
- Timestamp drift limit (+/- X s).
- Max block size, max tx size enforcement.
- Basic DoS guard (max orphan tx, per-peer rate limit).

## FÁZE 11 – Wallet MVP
- CLI: generate address, list balance (scan chain), send.
- Seed standard: BIP39 nebo vlastní? (DECISION)
- Později: light client proofs / merkle branch export.

## FÁZE 12 – Dokumentace
- CONSENSUS_SPEC.md (hashing, header, difficulty, emission).
- TX_FORMAT.md.
- P2P_PROTOCOL.md.
- ARCHITECTURE_OVERVIEW.md.
- CHANGELOG sekce pro chain evoluci.

## Milníky (návrh)
| Týden | Cíle |
|-------|------|
| 1 | Fáze 0 + část Fáze 1 (UTXO draft) |
| 2 | Dokončení Fáze 1 + P2P skeleton (F2) |
| 3 | Persistence + PoW v1 (F3+F4) |
| 4 | Mempool + RPC (F6+F7) |
| 5 | Observabilita + Wallet MVP (F9+F11) |
| 6 | Hardening + Dokumentace (F10+F12) |

## Quick Wins (proveditelné okamžitě)
- Přidat jest + test na genesis verifikaci.
- Logger modul (structured JSON, levels: debug/info/warn/error).
- Centralizovaný config (consensus constants, net params).

## Decision Items (otevřené)
| Téma | Možnosti | Stav |
|------|----------|------|
| Adresní formát | Base58Check, bech32 variant | OPEN |
| Emisní křivka | Lineární, exponenciální decay, tail emission | OPEN |
| Block interval | 30s / 60s / 120s | OPEN |
| PoW přepis do Rust/WASM | Po PoW v1 / odložit | OPEN |
| Seed standard | BIP39 vs vlastní deterministický | OPEN |

## Aktuální Stav (2025-09-29)
- Genesis: hotovo (numeric target + blake3+keccak, artefakty + verify script).
- Numeric difficulty: IMPLEMENTED (hash BigInt <= target, modul `consensus/params.ts`).
- Test harness: základní jest přidán (`genesis.spec.ts`).
- Dokumentace: README doplněno o genesis návod.
- Fáze 1 zahájena: TX typy + deterministická serializace + UTXO in-memory (`utxo/utxoSet.ts`).
- Block validator přidán (`core/blockValidator.ts`) – kontrola coinbase, merkle root, PoW, základní UTXO apply (shadow).
- Testy: `utxo.spec.ts`, `blockValidator.spec.ts` (merkle tamper, multi-coinbase, valid genesis).

## Doporučený Další Krok
Začít Fází 0 → implementovat numeric difficulty + přidat `jest` test `genesis.spec.ts` k ověření determinismu.

---
_Log generován automatizovaným asistentem 2025-09-29._
