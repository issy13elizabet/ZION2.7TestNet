# RANDOMX INTEGRATION PLAN (ZION NEXT CHAIN)

## Cíl
Zajistit, aby nový PoW rámec podporoval CPU-friendly RandomX (kompatibilní s mainstream CPU těžbou), zatímco se vyvíjí proprietární Cosmic Dharma algoritmus. RandomX bude používaný jako primární / fallback fáze pro rané testnety, dokud nebude Cosmic Dharma bezpečně auditován.

## Strategie Více Algoritmů
| Vrstva | Popis | Stav |
|--------|-------|------|
| pow/index.ts | Abstraktní výběr algoritmu podle height / epoch / config | ČÁSTEČNĚ (routing + stub) |
| randomx_pow.ts | Wrapper k nat. knihovně (C++ / WASM fallback) | STUB PŘED IMPLEMENTACÍ |
| composite (blake3+keccak) | Aktuální placeholder pro genesis a testy | HOTOVO |
| cosmic_pow.ts | Experimentální Phase A (memory-hard) | PROTOTYP |

## Fáze Implementace
1. Abstrakce API (ts interface):
   ```ts
   interface PowAlgorithm {
     name: string;
     init?(opts: any): Promise<void> | void;
     hash(headerBytes: Uint8Array, nonce: bigint, context: PowContext): string;
     verify(headerBytes: Uint8Array, nonce: bigint, target: bigint, context: PowContext): boolean;
   }
   interface PowContext { height: number; prevSeed?: string; epoch?: number; }
   ```
2. Konfig řízení: env / config soubor:
   - `POW_MODE=RANDOMX|COMPOSITE|COSMIC|HYBRID`
   - `POW_HYBRID_SWITCH_HEIGHT=<n>` (přepnutí z RandomX na Cosmic)
3. RandomX modul:
   - Minimální TypeScript wrapper: dynamicky načte nativní modul `randomx.node` (není-li, použije WASM fallback → low performance, jen test).
   - Seed per epoch (např. každých 2048 bloků: seed = blake3(hashPrevEpochSeed || epoch || chainId)).
   - Cache dataset (Full vs Light mode). V testnetu stačí Light dataset.
4. Integrace do genesis/miningu: genesis stále composite (deterministické), block >0 pokud POW_MODE=RANDOMX → RandomX hash.
5. Validátor: rozhoduje podle height a config který modul použít.
6. Testy: 
   - Mock dataset (zmenšený) pro CI.
   - Ověření determinismu seed derivace.
7. Performance / Benchmark: skript `scripts/bench-pow.ts` – měří H/s pro RandomX vs Composite vs Cosmic.

## RandomX Knihovna & Build
- Preferovaný upstream: https://github.com/tevador/RandomX
- Build kroky (native addon):
  - CMake build → `librandomx.a`
  - Node-API wrapper (bindings) → `randomx.node`
- Alternativa: WASM port (pomalejší) pro snadné CI / bez nativního toolchainu.

## Bezpečnostní Poznámky
- RandomX dataset memory footprint: Full (~2GB) / Light (~256MB); testnet běží Light.
- CPU fingerprinting minimalizovat – volitelně přidat "--soft-mode" bez agresivní detekce.
- Po přechodu na Cosmic Dharma: Dual-submit režim (akceptovat oba hashe po přechodové období).

## Hybridní Režim (Budoucí)
| Výška | Algoritmus | Motivace |
|-------|------------|----------|
| 0 - Hswitch | RandomX Light | rychlý onboarding CPU minerů |
| Hswitch - Hfinal | Hybrid (RandomX + Cosmic) – must satisfy both | postupný rollout |
| > Hfinal | Cosmic Dharma v1 | plný přechod |

## Minimální Požadavky Testnet (Milestone)
- POW_MODE=RANDOMX funkční s Light dataset.
- 1 blok / 60s ± tolerance s dynamickou retarget funkcí (ASERT adaptace pro RandomX variance).
- Validace nezávislá na lokální dataset regeneraci (deterministický seed + caching hash).
- Benchmark: `npm run bench:pow` pro srovnání COMPOSITE / COSMIC / RANDOMX (stub) – cílem je sledovat regresi výkonu.

## Task Breakdown
1. pow/index.ts + interface (routing)  ✅ (základ hotov)
2. randomx_pow.ts stub (bez nativního kódu)  ✅ (zatím hash = tagovaný composite)
3. Konfig env integration  ⏳
4. Validator refactor na interface  ⏳
5. Genesis unaffected (jen doc update)  ⏳
6. Seed schedule + test  ⏳
7. Native binding research doc  ⏳
8. Benchmark skript `bench-pow.ts`  ✅

## Seed Derivace (Návrh)
```
seed_0 = blake3("ZION-NEXT-RANDOMX-SEED-0")
seed_epoch_n = blake3(seed_epoch_{n-1} || u64le(n) || network_id)
```
Epoch = floor(height / EPOCH_BLOCKS), navrh EPOCH_BLOCKS = 2048 (laditelné).

## Možné Riziko
- Komplexní build pipeline (C++ addon) – navrhnout fallback (WASM) aby CI neblokovalo merge.
- Výkon WASM může zkreslit testy (nutné označit). 

## Další Dokumenty (navázat později)
- `POW_SPEC.md` sjednotí composite + RandomX + Cosmic.
- `EMISSION_SPEC.md` pro návaznost odměn vs difficulty adaptace.

---
Dokument vytvořen 2025-09-29 automatizovaným asistentem.
