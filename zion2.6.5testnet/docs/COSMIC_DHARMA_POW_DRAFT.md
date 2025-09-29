# COSMIC DHARMA PoW (Concept Draft v0.1)

Source Inspiration: `ZION_COSMIC_HARMONY_ALGORITHM_2025.md` & related victory / harmony logs.
Status: Draft – NOT IMPLEMENTED. This file specifies the target design that will replace the current simple PoW placeholder in `zion-next-chain`.

## Design Goals
1. ASIC Friction (18–24 month window) via memory hardness + mixed primitives.
2. Deterministic epochal evolution ("dharma cycles") so hardware cannot pre-compute across long horizons.
3. Verifier simplicity (single pass, O(N) memory read but low constant factors).
4. Algorithmic transparency + parameter upgradability (chain governance or version bits).

## High-Level Flow
```
Input: block_header_without_nonce (H), nonce (N), height (h), prev_epoch_seed (S_prev)
Output: 32-byte hash (compare ≤ target)

1. Seed Derivation:
   S = BLAKE3( H || N || h_le || S_prev )

2. Epoch Parameters (every EPOCH_BLOCKS):
   epoch_index = floor(h / EPOCH_BLOCKS)
   mem_size = BASE_MEM * growth_fn(epoch_index)
   rounds = BASE_ROUNDS + epoch_index % ROUND_MOD

3. Lattice Memory Fill (Pseudo-Random Matrix):
   M[0..mem_size-1] 64-bit words
   M[i] = FNV64( S xor i xor ROTR64(S, i % 63) )
   (Streaming: builder can discard tail; full array only needed for random read phase.)

4. Dharma Walk (Random Access Mixing):
   idx = lower_bits(S)
   acc = S
   For r in 0..rounds-1:
     x = M[idx]
     acc = Mix(acc, x, r)
     idx = ( (x >> (r % 17)) xor acc ) % mem_size

   Where Mix(a,b,r):
     t = a xor ROTL64(b, r % 61)
     if (r % 3 == 0): t = t + (a * 0x9e3779b185ebca87)
     if (r % 5 == 0): t = BLAKE3_CHUNK(t_bytes)  (reduced round flavor)
     if (r % 7 == 0): t = AES_SIM_ROUND(t, a)    (software S-box substitution)
     return t

5. Spiral Compression:
   spiral = Keccak256( acc || S || encode_u64(rounds) )
   cosmic  = BLAKE3( spiral || acc )

6. Final Hash = Keccak256( cosmic )
```

## Parameter Table (Initial Draft)
| Name | Value | Rationale |
|------|-------|-----------|
| EPOCH_BLOCKS | 2048 | Frequent but not too fast parameter churn |
| BASE_MEM | 32 MiB (4M 64-bit words) | Fits modern GPUs & higher-end CPUs |
| growth_fn | mem_size = BASE_MEM * (1 + epoch_index mod 4) | Periodic modulation |
| BASE_ROUNDS | 64 | Lower bound for security mixing |
| ROUND_MOD | 97 | Prime-ish variability spread |
| Reduced BLAKE3 rounds | 4 | Partial – keeps cost moderate |
| AES_SIM type | Simple S-box + MixColumns imitation in software | Anti-ASIC complexity |

## Security Considerations
- Full memory generation vs on-demand: We use streaming initial fill + random walk to defeat simple partial caching (needs >80% memory to avoid large penalty).
- Epoch seed ensures pre-computation not viable beyond next epoch.
- Mixed hash primitives diversification: Keccak + BLAKE3 + arithmetic/FNV.

## Verification Cost
Rough estimate (placeholder until benchmark):
- Memory fill: O(mem_size)
- Random walk: O(rounds)
Optimization: miners can reuse memory for attempts with same header & epoch; only seed diff tweaks portion.

## Upgrade / Versioning
- Header.version high bits encode POW_VERSION.
- POW_VERSION increment triggers clients to swap to new rule set.

## Minimal Initial Implementation Plan
Phase A (prototype):
- Implement memory fill + random walk with only FNV + rotations.
- Use Keccak256 at end.

Phase B:
- Add BLAKE3 and AES_SIM toggles.

Phase C:
- Add epochal growth & variability.

Phase D:
- Integrate into miner + chain validation; add benchmarks.

## Pseudocode (Phase A Minimal)
```
function cosmic_dharma_pow(header_bytes, nonce, height, prev_seed):
  S = blake3(header_bytes || u64le(nonce) || u64le(height) || prev_seed)
  mem_size = 32 MiB / 8
  M = array[mem_size]
  for i in 0..mem_size-1:
    M[i] = fnv64(S xor i)
  idx = S & (mem_size-1)
  acc = S
  for r in 0..63:
    x = M[idx]
    acc = (acc xor rotl(x, r % 61)) + (acc * 0x9e3779b185ebca87)
    idx = (Number(x ^ acc) & 0xffffffff) % mem_size
  h1 = keccak256(le_bytes(acc) || le_bytes(S))
  return keccak256(h1)
```

## Open Questions
- Exact memory hardness target vs CPU friendliness trade-off.
- Add optional lightweight verification mode? (Probably no for PoW fairness.)
- Parameter governance (on-chain vote vs static for 1 year).

## Next Steps
1. Implement Phase A in `consensus/pow/cosmic_pow.ts` (TODO)
2. Benchmark on CPU + GPU
3. Integrate target comparison & template generation
4. Decide final constants after micro-benchmarks

---
Draft maintained by Core Engineering.
