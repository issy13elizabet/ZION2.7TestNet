# ZION NEXT CHAIN SPEC (Draft v0.1)

Status: DRAFT (29 Sep 2025)
Mode: Path C – parallel new chain skeleton + finish strict bridge
Algorithm Phase: Use existing placeholders ("Cosmic Dharma" concept) – real bespoke PoW later

## 1. Scope (MVP)
- Pure PoW blockchain (no privacy layer v1)
- Transparent transactions (UTXO style)
- Deterministic genesis
- Block time target: 60s
- Difficulty algorithm: ASERT-like (simplified) placeholder
- Emission: Geometric decay with tail emission (configurable constants)
- Networking: Minimal peer gossip (handshake, inv, getdata, block, tx)
- RPC: get_info, get_block, get_block_template, submit_block
- Mining: External simple CPU miner + Stratum adapter later

## 2. Data Structures
### Block Header
```
version (u32)
height (u64)
prev_hash (32 bytes)
merkle_root (32 bytes)
timestamp (u64)
difficulty_target (u64)
nonce (u64)
pow_hash (32 bytes)  // computed, not stored inside serialized header (derived)
```
### Block Body
```
tx_count (varint)
[ transactions ... ]
```
### Transaction
```
version (u16)
lock_time (u64)
vin_count (varint)
  [ { prev_tx_hash (32b), prev_index (u32), script_sig (varbytes) } ]
vout_count (varint)
  [ { value (u64), script_pubkey (varbytes) } ]
witness_flag (u8) // reserved future
```
(Privacy layer hooks reserved: script_pubkey variants)

## 3. Hashing / IDs
- Transaction ID = BLAKE3(serialized_tx)
- Block ID = BLAKE3(serialized_header_without_pow_hash || nonce_le || difficulty_target_le)
- PoW Hash (ZIONHASH v0 placeholder) = keccak256( blake3(header || nonce) )
  - Later upgrade to Cosmic Dharma PoW (multi-phase memory & rotation scheme described in cosmic dharma logs).

## 4. Difficulty (ASERT Simplified)
Parameters:
```
TARGET_BLOCK_TIME = 60
HALF_LIFE = 300   # seconds
```
Formula each block:
```
new_diff = old_diff * 2^((actual_spacing - TARGET_BLOCK_TIME)/HALF_LIFE)
Clamp drift: timestamp in [prev_timestamp - 120, prev_timestamp + 600]
Minimum diff floor = 1000
```
Stored difficulty_target = (2^256 - 1) / diff (classic style) OR direct difficulty integer (implementation choice). For MVP: store compact target.

## 5. Emission
Constants:
```
INITIAL_REWARD = 32 * 10^12   # atomic units
DECAY_INTERVAL_BLOCKS = 105000
DECAY_FACTOR = 0.90
TAIL_EMISSION = 1 * 10^12
```
Reward(height): apply floor( INITIAL_REWARD * (DECAY_FACTOR)^(height / DECAY_INTERVAL_BLOCKS) ) but not below TAIL_EMISSION.

## 6. Genesis
Deterministic generation script will:
- Set height=0, prev_hash=0x00..00
- Coinbase TX paying INITIAL_REWARD to placeholder foundation address `Z3GENESISPLACEHOLDER...`
- Compute merkle_root = BLAKE3(coinbase_tx_id)
- Choose fixed timestamp (e.g. 2025-10-01T00:00:00Z)
- Derive nonce by searching first nonce such that keccak256(blake3(header||nonce)) < initial target
Export JSON + C++ header with values.

## 7. Networking (Phase 1)
Messages (framed length-prefix):
```
HANDSHAKE { version, node_id, height, top_hash }
INV { type (0=tx,1=block), ids[] }
GETDATA { type, ids[] }
TX { raw_tx }
BLOCK { raw_block }
PING { nonce }
PONG { nonce }
```
No encryption (future TLS/noise). Basic peer banning for malformed payload.

## 8. RPC (HTTP JSON)
Endpoints:
```
GET /info -> { height, difficulty, hash, peers, mempool, emission }
POST /json_rpc { method: get_block_template } -> { blocktemplate_blob, difficulty, height, target }
POST /json_rpc { method: submit_block, params:[hex] }
POST /json_rpc { method: get_block, params:{height} }
```
Block template blob = serialized header + placeholder extra nonce field region.

## 9. Cosmic Dharma PoW Placeholder Hooks
Reserve interface:
```
interface IPoW {
  init(seed: bytes32, height: u64): void;
  hash(header_without_nonce: bytes, nonce: u64): bytes32;
}
```
Current impl: SimpleKeccakPoW implements IPoW.
Planned Cosmic Dharma (phase upgrade):
1. Seeded memory lattice (size param per epoch)
2. Non-linear rotations using prime wheel schedule from `cosmic_dharma` log
3. Mixed rounds (BLAKE3 -> AES-like -> Keccak final)
4. Deterministic epoch re-seed every EPOCH_BLOCKS = 2048

## 10. Directory Skeleton (planned)
```
zion-next-chain/
  README.md
  LICENSE
  package.json (tools + scripts) or CMakeLists.txt (if C++)
  src/
    core/block.ts
    core/blockchain.ts
    core/tx.ts
    core/mempool.ts
    consensus/pow/simple_pow.ts
    consensus/difficulty/asert.ts
    rpc/server.ts
    p2p/node.ts
  scripts/genesis.ts
  tests/
    pow.test.ts
    difficulty.test.ts
```

## 11. Upgrade / Governance Flags
- Version field increments when consensus changes
- Future on-chain signaling reserved bits in header.version upper nibble

## 12. Security Considerations
- Timestamp sanity window
- Nonce space 64-bit (adequate for GPU miners before refilling template)
- DOS: limit tx size, block size soft limit (1 MB initial), reject >2 MB
- Validation pipeline order: header -> pow -> merkle -> tx structure -> tx inputs (Phase 2 add UTXO checks)

## 13. Test Strategy
- Unit: PoW hash determinism
- Unit: Difficulty retarget monotonic adaptation up/down
- Unit: Genesis reproducibility (hash stability)
- Integration: Mine 20 blocks locally, assert height progression & reward schedule

## 14. Open Questions
- Exact base58 address encoding spec alignment with current `ADDRESS_SPEC.md`
- Decide if we adopt ed25519 or secp256k1 for initial key scheme (LOG suggests ed25519 synergy with cosmic-layer). Pending decision.
- Fee model: flat per byte vs dynamic (placeholder: fee_per_kb = 10^9 atomic units)

## 15. Next Actions
1. Confirm key type & address format reuse (respond in issue / chat)
2. Implement skeleton (see TODO 11)
3. Write genesis generator (TODO 12)
4. Integrate simple miner script
5. Launch local test harness (20-block mining)

---
(End of Draft)
