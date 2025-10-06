# ZION 2.7 TestNet (Clean Start)

This directory is the clean, non-simulated reboot of the ZION network stack.

## Goal
Build a minimal, fully REAL (no mock / no demo) end-to-end testnet:
- Real Python blockchain core (will evolve into networked node)
- Real RandomX engine wrapper (hash fallback only if librandomx not present)
- Real Stratum pool listener (TCP 3333)
- Real wallet crypto (Ed25519 + AES-GCM) WITHOUT mock balances
- Deterministic testnet params (genesis + fixed reward schedule)

## Phase 1 Components (imported)
- core/blockchain.py (trimmed: remove demo harness if any, keep real logic)
- mining/randomx_engine.py (real wrapper)
- pool/stratum_pool.py (extracted from ai/zion_real_stratum_pool.py, cleaned)
- wallet/wallet_core.py (removed mock balance injection; require sync layer)

## Implemented (Phase 1 ✅)
1. Minimal blockchain core (genesis, mempool, block submit)
2. Persistent block storage (`data/blocks/*.json` + atomic writes)
3. Adaptive difficulty (sliding window)
4. Stratum pool integrated with real block templates
5. Basic share -> block acceptance (prefix rule)
6. RandomX wrapper (sha256 fallback)
7. Clean wallet key management (no mock balances)
8. P2P skeleton (hello + announce)
9. Unified run script (`run_node.py` with --pool / --p2p)

## Next TODO (Phase 2 → 3)
1. P2P block fetch (request missing blocks after announce)
2. Block propagation throttling + peer health
3. Replace prefix difficulty with real target computation (cumulative) 
4. Transaction format: inputs/outputs validation + fees
5. Wallet chain scanner (derive outputs, track balance)
6. Stratum: realistic difficulty / target conversion
7. Share validation: proper hash target vs difficulty mapping
8. Persistence: index (height → hash, hash → file) & compaction
9. CLI: `zion2.7 wallet create|list` & `zion2.7 submit-tx`
10. Basic test harness for end-to-end mined block

## Non-Goals (Phase 1)
- Fancy AI layers
- Cosmic narrative wrappers
- Mock dashboards

## Running
```bash
# Node only
python run_node.py

# Node + pool
python run_node.py --pool

# Node + pool + p2p (connect to peers)
python run_node.py --pool --p2p --peers 127.0.0.1:29876

# P2P only skeleton (second instance to test handshake)
python run_node.py --p2p --peers 127.0.0.1:29876 --interval 30
```

## Phase 2 Progress
Implemented:
* Block hash index (hash -> height O(1))
* Real 256-bit mining target (MAX_TARGET / difficulty) in templates
* Share validation using integer target comparison
* P2P block sync messages: inv_request, inv, getblocks, block, announce
* Basic UTXO transaction model & validation (inputs consume UTXOs, outputs create; fee check)
* Wallet scanner skeleton (full chain scan to derive balance)
* Enhanced status output (peers, pool stats, utxos, target_hex)

Upcoming / Planned:
* Incremental wallet scan + persistence of last scan height
* Transaction relay across P2P
* Reorg & fork handling (currently linear only)
* Coinbase maturity & reward schedule refinement
* Fee prioritization & mempool eviction policies
* Difficulty retarget robustness tests
* External miner protocol refinement (toward XMRig compatibility)
* Security hardening & performance profiling

## Policy
NO SIMULATION DATA. If data not available → return empty / unimplemented error.

## License
Copyright (c) 2025 ZION Network. All rights reserved.
