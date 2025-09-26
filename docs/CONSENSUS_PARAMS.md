# Zion Consensus Parameters

Status: Draft v0.1 (2025-09-26)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Symbol | ZION |  |
| Max Supply | 144,000,000,000 | 6 decimals (assumed) |
| Block Target Time | 120 s | Average intended interval |
| Initial Block Reward | 333 ZION | Pre-emission TBD |
| Emission Curve | Halving every 210,000 blocks | Approx ~292 days at 120s |
| Difficulty Algorithm | LWMA (assumed) + RandomX seed schedule | Verify in core code |
| PoW Algorithm | RandomX (variant rx/0) | No custom tweak yet |
| P2P Port | 18080 | Default in compose |
| RPC Port | 18081 | Default in compose |
| Shim RPC Port | 18089 | Internal Monero-like proxy |
| Stratum Port | 3333 | uzi-pool |
| Mempool TTL | TBD | Add after code audit |
| Coinbase Maturity | TBD (e.g. 60 blocks?) | Needed for payout validation |
| Min Fee | TBD | Extract from core |
| Address Prefix | 'Z3' | Base58 start |
| Extra Nonce Space | >= 16 bytes | Ensures multi-miner scaling |

## To Verify in Core
- Real emission formula and tail emission if any
- Difficulty window & target blocks for adjustment
- Coinbase maturity constant
- Min fee calculation method (static vs dynamic)
- Block size growth / penalty rules

## Block Template Flow
1. Pool -> rpc-shim: `getblocktemplate`
2. Shim caches template (default 12s)
3. Miner solves RandomX hash meets difficulty
4. Share -> pool -> (candidate) -> `submitblock`
5. On success: template cache invalidated

## Future Additions
- Tail emission activation height
- Soft fork upgrade heights table
- Version bits / feature flags description

---
Update this document whenever protocol-level constants change. A rendered version should be kept stable for explorers & tooling.
