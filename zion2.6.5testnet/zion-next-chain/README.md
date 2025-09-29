# ZION NEXT CHAIN (Skeleton)

Status: Experimental skeleton for new PoW chain (Path C)
Spec: See `../docs/ZION_NEXT_SPEC.md`

## Goals
- Minimal clean codebase for iterative consensus development
- Deterministic genesis & pluggable PoW
- Fast iteration before privacy + advanced features

## Quick Dev Start (after future package.json)
```
node scripts/genesis.js   # (pending)
node src/rpc/server.js    # (placeholder future)
```

## Layout
```
src/
  core/        # block, blockchain state, mempool (WIP)
  consensus/   # pow + difficulty
  rpc/         # http/json-rpc server
```

## Next Steps
1. Implement block & tx serialization
2. Add simple PoW (keccak(blake3(...)))
3. Difficulty (ASERT simplified)
4. Genesis generator script
5. Local mining harness

---
(Work in progress)
