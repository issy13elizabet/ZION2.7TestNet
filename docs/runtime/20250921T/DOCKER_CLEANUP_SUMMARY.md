# Docker Cleanup Summary (macOS frontend-only)

Date: 2025-09-21
Host: macOS

## Actions Performed

- Stopped and removed all Zion containers:
  - zion-rpc-shim, zion-seed1, zion-seed2, zion-walletd, zion-pool, zion-pool-nodejs, zion-wallet-adapter, zion-redis, zion-uzi-pool
- Removed Zion images:
  - zion:production-fixed, zion:rpc-shim, zion:wallet-adapter, zion:uzi-pool, zion:pool-nodejs
- Removed project volumes and zion-seeds network
- Optional prune available via scripts/macos-clean-frontend-only.sh --prune (not executed in this run)

## Resulting Docker State

- Zion containers: none
- Zion images: none
- Zion volumes: none

## Docker System DF

```
TYPE            TOTAL     ACTIVE    SIZE      RECLAIMABLE
Images          11        9         1.452GB   620.5MB (42%)
Containers      35        18        831.5kB   413.7kB (49%)
Local Volumes   0         0         0B        0B
Build Cache     0         0         0B        0B
```

## Notes
- Frontend-only workflow is now ready. Use scripts/frontend-dev.sh to run Next.js against the Ryzen backend.
- To re-clean in the future: scripts/macos-clean-frontend-only.sh
