## ZION 2.6.5 Strict Bridge Mode

Date: 2025-09-29
Status: ACTIVE (bridge-first, no mock fallbacks)

### Purpose
Guarantee that the integrated core refuses to operate with synthetic / mock blockchain data when a real legacy daemon is required. Ensures miners and tooling do not accidentally mine on placeholder state.

### Environment Flags
| Variable | Meaning | Behavior |
|----------|---------|----------|
| EXTERNAL_DAEMON_ENABLED | Enable legacy daemon bridge | Allows HTTP JSON-RPC passthrough to C++ daemon |
| STRICT_BRIDGE_REQUIRED | Enforce real chain only | Startup aborts if daemon disabled/unavailable; mock fallbacks disabled |
| DAEMON_RPC_URL | Bridge base URL | Default points to internal docker service `http://legacy-daemon:18081` |
| TEMPLATE_WALLET | Wallet for get_block_template | Used by bridge when fetching mining templates |
| TEMPLATE_RESERVE_SIZE | Extra nonce reserve bytes | Forwarded to get_block_template |
| BRIDGE_TIMEOUT_MS | RPC timeout | Avoid long stalls; default 4000ms |

### Startup Behavior
1. If `STRICT_BRIDGE_REQUIRED=true` and `EXTERNAL_DAEMON_ENABLED!=true` → process exits (code 66).
2. If daemon enabled but not reachable → exit (code 67).
3. MiningPool: No synthetic job rotation; waits for real block template.
4. RPC Adapter: All legacy mock fallbacks (get_info, get_height, get_block, get_block_template, submit_block, get_connections) disabled – errors if bridge fails.
5. BlockchainCore: Skips mock bootstrap; only height/difficulty from daemon polling.

### New Endpoints
| Endpoint | Description |
|----------|-------------|
| GET `/api/bridge/status` | Bridge health (height, difficulty) |
| POST `/api/tx/submit` | Submit raw transaction hex via bridge `send_raw_transaction` |

### Operational Runbook
1. Ensure legacy daemon container healthy:
   - `docker ps` → container `legacy-daemon` up
   - Healthcheck hits `http://localhost:18081/json_rpc`
2. Set in `.env`:
   ```
   EXTERNAL_DAEMON_ENABLED=true
   STRICT_BRIDGE_REQUIRED=true
   DAEMON_RPC_URL=http://legacy-daemon:18081
   ```
3. Launch stack: `docker-compose up --build legacy-daemon core`
4. Verify:
   - `curl http://localhost:8602/api/bridge/status`
   - Run smoke: `./scripts/smoke-strict.sh`

### Error Codes & Messages
| Code | Source | Meaning |
|------|--------|---------|
| 66 | server constructor | Strict required but bridge disabled |
| 67 | server start | Bridge unreachable under strict |
| BRIDGE_REQUIRED_DISABLED | DaemonBridge.requireAvailable() under strict disabled |
| BRIDGE_REQUIRED_UNAVAILABLE | DaemonBridge unavailable under strict |
| STRICT_RPC_* | RPC adapter bridge failure without fallback |
| STRICT_MINING_* | Mining pool cannot start due to missing template/bridge |

### Smoke Script
`scripts/smoke-strict.sh` checks:
1. `/api/bridge/status` enabled & height > 0
2. Optional `get_block_template` JSON-RPC availability

### Migration Notes
Previous migration log stated “migration complete” before strict enforcement existed. This document supersedes that statement for production readiness: real chain integration is now a hard requirement when strict mode is enabled.

### Future Enhancements
| Area | Planned |
|------|---------|
| Failover | Optional multi-daemon list + quorum selection |
| Caching | Smarter TTL adaptive to block rate |
| Metrics | Prometheus endpoint for bridge latency/error rates |
| Security | HMAC or mTLS between core and daemon container |

### Quick Troubleshooting
| Symptom | Cause | Fix |
|---------|-------|-----|
| Exit code 66 | STRICT true, daemon not enabled | Set EXTERNAL_DAEMON_ENABLED=true |
| Exit code 67 | Daemon not reachable | Check container logs / port mapping |
| submit_block errors | Invalid blob or daemon reject | Validate hex length & daemon logs |
| get_info STRICT_RPC_GET_INFO_FAIL | Bridge timeout | Increase BRIDGE_TIMEOUT_MS or inspect daemon performance |

---
Maintainer: Core Engineering
Revision: 1