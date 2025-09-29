# Zion Repository Migration Plan (2.6 → 2.6.5)

Date: 2025-09-29
Owner: Migration Working Thread
Status: DRAFT (Ready for execution once approved)

## 1. Objective
Create a clean, maintainable monorepo for Zion 2.6.5 with integrated services:
- Core runtime
- Integrated seed nodes (seed1, seed2)
- Integrated pool (Stratum) service
- Wallet adapter (unified API layer)
- RPC facade (no fragmented shim layers unless needed for backward compatibility)
- Miner tooling (v1.4.x)

Legacy (2.6 and earlier ad‑hoc stacks, C++ daemon v1.0.0, scattered Docker variants) will be archived for historical reference and reproducibility but not shipped forward.

## 2. High-Level Strategy
1. Freeze current 2.6 repo (tag `v2.6-backup-pre-migration`).
2. Classify components: KEEP → migrate / LEGACY → archive / MERGE → refactor into core.
3. Stand up new 2.6.5 repository skeleton (minimal, strongly opinionated layout).
4. Incrementally port only necessary code & configs; no blind copying.
5. Add automated bootstrap script to generate a functional local testnet (multi-seed + pool + wallet API + mining two blocks) in < 90s.
6. Embed single authoritative version file (`VERSION`) used by core + miner build pipelines.

## 3. Component Classification

| Domain | Current Location(s) | Action (2.6.5) | Notes |
|--------|---------------------|----------------|-------|
| Node/TS Core (RPC, P2P, blockchain modules) | `zion-core/` | KEEP → `core/` | Becomes orchestrator + service container |
| Integrated Pool (Stratum logic if present in TS) | `zion-core/modules/mining-pool.js` (and related) | MERGE / CONSOLIDATE | Replace external UZI adapter; expose /stratum & /stats |
| Wallet Adapter | `adapters/wallet-adapter/` | MERGE INTO core `wallet/` service | Single API gateway (REST+JSON-RPC) |
| RPC Shim (Node proxy) | `adapters/zion-rpc-shim/` | DEPRECATE (unless specific translation is still needed) | Remove indirection once core exposes stable API |
| External Pool Config (UZI) | `adapters/uzi-pool-config/` | ARCHIVE under `legacy/pool/` | Only reuse payout schema if needed |
| C++ Daemon (v1.0.0 banner) | C++ sources under root / `src/daemon` / Docker cryptonote | ARCHIVE → `legacy/daemon-cpp/` | Do not run in 2.6.5 stack |
| Miner 1.4.0 | `zion-miner-1.4.0/` | KEEP → `miner/` | Add version sync with root VERSION |
| Miner 1.3.0 | `zion-miner-1.3.0/` | ARCHIVE | No build in CI |
| Multi-platform build scripts | `zion-multi-platform/` | REVIEW → Keep only portable parts | Simplify to `scripts/build-miner.sh` |
| Dockerfiles (cryptonote*, runtime-local, minimal, prod variants) | `docker/` | CONSOLIDATE → `infra/docker/` (3 files) | `Dockerfile.core`, `Dockerfile.miner`, `Dockerfile.pool` (if still external) |
| Compose variants (`compose.*.yml`) | `docker/` root & others | REDUCE to 3 | `docker-compose.yml`, `docker-compose.dev.yml`, `docker-compose.lightning.yml` |
| Genesis / network config | `config/`, scattered JSON | UNIFY → `config/network/` | `genesis.json`, `consensus.json`, `pool.json` separated |
| Mining deployment scripts | `mining/` | SPLIT: active → `scripts/`, legacy → `legacy/mining/` | Keep only SSH deploy + bootstrap |
| Documentation (whitepapers, strategy, logs) | `docs/` | CURATE | Active guides vs historical archives in `docs/legacy/` |
| Session & audit logs | `docs/*LOG*`, `docs/sessions/` | ARCHIVE | Keep last 2 active for context |
| Lightning integration | `lightning/`, compose file | OPTIONAL MODULE | Gate by feature flag / separate compose override |
| Swap service | `swap-service/` | EVALUATE | Include only if maintained; else archive |
| Bridge / oracles / experimental dirs | `bridge/`, `oracles/`, `quantum/`, etc. | ARCHIVE (Phase 2) | Re-import selectively when stable |

## 4. New Repository Skeleton (2.6.5)
```text
zion-network-2.6.5/
  VERSION
  docker-compose.yml
  docker-compose.dev.yml
  infra/
    docker/
      Dockerfile.core
      Dockerfile.miner
      Dockerfile.pool   (optional if externalized)
    compose/
      lightning.override.yml (optional)
  core/
    package.json
    src/
      blockchain/
      p2p/
      rpc/
      pool/          # integrated stratum
      wallet/        # adapter merged
      seeds/         # seed1, seed2 config + launcher
  miner/
    CMakeLists.txt
    src/
  config/
    network/
      genesis.json
      consensus.json
      pool.json
    env/
      .env.example
  scripts/
    bootstrap-testnet.sh
    build-core.sh
    build-miner.sh
    mine-internal.sh
    verify-version-sync.sh
  docs/
    ARCHITECTURE.md
    CONSENSUS_PARAMS.md
    MINING_GUIDE.md
    DOCKER_GUIDE.md
    MIGRATION_LEGACY.md
    BOOTSTRAP_TESTNET.md
    RPC_API_REFERENCE.md
    legacy/ (archives)
  tests/
    integration/
      test_bootstrap.sh
  legacy/            # (populated only if absolutely required)
```

## 5. Migration Phases

| Phase | Goal | Deliverables | Exit Criteria |
|-------|------|--------------|----------------|
| 0 | Freeze & Backup | Tag `v2.6-backup-pre-migration` | Tag exists, push verified |
| 1 | Skeleton Create | Empty new repo scaffold + VERSION | CI passes lint/build skeleton |
| 2 | Core Import | Port `zion-core` modules (only required code) | `core` builds & runs `/get_info` |
| 3 | Pool Integration | Move TS pool logic or refactor from UZI config | `/stratum` listening; dummy template served |
| 4 | Wallet Merge | Integrate wallet adapter routes inside core RPC | Single API server running |
| 5 | Miner Align | Import miner 1.4.0, wire VERSION sync | `miner --version` shows 2.6.5 |
| 6 | Config Unification | Place consensus + genesis + pool config | `bootstrap-testnet.sh` mines 2 blocks |
| 7 | CI & Docs | Add minimal GitHub Actions + required docs | CI green, docs index complete |
| 8 | Cleanup Legacy | Archive unused modules in old repo + README pointer | Old repo has `MIGRATED.md` |
| 9 | Release | Tag `v2.6.5` + publish artifacts (docker images) | Images available & hash logged |

## 6. Bootstrap Testnet Flow (Target Script Behavior)
```bash
./scripts/bootstrap-testnet.sh \
  --seeds 2 \
  --mine-blocks 10 \
  --wallet-address Z3... \
  --stratum-port 3333

Steps:
1. Build core + miner images
2. Start seed1 + seed2 (shared genesis)
3. Wait for readiness (/healthz)
4. Start internal miner attached to seed1 until height >= N
5. Start integrated pool service (reads blocktemplate from core)
6. Display summary (height, peers, difficulty, active miners)
```

## 7. Version Synchronization
Single file: `VERSION` → e.g. `2.6.5`
| Consumer | Mechanism |
|----------|-----------|
| Core (Node) | `postinstall` script reads file & updates package.json if mismatch |
| Miner (CMake) | `configure_file(VERSION → version.h)` during configure |
| Docker labels | `LABEL org.zion.version` injected via build arg |
| Scripts | `scripts/verify-version-sync.sh` diff check |

## 8. Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Hidden dependency on C++ daemon internals | Broken runtime after removal | Keep daemon in `legacy/` for 1 cycle; instrumentation logs when features not yet ported |
| Stratum integration delays | Miners blocked | Provide temporary compatibility adapter using existing UZI logic behind feature flag |
| Config drift | Inconsistent startup | Central schema + validation step in bootstrap script |
| Build reproducibility gap | CI failures | Lock versions in `package-lock.json` & CMake toolchain notes |
| Over-pruning docs | Loss of tribal knowledge | Archive everything under `docs/legacy/` (no deletion) |

## 9. Rollback Strategy
If 2.6.5 migration blocks operations (>4h downtime):
1. Re-enable old repo + infrastructure using tag `v2.6-backup-pre-migration`.
2. Publish notice in `README` of new repo about rollback.
3. Diff new core changes to identify blocker, patch forward in isolated branch.

## 10. Approval Checklist (Before Executing Phase 2)
- [ ] This document reviewed & approved.
- [ ] New repo name confirmed (e.g. `zion-network-2.6.5`).
- [ ] Ownership & permissions prepared.
- [ ] Tag `v2.6-backup-pre-migration` pushed.
- [ ] Genesis / consensus configs exported & validated SHA256.
- [ ] Wallet address for bootstrap funded/verified (if required).

## 11. Execution Commands (Example Snippets)
```bash
# Tag backup
git tag v2.6-backup-pre-migration
git push origin v2.6-backup-pre-migration

# Create new repo skeleton (local)
mkdir zion-network-2.6.5 && cd zion-network-2.6.5
echo 2.6.5 > VERSION
mkdir -p core/src miner/src infra/docker config/network scripts docs tests

# Sync miner version via CMake (later)
cmake -DPROJECT_VERSION=$(cat ../VERSION) ...
```

## 12. Open Questions Requiring Clarification
1. Is Node/TS core authoritative for consensus state, or does it wrap a lower-level engine still in C++? (Defines how much C++ must be ported / embedded.)
2. Do we require multi-algo placeholders (kawpow/ethash) for roadmap or can we strip to RandomX + Cosmic Harmony only in 2.6.5?
3. Should integrated pool support variable difficulty (vardiff) from day 1, or can initial release fix baseline difficulty?
4. Are Lightning & swap services part of the near-term (2.6.5) release scope or scheduled for 2.7.x?
5. Will we keep RPC compatibility with existing external tools (naming: `/get_info`, `getblocktemplate`) or introduce versioned namespace (e.g. `/v1/`)?

## 13. Next Immediate Actions (Pending Approval)
1. Produce `VERSION` file in current repo (pre-step) → ensure miner & core sync approach works.
2. Prepare `MIGRATED.md` template for old repo (to commit after new repo public announcement).
3. Draft bootstrap script outline (skeleton) for early review.

---
END OF PLAN – Awaiting approval to begin Phase 0 & 1.
# Migration Plan: Zion 2.6  → Clean Zion 2.6.5 Repository

Prepared: 2025-09-29
Status: Draft (Execution Ready)
Owner: Migration Lead

## 1. Objectives
- Produce a **clean authoritative repository** for version `2.6.5` (Core + Miner + Pool adapter + Infra) without legacy noise.
- Preserve **full Git history** for the components that matter.
- Archive current repo as a frozen `2.6` backup (tag + branch + read-only note).
- Enable deterministic testnet bootstrap (first blocks mined automatically) in new repo.
- Reduce onboarding friction (single docker-compose, single .env schema, clear docs).

## 2. High-Level Strategy
| Phase | Goal | Output |
|-------|------|--------|
| 0 | Freeze current 2.6 | Tag & branch `archive/2.6` + backup note | 
| 1 | Classify content | Inclusion/Exclusion manifest | 
| 2 | Extract with history | New repo with filtered history (filter-repo / subtree) |
| 3 | Normalize structure | New directory layout + updated paths |
| 4 | Version alignment | Root `VERSION`=2.6.5 + synchronized builds |
| 5 | Infra & Docs refresh | Minimal compose + curated docs |
| 6 | Testnet bootstrap | Script mines first N blocks + health check |
| 7 | CI baseline | Build + lint + /get_info smoke |
| 8 | Decommission legacy | Mark old repo read-only (or warn in README) |

## 3. Component Classification
### 3.1 KEEP / MIGRATE (active code)
- `zion-core/` (Node/TS RPC + blockchain orchestration)
- `zion-miner-1.4.0/` (rename to `miner/`)
- `adapters/uzi-pool-config/` (merge into `pool/config/`)
- `adapters/wallet-adapter/` (keep if still required for payouts)
- `adapters/zion-rpc-shim/` (temporarily keep → aim to fold into core later)
- `pool/` (if separate implementation present / align with adapter naming)
- `bridge/` (only if required in live architecture; otherwise stage for Phase 2 optional)
- `config/` (prune → keep only consensus/network/pool JSON)
- `scripts/` (hosting build/deploy logic; prune duplicates)
- `docker-compose.yml` (will be rewritten)
- `docker/` selective Dockerfiles (core, miner, pool)

### 3.2 ARCHIVE (legacy / noise)
- `zion-miner-1.3.0/`
- All `docker/Dockerfile.zion-cryptonote*`
- `docker/Dockerfile.xmrig` (unless still used for alt PoC)
- `mining/cpuminer-opt/` (legacy experimentation)
- C++ daemon sources under `zion-cryptonote/` (if consensus now under Node core or replaced)
- Large conceptual / visionary documents (move to `legacy/docs/vision/`)
- Session logs older than last 14 days (`docs/sessions/`)
- Deprecated multi-algo placeholders (kawpow/ethash/ergo) unless on active roadmap

### 3.3 CONDITIONAL (evaluate)
| Directory | Criteria | Action |
|-----------|----------|--------|
| `swap-service/` | Needed for active testnet? | Include only if live feature.
| `lightning/` | If integrated & used | Keep minimal; else archive.
| `metaverse/`, `gaming/`, `music-ai/`, `bio-ai/` | Not core consensus | Archive; create separate domain repos later.
| `oracles/`, `quantum/` | Experimental | Archive.

## 4. New Repository Layout
```
/ (root)
  VERSION                 # 2.6.5
  core/                   # (from zion-core) TS/Node
  miner/                  # (from zion-miner-1.4.0)
  pool/                   # uzi pool adapter + unified config
  adapters/               # (wallet-adapter, rpc-shim) TEMPORARY
  infra/
    docker/               # Dockerfile.core, Dockerfile.miner, Dockerfile.pool
    compose/              # docker-compose.yml, docker-compose.dev.yml
    scripts/              # bootstrap-testnet.sh, health-check.sh
  config/                 # consensus.json, pool.json, network.json
  docs/
    ARCHITECTURE.md
    MINING_GUIDE.md
    DOCKER_GUIDE.md
    MIGRATION_LEGACY.md
    RPC_API_REFERENCE.md
    CHANGELOG.md
    sessions/ (recent logs)
  scripts/                # developer helpers (build, format, version sync)
  tests/                  # integration + API smoke
  .env.example
```

## 5. Git History Preservation
Preferred tool: `git filter-repo` (faster & more reliable than legacy `filter-branch`).

Example extraction (run from current repo clone):
```bash
git checkout main
git pull --ff-only
git tag archive/2.6
# Paths to keep history
KEEP_PATHS=(zion-core zion-miner-1.4.0 adapters/uzi-pool-config adapters/wallet-adapter adapters/zion-rpc-shim pool config docker scripts)
# Dry run manifest
printf '%s\n' "${KEEP_PATHS[@]}" > migration_paths.txt
# Clone a bare repo for filtering
git clone --no-local --bare . ../zion-2.6.5-filter.git
cd ../zion-2.6.5-filter.git
# Apply filter
git filter-repo $(printf -- '--path %s ' "${KEEP_PATHS[@]}") --force
# Rename dirs post-filter (in working clone later)
```
Alternative (subtree splits) if per-component modular repos wanted later.

## 6. Post-Filter Normalization (Working Tree)
```bash
git clone ../zion-2.6.5-filter.git ../zion-2.6.5
cd ../zion-2.6.5
mv zion-core core
mv zion-miner-1.4.0 miner
mkdir -p infra/docker infra/compose
mv docker/Dockerfile.zion-core infra/docker/Dockerfile.core
mv docker/Dockerfile.zion-unified infra/docker/Dockerfile.miner   # adjust if correct
# Create minimal compose file
cat > infra/compose/docker-compose.yml <<'YAML'
version: "3.9"
services:
  core:
    build: { context: ../.., dockerfile: infra/docker/Dockerfile.core }
    ports: ["18089:18089"]
  pool:
    build: { context: ../.., dockerfile: infra/docker/Dockerfile.pool }
    depends_on: [core, redis]
  miner:
    build: { context: ../.., dockerfile: infra/docker/Dockerfile.miner }
    deploy: { replicas: 0 }
  redis:
    image: redis:7-alpine
YAML
```
(Adjust final after validation.)

## 7. Version Synchronization
1. Add root `VERSION` with `2.6.5`.
2. `core/package.json` → set `"version": "2.6.5"`.
3. Miner CMakeLists: read file:
   ```cmake
   file(READ "${CMAKE_SOURCE_DIR}/../VERSION" PROJECT_VERSION_RAW)
   string(STRIP "${PROJECT_VERSION_RAW}" PROJECT_VERSION)
   add_definitions(-DZION_VERSION=\"${PROJECT_VERSION}\")
   ```
4. Add script `scripts/verify-version-sync.sh` that compares values.

## 8. Testnet Bootstrap Script
`scripts/bootstrap-testnet.sh` (outline):
```bash
#!/usr/bin/env bash
set -euo pipefail
./scripts/build-images.sh
cd infra/compose
docker compose up -d core redis
echo "Waiting for core RPC..."; ./scripts/wait-rpc.sh
# Pre-mine blocks (curl start_mining or internal method)
./scripts/mine-initial-blocks.sh 10
# Start pool & optional miner clients
docker compose up -d pool
```

## 9. CI Baseline (GitHub Actions)
Workflow: `.github/workflows/ci.yml`
- Install Node deps (core)
- `npm run build` (core)
- Build miner (cmake) in Release
- Start core (detached) → curl `/get_info`
- Lint / Typescript check

## 10. Rollback Plan
| Risk | Symptom | Rollback |
|------|---------|----------|
| Filter missed critical path | Missing code in new repo | Re-run filter-repo with added paths |
| Build broken after moves | CI red | Reintroduce missing Dockerfile or fix relative paths |
| Lost history for a file | Short log | Retrieve from `archive/2.6` tag via `git show` or cherry-pick |

## 11. Execution Checklist
- [ ] Tag `archive/2.6`
- [ ] Run filter-repo extraction
- [ ] Normalize directory names
- [ ] Insert VERSION & sync
- [ ] Create compose & docker pruning
- [ ] Add docs skeleton + migrate curated docs
- [ ] Add bootstrap & version scripts
- [ ] Add CI workflow stub
- [ ] Smoke test: build + /get_info + mine block
- [ ] Push new repo & create release draft `v2.6.5`

## 12. Open Decisions
| Item | Question | Default |
|------|----------|---------|
| Built-in C++ pool | Disable or keep? | Disable; rely on adapter |
| rpc-shim | Merge into core soon? | Keep transitional |
| Multi-algo placeholders | Keep for marketing? | Remove until real roadmap |
| AI / metaverse modules | Inline or separate repos? | Separate later |

## 13. Immediate Next Action
Prepare non-destructive PR in current repo that:
1. Adds `REPO_MIGRATION_PLAN_2.6_to_2.6.5.md` (this file)
2. Adds placeholder `VERSION` (2.6.5-draft)
3. Adds `CHANGELOG.md` with unreleased 2.6.5 section
4. Adds `scripts/verify-version-sync.sh`

After merge → perform extraction externally.

---
End of Plan (Ready for execution)
