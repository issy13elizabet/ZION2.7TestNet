# Zion 2.6 Repository Restructure Plan

Purpose: Remove confusion from mixed legacy (C++ v1.0.0 / miner 1.3.x / obsolete docker stacks) and establish a clean, authoritative layout centered on Zion Core 2.6 + Miner 1.4.0.

## 1. Current Problems
- Multiple Dockerfiles (≈30) with overlapping purposes (cryptonote, unified, runtime-local, prod, minimal)
- Two paradigms of core:
  - Legacy C++ daemon (banner: ZION CRYPTOCURRENCY v1.0.0)
  - Node/TypeScript `zion-core` (2.6) providing RPC + modular services
- Duplicate miner versions (`zion-miner-1.3.0`, `zion-miner-1.4.0`)
- Scattered configs: `config/`, `docker/`, `adapters/uzi-pool-config/`, root `.env`, mining scripts
- Pool logic duplicated (built‑in pool vs external pool adapter)
- Docs directory contains runbooks, visionary papers, logs, mining how‑tos, all flat => discovery pain
- RPC instability partly caused by running wrong daemon binary (legacy) side-by-side with core 2.6 components

## 2. Target Principles
1. Single Source of Truth per domain (core, miner, pool, adapters)
2. Clear separation: ACTIVE vs LEGACY
3. Minimal Docker surface (one build for core, one for miner, one for pool adapter if still needed)
4. Deterministic environment via consolidated `.env` + `compose.yml`
5. Self-describing docs: top-level architecture + focused guides
6. Easy testnet bootstrap (scripted: generate genesis, mine N blocks, start pool)

## 3. Proposed Directory Layout
```text
/ (root)
  core/                 # Zion Core 2.6 (Node/TS) – authoritative
  miner/                # Zion Miner 1.4.0 (C++ multi-platform)
  pool/                 # Pool adapter (UZI) + config (normalized)
  adapters/             # (Optional) bridge or wallet adapters kept if active
  infra/                # Docker & deployment (compose, Dockerfiles, k8s later)
    docker/             # Canonical Dockerfiles only
    compose/            # compose.yml, compose.dev.yml, compose.lightning.yml
    scripts/            # bootstrap, health, migration
  config/               # Unified runtime config (json/toml/yaml) referenced by services
  docs/                 # Curated docs (see section 5)
  legacy/               # Archived old components
    daemon-cpp/         # C++ v1.0.0 daemon + associated Dockerfiles
    miner-1.3.0/
    old-docker/
    experimental/
  scripts/              # Developer helper scripts (build, test, format)
  tests/                # Integration + unit tests
  .env.example          # Master env schema
  docker-compose.yml    # Primary stack (core + rpc-shim + pool + miner optional)
```

## 4. Docker Consolidation
| Keep | Path | Purpose |
|------|------|---------|
| ✅ | `infra/docker/Dockerfile.core` | Builds Zion Core 2.6 (Node) |
| ✅ | `infra/docker/Dockerfile.miner` | Builds miner 1.4.0 CPU/GPU variant |
| ✅ | `infra/docker/Dockerfile.pool` | Pool adapter (UZI) with slim base |
| ❌ | Deprecated cryptonote-only Dockerfiles | Move to legacy/daemon-cpp |
| ❌ | `Dockerfile.zion-cryptonote.*` | Archive (legacy)
| ❌ | Duplicated runtime-local/minimal variants | Merge via ARG flags

Single compose strategy:
- `docker-compose.yml` (core + rpc-shim + pool + redis)
- `docker-compose.dev.yml` (adds hot reload, mounting source)
- `docker-compose.miner.yml` (optional remote miners for lab)

## 5. Docs Restructure
### Keep (curate & rename if needed)
- `PROJECT_ARCHITECTURE_OVERVIEW.md` → `ARCHITECTURE.md`
- `CONSENSUS_PARAMS.md`
- `UNIFIED_ZION_CORE_IMPLEMENTATION.md` merge into ARCHITECTURE or keep as deep dive
- `DOCKER_WORKFLOW_GUIDE.md` → `DOCKER_GUIDE.md`
- `MINING_SSH_SETUP.md` + `ZION_MINING_SETUP.md` → `MINING_GUIDE.md`
- `ZION_MINER_1.4.0_COMPREHENSIVE_LOG_2025-09-29.md` (archive under `docs/sessions/`)

### Move to `docs/legacy/` (historical, not required for onboarding)
- Vision / manifesto / strategic expansions (e.g. LIBERATION, MASTER PLAN variants)
- Session logs older than last 2 weeks

### Add New
- `MIGRATION_LEGACY.md` (how to upgrade from C++ daemon to core 2.6)
- `BOOTSTRAP_TESTNET.md` (scripted local + remote procedure)
- `POOL_OPERATIONS.md` (jobs, payouts, troubleshooting)
- `RPC_API_REFERENCE.md` (get_info, getblocktemplate, mining methods)

## 6. Version Alignment
Single version authority file:
- `VERSION` (contains `2.6.x`)
- Core `package.json` version sync script → updates `VERSION`
- Miner build embeds version via CMake configure step reading root `VERSION`

## 7. Immediate Action Steps (Execution Order)
1. Create `legacy/` and move obsolete Dockerfiles & `zion-miner-1.3.0` there
2. Introduce `infra/docker/` & relocate kept Dockerfiles
3. Write `VERSION` file; patch core & miner build to read it
4. Draft new `docker-compose.yml` referencing trimmed images
5. Unify env variables (generate `.env.example` from superset + prune unused)
6. Create new docs structure and relocate historical logs
7. Add bootstrap script: `scripts/bootstrap-testnet.sh`
8. Add CI (GitHub Actions) placeholder that builds core + miner + runs `--help`

## 8. Risk Mitigation
| Risk | Mitigation |
|------|------------|
| Breaking existing external scripts | Provide symlink or README for 2 weeks |
| Confusing contributors about legacy removal | LEGACY README with rationale |
| Loss of historical logs | Move, never delete |
| Divergent configs after move | Add config loader that logs absolute path origins |

## 9. Open Questions
- Keep built-in pool inside C++ daemon? → Recommend disable & rely solely on adapter
- Is Node `zion-core` authoritative for consensus or just orchestration? (If orchestration only, clarify layering in ARCHITECTURE)
- Do we still require multi-algo placeholders (kawpow, ethash)? If not, strip for focus.

## 10. Success Criteria
- `docker compose up` (clean build) yields: core responding to `/get_info`, pool waiting for templates, miner attachable
- `scripts/bootstrap-testnet.sh` mines first 10 blocks automatically
- Only three Dockerfiles present in active tree
- No references to miner 1.3.0 outside `legacy/`
- Docs index links to all active guides without dead links

---
Prepared for execution. Next step: implement Step 1 (create legacy/ and move obsolete assets) + Step 2 (introduce infra/docker).