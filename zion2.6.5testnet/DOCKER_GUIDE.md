# Docker Guide (Simple Migration Phase)

This guide documents the minimal container setup for the 2.6.5 testnet skeleton.

## Services

- core: TypeScript Node service exposing HTTP health endpoint and embedded Stratum server
- frontend: Next.js UI (static build + runtime start)
- miner: Native C++ miner (placeholder build; no real share validation yet)

## Files

- `infra/docker/Dockerfile.core` – builds + runs core (no dev deps in runtime layer)
- `infra/docker/Dockerfile.frontend` – two stage Next.js build
- `infra/docker/Dockerfile.miner` – CMake build of miner
- `docker-compose.yml` – orchestrates the three containers

## Environment Variables

| Variable | Service | Default | Description |
|----------|---------|---------|-------------|
| PORT | core | 8601 | HTTP port for core API |
| STRATUM_PORT | core | 3333 | Stratum TCP port |
| INITIAL_DIFFICULTY | core | 1000 | Starting difficulty for new connections |

You can override by creating `.env` in repository root.

```
PORT=8601
STRATUM_PORT=3333
INITIAL_DIFFICULTY=500
```

## Usage

Build & start all:

```
docker compose build
docker compose up -d
```

Check health:
```
curl -f http://localhost:8601/healthz
```

Tail logs:
```
docker compose logs -f core
```

Connect external miner (example):
```
./zion-miner --pool=127.0.0.1:3333 --wallet=<yourAddress>
```

## Next Steps (Not Yet Implemented)

- Real block template generation
- Difficulty adjustment (vardiff)
- Share validation + reject logic
- Wallet adapter container
- Seed nodes replication

These will change Docker topology later; this document will then be upgraded.
