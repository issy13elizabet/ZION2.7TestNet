# Zion 2.6.5 Testnet - Architektura

## Přehled

Toto je čistá monorepo architektura pro Zion blockchain testnet verze 2.6.5. Nahrazuje roztříštěnou strukturu předchozí verze 2.6 jednotným, udržitelným řešením.

## Principy designu

### 1. Monorepo struktura
- Všechny komponenty v jednom repo
- Jednotné verzování across komponent
- Sdílené závislosti a build tooling
- Konzistentní CI/CD pipeline

### 2. Integrované služby
- Core obsahuje embedded Stratum pool server
- Eliminace externích pool adapterů
- Přímá komunikace mezi komponenty
- Redukce network latency

### 3. Kontejnerizace
- Každá služba má vlastní Dockerfile
- Jednotný docker-compose.yml
- Environment-based konfigurace
- Production-ready optimalizace

## Struktura adresářů

```
zion2.6.5testnet/
├── VERSION                      # Single source of truth pro verzování
├── .env.example                 # Environment template
├── docker-compose.yml           # Orchestrace služeb
├── 
├── core/                        # TypeScript core služba
│   ├── src/
│   │   ├── server.ts           # HTTP API + Stratum bootstrap
│   │   ├── pool/
│   │   │   └── StratumServer.ts # Embedded mining pool
│   │   ├── blockchain/         # Blockchain logic (z original zion-core)
│   │   ├── rpc/               # RPC endpoints
│   │   ├── p2p/               # Peer-to-peer networking
│   │   └── wallet/            # Wallet API
│   ├── package.json            # Node dependencies
│   └── tsconfig.json           # TypeScript konfigurace
├── 
├── miner/                      # C++ mining executable
│   ├── CMakeLists.txt         # Build konfigurace
│   ├── src/                   # Mining algorithm sources
│   └── include/               # Headers
├──
├── frontend/                   # Next.js web interface
│   ├── package.json           # React/Next dependencies
│   └── src/                   # UI components
├──
├── infra/
│   └── docker/                # Container definitions
│       ├── Dockerfile.core    # Core service image
│       ├── Dockerfile.miner   # Miner executable image
│       └── Dockerfile.frontend # Frontend image
├──
├── config/
│   └── network/               # Blockchain parameters
│       ├── genesis.json       # Genesis block definice
│       ├── consensus.json     # Consensus pravidla
│       └── pool.json          # Pool konfigurace
├──
├── scripts/                   # Utility skripty
│   ├── bootstrap-testnet.sh   # Testnet init
│   ├── genesis-hash-check.js  # Genesis validace
│   └── verify-version-sync.sh # Verze consistency check
├──
├── docs/                      # Dokumentace
├── ai/                        # AI integration placeholder
└── .github/
    └── workflows/
        └── ci.yml             # GitHub Actions CI
```
