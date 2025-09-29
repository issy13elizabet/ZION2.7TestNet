# Legacy Docker varianty

⚠️ **NEPOUŽÍVAT - ZASTARALÉ DOCKER SOUBORY**

## Původní obsah (přesunutý z `/docker/`)

Tato složka obsahovala původní docker infrastrukturu s mnoha variantami:
- ~30 různých Dockerfile souborů
- 6+ docker-compose variant
- Složité přepisování portů a konfigurací
- Duplicitní entrypointy a skripty

## Problémy s původní strukturou

- Příliš mnoho variant bez jasné hiearchie
- Duplicitní konfigurace napříč soubory
- Nejednotné verzování
- Složité debugging při problémech

## Nové Docker řešení

**Používej**: `../zion2.6.5testnet/infra/docker/` a `../zion2.6.5testnet/docker-compose.yml`

### Nová struktura (jednoduchá)
```
zion2.6.5testnet/
├── docker-compose.yml          # Jednotný compose file
├── .env.example                # Environment template
└── infra/docker/
    ├── Dockerfile.core         # Core TypeScript service
    ├── Dockerfile.miner        # C++ miner
    └── Dockerfile.frontend     # Next.js frontend
```

### Použití
```bash
cd ../zion2.6.5testnet
cp .env.example .env            # nastav porty dle potřeby
docker compose build
docker compose up -d
```

**3 služby, 3 Dockerfile, 1 compose** místo 30+ souborů.