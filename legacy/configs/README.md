# Zastaralé konfigurace

⚠️ **DUPLICITNÍ A ZASTARALÉ KONFIGURACE**

## Obsah

### Docker Compose varianty (zastaralé)
- `docker-compose-bootstrap.yml` - Bootstrap setup (nahrazeno)
- `docker-compose-production.yml` - Production setup (nahrazeno)

### Test soubory (zastaralé)  
- `test-*.json` - Starý test data
- `test-*.sh` - Legacy test skripty
- `*.bat` - Windows skripty (nahrazeno cross-platform řešením)

## Nové konfigurace

**Používej**: `../zion2.6.5testnet/` unified konfigurace:

### Docker
```bash
cd ../zion2.6.5testnet
cp .env.example .env  # upravit dle potřeby
docker compose up -d
```

### Network konfigurace
- `../zion2.6.5testnet/config/network/genesis.json`
- `../zion2.6.5testnet/config/network/consensus.json`  
- `../zion2.6.5testnet/config/network/pool.json`

### Environment
- `../zion2.6.5testnet/.env.example` - Template pro lokální nastavení

Všechny tyto starý konfigurace jsou nahrazeny jednotným systémem v novém repo.