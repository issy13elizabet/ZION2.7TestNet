# Legacy Components Archive

⚠️ **VAROVÁNÍ**: Tyto komponenty jsou zastaralé a neměly by být používány v nových deploymentech.

## Účel tohoto adresáře

Tento adresář obsahuje zastaralé komponenty z původní verze 2.6, které byly přesunuty sem během migrace na novou 2.6.5 architekturu. Komponenty jsou zachovány pro historické účely a možný rollback, ale nejsou aktivně udržovány.

## Struktura

- `miners/` - Staré verze minerů (1.3.0, 1.4.0, multi-platform)
- `experiments/` - Experimentální buildy a zastaralé CMake varianty
- `configs/` - Duplicitní konfigurační soubory a test skripty
- `docker-variants/` - Původní docker složka s mnoha variantami Dockerfile
- `docs-archive/` - Zastaralá dokumentace (připraveno)

## Migrace Status

**Datum migrace**: 29. září 2025
**Nová architektura**: `zion2.6.5testnet/` obsahuje čistou monorepo strukturu
**Backup tagy**: `v2.6-backup-final` pro kompletní rollback

## Co použít místo toho

- **Miner**: Používej `zion2.6.5testnet/miner/` (C++ sources integrované v monorepo)
- **Core**: Používej `zion2.6.5testnet/core/` (TypeScript/Node service s embedded Stratum)
- **Docker**: Používej `zion2.6.5testnet/docker-compose.yml` a `infra/docker/`
- **Konfigurace**: Používej `zion2.6.5testnet/config/network/`

## Rollback pokyny

V případě potřeby rollbacku:
```bash
git checkout v2.6-backup-final
# nebo extrahuj
tar -xzf Zion-2.6-final-backup.tar.gz
```

**NEDOPORUČUJE SE** používat tyto legacy komponenty pro nové instalace.