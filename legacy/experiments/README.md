# Experimentální komponenty

⚠️ **ZASTARALÉ EXPERIMENTY - NEPOUŽÍVAT**

## Obsah

### Build varianty (zastaralé)
- `CMakeLists-hybrid.txt` - Hybridní build (zrušeno)
- `CMakeLists-minimal.txt` - Minimální build (zrušeno) 
- `CMakeLists-simple.txt` - Jednoduchý build (zrušeno)
- `CMakeLists-xmrig.txt` - XMRig integrace (zrušeno)

### Build adresáře
- `build-*` - Různé build experimenty (všechny zrušené)

### Legacy komponenty
- `zion-cryptonote/` - Původní CryptoNote implementace (nahrazena)
- `xmrig-*` - XMRig konfigurační experimenty (zrušeno)

## Nový přístup

**Používej**: `../zion2.6.5testnet/` s unified build systémem:

```bash
# Core (TypeScript)
cd ../zion2.6.5testnet/core && npm run build

# Miner (C++) 
cd ../zion2.6.5testnet/miner && mkdir build && cd build
cmake .. && make

# Vše najednou
cd ../zion2.6.5testnet && docker compose build
```

Všechny tyto experimenty jsou nahrazeny čistou monorepo strukturou.