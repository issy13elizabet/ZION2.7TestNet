# Zastaralé Miner verze

⚠️ **TYTO MINERY JSOU ZASTARALÉ**

## Obsah

- `zion-miner-1.3.0/` - Původní miner verze 1.3.0
- `zion-miner-1.4.0/` - Poslední standalone verze před integrací
- `zion-multi-platform/` - Experimentální multi-platform build

## Místo toho použij

**Nový integrovaný miner**: `../zion2.6.5testnet/miner/`

Nový miner je součástí monorepo struktury a obsahuje:
- Unified build systém (CMake)
- Jednotné verzování s core komponentami  
- Optimalizované závislosti
- Docker integration

## Build nového mineru

```bash
cd ../zion2.6.5testnet
docker compose build miner
# nebo lokálně:
cd miner && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```