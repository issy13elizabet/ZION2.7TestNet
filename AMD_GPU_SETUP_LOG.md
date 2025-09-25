# ZION AMD GPU Mining Setup Log - 25. září 2025

## 🎯 Cíl
Rozchodit GPU mining na Ubuntu s AMD Radeon RX 5600 XT pro ZION TestNet.

## ✅ Co se podařilo

### 1. Hardware detekce
- **GPU**: AMD Radeon RX 5600 XT (Navi 10) detekována
- **CPU**: AMD Ryzen 5 3600 (6C/12T) - mining běží ✅
- **Systém**: Ubuntu 25.04 (Plucky)

### 2. Minery připraveny
- **XMRig v6.24.0**: ✅ Stažen, nakonfigurován, běží na CPU
- **SRBMiner-MULTI v2.9.7**: ✅ Stažen, detekuje algoritmy
- **Konfigurace**: Peněženky a pooly nakonfigurovány

### 3. XMRig CPU mining aktivní
```
✅ XMRig běží na AMD Ryzen 5 3600
✅ Pool: 91.98.122.165:3333 (RandomX)
✅ Wallet: Z3BDEEC2A0AE0F5D81B034308F99ECD8990D9B8B01BD9C7E7429392CA31861C6220DA3B30D74E809FA0A1FE069F1
✅ Status: Mining aktivní
```

### 4. AMD GPU ovladače - částečně
- **AMDGPU kernel modul**: ✅ Načten (lsmod | grep amdgpu)
- **ROCm OpenCL**: ✅ Nainstalován
- **Mesa OpenCL**: ✅ Nainstalován
- **OpenCL detekce**: ✅ AMD RX 5600 XT detekována přes clinfo

## ❌ Problém: SRBMiner GPU mining

### Chyba
```bash
./SRBMiner-MULTI --algorithm kawpow --pool 91.98.122.165:3334 --wallet ... --disable-cpu
Detecting GPU devices...
GPU mining disabled, OpenCL not installed ?
No devices available to mine with
```

### OpenCL status
```bash
clinfo
Number of platforms: 3
  1. Clover (Mesa) - AMD Radeon RX 5600 XT ✅ DETEKOVÁNO
  2. AMD Accelerated Parallel Processing - 0 devices
  3. rusticl - 0 devices
```

### Možné příčiny
1. **DKMS build failed**: AMD GPU DKMS modul se nepostavil správně
2. **OpenCL runtime issue**: SRBMiner nevidí Mesa OpenCL
3. **Permissions**: GPU přístupová práva
4. **Kernel compatibility**: Ubuntu 25.04 vs AMD drivers

## 🔧 Instalované komponenty

### AMD Software
```bash
- amdgpu-install_6.2.60204-1
- rocm-opencl-runtime
- mesa-opencl-icd
- libdrm-amdgpu-amdgpu1
- amdgpu kernel modul (načten)
```

### Mining Software
```bash
- XMRig 6.24.0 (CPU) ✅
- SRBMiner-MULTI 2.9.7 (GPU) ❌
```

## 📋 Další kroky po restartu

1. **Restart systému** - načtení všech kernel modulů
2. **Ověřit AMDGPU** - `lsmod | grep amdgpu`
3. **Test OpenCL** - `clinfo`
4. **Test SRBMiner** - `./SRBMiner-MULTI --list-devices`
5. **GPU permissions** - `ls -la /dev/dri/`

## 💾 Backup
- Všechny mining konfigurace: ✅ Připraveny
- Wallet backupy: ✅ V `/home/maitreya/backup-wallets/` (mimo git)
- Mining skripty: ✅ V `scripts/`

## 🚀 Mining Status
- **CPU Mining**: ✅ AKTIVNÍ (XMRig na Ryzen 5 3600)
- **GPU Mining**: ❌ Čeká na restart/driver fix
- **Pool**: ✅ 91.98.122.165:3333 (CPU), 3334 (GPU)
- **Cíl**: 60 bloků pro TestNet startup

---
**Poznámka**: CPU mining běží a těží. Po restartu by mělo GPU mining také fungovat.