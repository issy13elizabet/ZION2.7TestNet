# ZION Multi-Algorithm GPU Mining with SRBMiner-Multi

## Přehled
ZION Core v2.5.0 nyní používá **SRBMiner-Multi v2.9.7** jako výchozí GPU miner pro pokročilou multi-algo těžbu s AMD/NVIDIA podporou.

## GPU Mining Setup
- **Primary GPU Miner**: SRBMiner-Multi v2.9.7 
- **Hardware**: AMD Radeon RX 5600 XT (18 compute units, 5.8GB VRAM)
- **CPU Backup**: XMRig v6.21.3 (6 threads AMD Ryzen 5 3600)

## Podporované Algoritmy
1. **KawPow** (port 3334) - Ravencoin kompatibilní
2. **Octopus** (port 3337) - Conflux (CFX) 
3. **Ergo** (port 3338) - Autolykos2 algoritmus
4. **RandomX** (port 3333) - Monero kompatibilní
5. **Ethash** (port 3335) - Ethereum Classic
6. **CryptoNight** (port 3336) - ZION native

## Spuštění Mining
```bat
# Kompletní multi-algo mining
start-multi-algo-mining.bat

# Test jednotlivých algoritmů
test-srb-kawpow.bat     # KawPow GPU mining
test-srb-octopus.bat    # Octopus GPU mining  
test-srb-ergo.bat       # Ergo GPU mining
```

## Konfigurační Soubory
- `srb-kawpow-config.json` - Optimalizováno pro KawPow (intensity 23)
- `srb-octopus-config.json` - Optimalizováno pro Octopus (intensity 21)
- `srb-ergo-config.json` - Optimalizováno pro Ergo (intensity 20)
- `multi-algo-pools.json` - Centrální konfigurace všech poolů
- `zion-multi-algo-bridge.py` - Inteligentní přepínání algoritmů

## Profitabilita & Přepínání
Python bridge automaticky:
- Monitoruje síťovou obtížnost
- Kalkuluje denní zisk v USD
- Přepíná na nejziskovější algoritmus každých 5 minut
- Loguje všechny změny a výkon

## AMD RX 5600 XT Optimalizace
```json
{
    "gpu_intensity": 20-23,
    "gpu_worksize": 256, 
    "gpu_threads": 18,
    "gpu_boost": 100,
    "gpu_adl_type": 1
}
```

## Expected Hashrates
- **KawPow**: ~8.5 MH/s
- **Octopus**: ~45 MH/s  
- **Ergo**: ~85 MH/s
- **Ethash**: ~22 MH/s
- **RandomX**: ~1200 H/s
- **CryptoNight**: ~950 H/s

## Monitoring
- SRBMiner API: ports 21555-21557
- GPU statistiky: AMD Radeon Software
- Python bridge logy: console output
- Server monitoring: ZION Core dashboard

## Troubleshooting
1. Pokud SRBMiner nefunguje, automatický fallback na XMRig
2. Python bridge vyžaduje `requests` modul: `pip install requests`
3. GPU drivers: AMD Software Adrenalin najnovší verze
4. OpenCL runtime: AMD APP SDK nebo ROCm