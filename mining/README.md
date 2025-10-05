# 🚀 ZION Mining Suite v2.7.1

Kompletní sada optimalizovaných minerů pro ZION blockchain.

## 📂 Struktura souborů

### 🏆 Yescrypt CPU Mining (Doporučeno - Eco bonus +15%)
- `zion_yescrypt_hybrid.py` - **Hlavní hybrid miner** (C/Python)
- `zion_yescrypt_professional.py` - Profesionální implementace
- `zion_yescrypt_optimized.py` - Optimalizovaná verze
- `simple_yescrypt_test.py` - Rychlý test výkonu
- `yescrypt_fast.c` - C extension pro maximální rychlost
- `setup.py` - Build script pro C extension

### ⚙️ Konfigurace
- `yescrypt-miner-config.json` - Konfigurační template
- `xmrig-zion-yescrypt.json` - XMRig konfigurace

### 📚 Dokumentace
- `ZION_YESCRYPT_MINER_GUIDE.md` - **Kompletní návod**
- `test_yescrypt_miner.py` - Test suite

### 🔧 Legacy soubory
- `zion_yescrypt_cpu_miner.py` - Původní implementace
- `yescrypt_energy_efficient.py` - Energeticky účinná varianta

## 🚀 Rychlý start

### 1. Nainstalujte závislosti
```bash
python -m pip install --upgrade pip
```

### 2. Zkompilujte C extension (volitelné, ale doporučeno)
```bash
python setup.py build_ext --inplace
```

### 3. Spusťte miner
```bash
python zion_yescrypt_hybrid.py --wallet ZioniVAŠE_ADRESA_PENĚŽENKY
```

## 📊 Porovnání implementací

| Miner | Výkon | Funkce | Doporučení |
|-------|-------|---------|------------|
| **hybrid** | ⭐⭐⭐⭐⭐ | C/Python, eco-bonus, statistiky | **DOPORUČENO** |
| professional | ⭐⭐⭐⭐ | Plné Yescrypt, pool integrace | Pro pokročilé |
| optimized | ⭐⭐⭐ | Optimalizace, eco-bonus | Stabilní volba |
| simple_test | ⭐⭐ | Základní test, benchmark | Jen pro testování |

## 🌱 Eco-bonus systém

**Yescrypt algoritmus má nejlepší eco-bonus na ZION síti:**
- **+15% hashrate bonus** při eco módu
- **Nejnižší spotřeba**: ~80W
- **Nejvyšší účinnost**: 6-8 H/s/W
- **ASIC odolnost**: Memory-hard algoritmus

## ⚡ Očekávané výsledky

### S C extension (doporučeno)
```
🚀 C OPTIMIZED ZION Yescrypt Miner v2.7.1
⚡ Hashrate: 800-1200 H/s (4-8 vláken)
🌱 Eco Hashrate: 920-1380 H/s (+15%)
📈 Efficiency: 11.5-17.3 H/s/W
🏆 MAXIMUM PERFORMANCE MODE ACTIVE
```

### Python fallback
```
🐍 PYTHON ZION Yescrypt Miner v2.7.1
⚡ Hashrate: 200-400 H/s (4-8 vláken)  
🌱 Eco Hashrate: 230-460 H/s (+15%)
⚠️ Python fallback mode - compile C extension for 5-10x speed boost
```

## 🛠️ Pokročilé použití

### Vlastní pool
```bash
python zion_yescrypt_hybrid.py \
  --wallet ZioniADRESA \
  --host mining.zion.network \
  --port 4444 \
  --threads 6
```

### S config souborem
```bash
# Editujte yescrypt-miner-config.json
python zion_yescrypt_hybrid.py --config yescrypt-miner-config.json
```

### Benchmark test
```bash
python simple_yescrypt_test.py --duration 30
```

## 🔧 Troubleshooting

### C extension se nezkompiluje
```bash
# Windows: Nainstalujte Visual Studio Build Tools
# Linux: sudo apt install build-essential python3-dev
# macOS: xcode-select --install

python setup.py build_ext --inplace
```

### Nízký hashrate
1. **Zkompilujte C extension** - 5-10x rychlejší
2. **Optimalizujte počet vláken** - zkuste CPU_COUNT-1
3. **Zavřete jiné aplikace** - uvolněte CPU
4. **Aktualizujte Python** na nejnovější verzi

### Pool connection error
1. Zkontrolujte firewall nastavení
2. Ověřte pool host/port
3. Testujte: `telnet localhost 4444`

## 📈 Monitoring a statistiky

Miner automaticky zobrazuje každých 30 sekund:
- ⚡ **Hashrate**: Aktuální výkon
- 🎯 **Shares**: Odeslaná řešení
- 🌱 **Eco shares**: Řešení s eco-bonusem  
- 📈 **Efficiency**: H/s na watt
- 🔧 **Implementation**: C/Python status

## 💡 Tipy pro maximální ziskovost

1. **Používejte eco mód** - +15% bonus navždy
2. **Kompilujte C extension** - 5-10x rychlost
3. **Optimalizujte threads** - testujte různé hodnoty
4. **Monitorujte teplotu CPU** - throttling snižuje výkon
5. **Těžte v noci** - nižší ceny elektřiny

## 🏆 Úspěšné těžení!

Pro podporu a komunitu navštivte náš Discord/Telegram kanál.

---
*ZION Mining Suite v2.7.1 | Optimalizováno pro eco-bonus těžení*