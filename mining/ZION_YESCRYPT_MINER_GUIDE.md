# ZION Yescrypt CPU Miner - Kompletní návod

## 🚀 Přehled

ZION Yescrypt CPU Miner je optimalizovaný těžební software pro algoritmus Yescrypt na ZION blockchainu. Nabízí nejlepší energetickou účinnost ze všech podporovaných algoritmů s eco-bonusem +15%.

## 📊 Klíčové vlastnosti

- **Nejnižší spotřeba energie**: ~80W (nejúčinnější ze všech algoritmů)
- **Eco-bonus**: +15% k hashrate při aktivním eco módu
- **ASIC odolnost**: Memory-hard algoritmus zabraňuje ASIC dominanci
- **Hybridní implementace**: Automaticky používá C extension pokud je dostupná
- **Multi-threading**: Optimalizované pro moderní CPU
- **Real-time statistiky**: Podrobné sledování výkonu

## 🛠️ Instalace

### Základní instalace
```bash
cd mining/
python -m pip install --upgrade pip
```

### Kompilace C extension (doporučeno pro maximální výkon)
```bash
python setup.py build_ext --inplace
```
*Poznámka: C extension poskytuje 5-10x rychlejší výkon*

## ⚙️ Konfigurace

### Základní config soubor: `yescrypt-config.json`
```json
{
  "pool_host": "localhost",
  "pool_port": 4444,
  "wallet_address": "ZioniVAŠE_ADRESA_PENĚŽENKY_ZDE",
  "threads": null,
  "eco_mode": true,
  "mining_config": {
    "difficulty": 8000,
    "power_target": 80.0,
    "eco_bonus": 1.15
  }
}
```

### Parametry konfigurace

| Parametr | Popis | Výchozí | Doporučeno |
|----------|-------|---------|------------|
| `pool_host` | IP/doména mining poolu | localhost | IP vašeho poolu |
| `pool_port` | Port poolu | 4444 | Dle nastavení poolu |
| `wallet_address` | ZION adresa peněženky | - | **POVINNÉ** |
| `threads` | Počet vláken | auto | CPU_COUNT - 1 |
| `eco_mode` | Eco-bonus mód | true | true (doporučeno) |

## 🚀 Spuštění

### Rychlé spuštění
```bash
python zion_yescrypt_hybrid.py --wallet ZioniVAŠE_ADRESA
```

### Pokročilé možnosti
```bash
# Vlastní počet vláken
python zion_yescrypt_hybrid.py --wallet ZioniADRESA --threads 4

# Vlastní pool
python zion_yescrypt_hybrid.py --wallet ZioniADRESA --host 192.168.1.100 --port 4444

# S config souborem
python zion_yescrypt_hybrid.py --config yescrypt-config.json

# Bez eco-bonusu (nedoporučeno)
python zion_yescrypt_hybrid.py --wallet ZioniADRESA --no-eco
```

## 📊 Interpretace výsledků

### Ukázka statistik
```
📊 HYBRID MINER STATISTICS (60s)
-------------------------------------------------------
⚡ Hashrate:       512.34 H/s
🎯 Shares:        12 (12.0/min)
🌱 Eco Shares:        12 (100.0%)
💎 Total Hashes:          30,740
🧮 Threads Active:        4
⚡ Power Usage:     80.0W
📈 Efficiency:    6.404 H/s/W
🚀 Eco Hashrate:   589.19 H/s (+15%)
🌟 Eco Efficiency:    7.365 H/s/W
🔧 Implementation: C Extension
```

### Vysvětlení metrik

- **Hashrate**: Aktuální rychlost výpočtu hashů za sekundu
- **Eco Hashrate**: Efektivní hashrate s eco-bonusem (+15%)
- **Shares**: Počet validních výsledků odeslaných do poolu
- **Eco Shares**: Shares s eco-bonusem (vyšší odměna)
- **Efficiency**: Hashrate na watt (H/s/W) - čím vyšší, tím lepší
- **Implementation**: C Extension = maximální výkon, Python = fallback

## 🔧 Optimalizace výkonu

### 1. Kompilace C Extension
```bash
# Instalace build nástrojů (Windows)
pip install setuptools wheel

# Kompilace
python setup.py build_ext --inplace
```

### 2. Optimální počet vláken
- **Doporučeno**: CPU_COUNT - 1 (nechává 1 core pro systém)
- **Maximum**: 8 vláken (více nepřináší výhodu u Yescrypt)
- **Testování**: Zkuste různé hodnoty a sledujte hashrate

### 3. Systémové optimalizace
```bash
# Windows: Vysoká priorita
python -c "import os; os.system('wmic process where name=\"python.exe\" CALL setpriority \"high priority\"')"

# Vypnutí Windows Defenderu pro mining složku (dočasně)
# Přidejte mining složku do výjimek antivirového software
```

## 🌱 Eco-bonus systém

### Výhody eco módu
- **+15% bonus** k hashrate
- **Priorita v poolu** při výběru bloků  
- **Nižší poplatky** za transakce
- **Udržitelné těžení** s minimální spotřebou energie

### Eco metriky
- **Spotřeba**: ~80W (nejnižší ze všech algoritmů)
- **Účinnost**: 6-8 H/s/W (s eco-bonusem)
- **ROI**: Vyšší ziskovost díky bonusu a nižší spotřebě

## 🛠️ Troubleshooting

### Časté problémy

#### Nízký hashrate
```bash
# Zkontrolujte C extension
python -c "import yescrypt_fast; print('C extension OK')"

# Test výkonu
python simple_yescrypt_test.py
```

#### Chyby připojení k poolu
```bash
# Test spojení
telnet localhost 4444

# Zkontrolujte firewall
netstat -ano | findstr 4444
```

#### Vysoká spotřeba CPU
- Snižte počet vláken: `--threads 2`
- Zkontrolujte ostatní procesy: Task Manager
- Použijte nice/ionice na Linux

### Diagnostické příkazy

```bash
# Základní test
python simple_yescrypt_test.py --duration 10

# Benchmark různých implementací
python -c "
from zion_yescrypt_hybrid import HybridYescryptMiner
config = {'wallet_address': 'ZioniTEST123', 'threads': 1, 'eco_mode': True}
miner = HybridYescryptMiner(config)
print('Benchmark complete')
"

# Test síťového připojení
ping pool.zion.network
```

## 📈 Očekávané výsledky

### Výkon podle CPU

| CPU Typ | Threads | Hashrate (H/s) | Eco Hashrate | Efficiency |
|---------|---------|----------------|--------------|------------|
| Intel i3 | 2 | 200-300 | 230-345 | 2.9-4.3 H/s/W |
| Intel i5 | 4 | 400-600 | 460-690 | 5.8-8.6 H/s/W |
| Intel i7 | 6 | 600-900 | 690-1035 | 8.6-12.9 H/s/W |
| AMD Ryzen 5 | 6 | 650-950 | 748-1093 | 9.3-13.7 H/s/W |
| AMD Ryzen 7 | 8 | 800-1200 | 920-1380 | 11.5-17.3 H/s/W |

*Poznámka: Výsledky s C extension. Python fallback je 5-10x pomalejší.*

### Ziskovost (orientační)

S eco-bonusem a spotřebou 80W je Yescrypt často **nejziskovější** algoritmus pro CPU mining na ZION síti.

## 🔗 Užitečné odkazy

- **ZION Pool**: http://localhost:3334 (lokální pool)
- **Blockchain Explorer**: (TBA)
- **Komunita**: Discord/Telegram kanály
- **GitHub**: ZION2.7TestNet repository

## 📞 Podpora

### Rychlá pomoc
1. Zkontrolujte wallet adresu
2. Ověřte připojení k poolu  
3. Zkuste snížit počet vláken
4. Překompilujte C extension

### Reportování problémů
Při reportování problému přiložte:
- Verzi mineru a OS
- Konfiguračni soubor
- Výstup s chybou
- Specs CPU a RAM

---

**🏆 Úspěšné těžení s ZION Yescrypt minerem!**

*Verze dokumentace: 2.7.1 | Aktualizováno: Říjen 2025*