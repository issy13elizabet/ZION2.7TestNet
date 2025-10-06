# ZION Miner 1.4.0 - Professional Mining Client

## 🚀 Dva Mining Režimy

### 🎮 **zion-miner.py** - SIMULOVANÝ Mining
- **Testování** pool připojení a Stratum protokolu
- **Nulová** spotřeba CPU/elektřiny
- **Bezpečný** pro testování konfigurace
- **Ideální** pro vyzkoušení NiceHash nastavení

### 🔥 **zion-real-miner.py** - SKUTEČNÝ Mining  
- **Reálný RandomX** hashing algoritmus
- **100% CPU** vytížení všech threads
- **Skutečné** nalézání shares a rewards
- **POZOR:** Vysoká spotřeba elektřiny!

## 🌟 Features

- **GUI Interface** - Přehledné rozhraní pro snadnou konfiguraci
- **Multi-Pool Support** - ZION Pool, NiceHash, MineXMR a další
- **Stratum Protocol** - Plná podpora Stratum v1 protokolu  
- **Real-time Stats** - Živé statistiky hashrate, shares a uptime
- **Config Presets** - Rychlé nastavení pro populární pooly
- **Auto-save Config** - Automatické ukládání nastavení
- **Detailed Logging** - Kompletní logy všech operací
- **CPU Temperature Warning** - Upozornění na přehřátí

## 🚀 Rychlý Start

### Ubuntu/Debian:
```bash
# Spustit launcher (automaticky nainstaluje závislosti)
./zion-miner.sh

# Nebo manuálně:
sudo apt-get install python3-tk
python3 zion-miner.py
```

### Základní Konfigurace:
1. Otevřít záložku "⚙️ Konfigurace"
2. Nastavit Pool Host, Port a Wallet adresu
3. Vybrat počet threads (doporučeno: počet CPU cores)
4. Kliknout "🚀 Start Mining"

## 🔧 Pool Presets

### ZION Pool (vlastní)
- **Host:** 91.98.122.165
- **Port:** 3333
- **Algorithm:** rx/0 (RandomX)
- **Fee:** 2%

### NiceHash
- **Host:** randomxmonero.auto.nicehash.com  
- **Port:** 9200
- **Algorithm:** rx/0 (RandomX)
- **Pozor:** Zapnout "NiceHash kompatibilita"

### MineXMR  
- **Host:** pool.minexmr.com
- **Port:** 4444
- **Algorithm:** rx/0 (RandomX)
- **Fee:** 1%

## 📊 Mining Stats

Aplikace zobrazuje:
- **Status** - Aktuální stav připojení
- **Hashrate** - Výkon v H/s (hashes per second)
- **Shares** - Přijaté/odeslané shares
- **Uptime** - Doba běhu mining session
- **Acceptance Rate** - Úspěšnost shares v %

## 🛠️ Konfigurace

### Wallet Adresa
Pro ZION mining použij ZION wallet adresu. Pro ostatní pooly (NiceHash, MineXMR) použij Monero adresu.

### Worker Name
Identifikace workeru na poolu. Formát: `worker-001`, `rig1` apod.

### Threads
Počet CPU threads pro mining. Doporučení:
- **4-8 cores:** nastavit na počet cores
- **8+ cores:** nechat 1-2 cores pro systém
- **Laptop:** použít 50-75% cores (kvůli teplotě)

### Algorithm
- **rx/0** - RandomX (ZION, Monero)
- **cn/r** - CryptoNight R
- **cn/fast** - CryptoNight Fast
- **argon2/chukwa** - Argon2id Chukwa

## 🔍 Troubleshooting

### "Nepodařilo se připojit k poolu"
1. Zkontroluj internet připojení
2. Ověř host a port poolu
3. Zkus jiný port (pokud pool má více)
4. Zkontroluj firewall

### "Autorizace selhala"
1. Zkontroluj wallet adresu
2. Pro NiceHash zapni "NiceHash kompatibilita"
3. Zkontroluj worker name (bez speciálních znaků)

### "Share odmítnut"
1. Normální je odmítnutí 1-5% shares
2. Více než 10% = problém s připojením nebo HW
3. Zkus snížit počet threads

### Nízký hashrate
1. Zkontroluj vytížení CPU (htop)
2. Zkus snížit počet threads
3. Zavři ostatní náročné aplikace
4. Zkontroluj teplotu CPU

## 📁 Config Files

- **~/.zion-miner-config.ini** - Hlavní konfigurace
- **zion-miner.log** - Logy (při uložení)

## 🔒 Bezpečnost

- Aplikace neposílá privátní klíče
- Ukládá pouze pool nastavení lokálně
- Open source kód pro transparentnost

## 🐛 Bug Reports

Pokud najdeš bug nebo máš návrh na vylepšení:
1. Ulož logy ze záložky "📋 Logy"  
2. Napiš popis problému
3. Přilož system info (OS, CPU, RAM)

## 📈 Performance Tips

### CPU Optimalizace:
```bash
# Nastavit high performance governor
sudo cpupower frequency-set -g performance

# Vypnout CPU throttling
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### Systémové nastavení:
```bash
# Zvýšit prioritu procesu
sudo nice -n -10 python3 zion-miner.py

# Monitoring výkonu
htop
nvidia-smi  # pro GPU monitoring
```

## 🌟 Advanced Features

### Custom Pool konfigurace
Můžeš přidat vlastní pool do config.ini:
```ini
[custom_pool]
host = your.pool.com
port = 4444
algo = rx/0
```

### Batch Mining
Pro mining na více pool současně spusť více instancí s různými konfiguracemi.

---

**ZION Miner 1.4.0** - Professional Mining Solution  
*Compatible with ZION, Monero, NiceHash and more*