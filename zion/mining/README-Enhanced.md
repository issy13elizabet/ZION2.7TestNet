# ZION Real Miner 1.4.0 Enhanced 🔥🌡️

**SKUTEČNÝ Mining Client s Temperature Monitoring + NiceHash Support**

## ⚠️ DŮLEŽITÉ UPOZORNĚNÍ
Toto je **SKUTEČNÝ** mining software, který:
- 🔥 **100% vytíží CPU** - všechny cores
- ⚡ **Spotřebovává hodně elektřiny**
- 🌡️ **Může způsobit přehřátí** - integrovaný monitoring
- 💻 **Zpomalí systém** během mining
- 💎 **Těží skutečné kryptoměny** (ZION/Monero)

## 🌟 Nové Funkce v Enhanced verzi

### 🌡️ Temperature Monitoring
- **Real-time CPU temperature monitoring** pomocí lm-sensors
- **Automatické zastavení** při nebezpečné teplotě (85°C default)
- **Vizuální indikátory** stavu teploty (zelená/oranžová/červená)
- **Konfigurovatelný** maximální bezpečný limit
- **AMD Ryzen podporován** (k10temp driver)

### 💎 NiceHash Integration
- **Plná NiceHash kompatibilita** s RandomX
- **Automatické formatování** username (jen wallet adresa)
- **Preset konfigurace** pro rychlé připojení
- **Stratum protokol optimalizován** pro NiceHash

### 🎯 Pool Support
Předkonfigurované pooly:
- **ZION Pool** - `91.98.122.165:3333`
- **NiceHash RandomX** - `randomxmonero.auto.nicehash.com:9200`
- **MineXMR** - `pool.minexmr.com:4444`

## 📦 Instalace a Spuštění

### Systémové požadavky
```bash
# Ubuntu/Debian dependencies
sudo apt-get update
sudo apt-get install python3 python3-tk lm-sensors

# Temperature monitoring setup (DŮLEŽITÉ!)
sudo sensors-detect --auto
sudo modprobe k10temp  # Pro AMD
sudo modprobe nct6775  # Pro motherboard sensors
```

### Spuštění
1. **Desktop ikona**: `ZION-REAL-Miner-Enhanced.desktop`
2. **Script launcher**: `./zion-real-miner-enhanced.sh`
3. **Přímý spusť**: `python3 zion-real-miner-v2.py`

## 🎮 Použití GUI

### ⚙️ Tab: Konfigurace
- **Pool nastavení**: Host, port, wallet adresa
- **Mining nastavení**: Počet threads (doporučeno = počet CPU cores)
- **Temperature limity**: Bezpečný max (60-95°C)
- **Preset tlačítka**: Rychlé nastavení pro různé pooly

### 🔥 Tab: Real Mining
- **Start/Stop mining** tlačítka
- **Live temperature display** s barevnými indikátory
- **Real-time statistiky**: hashrate, shares, úspěšnost
- **Pool connection test**

### 📋 Tab: Mining Logy
- **Realtime logy** všech mining událostí
- **Export/import** logů
- **Timestamped** záznamy

## 🌡️ Temperature Safety Features

### Automatické monitorování
- **10 sekund interval** čtení teploty
- **Vizuální varování** při přiblížení k limitu
- **Auto-stop mining** při překročení max teploty
- **Emergency popup** při kritické teplotě

### Barevné indikátory
- 🟢 **Zelená**: Teplota OK (< max-15°C)
- 🟡 **Oranžová**: Teplota zvýšená (< max-5°C)
- 🔴 **Červená**: NEBEZPEČNÁ teplota (>= max°C)

## 💎 NiceHash Mining Setup

1. **Registrace** na [NiceHash.com](https://nicehash.com)
2. **Získej Monero adresu** (pro RandomX výplaty)
3. **Nastav v mineru**:
   - Pool: `randomxmonero.auto.nicehash.com:9200`
   - Wallet: Tvá Monero adresa
   - Zaškrtni "NiceHash kompatibilita"
4. **Spusť mining** - automaticky se připojí

## ⚡ Výkon a Optimalizace

### CPU Threads
- **Doporučeno**: Počet fyzických cores (ne hyperthreading)
- **AMD Ryzen 5 3600**: 6 threads optimal
- **Test různé hodnoty** pro najití nejlepšího poměru

### Temperature Management
- **Kvalitní CPU chladič** nutný!
- **Dobré větrání** skříně
- **Monitoring při mining** - sleduj teploty
- **Thermal paste** pravidelně měnit

## 📊 Mining Statistiky

### Zobrazované hodnoty
- **Hashrate**: H/s (hashes per second)
- **Shares**: Přijaté/Odeslané na pool
- **Úspěšnost**: % přijatých shares
- **Uptime**: Doba běhu mining
- **Temperature**: Aktuální CPU teplota

### Výkonnost Ryzen 5 3600
- **Očekávaný hashrate**: 3000-6000 H/s
- **Normální teplota**: 65-80°C
- **Power consumption**: 65W TDP

## 🛡️ Bezpečnost a Doporučení

### ⚠️ Před spuštěním mining
1. **Zkontroluj temperatures**: `sensors` command
2. **Nastav rozumný max limit** (85°C safe pro Ryzen)
3. **Monitoruj první hodinu** provozu
4. **Měj backup plan** pro emergency stop

### 🔒 Wallet Security
- **Nikdy nesdílej** private keys
- **Používej hardware wallets** pro větší částky
- **Double-check** adresy před mining
- **Test malou částkou** před full deployment

### ⚡ Systémová stabilita
- **Adequate PSU** pro mining load
- **Stability testing** před 24/7 mining
- **RAM stress test** - mining je memory intensive
- **Monitor systémové logy** pro errors

## 🏗️ Technické detaily

### RandomX Algorithm
- **CPU-optimized** Proof-of-Work
- **Memory-hard** (2GB+ RAM doporučeno)
- **ASIC-resistant** design
- **Monero/ZION compatible**

### Stratum Protocol
- **Mining.subscribe/authorize** handshake
- **Real-time job notification**
- **Share submission** s difficulty check
- **Error handling** a reconnection

### Temperature Integration
- **lm-sensors** backend pro hardware reading
- **k10temp** driver pro AMD CPUs
- **nct6775** driver pro motherboard sensors
- **Regex parsing** sensors output

## 🐛 Troubleshooting

### Temperature monitoring nefunguje
```bash
# Reinstall sensors
sudo apt-get remove --purge lm-sensors
sudo apt-get install lm-sensors
sudo sensors-detect --auto
sudo service kmod restart
```

### Pool connection selhává
1. **Check firewall**: Allow outgoing port 3333/9200
2. **Test network**: `telnet pool-host port`
3. **Check wallet format**: Správná adresa pro daný pool
4. **Try different pool**: Test jiný server

### Low hashrate
1. **CPU cores**: Nastav correct number threads
2. **Background apps**: Zavři unnecessary programy  
3. **Power settings**: High performance mode
4. **RAM speed**: Ensure optimal memory clocks

### Přehřívání systému
1. **Thermal paste**: Refresh CPU cooler
2. **Case fans**: Improve airflow
3. **Lower threads**: Reduce CPU load
4. **Ambient temp**: Improve room cooling

## 📝 Config soubor

Konfigurce se ukládá v: `~/.zion-real-miner-config.ini`

```ini
[mining]
pool_host = 91.98.122.165
pool_port = 3333
wallet_address = Z3BDEEC2A0AE0F5D...
worker_name = zion-real-miner
threads = 6
algorithm = rx/0
nicehash_mode = false
max_temp = 85
temp_check_interval = 10
```

## 🚀 Co je nového oproti základní verzi

### Enhanced Features
- ✅ **Real-time temperature monitoring** integrován přímo do GUI
- ✅ **NiceHash auto-configuration** s wallet format handling
- ✅ **Safety shutdown** při překročení temperature limits
- ✅ **Barevné temperature indikátory** pro rychlý overview
- ✅ **Improved error handling** a connection recovery
- ✅ **Enhanced logging** s temperature warnings
- ✅ **Desktop integration** s novým launcher scriptí

### Oproti původní verzi
- **Opraveny syntax errors** z předchozí update attempt
- **Kompletní rewrite** mining engine části  
- **Better modularity** - samostatné třídy pro mining a GUI
- **Improved stability** během dlouhodobého mining

## ⚖️ Legal Disclaimer

Tento software je určen pro:
- ✅ **Vzdělávací účely**
- ✅ **Testování mining algoritmů**  
- ✅ **Experimentální cryptocurrency mining**
- ✅ **Personal use s vlastní elektřinou**

⚠️ **UPOZORNĚNÍ**:
- Mining cryptocurrency může být **regulovaným** v některých jurisdikcích
- **Vysoká spotřeba elektřiny** - zkontroluj local energy costs
- **Hardware wear** - mining zkracuje životnost komponentů
- **Tax implications** - mining může podléhat danění
- **Pool terms** - respektuj terms of service jednotlivých poolů

## 🌟 Support a Community

- **GitHub Issues**: Report bugs a feature requests
- **Mining pools**: Připoj se k ZION community
- **Hardware tweaking**: Sdílej optimalizace pro různé CPUs
- **Temperature logs**: Sdílej safe operating ranges

---

**Happy Mining!** 🚀💎

*Vždy mining odpovědně s respektem k hardware limits a local regulations.*