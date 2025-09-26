# 🧪 Testing Configuration - ZION 2.6 TestNet

JSON konfigurace pro testování různých komponent ZION ekosystému.

## 📁 **Obsah**

### ⛏️ **Mining Tests**
- `test-mining.json` - Základní mining test konfigurace
- `test-xmrig.json` - XMRig specifická test konfigurace

## 📋 **Konfigurace**

### Mining Test (`test-mining.json`)
```json
{
  "pool": "stratum+tcp://localhost:3333",
  "algorithm": "randomx",
  "threads": 1,
  "test_duration": 60
}
```

### XMRig Test (`test-xmrig.json`)  
```json
{
  "api": {
    "port": 8080,
    "access-token": "test-token"
  },
  "pools": [{
    "url": "localhost:3333",
    "user": "test-wallet"
  }]
}
```

## 🚀 **Použití**

```bash
# Test s mining konfigurací
xmrig --config=config/testing/test-xmrig.json

# Načtení konfigurace v test skriptech  
CONFIG=$(cat config/testing/test-mining.json)
```

## 🔧 **Customizace**

Pro vlastní testy:
1. Zkopírujte existující konfiguraci
2. Upravte parametry podle potřeby
3. Použijte v test skriptech

## ⚠️ **Poznámky**

- Test konfigurace používají nízké hodnoty pro bezpečnost
- Nepoužívejte v produkci
- Test wallet adresy nejsou skutečné