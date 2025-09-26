# 🧪 Testing Scripts - ZION 2.6 TestNet

Test skripty pro ověření funkčnosti ZION komponenty.

## 📁 **Obsah**

### 🔗 **Core Testing**
- `test_zion.sh` - Hlavní test skript pro ZION core

## 🚀 **Použití**

```bash
# Spuštění core testů
./scripts/testing/test_zion.sh

# Nebo z root adresáře
bash scripts/testing/test_zion.sh
```

## 🔧 **Test Configuration**

Test konfigurace jsou v `config/testing/`:
- `test-mining.json` - Mining test konfigurace
- `test-xmrig.json` - XMRig test setup

## 📋 **Co se testuje**

- Blockchain synchronizace
- Mining pool konektivita
- Wallet funkčnost
- RPC endpoints
- Network komunikace
- Docker containers health

## 🎯 **Test Coverage**

- Unit testy pro core komponenty
- Integration testy pro services
- End-to-end testy pro user workflows
- Performance testy pro mining
- Security testy pro network layer

## 📊 **Reporting**

Testy generují:
- Detailní logy v `/tmp/zion-tests/`
- Performance metriky
- Error reports
- Coverage statistics