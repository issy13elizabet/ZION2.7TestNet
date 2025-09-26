# 🪟 Windows Scripts - ZION 2.6 TestNet

Kolekce Windows batch (.bat) souborů pro správu ZION ekosystému na Windows platformách.

## 📁 **Struktura**

```
scripts/windows/
├── README.md                   # Tento soubor
├── start-zion-pool.bat        # Spuštění ZION mining poolu
├── mining/                     # Mining skripty
│   ├── start-ai-gpu-hybrid.bat     # AI + GPU hybrid mining
│   ├── start-dual-mining.bat       # Dual algoritmus mining
│   └── start-multi-algo-mining.bat # Multi-algoritmus mining
├── testing/                    # Test skripty
│   ├── test-srb-ergo.bat           # Test SRB Ergo
│   ├── test-srb-kawpow.bat         # Test SRB KawPow
│   ├── test-srb-octopus.bat        # Test SRB Octopus
│   └── test-zion-addresses.bat     # Test ZION adres
└── setup/                      # Setup skripty
    └── srbminer-setup-complete.bat # Kompletní SRB Miner setup
```

## 🚀 **Použití**

### Pool Management
```cmd
# Spuštění ZION mining poolu
scripts\windows\start-zion-pool.bat
```

### Mining
```cmd
# Multi-algoritmus mining
scripts\windows\mining\start-multi-algo-mining.bat

# Dual mining
scripts\windows\mining\start-dual-mining.bat

# AI + GPU hybrid
scripts\windows\mining\start-ai-gpu-hybrid.bat
```

### Testing
```cmd
# Test různých algoritmů
scripts\windows\testing\test-srb-ergo.bat
scripts\windows\testing\test-srb-kawpow.bat
scripts\windows\testing\test-srb-octopus.bat

# Test ZION adres
scripts\windows\testing\test-zion-addresses.bat
```

### Setup
```cmd
# Kompletní setup SRB Mineru
scripts\windows\setup\srbminer-setup-complete.bat
```

## 📋 **Poznámky**

- Všechny skripty vyžadují Windows prostředí
- Pro mining je potřeba správně nakonfigurované GPU
- Pool skripty vyžadují Docker nebo přímé spuštění služeb
- Test skripty slouží k ověření funkčnosti před produkčním použitím

## 🔗 **Související**

- [Mining dokumentace](../../mining/)
- [Pool konfigurace](../../pool/)
- [Docker setup](../../docker/)