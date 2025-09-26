# ⚡ Lightning Network Configuration - ZION 2.6 TestNet

Konfigurační soubory pro Lightning Network integraci v ZION ekosystému.

## 📁 **Obsah**

### ₿ **Bitcoin Core**
- `bitcoin.conf` - Konfigurace Bitcoin Core pro Lightning Network

### ⚡ **Lightning Network Daemon (LND)**
- `lnd.conf` - Konfigurace LND pro ZION network

## 🔧 **Konfigurace**

### Bitcoin Core (`bitcoin.conf`)
```properties
# Testnet mode pro development
testnet=1
server=1
daemon=1
txindex=1

# RPC přístup
rpcuser=zion
rpcpassword=secure_password
rpcbind=0.0.0.0:18332
```

### LND (`lnd.conf`)
```properties
# Lightning Network Daemon
debuglevel=info
logdir=/root/.lnd/logs

# Network settings
listen=0.0.0.0:9735
rpclisten=0.0.0.0:10009
restlisten=0.0.0.0:8080
```

## 🚀 **Deployment**

Soubory jsou automaticky používány v Docker Compose:

```yaml
services:
  bitcoin:
    volumes:
      - ./config/lightning/bitcoin.conf:/root/.bitcoin/bitcoin.conf
  
  lnd:
    volumes:
      - ./config/lightning/lnd.conf:/root/.lnd/lnd.conf
```

## 🔐 **Bezpečnost**

⚠️ **Důležité:**
- Změňte default hesla před produkčním nasazením
- Používejte SSL certifikáty pro RPC
- Omezit přístup k RPC portům firewallem

## 📋 **Lightning Network Workflow**

1. **Bitcoin Core** - Poskytuje blockchain data
2. **LND** - Spravuje Lightning Network kanály  
3. **ZION** - Využívá Lightning pro rychlé platby

---

*"Thunder bridges connect mortal realm to star nations!"* ⚡🌟