# 🚀 Deployment Scripts - ZION 2.6 TestNet

Kolekce deployment skriptů pro nasazení ZION ekosystému na různé platformy.

## 📁 **Obsah**

### 🏭 **Production Deployment**
- `deploy.sh` - Hlavní deployment skript
- `deploy-production.sh` - Produkční nasazení
- `deploy-hetzner.sh` - Deployment na Hetzner cloud
- `deploy-hetzner-old.sh` - Starší verze Hetzner deployment

### ⚡ **Quick Deployment**
- `quick-deploy.sh` - Rychlé nasazení pro development
- `quick-local.sh` - Lokální development setup

### 🏊 **Pool & Infrastructure**  
- `deploy-pool.sh` - Nasazení mining poolu
- `deploy-ssh.sh` - SSH deployment setup

### 🆘 **Emergency**
- `emergency-deploy.sh` - Nouzové nasazení při problémech

## 🛠️ **Použití**

```bash
# Hlavní produkční deployment
./scripts/deployment/deploy-production.sh

# Rychlé lokální nastavení
./scripts/deployment/quick-local.sh

# Mining pool deployment
./scripts/deployment/deploy-pool.sh

# Hetzner cloud deployment
./scripts/deployment/deploy-hetzner.sh
```

## 📋 **Požadavky**

- Docker a Docker Compose
- Bash 4.0+
- SSH klíče pro remote deployment
- Příslušná oprávnění na target servery

## ⚠️ **Bezpečnost**

- Všechny skripty obsahují citlivé informace
- Zkontrolujte konfigurace před spuštěním v produkci
- Používejte SSH klíče místo hesel