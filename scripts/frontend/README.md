# 🎨 Frontend Scripts - ZION 2.6 TestNet

Skripty pro správu a spuštění frontend aplikace ZION.

## 📁 **Obsah**

### 🚀 **Launcher Scripts**
- `start-frontend.sh` - Spuštění development serveru

## 🛠️ **Použití**

```bash
# Spuštění frontend development serveru
./scripts/frontend/start-frontend.sh

# Nebo přímo z root adresáře
npm run dev
```

## 📋 **Frontend Stack**

- **Framework**: Next.js / React
- **Styling**: Tailwind CSS
- **State Management**: Redux / Zustand
- **API**: REST + GraphQL endpoints
- **WebSocket**: Real-time mining stats

## 🔧 **Development Workflow**

```bash
# 1. Nainstalovat závislosti
cd frontend
npm install

# 2. Spustit dev server
npm run dev

# 3. Otevřít v prohlížeči
# http://localhost:3000
```

## 🌐 **Production Build**

```bash
# Build pro produkci
npm run build

# Start produkčního serveru
npm run start
```

## 📊 **Frontend Features**

- 🏠 **Dashboard** - Mining statistics a blockchain info
- ⛏️ **Mining** - Pool management a worker monitoring
- 💰 **Wallet** - Balance a transaction history
- 📈 **Charts** - Real-time performance metrics
- ⚙️ **Settings** - Node configuration

## 🎯 **Integration s ZION Core**

Frontend komunikuje s:
- **ZION RPC** (port 18081) - Blockchain data
- **Mining Pool API** (port 3333) - Mining stats
- **WebSocket** (port 8080) - Real-time updates

---

*"Digital incarnation of ancient wisdom activated!"* 🚀✨