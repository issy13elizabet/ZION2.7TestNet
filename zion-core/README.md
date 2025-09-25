# ZION CORE v2.5.0 - TypeScript Edition

## 🚀 Multi-Chain Dharma Blockchain Platform

ZION CORE je unified TypeScript aplikace, která integruje všechny klíčové komponenty ZION blockchain ekosystému do jediné, vysoce výkonné a type-safe platformy. Tato refaktorizace přináší lepší architekturu, snadnější údržbu a robustní error handling.

### ✨ Klíčové Vlastnosti

#### 🔧 **Unified Architecture**
- **Všechno v jednom**: Mining pool, GPU mining, Lightning Network, Wallet služby, P2P síť, RPC adapter
- **TypeScript Type Safety**: Kompletní type definitions pro všechny interfaces a datové struktury
- **Modulární Design**: Každý modul má jasně definované rozhraní a zodpovědnosti
- **Graceful Shutdown**: Koordinované vypnutí všech modulů s proper cleanup

#### ⚡ **GPU-Accelerated Lightning Network**
- **Multi-Vendor Support**: NVIDIA (CUDA), AMD (OpenCL), Intel Arc (OpenCL)
- **Real-time GPU Monitoring**: Teplota, power consumption, hashrate tracking
- **GPU Benchmarking**: Automatické performance testing
- **Lightning Acceleration**: GPU-accelerované route calculation, cryptography, payment verification

#### ⛏️ **Advanced Mining Pool**
- **Stratum Protocol**: Plná kompatibilita s existujícími mining tools
- **Real-time Statistics**: Miner tracking, hashrate monitoring, share validation
- **Automatic Payouts**: Integrované zpracování výplat s pool fee managementem
- **Block Discovery**: Automatická detekce nových bloků a job distribution

#### 🔗 **Blockchain Core**
- **Multi-Chain Ready**: Připraveno pro cross-chain interoperabilitu
- **RandomX Consensus**: Testnet compatible s full sync capabilities
- **Transaction Management**: Mempool handling, block validation, chain state tracking
- **Network Synchronization**: P2P peer management a blockchain sync

#### 💰 **Integrated Wallet**
- **Balance Management**: Real-time balance tracking včetně unconfirmed transactions
- **Transaction Processing**: Send/receive functionality s automatic fee calculation
- **Security**: Type-safe transaction handling s proper validation

#### 🌐 **RPC Compatibility**
- **Monero-Compatible**: Plná kompatibilita s existujícími mining tools a wallets
- **JSON-RPC 2.0**: Standard compliant RPC interface
- **Legacy Support**: Backwards compatibility s daemon API endpoints

### 🏗️ Architektura

```
ZION CORE v2.5.0 (TypeScript)
├── 🖥️  HTTP Server (Express.js) - Port 8888
├── 🌐 WebSocket Server - Real-time stats
├── ⛏️  Stratum Mining Pool - Port 3333
├── ⚡ Lightning Network - Port 9735
├── 🔌 RPC Adapter - Monero compatibility
├── 🔗 Blockchain Core - Chain state management
├── 🖥️  GPU Mining - Multi-vendor support
├── 💰 Wallet Service - Balance & transactions
└── 🌍 P2P Network - Peer management
```

### 🚀 Quick Start

#### Prerequisites
- Node.js 18+ (doporučeno 20+)
- npm nebo yarn
- TypeScript (automaticky instalováno)

#### Installation & Build
```bash
# Clone repository
git clone <repo-url>
cd Zion-v2.5-Testnet/zion-core

# Install dependencies
npm install

# Build TypeScript
npm run build

# Start ZION CORE
npm start

# Development mode (s hot reload)
npm run dev
```

#### První Spuštění
```bash
# Zkontrolujte, že všechny moduly jsou ready
curl http://localhost:8888/health | jq .

# System statistics
curl http://localhost:8888/api/stats | jq .
```

### 📡 API Endpoints

#### 🏥 Health & System
```bash
# Health check
GET /health

# System statistics (včetně všech modulů)
GET /api/stats
```

#### 🔗 Blockchain
```bash
# Blockchain info
GET /api/blockchain/info
GET /api/blockchain/stats
GET /api/blockchain/status
```

#### ⛏️ Mining Pool
```bash
# Pool statistics
GET /api/mining/stats

# Active miners
GET /api/mining/miners

# Current mining job
GET /api/mining/job

# Shares history
GET /api/mining/shares?limit=100

# Process payout
POST /api/mining/payout
{
  "minerId": "miner_123",
  "amount": 1000000
}
```

#### 🖥️ GPU Mining
```bash
# Available GPUs
GET /api/gpu/devices

# GPU statistics
GET /api/gpu/stats

# Start mining
POST /api/gpu/start
{
  "gpuIds": [0, 1, 2],
  "pool": "stratum+tcp://localhost:3333",
  "wallet": "ZiONWalletAddress...",
  "algorithm": "randomx"
}

# Stop mining
POST /api/gpu/stop
{
  "gpuIds": [0, 1] // optional - všechny GPUs if not specified
}

# GPU benchmark
POST /api/gpu/benchmark
{
  "gpuId": 0,
  "duration": 60 // seconds
}
```

#### ⚡ Lightning Network
```bash
# Lightning stats
GET /api/lightning/stats

# List channels
GET /api/lightning/channels

# Channel details
GET /api/lightning/channels/:channelId

# Open channel
POST /api/lightning/channels/open
{
  "nodeId": "node_abc123...",
  "amount": 1000000,
  "pushAmount": 100000 // optional
}

# Close channel
POST /api/lightning/channels/:channelId/close
{
  "force": false // optional
}

# Create invoice
POST /api/lightning/invoices
{
  "amount": 1000,
  "description": "Coffee payment",
  "expiry": 3600 // optional, default 1 hour
}

# Pay invoice
POST /api/lightning/payments
{
  "bolt11": "lnzion1...",
  "amount": 1000 // optional for zero-amount invoices
}

# Find route
POST /api/lightning/routes/find
{
  "destination": "node_xyz...",
  "amount": 1000,
  "maxFee": 10 // optional
}

# Network info
GET /api/lightning/network
```

#### 💰 Wallet
```bash
# Balance
GET /api/wallet/balance

# Transaction history
GET /api/wallet/transactions

# Send transaction
POST /api/wallet/send
{
  "address": "ZiONAddress...",
  "amount": 1000000
}
```

#### 🌐 P2P Network
```bash
# Peer info
GET /api/p2p/peers

# Connect to peer
POST /api/p2p/connect
{
  "address": "peer.example.com:18080"
}
```

#### 🔌 RPC (Monero Compatible)
```bash
# JSON-RPC endpoint
POST /api/rpc/json_rpc
{
  "jsonrpc": "2.0",
  "method": "get_info",
  "params": {},
  "id": 1
}

# Legacy daemon endpoints
GET /api/rpc/get_info
GET /api/rpc/get_height
```

### 🌐 WebSocket Real-time Updates

```javascript
const ws = new WebSocket('ws://localhost:8888');

ws.onopen = () => {
  // Subscribe to updates
  ws.send(JSON.stringify({ 
    type: 'subscribe' 
  }));
  
  // Request current stats
  ws.send(JSON.stringify({ 
    type: 'stats' 
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch (data.type) {
    case 'welcome':
      console.log('Connected to ZION CORE:', data.data);
      break;
      
    case 'stats_update':
      console.log('Stats update:', data.data);
      // Automatic updates every 30 seconds
      break;
      
    case 'shutdown':
      console.log('Server shutting down');
      break;
  }
};
```

### 🔧 Development

#### Project Structure
```
zion-core/
├── src/
│   ├── server.ts              # Main application entry point
│   ├── types.ts               # TypeScript type definitions
│   └── modules/
│       ├── blockchain-core.ts # Blockchain management
│       ├── mining-pool.ts     # Stratum mining pool
│       ├── gpu-mining.ts      # GPU mining & acceleration
│       ├── lightning-network.ts # Lightning Network layer
│       ├── wallet-service.ts  # Wallet operations
│       ├── p2p-network.ts     # Peer-to-peer networking
│       └── rpc-adapter.ts     # RPC compatibility layer
├── dist/                      # Compiled JavaScript output
├── package.json               # Node.js dependencies
├── tsconfig.json             # TypeScript configuration
└── README.md                 # This file
```

#### Available Scripts
```bash
# Development s hot reload
npm run dev

# Build TypeScript
npm run build

# Start production server
npm start

# Run tests
npm test

# Type checking only (bez compilation)
npm run type-check

# Clean build artifacts
npm run clean
```

#### TypeScript Configuration
- **Target**: ES2022 pro moderní Node.js features
- **Modules**: ESNext s proper import/export
- **Strict Mode**: Enabled pro type safety
- **Source Maps**: Enabled pro debugging
- **Declaration Files**: Generated pro library usage

### ⚡ Performance Features

#### GPU Acceleration
- **Automatic GPU Detection**: Multi-vendor support (NVIDIA, AMD, Intel)
- **Lightning Network Acceleration**: Route calculation, cryptography, payment verification
- **Performance Monitoring**: Real-time GPU statistics a benchmarking
- **Smart GPU Selection**: Automatický výběr nejlepšího GPU pro každou úlohu

#### Optimizations
- **Clustering**: Production mode používá všechny CPU cores
- **WebSocket Efficiency**: Broadcast pouze při změnách
- **Memory Management**: Proper cleanup a garbage collection
- **Error Handling**: Comprehensive error handling s recovery

### 🔒 Security

#### Type Safety
- **Comprehensive Types**: Všechny interfaces a data structures
- **Runtime Validation**: Input validation na všech endpoints
- **Error Boundaries**: Isolated error handling per module

#### Network Security
- **CORS Configuration**: Configurable origins
- **Rate Limiting**: Configurable per endpoint
- **Input Sanitization**: Proper request validation

### 📊 Monitoring & Logging

#### Real-time Statistics
- **System Metrics**: CPU, memory, network usage
- **Blockchain Stats**: Height, difficulty, transaction pool
- **Mining Performance**: Hashrates, shares, block discovery
- **Lightning Network**: Channels, payments, routing performance
- **GPU Metrics**: Temperature, power, performance

#### Logging
- **Structured Logging**: Timestamp, module, level
- **Error Tracking**: Comprehensive error reporting
- **Performance Metrics**: Operation timing a success rates

### 🌍 Network Compatibility

#### Testnet Configuration
- **P2P Port**: 18080 (ZION testnet)
- **RPC Port**: 18081 (daemon compatibility)
- **Pool Port**: 3333 (mining)
- **Lightning Port**: 9735 (standard)
- **Core Port**: 8888 (unified API)

#### Cross-Platform
- **macOS**: Native support (tested on Apple Silicon)
- **Linux**: Full compatibility
- **Windows**: WSL recommended
- **Docker**: Container ready

### 🚀 Deployment

#### Production Setup
```bash
# Environment variables
export NODE_ENV=production
export PORT=8888
export CORS_ORIGINS="https://yourapp.com"

# Start with clustering
npm start
```

#### Docker Deployment
```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY dist ./dist
EXPOSE 8888 3333 9735
CMD ["npm", "start"]
```

### 📈 Roadmap

#### Phase 1 (Current - TypeScript Foundation) ✅
- ✅ Complete TypeScript conversion
- ✅ Unified architecture
- ✅ Basic GPU acceleration
- ✅ Lightning Network integration
- ✅ RPC compatibility

#### Phase 2 (Q1 2025)
- 🔄 Real GPU mining implementation
- 🔄 Advanced Lightning Network routing
- 🔄 Cross-chain bridge capabilities
- 🔄 Enhanced security features
- 🔄 Performance optimizations

#### Phase 3 (Q2 2025)
- 📋 Mainnet deployment
- 📋 Mobile wallet integration
- 📋 DeFi protocol integration
- 📋 Governance token implementation
- 📋 Enterprise features

### 🤝 Contributing

ZION CORE je open-source projekt. Contributions jsou vítány!

#### Development Setup
```bash
git clone <repo-url>
cd zion-core
npm install
npm run dev
```

#### Code Style
- TypeScript strict mode
- ESLint for code quality
- Prettier for formatting
- Comprehensive JSDoc comments

### 📄 License

MIT License - Viz LICENSE soubor pro details.

### 🙏 Acknowledgments

- **RandomX Algorithm**: Monero project
- **Lightning Network**: Bitcoin Lightning specification
- **GPU Mining**: Various mining software projects
- **TypeScript Community**: Type definitions a best practices

---

**ZION CORE v2.5.0** - Pioneering the future of multi-chain dharma blockchain technology with TypeScript excellence.

*Jai Ram Ram Ram Sita Ram Ram Ram Hanuman!* 🕉️