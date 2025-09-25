# ⚡🔄 ZION ↔ BTC Atomic Swap Service 🔄⚡

**Jai Ram Ram Ram Sita Ram Ram Ram Hanuman!** 🙏

## 🌟 Koncept: Trustless Cross-Chain Swaps

Atomic Swap Protocol pro instant ZION ↔ Bitcoin výměny bez třetí strany!

### 🎯 Architektura

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ZION Chain    │    │   Swap Service  │    │  Bitcoin Chain  │
│                 │    │                 │    │                 │
│  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌───────────┐  │
│  │ HTLC ZION │◄─┼────┼─►│Coordinator│◄─┼────┼─►│ HTLC BTC  │  │
│  └───────────┘  │    │  └───────────┘  │    │  └───────────┘  │
│                 │    │                 │    │                 │
│  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌───────────┐  │
│  │User ZION  │  │    │  │ Rate Feed │  │    │  │User BTC   │  │
│  │Wallet     │  │    │  │Oracle     │  │    │  │Wallet     │  │
│  └───────────┘  │    │  └───────────┘  │    │  └───────────┘  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Atomic Swap Protocol Flow

### 1. Swap Initiation
```
User A (ZION) wants to swap 1000 ZION for BTC
User B (BTC) wants to swap 0.001 BTC for ZION

1. Generate shared secret hash H(s)
2. User A locks ZION in HTLC with hash H(s) 
3. User B locks BTC in HTLC with same hash H(s)
4. Both HTLCs have timelock (24 hours)
```

### 2. Swap Execution
```
5. User B reveals secret 's' to claim ZION
6. User A uses revealed 's' to claim BTC
7. Both users get their desired coins
8. Swap completed trustlessly!
```

### 3. Safety Mechanisms
```
- Timelock refund if swap fails
- Cryptographic proof of funds
- No custodial risk
- Atomic execution
```

## ⚡ Technical Implementation

### Go Swap Coordinator Service

```go
type AtomicSwap struct {
    SwapID        string    `json:"swap_id"`
    InitiatorAddr string    `json:"initiator_addr"`
    ResponderAddr string    `json:"responder_addr"`
    ZionAmount    uint64    `json:"zion_amount"`
    BtcAmount     uint64    `json:"btc_amount"`
    SecretHash    []byte    `json:"secret_hash"`
    Status        string    `json:"status"`
    CreatedAt     time.Time `json:"created_at"`
    ExpiresAt     time.Time `json:"expires_at"`
}

type SwapCoordinator struct {
    zionClient *ZionClient
    btcClient  *BitcoinClient  
    swaps      map[string]*AtomicSwap
    rateFeed   *RateFeed
}

func (sc *SwapCoordinator) InitiateSwap(zionAddr, btcAddr string, zionAmount uint64) (*AtomicSwap, error) {
    // Generate secret and hash
    secret := generateSecret()
    secretHash := sha256.Sum256(secret)
    
    // Calculate BTC amount from rate feed
    rate := sc.rateFeed.GetZionBtcRate()
    btcAmount := uint64(float64(zionAmount) * rate)
    
    // Create swap
    swap := &AtomicSwap{
        SwapID:        generateSwapID(),
        InitiatorAddr: zionAddr,
        ResponderAddr: btcAddr,
        ZionAmount:    zionAmount,
        BtcAmount:     btcAmount,
        SecretHash:    secretHash[:],
        Status:        "pending",
        CreatedAt:     time.Now(),
        ExpiresAt:     time.Now().Add(24 * time.Hour),
    }
    
    // Lock ZION in HTLC
    err := sc.zionClient.CreateHTLC(zionAddr, zionAmount, secretHash[:], swap.ExpiresAt)
    if err != nil {
        return nil, err
    }
    
    swap.Status = "zion_locked"
    sc.swaps[swap.SwapID] = swap
    
    return swap, nil
}

func (sc *SwapCoordinator) AcceptSwap(swapID, btcAddr string) error {
    swap := sc.swaps[swapID]
    if swap == nil {
        return errors.New("swap not found")
    }
    
    // Lock BTC in HTLC
    err := sc.btcClient.CreateHTLC(btcAddr, swap.BtcAmount, swap.SecretHash, swap.ExpiresAt)
    if err != nil {
        return err
    }
    
    swap.Status = "both_locked"
    return nil
}

func (sc *SwapCoordinator) ClaimZion(swapID string, secret []byte) error {
    swap := sc.swaps[swapID]
    
    // Verify secret hash
    if !bytes.Equal(sha256.Sum256(secret)[:], swap.SecretHash) {
        return errors.New("invalid secret")
    }
    
    // Claim ZION
    err := sc.zionClient.ClaimHTLC(swap.InitiatorAddr, secret)
    if err != nil {
        return err
    }
    
    swap.Status = "zion_claimed"
    
    // Auto-claim BTC for initiator
    go sc.btcClient.ClaimHTLC(swap.ResponderAddr, secret)
    
    return nil
}
```

### ZION HTLC Smart Contract

```javascript
// ZION blockchain HTLC implementation
class ZionHTLC {
    constructor(sender, receiver, amount, secretHash, timelock) {
        this.sender = sender;
        this.receiver = receiver;
        this.amount = amount;
        this.secretHash = secretHash;
        this.timelock = timelock;
        this.claimed = false;
        this.refunded = false;
    }
    
    claim(secret) {
        // Verify secret hash
        const hash = crypto.createHash('sha256').update(secret).digest();
        if (!hash.equals(this.secretHash)) {
            throw new Error('Invalid secret');
        }
        
        if (this.claimed || this.refunded) {
            throw new Error('HTLC already settled');
        }
        
        // Transfer ZION to receiver
        this.transferZion(this.sender, this.receiver, this.amount);
        this.claimed = true;
        
        return true;
    }
    
    refund() {
        if (Date.now() < this.timelock) {
            throw new Error('Timelock not expired');
        }
        
        if (this.claimed || this.refunded) {
            throw new Error('HTLC already settled');
        }
        
        // Refund ZION to sender
        this.transferZion(this.sender, this.sender, this.amount);
        this.refunded = true;
        
        return true;
    }
}
```

### Bitcoin HTLC Script

```javascript
// Bitcoin Script HTLC
const bitcoinHTLC = `
OP_IF
    OP_SHA256 <secret_hash> OP_EQUALVERIFY 
    <receiver_pubkey> OP_CHECKSIG
OP_ELSE
    <timelock> OP_CHECKLOCKTIMEVERIFY OP_DROP
    <sender_pubkey> OP_CHECKSIG
OP_ENDIF
`;

// Create Bitcoin HTLC transaction
function createBitcoinHTLC(senderAddr, receiverAddr, amount, secretHash, timelock) {
    const script = bitcoin.script.compile([
        bitcoin.opcodes.OP_IF,
        bitcoin.opcodes.OP_SHA256,
        secretHash,
        bitcoin.opcodes.OP_EQUALVERIFY,
        bitcoin.address.toOutputScript(receiverAddr),
        bitcoin.opcodes.OP_ELSE,
        bitcoin.script.number.encode(timelock),
        bitcoin.opcodes.OP_CHECKLOCKTIMEVERIFY,
        bitcoin.opcodes.OP_DROP,
        bitcoin.address.toOutputScript(senderAddr),
        bitcoin.opcodes.OP_ENDIF
    ]);
    
    return script;
}
```

## 🌐 REST API Endpoints

```typescript
// Swap Service API
interface SwapAPI {
    // Create new swap
    POST /api/v1/swap/create
    {
        "zion_address": "ZionAddr123",
        "btc_address": "bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh", 
        "zion_amount": 1000000,
        "desired_btc_amount": 100000  // satoshis
    }
    
    // Accept existing swap
    POST /api/v1/swap/{swap_id}/accept
    {
        "btc_address": "bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh"
    }
    
    // Claim ZION with secret
    POST /api/v1/swap/{swap_id}/claim
    {
        "secret": "hex_secret_string"
    }
    
    // Get swap status
    GET /api/v1/swap/{swap_id}
    
    // List active swaps
    GET /api/v1/swaps?status=pending
    
    // Get current exchange rate
    GET /api/v1/rate/zion-btc
}
```

## 📱 Frontend Integration

```typescript
// Swap Widget v Stargate Portal
const SwapWidget = () => {
    const [swapAmount, setSwapAmount] = useState(0);
    const [swapDirection, setSwapDirection] = useState('ZION_TO_BTC');
    const [currentRate, setCurrentRate] = useState(0);
    
    const initiateSwap = async () => {
        const response = await fetch('/api/v1/swap/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                zion_address: wallet.zionAddress,
                btc_address: wallet.btcAddress,
                zion_amount: swapAmount,
                desired_btc_amount: swapAmount * currentRate
            })
        });
        
        const swap = await response.json();
        
        // Show QR code for counterparty
        showSwapQRCode(swap.swap_id);
    };
    
    return (
        <div className="swap-widget">
            <h3>⚡ Atomic Swap ZION ↔ BTC ⚡</h3>
            
            <div className="swap-input">
                <input 
                    type="number" 
                    value={swapAmount}
                    onChange={(e) => setSwapAmount(e.target.value)}
                    placeholder="Amount to swap"
                />
                <select 
                    value={swapDirection}
                    onChange={(e) => setSwapDirection(e.target.value)}
                >
                    <option value="ZION_TO_BTC">ZION → BTC</option>
                    <option value="BTC_TO_ZION">BTC → ZION</option>
                </select>
            </div>
            
            <div className="rate-display">
                Rate: 1 ZION = {currentRate} BTC
                Output: {swapAmount * currentRate} BTC
            </div>
            
            <button onClick={initiateSwap} className="swap-button">
                🔄 Initiate Atomic Swap
            </button>
        </div>
    );
};
```

## 🔥 Advanced Features

### 1. Multi-Hop Swaps
```
ZION → Lightning BTC → On-chain BTC
User gets best liquidity across all Bitcoin layers
```

### 2. Partial Swaps
```
Large swaps split into smaller atomic operations
Better price discovery and liquidity
```

### 3. Swap Marketplace
```
Users can post swap offers
Automated matching system
Competitive rates
```

### 4. Cross-Chain DeFi
```
ZION liquidity pools on Bitcoin
Bitcoin-backed ZION staking
Cross-chain yield farming
```

## 🛡️ Security Features

- **Cryptographic Secrets** - SHA256 hash locks
- **Timelock Safety** - Automatic refunds
- **No Custodial Risk** - Users control private keys
- **Atomic Execution** - All or nothing swaps
- **Rate Protection** - Price oracle integration
- **MEV Protection** - Private mempool for swaps

## 🚀 Deployment

```yaml
# docker-compose.swap.yml
version: '3.8'

services:
  zion-swap-service:
    build: ./swap-service
    ports:
      - "8091:8091"
    environment:
      - ZION_RPC_URL=http://zion-rpc-shim:18089
      - BITCOIN_RPC_URL=http://bitcoind:8332
      - SWAP_PORT=8091
      - RATE_ORACLE_URL=https://api.coingecko.com/api/v3
    depends_on:
      - bitcoind
      - zion-rpc-shim
    networks:
      - zion-lightning
      
  swap-frontend:
    build: ./swap-frontend  
    ports:
      - "3335:3335"
    environment:
      - SWAP_API_URL=http://zion-swap-service:8091
```

**⚡🔄 JAI RAM RAM RAM SITA RAM RAM RAM HANUMAN! 🔄⚡**

Chceš, abych implementoval celou atomic swap službu? 🚀