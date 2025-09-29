import express from 'express';

class WalletService {
  constructor() {
    this.status = 'stopped';
    this.balance = 1000000000; // 1000 ZION
    this.router = express.Router();
    this.setupRoutes();
  }

  async initialize() {
    console.log('💳 Initializing wallet service...');
    this.status = 'ready';
  }

  setupRoutes() {
    this.router.get('/balance', (req, res) => {
      res.json({ balance: this.balance });
    });
    
    this.router.post('/send', (req, res) => {
      const { address, amount } = req.body;
      res.json({ 
        success: true, 
        txid: 'mock_tx_' + Date.now(),
        amount,
        address 
      });
    });
  }

  getRouter() { return this.router; }
  getStatus() { return { status: this.status, balance: this.balance }; }
  async shutdown() { this.status = 'stopped'; }
}

export default WalletService;