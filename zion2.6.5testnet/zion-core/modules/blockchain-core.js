import express from 'express';

class BlockchainCore {
  constructor() {
    this.status = 'stopped';
    this.height = 0;
    this.difficulty = 1000;
    this.router = express.Router();
    this.setupRoutes();
  }

  async initialize() {
    console.log('⛓️ Initializing blockchain core...');
    this.height = 12345;
    this.status = 'synced';
  }

  setupRoutes() {
    this.router.get('/info', (req, res) => {
      res.json({
        height: this.height,
        difficulty: this.difficulty,
        status: this.status
      });
    });
  }

  getRouter() { return this.router; }
  getStatus() { return { status: this.status, height: this.height }; }
  getStats() { return { height: this.height, difficulty: this.difficulty }; }
  async shutdown() { this.status = 'stopped'; }
}

export default BlockchainCore;