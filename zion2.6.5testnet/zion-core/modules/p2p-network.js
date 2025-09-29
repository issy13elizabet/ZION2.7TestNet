import express from 'express';

class P2PNetwork {
  constructor() {
    this.status = 'stopped';
    this.peers = new Map();
    this.router = express.Router();
    this.setupRoutes();
  }

  async initialize() {
    console.log('📡 Initializing P2P network...');
    this.status = 'ready';
  }

  setupRoutes() {
    this.router.get('/peers', (req, res) => {
      res.json({ peer_count: this.peers.size });
    });
  }

  getRouter() { return this.router; }
  getStatus() { return { status: this.status, peers: this.peers.size }; }
  async shutdown() { this.status = 'stopped'; }
}

export default P2PNetwork;