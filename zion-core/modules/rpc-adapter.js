import express from 'express';

class RPCAdapter {
  constructor() {
    this.status = 'stopped';
    this.router = express.Router();
    this.setupRoutes();
  }

  async initialize() {
    console.log('🔄 Initializing RPC adapter...');
    this.status = 'ready';
  }

  setupRoutes() {
    this.router.post('/', (req, res) => {
      const { method, params, id } = req.body;
      
      switch (method) {
        case 'get_height':
          res.json({ id, result: { height: 12345 }, error: null });
          break;
        case 'get_info':
          res.json({ 
            id, 
            result: { 
              height: 12345, 
              difficulty: 1000,
              tx_count: 5678,
              tx_pool_size: 12
            }, 
            error: null 
          });
          break;
        default:
          res.json({ id, result: null, error: { code: -32601, message: 'Method not found' } });
      }
    });
  }

  getRouter() { return this.router; }
  getStatus() { return { status: this.status }; }
  async shutdown() { this.status = 'stopped'; }
}

export default RPCAdapter;