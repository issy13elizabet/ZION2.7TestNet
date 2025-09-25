import { Router } from 'express';
import {
  IWalletService,
  WalletBalance,
  Transaction,
  ModuleStatus,
  WalletError
} from '../types.js';

/**
 * ZION Wallet Service Module
 * 
 * Wallet management, balance tracking, and transaction handling.
 */
export class WalletService implements IWalletService {
  private status: ModuleStatus['status'] = 'stopped';
  private startTime: number = 0;
  private router: Router;
  
  // Wallet state
  private balance: WalletBalance = {
    balance: 1000000, // 1M ZION
    unconfirmed: 0,
    locked: 0
  };
  
  private transactions: Map<string, Transaction> = new Map();

  constructor() {
    this.router = Router();
    this.setupRoutes();
  }

  private setupRoutes(): void {
    this.router.get('/balance', (_req, res) => {
      res.json(this.getBalance());
    });

    this.router.get('/transactions', (_req, res) => {
      res.json(Array.from(this.transactions.values()));
    });

    this.router.post('/send', async (_req, res) => {
      try {
        const { address, amount } = _req.body;
        const transaction = await this.sendTransaction(address, amount);
        res.json(transaction);
      } catch (error) {
        const err = error as Error;
        res.status(400).json({ error: err.message });
      }
    });
  }

  public async initialize(): Promise<void> {
    console.log('💰 Initializing Wallet Service...');
    
    try {
      this.status = 'starting';
      this.startTime = Date.now();
      
      await this.loadWalletData();
      
      this.status = 'ready';
      console.log('✅ Wallet Service initialized successfully');
      
    } catch (error) {
      this.status = 'error';
      const err = error as Error;
      console.error('❌ Failed to initialize Wallet Service:', err.message);
      throw new WalletError(`Wallet initialization failed: ${err.message}`, 'WALLET_INIT_FAILED');
    }
  }

  public async shutdown(): Promise<void> {
    console.log('💰 Shutting down Wallet Service...');
    
    try {
      this.status = 'stopped';
      // Save wallet data
      console.log('✅ Wallet Service shut down successfully');
      
    } catch (error) {
      const err = error as Error;
      console.error('❌ Error shutting down Wallet Service:', err.message);
      throw error;
    }
  }

  public getStatus(): ModuleStatus {
    return {
      status: this.status,
      uptime: this.startTime ? Date.now() - this.startTime : 0
    };
  }

  public getBalance(): WalletBalance {
    return this.balance;
  }

  public async sendTransaction(address: string, amount: number): Promise<Transaction> {
    if (amount > this.balance.balance) {
      throw new WalletError('Insufficient balance', 'INSUFFICIENT_BALANCE');
    }

    const transaction: Transaction = {
      id: `tx_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      amount,
      address,
      confirmations: 0,
      timestamp: Date.now(),
      fee: Math.floor(amount * 0.001) // 0.1% fee
    };

    this.transactions.set(transaction.id, transaction);
    this.balance.balance -= amount;
    this.balance.unconfirmed += amount;

    // Simulate confirmation
    setTimeout(() => {
      transaction.confirmations = 6;
      this.balance.unconfirmed -= amount;
    }, 5000);

    return transaction;
  }

  public getRouter(): Router {
    return this.router;
  }

  private async loadWalletData(): Promise<void> {
    // Load wallet data from storage
    await new Promise(resolve => setTimeout(resolve, 500));
  }
}