// ZION Core TypeScript Types and Interfaces

export interface ZionConfig {
  readonly port: number;
  readonly nodeEnv: 'development' | 'production' | 'test';
  readonly corsOrigins: string[];
  readonly rateLimit: {
    windowMs: number;
    max: number;
  };
}

export interface ModuleStatus {
  readonly status: 'stopped' | 'starting' | 'ready' | 'error' | 'syncing' | 'mining';
  readonly uptime?: number;
  readonly lastError?: string;
}

export interface SystemStats {
  readonly system: {
    cpu: {
      manufacturer: string;
      brand: string;
      cores: number;
      speed: number;
    };
    memory: {
      total: number;
      used: number;
      free: number;
    };
    network: Record<string, unknown>;
  };
  readonly blockchain: BlockchainStats;
  readonly mining: MiningStats;
  readonly lightning: LightningStats;
  readonly gpu: GPUStats;
}

// Blockchain Types
export interface BlockchainStats {
  readonly height: number;
  readonly difficulty: number;
  readonly txCount?: number;
  readonly txPoolSize?: number;
}

export interface Block {
  readonly height: number;
  readonly hash: string;
  readonly timestamp: number;
  readonly difficulty: number;
  readonly reward: number;
  readonly minerId?: string;
}

// Mining Types
export interface MiningStats {
  readonly hashrate: number;
  readonly minersActive: number;
  readonly sharesAccepted: number;
  readonly sharesRejected: number;
  readonly blocksFound: number;
  readonly lastBlock: number | null;
}

export interface Miner {
  readonly id: string;
  address: string | null;
  worker: string | null;
  hashrate: number;
  shares: number;
  lastActivity: number;
  readonly connectedAt: number;
  readonly difficulty: number;
  subscribed: boolean;
  socket?: unknown; // Node.js socket
  algorithm?: string; // Multi-algo support
}

export interface MiningJob {
  readonly id: string;
  readonly prevhash: string;
  readonly coinb1: string;
  readonly coinb2: string;
  readonly merkleBranch: string[];
  readonly version: string;
  readonly nbits: string;
  readonly ntime: string;
  readonly cleanJobs: boolean;
  readonly createdAt: number;
  algorithm?: string; // Multi-algo support
}

export interface Share {
  readonly minerId: string;
  readonly jobId: string;
  readonly nonce: string;
  readonly timestamp: number;
  readonly difficulty: number;
  algorithm?: string; // Multi-algo support
}

// GPU Mining Types
export interface GPUStats {
  readonly gpus: GPUDevice[];
  readonly totalHashrate: number;
  readonly powerUsage: number;
  readonly averageTemperature: number;
}

export interface GPUDevice {
  readonly id: number;
  readonly name: string;
  readonly vendor: string;
  readonly vram: number;
  readonly bus: string;
  readonly supported: boolean;
  status: 'idle' | 'mining' | 'benchmark' | 'error';
  hashrate: number;
  power: number;
  temperature: number;
}

export interface GPUBenchmarkResult {
  readonly gpuId: number;
  readonly gpu: string;
  readonly duration: number;
  readonly hashrate: number;
  readonly powerUsage: number;
  readonly temperature: number;
  readonly efficiency: number;
  readonly score: number;
}

export interface MiningConfig {
  readonly gpuIds: number[];
  readonly pool: string;
  readonly wallet: string;
  readonly algorithm: string;
}

// Lightning Network Types
export interface LightningStats {
  readonly channelsActive: number;
  readonly channelsPending: number;
  readonly paymentsSent: number;
  readonly paymentsReceived: number;
  readonly totalVolume: number;
  readonly feesEarned: number;
  readonly gpuAcceleration: boolean;
  readonly networkNodes: number;
}

export interface LightningChannel {
  readonly id: string;
  readonly nodeId: string;
  readonly fundingTxid: string;
  readonly capacity: number;
  localBalance: number;
  remoteBalance: number;
  status: 'pending' | 'active' | 'closing' | 'force_closing' | 'closed';
  readonly createdAt: number;
  confirmations: number;
}

export interface LightningInvoice {
  readonly id: string;
  readonly paymentHash: string;
  readonly amount: number;
  readonly description: string;
  readonly expiry: number;
  readonly createdAt: number;
  status: 'pending' | 'paid' | 'expired';
  readonly bolt11: string;
}

export interface LightningPayment {
  readonly id: string;
  readonly bolt11: string;
  readonly amount: number;
  readonly destination: string;
  readonly route: RouteHop[];
  readonly fee: number;
  status: 'pending' | 'succeeded' | 'failed';
  readonly createdAt: number;
}

export interface RouteHop {
  readonly nodeId: string;
  readonly channelId: string;
  readonly fee: number;
  readonly delay: number;
}

export interface LightningRoute {
  readonly success: boolean;
  readonly path?: RouteHop[];
  readonly fee?: number;
  readonly totalDelay?: number;
  gpuAccelerated?: boolean;
  readonly error?: string;
  readonly maxFee?: number;
  readonly calculatedFee?: number;
}

// Network Types
export interface NetworkNode {
  readonly alias: string;
  readonly channels: number;
  readonly capacity: number;
  readonly lastUpdate: number;
}

// Wallet Types
export interface WalletBalance {
  balance: number;
  unconfirmed: number;
  locked: number;
}

export interface Transaction {
  readonly id: string;
  readonly amount: number;
  readonly address: string;
  confirmations: number;
  readonly timestamp: number;
  readonly fee?: number;
}

// WebSocket Types
export interface WebSocketMessage {
  readonly type: string;
  readonly data?: unknown;
  readonly timestamp?: number;
}

export interface WebSocketResponse {
  readonly type: string;
  readonly data: unknown;
  readonly timestamp: number;
}

// Module Interfaces
export interface IZionModule {
  initialize(): Promise<void>;
  shutdown(): Promise<void>;
  getStatus(): ModuleStatus;
  getRouter?(): unknown; // Express router
}

export interface IBlockchainCore extends IZionModule {
  getStats(): BlockchainStats;
  getHeight(): number;
  getDifficulty(): number;
}

export interface IMiningPool extends IZionModule {
  getStats(): MiningStats;
  getMiners(): Miner[];
  processPayout(minerId: string, amount: number): Promise<Transaction>;
}

export interface IGPUMining extends IZionModule {
  getStats(): GPUStats;
  startMining(config: MiningConfig): Promise<{ results: Array<{ gpuId: number; success: boolean; error?: string }> }>;
  stopMining(gpuIds?: number[]): Promise<{ results: Array<{ gpuId: number; success: boolean }> }>;
  runBenchmark(gpuId: number, duration: number): Promise<GPUBenchmarkResult>;
  accelerateLightningOperation(operation: string, data: unknown): Promise<unknown>;
}

export interface ILightningNetwork extends IZionModule {
  getStats(): LightningStats;
  createInvoice(amount: number, description?: string, expiry?: number): LightningInvoice;
  payInvoice(bolt11: string, amount?: number): Promise<LightningPayment>;
  openChannel(nodeId: string, amount: number, pushAmount?: number): Promise<LightningChannel>;
  closeChannel(channelId: string, force?: boolean): Promise<{ channelId: string; status: string; closingTxid: string }>;
  findRoute(destination: string, amount: number, maxFee?: number): Promise<LightningRoute>;
}

export interface IWalletService extends IZionModule {
  getBalance(): WalletBalance;
  sendTransaction(address: string, amount: number): Promise<Transaction>;
}

export interface IP2PNetwork extends IZionModule {
  getPeerCount(): number;
  connectPeer(address: string): Promise<boolean>;
  disconnectPeer(peerId: string): Promise<boolean>;
}

export interface IRPCAdapter extends IZionModule {
  handleRequest(method: string, params: unknown): Promise<unknown>;
}

// Error Types
export class ZionError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly module: string
  ) {
    super(message);
    this.name = 'ZionError';
  }
}

export class MiningError extends ZionError {
  constructor(message: string, code: string = 'MINING_ERROR') {
    super(message, code, 'mining');
  }
}

export class LightningError extends ZionError {
  constructor(message: string, code: string = 'LIGHTNING_ERROR') {
    super(message, code, 'lightning');
  }
}

export class GPUError extends ZionError {
  constructor(message: string, code: string = 'GPU_ERROR') {
    super(message, code, 'gpu');
  }
}

export class WalletError extends ZionError {
  constructor(message: string, code: string = 'WALLET_ERROR') {
    super(message, code, 'wallet');
  }
}

// Utility Types
export type RequiredNonNull<T> = {
  [P in keyof T]-?: NonNullable<T[P]>;
};

export type PartialBy<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

export type DeepReadonly<T> = {
  readonly [P in keyof T]: T[P] extends object ? DeepReadonly<T[P]> : T[P];
};

// Constants
export const ZION_CONSTANTS = {
  VERSION: '2.5.0',
  MAX_SUPPLY: 144_000_000_000,
  BLOCK_TIME: 120, // 2 minutes
  INITIAL_REWARD: 333_000_000, // 333 ZION in atomic units
  HALVENING_BLOCKS: 210_000,
  DEFAULT_PORTS: {
    P2P: 18080,
    RPC: 18081,
    POOL: 3333,
    CORE: 8888,
    LIGHTNING: 9735
  }
} as const;

export default {
  ZionError,
  MiningError,
  LightningError,
  GPUError,
  WalletError,
  ZION_CONSTANTS
};