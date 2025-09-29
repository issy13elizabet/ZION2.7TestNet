import axios, { AxiosInstance } from 'axios';
import { ZionError } from '../types.js';
// eslint-disable-next-line @typescript-eslint/no-explicit-any
declare const process: any; // fallback if @types/node absent

/**
 * DaemonBridge
 * 
 * Lightweight adapter mezi TypeScript integrací a legacy C++ CryptoNote daemonem.
 * Poskytuje: getInfo, getBlockTemplate, submitBlock, getBlock.
 * Implementuje jednoduché cache + retry aby nezatěžoval RPC.
 */
export class DaemonBridge {
  private rpc: AxiosInstance;
  private enabled: boolean;
  private strictRequired: boolean;
  private cache: {
    info?: { ts: number; data: any };
    template?: { ts: number; data: any };
    connections?: { ts: number; data: any };
    txPool?: { ts: number; data: any };
  } = {};
  private readonly infoTTL = 2000; // ms
  private readonly templateTTL = 5000; // ms
  private readonly connectionsTTL = 5000; // ms
  private readonly txPoolTTL = 3000; // ms
  private walletAddress: string;
  private reserveSize: number;

  constructor(opts?: {
    url?: string;
    enabled?: boolean;
    walletAddress?: string;
    reserveSize?: number;
    timeoutMs?: number;
  }) {
    const url = opts?.url || process.env.DAEMON_RPC_URL || 'http://127.0.0.1:18081';
    this.enabled = opts?.enabled ?? (process.env.EXTERNAL_DAEMON_ENABLED === 'true');
  this.strictRequired = process.env.STRICT_BRIDGE_REQUIRED === 'true';
    this.walletAddress = opts?.walletAddress || process.env.TEMPLATE_WALLET || 'Z3_PLACEHOLDER_POOL_ADDRESS';
    this.reserveSize = opts?.reserveSize ? Number(opts.reserveSize) : 8;
    const timeout = opts?.timeoutMs || Number(process.env.BRIDGE_TIMEOUT_MS || 4000);

    this.rpc = axios.create({ baseURL: url, timeout });
  }

  public isEnabled(): boolean { return this.enabled; }

  public async isAvailable(): Promise<boolean> {
    if (!this.enabled) return false;
    try {
      const info = await this.getInfo(true);
      return !!info && info.status === 'OK';
    } catch {
      return false;
    }
  }

  public requireAvailable = async (): Promise<void> => {
    if (!this.enabled) {
      if (this.strictRequired) {
        throw new ZionError('STRICT mode: external daemon bridge disabled', 'BRIDGE_REQUIRED_DISABLED', 'bridge');
      }
      throw new ZionError('Daemon bridge disabled', 'BRIDGE_DISABLED', 'bridge');
    }
    const ok = await this.isAvailable();
    if (!ok) {
      if (this.strictRequired) {
        throw new ZionError('STRICT mode: external daemon unavailable', 'BRIDGE_REQUIRED_UNAVAILABLE', 'bridge');
      }
      throw new ZionError('External daemon unavailable', 'BRIDGE_UNAVAILABLE', 'bridge');
    }
  };

  private async jsonRpc(method: string, params: any = {}, retries = 1): Promise<any> {
    try {
      const res = await this.rpc.post('/json_rpc', { jsonrpc: '2.0', id: 1, method, params });
      if (res.data?.error) throw new ZionError(res.data.error.message || 'Daemon RPC error', 'DAEMON_RPC_ERROR', 'bridge');
      return res.data?.result;
    } catch (e) {
      if (retries > 0) return this.jsonRpc(method, params, retries - 1);
      throw e;
    }
  }

  public async getInfo(force = false): Promise<any> {
    if (!this.enabled) throw new ZionError('Daemon bridge disabled', 'BRIDGE_DISABLED', 'bridge');
    const now = Date.now();
    if (!force && this.cache.info && now - this.cache.info.ts < this.infoTTL) {
      return this.cache.info.data;
    }
    const data = await this.jsonRpc('get_info', {});
    this.cache.info = { ts: now, data };
    return data;
  }

  public async getBlockTemplate(force = false): Promise<any> {
    if (!this.enabled) throw new ZionError('Daemon bridge disabled', 'BRIDGE_DISABLED', 'bridge');
    const now = Date.now();
    if (!force && this.cache.template && now - this.cache.template.ts < this.templateTTL) {
      return this.cache.template.data;
    }
    const data = await this.jsonRpc('get_block_template', { wallet_address: this.walletAddress, reserve_size: this.reserveSize });
    this.cache.template = { ts: now, data };
    return data;
  }

  public async submitBlock(blockBlob: string): Promise<any> {
    if (!this.enabled) throw new ZionError('Daemon bridge disabled', 'BRIDGE_DISABLED', 'bridge');
    return this.jsonRpc('submit_block', [blockBlob]);
  }

  public async getBlock(height: number): Promise<any> {
    if (!this.enabled) throw new ZionError('Daemon bridge disabled', 'BRIDGE_DISABLED', 'bridge');
    return this.jsonRpc('get_block', { height });
  }

  public async sendRawTransaction(hex: string): Promise<any> {
    if (!this.enabled) throw new ZionError('Daemon bridge disabled', 'BRIDGE_DISABLED', 'bridge');
    return this.jsonRpc('send_raw_transaction', { tx_as_hex: hex });
  }

  public async getConnections(force = false): Promise<any> {
    if (!this.enabled) throw new ZionError('Daemon bridge disabled', 'BRIDGE_DISABLED', 'bridge');
    const now = Date.now();
    if (!force && this.cache.connections && now - this.cache.connections.ts < this.connectionsTTL) {
      return this.cache.connections.data;
    }
    const data = await this.jsonRpc('get_connections');
    this.cache.connections = { ts: now, data };
    return data;
  }

  public async getTxPool(force = false): Promise<any> {
    if (!this.enabled) throw new ZionError('Daemon bridge disabled', 'BRIDGE_DISABLED', 'bridge');
    const now = Date.now();
    if (!force && this.cache.txPool && now - this.cache.txPool.ts < this.txPoolTTL) {
      return this.cache.txPool.data;
    }
    // Some CryptoNote daemons use 'get_tx_pool' or 'get_transaction_pool'; try sequentially
    try {
      const data = await this.jsonRpc('get_tx_pool');
      this.cache.txPool = { ts: now, data };
      return data;
    } catch (e) {
      try {
        const data2 = await this.jsonRpc('get_transaction_pool');
        this.cache.txPool = { ts: now, data: data2 };
        return data2;
      } catch (e2) {
        throw e2;
      }
    }
  }

  public async getHeight(): Promise<number> {
    const info = await this.getInfo();
    return info?.height || 0;
  }

  public async getBlockByHash(hash: string): Promise<any> {
    if (!this.enabled) throw new ZionError('Daemon bridge disabled', 'BRIDGE_DISABLED', 'bridge');
    return this.jsonRpc('get_block', { hash });
  }

  public async getLastBlockHeader(): Promise<any> {
    if (!this.enabled) throw new ZionError('Daemon bridge disabled', 'BRIDGE_DISABLED', 'bridge');
    return this.jsonRpc('get_last_block_header');
  }

  public async getBlockHeaderByHeight(height: number): Promise<any> {
    if (!this.enabled) throw new ZionError('Daemon bridge disabled', 'BRIDGE_DISABLED', 'bridge');
    return this.jsonRpc('get_block_header_by_height', { height });
  }

  public async getBlockHeaderByHash(hash: string): Promise<any> {
    if (!this.enabled) throw new ZionError('Daemon bridge disabled', 'BRIDGE_DISABLED', 'bridge');
    return this.jsonRpc('get_block_header_by_hash', { hash });
  }

  public async getBulkPayments(paymentId: string, minBlockHeight: number): Promise<any> {
    if (!this.enabled) throw new ZionError('Daemon bridge disabled', 'BRIDGE_DISABLED', 'bridge');
    return this.jsonRpc('get_bulk_payments', { payment_ids: [paymentId], min_block_height: minBlockHeight });
  }

  // Enhanced caching and metrics
  public getCacheStats(): any {
    const now = Date.now();
    return {
      info: {
        cached: !!this.cache.info,
        age: this.cache.info ? now - this.cache.info.ts : null,
        ttl: this.infoTTL
      },
      template: {
        cached: !!this.cache.template,
        age: this.cache.template ? now - this.cache.template.ts : null,
        ttl: this.templateTTL
      },
      connections: {
        cached: !!this.cache.connections,
        age: this.cache.connections ? now - this.cache.connections.ts : null,
        ttl: this.connectionsTTL
      },
      txPool: {
        cached: !!this.cache.txPool,
        age: this.cache.txPool ? now - this.cache.txPool.ts : null,
        ttl: this.txPoolTTL
      }
    };
  }

  public clearCache(): void {
    this.cache = {};
    console.log('[bridge] Cache cleared');
  }

  public async healthCheck(): Promise<{ available: boolean; latency?: number; error?: string }> {
    const start = Date.now();
    try {
      await this.getInfo(true);
      return {
        available: true,
        latency: Date.now() - start
      };
    } catch (error) {
      return {
        available: false,
        error: (error as Error).message
      };
    }
  }
}
