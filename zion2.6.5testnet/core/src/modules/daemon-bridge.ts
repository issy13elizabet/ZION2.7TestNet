import axios, { AxiosInstance } from 'axios';
import { ZionError } from '../types.js';

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
  private cache: {
    info?: { ts: number; data: any };
    template?: { ts: number; data: any };
  } = {};
  private readonly infoTTL = 2000; // ms
  private readonly templateTTL = 5000; // ms
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

  public async getHeight(): Promise<number> {
    const info = await this.getInfo();
    return info?.height || 0;
  }
}
