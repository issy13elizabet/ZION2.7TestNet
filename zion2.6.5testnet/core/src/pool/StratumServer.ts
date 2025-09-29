import net from 'net';
import { EventEmitter } from 'events';
import { randomBytes } from 'crypto';

export interface StratumConnection {
  id: number;
  socket: net.Socket;
  authorized: boolean;
  minerAddress?: string;
  extraNonce: string;
  difficulty: number;
}

export interface StratumServerOptions {
  host?: string;
  port?: number;
  initialDifficulty?: number;
  extraNoncePrefix?: string;
  log?: (...args: any[]) => void;
}

interface JsonRpcRequest {
  id: number | string | null;
  method: string;
  params?: any[] | Record<string, any>;
}

export class StratumServer extends EventEmitter {
  private server: net.Server;
  private conns = new Map<number, StratumConnection>();
  private currentJobId: string | null = null;
  private currentTemplate: string | null = null;
  private jobInterval?: any; // NodeJS.Timeout (typy pridame az doinstalujeme @types/node)
  private nextId = 1;
  private opts: Required<StratumServerOptions>;

  constructor(options: StratumServerOptions = {}) {
    super();
    this.opts = {
      host: options.host ?? '0.0.0.0',
      port: options.port ?? 3333,
      initialDifficulty: options.initialDifficulty ?? 1000,
      extraNoncePrefix: options.extraNoncePrefix ?? 'ZX',
      log: options.log ?? ((...a) => console.log('[stratum]', ...a)),
    };
  this.server = net.createServer((sock: any) => this.handleConnection(sock));
  }

  start(): Promise<void> {
    return new Promise(res => {
      this.server.listen(this.opts.port, this.opts.host, () => {
        this.opts.log(`listening on ${this.opts.host}:${this.opts.port}`);
        // Start job rotation placeholder every 30s
        this.rotateJob();
        this.jobInterval = setInterval(() => this.rotateJob(), 30000);
        res();
      });
    });
  }

  stop(): Promise<void> {
    return new Promise(res => {
      if (this.jobInterval) clearInterval(this.jobInterval);
      this.server.close(() => res());
      for (const c of this.conns.values()) {
        c.socket.destroy();
      }
      this.conns.clear();
    });
  }

  broadcast(obj: any) {
    const line = JSON.stringify(obj) + '\n';
    for (const c of this.conns.values()) {
      c.socket.write(line);
    }
  }

  private handleConnection(socket: net.Socket) {
    const id = this.nextId++;
    const conn: StratumConnection = {
      id,
      socket,
      authorized: false,
      extraNonce: this.opts.extraNoncePrefix + id.toString(16).padStart(4, '0'),
      difficulty: this.opts.initialDifficulty,
    };
    this.conns.set(id, conn);
    this.opts.log(`connection #${id} from ${socket.remoteAddress}`);

    let buffer = '';
  socket.on('data', (data: any) => {
      buffer += data.toString('utf8');
      let idx;
      while ((idx = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 1);
        if (!line) continue;
        this.handleLine(conn, line);
      }
    });

    socket.on('close', () => {
      this.conns.delete(id);
      this.opts.log(`connection #${id} closed`);
    });

  socket.on('error', (err: any) => {
      this.opts.log(`connection #${id} error`, err.message);
    });
  }

  private send(conn: StratumConnection, obj: any) {
    conn.socket.write(JSON.stringify(obj) + '\n');
  }

  private handleLine(conn: StratumConnection, line: string) {
    let msg: JsonRpcRequest | undefined;
    try {
      msg = JSON.parse(line);
    } catch {
      this.send(conn, { id: null, error: { code: -32700, message: 'Parse error' } });
      return;
    }
    if (!msg || typeof msg.method !== 'string') {
      this.send(conn, { id: msg?.id ?? null, error: { code: -32600, message: 'Invalid Request' } });
      return;
    }
    switch (msg.method) {
      case 'mining.subscribe':
        this.handleSubscribe(conn, msg);
        break;
      case 'mining.authorize':
        this.handleAuthorize(conn, msg);
        break;
      case 'mining.submit':
        this.handleSubmit(conn, msg);
        break;
      default:
        this.send(conn, { id: msg.id ?? null, error: { code: -32601, message: 'Method not found' } });
    }
  }

  private handleSubscribe(conn: StratumConnection, msg: JsonRpcRequest) {
    // Basic response structure similar to common stratum pools
    this.send(conn, { id: msg.id ?? 1, result: [["mining.set_difficulty","1"],["mining.notify","1"]], error: null });
    // Send difficulty
    this.send(conn, { id: null, method: 'mining.set_difficulty', params: [conn.difficulty] });
    // (Placeholder) send empty notify template
    if (this.currentJobId && this.currentTemplate) {
      this.send(conn, { id: null, method: 'mining.notify', params: this.buildNotifyParams(false) });
    }
  }

  private handleAuthorize(conn: StratumConnection, msg: JsonRpcRequest) {
    const params = Array.isArray(msg.params) ? msg.params : [];
    const miner = params[0];
    if (!miner) {
      this.send(conn, { id: msg.id ?? null, result: false, error: { code: 1, message: 'Missing miner address' } });
      return;
    }
    conn.minerAddress = miner;
    conn.authorized = true;
    this.send(conn, { id: msg.id ?? null, result: true, error: null });
  }

  private handleSubmit(conn: StratumConnection, msg: JsonRpcRequest) {
    if (!conn.authorized) {
      this.send(conn, { id: msg.id ?? null, result: false, error: { code: 2, message: 'Not authorized' } });
      return;
    }
    // Placeholder acceptance (no validation yet)
    this.send(conn, { id: msg.id ?? null, result: true, error: null });
  (this as any).emit('share', { connId: conn.id, params: msg.params });
  }

  private rotateJob() {
    // Placeholder job generation: random hex template + prev hash zero
    this.currentJobId = Date.now().toString(16);
  this.currentTemplate = randomBytes(16).toString('hex');
    const params = this.buildNotifyParams(true);
    this.broadcast({ id: null, method: 'mining.notify', params });
    this.opts.log(`new job ${this.currentJobId}`);
  }

  private buildNotifyParams(clean: boolean) {
    return [
      this.currentJobId,
      '00000000',                 // prev hash (placeholder)
      this.currentTemplate,       // coinbase / block template (placeholder)
      [],                         // merkle branch
      '00000000',                 // version
      '00000000',                 // nbits/ntime placeholder
      clean
    ];
  }
}
