import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import axios from 'axios';
import pino from 'pino';
import morgan from 'morgan';

const PORT = process.env.ADAPTER_PORT || 18099;
const BIND = process.env.ADAPTER_BIND || '0.0.0.0';
const WALLET_RPC = process.env.WALLET_RPC || 'http://walletd:8070/json_rpc';
const SHIM_RPC = process.env.SHIM_RPC || 'http://zion-rpc-shim:18089/json_rpc';
const CORS_ORIGINS_RAW = process.env.CORS_ORIGINS || '';
const CORS_ORIGINS = (CORS_ORIGINS_RAW ? CORS_ORIGINS_RAW.split(',') : []);
const API_KEY = process.env.ADAPTER_API_KEY || '';
const REQUIRE_API_KEY = (process.env.REQUIRE_API_KEY || '').toLowerCase() !== 'false';

const app = express();
app.use(helmet({ crossOriginEmbedderPolicy: false }));
app.use(express.json());
// CORS: v produkci povol jen whitelist, v dev fallback na localhost
app.use(cors({
  origin: (origin, cb) => {
    if (!origin) return cb(null, true);
    if (process.env.NODE_ENV !== 'production' && (CORS_ORIGINS.length === 0)) {
      const ok = /^(http:\/\/localhost(:\d+)?|http:\/\/127\.0\.0\.1(:\d+)?)$/.test(origin);
      return cb(ok ? null : new Error('CORS blocked'), ok);
    }
    if (CORS_ORIGINS.includes('*') && process.env.NODE_ENV !== 'production') return cb(null, true);
    return cb(CORS_ORIGINS.includes(origin) ? null : new Error('CORS blocked'), CORS_ORIGINS.includes(origin));
  },
  credentials: false,
}));

const logger = pino({ level: process.env.LOG_LEVEL || (process.env.NODE_ENV === 'production' ? 'info' : 'debug') });
if (process.env.NODE_ENV !== 'production') {
  app.use(morgan('tiny'));
}

// Lightweight rate limit for send endpoint
const sendLimiter = rateLimit({
  windowMs: Number(process.env.SEND_RATE_WINDOW_MS || 60_000),
  max: Number(process.env.SEND_RATE_MAX || 6),
  standardHeaders: true,
  legacyHeaders: false,
});

// In-memory metrics (Prometheus-friendly)
const metrics = {
  startedAt: Date.now(),
  requests: 0,
  routes: {
    balance: 0,
    send: 0,
    address: 0,
    history: 0,
    summary: 0,
  },
  sendOk: 0,
  sendErr: 0
};

// Tiny in-memory cache with TTL for explorer endpoints
const _cache = new Map(); // key -> { expires, value }
function cacheGet(key) {
  const hit = _cache.get(key);
  if (!hit) return undefined;
  if (Date.now() > hit.expires) { _cache.delete(key); return undefined; }
  return hit.value;
}
function cacheSet(key, value, ttlMs = 5000) {
  _cache.set(key, { expires: Date.now() + ttlMs, value });
}

// Helpers
async function walletRpc(method, params = {}) {
  const res = await axios.post(WALLET_RPC, { jsonrpc: '2.0', id: '0', method, params }, { timeout: 5000 });
  if (res.data.error) throw res.data.error;
  return res.data.result;
}

async function shimRpc(method, params = {}) {
  const res = await axios.post(SHIM_RPC, { jsonrpc: '2.0', id: '0', method, params }, { timeout: 5000 });
  if (res.data.error) throw res.data.error;
  return res.data.result || res.data;
}

async function walletTry(variants) {
  const errors = [];
  for (const v of variants) {
    try {
      const out = await walletRpc(v.method, v.params || {});
      return out;
    } catch (e) {
      errors.push(e);
    }
  }
  if (errors.length) throw errors[0];
  throw new Error('walletTry: no variants succeeded');
}

// Accept new mainnet addresses starting with "Z3" and legacy starting with "aj".
// Wallets may emit slightly different lengths; accept a safe range (92..100 total chars).
const ZION_ADDR_REGEX = /^(Z3|aj)[123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz]{90,98}$/;

// Try to normalize various wallet RPC responses into a single address string
function extractAddress(result) {
  if (!result || typeof result !== 'object') return undefined;
  // Common keys
  if (typeof result.address === 'string') return result.address;
  if (typeof result.addr === 'string') return result.addr;
  if (typeof result.mainAddress === 'string') return result.mainAddress;
  if (result.result && typeof result.result.address === 'string') return result.result.address;
  // Arrays of addresses
  if (Array.isArray(result.addresses) && result.addresses.length > 0) return result.addresses[0];
  if (Array.isArray(result.result?.addresses) && result.result.addresses.length > 0) return result.result.addresses[0];
  // Sometimes APIs wrap in {value: {address: "..."}}
  if (result.value && typeof result.value.address === 'string') return result.value.address;
  return undefined;
}

function requireApiKey(req, res) {
  if (!REQUIRE_API_KEY && process.env.NODE_ENV !== 'production') return true;
  if (!API_KEY) { res.status(401).json({ error: 'missing_api_key' }); return false; }
  const hdr = req.headers['x-api-key'];
  if (!hdr || hdr !== API_KEY) { res.status(401).json({ error: 'unauthorized' }); return false; }
  return true;
}

// Routes
app.get('/healthz', async (req, res) => {
  try {
    metrics.requests++;
    const [height, balance] = await Promise.all([
      shimRpc('getheight').catch(() => ({ height: -1 })),
      walletRpc('getBalance').catch(() => ({ availableBalance: -1, lockedAmount: -1 }))
    ]);
    res.json({ ok: true, height, balance });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message || e });
  }
});

app.get('/wallet/balance', async (req, res) => {
  try {
    metrics.requests++; metrics.routes.balance++;
    const result = await walletRpc('getBalance');
    res.json(result);
  } catch (e) {
    res.status(500).json({ error: e.message || e });
  }
});

app.get('/wallet/address', async (req, res) => {
  try {
    metrics.requests++; metrics.routes.address++;
    // Try broader set of wallet RPC method variants
    const r = await walletTry([
      { method: 'getAddress' },
      { method: 'get_address' },
      { method: 'getaddress' },
      { method: 'getAddresses' },
      { method: 'get_addresses' },
      { method: 'getMainAddress' },
      { method: 'get_main_address' }
    ]);
    const address = extractAddress(r);
    if (!address) return res.status(501).json({ error: 'address not available (unsupported RPC variant)' });
    res.json({ address, valid: ZION_ADDR_REGEX.test(address) });
  } catch (e) {
    res.status(500).json({ error: e.message || e });
  }
});

// Create a new wallet address (in current wallet container)
app.post('/wallet/create_address', async (req, res) => {
  try {
    if (!requireApiKey(req, res)) return;
    try {
      const r = await walletTry([
        { method: 'createAddress', params: {} },
        { method: 'create_address', params: {} },
        { method: 'createNewAddress', params: {} },
        { method: 'generateNewAddress', params: {} },
        { method: 'create_integrated_address', params: {} },
        { method: 'createIntegratedAddress', params: {} },
        { method: 'createAddressSimple', params: {} },
      ]);
      const address = extractAddress(r) || r?.address || r?.result?.address || r?.value?.address || r?.integratedAddress;
      if (!address) return res.status(500).json({ error: 'address not returned' });
      return res.json({ address });
    } catch (e) {
      // If method not supported, surface 501 to help UI
      const msg = (e && (e.message || e.toString())) || '';
      if (msg && /method not found|unsupported|not implemented/i.test(msg)) {
        return res.status(501).json({ error: 'create address not supported by wallet RPC' });
      }
      if (msg && /ECONNREFUSED|ECONNRESET|ENOTFOUND|ETIMEDOUT|timeout/i.test(msg)) {
        return res.status(503).json({ error: 'wallet RPC unavailable' });
      }
      throw e;
    }
  } catch (e) {
    res.status(500).json({ error: e.message || e });
  }
});

// Save wallet container to disk (flush)
app.post('/wallet/save', async (req, res) => {
  try {
    if (!requireApiKey(req, res)) return;
    const r = await walletTry([
      { method: 'save' },
      { method: 'store' },
      { method: 'saveWallet' }
    ]);
    res.json({ ok: true, result: r || true });
  } catch (e) {
    res.status(500).json({ error: e.message || e });
  }
});

// Export keys for given address (or primary)
app.get('/wallet/keys', async (req, res) => {
  try {
    if (!requireApiKey(req, res)) return;
    const qAddr = (req.query.address || '').toString().trim();
    let address = qAddr;
    if (!address) {
      try {
        const ra = await walletTry([
          { method: 'getAddress' },
          { method: 'get_address' },
          { method: 'getAddresses' },
          { method: 'get_addresses' }
        ]);
        address = extractAddress(ra);
      } catch (_) {/* ignore */}
    }
    // Gather spend/view keys via common variants
    let spend = {};
    try {
      const rs = await walletTry([
        { method: 'getSpendKeys', params: address ? { address } : {} },
        { method: 'get_spend_keys', params: address ? { address } : {} }
      ]);
      spend = {
        publicKey: rs?.spendPublicKey || rs?.publicKey || rs?.spend_key_public || rs?.spend_pub_key,
        secretKey: rs?.spendSecretKey || rs?.secretKey || rs?.spend_key_secret || rs?.spend_sec_key,
      };
    } catch (_) { /* optional */ }
    let view = {};
    try {
      const rv = await walletTry([
        { method: 'getViewKey', params: address ? { address } : {} },
        { method: 'get_view_key', params: address ? { address } : {} },
        { method: 'getViewKeys', params: address ? { address } : {} },
        { method: 'get_view_keys', params: address ? { address } : {} }
      ]);
      view = {
        publicKey: rv?.viewPublicKey || rv?.publicKey || rv?.view_key_public || rv?.view_pub_key,
        secretKey: rv?.viewSecretKey || rv?.secretKey || rv?.view_key_secret || rv?.view_sec_key,
      };
    } catch (_) { /* optional */ }
    if (!address && (!spend.secretKey && !view.secretKey)) {
      return res.status(501).json({ error: 'keys not available' });
    }
    res.json({ address, spend, view, ts: Date.now() });
  } catch (e) {
    res.status(500).json({ error: e.message || e });
  }
});

app.post('/wallet/send', sendLimiter, async (req, res) => {
  try {
    metrics.requests++; metrics.routes.send++;
    if (!requireApiKey(req, res)) return;
    const { address, amount, fee = 0, anonymity = 3, paymentId } = req.body || {};
    const parsedAmount = Number(amount);
    if (!address || !Number.isFinite(parsedAmount) || parsedAmount <= 0) {
      return res.status(400).json({ error: 'address and positive numeric amount are required' });
    }
    // Guardrails: max 1 recipient, basic upper bound to prevent abuse in dev
    if (parsedAmount > 1e16) {
      return res.status(400).json({ error: 'amount too large for adapter policy' });
    }
    if (!ZION_ADDR_REGEX.test(address)) {
      return res.status(400).json({ error: 'invalid address format' });
    }
    const params = {
  transfers: [ { address, amount: parsedAmount } ],
      fee,
      anonymity,
      unlockTime: 0,
      changeAddress: undefined,
      paymentId
    };
    const result = await walletRpc('sendTransaction', params);
    metrics.sendOk++;
    res.json(result);
  } catch (e) {
    metrics.sendErr++;
    res.status(500).json({ error: e.message || e });
  }
});

app.post('/wallet/validate', async (req, res) => {
  try {
    const { address } = req.body || {};
    if (typeof address !== 'string') return res.status(400).json({ error: 'address required' });
    const valid = ZION_ADDR_REGEX.test(address.trim());
    res.json({ valid });
  } catch (e) {
    res.status(500).json({ error: e.message || e });
  }
});

app.get('/chain/height', async (req, res) => {
  try {
    const result = await shimRpc('getheight');
    res.json(result);
  } catch (e) {
    res.status(500).json({ error: e.message || e });
  }
});

// Explorer endpoints
app.get('/explorer/summary', async (req, res) => {
  try {
    metrics.requests++; metrics.routes.summary++;
    const cached = cacheGet('explorer:summary');
    if (cached) return res.json(cached);
    const [h, last] = await Promise.all([
      shimRpc('getheight'),
      shimRpc('getlastblockheader').catch(async () => {
        // Fallback: derive last header by height
        const hh = await shimRpc('getheight');
        const tip = Number(hh.height || hh.count || 0);
        if (tip > 0) return shimRpc('getblockheaderbyheight', { height: tip });
        return { block_header: { height: 0 } };
      })
    ]);
    const out = { height: h.height || h.count || 0, last_block_header: last.block_header || last };
    cacheSet('explorer:summary', out, 3000);
    res.json(out);
  } catch (e) {
    res.status(500).json({ error: e.message || e });
  }
});

app.get('/explorer/blocks', async (req, res) => {
  try {
    // Params: start (height, default tip), limit (1..100), order (desc|asc)
    const limit = Math.max(1, Math.min(100, Number(req.query.limit || 10)));
    const order = (req.query.order || 'desc').toString().toLowerCase();
    const hh = await shimRpc('getheight');
    const tip = Number(hh.height || hh.count || 0);
    let start = req.query.start !== undefined ? Number(req.query.start) : tip;
    if (!Number.isFinite(start) || start < 0) start = tip;

    const blocks = [];
    if (order === 'desc') {
      let i = start;
      while (blocks.length < limit && i >= 0) {
        try {
          const r = await shimRpc('getblockheaderbyheight', { height: i });
          blocks.push({ height: i, header: r.block_header || r });
        } catch (err) {
          // If freshest height fails, skip it once
          if (i === start) { i--; continue; }
          blocks.push({ height: i, error: String(err?.message || err) });
        }
        i--;
      }
    } else {
      let i = start;
      const end = Math.min(tip, start + limit - 1);
      while (i <= end) {
        try {
          const r = await shimRpc('getblockheaderbyheight', { height: i });
          blocks.push({ height: i, header: r.block_header || r });
        } catch (err) {
          blocks.push({ height: i, error: String(err?.message || err) });
        }
        i++;
      }
    }
    res.json({ tip, start, order, count: blocks.length, blocks });
  } catch (e) {
    res.status(500).json({ error: e.message || e });
  }
});

app.get('/explorer/block/height/:height', async (req, res) => {
  try {
    const h = Number(req.params.height);
    if (!Number.isFinite(h) || h < 0) return res.status(400).json({ error: 'invalid height' });
    const key = `explorer:block:h:${h}`;
    const cached = cacheGet(key);
    if (cached) return res.json(cached);
    const r = await shimRpc('getblockheaderbyheight', { height: h });
    const out = { height: h, header: r.block_header || r };
    cacheSet(key, out, 5000);
    res.json(out);
  } catch (e) {
    res.status(500).json({ error: e.message || e });
  }
});

app.get('/explorer/block/hash/:hash', async (req, res) => {
  try {
    const hash = (req.params.hash || '').trim();
    if (!hash || !/^[0-9a-fA-F]{64}$/.test(hash)) return res.status(400).json({ error: 'invalid hash' });
    const key = `explorer:block:x:${hash}`;
    const cached = cacheGet(key);
    if (cached) return res.json(cached);
    const r = await shimRpc('getblockheaderbyhash', { hash });
    const out = { hash, header: r.block_header || r };
    cacheSet(key, out, 5000);
    res.json(out);
  } catch (e) {
    res.status(500).json({ error: e.message || e });
  }
});

app.get('/explorer/search', async (req, res) => {
  try {
    const q = (req.query.q || '').toString().trim();
    if (!q) return res.status(400).json({ error: 'missing q' });
    if (/^[0-9a-fA-F]{64}$/.test(q)) {
      // Try block by hash first; TODO: add tx lookup when daemon/shim supports it
      try {
        const r = await shimRpc('getblockheaderbyhash', { hash: q });
        return res.json({ type: 'block', hash: q, header: r.block_header || r });
      } catch (_) {
        return res.status(404).json({ error: 'not found', note: 'tx lookup not yet implemented' });
      }
    }
    // Maybe it's a height
    const h = Number(q);
    if (Number.isFinite(h) && h >= 0) {
      try {
        const r = await shimRpc('getblockheaderbyheight', { height: h });
        return res.json({ type: 'block', height: h, header: r.block_header || r });
      } catch (_) {}
    }
    return res.status(400).json({ error: 'unsupported query' });
  } catch (e) {
    res.status(500).json({ error: e.message || e });
  }
});

// Explorer stats: aggregates last N blocks
app.get('/explorer/stats', async (req, res) => {
  try {
    const n = Math.max(2, Math.min(200, Number(req.query.n || 120)));
    const cacheKey = `explorer:stats:${n}`;
    const cached = cacheGet(cacheKey);
    if (cached) return res.json(cached);

    const hh = await shimRpc('getheight');
    const tip = Number(hh.height || hh.count || 0);
    const start = Math.max(0, tip - n + 1);
    const ts = [];
    const headers = [];
    for (let h = start; h <= tip; h++) {
      try {
        const r = await shimRpc('getblockheaderbyheight', { height: h });
        const hdr = r.block_header || r;
        headers.push({ height: h, header: hdr });
        const t = Number(hdr.timestamp || hdr.ts || 0);
        if (Number.isFinite(t) && t > 0) ts.push({ h, t });
      } catch (_) {
        // ignore missing
      }
    }

    ts.sort((a, b) => a.h - b.h);
    const diffs = [];
    for (let i = 1; i < ts.length; i++) diffs.push(ts[i].t - ts[i-1].t);
    const avgIntervalSec = diffs.length ? Math.max(0, Math.round(diffs.reduce((a,b)=>a+b,0) / diffs.length)) : null;
    const now = Math.floor(Date.now() / 1000);
    const blocksLastHour = ts.filter(x => now - x.t <= 3600).length;
    const bphApprox = avgIntervalSec && avgIntervalSec > 0 ? Number((3600 / avgIntervalSec).toFixed(2)) : null;

    const out = {
      tip,
      window: { start, end: tip, captured: headers.length },
      avgIntervalSec,
      blocksLastHour,
      bphApprox,
    };
    cacheSet(cacheKey, out, 3000);
    res.json(out);
  } catch (e) {
    res.status(500).json({ error: e.message || e });
  }
});

app.get('/pool/blocks-recent', async (req, res) => {
  try {
    const n = Math.max(1, Math.min(50, Number(req.query.n || 10)));
    const h = await shimRpc('getheight');
    const tip = Number(h.height || h.count || 0);
    if (!Number.isFinite(tip) || tip <= 0) return res.json({ blocks: [], tip });
    const results = [];
    let i = tip;
    const seen = new Set();
    while (results.length < n && i >= 0) {
      try {
        const r = await shimRpc('getblockheaderbyheight', { height: i });
        if (!seen.has(i)) {
          results.push({ height: i, data: r });
          seen.add(i);
        }
        i--;
      } catch (err) {
        // If the freshest block is not yet queryable, skip it once
        if (i === tip) {
          i--;
          continue;
        }
        if (!seen.has(i)) {
          results.push({ height: i, error: String(err?.message || err) });
          seen.add(i);
        }
        i--;
      }
    }
    res.json({ tip, count: results.length, blocks: results });
  } catch (e) {
    res.status(500).json({ error: e.message || e });
  }
});

// Wallet history (best-effort across forks)
app.get('/wallet/history', async (req, res) => {
  try {
    metrics.requests++; metrics.routes.history++;
    const limit = Math.max(1, Math.min(200, Number(req.query.limit || 20)));
    const hh = await shimRpc('getheight').catch(() => ({ height: 0 }));
    const tip = Number(hh.height || 0);
    const firstBlockIndex = Math.max(0, tip - 1000);
    // Try common variants
    let r = null;
    try {
      r = await walletTry([
        { method: 'getTransactions', params: { firstBlockIndex, blockCount: 1000 } },
        { method: 'get_transactions', params: { firstBlockIndex, blockCount: 1000 } },
      ]);
    } catch (_) {
      // Try get_transfers style with minHeight
      try {
        r = await walletTry([
          { method: 'get_transfers', params: { in: true, out: true, pending: true, failed: false, pool: true, minHeight: firstBlockIndex } },
          { method: 'getTransfers', params: { in: true, out: true, pending: true, failed: false, pool: true, minHeight: firstBlockIndex } },
        ]);
      } catch (e2) {
        return res.status(501).json({ error: 'wallet history not available', details: String(e2?.message || e2) });
      }
    }
    // Normalize
    let txs = [];
    if (r && Array.isArray(r.transactions)) txs = r.transactions;
    else if (r && r.items && Array.isArray(r.items)) txs = r.items;
    else if (r && r.in && Array.isArray(r.in)) txs = [...r.in, ...(r.out || [])];
    // Map minimal view
    const mapped = txs.map((t) => ({
      hash: t.hash || t.transactionHash || t.id || undefined,
      ts: Number(t.timestamp || t.time || 0),
      amount: Number(t.totalAmount || t.amount || 0),
      fee: Number(t.fee || 0),
      blockIndex: Number(t.blockIndex || t.height || -1),
      paymentId: t.paymentId || t.payment_id || undefined,
      transfers: t.transfers || t.recipients || t.outputs || undefined,
      direction: (t.amount && t.amount < 0) ? 'out' : 'in'
    })).sort((a, b) => (b.ts - a.ts));
    res.json({ tip, count: mapped.length, txs: mapped.slice(0, limit) });
  } catch (e) {
    res.status(500).json({ error: e.message || e });
  }
});

// Prometheus metrics
app.get('/metrics', (_req, res) => {
  const uptimeSec = Math.floor((Date.now() - metrics.startedAt)/1000);
  const lines = [];
  lines.push(`# HELP zion_adapter_uptime_seconds Adapter uptime in seconds`);
  lines.push(`# TYPE zion_adapter_uptime_seconds gauge`);
  lines.push(`zion_adapter_uptime_seconds ${uptimeSec}`);
  lines.push(`# HELP zion_adapter_requests_total Total HTTP requests processed by adapter`);
  lines.push(`# TYPE zion_adapter_requests_total counter`);
  lines.push(`zion_adapter_requests_total ${metrics.requests}`);
  lines.push(`# HELP zion_adapter_route_requests_total Requests per logical route`);
  lines.push(`# TYPE zion_adapter_route_requests_total counter`);
  Object.entries(metrics.routes).forEach(([k,v]) => lines.push(`zion_adapter_route_requests_total{route="${k}"} ${v}`));
  lines.push(`# HELP zion_adapter_send_total Send transaction results`);
  lines.push(`# TYPE zion_adapter_send_total counter`);
  lines.push(`zion_adapter_send_total{status="ok"} ${metrics.sendOk}`);
  lines.push(`zion_adapter_send_total{status="error"} ${metrics.sendErr}`);
  res.set('Content-Type', 'text/plain; version=0.0.4');
  res.send(lines.join('\n'));
});

app.listen(PORT, BIND, () => {
  logger.info({ port: PORT, bind: BIND, walletRpc: WALLET_RPC, shimRpc: SHIM_RPC }, 'wallet-adapter listening');
  if (process.env.NODE_ENV === 'production') {
    if (!API_KEY && REQUIRE_API_KEY) logger.warn('API_KEY not set; requests will be rejected');
    if (CORS_ORIGINS.length === 0 || CORS_ORIGINS.includes('*')) logger.warn('CORS_ORIGINS is empty or wildcard in production');
  }
});
