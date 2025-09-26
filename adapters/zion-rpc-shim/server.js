// Zion RPC shim: exposes a Monero-like JSON-RPC for pool compatibility and calls ziond RPC underneath.

const express = require('express');
const axios = require('axios');
require('dotenv').config();

const app = express();
// Custom tolerant JSON parser to handle both single and batch requests
app.use((req, res, next) => {
  if (req.method !== 'POST') return next();
  let data = '';
  req.setEncoding('utf8');
  req.on('data', (chunk) => { data += chunk; });
  req.on('end', () => {
    try {
      req.body = JSON.parse(data);
    } catch (e) {
      console.error('shim JSON parse error:', e.message, 'body snippet=', (data || '').slice(0, 100));
      req.body = undefined;
    }
    next();
  });
});

const URL_ENV = process.env.ZION_RPC_URLS || process.env.ZION_RPC_URL || 'http://seed1:18081/json_rpc,http://seed2:18081/json_rpc';
const ZION_RPC_URLS = URL_ENV.split(',').map(s => s.trim()).filter(Boolean);
let CUR_URL_IDX = 0;
function currentRpcUrl() { return ZION_RPC_URLS[CUR_URL_IDX % ZION_RPC_URLS.length]; }
function nextRpcUrl() { CUR_URL_IDX++; return currentRpcUrl(); }
const PORT = parseInt(process.env.SHIM_PORT || '18089', 10);
let GBT_CACHE_MS = parseInt(process.env.GBT_CACHE_MS || '8000', 10); // serve cached template up to 8s by default
const BUSY_CACHE_FACTOR = parseInt(process.env.BUSY_CACHE_FACTOR || '5', 10); // multiply cache window during busy state
const MAX_BACKOFF_MS = parseInt(process.env.MAX_BACKOFF_MS || '5000', 10);
// Optional: disable cache after certain chain height for fresher templates during later testing
const GBT_DISABLE_CACHE_AFTER_HEIGHT = parseInt(process.env.GBT_DISABLE_CACHE_AFTER_HEIGHT || '-1', 10); // -1 => never
// Minimum sanity clamp so accidental very small values do not hammer daemon
if (GBT_CACHE_MS < 500) {
  console.warn(`[shim] GBT_CACHE_MS=${GBT_CACHE_MS}ms is very low; clamping to 500ms to protect daemon`);
  GBT_CACHE_MS = 500;
}
// Optional background prefetcher so we have a warm template even if first on-demand call hits busy
const PREFETCH_WALLET = (process.env.PREFETCH_WALLET || '').trim();
const PREFETCH_RESERVE = parseInt(process.env.PREFETCH_RESERVE || '16', 10);
const PREFETCH_INTERVAL_MS = parseInt(process.env.PREFETCH_INTERVAL_MS || '5000', 10);
// Submit tuning knobs
const SUBMIT_INITIAL_DELAY_MS = parseInt(process.env.SUBMIT_INITIAL_DELAY_MS || '1000', 10); // extra wait before first submits at very low heights
const SUBMIT_MAX_BACKOFF_MS = parseInt(process.env.SUBMIT_MAX_BACKOFF_MS || '15000', 10); // allow longer backoff on busy

// Basic in-memory metrics
const metrics = {
  startedAt: Date.now(),
  gbt: { requests: 0, cacheHits: 0, busyRetries: 0, errors: 0 },
  submit: { requests: 0, ok: 0, error: 0, busyRetries: 0 },
  lastHeight: null,
  lastSubmit: { when: null, ok: null }
};

// Cache of last successful block template (per wallet)
let lastTpl = null; // { mapped, raw, height, when, wal }
let lastErr = null; // { code, message, when }

// Low-level call to ziond; tries both JSON-RPC and REST with retry
async function zionRpc(method, params = {}, retryCount = 0) {
  const maxRetries = ZION_RPC_URLS.length;
  
  // First try JSON-RPC
  const payload = { jsonrpc: '2.0', id: '0', method, params };
  try {
    const { data } = await axios.post(currentRpcUrl(), payload, { timeout: 15000 });
    if (data && data.error) {
      // If method not found in JSON-RPC, try REST fallback
      if (data.error.code === -32601) {
        return await zionRestFallback(method, params);
      }
      const err = new Error(data.error.message || 'ziond error');
      if (typeof data.error.code !== 'undefined') err.code = data.error.code;
      err.data = data.error.data;
      throw err;
    }
    return data && data.result;
  } catch (e) {
    // Try next URL if available and this wasn't a method error
    if (retryCount < maxRetries - 1 && (!e.response || e.response.status >= 500 || e.code === 'ECONNREFUSED')) {
      console.warn(`[shim] Trying next URL (${retryCount + 1}/${maxRetries}) due to:`, e.message);
      nextRpcUrl();
      await sleep(100);
      return await zionRpc(method, params, retryCount + 1);
    }
    
    // Try REST fallback if JSON-RPC fails
    if (e.response && e.response.data && e.response.data.error && e.response.data.error.code === -32601) {
      return await zionRestFallback(method, params);
    }
    // Preserve axios response error codes if present
    if (e.response && e.response.data && e.response.data.error) {
      const de = e.response.data.error;
      const err = new Error(de.message || e.message);
      if (typeof de.code !== 'undefined') err.code = de.code;
      err.data = de.data;
      throw err;
    }
    throw e;
  }
}

// REST API fallback for methods not available in JSON-RPC
async function zionRestFallback(method, params) {
  const baseUrl = currentRpcUrl().replace('/json_rpc', '');
  
  switch (method) {
    case 'getheight':
    case 'get_height':
      const { data } = await axios.get(`${baseUrl}/getheight`, { timeout: 10000 });
      return data;
    
    case 'getblocktemplate':
      // Try REST getblocktemplate
      const wal = params.wallet_address || params.address;
      const reserve = params.reserve_size || 8;
      try {
        const { data } = await axios.get(`${baseUrl}/getblocktemplate?wallet_address=${wal}&reserve_size=${reserve}`, { timeout: 15000 });
        return data;
      } catch (e) {
        // If REST fails, throw busy error to trigger retry logic
        const err = new Error('Core is busy');
        err.code = -9;
        throw err;
      }
    
    case 'getblockheaderbyheight': {
      // REST fallback for block header by height
      let height;
      if (Array.isArray(params)) height = params[0];
      else height = params && (params.height || params.index);
      if (typeof height === 'undefined') throw new Error('height required');
      const { data } = await axios.get(`${baseUrl}/getblockheaderbyheight?height=${height}`, { timeout: 10000 });
      return data;
    }
    case 'getblockheaderbyhash': {
      // REST fallback for block header by hash
      let hash;
      if (Array.isArray(params)) hash = params[0];
      else hash = params && (params.hash || params.id || params.block_hash);
      if (!hash) throw new Error('hash required');
      const { data } = await axios.get(`${baseUrl}/getblockheaderbyhash?hash=${hash}`, { timeout: 10000 });
      return data;
    }
    
    case 'submitblock':
    case 'submit_block':
    case 'submit_raw_block':
    case 'submitrawblock':
    case 'submit_rawblock': {
      // Try multiple REST variants commonly seen in CryptoNote forks
      const blob = Array.isArray(params) ? params[0] : (params && (params.blob || params.block));
      if (!blob) throw new Error('block blob required');
      // Try GET style first
      try {
        const { data } = await axios.get(`${baseUrl}/submitblock`, { params: { blob }, timeout: 15000 });
        return data;
      } catch (_) {}
      // Try POST with {block}
      try {
        const { data } = await axios.post(`${baseUrl}/submitblock`, { block: blob }, { timeout: 15000 });
        return data;
      } catch (_) {}
      // Try POST with {blob}
      try {
        const { data } = await axios.post(`${baseUrl}/submitblock`, { blob }, { timeout: 15000 });
        return data;
      } catch (_) {}
      // Some forks expose /submit_raw_block
      try {
        const { data } = await axios.post(`${baseUrl}/submit_raw_block`, { block: blob }, { timeout: 15000 });
        return data;
      } catch (_) {}
      const err = new Error('Core is busy');
      err.code = -9;
      throw err;
    }
    default:
      throw new Error(`Method ${method} not available in REST fallback`);
  }
}

// Helper: try multiple method/param variants, return first success; else throw last error
async function tryVariants(variants) {
  const errors = [];
  for (const v of variants) {
    try {
      console.log('[shim->ziond]', v.method, JSON.stringify(v.params));
      const r = await zionRpc(v.method, v.params);
      return r;
    } catch (e) {
      const code = typeof e.code !== 'undefined' ? e.code : 'n/a';
      console.warn('[shim ziond error]', v.method, 'code=', code, 'msg=', e.message);
      errors.push(e);
      // Continue to next variant on method/param related errors
    }
  }
  // Prefer reporting a busy (-9) error if encountered, else the first error
  const busy = errors.find(er => typeof er.code !== 'undefined' && Number(er.code) === -9);
  if (busy) throw busy;
  if (errors.length) throw errors[0];
  throw new Error('all variants failed');
}

// Sleep helper
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// Simple mutex to serialize getblocktemplate calls
let gbtInFlight = false;
async function withGbtMutex(fn) {
  // Wait until not in flight
  while (gbtInFlight) {
    await sleep(50);
  }
  gbtInFlight = true;
  try {
    return await fn();
  } finally {
    gbtInFlight = false;
  }
}

// Simple mutex to serialize submitblock calls (avoid concurrent hammering while core is busy)
let submitInFlight = false;
async function withSubmitMutex(fn) {
  while (submitInFlight) {
    await sleep(50);
  }
  submitInFlight = true;
  try {
    return await fn();
  } finally {
    submitInFlight = false;
  }
}
// Minimal spacing between block submissions to avoid hammering the daemon
let lastSubmitAt = 0;

// Robust getblocktemplate with retry/backoff on transient busy errors
async function getBlockTemplateRobust(wal, reserve) {
  return withGbtMutex(async () => {
    metrics.gbt.requests++;
    // Respect requested reserve size but enforce a sane minimum of 16 bytes for extra nonce
    if (typeof reserve === 'undefined' || reserve === null) reserve = 16;
    if (reserve < 16) reserve = 16;

    // If we have a recent cached template for the same wallet, serve it immediately
    if (lastTpl && lastTpl.wal === wal) {
      if (GBT_DISABLE_CACHE_AFTER_HEIGHT >= 0 && metrics.lastHeight !== null && metrics.lastHeight >= GBT_DISABLE_CACHE_AFTER_HEIGHT) {
        // Ignore cache past configured height
      } else {
      const age = Date.now() - lastTpl.when;
      if (age <= GBT_CACHE_MS) {
        console.log(`[shim] serving cached blocktemplate age=${age}ms height=${lastTpl.height}`);
        metrics.gbt.cacheHits++;
        return lastTpl.mapped; // already mapped to Monero-like shape
      }
      }
    }

    const variants = [
      // Modern and legacy variants
      { method: 'getblocktemplate', params: { wallet_address: wal, reserve_size: reserve } },
      { method: 'getblocktemplate', params: { address: wal, reserve_size: reserve } },
      { method: 'getblocktemplate', params: [wal, reserve] },
      // Legacy underscore variant used by some CryptoNote daemons
      { method: 'get_block_template', params: { wallet_address: wal, reserve_size: reserve } },
      { method: 'get_block_template', params: { address: wal, reserve_size: reserve } },
      { method: 'get_block_template', params: [wal, reserve] }
    ];
    let lastErrLocal;
    for (let attempt = 0; attempt < 10; attempt++) {
      try {
        const r = await tryVariants(variants);
        // Debug: summarize daemon response
        try {
          const keys = Object.keys(r || {});
          const tplLen = ((r && (r.blocktemplate_blob || r.block_template_blob || r.blob || r.blockblob || r.template || r.block)) || '').length;
          const hashLen = ((r && (r.blockhashing_blob || r.hashing_blob)) || '').length;
          console.log(`[shim] gbt raw keys=${JSON.stringify(keys)} height=${r && r.height} diff=${r && r.difficulty} tplLen=${tplLen} hashBlobLen=${hashLen} reserved_offset=${r && (r.reserved_offset || r.reservedOffset)}`);
        } catch (e) {}

        // Map fields (template_blob, difficulty, height, seed_hash ... adjust as ziond provides)
        const blockTpl = r.blocktemplate_blob || r.block_template_blob || r.blob || r.blockblob || r.template || r.block || '';
        const mapped = {
          blocktemplate_blob: blockTpl,
          difficulty: r.difficulty,
          height: r.height,
          prev_hash: r.prev_hash || r.previous || r.prev || '',
          seed_hash: r.seed_hash || r.seed || '',
          reserved_offset: (typeof r.reserved_offset !== 'undefined' ? r.reserved_offset : (typeof r.reservedOffset !== 'undefined' ? r.reservedOffset : 0))
        };
        // Update cache
        lastTpl = { mapped, raw: r, height: r.height, when: Date.now(), wal };
        metrics.lastHeight = r.height;
        lastErr = null;
        return mapped;
      } catch (e) {
        lastErrLocal = e;
        lastErr = { code: typeof e.code !== 'undefined' ? Number(e.code) : undefined, message: e.message, when: Date.now() };
        if (typeof e.code !== 'undefined' && Number(e.code) === -9) {
          // Core is busy: back off and retry
          const delay = Math.min(MAX_BACKOFF_MS, 250 + attempt * 500);
          console.warn(`[shim] getblocktemplate busy (-9), retrying in ${delay}ms (attempt ${attempt+1}/10)`);
          metrics.gbt.busyRetries++;
          // rotate backend to spread load if multiple URLs configured
          if (ZION_RPC_URLS.length > 1) {
            CUR_URL_IDX = (CUR_URL_IDX + 1) % ZION_RPC_URLS.length;
            console.warn('[shim] rotating RPC backend to', currentRpcUrl());
          }
          // If we have a cached template for same wallet, serve it instead of hammering the daemon further
          if (lastTpl && lastTpl.wal === wal) {
            const age = Date.now() - lastTpl.when;
            if (age <= GBT_CACHE_MS * BUSY_CACHE_FACTOR) { // allow older cache while core is busy
              console.warn(`[shim] serving cached blocktemplate during busy state, age=${age}ms height=${lastTpl.height}`);
              return lastTpl.mapped;
            }
          }
          await sleep(delay);
          continue;
        }
        // Non-busy error: stop immediately
        metrics.gbt.errors++;
        break;
      }
    }
    throw lastErrLocal || new Error('getblocktemplate failed');
  });
}

async function handleSingle(id, method, params) {
  const m = (method || '').toString().toLowerCase();
  const mu = m.replace(/_/g, '');
  switch (mu) {
      // Height aliases
  case 'getheight': {
    const r = await tryVariants([
      { method: 'getheight', params: {} },
      { method: 'get_height', params: {} }
    ]);
        // Map to monero-like shape
        return { jsonrpc: '2.0', id, result: { height: r.height, count: r.height } };
      }
      case 'getblockheaderbyhash': {
        // Support both object and array params
        let hash;
        if (Array.isArray(params)) hash = params[0];
        else hash = params && (params.hash || params.id || params.block_hash);
        const r = await tryVariants([
          { method: 'getblockheaderbyhash', params: { hash } },
          { method: 'getblockheaderbyhash', params: [hash] },
          { method: 'get_block_header_by_hash', params: { hash } },
          { method: 'get_block_header_by_hash', params: [hash] }
        ]);
        return { jsonrpc: '2.0', id, result: r };
      }
      case 'getblockheaderbyheight': {
        // Support both object and array params
        let height;
        if (Array.isArray(params)) height = params[0];
        else height = params && params.height;
        const r = await tryVariants([
          { method: 'getblockheaderbyheight', params: { height } },
          { method: 'getblockheaderbyheight', params: [height] },
          { method: 'get_block_header_by_height', params: { height } },
          { method: 'get_block_header_by_height', params: [height] }
        ]);
        return { jsonrpc: '2.0', id, result: r };
      }
      case 'getinfo': {
        // Minimal getinfo emulation for pools that ping daemon state
        const h = await tryVariants([
          { method: 'getheight', params: {} },
          { method: 'get_height', params: {} }
        ]);
        return { jsonrpc: '2.0', id, result: { height: h.height, status: 'OK' } };
      }
      // Block template aliases
  case 'getblocktemplate': {
        // Pool may send object or array params. Accept both:
        // - Object: { wallet_address, reserve_size } or { address, reserve_size }
        // - Array: [wallet_address, reserve_size]
        let wal, reserve;
        if (Array.isArray(params)) {
          wal = params[0];
          reserve = params[1];
        } else {
          const { wallet_address, reserve_size, address, reserve_size: rs } = params || {};
          wal = address || wallet_address;
          reserve = typeof reserve_size !== 'undefined' ? reserve_size : rs;
        }
    // Sensible defaults
    if (typeof reserve === 'undefined' || reserve === null) reserve = 16;
    if (reserve < 16) reserve = 16;
        const result = await getBlockTemplateRobust(wal, reserve);
        return { jsonrpc: '2.0', id, result };
      }
      // Submit block aliases
  case 'submitblock': {
        const blob = Array.isArray(params) ? params[0] : (params && (params.blob || params.block)) || undefined;
        if (!blob) {
          return { jsonrpc: '2.0', id, error: { code: -32602, message: 'Invalid params: missing block blob' } };
        }
        return await withSubmitMutex(async () => {
          metrics.submit.requests++;
          let attemptLogCtx = { height: metrics.lastHeight, blobLen: blob.length, head: blob.slice(0,16) };
          try { console.log(`[shim] submitblock start ${JSON.stringify(attemptLogCtx)}`); } catch(_) {}
          // If we are at the bootstrap height (e.g., 0/1), give daemon a moment after GBT before first submit
          if (metrics.lastHeight !== null && Number(metrics.lastHeight) <= 1 && SUBMIT_INITIAL_DELAY_MS > 0) {
            console.warn(`[shim] initial submit delay ${SUBMIT_INITIAL_DELAY_MS}ms at height=${metrics.lastHeight}`);
            await sleep(SUBMIT_INITIAL_DELAY_MS);
          }
          // Respect a minimal cooldown between consecutive submits
          const sinceLast = Date.now() - (lastSubmitAt || 0);
          if (sinceLast < 250) {
            const wait = 250 - sinceLast;
            console.warn(`[shim] submit cooldown ${wait}ms to avoid rapid re-submits`);
            await sleep(wait);
          }
          // Retry submit on transient busy (-9)
          const variants = [
            { method: 'submitblock', params: [blob] },
            { method: 'submit_block', params: [blob] },
            { method: 'submit_raw_block', params: [blob] },
            { method: 'submitblock', params: { block: blob } },
            { method: 'submit_block', params: { block: blob } },
            { method: 'submit_raw_block', params: { block: blob } }
          ];
          let lastErr;
          for (let attempt = 0; attempt < 12; attempt++) {
            try {
              const r = await tryVariants(variants);
              metrics.submit.ok++;
              metrics.lastSubmit = { when: Date.now(), ok: true };
              lastSubmitAt = Date.now();
              // Invalidate cached template on success to force fresh height next call
              lastTpl = null;
              console.log(`[shim] submitblock accepted {"attempt":${attempt+1},"height":${metrics.lastHeight},"blobLen":${blob.length}}`);
              return { jsonrpc: '2.0', id, result: r || true };
            } catch (e) {
              lastErr = e;
              const code = typeof e.code !== 'undefined' ? Number(e.code) : undefined;
              if (code === -9) {
                // longer backoff to avoid hammering daemon
                const jitter = Math.floor(Math.random() * 200);
                const delay = Math.min(SUBMIT_MAX_BACKOFF_MS, 400 + attempt * 800 + jitter);
                console.warn(`[shim] submitblock busy (-9) retryDelay=${delay}ms attempt=${attempt+1}/12 height=${metrics.lastHeight}`);
                metrics.submit.busyRetries++;
                // Rotate backend every other attempt to reduce flapping
                if (ZION_RPC_URLS.length > 1 && attempt % 2 === 1) {
                  CUR_URL_IDX = (CUR_URL_IDX + 1) % ZION_RPC_URLS.length;
                  console.warn('[shim] rotating RPC backend to', currentRpcUrl());
                }
                // Also try REST fallbacks between attempts
                try {
                  await zionRestFallback('submitblock', [blob]);
                  metrics.submit.ok++;
                  metrics.lastSubmit = { when: Date.now(), ok: true };
                  lastSubmitAt = Date.now();
                  lastTpl = null;
                  console.log(`[shim] submitblock accepted via REST fallback {"attempt":${attempt+1},"height":${metrics.lastHeight},"blobLen":${blob.length}}`);
                  return { jsonrpc: '2.0', id, result: true };
                } catch (_) {}
                // Probe height to sync internal metrics and give core a breath
                try {
                  const h = await tryVariants([
                    { method: 'getheight', params: {} },
                    { method: 'get_height', params: {} }
                  ]);
                  if (h && typeof h.height !== 'undefined') metrics.lastHeight = h.height;
                } catch (_) {}
                await sleep(delay);
                continue;
              }
              console.warn(`[shim] submitblock error code=${code} msg=${e.message} attempt=${attempt+1} height=${metrics.lastHeight}`);
              break;
            }
          }
          metrics.submit.error++;
          metrics.lastSubmit = { when: Date.now(), ok: false };
          throw lastErr || new Error('submitblock failed');
        });
      }
      // Optional helpers used by some pools
  case 'getblockcount': {
        const r = await tryVariants([
          { method: 'getheight', params: {} },
          { method: 'get_height', params: {} }
        ]);
        return { jsonrpc: '2.0', id, result: { count: r.height } };
      }
      case 'getlastblockheader': {
        try {
          const r = await tryVariants([
            { method: 'getlastblockheader', params: {} },
            { method: 'get_last_block_header', params: {} }
          ]);
          return { jsonrpc: '2.0', id, result: r };
        } catch (e) {
          // Fallback: return minimal header
          const h = await tryVariants([
            { method: 'getheight', params: {} },
            { method: 'get_height', params: {} }
          ]);
          return { jsonrpc: '2.0', id, result: { block_header: { height: h.height } } };
        }
      }
      default:
        return { jsonrpc: '2.0', id, error: { code: -32601, message: 'Method not implemented in shim' } };
  }
}

// Simple health endpoint
app.get('/', async (req, res) => {
  try {
    const h = await tryVariants([
      { method: 'getheight', params: {} },
      { method: 'get_height', params: {} }
    ]).catch(() => ({ height: metrics.lastHeight }));
    res.json({ status: 'ok', height: h && h.height, uptimeSec: Math.floor((Date.now() - metrics.startedAt)/1000) });
  } catch (e) {
    res.status(500).json({ status: 'error', error: e.message || String(e) });
  }
});

// Unified metrics for monitoring / integrators
app.get('/metrics.json', async (req, res) => {
  try {
    const h = await tryVariants([
      { method: 'getheight', params: {} },
      { method: 'get_height', params: {} }
    ]).catch(() => ({ height: metrics.lastHeight }));
    res.json({
      status: 'ok',
      now: Date.now(),
      uptimeSec: Math.floor((Date.now() - metrics.startedAt)/1000),
      height: h && h.height,
      metrics,
      backends: {
        current: currentRpcUrl(),
        all: ZION_RPC_URLS
      }
    });
  } catch (e) {
    res.status(500).json({ status: 'error', error: e.message || String(e) });
  }
});

app.post('/json_rpc', async (req, res) => {
  const body = req.body;
  try {
    // Simple logging of incoming methods for debugging
    if (Array.isArray(body)) {
      console.log('shim batch size=', body.length, 'methods=', body.map(x => x && x.method));
      const results = [];
      for (const call of body) {
        const out = await handleSingle(call.id, call.method, call.params);
        results.push(out);
      }
      return res.json(results);
    } else {
      const { id, method, params } = body || {};
      console.log('shim method=', method, 'params=', JSON.stringify(params));
      const out = await handleSingle(id, method, params);
      return res.status(out.error ? (out.error.code === -32601 ? 501 : 500) : 200).json(out);
    }
  } catch (e) {
    const id = (body && body.id) || '0';
    const code = typeof e.code !== 'undefined' ? e.code : -32000;
    return res.status(500).json({ jsonrpc: '2.0', id, error: { code, message: e.message || 'shim error' } });
  }
});

// Convenience GET endpoints for manual testing (non-standard but handy with curl)
app.get('/getheight', async (_req, res) => {
  try {
    const r = await tryVariants([
      { method: 'getheight', params: {} },
      { method: 'get_height', params: {} }
    ]);
    return res.json({ height: r.height, status: 'OK' });
  } catch (e) {
    const code = typeof e.code !== 'undefined' ? e.code : -32000;
    return res.status(500).json({ error: { code, message: e.message || 'shim error' } });
  }
});

app.get('/getblocktemplate', async (req, res) => {
  try {
    const wal = req.query.wallet_address || req.query.address;
    let reserve = req.query.reserve_size;
    if (typeof reserve !== 'undefined') reserve = parseInt(reserve, 10);
    const result = await getBlockTemplateRobust(wal, reserve);
    return res.json(result);
  } catch (e) {
    const code = typeof e.code !== 'undefined' ? e.code : -32000;
    return res.status(code === -9 ? 503 : 500).json({ error: { code, message: e.message || 'shim error' } });
  }
});

// Non-standard helper to submit a block via GET for quick testing: /submit?blob=...
app.get('/submit', async (req, res) => {
  const blob = req.query.blob;
  if (!blob) return res.status(400).json({ error: { code: -32602, message: 'missing blob' } });
  try {
    // Reuse JSON-RPC handler path
    const out = await handleSingle('0', 'submitblock', [blob]);
    // If accepted, probe height so metrics gets updated quickly
    try {
      const h = await tryVariants([
        { method: 'getheight', params: {} },
        { method: 'get_height', params: {} }
      ]);
      metrics.lastHeight = h && h.height || metrics.lastHeight;
    } catch (_) {}
    return res.json(out.result === true ? { status: 'OK' } : (out.result || { status: 'OK' }));
  } catch (e) {
    const code = typeof e.code !== 'undefined' ? e.code : -32000;
    return res.status(code === -9 ? 503 : 500).json({ error: { code, message: e.message || 'shim error' } });
  }
});

// Simple healthcheck & info
app.get('/', (_req, res) => {
  const age = lastTpl ? (Date.now() - lastTpl.when) : null;
  res.json({
    status: 'ok',
    proxy: currentRpcUrl(),
    backends: ZION_RPC_URLS,
    lastGbt: lastTpl ? { height: lastTpl.height, ageMs: age, when: lastTpl.when } : null,
    lastError: lastErr || null,
    gbtMutexHeld: gbtInFlight,
    cacheMs: GBT_CACHE_MS
  });
});

// JSON metrics endpoint
app.get('/metrics.json', (_req, res) => {
  const age = lastTpl ? (Date.now() - lastTpl.when) : null;
  res.json({
    status: 'ok',
    metrics: {
      uptimeSec: Math.floor((Date.now() - metrics.startedAt) / 1000),
      gbt: metrics.gbt,
      submit: metrics.submit,
      lastHeight: metrics.lastHeight,
      lastGbtAgeMs: age
    },
    lastError: lastErr || null
  });
});

// Prometheus-style plaintext metrics
app.get('/metrics', (_req, res) => {
  const lines = [];
  const uptimeSec = Math.floor((Date.now() - metrics.startedAt) / 1000);
  const lastGbtAge = lastTpl ? (Date.now() - lastTpl.when) : -1;
  lines.push(`# HELP zion_shim_uptime_seconds Shim process uptime in seconds`);
  lines.push(`# TYPE zion_shim_uptime_seconds gauge`);
  lines.push(`zion_shim_uptime_seconds ${uptimeSec}`);
  lines.push(`# HELP zion_shim_gbt_requests_total Number of getblocktemplate requests`);
  lines.push(`# TYPE zion_shim_gbt_requests_total counter`);
  lines.push(`zion_shim_gbt_requests_total ${metrics.gbt.requests}`);
  lines.push(`# HELP zion_shim_gbt_cache_hits_total Number of cache hits for getblocktemplate`);
  lines.push(`# TYPE zion_shim_gbt_cache_hits_total counter`);
  lines.push(`zion_shim_gbt_cache_hits_total ${metrics.gbt.cacheHits}`);
  lines.push(`# HELP zion_shim_gbt_busy_retries_total Number of busy (-9) retries for getblocktemplate`);
  lines.push(`# TYPE zion_shim_gbt_busy_retries_total counter`);
  lines.push(`zion_shim_gbt_busy_retries_total ${metrics.gbt.busyRetries}`);
  lines.push(`# HELP zion_shim_gbt_errors_total Number of getblocktemplate errors`);
  lines.push(`# TYPE zion_shim_gbt_errors_total counter`);
  lines.push(`zion_shim_gbt_errors_total ${metrics.gbt.errors}`);
  lines.push(`# HELP zion_shim_submit_requests_total Number of submitblock requests`);
  lines.push(`# TYPE zion_shim_submit_requests_total counter`);
  lines.push(`zion_shim_submit_requests_total ${metrics.submit.requests}`);
  lines.push(`# HELP zion_shim_submit_ok_total Number of successful submitblock calls`);
  lines.push(`# TYPE zion_shim_submit_ok_total counter`);
  lines.push(`zion_shim_submit_ok_total ${metrics.submit.ok}`);
  lines.push(`# HELP zion_shim_submit_error_total Number of failed submitblock calls`);
  lines.push(`# TYPE zion_shim_submit_error_total counter`);
  lines.push(`zion_shim_submit_error_total ${metrics.submit.error}`);
  lines.push(`# HELP zion_shim_submit_busy_retries_total Number of busy (-9) retries during submitblock`);
  lines.push(`# TYPE zion_shim_submit_busy_retries_total counter`);
  lines.push(`zion_shim_submit_busy_retries_total ${metrics.submit.busyRetries}`);
  lines.push(`# HELP zion_shim_last_height Last known height from daemon`);
  lines.push(`# TYPE zion_shim_last_height gauge`);
  lines.push(`zion_shim_last_height ${metrics.lastHeight === null ? -1 : metrics.lastHeight}`);
  lines.push(`# HELP zion_shim_last_gbt_age_milliseconds Age of last block template in milliseconds`);
  lines.push(`# TYPE zion_shim_last_gbt_age_milliseconds gauge`);
  lines.push(`zion_shim_last_gbt_age_milliseconds ${lastGbtAge}`);
  res.set('Content-Type', 'text/plain; version=0.0.4');
  res.send(lines.join('\n'));
});

app.listen(PORT, () => {
  console.log(`zion-rpc-shim listening on 0.0.0.0:${PORT}, proxying to ${currentRpcUrl()}`);
});

// Start background prefetch loop to keep a warm template available
(async function startPrefetchLoop(){
  if (!PREFETCH_WALLET) return;
  // Slight initial delay to allow daemon to come up
  await sleep(500);
  console.log(`[shim] Prefetch loop enabled for wallet ${PREFETCH_WALLET.slice(0,8)}… every ${PREFETCH_INTERVAL_MS}ms`);
  while (true) {
    try {
      await getBlockTemplateRobust(PREFETCH_WALLET, PREFETCH_RESERVE);
    } catch (e) {
      const code = typeof e.code !== 'undefined' ? e.code : 'n/a';
      console.warn(`[shim] prefetch error code=${code} msg=${e.message}`);
    }
    await sleep(PREFETCH_INTERVAL_MS);
  }
})();
