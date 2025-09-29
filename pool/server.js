// Minimal Stratum-like TCP server (stub) for Zion
// - Listens on 0.0.0.0:3333 by default
// - Accepts JSON-RPC lines terminated by \n
const net = require('net');
const crypto = require('crypto');
const http = require('http');
require('dotenv').config();

const HOST = process.env.POOL_BIND || '0.0.0.0';
const PORT = parseInt(process.env.POOL_PORT || '3333', 10);
const DAEMON_HOST = process.env.DAEMON_HOST || '127.0.0.1';
const DAEMON_PORT = parseInt(process.env.DAEMON_PORT || '18081', 10); // JSON-RPC port per mainnet.conf
const TEMPLATE_ADDR = process.env.TEMPLATE_ADDRESS || 'Z3BDEEC2A0AE0F5D81B034308F99ECD8990D9B8B01BD9C7E7429392CA31861C6220DA3B30D74E809FA0A1FE069F1';

// Cache for latest block template
let latestTemplate = null;
let lastTemplateTs = 0;
const TEMPLATE_TTL_MS = 20_000; // refresh every 20s

const clients = new Set();

function randHex(n) {
  return crypto.randomBytes(n).toString('hex');
}

function makeResponse(id, result) {
  return JSON.stringify({ jsonrpc: '2.0', id, result, error: null });
}

function makeJobNotification() {
  // Simplified job; fields layout inspired by stratum, but not real mining yet
  const jobId = randHex(4);
  const header = randHex(32);
  const target = 'ffff' + '0'.repeat(60); // very easy target
  return JSON.stringify({ id: null, method: 'mining.notify', params: [jobId, header, target] });
}

function daemonRpc(method, params) {
  return new Promise((resolve, reject) => {
    const payload = JSON.stringify({ jsonrpc: '2.0', id: 1, method, params });
    const req = http.request({
      host: DAEMON_HOST,
      port: DAEMON_PORT,
      method: 'POST',
      path: '/json_rpc',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(payload)
      }
    }, (res) => {
      let data = '';
      res.on('data', d => data += d);
      res.on('end', () => {
        try {
          const parsed = JSON.parse(data);
          if (parsed.error) return reject(parsed.error);
          resolve(parsed.result);
        } catch (e) {
          reject(e);
        }
      });
    });
    req.on('error', reject);
    req.write(payload);
    req.end();
  });
}

async function refreshTemplate(force = false) {
  const now = Date.now();
  if (!force && latestTemplate && (now - lastTemplateTs) < TEMPLATE_TTL_MS) return latestTemplate;
  try {
    // get_block_template expects wallet_address + reserve_size
    const result = await daemonRpc('get_block_template', { wallet_address: TEMPLATE_ADDR, reserve_size: 8 });
    latestTemplate = result;
    lastTemplateTs = now;
    return latestTemplate;
  } catch (e) {
    console.error('Template refresh failed:', e.message || e);
    return latestTemplate; // return stale if available
  }
}

function difficultyToTarget(diff) {
  // CryptoNote target calculation: target = floor((2^256 - 1) / diff)
  // We'll approximate using BigInt.
  if (!diff || diff <= 0) diff = 1;
  const TWO_256 = (1n << 256n) - 1n;
  const targetBig = TWO_256 / BigInt(diff);
  let hex = targetBig.toString(16);
  if (hex.length < 64) hex = '0'.repeat(64 - hex.length) + hex;
  return hex;
}

function makeMoneroJobFromTemplate(tpl) {
  if (!tpl) {
    // fallback synthetic
    return {
      job_id: randHex(8),
      blob: randHex(152),
      target: '00ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
      algo: 'rx/0',
      height: 0,
      seed_hash: randHex(64),
      next_seed_hash: randHex(64)
    };
  }
  // tpl fields: blocktemplate_blob, difficulty, height, prev_hash, seed_hash, next_seed_hash (names may vary)
  return {
    job_id: randHex(8),
    blob: tpl.blocktemplate_blob || randHex(152),
    target: difficultyToTarget(tpl.difficulty || 1),
    algo: 'rx/0',
    height: tpl.height || 0,
    seed_hash: tpl.seed_hash || randHex(64),
    next_seed_hash: tpl.next_seed_hash || randHex(64)
  };
}

async function makeMoneroJob() {
  const tpl = await refreshTemplate();
  return makeMoneroJobFromTemplate(tpl);
}

async function handleLine(sock, line) {
  let msg;
  try {
    msg = JSON.parse(line);
  } catch {
    return; // ignore invalid JSON
  }

  const { id, method, params } = msg;
  // Monero/XMRig style: login
  if (method === 'login') {
    const sessionId = randHex(8);
    const job = await makeMoneroJob();
    const result = { status: 'OK', id: sessionId, job, extensions: [] };
    sock.write(makeResponse(id, result) + "\n");
    return;
  }
  // Monero/XMRig style: getjob
  if (method === 'getjob') {
    const job = await makeMoneroJob();
    sock.write(makeResponse(id, job) + "\n");
    return;
  }
  // Monero/XMRig style: keepalive
  if (method === 'keepalived') {
    sock.write(makeResponse(id, { status: 'KEEPALIVED' }) + "\n");
    return;
  }
  // Monero/XMRig style: submit
  if (method === 'submit') {
    // params could have: id, job_id, nonce, result (hash), algo
    try {
      console.log('Share submit params:', params);
      // Placeholder: we would reconstruct block blob with nonce and compute hash to check target
    } catch (e) {
      console.error('Submit processing error', e);
    }
    const job = await makeMoneroJob();
    sock.write(makeResponse(id, { status: 'OK' }) + "\n");
    sock.write(JSON.stringify({ jsonrpc: '2.0', id: null, method: 'job', params: job }) + "\n");
    return;
  }

  // Legacy mining.* style
  if (method === 'mining.subscribe') {
    sock.write(makeResponse(id, ["OK", "1"]) + "\n");
    // send first job
    sock.write(makeJobNotification() + "\n");
    return;
  }
  if (method === 'mining.authorize') {
    sock.write(makeResponse(id, true) + "\n");
    return;
  }
  if (method === 'mining.submit') {
    // accept everything in stub
    sock.write(makeResponse(id, true) + "\n");
    // optionally send another job
    sock.write(makeJobNotification() + "\n");
    return;
  }
  // generic ack
  sock.write(makeResponse(id, true) + "\n");
}

const server = net.createServer((sock) => {
  clients.add(sock);
  let buffer = '';
  sock.on('data', (chunk) => {
    buffer += chunk.toString('utf8');
    let idx;
    while ((idx = buffer.indexOf('\n')) !== -1) {
      const line = buffer.slice(0, idx).trim();
      buffer = buffer.slice(idx + 1);
      if (line) handleLine(sock, line);
    }
  });
  sock.on('end', () => clients.delete(sock));
  sock.on('error', () => clients.delete(sock));
});

server.listen(PORT, HOST, () => {
  console.log(`Zion Stratum stub listening on ${HOST}:${PORT}`);
});

// Periodic template refresh + broadcast if height changed
let lastBroadcastHeight = 0;
async function templateLoop() {
  await refreshTemplate(false);
  if (latestTemplate && latestTemplate.height && latestTemplate.height !== lastBroadcastHeight) {
    lastBroadcastHeight = latestTemplate.height;
    const job = makeMoneroJobFromTemplate(latestTemplate);
    const payload = JSON.stringify({ jsonrpc: '2.0', id: null, method: 'job', params: job }) + '\n';
    for (const c of clients) {
      try { c.write(payload); } catch {}
    }
    console.log('Broadcast new job height', lastBroadcastHeight);
  }
  setTimeout(templateLoop, 5000);
}
templateLoop();
