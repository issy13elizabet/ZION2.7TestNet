// Minimal Stratum-like TCP server (stub) for Zion
// - Listens on 0.0.0.0:3333 by default
// - Accepts JSON-RPC lines terminated by \n
const net = require('net');
const crypto = require('crypto');
require('dotenv').config();

const HOST = process.env.POOL_BIND || '0.0.0.0';
const PORT = parseInt(process.env.POOL_PORT || '3333', 10);

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

function makeMoneroJob() {
  // Minimal Monero-style job object for XMRig 'login' responses
  return {
    job_id: randHex(4),
    blob: randHex(39), // arbitrary size; xmrig will not validate in stub
    target: 'ffff' + '0'.repeat(60),
    algo: 'rx/0'
  };
}

function handleLine(sock, line) {
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
    const result = { status: 'OK', id: sessionId, job: makeMoneroJob(), extensions: [] };
    sock.write(makeResponse(id, result) + "\n");
    return;
  }
  // Monero/XMRig style: getjob
  if (method === 'getjob') {
    sock.write(makeResponse(id, makeMoneroJob()) + "\n");
    return;
  }
  // Monero/XMRig style: keepalive
  if (method === 'keepalived') {
    sock.write(makeResponse(id, { status: 'KEEPALIVED' }) + "\n");
    return;
  }
  // Monero/XMRig style: submit
  if (method === 'submit') {
    // Always accept in stub
    sock.write(makeResponse(id, { status: 'OK' }) + "\n");
    // Send a new job sometimes
    sock.write(JSON.stringify({ jsonrpc: '2.0', id: null, method: 'job', params: makeMoneroJob() }) + "\n");
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
