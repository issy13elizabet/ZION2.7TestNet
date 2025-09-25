// Minimal Stratum-like TCP server (stub) for Zion
// - Listens on 0.0.0.0:3333 by default
// - Accepts JSON-RPC lines terminated by \n
const net = require('net');
const crypto = require('crypto');
require('dotenv').config();

const HOST = process.env.POOL_BIND || '0.0.0.0';
const PORT = parseInt(process.env.POOL_PORT || '3333', 10);

const clients = new Set();

function makeResponse(id, result) {
  return JSON.stringify({ id, result, error: null });
}

function makeJobNotification() {
  // Simplified job; fields layout inspired by stratum, but not real mining yet
  const jobId = crypto.randomBytes(4).toString('hex');
  const header = crypto.randomBytes(32).toString('hex');
  const target = 'ffff' + '0'.repeat(60); // very easy target
  return JSON.stringify({
    id: null,
    method: 'mining.notify',
    params: [jobId, header, target]
  });
}

function handleLine(sock, line) {
  let msg;
  try {
    msg = JSON.parse(line);
  } catch {
    return; // ignore invalid JSON
  }

  const { id, method, params } = msg;
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
