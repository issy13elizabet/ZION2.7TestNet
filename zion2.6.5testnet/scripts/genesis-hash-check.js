#!/usr/bin/env node
import { createHash } from 'crypto';
import { readFileSync } from 'fs';
const genesisPath = new URL('../config/network/genesis.json', import.meta.url);
const raw = readFileSync(genesisPath, 'utf-8');
const json = JSON.parse(raw);
// Simple canonical hash: sha256( network + timestamp + merkle_root + nonce )
const h = createHash('sha256');
h.update(json.network);
h.update(String(json.timestamp));
h.update(json.block.merkle_root || '');
h.update(String(json.block.nonce));
const digest = h.digest('hex');
console.log('[genesis-hash-check] computed=', digest);
if (json.block.hash && json.block.hash !== 'PLACEHOLDER_GENESIS_HASH' && json.block.hash !== digest) {
  console.error('[WARN] genesis.block.hash mismatch expected', json.block.hash, 'got', digest);
  process.exitCode = 1;
}
