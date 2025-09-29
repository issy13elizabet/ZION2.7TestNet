#!/usr/bin/env ts-node
import { readFileSync } from 'fs';
import blake3 from 'blake3';
import { keccak256 } from 'js-sha3';
import { INITIAL_TARGET, meetsTarget } from '../src/consensus/params.js';
import { serializeHeader } from '../src/genesis/mineGenesis.js';

function fromHex(h: string): Uint8Array { if (h.length%2) throw new Error('hex'); const a=new Uint8Array(h.length/2); for(let i=0;i<a.length;i++) a[i]=parseInt(h.substr(i*2,2),16); return a; }
function blake3Hash(data: Uint8Array): Uint8Array { return blake3.hash(data); }
function keccakHashHex(data: Uint8Array): string { return keccak256.create().update(data).hex(); }
function powMix(headerBytes: Uint8Array, nonceHex: string): string {
  const h1 = blake3Hash(headerBytes);
  const nonceBytes = fromHex(nonceHex.padStart(16,'0'));
  const h2 = blake3Hash(new Uint8Array([...headerBytes, ...nonceBytes]));
  return keccakHashHex(new Uint8Array([...h1, ...h2]));
}

const manifest = JSON.parse(readFileSync('genesis/genesis.json','utf8'));
const header = manifest.header;
const coinbase = manifest.coinbase;
if (!header || !coinbase) throw new Error('manifest missing header/coinbase');

const headerBytes = serializeHeader(header);
const recomputed = powMix(headerBytes, header.nonce.toString(16));
if (recomputed !== header.powHash) {
  console.error('pow hash mismatch', { recomputed, stored: header.powHash });
  process.exit(1);
}
if (!meetsTarget(recomputed, INITIAL_TARGET)) {
  console.error('pow hash does not meet numeric target');
  process.exit(2);
}
console.log('Genesis verification OK');