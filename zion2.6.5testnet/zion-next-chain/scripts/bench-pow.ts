#!/usr/bin/env ts-node
import { powHash } from '../src/consensus/pow/index.js';
import { INITIAL_TARGET } from '../src/consensus/params.js';

interface Entry { mode: 'COMPOSITE'|'COSMIC'|'RANDOMX'; samples: number; }

const headerBytes = new Uint8Array(80); // synthetic header
for (let i=0;i<headerBytes.length;i++) headerBytes[i]=i & 0xff;

async function bench(entry: Entry) {
  const start = Date.now();
  let nonce = 0n;
  for (let i=0;i<entry.samples;i++) {
    powHash(headerBytes, nonce, { height: 100 }, entry.mode);
    nonce++;
  }
  const dt = (Date.now()-start)/1000;
  const hps = entry.samples / dt;
  return { mode: entry.mode, samples: entry.samples, seconds: dt, hps };
}

(async () => {
  const SAMPLES = Number(process.env.BENCH_SAMPLES || 5000);
  const modes: Entry[] = [
    { mode: 'COMPOSITE', samples: SAMPLES },
    { mode: 'COSMIC', samples: SAMPLES },
    { mode: 'RANDOMX', samples: SAMPLES }
  ];
  const results = [];
  for (const m of modes) results.push(await bench(m));
  console.table(results.map(r => ({ mode: r.mode, hps: Math.round(r.hps), seconds: r.seconds.toFixed(3) })));
})();
