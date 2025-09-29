#!/usr/bin/env ts-node
/*
 * ZION NEXT CHAIN - Genesis Generator
 * Deterministically constructs genesis block with:
 *  - Single coinbase paying 1,000,000,000 ZION (1e21 atomic units) to foundation address
 *  - Timestamp 2025-10-01T00:00:00Z
 *  - Easy target for fast mining (prefix 00000ffff...)
 *
 * NOTE: Placeholder hashing (no real blake3/keccak yet). Regenerate once real PoW integrated.
 */

import { writeFileSync, mkdirSync } from 'fs';
import { mineGenesis } from '../src/genesis/mineGenesis.js';
import { INITIAL_TARGET } from '../src/consensus/params.js';


const FOUNDATION_ADDRESS = "Z3B1A6AEA813825404D8FFA19821EE6CF2BBB8C3ECF38E8CD27DBE81E7144338932B0F8D7F62D752C3BAB727DADB";
const ATOMIC_PER_ZION = 10n**12n;
const ALLOCATION_ZION = 1_000_000_000n;
const ALLOCATION_ATOMIC = ALLOCATION_ZION * ATOMIC_PER_ZION; // 1e21
const GENESIS_TIMESTAMP = 1696118400n; // 2025-10-01T00:00:00Z
// Numeric target now handled via consensus INITIAL_TARGET (still easy). Prefix kept for legacy manifest compatibility.
const TARGET_PREFIX = INITIAL_TARGET.toString(16).slice(0,9); // for display only

(() => {
  const { header, coinbase } = mineGenesis({
    allocationAtomic: ALLOCATION_ATOMIC,
    foundationAddress: FOUNDATION_ADDRESS,
    timestamp: Number(GENESIS_TIMESTAMP)
  });
  const manifest = {
    network: 'ZION-NEXT-TESTNET-1',
    foundation_address: FOUNDATION_ADDRESS,
    allocation_zion: ALLOCATION_ZION.toString(),
    allocation_atomic: ALLOCATION_ATOMIC.toString(),
    timestamp: Number(GENESIS_TIMESTAMP),
    target_prefix: TARGET_PREFIX,
    numeric_target: INITIAL_TARGET.toString(16),
    header,
    coinbase,
    pow_note: 'blake3+keccak composite; numeric target enforced',
    deterministic: true
  };
  mkdirSync('genesis', { recursive: true });
  writeFileSync('genesis/genesis.json', JSON.stringify(manifest, null, 2));
  writeFileSync('genesis/header.hex', JSON.stringify(header, null, 2));
  writeFileSync('genesis/coinbase.hex', coinbase.raw + '\n');
  console.log('[genesis] written genesis/genesis.json');
})();
