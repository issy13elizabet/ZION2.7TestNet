import { mineGenesis } from '../src/genesis/mineGenesis.js';
import { INITIAL_TARGET, meetsTarget } from '../src/consensus/params.js';

const FOUNDATION_ADDRESS = 'Z3B1A6AEA813825404D8FFA19821EE6CF2BBB8C3ECF38E8CD27DBE81E7144338932B0F8D7F62D752C3BAB727DADB';
const ATOMIC_PER_ZION = 10n ** 12n;
const ALLOCATION_ZION = 1_000_000_000n;
const ALLOCATION_ATOMIC = ALLOCATION_ZION * ATOMIC_PER_ZION;
const GENESIS_TIMESTAMP = 1696118400; // 2025-10-01T00:00:00Z

describe('genesis determinism', () => {
  test('mines a block meeting target', () => {
    const result = mineGenesis({
      allocationAtomic: ALLOCATION_ATOMIC,
      foundationAddress: FOUNDATION_ADDRESS,
      timestamp: GENESIS_TIMESTAMP,
      maxAttempts: 5_000_000n // safety cap
    });
    expect(result.header.height).toBe(0);
    expect(result.header.powHash).toBeDefined();
    expect(meetsTarget(result.header.powHash!, INITIAL_TARGET)).toBe(true);
    expect(result.coinbase.txid.length).toBeGreaterThan(10);
  });

  test('is deterministic for same params', () => {
    const a = mineGenesis({ allocationAtomic: ALLOCATION_ATOMIC, foundationAddress: FOUNDATION_ADDRESS, timestamp: GENESIS_TIMESTAMP, maxAttempts: 5_000_000n });
    const b = mineGenesis({ allocationAtomic: ALLOCATION_ATOMIC, foundationAddress: FOUNDATION_ADDRESS, timestamp: GENESIS_TIMESTAMP, maxAttempts: 5_000_000n });
    expect(a.header.powHash).toBe(b.header.powHash);
    expect(a.header.nonce).toBe(b.header.nonce);
  });
});
