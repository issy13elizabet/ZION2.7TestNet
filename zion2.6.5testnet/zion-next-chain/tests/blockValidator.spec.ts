import { mineGenesis } from '../src/genesis/mineGenesis.js';
import { validateBlock } from '../src/core/blockValidator.js';
import { UTXOSet } from '../src/utxo/utxoSet.js';
import { finalizeTx } from '../src/tx/serialize.js';
import { Transaction } from '../src/tx/types.js';

const FOUNDATION_ADDRESS = 'FOUND_X';
const ATOMIC_PER_ZION = 10n ** 12n;
const ALLOCATION_ZION = 1_000_000_000n;
const ALLOCATION_ATOMIC = ALLOCATION_ZION * ATOMIC_PER_ZION;
const GENESIS_TIMESTAMP = 1696118400; // 2025-10-01

function buildGenesisBlock() {
  const { header, coinbase } = mineGenesis({
    allocationAtomic: ALLOCATION_ATOMIC,
    foundationAddress: FOUNDATION_ADDRESS,
    timestamp: GENESIS_TIMESTAMP,
    maxAttempts: 2_000_000n
  });
  // Structured tx already stored in coinbase.structured
  const tx = coinbase.structured!;
  return { header, transactions: [tx] };
}

describe('blockValidator', () => {
  test('valid genesis block', () => {
    const block = buildGenesisBlock();
    const res = validateBlock(block as any, new UTXOSet());
    expect(res.ok).toBe(true);
  });

  test('tampered merkle fails', () => {
    const block = buildGenesisBlock();
    block.header.merkleRoot = 'ff'.repeat(32);
    const res = validateBlock(block as any, new UTXOSet());
    expect(res.ok).toBe(false);
    expect(res.error).toMatch(/merkle/);
  });

  test('additional fake coinbase rejected', () => {
    const block = buildGenesisBlock();
    const extraCoinbase: Transaction = {
      version: 1,
      vin: [{ prevTxId: '0'.repeat(64), vout: 0xffffffff, scriptSig: '', sequence: 0xffffffff }],
      vout: [{ value: 1n, scriptPubKey: '51' }],
      lockTime: 0
    };
    block.transactions.push(finalizeTx(extraCoinbase));
    const res = validateBlock(block as any, new UTXOSet());
    expect(res.ok).toBe(false);
    expect(res.error).toMatch(/multiple coinbase/);
  });
});
