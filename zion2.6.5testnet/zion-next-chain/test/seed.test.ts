import { epochFromHeight, deriveSeed, getSeedInfo, __clearSeedCache } from '../src/consensus/pow/seed.js';
import { __resetPowConfigForTests } from '../src/config/powConfig.js';

beforeEach(() => { __clearSeedCache(); __resetPowConfigForTests({ POW_EPOCH_BLOCKS: '8' }); });

describe('seed derivation', () => {
  test('same epoch same seed', () => {
    const s1 = getSeedInfo(3).seed; // height 3 epoch 0
    const s2 = getSeedInfo(7).seed; // same epoch 0
    expect(s1).toBe(s2);
  });

  test('epoch rollover changes seed', () => {
    const seedEpoch0 = getSeedInfo(7).seed; // last of epoch 0
    const seedEpoch1 = getSeedInfo(8).seed; // first of epoch 1
    expect(seedEpoch0).not.toBe(seedEpoch1);
  });

  test('deterministic across clears', () => {
    const h = 15; // epoch 1 (since 8 blocks per epoch)
    const first = getSeedInfo(h).seed;
    __clearSeedCache();
    const second = getSeedInfo(h).seed;
    expect(first).toBe(second);
  });
});
