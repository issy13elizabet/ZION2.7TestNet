import { currentPowAlgorithm, powHash } from '../src/consensus/pow/index.js';
import { __resetPowConfigForTests } from '../src/config/powConfig.js';

function dummyHeader(): Uint8Array { return new TextEncoder().encode('HEADER'); }

describe('pow router selection', () => {
  afterEach(() => __resetPowConfigForTests());

  test('composite default', () => {
    __resetPowConfigForTests({});
    expect(currentPowAlgorithm(0,'COMPOSITE')).toBe('composite');
  });

  test('hybrid before and after switch', () => {
    __resetPowConfigForTests({ POW_MODE: 'HYBRID', POW_HYBRID_SWITCH_HEIGHT: '10' });
    expect(currentPowAlgorithm(0,'HYBRID')).toBe('randomx');
    expect(currentPowAlgorithm(9,'HYBRID')).toBe('randomx');
    expect(currentPowAlgorithm(10,'HYBRID')).toBe('cosmic');
  });
});
