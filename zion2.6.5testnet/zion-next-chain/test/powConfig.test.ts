import { getPowConfig, __resetPowConfigForTests } from '../src/config/powConfig.js';

describe('powConfig env parsing', () => {
  afterEach(() => __resetPowConfigForTests());

  test('defaults', () => {
    __resetPowConfigForTests({});
    const cfg = getPowConfig();
    expect(cfg.mode).toBe('COMPOSITE');
    expect(cfg.epochBlocks).toBe(2048);
    expect(cfg.hybridSwitchHeight).toBe(1000);
  });

  test('custom values within bounds', () => {
    __resetPowConfigForTests({ POW_MODE: 'hybrid', POW_EPOCH_BLOCKS: '4096', POW_HYBRID_SWITCH_HEIGHT: '500' });
    const cfg = getPowConfig();
    expect(cfg.mode).toBe('HYBRID');
    expect(cfg.epochBlocks).toBe(4096);
    expect(cfg.hybridSwitchHeight).toBe(500);
  });

  test('invalid mode falls back', () => {
    __resetPowConfigForTests({ POW_MODE: 'unknown' });
    expect(getPowConfig().mode).toBe('COMPOSITE');
  });
});
