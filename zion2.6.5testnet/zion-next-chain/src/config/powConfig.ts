// Power-of-Work configuration loader
// Reads environment variables with safe defaults.

export interface PowEnvConfig {
  mode: 'COMPOSITE' | 'COSMIC' | 'RANDOMX' | 'HYBRID';
  epochBlocks: number; // blocks per seed epoch
  hybridSwitchHeight: number; // for HYBRID mode initial switch
}

function parseMode(raw?: string): PowEnvConfig['mode'] {
  if (!raw) return 'COMPOSITE';
  const up = raw.toUpperCase();
  if (['COMPOSITE','COSMIC','RANDOMX','HYBRID'].includes(up)) return up as any;
  return 'COMPOSITE';
}

export function loadPowConfig(env = process.env): PowEnvConfig {
  return {
    mode: parseMode(env.POW_MODE),
    epochBlocks: clampInt(env.POW_EPOCH_BLOCKS, 2048, 32, 1000000),
    hybridSwitchHeight: clampInt(env.POW_HYBRID_SWITCH_HEIGHT, 1000, 1, 10_000_000)
  };
}

function clampInt(val: any, def: number, min: number, max: number): number {
  const n = Number(val);
  if (!Number.isFinite(n)) return def;
  return Math.min(Math.max(Math.floor(n), min), max);
}

// Singleton pattern (lazy)
let cached: PowEnvConfig | null = null;
export function getPowConfig(): PowEnvConfig {
  if (!cached) cached = loadPowConfig();
  return cached;
}

// Test helper to reset cached config (not for production use)
export function __resetPowConfigForTests(newEnv?: any) {
  cached = null;
  if (newEnv) {
    cached = loadPowConfig(newEnv);
  }
}
