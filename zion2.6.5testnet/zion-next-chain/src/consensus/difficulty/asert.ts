// ASERT-like simplified difficulty adjustment (skeleton)
// new_diff = old_diff * 2^((actual - target)/half_life)
// Using floating approximation for draft; convert to bigint later.

export interface AsertParams {
  halfLife: number; // seconds
  targetTime: number; // seconds per block
  minDifficulty: number;
}

export function adjustDifficulty(oldDiff: number, actualSpacing: number, params: AsertParams): number {
  const { halfLife, targetTime, minDifficulty } = params;
  const exponent = (actualSpacing - targetTime) / halfLife;
  const factor = Math.pow(2, exponent);
  const next = Math.max(minDifficulty, Math.floor(oldDiff * factor));
  return next;
}
