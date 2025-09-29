/**
 * ShareValidator (Phase 3 – placeholder)
 *
 * Cíl: Poskytnout jednotné rozhraní pro ověřování share / kandidátního bloku.
 * Budoucí implementace: RandomX / CryptoNight hashing (native addon nebo externí worker).
 */

export interface ShareValidationInput {
  jobId?: string;
  nonce: string;            // hex nonce / extranonce
  data: string;             // block hashing blob nebo kombinace (template + nonce)
  target?: string;          // explicitní target (hex, little-endian) pokud miner posílá
  difficulty?: number;      // případně diff ze strany poolu
  address?: string | null;  // miner address (kvůli statistikám / penalizacím)
  algorithm?: string;       // randomx | cryptonight | kawpow ...
}

export interface ShareValidationResult {
  valid: boolean;
  meetsTarget: boolean;
  hash: string;             // simulovaný / skutečný PoW hash
  effectiveDifficulty: number; // spočítaný diff z hash
  targetUsed: string;       // target proti kterému se validovalo (hex)
  reason?: string;          // pokud nevalidní
  elapsedMs: number;        // čas validace
}

export interface IShareValidator {
  validate(input: ShareValidationInput): Promise<ShareValidationResult>;
  isEnabled(): boolean;
}

/**
 * Jednoduchá implementace bez skutečného PoW.
 * Hash = keccak simulace náhradou: vezmeme simple FNV32 přes data+nonce a rozšíříme.
 * Target logika: pokud konfigurační target existuje (ENV POW_PLACEHOLDER_TARGET) -> použijeme.
 * Difficulty výpočet: diff = (MAX_TARGET / targetUsed) approximováno dle délky prefixu nul.
 */
export class PlaceholderShareValidator implements IShareValidator {
  private enabled: boolean;
  private placeholderTarget: string; // hex target (little-endian styl; pro jednoduchost big-endian interpretace)
  private readonly MAX_HEX = 'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff';

  constructor() {
    this.enabled = process.env.POW_VALIDATION_ENABLED === 'true';
    this.placeholderTarget = (process.env.POW_PLACEHOLDER_TARGET || '00000fffffffffffffffffffffffffffffffffffffffffffffffffffffffffff').toLowerCase();
  }

  public isEnabled(): boolean { return this.enabled; }

  public async validate(input: ShareValidationInput): Promise<ShareValidationResult> {
    const start = Date.now();
    // Rychlá syntetická kontrola formátu
    if (!/^([0-9a-fA-F]+)$/.test(input.nonce)) {
      return this.result(false, false, '00'.repeat(32), 0, 'malformed_nonce', start);
    }
    if (!/^([0-9a-fA-F]+)$/.test(input.data)) {
      return this.result(false, false, '00'.repeat(32), 0, 'malformed_data', start);
    }

    // Syntetický hash – FNV-like + drobná permutace
    const hash = this.syntheticHash(input.data + input.nonce);

    const targetUsed = (input.target || this.placeholderTarget).padStart(64, '0');

    // meetsTarget – prosté lexikografické srovnání (big-endian) jako placeholder
    const meetsTarget = hash < targetUsed;

    const effectiveDifficulty = this.estimateDifficulty(targetUsed);

    const valid = meetsTarget; // pro placeholder – validní jen pokud target splněn

    return this.result(valid, meetsTarget, hash, effectiveDifficulty, valid ? undefined : 'low_difficulty', start, targetUsed);
  }

  private result(valid: boolean, meetsTarget: boolean, hash: string, diff: number, reason: string | undefined, start: number, targetUsed?: string): ShareValidationResult {
    return {
      valid,
      meetsTarget,
      hash,
      effectiveDifficulty: diff,
      targetUsed: targetUsed || this.placeholderTarget,
      reason,
      elapsedMs: Date.now() - start
    };
  }

  private syntheticHash(data: string): string {
    let h = 0x811c9dc5;
    for (let i = 0; i < data.length; i++) {
      h ^= data.charCodeAt(i);
      h = (h * 0x01000193) >>> 0;
    }
    // Expand to 256-bit hex by repeating pattern & mixing
    const part = h.toString(16).padStart(8, '0');
    return (part + part.split('').reverse().join('') + part + part).slice(0, 64);
  }

  private estimateDifficulty(targetHex: string): number {
    // velmi hrubá aproximace – počet leading zeros * konstanta
    const match = targetHex.match(/^(0+)/);
    const zeros = match ? match[1].length : 0;
    return Math.max(1, zeros * 1000);
  }
}
