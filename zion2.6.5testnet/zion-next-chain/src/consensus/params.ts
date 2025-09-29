// Consensus parameters for Zion Next Chain (initial draft)
// Numeric difficulty: hash_as_bigint <= target

export const CONSENSUS = {
  NETWORK_ID: 'ZION-NEXT-TESTNET-1',
  TARGET_BLOCK_TIME_SEC: 60,
  // Initial target chosen to be easy: treat hash as 256-bit big-endian
  // target = 0x0000ffff........................ (approx difficulty placeholder)
  INITIAL_TARGET_HEX: '0000ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
  MAX_TARGET_HEX:    '0000ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
};

export function hexToBigInt(hex: string): bigint {
  return BigInt('0x' + hex.replace(/^0x/, ''));
}

export const INITIAL_TARGET = hexToBigInt(CONSENSUS.INITIAL_TARGET_HEX);
export const MAX_TARGET = hexToBigInt(CONSENSUS.MAX_TARGET_HEX);

export function hashHexToBigInt(hashHex: string): bigint {
  return BigInt('0x' + hashHex);
}

export function meetsTarget(hashHex: string, target: bigint): boolean {
  // Hash is interpreted as big-endian integer
  const hv = hashHexToBigInt(hashHex);
  return hv <= target;
}
