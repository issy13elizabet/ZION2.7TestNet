// Zion coin stub module used by patched pool runtime.
// Provides minimal interface expected by node-cryptonote-pool without multi-hashing.
module.exports = {
  name: 'Zion',
  symbol: 'ZION',
  algorithm: 'randomx',
  coinUnits: 1000000000000, // 1e12 atomic units
  addressPrefix: 'Z3',
  validateAddress: function(addr){
    return typeof addr === 'string' && /^Z3[1-9A-HJ-NP-Za-km-z]{90,98}$/.test(addr);
  },
  getDifficultyTarget: function(){ return 120; },
  getBlockTime: function(){ return 120; },
  // Minimal stubs used by pool.js (avoid crashes where original cn functions lived)
  convertBlob: function(blob){ return blob; },
  constructNewBlob: function(blob){ return blob; },
  hashFast: function(blob){ return '0'.repeat(64); },
  getDaemonResponse: function(resp){ return resp; }
};
