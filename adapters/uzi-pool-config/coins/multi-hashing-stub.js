// Enhanced multi-hashing stub to keep pool initialization paths alive
// Provides minimal placeholders for functions typically expected by node-cryptonote-pool.
module.exports = {
  cryptonight: function(){ throw new Error('cryptonight disabled (stub)'); },
  // typical utilities that pool may probe
  convert_blob: function(blob){ return blob; },
  construct_block_blob: function(header, nonceBuf){
    // return header unchanged as fake block blob
    return Buffer.isBuffer(header) ? header : Buffer.from(header, 'hex');
  },
  get_block_id: function(blob){ return '0'.repeat(64); },
  hash_fast: function(data){ return '0'.repeat(64); }
};