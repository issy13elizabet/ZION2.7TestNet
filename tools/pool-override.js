(function(){
  try {
    if (typeof Miner !== 'undefined' && Miner && Miner.prototype) {
      Miner.prototype.getTargetHex = function(target){
        try {
          var buf;
          if (Buffer.isBuffer(target)) {
            buf = Buffer.from(target);
          } else if (typeof target === 'string') {
            var hex = target.replace(/^0x/, '');
            buf = Buffer.from(hex, 'hex');
          } else if (target && target.type === 'Buffer' && Array.isArray(target.data)) {
            buf = Buffer.from(target.data);
          } else if (target && typeof target.toBuffer === 'function') {
            buf = Buffer.from(target.toBuffer());
          } else if (Array.isArray(target)) {
            buf = Buffer.from(target);
          } else if (target && target.toJSON) {
            try {
              var j = target.toJSON();
              if (j && Array.isArray(j.data)) buf = Buffer.from(j.data);
            } catch(e) {}
          }
          if (!buf) { buf = Buffer.alloc(0); }
          var arr = [];
          try { arr = Array.from(buf); } catch(e) { arr = []; }
          arr.reverse();
          return Buffer.from(arr).toString('hex');
        } catch (e) {
          try { console.error('[override] getTargetHex failed:', e && e.message ? e.message : e); } catch(_) {}
          return '';
        }
      };
      try { console.log('[override] Miner.getTargetHex patched'); } catch(_) {}
    }
  } catch (e) {
    try { console.error('[override] patch failed:', e && e.message ? e.message : e); } catch(_) {}
  }
})();
