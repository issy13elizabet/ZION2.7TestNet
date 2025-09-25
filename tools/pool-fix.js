#!/usr/bin/env node
/*
  Robust patcher for node-cryptonote-pool to harden Miner.getTargetHex for RandomX
  - Accept Buffer/string/array/{type:'Buffer',data:[..]}/bignum-like (toBuffer)
  - Avoid reverse() on non-array; always convert to Array of bytes first
*/
const fs = require('fs');
const path = '/app/lib/pool.js';
function makeGetTargetHex(assignPrefix){
  return (
    assignPrefix + 'function(target){\n' +
    '    try {\n' +
    '        var buf;\n' +
    '        if (Buffer.isBuffer(target)) {\n' +
    '            buf = Buffer.from(target);\n' +
    '        } else if (typeof target === "string") {\n' +
    '            var hex = target.replace(/^0x/, "");\n' +
    '            buf = Buffer.from(hex, "hex");\n' +
    '        } else if (target && typeof target.toBuffer === "function") {\n' +
    '            buf = Buffer.from(target.toBuffer());\n' +
    '        } else if (Array.isArray(target)) {\n' +
    '            buf = Buffer.from(target);\n' +
    '        } else if (target && target.type === "Buffer" && Array.isArray(target.data)) {\n' +
    '            buf = Buffer.from(target.data);\n' +
    '        } else if (target && typeof target === "object") {\n' +
    '            try { var j = target.toJSON ? target.toJSON() : target; if (j && Array.isArray(j.data)) buf = Buffer.from(j.data); } catch(e) {}\n' +
    '        }\n' +
    '        if (!buf) buf = Buffer.alloc(0);\n' +
    '        var arr;\n' +
    '        try { arr = Array.from(buf); } catch(e) { arr = []; }\n' +
    '        arr.reverse();\n' +
    '        return Buffer.from(arr).toString("hex");\n' +
    '    } catch (e) {\n' +
    "        try { log('warn', logSystem, 'getTargetHex failed: %j', [e && e.message ? e.message : e]); } catch(_) {}\n" +
    '        return "";\n' +
    '    }\n' +
    '}\n'
  );
}
function patch(){
  if (!fs.existsSync(path)) { console.error('[pool-fix] pool.js not found'); process.exit(2); }
  let src = fs.readFileSync(path, 'utf8');
  let changed = false;
  let mProt = src.match(/Miner\.prototype\.getTargetHex\s*=\s*function\s*\([^)]*\)\s*\{[\s\S]*?\n\}/);
  if (mProt) {
    const start = mProt.index, end = start + mProt[0].length;
    src = src.slice(0, start) + 'Miner.prototype.getTargetHex = ' + makeGetTargetHex('') + src.slice(end);
    changed = true;
    console.log('[pool-fix] Replaced Miner.prototype.getTargetHex');
  } else {
    let mFunc = src.match(/function\s+Miner\.getTargetHex\s*\(\s*[^)]*\)\s*\{[\s\S]*?\n\}/);
    if (mFunc) {
      const start = mFunc.index, end = start + mFunc[0].length;
      src = src.slice(0, start) + 'function Miner.getTargetHex' + makeGetTargetHex('') + src.slice(end);
      changed = true;
      console.log('[pool-fix] Replaced function Miner.getTargetHex');
    }
  }
  // Also normalize any direct buffArray = X.toJSON() occurrences to handle .data
  const before = src;
  src = src.replace(/var\s+buffArray\s*=\s*(buff|target)\.toJSON\(\);/g,
    'var buffArray = $1.toJSON();\n        if (buffArray && buffArray.data) buffArray = buffArray.data;\n        if (!Array.isArray(buffArray)) buffArray = Array.prototype.slice.call($1);');
  if (src !== before) { changed = true; console.log('[pool-fix] Hardened buffArray.toJSON() usages'); }

  if (changed) {
    fs.writeFileSync(path, src, 'utf8');
    console.log('[pool-fix] pool.js updated');
  } else {
    console.log('[pool-fix] No matching patterns found (maybe already patched)');
  }
}
patch();
