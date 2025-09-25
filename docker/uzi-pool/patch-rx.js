#!/usr/bin/env node
/*
  Runtime patcher for node-cryptonote-pool to support RandomX on ARM64 by:
  - Removing/bypassing CryptoNight multi-hashing paths (which crash on ARM64)
  - Injecting a processShare implementation that trusts miner's resultHash and
    computes difficulty via bignum only
  - Simplifying IsBannedIp to just return ban status (no share processing there)
*/

const fs = require('fs');
const path = '/app/lib/pool.js';

function apply() {
  if (!fs.existsSync(path)) {
    console.log('[patch-rx] pool.js not found, skipping');
    return;
  }
  let src = fs.readFileSync(path, 'utf8');
  const original = src;

  // 1) Simplify IsBannedIp function (remove corrupted cryptoNight code)
  src = src.replace(/function IsBannedIp\(ip\)\{[\s\S]*?\n\}\n/, (
    'function IsBannedIp(ip){\n' +
    '    if (!banningEnabled || !bannedIPs[ip]) return false;\n' +
    '    var bannedTime = bannedIPs[ip];\n' +
    '    var bannedTimeAgo = Date.now() - bannedTime;\n' +
    '    var timeLeft = config.poolServer.banning.time * 1000 - bannedTimeAgo;\n' +
    '    return timeLeft > 0;\n' +
    '}\n'
  ));

  // 2) Replace existing processShare (post-recordShareData) with RandomX-trusting version
  (function replaceProcessShare(){
    var anchor = src.indexOf('function recordShareData');
    if (anchor === -1) return;
    var tail = src.slice(anchor);
    var m = tail.match(/function\s+processShare\s*\([\s\S]*?\)\s*\{[\s\S]*?\n\}/);
    if (!m) return;
    var start = anchor + m.index;
    var end = start + m[0].length;
    var replacement = [
      "function processShare(miner, job, blockTemplate, nonce, resultHash){",
      "    var shareType = 'trusted';",
      "    if (!resultHash || typeof resultHash !== 'string' || resultHash.length !== 64){",
      "        log('warn', logSystem, 'Malformed result hash from %s@%s', [miner.login, miner.ip]);",
      "        return false;",
      "    }",
  "    var hash;",
  "    try{ hash = Buffer.from(resultHash, 'hex'); } catch(e){",
      "        log('warn', logSystem, 'Invalid result hash hex from %s@%s', [miner.login, miner.ip]);",
      "        return false;",
      "    }",
      "    if (hash.toString('hex') !== resultHash.toLowerCase()){",
      "        log('warn', logSystem, 'Bad hash from miner %s@%s', [miner.login, miner.ip]);",
      "        return false;",
      "    }",
  "    var hashArray = (function(){",
  "        try {",
  "            var h = hash.toJSON();",
  "            if (h && h.data) return h.data.slice();",
  "        } catch(e) {}",
  "        try { return Array.from(hash); } catch(e) { return []; }",
  "    })();",
  "    hashArray.reverse();",
  "    var hashNum = bignum.fromBuffer(Buffer.from(hashArray));",
      "    var hashDiff = diff1.div(hashNum);",
      "    if (hashDiff.ge(blockTemplate.difficulty)){",
      "        log('info', logSystem, 'Block candidate share from %s@%s at height %d, shareDiff=%s >= blockDiff=%s', [miner.login, miner.ip, job.height, hashDiff.toString(), String(blockTemplate.difficulty)]);",
      "        try {",
  "            if (cnUtil && cnUtil.construct_block_blob){",
  "                var headerBuffer = Buffer.from(blockTemplate.buffer);",
  "                try { headerBuffer.writeUInt32BE(job.extraNonce >>> 0, blockTemplate.reserveOffset); } catch(e) {}",
  "                var shareBuffer = cnUtil.construct_block_blob(headerBuffer, Buffer.from(nonce, 'hex'));",
      "                apiInterfaces.rpcDaemon('submitblock', [shareBuffer.toString('hex')], function(error, result){",
      "                    if (error){",
      "                        log('error', logSystem, 'Error submitting block at height %d from %s@%s, share type: %s - %j', [job.height, miner.login, miner.ip, shareType, error]);",
      "                        recordShareData(miner, job, hashDiff.toString(), false, null, shareType);",
      "                    } else {",
      "                        var blockFastHash = resultHash;",
      "                        log('info', logSystem, 'Block %s found at height %d by miner %s@%s - submit result: %j', [blockFastHash.substr(0, 6), job.height, miner.login, miner.ip, result]);",
      "                        recordShareData(miner, job, hashDiff.toString(), true, blockFastHash, shareType, blockTemplate);",
      "                        jobRefresh();",
      "                    }",
      "                });",
      "            } else {",
      "                recordShareData(miner, job, hashDiff.toString(), true, resultHash, shareType, blockTemplate);",
      "                jobRefresh();",
      "            }",
      "        } catch(e){",
      "            log('error', logSystem, 'Block submit path failed: %j', [e && e.message ? e.message : e]);",
      "            recordShareData(miner, job, hashDiff.toString(), true, resultHash, shareType, blockTemplate);",
      "            jobRefresh();",
      "        }",
      "        return true;",
      "    } else if (hashDiff.lt(job.difficulty)){",
      "        log('warn', logSystem, 'Rejected low difficulty share of %s from %s@%s', [hashDiff.toString(), miner.login, miner.ip]);",
      "        return false;",
      "    } else {",
      "        log('info', logSystem, 'Accepted share from %s@%s on height %d: shareDiff=%s >= jobDiff=%s', [miner.login, miner.ip, job.height, hashDiff.toString(), String(job.difficulty)]);",
      "        recordShareData(miner, job, hashDiff.toString(), false, null, shareType);",
      "        return true;",
      "    }",
      "}"
    ].join('\n');
    src = src.slice(0, start) + replacement + src.slice(end);
  })();

  // 2b) Harden Miner.getTargetHex/Miner.prototype.getTargetHex to accept Buffer/Array/hex and avoid reverse() on non-array types
  (function replaceGetTargetHex(){
    function buildReplacement(assignPrefix){
      return [
        assignPrefix + "function(target){",
        "    try {",
        "        var buf;",
        "        if (Buffer.isBuffer(target)) {",
        "            buf = Buffer.from(target);",
        "        } else if (typeof target === 'string') {",
        "            var hex = target.replace(/^0x/, '');",
        "            buf = Buffer.from(hex, 'hex');",
        "        } else if (target && typeof target.toBuffer === 'function') {",
        "            buf = Buffer.from(target.toBuffer());",
        "        } else if (Array.isArray(target)) {",
        "            buf = Buffer.from(target);",
        "        } else if (target && target.type === 'Buffer' && Array.isArray(target.data)) {",
        "            buf = Buffer.from(target.data);",
        "        } else if (target && typeof target === 'object') {",
        "            try {",
        "                var j = target.toJSON ? target.toJSON() : target;",
        "                if (j && Array.isArray(j.data)) buf = Buffer.from(j.data);",
        "            } catch(e) {}",
        "        }",
        "        if (!buf) { buf = Buffer.alloc(0); }",
        "        var arr = [];",
        "        try { arr = Array.from(buf); } catch(e) { arr = []; }",
        "        arr.reverse();",
        "        return Buffer.from(arr).toString('hex');",
        "    } catch (e) {",
        "        try { log('warn', logSystem, 'getTargetHex failed: %j', [e && e.message ? e.message : e]); } catch(_) {}",
        "        return '';",
        "    }",
        "}"
      ].join('\n');
    }
    // Try prototype form first
    var mProt = src.match(/Miner\.prototype\.getTargetHex\s*=\s*function\s*\([^)]*\)\s*\{[\s\S]*?\n\}/);
    if (mProt) {
      var start = mProt.index;
      var end = start + mProt[0].length;
      var replacement = 'Miner.prototype.getTargetHex = ' + buildReplacement('');
      src = src.slice(0, start) + replacement + src.slice(end);
      console.log('[patch-rx] Replaced Miner.prototype.getTargetHex');
      return;
    }
    // Fallback: function Miner.getTargetHex() form
    var mFunc = src.match(/function\s+Miner\.getTargetHex\s*\(\s*[^)]*\)\s*\{[\s\S]*?\n\}/);
    if (mFunc) {
      var start2 = mFunc.index;
      var end2 = start2 + mFunc[0].length;
      var replacement2 = 'function Miner.getTargetHex' + buildReplacement('');
      src = src.slice(0, start2) + replacement2 + src.slice(end2);
      console.log('[patch-rx] Replaced function Miner.getTargetHex');
    }
  })();

  // 3) Capture seed_hash in BlockTemplate for RandomX miners (insert after height line)
  if (!/seedHash\s*=/.test(src)) {
    src = src.replace(/(this\.height\s*=\s*template\.height;)/, "$1\n    this.seedHash = (template.seed_hash || template.seedHash || '');\n    if (!this.seedHash || this.seedHash.length === 0) this.seedHash = '0000000000000000000000000000000000000000000000000000000000000000';\n");
  }

  // 4) Include algo and seed_hash in Miner.getJob return so XMRig can mine RandomX
  // 4) Include algo and seed_hash in Miner.getJob return so XMRig can mine RandomX
  if (!/algo:\s*'rx\/0'/.test(src)) {
    var jobReturnRegex = /return\s*\{\s*\n\s*blob:\s*blob,\s*\n\s*job_id:\s*newJob\.id,\s*\n\s*target:\s*target,\s*\n\s*id:\s*this\.id\s*\n\s*\};/;
    if (jobReturnRegex.test(src)) {
      src = src.replace(jobReturnRegex, [
        'return {',
        '            blob: blob,',
        '            job_id: newJob.id,',
        '            target: target,',
        "            algo: 'rx/0',",
        '            seed_hash: (currentBlockTemplate.seedHash || "0000000000000000000000000000000000000000000000000000000000000000"),',
        '            height: currentBlockTemplate.height,',
        '            id: this.id',
        '        };'
      ].join('\n'));
    } else {
      src = src.replace(/(\btarget:\s*target,)/, "$1\n            algo: 'rx/0',\n            seed_hash: (currentBlockTemplate.seedHash || '0000000000000000000000000000000000000000000000000000000000000000'),\n            height: currentBlockTemplate.height,");
    }
  }

  // 5) Fallback fix: if existing processShare code still calls construct_block_blob(blockTemplate,...), rewrite it to use headerBuffer with proper Buffer arguments
  src = src.replace(
    /var\s+shareBuffer\s*=\s*cnUtil\.construct_block_blob\(\s*blockTemplate\s*,\s*Buffer\.from\(nonce,\s*'hex'\)\s*\)\s*;/,
    [
      'var headerBuffer = Buffer.from(blockTemplate.buffer);',
      'try { headerBuffer.writeUInt32BE(job.extraNonce >>> 0, blockTemplate.reserveOffset); } catch(e) {}',
      "var shareBuffer = cnUtil.construct_block_blob(headerBuffer, Buffer.from(nonce, 'hex'));"
    ].join('\n') + '\n'
  );

  if (src !== original) {
    fs.writeFileSync(path, src, 'utf8');
    console.log('[patch-rx] Patched pool.js for RandomX/ARM64');
  } else {
    console.log('[patch-rx] No changes applied (already patched?)');
  }
}

try { apply(); } catch (e) { console.error('[patch-rx] failed:', e && e.message ? e.message : e); process.exit(1); }
