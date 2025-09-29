#!/usr/bin/env node
/* Minimal RandomX enablement patch for node-cryptonote-pool (ARM64 safe)
   Changes:
   - Simplify IsBannedIp
   - Inject seedHash & algo into jobs
   - Relax init gating so pool starts even if first getblocktemplate fails (daemon busy)
*/
const fs = require('fs');
const target = '/app/lib/pool.js';

function safeReplace(src, pattern, replacement, label){
  const before = src;
  try { src = src.replace(pattern, replacement); } catch(e){ console.log('[patch-rx] replace failed', label, e.message); }
  if (before !== src) console.log('[patch-rx] applied', label); else console.log('[patch-rx] pattern not found for', label);
  return src;
}

function run(){
  if(!fs.existsSync(target)) { console.log('[patch-rx] pool.js missing'); return; }
  let src = fs.readFileSync(target,'utf8');
  const original = src;
  console.log('[patch-rx] original length', src.length);

  // 1. Simplify IsBannedIp
  src = safeReplace(src, /function IsBannedIp\(ip\)\{[\s\S]*?\n\}\n/, 'function IsBannedIp(ip){\n    if (!banningEnabled || !bannedIPs[ip]) return false;\n    var bannedTime = bannedIPs[ip];\n    var bannedTimeAgo = Date.now() - bannedTime;\n    var timeLeft = config.poolServer.banning.time * 1000 - bannedTimeAgo;\n    return timeLeft > 0;\n}\n', 'IsBannedIp');

  // 2. Add seedHash to BlockTemplate
  if(!/this\.seedHash/.test(src)) {
    src = safeReplace(src, /(this\.height\s*=\s*template\.height;)/, '$1\n    this.seedHash = (template.seed_hash || template.seedHash || \"0000000000000000000000000000000000000000000000000000000000000000\");', 'seedHash');
  }

  // 3. Add algo & seed_hash & height to Miner.getJob return
  if(!/algo:\s*'rx\/0'/.test(src)) {
    src = safeReplace(src,
      /return\s*\{\s*\n\s*blob:\s*blob,\s*\n\s*job_id:\s*newJob\.id,\s*\n\s*target:\s*target,\s*\n\s*id:\s*this\.id\s*\n\s*\};/,
      'return {\n            blob: blob,\n            job_id: newJob.id,\n            target: target,\n            algo: \"rx/0\",\n            seed_hash: (currentBlockTemplate.seedHash || \"0000000000000000000000000000000000000000000000000000000000000000\"),\n            height: currentBlockTemplate.height,\n            id: this.id\n        };',
      'jobReturn');
  }

  // 4. Relax init gating (start server even if first jobRefresh fails)
  // Instead of starting without a block template, disable original init IIFE and inject retry wrapper
  src = safeReplace(src,
    /\(function init\(\)\{/, '(function init_disabled(){ /* disabled by patch-rx */');
  // Inject retry logic appended near EOF (before final module export / end). We'll append a marker block.
  if(!/__patch_rx_init_retry/.test(src)){
    src += '\n// __patch_rx_init_retry\n' +
      '(function(){\n' +
      '  try {\n' +
      '    if (typeof jobRefresh === "function" && typeof startPoolServerTcp === "function") {\n' +
      '      var attempts = 0;\n' +
      '      function attempt(){\n' +
      '        jobRefresh(true, function(success){\n' +
      '          if (!success){\n' +
      '            attempts++;\n' +
      '            if (attempts < 20){\n' +
      '              try { log(\'warn\', logSystem, \"Initial jobRefresh failed (attempt %d)\", [attempts]); } catch(e){ console.log(\'[patch-rx] jobRefresh fail\', e.message); }\n' +
      '              return setTimeout(attempt, 2000);\n' +
      '            } else {\n' +
      '              console.log(\'[patch-rx] giving up after 20 jobRefresh attempts\');\n' +
      '              return;\n' +
      '            }\n' +
      '          }\n' +
      '          try { log(\'info\', logSystem, \"Initial jobRefresh succeeded after %d attempt(s)\", [attempts+1]); } catch(e){}\n' +
      '          startPoolServerTcp(function(){ console.log(\'[patch-rx] stratum servers started\'); });\n' +
      '        });\n' +
      '      }\n' +
      '      if (!currentBlockTemplate){ attempt(); }\n' +
      '    }\n' +
      '  } catch(e){ console.log(\'[patch-rx] init retry wrapper error\', e.message); }\n' +
      '})();\n';
  }

  // 5. Log ports summary (non-fatal if missing)
  try {
    const cfgPath = '/app/config.json';
    if (fs.existsSync(cfgPath)) {
      const cfg = JSON.parse(fs.readFileSync(cfgPath,'utf8'));
      if (cfg.poolServer && Array.isArray(cfg.poolServer.ports)) {
        console.log('[patch-rx] ports configured:', cfg.poolServer.ports.map(p=>p.port+':' + (p.algo||'n/a')).join(','));
      }
    }
  } catch(e){ console.log('[patch-rx] port summary failed', e.message); }

  if(src !== original){
    fs.writeFileSync(target, src, 'utf8');
    console.log('[patch-rx] patch applied OK');
  } else {
    console.log('[patch-rx] no changes (already patched)');
  }

  // Patch utils.isValidAddress to accept Z3 addresses (temporary relaxed validation)
  try {
    const utilPath = '/app/lib/utils.js';
    if (fs.existsSync(utilPath)) {
      let u = fs.readFileSync(utilPath,'utf8');
      if (!/__patch_rx_addr/.test(u)) {
        u += '\n// __patch_rx_addr\nexports.isValidAddress = function(address){ return typeof address === "string" && /^Z3[1-9A-HJ-NP-Za-km-z]{90,110}$/.test(address); };\n';
        fs.writeFileSync(utilPath, u, 'utf8');
        console.log('[patch-rx] utils.isValidAddress overridden for Z3');
      }
    }
  } catch(e){ console.log('[patch-rx] utils patch failed', e.message); }

  // Secondary instrumentation patch applied after primary write (in-place edit of pool.js)
  try {
    let p = fs.readFileSync(target,'utf8');
    if (!/__patch_rx_force_start/.test(p)) {
      // Instrument startPoolServerTcp
      p = p.replace(/function startPoolServerTcp\(callback\)\{/, 'function startPoolServerTcp(callback){\n    console.log(\'[patch-rx] entering startPoolServerTcp\');');
      // Force start after definitions
      p += '\n// __patch_rx_force_start\nsetTimeout(function(){\n  try {\n    if (typeof startPoolServerTcp === "function") {\n      console.log("[patch-rx] forcing stratum start");\n      startPoolServerTcp(function(ok){ console.log("[patch-rx] forced start callback", ok); });\n    } else { console.log("[patch-rx] startPoolServerTcp undefined"); }\n  } catch(e){ console.log("[patch-rx] force start error", e && e.message); }\n}, 1500);\n';
      fs.writeFileSync(target, p, 'utf8');
      console.log('[patch-rx] added force start instrumentation');
    }
  } catch(e){ console.log('[patch-rx] instrumentation add failed', e.message); }

  // Patch login handler to defer job until block template exists
  try {
    let p2 = fs.readFileSync(target,'utf8');
    if (!/__patch_rx_login_wait/.test(p2)) {
      const loginPattern = /case 'login':[\s\S]*?break;/;
      if (loginPattern.test(p2)) {
        p2 = p2.replace(loginPattern, function(block){
          if (block.indexOf('currentBlockTemplate') !== -1) return block; // already patched
          return block.replace(/sendReply\(null, \{[\s\S]*?status: 'OK'\n\s*\}\);/, "if (!currentBlockTemplate){ try { log('warn', logSystem, 'No block template yet for miner %s', [params.login]); } catch(e){}\n                // queue minimal response; miner will request getjob repeatedly\n                sendReply(null, { id: minerId, job: { status: 'NOJOB' }, status: 'OK' });\n            } else {\n                sendReply(null, { id: minerId, job: miner.getJob(), status: 'OK' });\n            }") + '\n            // __patch_rx_login_wait';
        });
        fs.writeFileSync(target, p2, 'utf8');
        console.log('[patch-rx] login handler patched for template wait');
      } else {
        console.log('[patch-rx] login pattern not found');
      }
    }
  } catch(e){ console.log('[patch-rx] login patch failed', e.message); }
}

try { run(); } catch(e){ console.error('[patch-rx] fatal', e); process.exit(1); }
