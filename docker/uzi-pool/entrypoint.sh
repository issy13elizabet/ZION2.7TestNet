#!/bin/sh
set -e

# Prepare runtime config in writable location
RUNTIME_CFG="/app/run-config.json"
SRC_CFG=""
if [ -f "$POOL_CONFIG" ]; then
  SRC_CFG="$POOL_CONFIG"
elif [ -f "/app/config.json" ]; then
  SRC_CFG="/app/config.json"
fi
if [ -n "$SRC_CFG" ]; then
  cp "$SRC_CFG" "$RUNTIME_CFG" || echo "[entrypoint] WARN: failed to copy config from $SRC_CFG"
fi
# Point pool to runtime config via symlink
ln -sf "$RUNTIME_CFG" /app/config.json
# Link coins directory if provided
if [ -d "$COINS_DIR" ]; then
  ln -sfn "$COINS_DIR" /app/coins
fi

# Apply RandomX patch (disables CryptoNight/multi-hashing paths)
if [ -f "/patch-rx.js" ]; then
  node /patch-rx.js || echo "[entrypoint] patch-rx failed (continuing)"
fi

# Buffer.toJSON() safety in pool.js (legacy code uses buff.toJSON().data)
if [ -f "/app/lib/pool.js" ]; then
  sed -i "s/var buffArray = buff.toJSON();/var buffArray = buff.toJSON();\n        if (buffArray \&\& buffArray.data) buffArray = buffArray.data;/" /app/lib/pool.js || true
fi

# Inject runtime-configured addresses (pool + donations) if provided via env
if [ -f "$RUNTIME_CFG" ]; then
  node -e '
    const fs = require("fs");
    const path = process.env.RUNTIME_CFG || "/app/run-config.json";
    let cfg = JSON.parse(fs.readFileSync(path, "utf8"));
    const env = process.env;
    let changed = false;
    if (env.POOL_ADDRESS && env.POOL_ADDRESS.trim()) {
      cfg.poolServer = cfg.poolServer || {};
      cfg.poolServer.poolAddress = env.POOL_ADDRESS.trim();
      changed = true;
    }
    if (env.DEV_ADDRESS && env.DEV_ADDRESS.trim()) {
      cfg.blockUnlocker = cfg.blockUnlocker || {};
      cfg.blockUnlocker.devAddress = env.DEV_ADDRESS.trim();
      changed = true;
    }
    if (env.CORE_DEV_ADDRESS && env.CORE_DEV_ADDRESS.trim()) {
      cfg.blockUnlocker = cfg.blockUnlocker || {};
      cfg.blockUnlocker.coreDevAddress = env.CORE_DEV_ADDRESS.trim();
      changed = true;
    }
    if (changed) {
      fs.writeFileSync(path, JSON.stringify(cfg, null, 2));
      console.log("[entrypoint] Applied runtime addresses to config.json");
    }
  '
fi

# Wait for rpc-shim health (best-effort)
if command -v curl >/dev/null 2>&1; then
  i=0; until [ $i -ge 30 ]; do
    if curl -s --max-time 2 http://rpc-shim:18089/ | grep -q "status"; then
      echo "[entrypoint] rpc-shim is up"; break; fi
    i=$((i+1)); echo "[entrypoint] waiting rpc-shim ($i/30)"; sleep 2
  done
fi

exec node init.js
# not reached
