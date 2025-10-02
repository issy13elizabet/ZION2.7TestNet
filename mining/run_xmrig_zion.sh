#!/usr/bin/env bash
set -euo pipefail
CFG="$(dirname "$0")/xmrig-zion-cpu.json"
if ! command -v xmrig >/dev/null 2>&1; then
  echo "❌ xmrig binárka nenalezena v PATH. Stáhni nebo nainstaluj ji (git clone && cmake .. && make -j)." >&2
  exit 1
fi
echo "🚀 Spouštím XMRig se ZION konfigurací: $CFG"
exec xmrig -c "$CFG" --http-host=0.0.0.0 --http-port=16000 --http-access-token=zion --http-enabled
