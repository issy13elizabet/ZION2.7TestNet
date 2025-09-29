#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT=${PORT:-8601}

echo "[bootstrap] Overuji verzi..."
"$ROOT/scripts/verify-version-sync.sh"

echo "[bootstrap] Spoustim core (background)"
(node "$ROOT/core/dist/server.js" &) 2>/dev/null || echo "[WARN] core dist neni sestaven (spust 'cd core && npm install && npm run build')"

echo "[bootstrap] Cekam na /healthz"
for i in {1..20}; do
  if curl -fsS http://localhost:$PORT/healthz >/dev/null 2>&1; then
    echo "[OK] Core pripraveno"; break; fi
  sleep 0.5
  if [[ $i -eq 20 ]]; then echo "[ERR] Core nepripravene"; exit 1; fi
done

echo "[bootstrap] DONE"
