#!/usr/bin/env bash
set -euo pipefail

echo "[monitor] uzi-pool logs (tail -n 50)" && docker logs zion-uzi-pool --tail 50 || true
echo "\n[monitor] rpc-shim health" && curl -s http://localhost:18089/ || true
echo "\n[monitor] seed1 last lines" && docker logs zion-seed1 --tail 30 || true
echo "\n[monitor] seed2 last lines" && docker logs zion-seed2 --tail 30 || true
