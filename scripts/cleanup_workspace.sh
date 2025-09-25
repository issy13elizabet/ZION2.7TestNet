#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[cleanup] Docker: zastavuji orphan test miner kontejnery (pokud existují)…"
docker ps -aq --filter name=zion-xmrig-test | xargs -r docker rm -f || true

echo "[cleanup] Docker: prune dangling image a builder cache (bez dopadu na běžící kontejnery)…"
docker image prune -f || true
docker builder prune -f || true

echo "[cleanup] Git status (pro kontrolu před push)…"
git status --short || true

echo "[cleanup] Hotovo. Pro push použij: scripts/push_logs.sh"