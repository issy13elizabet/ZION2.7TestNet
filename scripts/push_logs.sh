#!/usr/bin/env bash
set -euo pipefail

MSG=${1:-"chore(logs): update deployment + handover ($(date -u +%Y-%m-%dT%H:%M:%SZ))"}

git add -A
git commit -m "$MSG" || true
git pull --rebase
git push

echo "[push_logs] Pushed with message: $MSG"
