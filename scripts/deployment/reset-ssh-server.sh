#!/usr/bin/env bash
set -euo pipefail

# ZION Reset vzdáleného serveru (SSH)
# ====================================
# Použití: ./scripts/deployment/reset-ssh-server.sh <server-ip> [user] [CLEAN=0|1]
#  - Bezpečně zastaví a odstraní ZION kontejnery, síť, systemd službu a /opt/zion adresář
#  - CLEAN=1 navíc odstraní Docker volumes a provede docker system prune -af

SERVER_IP="${1:-}"
SERVER_USER="${2:-root}"
CLEAN_FLAG="${CLEAN:-0}"

if [[ -z "${SERVER_IP}" ]]; then
  echo "Použití: $0 <server-ip> [user]"
  exit 1
fi

echo "[local] Reset vzdáleného serveru ${SERVER_USER}@${SERVER_IP} (CLEAN=${CLEAN_FLAG})…"
ssh "${SERVER_USER}@${SERVER_IP}" CLEAN="${CLEAN_FLAG}" bash -s <<'REMOTE'
set -euo pipefail
echo "[remote] Zastavuji systemd službu zion (pokud existuje)…"
systemctl stop zion 2>/dev/null || true
systemctl disable zion 2>/dev/null || true
rm -f /etc/systemd/system/zion.service || true
systemctl daemon-reload || true

echo "[remote] Odstraňuji kontejnery zion-* (pokud běží)…"
docker rm -f \
  zion-production \
  zion-pool \
  zion-xmrig-test 2>/dev/null || true

echo "[remote] Odstraňuji externí síť zion-seeds (pokud existuje)…"
docker network rm zion-seeds 2>/dev/null || true

if [[ "${CLEAN:-0}" == "1" ]]; then
  echo "[remote] CLEAN=1 → odstraním volumes a provedu prune…"
  docker compose down -v 2>/dev/null || true
  docker volume prune -f || true
  docker system prune -af || true
fi

echo "[remote] Odstraňuji /opt/zion a pracovní soubory…"
rm -rf /opt/zion
rm -f /tmp/zion-ssh-deploy.tar.gz

echo "[remote] Hotovo. Server je připraven na čisté nasazení."
REMOTE

echo "[local] Reset dokončen."
