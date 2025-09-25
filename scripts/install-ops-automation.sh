#!/usr/bin/env bash
# Install systemd services for Zion ops:
#  - zion-monitor-shim.service (runs monitor-shim.sh continuously)
#  - zion-status.service + zion-status.timer (runs status.sh every 10 minutes)
#
# Usage:
#   sudo ./scripts/install-ops-automation.sh \
#     --repo /opt/Zion \
#     --base http://localhost:8080 \
#     --webhook https://example.com/webhook \
#     --interval 30 --stall-min 5

set -euo pipefail

REPO="$(cd "$(dirname "$0")"/.. && pwd)"
BASE="http://localhost:8080"
WEBHOOK=""
INTERVAL=30
STALL_MIN=5

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --base) BASE="$2"; shift 2 ;;
    --webhook) WEBHOOK="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --stall-min) STALL_MIN="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

if [[ $EUID -ne 0 ]]; then
  echo "Please run as root (sudo)." >&2
  exit 1
fi

echo "[install] Repo: $REPO"
echo "[install] Base URL: $BASE"
[[ -n "$WEBHOOK" ]] && echo "[install] Webhook: $WEBHOOK"

mkdir -p /etc/zion
cat > /etc/zion/monitor.env <<EOF
BASE="$BASE"
WEBHOOK_URL="$WEBHOOK"
INTERVAL="$INTERVAL"
STALL_MIN="$STALL_MIN"
EOF

MON_SERVICE=/etc/systemd/system/zion-monitor-shim.service
cat > "$MON_SERVICE" <<EOF
[Unit]
Description=Zion RPC Shim Monitor
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
EnvironmentFile=/etc/zion/monitor.env
WorkingDirectory=$REPO
ExecStart=$REPO/scripts/monitor-shim.sh --url \
  ${BASE} --interval ${INTERVAL} --stall-min ${STALL_MIN}
Restart=always
RestartSec=5
User=root

[Install]
WantedBy=multi-user.target
EOF

STATUS_SERVICE=/etc/systemd/system/zion-status.service
cat > "$STATUS_SERVICE" <<EOF
[Unit]
Description=Zion Status Snapshot (writes to /var/log/zion/status.log)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
EnvironmentFile=/etc/zion/monitor.env
WorkingDirectory=$REPO
ExecStart=/bin/sh -c 'mkdir -p /var/log/zion; $REPO/scripts/status.sh --base ${BASE} >> /var/log/zion/status.log 2>&1'
User=root
EOF

STATUS_TIMER=/etc/systemd/system/zion-status.timer
cat > "$STATUS_TIMER" <<'EOF'
[Unit]
Description=Run Zion Status Snapshot every 10 minutes

[Timer]
OnBootSec=2m
OnUnitActiveSec=10m
AccuracySec=1m
Unit=zion-status.service

[Install]
WantedBy=timers.target
EOF

systemctl daemon-reload
systemctl enable --now zion-monitor-shim.service
systemctl enable --now zion-status.timer

echo "[install] Done. Check services with: systemctl status zion-monitor-shim zion-status.timer"
