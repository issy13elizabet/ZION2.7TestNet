#!/usr/bin/env bash
set -euo pipefail

# SSH key + multiplex setup to reduce password prompts
# Usage: ./scripts/ssh-key-setup.sh <server-ip> [user]

SERVER_IP="${1:-}"
SERVER_USER="${2:-root}"

if [[ -z "${SERVER_IP}" ]]; then
  echo "Usage: $0 <server-ip> [user]" >&2
  exit 1
fi

SSH_DIR="$HOME/.ssh"
KEY_FILE="$SSH_DIR/id_ed25519"
CONFIG_FILE="$SSH_DIR/config"
CONTROL_PATH="$SSH_DIR/cm_%r@%h:%p"

mkdir -p "$SSH_DIR"
chmod 700 "$SSH_DIR"

if [[ ! -f "$KEY_FILE" ]]; then
  echo "[local] Generating new ED25519 key: $KEY_FILE"
  ssh-keygen -t ed25519 -C "zion-deploy" -f "$KEY_FILE" -N ""
fi

PUB_KEY_FILE="${KEY_FILE}.pub"

echo "[local] Installing public key to ${SERVER_USER}@${SERVER_IP} (you may be prompted once)…"
if command -v ssh-copy-id >/dev/null 2>&1; then
  ssh-copy-id -i "$PUB_KEY_FILE" "${SERVER_USER}@${SERVER_IP}"
else
  # Portable fallback for systems without ssh-copy-id (e.g., macOS)
  cat "$PUB_KEY_FILE" | ssh "${SERVER_USER}@${SERVER_IP}" 'umask 077; mkdir -p ~/.ssh; touch ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys; cat >> ~/.ssh/authorized_keys'
fi

# Ensure SSH config has a per-host block enabling multiplexing and keepalive
if [[ ! -f "$CONFIG_FILE" ]]; then
  touch "$CONFIG_FILE"
  chmod 600 "$CONFIG_FILE"
fi

HOST_BLOCK=$(cat <<EOF
Host ${SERVER_IP}
  HostName ${SERVER_IP}
  User ${SERVER_USER}
  ControlMaster auto
  ControlPersist 10m
  ControlPath ${CONTROL_PATH}
  ServerAliveInterval 60
  ServerAliveCountMax 3
  StrictHostKeyChecking accept-new
EOF
)

if ! grep -q "^Host ${SERVER_IP}$" "$CONFIG_FILE" 2>/dev/null; then
  echo "[local] Adding SSH config block for ${SERVER_IP} with multiplexing…"
  printf "\n%s\n" "$HOST_BLOCK" >> "$CONFIG_FILE"
else
  echo "[local] SSH config for ${SERVER_IP} already present."
fi

echo "[local] Starting persistent master connection (keeps auth for 10 minutes)…"
ssh -fN -o ControlMaster=auto -o ControlPersist=600 -o ControlPath="$CONTROL_PATH" "${SERVER_USER}@${SERVER_IP}" || true

echo "[local] Verifying passwordless access…"
if ssh -o BatchMode=yes "${SERVER_USER}@${SERVER_IP}" 'echo SSH_OK' 2>/dev/null | grep -q SSH_OK; then
  echo "🟢 SSH key auth is working for ${SERVER_USER}@${SERVER_IP}."
else
  echo "🟡 Could not verify passwordless login yet. You may need to try once more or check server's sshd settings." >&2
fi

echo "Done. Subsequent SSH commands to ${SERVER_IP} should reuse the connection for ~10 minutes."
