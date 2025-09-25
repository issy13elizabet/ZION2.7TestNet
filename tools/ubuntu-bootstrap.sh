#!/usr/bin/env bash
set -euo pipefail

# Ubuntu bootstrap skript pro ZION v2.6 (lokální PC)

echo "[1/5] Instalace Dockeru a Compose"
curl -fsSL https://get.docker.com | sh
usermod -aG docker "${SUDO_USER:-$USER}" || true

echo "[2/5] Instalace Git"
apt update && apt install -y git curl

echo "[3/5] (Volitelné) Node.js LTS"
curl -fsSL https://deb.nodesource.com/setup_lts.x | bash -
apt install -y nodejs || true

echo "[4/5] Klon repozitáře"
mkdir -p /opt/zion && cd /opt/zion
if [ ! -d zion-repo ]; then
  git clone https://github.com/Maitreya-ZionNet/Zion-2.6-TestNet.git zion-repo
fi
cd zion-repo

echo "[5/5] Příprava .env (pokud chybí)"
if [ ! -f .env ]; then
  cp .env.example .env || true
  sed -i 's/^ZION_RPC_BIND=.*/ZION_RPC_BIND=127.0.0.1/' .env || true
  sed -i 's/^POOL_PORT=.*/POOL_PORT=3334/' .env || echo 'POOL_PORT=3334' >> .env
fi

echo "Spuštění služeb (node + pool profil)"
docker compose -f docker-compose.prod.yml down --remove-orphans || true
docker compose -f docker-compose.prod.yml --profile pool up -d

echo "Hotovo. Ověřte běh: docker ps"
