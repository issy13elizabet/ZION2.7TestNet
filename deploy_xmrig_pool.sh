#!/bin/bash
# Robust deploy script for ZION Universal Pool (RandomX + KawPow placeholder)

set -e

SSH_HOST="91.98.122.165"
SSH_USER="root"
REMOTE_FILE="zion_universal_pool.py"

echo "🚀 Deploying ZION Universal Pool (multi-algo) to ${SSH_HOST} ..."

if [ ! -f "$REMOTE_FILE" ]; then
	echo "❌ Missing $REMOTE_FILE in current directory" >&2
	exit 1
fi

LOCAL_HASH=$(sha256sum "$REMOTE_FILE" | awk '{print $1}')
echo "📦 Local file hash: $LOCAL_HASH"

echo "📤 Copying pool file..."
scp "$REMOTE_FILE" ${SSH_USER}@${SSH_HOST}:~/

echo "🛠  Preparing remote start script..."
ssh ${SSH_USER}@${SSH_HOST} bash -s <<'EOS'
set -e
echo "#!/usr/bin/env bash" > start_pool.sh
echo "set -e" >> start_pool.sh
cat >> start_pool.sh <<'INNER'
echo "🔧 Stopping previous pool (if any)..."
pkill -f 'python3.*zion_universal_pool.py' 2>/dev/null || true
pkill -f 'python .*zion_universal_pool.py' 2>/dev/null || true
sleep 1

PY_BIN="python3"
if ! command -v python3 >/dev/null 2>&1; then
	if command -v python >/dev/null 2>&1; then
		PY_BIN=python
	else
		echo "❌ No python interpreter found" >&2
		exit 1
	fi
fi

echo "🚀 Starting pool with \$PY_BIN ..."
nohup \$PY_BIN zion_universal_pool.py > pool.log 2>&1 &
echo $! > pool.pid
sleep 2

if ! ps -p \\$(cat pool.pid) >/dev/null 2>&1; then
	echo "❌ Pool process not running" >&2
	exit 1
fi

echo "✅ Pool running with PID \\$(cat pool.pid)"
echo "📊 Recent log (tail):"
tail -n 25 pool.log || true

echo "🌐 Port check (3333):"
if command -v ss >/dev/null 2>&1; then
	ss -tlnp | grep 3333 || echo "(no listener yet)"
elif command -v netstat >/dev/null 2>&1; then
	netstat -tlnp | grep 3333 || echo "(no listener yet)"
else
	echo "ss/netstat not available"
fi
INNER

chmod +x start_pool.sh
./start_pool.sh || { echo "❌ start_pool.sh failed"; exit 1; }
EOS

echo "🔁 Verifying remote hash..."
REMOTE_HASH=$(ssh ${SSH_USER}@${SSH_HOST} "sha256sum ${REMOTE_FILE} 2>/dev/null | awk '{print \$1}'") || true
echo "📦 Remote file hash: ${REMOTE_HASH:-N/A}"

if [ -n "$REMOTE_HASH" ] && [ "$REMOTE_HASH" = "$LOCAL_HASH" ]; then
	echo "✅ Hash match confirmed"
else
	echo "⚠️  Hash mismatch or unavailable (REMOTE_HASH=$REMOTE_HASH)" >&2
fi

echo "✅ Deployment complete. Test login (RandomX):"
echo "   nc ${SSH_HOST} 3333  (pak poslat JSON login)"
echo "   nebo spustit XMRig: xmrig -o ${SSH_HOST}:3333 -u ZION_<addr> -p x"

echo "ℹ️ KawPow placeholder: použij --algo kawpow (custom client test skript)"