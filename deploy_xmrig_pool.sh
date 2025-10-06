#!/bin/bash
# Robust deploy script for ZION Universal Pool v2 (Real Hash Validation + Rewards)
# Features: Real ProgPow validation, proportional rewards, wallet integration
# Vylepšeno: detailní diagnostika, py_compile, environment info, fallback SSH options

set -euo pipefail
trap 'echo "[LOCAL] ❌ Deploy aborted (line $LINENO)"' ERR

SSH_HOST="91.98.122.165"
SSH_USER="root"
REMOTE_FILE="zion_universal_pool_v2.py"
POOL_PORT="${POOL_PORT:-3333}"

echo "🚀 Deploying ZION Universal Pool v2 (Real Validation + Rewards) to ${SSH_HOST} ..."

if [ ! -f "$REMOTE_FILE" ]; then
	echo "❌ Missing $REMOTE_FILE in current directory" >&2
	exit 1
fi

LOCAL_HASH=$(sha256sum "$REMOTE_FILE" | awk '{print $1}')
echo "📦 Local file hash: $LOCAL_HASH"

echo "🔗 Testing SSH connectivity..."
SSH_OPTS="-o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ConnectTimeout=15"
if ssh -o BatchMode=yes -o ConnectTimeout=8 ${SSH_USER}@${SSH_HOST} "echo SSH_OK" 2>/dev/null; then
  echo "✅ SSH basic check passed"
else
  echo "⚠️  Initial SSH check failed (will still attempt)." >&2
fi

echo "� Copying pool file..."
scp $SSH_OPTS "$REMOTE_FILE" ${SSH_USER}@${SSH_HOST}:~/

echo "🛠  Preparing remote start script (port $POOL_PORT)..."
ssh $SSH_OPTS ${SSH_USER}@${SSH_HOST} bash -s <<'EOS'
set -euo pipefail
trap 'echo "[REMOTE] 🔥 Failure at line $LINENO"' ERR
echo "#!/usr/bin/env bash" > start_pool.sh
echo "set -euo pipefail" >> start_pool.sh
echo "trap 'echo \"[POOL] Crash line \\${LINENO}\"' ERR" >> start_pool.sh
echo "set -x" >> start_pool.sh
cat >> start_pool.sh <<'INNER'
PORT_TO_USE=${ZION_POOL_PORT:-${POOL_PORT:-3333}}
echo "🌍 Target port: $PORT_TO_USE"
echo "🔎 Checking existing listener on port $PORT_TO_USE ..."
if command -v ss >/dev/null 2>&1; then
	EXIST_LINE=$(ss -tlnp | grep ":$PORT_TO_USE " || true)
elif command -v netstat >/dev/null 2>&1; then
	EXIST_LINE=$(netstat -tlnp 2>/dev/null | grep ":$PORT_TO_USE " || true)
else
	EXIST_LINE=""
fi
if [ -n "$EXIST_LINE" ]; then
	echo "⚠️  Port $PORT_TO_USE already in use: $EXIST_LINE" >&2
	PID_TO_KILL=$(echo "$EXIST_LINE" | sed -n 's/.*pid=\([0-9]*\).*/\1/p')
	if [ -z "$PID_TO_KILL" ]; then
		PID_TO_KILL=$(echo "$EXIST_LINE" | awk -F',' '{for(i=1;i<=NF;i++){if($i ~ /[0-9]+\//){split($i,a,"/");print a[1];exit}}}')
	fi
	if [ -n "$PID_TO_KILL" ]; then
		echo "🔨 Killing process PID=$PID_TO_KILL on port $PORT_TO_USE" >&2
		kill $PID_TO_KILL 2>/dev/null || true
		sleep 1
	else
		echo "⚠️  Could not parse PID holding port $PORT_TO_USE" >&2
	fi
fi
echo "🔧 Stopping previous pool (if any)..."
pkill -f 'python3.*zion_universal_pool.py' 2>/dev/null || true
pkill -f 'python .*zion_universal_pool.py' 2>/dev/null || true
pkill -f 'python3.*zion_universal_pool_v2.py' 2>/dev/null || true
pkill -f 'python .*zion_universal_pool_v2.py' 2>/dev/null || true
sleep 1

echo "🧭 PATH: $PATH"
echo "🔍 Searching for python candidates..."
PY_CANDIDATES="/usr/bin/python3 /usr/local/bin/python3 /usr/bin/python /usr/local/bin/python python3 python"
FOUND=""
for c in $PY_CANDIDATES; do
	if command -v $c >/dev/null 2>&1; then
		echo "✅ Found candidate: $c"
		FOUND="$FOUND $c"
	fi
done
if [ -z "$FOUND" ]; then
	echo "❌ No python interpreter found in candidates." >&2
	if command -v apt-get >/dev/null 2>&1; then
		echo "💡 You can install via: apt-get update && apt-get install -y python3" >&2
	elif command -v yum >/dev/null 2>&1; then
		echo "💡 You can install via: yum install -y python3" >&2
	else
		echo "💡 Manual python install required." >&2
	fi
	exit 1
fi
PY_BIN=$(echo "$FOUND" | awk '{print $1}')
echo "🐍 Using PY_BIN=$PY_BIN"
if [ ! -x "$PY_BIN" ]; then
	echo "❗ Selected PY_BIN not executable, listing perms:" >&2
	ls -l "$PY_BIN" >&2 || true
fi
ln -sf "$PY_BIN" /tmp/zion_py_bin 2>/dev/null || true
PY_RUN=/tmp/zion_py_bin
if [ ! -x "$PY_RUN" ]; then PY_RUN="$PY_BIN"; fi
echo "▶ PY_RUN=$PY_RUN"

echo "🧪 Syntax pre-check..."
"$PY_RUN" -m py_compile zion_universal_pool_v2.py || { echo "❌ py_compile failed"; exit 2; }

echo "🚀 Starting pool v2 with \$PY_BIN (nohup)..."
export ZION_POOL_PORT="$PORT_TO_USE"
nohup "$PY_RUN" zion_universal_pool_v2.py > pool.log 2>&1 &
echo $! > pool.pid
sleep 2

if ! ps -p $(cat pool.pid) >/dev/null 2>&1; then
    sleep 1
    if ! ps -p $(cat pool.pid) >/dev/null 2>&1; then
        echo "❌ Pool process not running" >&2
	echo "---- pool.log (last 60 lines) ----" >&2
	tail -n 60 pool.log 2>&1 >&2 || true
	echo "---- python info ----" >&2
	"$PY_RUN" --version >&2 || true
	which "$PY_RUN" >&2 || true
	echo "---- processes (python) ----" >&2
	ps -ef | grep python | grep -v grep >&2 || true
	echo "---- free -m ----" >&2
	free -m 2>&1 || true
	echo "---- ls -l ----" >&2
	ls -l >&2 || true
	exit 1
    fi
fi

echo "✅ Pool running with PID $(cat pool.pid)"
echo "🔎 Health check: waiting 1s for stability..."
sleep 1
if ! ps -p $(cat pool.pid) >/dev/null 2>&1; then
  echo "⚠️ Pool died after initial start window" >&2
  tail -n 80 pool.log >&2 || true
  exit 1
fi
echo "💚 POOL_HEALTH_OK"
echo "📊 Recent log (tail 40):"
tail -n 40 pool.log || true

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
./start_pool.sh || { echo "❌ start_pool.sh failed (above diagnostics)"; exit 1; }
EOS

echo "🔁 Verifying remote hash..."
REMOTE_HASH=$(ssh $SSH_OPTS ${SSH_USER}@${SSH_HOST} "sha256sum ${REMOTE_FILE} 2>/dev/null | awk '{print \$1}'") || true
echo "📦 Remote file hash: ${REMOTE_HASH:-N/A}"

if [ -n "$REMOTE_HASH" ] && [ "$REMOTE_HASH" = "$LOCAL_HASH" ]; then
	echo "✅ Hash match confirmed"
else
	echo "⚠️  Hash mismatch or unavailable (REMOTE_HASH=$REMOTE_HASH)" >&2
fi

echo "✅ Deployment complete. Test login (RandomX):"
echo "   nc ${SSH_HOST} 3333  (pak poslat JSON login)"
echo "   nebo spustit XMRig: xmrig -o ${SSH_HOST}:3333 -u ZION_<addr> -p x"
echo ""
echo "🎯 New Features Available:"
echo "   ✅ Real ProgPow/KawPow validation (no more placeholders)"
echo "   ✅ Proportional reward system (50 ZION per block)"
echo "   ✅ Automatic payouts (1 ZION threshold)"
echo "   ✅ Pool fee: 1% | Duplicate share detection"
echo "   ✅ Enhanced statistics & miner tracking"
echo ""
echo "⛏️ KawPow GPU Mining: SRBMiner-MULTI --algorithm kawpow --pool ${SSH_HOST}:3333 --wallet ZION_<addr>"