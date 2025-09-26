param(
  [Parameter(Mandatory=$true)][string]$ServerIp,
  [string]$KeyFile
)

function Invoke-Remote($cmd) {
  $sshArgs = @('-o','StrictHostKeyChecking=no','-o','UserKnownHostsFile=/dev/null')
  if ($KeyFile -and (Test-Path $KeyFile)) { $sshArgs += @('-i', $KeyFile) }
  $sshArgs += @("root@${ServerIp}", $cmd)
  & ssh @sshArgs
}

$script = @'
set -e
docker network inspect zion-seeds >/dev/null 2>&1 || docker network create --driver bridge zion-seeds
cd /root/Zion
echo "[remote] Updating repo..."
if command -v git >/dev/null 2>&1; then
  git fetch --all --prune || true
  # Force sync to origin/master to avoid local drift
  git reset --hard origin/master || true
  # No submodules required (core vendored)
else
  echo "[remote] git not found; skipping repo update" >&2
fi
docker compose -f docker/compose.pool-seeds.yml down || docker-compose -f docker/compose.pool-seeds.yml down
docker compose -f docker/compose.pool-seeds.yml up -d --build || docker-compose -f docker/compose.pool-seeds.yml up -d --build
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
'@

# Normalize to LF and send as base64 to avoid CRLF issues
$scriptLF = ($script -replace "`r", "")
$b64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($scriptLF))
$remote = "bash -lc 'echo $b64 | base64 -d >/tmp/zion-restart.sh && chmod +x /tmp/zion-restart.sh && /bin/bash /tmp/zion-restart.sh'"
Invoke-Remote $remote
