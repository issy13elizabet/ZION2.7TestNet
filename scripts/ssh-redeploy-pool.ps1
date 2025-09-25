[CmdletBinding()]
Param(
  [Parameter(Mandatory=$true)][string]$ServerIp,
  [string]$User = 'root',
  [switch]$Clean,
  # Optional path to SSH private key (e.g. a key stored in the repo)
  [string]$KeyPath
)

$ErrorActionPreference = 'Stop'

function Info($m){ Write-Host $m }
function Warn($m){ Write-Host $m -ForegroundColor Yellow }
function Err($m){ Write-Host $m -ForegroundColor Red }

# Use isolated SSH workspace prepared by ssh-key-setup.ps1 unless a custom -KeyPath is provided
$workDir = Join-Path $env:LOCALAPPDATA 'Zion\ssh'
if ([string]::IsNullOrWhiteSpace($KeyPath)) {
  $keyPath = Join-Path $workDir 'id_ed25519'
} else {
  $keyPath = $KeyPath
}
$knownHosts = Join-Path $workDir 'known_hosts'
if (-not (Test-Path $workDir)) { New-Item -ItemType Directory -Path $workDir -Force | Out-Null }
if (-not (Test-Path $knownHosts)) { New-Item -ItemType File -Path $knownHosts -Force | Out-Null }
if (-not (Test-Path $keyPath)) { Err "SSH key not found: $keyPath. Provide -KeyPath or run scripts/ssh-key-setup.ps1."; exit 1 }

# Base SSH/SCP args and connectivity test
$sshBaseArgs = @('-F','NUL','-i', $keyPath, '-o', "UserKnownHostsFile=$knownHosts", '-o', 'GlobalKnownHostsFile=NUL', '-o','StrictHostKeyChecking=accept-new')
Info "[ssh] Testing SSH connectivity to $User@$ServerIp..."
# Try with BatchMode first; if it fails (e.g., key requires agent), retry without BatchMode
$tmpOut = New-TemporaryFile
$testArgs1 = $sshBaseArgs + @('-o','BatchMode=yes',"$User@$ServerIp",'echo SSH_OK')
$p = Start-Process -FilePath ssh -ArgumentList $testArgs1 -NoNewWindow -Wait -PassThru -RedirectStandardOutput $tmpOut
if ($p.ExitCode -ne 0) {
  Warn "[ssh] BatchMode test failed (code $($p.ExitCode)). Retrying without BatchMode..."
  $testArgs2 = $sshBaseArgs + @("$User@$ServerIp",'echo SSH_OK')
  $p = Start-Process -FilePath ssh -ArgumentList $testArgs2 -NoNewWindow -Wait -PassThru -RedirectStandardOutput $tmpOut
  if ($p.ExitCode -ne 0) { Err "SSH connection failed (code $($p.ExitCode))."; exit 2 }
}

# Prepare remote deploy script content (remote will extract uploaded tarball)
$remoteScript = (
@'
set -euxo pipefail
REMOTE_BASE="/opt/zion"
REMOTE_REPO_DIR="$REMOTE_BASE/Zion"
TARBALL="/tmp/zion-src.tgz"

echo "[remote] Preparing $REMOTE_BASE..."
sudo mkdir -p "$REMOTE_BASE"
sudo chown -R "$USER":"$USER" "$REMOTE_BASE"

echo "[remote] Deploying from tarball: $TARBALL"
rm -rf "$REMOTE_REPO_DIR"
mkdir -p "$REMOTE_REPO_DIR"
if [ -f "$TARBALL" ]; then
  tar -xzf "$TARBALL" -C "$REMOTE_REPO_DIR"
else
  echo "[remote] ERROR: Tarball not found: $TARBALL" >&2
  exit 42
fi

cd "$REMOTE_REPO_DIR"

echo "[remote] Compose services present:"
docker compose -f docker/compose.pool-seeds.yml config --services || true

echo "[remote] Docker network ensure: zion-seeds"
docker network create zion-seeds >/dev/null 2>&1 || true

if [ "${CLEAN:-0}" = "1" ]; then
  echo "[remote] CLEAN: stopping and removing volumes..."
  docker compose -f docker/compose.pool-seeds.yml down -v || true
  for v in seed1-data seed2-data pool-data; do docker volume rm "$v" >/dev/null 2>&1 || true; done
  for c in zion-uzi-pool zion-rpc-shim zion-seed1 zion-seed2 zion-redis; do docker rm -f "$c" >/dev/null 2>&1 || true; done
fi

echo "[remote] Building images..."
echo "[remote] Dockerfile snippet (UPNP flags):"
grep -n "USE_UPNP\|NO_UPNP\|miniupnpc" docker/Dockerfile.zion-cryptonote.minimal || true
# Build core runtime image used by walletd (includes zion_walletd)
# Use minimal Dockerfile variant which can self-fetch zion-cryptonote if local dir is empty
docker build --no-cache -t zion:production-fixed -f docker/Dockerfile.zion-cryptonote.minimal .
docker build -t zion:uzi-pool -f docker/uzi-pool/Dockerfile .
docker compose -f docker/compose.pool-seeds.yml build --no-cache seed1 seed2 rpc-shim

echo "[remote] Starting core services..."
docker compose -f docker/compose.pool-seeds.yml up -d --force-recreate seed1 seed2 redis

echo "[remote] Waiting for seed1 to become healthy..."
for i in $(seq 1 60); do
  st=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' zion-seed1 2>/dev/null || echo none)
  if [ "$st" = "healthy" ]; then echo "[remote] seed1 is healthy"; break; fi
  if [ $i -eq 60 ]; then echo "[remote] WARN: seed1 not healthy after wait"; fi
  sleep 3
done

echo "[remote] Restarting rpc-shim & uzi-pool..."
docker compose -f docker/compose.pool-seeds.yml up -d --force-recreate rpc-shim uzi-pool

echo "[remote] Starting walletd (no deps)..."
docker compose -f docker/compose.pool-seeds.yml up -d --no-deps walletd

echo "[remote] Waiting for walletd to become healthy..."
for i in $(seq 1 40); do
  st=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' zion-walletd 2>/dev/null || echo none)
  if [ "$st" = "healthy" ]; then echo "[remote] walletd is healthy"; break; fi
  if [ $i -eq 40 ]; then echo "[remote] WARN: walletd not healthy after wait"; fi
  sleep 3
done

echo "[remote] Starting wallet-adapter..."
services=$(docker compose -f docker/compose.pool-seeds.yml config --services | tr -d '\r')
if echo "$services" | grep -qx 'wallet-adapter'; then
  srv_wallet_adapter='wallet-adapter'
else
  echo "[remote] ERROR: wallet-adapter service not found in compose services: $services" >&2
  exit 17
fi
docker compose -f docker/compose.pool-seeds.yml up -d --no-deps "$srv_wallet_adapter"

echo "[remote] ps:"
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | (head -n1; grep -E 'zion-(seed1|seed2|redis|rpc-shim|uzi-pool|walletd|wallet-adapter)') || true

echo "[remote] Probing shim health:"
curl -s http://localhost:18089/ || true
echo

echo "[remote] Listening sockets:"
ss -tulpn | grep -E ':3333|:18089' || true

echo "[remote] Seed1/Seed2 logs (last 200 lines):"
docker logs --tail=200 zion-seed1 || true
echo '---'
docker logs --tail=200 zion-seed2 || true
echo '---'
echo "[remote] walletd last 80 lines:"
docker logs --tail=80 zion-walletd || true
echo '---'
echo "[remote] wallet-adapter last 80 lines:"
docker logs --tail=80 zion-wallet-adapter || true
echo '---'
echo "[remote] Seed1 health object:"
docker inspect --format '{{json .State.Health}}' zion-seed1 || true
echo "[remote] Check seed1 conf & RPC port inside container:"
docker exec zion-seed1 sh -lc 'ls -l /config; echo === /config/zion.conf ===; sed -n "1,160p" /config/zion.conf; echo === LISTEN 18081 ===; ss -tulpn | grep :18081 || true' || true

# Best-effort firewall rules if ufw is active
if command -v ufw >/dev/null 2>&1; then
  status=$(ufw status 2>/dev/null || true)
  echo "$status" | grep -qi inactive || { ufw allow 3333/tcp || true; ufw allow 18089/tcp || true; ufw allow 18099/tcp || true; ufw status || true; }
fi
'@
).Replace("`r`n","`n")

# Write remote script to temp and copy up
$tmpScript = New-TemporaryFile
Set-Content -Path $tmpScript -Value $remoteScript -NoNewline -Encoding UTF8

Info "[ssh] Uploading deploy script..."
$scpArgs = $sshBaseArgs + @($tmpScript, "${User}@${ServerIp}:/tmp/zion-redeploy.sh")
$scp = Start-Process -FilePath scp -ArgumentList $scpArgs -NoNewWindow -Wait -PassThru
if ($scp.ExitCode -ne 0) { Err "SCP failed (code $($scp.ExitCode))."; exit 3 }

# Create a tarball of current repo and upload it to the server
Info "[ssh] Creating source tarball..."
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$tarLocal = Join-Path $env:TEMP ("zion-src-" + [Guid]::NewGuid().ToString() + ".tgz")
& tar -czf $tarLocal -C $repoRoot . | Out-Null
if (-not (Test-Path $tarLocal)) { Err "Failed to create source tarball at $tarLocal"; exit 3 }

Info "[ssh] Uploading source tarball..."
$scpArgsTar = $sshBaseArgs + @($tarLocal, "${User}@${ServerIp}:/tmp/zion-src.tgz")
$scpTar = Start-Process -FilePath scp -ArgumentList $scpArgsTar -NoNewWindow -Wait -PassThru
if ($scpTar.ExitCode -ne 0) { Err "SCP tarball failed (code $($scpTar.ExitCode))."; exit 3 }

# Skipping per-file SCP because we deploy the full source tarball

Info "[ssh] Executing remote deploy script..."
$envMap = @{}
if ($Clean) { $envMap['CLEAN'] = '1' } else { $envMap['CLEAN'] = '0' }
$envAssign = ($envMap.GetEnumerator() | ForEach-Object { $_.Key + '=' + $_.Value }) -join ' '
$cmd = 'bash -lc ' + "'" + $envAssign + '; set -euo pipefail; sed -i ' + '"' + 's/\r$//' + '"' + ' /tmp/zion-redeploy.sh; bash /tmp/zion-redeploy.sh' + "'"
$sshArgs2 = $sshBaseArgs + ("${User}@${ServerIp}", $cmd)
$run = Start-Process -FilePath ssh -ArgumentList $sshArgs2 -NoNewWindow -Wait -PassThru
if ($run.ExitCode -ne 0) { Err "Remote deploy failed (code $($run.ExitCode))."; exit 4 }

Info "[done] Remote redeploy finished. Re-test from Windows:"
Info "       Test-NetConnection -ComputerName $ServerIp -Port 3333"
Info "       Test-NetConnection -ComputerName $ServerIp -Port 18089"
