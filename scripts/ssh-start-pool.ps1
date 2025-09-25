[CmdletBinding()]
param(
  [string]$ServerIp,
  [string]$User,
  [string]$KeyPath,
  [string]$Defaults = ''
)

$ErrorActionPreference = 'Stop'
function Info($m){ Write-Host $m }
function Warn($m){ Write-Host $m -ForegroundColor Yellow }
function Err($m){ Write-Host $m -ForegroundColor Red }

# Load defaults if any param is missing
if (-not $ServerIp -or -not $User -or -not $KeyPath) {
  try {
    $json = & (Join-Path $PSScriptRoot 'ssh-defaults.ps1') -DefaultsPath $Defaults | ConvertFrom-Json
    if (-not $ServerIp) { $ServerIp = $json.ServerIp }
    if (-not $User)     { $User     = $json.User }
    if (-not $KeyPath)  { $KeyPath  = $json.KeyPath }
  } catch {
    Write-Host "[warn] ssh-defaults not loaded: $($_.Exception.Message)" -ForegroundColor Yellow
  }
}

# SSH base args
$workDir = Join-Path $env:LOCALAPPDATA 'Zion\ssh'
if (-not (Test-Path $workDir)) { New-Item -ItemType Directory -Path $workDir -Force | Out-Null }
if ([string]::IsNullOrWhiteSpace($KeyPath)) { $KeyPath = Join-Path $workDir 'id_ed25519' }
$knownHosts = Join-Path $workDir 'known_hosts'
if (-not (Test-Path $knownHosts)) { New-Item -ItemType File -Path $knownHosts -Force | Out-Null }
$sshBase = @('-F','NUL')
$useKey = $false
if (Test-Path $KeyPath) {
  try { Get-Content -LiteralPath $KeyPath -TotalCount 1 -ErrorAction Stop | Out-Null; $useKey = $true } catch { Warn "SSH key not readable at $KeyPath; relying on agent/default key" }
} else { Warn "SSH key not found at $KeyPath; relying on agent/default key" }
if ($useKey) { $sshBase += @('-i', $KeyPath) }
$sshBase += @('-o', "UserKnownHostsFile=$knownHosts", '-o', 'GlobalKnownHostsFile=NUL', '-o','StrictHostKeyChecking=accept-new')

# Remote script: start uzi-pool service via docker compose
$remoteScript = (@'
#!/usr/bin/env bash
set -euo pipefail
# Find repo dir
CD=""
for d in /root/Zion /opt/Zion "$HOME/Zion"; do
  if [ -d "$d" ] && [ -f "$d/docker/compose.pool-seeds.yml" ]; then CD="$d"; break; fi
done
if [ -z "$CD" ]; then echo "[remote] Zion repo not found (expected /root/Zion or /opt/Zion)" >&2; exit 4; fi
cd "$CD"

# Ensure network
(docker network inspect zion-seeds >/dev/null 2>&1) || docker network create zion-seeds >/dev/null

# Start uzi-pool
if command -v docker compose >/dev/null 2>&1; then
  docker compose -f docker/compose.pool-seeds.yml up -d uzi-pool
else
  docker-compose -f docker/compose.pool-seeds.yml up -d uzi-pool
fi

# Show status and recent logs
sleep 2
echo "[remote] ps:"
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | { head -n1; grep -E "zion-(uzi-pool|rpc-shim|seed1|seed2)"; } || true

echo "[remote] uzi-pool logs (tail 60):"
docker logs --tail=60 zion-uzi-pool 2>/dev/null || true
'@).Replace("`r`n","`n")

$tmpFile = New-TemporaryFile
Set-Content -Path $tmpFile -Value $remoteScript -NoNewline -Encoding UTF8

Info "[ssh] Uploading start script to $User@$ServerIp..."
$scpArgs = $sshBase + @($tmpFile, "${User}@${ServerIp}:/tmp/zion-start-pool.sh")
$scp = Start-Process -FilePath scp -ArgumentList $scpArgs -NoNewWindow -Wait -PassThru
if ($scp.ExitCode -ne 0) { Err "SCP upload failed ($($scp.ExitCode))"; exit 1 }

Info "[ssh] Executing remote start script..."
$sshArgs = $sshBase + ("$User@$ServerIp", 'bash -lc ' + "'chmod +x /tmp/zion-start-pool.sh && bash /tmp/zion-start-pool.sh'")
$p = Start-Process -FilePath ssh -ArgumentList $sshArgs -NoNewWindow -Wait -PassThru
if ($p.ExitCode -ne 0) { Err "Remote start failed ($($p.ExitCode))"; exit 2 }

Info "[done] Pool start attempted. Use ssh-restart-services to recheck health."
