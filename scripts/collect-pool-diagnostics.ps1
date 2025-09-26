<#
Collect-Pool-Diagnostics.ps1

Skript pro lokální spuštění na vývojářském stroji (PowerShell). Připojí se přes SSH na vzdálený server, vytvoří `/tmp/pool-diagnostics.tar.gz` obsahující:
 - `docker logs` pro `zion-uzi-pool`
 - `/app/config.json` z uzi-pool kontejneru
 - `/app/lib/pool.js` (prvních ~2000 řádků) z uzi-pool kontejneru
 - rpc-shim metriky (`/metrics.json`)
 - seznam Redis klíčů

Pak archiv stáhne přes `scp` do lokálního adresáře `./diagnostics/<server>/`.

Použití:
  .\collect-pool-diagnostics.ps1 -ServerIp 91.98.122.165 -User root -KeyPath C:\Users\you\.ssh\id_ed25519 -OutDir .\diagnostics -Port 22

Parametry:
  -ServerIp (povinné)
  -User (default: root)
  -KeyPath (cesta k privátnímu klíči pro SSH)
  -OutDir (lokální adresář pro uložení archivu)
  -Port (SSH port, default 22)
#>

param(
  [Parameter(Mandatory=$true)][string]$ServerIp,
  [string]$User = 'root',
  [string]$KeyPath = '',
  [string]$OutDir = '.\diagnostics',
  [int]$Port = 22
)

$ErrorActionPreference = 'Stop'
function Info([string]$m){ Write-Host $m -ForegroundColor Cyan }
function Warn([string]$m){ Write-Host $m -ForegroundColor Yellow }
function Err([string]$m){ Write-Host $m -ForegroundColor Red }

# Resolve key path fallback
if ([string]::IsNullOrWhiteSpace($KeyPath)) {
  $keyDir = Join-Path $env:LOCALAPPDATA 'Zion\ssh'
  $defaultKey = Join-Path $keyDir 'id_ed25519'
  if (Test-Path $defaultKey) { $KeyPath = $defaultKey }
}
if (-not (Test-Path $KeyPath)) { Warn "SSH key not found at '$KeyPath'. Continuing without explicit key (will use agent or default key)." }

# Prepare local outdir
$absOut = Resolve-Path -LiteralPath $OutDir -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Path -ErrorAction SilentlyContinue
if (-not $absOut) { New-Item -ItemType Directory -Path $OutDir -Force | Out-Null; $absOut = Resolve-Path -LiteralPath $OutDir }
$absOut = $absOut.Path

# Build ssh/scp base args
$sshBaseArgs = @('-F','NUL')
if ($KeyPath -and (Test-Path $KeyPath)) { $sshBaseArgs += @('-i',$KeyPath) }
$sshBaseArgs += @('-o', 'UserKnownHostsFile=NUL', '-o', 'GlobalKnownHostsFile=NUL', '-o', 'StrictHostKeyChecking=accept-new', '-p', [string]$Port)

# Remote commands to collect artifacts
$remoteCmd = @'
set -euo pipefail
TMPDIR=/tmp/pool-diagnostics-$(date +%s)
mkdir -p "$TMPDIR"
# uzi-pool logs
if docker ps --format "{{.Names}}" | grep -q "zion-uzi-pool"; then
  docker logs --tail 2000 zion-uzi-pool > "$TMPDIR/uzi-pool.log" 2>&1 || true
  docker exec zion-uzi-pool sh -lc 'cat /app/config.json' > "$TMPDIR/uzi-pool.config.json" 2>/dev/null || true
  docker exec zion-uzi-pool sh -lc "sed -n '1,2000p' /app/lib/pool.js" > "$TMPDIR/uzi-pool.lib.pool.js" 2>/dev/null || true
else
  echo "zion-uzi-pool container not found" > "$TMPDIR/uzi-pool-missing.txt"
fi
# rpc-shim metrics
if command -v curl >/dev/null 2>&1; then
  curl -sS --max-time 3 http://127.0.0.1:18089/metrics.json > "$TMPDIR/rpc-shim.metrics.json" || true
else
  echo "curl not available" > "$TMPDIR/curl-missing.txt"
fi
# redis keys (best-effort)
if docker ps --format "{{.Names}}" | grep -q "zion-redis"; then
  docker exec zion-redis redis-cli -p 6379 --raw KEYS '*' > "$TMPDIR/redis-keys.txt" 2>/dev/null || true
else
  echo "zion-redis container not found" > "$TMPDIR/redis-missing.txt"
fi
# package results
tar -czf /tmp/pool-diagnostics.tar.gz -C /tmp $(basename "$TMPDIR")
echo "/tmp/pool-diagnostics.tar.gz"
'@

# Run remote ssh command
$sshArgs = $sshBaseArgs + @("$User@$ServerIp", $remoteCmd)
Info "Spouštím remote sběr diagnostiky na $User@$ServerIp..."
$proc = Start-Process -FilePath ssh -ArgumentList $sshArgs -NoNewWindow -Wait -PassThru -RedirectStandardOutput -RedirectStandardError
if ($proc.ExitCode -ne 0) { Err "SSH command failed (exit $($proc.ExitCode)). Ujistěte se, že SSH přístup funguje."; exit 1 }

# Now scp the archive back
$localArchive = Join-Path $absOut ("pool-diagnostics-$ServerIp-$(Get-Date -Format yyyyMMddHHmmss).tar.gz")
$remotePath = "$User@$ServerIp:/tmp/pool-diagnostics.tar.gz"
$scpArgs = $sshBaseArgs + @($remotePath, $localArchive)
Info "Stahuji archiv přes SCP do $localArchive..."
$scp = Start-Process -FilePath scp -ArgumentList $scpArgs -NoNewWindow -Wait -PassThru
if ($scp.ExitCode -ne 0) { Err "SCP selhalo (exit $($scp.ExitCode))."; exit 2 }

Info "Archiv stažen: $localArchive"
Write-Host "Hotovo. Rozbalte archiv a pošlete mi relevantní logy (uži-pool.log, uzi-pool.config.json, uzi-pool.lib.pool.js, rpc-shim.metrics.json)." -ForegroundColor Green
