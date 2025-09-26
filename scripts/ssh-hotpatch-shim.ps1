[CmdletBinding()]
Param(
  [Parameter(Mandatory=$true)][string]$ServerIp,
  [string]$User = 'root',
  # Optional path to SSH private key (e.g., repo-managed key)
  [string]$KeyPath,
  # Also bounce uzi-pool to pick up jobs
  [switch]$RestartPool
)

$ErrorActionPreference = 'Stop'
function Info($m){ Write-Host $m }
function Warn($m){ Write-Host $m -ForegroundColor Yellow }
function Err($m){ Write-Host $m -ForegroundColor Red }

# Resolve local file to hotpatch
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$shimFile = Join-Path $repoRoot 'adapters\zion-rpc-shim\server.js'
if (-not (Test-Path $shimFile)) { Err "Local shim file not found: $shimFile"; exit 1 }

# SSH workspace (re-use from ssh-redeploy)
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

$sshBaseArgs = @('-F','NUL','-i', $keyPath, '-o', "UserKnownHostsFile=$knownHosts", '-o', 'GlobalKnownHostsFile=NUL', '-o','StrictHostKeyChecking=accept-new')
Info "[ssh] Testing SSH connectivity to $User@$ServerIp..."
$tmpOut = New-TemporaryFile
$testArgs1 = $sshBaseArgs + @('-o','BatchMode=yes',"$User@$ServerIp",'echo SSH_OK')
$p = Start-Process -FilePath ssh -ArgumentList $testArgs1 -NoNewWindow -Wait -PassThru -RedirectStandardOutput $tmpOut
if ($p.ExitCode -ne 0) {
  Warn "[ssh] BatchMode test failed (code $($p.ExitCode)). Retrying without BatchMode..."
  $testArgs2 = $sshBaseArgs + @("$User@$ServerIp",'echo SSH_OK')
  $p = Start-Process -FilePath ssh -ArgumentList $testArgs2 -NoNewWindow -Wait -PassThru -RedirectStandardOutput $tmpOut
  if ($p.ExitCode -ne 0) { Err "SSH connection failed (code $($p.ExitCode))."; exit 2 }
}

# Upload only the single file to a temp location
Info "[ssh] Uploading server.js hotpatch..."
# Quote local path to handle spaces (e.g., 'Zion TestNet')
$localShim = '"' + $shimFile + '"'
$scpArgs = $sshBaseArgs + @($localShim, "${User}@${ServerIp}:/tmp/zion-rpc-shim.server.js")
$scp = Start-Process -FilePath scp -ArgumentList $scpArgs -NoNewWindow -Wait -PassThru
if ($scp.ExitCode -ne 0) { Err "SCP of server.js failed (code $($scp.ExitCode))."; exit 3 }

# Remote command to inject file into the running container and restart it
$envAssign = @{}
if ($RestartPool) { $envAssign['RESTART_POOL'] = '1' } else { $envAssign['RESTART_POOL'] = '0' }
$envStr = ($envAssign.GetEnumerator() | ForEach-Object { $_.Key + '=' + $_.Value }) -join ' '
$cmd = 'bash -lc ' + "'" + $envStr + '; set -euo pipefail; ' +
       'CONT=zion-rpc-shim; SRC=/tmp/zion-rpc-shim.server.js; ' +
  'if [ -d "$SRC" ]; then echo "[hotpatch] SRC is a directory, using $SRC/server.js"; SRC="$SRC/server.js"; fi; ' +
       'if ! docker ps --format "{{.Names}}" | grep -qx "$CONT"; then echo "[hotpatch] container not running: $CONT" >&2; exit 5; fi; ' +
       'echo "[hotpatch] backup original"; docker cp "$CONT:/app/server.js" /tmp/server.js.bak 2>/dev/null || true; ' +
       'echo "[hotpatch] copy new server.js"; docker cp "$SRC" "$CONT:/app/server.js"; ' +
       'echo "[hotpatch] restart shim"; docker restart "$CONT" >/dev/null; ' +
       'echo "[hotpatch] wait for shim"; for i in $(seq 1 30); do curl -sf --max-time 2 http://127.0.0.1:18089/metrics.json >/dev/null && break; sleep 1; done; ' +
       'echo "[hotpatch] metrics:"; curl -sS --max-time 3 http://127.0.0.1:18089/metrics.json || true; ' +
       'echo "[hotpatch] logs:"; docker logs --tail=80 "$CONT" || true; ' +
       'if [ "${RESTART_POOL}" = "1" ]; then if docker ps --format "{{.Names}}" | grep -qx zion-uzi-pool; then echo "[hotpatch] restart pool"; docker restart zion-uzi-pool >/dev/null || true; sleep 2; docker logs --tail=80 zion-uzi-pool || true; fi; fi' + "'"

Info "[ssh] Executing remote hotpatch..."
$sshArgs = $sshBaseArgs + ("${User}@${ServerIp}", $cmd)
$run = Start-Process -FilePath ssh -ArgumentList $sshArgs -NoNewWindow -Wait -PassThru
if ($run.ExitCode -ne 0) { Err "Remote hotpatch failed (code $($run.ExitCode))."; exit 4 }

Info "[done] rpc-shim hotpatch completed."
