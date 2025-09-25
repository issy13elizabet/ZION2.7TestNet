[CmdletBinding()]
param(
  [string]$ServerIp,
  [string]$User,
  [string]$KeyPath,
  [switch]$NoRestart,
  [string]$Defaults = ''
)

$ErrorActionPreference = 'Stop'
function Info($m){ Write-Host $m }
function Warn($m){ Write-Host $m -ForegroundColor Yellow }
function Err($m){ Write-Host $m -ForegroundColor Red }

# Load defaults if inputs missing
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

# Resolve local patch file
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$patchFile = Join-Path $repoRoot 'docker\uzi-pool\patch-rx.js'
if (-not (Test-Path $patchFile)) { Err "Local patch file not found: $patchFile"; exit 1 }

# SSH workspace
$workDir = Join-Path $env:LOCALAPPDATA 'Zion\ssh'
if (-not (Test-Path $workDir)) { New-Item -ItemType Directory -Path $workDir -Force | Out-Null }
$knownHosts = Join-Path $workDir 'known_hosts'
if (-not (Test-Path $knownHosts)) { New-Item -ItemType File -Path $knownHosts -Force | Out-Null }
if ([string]::IsNullOrWhiteSpace($KeyPath)) { $KeyPath = Join-Path $workDir 'id_ed25519' }
if (-not (Test-Path $KeyPath)) { Warn "SSH key not found: $KeyPath. Continuing may rely on agent or default key." }
$sshBase = @('-F','NUL')
if (Test-Path $KeyPath) { $sshBase += @('-i', $KeyPath) }
$sshBase += @('-o', "UserKnownHostsFile=$knownHosts", '-o', 'GlobalKnownHostsFile=NUL', '-o','StrictHostKeyChecking=accept-new')

# Test SSH
Info "[ssh] Testing SSH connectivity to $User@$ServerIp..."
$tmpOut = New-TemporaryFile
$p = Start-Process -FilePath ssh -ArgumentList ($sshBase + @('-o','BatchMode=yes',"$User@$ServerIp",'echo SSH_OK')) -NoNewWindow -Wait -PassThru -RedirectStandardOutput $tmpOut
if ($p.ExitCode -ne 0) {
  Warn "[ssh] BatchMode test failed, retrying..."
  $p = Start-Process -FilePath ssh -ArgumentList ($sshBase + @("$User@$ServerIp",'echo SSH_OK')) -NoNewWindow -Wait -PassThru -RedirectStandardOutput $tmpOut
  if ($p.ExitCode -ne 0) { Err "SSH connection failed ($($p.ExitCode))"; exit 2 }
}

# Upload patch
Info "[ssh] Uploading patch-rx.js..."
# Pass raw path to -ArgumentList; PowerShell will quote as needed
$scpArgs = $sshBase + @($patchFile, "${User}@${ServerIp}:/tmp/patch-rx.js")
$scp = Start-Process -FilePath scp -ArgumentList $scpArgs -NoNewWindow -Wait -PassThru
if ($scp.ExitCode -ne 0) { Err "SCP failed ($($scp.ExitCode))"; exit 3 }

# Remote patch apply and optional restart
$envStr = if ($NoRestart) { 'RESTART=0' } else { 'RESTART=1' }
$cmd = 'bash -lc ' + "'" + $envStr + '; set -euo pipefail; ' +
  'CONT=zion-uzi-pool; SRC=/tmp/patch-rx.js; ' +
  'if ! docker ps --format "{{.Names}}" | grep -qx "$CONT"; then echo "[pool-hotpatch] container not running: $CONT" >&2; exit 5; fi; ' +
  'echo "[pool-hotpatch] backup pool.js"; docker cp "$CONT:/app/lib/pool.js" /tmp/pool.js.bak 2>/dev/null || true; ' +
  'echo "[pool-hotpatch] copy patch-rx.js"; docker cp "$SRC" "$CONT:/patch-rx.js"; ' +
  'echo "[pool-hotpatch] apply patch (node /patch-rx.js)"; docker exec "$CONT" node /patch-rx.js || true; ' +
  'if [ "${RESTART}" = "1" ]; then echo "[pool-hotpatch] restart pool"; docker restart "$CONT" >/dev/null || true; sleep 2; fi; ' +
  'echo "[pool-hotpatch] tail logs"; docker logs --tail=120 ' + "$CONT" + ' || true' + "'"

Info "[ssh] Executing remote hotpatch..."
$sshArgs = $sshBase + ("${User}@${ServerIp}", $cmd)
$run = Start-Process -FilePath ssh -ArgumentList $sshArgs -NoNewWindow -Wait -PassThru
if ($run.ExitCode -ne 0) { Err "Remote hotpatch failed ($($run.ExitCode))"; exit 4 }

Info "[done] pool hotpatch completed."
