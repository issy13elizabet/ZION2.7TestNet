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

# SSH base args (use repo-managed default if not provided)
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

# Upload and execute a remote script to avoid quoting pitfalls
$remoteScript = (@'
#!/usr/bin/env bash
set -euo pipefail
C=(zion-uzi-pool zion-rpc-shim zion-seed1 zion-seed2 zion-redis zion-walletd zion-wallet-adapter)
for name in "${C[@]}"; do
  if docker ps --format '{{.Names}}' | grep -qx "$name"; then
    echo "[remote] restarting $name"; docker restart "$name" >/dev/null || true
  else
    echo "[remote] container not found: $name" >&2
  fi
done

echo "[remote] ps after restart:"
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | { head -n1; grep -E "zion-(seed1|seed2|redis|rpc-shim|uzi-pool|walletd|wallet-adapter)"; } || true

echo "[remote] probe shim metrics:"
curl -sS --max-time 3 http://127.0.0.1:18089/metrics.json || true

echo "[remote] probe pool live_stats:"
curl -sS --max-time 3 http://127.0.0.1:8117/live_stats | head -c 800 || true

echo "[remote] recent logs (shim & pool):"
docker logs --tail=80 zion-rpc-shim 2>/dev/null | sed -e 's/^/[shim] /' || true
docker logs --tail=80 zion-uzi-pool 2>/dev/null | sed -e 's/^/[pool] /' || true
'@).Replace("`r`n","`n")

$tmpFile = New-TemporaryFile
Set-Content -Path $tmpFile -Value $remoteScript -NoNewline -Encoding UTF8

Info "[ssh] Uploading restart script to $User@$ServerIp..."
$scpArgs = $sshBase + @($tmpFile, "${User}@${ServerIp}:/tmp/zion-restart.sh")
$scp = Start-Process -FilePath scp -ArgumentList $scpArgs -NoNewWindow -Wait -PassThru
if ($scp.ExitCode -ne 0) { Err "SCP upload failed ($($scp.ExitCode))"; exit 1 }

Info "[ssh] Executing remote restart script..."
$sshArgs = $sshBase + ("$User@$ServerIp", 'bash -lc ' + "'chmod +x /tmp/zion-restart.sh && bash /tmp/zion-restart.sh'")
$p = Start-Process -FilePath ssh -ArgumentList $sshArgs -NoNewWindow -Wait -PassThru
if ($p.ExitCode -ne 0) { Err "Remote restart failed ($($p.ExitCode))"; exit 1 }

Info "[done] Remote restart complete."
