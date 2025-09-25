[CmdletBinding()]
param(
  [string]$ServerIp,
  [string]$User,
  [string]$KeyPath,
  [string]$Defaults = '',
  [switch]$NoTunnel
)

$ErrorActionPreference = 'Stop'
function Info($m){ Write-Host $m -ForegroundColor Cyan }
function Warn($m){ Write-Host $m -ForegroundColor Yellow }
function Err($m){ Write-Host $m -ForegroundColor Red }

# 0) Load defaults if needed
try {
  if (-not $ServerIp -or -not $User -or -not $KeyPath) {
    $json = & (Join-Path $PSScriptRoot 'ssh-defaults.ps1') -DefaultsPath $Defaults | ConvertFrom-Json
    if (-not $ServerIp) { $ServerIp = $json.ServerIp }
    if (-not $User)     { $User     = $json.User }
    if (-not $KeyPath)  { $KeyPath  = $json.KeyPath }
  }
} catch { Warn "ssh-defaults not loaded: $($_.Exception.Message)" }

# 1) Ensure pool container is running (best-effort)
try {
  & (Join-Path $PSScriptRoot 'ssh-start-pool.ps1') -ServerIp $ServerIp -User $User -KeyPath $KeyPath -Defaults $Defaults | Write-Host
} catch { Warn "ssh-start-pool.ps1 failed: $($_.Exception.Message)" }

# 2) Restart services and print health
try {
  & (Join-Path $PSScriptRoot 'ssh-restart-services.ps1') -ServerIp $ServerIp -User $User -KeyPath $KeyPath -Defaults $Defaults | Write-Host
} catch { Warn "ssh-restart-services.ps1 failed: $($_.Exception.Message)" }

# 3) Start SSH tunnel (3333/18089/8117)
if (-not $NoTunnel) {
  try {
    & (Join-Path $PSScriptRoot 'ssh-tunnel-maitreya.ps1') -Action start -ServerIp $ServerIp -User $User -KeyPath $KeyPath -Defaults $Defaults | Write-Host
  } catch { Warn "ssh-tunnel-maitreya.ps1 failed: $($_.Exception.Message)" }
}

# 4) Start miner against localhost:3333
Info "Starting miner via scripts/start-mining.ps1 against 127.0.0.1:3333 ..."
& (Join-Path $PSScriptRoot 'start-mining.ps1') -ServerIp 127.0.0.1 -Ports 3333
