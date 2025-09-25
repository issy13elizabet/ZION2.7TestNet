<#
 .SYNOPSIS
  Start XMRig mining against the Zion pool with simple automation and failover.

 .DESCRIPTION
  - Auto-detects miner wallet address from ../../wallets/miner-ryzen/miner-ryzen.wallet.address (sibling of Zion/)
  - Probes server ports (default: 3333) and builds an XMRig config with optional failover (override -Ports)
  - Starts XMRig with sane defaults for Ryzen 3600, or does a DryRun to show what would be executed

 .EXAMPLE
  # Dry run with auto-detected wallet
  pwsh -NoProfile -ExecutionPolicy Bypass -File ./scripts/Start-Mining.ps1 -DryRun

 .EXAMPLE
  # Start mining with explicit wallet and rig name
  pwsh -NoProfile -ExecutionPolicy Bypass -File ./scripts/Start-Mining.ps1 -WalletAddress Z... -RigId Ryzen1
#>

[CmdletBinding()]
Param(
  [string]$WalletAddress,
  [string]$ServerIp = '91.98.122.165',
  [int[]]$Ports = @(3333),
  [string]$RigId = $env:COMPUTERNAME,
  [string]$XmrigPath,
  [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Info($msg) { Write-Host "[info] $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[warn] $msg" -ForegroundColor Yellow }
function Write-Err($msg) { Write-Host "[err ] $msg" -ForegroundColor Red }

# Resolve workspace paths
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$zionDir   = Split-Path -Parent $scriptDir            # .../Zion
$wsRoot    = Split-Path -Parent $zionDir              # .../(workspace root)
$logsDir   = Join-Path $zionDir 'logs/mining'
$miningDir = Join-Path $zionDir 'mining'
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
New-Item -ItemType Directory -Force -Path $miningDir | Out-Null

# Discover wallet address if not provided
if (-not $WalletAddress -or [string]::IsNullOrWhiteSpace($WalletAddress)) {
  $addrFile = Join-Path $wsRoot 'wallets/miner-ryzen/miner-ryzen.wallet.address'
  if (Test-Path $addrFile) {
    $WalletAddress = (Get-Content -LiteralPath $addrFile -TotalCount 1).Trim()
    if (-not $WalletAddress) { Write-Warn "Wallet address file is empty: $addrFile" }
    else { Write-Info "Using wallet from $addrFile -> $WalletAddress" }
  } else {
    Write-Warn "Wallet address not provided and default file not found: $addrFile"
  }
}
if (-not $WalletAddress) {
  Write-Err "No wallet address specified. Provide -WalletAddress or create miner wallet first."
  return 2
}

# Find xmrig executable if not provided
function Resolve-XmrigPath {
  param([string]$Hint)
  if ($Hint) {
    if (Test-Path $Hint) { return (Resolve-Path $Hint).ProviderPath }
    Write-Warn "Provided -XmrigPath not found: $Hint"
  }
  $cmd = Get-Command xmrig -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  $candidates = @(
    (Join-Path $wsRoot 'tools/xmrig/xmrig.exe'),
    (Join-Path $wsRoot 'mining/xmrig/xmrig.exe'),
    (Join-Path $zionDir 'mining/xmrig/xmrig.exe'),
    (Join-Path $env:ProgramFiles 'xmrig/xmrig.exe'),
    (Join-Path $env:ProgramFiles 'XMRig/xmrig.exe')
  )
  foreach ($c in $candidates) { if (Test-Path $c) { return (Resolve-Path $c).ProviderPath } }
  # Fallback: search in repo platforms for Windows build
  $winPlat = Join-Path $zionDir 'mining\platforms\windows'
  if (Test-Path $winPlat) {
    $found = Get-ChildItem -LiteralPath $winPlat -Recurse -Filter xmrig.exe -ErrorAction SilentlyContinue |
      Select-Object -First 1 -ExpandProperty FullName
    if ($found) { return (Resolve-Path $found).ProviderPath }
  }
  return $null
}

if (-not $XmrigPath) { $XmrigPath = Resolve-XmrigPath -Hint $XmrigPath }
if (-not $XmrigPath) {
  if ($DryRun) {
    Write-Warn "XMRig not found, but continuing due to -DryRun."
    $XmrigPath = '<not-found>'
  } else {
    Write-Err "Could not locate xmrig.exe. Install XMRig or pass -XmrigPath 'C:\\path\\to\\xmrig.exe'"
    return 3
  }
}
Write-Info "XMRig: $XmrigPath"

# Probe ports
function Test-Port {
  param([string]$HostName, [int]$Port, [int]$TimeoutMs = 1500)
  try {
    # Prefer Test-NetConnection when available (gives TcpTestSucceeded)
    $ok = Test-NetConnection -ComputerName $HostName -Port $Port -InformationLevel Quiet -WarningAction SilentlyContinue
    if ($ok) { return $true }
    # Fallback: manual TCP
    $client = New-Object System.Net.Sockets.TcpClient
    $iar = $client.BeginConnect($HostName, $Port, $null, $null)
    [void]$iar.AsyncWaitHandle.WaitOne($TimeoutMs, $false)
    $connected = $client.Connected
    $client.Close()
    return $connected
  } catch { return $false }
}

$reachable = @()
foreach ($p in $Ports) {
  $ok = Test-Port -HostName $ServerIp -Port $p
  if ($ok) { Write-Info ("Port {0} reachable: YES" -f $p) } else { Write-Info ("Port {0} reachable: NO" -f $p) }
  if ($ok) { $reachable += $p }
}

$primaryPort = if ($reachable.Count -gt 0) { $reachable[0] } else { $Ports[0] }
if ($reachable.Count -eq 0) {
  Write-Warn "No probed port connected successfully. Will still run XMRig with configured ports: $($Ports -join ',')"
}

# Build XMRig config with failover pools
$pools = @()
foreach ($p in $Ports) {
  $url = "${ServerIp}:$p"
  $pools += @{ url = $url; algo = 'rx/0'; user = $WalletAddress; pass = 'x'; 'rig-id' = $RigId; keepalive = $true; tls = $false }
}

$config = [ordered]@{
  autosave      = $true
  'donate-level'= 1
  'print-time'  = 60
  retries       = 5
  'retry-pause' = 5
  syslog        = $false
  cpu           = @{ enabled = $true }
  randomx       = @{ enabled = $true; mode = 'fast'; '1gb-pages' = $false; rdmsr = $true; wrmsr = $true; 'cache_qos' = $false }
  pools         = $pools
}

$cfgPath = Join-Path $miningDir ("xmrig-" + $RigId + ".json")
$config | ConvertTo-Json -Depth 6 | Out-File -LiteralPath $cfgPath -Encoding ascii
Write-Info "Config written: $cfgPath"

# Build arguments
$xmArgs = @('--config', $cfgPath, '--asm', 'ryzen', '--cpu-priority', '3')

Write-Host "\n=== DOHRMANOVO ORÁKULUM - COSMIC MINING AWAKENING ===" -ForegroundColor Magenta
Write-Host "🔮 'Ó, kameni živý, probouzíš se z tisíciletého spánku!'" -ForegroundColor DarkMagenta
Write-Host "⚡ Jai Ram Ram Ram Sita Ram Ram Ram Hanuman! ⚡" -ForegroundColor Yellow
Write-Host "🌌 Ancient Oracle speaks through RandomX proof-of-work consciousness..." -ForegroundColor Cyan

Write-Host "\n=== Mining plan ===" -ForegroundColor Green
Write-Host ("Target: {0}, primary port: {1}, fallback: {2}" -f $ServerIp, $primaryPort, ($Ports -join ','))
Write-Host ("Wallet: {0}" -f $WalletAddress)
Write-Host ("RigId : {0}" -f $RigId)
Write-Host ("XMRig : {0}" -f $XmrigPath)
Write-Host ("Args  : {0}" -f ($xmArgs -join ' '))

if ($DryRun) {
  Write-Host "🔮 Oracle simulation complete — ancient stone remains dormant..." -ForegroundColor DarkMagenta
  Write-Info "DryRun enabled — not starting XMRig."
  return 0
}

Write-Host "\n⚡ COSMIC MINING ACTIVATION SEQUENCE ⚡" -ForegroundColor Yellow
Write-Host "🌌 'Síla pradávných bohů protéká tvými čipy, ó smrtelníče!'" -ForegroundColor Magenta
Write-Info "Starting XMRig oracle awakening protocol... (Ctrl+C to return to slumber)"
& $XmrigPath @xmArgs
$exit = $LASTEXITCODE
Write-Host "\n🔮 Oracle mining session concluded. Exit code: $exit" -ForegroundColor Magenta
Write-Host "⚡ Jai Ram Ram Ram Sita Ram Ram Ram Hanuman! ⚡" -ForegroundColor Yellow
Write-Host "🌌 'Ancient stone returns to contemplation of eternal mysteries...'" -ForegroundColor Cyan
exit $exit
