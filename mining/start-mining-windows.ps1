[CmdletBinding()]
param(
    [string]$PoolHost = "91.98.122.165",
    [int]$PoolPort = 3333,
    [string]$Wallet = "ajmrDtnSJCqchjF3wuiceJRCpAumA3wGLhQjtzf7B9uELFYrbtURHTFabiC8RmVbSkPGznaPhsehxYJvwcSGcwXV495Ytpr7wf",
    [string]$RigId = "windows-ryzen-1",
    [switch]$AddDefenderExclusion,
    [switch]$VerboseXMRig
)

function Write-Section($text) { Write-Host "`n===== $text =====" -ForegroundColor Cyan }
function Write-Warn($text) { Write-Host "[warn] $text" -ForegroundColor Yellow }
function Write-Err($text) { Write-Host "[error] $text" -ForegroundColor Red }
function Test-Admin {
    try { return ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator) } catch { return $false }
}

$root = $PSScriptRoot
if (-not $root) { $root = (Get-Location).Path }
$xmrigDir = Join-Path $root "platforms\windows\xmrig-6.21.3"
$xmrigExe = Join-Path $xmrigDir "xmrig.exe"
$cfgPath  = Join-Path $xmrigDir "config-zion.json"
$zipPath  = Join-Path $root "xmrig-6.21.3-msvc-win64.zip"
$zipUrl   = "https://github.com/xmrig/xmrig/releases/download/v6.21.3/xmrig-6.21.3-msvc-win64.zip"

Write-Section "ZION Mining for Windows 11 (Ryzen)"
Write-Host "Pool     : ${PoolHost}:${PoolPort}"
Write-Host "Wallet   : $Wallet"
Write-Host "Rig ID   : $RigId"
Write-Host "Root dir : $root"

# 1) Prepare directory
if (-not (Test-Path $xmrigDir)) {
    Write-Host "[setup] Creating $xmrigDir"
    New-Item -ItemType Directory -Force -Path $xmrigDir | Out-Null
}

# 2) Ensure XMRig binary exists
if (-not (Test-Path $xmrigExe)) {
    if (-not (Test-Path $zipPath)) {
        Write-Host "[setup] Downloading XMRig from $zipUrl"
        try {
            Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -UseBasicParsing
        } catch {
            Write-Err "Failed to download XMRig ZIP. Download manually to $zipPath"
            exit 1
        }
    }
    Write-Host "[setup] Extracting XMRig ZIP to $xmrigDir"
    try {
        Expand-Archive -Path $zipPath -DestinationPath $xmrigDir -Force
    } catch {
        Write-Err "Failed to extract XMRig ZIP: $($_.Exception.Message)"
        exit 1
    }
}

# 3) Optional: Add Windows Defender exclusion
if ($AddDefenderExclusion) {
    if (-not (Test-Admin)) {
        Write-Warn "Defender exclusion requires admin. Re-run PowerShell as Administrator or omit -AddDefenderExclusion."
    } else {
        try {
            Add-MpPreference -ExclusionPath $xmrigDir -ErrorAction SilentlyContinue
            Write-Host "[ok] Defender exclusion added for $xmrigDir"
        } catch {
            Write-Warn "Could not add Defender exclusion: $($_.Exception.Message)"
        }
    }
}

# 4) Generate config
$poolUrl = "stratum+tcp://${PoolHost}:${PoolPort}"
@{
  autosave = $true
  "donate-level" = 0
  randomx = @{ "init" = -1; "init-avx2" = -1; "mode" = "auto"; "1gb-pages" = $false; rdmsr = $true; wrmsr = $true; cache_qos = $false; numa = $true; scratchpad_prefetch_mode = 1 }
  cpu = @{ enabled = $true; "huge-pages" = $true; "huge-pages-jit" = $false; priority = $null; "memory-pool" = $false; yield = $true; "max-threads-hint" = 100; asm = $true }
    pools = @(@{ algo = "rx/0"; url = $poolUrl; user = $Wallet; pass = $RigId; "rig-id" = $RigId; keepalive = $true; tls = $false })
} | ConvertTo-Json -Depth 5 | Out-File -FilePath $cfgPath -Encoding ASCII
Write-Host "[ok] Config written: $cfgPath"

# 5) Connectivity test
Write-Host "[check] Testing connectivity to ${PoolHost}:${PoolPort}"
try {
    $r = Test-NetConnection -ComputerName $PoolHost -Port $PoolPort -WarningAction SilentlyContinue
    if (-not $r.TcpTestSucceeded) {
        Write-Warn "Could not connect to ${PoolHost}:${PoolPort}. Check firewall/NAT. Proceeding anyway."
    } else { Write-Host "[ok] TCP connect succeeded" }
} catch { Write-Warn "Connectivity test failed: $($_.Exception.Message)" }

# 6) Run XMRig
if (-not (Test-Path $xmrigExe)) { Write-Err "xmrig.exe not found after extraction"; exit 1 }

$argList = @("--config=`"$cfgPath`"")
if ($VerboseXMRig) { $argList += "--verbose" }

Write-Host "[run] Launching XMRig ..."
Start-Process -FilePath $xmrigExe -ArgumentList $argList -WorkingDirectory $xmrigDir -WindowStyle Normal
Write-Host "[ok] XMRig started. Monitor the window for hashrate and shares."
