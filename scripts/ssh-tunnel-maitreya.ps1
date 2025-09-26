Param(
  [ValidateSet('start','stop')][string]$Action = 'start',
  [string]$ServerIp,
  [string]$User,
  [string]$KeyPath,
  [string]$Defaults = ''
)

$ErrorActionPreference = 'Stop'

function Get-Defaults {
  param([string]$DefaultsPath)
  try {
    return & (Join-Path $PSScriptRoot 'ssh-defaults.ps1') -DefaultsPath $DefaultsPath | ConvertFrom-Json
  } catch {
    return $null
  }
}

# Apply defaults if params missing
if (-not $ServerIp -or -not $User -or -not $KeyPath) {
  $d = Get-Defaults -DefaultsPath $Defaults
  if ($d) {
    if (-not $ServerIp) { $ServerIp = $d.ServerIp }
    if (-not $User)     { $User     = $d.User }
    if (-not $KeyPath)  { $KeyPath  = $d.KeyPath }
  }
}

# SSH env
$work = Join-Path $env:LOCALAPPDATA 'Zion\ssh'
if (-not (Test-Path $work)) { New-Item -ItemType Directory -Path $work -Force | Out-Null }
if ([string]::IsNullOrWhiteSpace($KeyPath)) { $KeyPath = Join-Path $work 'id_ed25519' }
$kh = Join-Path $work 'known_hosts'
if (-not (Test-Path $kh)) { New-Item -ItemType File -Path $kh -Force | Out-Null }
if (-not (Test-Path $KeyPath)) { Write-Host "[tunnel] SSH key not found at $KeyPath; relying on agent/default" -ForegroundColor Yellow }

function Start-Tunnel {
  $sshArgs = @('-F','NUL')
  if (Test-Path $KeyPath) { $sshArgs += @('-i', $KeyPath) }
  $sshArgs += @('-o',"UserKnownHostsFile=$kh",'-o','GlobalKnownHostsFile=NUL','-o','StrictHostKeyChecking=accept-new','-N','-L','127.0.0.1:3333:127.0.0.1:3333','-L','127.0.0.1:18089:127.0.0.1:18089','-L','127.0.0.1:8117:127.0.0.1:8117',"$User@$ServerIp")
  Write-Host "[tunnel] Starting SSH tunnels -> $User@$ServerIp (3333, 18089, 8117)"
  Start-Process -FilePath ssh -ArgumentList $sshArgs -WindowStyle Hidden | Out-Null
  Start-Sleep -Seconds 2
  Test-NetConnection -ComputerName 127.0.0.1 -Port 3333 | Out-Null
  Test-NetConnection -ComputerName 127.0.0.1 -Port 18089 | Out-Null
  Test-NetConnection -ComputerName 127.0.0.1 -Port 8117  | Out-Null
  Write-Host "[tunnel] Ready: 127.0.0.1:3333, :18089, :8117"
}

function Stop-Tunnel {
  Write-Host "[tunnel] Stopping tunnels (best-effort)"
  Get-Process ssh -ErrorAction SilentlyContinue | Where-Object { $_.Path -like '*\\ssh.exe' } | Stop-Process -Force -ErrorAction SilentlyContinue
}

if ($Action -eq 'start') { Start-Tunnel } else { Stop-Tunnel }
