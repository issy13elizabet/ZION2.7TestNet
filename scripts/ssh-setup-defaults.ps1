[CmdletBinding()]
param(
  [string]$ServerIp = '91.98.122.165',
  [string]$User = 'root',
  [string]$KeyPath = ''
)

$ErrorActionPreference = 'Stop'
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$defaultsPath = Join-Path $PSScriptRoot 'ssh-defaults.json'

# If KeyPath not specified, try repo-managed path
if ([string]::IsNullOrWhiteSpace($KeyPath)) {
  $workDir = Join-Path $env:LOCALAPPDATA 'Zion\ssh'
  $tryKey = Join-Path $workDir 'id_ed25519'
  if (Test-Path $tryKey) { $KeyPath = $tryKey } else { $KeyPath = 'C:\\Users\\you\\.ssh\\id_ed25519' }
}

$cfg = [ordered]@{ ServerIp = $ServerIp; User = $User; KeyPath = $KeyPath }
$cfg | ConvertTo-Json | Set-Content -Path $defaultsPath -Encoding UTF8

Write-Host "Vytvořeno: $defaultsPath"
Write-Host "Obsah:"; Get-Content -Path $defaultsPath
