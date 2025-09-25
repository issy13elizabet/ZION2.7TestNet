[CmdletBinding()]
param(
  [string]$DefaultsPath = ''
)

$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($DefaultsPath)) {
  $DefaultsPath = Join-Path $PSScriptRoot 'ssh-defaults.json'
}
if (-not (Test-Path $DefaultsPath)) {
  # fallback to sample for info
  $sample = Join-Path $PSScriptRoot 'ssh-defaults.sample.json'
  if (Test-Path $sample) {
    throw "SSH defaults not found. Create '$DefaultsPath' based on '$sample' and fill ServerIp/User/KeyPath."
  } else {
    throw "SSH defaults not found: $DefaultsPath"
  }
}

try {
  $json = Get-Content -Raw -Path $DefaultsPath | ConvertFrom-Json
} catch {
  $msg = $_.Exception.Message
  throw ("Invalid JSON in {0}: {1}" -f $DefaultsPath, $msg)
}

$out = [ordered]@{
  ServerIp = [string]$json.ServerIp
  User     = [string]$json.User
  KeyPath  = [string]$json.KeyPath
}

if (-not $out.ServerIp) { throw "ssh-defaults: missing ServerIp" }
if (-not $out.User)     { $out.User = 'root' }
if (-not $out.KeyPath)  { $out.KeyPath = '' }

$out | ConvertTo-Json -Compress
