param(
  [string]$Branch = "main",
  [switch]$HardReset,
  [switch]$RestartServices,
  [switch]$Watch,
  [int]$IntervalSeconds = 60
)

$ErrorActionPreference = 'Stop'

function Invoke-Resync {
  Write-Host "[resync] Fetching from origin..." -ForegroundColor Cyan
  git fetch origin $Branch

  $local = (git rev-parse $Branch).Trim()
  $remote = (git rev-parse origin/$Branch).Trim()
  $base = (git merge-base $Branch origin/$Branch).Trim()

  if ($local -eq $remote) {
    Write-Host "[resync] Up to date ($Branch)" -ForegroundColor Green
    return $false
  }

  if ($local -eq $base) {
    Write-Host "[resync] Need pull (fast-forward)" -ForegroundColor Yellow
    git pull --ff-only origin $Branch
  } elseif ($remote -eq $base) {
    Write-Host "[resync] Local ahead of remote, no action" -ForegroundColor Yellow
  } else {
    if ($HardReset) {
      Write-Host "[resync] Divergence detected, hard reset to origin/$Branch" -ForegroundColor Red
      git reset --hard origin/$Branch
    } else {
      Write-Host "[resync] Divergence detected, skipping (use -HardReset to force)" -ForegroundColor Red
      return $false
    }
  }

  if ($RestartServices) {
    Write-Host "[resync] Restarting services via docker compose..." -ForegroundColor Cyan
    docker compose -f "$(Resolve-Path ./docker/compose.pool-seeds.yml)" up -d
  }
  return $true
}

if ($Watch) {
  Write-Host "[resync] Watch mode every $IntervalSeconds s on branch $Branch" -ForegroundColor Magenta
  while ($true) {
    try { Invoke-Resync | Out-Null } catch { Write-Warning $_ }
    Start-Sleep -Seconds $IntervalSeconds
  }
} else {
  Invoke-Resync | Out-Null
}
