Param(
  [string]$DestDir = 'logs/archive'
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

try {
  # Ensure destination directory exists
  if (-not (Test-Path $DestDir)) { New-Item -ItemType Directory -Path $DestDir -Force | Out-Null }
  $ts = Get-Date -Format 'yyyyMMdd_HHmmss'
  $zip = Join-Path $DestDir ("Zion_backup_$ts.zip")
  $tmp = Join-Path $env:TEMP ("zion_backup_$ts")

  if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
  New-Item -ItemType Directory -Path $tmp | Out-Null

  # Build exclusion list (absolute paths for robocopy /XD)
  $root = (Resolve-Path '.').Path
  $excludeNames = @('.git','node_modules','logs\archive')
  $excludes = @()
  foreach($name in $excludeNames){ $excludes += (Join-Path $root $name) }

  # Mirror working dir to temp excluding heavy/irrelevant folders
  $xdArgs = @('/MIR','/NFL','/NDL','/NJH','/NJS','/NP') + @('/XD') + $excludes
  & robocopy $root $tmp $xdArgs | Out-Null
  $rc = $LASTEXITCODE
  # Robocopy non-zero codes do not necessarily indicate errors; continue regardless

  if (Test-Path $zip) { Remove-Item $zip -Force }
  Compress-Archive -Path ("$tmp\*") -DestinationPath $zip -Force
  Remove-Item -Recurse -Force $tmp

  Write-Host $zip
} catch {
  Write-Error $_
  exit 1
}
