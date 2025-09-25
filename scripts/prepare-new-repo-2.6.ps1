Param(
  [string]$OutDir = "$env:TEMP\\Zion-v2.6",
  [switch]$Force = $false
)

$ErrorActionPreference = 'Stop'

Write-Host "[prep] Exporting sanitized source tree for v2.6..."

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')

# Paths to exclude (folders or masks)
$excludeDirs = @('.git','Zion-clone','wallets','keys','secrets','node_modules','logs','data','.vscode')
$excludeFiles = @('*.tgz','*.tar.gz','*.zip','.env','*.env')

if (Test-Path $OutDir) {
  if (-not $Force) { throw "Output directory exists: $OutDir. Use -Force to overwrite." }
  Remove-Item -Recurse -Force $OutDir
}
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

Write-Host "[prep] Copying files to $OutDir ..."

# Use robocopy with excludes to handle Windows path quirks
$src = $repoRoot.Path.TrimEnd('\\')
$dst = (Resolve-Path $OutDir).Path

# Build robocopy exclude args
$xd = @()
foreach($d in $excludeDirs){ $xd += @('/XD', (Join-Path $src $d)) }
$xf = @()
foreach($f in $excludeFiles){ $xf += @('/XF', $f) }

# Quote args that contain spaces because Start-Process does not auto-quote array items
function QuoteArg([string]$s){ if ($s -match '\s') { '"' + $s + '"' } else { $s } }

$baseArgs = @($src, $dst, '/MIR', '/NFL','/NDL','/NJH','/NJS','/NP','/XJ')
$allArgs = @()
$allArgs += $baseArgs
$allArgs += $xd
$allArgs += $xf

$argString = ($allArgs | ForEach-Object { QuoteArg $_ }) -join ' '
$p = Start-Process -FilePath robocopy.exe -ArgumentList $argString -NoNewWindow -Wait -PassThru
if ($p.ExitCode -gt 8) { throw "robocopy failed with code $($p.ExitCode)" }

Write-Host "[prep] Done. Next steps:"
Write-Host "  1) Create a new empty GitHub repo (e.g., Maitreya-ZionNet/Zion-v2.6)"
Write-Host "  2) Initialize Git in exported directory:"
Write-Host "     cd $OutDir"
Write-Host "     git init"
Write-Host "     git add ."
Write-Host "     git commit -m 'chore: bootstrap 2.6 from v2.5 sanitized'"
Write-Host "     git remote add origin https://github.com/Maitreya-ZionNet/Zion-v2.6.git"
Write-Host "     git branch -M main"
Write-Host "     git push -u origin main"
Write-Host "  3) Update CI/CD and deployment scripts to point to the new repo URL if needed."
