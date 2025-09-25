param(
    [Parameter(Mandatory=$true)][string]$ServerIp,
    [Parameter(Mandatory=$true)][string]$User,
    [Parameter(Mandatory=$true)][string]$Address,
    [int]$Threads = 8,
    [int]$TargetBlocks = 5,
    [int]$PollSeconds = 5,
    [int]$TimeoutSeconds = 900,
    [string]$KeyFile
)

${global:ErrorActionPreference} = 'Continue'

Write-Host "Bootstrap SOLO mining via XMRig (daemon RPC) starting..." -ForegroundColor Cyan
Write-Host "Server: $ServerIp User: $User Threads: $Threads TargetBlocks: $TargetBlocks" -ForegroundColor DarkGray
Write-Host "Using miner address: $Address" -ForegroundColor DarkGray

function Invoke-Remote([string]$cmd) {
    $sshArgs = @('-o','StrictHostKeyChecking=no','-o','UserKnownHostsFile=/dev/null','-o','BatchMode=yes')
    if ($KeyFile -and (Test-Path $KeyFile)) { $sshArgs += @('-i', $KeyFile) }
    $sshArgs += @("{0}@{1}" -f $User, $ServerIp, $null) | Where-Object { $_ }
    $sshArgs += $cmd
    Write-Verbose ("REMOTE ssh args: {0}" -f ($sshArgs -join ' '))
    try {
        $out = & ssh @sshArgs 2>&1
        if ($out) { return ($out -join "`n").Trim() } else { return "" }
    } catch {
        Write-Verbose ("REMOTE exception: {0}" -f $_)
        return ""
    }
}

function Convert-ToShSingleQuote([string]$s) {
    if ($null -eq $s) { return "''" }
    return "'" + ($s -replace "'", "'\\''") + "'"
}

function Get-RemoteHeight() {
    $cmd = "docker exec zion-seed1 sh -lc 'curl -s http://127.0.0.1:18081/getheight'"
    $json = Invoke-Remote $cmd
    if (-not $json) { return $null }
    try { return ([int]((ConvertFrom-Json $json).height)) } catch { return $null }
}

# 1) Build XMRig image from Dockerfile.xmrig (idempotent)
Write-Host 'Ensuring XMRig image is available on server...' -ForegroundColor Cyan
# Assumption: repo is cloned at /root/Zion (or ~/Zion) on the server
$builderScript = 'set -e; REPO_DIR=/root/Zion; if [ ! -d "$REPO_DIR" ]; then REPO_DIR="$HOME/Zion"; fi; if [ ! -d "$REPO_DIR" ]; then echo RepoDirMissing; exit 2; fi; docker build -q -f "$REPO_DIR/docker/Dockerfile.xmrig" -t zion-xmrig:solo "$REPO_DIR" >/dev/null && echo built || echo reused'
$remoteBuild = "bash -lc " + (Convert-ToShSingleQuote $builderScript)
$buildOut = Invoke-Remote $remoteBuild
if ($buildOut -match 'RepoDirMissing') { throw "Remote repo directory not found (/root/Zion or ~/Zion)." }

# 2) Start xmrig container in daemon (solo) mode against zion-seed1:18081
Write-Host 'Starting XMRig (daemon RPC) container...' -ForegroundColor Cyan

$confJson = @{
  autosave = $false;
  cpu = @{ enabled = $true };
  opencl = @{ enabled = $false };
  cuda = @{ enabled = $false };
  pools = @(@{
    url = "zion-seed1:18081";
    daemon = $true;
    algo = @("rx/0");
    user = $Address;
    pass = "bootstrap";
    tls = $false;
  })
} | ConvertTo-Json -Depth 6 -Compress

$heredoc = @()
$heredoc += "cat > /tmp/xmrig-config.json <<'JSON'"
$heredoc += $confJson
$heredoc += 'JSON'
$remoteCfg = "bash -lc " + (Convert-ToShSingleQuote ($heredoc -join "; "))
Invoke-Remote $remoteCfg | Out-Null

$runScript = "docker rm -f xmrig-bootstrap >/dev/null 2>&1 || true; docker run -d --name xmrig-bootstrap --network zion-seeds -v /tmp/xmrig-config.json:/app/config.json zion-xmrig:solo xmrig --config=/app/config.json -t $Threads --print-time 30 --retries 0 --retry-pause 1 --donate-level 1"
$runCmd = "bash -lc " + (Convert-ToShSingleQuote $runScript)
$out = Invoke-Remote $runCmd
if ($out) { Write-Host ("xmrig container: {0}" -f $out) -ForegroundColor DarkGray }

# 3) Poll height until target
$startHeight = Get-RemoteHeight
if ($null -eq $startHeight) { $startHeight = 0 }
$targetHeight = $startHeight + [Math]::Max(1, $TargetBlocks)
Write-Host ("Start height: {0} -> Target height: {1}" -f $startHeight, $targetHeight)

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ($true) {
    Start-Sleep -Seconds $PollSeconds
    $h = Get-RemoteHeight
    if ($null -eq $h) { Write-Host '[poll] height: unavailable' -ForegroundColor DarkYellow; continue }
    Write-Host ("[poll] height: {0} / {1}" -f $h, $targetHeight)
    if ($h -ge $targetHeight) { break }
    if ((Get-Date) -gt $deadline) { Write-Warning "Timeout reached before target height."; break }
}

# 4) Stop container
Write-Host 'Stopping XMRig container...' -ForegroundColor Cyan
Invoke-Remote "bash -lc 'docker stop xmrig-bootstrap >/dev/null 2>&1 || true'" | Out-Null
Write-Host 'Done.' -ForegroundColor Green

exit 0
