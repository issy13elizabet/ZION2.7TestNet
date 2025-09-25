param(
    [Parameter(Mandatory=$true)][string]$ServerIp,
    [string]$User = "root",
    [string]$KeyFile = "",
    [string]$SshOptions = "",
    [int]$TargetBlocks = 60,
    [int]$Threads = 4,
    [string]$Address = "",
    [int]$PollSeconds = 5,
    [int]$TimeoutSeconds = 1800
)

# Helper to run a remote command over SSH and return stdout as string
function Invoke-Remote {
    param([string]$Cmd)
    # Determine interactive mode based on SshOptions (BatchMode=no implies interactive/password allowed)
    $isInteractive = $false
    if ($SshOptions -and ($SshOptions -match 'BatchMode\s*=\s*no')) { $isInteractive = $true }

    # Build ssh argument list robustly
    $sshArgs = @()
    # Host key checks disabled to avoid first-connection prompts
    $sshArgs += @('-o','StrictHostKeyChecking=no','-o','UserKnownHostsFile=/dev/null')
    if ($isInteractive) {
        # Allow password prompting and allocate TTY
        $sshArgs += @('-o','BatchMode=no','-tt')
    } else {
        $sshArgs += @('-o','BatchMode=yes')
    }
    if ($SshOptions -and $SshOptions.Trim().Length -gt 0) {
        # Append raw custom options as tokens (split on whitespace)
        $sshArgs += ($SshOptions -split '\s+')
    }
    if ($KeyFile -and (Test-Path $KeyFile)) {
        $sshArgs += @('-i', $KeyFile)
    }
    $sshArgs += ("$User@$ServerIp")
    $sshArgs += $Cmd

    Write-Verbose ("REMOTE ssh args: {0}" -f ($sshArgs -join ' '))

    if ($isInteractive) {
        # In interactive mode, do not suppress stderr; let prompts show up, but still capture stdout
        $out = & ssh @sshArgs | Out-String
        $exit = $LASTEXITCODE
        if ($exit -ne 0) { return $null }
        return $out
    } else {
        # Capture stdout, suppress stderr noise
        $out = & ssh @sshArgs 2>$null | Out-String
        return $out
    }
}

Write-Host 'Bootstrap mining starting...' -ForegroundColor Cyan
Write-Host ("Server: {0} User: {1} TargetBlocks: {2} Threads: {3}" -f $ServerIp,$User,$TargetBlocks,$Threads) -ForegroundColor Cyan

# Resolve pool address if not provided
if (-not $Address -or $Address.Trim().Length -eq 0) {
    $cfgPath = Join-Path $PSScriptRoot "..\adapters\uzi-pool-config\config.json"
    if (-not (Test-Path $cfgPath)) { throw "Pool config not found at $cfgPath. Provide -Address explicitly." }
    $cfg = Get-Content $cfgPath -Raw | ConvertFrom-Json
    $Address = $cfg.poolServer.poolAddress
}
if (-not $Address -or $Address.Trim().Length -lt 10) { throw "Invalid pool address. Provide -Address explicitly." }

Write-Host ("Using miner address: {0}" -f $Address) -ForegroundColor Yellow

# Get current height from daemon (inside seed1 container)
function Get-RemoteHeight {
    $json = Invoke-Remote "docker exec zion-seed1 sh -lc 'curl -s http://127.0.0.1:18081/getheight'"
    if (-not $json) { return $null }
    try {
        $obj = $json | ConvertFrom-Json
        return [int]$obj.height
    } catch {
        return $null
    }
}

$startHeight = Get-RemoteHeight
if ($null -eq $startHeight) { throw "Cannot read remote height from daemon." }
$targetHeight = $startHeight + $TargetBlocks
Write-Host ("Start height: {0} -> Target height: {1}" -f $startHeight,$targetHeight) -ForegroundColor Green

# Helper to safely embed JSON into sh -lc single-quoted string
function Convert-ToShSingleQuote([string]$s) {
    if ($null -eq $s) { return "" }
    return $s -replace "'", "'""'""'"  # close ', insert '"'", reopen '
}

# Start mining via JSON-RPC (base64 payload to avoid quoting issues), then fallback to REST (miner_address)
Write-Host 'Requesting daemon to start mining (JSON-RPC)...' -ForegroundColor Cyan
$startPayload = @{ jsonrpc = "2.0"; id = "0"; method = "start_mining"; params = @{ miner_address = $Address; threads_count = $Threads; do_background_mining = $false; ignore_battery = $true } } | ConvertTo-Json -Compress
$startB64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($startPayload))
$startRpcCmd = "docker exec zion-seed1 sh -lc 'echo $startB64 | base64 -d | curl -s -X POST -H `"Content-Type: application/json`" --data @- http://127.0.0.1:18081/json_rpc'"
$resp = Invoke-Remote $startRpcCmd
if ($resp) { Write-Host ("start_mining RPC response: {0}" -f $resp) -ForegroundColor DarkGray }
if (-not $resp -or $resp -match 'error' -or $resp -match 'Parse error') {
    Write-Host 'RPC failed, trying REST (miner_address)...' -ForegroundColor Yellow
    $startUrl1 = "http://127.0.0.1:18081/start_mining?miner_address=$Address&threads_count=$Threads&do_background_mining=false&ignore_battery=true"
    $uq = Convert-ToShSingleQuote $startUrl1
    $cmd = "docker exec zion-seed1 sh -lc 'curl -s $uq'"
    $resp = Invoke-Remote $cmd
    if ($resp) { Write-Host ("start_mining REST response: {0}" -f $resp) -ForegroundColor DarkGray }
}
Write-Host 'Daemon getinfo snapshot:' -ForegroundColor DarkGray
Invoke-Remote "docker exec zion-seed1 sh -lc 'curl -s http://127.0.0.1:18081/getinfo'" | Write-Host

# Poll until target height or timeout
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ($true) {
    Start-Sleep -Seconds $PollSeconds
    $h = Get-RemoteHeight
    if ($null -eq $h) { Write-Host '[poll] height: unavailable' -ForegroundColor DarkYellow; continue }
    Write-Host ("[poll] height: {0} / {1}" -f $h, $targetHeight)
    if ($h -le $startHeight) {
        # If stuck, re-send JSON-RPC once more
        $r2 = Invoke-Remote $startRpcCmd
        if ($r2) { Write-Host ("re-try start_mining RPC response: {0}" -f $r2) -ForegroundColor DarkGray }
    }
    if ($h -ge $targetHeight) { break }
    if ((Get-Date) -gt $deadline) { Write-Warning "Timeout reached before target height."; break }
}

# Stop mining
Write-Host 'Requesting daemon to stop mining (JSON-RPC, fallback REST)...' -ForegroundColor Cyan
$stopPayload = @{ jsonrpc = "2.0"; id = "0"; method = "stop_mining"; params = @{} } | ConvertTo-Json -Compress
$stopB64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($stopPayload))
$stopRpcCmd = "docker exec zion-seed1 sh -lc 'echo $stopB64 | base64 -d | curl -s -X POST -H `"Content-Type: application/json`" --data @- http://127.0.0.1:18081/json_rpc'"
$resp2 = Invoke-Remote $stopRpcCmd
if (-not $resp2 -or $resp2 -match 'error') {
    $stopCmd = "docker exec zion-seed1 sh -lc 'curl -s http://127.0.0.1:18081/stop_mining'"
    $resp2 = Invoke-Remote $stopCmd
}
if ($resp2) { Write-Host ("stop_mining response: {0}" -f $resp2) -ForegroundColor DarkGray }

$finalHeight = Get-RemoteHeight
Write-Host ("Final height on daemon: {0}" -f $finalHeight) -ForegroundColor Green
Write-Host 'Bootstrap mining script finished.' -ForegroundColor Cyan
