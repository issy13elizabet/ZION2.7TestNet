Param(
  [string]$ServerIp,
  [string]$User,
  [string]$KeyPath,
  [string]$Defaults = ''
)

$ErrorActionPreference = 'Stop'
function Info($m){ Write-Host $m }
function Err($m){ Write-Host $m -ForegroundColor Red }

# Load defaults if needed
try {
  if (-not $ServerIp -or -not $User -or -not $KeyPath) {
    $json = & (Join-Path $PSScriptRoot 'ssh-defaults.ps1') -DefaultsPath $Defaults | ConvertFrom-Json
    if (-not $ServerIp) { $ServerIp = $json.ServerIp }
    if (-not $User)     { $User     = $json.User }
    if (-not $KeyPath)  { $KeyPath  = $json.KeyPath }
  }
} catch {
  Write-Host "[warn] ssh-defaults not loaded: $($_.Exception.Message)" -ForegroundColor Yellow
}

$workDir = Join-Path $env:LOCALAPPDATA 'Zion\ssh'
if (-not (Test-Path $workDir)) { New-Item -ItemType Directory -Path $workDir -Force | Out-Null }
if ([string]::IsNullOrWhiteSpace($KeyPath)) { $KeyPath = Join-Path $workDir 'id_ed25519' }
$knownHosts = Join-Path $workDir 'known_hosts'
if (-not (Test-Path $knownHosts)) { New-Item -ItemType File -Path $knownHosts -Force | Out-Null }

$sshBaseArgs = @('-F','NUL')
if (Test-Path $KeyPath) { $sshBaseArgs += @('-i', $KeyPath) } else { Write-Host "[warn] SSH key not found at $KeyPath; relying on agent/default" -ForegroundColor Yellow }
$sshBaseArgs += @('-o', "UserKnownHostsFile=$knownHosts", '-o', 'GlobalKnownHostsFile=NUL', '-o','StrictHostKeyChecking=accept-new')

# Build compact bash probe to avoid CRLF/quoting pitfalls
# Prepare remote script content (Unix newlines)
$probe = (
@'
#!/bin/bash
set -e
echo "[probe] docker ps (core):"
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | (head -n1; grep -E '^zion-(seed1|seed2|rpc-shim|uzi-pool|redis)\b' || true)
echo
echo "[probe] docker ps -a (core):"
docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | (head -n1; grep -E '^zion-(seed1|seed2|rpc-shim|uzi-pool|redis)\b' || true)
echo
for s in zion-seed1 zion-seed2; do
  echo "[probe] inspect $s:"
  docker inspect -f '{{.Name}} state={{.State.Status}} health={{if .State.Health}}{{.State.Health.Status}}{{else}}n/a{{end}} restartCount={{.RestartCount}}' "$s" 2>/dev/null || true
  echo "[probe] pgrep ziond in $s:"
  docker exec "$s" sh -lc 'pgrep -a ziond || ps aux | grep ziond | grep -v grep || true' 2>&1 || true
  echo
done
echo
echo "[probe] shim metrics:"
curl -s http://localhost:18089/metrics.json || echo '(shim not responding)'
echo
echo "[probe] DNS inside rpc-shim: getent hosts zion-seed1 / seed1:"
docker exec zion-rpc-shim sh -lc 'getent hosts zion-seed1 || echo "no zion-seed1"; getent hosts seed1 || echo "no seed1"' 2>&1 || true
echo
echo "[probe] curl seed RPC from shim container (json_rpc get_height):"
docker exec zion-rpc-shim sh -lc 'apk add --no-cache curl >/dev/null 2>&1 || true; curl -s http://zion-seed1:18081/getheight || curl -s http://seed1:18081/getheight || echo "(curl failed)"' 2>&1 || true
echo
echo "[probe] direct JSON-RPC submit method existence check (seed1):"
docker exec zion-seed1 sh -lc "curl -s -X POST -H 'Content-Type: application/json' --data '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"submitblock\",\"params\":[\"00\"]}' http://127.0.0.1:18081/json_rpc || true" 2>&1 || true
echo
docker exec zion-seed1 sh -lc "curl -s -X POST -H 'Content-Type: application/json' --data '{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"submit_block\",\"params\":[\"00\"]}' http://127.0.0.1:18081/json_rpc || true" 2>&1 || true
echo
docker exec zion-seed1 sh -lc "curl -s -X POST -H 'Content-Type: application/json' --data '{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"submit_raw_block\",\"params\":[\"00\"]}' http://127.0.0.1:18081/json_rpc || true" 2>&1 || true
echo
echo "[probe] seed1 last 40 lines:"
docker logs --tail=40 zion-seed1 2>&1 || true
echo
echo "[probe] seed2 last 40 lines:"
docker logs --tail=40 zion-seed2 2>&1 || true
echo
echo "[probe] rpc-shim last 40 lines:"
docker logs --tail=40 zion-rpc-shim 2>&1 || true
echo
echo "[probe] uzi-pool last 40 lines:"
docker logs --tail=40 zion-uzi-pool 2>&1 || true
'@
).Replace("`r`n","`n")

# Write temp and upload via scp
$tmp = New-TemporaryFile
Set-Content -Path $tmp -Value $probe -NoNewline -Encoding UTF8
$scpArgs = $sshBaseArgs + @($tmp, "${User}@${ServerIp}:/tmp/zion-probe.sh")
$scp = Start-Process -FilePath scp -ArgumentList $scpArgs -NoNewWindow -Wait -PassThru
if ($scp.ExitCode -ne 0) { Err "SCP probe upload failed (code $($scp.ExitCode))."; exit 2 }

# Run the probe script
$cmd = 'bash -lc ' + "'chmod +x /tmp/zion-probe.sh; sed -i '" + '"' + 's/\r$//' + '"' + "' /tmp/zion-probe.sh; bash /tmp/zion-probe.sh'"
$sshArgs = $sshBaseArgs + ("${User}@${ServerIp}", $cmd)
$proc = Start-Process -FilePath ssh -ArgumentList $sshArgs -NoNewWindow -Wait -PassThru
if ($proc.ExitCode -ne 0) { Err "Remote probe failed (code $($proc.ExitCode))."; exit 2 }
