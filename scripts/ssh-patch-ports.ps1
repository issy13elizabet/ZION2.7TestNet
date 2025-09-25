[CmdletBinding()]
Param(
  [Parameter(Mandatory=$true)][string]$ServerIp,
  [string]$User = 'root',
  # Optional path to SSH private key (e.g. repo-stored key)
  [string]$KeyPath,
  # Also publish HTTP (80) in addition to HTTPS (443)
  [switch]$AlsoHttp
)

$ErrorActionPreference = 'Stop'
function Info($m){ Write-Host $m }
function Err($m){ Write-Host $m -ForegroundColor Red }

# SSH base
$workDir = Join-Path $env:LOCALAPPDATA 'Zion\ssh'
if ([string]::IsNullOrWhiteSpace($KeyPath)) {
  $keyPath = Join-Path $workDir 'id_ed25519'
} else {
  $keyPath = $KeyPath
}
$knownHosts = Join-Path $workDir 'known_hosts'
if (-not (Test-Path $workDir)) { New-Item -ItemType Directory -Path $workDir -Force | Out-Null }
if (-not (Test-Path $knownHosts)) { New-Item -ItemType File -Path $knownHosts -Force | Out-Null }
if (-not (Test-Path $keyPath)) { Err "SSH key not found: $keyPath."; exit 1 }
$sshBaseArgs = @('-F','NUL','-i', $keyPath, '-o', "UserKnownHostsFile=$knownHosts", '-o', 'GlobalKnownHostsFile=NUL', '-o','StrictHostKeyChecking=accept-new')

# Create remote script content and upload via scp to avoid quoting issues
$remoteScript = @'
#!/bin/sh
set -eu
cd /opt/zion/Zion
mkdir -p docker
OVR=docker/compose.override-ports.yml

# Detect if host ports are already in use
busy_443=0
busy_80=0
if ss -ltn | awk '{print $4}' | grep -qE '(^|:)443$'; then busy_443=1; fi
if ss -ltn | awk '{print $4}' | grep -qE '(^|:)80$'; then busy_80=1; fi

echo "[remote] Port 443 busy=$busy_443, Port 80 busy=$busy_80"

# Build ports mapping dynamically
{
  printf '%s\n' 'services:'
  printf '%s\n' '  uzi-pool:'
  printf '%s\n' '    ports:'
  printf '%s\n' '      - "3333:3333"'
  if [ "$busy_443" -eq 0 ]; then
    printf '%s\n' '      - "443:3333"'
  fi
} > "$OVR"

if [ "${INCLUDE_HTTP:-0}" = "1" ] && [ "$busy_80" -eq 0 ]; then
  printf '%s\n' '      - "80:3333"' >> "$OVR"
fi

if docker compose -f docker/compose.pool-seeds.yml -f "$OVR" config >/dev/null 2>&1; then
  docker compose -f docker/compose.pool-seeds.yml -f "$OVR" up -d --no-deps uzi-pool
else
  echo "[remote] ERROR: override config invalid" >&2
  exit 21
fi

if command -v ufw >/dev/null 2>&1; then
  ufw allow 443/tcp || true
  ufw allow 80/tcp || true
  ufw status || true
fi

echo "[remote] Listening (post-change):"
ss -tulpn | grep -E ':3333|:443|:80' || true
'@

$tmp = New-TemporaryFile
# Force LF endings to avoid /bin/sh parsing issues
$remoteScript = $remoteScript -replace "`r", ""
Set-Content -Path $tmp -Value $remoteScript -NoNewline -Encoding UTF8

Info "[ssh] Uploading remote patch script..."
$scpArgs = $sshBaseArgs + @($tmp, "${User}@${ServerIp}:/tmp/zion-patch-ports.sh")
$scp = Start-Process -FilePath scp -ArgumentList $scpArgs -NoNewWindow -Wait -PassThru
if ($scp.ExitCode -ne 0) { Err "SCP failed (code $($scp.ExitCode))."; exit 2 }

if ($AlsoHttp) { $include = '1' } else { $include = '0' }
Info "[ssh] Running remote patch (INCLUDE_HTTP=$include)..."
$cmd = "sh -lc 'INCLUDE_HTTP=$include sh /tmp/zion-patch-ports.sh'"
$sshArgs = $sshBaseArgs + ("${User}@${ServerIp}", $cmd)
$rc = Start-Process -FilePath ssh -ArgumentList $sshArgs -NoNewWindow -Wait -PassThru
if ($rc.ExitCode -ne 0) { Err "Remote port patch failed (code $($rc.ExitCode))."; exit 2 }

Info "[done] Ports patched. Test connectivity:"
Info "       Test-NetConnection -ComputerName $ServerIp -Port 443"
if ($AlsoHttp) { Info "       Test-NetConnection -ComputerName $ServerIp -Port 80" }
