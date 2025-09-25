[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$ServerIp,
  [string]$User = "root",
  [switch]$Quiet
)

function W($m){ if(-not $Quiet){ Write-Host $m } }
function E($m){ Write-Host $m -ForegroundColor Red }

# 0) Sanity
if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) { E "OpenSSH client (ssh) not found. Install 'OpenSSH Client' in Optional Features."; exit 1 }

# 1) Paths (use isolated workspace to avoid broken %USERPROFILE%\.ssh)
$workDir = Join-Path $env:LOCALAPPDATA "Zion\ssh"
$null = New-Item -ItemType Directory -Path $workDir -Force -ErrorAction SilentlyContinue
$keyName = "id_ed25519"
$keyPath = Join-Path $workDir $keyName
$pubPath = "$keyPath.pub"
$knownHostsPath = Join-Path $workDir 'known_hosts'
if (-not (Test-Path $knownHostsPath)) { New-Item -ItemType File -Path $knownHostsPath -Force | Out-Null }

# 2) Generate key if missing (use argument array to avoid quoting issues)
if (-not (Test-Path $keyPath)) {
  W "[local] Generating ED25519 key: $keyPath"
  $cmdLine = 'ssh-keygen -q -t ed25519 -C zion-deploy -f ' + '"' + $keyPath + '"' + ' -N ""'
  $p = Start-Process -FilePath cmd.exe -ArgumentList @('/c', $cmdLine) -NoNewWindow -Wait -PassThru
  if ($p.ExitCode -ne 0) { E "ssh-keygen failed with code $($p.ExitCode)"; exit 1 }
}

# 3) Ensure ssh-agent is running
$svc = Get-Service ssh-agent -ErrorAction SilentlyContinue
if (-not $svc) {
  E "ssh-agent service is not installed. Install OpenSSH Authentication Agent in Optional Features."
  exit 1
} else {
  if ($svc.StartType -ne 'Automatic') { Set-Service ssh-agent -StartupType Automatic }
  if ($svc.Status -ne 'Running') { Start-Service ssh-agent }
}

# 4) Add key to ssh-agent
try {
  & ssh-add $keyPath | Out-Null
} catch {
  E "ssh-add failed: $($_.Exception.Message)"; exit 1
}

# 5) Append public key to remote authorized_keys (first login prompts password)
W "[remote] Installing public key to $User@$ServerIp (you may be prompted for password once)"
try {
  $pub = Get-Content -Raw -Path $pubPath
  $cmd = "umask 077; mkdir -p ~/.ssh; touch ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys; grep -qx '""$pub""' ~/.ssh/authorized_keys || echo '""$pub""' >> ~/.ssh/authorized_keys"
  # Use -F NUL and isolated known_hosts to bypass broken default config/known_hosts
  ssh -F NUL -i $keyPath -o UserKnownHostsFile=$knownHostsPath -o GlobalKnownHostsFile=NUL -o StrictHostKeyChecking=accept-new "$User@$ServerIp" $cmd
} catch {
  E "Failed to copy public key: $($_.Exception.Message)"; exit 2
}

# 6) Skip writing to ~/.ssh/config (may be broken). We rely on explicit CLI options.

# 7) Verify passwordless access
W "[check] Verifying key auth..."
$tmpOut = New-TemporaryFile
$sshArgs = @('-F','NUL','-i', $keyPath, '-o', "UserKnownHostsFile=$knownHostsPath", '-o','GlobalKnownHostsFile=NUL','-o','BatchMode=yes',"$User@$ServerIp",'echo SSH_OK')
$proc = Start-Process -FilePath ssh -ArgumentList $sshArgs -NoNewWindow -PassThru -Wait -RedirectStandardOutput $tmpOut.FullName
$out = Get-Content $tmpOut.FullName -ErrorAction SilentlyContinue
Remove-Item $tmpOut -Force -ErrorAction SilentlyContinue
if ($out -match 'SSH_OK') {
  W "[ok] SSH key auth working for $User@$ServerIp"
  exit 0
} else {
  E "[warn] Could not verify key auth. Try: ssh $User@$ServerIp"
  exit 3
}
