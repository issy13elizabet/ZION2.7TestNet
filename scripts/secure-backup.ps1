param(
  [string]$Container = "zion-walletd",
  [string]$Passphrase,
  [string]$Dest
)
if (-not $Passphrase) { throw "-Passphrase is required" }
if (-not $Dest) { $Dest = "logs/runtime/" + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ') + "/vault" }
# Ensure destination directory exists and get absolute path
$destItem = New-Item -ItemType Directory -Force -Path $Dest
$DestAbs = ($destItem).FullName
# Temp workspace
$Tmp = New-Item -ItemType Directory -Force -Path ([System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), [System.Guid]::NewGuid().ToString()))
# Copy entire wallet directory from container to temp (more reliable than wildcards with docker cp)
try {
  docker cp "${Container}:/home/zion/.zion" $Tmp.FullName | Out-Null
} catch {
  throw "Failed to copy /home/zion/.zion from container '${Container}'. Ensure the container is running and path exists."
}
# Zip only the copied .zion folder (avoid placing zip inside the source tree being zipped)
$Src = Join-Path $Tmp.FullName ".zion"
if (-not (Test-Path $Src)) { throw "Source folder not found: $Src" }
$zipPath = Join-Path $Tmp.FullName ("zion-wallet-" + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ') + ".zip")
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory($Src, $zipPath)
# AES-256 encryption
$Out = Join-Path $DestAbs ("zion-wallet-" + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ') + ".zip.aes")
# Derive key using PBKDF2
Add-Type -AssemblyName System.Security
$Salt = New-Object byte[] 16; (New-Object System.Random).NextBytes($Salt)
$Kdf = New-Object System.Security.Cryptography.Rfc2898DeriveBytes($Passphrase, $Salt, 100000)
$Key = $Kdf.GetBytes(32)
$IV = New-Object byte[] 16; (New-Object System.Random).NextBytes($IV)
$Aes = [System.Security.Cryptography.Aes]::Create(); $Aes.KeySize = 256; $Aes.Key = $Key; $Aes.IV = $IV; $Aes.Mode = 'CBC'; $Aes.Padding = 'PKCS7'
$InputStream = [System.IO.File]::OpenRead($zipPath)
$OutStream = [System.IO.File]::Open($Out, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
# Write header: Salt + IV (plaintext) for decryption
$OutStream.Write($Salt,0,$Salt.Length); $OutStream.Write($IV,0,$IV.Length)
$CryptoStream = New-Object System.Security.Cryptography.CryptoStream($OutStream, $Aes.CreateEncryptor(), [System.Security.Cryptography.CryptoStreamMode]::Write)
$InputStream.CopyTo($CryptoStream)
$CryptoStream.FlushFinalBlock(); $CryptoStream.Close(); $InputStream.Close(); $OutStream.Close()
# Clean plaintext
try { Remove-Item -Force -Recurse $Tmp.FullName } catch {}
Write-Host "Encrypted backup created:" $Out
