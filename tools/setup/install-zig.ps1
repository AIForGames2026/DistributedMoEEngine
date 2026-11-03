# Installs the Zig version pinned in .zig-version into $Prefix\zig (default
# ~/tools/zig, like install-zig.sh), verifying the zip's SHA-256 against
# ziglang.org's index. Windows x86_64, PowerShell 5.1 or newer, no admin.
# The Windows counterpart of install-zig.sh.
param([string]$Prefix = (Join-Path $HOME 'tools'))

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'  # Invoke-WebRequest is very slow with the progress bar
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$repo = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$version = (Get-Content (Join-Path $repo '.zig-version') -Raw).Trim()

$index = Invoke-RestMethod 'https://ziglang.org/download/index.json'
$entry = $index.$version.'x86_64-windows'
if ($null -eq $entry) { throw "ziglang.org has no x86_64-windows build of Zig $version" }

$tmp = Join-Path ([IO.Path]::GetTempPath()) ("zig-" + [Guid]::NewGuid())
New-Item -ItemType Directory $tmp | Out-Null
try {
    $zip = Join-Path $tmp 'zig.zip'
    Write-Host "Downloading $($entry.tarball)"
    Invoke-WebRequest $entry.tarball -OutFile $zip -UseBasicParsing
    $sum = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
    if ($sum -ne $entry.shasum) { throw "SHA-256 mismatch: got $sum, expected $($entry.shasum)" }
    Write-Host "SHA-256 OK"

    Expand-Archive $zip -DestinationPath $tmp
    $unpacked = Get-ChildItem $tmp -Directory | Select-Object -First 1
    $dest = Join-Path $Prefix 'zig'
    New-Item -ItemType Directory -Force $Prefix | Out-Null
    if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
    Move-Item $unpacked.FullName $dest
} finally {
    Remove-Item -Recurse -Force $tmp
}

& (Join-Path $dest 'zig.exe') version
Write-Host "Installed to $dest (build.ps1 finds it there; add it to PATH to use zig directly)"
