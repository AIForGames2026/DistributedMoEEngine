# Native Windows 11 x86_64 build of the engine (tfswarm.exe, TASK-40), its tray
# launcher (tfswarm-operator.exe, TASK-44) and its PoC programs (dispatch.exe, tfswarm-poc.exe). Runs in Windows PowerShell 5.1
# or PowerShell 7, no admin.
#
#   .\build.ps1             build, run the unit tests, run the PoC checks
#   .\build.ps1 -SkipChecks build and unit-test only
#   .\build.ps1 -Package    also zip bin\ for GitHub Releases into dist\
#   .\build.ps1 -Model M.gguf -Golden G.tfref   also run tfswarm's golden check, local and shm
#   .\build.ps1 -Gpu        build tfswarm.exe with CUDA (needs the CUDA toolkit and Visual
#                           Studio's C++ tools, whose cl.exe nvcc uses) and check its kernels
#
# Uses Zig from PATH if it matches .zig-version, else ~\tools\zig, and
# installs it there (tools\setup\install-zig.ps1, SHA-256 checked) if missing.
#
# Windows covers the PoC check (the two-process MoE over shared memory and
# TCP) and tfswarm's local, shm, expert and router modes on the CPU. io_uring, CUDA, and RDMA steps
# (io-test, gate) are Linux only.
param(
    [switch]$SkipChecks,
    [switch]$Gpu,
    [switch]$Package,
    [string]$Model,
    [string]$Golden,
    [ValidateSet('Debug', 'ReleaseSafe', 'ReleaseFast', 'ReleaseSmall')]
    [string]$Optimize = 'ReleaseFast',
    [int]$Port = 7470
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$version = (Get-Content .zig-version -Raw).Trim()

function Find-Zig {
    $onPath = Get-Command zig -ErrorAction SilentlyContinue
    if ($onPath -and ((& $onPath.Source version) -eq $version)) { return $onPath.Source }
    $local = Join-Path $HOME 'tools\zig\zig.exe'
    if ((Test-Path $local) -and ((& $local version) -eq $version)) { return $local }
    Write-Host "Zig $version not found; installing it into ~\tools\zig"
    & (Join-Path $PSScriptRoot 'tools\setup\install-zig.ps1')
    return $local
}

# The newest MSVC host compiler nvcc accepts (CUDA 13.1 takes MSVC up to 19.49, toolset 14.4x).
function Find-ClDir {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { throw "-Gpu needs Visual Studio's C++ tools (vswhere.exe not found)" }
    $cls = & $vswhere -all -products * -find 'VC\Tools\MSVC\**\bin\Hostx64\x64\cl.exe' |
        Where-Object { $_ -match '\\MSVC\\14\.(\d+)\.' -and [int]$Matches[1] -lt 50 } |
        Sort-Object { [version](($_ -split '\\MSVC\\')[1] -split '\\')[0] } -Descending
    if (-not $cls) { throw "-Gpu needs MSVC 14.4x or older (Visual Studio 2022's C++ tools) for nvcc" }
    return Split-Path @($cls)[0]
}

# Native commands write progress to stderr, so only the exit code means failure.
function Invoke-Step([string]$Name, [scriptblock]$Command) {
    Write-Host ""
    Write-Host "== $Name" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "$Name failed (exit $LASTEXITCODE)" }
}

# Starts the TCP expert, runs the router against it, and requires both to exit 0.
function Invoke-TcpCheck([string]$Name, [string[]]$Extra) {
    Write-Host ""
    Write-Host "== $Name" -ForegroundColor Cyan
    $expertArgs = @('expert', '--port', "$Port") + $Extra
    $expert = Start-Process $dispatch -ArgumentList $expertArgs -NoNewWindow -PassThru
    $null = $expert.Handle  # without this, ExitCode stays empty after exit
    & $dispatch router --host 127.0.0.1 --port $Port @Extra
    $routerExit = $LASTEXITCODE
    if (-not $expert.WaitForExit(15000)) { $expert.Kill(); throw "$Name failed: expert did not exit" }
    if ($routerExit -ne 0 -or $expert.ExitCode -ne 0) {
        throw "$Name failed (router exit $routerExit, expert exit $($expert.ExitCode))"
    }
}

$zig = Find-Zig
Write-Host "Zig: $zig"

# Release binaries target x86_64_v2 (SSE4.2 CRC32C), like the Linux x86_64 build.
$buildArgs = @('-Dtarget=x86_64-windows-gnu', '-Dcpu=x86_64_v2', "-Doptimize=$Optimize")
$gpuFlag = @()
if ($Gpu) {
    $buildArgs += @('-Dgpu=true', "-Dccbin=$(Find-ClDir)")
    $gpuFlag = @('--gpu')
}
Invoke-Step 'Unit tests' { & $zig build test --summary all }
Invoke-Step "Build ($Optimize)" { & $zig build install tfswarm @buildArgs }
$dispatch = Join-Path $PSScriptRoot 'zig-out\bin\dispatch.exe'
$tfswarm = Join-Path $PSScriptRoot 'zig-out\bin\tfswarm.exe'

if (-not $SkipChecks) {
    Invoke-Step 'Dispatch check, shared memory (4,096 + 4,096 packets)' { & $dispatch shm }
    Invoke-Step 'PoC check: two-process MoE vs PyTorch goldens, shared memory' { & $dispatch shm --moe }
    Invoke-TcpCheck 'Dispatch check, TCP loopback' @()
    Invoke-TcpCheck 'PoC check: two-process MoE vs PyTorch goldens, TCP loopback' @('--moe')
    if ($Model -and $Golden) {
        Invoke-Step 'tfswarm local: one process vs the transformers golden' { & $tfswarm local $Model --golden $Golden @gpuFlag }
        Invoke-Step 'tfswarm shm: two processes vs the same golden' { & $tfswarm shm $Model --golden $Golden --payload f32 @gpuFlag }
    }
    if ($Gpu) {
        Invoke-Step 'GPU kernels for every weight format vs the CPU' { & $tfswarm gpu-formats }
    }
}

if ($Package) {
    $tag = (git describe --tags 2>$null)
    if (-not $tag) { $tag = 'dev' }
    $name = "tensorfold-swarm-$tag-windows-x86_64"
    $stage = Join-Path $PSScriptRoot "dist\$name"
    if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
    New-Item -ItemType Directory -Force (Join-Path $stage 'bin') | Out-Null
    Copy-Item zig-out\bin\*.exe (Join-Path $stage 'bin')
    Copy-Item README.md, POC.md, LICENSE $stage
    $zip = "$stage.zip"
    if (Test-Path $zip) { Remove-Item -Force $zip }
    Compress-Archive -Path $stage -DestinationPath $zip
    $hash = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
    "$hash  $name.zip" | Set-Content -Encoding ascii "$zip.sha256"
    Write-Host ""
    Write-Host "Packaged $zip"
    Write-Host "SHA-256  $hash"
}

Write-Host ""
Write-Host "Done. Binaries are in zig-out\bin." -ForegroundColor Green
