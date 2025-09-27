# ZION AI Miner - Cross-Platform Build Script
# Usage: build.ps1 [platform] [config]
# Platforms: windows, linux, macos, android, ios, all
# Configs: Debug, Release, RelWithDebInfo

param(
    [string]$Platform = "windows",
    [string]$Config = "Release",
    [switch]$Clean = $false,
    [switch]$Install = $false,
    [switch]$Package = $false,
    [switch]$Test = $false,
    [switch]$Verbose = $false
)

# Colors for output
$ColorReset = "`e[0m"
$ColorRed = "`e[31m"
$ColorGreen = "`e[32m"
$ColorYellow = "`e[33m"
$ColorBlue = "`e[34m"
$ColorMagenta = "`e[35m"
$ColorCyan = "`e[36m"

function Write-ColorOutput {
    param($Color, $Message)
    Write-Host "${Color}${Message}${ColorReset}"
}

function Write-Banner {
    Write-Host ""
    Write-ColorOutput $ColorCyan "╔════════════════════════════════════════════════════════════════╗"
    Write-ColorOutput $ColorCyan "║                    🌟 ZION AI MINER 🌟                        ║"
    Write-ColorOutput $ColorCyan "║                                                                ║"
    Write-ColorOutput $ColorCyan "║              Cross-Platform Build System                      ║"
    Write-ColorOutput $ColorCyan "║                   Copyright © 2025                             ║"
    Write-ColorOutput $ColorCyan "║                  Maitreya-ZionNet                              ║"
    Write-ColorOutput $ColorCyan "╚════════════════════════════════════════════════════════════════╝"
    Write-Host ""
}

function Test-Dependencies {
    Write-ColorOutput $ColorYellow "🔍 Checking build dependencies..."
    
    # Check CMake
    try {
        $cmakeVersion = cmake --version | Select-String "version" | ForEach-Object { $_.Line.Split(' ')[2] }
        Write-ColorOutput $ColorGreen "✅ CMake found: $cmakeVersion"
    }
    catch {
        Write-ColorOutput $ColorRed "❌ CMake not found! Please install CMake 3.16 or higher."
        exit 1
    }
    
    # Check compilers based on platform
    switch ($Platform) {
        "windows" {
            try {
                $msvcVersion = cl 2>&1 | Select-String "Version" | ForEach-Object { $_.Line }
                Write-ColorOutput $ColorGreen "✅ MSVC found: $msvcVersion"
            }
            catch {
                Write-ColorOutput $ColorYellow "⚠️  MSVC not found, trying to find Visual Studio..."
                $vsPath = Get-ChildItem -Path "${env:ProgramFiles(x86)}\Microsoft Visual Studio" -Directory -ErrorAction SilentlyContinue |
                          Get-ChildItem -Directory |
                          Where-Object { $_.Name -match "^(2019|2022)$" } |
                          Select-Object -First 1
                if ($vsPath) {
                    Write-ColorOutput $ColorGreen "✅ Visual Studio found: $($vsPath.FullName)"
                } else {
                    Write-ColorOutput $ColorRed "❌ Visual Studio not found! Please install Visual Studio 2019/2022."
                    exit 1
                }
            }
        }
        "linux" {
            try {
                $gccVersion = gcc --version | Select-Object -First 1
                Write-ColorOutput $ColorGreen "✅ GCC found: $gccVersion"
            }
            catch {
                Write-ColorOutput $ColorRed "❌ GCC not found! Please install build-essential."
                exit 1
            }
        }
        "macos" {
            try {
                $clangVersion = clang --version | Select-Object -First 1
                Write-ColorOutput $ColorGreen "✅ Clang found: $clangVersion"
            }
            catch {
                Write-ColorOutput $ColorRed "❌ Clang not found! Please install Xcode Command Line Tools."
                exit 1
            }
        }
        "android" {
            if (-not $env:ANDROID_NDK_ROOT) {
                Write-ColorOutput $ColorRed "❌ ANDROID_NDK_ROOT not set! Please set Android NDK path."
                exit 1
            }
            Write-ColorOutput $ColorGreen "✅ Android NDK found: $env:ANDROID_NDK_ROOT"
        }
        "ios" {
            if ($IsLinux -or $IsWindows) {
                Write-ColorOutput $ColorRed "❌ iOS builds are only supported on macOS!"
                exit 1
            }
            try {
                $xcodeVersion = xcodebuild -version | Select-Object -First 1
                Write-ColorOutput $ColorGreen "✅ Xcode found: $xcodeVersion"
            }
            catch {
                Write-ColorOutput $ColorRed "❌ Xcode not found! Please install Xcode."
                exit 1
            }
        }
    }
}

function Install-VcpkgDependencies {
    param($TargetTriplet)
    
    Write-ColorOutput $ColorYellow "📦 Installing vcpkg dependencies for $TargetTriplet..."
    
    $vcpkgPath = ".\vcpkg"
    if (-not (Test-Path $vcpkgPath)) {
        Write-ColorOutput $ColorYellow "📥 Cloning vcpkg..."
        git clone https://github.com/Microsoft/vcpkg.git $vcpkgPath
        
        if ($IsWindows) {
            & "$vcpkgPath\bootstrap-vcpkg.bat"
        } else {
            & "$vcpkgPath/bootstrap-vcpkg.sh"
        }
    }
    
    $vcpkgExe = if ($IsWindows) { "$vcpkgPath\vcpkg.exe" } else { "$vcpkgPath/vcpkg" }
    
    # Install required packages
    $packages = @(
        "boost-system:$TargetTriplet",
        "boost-thread:$TargetTriplet",
        "boost-program-options:$TargetTriplet",
        "openssl:$TargetTriplet",
        "nlohmann-json:$TargetTriplet"
    )
    
    foreach ($package in $packages) {
        Write-ColorOutput $ColorBlue "Installing $package..."
        & $vcpkgExe install $package
    }
    
    return "$vcpkgPath\scripts\buildsystems\vcpkg.cmake"
}

function Build-Platform {
    param($PlatformName, $ConfigName)
    
    Write-ColorOutput $ColorMagenta "🔨 Building ZION AI Miner for $PlatformName ($ConfigName)..."
    
    $buildDir = "build\$PlatformName-$ConfigName"
    $sourceDir = "src"
    
    # Clean if requested
    if ($Clean -and (Test-Path $buildDir)) {
        Write-ColorOutput $ColorYellow "🧹 Cleaning build directory..."
        Remove-Item -Recurse -Force $buildDir
    }
    
    # Create build directory
    New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
    Set-Location $buildDir
    
    try {
        # Platform-specific CMake configuration
        $cmakeArgs = @(
            "-DCMAKE_BUILD_TYPE=$ConfigName",
            "-DENABLE_GPU_MINING=ON",
            "-DENABLE_AI_ACCELERATION=ON"
        )
        
        if ($Test) {
            $cmakeArgs += "-DENABLE_TESTING=ON"
        }
        
        switch ($PlatformName) {
            "windows" {
                $triplet = if ([Environment]::Is64BitOperatingSystem) { "x64-windows" } else { "x86-windows" }
                $vcpkgToolchain = Install-VcpkgDependencies $triplet
                
                $cmakeArgs += @(
                    "-G", "Visual Studio 17 2022",
                    "-A", "x64",
                    "-DCMAKE_TOOLCHAIN_FILE=$vcpkgToolchain",
                    "-DVCPKG_TARGET_TRIPLET=$triplet",
                    "-DDESKTOP_BUILD=ON"
                )
            }
            "linux" {
                $triplet = "x64-linux"
                $vcpkgToolchain = Install-VcpkgDependencies $triplet
                
                $cmakeArgs += @(
                    "-G", "Unix Makefiles",
                    "-DCMAKE_TOOLCHAIN_FILE=$vcpkgToolchain",
                    "-DVCPKG_TARGET_TRIPLET=$triplet",
                    "-DDESKTOP_BUILD=ON"
                )
            }
            "macos" {
                $triplet = "x64-osx"
                $vcpkgToolchain = Install-VcpkgDependencies $triplet
                
                $cmakeArgs += @(
                    "-G", "Xcode",
                    "-DCMAKE_TOOLCHAIN_FILE=$vcpkgToolchain",
                    "-DVCPKG_TARGET_TRIPLET=$triplet",
                    "-DDESKTOP_BUILD=ON",
                    "-DCMAKE_OSX_DEPLOYMENT_TARGET=10.15"
                )
            }
            "android" {
                $cmakeArgs += @(
                    "-G", "Ninja",
                    "-DCMAKE_TOOLCHAIN_FILE=$env:ANDROID_NDK_ROOT\build\cmake\android.toolchain.cmake",
                    "-DANDROID_ABI=arm64-v8a",
                    "-DANDROID_PLATFORM=android-21",
                    "-DANDROID_BUILD=ON"
                )
            }
            "ios" {
                $cmakeArgs += @(
                    "-G", "Xcode",
                    "-DCMAKE_TOOLCHAIN_FILE=ios.toolchain.cmake",
                    "-DPLATFORM=OS64",
                    "-DDEPLOYMENT_TARGET=13.0",
                    "-DIOS_BUILD=ON"
                )
            }
        }
        
        # Configure
        Write-ColorOutput $ColorBlue "⚙️  Configuring CMake..."
        if ($Verbose) {
            Write-ColorOutput $ColorBlue "CMake args: $($cmakeArgs -join ' ')"
        }
        
        & cmake $cmakeArgs "..\..\$sourceDir"
        if ($LASTEXITCODE -ne 0) {
            throw "CMake configuration failed"
        }
        
        # Build
        Write-ColorOutput $ColorBlue "🔧 Building..."
        $buildArgs = @("--build", ".", "--config", $ConfigName)
        if ($Verbose) {
            $buildArgs += "--verbose"
        }
        
        & cmake $buildArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Build failed"
        }
        
        # Test
        if ($Test) {
            Write-ColorOutput $ColorBlue "🧪 Running tests..."
            & ctest --build-config $ConfigName --verbose
        }
        
        # Install
        if ($Install) {
            Write-ColorOutput $ColorBlue "📦 Installing..."
            & cmake --install . --config $ConfigName
        }
        
        # Package
        if ($Package) {
            Write-ColorOutput $ColorBlue "📦 Creating package..."
            & cpack -C $ConfigName
        }
        
        Write-ColorOutput $ColorGreen "✅ Build completed successfully for $PlatformName!"
        
    }
    catch {
        Write-ColorOutput $ColorRed "❌ Build failed for ${PlatformName}: $_"
        Set-Location ..\..
        exit 1
    }
    finally {
        Set-Location ..\..
    }
}

function Build-All {
    $platforms = @("windows", "linux", "macos")
    
    foreach ($platform in $platforms) {
        if ($platform -eq "linux" -and $IsWindows) {
            Write-ColorOutput $ColorYellow "⏭️  Skipping Linux build on Windows"
            continue
        }
        if ($platform -eq "macos" -and -not $IsMacOS) {
            Write-ColorOutput $ColorYellow "⏭️  Skipping macOS build on non-Mac"
            continue
        }
        
        Build-Platform $platform $Config
    }
}

# Main execution
Write-Banner

Write-ColorOutput $ColorYellow "🚀 Starting ZION AI Miner build process..."
Write-ColorOutput $ColorBlue "Platform: $Platform"
Write-ColorOutput $ColorBlue "Configuration: $Config"
Write-ColorOutput $ColorBlue "Clean: $Clean"
Write-ColorOutput $ColorBlue "Install: $Install"
Write-ColorOutput $ColorBlue "Package: $Package"
Write-ColorOutput $ColorBlue "Test: $Test"
Write-Host ""

# Validate parameters
$validPlatforms = @("windows", "linux", "macos", "android", "ios", "all")
$validConfigs = @("Debug", "Release", "RelWithDebInfo")

if ($Platform -notin $validPlatforms) {
    Write-ColorOutput $ColorRed "❌ Invalid platform: $Platform"
    Write-ColorOutput $ColorYellow "Valid platforms: $($validPlatforms -join ', ')"
    exit 1
}

if ($Config -notin $validConfigs) {
    Write-ColorOutput $ColorRed "❌ Invalid configuration: $Config"
    Write-ColorOutput $ColorYellow "Valid configurations: $($validConfigs -join ', ')"
    exit 1
}

# Check dependencies
Test-Dependencies

# Execute build
if ($Platform -eq "all") {
    Build-All
} else {
    Build-Platform $Platform $Config
}

Write-ColorOutput $ColorGreen "🎉 Build process completed successfully!"
Write-ColorOutput $ColorCyan "✨ May the Cosmic Harmony be with you! ✨"