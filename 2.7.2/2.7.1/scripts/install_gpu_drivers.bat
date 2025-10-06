@echo off
REM ZION GPU Driver Installation Script
REM Supports AMD RX 5600 XT and NVIDIA CUDA

echo ============================================
echo ZION AI GPU Driver Setup
echo AMD RX 5600 XT + NVIDIA CUDA Support
echo ============================================

REM Check for AMD GPU
echo Checking for AMD GPU...
wmic path win32_VideoController get name | findstr /i "radeon" >nul
if %errorlevel% equ 0 (
    echo AMD GPU detected - installing AMD drivers...
    goto :amd_setup
)

REM Check for NVIDIA GPU
echo Checking for NVIDIA GPU...
nvidia-smi --query-gpu=name --format=csv,noheader,nounits >nul 2>&1
if %errorlevel% equ 0 (
    echo NVIDIA GPU detected - installing CUDA...
    goto :cuda_setup
)

echo No compatible GPU found!
pause
exit /b 1

:amd_setup
echo.
echo ============================================
echo AMD GPU Setup for RX 5600 XT
echo ============================================
echo.

echo Installing AMD GPU drivers...
echo Please download and install from:
echo https://www.amd.com/en/support/graphics/radeon-500-series/radeon-rx-500-series/radeon-rx-5600-xt
echo.
echo Or install ROCm for mining:
echo https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.4.3/page/Introduction_to_ROCm_Installation.html
echo.

echo After driver installation, run:
echo pip install pyopencl
echo pip install rocm-smi
echo.

pause
goto :common_setup

:cuda_setup
echo.
echo ============================================
echo NVIDIA CUDA Setup
echo ============================================
echo.

echo Installing NVIDIA CUDA Toolkit...
echo Please download from:
echo https://developer.nvidia.com/cuda-downloads
echo.

echo Required: CUDA 11.8 or higher
echo.

echo After installation, run:
echo pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo.

pause
goto :common_setup

:common_setup
echo.
echo ============================================
echo Common GPU Libraries
echo ============================================
echo.

echo Installing common Python GPU libraries...
pip install --upgrade pip
pip install numpy
pip install psutil

echo.
echo Testing GPU detection...
python -c "
import subprocess
import sys

print('Testing GPU detection...')

# Test NVIDIA
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                          capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        print('✓ NVIDIA GPU detected:', result.stdout.strip())
except:
    print('✗ NVIDIA GPU not detected')

# Test AMD
try:
    result = subprocess.run(['rocm-smi', '--showid'], capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        print('✓ AMD GPU detected via ROCm')
except:
    print('✗ AMD ROCm not detected')

# Test OpenCL
try:
    result = subprocess.run(['clinfo', '--list'], capture_output=True, text=True, timeout=10)
    if 'AMD' in result.stdout.upper() or 'NVIDIA' in result.stdout.upper():
        print('✓ OpenCL GPU detected')
except:
    print('✗ OpenCL not available')

print('GPU setup test completed!')
"

echo.
echo ============================================
echo Setup Complete!
echo ============================================
echo.
echo Next steps:
echo 1. Restart your computer
echo 2. Run: python zion_gpu_miner.py
echo 3. Test mining: python -c "from zion_gpu_miner import ZionGPUMiner; m = ZionGPUMiner(); print(f'GPU: {m.gpu_available}')"

pause