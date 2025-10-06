#!/bin/bash
# ZION 2.7.1 - GPU Setup Script
# Install GPU mining dependencies for CUDA and OpenCL

echo "🎮 ZION 2.7.1 GPU Setup"
echo "========================"

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "🐧 Linux detected"

    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        echo "✅ NVIDIA GPU detected"
        echo "📦 Installing PyCUDA..."

        # Install CUDA toolkit if not present
        if ! command -v nvcc &> /dev/null; then
            echo "⚠️ CUDA toolkit not found. Please install CUDA toolkit first:"
            echo "   Ubuntu/Debian: sudo apt install nvidia-cuda-toolkit"
            echo "   Or download from: https://developer.nvidia.com/cuda-downloads"
            exit 1
        fi

        pip3 install pycuda --verbose
    else
        echo "⚠️ No NVIDIA GPU detected, skipping PyCUDA"
    fi

    # Check for AMD/Intel GPU or OpenCL
    if command -v clinfo &> /dev/null || command -v glxinfo &> /dev/null; then
        echo "✅ OpenCL capable GPU detected"
        echo "📦 Installing PyOpenCL..."

        # Install OpenCL headers if needed
        if [[ -f /etc/debian_version ]]; then
            sudo apt update
            sudo apt install -y opencl-headers ocl-icd-opencl-dev
        elif [[ -f /etc/redhat-release ]]; then
            sudo dnf install -y opencl-headers ocl-icd-devel
        fi

        pip3 install pyopencl --verbose
    else
        echo "⚠️ No OpenCL GPU detected, skipping PyOpenCL"
    fi

elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 macOS detected"

    # macOS has OpenCL built-in
    echo "📦 Installing PyOpenCL for macOS..."
    pip3 install pyopencl --verbose

elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo "🪟 Windows detected"

    # Windows CUDA setup
    if nvidia-smi &> /dev/null; then
        echo "✅ NVIDIA GPU detected on Windows"
        echo "📦 Installing PyCUDA..."

        # Check if CUDA is installed
        if ! nvcc --version &> /dev/null; then
            echo "⚠️ CUDA toolkit not found. Please install CUDA toolkit from:"
            echo "   https://developer.nvidia.com/cuda-downloads"
            exit 1
        fi

        pip3 install pycuda --verbose
    fi

    # Windows OpenCL
    echo "📦 Installing PyOpenCL for Windows..."
    pip3 install pyopencl --verbose

else
    echo "❌ Unsupported OS: $OSTYPE"
    exit 1
fi

echo ""
echo "🧪 Testing GPU setup..."

# Test PyCUDA if available
python3 -c "
try:
    import pycuda.driver as cuda
    cuda.init()
    print('✅ PyCUDA: Found', cuda.Device.count(), 'GPU(s)')
    for i in range(cuda.Device.count()):
        dev = cuda.Device(i)
        print(f'   GPU{i}: {dev.name()}')
except ImportError:
    print('⚠️ PyCUDA not available')
except Exception as e:
    print('❌ PyCUDA error:', e)
"

# Test PyOpenCL if available
python3 -c "
try:
    import pyopencl as cl
    platforms = cl.get_platforms()
    print('✅ PyOpenCL: Found', len(platforms), 'platform(s)')
    for i, platform in enumerate(platforms):
        devices = platform.get_devices()
        print(f'   Platform {i}: {platform.name} ({len(devices)} device(s))')
        for j, device in enumerate(devices):
            print(f'     Device {j}: {device.name}')
except ImportError:
    print('⚠️ PyOpenCL not available')
except Exception as e:
    print('❌ PyOpenCL error:', e)
"

echo ""
echo "🎉 GPU setup complete!"
echo ""
echo "💡 Usage:"
echo "   python3 zion_cli.py algorithms benchmark  # Test GPU performance"
echo "   python3 zion_cli.py algorithms set gpu     # Use GPU mining"
echo "   python3 zion_cli.py mine --address <addr>  # Start GPU mining"