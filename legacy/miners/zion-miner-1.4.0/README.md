# ZION Miner 1.4.0 - Cosmic Harmony Mining

## Overview
ZION Miner 1.4.0 represents the complete implementation of the ZION Cosmic Harmony algorithm with full GPU and CPU mining support. This miner uses advanced cryptographic operations including Blake3, Keccak-256, and SHA3-512 to create a unique and secure mining algorithm.

## Features

### 🚀 Cosmic Harmony Algorithm
- **Blake3 Foundation**: Ultra-fast cryptographic hash as the base layer
- **Keccak-256 Galactic Matrix**: For complex galactic matrix operations
- **SHA3-512 Stellar Harmony**: For stellar harmony processing
- **Golden Ratio Transformations**: Mathematical constants for cosmic operations
- **Multi-stage Hashing**: Advanced security through layered cryptographic operations

### 💻 Multi-Platform GPU Support
- **NVIDIA CUDA**: Native CUDA kernels for maximum performance
- **AMD OpenCL**: Cross-platform OpenCL kernels for AMD GPUs
- **Intel OpenCL**: Support for Intel integrated and discrete GPUs
- **Unified Architecture**: Single codebase managing all GPU platforms
- **Auto-Detection**: Automatic discovery and optimal configuration

### ⚡ Performance Optimizations
- **AVX2/AVX512 Instructions**: Hardware-accelerated cryptographic operations
- **Multi-threading**: Full CPU core utilization
- **Memory Optimization**: Efficient memory management and caching
- **Assembly Optimizations**: Hand-tuned critical path optimizations
- **Adaptive Intensity**: Dynamic workload adjustment based on hardware

## Requirements

### System Requirements
- **Linux**: Ubuntu 20.04+ (recommended), CentOS 8+, or similar
- **CPU**: Modern x86_64 processor with AVX2 support
- **RAM**: Minimum 4GB, recommended 8GB+
- **GPU** (optional): NVIDIA GTX 900+ or AMD RX 400+ series

### Software Dependencies
- **GCC 9+** or **Clang 10+** with C++17 support
- **CMake 3.16+** for build system
- **CUDA Toolkit 11.0+** (for NVIDIA GPU mining)
- **OpenCL 1.2+** (for AMD/Intel GPU mining)

### Required Libraries
- **Blake3**: Manually compiled from source
- **OpenSSL 1.1+**: For Keccak-256 and SHA3-512
- **Boost 1.70+**: For system utilities and threading
- **Intel TBB**: For parallel processing
- **GSL**: GNU Scientific Library for mathematical operations
- **FFTW3**: For Fourier transforms in cosmic calculations
- **Eigen3**: For linear algebra and matrix operations

## Installation

### 1. Install System Dependencies

#### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install build-essential cmake git
sudo apt install libssl-dev libboost-all-dev libtbb-dev
sudo apt install libgsl-dev libfftw3-dev libeigen3-dev
sudo apt install opencl-headers ocl-icd-opencl-dev
```

#### CentOS/RHEL:
```bash
sudo yum groupinstall "Development Tools"
sudo yum install cmake3 git openssl-devel boost-devel tbb-devel
sudo yum install gsl-devel fftw-devel eigen3-devel
sudo yum install opencl-headers ocl-icd-devel
```

### 2. Install CUDA (for NVIDIA GPUs)
Download and install CUDA Toolkit from [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads):
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-2
```

### 3. Compile Blake3 Library
```bash
git clone https://github.com/BLAKE3-team/BLAKE3.git
cd BLAKE3/c
make
sudo make install
sudo ldconfig
cd ../..
```

### 4. Build ZION Miner
```bash
git clone <zion-miner-repository>
cd zion-miner-1.4.0
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Usage

### Basic Mining
```bash
# Mine with all available hardware (CPU + GPU)
./zion-miner --pool 91.98.122.165:3333 --wallet YOUR_WALLET_ADDRESS

# CPU-only mining
./zion-miner --cpu-only --pool 91.98.122.165:3333 --wallet YOUR_WALLET_ADDRESS

# GPU-only mining
./zion-miner --gpu-only --pool 91.98.122.165:3333 --wallet YOUR_WALLET_ADDRESS
```

### Command Line Options
```
--pool <host:port>    Mining pool address (default: 91.98.122.165:3333)
--wallet <address>    Your ZION wallet address for rewards
--cpu-only           Use CPU mining only
--gpu-only           Use GPU mining only  
--threads <n>        Number of CPU threads (default: auto-detect)
--intensity <n>      GPU mining intensity (default: auto)
--help               Show help message
```

### Example with Custom Pool
```bash
./zion-miner \
  --pool your-pool.com:4444 \
  --wallet ZiYgACwXhrYG9iLjRfBgEdgsGsT6DqQ2brtM8j9iR3Rs7geE5kyj7oEGkw9LpjaGX9p1h7uRNJg5BkWKu8HD28EMPpJAYUdJ4 \
  --threads 8
```

## Mining Statistics

The miner displays real-time statistics including:
- **Hashrate**: Current hashes per second (H/s)
- **Total Hashes**: Cumulative hashes computed
- **Shares Found**: Valid shares submitted to pool
- **Hardware Status**: CPU and GPU utilization
- **Connection Status**: Pool connectivity

Example output:
```
📊 MINING STATS (Runtime: 300s)
   CPU: 15,420 H/s (4,626,000 total)
   GPU: 145,680 H/s (43,704,000 total)
 TOTAL: 161,100 H/s (48,330,000 total)
STATUS: Connected ✅
```

## Configuration

### GPU Selection
The miner automatically detects and ranks all available GPUs by performance. You can view detected devices at startup:
```
Found 3 GPU devices:
  [0] CUDA: NVIDIA GeForce RTX 3080 (10240 MB, 68 CUs, Score: 174080)
  [1] OpenCL-AMD: AMD Radeon RX 6800 XT (16384 MB, 72 CUs, Score: 4608)
  [2] OpenCL-Intel: Intel UHD Graphics 630 (2048 MB, 24 CUs, Score: 1536)
```

### Performance Tuning

#### CPU Optimization
- Use `--threads` to match your CPU core count
- Ensure adequate cooling for sustained mining
- Close unnecessary applications to maximize CPU availability

#### GPU Optimization
- Increase GPU power limits and memory clocks
- Ensure adequate PSU capacity and cooling
- Monitor temperatures to prevent throttling
- Use `--intensity` to fine-tune GPU workload

## Algorithm Details

### ZION Cosmic Harmony Algorithm
The ZION Cosmic Harmony algorithm implements a multi-stage cryptographic process:

1. **Blake3 Base Hash**: Initial high-speed hashing of input data
2. **Galactic Matrix Operations**: Keccak-256 based matrix transformations
3. **Stellar Harmony Processing**: SHA3-512 harmonic calculations  
4. **Golden Ratio Transform**: Mathematical constant-based final mixing
5. **Difficulty Validation**: Target comparison and nonce verification

### Cryptographic Security
- **Blake3**: Provides 256-bit base security with extreme performance
- **Keccak-256**: NIST-standardized secure hash for matrix operations
- **SHA3-512**: 512-bit security for stellar processing
- **Combined Security**: Multi-algorithm approach prevents specialized attacks

## Troubleshooting

### Build Issues

#### Blake3 Not Found
```bash
# Manually specify Blake3 location
cmake -DBLAKE3_LIBRARY=/usr/local/lib/libblake3.a \
      -DBLAKE3_INCLUDE_DIR=/usr/local/include ..
```

#### CUDA Issues
```bash
# Verify CUDA installation
nvidia-smi
nvcc --version

# Check device detection
/usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery
```

#### OpenCL Issues
```bash
# Check OpenCL platforms and devices
clinfo

# Install additional drivers if no devices found
sudo apt install mesa-opencl-icd
```

### Runtime Issues

#### Low Hashrate
- Verify GPU drivers are up to date
- Check thermal throttling with monitoring tools
- Increase GPU power limits and clocks
- Ensure adequate system cooling

#### Connection Issues
- Verify pool address and port
- Check firewall settings
- Confirm wallet address format
- Test network connectivity to pool

#### High Rejection Rate
- Verify system time synchronization
- Check for hardware instability
- Reduce mining intensity if overclocked
- Monitor for thermal issues

## Performance Benchmarks

### Expected Hashrates (approximate)

#### CPU Mining
- Intel i7-8700K: ~12,000 H/s
- AMD Ryzen 7 3700X: ~18,000 H/s  
- Intel i9-10900K: ~22,000 H/s
- AMD Ryzen 9 5900X: ~28,000 H/s

#### GPU Mining
- NVIDIA GTX 1660 Ti: ~35,000 H/s
- NVIDIA RTX 3070: ~85,000 H/s
- NVIDIA RTX 3080: ~125,000 H/s
- AMD RX 5700 XT: ~65,000 H/s
- AMD RX 6800 XT: ~95,000 H/s

*Note: Actual performance varies based on hardware configuration, cooling, and system optimization.*

## Development

### Building from Source
```bash
git clone <repository>
cd zion-miner-1.4.0
mkdir build && cd build

# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j$(nproc)

# Release build with optimizations
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### Code Structure
- `zion-cosmic-harmony.h/cpp`: Core algorithm implementation
- `zion-cuda-kernel.cu`: NVIDIA CUDA mining kernels
- `zion-opencl-kernel.cl`: OpenCL kernels for AMD/Intel
- `zion-gpu-miner-unified.h/cpp`: Unified GPU mining interface
- `zion-miner-main.cpp`: Main application and stratum client

## License
[Specify your license here]

## Support
For technical support, bug reports, or feature requests:
- GitHub Issues: [Repository Issues Page]
- Discord: [Community Discord Server]
- Email: [Support Email]

## Acknowledgments
- BLAKE3 Team for the exceptional cryptographic hash function
- NVIDIA and AMD for GPU development tools and documentation
- OpenCL Working Group for cross-platform GPU computing standards
- ZION Community for testing and feedback

---

**Happy Mining! ⛏️ 💎**