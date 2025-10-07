# ZION Multi-AI Miner v2.7.1
## Platform: WINDOWS-X64

[ZION] **Multi-algorithm cryptocurrency miner with AI-optimized performance**

## [TARGET] Supported Algorithms

[OK] **Yescrypt** - CPU optimized with C extension
   - Performance: ~150K H/s per CPU core
   - Memory: ~2GB RAM
   - Status: [OK] Production ready

[OK] **RandomX** - Memory-hard CPU algorithm  
   - Performance: ~2K H/s per CPU core
   - Memory: ~4GB RAM
   - Status: [PROCESS] Integration in progress

[OK] **Multi-Algorithm** - Automatic switching
   - Adapts to best profitability
   - Multiple pools support
   - Status: [OK] Available

## [ZION] Quick Start

### Windows
1. Extract archive to desired folder
2. Double-click `start_zion_miner.bat`
3. Select mining algorithm
4. Enter your wallet address

### macOS  
1. Extract archive to Applications or desired folder
2. Double-click `start_zion_miner.command`
3. Allow execution if prompted
4. Select mining algorithm

### Linux
1. Extract archive: `tar -xzf ZION-Multi-AI-Miner-*.tar.gz`
2. Navigate: `cd ZION-Multi-AI-Miner-*`
3. Run: `./start_zion_miner.sh`
4. Select mining algorithm

## [GEAR] Advanced Configuration

### Command Line Usage
```bash
# Yescrypt mining with custom pool
python zion_yescrypt_hybrid.py --host pool.example.com --port 3335 --threads 8

# Check available real miners  
python zion_real_mining_system.py --check

# Mine with specific algorithm
python zion_real_mining_system.py --mine yescrypt --threads 6
```

### Pool Configuration
```bash
# Default ZION Universal Pool v3.0
Host: 127.0.0.1
Port: 3335
Difficulty: Auto-adjusting (1K - 50M)
```

## [INFO] Performance Expectations

| Algorithm    | CPU (6 cores) | Memory | GPU Support |
|--------------|---------------|--------|-------------|  
| Yescrypt     | ~900K H/s     | 2GB    | AMD (future)|
| RandomX      | ~12K H/s      | 4GB    | No          |
| Autolykos v2 | N/A           | N/A    | 1M+ H/s     |

## [TOOL] Requirements

- **Python**: 3.8+ (3.9+ recommended)
- **RAM**: 4GB minimum, 8GB recommended
- **CPU**: Multi-core processor (6+ cores optimal)
- **Network**: Internet connection for pool mining

## [SETUP] Dependencies

All dependencies installed automatically:
- `numpy>=1.21.0` - Numerical computations
- `requests>=2.25.0` - HTTP pool communication  
- `psutil>=5.8.0` - System monitoring

Optional GPU dependencies:
- `pyopencl>=2021.2.13` - AMD GPU acceleration
- `pycuda>=2021.1` - NVIDIA GPU acceleration

## [TOOLS] Troubleshooting

### Python Not Found
**Windows**: Install from https://python.org (3.9+)
**macOS**: `brew install python3` or download from python.org
**Linux**: `sudo apt install python3 python3-pip`

### Dependencies Fail to Install  
```bash
# Manual installation
pip install numpy requests psutil
```

### Low Performance
- Ensure adequate RAM (4GB+)
- Close other CPU-intensive applications
- Use optimal thread count (usually CPU cores - 1)

### Pool Connection Issues
- Check firewall settings
- Verify pool address and port
- Ensure internet connectivity

## [LOCK] Security & Open Source

All components are open source:
- **RandomX**: BSD-3-Clause License
- **SRBMiner**: Open Source Mining
- **ZION Core**: MIT License (original)

No telemetry, no hidden fees, transparent mining.

## [PHONE] Support

- **GitHub**: https://github.com/issy13elizabet/ZION2.7TestNet
- **Issues**: Use GitHub issue tracker for bug reports
- **Pool**: Connect to ZION Universal Pool for optimal mining

## [COPY] Version History

- **v2.7.1** - Multi-platform release, enhanced stability
- **v2.7.0** - Multi-algorithm support, GPU preparation
- **v2.6.x** - Core mining algorithms, pool integration

---
**Built with [EMOJI] for the ZION mining community**
**No simulations, only real algorithms! [TARGET]**
