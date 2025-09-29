#pragma once

#ifndef ZION_GPU_MINER_H
#define ZION_GPU_MINER_H

#include <memory>
#include <vector>
#include <string>
#include <atomic>

namespace zion {

enum class GPUType {
    NONE,
    NVIDIA_CUDA,
    AMD_OPENCL,
    INTEL_OPENCL
};

struct GPUInfo {
    GPUType type;
    std::string name;
    size_t memory_mb;
    uint32_t compute_units;
    bool available;
    
    GPUInfo() : type(GPUType::NONE), memory_mb(0), compute_units(0), available(false) {}
};

class ZionGPUMiner {
public:
    ZionGPUMiner();
    ~ZionGPUMiner();
    
    // GPU Detection
    std::vector<GPUInfo> detect_gpus();
    bool initialize_gpu(int device_id = 0);
    
    // Mining operations
    bool start_mining();
    bool stop_mining();
    bool is_mining() const;
    
    // Statistics
    double get_hashrate() const;
    uint64_t get_total_hashes() const;
    
    // Configuration
    void set_threads(uint32_t thread_count);
    void set_intensity(uint32_t intensity);
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
    
    std::atomic<bool> mining_active_{false};
    std::atomic<uint64_t> total_hashes_{0};
    std::atomic<double> current_hashrate_{0.0};
};

// Stub implementations for when CUDA/OpenCL not available
class StubGPUMiner {
public:
    std::vector<GPUInfo> detect_gpus() { return {}; }
    bool initialize_gpu(int device_id = 0) { (void)device_id; return false; }
    bool start_mining() { return false; }
    bool stop_mining() { return true; }
    bool is_mining() const { return false; }
    double get_hashrate() const { return 0.0; }
    uint64_t get_total_hashes() const { return 0; }
    void set_threads(uint32_t thread_count) { (void)thread_count; }
    void set_intensity(uint32_t intensity) { (void)intensity; }
};

} // namespace zion

#endif // ZION_GPU_MINER_H