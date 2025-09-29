#ifndef ZION_GPU_MINER_UNIFIED_H
#define ZION_GPU_MINER_UNIFIED_H

#include <vector>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include "zion-cosmic-harmony.h"

namespace zion {

enum class GPUPlatform {
    CUDA_NVIDIA,
    OPENCL_AMD,
    OPENCL_INTEL,
    OPENCL_NVIDIA
};

struct GPUDevice {
    GPUPlatform platform;
    std::string name;
    size_t memory_mb;
    int compute_units;
    int device_id;
    double performance_score;
    bool available;
};

struct GPUMiningResult {
    bool found_share;
    uint32_t nonce;
    uint8_t hash[32];
    uint64_t hashes_computed;
    std::chrono::milliseconds compute_time;
    GPUDevice device;
};

class UnifiedGPUMiner {
public:
    UnifiedGPUMiner();
    ~UnifiedGPUMiner();
    
    // Device management
    bool initialize();
    std::vector<GPUDevice> enumerate_devices();
    bool select_devices(const std::vector<int>& device_ids);
    
    // Mining operations
    bool start_mining();
    void stop_mining();
    bool is_mining() const { return mining_active_.load(); }
    
    // Job management
    void set_work(const uint8_t* header, uint64_t target_difficulty);
    std::vector<GPUMiningResult> get_results();
    
    // Statistics
    uint64_t get_total_hashrate() const { return total_hashrate_.load(); }
    uint64_t get_total_hashes() const { return total_hashes_.load(); }
    size_t get_active_devices() const { return active_devices_.size(); }
    std::vector<GPUDevice> get_available_devices() const { return available_devices_; }
    
    // Configuration
    void set_intensity(int intensity) { intensity_ = intensity; }
    void set_worksize(size_t worksize) { worksize_ = worksize; }
    
public:
    struct GPUWorker;
private:
    void worker_thread_function(GPUWorker& worker);
    
    std::atomic<bool> mining_active_{false};
    std::atomic<bool> shutdown_requested_{false};
    std::atomic<uint64_t> total_hashrate_{0};
    std::atomic<uint64_t> total_hashes_{0};
    
    std::vector<GPUDevice> available_devices_;
    std::vector<GPUDevice> active_devices_;
    std::vector<std::unique_ptr<GPUWorker>> workers_;
    
    std::mutex work_mutex_;
    uint8_t current_header_[80];
    uint64_t current_target_;
    bool work_available_;
    
    std::mutex results_mutex_;
    std::vector<GPUMiningResult> pending_results_;
    
    int intensity_ = 20;  // Default intensity
    size_t worksize_ = 256;  // Default workgroup size
    
    bool initialize_cuda();
    bool initialize_opencl();
    void cleanup();
};

// CUDA-specific worker
class CUDAWorker {
public:
    CUDAWorker(const GPUDevice& device);
    ~CUDAWorker();
    
    bool initialize();
    bool mine_work(const uint8_t* header, uint64_t target, GPUMiningResult& result);
    void cleanup();
    
private:
    GPUDevice device_;
    void* cuda_context_;
    bool initialized_;
};

// OpenCL-specific worker  
class OpenCLWorker {
public:
    OpenCLWorker(const GPUDevice& device);
    ~OpenCLWorker();
    
    bool initialize();
    bool mine_work(const uint8_t* header, uint64_t target, GPUMiningResult& result);
    void cleanup();
    
private:
    GPUDevice device_;
    void* opencl_context_;
    void* opencl_queue_;
    void* opencl_program_;
    void* opencl_kernel_;
    bool initialized_;
};

} // namespace zion

#endif // ZION_GPU_MINER_UNIFIED_H