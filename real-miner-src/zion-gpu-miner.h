#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <atomic>
#include <thread>
#include <chrono>
#include <mutex>

// OpenCL and CUDA headers (optional, fallback to simulation if not available)
#if defined(OPENCL_SUPPORT) && OPENCL_SUPPORT
    #ifdef _WIN32
        #define CL_TARGET_OPENCL_VERSION 200
        #include <CL/cl.h>
    #else
        #ifdef __APPLE__
            #include <OpenCL/cl.h>
        #else
            #include <CL/cl.h>
        #endif
    #endif
#else
    // Fallback OpenCL types for compilation without OpenCL
    typedef void* cl_context;
    typedef void* cl_command_queue;
    typedef void* cl_program;
    typedef void* cl_kernel;
    typedef void* cl_mem;
    typedef int cl_int;
    typedef unsigned int cl_uint;
    typedef unsigned long cl_ulong;
    typedef void* cl_platform_id;
    typedef void* cl_device_id;
    #define CL_SUCCESS 0
#endif

#if defined(CUDA_SUPPORT) && CUDA_SUPPORT
    #include <cuda_runtime.h>
    #include <device_launch_parameters.h>
#else
    // Fallback CUDA types for compilation without CUDA
    typedef int cudaError_t;
    typedef void* cudaStream_t;
    #define cudaSuccess 0
#endif

enum class GPUType {
    AUTO_DETECT,
    OPENCL_AMD,
    OPENCL_NVIDIA, 
    CUDA_NVIDIA,
    OPENCL_INTEL,
    VULKAN_COMPUTE
};

enum class GPUOptimization {
    BALANCED,
    POWER_EFFICIENT,
    MAX_PERFORMANCE,
    AI_ENHANCED,
    COSMIC_HARMONY
};

struct GPUDevice {
    std::string name;
    std::string vendor;
    GPUType type;
    size_t memory_mb;
    uint32_t compute_units;
    uint32_t max_work_group_size;
    uint32_t max_frequency_mhz;
    double performance_score;
    bool ai_acceleration_support;
    
    GPUDevice() : memory_mb(0), compute_units(0), max_work_group_size(0), 
                  max_frequency_mhz(0), performance_score(0.0), 
                  ai_acceleration_support(false) {}
};

struct GPUMiningStats {
    std::atomic<uint64_t> gpu_hashrate{0};
    std::atomic<uint64_t> gpu_shares_found{0};
    std::atomic<uint64_t> gpu_shares_accepted{0};
    std::atomic<double> gpu_temperature{0.0};
    std::atomic<uint32_t> gpu_power_usage{0};
    std::atomic<double> gpu_efficiency{0.0}; // hashes per watt
    std::chrono::steady_clock::time_point start_time;
    
    GPUMiningStats() : start_time(std::chrono::steady_clock::now()) {}
    
    // Copy constructor for atomic members
    GPUMiningStats(const GPUMiningStats& other) : start_time(other.start_time) {
        gpu_hashrate.store(other.gpu_hashrate.load());
        gpu_shares_found.store(other.gpu_shares_found.load());
        gpu_shares_accepted.store(other.gpu_shares_accepted.load());
        gpu_temperature.store(other.gpu_temperature.load());
        gpu_power_usage.store(other.gpu_power_usage.load());
        gpu_efficiency.store(other.gpu_efficiency.load());
    }
    
    GPUMiningStats& operator=(const GPUMiningStats& other) {
        if (this != &other) {
            gpu_hashrate.store(other.gpu_hashrate.load());
            gpu_shares_found.store(other.gpu_shares_found.load());
            gpu_shares_accepted.store(other.gpu_shares_accepted.load());
            gpu_temperature.store(other.gpu_temperature.load());
            gpu_power_usage.store(other.gpu_power_usage.load());
            gpu_efficiency.store(other.gpu_efficiency.load());
            start_time = other.start_time;
        }
        return *this;
    }
};

class ZionGPUMiner {
private:
    std::vector<GPUDevice> available_gpus_;
    std::vector<GPUDevice> selected_gpus_;
    GPUOptimization optimization_mode_;
    std::atomic<bool> is_mining_{false};
    std::vector<std::thread> gpu_threads_;
    GPUMiningStats stats_;
    
    // OpenCL context and objects
    cl_context opencl_context_;
    cl_command_queue opencl_queue_;
    cl_program opencl_program_;
    cl_kernel opencl_kernel_;
    cl_mem opencl_input_buffer_;
    cl_mem opencl_output_buffer_;
    
    // CUDA context and objects  
    void* cuda_context_;
    void* cuda_stream_;
    void* cuda_input_buffer_;
    void* cuda_output_buffer_;
    
    // AI Enhancement
    std::atomic<double> ai_gpu_multiplier_{2.718}; // Euler's number for GPU enhancement
    std::atomic<uint32_t> cosmic_gpu_level_{88}; // Cosmic GPU level
    
public:
    ZionGPUMiner();
    ~ZionGPUMiner();
    
    // GPU Detection and Management
    bool detect_gpus();
    std::vector<GPUDevice> get_available_gpus() const { return available_gpus_; }
    bool select_gpu(int gpu_index);
    bool select_all_gpus();
    bool select_best_gpus(int count = 1);
    
    // Mining Control
    bool start_gpu_mining(GPUOptimization mode = GPUOptimization::AI_ENHANCED);
    void stop_gpu_mining();
    bool is_mining() const { return is_mining_.load(); }
    
    // Statistics and Monitoring
    GPUMiningStats get_stats() const { return stats_; }
    void print_gpu_stats() const;
    std::string get_gpu_info() const;
    
    // AI Enhancement
    void enhance_with_gpu_ai();
    void optimize_for_zion_algorithm();
    
    // ZION Cosmic Harmony GPU Implementation
    uint64_t cosmic_harmony_gpu_hash(const std::vector<uint8_t>& data, 
                                    uint32_t nonce, int gpu_index);
    
private:
    // OpenCL Implementation
    bool init_opencl(int gpu_index);
    void cleanup_opencl();
    bool compile_opencl_kernel();
    void gpu_mining_thread_opencl(int gpu_index);
    
    // CUDA Implementation  
    bool init_cuda(int gpu_index);
    void cleanup_cuda();
    void gpu_mining_thread_cuda(int gpu_index);
    
    // GPU Monitoring
    void monitor_gpu_health(int gpu_index);
    double get_gpu_temperature(int gpu_index);
    uint32_t get_gpu_power_usage(int gpu_index);
    
    // Performance Optimization
    void optimize_gpu_settings(int gpu_index);
    uint32_t calculate_optimal_work_group_size(int gpu_index);
    uint32_t calculate_optimal_global_size(int gpu_index);
};