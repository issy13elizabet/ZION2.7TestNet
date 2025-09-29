#include "zion-gpu-miner-unified.h"
#include <iostream>
#include <chrono>
#include <algorithm>

#ifdef __NVCC__
#include <cuda_runtime.h>
#include <cuda.h>
extern "C" void cuda_zion_mine(const uint8_t*, uint32_t, uint32_t, uint64_t, uint32_t*, uint8_t*, bool*);
#endif

#ifdef OPENCL_ENABLED
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#endif

namespace zion {

struct UnifiedGPUMiner::GPUWorker {
    GPUDevice device;
    std::unique_ptr<CUDAWorker> cuda_worker;
    std::unique_ptr<OpenCLWorker> opencl_worker;
    std::thread worker_thread;
    std::atomic<bool> active{false};
    std::atomic<uint64_t> hashrate{0};
    
    GPUWorker(const GPUDevice& dev) : device(dev) {
        if (device.platform == GPUPlatform::CUDA_NVIDIA) {
            cuda_worker = std::make_unique<CUDAWorker>(device);
        } else {
            opencl_worker = std::make_unique<OpenCLWorker>(device);
        }
    }
};

UnifiedGPUMiner::UnifiedGPUMiner() : work_available_(false) {
    memset(current_header_, 0, sizeof(current_header_));
    current_target_ = 1000000;
}

UnifiedGPUMiner::~UnifiedGPUMiner() {
    stop_mining();
    cleanup();
}

bool UnifiedGPUMiner::initialize() {
    std::cout << "Initializing Unified GPU Miner..." << std::endl;
    
    bool cuda_ok = initialize_cuda();
    bool opencl_ok = initialize_opencl();
    
    if (!cuda_ok && !opencl_ok) {
        std::cerr << "No GPU platforms available!" << std::endl;
        return false;
    }
    
    available_devices_ = enumerate_devices();
    std::cout << "Found " << available_devices_.size() << " GPU devices:" << std::endl;
    
    for (const auto& dev : available_devices_) {
        std::string platform_name;
        switch (dev.platform) {
            case GPUPlatform::CUDA_NVIDIA: platform_name = "CUDA"; break;
            case GPUPlatform::OPENCL_AMD: platform_name = "OpenCL-AMD"; break;
            case GPUPlatform::OPENCL_INTEL: platform_name = "OpenCL-Intel"; break;
            case GPUPlatform::OPENCL_NVIDIA: platform_name = "OpenCL-NVIDIA"; break;
        }
        
        std::cout << "  [" << dev.device_id << "] " << platform_name 
                  << ": " << dev.name << " (" << dev.memory_mb << " MB, " 
                  << dev.compute_units << " CUs, Score: " << dev.performance_score << ")" << std::endl;
    }
    
    return !available_devices_.empty();
}

bool UnifiedGPUMiner::initialize_cuda() {
#ifdef __NVCC__
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
        std::cout << "CUDA not available or no CUDA devices found" << std::endl;
        return false;
    }
    
    std::cout << "CUDA initialized successfully with " << device_count << " devices" << std::endl;
    return true;
#else
    std::cout << "CUDA support not compiled" << std::endl;
    return false;
#endif
}

bool UnifiedGPUMiner::initialize_opencl() {
#ifdef OPENCL_ENABLED
    try {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        
        if (platforms.empty()) {
            std::cout << "No OpenCL platforms found" << std::endl;
            return false;
        }
        
        std::cout << "OpenCL initialized with " << platforms.size() << " platforms" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cout << "OpenCL initialization failed: " << e.what() << std::endl;
        return false;
    }
#else
    std::cout << "OpenCL support not compiled" << std::endl;
    return false;
#endif
}

std::vector<GPUDevice> UnifiedGPUMiner::enumerate_devices() {
    std::vector<GPUDevice> devices;
    
#ifdef __NVCC__
    // Enumerate CUDA devices
    int cuda_device_count;
    if (cudaGetDeviceCount(&cuda_device_count) == cudaSuccess) {
        for (int i = 0; i < cuda_device_count; i++) {
            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
                GPUDevice dev;
                dev.platform = GPUPlatform::CUDA_NVIDIA;
                dev.name = prop.name;
                dev.memory_mb = prop.totalGlobalMem / (1024 * 1024);
                dev.compute_units = prop.multiProcessorCount;
                dev.device_id = devices.size();
                dev.performance_score = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor;
                dev.available = true;
                devices.push_back(dev);
            }
        }
    }
#endif
    
    // Enumerate OpenCL devices
#ifdef OPENCL_ENABLED
    try {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        
        for (const auto& platform : platforms) {
            std::vector<cl::Device> platform_devices;
            platform.getDevices(CL_DEVICE_TYPE_GPU, &platform_devices);
            
            std::string platform_name = platform.getInfo<CL_PLATFORM_NAME>();
            
            for (const auto& cl_device : platform_devices) {
                GPUDevice dev;
                
                // Determine platform type
                if (platform_name.find("AMD") != std::string::npos ||
                    platform_name.find("Advanced Micro Devices") != std::string::npos) {
                    dev.platform = GPUPlatform::OPENCL_AMD;
                } else if (platform_name.find("Intel") != std::string::npos) {
                    dev.platform = GPUPlatform::OPENCL_INTEL;
                } else if (platform_name.find("NVIDIA") != std::string::npos) {
                    dev.platform = GPUPlatform::OPENCL_NVIDIA;
                } else {
                    dev.platform = GPUPlatform::OPENCL_AMD; // Default
                }
                
                dev.name = cl_device.getInfo<CL_DEVICE_NAME>();
                dev.memory_mb = cl_device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / (1024 * 1024);
                dev.compute_units = cl_device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
                dev.device_id = devices.size();
                dev.performance_score = dev.compute_units * 64; // Rough estimate
                dev.available = true;
                devices.push_back(dev);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "OpenCL enumeration error: " << e.what() << std::endl;
    }
#endif
    
    // Sort by performance score (descending)
    std::sort(devices.begin(), devices.end(), 
              [](const GPUDevice& a, const GPUDevice& b) {
                  return a.performance_score > b.performance_score;
              });
    
    // Reassign device IDs after sorting
    for (size_t i = 0; i < devices.size(); i++) {
        devices[i].device_id = i;
    }
    
    return devices;
}

bool UnifiedGPUMiner::select_devices(const std::vector<int>& device_ids) {
    active_devices_.clear();
    
    for (int id : device_ids) {
        if (id >= 0 && id < (int)available_devices_.size()) {
            active_devices_.push_back(available_devices_[id]);
        }
    }
    
    if (active_devices_.empty()) {
        std::cerr << "No valid devices selected!" << std::endl;
        return false;
    }
    
    std::cout << "Selected " << active_devices_.size() << " GPU devices for mining" << std::endl;
    return true;
}

bool UnifiedGPUMiner::start_mining() {
    if (mining_active_.load() || active_devices_.empty()) {
        return false;
    }
    
    std::cout << "Starting GPU mining on " << active_devices_.size() << " devices..." << std::endl;
    
    workers_.clear();
    workers_.reserve(active_devices_.size());
    
    for (const auto& device : active_devices_) {
        auto worker = std::make_unique<GPUWorker>(device);
        
        // Initialize worker
        bool init_ok = false;
        if (device.platform == GPUPlatform::CUDA_NVIDIA) {
            init_ok = worker->cuda_worker->initialize();
        } else {
            init_ok = worker->opencl_worker->initialize();
        }
        
        if (init_ok) {
            worker->active.store(true);
            worker->worker_thread = std::thread([this, &worker = *worker]() {
                worker_thread_function(worker);
            });
            workers_.push_back(std::move(worker));
        } else {
            std::cerr << "Failed to initialize device: " << device.name << std::endl;
        }
    }
    
    if (!workers_.empty()) {
        mining_active_.store(true);
        std::cout << "GPU mining started with " << workers_.size() << " workers" << std::endl;
        return true;
    }
    
    return false;
}

void UnifiedGPUMiner::stop_mining() {
    if (!mining_active_.load()) {
        return;
    }
    
    std::cout << "Stopping GPU mining..." << std::endl;
    shutdown_requested_.store(true);
    mining_active_.store(false);
    
    for (auto& worker : workers_) {
        worker->active.store(false);
        if (worker->worker_thread.joinable()) {
            worker->worker_thread.join();
        }
    }
    
    workers_.clear();
    shutdown_requested_.store(false);
    std::cout << "GPU mining stopped" << std::endl;
}

void UnifiedGPUMiner::set_work(const uint8_t* header, uint64_t target_difficulty) {
    std::lock_guard<std::mutex> lock(work_mutex_);
    memcpy(current_header_, header, 80);
    current_target_ = target_difficulty;
    work_available_ = true;
}

std::vector<GPUMiningResult> UnifiedGPUMiner::get_results() {
    std::lock_guard<std::mutex> lock(results_mutex_);
    std::vector<GPUMiningResult> results = std::move(pending_results_);
    pending_results_.clear();
    return results;
}

void UnifiedGPUMiner::worker_thread_function(GPUWorker& worker) {
    std::cout << "GPU worker started for: " << worker.device.name << std::endl;
    
    auto last_hashrate_update = std::chrono::steady_clock::now();
    uint64_t hashes_since_update = 0;
    
    while (worker.active.load() && !shutdown_requested_.load()) {
        // Get current work
        uint8_t header[80];
        uint64_t target;
        bool has_work;
        
        {
            std::lock_guard<std::mutex> lock(work_mutex_);
            has_work = work_available_;
            if (has_work) {
                memcpy(header, current_header_, 80);
                target = current_target_;
            }
        }
        
        if (!has_work) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        // Mine work
        GPUMiningResult result;
        result.device = worker.device;
        
        bool mine_ok = false;
        if (worker.device.platform == GPUPlatform::CUDA_NVIDIA) {
            mine_ok = worker.cuda_worker->mine_work(header, target, result);
        } else {
            mine_ok = worker.opencl_worker->mine_work(header, target, result);
        }
        
        if (mine_ok) {
            hashes_since_update += result.hashes_computed;
            
            // Update hashrate every 5 seconds
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_hashrate_update);
            if (elapsed.count() >= 5) {
                double hashrate = hashes_since_update / elapsed.count();
                worker.hashrate.store((uint64_t)hashrate);
                last_hashrate_update = now;
                hashes_since_update = 0;
                
                // Update total hashrate
                uint64_t total = 0;
                for (const auto& w : workers_) {
                    total += w->hashrate.load();
                }
                total_hashrate_.store(total);
            }
            
            // Store results if share found
            if (result.found_share) {
                std::lock_guard<std::mutex> lock(results_mutex_);
                pending_results_.push_back(result);
            }
            
            total_hashes_.fetch_add(result.hashes_computed);
        }
    }
    
    std::cout << "GPU worker stopped for: " << worker.device.name << std::endl;
}

void UnifiedGPUMiner::cleanup() {
    // Cleanup handled in destructor and stop_mining()
}

// CUDA Worker Implementation
CUDAWorker::CUDAWorker(const GPUDevice& device) 
    : device_(device), cuda_context_(nullptr), initialized_(false) {
}

CUDAWorker::~CUDAWorker() {
    cleanup();
}

bool CUDAWorker::initialize() {
#ifdef __NVCC__
    cudaError_t error = cudaSetDevice(device_.device_id);
    if (error != cudaSuccess) {
        std::cerr << "Failed to set CUDA device " << device_.device_id << std::endl;
        return false;
    }
    
    initialized_ = true;
    std::cout << "CUDA worker initialized for: " << device_.name << std::endl;
    return true;
#else
    return false;
#endif
}

bool CUDAWorker::mine_work(const uint8_t* header, uint64_t target, GPUMiningResult& result) {
#ifdef __NVCC__
    if (!initialized_) return false;
    
    auto start_time = std::chrono::steady_clock::now();
    
    uint32_t start_nonce = rand() % 0xFFFFFFFF;
    uint32_t nonce_range = 1024 * intensity_;  // Use intensity from miner
    
    uint32_t found_nonce = 0;
    uint8_t found_hash[32];
    bool found = false;
    
    // Call CUDA kernel
    cuda_zion_mine(header, start_nonce, nonce_range, target, &found_nonce, found_hash, &found);
    
    auto end_time = std::chrono::steady_clock::now();
    result.compute_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    result.hashes_computed = nonce_range;
    result.found_share = found;
    
    if (found) {
        result.nonce = found_nonce;
        memcpy(result.hash, found_hash, 32);
    }
    
    return true;
#else
    return false;
#endif
}

void CUDAWorker::cleanup() {
    initialized_ = false;
}

// OpenCL Worker Implementation
OpenCLWorker::OpenCLWorker(const GPUDevice& device) 
    : device_(device), opencl_context_(nullptr), opencl_queue_(nullptr), 
      opencl_program_(nullptr), opencl_kernel_(nullptr), initialized_(false) {
}

OpenCLWorker::~OpenCLWorker() {
    cleanup();
}

bool OpenCLWorker::initialize() {
    try {
        // This would contain full OpenCL initialization
        // For now, return a placeholder
        std::cout << "OpenCL worker initialized for: " << device_.name << std::endl;
        initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "OpenCL worker init failed: " << e.what() << std::endl;
        return false;
    }
}

bool OpenCLWorker::mine_work(const uint8_t* header, uint64_t target, GPUMiningResult& result) {
    if (!initialized_) return false;
    
    // Placeholder implementation for OpenCL mining
    auto start_time = std::chrono::steady_clock::now();
    
    // Simulate mining work
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    auto end_time = std::chrono::steady_clock::now();
    result.compute_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    result.hashes_computed = 1000000; // Simulated
    result.found_share = false; // Placeholder
    
    return true;
}

void OpenCLWorker::cleanup() {
    initialized_ = false;
}

} // namespace zion