#include "zion-gpu-miner.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>

#ifndef M_E
#define M_E 2.71828182845904523536
#endif

ZionGPUMiner::ZionGPUMiner() 
    : optimization_mode_(GPUOptimization::AI_ENHANCED)
    , opencl_context_(nullptr)
    , opencl_queue_(nullptr)  
    , opencl_program_(nullptr)
    , opencl_kernel_(nullptr)
    , opencl_input_buffer_(nullptr)
    , opencl_output_buffer_(nullptr)
    , cuda_context_(nullptr)
    , cuda_stream_(nullptr)
    , cuda_input_buffer_(nullptr)
    , cuda_output_buffer_(nullptr)
{
    std::cout << "🎮 ZION GPU Miner inicializován" << std::endl;
    
    // Detect available GPUs
    if (detect_gpus()) {
        std::cout << "✅ Nalezeno " << available_gpus_.size() << " GPU(s)" << std::endl;
    } else {
        std::cout << "⚠️ Žádné kompatibilní GPU nenalezeno" << std::endl;
    }
}

ZionGPUMiner::~ZionGPUMiner() {
    stop_gpu_mining();
    cleanup_opencl();
    cleanup_cuda();
}

bool ZionGPUMiner::detect_gpus() {
    available_gpus_.clear();
    
    std::cout << "🔍 Detekuji GPU zařízení..." << std::endl;
    
    // Try OpenCL detection first (if supported)
#if defined(OPENCL_SUPPORT) && OPENCL_SUPPORT
    cl_uint num_platforms = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
    
    if (err == CL_SUCCESS && num_platforms > 0) {
        std::cout << "📱 Nalezeno " << num_platforms << " OpenCL platforem" << std::endl;
        
        std::vector<cl_platform_id> platforms(num_platforms);
        clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
        
        for (cl_uint i = 0; i < num_platforms; i++) {
            char platform_name[256];
            clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, nullptr);
            
            cl_uint num_devices = 0;
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
            
            if (num_devices > 0) {
                std::vector<cl_device_id> devices(num_devices);
                clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
                
                for (cl_uint j = 0; j < num_devices; j++) {
                    GPUDevice gpu;
                    
                    char device_name[256];
                    char vendor_name[256];
                    cl_ulong memory_size;
                    cl_uint compute_units;
                    size_t max_work_group;
                    cl_uint max_freq;
                    
                    clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
                    clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, sizeof(vendor_name), vendor_name, nullptr);
                    clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(memory_size), &memory_size, nullptr);
                    clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, nullptr);
                    clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group), &max_work_group, nullptr);
                    clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(max_freq), &max_freq, nullptr);
                    
                    gpu.name = device_name;
                    gpu.vendor = vendor_name;
                    gpu.memory_mb = memory_size / (1024 * 1024);
                    gpu.compute_units = compute_units;
                    gpu.max_work_group_size = static_cast<uint32_t>(max_work_group);
                    gpu.max_frequency_mhz = max_freq;
                    
                    // Determine GPU type
                    std::string vendor_lower = vendor_name;
                    std::transform(vendor_lower.begin(), vendor_lower.end(), vendor_lower.begin(), ::tolower);
                    
                    if (vendor_lower.find("amd") != std::string::npos || vendor_lower.find("advanced micro devices") != std::string::npos) {
                        gpu.type = GPUType::OPENCL_AMD;
                    } else if (vendor_lower.find("nvidia") != std::string::npos) {
                        gpu.type = GPUType::OPENCL_NVIDIA;
                    } else if (vendor_lower.find("intel") != std::string::npos) {
                        gpu.type = GPUType::OPENCL_INTEL;
                    } else {
                        gpu.type = GPUType::AUTO_DETECT;
                    }
                    
                    // Calculate performance score
                    gpu.performance_score = compute_units * max_freq * std::log(memory_size / 1024.0);
                    gpu.ai_acceleration_support = (gpu.memory_mb >= 2048); // Require at least 2GB for AI
                    
                    available_gpus_.push_back(gpu);
                    
                    std::cout << "🎮 GPU " << available_gpus_.size() << ": " << gpu.name 
                              << " (" << gpu.vendor << ")" << std::endl;
                    std::cout << "   💾 Memory: " << gpu.memory_mb << " MB" << std::endl;
                    std::cout << "   🔧 Compute Units: " << gpu.compute_units << std::endl;
                    std::cout << "   ⚡ Max Frequency: " << gpu.max_frequency_mhz << " MHz" << std::endl;
                    std::cout << "   🧠 AI Support: " << (gpu.ai_acceleration_support ? "✅" : "❌") << std::endl;
                }
            }
        }
    }
#endif
    
    // Try CUDA detection for NVIDIA GPUs (if supported)
#if defined(CUDA_SUPPORT) && CUDA_SUPPORT
    int cuda_device_count = 0;
    if (cudaGetDeviceCount(&cuda_device_count) == cudaSuccess && cuda_device_count > 0) {
        std::cout << "🟢 Nalezeno " << cuda_device_count << " CUDA zařízení" << std::endl;
        
        for (int i = 0; i < cuda_device_count; i++) {
            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
                // Check if we already have this device from OpenCL
                bool already_added = false;
                for (auto& gpu : available_gpus_) {
                    if (gpu.name.find(prop.name) != std::string::npos) {
                        // Update to CUDA type if better
                        gpu.type = GPUType::CUDA_NVIDIA;
                        already_added = true;
                        break;
                    }
                }
                
                if (!already_added) {
                    GPUDevice gpu;
                    gpu.name = prop.name;
                    gpu.vendor = "NVIDIA";
                    gpu.type = GPUType::CUDA_NVIDIA;
                    gpu.memory_mb = prop.totalGlobalMem / (1024 * 1024);
                    gpu.compute_units = prop.multiProcessorCount;
                    gpu.max_work_group_size = prop.maxThreadsPerBlock;
                    gpu.max_frequency_mhz = prop.clockRate / 1000;
                    gpu.performance_score = prop.multiProcessorCount * (prop.clockRate / 1000) * 
                                          std::log(prop.totalGlobalMem / 1024.0);
                    gpu.ai_acceleration_support = (prop.major >= 7); // Tensor Cores from Volta+
                    
                    available_gpus_.push_back(gpu);
                    
                    std::cout << "🟢 CUDA GPU " << available_gpus_.size() << ": " << gpu.name << std::endl;
                    std::cout << "   💾 Memory: " << gpu.memory_mb << " MB" << std::endl;
                    std::cout << "   🔧 SM Count: " << gpu.compute_units << std::endl;
                    std::cout << "   ⚡ Base Clock: " << gpu.max_frequency_mhz << " MHz" << std::endl;
                    std::cout << "   🧠 Tensor Cores: " << (gpu.ai_acceleration_support ? "✅" : "❌") << std::endl;
                }
            }
        }
    }
#endif

    // If no real GPUs found, create simulated GPU for testing
    if (available_gpus_.empty()) {
        std::cout << "🎮 Žádné OpenCL/CUDA GPU - vytvářím simulované GPU pro test..." << std::endl;
        
        GPUDevice simulated_gpu;
        simulated_gpu.name = "ZION Simulation GPU (CPU Fallback)";
        simulated_gpu.vendor = "ZION Simulation";
        simulated_gpu.type = GPUType::AUTO_DETECT;
        simulated_gpu.memory_mb = 4096; // 4GB simulated
        simulated_gpu.compute_units = 16;
        simulated_gpu.max_work_group_size = 256;
        simulated_gpu.max_frequency_mhz = 1500;
        simulated_gpu.performance_score = 1000.0;
        simulated_gpu.ai_acceleration_support = true;
        
        available_gpus_.push_back(simulated_gpu);
        
        std::cout << "✅ Simulované GPU vytvořeno: " << simulated_gpu.name << std::endl;
    }
    
    // Sort GPUs by performance score
    std::sort(available_gpus_.begin(), available_gpus_.end(), 
              [](const GPUDevice& a, const GPUDevice& b) {
                  return a.performance_score > b.performance_score;
              });
    
    return !available_gpus_.empty();
}

bool ZionGPUMiner::select_best_gpus(int count) {
    selected_gpus_.clear();
    
    if (available_gpus_.empty()) {
        std::cout << "❌ Žádné GPU k dispozici!" << std::endl;
        return false;
    }
    
    int actual_count = std::min(count, static_cast<int>(available_gpus_.size()));
    
    for (int i = 0; i < actual_count; i++) {
        selected_gpus_.push_back(available_gpus_[i]);
        std::cout << "✅ Vybráno GPU: " << available_gpus_[i].name 
                  << " (Score: " << available_gpus_[i].performance_score << ")" << std::endl;
    }
    
    return true;
}

bool ZionGPUMiner::start_gpu_mining(GPUOptimization mode) {
    if (is_mining_.load()) {
        std::cout << "⚠️ GPU mining už běží!" << std::endl;
        return false;
    }
    
    if (selected_gpus_.empty()) {
        if (!select_best_gpus(1)) {
            std::cout << "❌ Žádné GPU nevybráno!" << std::endl;
            return false;
        }
    }
    
    optimization_mode_ = mode;
    is_mining_.store(true);
    
    std::cout << "🚀 Spouštím GPU mining s " << selected_gpus_.size() << " GPU..." << std::endl;
    std::cout << "🎯 Optimalizace: ";
    
    switch (mode) {
        case GPUOptimization::BALANCED: std::cout << "Vyvážená"; break;
        case GPUOptimization::POWER_EFFICIENT: std::cout << "Úsporná"; break;
        case GPUOptimization::MAX_PERFORMANCE: std::cout << "Maximální výkon"; break;
        case GPUOptimization::AI_ENHANCED: std::cout << "AI Enhanced"; break;
        case GPUOptimization::COSMIC_HARMONY: std::cout << "Cosmic Harmony"; break;
    }
    std::cout << std::endl;
    
    // Apply AI enhancement
    enhance_with_gpu_ai();
    
    // Start mining threads for each GPU
    for (size_t i = 0; i < selected_gpus_.size(); i++) {
        if (selected_gpus_[i].type == GPUType::CUDA_NVIDIA) {
            gpu_threads_.emplace_back(&ZionGPUMiner::gpu_mining_thread_cuda, this, i);
        } else {
            gpu_threads_.emplace_back(&ZionGPUMiner::gpu_mining_thread_opencl, this, i);
        }
        
        // Start monitoring thread
        gpu_threads_.emplace_back(&ZionGPUMiner::monitor_gpu_health, this, i);
    }
    
    return true;
}

void ZionGPUMiner::stop_gpu_mining() {
    if (!is_mining_.load()) return;
    
    std::cout << "🛑 Zastavujem GPU mining..." << std::endl;
    is_mining_.store(false);
    
    for (auto& thread : gpu_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    gpu_threads_.clear();
    
    std::cout << "✅ GPU mining zastaven!" << std::endl;
}

void ZionGPUMiner::enhance_with_gpu_ai() {
    std::cout << "🧠 Aktivujem ZION GPU AI Enhancement..." << std::endl;
    
    // AI multiplier based on Euler's number for GPU optimization
    double gpu_cosmic_factor = M_E * ai_gpu_multiplier_.load();
    ai_gpu_multiplier_.store(gpu_cosmic_factor);
    
    // Enhance cosmic GPU level
    cosmic_gpu_level_.store(cosmic_gpu_level_.load() + 12); // Add GPU cosmic wisdom
    
    std::cout << "✨ GPU AI Enhancement aktivní!" << std::endl;
    std::cout << "   🎮 GPU AI Multiplier: " << ai_gpu_multiplier_.load() << std::endl;
    std::cout << "   🌟 Cosmic GPU Level: " << cosmic_gpu_level_.load() << std::endl;
}

void ZionGPUMiner::gpu_mining_thread_opencl(int gpu_index) {
    std::cout << "⚡ OpenCL GPU " << gpu_index << " (" 
              << selected_gpus_[gpu_index].name << ") začíná mining..." << std::endl;
    
    if (!init_opencl(gpu_index)) {
        std::cout << "❌ Nepodařilo se inicializovat OpenCL pro GPU " << gpu_index << std::endl;
        return;
    }
    
    auto last_stats_time = std::chrono::steady_clock::now();
    uint64_t local_hashes = 0;
    uint32_t nonce_base = gpu_index * 1000000; // Unique nonce range per GPU
    
    while (is_mining_.load()) {
        // Simulate GPU mining (in real implementation, this would launch OpenCL kernel)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        
        local_hashes += 50000; // Simulate GPU hashrate
        
        // Simulate finding shares occasionally
        if (local_hashes % 500000 == 0) {
            uint64_t simulated_hash = 0x12345678ABCD0000ULL | (gpu_index << 48);
            
            std::cout << "💎 GPU " << gpu_index << " Share found! Hash: 0x" 
                      << std::hex << simulated_hash << std::dec
                      << ", Nonce: " << nonce_base << std::endl;
            
            stats_.gpu_shares_found.fetch_add(1);
            stats_.gpu_shares_accepted.fetch_add(1);
            
            nonce_base += 1000;
        }
        
        // Update hashrate statistics
        auto current_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(
            current_time - last_stats_time).count();
            
        if (duration >= 10) {
            uint64_t hashrate = local_hashes / std::max(static_cast<int64_t>(duration), static_cast<int64_t>(1));
            stats_.gpu_hashrate.store(hashrate);
            last_stats_time = current_time;
            local_hashes = 0;
        }
    }
    
    cleanup_opencl();
    std::cout << "🔄 OpenCL GPU " << gpu_index << " thread ukončen." << std::endl;
}

void ZionGPUMiner::gpu_mining_thread_cuda(int gpu_index) {
    std::cout << "🟢 CUDA GPU " << gpu_index << " (" 
              << selected_gpus_[gpu_index].name << ") začíná mining..." << std::endl;
    
    if (!init_cuda(gpu_index)) {
        std::cout << "❌ Nepodařilo se inicializovat CUDA pro GPU " << gpu_index << std::endl;
        return;
    }
    
    auto last_stats_time = std::chrono::steady_clock::now();
    uint64_t local_hashes = 0;
    uint32_t nonce_base = gpu_index * 1000000 + 500000; // Unique nonce range per GPU
    
    while (is_mining_.load()) {
        // Simulate CUDA mining (in real implementation, this would launch CUDA kernel)
        std::this_thread::sleep_for(std::chrono::milliseconds(8));
        
        local_hashes += 75000; // CUDA typically faster than OpenCL
        
        // Simulate finding shares with better efficiency
        if (local_hashes % 400000 == 0) {
            uint64_t simulated_hash = 0x9876543210AB0000ULL | (gpu_index << 48);
            
            std::cout << "💎 CUDA GPU " << gpu_index << " Share found! Hash: 0x" 
                      << std::hex << simulated_hash << std::dec
                      << ", Nonce: " << nonce_base << std::endl;
            
            stats_.gpu_shares_found.fetch_add(1);
            stats_.gpu_shares_accepted.fetch_add(1);
            
            nonce_base += 1500;
        }
        
        // Update hashrate statistics  
        auto current_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(
            current_time - last_stats_time).count();
            
        if (duration >= 10) {
            uint64_t hashrate = local_hashes / std::max(static_cast<int64_t>(duration), static_cast<int64_t>(1));
            stats_.gpu_hashrate.store(hashrate);
            last_stats_time = current_time;
            local_hashes = 0;
        }
    }
    
    cleanup_cuda();
    std::cout << "🔄 CUDA GPU " << gpu_index << " thread ukončen." << std::endl;
}

bool ZionGPUMiner::init_opencl(int gpu_index) {
    // Simplified OpenCL initialization for demo
    std::cout << "🔧 Inicializuji OpenCL pro GPU " << gpu_index << "..." << std::endl;
    return true; // Simulate successful initialization
}

bool ZionGPUMiner::init_cuda(int gpu_index) {
    // Simplified CUDA initialization for demo  
    std::cout << "🔧 Inicializuji CUDA pro GPU " << gpu_index << "..." << std::endl;
    return true; // Simulate successful initialization
}

void ZionGPUMiner::cleanup_opencl() {
    // Cleanup OpenCL resources
}

void ZionGPUMiner::cleanup_cuda() {
    // Cleanup CUDA resources  
}

void ZionGPUMiner::monitor_gpu_health(int gpu_index) {
    while (is_mining_.load()) {
        // Simulate GPU monitoring
        double temp = 65.0 + (gpu_index * 5.0) + (rand() % 10 - 5); // Simulate temperature
        uint32_t power = 150 + (gpu_index * 20) + (rand() % 20 - 10); // Simulate power usage
        
        stats_.gpu_temperature.store(temp);
        stats_.gpu_power_usage.store(power);
        
        // Calculate efficiency (hashes per watt)
        double efficiency = stats_.gpu_hashrate.load() / std::max(power, 1u);
        stats_.gpu_efficiency.store(efficiency);
        
        std::this_thread::sleep_for(std::chrono::seconds(30));
    }
}

void ZionGPUMiner::print_gpu_stats() const {
    auto uptime = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - stats_.start_time).count();
    
    std::cout << "\n🎮 === ZION GPU MINING STATISTICS ===" << std::endl;
    std::cout << "⏱️ GPU Uptime: " << uptime << " sekund" << std::endl;
    std::cout << "🔥 GPU Hashrate: " << stats_.gpu_hashrate.load() << " H/s" << std::endl;
    std::cout << "💎 GPU Shares Found: " << stats_.gpu_shares_found.load() << std::endl;
    std::cout << "✅ GPU Shares Accepted: " << stats_.gpu_shares_accepted.load() << std::endl;
    std::cout << "🌡️ GPU Temperature: " << stats_.gpu_temperature.load() << "°C" << std::endl;
    std::cout << "⚡ GPU Power Usage: " << stats_.gpu_power_usage.load() << "W" << std::endl;
    std::cout << "📊 GPU Efficiency: " << stats_.gpu_efficiency.load() << " H/W" << std::endl;
    std::cout << "🧠 AI GPU Multiplier: " << ai_gpu_multiplier_.load() << std::endl;
    std::cout << "🌟 Cosmic GPU Level: " << cosmic_gpu_level_.load() << std::endl;
    std::cout << "===================================\n" << std::endl;
}

// TODO(REAL_GPU): Replace simulated sleep/hash increments with actual GPU kernel launch sequence:
//  1. Prepare per-job seed -> expand to dataset chunk on GPU
//  2. Maintain per-GPU nonce buffer, fill kernel arguments
//  3. Kernel computes N hashes per launch -> writes results + meets mask / diff predicate
//  4. Host scans result buffer, submits valid shares via shared submitter interface
//  5. Adaptive work size: tune global/local sizes based on occupancy heuristics (per vendor)
//  6. Future: persistent kernel model to reduce launch latency