/*
 * ZION GPU Miner - Unified Implementation
 * Automatic NVIDIA/AMD GPU detection and mining
 * Author: Maitreya ZionNet Team
 * Date: September 28, 2025
 */

#include "zion-gpu-miner.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <thread>
#include <vector>
#include <chrono>
#include <atomic>
#include <mutex>

// Global GPU device registry
static std::vector<zion_gpu_device_t> gpu_devices;
static std::vector<std::atomic<bool>> mining_status;
static std::vector<zion_gpu_stats_t> gpu_stats;
static std::mutex gpu_mutex;

// Auto-detect available GPU devices
int zion_gpu_detect_devices(zion_gpu_device_t** devices, int* count) {
    std::lock_guard<std::mutex> lock(gpu_mutex);
    
    gpu_devices.clear();
    int device_count = 0;
    
    printf("[ZION GPU] Detecting available GPU devices...\n");
    
    // Try to detect NVIDIA CUDA devices
#ifdef ZION_CUDA_SUPPORT
    int cuda_count = zion_cuda_get_device_count();
    for (int i = 0; i < cuda_count; i++) {
        zion_gpu_device_t device = {};
        device.device_id = device_count;
        device.type = GPU_TYPE_NVIDIA;
        
        // Initialize CUDA to get device info
        if (zion_cuda_init(i) == 0) {
            snprintf(device.name, sizeof(device.name), "NVIDIA GPU #%d", i);
            device.memory_size = 8ULL * 1024 * 1024 * 1024; // Default 8GB
            device.compute_units = 32; // Estimate
            device.estimated_hashrate = 50000000.0; // 50 MH/s estimate
            device.is_available = true;
            
            gpu_devices.push_back(device);
            device_count++;
            
            printf("[ZION GPU] Found NVIDIA GPU: %s\n", device.name);
        }
    }
#endif
    
    // Try to detect AMD OpenCL devices
#ifdef ZION_OPENCL_SUPPORT
    int opencl_count = zion_opencl_get_device_count();
    for (int i = 0; i < opencl_count; i++) {
        zion_gpu_device_t device = {};
        device.device_id = device_count;
        device.type = GPU_TYPE_AMD;
        
        // Initialize OpenCL to get device info
        if (zion_opencl_init(0, i) == 0) {
            snprintf(device.name, sizeof(device.name), "AMD GPU #%d", i);
            device.memory_size = 8ULL * 1024 * 1024 * 1024; // Default 8GB
            device.compute_units = 64; // AMD typically has more CUs
            device.estimated_hashrate = 45000000.0; // 45 MH/s estimate
            device.is_available = true;
            
            gpu_devices.push_back(device);
            device_count++;
            
            printf("[ZION GPU] Found AMD GPU: %s\n", device.name);
        }
    }
#endif
    
    // Initialize mining status and stats vectors
    mining_status.resize(device_count);
    gpu_stats.resize(device_count);
    
    for (int i = 0; i < device_count; i++) {
        mining_status[i] = false;
        memset(&gpu_stats[i], 0, sizeof(zion_gpu_stats_t));
    }
    
    *count = device_count;
    if (device_count > 0) {
        *devices = gpu_devices.data();
        printf("[ZION GPU] Total GPU devices detected: %d\n", device_count);
        return 0;
    } else {
        printf("[ZION GPU] No compatible GPU devices found\n");
        *devices = nullptr;
        return -1;
    }
}

// Initialize specific GPU device
int zion_gpu_init_device(int device_id, gpu_type_t type) {
    if (device_id >= (int)gpu_devices.size()) {
        printf("[ZION GPU] Invalid device ID: %d\n", device_id);
        return -1;
    }
    
    printf("[ZION GPU] Initializing device %d (%s)...\n", 
           device_id, zion_gpu_type_string(type));
    
    switch (type) {
#ifdef ZION_CUDA_SUPPORT
        case GPU_TYPE_NVIDIA:
            return zion_cuda_init(device_id);
#endif
            
#ifdef ZION_OPENCL_SUPPORT
        case GPU_TYPE_AMD:
            return zion_opencl_init(0, device_id);
#endif
            
        default:
            printf("[ZION GPU] Unsupported GPU type\n");
            return -1;
    }
}

// Unified GPU mining function
uint32_t zion_gpu_mine_hash(
    int device_id,
    uint8_t* block_data,
    uint32_t target_difficulty,
    uint32_t start_nonce,
    uint32_t max_iterations,
    uint32_t* hash_count_out
) {
    if (device_id >= (int)gpu_devices.size()) {
        printf("[ZION GPU] Invalid device ID: %d\n", device_id);
        return 0;
    }
    
    zion_gpu_device_t* device = &gpu_devices[device_id];
    
    switch (device->type) {
#ifdef ZION_CUDA_SUPPORT
        case GPU_TYPE_NVIDIA:
            return zion_cuda_mine_hash(block_data, target_difficulty, 
                                       start_nonce, max_iterations, hash_count_out);
#endif
            
#ifdef ZION_OPENCL_SUPPORT
        case GPU_TYPE_AMD:
            return zion_opencl_mine_hash(block_data, target_difficulty, 
                                         start_nonce, max_iterations, hash_count_out);
#endif
            
        default:
            printf("[ZION GPU] Unsupported GPU type for mining\n");
            return 0;
    }
}

// Start mining on specific GPU
int zion_gpu_start_mining(int device_id, zion_gpu_config_t* config) {
    if (device_id >= (int)gpu_devices.size()) {
        return -1;
    }
    
    mining_status[device_id] = true;
    
    // Reset stats
    memset(&gpu_stats[device_id], 0, sizeof(zion_gpu_stats_t));
    gpu_stats[device_id].uptime_seconds = 0;
    
    printf("[ZION GPU] Started mining on device %d\n", device_id);
    return 0;
}

// Stop mining on specific GPU
void zion_gpu_stop_mining(int device_id) {
    if (device_id < (int)gpu_devices.size()) {
        mining_status[device_id] = false;
        printf("[ZION GPU] Stopped mining on device %d\n", device_id);
    }
}

// Check if GPU is currently mining
bool zion_gpu_is_mining(int device_id) {
    if (device_id >= (int)gpu_devices.size()) {
        return false;
    }
    return mining_status[device_id];
}

// Get mining statistics for specific GPU
int zion_gpu_get_stats(int device_id, zion_gpu_stats_t* stats) {
    if (device_id >= (int)gpu_devices.size() || !stats) {
        return -1;
    }
    
    *stats = gpu_stats[device_id];
    return 0;
}

// Get current hashrate for specific GPU
double zion_gpu_get_hashrate(int device_id) {
    if (device_id >= (int)gpu_devices.size()) {
        return 0.0;
    }
    
    return gpu_stats[device_id].current_hashrate;
}

// Benchmark GPU performance
int zion_gpu_benchmark(int device_id, double* hashrate_out) {
    if (device_id >= (int)gpu_devices.size()) {
        return -1;
    }
    
    printf("[ZION GPU] Benchmarking device %d...\n", device_id);
    
    // Prepare test data
    uint8_t test_block[32] = {0};
    uint32_t test_target = 0xFFFFFFFF; // Easy target for benchmark
    uint32_t hash_count = 0;
    
    // Time the mining operation
    auto start_time = std::chrono::high_resolution_clock::now();
    
    zion_gpu_mine_hash(device_id, test_block, test_target, 0, 100000, &hash_count);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Calculate hashrate (hashes per second)
    double seconds = duration.count() / 1000000.0;
    *hashrate_out = hash_count / seconds;
    
    printf("[ZION GPU] Device %d benchmark: %.2f MH/s\n", 
           device_id, *hashrate_out / 1000000.0);
    
    return 0;
}

// Auto-tune GPU settings for optimal performance
int zion_gpu_auto_tune(int device_id, zion_gpu_config_t* config) {
    if (device_id >= (int)gpu_devices.size() || !config) {
        return -1;
    }
    
    zion_gpu_device_t* device = &gpu_devices[device_id];
    
    printf("[ZION GPU] Auto-tuning device %d...\n", device_id);
    
    // Set optimal defaults based on GPU type
    switch (device->type) {
        case GPU_TYPE_NVIDIA:
            config->threads_per_gpu = 256;
            config->intensity = 20;
            config->enable_auto_tuning = true;
            config->memory_clock_offset = 500;
            config->core_clock_offset = 100;
            config->power_limit = 85;
            config->temperature_limit = 83;
            break;
            
        case GPU_TYPE_AMD:
            config->threads_per_gpu = 256;
            config->intensity = 18;
            config->enable_auto_tuning = true;
            config->memory_clock_offset = 900;
            config->core_clock_offset = 50;
            config->power_limit = 90;
            config->temperature_limit = 85;
            break;
            
        default:
            return -1;
    }
    
    // Run quick benchmarks to find optimal settings
    double best_hashrate = 0.0;
    int best_intensity = config->intensity;
    
    for (int intensity = 16; intensity <= 22; intensity++) {
        config->intensity = intensity;
        
        double hashrate;
        if (zion_gpu_benchmark(device_id, &hashrate) == 0) {
            if (hashrate > best_hashrate) {
                best_hashrate = hashrate;
                best_intensity = intensity;
            }
        }
    }
    
    config->intensity = best_intensity;
    device->estimated_hashrate = best_hashrate;
    
    printf("[ZION GPU] Auto-tune complete. Optimal intensity: %d, Hashrate: %.2f MH/s\n", 
           best_intensity, best_hashrate / 1000000.0);
    
    return 0;
}

// Utility function to get GPU type string
const char* zion_gpu_type_string(gpu_type_t type) {
    switch (type) {
        case GPU_TYPE_NVIDIA: return "NVIDIA";
        case GPU_TYPE_AMD: return "AMD";
        default: return "Unknown";
    }
}

// Print detailed device information
void zion_gpu_print_device_info(zion_gpu_device_t* device) {
    if (!device) return;
    
    printf("=== GPU Device %d ===\n", device->device_id);
    printf("Name: %s\n", device->name);
    printf("Type: %s\n", zion_gpu_type_string(device->type));
    printf("Memory: %.1f GB\n", device->memory_size / (1024.0 * 1024.0 * 1024.0));
    printf("Compute Units: %d\n", device->compute_units);
    printf("Estimated Hashrate: %.2f MH/s\n", device->estimated_hashrate / 1000000.0);
    printf("Available: %s\n", device->is_available ? "Yes" : "No");
    printf("====================\n");
}

// Check GPU compatibility with ZION mining
bool zion_gpu_check_compatibility(zion_gpu_device_t* device) {
    if (!device || !device->is_available) {
        return false;
    }
    
    // Check minimum memory requirement (2GB)
    if (device->memory_size < 2ULL * 1024 * 1024 * 1024) {
        printf("[ZION GPU] Device %d has insufficient memory (< 2GB)\n", device->device_id);
        return false;
    }
    
    // Check minimum compute units
    if (device->compute_units < 8) {
        printf("[ZION GPU] Device %d has insufficient compute units (< 8)\n", device->device_id);
        return false;
    }
    
    return true;
}

// Cleanup all GPU resources
void zion_gpu_cleanup_all() {
    std::lock_guard<std::mutex> lock(gpu_mutex);
    
    // Stop all mining
    for (size_t i = 0; i < mining_status.size(); i++) {
        mining_status[i] = false;
    }
    
#ifdef ZION_OPENCL_SUPPORT
    zion_opencl_cleanup();
#endif
    
    gpu_devices.clear();
    mining_status.clear();
    gpu_stats.clear();
    
    printf("[ZION GPU] All GPU resources cleaned up\n");
}