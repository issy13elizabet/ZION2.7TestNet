/*
 * ZION GPU Miner - Unified CUDA/OpenCL Interface
 * Automatic detection and optimal usage of NVIDIA/AMD GPUs
 * Author: Maitreya ZionNet Team
 * Date: September 28, 2025
 */

#ifndef ZION_GPU_MINER_H
#define ZION_GPU_MINER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// GPU types
typedef enum {
    GPU_TYPE_UNKNOWN = 0,
    GPU_TYPE_NVIDIA = 1,
    GPU_TYPE_AMD = 2
} gpu_type_t;

// GPU device information
typedef struct {
    int device_id;
    gpu_type_t type;
    char name[256];
    size_t memory_size;
    int compute_units;
    double estimated_hashrate;
    bool is_available;
} zion_gpu_device_t;

// GPU mining statistics
typedef struct {
    uint64_t total_hashes;
    uint32_t accepted_shares;
    uint32_t rejected_shares;
    double current_hashrate;
    double average_hashrate;
    uint64_t uptime_seconds;
    double gpu_temperature;
    double power_usage;
} zion_gpu_stats_t;

// GPU mining configuration
typedef struct {
    int threads_per_gpu;
    int intensity;
    bool enable_auto_tuning;
    int memory_clock_offset;
    int core_clock_offset;
    int power_limit;
    int temperature_limit;
} zion_gpu_config_t;

// Function declarations

// Device detection and initialization
int zion_gpu_detect_devices(zion_gpu_device_t** devices, int* count);
int zion_gpu_init_device(int device_id, gpu_type_t type);
void zion_gpu_cleanup_device(int device_id);
void zion_gpu_cleanup_all();

// Mining functions
uint32_t zion_gpu_mine_hash(
    int device_id,
    uint8_t* block_data,
    uint32_t target_difficulty,
    uint32_t start_nonce,
    uint32_t max_iterations,
    uint32_t* hash_count_out
);

int zion_gpu_start_mining(int device_id, zion_gpu_config_t* config);
void zion_gpu_stop_mining(int device_id);
bool zion_gpu_is_mining(int device_id);

// Statistics and monitoring
int zion_gpu_get_stats(int device_id, zion_gpu_stats_t* stats);
double zion_gpu_get_hashrate(int device_id);
double zion_gpu_get_temperature(int device_id);
int zion_gpu_set_power_limit(int device_id, int power_limit);

// Auto-tuning and optimization
int zion_gpu_auto_tune(int device_id, zion_gpu_config_t* config);
int zion_gpu_benchmark(int device_id, double* hashrate_out);
void zion_gpu_optimize_settings(zion_gpu_device_t* device, zion_gpu_config_t* config);

// Utility functions
const char* zion_gpu_type_string(gpu_type_t type);
void zion_gpu_print_device_info(zion_gpu_device_t* device);
bool zion_gpu_check_compatibility(zion_gpu_device_t* device);

// CUDA-specific functions (when available)
#ifdef ZION_CUDA_SUPPORT
int zion_cuda_init(int device_id);
int zion_cuda_get_device_count();
uint32_t zion_cuda_mine_hash(
    uint8_t* block_data,
    uint32_t target_difficulty,
    uint32_t start_nonce,
    uint32_t max_iterations,
    uint32_t* hash_count_out
);
#endif

// OpenCL-specific functions (when available)
#ifdef ZION_OPENCL_SUPPORT
int zion_opencl_init(int platform_id, int device_id);
int zion_opencl_get_device_count();
uint32_t zion_opencl_mine_hash(
    uint8_t* block_data,
    uint32_t target_difficulty,
    uint32_t start_nonce,
    uint32_t max_iterations,
    uint32_t* hash_count_out
);
void zion_opencl_cleanup();
#endif

#ifdef __cplusplus
}
#endif

#endif // ZION_GPU_MINER_H