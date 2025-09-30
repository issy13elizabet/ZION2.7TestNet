/*
 * ZION Mining Engine - MIT Licensed
 * Copyright (c) 2025 Maitreya-ZionNet
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <functional>

namespace zion::mining {

/**
 * ZION Mining Engine
 * 
 * Multi-algorithm, multi-device mining engine inspired by open source
 * solutions but designed specifically for ZION Cosmic Harmony algorithm.
 * 
 * Features:
 * - CPU Mining: ZION Cosmic Harmony + RandomX fallback
 * - GPU Mining: OpenCL/CUDA support for AMD/NVIDIA
 * - Pool Protocol: Stratum v1/v2 compatible
 * - Zero Dependencies: Self-contained with minimal external libs
 */
class ZionMiningEngine {
public:
    enum class Algorithm {
        ZION_COSMIC_HARMONY,
        RANDOMX_FALLBACK,
        AUTO_DETECT
    };
    
    enum class DeviceType {
        CPU,
        GPU_OPENCL,
        GPU_CUDA,
        HYBRID
    };
    
    struct MiningConfig {
        std::string pool_host = "127.0.0.1";
        int pool_port = 3333;
        std::string wallet_address;
        std::string worker_name = "zion-miner";
        
        Algorithm algorithm = Algorithm::AUTO_DETECT;
        DeviceType device_type = DeviceType::CPU;
        
        int cpu_threads = 0; // 0 = auto-detect
        std::vector<int> gpu_devices; // empty = auto-detect
        
        bool enable_logging = true;
        int log_level = 1; // 0=error, 1=info, 2=debug
        
        // Performance tuning
        int difficulty_target = 1000;
        int submit_stale_threshold = 5000; // ms
        bool optimize_for_mobile = false;
        
        // Advanced options
        bool enable_opencl = true;
        bool enable_cuda = true;
        bool enable_cpu_affinity = false;
        std::string opencl_platform;
    };
    
    struct HashingStats {
        std::atomic<uint64_t> total_hashes{0};
        std::atomic<uint64_t> accepted_shares{0};
        std::atomic<uint64_t> rejected_shares{0};
        std::atomic<uint64_t> stale_shares{0};
        std::atomic<double> current_hashrate{0.0};
        std::atomic<uint64_t> uptime_seconds{0};
        
        // Device-specific stats
        std::vector<double> cpu_hashrates;
        std::vector<double> gpu_hashrates;
        std::vector<double> gpu_temperatures;
        std::vector<double> gpu_power_usage;
    };
    
    // Callback types
    using ShareCallback = std::function<void(bool accepted, const std::string& error)>;
    using StatsCallback = std::function<void(const HashingStats& stats)>;
    using LogCallback = std::function<void(int level, const std::string& message)>;

public:
    ZionMiningEngine();
    ~ZionMiningEngine();
    
    // Core functionality
    bool initialize(const MiningConfig& config);
    bool start_mining();
    bool stop_mining();
    bool is_mining() const;
    
    // Configuration
    void set_config(const MiningConfig& config);
    const MiningConfig& get_config() const;
    
    // Statistics
    HashingStats get_stats() const;
    double get_current_hashrate() const;
    uint64_t get_total_hashes() const;
    
    // Callbacks
    void set_share_callback(ShareCallback callback);
    void set_stats_callback(StatsCallback callback);
    void set_log_callback(LogCallback callback);
    
    // Device detection
    std::vector<std::string> detect_cpu_features() const;
    std::vector<std::string> detect_gpu_devices() const;
    std::vector<std::string> detect_opencl_devices() const;
    
    // Algorithm support
    bool supports_algorithm(Algorithm algo) const;
    Algorithm detect_best_algorithm() const;
    
    // Pool connectivity
    bool test_pool_connection() const;
    std::string get_pool_info() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * ZION Cosmic Harmony Algorithm Implementation
 * 
 * Custom proof-of-work algorithm optimized for:
 * - CPU efficiency on modern processors (AVX2, AVX-512)
 * - GPU efficiency on AMD/NVIDIA architectures
 * - Memory-hard properties for ASIC resistance
 * - Quantum-resistant cryptographic primitives
 */
class CosmicHarmonyHasher {
public:
    struct HashInput {
        std::vector<uint8_t> block_header;
        uint64_t nonce;
        uint64_t timestamp;
        std::string extra_data;
    };
    
    struct HashResult {
        std::vector<uint8_t> hash;
        uint64_t difficulty;
        double computation_time;
        bool meets_target;
    };
    
public:
    CosmicHarmonyHasher();
    ~CosmicHarmonyHasher();
    
    bool initialize();
    HashResult compute_hash(const HashInput& input, uint64_t target_difficulty) const;
    bool verify_hash(const HashInput& input, const HashResult& result) const;
    
    // Performance optimization
    void enable_cpu_optimizations();
    void enable_gpu_optimizations(DeviceType device_type);
    
    // Algorithm parameters
    void set_memory_requirement(size_t bytes);
    void set_iteration_count(uint32_t iterations);
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * Multi-Device Mining Controller
 * 
 * Coordinates mining across multiple devices:
 * - Load balancing between CPU and GPU
 * - Thermal management and power optimization
 * - Automatic failover and device health monitoring
 */
class MultiDeviceController {
public:
    struct DeviceInfo {
        std::string device_name;
        DeviceType type;
        int device_id;
        bool is_available;
        double temperature;
        double power_usage;
        double current_hashrate;
        uint64_t total_hashes;
    };

public:
    MultiDeviceController();
    ~MultiDeviceController();
    
    std::vector<DeviceInfo> enumerate_devices() const;
    bool enable_device(int device_id, const MiningConfig& config);
    bool disable_device(int device_id);
    
    void set_thermal_limits(double max_cpu_temp, double max_gpu_temp);
    void set_power_limits(double max_cpu_watts, double max_gpu_watts);
    
    std::vector<DeviceInfo> get_device_stats() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * Stratum Protocol Implementation
 * 
 * Full-featured stratum client for mining pool connectivity:
 * - Stratum v1 and v2 protocol support
 * - Automatic reconnection and failover
 * - Difficulty adjustment and job management
 * - Share submission and validation
 */
class StratumClient {
public:
    struct JobInfo {
        std::string job_id;
        std::vector<uint8_t> block_header;
        uint64_t target_difficulty;
        uint64_t timestamp;
        std::string extra_nonce;
    };
    
    using JobCallback = std::function<void(const JobInfo& job)>;
    using DifficultyCallback = std::function<void(uint64_t new_difficulty)>;

public:
    StratumClient();
    ~StratumClient();
    
    bool connect(const std::string& host, int port);
    bool authenticate(const std::string& username, const std::string& password);
    bool subscribe();
    void disconnect();
    
    bool submit_share(const std::string& job_id, uint64_t nonce, const std::vector<uint8_t>& hash);
    
    void set_job_callback(JobCallback callback);
    void set_difficulty_callback(DifficultyCallback callback);
    
    bool is_connected() const;
    std::string get_connection_info() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace zion::mining