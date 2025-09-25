#pragma once

#ifdef GPU_MINING_ENABLED
#include <cstdint>
#include <string>

namespace zion {
namespace gpu {

class AutolykosGPU {
public:
    AutolykosGPU();
    ~AutolykosGPU();
    
    bool initialize(int device_id = 0);
    bool mine(const std::string& header, uint64_t target, uint64_t& nonce);
    void shutdown();
    
private:
    bool initialized_;
    int device_id_;
};

} // namespace gpu
} // namespace zion

#endif // GPU_MINING_ENABLED
