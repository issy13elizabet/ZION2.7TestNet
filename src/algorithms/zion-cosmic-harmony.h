#pragma once

#include <cstdint>
#include <cstddef>

namespace ZionAlgorithm {
    // ZION Cosmic Harmony Hash Function
    uint64_t zion_cosmic_harmony_hash(const uint8_t* data, size_t length, uint32_t nonce);
}