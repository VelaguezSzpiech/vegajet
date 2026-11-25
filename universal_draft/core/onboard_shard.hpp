#pragma once

#include "core/universal_ir.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace universal::onboard {
struct Device { std::string name; uint64_t vram_bytes=0; double speed=1.0; };
struct Profile {
    std::string name;
    uint64_t parameters=0, layers=0;
    double bits_per_weight=5.0;      // includes quant scales/metadata
    uint32_t context_tokens=4096;
    uint32_t heads=0, kv_heads=0, head_dim=0;
    uint32_t kv_block_bytes[3]={34,18,14}; // per 32 scalar values, including scales
};
struct GpuSlice {
    uint64_t weight_bytes=0, kv_bytes=0, arena_bytes=0, resident_bytes=0;
    uint32_t q_head_begin=0, q_head_end=0, kv_head_begin=0, kv_head_end=0;
    double sequence_fraction=0.0;
    uint32_t kv_tier=0;
};
struct Plan {
    bool feasible=false;
    bool full_resident=true;
    uint64_t model_weight_bytes=0, kv_bytes=0;
    std::string error;
    std::vector<GpuSlice> gpus;
};

// Uses the declared full-model profile, bounded below by known compiler tensor
// payloads. Tiny compiler fixtures must use a separate, tiny Profile.
uint64_t full_weight_bytes(const Model&, const Profile&);
Plan plan(const Model&, const Profile&, const std::vector<Device>&);

} // namespace universal::onboard
