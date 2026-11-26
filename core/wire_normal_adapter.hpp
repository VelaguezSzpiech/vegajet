// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Normal-engine adapter for the sealed gfx906 Q8_0/Q4_0 Split-KV pair.
//
// It owns no device memory and performs no allocation: the engine allocates the wire Q, wire K/V
// cache, scratch arena and FP32 merge output once, then calls run() for each eligible layer step.
// run() enqueues (all with queue barriers, so producer -> partition -> merge -> epilogue cannot
// overlap):
//   1. vj_wire_{q8,q4}_kv_kernel  for the cache rows that are not valid in the wire cache yet,
//   2. vj_wire_{q8,q4}_q_kernel   for the current batch's FP32 q_buf rows,
//   3. vj_flash_decode_{q8,q4}_partition,
//   4. vj_flash_decode_merge,
//   5. vj_wire_attn_epilogue_kernel (sigmoid gate + block-32 INT8, the active path's outputs).
//
// A typed fallback is returned for every unsupported shape/storage/budget case and the caller
// must run the active single-pass path. Nothing is counted as exercise unless every dispatch was
// enqueued with a nonzero grid.
#pragma once

#include "core/kfd_runtime.hpp"
#include "ops/flash_decode_splitkv_launch.hpp"
#include "ops/wire_normal_launch.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vegajet::wire_normal {

struct Kernels {
    const kfd::LoadedKernel* q8_q = nullptr;
    const kfd::LoadedKernel* q4_q = nullptr;
    const kfd::LoadedKernel* q8_kv = nullptr;
    const kfd::LoadedKernel* q4_kv = nullptr;
    const kfd::LoadedKernel* epilogue = nullptr;
    const kfd::LoadedKernel* sealed_q8_partition = nullptr;
    const kfd::LoadedKernel* sealed_q4_partition = nullptr;
    const kfd::LoadedKernel* sealed_merge = nullptr;
};

struct Buffers {
    uint64_t native_k = 0; // active engine 272 B K cache base (producer input)
    uint64_t native_v = 0; // active engine 272 B V cache base (producer input)
    uint64_t q_wire = 0;
    uint64_t k_wire = 0;
    uint64_t v_wire = 0;
    uint64_t scratch = 0;
    uint64_t out = 0;
    uint64_t out_lse = 0;
    // Capacities of the allocations above; a plan that does not fit falls back immediately.
    uint32_t cache_rows = 0;
    uint64_t q_bytes = 0;
    uint64_t scratch_bytes = 0;
};

struct CacheState {
    std::vector<uint32_t> produced_rows; // one high-water mark per attention layer

    void reset(size_t layers) { produced_rows.assign(layers, 0u); }
    uint32_t get(size_t layer) const {
        return layer < produced_rows.size() ? produced_rows[layer] : 0u;
    }
    void set(size_t layer, uint32_t rows) {
        if (layer < produced_rows.size()) produced_rows[layer] = rows;
    }
};

constexpr size_t kFallbackCount = static_cast<size_t>(Fallback::LaunchFailed) + 1u;

struct Counters {
    uint64_t attempts = 0;
    uint64_t engaged = 0;
    uint64_t q_dispatches = 0;
    uint64_t kv_dispatches = 0;
    uint64_t partition_dispatches = 0;
    uint64_t merge_dispatches = 0;
    uint64_t epilogue_dispatches = 0;
    uint64_t rows_produced = 0;
    uint64_t fallbacks[kFallbackCount] = {};
    bool printed[kFallbackCount] = {};
    uint32_t last_splits = 0;
    uint32_t last_rows = 0;
    WireFormat last_format = WireFormat::None;
    uint32_t last_kv_begin = 0;
    uint32_t last_kv_end = 0;

    void reset() { *this = Counters{}; }
};

// Returns the decision even on fallback so the caller can report the typed reason. On success the
// dispatches are already enqueued; the caller must not enqueue the active attention as well.
Attempt run(kfd::Runtime& rt, uint32_t gpu, const Kernels& kernels, const Buffers& buffers,
            const Policy& policy, uint32_t layer, uint32_t pos0, uint32_t rows,
            uint64_t q_f32, uint64_t gate_f32, uint64_t out_q8, uint64_t out_ds,
            CacheState& cache, Counters& counters, std::string* error);

// Human-readable one-line summary for the engine's end-of-run report / A-B evidence.
std::string summary(const Counters& counters);

} // namespace vegajet::wire_normal
