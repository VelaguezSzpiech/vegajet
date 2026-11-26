// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Host launcher + opt-in switch for the separate 240-WG router / shared Gate+Up kernel pair
// (ops/kernels/vj_ornith_router_s240.hip). Header-only so the engine integration stays a call site.
//
// Opt-in is read once per process:
//   VEGAJET_ROUTER_S240=1        enable the kernel pair instead of the fused/router M=1 front.
//                                Takes effect on the per-layer M=1 path; the M=1 super-kernel path is
//                                selected first, so run engine A/B with VEGAJET_ENABLE_SUPER=0.
//   VEGAJET_ROUTER_S240_SPLIT=N  deterministic Split-K factor, 1..8 (default 1: no extra dispatch effect,
//                                bit-identical to the existing single-pass accumulation)
//   VEGAJET_ROUTER_S240_XOR=bits 1 = router FP16 plane XOR, 2 = shared Gate/Up qs plane XOR, 3 = both.
//                                XOR requires weights packed by core/router_s240_pack.hpp with the same
//                                PlaneLayout. tests/vegajet_layer_test.cpp honors it by repacking the fixture
//                                weights; the engine path keeps 0 because prefill/GEMM readers share the planes.
#pragma once

#include "core/kfd_runtime.hpp"
#include "core/router_s240_layout.hpp"

#include <cstdlib>
#include <string>

namespace vegajet {
namespace ornith {

struct RouterS240Options {
    bool enabled = false;
    uint32_t split_k = 1u;
    uint32_t xor_bits = 0u;
};

inline RouterS240Options router_s240_options_from_env() {
    RouterS240Options o;
    const char* e = std::getenv("VEGAJET_ROUTER_S240");
    o.enabled = (e != nullptr) && !(e[0] == '0' && e[1] == '\0');
    if (!o.enabled) return o;
    if (const char* s = std::getenv("VEGAJET_ROUTER_S240_SPLIT")) {
        const long v = std::strtol(s, nullptr, 10);
        if (v >= 1 && v <= static_cast<long>(router_s240::kMaxSplitK)) o.split_k = static_cast<uint32_t>(v);
    }
    if (const char* x = std::getenv("VEGAJET_ROUTER_S240_XOR")) {
        const long v = std::strtol(x, nullptr, 10);
        if (v >= 0 && v <= 3) o.xor_bits = static_cast<uint32_t>(v);
    }
    return o;
}

struct RouterS240Scratch {
    uint64_t gate_part = 0;
    uint64_t rtr_part = 0;
    uint64_t sh_part = 0;
};

// Carves the three FP32 partial arrays out of one zero-filled scratch allocation.
inline bool router_s240_scratch(uint64_t base, uint64_t bytes_available, uint32_t sh_rows, uint32_t split_k,
                                RouterS240Scratch* out, std::string* err) {
    if (base == 0u) {
        if (err) *err = "router_s240: null scratch base";
        return false;
    }
    if (split_k < 1u || split_k > router_s240::kMaxSplitK) {
        if (err) *err = "router_s240: split_k outside 1..8";
        return false;
    }
    const router_s240::ScratchPlan p = router_s240::make_scratch_plan(sh_rows, split_k);
    if (p.bytes > bytes_available) {
        if (err) *err = "router_s240: scratch needs " + std::to_string(p.bytes) + " B, have " +
                        std::to_string(bytes_available) + " B";
        return false;
    }
    out->gate_part = base + p.gate_off;
    out->rtr_part = base + p.rtr_off;
    out->sh_part = base + p.sh_off;
    return true;
}

// Phase 1 always runs on exactly 240 WGs; phase 2 is the deterministic fixed-order reduction.
inline bool launch_router_s240(kfd::Runtime& rt, size_t gpu, const kfd::LoadedKernel& phase1,
                               const kfd::LoadedKernel& reduce, const router_s240::Phase1Args& pa,
                               const router_s240::ReduceArgs& ra, std::string* err) {
    if (!rt.enqueue_kernel(gpu, phase1, router_s240::kPhase1Wgs, 1, 1, router_s240::kThreads, 1, 1, &pa,
                           sizeof(pa), true)) {
        if (err) *err = "router_s240: phase-1 enqueue failed";
        return false;
    }
    const uint32_t total = ra.sh_rows + router_s240::kRouterRows + 1u;
    const uint32_t grid = (total + router_s240::kThreads - 1u) / router_s240::kThreads;
    if (!rt.enqueue_kernel(gpu, reduce, grid, 1, 1, router_s240::kThreads, 1, 1, &ra, sizeof(ra), true)) {
        if (err) *err = "router_s240: reduce enqueue failed";
        return false;
    }
    return true;
}

} // namespace ornith
} // namespace vegajet
