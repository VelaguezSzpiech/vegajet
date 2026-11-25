// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Residency-aware launch geometry for gfx906 (reference for the "planned grids" phase).
//
// planner::plan_gemv_splitk() in core/grid_planner.hpp scores fill as if each CU ran one workgroup at a time.
// Real gfx906 CUs keep several workgroups resident (6 for a 38-VGPR, 256-thread GEMV), so a 384-WG grid on 60 CUs is
// 1.07 resident waves, not 6.4 CU-waves. The planner below works on resident slots:
//
//   slots      = n_cu * resident_per_cu          (resident_per_cu <= occupancy limit from compute_gfx906_occupancy)
//   waves      = ceil(wgs / slots)
//   fill       = wgs / (waves * slots)           (1.0 = no idle slot in any wave)
//   tail_waste = idle slots in the last wave / slots
//
// plan_gemv_launch() searches rows-per-WG x split-K S in {1,2,4,8} x resident cap and returns the best
// deterministic plan. plan_split_kv() does the same for decode attention when the local query heads cannot fill the CUs.
#pragma once

#include <cstdint>
#include <vector>

namespace vegajet::planner {

struct WaveFill {
    uint32_t wgs = 0, slots = 0, waves = 0;
    double fill = 0.0;        // useful slot fraction over all waves
    double tail_waste = 0.0;  // idle fraction of the final wave
};
WaveFill wave_fill(uint32_t wgs, uint32_t slots);

struct GemvLaunchLimits {
    uint32_t vgpr = 38, sgpr = 24, lds_bytes = 0, wg_threads = 256;   // from the kernel's code-object metadata
    std::vector<uint32_t> rows_per_wg = {2, 4, 8, 16};                 // row tilings the kernel family supports
    std::vector<uint32_t> split_k = {1, 2, 4, 8};                      // split factors the reduction supports
    uint32_t min_k_chunk = 256;        // smallest K slice worth a partial sum (elements, multiple of 32)
    uint32_t min_resident_per_cu = 2;  // never cap residency below this (latency hiding for HBM streams)
    uint32_t min_elems_per_wg = 2048;  // rows_per_wg * k_chunk floor, keeps per-WG work above dispatch cost
    double split_penalty = 0.010;      // score cost per extra split (partial-sum write + reduce)
    double residency_penalty = 0.010;  // score cost per resident WG given up below the occupancy limit
    double tail_penalty = 0.020;       // score cost per unit of idle final-wave fraction
};

struct GemvLaunchPlan {
    uint32_t n = 0, k = 0, n_cu = 0;
    uint32_t rows_per_wg = 1, split_k = 1, k_chunk = 0;  // k_chunk: multiple of 32, (S-1)*k_chunk < k <= S*k_chunk
    uint32_t row_wgs = 0, total_wgs = 0;                // total = row_wgs * split_k
    uint32_t resident_per_cu = 1, max_resident_per_cu = 1;
    WaveFill fill;
    double score = 0.0;
    bool valid = false;
};

// Score used by the search (exposed so tests can re-derive the optimum independently).
double gemv_plan_score(const WaveFill& f, uint32_t split_k, uint32_t resident_per_cu, uint32_t max_resident,
                       const GemvLaunchLimits& lim);
GemvLaunchPlan plan_gemv_launch(uint32_t n, uint32_t k, uint32_t n_cu, const GemvLaunchLimits& lim = {});

struct SplitKvPlan {
    uint32_t n_q_local = 0, ctx_tokens = 0, n_cu = 0;
    uint32_t chunks = 1, chunk_tokens = 0, total_wgs = 0;  // one WG per (local q head, KV chunk)
    WaveFill fill;
    double score = 0.0;
};

// Splits the KV sequence when n_q_local heads alone underfill n_cu * resident_per_cu slots.
// chunk_tokens is a multiple of 64 and >= min_chunk_tokens (unless the whole context is one chunk).
SplitKvPlan plan_split_kv(uint32_t n_q_local, uint32_t ctx_tokens, uint32_t n_cu, uint32_t resident_per_cu = 1,
                          uint32_t min_chunk_tokens = 256, uint32_t max_chunks = 64, double merge_penalty = 0.002);

} // namespace vegajet::planner
