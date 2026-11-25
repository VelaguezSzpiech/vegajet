// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// CPU reference for the expert-grouped routed MoE Down (ops/kernels/vj_moe_down_grouped.hip[h]):
//   * moe_route_group_ref        : the routing-group outputs (counting sort by expert, stable in route index).
//   * moe_down_*_ref             : bit-level CPU models of the device float arithmetic of
//                                  vj_moe_down_dot8_kernel (per token) and of the grouped + combine kernels.
//   * moe_down_bytes             : DRAM byte model of the per-token vs grouped schemes.
// The float models must be compiled without FP contraction (each expression rounds like the device code;
// fmaf is explicit). Pure CPU, no HIP.
#pragma once

#include "ops/kernels/vj_moe_down_grouped.hiph"

#include <cstdint>
#include <vector>

namespace vegajet {

struct MoeRouteGroup {
    uint32_t hdr[moe_grp::kHdrWords] = {0, 0, 0, 0};  // unique, items, grouped routes, R
    std::vector<uint32_t> exp_cnt;   // [256]
    std::vector<uint32_t> exp_off;   // [257]
    std::vector<uint32_t> uniq;      // [hdr[0]]   ascending expert id
    std::vector<uint32_t> items;     // [hdr[1]]   moe_grp::item_pack(e, start, len), ascending e then start
    std::vector<uint32_t> routes;    // [R]        m*8+s, grouped by expert, ascending inside an expert
};

// top_ids: [M*8] (low 8 bits used, like the device). min_group: experts with count >= min_group get items.
void moe_route_group_ref(const int32_t* top_ids, uint32_t M, uint32_t min_group, MoeRouteGroup* out);

// Inputs shared by all Down models (same meaning / layout as the kernels' kernargs).
struct MoeDownProblem {
    const uint8_t* w_qs = nullptr;   uint32_t w_stride = 0;   // [256 * D rows] codes, 16 B per 32-weight block
    const uint8_t* w_sc = nullptr;   uint32_t s_stride = 0;   // [256 * D rows] {u32 nibbles, f32 d} x 2
    const int8_t*  mid_q8 = nullptr;                          // [M][8][512]
    const float*   mid_d = nullptr;                           // [M][8][16]
    const int32_t* top_ids = nullptr;                         // [M][8]
    const float*   top_w = nullptr;                           // [M][8]
    const float*   sh_down_in = nullptr;                      // [M][D] or null
    uint32_t M = 0, D = 0;
};

// Exact int32 block dot of lane `lane16` of (expert e, row) against route r's activation.
int32_t moe_down_block_dp(const MoeDownProblem& p, uint32_t e, uint32_t row, uint32_t lane16, uint32_t r);
// Device gdb expression for (expert e, row, lane16).
float moe_down_block_gdb(const MoeDownProblem& p, uint32_t e, uint32_t row, uint32_t lane16);
// ds_swizzle butterfly (xor 8, 4, 2, 1) as in dot8_dev::sum16; returns lane 0's value.
float moe_sum16_ref(const float v[16]);

// vj_moe_down_dot8_kernel: out[m*D+row].
void moe_down_per_token_ref(const MoeDownProblem& p, float* out);
// vj_moe_down_grouped_dp_kernel: dp[(r*D + row)*16 + lane] for every route of every work item.
void moe_down_grouped_dp_ref(const MoeDownProblem& p, const MoeRouteGroup& g, int32_t* dp);
// vj_moe_down_combine_dp_kernel.
void moe_down_combine_dp_ref(const MoeDownProblem& p, const uint32_t* exp_cnt, uint32_t min_group,
                             const int32_t* dp, float* out);
// vj_moe_down_grouped_f32_kernel: part[r*D + row].
void moe_down_grouped_f32_ref(const MoeDownProblem& p, const MoeRouteGroup& g, float* part);
// vj_moe_down_combine_f32_kernel.
void moe_down_combine_f32_ref(const MoeDownProblem& p, const uint32_t* exp_cnt, uint32_t min_group,
                              const float* part, float* out);

// DRAM byte model (weights, intermediate buffers; activations / outputs are equal in both schemes and are
// L2-resident re-reads in the grouped kernel, so they are left out). Down row = 256 B codes + 16 B scales.
struct MoeDownBytes {
    uint64_t per_token = 0;        // vj_moe_down_dot8_kernel: R * D * 272
    uint64_t grouped_weights = 0;  // grouped kernel weight reads (items * D * (256 | 272))
    uint64_t buf_write = 0;        // dp (64 B) / part (4 B) per wide route-row
    uint64_t combine_reads = 0;    // skinny routes 272 B + wide routes (64 + 16 | 4) B per row
    uint64_t grouped_total() const { return grouped_weights + buf_write + combine_reads; }
};
// exact = true: _dp scheme, false: _f32 scheme. exp_cnt: [256].
MoeDownBytes moe_down_bytes(const uint32_t* exp_cnt, uint32_t D, uint32_t min_group, bool exact);

} // namespace vegajet
