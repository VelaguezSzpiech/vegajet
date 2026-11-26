// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Selectable dequant-once-to-LDS routed MoE prefill for gfx906 on ACTUAL IQ2_S / IQ3_XXS packed bytes
// (the production merged Gate+Up plane and the compact Down split plane, both lossless re-layouts of
// ggml block bytes; see ops/kernels/vj_prefill_iq_lds_math.hpp).
//
// Two work-item shapes, both on the same packed bytes and the same output contracts:
//
//  A. THE REQUESTED SINGLE 64x64x64 ITEM (`..._gu_iq2s_lds64x64_kernel`, `..._down_iq3_lds64x64_kernel`):
//     one work item = one expert x 64 weight rows x 64 routed tokens, K advanced in steps of 64
//     elements (two 32-element IQ groups). One 512-thread work-group = two 256-thread row cohorts;
//     cohort c owns rows c*32..c*32+31 (its own 32-row LDS slice, also exactly one block-32 quant
//     group of local_ff) and both cohorts share the activation tile staged cooperatively by the whole
//     work-group behind common barriers. 16 FP32 accumulators per thread (1 row x 8 tokens x
//     {gate, up}), ALL 64 tokens in ONE pass, so each of the 512 decode units of a step is staged
//     once and consumed by all 64 routed tokens: exactly one dequantisation per weight per work item,
//     4x fewer decode units than the 16-token production tiles (CPU model, D = 3072).
//  B. THE 32-ROW DIAGNOSTIC TILE (`..._gu_iq2s_lds32x64_kernel`, `..._down_iq3_lds32x64_kernel`),
//     kept selectable (`--pf-moe-iq-lds32`) for A/B: one 256-thread work-group per 32 rows x 64
//     tokens, two 32-token passes, 8 FP32 accumulators per thread (measured 62/61 VGPR, 0 B private).
//
//   * Gate+Up (`vj_prefill_moe_gu_iq2s_lds32x64_kernel`): 256 threads = 32 rows x 8 (ib, matrix, half)
//     decode units per step. Output contract identical to vj_prefill_moe_gu_iq2s_tiled_kernel:
//     mid_q8[route][local_ff] INT8 + mid_d[route][local_ff/32] FP32 block scales, y = silu(g) * u.
//   * Down (`vj_prefill_moe_down_iq3_lds32x64_kernel`): 64 threads decode 32 rows x 2 compact-plane lanes
//     (64 weights) with 16 grid lookups per row-step; output contract identical to
//     vj_prefill_moe_down_iq3_tiled_kernel: part_out[route * D + row] FP32 (one 32-block dot chain +
//     one fma per block), combined by vj_moe_down_combine_all_f32_kernel.
//
// Accumulation order matches production: one int32 v_dot4 chain per production dot group (4 dwords for
// each 16-weight IQ2_S half block, 8 dwords for each 32-weight IQ3_XXS block) and one FP32 fma per
// group with (weight scale * activation scale), in increasing K order.
//
// LDS (measured with the pinned compiler): shape A GU 18,704 B / Down 12,048 B -> 2 work-groups/CU at
// 512 threads and 64 VGPR (2 x 18,704 <= 65,536 B); shape B GU 9,616 B / Down 6,288 B -> 4
// work-groups/CU. Persistent work-stealing queue (queue[0] items, queue[16] drain counter) exactly
// like the tiled production kernels: any grid size is correct; 240 WGs (x 512 threads for shape A)
// are recommended, counted as wavefronts/BAR0 blocks by the host. __launch_bounds__(512, 2) for shape
// A and __launch_bounds__(256, 4) for shape B, both with amdgpu_num_vgpr(64), hold the tiled gate:
// measured VGPR 63/63 (A) and 62/61 (B), private_segment_fixed_size 0 B on all four kernels. Every
// global read is a register-targeted MUBUF load written to LDS explicitly; no load carries the `lds`
// modifier.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#include "core/ggml_iq_tables.hpp"
#include "ops/kernels/vj_prefill_iq_lds_math.hpp"

namespace vegajet {
namespace il_moe {

using namespace vegajet::il_lds;

__device__ __forceinline__ __amdgpu_buffer_rsrc_t il_rsrc(const void* p) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(p), short(0), 0x7fffffff, 0x00027000);
}
// LDS tile traffic only: wait for outstanding MUBUF results and LDS writes, then cross the work-group
// barrier. Nothing is published outside the work-group, so no system-scope fence is required.
__device__ __forceinline__ void il_barrier() {
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
    __syncthreads();
}
__device__ __forceinline__ uint4 il_ld128(const uint32_t* p) { return *reinterpret_cast<const uint4*>(p); }
__device__ __forceinline__ void il_st128(uint32_t* p, uint4 v) { *reinterpret_cast<uint4*>(p) = v; }

// Exclusive prefix of work items per expert (Hillis-Steele over 256 experts, old values only). uint16
// is safe: total items <= (routes/64 + 256) * row_tiles and routes <= 4096.
__device__ __forceinline__ void il_prefix(const uint32_t* exp_cnt, uint32_t rt, uint16_t* pref, uint32_t tid) {
    pref[tid + 1u] = static_cast<uint16_t>(il_items_per_expert(exp_cnt[tid], rt));
    il_barrier();
    #pragma unroll
    for (uint32_t step = 1u; step < 256u; step <<= 1) {
        const uint32_t add = (tid >= step) ? pref[tid + 1u - step] : 0u;
        il_barrier();
        pref[tid + 1u] = static_cast<uint16_t>(pref[tid + 1u] + add);
        il_barrier();
    }
    if (tid == 0u) pref[0] = 0u;
    il_barrier();
}

__device__ __forceinline__ void il_queue_drain(uint32_t* queue) {
    if (threadIdx.x == 0u) {
        __threadfence();
        const uint32_t done = atomicAdd(&queue[16], 1u) + 1u;
        if (done == gridDim.x) {
            __atomic_store_n(&queue[0], 0u, __ATOMIC_RELEASE);
            __atomic_store_n(&queue[16], 0u, __ATOMIC_RELEASE);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Gate+Up on the merged IQ2_S plane: 32 rows x 64 tokens x K step 64 (2 x 32-element groups).
// ---------------------------------------------------------------------------------------------
struct GuSmem {
    alignas(16) uint32_t w[2][IL_ROWS * IL_WS];  //  5,120 B decoded INT8 (gate, up)
    alignas(16) uint32_t x[IL_TPASS * IL_WS];    //  2,560 B INT8 activations of the current pass
    float wsc[2][IL_ROWS * 4u];                  //  1,024 B one scale per 16-weight group of the step
    float xsc[IL_TPASS * 2u];                    //    256 B two 32-element activation scales per step
    uint16_t pref[257];                          //    514 B exclusive work items per expert
    uint32_t ms[IL_TPASS];                       //    128 B route ids of the current pass
    uint32_t item;                               //      4 B
};
static_assert(sizeof(GuSmem) <= 16384u, "Gate+Up LDS must keep 4 work-groups/CU");

} // namespace il_moe
} // namespace vegajet

extern "C" __global__ __launch_bounds__(256, 4) __attribute__((amdgpu_num_vgpr(64)))
void vj_prefill_moe_gu_iq2s_lds32x64_kernel(
    const uint4*  __restrict__ wgu_merged,
    const int8_t* __restrict__ x_q8,
    const float2* __restrict__ x_ds,
    const uint32_t* __restrict__ exp_off,
    const uint32_t* __restrict__ exp_cnt,
    const uint32_t* __restrict__ sorted_ms,
    int8_t*       __restrict__ mid_q8,
    float*        __restrict__ mid_d,
    uint32_t*     __restrict__ queue,
    uint32_t local_ff,
    uint32_t D) {
    using namespace vegajet::il_moe;
    using namespace vegajet::il_lds;
    __shared__ GuSmem sm;
    const uint32_t tid = threadIdx.x;
    const uint32_t nib = D >> 5;                       // 32-element groups along K
    if (!local_ff || (local_ff & 31u) || !nib || (nib & 7u) || D != nib * 32u) return;

    const uint32_t rt_cnt = il_row_tiles(local_ff);    // 32-row column tiles per expert
    const uint32_t step_cnt = nib >> 1;                // 64-element steps
    il_prefix(exp_cnt, rt_cnt, sm.pref, tid);
    const uint32_t total = __builtin_amdgcn_readfirstlane(sm.pref[256]);

    // Decode role: row = tid & 31, unit = tid >> 5 = (ib-in-step << 2) | (matrix << 1) | half
    // (32 rows x 8 units = the 256 decode units of one 64-weight step).
    const uint32_t drow = tid & 31u;
    const uint32_t unit = tid >> 5;
    const uint32_t ibs  = unit >> 2;
    const uint32_t mat  = (unit >> 1) & 1u;
    const uint32_t half = unit & 1u;
    // Compute role: one row per thread (rc IS the block-32 row) and four tokens per pass.
    const uint32_t rc = tid & 31u;
    const uint32_t tc = tid >> 5;
    const __amdgpu_buffer_rsrc_t r_x = il_rsrc(x_q8);
    const __amdgpu_buffer_rsrc_t r_s = il_rsrc(x_ds);
    const uint64_t* grid = vegajet::ggml_tables::iq2s_grid;

    #pragma nounroll
    for (;;) {
        if (tid == 0u) sm.item = atomicAdd(&queue[0], 1u);
        il_barrier();
        const uint32_t item = __builtin_amdgcn_readfirstlane(sm.item);
        if (item >= total) break;

        uint32_t lo = 0u, hi = 256u;
        while (hi - lo > 1u) {
            const uint32_t mid = (lo + hi) >> 1;
            if (__builtin_amdgcn_readfirstlane(sm.pref[mid]) <= item) lo = mid; else hi = mid;
        }
        const uint32_t e     = lo;
        const uint32_t local = item - __builtin_amdgcn_readfirstlane(sm.pref[e]);
        const uint32_t cnt   = __builtin_amdgcn_readfirstlane(exp_cnt[e]);
        const uint32_t off   = __builtin_amdgcn_readfirstlane(exp_off[e]);
        const uint32_t tt    = local / rt_cnt;
        const uint32_t rt    = local - tt * rt_cnt;
        const uint32_t t0    = tt * IL_TOKENS;
        const uint32_t tcnt  = (cnt - t0 < IL_TOKENS) ? (cnt - t0) : uint32_t(IL_TOKENS);
        // Per-item base so every thread offset stays a 32-bit voffset (rows are up to ~2 KiB each).
        const uint32_t row_bytes = il_gu_row_bytes(nib);
        const uint32_t v_row = drow * row_bytes;
        const uint32_t v_q   = v_row + mat * 8u + half * 4u;
        const uint32_t v_hs  = v_row;
        const __amdgpu_buffer_rsrc_t r_wi =
            il_rsrc(wgu_merged + static_cast<uint64_t>(e * local_ff + rt * IL_ROWS) * row_bytes);

        #pragma nounroll
        for (uint32_t pass = 0u; pass < 2u; ++pass) {
            const uint32_t pw = (tcnt > IL_TPASS * pass) ? ((tcnt - IL_TPASS * pass < IL_TPASS) ? tcnt - IL_TPASS * pass : uint32_t(IL_TPASS)) : 0u;
            if (!pw) break;
            if (tid < IL_TPASS) sm.ms[tid] = sorted_ms[off + t0 + IL_TPASS * pass + (tid < pw ? tid : pw - 1u)];
            il_barrier();

            float ag[4], au[4];
            #pragma unroll
            for (uint32_t j = 0u; j < 4u; ++j) { ag[j] = 0.0f; au[j] = 0.0f; }

            #pragma nounroll
            for (uint32_t s = 0u; s < step_cnt; ++s) {
                // ---- decode 32 rows x 64 weights x {gate, up} into LDS, once per step ----
                const uint32_t ib = 2u * s + ibs;
                const uint32_t sd = __builtin_amdgcn_raw_buffer_load_b32(r_wi, v_hs, nib * 20u + (s >> 2) * 4u, 0u);
                const uint32_t sq = __builtin_amdgcn_raw_buffer_load_b32(r_wi, v_q, ib * 16u, 3u);
                const uint32_t shs = __builtin_amdgcn_raw_buffer_load_b32(r_wi, v_hs, nib * 16u + ib * 4u, 3u);
                const IlIq2Unit u = il_iq2_unit(sq, shs, sd, mat, half, grid);
                il_st128(&sm.w[mat][drow * IL_WS + ibs * 8u + half * 4u],
                         make_uint4(u.w[0], u.w[1], u.w[2], u.w[3]));
                sm.wsc[mat][drow * 4u + ibs * 2u + half] = u.scale;

                // ---- stage the pass activations (32 tokens x 64 INT8) and their two block scales ----
                if (tid < IL_TPASS * 4u) {
                    const uint32_t tok = tid >> 2, quarter = tid & 3u;
                    if (tok < pw) {
                        const uint32_t ms = sm.ms[tok];
                        const uint4 xv = __builtin_bit_cast(
                            uint4, __builtin_amdgcn_raw_buffer_load_b128(r_x, ms * D + s * IL_KSTEP + quarter * 16u, 0u, 3u));
                        il_st128(&sm.x[tok * IL_WS + quarter * 4u], xv);
                    }
                } else if (tid < IL_TPASS * 4u + IL_TPASS) {
                    const uint32_t tok = tid - IL_TPASS * 4u;
                    if (tok < pw) {
                        const uint32_t ms = sm.ms[tok];
                        const uint2 xsv = __builtin_bit_cast(
                            uint2, __builtin_amdgcn_raw_buffer_load_b64(r_s, ms * (nib * 8u) + s * 8u, 0u, 3u));
                        sm.xsc[tok * 2u + 0u] = __uint_as_float(xsv.x);
                        sm.xsc[tok * 2u + 1u] = __uint_as_float(xsv.y);
                    }
                }
                il_barrier();

                // ---- 1 row x 4 tokens x {gate, up}: one int32 chain + one FP32 fma per 16-group ----
                #pragma nounroll
                for (uint32_t g = 0u; g < 4u; ++g) {
                    __builtin_amdgcn_sched_barrier(0);
                    const uint32_t xoff = g * 4u;
                    const uint32_t xsi = (g >> 1);
                    {   // gate
                        const uint32_t* wg = &sm.w[0][rc * IL_WS + xoff];
                        const float sg = sm.wsc[0][rc * 4u + g];
                        #pragma unroll
                        for (uint32_t j = 0u; j < 4u; ++j) {
                            __builtin_amdgcn_sched_barrier(0);
                            const uint32_t tok = tc * 4u + j;
                            const uint32_t* xw = &sm.x[tok * IL_WS + xoff];
                            const int32_t dg = il_dot4x4(wg[0], wg[1], wg[2], wg[3], xw[0], xw[1], xw[2], xw[3]);
                            ag[j] = fmaf(sg * sm.xsc[tok * 2u + xsi], static_cast<float>(dg), ag[j]);
                        }
                    }
                    {   // up
                        const uint32_t* wu = &sm.w[1][rc * IL_WS + xoff];
                        const float su = sm.wsc[1][rc * 4u + g];
                        #pragma unroll
                        for (uint32_t j = 0u; j < 4u; ++j) {
                            __builtin_amdgcn_sched_barrier(0);
                            const uint32_t tok = tc * 4u + j;
                            const uint32_t* xw = &sm.x[tok * IL_WS + xoff];
                            const int32_t du = il_dot4x4(wu[0], wu[1], wu[2], wu[3], xw[0], xw[1], xw[2], xw[3]);
                            au[j] = fmaf(su * sm.xsc[tok * 2u + xsi], static_cast<float>(du), au[j]);
                        }
                    }
                    __builtin_amdgcn_sched_barrier(0);
                }
                il_barrier();
            }

            // ---- SwiGLU + block-32 quant along local_ff (rc IS one full 32-row block) ----
            #pragma unroll
            for (uint32_t j = 0u; j < 4u; ++j) {
                const uint32_t tok = tc * 4u + j;
                const float gv = ag[j];
                const float y = (gv / (1.0f + __expf(-gv))) * au[j];
                float amax = fabsf(y);
                amax = fmaxf(amax, __shfl_xor(amax, 16, 32));
                amax = fmaxf(amax, __shfl_xor(amax, 8, 32));
                amax = fmaxf(amax, __shfl_xor(amax, 4, 32));
                amax = fmaxf(amax, __shfl_xor(amax, 2, 32));
                amax = fmaxf(amax, __shfl_xor(amax, 1, 32));
                if (tok < pw) {
                    const float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                    const uint32_t ms = sm.ms[tok];
                    mid_q8[static_cast<uint64_t>(ms) * local_ff + rt * IL_ROWS + rc] =
                        static_cast<int8_t>(max(-127, min(127, __float2int_rn(y * inv))));
                    if (rc == 0u) mid_d[static_cast<uint64_t>(ms) * (local_ff >> 5) + rt] = amax * (1.0f / 127.0f);
                }
            }
            il_barrier();   // sm.ms / sm.x reuse on the next pass
        }
    }
    il_queue_drain(queue);
}

// ---------------------------------------------------------------------------------------------
// Down on the compact IQ3_XXS split plane (H = 512): 32 rows x 64 tokens x K step 64 (2 lanes).
// ---------------------------------------------------------------------------------------------
namespace vegajet {
namespace il_moe {

struct DownSmem {
    alignas(16) uint32_t w[il_lds::IL_ROWS * il_lds::IL_WS];   //  2,560 B decoded INT8 (64 weights per row-step)
    alignas(16) uint32_t x[il_lds::IL_TPASS * il_lds::IL_WS];  //  2,560 B INT8 activations of the current pass
    float wsc[il_lds::IL_ROWS * 2u];                           //    256 B one scale per 32-weight block of the step
    float xsc[il_lds::IL_TPASS * 2u];                          //    256 B ew * xd for the two 32-blocks of the step
    uint16_t pref[257];                                        //    514 B exclusive work items per expert
    uint32_t ms[il_lds::IL_TPASS];                             //    128 B route ids of the current pass
    uint32_t item;                                             //      4 B
};
static_assert(sizeof(DownSmem) <= 16384u, "Down LDS must keep 4 work-groups/CU");

} // namespace il_moe
} // namespace vegajet

extern "C" __global__ __launch_bounds__(256, 4) __attribute__((amdgpu_num_vgpr(64)))
void vj_prefill_moe_down_iq3_lds32x64_kernel(
    const uint8_t* __restrict__ w_qs,
    const int8_t*  __restrict__ mid_q8,
    const float*   __restrict__ mid_d,
    const float*   __restrict__ top_w,
    const uint32_t* __restrict__ exp_off,
    const uint32_t* __restrict__ exp_cnt,
    const uint32_t* __restrict__ sorted_ms,
    float*         __restrict__ part_out,
    uint32_t*      __restrict__ queue,
    uint32_t D) {
    using namespace vegajet::il_moe;
    using namespace vegajet::il_lds;
    __shared__ DownSmem sm;
    const uint32_t tid = threadIdx.x;
    if (!D || (D & 31u)) return;                       // 32-row column tiles

    const uint32_t rt_cnt = D / IL_ROWS;               // 32-row tiles per expert
    const uint32_t step_cnt = IL_H / IL_KSTEP;          // 8 steps of 64 hidden elements
    il_prefix(exp_cnt, rt_cnt, sm.pref, tid);
    const uint32_t total = __builtin_amdgcn_readfirstlane(sm.pref[256]);

    const uint32_t drow = tid & 31u;                   // decode: 32 rows x 2 lanes
    const uint32_t dhalf = tid >> 5;                   // 0..7: only 0/1 decode, others stage
    const uint32_t rc = tid & 31u;                     // compute: one D row per thread
    const uint32_t tc = tid >> 5;
    const __amdgpu_buffer_rsrc_t r_x = il_rsrc(mid_q8);
    const __amdgpu_buffer_rsrc_t r_d = il_rsrc(mid_d);
    const __amdgpu_buffer_rsrc_t r_e = il_rsrc(top_w);
    const uint32_t* grid = vegajet::ggml_tables::iq3xxs_grid;

    #pragma nounroll
    for (;;) {
        if (tid == 0u) sm.item = atomicAdd(&queue[0], 1u);
        il_barrier();
        const uint32_t item = __builtin_amdgcn_readfirstlane(sm.item);
        if (item >= total) break;

        uint32_t lo = 0u, hi = 256u;
        while (hi - lo > 1u) {
            const uint32_t mid = (lo + hi) >> 1;
            if (__builtin_amdgcn_readfirstlane(sm.pref[mid]) <= item) lo = mid; else hi = mid;
        }
        const uint32_t e     = lo;
        const uint32_t local = item - __builtin_amdgcn_readfirstlane(sm.pref[e]);
        const uint32_t cnt   = __builtin_amdgcn_readfirstlane(exp_cnt[e]);
        const uint32_t off   = __builtin_amdgcn_readfirstlane(exp_off[e]);
        const uint32_t tt    = local / rt_cnt;
        const uint32_t rt    = local - tt * rt_cnt;
        const uint32_t t0    = tt * IL_TOKENS;
        const uint32_t tcnt  = (cnt - t0 < IL_TOKENS) ? (cnt - t0) : uint32_t(IL_TOKENS);
        const uint32_t v_row = drow * IL_IQ3_ROW;
        const __amdgpu_buffer_rsrc_t r_wi =
            il_rsrc(w_qs + static_cast<uint64_t>(e * D + rt * IL_ROWS) * IL_IQ3_ROW);

        #pragma nounroll
        for (uint32_t pass = 0u; pass < 2u; ++pass) {
            const uint32_t pw = (tcnt > IL_TPASS * pass) ? ((tcnt - IL_TPASS * pass < IL_TPASS) ? tcnt - IL_TPASS * pass : uint32_t(IL_TPASS)) : 0u;
            if (!pw) break;
            if (tid < IL_TPASS) sm.ms[tid] = sorted_ms[off + t0 + IL_TPASS * pass + (tid < pw ? tid : pw - 1u)];
            il_barrier();

            float acc[4];
            #pragma unroll
            for (uint32_t j = 0u; j < 4u; ++j) acc[j] = 0.0f;

            #pragma nounroll
            for (uint32_t s = 0u; s < step_cnt; ++s) {
                // ---- decode 32 rows x 2 lanes x 32 weights into LDS, once per step ----
                if (tid < IL_ROWS * 2u) {
                    const uint32_t lane16 = 2u * s + (tid >> 5);   // tid < 64 -> half = tid >> 5
                    const uint2 q2 = __builtin_bit_cast(
                        uint2, __builtin_amdgcn_raw_buffer_load_b64(r_wi, v_row + il_iq3_q_off(lane16), 0u, 3u));
                    const uint32_t aux = __builtin_amdgcn_raw_buffer_load_b32(r_wi, v_row + il_iq3_aux_off(lane16), 0u, 3u);
                    const uint32_t d2  = __builtin_amdgcn_raw_buffer_load_b32(r_wi, v_row + il_iq3_d_off(), 0u, 0u);
                    const uint32_t qw[2] = {q2.x, q2.y};
                    const IlIq3Unit u = il_iq3_unit(qw, aux, d2, lane16, grid);
                    const uint32_t half = tid >> 5;
                    il_st128(&sm.w[drow * IL_WS + half * 8u], make_uint4(u.w[0], u.w[1], u.w[2], u.w[3]));
                    il_st128(&sm.w[drow * IL_WS + half * 8u + 4u], make_uint4(u.w[4], u.w[5], u.w[6], u.w[7]));
                    sm.wsc[drow * 2u + half] = u.scale;
                } else {
                    // ---- stage 32 tokens x 64 INT8 activations ----
                    const uint32_t u = tid - IL_ROWS * 2u;         // 0..191
                    if (u < IL_TPASS * 4u) {
                        const uint32_t tok = u >> 2, quarter = u & 3u;
                        if (tok < pw) {
                            const uint32_t ms = sm.ms[tok];
                            const uint4 xv = __builtin_bit_cast(
                                uint4, __builtin_amdgcn_raw_buffer_load_b128(r_x, ms * IL_H + s * IL_KSTEP + quarter * 16u, 0u, 3u));
                            il_st128(&sm.x[tok * IL_WS + quarter * 4u], xv);
                        }
                    }
                }
                if (tid >= 192u && tid < 192u + IL_TPASS) {   // ew * xd for the two 32-blocks of the step
                    const uint32_t tok = tid - 192u;
                    if (tok < pw) {
                        const uint32_t ms = sm.ms[tok];
                        const float ew = __uint_as_float(__builtin_amdgcn_raw_buffer_load_b32(r_e, ms * 4u, 0u, 0u));
                        const uint2 ds = __builtin_bit_cast(
                            uint2, __builtin_amdgcn_raw_buffer_load_b64(r_d, ms * (IL_H / 32u) * 4u + s * 8u, 0u, 3u));
                        sm.xsc[tok * 2u + 0u] = ew * __uint_as_float(ds.x);
                        sm.xsc[tok * 2u + 1u] = ew * __uint_as_float(ds.y);
                    }
                }
                il_barrier();

                // ---- 1 row x 4 tokens: one int32 chain (8 v_dot4) + one FP32 fma per 32-block ----
                #pragma nounroll
                for (uint32_t b = 0u; b < 2u; ++b) {
                    __builtin_amdgcn_sched_barrier(0);
                    const uint32_t* wr = &sm.w[rc * IL_WS + b * 8u];
                    const float sc = sm.wsc[rc * 2u + b];
                    #pragma unroll
                    for (uint32_t j = 0u; j < 4u; ++j) {
                        __builtin_amdgcn_sched_barrier(0);
                        const uint32_t tok = tc * 4u + j;
                        const uint32_t* xw = &sm.x[tok * IL_WS + b * 8u];
                        const int32_t dp = il_dot8x8(wr, xw);
                        acc[j] = fmaf(sc * sm.xsc[tok * 2u + b], static_cast<float>(dp), acc[j]);
                    }
                    __builtin_amdgcn_sched_barrier(0);
                }
                il_barrier();
            }

            #pragma unroll
            for (uint32_t j = 0u; j < 4u; ++j) {
                const uint32_t tok = tc * 4u + j;
                if (tok >= pw) continue;
                const uint32_t r = sm.ms[tok];
                part_out[static_cast<uint64_t>(r) * D + rt * IL_ROWS + rc] = acc[j];
            }
            il_barrier();   // sm.ms / sm.x reuse on the next pass
        }
    }
    il_queue_drain(queue);
}

// =============================================================================================
// The REQUESTED single 64x64x64 work item: 64 expert rows x 64 routed tokens x K step 64 in ONE
// work-group of 512 threads = two 256-thread cohorts. Holding a second work-group in the same
// item is impossible without spilling (16 FP32 accumulators/thread), so the 64 rows are split
// into two INDEPENDENT 32-row cohorts inside one work item instead:
//
//   * cohort c = tid / 256 owns weight rows c*32 .. c*32+31 and accumulates them (its own 32-row
//     LDS slice of the shared tile; the widths are disjoint, so no cohort ever reads the other's
//     rows). Both cohorts run the SAME K step and the SAME token pass, so the activation tile
//     (64 tokens x 64 K) and the item's route ids are staged once by the whole work-group and
//     shared through a common work-group barrier.
//   * decode role: one 16-weight IQ2_S unit per thread (64 rows x 8 units = 512), each weight of
//     the step decoded exactly once, gate and up together, and consumed by all 64 routed tokens
//     of the item (the 32-row kernel needs two 32-token passes for the same 64 rows, i.e. twice
//     the decode units; the production 16-token tiles need 4x).
//   * accumulate role: one row x 8 tokens x {gate, up} = 16 FP32 accumulators/thread, so every
//     (row, token, matrix) of the item has exactly one writer thread.
//
// 18,694 B (GU) / 17,042 B (Down) LDS keeps 2 work-groups/CU at 512 threads and 64 VGPR
// (2 x 18,694 <= 65,536 B); __launch_bounds__(512, 2) + amdgpu_num_vgpr(64) holds the tiled gate.
// Output contracts are bit-identical in layout to the 32-row variants and the tiled production
// kernels: mid_q8[route][local_ff] + mid_d[route][local_ff/32] for Gate+Up (y = silu(g) * u, one
// block scale per block-32 quant group = one cohort's 32 rows), and part_out[route * D + row]
// for Down (2 blocks x 8 steps of 64 hidden elements, one 8-dword chain + one FP32 fma per block).
// =============================================================================================
namespace vegajet {
namespace il_moe {

struct GuSmem64 {
    alignas(16) uint32_t w[2][il_lds::IL_ROWS64 * il_lds::IL_WS];  // 10,240 B decoded INT8 (gate, up)
    alignas(16) uint32_t x[il_lds::IL_TOKENS * il_lds::IL_WS];     //  5,120 B activations, all 64 tokens
    float wsc[2][il_lds::IL_ROWS64 * 4u];                          //  2,048 B scale per 16-weight group
    float xsc[il_lds::IL_TOKENS * 2u];                             //    512 B two 32-element x scales
    uint16_t pref[257];                                            //    514 B exclusive work items
    uint32_t ms[il_lds::IL_TOKENS];                                //    256 B route ids of the item
    uint32_t item;                                                 //      4 B
};
static_assert(sizeof(GuSmem64) <= 32768u, "64-row Gate+Up LDS must keep 2 work-groups/CU");

struct DownSmem64 {
    alignas(16) uint32_t w[il_lds::IL_ROWS64 * il_lds::IL_WS];     //  5,120 B decoded INT8 (2 x 32 weights)
    alignas(16) uint32_t x[il_lds::IL_TOKENS * il_lds::IL_WS];     //  5,120 B activations, all 64 tokens
    float wsc[il_lds::IL_ROWS64 * 2u];                             //    512 B scale per 32-weight block
    float xsc[il_lds::IL_TOKENS * 2u];                             //    512 B ew * xd of the two blocks
    uint16_t pref[257];                                            //    514 B
    uint32_t ms[il_lds::IL_TOKENS];                                //    256 B
    uint32_t item;                                                 //      4 B
};
static_assert(sizeof(DownSmem64) <= 32768u, "64-row Down LDS must keep 2 work-groups/CU");

// Exclusive prefix of work items per expert for a >256-thread work-group: threads 0..255 own one
// expert each, the rest only cross the barriers (identical old-value scan semantics as il_prefix).
__device__ __forceinline__ void il_prefix512(const uint32_t* exp_cnt, uint32_t rt, uint16_t* pref, uint32_t tid) {
    if (tid < 256u) pref[tid + 1u] = static_cast<uint16_t>(il_lds::il_items_per_expert64(exp_cnt[tid], rt));
    il_barrier();
    #pragma unroll
    for (uint32_t step = 1u; step < 256u; step <<= 1) {
        const uint32_t add = (tid < 256u && tid >= step) ? pref[tid + 1u - step] : 0u;
        il_barrier();
        if (tid < 256u) pref[tid + 1u] = static_cast<uint16_t>(pref[tid + 1u] + add);
        il_barrier();
    }
    if (tid == 0u) pref[0] = 0u;
    il_barrier();
}

} // namespace il_moe
} // namespace vegajet

extern "C" __global__ __launch_bounds__(512, 2) __attribute__((amdgpu_num_vgpr(64)))
void vj_prefill_moe_gu_iq2s_lds64x64_kernel(
    const uint4*  __restrict__ wgu_merged,
    const int8_t* __restrict__ x_q8,
    const float2* __restrict__ x_ds,
    const uint32_t* __restrict__ exp_off,
    const uint32_t* __restrict__ exp_cnt,
    const uint32_t* __restrict__ sorted_ms,
    int8_t*       __restrict__ mid_q8,
    float*        __restrict__ mid_d,
    uint32_t*     __restrict__ queue,
    uint32_t local_ff,
    uint32_t D) {
    using namespace vegajet::il_moe;
    using namespace vegajet::il_lds;
    __shared__ GuSmem64 sm;
    const uint32_t tid = threadIdx.x;
    const uint32_t nib = D >> 5;                       // 32-element groups along K
    if (!local_ff || (local_ff & 63u) || !nib || (nib & 7u) || D != nib * 32u) return;

    const uint32_t rt_cnt = il_row_tiles64(local_ff);   // 64-row work items per expert (local_ff/64)
    const uint32_t step_cnt = nib >> 1;                 // 64-element steps
    il_prefix512(exp_cnt, rt_cnt, sm.pref, tid);
    const uint32_t total = __builtin_amdgcn_readfirstlane(sm.pref[256]);

    // Decode role: row 0..63 (tid & 63) x unit (tid / 64) = (ib-in-step, matrix, half); 512 units.
    const uint32_t drow = il_dec_row64(tid);
    const uint32_t unit = il_dec_unit64(tid);
    const uint32_t ibs  = unit >> 2;
    const uint32_t mat  = (unit >> 1) & 1u;
    const uint32_t half = unit & 1u;
    // Accumulate role: cohort c owns rows c*32 + (tid & 31); token slot q covers tokens q*8 .. q*8+7.
    const uint32_t arow = il_acc_row64(tid);
    const uint32_t q    = il_acc_slot64(tid);
    const __amdgpu_buffer_rsrc_t r_x = il_rsrc(x_q8);
    const __amdgpu_buffer_rsrc_t r_s = il_rsrc(x_ds);
    const uint64_t* grid = vegajet::ggml_tables::iq2s_grid;

    #pragma nounroll
    for (;;) {
        if (tid == 0u) sm.item = atomicAdd(&queue[0], 1u);
        il_barrier();
        const uint32_t item = __builtin_amdgcn_readfirstlane(sm.item);
        if (item >= total) break;

        uint32_t lo = 0u, hi = 256u;
        while (hi - lo > 1u) {
            const uint32_t mid = (lo + hi) >> 1;
            if (__builtin_amdgcn_readfirstlane(sm.pref[mid]) <= item) lo = mid; else hi = mid;
        }
        const uint32_t e     = lo;
        const uint32_t local = item - __builtin_amdgcn_readfirstlane(sm.pref[e]);
        const uint32_t cnt   = __builtin_amdgcn_readfirstlane(exp_cnt[e]);
        const uint32_t off   = __builtin_amdgcn_readfirstlane(exp_off[e]);
        const uint32_t tt    = local / rt_cnt;
        const uint32_t rt    = local - tt * rt_cnt;
        const uint32_t t0    = tt * IL_TOKENS;
        const uint32_t tcnt  = (cnt - t0 < IL_TOKENS) ? (cnt - t0) : uint32_t(IL_TOKENS);
        if (tid < IL_TOKENS) sm.ms[tid] = sorted_ms[off + t0 + (tid < tcnt ? tid : tcnt - 1u)];
        il_barrier();

        const uint32_t row_bytes = il_gu_row_bytes(nib);
        const uint32_t v_row = drow * row_bytes;
        const uint32_t v_q   = v_row + mat * 8u + half * 4u;
        const uint32_t v_hs  = v_row;
        const __amdgpu_buffer_rsrc_t r_wi =
            il_rsrc(wgu_merged + static_cast<uint64_t>(e * local_ff + rt * IL_ROWS64) * row_bytes);

        float ag[IL_ACC64], au[IL_ACC64];
        #pragma unroll
        for (uint32_t j = 0u; j < IL_ACC64; ++j) { ag[j] = 0.0f; au[j] = 0.0f; }

        #pragma nounroll
        for (uint32_t s = 0u; s < step_cnt; ++s) {
            // ---- decode 64 rows x 64 weights x {gate, up} into LDS, ONCE for all 64 tokens ----
            const uint32_t ib = 2u * s + ibs;
            const uint32_t sd = __builtin_amdgcn_raw_buffer_load_b32(r_wi, v_hs, nib * 20u + (s >> 2) * 4u, 0u);
            const uint32_t sq = __builtin_amdgcn_raw_buffer_load_b32(r_wi, v_q, ib * 16u, 3u);
            const uint32_t shs = __builtin_amdgcn_raw_buffer_load_b32(r_wi, v_hs, nib * 16u + ib * 4u, 3u);
            const IlIq2Unit u = il_iq2_unit(sq, shs, sd, mat, half, grid);
            il_st128(&sm.w[mat][drow * IL_WS + ibs * 8u + half * 4u],
                     make_uint4(u.w[0], u.w[1], u.w[2], u.w[3]));
            sm.wsc[mat][drow * 4u + ibs * 2u + half] = u.scale;

            // ---- stage the item's 64 tokens x 64 INT8 activations and their two block scales ----
            if (tid < IL_TOKENS * 4u) {
                const uint32_t tok = tid >> 2, quarter = tid & 3u;
                if (tok < tcnt) {
                    const uint32_t ms = sm.ms[tok];
                    const uint4 xv = __builtin_bit_cast(
                        uint4, __builtin_amdgcn_raw_buffer_load_b128(r_x, ms * D + s * IL_KSTEP + quarter * 16u, 0u, 3u));
                    il_st128(&sm.x[tok * IL_WS + quarter * 4u], xv);
                }
            } else if (tid < IL_TOKENS * 5u) {
                const uint32_t tok = tid - IL_TOKENS * 4u;
                if (tok < tcnt) {
                    const uint32_t ms = sm.ms[tok];
                    const uint2 xsv = __builtin_bit_cast(
                        uint2, __builtin_amdgcn_raw_buffer_load_b64(r_s, ms * (nib * 8u) + s * 8u, 0u, 3u));
                    sm.xsc[tok * 2u + 0u] = __uint_as_float(xsv.x);
                    sm.xsc[tok * 2u + 1u] = __uint_as_float(xsv.y);
                }
            }
            il_barrier();

            // ---- 1 row x 8 tokens x {gate, up}: one int32 chain + one FP32 fma per 16-group ----
            #pragma nounroll
            for (uint32_t g = 0u; g < 4u; ++g) {
                __builtin_amdgcn_sched_barrier(0);
                const uint32_t* wg = &sm.w[0][arow * IL_WS + g * 4u];
                const uint32_t* wu = &sm.w[1][arow * IL_WS + g * 4u];
                const float sg = sm.wsc[0][arow * 4u + g];
                const float su = sm.wsc[1][arow * 4u + g];
                const uint32_t* xb = &sm.x[(q * IL_ACC64) * IL_WS + g * 4u];
                const float* xsb = &sm.xsc[(q * IL_ACC64) * 2u + (g >> 1)];
                #pragma unroll
                for (uint32_t j = 0u; j < IL_ACC64; ++j) {
                    __builtin_amdgcn_sched_barrier(0);
                    const uint32_t* xw = xb + j * IL_WS;
                    const int32_t dg = il_dot4x4(wg[0], wg[1], wg[2], wg[3], xw[0], xw[1], xw[2], xw[3]);
                    const float xsf = xsb[j * 2u];
                    const int32_t du = il_dot4x4(wu[0], wu[1], wu[2], wu[3], xw[0], xw[1], xw[2], xw[3]);
                    ag[j] = fmaf(sg * xsf, static_cast<float>(dg), ag[j]);
                    au[j] = fmaf(su * xsf, static_cast<float>(du), au[j]);
                }
                __builtin_amdgcn_sched_barrier(0);
            }
            il_barrier();
        }

        // ---- SwiGLU + block-32 quant: a cohort IS one full 32-row quant group of local_ff ----
        const uint32_t block = rt * 2u + (arow >> 5);          // block-32 index inside local_ff
        #pragma unroll
        for (uint32_t j = 0u; j < IL_ACC64; ++j) {
            const uint32_t tok = q * IL_ACC64 + j;
            const float gv = ag[j];
            const float y = (gv / (1.0f + __expf(-gv))) * au[j];
            float amax = fabsf(y);
            amax = fmaxf(amax, __shfl_xor(amax, 16, 32));
            amax = fmaxf(amax, __shfl_xor(amax, 8, 32));
            amax = fmaxf(amax, __shfl_xor(amax, 4, 32));
            amax = fmaxf(amax, __shfl_xor(amax, 2, 32));
            amax = fmaxf(amax, __shfl_xor(amax, 1, 32));
            if (tok < tcnt) {
                const float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                const uint32_t ms = sm.ms[tok];
                mid_q8[static_cast<uint64_t>(ms) * local_ff + rt * IL_ROWS64 + arow] =
                    static_cast<int8_t>(max(-127, min(127, __float2int_rn(y * inv))));
                if ((tid & 31u) == 0u)
                    mid_d[static_cast<uint64_t>(ms) * (local_ff >> 5) + block] = amax * (1.0f / 127.0f);
            }
        }
        il_barrier();   // sm.ms / sm.x reuse by the next work item
    }
    il_queue_drain(queue);
}

// ---------------------------------------------------------------------------------------------
// Down on the compact IQ3_XXS split plane (H = 512) as the same single 64x64x64 work item:
// 64 hidden rows x 64 routed tokens x K step 64 (two 32-weight lanes), 512 threads, 8 FP32
// accumulators/thread (1 row x 8 tokens), one decoded lane consumed by all 64 tokens.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(512, 2) __attribute__((amdgpu_num_vgpr(64)))
void vj_prefill_moe_down_iq3_lds64x64_kernel(
    const uint8_t* __restrict__ w_qs,
    const int8_t*  __restrict__ mid_q8,
    const float*   __restrict__ mid_d,
    const float*   __restrict__ top_w,
    const uint32_t* __restrict__ exp_off,
    const uint32_t* __restrict__ exp_cnt,
    const uint32_t* __restrict__ sorted_ms,
    float*         __restrict__ part_out,
    uint32_t*      __restrict__ queue,
    uint32_t D) {
    using namespace vegajet::il_moe;
    using namespace vegajet::il_lds;
    __shared__ DownSmem64 sm;
    const uint32_t tid = threadIdx.x;
    if (!D || (D & 63u)) return;                       // 64-row work items

    const uint32_t rt_cnt = D / IL_ROWS64;             // 64-row work items per expert
    const uint32_t step_cnt = IL_H / IL_KSTEP;         // 8 steps of 64 hidden elements
    il_prefix512(exp_cnt, rt_cnt, sm.pref, tid);
    const uint32_t total = __builtin_amdgcn_readfirstlane(sm.pref[256]);

    // Decode role: threads 0..127 = 64 rows x 2 lanes; everyone else stages, then all accumulate.
    const uint32_t drow = tid & 63u;
    const uint32_t dhf  = (tid >> 6) & 1u;
    const uint32_t arow = il_acc_row64(tid);
    const uint32_t q    = il_acc_slot64(tid);
    const __amdgpu_buffer_rsrc_t r_x = il_rsrc(mid_q8);
    const __amdgpu_buffer_rsrc_t r_d = il_rsrc(mid_d);
    const __amdgpu_buffer_rsrc_t r_e = il_rsrc(top_w);
    const uint32_t* grid = vegajet::ggml_tables::iq3xxs_grid;

    #pragma nounroll
    for (;;) {
        if (tid == 0u) sm.item = atomicAdd(&queue[0], 1u);
        il_barrier();
        const uint32_t item = __builtin_amdgcn_readfirstlane(sm.item);
        if (item >= total) break;

        uint32_t lo = 0u, hi = 256u;
        while (hi - lo > 1u) {
            const uint32_t mid = (lo + hi) >> 1;
            if (__builtin_amdgcn_readfirstlane(sm.pref[mid]) <= item) lo = mid; else hi = mid;
        }
        const uint32_t e     = lo;
        const uint32_t local = item - __builtin_amdgcn_readfirstlane(sm.pref[e]);
        const uint32_t cnt   = __builtin_amdgcn_readfirstlane(exp_cnt[e]);
        const uint32_t off   = __builtin_amdgcn_readfirstlane(exp_off[e]);
        const uint32_t tt    = local / rt_cnt;
        const uint32_t rt    = local - tt * rt_cnt;
        const uint32_t t0    = tt * IL_TOKENS;
        const uint32_t tcnt  = (cnt - t0 < IL_TOKENS) ? (cnt - t0) : uint32_t(IL_TOKENS);
        if (tid < IL_TOKENS) sm.ms[tid] = sorted_ms[off + t0 + (tid < tcnt ? tid : tcnt - 1u)];
        il_barrier();

        const uint32_t v_row = drow * IL_IQ3_ROW;
        const __amdgpu_buffer_rsrc_t r_wi =
            il_rsrc(w_qs + static_cast<uint64_t>(e * D + rt * IL_ROWS64) * IL_IQ3_ROW);

        float acc[IL_ACC64];
        #pragma unroll
        for (uint32_t j = 0u; j < IL_ACC64; ++j) acc[j] = 0.0f;

        #pragma nounroll
        for (uint32_t s = 0u; s < step_cnt; ++s) {
            // ---- decode 64 rows x 2 lanes x 32 weights into LDS, ONCE for all 64 tokens ----
            if (tid < IL_ROWS64 * 2u) {
                const uint32_t lane16 = 2u * s + dhf;
                const uint2 q2 = __builtin_bit_cast(
                    uint2, __builtin_amdgcn_raw_buffer_load_b64(r_wi, v_row + il_iq3_q_off(lane16), 0u, 3u));
                const uint32_t aux = __builtin_amdgcn_raw_buffer_load_b32(r_wi, v_row + il_iq3_aux_off(lane16), 0u, 3u);
                const uint32_t d2  = __builtin_amdgcn_raw_buffer_load_b32(r_wi, v_row + il_iq3_d_off(), 0u, 0u);
                const uint32_t qw[2] = {q2.x, q2.y};
                const IlIq3Unit u = il_iq3_unit(qw, aux, d2, lane16, grid);
                il_st128(&sm.w[drow * IL_WS + dhf * 8u], make_uint4(u.w[0], u.w[1], u.w[2], u.w[3]));
                il_st128(&sm.w[drow * IL_WS + dhf * 8u + 4u], make_uint4(u.w[4], u.w[5], u.w[6], u.w[7]));
                sm.wsc[drow * 2u + dhf] = u.scale;
            } else {
                const uint32_t u = tid - IL_ROWS64 * 2u;              // 0..383
                if (u < IL_TOKENS * 4u) {                             // 64 tokens x 4 quarters
                    const uint32_t tok = u >> 2, quarter = u & 3u;
                    if (tok < tcnt) {
                        const uint32_t ms = sm.ms[tok];
                        const uint4 xv = __builtin_bit_cast(
                            uint4, __builtin_amdgcn_raw_buffer_load_b128(r_x, ms * IL_H + s * IL_KSTEP + quarter * 16u, 0u, 3u));
                        il_st128(&sm.x[tok * IL_WS + quarter * 4u], xv);
                    }
                } else if (u < IL_TOKENS * 5u) {                      // ew * xd of the step's two blocks
                    const uint32_t tok = u - IL_TOKENS * 4u;
                    if (tok < tcnt) {
                        const uint32_t ms = sm.ms[tok];
                        const float ew = __uint_as_float(__builtin_amdgcn_raw_buffer_load_b32(r_e, ms * 4u, 0u, 0u));
                        const uint2 ds = __builtin_bit_cast(
                            uint2, __builtin_amdgcn_raw_buffer_load_b64(r_d, ms * (IL_H / 32u) * 4u + s * 8u, 0u, 3u));
                        sm.xsc[tok * 2u + 0u] = ew * __uint_as_float(ds.x);
                        sm.xsc[tok * 2u + 1u] = ew * __uint_as_float(ds.y);
                    }
                }
            }
            il_barrier();

            // ---- 1 row x 8 tokens: one int32 chain + one FP32 fma per 32-weight block ----
            #pragma nounroll
            for (uint32_t b = 0u; b < 2u; ++b) {
                __builtin_amdgcn_sched_barrier(0);
                const uint32_t* wr = &sm.w[arow * IL_WS + b * 8u];
                const float sc = sm.wsc[arow * 2u + b];
                const uint32_t* xb = &sm.x[(q * IL_ACC64) * IL_WS + b * 8u];
                const float* xsb = &sm.xsc[(q * IL_ACC64) * 2u + b];
                #pragma unroll
                for (uint32_t j = 0u; j < IL_ACC64; ++j) {
                    __builtin_amdgcn_sched_barrier(0);
                    const int32_t dp = il_dot8x8(wr, xb + j * IL_WS);
                    acc[j] = fmaf(sc * xsb[j * 2u], static_cast<float>(dp), acc[j]);
                }
                __builtin_amdgcn_sched_barrier(0);
            }
            il_barrier();
        }

        #pragma unroll
        for (uint32_t j = 0u; j < IL_ACC64; ++j) {
            const uint32_t tok = q * IL_ACC64 + j;
            if (tok >= tcnt) continue;
            part_out[static_cast<uint64_t>(sm.ms[tok]) * D + rt * IL_ROWS64 + arow] = acc[j];
        }
        il_barrier();   // sm.ms / sm.x reuse by the next work item
    }
    il_queue_drain(queue);
}
