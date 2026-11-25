// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Standalone A/B test kernels for the VJ_Q4_DOT8 routed-MoE tile bodies (ops/kernels/vj_moe_dot8.hiph).
// Same launch geometry, LDS staging and outputs as kernels 9b / 9c in vj_ornith_ops.hip, plus the new weight
// pointers / strides. Output buffers must match 9b / 9c bit for bit on the same inputs.
#include <hip/hip_runtime.h>
#include <stdint.h>

#include "core/ggml_iq_tables.hpp"
#include "ops/kernels/vj_moe_dot8.hiph"

// ============================================================================
// Down (A/B against ornith_moe_down_iq3xxs_split_kernel, 9c): grid = (D / 16, M), block = 256.
//   w_qs / w_stride: weight plane [256 * D rows], w_sc / s_stride: scale region [256 * D rows].
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 5)
void vj_moe_down_dot8_kernel(
    const uint8_t* __restrict__ w_qs,
    const uint8_t* __restrict__ w_sc,
    const int8_t*  __restrict__ mid_q8,
    const float*   __restrict__ mid_d,
    const int32_t* __restrict__ top_ids,
    const float*   __restrict__ top_w,
    const float*   __restrict__ sh_down_in,
    float*         __restrict__ out_f32,
    uint32_t M,
    uint32_t D,
    uint32_t w_stride,
    uint32_t s_stride) {
    __shared__ uint4   sm_mq[256];
    __shared__ float   sm_md[128];
    __shared__ int32_t sm_ids[8];
    __shared__ float   sm_w[8];

    const uint32_t m      = blockIdx.y;
    const uint32_t lane16 = threadIdx.x & 15u;
    const uint32_t row    = blockIdx.x * 16u + (threadIdx.x >> 4);
    if (m >= M) return;

    sm_mq[threadIdx.x] = reinterpret_cast<const uint4*>(mid_q8 + static_cast<uint64_t>(m) * 4096u)[threadIdx.x];
    if (threadIdx.x < 128u) sm_md[threadIdx.x] = mid_d[static_cast<uint64_t>(m) * 128u + threadIdx.x];
    if (threadIdx.x <   8u) {
        sm_ids[threadIdx.x] = top_ids[static_cast<uint64_t>(m) * 8u + threadIdx.x];
        sm_w[threadIdx.x]   = top_w[static_cast<uint64_t>(m) * 8u + threadIdx.x];
    }
    __syncthreads();
    if (row >= D) return;

    const float total = vegajet::dot8_dev::vj_dot8_down_row_iq3<4>(w_qs, w_stride, w_sc, s_stride,
                                                                  sm_mq, sm_md, sm_ids, sm_w, row, D, lane16);
    if (lane16 == 0u) {
        const float sh = sh_down_in ? sh_down_in[static_cast<uint64_t>(m) * D + row] : 0.0f;
        out_f32[static_cast<uint64_t>(m) * D + row] = total + sh;
    }
}

// Down with optional FP16 BAR0 epilogue prepush for M<=12 speculative verify (0 extra VGPRs, 64 B extra LDS).
extern "C" __global__ __launch_bounds__(256, 5)
void vj_moe_down_dot8_prepush_kernel(
    const uint8_t* __restrict__ w_qs,
    const uint8_t* __restrict__ w_sc,
    const int8_t*  __restrict__ mid_q8,
    const float*   __restrict__ mid_d,
    const int32_t* __restrict__ top_ids,
    const float*   __restrict__ top_w,
    const float*   __restrict__ sh_down_in,
    float*         __restrict__ out_f32,
    uint32_t M,
    uint32_t D,
    uint32_t w_stride,
    uint32_t s_stride,
    uint16_t*      __restrict__ peer_tx) {
    __shared__ uint4   sm_mq[256];
    __shared__ float   sm_md[128];
    __shared__ int32_t sm_ids[8];
    __shared__ float   sm_w[8];
    __shared__ float   sm_y[16];

    const uint32_t m      = blockIdx.y;
    const uint32_t grp16  = threadIdx.x >> 4;
    const uint32_t lane16 = threadIdx.x & 15u;
    const uint32_t row    = blockIdx.x * 16u + grp16;
    if (m >= M) return;

    sm_mq[threadIdx.x] = reinterpret_cast<const uint4*>(mid_q8 + static_cast<uint64_t>(m) * 4096u)[threadIdx.x];
    if (threadIdx.x < 128u) sm_md[threadIdx.x] = mid_d[static_cast<uint64_t>(m) * 128u + threadIdx.x];
    if (threadIdx.x <   8u) {
        sm_ids[threadIdx.x] = top_ids[static_cast<uint64_t>(m) * 8u + threadIdx.x];
        sm_w[threadIdx.x]   = top_w[static_cast<uint64_t>(m) * 8u + threadIdx.x];
    }
    __syncthreads();

    if (row < D) {
        // Speculative verify (M = 2..16): L2-cached loads reuse experts hit by multiple tokens in the batch (bit-identical).
        const float total = vegajet::dot8_dev::vj_dot8_down_row_iq3<4, true>(w_qs, w_stride, w_sc, s_stride,
                                                                            sm_mq, sm_md, sm_ids, sm_w, row, D, lane16);
        if (lane16 == 0u) {
            const float sh = sh_down_in ? sh_down_in[static_cast<uint64_t>(m) * D + row] : 0.0f;
            const float v  = total + sh;
            out_f32[static_cast<uint64_t>(m) * D + row] = v;
            sm_y[grp16] = v;
        }
    }
    if (peer_tx != nullptr && (D & 15u) == 0u) {
        __syncthreads();
        if (threadIdx.x < 2u) {
            const uint32_t o = threadIdx.x * 8u;
            const __half2 h0 = __float22half2_rn(make_float2(sm_y[o + 0u], sm_y[o + 1u]));
            const __half2 h1 = __float22half2_rn(make_float2(sm_y[o + 2u], sm_y[o + 3u]));
            const __half2 h2 = __float22half2_rn(make_float2(sm_y[o + 4u], sm_y[o + 5u]));
            const __half2 h3 = __float22half2_rn(make_float2(sm_y[o + 6u], sm_y[o + 7u]));
            typedef uint32_t u32x4_t __attribute__((ext_vector_type(4)));
            u32x4_t pv = {
                *reinterpret_cast<const uint32_t*>(&h0),
                *reinterpret_cast<const uint32_t*>(&h1),
                *reinterpret_cast<const uint32_t*>(&h2),
                *reinterpret_cast<const uint32_t*>(&h3)
            };
            __builtin_nontemporal_store(pv, reinterpret_cast<u32x4_t*>(peer_tx + static_cast<uint64_t>(m) * D + blockIdx.x * 16u + o));
        }
    }
}

// ============================================================================
// Gate+Up + SiLU*up + INT8 quant (A/B against ornith_moe_gu_iq2s_merged_kernel, 9b):
//   grid = (8 * local_ff / 32, M), block = 256, D = 3072 (x_q8 rows of 3072, 96 blocks, n_it = 6).
//   w_gu / w_stride: merged weight plane [256 * local_ff rows], w_sc / s_stride: scale region.
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 5)
void vj_moe_gu_dot8_kernel(
    const uint8_t* __restrict__ w_gu,
    const uint8_t* __restrict__ w_sc,
    const int8_t*  __restrict__ x_q8,
    const float2*  __restrict__ x_ds,
    const int32_t* __restrict__ top_ids,
    int8_t*        __restrict__ mid_q8,
    float*         __restrict__ mid_d,
    uint32_t M,
    uint32_t local_ff,
    uint32_t w_stride,
    uint32_t s_stride) {
    __shared__ uint4 sm_xq[192];
    __shared__ float sm_xd[96];
    __shared__ float sm_y[32];

    const uint32_t m      = blockIdx.y;
    const uint32_t nblk32 = local_ff >> 5;
    const uint32_t slot   = blockIdx.x / nblk32;
    const uint32_t rblk   = blockIdx.x - slot * nblk32;
    if (m >= M || slot >= 8u) return;

    const uint4*  xq4_g = reinterpret_cast<const uint4*>(x_q8 + static_cast<uint64_t>(m) * 3072u);
    const float2* xds_g = x_ds + static_cast<uint64_t>(m) * 96u;
    if (threadIdx.x < 192u) sm_xq[threadIdx.x] = xq4_g[threadIdx.x];
    if (threadIdx.x <  96u) sm_xd[threadIdx.x] = xds_g[threadIdx.x].x;
    __syncthreads();

    const uint32_t exp_id = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(top_ids[static_cast<uint64_t>(m) * 8u + slot]));
    const uint32_t sub_id = threadIdx.x >> 4;
    const uint32_t lane16 = threadIdx.x & 15u;

    #pragma nounroll
    for (uint32_t half = 0u; half < 2u; ++half) {
        const uint32_t row = rblk * 32u + half * 16u + sub_id;
        const float2 gu = vegajet::dot8_dev::vj_dot8_gu_row_iq2s<6, 2>(w_gu, w_stride, w_sc, s_stride, sm_xq, sm_xd,
                                                                      exp_id * local_ff, row, lane16);
        if (lane16 == 0u) sm_y[half * 16u + sub_id] = (gu.x / (1.0f + __expf(-gu.x))) * gu.y;
    }
    __syncthreads();

    if (threadIdx.x < 32u) {
        float y = sm_y[threadIdx.x];
        float amax = fabsf(y);
        amax = fmaxf(amax, __shfl_xor(amax, 16, 32));
        amax = fmaxf(amax, __shfl_xor(amax, 8,  32));
        amax = fmaxf(amax, __shfl_xor(amax, 4,  32));
        amax = fmaxf(amax, __shfl_xor(amax, 2,  32));
        amax = fmaxf(amax, __shfl_xor(amax, 1,  32));
        const float d   = amax * (1.0f / 127.0f);
        const float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        mid_q8[(static_cast<uint64_t>(m) * 8u + slot) * local_ff + rblk * 32u + threadIdx.x] =
            static_cast<int8_t>(max(-127, min(127, __float2int_rn(y * inv))));
        if (threadIdx.x == 0u) mid_d[(static_cast<uint64_t>(m) * 8u + slot) * nblk32 + rblk] = d;
    }
}

// Lossless 3-bit bit-sliced Gate+Up kernel (VJ_Q3_IQ2S, L2-cached for M >= 2 prompt/verify; <=48 VGPRs, 0 B spill).
extern "C" __global__ __launch_bounds__(256, 5)
void vj_moe_gu_q3_kernel(
    const uint8_t* __restrict__ w_gu,
    const uint8_t* __restrict__ w_sc,
    const int8_t*  __restrict__ x_q8,
    const float2*  __restrict__ x_ds,
    const int32_t* __restrict__ top_ids,
    int8_t*        __restrict__ mid_q8,
    float*         __restrict__ mid_d,
    uint32_t M,
    uint32_t local_ff,
    uint32_t w_stride,
    uint32_t s_stride) {
    __shared__ uint4 sm_xq[192];
    __shared__ float sm_xd[96];
    __shared__ float sm_y[32];

    const uint32_t m      = blockIdx.y;
    const uint32_t nblk32 = local_ff >> 5;
    const uint32_t slot   = blockIdx.x / nblk32;
    const uint32_t rblk   = blockIdx.x - slot * nblk32;
    if (m >= M || slot >= 8u) return;

    const uint4*  xq4_g = reinterpret_cast<const uint4*>(x_q8 + static_cast<uint64_t>(m) * 3072u);
    const float2* xds_g = x_ds + static_cast<uint64_t>(m) * 96u;
    if (threadIdx.x < 192u) sm_xq[threadIdx.x] = xq4_g[threadIdx.x];
    if (threadIdx.x <  96u) sm_xd[threadIdx.x] = xds_g[threadIdx.x].x;
    __syncthreads();

    const uint32_t exp_id = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(top_ids[static_cast<uint64_t>(m) * 8u + slot]));
    const uint32_t sub_id = threadIdx.x >> 4;
    const uint32_t lane16 = threadIdx.x & 15u;

    #pragma nounroll
    for (uint32_t half = 0u; half < 2u; ++half) {
        const uint32_t row = rblk * 32u + half * 16u + sub_id;
        const float2 gu = vegajet::dot8_dev::vj_q3_gu_row_iq2s<6, 1, true>(w_gu, w_stride, w_sc, s_stride, sm_xq, sm_xd,
                                                                           exp_id * local_ff, row, lane16);
        if (lane16 == 0u) sm_y[half * 16u + sub_id] = (gu.x / (1.0f + __expf(-gu.x))) * gu.y;
    }
    __syncthreads();

    if (threadIdx.x < 32u) {
        float y = sm_y[threadIdx.x];
        float amax = fabsf(y);
        amax = fmaxf(amax, __shfl_xor(amax, 16, 32));
        amax = fmaxf(amax, __shfl_xor(amax, 8,  32));
        amax = fmaxf(amax, __shfl_xor(amax, 4,  32));
        amax = fmaxf(amax, __shfl_xor(amax, 2,  32));
        amax = fmaxf(amax, __shfl_xor(amax, 1,  32));
        const float d   = amax * (1.0f / 127.0f);
        const float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        mid_q8[(static_cast<uint64_t>(m) * 8u + slot) * local_ff + rblk * 32u + threadIdx.x] =
            static_cast<int8_t>(max(-127, min(127, __float2int_rn(y * inv))));
        if (threadIdx.x == 0u) mid_d[(static_cast<uint64_t>(m) * 8u + slot) * nblk32 + rblk] = d;
    }
}

// Reuse-driven L2-cached Gate+Up DOT8 kernel (glc=0 slc=0) for M >= 2 speculative verify batches
// where multiple tokens hit the same expert (bit-identical to vj_moe_gu_dot8_kernel).
extern "C" __global__ __launch_bounds__(256, 5)
void vj_moe_gu_dot8_cached_kernel(
    const uint8_t* __restrict__ w_gu,
    const uint8_t* __restrict__ w_sc,
    const int8_t*  __restrict__ x_q8,
    const float2*  __restrict__ x_ds,
    const int32_t* __restrict__ top_ids,
    int8_t*        __restrict__ mid_q8,
    float*         __restrict__ mid_d,
    uint32_t M,
    uint32_t local_ff,
    uint32_t w_stride,
    uint32_t s_stride) {
    __shared__ uint4 sm_xq[192];
    __shared__ float sm_xd[96];
    __shared__ float sm_y[32];

    const uint32_t m      = blockIdx.y;
    const uint32_t nblk32 = local_ff >> 5;
    const uint32_t slot   = blockIdx.x / nblk32;
    const uint32_t rblk   = blockIdx.x - slot * nblk32;
    if (m >= M || slot >= 8u) return;

    const uint4*  xq4_g = reinterpret_cast<const uint4*>(x_q8 + static_cast<uint64_t>(m) * 3072u);
    const float2* xds_g = x_ds + static_cast<uint64_t>(m) * 96u;
    if (threadIdx.x < 192u) sm_xq[threadIdx.x] = xq4_g[threadIdx.x];
    if (threadIdx.x <  96u) sm_xd[threadIdx.x] = xds_g[threadIdx.x].x;
    __syncthreads();

    const uint32_t exp_id = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(top_ids[static_cast<uint64_t>(m) * 8u + slot]));
    const uint32_t sub_id = threadIdx.x >> 4;
    const uint32_t lane16 = threadIdx.x & 15u;

    #pragma nounroll
    for (uint32_t half = 0u; half < 2u; ++half) {
        const uint32_t row = rblk * 32u + half * 16u + sub_id;
        const float2 gu = vegajet::dot8_dev::vj_dot8_gu_row_iq2s<6, 2, true>(w_gu, w_stride, w_sc, s_stride, sm_xq, sm_xd,
                                                                            exp_id * local_ff, row, lane16);
        if (lane16 == 0u) sm_y[half * 16u + sub_id] = (gu.x / (1.0f + __expf(-gu.x))) * gu.y;
    }
    __syncthreads();

    if (threadIdx.x < 32u) {
        float y = sm_y[threadIdx.x];
        float amax = fabsf(y);
        amax = fmaxf(amax, __shfl_xor(amax, 16, 32));
        amax = fmaxf(amax, __shfl_xor(amax, 8,  32));
        amax = fmaxf(amax, __shfl_xor(amax, 4,  32));
        amax = fmaxf(amax, __shfl_xor(amax, 2,  32));
        amax = fmaxf(amax, __shfl_xor(amax, 1,  32));
        const float d   = amax * (1.0f / 127.0f);
        const float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        mid_q8[(static_cast<uint64_t>(m) * 8u + slot) * local_ff + rblk * 32u + threadIdx.x] =
            static_cast<int8_t>(max(-127, min(127, __float2int_rn(y * inv))));
        if (threadIdx.x == 0u) mid_d[(static_cast<uint64_t>(m) * 8u + slot) * nblk32 + rblk] = d;
    }
}

// ============================================================================
// Wave 5 Dynamic LRU MoE Cache: On-GPU IQ2_S -> exact VJ_Q4_DOT8 Gate+Up Repack
// Grid: local_ff (512 WGs, 1 row/WG), Block: 128 thr (tid = ib in 0..95 for D=3072).
// Bit-exact match with core/q4_dot8_repack.cpp (repacks 1 expert in ~8 us in VRAM!).
// ============================================================================
__device__ __forceinline__ uint32_t iq2s_code4_swar(uint32_t mag4_sel, uint32_t sign4) {
    const uint32_t neg = (sign4 * 0x00204081u) & 0x01010101u;
    return (mag4_sel & 0x03030303u) + neg * 3u;
}

__device__ __forceinline__ uint32_t iq2s_perm4_lru(uint32_t mag4_sel, uint32_t sign4) {
    const uint32_t sel = (mag4_sel & 0x03030303u) | ((sign4 * 0x00810204u) & 0x04040404u);
    return __builtin_amdgcn_perm(0x00D5E7F8u, 0x002B1908u, sel);
}

extern "C" __global__ __launch_bounds__(128, 4)
void vj_repack_iq2s_to_dot8_gu_kernel(
    const uint4* __restrict__ wgu_iq2s,
    uint8_t*     __restrict__ cache_w,
    uint8_t*     __restrict__ cache_sc,
    uint32_t exp_id,
    uint32_t slot_id,
    uint32_t local_ff,
    uint32_t w_stride,
    uint32_t s_stride) {
    const uint32_t row = blockIdx.x;
    const uint32_t ib  = threadIdx.x;
    if (row >= local_ff || ib >= 96u) return;

    const uint4* qp_p = wgu_iq2s + (static_cast<uint64_t>(exp_id) * local_ff + row) * 123ULL;
    const uint4    wq  = qp_p[ib];
    const uint32_t hs2 = reinterpret_cast<const uint32_t*>(qp_p + 96u)[ib];
    const uint64_t dst_row = static_cast<uint64_t>(slot_id) * local_ff + row;
    uint8_t* w_dst = cache_w  + dst_row * w_stride + ib * 32u;
    uint8_t* s_dst = cache_sc + dst_row * s_stride;

    #pragma unroll
    for (uint32_t part = 0u; part < 2u; ++part) {
        const uint32_t qa  = part ? wq.z : wq.x;
        const uint32_t qb  = part ? wq.w : wq.y;
        const uint32_t hsp = part ? (hs2 >> 16) : (hs2 & 0xFFFFu);
        const uint32_t qh  = hsp & 0xFFu;
        const uint32_t g0  = vegajet::ggml_tables::iq2s_grid32[(qa & 0xFFu)         | ((qh << 8) & 0x300u)];
        const uint32_t g1  = vegajet::ggml_tables::iq2s_grid32[((qa >> 16) & 0xFFu) | ((qh << 6) & 0x300u)];
        const uint32_t g2  = vegajet::ggml_tables::iq2s_grid32[(qb & 0xFFu)         | ((qh << 4) & 0x300u)];
        const uint32_t g3  = vegajet::ggml_tables::iq2s_grid32[((qb >> 16) & 0xFFu) | ((qh << 2) & 0x300u)];

        uint4 packed;
        packed.x = iq2s_code4_swar(g0,      (qa >> 8)  & 0xFu) | (iq2s_code4_swar(g2,      (qb >> 8)  & 0xFu) << 4);
        packed.y = iq2s_code4_swar(g0 >> 2, (qa >> 12) & 0xFu) | (iq2s_code4_swar(g2 >> 2, (qb >> 12) & 0xFu) << 4);
        packed.z = iq2s_code4_swar(g1,      (qa >> 24) & 0xFu) | (iq2s_code4_swar(g3,      (qb >> 24) & 0xFu) << 4);
        packed.w = iq2s_code4_swar(g1 >> 2,  qa >> 28)         | (iq2s_code4_swar(g3 >> 2,  qb >> 28)         << 4);
        *reinterpret_cast<uint4*>(w_dst + part * 16u) = packed;
    }

    const uint16_t sc16 = static_cast<uint16_t>(((hs2 >> 8) & 0xFFu) | (((hs2 >> 24) & 0xFFu) << 8));
    reinterpret_cast<uint16_t*>(s_dst)[ib] = sc16;
    if ((ib & 7u) == 0u) {
        const uint32_t sb = ib >> 3;
        *reinterpret_cast<uint32_t*>(s_dst + 96u * 2u + sb * 4u) = reinterpret_cast<const uint32_t*>(qp_p + 120u)[sb];
    }
}

// ============================================================================
// Wave 5 Dynamic LRU-Cached Gate+Up Kernel (M >= 2 verify):
//   - Cache HIT  (exp_to_slot[exp] >= 0): L2-cached VJ_Q4_DOT8 (glc=0 slc=0, 0 sm_grid!)
//   - Cache MISS (exp_to_slot[exp] <  0): lazy 4-KiB sm_grid32 + iq2s_perm4 fallback
//   - 100% bit-exact output on both HIT and MISS!
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 4)
void vj_moe_gu_dot8_lru_kernel(
    const uint4*   __restrict__ wgu_iq2s,
    const uint8_t* __restrict__ cache_w,
    const uint8_t* __restrict__ cache_sc,
    const int32_t* __restrict__ exp_to_slot,
    const int8_t*  __restrict__ x_q8,
    const float2*  __restrict__ x_ds,
    const int32_t* __restrict__ top_ids,
    int8_t*        __restrict__ mid_q8,
    float*         __restrict__ mid_d,
    uint32_t M,
    uint32_t local_ff,
    uint32_t w_stride,
    uint32_t s_stride) {
    __shared__ uint32_t sm_grid32[1024]; // 4,096 B (loaded lazily ONLY on cache miss!)
    __shared__ uint4    sm_xq[192];      // 3,072 B
    __shared__ float    sm_xd[96];       //   384 B
    __shared__ float    sm_y[32];        //   128 B -> 7,680 B total

    const uint32_t m      = blockIdx.y;
    const uint32_t nblk32 = local_ff >> 5;
    const uint32_t slot   = blockIdx.x / nblk32;
    const uint32_t rblk   = blockIdx.x - slot * nblk32;
    if (m >= M || slot >= 8u) return;

    const uint32_t tid    = threadIdx.x;
    const uint32_t exp_id = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(top_ids[static_cast<uint64_t>(m) * 8u + slot]));
    const int32_t  cslot  = __builtin_amdgcn_readfirstlane(exp_to_slot[exp_id]);

    const uint4*  xq4_g = reinterpret_cast<const uint4*>(x_q8 + static_cast<uint64_t>(m) * 3072u);
    const float2* xds_g = x_ds + static_cast<uint64_t>(m) * 96u;
    if (tid < 192u) sm_xq[tid] = xq4_g[tid];
    if (tid <  96u) sm_xd[tid] = xds_g[tid].x;
    if (cslot < 0) {
        reinterpret_cast<uint4*>(sm_grid32)[tid] = reinterpret_cast<const uint4*>(vegajet::ggml_tables::iq2s_grid32)[tid];
    }
    __syncthreads();

    const uint32_t sub_id = tid >> 4;
    const uint32_t lane16 = tid & 15u;

    if (cslot >= 0) {
        #pragma nounroll
        for (uint32_t half = 0u; half < 2u; ++half) {
            const uint32_t row = rblk * 32u + half * 16u + sub_id;
            const float2 gu = vegajet::dot8_dev::vj_dot8_gu_row_iq2s<6, 1, true>(
                cache_w, w_stride, cache_sc, s_stride, sm_xq, sm_xd,
                static_cast<uint32_t>(cslot) * local_ff, row, lane16);
            if (lane16 == 0u) sm_y[half * 16u + sub_id] = (gu.x / (1.0f + __expf(-gu.x))) * gu.y;
        }
    } else {
        #pragma nounroll
        for (uint32_t half = 0u; half < 2u; ++half) {
            const uint32_t row = rblk * 32u + half * 16u + sub_id;
            const uint4* qp_p = wgu_iq2s + (static_cast<uint64_t>(exp_id) * local_ff + row) * 123ULL;
            float ag = 0.0f, au = 0.0f;
            #pragma nounroll
            for (uint32_t it = 0u; it < 6u; ++it) {
                const uint32_t ib  = it * 16u + lane16;
                const uint4    wq  = qp_p[ib];
                const uint32_t hs2 = reinterpret_cast<const uint32_t*>(qp_p + 96u)[ib];
                const uint32_t d2  = reinterpret_cast<const uint32_t*>(qp_p + 120u)[ib >> 3];
                const uint4    xa  = sm_xq[ib * 2u];
                const uint4    xb  = sm_xq[ib * 2u + 1u];
                const float    xda = sm_xd[ib];
                #pragma nounroll
                for (uint32_t part = 0u; part < 2u; ++part) {
                    const uint32_t qa  = part ? wq.z : wq.x;
                    const uint32_t qb  = part ? wq.w : wq.y;
                    const uint32_t hsp = part ? (hs2 >> 16) : (hs2 & 0xFFFFu);
                    const uint32_t qh  = hsp & 0xFFu, sc = hsp >> 8;
                    const float pd = vegajet::dot8_dev::h2f(part ? (d2 >> 16) : d2) * 0.25f * xda;
                    float acc = part ? au : ag;
                    const uint32_t g0 = sm_grid32[(qa & 0xFFu)         | ((qh << 8) & 0x300u)];
                    const uint32_t g1 = sm_grid32[((qa >> 16) & 0xFFu) | ((qh << 6) & 0x300u)];
                    const uint32_t g2 = sm_grid32[(qb & 0xFFu)         | ((qh << 4) & 0x300u)];
                    const uint32_t g3 = sm_grid32[((qb >> 16) & 0xFFu) | ((qh << 2) & 0x300u)];
                    int32_t dp0 = vegajet::dot8_dev::sdot4(iq2s_perm4_lru(g0,      (qa >> 8)  & 0xFu), xa.x, 0);
                    dp0         = vegajet::dot8_dev::sdot4(iq2s_perm4_lru(g0 >> 2, (qa >> 12) & 0xFu), xa.y, dp0);
                    dp0         = vegajet::dot8_dev::sdot4(iq2s_perm4_lru(g1,      (qa >> 24) & 0xFu), xa.z, dp0);
                    dp0         = vegajet::dot8_dev::sdot4(iq2s_perm4_lru(g1 >> 2,  qa >> 28),          xa.w, dp0);
                    acc = fmaf(pd * (0.5f + static_cast<float>(sc & 0xFu)), static_cast<float>(dp0), acc);
                    int32_t dp1 = vegajet::dot8_dev::sdot4(iq2s_perm4_lru(g2,      (qb >> 8)  & 0xFu), xb.x, 0);
                    dp1         = vegajet::dot8_dev::sdot4(iq2s_perm4_lru(g2 >> 2, (qb >> 12) & 0xFu), xb.y, dp1);
                    dp1         = vegajet::dot8_dev::sdot4(iq2s_perm4_lru(g3,      (qb >> 24) & 0xFu), xb.z, dp1);
                    dp1         = vegajet::dot8_dev::sdot4(iq2s_perm4_lru(g3 >> 2,  qb >> 28),          xb.w, dp1);
                    acc = fmaf(pd * (0.5f + static_cast<float>(sc >> 4)), static_cast<float>(dp1), acc);
                    if (part) au = acc; else ag = acc;
                }
            }
            ag = vegajet::dot8_dev::sum16(ag);
            au = vegajet::dot8_dev::sum16(au);
            if (lane16 == 0u) sm_y[half * 16u + sub_id] = (ag / (1.0f + __expf(-ag))) * au;
        }
    }
    __syncthreads();

    if (tid < 32u) {
        float y = sm_y[tid];
        float amax = fabsf(y);
        amax = fmaxf(amax, __shfl_xor(amax, 16, 32));
        amax = fmaxf(amax, __shfl_xor(amax, 8,  32));
        amax = fmaxf(amax, __shfl_xor(amax, 4,  32));
        amax = fmaxf(amax, __shfl_xor(amax, 2,  32));
        amax = fmaxf(amax, __shfl_xor(amax, 1,  32));
        const float d   = amax * (1.0f / 127.0f);
        const float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        mid_q8[(static_cast<uint64_t>(m) * 8u + slot) * local_ff + rblk * 32u + tid] =
            static_cast<int8_t>(max(-127, min(127, __float2int_rn(y * inv))));
        if (tid == 0u) mid_d[(static_cast<uint64_t>(m) * 8u + slot) * nblk32 + rblk] = d;
    }
}
