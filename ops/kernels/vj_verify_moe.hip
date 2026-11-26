// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Exact pair reuse for Q3 speculative verification. Default kernels stay unchanged.
#include <hip/hip_runtime.h>
#include <stdint.h>
#include "ops/verify_routes.hpp"
#include "ops/kernels/vj_moe_dot8.hiph"
namespace vegajet { namespace dot8_dev {
template <uint32_t kNIt, uint32_t kBatch, bool kCached = false, typename WPtr = const uint8_t*, typename SPtr = const uint8_t*>
__device__ __forceinline__ void verify_q3_pair_row(
    WPtr __restrict__ w_gu, uint32_t w_stride,
    SPtr __restrict__ w_sc, uint32_t s_stride,
    const uint4* sm_xq, const float* sm_xd,
    uint32_t erow_base, uint32_t row, uint32_t lane16, bool paired, float2 (&out)[2]) {
    static_assert(kNIt % kBatch == 0u, "kBatch must divide kNIt");
    constexpr bool kIsG = __is_same(WPtr, GPtr<const uint8_t>);
    using U4P  = typename CondT<kIsG, GPtr<const uint4>, const uint4*>::type;
    using U2P  = typename CondT<kIsG, GPtr<const uint2>, const uint2*>::type;
    using U16P = typename CondT<kIsG, GPtr<const uint16_t>, const uint16_t*>::type;
    using U32P = typename CondT<kIsG, GPtr<const uint32_t>, const uint32_t*>::type;
    constexpr uint32_t nblk = kNIt * 16u;
    constexpr uint32_t nsb  = nblk >> 3;
    WPtr wb = w_gu + uniform_u64(static_cast<uint64_t>(erow_base) * w_stride);
    SPtr sb = w_sc + uniform_u64(static_cast<uint64_t>(erow_base) * s_stride);
    const uint32_t r_w_off = row * w_stride;
    const uint32_t r_s_off = row * s_stride;
    uint32_t d_off = r_s_off + nblk * 2u + lane16 * 4u;
    asm volatile("" : "+v"(d_off));
    const uint32_t d2_mine = (lane16 < nsb)
        ? (kCached ? *reinterpret_cast<U32P>(sb + d_off) : ldg_u32_nt(reinterpret_cast<U32P>(sb + d_off)))
        : 0u;

    float ag[2] = {}, au[2] = {};
    #pragma nounroll
    for (uint32_t it0 = 0u; it0 < kNIt; it0 += kBatch) {
        uint4    mgv[kBatch];
        uint2    sgv[kBatch];
        uint32_t scv[kBatch];
        #pragma unroll
        for (uint32_t k = 0u; k < kBatch; ++k) {
            const uint32_t ib = (it0 + k) * 16u + lane16;
            uint32_t m_off = r_w_off + ib * 16u;
            uint32_t g_off = r_w_off + nblk * 16u + ib * 8u;
            uint32_t s_off = r_s_off + ib * 2u;
            asm volatile("" : "+v"(m_off), "+v"(g_off), "+v"(s_off));
            if constexpr (kCached) {
                mgv[k] = *reinterpret_cast<U4P>(wb + m_off);
                sgv[k] = *reinterpret_cast<U2P>(wb + g_off);
                scv[k] = *reinterpret_cast<U16P>(sb + s_off);
            } else {
                mgv[k] = ldg_x4_nt(reinterpret_cast<U4P>(wb + m_off));
                sgv[k] = ldg_x2_nt(reinterpret_cast<U2P>(wb + g_off));
                scv[k] = ldg_u16_nt(reinterpret_cast<U16P>(sb + s_off));
            }
        }
        __builtin_amdgcn_sched_barrier(0);
        #pragma unroll
        for (uint32_t k = 0u; k < kBatch; ++k) {
            if constexpr (kBatch == 2u) {
                if (k == 0u) asm volatile("s_waitcnt vmcnt(3)" : "+v"(mgv[0].x), "+v"(sgv[0].x), "+v"(scv[0]) :: "memory");
                else         asm volatile("s_waitcnt vmcnt(0)" : "+v"(mgv[1].x), "+v"(sgv[1].x), "+v"(scv[1]) :: "memory");
            } else if constexpr (kBatch == 3u) {
                if (k == 0u)      asm volatile("s_waitcnt vmcnt(6)" : "+v"(mgv[0].x), "+v"(sgv[0].x), "+v"(scv[0]) :: "memory");
                else if (k == 1u) asm volatile("s_waitcnt vmcnt(3)" : "+v"(mgv[1].x), "+v"(sgv[1].x), "+v"(scv[1]) :: "memory");
                else              asm volatile("s_waitcnt vmcnt(0)" : "+v"(mgv[2].x), "+v"(sgv[2].x), "+v"(scv[2]) :: "memory");
            }
            const uint32_t ib  = (it0 + k) * 16u + lane16;
            const uint32_t d2k = __shfl(static_cast<int>(d2_mine), static_cast<int>(ib >> 3), 16);
            #pragma unroll
            for (uint32_t t = 0; t < 2u; ++t) {
            if (t == 1u && !paired) break;
            const float    xda = sm_xd[t * 96u + ib];
            const float    pg  = h2f(d2k) * 0.25f * xda;
            const float    pu  = h2f(d2k >> 16) * 0.25f * xda;
            const uint4    xa  = sm_xq[t * 192u + ib * 2u];
            const int32_t  g0  = dot16_lo_q3(mgv[k].x, sgv[k].x, xa);
            const int32_t  u0  = dot16_lo_q3(mgv[k].z, sgv[k].y, xa);
            ag[t] = fmaf(pg * (0.5f + static_cast<float>(scv[k] & 0xFu)), static_cast<float>(g0), ag[t]);
            au[t] = fmaf(pu * (0.5f + static_cast<float>((scv[k] >> 8) & 0xFu)), static_cast<float>(u0), au[t]);
            const uint4    xb  = sm_xq[t * 192u + ib * 2u + 1u];
            const int32_t  g1  = dot16_hi_q3(mgv[k].y, sgv[k].x, xb);
            const int32_t  u1  = dot16_hi_q3(mgv[k].w, sgv[k].y, xb);
            ag[t] = fmaf(pg * (0.5f + static_cast<float>((scv[k] >> 4) & 0xFu)), static_cast<float>(g1), ag[t]);
            au[t] = fmaf(pu * (0.5f + static_cast<float>((scv[k] >> 12) & 0xFu)), static_cast<float>(u1), au[t]);
            }
        }
    }
    for (uint32_t t = 0; t < 2u; ++t) out[t] = make_float2(sum16(ag[t]), sum16(au[t]));
}

}}
extern "C" __global__ __launch_bounds__(256, 4)
void vj_verify_gu_q3_pair_kernel(
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
    __shared__ uint4 sm_xq[384];
    __shared__ float sm_xd[192];
    __shared__ float sm_y[64];

    const uint32_t m      = blockIdx.y;
    const uint32_t nblk32 = local_ff >> 5;
    const uint32_t slot   = blockIdx.x / nblk32;
    const uint32_t rblk   = blockIdx.x - slot * nblk32;
    if (m >= M || slot >= 8u) return;

    // Pair occurrences in route order. Odd occurrences are produced by their even predecessor.
    const uint32_t route = m * 8u + slot;
    const uint32_t exp_id = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(top_ids[route]));
    const uint32_t partner = vegajet::verify::partner(top_ids, M * 8u, route);
    if (partner == M * 8u) return;
    const uint32_t rows[2] = {m, partner / 8u};
    for (uint32_t t = 0; t < 2u; ++t) {
        const auto* xq4_g = reinterpret_cast<const uint4*>(x_q8 + static_cast<uint64_t>(rows[t]) * 3072u);
        const auto* xds_g = x_ds + static_cast<uint64_t>(rows[t]) * 96u;
        if (threadIdx.x < 192u) sm_xq[t * 192u + threadIdx.x] = xq4_g[threadIdx.x];
        if (threadIdx.x < 96u) sm_xd[t * 96u + threadIdx.x] = xds_g[threadIdx.x].x;
    }
    __syncthreads();
    const uint32_t sub_id = threadIdx.x >> 4;
    const uint32_t lane16 = threadIdx.x & 15u;

    #pragma nounroll
    for (uint32_t half = 0u; half < 2u; ++half) {
        const uint32_t row = rblk * 32u + half * 16u + sub_id;
        float2 gu[2];
        vegajet::dot8_dev::verify_q3_pair_row<6, 1, true>(w_gu, w_stride, w_sc, s_stride, sm_xq, sm_xd,
                                                       exp_id * local_ff, row, lane16, partner != route, gu);
        for (uint32_t t = 0; t < 2u; ++t)
            if (lane16 == 0u) sm_y[t * 32u + half * 16u + sub_id] = (gu[t].x / (1.0f + __expf(-gu[t].x))) * gu[t].y;
    }
    __syncthreads();

    for (uint32_t t = 0; t < 2u; ++t) {
      if (t == 1u && partner == route) break;
      const uint32_t dst_route = t ? partner : route;
      if (threadIdx.x < 32u) {
        float y = sm_y[t * 32u + threadIdx.x];
        float amax = fabsf(y);
        amax = fmaxf(amax, __shfl_xor(amax, 16, 32));
        amax = fmaxf(amax, __shfl_xor(amax, 8,  32));
        amax = fmaxf(amax, __shfl_xor(amax, 4,  32));
        amax = fmaxf(amax, __shfl_xor(amax, 2,  32));
        amax = fmaxf(amax, __shfl_xor(amax, 1,  32));
        const float d   = amax * (1.0f / 127.0f);
        const float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        mid_q8[static_cast<uint64_t>(dst_route) * local_ff + rblk * 32u + threadIdx.x] =
            static_cast<int8_t>(max(-127, min(127, __float2int_rn(y * inv))));
        if (threadIdx.x == 0u) mid_d[static_cast<uint64_t>(dst_route) * nblk32 + rblk] = d;
    }
    }
}
