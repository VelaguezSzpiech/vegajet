// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Opt-in 240-WG deterministic Split-K router + shared Gate/Up front (separate from the fused
// ornith_p2p_ar_moe_m1_front_s240_kernel and from ornith_moe_m1_router_shared_kernel). Used by the
// per-layer M=1 path when VEGAJET_ROUTER_S240=1 (engine A/B: also VEGAJET_ENABLE_SUPER=0).
//
// Phase 1 (ornith_moe_router_shgu_s240_kernel), exactly 240 WGs x 256 threads:
//   (sh_rows + 256) row-parts per split part, spread over the 239 weight WGs by two round-robin group
//   streams (shared Gate/Up weight 1, router weight 4). Each 32-lane group owns one row-part:
//   VJ_Q4_A8 shared Gate/Up rows use the engine's INT8-dot arithmetic, router experts use the engine's
//   FP16 FMA chain. Both keep the per-lane block order (lane, lane+32, ...) and block32_sum of the
//   existing kernels, so split_k == 1 and xor_mask == 0 reproduces their outputs bit-for-bit.
//   The shared gate scalar is one WG with 64 lanes (wave64_sum), matching the 33-WG router kernel.
//   Weights are read straight into registers (uint4/uint32; no global->LDS), with an optional per-plane
//   256-B XOR tile swizzle whose formula comes from core/router_s240_layout.hpp (same header as the host
//   packer). XOR mode and split_k are independent runtime arguments.
//
// Phase 2 (ornith_moe_router_shgu_s240_reduce_kernel): sums partial part 0..split_k-1 in fixed order.
//   No FP atomics, no grid barrier, no sense slot: a repeat run is bit-identical.
#include "ops/kernels/vj_ornith_common.hiph"

#include "core/router_s240_layout.hpp"

namespace {

using vegajet::router_s240::Phase1Args;
using vegajet::router_s240::ReduceArgs;

// One VJ_Q4_A8 row-part. `nb` = D/32 blocks; [b_beg, b_end) is this split's contiguous block range.
// Arithmetic is copied from ornith_moe_m1_router_shared_kernel's shared Gate/Up branch: the plain loop
// (lane, lane+32, ...) produces the same per-lane order as its 3-block load batches.
__device__ __forceinline__ float shgu_row_part(const Phase1Args& a, uint32_t row, uint32_t b_beg,
                                               uint32_t b_end, uint32_t lane) {
    float acc = 0.0f;
    if (b_beg < b_end) {
        const uint4* w4 = reinterpret_cast<const uint4*>(
            reinterpret_cast<const uint8_t*>(a.sh_w_qs) + static_cast<uint64_t>(row) * a.sh_qs_stride_bytes);
        const uint32_t* dm = reinterpret_cast<const uint32_t*>(a.sh_w_dm) +
                             static_cast<uint64_t>(row) * a.sh_dm_stride_u32;
        const uint4* x4 = reinterpret_cast<const uint4*>(a.x_q8);
        const float2* xds = reinterpret_cast<const float2*>(a.x_ds);
        const uint32_t xor_mask = a.sh_xor_mask;
        #pragma nounroll
        for (uint32_t b = b_beg + lane; b < b_end; b += 32u) {
            const uint32_t wi = ((b >> 4) ^ (row & xor_mask)) * 16u + (b & 15u);
            const uint4    wv = vegajet::ornith_dev::ldg_u32x4_stream(w4 + wi);
            const uint32_t dmv = vegajet::ornith_dev::ldg_u32_stream(dm + b);
            const uint4    xa = x4[b * 2u];
            const uint4    xb = x4[b * 2u + 1u];
            const float2   xd = xds[b];
            int32_t dp = vegajet::ornith_dev::sdot4(wv.x & 0x0F0F0F0Fu, xa.x, 0);
            dp         = vegajet::ornith_dev::sdot4(wv.y & 0x0F0F0F0Fu, xa.y, dp);
            dp         = vegajet::ornith_dev::sdot4(wv.z & 0x0F0F0F0Fu, xa.z, dp);
            dp         = vegajet::ornith_dev::sdot4(wv.w & 0x0F0F0F0Fu, xa.w, dp);
            dp         = vegajet::ornith_dev::sdot4((wv.x >> 4) & 0x0F0F0F0Fu, xb.x, dp);
            dp         = vegajet::ornith_dev::sdot4((wv.y >> 4) & 0x0F0F0F0Fu, xb.y, dp);
            dp         = vegajet::ornith_dev::sdot4((wv.z >> 4) & 0x0F0F0F0Fu, xb.z, dp);
            dp         = vegajet::ornith_dev::sdot4((wv.w >> 4) & 0x0F0F0F0Fu, xb.w, dp);
            const float wd = vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(dmv & 0xFFFFu));
            const float wm = vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(dmv >> 16));
            acc = fmaf(wd * xd.x, static_cast<float>(dp), fmaf(wm, xd.y, acc));
        }
    }
    return vegajet::ornith_dev::block32_sum(acc);
}

// One FP16 router expert row-part. 32 weights = 4 uint4; [b_beg, b_end) is the split's block range.
__device__ __forceinline__ float router_row_part(const Phase1Args& a, uint32_t row, uint32_t b_beg,
                                                 uint32_t b_end, uint32_t lane) {
    float acc = 0.0f;
    const uint32_t i_beg = b_beg * 4u;
    const uint32_t i_end = b_end * 4u;
    if (i_beg < i_end) {
        const uint16_t* rp = reinterpret_cast<const uint16_t*>(a.w_router_f16) +
                             static_cast<uint64_t>(row) * a.wr_stride_elems;
        const uint4*    w4 = reinterpret_cast<const uint4*>(rp);
        const float4*   x4f = reinterpret_cast<const float4*>(a.x_norm);
        const uint32_t  xor_mask = a.wr_xor_mask;
        #pragma nounroll
        for (uint32_t i = i_beg + lane; i < i_end; i += 32u) {
            const uint32_t wi = ((i >> 4) ^ (row & xor_mask)) * 16u + (i & 15u);
            const uint4    wv = vegajet::ornith_dev::ldg_u32x4_stream(w4 + wi);
            const float4   xa = x4f[i * 2u];
            const float4   xb = x4f[i * 2u + 1u];
            const float2   h0 = __half22float2(*reinterpret_cast<const __half2*>(&wv.x));
            const float2   h1 = __half22float2(*reinterpret_cast<const __half2*>(&wv.y));
            const float2   h2 = __half22float2(*reinterpret_cast<const __half2*>(&wv.z));
            const float2   h3 = __half22float2(*reinterpret_cast<const __half2*>(&wv.w));
            acc = fmaf(h0.x, xa.x, fmaf(h0.y, xa.y, fmaf(h1.x, xa.z, fmaf(h1.y, xa.w, acc))));
            acc = fmaf(h2.x, xb.x, fmaf(h2.y, xb.y, fmaf(h3.x, xb.z, fmaf(h3.y, xb.w, acc))));
        }
    }
    return vegajet::ornith_dev::block32_sum(acc);
}

} // namespace

extern "C" __global__ __launch_bounds__(vegajet::router_s240::kThreads, 4)
void ornith_moe_router_shgu_s240_kernel(const vegajet::router_s240::Phase1Args a) {
    using namespace vegajet::router_s240;
    const uint32_t wg   = blockIdx.x;
    const uint32_t tid  = threadIdx.x;
    const uint32_t grp  = tid >> 5;
    const uint32_t lane = tid & 31u;
    if (wg >= kPhase1Wgs || a.split_k == 0u || a.split_k > kMaxSplitK || a.sh_rows == 0u || a.D < 32u ||
        (a.D & 31u) != 0u) {
        return;
    }

    const uint32_t nb = a.D >> 5;
    const TaskPlan plan = make_task_plan(a.sh_rows, a.split_k);
    if (wg < kWorkerWgs) {
        const uint32_t first = grp * kWorkerWgs + wg;
        for (uint32_t t = first; t < plan.sh_tasks; t += plan.stride) {
            uint32_t part = 0u, row = 0u;
            sh_task_split(t, a.sh_rows, a.split_k, &part, &row);
            const float acc = shgu_row_part(a, row, split_begin(nb, a.split_k, part),
                                            split_end(nb, a.split_k, part), lane);
            if (lane == 0u) {
                reinterpret_cast<float*>(a.sh_part)[static_cast<uint64_t>(part) * a.sh_rows + row] = acc;
            }
        }
        for (uint32_t t = first; t < plan.rtr_tasks; t += plan.stride) {
            uint32_t part = 0u, row = 0u;
            rtr_task_split(t, a.split_k, &part, &row);
            const float acc = router_row_part(a, row, split_begin(nb, a.split_k, part),
                                              split_end(nb, a.split_k, part), lane);
            if (lane == 0u) {
                reinterpret_cast<float*>(a.rtr_part)[static_cast<uint64_t>(part) * kRouterRows + row] = acc;
            }
        }
    }

    // Shared gate scalar: exactly 64 lanes on the last WG, same per-lane order and wave64 tree as the
    // 33-WG ornith_moe_router_fp16_logits_kernel gate branch (split_k == 1 is bit-identical).
    if (wg == kPhase1Wgs - 1u && tid < 64u) {
        const float4* w4  = reinterpret_cast<const float4*>(a.w_sh_gate);
        const float4* x4f = reinterpret_cast<const float4*>(a.x_norm);
        const uint32_t n4 = a.D >> 2;
        for (uint32_t part = 0; part < a.split_k; ++part) {
            const uint32_t b_beg = split_begin(nb, a.split_k, part);
            const uint32_t b_end = split_end(nb, a.split_k, part);
            const uint32_t i_beg = b_beg * 8u;  // 32 floats per block = 8 float4
            const uint32_t i_end = b_end * 8u;
            float acc = 0.0f;
            #pragma nounroll
            for (uint32_t i = i_beg + tid; i < i_end && i < n4; i += 64u) {
                const float4 wv = w4[i];
                const float4 xv = x4f[i];
                acc = fmaf(wv.x, xv.x, fmaf(wv.y, xv.y, fmaf(wv.z, xv.z, fmaf(wv.w, xv.w, acc))));
            }
            acc = vegajet::ornith_dev::wave64_sum(acc);
            if (tid == 0u) reinterpret_cast<float*>(a.gate_part)[part] = acc;
        }
    }
}

extern "C" __global__ __launch_bounds__(vegajet::router_s240::kThreads, 4)
void ornith_moe_router_shgu_s240_reduce_kernel(const vegajet::router_s240::ReduceArgs a) {
    using namespace vegajet::router_s240;
    const uint32_t idx = blockIdx.x * kThreads + threadIdx.x;
    const uint32_t total = a.sh_rows + kRouterRows + 1u;
    if (idx >= total || a.split_k == 0u) return;

    const float* sh_part = reinterpret_cast<const float*>(a.sh_part);
    const float* rtr_part = reinterpret_cast<const float*>(a.rtr_part);
    const float* gate_part = reinterpret_cast<const float*>(a.gate_part);

    if (idx < a.sh_rows) {
        float s = 0.0f;
        if (a.split_k == 1u) {
            s = sh_part[idx];
        } else {
            #pragma nounroll
            for (uint32_t p = 0; p < a.split_k; ++p) s += sh_part[static_cast<uint64_t>(p) * a.sh_rows + idx];
        }
        reinterpret_cast<float*>(a.sh_gu)[idx] = s;
    } else if (idx < a.sh_rows + kRouterRows) {
        const uint32_t row = idx - a.sh_rows;
        float s = 0.0f;
        if (a.split_k == 1u) {
            s = rtr_part[row];
        } else {
            #pragma nounroll
            for (uint32_t p = 0; p < a.split_k; ++p) s += rtr_part[static_cast<uint64_t>(p) * kRouterRows + row];
        }
        reinterpret_cast<float*>(a.logits)[row] = s;
    } else {
        float s = 0.0f;
        if (a.split_k == 1u) {
            s = gate_part[0];
        } else {
            #pragma nounroll
            for (uint32_t p = 0; p < a.split_k; ++p) s += gate_part[p];
        }
        reinterpret_cast<float*>(a.sh_gate_sc)[0] = 1.0f / (1.0f + __expf(-s));
    }
}
