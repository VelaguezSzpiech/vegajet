// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "ops/kernels/vj_ornith_common.hiph"
#include "core/acceptmoe.hpp"

// ============================================================================
// 8. 33-WG FP16 MoE Router Logits + Shared Gate (8-token tiled for M > 1, 0 B spill!)
//    32 WGs x 8 experts (32 lanes/expert, 12 uint4/lane exact!) + WG 32 = shared gate!
//    At M > 1, each WG tiles up to 8 tokens so the 1.57 MB FP16 router stays L2-resident
//    and weight reads drop 8x while preserving bit-identical FMA accumulation order.
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_moe_router_fp16_logits_kernel(
    const uint16_t* __restrict__ w_router_f16, // [256, D] FP16
    const float*    __restrict__ w_sh_gate,    // [D] FP32
    const float*    __restrict__ x_norm,       // [M, D] FP32
    float*          __restrict__ logits_out,   // [M, 256] FP32
    float*          __restrict__ sh_gate_out,  // [M] FP32
    uint32_t M,
    uint32_t D) {
    const uint32_t m0   = blockIdx.y * 8u;
    const uint32_t grp  = threadIdx.x >> 5;  // 0..7 (8 experts per WG!)
    const uint32_t lane = threadIdx.x & 31u;
    if (m0 >= M) return;
    const uint32_t T = min(8u, M - m0);

    if (blockIdx.x < 32u) {
        const uint32_t e  = blockIdx.x * 8u + grp;
        const uint32_t wr_stride = (D == 3072u) ? 3200u : D;
        const uint4*   w4 = reinterpret_cast<const uint4*>(w_router_f16 + static_cast<uint64_t>(e) * wr_stride);
        const uint32_t n8 = D >> 3; // 384 for D=3072 -> 12 exact iterations/lane!
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        #pragma nounroll
        for (uint32_t i = lane; i < n8; i += 32u) {
            const uint4  wv = (M == 1u) ? vegajet::ornith_dev::ldg_u32x4_stream(w4 + i) : w4[i];
            const float2 h0 = __half22float2(*reinterpret_cast<const __half2*>(&wv.x));
            const float2 h1 = __half22float2(*reinterpret_cast<const __half2*>(&wv.y));
            const float2 h2 = __half22float2(*reinterpret_cast<const __half2*>(&wv.z));
            const float2 h3 = __half22float2(*reinterpret_cast<const __half2*>(&wv.w));
            for (uint32_t t = 0u; t < T; ++t) {
                const float4* x4 = reinterpret_cast<const float4*>(x_norm + static_cast<uint64_t>(m0 + t) * D);
                const float4 xa = x4[i * 2u];
                const float4 xb = x4[i * 2u + 1u];
                float a = acc[t];
                a = fmaf(h0.x, xa.x, fmaf(h0.y, xa.y, fmaf(h1.x, xa.z, fmaf(h1.y, xa.w, a))));
                a = fmaf(h2.x, xb.x, fmaf(h2.y, xb.y, fmaf(h3.x, xb.z, fmaf(h3.y, xb.w, a))));
                acc[t] = a;
            }
        }
        for (uint32_t t = 0u; t < T; ++t) {
            const float s = vegajet::ornith_dev::block32_sum(acc[t]);
            if (lane == 0u) logits_out[static_cast<uint64_t>(m0 + t) * 256u + e] = s;
        }
    } else if (blockIdx.x == 32u && grp < 2u) {
        const uint32_t lane64 = threadIdx.x; // 0..63
        const float4* w4 = reinterpret_cast<const float4*>(w_sh_gate);
        const uint32_t n4 = D >> 2;
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        #pragma nounroll
        for (uint32_t i = lane64; i < n4; i += 64u) {
            const float4 wv = w4[i];
            for (uint32_t t = 0u; t < T; ++t) {
                const float4* x4 = reinterpret_cast<const float4*>(x_norm + static_cast<uint64_t>(m0 + t) * D);
                const float4 xv = x4[i];
                acc[t] = fmaf(wv.x, xv.x, fmaf(wv.y, xv.y, fmaf(wv.z, xv.z, fmaf(wv.w, xv.w, acc[t]))));
            }
        }
        for (uint32_t t = 0u; t < T; ++t) {
            const float s = vegajet::ornith_dev::wave64_sum(acc[t]);
            if (lane64 == 0u) sh_gate_out[m0 + t] = 1.0f / (1.0f + __expf(-s));
        }
    }
}

// 256-Thread Parallel Wave64 DPP Softmax + Top-8 + Renormalize (<0.5 us/layer vs 65 us serial!)
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_moe_router_top8_kernel(
    const float* __restrict__ logits_in,
    int32_t*     __restrict__ top_ids,
    float*       __restrict__ top_w,
    uint32_t M) {
    const uint32_t m = blockIdx.x;
    const uint32_t e = threadIdx.x;
    if (m >= M || e >= 64u) return;
    vegajet::ornith_dev::fast_softmax_top8_wave64(
        logits_in + static_cast<uint64_t>(m) * 256u,
        top_ids   + static_cast<uint64_t>(m) * 8u,
        top_w     + static_cast<uint64_t>(m) * 8u, e);
}

// ============================================================================
// 8b. M=1 fused MoE front end (Phase 2, super-kernel B stage 1+2): ONE dispatch replaces
//     router logits (33 WG) + shared-expert Gate/Up GEMV (rows/8 WG) + top-8 (1 WG) + SwiGLU quant (1 WG).
//     Stage 1: WG 0..31 router logits, WG 32 shared gate, WG 33.. shared Gate/Up GEMV (8 rows/WG).
//     Grid barrier over all workgroups, then WG 0 = softmax/top-8/renorm, WG 32 = SwiGLU*gate + b=32 INT8 quant.
//     Arithmetic is copied 1:1 from kernels 8, 4a and 9a so results are bit-identical. Values produced by other
//     workgroups are re-read after the barrier with glc=1 (non-temporal) loads only.
//     Grid = (33 + sh_rows/8) * 256 threads. bar = 128 B zeroed scratch, sense = value unique per dispatch.
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_moe_m1_router_shared_kernel(
    const uint16_t* __restrict__ w_router_f16, // [256, D] FP16
    const float*    __restrict__ w_sh_gate,    // [D] FP32
    const float*    __restrict__ x_norm,       // [D] FP32
    const uint8_t*  __restrict__ sh_w_qs,      // shared Gate/Up VJ_Q4_A8 rows
    const uint32_t* __restrict__ sh_w_dm,
    const int8_t*   __restrict__ x_q8,
    const float2*   __restrict__ x_ds,
    float*          __restrict__ logits,       // [256] scratch
    float*          __restrict__ sh_gu,        // [2 * local_ff] scratch
    int32_t*        __restrict__ top_ids,      // [8]
    float*          __restrict__ top_w,        // [8]
    int8_t*         __restrict__ sh_q8,        // [local_ff]
    float2*         __restrict__ sh_ds,        // [local_ff / 32]
    uint32_t*       __restrict__ bar,
    uint32_t D,
    uint32_t sh_rows,
    uint32_t sh_row_stride_bytes,
    uint32_t local_ff,
    uint32_t sense) {
    const uint32_t wg   = blockIdx.x;
    const uint32_t tid  = threadIdx.x;
    const uint32_t grp  = tid >> 5;
    const uint32_t l32  = tid & 31u;

    __shared__ float   sm_v[4];
    __shared__ int32_t sm_i[4];
    __shared__ float   sm_gate;
    alignas(16) __shared__ float sm_p[256]; // read back as float4 (ds_read_b128)
    __shared__ float   sm_tw[8];
    __shared__ int32_t sm_ti[8];

    const float4* x4f = reinterpret_cast<const float4*>(x_norm);

    // ---------------- stage 1 ----------------
    // Balance router FP16 rows (256 x 6400 B odd-256B TCC stride) across 64 WGs (all 60 CUs!) by pairing
    // 4 router rows (grp 0..3) with 4 shared Gate/Up rows (grp 4..7) on WGs 0..63.
    const uint32_t gate_wg = gridDim.x - 1u;
    const bool split64 = (sh_rows >= 256u && gate_wg >= 64u);
    if (wg == gate_wg) {
        if (grp < 2u) {
            const float4* w4 = reinterpret_cast<const float4*>(w_sh_gate);
            const uint32_t n4 = D >> 2;
            float acc = 0.0f;
            #pragma nounroll
            for (uint32_t i = tid; i < n4; i += 64u) {
                const float4 wv = w4[i];
                const float4 xv = x4f[i];
                acc = fmaf(wv.x, xv.x, fmaf(wv.y, xv.y, fmaf(wv.z, xv.z, fmaf(wv.w, xv.w, acc))));
            }
            acc = vegajet::ornith_dev::wave64_sum(acc);
            if (tid == 0u) sm_gate = 1.0f / (1.0f + __expf(-acc));
        }
    } else if (split64 ? (wg < 64u && grp < 4u) : (wg < 32u)) {
        const uint32_t e  = split64 ? (wg * 4u + grp) : (wg * 8u + grp);
        const uint32_t wr_stride = (D == 3072u) ? 3200u : D;
        const uint4*   w4 = reinterpret_cast<const uint4*>(w_router_f16 + static_cast<uint64_t>(e) * wr_stride);
        const uint32_t n8 = D >> 3;
        float acc = 0.0f;
        const uint32_t i_last = n8 - 1u;
        #pragma nounroll
        for (uint32_t i0 = l32; i0 < n8; i0 += 128u) {
            uint4 wvs[4];
            #pragma unroll
            for (uint32_t r = 0u; r < 4u; ++r) wvs[r] = vegajet::ornith_dev::ldg_u32x4_stream(w4 + min(i0 + r * 32u, i_last));
            #pragma unroll
            for (uint32_t r = 0u; r < 4u; ++r) {
                const uint32_t i = i0 + r * 32u;
                if (i < n8) {
                    const uint4  wv = wvs[r];
                    const float4 xa = x4f[i * 2u];
                    const float4 xb = x4f[i * 2u + 1u];
                    const float2 h0 = __half22float2(*reinterpret_cast<const __half2*>(&wv.x));
                    const float2 h1 = __half22float2(*reinterpret_cast<const __half2*>(&wv.y));
                    const float2 h2 = __half22float2(*reinterpret_cast<const __half2*>(&wv.z));
                    const float2 h3 = __half22float2(*reinterpret_cast<const __half2*>(&wv.w));
                    acc = fmaf(h0.x, xa.x, fmaf(h0.y, xa.y, fmaf(h1.x, xa.z, fmaf(h1.y, xa.w, acc))));
                    acc = fmaf(h2.x, xb.x, fmaf(h2.y, xb.y, fmaf(h3.x, xb.z, fmaf(h3.y, xb.w, acc))));
                }
            }
        }
        acc = vegajet::ornith_dev::block32_sum(acc);
        if (l32 == 0u) logits[e] = acc;
    } else {
        const uint32_t row = split64
            ? ((wg < 64u) ? (wg * 4u + (grp - 4u)) : (256u + (wg - 64u) * 8u + grp))
            : ((wg - 32u) * 8u + grp);
        if (row < sh_rows) {
            const uint32_t  nb = D >> 5;
            const uint4*    w4 = reinterpret_cast<const uint4*>(sh_w_qs + static_cast<uint64_t>(row) * sh_row_stride_bytes);
            const uint32_t* dm = sh_w_dm + static_cast<uint64_t>(row) * nb;
            const uint4*    x4 = reinterpret_cast<const uint4*>(x_q8);
            // Same 3-blocks-per-round load batching as kernel 4a (bit-identical accumulation order).
            float acc = 0.0f;
            const uint32_t b_last = nb - 1u;
            #pragma nounroll
            for (uint32_t b0 = l32; b0 < nb; b0 += 96u) {
                uint4    wvs[3];
                uint32_t dms[3];
                #pragma unroll
                for (uint32_t r = 0u; r < 3u; ++r) {
                    const uint32_t bc = min(b0 + r * 32u, b_last);
                    wvs[r] = vegajet::ornith_dev::ldg_u32x4_stream(w4 + bc);
                    dms[r] = vegajet::ornith_dev::ldg_u32_stream(dm + bc);
                }
                #pragma unroll
                for (uint32_t r = 0u; r < 3u; ++r) {
                    const uint32_t bq = b0 + r * 32u;
                    if (bq < nb) {
                        const uint4    wv  = wvs[r];
                        const uint32_t dmv = dms[r];
                        const uint4    xa  = x4[bq * 2u];
                        const uint4    xb  = x4[bq * 2u + 1u];
                        const float2   xds = x_ds[bq];

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
                        acc = fmaf(wd * xds.x, static_cast<float>(dp), fmaf(wm, xds.y, acc));
                    }
                }
            }
            acc = vegajet::ornith_dev::block32_sum(acc);
            if (l32 == 0u) sh_gu[row] = acc;
        }
    }

    // ---------------- device-local grid barrier + L1 invalidate for L2-cached stage-2 reads ----------------
    __threadfence();
    vegajet::ornith_dev::grid_barrier_m(bar, gridDim.x, sense);
    if (tid == 0u) asm volatile("buffer_wbinvl1\n" ::: "memory");
    __syncthreads();

    // ---------------- stage 2 ----------------
    if (wg == 0u) {
        if (tid < 64u) vegajet::ornith_dev::fast_softmax_top8_wave64(logits, top_ids, top_w, tid);
    } else if (wg == gate_wg) {
        const float sc_g = sm_gate;
        const float* g_p = sh_gu;
        const float* u_p = g_p + local_ff;
        #pragma nounroll
        for (uint32_t blk = tid >> 5; blk < (local_ff >> 5); blk += 8u) {
            const uint32_t c = (blk << 5) + l32;
            const float gv = g_p[c];
            const float uv = u_p[c];
            const float y  = (gv / (1.0f + __expf(-gv))) * uv * sc_g;
            const float amax  = vegajet::ornith_dev::block32_max(fabsf(y));
            const float bsum  = vegajet::ornith_dev::block32_sum(y);
            const float d     = amax * (1.0f / 127.0f);
            const float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            sh_q8[c] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(y * inv_d))));
            if (l32 == 0u) sh_ds[blk] = make_float2(d, bsum);
        }
    }
}

// ============================================================================
// 9a. Shared Expert Fused SwiGLU * sh_gate -> Inline b=32 INT8 Quantize (+ out_ds)
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 6)
void ornith_swiglu_quant_q8_kernel(
    const float* __restrict__ sh_gu_in,
    const float* __restrict__ sh_gate_sc,
    int8_t*      __restrict__ out_q8,
    float2*      __restrict__ out_ds,
    uint32_t M,
    uint32_t local_ff) {
    const uint32_t m   = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t sub = tid & 31u;
    if (m >= M) return;

    const float sc_g = sh_gate_sc[m];
    const float* g_p = sh_gu_in + static_cast<uint64_t>(m) * (2u * local_ff);
    const float* u_p = g_p + local_ff;
    int8_t* oq       = out_q8 + static_cast<uint64_t>(m) * local_ff;
    float2* od       = out_ds + static_cast<uint64_t>(m) * (local_ff >> 5);

    #pragma nounroll
    for (uint32_t blk = tid >> 5; blk < (local_ff >> 5); blk += 8u) {
        const uint32_t c = (blk << 5) + sub;
        const float gv = g_p[c];
        const float y  = (gv / (1.0f + __expf(-gv))) * u_p[c] * sc_g;
        const float amax  = vegajet::ornith_dev::block32_max(fabsf(y));
        const float bsum  = vegajet::ornith_dev::block32_sum(y);
        const float d     = amax * (1.0f / 127.0f);
        const float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        oq[c] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(y * inv_d))));
        if (sub == 0u) od[blk] = make_float2(d, bsum);
    }
}

// ============================================================================
// 9b. LDS-Cached + 5-Inst SWAR Coalesced uint4 MERGED IQ2_S Gate+Up + SiLU*Mul + Quant!
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 3)
void ornith_moe_gu_iq2s_merged_kernel(
    const uint4*   __restrict__ wgu_merged,
    const int8_t*  __restrict__ x_q8,
    const float2*  __restrict__ x_ds,
    const int32_t* __restrict__ top_ids,
    int8_t*        __restrict__ mid_q8,
    float*         __restrict__ mid_d,
    uint32_t M,
    uint32_t local_ff) {
    __shared__ uint64_t sm_grid[1024];
    __shared__ uint4    sm_xq[192];
    __shared__ float    sm_xd[96];
    __shared__ float    sm_y[32];

    const uint32_t m      = blockIdx.y;
    const uint32_t nblk32 = local_ff >> 5;
    const uint32_t slot   = blockIdx.x / nblk32;
    const uint32_t rblk   = blockIdx.x - slot * nblk32;
    if (m >= M || slot >= 8u) return;

    reinterpret_cast<uint4*>(sm_grid)[threadIdx.x]        = reinterpret_cast<const uint4*>(vegajet::ggml_tables::iq2s_grid)[threadIdx.x];
    reinterpret_cast<uint4*>(sm_grid)[threadIdx.x + 256u] = reinterpret_cast<const uint4*>(vegajet::ggml_tables::iq2s_grid)[threadIdx.x + 256u];
    const uint4*  xq4_g = reinterpret_cast<const uint4*>(x_q8 + static_cast<uint64_t>(m) * 3072u);
    const float2* xds_g = x_ds + static_cast<uint64_t>(m) * 96u;
    if (threadIdx.x < 192u) sm_xq[threadIdx.x] = xq4_g[threadIdx.x];
    if (threadIdx.x <  96u) sm_xd[threadIdx.x] = xds_g[threadIdx.x].x;
    __syncthreads();

    const uint32_t exp_id = static_cast<uint32_t>(top_ids[static_cast<uint64_t>(m) * 8u + slot]);
    const uint32_t sub_id = threadIdx.x >> 4;
    const uint32_t lane16 = threadIdx.x & 15u;

    #pragma nounroll
    for (uint32_t half = 0u; half < 2u; ++half) {
        const uint32_t row  = rblk * 32u + half * 16u + sub_id;
        const uint64_t erow = static_cast<uint64_t>(exp_id) * local_ff + row;
        const uint4*    qp_p = wgu_merged + erow * 123ULL;
        const uint32_t* hs_p = reinterpret_cast<const uint32_t*>(qp_p + 96u);
        const uint32_t* d_p  = reinterpret_cast<const uint32_t*>(qp_p + 120u);

        float ag = 0.0f, au = 0.0f;
        #pragma nounroll
        for (uint32_t it = 0u; it < 6u; ++it) {
            const uint32_t ib  = it * 16u + lane16;
            // M = 2..16 verify path: cached loads hit L2 on duplicate experts across verify tokens (bit-identical).
            const uint4    q   = qp_p[ib];
            const uint32_t hs2 = hs_p[ib];
            const uint32_t d2  = d_p[ib >> 3];
            const uint4    xa  = sm_xq[ib * 2u];
            const uint4    xb  = sm_xq[ib * 2u + 1u];
            const float    xda = sm_xd[ib];

            {
                const uint32_t qh = hs2 & 0xFFu, sc = (hs2 >> 8) & 0xFFu;
                const float gd = vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(d2 & 0xFFFFu)) * 0.25f * xda;
                uint64_t g0 = sm_grid[(q.x & 0xFFu)         | ((qh << 8) & 0x300u)];
                uint64_t g1 = sm_grid[((q.x >> 16) & 0xFFu) | ((qh << 6) & 0x300u)];
                int32_t dp0 = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g0),       (q.x >> 8)  & 0xFu), xa.x, 0);
                dp0         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g0 >> 32), (q.x >> 12) & 0xFu), xa.y, dp0);
                dp0         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g1),       (q.x >> 24) & 0xFu), xa.z, dp0);
                dp0         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g1 >> 32), q.x >> 28),          xa.w, dp0);
                ag = fmaf(gd * (0.5f + static_cast<float>(sc & 0xFu)), static_cast<float>(dp0), ag);

                uint64_t g2 = sm_grid[(q.y & 0xFFu)         | ((qh << 4) & 0x300u)];
                uint64_t g3 = sm_grid[((q.y >> 16) & 0xFFu) | ((qh << 2) & 0x300u)];
                int32_t dp1 = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g2),       (q.y >> 8)  & 0xFu), xb.x, 0);
                dp1         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g2 >> 32), (q.y >> 12) & 0xFu), xb.y, dp1);
                dp1         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g3),       (q.y >> 24) & 0xFu), xb.z, dp1);
                dp1         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g3 >> 32), q.y >> 28),          xb.w, dp1);
                ag = fmaf(gd * (0.5f + static_cast<float>(sc >> 4)), static_cast<float>(dp1), ag);
            }
            {
                const uint32_t qh = (hs2 >> 16) & 0xFFu, sc = hs2 >> 24;
                const float ud = vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(d2 >> 16)) * 0.25f * xda;
                uint64_t g0 = sm_grid[(q.z & 0xFFu)         | ((qh << 8) & 0x300u)];
                uint64_t g1 = sm_grid[((q.z >> 16) & 0xFFu) | ((qh << 6) & 0x300u)];
                int32_t dp0 = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g0),       (q.z >> 8)  & 0xFu), xa.x, 0);
                dp0         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g0 >> 32), (q.z >> 12) & 0xFu), xa.y, dp0);
                dp0         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g1),       (q.z >> 24) & 0xFu), xa.z, dp0);
                dp0         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g1 >> 32), q.z >> 28),          xa.w, dp0);
                au = fmaf(ud * (0.5f + static_cast<float>(sc & 0xFu)), static_cast<float>(dp0), au);

                uint64_t g2 = sm_grid[(q.w & 0xFFu)         | ((qh << 4) & 0x300u)];
                uint64_t g3 = sm_grid[((q.w >> 16) & 0xFFu) | ((qh << 2) & 0x300u)];
                int32_t dp1 = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g2),       (q.w >> 8)  & 0xFu), xb.x, 0);
                dp1         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g2 >> 32), (q.w >> 12) & 0xFu), xb.y, dp1);
                dp1         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g3),       (q.w >> 24) & 0xFu), xb.z, dp1);
                dp1         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g3 >> 32), q.w >> 28),          xb.w, dp1);
                au = fmaf(ud * (0.5f + static_cast<float>(sc >> 4)), static_cast<float>(dp1), au);
            }
        }
        ag += __shfl_xor(ag, 8, 16); ag += __shfl_xor(ag, 4, 16); ag += __shfl_xor(ag, 2, 16); ag += __shfl_xor(ag, 1, 16);
        au += __shfl_xor(au, 8, 16); au += __shfl_xor(au, 4, 16); au += __shfl_xor(au, 2, 16); au += __shfl_xor(au, 1, 16);
        if (lane16 == 0u) sm_y[half * 16u + sub_id] = (ag / (1.0f + __expf(-ag))) * au;
    }
    __syncthreads();

    if (threadIdx.x < 32u) {
        const float y    = sm_y[threadIdx.x];
        const float amax = vegajet::ornith_dev::block32_max(fabsf(y));
        const float d    = amax * (1.0f / 127.0f);
        const float inv  = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        mid_q8[(static_cast<uint64_t>(m) * 8u + slot) * local_ff + rblk * 32u + threadIdx.x] =
            static_cast<int8_t>(max(-127, min(127, __float2int_rn(y * inv))));
        if (threadIdx.x == 0u) mid_d[(static_cast<uint64_t>(m) * 8u + slot) * nblk32 + rblk] = d;
    }
}

// ============================================================================
// 9c. LDS-Cached + 5-Inst SWAR Coalesced uint2/uint32 Split-Plane IQ3_XXS Down!
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 5)
void ornith_moe_down_iq3xxs_split_kernel(
    const uint4*   __restrict__ w_down_split,
    const int8_t*  __restrict__ mid_q8,
    const float*   __restrict__ mid_d,
    const int32_t* __restrict__ top_ids,
    const float*   __restrict__ top_w,
    const float*   __restrict__ sh_down_in,
    float*         __restrict__ out_f32,
    uint32_t M,
    uint32_t D) {
    __shared__ uint32_t sm_grid[256];
    __shared__ uint4    sm_mq[256]; // 8 * 512 B = 256 uint4s (4 KiB!)
    __shared__ float    sm_md[128]; // 8 * 16 floats
    __shared__ int32_t  sm_ids[8];
    __shared__ float    sm_w[8];

    const uint32_t m      = blockIdx.y;
    const uint32_t sub_r  = threadIdx.x >> 4;
    const uint32_t lane16 = threadIdx.x & 15u;
    const uint32_t row    = blockIdx.x * 16u + sub_r;
    if (m >= M) return;

    sm_grid[threadIdx.x] = vegajet::ggml_tables::iq3xxs_grid[threadIdx.x];
    sm_mq[threadIdx.x]   = reinterpret_cast<const uint4*>(mid_q8 + static_cast<uint64_t>(m) * 4096u)[threadIdx.x];
    if (threadIdx.x < 128u) sm_md[threadIdx.x] = mid_d[static_cast<uint64_t>(m) * 128u + threadIdx.x];
    if (threadIdx.x <   8u) {
        sm_ids[threadIdx.x] = top_ids[static_cast<uint64_t>(m) * 8u + threadIdx.x];
        sm_w[threadIdx.x]   = top_w[static_cast<uint64_t>(m) * 8u + threadIdx.x];
    }
    __syncthreads();
    if (row >= D) return;

    float total = 0.0f;
    #pragma nounroll
    for (uint32_t slot = 0u; slot < 8u; ++slot) {
        const uint32_t exp_id = static_cast<uint32_t>(sm_ids[slot]);
        const float    ew     = sm_w[slot];
        const uint4*   rbase  = w_down_split + (static_cast<uint64_t>(exp_id) * D + row) * 13ULL;
        const uint2*   q_p    = reinterpret_cast<const uint2*>(rbase);
        const uint32_t* aux_p = reinterpret_cast<const uint32_t*>(rbase + 8u);
        const uint32_t d2     = reinterpret_cast<const uint32_t*>(rbase + 12u)[0];

        const uint2    q2 = vegajet::ornith_dev::ldg_u32x2_stream(q_p + lane16);
        const uint32_t ga = vegajet::ornith_dev::ldg_u32_stream(aux_p + lane16);
        const uint16_t d16 = (lane16 < 8u) ? static_cast<uint16_t>(d2 & 0xFFFFu) : static_cast<uint16_t>(d2 >> 16);
        const float gdb = vegajet::ornith_dev::fp16_to_f32(d16) * (0.5f + static_cast<float>(ga >> 28)) * 0.5f;

        uint32_t s0 = ga & 127u;         s0 |= ((__popc(s0) & 1u) << 7);
        uint32_t s1 = (ga >> 7)  & 127u; s1 |= ((__popc(s1) & 1u) << 7);
        uint32_t s2 = (ga >> 14) & 127u; s2 |= ((__popc(s2) & 1u) << 7);
        uint32_t s3 = (ga >> 21) & 127u; s3 |= ((__popc(s3) & 1u) << 7);

        const uint4 xa = sm_mq[slot * 32u + lane16 * 2u + 0u];
        const uint4 xb = sm_mq[slot * 32u + lane16 * 2u + 1u];
        const float xd = sm_md[slot * 16u + lane16];

        int32_t dp = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(sm_grid[(q2.x >>  0) & 0xFFu], s0 & 0xFu), xa.x, 0);
        dp         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(sm_grid[(q2.x >>  8) & 0xFFu], s0 >> 4),   xa.y, dp);
        dp         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(sm_grid[(q2.x >> 16) & 0xFFu], s1 & 0xFu), xa.z, dp);
        dp         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(sm_grid[(q2.x >> 24) & 0xFFu], s1 >> 4),   xa.w, dp);
        dp         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(sm_grid[(q2.y >>  0) & 0xFFu], s2 & 0xFu), xb.x, dp);
        dp         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(sm_grid[(q2.y >>  8) & 0xFFu], s2 >> 4),   xb.y, dp);
        dp         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(sm_grid[(q2.y >> 16) & 0xFFu], s3 & 0xFu), xb.z, dp);
        dp         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(sm_grid[(q2.y >> 24) & 0xFFu], s3 >> 4),   xb.w, dp);
        total = fmaf(ew * gdb * xd, static_cast<float>(dp), total);
    }

    total += __shfl_xor(total, 8, 16);
    total += __shfl_xor(total, 4, 16);
    total += __shfl_xor(total, 2, 16);
    total += __shfl_xor(total, 1, 16);
    if (lane16 == 0u) {
        const float sh = sh_down_in ? sh_down_in[static_cast<uint64_t>(m) * D + row] : 0.0f;
        out_f32[static_cast<uint64_t>(m) * D + row] = total + sh;
    }
}

// ============================================================================
// 12. Expert-Grouped Prefill MoE Kernels (M >= 16, up to 4x weight & SWAR reuse on M=128!)
//     (a) ornith_moe_group_sort_kernel:        256-thread counting sort of M*8 pairs by expert
//     (b) ornith_moe_gu_iq2s_grouped_kernel:   amortizes IQ2_S loads + SWAR unpack across all tokens per expert!
//     (c) ornith_moe_down_iq3xxs_grouped_kernel: amortizes IQ3_XXS loads + SWAR unpack across all tokens per expert!
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_moe_group_sort_kernel(
    const int32_t* __restrict__ top_ids,   // [M * 8]
    uint32_t*      __restrict__ exp_off,   // [257]
    uint32_t*      __restrict__ exp_cnt,   // [256]
    uint32_t*      __restrict__ sorted_ms, // [M * 8]
    uint32_t M) {
    const uint32_t tid   = threadIdx.x;
    const uint32_t total = M * 8u;
    __shared__ uint32_t sm_cnt[256];
    __shared__ uint32_t sm_off[256];
    __shared__ uint32_t sm_cur[256];

    sm_cnt[tid] = 0u;
    __syncthreads();

    for (uint32_t idx = tid; idx < total; idx += 256u) {
        const uint32_t e = static_cast<uint32_t>(top_ids[idx]) & 255u;
        atomicAdd(&sm_cnt[e], 1u);
    }
    __syncthreads();

    const uint32_t c = sm_cnt[tid];
    exp_cnt[tid] = c;
    sm_off[tid]  = c;
    __syncthreads();

    // Hillis-Steele inclusive scan -> exclusive offset
    #pragma unroll
    for (uint32_t step = 1u; step < 256u; step <<= 1) {
        const uint32_t add = (tid >= step) ? sm_off[tid - step] : 0u;
        __syncthreads();
        sm_off[tid] += add;
        __syncthreads();
    }
    const uint32_t ex_off = sm_off[tid] - c;
    exp_off[tid] = ex_off;
    sm_cur[tid]  = ex_off;
    if (tid == 255u) exp_off[256] = sm_off[255];
    __syncthreads();

    for (uint32_t idx = tid; idx < total; idx += 256u) {
        const uint32_t e   = static_cast<uint32_t>(top_ids[idx]) & 255u;
        const uint32_t dst = atomicAdd(&sm_cur[e], 1u);
        sorted_ms[dst] = idx;
    }
}

extern "C" __global__ __launch_bounds__(256, 2) __attribute__((amdgpu_num_vgpr(64)))
void ornith_moe_gu_iq2s_grouped_kernel(
    const uint4*    __restrict__ wgu_merged,
    const int8_t*   __restrict__ x_q8,      // [M, 3072]
    const float2*   __restrict__ x_ds,      // [M, 96]
    const uint32_t* __restrict__ exp_off,   // [257]
    const uint32_t* __restrict__ exp_cnt,   // [256]
    const uint32_t* __restrict__ sorted_ms, // [M * 8]
    int8_t*         __restrict__ mid_q8,    // [M * 8, local_ff]
    float*          __restrict__ mid_d,     // [M * 8, local_ff / 32]
    uint32_t local_ff) {
    const uint32_t exp_id = blockIdx.y;
    const uint32_t rblk   = blockIdx.x;
    const uint32_t cnt    = exp_cnt[exp_id];
    if (cnt == 0u) return;

    __shared__ uint64_t sm_grid[1024];
    __shared__ uint32_t sm_ms[8];
    __shared__ float    sm_y[8 * 32];

    reinterpret_cast<uint4*>(sm_grid)[threadIdx.x]        = reinterpret_cast<const uint4*>(vegajet::ggml_tables::iq2s_grid)[threadIdx.x];
    reinterpret_cast<uint4*>(sm_grid)[threadIdx.x + 256u] = reinterpret_cast<const uint4*>(vegajet::ggml_tables::iq2s_grid)[threadIdx.x + 256u];
    __syncthreads();

    const uint32_t off    = exp_off[exp_id];
    const uint32_t sub_id = threadIdx.x >> 4;
    const uint32_t lane16 = threadIdx.x & 15u;
    const uint32_t nblk32 = local_ff >> 5;

    for (uint32_t t0 = 0u; t0 < cnt; t0 += 8u) {
        const uint32_t T = min(8u, cnt - t0);
        if (threadIdx.x < T) sm_ms[threadIdx.x] = sorted_ms[off + t0 + threadIdx.x];
        __syncthreads();

        #pragma nounroll
        for (uint32_t half = 0u; half < 2u; ++half) {
            const uint32_t row  = rblk * 32u + half * 16u + sub_id;
            const uint64_t erow = static_cast<uint64_t>(exp_id) * local_ff + row;
            const uint4*    qp_p = wgu_merged + erow * 123ULL;
            const uint32_t* hs_p = reinterpret_cast<const uint32_t*>(qp_p + 96u);
            const uint32_t* d_p  = reinterpret_cast<const uint32_t*>(qp_p + 120u);

            // Keep one eight-token accumulator bank live. Each projection retains the
            // original six-block FMA order and 16-lane reduction; sm_y holds gate until up.
            #pragma nounroll
            for (uint32_t projection = 0u; projection < 2u; ++projection) {
                float acc[8] = {0,0,0,0,0,0,0,0};

                #pragma nounroll
                for (uint32_t it = 0u; it < 6u; ++it) {
                    const uint32_t ib  = it * 16u + lane16;
                    const uint4    q   = vegajet::ornith_dev::ldg_u32x4_stream(qp_p + ib);
                    const uint32_t hs2 = vegajet::ornith_dev::ldg_u32_stream(hs_p + ib);
                    const uint32_t d2  = d_p[ib >> 3];
                    const uint32_t q0  = (projection == 0u) ? q.x : q.z;
                    const uint32_t q1  = (projection == 0u) ? q.y : q.w;
                    const uint32_t hs  = hs2 >> (projection * 16u);
                    const uint32_t qh  = hs & 0xFFu, sc = (hs >> 8) & 0xFFu;
                    const float d     = vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(d2 >> (projection * 16u))) * 0.25f;
                    const float d0    = d * (0.5f + static_cast<float>(sc & 0xFu));
                    const float d1    = d * (0.5f + static_cast<float>(sc >> 4));
                    const uint64_t g0 = sm_grid[(q0 & 0xFFu)         | ((qh << 8) & 0x300u)];
                    const uint64_t g1 = sm_grid[((q0 >> 16) & 0xFFu) | ((qh << 6) & 0x300u)];
                    const uint64_t g2 = sm_grid[(q1 & 0xFFu)         | ((qh << 4) & 0x300u)];
                    const uint64_t g3 = sm_grid[((q1 >> 16) & 0xFFu) | ((qh << 2) & 0x300u)];
                    const uint32_t w0 = vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g0),       (q0 >> 8)  & 0xFu);
                    const uint32_t w1 = vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g0 >> 32), (q0 >> 12) & 0xFu);
                    const uint32_t w2 = vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g1),       (q0 >> 24) & 0xFu);
                    const uint32_t w3 = vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g1 >> 32), q0 >> 28);
                    const uint32_t w4 = vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g2),       (q1 >> 8)  & 0xFu);
                    const uint32_t w5 = vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g2 >> 32), (q1 >> 12) & 0xFu);
                    const uint32_t w6 = vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g3),       (q1 >> 24) & 0xFu);
                    const uint32_t w7 = vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g3 >> 32), q1 >> 28);

                    // Unpack once per projection/block and reuse across all T tokens.
                    for (uint32_t t = 0u; t < T; ++t) {
                        const uint32_t m   = sm_ms[t] >> 3;
                        const uint4*   xq4 = reinterpret_cast<const uint4*>(x_q8 + static_cast<uint64_t>(m) * 3072u);
                        const uint4    xa  = xq4[ib * 2u];
                        const uint4    xb  = xq4[ib * 2u + 1u];
                        const float    xda = x_ds[static_cast<uint64_t>(m) * 96u + ib].x;
                        int32_t dp0 = vegajet::ornith_dev::sdot4(w3, xa.w, vegajet::ornith_dev::sdot4(w2, xa.z, vegajet::ornith_dev::sdot4(w1, xa.y, vegajet::ornith_dev::sdot4(w0, xa.x, 0))));
                        int32_t dp1 = vegajet::ornith_dev::sdot4(w7, xb.w, vegajet::ornith_dev::sdot4(w6, xb.z, vegajet::ornith_dev::sdot4(w5, xb.y, vegajet::ornith_dev::sdot4(w4, xb.x, 0))));
                        acc[t] = fmaf(d0 * xda, static_cast<float>(dp0), fmaf(d1 * xda, static_cast<float>(dp1), acc[t]));
                    }
                }

                for (uint32_t t = 0u; t < T; ++t) {
                    float v = acc[t];
                    v += __shfl_xor(v, 8, 16); v += __shfl_xor(v, 4, 16); v += __shfl_xor(v, 2, 16); v += __shfl_xor(v, 1, 16);
                    if (lane16 == 0u) {
                        const uint32_t idx = t * 32u + half * 16u + sub_id;
                        if (projection == 0u) {
                            sm_y[idx] = v;
                        } else {
                            const float gv = sm_y[idx]; // same lane stored the gate result
                            sm_y[idx] = (gv / (1.0f + __expf(-gv))) * v;
                        }
                    }
                }
            }
        }
        __syncthreads();

        // All 256 threads quantize up to 8 tokens (32 elems each) in parallel!
        const uint32_t tq  = threadIdx.x >> 5;
        const uint32_t sub = threadIdx.x & 31u;
        if (tq < T) {
            const uint32_t ms   = sm_ms[tq];
            const float    y    = sm_y[tq * 32u + sub];
            const float    amax = vegajet::ornith_dev::block32_max(fabsf(y));
            const float    d    = amax * (1.0f / 127.0f);
            const float    inv  = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            mid_q8[static_cast<uint64_t>(ms) * local_ff + rblk * 32u + sub] =
                static_cast<int8_t>(max(-127, min(127, __float2int_rn(y * inv))));
            if (sub == 0u) mid_d[static_cast<uint64_t>(ms) * nblk32 + rblk] = d;
        }
        __syncthreads();
    }
}

extern "C" __global__ __launch_bounds__(256, 4)
void ornith_moe_down_iq3xxs_grouped_kernel(
    const uint4*    __restrict__ w_down_split,
    const int8_t*   __restrict__ mid_q8,    // [M * 8, 512]
    const float*    __restrict__ mid_d,     // [M * 8, 16]
    const float*    __restrict__ top_w,     // [M * 8]
    const uint32_t* __restrict__ exp_off,   // [257]
    const uint32_t* __restrict__ exp_cnt,   // [256]
    const uint32_t* __restrict__ sorted_ms, // [M * 8]
    float*          __restrict__ out_f32,   // [M, D] (pre-initialized with sh_down_in!)
    uint32_t D) {
    const uint32_t exp_id = blockIdx.y;
    const uint32_t cnt    = exp_cnt[exp_id];
    if (cnt == 0u) return;

    __shared__ uint32_t sm_grid[256];
    sm_grid[threadIdx.x] = vegajet::ggml_tables::iq3xxs_grid[threadIdx.x];
    __syncthreads();

    const uint32_t sub_r  = threadIdx.x >> 4;
    const uint32_t lane16 = threadIdx.x & 15u;
    const uint32_t row    = blockIdx.x * 16u + sub_r;
    if (row >= D) return;

    // Load & unpack IQ3_XXS weights ONCE for this (expert, row, lane16) and reuse across ALL cnt tokens!
    const uint4*    rbase = w_down_split + (static_cast<uint64_t>(exp_id) * D + row) * 13ULL;
    const uint2*    q_p   = reinterpret_cast<const uint2*>(rbase);
    const uint32_t* aux_p = reinterpret_cast<const uint32_t*>(rbase + 8u);
    const uint32_t  d2    = reinterpret_cast<const uint32_t*>(rbase + 12u)[0];

    const uint2    q2  = vegajet::ornith_dev::ldg_u32x2_stream(q_p + lane16);
    const uint32_t ga  = vegajet::ornith_dev::ldg_u32_stream(aux_p + lane16);
    const uint16_t d16 = (lane16 < 8u) ? static_cast<uint16_t>(d2 & 0xFFFFu) : static_cast<uint16_t>(d2 >> 16);
    const float    gdb = vegajet::ornith_dev::fp16_to_f32(d16) * (0.5f + static_cast<float>(ga >> 28)) * 0.5f;

    uint32_t s0 = ga & 127u;         s0 |= ((__popc(s0) & 1u) << 7);
    uint32_t s1 = (ga >> 7)  & 127u; s1 |= ((__popc(s1) & 1u) << 7);
    uint32_t s2 = (ga >> 14) & 127u; s2 |= ((__popc(s2) & 1u) << 7);
    uint32_t s3 = (ga >> 21) & 127u; s3 |= ((__popc(s3) & 1u) << 7);

    const uint32_t w0 = vegajet::ornith_dev::apply_4signs_i8x4(sm_grid[(q2.x >>  0) & 0xFFu], s0 & 0xFu);
    const uint32_t w1 = vegajet::ornith_dev::apply_4signs_i8x4(sm_grid[(q2.x >>  8) & 0xFFu], s0 >> 4);
    const uint32_t w2 = vegajet::ornith_dev::apply_4signs_i8x4(sm_grid[(q2.x >> 16) & 0xFFu], s1 & 0xFu);
    const uint32_t w3 = vegajet::ornith_dev::apply_4signs_i8x4(sm_grid[(q2.x >> 24) & 0xFFu], s1 >> 4);
    const uint32_t w4 = vegajet::ornith_dev::apply_4signs_i8x4(sm_grid[(q2.y >>  0) & 0xFFu], s2 & 0xFu);
    const uint32_t w5 = vegajet::ornith_dev::apply_4signs_i8x4(sm_grid[(q2.y >>  8) & 0xFFu], s2 >> 4);
    const uint32_t w6 = vegajet::ornith_dev::apply_4signs_i8x4(sm_grid[(q2.y >> 16) & 0xFFu], s3 & 0xFu);
    const uint32_t w7 = vegajet::ornith_dev::apply_4signs_i8x4(sm_grid[(q2.y >> 24) & 0xFFu], s3 >> 4);

    const uint32_t off = exp_off[exp_id];
    #pragma nounroll
    for (uint32_t t = 0u; t < cnt; ++t) {
        const uint32_t ms  = sorted_ms[off + t];
        const uint32_t m   = ms >> 3;
        const float    ew  = top_w[ms];
        const uint4*   xq4 = reinterpret_cast<const uint4*>(mid_q8 + static_cast<uint64_t>(ms) * 512u);
        const uint4    xa  = xq4[lane16 * 2u + 0u];
        const uint4    xb  = xq4[lane16 * 2u + 1u];
        const float    xd  = mid_d[static_cast<uint64_t>(ms) * 16u + lane16];

        int32_t dp = vegajet::ornith_dev::sdot4(w0, xa.x, 0);
        dp         = vegajet::ornith_dev::sdot4(w1, xa.y, dp);
        dp         = vegajet::ornith_dev::sdot4(w2, xa.z, dp);
        dp         = vegajet::ornith_dev::sdot4(w3, xa.w, dp);
        dp         = vegajet::ornith_dev::sdot4(w4, xb.x, dp);
        dp         = vegajet::ornith_dev::sdot4(w5, xb.y, dp);
        dp         = vegajet::ornith_dev::sdot4(w6, xb.z, dp);
        dp         = vegajet::ornith_dev::sdot4(w7, xb.w, dp);
        float v = ew * gdb * xd * static_cast<float>(dp);
        v += __shfl_xor(v, 8, 16);
        v += __shfl_xor(v, 4, 16);
        v += __shfl_xor(v, 2, 16);
        v += __shfl_xor(v, 1, 16);
        if (lane16 == 0u) atomicAdd(&out_f32[static_cast<uint64_t>(m) * D + row], v);
    }
}

extern "C" __global__ __launch_bounds__(256, 4)
void ornith_moe_gu_q4a8_kernel(
    const uint8_t*  __restrict__ wg_qs,    // [256 * local_ff, stride_b]
    const uint32_t* __restrict__ wg_dm,    // [256 * local_ff, 96]
    const uint8_t*  __restrict__ wu_qs,    // [256 * local_ff, stride_b]
    const uint32_t* __restrict__ wu_dm,    // [256 * local_ff, 96]
    const int8_t*   __restrict__ x_q8,     // [M, 3072]
    const float2*   __restrict__ x_ds,     // [M, 96]
    const int32_t*  __restrict__ top_ids,  // [M, 8]
    int8_t*         __restrict__ mid_q8,   // [M, 8, local_ff]
    float2*         __restrict__ mid_ds,   // [M, 8, local_ff/32]
    uint32_t M,
    uint32_t local_ff,
    uint32_t stride_b) {
    __shared__ uint4  sm_xq[192];
    __shared__ float2 sm_xds[96];
    __shared__ float  sm_y[32];

    const uint32_t m      = blockIdx.y;
    const uint32_t nblk32 = local_ff >> 5;
    const uint32_t slot   = blockIdx.x / nblk32;
    const uint32_t rblk   = blockIdx.x - slot * nblk32;
    if (m >= M || slot >= 8u) return;

    const uint4*  xq4_g = reinterpret_cast<const uint4*>(x_q8 + static_cast<uint64_t>(m) * 3072u);
    const float2* xds_g = x_ds + static_cast<uint64_t>(m) * 96u;
    if (threadIdx.x < 192u) sm_xq[threadIdx.x]  = xq4_g[threadIdx.x];
    if (threadIdx.x <  96u) sm_xds[threadIdx.x] = xds_g[threadIdx.x];
    __syncthreads();

    const uint32_t exp_id = static_cast<uint32_t>(top_ids[static_cast<uint64_t>(m) * 8u + slot]);
    const uint32_t grp    = threadIdx.x >> 5; // 0..7 (8 rows per pass, 4 passes = 32 rows!)
    const uint32_t lane   = threadIdx.x & 31u;

    #pragma nounroll
    for (uint32_t pass = 0u; pass < 4u; ++pass) {
        const uint32_t row  = rblk * 32u + pass * 8u + grp;
        const uint64_t erow = static_cast<uint64_t>(exp_id) * local_ff + row;
        const uint4*    gw4 = reinterpret_cast<const uint4*>(wg_qs + erow * stride_b);
        const uint32_t* gdm = wg_dm + erow * 96ULL;
        const uint4*    uw4 = reinterpret_cast<const uint4*>(wu_qs + erow * stride_b);
        const uint32_t* udm = wu_dm + erow * 96ULL;

        float ag = 0.0f, au = 0.0f;
        #pragma nounroll
        for (uint32_t it = 0u; it < 3u; ++it) {
            const uint32_t b   = it * 32u + lane;
            const uint4    xa  = sm_xq[b * 2u];
            const uint4    xb  = sm_xq[b * 2u + 1u];
            const float2   xds = sm_xds[b];

            const uint4    gv  = vegajet::ornith_dev::ldg_u32x4_stream(gw4 + b);
            const uint32_t gdv = vegajet::ornith_dev::ldg_u32_stream(gdm + b);
            int32_t dpg = vegajet::ornith_dev::sdot4(gv.x & 0x0F0F0F0Fu, xa.x, 0);
            dpg         = vegajet::ornith_dev::sdot4(gv.y & 0x0F0F0F0Fu, xa.y, dpg);
            dpg         = vegajet::ornith_dev::sdot4(gv.z & 0x0F0F0F0Fu, xa.z, dpg);
            dpg         = vegajet::ornith_dev::sdot4(gv.w & 0x0F0F0F0Fu, xa.w, dpg);
            dpg         = vegajet::ornith_dev::sdot4((gv.x >> 4) & 0x0F0F0F0Fu, xb.x, dpg);
            dpg         = vegajet::ornith_dev::sdot4((gv.y >> 4) & 0x0F0F0F0Fu, xb.y, dpg);
            dpg         = vegajet::ornith_dev::sdot4((gv.z >> 4) & 0x0F0F0F0Fu, xb.z, dpg);
            dpg         = vegajet::ornith_dev::sdot4((gv.w >> 4) & 0x0F0F0F0Fu, xb.w, dpg);
            ag = fmaf(vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(gdv & 0xFFFFu)) * xds.x, static_cast<float>(dpg),
                      fmaf(vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(gdv >> 16)), xds.y, ag));

            const uint4    uv  = vegajet::ornith_dev::ldg_u32x4_stream(uw4 + b);
            const uint32_t udv = vegajet::ornith_dev::ldg_u32_stream(udm + b);
            int32_t dpu = vegajet::ornith_dev::sdot4(uv.x & 0x0F0F0F0Fu, xa.x, 0);
            dpu         = vegajet::ornith_dev::sdot4(uv.y & 0x0F0F0F0Fu, xa.y, dpu);
            dpu         = vegajet::ornith_dev::sdot4(uv.z & 0x0F0F0F0Fu, xa.z, dpu);
            dpu         = vegajet::ornith_dev::sdot4(uv.w & 0x0F0F0F0Fu, xa.w, dpu);
            dpu         = vegajet::ornith_dev::sdot4((uv.x >> 4) & 0x0F0F0F0Fu, xb.x, dpu);
            dpu         = vegajet::ornith_dev::sdot4((uv.y >> 4) & 0x0F0F0F0Fu, xb.y, dpu);
            dpu         = vegajet::ornith_dev::sdot4((uv.z >> 4) & 0x0F0F0F0Fu, xb.z, dpu);
            dpu         = vegajet::ornith_dev::sdot4((uv.w >> 4) & 0x0F0F0F0Fu, xb.w, dpu);
            au = fmaf(vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(udv & 0xFFFFu)) * xds.x, static_cast<float>(dpu),
                      fmaf(vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(udv >> 16)), xds.y, au));
        }
        ag = vegajet::ornith_dev::block32_sum(ag);
        au = vegajet::ornith_dev::block32_sum(au);
        if (lane == 0u) sm_y[pass * 8u + grp] = (ag / (1.0f + __expf(-ag))) * au;
    }
    __syncthreads();

    if (threadIdx.x < 32u) {
        const float y    = sm_y[threadIdx.x];
        const float amax = vegajet::ornith_dev::block32_max(fabsf(y));
        const float bsum = vegajet::ornith_dev::block32_sum(y);
        const float d    = amax * (1.0f / 127.0f);
        const float inv  = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        mid_q8[(static_cast<uint64_t>(m) * 8u + slot) * local_ff + rblk * 32u + threadIdx.x] =
            static_cast<int8_t>(max(-127, min(127, __float2int_rn(y * inv))));
        if (threadIdx.x == 0u) mid_ds[(static_cast<uint64_t>(m) * 8u + slot) * nblk32 + rblk] = make_float2(d, bsum);
    }
}

extern "C" __global__ __launch_bounds__(256, 5)
void ornith_moe_down_q4a8_kernel(
    const uint8_t*  __restrict__ wd_qs,    // [256 * D, local_ff / 2] (e.g. 512->256B, 384->192B, 320->160B)
    const uint32_t* __restrict__ wd_dm,    // [256 * D, local_ff / 32]
    const int8_t*   __restrict__ mid_q8,   // [M, 8, local_ff]
    const float2*   __restrict__ mid_ds,   // [M, 8, local_ff / 32]
    const int32_t*  __restrict__ top_ids,  // [M, 8]
    const float*    __restrict__ top_w,    // [M, 8]
    const float*    __restrict__ sh_down_in,
    float*          __restrict__ out_f32,
    uint32_t M,
    uint32_t D,
    uint32_t stride_b,
    uint32_t local_ff) {
    __shared__ uint4   sm_mq[256];
    __shared__ float2  sm_mds[128];
    __shared__ int32_t sm_ids[8];
    __shared__ float   sm_w[8];

    const uint32_t m      = blockIdx.y;
    const uint32_t sub_r  = threadIdx.x >> 4; // 0..15 (16 rows/WG)
    const uint32_t lane16 = threadIdx.x & 15u;
    const uint32_t row    = blockIdx.x * 16u + sub_r;
    if (m >= M) return;

    const uint32_t n_blk   = local_ff >> 5;           // <= 16 (16 on TP=2, 12/10/10 on TP=3)
    const uint32_t n_q8_u4 = 8u * (local_ff >> 4);    // <= 256
    const uint32_t n_ds    = 8u * n_blk;              // <= 128

    if (threadIdx.x < n_q8_u4)
        sm_mq[threadIdx.x] = reinterpret_cast<const uint4*>(mid_q8 + static_cast<uint64_t>(m) * (8u * local_ff))[threadIdx.x];
    if (threadIdx.x < n_ds)
        sm_mds[threadIdx.x] = mid_ds[static_cast<uint64_t>(m) * n_ds + threadIdx.x];
    if (threadIdx.x < 8u) {
        sm_ids[threadIdx.x] = top_ids[static_cast<uint64_t>(m) * 8u + threadIdx.x];
        sm_w[threadIdx.x]   = top_w[static_cast<uint64_t>(m) * 8u + threadIdx.x];
    }
    __syncthreads();
    if (row >= D) return;

    float total = 0.0f;
    if (lane16 < n_blk) {
        const uint32_t n_blk2 = n_blk << 1;
        #pragma nounroll
        for (uint32_t slot = 0u; slot < 8u; ++slot) {
            const uint32_t exp_id = static_cast<uint32_t>(sm_ids[slot]);
            const float    ew     = sm_w[slot];
            const uint64_t erow   = static_cast<uint64_t>(exp_id) * D + row;
            const uint4*    w4    = reinterpret_cast<const uint4*>(wd_qs + erow * stride_b);
            const uint32_t* dm    = wd_dm + erow * n_blk;

            const uint4    wv  = vegajet::ornith_dev::ldg_u32x4_stream(w4 + lane16);
            const uint32_t dmv = vegajet::ornith_dev::ldg_u32_stream(dm + lane16);
            const uint4    xa  = sm_mq[slot * n_blk2 + lane16 * 2u + 0u];
            const uint4    xb  = sm_mq[slot * n_blk2 + lane16 * 2u + 1u];
            const float2   xds = sm_mds[slot * n_blk + lane16];

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
            total = fmaf(ew, fmaf(wd * xds.x, static_cast<float>(dp), wm * xds.y), total);
        }
    }

    total += __shfl_xor(total, 8, 16);
    total += __shfl_xor(total, 4, 16);
    total += __shfl_xor(total, 2, 16);
    total += __shfl_xor(total, 1, 16);
    if (lane16 == 0u) {
        const float sh = sh_down_in ? sh_down_in[static_cast<uint64_t>(m) * D + row] : 0.0f;
        out_f32[static_cast<uint64_t>(m) * D + row] = total + sh;
    }
}


// ============================================================================
// 8z. AcceptMoE (VEGAJET_SPEC_ACCEPTMOE=1, opt-in, LOSSY): speculative-verify (M >= 2) top-8 routing that
//     shrinks the per-batch expert union. Token 0 (the committed context token) is routed exactly as
//     ornith_moe_router_top8_kernel. Draft token m >= 1 keeps its own top-`keep` experts, then fills the
//     remaining 8 - keep slots ranking commitment-weighted union and cold-expert scores.
//     Position m uses acceptance^m; nonunion multiplier = 1 - acceptance^m * (1-penalty).
//     Recency is frozen for the cycle and updated after all rows; window=0 disables cold bias.
//     Weights are the true softmax
//     probabilities of the chosen experts renormalized like the baseline. Draft-position outputs change, so
//     greedy acceptance is no longer guaranteed identical to raw decode: measure KL / parity before use.
//     Grid = 64 threads (one wave), tokens processed in order.
// ============================================================================
extern "C" __global__ __launch_bounds__(64, 1)
void vj_moe_top8_acceptmoe_kernel(
    const float* __restrict__ logits_in,
    int32_t*     __restrict__ top_ids,
    float*       __restrict__ top_w,
    uint32_t M,
    uint32_t keep,
    float    penalty,
    float    acceptance,
    float    cold_penalty,
    uint32_t window,
    uint32_t tick,
    uint32_t* __restrict__ recency) {
    using namespace vegajet::ornith_dev;
    const uint32_t l = threadIdx.x;
    __shared__ uint32_t sm_union[8];
    if (l < 8u) sm_union[l] = 0u;
    __syncthreads();
    float commitment = 1.0f;
    #pragma nounroll
    for (uint32_t m = 0u; m < M; ++m) {
        if (m != 0u) commitment *= acceptance;
        const float* lg = logits_in + static_cast<uint64_t>(m) * 256u;
        float x0 = lg[l], x1 = lg[l + 64u], x2 = lg[l + 128u], x3 = lg[l + 192u];
        float mx = fmaxf(fmaxf(x0, x1), fmaxf(x2, x3));
        mx = fmaxf(mx, xor_shfl_f32_fast<32>(mx));
        mx = fmaxf(mx, xor_shfl_f32_fast<16>(mx));
        mx = fmaxf(mx, xor_shfl_f32_fast<8>(mx));
        mx = fmaxf(mx, xor_shfl_f32_fast<4>(mx));
        mx = fmaxf(mx, xor_shfl_f32_fast<2>(mx));
        mx = fmaxf(mx, xor_shfl_f32_fast<1>(mx));
        float p[4] = {__expf(x0 - mx), __expf(x1 - mx), __expf(x2 - mx), __expf(x3 - mx)};
        const float s = ((wave64_sum(p[0]) + wave64_sum(p[1])) + wave64_sum(p[2])) + wave64_sum(p[3]);
        const float inv_s = 1.0f / fmaxf(s, 1e-30f);
        uint64_t kp[4], ka[4];
        const uint32_t keep_m = (m == 0u) ? 8u : min(keep, 8u);
        #pragma unroll
        for (uint32_t c = 0u; c < 4u; ++c) {
            p[c] *= inv_s;
            const uint32_t e = l + 64u * c;
            const bool in_u = ((sm_union[e >> 5] >> (e & 31u)) & 1u) != 0u;
            const float pa = p[c] * vegajet::acceptmoe::multiplier(in_u,
                m != 0u && vegajet::acceptmoe::cold(recency[e], tick, window),
                commitment, penalty, cold_penalty);
            kp[c] = (static_cast<uint64_t>(__float_as_uint(p[c])) << 32) | static_cast<uint64_t>(255u - e);
            ka[c] = (static_cast<uint64_t>(__float_as_uint(pa)) << 32) | static_cast<uint64_t>(255u - e);
        }
        float top_sum = 0.0f;
        #pragma nounroll
        for (uint32_t r = 0u; r < 8u; ++r) {
            const bool own = (r < keep_m);
            uint64_t a0 = own ? kp[0] : ka[0], a1 = own ? kp[1] : ka[1], a2 = own ? kp[2] : ka[2], a3 = own ? kp[3] : ka[3];
            uint64_t bk = (a0 > a1) ? a0 : a1;
            const uint64_t b2 = (a2 > a3) ? a2 : a3;
            if (b2 > bk) bk = b2;
            bk = step_max_u64<32>(bk);
            bk = step_max_u64<16>(bk);
            bk = step_max_u64<8>(bk);
            bk = step_max_u64<4>(bk);
            bk = step_max_u64<2>(bk);
            bk = step_max_u64<1>(bk);
            const uint32_t e_sel = 255u - static_cast<uint32_t>(bk & 255u);
            // true probability of the chosen expert lives in lane e_sel & 63, component e_sel >> 6
            const uint32_t c_sel = e_sel >> 6;
            const float pc = (c_sel == 0u) ? p[0] : (c_sel == 1u) ? p[1] : (c_sel == 2u) ? p[2] : p[3];
            const float v = __shfl(pc, static_cast<int>(e_sel & 63u), 64);
            #pragma unroll
            for (uint32_t c = 0u; c < 4u; ++c) {
                if (l + 64u * c == e_sel) { kp[c] = 0ULL; ka[c] = 0ULL; }
            }
            if (l == 0u) {
                top_ids[static_cast<uint64_t>(m) * 8u + r] = static_cast<int32_t>(e_sel);
                top_w[static_cast<uint64_t>(m) * 8u + r]   = v;
                top_sum += v;
                sm_union[e_sel >> 5] |= (1u << (e_sel & 31u));
            }
        }
        if (l == 0u) {
            const float inv_w = 1.0f / fmaxf(top_sum, 1e-30f);
            #pragma unroll
            for (uint32_t r = 0u; r < 8u; ++r) top_w[static_cast<uint64_t>(m) * 8u + r] *= inv_w;
        }
        __syncthreads();
    }
    #pragma unroll
    for (uint32_t c = 0u; c < 4u; ++c) {
        const uint32_t e = l + 64u * c;
        if ((sm_union[e >> 5] >> (e & 31u)) & 1u) recency[e] = tick;
    }
}
