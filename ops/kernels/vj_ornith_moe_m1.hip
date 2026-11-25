// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "ops/kernels/vj_ornith_common.hiph"

// ============================================================================
// 9d. M=1 MoE back half as ONE persistent work-stealing dispatch (3 -> 1):
//     stage A tiles: routed IQ2_S Gate+Up+SiLU*up+quant (8 slots x local_ff/32) and shared-expert Down GEMV
//     (16 rows/tile); grid barrier; stage B tiles: routed Down (DOT8 repack of IQ3_XXS) + shared Down (16 rows/tile).
//     WGs pull tiles from atomic counters, so the grid size (<= 4 WGs/CU resident) is independent of the
//     tile count. Per-tile math is copied from kernels 9b / 4b / 9c and is bit-identical to them.
//     q[0]=arrivals, q[16]=sense, q[32]=stage-A tile counter, q[48]=stage-B tile counter (host zeroes once).
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_moe_m1_back_kernel(
    const uint4*    __restrict__ wgu_merged,
    const int8_t*   __restrict__ x_q8,
    const float2*   __restrict__ x_ds,
    const int32_t*  __restrict__ top_ids,
    const float*    __restrict__ top_w,
    int8_t*         __restrict__ mid_q8,
    float*          __restrict__ mid_d,
    const uint8_t*  __restrict__ sh_wqs,
    const uint32_t* __restrict__ sh_wdm,
    const int8_t*   __restrict__ sh_q8,
    const float2*   __restrict__ sh_ds,
    float*          __restrict__ sh_down,
    const uint8_t*  __restrict__ wdn_d8,     // routed Down, exact 16-level DOT8 codes (256 B/row)
    const uint8_t*  __restrict__ wdn_d8_sc,  // routed Down scale plane (16 B/row)
    float*          __restrict__ out_f32,
    uint32_t*       __restrict__ q,
    uint32_t local_ff,
    uint32_t D,
    uint32_t sh_rows,
    uint32_t sh_stride,
    uint32_t sense,
    uint32_t dn_w_stride,
    uint32_t dn_s_stride,
    uint16_t*       __restrict__ peer_tx) {
    __shared__ uint64_t sm_grid[1024];
    __shared__ uint4    sm_x[256];   // stage A: x_q8 (192 used); stage B: mid_q8 (8 x 512 B)
    __shared__ float    sm_xd[128];  // stage A: x block scales (96 used); stage B: mid_d (8 x 16)
    __shared__ float    sm_y[32];
    __shared__ int32_t  sm_ids[8];
    __shared__ float    sm_w[8];
    __shared__ uint32_t sm_tile;

    const uint32_t wg     = blockIdx.x;
    const uint32_t nblk32 = local_ff >> 5;
    const uint32_t n_gu   = 8u * nblk32;
    const uint32_t n_sh   = (sh_rows + 15u) / 16u;
    const uint32_t n_a    = n_gu + n_sh;
    const uint32_t n_b    = (D + 15u) / 16u;
    const uint32_t lane16 = threadIdx.x & 15u;
    const bool static_224 = (__builtin_amdgcn_readfirstlane(gridDim.x) == 224u && n_gu == 128u && n_sh == 192u && n_b == 192u);

    if (threadIdx.x < 192u) sm_x[threadIdx.x]  = reinterpret_cast<const uint4*>(x_q8)[threadIdx.x];
    if (threadIdx.x <  96u) sm_xd[threadIdx.x] = x_ds[threadIdx.x].x;
    if (threadIdx.x <   8u) {
        sm_ids[threadIdx.x] = top_ids[threadIdx.x];
        sm_w[threadIdx.x]   = top_w[threadIdx.x];
    }
    __syncthreads();

    // ---- stage A ----
    bool grid_loaded = false;
    uint32_t static_step = 0u;
    #pragma nounroll
    for (;;) {
        uint32_t t;
        if (static_224) {
            if (wg < 128u) {
                if (static_step > 0u) break;
                t = wg;
            } else {
                if (static_step >= 2u) break;
                t = 128u + (wg - 128u) * 2u + static_step;
            }
            ++static_step;
        } else {
            if (threadIdx.x == 0u) sm_tile = atomicAdd(&q[32], 1u);
            __syncthreads();
            t = __builtin_amdgcn_readfirstlane(sm_tile);
            if (t >= n_a) break;
        }
        t = __builtin_amdgcn_readfirstlane(t);
        if (t < n_gu) {
            if (!grid_loaded) {
                reinterpret_cast<uint4*>(sm_grid)[threadIdx.x]        = reinterpret_cast<const uint4*>(vegajet::ggml_tables::iq2s_grid)[threadIdx.x];
                reinterpret_cast<uint4*>(sm_grid)[threadIdx.x + 256u] = reinterpret_cast<const uint4*>(vegajet::ggml_tables::iq2s_grid)[threadIdx.x + 256u];
                __syncthreads();
                grid_loaded = true;
            }
            const uint32_t slot   = __builtin_amdgcn_readfirstlane(t / nblk32);
            const uint32_t rblk   = __builtin_amdgcn_readfirstlane(t - slot * nblk32);
            const uint32_t exp_id = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(sm_ids[slot]));
            const uint32_t sub_id = threadIdx.x >> 4;

            #pragma nounroll
            for (uint32_t half = 0u; half < 2u; ++half) {
                const uint32_t row  = rblk * 32u + half * 16u + sub_id;
                const uint64_t erow = static_cast<uint64_t>(exp_id) * local_ff + row;
                const uint4*   qp_p = wgu_merged + erow * 123ULL;

                float ag = 0.0f, au = 0.0f;
                #pragma nounroll
                for (uint32_t it = 0u; it < 6u; ++it) {
                    const uint32_t ib  = it * 16u + lane16;
                    const uint4    wq  = vegajet::ornith_dev::ldg_u32x4_stream(qp_p + ib);
                    const uint32_t hs2 = vegajet::ornith_dev::ldg_u32_stream(reinterpret_cast<const uint32_t*>(qp_p + 96u) + ib);
                    const uint32_t d2  = reinterpret_cast<const uint32_t*>(qp_p + 120u)[ib >> 3];
                    const uint4    xa  = sm_x[ib * 2u];
                    const uint4    xb  = sm_x[ib * 2u + 1u];
                    const float    xda = sm_xd[ib];
                    __builtin_amdgcn_sched_barrier(0);

                    #pragma nounroll
                    for (uint32_t part = 0u; part < 2u; ++part) {
                        const uint32_t qa  = part ? wq.z : wq.x;
                        const uint32_t qb  = part ? wq.w : wq.y;
                        const uint32_t hsp = part ? (hs2 >> 16) : (hs2 & 0xFFFFu);
                        const uint32_t qh  = hsp & 0xFFu, sc = hsp >> 8;
                        const float pd = vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(part ? (d2 >> 16) : (d2 & 0xFFFFu))) * 0.25f * xda;
                        float acc = part ? au : ag;
                        uint64_t g0 = sm_grid[(qa & 0xFFu)         | ((qh << 8) & 0x300u)];
                        uint64_t g1 = sm_grid[((qa >> 16) & 0xFFu) | ((qh << 6) & 0x300u)];
                        int32_t dp0 = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g0),       (qa >> 8)  & 0xFu), xa.x, 0);
                        dp0         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g0 >> 32), (qa >> 12) & 0xFu), xa.y, dp0);
                        dp0         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g1),       (qa >> 24) & 0xFu), xa.z, dp0);
                        dp0         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g1 >> 32), qa >> 28),          xa.w, dp0);
                        acc = fmaf(pd * (0.5f + static_cast<float>(sc & 0xFu)), static_cast<float>(dp0), acc);
                        __builtin_amdgcn_sched_barrier(0);

                        uint64_t g2 = sm_grid[(qb & 0xFFu)         | ((qh << 4) & 0x300u)];
                        uint64_t g3 = sm_grid[((qb >> 16) & 0xFFu) | ((qh << 2) & 0x300u)];
                        int32_t dp1 = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g2),       (qb >> 8)  & 0xFu), xb.x, 0);
                        dp1         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g2 >> 32), (qb >> 12) & 0xFu), xb.y, dp1);
                        dp1         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g3),       (qb >> 24) & 0xFu), xb.z, dp1);
                        dp1         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g3 >> 32), qb >> 28),          xb.w, dp1);
                        acc = fmaf(pd * (0.5f + static_cast<float>(sc >> 4)), static_cast<float>(dp1), acc);
                        if (part) au = acc; else ag = acc;
                        __builtin_amdgcn_sched_barrier(0);
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
                mid_q8[(static_cast<uint64_t>(slot)) * local_ff + rblk * 32u + threadIdx.x] =
                    static_cast<int8_t>(max(-127, min(127, __float2int_rn(y * inv))));
                if (threadIdx.x == 0u) mid_d[(static_cast<uint64_t>(slot)) * nblk32 + rblk] = d;
            }
        } else {
            const uint32_t row = (t - n_gu) * 16u + (threadIdx.x >> 4);
            if (row < sh_rows) {
                const uint32_t  nb = local_ff >> 5;
                const uint4*    w4 = reinterpret_cast<const uint4*>(sh_wqs + static_cast<uint64_t>(row) * sh_stride);
                const uint32_t* dm = sh_wdm + static_cast<uint64_t>(row) * nb;
                const uint4*    x4 = reinterpret_cast<const uint4*>(sh_q8);
                float acc = 0.0f;
                #pragma nounroll
                for (uint32_t b = lane16; b < nb; b += 16u) {
                    const uint4    wv  = vegajet::ornith_dev::ldg_u32x4_stream(w4 + b);
                    const uint32_t dmv = vegajet::ornith_dev::ldg_u32_stream(dm + b);
                    const uint4    xa  = x4[b * 2u];
                    const uint4    xb  = x4[b * 2u + 1u];
                    const float2   xds = sh_ds[b];
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
                acc += __shfl_xor(acc, 8, 16);
                acc += __shfl_xor(acc, 4, 16);
                acc += __shfl_xor(acc, 2, 16);
                acc += __shfl_xor(acc, 1, 16);
                if (lane16 == 0u) sh_down[row] = acc;
            }
        }
        __syncthreads();
    }

    // ---- device-local grid barrier: the last arriver rearms counters before releasing ----
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0u) {
        __threadfence();
        const uint32_t arr = atomicAdd(&q[0], 1u) + 1u;
        if (arr == gridDim.x) {
            __atomic_store_n(&q[0], 0u, __ATOMIC_RELEASE);
            if (!static_224) {
                __atomic_store_n(&q[32], 0u, __ATOMIC_RELEASE);
                __atomic_store_n(&q[48], 0u, __ATOMIC_RELEASE);
            }
            __atomic_store_n(&q[16], sense, __ATOMIC_RELEASE);
        } else {
            while (__atomic_load_n(&q[16], __ATOMIC_ACQUIRE) != sense) {
                __builtin_amdgcn_s_sleep(1);
            }
        }
    }
    if (threadIdx.x == 0u) asm volatile("buffer_wbinvl1\n" ::: "memory");
    __syncthreads();

    // ---- stage B ----
    if (static_224 && wg >= 192u) return;
    sm_x[threadIdx.x] = reinterpret_cast<const uint4*>(mid_q8)[threadIdx.x];
    if (threadIdx.x < 128u) sm_xd[threadIdx.x] = mid_d[threadIdx.x];
    __syncthreads();

    uint32_t b_step = 0u;
    #pragma nounroll
    for (;;) {
        uint32_t t;
        if (static_224) {
            if (b_step > 0u) break;
            t = wg;
            ++b_step;
        } else {
            if (threadIdx.x == 0u) sm_tile = atomicAdd(&q[48], 1u);
            __syncthreads();
            t = sm_tile;
            if (t >= n_b) break;
        }
        const uint32_t grp16 = threadIdx.x >> 4;
        const uint32_t row   = t * 16u + grp16;
        if (row < D) {
            const float total = vegajet::dot8_dev::vj_dot8_down_row_iq3<4>(wdn_d8, dn_w_stride, wdn_d8_sc, dn_s_stride,
                                                                          sm_x, sm_xd, sm_ids, sm_w, row, D, lane16);
            if (lane16 == 0u) {
                const float v = total + sh_down[row];
                out_f32[row] = v;
                sm_y[grp16]  = v;
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
                __builtin_nontemporal_store(pv, reinterpret_cast<u32x4_t*>(peer_tx + t * 16u + o));
            }
        }
        if (!static_224) __syncthreads();
    }
}

// ============================================================================
// 8c. Stage 4B Fused Post-Attn P2P AllReduce (Pre-Pushed) + M=1 MoE Front (-48 dispatches/tok!)
//     WG 0 executes bit-identical ornith_p2p_ar_fp16_body<true>, syncs all 161 WGs via bar+32,
//     then executes Stage 1 (odd-256B router + sh_gu) -> bar -> Stage 2 (Top-8 + sh SwiGLU).
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_p2p_ar_moe_m1_front_kernel(
    const float*    __restrict__ x_partial,
    uint16_t*       __restrict__ peer_tx,
    const uint16_t* __restrict__ local_rx,
    uint32_t*       __restrict__ peer_flag,
    const uint32_t* __restrict__ local_flag,
    float*          __restrict__ h_io,
    const float*    __restrict__ w_rms,
    float*          __restrict__ x_norm,
    int8_t*         __restrict__ x_q8,
    float2*         __restrict__ x_ds,
    const uint16_t* __restrict__ w_router_f16,
    const float*    __restrict__ w_sh_gate,
    const uint8_t*  __restrict__ sh_w_qs,
    const uint32_t* __restrict__ sh_w_dm,
    float*          __restrict__ logits,
    float*          __restrict__ sh_gu,
    int32_t*        __restrict__ top_ids,
    float*          __restrict__ top_w,
    int8_t*         __restrict__ sh_q8,
    float2*         __restrict__ sh_ds,
    uint32_t*       __restrict__ bar,
    uint32_t D,
    uint32_t epoch,
    float    eps,
    uint32_t sh_rows,
    uint32_t sh_row_stride_bytes,
    uint32_t local_ff,
    uint32_t sense) {
    const uint32_t wg  = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t grp = tid >> 5;
    const uint32_t l32 = tid & 31u;

    __shared__ float   sm_v[4];
    __shared__ float   sm_gate;
    alignas(16) __shared__ float sm_p[256];
    __shared__ float   sm_tw[8];
    __shared__ int32_t sm_ti[8];

    // ---- Stage 0: Pre-pushed FP16 P2P AllReduce + Residual + RMSNorm + Q8 Quant on WG 0 ----
    if (wg == 0u) {
        ornith_p2p_ar_fp16_body<true>(x_partial, peer_tx, local_rx, peer_flag, local_flag, bar,
                                      h_io, w_rms, x_norm, x_q8, x_ds, 1u, D, epoch, 1u, eps, nullptr);
    }
    __threadfence();
    vegajet::ornith_dev::grid_barrier_m(bar + 32u, gridDim.x, sense);
    if (tid == 0u) asm volatile("buffer_wbinvl1\n" ::: "memory");
    __syncthreads();

    const float4* x4f = reinterpret_cast<const float4*>(x_norm);
    const uint32_t gate_wg = gridDim.x - 1u;
    const bool split64 = (sh_rows >= 256u && gate_wg >= 64u);

    // ---- Stage 1: Router FP16 (odd-256B TCC stride) + Shared Gate/Up + Shared Gate ----
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

    __threadfence();
    vegajet::ornith_dev::grid_barrier_m(bar, gridDim.x, sense);
    if (tid == 0u) asm volatile("buffer_wbinvl1\n" ::: "memory");
    __syncthreads();

    // ---- Stage 2: Top-8 Softmax + Shared SwiGLU + Q8 Quant ----
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
// 9e. Stage 4B Specialized Static-224 M=1 MoE Back (2-deep it prefetch in IQ2_S Gate+Up, 0 atomic loop!)
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_moe_m1_back_s224_kernel(
    const uint4*    __restrict__ wgu_merged,
    const int8_t*   __restrict__ x_q8,
    const float2*   __restrict__ x_ds,
    const int32_t*  __restrict__ top_ids,
    const float*    __restrict__ top_w,
    int8_t*         __restrict__ mid_q8,
    float*          __restrict__ mid_d,
    const uint8_t*  __restrict__ sh_wqs,
    const uint32_t* __restrict__ sh_wdm,
    const int8_t*   __restrict__ sh_q8,
    const float2*   __restrict__ sh_ds,
    float*          __restrict__ sh_down,
    const uint8_t*  __restrict__ wdn_d8,
    const uint8_t*  __restrict__ wdn_d8_sc,
    float*          __restrict__ out_f32,
    uint32_t*       __restrict__ q,
    uint32_t local_ff,
    uint32_t D,
    uint32_t sh_rows,
    uint32_t sh_stride,
    uint32_t sense,
    uint32_t dn_w_stride,
    uint32_t dn_s_stride,
    uint16_t*       __restrict__ peer_tx) {
    __shared__ uint32_t sm_grid32[1024]; // 4,096 B (half of sm_grid -> 1-bank ds_read_b32 + v_perm_b32!)
    __shared__ uint4    sm_x[256];       // 4,096 B
    __shared__ float    sm_xd[128];      //   512 B
    __shared__ float    sm_y[32];        //   128 B
    __shared__ int32_t  sm_ids[8];
    __shared__ float    sm_w[8];

    const uint32_t wg     = blockIdx.x;
    const uint32_t tid    = threadIdx.x;
    const uint32_t lane16 = tid & 15u;

    if (tid < 192u) sm_x[tid]  = reinterpret_cast<const uint4*>(x_q8)[tid];
    if (tid <  96u) sm_xd[tid] = x_ds[tid].x;
    if (tid <   8u) {
        sm_ids[tid] = top_ids[tid];
        sm_w[tid]   = top_w[tid];
    }
    if (wg < 128u) {
        reinterpret_cast<uint4*>(sm_grid32)[tid] = reinterpret_cast<const uint4*>(vegajet::ggml_tables::iq2s_grid32)[tid];
    }
    __syncthreads();

    // ---- Stage A (zero dynamic atomic loop!) ----
    if (wg < 128u) {
        const uint32_t slot   = wg >> 4;
        const uint32_t rblk   = wg & 15u;
        const uint32_t exp_id = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(sm_ids[slot]));
        const uint32_t sub_id = tid >> 4;

        #pragma nounroll
        for (uint32_t half = 0u; half < 2u; ++half) {
            const uint32_t row  = rblk * 32u + half * 16u + sub_id;
            const uint64_t erow = static_cast<uint64_t>(exp_id) * 512ULL + row;
            const uint4*   qp_p = wgu_merged + erow * 123ULL;

            // 2-deep software prefetch across it = 0..5: next block's HBM2 loads overlap current sdot4/sm_grid!
            uint4    wq  = vegajet::ornith_dev::ldg_u32x4_stream(qp_p + lane16);
            uint32_t hs2 = vegajet::ornith_dev::ldg_u32_stream(reinterpret_cast<const uint32_t*>(qp_p + 96u) + lane16);
            uint32_t d2  = reinterpret_cast<const uint32_t*>(qp_p + 120u)[lane16 >> 3];
            float ag = 0.0f, au = 0.0f;

            #pragma nounroll
            for (uint32_t it = 0u; it < 6u; ++it) {
                const uint32_t ib = it * 16u + lane16;
                uint4    wq_n;
                uint32_t hs2_n, d2_n;
                if (it < 5u) {
                    const uint32_t ibn = ib + 16u;
                    wq_n  = vegajet::ornith_dev::ldg_u32x4_stream(qp_p + ibn);
                    hs2_n = vegajet::ornith_dev::ldg_u32_stream(reinterpret_cast<const uint32_t*>(qp_p + 96u) + ibn);
                    d2_n  = reinterpret_cast<const uint32_t*>(qp_p + 120u)[ibn >> 3];
                }
                const uint4 xa  = sm_x[ib * 2u];
                const uint4 xb  = sm_x[ib * 2u + 1u];
                const float xda = sm_xd[ib];
                __builtin_amdgcn_sched_barrier(0);

                #pragma nounroll
                for (uint32_t part = 0u; part < 2u; ++part) {
                    const uint32_t qa  = part ? wq.z : wq.x;
                    const uint32_t qb  = part ? wq.w : wq.y;
                    const uint32_t hsp = part ? (hs2 >> 16) : (hs2 & 0xFFFFu);
                    const uint32_t qh  = hsp & 0xFFu, sc = hsp >> 8;
                    const float pd = vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(part ? (d2 >> 16) : (d2 & 0xFFFFu))) * 0.25f * xda;
                    float acc = part ? au : ag;
                    const uint32_t g0 = sm_grid32[(qa & 0xFFu)         | ((qh << 8) & 0x300u)];
                    const uint32_t g1 = sm_grid32[((qa >> 16) & 0xFFu) | ((qh << 6) & 0x300u)];
                    const uint32_t g2 = sm_grid32[(qb & 0xFFu)         | ((qh << 4) & 0x300u)];
                    const uint32_t g3 = sm_grid32[((qb >> 16) & 0xFFu) | ((qh << 2) & 0x300u)];
                    int32_t dp0 = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::iq2s_perm4(g0,      (qa >> 8)  & 0xFu), xa.x, 0);
                    dp0         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::iq2s_perm4(g0 >> 2, (qa >> 12) & 0xFu), xa.y, dp0);
                    dp0         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::iq2s_perm4(g1,      (qa >> 24) & 0xFu), xa.z, dp0);
                    dp0         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::iq2s_perm4(g1 >> 2, qa >> 28),           xa.w, dp0);
                    acc = fmaf(pd * (0.5f + static_cast<float>(sc & 0xFu)), static_cast<float>(dp0), acc);
                    int32_t dp1 = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::iq2s_perm4(g2,      (qb >> 8)  & 0xFu), xb.x, 0);
                    dp1         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::iq2s_perm4(g2 >> 2, (qb >> 12) & 0xFu), xb.y, dp1);
                    dp1         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::iq2s_perm4(g3,      (qb >> 24) & 0xFu), xb.z, dp1);
                    dp1         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::iq2s_perm4(g3 >> 2, qb >> 28),           xb.w, dp1);
                    acc = fmaf(pd * (0.5f + static_cast<float>(sc >> 4)), static_cast<float>(dp1), acc);
                    if (part) au = acc; else ag = acc;
                }
                if (it < 5u) { wq = wq_n; hs2 = hs2_n; d2 = d2_n; }
            }
            ag = vegajet::dot8_dev::sum16(ag);
            au = vegajet::dot8_dev::sum16(au);
            if (lane16 == 0u) sm_y[half * 16u + sub_id] = (ag / (1.0f + __expf(-ag))) * au;
        }
        __syncthreads();
        if (tid < 32u) {
            const float y    = sm_y[tid];
            const float amax = vegajet::ornith_dev::block32_max(fabsf(y));
            const float d    = amax * (1.0f / 127.0f);
            const float inv  = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            mid_q8[static_cast<uint64_t>(slot) * 512ULL + rblk * 32u + tid] =
                static_cast<int8_t>(max(-127, min(127, __float2int_rn(y * inv))));
            if (tid == 0u) mid_d[static_cast<uint64_t>(slot) * 16ULL + rblk] = d;
        }
    } else {
        const uint4* x4 = reinterpret_cast<const uint4*>(sh_q8);
        #pragma nounroll
        for (uint32_t step = 0u; step < 2u; ++step) {
            const uint32_t row = ((wg - 128u) * 2u + step) * 16u + (tid >> 4);
            const uint4*    w4 = reinterpret_cast<const uint4*>(sh_wqs + static_cast<uint64_t>(row) * sh_stride);
            const uint32_t* dm = sh_wdm + static_cast<uint64_t>(row) * 16u;
            const uint4    wv  = vegajet::ornith_dev::ldg_u32x4_stream(w4 + lane16);
            const uint32_t dmv = vegajet::ornith_dev::ldg_u32_stream(dm + lane16);
            const uint4    xa  = x4[lane16 * 2u];
            const uint4    xb  = x4[lane16 * 2u + 1u];
            const float2   xds = sh_ds[lane16];
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
            float acc = vegajet::dot8_dev::sum16(fmaf(wd * xds.x, static_cast<float>(dp), wm * xds.y));
            if (lane16 == 0u) sh_down[row] = acc;
        }
    }

    __threadfence();
    vegajet::ornith_dev::grid_barrier_m(q, 224u, sense);
    if (tid == 0u) asm volatile("buffer_wbinvl1\n" ::: "memory");
    __syncthreads();

    // ---- Stage B: 192 WGs Routed Down + sh_down + FP16 BAR0 prepush ----
    if (wg >= 192u) return;
    sm_x[tid] = reinterpret_cast<const uint4*>(mid_q8)[tid];
    if (tid < 128u) sm_xd[tid] = mid_d[tid];
    __syncthreads();

    const uint32_t grp16 = tid >> 4;
    const uint32_t row   = wg * 16u + grp16;
    const float total = vegajet::dot8_dev::vj_dot8_down_row_iq3<4>(wdn_d8, dn_w_stride, wdn_d8_sc, dn_s_stride,
                                                                  sm_x, sm_xd, sm_ids, sm_w, row, D, lane16);
    if (lane16 == 0u) {
        const float v = total + sh_down[row];
        out_f32[row] = v;
        sm_y[grp16]  = v;
    }
    if (peer_tx != nullptr) {
        __syncthreads();
        if (tid < 2u) {
            const uint32_t o = tid * 8u;
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
            __builtin_nontemporal_store(pv, reinterpret_cast<u32x4_t*>(peer_tx + wg * 16u + o));
        }
    }
}

// ============================================================================
// 8d. Wave 1D Zero-Skew 240-WG Fused Post-Attn P2P AllReduce + M=1 MoE Front
//     240 resident WGs (4 WGs/CU x 60 CUs):
//       - WGs 0..127   (128 WGs): Shared Gate+Up (8 rows/WG, 32 lanes/row = 15.4 KB/WG, grid-strided)
//       - WGs 128..238 (111 WGs): Router FP16 (2-3 rows/WG, 128 lanes/row = 12.3-18.4 KB/WG, 0 Split-K buf)
//       - WG 239       (1 WG):    Shared gate scalar (12.3 KB/WG)
//     Peak WG work drops from 47.7 KB/WG -> 18.4 KB/WG (2.60x reduction, 1.20x skew, 100% chip saturation!).
//     When fuse_ar == 0 (standalone / --profile-stages), skips Stage 0 AR cleanly.
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_p2p_ar_moe_m1_front_s240_kernel(
    const float*    __restrict__ x_partial,
    uint16_t*       __restrict__ peer_tx,
    const uint16_t* __restrict__ local_rx,
    uint32_t*       __restrict__ peer_flag,
    const uint32_t* __restrict__ local_flag,
    float*          __restrict__ h_io,
    const float*    __restrict__ w_rms,
    float*          __restrict__ x_norm,
    int8_t*         __restrict__ x_q8,
    float2*         __restrict__ x_ds,
    const uint16_t* __restrict__ w_router_f16,
    const float*    __restrict__ w_sh_gate,
    const uint8_t*  __restrict__ sh_w_qs,
    const uint32_t* __restrict__ sh_w_dm,
    float*          __restrict__ logits,
    float*          __restrict__ sh_gu,
    int32_t*        __restrict__ top_ids,
    float*          __restrict__ top_w,
    int8_t*         __restrict__ sh_q8,
    float2*         __restrict__ sh_ds,
    uint32_t*       __restrict__ bar,
    uint32_t D,
    uint32_t epoch,
    float    eps,
    uint32_t sh_rows,
    uint32_t sh_row_stride_bytes,
    uint32_t local_ff,
    uint32_t sense,
    uint32_t fuse_ar) {
    const uint32_t wg  = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t grp = tid >> 5;
    const uint32_t l32 = tid & 31u;

    __shared__ float   sm_v[4];
    __shared__ float   sm_gate;
    alignas(16) __shared__ float sm_p[256];
    __shared__ float   sm_tw[8];
    __shared__ int32_t sm_ti[8];

    // ---- Stage 0: Optional Pre-pushed FP16 P2P AllReduce + Residual + RMSNorm + Q8 Quant on WG 0 ----
    if (fuse_ar != 0u) {
        if (wg == 0u) {
            if (fuse_ar == 1u) {
                ornith_p2p_ar_fp16_body<true>(x_partial, peer_tx, local_rx, peer_flag, local_flag, bar,
                                              h_io, w_rms, x_norm, x_q8, x_ds, 1u, D, epoch, 1u, eps, nullptr);
            } else {
                ornith_p2p_ar_fp16_body<false>(x_partial, peer_tx, local_rx, peer_flag, local_flag, bar,
                                               h_io, w_rms, x_norm, x_q8, x_ds, 1u, D, epoch, 1u, eps, nullptr);
            }
        }
        __threadfence();
        vegajet::ornith_dev::grid_barrier_m(bar + 32u, gridDim.x, sense);
        if (tid == 0u) asm volatile("buffer_wbinvl1\n" ::: "memory");
        __syncthreads();
    }

    const float4*  x4f     = reinterpret_cast<const float4*>(x_norm);
    const uint32_t n_wgs   = gridDim.x;
    const uint32_t gate_wg = n_wgs - 1u;
    const uint32_t sh_wgs  = min(128u, (sh_rows + 7u) >> 3); // 128 WGs for sh_rows=1024
    const uint32_t rtr_wgs = (gate_wg > sh_wgs) ? (gate_wg - sh_wgs) : 1u;

    // ---- Stage 1: Zero-skew 240-WG Router FP16 + Shared Gate/Up + Shared Gate ----
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
    } else if (wg < sh_wgs) {
        const uint32_t nb = D >> 5;
        const uint32_t b_last = nb - 1u;
        const uint4*   x4 = reinterpret_cast<const uint4*>(x_q8);
        #pragma nounroll
        for (uint32_t row = wg * 8u + grp; row < sh_rows; row += sh_wgs * 8u) {
            const uint4*    w4 = reinterpret_cast<const uint4*>(sh_w_qs + static_cast<uint64_t>(row) * sh_row_stride_bytes);
            const uint32_t* dm = sh_w_dm + static_cast<uint64_t>(row) * nb;
            float acc = 0.0f;
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
    } else {
        // 111 Router WGs (wg = 128..238): 2 rows/iter (128 lanes/row = 2 Wave64s, 3 uint4s/lane at D=3072!)
        const uint32_t rwg       = wg - sh_wgs;
        const uint32_t wr_stride = (D == 3072u) ? 3200u : D;
        const uint32_t n8        = D >> 3;
        const uint32_t half_id   = tid >> 7;     // 0 or 1 (2 rows in parallel!)
        const uint32_t l128      = tid & 127u;   // 0..127
        const uint32_t wave      = tid >> 6;     // 0..3
        const uint32_t lane64    = tid & 63u;
        #pragma nounroll
        for (uint32_t e_base = rwg * 2u; e_base < 256u; e_base += rtr_wgs * 2u) {
            const uint32_t e = e_base + half_id;
            float acc = 0.0f;
            if (e < 256u) {
                const uint4* w4 = reinterpret_cast<const uint4*>(w_router_f16 + static_cast<uint64_t>(e) * wr_stride);
                if (n8 == 384u) {
                    const uint4 w0 = vegajet::ornith_dev::ldg_u32x4_stream(w4 + l128);
                    const uint4 w1 = vegajet::ornith_dev::ldg_u32x4_stream(w4 + l128 + 128u);
                    const uint4 w2 = vegajet::ornith_dev::ldg_u32x4_stream(w4 + l128 + 256u);
                    #pragma unroll
                    for (uint32_t r = 0u; r < 3u; ++r) {
                        const uint32_t i  = l128 + r * 128u;
                        const uint4    wv = (r == 0u) ? w0 : ((r == 1u) ? w1 : w2);
                        const float4   xa = x4f[i * 2u];
                        const float4   xb = x4f[i * 2u + 1u];
                        const float2   h0 = __half22float2(*reinterpret_cast<const __half2*>(&wv.x));
                        const float2   h1 = __half22float2(*reinterpret_cast<const __half2*>(&wv.y));
                        const float2   h2 = __half22float2(*reinterpret_cast<const __half2*>(&wv.z));
                        const float2   h3 = __half22float2(*reinterpret_cast<const __half2*>(&wv.w));
                        acc = fmaf(h0.x, xa.x, fmaf(h0.y, xa.y, fmaf(h1.x, xa.z, fmaf(h1.y, xa.w, acc))));
                        acc = fmaf(h2.x, xb.x, fmaf(h2.y, xb.y, fmaf(h3.x, xb.z, fmaf(h3.y, xb.w, acc))));
                    }
                } else {
                    #pragma nounroll
                    for (uint32_t i = l128; i < n8; i += 128u) {
                        const uint4  wv = vegajet::ornith_dev::ldg_u32x4_stream(w4 + i);
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
            acc = vegajet::ornith_dev::wave64_sum(acc);
            if (lane64 == 0u) sm_v[wave] = acc;
            __syncthreads();
            if (tid < 2u) {
                const uint32_t e_out = e_base + tid;
                if (e_out < 256u) logits[e_out] = sm_v[tid * 2u] + sm_v[tid * 2u + 1u];
            }
            __syncthreads();
        }
    }

    __threadfence();
    vegajet::ornith_dev::grid_barrier_m(bar, n_wgs, sense);
    if (tid == 0u) asm volatile("buffer_wbinvl1\n" ::: "memory");
    __syncthreads();

    // ---- Stage 2: Top-8 Softmax (WG 0) + Shared SwiGLU + Q8 Quant (WG gate_wg) ----
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
// 9f. Wave 1D Zero-Skew 240-WG M=1 MoE Back (Shared Down moved to Phase 2!)
//     Phase 1 (ALL 240 WGs 100% on Routed Gate+Up):
//       - First wq/hs2 non-temporal prefetch issued BEFORE 8 KiB iq2s_grid LDS copy + __syncthreads()!
//       - 240 WGs x 16 primary rows (3840 rows) + 1-2 remainder rows in 1 single parallel pass (1.06x skew!)
//       - Critical-path load drops 63.0 -> 33.5-35.4 KB/WG (-46.7%!). Writes 16 KiB mid_f32 (L2-resident).
//     Phase 2 (ALL 240 WGs on Routed Down + IN-REGISTER Shared Down):
//       - Every WG quantizes L2-resident mid_f32 -> LDS sm_x/sm_xd (exact block-32 INT8)
//       - 96 WGs x 14 rows + 144 WGs x 12 rows = 3072 rows (29.9-34.9 KB/WG, 1.16x skew, 0 sh_down buffer!)
//       - Coalesced 4-byte __half2 BAR0 prepush (write-combining inside each WG).
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_moe_m1_back_s240_kernel(
    const uint4*    __restrict__ wgu_merged,
    const int8_t*   __restrict__ x_q8,
    const float2*   __restrict__ x_ds,
    const int32_t*  __restrict__ top_ids,
    const float*    __restrict__ top_w,
    float*          __restrict__ mid_f32,     // [8 * 512] FP32 SwiGLU intermediate (16 KiB, L2-resident)
    const uint8_t*  __restrict__ sh_wqs,
    const uint32_t* __restrict__ sh_wdm,
    const int8_t*   __restrict__ sh_q8,
    const float2*   __restrict__ sh_ds,
    const uint8_t*  __restrict__ wdn_d8,
    const uint8_t*  __restrict__ wdn_d8_sc,
    float*          __restrict__ out_f32,
    uint32_t*       __restrict__ q,
    uint32_t local_ff,
    uint32_t D,
    uint32_t sh_stride,
    uint32_t sense,
    uint32_t dn_w_stride,
    uint32_t dn_s_stride,
    uint16_t*       __restrict__ peer_tx) {
    __shared__ uint64_t sm_grid[1024]; // 8,192 B
    __shared__ uint4    sm_x[256];     // 4,096 B
    __shared__ float    sm_xd[128];    //   512 B
    __shared__ float    sm_y[32];      //   128 B
    __shared__ int32_t  sm_ids[8];
    __shared__ float    sm_w[8];

    const uint32_t wg     = blockIdx.x;
    const uint32_t tid    = threadIdx.x;
    const uint32_t lane16 = tid & 15u;
    const uint32_t sub_id = tid >> 4;

    if (tid < 192u) sm_x[tid]  = reinterpret_cast<const uint4*>(x_q8)[tid];
    if (tid <  96u) sm_xd[tid] = x_ds[tid].x;
    if (tid <   8u) {
        sm_ids[tid] = top_ids[tid];
        sm_w[tid]   = top_w[tid];
    }

    // Issue primary-row it=0 non-temporal prefetch BEFORE staging 8 KiB iq2s_grid into LDS!
    const uint32_t gidx0  = wg * 16u + sub_id; // 0..3839 (16 primary rows/WG)
    const uint32_t slot0  = gidx0 >> 9;        // / 512
    const uint32_t rloc0  = gidx0 & 511u;
    const uint32_t exp_id0 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(top_ids[slot0]));
    const uint4*   qp_p0  = wgu_merged + (static_cast<uint64_t>(exp_id0) * 512ULL + rloc0) * 123ULL;
    uint4          wq     = vegajet::ornith_dev::ldg_u32x4_stream(qp_p0 + lane16);
    uint32_t       hs2    = vegajet::ornith_dev::ldg_u32_stream(reinterpret_cast<const uint32_t*>(qp_p0 + 96u) + lane16);
    uint32_t       d2     = reinterpret_cast<const uint32_t*>(qp_p0 + 120u)[lane16 >> 3];

    reinterpret_cast<uint4*>(sm_grid)[tid]        = reinterpret_cast<const uint4*>(vegajet::ggml_tables::iq2s_grid)[tid];
    reinterpret_cast<uint4*>(sm_grid)[tid + 256u] = reinterpret_cast<const uint4*>(vegajet::ggml_tables::iq2s_grid)[tid + 256u];
    __syncthreads();

    // ---- Phase 1a: 16 primary rows/WG across ALL 240 WGs (rows 0..3839, 100% active threads) ----
    {
        float ag = 0.0f, au = 0.0f;
        #pragma nounroll
        for (uint32_t it = 0u; it < 6u; ++it) {
            const uint32_t ib = it * 16u + lane16;
            uint4    wq_n;
            uint32_t hs2_n, d2_n;
            if (it < 5u) {
                const uint32_t ibn = ib + 16u;
                wq_n  = vegajet::ornith_dev::ldg_u32x4_stream(qp_p0 + ibn);
                hs2_n = vegajet::ornith_dev::ldg_u32_stream(reinterpret_cast<const uint32_t*>(qp_p0 + 96u) + ibn);
                d2_n  = reinterpret_cast<const uint32_t*>(qp_p0 + 120u)[ibn >> 3];
            }
            const uint4 xa  = sm_x[ib * 2u];
            const uint4 xb  = sm_x[ib * 2u + 1u];
            const float xda = sm_xd[ib];
            __builtin_amdgcn_sched_barrier(0);

            #pragma nounroll
            for (uint32_t part = 0u; part < 2u; ++part) {
                const uint32_t qa  = part ? wq.z : wq.x;
                const uint32_t qb  = part ? wq.w : wq.y;
                const uint32_t hsp = part ? (hs2 >> 16) : (hs2 & 0xFFFFu);
                const uint32_t qh  = hsp & 0xFFu, sc = hsp >> 8;
                const float pd = vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(part ? (d2 >> 16) : (d2 & 0xFFFFu))) * 0.25f * xda;
                float acc = part ? au : ag;
                uint64_t g0 = sm_grid[(qa & 0xFFu)         | ((qh << 8) & 0x300u)];
                uint64_t g1 = sm_grid[((qa >> 16) & 0xFFu) | ((qh << 6) & 0x300u)];
                int32_t dp0 = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g0),       (qa >> 8)  & 0xFu), xa.x, 0);
                dp0         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g0 >> 32), (qa >> 12) & 0xFu), xa.y, dp0);
                dp0         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g1),       (qa >> 24) & 0xFu), xa.z, dp0);
                dp0         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g1 >> 32), qa >> 28),          xa.w, dp0);
                acc = fmaf(pd * (0.5f + static_cast<float>(sc & 0xFu)), static_cast<float>(dp0), acc);
                __builtin_amdgcn_sched_barrier(0);

                uint64_t g2 = sm_grid[(qb & 0xFFu)         | ((qh << 4) & 0x300u)];
                uint64_t g3 = sm_grid[((qb >> 16) & 0xFFu) | ((qh << 2) & 0x300u)];
                int32_t dp1 = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g2),       (qb >> 8)  & 0xFu), xb.x, 0);
                dp1         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g2 >> 32), (qb >> 12) & 0xFu), xb.y, dp1);
                dp1         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g3),       (qb >> 24) & 0xFu), xb.z, dp1);
                dp1         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g3 >> 32), qb >> 28),          xb.w, dp1);
                acc = fmaf(pd * (0.5f + static_cast<float>(sc >> 4)), static_cast<float>(dp1), acc);
                if (part) au = acc; else ag = acc;
                __builtin_amdgcn_sched_barrier(0);
            }
            if (it < 5u) { wq = wq_n; hs2 = hs2_n; d2 = d2_n; }
        }
        ag = vegajet::dot8_dev::sum16(ag);
        au = vegajet::dot8_dev::sum16(au);
        if (lane16 == 0u) mid_f32[gidx0] = (ag / (1.0f + __expf(-ag))) * au;
    }

    // ---- Phase 1b: Remaining 256 rows (3840..4095) in ONE parallel 96-thread/row pass (0 extra loop!) ----
    //   Half 0 (tid 0..127):   row 3840 + wg (all 240 WGs -> 3840..4079)
    //   Half 1 (tid 128..255): row 4080 + wg (wg < 16 -> 4080..4095)
    {
        const uint32_t h_id  = tid >> 7;
        const uint32_t ib    = tid & 127u;
        const uint32_t gidx  = 3840u + h_id * 240u + wg;
        float ag = 0.0f, au = 0.0f;
        if (gidx < 4096u && ib < 96u) {
            const uint32_t slot   = gidx >> 9;
            const uint32_t rloc   = gidx & 511u;
            const uint32_t exp_id = static_cast<uint32_t>(sm_ids[slot]);
            const uint4*   qp_p   = wgu_merged + (static_cast<uint64_t>(exp_id) * 512ULL + rloc) * 123ULL;
            const uint4    wqr    = vegajet::ornith_dev::ldg_u32x4_stream(qp_p + ib);
            const uint32_t hs2r   = vegajet::ornith_dev::ldg_u32_stream(reinterpret_cast<const uint32_t*>(qp_p + 96u) + ib);
            const uint32_t d2r    = reinterpret_cast<const uint32_t*>(qp_p + 120u)[ib >> 3];
            const uint4    xa     = sm_x[ib * 2u];
            const uint4    xb     = sm_x[ib * 2u + 1u];
            const float    xda    = sm_xd[ib];
            #pragma nounroll
            for (uint32_t part = 0u; part < 2u; ++part) {
                const uint32_t qa  = part ? wqr.z : wqr.x;
                const uint32_t qb  = part ? wqr.w : wqr.y;
                const uint32_t hsp = part ? (hs2r >> 16) : (hs2r & 0xFFFFu);
                const uint32_t qh  = hsp & 0xFFu, sc = hsp >> 8;
                const float pd = vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(part ? (d2r >> 16) : (d2r & 0xFFFFu))) * 0.25f * xda;
                uint64_t g0 = sm_grid[(qa & 0xFFu)         | ((qh << 8) & 0x300u)];
                uint64_t g1 = sm_grid[((qa >> 16) & 0xFFu) | ((qh << 6) & 0x300u)];
                int32_t dp0 = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g0),       (qa >> 8)  & 0xFu), xa.x, 0);
                dp0         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g0 >> 32), (qa >> 12) & 0xFu), xa.y, dp0);
                dp0         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g1),       (qa >> 24) & 0xFu), xa.z, dp0);
                dp0         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g1 >> 32), qa >> 28),          xa.w, dp0);
                float acc = pd * (0.5f + static_cast<float>(sc & 0xFu)) * static_cast<float>(dp0);

                uint64_t g2 = sm_grid[(qb & 0xFFu)         | ((qh << 4) & 0x300u)];
                uint64_t g3 = sm_grid[((qb >> 16) & 0xFFu) | ((qh << 2) & 0x300u)];
                int32_t dp1 = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g2),       (qb >> 8)  & 0xFu), xb.x, 0);
                dp1         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g2 >> 32), (qb >> 12) & 0xFu), xb.y, dp1);
                dp1         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g3),       (qb >> 24) & 0xFu), xb.z, dp1);
                dp1         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g3 >> 32), qb >> 28),          xb.w, dp1);
                acc = fmaf(pd * (0.5f + static_cast<float>(sc >> 4)), static_cast<float>(dp1), acc);
                if (part) au = acc; else ag = acc;
            }
        }
        ag = vegajet::ornith_dev::wave64_sum(ag);
        au = vegajet::ornith_dev::wave64_sum(au);
        const uint32_t wave = tid >> 6;
        if ((tid & 63u) == 0u) { sm_y[wave] = ag; sm_y[wave + 4u] = au; }
        __syncthreads();
        if (tid < 2u) {
            const uint32_t go = 3840u + tid * 240u + wg;
            if (go < 4096u) {
                const float gsum = sm_y[tid * 2u] + sm_y[tid * 2u + 1u];
                const float usum = sm_y[tid * 2u + 4u] + sm_y[tid * 2u + 5u];
                mid_f32[go] = (gsum / (1.0f + __expf(-gsum))) * usum;
            }
        }
    }

    __threadfence();
    vegajet::ornith_dev::grid_barrier_m(q, 240u, sense);
    if (tid == 0u) asm volatile("buffer_wbinvl1\n" ::: "memory");
    __syncthreads();

    // ---- Phase 2a: Quantize 16 KiB L2-resident mid_f32 -> LDS sm_x[256] + sm_xd[128] ----
    {
        const uint32_t b_in = tid >> 5;   // 0..7
        const uint32_t l32  = tid & 31u;
        int8_t* sm_x_i8 = reinterpret_cast<int8_t*>(sm_x);
        #pragma nounroll
        for (uint32_t iter = 0u; iter < 16u; ++iter) {
            const uint32_t blk  = iter * 8u + b_in;
            const float    yv   = mid_f32[blk * 32u + l32];
            const float    amax = vegajet::ornith_dev::block32_max(fabsf(yv));
            const float    d    = amax * (1.0f / 127.0f);
            const float    inv  = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            sm_x_i8[blk * 32u + l32] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(yv * inv))));
            if (l32 == 0u) sm_xd[blk] = d;
        }
    }
    __syncthreads();

    // ---- Phase 2b: ALL 240 WGs compute Routed Down + IN-REGISTER Shared Down (1.16x skew!) ----
    //   WGs 0..95:   14 rows each (0..1343)   | WGs 96..239: 12 rows each (1344..3071) -> 3072 EXACT!
    const uint32_t row0   = (wg < 96u) ? (wg * 14u) : (1344u + (wg - 96u) * 12u);
    const uint32_t n_rows = (wg < 96u) ? 14u : 12u;
    if (sub_id < n_rows) {
        const uint32_t row = row0 + sub_id;
        // 1. Routed Down (8 experts, exact DOT8)
        const float total = vegajet::dot8_dev::vj_dot8_down_row_iq3<4>(wdn_d8, dn_w_stride, wdn_d8_sc, dn_s_stride,
                                                                      sm_x, sm_xd, sm_ids, sm_w, row, D, lane16);
        // 2. Fused Shared Down (1 block/lane for local_ff=512 -> 0 global sh_down write/read!)
        const uint4*    sw4 = reinterpret_cast<const uint4*>(sh_wqs + static_cast<uint64_t>(row) * sh_stride);
        const uint32_t* sdm = sh_wdm + static_cast<uint64_t>(row) * 16u;
        const uint4*    sx4 = reinterpret_cast<const uint4*>(sh_q8);
        const uint4     wv  = vegajet::ornith_dev::ldg_u32x4_stream(sw4 + lane16);
        const uint32_t  dmv = vegajet::ornith_dev::ldg_u32_stream(sdm + lane16);
        const uint4     xa  = sx4[lane16 * 2u];
        const uint4     xb  = sx4[lane16 * 2u + 1u];
        const float2    xds = sh_ds[lane16];
        int32_t dp = vegajet::ornith_dev::sdot4(wv.x & 0x0F0F0F0Fu, xa.x, 0);
        dp         = vegajet::ornith_dev::sdot4(wv.y & 0x0F0F0F0Fu, xa.y, dp);
        dp         = vegajet::ornith_dev::sdot4(wv.z & 0x0F0F0F0Fu, xa.z, dp);
        dp         = vegajet::ornith_dev::sdot4(wv.w & 0x0F0F0F0Fu, xa.w, dp);
        dp         = vegajet::ornith_dev::sdot4((wv.x >> 4) & 0x0F0F0F0Fu, xb.x, dp);
        dp         = vegajet::ornith_dev::sdot4((wv.y >> 4) & 0x0F0F0F0Fu, xb.y, dp);
        dp         = vegajet::ornith_dev::sdot4((wv.z >> 4) & 0x0F0F0F0Fu, xb.z, dp);
        dp         = vegajet::ornith_dev::sdot4((wv.w >> 4) & 0x0F0F0F0Fu, xb.w, dp);
        const float wd  = vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(dmv & 0xFFFFu));
        const float wm  = vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(dmv >> 16));
        const float shv = vegajet::dot8_dev::sum16(fmaf(wd * xds.x, static_cast<float>(dp), wm * xds.y));
        if (lane16 == 0u) {
            const float v = total + shv;
            out_f32[row] = v;
            sm_y[sub_id] = v;
        }
    }
    if (peer_tx != nullptr) {
        __syncthreads();
        const uint32_t n_pairs = n_rows >> 1; // 7 or 6 __half2s (28 B or 24 B contiguous per WG -> write-combined!)
        if (tid < n_pairs) {
            const __half2 h = __float22half2_rn(make_float2(sm_y[tid * 2u], sm_y[tid * 2u + 1u]));
            const uint32_t w = *reinterpret_cast<const uint32_t*>(&h);
            __builtin_nontemporal_store(w, reinterpret_cast<uint32_t*>(peer_tx + row0) + tid);
        }
    }
}

// ============================================================================
// 8e. Wave 1D Barrier-Free High-Occupancy M=1 MoE Front (2 tiny dispatches, 0 grid_barrier_m!)
//     (1) ornith_moe_m1_rtr_shgu_kernel: 64 thr/WG (0 B LDS, 16 WGs/CU = 960 resident WGs!),
//         128 WGs Router FP16 (2 rows/WG, exact block32_sum) + (sh_rows/4) WGs Shared Gate/Up (4 rows/WG,
//         exact block32_sum) + 1 WG Shared Gate scalar. Zero grid barrier, zero __syncthreads()!
//     (2) ornith_moe_m1_top8_swiglu_kernel: 2 WGs x 256 thr (WG 0 = Top-8 softmax, WG 1 = Shared SwiGLU+Q8).
// ============================================================================
extern "C" __global__ __launch_bounds__(64, 5)
void ornith_moe_m1_rtr_shgu_kernel(
    const uint16_t* __restrict__ w_router_f16,
    const float*    __restrict__ w_sh_gate,
    const float*    __restrict__ x_norm,
    const uint8_t*  __restrict__ sh_w_qs,
    const uint32_t* __restrict__ sh_w_dm,
    const int8_t*   __restrict__ x_q8,
    const float2*   __restrict__ x_ds,
    float*          __restrict__ logits,
    float*          __restrict__ sh_gate_sc,
    float*          __restrict__ sh_gu,
    uint32_t D,
    uint32_t sh_rows,
    uint32_t sh_row_stride_bytes) {
    const uint32_t wg      = blockIdx.x;
    const uint32_t tid     = threadIdx.x; // 0..63
    const uint32_t grp     = tid >> 5;    // 0..1
    const uint32_t l32     = tid & 31u;
    const uint32_t gate_wg = gridDim.x - 1u;

    if (wg < 128u) {
        const uint32_t e = wg * 2u + grp;
        const uint32_t wr_stride = (D == 3072u) ? 3200u : D;
        const uint4*   w4  = reinterpret_cast<const uint4*>(w_router_f16 + static_cast<uint64_t>(e) * wr_stride);
        const float4*  x4f = reinterpret_cast<const float4*>(x_norm);
        const uint32_t n8  = D >> 3;
        const uint32_t i_last = n8 - 1u;
        float acc = 0.0f;
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
    } else if (wg == gate_wg) {
        const float4* w4  = reinterpret_cast<const float4*>(w_sh_gate);
        const float4* x4f = reinterpret_cast<const float4*>(x_norm);
        const uint32_t n4 = D >> 2;
        float acc = 0.0f;
        #pragma nounroll
        for (uint32_t i = tid; i < n4; i += 64u) {
            const float4 wv = w4[i];
            const float4 xv = x4f[i];
            acc = fmaf(wv.x, xv.x, fmaf(wv.y, xv.y, fmaf(wv.z, xv.z, fmaf(wv.w, xv.w, acc))));
        }
        acc = vegajet::ornith_dev::wave64_sum(acc);
        if (tid == 0u) sh_gate_sc[0] = 1.0f / (1.0f + __expf(-acc));
    } else {
        const uint32_t sh_wg  = wg - 128u;
        const uint32_t nb     = D >> 5;
        const uint32_t b_last = nb - 1u;
        const uint4*   x4     = reinterpret_cast<const uint4*>(x_q8);
        #pragma nounroll
        for (uint32_t pass = 0u; pass < 2u; ++pass) {
            const uint32_t row = sh_wg * 4u + pass * 2u + grp;
            if (row < sh_rows) {
                const uint4*    w4 = reinterpret_cast<const uint4*>(sh_w_qs + static_cast<uint64_t>(row) * sh_row_stride_bytes);
                const uint32_t* dm = sh_w_dm + static_cast<uint64_t>(row) * nb;
                float acc = 0.0f;
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
    }
}

extern "C" __global__ __launch_bounds__(256, 4)
void ornith_moe_m1_top8_swiglu_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ sh_gu,
    const float* __restrict__ sh_gate_sc,
    int32_t*     __restrict__ top_ids,
    float*       __restrict__ top_w,
    int8_t*      __restrict__ sh_q8,
    float2*      __restrict__ sh_ds,
    uint32_t local_ff) {
    const uint32_t tid = threadIdx.x;
    if (blockIdx.x == 0u) {
        if (tid < 64u) vegajet::ornith_dev::fast_softmax_top8_wave64(logits, top_ids, top_w, tid);
    } else if (blockIdx.x == 1u) {
        const uint32_t l32 = tid & 31u;
        const float sc_g = sh_gate_sc[0];
        const float* g_p = sh_gu;
        const float* u_p = g_p + local_ff;
        #pragma nounroll
        for (uint32_t blk = tid >> 5; blk < (local_ff >> 5); blk += 8u) {
            const uint32_t c = (blk << 5) + l32;
            const float gv = g_p[c];
            const float y  = (gv / (1.0f + __expf(-gv))) * u_p[c] * sc_g;
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
// 9g. Wave 1D Barrier-Free High-Occupancy M=1 MoE Back (0 grid_barrier_m, 100% EXACT sum16!)
//     (1) ornith_moe_gu_iq2s_m1_8r_kernel: 512 WGs x 128 thr (8 rows/WG, 8,192 B LDS -> 8 WGs/CU = 480 resident!)
//     (2) ornith_moe_mid_f32_to_q8_kernel: 16 WGs x 256 thr (quantizes 16 KiB L2-hot mid_f32 -> mid_q8/mid_d)
//     (3) ornith_moe_down_d8_sh_m1_kernel: 384 WGs x 128 thr (8 rows/WG, 4,704 B LDS -> 10+ WGs/CU, fused sh_down + 16B PP!)
// ============================================================================
extern "C" __global__ __launch_bounds__(128, 4)
void ornith_moe_gu_iq2s_m1_8r_kernel(
    const uint4*   __restrict__ wgu_merged,
    const int8_t*  __restrict__ x_q8,
    const float2*  __restrict__ x_ds,
    const int32_t* __restrict__ top_ids,
    float*         __restrict__ mid_f32,
    uint32_t local_ff) {
    __shared__ uint64_t sm_grid[1024]; // 8,192 B -> 8 WGs/CU = 480 resident WGs on 60 CUs!

    const uint32_t tid    = threadIdx.x; // 0..127
    const uint32_t sub_id = tid >> 4;    // 0..7 (8 rows/WG)
    const uint32_t lane16 = tid & 15u;
    const uint32_t gidx   = blockIdx.x * 8u + sub_id;
    const uint32_t slot   = gidx / local_ff;
    const uint32_t rloc   = gidx - slot * local_ff;

    const uint32_t exp_id = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(top_ids[slot]));
    const uint4*   qp_p   = wgu_merged + (static_cast<uint64_t>(exp_id) * local_ff + rloc) * 123ULL;
    const uint4*   x4     = reinterpret_cast<const uint4*>(x_q8);

    // Issue it=0 non-temporal prefetch BEFORE staging 8 KiB iq2s_grid into LDS!
    uint4    wq  = vegajet::ornith_dev::ldg_u32x4_stream(qp_p + lane16);
    uint32_t hs2 = vegajet::ornith_dev::ldg_u32_stream(reinterpret_cast<const uint32_t*>(qp_p + 96u) + lane16);
    uint32_t d2  = reinterpret_cast<const uint32_t*>(qp_p + 120u)[lane16 >> 3];

    uint4* sm_g4 = reinterpret_cast<uint4*>(sm_grid);
    const uint4* src_g4 = reinterpret_cast<const uint4*>(vegajet::ggml_tables::iq2s_grid);
    sm_g4[tid]        = src_g4[tid];
    sm_g4[tid + 128u] = src_g4[tid + 128u];
    sm_g4[tid + 256u] = src_g4[tid + 256u];
    sm_g4[tid + 384u] = src_g4[tid + 384u];
    __syncthreads();

    float ag = 0.0f, au = 0.0f;
    #pragma nounroll
    for (uint32_t it = 0u; it < 6u; ++it) {
        const uint32_t ib = it * 16u + lane16;
        uint4    wq_n;
        uint32_t hs2_n, d2_n;
        if (it < 5u) {
            const uint32_t ibn = ib + 16u;
            wq_n  = vegajet::ornith_dev::ldg_u32x4_stream(qp_p + ibn);
            hs2_n = vegajet::ornith_dev::ldg_u32_stream(reinterpret_cast<const uint32_t*>(qp_p + 96u) + ibn);
            d2_n  = reinterpret_cast<const uint32_t*>(qp_p + 120u)[ibn >> 3];
        }
        const uint4 xa  = x4[ib * 2u];
        const uint4 xb  = x4[ib * 2u + 1u];
        const float xda = x_ds[ib].x;
        __builtin_amdgcn_sched_barrier(0);

        #pragma nounroll
        for (uint32_t part = 0u; part < 2u; ++part) {
            const uint32_t qa  = part ? wq.z : wq.x;
            const uint32_t qb  = part ? wq.w : wq.y;
            const uint32_t hsp = part ? (hs2 >> 16) : (hs2 & 0xFFFFu);
            const uint32_t qh  = hsp & 0xFFu, sc = hsp >> 8;
            const float pd = vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(part ? (d2 >> 16) : (d2 & 0xFFFFu))) * 0.25f * xda;
            float acc = part ? au : ag;
            uint64_t g0 = sm_grid[(qa & 0xFFu)         | ((qh << 8) & 0x300u)];
            uint64_t g1 = sm_grid[((qa >> 16) & 0xFFu) | ((qh << 6) & 0x300u)];
            int32_t dp0 = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g0),       (qa >> 8)  & 0xFu), xa.x, 0);
            dp0         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g0 >> 32), (qa >> 12) & 0xFu), xa.y, dp0);
            dp0         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g1),       (qa >> 24) & 0xFu), xa.z, dp0);
            dp0         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g1 >> 32), qa >> 28),          xa.w, dp0);
            acc = fmaf(pd * (0.5f + static_cast<float>(sc & 0xFu)), static_cast<float>(dp0), acc);
            __builtin_amdgcn_sched_barrier(0);

            uint64_t g2 = sm_grid[(qb & 0xFFu)         | ((qh << 4) & 0x300u)];
            uint64_t g3 = sm_grid[((qb >> 16) & 0xFFu) | ((qh << 2) & 0x300u)];
            int32_t dp1 = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g2),       (qb >> 8)  & 0xFu), xb.x, 0);
            dp1         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g2 >> 32), (qb >> 12) & 0xFu), xb.y, dp1);
            dp1         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g3),       (qb >> 24) & 0xFu), xb.z, dp1);
            dp1         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g3 >> 32), qb >> 28),          xb.w, dp1);
            acc = fmaf(pd * (0.5f + static_cast<float>(sc >> 4)), static_cast<float>(dp1), acc);
            if (part) au = acc; else ag = acc;
            __builtin_amdgcn_sched_barrier(0);
        }
        if (it < 5u) { wq = wq_n; hs2 = hs2_n; d2 = d2_n; }
    }
    ag = vegajet::dot8_dev::sum16(ag);
    au = vegajet::dot8_dev::sum16(au);
    if (lane16 == 0u) mid_f32[gidx] = (ag / (1.0f + __expf(-ag))) * au;
}


extern "C" __global__ __launch_bounds__(256, 8)
void ornith_moe_mid_f32_to_q8_kernel(
    const float* __restrict__ mid_f32,
    int8_t*      __restrict__ mid_q8,
    float*       __restrict__ mid_d,
    uint32_t n_blocks32) {
    const uint32_t blk = blockIdx.x * 8u + (threadIdx.x >> 5);
    const uint32_t l32 = threadIdx.x & 31u;
    if (blk >= n_blocks32) return;
    const float y    = mid_f32[blk * 32u + l32];
    const float amax = vegajet::ornith_dev::block32_max(fabsf(y));
    const float d    = amax * (1.0f / 127.0f);
    const float inv  = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    mid_q8[blk * 32u + l32] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(y * inv))));
    if (l32 == 0u) mid_d[blk] = d;
}

extern "C" __global__ __launch_bounds__(128, 5)
void ornith_moe_down_d8_sh_m1_kernel(
    const uint8_t*  __restrict__ wdn_d8,
    const uint8_t*  __restrict__ wdn_d8_sc,
    const int8_t*   __restrict__ mid_q8,
    const float*    __restrict__ mid_d,
    const int32_t*  __restrict__ top_ids,
    const float*    __restrict__ top_w,
    const uint8_t*  __restrict__ sh_wqs,
    const uint32_t* __restrict__ sh_wdm,
    const int8_t*   __restrict__ sh_q8,
    const float2*   __restrict__ sh_ds,
    float*          __restrict__ out_f32,
    uint32_t D,
    uint32_t sh_stride,
    uint32_t dn_w_stride,
    uint32_t dn_s_stride,
    uint16_t*       __restrict__ peer_tx) {
    __shared__ uint4   sm_x[256];  // 4,096 B
    __shared__ float   sm_xd[128]; //   512 B
    __shared__ int32_t sm_ids[8];
    __shared__ float   sm_w[8];
    __shared__ float   sm_y[8];

    const uint32_t tid    = threadIdx.x; // 0..127
    const uint32_t sub_id = tid >> 4;    // 0..7 (8 rows/WG)
    const uint32_t lane16 = tid & 15u;
    const uint32_t row    = blockIdx.x * 8u + sub_id;

    const uint4* mq4 = reinterpret_cast<const uint4*>(mid_q8);
    sm_x[tid]        = mq4[tid];
    sm_x[tid + 128u] = mq4[tid + 128u];
    sm_xd[tid]       = mid_d[tid];
    if (tid < 8u) { sm_ids[tid] = top_ids[tid]; sm_w[tid] = top_w[tid]; }
    __syncthreads();
    if (row >= D) return;

    const float total = vegajet::dot8_dev::vj_dot8_down_row_iq3<2>(wdn_d8, dn_w_stride, wdn_d8_sc, dn_s_stride,
                                                                  sm_x, sm_xd, sm_ids, sm_w, row, D, lane16);
    // In-register Shared Down (local_ff=512 -> 1 block/lane, 0 global sh_down write/read!)
    const uint4*    sw4 = reinterpret_cast<const uint4*>(sh_wqs + static_cast<uint64_t>(row) * sh_stride);
    const uint32_t* sdm = sh_wdm + static_cast<uint64_t>(row) * 16u;
    const uint4*    sx4 = reinterpret_cast<const uint4*>(sh_q8);
    const uint4     wv  = vegajet::ornith_dev::ldg_u32x4_stream(sw4 + lane16);
    const uint32_t  dmv = vegajet::ornith_dev::ldg_u32_stream(sdm + lane16);
    const uint4     xa  = sx4[lane16 * 2u];
    const uint4     xb  = sx4[lane16 * 2u + 1u];
    const float2    xds = sh_ds[lane16];
    int32_t dp = vegajet::ornith_dev::sdot4(wv.x & 0x0F0F0F0Fu, xa.x, 0);
    dp         = vegajet::ornith_dev::sdot4(wv.y & 0x0F0F0F0Fu, xa.y, dp);
    dp         = vegajet::ornith_dev::sdot4(wv.z & 0x0F0F0F0Fu, xa.z, dp);
    dp         = vegajet::ornith_dev::sdot4(wv.w & 0x0F0F0F0Fu, xa.w, dp);
    dp         = vegajet::ornith_dev::sdot4((wv.x >> 4) & 0x0F0F0F0Fu, xb.x, dp);
    dp         = vegajet::ornith_dev::sdot4((wv.y >> 4) & 0x0F0F0F0Fu, xb.y, dp);
    dp         = vegajet::ornith_dev::sdot4((wv.z >> 4) & 0x0F0F0F0Fu, xb.z, dp);
    dp         = vegajet::ornith_dev::sdot4((wv.w >> 4) & 0x0F0F0F0Fu, xb.w, dp);
    const float wd  = vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(dmv & 0xFFFFu));
    const float wm  = vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(dmv >> 16));
    const float shv = vegajet::dot8_dev::sum16(fmaf(wd * xds.x, static_cast<float>(dp), wm * xds.y));
    if (lane16 == 0u) {
        const float v = total + shv;
        out_f32[row] = v;
        sm_y[sub_id] = v;
    }
    if (peer_tx != nullptr) {
        __syncthreads();
        if (tid == 0u) {
            const __half2 h0 = __float22half2_rn(make_float2(sm_y[0], sm_y[1]));
            const __half2 h1 = __float22half2_rn(make_float2(sm_y[2], sm_y[3]));
            const __half2 h2 = __float22half2_rn(make_float2(sm_y[4], sm_y[5]));
            const __half2 h3 = __float22half2_rn(make_float2(sm_y[6], sm_y[7]));
            typedef uint32_t u32x4_t __attribute__((ext_vector_type(4)));
            u32x4_t pv = {
                *reinterpret_cast<const uint32_t*>(&h0),
                *reinterpret_cast<const uint32_t*>(&h1),
                *reinterpret_cast<const uint32_t*>(&h2),
                *reinterpret_cast<const uint32_t*>(&h3)
            };
            __builtin_nontemporal_store(pv, reinterpret_cast<u32x4_t*>(peer_tx + blockIdx.x * 8u));
        }
    }
}

// Experimental A/B: 0-LDS 512x128 IQ2_S Routed GU (reads 8 KiB iq2s_grid directly from 16 KiB L1D -> 0 B LDS, 10 WGs/CU!)
extern "C" __global__ __launch_bounds__(128, 5)
void ornith_moe_gu_iq2s_m1_8r_nolut_kernel(
    const uint4*   __restrict__ wgu_merged,
    const int8_t*  __restrict__ x_q8,
    const float2*  __restrict__ x_ds,
    const int32_t* __restrict__ top_ids,
    float*         __restrict__ mid_f32,
    uint32_t local_ff) {
    const uint32_t tid    = threadIdx.x;
    const uint32_t sub_id = tid >> 4;
    const uint32_t lane16 = tid & 15u;
    const uint32_t gidx   = blockIdx.x * 8u + sub_id;
    const uint32_t slot   = gidx / local_ff;
    const uint32_t rloc   = gidx - slot * local_ff;

    const uint32_t exp_id = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(top_ids[slot]));
    const uint4*   qp_p   = wgu_merged + (static_cast<uint64_t>(exp_id) * local_ff + rloc) * 123ULL;
    const uint4*   x4     = reinterpret_cast<const uint4*>(x_q8);
    const uint64_t* grid  = vegajet::ggml_tables::iq2s_grid;

    uint4    wq  = vegajet::ornith_dev::ldg_u32x4_stream(qp_p + lane16);
    uint32_t hs2 = vegajet::ornith_dev::ldg_u32_stream(reinterpret_cast<const uint32_t*>(qp_p + 96u) + lane16);
    uint32_t d2  = reinterpret_cast<const uint32_t*>(qp_p + 120u)[lane16 >> 3];

    float ag = 0.0f, au = 0.0f;
    #pragma nounroll
    for (uint32_t it = 0u; it < 6u; ++it) {
        const uint32_t ib = it * 16u + lane16;
        uint4    wq_n;
        uint32_t hs2_n, d2_n;
        if (it < 5u) {
            const uint32_t ibn = ib + 16u;
            wq_n  = vegajet::ornith_dev::ldg_u32x4_stream(qp_p + ibn);
            hs2_n = vegajet::ornith_dev::ldg_u32_stream(reinterpret_cast<const uint32_t*>(qp_p + 96u) + ibn);
            d2_n  = reinterpret_cast<const uint32_t*>(qp_p + 120u)[ibn >> 3];
        }
        const uint4 xa  = x4[ib * 2u];
        const uint4 xb  = x4[ib * 2u + 1u];
        const float xda = x_ds[ib].x;
        #pragma nounroll
        for (uint32_t part = 0u; part < 2u; ++part) {
            const uint32_t qa  = part ? wq.z : wq.x;
            const uint32_t qb  = part ? wq.w : wq.y;
            const uint32_t hsp = part ? (hs2 >> 16) : (hs2 & 0xFFFFu);
            const uint32_t qh  = hsp & 0xFFu, sc = hsp >> 8;
            const float pd = vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(part ? (d2 >> 16) : (d2 & 0xFFFFu))) * 0.25f * xda;
            float acc = part ? au : ag;
            uint64_t g0 = grid[(qa & 0xFFu)         | ((qh << 8) & 0x300u)];
            uint64_t g1 = grid[((qa >> 16) & 0xFFu) | ((qh << 6) & 0x300u)];
            int32_t dp0 = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g0),       (qa >> 8)  & 0xFu), xa.x, 0);
            dp0         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g0 >> 32), (qa >> 12) & 0xFu), xa.y, dp0);
            dp0         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g1),       (qa >> 24) & 0xFu), xa.z, dp0);
            dp0         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g1 >> 32), qa >> 28),          xa.w, dp0);
            acc = fmaf(pd * (0.5f + static_cast<float>(sc & 0xFu)), static_cast<float>(dp0), acc);
            uint64_t g2 = grid[(qb & 0xFFu)         | ((qh << 4) & 0x300u)];
            uint64_t g3 = grid[((qb >> 16) & 0xFFu) | ((qh << 2) & 0x300u)];
            int32_t dp1 = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g2),       (qb >> 8)  & 0xFu), xb.x, 0);
            dp1         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g2 >> 32), (qb >> 12) & 0xFu), xb.y, dp1);
            dp1         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g3),       (qb >> 24) & 0xFu), xb.z, dp1);
            dp1         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g3 >> 32), qb >> 28),          xb.w, dp1);
            acc = fmaf(pd * (0.5f + static_cast<float>(sc >> 4)), static_cast<float>(dp1), acc);
            if (part) au = acc; else ag = acc;
        }
        if (it < 5u) { wq = wq_n; hs2 = hs2_n; d2 = d2_n; }
    }
    ag = vegajet::dot8_dev::sum16(ag);
    au = vegajet::dot8_dev::sum16(au);
    if (lane16 == 0u) mid_f32[gidx] = (ag / (1.0f + __expf(-ag))) * au;
}


// Experimental A/B: 256x256 IQ2_S Routed GU (16 rows/WG, 2x fewer sm_grid copies, sm_xq + sm_xd in LDS!)
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_moe_gu_iq2s_m1_16r_kernel(
    const uint4*   __restrict__ wgu_merged,
    const int8_t*  __restrict__ x_q8,
    const float2*  __restrict__ x_ds,
    const int32_t* __restrict__ top_ids,
    float*         __restrict__ mid_f32,
    uint32_t local_ff) {
    __shared__ uint64_t sm_grid[1024];
    __shared__ uint4    sm_xq[192];
    __shared__ float    sm_xd[96];

    const uint32_t tid    = threadIdx.x;
    const uint32_t sub_id = tid >> 4;
    const uint32_t lane16 = tid & 15u;
    const uint32_t gidx   = blockIdx.x * 16u + sub_id;
    const uint32_t slot   = gidx / local_ff;
    const uint32_t rloc   = gidx - slot * local_ff;

    const uint32_t exp_id = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(top_ids[slot]));
    const uint4*   qp_p   = wgu_merged + (static_cast<uint64_t>(exp_id) * local_ff + rloc) * 123ULL;

    uint4    wq  = vegajet::ornith_dev::ldg_u32x4_stream(qp_p + lane16);
    uint32_t hs2 = vegajet::ornith_dev::ldg_u32_stream(reinterpret_cast<const uint32_t*>(qp_p + 96u) + lane16);
    uint32_t d2  = reinterpret_cast<const uint32_t*>(qp_p + 120u)[lane16 >> 3];

    reinterpret_cast<uint4*>(sm_grid)[tid]        = reinterpret_cast<const uint4*>(vegajet::ggml_tables::iq2s_grid)[tid];
    reinterpret_cast<uint4*>(sm_grid)[tid + 256u] = reinterpret_cast<const uint4*>(vegajet::ggml_tables::iq2s_grid)[tid + 256u];
    if (tid < 192u) sm_xq[tid] = reinterpret_cast<const uint4*>(x_q8)[tid];
    if (tid <  96u) sm_xd[tid] = x_ds[tid].x;
    __syncthreads();

    float ag = 0.0f, au = 0.0f;
    #pragma nounroll
    for (uint32_t it = 0u; it < 6u; ++it) {
        const uint32_t ib = it * 16u + lane16;
        uint4    wq_n;
        uint32_t hs2_n, d2_n;
        if (it < 5u) {
            const uint32_t ibn = ib + 16u;
            wq_n  = vegajet::ornith_dev::ldg_u32x4_stream(qp_p + ibn);
            hs2_n = vegajet::ornith_dev::ldg_u32_stream(reinterpret_cast<const uint32_t*>(qp_p + 96u) + ibn);
            d2_n  = reinterpret_cast<const uint32_t*>(qp_p + 120u)[ibn >> 3];
        }
        const uint4 xa  = sm_xq[ib * 2u];
        const uint4 xb  = sm_xq[ib * 2u + 1u];
        const float xda = sm_xd[ib];
        #pragma nounroll
        for (uint32_t part = 0u; part < 2u; ++part) {
            const uint32_t qa  = part ? wq.z : wq.x;
            const uint32_t qb  = part ? wq.w : wq.y;
            const uint32_t hsp = part ? (hs2 >> 16) : (hs2 & 0xFFFFu);
            const uint32_t qh  = hsp & 0xFFu, sc = hsp >> 8;
            const float pd = vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(part ? (d2 >> 16) : (d2 & 0xFFFFu))) * 0.25f * xda;
            float acc = part ? au : ag;
            uint64_t g0 = sm_grid[(qa & 0xFFu)         | ((qh << 8) & 0x300u)];
            uint64_t g1 = sm_grid[((qa >> 16) & 0xFFu) | ((qh << 6) & 0x300u)];
            int32_t dp0 = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g0),       (qa >> 8)  & 0xFu), xa.x, 0);
            dp0         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g0 >> 32), (qa >> 12) & 0xFu), xa.y, dp0);
            dp0         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g1),       (qa >> 24) & 0xFu), xa.z, dp0);
            dp0         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g1 >> 32), qa >> 28),          xa.w, dp0);
            acc = fmaf(pd * (0.5f + static_cast<float>(sc & 0xFu)), static_cast<float>(dp0), acc);
            uint64_t g2 = sm_grid[(qb & 0xFFu)         | ((qh << 4) & 0x300u)];
            uint64_t g3 = sm_grid[((qb >> 16) & 0xFFu) | ((qh << 2) & 0x300u)];
            int32_t dp1 = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g2),       (qb >> 8)  & 0xFu), xb.x, 0);
            dp1         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g2 >> 32), (qb >> 12) & 0xFu), xb.y, dp1);
            dp1         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g3),       (qb >> 24) & 0xFu), xb.z, dp1);
            dp1         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::apply_4signs_i8x4(static_cast<uint32_t>(g3 >> 32), qb >> 28),          xb.w, dp1);
            acc = fmaf(pd * (0.5f + static_cast<float>(sc >> 4)), static_cast<float>(dp1), acc);
            if (part) au = acc; else ag = acc;
        }
        if (it < 5u) { wq = wq_n; hs2 = hs2_n; d2 = d2_n; }
    }
    ag = vegajet::dot8_dev::sum16(ag);
    au = vegajet::dot8_dev::sum16(au);
    if (lane16 == 0u) mid_f32[gidx] = (ag / (1.0f + __expf(-ag))) * au;
}

// ============================================================================
// 9h. Wave 5 Compact 4-KiB iq2s_grid32 + v_perm_b32 Routed IQ2_S Gate+Up A/B!
//     Halves LDS footprint (4 KiB vs 8 KiB) & LDS bank traffic (ds_read_b32 vs ds_read_b64),
//     replaces 5-op apply_4signs_i8x4 with v_perm_b32. 100% bit-identical to 9g/s224!
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 5)
void ornith_moe_gu_iq2s_m1_16r_g32_kernel(
    const uint4*   __restrict__ wgu_merged,
    const int8_t*  __restrict__ x_q8,
    const float2*  __restrict__ x_ds,
    const int32_t* __restrict__ top_ids,
    float*         __restrict__ mid_f32,
    uint32_t local_ff) {
    __shared__ uint32_t sm_grid32[1024]; // 4,096 B (half of sm_grid!)
    __shared__ uint4    sm_xq[192];      // 3,072 B
    __shared__ float    sm_xd[96];       //   384 B  -> total 7,552 B (5+ WGs/CU!)

    const uint32_t tid    = threadIdx.x;
    const uint32_t sub_id = tid >> 4;
    const uint32_t lane16 = tid & 15u;
    const uint32_t gidx   = blockIdx.x * 16u + sub_id;
    const uint32_t slot   = gidx / local_ff;
    const uint32_t rloc   = gidx - slot * local_ff;

    const uint32_t exp_id = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(top_ids[slot]));
    const uint4*   qp_p   = wgu_merged + (static_cast<uint64_t>(exp_id) * local_ff + rloc) * 123ULL;

    uint4    wq  = vegajet::ornith_dev::ldg_u32x4_stream(qp_p + lane16);
    uint32_t hs2 = vegajet::ornith_dev::ldg_u32_stream(reinterpret_cast<const uint32_t*>(qp_p + 96u) + lane16);
    uint32_t d2  = reinterpret_cast<const uint32_t*>(qp_p + 120u)[lane16 >> 3];

    reinterpret_cast<uint4*>(sm_grid32)[tid] = reinterpret_cast<const uint4*>(vegajet::ggml_tables::iq2s_grid32)[tid];
    if (tid < 192u) sm_xq[tid] = reinterpret_cast<const uint4*>(x_q8)[tid];
    if (tid <  96u) sm_xd[tid] = x_ds[tid].x;
    __syncthreads();

    float ag = 0.0f, au = 0.0f;
    #pragma nounroll
    for (uint32_t it = 0u; it < 6u; ++it) {
        const uint32_t ib = it * 16u + lane16;
        uint4    wq_n;
        uint32_t hs2_n, d2_n;
        if (it < 5u) {
            const uint32_t ibn = ib + 16u;
            wq_n  = vegajet::ornith_dev::ldg_u32x4_stream(qp_p + ibn);
            hs2_n = vegajet::ornith_dev::ldg_u32_stream(reinterpret_cast<const uint32_t*>(qp_p + 96u) + ibn);
            d2_n  = reinterpret_cast<const uint32_t*>(qp_p + 120u)[ibn >> 3];
        }
        const uint4 xa  = sm_xq[ib * 2u];
        const uint4 xb  = sm_xq[ib * 2u + 1u];
        const float xda = sm_xd[ib];
        #pragma nounroll
        for (uint32_t part = 0u; part < 2u; ++part) {
            const uint32_t qa  = part ? wq.z : wq.x;
            const uint32_t qb  = part ? wq.w : wq.y;
            const uint32_t hsp = part ? (hs2 >> 16) : (hs2 & 0xFFFFu);
            const uint32_t qh  = hsp & 0xFFu, sc = hsp >> 8;
            const float pd = vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(part ? (d2 >> 16) : (d2 & 0xFFFFu))) * 0.25f * xda;
            float acc = part ? au : ag;
            const uint32_t g0 = sm_grid32[(qa & 0xFFu)         | ((qh << 8) & 0x300u)];
            const uint32_t g1 = sm_grid32[((qa >> 16) & 0xFFu) | ((qh << 6) & 0x300u)];
            int32_t dp0 = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::iq2s_perm4(g0,      (qa >> 8)  & 0xFu), xa.x, 0);
            dp0         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::iq2s_perm4(g0 >> 2, (qa >> 12) & 0xFu), xa.y, dp0);
            dp0         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::iq2s_perm4(g1,      (qa >> 24) & 0xFu), xa.z, dp0);
            dp0         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::iq2s_perm4(g1 >> 2, qa >> 28),           xa.w, dp0);
            acc = fmaf(pd * (0.5f + static_cast<float>(sc & 0xFu)), static_cast<float>(dp0), acc);

            const uint32_t g2 = sm_grid32[(qb & 0xFFu)         | ((qh << 4) & 0x300u)];
            const uint32_t g3 = sm_grid32[((qb >> 16) & 0xFFu) | ((qh << 2) & 0x300u)];
            int32_t dp1 = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::iq2s_perm4(g2,      (qb >> 8)  & 0xFu), xb.x, 0);
            dp1         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::iq2s_perm4(g2 >> 2, (qb >> 12) & 0xFu), xb.y, dp1);
            dp1         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::iq2s_perm4(g3,      (qb >> 24) & 0xFu), xb.z, dp1);
            dp1         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::iq2s_perm4(g3 >> 2, qb >> 28),           xb.w, dp1);
            acc = fmaf(pd * (0.5f + static_cast<float>(sc >> 4)), static_cast<float>(dp1), acc);
            if (part) au = acc; else ag = acc;
        }
        if (it < 5u) { wq = wq_n; hs2 = hs2_n; d2 = d2_n; }
    }
    ag = vegajet::dot8_dev::sum16(ag);
    au = vegajet::dot8_dev::sum16(au);
    if (lane16 == 0u) mid_f32[gidx] = (ag / (1.0f + __expf(-ag))) * au;
}

extern "C" __global__ __launch_bounds__(128, 5)
void ornith_moe_gu_iq2s_m1_8r_g32_kernel(
    const uint4*   __restrict__ wgu_merged,
    const int8_t*  __restrict__ x_q8,
    const float2*  __restrict__ x_ds,
    const int32_t* __restrict__ top_ids,
    float*         __restrict__ mid_f32,
    uint32_t local_ff) {
    __shared__ uint32_t sm_grid32[1024]; // 4,096 B -> 16 WGs/CU possible (10 waves/SIMD cap = 10 WGs/CU)!

    const uint32_t tid    = threadIdx.x;
    const uint32_t sub_id = tid >> 4;
    const uint32_t lane16 = tid & 15u;
    const uint32_t gidx   = blockIdx.x * 8u + sub_id;
    const uint32_t slot   = gidx / local_ff;
    const uint32_t rloc   = gidx - slot * local_ff;

    const uint32_t exp_id = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(top_ids[slot]));
    const uint4*   qp_p   = wgu_merged + (static_cast<uint64_t>(exp_id) * local_ff + rloc) * 123ULL;
    const uint4*   x4     = reinterpret_cast<const uint4*>(x_q8);

    uint4    wq  = vegajet::ornith_dev::ldg_u32x4_stream(qp_p + lane16);
    uint32_t hs2 = vegajet::ornith_dev::ldg_u32_stream(reinterpret_cast<const uint32_t*>(qp_p + 96u) + lane16);
    uint32_t d2  = reinterpret_cast<const uint32_t*>(qp_p + 120u)[lane16 >> 3];

    uint4* sm_g4 = reinterpret_cast<uint4*>(sm_grid32);
    const uint4* src_g4 = reinterpret_cast<const uint4*>(vegajet::ggml_tables::iq2s_grid32);
    sm_g4[tid]        = src_g4[tid];
    sm_g4[tid + 128u] = src_g4[tid + 128u];
    __syncthreads();

    float ag = 0.0f, au = 0.0f;
    #pragma nounroll
    for (uint32_t it = 0u; it < 6u; ++it) {
        const uint32_t ib = it * 16u + lane16;
        uint4    wq_n;
        uint32_t hs2_n, d2_n;
        if (it < 5u) {
            const uint32_t ibn = ib + 16u;
            wq_n  = vegajet::ornith_dev::ldg_u32x4_stream(qp_p + ibn);
            hs2_n = vegajet::ornith_dev::ldg_u32_stream(reinterpret_cast<const uint32_t*>(qp_p + 96u) + ibn);
            d2_n  = reinterpret_cast<const uint32_t*>(qp_p + 120u)[ibn >> 3];
        }
        const uint4 xa  = x4[ib * 2u];
        const uint4 xb  = x4[ib * 2u + 1u];
        const float xda = x_ds[ib].x;
        #pragma nounroll
        for (uint32_t part = 0u; part < 2u; ++part) {
            const uint32_t qa  = part ? wq.z : wq.x;
            const uint32_t qb  = part ? wq.w : wq.y;
            const uint32_t hsp = part ? (hs2 >> 16) : (hs2 & 0xFFFFu);
            const uint32_t qh  = hsp & 0xFFu, sc = hsp >> 8;
            const float pd = vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(part ? (d2 >> 16) : (d2 & 0xFFFFu))) * 0.25f * xda;
            float acc = part ? au : ag;
            const uint32_t g0 = sm_grid32[(qa & 0xFFu)         | ((qh << 8) & 0x300u)];
            const uint32_t g1 = sm_grid32[((qa >> 16) & 0xFFu) | ((qh << 6) & 0x300u)];
            int32_t dp0 = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::iq2s_perm4(g0,      (qa >> 8)  & 0xFu), xa.x, 0);
            dp0         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::iq2s_perm4(g0 >> 2, (qa >> 12) & 0xFu), xa.y, dp0);
            dp0         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::iq2s_perm4(g1,      (qa >> 24) & 0xFu), xa.z, dp0);
            dp0         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::iq2s_perm4(g1 >> 2, qa >> 28),           xa.w, dp0);
            acc = fmaf(pd * (0.5f + static_cast<float>(sc & 0xFu)), static_cast<float>(dp0), acc);

            const uint32_t g2 = sm_grid32[(qb & 0xFFu)         | ((qh << 4) & 0x300u)];
            const uint32_t g3 = sm_grid32[((qb >> 16) & 0xFFu) | ((qh << 2) & 0x300u)];
            int32_t dp1 = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::iq2s_perm4(g2,      (qb >> 8)  & 0xFu), xb.x, 0);
            dp1         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::iq2s_perm4(g2 >> 2, (qb >> 12) & 0xFu), xb.y, dp1);
            dp1         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::iq2s_perm4(g3,      (qb >> 24) & 0xFu), xb.z, dp1);
            dp1         = vegajet::ornith_dev::sdot4(vegajet::ornith_dev::iq2s_perm4(g3 >> 2, qb >> 28),           xb.w, dp1);
            acc = fmaf(pd * (0.5f + static_cast<float>(sc >> 4)), static_cast<float>(dp1), acc);
            if (part) au = acc; else ag = acc;
        }
        if (it < 5u) { wq = wq_n; hs2 = hs2_n; d2 = d2_n; }
    }
    ag = vegajet::dot8_dev::sum16(ag);
    au = vegajet::dot8_dev::sum16(au);
    if (lane16 == 0u) mid_f32[gidx] = (ag / (1.0f + __expf(-ag))) * au;
}
