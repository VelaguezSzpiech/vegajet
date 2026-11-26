// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Toggle-only, 32-column-granular expert TP. Legacy kernels remain unchanged.
#include "ops/kernels/vj_ornith_common.hiph"
#include "ops/kernels/vj_moe_dot8.hiph"

__device__ __forceinline__ float asym_down_row(
    const uint8_t* w, uint32_t wst, const uint8_t* s, uint32_t sst,
    const uint4* mq, const float* md, const int32_t* ids, const float* weights,
    uint32_t row, uint32_t D, uint32_t lane, uint32_t lff) {
    const uint32_t nb = lff / 32u;
    float total = 0.0f;
    #pragma nounroll
    for (uint32_t slot = 0; slot < 8u; ++slot) {
        const uint64_t er = static_cast<uint64_t>(ids[slot]) * D + row;
        #pragma nounroll
        for (uint32_t b = lane; b < nb; b += 16u) {
            const uint4 wv = *reinterpret_cast<const uint4*>(w + er * wst + b * 16u);
            const uint2 sc = *reinterpret_cast<const uint2*>(s + er * sst + b * 8u);
            const int32_t dp = vegajet::dot8_dev::dot32_iq3(wv, mq[slot * nb * 2u + b * 2u],
                                                         mq[slot * nb * 2u + b * 2u + 1u]);
            const float gdb = __uint_as_float(sc.y) * (0.5f + static_cast<float>(sc.x)) * 0.5f;
            total = fmaf(weights[slot] * gdb * md[slot * nb + b], static_cast<float>(dp), total);
        }
    }
    return vegajet::dot8_dev::sum16(total);
}

extern "C" __global__ __launch_bounds__(256, 4)
void vj_moe_m1_back_asym_s224_kernel(
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
    __shared__ uint4    sm_x[512];       // up to 8 KiB of routed mid_q8
    __shared__ float    sm_xd[256];      // up to 1 KiB of mid scales
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
    reinterpret_cast<uint4*>(sm_grid32)[tid] = reinterpret_cast<const uint4*>(vegajet::ggml_tables::iq2s_grid32)[tid];
    __syncthreads();

    // Static strided work assignment keeps all 224 WGs resident at the grid barrier.
    const uint32_t nblk = local_ff / 32u;
    const uint32_t n_gu = 8u * nblk;
    for (uint32_t task = wg; task < n_gu + 96u; task += 224u) {
    if (task < n_gu) {
        const uint32_t slot   = task / nblk;
        const uint32_t rblk   = task % nblk;
        const uint32_t exp_id = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(sm_ids[slot]));
        const uint32_t sub_id = tid >> 4;

        #pragma nounroll
        for (uint32_t half = 0u; half < 2u; ++half) {
            const uint32_t row  = rblk * 32u + half * 16u + sub_id;
            const uint64_t erow = static_cast<uint64_t>(exp_id) * local_ff + row;
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
            mid_q8[static_cast<uint64_t>(slot) * local_ff + rblk * 32u + tid] =
                static_cast<int8_t>(max(-127, min(127, __float2int_rn(y * inv))));
            if (tid == 0u) mid_d[static_cast<uint64_t>(slot) * nblk + rblk] = d;
        }
    } else {
        const uint4* x4 = reinterpret_cast<const uint4*>(sh_q8);
        #pragma nounroll
        for (uint32_t step = 0u; step < 2u; ++step) {
            const uint32_t row = ((task - n_gu) * 2u + step) * 16u + (tid >> 4);
            const uint4*    w4 = reinterpret_cast<const uint4*>(sh_wqs + static_cast<uint64_t>(row) * sh_stride);
            const uint32_t* dm = sh_wdm + static_cast<uint64_t>(row) * nblk;
            float acc = 0.0f;
            for (uint32_t b = lane16; b < nblk; b += 16u) {
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
            acc += fmaf(wd * xds.x, static_cast<float>(dp), wm * xds.y);
            }
            acc = vegajet::dot8_dev::sum16(acc);
            if (lane16 == 0u) sh_down[row] = acc;
        }
    }

    __syncthreads();
    }

    __threadfence();
    vegajet::ornith_dev::grid_barrier_m(q, 224u, sense);
    if (tid == 0u) asm volatile("buffer_wbinvl1\n" ::: "memory");
    __syncthreads();

    // ---- Stage B: 192 WGs Routed Down + sh_down + FP16 BAR0 prepush ----
    if (wg >= 192u) return;
    for (uint32_t i = tid; i < local_ff / 2u; i += 256u)
        sm_x[i] = reinterpret_cast<const uint4*>(mid_q8)[i];
    for (uint32_t i = tid; i < local_ff / 4u; i += 256u) sm_xd[i] = mid_d[i];
    __syncthreads();

    const uint32_t grp16 = tid >> 4;
    const uint32_t row   = wg * 16u + grp16;
    const float total = asym_down_row(wdn_d8, dn_w_stride, wdn_d8_sc, dn_s_stride,
                                                                  sm_x, sm_xd, sm_ids, sm_w, row, D, lane16, local_ff);
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

extern "C" __global__ __launch_bounds__(256, 4)
void vj_moe_down_asym_kernel(
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
    uint32_t local_ff,
    uint16_t*      __restrict__ peer_tx) {
    __shared__ uint4   sm_mq[512];
    __shared__ float   sm_md[256];
    __shared__ int32_t sm_ids[8];
    __shared__ float   sm_w[8];
    __shared__ float   sm_y[16];

    const uint32_t m      = blockIdx.y;
    const uint32_t grp16  = threadIdx.x >> 4;
    const uint32_t lane16 = threadIdx.x & 15u;
    const uint32_t row    = blockIdx.x * 16u + grp16;
    if (m >= M) return;

    for (uint32_t i = threadIdx.x; i < local_ff / 2u; i += 256u)
        sm_mq[i] = reinterpret_cast<const uint4*>(mid_q8 + static_cast<uint64_t>(m) * 8u * local_ff)[i];
    for (uint32_t i = threadIdx.x; i < local_ff / 4u; i += 256u)
        sm_md[i] = mid_d[static_cast<uint64_t>(m) * (local_ff / 4u) + i];
    if (threadIdx.x <   8u) {
        sm_ids[threadIdx.x] = top_ids[static_cast<uint64_t>(m) * 8u + threadIdx.x];
        sm_w[threadIdx.x]   = top_w[static_cast<uint64_t>(m) * 8u + threadIdx.x];
    }
    __syncthreads();

    if (row < D) {
        // Speculative verify (M = 2..16): L2-cached loads reuse experts hit by multiple tokens in the batch (bit-identical).
        const float total = asym_down_row(w_qs, w_stride, w_sc, s_stride,
                                                                            sm_mq, sm_md, sm_ids, sm_w, row, D, lane16, local_ff);
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

extern "C" __global__ __launch_bounds__(256, 4)
void vj_moe_down_asym_q4a8_kernel(
    const uint8_t*  __restrict__ wd_qs,    // [256 * D, 256] (cols=512 -> 256 B/row!)
    const uint32_t* __restrict__ wd_dm,    // [256 * D, 16]
    const int8_t*   __restrict__ mid_q8,   // [M, 8, 512]
    const float2*   __restrict__ mid_ds,   // [M, 8, 16]
    const int32_t*  __restrict__ top_ids,  // [M, 8]
    const float*    __restrict__ top_w,    // [M, 8]
    const float*    __restrict__ sh_down_in,
    float*          __restrict__ out_f32,
    uint32_t M,
    uint32_t D,
    uint32_t stride_b,
    uint32_t local_ff) {
    const uint32_t nb = local_ff / 32u;
    __shared__ uint4   sm_mq[512];
    __shared__ float2  sm_mds[256];
    __shared__ int32_t sm_ids[8];
    __shared__ float   sm_w[8];

    const uint32_t m      = blockIdx.y;
    const uint32_t sub_r  = threadIdx.x >> 4; // 0..15 (16 rows/WG)
    const uint32_t lane16 = threadIdx.x & 15u;
    const uint32_t row    = blockIdx.x * 16u + sub_r;
    if (m >= M) return;

    for (uint32_t i = threadIdx.x; i < local_ff / 2u; i += 256u)
        sm_mq[i] = reinterpret_cast<const uint4*>(mid_q8 + static_cast<uint64_t>(m) * 8u * local_ff)[i];
    for (uint32_t i = threadIdx.x; i < local_ff / 4u; i += 256u)
        sm_mds[i] = mid_ds[static_cast<uint64_t>(m) * (local_ff / 4u) + i];
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
        const uint64_t erow   = static_cast<uint64_t>(exp_id) * D + row;
        const uint4*    w4    = reinterpret_cast<const uint4*>(wd_qs + erow * stride_b);
        const uint32_t* dm    = wd_dm + erow * nb;

        for (uint32_t b = lane16; b < nb; b += 16u) {
        const uint4    wv  = vegajet::ornith_dev::ldg_u32x4_stream(w4 + b);
        const uint32_t dmv = vegajet::ornith_dev::ldg_u32_stream(dm + b);
        const uint4    xa  = sm_mq[slot * nb * 2u + b * 2u + 0u];
        const uint4    xb  = sm_mq[slot * nb * 2u + b * 2u + 1u];
        const float2   xds = sm_mds[slot * nb + b];

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
