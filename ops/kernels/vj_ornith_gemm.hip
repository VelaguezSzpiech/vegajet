// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "ops/kernels/vj_ornith_common.hiph"

// ============================================================================
// 4a. ULTRA-FAST Asymmetric VJ_Q4_A8 M=1 Streaming GEMV (32 threads/row = 8 rows/WG!)
//     Zero wave-tail waste for cols=3072 (3x32) & cols=4096 (4x32), TCC odd-256B anti-camping!
//     w_qs: uint4 = 32 u4 weights (low nibbles=0..15, high nibbles=16..31), 0 sign-ext, v_dot4_i32_i8!
//     w_dm: uint32_t = fp16(d) | (fp16(m) << 16) per 32-element block
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 5)
void ornith_gemv_q4a8_m1_w32_kernel(
    const uint8_t*  __restrict__ w_qs,
    const uint32_t* __restrict__ w_dm,
    const int8_t*   __restrict__ x_q8,
    const float2*   __restrict__ x_ds,
    float*          __restrict__ y_out,
    uint32_t rows,
    uint32_t cols,
    uint32_t row_stride_bytes,
    uint32_t dm_stride_u32,
    uint16_t*       __restrict__ peer_tx) {
    const uint32_t grp  = threadIdx.x >> 5;  // 0..7 (8 rows/WG!)
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row  = blockIdx.x * 8u + grp;
    if (row >= rows) return;

    const uint32_t  nb = cols >> 5; // 96 for 3072, 128 for 4096
    const uint32_t  dm_st = dm_stride_u32 ? dm_stride_u32 : nb;
    const uint4*    w4 = reinterpret_cast<const uint4*>(w_qs + static_cast<uint64_t>(row) * row_stride_bytes);
    const uint32_t* dm = w_dm + static_cast<uint64_t>(row) * dm_st;
    const uint4*    x4 = reinterpret_cast<const uint4*>(x_q8);

    // Weight loads are requested in batches before consumption to maximize HBM2 MLP while preserving exact
    // ascending block accumulation order. Uniform fast-path for nb == 128 (cols == 4096: w_ssm_out, w_o)
    // runs 2 exact rounds of 2 blocks (0 clamped tail re-loads, 0 b<nb branch, 33% fewer HBM2 requests!).
    float acc = 0.0f;
    if (nb == 128u) {
        #pragma unroll
        for (uint32_t r = 0u; r < 2u; ++r) {
            const uint32_t b0 = lane + r * 64u;
            const uint32_t b1 = b0 + 32u;
            const uint4    wv0 = vegajet::ornith_dev::ldg_u32x4_stream(w4 + b0);
            const uint32_t dm0 = vegajet::ornith_dev::ldg_u32_stream(dm + b0);
            const uint4    wv1 = vegajet::ornith_dev::ldg_u32x4_stream(w4 + b1);
            const uint32_t dm1 = vegajet::ornith_dev::ldg_u32_stream(dm + b1);
            {
                const uint4  xa = x4[b0 * 2u], xb = x4[b0 * 2u + 1u];
                const float2 xds = x_ds[b0];
                int32_t dp = vegajet::ornith_dev::sdot4(wv0.x & 0x0F0F0F0Fu, xa.x, 0);
                dp         = vegajet::ornith_dev::sdot4(wv0.y & 0x0F0F0F0Fu, xa.y, dp);
                dp         = vegajet::ornith_dev::sdot4(wv0.z & 0x0F0F0F0Fu, xa.z, dp);
                dp         = vegajet::ornith_dev::sdot4(wv0.w & 0x0F0F0F0Fu, xa.w, dp);
                dp         = vegajet::ornith_dev::sdot4((wv0.x >> 4) & 0x0F0F0F0Fu, xb.x, dp);
                dp         = vegajet::ornith_dev::sdot4((wv0.y >> 4) & 0x0F0F0F0Fu, xb.y, dp);
                dp         = vegajet::ornith_dev::sdot4((wv0.z >> 4) & 0x0F0F0F0Fu, xb.z, dp);
                dp         = vegajet::ornith_dev::sdot4((wv0.w >> 4) & 0x0F0F0F0Fu, xb.w, dp);
                const float wd = vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(dm0 & 0xFFFFu));
                const float wm = vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(dm0 >> 16));
                acc = fmaf(wd * xds.x, static_cast<float>(dp), fmaf(wm, xds.y, acc));
            }
            {
                const uint4  xa = x4[b1 * 2u], xb = x4[b1 * 2u + 1u];
                const float2 xds = x_ds[b1];
                int32_t dp = vegajet::ornith_dev::sdot4(wv1.x & 0x0F0F0F0Fu, xa.x, 0);
                dp         = vegajet::ornith_dev::sdot4(wv1.y & 0x0F0F0F0Fu, xa.y, dp);
                dp         = vegajet::ornith_dev::sdot4(wv1.z & 0x0F0F0F0Fu, xa.z, dp);
                dp         = vegajet::ornith_dev::sdot4(wv1.w & 0x0F0F0F0Fu, xa.w, dp);
                dp         = vegajet::ornith_dev::sdot4((wv1.x >> 4) & 0x0F0F0F0Fu, xb.x, dp);
                dp         = vegajet::ornith_dev::sdot4((wv1.y >> 4) & 0x0F0F0F0Fu, xb.y, dp);
                dp         = vegajet::ornith_dev::sdot4((wv1.z >> 4) & 0x0F0F0F0Fu, xb.z, dp);
                dp         = vegajet::ornith_dev::sdot4((wv1.w >> 4) & 0x0F0F0F0Fu, xb.w, dp);
                const float wd = vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(dm1 & 0xFFFFu));
                const float wm = vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(dm1 >> 16));
                acc = fmaf(wd * xds.x, static_cast<float>(dp), fmaf(wm, xds.y, acc));
            }
        }
    } else {
    const uint32_t b_last = nb - 1u;
    #pragma nounroll
    for (uint32_t b0 = lane; b0 < nb; b0 += 96u) {
        uint4    wvs[3];
        uint32_t dms[3];
        #pragma unroll
        for (uint32_t k = 0u; k < 3u; ++k) {
            const uint32_t bc = min(b0 + k * 32u, b_last); // out-of-row blocks re-read the last one and are skipped below
            wvs[k] = vegajet::ornith_dev::ldg_u32x4_stream(w4 + bc);
            dms[k] = vegajet::ornith_dev::ldg_u32_stream(dm + bc);
        }
        #pragma unroll
        for (uint32_t k = 0u; k < 3u; ++k) {
            const uint32_t b = b0 + k * 32u;
            if (b < nb) {
                const uint4    wv  = wvs[k];
                const uint32_t dmv = dms[k];
                const uint4    xa  = x4[b * 2u];
                const uint4    xb  = x4[b * 2u + 1u];
                const float2   xds = x_ds[b];

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
    }
    acc = vegajet::ornith_dev::block32_sum(acc);
    if (lane == 0u) y_out[row] = acc;
    if (peer_tx != nullptr && (rows & 7u) == 0u) {
        __shared__ float sm_out[8];
        if (lane == 0u) sm_out[grp] = acc;
        __syncthreads();
        if (threadIdx.x == 0u) {
            const __half2 h0 = __float22half2_rn(make_float2(sm_out[0], sm_out[1]));
            const __half2 h1 = __float22half2_rn(make_float2(sm_out[2], sm_out[3]));
            const __half2 h2 = __float22half2_rn(make_float2(sm_out[4], sm_out[5]));
            const __half2 h3 = __float22half2_rn(make_float2(sm_out[6], sm_out[7]));
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

// ============================================================================
// v1.2 D13: persistent work-stealing grid for the same dense M=1 GEMV
// (VEGAJET_V12_DENSE_PERSIST, default off).
//     (rows+7)/8 WGs of 256 threads overshoot the 300 resident slots (rows=3072 -> 384 WGs =
//     1.28 waves), so the tail runs with a third of the CUs issuing. This variant launches a
//     fixed resident grid (224-240 WGs) and lets each WG pull 8-row tiles from an atomic
//     counter, exactly like the M=1 MoE back path. Per-tile math, per-row accumulation order
//     and the peer_tx layout (8 halves starting at row 8*tile) are identical to the
//     blockIdx.x variant, so tokens do not change.
//     q[0] = tile counter, q[1] = WG arrivals. The last WG out zeroes both, so the host never
//     has to memset per dispatch; the buffer only has to be zeroed once at setup.
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 5)
void ornith_gemv_q4a8_m1_w32_persist_kernel(
    const uint8_t*  __restrict__ w_qs,
    const uint32_t* __restrict__ w_dm,
    const int8_t*   __restrict__ x_q8,
    const float2*   __restrict__ x_ds,
    float*          __restrict__ y_out,
    uint32_t rows,
    uint32_t cols,
    uint32_t row_stride_bytes,
    uint32_t dm_stride_u32,
    uint16_t*       __restrict__ peer_tx,
    uint32_t*       __restrict__ q,
    uint32_t n_tiles) {
    const uint32_t grp  = threadIdx.x >> 5;  // 0..7 (8 rows/WG)
    const uint32_t lane = threadIdx.x & 31u;

    const uint32_t  nb    = cols >> 5;
    const uint32_t  dm_st = dm_stride_u32 ? dm_stride_u32 : nb;
    const uint4*    x4    = reinterpret_cast<const uint4*>(x_q8);
    const uint32_t  b_last = nb - 1u;

    __shared__ uint32_t sm_tile;
    __shared__ float    sm_out[8];

    for (;;) {
        if (threadIdx.x == 0u) sm_tile = atomicAdd(&q[0], 1u);
        __syncthreads();
        const uint32_t t = sm_tile;
        __syncthreads();                     // sm_tile is free again for the next pull
        if (t >= n_tiles) break;             // uniform across the WG
        const uint32_t row = t * 8u + grp;

        float acc = 0.0f;
        if (row < rows) {
            const uint4*    w4 = reinterpret_cast<const uint4*>(w_qs + static_cast<uint64_t>(row) * row_stride_bytes);
            const uint32_t* dm = w_dm + static_cast<uint64_t>(row) * dm_st;
            // Same two paths, same accumulation order as ornith_gemv_q4a8_m1_w32_kernel.
            if (nb == 128u) {
                #pragma unroll
                for (uint32_t r = 0u; r < 2u; ++r) {
                    const uint32_t b0 = lane + r * 64u;
                    const uint32_t b1 = b0 + 32u;
                    const uint4    wv0 = vegajet::ornith_dev::ldg_u32x4_stream(w4 + b0);
                    const uint32_t dm0 = vegajet::ornith_dev::ldg_u32_stream(dm + b0);
                    const uint4    wv1 = vegajet::ornith_dev::ldg_u32x4_stream(w4 + b1);
                    const uint32_t dm1 = vegajet::ornith_dev::ldg_u32_stream(dm + b1);
                    {
                        const uint4  xa = x4[b0 * 2u], xb = x4[b0 * 2u + 1u];
                        const float2 xds = x_ds[b0];
                        int32_t dp = vegajet::ornith_dev::sdot4(wv0.x & 0x0F0F0F0Fu, xa.x, 0);
                        dp         = vegajet::ornith_dev::sdot4(wv0.y & 0x0F0F0F0Fu, xa.y, dp);
                        dp         = vegajet::ornith_dev::sdot4(wv0.z & 0x0F0F0F0Fu, xa.z, dp);
                        dp         = vegajet::ornith_dev::sdot4(wv0.w & 0x0F0F0F0Fu, xa.w, dp);
                        dp         = vegajet::ornith_dev::sdot4((wv0.x >> 4) & 0x0F0F0F0Fu, xb.x, dp);
                        dp         = vegajet::ornith_dev::sdot4((wv0.y >> 4) & 0x0F0F0F0Fu, xb.y, dp);
                        dp         = vegajet::ornith_dev::sdot4((wv0.z >> 4) & 0x0F0F0F0Fu, xb.z, dp);
                        dp         = vegajet::ornith_dev::sdot4((wv0.w >> 4) & 0x0F0F0F0Fu, xb.w, dp);
                        const float wd = vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(dm0 & 0xFFFFu));
                        const float wm = vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(dm0 >> 16));
                        acc = fmaf(wd * xds.x, static_cast<float>(dp), fmaf(wm, xds.y, acc));
                    }
                    {
                        const uint4  xa = x4[b1 * 2u], xb = x4[b1 * 2u + 1u];
                        const float2 xds = x_ds[b1];
                        int32_t dp = vegajet::ornith_dev::sdot4(wv1.x & 0x0F0F0F0Fu, xa.x, 0);
                        dp         = vegajet::ornith_dev::sdot4(wv1.y & 0x0F0F0F0Fu, xa.y, dp);
                        dp         = vegajet::ornith_dev::sdot4(wv1.z & 0x0F0F0F0Fu, xa.z, dp);
                        dp         = vegajet::ornith_dev::sdot4(wv1.w & 0x0F0F0F0Fu, xa.w, dp);
                        dp         = vegajet::ornith_dev::sdot4((wv1.x >> 4) & 0x0F0F0F0Fu, xb.x, dp);
                        dp         = vegajet::ornith_dev::sdot4((wv1.y >> 4) & 0x0F0F0F0Fu, xb.y, dp);
                        dp         = vegajet::ornith_dev::sdot4((wv1.z >> 4) & 0x0F0F0F0Fu, xb.z, dp);
                        dp         = vegajet::ornith_dev::sdot4((wv1.w >> 4) & 0x0F0F0F0Fu, xb.w, dp);
                        const float wd = vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(dm1 & 0xFFFFu));
                        const float wm = vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(dm1 >> 16));
                        acc = fmaf(wd * xds.x, static_cast<float>(dp), fmaf(wm, xds.y, acc));
                    }
                }
            } else {
                #pragma nounroll
                for (uint32_t b0 = lane; b0 < nb; b0 += 96u) {
                    uint4    wvs[3];
                    uint32_t dms[3];
                    #pragma unroll
                    for (uint32_t k = 0u; k < 3u; ++k) {
                        const uint32_t bc = min(b0 + k * 32u, b_last); // out-of-row blocks re-read the last one and are skipped below
                        wvs[k] = vegajet::ornith_dev::ldg_u32x4_stream(w4 + bc);
                        dms[k] = vegajet::ornith_dev::ldg_u32_stream(dm + bc);
                    }
                    #pragma unroll
                    for (uint32_t k = 0u; k < 3u; ++k) {
                        const uint32_t b = b0 + k * 32u;
                        if (b < nb) {
                            const uint4    wv  = wvs[k];
                            const uint32_t dmv = dms[k];
                            const uint4    xa  = x4[b * 2u];
                            const uint4    xb  = x4[b * 2u + 1u];
                            const float2   xds = x_ds[b];

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
            }
            acc = vegajet::ornith_dev::block32_sum(acc);
            if (lane == 0u) y_out[row] = acc;
        }

        // peer_tx layout is per tile (rows 8*t .. 8*t+7), so tiles may be claimed in any order.
        if (peer_tx != nullptr && (rows & 7u) == 0u) {
            if (lane == 0u) sm_out[grp] = acc;
            __syncthreads();
            if (threadIdx.x == 0u) {
                const __half2 h0 = __float22half2_rn(make_float2(sm_out[0], sm_out[1]));
                const __half2 h1 = __float22half2_rn(make_float2(sm_out[2], sm_out[3]));
                const __half2 h2 = __float22half2_rn(make_float2(sm_out[4], sm_out[5]));
                const __half2 h3 = __float22half2_rn(make_float2(sm_out[6], sm_out[7]));
                typedef uint32_t u32x4_t __attribute__((ext_vector_type(4)));
                u32x4_t pv = {
                    *reinterpret_cast<const uint32_t*>(&h0),
                    *reinterpret_cast<const uint32_t*>(&h1),
                    *reinterpret_cast<const uint32_t*>(&h2),
                    *reinterpret_cast<const uint32_t*>(&h3)
                };
                __builtin_nontemporal_store(pv, reinterpret_cast<u32x4_t*>(peer_tx + t * 8u));
            }
            __syncthreads();                 // sm_out is reused by the next tile
        }
    }

    // Last WG out resets both counters: the next dispatch starts from an empty queue with no host memset.
    if (threadIdx.x == 0u) {
        const uint32_t arr = atomicAdd(&q[1], 1u) + 1u;
        if (arr == gridDim.x) {
            __atomic_store_n(&q[0], 0u, __ATOMIC_RELEASE);
            __atomic_store_n(&q[1], 0u, __ATOMIC_RELEASE);
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 6)
void ornith_gemv_q4a8_m1_w16_kernel(
    const uint8_t*  __restrict__ w_qs,
    const uint32_t* __restrict__ w_dm,
    const int8_t*   __restrict__ x_q8,
    const float2*   __restrict__ x_ds,
    float*          __restrict__ y_out,
    uint32_t rows,
    uint32_t cols,
    uint32_t row_stride_bytes) {
    const uint32_t grp  = threadIdx.x >> 4;  // 0..15 (16 rows/WG!)
    const uint32_t lane = threadIdx.x & 15u;
    const uint32_t row  = blockIdx.x * 16u + grp;
    if (row >= rows) return;

    const uint32_t  nb = cols >> 5; // 16 for 512
    const uint4*    w4 = reinterpret_cast<const uint4*>(w_qs + static_cast<uint64_t>(row) * row_stride_bytes);
    const uint32_t* dm = w_dm + static_cast<uint64_t>(row) * nb;
    const uint4*    x4 = reinterpret_cast<const uint4*>(x_q8);

    float acc = 0.0f;
    #pragma nounroll
    for (uint32_t b = lane; b < nb; b += 16u) {
        const uint4    wv  = vegajet::ornith_dev::ldg_u32x4_stream(w4 + b);
        const uint32_t dmv = vegajet::ornith_dev::ldg_u32_stream(dm + b);
        const uint4    xa  = x4[b * 2u];
        const uint4    xb  = x4[b * 2u + 1u];
        const float2   xds = x_ds[b];

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
    if (lane == 0u) y_out[row] = acc;
}

// ============================================================================
// 4c. LDS-Cached 16-Token Register-Tiled Batched VJ_Q4_A8 Prefill GEMM (M=2..64!)
//     Cooperatively caches x_q8[16][32 blk] (16 KiB) + x_ds (4 KiB) in LDS -> 8x fewer x loads!
//     32 threads/row (8 rows/WG) x up to 16 tokens/WG -> 16x weight BW reuse on prefill!
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 2)
void ornith_gemm_q4a8_tcc_kernel(
    const uint8_t*  __restrict__ w_qs,
    const uint32_t* __restrict__ w_dm,
    const int8_t*   __restrict__ x_q8,
    const float2*   __restrict__ x_ds,
    float*          __restrict__ y_out,
    uint32_t rows,
    uint32_t cols,
    uint32_t M,
    uint32_t row_stride_bytes) {
    const uint32_t tid  = threadIdx.x;
    const uint32_t grp  = tid >> 5; // 0..7
    const uint32_t lane = tid & 31u;
    const uint32_t row  = blockIdx.x * 8u + grp;
    const uint32_t m0   = blockIdx.y * 16u;
    if (m0 >= M) return;

    __shared__ uint4  sm_x[16 * 64];  // 16 KiB: 16 tokens x 32 blocks x 2 uint4s
    __shared__ float2 sm_xds[16 * 32]; // 4 KiB: 16 tokens x 32 float2s

    const uint32_t  m_cnt = min(16u, M - m0);
    const uint32_t  nb    = cols >> 5;
    const bool      valid_row = (row < rows);
    const uint4*    w4    = valid_row ? reinterpret_cast<const uint4*>(w_qs + static_cast<uint64_t>(row) * row_stride_bytes) : nullptr;
    const uint32_t* dm    = valid_row ? (w_dm + static_cast<uint64_t>(row) * nb) : nullptr;

    float a0=0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0, a7=0;
    float a8=0, a9=0, a10=0, a11=0, a12=0, a13=0, a14=0, a15=0;

    #pragma nounroll
    for (uint32_t b_base = 0u; b_base < nb; b_base += 32u) {
        const uint32_t tile_nb = min(32u, nb - b_base);
        // Cooperatively load x_q8[m0..m0+m_cnt-1][b_base..b_base+tile_nb-1] into LDS
        #pragma unroll
        for (uint32_t idx = tid; idx < m_cnt * 64u; idx += 256u) {
            const uint32_t dm_i = idx >> 6;
            const uint32_t sub  = idx & 63u;
            if ((sub >> 1) < tile_nb) {
                const uint4* gx4 = reinterpret_cast<const uint4*>(x_q8 + static_cast<uint64_t>(m0 + dm_i) * cols);
                sm_x[idx] = gx4[b_base * 2u + sub];
            }
        }
        #pragma unroll
        for (uint32_t idx = tid; idx < m_cnt * 32u; idx += 256u) {
            const uint32_t dm_i = idx >> 5;
            const uint32_t sub  = idx & 31u;
            if (sub < tile_nb) {
                sm_xds[idx] = x_ds[static_cast<uint64_t>(m0 + dm_i) * nb + b_base + sub];
            }
        }
        __syncthreads();

        if (valid_row && lane < tile_nb) {
            const uint32_t b   = b_base + lane;
            const uint4    wv  = w4[b];
            const uint32_t dmv = dm[b];
            const float    wd  = vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(dmv & 0xFFFFu));
            const float    wm  = vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(dmv >> 16));
            const uint32_t wl0 = wv.x & 0x0F0F0F0Fu, wh0 = (wv.x >> 4) & 0x0F0F0F0Fu;
            const uint32_t wl1 = wv.y & 0x0F0F0F0Fu, wh1 = (wv.y >> 4) & 0x0F0F0F0Fu;
            const uint32_t wl2 = wv.z & 0x0F0F0F0Fu, wh2 = (wv.z >> 4) & 0x0F0F0F0Fu;
            const uint32_t wl3 = wv.w & 0x0F0F0F0Fu, wh3 = (wv.w >> 4) & 0x0F0F0F0Fu;

            auto mac = [&](uint32_t dm_i, float& acc) {
                const uint4  xa  = sm_x[dm_i * 64u + lane * 2u];
                const uint4  xb  = sm_x[dm_i * 64u + lane * 2u + 1u];
                const float2 xds = sm_xds[dm_i * 32u + lane];
                int32_t dp = vegajet::ornith_dev::sdot4(wl0, xa.x, 0);
                dp         = vegajet::ornith_dev::sdot4(wl1, xa.y, dp);
                dp         = vegajet::ornith_dev::sdot4(wl2, xa.z, dp);
                dp         = vegajet::ornith_dev::sdot4(wl3, xa.w, dp);
                dp         = vegajet::ornith_dev::sdot4(wh0, xb.x, dp);
                dp         = vegajet::ornith_dev::sdot4(wh1, xb.y, dp);
                dp         = vegajet::ornith_dev::sdot4(wh2, xb.z, dp);
                dp         = vegajet::ornith_dev::sdot4(wh3, xb.w, dp);
                acc = fmaf(wd * xds.x, static_cast<float>(dp), fmaf(wm, xds.y, acc));
            };
            mac(0u, a0);
            if (m_cnt >  1u) mac( 1u,  a1);
            if (m_cnt >  2u) mac( 2u,  a2);
            if (m_cnt >  3u) mac( 3u,  a3);
            if (m_cnt >  4u) mac( 4u,  a4);
            if (m_cnt >  5u) mac( 5u,  a5);
            if (m_cnt >  6u) mac( 6u,  a6);
            if (m_cnt >  7u) mac( 7u,  a7);
            if (m_cnt >  8u) mac( 8u,  a8);
            if (m_cnt >  9u) mac( 9u,  a9);
            if (m_cnt > 10u) mac(10u, a10);
            if (m_cnt > 11u) mac(11u, a11);
            if (m_cnt > 12u) mac(12u, a12);
            if (m_cnt > 13u) mac(13u, a13);
            if (m_cnt > 14u) mac(14u, a14);
            if (m_cnt > 15u) mac(15u, a15);
        }
        __syncthreads();
    }

    if (!valid_row) return;
    auto emit = [&](uint32_t dm_i, float v) {
        v = vegajet::ornith_dev::block32_sum(v);
        if (lane == 0u) y_out[static_cast<uint64_t>(m0 + dm_i) * rows + row] = v;
    };
    emit(0u, a0);
    if (m_cnt >  1u) emit( 1u,  a1);
    if (m_cnt >  2u) emit( 2u,  a2);
    if (m_cnt >  3u) emit( 3u,  a3);
    if (m_cnt >  4u) emit( 4u,  a4);
    if (m_cnt >  5u) emit( 5u,  a5);
    if (m_cnt >  6u) emit( 6u,  a6);
    if (m_cnt >  7u) emit( 7u,  a7);
    if (m_cnt >  8u) emit( 8u,  a8);
    if (m_cnt >  9u) emit( 9u,  a9);
    if (m_cnt > 10u) emit(10u, a10);
    if (m_cnt > 11u) emit(11u, a11);
    if (m_cnt > 12u) emit(12u, a12);
    if (m_cnt > 13u) emit(13u, a13);
    if (m_cnt > 14u) emit(14u, a14);
    if (m_cnt > 15u) emit(15u, a15);
}

// 4d. High-occupancy M=2..8 batched VJ_Q4_A8 GEMM for speculative verify (M=5 MTP, M<=8 Suffix):
//     Half the accumulators (a0..a7) and half the LDS (10 KiB vs 20 KiB) -> <= 64 VGPRs & 4 WGs/CU
//     (2x occupancy vs 73-VGPR 2-WG/CU tcc). Bit-identical float order to ornith_gemm_q4a8_tcc_kernel.
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_gemm_q4a8_m8_kernel(
    const uint8_t*  __restrict__ w_qs,
    const uint32_t* __restrict__ w_dm,
    const int8_t*   __restrict__ x_q8,
    const float2*   __restrict__ x_ds,
    float*          __restrict__ y_out,
    uint32_t rows,
    uint32_t cols,
    uint32_t M,
    uint32_t row_stride_bytes,
    uint16_t*       __restrict__ peer_tx) {
    const uint32_t tid  = threadIdx.x;
    const uint32_t grp  = tid >> 5; // 0..7
    const uint32_t lane = tid & 31u;
    const uint32_t row  = blockIdx.x * 8u + grp;
    const uint32_t m0   = blockIdx.y * 8u;
    if (m0 >= M) return;

    __shared__ uint4  sm_x[8 * 64];  // 8 KiB (reused as sm_out[8*8] at epilogue)
    __shared__ float2 sm_xds[8 * 32]; // 2 KiB -> 10 KiB total (4 WGs/CU!)

    const uint32_t  m_cnt = min(8u, M - m0);
    const uint32_t  nb    = cols >> 5;
    const bool      valid_row = (row < rows);
    const uint4*    w4    = valid_row ? reinterpret_cast<const uint4*>(w_qs + static_cast<uint64_t>(row) * row_stride_bytes) : nullptr;
    const uint32_t* dm    = valid_row ? (w_dm + static_cast<uint64_t>(row) * nb) : nullptr;

    float a0=0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0, a7=0;

    #pragma nounroll
    for (uint32_t b_base = 0u; b_base < nb; b_base += 32u) {
        const uint32_t tile_nb = min(32u, nb - b_base);
        #pragma unroll
        for (uint32_t idx = tid; idx < m_cnt * 64u; idx += 256u) {
            const uint32_t dm_i = idx >> 6;
            const uint32_t sub  = idx & 63u;
            if ((sub >> 1) < tile_nb) {
                const uint4* gx4 = reinterpret_cast<const uint4*>(x_q8 + static_cast<uint64_t>(m0 + dm_i) * cols);
                sm_x[idx] = gx4[b_base * 2u + sub];
            }
        }
        if (tid < m_cnt * 32u) {
            const uint32_t dm_i = tid >> 5;
            const uint32_t sub  = tid & 31u;
            if (sub < tile_nb) sm_xds[tid] = x_ds[static_cast<uint64_t>(m0 + dm_i) * nb + b_base + sub];
        }
        __syncthreads();

        if (valid_row && lane < tile_nb) {
            const uint32_t b   = b_base + lane;
            const uint4    wv  = w4[b];
            const uint32_t dmv = dm[b];
            const float    wd  = vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(dmv & 0xFFFFu));
            const float    wm  = vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(dmv >> 16));
            const uint32_t wl0 = wv.x & 0x0F0F0F0Fu, wh0 = (wv.x >> 4) & 0x0F0F0F0Fu;
            const uint32_t wl1 = wv.y & 0x0F0F0F0Fu, wh1 = (wv.y >> 4) & 0x0F0F0F0Fu;
            const uint32_t wl2 = wv.z & 0x0F0F0F0Fu, wh2 = (wv.z >> 4) & 0x0F0F0F0Fu;
            const uint32_t wl3 = wv.w & 0x0F0F0F0Fu, wh3 = (wv.w >> 4) & 0x0F0F0F0Fu;

            auto mac = [&](uint32_t dm_i, float& acc) {
                const uint4  xa  = sm_x[dm_i * 64u + lane * 2u];
                const uint4  xb  = sm_x[dm_i * 64u + lane * 2u + 1u];
                const float2 xds = sm_xds[dm_i * 32u + lane];
                int32_t dp = vegajet::ornith_dev::sdot4(wl0, xa.x, 0);
                dp         = vegajet::ornith_dev::sdot4(wl1, xa.y, dp);
                dp         = vegajet::ornith_dev::sdot4(wl2, xa.z, dp);
                dp         = vegajet::ornith_dev::sdot4(wl3, xa.w, dp);
                dp         = vegajet::ornith_dev::sdot4(wh0, xb.x, dp);
                dp         = vegajet::ornith_dev::sdot4(wh1, xb.y, dp);
                dp         = vegajet::ornith_dev::sdot4(wh2, xb.z, dp);
                dp         = vegajet::ornith_dev::sdot4(wh3, xb.w, dp);
                acc = fmaf(wd * xds.x, static_cast<float>(dp), fmaf(wm, xds.y, acc));
            };
            mac(0u, a0);
            if (m_cnt > 1u) mac(1u, a1);
            if (m_cnt > 2u) mac(2u, a2);
            if (m_cnt > 3u) mac(3u, a3);
            if (m_cnt > 4u) mac(4u, a4);
            if (m_cnt > 5u) mac(5u, a5);
            if (m_cnt > 6u) mac(6u, a6);
            if (m_cnt > 7u) mac(7u, a7);
        }
        __syncthreads();
    }

    float* sm_out = reinterpret_cast<float*>(sm_x);
    const bool do_pp = (peer_tx != nullptr && (rows & 7u) == 0u);
    if (valid_row) {
        auto emit = [&](uint32_t dm_i, float v) {
            v = vegajet::ornith_dev::block32_sum(v);
            if (lane == 0u) {
                y_out[static_cast<uint64_t>(m0 + dm_i) * rows + row] = v;
                if (do_pp) sm_out[dm_i * 8u + grp] = v;
            }
        };
        emit(0u, a0);
        if (m_cnt > 1u) emit(1u, a1);
        if (m_cnt > 2u) emit(2u, a2);
        if (m_cnt > 3u) emit(3u, a3);
        if (m_cnt > 4u) emit(4u, a4);
        if (m_cnt > 5u) emit(5u, a5);
        if (m_cnt > 6u) emit(6u, a6);
        if (m_cnt > 7u) emit(7u, a7);
    }
    if (do_pp) {
        __syncthreads();
        if (tid < m_cnt) {
            const float* so = sm_out + tid * 8u;
            const __half2 h0 = __float22half2_rn(make_float2(so[0], so[1]));
            const __half2 h1 = __float22half2_rn(make_float2(so[2], so[3]));
            const __half2 h2 = __float22half2_rn(make_float2(so[4], so[5]));
            const __half2 h3 = __float22half2_rn(make_float2(so[6], so[7]));
            typedef uint32_t u32x4_t __attribute__((ext_vector_type(4)));
            u32x4_t pv = {
                *reinterpret_cast<const uint32_t*>(&h0),
                *reinterpret_cast<const uint32_t*>(&h1),
                *reinterpret_cast<const uint32_t*>(&h2),
                *reinterpret_cast<const uint32_t*>(&h3)
            };
            __builtin_nontemporal_store(pv, reinterpret_cast<u32x4_t*>(peer_tx + static_cast<uint64_t>(m0 + tid) * rows + blockIdx.x * 8u));
        }
    }
}

// 4e. High-occupancy M=9..12 batched VJ_Q4_A8 GEMM for Suffix/Hybrid verify (replaces 73-VGPR 2-WG/CU tcc):
//     12 accumulators + 15 KiB LDS -> <= 64 VGPRs & 4 WGs/CU (2x occupancy, 0 wasted slots at M=12!) + BAR0 prepush.
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_gemm_q4a8_m12_kernel(
    const uint8_t*  __restrict__ w_qs,
    const uint32_t* __restrict__ w_dm,
    const int8_t*   __restrict__ x_q8,
    const float2*   __restrict__ x_ds,
    float*          __restrict__ y_out,
    uint32_t rows,
    uint32_t cols,
    uint32_t M,
    uint32_t row_stride_bytes,
    uint16_t*       __restrict__ peer_tx) {
    const uint32_t tid  = threadIdx.x;
    const uint32_t grp  = tid >> 5; // 0..7
    const uint32_t lane = tid & 31u;
    const uint32_t row  = blockIdx.x * 8u + grp;
    const uint32_t m0   = blockIdx.y * 12u;
    if (m0 >= M) return;

    __shared__ uint4  sm_x[12 * 64];  // 12 KiB (reused as sm_out[12*8] at epilogue)
    __shared__ float2 sm_xds[12 * 32]; // 3 KiB -> 15 KiB total (4 WGs/CU!)

    const uint32_t  m_cnt = min(12u, M - m0);
    const uint32_t  nb    = cols >> 5;
    const bool      valid_row = (row < rows);
    const uint4*    w4    = valid_row ? reinterpret_cast<const uint4*>(w_qs + static_cast<uint64_t>(row) * row_stride_bytes) : nullptr;
    const uint32_t* dm    = valid_row ? (w_dm + static_cast<uint64_t>(row) * nb) : nullptr;

    float a0=0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0, a7=0, a8=0, a9=0, a10=0, a11=0;

    #pragma nounroll
    for (uint32_t b_base = 0u; b_base < nb; b_base += 32u) {
        const uint32_t tile_nb = min(32u, nb - b_base);
        #pragma nounroll
        for (uint32_t idx = tid; idx < m_cnt * 64u; idx += 256u) {
            const uint32_t dm_i = idx >> 6;
            const uint32_t sub  = idx & 63u;
            if ((sub >> 1) < tile_nb) {
                const uint4* gx4 = reinterpret_cast<const uint4*>(x_q8 + static_cast<uint64_t>(m0 + dm_i) * cols);
                sm_x[idx] = gx4[b_base * 2u + sub];
            }
        }
        #pragma nounroll
        for (uint32_t idx = tid; idx < m_cnt * 32u; idx += 256u) {
            const uint32_t dm_i = idx >> 5;
            const uint32_t sub  = idx & 31u;
            if (sub < tile_nb) sm_xds[idx] = x_ds[static_cast<uint64_t>(m0 + dm_i) * nb + b_base + sub];
        }
        __syncthreads();

        if (valid_row && lane < tile_nb) {
            const uint32_t b   = b_base + lane;
            const uint4    wv  = w4[b];
            const uint32_t dmv = dm[b];
            const float    wd  = vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(dmv & 0xFFFFu));
            const float    wm  = vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(dmv >> 16));
            const uint32_t wl0 = wv.x & 0x0F0F0F0Fu, wh0 = (wv.x >> 4) & 0x0F0F0F0Fu;
            const uint32_t wl1 = wv.y & 0x0F0F0F0Fu, wh1 = (wv.y >> 4) & 0x0F0F0F0Fu;
            const uint32_t wl2 = wv.z & 0x0F0F0F0Fu, wh2 = (wv.z >> 4) & 0x0F0F0F0Fu;
            const uint32_t wl3 = wv.w & 0x0F0F0F0Fu, wh3 = (wv.w >> 4) & 0x0F0F0F0Fu;

            auto mac = [&](uint32_t dm_i, float& acc) {
                const uint4  xa  = sm_x[dm_i * 64u + lane * 2u];
                const uint4  xb  = sm_x[dm_i * 64u + lane * 2u + 1u];
                const float2 xds = sm_xds[dm_i * 32u + lane];
                int32_t dp = vegajet::ornith_dev::sdot4(wl0, xa.x, 0);
                dp         = vegajet::ornith_dev::sdot4(wl1, xa.y, dp);
                dp         = vegajet::ornith_dev::sdot4(wl2, xa.z, dp);
                dp         = vegajet::ornith_dev::sdot4(wl3, xa.w, dp);
                dp         = vegajet::ornith_dev::sdot4(wh0, xb.x, dp);
                dp         = vegajet::ornith_dev::sdot4(wh1, xb.y, dp);
                dp         = vegajet::ornith_dev::sdot4(wh2, xb.z, dp);
                dp         = vegajet::ornith_dev::sdot4(wh3, xb.w, dp);
                acc = fmaf(wd * xds.x, static_cast<float>(dp), fmaf(wm, xds.y, acc));
            };
            mac(0u, a0);
            if (m_cnt > 1u) mac(1u, a1);
            if (m_cnt > 2u) mac(2u, a2);
            if (m_cnt > 3u) mac(3u, a3);
            if (m_cnt > 4u) mac(4u, a4);
            if (m_cnt > 5u) mac(5u, a5);
            __builtin_amdgcn_sched_barrier(0);
            if (m_cnt > 6u) mac(6u, a6);
            if (m_cnt > 7u) mac(7u, a7);
            if (m_cnt > 8u) mac(8u, a8);
            if (m_cnt > 9u) mac(9u, a9);
            if (m_cnt > 10u) mac(10u, a10);
            if (m_cnt > 11u) mac(11u, a11);
        }
        __syncthreads();
    }

    float* sm_out = reinterpret_cast<float*>(sm_x);
    const bool do_pp = (peer_tx != nullptr && (rows & 7u) == 0u);
    if (valid_row) {
        auto emit = [&](uint32_t dm_i, float v) {
            v = vegajet::ornith_dev::block32_sum(v);
            if (lane == 0u) {
                y_out[static_cast<uint64_t>(m0 + dm_i) * rows + row] = v;
                if (do_pp) sm_out[dm_i * 8u + grp] = v;
            }
        };
        emit(0u, a0);
        if (m_cnt > 1u) emit(1u, a1);
        if (m_cnt > 2u) emit(2u, a2);
        if (m_cnt > 3u) emit(3u, a3);
        if (m_cnt > 4u) emit(4u, a4);
        if (m_cnt > 5u) emit(5u, a5);
        if (m_cnt > 6u) emit(6u, a6);
        if (m_cnt > 7u) emit(7u, a7);
        if (m_cnt > 8u) emit(8u, a8);
        if (m_cnt > 9u) emit(9u, a9);
        if (m_cnt > 10u) emit(10u, a10);
        if (m_cnt > 11u) emit(11u, a11);
    }
    if (do_pp) {
        __syncthreads();
        if (tid < m_cnt) {
            const float* so = sm_out + tid * 8u;
            const __half2 h0 = __float22half2_rn(make_float2(so[0], so[1]));
            const __half2 h1 = __float22half2_rn(make_float2(so[2], so[3]));
            const __half2 h2 = __float22half2_rn(make_float2(so[4], so[5]));
            const __half2 h3 = __float22half2_rn(make_float2(so[6], so[7]));
            typedef uint32_t u32x4_t __attribute__((ext_vector_type(4)));
            u32x4_t pv = {
                *reinterpret_cast<const uint32_t*>(&h0),
                *reinterpret_cast<const uint32_t*>(&h1),
                *reinterpret_cast<const uint32_t*>(&h2),
                *reinterpret_cast<const uint32_t*>(&h3)
            };
            __builtin_nontemporal_store(pv, reinterpret_cast<u32x4_t*>(peer_tx + static_cast<uint64_t>(m0 + tid) * rows + blockIdx.x * 8u));
        }
    }
}

// 4f. Bit-identical 32-lane LDS-tiled M=9..16 batched VJ_Q4_A8 GEMM for native block_size=16 DFlash2 & Suffix-15 verify:
//     Loads w_qs/w_dm ONCE per 32-block K-tile and reuses 10 KiB LDS (4 WGs/CU) across 2 sub-batches of <=8 tokens
//     (a0..a7, a8..a15). Exact 32-lane block32_sum order = 100% bit-identical to M=1 gemv_q4_w32 & m8/m12!
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_gemm_q4a8_m16_l2_kernel(
    const uint8_t*  __restrict__ w_qs,
    const uint32_t* __restrict__ w_dm,
    const int8_t*   __restrict__ x_q8,
    const float2*   __restrict__ x_ds,
    float*          __restrict__ y_out,
    uint32_t rows,
    uint32_t cols,
    uint32_t M,
    uint32_t row_stride_bytes,
    uint16_t*       __restrict__ peer_tx) {
    const uint32_t tid  = threadIdx.x;
    const uint32_t grp  = tid >> 5;  // 0..7 (8 rows per WG, 32 lanes/row = bit-identical to M=1!)
    const uint32_t lane = tid & 31u;
    const uint32_t row  = blockIdx.x * 8u + grp;
    const uint32_t m0   = blockIdx.y * 16u;
    if (m0 >= M) return;

    __shared__ uint4  sm_x[8 * 64];   // 8,192 B (reused for tokens 0..7, then 8..15, then sm_out[16*8]!)
    __shared__ float2 sm_xds[8 * 32]; // 2,048 B
    __shared__ uint4  sm_w[256];      // 4,096 B -> 14,336 B total (4 WGs/CU, frees 4 VGPRs -> 0 B spill!)

    const uint32_t  m_cnt = min(16u, M - m0);
    const uint32_t  m_lo  = min(8u, m_cnt);
    const uint32_t  m_hi  = (m_cnt > 8u) ? (m_cnt - 8u) : 0u;
    const uint32_t  nb    = cols >> 5;
    const bool      valid_row = (row < rows);
    const uint4*    w4    = valid_row ? reinterpret_cast<const uint4*>(w_qs + static_cast<uint64_t>(row) * row_stride_bytes) : nullptr;
    const uint32_t* dm    = valid_row ? (w_dm + static_cast<uint64_t>(row) * nb) : nullptr;

    float a0=0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0, a7=0;
    float a8=0, a9=0, a10=0, a11=0, a12=0, a13=0, a14=0, a15=0;

    #pragma nounroll
    for (uint32_t b_base = 0u; b_base < nb; b_base += 32u) {
        const uint32_t tile_nb = min(32u, nb - b_base);
        const bool active = (valid_row && lane < tile_nb);

        // Sub-batch 0: stage tokens 0..m_lo-1 BEFORE loading weights (zero live weight VGPRs across copy!)
        #pragma nounroll
        for (uint32_t idx = tid; idx < m_lo * 64u; idx += 256u) {
            const uint32_t dm_i = idx >> 6;
            const uint32_t sub  = idx & 63u;
            if ((sub >> 1) < tile_nb) {
                const uint4* gx4 = reinterpret_cast<const uint4*>(x_q8 + static_cast<uint64_t>(m0 + dm_i) * cols);
                sm_x[idx] = gx4[b_base * 2u + sub];
            }
        }
        if (tid < m_lo * 32u) {
            const uint32_t dm_i = tid >> 5;
            const uint32_t sub  = tid & 31u;
            if (sub < tile_nb) sm_xds[tid] = x_ds[static_cast<uint64_t>(m0 + dm_i) * nb + b_base + sub];
        }
        __syncthreads();

        uint32_t dmv = 0u;
        if (active) {
            const uint32_t b = b_base + lane;
            sm_w[tid] = vegajet::ornith_dev::ldg_u32x4_stream(w4 + b);
            dmv = vegajet::ornith_dev::ldg_u32_stream(dm + b);
        }

        auto mac = [&](uint32_t sm_t, float& acc) {
            const uint4 wv = sm_w[tid];
            const uint4 xa = sm_x[sm_t * 64u + lane * 2u];
            int32_t dp = vegajet::ornith_dev::sdot4(wv.x & 0x0F0F0F0Fu, xa.x, 0);
            dp         = vegajet::ornith_dev::sdot4(wv.y & 0x0F0F0F0Fu, xa.y, dp);
            dp         = vegajet::ornith_dev::sdot4(wv.z & 0x0F0F0F0Fu, xa.z, dp);
            dp         = vegajet::ornith_dev::sdot4(wv.w & 0x0F0F0F0Fu, xa.w, dp);
            __builtin_amdgcn_sched_barrier(0);
            const uint4 xb = sm_x[sm_t * 64u + lane * 2u + 1u];
            dp         = vegajet::ornith_dev::sdot4((wv.x >> 4) & 0x0F0F0F0Fu, xb.x, dp);
            dp         = vegajet::ornith_dev::sdot4((wv.y >> 4) & 0x0F0F0F0Fu, xb.y, dp);
            dp         = vegajet::ornith_dev::sdot4((wv.z >> 4) & 0x0F0F0F0Fu, xb.z, dp);
            dp         = vegajet::ornith_dev::sdot4((wv.w >> 4) & 0x0F0F0F0Fu, xb.w, dp);
            const float2 xds = sm_xds[sm_t * 32u + lane];
            const float  wd  = vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(dmv & 0xFFFFu));
            const float  wm  = vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(dmv >> 16));
            acc = fmaf(wd * xds.x, static_cast<float>(dp), fmaf(wm, xds.y, acc));
            __builtin_amdgcn_sched_barrier(0);
        };

        if (active) {
            mac(0u, a0);
            if (m_lo > 1u) mac(1u, a1);
            if (m_lo > 2u) mac(2u, a2);
            if (m_lo > 3u) mac(3u, a3);
            if (m_lo > 4u) mac(4u, a4);
            if (m_lo > 5u) mac(5u, a5);
            if (m_lo > 6u) mac(6u, a6);
            if (m_lo > 7u) mac(7u, a7);
        }
        __syncthreads();

        // Sub-batch 1: tokens 8..m_cnt-1 (reuses wv/dmv in registers and sm_x/sm_xds in LDS!)
        if (m_hi > 0u) {
            #pragma nounroll
            for (uint32_t idx = tid; idx < m_hi * 64u; idx += 256u) {
                const uint32_t dm_i = idx >> 6;
                const uint32_t sub  = idx & 63u;
                if ((sub >> 1) < tile_nb) {
                    const uint4* gx4 = reinterpret_cast<const uint4*>(x_q8 + static_cast<uint64_t>(m0 + 8u + dm_i) * cols);
                    sm_x[idx] = gx4[b_base * 2u + sub];
                }
            }
            if (tid < m_hi * 32u) {
                const uint32_t dm_i = tid >> 5;
                const uint32_t sub  = tid & 31u;
                if (sub < tile_nb) sm_xds[tid] = x_ds[static_cast<uint64_t>(m0 + 8u + dm_i) * nb + b_base + sub];
            }
            __syncthreads();

            if (active) {
                mac(0u, a8);
                if (m_hi > 1u) mac(1u, a9);
                if (m_hi > 2u) mac(2u, a10);
                if (m_hi > 3u) mac(3u, a11);
                if (m_hi > 4u) mac(4u, a12);
                if (m_hi > 5u) mac(5u, a13);
                if (m_hi > 6u) mac(6u, a14);
                if (m_hi > 7u) mac(7u, a15);
            }
            __syncthreads();
        }
    }

    float* sm_out = reinterpret_cast<float*>(sm_x);
    const bool do_pp = (peer_tx != nullptr && (rows & 7u) == 0u);
    if (valid_row) {
        auto emit = [&](uint32_t dm_i, float v) {
            v = vegajet::ornith_dev::block32_sum(v);
            if (lane == 0u) {
                y_out[static_cast<uint64_t>(m0 + dm_i) * rows + row] = v;
                if (do_pp) sm_out[dm_i * 8u + grp] = v;
            }
        };
        emit(0u, a0);
        if (m_cnt > 1u)  emit(1u, a1);
        if (m_cnt > 2u)  emit(2u, a2);
        if (m_cnt > 3u)  emit(3u, a3);
        if (m_cnt > 4u)  emit(4u, a4);
        if (m_cnt > 5u)  emit(5u, a5);
        if (m_cnt > 6u)  emit(6u, a6);
        if (m_cnt > 7u)  emit(7u, a7);
        if (m_cnt > 8u)  emit(8u, a8);
        if (m_cnt > 9u)  emit(9u, a9);
        if (m_cnt > 10u) emit(10u, a10);
        if (m_cnt > 11u) emit(11u, a11);
        if (m_cnt > 12u) emit(12u, a12);
        if (m_cnt > 13u) emit(13u, a13);
        if (m_cnt > 14u) emit(14u, a14);
        if (m_cnt > 15u) emit(15u, a15);
    }
    if (do_pp) {
        __syncthreads();
        if (tid < m_cnt) {
            const float* so = sm_out + tid * 8u;
            const __half2 h0 = __float22half2_rn(make_float2(so[0], so[1]));
            const __half2 h1 = __float22half2_rn(make_float2(so[2], so[3]));
            const __half2 h2 = __float22half2_rn(make_float2(so[4], so[5]));
            const __half2 h3 = __float22half2_rn(make_float2(so[6], so[7]));
            typedef uint32_t u32x4_t __attribute__((ext_vector_type(4)));
            u32x4_t pv = {
                *reinterpret_cast<const uint32_t*>(&h0),
                *reinterpret_cast<const uint32_t*>(&h1),
                *reinterpret_cast<const uint32_t*>(&h2),
                *reinterpret_cast<const uint32_t*>(&h3)
            };
            __builtin_nontemporal_store(pv, reinterpret_cast<u32x4_t*>(peer_tx + static_cast<uint64_t>(m0 + tid) * rows + blockIdx.x * 8u));
        }
    }
}

