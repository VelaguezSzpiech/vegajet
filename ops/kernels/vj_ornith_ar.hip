// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "ops/kernels/vj_ornith_common.hiph"
#include "ops/kernels/vj_ar_fused.hiph"

// ============================================================================
// 1. Token Embedding Gather: VJ_Q8_DP4A [vocab, D] -> FP32 h [M, D]
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_embed_q8_kernel(
    const int8_t*   __restrict__ emb_qs,
    const uint16_t* __restrict__ emb_ds,
    const int32_t*  __restrict__ tokens,
    float*          __restrict__ h_out,
    uint32_t M,
    uint32_t D) {
    const uint32_t m = blockIdx.x;
    if (m >= M) return;
    const uint64_t tok = static_cast<uint64_t>(tokens[m]);
    const int8_t*   qs = emb_qs + tok * D;
    const uint16_t* ds = emb_ds + tok * (D >> 5);
    float* dst = h_out + static_cast<uint64_t>(m) * D;
    for (uint32_t c = threadIdx.x; c < D; c += 256u) {
        dst[c] = static_cast<float>(qs[c]) * vegajet::ornith_dev::fp16_to_f32(ds[c >> 5]);
    }
}

// ============================================================================
// 2. Fused (Optional Residual Add) + RMSNorm + b=32 INT8 Quantize (+ FP32 x_norm & x_ds={d, sum})
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_rms_quant_q8_kernel(
    float*       __restrict__ h_io,
    const float* __restrict__ x_add,
    const float* __restrict__ w_rms,
    float*       __restrict__ x_norm_out,
    int8_t*      __restrict__ x_q8_out,
    float2*      __restrict__ x_ds_out,
    uint32_t M,
    uint32_t D,
    float    eps) {
    const uint32_t m    = blockIdx.x;
    const uint32_t tid  = threadIdx.x;
    const uint32_t wave = tid >> 6;
    const uint32_t lane = tid & 63u;
    if (m >= M) return;

    __shared__ float sm_red[4];
    float* h = h_io + static_cast<uint64_t>(m) * D;
    const float* xa = x_add ? (x_add + static_cast<uint64_t>(m) * D) : nullptr;

    float ss = 0.0f;
    #pragma nounroll
    for (uint32_t c = tid; c < D; c += 256u) {
        float v = h[c];
        if (xa) { v += xa[c]; h[c] = v; }
        ss = fmaf(v, v, ss);
    }
    ss = vegajet::ornith_dev::wave64_sum(ss);
    if (lane == 0u) sm_red[wave] = ss;
    __syncthreads();
    const float inv_rms = rsqrtf((sm_red[0] + sm_red[1] + sm_red[2] + sm_red[3]) / static_cast<float>(D) + eps);

    float*  xn = x_norm_out ? (x_norm_out + static_cast<uint64_t>(m) * D) : nullptr;
    int8_t* xq = x_q8_out + static_cast<uint64_t>(m) * D;
    float2* xd = x_ds_out + static_cast<uint64_t>(m) * (D >> 5);
    const uint32_t sub = tid & 31u;
    #pragma nounroll
    for (uint32_t blk = tid >> 5; blk < (D >> 5); blk += 8u) {
        const uint32_t c = (blk << 5) + sub;
        const float y = h[c] * inv_rms * w_rms[c];
        if (xn) xn[c] = y;
        const float amax  = vegajet::ornith_dev::block32_max(fabsf(y));
        const float bsum  = vegajet::ornith_dev::block32_sum(y);
        const float d     = amax * (1.0f / 127.0f);
        const float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        xq[c] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(y * inv_d))));
        if (sub == 0u) xd[blk] = make_float2(d, bsum);
    }
}

extern "C" __global__ __launch_bounds__(256, 4)
void ornith_p2p_ar_res_rms_quant_kernel(
    const float*    __restrict__ x_partial,
    uint16_t*       __restrict__ peer_tx,
    const uint16_t* __restrict__ local_rx,
    uint32_t*       __restrict__ peer_flag,
    const uint32_t* __restrict__ local_flag,
    uint32_t*       __restrict__ bar_scratch,
    float*          __restrict__ h_io,
    const float*    __restrict__ w_rms,
    float*          __restrict__ x_norm_out,
    int8_t*         __restrict__ x_q8_out,
    float2*         __restrict__ x_ds_out,
    uint32_t M,
    uint32_t D,
    uint32_t epoch,
    uint32_t do_norm,
    float    eps) {
    if (M == 1u && D == 3072u) {
        if (blockIdx.x != 0u) return;
        // Three 128-packet stripes; publish only after every writer drains.
        if (threadIdx.x < 128u) {
            const unsigned t = threadIdx.x;
            const float4* x = reinterpret_cast<const float4*>(x_partial) + 2u * t;
            using U4 = vegajet::ornith_dev::u32x4;
            U4 packets[3];
            #pragma unroll
            for (unsigned half = 0; half < 2; ++half) {
                float4 v[3];
                #pragma unroll
                for (unsigned k = 0; k < 3; ++k) v[k] = x[256u * k + half];
                asm volatile("" ::: "memory");
                #pragma unroll
                for (unsigned k = 0; k < 3; ++k) {
                    packets[k][2u * half] = __builtin_bit_cast(uint32_t, __float22half2_rn(make_float2(v[k].x, v[k].y)));
                    packets[k][2u * half + 1u] = __builtin_bit_cast(uint32_t, __float22half2_rn(make_float2(v[k].z, v[k].w)));
                }
                asm volatile("" ::: "memory");
            }
            #pragma unroll
            for (unsigned k = 0; k < 3; ++k)
                __builtin_nontemporal_store(packets[k], reinterpret_cast<U4*>(peer_tx) + t + 128u * k);
        }
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        __threadfence_system();
        __syncthreads();
        ornith_p2p_ar_fp16_body<true>(x_partial, peer_tx, local_rx, peer_flag, local_flag, bar_scratch, h_io, w_rms, x_norm_out, x_q8_out, x_ds_out, 1u, 3072u, epoch, do_norm, eps, nullptr);
        return;
    }
    ornith_p2p_ar_fp16_body<false>(x_partial, peer_tx, local_rx, peer_flag, local_flag, bar_scratch, h_io, w_rms, x_norm_out, x_q8_out, x_ds_out, M, D, epoch, do_norm, eps, nullptr);
}

// Pre-pushed M=1 variant: the preceding GEMV / MoE epilogue already posted the 16B FP16 BAR0 payload into peer_tx,
// so PCIe flight overlapped with trailing GEMV waves; this kernel only signals the peer, waits, reduces & normalizes.
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_p2p_ar_res_rms_quant_prepush_kernel(
    const float*    __restrict__ x_partial,
    uint16_t*       __restrict__ peer_tx,
    const uint16_t* __restrict__ local_rx,
    uint32_t*       __restrict__ peer_flag,
    const uint32_t* __restrict__ local_flag,
    uint32_t*       __restrict__ bar_scratch,
    float*          __restrict__ h_io,
    const float*    __restrict__ w_rms,
    float*          __restrict__ x_norm_out,
    int8_t*         __restrict__ x_q8_out,
    float2*         __restrict__ x_ds_out,
    uint32_t M,
    uint32_t D,
    uint32_t epoch,
    uint32_t do_norm,
    float    eps) {
    ornith_p2p_ar_fp16_body<true>(x_partial, peer_tx, local_rx, peer_flag, local_flag, bar_scratch, h_io, w_rms, x_norm_out, x_q8_out, x_ds_out, M, D, epoch, do_norm, eps, nullptr);
}

// Timeline variant of (3): identical work, plus six s_memrealtime stamps per AllReduce (--ar-timeline only).
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_p2p_ar_res_rms_quant_ts_kernel(
    const float*    __restrict__ x_partial,
    uint16_t*       __restrict__ peer_tx,
    const uint16_t* __restrict__ local_rx,
    uint32_t*       __restrict__ peer_flag,
    const uint32_t* __restrict__ local_flag,
    uint32_t*       __restrict__ bar_scratch,
    float*          __restrict__ h_io,
    const float*    __restrict__ w_rms,
    float*          __restrict__ x_norm_out,
    int8_t*         __restrict__ x_q8_out,
    float2*         __restrict__ x_ds_out,
    uint32_t M,
    uint32_t D,
    uint32_t epoch,
    uint32_t do_norm,
    float    eps,
    uint64_t* __restrict__ ts) {
    ornith_p2p_ar_fp16_body(x_partial, peer_tx, local_rx, peer_flag, local_flag, bar_scratch, h_io, w_rms, x_norm_out, x_q8_out, x_ds_out, M, D, epoch, do_norm, eps, ts);
}

// Reference for (3): the former two-pass epilogue, kept only so `--bench-ar` can prove the single-pass kernel
// bit-identical on hardware (h, x_norm, x_q8, x_ds).
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_p2p_ar_res_rms_quant_ref_kernel(
    const float*    __restrict__ x_partial,
    uint16_t*       __restrict__ peer_tx,
    const uint16_t* __restrict__ local_rx,
    uint32_t*       __restrict__ peer_flag,
    const uint32_t* __restrict__ local_flag,
    uint32_t*       __restrict__ bar_scratch,
    float*          __restrict__ h_io,
    const float*    __restrict__ w_rms,
    float*          __restrict__ x_norm_out,
    int8_t*         __restrict__ x_q8_out,
    float2*         __restrict__ x_ds_out,
    uint32_t M,
    uint32_t D,
    uint32_t epoch,
    uint32_t do_norm,
    float    eps) {
    const uint32_t m    = blockIdx.x;
    const uint32_t tid  = threadIdx.x;
    const uint32_t wave = tid >> 6;
    const uint32_t lane = tid & 63u;
    if (m >= M) return;

    __shared__ float sm_red[4];
    const uint64_t off = static_cast<uint64_t>(m) * D;
    const float4*  xp4 = reinterpret_cast<const float4*>(x_partial + off);
    uint4*         pt4 = reinterpret_cast<uint4*>(peer_tx + off);
    const uint32_t n8  = D >> 3; // 384 uint4s (16B each) = 3072 FP16s -> 8x fewer PCIe TLPs!

    #pragma nounroll
    for (uint32_t i = tid; i < n8; i += 256u) {
        const float4 a = xp4[i * 2u];
        const float4 b = xp4[i * 2u + 1u];
        const __half2 h0 = __float22half2_rn(make_float2(a.x, a.y));
        const __half2 h1 = __float22half2_rn(make_float2(a.z, a.w));
        const __half2 h2 = __float22half2_rn(make_float2(b.x, b.y));
        const __half2 h3 = __float22half2_rn(make_float2(b.z, b.w));
        uint4 pack;
        pack.x = *reinterpret_cast<const uint32_t*>(&h0);
        pack.y = *reinterpret_cast<const uint32_t*>(&h1);
        pack.z = *reinterpret_cast<const uint32_t*>(&h2);
        pack.w = *reinterpret_cast<const uint32_t*>(&h3);
        pt4[i] = pack;
    }
    if (M > 1u) {
        vegajet::ornith_dev::grid_barrier_m(bar_scratch, M, epoch * 2u - 1u);
    } else {
        __syncthreads();
    }

    if (m == 0u && tid == 0u) {
        __threadfence_system();
        __atomic_store_n(peer_flag, epoch, __ATOMIC_RELEASE);
        while (__atomic_load_n(local_flag, __ATOMIC_ACQUIRE) != epoch) {
            __builtin_amdgcn_s_sleep(1);
        }
        __threadfence_system();
    }
    if (M > 1u) {
        vegajet::ornith_dev::grid_barrier_m(bar_scratch, M, epoch * 2u);
    } else {
        __syncthreads();
    }

    const uint4* lr4 = reinterpret_cast<const uint4*>(local_rx + off);
    float4*      h4  = reinterpret_cast<float4*>(h_io + off);
    float ss = 0.0f;
    #pragma nounroll
    for (uint32_t i = tid; i < n8; i += 256u) {
        const float4 la = xp4[i * 2u];
        const float4 lb = xp4[i * 2u + 1u];
        const uint4  rv = vegajet::ornith_dev::ldg_u32x4_stream(lr4 + i);
        const float2 p0 = __half22float2(*reinterpret_cast<const __half2*>(&rv.x));
        const float2 p1 = __half22float2(*reinterpret_cast<const __half2*>(&rv.y));
        const float2 p2 = __half22float2(*reinterpret_cast<const __half2*>(&rv.z));
        const float2 p3 = __half22float2(*reinterpret_cast<const __half2*>(&rv.w));
        float4 ha = h4[i * 2u];
        float4 hb = h4[i * 2u + 1u];
        ha.x += __half2float(__float2half(la.x)) + p0.x;
        ha.y += __half2float(__float2half(la.y)) + p0.y;
        ha.z += __half2float(__float2half(la.z)) + p1.x;
        ha.w += __half2float(__float2half(la.w)) + p1.y;
        hb.x += __half2float(__float2half(lb.x)) + p2.x;
        hb.y += __half2float(__float2half(lb.y)) + p2.y;
        hb.z += __half2float(__float2half(lb.z)) + p3.x;
        hb.w += __half2float(__float2half(lb.w)) + p3.y;
        h4[i * 2u]      = ha;
        h4[i * 2u + 1u] = hb;
        ss = fmaf(ha.x, ha.x, fmaf(ha.y, ha.y, fmaf(ha.z, ha.z, fmaf(ha.w, ha.w, ss))));
        ss = fmaf(hb.x, hb.x, fmaf(hb.y, hb.y, fmaf(hb.z, hb.z, fmaf(hb.w, hb.w, ss))));
    }
    if (do_norm == 0u) return;

    float* h = h_io + off;

    ss = vegajet::ornith_dev::wave64_sum(ss);
    if (lane == 0u) sm_red[wave] = ss;
    __syncthreads();
    const float inv_rms = rsqrtf((sm_red[0] + sm_red[1] + sm_red[2] + sm_red[3]) / static_cast<float>(D) + eps);

    float*  xn = x_norm_out ? (x_norm_out + off) : nullptr;
    int8_t* xq = x_q8_out + off;
    float2* xd = x_ds_out + static_cast<uint64_t>(m) * (D >> 5);
    const uint32_t sub = tid & 31u;
    #pragma nounroll
    for (uint32_t blk = tid >> 5; blk < (D >> 5); blk += 8u) {
        const uint32_t c = (blk << 5) + sub;
        const float y = h[c] * inv_rms * w_rms[c];
        if (xn) xn[c] = y;
        const float amax  = vegajet::ornith_dev::block32_max(fabsf(y));
        const float bsum  = vegajet::ornith_dev::block32_sum(y);
        const float d     = amax * (1.0f / 127.0f);
        const float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        xq[c] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(y * inv_d))));
        if (sub == 0u) xd[blk] = make_float2(d, bsum);
    }
}

// ============================================================================
// 3b. INT8-wire variant of (3): the partial is sent as block-32 INT8 + one FP32 scale per block
//     (D + D/8 bytes per row instead of 2*D: 3456 B vs 6144 B for D=3072; 216 vs 384 16-byte posted writes).
//     Both ranks add dequant(local) + dequant(peer) with un-contracted multiplies, so the two residual
//     streams stay bit-identical. Same kernargs as (3); peer_tx / local_rx are byte buffers. D % 128 == 0.
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_p2p_ar_i8_res_rms_quant_kernel(
    const float*    __restrict__ x_partial,
    uint8_t*        __restrict__ peer_tx,
    const uint8_t*  __restrict__ local_rx,
    uint32_t*       __restrict__ peer_flag,
    const uint32_t* __restrict__ local_flag,
    uint32_t*       __restrict__ bar_scratch,
    float*          __restrict__ h_io,
    const float*    __restrict__ w_rms,
    float*          __restrict__ x_norm_out,
    int8_t*         __restrict__ x_q8_out,
    float2*         __restrict__ x_ds_out,
    uint32_t M,
    uint32_t D,
    uint32_t epoch,
    uint32_t do_norm,
    float    eps) {
    #pragma clang fp contract(off)
    const uint32_t m    = blockIdx.x;
    const uint32_t tid  = threadIdx.x;
    const uint32_t wave = tid >> 6;
    const uint32_t lane = tid & 63u;
    if (m >= M) return;

    __shared__ float sm_red[4];
    __shared__ float sm_d[128];
    const uint32_t nb   = D >> 5;                      // 32-element blocks (96)
    const uint32_t nv   = D >> 4;                      // 16-byte payload vectors (192 <= 256 threads)
    const uint64_t rec  = static_cast<uint64_t>(D) + static_cast<uint64_t>(nb) * 4u;   // wire bytes per row
    const uint32_t blk  = tid >> 1;
    const uint32_t nwg  = gridDim.x;
    const uint32_t m0   = blockIdx.x;
    const uint32_t m1   = m0 + nwg;
    const uint32_t m2   = m0 + nwg * 2u;
    const uint32_t m3   = m0 + nwg * 3u;
    const bool     has1 = (m1 < M);
    const bool     has2 = (m2 < M);
    const bool     has3 = (m3 < M);

    auto pack_push_row = [&](uint32_t m_row, uint4& lq_out, float& ld_out) {
        const uint64_t off = static_cast<uint64_t>(m_row) * D;
        const float4*  xp4 = reinterpret_cast<const float4*>(x_partial + off);
        lq_out = make_uint4(0u, 0u, 0u, 0u);
        ld_out = 0.0f;
        if (tid < nv) {
            const float4* o4 = xp4 + blk * 8u + (tid & 1u) * 4u;
            const float4 v0 = o4[0], v1 = o4[1], v2 = o4[2], v3 = o4[3];
            float amax = fmaxf(fmaxf(fabsf(v0.x), fabsf(v0.y)), fmaxf(fabsf(v0.z), fabsf(v0.w)));
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v1.x), fabsf(v1.y)), fmaxf(fabsf(v1.z), fabsf(v1.w))));
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v2.x), fabsf(v2.y)), fmaxf(fabsf(v2.z), fabsf(v2.w))));
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v3.x), fabsf(v3.y)), fmaxf(fabsf(v3.z), fabsf(v3.w))));
            amax = fmaxf(amax, __shfl_xor(amax, 1, 64));
            ld_out = amax * (1.0f / 127.0f);
            const float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            lq_out.x = vegajet::ornith_dev::ar_q8_pack4(v0, inv);
            lq_out.y = vegajet::ornith_dev::ar_q8_pack4(v1, inv);
            lq_out.z = vegajet::ornith_dev::ar_q8_pack4(v2, inv);
            lq_out.w = vegajet::ornith_dev::ar_q8_pack4(v3, inv);
            typedef uint32_t u32x4_nt __attribute__((ext_vector_type(4)));
            const u32x4_nt p_nt = {lq_out.x, lq_out.y, lq_out.z, lq_out.w};
            __builtin_nontemporal_store(p_nt, reinterpret_cast<u32x4_nt*>(peer_tx + static_cast<uint64_t>(m_row) * rec) + tid);
            if ((tid & 1u) == 0u) sm_d[blk] = ld_out;
        }
        __syncthreads();
        if (tid < (nb >> 2)) {
            typedef uint32_t u32x4_nt __attribute__((ext_vector_type(4)));
            const u32x4_nt d_nt = {
                __float_as_uint(sm_d[tid * 4u]),
                __float_as_uint(sm_d[tid * 4u + 1u]),
                __float_as_uint(sm_d[tid * 4u + 2u]),
                __float_as_uint(sm_d[tid * 4u + 3u])
            };
            __builtin_nontemporal_store(d_nt, reinterpret_cast<u32x4_nt*>(peer_tx + static_cast<uint64_t>(m_row) * rec + D) + tid);
        }
    };

    uint4 lq0, lq1, lq2, lq3;
    float ld0, ld1, ld2, ld3;
    pack_push_row(m0, lq0, ld0);
    if (has1) { __syncthreads(); pack_push_row(m1, lq1, ld1); }
    if (has2) { __syncthreads(); pack_push_row(m2, lq2, ld2); }
    if (has3) { __syncthreads(); pack_push_row(m3, lq3, ld3); }

    asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
    __threadfence_system();
    if (nwg > 1u) {
        vegajet::ornith_dev::grid_barrier_m(bar_scratch, nwg, epoch * 2u - 1u);
    } else {
        __syncthreads();
    }

    if (m0 == 0u && tid == 0u) {
        __threadfence_system();
        __atomic_store_n(peer_flag, epoch, __ATOMIC_RELEASE);
        while (__atomic_load_n(local_flag, __ATOMIC_ACQUIRE) != epoch) {
            __builtin_amdgcn_s_sleep(1);
        }
        __threadfence_system();
    }
    if (nwg > 1u) {
        vegajet::ornith_dev::grid_barrier_m(bar_scratch, nwg, epoch * 2u);
    } else {
        __syncthreads();
    }

    auto finish_row = [&](uint32_t m_row, const uint4& lq, float ld) {
        const uint64_t off = static_cast<uint64_t>(m_row) * D;
        float ss = 0.0f;
        if (tid < nv) {
            const uint8_t* lrow = local_rx + static_cast<uint64_t>(m_row) * rec;
            const uint4 rv = vegajet::ornith_dev::ldg_u32x4_stream(reinterpret_cast<const uint4*>(lrow) + tid);
            const float pd = __uint_as_float(vegajet::ornith_dev::ldg_u32_stream(reinterpret_cast<const uint32_t*>(lrow + D) + blk));
            float4* h4 = reinterpret_cast<float4*>(h_io + off) + tid * 4u;
            ss = vegajet::ornith_dev::ar_q8_accum4(h4 + 0, lq.x, ld, rv.x, pd, ss);
            ss = vegajet::ornith_dev::ar_q8_accum4(h4 + 1, lq.y, ld, rv.y, pd, ss);
            ss = vegajet::ornith_dev::ar_q8_accum4(h4 + 2, lq.z, ld, rv.z, pd, ss);
            ss = vegajet::ornith_dev::ar_q8_accum4(h4 + 3, lq.w, ld, rv.w, pd, ss);
        }
        if (do_norm == 0u) return;

        float* h = h_io + off;
        __syncthreads();
        ss = vegajet::ornith_dev::wave64_sum(ss);
        if (lane == 0u) sm_red[wave] = ss;
        __syncthreads();
        const float inv_rms = rsqrtf((sm_red[0] + sm_red[1] + sm_red[2] + sm_red[3]) / static_cast<float>(D) + eps);

        float*  xn = x_norm_out ? (x_norm_out + off) : nullptr;
        int8_t* xq = x_q8_out + off;
        float2* xd = x_ds_out + static_cast<uint64_t>(m_row) * (D >> 5);
        const uint32_t sub = tid & 31u;
        #pragma nounroll
        for (uint32_t b = tid >> 5; b < (D >> 5); b += 8u) {
            const uint32_t c = (b << 5) + sub;
            const float y = h[c] * inv_rms * w_rms[c];
            if (xn) xn[c] = y;
            const float amax  = vegajet::ornith_dev::block32_max(fabsf(y));
            const float bsum  = vegajet::ornith_dev::block32_sum(y);
            const float d     = amax * (1.0f / 127.0f);
            const float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            xq[c] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(y * inv_d))));
            if (sub == 0u) xd[b] = make_float2(d, bsum);
        }
    };

    finish_row(m0, lq0, ld0);
    if (has1) { __syncthreads(); finish_row(m1, lq1, ld1); }
    if (has2) { __syncthreads(); finish_row(m2, lq2, ld2); }
    if (has3) { __syncthreads(); finish_row(m3, lq3, ld3); }
}

// ============================================================================
// 3c. Dynamic N-Rank (N=2..4) 1-Hop Push-Only Dual-BAR0 Lossless FP16 AllReduce
//     + ResidualAdd + optional RMSNorm + b=32 INT8 Quantize (+ x_ds).
//     Canonical rank-order FP32 accumulation (((p0 + p1) + p2) + p3) guarantees
//     100% bit-identical outputs across all ranks.
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_p2p_ar_n_fp16_kernel(
    const float* __restrict__ x_partial,
    uint64_t ptx0, uint64_t ptx1, uint64_t ptx2, uint64_t ptx3,
    uint64_t lrx0, uint64_t lrx1, uint64_t lrx2, uint64_t lrx3,
    uint64_t pfl0, uint64_t pfl1, uint64_t pfl2, uint64_t pfl3,
    uint64_t lfl0, uint64_t lfl1, uint64_t lfl2, uint64_t lfl3,
    uint32_t*    __restrict__ bar_scratch,
    float*       __restrict__ h_io,
    const float* __restrict__ w_rms,
    float*       __restrict__ x_norm_out,
    int8_t*      __restrict__ x_q8_out,
    float2*      __restrict__ x_ds_out,
    uint32_t M,
    uint32_t D,
    uint32_t epoch,
    uint32_t do_norm,
    uint32_t rank,
    uint32_t n_ranks,
    uint32_t pre_pushed,
    float    eps) {
    const uint32_t m    = blockIdx.x;
    const uint32_t tid  = threadIdx.x;
    const uint32_t wave = tid >> 6;
    const uint32_t lane = tid & 63u;
    if (m >= M) return;

    __shared__ float sm_red[4];
    const uint64_t off = static_cast<uint64_t>(m) * D;
    const float4*  xp4 = reinterpret_cast<const float4*>(x_partial + off);
    const uint32_t n8  = D >> 3;

    auto send_to_peer = [&](uint32_t dst, uint64_t ptx_va, uint32_t i, const vegajet::ornith_dev::u32x4& pack) {
        if (dst < n_ranks && dst != rank && ptx_va != 0ULL) {
            __builtin_nontemporal_store(pack, reinterpret_cast<vegajet::ornith_dev::u32x4*>(ptx_va + off * 2ULL) + i);
        }
    };
    auto set_peer_flag = [&](uint32_t dst, uint64_t pfl_va) {
        if (dst < n_ranks && dst != rank && pfl_va != 0ULL) {
            __atomic_store_n(reinterpret_cast<uint32_t*>(pfl_va), epoch, __ATOMIC_RELEASE);
        }
    };
    auto wait_peer_flag = [&](uint32_t src, uint64_t lfl_va) {
        if (src < n_ranks && src != rank && lfl_va != 0ULL) {
            const uint32_t* fp = reinterpret_cast<const uint32_t*>(lfl_va);
            while (__atomic_load_n(fp, __ATOMIC_ACQUIRE) != epoch) {
                __builtin_amdgcn_s_sleep(1);
            }
        }
    };

    if (pre_pushed == 0u && n_ranks > 1u) {
        using U4 = vegajet::ornith_dev::u32x4;
        #pragma nounroll
        for (uint32_t i = tid; i < n8; i += 256u) {
            const float4 a = xp4[i * 2u];
            const float4 b = xp4[i * 2u + 1u];
            const __half2 h0 = __float22half2_rn(make_float2(a.x, a.y));
            const __half2 h1 = __float22half2_rn(make_float2(a.z, a.w));
            const __half2 h2 = __float22half2_rn(make_float2(b.x, b.y));
            const __half2 h3 = __float22half2_rn(make_float2(b.z, b.w));
            U4 pack = {
                *reinterpret_cast<const uint32_t*>(&h0),
                *reinterpret_cast<const uint32_t*>(&h1),
                *reinterpret_cast<const uint32_t*>(&h2),
                *reinterpret_cast<const uint32_t*>(&h3)
            };
            send_to_peer(0u, ptx0, i, pack);
            send_to_peer(1u, ptx1, i, pack);
            send_to_peer(2u, ptx2, i, pack);
            send_to_peer(3u, ptx3, i, pack);
        }
        asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
        __threadfence_system();
        if (M > 1u) {
            vegajet::ornith_dev::grid_barrier_m(bar_scratch, M, epoch * 2u - 1u);
        } else {
            __syncthreads();
        }
    }

    if (n_ranks > 1u) {
        if (m == 0u && tid == 0u && pre_pushed != 4u) {
            __threadfence_system();
            // pre_pushed: 0 = push here + flag; 1 = super-kernel pushed AND flagged;
            //             2 = GEMM/MoE epilogue pushed (TP=3 verify), flag here;
            //             3 = super-kernel pushed with per-chunk ready tags (VEGAJET_AR_TILE_PIPE, M = 1): no flags.
            if (pre_pushed == 0u || pre_pushed == 2u) {
                set_peer_flag(0u, pfl0);
                set_peer_flag(1u, pfl1);
                set_peer_flag(2u, pfl2);
                set_peer_flag(3u, pfl3);
            }
            if (pre_pushed != 3u) wait_peer_flag(0u, lfl0);
            if (pre_pushed != 3u) wait_peer_flag(1u, lfl1);
            if (pre_pushed != 3u) wait_peer_flag(2u, lfl2);
            if (pre_pushed != 3u) wait_peer_flag(3u, lfl3);
            __threadfence_system();
        }
        if (M > 1u) {
            vegajet::ornith_dev::grid_barrier_m(bar_scratch, M, epoch * 2u);
        } else {
            __syncthreads();
        }
    }

    const float4* h4c = reinterpret_cast<const float4*>(h_io + off);
    float4*       h4  = reinterpret_cast<float4*>(h_io + off);
    float*  xn = x_norm_out ? (x_norm_out + off) : nullptr;
    int8_t* xq = x_q8_out + off;
    float2* xd = x_ds_out + static_cast<uint64_t>(m) * (D >> 5);

    auto accum_rank = [&](uint32_t r, uint64_t lrx_va, uint32_t ic, const float4& la, const float4& lb, float4& sa, float4& sb) {
        #pragma clang fp contract(off)
        if (r >= n_ranks) return;
        float4 va, vb;
        if (r == rank) {
            va = make_float4(__half2float(__float2half(la.x)), __half2float(__float2half(la.y)),
                             __half2float(__float2half(la.z)), __half2float(__float2half(la.w)));
            vb = make_float4(__half2float(__float2half(lb.x)), __half2float(__float2half(lb.y)),
                             __half2float(__float2half(lb.z)), __half2float(__float2half(lb.w)));
        } else {
            if (pre_pushed == 3u) {
                // Tag area = last 4 KiB of the 2 MiB RX slot (VJ_SUPER_AR_TAG_OFF in vj_ornith_super.hip), one u32 per chunk.
                const uint32_t* tp = reinterpret_cast<const uint32_t*>(lrx_va + (2ULL * 1048576ULL - 4096ULL)) + ic;
                #pragma nounroll
                for (uint32_t spin = 0u; spin < 5000000u; ++spin) {
                    if (__atomic_load_n(tp, __ATOMIC_ACQUIRE) == epoch) break;
                    __builtin_amdgcn_s_sleep(1);
                }
            }
            const uint4* lr4 = reinterpret_cast<const uint4*>(lrx_va + off * 2ULL);
            const uint4 rv = pre_pushed == 4u
                ? vegajet::ar_fused::load8(reinterpret_cast<const uint32_t*>(lrx_va + vegajet::ar_fused::kOffset) + ic * 8u, epoch)
                : vegajet::ornith_dev::ldg_u32x4_stream(lr4 + ic);
            const float2 p0  = __half22float2(*reinterpret_cast<const __half2*>(&rv.x));
            const float2 p1  = __half22float2(*reinterpret_cast<const __half2*>(&rv.y));
            const float2 p2  = __half22float2(*reinterpret_cast<const __half2*>(&rv.z));
            const float2 p3  = __half22float2(*reinterpret_cast<const __half2*>(&rv.w));
            va = make_float4(p0.x, p0.y, p1.x, p1.y);
            vb = make_float4(p2.x, p2.y, p3.x, p3.y);
        }
        if (r == 0u) { sa = va; sb = vb; }
        else {
            sa.x += va.x; sa.y += va.y; sa.z += va.z; sa.w += va.w;
            sb.x += vb.x; sb.y += vb.y; sb.z += vb.z; sb.w += vb.w;
        }
    };

    float4 ha[2], hb[2];
    float ss = 0.0f;
    #pragma unroll
    for (uint32_t it = 0u; it < 2u; ++it) {
        const uint32_t i = tid + 256u * it;
        const uint32_t ic = min(i, n8 - 1u);
        const float4 la = xp4[ic * 2u];
        const float4 lb = xp4[ic * 2u + 1u];
        ha[it] = h4c[ic * 2u];
        hb[it] = h4c[ic * 2u + 1u];
        if (n_ranks <= 1u) {
            ha[it].x += la.x; ha[it].y += la.y; ha[it].z += la.z; ha[it].w += la.w;
            hb[it].x += lb.x; hb[it].y += lb.y; hb[it].z += lb.z; hb[it].w += lb.w;
        } else {
            float4 sa = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 sb = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            accum_rank(0u, lrx0, ic, la, lb, sa, sb);
            accum_rank(1u, lrx1, ic, la, lb, sa, sb);
            accum_rank(2u, lrx2, ic, la, lb, sa, sb);
            accum_rank(3u, lrx3, ic, la, lb, sa, sb);
            ha[it].x += sa.x; ha[it].y += sa.y; ha[it].z += sa.z; ha[it].w += sa.w;
            hb[it].x += sb.x; hb[it].y += sb.y; hb[it].z += sb.z; hb[it].w += sb.w;
        }
        if (i < n8) {
            h4[i * 2u]      = ha[it];
            h4[i * 2u + 1u] = hb[it];
            ss = fmaf(ha[it].x, ha[it].x, fmaf(ha[it].y, ha[it].y, fmaf(ha[it].z, ha[it].z, fmaf(ha[it].w, ha[it].w, ss))));
            ss = fmaf(hb[it].x, hb[it].x, fmaf(hb[it].y, hb[it].y, fmaf(hb[it].z, hb[it].z, fmaf(hb[it].w, hb[it].w, ss))));
        }
    }
    if (do_norm == 0u) return;

    ss = vegajet::ornith_dev::wave64_sum(ss);
    if (lane == 0u) sm_red[wave] = ss;
    __syncthreads();
    const float inv_rms = rsqrtf((sm_red[0] + sm_red[1] + sm_red[2] + sm_red[3]) / static_cast<float>(D) + eps);

    const float4* w4 = reinterpret_cast<const float4*>(w_rms);
    #pragma unroll
    for (uint32_t it = 0u; it < 2u; ++it) {
        #pragma clang fp contract(off)
        const uint32_t i  = tid + 256u * it;
        const uint32_t ic = min(i, n8 - 1u);
        const float4 wa = w4[ic * 2u];
        const float4 wb = w4[ic * 2u + 1u];
        float y[8], hs[8], ws[8];
        hs[0] = ha[it].x * inv_rms; hs[1] = ha[it].y * inv_rms; hs[2] = ha[it].z * inv_rms; hs[3] = ha[it].w * inv_rms;
        hs[4] = hb[it].x * inv_rms; hs[5] = hb[it].y * inv_rms; hs[6] = hb[it].z * inv_rms; hs[7] = hb[it].w * inv_rms;
        ws[0] = wa.x; ws[1] = wa.y; ws[2] = wa.z; ws[3] = wa.w;
        ws[4] = wb.x; ws[5] = wb.y; ws[6] = wb.z; ws[7] = wb.w;
        float amax = 0.0f, t8[8];
        #pragma unroll
        for (uint32_t e = 0u; e < 8u; ++e) {
            y[e] = hs[e] * ws[e];
            amax = fmaxf(amax, fabsf(y[e]));
            float v = y[e];
            v += __shfl_xor(v, 2, 4);
            v += __shfl_xor(v, 1, 4);
            t8[e] = v;
        }
        amax = fmaxf(amax, __shfl_xor(amax, 2, 4));
        amax = fmaxf(amax, __shfl_xor(amax, 1, 4));
        const float bsum  = (((t8[0] + t8[4]) + (t8[2] + t8[6])) + ((t8[1] + t8[5]) + (t8[3] + t8[7])));
        const float d     = amax * (1.0f / 127.0f);
        const float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        uint32_t q[2] = {0u, 0u};
        #pragma unroll
        for (uint32_t e = 0u; e < 8u; ++e)
            q[e >> 2] |= (static_cast<uint32_t>(max(-127, min(127, __float2int_rn(y[e] * inv_d)))) & 0xFFu) << ((e & 3u) * 8u);
        if (i < n8) {
            if (xn) {
                reinterpret_cast<float4*>(xn)[i * 2u]      = make_float4(y[0], y[1], y[2], y[3]);
                reinterpret_cast<float4*>(xn)[i * 2u + 1u] = make_float4(y[4], y[5], y[6], y[7]);
            }
            reinterpret_cast<uint2*>(xq)[i] = make_uint2(q[0], q[1]);
            if ((tid & 3u) == 0u) xd[i >> 2] = make_float2(d, bsum);
        }
    }
}

