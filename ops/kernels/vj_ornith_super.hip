// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// VegaJet Wave 5 Persistent M=1 Super-Kernels for gfx906 (2x MI50):
//   - Collapses 291 -> 100 dispatches/token (<=104 target): 2 super-kernels/layer!
//   - Strictly 0 B private spill, <= 64 VGPRs (4 WGs/CU = 240 resident WGs), < 24,576 B .text, lds_dword = 0.
//   - Hardware coherency: s_waitcnt vmcnt(0); __threadfence_system(); before exit on P2P prepush paths;
//     full buffer_wbinvl1 after every grid_barrier_m boundary.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#include "core/ggml_iq_tables.hpp"
#include "ops/kernels/vj_moe_dot8.hiph"
#include "ops/kernels/vj_ornith_super_desc.hpp"

#define VJ_SUPER_RING_SLOTS 17u
#define VJ_SUPER_AR_SLOT_BYTES (2ULL * 1024ULL * 1024ULL)

namespace vegajet {
namespace super_dev {

typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
template <typename T>
using GPtr = T __attribute__((address_space(1)))*;
template <typename T>
__device__ __forceinline__ GPtr<T> as_gptr(uint64_t va) {
    return reinterpret_cast<GPtr<T>>(va);
}

__device__ __forceinline__ int32_t sdot4(uint32_t a, uint32_t b, int32_t c) {
    return __builtin_amdgcn_sdot4(static_cast<int32_t>(a), static_cast<int32_t>(b), c, false);
}

__device__ __forceinline__ float fp16_to_f32(uint16_t h) {
    return __half2float(__ushort_as_half(h));
}

__device__ __forceinline__ uint4 ldg_u32x4_stream(GPtr<const uint4> ptr) {
    u32x4 v = __builtin_nontemporal_load(reinterpret_cast<GPtr<const u32x4>>(ptr));
    return make_uint4(v[0], v[1], v[2], v[3]);
}

__device__ __forceinline__ uint32_t ldg_u32_stream(GPtr<const uint32_t> ptr) {
    return __builtin_nontemporal_load(ptr);
}

__device__ __forceinline__ float ldg_f32_stream(GPtr<const float> ptr) {
    return __builtin_nontemporal_load(ptr);
}

__device__ __forceinline__ void stg_f32_stream(GPtr<float> ptr, float v) {
    __builtin_nontemporal_store(v, ptr);
}

// Bit-exact hybrid DPP (xor 1, 2: 0 LDS!) + ds_swizzle (xor 4, 8, 16) + shfl_xor 32 wave64 sum.
__device__ __forceinline__ float wave64_sum(float v) {
    v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0xB1, 0xF, 0xF, true));
    v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x4E, 0xF, 0xF, true));
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x101F));
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x201F));
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x401F));
    v += __shfl_xor(v, 32, 64);
    return v;
}

__device__ __forceinline__ float block32_sum(float v) {
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x401F));
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x201F));
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x101F));
    v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x4E, 0xF, 0xF, true));
    v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0xB1, 0xF, 0xF, true));
    return v;
}

__device__ __forceinline__ float block32_max(float v) {
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x401F)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x201F)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x101F)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x4E, 0xF, 0xF, true)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0xB1, 0xF, 0xF, true)));
    return v;
}

__device__ __forceinline__ uint32_t iq2s_perm4(uint32_t mag4_sel, uint32_t sign4) {
    const uint32_t sel = (mag4_sel & 0x03030303u) | (__umul24(sign4, 0x00810204u) & 0x04040404u);
    return __builtin_amdgcn_perm(0x00D5E7F8u, 0x002B1908u, sel);
}

__device__ __forceinline__ void signal_out_peer_flag_n(
    GPtr<uint32_t>  sbar,
    const uint64_t* out_peer_flag,
    uint32_t        rank,
    uint32_t        n_ranks,
    uint32_t        n_active_wgs,
    uint32_t        out_ar_epoch) {
    if (n_ranks <= 1u) return;
    asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0u) {
        const uint32_t arr = __atomic_fetch_add(&sbar[160], 1u, __ATOMIC_RELAXED) + 1u;
        if (arr == n_active_wgs) {
            __atomic_store_n(&sbar[160], 0u, __ATOMIC_RELAXED);
            __threadfence_system();
            #pragma nounroll
            for (uint32_t dst = 0u; dst < n_ranks; ++dst) {
                if (dst != rank && out_peer_flag[dst] != 0ULL) {
                    __atomic_store_n(as_gptr<uint32_t>(out_peer_flag[dst]), out_ar_epoch, __ATOMIC_RELEASE);
                }
            }
        }
    }
}

__device__ __forceinline__ void prof_acc(GPtr<uint64_t> prof, uint32_t slot, uint64_t& t_prev, bool on) {
    if (on) {
        const uint64_t now = __builtin_amdgcn_s_memrealtime();
        prof[slot] += (now - t_prev);
        t_prev = now;
    }
}

__device__ __forceinline__ uint32_t load_u32_glc(GPtr<const uint32_t> ptr) {
    uint32_t v;
    asm volatile("global_load_dword %0, %1, off glc slc\n\ts_waitcnt vmcnt(0)\n"
                 : "=v"(v) : "v"(ptr) : "memory");
    return v;
}

__device__ __forceinline__ void grid_barrier_glc_only(GPtr<uint32_t> bar, uint32_t n_wgs, uint32_t sense) {
    asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
    __syncthreads();
    if (threadIdx.x == 0u) {
        const uint32_t arr = __atomic_fetch_add(&bar[0], 1u, __ATOMIC_RELAXED) + 1u;
        if (arr == n_wgs) {
            __atomic_store_n(&bar[0], 0u, __ATOMIC_RELAXED);
            asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
            __atomic_store_n(&bar[16], sense, __ATOMIC_RELEASE);
        } else {
            #pragma nounroll
            for (uint32_t spin = 0u; spin < 5000000u; ++spin) {
                if (load_u32_glc(&bar[16]) == sense) break;
                __builtin_amdgcn_s_sleep(1);
            }
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void grid_barrier_wbinvl1(GPtr<uint32_t> bar, uint32_t n_wgs, uint32_t sense) {
    asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
    __syncthreads();
    if (threadIdx.x == 0u) {
        const uint32_t arr = __atomic_fetch_add(&bar[0], 1u, __ATOMIC_RELAXED) + 1u;
        if (arr == n_wgs) {
            __atomic_store_n(&bar[0], 0u, __ATOMIC_RELAXED);
            asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
            __atomic_store_n(&bar[16], sense, __ATOMIC_RELEASE);
        } else {
            #pragma nounroll
            for (uint32_t spin = 0u; spin < 5000000u; ++spin) {
                if (load_u32_glc(&bar[16]) == sense) break;
                __builtin_amdgcn_s_sleep(1);
            }
        }
        asm volatile("buffer_wbinvl1\n" ::: "memory");
    }
    __syncthreads();
}

template<int Off>
__device__ __forceinline__ uint32_t xor_shfl32_fast(uint32_t v) {
    if constexpr (Off == 32) return __shfl_xor(v, 32, 64);
    else return static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int>(v), (Off << 10) | 0x1F));
}

template<int Off>
__device__ __forceinline__ float xor_shfl_f32_fast(float v) {
    return __uint_as_float(xor_shfl32_fast<Off>(__float_as_uint(v)));
}

template<int Off>
__device__ __forceinline__ uint64_t step_max_u64(uint64_t bk) {
    uint32_t lo = xor_shfl32_fast<Off>(static_cast<uint32_t>(bk));
    uint32_t hi = xor_shfl32_fast<Off>(static_cast<uint32_t>(bk >> 32));
    uint64_t ok = (static_cast<uint64_t>(hi) << 32) | lo;
    return (ok > bk) ? ok : bk;
}

// Pure-DPP zero-LDS wave64 max (returns uniform max across all 64 lanes via v_readlane_b32 on lanes 15,31,47,63).
__device__ __forceinline__ float wave64_max_dpp(float v) {
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0xB1, 0xF, 0xF, true)));   // quad_perm [1,0,3,2]
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x4E, 0xF, 0xF, true)));   // quad_perm [2,3,0,1]
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x114, 0xF, 0xF, false))); // row_shr:4
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x118, 0xF, 0xF, false))); // row_shr:8
    const float m0 = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 15));
    const float m1 = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 31));
    const float m2 = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 47));
    const float m3 = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 63));
    return fmaxf(fmaxf(m0, m1), fmaxf(m2, m3));
}

__device__ __forceinline__ void fast_softmax_top8_wave64(
    GPtr<const float> __restrict__ logits,
    int32_t*          __restrict__ top_ids,
    float*            __restrict__ top_w,
    uint32_t l) {
    // 1 coalesced 128-bit float4 load per lane (4x fewer L2 requests across 240 WGs vs 4 scalar loads!).
    const float4 xv = reinterpret_cast<GPtr<const float4>>(logits)[l];
    float x0 = xv.x, x1 = xv.y, x2 = xv.z, x3 = xv.w;
    float lm = fmaxf(fmaxf(x0, x1), fmaxf(x2, x3));
    float mx = 0.0f;
    #pragma nounroll
    for (uint32_t r = 0u; r < 8u; ++r) {
        const float best = wave64_max_dpp(lm);
        if (r == 0u) mx = best;
        const uint64_t m0 = __ballot(x0 == best);
        const uint64_t m1 = __ballot(x1 == best);
        const uint64_t m2 = __ballot(x2 == best);
        const uint64_t m3 = __ballot(x3 == best);
        const uint64_t ma = m0 | m1 | m2 | m3;
        const uint32_t wl = static_cast<uint32_t>(__ffsll(static_cast<long long>(ma)) - 1);
        const uint32_t wq = ((m0 >> wl) & 1ULL) ? 0u
                          : (((m1 >> wl) & 1ULL) ? 1u
                          : (((m2 >> wl) & 1ULL) ? 2u : 3u));
        const uint32_t win_id = wl * 4u + wq;
        if (l == wl) {
            if (wq == 0u) x0 = -1e30f; else if (wq == 1u) x1 = -1e30f; else if (wq == 2u) x2 = -1e30f; else x3 = -1e30f;
            lm = fmaxf(fmaxf(x0, x1), fmaxf(x2, x3));
        }
        if (l == 0u) {
            top_ids[r] = static_cast<int32_t>(win_id);
            top_w[r]   = best;
        }
    }
    if (l < 8u) {
        const float ev = __expf(top_w[l] - mx);
        top_w[l] = ev;
    }
    if (l == 0u) {
        float top_sum = 0.0f;
        #pragma unroll
        for (uint32_t r = 0u; r < 8u; ++r) top_sum += top_w[r];
        const float inv_w = 1.0f / fmaxf(top_sum, 1e-30f);
        #pragma unroll
        for (uint32_t r = 0u; r < 8u; ++r) top_w[r] *= inv_w;
    }
}

// Bit-identical pre-pushed FP16 P2P AllReduce (N=1..4) + Residual + RMSNorm + Q8 Quant (WG 0 only, M == 1).
// Canonical rank-order accumulation (0, 1, ..., n_ranks-1) guarantees bit-identical h/x_norm/x_q8/x_ds on every rank.
__device__ __forceinline__ void p2p_ar_prepush_m1_n(
    GPtr<const float> __restrict__ x_partial,
    const uint64_t*                ar_local_rx_base,
    const uint64_t*                ar_local_flag,
    uint64_t                       slot_off,
    GPtr<float>       __restrict__ h_io,
    GPtr<const float> __restrict__ w_rms,
    GPtr<float>       __restrict__ x_norm_out,
    GPtr<int8_t>      __restrict__ x_q8_out,
    GPtr<float2>      __restrict__ x_ds_out,
    float*                         sm_red4,
    uint32_t D,
    uint32_t epoch,
    uint32_t rank,
    uint32_t n_ranks,
    float    eps) {
    const uint32_t tid  = threadIdx.x;
    const uint32_t wave = tid >> 6;
    const uint32_t lane = tid & 63u;
    const uint32_t n8   = D >> 3;

    if (n_ranks > 1u) {
        if (tid == 0u) {
            #pragma nounroll
            for (uint32_t src = 0u; src < n_ranks; ++src) {
                if (src != rank && ar_local_flag[src] != 0ULL) {
                    GPtr<const uint32_t> fp = as_gptr<const uint32_t>(ar_local_flag[src]);
                    #pragma nounroll
                    for (uint32_t spin = 0u; spin < 5000000u; ++spin) {
                        if (load_u32_glc(fp) == epoch) break;
                        __builtin_amdgcn_s_sleep(1);
                    }
                }
            }
        }
        __syncthreads();
    }

    GPtr<const float4> xp4 = reinterpret_cast<GPtr<const float4>>(x_partial);
    GPtr<const float4> h4c = reinterpret_cast<GPtr<const float4>>(h_io);
    GPtr<float4>       h4  = reinterpret_cast<GPtr<float4>>(h_io);

    float4 ha[2], hb[2];
    float ss = 0.0f;
    #pragma unroll
    for (uint32_t it = 0u; it < 2u; ++it) {
        const uint32_t i  = tid + 256u * it;
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
            #pragma nounroll
            for (uint32_t r = 0u; r < n_ranks; ++r) {
                #pragma clang fp contract(off)
                float4 va, vb;
                if (r == rank) {
                    va = make_float4(__half2float(__float2half(la.x)), __half2float(__float2half(la.y)),
                                     __half2float(__float2half(la.z)), __half2float(__float2half(la.w)));
                    vb = make_float4(__half2float(__float2half(lb.x)), __half2float(__float2half(lb.y)),
                                     __half2float(__float2half(lb.z)), __half2float(__float2half(lb.w)));
                } else {
                    const uint4 rv = ldg_u32x4_stream(as_gptr<const uint4>(ar_local_rx_base[r] + slot_off) + ic);
                    const float2 p0 = __half22float2(*reinterpret_cast<const __half2*>(&rv.x));
                    const float2 p1 = __half22float2(*reinterpret_cast<const __half2*>(&rv.y));
                    const float2 p2 = __half22float2(*reinterpret_cast<const __half2*>(&rv.z));
                    const float2 p3 = __half22float2(*reinterpret_cast<const __half2*>(&rv.w));
                    va = make_float4(p0.x, p0.y, p1.x, p1.y);
                    vb = make_float4(p2.x, p2.y, p3.x, p3.y);
                }
                if (r == 0u) { sa = va; sb = vb; }
                else {
                    sa.x += va.x; sa.y += va.y; sa.z += va.z; sa.w += va.w;
                    sb.x += vb.x; sb.y += vb.y; sb.z += vb.z; sb.w += vb.w;
                }
            }
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

    ss = wave64_sum(ss);
    if (lane == 0u) sm_red4[wave] = ss;
    __syncthreads();
    const float inv_rms = rsqrtf((sm_red4[0] + sm_red4[1] + sm_red4[2] + sm_red4[3]) / static_cast<float>(D) + eps);

    GPtr<const float4> w4 = reinterpret_cast<GPtr<const float4>>(w_rms);
    float4 wa[2], wb[2];
    #pragma unroll
    for (uint32_t it = 0u; it < 2u; ++it) {
        const uint32_t i = min(tid + 256u * it, n8 - 1u);
        wa[it] = w4[i * 2u];
        wb[it] = w4[i * 2u + 1u];
    }
    __builtin_amdgcn_sched_barrier(0);
    #pragma unroll
    for (uint32_t it = 0u; it < 2u; ++it) {
        #pragma clang fp contract(off)
        const uint32_t i = tid + 256u * it;
        float y[8], hs[8], ws[8];
        hs[0] = ha[it].x * inv_rms; hs[1] = ha[it].y * inv_rms; hs[2] = ha[it].z * inv_rms; hs[3] = ha[it].w * inv_rms;
        hs[4] = hb[it].x * inv_rms; hs[5] = hb[it].y * inv_rms; hs[6] = hb[it].z * inv_rms; hs[7] = hb[it].w * inv_rms;
        ws[0] = wa[it].x; ws[1] = wa[it].y; ws[2] = wa[it].z; ws[3] = wa[it].w;
        ws[4] = wb[it].x; ws[5] = wb[it].y; ws[6] = wb[it].z; ws[7] = wb[it].w;
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
            if (x_norm_out) {
                reinterpret_cast<GPtr<float4>>(x_norm_out)[i * 2u]      = make_float4(y[0], y[1], y[2], y[3]);
                reinterpret_cast<GPtr<float4>>(x_norm_out)[i * 2u + 1u] = make_float4(y[4], y[5], y[6], y[7]);
            }
            reinterpret_cast<GPtr<uint2>>(x_q8_out)[i] = make_uint2(q[0], q[1]);
            if ((tid & 3u) == 0u) x_ds_out[i >> 2] = make_float2(d, bsum);
        }
    }
}

// Persistent grid-strided VJ_Q4_A8 GEMV helper (exact bit-identical to ornith_gemv_q4a8_m1_w32_kernel!).
// Caches x_q8 + x_ds in LDS ONCE per WG; full-batch weight loads + sched_barrier(0) for nb==96 and nb==128.
__device__ __forceinline__ void gemv_q4a8_persistent(
    GPtr<const uint8_t>  __restrict__ w_qs,
    GPtr<const uint32_t> __restrict__ w_dm,
    GPtr<const int8_t>   __restrict__ x_q8,
    GPtr<const float2>   __restrict__ x_ds,
    GPtr<float>          __restrict__ y_out,
    const uint64_t*                   out_peer_tx_base,
    uint64_t                          slot_off,
    uint32_t                          rank,
    uint32_t                          n_ranks,
    bool                              do_prepush,
    uint4*               sm_xq,
    float2*              sm_xds,
    float*               sm_out8,
    uint32_t rows,
    uint32_t cols,
    uint32_t qs_stride,
    uint32_t dm_stride) {
    const uint32_t tid   = threadIdx.x;
    const uint32_t grp   = tid >> 5;
    const uint32_t lane  = tid & 31u;
    const uint32_t nb    = cols >> 5;
    const uint32_t dm_st = dm_stride ? dm_stride : nb;
    const uint32_t n_rg  = (rows + 7u) >> 3;
    GPtr<const uint4> x4 = reinterpret_cast<GPtr<const uint4>>(x_q8);
    if (tid < (nb << 1)) sm_xq[tid]  = x4[tid];
    if (tid < nb)        sm_xds[tid] = x_ds[tid];
    __syncthreads();

    // Untyped MUBUF SRDs via LLVM builtins (32-bit offen glc slc, compiler-tracked vmcnt, LDS=0!)
    auto make_srd = [](const void* p) -> __amdgpu_buffer_rsrc_t {
        return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(p), static_cast<short>(0), 0x7FFFFFFF, 0x00027000);
    };
    auto mubuf_x4 = [](__amdgpu_buffer_rsrc_t rsrc, uint32_t voff, uint32_t soff) -> uint4 {
        return __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(rsrc, voff, soff, 3));
    };
    auto mubuf_u32 = [](__amdgpu_buffer_rsrc_t rsrc, uint32_t voff, uint32_t soff) -> uint32_t {
        return __builtin_amdgcn_raw_buffer_load_b32(rsrc, voff, soff, 3);
    };
    const __amdgpu_buffer_rsrc_t srd_w  = make_srd(reinterpret_cast<const void*>(w_qs));
    const __amdgpu_buffer_rsrc_t srd_dm = make_srd(reinterpret_cast<const void*>(w_dm));
    const uint32_t dm_row_bytes = dm_st * 4u;

    uint32_t iter = 0u;
    #pragma nounroll
    for (uint32_t rg = blockIdx.x; rg < n_rg; rg += gridDim.x) {
        const uint32_t row = rg * 8u + grp;
        float acc = 0.0f;
        if (row < rows) {
            const uint32_t w_voff  = row * qs_stride + lane * 16u;
            const uint32_t dm_voff = row * dm_row_bytes + lane * 4u;
            if (nb == 128u) {
                uint4    wvs[4];
                uint32_t dms[4];
                #pragma unroll
                for (uint32_t k = 0u; k < 4u; ++k) {
                    wvs[k] = mubuf_x4(srd_w,  w_voff,  k * 512u);
                    dms[k] = mubuf_u32(srd_dm, dm_voff, k * 128u);
                }
                __builtin_amdgcn_sched_barrier(0);
                #pragma unroll
                for (uint32_t k = 0u; k < 4u; ++k) {
                    const uint32_t b   = lane + k * 32u;
                    const uint4    wv  = wvs[k];
                    const uint32_t dmv = dms[k];
                    const uint4    xa  = sm_xq[b * 2u];
                    const uint4    xb  = sm_xq[b * 2u + 1u];
                    const float2   xds = sm_xds[b];
                    int32_t dp = sdot4(wv.x & 0x0F0F0F0Fu, xa.x, 0);
                    dp         = sdot4(wv.y & 0x0F0F0F0Fu, xa.y, dp);
                    dp         = sdot4(wv.z & 0x0F0F0F0Fu, xa.z, dp);
                    dp         = sdot4(wv.w & 0x0F0F0F0Fu, xa.w, dp);
                    dp         = sdot4((wv.x >> 4) & 0x0F0F0F0Fu, xb.x, dp);
                    dp         = sdot4((wv.y >> 4) & 0x0F0F0F0Fu, xb.y, dp);
                    dp         = sdot4((wv.z >> 4) & 0x0F0F0F0Fu, xb.z, dp);
                    dp         = sdot4((wv.w >> 4) & 0x0F0F0F0Fu, xb.w, dp);
                    const float wd = fp16_to_f32(static_cast<uint16_t>(dmv & 0xFFFFu));
                    const float wm = fp16_to_f32(static_cast<uint16_t>(dmv >> 16));
                    acc = fmaf(wd * xds.x, static_cast<float>(dp), fmaf(wm, xds.y, acc));
                }
            } else if (nb == 96u) {
                uint4    wvs[3];
                uint32_t dms[3];
                #pragma unroll
                for (uint32_t k = 0u; k < 3u; ++k) {
                    wvs[k] = mubuf_x4(srd_w,  w_voff,  k * 512u);
                    dms[k] = mubuf_u32(srd_dm, dm_voff, k * 128u);
                }
                __builtin_amdgcn_sched_barrier(0);
                #pragma unroll
                for (uint32_t k = 0u; k < 3u; ++k) {
                    const uint32_t b   = lane + k * 32u;
                    const uint4    wv  = wvs[k];
                    const uint32_t dmv = dms[k];
                    const uint4    xa  = sm_xq[b * 2u];
                    const uint4    xb  = sm_xq[b * 2u + 1u];
                    const float2   xds = sm_xds[b];
                    int32_t dp = sdot4(wv.x & 0x0F0F0F0Fu, xa.x, 0);
                    dp         = sdot4(wv.y & 0x0F0F0F0Fu, xa.y, dp);
                    dp         = sdot4(wv.z & 0x0F0F0F0Fu, xa.z, dp);
                    dp         = sdot4(wv.w & 0x0F0F0F0Fu, xa.w, dp);
                    dp         = sdot4((wv.x >> 4) & 0x0F0F0F0Fu, xb.x, dp);
                    dp         = sdot4((wv.y >> 4) & 0x0F0F0F0Fu, xb.y, dp);
                    dp         = sdot4((wv.z >> 4) & 0x0F0F0F0Fu, xb.z, dp);
                    dp         = sdot4((wv.w >> 4) & 0x0F0F0F0Fu, xb.w, dp);
                    const float wd = fp16_to_f32(static_cast<uint16_t>(dmv & 0xFFFFu));
                    const float wm = fp16_to_f32(static_cast<uint16_t>(dmv >> 16));
                    acc = fmaf(wd * xds.x, static_cast<float>(dp), fmaf(wm, xds.y, acc));
                }
            } else if (nb == 80u) {
                // TP=3 GPU1/2 exact fast path (cols=2560, 0 clamped block-79 re-loads!)
                uint4    wvs[3];
                uint32_t dms[3];
                wvs[0] = mubuf_x4(srd_w,  w_voff,  0u);
                dms[0] = mubuf_u32(srd_dm, dm_voff, 0u);
                wvs[1] = mubuf_x4(srd_w,  w_voff,  512u);
                dms[1] = mubuf_u32(srd_dm, dm_voff, 128u);
                if (lane < 16u) {
                    wvs[2] = mubuf_x4(srd_w,  w_voff,  1024u);
                    dms[2] = mubuf_u32(srd_dm, dm_voff, 256u);
                }
                __builtin_amdgcn_sched_barrier(0);
                #pragma unroll
                for (uint32_t k = 0u; k < 3u; ++k) {
                    const uint32_t b = lane + k * 32u;
                    if (k < 2u || lane < 16u) {
                        const uint4    wv  = wvs[k];
                        const uint32_t dmv = dms[k];
                        const uint4    xa  = sm_xq[b * 2u];
                        const uint4    xb  = sm_xq[b * 2u + 1u];
                        const float2   xds = sm_xds[b];
                        int32_t dp = sdot4(wv.x & 0x0F0F0F0Fu, xa.x, 0);
                        dp         = sdot4(wv.y & 0x0F0F0F0Fu, xa.y, dp);
                        dp         = sdot4(wv.z & 0x0F0F0F0Fu, xa.z, dp);
                        dp         = sdot4(wv.w & 0x0F0F0F0Fu, xa.w, dp);
                        dp         = sdot4((wv.x >> 4) & 0x0F0F0F0Fu, xb.x, dp);
                        dp         = sdot4((wv.y >> 4) & 0x0F0F0F0Fu, xb.y, dp);
                        dp         = sdot4((wv.z >> 4) & 0x0F0F0F0Fu, xb.z, dp);
                        dp         = sdot4((wv.w >> 4) & 0x0F0F0F0Fu, xb.w, dp);
                        const float wd = fp16_to_f32(static_cast<uint16_t>(dmv & 0xFFFFu));
                        const float wm = fp16_to_f32(static_cast<uint16_t>(dmv >> 16));
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
                        const uint32_t bc = min(b0 + k * 32u, b_last);
                        wvs[k] = mubuf_x4(srd_w,  row * qs_stride + bc * 16u, 0u);
                        dms[k] = mubuf_u32(srd_dm, row * dm_row_bytes + bc * 4u, 0u);
                    }
                    __builtin_amdgcn_sched_barrier(0);
                    #pragma unroll
                    for (uint32_t k = 0u; k < 3u; ++k) {
                        const uint32_t b = b0 + k * 32u;
                        if (b < nb) {
                            const uint4    wv  = wvs[k];
                            const uint32_t dmv = dms[k];
                            const uint4    xa  = sm_xq[b * 2u];
                            const uint4    xb  = sm_xq[b * 2u + 1u];
                            const float2   xds = sm_xds[b];
                            int32_t dp = sdot4(wv.x & 0x0F0F0F0Fu, xa.x, 0);
                            dp         = sdot4(wv.y & 0x0F0F0F0Fu, xa.y, dp);
                            dp         = sdot4(wv.z & 0x0F0F0F0Fu, xa.z, dp);
                            dp         = sdot4(wv.w & 0x0F0F0F0Fu, xa.w, dp);
                            dp         = sdot4((wv.x >> 4) & 0x0F0F0F0Fu, xb.x, dp);
                            dp         = sdot4((wv.y >> 4) & 0x0F0F0F0Fu, xb.y, dp);
                            dp         = sdot4((wv.z >> 4) & 0x0F0F0F0Fu, xb.z, dp);
                            dp         = sdot4((wv.w >> 4) & 0x0F0F0F0Fu, xb.w, dp);
                            const float wd = fp16_to_f32(static_cast<uint16_t>(dmv & 0xFFFFu));
                            const float wm = fp16_to_f32(static_cast<uint16_t>(dmv >> 16));
                            acc = fmaf(wd * xds.x, static_cast<float>(dp), fmaf(wm, xds.y, acc));
                        }
                    }
                }
            }
            acc = block32_sum(acc);
            if (lane == 0u) y_out[row] = acc;
        }
        if (do_prepush && n_ranks > 1u && (rows & 7u) == 0u) {
            float* s8 = sm_out8 + ((iter & 1u) << 3);
            if (lane == 0u) s8[grp] = acc;
            __syncthreads();
            if (tid == 0u) {
                const __half2 h0 = __float22half2_rn(make_float2(s8[0], s8[1]));
                const __half2 h1 = __float22half2_rn(make_float2(s8[2], s8[3]));
                const __half2 h2 = __float22half2_rn(make_float2(s8[4], s8[5]));
                const __half2 h3 = __float22half2_rn(make_float2(s8[6], s8[7]));
                u32x4 pv = {
                    *reinterpret_cast<const uint32_t*>(&h0),
                    *reinterpret_cast<const uint32_t*>(&h1),
                    *reinterpret_cast<const uint32_t*>(&h2),
                    *reinterpret_cast<const uint32_t*>(&h3)
                };
                #pragma nounroll
                for (uint32_t dst = 0u; dst < n_ranks; ++dst) {
                    if (dst != rank && out_peer_tx_base[dst] != 0ULL) {
                        __builtin_nontemporal_store(pv, reinterpret_cast<GPtr<u32x4>>(out_peer_tx_base[dst] + slot_off) + rg);
                    }
                }
            }
        }
        ++iter;
    }
}

__device__ __forceinline__ void divmod_slot_u(uint32_t tile, uint32_t nblk16, uint32_t& slot, uint32_t& t_in_slot) {
    uint32_t rem = __builtin_amdgcn_readfirstlane(tile);
    const uint32_t nb = __builtin_amdgcn_readfirstlane(nblk16);
    uint32_t s = 0u;
    #pragma nounroll
    while (rem >= nb) { rem -= nb; ++s; }
    slot = s;
    t_in_slot = rem;
}

// Helper for 16-row IQ2_S Gate+Up tile using 4-KiB sm_grid32 + v_perm_b32 -> writes SiLU*Up to LDS sm_y (bit-exact!).
// Batches 2 K-iterations (6 global_load saddr ops) in flight with uniform_u64 SGPR base + sched_barrier(0).
__device__ __forceinline__ void moe_gu_16r_tile_g32_sm(
    GPtr<const uint8_t> __restrict__ wgu_merged_b,
    const uint32_t*     sm_grid32,
    const uint4*        sm_xq,
    const float*        sm_xd,
    const int32_t*      sm_ids,
    float*              sm_y_dst,
    uint32_t tile_idx,
    uint32_t local_ff,
    uint32_t nblk16) {
    const uint32_t sub_id = threadIdx.x >> 4;
    const uint32_t lane16 = threadIdx.x & 15u;
    uint32_t slot, t_in_slot;
    divmod_slot_u(tile_idx, nblk16, slot, t_in_slot);
    const uint32_t rloc   = t_in_slot * 16u + sub_id;
    const uint32_t exp_id = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(sm_ids[slot]));
    GPtr<const uint8_t> eb = wgu_merged_b + vegajet::dot8_dev::uniform_u64(static_cast<uint64_t>(exp_id) * local_ff * 1968ULL);
    const uint32_t r_off  = rloc * 1968u;
    // Preload all 12 FP16 scale pairs (48 B) ONCE across lanes 0..11 of each 16-lane group!
    const uint32_t d2_mine = (lane16 < 12u)
        ? ldg_u32_stream(reinterpret_cast<GPtr<const uint32_t>>(eb + r_off + 1920u + lane16 * 4u))
        : 0u;

    float ag = 0.0f, au = 0.0f;
    #pragma nounroll
    for (uint32_t it0 = 0u; it0 < 6u; it0 += 2u) {
        uint4    wq[2];
        uint32_t hs2[2];
        #pragma unroll
        for (uint32_t k = 0u; k < 2u; ++k) {
            const uint32_t ib = (it0 + k) * 16u + lane16;
            uint32_t q_off = r_off + ib * 16u;
            uint32_t h_off = r_off + 1536u + ib * 4u;
            asm volatile("" : "+v"(q_off), "+v"(h_off));
            wq[k]  = ldg_u32x4_stream(reinterpret_cast<GPtr<const uint4>>(eb + q_off));
            hs2[k] = ldg_u32_stream(reinterpret_cast<GPtr<const uint32_t>>(eb + h_off));
        }
        __builtin_amdgcn_sched_barrier(0);
        #pragma unroll
        for (uint32_t k = 0u; k < 2u; ++k) {
            const uint32_t ib  = (it0 + k) * 16u + lane16;
            const uint32_t d2k = __shfl(static_cast<int>(d2_mine), static_cast<int>(ib >> 3), 16);
            const uint4    xa  = sm_xq[ib * 2u];
            const uint4    xb  = sm_xq[ib * 2u + 1u];
            const float    xda = sm_xd[ib];
            #pragma nounroll
            for (uint32_t part = 0u; part < 2u; ++part) {
                const uint32_t qa  = part ? wq[k].z : wq[k].x;
                const uint32_t qb  = part ? wq[k].w : wq[k].y;
                const uint32_t hsp = part ? (hs2[k] >> 16) : (hs2[k] & 0xFFFFu);
                const uint32_t qh  = hsp & 0xFFu, sc = hsp >> 8;
                const float pd = fp16_to_f32(static_cast<uint16_t>(part ? (d2k >> 16) : (d2k & 0xFFFFu))) * 0.25f * xda;
                float acc = part ? au : ag;
                const uint32_t g0 = sm_grid32[(qa & 0xFFu)         | ((qh << 8) & 0x300u)];
                const uint32_t g1 = sm_grid32[((qa >> 16) & 0xFFu) | ((qh << 6) & 0x300u)];
                const uint32_t g2 = sm_grid32[(qb & 0xFFu)         | ((qh << 4) & 0x300u)];
                const uint32_t g3 = sm_grid32[((qb >> 16) & 0xFFu) | ((qh << 2) & 0x300u)];
                int32_t dp0 = sdot4(iq2s_perm4(g0,      (qa >> 8)  & 0xFu), xa.x, 0);
                dp0         = sdot4(iq2s_perm4(g0 >> 2, (qa >> 12) & 0xFu), xa.y, dp0);
                dp0         = sdot4(iq2s_perm4(g1,      (qa >> 24) & 0xFu), xa.z, dp0);
                dp0         = sdot4(iq2s_perm4(g1 >> 2, qa >> 28),           xa.w, dp0);
                acc = fmaf(pd * (0.5f + static_cast<float>(sc & 0xFu)), static_cast<float>(dp0), acc);
                int32_t dp1 = sdot4(iq2s_perm4(g2,      (qb >> 8)  & 0xFu), xb.x, 0);
                dp1         = sdot4(iq2s_perm4(g2 >> 2, (qb >> 12) & 0xFu), xb.y, dp1);
                dp1         = sdot4(iq2s_perm4(g3,      (qb >> 24) & 0xFu), xb.z, dp1);
                dp1         = sdot4(iq2s_perm4(g3 >> 2, qb >> 28),           xb.w, dp1);
                acc = fmaf(pd * (0.5f + static_cast<float>(sc >> 4)), static_cast<float>(dp1), acc);
                if (part) au = acc; else ag = acc;
            }
        }
    }
    ag = vegajet::dot8_dev::sum16(ag);
    au = vegajet::dot8_dev::sum16(au);
    if (lane16 == 0u) sm_y_dst[sub_id] = (ag / (1.0f + __expf(-ag))) * au;
}

} // namespace super_dev
} // namespace vegajet

// ============================================================================
// SUPER-KERNEL A-GDN: [Optional Prev Post-MoE AR] -> w_gdn_qkv_gab -> GDN Conv+SplitV Recur -> w_ssm_out + BAR0 PP
// Grid: 240 WGs x 256 thr (4 WGs/CU x 60 CUs). 36 dispatches/token!
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_m1_gdn_super_kernel(
    const vegajet::ornith_super::GdnSuperDesc* __restrict__ desc,
    uint64_t ssm_ring_override,
    uint32_t prev_ar_epoch,
    uint32_t out_ar_epoch,
    uint32_t conv_in_slot,
    uint32_t conv_out_slot,
    uint32_t ssm_ring_head,
    uint32_t use_ring,
    uint32_t super_sense,
    uint32_t gdn_sense) {
    using namespace vegajet::super_dev;
    __shared__ uint4  sm_xq[256];
    __shared__ float2 sm_xds[128];
    __shared__ float  sm_q[128];
    __shared__ float  sm_k[128];
    __shared__ float  sm_red[8 * 32];
    __shared__ float  sm_dv[32];
    __shared__ float  sm_fo[32];

    const uint32_t wg  = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    GPtr<uint32_t> sbar = as_gptr<uint32_t>(desc->super_bar);
    const bool prof_on = ((super_sense & 0x80000000u) != 0u) && (wg == 0u) && (tid == 0u);
    auto p_acc = [&](uint32_t slot) {
        if (prof_on) {
            GPtr<uint64_t> pt = as_gptr<uint64_t>(desc->super_bar + 1280u);
            const uint64_t now = __builtin_amdgcn_s_memrealtime();
            as_gptr<uint64_t>(desc->super_bar + 1024u)[slot] += (now - pt[0]);
            pt[0] = now;
        }
    };
    if (prof_on) as_gptr<uint64_t>(desc->super_bar + 1280u)[0] = __builtin_amdgcn_s_memrealtime();

    // ---- Stage 0: Optional previous layer Post-MoE P2P AllReduce + RMSNorm + Q8 Quant ----
    if (prev_ar_epoch != 0u) {
        if (wg == 0u) {
            const uint64_t slot_off = static_cast<uint64_t>(prev_ar_epoch & 1u) * VJ_SUPER_AR_SLOT_BYTES;
            p2p_ar_prepush_m1_n(
                as_gptr<const float>(desc->x_partial),
                desc->ar_local_rx_base,
                desc->ar_local_flag,
                slot_off,
                as_gptr<float>(desc->h_io),
                as_gptr<const float>(desc->w_attn_norm),
                nullptr,
                as_gptr<int8_t>(desc->x_q8),
                as_gptr<float2>(desc->x_ds),
                sm_fo,
                desc->D, prev_ar_epoch, desc->rank, desc->n_ranks, desc->eps);
        }
        grid_barrier_wbinvl1(sbar, gridDim.x, super_sense);
        p_acc(0u);
    }

    // ---- Stage 1: w_gdn_qkv_gab GEMV [qkv_rows, 3072] across all WGs ----
    {
        gemv_q4a8_persistent(
            as_gptr<const uint8_t>(desc->w_qkv_qs),
            as_gptr<const uint32_t>(desc->w_qkv_dm),
            as_gptr<const int8_t>(desc->x_q8),
            as_gptr<const float2>(desc->x_ds),
            as_gptr<float>(desc->gdn_qkv_gab),
            nullptr, 0ULL, 0u, 1u, false, sm_xq, sm_xds, sm_fo,
            desc->qkv_rows, desc->qkv_cols, desc->qkv_qs_stride, desc->qkv_dm_stride);
        grid_barrier_wbinvl1(sbar + 32u, gridDim.x, super_sense);
        p_acc(1u);
    }

    // ---- Stage 2: GDN Conv1D + L2 + Gate + Split-V=4 Recurrence + Gated RMSNorm + Quant (nv * 4 WGs) ----
    if (wg < desc->nv * 4u) {
        const uint32_t nk       = desc->nk;
        const uint32_t nv       = desc->nv;
        const uint32_t conv_dim = desc->conv_dim;
        const uint32_t hv       = wg >> 2;
        const uint32_t part     = wg & 3u;
        const uint32_t col0     = part * 32u;
        const uint32_t hk       = hv % nk;
        const float    eps      = desc->eps;
        GPtr<const float> qkv_in = as_gptr<const float>(desc->gdn_qkv_gab);
        GPtr<const float> conv_w = as_gptr<const float>(desc->conv_w);
        GPtr<float> conv_rg = as_gptr<float>(desc->conv_ring);

        {
            const uint64_t cslot_stride = 3ULL * conv_dim;
            GPtr<const float> cr_in = conv_rg + static_cast<uint64_t>(conv_in_slot)  * cslot_stride;
            GPtr<float> cr_out = conv_rg + static_cast<uint64_t>(conv_out_slot) * cslot_stride;

            const uint32_t is_k = tid >> 7;
            const uint32_t l    = tid & 127u;
            const uint32_t ch   = (is_k ? (nk + hk) : hk) * 128u + l;
            const float4   w    = reinterpret_cast<GPtr<const float4>>(conv_w)[ch];
            const float    s0   = cr_in[0ULL * conv_dim + ch];
            const float    s1   = cr_in[1ULL * conv_dim + ch];
            const float    s2   = cr_in[2ULL * conv_dim + ch];
            const float    x    = qkv_in[ch];
            if (hv < nk && part == 0u) {
                cr_out[0ULL * conv_dim + ch] = s1;
                cr_out[1ULL * conv_dim + ch] = s2;
                cr_out[2ULL * conv_dim + ch] = x;
            }
            const float acc  = fmaf(w.x, s0, fmaf(w.y, s1, fmaf(w.z, s2, w.w * x)));
            float       y    = acc / (1.0f + __expf(-acc));
            const float sq_w = wave64_sum(y * y);
            if ((tid & 63u) == 0u) sm_red[tid >> 6] = sq_w;
            __syncthreads();
            float inv_n = rsqrtf(sm_red[is_k ? 2u : 0u] + sm_red[is_k ? 3u : 1u] + eps);
            if (!is_k) inv_n *= 0.08838834764831845f;
            y *= inv_n;
            if (!is_k) sm_q[l] = y; else sm_k[l] = y;

            if (tid < 32u) {
                const uint32_t ch_v = (2u * nk + hv) * 128u + col0 + tid;
                const float4   wv   = reinterpret_cast<GPtr<const float4>>(conv_w)[ch_v];
                const float    sv0  = cr_in[0ULL * conv_dim + ch_v];
                const float    sv1  = cr_in[1ULL * conv_dim + ch_v];
                const float    sv2  = cr_in[2ULL * conv_dim + ch_v];
                const float    xv   = qkv_in[ch_v];
                const float    av   = fmaf(wv.x, sv0, fmaf(wv.y, sv1, fmaf(wv.z, sv2, wv.w * xv)));
                sm_dv[tid] = av / (1.0f + __expf(-av));
                cr_out[0ULL * conv_dim + ch_v] = sv1;
                cr_out[1ULL * conv_dim + ch_v] = sv2;
                cr_out[2ULL * conv_dim + ch_v] = xv;
            }
            if (tid == 0u) {
                const float a  = qkv_in[desc->alpha_off + hv] + as_gptr<const float>(desc->dt_bias)[hv];
                const float sp = (a > 20.0f) ? a : log1pf(__expf(a));
                sm_fo[0] = __expf(sp * as_gptr<const float>(desc->ssm_a)[hv]);
                sm_fo[1] = 1.0f / (1.0f + __expf(-qkv_in[desc->beta_off + hv]));
            }
        }
        __syncthreads();

        const uint64_t head_elems = 128ULL * 128ULL;
        const uint64_t slot_elems = static_cast<uint64_t>(nv) * head_elems;
        GPtr<float> ssm_ring = as_gptr<float>(ssm_ring_override ? ssm_ring_override : desc->ssm_ring_base);
        GPtr<float> S_base = ssm_ring + static_cast<uint64_t>(ssm_ring_head) * slot_elems + static_cast<uint64_t>(hv) * head_elems;
        const uint32_t rg4   = tid >> 3;           // 0..31 (4 rows per thread)
        const uint32_t c4    = tid & 7u;           // 0..7  (4 contiguous cols per thread)
        const uint32_t r0    = rg4 * 4u;
        const uint32_t col4  = (col0 >> 2) + c4;   // float4 column index (0..31)
        GPtr<const float4> S_in4 = reinterpret_cast<GPtr<const float4>>(S_base) + static_cast<uint64_t>(r0) * 32u + col4;

        typedef float f32x4 __attribute__((ext_vector_type(4)));
        auto ldg_f4 = [](GPtr<const float4> p) -> float4 {
            f32x4 v = __builtin_nontemporal_load(reinterpret_cast<GPtr<const f32x4>>(p));
            return make_float4(v[0], v[1], v[2], v[3]);
        };
        auto stg_f4 = [](GPtr<float4> p, float4 v) {
            f32x4 fv = {v.x, v.y, v.z, v.w};
            __builtin_nontemporal_store(fv, reinterpret_cast<GPtr<f32x4>>(p));
        };

        float4 s0 = ldg_f4(S_in4 + 0u  * 32u);
        float4 s1 = ldg_f4(S_in4 + 1u  * 32u);
        float4 s2 = ldg_f4(S_in4 + 2u  * 32u);
        float4 s3 = ldg_f4(S_in4 + 3u  * 32u);
        __builtin_amdgcn_sched_barrier(0);

        const float g_exp = sm_fo[0];
        const float beta  = sm_fo[1];
        const float k0 = sm_k[r0 + 0u], k1 = sm_k[r0 + 1u], k2 = sm_k[r0 + 2u], k3 = sm_k[r0 + 3u];

        float4 pv;
        pv.x = fmaf(s3.x, k3, fmaf(s2.x, k2, fmaf(s1.x, k1, s0.x * k0)));
        pv.y = fmaf(s3.y, k3, fmaf(s2.y, k2, fmaf(s1.y, k1, s0.y * k0)));
        pv.z = fmaf(s3.z, k3, fmaf(s2.z, k2, fmaf(s1.z, k1, s0.z * k0)));
        pv.w = fmaf(s3.w, k3, fmaf(s2.w, k2, fmaf(s1.w, k1, s0.w * k0)));

        auto wave_red8 = [](float v) -> float {
            v += __shfl_xor(v, 32, 64);
            v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x401F));
            v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x201F));
            return v;
        };
        pv.x = wave_red8(pv.x); pv.y = wave_red8(pv.y); pv.z = wave_red8(pv.z); pv.w = wave_red8(pv.w);
        if ((tid & 56u) == 0u) {
            const uint32_t wv = tid >> 6;
            reinterpret_cast<float4*>(sm_red)[wv * 8u + c4] = pv;
        }
        __syncthreads();

        if (tid < 32u) {
            const float pred = (sm_red[tid] + sm_red[32u + tid]) + (sm_red[64u + tid] + sm_red[96u + tid]);
            sm_dv[tid] = beta * (sm_dv[tid] - g_exp * pred);
        }
        __syncthreads();

        const float4 dv4 = reinterpret_cast<const float4*>(sm_dv)[c4];
        GPtr<float4> S_out4 = (use_ring != 0u)
            ? reinterpret_cast<GPtr<float4>>(ssm_ring + static_cast<uint64_t>((ssm_ring_head + 1u) % VJ_SUPER_RING_SLOTS) * slot_elems + static_cast<uint64_t>(hv) * head_elems) + static_cast<uint64_t>(r0) * 32u + col4
            : const_cast<GPtr<float4>>(S_in4);

        const float q0 = sm_q[r0 + 0u], q1 = sm_q[r0 + 1u], q2 = sm_q[r0 + 2u], q3 = sm_q[r0 + 3u];
        auto upd_f4 = [&](float4 s, float kv) -> float4 {
            return make_float4(fmaf(kv, dv4.x, g_exp * s.x), fmaf(kv, dv4.y, g_exp * s.y),
                               fmaf(kv, dv4.z, g_exp * s.z), fmaf(kv, dv4.w, g_exp * s.w));
        };
        s0 = upd_f4(s0, k0); stg_f4(S_out4 + 0u * 32u, s0);
        s1 = upd_f4(s1, k1); stg_f4(S_out4 + 1u * 32u, s1);
        s2 = upd_f4(s2, k2); stg_f4(S_out4 + 2u * 32u, s2);
        s3 = upd_f4(s3, k3); stg_f4(S_out4 + 3u * 32u, s3);

        float4 ov;
        ov.x = fmaf(s3.x, q3, fmaf(s2.x, q2, fmaf(s1.x, q1, s0.x * q0)));
        ov.y = fmaf(s3.y, q3, fmaf(s2.y, q2, fmaf(s1.y, q1, s0.y * q0)));
        ov.z = fmaf(s3.z, q3, fmaf(s2.z, q2, fmaf(s1.z, q1, s0.z * q0)));
        ov.w = fmaf(s3.w, q3, fmaf(s2.w, q2, fmaf(s1.w, q1, s0.w * q0)));
        ov.x = wave_red8(ov.x); ov.y = wave_red8(ov.y); ov.z = wave_red8(ov.z); ov.w = wave_red8(ov.w);
        if ((tid & 56u) == 0u) {
            const uint32_t wv = tid >> 6;
            reinterpret_cast<float4*>(sm_red)[wv * 8u + c4] = ov;
        }
        __syncthreads();

        if (tid < 32u) {
            sm_fo[tid] = (sm_red[tid] + sm_red[32u + tid]) + (sm_red[64u + tid] + sm_red[96u + tid]);
        }
        __syncthreads();

        GPtr<float> sq = as_gptr<float>(desc->gdn_sq) + static_cast<uint64_t>(hv) * 64u;
        const uint32_t l16 = tid & 15u;
        const float fo0 = sm_fo[l16 * 2u];
        const float fo1 = sm_fo[l16 * 2u + 1u];
        if (tid < 16u) stg_f32_stream(sq + part * 16u + tid, fo0 * fo0 + fo1 * fo1);
        grid_barrier_glc_only(as_gptr<uint32_t>(desc->gdn_bar) + static_cast<uint64_t>(hv) * 32u, 4u, gdn_sense);

        if (tid < 64u) {
            const float total_sq = wave64_sum(ldg_f32_stream(sq + tid));
            if (tid < 16u) {
                const float inv_r = rsqrtf(total_sq * (1.0f / 128.0f) + eps);
                const uint32_t j0 = col0 + tid * 2u;
                GPtr<const float> zp = qkv_in + desc->z_off + hv * 128u;
                GPtr<const float> nw = as_gptr<const float>(desc->ssm_norm_w);
                const float z0 = zp[j0], z1 = zp[j0 + 1u];
                const float y0 = (fo0 * inv_r * nw[j0])      * (z0 / (1.0f + __expf(-z0)));
                const float y1 = (fo1 * inv_r * nw[j0 + 1u]) * (z1 / (1.0f + __expf(-z1)));

                float amax = fmaxf(fabsf(y0), fabsf(y1));
                amax = fmaxf(amax, __shfl_xor(amax, 8, 16));
                amax = fmaxf(amax, __shfl_xor(amax, 4, 16));
                amax = fmaxf(amax, __shfl_xor(amax, 2, 16));
                amax = fmaxf(amax, __shfl_xor(amax, 1, 16));

                float bsum = y0 + y1;
                bsum += __shfl_xor(bsum, 8, 16);
                bsum += __shfl_xor(bsum, 4, 16);
                bsum += __shfl_xor(bsum, 2, 16);
                bsum += __shfl_xor(bsum, 1, 16);

                const float d     = amax * (1.0f / 127.0f);
                const float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                GPtr<int8_t> oq = as_gptr<int8_t>(desc->mid_q8) + static_cast<uint64_t>(hv) * 128u;
                oq[j0]      = static_cast<int8_t>(max(-127, min(127, __float2int_rn(y0 * inv_d))));
                oq[j0 + 1u] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(y1 * inv_d))));
                if (tid == 0u) as_gptr<float2>(desc->mid_ds)[static_cast<uint64_t>(hv) * 4u + part] = make_float2(d, bsum);
            }
        }
    }
    grid_barrier_wbinvl1(sbar + 64u, gridDim.x, super_sense);
    p_acc(2u);

    // ---- Stage 3: w_ssm_out GEMV [3072, nv*128] + FP16 BAR0 prepush ----
    {
        const uint64_t slot_off = static_cast<uint64_t>(out_ar_epoch & 1u) * VJ_SUPER_AR_SLOT_BYTES;
        const bool do_pp = (out_ar_epoch != 0u) && (desc->n_ranks > 1u);
        gemv_q4a8_persistent(
            as_gptr<const uint8_t>(desc->w_out_qs),
            as_gptr<const uint32_t>(desc->w_out_dm),
            as_gptr<const int8_t>(desc->mid_q8),
            as_gptr<const float2>(desc->mid_ds),
            as_gptr<float>(desc->x_partial),
            desc->out_peer_tx_base, slot_off, desc->rank, desc->n_ranks, do_pp,
            sm_xq, sm_xds, sm_fo,
            desc->out_rows, desc->out_cols, desc->out_qs_stride, desc->out_dm_stride);
        if (do_pp) {
            signal_out_peer_flag_n(sbar, desc->out_peer_flag, desc->rank, desc->n_ranks, gridDim.x, out_ar_epoch);
        }
        p_acc(3u);
    }
}

// ============================================================================
// SUPER-KERNEL A-GQA: [Optional Prev Post-MoE AR] -> w_attn_qkv -> Prep + Split-KV Flash -> w_o + BAR0 PP
// Grid: 240 WGs x 256 thr (4 WGs/CU x 60 CUs). 12 dispatches/token!
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_m1_gqa_super_kernel(
    const vegajet::ornith_super::GqaSuperDesc* __restrict__ desc,
    uint32_t prev_ar_epoch,
    uint32_t out_ar_epoch,
    uint32_t pos0,
    uint32_t super_sense) {
    using namespace vegajet::super_dev;
    __shared__ uint4  sm_xq[256];
    __shared__ float2 sm_xds[128];
    __shared__ float  sm_v[256];
    __shared__ float  sm_red[32];

    const uint32_t wg   = blockIdx.x;
    const uint32_t d    = threadIdx.x;
    const uint32_t wave = d >> 6;
    const uint32_t lane = d & 63u;
    const uint32_t blk  = d >> 5;
    const uint32_t sub  = d & 31u;
    GPtr<uint32_t> sbar = as_gptr<uint32_t>(desc->super_bar);
    const bool prof_on = ((super_sense & 0x80000000u) != 0u) && (wg == 0u) && (d == 0u);
    auto p_acc = [&](uint32_t slot) {
        if (prof_on) {
            GPtr<uint64_t> pt = as_gptr<uint64_t>(desc->super_bar + 1280u);
            const uint64_t now = __builtin_amdgcn_s_memrealtime();
            as_gptr<uint64_t>(desc->super_bar + 1024u)[slot] += (now - pt[0]);
            pt[0] = now;
        }
    };
    if (prof_on) as_gptr<uint64_t>(desc->super_bar + 1280u)[0] = __builtin_amdgcn_s_memrealtime();

    // ---- Stage 0: Optional previous Post-MoE P2P AllReduce + RMSNorm + Q8 Quant ----
    if (prev_ar_epoch != 0u) {
        if (wg == 0u) {
            const uint64_t slot_off = static_cast<uint64_t>(prev_ar_epoch & 1u) * VJ_SUPER_AR_SLOT_BYTES;
            p2p_ar_prepush_m1_n(
                as_gptr<const float>(desc->x_partial),
                desc->ar_local_rx_base,
                desc->ar_local_flag,
                slot_off,
                as_gptr<float>(desc->h_io),
                as_gptr<const float>(desc->w_attn_norm),
                nullptr,
                as_gptr<int8_t>(desc->x_q8),
                as_gptr<float2>(desc->x_ds),
                sm_red,
                desc->D, prev_ar_epoch, desc->rank, desc->n_ranks, desc->eps);
        }
        grid_barrier_wbinvl1(sbar, gridDim.x, super_sense);
        p_acc(4u);
    }

    // ---- Stage 1: w_attn_qkv GEMV [qkv_rows, 3072] across all WGs ----
    {
        gemv_q4a8_persistent(
            as_gptr<const uint8_t>(desc->w_qkv_qs),
            as_gptr<const uint32_t>(desc->w_qkv_dm),
            as_gptr<const int8_t>(desc->x_q8),
            as_gptr<const float2>(desc->x_ds),
            as_gptr<float>(desc->attn_qkv),
            nullptr, 0ULL, 0u, 1u, false, sm_xq, sm_xds, sm_red,
            desc->qkv_rows, desc->qkv_cols, desc->qkv_qs_stride, desc->qkv_dm_stride);
        grid_barrier_wbinvl1(sbar + 32u, gridDim.x, super_sense);
        p_acc(5u);
    }

    // ---- Stage 2a: Q/K Norm + RoPE + KV Q8_0 cache write (17 WGs: 0..15 Q, 16 K+V) ----
    const uint32_t n_q_local  = desc->n_q_local;
    const uint32_t n_kv_local = desc->n_kv_local;
    {
        GPtr<const float> qkv_f = as_gptr<const float>(desc->attn_qkv);
        const uint32_t qg_tot = n_q_local * 512u;
        const uint32_t kv_tot = n_kv_local * 256u;
        const uint32_t rdims  = desc->rope_dims;
        const uint32_t half_r = rdims >> 1;
        const float    rbase  = desc->rope_base;
        const float    eps    = desc->eps;
        const float    pos_f  = static_cast<float>(pos0);

        if (wg < n_q_local) {
            GPtr<const float> qg = qkv_f + wg * 512u;
            const float qr  = qg[d];
            const float sq  = wave64_sum(qr * qr);
            if (lane == 0u) sm_red[wave] = sq;
            __syncthreads();
            const float inv_r = rsqrtf((sm_red[0] + sm_red[1] + sm_red[2] + sm_red[3]) * (1.0f / 256.0f) + eps);
            sm_v[d] = qr * inv_r * as_gptr<const float>(desc->q_norm_w)[d];
            __syncthreads();
            float qv = sm_v[d];
            if (d < rdims) {
                const uint32_t p = (d < half_r) ? d : (d - half_r);
                const float theta = pos_f * __powf(rbase, -2.0f * static_cast<float>(p) / static_cast<float>(rdims));
                float s_th, c_th;
                __sincosf(theta, &s_th, &c_th);
                qv = (d < half_r) ? (sm_v[p] * c_th - sm_v[p + half_r] * s_th)
                                  : (sm_v[p] * s_th + sm_v[p + half_r] * c_th);
            }
            as_gptr<float>(desc->q_buf)[static_cast<uint64_t>(wg) * 256u + d] = qv;
        } else if (wg < n_q_local + n_kv_local) {
            const uint32_t kv_h = wg - n_q_local;
            GPtr<const float> kr = qkv_f + qg_tot + kv_h * 256u;
            GPtr<const float> vr = qkv_f + qg_tot + kv_tot + kv_h * 256u;
            const float kv_raw = kr[d];
            const float sq = wave64_sum(kv_raw * kv_raw);
            if (lane == 0u) sm_red[wave] = sq;
            __syncthreads();
            const float inv_r = rsqrtf((sm_red[0] + sm_red[1] + sm_red[2] + sm_red[3]) * (1.0f / 256.0f) + eps);
            sm_v[d] = kv_raw * inv_r * as_gptr<const float>(desc->k_norm_w)[d];
            __syncthreads();
            float k_rot = sm_v[d];
            if (d < rdims) {
                const uint32_t p = (d < half_r) ? d : (d - half_r);
                const float theta = pos_f * __powf(rbase, -2.0f * static_cast<float>(p) / static_cast<float>(rdims));
                float s_th, c_th;
                __sincosf(theta, &s_th, &c_th);
                k_rot = (d < half_r) ? (sm_v[p] * c_th - sm_v[p + half_r] * s_th)
                                     : (sm_v[p] * s_th + sm_v[p + half_r] * c_th);
            }
            const uint64_t slot_off = (static_cast<uint64_t>(pos0) * n_kv_local + kv_h) * 272ULL;
            GPtr<uint8_t> kc = as_gptr<uint8_t>(desc->k_cache_q8) + slot_off;
            GPtr<uint8_t> vc = as_gptr<uint8_t>(desc->v_cache_q8) + slot_off;
            {
                const float amax = block32_max(fabsf(k_rot));
                const float sc   = amax * (1.0f / 127.0f);
                const float inv  = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                reinterpret_cast<GPtr<int8_t>>(kc + 16u)[d] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(k_rot * inv))));
                if (sub == 0u) reinterpret_cast<GPtr<uint16_t>>(kc)[blk] = __half_as_ushort(__float2half(sc));
            }
            {
                const float vv   = vr[d];
                const float amax = block32_max(fabsf(vv));
                const float sc   = amax * (1.0f / 127.0f);
                const float inv  = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                reinterpret_cast<GPtr<int8_t>>(vc + 16u)[d] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(vv * inv))));
                if (sub == 0u) reinterpret_cast<GPtr<uint16_t>>(vc)[blk] = __half_as_ushort(__float2half(sc));
            }
        }
        grid_barrier_wbinvl1(sbar + 64u, gridDim.x, super_sense);
        p_acc(6u);
    }

    // ---- Stage 2b: In-Kernel 192-WG Split-KV FlashAttention Partial (constant chunk_len=32 up to 1920, bit-exact with verify!) ----
    const uint32_t span = pos0 + 1u;
    const uint32_t chunk_len = (span <= 1920u) ? 32u
                             : ((span <= 3840u) ? 64u
                             : ((span <= 7680u) ? 128u : 256u));
    const uint32_t n_chunks = (span + chunk_len - 1u) / chunk_len;
    const uint32_t max_ch   = gridDim.x / n_q_local; // 12 chunk WGs/head for 192 WGs / 16 heads

    {
        const uint32_t active_ch = min(n_chunks, max_ch);
        if (wg < n_q_local * active_ch) {
            const uint32_t h   = wg / active_ch;
            const uint32_t ch0 = wg - h * active_ch;
            const uint32_t kv_h = (desc->gqa_group != 0u)
                ? (((desc->q_head_off + h) / desc->gqa_group) - desc->kv_head_off)
                : ((h * n_kv_local) / n_q_local);
            const float qv = as_gptr<const float>(desc->q_buf)[static_cast<uint64_t>(h) * 256u + d];
            GPtr<const uint8_t> k_cache = as_gptr<const uint8_t>(desc->k_cache_q8);
            GPtr<const uint8_t> v_cache = as_gptr<const uint8_t>(desc->v_cache_q8);

            #pragma nounroll
            for (uint32_t ch = ch0; ch < n_chunks; ch += active_ch) {
                const uint32_t t_beg = ch * chunk_len;
                const uint32_t t_end = min(t_beg + chunk_len, span);
                float m_run = -1e30f, l_run = 0.0f, o_d = 0.0f;
                #pragma nounroll
                for (uint32_t t0 = t_beg; t0 < t_end; t0 += 4u) {
                    float dots[4], vvs[4];
                    #pragma unroll
                    for (uint32_t j = 0u; j < 4u; ++j) {
                        const uint32_t t = min(t0 + j, t_end - 1u);
                        const uint64_t off = (static_cast<uint64_t>(t) * n_kv_local + kv_h) * 272ULL;
                        GPtr<const uint8_t> kc = k_cache + off;
                        GPtr<const uint8_t> vc = v_cache + off;
                        const float ksc = fp16_to_f32(reinterpret_cast<GPtr<const uint16_t>>(kc)[blk]);
                        const float kv  = ksc * static_cast<float>(reinterpret_cast<GPtr<const int8_t>>(kc + 16u)[d]);
                        const float vsc = fp16_to_f32(reinterpret_cast<GPtr<const uint16_t>>(vc)[blk]);
                        vvs[j]  = vsc * static_cast<float>(reinterpret_cast<GPtr<const int8_t>>(vc + 16u)[d]);
                        dots[j] = qv * kv;
                    }
                    #pragma unroll
                    for (uint32_t j = 0u; j < 4u; ++j) {
                        const float dot = wave64_sum(dots[j]);
                        if (lane == 0u) sm_red[j * 4u + wave] = dot;
                    }
                    __syncthreads();
                    #pragma unroll
                    for (uint32_t j = 0u; j < 4u; ++j) {
                        if (t0 + j < t_end) {
                            const float score = (sm_red[j * 4u] + sm_red[j * 4u + 1u] + sm_red[j * 4u + 2u] + sm_red[j * 4u + 3u]) * 0.0625f;
                            const float m_new = fmaxf(m_run, score);
                            const float alpha = __expf(m_run - m_new);
                            const float beta  = __expf(score - m_new);
                            l_run = l_run * alpha + beta;
                            m_run = m_new;
                            o_d = fmaf(beta, vvs[j], o_d * alpha);
                        }
                    }
                    __syncthreads();
                }

                if (n_chunks == 1u) {
                    const float attn_v = (l_run > 0.0f) ? (o_d / l_run) : 0.0f;
                    const float gv     = as_gptr<const float>(desc->attn_qkv)[h * 512u + 256u + d];
                    const float y_d    = attn_v * (1.0f / (1.0f + __expf(-gv)));
                    const float amax   = block32_max(fabsf(y_d));
                    const float bsum   = block32_sum(y_d);
                    const float sc     = amax * (1.0f / 127.0f);
                    const float inv_d  = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                    as_gptr<int8_t>(desc->mid_q8)[static_cast<uint64_t>(h) * 256u + d] =
                        static_cast<int8_t>(max(-127, min(127, __float2int_rn(y_d * inv_d))));
                    if (sub == 0u) as_gptr<float2>(desc->mid_ds)[static_cast<uint64_t>(h) * 8u + blk] = make_float2(sc, bsum);
                } else {
                    const uint64_t idx = static_cast<uint64_t>(h) * n_chunks + ch;
                    as_gptr<float>(desc->part_o)[idx * 256u + d] = o_d;
                    if (d == 0u) as_gptr<float2>(desc->part_ml)[idx] = make_float2(m_run, l_run);
                }
            }
        }
        grid_barrier_wbinvl1(sbar + 96u, gridDim.x, super_sense);
        p_acc(7u);
    }

    // ---- Stage 2c: Merge Split-KV partials + Gate + Q8 Quant (16 WGs, only when n_chunks > 1; bit-exact with vj_flash_splitkv_any_merge_quant_kernel!) ----
    if (n_chunks > 1u) {
        if (wg < n_q_local) {
            GPtr<const float> po = as_gptr<const float>(desc->part_o) + static_cast<uint64_t>(wg) * n_chunks * 256u;
            GPtr<const float2> pml = as_gptr<const float2>(desc->part_ml) + static_cast<uint64_t>(wg) * n_chunks;
            float g_max = -1e30f;
            #pragma nounroll
            for (uint32_t ch = 0u; ch < n_chunks; ++ch) {
                const float2 ml = pml[ch];
                if (ml.y > 0.0f) g_max = fmaxf(g_max, ml.x);
            }
            float o_tot = 0.0f, l_tot = 0.0f;
            #pragma nounroll
            for (uint32_t ch = 0u; ch < n_chunks; ++ch) {
                const float2 ml = pml[ch];
                if (ml.y > 0.0f) {
                    const float w = __expf(ml.x - g_max);
                    l_tot = fmaf(w, ml.y, l_tot);
                    o_tot = fmaf(w, po[static_cast<uint64_t>(ch) * 256u + d], o_tot);
                }
            }
            const float attn_v = (l_tot > 0.0f) ? (o_tot / l_tot) : 0.0f;
            const float gv     = as_gptr<const float>(desc->attn_qkv)[wg * 512u + 256u + d];
            const float y_d    = attn_v * (1.0f / (1.0f + __expf(-gv)));
            const float amax   = block32_max(fabsf(y_d));
            const float bsum   = block32_sum(y_d);
            const float sc     = amax * (1.0f / 127.0f);
            const float inv_d  = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            as_gptr<int8_t>(desc->mid_q8)[static_cast<uint64_t>(wg) * 256u + d] =
                static_cast<int8_t>(max(-127, min(127, __float2int_rn(y_d * inv_d))));
            if (sub == 0u) as_gptr<float2>(desc->mid_ds)[static_cast<uint64_t>(wg) * 8u + blk] = make_float2(sc, bsum);
        }
        grid_barrier_wbinvl1(sbar + 128u, gridDim.x, super_sense);
        p_acc(8u);
    }

    // ---- Stage 3: w_o GEMV [3072, n_q_local*256] + FP16 BAR0 prepush ----
    {
        const uint64_t slot_off = static_cast<uint64_t>(out_ar_epoch & 1u) * VJ_SUPER_AR_SLOT_BYTES;
        const bool do_pp = (out_ar_epoch != 0u) && (desc->n_ranks > 1u);
        gemv_q4a8_persistent(
            as_gptr<const uint8_t>(desc->w_o_qs),
            as_gptr<const uint32_t>(desc->w_o_dm),
            as_gptr<const int8_t>(desc->mid_q8),
            as_gptr<const float2>(desc->mid_ds),
            as_gptr<float>(desc->x_partial),
            desc->out_peer_tx_base, slot_off, desc->rank, desc->n_ranks, do_pp,
            sm_xq, sm_xds, sm_red,
            desc->o_rows, desc->o_cols, desc->o_qs_stride, desc->o_dm_stride);
        if (do_pp) {
            signal_out_peer_flag_n(sbar, desc->out_peer_flag, desc->rank, desc->n_ranks, gridDim.x, out_ar_epoch);
        }
        p_acc(9u);
    }
}

// ============================================================================
// SUPER-KERNEL B-MoE: [Post-Mixer AR] -> [Balanced 240-WG Router+ShGU] -> [Top8+ShSwiGLU]
//                     -> [240-WG 4-KiB grid32 IQ2_S Gate+Up + Shared Down] -> [192-WG DOT8 Down + BAR0 PP]
// Grid: 240 WGs x 256 thr (4 WGs/CU x 60 CUs). 48 dispatches/token! Total LDS = 8,928 B!
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_m1_moe_super_kernel(
    const vegajet::ornith_super::MoeSuperDesc* __restrict__ desc,
    uint32_t ar_epoch,
    uint32_t out_ar_epoch,
    uint32_t super_sense) {
    using namespace vegajet::super_dev;
    __shared__ uint32_t sm_grid32[1024]; // 4,096 B (half of sm_grid!)
    __shared__ uint4    sm_x[256];       // 4,096 B
    __shared__ float    sm_xd[128];      //   512 B
    __shared__ float    sm_y[32];        //   128 B
    __shared__ int32_t  sm_ids[8];       //    32 B
    __shared__ float    sm_w[8];         //    32 B -> total 8,896 B!

    const uint32_t wg  = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t grp = tid >> 5;
    const uint32_t l32 = tid & 31u;
    GPtr<uint32_t> sbar = as_gptr<uint32_t>(desc->super_bar);
    const bool prof_on = ((super_sense & 0x80000000u) != 0u) && (wg == 0u) && (tid == 0u);
    GPtr<uint64_t> prof = as_gptr<uint64_t>(desc->super_bar + 1024u);
    uint64_t t_p = prof_on ? __builtin_amdgcn_s_memrealtime() : 0ULL;

    // ---- Stage 0: Optional Post-Mixer Prepush FP16 AR (WG 0) overlapped with 4-KiB sm_grid32 LDS preload! ----
    if (ar_epoch != 0u && wg == 0u) {
        const uint64_t slot_off = static_cast<uint64_t>(ar_epoch & 1u) * VJ_SUPER_AR_SLOT_BYTES;
        p2p_ar_prepush_m1_n(
            as_gptr<const float>(desc->x_partial),
            desc->ar_local_rx_base,
            desc->ar_local_flag,
            slot_off,
            as_gptr<float>(desc->h_io),
            as_gptr<const float>(desc->w_post_attn_norm),
            as_gptr<float>(desc->x_norm),
            as_gptr<int8_t>(desc->x_q8),
            as_gptr<float2>(desc->x_ds),
            sm_y,
            desc->D, ar_epoch, desc->rank, desc->n_ranks, desc->eps);
    }
    // Preload 4-KiB sm_grid32 into LDS ONLY on non-DOT8 (IQ2_S) layers!
    if (desc->w_gu_d8_w == 0ULL) {
        reinterpret_cast<uint4*>(sm_grid32)[tid] = reinterpret_cast<const uint4*>(vegajet::ggml_tables::iq2s_grid32)[tid];
    }
    if (ar_epoch != 0u) {
        grid_barrier_wbinvl1(sbar, gridDim.x, super_sense);
    } else {
        __syncthreads();
    }
    prof_acc(prof, 10u, t_p, prof_on);

    // ---- Stage 1: Balanced 240-WG Router FP16 (128 lanes/row, all 256 thr!) + Shared Gate/Up (LDS x_q8/x_ds!) + Shared Gate ----
    //   WGs   0..127 (128 WGs): Shared Gate/Up (8 rows/WG = 1024 exact, 15.4 KB/WG, reads x_q8/x_ds from LDS!)
    //   WGs 128..239 (112 WGs): Router FP16 (2-3 rows/WG = 256 exact, 12.3-18.4 KB/WG) + Shared Gate on WG 239 half 1!
    {
        // Preload x_q8 (sm_x[0..191]) + x_ds (sm_x[192..239] & sm_xd[0..95]) into LDS ONCE for BOTH Stage 1 (sh_gu) AND Stage 3 (moe_gu)!
        if (tid < 192u) sm_x[tid] = as_gptr<const uint4>(desc->x_q8)[tid];
        if (tid <  96u) {
            const float2 xds = as_gptr<const float2>(desc->x_ds)[tid];
            reinterpret_cast<float2*>(sm_x + 192u)[tid] = xds;
            sm_xd[tid] = xds.x;
        }
        __syncthreads();

        const uint32_t D = desc->D;
        GPtr<const float4> x4f = as_gptr<const float4>(desc->x_norm);
        const uint32_t sh_wgs = (gridDim.x == 240u && desc->sh_gu_rows <= 768u) ? 96u : 128u;
        if (wg < sh_wgs) {
            const uint32_t sh_rows = desc->sh_gu_rows;
            const uint32_t row = wg * 8u + grp;
            if (row < sh_rows) {
                const uint32_t  nb = D >> 5;
                GPtr<const uint4> w4 = as_gptr<const uint4>(desc->sh_gu_qs + static_cast<uint64_t>(row) * desc->sh_gu_stride);
                GPtr<const uint32_t> dm = as_gptr<const uint32_t>(desc->sh_gu_dm) + static_cast<uint64_t>(row) * nb;
                const float2*   xd = reinterpret_cast<const float2*>(sm_x + 192u);
                float acc = 0.0f;
                // D == 3072 -> nb == 96 exact: single 3-batch unrolled pass, 0 loop overhead!
                uint4 wvs[3]; uint32_t dms[3];
                #pragma unroll
                for (uint32_t r = 0u; r < 3u; ++r) {
                    const uint32_t bc = l32 + r * 32u;
                    wvs[r] = ldg_u32x4_stream(w4 + bc);
                    dms[r] = ldg_u32_stream(dm + bc);
                }
                __builtin_amdgcn_sched_barrier(0);
                #pragma unroll
                for (uint32_t r = 0u; r < 3u; ++r) {
                    const uint32_t bq = l32 + r * 32u;
                    const uint4 wv = wvs[r]; const uint32_t dmv = dms[r];
                    const uint4 xa = sm_x[bq * 2u], xb = sm_x[bq * 2u + 1u];
                    const float2 xds = xd[bq];
                    int32_t dp = sdot4(wv.x & 0x0F0F0F0Fu, xa.x, 0);
                    dp = sdot4(wv.y & 0x0F0F0F0Fu, xa.y, dp);
                    dp = sdot4(wv.z & 0x0F0F0F0Fu, xa.z, dp);
                    dp = sdot4(wv.w & 0x0F0F0F0Fu, xa.w, dp);
                    dp = sdot4((wv.x >> 4) & 0x0F0F0F0Fu, xb.x, dp);
                    dp = sdot4((wv.y >> 4) & 0x0F0F0F0Fu, xb.y, dp);
                    dp = sdot4((wv.z >> 4) & 0x0F0F0F0Fu, xb.z, dp);
                    dp = sdot4((wv.w >> 4) & 0x0F0F0F0Fu, xb.w, dp);
                    const float wd = fp16_to_f32(static_cast<uint16_t>(dmv & 0xFFFFu));
                    const float wm = fp16_to_f32(static_cast<uint16_t>(dmv >> 16));
                    acc = fmaf(wd * xds.x, static_cast<float>(dp), fmaf(wm, xds.y, acc));
                }
                acc = block32_sum(acc);
                if (l32 == 0u) as_gptr<float>(desc->sh_gu_out)[row] = acc;
            }
        } else {
            const uint32_t rwg = wg - sh_wgs;
            uint32_t e_beg = 256u, e_end = 256u;
            bool do_sh_gate = false;
            if (sh_wgs == 96u) {
                if (rwg < 128u)       { e_beg = rwg * 2u; e_end = e_beg + 2u; }
                else if (rwg == 128u) { do_sh_gate = true; }
            } else if (gridDim.x == 240u) {
                if (rwg < 78u)       { e_beg = rwg * 2u; e_end = e_beg + 2u; }
                else if (rwg < 111u) { e_beg = 156u + (rwg - 78u) * 3u; e_end = e_beg + 3u; }
                else if (rwg == 111u){ e_beg = 255u; e_end = 256u; do_sh_gate = true; }
            } else {
                e_beg = rwg * 4u;
                e_end = min(e_beg + 4u, 256u);
                do_sh_gate = (rwg == 0u);
            }
            const uint32_t half_id = tid >> 7;
            const uint32_t l128    = tid & 127u;
            const uint32_t wr_stride = (D == 3072u) ? 3200u : D;
            const uint32_t n8 = D >> 3;
            #pragma nounroll
            for (uint32_t e0 = e_beg; e0 < e_end; e0 += 2u) {
                const uint32_t e = e0 + half_id;
                float acc = 0.0f;
                if (e < e_end) {
                    GPtr<const uint4> w4 = reinterpret_cast<GPtr<const uint4>>(
                        as_gptr<const uint16_t>(desc->w_router_f16) + static_cast<uint64_t>(e) * wr_stride);
                    // D=3072 -> n8=384 -> exact 3 loads per 128 lanes (0 min/branch)!
                    uint4 wvs[3];
                    #pragma unroll
                    for (uint32_t r = 0u; r < 3u; ++r) {
                        wvs[r] = ldg_u32x4_stream(w4 + l128 + r * 128u);
                    }
                    __builtin_amdgcn_sched_barrier(0);
                    #pragma unroll
                    for (uint32_t r = 0u; r < 3u; ++r) {
                        const uint32_t i = l128 + r * 128u;
                        const uint4 wv = wvs[r];
                        const float4 xa = x4f[i * 2u], xb = x4f[i * 2u + 1u];
                        const float2 h0 = __half22float2(*reinterpret_cast<const __half2*>(&wv.x));
                        const float2 h1 = __half22float2(*reinterpret_cast<const __half2*>(&wv.y));
                        const float2 h2 = __half22float2(*reinterpret_cast<const __half2*>(&wv.z));
                        const float2 h3 = __half22float2(*reinterpret_cast<const __half2*>(&wv.w));
                        acc = fmaf(h0.x, xa.x, fmaf(h0.y, xa.y, fmaf(h1.x, xa.z, fmaf(h1.y, xa.w, acc))));
                        acc = fmaf(h2.x, xb.x, fmaf(h2.y, xb.y, fmaf(h3.x, xb.z, fmaf(h3.y, xb.w, acc))));
                    }
                } else if (do_sh_gate && half_id == 1u) {
                    GPtr<const float4> w4 = as_gptr<const float4>(desc->w_sh_gate);
                    const uint32_t n4 = D >> 2;
                    #pragma nounroll
                    for (uint32_t i = l128; i < n4; i += 128u) {
                        const float4 wv = w4[i], xv = x4f[i];
                        acc = fmaf(wv.x, xv.x, fmaf(wv.y, xv.y, fmaf(wv.z, xv.z, fmaf(wv.w, xv.w, acc))));
                    }
                }
                acc = wave64_sum(acc);
                if ((tid & 63u) == 0u) sm_y[tid >> 6] = acc;
                __syncthreads();
                if (l128 == 0u) {
                    const float tot = sm_y[half_id * 2u] + sm_y[half_id * 2u + 1u];
                    if (e < e_end) {
                        as_gptr<float>(desc->router_logits)[e] = tot;
                    } else if (do_sh_gate && half_id == 1u) {
                        as_gptr<float>(desc->sh_gate_sc)[0] = 1.0f / (1.0f + __expf(-tot));
                    }
                }
                __syncthreads();
            }
            if (do_sh_gate && (gridDim.x != 240u || sh_wgs == 96u)) {
                GPtr<const float4> w4 = as_gptr<const float4>(desc->w_sh_gate);
                const uint32_t n4 = D >> 2;
                float acc = 0.0f;
                #pragma nounroll
                for (uint32_t i = tid; i < n4; i += 256u) {
                    const float4 wv = w4[i], xv = x4f[i];
                    acc = fmaf(wv.x, xv.x, fmaf(wv.y, xv.y, fmaf(wv.z, xv.z, fmaf(wv.w, xv.w, acc))));
                }
                acc = wave64_sum(acc);
                if ((tid & 63u) == 0u) sm_y[tid >> 6] = acc;
                __syncthreads();
                if (tid == 0u) {
                    const float tot = (sm_y[0] + sm_y[1]) + (sm_y[2] + sm_y[3]);
                    as_gptr<float>(desc->sh_gate_sc)[0] = 1.0f / (1.0f + __expf(-tot));
                }
            }
        }
        grid_barrier_wbinvl1(sbar + 32u, gridDim.x, super_sense);
        prof_acc(prof, 11u, t_p, prof_on);
    }

    // ---- Stage 2 (BARRIER-FREE!): Every WG computes local Top-8 in Wave 0 from L2-hot 1 KiB router_logits! ----
    if (tid < 64u) {
        fast_softmax_top8_wave64(
            as_gptr<const float>(desc->router_logits),
            sm_ids, sm_w, tid);
    }
    __syncthreads();
    if (wg == 0u && tid < 8u) {
        as_gptr<int32_t>(desc->top_ids)[tid] = sm_ids[tid];
        as_gptr<float>(desc->top_w)[tid]     = sm_w[tid];
    }
    prof_acc(prof, 12u, t_p, prof_on);

    // ---- Stage 3: ALL WGs (240 WGs = 4 WGs/CU!) Exact VJ_Q3_IQ2S (kBatch=3 + v_bfi_b32) / DOT8 / IQ2_S Gate+Up + 64-WG Shared Down ----
    {
        const uint32_t local_ff = desc->local_ff;
        const uint32_t nblk16   = local_ff >> 4;
        const uint32_t n_tiles  = 8u * nblk16;
        GPtr<const uint8_t> wgu_d8_w = as_gptr<const uint8_t>(desc->w_gu_d8_w);
        GPtr<const uint8_t> wgu_d8_s = as_gptr<const uint8_t>(desc->w_gu_d8_s);
        const uint32_t gu_w_st  = desc->gu_d8_w_stride;
        const uint32_t gu_s_st  = desc->gu_d8_s_stride;
        GPtr<const uint8_t> wgu_b = as_gptr<const uint8_t>(desc->w_gu_iq2s);
        GPtr<uint32_t> pair_cnt = as_gptr<uint32_t>(desc->super_bar + 1536u);
        GPtr<float> mf32 = as_gptr<float>(desc->mid_f32);
        #pragma nounroll
        for (uint32_t tile = wg; tile < n_tiles; tile += gridDim.x) {
            if (wgu_d8_w != nullptr) {
                const uint32_t sub_id = tid >> 4;
                const uint32_t lane16 = tid & 15u;
                uint32_t slot, t_in_slot;
                divmod_slot_u(tile, nblk16, slot, t_in_slot);
                const uint32_t rloc   = t_in_slot * 16u + sub_id;
                const uint32_t exp_id = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(sm_ids[slot]));
                const float2 gu = (gu_w_st == 2304u)
                    ? vegajet::dot8_dev::vj_q3_gu_row_iq2s<6, 2>(
                          wgu_d8_w, gu_w_st, wgu_d8_s, gu_s_st,
                          sm_x, sm_xd, exp_id * local_ff, rloc, lane16)
                    : vegajet::dot8_dev::vj_dot8_gu_row_iq2s<6, 2>(
                          wgu_d8_w, gu_w_st, wgu_d8_s, gu_s_st,
                          sm_x, sm_xd, exp_id * local_ff, rloc, lane16);
                if (lane16 == 0u) sm_y[sub_id] = (gu.x / (1.0f + __expf(-gu.x))) * gu.y;
            } else {
                moe_gu_16r_tile_g32_sm(wgu_b, sm_grid32, sm_x, sm_xd, sm_ids, sm_y, tile, local_ff, nblk16);
            }
            __syncthreads();
            const uint32_t blk    = tile >> 1;
            const uint32_t is_odd = tile & 1u;
            GPtr<float> dst = mf32 + tile * 16u;
            if (tid < 16u) dst[tid] = sm_y[tid];
            asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
            if (tid == 0u) sm_y[16] = __uint_as_float(__atomic_fetch_xor(pair_cnt + blk, 1u, __ATOMIC_RELAXED));
            __syncthreads();
            if (__float_as_uint(sm_y[16]) == 1u && tid < 32u) {
                GPtr<const float> other = mf32 + (tile ^ 1u) * 16u;
                const float y = (tid < 16u)
                    ? (is_odd ? ldg_f32_stream(other + tid) : sm_y[tid])
                    : (is_odd ? sm_y[tid - 16u] : ldg_f32_stream(other + tid - 16u));
                const float amax = block32_max(fabsf(y));
                const float d    = amax * (1.0f / 127.0f);
                const float inv  = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                as_gptr<int8_t>(desc->mid_q8)[blk * 32u + tid] =
                    static_cast<int8_t>(max(-127, min(127, __float2int_rn(y * inv))));
                if (tid == 0u) as_gptr<float>(desc->mid_d)[blk] = d;
            }
            __syncthreads();
        }

        // TP=3 (n_tiles <= 192): run 48 Shared Down WGs x 64 rows strictly on IDLE WGs 192..239 (0 Gate+Up collision!).
        // TP=2 (n_tiles == 256): 64 tiles x 48 rows on wg 16..59 + 76..95 (0 overlap with 2-tile CUs 0..15).
        uint32_t sh_tile = 64u;
        uint32_t sh_steps = 3u;
        uint32_t sh_stride_rows = 48u;
        if (gridDim.x == 240u && n_tiles <= 192u) {
            if (wg >= 192u && wg < 240u) {
                sh_tile = wg - 192u;
                sh_steps = 4u;
                sh_stride_rows = 64u;
            }
        } else if (gridDim.x == 240u) {
            if (wg >= 16u && wg < 60u)      sh_tile = wg - 16u;
            else if (wg >= 76u && wg < 96u) sh_tile = 44u + (wg - 76u);
        } else if (wg >= 128u && wg < 192u) {
            sh_tile = wg - 128u;
        }
        if (sh_tile < 64u) {
            const uint32_t n_blk = local_ff >> 5;
            const float sc_g = as_gptr<const float>(desc->sh_gate_sc)[0];
            GPtr<const float> g_p = as_gptr<const float>(desc->sh_gu_out);
            GPtr<const float> u_p = g_p + local_ff;
            int8_t* sh_q8_sm = reinterpret_cast<int8_t*>(sm_x);
            float2* sh_ds_sm = reinterpret_cast<float2*>(sm_x + 32u);
            #pragma nounroll
            for (uint32_t blk = tid >> 5; blk < n_blk; blk += 8u) {
                const uint32_t c = (blk << 5) + l32;
                const float gv = g_p[c], uv = u_p[c];
                const float y  = (gv / (1.0f + __expf(-gv))) * uv * sc_g;
                const float amax  = block32_max(fabsf(y));
                const float bsum  = block32_sum(y);
                const float d     = amax * (1.0f / 127.0f);
                const float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                sh_q8_sm[c] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(y * inv_d))));
                if (l32 == 0u) sh_ds_sm[blk] = make_float2(d, bsum);
            }
            __syncthreads();
            const uint32_t lane16 = tid & 15u;
            const uint32_t grp16  = tid >> 4;
            const uint32_t base   = sh_tile * sh_stride_rows;
            const bool     act16  = (lane16 < n_blk);
            const uint4    xa     = act16 ? sm_x[lane16 * 2u]      : make_uint4(0u, 0u, 0u, 0u);
            const uint4    xb     = act16 ? sm_x[lane16 * 2u + 1u] : make_uint4(0u, 0u, 0u, 0u);
            const float2   xds    = act16 ? sh_ds_sm[lane16]       : make_float2(0.0f, 0.0f);
            #pragma nounroll
            for (uint32_t step = 0u; step < sh_steps; ++step) {
                const uint32_t row = base + step * 16u + grp16;
                float part_acc = 0.0f;
                if (act16) {
                    GPtr<const uint4> w4 = as_gptr<const uint4>(desc->sh_dn_qs + static_cast<uint64_t>(row) * desc->sh_dn_stride);
                    GPtr<const uint32_t> dm = as_gptr<const uint32_t>(desc->sh_dn_dm) + static_cast<uint64_t>(row) * n_blk;
                    const uint4     wv  = ldg_u32x4_stream(w4 + lane16);
                    const uint32_t  dmv = ldg_u32_stream(dm + lane16);
                    int32_t dp = sdot4(wv.x & 0x0F0F0F0Fu, xa.x, 0);
                    dp = sdot4(wv.y & 0x0F0F0F0Fu, xa.y, dp);
                    dp = sdot4(wv.z & 0x0F0F0F0Fu, xa.z, dp);
                    dp = sdot4(wv.w & 0x0F0F0F0Fu, xa.w, dp);
                    dp = sdot4((wv.x >> 4) & 0x0F0F0F0Fu, xb.x, dp);
                    dp = sdot4((wv.y >> 4) & 0x0F0F0F0Fu, xb.y, dp);
                    dp = sdot4((wv.z >> 4) & 0x0F0F0F0Fu, xb.z, dp);
                    dp = sdot4((wv.w >> 4) & 0x0F0F0F0Fu, xb.w, dp);
                    const float wd = fp16_to_f32(static_cast<uint16_t>(dmv & 0xFFFFu));
                    const float wm = fp16_to_f32(static_cast<uint16_t>(dmv >> 16));
                    part_acc = fmaf(wd * xds.x, static_cast<float>(dp), wm * xds.y);
                }
                const float acc = vegajet::dot8_dev::sum16(part_acc);
                if (lane16 == 0u) as_gptr<float>(desc->sh_down_out)[row] = acc;
            }
        }
        grid_barrier_wbinvl1(sbar + 96u, gridDim.x, super_sense);
        prof_acc(prof, (desc->w_gu_d8_w != 0ULL) ? 15u : 13u, t_p, prof_on);
    }

    // ---- Stage 4: CU-balanced 240-WG (144x16 + 96x8 = 3072) or 192-WG Routed DOT8 Down + sh_down + 16B FP16 BAR0 PP ----
    {
        const bool use_240 = (gridDim.x == 240u && desc->D == 3072u);
        const uint32_t dn_wgs = use_240 ? 240u : 192u;
        if (wg < dn_wgs) {
            const uint32_t n_blk   = desc->local_ff >> 5;
            const uint32_t n_q8_u4 = 8u * (n_blk << 1);
            const uint32_t n_d     = 8u * n_blk;
            uint32_t t4 = tid;
            asm volatile("" : "+v"(t4));
            if (t4 < n_q8_u4) sm_x[t4]  = as_gptr<const uint4>(desc->mid_q8)[t4];
            if (t4 < n_d)     sm_xd[t4] = as_gptr<const float>(desc->mid_d)[t4];
            __syncthreads();

            const uint32_t grp16  = t4 >> 4;
            const uint32_t lane16 = t4 & 15u;
            const uint32_t n_rows = (use_240 && wg >= 144u) ? 8u : 16u;
            const uint32_t row0   = (use_240 && wg >= 144u) ? (2304u + (wg - 144u) * 8u) : (wg * 16u);
            if (grp16 < n_rows) {
                const uint32_t row = row0 + grp16;
                const float total = vegajet::dot8_dev::vj_dot8_down_row_iq3_flex<2>(
                    as_gptr<const uint8_t>(desc->w_dn_d8_w), desc->dn_w_stride,
                    as_gptr<const uint8_t>(desc->w_dn_d8_s), desc->dn_s_stride,
                    sm_x, sm_xd, sm_ids, sm_w, row, desc->D, n_blk, desc->dn_is_sub32 != 0u, lane16);
                if (lane16 == 0u) {
                    const float v = total + as_gptr<const float>(desc->sh_down_out)[row];
                    as_gptr<float>(desc->x_partial)[row] = v;
                    sm_y[grp16] = v;
                }
            }
            if (out_ar_epoch != 0u && desc->n_ranks > 1u) {
                __syncthreads();
                const uint64_t slot_off = static_cast<uint64_t>(out_ar_epoch & 1u) * VJ_SUPER_AR_SLOT_BYTES;
                if (tid < (n_rows >> 3)) {
                    const uint32_t o = tid * 8u;
                    const __half2 h0 = __float22half2_rn(make_float2(sm_y[o + 0u], sm_y[o + 1u]));
                    const __half2 h1 = __float22half2_rn(make_float2(sm_y[o + 2u], sm_y[o + 3u]));
                    const __half2 h2 = __float22half2_rn(make_float2(sm_y[o + 4u], sm_y[o + 5u]));
                    const __half2 h3 = __float22half2_rn(make_float2(sm_y[o + 6u], sm_y[o + 7u]));
                    u32x4 pv = {
                        *reinterpret_cast<const uint32_t*>(&h0),
                        *reinterpret_cast<const uint32_t*>(&h1),
                        *reinterpret_cast<const uint32_t*>(&h2),
                        *reinterpret_cast<const uint32_t*>(&h3)
                    };
                    #pragma nounroll
                    for (uint32_t dst = 0u; dst < desc->n_ranks; ++dst) {
                        if (dst != desc->rank && desc->out_peer_tx_base[dst] != 0ULL) {
                            __builtin_nontemporal_store(pv, reinterpret_cast<GPtr<u32x4>>(
                                as_gptr<uint16_t>(desc->out_peer_tx_base[dst] + slot_off) + row0 + o));
                        }
                    }
                }
                signal_out_peer_flag_n(sbar, desc->out_peer_flag, desc->rank, desc->n_ranks, dn_wgs, out_ar_epoch);
            }
            prof_acc(prof, 14u, t_p, prof_on);
        }
    }
}

// ============================================================================
// DEDICATED Q3 SUPER-KERNEL B-MoE (240 WGs, kBatch=3 deep prefetch, balanced 240-WG 16B-BAR0 Down, ~16 KB .text!)
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_m1_moe_super_q3_kernel(
    const vegajet::ornith_super::MoeSuperDesc* __restrict__ desc,
    uint32_t ar_epoch,
    uint32_t out_ar_epoch,
    uint32_t super_sense) {
    using namespace vegajet::super_dev;
    __shared__ uint4   sm_x[256];   // 4,096 B
    __shared__ float   sm_xd[128];  //   512 B
    __shared__ float   sm_y[32];    //   128 B
    __shared__ int32_t sm_ids[8];   //    32 B
    __shared__ float   sm_w[8];     //    32 B -> total 4,800 B!

    const uint32_t wg  = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    GPtr<uint32_t> sbar = as_gptr<uint32_t>(desc->super_bar);
    const bool prof_on = ((super_sense & 0x80000000u) != 0u) && (wg == 0u) && (tid == 0u);
    auto p_acc = [&](uint32_t slot) {
        if (prof_on) {
            GPtr<uint64_t> pt = as_gptr<uint64_t>(desc->super_bar + 1280u);
            const uint64_t now = __builtin_amdgcn_s_memrealtime();
            as_gptr<uint64_t>(desc->super_bar + 1024u)[slot] += (now - pt[0]);
            pt[0] = now;
        }
    };
    if (prof_on) as_gptr<uint64_t>(desc->super_bar + 1280u)[0] = __builtin_amdgcn_s_memrealtime();

    // ---- Stage 0: Post-Mixer Prepush FP16 AR (WG 0) ----
    if (ar_epoch != 0u) {
        if (wg == 0u) {
            const uint64_t slot_off = static_cast<uint64_t>(ar_epoch & 1u) * VJ_SUPER_AR_SLOT_BYTES;
            p2p_ar_prepush_m1_n(
                as_gptr<const float>(desc->x_partial),
                desc->ar_local_rx_base,
                desc->ar_local_flag,
                slot_off,
                as_gptr<float>(desc->h_io),
                as_gptr<const float>(desc->w_post_attn_norm),
                as_gptr<float>(desc->x_norm),
                as_gptr<int8_t>(desc->x_q8),
                as_gptr<float2>(desc->x_ds),
                sm_y,
                desc->D, ar_epoch, desc->rank, desc->n_ranks, desc->eps);
        }
        grid_barrier_wbinvl1(sbar, gridDim.x, super_sense);
        p_acc(10u);
    }

    // ---- Stage 1: Balanced 240-WG ShGU + Router + ShGate ----
    {
        uint32_t t1 = tid;
        asm volatile("" : "+v"(t1));
        const uint32_t grp = t1 >> 5;
        const uint32_t l32 = t1 & 31u;
        if (t1 < 192u) sm_x[t1] = as_gptr<const uint4>(desc->x_q8)[t1];
        if (t1 <  96u) {
            const float2 xds = as_gptr<const float2>(desc->x_ds)[t1];
            reinterpret_cast<float2*>(sm_x + 192u)[t1] = xds;
            sm_xd[t1] = xds.x;
        }
        __syncthreads();

        GPtr<const float4> x4f = as_gptr<const float4>(desc->x_norm);
        const uint32_t sh_wgs = (gridDim.x == 240u && desc->sh_gu_rows <= 768u) ? 96u : 128u;
        if (wg < sh_wgs) {
            const uint32_t row = wg * 8u + grp;
            if (row < desc->sh_gu_rows) {
                GPtr<const uint4> w4 = as_gptr<const uint4>(desc->sh_gu_qs + static_cast<uint64_t>(row) * desc->sh_gu_stride);
                GPtr<const uint32_t> dm = as_gptr<const uint32_t>(desc->sh_gu_dm) + static_cast<uint64_t>(row) * 96u;
                const float2* xd = reinterpret_cast<const float2*>(sm_x + 192u);
                float acc = 0.0f;
                uint4 wvs[3]; uint32_t dms[3];
                #pragma unroll
                for (uint32_t r = 0u; r < 3u; ++r) {
                    const uint32_t bc = l32 + r * 32u;
                    wvs[r] = ldg_u32x4_stream(w4 + bc);
                    dms[r] = ldg_u32_stream(dm + bc);
                }
                __builtin_amdgcn_sched_barrier(0);
                #pragma unroll
                for (uint32_t r = 0u; r < 3u; ++r) {
                    const uint32_t bq = l32 + r * 32u;
                    const uint4 wv = wvs[r]; const uint32_t dmv = dms[r];
                    const uint4 xa = sm_x[bq * 2u], xb = sm_x[bq * 2u + 1u];
                    const float2 xds = xd[bq];
                    int32_t dp = sdot4(wv.x & 0x0F0F0F0Fu, xa.x, 0);
                    dp = sdot4(wv.y & 0x0F0F0F0Fu, xa.y, dp);
                    dp = sdot4(wv.z & 0x0F0F0F0Fu, xa.z, dp);
                    dp = sdot4(wv.w & 0x0F0F0F0Fu, xa.w, dp);
                    dp = sdot4((wv.x >> 4) & 0x0F0F0F0Fu, xb.x, dp);
                    dp = sdot4((wv.y >> 4) & 0x0F0F0F0Fu, xb.y, dp);
                    dp = sdot4((wv.z >> 4) & 0x0F0F0F0Fu, xb.z, dp);
                    dp = sdot4((wv.w >> 4) & 0x0F0F0F0Fu, xb.w, dp);
                    const float wd = fp16_to_f32(static_cast<uint16_t>(dmv & 0xFFFFu));
                    const float wm = fp16_to_f32(static_cast<uint16_t>(dmv >> 16));
                    acc = fmaf(wd * xds.x, static_cast<float>(dp), fmaf(wm, xds.y, acc));
                }
                acc = block32_sum(acc);
                if (l32 == 0u) as_gptr<float>(desc->sh_gu_out)[row] = acc;
            }
        } else {
            const uint32_t rwg = wg - sh_wgs;
            uint32_t e_beg = 256u, e_end = 256u;
            bool do_sh_gate = false;
            if (sh_wgs == 96u) {
                if (rwg < 128u)       { e_beg = rwg * 2u; e_end = e_beg + 2u; }
                else if (rwg == 128u) { do_sh_gate = true; }
            } else {
                if (rwg < 78u)        { e_beg = rwg * 2u; e_end = e_beg + 2u; }
                else if (rwg < 111u)  { e_beg = 156u + (rwg - 78u) * 3u; e_end = e_beg + 3u; }
                else if (rwg == 111u) { e_beg = 255u; e_end = 256u; do_sh_gate = true; }
            }
            const uint32_t half_id = tid >> 7;
            const uint32_t l128    = tid & 127u;
            #pragma nounroll
            for (uint32_t e0 = e_beg; e0 < e_end; e0 += 2u) {
                const uint32_t e = e0 + half_id;
                float acc = 0.0f;
                if (e < e_end) {
                    GPtr<const uint4> w4 = reinterpret_cast<GPtr<const uint4>>(
                        as_gptr<const uint16_t>(desc->w_router_f16) + static_cast<uint64_t>(e) * 3200u);
                    uint4 wvs[3];
                    #pragma unroll
                    for (uint32_t r = 0u; r < 3u; ++r) wvs[r] = ldg_u32x4_stream(w4 + l128 + r * 128u);
                    __builtin_amdgcn_sched_barrier(0);
                    #pragma unroll
                    for (uint32_t r = 0u; r < 3u; ++r) {
                        const uint32_t i = l128 + r * 128u;
                        const uint4 wv = wvs[r];
                        const float4 xa = x4f[i * 2u], xb = x4f[i * 2u + 1u];
                        const float2 h0 = __half22float2(*reinterpret_cast<const __half2*>(&wv.x));
                        const float2 h1 = __half22float2(*reinterpret_cast<const __half2*>(&wv.y));
                        const float2 h2 = __half22float2(*reinterpret_cast<const __half2*>(&wv.z));
                        const float2 h3 = __half22float2(*reinterpret_cast<const __half2*>(&wv.w));
                        acc = fmaf(h0.x, xa.x, fmaf(h0.y, xa.y, fmaf(h1.x, xa.z, fmaf(h1.y, xa.w, acc))));
                        acc = fmaf(h2.x, xb.x, fmaf(h2.y, xb.y, fmaf(h3.x, xb.z, fmaf(h3.y, xb.w, acc))));
                    }
                } else if (do_sh_gate && half_id == 1u) {
                    GPtr<const float4> w4 = as_gptr<const float4>(desc->w_sh_gate);
                    #pragma nounroll
                    for (uint32_t i = l128; i < 768u; i += 128u) {
                        const float4 wv = w4[i], xv = x4f[i];
                        acc = fmaf(wv.x, xv.x, fmaf(wv.y, xv.y, fmaf(wv.z, xv.z, fmaf(wv.w, xv.w, acc))));
                    }
                }
                acc = wave64_sum(acc);
                if ((tid & 63u) == 0u) sm_y[tid >> 6] = acc;
                __syncthreads();
                if (l128 == 0u) {
                    const float tot = sm_y[half_id * 2u] + sm_y[half_id * 2u + 1u];
                    if (e < e_end) as_gptr<float>(desc->router_logits)[e] = tot;
                    else if (do_sh_gate && half_id == 1u) as_gptr<float>(desc->sh_gate_sc)[0] = 1.0f / (1.0f + __expf(-tot));
                }
                __syncthreads();
            }
            if (do_sh_gate && sh_wgs == 96u) {
                GPtr<const float4> w4 = as_gptr<const float4>(desc->w_sh_gate);
                float acc = 0.0f;
                #pragma nounroll
                for (uint32_t i = tid; i < 768u; i += 256u) {
                    const float4 wv = w4[i], xv = x4f[i];
                    acc = fmaf(wv.x, xv.x, fmaf(wv.y, xv.y, fmaf(wv.z, xv.z, fmaf(wv.w, xv.w, acc))));
                }
                acc = wave64_sum(acc);
                if ((tid & 63u) == 0u) sm_y[tid >> 6] = acc;
                __syncthreads();
                if (tid == 0u) {
                    const float tot = (sm_y[0] + sm_y[1]) + (sm_y[2] + sm_y[3]);
                    as_gptr<float>(desc->sh_gate_sc)[0] = 1.0f / (1.0f + __expf(-tot));
                }
            }
        }
        grid_barrier_wbinvl1(sbar + 32u, gridDim.x, super_sense);
        p_acc(11u);
    }

    // ---- Stage 2 (BARRIER-FREE!): Local Top-8 in Wave 0 ----
    if (tid < 64u) fast_softmax_top8_wave64(as_gptr<const float>(desc->router_logits), sm_ids, sm_w, tid);
    __syncthreads();
    if (wg == 0u && tid < 8u) {
        as_gptr<int32_t>(desc->top_ids)[tid] = sm_ids[tid];
        as_gptr<float>(desc->top_w)[tid]     = sm_w[tid];
    }
    p_acc(12u);

    // ---- Stage 3: Pure VJ_Q3_IQ2S Gate+Up (kBatch=3 deep prefetch!) + Parallel Shared Down! ----
    {
        const uint32_t local_ff = desc->local_ff;
        const uint32_t nblk16   = local_ff >> 4;
        const uint32_t n_tiles  = 8u * nblk16;
        #pragma nounroll
        for (uint32_t tile = wg; tile < n_tiles; tile += gridDim.x) {
            const uint32_t sub_id = tid >> 4;
            const uint32_t lane16 = tid & 15u;
            uint32_t slot, t_in_slot;
            divmod_slot_u(tile, nblk16, slot, t_in_slot);
            const uint32_t rloc   = t_in_slot * 16u + sub_id;
            const uint32_t exp_id = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(sm_ids[slot]));
            const float2 gu = vegajet::dot8_dev::vj_q3_gu_row_iq2s<6, 3>(
                as_gptr<const uint8_t>(desc->w_gu_d8_w), desc->gu_d8_w_stride,
                as_gptr<const uint8_t>(desc->w_gu_d8_s), desc->gu_d8_s_stride,
                sm_x, sm_xd, exp_id * local_ff, rloc, lane16);
            if (lane16 == 0u) sm_y[sub_id] = (gu.x / (1.0f + __expf(-gu.x))) * gu.y;
            __syncthreads();
            GPtr<uint32_t> pair_cnt = as_gptr<uint32_t>(desc->super_bar + 1536u);
            GPtr<float> mf32 = as_gptr<float>(desc->mid_f32);
            const uint32_t blk    = tile >> 1;
            const uint32_t is_odd = tile & 1u;
            GPtr<float> dst = mf32 + tile * 16u;
            if (tid < 16u) dst[tid] = sm_y[tid];
            asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
            __syncthreads();
            if (tid == 0u) sm_y[16] = __uint_as_float(__atomic_fetch_xor(pair_cnt + blk, 1u, __ATOMIC_RELAXED));
            __syncthreads();
            if (__float_as_uint(sm_y[16]) == 1u && tid < 32u) {
                GPtr<const float> other = mf32 + (tile ^ 1u) * 16u;
                const float y = (tid < 16u)
                    ? (is_odd ? ldg_f32_stream(other + tid) : sm_y[tid])
                    : (is_odd ? sm_y[tid - 16u] : ldg_f32_stream(other + tid - 16u));
                const float amax = block32_max(fabsf(y));
                const float d    = amax * (1.0f / 127.0f);
                const float inv  = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                as_gptr<int8_t>(desc->mid_q8)[blk * 32u + tid] =
                    static_cast<int8_t>(max(-127, min(127, __float2int_rn(y * inv))));
                if (tid == 0u) as_gptr<float>(desc->mid_d)[blk] = d;
            }
            __syncthreads();
        }

        // TP=3 (n_tiles <= 192): run 48 Shared Down WGs x 64 rows strictly on IDLE WGs 192..239 (0 Gate+Up collision!).
        // TP=2 (n_tiles == 256): 64 tiles x 48 rows on wg 16..59 + 76..95 (0 overlap with 2-tile CUs 0..15!).
        uint32_t sh_tile = 64u;
        uint32_t sh_steps = 3u;
        uint32_t sh_stride_rows = 48u;
        if (n_tiles <= 192u) {
            if (wg >= 192u && wg < 240u) {
                sh_tile = wg - 192u;
                sh_steps = 4u;
                sh_stride_rows = 64u;
            }
        } else {
            if (wg >= 16u && wg < 60u)      sh_tile = wg - 16u;
            else if (wg >= 76u && wg < 96u) sh_tile = 44u + (wg - 76u);
        }
        asm volatile("" : "+s"(sh_tile));
        if (sh_tile < 64u) {
            const uint32_t n_blk = local_ff >> 5;
            const float sc_g = as_gptr<const float>(desc->sh_gate_sc)[0];
            GPtr<const float> g_p = as_gptr<const float>(desc->sh_gu_out);
            GPtr<const float> u_p = g_p + local_ff;
            int8_t* sh_q8_sm = reinterpret_cast<int8_t*>(sm_x);
            float2* sh_ds_sm = reinterpret_cast<float2*>(sm_x + 32u);
            const uint32_t l32 = tid & 31u;
            #pragma nounroll
            for (uint32_t blk = tid >> 5; blk < n_blk; blk += 8u) {
                const uint32_t c = (blk << 5) + l32;
                const float gv = g_p[c], uv = u_p[c];
                const float y  = (gv / (1.0f + __expf(-gv))) * uv * sc_g;
                const float amax  = block32_max(fabsf(y));
                const float bsum  = block32_sum(y);
                const float d     = amax * (1.0f / 127.0f);
                const float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                sh_q8_sm[c] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(y * inv_d))));
                if (l32 == 0u) sh_ds_sm[blk] = make_float2(d, bsum);
            }
            __syncthreads();
            const uint32_t lane16 = tid & 15u;
            const uint32_t grp16  = tid >> 4;
            const uint32_t base   = sh_tile * sh_stride_rows;
            const bool     act16  = (lane16 < n_blk);
            const uint4    xa     = act16 ? sm_x[lane16 * 2u]      : make_uint4(0u, 0u, 0u, 0u);
            const uint4    xb     = act16 ? sm_x[lane16 * 2u + 1u] : make_uint4(0u, 0u, 0u, 0u);
            const float2   xds    = act16 ? sh_ds_sm[lane16]       : make_float2(0.0f, 0.0f);
            #pragma nounroll
            for (uint32_t st = 0u; st < sh_steps; ++st) {
                const uint32_t row = base + st * 16u + grp16;
                float part_acc = 0.0f;
                if (act16) {
                    GPtr<const uint4> w4 = as_gptr<const uint4>(desc->sh_dn_qs + static_cast<uint64_t>(row) * desc->sh_dn_stride);
                    GPtr<const uint32_t> dm = as_gptr<const uint32_t>(desc->sh_dn_dm) + static_cast<uint64_t>(row) * n_blk;
                    const uint4 wv = ldg_u32x4_stream(w4 + lane16);
                    const uint32_t dmv = ldg_u32_stream(dm + lane16);
                    int32_t dp = sdot4(wv.x & 0x0F0F0F0Fu, xa.x, 0);
                    dp = sdot4(wv.y & 0x0F0F0F0Fu, xa.y, dp);
                    dp = sdot4(wv.z & 0x0F0F0F0Fu, xa.z, dp);
                    dp = sdot4(wv.w & 0x0F0F0F0Fu, xa.w, dp);
                    dp = sdot4((wv.x >> 4) & 0x0F0F0F0Fu, xb.x, dp);
                    dp = sdot4((wv.y >> 4) & 0x0F0F0F0Fu, xb.y, dp);
                    dp = sdot4((wv.z >> 4) & 0x0F0F0F0Fu, xb.z, dp);
                    dp = sdot4((wv.w >> 4) & 0x0F0F0F0Fu, xb.w, dp);
                    const float wd = fp16_to_f32(static_cast<uint16_t>(dmv & 0xFFFFu));
                    const float wm = fp16_to_f32(static_cast<uint16_t>(dmv >> 16));
                    part_acc = fmaf(wd * xds.x, static_cast<float>(dp), wm * xds.y);
                }
                const float acc = vegajet::dot8_dev::sum16(part_acc);
                if (lane16 == 0u) as_gptr<float>(desc->sh_down_out)[row] = acc;
            }
        }
        grid_barrier_wbinvl1(sbar + 96u, gridDim.x, super_sense);
        p_acc(15u);
    }

    // ---- Stage 4: CU-balanced 240-WG (144x16 + 96x8 = 3072, 16B u32x4 BAR0 PP!) Routed DOT8 Down ----
    if (wg < 240u) {
        const uint32_t n_blk   = desc->local_ff >> 5;
        const uint32_t n_q8_u4 = 8u * (n_blk << 1);
        const uint32_t n_d     = 8u * n_blk;
        uint32_t t4 = tid;
        asm volatile("" : "+v"(t4));
        if (t4 < n_q8_u4) sm_x[t4]  = as_gptr<const uint4>(desc->mid_q8)[t4];
        if (t4 < n_d)     sm_xd[t4] = as_gptr<const float>(desc->mid_d)[t4];
        __syncthreads();
        const uint32_t grp16  = t4 >> 4;
        const uint32_t lane16 = t4 & 15u;
        const uint32_t n_rows = (wg < 144u) ? 16u : 8u;
        const uint32_t row0   = (wg < 144u) ? (wg * 16u) : (2304u + (wg - 144u) * 8u);
        if (grp16 < n_rows) {
            const uint32_t row = row0 + grp16;
            const float total = vegajet::dot8_dev::vj_dot8_down_row_iq3_flex<2>(
                as_gptr<const uint8_t>(desc->w_dn_d8_w), desc->dn_w_stride,
                as_gptr<const uint8_t>(desc->w_dn_d8_s), desc->dn_s_stride,
                sm_x, sm_xd, sm_ids, sm_w, row, desc->D, n_blk, desc->dn_is_sub32 != 0u, lane16);
            if (lane16 == 0u) {
                const float v = total + as_gptr<const float>(desc->sh_down_out)[row];
                as_gptr<float>(desc->x_partial)[row] = v;
                sm_y[grp16] = v;
            }
        }
        if (out_ar_epoch != 0u && desc->n_ranks > 1u) {
            __syncthreads();
            const uint64_t slot_off = static_cast<uint64_t>(out_ar_epoch & 1u) * VJ_SUPER_AR_SLOT_BYTES;
            if (tid < (n_rows >> 3)) {
                const uint32_t o = tid * 8u;
                const __half2 h0 = __float22half2_rn(make_float2(sm_y[o + 0u], sm_y[o + 1u]));
                const __half2 h1 = __float22half2_rn(make_float2(sm_y[o + 2u], sm_y[o + 3u]));
                const __half2 h2 = __float22half2_rn(make_float2(sm_y[o + 4u], sm_y[o + 5u]));
                const __half2 h3 = __float22half2_rn(make_float2(sm_y[o + 6u], sm_y[o + 7u]));
                u32x4 pv = {
                    *reinterpret_cast<const uint32_t*>(&h0),
                    *reinterpret_cast<const uint32_t*>(&h1),
                    *reinterpret_cast<const uint32_t*>(&h2),
                    *reinterpret_cast<const uint32_t*>(&h3)
                };
                #pragma nounroll
                for (uint32_t dst = 0u; dst < desc->n_ranks; ++dst) {
                    if (dst != desc->rank && desc->out_peer_tx_base[dst] != 0ULL) {
                        __builtin_nontemporal_store(pv, reinterpret_cast<GPtr<u32x4>>(
                            as_gptr<uint16_t>(desc->out_peer_tx_base[dst] + slot_off) + row0 + o));
                    }
                }
            }
            signal_out_peer_flag_n(sbar, desc->out_peer_flag, desc->rank, desc->n_ranks, 240u, out_ar_epoch);
        }
        p_acc(14u);
    }
}

// Ring-verify (M = 2..8) router front end in the exact M = 1 super order: each expert logit is one 128-lane
// half-WG dot (3 uint4 packets/lane at D = 3072, wave64_sum per wave, wave0 + wave1), the shared scalar gate is
// the super's 128-lane half-WG loop, and top-8 selects raw logits then normalises the eight exponentials.
// Grid (129 WG, M): WG 0..127 -> experts 2*wg + half, WG 128 half 1 -> shared gate. Same kernargs as
// ornith_moe_router_fp16_logits_kernel.
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_ring_router_super_order_kernel(
    uint64_t w_router_f16, uint64_t w_sh_gate, uint64_t x_norm, uint64_t logits_out, uint64_t sh_gate_out,
    uint32_t M, uint32_t D) {
    using namespace vegajet::super_dev;
    __shared__ float sm_y[4];
    const uint32_t m = blockIdx.y;
    if (m >= M || D != 3072u) return;
    const uint32_t tid = threadIdx.x, wg = blockIdx.x;
    const uint32_t half_id = tid >> 7, l128 = tid & 127u;
    GPtr<const float4> x4f = as_gptr<const float4>(x_norm) + static_cast<uint64_t>(m) * (D >> 2);
    float acc = 0.0f;
    if (wg < 128u) {
        const uint32_t e = wg * 2u + half_id;
        GPtr<const uint4> w4 = reinterpret_cast<GPtr<const uint4>>(
            as_gptr<const uint16_t>(w_router_f16) + static_cast<uint64_t>(e) * 3200u);
        uint4 wvs[3];
        #pragma unroll
        for (uint32_t r = 0u; r < 3u; ++r) wvs[r] = ldg_u32x4_stream(w4 + l128 + r * 128u);
        #pragma unroll
        for (uint32_t r = 0u; r < 3u; ++r) {
            const uint32_t i = l128 + r * 128u;
            const uint4 wv = wvs[r];
            const float4 xa = x4f[i * 2u], xb = x4f[i * 2u + 1u];
            const float2 h0 = __half22float2(*reinterpret_cast<const __half2*>(&wv.x));
            const float2 h1 = __half22float2(*reinterpret_cast<const __half2*>(&wv.y));
            const float2 h2 = __half22float2(*reinterpret_cast<const __half2*>(&wv.z));
            const float2 h3 = __half22float2(*reinterpret_cast<const __half2*>(&wv.w));
            acc = fmaf(h0.x, xa.x, fmaf(h0.y, xa.y, fmaf(h1.x, xa.z, fmaf(h1.y, xa.w, acc))));
            acc = fmaf(h2.x, xb.x, fmaf(h2.y, xb.y, fmaf(h3.x, xb.z, fmaf(h3.y, xb.w, acc))));
        }
    } else if (half_id == 1u) {
        GPtr<const float4> w4 = as_gptr<const float4>(w_sh_gate);
        const uint32_t n4 = D >> 2;
        #pragma nounroll
        for (uint32_t i = l128; i < n4; i += 128u) {
            const float4 wv = w4[i], xv = x4f[i];
            acc = fmaf(wv.x, xv.x, fmaf(wv.y, xv.y, fmaf(wv.z, xv.z, fmaf(wv.w, xv.w, acc))));
        }
    }
    acc = wave64_sum(acc);
    if ((tid & 63u) == 0u) sm_y[tid >> 6] = acc;
    __syncthreads();
    if (l128 == 0u) {
        const float tot = sm_y[half_id * 2u] + sm_y[half_id * 2u + 1u];
        if (wg < 128u) as_gptr<float>(logits_out)[static_cast<uint64_t>(m) * 256u + wg * 2u + half_id] = tot;
        else if (half_id == 1u) as_gptr<float>(sh_gate_out)[m] = 1.0f / (1.0f + __expf(-tot));
    }
}

extern "C" __global__ __launch_bounds__(256, 4)
void ornith_ring_top8_super_order_kernel(uint64_t logits_in, uint64_t top_ids, uint64_t top_w, uint32_t M) {
    using namespace vegajet::super_dev;
    __shared__ int32_t sm_ids[8];
    __shared__ float   sm_w[8];
    const uint32_t m = blockIdx.x, tid = threadIdx.x;
    if (m >= M) return;
    if (tid < 64u) fast_softmax_top8_wave64(as_gptr<const float>(logits_in) + static_cast<uint64_t>(m) * 256u, sm_ids, sm_w, tid);
    __syncthreads();
    if (tid < 8u) {
        as_gptr<int32_t>(top_ids)[static_cast<uint64_t>(m) * 8u + tid] = sm_ids[tid];
        as_gptr<float>(top_w)[static_cast<uint64_t>(m) * 8u + tid]     = sm_w[tid];
    }
}
