// Isolated L4 copy of the baseline GQA super-kernel and its device helpers.
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


#include "ops/kernels/vj_kvq_device.hiph"
#include "ops/ornith_gqa_kvq_launch.hpp"
namespace kvq_dev = vegajet::kvq_dev;
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_m1_gqa_super_kvq_kernel(
    const vegajet::ornith_super::GqaSuperDesc* __restrict__ desc,
    uint32_t prev_ar_epoch,
    uint32_t out_ar_epoch,
    uint32_t pos0,
    uint32_t super_sense,
    const int8_t* signs, uint32_t format, uint32_t chronological) {
    using namespace vegajet::super_dev;
    __shared__ uint4  sm_xq[256];
    __shared__ float2 sm_xds[128];
    __shared__ float  sm_v[256];
    __shared__ float  sm_red[32];

    // Opaque uniform mode reads at entry; do not fold mode tests into loop latches.
    format = __builtin_amdgcn_readfirstlane(format);
    chronological = __builtin_amdgcn_readfirstlane(chronological);
    asm volatile("" : "+s"(format), "+s"(chronological));
    const uint32_t cache_stride = kvq_dev::stride(format, 256);
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

        if (wg < n_q_local + n_kv_local) {
            const bool is_q = wg < n_q_local;
            const uint32_t kv_h = wg - n_q_local;
            GPtr<const float> raw = qkv_f + (is_q ? wg * 512u : qg_tot + kv_h * 256u);
            GPtr<const float> vr = qkv_f + qg_tot + kv_tot + kv_h * 256u;
            const float x = raw[d];
            const float sq = wave64_sum(x * x);
            if (lane == 0u) sm_red[wave] = sq;
            __syncthreads();
            const float inv_r = rsqrtf((sm_red[0] + sm_red[1] + sm_red[2] + sm_red[3]) * (1.0f / 256.0f) + eps);
            sm_v[d] = x * inv_r * as_gptr<const float>(is_q ? desc->q_norm_w : desc->k_norm_w)[d];
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
            // Every wave must finish reading the RoPE partner before LDS is reused.
            __syncthreads();
            if (is_q) {
                if (format == 3u) {
                    sm_v[d] = k_rot * signs[d];
                    __syncthreads();
                    kvq_dev::transform(sm_v, 256, d);
                    k_rot = sm_v[d];
                }
                as_gptr<float>(desc->q_buf)[static_cast<uint64_t>(wg) * 256u + d] = k_rot;
            } else {
                const uint64_t slot_off = (static_cast<uint64_t>(pos0) * n_kv_local + kv_h) * cache_stride;
                // One encoding body shared by K/V keeps the refit loop below the ISA-size gate.
                #pragma nounroll
                for (uint32_t which = 0; which < 2; ++which) {
                    auto cache = reinterpret_cast<uint8_t*>((which ? desc->v_cache_q8 : desc->k_cache_q8) + slot_off);
                    sm_v[d] = (which ? vr[d] : k_rot) * (format == 3u ? signs[d] : 1);
                    __syncthreads();
                    if (format == 3u) kvq_dev::transform(sm_v, 256, d);
                    kvq_dev::append_vector(sm_v, cache, 256, format, d);
                    __syncthreads();
                }
            }
        }
        grid_barrier_wbinvl1(sbar + 64u, gridDim.x, super_sense);
        p_acc(6u);
    }

    // ---- Stage 2b: In-Kernel 192-WG Split-KV FlashAttention Partial (constant chunk_len=32 up to 1920, bit-exact with verify!) ----
    const uint32_t span = pos0 + 1u;
    const uint32_t chunk_len = vegajet::ornith::kvq_super::chunk_length(span, chronological != 0u);
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
                    #pragma nounroll
                    for (uint32_t j = 0u; j < 4u; ++j) {
                        const uint32_t t = min(t0 + j, t_end - 1u);
                        const uint64_t off = (static_cast<uint64_t>(t) * n_kv_local + kv_h) * cache_stride;
                        GPtr<const uint8_t> kc = k_cache + off;
                        GPtr<const uint8_t> vc = v_cache + off;
                        const float kv = kvq_dev::elem(reinterpret_cast<const uint8_t*>(kc), format, 256, d);
                        vvs[j] = kvq_dev::elem(reinterpret_cast<const uint8_t*>(vc), format, 256, d);
                        dots[j] = qv * kv;
                    }
                    #pragma nounroll
                    for (uint32_t j = 0u; j < 4u; ++j) {
                        const float dot = wave64_sum(dots[j]);
                        if (lane == 0u) sm_red[j * 4u + wave] = dot;
                    }
                    __syncthreads();
                    #pragma nounroll
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

                const uint64_t idx = static_cast<uint64_t>(h) * n_chunks + ch;
                as_gptr<float>(desc->part_o)[idx * 256u + d] = o_d;
                if (d == 0u) as_gptr<float2>(desc->part_ml)[idx] = make_float2(m_run, l_run);
            }
        }
        grid_barrier_wbinvl1(sbar + 96u, gridDim.x, super_sense);
        p_acc(7u);
    }

    // ---- Stage 2c: Always merge, inverse-rotate Turbo3 once, then gate + Q8 quant. ----
    {
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
            float attn_v = (l_tot > 0.0f) ? (o_tot / l_tot) : 0.0f;
            if (format == 3u) {
                sm_v[d] = attn_v;
                __syncthreads();
                kvq_dev::transform(sm_v, 256, d);
                attn_v = sm_v[d] * signs[d];
            }
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

