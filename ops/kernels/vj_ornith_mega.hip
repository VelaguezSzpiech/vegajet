// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// VegaJet Wave 2: Persistent Block-Fused M=1 Megakernels for Ornith-122B TP=3 (gfx906).
//   1. ornith_m1_gdn_moe_loop_q3_kernel : runs 3 consecutive (GDN + MoE Q3) layers in 1 dispatch -> 12 dispatches!
//   2. ornith_m1_gqa_moe_q3_kernel      : runs 1 (GQA + MoE Q3) layer (+ optional Final AR) in 1 dispatch -> 12 dispatches!
// Total: 100 -> 27 dispatches/token (-73 AQL launches): embed + norm + 24 megakernels + vocab.
// gfx906 contract: 240 WGs, TP=3, D=3072; 0 B private, <=64 VGPRs, <24,576 B per entry plus shared helpers.
// Projection row counts are multiples of eight; peer pointers for both other ranks are valid.
// Decode loop/barrier generations remain in LDS; build with -mllvm -disable-machine-licm.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#include "ops/kernels/vj_moe_dot8.hiph"
#include "ops/kernels/vj_ornith_super_desc.hpp"

#define VJ_MEGA_RING_SLOTS 17u
#define VJ_MEGA_AR_SLOT_BYTES (2ULL * 1024ULL * 1024ULL)

namespace vegajet {
namespace mega_dev {

typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef float    f32x4 __attribute__((ext_vector_type(4)));
template <typename T>
using GPtr = T __attribute__((address_space(1)))*;
template <typename T>
__device__ __forceinline__ GPtr<T> as_gptr(uint64_t va) {
    return reinterpret_cast<GPtr<T>>(va);
}

__device__ __forceinline__ uint32_t flag_opaque(uint32_t value, uint32_t bit) {
    uint32_t b = __builtin_amdgcn_readfirstlane((value >> bit) & 1u);
    asm volatile("" : "+s"(b));
    return b;
}

__device__ __forceinline__ uint32_t fresh_tid() {
    uint32_t t = threadIdx.x;
    asm volatile("" : "+v"(t));
    return t;
}
__device__ __forceinline__ uint32_t fresh_wg() {
    return __builtin_amdgcn_readfirstlane(blockIdx.x);
}
__device__ __forceinline__ GPtr<const vegajet::ornith_super::Mega48Desc> fresh_mega(
    GPtr<const vegajet::ornith_super::Mega48Desc> m) {
    return m;
}
__device__ __forceinline__ uint32_t rfl(uint32_t x) {
    return __builtin_amdgcn_readfirstlane(x);
}

__device__ __forceinline__ int32_t sdot4(uint32_t a, uint32_t b, int32_t c) {
    return __builtin_amdgcn_sdot4(static_cast<int32_t>(a), static_cast<int32_t>(b), c, false);
}
// Exact division with shared denominator computation for SiLU and scalar sigmoid.
__device__ __noinline__ float activation_exact(float x, float numerator) {
    return numerator / (1.0f + __expf(-x));
}
__device__ __forceinline__ float silu_exact(float x) { return activation_exact(x, x); }
__device__ __forceinline__ float sigmoid_exact(float x) { return activation_exact(x, 1.0f); }
__device__ __noinline__ uint32_t quant_pack4(float4 y, float inv_d) {
    return (static_cast<uint32_t>(max(-127, min(127, __float2int_rn(y.x * inv_d)))) & 0xFFu)
         | ((static_cast<uint32_t>(max(-127, min(127, __float2int_rn(y.y * inv_d)))) & 0xFFu) << 8)
         | ((static_cast<uint32_t>(max(-127, min(127, __float2int_rn(y.z * inv_d)))) & 0xFFu) << 16)
         | ((static_cast<uint32_t>(max(-127, min(127, __float2int_rn(y.w * inv_d)))) & 0xFFu) << 24);
}
__device__ __noinline__ int32_t dot32_q4(uint4 w, uint4 a, uint4 b) {
    int32_t dp = sdot4(w.x & 0x0F0F0F0Fu, a.x, 0);
    dp = sdot4(w.y & 0x0F0F0F0Fu, a.y, dp);
    dp = sdot4(w.z & 0x0F0F0F0Fu, a.z, dp);
    dp = sdot4(w.w & 0x0F0F0F0Fu, a.w, dp);
    dp = sdot4((w.x >> 4) & 0x0F0F0F0Fu, b.x, dp);
    dp = sdot4((w.y >> 4) & 0x0F0F0F0Fu, b.y, dp);
    dp = sdot4((w.z >> 4) & 0x0F0F0F0Fu, b.z, dp);
    return sdot4((w.w >> 4) & 0x0F0F0F0Fu, b.w, dp);
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

__device__ __noinline__ float wave64_sum(float v) {
    v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0xB1, 0xF, 0xF, true));
    v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x4E, 0xF, 0xF, true));
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x101F));
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x201F));
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x401F));
    const uint32_t lane = __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
    v += __int_as_float(__builtin_amdgcn_ds_bpermute((lane ^ 32u) * 4u, __float_as_int(v)));
    return v;
}
__device__ __noinline__ float block32_sum(float v) {
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x401F));
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x201F));
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x101F));
    v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x4E, 0xF, 0xF, true));
    v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0xB1, 0xF, 0xF, true));
    return v;
}
__device__ __noinline__ float block32_max(float v) {
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x401F)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x201F)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x101F)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x4E, 0xF, 0xF, true)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0xB1, 0xF, 0xF, true)));
    return v;
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

// The barrier generation and slot state are shared by both persistent kernels.
// Outlining the complete operation avoids repeating its LDS/readfirstlane prologue.
__device__ __forceinline__ GPtr<uint32_t> scratch_bar(uint32_t* sm_st) {
    return as_gptr<uint32_t>(static_cast<uint64_t>(rfl(sm_st[11])) |
                            (static_cast<uint64_t>(rfl(sm_st[12])) << 32));
}
// Event encoding: bit31 executes the grid barrier; low slot31 means no stamp.
// Start events encode (record << 5) | 16; endpoints use SUPER_PROFILE=2 slots.
// sm_st[8]=profile enable, [9]=record, [10]=reserved, [11..12]=immutable scratch VA.
__device__ __noinline__ void stage_event(uint32_t* sm_st, uint32_t slot) {
    slot = rfl(slot);
    asm volatile("" : "+s"(slot));
    if ((slot & 0x80000000u) != 0u) {
        // Only lane zero consumes/updates generation state. The entry/exit barriers
        // suffice; reading LDS state in every wave required a third barrier before.
        asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
        __syncthreads();
        if (threadIdx.x == 0u) {
            const uint32_t ns = sm_st[2] + 1u;
            const uint32_t bi = sm_st[3];
            sm_st[2] = ns; sm_st[3] = bi + 1u;
            auto bar = scratch_bar(sm_st) + 512u + ((bi & 3u) << 5);
            const uint32_t arr = __atomic_fetch_add(&bar[0], 1u, __ATOMIC_RELAXED) + 1u;
            if (arr == 240u) {
                __atomic_store_n(&bar[0], 0u, __ATOMIC_RELAXED);
                asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
                __atomic_store_n(&bar[16], ns, __ATOMIC_RELEASE);
            } else {
                #pragma nounroll
                for (uint32_t spin = 0u; spin < 5000000u; ++spin) {
                    if (load_u32_glc(&bar[16]) == ns) break;
                    __builtin_amdgcn_s_sleep(1);
                }
            }
            asm volatile("buffer_wbinvl1\n" ::: "memory");
        }
        __syncthreads();
    }
    slot &= 0x7fffffffu;
    if (rfl(sm_st[8]) == 0u || slot == 31u) return;
    if ((slot & 16u) != 0u) {
        if (threadIdx.x == 0u) sm_st[9] = slot >> 5u;
        slot = 16u;
    }
    __syncthreads();
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
    if (threadIdx.x == 0u) {
        const uint64_t now = __builtin_amdgcn_s_memrealtime();
        auto p = reinterpret_cast<GPtr<uint64_t>>(scratch_bar(sm_st) + 1024u) + rfl(sm_st[9]) * 17u + slot;
        if (slot == 16u) {
            if (blockIdx.x == 0u) *p = now;
        } else {
            __atomic_fetch_max(p, now, __ATOMIC_RELAXED);
        }
    }
}

__device__ __forceinline__ void signal_out_peer_flag_mega(
    GPtr<uint32_t>  sbar,
    const uint64_t* out_peer_flag,
    uint32_t        rank,
    uint32_t        /*n_ranks_arg*/,
    uint32_t        out_ar_epoch) {
    constexpr uint32_t n_ranks = 3u;
    asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0u) {
        const uint32_t arr = __atomic_fetch_add(&sbar[160], 1u, __ATOMIC_RELAXED) + 1u;
        if (arr == 240u) {
            __atomic_store_n(&sbar[160], 0u, __ATOMIC_RELAXED);
            if (n_ranks > 1u) {
                __threadfence_system();
                #pragma nounroll
                for (uint32_t dst = 0u; dst < n_ranks; ++dst) {
                    if (dst != rank) {
                        __atomic_store_n(as_gptr<uint32_t>(out_peer_flag[dst]), out_ar_epoch, __ATOMIC_RELEASE);
                    }
                }
            }
            asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
            __atomic_store_n(&sbar[176], out_ar_epoch, __ATOMIC_RELEASE);
        }
    }
}

__device__ __forceinline__ float wave64_max_dpp(float v) {
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0xB1, 0xF, 0xF, true)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x4E, 0xF, 0xF, true)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x114, 0xF, 0xF, false)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x118, 0xF, 0xF, false)));
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
    const float4 xv = reinterpret_cast<GPtr<const float4>>(logits)[l];
    float x0 = xv.x, x1 = xv.y, x2 = xv.z, x3 = xv.w;
    float lm = fmaxf(fmaxf(x0, x1), fmaxf(x2, x3));
    float mx = 0.0f;
    #pragma nounroll
    for (uint32_t r = 0u; r < 8u; ++r) {
        const float best = wave64_max_dpp(lm);
        if (r == 0u) mx = best;
        const uint64_t ma = __ballot(lm == best);
        const uint32_t wl = static_cast<uint32_t>(__ffsll(static_cast<long long>(ma)) - 1);
        const uint32_t local_q = (x0 == best) ? 0u : ((x1 == best) ? 1u : ((x2 == best) ? 2u : 3u));
        const uint32_t wq = __shfl(local_q, wl, 64);
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
    if (l < 8u) top_w[l] = __expf(top_w[l] - mx);
    if (l == 0u) {
        float top_sum = 0.0f;
        #pragma nounroll
        for (uint32_t r = 0u; r < 8u; ++r) top_sum += top_w[r];
        const float inv_w = 1.0f / fmaxf(top_sum, 1e-30f);
        #pragma nounroll
        for (uint32_t r = 0u; r < 8u; ++r) top_w[r] *= inv_w;
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

__device__ __forceinline__ __amdgpu_buffer_rsrc_t make_srd(uint64_t va) {
    const uint64_t a = vegajet::dot8_dev::uniform_u64(va);
    return __builtin_amdgcn_make_buffer_rsrc(reinterpret_cast<void*>(a), static_cast<short>(0), 0x7FFFFFFF, 0x00027000);
}
__device__ __forceinline__ uint4 mubuf_x4(__amdgpu_buffer_rsrc_t rsrc, uint32_t voff, uint32_t soff) {
    return __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(rsrc, voff, soff, 3));
}
__device__ __forceinline__ uint32_t mubuf_u32(__amdgpu_buffer_rsrc_t rsrc, uint32_t voff, uint32_t soff) {
    return __builtin_amdgcn_raw_buffer_load_b32(rsrc, voff, soff, 3);
}

// Single-instance compact MUBUF Q4_A8 persistent GEMV (zero duplication across GDN/GQA QKV & Out!).
__device__ __forceinline__ void gemv_q4a8_mega(
    uint64_t w_qs_va, uint64_t w_dm_va,
    GPtr<const int8_t> __restrict__ x_q8,
    GPtr<const float2> __restrict__ x_ds,
    GPtr<float>        __restrict__ y_out,
    const uint64_t*                 out_peer_tx_base,
    uint64_t                        slot_off,
    uint32_t                        rank,
    uint32_t                        n_ranks,
    bool                            do_prepush,
    uint4*             sm_xq,
    float2*            sm_xds,
    float*             sm_out8,
    uint32_t rows, uint32_t cols, uint32_t qs_stride) {
    const uint32_t tid   = fresh_tid();
    const uint32_t wg    = fresh_wg();
    const uint32_t grp   = tid >> 5;
    const uint32_t lane  = tid & 31u;
    const uint32_t nb    = cols >> 5;
    const uint32_t n_rg  = rows >> 3;
    GPtr<const uint4> x4 = reinterpret_cast<GPtr<const uint4>>(x_q8);
    if (tid < (nb << 1)) sm_xq[tid]  = x4[tid];
    if (tid < nb)        sm_xds[tid] = x_ds[tid];
    __syncthreads();

    const __amdgpu_buffer_rsrc_t srd_w  = make_srd(w_qs_va);
    const __amdgpu_buffer_rsrc_t srd_dm = make_srd(w_dm_va);
    const uint32_t dm_row_bytes = nb * 4u;

    uint32_t iter = 0u;
    #pragma nounroll
    for (uint32_t rg = wg; rg < n_rg; rg += 240u) {
        const uint32_t row = rg * 8u + grp;
        float acc = 0.0f;
        {
            const uint32_t w_voff  = row * qs_stride;
            const uint32_t dm_voff = row * dm_row_bytes;
            uint4    wv_cur  = mubuf_x4(srd_w,  w_voff  + lane * 16u, 0u);
            uint32_t dmv_cur = mubuf_u32(srd_dm, dm_voff + lane * 4u, 0u);
            #pragma nounroll
            for (uint32_t b = lane; b < nb; b += 32u) {
                const uint32_t bn = b + 32u;
                uint4    wv_nxt  = (bn < nb) ? mubuf_x4(srd_w,  w_voff  + bn * 16u, 0u) : wv_cur;
                uint32_t dmv_nxt = (bn < nb) ? mubuf_u32(srd_dm, dm_voff + bn * 4u, 0u) : dmv_cur;
                const uint4    xa  = sm_xq[b * 2u];
                const uint4    xb  = sm_xq[b * 2u + 1u];
                const float2   xds = sm_xds[b];
                const int32_t dp = dot32_q4(wv_cur, xa, xb);
                const float wd = fp16_to_f32(static_cast<uint16_t>(dmv_cur & 0xFFFFu));
                const float wm = fp16_to_f32(static_cast<uint16_t>(dmv_cur >> 16));
                acc = fmaf(wd * xds.x, static_cast<float>(dp), fmaf(wm, xds.y, acc));
                wv_cur = wv_nxt; dmv_cur = dmv_nxt;
            }
            acc = block32_sum(acc);
            if (lane == 0u) y_out[row] = acc;
        }
        if (do_prepush) {
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
                    if (dst != rank) {
                        __builtin_nontemporal_store(pv, reinterpret_cast<GPtr<u32x4>>(out_peer_tx_base[dst] + slot_off) + rg);
                    }
                }
            }
        }
        ++iter;
    }
}

// Compact low-VGPR P2P Prepush FP16 AR + Residual + Optional RMSNorm + Q8 Quant (WG 0 only).
// Re-reads L1-hot h_io in pass 2 instead of holding ha[2]/hb[2] in 16 live VGPRs -> saves 16 VGPRs & ~2 KB .text!
__device__ __forceinline__ void p2p_ar_prepush_mega(
    GPtr<uint32_t>                 sbar,
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
    uint32_t /*D_arg*/, uint32_t epoch, uint32_t rank, uint32_t /*n_ranks_arg*/,
    bool do_norm, float eps) {
    constexpr uint32_t D = 3072u, n_ranks = 3u;
    const uint32_t tid  = fresh_tid();
    const uint32_t wave = tid >> 6;
    const uint32_t lane = tid & 63u;
    const uint32_t n8   = D >> 3;

    if (tid == 0u) {
        #pragma nounroll
        for (uint32_t spin = 0u; spin < 5000000u; ++spin) {
            if (load_u32_glc(&sbar[176]) == epoch) break;
            __builtin_amdgcn_s_sleep(1);
        }
        if (n_ranks > 1u) {
            #pragma nounroll
            for (uint32_t src = 0u; src < n_ranks; ++src) {
                if (src != rank) {
                    GPtr<const uint32_t> fp = as_gptr<const uint32_t>(ar_local_flag[src]);
                    #pragma nounroll
                    for (uint32_t spin = 0u; spin < 5000000u; ++spin) {
                        if (load_u32_glc(fp) == epoch) break;
                        __builtin_amdgcn_s_sleep(1);
                    }
                }
            }
        }
        asm volatile("buffer_wbinvl1\n" ::: "memory");
    }
    __syncthreads();

    GPtr<const float4> xp4 = reinterpret_cast<GPtr<const float4>>(x_partial);
    GPtr<float4>       h4  = reinterpret_cast<GPtr<float4>>(h_io);

    float ss = 0.0f;
    #pragma nounroll
    for (uint32_t i = tid; i < n8; i += 256u) {
        const float4 la = xp4[i * 2u];
        const float4 lb = xp4[i * 2u + 1u];
        float4 ha = h4[i * 2u];
        float4 hb = h4[i * 2u + 1u];
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
                    const uint4 rv = ldg_u32x4_stream(as_gptr<const uint4>(ar_local_rx_base[r] + slot_off) + i);
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
            ha.x += sa.x; ha.y += sa.y; ha.z += sa.z; ha.w += sa.w;
            hb.x += sb.x; hb.y += sb.y; hb.z += sb.z; hb.w += sb.w;
        h4[i * 2u]      = ha;
        h4[i * 2u + 1u] = hb;
        ss = fmaf(ha.x, ha.x, fmaf(ha.y, ha.y, fmaf(ha.z, ha.z, fmaf(ha.w, ha.w, ss))));
        ss = fmaf(hb.x, hb.x, fmaf(hb.y, hb.y, fmaf(hb.z, hb.z, fmaf(hb.w, hb.w, ss))));
    }
    if (!do_norm) return;

    ss = wave64_sum(ss);
    if (lane == 0u) sm_red4[wave] = ss;
    __syncthreads();
    const float inv_rms = rsqrtf((sm_red4[0] + sm_red4[1] + sm_red4[2] + sm_red4[3]) / static_cast<float>(D) + eps);

    GPtr<const float4> w4 = reinterpret_cast<GPtr<const float4>>(w_rms);
    #pragma nounroll
    for (uint32_t it = 0u; it < 2u; ++it) {
        #pragma clang fp contract(off)
        const uint32_t i  = tid + 256u * it;
        const uint32_t ic = min(i, n8 - 1u);
        const float4 ha = h4[ic * 2u],      hb = h4[ic * 2u + 1u];
        const float4 wa = w4[ic * 2u],      wb = w4[ic * 2u + 1u];
        const float4 ya = make_float4((ha.x * inv_rms) * wa.x, (ha.y * inv_rms) * wa.y, (ha.z * inv_rms) * wa.z, (ha.w * inv_rms) * wa.w);
        const float4 yb = make_float4((hb.x * inv_rms) * wb.x, (hb.y * inv_rms) * wb.y, (hb.z * inv_rms) * wb.z, (hb.w * inv_rms) * wb.w);
        auto shfl4 = [](float v) -> float {
            v += __shfl_xor(v, 2, 4);
            v += __shfl_xor(v, 1, 4);
            return v;
        };
        float amax = fmaxf(fmaxf(fmaxf(fabsf(ya.x), fabsf(ya.y)), fmaxf(fabsf(ya.z), fabsf(ya.w))),
                           fmaxf(fmaxf(fabsf(yb.x), fabsf(yb.y)), fmaxf(fabsf(yb.z), fabsf(yb.w))));
        amax = fmaxf(amax, __shfl_xor(amax, 2, 4));
        amax = fmaxf(amax, __shfl_xor(amax, 1, 4));
        const float bsum = (((shfl4(ya.x) + shfl4(yb.x)) + (shfl4(ya.z) + shfl4(yb.z)))
                          + ((shfl4(ya.y) + shfl4(yb.y)) + (shfl4(ya.w) + shfl4(yb.w))));
        const float d     = amax * (1.0f / 127.0f);
        const float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        if (i < n8) {
            if (x_norm_out) {
                reinterpret_cast<GPtr<float4>>(x_norm_out)[i * 2u]      = ya;
                reinterpret_cast<GPtr<float4>>(x_norm_out)[i * 2u + 1u] = yb;
            }
            reinterpret_cast<GPtr<uint2>>(x_q8_out)[i] = make_uint2(quant_pack4(ya, inv_d), quant_pack4(yb, inv_d));
            if ((tid & 3u) == 0u) x_ds_out[i >> 2] = make_float2(d, bsum);
        }
    }
}

// Shared MoE Q3 Stage 1..4 body (sub32-specialized for TP=3, fresh_tid/fresh_wg to defeat MachineLICM hoisting!).
template <typename GBarFn>
__device__ __forceinline__ void moe_q3_body(
    GPtr<const vegajet::ornith_super::MoeSuperDesc> md,
    uint32_t wg_in,
    uint32_t out_ep, uint32_t rank, uint32_t n_ranks, uint32_t D,
    uint4* sm_xq, float* sm_xd, float* sm_y, int32_t* sm_ids, float* sm_w,
    GBarFn&& gbar, uint32_t* sm_st) {
    // Stage 1: Balanced 240-WG ShGU (96 WGs) + Router (128 WGs x 2 rows) + ShGate (1 WG)
    {
        const uint32_t t1  = fresh_tid();
        const uint32_t wg  = wg_in;
        const uint32_t grp = t1 >> 5;
        const uint32_t l32 = t1 & 31u;
        if (t1 < 192u) sm_xq[t1] = as_gptr<const uint4>(md->x_q8)[t1];
        if (t1 <  96u) {
            const float2 xds = as_gptr<const float2>(md->x_ds)[t1];
            reinterpret_cast<float2*>(sm_xq + 192u)[t1] = xds;
            sm_xd[t1] = xds.x;
        }
        __syncthreads();

        GPtr<const float4> x4f = as_gptr<const float4>(md->x_norm);
        if (wg < 96u) {
            const uint32_t row = wg * 8u + grp;
            if (row < md->sh_gu_rows) {
                const __amdgpu_buffer_rsrc_t srd_w  = make_srd(md->sh_gu_qs);
                const __amdgpu_buffer_rsrc_t srd_dm = make_srd(md->sh_gu_dm);
                const uint32_t w_voff  = row * md->sh_gu_stride + l32 * 16u;
                const uint32_t dm_voff = row * 384u + l32 * 4u;
                const float2* xd = reinterpret_cast<const float2*>(sm_xq + 192u);
                float acc = 0.0f;
                uint4    wv_cur  = mubuf_x4(srd_w,  w_voff,  0u);
                uint32_t dmv_cur = mubuf_u32(srd_dm, dm_voff, 0u);
                #pragma nounroll
                for (uint32_t r = 0u; r < 3u; ++r) {
                    uint4    wv_nxt  = (r < 2u) ? mubuf_x4(srd_w,  w_voff  + (r + 1u) * 512u, 0u) : wv_cur;
                    uint32_t dmv_nxt = (r < 2u) ? mubuf_u32(srd_dm, dm_voff + (r + 1u) * 128u, 0u) : dmv_cur;
                    const uint32_t bq = l32 + r * 32u;
                    const uint4 xa = sm_xq[bq * 2u], xb = sm_xq[bq * 2u + 1u];
                    const float2 xds = xd[bq];
                    const int32_t dp = dot32_q4(wv_cur, xa, xb);
                    const float wd = fp16_to_f32(static_cast<uint16_t>(dmv_cur & 0xFFFFu));
                    const float wm = fp16_to_f32(static_cast<uint16_t>(dmv_cur >> 16));
                    acc = fmaf(wd * xds.x, static_cast<float>(dp), fmaf(wm, xds.y, acc));
                    wv_cur = wv_nxt; dmv_cur = dmv_nxt;
                }
                acc = block32_sum(acc);
                if (l32 == 0u) as_gptr<float>(md->sh_gu_out)[row] = acc;
            }
        } else if (wg < 224u) {
            const uint32_t half_id = t1 >> 7;
            const uint32_t l128    = t1 & 127u;
            const uint32_t e       = (wg - 96u) * 2u + half_id;
            const __amdgpu_buffer_rsrc_t srd_r = make_srd(md->w_router_f16);
            const uint32_t r_voff = e * 6400u + l128 * 16u;
            uint4 wv_cur = mubuf_x4(srd_r, r_voff, 0u);
            float acc = 0.0f;
            #pragma nounroll
            for (uint32_t r = 0u; r < 3u; ++r) {
                uint4 wv_nxt = (r < 2u) ? mubuf_x4(srd_r, r_voff + (r + 1u) * 2048u, 0u) : wv_cur;
                const uint32_t i = l128 + r * 128u;
                const float4 xa = x4f[i * 2u], xb = x4f[i * 2u + 1u];
                const float2 h0 = __half22float2(*reinterpret_cast<const __half2*>(&wv_cur.x));
                const float2 h1 = __half22float2(*reinterpret_cast<const __half2*>(&wv_cur.y));
                const float2 h2 = __half22float2(*reinterpret_cast<const __half2*>(&wv_cur.z));
                const float2 h3 = __half22float2(*reinterpret_cast<const __half2*>(&wv_cur.w));
                acc = fmaf(h0.x, xa.x, fmaf(h0.y, xa.y, fmaf(h1.x, xa.z, fmaf(h1.y, xa.w, acc))));
                acc = fmaf(h2.x, xb.x, fmaf(h2.y, xb.y, fmaf(h3.x, xb.z, fmaf(h3.y, xb.w, acc))));
                wv_cur = wv_nxt;
            }
            acc = wave64_sum(acc);
            if ((t1 & 63u) == 0u) sm_y[t1 >> 6] = acc;
            __syncthreads();
            if (l128 == 0u) as_gptr<float>(md->router_logits)[e] = sm_y[half_id * 2u] + sm_y[half_id * 2u + 1u];
        } else if (wg == 224u) {
            GPtr<const float4> w4 = as_gptr<const float4>(md->w_sh_gate);
            float acc = 0.0f;
            #pragma nounroll
            for (uint32_t i = t1; i < 768u; i += 256u) {
                const float4 wv = w4[i], xv = x4f[i];
                acc = fmaf(wv.x, xv.x, fmaf(wv.y, xv.y, fmaf(wv.z, xv.z, fmaf(wv.w, xv.w, acc))));
            }
            acc = wave64_sum(acc);
            if ((t1 & 63u) == 0u) sm_y[t1 >> 6] = acc;
            __syncthreads();
            if (t1 == 0u) as_gptr<float>(md->sh_gate_sc)[0] = sigmoid_exact((sm_y[0] + sm_y[1]) + (sm_y[2] + sm_y[3]));
        }
        gbar(11u);
    }

    // Restore default-super local Top8: no global broadcast barrier.
    {
        const uint32_t t2 = fresh_tid();
        if (t2 < 64u) fast_softmax_top8_wave64(as_gptr<const float>(md->router_logits), sm_ids, sm_w, t2);
        __syncthreads();
        if (wg_in == 0u && t2 < 8u) {
            as_gptr<int32_t>(md->top_ids)[t2] = sm_ids[t2];
            as_gptr<float>(md->top_w)[t2] = sm_w[t2];
        }
        stage_event(sm_st, 12u);
    }

    // Stage 3: Pure VJ_Q3_IQ2S Gate+Up (WGs 0..n_tiles-1) + Idle-WG Shared Down (WGs 192..239)
    {
        const uint32_t t3       = fresh_tid();
        const uint32_t wg       = wg_in;
        const uint32_t local_ff = md->local_ff;
        const uint32_t nblk16   = local_ff >> 4;
        const uint32_t n_tiles  = 8u * nblk16;
        if (wg < n_tiles) {
            const uint32_t sub_id = t3 >> 4;
            const uint32_t lane16 = t3 & 15u;
            uint32_t slot, t_in_slot;
            divmod_slot_u(wg, nblk16, slot, t_in_slot);
            const uint32_t rloc   = t_in_slot * 16u + sub_id;
            const uint32_t exp_id = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(sm_ids[slot]));
            const float2 gu = vegajet::dot8_dev::vj_q3_gu_row_iq2s<6, 3>(
                as_gptr<const uint8_t>(md->w_gu_d8_w), md->gu_d8_w_stride,
                as_gptr<const uint8_t>(md->w_gu_d8_s), md->gu_d8_s_stride,
                sm_xq, sm_xd, exp_id * local_ff, rloc, lane16);
            if (lane16 == 0u) sm_y[sub_id] = silu_exact(gu.x) * gu.y;
            __syncthreads();
            GPtr<uint32_t> pair_cnt = as_gptr<uint32_t>(md->super_bar + 1536u);
            GPtr<float> mf32 = as_gptr<float>(md->mid_f32);
            const uint32_t blk    = wg >> 1;
            const uint32_t is_odd = wg & 1u;
            GPtr<float> dst = mf32 + wg * 16u;
            if (t3 < 16u) dst[t3] = sm_y[t3];
            asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
            __syncthreads();
            if (t3 == 0u) sm_y[16] = __uint_as_float(__atomic_fetch_xor(pair_cnt + blk, 1u, __ATOMIC_RELAXED));
            __syncthreads();
            if (__float_as_uint(sm_y[16]) == 1u && t3 < 32u) {
                GPtr<const float> other = mf32 + (wg ^ 1u) * 16u;
                const float y = (t3 < 16u)
                    ? (is_odd ? ldg_f32_stream(other + t3) : sm_y[t3])
                    : (is_odd ? sm_y[t3 - 16u] : ldg_f32_stream(other + t3 - 16u));
                const float amax = block32_max(fabsf(y));
                const float d    = amax * (1.0f / 127.0f);
                const float inv  = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                as_gptr<int8_t>(md->mid_q8)[blk * 32u + t3] =
                    static_cast<int8_t>(max(-127, min(127, __float2int_rn(y * inv))));
                if (t3 == 0u) as_gptr<float>(md->mid_d)[blk] = d;
            }
        } else if (wg >= 192u) {
            const uint32_t sh_tile = wg - 192u;
            const uint32_t n_blk   = local_ff >> 5;
            const float sc_g = as_gptr<const float>(md->sh_gate_sc)[0];
            GPtr<const float> g_p = as_gptr<const float>(md->sh_gu_out);
            GPtr<const float> u_p = g_p + local_ff;
            int8_t* sh_q8_sm = reinterpret_cast<int8_t*>(sm_xq);
            float2* sh_ds_sm = reinterpret_cast<float2*>(sm_xq + 32u);
            const uint32_t l32 = t3 & 31u;
            #pragma nounroll
            for (uint32_t blk = t3 >> 5; blk < n_blk; blk += 8u) {
                const uint32_t c = (blk << 5) + l32;
                const float gv = g_p[c], uv = u_p[c];
                const float y  = silu_exact(gv) * uv * sc_g;
                const float amax  = block32_max(fabsf(y));
                const float bsum  = block32_sum(y);
                const float d     = amax * (1.0f / 127.0f);
                const float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                sh_q8_sm[c] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(y * inv_d))));
                if (l32 == 0u) sh_ds_sm[blk] = make_float2(d, bsum);
            }
            __syncthreads();
            const uint32_t lane16 = t3 & 15u;
            const uint32_t grp16  = t3 >> 4;
            const uint32_t base   = sh_tile * 64u;
            const bool     act16  = (lane16 < n_blk);
            const uint4    xa     = act16 ? sm_xq[lane16 * 2u]      : make_uint4(0u, 0u, 0u, 0u);
            const uint4    xb     = act16 ? sm_xq[lane16 * 2u + 1u] : make_uint4(0u, 0u, 0u, 0u);
            const float2   xds    = act16 ? sh_ds_sm[lane16]        : make_float2(0.0f, 0.0f);
            #pragma nounroll
            for (uint32_t st = 0u; st < 4u; ++st) {
                const uint32_t row = base + st * 16u + grp16;
                float part_acc = 0.0f;
                if (act16) {
                    GPtr<const uint4> w4 = as_gptr<const uint4>(md->sh_dn_qs + static_cast<uint64_t>(row) * md->sh_dn_stride);
                    GPtr<const uint32_t> dm = as_gptr<const uint32_t>(md->sh_dn_dm) + static_cast<uint64_t>(row) * n_blk;
                    const uint4 wv = ldg_u32x4_stream(w4 + lane16);
                    const uint32_t dmv = ldg_u32_stream(dm + lane16);
                    const int32_t dp = dot32_q4(wv, xa, xb);
                    const float wd = fp16_to_f32(static_cast<uint16_t>(dmv & 0xFFFFu));
                    const float wm = fp16_to_f32(static_cast<uint16_t>(dmv >> 16));
                    part_acc = fmaf(wd * xds.x, static_cast<float>(dp), wm * xds.y);
                }
                const float acc = vegajet::dot8_dev::sum16(part_acc);
                if (lane16 == 0u) as_gptr<float>(md->sh_down_out)[row] = acc;
            }
        }
        gbar(15u);
    }

    // Stage 4: Balanced 240-WG sub32 Routed DOT8 Down + Shared Down + Dual-BAR0 Prepush
    {
        const uint32_t t4      = fresh_tid();
        const uint32_t wg      = wg_in;
        const uint32_t n_blk   = md->local_ff >> 5;
        const uint32_t n_q8_u4 = 8u * (n_blk << 1);
        const uint32_t n_d     = 8u * n_blk;
        if (t4 < n_q8_u4) sm_xq[t4] = as_gptr<const uint4>(md->mid_q8)[t4];
        if (t4 < n_d)     sm_xd[t4] = as_gptr<const float>(md->mid_d)[t4];
        __syncthreads();
        const uint32_t grp16  = t4 >> 4;
        const uint32_t lane16 = t4 & 15u;
        const uint32_t n_rows = (wg < 144u) ? 16u : 8u;
        const uint32_t row0   = (wg < 144u) ? (wg * 16u) : (2304u + (wg - 144u) * 8u);
        if (grp16 < n_rows) {
            const uint32_t row = row0 + grp16;
            const float total = vegajet::dot8_dev::vj_dot8_down_row_iq3_sub32<2>(
                as_gptr<const uint8_t>(md->w_dn_d8_w), md->dn_w_stride,
                as_gptr<const uint8_t>(md->w_dn_d8_s), md->dn_s_stride,
                sm_xq, sm_xd, sm_ids, sm_w, row, D, n_blk, lane16);
            if (lane16 == 0u) {
                const float v = total + as_gptr<const float>(md->sh_down_out)[row];
                as_gptr<float>(md->x_partial)[row] = v;
                sm_y[grp16] = v;
            }
        }
        if (n_ranks > 1u) {
            __syncthreads();
            const uint64_t slot_off = static_cast<uint64_t>(out_ep & 1u) * VJ_MEGA_AR_SLOT_BYTES;
            if (t4 < (n_rows >> 3)) {
                const uint32_t o = t4 * 8u;
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
                for (uint32_t dst = 0u; dst < n_ranks; ++dst) {
                    if (dst != rank) {
                        __builtin_nontemporal_store(pv, reinterpret_cast<GPtr<u32x4>>(
                            as_gptr<uint16_t>(md->out_peer_tx_base[dst] + slot_off) + row0 + o));
                    }
                }
            }
        }
        signal_out_peer_flag_mega(scratch_bar(sm_st), md->out_peer_flag, rank, n_ranks, out_ep);
        stage_event(sm_st, 14u);
    }
}

} // namespace mega_dev
} // namespace vegajet

// ============================================================================
// KERNEL 1: Persistent N-Layer (GDN + MoE Q3) Loop (3 consecutive layers -> 12 dispatches!)
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_m1_gdn_moe_loop_q3_kernel(
    vegajet::mega_dev::GPtr<const vegajet::ornith_super::Mega48Desc> __restrict__ mega_in,
    uint64_t spec_gdn_base,
    uint32_t il0,
    uint32_t gi0,
    uint32_t n_layers,
    uint32_t ar_epoch0,
    uint32_t conv_in_slot,
    uint32_t conv_out_slot,
    uint32_t ssm_ring_head,
    uint32_t use_ring,
    uint32_t super_sense0,
    uint32_t gdn_sense0,
    uint32_t flags) {
    // flags: bit 0 = skip_first_ar (layer 0 already embedded+normed), bit 2 = profile
    using namespace vegajet::mega_dev;
    const uint32_t profile_on = flag_opaque(flags, 2u);
    const uint32_t skip_first_ar = flag_opaque(flags, 0u);
    const uint32_t ring_on = flag_opaque(use_ring, 0u);
    __shared__ uint4    sm_xq[256];
    __shared__ float2   sm_xds[128];
    __shared__ float    sm_aux[256];
    __shared__ float    sm_fo[64];
    __shared__ uint32_t sm_st[13]; // [0]=st, [1]=cur_ep, [2]=sense, [3]=bar_idx, [4]=gdn_sense, [5..6]=mega_in, [7]=wg

    if (threadIdx.x == 0u) {
        const uint64_t mu = reinterpret_cast<uint64_t>(mega_in);
        sm_st[0] = 0u;
        sm_st[1] = ar_epoch0;
        sm_st[2] = super_sense0;
        sm_st[3] = 0u;
        sm_st[4] = gdn_sense0;
        sm_st[5] = static_cast<uint32_t>(mu);
        sm_st[6] = static_cast<uint32_t>(mu >> 32);
        sm_st[7] = blockIdx.x;
        sm_st[8] = profile_on;
        const uint64_t sb = mega_in->super_bar;
        sm_st[11] = static_cast<uint32_t>(sb); sm_st[12] = static_cast<uint32_t>(sb >> 32);

    }
    __syncthreads();

    auto load_mega = [&]() -> GPtr<const vegajet::ornith_super::Mega48Desc> {
        const uint64_t mu = static_cast<uint64_t>(rfl(sm_st[5])) | (static_cast<uint64_t>(rfl(sm_st[6])) << 32);
        return reinterpret_cast<GPtr<const vegajet::ornith_super::Mega48Desc>>(mu);
    };
    auto gbar = [&](uint32_t slot = 31u) {
        stage_event(sm_st, slot | 0x80000000u);
    };

    #pragma nounroll
    for (;;) {
        const uint32_t st = rfl(sm_st[0]);
        if (st >= (n_layers * 2u)) break;
        const auto* mega = load_mega();
        const uint32_t wg_cur = rfl(sm_st[7]);
        const uint32_t k      = st >> 1;
        const bool     is_moe = (st & 1u) != 0u;
        const uint32_t il     = il0 + k;
        const uint32_t gi     = gi0 + k;

        stage_event(sm_st, ((il * 2u + (st & 1u)) << 5u) | 16u);
        if (!((skip_first_ar != 0u) && st == 0u)) {
            const uint32_t prev_ep = rfl(sm_st[1]) - 1u;
            if (wg_cur == 0u) {
                const auto* gdn_arr = reinterpret_cast<GPtr<const vegajet::ornith_super::GdnSuperDesc>>(mega->gdn_descs);
                const auto* moe_arr = reinterpret_cast<GPtr<const vegajet::ornith_super::MoeSuperDesc>>(mega->moe_descs);
                const uint64_t slot_off = static_cast<uint64_t>(prev_ep & 1u) * VJ_MEGA_AR_SLOT_BYTES;
                const uint64_t* rx_arr  = is_moe ? moe_arr[il].ar_local_rx_base : gdn_arr[gi].ar_local_rx_base;
                const uint64_t* fl_arr  = is_moe ? moe_arr[il].ar_local_flag    : gdn_arr[gi].ar_local_flag;
                const uint64_t w_rms_va = is_moe ? moe_arr[il].w_post_attn_norm : gdn_arr[gi].w_attn_norm;
                const uint64_t xn_va    = is_moe ? moe_arr[il].x_norm           : 0ULL;
                p2p_ar_prepush_mega(
                    as_gptr<uint32_t>(mega->super_bar), as_gptr<const float>(mega->x_partial), rx_arr, fl_arr, slot_off,
                    as_gptr<float>(mega->h_io), as_gptr<const float>(w_rms_va), as_gptr<float>(xn_va),
                    as_gptr<int8_t>(mega->x_q8), as_gptr<float2>(mega->x_ds), sm_fo,
                    3072u, prev_ep, mega->rank, 3u, true, mega->eps);
            }
            gbar();
        }

        stage_event(sm_st, is_moe ? 10u : 0u);
        const uint32_t out_ep = rfl(sm_st[1]);
        __syncthreads();
        if (threadIdx.x == 0u) { sm_st[1] = out_ep + 1u; sm_st[0] = st + 1u; }
        __syncthreads();

        if (!is_moe) {
            const auto* gd = reinterpret_cast<GPtr<const vegajet::ornith_super::GdnSuperDesc>>(mega->gdn_descs) + gi;
            #pragma nounroll
            for (uint32_t gp = 0u; gp < 2u; ++gp) {
                const bool     do_pp    = (gp == 1u) && (3u > 1u);
                const uint64_t slot_off = static_cast<uint64_t>(out_ep & 1u) * VJ_MEGA_AR_SLOT_BYTES;
                gemv_q4a8_mega(
                    gp ? gd->w_out_qs : gd->w_qkv_qs,
                    gp ? gd->w_out_dm : gd->w_qkv_dm,
                    as_gptr<const int8_t>(gp ? gd->mid_q8 : gd->x_q8),
                    as_gptr<const float2>(gp ? gd->mid_ds : gd->x_ds),
                    as_gptr<float>(gp ? gd->x_partial : gd->gdn_qkv_gab),
                    gp ? gd->out_peer_tx_base : nullptr, slot_off,
                    mega->rank, 3u, do_pp,
                    sm_xq, sm_xds, sm_fo,
                    gp ? gd->out_rows : gd->qkv_rows,
                    gp ? gd->out_cols : gd->qkv_cols,
                    gp ? gd->out_qs_stride : gd->qkv_qs_stride);
                if (gp == 1u) {
                    signal_out_peer_flag_mega(as_gptr<uint32_t>(mega->super_bar), gd->out_peer_flag, mega->rank, 3u, out_ep);
                    stage_event(sm_st, 3u);
                    break;
                }
                gbar(1u);

                const uint32_t gs = rfl(sm_st[4]) + 1u;
                __syncthreads();
                if (threadIdx.x == 0u) sm_st[4] = gs;
                __syncthreads();

                const uint32_t wg_g  = rfl(sm_st[7]);
                const uint32_t tid_g = fresh_tid();
                if (wg_g < gd->nv * 4u) {
                    float* sm_q   = reinterpret_cast<float*>(sm_xds);
                    float* sm_k   = sm_q + 128u;
                    float* sm_red = sm_aux;
                    float* sm_dv  = sm_fo;
                    float* sm_fg  = sm_fo + 32u;
                    const uint32_t nk       = gd->nk;
                    const uint32_t nv       = gd->nv;
                    const uint32_t conv_dim = gd->conv_dim;
                    const uint32_t hv       = wg_g >> 2;
                    const uint32_t part     = wg_g & 3u;
                    const uint32_t col0     = part * 32u;
                    const uint32_t hk       = hv % nk;
                    const float    eps      = mega->eps;
                    GPtr<const float> qkv_in = as_gptr<const float>(gd->gdn_qkv_gab);
                    GPtr<const float> conv_w = as_gptr<const float>(gd->conv_w);
                    GPtr<float> conv_rg = as_gptr<float>(gd->conv_ring);
                    {
                        const uint64_t cslot_stride = 3ULL * conv_dim;
                        GPtr<const float> cr_in  = conv_rg + static_cast<uint64_t>(conv_in_slot)  * cslot_stride;
                        GPtr<float>       cr_out = conv_rg + static_cast<uint64_t>(conv_out_slot) * cslot_stride;
                        const uint32_t is_k = tid_g >> 7;
                        const uint32_t l    = tid_g & 127u;
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
                        float       y    = silu_exact(acc);
                        const float sq_w = wave64_sum(y * y);
                        if ((tid_g & 63u) == 0u) sm_red[tid_g >> 6] = sq_w;
                        __syncthreads();
                        float inv_n = rsqrtf(sm_red[is_k ? 2u : 0u] + sm_red[is_k ? 3u : 1u] + eps);
                        if (!is_k) inv_n *= 0.08838834764831845f;
                        y *= inv_n;
                        if (!is_k) sm_q[l] = y; else sm_k[l] = y;

                        if (tid_g < 32u) {
                            const uint32_t ch_v = (2u * nk + hv) * 128u + col0 + tid_g;
                            const float4   wv   = reinterpret_cast<GPtr<const float4>>(conv_w)[ch_v];
                            const float    sv0  = cr_in[0ULL * conv_dim + ch_v];
                            const float    sv1  = cr_in[1ULL * conv_dim + ch_v];
                            const float    sv2  = cr_in[2ULL * conv_dim + ch_v];
                            const float    xv   = qkv_in[ch_v];
                            const float    av   = fmaf(wv.x, sv0, fmaf(wv.y, sv1, fmaf(wv.z, sv2, wv.w * xv)));
                            sm_dv[tid_g] = silu_exact(av);
                            cr_out[0ULL * conv_dim + ch_v] = sv1;
                            cr_out[1ULL * conv_dim + ch_v] = sv2;
                            cr_out[2ULL * conv_dim + ch_v] = xv;
                        }
                        if (tid_g == 0u) {
                            const float a  = qkv_in[gd->alpha_off + hv] + as_gptr<const float>(gd->dt_bias)[hv];
                            const float sp = (a > 20.0f) ? a : log1pf(__expf(a));
                            sm_fg[0] = __expf(sp * as_gptr<const float>(gd->ssm_a)[hv]);
                            sm_fg[1] = sigmoid_exact(qkv_in[gd->beta_off + hv]);
                        }
                    }
                    __syncthreads();

                    const uint64_t head_elems = 128ULL * 128ULL;
                    const uint64_t slot_elems = static_cast<uint64_t>(nv) * head_elems;
                    const uint64_t ssm_base_va = (spec_gdn_base != 0ULL)
                        ? (spec_gdn_base + static_cast<uint64_t>(gi) * mega->ssm_layer_bytes)
                        : gd->ssm_ring_base;
                    GPtr<float> ssm_ring = as_gptr<float>(ssm_base_va);
                    GPtr<float> S_base = ssm_ring + static_cast<uint64_t>(ssm_ring_head) * slot_elems + static_cast<uint64_t>(hv) * head_elems;
                    const uint32_t rg4  = tid_g >> 3;
                    const uint32_t c4   = tid_g & 7u;
                    const uint32_t r0   = rg4 * 4u;
                    const uint32_t col4 = (col0 >> 2) + c4;
                    GPtr<const float4> S_in4 = reinterpret_cast<GPtr<const float4>>(S_base) + static_cast<uint64_t>(r0) * 32u + col4;
                    auto ldg_f4 = [](GPtr<const float4> p) -> float4 {
                        f32x4 v = __builtin_nontemporal_load(reinterpret_cast<GPtr<const f32x4>>(p));
                        return make_float4(v[0], v[1], v[2], v[3]);
                    };
                    auto stg_f4 = [](GPtr<float4> p, float4 v) {
                        f32x4 fv = {v.x, v.y, v.z, v.w};
                        __builtin_nontemporal_store(fv, reinterpret_cast<GPtr<f32x4>>(p));
                    };
                    float4 s0 = ldg_f4(S_in4 + 0u * 32u);
                    float4 s1 = ldg_f4(S_in4 + 1u * 32u);
                    float4 s2 = ldg_f4(S_in4 + 2u * 32u);
                    float4 s3 = ldg_f4(S_in4 + 3u * 32u);
                    __builtin_amdgcn_sched_barrier(0);
                    const float g_exp = sm_fg[0];
                    const float beta  = sm_fg[1];
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
                    if ((tid_g & 56u) == 0u) reinterpret_cast<float4*>(sm_red)[(tid_g >> 6) * 8u + c4] = pv;
                    __syncthreads();
                    if (tid_g < 32u) {
                        const float pred = (sm_red[tid_g] + sm_red[32u + tid_g]) + (sm_red[64u + tid_g] + sm_red[96u + tid_g]);
                        sm_dv[tid_g] = beta * (sm_dv[tid_g] - g_exp * pred);
                    }
                    __syncthreads();
                    const float4 dv4 = reinterpret_cast<const float4*>(sm_dv)[c4];
                    GPtr<float4> S_out4 = (ring_on != 0u)
                        ? reinterpret_cast<GPtr<float4>>(ssm_ring + static_cast<uint64_t>((ssm_ring_head + 1u) % VJ_MEGA_RING_SLOTS) * slot_elems + static_cast<uint64_t>(hv) * head_elems) + static_cast<uint64_t>(r0) * 32u + col4
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
                    if ((tid_g & 56u) == 0u) reinterpret_cast<float4*>(sm_red)[(tid_g >> 6) * 8u + c4] = ov;
                    __syncthreads();
                    if (tid_g < 32u) sm_fg[tid_g] = (sm_red[tid_g] + sm_red[32u + tid_g]) + (sm_red[64u + tid_g] + sm_red[96u + tid_g]);
                    __syncthreads();

                    GPtr<float> sq = as_gptr<float>(gd->gdn_sq) + static_cast<uint64_t>(hv) * 64u;
                    const uint32_t l16 = tid_g & 15u;
                    const float fo0 = sm_fg[l16 * 2u], fo1 = sm_fg[l16 * 2u + 1u];
                    if (tid_g < 16u) stg_f32_stream(sq + part * 16u + tid_g, fo0 * fo0 + fo1 * fo1);
                    grid_barrier_glc_only(as_gptr<uint32_t>(gd->gdn_bar) + static_cast<uint64_t>(hv) * 32u, 4u, gs);
                    if (tid_g < 64u) {
                        const float total_sq = wave64_sum(ldg_f32_stream(sq + tid_g));
                        if (tid_g < 16u) {
                            const float inv_r = rsqrtf(total_sq * (1.0f / 128.0f) + eps);
                            const uint32_t j0 = col0 + tid_g * 2u;
                            GPtr<const float> zp = qkv_in + gd->z_off + hv * 128u;
                            GPtr<const float> nw = as_gptr<const float>(gd->ssm_norm_w);
                            const float z0 = zp[j0], z1 = zp[j0 + 1u];
                            const float y0 = (fo0 * inv_r * nw[j0])      * silu_exact(z0);
                            const float y1 = (fo1 * inv_r * nw[j0 + 1u]) * silu_exact(z1);
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
                            GPtr<int8_t> oq = as_gptr<int8_t>(gd->mid_q8) + static_cast<uint64_t>(hv) * 128u;
                            oq[j0]      = static_cast<int8_t>(max(-127, min(127, __float2int_rn(y0 * inv_d))));
                            oq[j0 + 1u] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(y1 * inv_d))));
                            if (tid_g == 0u) as_gptr<float2>(gd->mid_ds)[static_cast<uint64_t>(hv) * 4u + part] = make_float2(d, bsum);
                        }
                    }
                }
                gbar(2u);
            }
        } else {
            const auto* md = reinterpret_cast<GPtr<const vegajet::ornith_super::MoeSuperDesc>>(mega->moe_descs) + il;
            moe_q3_body(md, rfl(sm_st[7]), out_ep, mega->rank, 3u, 3072u,
                        sm_xq, reinterpret_cast<float*>(sm_xds), sm_aux,
                        reinterpret_cast<int32_t*>(sm_fo), sm_fo + 8u, gbar, sm_st);
        }
    }
}

// ============================================================================
// KERNEL 2: Persistent (GQA + MoE Q3) Layer (+ optional Final AR + out_norm)
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_m1_gqa_moe_q3_kernel(
    vegajet::mega_dev::GPtr<const vegajet::ornith_super::Mega48Desc> __restrict__ mega_in,
    uint32_t il,
    uint32_t ai,
    uint32_t ar_epoch0,
    uint32_t pos0,
    uint32_t super_sense0,
    uint32_t flags) {
    // flags: bit 0 = do_final_ar, bit 1 = do_final_norm, bit 2 = profile
    using namespace vegajet::mega_dev;
    const uint32_t profile_on = flag_opaque(flags, 2u);
    const uint32_t final_ar = flag_opaque(flags, 0u);
    const uint32_t final_norm = flag_opaque(flags, 1u);
    __shared__ uint4    sm_xq[256];
    __shared__ float2   sm_xds[128];
    __shared__ float    sm_aux[256];
    __shared__ float    sm_fo[64];
    __shared__ uint32_t sm_st[13]; // [0]=st, [1]=cur_ep, [2]=sense, [3]=bar_idx, [4..5]=mega_in, [6]=wg

    if (threadIdx.x == 0u) {
        const uint64_t mu = reinterpret_cast<uint64_t>(mega_in);
        sm_st[0] = 0u;
        sm_st[1] = ar_epoch0;
        sm_st[2] = super_sense0;
        sm_st[3] = 0u;
        sm_st[4] = static_cast<uint32_t>(mu);
        sm_st[5] = static_cast<uint32_t>(mu >> 32);
        sm_st[6] = blockIdx.x;
        sm_st[8] = profile_on;
        const uint64_t sb = mega_in->super_bar;
        sm_st[11] = static_cast<uint32_t>(sb); sm_st[12] = static_cast<uint32_t>(sb >> 32);

    }
    __syncthreads();

    auto load_mega = [&]() -> GPtr<const vegajet::ornith_super::Mega48Desc> {
        const uint64_t mu = static_cast<uint64_t>(rfl(sm_st[4])) | (static_cast<uint64_t>(rfl(sm_st[5])) << 32);
        return reinterpret_cast<GPtr<const vegajet::ornith_super::Mega48Desc>>(mu);
    };
    auto gbar = [&](uint32_t slot = 31u) {
        stage_event(sm_st, slot | 0x80000000u);
    };

    #pragma nounroll
    for (;;) {
        const uint32_t st = rfl(sm_st[0]);
        if (st >= ((final_ar != 0u) ? 3u : 2u)) break;
        const auto* mega = load_mega();
        stage_event(sm_st, ((st == 2u ? 96u : il * 2u + st) << 5u) | 16u);

        {
            const uint32_t prev_ep = rfl(sm_st[1]) - 1u;
            if (rfl(sm_st[6]) == 0u) {
                const auto* qd = reinterpret_cast<GPtr<const vegajet::ornith_super::GqaSuperDesc>>(mega->gqa_descs) + ai;
                const auto* md = reinterpret_cast<GPtr<const vegajet::ornith_super::MoeSuperDesc>>(mega->moe_descs) + il;
                const uint64_t slot_off = static_cast<uint64_t>(prev_ep & 1u) * VJ_MEGA_AR_SLOT_BYTES;
                const uint64_t* rx_arr  = (st == 2u) ? mega->final_ar_rx   : (st == 1u ? md->ar_local_rx_base : qd->ar_local_rx_base);
                const uint64_t* fl_arr  = (st == 2u) ? mega->final_ar_flag : (st == 1u ? md->ar_local_flag    : qd->ar_local_flag);
                const uint64_t w_rms_va = (st == 2u) ? mega->w_out_norm    : (st == 1u ? md->w_post_attn_norm : qd->w_attn_norm);
                const uint64_t xn_va    = (st == 2u) ? mega->x_norm        : (st == 1u ? md->x_norm           : 0ULL);
                const bool     dn       = (st == 2u) ? (final_norm != 0u) : true;
                p2p_ar_prepush_mega(
                    as_gptr<uint32_t>(mega->super_bar), as_gptr<const float>(mega->x_partial), rx_arr, fl_arr, slot_off,
                    as_gptr<float>(mega->h_io), as_gptr<const float>(w_rms_va), as_gptr<float>(xn_va),
                    as_gptr<int8_t>(mega->x_q8), as_gptr<float2>(mega->x_ds), sm_fo,
                    3072u, prev_ep, mega->rank, 3u, dn, mega->eps);
            }
            gbar();
            stage_event(sm_st, st == 1u ? 10u : 4u);
            if (st == 2u) break;
        }

        const uint32_t out_ep = rfl(sm_st[1]);
        __syncthreads();
        if (threadIdx.x == 0u) { sm_st[1] = out_ep + 1u; sm_st[0] = st + 1u; }
        __syncthreads();

        if (st == 0u) {
            const auto* qd = reinterpret_cast<GPtr<const vegajet::ornith_super::GqaSuperDesc>>(mega->gqa_descs) + ai;
            #pragma nounroll
            for (uint32_t gp = 0u; gp < 2u; ++gp) {
                const bool     do_pp    = (gp == 1u) && (3u > 1u);
                const uint64_t slot_off = static_cast<uint64_t>(out_ep & 1u) * VJ_MEGA_AR_SLOT_BYTES;
                gemv_q4a8_mega(
                    gp ? qd->w_o_qs : qd->w_qkv_qs,
                    gp ? qd->w_o_dm : qd->w_qkv_dm,
                    as_gptr<const int8_t>(gp ? qd->mid_q8 : qd->x_q8),
                    as_gptr<const float2>(gp ? qd->mid_ds : qd->x_ds),
                    as_gptr<float>(gp ? qd->x_partial : qd->attn_qkv),
                    gp ? qd->out_peer_tx_base : nullptr, slot_off,
                    mega->rank, 3u, do_pp,
                    sm_xq, sm_xds, sm_fo,
                    gp ? qd->o_rows : qd->qkv_rows,
                    gp ? qd->o_cols : qd->qkv_cols,
                    gp ? qd->o_qs_stride : qd->qkv_qs_stride);
                if (gp == 1u) {
                    signal_out_peer_flag_mega(as_gptr<uint32_t>(mega->super_bar), qd->out_peer_flag, mega->rank, 3u, out_ep);
                    stage_event(sm_st, 9u);
                    break;
                }
                gbar(5u);

                const uint32_t d    = fresh_tid();
                const uint32_t wg_q = rfl(sm_st[6]);
                const uint32_t wave = d >> 6;
                const uint32_t lane = d & 63u;
                const uint32_t blk  = d >> 5;
                const uint32_t sub  = d & 31u;
                const uint32_t n_q_local  = qd->n_q_local;
                const uint32_t n_kv_local = qd->n_kv_local;
                float* sm_v   = sm_aux;
                float* sm_red = sm_fo;
                {
                    GPtr<const float> qkv_f = as_gptr<const float>(qd->attn_qkv);
                    const uint32_t qg_tot = n_q_local * 512u;
                    const uint32_t kv_tot = n_kv_local * 256u;
                    const uint32_t rdims  = qd->rope_dims;
                    const uint32_t half_r = rdims >> 1;
                    const float    rbase  = qd->rope_base;
                    const float    eps    = mega->eps;
                    const float    pos_f  = static_cast<float>(pos0);
                    if (wg_q < n_q_local + n_kv_local) {
                        const bool is_q = (wg_q < n_q_local);
                        const uint32_t kv_h = is_q ? 0u : (wg_q - n_q_local);
                        GPtr<const float> src = is_q ? (qkv_f + wg_q * 512u) : (qkv_f + qg_tot + kv_h * 256u);
                        const float raw = src[d];
                        const float sq  = wave64_sum(raw * raw);
                        if (lane == 0u) sm_red[wave] = sq;
                        __syncthreads();
                        const float inv_r = rsqrtf((sm_red[0] + sm_red[1] + sm_red[2] + sm_red[3]) * (1.0f / 256.0f) + eps);
                        sm_v[d] = raw * inv_r * as_gptr<const float>(is_q ? qd->q_norm_w : qd->k_norm_w)[d];
                        __syncthreads();
                        float rot = sm_v[d];
                        if (d < rdims) {
                            const uint32_t p = (d < half_r) ? d : (d - half_r);
                            const float theta = pos_f * __powf(rbase, -2.0f * static_cast<float>(p) / static_cast<float>(rdims));
                            float s_th, c_th;
                            __sincosf(theta, &s_th, &c_th);
                            rot = (d < half_r) ? (sm_v[p] * c_th - sm_v[p + half_r] * s_th)
                                               : (sm_v[p] * s_th + sm_v[p + half_r] * c_th);
                        }
                        if (is_q) {
                            as_gptr<float>(qd->q_buf)[static_cast<uint64_t>(wg_q) * 256u + d] = rot;
                        } else {
                            const uint64_t slot_off2 = (static_cast<uint64_t>(pos0) * n_kv_local + kv_h) * 272ULL;
                            #pragma nounroll
                            for (uint32_t is_v = 0u; is_v < 2u; ++is_v) {
                                GPtr<uint8_t> dst_c = as_gptr<uint8_t>(is_v ? qd->v_cache_q8 : qd->k_cache_q8) + slot_off2;
                                const float val  = is_v ? (qkv_f + qg_tot + kv_tot + kv_h * 256u)[d] : rot;
                                const float amax = block32_max(fabsf(val));
                                const float sc   = amax * (1.0f / 127.0f);
                                const float inv  = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                                reinterpret_cast<GPtr<int8_t>>(dst_c + 16u)[d] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(val * inv))));
                                if (sub == 0u) reinterpret_cast<GPtr<uint16_t>>(dst_c)[blk] = __half_as_ushort(__float2half(sc));
                            }
                        }
                    }
                    gbar(6u);
                }

                const uint32_t span = pos0 + 1u;
                const uint32_t chunk_len = (span <= 1920u) ? 32u : ((span <= 3840u) ? 64u : ((span <= 7680u) ? 128u : 256u));
                const uint32_t n_chunks  = (span + chunk_len - 1u) / chunk_len;
                const uint32_t active_ch = min(n_chunks, 240u / n_q_local);
                {
                    if (wg_q < n_q_local * active_ch) {
                        const uint32_t h   = wg_q / active_ch;
                        const uint32_t ch0 = wg_q - h * active_ch;
                        const uint32_t kv_h = (qd->gqa_group != 0u)
                            ? (((qd->q_head_off + h) / qd->gqa_group) - qd->kv_head_off)
                            : ((h * n_kv_local) / n_q_local);
                        const float qv = as_gptr<const float>(qd->q_buf)[static_cast<uint64_t>(h) * 256u + d];
                        GPtr<const uint8_t> k_cache = as_gptr<const uint8_t>(qd->k_cache_q8);
                        GPtr<const uint8_t> v_cache = as_gptr<const uint8_t>(qd->v_cache_q8);
                        #pragma nounroll
                        for (uint32_t ch = ch0; ch < n_chunks; ch += active_ch) {
                            const uint32_t t_beg = ch * chunk_len;
                            const uint32_t t_end = min(t_beg + chunk_len, span);
                            float m_run = -1e30f, l_run = 0.0f, o_d = 0.0f;
                            #pragma nounroll
                            for (uint32_t t = t_beg; t < t_end; ++t) {
                                const uint64_t off = (static_cast<uint64_t>(t) * n_kv_local + kv_h) * 272ULL;
                                GPtr<const uint8_t> kc = k_cache + off;
                                GPtr<const uint8_t> vc = v_cache + off;
                                const float kv = fp16_to_f32(reinterpret_cast<GPtr<const uint16_t>>(kc)[blk]) *
                                                 static_cast<float>(reinterpret_cast<GPtr<const int8_t>>(kc + 16u)[d]);
                                const float vv = fp16_to_f32(reinterpret_cast<GPtr<const uint16_t>>(vc)[blk]) *
                                                 static_cast<float>(reinterpret_cast<GPtr<const int8_t>>(vc + 16u)[d]);
                                const float dot = wave64_sum(qv * kv);
                                if (lane == 0u) sm_red[wave] = dot;
                                __syncthreads();
                                const float score = (sm_red[0] + sm_red[1] + sm_red[2] + sm_red[3]) * 0.0625f;
                                const float m_new = fmaxf(m_run, score);
                                const float alpha = __expf(m_run - m_new);
                                const float beta  = __expf(score - m_new);
                                l_run = l_run * alpha + beta;
                                m_run = m_new;
                                o_d   = fmaf(beta, vv, o_d * alpha);
                                __syncthreads();
                            }
                            const uint64_t idx = static_cast<uint64_t>(h) * n_chunks + ch;
                            as_gptr<float>(qd->part_o)[idx * 256u + d] = o_d;
                            if (d == 0u) as_gptr<float2>(qd->part_ml)[idx] = make_float2(m_run, l_run);
                        }
                    }
                    gbar(7u);
                }
                {
                    if (wg_q < n_q_local) {
                        GPtr<const float>  po  = as_gptr<const float>(qd->part_o)   + static_cast<uint64_t>(wg_q) * n_chunks * 256u;
                        GPtr<const float2> pml = as_gptr<const float2>(qd->part_ml) + static_cast<uint64_t>(wg_q) * n_chunks;
                        float o_tot, l_tot;
                        if (n_chunks == 1u) {
                            o_tot = po[d];
                            l_tot = pml[0].y;
                        } else {
                            float g_max = -1e30f;
                            #pragma nounroll
                            for (uint32_t ch = 0u; ch < n_chunks; ++ch) {
                                const float2 ml = pml[ch];
                                if (ml.y > 0.0f) g_max = fmaxf(g_max, ml.x);
                            }
                            o_tot = 0.0f; l_tot = 0.0f;
                            #pragma nounroll
                            for (uint32_t ch = 0u; ch < n_chunks; ++ch) {
                                const float2 ml = pml[ch];
                                if (ml.y > 0.0f) {
                                    const float w = __expf(ml.x - g_max);
                                    l_tot = fmaf(w, ml.y, l_tot);
                                    o_tot = fmaf(w, po[static_cast<uint64_t>(ch) * 256u + d], o_tot);
                                }
                            }
                        }
                        const float attn_v = (l_tot > 0.0f) ? (o_tot / l_tot) : 0.0f;
                        const float gv     = as_gptr<const float>(qd->attn_qkv)[wg_q * 512u + 256u + d];
                        const float y_d    = attn_v * (sigmoid_exact(gv));
                        const float amax   = block32_max(fabsf(y_d));
                        const float bsum   = block32_sum(y_d);
                        const float sc     = amax * (1.0f / 127.0f);
                        const float inv_d  = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                        as_gptr<int8_t>(qd->mid_q8)[static_cast<uint64_t>(wg_q) * 256u + d] =
                            static_cast<int8_t>(max(-127, min(127, __float2int_rn(y_d * inv_d))));
                        if (sub == 0u) as_gptr<float2>(qd->mid_ds)[static_cast<uint64_t>(wg_q) * 8u + blk] = make_float2(sc, bsum);
                    }
                    gbar(8u);
                }
            }
        } else {
            const auto* md = reinterpret_cast<GPtr<const vegajet::ornith_super::MoeSuperDesc>>(mega->moe_descs) + il;
            moe_q3_body(md, rfl(sm_st[6]), out_ep, mega->rank, 3u, 3072u,
                        sm_xq, reinterpret_cast<float*>(sm_xds), sm_aux,
                        reinterpret_cast<int32_t*>(sm_fo), sm_fo + 8u, gbar, sm_st);
        }
    }
}
