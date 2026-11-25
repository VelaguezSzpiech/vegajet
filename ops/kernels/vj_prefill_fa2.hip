// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// VegaJet Big Lever 3: Br=16 / Bc=32 head-packed FlashAttention-2 for batched GQA prefill (gfx906).
//
// Drop-in replacement for ornith_attn_flash_quant_kernel on the M > 1 path (same kernargs, same outputs:
// sigmoid(gate)-gated attention, block-32 INT8 + float2{scale, sum} per 32 dims).
//
// Packing (16:1 head packing): the 16 query rows of a workgroup are the rows i = blockIdx.x*16 .. +15 of the
// (token, q-head-in-group) space of one KV head, ordered token-major / head-minor: i = m*G + g with
// G = n_q_local / n_kv_local. For Ornith (G = 16) one workgroup is exactly ONE token and all 16 Q heads that
// share the local KV head. Why heads rather than tokens:
//   * all 16 rows share one causal limit (pos0 + m), so there is no per-row masked waste inside a tile and the
//     key loop length is exact for every row;
//   * every K/V tile pulled into LDS is reused by 16 rows, the same reuse as token packing, but the 16 Q rows
//     and gate rows are one contiguous 16 KiB span in memory;
//   * token packing for G = 16 would need 16 separate (head) passes over the same KV prefix per token block.
// For G < 16 the same mapping packs 16/G tokens per workgroup and masks per row (still correct).
//
// Per workgroup: 256 threads = 4 Wave64; wave w owns query rows 4w..4w+3; lane l owns head dims 4l..4l+3.
//   Q   : 4 rows x 4 dims FP16 (8 VGPRs)        O : 4 rows x 4 dims FP32 (16 VGPRs)
//   m,l : one row per 16-lane quarter (2 VGPRs; per-row alpha / l are broadcast with v_readlane)
//   Measured: 63 VGPRs (FP32-Q variant 61), 0 spill (spec target was 56: the K/V tile staging registers and LDS addresses cost
//   the difference; a next-K-tile prefetch held across the PV loop spilled and was dropped).
// LDS: K or V tile 32 x (256 + 8) FP16 (16,896 B; INT8 values are exact in FP16, the FP16 block scales are kept
// separately in FP32: 1,024 B) + P[32][16] FP32 (2,048 B) = 19,968 B/WG -> 3 WGs/CU by LDS, VGPR-limited to 4
// waves/SIMD = 4 WGs/CU at <= 64 VGPRs (2 WGs/CU guaranteed).
// S = Q K^T: per lane v_dot2_f32_f16 over its 4 dims, then an 8-value reduce-scatter across the wave
// (4+2+1+3 = 10 cross-lane ops per 8 scores instead of 6 per score).
// Online softmax is updated once per 32-key tile (FA2), causal mask by absolute position.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

namespace vegajet {
namespace pfa2 {

typedef _Float16 hf2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ float fp16_to_f32(uint16_t h) {
    return __half2float(__ushort_as_half(h));
}

template <int PAT>
__device__ __forceinline__ float swz(float v) {
    return __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), PAT));
}
// ds_swizzle bit-mask mode: and=0x1F, xor = PAT >> 10 (within 32-lane halves).
#define VJ_SWZ_X1  0x041F
#define VJ_SWZ_X2  0x081F
#define VJ_SWZ_X4  0x101F
#define VJ_SWZ_X8  0x201F
#define VJ_SWZ_X16 0x401F

__device__ __forceinline__ float xor_lane(float v, uint32_t mask) {
    switch (mask) {
        case 1u:  return swz<VJ_SWZ_X1>(v);
        case 2u:  return swz<VJ_SWZ_X2>(v);
        case 4u:  return swz<VJ_SWZ_X4>(v);
        case 8u:  return swz<VJ_SWZ_X8>(v);
        case 16u: return swz<VJ_SWZ_X16>(v);
        default:  return __shfl_xor(v, 32, 64);
    }
}

// 4 INT8 (one dword) -> 4 exact FP16 values (2 x half2) with the 0x6400 magic: (1024 + 128 + b) - 1152 = b.
__device__ __forceinline__ void i8x4_to_hf2x2(uint32_t w, uint32_t& lo, uint32_t& hi) {
    const uint32_t x = w ^ 0x80808080u;
    const uint32_t a = __builtin_amdgcn_perm(0x64646464u, x, 0x04010400u); // bytes {x0, 64, x1, 64}
    const uint32_t b = __builtin_amdgcn_perm(0x64646464u, x, 0x04030402u); // bytes {x2, 64, x3, 64}
    const hf2 bias = {static_cast<_Float16>(1152.0f), static_cast<_Float16>(1152.0f)};
    hf2 ha = __builtin_bit_cast(hf2, a) - bias;
    hf2 hb = __builtin_bit_cast(hf2, b) - bias;
    lo = __builtin_bit_cast(uint32_t, ha);
    hi = __builtin_bit_cast(uint32_t, hb);
}

} // namespace pfa2
} // namespace vegajet

// ============================================================================
// Grid: x = ceil(M * G / 16) workgroups, y = n_kv_local; block = 256. (G = n_q_local / n_kv_local.)
// enqueue_kernel(g, k, ceil(M*G/16) * 256, n_kv_local, 1, 256, 1, 1, ...)
// ============================================================================
template <bool F32Q>
__device__ __forceinline__ void vj_fa2_body(
    const float*   __restrict__ q_all,      // [M][n_q_local][256] FP32 (normed + RoPE)
    const float*   __restrict__ gate_all,   // [M][n_q_local][256] FP32
    const uint8_t* __restrict__ k_cache_q8, // [pos][n_kv_local][272]: 8 x FP16 scale, 256 x INT8
    const uint8_t* __restrict__ v_cache_q8,
    int8_t*        __restrict__ out_q8,     // [M][n_q_local][256]
    float2*        __restrict__ out_ds,     // [M][n_q_local][8]
    uint32_t M,
    uint32_t pos0,
    uint32_t n_q_local,
    uint32_t n_kv_local) {
    using namespace vegajet::pfa2;
    constexpr uint32_t KSTR_W = 68u;          // uint32 words per LDS key row (256 B INT8 + 16 B pad -> 16B-aligned, 8,704 B -> 4 WGs/CU!)

    __shared__ __attribute__((aligned(16))) uint32_t sm_kv[32u * KSTR_W];
    __shared__ float sm_sc[32u * 8u];
    __shared__ __attribute__((aligned(16))) float sm_p[32u * 16u];

    const uint32_t tid  = threadIdx.x;
    const uint32_t wave = __builtin_amdgcn_readfirstlane(tid >> 6);   // wave-uniform -> SGPR row math
    const uint32_t lane = tid & 63u;
    const uint32_t kv_h = blockIdx.y;
    const uint32_t G    = n_q_local / n_kv_local;
    const uint32_t rows_total = M * G;
    const uint32_t i0   = blockIdx.x * 16u;
    if (i0 >= rows_total || kv_h >= n_kv_local) return;

    // Key limit (inclusive) of this lane's softmax row and of the whole workgroup.
    const uint32_t i_last = min(i0 + 15u, rows_total - 1u);
    const uint32_t n_keys = pos0 + i_last / G + 1u;
    const uint32_t r_s    = lane >> 4;                         // softmax row (within the wave) of this lane
    const uint32_t i_s    = min(i0 + wave * 4u + r_s, i_last);
    const uint32_t lim_s  = pos0 + i_s / G;

    // Q rows. FP16 variant: FP16 pairs held for the whole kernel (8 VGPRs). F32Q variant: nothing held; the four
    // FP32 Q rows are re-read (L1/L2-resident, 16 KiB per WG) at the start of every S phase and live only there.
    uint32_t q01[4], q23[4];
    auto q_ptr = [&](uint32_t r) {
        const uint32_t i  = min(i0 + wave * 4u + r, i_last);
        const uint32_t m  = i / G;
        const uint32_t hq = kv_h * G + (i - m * G);
        return reinterpret_cast<const float4*>(q_all + (static_cast<uint64_t>(m) * n_q_local + hq) * 256u + lane * 4u);
    };
    if constexpr (!F32Q) {
        #pragma unroll
        for (uint32_t r = 0u; r < 4u; ++r) {
            const float4 qv = *q_ptr(r);
            const hf2 a = {static_cast<_Float16>(qv.x), static_cast<_Float16>(qv.y)};
            const hf2 b = {static_cast<_Float16>(qv.z), static_cast<_Float16>(qv.w)};
            q01[r] = __builtin_bit_cast(uint32_t, a);
            q23[r] = __builtin_bit_cast(uint32_t, b);
        }
    }

    float o[4][4];
    #pragma unroll
    for (uint32_t r = 0u; r < 4u; ++r) {
        #pragma unroll
        for (uint32_t e = 0u; e < 4u; ++e) o[r][e] = 0.0f;
    }
    float m_run = -1e30f, l_run = 0.0f;

    // Tile loader mapping: thread -> (key kk = tid>>3, 32-dim block seg = tid&7).
    const uint32_t ld_k   = tid >> 3;
    const uint32_t ld_seg = tid & 7u;
    const uint32_t blk_l  = lane >> 3;                           // quant block of this lane's 4 dims

    auto load_tile = [&](const uint8_t* base, uint32_t t0, uint4& a, uint4& b, uint32_t& s) {
        const uint32_t t = min(t0 + ld_k, n_keys - 1u);          // tail keys re-read the last row, masked later
        const uint8_t* row = base + (static_cast<uint64_t>(t) * n_kv_local + kv_h) * 272ULL;
        const uint4* p = reinterpret_cast<const uint4*>(row + 16u + ld_seg * 32u);
        a = p[0];
        b = p[1];
        s = reinterpret_cast<const uint16_t*>(row)[ld_seg];
    };
    auto store_tile = [&](const uint4& a, const uint4& b, uint32_t s) {
        // Store raw INT8 directly into LDS (half the LDS footprint -> 4 WGs/CU, zero conversion in store_tile).
        uint4* dst = reinterpret_cast<uint4*>(sm_kv + ld_k * KSTR_W + ld_seg * 8u);
        dst[0] = a;
        dst[1] = b;
        sm_sc[ld_k * 8u + ld_seg] = fp16_to_f32(static_cast<uint16_t>(s));
    };


    #pragma nounroll
    for (uint32_t t0 = 0u; t0 < n_keys; t0 += 32u) {
        __syncthreads();                                         // previous PV finished with sm_kv / sm_p
        {
            uint4 ka, kb; uint32_t ks;
            load_tile(k_cache_q8, t0, ka, kb, ks);
            store_tile(ka, kb, ks);
        }
        __syncthreads();

        if constexpr (!F32Q) {
            // ---- S = Q K^T in 16 chunks of (4 rows x 2 keys), FP16 Q via v_dot2_f32_f16 ----
            #pragma nounroll
            for (uint32_t c = 0u; c < 16u; ++c) {
                float part[8];
                #pragma unroll
                for (uint32_t jj = 0u; jj < 2u; ++jj) {
                    const uint32_t j  = c * 2u + jj;
                    const uint32_t kw = sm_kv[j * KSTR_W + lane];
                    const float ksc   = sm_sc[j * 8u + blk_l];
                    uint32_t k01_u, k23_u;
                    i8x4_to_hf2x2(kw, k01_u, k23_u);
                    const hf2 k01 = __builtin_bit_cast(hf2, k01_u);
                    const hf2 k23 = __builtin_bit_cast(hf2, k23_u);
                    #pragma unroll
                    for (uint32_t r = 0u; r < 4u; ++r) {
                        const float d = __builtin_amdgcn_fdot2(__builtin_bit_cast(hf2, q01[r]), k01,
                                        __builtin_amdgcn_fdot2(__builtin_bit_cast(hf2, q23[r]), k23, 0.0f, false), false);
                        part[r * 2u + jj] = d * ksc;
                    }
                    __builtin_amdgcn_sched_barrier(0);
                }
                // Reduce-scatter over the wave: after the halving steps xor 32, 16, 8 lane l holds index
                // k = (l >> 3) & 7 (= r*2 + jj, r = l >> 4) summed over the 8 lanes differing in bits 5..3, and
                // xor 4, 2, 1 finish the 64-lane sum: 4 + 2 + 1 + 3 = 10 cross-lane ops per 8 scores.
                #pragma unroll
                for (uint32_t i = 0u; i < 4u; ++i) {
                    const bool up = (lane & 32u) != 0u;
                    const float send = up ? part[i] : part[i + 4u];
                    const float keep = up ? part[i + 4u] : part[i];
                    part[i] = keep + xor_lane(send, 32u);
                }
                #pragma unroll
                for (uint32_t i = 0u; i < 2u; ++i) {
                    const bool up = (lane & 16u) != 0u;
                    const float send = up ? part[i] : part[i + 2u];
                    const float keep = up ? part[i + 2u] : part[i];
                    part[i] = keep + swz<VJ_SWZ_X16>(send);
                }
                float s;
                {
                    const bool up = (lane & 8u) != 0u;
                    const float send = up ? part[0] : part[1];
                    const float keep = up ? part[1] : part[0];
                    s = keep + swz<VJ_SWZ_X8>(send);
                }
                s += swz<VJ_SWZ_X4>(s);
                s += swz<VJ_SWZ_X2>(s);
                s += swz<VJ_SWZ_X1>(s);
                if ((lane & 7u) == 0u) {
                    const uint32_t j = c * 2u + ((lane >> 3) & 1u);   // this lane's row is r_s = lane >> 4
                    const bool ok = (t0 + j) <= lim_s;
                    sm_p[j * 16u + wave * 4u + r_s] = ok ? s * 0.0625f : -INFINITY;
                }
            }
        } else {
            // ---- S = Q K^T with FP32 Q: two passes of 2 rows (8 Q VGPRs live), one key per step ----
            // Reduce-scatter: xor 32 halves 2 -> 1 (row rp*2 + (l >> 5)), then xor 16, 8, 4, 2, 1: 6 ops / 2 scores.
            #pragma nounroll
            for (uint32_t rp = 0u; rp < 2u; ++rp) {
                const float4 qa = *q_ptr(rp * 2u);
                const float4 qb = *q_ptr(rp * 2u + 1u);
                const uint32_t r_w = rp * 2u + (lane >> 5);                   // row written by lanes 0 / 32
                const uint32_t lim_w = pos0 + min(i0 + wave * 4u + r_w, i_last) / G;
                #pragma nounroll
                for (uint32_t j = 0u; j < 32u; ++j) {
                    const uint32_t kw = sm_kv[j * KSTR_W + lane];
                    const float ksc = sm_sc[j * 8u + blk_l];
                    uint32_t k01_u, k23_u;
                    i8x4_to_hf2x2(kw, k01_u, k23_u);
                    const hf2 k01 = __builtin_bit_cast(hf2, k01_u);
                    const hf2 k23 = __builtin_bit_cast(hf2, k23_u);
                    const float k0 = static_cast<float>(k01.x), k1 = static_cast<float>(k01.y);
                    const float k2 = static_cast<float>(k23.x), k3 = static_cast<float>(k23.y);
                    __builtin_amdgcn_sched_barrier(0);
                    const float pa = fmaf(qa.w, k3, fmaf(qa.z, k2, fmaf(qa.y, k1, qa.x * k0))) * ksc;
                    const float pb = fmaf(qb.w, k3, fmaf(qb.z, k2, fmaf(qb.y, k1, qb.x * k0))) * ksc;
                    const bool up = (lane & 32u) != 0u;
                    float s = (up ? pb : pa) + xor_lane(up ? pa : pb, 32u);
                    s += swz<VJ_SWZ_X16>(s);
                    s += swz<VJ_SWZ_X8>(s);
                    s += swz<VJ_SWZ_X4>(s);
                    s += swz<VJ_SWZ_X2>(s);
                    s += swz<VJ_SWZ_X1>(s);
                    if ((lane & 31u) == 0u) {
                        const bool ok = (t0 + j) <= lim_w;
                        sm_p[j * 16u + wave * 4u + r_w] = ok ? s * 0.0625f : -INFINITY;
                    }
                }
            }
        }
        __syncthreads();                                         // all K reads done, raw scores visible

        {
            // Online softmax for row r_s over this tile: 16 lanes x 2 keys.
            const uint32_t jp = (lane & 15u) * 2u;
            float* p0p = sm_p + jp * 16u + wave * 4u + r_s;
            const float s0 = p0p[0];
            const float s1 = p0p[16];
            float mx = fmaxf(s0, s1);
            mx = fmaxf(mx, swz<VJ_SWZ_X1>(mx));
            mx = fmaxf(mx, swz<VJ_SWZ_X2>(mx));
            mx = fmaxf(mx, swz<VJ_SWZ_X4>(mx));
            mx = fmaxf(mx, swz<VJ_SWZ_X8>(mx));
            const float m_new = fmaxf(m_run, mx);
            const float alpha = __expf(m_run - m_new);
            const float p0 = __expf(s0 - m_new);
            const float p1 = __expf(s1 - m_new);
            float ps = p0 + p1;
            ps += swz<VJ_SWZ_X1>(ps);
            ps += swz<VJ_SWZ_X2>(ps);
            ps += swz<VJ_SWZ_X4>(ps);
            ps += swz<VJ_SWZ_X8>(ps);
            l_run = fmaf(l_run, alpha, ps);
            m_run = m_new;
            p0p[0]  = p0;
            p0p[16] = p1;
            #pragma unroll
            for (uint32_t r = 0u; r < 4u; ++r) {
                const float a_r = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(alpha), r * 16u));
                #pragma unroll
                for (uint32_t e = 0u; e < 4u; ++e) o[r][e] *= a_r;
            }
        }
        {
            uint4 va, vb; uint32_t vs;
            load_tile(v_cache_q8, t0, va, vb, vs);
            store_tile(va, vb, vs);                              // V tile into the same LDS buffer
        }
        __syncthreads();

        // ---- O += P V ----
        #pragma nounroll
        for (uint32_t j = 0u; j < 32u; ++j) {
            const float4 pr = *reinterpret_cast<const float4*>(sm_p + j * 16u + wave * 4u);
            const float vsc = sm_sc[j * 8u + blk_l];
            const uint32_t vw = sm_kv[j * KSTR_W + lane];
            uint32_t v01_u, v23_u;
            i8x4_to_hf2x2(vw, v01_u, v23_u);
            const hf2 v01 = __builtin_bit_cast(hf2, v01_u);
            const hf2 v23 = __builtin_bit_cast(hf2, v23_u);
            const float v0 = static_cast<float>(v01.x), v1 = static_cast<float>(v01.y);
            const float v2 = static_cast<float>(v23.x), v3 = static_cast<float>(v23.y);
            const float pw[4] = {pr.x * vsc, pr.y * vsc, pr.z * vsc, pr.w * vsc};
            #pragma unroll
            for (uint32_t r = 0u; r < 4u; ++r) {
                o[r][0] = fmaf(pw[r], v0, o[r][0]);
                o[r][1] = fmaf(pw[r], v1, o[r][1]);
                o[r][2] = fmaf(pw[r], v2, o[r][2]);
                o[r][3] = fmaf(pw[r], v3, o[r][3]);
            }
            __builtin_amdgcn_sched_barrier(0);
        }
    }

    // ---- Epilogue: normalise, sigmoid gate, block-32 INT8 quant ----
    #pragma unroll
    for (uint32_t r = 0u; r < 4u; ++r) {
        const uint32_t i = i0 + wave * 4u + r;
        const float l_r = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(l_run), r * 16u));
        if (i >= rows_total) continue;                            // wave-uniform
        const uint32_t m  = i / G;
        const uint32_t hq = kv_h * G + (i - m * G);
        const uint64_t rb = (static_cast<uint64_t>(m) * n_q_local + hq);
        const float4 gv = *reinterpret_cast<const float4*>(gate_all + rb * 256u + lane * 4u);
        float y[4];
        const float gg[4] = {gv.x, gv.y, gv.z, gv.w};
        #pragma unroll
        for (uint32_t e = 0u; e < 4u; ++e) {
            const float attn_v = (l_r > 0.0f) ? (o[r][e] / l_r) : 0.0f;
            y[e] = attn_v * (1.0f / (1.0f + __expf(-gg[e])));
        }
        float amax = fmaxf(fmaxf(fabsf(y[0]), fabsf(y[1])), fmaxf(fabsf(y[2]), fabsf(y[3])));
        amax = fmaxf(amax, swz<VJ_SWZ_X1>(amax));
        amax = fmaxf(amax, swz<VJ_SWZ_X2>(amax));
        amax = fmaxf(amax, swz<VJ_SWZ_X4>(amax));
        float bsum = (y[0] + y[1]) + (y[2] + y[3]);
        bsum += swz<VJ_SWZ_X1>(bsum);
        bsum += swz<VJ_SWZ_X2>(bsum);
        bsum += swz<VJ_SWZ_X4>(bsum);
        const float sc    = amax * (1.0f / 127.0f);
        const float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        uint32_t packed = 0u;
        #pragma unroll
        for (uint32_t e = 0u; e < 4u; ++e) {
            const uint32_t q = static_cast<uint32_t>(max(-127, min(127, __float2int_rn(y[e] * inv_d)))) & 0xFFu;
            packed |= q << (8u * e);
        }
        reinterpret_cast<uint32_t*>(out_q8 + rb * 256u)[lane] = packed;
        if ((lane & 7u) == 0u) out_ds[rb * 8u + blk_l] = make_float2(sc, bsum);
    }
}

// LDS (19,968 B) caps residency at 3 WGs/CU, so the compiler would size registers for 3 waves/SIMD (84 VGPRs);
// amdgpu_num_vgpr(64) holds it to the tiled-class gate.
//   vj_prefill_fa2_quant_kernel      : FP16 Q (8 VGPRs)                       -> 63 VGPRs, 0 spill
//   vj_prefill_fa2_quant_f32q_kernel : FP32 Q (re-read per tile, FP32 FMAs on the exact INT8 K values): QK^T at
//                                      FP32 accuracy for 2x the S-phase VALU; use it if the FP16-Q variant breaks
//                                      first-token parity.
extern "C" __global__ __launch_bounds__(256, 4) __attribute__((amdgpu_num_vgpr(64)))
void vj_prefill_fa2_quant_kernel(
    const float* __restrict__ q_all, const float* __restrict__ gate_all,
    const uint8_t* __restrict__ k_cache_q8, const uint8_t* __restrict__ v_cache_q8,
    int8_t* __restrict__ out_q8, float2* __restrict__ out_ds,
    uint32_t M, uint32_t pos0, uint32_t n_q_local, uint32_t n_kv_local) {
    vj_fa2_body<false>(q_all, gate_all, k_cache_q8, v_cache_q8, out_q8, out_ds, M, pos0, n_q_local, n_kv_local);
}

extern "C" __global__ __launch_bounds__(256, 4) __attribute__((amdgpu_num_vgpr(64)))
void vj_prefill_fa2_quant_f32q_kernel(
    const float* __restrict__ q_all, const float* __restrict__ gate_all,
    const uint8_t* __restrict__ k_cache_q8, const uint8_t* __restrict__ v_cache_q8,
    int8_t* __restrict__ out_q8, float2* __restrict__ out_ds,
    uint32_t M, uint32_t pos0, uint32_t n_q_local, uint32_t n_kv_local) {
    vj_fa2_body<true>(q_all, gate_all, k_cache_q8, v_cache_q8, out_q8, out_ds, M, pos0, n_q_local, n_kv_local);
}

#include "vj_prefill_gqa_fa2.hiph"
