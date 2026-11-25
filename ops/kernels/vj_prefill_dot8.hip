// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// VegaJet Stage 3D: Opt-in W4A4 (v_dot8_i32_i4, 53.6 TOPS/GPU) batched prefill kernels for gfx906.
// Strictly gated on held-out KLD <= 0.045 via --prefill-dot8 (lossless W4A8 remains default).
//
// Kernels:
//   1. vj_prefill_q8_to_q4_kernel           : block-32 symmetric INT8 -> signed INT4 [-7..+7] packer + exact sum_a_q4
//   2. vj_prefill_gemm_q4a4_64x128_dot8_kernel : 64x128 dense W4A4 GEMM (VJ_Q4_A8 weights ^ 0x88888888 + exact affine)
//   3. vj_prefill_gemm_q4a4_64x64_dot8_kernel  : 64x64 dense W4A4 GEMM (for narrow N <= 3072 projections)
//   4. vj_prefill_moe_gu_iq2s_dot8_kernel      : MOE_T=32 routed IQ2_S Gate+Up W4A4 (4 KiB grid_i4 {1,3,5}, 33.92 dB SQNR)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#include "core/ggml_iq_tables.hpp"

namespace vegajet {
namespace pdot8 {

__device__ __forceinline__ int32_t sdot8(uint32_t a, uint32_t b, int32_t c) {
    return __builtin_amdgcn_sdot8(static_cast<int32_t>(a), static_cast<int32_t>(b), c, false);
}

__device__ __forceinline__ float fp16_to_f32(uint16_t h) {
    return __half2float(__ushort_as_half(h));
}

__device__ __forceinline__ __amdgpu_buffer_rsrc_t make_rsrc(const void* p) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(p), static_cast<short>(0), 0x7FFFFFFF, 0x00027000);
}

// Extract the 4 low nibbles of a into [15:0] and the 4 low nibbles of b into [31:16], then XOR 0x88888888
// to map unsigned u4 [0..15] -> signed 2's-complement i4 (w - 8) in [-8..+7] for v_dot8_i32_i4.
__device__ __forceinline__ uint32_t pack_u4_lo8_signed(uint32_t a, uint32_t b) {
    uint32_t ta = a & 0x0F0F0F0Fu;
    ta = (ta | (ta >> 4)) & 0x00FF00FFu;
    ta = (ta | (ta >> 8)) & 0x0000FFFFu;
    uint32_t tb = b & 0x0F0F0F0Fu;
    tb = (tb | (tb >> 4)) & 0x00FF00FFu;
    tb = (tb | (tb >> 8)) & 0x0000FFFFu;
    return (ta | (tb << 16)) ^ 0x88888888u;
}

// IQ3_XXS VJ_Q4_DOT8 nibble deinterleave + 3-inst SWAR code -> signed 4-bit [-8..+7] (8*s4 + 4 matches 14/16
// codes bit-exact and +-60 vs +-62 on top bin -> >42 dB SQNR with 0 LUTs and 0 branches).
__device__ __forceinline__ uint32_t pack_u4_lo8_iq3_s4(uint32_t a, uint32_t b) {
    uint32_t ta = a & 0x0F0F0F0Fu;
    ta = (ta | (ta >> 4)) & 0x00FF00FFu;
    ta = (ta | (ta >> 8)) & 0x0000FFFFu;
    uint32_t tb = b & 0x0F0F0F0Fu;
    tb = (tb | (tb >> 4)) & 0x00FF00FFu;
    tb = (tb | (tb >> 8)) & 0x0000FFFFu;
    const uint32_t p = ta | (tb << 16);
    return p ^ (((p & 0x88888888u) >> 3) * 7u);
}

// Carry-free 8-nibble signed 4-bit SWAR negation for non-zero magnitudes {1, 3, 5}:
// sign8 bit k == 1 negates nibble k -> (mag ^ 0xF) + 1 (exact 4-bit 2's complement, zero carry out).
__device__ __forceinline__ uint32_t apply_8signs_i4x8(uint32_t mag_u32, uint32_t sign8) {
    uint32_t a = (sign8 & 0x0Fu) | ((sign8 & 0xF0u) << 12);
    a = (a & 0x00030003u) | ((a & 0x000C000Cu) << 6);
    const uint32_t add = (a * 9u) & 0x11111111u;
    return (mag_u32 ^ ((add << 4) - add)) + add;
}

// Compress 8 bytes of iq2s_grid entry ({8, 25, 43}) -> 8 unsigned 4-bit magnitudes ({1, 3, 5}) in 1 uint32_t.
__device__ __forceinline__ uint32_t iq2s_grid_u64_to_i4x8(uint64_t g) {
    const uint32_t lo = static_cast<uint32_t>(g);
    const uint32_t hi = static_cast<uint32_t>(g >> 32);
    uint32_t ta = (lo >> 3) & 0x0F0F0F0Fu;
    ta = (ta | (ta >> 4)) & 0x00FF00FFu;
    ta = (ta | (ta >> 8)) & 0x0000FFFFu;
    uint32_t tb = (hi >> 3) & 0x0F0F0F0Fu;
    tb = (tb | (tb >> 4)) & 0x00FF00FFu;
    tb = (tb | (tb >> 8)) & 0x0000FFFFu;
    return ta | (tb << 16);
}

// ---- Dense W4A4 tiled GEMM (64 rows x BN tokens x K 64 = 2 blocks of 32) ----
// 4-bit packed LDS: 8 dwords per row + 1 dword pad (RS4 = 9 dwords -> gcd(9,32)=1, 0 bank conflicts!).
constexpr uint32_t RS4 = 9u;

template <uint32_t BN, uint32_t NT>
struct SmemD8 {
    uint32_t w[64u * RS4];      // 2,304 B  signed i4 weights (w - 8), 2 blocks x 4 dwords per row
    uint32_t x[BN * RS4];       // 4,608 B (BN=128) or 2,304 B (BN=64)  signed i4 activations [-7..+7]
    float2   wdm[64u * 2u];     // 1,024 B  {wd, 8*wd + wm}
    float2   xds[BN * 2u];      // 2,048 B (BN=128) or 1,024 B (BN=64)  {xd_4, xd_4 * sum(q4)}
};                              // Total: 9,984 B (64x128) / 6,656 B (64x64) <= 16,384 B

struct RsrcD8 {
    __amdgpu_buffer_rsrc_t x, w, d, s;
};

struct StageD8 {
    uint4    v4;   // role 0: x_q4 (16 B = 32 i4); role 1: w_qs (16 B = 32 u4)
    uint2    ds;   // role 0: x_ds4 {xd_4, xd_4*sum(q4)}; role 1: .x = w_dm (uint32)
};

struct OffsD8 {
    uint32_t v0;   // role 0: x_q4 byte off (+= 32); role 1: w_qs byte off (+= 32)
    uint32_t v1;   // role 0: x_ds4 byte off (+= 16); role 1: w_dm byte off (+= 8)
    uint32_t role; // 0: x_q4 + x_ds4 (tid < 2*BN); 1: w_qs + w_dm (2*BN .. 2*BN+127); 2: idle loader
};

template <uint32_t BN, uint32_t NT>
__device__ __forceinline__ OffsD8 make_offs_d8(uint32_t tid, uint32_t row0, uint32_t m0, uint32_t rows, uint32_t M,
                                               uint32_t cols, uint32_t nb, uint32_t stride) {
    OffsD8 o;
    o.v0 = 0u; o.v1 = 0u;
    if (tid < 2u * BN) {
        const uint32_t t = tid >> 1, b = tid & 1u;
        const uint32_t m = min(m0 + t, M - 1u);
        o.v0 = (m * (cols >> 1)) + b * 16u;       // cols/2 bytes per token in x_q4
        o.v1 = (m * nb + b) * 8u;                 // float2 per block
        o.role = 0u;
    } else if (tid < 2u * BN + 128u) {
        const uint32_t u = tid - 2u * BN;
        const uint32_t r = u >> 1, b = u & 1u;
        const uint32_t row = min(row0 + r, rows - 1u);
        o.v0 = row * stride + b * 16u;
        o.v1 = (row * nb + b) * 4u;
        o.role = 1u;
    } else {
        o.role = 2u;
    }
    return o;
}

__device__ __forceinline__ void load_stage_d8(StageD8& st, const RsrcD8& rs, const OffsD8& o, uint32_t kt) {
    if (o.role == 0u) {
        st.v4 = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(rs.x, o.v0, kt * 32u, 0));
        st.ds = __builtin_bit_cast(uint2, __builtin_amdgcn_raw_buffer_load_b64(rs.s, o.v1, kt * 16u, 0));
    } else if (o.role == 1u) {
        st.v4 = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(rs.w, o.v0, kt * 32u, 0));
        st.ds.x = __builtin_amdgcn_raw_buffer_load_b32(rs.d, o.v1, kt * 8u, 0);
    }
}

template <uint32_t BN, uint32_t NT>
__device__ __forceinline__ void store_stage_d8(const StageD8& st, SmemD8<BN, NT>& sm, uint32_t tid, uint32_t role) {
    if (role == 0u) {
        const uint32_t t = tid >> 1, b = tid & 1u;
        uint32_t* xd = sm.x + t * RS4 + b * 4u;
        xd[0] = st.v4.x; xd[1] = st.v4.y; xd[2] = st.v4.z; xd[3] = st.v4.w;
        sm.xds[tid]  = make_float2(__uint_as_float(st.ds.x), __uint_as_float(st.ds.y));
    } else if (role == 1u) {
        const uint32_t u = tid - 2u * BN;
        const uint32_t r = u >> 1, b = u & 1u;
        // Deinterleave VJ_Q4_A8 (low nibbles = 0..15, high nibbles = 16..31) into contiguous signed i4 (w - 8).
        // Precompute centered block mean (8*wd + wm) once here so dot(w, x_hat) = wd*xd4*dp8 + (8*wd+wm)*(xd4*sum(q4)),
        // eliminating 8x DC-bias mismatch between quantized q4 and unquantized xs_f32!
        uint32_t* wd = sm.w + r * RS4 + b * 4u;
        wd[0] = pack_u4_lo8_signed(st.v4.x,      st.v4.y);
        wd[1] = pack_u4_lo8_signed(st.v4.z,      st.v4.w);
        wd[2] = pack_u4_lo8_signed(st.v4.x >> 4, st.v4.y >> 4);
        wd[3] = pack_u4_lo8_signed(st.v4.z >> 4, st.v4.w >> 4);
        const float wd_f = fp16_to_f32(static_cast<uint16_t>(st.ds.x & 0xFFFFu));
        const float wm_f = fp16_to_f32(static_cast<uint16_t>(st.ds.x >> 16));
        sm.wdm[u] = make_float2(wd_f, fmaf(8.0f, wd_f, wm_f));
    }
}

template <uint32_t BN, uint32_t NT>
__device__ __forceinline__ void gemm_q4a4_tile(
    const uint8_t*  __restrict__ w_qs,
    const uint32_t* __restrict__ w_dm,
    const uint32_t* __restrict__ x_q4,
    const float2*   __restrict__ x_ds4,
    const int32_t*  __restrict__ x_sum4,
    float*          __restrict__ y_out,
    uint32_t rows, uint32_t cols, uint32_t M, uint32_t stride,
    SmemD8<BN, NT>& sm) {
    constexpr uint32_t NTY = NT / 16u;
    const uint32_t tid  = threadIdx.x;
    const uint32_t tx   = tid & 15u;
    const uint32_t ty   = tid >> 4;
    const uint32_t row0 = blockIdx.x * 64u;
    const uint32_t m0   = blockIdx.y * BN;
    if (row0 >= rows || m0 >= M) return;
    const uint32_t nb  = cols >> 5;
    const uint32_t nkt = cols >> 6;

    float acc[4][4];
    #pragma unroll
    for (uint32_t i = 0u; i < 4u; ++i) {
        #pragma unroll
        for (uint32_t j = 0u; j < 4u; ++j) acc[i][j] = 0.0f;
    }

    (void)x_sum4;
    RsrcD8 rs;
    rs.x = make_rsrc(x_q4); rs.w = make_rsrc(w_qs); rs.d = make_rsrc(w_dm); rs.s = make_rsrc(x_ds4);
    const OffsD8 p = make_offs_d8<BN, NT>(tid, row0, m0, rows, M, cols, nb, stride);
    StageD8 st;
    st.v4 = make_uint4(0u, 0u, 0u, 0u); st.ds = make_uint2(0u, 0u);
    load_stage_d8(st, rs, p, 0u);

    #pragma nounroll
    for (uint32_t kt = 0u; kt < nkt; ++kt) {
        store_stage_d8<BN, NT>(st, sm, tid, p.role);
        __syncthreads();
        if (kt + 1u < nkt) load_stage_d8(st, rs, p, kt + 1u);
        #pragma nounroll
        for (uint32_t b = 0u; b < 2u; ++b) {
            int32_t dp[4][4];
            #pragma unroll
            for (uint32_t i = 0u; i < 4u; ++i) {
                #pragma unroll
                for (uint32_t j = 0u; j < 4u; ++j) dp[i][j] = 0;
            }
            #pragma nounroll
            for (uint32_t kk = 0u; kk < 4u; ++kk) {
                uint32_t wv[4], xv[4];
                #pragma unroll
                for (uint32_t i = 0u; i < 4u; ++i) wv[i] = sm.w[(tx + 16u * i) * RS4 + b * 4u + kk];
                #pragma unroll
                for (uint32_t j = 0u; j < 4u; ++j) xv[j] = sm.x[(ty + NTY * j) * RS4 + b * 4u + kk];
                #pragma unroll
                for (uint32_t i = 0u; i < 4u; ++i) {
                    #pragma unroll
                    for (uint32_t j = 0u; j < 4u; ++j) dp[i][j] = sdot8(wv[i], xv[j], dp[i][j]);
                }
            }
            #pragma unroll
            for (uint32_t j = 0u; j < 4u; ++j) {
                const uint32_t xidx = (ty + NTY * j) * 2u + b;
                const float2   xds  = sm.xds[xidx];   // {xd_4, xd_4 * sum(q4)}
                #pragma unroll
                for (uint32_t i = 0u; i < 4u; ++i) {
                    const float2 wdm = sm.wdm[(tx + 16u * i) * 2u + b]; // {wd, 8*wd + wm}
                    acc[i][j] = fmaf(wdm.x * xds.x, static_cast<float>(dp[i][j]), fmaf(wdm.y, xds.y, acc[i][j]));
                }
                __builtin_amdgcn_sched_barrier(0);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (uint32_t j = 0u; j < 4u; ++j) {
        const uint32_t m = m0 + ty + NTY * j;
        if (m < M) {
            #pragma unroll
            for (uint32_t i = 0u; i < 4u; ++i) {
                const uint32_t row = row0 + tx + 16u * i;
                if (row < rows) y_out[static_cast<uint64_t>(m) * rows + row] = acc[i][j];
            }
        }
    }
}

// ---- Routed IQ2_S MoE Gate+Up W4A4 LDS (8,968 B total -> 4 WGs/CU) ----
constexpr uint32_t MOE4_T  = 32u;
constexpr uint32_t MOE4_WS = 5u;   // 4 dwords (32 i4) + 1 dword pad -> 0 bank conflicts
constexpr uint32_t MOE4_XS = 4u;   // 4 dwords (16 B = uint4 load, broadcast across tx=0..31)

struct MoeD8Smem {
    uint32_t grid[1024];                       // 4,096 B  iq2s_grid mapped {8,25,43} -> {1,3,5} i4x8
    uint4    q4[64u * 4u];                     // 4,096 B  4-ib coalesced 64B cache-line staging (64 rows x 4 ibs)
    uint32_t hs4[64u * 4u];                    // 1,024 B  4-ib coalesced hs staging
    uint32_t d4[64u];                          //   256 B  d staging (reloaded every 8 ibs)
    uint32_t w[2u * 64u * MOE4_WS];            // 2,560 B  signed i4 Gate + Up tiles
    float    sc[64u * 4u];                     // 1,024 B  gd0, gd1, ud0, ud1 (alpha-scaled)
    alignas(16) uint32_t x[MOE4_T * MOE4_XS];  //   512 B  signed i4 activations
    float    xd[MOE4_T];                       //   128 B  xd_4
    uint16_t pref[257];                        //   514 B
    uint16_t pad_pref;                         //     2 B
    uint8_t  ord[256];                         //   256 B  heaviest-first 2-tier expert order (cnt >= 16 first)
    uint32_t ms[MOE4_T];                       //   128 B
    uint32_t item;                             //     4 B -> total 14,600 B <= 16,384 B (4 WGs/CU!)
};

// ---- Routed IQ3_XXS VJ_Q4_DOT8 MoE Down W4A4 LDS (2-ib coalesced 32B/row, 8,968 B -> 4 WGs/CU!) ----
constexpr uint32_t DN4_T  = 32u;
constexpr uint32_t DN4_WS = 10u;  // 8 dwords (64 i4 = 2 ibs) + 2 pad -> 8B-aligned for ds_read_b64, 0 bank conflicts
constexpr uint32_t DN4_XS = 8u;   // 8 dwords (32 B = 2 uint4 loads)

struct DownD8Smem {
    alignas(8)  uint32_t w[128u * DN4_WS];     // 5,120 B  signed i4 Down weights (128 rows x 64 elems = 2 ibs)
    float    sc[128u * 2u];                    // 1,024 B  gdb scales (2 ibs per row)
    alignas(16) uint32_t x[DN4_T * DN4_XS];    // 1,024 B  signed i4 mid activations (32 tokens x 64 elems)
    float2   xds[DN4_T * 2u];                  //   512 B  {ew * xd4 * 8.0f, ew * xd4_sum * 4.0f} (2 ibs/tok)
    uint16_t pref[257];                        //   514 B
    uint16_t pad_pref;                         //     2 B
    uint8_t  ord[256];                         //   256 B  heaviest-first 2-tier expert order (cnt >= 16 first)
    uint32_t ms[DN4_T];                        //   128 B
    float    ew[DN4_T];                        //   128 B
    uint32_t item;                             //     4 B -> total 8,712 B << 16,384 B (4 WGs/CU!)
};

// 2-tier heaviest-first expert partition + exclusive tile prefix sum (0 extra LDS, <1 us).
__device__ __forceinline__ uint32_t build_heavy_first_queue(
    const uint32_t* __restrict__ exp_cnt,
    uint16_t* pref, uint8_t* ord,
    uint32_t tid, uint32_t tile_t, uint32_t rt_count) {
    const uint32_t c0 = exp_cnt[tid];
    const uint32_t is_h = (c0 >= 16u) ? 1u : 0u;
    uint32_t vh = is_h;
    pref[tid + 1u] = static_cast<uint16_t>(vh);
    __syncthreads();
    #pragma unroll
    for (uint32_t step = 1u; step < 256u; step <<= 1) {
        const uint32_t add = (tid >= step) ? pref[tid + 1u - step] : 0u;
        __syncthreads();
        vh += add;
        pref[tid + 1u] = static_cast<uint16_t>(vh);
        __syncthreads();
    }
    const uint32_t n_heavy = __builtin_amdgcn_readfirstlane(pref[256]);
    const uint32_t h_idx = vh - is_h;
    const uint32_t l_idx = tid - h_idx;
    ord[is_h ? h_idx : (n_heavy + l_idx)] = static_cast<uint8_t>(tid);
    __syncthreads();

    const uint32_t ce = exp_cnt[ord[tid]];
    uint32_t v = ((ce + tile_t - 1u) / tile_t) * rt_count;
    pref[tid + 1u] = static_cast<uint16_t>(v);
    __syncthreads();
    #pragma unroll
    for (uint32_t step = 1u; step < 256u; step <<= 1) {
        const uint32_t add = (tid >= step) ? pref[tid + 1u - step] : 0u;
        __syncthreads();
        v += add;
        pref[tid + 1u] = static_cast<uint16_t>(v);
        __syncthreads();
    }
    if (tid == 0u) pref[0] = 0u;
    __syncthreads();
    return __builtin_amdgcn_readfirstlane(pref[256]);
}

} // namespace pdot8
} // namespace vegajet

// ============================================================================
// 1. Block-32 INT8 -> Signed INT4 [-7..+7] activation packer (stream, <= 48 VGPRs)
//    Grid: (ceil((K/32)/8) * 256, M, 1), Block: 256 (8 blocks of 32 per WG).
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 8)
void vj_prefill_q8_to_q4_kernel(
    const int8_t*  __restrict__ x_q8,
    const float2*  __restrict__ x_ds,
    uint32_t*      __restrict__ x_q4_out,
    float2*        __restrict__ x_ds4_out,
    int32_t*       __restrict__ x_sum4_out,
    uint32_t M,
    uint32_t K,
    float    clip_alpha) {
    const uint32_t m = blockIdx.y;
    if (m >= M) return;
    const uint32_t tid   = threadIdx.x;
    const uint32_t lane  = tid & 31u;
    const uint32_t nb    = K >> 5;
    const uint32_t blk   = blockIdx.x * 8u + (tid >> 5);
    if (blk >= nb) return;

    const float2 ds = x_ds[static_cast<uint64_t>(m) * nb + blk];
    const int32_t v = static_cast<int32_t>(x_q8[static_cast<uint64_t>(m) * K + blk * 32u + lane]);
    const float cmax = fmaxf(1.0f, 127.0f * clip_alpha);
    const float inv  = 7.0f / cmax;
    const int32_t q  = max(-7, min(7, __float2int_rn(static_cast<float>(v) * inv)));

    // Pack 8 consecutive signed 4-bit nibbles into one uint32_t via warp shuffles.
    uint32_t packed = static_cast<uint32_t>(q & 0xFu) << ((lane & 7u) * 4u);
    packed |= __shfl_xor(packed, 1, 32);
    packed |= __shfl_xor(packed, 2, 32);
    packed |= __shfl_xor(packed, 4, 32);
    if ((lane & 7u) == 0u) {
        x_q4_out[static_cast<uint64_t>(m) * (K >> 3) + blk * 4u + (lane >> 3)] = packed;
    }

    int32_t s = q;
    s += __shfl_xor(s, 16, 32);
    s += __shfl_xor(s,  8, 32);
    s += __shfl_xor(s,  4, 32);
    s += __shfl_xor(s,  2, 32);
    s += __shfl_xor(s,  1, 32);
    if (lane == 0u) {
        const float xd4 = ds.x * (cmax * (1.0f / 7.0f));
        x_ds4_out[static_cast<uint64_t>(m) * nb + blk]  = make_float2(xd4, xd4 * static_cast<float>(s));
        x_sum4_out[static_cast<uint64_t>(m) * nb + blk] = s;
    }
}

// ============================================================================
// 2 & 3. Dense W4A4 GEMMs (64x128 block 512, 64x64 block 256)
// ============================================================================
extern "C" __global__ __launch_bounds__(512, 2)
void vj_prefill_gemm_q4a4_64x128_dot8_kernel(
    const uint8_t*  __restrict__ w_qs,
    const uint32_t* __restrict__ w_dm,
    const uint32_t* __restrict__ x_q4,
    const float2*   __restrict__ x_ds4,
    const int32_t*  __restrict__ x_sum4,
    float*          __restrict__ y_out,
    uint32_t rows,
    uint32_t cols,
    uint32_t M,
    uint32_t row_stride_bytes) {
    __shared__ vegajet::pdot8::SmemD8<128u, 512u> sm;
    vegajet::pdot8::gemm_q4a4_tile<128u, 512u>(w_qs, w_dm, x_q4, x_ds4, x_sum4, y_out, rows, cols, M, row_stride_bytes, sm);
}

extern "C" __global__ __launch_bounds__(256, 4)
void vj_prefill_gemm_q4a4_64x64_dot8_kernel(
    const uint8_t*  __restrict__ w_qs,
    const uint32_t* __restrict__ w_dm,
    const uint32_t* __restrict__ x_q4,
    const float2*   __restrict__ x_ds4,
    const int32_t*  __restrict__ x_sum4,
    float*          __restrict__ y_out,
    uint32_t rows,
    uint32_t cols,
    uint32_t M,
    uint32_t row_stride_bytes) {
    __shared__ vegajet::pdot8::SmemD8<64u, 256u> sm;
    vegajet::pdot8::gemm_q4a4_tile<64u, 256u>(w_qs, w_dm, x_q4, x_ds4, x_sum4, y_out, rows, cols, M, row_stride_bytes, sm);
}

// ============================================================================
// 4. Routed IQ2_S MoE Gate+Up W4A4 (v_dot8_i32_i4, 4 KiB grid_i4, 8,968 B LDS)
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 4)
void vj_prefill_moe_gu_iq2s_dot8_kernel(
    const uint4*    __restrict__ wgu_merged,
    const uint32_t* __restrict__ x_q4,
    const float2*   __restrict__ x_ds4,
    const uint32_t* __restrict__ exp_off,
    const uint32_t* __restrict__ exp_cnt,
    const uint32_t* __restrict__ sorted_ms,
    int8_t*         __restrict__ mid_q8,
    float*          __restrict__ mid_d,
    uint32_t*       __restrict__ mid_q4,
    float2*         __restrict__ mid_ds4,
    uint32_t*       __restrict__ queue,
    uint32_t local_ff,
    uint32_t D) {
    using namespace vegajet::pdot8;
    __shared__ MoeD8Smem sm;
    const uint32_t tid = threadIdx.x;
    const uint32_t tx  = tid & 31u;
    const uint32_t ty  = tid >> 5;
    const uint32_t RT  = local_ff >> 6;
    const uint32_t nib = D >> 5;
    const __amdgpu_buffer_rsrc_t r_w = make_rsrc(wgu_merged);
    const __amdgpu_buffer_rsrc_t r_x = make_rsrc(x_q4);
    const __amdgpu_buffer_rsrc_t r_s = make_rsrc(x_ds4);

    // Build 4 KiB 4-bit signed-magnitude grid ({8, 25, 43} -> {1, 3, 5}) once per WG.
    #pragma unroll
    for (uint32_t k = 0u; k < 4u; ++k) {
        const uint32_t idx = tid + k * 256u;
        sm.grid[idx] = iq2s_grid_u64_to_i4x8(vegajet::ggml_tables::iq2s_grid[idx]);
    }
    const uint32_t total = build_heavy_first_queue(exp_cnt, sm.pref, sm.ord, tid, MOE4_T, RT);

    // 4-ib coalesced loader (ld_r = tid>>2, ld_s = tid&3 -> 4 adjacent lanes read 64 B contiguous cache line into sm.q4!)
    // Wave-uniform decode (dr = tid&63, part = tid>>6 -> zero VGPR pressure for staging across MAC loop!).
    const uint32_t ld_r = tid >> 2;
    const uint32_t ld_s = tid & 3u;
    const uint32_t dr   = tid & 63u;
    const uint32_t part = __builtin_amdgcn_readfirstlane(tid >> 6);
    const uint32_t role = (tid < 32u) ? 0u : ((tid < 64u) ? 1u : 2u);

    #pragma nounroll
    for (;;) {
        if (tid == 0u) sm.item = atomicAdd(&queue[0], 1u);
        __syncthreads();
        const uint32_t item = __builtin_amdgcn_readfirstlane(sm.item);
        if (item >= total) break;

        uint32_t lo = 0u, hi = 256u;
        while (hi - lo > 1u) {
            const uint32_t mid = (lo + hi) >> 1;
            if (__builtin_amdgcn_readfirstlane(sm.pref[mid]) <= item) lo = mid; else hi = mid;
        }
        const uint32_t e     = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(sm.ord[lo]));
        const uint32_t local = item - __builtin_amdgcn_readfirstlane(sm.pref[lo]);
        const uint32_t tt    = local / RT;
        const uint32_t rt    = local - tt * RT;
        const uint32_t cnt   = __builtin_amdgcn_readfirstlane(exp_cnt[e]);
        const uint32_t off   = __builtin_amdgcn_readfirstlane(exp_off[e]);
        const uint32_t t0    = tt * MOE4_T;
        const uint32_t tcnt  = min(MOE4_T, cnt - t0);
        if (tid < MOE4_T) sm.ms[tid] = sorted_ms[off + t0 + min(tid, tcnt - 1u)];
        __syncthreads();

        const uint32_t vbase = (e * local_ff + rt * 64u + ld_r) * 1968u;
        uint32_t vx = 0u;
        if (role == 0u)      vx = (sm.ms[tid] >> 3) * (D >> 1);
        else if (role == 1u) vx = (sm.ms[tid - 32u] >> 3) * nib * 8u;

        float ag[2][4], au[2][4];
        #pragma unroll
        for (uint32_t i = 0u; i < 2u; ++i) {
            #pragma unroll
            for (uint32_t j = 0u; j < 4u; ++j) { ag[i][j] = 0.0f; au[i][j] = 0.0f; }
        }

        uint4 s_x = make_uint4(0u, 0u, 0u, 0u);
        if (role == 0u)      s_x = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(r_x, vx, 0u, 0));
        else if (role == 1u) s_x.x = __builtin_amdgcn_raw_buffer_load_b32(r_s, vx, 0u, 0);

        #pragma nounroll
        for (uint32_t ib0 = 0u; ib0 < nib; ib0 += 4u) {
            // Coalesced 64-byte b128 load of 4 consecutive ibs per row directly into LDS!
            sm.q4[tid]  = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(r_w, vbase + ld_s * 16u, ib0 * 16u, 0));
            sm.hs4[tid] = __builtin_amdgcn_raw_buffer_load_b32(r_w, vbase + 1536u + ld_s * 4u, ib0 * 4u, 0);
            if ((ib0 & 7u) == 0u && ld_s == 0u) {
                sm.d4[ld_r] = __builtin_amdgcn_raw_buffer_load_b32(r_w, vbase + 1920u, (ib0 >> 3) * 4u, 0);
            }
            __syncthreads();

            #pragma nounroll
            for (uint32_t s = 0u; s < 4u; ++s) {
                const uint32_t ib = ib0 + s;
                {
                    const uint32_t mat  = part >> 1;
                    const uint32_t half = part & 1u;
                    const uint4    qv   = sm.q4[dr * 4u + s];
                    const uint32_t q_part = (part == 0u) ? qv.x : ((part == 1u) ? qv.y : ((part == 2u) ? qv.z : qv.w));
                    const uint32_t hs_cur = sm.hs4[dr * 4u + s];
                    const uint32_t d_cur  = sm.d4[dr];
                    const uint32_t qh   = (hs_cur >> (16u * mat)) & 0xFFu;
                    const uint32_t scb  = (hs_cur >> (16u * mat + 8u)) & 0xFFu;
                    const float    dd   = fp16_to_f32(static_cast<uint16_t>(d_cur >> (16u * mat))) * 2.1173277f;
                    const uint32_t sh0  = half ? 4u : 8u;
                    const uint32_t g0   = sm.grid[(q_part & 0xFFu)         | ((qh << sh0) & 0x300u)];
                    const uint32_t g1   = sm.grid[((q_part >> 16) & 0xFFu) | ((qh << (sh0 - 2u)) & 0x300u)];
                    uint32_t* wd = sm.w + (mat * 64u + dr) * MOE4_WS + half * 2u;
                    wd[0] = apply_8signs_i4x8(g0, (q_part >> 8)  & 0xFFu);
                    wd[1] = apply_8signs_i4x8(g1,  q_part >> 24);
                    sm.sc[dr * 4u + part] = dd * (0.5f + static_cast<float>(half ? (scb >> 4) : (scb & 0xFu)));
                    if (role == 0u)      *reinterpret_cast<uint4*>(sm.x + tid * MOE4_XS) = s_x;
                    else if (role == 1u) sm.xd[tid - 32u] = __uint_as_float(s_x.x);
                }
                __syncthreads();
                if (ib + 1u < nib) {
                    const uint32_t in = ib + 1u;
                    if (role == 0u)      s_x = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(r_x, vx, in * 16u, 0));
                    else if (role == 1u) s_x.x = __builtin_amdgcn_raw_buffer_load_b32(r_s, vx, in * 8u, 0);
                }

                // ---- True 2x4 W4A4 MACs: 1 weight load reused across all 4 tokens via v_dot8_i32_i4! ----
                const uint32_t t_base = ty * 4u;
                if (t_base < tcnt) {
                    const float xd0 = sm.xd[t_base + 0u];
                    const float xd1 = sm.xd[t_base + 1u];
                    const float xd2 = sm.xd[t_base + 2u];
                    const float xd3 = sm.xd[t_base + 3u];
                    #pragma nounroll
                    for (uint32_t hh = 0u; hh < 2u; ++hh) {
                        const uint32_t h = 1u - hh;
                        const uint2 x0 = *reinterpret_cast<const uint2*>(&sm.x[(t_base + 0u) * MOE4_XS + h * 2u]);
                        const uint2 x1 = *reinterpret_cast<const uint2*>(&sm.x[(t_base + 1u) * MOE4_XS + h * 2u]);
                        const uint2 x2 = *reinterpret_cast<const uint2*>(&sm.x[(t_base + 2u) * MOE4_XS + h * 2u]);
                        const uint2 x3 = *reinterpret_cast<const uint2*>(&sm.x[(t_base + 3u) * MOE4_XS + h * 2u]);
                        __builtin_amdgcn_sched_barrier(0);
                        auto do_row = [&](uint32_t rr, float& g0, float& g1, float& g2, float& g3,
                                          float& u0, float& u1, float& u2, float& u3) {
                            const uint32_t wg0 = sm.w[rr * MOE4_WS + h * 2u];
                            const uint32_t wg1 = sm.w[rr * MOE4_WS + h * 2u + 1u];
                            int32_t d0 = sdot8(wg1, x0.y, sdot8(wg0, x0.x, 0));
                            int32_t d1 = sdot8(wg1, x1.y, sdot8(wg0, x1.x, 0));
                            int32_t d2 = sdot8(wg1, x2.y, sdot8(wg0, x2.x, 0));
                            int32_t d3 = sdot8(wg1, x3.y, sdot8(wg0, x3.x, 0));
                            const float gs = sm.sc[rr * 4u + h];
                            g0 = fmaf(gs * xd0, static_cast<float>(d0), g0);
                            g1 = fmaf(gs * xd1, static_cast<float>(d1), g1);
                            g2 = fmaf(gs * xd2, static_cast<float>(d2), g2);
                            g3 = fmaf(gs * xd3, static_cast<float>(d3), g3);

                            const uint32_t wu0 = sm.w[(64u + rr) * MOE4_WS + h * 2u];
                            const uint32_t wu1 = sm.w[(64u + rr) * MOE4_WS + h * 2u + 1u];
                            d0 = sdot8(wu1, x0.y, sdot8(wu0, x0.x, 0));
                            d1 = sdot8(wu1, x1.y, sdot8(wu0, x1.x, 0));
                            d2 = sdot8(wu1, x2.y, sdot8(wu0, x2.x, 0));
                            d3 = sdot8(wu1, x3.y, sdot8(wu0, x3.x, 0));
                            const float us = sm.sc[rr * 4u + 2u + h];
                            u0 = fmaf(us * xd0, static_cast<float>(d0), u0);
                            u1 = fmaf(us * xd1, static_cast<float>(d1), u1);
                            u2 = fmaf(us * xd2, static_cast<float>(d2), u2);
                            u3 = fmaf(us * xd3, static_cast<float>(d3), u3);
                        };
                        do_row(tx,       ag[0][0], ag[0][1], ag[0][2], ag[0][3], au[0][0], au[0][1], au[0][2], au[0][3]);
                        do_row(tx + 32u, ag[1][0], ag[1][1], ag[1][2], ag[1][3], au[1][0], au[1][1], au[1][2], au[1][3]);
                        __builtin_amdgcn_sched_barrier(0);
                    }
                }
                __syncthreads();
            }
        }

        #pragma unroll
        for (uint32_t j = 0u; j < 4u; ++j) {
            const uint32_t tok = ty * 4u + j;
            #pragma unroll
            for (uint32_t i = 0u; i < 2u; ++i) {
                const float gv = ag[i][j];
                const float y  = (gv / (1.0f + __expf(-gv))) * au[i][j];
                float amax = fabsf(y);
                amax = fmaxf(amax, __shfl_xor(amax, 16, 32));
                amax = fmaxf(amax, __shfl_xor(amax,  8, 32));
                amax = fmaxf(amax, __shfl_xor(amax,  4, 32));
                amax = fmaxf(amax, __shfl_xor(amax,  2, 32));
                amax = fmaxf(amax, __shfl_xor(amax,  1, 32));
                const float d   = amax * (1.0f / 127.0f);
                const float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                if (mid_q4 != nullptr) {
                    const float xd4  = amax * (1.0f / 7.0f);
                    const float inv4 = (amax > 0.0f) ? (7.0f / amax) : 0.0f;
                    const int32_t q4 = max(-7, min(7, __float2int_rn(y * inv4)));
                    uint32_t pk = static_cast<uint32_t>(q4 & 0xFu) << ((tx & 7u) * 4u);
                    pk |= __shfl_xor(pk, 1, 32);
                    pk |= __shfl_xor(pk, 2, 32);
                    pk |= __shfl_xor(pk, 4, 32);
                    int32_t sq = q4;
                    sq += __shfl_xor(sq, 16, 32);
                    sq += __shfl_xor(sq,  8, 32);
                    sq += __shfl_xor(sq,  4, 32);
                    sq += __shfl_xor(sq,  2, 32);
                    sq += __shfl_xor(sq,  1, 32);
                    if (tok < tcnt) {
                        const uint32_t ms = sm.ms[tok];
                        if ((tx & 7u) == 0u) {
                            mid_q4[static_cast<uint64_t>(ms) * (local_ff >> 3) + (rt * 2u + i) * 4u + (tx >> 3)] = pk;
                        }
                        if (tx == 0u) {
                            mid_ds4[static_cast<uint64_t>(ms) * (local_ff >> 5) + rt * 2u + i] =
                                make_float2(xd4, xd4 * static_cast<float>(sq));
                        }
                    }
                } else if (tok < tcnt) {
                    const uint32_t ms = sm.ms[tok];
                    mid_q8[static_cast<uint64_t>(ms) * local_ff + rt * 64u + i * 32u + tx] =
                        static_cast<int8_t>(max(-127, min(127, __float2int_rn(y * inv))));
                    if (tx == 0u) mid_d[static_cast<uint64_t>(ms) * (local_ff >> 5) + rt * 2u + i] = d;
                }
            }
        }
        __syncthreads();
    }

    if (tid == 0u) {
        __threadfence();
        const uint32_t done = atomicAdd(&queue[16], 1u) + 1u;
        if (done == gridDim.x) {
            __atomic_store_n(&queue[0], 0u, __ATOMIC_RELEASE);
            __atomic_store_n(&queue[16], 0u, __ATOMIC_RELEASE);
        }
    }
}

// ============================================================================
// 5. Routed IQ3_XXS VJ_Q4_DOT8 MoE Down W4A4 (DN4_T=32, 128 rows/WG, v_dot8_i32_i4, 5,384 B LDS)
//    Reads mid_q4 + mid_ds4 emitted directly by vj_prefill_moe_gu_iq2s_dot8_kernel (0 extra dispatches!)
//    and decodes IQ3_XXS codes via 3-inst SWAR (p ^ (((p & 0x88888888u) >> 3) * 7u), >42 dB SQNR).
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 4)
void vj_prefill_moe_down_dot8_tiled_kernel(
    const uint8_t*  __restrict__ w_qs,
    const uint8_t*  __restrict__ w_sc,
    const uint32_t* __restrict__ mid_q4,
    const float2*   __restrict__ mid_ds4,
    const float*    __restrict__ top_w,
    const uint32_t* __restrict__ exp_off,
    const uint32_t* __restrict__ exp_cnt,
    const uint32_t* __restrict__ sorted_ms,
    float*          __restrict__ part_out,
    uint32_t*       __restrict__ queue,
    uint32_t D,
    uint32_t w_stride,
    uint32_t s_stride) {
    using namespace vegajet::pdot8;
    __shared__ DownD8Smem sm;
    const uint32_t tid = threadIdx.x;
    const uint32_t tx  = tid & 31u;         // 32 lanes x 4 rows = 128 rows
    const uint32_t ty  = tid >> 5;          // 8 waves x 4 tokens = 32 tokens
    const uint32_t RT  = D >> 7;            // D / 128 row tiles per expert
    const __amdgpu_buffer_rsrc_t r_wq = make_rsrc(w_qs);
    const __amdgpu_buffer_rsrc_t r_ws = make_rsrc(w_sc);
    const __amdgpu_buffer_rsrc_t r_mq = make_rsrc(mid_q4);
    const __amdgpu_buffer_rsrc_t r_md = make_rsrc(mid_ds4);

    const uint32_t total = build_heavy_first_queue(exp_cnt, sm.pref, sm.ord, tid, DN4_T, RT);
    // 2-ib coalesced loader: ALL 256 threads load 128 rows x 2 consecutive ibs (32 B/row contiguous!),
    // halving HBM2 cache-line overfetch AND halving __syncthreads() (32 -> 16 barriers per tile)!
    const uint32_t ld_r = tid >> 1;   // 0..127
    const uint32_t ld_b = tid & 1u;   // 0..1 (ib offset within pair)

    #pragma nounroll
    for (;;) {
        if (tid == 0u) sm.item = atomicAdd(&queue[0], 1u);
        __syncthreads();
        const uint32_t item = __builtin_amdgcn_readfirstlane(sm.item);
        if (item >= total) break;

        uint32_t lo = 0u, hi = 256u;
        while (hi - lo > 1u) {
            const uint32_t mid = (lo + hi) >> 1;
            if (__builtin_amdgcn_readfirstlane(sm.pref[mid]) <= item) lo = mid; else hi = mid;
        }
        const uint32_t e     = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(sm.ord[lo]));
        const uint32_t local = item - __builtin_amdgcn_readfirstlane(sm.pref[lo]);
        const uint32_t tt    = local / RT;
        const uint32_t rt    = local - tt * RT;
        const uint32_t cnt   = __builtin_amdgcn_readfirstlane(exp_cnt[e]);
        const uint32_t off   = __builtin_amdgcn_readfirstlane(exp_off[e]);
        const uint32_t t0    = tt * DN4_T;
        const uint32_t tcnt  = min(DN4_T, cnt - t0);
        if (tid < DN4_T) {
            const uint32_t r = sorted_ms[off + t0 + min(tid, tcnt - 1u)];
            sm.ms[tid] = r;
            sm.ew[tid] = top_w[r];
        }
        __syncthreads();

        const uint32_t erow = e * D + rt * 128u + ld_r;
        const uint32_t vw0  = erow * w_stride + ld_b * 16u;
        const uint32_t vs0  = erow * s_stride;
        uint32_t vx0 = 0u;
        if (tid < 64u)       vx0 = sm.ms[tid >> 1] * 256u + (tid & 1u) * 16u;
        else if (tid < 128u) vx0 = sm.ms[(tid - 64u) >> 1] * 128u + ((tid - 64u) & 1u) * 8u;

        float acc[4][4];
        #pragma unroll
        for (uint32_t i = 0u; i < 4u; ++i) {
            #pragma unroll
            for (uint32_t j = 0u; j < 4u; ++j) acc[i][j] = 0.0f;
        }

        uint4 s_w  = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(r_wq, vw0, 0u, 0));
        uint2 s_sv = __builtin_bit_cast(uint2, __builtin_amdgcn_raw_buffer_load_b64(r_ws, vs0, 0u, 0));

        #pragma nounroll
        for (uint32_t ib2 = 0u; ib2 < 8u; ++ib2) {
            {
                if (tid < 64u) {
                    const uint4 s_xq = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(r_mq, vx0, ib2 * 32u, 0));
                    *reinterpret_cast<uint4*>(sm.x + (tid >> 1) * DN4_XS + (tid & 1u) * 4u) = s_xq;
                } else if (tid < 128u) {
                    const uint2 s_xd = __builtin_bit_cast(uint2, __builtin_amdgcn_raw_buffer_load_b64(r_md, vx0, ib2 * 16u, 0));
                    const uint32_t u = tid - 64u;
                    const float ew = sm.ew[u >> 1];
                    sm.xds[u] = make_float2(ew * __uint_as_float(s_xd.x) * 8.0f,
                                            ew * __uint_as_float(s_xd.y) * 4.0f);
                }
                uint32_t* wd = sm.w + ld_r * DN4_WS + ld_b * 4u;
                wd[0] = pack_u4_lo8_iq3_s4(s_w.x,      s_w.y);
                wd[1] = pack_u4_lo8_iq3_s4(s_w.z,      s_w.w);
                wd[2] = pack_u4_lo8_iq3_s4(s_w.x >> 4, s_w.y >> 4);
                wd[3] = pack_u4_lo8_iq3_s4(s_w.z >> 4, s_w.w >> 4);
                const uint32_t ib = ib2 * 2u + ld_b;
                const uint32_t sc = (s_sv.x >> ((ib & 7u) * 4u)) & 0xFu;
                sm.sc[ld_r * 2u + ld_b] = __uint_as_float(s_sv.y) * (0.5f + static_cast<float>(sc)) * 0.5f;
            }
            __syncthreads();
            if (ib2 + 1u < 8u) {
                const uint32_t nxt = ib2 + 1u;
                s_w = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(r_wq, vw0, nxt * 32u, 0));
                if (nxt == 4u) s_sv = __builtin_bit_cast(uint2, __builtin_amdgcn_raw_buffer_load_b64(r_ws, vs0, 8u, 0));
            }

            // Wave-balanced 2-pass MACs: ALL 8 waves active on tokens 0..15; pass 2 (16..31) skipped when tcnt <= 16!
            auto run_dn4_pair = [&](uint32_t t_a, float& a00, float& a01, float& a10, float& a11,
                                    float& a20, float& a21, float& a30, float& a31) {
                const uint32_t t_b = t_a + 1u;
                #pragma nounroll
                for (uint32_t b = 0u; b < 2u; ++b) {
                    const uint4 xa = *reinterpret_cast<const uint4*>(&sm.x[t_a * DN4_XS + b * 4u]);
                    const uint4 xb = *reinterpret_cast<const uint4*>(&sm.x[t_b * DN4_XS + b * 4u]);
                    const float2 sa = sm.xds[t_a * 2u + b];
                    const float2 sb = sm.xds[t_b * 2u + b];
                    __builtin_amdgcn_sched_barrier(0);
                    auto do_row = [&](uint32_t rr, float& ra, float& rb) {
                        const uint2 w01 = *reinterpret_cast<const uint2*>(&sm.w[rr * DN4_WS + b * 4u + 0u]);
                        const uint2 w23 = *reinterpret_cast<const uint2*>(&sm.w[rr * DN4_WS + b * 4u + 2u]);
                        const int32_t da = sdot8(w23.y, xa.w, sdot8(w23.x, xa.z, sdot8(w01.y, xa.y, sdot8(w01.x, xa.x, 0))));
                        const int32_t db = sdot8(w23.y, xb.w, sdot8(w23.x, xb.z, sdot8(w01.y, xb.y, sdot8(w01.x, xb.x, 0))));
                        const float g = sm.sc[rr * 2u + b];
                        ra = fmaf(g, fmaf(sa.x, static_cast<float>(da), sa.y), ra);
                        rb = fmaf(g, fmaf(sb.x, static_cast<float>(db), sb.y), rb);
                    };
                    do_row(tx,       a00, a01);
                    do_row(tx + 32u, a10, a11);
                    do_row(tx + 64u, a20, a21);
                    do_row(tx + 96u, a30, a31);
                    __builtin_amdgcn_sched_barrier(0);
                }
            };
            if (ty * 2u < tcnt)
                run_dn4_pair(ty * 2u,      acc[0][0], acc[0][1], acc[1][0], acc[1][1], acc[2][0], acc[2][1], acc[3][0], acc[3][1]);
            if (16u + ty * 2u < tcnt)
                run_dn4_pair(16u + ty * 2u, acc[0][2], acc[0][3], acc[1][2], acc[1][3], acc[2][2], acc[2][3], acc[3][2], acc[3][3]);
            __syncthreads();
        }

        #pragma unroll
        for (uint32_t j = 0u; j < 4u; ++j) {
            const uint32_t tok = ((j >> 1) * 16u) + ty * 2u + (j & 1u);
            if (tok < tcnt) {
                const uint32_t r = sm.ms[tok];
                float* dst = part_out + static_cast<uint64_t>(r) * D + rt * 128u + tx;
                #pragma unroll
                for (uint32_t i = 0u; i < 4u; ++i) dst[32u * i] = acc[i][j];
            }
        }
        __syncthreads();
    }

    if (tid == 0u) {
        __threadfence();
        const uint32_t done = atomicAdd(&queue[16], 1u) + 1u;
        if (done == gridDim.x) {
            __atomic_store_n(&queue[0], 0u, __ATOMIC_RELEASE);
            __atomic_store_n(&queue[16], 0u, __ATOMIC_RELEASE);
        }
    }
}
