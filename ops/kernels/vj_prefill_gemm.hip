// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// VegaJet Big Lever 3: LDS-tiled W4A8 prefill GEMMs for gfx906 (v_dot4_i32_i8), lossless VJ_Q4_A8.
//
//   y[m][row] = sum_b  wd[row][b] * xd[m][b] * dp(row, m, b)  +  wm[row][b] * xs[m][b]
//   dp = exact INT32 dot of 32 u4 weights (0..15) with 32 INT8 activations of block b.
//
// Same kernargs, same output layout (FP32 y[M][rows]) as ornith_gemm_q4a8_tcc_kernel, so a dispatch swap is
// enough. The activations stay INT8 block-32 + float2{d, sum}; no FP32 activation buffer is read or created.
//
// Geometries (one template):
//   vj_prefill_gemm_q4a8_64x128_kernel : 64 weight rows x 128 tokens x K 64, 512 threads, 4x4 micro-tile
//   vj_prefill_gemm_q4a8_64x64_kernel  : 64 weight rows x  64 tokens x K 64, 256 threads, 4x4 micro-tile
// Why 4x4 and not 4x8 for the wide tile: the block-32 scales force an exact INT32 dp per (output, block) before
// the float update, so every output costs one float accumulator AND one INT32 accumulator. A 4x8 micro-tile is
// 32 + 32 = 64 VGPRs of accumulators alone (before operands, staging and addresses), which cannot meet the
// 64-VGPR gate without spilling. The 64x128 tile therefore uses 512 threads (still 4 waves/SIMD at 2 WGs/CU)
// with the same 4x4 micro-tile.
// Measured (therock-7.12, gfx906): both kernels 63 VGPRs, 0 spill, 4 waves/SIMD. The 64x64 target of <= 48
// VGPRs / 5 waves was not reachable: 16 float + 16 INT32 accumulators + 8 operand registers are 40 before
// addressing, and a no-deferral variant still spilled 23 VGPRs at a 48 cap, so it was dropped.
//
// Thread mapping: tx = tid & 15 -> weight rows tx + 16*i (i < 4); ty = tid >> 4 -> tokens ty + NTY*j (j < 4).
//   Weight LDS reads: 16 distinct rows per wave, row stride 68 B (= 17 dwords) -> 16 distinct banks.
//   Activation LDS reads: 4 distinct tokens per wave -> broadcast.
//   Stores: 16 consecutive rows of one token per 16 lanes (64 B segments).
// LDS (per WG): weights unpacked to INT8 64 x (64+4) B + activations BN x (64+4) B + float2 scale planes.
//   64x64 : 4,352 + 4,352 + 1,024 + 1,024 = 10,752 B
//   64x128: 4,352 + 8,704 + 1,024 + 2,048 = 16,128 B
// Pipeline: register-targeted global loads (plain cached: prefill weights are multi-hit), one stage of load
// deferral (tile k+1 is in flight in registers while tile k is computed), then ds_write into the padded rows.
// No direct-to-LDS loads, no 2-accumulator DP4A split, no inline-asm prefetch.
//
// Per K-tile (64) and per thread: 16 weight LDS dword reads + 16 activation LDS dword reads per block x 2 blocks
// = 64 ds_read_b32 for 256 v_dot4 (4x4 outputs x 16 dwords); 2 B of LDS read per v_dot4.
//
// Accumulation order vs ornith_gemm_q4a8_tcc_kernel: each block contributes exactly
//   acc = fmaf(wd * xd, (float)dp, fmaf(wm, xs, acc))
// (same integer dp, same expression, same rounding per block). The tcc kernel accumulates blocks b, b+32, b+64..
// in 32 lane-private partials and then sums the 32 partials with a xor butterfly; this kernel accumulates the
// blocks of a row sequentially b = 0, 1, 2, ... in one accumulator. Reproducing the butterfly tree would need a
// 5-deep partial stack per output (80 extra VGPRs for 16 outputs), so the two results agree to float rounding
// (a few ulp of the largest partial), not bit-for-bit. tests/test_prefill_ref.cpp measures the difference.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#include "core/ggml_iq_tables.hpp"
#include "ops/kernels/vj_moe_dot8.hiph"

namespace vegajet {
namespace pgemm {

__device__ __forceinline__ int32_t sdot4(uint32_t a, uint32_t b, int32_t c) {
    return __builtin_amdgcn_sdot4(static_cast<int32_t>(a), static_cast<int32_t>(b), c, false);
}
__device__ __forceinline__ float fp16_to_f32(uint16_t h) {
    return __half2float(__ushort_as_half(h));
}

constexpr uint32_t RS_W = 68u; // 64 INT8 + 4 B pad (17 dwords -> 0 bank conflicts on ds_read_b32 across tx=0..15)
constexpr uint32_t RS_X = 68u; // 64 INT8 + 4 B pad (17 dwords -> 0 bank conflicts across ty=0..3)

template <uint32_t BN, uint32_t NT>
struct Smem {
    uint32_t w[64u * (RS_W / 4u)];             // weights, INT8 0..15, row = weight row, 2 blocks per row
    uint32_t x[BN * (RS_X / 4u)];              // activations, row = token
    float2   wdm[64u * 2u];                    // {wd, wm} per (row, block)
    float2   xds[BN * 2u];                     // {xd, xs} per (token, block)
};

// Staging registers for one K-tile (9 VGPRs): 16 B of activations for every thread, plus either 16 B of packed
// weights + one dm word (threads 0..127) or one activation scale pair (threads 128 .. 128 + 2*BN - 1).
struct Stage {
    uint4    xa;
    uint4    wq;     // packed weights, or {xd, xs} bit patterns in .x/.y
    uint32_t dm;
};

// Global addressing: one buffer resource (SGPRs) per stream and ONE 32-bit VGPR offset per stream and thread;
// the per-K-tile advance is a scalar soffset. (64-bit pointers held across the MAC loop next to the staging
// registers spilled.) Plain buffer_load_dword{x2,x4} with glc=0 slc=0, LDS=0: register-targeted, cached.
struct Rsrc {
    __amdgpu_buffer_rsrc_t x, w, d, s;
};
struct Offs {
    uint32_t va;    // activation chunk byte offset (soffset += 64 per K-tile)
    uint32_t vb;    // role 0: weight row-block byte offset (+= 32); role 1: x_ds byte offset (+= 16)
    uint32_t vc;    // role 0: dm byte offset (+= 8)
    uint32_t role;  // 0: weights + dm, 1: activation scales, 2: activations only
};

__device__ __forceinline__ __amdgpu_buffer_rsrc_t make_rsrc(const void* p) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(p), static_cast<short>(0), 0x7FFFFFFF, 0x00027000);
}

template <uint32_t BN, uint32_t NT>
__device__ __forceinline__ Offs make_offs(uint32_t tid, uint32_t row0, uint32_t m0, uint32_t rows, uint32_t M,
                                          uint32_t cols, uint32_t nb, uint32_t stride) {
    Offs o;
    {
        const uint32_t t = tid >> 2, q = tid & 3u;
        const uint32_t m = min(m0 + t, M - 1u);                  // tail tokens re-read the last one (not stored)
        o.va = m * cols + q * 16u;
    }
    o.vb = 0u; o.vc = 0u;
    if (tid < 128u) {
        const uint32_t r = tid >> 1, b = tid & 1u;
        const uint32_t row = min(row0 + r, rows - 1u);           // tail rows re-read the last one (not stored)
        o.vb = row * stride + b * 16u;
        o.vc = (row * nb + b) * 4u;
        o.role = 0u;
    } else if (tid < 128u + 2u * BN) {
        const uint32_t u = tid - 128u;
        const uint32_t t = u >> 1, b = u & 1u;
        const uint32_t m = min(m0 + t, M - 1u);
        o.vb = (m * nb + b) * 8u;
        o.role = 1u;
    } else {
        o.role = 2u;
    }
    return o;
}

__device__ __forceinline__ void load_stage(Stage& st, const Rsrc& rs, const Offs& o, uint32_t kt) {
    st.xa = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(rs.x, o.va, kt * 64u, 0));
    if (o.role == 0u) {
        st.wq = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(rs.w, o.vb, kt * 32u, 0));
        st.dm = __builtin_amdgcn_raw_buffer_load_b32(rs.d, o.vc, kt * 8u, 0);
    } else if (o.role == 1u) {
        const uint2 v = __builtin_bit_cast(uint2, __builtin_amdgcn_raw_buffer_load_b64(rs.s, o.vb, kt * 16u, 0));
        st.wq.x = v.x; st.wq.y = v.y;
    }
}

template <uint32_t BN, uint32_t NT>
__device__ __forceinline__ void store_stage(const Stage& st, Smem<BN, NT>& sm, uint32_t tid, uint32_t role) {
    {
        const uint32_t t = tid >> 2, q = tid & 3u;
        uint32_t* d = sm.x + t * (RS_X / 4u) + q * 4u;
        d[0] = st.xa.x; d[1] = st.xa.y; d[2] = st.xa.z; d[3] = st.xa.w;
    }
    if (role == 0u) {
        const uint32_t r = tid >> 1, b = tid & 1u;
        // Low nibbles are elements 0..15 of the block, high nibbles 16..31 (VJ_Q4_A8 layout).
        uint32_t* d = sm.w + r * (RS_W / 4u) + b * 8u;
        d[0] = st.wq.x & 0x0F0F0F0Fu;         d[1] = st.wq.y & 0x0F0F0F0Fu;
        d[2] = st.wq.z & 0x0F0F0F0Fu;         d[3] = st.wq.w & 0x0F0F0F0Fu;
        d[4] = (st.wq.x >> 4) & 0x0F0F0F0Fu;  d[5] = (st.wq.y >> 4) & 0x0F0F0F0Fu;
        d[6] = (st.wq.z >> 4) & 0x0F0F0F0Fu;  d[7] = (st.wq.w >> 4) & 0x0F0F0F0Fu;
        sm.wdm[r * 2u + b] = make_float2(fp16_to_f32(static_cast<uint16_t>(st.dm & 0xFFFFu)),
                                         fp16_to_f32(static_cast<uint16_t>(st.dm >> 16)));
    } else if (role == 1u) {
        sm.xds[tid - 128u] = make_float2(__uint_as_float(st.wq.x), __uint_as_float(st.wq.y));
    }
}

template <uint32_t BN, uint32_t NT, bool DEFER>
__device__ __forceinline__ void gemm_q4a8_tile(
    const uint8_t*  __restrict__ w_qs,
    const uint32_t* __restrict__ w_dm,
    const int8_t*   __restrict__ x_q8,
    const float2*   __restrict__ x_ds,
    float*          __restrict__ y_out,
    uint32_t rows, uint32_t cols, uint32_t M, uint32_t stride,
    Smem<BN, NT>& sm) {
    constexpr uint32_t NTY = NT / 16u;           // token stride between a thread's 4 tokens (16 or 32)
    static_assert(NT == 4u * BN, "one 16-B activation chunk per thread");
    const uint32_t tid  = threadIdx.x;
    const uint32_t tx   = tid & 15u;
    const uint32_t ty   = tid >> 4;
    const uint32_t row0 = blockIdx.x * 64u;
    const uint32_t m0   = blockIdx.y * BN;
    if (row0 >= rows || m0 >= M) return;
    const uint32_t nb   = cols >> 5;
    const uint32_t nkt  = cols >> 6;              // K-tiles of 2 blocks (cols % 64 == 0)

    float acc[4][4];
    #pragma unroll
    for (uint32_t i = 0u; i < 4u; ++i) {
        #pragma unroll
        for (uint32_t j = 0u; j < 4u; ++j) acc[i][j] = 0.0f;
    }

    Rsrc rs;
    rs.x = make_rsrc(x_q8); rs.w = make_rsrc(w_qs); rs.d = make_rsrc(w_dm); rs.s = make_rsrc(x_ds);
    const Offs p = make_offs<BN, NT>(tid, row0, m0, rows, M, cols, nb, stride);
    Stage st;
    st.wq = make_uint4(0u, 0u, 0u, 0u); st.dm = 0u;
    if (DEFER) load_stage(st, rs, p, 0u);

    #pragma nounroll
    for (uint32_t kt = 0u; kt < nkt; ++kt) {
        if (!DEFER) load_stage(st, rs, p, kt);
        store_stage<BN, NT>(st, sm, tid, p.role);
        __syncthreads();
        if (DEFER && kt + 1u < nkt) load_stage(st, rs, p, kt + 1u);  // one stage of deferral: in flight during the MACs
        #pragma nounroll
        for (uint32_t b = 0u; b < 2u; ++b) {
            int32_t dp[4][4];
            #pragma unroll
            for (uint32_t i = 0u; i < 4u; ++i) {
                #pragma unroll
                for (uint32_t j = 0u; j < 4u; ++j) dp[i][j] = 0;
            }
            #pragma nounroll
            for (uint32_t kk = 0u; kk < 8u; ++kk) {
                uint32_t wv[4], xv[4];
                #pragma unroll
                for (uint32_t i = 0u; i < 4u; ++i) wv[i] = sm.w[(tx + 16u * i) * (RS_W / 4u) + b * 8u + kk];
                #pragma unroll
                for (uint32_t j = 0u; j < 4u; ++j) xv[j] = sm.x[(ty + NTY * j) * (RS_X / 4u) + b * 8u + kk];
                #pragma unroll
                for (uint32_t i = 0u; i < 4u; ++i) {
                    #pragma unroll
                    for (uint32_t j = 0u; j < 4u; ++j) dp[i][j] = sdot4(wv[i], xv[j], dp[i][j]);
                }
            }
            #pragma unroll
            for (uint32_t j = 0u; j < 4u; ++j) {
                const float2 xds = sm.xds[(ty + NTY * j) * 2u + b];
                #pragma unroll
                for (uint32_t i = 0u; i < 4u; ++i) {
                    const float2 wdm = sm.wdm[(tx + 16u * i) * 2u + b];
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

} // namespace pgemm
} // namespace vegajet

// ============================================================================
// Kernargs (identical to ornith_gemm_q4a8_tcc_kernel):
//   w_qs, w_dm, x_q8, x_ds, y_out, rows, cols, M, row_stride_bytes      (cols % 64 == 0: all Ornith dense
//   projections have cols in {512, 3072, 4096}; keep the tcc kernel for any cols % 64 == 32 shape)
// 64x128: grid x = ceil(rows/64) WGs, y = ceil(M/128); block 512.
//   enqueue_kernel(g, k, ceil(rows/64) * 512, ceil(M/128), 1, 512, 1, 1, ...)
// 64x64 : grid x = ceil(rows/64) WGs, y = ceil(M/64);  block 256.
//   enqueue_kernel(g, k, ceil(rows/64) * 256, ceil(M/64), 1, 256, 1, 1, ...)
// ============================================================================
extern "C" __global__ __launch_bounds__(512, 2)
void vj_prefill_gemm_q4a8_64x128_kernel(
    const uint8_t*  __restrict__ w_qs,
    const uint32_t* __restrict__ w_dm,
    const int8_t*   __restrict__ x_q8,
    const float2*   __restrict__ x_ds,
    float*          __restrict__ y_out,
    uint32_t rows,
    uint32_t cols,
    uint32_t M,
    uint32_t row_stride_bytes) {
    __shared__ vegajet::pgemm::Smem<128u, 512u> sm;
    vegajet::pgemm::gemm_q4a8_tile<128u, 512u, true>(w_qs, w_dm, x_q8, x_ds, y_out, rows, cols, M, row_stride_bytes, sm);
}

extern "C" __global__ __launch_bounds__(256, 4)
void vj_prefill_gemm_q4a8_64x64_kernel(
    const uint8_t*  __restrict__ w_qs,
    const uint32_t* __restrict__ w_dm,
    const int8_t*   __restrict__ x_q8,
    const float2*   __restrict__ x_ds,
    float*          __restrict__ y_out,
    uint32_t rows,
    uint32_t cols,
    uint32_t M,
    uint32_t row_stride_bytes) {
    __shared__ vegajet::pgemm::Smem<64u, 256u> sm;
    vegajet::pgemm::gemm_q4a8_tile<64u, 256u, true>(w_qs, w_dm, x_q8, x_ds, y_out, rows, cols, M, row_stride_bytes, sm);
}

// ============================================================================
// MoE Gate+Up for unpromoted IQ2_S experts (merged 123-uint4 row layout), expert-grouped, dequant-once tiles.
//
// Work item = (expert e, token tile of up to 16 routed (token, slot) pairs, row tile of 64 local_ff rows).
// For each K step (one 32-element ib block) the WG decodes the 64-row Gate tile AND the 64-row Up tile ONCE
// into padded INT8 LDS (grid lookups + SWAR signs paid once per tile; wave w decodes 16-element group w of all
// 64 rows, so the decode shifts are scalar),
// then every thread runs v_dot4 for 4 rows x 1 token x {gate, up} against the gathered activations.
// Persistent work-stealing queue: WGs pop items from queue[0]; the last WG to drain the queue (queue[16])
// rearms both counters for the next dispatch. There is no grid barrier, so residency is not required for
// correctness; LDS 16,256 B and <= 64 VGPRs keep 4 WGs/CU resident (grid 240 = 4 x 60 recommended).
//
// Output (identical layout to ornith_moe_gu_iq2s_grouped_kernel): mid_q8[ms][local_ff] INT8 and
// mid_d[ms][local_ff/32] float, y = silu(g) * u, block-32 quantised along local_ff.
// Float order per ib matches the grouped kernel's per-ib expression:
//   acc = fmaf(gd0 * xd, dp0, fmaf(gd1 * xd, dp1, acc))
// but ibs are accumulated sequentially (0..95) instead of 16 lane partials + xor butterfly (same caveat as the
// dense kernels: equal to float rounding, not bit-for-bit).
// Per ib and thread: 3 global loads (q dword, hs, d) for the decode, 2 grid LDS lookups (uint64), 64 weight
// + 8 activation LDS dword reads for 64 v_dot4 (4 rows x 1 token x 2 halves x 2 mats x 4 dwords). A token tile
// of 32 (4 rows x 2 tokens) needed 16 float + 16 INT32 accumulators and spilled 20 VGPRs.
// ============================================================================
namespace vegajet {
namespace pgemm {

__device__ __forceinline__ uint32_t iq2s_perm4(uint32_t mag4_sel, uint32_t sign4) {
    const uint32_t sel = (mag4_sel & 0x03030303u) | ((sign4 * 0x00810204u) & 0x04040404u);
    return __builtin_amdgcn_perm(0x00D5E7F8u, 0x002B1908u, sel);
}

constexpr uint32_t MOE_T  = 32u;   // token tile (covers 95%+ experts in 1 decode pass at M=512; 4 tokens/thread!)
constexpr uint32_t MOE_WS = 10u;   // 8B-aligned padded LDS stride for w (enables ds_read_b64 + low bank conflict)
constexpr uint32_t MOE_XS = 8u;    // 8B-aligned LDS stride for x (broadcast across tx=0..31 -> 0 bank conflicts)

struct MoeSmem {
    uint32_t grid32[1024];                    // 4,096 B  compact 2-bit iq2s_grid32 (1-bank ds_read_b32 + v_perm_b32!)
    alignas(8) uint32_t w[2u * 64u * MOE_WS]; // 5,120 B  gate / up INT8 tiles (8B aligned)
    float    sc[64u * 4u];                    // 1,024 B  gd0, gd1, ud0, ud1 per row
    alignas(16) uint32_t x[MOE_T * MOE_XS];   // 1,024 B  activations (16B aligned -> ds_read_b128)
    float    xd[MOE_T];                       //   128 B
    uint16_t pref[257];                       //   514 B  exclusive prefix of work items per expert
    uint16_t pad_pref;                        //     2 B
    uint32_t ms[MOE_T];                       //   128 B
    uint32_t item;                            //     4 B  -> total 12,040 B (5 WGs/CU LDS budget!)
};

} // namespace pgemm
} // namespace vegajet

// Kernargs: wgu_merged, x_q8 [M][D], x_ds [M][D/32], exp_off[257], exp_cnt[256], sorted_ms[M*8],
//           mid_q8 [M*8][local_ff], mid_d [M*8][local_ff/32], queue (uint32[32], zero before first use),
//           local_ff (multiple of 64), D (multiple of 32)
// Grid: any number of WGs (recommended 240), block 256.
extern "C" __global__ __launch_bounds__(256, 4)
void vj_prefill_moe_gu_iq2s_tiled_kernel(
    const uint4*    __restrict__ wgu_merged,
    const int8_t*   __restrict__ x_q8,
    const float2*   __restrict__ x_ds,
    const uint32_t* __restrict__ exp_off,
    const uint32_t* __restrict__ exp_cnt,
    const uint32_t* __restrict__ sorted_ms,
    int8_t*         __restrict__ mid_q8,
    float*          __restrict__ mid_d,
    uint32_t*       __restrict__ queue,
    uint32_t local_ff,
    uint32_t D) {
    using namespace vegajet::pgemm;
    __shared__ MoeSmem sm;
    const uint32_t tid = threadIdx.x;
    const uint32_t tx  = tid & 31u;         // 32 row pairs -> 64 rows (2 rows/thread = 2 full block-32 quant blocks)
    const uint32_t ty  = tid >> 5;          // 8 token pairs -> 16 tokens (2 tokens/thread -> 2x2 micro-tile!)
    const uint32_t RT  = local_ff >> 6;     // row tiles per expert
    const uint32_t nib = D >> 5;
    const __amdgpu_buffer_rsrc_t r_w = make_rsrc(wgu_merged);
    const __amdgpu_buffer_rsrc_t r_x = make_rsrc(x_q8);
    const __amdgpu_buffer_rsrc_t r_s = make_rsrc(x_ds);

    reinterpret_cast<uint4*>(sm.grid32)[tid] = reinterpret_cast<const uint4*>(vegajet::ggml_tables::iq2s_grid32)[tid];
    {
        // Exclusive prefix of items per expert (Hillis-Steele over 256 experts).
        const uint32_t c = exp_cnt[tid];
        uint32_t v = ((c + MOE_T - 1u) / MOE_T) * RT;
        sm.pref[tid + 1u] = v;
        __syncthreads();
        #pragma unroll
        for (uint32_t step = 1u; step < 256u; step <<= 1) {
            const uint32_t add = (tid >= step) ? sm.pref[tid + 1u - step] : 0u;
            __syncthreads();
            v += add;
            sm.pref[tid + 1u] = v;
            __syncthreads();
        }
        if (tid == 0u) sm.pref[0] = 0u;
        __syncthreads();
    }
    const uint32_t total = __builtin_amdgcn_readfirstlane(sm.pref[256]);

    // Decode role: row dr = tid & 63, part = wave (0: gate elems 0..15, 1: gate 16..31, 2: up 0..15, 3: up 16..31).
    // part is wave-uniform, so every decode shift / select is scalar.
    const uint32_t dr   = tid & 63u;
    const uint32_t part = __builtin_amdgcn_readfirstlane(tid >> 6);
    // Activation role (tid < 64): token tid >> 1, half tid & 1 (16 B each); tid 64..95: xd of token tid - 64.

    #pragma nounroll
    for (;;) {
        if (tid == 0u) sm.item = atomicAdd(&queue[0], 1u);
        __syncthreads();
        const uint32_t item = __builtin_amdgcn_readfirstlane(sm.item);   // uniform -> SGPR
        if (item >= total) break;

        // Expert = last e with pref[e] <= item (binary search, wave-uniform).
        uint32_t lo = 0u, hi = 256u;
        while (hi - lo > 1u) {
            const uint32_t mid = (lo + hi) >> 1;
            if (__builtin_amdgcn_readfirstlane(sm.pref[mid]) <= item) lo = mid; else hi = mid;
        }
        const uint32_t e     = lo;
        const uint32_t local = item - __builtin_amdgcn_readfirstlane(sm.pref[e]);
        const uint32_t tt    = local / RT;
        const uint32_t rt    = local - tt * RT;
        const uint32_t cnt   = __builtin_amdgcn_readfirstlane(exp_cnt[e]);
        const uint32_t off   = __builtin_amdgcn_readfirstlane(exp_off[e]);
        const uint32_t t0    = tt * MOE_T;
        const uint32_t tcnt  = min(MOE_T, cnt - t0);
        if (tid < MOE_T) sm.ms[tid] = sorted_ms[off + t0 + min(tid, tcnt - 1u)];
        __syncthreads();

        // Buffer resources + one 32-bit VGPR offset per stream; the ib advance is a scalar soffset.
        const uint32_t erow  = e * local_ff + rt * 64u + dr;
        const uint32_t vbase = erow * 1968u;                       // 123 x 16 B per merged row
        const uint32_t vq    = vbase + part * 4u;                  // q.{x,y,z,w} of each ib at + ib*16
        const uint32_t role  = (tid < 64u) ? 0u : ((tid < 96u) ? 1u : 2u);
        uint32_t vx = 0u;
        if (role == 0u) vx = (sm.ms[tid >> 1] >> 3) * D + (tid & 1u) * 16u;
        else if (role == 1u) vx = (sm.ms[tid - 64u] >> 3) * nib * 8u;

        float ag[2][4], au[2][4];
        #pragma unroll
        for (uint32_t i = 0u; i < 2u; ++i) {
            #pragma unroll
            for (uint32_t j = 0u; j < 4u; ++j) { ag[i][j] = 0.0f; au[i][j] = 0.0f; }
        }

        // Stage registers (one ib ahead).
        uint32_t s_q  = __builtin_amdgcn_raw_buffer_load_b32(r_w, vq, 0u, 0);
        uint32_t s_hs = __builtin_amdgcn_raw_buffer_load_b32(r_w, vbase + 1536u, 0u, 0);
        uint32_t s_d  = __builtin_amdgcn_raw_buffer_load_b32(r_w, vbase + 1920u, 0u, 0);
        uint4 s_x = make_uint4(0u, 0u, 0u, 0u);
        if (role == 0u) s_x = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(r_x, vx, 0u, 0));
        else if (role == 1u) s_x.x = __builtin_amdgcn_raw_buffer_load_b32(r_s, vx, 0u, 0);   // xd bits

        #pragma nounroll
        for (uint32_t ib = 0u; ib < nib; ++ib) {
            // ---- decode one 16-element group into LDS (dequant once per tile) ----
            {
                const uint32_t mat  = part >> 1;                 // 0 gate, 1 up
                const uint32_t half = part & 1u;
                const uint32_t qh   = (s_hs >> (16u * mat)) & 0xFFu;
                const uint32_t scb  = (s_hs >> (16u * mat + 8u)) & 0xFFu;
                const float    dd   = fp16_to_f32(static_cast<uint16_t>(s_d >> (16u * mat))) * 0.25f;
                const uint32_t sh0  = half ? 4u : 8u;            // qh bit offsets for the two grid entries
                const uint32_t g0   = sm.grid32[(s_q & 0xFFu)         | ((qh << sh0) & 0x300u)];
                const uint32_t g1   = sm.grid32[((s_q >> 16) & 0xFFu) | ((qh << (sh0 - 2u)) & 0x300u)];
                uint32_t* wd = sm.w + (mat * 64u + dr) * MOE_WS + half * 4u;
                wd[0] = iq2s_perm4(g0,      (s_q >> 8)  & 0xFu);
                wd[1] = iq2s_perm4(g0 >> 2, (s_q >> 12) & 0xFu);
                wd[2] = iq2s_perm4(g1,      (s_q >> 24) & 0xFu);
                wd[3] = iq2s_perm4(g1 >> 2,  s_q >> 28);
                sm.sc[dr * 4u + part] = dd * (0.5f + static_cast<float>(half ? (scb >> 4) : (scb & 0xFu)));
                if (role == 0u) {
                    *reinterpret_cast<uint4*>(sm.x + (tid >> 1) * MOE_XS + (tid & 1u) * 4u) = s_x;
                } else if (role == 1u) {
                    sm.xd[tid - 64u] = __uint_as_float(s_x.x);
                }
            }
            __syncthreads();
            if (ib + 1u < nib) {
                const uint32_t in = ib + 1u;
                s_q  = __builtin_amdgcn_raw_buffer_load_b32(r_w, vq, in * 16u, 0);
                s_hs = __builtin_amdgcn_raw_buffer_load_b32(r_w, vbase + 1536u, in * 4u, 0);
                s_d  = __builtin_amdgcn_raw_buffer_load_b32(r_w, vbase + 1920u, (in >> 3) * 4u, 0);
                if (role == 0u) s_x = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(r_x, vx, in * 32u, 0));
                else if (role == 1u) s_x.x = __builtin_amdgcn_raw_buffer_load_b32(r_s, vx, in * 8u, 0);
            }
            // ---- Wave-balanced 2x2 micro-tile MACs (ALL 8 half-waves active on tokens 0..15; pass 2 skipped when tcnt <= 16!) ----
            #pragma nounroll
            for (uint32_t hh = 0u; hh < 2u; ++hh) {
                const uint32_t h = 1u - hh;
                auto run_pair = [&](uint32_t t_a, float& ag00, float& ag01, float& ag10, float& ag11,
                                    float& au00, float& au01, float& au10, float& au11) {
                    const uint32_t t_b = t_a + 1u;
                    const float xda = sm.xd[t_a];
                    const float xdb = sm.xd[t_b];
                    const uint4 xa = *reinterpret_cast<const uint4*>(&sm.x[t_a * MOE_XS + h * 4u]);
                    const uint4 xb = *reinterpret_cast<const uint4*>(&sm.x[t_b * MOE_XS + h * 4u]);
                    __builtin_amdgcn_sched_barrier(0);
                    auto do_row = [&](uint32_t rr, float& g0, float& g1, float& u0, float& u1) {
                        uint2 wg = *reinterpret_cast<const uint2*>(&sm.w[rr * MOE_WS + h * 4u]);
                        int32_t d0 = sdot4(wg.x, xa.x, 0);
                        int32_t d1 = sdot4(wg.x, xb.x, 0);
                        d0 = sdot4(wg.y, xa.y, d0);  d1 = sdot4(wg.y, xb.y, d1);
                        wg = *reinterpret_cast<const uint2*>(&sm.w[rr * MOE_WS + h * 4u + 2u]);
                        d0 = sdot4(wg.x, xa.z, d0);  d1 = sdot4(wg.x, xb.z, d1);
                        d0 = sdot4(wg.y, xa.w, d0);  d1 = sdot4(wg.y, xb.w, d1);
                        const float gs = sm.sc[rr * 4u + h];
                        g0 = fmaf(gs * xda, static_cast<float>(d0), g0);
                        g1 = fmaf(gs * xdb, static_cast<float>(d1), g1);

                        uint2 wu = *reinterpret_cast<const uint2*>(&sm.w[(64u + rr) * MOE_WS + h * 4u]);
                        d0 = sdot4(wu.x, xa.x, 0);
                        d1 = sdot4(wu.x, xb.x, 0);
                        d0 = sdot4(wu.y, xa.y, d0);  d1 = sdot4(wu.y, xb.y, d1);
                        wu = *reinterpret_cast<const uint2*>(&sm.w[(64u + rr) * MOE_WS + h * 4u + 2u]);
                        d0 = sdot4(wu.x, xa.z, d0);  d1 = sdot4(wu.x, xb.z, d1);
                        d0 = sdot4(wu.y, xa.w, d0);  d1 = sdot4(wu.y, xb.w, d1);
                        const float us = sm.sc[rr * 4u + 2u + h];
                        u0 = fmaf(us * xda, static_cast<float>(d0), u0);
                        u1 = fmaf(us * xdb, static_cast<float>(d1), u1);
                    };
                    do_row(tx,       ag00, ag01, au00, au01);
                    do_row(tx + 32u, ag10, ag11, au10, au11);
                    __builtin_amdgcn_sched_barrier(0);
                };
                if (ty * 2u < tcnt)
                    run_pair(ty * 2u,       ag[0][0], ag[0][1], ag[1][0], ag[1][1], au[0][0], au[0][1], au[1][0], au[1][1]);
                if (16u + ty * 2u < tcnt)
                    run_pair(16u + ty * 2u, ag[0][2], ag[0][3], ag[1][2], ag[1][3], au[0][2], au[0][3], au[1][2], au[1][3]);
            }
            __syncthreads();
        }

        // ---- SwiGLU + block-32 quant along local_ff: tx=0..31 IS one full 32-row block (i=0 -> blk0, i=1 -> blk1) ----
        #pragma unroll
        for (uint32_t j = 0u; j < 4u; ++j) {
            const uint32_t tok = ((j >> 1) * 16u) + ty * 2u + (j & 1u);
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
                if (tok < tcnt) {
                    const uint32_t ms = sm.ms[tok];
                    mid_q8[static_cast<uint64_t>(ms) * local_ff + rt * 64u + i * 32u + tx] =
                        static_cast<int8_t>(max(-127, min(127, __float2int_rn(y * inv))));
                    if (tx == 0u) mid_d[static_cast<uint64_t>(ms) * (local_ff >> 5) + rt * 2u + i] = d;
                }
            }
        }
        __syncthreads();   // sm.ms / sm.item reuse
    }

    // Drain: the last WG out rearms the queue for the next dispatch.
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
// MoE Gate+Up for exact VJ_Q4_DOT8 (IQ2_S-sourced), LDS-tiled 64 rows x 32 tokens.
// Zero IQ2_S grid LUT in LDS (7,944 B total -> 8 WGs/CU!), dec4_iq2 register decode,
// supports any local_ff multiple of 32 (320 / 352 / 384 / 512) for TP=2/3/4!
// ============================================================================
namespace vegajet {
namespace pgemm {

struct MoeDot8Smem {
    alignas(8) uint32_t w[2u * 64u * MOE_WS]; // 5,120 B  gate / up INT8 tiles (8B aligned)
    float    sc[64u * 4u];                    // 1,024 B  gd0, gd1, ud0, ud1 per row
    alignas(16) uint32_t x[MOE_T * MOE_XS];   // 1,024 B  activations (16B aligned -> ds_read_b128)
    float    xd[MOE_T];                       //   128 B
    uint16_t pref[257];                       //   514 B  exclusive prefix of work items per expert
    uint16_t pad_pref;                        //     2 B
    uint32_t ms[MOE_T];                       //   128 B
    uint32_t item;                            //     4 B  -> total 7,944 B <= 8,192 B (8 WGs/CU!)
};

} // namespace pgemm
} // namespace vegajet

extern "C" __global__ __launch_bounds__(256, 4)
void vj_prefill_moe_gu_dot8_tiled_kernel(
    const uint8_t*  __restrict__ w_gu,
    const uint8_t*  __restrict__ w_sc,
    const int8_t*   __restrict__ x_q8,
    const float2*   __restrict__ x_ds,
    const uint32_t* __restrict__ exp_off,
    const uint32_t* __restrict__ exp_cnt,
    const uint32_t* __restrict__ sorted_ms,
    int8_t*         __restrict__ mid_q8,
    float*          __restrict__ mid_d,
    uint32_t*       __restrict__ queue,
    uint32_t local_ff,
    uint32_t D,
    uint32_t w_stride,
    uint32_t s_stride) {
    using namespace vegajet::pgemm;
    using vegajet::dot8_dev::dec4_iq2;
    __shared__ MoeDot8Smem sm;
    const uint32_t tid = threadIdx.x;
    const uint32_t tx  = tid & 31u;
    const uint32_t ty  = tid >> 5;
    const uint32_t RT  = (local_ff + 63u) >> 6;
    const uint32_t nib = D >> 5;
    const __amdgpu_buffer_rsrc_t r_w  = make_rsrc(w_gu);
    const __amdgpu_buffer_rsrc_t r_ws = make_rsrc(w_sc);
    const __amdgpu_buffer_rsrc_t r_x  = make_rsrc(x_q8);
    const __amdgpu_buffer_rsrc_t r_xs = make_rsrc(x_ds);

    {
        const uint32_t c = exp_cnt[tid];
        uint32_t v = ((c + MOE_T - 1u) / MOE_T) * RT;
        sm.pref[tid + 1u] = static_cast<uint16_t>(v);
        __syncthreads();
        #pragma unroll
        for (uint32_t step = 1u; step < 256u; step <<= 1) {
            const uint32_t add = (tid >= step) ? sm.pref[tid + 1u - step] : 0u;
            __syncthreads();
            v += add;
            sm.pref[tid + 1u] = static_cast<uint16_t>(v);
            __syncthreads();
        }
        if (tid == 0u) sm.pref[0] = 0u;
        __syncthreads();
    }
    const uint32_t total = __builtin_amdgcn_readfirstlane(sm.pref[256]);
    const uint32_t role  = (tid < 128u) ? 0u : ((tid < 192u) ? 1u : ((tid < 224u) ? 2u : 3u));
    const uint32_t dr    = tid & 63u;
    const uint32_t mat   = __builtin_amdgcn_readfirstlane((tid >> 6) & 1u);

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
        const uint32_t e     = lo;
        const uint32_t local = item - __builtin_amdgcn_readfirstlane(sm.pref[e]);
        const uint32_t tt    = local / RT;
        const uint32_t rt    = local - tt * RT;
        const uint32_t cnt   = __builtin_amdgcn_readfirstlane(exp_cnt[e]);
        const uint32_t off   = __builtin_amdgcn_readfirstlane(exp_off[e]);
        const uint32_t t0    = tt * MOE_T;
        const uint32_t tcnt  = min(MOE_T, cnt - t0);
        if (tid < MOE_T) sm.ms[tid] = sorted_ms[off + t0 + min(tid, tcnt - 1u)];
        __syncthreads();

        uint32_t v0 = 0u, v1 = 0u, v2 = 0u;
        if (role == 0u) {
            const uint32_t r_in_e = min(rt * 64u + dr, local_ff - 1u);
            const uint32_t erow   = e * local_ff + r_in_e;
            v0 = erow * w_stride + mat * 16u;
            v1 = erow * s_stride;
            v2 = v1 + nib * 2u;
        } else if (role == 1u) {
            const uint32_t u = tid - 128u;
            v0 = (sm.ms[u >> 1] >> 3) * D + (u & 1u) * 16u;
        } else if (role == 2u) {
            v0 = (sm.ms[tid - 192u] >> 3) * nib * 8u;
        }

        float ag[2][4], au[2][4];
        #pragma unroll
        for (uint32_t i = 0u; i < 2u; ++i) {
            #pragma unroll
            for (uint32_t j = 0u; j < 4u; ++j) { ag[i][j] = 0.0f; au[i][j] = 0.0f; }
        }

        uint4    s_q  = make_uint4(0u, 0u, 0u, 0u);
        uint32_t s_sc = 0u, s_d = 0u;
        if (role == 0u) {
            s_q  = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(r_w, v0, 0u, 0));
            s_sc = __builtin_amdgcn_raw_buffer_load_b32(r_ws, v1, 0u, 0);
            s_d  = __builtin_amdgcn_raw_buffer_load_b32(r_ws, v2, 0u, 0);
        } else if (role == 1u) {
            s_q = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(r_x, v0, 0u, 0));
        } else if (role == 2u) {
            s_q.x = __builtin_amdgcn_raw_buffer_load_b32(r_xs, v0, 0u, 0);
        }

        #pragma nounroll
        for (uint32_t ib = 0u; ib < nib; ++ib) {
            if (role == 0u) {
                uint32_t* wd = sm.w + (mat * 64u + dr) * MOE_WS;
                wd[0] = dec4_iq2(s_q.x);       wd[1] = dec4_iq2(s_q.y);
                wd[2] = dec4_iq2(s_q.z);       wd[3] = dec4_iq2(s_q.w);
                wd[4] = dec4_iq2(s_q.x >> 4);  wd[5] = dec4_iq2(s_q.y >> 4);
                wd[6] = dec4_iq2(s_q.z >> 4);  wd[7] = dec4_iq2(s_q.w >> 4);
                const uint32_t scb = (s_sc >> ((ib & 1u) * 16u + mat * 8u)) & 0xFFu;
                const float    dd  = fp16_to_f32(static_cast<uint16_t>(s_d >> (mat * 16u))) * 0.25f;
                sm.sc[dr * 4u + mat * 2u + 0u] = dd * (0.5f + static_cast<float>(scb & 0xFu));
                sm.sc[dr * 4u + mat * 2u + 1u] = dd * (0.5f + static_cast<float>(scb >> 4));
            } else if (role == 1u) {
                const uint32_t u = tid - 128u;
                *reinterpret_cast<uint4*>(sm.x + (u >> 1) * MOE_XS + (u & 1u) * 4u) = s_q;
            } else if (role == 2u) {
                sm.xd[tid - 192u] = __uint_as_float(s_q.x);
            }
            __syncthreads();
            if (ib + 1u < nib) {
                const uint32_t in = ib + 1u;
                if (role == 0u) {
                    s_q = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(r_w, v0, in * 32u, 0));
                    if ((in & 1u) == 0u) s_sc = __builtin_amdgcn_raw_buffer_load_b32(r_ws, v1, in * 2u, 0);
                    if ((in & 7u) == 0u) s_d  = __builtin_amdgcn_raw_buffer_load_b32(r_ws, v2, (in >> 3) * 4u, 0);
                } else if (role == 1u) {
                    s_q = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(r_x, v0, in * 32u, 0));
                } else if (role == 2u) {
                    s_q.x = __builtin_amdgcn_raw_buffer_load_b32(r_xs, v0, in * 8u, 0);
                }
            }
            #pragma nounroll
            for (uint32_t hh = 0u; hh < 2u; ++hh) {
                const uint32_t h = 1u - hh;
                auto run_pair = [&](uint32_t t_a, float& ag00, float& ag01, float& ag10, float& ag11,
                                    float& au00, float& au01, float& au10, float& au11) {
                    const uint32_t t_b = t_a + 1u;
                    const float xda = sm.xd[t_a];
                    const float xdb = sm.xd[t_b];
                    const uint4 xa = *reinterpret_cast<const uint4*>(&sm.x[t_a * MOE_XS + h * 4u]);
                    const uint4 xb = *reinterpret_cast<const uint4*>(&sm.x[t_b * MOE_XS + h * 4u]);
                    __builtin_amdgcn_sched_barrier(0);
                    auto do_row = [&](uint32_t rr, float& g0, float& g1, float& u0, float& u1) {
                        uint2 wg = *reinterpret_cast<const uint2*>(&sm.w[rr * MOE_WS + h * 4u]);
                        int32_t d0 = sdot4(wg.x, xa.x, 0);
                        int32_t d1 = sdot4(wg.x, xb.x, 0);
                        d0 = sdot4(wg.y, xa.y, d0);  d1 = sdot4(wg.y, xb.y, d1);
                        wg = *reinterpret_cast<const uint2*>(&sm.w[rr * MOE_WS + h * 4u + 2u]);
                        d0 = sdot4(wg.x, xa.z, d0);  d1 = sdot4(wg.x, xb.z, d1);
                        d0 = sdot4(wg.y, xa.w, d0);  d1 = sdot4(wg.y, xb.w, d1);
                        const float gs = sm.sc[rr * 4u + h];
                        g0 = fmaf(gs * xda, static_cast<float>(d0), g0);
                        g1 = fmaf(gs * xdb, static_cast<float>(d1), g1);

                        uint2 wu = *reinterpret_cast<const uint2*>(&sm.w[(64u + rr) * MOE_WS + h * 4u]);
                        d0 = sdot4(wu.x, xa.x, 0);
                        d1 = sdot4(wu.x, xb.x, 0);
                        d0 = sdot4(wu.y, xa.y, d0);  d1 = sdot4(wu.y, xb.y, d1);
                        wu = *reinterpret_cast<const uint2*>(&sm.w[(64u + rr) * MOE_WS + h * 4u + 2u]);
                        d0 = sdot4(wu.x, xa.z, d0);  d1 = sdot4(wu.x, xb.z, d1);
                        d0 = sdot4(wu.y, xa.w, d0);  d1 = sdot4(wu.y, xb.w, d1);
                        const float us = sm.sc[rr * 4u + 2u + h];
                        u0 = fmaf(us * xda, static_cast<float>(d0), u0);
                        u1 = fmaf(us * xdb, static_cast<float>(d1), u1);
                    };
                    do_row(tx,       ag00, ag01, au00, au01);
                    do_row(tx + 32u, ag10, ag11, au10, au11);
                    __builtin_amdgcn_sched_barrier(0);
                };
                if (ty * 2u < tcnt)
                    run_pair(ty * 2u,       ag[0][0], ag[0][1], ag[1][0], ag[1][1], au[0][0], au[0][1], au[1][0], au[1][1]);
                if (16u + ty * 2u < tcnt)
                    run_pair(16u + ty * 2u, ag[0][2], ag[0][3], ag[1][2], ag[1][3], au[0][2], au[0][3], au[1][2], au[1][3]);
            }
            __syncthreads();
        }

        #pragma unroll
        for (uint32_t j = 0u; j < 4u; ++j) {
            const uint32_t tok = ((j >> 1) * 16u) + ty * 2u + (j & 1u);
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
                const uint32_t out_r = rt * 64u + i * 32u + tx;
                if (tok < tcnt && out_r < local_ff) {
                    const uint32_t ms = sm.ms[tok];
                    mid_q8[static_cast<uint64_t>(ms) * local_ff + out_r] =
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
// MoE Down for VJ_Q4_DOT8 (IQ3_XXS-sourced, K = 512 = 16 blocks of 32), LDS-tiled 128 rows x 16 tokens.
//
// Eliminates the 309 GB/req dp_out HBM2 round-trip of vj_moe_down_grouped_dp_kernel AND the 32-row uncoalesced
// stores of vj_moe_down_grouped_f32_kernel:
//   * Work item = (expert e, up to 16 routes, 128-row tile of D). Persistent work-stealing queue (240 WGs).
//   * Per ib (0..15): 128 threads decode 128 rows x 32 weights into LDS via 5-op bitwise dec4_iq3 (0 grid lookups),
//     32 threads stage 16 routes x 32 INT8 activations, 16 threads stage ew * xd.
//   * 4x2 micro-tile (tx = 0..31, ty = 0..7): 4 rows x 2 tokens = 8 float accumulators, ds_read_b128 on x,
//     ds_read_b64 on w -> 64 v_dot4 per thread per ib (4:1 compute:LDS ratio), 100% coalesced 128 B stores to
//     part_out[route * D + row]. LDS = 7,368 B (4 WGs/CU).
//   * Paired with vj_moe_down_combine_all_f32_kernel (12 x M WGs, coalesced 8-slot sum + sh_down).
// ============================================================================
namespace vegajet {
namespace pgemm {

constexpr uint32_t DN_T  = 32u;
constexpr uint32_t DN_WS = 10u; // 8B-aligned (40 B/row -> 0 bank conflicts on ds_read_b64 across tx=0..15)
constexpr uint32_t DN_XS = 8u;  // 16B-aligned (32 B/tok -> ds_read_b128, broadcast across tx=0..31)

struct DownSmem {
    alignas(8)  uint32_t w[128u * DN_WS];   // 5,120 B  decoded INT8 weights (128 rows x 32 elems)
    alignas(16) uint32_t x[DN_T * DN_XS];   // 1,024 B  INT8 activations (32 tokens x 32 elems)
    float    sc[128u];                      //   512 B  gdb scale per row for current ib
    float    xew[DN_T];                     //   128 B  ew[t] * xd[t]
    float    ew[DN_T];                      //   128 B  top_w[r]
    uint32_t ms[DN_T];                      //   128 B  route r = m*8+s
    uint16_t pref[257];                     //   514 B  exclusive prefix of items per expert
    uint16_t pad_pref;                      //     2 B
    uint8_t  ord[256];                      //   256 B  heaviest-first 2-tier expert order (cnt >= 16 first)
    uint32_t item;                          //     4 B  -> total 7,816 B <= 8,192 B (4 WGs/CU)
};

} // namespace pgemm
} // namespace vegajet

extern "C" __global__ __launch_bounds__(256, 4)
void vj_prefill_moe_down_iq3_tiled_kernel(
    const uint8_t*  __restrict__ w_qs,
    const uint8_t*  __restrict__ w_sc,
    const int8_t*   __restrict__ mid_q8,
    const float*    __restrict__ mid_d,
    const float*    __restrict__ top_w,
    const uint32_t* __restrict__ exp_off,
    const uint32_t* __restrict__ exp_cnt,
    const uint32_t* __restrict__ sorted_ms,
    float*          __restrict__ part_out,
    uint32_t*       __restrict__ queue,
    uint32_t D,
    uint32_t w_stride,
    uint32_t s_stride,
    uint32_t local_ff,
    uint32_t is_sub32) {
    using namespace vegajet::pgemm;
    using vegajet::dot8_dev::dec4_iq3;
    __shared__ DownSmem sm;
    const uint32_t tid = threadIdx.x;
    const uint32_t tx  = tid & 31u;         // 32 lanes x 4 rows = 128 rows
    const uint32_t ty  = tid >> 5;          // 8 pairs x 2 tokens = 16 tokens
    const uint32_t RT  = D >> 7;            // D / 128 row tiles per expert
    const uint32_t nib = __builtin_amdgcn_readfirstlane(local_ff >> 5);
    const bool sub32_u = (__builtin_amdgcn_readfirstlane(is_sub32) != 0u);
    const __amdgpu_buffer_rsrc_t r_wq = make_rsrc(w_qs);
    const __amdgpu_buffer_rsrc_t r_ws = make_rsrc(w_sc);
    const __amdgpu_buffer_rsrc_t r_mq = make_rsrc(mid_q8);
    const __amdgpu_buffer_rsrc_t r_md = make_rsrc(mid_d);

    {
        const uint32_t c0 = exp_cnt[tid];
        const uint32_t is_h = (c0 >= 16u) ? 1u : 0u;
        uint32_t vh = is_h;
        sm.pref[tid + 1u] = static_cast<uint16_t>(vh);
        __syncthreads();
        #pragma unroll
        for (uint32_t step = 1u; step < 256u; step <<= 1) {
            const uint32_t add = (tid >= step) ? sm.pref[tid + 1u - step] : 0u;
            __syncthreads();
            vh += add;
            sm.pref[tid + 1u] = static_cast<uint16_t>(vh);
            __syncthreads();
        }
        const uint32_t n_heavy = __builtin_amdgcn_readfirstlane(sm.pref[256]);
        const uint32_t h_idx = vh - is_h;
        sm.ord[is_h ? h_idx : (n_heavy + tid - h_idx)] = static_cast<uint8_t>(tid);
        __syncthreads();

        const uint32_t c = exp_cnt[sm.ord[tid]];
        uint32_t v = ((c + DN_T - 1u) / DN_T) * RT;
        sm.pref[tid + 1u] = static_cast<uint16_t>(v);
        __syncthreads();
        #pragma unroll
        for (uint32_t step = 1u; step < 256u; step <<= 1) {
            const uint32_t add = (tid >= step) ? sm.pref[tid + 1u - step] : 0u;
            __syncthreads();
            v += add;
            sm.pref[tid + 1u] = static_cast<uint16_t>(v);
            __syncthreads();
        }
        if (tid == 0u) sm.pref[0] = 0u;
        __syncthreads();
    }
    const uint32_t total = __builtin_amdgcn_readfirstlane(sm.pref[256]);
    const uint32_t role  = (tid < 128u) ? 0u : ((tid < 192u) ? 1u : ((tid < 224u) ? 2u : 3u));

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
        const uint32_t t0    = tt * DN_T;
        const uint32_t tcnt  = min(DN_T, cnt - t0);
        if (tid < DN_T) {
            const uint32_t r = sorted_ms[off + t0 + min(tid, tcnt - 1u)];
            sm.ms[tid] = r;
            sm.ew[tid] = top_w[r];
        }
        __syncthreads();

        uint32_t v0 = 0u, v1 = 0u;
        if (role == 0u) {
            const uint32_t erow = e * D + rt * 128u + tid;
            v0 = erow * w_stride;
            v1 = erow * s_stride;
        } else if (role == 1u) {
            const uint32_t u = tid - 128u;
            v0 = sm.ms[u >> 1] * local_ff + (u & 1u) * 16u;
        } else if (role == 2u) {
            v0 = sm.ms[tid - 192u] * (nib * 4u);
        }

        float acc[4][4];
        #pragma unroll
        for (uint32_t i = 0u; i < 4u; ++i) {
            #pragma unroll
            for (uint32_t j = 0u; j < 4u; ++j) acc[i][j] = 0.0f;
        }

        uint4 s_q  = make_uint4(0u, 0u, 0u, 0u);
        uint2 s_sv = make_uint2(0u, 0u);
        if (role == 0u) {
            s_q  = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(r_wq, v0, 0u, 0));
            if (sub32_u) s_sv.x = __builtin_amdgcn_raw_buffer_load_b32(r_ws, v1, 0u, 0);
            else         s_sv   = __builtin_bit_cast(uint2, __builtin_amdgcn_raw_buffer_load_b64(r_ws, v1, 0u, 0));
        } else if (role == 1u) {
            s_q = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(r_mq, v0, 0u, 0));
        } else if (role == 2u) {
            s_q.x = __builtin_amdgcn_raw_buffer_load_b32(r_md, v0, 0u, 0);
        }

        #pragma nounroll
        for (uint32_t ib = 0u; ib < nib; ++ib) {
            if (role == 0u) {
                uint32_t* wd = sm.w + tid * DN_WS;
                wd[0] = dec4_iq3(s_q.x);       wd[1] = dec4_iq3(s_q.y);
                wd[2] = dec4_iq3(s_q.z);       wd[3] = dec4_iq3(s_q.w);
                wd[4] = dec4_iq3(s_q.x >> 4);  wd[5] = dec4_iq3(s_q.y >> 4);
                wd[6] = dec4_iq3(s_q.z >> 4);  wd[7] = dec4_iq3(s_q.w >> 4);
                if (sub32_u) {
                    sm.sc[tid] = __uint_as_float(s_sv.x);
                } else {
                    const uint32_t sc = (s_sv.x >> ((ib & 7u) * 4u)) & 0xFu;
                    sm.sc[tid] = __uint_as_float(s_sv.y) * (0.5f + static_cast<float>(sc)) * 0.5f;
                }
            } else if (role == 1u) {
                const uint32_t u = tid - 128u;
                *reinterpret_cast<uint4*>(sm.x + (u >> 1) * DN_XS + (u & 1u) * 4u) = s_q;
            } else if (role == 2u) {
                const uint32_t u = tid - 192u;
                sm.xew[u] = sm.ew[u] * __uint_as_float(s_q.x);
            }
            __syncthreads();
            if (ib + 1u < nib) {
                const uint32_t in = ib + 1u;
                if (role == 0u) {
                    s_q = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(r_wq, v0, in * 16u, 0));
                    if (sub32_u)       s_sv.x = __builtin_amdgcn_raw_buffer_load_b32(r_ws, v1, in * 4u, 0);
                    else if (in == 8u) s_sv   = __builtin_bit_cast(uint2, __builtin_amdgcn_raw_buffer_load_b64(r_ws, v1, 8u, 0));
                } else if (role == 1u) {
                    s_q = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(r_mq, v0, in * 32u, 0));
                } else if (role == 2u) {
                    s_q.x = __builtin_amdgcn_raw_buffer_load_b32(r_md, v0, in * 4u, 0);
                }
            }
            // 4x4 micro-tile in 2 token pairs with single int32 dot per 32-elem block + empty-wave skip (0 B spill)
            const uint32_t t_base = ty * 4u;
            auto run_dn_pair = [&](uint32_t t_a, float& a00, float& a01, float& a10, float& a11,
                                   float& a20, float& a21, float& a30, float& a31) {
                const uint32_t t_b = t_a + 1u;
                int32_t d0[4] = {0, 0, 0, 0};
                int32_t d1[4] = {0, 0, 0, 0};
                #pragma nounroll
                for (uint32_t h = 0u; h < 2u; ++h) {
                    const uint4 xa = *reinterpret_cast<const uint4*>(&sm.x[t_a * DN_XS + h * 4u]);
                    const uint4 xb = *reinterpret_cast<const uint4*>(&sm.x[t_b * DN_XS + h * 4u]);
                    __builtin_amdgcn_sched_barrier(0);
                    #pragma unroll
                    for (uint32_t i = 0u; i < 4u; ++i) {
                        const uint32_t rr = tx + 32u * i;
                        uint2 wv = *reinterpret_cast<const uint2*>(&sm.w[rr * DN_WS + h * 4u]);
                        d0[i] = sdot4(wv.x, xa.x, d0[i]);  d1[i] = sdot4(wv.x, xb.x, d1[i]);
                        d0[i] = sdot4(wv.y, xa.y, d0[i]);  d1[i] = sdot4(wv.y, xb.y, d1[i]);
                        wv = *reinterpret_cast<const uint2*>(&sm.w[rr * DN_WS + h * 4u + 2u]);
                        d0[i] = sdot4(wv.x, xa.z, d0[i]);  d1[i] = sdot4(wv.x, xb.z, d1[i]);
                        d0[i] = sdot4(wv.y, xa.w, d0[i]);  d1[i] = sdot4(wv.y, xb.w, d1[i]);
                    }
                    __builtin_amdgcn_sched_barrier(0);
                }
                const float xa_sc = sm.xew[t_a];
                const float xb_sc = sm.xew[t_b];
                const float g0 = sm.sc[tx], g1 = sm.sc[tx + 32u], g2 = sm.sc[tx + 64u], g3 = sm.sc[tx + 96u];
                a00 = fmaf(g0 * xa_sc, static_cast<float>(d0[0]), a00);
                a01 = fmaf(g0 * xb_sc, static_cast<float>(d1[0]), a01);
                a10 = fmaf(g1 * xa_sc, static_cast<float>(d0[1]), a10);
                a11 = fmaf(g1 * xb_sc, static_cast<float>(d1[1]), a11);
                a20 = fmaf(g2 * xa_sc, static_cast<float>(d0[2]), a20);
                a21 = fmaf(g2 * xb_sc, static_cast<float>(d1[2]), a21);
                a30 = fmaf(g3 * xa_sc, static_cast<float>(d0[3]), a30);
                a31 = fmaf(g3 * xb_sc, static_cast<float>(d1[3]), a31);
                __builtin_amdgcn_sched_barrier(0);
            };
            if (ty * 2u < tcnt)
                run_dn_pair(ty * 2u,      acc[0][0], acc[0][1], acc[1][0], acc[1][1], acc[2][0], acc[2][1], acc[3][0], acc[3][1]);
            if (16u + ty * 2u < tcnt)
                run_dn_pair(16u + ty * 2u, acc[0][2], acc[0][3], acc[1][2], acc[1][3], acc[2][2], acc[2][3], acc[3][2], acc[3][3]);
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

// Coalesced 8-slot combine for vj_prefill_moe_down_iq3_tiled_kernel: grid (ceil(D/256), M), block 256.
extern "C" __global__ __launch_bounds__(256, 4)
void vj_moe_down_combine_all_f32_kernel(
    const float* __restrict__ part_in,
    const float* __restrict__ sh_down_in,
    float*       __restrict__ out_f32,
    uint32_t M,
    uint32_t D) {
    const uint32_t m   = blockIdx.y;
    const uint32_t row = blockIdx.x * 256u + threadIdx.x;
    if (m >= M || row >= D) return;
    const float* base = part_in + static_cast<uint64_t>(m) * 8u * D + row;
    float acc = 0.0f;
    #pragma unroll
    for (uint32_t s = 0u; s < 8u; ++s) acc += base[static_cast<uint64_t>(s) * D];
    if (sh_down_in) acc += sh_down_in[static_cast<uint64_t>(m) * D + row];
    out_f32[static_cast<uint64_t>(m) * D + row] = acc;
}

// ============================================================================
// High-occupancy 32x32xK32 LDS-tiled FP16xFP16 (v_dot2_f32_f16) MoE router logits + shared gate.
// Grid: (9, ceil(M / 32)) = 144 WGs at M=512 (saturates all 60 CUs!), block: 128.
//   blockIdx.x < 8 : 32 experts x 32 tokens x K=32 tiled fdot2 GEMM (LDS = 4,352 B -> 8 WGs/CU!).
//   blockIdx.x == 8: shared gate scalar (1 row x 32 tokens, 4 threads/token).
// ============================================================================
extern "C" __global__ __launch_bounds__(128, 4)
void vj_prefill_router_fp16_tiled_kernel(
    const uint16_t* __restrict__ w_router_f16,
    const float*    __restrict__ w_sh_gate,
    const float*    __restrict__ x_norm,
    float*          __restrict__ logits_out,
    float*          __restrict__ sh_gate_out,
    uint32_t M,
    uint32_t D) {
    typedef _Float16 hf2_t __attribute__((ext_vector_type(2)));
    __shared__ uint32_t sm_w[32u * 17u]; // 2,176 B (16 half2 + 1 pad = 0 bank conflicts!)
    __shared__ uint32_t sm_x[32u * 17u]; // 2,176 B -> 4,352 B total (HALF of FP32 LDS!)
    const uint32_t tid = threadIdx.x; // 0..127
    const uint32_t m0  = blockIdx.y * 32u;
    if (m0 >= M) return;

    if (blockIdx.x < 8u) {
        const uint32_t e0 = blockIdx.x * 32u;
        const uint32_t tx = tid & 7u;        // 0..7 -> 4 experts tx + 8*i
        const uint32_t ty = tid >> 3;        // 0..15 -> 2 tokens ty + 16*j
        const uint32_t lr = tid >> 2;        // 0..31
        const uint32_t lc = (tid & 3u) * 4u; // 0, 4, 8, 12 (half2 index -> 8 FP16s = 16 B)
        const uint32_t lm = min(m0 + lr, M - 1u);

        float acc[4][2];
        #pragma unroll
        for (uint32_t i = 0u; i < 4u; ++i) { acc[i][0] = 0.0f; acc[i][1] = 0.0f; }

        const uint32_t wr_stride = (D == 3072u) ? 3200u : D;
        #pragma nounroll
        for (uint32_t k0 = 0u; k0 < D; k0 += 32u) {
            const uint4 wh = *reinterpret_cast<const uint4*>(w_router_f16 + static_cast<uint64_t>(e0 + lr) * wr_stride + k0 + lc * 2u);
            uint32_t* wd = sm_w + lr * 17u + lc;
            wd[0] = wh.x; wd[1] = wh.y; wd[2] = wh.z; wd[3] = wh.w;

            const float4* xp = reinterpret_cast<const float4*>(x_norm + static_cast<uint64_t>(lm) * D + k0 + lc * 2u);
            const float4 xv0 = xp[0], xv1 = xp[1];
            const __half2 h0 = __float22half2_rn(make_float2(xv0.x, xv0.y));
            const __half2 h1 = __float22half2_rn(make_float2(xv0.z, xv0.w));
            const __half2 h2 = __float22half2_rn(make_float2(xv1.x, xv1.y));
            const __half2 h3 = __float22half2_rn(make_float2(xv1.z, xv1.w));
            uint32_t* xd = sm_x + lr * 17u + lc;
            xd[0] = *reinterpret_cast<const uint32_t*>(&h0);
            xd[1] = *reinterpret_cast<const uint32_t*>(&h1);
            xd[2] = *reinterpret_cast<const uint32_t*>(&h2);
            xd[3] = *reinterpret_cast<const uint32_t*>(&h3);
            __syncthreads();

            #pragma unroll 4
            for (uint32_t k = 0u; k < 16u; ++k) {
                const hf2_t x0 = __builtin_bit_cast(hf2_t, sm_x[ty * 17u + k]);
                const hf2_t x1 = __builtin_bit_cast(hf2_t, sm_x[(ty + 16u) * 17u + k]);
                #pragma unroll
                for (uint32_t i = 0u; i < 4u; ++i) {
                    const hf2_t wv = __builtin_bit_cast(hf2_t, sm_w[(tx + 8u * i) * 17u + k]);
                    acc[i][0] = __builtin_amdgcn_fdot2(wv, x0, acc[i][0], false);
                    acc[i][1] = __builtin_amdgcn_fdot2(wv, x1, acc[i][1], false);
                }
            }
            __syncthreads();
        }

        #pragma unroll
        for (uint32_t j = 0u; j < 2u; ++j) {
            const uint32_t m = m0 + ty + 16u * j;
            if (m < M) {
                #pragma unroll
                for (uint32_t i = 0u; i < 4u; ++i)
                    logits_out[static_cast<uint64_t>(m) * 256u + e0 + tx + 8u * i] = acc[i][j];
            }
        }
    } else {
        float* sm_wf = reinterpret_cast<float*>(sm_w);
        const uint32_t t     = tid >> 2;
        const uint32_t lane4 = tid & 3u;
        const uint32_t m     = m0 + t;
        float acc = 0.0f;
        #pragma nounroll
        for (uint32_t k0 = 0u; k0 < D; k0 += 128u) {
            sm_wf[tid] = (k0 + tid < D) ? w_sh_gate[k0 + tid] : 0.0f;
            __syncthreads();
            if (m < M) {
                const uint32_t lim = min(128u, D - k0);
                for (uint32_t kk = lane4 * 4u; kk < lim; kk += 16u) {
                    const float4 xv = *reinterpret_cast<const float4*>(x_norm + static_cast<uint64_t>(m) * D + k0 + kk);
                    acc = fmaf(sm_wf[kk + 0u], xv.x, fmaf(sm_wf[kk + 1u], xv.y, fmaf(sm_wf[kk + 2u], xv.z, fmaf(sm_wf[kk + 3u], xv.w, acc))));
                }
            }
            __syncthreads();
        }
        acc += __shfl_xor(acc, 2, 4);
        acc += __shfl_xor(acc, 1, 4);
        if (lane4 == 0u && m < M) sh_gate_out[m] = 1.0f / (1.0f + __expf(-acc));
    }
}
