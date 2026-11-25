// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Expert-grouped routed MoE Down for multi-token batches (M = 2..512). Design, exactness and byte budget:
// ops/kernels/vj_moe_down_grouped.hiph. CPU reference + arithmetic model: core/moe_route_group.*.
#include <hip/hip_runtime.h>
#include <stdint.h>

#include "ops/kernels/vj_moe_down_grouped.hiph"

using namespace vegajet::moe_grp;
using namespace vegajet::moe_grp_dev;

// ============================================================================
// 1. Routing group: grid = (1), block = 256 (4 waves). M <= kMaxM.
//    Stable counting sort of the R = M*8 routes by expert without atomics: wave w owns the contiguous route
//    range [w*Q, w*Q+Q); within a 64-route chunk equal-expert peers are found with 8 ballots, ranked with
//    mbcnt, and the lowest peer advances the wave's private cursor. Cursors start at
//    exp_off[e] + (routes of e in waves < w), so the list is ascending route index (token, then slot) per expert.
//    Outputs: hdr[4], exp_cnt[256], exp_off[257], uniq[256] (first hdr[0] valid, ascending expert id),
//             items[item_cap] (first hdr[1] valid, ascending expert then start), routes[R] (value m*8+s).
// ============================================================================
namespace {
__device__ __forceinline__ uint64_t equal_peers8(uint32_t e, bool valid) {
    uint64_t m = __ballot(valid);
    #pragma unroll
    for (uint32_t b = 0u; b < 8u; ++b) {
        const bool     bit = (e >> b) & 1u;
        const uint64_t bb  = __ballot(bit);
        m &= bit ? bb : ~bb;
    }
    return m;
}
__device__ __forceinline__ uint32_t mbcnt64(uint64_t m) {
    return __builtin_amdgcn_mbcnt_hi(static_cast<uint32_t>(m >> 32),
                                     __builtin_amdgcn_mbcnt_lo(static_cast<uint32_t>(m), 0u));
}
} // namespace

extern "C" __global__ __launch_bounds__(256, 1)
void vj_moe_route_group_kernel(
    const int32_t* __restrict__ top_ids,
    uint32_t*      __restrict__ hdr,
    uint32_t*      __restrict__ exp_cnt,
    uint32_t*      __restrict__ exp_off,
    uint32_t*      __restrict__ uniq,
    uint32_t*      __restrict__ items,
    uint32_t*      __restrict__ routes,
    uint32_t M,
    uint32_t min_group,
    uint32_t item_cap_n) {
    __shared__ uint8_t  sm_id[kMaxM * kTopK];   // 4 KiB
    __shared__ uint32_t sm_wc[4u * 256u];       // per-wave counts, then per-wave cursors (4 KiB)
    __shared__ uint32_t sm_scan[256];           // packed scan (1 KiB)

    const uint32_t tid  = threadIdx.x;
    const uint32_t wave = tid >> 6;
    const uint32_t lane = tid & 63u;
    const uint32_t R    = min(M, kMaxM) * kTopK;
    const uint32_t Q    = ((R + 255u) / 256u) * 64u;   // per-wave range, multiple of 64
    const uint32_t r_lo = min(R, wave * Q);
    const uint32_t r_hi = min(R, r_lo + Q);

    for (uint32_t i = tid; i < R; i += 256u) sm_id[i] = static_cast<uint8_t>(top_ids[i] & 255);
    #pragma unroll
    for (uint32_t w = 0u; w < 4u; ++w) sm_wc[w * 256u + tid] = 0u;
    __syncthreads();

    // Pass 1: per-wave counts (leader-only plain LDS writes, no atomics).
    for (uint32_t base = r_lo; base < r_hi; base += 64u) {
        const uint32_t idx   = base + lane;
        const bool     valid = idx < r_hi;
        const uint32_t e     = valid ? sm_id[idx] : 0u;
        const uint64_t peers = equal_peers8(e, valid);
        if (valid && mbcnt64(peers) == 0u) sm_wc[wave * 256u + e] += static_cast<uint32_t>(__popcll(peers));
    }
    __syncthreads();

    // Thread = expert: packed scan of {count (13 b), unique flag (9 b), item count (10 b)}.
    const uint32_t c0 = sm_wc[0u * 256u + tid], c1 = sm_wc[1u * 256u + tid];
    const uint32_t c2 = sm_wc[2u * 256u + tid], c3 = sm_wc[3u * 256u + tid];
    const uint32_t c   = c0 + c1 + c2 + c3;
    const uint32_t nit = items_for_count(c, min_group);
    const uint32_t pk  = c | ((c != 0u ? 1u : 0u) << 13) | (nit << 22);
    sm_scan[tid] = pk;
    __syncthreads();
    #pragma unroll
    for (uint32_t step = 1u; step < 256u; step <<= 1) {
        const uint32_t add = (tid >= step) ? sm_scan[tid - step] : 0u;
        __syncthreads();
        sm_scan[tid] += add;
        __syncthreads();
    }
    const uint32_t ex   = sm_scan[tid] - pk;             // exclusive
    const uint32_t off  = ex & 0x1FFFu;
    const uint32_t uoff = (ex >> 13) & 0x1FFu;
    const uint32_t ioff = ex >> 22;

    exp_cnt[tid] = c;
    exp_off[tid] = off;
    if (c != 0u) uniq[uoff] = tid;
    for (uint32_t k = 0u; k < nit; ++k) {
        const uint32_t st = off + k * kItemRoutes;
        if (ioff + k < item_cap_n) items[ioff + k] = item_pack(tid, st, min(kItemRoutes, c - k * kItemRoutes));
    }
    sm_wc[0u * 256u + tid] = off;
    sm_wc[1u * 256u + tid] = off + c0;
    sm_wc[2u * 256u + tid] = off + c0 + c1;
    sm_wc[3u * 256u + tid] = off + c0 + c1 + c2;
    if (tid == 255u) {
        const uint32_t inc = sm_scan[255];
        exp_off[256]           = inc & 0x1FFFu;
        hdr[kHdrUnique]        = (inc >> 13) & 0x1FFu;
        hdr[kHdrItems]         = min(inc >> 22, item_cap_n);
        hdr[kHdrRoutes]        = R;
    }
    // grouped-route total: experts with count >= min_group (one wave-reduction-free pass via LDS reuse below)
    __syncthreads();
    sm_scan[tid] = (nit != 0u) ? c : 0u;
    __syncthreads();

    // Pass 2: stable scatter (each wave only touches its own cursors; LDS ops of a wave stay in order).
    for (uint32_t base = r_lo; base < r_hi; base += 64u) {
        const uint32_t idx   = base + lane;
        const bool     valid = idx < r_hi;
        const uint32_t e     = valid ? sm_id[idx] : 0u;
        const uint64_t peers = equal_peers8(e, valid);
        const uint32_t rank  = mbcnt64(peers);
        const uint32_t cur   = sm_wc[wave * 256u + e];
        if (valid) routes[cur + rank] = idx;
        if (valid && rank == 0u) sm_wc[wave * 256u + e] = cur + static_cast<uint32_t>(__popcll(peers));
    }

    if (tid < 64u) {
        uint32_t g = sm_scan[tid] + sm_scan[tid + 64u] + sm_scan[tid + 128u] + sm_scan[tid + 192u];
        #pragma unroll
        for (uint32_t o = 32u; o > 0u; o >>= 1) g += __shfl_xor(g, o, 64);
        if (tid == 0u) hdr[kHdrGroupedRoutes] = g;
    }
}

// ============================================================================
// 2. Grouped Down: grid = (ceil(D / (16*kR)), item_cap(M)), block = 256 (16 row groups x 16 lanes),
//    kR = 4 (_dp, 64-row tiles) or 2 (_f32, 32-row tiles: 4 rows + gdb + 4 sum16 chains spill 11 VGPRs).
//    Lane (rg, l) owns rows tile*16*kR + rg + 16*j (j < kR), block l: codes are loaded once (glc slc: every
//    weight tile is read by exactly one workgroup per item) and decoded once into 8*kR VGPRs of int8.
//    Routes of the item are staged 16 at a time in LDS (8 KiB of int8 activations, normal loads: each
//    activation is shared by all row tiles of the item), then every staged route costs 2 ds_read_b128
//    + 32 v_dot4 per lane. Workgroups with blockIdx.y >= hdr[1] exit immediately.
//    Requires D % (16 * kR) == 0 (D = 3072): no row guards inside the token loop.
//    _dp : dp_out[(route * D + row) * 16 + lane] = exact int32 block dot.
//    _f32: part_out[route * D + row] = sum16((ew * gdb * xd) * float(dp)).
// ============================================================================
template <bool kF32, uint32_t kR>
__device__ __forceinline__ void grouped_body(
    const uint8_t* __restrict__ w_qs, const uint8_t* __restrict__ w_sc,
    const int8_t* __restrict__ mid_q8, const float* __restrict__ mid_d, const float* __restrict__ top_w,
    const uint32_t* __restrict__ hdr, const uint32_t* __restrict__ items, const uint32_t* __restrict__ routes,
    int32_t* __restrict__ dp_out, float* __restrict__ part_out,
    uint32_t D, uint32_t w_stride, uint32_t s_stride) {
    #pragma clang fp contract(off)
    __shared__ uint4    sm_x[kStageRoutes * 32u];   // 8 KiB
    __shared__ uint32_t sm_r[kStageRoutes];
    __shared__ float    sm_d[kStageRoutes * 16u];   // f32 mode only (1 KiB)
    __shared__ float    sm_w[kStageRoutes];

    if (blockIdx.y >= hdr[kHdrItems]) return;
    const uint32_t it    = __builtin_amdgcn_readfirstlane(items[blockIdx.y]);
    const uint32_t e     = item_expert(it);
    const uint32_t start = item_start(it);
    const uint32_t len   = item_len(it);

    const uint32_t tid    = threadIdx.x;
    const uint32_t lane16 = tid & 15u;
    const uint32_t row0   = blockIdx.x * (16u * kR) + (tid >> 4);

    const uint64_t eb = static_cast<uint64_t>(e) * D;
    const uint8_t* wb = w_qs + uniform_u64(eb * w_stride);
    Dec32 wd[kR];
    float gdb[kR];
    #pragma unroll
    for (uint32_t j = 0u; j < kR; ++j) {
        const uint32_t row = row0 + 16u * j;
        uint32_t wo = row * w_stride + lane16 * 16u;   // opaque: keep the saddr form (SGPR base + 32-bit offset)
        asm volatile("" : "+v"(wo));
        const uint4 q = ldg_x4_nt(reinterpret_cast<const uint4*>(wb + wo));
        wd[j] = decode32_iq3(q);
        gdb[j] = 0.0f;
        if (kF32) {
            const uint8_t* sb = w_sc + uniform_u64(eb * s_stride);
            uint32_t so = row * s_stride + (lane16 >> 3) * 8u;
            asm volatile("" : "+v"(so));
            const uint2 sv = ldg_x2_nt(reinterpret_cast<const uint2*>(sb + so));
            gdb[j] = gdb_of(sv, lane16);
        }
    }

    for (uint32_t st = 0u; st < len; st += kStageRoutes) {
        const uint32_t n = min(kStageRoutes, len - st);
        if (st != 0u) __syncthreads();
        // Pipelined stage loader: fetch route indices once in parallel, then issue both mid_q8 + mid_d/top_w loads
        // concurrently before writing LDS (2 vmcnt stalls instead of 5).
        const uint32_t j0 = tid >> 5;
        const uint32_t j1 = (tid + 256u) >> 5;
        const uint32_t r0 = (j0 < n) ? routes[start + st + j0] : 0u;
        const uint32_t r1 = (j1 < n) ? routes[start + st + j1] : 0u;
        const uint32_t rt = (tid < n) ? routes[start + st + tid] : 0u;
        uint32_t rf = 0u;
        if (kF32) {
            const uint32_t jf = tid >> 4;
            if (jf < n) rf = routes[start + st + jf];
        }
        uint4 v0 = make_uint4(0u, 0u, 0u, 0u), v1 = make_uint4(0u, 0u, 0u, 0u);
        if (j0 < n) v0 = reinterpret_cast<const uint4*>(mid_q8 + static_cast<uint64_t>(r0) * kCols)[tid & 31u];
        if (j1 < n) v1 = reinterpret_cast<const uint4*>(mid_q8 + static_cast<uint64_t>(r1) * kCols)[tid & 31u];
        float fd = 0.0f, fw = 0.0f;
        if (kF32 && (tid >> 4) < n) {
            fd = mid_d[static_cast<uint64_t>(rf) * 16u + lane16];
            if (lane16 == 0u) fw = top_w[rf];
        }
        if (j0 < n) sm_x[tid]        = v0;
        if (j1 < n) sm_x[tid + 256u] = v1;
        if (tid < n) sm_r[tid]       = rt;
        if (kF32 && (tid >> 4) < n) {
            sm_d[tid] = fd;
            if (lane16 == 0u) sm_w[tid >> 4] = fw;
        }
        __syncthreads();

        #pragma nounroll
        for (uint32_t t = 0u; t < n; ++t) {
            const uint32_t r  = __builtin_amdgcn_readfirstlane(sm_r[t]);
            const uint4    xa = sm_x[t * 32u + lane16 * 2u + 0u];
            const uint4    xb = sm_x[t * 32u + lane16 * 2u + 1u];
            if (!kF32) {
                // SGPR base (route record) + one lane offset (tid * 4); rows j fold into the immediate offset.
                uint8_t* ob = reinterpret_cast<uint8_t*>(dp_out) +
                              uniform_u64((static_cast<uint64_t>(r) * D + blockIdx.x * (16u * kR)) * 64u);
                uint32_t to = tid * 4u;
                asm volatile("" : "+v"(to));
                #pragma unroll
                for (uint32_t j = 0u; j < kR; ++j)
                    *reinterpret_cast<int32_t*>(ob + to + 1024u * j) = dot32_dec(wd[j], xa, xb);
            } else {
                const float ew = sm_w[t];
                const float xd = sm_d[t * 16u + lane16];
                float* ob = part_out + uniform_u64(static_cast<uint64_t>(r) * D);
                #pragma unroll
                for (uint32_t j = 0u; j < kR; ++j) {
                    const uint32_t row = row0 + 16u * j;
                    const int32_t  dp  = dot32_dec(wd[j], xa, xb);
                    const float    a   = ew * gdb[j] * xd;
                    const float    v   = sum16(a * static_cast<float>(dp));
                    if (lane16 == 0u) ob[row] = v;
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 4)
void vj_moe_down_grouped_dp_kernel(
    const uint8_t*  __restrict__ w_qs,
    const int8_t*   __restrict__ mid_q8,
    const uint32_t* __restrict__ grp_hdr,
    const uint32_t* __restrict__ grp_items,
    const uint32_t* __restrict__ grp_routes,
    int32_t*        __restrict__ dp_out,
    uint32_t D,
    uint32_t w_stride) {
    grouped_body<false, kRowsPerThreadDp>(w_qs, nullptr, mid_q8, nullptr, nullptr, grp_hdr, grp_items, grp_routes,
                        dp_out, nullptr, D, w_stride, 0u);
}

extern "C" __global__ __launch_bounds__(256, 4)
void vj_moe_down_grouped_f32_kernel(
    const uint8_t*  __restrict__ w_qs,
    const uint8_t*  __restrict__ w_sc,
    const int8_t*   __restrict__ mid_q8,
    const float*    __restrict__ mid_d,
    const float*    __restrict__ top_w,
    const uint32_t* __restrict__ grp_hdr,
    const uint32_t* __restrict__ grp_items,
    const uint32_t* __restrict__ grp_routes,
    float*          __restrict__ part_out,
    uint32_t D,
    uint32_t w_stride,
    uint32_t s_stride) {
    grouped_body<true, kRowsPerThreadF32>(w_qs, w_sc, mid_q8, mid_d, top_w, grp_hdr, grp_items, grp_routes,
                       nullptr, part_out, D, w_stride, s_stride);
}

// ============================================================================
// 3. Combine: grid = (D / 16, M), block = 256, same LDS staging as vj_moe_down_dot8_kernel.
//    Slot s of token m is wide iff exp_cnt[top_ids[m*8+s]] >= min_group (same predicate as the routing kernel).
// ============================================================================
#define VJ_COMBINE_PROLOGUE()                                                                                   \
    __shared__ uint4   sm_mq[256];                                                                              \
    __shared__ float   sm_md[128];                                                                              \
    __shared__ int32_t sm_ids[8];                                                                               \
    __shared__ float   sm_w[8];                                                                                 \
    __shared__ int32_t sm_g[8];                                                                                 \
    const uint32_t m      = blockIdx.y;                                                                         \
    const uint32_t lane16 = threadIdx.x & 15u;                                                                  \
    const uint32_t row    = blockIdx.x * 16u + (threadIdx.x >> 4);                                              \
    if (m >= M) return;                                                                                         \
    sm_mq[threadIdx.x] = reinterpret_cast<const uint4*>(mid_q8 + static_cast<uint64_t>(m) * 4096u)[threadIdx.x]; \
    if (threadIdx.x < 128u) sm_md[threadIdx.x] = mid_d[static_cast<uint64_t>(m) * 128u + threadIdx.x];          \
    if (threadIdx.x < 8u) {                                                                                     \
        const int32_t id = top_ids[static_cast<uint64_t>(m) * 8u + threadIdx.x];                                \
        sm_ids[threadIdx.x] = id;                                                                               \
        sm_w[threadIdx.x]   = top_w[static_cast<uint64_t>(m) * 8u + threadIdx.x];                               \
        sm_g[threadIdx.x]   = (exp_cnt[static_cast<uint32_t>(id) & 255u] >= min_group) ? 1 : 0;                 \
    }                                                                                                           \
    __syncthreads();                                                                                            \
    if (row >= D) return;                                                                                       \
    uint32_t gmask = 0u;                                                                                        \
    _Pragma("unroll") for (uint32_t s = 0u; s < 8u; ++s)                                                        \
        gmask |= static_cast<uint32_t>(__builtin_amdgcn_readfirstlane(sm_g[s])) << s;

extern "C" __global__ __launch_bounds__(256, 5)
void vj_moe_down_combine_dp_kernel(
    const uint8_t*  __restrict__ w_qs,
    const uint8_t*  __restrict__ w_sc,
    const int8_t*   __restrict__ mid_q8,
    const float*    __restrict__ mid_d,
    const int32_t*  __restrict__ top_ids,
    const float*    __restrict__ top_w,
    const float*    __restrict__ sh_down_in,
    const uint32_t* __restrict__ exp_cnt,
    const int32_t*  __restrict__ dp_in,
    float*          __restrict__ out_f32,
    uint32_t M,
    uint32_t D,
    uint32_t w_stride,
    uint32_t s_stride,
    uint32_t min_group) {
    VJ_COMBINE_PROLOGUE()
    const int32_t* dp_tok = dp_in + static_cast<uint64_t>(m) * 8u * D * 16u;
    // gmask == 0 (all skinny) runs the mixed body with every select false: the same loads, LDS reads, dot and
    // float chain as vj_dot8_down_row_iq3<4> (plus one v_cndmask per slot). A separate call of that helper
    // here costs 2 spilled VGPRs at the 48 cap, so it is not used.
    float total;
    if (gmask == 0xFFu) {   // all slots wide: dp + scales only
        total = down_row_mixed_dp<8, true>(w_qs, w_stride, w_sc, s_stride, dp_tok,
                                           sm_mq, sm_md, sm_ids, sm_w, gmask, row, D, lane16);
    } else {
        total = down_row_mixed_dp<4, false>(w_qs, w_stride, w_sc, s_stride, dp_tok,
                                            sm_mq, sm_md, sm_ids, sm_w, gmask, row, D, lane16);
    }
    if (lane16 == 0u) {
        const float sh = sh_down_in ? sh_down_in[static_cast<uint64_t>(m) * D + row] : 0.0f;
        out_f32[static_cast<uint64_t>(m) * D + row] = total + sh;
    }
}

extern "C" __global__ __launch_bounds__(256, 5)
void vj_moe_down_combine_f32_kernel(
    const uint8_t*  __restrict__ w_qs,
    const uint8_t*  __restrict__ w_sc,
    const int8_t*   __restrict__ mid_q8,
    const float*    __restrict__ mid_d,
    const int32_t*  __restrict__ top_ids,
    const float*    __restrict__ top_w,
    const float*    __restrict__ sh_down_in,
    const uint32_t* __restrict__ exp_cnt,
    const float*    __restrict__ part_in,
    float*          __restrict__ out_f32,
    uint32_t M,
    uint32_t D,
    uint32_t w_stride,
    uint32_t s_stride,
    uint32_t min_group) {
    VJ_COMBINE_PROLOGUE()
    float total = down_row_skinny_only<4>(w_qs, w_stride, w_sc, s_stride,
                                          sm_mq, sm_md, sm_ids, sm_w, gmask, row, D, lane16);
    if (lane16 == 0u) {
        #pragma unroll
        for (uint32_t s = 0u; s < 8u; ++s)
            if ((gmask >> s) & 1u) total += part_in[(static_cast<uint64_t>(m) * 8u + s) * D + row];
        const float sh = sh_down_in ? sh_down_in[static_cast<uint64_t>(m) * D + row] : 0.0f;
        out_f32[static_cast<uint64_t>(m) * D + row] = total + sh;
    }
}
