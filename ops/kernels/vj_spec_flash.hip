// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// 16:1 head-packed Split-KV FlashDecoding for gfx906 (M = 1 decode and small-M verify), a drop-in for
// ornith_attn_flash_quant_kernel (same q_all / gate_all / KV-cache / out_q8 / out_ds layouts).
//
// Why: ornith_attn_flash_quant_kernel launches one WG per Q head, each walking all seqlen positions in order,
// so 16 WGs per GPU (16 of 60 CUs busy) re-read the single local KV head 16 times and the dependency chain
// grows with seqlen (the tg128 -> tg512 slowdown). Here:
//   stage 1  vj_spec_flash_partial_kernel  grid (n_chunks, M, n_kv_local) x 512 threads
//            WG (c, m, kvh) owns positions [c*chunk_len, min((c+1)*chunk_len, pos0+m+1)) of KV head kvh and
//            scores ALL 16 Q heads of that group against each K row: wave w takes heads 2w and 2w+1, a 32-lane
//            half-wave per head, each lane owning 8 of the 256 dims (q and o in registers, 4 positions of K
//            and V in flight). Each K/V row is fetched from HBM once per WG (the 8 waves hit the same lines
//            in L1/L2). Output per (m, head, chunk): the unnormalised online-softmax state (m_c, l_c, o_c[256])
//            in FP32.
//   stage 2  vj_spec_flash_merge_quant_kernel  grid (n_q_local, M) x 256 threads
//            merges the chunk states of one head (log-sum-exp rescale), divides by l, applies the sigmoid
//            gate and writes the same block-32 INT8 + (scale, sum) output as the sequential kernel.
//
// Numerics: positions are combined in a different floating-point association than the sequential kernel
// (chunked partial sums, 4 positions per online-softmax rescale, K scale factored out of each lane's 8-dim
// partial dot, 32-lane tree reduction). The result is therefore NOT bit-identical to ornith_attn_flash_quant_kernel;
// it is equivalent at the token level only (tests/test_spec_flash_ref.cpp bounds the difference on the CPU
// against an FP64 reference and the sequential order). A lossless-gate run is required before it replaces
// the sequential kernel in a parity-gated path.
//
// Host contract:
//   n_q_local / n_kv_local must be 16 (Ornith: 16 local Q heads, 1 local KV head); head_dim 256;
//   KV row = 16 B FP16 block-32 scales + 256 INT8 at ((t * n_kv_local + kvh) * 272) (same as the prep kernel);
//   chunk_len a multiple of 4 with n_chunks * chunk_len >= pos0 + M (suggested:
//     n_chunks = clamp(ceil((pos0 + M) / 32), 1, 60), chunk_len = round_up(ceil((pos0 + M) / n_chunks), 4));
//   part_o  : M * n_q_local * n_chunks * 256 floats, part_ml : M * n_q_local * n_chunks * 2 floats;
//   the merge dispatch must follow the partial dispatch with a barrier (it reads partials with glc loads).
// Gates: no LDS, no grid barrier, no buffer_load ... lds.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

namespace vegajet {
namespace spec_flash_dev {

__device__ __forceinline__ float fp16_to_f32(uint16_t h) { return __half2float(__ushort_as_half(h)); }

__device__ __forceinline__ float wave64_sum(float v) {
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x041F));
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x081F));
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x101F));
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x201F));
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x401F));
    v += __shfl_xor(v, 32, 64);
    return v;
}

__device__ __forceinline__ float block32_sum(float v) {
    v += __shfl_xor(v, 16, 32);
    v += __shfl_xor(v, 8, 32);
    v += __shfl_xor(v, 4, 32);
    v += __shfl_xor(v, 2, 32);
    v += __shfl_xor(v, 1, 32);
    return v;
}

__device__ __forceinline__ float block32_max(float v) {
    v = fmaxf(v, __shfl_xor(v, 16, 32));
    v = fmaxf(v, __shfl_xor(v, 8, 32));
    v = fmaxf(v, __shfl_xor(v, 4, 32));
    v = fmaxf(v, __shfl_xor(v, 2, 32));
    v = fmaxf(v, __shfl_xor(v, 1, 32));
    return v;
}

__device__ __forceinline__ float i8_at(uint32_t w, uint32_t k) {
    return static_cast<float>(static_cast<int8_t>((w >> (8u * k)) & 0xFFu));
}

// S3 tree verify: logical serial index t -> physical KV row (j-th set ancestor bit for the tree suffix).
__device__ __forceinline__ uint32_t tree_phys_row(uint32_t t, uint32_t pos0, uint32_t mask) {
    if (t < pos0) return t;
    #pragma nounroll
    for (uint32_t j = t - pos0; j > 0u; --j) mask &= mask - 1u;
    return pos0 + static_cast<uint32_t>(__builtin_ctz(mask));
}

} // namespace spec_flash_dev
} // namespace vegajet

// ============================================================================
// Stage 1: per-chunk online-softmax partial states for all 16 Q heads of one KV head.
// 512 threads = 8 waves; wave w scores heads 2w and 2w+1, a 32-lane half-wave per head, each lane owning 8
// of the 256 dims (one uint2 of a KV row; a block-32 scale is shared by 4 lanes).
// ============================================================================
extern "C" __global__ __launch_bounds__(512, 2)
void vj_spec_flash_partial_kernel(
    const float*   __restrict__ q_all,      // [M, n_q_local, 256]
    const uint8_t* __restrict__ k_cache_q8, // [pos, n_kv_local, 272]
    const uint8_t* __restrict__ v_cache_q8,
    float*         __restrict__ part_o,     // [M, n_q_local, n_chunks, 256]
    float*         __restrict__ part_ml,    // [M, n_q_local, n_chunks, 2]
    uint32_t M,
    uint32_t pos0,
    uint32_t n_q_local,
    uint32_t n_kv_local,
    uint32_t n_chunks,
    uint32_t chunk_len) {
    using namespace vegajet::spec_flash_dev;
    const uint32_t c    = blockIdx.x;
    const uint32_t m    = blockIdx.y;
    const uint32_t kvh  = blockIdx.z;
    if (m >= M || c >= n_chunks || kvh >= n_kv_local || n_q_local != n_kv_local * 16u) return;

    const uint32_t wave = threadIdx.x >> 6;
    const uint32_t lane = threadIdx.x & 63u;
    const uint32_t l32  = lane & 31u;
    const uint32_t h    = kvh * 16u + wave * 2u + (lane >> 5);
    const uint32_t seqlen  = pos0 + m + 1u;
    const uint32_t t_begin = c * chunk_len;
    const uint32_t t_end   = min(t_begin + chunk_len, seqlen);
    const uint64_t st_idx  = (static_cast<uint64_t>(m) * n_q_local + h) * n_chunks + c;

    if (t_begin >= t_end) {  // empty chunk (row m shorter than the grid's span): identity state, o not written
        if (l32 == 0u) {
            __builtin_nontemporal_store(-1e30f, part_ml + st_idx * 2u);
            __builtin_nontemporal_store(0.0f, part_ml + st_idx * 2u + 1u);
        }
        return;
    }

    float q[8];
    {
        const float4* q4 = reinterpret_cast<const float4*>(q_all + (static_cast<uint64_t>(m) * n_q_local + h) * 256u + l32 * 8u);
        const float4 a = q4[0], b = q4[1];
        q[0] = a.x; q[1] = a.y; q[2] = a.z; q[3] = a.w; q[4] = b.x; q[5] = b.y; q[6] = b.z; q[7] = b.w;
    }
    float o[8];
    #pragma unroll
    for (uint32_t i = 0; i < 8u; ++i) o[i] = 0.0f;
    float m_run = -1e30f, l_run = 0.0f;
    const uint32_t row_stride = n_kv_local * 272u;
    const uint32_t qs_off = kvh * 272u + 16u + l32 * 8u;
    const uint32_t sc_off = kvh * 272u + 2u * (l32 >> 2);

    #pragma nounroll
    for (uint32_t t0 = t_begin; t0 < t_end; t0 += 4u) {
        // K and V rows of 4 positions requested together (tail re-reads the last valid row, masked below).
        // KV rows are multi-hit (8 waves of this WG, other verify rows m): plain cached loads.
        uint2 kw[4], vw[4];
        uint16_t ks[4], vs[4];
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) {
            const uint64_t ro = static_cast<uint64_t>(min(t0 + j, t_end - 1u)) * row_stride;
            kw[j] = *reinterpret_cast<const uint2*>(k_cache_q8 + ro + qs_off);
            ks[j] = *reinterpret_cast<const uint16_t*>(k_cache_q8 + ro + sc_off);
        }
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) {
            const uint64_t ro = static_cast<uint64_t>(min(t0 + j, t_end - 1u)) * row_stride;
            vw[j] = *reinterpret_cast<const uint2*>(v_cache_q8 + ro + qs_off);
            vs[j] = *reinterpret_cast<const uint16_t*>(v_cache_q8 + ro + sc_off);
        }
        float s[4];
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) {
            float a0 = 0.0f, a1 = 0.0f;
            #pragma unroll
            for (uint32_t k = 0; k < 4u; k += 2u) {
                a0 = fmaf(q[k], i8_at(kw[j].x, k), a0);
                a1 = fmaf(q[k + 1u], i8_at(kw[j].x, k + 1u), a1);
                a0 = fmaf(q[4u + k], i8_at(kw[j].y, k), a0);
                a1 = fmaf(q[5u + k], i8_at(kw[j].y, k + 1u), a1);
            }
            // The K scale is per block-32 = per 4 lanes: apply it before the cross-lane sum.
            s[j] = block32_sum((a0 + a1) * fp16_to_f32(ks[j])) * 0.0625f;
        }
        float m_new = m_run;
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j)
            if (t0 + j < t_end) m_new = fmaxf(m_new, s[j]);
        const float alpha = __expf(m_run - m_new);
        l_run *= alpha;
        #pragma unroll
        for (uint32_t i = 0; i < 8u; ++i) o[i] *= alpha;
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) {
            const float beta = (t0 + j < t_end) ? __expf(s[j] - m_new) : 0.0f;
            l_run += beta;
            const float bv = beta * fp16_to_f32(vs[j]);
            #pragma unroll
            for (uint32_t k = 0; k < 4u; ++k) {
                o[k]      = fmaf(bv, i8_at(vw[j].x, k), o[k]);
                o[4u + k] = fmaf(bv, i8_at(vw[j].y, k), o[4u + k]);
            }
        }
        m_run = m_new;
    }

    float* po = part_o + st_idx * 256u + l32 * 8u;
    #pragma unroll
    for (uint32_t i = 0; i < 8u; ++i) __builtin_nontemporal_store(o[i], po + i);
    if (l32 == 0u) {
        __builtin_nontemporal_store(m_run, part_ml + st_idx * 2u);
        __builtin_nontemporal_store(l_run, part_ml + st_idx * 2u + 1u);
    }
}

// ============================================================================
// Stage 2: merge chunk states per head + sigmoid gate + block-32 INT8 quant (+ (scale, sum)).
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 4)
void vj_spec_flash_merge_quant_kernel(
    const float* __restrict__ part_o,
    const float* __restrict__ part_ml,
    const float* __restrict__ gate_all,  // [M, n_q_local, 256]
    int8_t*      __restrict__ out_q8,    // [M, n_q_local, 256]
    float2*      __restrict__ out_ds,    // [M, n_q_local, 8]
    uint32_t M,
    uint32_t n_q_local,
    uint32_t n_chunks) {
    using namespace vegajet::spec_flash_dev;
    const uint32_t h = blockIdx.x;
    const uint32_t m = blockIdx.y;
    const uint32_t d = threadIdx.x;
    if (h >= n_q_local || m >= M) return;
    const uint64_t base = (static_cast<uint64_t>(m) * n_q_local + h) * n_chunks;

    // Pass 1: global max over non-empty chunks (glc loads: partials come from the previous dispatch).
    float g_max = -1e30f;
    #pragma unroll 4
    for (uint32_t c = 0; c < n_chunks; ++c) {
        const float mc = __builtin_nontemporal_load(part_ml + (base + c) * 2u);
        const float lc = __builtin_nontemporal_load(part_ml + (base + c) * 2u + 1u);
        if (lc > 0.0f) g_max = fmaxf(g_max, mc);
    }
    // Pass 2: rescaled sums.
    float o = 0.0f, l = 0.0f;
    #pragma unroll 4
    for (uint32_t c = 0; c < n_chunks; ++c) {
        const float mc = __builtin_nontemporal_load(part_ml + (base + c) * 2u);
        const float lc = __builtin_nontemporal_load(part_ml + (base + c) * 2u + 1u);
        if (lc > 0.0f) {
            const float w = __expf(mc - g_max);
            l = fmaf(w, lc, l);
            o = fmaf(w, __builtin_nontemporal_load(part_o + (base + c) * 256u + d), o);
        }
    }

    const uint64_t oi    = (static_cast<uint64_t>(m) * n_q_local + h) * 256u + d;
    const float attn_v   = (l > 0.0f) ? (o / l) : 0.0f;
    const float gv       = gate_all[oi];
    const float y_d      = attn_v * (1.0f / (1.0f + __expf(-gv)));
    const float amax     = block32_max(fabsf(y_d));
    const float bsum     = block32_sum(y_d);
    const float sc       = amax * (1.0f / 127.0f);
    const float inv_d    = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    out_q8[oi] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(y_d * inv_d))));
    if ((d & 31u) == 0u) out_ds[(static_cast<uint64_t>(m) * n_q_local + h) * 8u + (d >> 5)] = make_float2(sc, bsum);
}

// ============================================================================
// S3 tree verify (VEGAJET_V12_TREE_VERIFY, opt-in): ancestor-masked copy of vj_spec_flash_partial_kernel.
// Row m is a tree node: KV at physical row pos0 + m, logical position pos0 + depth[m]. Row m attends over its
// LOGICAL serial context [0, L), L = pos0 + depth[m] + 1: logical t < pos0 is physical row t, logical
// t = pos0 + j is physical row pos0 + (j-th set bit of mask[m]) (ancestors are parent-before-child, so the
// set bits are in depth order; bit m = self). chunk_len is derived per row from L with the serial M = 1
// thresholds (32 <= 1920, 64 <= 3840, 128 <= 7680, else 256), so chunk boundaries, per-chunk arithmetic and the
// merge order equal a serial M = 1 decode of that path. The chunk_len argument is ignored (kept for the ABI).
// Host contract: M <= 32; mask[m] has bit m set, no higher bit, popcount == depth[m] + 1;
//   n_chunks >= max over rows of ceil(L / chunk_len(L)) (larger is fine: extra chunks are identity states).
//   A row that breaks this writes identity states for all chunks (zero attention), never reads out of bounds.
// Merge: the unchanged vj_spec_flash_merge_quant_kernel with the same n_chunks (it skips identity chunks).
// ============================================================================
extern "C" __global__ __launch_bounds__(512, 2)
void vj_tree_flash_partial_kernel(
    const float*   __restrict__ q_all,      // [M, n_q_local, 256]
    const uint8_t* __restrict__ k_cache_q8, // [pos, n_kv_local, 272]
    const uint8_t* __restrict__ v_cache_q8,
    float*         __restrict__ part_o,     // [M, n_q_local, n_chunks, 256]
    float*         __restrict__ part_ml,    // [M, n_q_local, n_chunks, 2]
    uint32_t M,
    uint32_t pos0,
    uint32_t n_q_local,
    uint32_t n_kv_local,
    uint32_t n_chunks,
    uint32_t /*chunk_len: ignored, per row*/,
    const uint32_t* __restrict__ tree_mask,    // [M] ancestor-plus-self bits
    const uint32_t* __restrict__ tree_depth) { // [M] logical depth (anchor = 0)
    using namespace vegajet::spec_flash_dev;
    const uint32_t c    = blockIdx.x;
    const uint32_t m    = blockIdx.y;
    const uint32_t kvh  = blockIdx.z;
    if (m >= M || c >= n_chunks || kvh >= n_kv_local || n_q_local != n_kv_local * 16u) return;

    const uint32_t wave = threadIdx.x >> 6;
    const uint32_t lane = threadIdx.x & 63u;
    const uint32_t l32  = lane & 31u;
    const uint32_t h    = kvh * 16u + wave * 2u + (lane >> 5);
    const uint32_t mask    = tree_mask[m];
    const uint32_t depth   = tree_depth[m];
    const uint32_t seqlen  = pos0 + depth + 1u;  // logical serial span of this row
    const uint32_t chunk_len = (seqlen <= 1920u) ? 32u : ((seqlen <= 3840u) ? 64u : ((seqlen <= 7680u) ? 128u : 256u));
    const bool row_ok = M <= 32u && (mask >> m) == 1u && static_cast<uint32_t>(__builtin_popcount(mask)) == depth + 1u &&
                        (seqlen + chunk_len - 1u) / chunk_len <= n_chunks;
    const uint32_t t_begin = c * chunk_len;
    const uint32_t t_end   = row_ok ? min(t_begin + chunk_len, seqlen) : 0u;
    const uint64_t st_idx  = (static_cast<uint64_t>(m) * n_q_local + h) * n_chunks + c;

    if (t_begin >= t_end) {  // empty chunk (row's logical span ends earlier) or malformed row: identity state
        if (l32 == 0u) {
            __builtin_nontemporal_store(-1e30f, part_ml + st_idx * 2u);
            __builtin_nontemporal_store(0.0f, part_ml + st_idx * 2u + 1u);
        }
        return;
    }

    float q[8];
    {
        const float4* q4 = reinterpret_cast<const float4*>(q_all + (static_cast<uint64_t>(m) * n_q_local + h) * 256u + l32 * 8u);
        const float4 a = q4[0], b = q4[1];
        q[0] = a.x; q[1] = a.y; q[2] = a.z; q[3] = a.w; q[4] = b.x; q[5] = b.y; q[6] = b.z; q[7] = b.w;
    }
    float o[8];
    #pragma unroll
    for (uint32_t i = 0; i < 8u; ++i) o[i] = 0.0f;
    float m_run = -1e30f, l_run = 0.0f;
    const uint32_t row_stride = n_kv_local * 272u;
    const uint32_t qs_off = kvh * 272u + 16u + l32 * 8u;
    const uint32_t sc_off = kvh * 272u + 2u * (l32 >> 2);

    #pragma nounroll
    for (uint32_t t0 = t_begin; t0 < t_end; t0 += 4u) {
        // K and V rows of 4 positions requested together (tail re-reads the last valid row, masked below).
        // KV rows are multi-hit (8 waves of this WG, other verify rows m): plain cached loads.
        uint2 kw[4], vw[4];
        uint16_t ks[4], vs[4];
        uint32_t pr[4];  // physical KV rows of the 4 logical positions (wave-uniform: forced to SGPRs)
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j)
            pr[j] = __builtin_amdgcn_readfirstlane(tree_phys_row(min(t0 + j, t_end - 1u), pos0, mask));
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) {
            const uint64_t ro = static_cast<uint64_t>(pr[j]) * row_stride;
            kw[j] = *reinterpret_cast<const uint2*>(k_cache_q8 + ro + qs_off);
            ks[j] = *reinterpret_cast<const uint16_t*>(k_cache_q8 + ro + sc_off);
        }
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) {
            const uint64_t ro = static_cast<uint64_t>(pr[j]) * row_stride;
            vw[j] = *reinterpret_cast<const uint2*>(v_cache_q8 + ro + qs_off);
            vs[j] = *reinterpret_cast<const uint16_t*>(v_cache_q8 + ro + sc_off);
        }
        float s[4];
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) {
            float a0 = 0.0f, a1 = 0.0f;
            #pragma unroll
            for (uint32_t k = 0; k < 4u; k += 2u) {
                a0 = fmaf(q[k], i8_at(kw[j].x, k), a0);
                a1 = fmaf(q[k + 1u], i8_at(kw[j].x, k + 1u), a1);
                a0 = fmaf(q[4u + k], i8_at(kw[j].y, k), a0);
                a1 = fmaf(q[5u + k], i8_at(kw[j].y, k + 1u), a1);
            }
            // The K scale is per block-32 = per 4 lanes: apply it before the cross-lane sum.
            s[j] = block32_sum((a0 + a1) * fp16_to_f32(ks[j])) * 0.0625f;
        }
        float m_new = m_run;
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j)
            if (t0 + j < t_end) m_new = fmaxf(m_new, s[j]);
        const float alpha = __expf(m_run - m_new);
        l_run *= alpha;
        #pragma unroll
        for (uint32_t i = 0; i < 8u; ++i) o[i] *= alpha;
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) {
            const float beta = (t0 + j < t_end) ? __expf(s[j] - m_new) : 0.0f;
            l_run += beta;
            const float bv = beta * fp16_to_f32(vs[j]);
            #pragma unroll
            for (uint32_t k = 0; k < 4u; ++k) {
                o[k]      = fmaf(bv, i8_at(vw[j].x, k), o[k]);
                o[4u + k] = fmaf(bv, i8_at(vw[j].y, k), o[4u + k]);
            }
        }
        m_run = m_new;
    }

    float* po = part_o + st_idx * 256u + l32 * 8u;
    #pragma unroll
    for (uint32_t i = 0; i < 8u; ++i) __builtin_nontemporal_store(o[i], po + i);
    if (l32 == 0u) {
        __builtin_nontemporal_store(m_run, part_ml + st_idx * 2u);
        __builtin_nontemporal_store(l_run, part_ml + st_idx * 2u + 1u);
    }
}

// ============================================================================
// Engine-wide Model-Agnostic Split-KV FlashDecoding (ANY n_q_local / n_kv_local GQA ratio!)
//   Stage 1: Grid (n_chunks * 256, n_q_local, M), Block 256 (<= 48 VGPRs, 0 B spill)
//     Each WG owns 1 (Q head h, token m, KV chunk c), fills n_q_local * n_chunks WGs (>= 80%-100% CUs)
//     and uses exact 256-lane wave64_sum per position (matching sequential attention to ~1e-7).
//   Stage 2: Grid (n_q_local * 256, M, 1), Block 256 (L2-cached merge + sigmoid gate + Q8 quant)
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 5)
void vj_flash_splitkv_any_partial_kernel(
    const float*   __restrict__ q_all,      // [M, n_q_local, 256]
    const uint8_t* __restrict__ k_cache_q8, // [pos, n_kv_local, 272]
    const uint8_t* __restrict__ v_cache_q8,
    float*         __restrict__ part_o,     // [M, n_q_local, n_chunks, 256]
    float*         __restrict__ part_ml,    // [M, n_q_local, n_chunks, 2]
    uint32_t M,
    uint32_t pos0,
    uint32_t n_q_local,
    uint32_t n_kv_local,
    uint32_t n_chunks,
    uint32_t chunk_len) {
    using namespace vegajet::spec_flash_dev;
    const uint32_t c = blockIdx.x;
    const uint32_t h = blockIdx.y;
    const uint32_t m = blockIdx.z;
    const uint32_t d = threadIdx.x;
    if (c >= n_chunks || h >= n_q_local || m >= M) return;

    const uint32_t seqlen  = pos0 + m + 1u;
    const uint32_t t_begin = c * chunk_len;
    const uint32_t t_end   = min(t_begin + chunk_len, seqlen);
    const uint64_t st_idx  = (static_cast<uint64_t>(m) * n_q_local + h) * n_chunks + c;

    if (t_begin >= t_end) {
        if (d == 0u) {
            part_ml[st_idx * 2u]      = -1e30f;
            part_ml[st_idx * 2u + 1u] = 0.0f;
        }
        return;
    }

    const uint32_t kv_h = (h * n_kv_local) / n_q_local;
    const uint32_t wave = d >> 6;
    const uint32_t lane = d & 63u;
    const uint32_t blk  = d >> 5;
    const float    qv   = q_all[(static_cast<uint64_t>(m) * n_q_local + h) * 256u + d];

    __shared__ float sm_red[16]; // 4 positions x 4 waves
    float m_run = -1e30f, l_run = 0.0f, o_d = 0.0f;
    const uint32_t t_last = t_end - 1u;

    #pragma nounroll
    for (uint32_t t0 = t_begin; t0 < t_end; t0 += 4u) {
        float dots[4];
        float vvs[4];
        #pragma unroll
        for (uint32_t j = 0u; j < 4u; ++j) {
            const uint32_t t   = min(t0 + j, t_last);
            const uint64_t off = (static_cast<uint64_t>(t) * n_kv_local + kv_h) * 272ULL;
            const uint8_t* kc  = k_cache_q8 + off;
            const uint8_t* vc  = v_cache_q8 + off;
            const float ksc    = fp16_to_f32(reinterpret_cast<const uint16_t*>(kc)[blk]);
            const float kv     = ksc * static_cast<float>(reinterpret_cast<const int8_t*>(kc + 16u)[d]);
            const float vsc    = fp16_to_f32(reinterpret_cast<const uint16_t*>(vc)[blk]);
            vvs[j]  = vsc * static_cast<float>(reinterpret_cast<const int8_t*>(vc + 16u)[d]);
            dots[j] = qv * kv;
        }
        #pragma unroll
        for (uint32_t j = 0u; j < 4u; ++j) {
            const float v = wave64_sum(dots[j]);
            if (lane == 0u) sm_red[j * 4u + wave] = v;
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
                o_d   = fmaf(beta, vvs[j], o_d * alpha);
            }
        }
        __syncthreads();
    }

    part_o[st_idx * 256u + d] = o_d;
    if (d == 0u) {
        part_ml[st_idx * 2u]      = m_run;
        part_ml[st_idx * 2u + 1u] = l_run;
    }
}

extern "C" __global__ __launch_bounds__(256, 5)
void vj_flash_splitkv_any_merge_quant_kernel(
    const float* __restrict__ part_o,
    const float* __restrict__ part_ml,
    const float* __restrict__ gate_all,
    int8_t*      __restrict__ out_q8,
    float2*      __restrict__ out_ds,
    uint32_t M,
    uint32_t n_q_local,
    uint32_t n_chunks) {
    using namespace vegajet::spec_flash_dev;
    const uint32_t h = blockIdx.x;
    const uint32_t m = blockIdx.y;
    const uint32_t d = threadIdx.x;
    if (h >= n_q_local || m >= M) return;
    const uint64_t base = (static_cast<uint64_t>(m) * n_q_local + h) * n_chunks;

    float g_max = -1e30f;
    #pragma nounroll
    for (uint32_t c = 0u; c < n_chunks; ++c) {
        const float mc = part_ml[(base + c) * 2u];
        const float lc = part_ml[(base + c) * 2u + 1u];
        if (lc > 0.0f) g_max = fmaxf(g_max, mc);
    }
    float o = 0.0f, l = 0.0f;
    #pragma nounroll
    for (uint32_t c = 0u; c < n_chunks; ++c) {
        const float mc = part_ml[(base + c) * 2u];
        const float lc = part_ml[(base + c) * 2u + 1u];
        if (lc > 0.0f) {
            const float w = __expf(mc - g_max);
            l = fmaf(w, lc, l);
            o = fmaf(w, part_o[(base + c) * 256u + d], o);
        }
    }

    const uint64_t oi   = (static_cast<uint64_t>(m) * n_q_local + h) * 256u + d;
    const float attn_v  = (l > 0.0f) ? (o / l) : 0.0f;
    const float gv      = gate_all[oi];
    const float y_d     = attn_v * (1.0f / (1.0f + __expf(-gv)));
    const float amax    = block32_max(fabsf(y_d));
    const float bsum    = block32_sum(y_d);
    const float sc      = amax * (1.0f / 127.0f);
    const float inv_d   = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    out_q8[oi] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(y_d * inv_d))));
    if ((d & 31u) == 0u) out_ds[(static_cast<uint64_t>(m) * n_q_local + h) * 8u + (d >> 5)] = make_float2(sc, bsum);
}
