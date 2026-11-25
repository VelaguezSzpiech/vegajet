// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Bare-metal gfx906 (MI50) kernels for Ornith-1.5-122B-DFlash2 block-diffusion speculative drafter:
//   - Target hidden-state tap capture (8 layers -> [M, 8*D] contiguous FP32)
//   - Block embedding init (slot 0 from Q8_0 token_embd, slots 1..B-1 from exact BF16->FP32 mask_embedding)
//   - Dynamic grouped causal depthwise conv (K=2, G=16, side=0 with fused Q8 quant, side=1 with fused residual add)
//   - Context KV projection norm + NeoX RoPE (HD=128, theta=1e7) + FP16 sliding-window ring write (W=2048)
//   - Non-causal sliding-window GQA (32Q/8KV, HD=128) over [K_ctx, K_blk] & [V_ctx, V_blk] + fused Q8 quant
//   - SwiGLU + block-32 Q8 quant ([B, 9216]) & standalone FP32 -> block-32 Q8 quant
//   - TP=2 single-pass LM-head Top-16 over output.weight for slots 1..B-1 (M <= 15)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

namespace vegajet {
namespace dflash2_dev {

typedef uint32_t df_u32x4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float fp16_to_f32(uint16_t h) {
    return __half2float(__ushort_as_half(h));
}

__device__ __forceinline__ uint4 ldg_u32x4_stream(const uint4* p) {
    const df_u32x4 v = __builtin_nontemporal_load(reinterpret_cast<const df_u32x4*>(p));
    return make_uint4(v[0], v[1], v[2], v[3]);
}

__device__ __forceinline__ int32_t sdot4(uint32_t a, uint32_t b, int32_t c) {
    return __builtin_amdgcn_sdot4(static_cast<int32_t>(a), static_cast<int32_t>(b), c, false);
}

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
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x041F));
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x081F));
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x101F));
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x201F));
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x401F));
    return v;
}

__device__ __forceinline__ float block32_max(float v) {
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x041F)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x081F)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x101F)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x201F)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x401F)));
    return v;
}

} // namespace dflash2_dev
} // namespace vegajet

// 1. Copy target residual stream h [M, D] into strided tap slot k of [M, n_taps, D].
//    grid = (ceil(D/256), M), block = 256.
extern "C" __global__ __launch_bounds__(256, 4)
void vj_dflash2_tap_copy_kernel(
    const float* __restrict__ h_src,
    float*       __restrict__ taps_dst,
    uint32_t M,
    uint32_t D,
    uint32_t n_taps,
    uint32_t tap_idx) {
    const uint32_t m = blockIdx.y;
    const uint32_t c = blockIdx.x * 256u + threadIdx.x;
    if (m >= M || c >= D) return;
    taps_dst[(static_cast<uint64_t>(m) * n_taps + tap_idx) * D + c] = h_src[static_cast<uint64_t>(m) * D + c];
}

// 2. Standalone FP32 [M, K] -> block-32 INT8 q8 + float2(d, bsum).
//    grid = (ceil(K/256), M), block = 256 (8 blocks of 32 per WG).
extern "C" __global__ __launch_bounds__(256, 4)
void vj_dflash2_quant_q8_kernel(
    const float* __restrict__ x_in,
    int8_t*      __restrict__ q8_out,
    float2*      __restrict__ ds_out,
    uint32_t M,
    uint32_t K) {
    const uint32_t m   = blockIdx.y;
    const uint32_t blk = blockIdx.x * 8u + (threadIdx.x >> 5);
    const uint32_t l32 = threadIdx.x & 31u;
    const uint32_t nb  = K >> 5;
    if (m >= M || blk >= nb) return;
    const uint64_t off = static_cast<uint64_t>(m) * K + (blk << 5) + l32;
    const float v     = x_in[off];
    const float amax  = vegajet::dflash2_dev::block32_max(fabsf(v));
    const float bsum  = vegajet::dflash2_dev::block32_sum(v);
    const float d     = amax * (1.0f / 127.0f);
    const float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    q8_out[off] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(v * inv_d))));
    if (l32 == 0u) ds_out[static_cast<uint64_t>(m) * nb + blk] = make_float2(d, bsum);
}

// 3. Initialize DFlash2 block hidden state [B, D]:
//    slot 0 = dequant(target token_embd[anchor_tok]), slots 1..B-1 = exact FP32 mask_embd_f32[0..D-1].
//    grid = (B, 1), block = 256.
extern "C" __global__ __launch_bounds__(256, 4)
void vj_dflash2_embed_block_kernel(
    const int8_t*   __restrict__ emb_qs,
    const uint16_t* __restrict__ emb_ds,
    const float*    __restrict__ mask_embd_f32,
    float*          __restrict__ h_out,
    uint32_t anchor_tok,
    uint32_t B,
    uint32_t D) {
    const uint32_t m = blockIdx.x;
    if (m >= B) return;
    float* dst = h_out + static_cast<uint64_t>(m) * D;
    if (m == 0u) {
        const int8_t*   qs = emb_qs + static_cast<uint64_t>(anchor_tok) * D;
        const uint16_t* ds = emb_ds + static_cast<uint64_t>(anchor_tok) * (D >> 5);
        for (uint32_t c = threadIdx.x; c < D; c += 256u) {
            dst[c] = static_cast<float>(qs[c]) * vegajet::dflash2_dev::fp16_to_f32(ds[c >> 5]);
        }
    } else {
        for (uint32_t c = threadIdx.x; c < D; c += 256u) {
            dst[c] = mask_embd_f32[c];
        }
    }
}

// 4. Dynamic grouped causal depthwise conv (K=2, G=16, n_groups = D/16 = 192):
//    dyn  [B, 768] = [B, side=2, tap=2, group=192]
//    base [2, 2, D]
//    side == 0: writes block-32 INT8 q8_out & ds_out directly (pre-sublayer)
//    side == 1: h_io[i, c] += conv_out (post-sublayer residual add!)
//    grid = (ceil(D/256), B), block = 256.
extern "C" __global__ __launch_bounds__(256, 4)
void vj_dflash2_dyn_conv_kernel(
    const float* __restrict__ u_in,
    const float* __restrict__ dyn_in,
    const float* __restrict__ base_in,
    float*       __restrict__ h_io,
    int8_t*      __restrict__ q8_out,
    float2*      __restrict__ ds_out,
    uint32_t B,
    uint32_t D,
    uint32_t side) {
    const uint32_t i   = blockIdx.y;
    const uint32_t blk = blockIdx.x * 8u + (threadIdx.x >> 5);
    const uint32_t l32 = threadIdx.x & 31u;
    const uint32_t nb  = D >> 5;
    if (i >= B || blk >= nb) return;

    const uint32_t c  = (blk << 5) + l32;
    const uint32_t g  = c >> 4; // group_size = 16 -> 0..191
    const uint32_t ng = D >> 4; // 192
    const float* dp = dyn_in + static_cast<uint64_t>(i) * (4u * ng) + static_cast<uint64_t>(side * 2u) * ng;
    const float* bp = base_in + static_cast<uint64_t>(side * 2u) * D;

    const float w0 = bp[c]     + dp[g];
    const float w1 = bp[D + c] + dp[ng + g];
    const float u0 = u_in[static_cast<uint64_t>(i) * D + c];
    const float u1 = (i > 0u) ? u_in[static_cast<uint64_t>(i - 1u) * D + c] : 0.0f;
    const float y  = fmaf(w0, u0, w1 * u1);

    if (side == 0u) {
        const float amax  = vegajet::dflash2_dev::block32_max(fabsf(y));
        const float bsum  = vegajet::dflash2_dev::block32_sum(y);
        const float d     = amax * (1.0f / 127.0f);
        const float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        q8_out[static_cast<uint64_t>(i) * D + c] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(y * inv_d))));
        if (l32 == 0u) ds_out[static_cast<uint64_t>(i) * nb + blk] = make_float2(d, bsum);
    } else {
        h_io[static_cast<uint64_t>(i) * D + c] += y;
    }
}

// 5. Context KV inject: RMSNorm(K, k_norm_w) + NeoX RoPE (HD=128, theta=1e7) -> FP16 sliding-window ring [2048, n_kv, 128].
//    kv_f32 [n, 2 * n_kv * 128] (first n_kv*128 is K, second is V).
//    grid = (n_kv, n), block = 128.
extern "C" __global__ __launch_bounds__(128, 4)
void vj_dflash2_kv_inject_kernel(
    const float* __restrict__ kv_f32,
    const float* __restrict__ k_norm_w,
    uint16_t*    __restrict__ k_cache_f16,
    uint16_t*    __restrict__ v_cache_f16,
    uint32_t pos_beg,
    uint32_t n,
    uint32_t n_kv,
    uint32_t win_mask,   // 2047u for sliding_window = 2048
    float    rope_base,
    float    eps) {
    const uint32_t kh = blockIdx.x;
    const uint32_t m  = blockIdx.y;
    const uint32_t d  = threadIdx.x; // 0..127
    if (kh >= n_kv || m >= n) return;

    __shared__ float sm_k[128];
    __shared__ float sm_red[2];

    const uint32_t kv_dim = n_kv * 128u;
    const float* kr = kv_f32 + static_cast<uint64_t>(m) * (2u * kv_dim) + kh * 128u;
    const float* vr = kr + kv_dim;

    const float kv_raw = kr[d];
    const float sq = vegajet::dflash2_dev::wave64_sum(kv_raw * kv_raw);
    if ((d & 63u) == 0u) sm_red[d >> 6] = sq;
    __syncthreads();
    const float inv_r = rsqrtf((sm_red[0] + sm_red[1]) * (1.0f / 128.0f) + eps);
    sm_k[d] = kv_raw * inv_r * k_norm_w[d];
    __syncthreads();

    // NeoX rotate-half RoPE over full 128 dims: pairs (p, p + 64) for p = 0..63.
    const uint32_t p = (d < 64u) ? d : (d - 64u);
    const float theta = static_cast<float>(pos_beg + m) * __powf(rope_base, -2.0f * static_cast<float>(p) / 128.0f);
    float s_th, c_th;
    __sincosf(theta, &s_th, &c_th);
    const float k_rot = (d < 64u) ? (sm_k[p] * c_th - sm_k[p + 64u] * s_th)
                                  : (sm_k[p] * s_th + sm_k[p + 64u] * c_th);

    const uint32_t slot = (pos_beg + m) & win_mask;
    const uint64_t dst  = (static_cast<uint64_t>(slot) * n_kv + kh) * 128ULL + d;
    k_cache_f16[dst] = __half_as_ushort(__float2half(k_rot));
    v_cache_f16[dst] = __half_as_ushort(__float2half(vr[d]));
}

// 6. DFlash2 Non-Causal Sliding-Window GQA (32Q/8KV, HD=128) over [K_ctx(lo..s-1), K_blk(0..B-1)] + Q8 quant.
//    qkv_f32 [B, 6144]: per token [Q(4096) | K_blk(1024) | V_blk(1024)].
//    grid = (n_q=32, B), block = 128.
extern "C" __global__ __launch_bounds__(128, 4)
void vj_dflash2_attn_kernel(
    const float*    __restrict__ qkv_f32,
    const float*    __restrict__ q_norm_w,
    const float*    __restrict__ k_norm_w,
    const uint16_t* __restrict__ k_cache_f16,
    const uint16_t* __restrict__ v_cache_f16,
    int8_t*         __restrict__ out_q8,
    float2*         __restrict__ out_ds,
    uint32_t s_pos,      // anchor position (context KV holds 0 .. s_pos-1)
    uint32_t B,
    uint32_t n_q,
    uint32_t n_kv,
    uint32_t swa_win,    // 2048
    float    rope_base,
    float    eps) {
    const uint32_t qh = blockIdx.x;
    const uint32_t i  = blockIdx.y;
    const uint32_t d  = threadIdx.x; // 0..127
    if (qh >= n_q || i >= B) return;

    __shared__ float sm_buf[128];
    __shared__ float sm_red[2];

    const uint32_t kh      = (qh * n_kv) / n_q;
    const uint32_t q_dim   = n_q * 128u;
    const uint32_t kv_dim  = n_kv * 128u;
    const uint32_t qkv_str = q_dim + 2u * kv_dim;
    const uint32_t p       = (d < 64u) ? d : (d - 64u);
    const float    inv_f   = __powf(rope_base, -2.0f * static_cast<float>(p) / 128.0f);
    const float    q_pos   = static_cast<float>(s_pos + i);

    // 1. Q[i, qh]: RMSNorm + NeoX RoPE -> register qv.
    const float* qr = qkv_f32 + static_cast<uint64_t>(i) * qkv_str + qh * 128u;
    const float q_raw = qr[d];
    const float q_sq  = vegajet::dflash2_dev::wave64_sum(q_raw * q_raw);
    if ((d & 63u) == 0u) sm_red[d >> 6] = q_sq;
    __syncthreads();
    const float q_inv = rsqrtf((sm_red[0] + sm_red[1]) * (1.0f / 128.0f) + eps);
    sm_buf[d] = q_raw * q_inv * q_norm_w[d];
    __syncthreads();
    float s_th, c_th;
    __sincosf(q_pos * inv_f, &s_th, &c_th);
    const float qv = (d < 64u) ? (sm_buf[p] * c_th - sm_buf[p + 64u] * s_th)
                               : (sm_buf[p] * s_th + sm_buf[p + 64u] * c_th);

    // 2. Context keys j in [lo .. s_pos - 1], strict |q_pos - j| < swa_win.
    const uint32_t win_mask = swa_win - 1u;
    const uint32_t lo = (s_pos + i >= swa_win) ? (s_pos + i - swa_win + 1u) : 0u;
    float m_run = -1e30f, l_run = 0.0f, o_d = 0.0f;
    constexpr float kScale = 0.08838834764831845f; // 1/sqrt(128)

    #pragma nounroll
    for (uint32_t j = lo; j < s_pos; ++j) {
        const uint32_t slot = j & win_mask;
        const uint64_t off  = (static_cast<uint64_t>(slot) * n_kv + kh) * 128ULL + d;
        const float kv = vegajet::dflash2_dev::fp16_to_f32(k_cache_f16[off]);
        const float vv = vegajet::dflash2_dev::fp16_to_f32(v_cache_f16[off]);
        const float wdot = vegajet::dflash2_dev::wave64_sum(qv * kv);
        if ((d & 63u) == 0u) sm_red[d >> 6] = wdot;
        __syncthreads();
        const float score = (sm_red[0] + sm_red[1]) * kScale;
        const float m_new = fmaxf(m_run, score);
        const float alpha = __expf(m_run - m_new);
        const float beta  = __expf(score - m_new);
        l_run = l_run * alpha + beta;
        m_run = m_new;
        o_d   = fmaf(beta, vv, o_d * alpha);
        __syncthreads();
    }

    // 3. Bidirectional block keys jb in [0 .. B - 1] (ephemeral, never written to KV cache!).
    #pragma nounroll
    for (uint32_t jb = 0u; jb < B; ++jb) {
        const float* kr = qkv_f32 + static_cast<uint64_t>(jb) * qkv_str + q_dim + kh * 128u;
        const float* vr = kr + kv_dim;
        const float k_raw = kr[d];
        const float k_sq  = vegajet::dflash2_dev::wave64_sum(k_raw * k_raw);
        if ((d & 63u) == 0u) sm_red[d >> 6] = k_sq;
        __syncthreads();
        const float k_inv = rsqrtf((sm_red[0] + sm_red[1]) * (1.0f / 128.0f) + eps);
        sm_buf[d] = k_raw * k_inv * k_norm_w[d];
        __syncthreads();
        float ks_th, kc_th;
        __sincosf(static_cast<float>(s_pos + jb) * inv_f, &ks_th, &kc_th);
        const float kv = (d < 64u) ? (sm_buf[p] * kc_th - sm_buf[p + 64u] * ks_th)
                                   : (sm_buf[p] * ks_th + sm_buf[p + 64u] * kc_th);
        const float wdot = vegajet::dflash2_dev::wave64_sum(qv * kv);
        if ((d & 63u) == 0u) sm_red[d >> 6] = wdot;
        __syncthreads();
        const float score = (sm_red[0] + sm_red[1]) * kScale;
        const float m_new = fmaxf(m_run, score);
        const float alpha = __expf(m_run - m_new);
        const float beta  = __expf(score - m_new);
        l_run = l_run * alpha + beta;
        m_run = m_new;
        o_d   = fmaf(beta, vr[d], o_d * alpha);
        __syncthreads();
    }

    // 4. Normalize and quantize block-32 INT8 directly into out_q8 / out_ds!
    const float y     = (l_run > 0.0f) ? (o_d / l_run) : 0.0f;
    const float amax  = vegajet::dflash2_dev::block32_max(fabsf(y));
    const float bsum  = vegajet::dflash2_dev::block32_sum(y);
    const float sc    = amax * (1.0f / 127.0f);
    const float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    out_q8[static_cast<uint64_t>(i) * q_dim + qh * 128u + d] =
        static_cast<int8_t>(max(-127, min(127, __float2int_rn(y * inv_d))));
    if ((d & 31u) == 0u) out_ds[(static_cast<uint64_t>(i) * n_q + qh) * 4u + (d >> 5)] = make_float2(sc, bsum);
}

// 7. DFlash2 SwiGLU + block-32 INT8 quantize (gu [B, 2*ff] -> q8 [B, ff], ds [B, ff/32]).
//    grid = (ceil(ff/256), B), block = 256.
extern "C" __global__ __launch_bounds__(256, 4)
void vj_dflash2_swiglu_quant_kernel(
    const float* __restrict__ gu_in,
    int8_t*      __restrict__ q8_out,
    float2*      __restrict__ ds_out,
    uint32_t B,
    uint32_t ff) {
    const uint32_t m   = blockIdx.y;
    const uint32_t blk = blockIdx.x * 8u + (threadIdx.x >> 5);
    const uint32_t l32 = threadIdx.x & 31u;
    const uint32_t nb  = ff >> 5;
    if (m >= B || blk >= nb) return;
    const uint32_t c = (blk << 5) + l32;
    const float* gp  = gu_in + static_cast<uint64_t>(m) * (2u * ff);
    const float gv   = gp[c];
    const float uv   = gp[ff + c];
    const float y    = (gv / (1.0f + __expf(-gv))) * uv;
    const float amax  = vegajet::dflash2_dev::block32_max(fabsf(y));
    const float bsum  = vegajet::dflash2_dev::block32_sum(y);
    const float sc    = amax * (1.0f / 127.0f);
    const float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    q8_out[static_cast<uint64_t>(m) * ff + c] =
        static_cast<int8_t>(max(-127, min(127, __float2int_rn(y * inv_d))));
    if (l32 == 0u) ds_out[static_cast<uint64_t>(m) * nb + blk] = make_float2(sc, bsum);
}

// 8. TP=2 LDS-staged LM-head Top-16 over Q8_0 output.weight (or FR-Spec shortlist when tok_map != nullptr)
//    for T <= 15 draft slots, tiling 4 slots per blockIdx.y so x_q8/x_ds live 100% in LDS (16,000 B -> 4 WGs/CU!).
//    grid = (n_blocks, ceil(T/4)), block = 256 (4 waves/WG).
//    Emits per-block top-16 into blk_val/blk_id [T, gridDim.x, 16] for fast host merge across 2 GPUs.
extern "C" __global__ __launch_bounds__(256, 4)
void vj_dflash2_vocab_topk16_kernel(
    const int8_t*   __restrict__ w_qs,
    const uint16_t* __restrict__ w_ds,
    const int8_t*   __restrict__ x_q8,
    const float2*   __restrict__ x_ds,
    const int32_t*  __restrict__ tok_map,
    float*          __restrict__ blk_val,
    int32_t*        __restrict__ blk_id,
    uint32_t local_rows,
    uint32_t cols,
    uint32_t row_off,
    uint32_t row_stride_bytes,
    uint32_t T) {
    const uint32_t tid  = threadIdx.x;
    const uint32_t wave = tid >> 6;
    const uint32_t lane = tid & 63u;
    const uint32_t t0   = blockIdx.y * 4u;
    if (t0 >= T || cols != 3072u) return;
    const uint32_t tc   = min(4u, T - t0);
    constexpr uint32_t kN16 = 192u;
    constexpr uint32_t kNB  = 96u;

    // 12,288 + 1,536 + 1,024 + 1,024 + 64 + 64 = 16,000 B <= 16,384 B (4 WGs/CU!)
    __shared__ uint4    sm_x[4u * kN16];
    __shared__ float    sm_xd[4u * kNB];
    __shared__ float    sm_v[4u * 4u * 16u];
    __shared__ int32_t  sm_i[4u * 4u * 16u];
    __shared__ float    sm_minv[4u * 4u];
    __shared__ uint32_t sm_mini[4u * 4u];

    const uint4*  xq4_base = reinterpret_cast<const uint4*>(x_q8) + static_cast<uint64_t>(t0) * kN16;
    const float2* xds_base = x_ds + static_cast<uint64_t>(t0) * kNB;
    for (uint32_t k = tid; k < tc * kN16; k += 256u) sm_x[k] = xq4_base[k];
    for (uint32_t k = tid; k < tc * kNB;  k += 256u) sm_xd[k] = xds_base[k].x;
    if (tid < tc * 64u) {
        sm_v[tid] = -1e30f;
        sm_i[tid] = -1;
    }
    if (tid < tc * 4u) {
        sm_minv[tid] = -1e30f;
        sm_mini[tid] = 0u;
    }
    __syncthreads();

    for (uint32_t row = blockIdx.x * 4u + wave; row < local_rows; row += gridDim.x * 4u) {
        const uint4*    w4 = reinterpret_cast<const uint4*>(w_qs + static_cast<uint64_t>(row) * row_stride_bytes);
        const uint16_t* ds = w_ds + static_cast<uint64_t>(row) * kNB;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        #pragma unroll
        for (uint32_t step = 0u; step < 3u; ++step) {
            const uint32_t i   = lane + step * 64u;
            const uint4    wv  = vegajet::dflash2_dev::ldg_u32x4_stream(w4 + i);
            const uint32_t bi  = i >> 1;
            const float    wdi = vegajet::dflash2_dev::fp16_to_f32(ds[bi]);
            auto mac_t = [&](uint32_t tl, float& acc) {
                const uint4 xv = sm_x[tl * kN16 + i];
                int32_t dp = vegajet::dflash2_dev::sdot4(wv.w, xv.w,
                             vegajet::dflash2_dev::sdot4(wv.z, xv.z,
                             vegajet::dflash2_dev::sdot4(wv.y, xv.y,
                             vegajet::dflash2_dev::sdot4(wv.x, xv.x, 0))));
                acc = fmaf(wdi * sm_xd[tl * kNB + bi], static_cast<float>(dp), acc);
            };
            mac_t(0u, a0);
            if (tc > 1u) mac_t(1u, a1);
            if (tc > 2u) mac_t(2u, a2);
            if (tc > 3u) mac_t(3u, a3);
        }
        const int32_t gid = tok_map ? tok_map[row] : static_cast<int32_t>(row_off + row);
        auto push_t = [&](uint32_t tl, float acc) {
            const float s = vegajet::dflash2_dev::wave64_sum(acc);
            const uint32_t widx = tl * 4u + wave;
            if (lane == 0u && s > sm_minv[widx]) {
                const uint32_t base = widx * 16u;
                const uint32_t pos  = sm_mini[widx];
                sm_v[base + pos] = s;
                sm_i[base + pos] = gid;
                float    mv = sm_v[base];
                uint32_t mi = 0u;
                #pragma unroll
                for (uint32_t k = 1u; k < 16u; ++k) {
                    const float vk = sm_v[base + k];
                    if (vk < mv) { mv = vk; mi = k; }
                }
                sm_minv[widx] = mv;
                sm_mini[widx] = mi;
            }
        };
        push_t(0u, a0);
        if (tc > 1u) push_t(1u, a1);
        if (tc > 2u) push_t(2u, a2);
        if (tc > 3u) push_t(3u, a3);
    }
    __syncthreads();

    // Merge 4 waves' 64 entries -> top-16 per slot tl < tc (1 thread per slot).
    if (tid < tc) {
        const uint32_t tl   = tid;
        const uint32_t base = tl * 64u;
        float*   out_v = blk_val + (static_cast<uint64_t>(t0 + tl) * gridDim.x + blockIdx.x) * 16ULL;
        int32_t* out_i = blk_id  + (static_cast<uint64_t>(t0 + tl) * gridDim.x + blockIdx.x) * 16ULL;
        #pragma nounroll
        for (uint32_t k = 0u; k < 16u; ++k) {
            float    bv = -1e30f;
            int32_t  bi = -1;
            uint32_t bj = 0u;
            #pragma nounroll
            for (uint32_t j = 0u; j < 64u; ++j) {
                const float vj = sm_v[base + j];
                if (vj > bv) { bv = vj; bi = sm_i[base + j]; bj = j; }
            }
            out_v[k] = bv;
            out_i[k] = bi;
            sm_v[base + bj] = -1e30f;
        }
    }
}
