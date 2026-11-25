// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// VegaJet Big Lever 3: batched GDN prefill kernels with >= 128 WGs per GPU (the current WY prefill kernel runs
// 32 WGs, one per V head). Two candidates for A/B; both write the same outputs as ornith_gdn_recur_quant_kernel
// (kernel 5b): out_q8[m][nv][128] INT8 and out_ds[m][nv][4] = {d, sum} of the gated RMSNorm output, and leave the
// final state S[hv][128][128] (row i = key dim, column j = value dim) in the ring slot.
//
//  (b) vj_prefill_gdn_splitv_batch_kernel   -- register-cached sequential Split-V recurrence over a batch.
//      4 WGs per head (nv*4 = 128 WGs), each owns 32 value columns, a thread owns 16 rows of one column in
//      registers for the whole batch. Per token the arithmetic and the reduction order are exactly those of 5b
//      (8 row groups x fmaf chain over 16 rows, partials summed in group order), so S, fo and the outputs are
//      bit-identical to 5b. The head-wide RMS only feeds the output, never the state, so it is deferred: per token
//      the WG stores its 32 fo values (FP32, own columns) and 16 pair-squares (cross-WG), and after the batch one
//      grid barrier lets every WG normalise its own columns for all tokens with the same wave64 tree as 5b.
//      One grid barrier per DISPATCH instead of one per token (the M = 1 split-V kernel's per-token barrier would
//      cost ~M x barrier latency).
//      Grid must be exactly nv * 4 WGs and all resident (grid barrier): 128 WGs <= 4/CU x 60.
//
//  (a) vj_prefill_gdn_wy_split_kernel       -- chunked WY / Neumann form, T = prod_k (I + A^(2^k)), zero
//      SOLVE_TRI, C = 16, each head split over 4 WGs by value columns (128 WGs). Every WG
//      recomputes the head's small C x C chunk matrices (T, QK, decay ratios) redundantly and owns 32 value
//      columns of S, dv, intra-chunk output and the state update; the RMS is deferred to
//      vj_prefill_gdn_normq_kernel (fo / sq scratch shared with (b)).
//      Numerically this is the WY reformulation (same math as ornith_gdn_wy_prefill_quant_kernel, different
//      rounding from 5b); it is offered for the speed A/B only.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

namespace vegajet {
namespace pgdn {

__device__ __forceinline__ float wave64_sum(float v) {   // identical tree to ornith_dev::wave64_sum
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x041F));
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x081F));
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x101F));
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x201F));
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x401F));
    v += __shfl_xor(v, 32, 64);
    return v;
}

// Opaque copy of a per-thread value: an empty asm register fence (no instruction is emitted) so that each phase
// of the chunk loop recomputes its own thread-derived LDS / global offsets instead of the compiler hoisting ~30
// of them out of the loop and holding them in VGPRs for the whole kernel (the M = 1 split-V kernel's note).
__device__ __forceinline__ uint32_t opaque_u32(uint32_t v) {
    asm volatile("" : "+v"(v));
    return v;
}

__device__ __forceinline__ void grid_barrier(uint32_t* bar, uint32_t n_wgs, uint32_t sense) {
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0u) {
        const uint32_t arr = atomicAdd(&bar[0], 1u) + 1u;
        if (arr == n_wgs) {
            __atomic_store_n(&bar[0], 0u, __ATOMIC_RELEASE);
            __atomic_store_n(&bar[16], sense, __ATOMIC_RELEASE);
        } else {
            for (uint32_t spin = 0u; spin < 5000000u; ++spin) {
                if (__atomic_load_n(&bar[16], __ATOMIC_ACQUIRE) == sense) break;
                __builtin_amdgcn_s_sleep(1);
            }
        }
        asm volatile("buffer_wbinvl1\n" ::: "memory");
    }
    __syncthreads();
}

// Deferred gated RMSNorm + block-32 quant for the 32 columns [col0, col0+32) of head hv, all M tokens.
// Wave w handles tokens w, w+4, ...: the 64 lanes reduce the head's 64 pair-squares with the 5b wave64 tree,
// lanes 0..15 then finish 2 columns each with 5b's 16-lane amax / sum trees.
__device__ __forceinline__ void norm_quant_part(const float* __restrict__ fo_scratch, const float* __restrict__ sq_scratch,
                                                const float* __restrict__ qkv_gab_in, const float* __restrict__ ssm_norm_w,
                                                int8_t* __restrict__ out_q8, float2* __restrict__ out_ds,
                                                uint32_t M, uint32_t nv, uint32_t hv, uint32_t part,
                                                uint32_t stride_in, uint32_t z_off, float eps) {
    const uint32_t wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    const uint32_t lane = threadIdx.x & 63u;
    const uint32_t l16  = lane & 15u;
    const uint32_t col0 = part * 32u;
    const uint32_t j0   = col0 + l16 * 2u;
    const float w0 = ssm_norm_w[j0], w1 = ssm_norm_w[j0 + 1u];
    #pragma nounroll
    for (uint32_t m = wave; m < M; m += 4u) {
        const uint64_t mh = static_cast<uint64_t>(m) * nv + hv;
        const float lane_sq  = sq_scratch[mh * 64u + lane];
        const float total_sq = wave64_sum(lane_sq);
        const float inv_r = rsqrtf(total_sq * (1.0f / 128.0f) + eps);
        const float fo0 = fo_scratch[mh * 128u + j0];
        const float fo1 = fo_scratch[mh * 128u + j0 + 1u];
        const float* zp = qkv_gab_in + static_cast<uint64_t>(m) * stride_in + z_off + hv * 128u;
        const float z0 = zp[j0], z1 = zp[j0 + 1u];
        const float y0 = (fo0 * inv_r * w0) * (z0 / (1.0f + __expf(-z0)));
        const float y1 = (fo1 * inv_r * w1) * (z1 / (1.0f + __expf(-z1)));

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
        if (lane < 16u) {
            int8_t* oq = out_q8 + mh * 128u;
            oq[j0]      = static_cast<int8_t>(max(-127, min(127, __float2int_rn(y0 * inv_d))));
            oq[j0 + 1u] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(y1 * inv_d))));
            if (lane == 0u) out_ds[mh * 4u + part] = make_float2(d, bsum);
        }
    }
}

// Per token, the 32 fo values of this part go to fo_scratch and the 16 pair-squares (fo[2l]^2 + fo[2l+1]^2, the
// same expression as 5b) to sq_scratch[m][hv][part*16 + l]. Called by threads 0..31 (one wave half).
__device__ __forceinline__ void emit_fo(float fo, float* __restrict__ fo_scratch, float* __restrict__ sq_scratch,
                                        uint64_t mh, uint32_t part, uint32_t c) {
    fo_scratch[mh * 128u + part * 32u + c] = fo;
    const float fo1 = __shfl_down(fo, 1, 64);
    const float fo0 = fo;
    if ((c & 1u) == 0u) sq_scratch[mh * 64u + part * 16u + (c >> 1)] = fo0 * fo0 + fo1 * fo1;
}

} // namespace pgdn
} // namespace vegajet

// ============================================================================
// (c) Parallel-over-tokens batched GDN Conv1D + SiLU + Q/K L2-Norm + Alpha/Beta
//     Grid: (2*nk + nv + 1, ceil(M / 16)) = (49, 32) = 1,568 WGs at M=512 (vs 49 WGs sequential!).
//     Block: 128. Bit-identical to ornith_gdn_conv_l2_gate_kernel (use_ring == 0 prefill path).
// ============================================================================
extern "C" __global__ __launch_bounds__(128, 6)
void vj_prefill_gdn_conv_l2_gate_kernel(
    const float* __restrict__ qkv_gab_in,
    const float* __restrict__ conv_w,
    float*       __restrict__ conv_ring,
    float*       __restrict__ conv_out,
    const float* __restrict__ dt_bias,
    const float* __restrict__ ssm_a,
    float*       __restrict__ decay_out,
    float*       __restrict__ beta_out,
    uint32_t M,
    uint32_t nk,
    uint32_t nv,
    uint32_t conv_dim,
    uint32_t stride_in,
    uint32_t alpha_off,
    uint32_t beta_off,
    uint32_t ring_head,
    uint32_t ring_out,
    float    eps) {
    constexpr uint32_t kTileM = 16u;
    const uint32_t grp = blockIdx.x;
    const uint32_t m0  = blockIdx.y * kTileM;
    const uint32_t tid = threadIdx.x; // 0..127
    const uint32_t n_head_wgs = 2u * nk + nv; // 48
    if (m0 >= M) return;
    const uint32_t mc = min(kTileM, M - m0);

    if (grp < n_head_wgs) {
        const uint32_t ch = grp * 128u + tid;
        const float4 w = reinterpret_cast<const float4*>(conv_w)[ch];
        const uint64_t slot_stride = 3ULL * conv_dim;
        const float* r_in = conv_ring + static_cast<uint64_t>(ring_head) * slot_stride;
        float s0 = (m0 < 3u) ? r_in[static_cast<uint64_t>(m0) * conv_dim + ch]
                             : qkv_gab_in[static_cast<uint64_t>(m0 - 3u) * stride_in + ch];
        float s1 = (m0 + 1u < 3u) ? r_in[static_cast<uint64_t>(m0 + 1u) * conv_dim + ch]
                                  : qkv_gab_in[static_cast<uint64_t>(m0 - 2u) * stride_in + ch];
        float s2 = (m0 + 2u < 3u) ? r_in[static_cast<uint64_t>(m0 + 2u) * conv_dim + ch]
                                  : qkv_gab_in[static_cast<uint64_t>(m0 - 1u) * stride_in + ch];
        __shared__ float sm_sq[2];

        #pragma nounroll
        for (uint32_t t = 0u; t < mc; ++t) {
            const uint32_t m = m0 + t;
            const float x = qkv_gab_in[static_cast<uint64_t>(m) * stride_in + ch];
            const float acc = fmaf(w.x, s0, fmaf(w.y, s1, fmaf(w.z, s2, w.w * x)));
            float y = acc / (1.0f + __expf(-acc));
            s0 = s1; s1 = s2; s2 = x;
            if (grp < 2u * nk) {
                const float sq = vegajet::pgdn::wave64_sum(y * y);
                if ((tid & 63u) == 0u) sm_sq[tid >> 6] = sq;
                __syncthreads();
                float inv_n = rsqrtf(sm_sq[0] + sm_sq[1] + eps);
                if (grp < nk) inv_n *= 0.08838834764831845f; // 1/sqrt(128)
                y *= inv_n;
                __syncthreads();
            }
            conv_out[static_cast<uint64_t>(m) * conv_dim + ch] = y;
        }
        if (m0 + kTileM >= M) {
            float* r_out = conv_ring + static_cast<uint64_t>(ring_out) * slot_stride;
            r_out[0ULL * conv_dim + ch] = s0;
            r_out[1ULL * conv_dim + ch] = s1;
            r_out[2ULL * conv_dim + ch] = s2;
        }
    } else if (grp == n_head_wgs && tid < nv) {
        const float dtb = dt_bias[tid];
        const float sa  = ssm_a[tid];
        #pragma nounroll
        for (uint32_t t = 0u; t < mc; ++t) {
            const uint32_t m = m0 + t;
            const float* row = qkv_gab_in + static_cast<uint64_t>(m) * stride_in;
            const float a  = row[alpha_off + tid] + dtb;
            const float sp = (a > 20.0f) ? a : log1pf(__expf(a));
            const uint64_t idx = static_cast<uint64_t>(m) * nv + tid;
            decay_out[idx] = __expf(sp * sa);
            beta_out[idx]  = 1.0f / (1.0f + __expf(-row[beta_off + tid]));
        }
    }
}

// ============================================================================
// (b) Split-V batch recurrence.
// Kernargs: ssm_ring, conv_qkv, qkv_gab_in, decay_in, beta_in, ssm_norm_w, out_q8, out_ds, bar, fo_scratch,
//           sq_scratch, M, nk, nv, conv_dim, stride_in, z_off, ring_head, use_ring (must be 0), sense, eps, fuse_norm
//   fo_scratch: float [M][nv][128]   sq_scratch: float [M][nv][64]   bar: 2 x 64 B (zeroed once; sense-reversing)
//   fuse_norm = 1: one grid barrier then the gated RMSNorm + quant in-kernel (all nv*4 WGs must be resident:
//                  52 VGPRs, 2.3 KiB LDS -> 4 WGs/CU, 128 <= 240);
//   fuse_norm = 0: return after the recurrence; launch vj_prefill_gdn_normq_kernel (same grid) afterwards.
// Grid: exactly nv * 4 WGs (128 for Ornith), block 256.
//   enqueue_kernel(g, k, nv * 4 * 256, 1, 1, 256, 1, 1, ...)
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 4)
void vj_prefill_gdn_splitv_batch_kernel(
    float*       __restrict__ ssm_ring,
    const float* __restrict__ conv_qkv,
    const float* __restrict__ qkv_gab_in,
    const float* __restrict__ decay_in,
    const float* __restrict__ beta_in,
    const float* __restrict__ ssm_norm_w,
    int8_t*      __restrict__ out_q8,
    float2*      __restrict__ out_ds,
    uint32_t*    __restrict__ bar,
    float*       __restrict__ fo_scratch,
    float*       __restrict__ sq_scratch,
    uint32_t M,
    uint32_t nk,
    uint32_t nv,
    uint32_t conv_dim,
    uint32_t stride_in,
    uint32_t z_off,
    uint32_t ring_head,
    uint32_t use_ring,
    uint32_t sense,
    float    eps,
    uint32_t fuse_norm) {
    const uint32_t hv   = blockIdx.x >> 2;
    const uint32_t part = blockIdx.x & 3u;
    const uint32_t tid  = threadIdx.x;
    const uint32_t c    = tid & 31u;
    const uint32_t rg   = tid >> 5;
    const uint32_t col0 = part * 32u;
    const uint32_t hk   = hv % nk;
    const uint64_t head_elems = 128ULL * 128ULL;
    const uint64_t slot_elems = static_cast<uint64_t>(nv) * head_elems;

    // Stage T=8 tokens of q, k, v, decay, beta in LDS (~11.3 KiB <= 16 KiB -> 4 WGs/CU) so the inner
    // recurrence loop has zero VMEM stalls and only 2 __syncthreads() barriers per token (fusing o(m) & p(m+1)).
    constexpr uint32_t kStageT = 8u;
    __shared__ float sm_q[kStageT * 128u];     // 4,096 B
    __shared__ float sm_k[kStageT * 128u];     // 4,096 B
    __shared__ float sm_v[kStageT * 32u];      // 1,024 B
    __shared__ float sm_g[kStageT];            //    32 B
    __shared__ float sm_b[kStageT];            //    32 B
    __shared__ float sm_red_p[8 * 32];         // 1,024 B
    __shared__ float sm_red_o[8 * 32];         // 1,024 B
    __shared__ float sm_dv[32];                //   128 B

    const uint32_t i_beg = rg * 16u;
    float* S_thr = ssm_ring + static_cast<uint64_t>(ring_head) * slot_elems + static_cast<uint64_t>(hv) * head_elems
                 + (i_beg + 8u) * 128u + col0 + c;
    float s_reg[16];
    #pragma unroll
    for (uint32_t r = 0u; r < 16u; ++r) s_reg[r] = S_thr[(static_cast<int32_t>(r) - 8) * 128];

    const uint32_t ld_off = (tid < 128u) ? (hk * 128u + tid) : ((nk + hk) * 128u + (tid - 128u));
    const uint32_t v_off  = (2u * nk + hv) * 128u + col0 + c;

    #pragma nounroll
    for (uint32_t m0 = 0u; m0 < M; m0 += kStageT) {
        const uint32_t tc = min(kStageT, M - m0);
        // Cooperatively stage all tc tokens of q, k, v, g_exp, beta in one burst.
        #pragma unroll
        for (uint32_t t = 0u; t < kStageT; ++t) {
            if (t < tc) {
                const float* base = conv_qkv + static_cast<uint64_t>(m0 + t) * conv_dim;
                const float qkv = base[ld_off];
                if (tid < 128u) sm_q[t * 128u + tid] = qkv;
                else            sm_k[t * 128u + (tid - 128u)] = qkv;
            }
        }
        if (tid < tc * 32u) {
            const uint32_t t = tid >> 5;
            const uint32_t l = tid & 31u;
            sm_v[tid] = conv_qkv[static_cast<uint64_t>(m0 + t) * conv_dim + (2u * nk + hv) * 128u + col0 + l];
        }
        if (tid < tc) {
            const uint64_t mh = static_cast<uint64_t>(m0 + tid) * nv + hv;
            sm_g[tid] = decay_in[mh];
            sm_b[tid] = beta_in[mh];
        }
        __syncthreads();

        // Prime p for t = 0
        {
            const float* kp = sm_k;
            float p = 0.0f;
            #pragma unroll
            for (uint32_t r = 0u; r < 16u; ++r) p = fmaf(s_reg[r], kp[i_beg + r], p);
            sm_red_p[rg * 32u + c] = p;
        }
        __syncthreads();

        #pragma nounroll
        for (uint32_t t = 0u; t < tc; ++t) {
            const float g_exp = sm_g[t];
            const float beta  = sm_b[t];
            if (tid < 32u) {
                const float pred = sm_red_p[tid] + sm_red_p[32u + tid] + sm_red_p[64u + tid] + sm_red_p[96u + tid]
                                 + sm_red_p[128u + tid] + sm_red_p[160u + tid] + sm_red_p[192u + tid] + sm_red_p[224u + tid];
                sm_dv[tid] = beta * (sm_v[t * 32u + tid] - g_exp * pred);
            }
            __syncthreads();

            const float dv = sm_dv[c];
            const float* kp = sm_k + t * 128u;
            const float* qp = sm_q + t * 128u;
            float o = 0.0f;
            #pragma unroll
            for (uint32_t r = 0u; r < 16u; ++r) {
                s_reg[r] = fmaf(kp[i_beg + r], dv, g_exp * s_reg[r]);
                o = fmaf(s_reg[r], qp[i_beg + r], o);
                if ((r & 3u) == 3u) __builtin_amdgcn_sched_barrier(0);
            }
            sm_red_o[rg * 32u + c] = o;
            // Overlap next token's p(t+1) BEFORE the barrier so o(t) and p(t+1) share ONE barrier!
            if (t + 1u < tc) {
                const float* kpn = sm_k + (t + 1u) * 128u;
                float pn = 0.0f;
                #pragma unroll
                for (uint32_t r = 0u; r < 16u; ++r) pn = fmaf(s_reg[r], kpn[i_beg + r], pn);
                sm_red_p[rg * 32u + c] = pn;
            }
            __syncthreads();

            if (tid < 32u) {
                const float fo = sm_red_o[tid] + sm_red_o[32u + tid] + sm_red_o[64u + tid] + sm_red_o[96u + tid]
                               + sm_red_o[128u + tid] + sm_red_o[160u + tid] + sm_red_o[192u + tid] + sm_red_o[224u + tid];
                const uint64_t mh = static_cast<uint64_t>(m0 + t) * nv + hv;
                vegajet::pgdn::emit_fo(fo, fo_scratch, sq_scratch, mh, part, tid);
            }
        }
        __syncthreads(); // ensure sm_red_o / sm_q / sm_k reads complete before next chunk staging
    }
    (void)v_off;

    (void)use_ring;   // prefill only: per-token ring snapshots (speculative verify) stay on kernel 5b
    #pragma unroll
    for (uint32_t r = 0u; r < 16u; ++r) S_thr[(static_cast<int32_t>(r) - 8) * 128] = s_reg[r];

    if (fuse_norm == 0u) return;   // the main thread runs vj_prefill_gdn_normq_kernel next (no residency rule)
    __threadfence_system();
    vegajet::pgdn::grid_barrier(bar, gridDim.x, sense);
    vegajet::pgdn::norm_quant_part(fo_scratch, sq_scratch, qkv_gab_in, ssm_norm_w, out_q8, out_ds,
                                   M, nv, hv, part, stride_in, z_off, eps);
}

// ============================================================================
// Deferred gated RMSNorm + block-32 quant (pairs with fuse_norm = 0 of (b) and with (a)).
// Kernargs: fo_scratch, sq_scratch, qkv_gab_in, ssm_norm_w, out_q8, out_ds, M, nv, stride_in, z_off, eps
// Grid: nv * 4 WGs (WG = (head, 32-column part)), block 256.
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 4)
void vj_prefill_gdn_normq_kernel(
    const float* __restrict__ fo_scratch,
    const float* __restrict__ sq_scratch,
    const float* __restrict__ qkv_gab_in,
    const float* __restrict__ ssm_norm_w,
    int8_t*      __restrict__ out_q8,
    float2*      __restrict__ out_ds,
    uint32_t M,
    uint32_t nv,
    uint32_t stride_in,
    uint32_t z_off,
    float    eps) {
    vegajet::pgdn::norm_quant_part(fo_scratch, sq_scratch, qkv_gab_in, ssm_norm_w, out_q8, out_ds,
                                   M, nv, blockIdx.x >> 2, blockIdx.x & 3u, stride_in, z_off, eps);
}

// ============================================================================
// (a) Chunked WY / Neumann prefill, each head split over 4 WGs by value columns (nv * 4 = 128 WGs), C = 16.
//   Per chunk of C tokens (cum_g[t] = prod_{u <= t} g_u):
//     A[r][c]  = -beta_r (cum_g[r] / cum_g[c]) (k_r . k_c)   (r > c),  T = (I - A)^-1 = (I+A)(I+A^2)(I+A^4)(I+A^8)
//     QK[r][c] =  (cum_g[r] / cum_g[c]) (q_r . k_c)            (r >= c)
//     rhs[t][j] = beta_t (v_t[j] - cum_g[t] (k_t . S[:, j])),  dv = T rhs
//     O[t][j]   = cum_g[t] (q_t . S[:, j]) + sum_{u <= t} QK[t][u] dv[u][j]
//     S[i][j]   = cum_g[C-1] S[i][j] + sum_u (cum_g[C-1] / cum_g[u]) k_u[i] dv[u][j]
//   Same algebra and the same per-entry expressions as ornith_gdn_wy_prefill_quant_kernel; the C x C matrices are
//   recomputed by each of a head's 4 WGs (32 K MACs each), S lives in registers (thread = 16 rows x 1 column).
//   C = 32 is not offered: K and Q chunks alone would be 32 KiB of LDS (1 WG/CU).
//   No grid barrier: the RMS is deferred to vj_prefill_gdn_normq_kernel (same fo / sq scratch as (b)).
// Kernargs: S_all (state, [nv][128][128]), conv_qkv, decay_in, beta_in, fo_scratch, sq_scratch, M, nk, nv, conv_dim
// Grid: nv * 4 WGs, block 256. LDS 30,848 B -> 2 WGs/CU (120 resident; no barrier, so 128 WGs is fine).
// Measured: 37 VGPRs (opaque per-phase thread ids), 0 spill.
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 2)
void vj_prefill_gdn_wy_split_kernel(
    float*       __restrict__ S_all,
    const float* __restrict__ conv_qkv,
    const float* __restrict__ decay_in,
    const float* __restrict__ beta_in,
    float*       __restrict__ fo_scratch,
    float*       __restrict__ sq_scratch,
    uint32_t M,
    uint32_t nk,
    uint32_t nv,
    uint32_t conv_dim) {
    constexpr uint32_t C = 16u;
    const uint32_t hv   = blockIdx.x >> 2;
    const uint32_t part = blockIdx.x & 3u;
    const uint32_t tid  = threadIdx.x;
    const uint32_t col0 = part * 32u;
    const uint32_t hk   = hv % nk;

    __shared__ float sm_k[C * 128u];          // 8 KiB
    __shared__ float sm_q[C * 128u];          // 8 KiB
    __shared__ float sm_T[C * C];             // 1 KiB
    __shared__ float sm_QK[C * C];            // 1 KiB
    __shared__ float sm_u[2u * 4u * 8u * 32u];// 8 KiB union: Neumann P, tmp | row-group partials | dv
    __shared__ float sm_rhs[C * 32u];         // 2 KiB (rhs, later dv)
    __shared__ float sm_qs[C * 32u];          // 2 KiB
    __shared__ float sm_cg[C];
    __shared__ float sm_beta[C];

    float s_reg[16];
    {
        float* S_thr = S_all + static_cast<uint64_t>(hv) * 16384ULL + ((tid >> 5) * 16u + 8u) * 128u + col0 + (tid & 31u);
        #pragma unroll
        for (uint32_t r = 0u; r < 16u; ++r) s_reg[r] = S_thr[(static_cast<int32_t>(r) - 8) * 128];
    }

    #pragma nounroll
    for (uint32_t m0 = 0u; m0 < M; m0 += C) {
        const uint32_t clen = min(C, M - m0);
        if (tid == 0u) {
            float acc = 1.0f;
            for (uint32_t t = 0u; t < C; ++t) {
                if (t < clen) acc *= decay_in[static_cast<uint64_t>(m0 + t) * nv + hv];
                sm_cg[t] = acc;
            }
        }
        {
            const uint32_t tq = vegajet::pgdn::opaque_u32(threadIdx.x);
            if (tq < C) sm_beta[tq] = (tq < clen) ? beta_in[static_cast<uint64_t>(m0 + tq) * nv + hv] : 0.0f;
        }
        #pragma unroll
        for (uint32_t e = vegajet::pgdn::opaque_u32(threadIdx.x); e < C * 128u; e += 256u) {
            const uint32_t t = e >> 7, d = e & 127u;
            const float* br = conv_qkv + static_cast<uint64_t>(m0 + min(t, clen - 1u)) * conv_dim;
            sm_q[e] = (t < clen) ? br[hk * 128u + d] : 0.0f;
            sm_k[e] = (t < clen) ? br[(nk + hk) * 128u + d] : 0.0f;
        }
        __syncthreads();

        // C x C matrices (redundant per WG): thread (r, cc) = (tid >> 4, tid & 15).
        float* sm_P = sm_u;
        float* sm_tmp = sm_u + C * C;
        {
            const uint32_t tq = vegajet::pgdn::opaque_u32(threadIdx.x);
            const uint32_t r = tq >> 4, cc = tq & 15u;
            const bool valid = (r < clen && cc < clen);
            const float ratio = (valid && r >= cc) ? (sm_cg[r] / fmaxf(sm_cg[cc], 1e-30f)) : 0.0f;
            float kk = 0.0f, qk = 0.0f;
            if (valid && r >= cc) {
                #pragma nounroll
                for (uint32_t d = 0u; d < 128u; ++d) {
                    const float kcv = sm_k[cc * 128u + d];
                    if (r > cc) kk = fmaf(sm_k[r * 128u + d], kcv, kk);
                    qk = fmaf(sm_q[r * 128u + d], kcv, qk);
                }
            }
            const float a_val = (valid && r > cc) ? (-sm_beta[r] * ratio * kk) : 0.0f;
            sm_P[tq]  = a_val;
            sm_T[tq]  = (r == cc) ? 1.0f : a_val;
            sm_QK[tq] = (valid && r >= cc) ? (ratio * qk) : 0.0f;
            __syncthreads();
            #pragma unroll
            for (uint32_t st = 0u; st < 3u; ++st) {
                float p2 = 0.0f;
                if (r > cc) {
                    #pragma nounroll
                    for (uint32_t k = cc + 1u; k < r; ++k) p2 = fmaf(sm_P[r * C + k], sm_P[k * C + cc], p2);
                }
                sm_tmp[tq] = p2; __syncthreads();
                sm_P[tq] = p2;   __syncthreads();
                float tp = sm_T[tq];
                if (r > cc) {
                    #pragma nounroll
                    for (uint32_t k = cc + 1u; k <= r; ++k) tp = fmaf(sm_T[r * C + k], sm_P[k * C + cc], tp);
                }
                sm_tmp[tq] = tp; __syncthreads();
                sm_T[tq] = tp;   __syncthreads();
            }
        }

        // k_t . S[:, j] and q_t . S[:, j] for this WG's 32 columns: row-group partials, 4 tokens per round.
        #pragma nounroll
        for (uint32_t tb = 0u; tb < C; tb += 4u) {
            const uint32_t tq0 = vegajet::pgdn::opaque_u32(threadIdx.x);
            const uint32_t c = tq0 & 31u, rg = tq0 >> 5, i_beg = rg * 16u;
            #pragma nounroll
            for (uint32_t tl = 0u; tl < 4u; ++tl) {
                const uint32_t t = tb + tl;
                float pk = 0.0f, pq = 0.0f;
                #pragma unroll
                for (uint32_t r = 0u; r < 16u; ++r) {
                    pk = fmaf(s_reg[r], sm_k[t * 128u + i_beg + r], pk);
                    pq = fmaf(s_reg[r], sm_q[t * 128u + i_beg + r], pq);
                    if ((r & 3u) == 3u) __builtin_amdgcn_sched_barrier(0);
                }
                sm_u[((0u * 4u + tl) * 8u + rg) * 32u + c] = pk;
                sm_u[((1u * 4u + tl) * 8u + rg) * 32u + c] = pq;
            }
            __syncthreads();
            {
                const uint32_t tq = vegajet::pgdn::opaque_u32(threadIdx.x);
                const uint32_t kind = tq >> 7, tl = (tq >> 5) & 3u, cc = tq & 31u, t = tb + tl;
                const float* pr = sm_u + ((kind * 4u + tl) * 8u) * 32u + cc;
                const float sum = pr[0] + pr[32] + pr[64] + pr[96] + pr[128] + pr[160] + pr[192] + pr[224];
                if (kind == 0u) {
                    float rhs = 0.0f;
                    if (t < clen) {
                        const float vj = conv_qkv[static_cast<uint64_t>(m0 + t) * conv_dim + (2u * nk + hv) * 128u + col0 + cc];
                        rhs = sm_beta[t] * (vj - sum * sm_cg[t]);
                    }
                    sm_rhs[t * 32u + cc] = rhs;
                } else {
                    sm_qs[t * 32u + cc] = (t < clen) ? sum * sm_cg[t] : 0.0f;
                }
            }
            __syncthreads();
        }

        // dv = T rhs (in place into sm_u as scratch, then back), O = qs + QK dv -> fo / sq scratch.
        float* sm_dv = sm_u;
        #pragma unroll
        for (uint32_t h = 0u; h < 2u; ++h) {
            const uint32_t idx = vegajet::pgdn::opaque_u32(threadIdx.x) + 256u * h, t = idx >> 5, cc = idx & 31u;
            float dv = 0.0f;
            if (t < clen) {
                #pragma nounroll
                for (uint32_t u = 0u; u <= t; ++u) dv = fmaf(sm_T[t * C + u], sm_rhs[u * 32u + cc], dv);
            }
            sm_dv[t * 32u + cc] = dv;
        }
        __syncthreads();
        #pragma unroll
        for (uint32_t h = 0u; h < 2u; ++h) {
            const uint32_t idx = vegajet::pgdn::opaque_u32(threadIdx.x) + 256u * h, t = idx >> 5, cc = idx & 31u;
            if (t < clen) {
                float intra = 0.0f;
                #pragma nounroll
                for (uint32_t u = 0u; u <= t; ++u) intra = fmaf(sm_QK[t * C + u], sm_dv[u * 32u + cc], intra);
                const float fo = sm_qs[t * 32u + cc] + intra;
                const uint64_t mh = static_cast<uint64_t>(m0 + t) * nv + hv;
                vegajet::pgdn::emit_fo(fo, fo_scratch, sq_scratch, mh, part, cc);
            }
        }

        // State update in registers.
        const float g_end = sm_cg[clen - 1u];
        const uint32_t tqs = vegajet::pgdn::opaque_u32(threadIdx.x);
        const uint32_t c = tqs & 31u, i_beg = (tqs >> 5) * 16u;
        #pragma unroll
        for (uint32_t r = 0u; r < 16u; ++r) s_reg[r] *= g_end;
        #pragma nounroll
        for (uint32_t u = 0u; u < clen; ++u) {
            const float coef = g_end / fmaxf(sm_cg[u], 1e-30f);
            const float dvu  = sm_dv[u * 32u + c];
            #pragma unroll
            for (uint32_t r = 0u; r < 16u; ++r) {
                s_reg[r] = fmaf(coef * sm_k[u * 128u + i_beg + r], dvu, s_reg[r]);
                if ((r & 3u) == 3u) __builtin_amdgcn_sched_barrier(0);
            }
        }
        __syncthreads();
    }

    {
        const uint32_t tq = vegajet::pgdn::opaque_u32(threadIdx.x);
        float* S_thr = S_all + static_cast<uint64_t>(hv) * 16384ULL + ((tq >> 5) * 16u + 8u) * 128u + col0 + (tq & 31u);
        #pragma unroll
        for (uint32_t r = 0u; r < 16u; ++r) S_thr[(static_cast<int32_t>(r) - 8) * 128] = s_reg[r];
    }
}

#include "vj_prefill_gdn_128wg.hiph"
