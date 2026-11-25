// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Single-write Split-V GDN recurrence for speculative verification (M = 1..12) on gfx906.
//
// Today (ornith_gdn_recur_quant_kernel with use_ring=1) a verify batch of M tokens writes the full 128x128
// state of every V head after EVERY token into a 6-slot ring (M x 2 MiB per GDN layer per GPU), so that the
// host can commit any accepted prefix by moving ring_head; the ring also caps verification at M <= 5.
// Here the state is read once, stepped through M tokens in registers and written ONCE (S_M) to a separate
// slot. If the verifier rejects at position n < M, the accepted prefix is recomputed from S_0 with the SAME
// kernel (recompute-on-reject), from FP32 copies of the per-token inputs (k, v, decay, beta) recorded during
// the verify pass, for all GDN layers in one dispatch (grid.y = layer).
//
// Kernels
//   vj_spec_gdn_verify_kernel       grid (nv*4, n_layers) x 256 threads, 128 WGs per layer and GPU (nv = 32).
//       WG (4*hv + part) owns columns [32*part, 32*part+32) of V head hv; thread (rg = tid>>5, c = tid&31)
//       owns rows 16*rg..16*rg+15 of column 32*part+c (16 VGPRs of state). Columns of the GDN state are
//       independent: pred_j = sum_i S_ij k_i, dv_j = beta (v_j - g pred_j), S_ij = g S_ij + k_i dv_j,
//       o_j = sum_i S_ij q_i. The only head-wide coupling is the gated RMSNorm of the OUTPUT, which does not
//       feed back into the state. So the token loop needs no cross-WG exchange at all: each WG writes its raw
//       o_j per token to o_out, and the norm is done by
//   vj_spec_gdn_norm_quant_kernel   grid ceil(M*nv/4) x 256 threads, one wave per (token, head): head-wide
//       sum of squares with the same wave64 tree as kernel 5b, RMSNorm * w * SiLU(z), block-32 INT8 + (d, sum).
//   Choice: two dispatches (~2.2 us extra) instead of a grid barrier per token (M barriers, whole grid parked,
//   residency constraints) or a per-step partial buffer with spin-waits. Dispatch 1 has zero inter-WG sync.
//
// Arithmetic per token mirrors kernels 5b / 5b-SV term for term (same 16-row partials, same 8-way left-to-right
// LDS sums, same fmaf forms, same norm tree), so per-token outputs and states are expected to be bit-identical
// to ornith_gdn_recur_quant_kernel; recompute uses the same code on bit-identical FP32 inputs, so the
// recomputed S_n equals what a non-speculative pass would hold. (FP16 records were considered and rejected:
// they would perturb the committed state after a rejection and break the lossless 129/129 gate.)
// Bit identity against 5b must still be confirmed on the GPU (compiler contraction choices).
//
// Host contract (per GPU, Ornith: nk = 8, nv = 32, conv_dim = 6144):
//   verify : S_in = layer state slot A, S_out = slot B (B != A), q_src = conv_out, k_src = conv_out + nk*128,
//            v_src = conv_out + 2*nk*128, src_stride = conv_dim, dec/beta = gdn_dec / gdn_bsig ([M][nv]),
//            o_out = [M][nv][128] floats, rec_* = per-layer record base, layer_base = GDN layer index,
//            grid (nv*4) x 1, all layer strides except rec_* = 0.
//   commit all M : swap A/B for every GDN layer (pointer flip on the host).
//   reject at n  : one dispatch, grid (nv*4) x n_gdn_layers, S_in = S_out = slot A base (in place),
//            q_src = 0, o_out = 0, rec_kv/rec_dec/rec_beta = 0 (no re-recording), k_src = rec_kv base,
//            v_src = rec_kv base + nk*128, src_stride = rec_stride = (nk+nv)*128, dec/beta = rec_dec/rec_beta,
//            M = n, layer strides = the per-layer strides of slot A and of the records.
//   Sizes: records per layer = kMaxM * ((nk+nv)*128 + 2*nv) floats (12 * 5184 * 4 B = 243 KiB); o_out =
//   M*nv*128 floats (192 KiB at M = 12). M <= VJ_SPEC_GDN_MAX_M. LDS 15.6 KiB, no grid barrier.
// The causal-conv state (ornith_gdn_conv_l2_gate_kernel ring, m < 5) needs a matching rollback for M > 5.
#include <hip/hip_runtime.h>
#include <stdint.h>

#ifndef VJ_SPEC_GDN_MAX_M
#define VJ_SPEC_GDN_MAX_M 16
#endif

namespace vegajet {
namespace spec_gdn_dev {

__device__ __forceinline__ float wave64_sum(float v) {  // same tree as ornith_dev::wave64_sum
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x041F));
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x081F));
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x101F));
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x201F));
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x401F));
    v += __shfl_xor(v, 32, 64);
    return v;
}

__device__ __forceinline__ float sum8(const float* r, uint32_t i) {  // left-to-right, as in kernels 5b / 5b-SV
    return r[i] + r[32u + i] + r[64u + i] + r[96u + i] + r[128u + i] + r[160u + i] + r[192u + i] + r[224u + i];
}

} // namespace spec_gdn_dev
} // namespace vegajet

extern "C" __global__ __launch_bounds__(256, 3)
void vj_spec_gdn_verify_kernel(
    const float* S_in,                       // may alias S_out (recompute in place)
    float*       S_out,
    const float* __restrict__ q_src,         // nullable: no output (recompute)
    const float* __restrict__ k_src,
    const float* __restrict__ v_src,
    const float* __restrict__ dec_in,        // [M][nv]
    const float* __restrict__ beta_in,       // [M][nv]
    float*       __restrict__ o_out,         // nullable: [M][nv][128] raw o
    float*       __restrict__ rec_kv,        // nullable: [M][(nk+nv)*128] (k section, then v section)
    float*       __restrict__ rec_dec,       // nullable: [M][nv]
    float*       __restrict__ rec_beta,      // nullable: [M][nv]
    uint32_t M,
    uint32_t nk,
    uint32_t nv,
    uint32_t src_stride,                     // floats between token rows of q/k/v sources
    uint32_t rec_stride,                     // floats between token rows of rec_kv
    uint32_t layer_base,
    uint32_t s_layer_stride,                 // floats between layers of S (0 for single-layer verify)
    uint32_t src_layer_stride,               // floats between layers of k/v sources
    uint32_t gb_layer_stride,                // floats between layers of dec/beta sources
    uint32_t rec_layer_stride,               // floats between layers of rec_kv
    uint32_t rec_gb_layer_stride) {          // floats between layers of rec_dec / rec_beta
    using namespace vegajet::spec_gdn_dev;
    const uint32_t hv   = blockIdx.x >> 2;
    const uint32_t part = blockIdx.x & 3u;
    const uint32_t tid  = threadIdx.x;
    const uint32_t c    = tid & 31u;
    const uint32_t rg   = tid >> 5;
    const uint32_t col0 = part * 32u;
    if (hv >= nv || M == 0u || M > VJ_SPEC_GDN_MAX_M) return;
    const uint32_t hk  = hv % nk;
    const uint64_t lid = static_cast<uint64_t>(layer_base) + blockIdx.y;

    __shared__ float sm_q[VJ_SPEC_GDN_MAX_M * 128];
    __shared__ float sm_k[VJ_SPEC_GDN_MAX_M * 128];
    __shared__ float sm_v[VJ_SPEC_GDN_MAX_M * 32];
    __shared__ float sm_g[VJ_SPEC_GDN_MAX_M];
    __shared__ float sm_b[VJ_SPEC_GDN_MAX_M];
    __shared__ float sm_p[8 * 32];
    __shared__ float sm_o[8 * 32];
    __shared__ float sm_dv[32];

    // ---- stage all M tokens' inputs in LDS (and record them for recompute-on-reject) ----
    {
        const float* ks = k_src + lid * src_layer_stride;
        const float* vs = v_src + lid * src_layer_stride;
        float* rk = rec_kv ? rec_kv + lid * rec_layer_stride : nullptr;
        const bool rec_k = rk != nullptr && part == 0u && hv < nk;  // hv < nk  =>  hk == hv
        for (uint32_t e = tid; e < M * 128u; e += 256u) {
            const uint32_t m = e >> 7, i = e & 127u;
            const float kv = ks[static_cast<uint64_t>(m) * src_stride + hk * 128u + i];
            sm_k[e] = kv;
            if (q_src) sm_q[e] = q_src[lid * src_layer_stride + static_cast<uint64_t>(m) * src_stride + hk * 128u + i];
            if (rec_k) rk[static_cast<uint64_t>(m) * rec_stride + hv * 128u + i] = kv;
        }
        for (uint32_t e = tid; e < M * 32u; e += 256u) {
            const uint32_t m = e >> 5, cc = e & 31u;
            const float vv = vs[static_cast<uint64_t>(m) * src_stride + hv * 128u + col0 + cc];
            sm_v[e] = vv;
            if (rk) rk[static_cast<uint64_t>(m) * rec_stride + nk * 128u + hv * 128u + col0 + cc] = vv;
        }
        if (tid < M) {
            const float g = dec_in[lid * gb_layer_stride + static_cast<uint64_t>(tid) * nv + hv];
            const float b = beta_in[lid * gb_layer_stride + static_cast<uint64_t>(tid) * nv + hv];
            sm_g[tid] = g;
            sm_b[tid] = b;
            if (rec_dec && part == 0u) {
                rec_dec[lid * rec_gb_layer_stride + static_cast<uint64_t>(tid) * nv + hv] = g;
                rec_beta[lid * rec_gb_layer_stride + static_cast<uint64_t>(tid) * nv + hv] = b;
            }
        }
    }

    // ---- S_0: this thread's 16 rows of one column, read once (one-shot stream) ----
    const uint32_t i_beg = rg * 16u;
    const uint32_t col   = col0 + c;
    const uint64_t s_off = lid * s_layer_stride + static_cast<uint64_t>(hv) * 16384u + (i_beg + 8u) * 128u + col;
    const float* S_thr_in = S_in + s_off;
    float s_reg[16];
    #pragma unroll
    for (uint32_t r = 0u; r < 16u; ++r) s_reg[r] = __builtin_nontemporal_load(S_thr_in + (static_cast<int32_t>(r) - 8) * 128);
    __syncthreads();

    // do-while: M >= 1 is guaranteed above; a zero-trip-capable loop makes the compiler keep a second copy of
    // the 16 state registers alive across the loop (and spill them).
    uint32_t m = 0u;
    #pragma nounroll
    do {
        const float* km = sm_k + m * 128u + i_beg;
        float p = 0.0f;
        #pragma unroll
        for (uint32_t r = 0u; r < 16u; ++r) p = fmaf(s_reg[r], km[r], p);
        sm_p[rg * 32u + c] = p;
        __syncthreads();  // A: pred partials of token m visible; o partials of token m-1 visible

        if (tid < 32u) {
            const float pred = sum8(sm_p, tid);
            const float vj = sm_v[m * 32u + tid];
            sm_dv[tid] = sm_b[m] * (vj - sm_g[m] * pred);
        } else if (o_out != nullptr && q_src != nullptr && m > 0u && tid >= 64u && tid < 96u) {
            o_out[(static_cast<uint64_t>(m - 1u) * nv + hv) * 128u + col0 + (tid - 64u)] = sum8(sm_o, tid - 64u);
        }
        __syncthreads();  // B: dv visible; sm_o free for token m

        const float dv = sm_dv[c];
        const float g_exp = sm_g[m];
        if (q_src) {
            const float* qm = sm_q + m * 128u + i_beg;
            float o = 0.0f;
            // Two halves with a scheduling fence: bounds the LDS operands in flight to 8 k + 8 q values
            // (47 VGPRs; unfenced the scheduler hoists all 32 reads: 55 VGPRs).
            #pragma unroll
            for (uint32_t r = 0u; r < 8u; ++r) {
                s_reg[r] = fmaf(km[r], dv, g_exp * s_reg[r]);
                o = fmaf(s_reg[r], qm[r], o);
            }
            __builtin_amdgcn_sched_barrier(0);
            #pragma unroll
            for (uint32_t r = 8u; r < 16u; ++r) {
                s_reg[r] = fmaf(km[r], dv, g_exp * s_reg[r]);
                o = fmaf(s_reg[r], qm[r], o);
            }
            sm_o[rg * 32u + c] = o;
        } else {
            #pragma unroll
            for (uint32_t r = 0u; r < 16u; ++r) s_reg[r] = fmaf(km[r], dv, g_exp * s_reg[r]);
        }
    } while (++m < M);
    __syncthreads();
    if (o_out != nullptr && q_src != nullptr && tid >= 64u && tid < 96u)
        o_out[(static_cast<uint64_t>(M - 1u) * nv + hv) * 128u + col0 + (tid - 64u)] = sum8(sm_o, tid - 64u);

    // ---- S_M: written once ----
    float* S_thr_out = S_out + s_off;
    #pragma unroll
    for (uint32_t r = 0u; r < 16u; ++r) S_thr_out[(static_cast<int32_t>(r) - 8) * 128] = s_reg[r];
}

// One wave per (token m, head hv): gated RMSNorm + block-32 INT8 quant, term for term as kernel 5b (wave 0).
extern "C" __global__ __launch_bounds__(256, 4)
void vj_spec_gdn_norm_quant_kernel(
    const float* __restrict__ o_buf,       // [M][nv][128] from vj_spec_gdn_verify_kernel (previous dispatch)
    const float* __restrict__ qkv_gab_in,  // z gate rows: [M][stride_in], z at z_off + hv*128
    const float* __restrict__ ssm_norm_w,  // [128]
    int8_t*      __restrict__ out_q8,      // [M][nv][128]
    float2*      __restrict__ out_ds,      // [M][nv][4]
    uint32_t M,
    uint32_t nv,
    uint32_t stride_in,
    uint32_t z_off,
    float    eps) {
    using namespace vegajet::spec_gdn_dev;
    const uint32_t wave = threadIdx.x >> 6;
    const uint32_t lane = threadIdx.x & 63u;
    const uint32_t idx  = blockIdx.x * 4u + wave;
    if (idx >= M * nv) return;
    const uint32_t m  = idx / nv;
    const uint32_t hv = idx - m * nv;
    const uint32_t j0 = lane * 2u;

    const float* ob = o_buf + (static_cast<uint64_t>(m) * nv + hv) * 128u;
    const float fo0 = __builtin_nontemporal_load(ob + j0);
    const float fo1 = __builtin_nontemporal_load(ob + j0 + 1u);
    const float total_sq = wave64_sum(fo0 * fo0 + fo1 * fo1);
    const float inv_r = rsqrtf(total_sq * (1.0f / 128.0f) + eps);
    const float* zp = qkv_gab_in + static_cast<uint64_t>(m) * stride_in + z_off + hv * 128u;
    const float z0 = zp[j0], z1 = zp[j0 + 1u];
    const float y0 = (fo0 * inv_r * ssm_norm_w[j0])      * (z0 / (1.0f + __expf(-z0)));
    const float y1 = (fo1 * inv_r * ssm_norm_w[j0 + 1u]) * (z1 / (1.0f + __expf(-z1)));

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
    int8_t* oq = out_q8 + (static_cast<uint64_t>(m) * nv + hv) * 128u;
    oq[j0]      = static_cast<int8_t>(max(-127, min(127, __float2int_rn(y0 * inv_d))));
    oq[j0 + 1u] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(y1 * inv_d))));
    if ((lane & 15u) == 0u) out_ds[(static_cast<uint64_t>(m) * nv + hv) * 4u + (lane >> 4)] = make_float2(d, bsum);
}
