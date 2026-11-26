// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// v1.2 S3 tree verify: ancestry-aware GDN conv and recurrence (VEGAJET_V12_TREE_VERIFY=1 only).
//
// Separate code object (vj_tree_gdn.hsaco): the chain kernels in vj_ornith_attn.hip / vj_spec_gdn.hip are
// unchanged. Both kernels here copy the chain arithmetic term for term; only the input/state SOURCE of each
// row changes from "previous flattened row" to "this row's tree ancestors" (core/spec_tree.hpp).
//
//   vj_tree_gdn_conv_kernel    = ornith_gdn_conv_l2_gate_kernel + conv_inputs[M*3] (oldest first, per row).
//       Index -3,-2,-1 = immutable history slots 0,1,2 at ring_head; index j >= 0 = raw qkv_gab_in row j
//       (an ancestor, j < m). No mutable sibling state is read. With use_ring != 0, the post-row history
//       {h1, h2, x_m} goes to ring slot (ring_head + m + 1) % VJ_GDN_RING_SLOTS (m < slots - 1), the same slot
//       as the chain kernel. With use_ring == 0 nothing is written (a flattened last row is not a commit state).
//   vj_tree_gdn_verify_kernel  = vj_spec_gdn_verify_kernel + parents[M] + node_states[M][nv][128][128].
//       Row m starts from S_in (parent < 0) or node_states[parent] (skipped when parent == m-1: the registers
//       hold exactly those values). After the update, node_states[m] is saved. o_out / rec_* are unchanged.
//       Each thread saves and restores only its own 16 rows x 1 column: no cross-thread or cross-WG data.
//       S_out receives the LAST row's state only; the host commits via record replay or node_states.
//       Single-layer verify only (gridDim.y == 1). Invalid parents make the whole grid return before any write.
#include "ops/kernels/vj_ornith_common.hiph"
#include <stdint.h>

#ifndef VJ_SPEC_GDN_MAX_M
#define VJ_SPEC_GDN_MAX_M 16
#endif

namespace vegajet {
namespace tree_gdn_dev {

__device__ __forceinline__ float sum8(const float* r, uint32_t i) {  // left-to-right, as vj_spec_gdn
    return r[i] + r[32u + i] + r[64u + i] + r[96u + i] + r[128u + i] + r[160u + i] + r[192u + i] + r[224u + i];
}

// idx in [-3, m): history slot or ancestor raw row; anything else is a host bug -> NaN (visible, no OOB read).
__device__ __forceinline__ float conv_hist(int32_t idx, uint32_t m, float s0, float s1, float s2,
                                           const float* __restrict__ qkv_gab_in, uint32_t stride_in, uint32_t ch) {
    if (idx == -3) return s0;
    if (idx == -2) return s1;
    if (idx == -1) return s2;
    if (idx >= 0 && static_cast<uint32_t>(idx) < m)
        return qkv_gab_in[static_cast<uint64_t>(idx) * stride_in + ch];
    return __int_as_float(0x7fc00000);
}

} // namespace tree_gdn_dev
} // namespace vegajet

extern "C" __global__ __launch_bounds__(128, 6)
void vj_tree_gdn_conv_kernel(
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
    uint32_t use_ring,
    float    eps,
    const int32_t* __restrict__ conv_inputs) {   // [M][3], oldest first
    using vegajet::tree_gdn_dev::conv_hist;
    const uint32_t grp = blockIdx.x;
    const uint32_t tid = threadIdx.x; // 0..127
    const uint32_t n_head_wgs = 2u * nk + nv; // 48

    if (grp < n_head_wgs) {
        const uint32_t ch = grp * 128u + tid;
        const float4 w = reinterpret_cast<const float4*>(conv_w)[ch];
        const uint64_t slot_stride = 3ULL * conv_dim;
        const float* r_in = conv_ring + static_cast<uint64_t>(ring_head) * slot_stride;
        // Immutable pre-verify history; later slot writes never target ring_head (m + 1 < slots).
        const float s0 = r_in[0ULL * conv_dim + ch];
        const float s1 = r_in[1ULL * conv_dim + ch];
        const float s2 = r_in[2ULL * conv_dim + ch];
        __shared__ float sm_sq[2];

        for (uint32_t m = 0u; m < M; ++m) {
            const int32_t* ci = conv_inputs + 3u * m;
            const float h0 = conv_hist(ci[0], m, s0, s1, s2, qkv_gab_in, stride_in, ch);
            const float h1 = conv_hist(ci[1], m, s0, s1, s2, qkv_gab_in, stride_in, ch);
            const float h2 = conv_hist(ci[2], m, s0, s1, s2, qkv_gab_in, stride_in, ch);
            const float x = qkv_gab_in[static_cast<uint64_t>(m) * stride_in + ch];
            const float acc = fmaf(w.x, h0, fmaf(w.y, h1, fmaf(w.z, h2, w.w * x)));
            float y = acc / (1.0f + __expf(-acc));
            if (use_ring != 0u && m < VJ_GDN_RING_SLOTS - 1u) {
                const uint32_t dst_slot = (ring_head + m + 1u) % VJ_GDN_RING_SLOTS;
                float* r_out = conv_ring + static_cast<uint64_t>(dst_slot) * slot_stride;
                r_out[0ULL * conv_dim + ch] = h1;
                r_out[1ULL * conv_dim + ch] = h2;
                r_out[2ULL * conv_dim + ch] = x;
            }
            if (grp < 2u * nk) {
                float sq = vegajet::ornith_dev::wave64_sum(y * y);
                if ((tid & 63u) == 0u) sm_sq[tid >> 6] = sq;
                __syncthreads();
                float inv_n = rsqrtf(sm_sq[0] + sm_sq[1] + eps);
                if (grp < nk) inv_n *= 0.08838834764831845f; // 1/sqrt(128)
                y *= inv_n;
                __syncthreads();
            }
            conv_out[static_cast<uint64_t>(m) * conv_dim + ch] = y;
        }
    } else if (grp == n_head_wgs && tid < nv) {
        const float dtb = dt_bias[tid];
        const float sa  = ssm_a[tid];
        for (uint32_t m = 0u; m < M; ++m) {
            const float* row = qkv_gab_in + static_cast<uint64_t>(m) * stride_in;
            const float a  = row[alpha_off + tid] + dtb;
            const float sp = (a > 20.0f) ? a : log1pf(__expf(a));
            const uint64_t idx = static_cast<uint64_t>(m) * nv + tid;
            decay_out[idx] = __expf(sp * sa);
            beta_out[idx]  = 1.0f / (1.0f + __expf(-row[beta_off + tid]));
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 3)
void vj_tree_gdn_verify_kernel(
    const float* S_in,                       // immutable pre-verify layer state; may alias S_out
    float*       S_out,                      // last row's state only
    const float* __restrict__ q_src,         // nullable: no output
    const float* __restrict__ k_src,
    const float* __restrict__ v_src,
    const float* __restrict__ dec_in,        // [M][nv]
    const float* __restrict__ beta_in,       // [M][nv]
    float*       __restrict__ o_out,         // nullable: [M][nv][128] raw o
    float*       __restrict__ rec_kv,        // nullable: [M][(nk+nv)*128]
    float*       __restrict__ rec_dec,       // nullable: [M][nv]
    float*       __restrict__ rec_beta,      // nullable: [M][nv]
    uint32_t M,
    uint32_t nk,
    uint32_t nv,
    uint32_t src_stride,
    uint32_t rec_stride,
    uint32_t layer_base,
    uint32_t s_layer_stride,
    uint32_t src_layer_stride,
    uint32_t gb_layer_stride,
    uint32_t rec_layer_stride,
    uint32_t rec_gb_layer_stride,
    const int32_t* __restrict__ parents,     // [M]: parents[0] == -1, 0 <= parents[m] < m otherwise
    float*       __restrict__ node_states) { // [M][nv][128][128], current layer only
    using namespace vegajet::tree_gdn_dev;
    const uint32_t hv   = blockIdx.x >> 2;
    const uint32_t part = blockIdx.x & 3u;
    const uint32_t tid  = threadIdx.x;
    const uint32_t c    = tid & 31u;
    const uint32_t rg   = tid >> 5;
    const uint32_t col0 = part * 32u;
    if (hv >= nv || M == 0u || M > VJ_SPEC_GDN_MAX_M || gridDim.y != 1u || node_states == nullptr) return;
    // Uniform validation (same values in every thread): reject before any write.
    if (parents[0] != -1) return;
    for (uint32_t m = 1u; m < M; ++m) {
        const int32_t p = parents[m];
        if (p < 0 || static_cast<uint32_t>(p) >= m) return;
    }
    const uint32_t hk  = hv % nk;
    const uint64_t lid = static_cast<uint64_t>(layer_base);

    __shared__ float sm_q[VJ_SPEC_GDN_MAX_M * 128];
    __shared__ float sm_k[VJ_SPEC_GDN_MAX_M * 128];
    __shared__ float sm_v[VJ_SPEC_GDN_MAX_M * 32];
    __shared__ float sm_g[VJ_SPEC_GDN_MAX_M];
    __shared__ float sm_b[VJ_SPEC_GDN_MAX_M];
    __shared__ int32_t sm_par[VJ_SPEC_GDN_MAX_M];
    __shared__ float sm_p[8 * 32];
    __shared__ float sm_o[8 * 32];
    __shared__ float sm_dv[32];

    // ---- stage all M rows' inputs in LDS and record them (verbatim vj_spec_gdn_verify_kernel) ----
    {
        const float* ks = k_src + lid * src_layer_stride;
        const float* vs = v_src + lid * src_layer_stride;
        float* rk = rec_kv ? rec_kv + lid * rec_layer_stride : nullptr;
        const bool rec_k = rk != nullptr && part == 0u && hv < nk;
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
            sm_par[tid] = parents[tid];
            if (rec_dec && part == 0u) {
                rec_dec[lid * rec_gb_layer_stride + static_cast<uint64_t>(tid) * nv + hv] = g;
                rec_beta[lid * rec_gb_layer_stride + static_cast<uint64_t>(tid) * nv + hv] = b;
            }
        }
    }

    // ---- root state: this thread's 16 rows of one column ----
    const uint32_t i_beg = rg * 16u;
    const uint32_t col   = col0 + c;
    // Same element offset as vj_spec_gdn's s_off minus the uniform layer term. All state addresses are
    // uniform base + 32-bit byte offset (M*nv*64 KiB < 4 GiB): one shared VGPR offset, SADDR stores.
    const uint32_t n_off = hv * 16384u + (i_beg + 8u) * 128u + col;
    const uint32_t node_stride = nv * 16384u;  // floats per node snapshot (single layer)
    const float* S_thr_in = reinterpret_cast<const float*>(
        reinterpret_cast<const char*>(S_in + lid * s_layer_stride) + n_off * 4u);
    float s_reg[16];
    #pragma unroll
    for (uint32_t r = 0u; r < 16u; ++r) s_reg[r] = S_thr_in[(static_cast<int32_t>(r) - 8) * 128];
    __syncthreads();

    uint32_t m = 0u;
    #pragma nounroll
    do {
        // Restore the parent snapshot this thread saved earlier (same thread, same addresses).
        const int32_t par = __builtin_amdgcn_readfirstlane(sm_par[m]);  // uniform: scalar branch
        if (m > 0u && static_cast<uint32_t>(par) + 1u != m) {
            const float* src = reinterpret_cast<const float*>(
                reinterpret_cast<const char*>(node_states) + (static_cast<uint32_t>(par) * node_stride + n_off) * 4u);
            #pragma unroll
            for (uint32_t r = 0u; r < 16u; ++r) s_reg[r] = src[(static_cast<int32_t>(r) - 8) * 128];
        }
        const float* km = sm_k + m * 128u + i_beg;
        float p = 0.0f;
        #pragma unroll
        for (uint32_t r = 0u; r < 16u; ++r) p = fmaf(s_reg[r], km[r], p);
        sm_p[rg * 32u + c] = p;
        __syncthreads();  // A

        if (tid < 32u) {
            const float pred = sum8(sm_p, tid);
            const float vj = sm_v[m * 32u + tid];
            sm_dv[tid] = sm_b[m] * (vj - sm_g[m] * pred);
        } else if (o_out != nullptr && q_src != nullptr && m > 0u && tid >= 64u && tid < 96u) {
            o_out[(static_cast<uint64_t>(m - 1u) * nv + hv) * 128u + col0 + (tid - 64u)] = sum8(sm_o, tid - 64u);
        }
        __syncthreads();  // B

        const float dv = sm_dv[c];
        const float g_exp = sm_g[m];
        if (q_src) {
            const float* qm = sm_q + m * 128u + i_beg;
            float o = 0.0f;
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
        __builtin_amdgcn_sched_barrier(0);  // keep snapshot stores out of the q/k operand window
        float* dst = reinterpret_cast<float*>(reinterpret_cast<char*>(node_states) + (m * node_stride + n_off) * 4u);
        #pragma unroll
        for (uint32_t r = 0u; r < 16u; ++r) dst[(static_cast<int32_t>(r) - 8) * 128] = s_reg[r];
    } while (++m < M);
    __syncthreads();
    if (o_out != nullptr && q_src != nullptr && tid >= 64u && tid < 96u)
        o_out[(static_cast<uint64_t>(M - 1u) * nv + hv) * 128u + col0 + (tid - 64u)] = sum8(sm_o, tid - 64u);

    float* S_thr_out = reinterpret_cast<float*>(reinterpret_cast<char*>(S_out + lid * s_layer_stride) + n_off * 4u);
    #pragma unroll
    for (uint32_t r = 0u; r < 16u; ++r) S_thr_out[(static_cast<int32_t>(r) - 8) * 128] = s_reg[r];
}
