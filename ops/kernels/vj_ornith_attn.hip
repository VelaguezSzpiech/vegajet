// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "ops/kernels/vj_ornith_common.hiph"

// ============================================================================
// 5a. FUSED GDN Phase A: Causal Conv1D (k=4) + SiLU + Inline L2-Norm + Alpha/Beta Gate!
//     49 WGs x 128 threads:
//       - blockIdx.x 0..7   : Q heads (Conv1D + SiLU + L2-Norm * 1/sqrt(128))
//       - blockIdx.x 8..15  : K heads (Conv1D + SiLU + L2-Norm)
//       - blockIdx.x 16..47 : V heads (Conv1D + SiLU)
//       - blockIdx.x 48     : Alpha/Beta -> decay/beta for all 32 V-heads
//     Collapses 2 dispatches into 1 AND eliminates Q/K global round-trip!
// ============================================================================
extern "C" __global__ __launch_bounds__(128, 6)
void ornith_gdn_conv_l2_gate_kernel(
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
    float    eps) {
    const uint32_t grp = blockIdx.x;
    const uint32_t tid = threadIdx.x; // 0..127
    const uint32_t n_head_wgs = 2u * nk + nv; // 48

    if (grp < n_head_wgs) {
        const uint32_t ch = grp * 128u + tid;
        const float4 w = reinterpret_cast<const float4*>(conv_w)[ch];
        const uint64_t slot_stride = 3ULL * conv_dim;
        const float* r_in = conv_ring + static_cast<uint64_t>(use_ring == 2u ? 0u : ring_head) * slot_stride;
        float s0 = r_in[0ULL * conv_dim + ch];
        float s1 = r_in[1ULL * conv_dim + ch];
        float s2 = r_in[2ULL * conv_dim + ch];
        __shared__ float sm_sq[2];

        // use_ring == 2 (VEGAJET_GPU_BATCH): row m is an independent sequence whose conv state lives in slot
        // the 3-bit row entry packed in ring_head and is updated in place (no carry between rows).
        const bool per_row = (use_ring == 2u);
        for (uint32_t m = 0u; m < M; ++m) {
            const uint32_t slot = per_row ? ((ring_head >> (3u * m)) & 7u) : ring_head + m;
            float* r_row = conv_ring + static_cast<uint64_t>(slot) * slot_stride;
            if (per_row) { s0 = r_row[0ULL * conv_dim + ch]; s1 = r_row[1ULL * conv_dim + ch]; s2 = r_row[2ULL * conv_dim + ch]; }
            const float x = qkv_gab_in[static_cast<uint64_t>(m) * stride_in + ch];
            const float acc = fmaf(w.x, s0, fmaf(w.y, s1, fmaf(w.z, s2, w.w * x)));
            float y = acc / (1.0f + __expf(-acc));
            s0 = s1; s1 = s2; s2 = x;
            if (per_row) { r_row[0ULL * conv_dim + ch] = s0; r_row[1ULL * conv_dim + ch] = s1; r_row[2ULL * conv_dim + ch] = s2; }
            if (use_ring == 1u && m < VJ_GDN_RING_SLOTS - 1u) {
                const uint32_t dst_slot = (ring_head + m + 1u) % VJ_GDN_RING_SLOTS;
                float* r_out = conv_ring + static_cast<uint64_t>(dst_slot) * slot_stride;
                r_out[0ULL * conv_dim + ch] = s0;
                r_out[1ULL * conv_dim + ch] = s1;
                r_out[2ULL * conv_dim + ch] = s2;
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
        if (use_ring == 0u) {
            float* r_out = conv_ring + static_cast<uint64_t>(ring_head) * slot_stride;
            r_out[0ULL * conv_dim + ch] = s0;
            r_out[1ULL * conv_dim + ch] = s1;
            r_out[2ULL * conv_dim + ch] = s2;
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

// ============================================================================
// 5b. 1024-thread GDN recurrence. Ring verification follows production super M=1;
//     non-ring prefill/fallback retains its original eight-row arithmetic.
// ============================================================================
extern "C" __global__ __launch_bounds__(1024, 1)
void ornith_gdn_recur_quant_kernel(
    float*       __restrict__ ssm_ring,
    const float* __restrict__ conv_qkv,
    const float* __restrict__ qkv_gab_in,
    const float* __restrict__ decay_in,
    const float* __restrict__ beta_in,
    const float* __restrict__ ssm_norm_w,
    int8_t*      __restrict__ out_q8,
    float2*      __restrict__ out_ds,
    uint32_t M,
    uint32_t nk,
    uint32_t nv,
    uint32_t conv_dim,
    uint32_t stride_in,
    uint32_t z_off,
    uint32_t ring_head,
    uint32_t use_ring,
    float    eps) {
    // Non-ring calls also build the raw reference's prompt state. Do not change
    // that state while fixing verifier order; the M=1 split/super kernels are unchanged.
    // use_ring == 2 (VEGAJET_GPU_BATCH): row m is an independent sequence; its state slot is packed in the 3-bit row entry of ring_head,
    // loaded and stored in place around that row's step (same per-step arithmetic as the use_ring == 0 path).
    if (use_ring == 0u) {
        const bool per_row = (use_ring == 2u);
        const uint32_t hv = blockIdx.x, tid = threadIdx.x;
        const uint32_t wave = tid >> 6, lane = tid & 63u;
        if (hv >= nv) return;
        const uint32_t hk = hv % nk, i_beg = wave * 8u, j0 = lane * 2u;
        const uint64_t slot_elems = static_cast<uint64_t>(nv) * 128u * 128u;
        float* S_base = ssm_ring + static_cast<uint64_t>(ring_head) * slot_elems
                       + static_cast<uint64_t>(hv) * 128u * 128u;
        __shared__ float sm_q[128], sm_k[128], sm_red[16 * 128], sm_dv[128];
        float2 s_reg[8];
        const float2* S_src2 = reinterpret_cast<const float2*>(S_base) + (i_beg + 4u) * 64u + lane;
        if (!per_row) {
            #pragma unroll
            for (uint32_t r = 0u; r < 8u; ++r)
                s_reg[r] = S_src2[(static_cast<int32_t>(r) - 4) * 64];
        }
        #pragma nounroll
        for (uint32_t m = 0u; m < M; ++m) {
            uint32_t mo = __builtin_amdgcn_readfirstlane(m);
            asm volatile("" : "+s"(mo));
            float2* S_row2 = reinterpret_cast<float2*>(S_base + static_cast<uint64_t>(mo) * slot_elems) + (i_beg + 4u) * 64u + lane;
            if (per_row) {
                #pragma unroll
                for (uint32_t r = 0u; r < 8u; ++r)
                    s_reg[r] = S_row2[(static_cast<int32_t>(r) - 4) * 64];
            }
            const float* base = conv_qkv + static_cast<uint64_t>(mo) * conv_dim;
            if (tid < 128u) {
                sm_q[tid] = base[hk * 128u + tid];
                sm_k[tid] = base[(nk + hk) * 128u + tid];
            }
            __syncthreads();
            const float g_exp = decay_in[static_cast<uint64_t>(mo) * nv + hv];
            const float beta = beta_in[static_cast<uint64_t>(mo) * nv + hv];
            float p0 = 0.0f, p1 = 0.0f;
            #pragma unroll
            for (uint32_t r = 0u; r < 8u; ++r) {
                const float ki = sm_k[i_beg + r];
                p0 = fmaf(s_reg[r].x, ki, p0);
                p1 = fmaf(s_reg[r].y, ki, p1);
            }
            sm_red[wave * 128u + j0] = p0;
            sm_red[wave * 128u + j0 + 1u] = p1;
            __syncthreads();
            if (tid < 128u) {
                float pred = 0.0f;
                #pragma nounroll
                for (uint32_t w = 0u; w < 16u; ++w) pred += sm_red[w * 128u + tid];
                const float vj = base[(2u * nk + hv) * 128u + tid];
                sm_dv[tid] = beta * (vj - g_exp * pred);
            }
            __syncthreads();
            const float dv0 = sm_dv[j0], dv1 = sm_dv[j0 + 1u];
            float o0 = 0.0f, o1 = 0.0f;
            #pragma unroll
            for (uint32_t r = 0u; r < 8u; ++r) {
                const float ki = sm_k[i_beg + r], qi = sm_q[i_beg + r];
                s_reg[r].x = fmaf(ki, dv0, g_exp * s_reg[r].x);
                s_reg[r].y = fmaf(ki, dv1, g_exp * s_reg[r].y);
                o0 = fmaf(s_reg[r].x, qi, o0);
                o1 = fmaf(s_reg[r].y, qi, o1);
            }
            sm_red[wave * 128u + j0] = o0;
            sm_red[wave * 128u + j0 + 1u] = o1;
            __syncthreads();
            if (wave == 0u) {
                float fo0 = 0.0f, fo1 = 0.0f;
                #pragma nounroll
                for (uint32_t w = 0u; w < 16u; ++w) {
                    fo0 += sm_red[w * 128u + j0];
                    fo1 += sm_red[w * 128u + j0 + 1u];
                }
                const float total_sq = vegajet::ornith_dev::wave64_sum(fo0 * fo0 + fo1 * fo1);
                const float inv_r = rsqrtf(total_sq * (1.0f / 128.0f) + eps);
                const float* zp = qkv_gab_in + static_cast<uint64_t>(mo) * stride_in + z_off + hv * 128u;
                const float z0 = zp[j0], z1 = zp[j0 + 1u];
                const float y0 = (fo0 * inv_r * ssm_norm_w[j0]) * (z0 / (1.0f + __expf(-z0)));
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
                const float d = amax * (1.0f / 127.0f);
                const float inv_d = amax > 0.0f ? 127.0f / amax : 0.0f;
                int8_t* oq = out_q8 + (static_cast<uint64_t>(mo) * nv + hv) * 128u;
                oq[j0] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(y0 * inv_d))));
                oq[j0 + 1u] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(y1 * inv_d))));
                if ((lane & 15u) == 0u)
                    out_ds[(static_cast<uint64_t>(mo) * nv + hv) * 4u + (lane >> 4)] = make_float2(d, bsum);
            }
            if (per_row) {
                #pragma unroll
                for (uint32_t r = 0u; r < 8u; ++r)
                    S_row2[(static_cast<int32_t>(r) - 4) * 64] = s_reg[r];
            }
            __syncthreads();
        }
        if (per_row) return;
        float2* S_dst2 = reinterpret_cast<float2*>(S_base) + (i_beg + 4u) * 64u + lane;
        #pragma unroll
        for (uint32_t r = 0u; r < 8u; ++r)
            S_dst2[(static_cast<int32_t>(r) - 4) * 64] = s_reg[r];
        return;
    }

    const uint32_t hv   = blockIdx.x;
    const uint32_t tid  = threadIdx.x; // 0..1023: four M=1 superkernel parts in one WG
    const uint32_t wave = tid >> 6;
    const uint32_t lane = tid & 63u;
    const uint32_t rg4  = (tid & 255u) >> 3;
    const uint32_t col4 = (tid >> 8) * 8u + (tid & 7u);
    const uint32_t wv   = (tid >> 6) & 3u;
    if (hv >= nv) return;

    const uint32_t hk = hv % nk;
    const uint64_t head_elems = 128ULL * 128ULL;
    const uint64_t slot_elems = static_cast<uint64_t>(nv) * head_elems;
    const float*   S_base     = ssm_ring + static_cast<uint64_t>(use_ring == 2u ? 0u : ring_head) * slot_elems + static_cast<uint64_t>(hv) * head_elems;

    __shared__ float sm_q[128];
    __shared__ float sm_k[128];
    __shared__ float sm_red[4 * 128]; // four balanced wave partials per column
    __shared__ float sm_dv[128];

    const uint32_t i_beg = rg4 * 4u;
    const uint32_t j0    = lane * 2u;

    float4 s_reg[4];
    const float4* S_src4 = reinterpret_cast<const float4*>(S_base) + (i_beg + 2u) * 32u + col4;
    #pragma unroll
    for (uint32_t r = 0u; r < 4u; ++r)
        s_reg[r] = S_src4[(static_cast<int32_t>(r) - 2) * 32];

    // Match ornith_m1_gdn_super_kernel: XOR 32,16,8 within each wave, then
    // (wave0 + wave1) + (wave2 + wave3). Eight-/sixteen-row FMA chains differ.
    auto wave_red8 = [](float v) -> float {
        v += __shfl_xor(v, 32, 64);
        v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x401F));
        v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x201F));
        return v;
    };

    #pragma nounroll
    for (uint32_t m = 0u; m < M; ++m) {
        uint32_t mo = __builtin_amdgcn_readfirstlane(m);
        asm volatile("" : "+s"(mo));
        // L6: independent rows use the production M=1 reduction tree, loading
        // their own state before each row and committing to that same slot.
        if (use_ring == 2u) {
            const float4* row_state = reinterpret_cast<const float4*>(S_base + static_cast<uint64_t>((ring_head >> (3u * mo)) & 7u) * slot_elems)
                                     + (i_beg + 2u) * 32u + col4;
            #pragma unroll
            for (uint32_t r = 0u; r < 4u; ++r) s_reg[r] = row_state[(static_cast<int32_t>(r) - 2) * 32];
        }
        const float* base = conv_qkv + static_cast<uint64_t>(mo) * conv_dim;
        if (tid < 128u) {
            sm_q[tid] = base[hk * 128u + tid];
            sm_k[tid] = base[(nk + hk) * 128u + tid];
        }
        __syncthreads();

        const float g_exp = decay_in[static_cast<uint64_t>(mo) * nv + hv];
        const float beta  = beta_in[static_cast<uint64_t>(mo) * nv + hv];

        const float k0 = sm_k[i_beg];
        float4 p = make_float4(s_reg[0].x * k0, s_reg[0].y * k0,
                              s_reg[0].z * k0, s_reg[0].w * k0);
        #pragma unroll
        for (uint32_t r = 1u; r < 4u; ++r) {
            const float ki = sm_k[i_beg + r];
            p.x = fmaf(s_reg[r].x, ki, p.x);
            p.y = fmaf(s_reg[r].y, ki, p.y);
            p.z = fmaf(s_reg[r].z, ki, p.z);
            p.w = fmaf(s_reg[r].w, ki, p.w);
        }
        p.x = wave_red8(p.x); p.y = wave_red8(p.y);
        p.z = wave_red8(p.z); p.w = wave_red8(p.w);
        if ((tid & 56u) == 0u) reinterpret_cast<float4*>(sm_red)[wv * 32u + col4] = p;
        __syncthreads();

        if (tid < 128u) {
            const float pred = (sm_red[tid] + sm_red[128u + tid])
                             + (sm_red[256u + tid] + sm_red[384u + tid]);
            const float vj = base[(2u * nk + hv) * 128u + tid];
            sm_dv[tid] = beta * (vj - g_exp * pred);
        }
        __syncthreads();

        const float4 dv = reinterpret_cast<const float4*>(sm_dv)[col4];
        const uint64_t out_slot = use_ring == 2u ? ((ring_head >> (3u * mo)) & 7u)
            : static_cast<uint64_t>((ring_head + mo + 1u) % VJ_GDN_RING_SLOTS);
        float4* S_out4 = reinterpret_cast<float4*>(ssm_ring + out_slot * slot_elems
                             + static_cast<uint64_t>(hv) * head_elems) + (i_beg + 2u) * 32u + col4;
        float4 o;
        #pragma unroll
        for (uint32_t r = 0u; r < 4u; ++r) {
            const float ki = sm_k[i_beg + r];
            const float qi = sm_q[i_beg + r];
            s_reg[r].x = fmaf(ki, dv.x, g_exp * s_reg[r].x);
            s_reg[r].y = fmaf(ki, dv.y, g_exp * s_reg[r].y);
            s_reg[r].z = fmaf(ki, dv.z, g_exp * s_reg[r].z);
            s_reg[r].w = fmaf(ki, dv.w, g_exp * s_reg[r].w);
            S_out4[(static_cast<int32_t>(r) - 2) * 32] = s_reg[r];
            if (r == 0u) {
                o = make_float4(s_reg[r].x * qi, s_reg[r].y * qi,
                                s_reg[r].z * qi, s_reg[r].w * qi);
            } else {
                o.x = fmaf(s_reg[r].x, qi, o.x);
                o.y = fmaf(s_reg[r].y, qi, o.y);
                o.z = fmaf(s_reg[r].z, qi, o.z);
                o.w = fmaf(s_reg[r].w, qi, o.w);
            }
        }
        o.x = wave_red8(o.x); o.y = wave_red8(o.y);
        o.z = wave_red8(o.z); o.w = wave_red8(o.w);
        if ((tid & 56u) == 0u) reinterpret_cast<float4*>(sm_red)[wv * 32u + col4] = o;
        __syncthreads();

        if (wave == 0u) {
            const float fo0 = (sm_red[j0] + sm_red[128u + j0])
                            + (sm_red[256u + j0] + sm_red[384u + j0]);
            const float fo1 = (sm_red[j0 + 1u] + sm_red[128u + j0 + 1u])
                            + (sm_red[256u + j0 + 1u] + sm_red[384u + j0 + 1u]);
            const float total_sq = vegajet::ornith_dev::wave64_sum(fo0 * fo0 + fo1 * fo1);
            const float inv_r = rsqrtf(total_sq * (1.0f / 128.0f) + eps);
            const float* zp = qkv_gab_in + static_cast<uint64_t>(mo) * stride_in + z_off + hv * 128u;
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
            int8_t* oq = out_q8 + (static_cast<uint64_t>(mo) * nv + hv) * 128u;
            oq[j0]      = static_cast<int8_t>(max(-127, min(127, __float2int_rn(y0 * inv_d))));
            oq[j0 + 1u] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(y1 * inv_d))));
            if ((lane & 15u) == 0u) {
                out_ds[(static_cast<uint64_t>(mo) * nv + hv) * 4u + (lane >> 4)] = make_float2(d, bsum);
            }
        }
        __syncthreads();
    }

}

// ============================================================================
// 5b-SV. Split-V GDN Recurrence + Gated RMSNorm + Quant: 4 WGs per head (128 WGs/GPU), 16 VGPRs of state.
//     Each WG owns 32 columns of S (one b=32 quant block); a thread owns 16 rows of one column. Columns are
//     independent except for the head-wide RMS, which is exchanged through sq_scratch across one grid
//     barrier per token. The 64 per-lane squares are reduced with the same wave64 tree as kernel 5b, so the
//     result is bit-identical to it. Grid must be exactly nv * 4 WGs (barrier counts gridDim.x). M == 1 only.
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_gdn_recur_splitv_quant_kernel(
    float*       __restrict__ ssm_ring,
    const float* __restrict__ conv_qkv,
    const float* __restrict__ qkv_gab_in,
    const float* __restrict__ decay_in,
    const float* __restrict__ beta_in,
    const float* __restrict__ ssm_norm_w,
    int8_t*      __restrict__ out_q8,
    float2*      __restrict__ out_ds,
    uint32_t*    __restrict__ bar,
    float*       __restrict__ sq_scratch,
    uint32_t nk,
    uint32_t nv,
    uint32_t conv_dim,
    uint32_t stride_in,
    uint32_t z_off,
    uint32_t ring_head,
    uint32_t use_ring,
    uint32_t sense,
    float    eps) {
    const uint32_t hv   = blockIdx.x >> 2;
    const uint32_t part = blockIdx.x & 3u;   // columns 32*part .. 32*part+31
    const uint32_t tid  = threadIdx.x;       // 0..255
    const uint32_t c    = tid & 31u;         // column within the part
    const uint32_t rg   = tid >> 5;          // 0..7: rows 16*rg .. 16*rg+15
    const uint32_t col0 = part * 32u;

    const uint32_t hk = hv % nk;
    const uint64_t head_elems = 128ULL * 128ULL;
    const uint64_t slot_elems = static_cast<uint64_t>(nv) * head_elems;
    float* S_base = ssm_ring + static_cast<uint64_t>(ring_head) * slot_elems + static_cast<uint64_t>(hv) * head_elems;

    __shared__ float sm_q[128];
    __shared__ float sm_k[128];
    __shared__ float sm_red[8 * 32];
    __shared__ float sm_dv[32];
    __shared__ float sm_fo[32];

    const uint32_t i_beg = rg * 16u;
    const uint32_t col   = col0 + c;

    // This thread's 16 rows of one column, addressed around the middle row so every access is one base
    // register plus a signed 13-bit immediate (-4096 .. +3584 B) instead of 16 separate 64-bit addresses.
    float* S_thr = S_base + (i_beg + 8u) * 128u + col;
    float s_reg[16];
    #pragma unroll
    for (uint32_t r = 0u; r < 16u; ++r) {
        s_reg[r] = S_thr[(static_cast<int32_t>(r) - 8) * 128];
    }

    {
        // M == 1 only. A token loop here hoists ~30 per-thread invariants and pushes the kernel from 45 to
        // 80 VGPRs; batches (M > 1) stay on kernel 5b / the WY prefill kernel.
        const uint32_t m = 0u;
        const float* base = conv_qkv + static_cast<uint64_t>(m) * conv_dim;
        if (tid < 128u) {
            sm_q[tid] = base[hk * 128u + tid];
            sm_k[tid] = base[(nk + hk) * 128u + tid];
        }
        __syncthreads();

        const float g_exp = decay_in[static_cast<uint64_t>(m) * nv + hv];
        const float beta  = beta_in[static_cast<uint64_t>(m) * nv + hv];

        float p = 0.0f;
        #pragma unroll
        for (uint32_t r = 0u; r < 16u; ++r) {
            p = fmaf(s_reg[r], sm_k[i_beg + r], p);
        }
        sm_red[rg * 32u + c] = p;
        __syncthreads();

        if (tid < 32u) {
            const float pred = sm_red[tid] + sm_red[32u + tid] + sm_red[64u + tid] + sm_red[96u + tid]
                             + sm_red[128u + tid] + sm_red[160u + tid] + sm_red[192u + tid] + sm_red[224u + tid];
            const float vj = base[(2u * nk + hv) * 128u + col0 + tid];
            sm_dv[tid] = beta * (vj - g_exp * pred);
        }
        __syncthreads();

        const float dv = sm_dv[c];
        float* S_out = (use_ring != 0u)
            ? ssm_ring + static_cast<uint64_t>((ring_head + m + 1u) % VJ_GDN_RING_SLOTS) * slot_elems + static_cast<uint64_t>(hv) * head_elems + (i_beg + 8u) * 128u + col
            : nullptr;

        float o = 0.0f;
        #pragma unroll
        for (uint32_t r = 0u; r < 16u; ++r) {
            const float ki = sm_k[i_beg + r];
            const float qi = sm_q[i_beg + r];
            s_reg[r] = fmaf(ki, dv, g_exp * s_reg[r]);
            o = fmaf(s_reg[r], qi, o);
        }
        sm_red[rg * 32u + c] = o;
        __syncthreads();

        if (tid < 32u) {
            sm_fo[tid] = sm_red[tid] + sm_red[32u + tid] + sm_red[64u + tid] + sm_red[96u + tid]
                       + sm_red[128u + tid] + sm_red[160u + tid] + sm_red[192u + tid] + sm_red[224u + tid];
        }
        __syncthreads();

        // Publish this part's 16 sq values BEFORE writing 2 MB of S_out, and sync ONLY the 4 WGs of head hv
        // with a device-local fence (eliminates __threadfence_system() + 128-WG global barrier + S_out drain stall).
        float* sq = sq_scratch + static_cast<uint64_t>(hv) * 64u;
        const uint32_t l16 = tid & 15u;
        const float fo0 = sm_fo[l16 * 2u];
        const float fo1 = sm_fo[l16 * 2u + 1u];
        if (tid < 16u) {
            sq[part * 16u + tid] = fo0 * fo0 + fo1 * fo1;
        }
        __threadfence();
        vegajet::ornith_dev::grid_barrier_m(bar + static_cast<uint64_t>(hv) * 32u, 4u, sense);
        if (tid == 0u) asm volatile("buffer_wbinvl1\n" ::: "memory");
        __syncthreads();

        // Write S_out right after the 4-WG barrier so s_reg[16] dies before out_q8/out_ds (0 extra VGPRs!).
        if (S_out) {
            #pragma unroll
            for (uint32_t r = 0u; r < 16u; ++r) S_out[(static_cast<int32_t>(r) - 8) * 128] = s_reg[r];
        }

        if (tid < 64u) {
            const float lane_sq = sq[tid];
            const float total_sq = vegajet::ornith_dev::wave64_sum(lane_sq);
            if (tid < 16u) {
                const float inv_r = rsqrtf(total_sq * (1.0f / 128.0f) + eps);
                const uint32_t j0 = col0 + tid * 2u;
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
                if (tid == 0u) {
                    out_ds[(static_cast<uint64_t>(m) * nv + hv) * 4u + part] = make_float2(d, bsum);
                }
            }
        }
        __syncthreads();
    }

    if (use_ring == 0u) {
        #pragma unroll
        for (uint32_t r = 0u; r < 16u; ++r) {
            S_thr[(static_cast<int32_t>(r) - 8) * 128] = s_reg[r];
        }
    }
}

// ============================================================================
// 5b3. Fused M=1 GDN Conv1D + L2-Norm + Gate + Split-V Recurrence + Gated RMSNorm + Quant (0 B spill!)
//      Eliminates 36 ornith_gdn_conv_l2_gate_kernel dispatches/tok and 0.88 MB conv_out roundtrip.
//      128 WGs x 256 thr; race-free for both use_ring != 0 (next slot) and use_ring == 0 (16-WG hk barrier).
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_gdn_conv_recur_splitv_quant_m1_kernel(
    const float* __restrict__ qkv_gab_in,
    const float* __restrict__ conv_w,
    float*       __restrict__ conv_ring,
    const float* __restrict__ dt_bias,
    const float* __restrict__ ssm_a,
    float*       __restrict__ ssm_ring,
    const float* __restrict__ ssm_norm_w,
    int8_t*      __restrict__ out_q8,
    float2*      __restrict__ out_ds,
    uint32_t*    __restrict__ bar,
    float*       __restrict__ sq_scratch,
    uint32_t nk,
    uint32_t nv,
    uint32_t conv_dim,
    uint32_t stride_in,
    uint32_t z_off,
    uint32_t alpha_off,
    uint32_t beta_off,
    uint32_t conv_in_slot,
    uint32_t conv_out_slot,
    uint32_t ssm_ring_head,
    uint32_t use_ring,
    uint32_t sense,
    float    eps) {
    const uint32_t hv   = blockIdx.x >> 2;
    const uint32_t part = blockIdx.x & 3u;
    const uint32_t tid  = threadIdx.x;
    const uint32_t c    = tid & 31u;
    const uint32_t rg   = tid >> 5;
    const uint32_t col0 = part * 32u;
    const uint32_t hk   = hv % nk;

    __shared__ float sm_q[128];
    __shared__ float sm_k[128];
    __shared__ float sm_red[8 * 32];
    __shared__ float sm_dv[32];
    __shared__ float sm_fo[32];

    // ---- Phase 0: Conv1D + SiLU + L2-norm (Q/K) + V conv + alpha/beta gate ----
    // conv_in_slot != conv_out_slot always -> 100% race-free with ZERO inter-WG barrier!
    {
        const uint64_t cslot_stride = 3ULL * conv_dim;
        const float*   cr_in  = conv_ring + static_cast<uint64_t>(conv_in_slot)  * cslot_stride;
        float*         cr_out = conv_ring + static_cast<uint64_t>(conv_out_slot) * cslot_stride;

        const uint32_t is_k = tid >> 7;        // 0=Q (tid 0..127), 1=K (tid 128..255)
        const uint32_t l    = tid & 127u;
        const uint32_t ch   = (is_k ? (nk + hk) : hk) * 128u + l;
        const float4   w    = reinterpret_cast<const float4*>(conv_w)[ch];
        const float    s0   = cr_in[0ULL * conv_dim + ch];
        const float    s1   = cr_in[1ULL * conv_dim + ch];
        const float    s2   = cr_in[2ULL * conv_dim + ch];
        const float    x    = qkv_gab_in[ch];
        if (hv < nk && part == 0u) {
            cr_out[0ULL * conv_dim + ch] = s1;
            cr_out[1ULL * conv_dim + ch] = s2;
            cr_out[2ULL * conv_dim + ch] = x;
        }
        const float    acc  = fmaf(w.x, s0, fmaf(w.y, s1, fmaf(w.z, s2, w.w * x)));
        float          y    = acc / (1.0f + __expf(-acc));

        const float sq_w = vegajet::ornith_dev::wave64_sum(y * y);
        if ((tid & 63u) == 0u) sm_red[tid >> 6] = sq_w;
        __syncthreads();
        float inv_n = rsqrtf(sm_red[is_k ? 2u : 0u] + sm_red[is_k ? 3u : 1u] + eps);
        if (!is_k) inv_n *= 0.08838834764831845f; // 1/sqrt(128)
        y *= inv_n;
        if (!is_k) sm_q[l] = y; else sm_k[l] = y;

        // Disjoint V channels (owned exclusively by (hv, part)): compute + update conv_ring immediately.
        if (tid < 32u) {
            const uint32_t ch_v = (2u * nk + hv) * 128u + col0 + tid;
            const float4   wv   = reinterpret_cast<const float4*>(conv_w)[ch_v];
            const float    sv0  = cr_in[0ULL * conv_dim + ch_v];
            const float    sv1  = cr_in[1ULL * conv_dim + ch_v];
            const float    sv2  = cr_in[2ULL * conv_dim + ch_v];
            const float    xv   = qkv_gab_in[ch_v];
            const float    av   = fmaf(wv.x, sv0, fmaf(wv.y, sv1, fmaf(wv.z, sv2, wv.w * xv)));
            sm_dv[tid] = av / (1.0f + __expf(-av));
            cr_out[0ULL * conv_dim + ch_v] = sv1;
            cr_out[1ULL * conv_dim + ch_v] = sv2;
            cr_out[2ULL * conv_dim + ch_v] = xv;
        }
        if (tid == 0u) {
            const float a  = qkv_gab_in[alpha_off + hv] + dt_bias[hv];
            const float sp = (a > 20.0f) ? a : log1pf(__expf(a));
            sm_fo[0] = __expf(sp * ssm_a[hv]);                         // g_exp
            sm_fo[1] = 1.0f / (1.0f + __expf(-qkv_gab_in[beta_off + hv])); // beta
        }
    }
    __syncthreads();

    // ---- Phase 1: Split-V Recurrence + 4-WG Head Barrier + Gated RMSNorm + Quant ----
    const uint64_t head_elems = 128ULL * 128ULL;
    const uint64_t slot_elems = static_cast<uint64_t>(nv) * head_elems;
    float* S_base = ssm_ring + static_cast<uint64_t>(ssm_ring_head) * slot_elems + static_cast<uint64_t>(hv) * head_elems;
    const uint32_t i_beg = rg * 16u;
    const uint32_t col   = col0 + c;
    float* S_thr = S_base + (i_beg + 8u) * 128u + col;

    float s_reg[16];
    #pragma unroll
    for (uint32_t r = 0u; r < 16u; ++r) {
        s_reg[r] = S_thr[(static_cast<int32_t>(r) - 8) * 128];
    }

    const float g_exp = sm_fo[0];
    const float beta  = sm_fo[1];

    float p = 0.0f;
    #pragma unroll
    for (uint32_t r = 0u; r < 16u; ++r) {
        p = fmaf(s_reg[r], sm_k[i_beg + r], p);
    }
    sm_red[rg * 32u + c] = p;
    __syncthreads();

    if (tid < 32u) {
        const float pred = sm_red[tid] + sm_red[32u + tid] + sm_red[64u + tid] + sm_red[96u + tid]
                         + sm_red[128u + tid] + sm_red[160u + tid] + sm_red[192u + tid] + sm_red[224u + tid];
        const float vj = sm_dv[tid];
        sm_dv[tid] = beta * (vj - g_exp * pred);
    }
    __syncthreads();

    const float dv = sm_dv[c];
    float* S_out = (use_ring != 0u)
        ? ssm_ring + static_cast<uint64_t>((ssm_ring_head + 1u) % VJ_GDN_RING_SLOTS) * slot_elems + static_cast<uint64_t>(hv) * head_elems + (i_beg + 8u) * 128u + col
        : S_thr;

    float o = 0.0f;
    #pragma unroll
    for (uint32_t r = 0u; r < 16u; ++r) {
        const float ki = sm_k[i_beg + r];
        const float qi = sm_q[i_beg + r];
        const float sv = fmaf(ki, dv, g_exp * s_reg[r]);
        S_out[(static_cast<int32_t>(r) - 8) * 128] = sv;
        o = fmaf(sv, qi, o);
    }
    sm_red[rg * 32u + c] = o;
    __syncthreads();

    if (tid < 32u) {
        sm_fo[tid] = sm_red[tid] + sm_red[32u + tid] + sm_red[64u + tid] + sm_red[96u + tid]
                   + sm_red[128u + tid] + sm_red[160u + tid] + sm_red[192u + tid] + sm_red[224u + tid];
    }
    __syncthreads();

    float* sq = sq_scratch + static_cast<uint64_t>(hv) * 64u;
    const uint32_t l16 = tid & 15u;
    const float fo0 = sm_fo[l16 * 2u];
    const float fo1 = sm_fo[l16 * 2u + 1u];
    if (tid < 16u) {
        sq[part * 16u + tid] = fo0 * fo0 + fo1 * fo1;
    }
    __threadfence();
    vegajet::ornith_dev::grid_barrier_m(bar + static_cast<uint64_t>(hv) * 32u, 4u, sense);
    if (tid == 0u) asm volatile("buffer_wbinvl1\n" ::: "memory");
    __syncthreads();

    if (tid < 64u) {
        const float lane_sq  = sq[tid];
        const float total_sq = vegajet::ornith_dev::wave64_sum(lane_sq);
        if (tid < 16u) {
            const float inv_r = rsqrtf(total_sq * (1.0f / 128.0f) + eps);
            const uint32_t j0 = col0 + tid * 2u;
            const float* zp = qkv_gab_in + z_off + hv * 128u;
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
            int8_t* oq = out_q8 + static_cast<uint64_t>(hv) * 128u;
            oq[j0]      = static_cast<int8_t>(max(-127, min(127, __float2int_rn(y0 * inv_d))));
            oq[j0 + 1u] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(y1 * inv_d))));
            if (tid == 0u) {
                out_ds[static_cast<uint64_t>(hv) * 4u + part] = make_float2(d, bsum);
            }
        }
    }
}

// ============================================================================
// 5c. GDN Chunked WY/Neumann Prefill (C=16, ZERO SOLVE_TRI, 0 B spill!)
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 3) __attribute__((amdgpu_num_vgpr(60)))
void ornith_gdn_wy_prefill_quant_kernel(
    float*       __restrict__ S_all,
    const float* __restrict__ conv_qkv,
    const float* __restrict__ qkv_gab_in,
    const float* __restrict__ decay_in,
    const float* __restrict__ beta_in,
    const float* __restrict__ ssm_norm_w,
    int8_t*      __restrict__ out_q8,
    float2*      __restrict__ out_ds,
    uint32_t M,
    uint32_t nk,
    uint32_t nv,
    uint32_t conv_dim,
    uint32_t stride_in,
    uint32_t z_off,
    float    eps) {
    const uint32_t hv  = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (hv >= nv) return;

    constexpr uint32_t C = 16u;
    const uint32_t r  = tid >> 4;
    const uint32_t c  = tid & 15u;
    const uint32_t hk = hv % nk;
    float* Sh = S_all + static_cast<uint64_t>(hv) * 128ULL * 128ULL;

    __shared__ float sm_cum_g[C];
    __shared__ float sm_P[C * C];
    __shared__ float sm_T[C * C];
    __shared__ float sm_QK[C * C];
    __shared__ float sm_tmp[C * C];
    __shared__ float sm_rhs[C * C];
    __shared__ float sm_dv[C * C];
    __shared__ float sm_chunk_O[C * 128];
    __shared__ float sm_S[128 * 16]; // 8 KiB LDS tile of Sh -> 16x fewer global Sh reads!
    __shared__ float sm_sq[2];

    for (uint32_t m0 = 0u; m0 < M; m0 += C) {
        const uint32_t clen = min(C, M - m0);
        if (tid == 0u) {
            float acc = 1.0f;
            #pragma nounroll
            for (uint32_t t = 0u; t < C; ++t) {
                if (t < clen) acc *= decay_in[static_cast<uint64_t>(m0 + t) * nv + hv];
                sm_cum_g[t] = acc;
            }
        }
        __syncthreads();

        const bool valid_rc = (r < clen && c < clen);
        const float ratio_rc = (valid_rc && r >= c) ? (sm_cum_g[r] / fmaxf(sm_cum_g[c], 1e-30f)) : 0.0f;
        float kk = 0.0f, qk = 0.0f;
        if (valid_rc && r >= c) {
            const float* br = conv_qkv + static_cast<uint64_t>(m0 + r) * conv_dim;
            const float* bc = conv_qkv + static_cast<uint64_t>(m0 + c) * conv_dim;
            const float* qr = br + hk * 128u;
            const float* kr = br + (nk + hk) * 128u;
            const float* kc = bc + (nk + hk) * 128u;
            #pragma nounroll
            for (uint32_t d = 0u; d < 128u; ++d) {
                const float kcv = kc[d];
                if (r > c) kk = fmaf(kr[d], kcv, kk);
                qk = fmaf(qr[d], kcv, qk);
            }
        }
        const float br_beta = (r < clen) ? beta_in[static_cast<uint64_t>(m0 + r) * nv + hv] : 0.0f;
        const float a_val = (valid_rc && r > c) ? (-br_beta * ratio_rc * kk) : 0.0f;
        sm_P[tid]  = a_val;
        sm_T[tid]  = (r == c) ? 1.0f : a_val;
        sm_QK[tid] = (valid_rc && r >= c) ? (ratio_rc * qk) : 0.0f;
        __syncthreads();

        #pragma nounroll
        for (uint32_t s = 0u; s < 3u; ++s) {
            float p2 = 0.0f;
            if (r > c) {
                #pragma nounroll
                for (uint32_t k = c + 1u; k < r; ++k) p2 = fmaf(sm_P[r * C + k], sm_P[k * C + c], p2);
            }
            sm_tmp[tid] = p2; __syncthreads();
            sm_P[tid] = p2;   __syncthreads();

            float tp = sm_T[tid];
            if (r > c) {
                #pragma nounroll
                for (uint32_t k = c + 1u; k <= r; ++k) tp = fmaf(sm_T[r * C + k], sm_P[k * C + c], tp);
            }
            sm_tmp[tid] = tp; __syncthreads();
            sm_T[tid] = tp;   __syncthreads();
        }

        const float g_end = sm_cum_g[clen - 1u];
        for (uint32_t j0 = 0u; j0 < 128u; j0 += 16u) {
            const uint32_t j = j0 + c;
            // Cooperatively load 128x16 tile of Sh ONCE into LDS!
            #pragma nounroll
            for (uint32_t i = r; i < 128u; i += C) {
                sm_S[i * 16u + c] = Sh[i * 128u + j];
            }
            __syncthreads();

            float sk = 0.0f, sq = 0.0f;
            if (r < clen) {
                const float* br = conv_qkv + static_cast<uint64_t>(m0 + r) * conv_dim;
                const float* qr = br + hk * 128u;
                const float* kr = br + (nk + hk) * 128u;
                #pragma nounroll
                for (uint32_t i = 0u; i < 128u; ++i) {
                    const float sij = sm_S[i * 16u + c];
                    sk = fmaf(kr[i], sij, sk);
                    sq = fmaf(qr[i], sij, sq);
                }
                sk *= sm_cum_g[r];
                sq *= sm_cum_g[r];
                sm_rhs[tid] = br_beta * (br[(2u * nk + hv) * 128u + j] - sk);
            } else {
                sm_rhs[tid] = 0.0f;
            }
            __syncthreads();

            float dv_rj = 0.0f;
            if (r < clen) {
                #pragma nounroll
                for (uint32_t u = 0u; u <= r; ++u) dv_rj = fmaf(sm_T[r * C + u], sm_rhs[u * C + c], dv_rj);
            }
            sm_dv[tid] = dv_rj;
            __syncthreads();

            if (r < clen) {
                float intra = 0.0f;
                #pragma nounroll
                for (uint32_t u = 0u; u <= r; ++u) intra = fmaf(sm_QK[r * C + u], sm_dv[u * C + c], intra);
                sm_chunk_O[r * 128u + j] = sq + intra;
            }
            #pragma nounroll
            for (uint32_t i = r; i < 128u; i += C) {
                float s_upd = g_end * sm_S[i * 16u + c];
                #pragma nounroll
                for (uint32_t u = 0u; u < clen; ++u) {
                    const float ku_i = conv_qkv[static_cast<uint64_t>(m0 + u) * conv_dim + (nk + hk) * 128u + i];
                    s_upd = fmaf((g_end / fmaxf(sm_cum_g[u], 1e-30f)) * ku_i, sm_dv[u * C + c], s_upd);
                }
                Sh[i * 128u + j] = s_upd;
            }
            __syncthreads();
        }

        for (uint32_t t = 0u; t < clen; ++t) {
            if (tid < 128u) {
                const float o_j = sm_chunk_O[t * 128u + tid];
                float sq_v = vegajet::ornith_dev::wave64_sum(o_j * o_j);
                if ((tid & 63u) == 0u) sm_sq[tid >> 6] = sq_v;
            }
            __syncthreads();
            if (tid < 128u) {
                const float inv_r  = rsqrtf((sm_sq[0] + sm_sq[1]) * (1.0f / 128.0f) + eps);
                const float z_j    = qkv_gab_in[static_cast<uint64_t>(m0 + t) * stride_in + z_off + hv * 128u + tid];
                const float silu_z = z_j / (1.0f + __expf(-z_j));
                const float y_j    = (sm_chunk_O[t * 128u + tid] * inv_r * ssm_norm_w[tid]) * silu_z;
                const float amax   = vegajet::ornith_dev::block32_max(fabsf(y_j));
                const float bsum   = vegajet::ornith_dev::block32_sum(y_j);
                const float d      = amax * (1.0f / 127.0f);
                const float inv_d  = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                out_q8[(static_cast<uint64_t>(m0 + t) * nv + hv) * 128u + tid] =
                    static_cast<int8_t>(max(-127, min(127, __float2int_rn(y_j * inv_d))));
                if ((tid & 31u) == 0u) {
                    out_ds[(static_cast<uint64_t>(m0 + t) * nv + hv) * 4u + (tid >> 5)] = make_float2(d, bsum);
                }
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// 6. Full GQA Attention Prep (Fused Q+K+V input buffer [M, 8704] -> Q/Gate/Q8_0 KV Cache)
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_attn_qkv_prep_kernel(
    const float* __restrict__ qkv_fused,
    const float* __restrict__ q_norm_w,
    const float* __restrict__ k_norm_w,
    float*       __restrict__ q_out,
    float*       __restrict__ gate_out,
    uint8_t*     __restrict__ k_cache_q8,
    uint8_t*     __restrict__ v_cache_q8,
    uint32_t M,
    uint32_t pos0,
    uint32_t n_q_local,
    uint32_t n_kv_local,
    uint32_t rope_dims,
    float    rope_base,
    float    eps) {
    const uint32_t h_idx = blockIdx.x;
    const uint32_t m     = blockIdx.y;
    const uint32_t d     = threadIdx.x;
    if (m >= M) return;

    __shared__ float sm_v[256];
    __shared__ float sm_sq[4];
    const uint32_t wave = d >> 6;
    const uint32_t lane = d & 63u;
    const float pos_f = static_cast<float>(pos0 + m);
    const uint32_t half_r = rope_dims >> 1;
    const uint32_t qg_tot = n_q_local * 512u;
    const uint32_t kv_tot = n_kv_local * 256u;
    const float* row = qkv_fused + static_cast<uint64_t>(m) * (qg_tot + 2u * kv_tot);

    if (h_idx < n_q_local) {
        const float* qg = row + h_idx * 512u;
        const float qv  = qg[d];
        gate_out[(static_cast<uint64_t>(m) * n_q_local + h_idx) * 256u + d] = qg[256u + d];

        float sq = vegajet::ornith_dev::wave64_sum(qv * qv);
        if (lane == 0u) sm_sq[wave] = sq;
        __syncthreads();
        const float inv_r = rsqrtf((sm_sq[0] + sm_sq[1] + sm_sq[2] + sm_sq[3]) * (1.0f / 256.0f) + eps);
        sm_v[d] = qv * inv_r * q_norm_w[d];
        __syncthreads();

        float q_rot = sm_v[d];
        if (d < rope_dims) {
            const uint32_t p = (d < half_r) ? d : (d - half_r);
            const float theta = pos_f * __powf(rope_base, -2.0f * static_cast<float>(p) / static_cast<float>(rope_dims));
            float s_th, c_th;
            __sincosf(theta, &s_th, &c_th);
            q_rot = (d < half_r) ? (sm_v[p] * c_th - sm_v[p + half_r] * s_th)
                                 : (sm_v[p] * s_th + sm_v[p + half_r] * c_th);
        }
        q_out[(static_cast<uint64_t>(m) * n_q_local + h_idx) * 256u + d] = q_rot;
    } else if (h_idx < n_q_local + n_kv_local) {
        const uint32_t kv_h = h_idx - n_q_local;
        const float* kr = row + qg_tot + kv_h * 256u;
        const float* vr = row + qg_tot + kv_tot + kv_h * 256u;
        const float kv  = kr[d];

        float sq = vegajet::ornith_dev::wave64_sum(kv * kv);
        if (lane == 0u) sm_sq[wave] = sq;
        __syncthreads();
        const float inv_r = rsqrtf((sm_sq[0] + sm_sq[1] + sm_sq[2] + sm_sq[3]) * (1.0f / 256.0f) + eps);
        sm_v[d] = kv * inv_r * k_norm_w[d];
        __syncthreads();

        float k_rot = sm_v[d];
        if (d < rope_dims) {
            const uint32_t p = (d < half_r) ? d : (d - half_r);
            const float theta = pos_f * __powf(rope_base, -2.0f * static_cast<float>(p) / static_cast<float>(rope_dims));
            float s_th, c_th;
            __sincosf(theta, &s_th, &c_th);
            k_rot = (d < half_r) ? (sm_v[p] * c_th - sm_v[p + half_r] * s_th)
                                 : (sm_v[p] * s_th + sm_v[p + half_r] * c_th);
        }

        const uint64_t slot_off = (static_cast<uint64_t>(pos0 + m) * n_kv_local + kv_h) * 272ULL;
        uint8_t* kc = k_cache_q8 + slot_off;
        uint8_t* vc = v_cache_q8 + slot_off;
        const uint32_t blk = d >> 5;
        const uint32_t sub = d & 31u;
        {
            const float amax = vegajet::ornith_dev::block32_max(fabsf(k_rot));
            const float sc   = amax * (1.0f / 127.0f);
            const float inv  = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            reinterpret_cast<int8_t*>(kc + 16u)[d] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(k_rot * inv))));
            if (sub == 0u) reinterpret_cast<uint16_t*>(kc)[blk] = __half_as_ushort(__float2half(sc));
        }
        {
            const float vv   = vr[d];
            const float amax = vegajet::ornith_dev::block32_max(fabsf(vv));
            const float sc   = amax * (1.0f / 127.0f);
            const float inv  = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            reinterpret_cast<int8_t*>(vc + 16u)[d] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(vv * inv))));
            if (sub == 0u) reinterpret_cast<uint16_t*>(vc)[blk] = __half_as_ushort(__float2half(sc));
        }
    }
}

// ============================================================================
// 7. Causal Wave64 FlashAttention on Q8_0 KV Cache + Sigmoid(Gate) + INT8 Quant (+ out_ds)
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_attn_flash_quant_kernel(
    const float*   __restrict__ q_all,
    const float*   __restrict__ gate_all,
    const uint8_t* __restrict__ k_cache_q8,
    const uint8_t* __restrict__ v_cache_q8,
    int8_t*        __restrict__ out_q8,
    float2*        __restrict__ out_ds,
    uint32_t M,
    uint32_t pos0,
    uint32_t n_q_local,
    uint32_t n_kv_local) {
    const uint32_t h    = blockIdx.x;
    const uint32_t m    = blockIdx.y;
    const uint32_t d    = threadIdx.x;
    if (h >= n_q_local || m >= M) return;

    const uint32_t n_kv_raw = n_kv_local & 0xFFu;
    const uint32_t gqa_grp  = (n_kv_local >> 24) & 0xFFu;
    const uint32_t kv_h     = (gqa_grp != 0u)
        ? ((((n_kv_local >> 8) & 0xFFu) + h) / gqa_grp - ((n_kv_local >> 16) & 0xFFu))
        : ((h * n_kv_raw) / n_q_local);
    const uint32_t seqlen = pos0 + m + 1u;
    const uint32_t wave   = d >> 6;
    const uint32_t lane   = d & 63u;
    const uint32_t blk    = d >> 5;
    const float    qv     = q_all[(static_cast<uint64_t>(m) * n_q_local + h) * 256u + d];

    // Positions are processed 8 per step: all 8 K rows and 8 V rows are requested up front (16 loads in flight
    // instead of 1) and the WG synchronises twice per 8 positions instead of twice per position. The online
    // softmax update still runs position by position in the original order, so the result is bit-identical.
    __shared__ float sm_red[32];
    float m_run = -1e30f, l_run = 0.0f, o_d = 0.0f;
    const uint32_t t_last = seqlen - 1u;

    #pragma nounroll
    for (uint32_t t0 = 0u; t0 < seqlen; t0 += 8u) {
        float dots[8];
        float vvs[8];
        #pragma unroll
        for (uint32_t j = 0u; j < 8u; ++j) {
            const uint32_t t   = min(t0 + j, t_last); // tail positions re-read the last row and are ignored below
            const uint64_t off = (static_cast<uint64_t>(t) * n_kv_raw + kv_h) * 272ULL;
            const uint8_t* kc  = k_cache_q8 + off;
            const uint8_t* vc  = v_cache_q8 + off;
            const float ksc    = vegajet::ornith_dev::fp16_to_f32(reinterpret_cast<const uint16_t*>(kc)[blk]);
            const float kv     = ksc * static_cast<float>(reinterpret_cast<const int8_t*>(kc + 16u)[d]);
            const float vsc    = vegajet::ornith_dev::fp16_to_f32(reinterpret_cast<const uint16_t*>(vc)[blk]);
            vvs[j]  = vsc * static_cast<float>(reinterpret_cast<const int8_t*>(vc + 16u)[d]);
            dots[j] = qv * kv;
        }
        #pragma unroll
        for (uint32_t j = 0u; j < 8u; ++j) {
            const float dot = vegajet::ornith_dev::wave64_sum(dots[j]);
            if (lane == 0u) sm_red[j * 4u + wave] = dot;
        }
        __syncthreads();
        #pragma unroll
        for (uint32_t j = 0u; j < 8u; ++j) {
            if (t0 + j < seqlen) {
                const float score = (sm_red[j * 4u] + sm_red[j * 4u + 1u] + sm_red[j * 4u + 2u] + sm_red[j * 4u + 3u]) * 0.0625f;
                const float m_new = fmaxf(m_run, score);
                const float alpha = __expf(m_run - m_new);
                const float beta  = __expf(score - m_new);
                l_run = l_run * alpha + beta;
                m_run = m_new;
                o_d = fmaf(beta, vvs[j], o_d * alpha);
            }
        }
        __syncthreads();
    }

    const float attn_v = (l_run > 0.0f) ? (o_d / l_run) : 0.0f;
    const float gv     = gate_all[(static_cast<uint64_t>(m) * n_q_local + h) * 256u + d];
    const float y_d    = attn_v * (1.0f / (1.0f + __expf(-gv)));
    const float amax   = vegajet::ornith_dev::block32_max(fabsf(y_d));
    const float bsum   = vegajet::ornith_dev::block32_sum(y_d);
    const float sc     = amax * (1.0f / 127.0f);
    const float inv_d  = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    out_q8[(static_cast<uint64_t>(m) * n_q_local + h) * 256u + d] =
        static_cast<int8_t>(max(-127, min(127, __float2int_rn(y_d * inv_d))));
    if ((d & 31u) == 0u) out_ds[(static_cast<uint64_t>(m) * n_q_local + h) * 8u + blk] = make_float2(sc, bsum);
}

// 7b. Fused M=1 GQA Attention Prep + FlashAttention + Gate + Quant (-12 dispatches/tok + zero q_buf/gate_buf roundtrip!)
//     16 WGs x 256 thr: all WGs compute K/V at pos0 into sm_k_cur/sm_v_cur (WG 0 writes KV cache), each WG
//     norms+RoPEs its own Q head into qv, runs cached 8-pos flash over 0..pos0-1, then final step t=pos0.
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_attn_prep_flash_m1_kernel(
    const float*   __restrict__ qkv_fused,
    const float*   __restrict__ q_norm_w,
    const float*   __restrict__ k_norm_w,
    uint8_t*       __restrict__ k_cache_q8,
    uint8_t*       __restrict__ v_cache_q8,
    int8_t*        __restrict__ out_q8,
    float2*        __restrict__ out_ds,
    uint32_t pos0,
    uint32_t n_q_local,
    uint32_t n_kv_local,
    uint32_t rope_dims,
    float    rope_base,
    float    eps) {
    const uint32_t h    = blockIdx.x;
    const uint32_t d    = threadIdx.x;
    if (h >= n_q_local) return;

    __shared__ float sm_v[256];
    __shared__ float sm_k_cur[256];
    __shared__ float sm_v_cur[256];
    __shared__ float sm_red[32];

    const uint32_t wave   = d >> 6;
    const uint32_t lane   = d & 63u;
    const uint32_t blk    = d >> 5;
    const uint32_t sub    = d & 31u;
    const uint32_t n_kv_raw = n_kv_local & 0xFFu;
    const uint32_t gqa_grp  = (n_kv_local >> 24) & 0xFFu;
    const uint32_t q_off_p  = (n_kv_local >> 8) & 0xFFu;
    const uint32_t kv_off_p = (n_kv_local >> 16) & 0xFFu;
    const uint32_t kv_h     = (gqa_grp != 0u) ? ((q_off_p + h) / gqa_grp - kv_off_p) : ((h * n_kv_raw) / n_q_local);
    const uint32_t kv_prev  = (h > 0u) ? ((gqa_grp != 0u) ? ((q_off_p + h - 1u) / gqa_grp - kv_off_p) : (((h - 1u) * n_kv_raw) / n_q_local)) : 0xFFFFFFFFu;
    const uint32_t qg_tot = n_q_local * 512u;
    const uint32_t kv_tot = n_kv_raw * 256u;
    const uint32_t half_r = rope_dims >> 1;
    const float    pos_f  = static_cast<float>(pos0);

    // 1. K/V at pos0: norm + RoPE + Q8_0 quantize -> sm_k_cur/sm_v_cur; only WG 0 (per kv_h) writes cache.
    {
        const float* kr = qkv_fused + qg_tot + kv_h * 256u;
        const float* vr = qkv_fused + qg_tot + kv_tot + kv_h * 256u;
        const float kv_raw = kr[d];
        const float sq = vegajet::ornith_dev::wave64_sum(kv_raw * kv_raw);
        if (lane == 0u) sm_red[wave] = sq;
        __syncthreads();
        const float inv_r = rsqrtf((sm_red[0] + sm_red[1] + sm_red[2] + sm_red[3]) * (1.0f / 256.0f) + eps);
        sm_v[d] = kv_raw * inv_r * k_norm_w[d];
        __syncthreads();

        float k_rot = sm_v[d];
        if (d < rope_dims) {
            const uint32_t p = (d < half_r) ? d : (d - half_r);
            const float theta = pos_f * __powf(rope_base, -2.0f * static_cast<float>(p) / static_cast<float>(rope_dims));
            float s_th, c_th;
            __sincosf(theta, &s_th, &c_th);
            k_rot = (d < half_r) ? (sm_v[p] * c_th - sm_v[p + half_r] * s_th)
                                 : (sm_v[p] * s_th + sm_v[p + half_r] * c_th);
        }
        const uint64_t slot_off = (static_cast<uint64_t>(pos0) * n_kv_raw + kv_h) * 272ULL;
        const bool write_kv = (h == 0u) || (kv_h != kv_prev);
        {
            const float amax = vegajet::ornith_dev::block32_max(fabsf(k_rot));
            const float sc   = amax * (1.0f / 127.0f);
            const float inv  = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            const int8_t kq  = static_cast<int8_t>(max(-127, min(127, __float2int_rn(k_rot * inv))));
            const uint16_t hsc = __half_as_ushort(__float2half(sc));
            sm_k_cur[d] = vegajet::ornith_dev::fp16_to_f32(hsc) * static_cast<float>(kq);
            if (write_kv) {
                reinterpret_cast<int8_t*>(k_cache_q8 + slot_off + 16u)[d] = kq;
                if (sub == 0u) reinterpret_cast<uint16_t*>(k_cache_q8 + slot_off)[blk] = hsc;
            }
        }
        {
            const float vv   = vr[d];
            const float amax = vegajet::ornith_dev::block32_max(fabsf(vv));
            const float sc   = amax * (1.0f / 127.0f);
            const float inv  = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            const int8_t vq  = static_cast<int8_t>(max(-127, min(127, __float2int_rn(vv * inv))));
            const uint16_t hsc = __half_as_ushort(__float2half(sc));
            sm_v_cur[d] = vegajet::ornith_dev::fp16_to_f32(hsc) * static_cast<float>(vq);
            if (write_kv) {
                reinterpret_cast<int8_t*>(v_cache_q8 + slot_off + 16u)[d] = vq;
                if (sub == 0u) reinterpret_cast<uint16_t*>(v_cache_q8 + slot_off)[blk] = hsc;
            }
        }
    }
    __syncthreads();

    // 2. Q head h: norm + RoPE directly into register qv.
    float qv;
    {
        const float* qg = qkv_fused + h * 512u;
        const float qr = qg[d];
        const float sq = vegajet::ornith_dev::wave64_sum(qr * qr);
        if (lane == 0u) sm_red[wave] = sq;
        __syncthreads();
        const float inv_r = rsqrtf((sm_red[0] + sm_red[1] + sm_red[2] + sm_red[3]) * (1.0f / 256.0f) + eps);
        sm_v[d] = qr * inv_r * q_norm_w[d];
        __syncthreads();
        qv = sm_v[d];
        if (d < rope_dims) {
            const uint32_t p = (d < half_r) ? d : (d - half_r);
            const float theta = pos_f * __powf(rope_base, -2.0f * static_cast<float>(p) / static_cast<float>(rope_dims));
            float s_th, c_th;
            __sincosf(theta, &s_th, &c_th);
            qv = (d < half_r) ? (sm_v[p] * c_th - sm_v[p + half_r] * s_th)
                              : (sm_v[p] * s_th + sm_v[p + half_r] * c_th);
        }
    }
    __syncthreads();

    // 3. Causal 8-pos cached flash over 0..pos0-1, then final step t=pos0 from sm_k_cur/sm_v_cur.
    float m_run = -1e30f, l_run = 0.0f, o_d = 0.0f;
    if (pos0 > 0u) {
        const uint32_t t_last = pos0 - 1u;
        #pragma nounroll
        for (uint32_t t0 = 0u; t0 < pos0; t0 += 8u) {
            float dots[8];
            float vvs[8];
            #pragma unroll
            for (uint32_t j = 0u; j < 8u; ++j) {
                const uint32_t t   = min(t0 + j, t_last);
                const uint64_t off = (static_cast<uint64_t>(t) * n_kv_raw + kv_h) * 272ULL;
                const uint8_t* kc  = k_cache_q8 + off;
                const uint8_t* vc  = v_cache_q8 + off;
                const float ksc    = vegajet::ornith_dev::fp16_to_f32(reinterpret_cast<const uint16_t*>(kc)[blk]);
                const float kv     = ksc * static_cast<float>(reinterpret_cast<const int8_t*>(kc + 16u)[d]);
                const float vsc    = vegajet::ornith_dev::fp16_to_f32(reinterpret_cast<const uint16_t*>(vc)[blk]);
                vvs[j]  = vsc * static_cast<float>(reinterpret_cast<const int8_t*>(vc + 16u)[d]);
                dots[j] = qv * kv;
            }
            #pragma unroll
            for (uint32_t j = 0u; j < 8u; ++j) {
                const float dot = vegajet::ornith_dev::wave64_sum(dots[j]);
                if (lane == 0u) sm_red[j * 4u + wave] = dot;
            }
            __syncthreads();
            #pragma unroll
            for (uint32_t j = 0u; j < 8u; ++j) {
                if (t0 + j < pos0) {
                    const float score = (sm_red[j * 4u] + sm_red[j * 4u + 1u] + sm_red[j * 4u + 2u] + sm_red[j * 4u + 3u]) * 0.0625f;
                    const float m_new = fmaxf(m_run, score);
                    const float alpha = __expf(m_run - m_new);
                    const float beta  = __expf(score - m_new);
                    l_run = l_run * alpha + beta;
                    m_run = m_new;
                    o_d = fmaf(beta, vvs[j], o_d * alpha);
                }
            }
            __syncthreads();
        }
    }
    // Final position t = pos0 (exact bit-identical quantized K/V from sm_k_cur/sm_v_cur).
    {
        const float dot = vegajet::ornith_dev::wave64_sum(qv * sm_k_cur[d]);
        if (lane == 0u) sm_red[wave] = dot;
        __syncthreads();
        const float score = (sm_red[0] + sm_red[1] + sm_red[2] + sm_red[3]) * 0.0625f;
        const float m_new = fmaxf(m_run, score);
        const float alpha = __expf(m_run - m_new);
        const float beta  = __expf(score - m_new);
        l_run = l_run * alpha + beta;
        m_run = m_new;
        o_d = fmaf(beta, sm_v_cur[d], o_d * alpha);
    }

    const float attn_v = (l_run > 0.0f) ? (o_d / l_run) : 0.0f;
    const float gv     = qkv_fused[h * 512u + 256u + d];
    const float y_d    = attn_v * (1.0f / (1.0f + __expf(-gv)));
    const float amax   = vegajet::ornith_dev::block32_max(fabsf(y_d));
    const float bsum   = vegajet::ornith_dev::block32_sum(y_d);
    const float sc     = amax * (1.0f / 127.0f);
    const float inv_d  = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    out_q8[static_cast<uint64_t>(h) * 256u + d] =
        static_cast<int8_t>(max(-127, min(127, __float2int_rn(y_d * inv_d))));
    if (sub == 0u) out_ds[static_cast<uint64_t>(h) * 8u + blk] = make_float2(sc, bsum);
}

