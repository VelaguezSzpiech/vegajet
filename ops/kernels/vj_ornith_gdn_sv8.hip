// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// A2: separate, selectable Split-V=8 fused GDN ConvRec for the current symmetric head split
// (nk = 8, nv = 32; one value head = 8 WGs x 16 columns of S), emitted as TWO ORDERED DISPATCHES.
//
// Numerically identical to ornith_gdn_conv_recur_splitv_quant_m1_kernel:
//   * Phase 0 conv1d+SiLU, Q/K wave64 L2-norm and alpha/beta are the same per-channel order;
//   * the recurrence keeps the same 8 row groups x 16 rows ascending fma order, the same
//     8-term ascending pred / o sums and the same g*s + k*dv state update;
//   * the head RMS uses the same 64 pair-squares in the same lane positions (global column
//     pair index -> sq[lane]) and the same wave64 tree;
//   * the block-32 quant uses the same 16-lane amax/sum XOR trees over the full 32-column
//     block; the two WGs sharing a block recompute it from identical data, so d/bsum/q8 are
//     bit-identical to the Split-V=4 kernel.
//
// Why two dispatches (A2 audit): the earlier single kernel published fo + pair-squares, then
// met the other 7 WGs of the head at a global spin barrier before reading the partner half.
// HIP/AQL provides no full-grid co-residency guarantee for a plain dispatch (there is no
// cooperative-launch contract here), and the barrier helper's bounded spin (grid_barrier_m)
// silently falls through after its 5e6-iteration cap, i.e. a late sibling turns into a data race
// rather than a visible hang. Register counting cannot fix that: at the 48-VGPR `stream`
// ceiling the modeled occupancy is 5 WGs/CU (300 WGs of the 256 launched), i.e. a resource
// estimate for an otherwise idle GPU, not a hardware reservation; any co-scheduled dispatch, a
// different realized occupancy, or a stale/reset sense counter invalidates it, and the failure
// mode is silent wrong output. The split below removes the barrier instead of trusting it:
//
//   stage 1  ornith_gdn_conv_recur_splitv8_recur_m1_kernel  (nv*8 WGs x 256 thr)
//            Conv1D + L2-norm + gate + Split-V=8 recurrence; writes conv_ring, S, and publishes
//            fo (16 floats/part) and 8 pair-squares/part into sq_scratch.
//   stage 2  ornith_gdn_conv_recur_splitv8_quant_m1_kernel  (nv*8 WGs x 256 thr)
//            Gated head RMSNorm + block-32 quant; consumes ONLY the published sq_scratch,
//            ssm_norm_w and the z gate, and writes q8 / d+bsum.
//
// Ordering: the two kernels are enqueued back to back on the same AQL queue with the barrier
// bit set (Runtime::enqueue_kernel(..., barrier=true)), which is the same dependency mechanism
// every other stage pair in this engine uses. No device-side cross-WG handshake is left, so no
// residency assumption (and no spin) exists at all.
//
// Thread mapping: only tid 0..127 own state (rg = tid >> 4 is 0..7, 16 rows per group). Threads
// 128..255 never form an S/head address: rg would be 8..15 and i_beg 128..240, i.e. past the
// 128-row head (the out-of-bounds Split-V=8 layout this kernel deliberately does not use).
//
// Scratch contract: sq_scratch holds 192 f32 per head
//   [0..127]  pre-norm fo values written by the 8 parts of stage 1
//   [128..191] the 64 pair-squares (global column pair index)
// The host allocation must therefore be nv * 192 * 4 B. Kernels that still index hv * 64 are
// unaffected by the larger allocation. sq_scratch is the only channel between the dispatches.
#include "ops/kernels/vj_ornith_common.hiph"

extern "C" __global__ __launch_bounds__(256, 4)
void ornith_gdn_conv_recur_splitv8_recur_m1_kernel(
    const float* __restrict__ qkv_gab_in,
    const float* __restrict__ conv_w,
    float*       __restrict__ conv_ring,
    const float* __restrict__ dt_bias,
    const float* __restrict__ ssm_a,
    float*       __restrict__ ssm_ring,
    float*       __restrict__ sq_scratch,
    uint32_t nk,
    uint32_t nv,
    uint32_t conv_dim,
    uint32_t stride_in,
    uint32_t alpha_off,
    uint32_t beta_off,
    uint32_t conv_in_slot,
    uint32_t conv_out_slot,
    uint32_t ssm_ring_head,
    uint32_t use_ring,
    float    eps) {
    (void)stride_in;                       // M == 1 only: the token row starts at offset 0
    const uint32_t hv   = blockIdx.x >> 3;
    const uint32_t part = blockIdx.x & 7u;
    const uint32_t tid  = threadIdx.x;
    const uint32_t c    = tid & 15u;   // column inside the part (16 columns)
    const uint32_t rg   = tid >> 4;    // tid<128: row group 0..7; tid>=128: 8..15, never used for S
    const uint32_t col0 = part * 16u;
    const uint32_t hk   = hv % nk;

    __shared__ float sm_q[128];
    __shared__ float sm_k[128];
    __shared__ float sm_red[8 * 16];
    __shared__ float sm_dv[16];
    __shared__ float sm_ab[2];
    __shared__ float sm_fo[16];

    // ---- Phase 0: Conv1D + SiLU + L2-norm (Q/K) + V conv + alpha/beta gate ----
    // conv_in_slot != conv_out_slot always -> race-free with no cross-WG publication.
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
        if (tid < 16u) {
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
            sm_ab[0] = __expf(sp * ssm_a[hv]);                            // g_exp
            sm_ab[1] = 1.0f / (1.0f + __expf(-qkv_gab_in[beta_off + hv])); // beta
        }
    }
    __syncthreads();

    // ---- Phase 1: Split-V=8 recurrence (16 columns x 8 row groups per WG) ----
    const uint64_t head_elems = 128ULL * 128ULL;
    const uint64_t slot_elems = static_cast<uint64_t>(nv) * head_elems;
    float* S_base = ssm_ring + static_cast<uint64_t>(ssm_ring_head) * slot_elems + static_cast<uint64_t>(hv) * head_elems;
    const float g_exp = sm_ab[0];
    const float beta  = sm_ab[1];

    float s_reg[16];
    if (tid < 128u) {
        const uint32_t i_beg = rg * 16u;
        const uint32_t col   = col0 + c;
        // One base register plus a signed 13-bit immediate (-4096 .. +3584 B) per row.
        float* S_thr = S_base + (i_beg + 8u) * 128u + col;
        #pragma unroll
        for (uint32_t r = 0u; r < 16u; ++r) {
            s_reg[r] = S_thr[(static_cast<int32_t>(r) - 8) * 128];
        }
        float p = 0.0f;
        #pragma unroll
        for (uint32_t r = 0u; r < 16u; ++r) {
            p = fmaf(s_reg[r], sm_k[i_beg + r], p);
        }
        sm_red[rg * 16u + c] = p;
    }
    __syncthreads();

    if (tid < 16u) {
        const float pred = sm_red[tid] + sm_red[16u + tid] + sm_red[32u + tid] + sm_red[48u + tid]
                         + sm_red[64u + tid] + sm_red[80u + tid] + sm_red[96u + tid] + sm_red[112u + tid];
        const float vj = sm_dv[tid];
        sm_dv[tid] = beta * (vj - g_exp * pred);
    }
    __syncthreads();

    if (tid < 128u) {
        const uint32_t i_beg = rg * 16u;
        const uint32_t col   = col0 + c;
        const float dv = sm_dv[c];
        float* S_out = (use_ring != 0u)
            ? ssm_ring + static_cast<uint64_t>((ssm_ring_head + 1u) % VJ_GDN_RING_SLOTS) * slot_elems + static_cast<uint64_t>(hv) * head_elems + (i_beg + 8u) * 128u + col
            : S_base + (i_beg + 8u) * 128u + col;

        float o = 0.0f;
        #pragma unroll
        for (uint32_t r = 0u; r < 16u; ++r) {
            const float ki = sm_k[i_beg + r];
            const float qi = sm_q[i_beg + r];
            const float sv = fmaf(ki, dv, g_exp * s_reg[r]);
            S_out[(static_cast<int32_t>(r) - 8) * 128] = sv;
            o = fmaf(sv, qi, o);
        }
        sm_red[rg * 16u + c] = o;
    }
    __syncthreads();

    if (tid < 16u) {
        sm_fo[tid] = sm_red[tid] + sm_red[16u + tid] + sm_red[32u + tid] + sm_red[48u + tid]
                   + sm_red[64u + tid] + sm_red[80u + tid] + sm_red[96u + tid] + sm_red[112u + tid];
    }
    __syncthreads();

    // ---- Publish this part's fo + pair-squares for stage 2 (no cross-WG wait) ----
    // Dispatch boundary: the next packet on this queue waits for this kernel to complete
    // (AQL barrier bit), so plain global stores are the hand-off. No __threadfence and no
    // spin: a partially resident grid is harmless because no WG ever waits for another.
    float* base_scr = sq_scratch + static_cast<uint64_t>(hv) * 192u;
    float* fo_scr   = base_scr;          // [0..127]
    float* sq       = base_scr + 128u;   // [0..63]
    if (tid < 16u) {
        const float fo0 = sm_fo[tid];
        fo_scr[static_cast<uint64_t>(part) * 16u + tid] = fo0;
        if ((tid & 1u) == 0u) {
            const float fo1 = sm_fo[tid + 1u];
            sq[8u * part + (tid >> 1)] = fo0 * fo0 + fo1 * fo1;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Stage 2: gated head RMSNorm + block-32 quant from the published sq_scratch (ordered dispatch).
// Same per-part thread mapping and the same reduction/rounding trees as the Split-V=4 epilogue:
// tid 0..63 replays the 64-lane pair-square tree; tid 0..15 of each part owns 16 columns of the
// 32-column block; the even part records d/bsum. Output bytes are bit-identical because every
// operand is the exact float stage 1 published.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_gdn_conv_recur_splitv8_quant_m1_kernel(
    const float* __restrict__ qkv_gab_in,
    const float* __restrict__ ssm_norm_w,
    int8_t*      __restrict__ out_q8,
    float2*      __restrict__ out_ds,
    const float* __restrict__ sq_scratch,
    uint32_t z_off,
    float    eps) {
    const uint32_t hv    = blockIdx.x >> 3;
    const uint32_t part  = blockIdx.x & 7u;
    const uint32_t tid   = threadIdx.x;
    const uint32_t block = part >> 1u;
    const uint32_t col_b = block * 32u;

    const float* base_scr = sq_scratch + static_cast<uint64_t>(hv) * 192u;
    const float* fo_scr   = base_scr;          // [0..127]
    const float* sq       = base_scr + 128u;   // [0..63]

    if (tid < 64u) {
        const float lane_sq  = sq[tid];
        const float total_sq = vegajet::ornith_dev::wave64_sum(lane_sq);
        if (tid < 16u) {
            const float inv_r = rsqrtf(total_sq * (1.0f / 128.0f) + eps);
            const uint32_t j0 = col_b + tid * 2u;
            const float fo0 = fo_scr[j0];
            const float fo1 = fo_scr[j0 + 1u];
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
            if ((j0 >> 4u) == part) {
                int8_t* oq = out_q8 + static_cast<uint64_t>(hv) * 128u;
                oq[j0]      = static_cast<int8_t>(max(-127, min(127, __float2int_rn(y0 * inv_d))));
                oq[j0 + 1u] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(y1 * inv_d))));
            }
            if ((part & 1u) == 0u && tid == 0u) {
                out_ds[static_cast<uint64_t>(hv) * 4u + block] = make_float2(d, bsum);
            }
        }
    }
}
