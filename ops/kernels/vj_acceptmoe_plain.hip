// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "ops/kernels/vj_ornith_common.hiph"

// Historical pre-F implementation, isolated to preserve its arithmetic and code generation.
// ============================================================================
// 8z. AcceptMoE (VEGAJET_SPEC_ACCEPTMOE=1, opt-in, LOSSY): speculative-verify (M >= 2) top-8 routing that
//     shrinks the per-batch expert union. Token 0 (the committed context token) is routed exactly as
//     ornith_moe_router_top8_kernel. Draft token m >= 1 keeps its own top-`keep` experts, then fills the
//     remaining 8 - keep slots ranking p_e * (e in union ? 1 : penalty). Weights are the true softmax
//     probabilities of the chosen experts renormalized like the baseline. Draft-position outputs change, so
//     greedy acceptance is no longer guaranteed identical to raw decode: measure KL / parity before use.
//     Grid = 64 threads (one wave), tokens processed in order.
// ============================================================================
extern "C" __global__ __launch_bounds__(64, 1)
void vj_moe_top8_acceptmoe_plain_kernel(
    const float* __restrict__ logits_in,
    int32_t*     __restrict__ top_ids,
    float*       __restrict__ top_w,
    uint32_t M,
    uint32_t keep,
    float    penalty) {
    using namespace vegajet::ornith_dev;
    const uint32_t l = threadIdx.x;
    __shared__ uint32_t sm_union[8];
    if (l < 8u) sm_union[l] = 0u;
    __syncthreads();
    #pragma nounroll
    for (uint32_t m = 0u; m < M; ++m) {
        const float* lg = logits_in + static_cast<uint64_t>(m) * 256u;
        float x0 = lg[l], x1 = lg[l + 64u], x2 = lg[l + 128u], x3 = lg[l + 192u];
        float mx = fmaxf(fmaxf(x0, x1), fmaxf(x2, x3));
        mx = fmaxf(mx, xor_shfl_f32_fast<32>(mx));
        mx = fmaxf(mx, xor_shfl_f32_fast<16>(mx));
        mx = fmaxf(mx, xor_shfl_f32_fast<8>(mx));
        mx = fmaxf(mx, xor_shfl_f32_fast<4>(mx));
        mx = fmaxf(mx, xor_shfl_f32_fast<2>(mx));
        mx = fmaxf(mx, xor_shfl_f32_fast<1>(mx));
        float p[4] = {__expf(x0 - mx), __expf(x1 - mx), __expf(x2 - mx), __expf(x3 - mx)};
        const float s = ((wave64_sum(p[0]) + wave64_sum(p[1])) + wave64_sum(p[2])) + wave64_sum(p[3]);
        const float inv_s = 1.0f / fmaxf(s, 1e-30f);
        uint64_t kp[4], ka[4];
        const uint32_t keep_m = (m == 0u) ? 8u : min(keep, 8u);
        #pragma unroll
        for (uint32_t c = 0u; c < 4u; ++c) {
            p[c] *= inv_s;
            const uint32_t e = l + 64u * c;
            const bool in_u = ((sm_union[e >> 5] >> (e & 31u)) & 1u) != 0u;
            const float pa = in_u ? p[c] : p[c] * penalty;
            kp[c] = (static_cast<uint64_t>(__float_as_uint(p[c])) << 32) | static_cast<uint64_t>(255u - e);
            ka[c] = (static_cast<uint64_t>(__float_as_uint(pa)) << 32) | static_cast<uint64_t>(255u - e);
        }
        float top_sum = 0.0f;
        #pragma nounroll
        for (uint32_t r = 0u; r < 8u; ++r) {
            const bool own = (r < keep_m);
            uint64_t a0 = own ? kp[0] : ka[0], a1 = own ? kp[1] : ka[1], a2 = own ? kp[2] : ka[2], a3 = own ? kp[3] : ka[3];
            uint64_t bk = (a0 > a1) ? a0 : a1;
            const uint64_t b2 = (a2 > a3) ? a2 : a3;
            if (b2 > bk) bk = b2;
            bk = step_max_u64<32>(bk);
            bk = step_max_u64<16>(bk);
            bk = step_max_u64<8>(bk);
            bk = step_max_u64<4>(bk);
            bk = step_max_u64<2>(bk);
            bk = step_max_u64<1>(bk);
            const uint32_t e_sel = 255u - static_cast<uint32_t>(bk & 255u);
            // true probability of the chosen expert lives in lane e_sel & 63, component e_sel >> 6
            const uint32_t c_sel = e_sel >> 6;
            const float pc = (c_sel == 0u) ? p[0] : (c_sel == 1u) ? p[1] : (c_sel == 2u) ? p[2] : p[3];
            const float v = __shfl(pc, static_cast<int>(e_sel & 63u), 64);
            #pragma unroll
            for (uint32_t c = 0u; c < 4u; ++c) {
                if (l + 64u * c == e_sel) { kp[c] = 0ULL; ka[c] = 0ULL; }
            }
            if (l == 0u) {
                top_ids[static_cast<uint64_t>(m) * 8u + r] = static_cast<int32_t>(e_sel);
                top_w[static_cast<uint64_t>(m) * 8u + r]   = v;
                top_sum += v;
                sm_union[e_sel >> 5] |= (1u << (e_sel & 31u));
            }
        }
        if (l == 0u) {
            const float inv_w = 1.0f / fmaxf(top_sum, 1e-30f);
            #pragma unroll
            for (uint32_t r = 0u; r < 8u; ++r) top_w[static_cast<uint64_t>(m) * 8u + r] *= inv_w;
        }
        __syncthreads();
    }
}
