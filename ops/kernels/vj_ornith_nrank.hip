// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "ops/kernels/vj_ornith_common.hiph"
// Explicit local head map also covers unequal owner-selected KV replication.
extern "C" __global__ __launch_bounds__(64)
void ornith_local_argmax_nrank(const float* logits, float* value, int32_t* index, uint32_t count, uint32_t first) {
    if (threadIdx.x != 0) return;
    float best = -INFINITY;
    int32_t selected = -1;
    for (uint32_t i = 0; i < count; ++i) {
        const float x = logits[i];
        if (!isfinite(x)) { *value = x; *index = -1; return; }
        if (x > best) { best = x; selected = static_cast<int32_t>(first + i); }
    }
    *value = best; *index = selected;
}
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_attn_flash_nrank_kernel(
    const float*   __restrict__ q_all,
    const float*   __restrict__ gate_all,
    const uint8_t* __restrict__ k_cache_q8,
    const uint8_t* __restrict__ v_cache_q8,
    int8_t*        __restrict__ out_q8,
    float2*        __restrict__ out_ds,
    uint32_t M,
    uint32_t pos0,
    uint32_t n_q_local,
    uint32_t n_kv_local,
    const uint32_t* __restrict__ q_to_kv) {
    const uint32_t h    = blockIdx.x;
    const uint32_t m    = blockIdx.y;
    const uint32_t d    = threadIdx.x;
    if (h >= n_q_local || m >= M) return;

    const uint32_t kv_h   = q_to_kv[h];
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
            const uint64_t off = (static_cast<uint64_t>(t) * n_kv_local + kv_h) * 272ULL;
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
