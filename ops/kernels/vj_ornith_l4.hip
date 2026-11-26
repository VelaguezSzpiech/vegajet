// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "ops/kernels/vj_ornith_common.hiph"
extern "C" __global__ __launch_bounds__(256, 4)
void vj_l4_prep_kernel(
    const float* __restrict__ qkv_fused,
    const float* __restrict__ q_norm_w,
    const float* __restrict__ k_norm_w,
    float*       __restrict__ q_out,
    float*       __restrict__ gate_out,
    float*       __restrict__ k_cache_q8,
    float*       __restrict__ v_cache_q8,
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

        const uint64_t slot = (static_cast<uint64_t>(m) * n_kv_local + kv_h) * 256u + d;
        k_cache_q8[slot] = k_rot;
        v_cache_q8[slot] = vr[d];
    }
}

// One KV head / token: lossless native Q8 -> sealed wire copy.
extern "C" __global__ __launch_bounds__(256)
void vj_l4_wire_kv_kernel(const uint8_t* k, const uint8_t* v, uint8_t* kw, uint8_t* vw,
                         uint32_t heads, uint32_t head, uint32_t token) {
    const uint32_t d=threadIdx.x, b=d/32, j=d%32;
    const uint64_t src=(uint64_t(token)*heads+head)*272;
    const uint64_t dst=uint64_t(token)*384+b*48;
    kw[dst+16+j]=k[src+16+d]; vw[dst+16+j]=v[src+16+d];
    if(j<16) { kw[dst+j]=j<2?k[src+2*b+j]:0; vw[dst+j]=j<2?v[src+2*b+j]:0; }
}
// Each real local head occupies one sealed slot; padded heads have zero Q.
extern "C" __global__ __launch_bounds__(256)
void vj_l4_wire_q_kernel(const float* q, uint8_t* wire, uint32_t begin, uint32_t end) {
    const uint32_t h=blockIdx.x, d=threadIdx.x, b=d/32, j=d%32;
    float x=h<end-begin?q[(begin+h)*256+d]:0;
    float am=vegajet::ornith_dev::block32_max(fabsf(x));
    uint8_t* p=wire+h*384+b*48;
    p[16+j]=static_cast<uint8_t>(static_cast<int8_t>(max(-127,min(127,__float2int_rn(x*(am>0?127/am:0))))));
    if(j>=2 && j<16) p[j]=0;
    if(j==0) reinterpret_cast<uint16_t*>(p)[0]=__half_as_ushort(__float2half(am/127));
}
// Scatter only real heads; output is the normal gate + block32 quantizer.
extern "C" __global__ __launch_bounds__(256)
void vj_l4_epilogue_kernel(const float* attn, const float* gate, int8_t* out_q8, float2* out_ds,
                           uint32_t begin, uint32_t end) {
    const uint32_t h=blockIdx.x, d=threadIdx.x, blk=d/32;
    if(h>=end-begin) return;
    float x=attn[h*256+d]*(1.0f/(1.0f+__expf(-gate[(begin+h)*256+d])));
    float am=vegajet::ornith_dev::block32_max(fabsf(x));
    float sum=vegajet::ornith_dev::block32_sum(x);
    out_q8[(begin+h)*256+d]=static_cast<int8_t>(max(-127,min(127,__float2int_rn(x*(am>0?127/am:0)))));
    if((d&31)==0) out_ds[(begin+h)*8+blk]=make_float2(am/127,sum);
}
