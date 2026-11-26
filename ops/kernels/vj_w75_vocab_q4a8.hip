// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "ops/kernels/vj_ornith_common.hiph"

// Qualification-only optional module. Launch (256,1,1), grid (G,M,1).
// cols must be positive and divisible by 32; row_off+local_rows <= INT32_MAX.
// w_qs rows have cols/2 bytes (16-byte aligned stride); w_dm stores half(d)
// in bits 0..15 and half(m) in bits 16..31. dm_stride_u32==0 means cols/32.
// x_ds=(activation scale, original activation block sum), NOT sum of x_q8.
// Output layout and strict-'>' tie priority match scalar Q8: earliest visited
// row within each wave, then lowest wave index. This is NOT lowest global ID.
// Only 32 B scalar LDS for four partial winners; no activation LDS staging.
__device__ __forceinline__ float w75_dot_blk(const uint4 w, const uint32_t d, const uint4* __restrict__ x4,
                                             const float2* __restrict__ xd, uint32_t b, float acc) {
    using namespace vegajet::ornith_dev;
    const uint4 a = x4[2u*b], z = x4[2u*b+1u];
    int32_t dp = sdot4(w.x & 0x0f0f0f0fu, a.x, 0);
    dp = sdot4(w.y & 0x0f0f0f0fu, a.y, dp);
    dp = sdot4(w.z & 0x0f0f0f0fu, a.z, dp);
    dp = sdot4(w.w & 0x0f0f0f0fu, a.w, dp);
    dp = sdot4((w.x >> 4) & 0x0f0f0f0fu, z.x, dp);
    dp = sdot4((w.y >> 4) & 0x0f0f0f0fu, z.y, dp);
    dp = sdot4((w.z >> 4) & 0x0f0f0f0fu, z.z, dp);
    dp = sdot4((w.w >> 4) & 0x0f0f0f0fu, z.w, dp);
    const float2 ds = xd[b];
    return fmaf(fp16_to_f32(uint16_t(d))*ds.x, float(dp), fmaf(fp16_to_f32(uint16_t(d >> 16)), ds.y, acc));
}

extern "C" __global__ __launch_bounds__(256, 5)
void ornith_vocab_argmax_q4a8_w75_kernel(
    const uint8_t* __restrict__ w_qs,
    const uint32_t* __restrict__ w_dm,
    const int8_t* __restrict__ x_q8,
    const float2* __restrict__ x_ds,
    float* __restrict__ blk_val,
    int32_t* __restrict__ blk_id,
    uint32_t local_rows, uint32_t cols, uint32_t row_off,
    uint32_t row_stride_bytes, uint32_t M, uint32_t dm_stride_u32) {
    using namespace vegajet::ornith_dev;
    const uint32_t m = blockIdx.y, wave = threadIdx.x >> 6;
    const uint32_t lane = threadIdx.x & 63u;
    if (m >= M) return;
    const uint32_t nb = cols >> 5;
    const uint32_t dm_st = dm_stride_u32 ? dm_stride_u32 : nb;
    const uint4* x4 = reinterpret_cast<const uint4*>(x_q8 + uint64_t(m)*cols);
    const float2* xd = x_ds + uint64_t(m)*nb;
    // Both 32-lane halves of a wave work: half h takes row 2*k + h. Per row the lane/block order and the fmaf chain are
    // unchanged, so logits are bit-identical to the one-row-per-wave form. nb == 96 issues all three block loads first.
    const uint32_t half = lane >> 5, l32 = lane & 31u;
    float best_v = -1e30f;
    int32_t best_i = 0x7fffffff;
    for (uint32_t r0 = (blockIdx.x*4u + wave)*2u; r0 < local_rows; r0 += gridDim.x*8u) {
        const uint32_t row = r0 + half;
        float acc = 0.0f;
        if (row < local_rows) {
            const uint4* w4 = reinterpret_cast<const uint4*>(w_qs + uint64_t(row)*row_stride_bytes);
            const uint32_t* dm = w_dm + uint64_t(row)*dm_st;
            #pragma nounroll
                for (uint32_t b = l32; b < nb; b += 32u) acc = w75_dot_blk(ldg_u32x4_stream(w4+b), ldg_u32_stream(dm+b), x4, xd, b, acc);
        }
        acc = block32_sum(acc);
        if (l32 == 0u && row < local_rows) {
            const int32_t id = int32_t(row_off + row);
            if (acc > best_v || (acc == best_v && id < best_i)) { best_v = acc; best_i = id; }
        }
    }
    __shared__ float values[8];
    __shared__ int32_t ids[8];
    if (l32 == 0u) { values[wave*2u + half] = best_v; ids[wave*2u + half] = best_i; }
    __syncthreads();
    if (threadIdx.x == 0u) {
        float v = values[0];
        int32_t id = ids[0];
        for (int k = 1; k < 8; ++k) {
            if (values[k] > v || (values[k] == v && ids[k] < id)) { v = values[k]; id = ids[k]; }
        }
        if (id == 0x7fffffff) id = -1;
        blk_val[uint64_t(m)*gridDim.x + blockIdx.x] = v;
        blk_id[uint64_t(m)*gridDim.x + blockIdx.x] = id;
    }
}

extern "C" __global__ __launch_bounds__(256, 5)
void ornith_vocab_logits_q4a8_w75_kernel(
    const uint8_t* __restrict__ w_qs, const uint32_t* __restrict__ w_dm,
    const int8_t* __restrict__ x_q8, const float2* __restrict__ x_ds,
    float* __restrict__ logits, uint32_t local_rows, uint32_t cols,
    uint32_t row_stride_bytes) {
    using namespace vegajet::ornith_dev;
    const uint32_t wave = threadIdx.x >> 6, lane = threadIdx.x & 63u;
    const uint32_t nb = cols >> 5;
    const uint4* x4 = reinterpret_cast<const uint4*>(x_q8);
    for (uint32_t row = blockIdx.x*4u + wave; row < local_rows; row += gridDim.x*4u) {
        const uint4* w4 = reinterpret_cast<const uint4*>(w_qs + uint64_t(row)*row_stride_bytes);
        const uint32_t* dm = w_dm + uint64_t(row)*nb;
        float acc = 0.0f;
        if (lane < 32u) {
            #pragma nounroll
            for (uint32_t b = lane; b < nb; b += 32u) {
                const uint4 w = ldg_u32x4_stream(w4+b);
                const uint32_t d = ldg_u32_stream(dm+b);
                const uint4 a = x4[2u*b], z = x4[2u*b+1u];
                int32_t dp = sdot4(w.x & 0x0f0f0f0fu, a.x, 0);
                dp = sdot4(w.y & 0x0f0f0f0fu, a.y, dp);
                dp = sdot4(w.z & 0x0f0f0f0fu, a.z, dp);
                dp = sdot4(w.w & 0x0f0f0f0fu, a.w, dp);
                dp = sdot4((w.x >> 4) & 0x0f0f0f0fu, z.x, dp);
                dp = sdot4((w.y >> 4) & 0x0f0f0f0fu, z.y, dp);
                dp = sdot4((w.z >> 4) & 0x0f0f0f0fu, z.z, dp);
                dp = sdot4((w.w >> 4) & 0x0f0f0f0fu, z.w, dp);
                const float2 ds = x_ds[b];
                acc = fmaf(fp16_to_f32(uint16_t(d))*ds.x, float(dp),
                           fmaf(fp16_to_f32(uint16_t(d >> 16)), ds.y, acc));
            }
        }
        acc = block32_sum(acc);
        if (lane == 0u) logits[row] = acc;
    }
}

// Batched Q4 LM-head argmax for verify rows (M = 2..16 per y-block): every weight block is read once per row group
// and dotted with up to 16 tokens. Per (row, token) the lane/block order, fmaf chain (w75_dot_blk) and block32_sum
// match ornith_vocab_argmax_q4a8_w75_kernel exactly, and rows map to blocks identically, so the per-block
// candidates are bit-identical to running the M = 1 kernel once per token.
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_vocab_argmax_q4a8_batch16_kernel(
    const uint8_t* __restrict__ w_qs,
    const uint32_t* __restrict__ w_dm,
    const int8_t* __restrict__ x_q8,
    const float2* __restrict__ x_ds,
    float* __restrict__ blk_val,
    int32_t* __restrict__ blk_id,
    uint32_t local_rows, uint32_t cols, uint32_t row_off,
    uint32_t row_stride_bytes, uint32_t M, uint32_t dm_stride_u32) {
    using namespace vegajet::ornith_dev;
    const uint32_t m0 = blockIdx.y * 16u;
    if (m0 >= M) return;
    const uint32_t T = min(16u, M - m0);
    const uint32_t wave = threadIdx.x >> 6, lane = threadIdx.x & 63u;
    const uint32_t half = lane >> 5, l32 = lane & 31u;
    const uint32_t nb = cols >> 5;
    const uint32_t dm_st = dm_stride_u32 ? dm_stride_u32 : nb;
    __shared__ float   sm_v[16 * 8];
    __shared__ int32_t sm_i[16 * 8];
    if (threadIdx.x < 128u) { sm_v[threadIdx.x] = -1e30f; sm_i[threadIdx.x] = 0x7fffffff; }
    __syncthreads();
    const uint32_t cell = wave * 2u + half;
    for (uint32_t r0 = (blockIdx.x*4u + wave)*2u; r0 < local_rows; r0 += gridDim.x*8u) {
        const uint32_t row = r0 + half;
        float acc[16];
        #pragma unroll
        for (uint32_t t = 0u; t < 16u; ++t) acc[t] = 0.0f;
        if (row < local_rows) {
            const uint4* w4 = reinterpret_cast<const uint4*>(w_qs + uint64_t(row)*row_stride_bytes);
            const uint32_t* dm = w_dm + uint64_t(row)*dm_st;
            #pragma nounroll
            for (uint32_t b = l32; b < nb; b += 32u) {
                const uint4 w = ldg_u32x4_stream(w4 + b);
                const uint32_t d = ldg_u32_stream(dm + b);
                #pragma unroll
                for (uint32_t t = 0u; t < 16u; ++t) {
                    if (t < T) {
                        const uint4* x4 = reinterpret_cast<const uint4*>(x_q8 + uint64_t(m0 + t)*cols);
                        acc[t] = w75_dot_blk(w, d, x4, x_ds + uint64_t(m0 + t)*nb, b, acc[t]);
                    }
                }
            }
        }
        #pragma unroll
        for (uint32_t t = 0u; t < 16u; ++t) {
            if (t < T) {
                const float s = block32_sum(acc[t]);
                if (l32 == 0u && row < local_rows) {
                    const int32_t rid = int32_t(row_off + row);
                    const uint32_t c = t * 8u + cell;
                    if (s > sm_v[c] || (s == sm_v[c] && rid < sm_i[c])) { sm_v[c] = s; sm_i[c] = rid; }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < T) {
        const uint32_t t = threadIdx.x;
        float v = sm_v[t * 8u];
        int32_t rid = sm_i[t * 8u];
        for (uint32_t c = 1u; c < 8u; ++c) {
            const float vc = sm_v[t * 8u + c];
            const int32_t ic = sm_i[t * 8u + c];
            if (vc > v || (vc == v && ic < rid)) { v = vc; rid = ic; }
        }
        if (rid == 0x7fffffff) rid = -1;
        blk_val[uint64_t(m0 + t)*gridDim.x + blockIdx.x] = v;
        blk_id[uint64_t(m0 + t)*gridDim.x + blockIdx.x] = rid;
    }
}
