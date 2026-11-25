// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "ops/kernels/vj_ornith_common.hiph"

// ============================================================================
// 10. TP=2 Sharded Vocab LM Head VJ_Q8_DP4A (Q6_K -> Q8 high precision!) + Inline Argmax
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 6)
void ornith_vocab_argmax_q8_kernel(
    const int8_t*   __restrict__ w_qs,
    const uint16_t* __restrict__ w_ds,
    const int8_t*   __restrict__ x_q8,
    const float2*   __restrict__ x_ds,
    float*          __restrict__ blk_val,
    int32_t*        __restrict__ blk_id,
    uint32_t local_rows,
    uint32_t cols,
    uint32_t row_off,
    uint32_t row_stride_bytes,
    uint32_t M) {
    const uint32_t m    = blockIdx.y;
    const uint32_t wave = threadIdx.x >> 6;
    const uint32_t lane = threadIdx.x & 63u;
    if (m >= M) return;

    __shared__ uint4 sm_x[192];
    __shared__ float sm_xd[96];

    const uint4*  x4  = reinterpret_cast<const uint4*>(x_q8 + static_cast<uint64_t>(m) * cols);
    const float2* xds = x_ds + static_cast<uint64_t>(m) * (cols >> 5);
    const uint32_t n16 = cols >> 4;
    const bool use_sm = (cols <= 3072u);
    if (use_sm) {
        if (threadIdx.x < n16) sm_x[threadIdx.x] = x4[threadIdx.x];
        if (threadIdx.x < (cols >> 5)) sm_xd[threadIdx.x] = xds[threadIdx.x].x;
        __syncthreads();
    }

    float   best_v = -1e30f;
    int32_t best_i = -1;

    for (uint32_t row = blockIdx.x * 4u + wave; row < local_rows; row += gridDim.x * 4u) {
        const uint4*    w4 = reinterpret_cast<const uint4*>(w_qs + static_cast<uint64_t>(row) * row_stride_bytes);
        const uint16_t* ds = w_ds + static_cast<uint64_t>(row) * (cols >> 5);
        float acc = 0.0f;
        #pragma nounroll
        for (uint32_t i = lane; i < n16; i += 64u) {
            const uint4 wv  = vegajet::ornith_dev::ldg_u32x4_stream(w4 + i);
            const uint4 xv  = use_sm ? sm_x[i] : x4[i];
            const float xdi = use_sm ? sm_xd[i >> 1] : xds[i >> 1].x;
            int32_t dp = vegajet::ornith_dev::sdot4(wv.w, xv.w,
                         vegajet::ornith_dev::sdot4(wv.z, xv.z,
                         vegajet::ornith_dev::sdot4(wv.y, xv.y,
                         vegajet::ornith_dev::sdot4(wv.x, xv.x, 0))));
            acc = fmaf(vegajet::ornith_dev::fp16_to_f32(ds[i >> 1]) * xdi, static_cast<float>(dp), acc);
        }
        acc = vegajet::ornith_dev::wave64_sum(acc);
        if (lane == 0u && acc > best_v) {
            best_v = acc;
            best_i = static_cast<int32_t>(row_off + row);
        }
    }

    __shared__ float   sm_v[4];
    __shared__ int32_t sm_i[4];
    if (lane == 0u) { sm_v[wave] = best_v; sm_i[wave] = best_i; }
    __syncthreads();
    if (threadIdx.x == 0u) {
        float   bv = sm_v[0];
        int32_t bi = sm_i[0];
        for (int k = 1; k < 4; ++k) {
            if (sm_v[k] > bv) { bv = sm_v[k]; bi = sm_i[k]; }
        }
        blk_val[static_cast<uint64_t>(m) * gridDim.x + blockIdx.x] = bv;
        blk_id[static_cast<uint64_t>(m) * gridDim.x + blockIdx.x]  = bi;
    }
}

namespace {
using v4 = unsigned __attribute__((ext_vector_type(4)));
__device__ __forceinline__ v4 resource(const void* p) {
    auto a = reinterpret_cast<uintptr_t>(p);
    return {unsigned(a), unsigned(a >> 32), 0xffffffffu, 0x27000u};
}
struct stripe { v4 w; unsigned h; };
__device__ __forceinline__ void issue(stripe& s, v4 w, v4 d, unsigned i,unsigned wo,unsigned so) {
    asm volatile("buffer_load_dwordx4 %0,%2,%4,0 offen glc slc\n\t"
                 "buffer_load_ushort %1,%3,%5,0 offen glc slc"
                 : "=&v"(s.w), "=&v"(s.h) : "v"(wo+16*i), "v"(so+(i>>1)*2), "s"(w), "s"(d) : "memory");
}
template<unsigned Wait> __device__ __forceinline__ float consume(stripe& s,unsigned i,const uint4* sm_x,const float* sm_xd,float acc){
    asm volatile("s_waitcnt vmcnt(%2)" : "+v"(s.w), "+v"(s.h) : "n"(Wait) : "memory");
    const auto wv = __builtin_bit_cast(uint4,s.w); const auto xv = sm_x[i]; unsigned h=s.h;
    int32_t dp = vegajet::ornith_dev::sdot4(wv.w, xv.w,
         vegajet::ornith_dev::sdot4(wv.z, xv.z,
         vegajet::ornith_dev::sdot4(wv.y, xv.y,
         vegajet::ornith_dev::sdot4(wv.x, xv.x, 0))));
    return fmaf(vegajet::ornith_dev::fp16_to_f32(static_cast<uint16_t>(h)) * sm_xd[i >> 1], static_cast<float>(dp), acc);
}
}
extern "C" __global__ __launch_bounds__(256, 6)
void ornith_vocab_argmax_q8_mubuf_kernel(
    const int8_t*   __restrict__ w_qs,
    const uint16_t* __restrict__ w_ds,
    const int8_t*   __restrict__ x_q8,
    const float2*   __restrict__ x_ds,
    float*          __restrict__ blk_val,
    int32_t*        __restrict__ blk_id,
    uint32_t local_rows,
    uint32_t cols,
    uint32_t row_off,
    uint32_t row_stride_bytes,
    uint32_t M) {
    const uint32_t m    = blockIdx.y;
    const uint32_t wave = threadIdx.x >> 6;
    const uint32_t lane = threadIdx.x & 63u;
    if (M != 1 || m || cols != 3072) return;

    __shared__ uint4 sm_x[192];
    __shared__ float sm_xd[96];

    const uint4*  x4  = reinterpret_cast<const uint4*>(x_q8 + static_cast<uint64_t>(m) * cols);
    const float2* xds = x_ds + static_cast<uint64_t>(m) * (cols >> 5);
    if (threadIdx.x < 192u) sm_x[threadIdx.x] = x4[threadIdx.x];
    if (threadIdx.x < 96u) sm_xd[threadIdx.x] = xds[threadIdx.x].x;
    __syncthreads();

    float   best_v = -1e30f;
    int32_t best_i = -1;

    for (uint32_t row = blockIdx.x * 4u + wave; row < local_rows; row += gridDim.x * 4u) {
        float acc = 0.0f;
        // row-wave is identical across all active lanes in the wave.
        const unsigned group_row=__builtin_amdgcn_readfirstlane(row-wave);
        const auto wr=resource(w_qs+uint64_t(group_row)*row_stride_bytes), dr=resource(w_ds+uint64_t(group_row)*96);
        stripe current, queue[2];
        issue(current,wr,dr,lane,wave*row_stride_bytes,wave*192); issue(queue[0],wr,dr,lane+64,wave*row_stride_bytes,wave*192); issue(queue[1],wr,dr,lane+128,wave*row_stride_bytes,wave*192);
        acc=consume<4>(current,lane,sm_x,sm_xd,acc);
        asm volatile("" : "+v"(acc) :: "memory");
        acc=consume<2>(queue[0],lane+64,sm_x,sm_xd,acc);
        asm volatile("" : "+v"(acc) :: "memory");
        acc=consume<0>(queue[1],lane+128,sm_x,sm_xd,acc);
        acc = vegajet::ornith_dev::wave64_sum(acc);
        if (lane == 0u && acc > best_v) {
            best_v = acc;
            best_i = static_cast<int32_t>(row_off + row);
        }
    }

    __shared__ float   sm_v[4];
    __shared__ int32_t sm_i[4];
    if (lane == 0u) { sm_v[wave] = best_v; sm_i[wave] = best_i; }
    __syncthreads();
    if (threadIdx.x == 0u) {
        float   bv = sm_v[0];
        int32_t bi = sm_i[0];
        for (int k = 1; k < 4; ++k) {
            if (sm_v[k] > bv) { bv = sm_v[k]; bi = sm_i[k]; }
        }
        blk_val[static_cast<uint64_t>(m) * gridDim.x + blockIdx.x] = bv;
        blk_id[static_cast<uint64_t>(m) * gridDim.x + blockIdx.x]  = bi;
    }
}

// Full-vocab FP32 logits dump kernel (for exact KLD evaluation via --dump-logits; M=1).
extern "C" __global__ __launch_bounds__(256, 6)
void ornith_vocab_logits_q8_kernel(
    const int8_t*   __restrict__ w_qs,
    const uint16_t* __restrict__ w_ds,
    const int8_t*   __restrict__ x_q8,
    const float2*   __restrict__ x_ds,
    float*          __restrict__ logits_out,
    uint32_t local_rows,
    uint32_t cols,
    uint32_t row_stride_bytes) {
    const uint32_t wave = threadIdx.x >> 6;
    const uint32_t lane = threadIdx.x & 63u;
    __shared__ uint4 sm_x[192];
    __shared__ float sm_xd[96];
    const uint4*  x4  = reinterpret_cast<const uint4*>(x_q8);
    const float2* xds = x_ds;
    const uint32_t n16 = cols >> 4;
    const bool use_sm = (cols <= 3072u);
    if (use_sm) {
        if (threadIdx.x < n16) sm_x[threadIdx.x] = x4[threadIdx.x];
        if (threadIdx.x < (cols >> 5)) sm_xd[threadIdx.x] = xds[threadIdx.x].x;
        __syncthreads();
    }
    for (uint32_t row = blockIdx.x * 4u + wave; row < local_rows; row += gridDim.x * 4u) {
        const uint4*    w4 = reinterpret_cast<const uint4*>(w_qs + static_cast<uint64_t>(row) * row_stride_bytes);
        const uint16_t* ds = w_ds + static_cast<uint64_t>(row) * (cols >> 5);
        float acc = 0.0f;
        #pragma nounroll
        for (uint32_t i = lane; i < n16; i += 64u) {
            const uint4 wv  = vegajet::ornith_dev::ldg_u32x4_stream(w4 + i);
            const uint4 xv  = use_sm ? sm_x[i] : x4[i];
            const float xdi = use_sm ? sm_xd[i >> 1] : xds[i >> 1].x;
            int32_t dp = vegajet::ornith_dev::sdot4(wv.w, xv.w,
                         vegajet::ornith_dev::sdot4(wv.z, xv.z,
                         vegajet::ornith_dev::sdot4(wv.y, xv.y,
                         vegajet::ornith_dev::sdot4(wv.x, xv.x, 0))));
            acc = fmaf(vegajet::ornith_dev::fp16_to_f32(ds[i >> 1]) * xdi, static_cast<float>(dp), acc);
        }
        acc = vegajet::ornith_dev::wave64_sum(acc);
        if (lane == 0u) logits_out[row] = acc;
    }
}

// Batched single-read LM head argmax for speculative verification (M = 2..12, cols <= 3072):
// stages up to 6 tokens of x_q8 + x_ds in LDS (20.4 KiB -> 3 WGs/CU) and streams each output.weight
// row ONCE across all T tokens (5x less HBM2 at M=5, 6x less at M=12). Bit-identical to kernel 10.
extern "C" __global__ __launch_bounds__(256, 3)
void ornith_vocab_argmax_q8_batch6_kernel(
    const int8_t*   __restrict__ w_qs,
    const uint16_t* __restrict__ w_ds,
    const int8_t*   __restrict__ x_q8,
    const float2*   __restrict__ x_ds,
    float*          __restrict__ blk_val,
    int32_t*        __restrict__ blk_id,
    uint32_t local_rows,
    uint32_t cols,
    uint32_t row_off,
    uint32_t row_stride_bytes,
    uint32_t M) {
    const uint32_t m0   = blockIdx.y * 6u;
    const uint32_t wave = threadIdx.x >> 6;
    const uint32_t lane = threadIdx.x & 63u;
    if (m0 >= M || cols > 3072u) return;
    const uint32_t T   = min(6u, M - m0);
    const uint32_t n16 = cols >> 4;
    const uint32_t nb  = cols >> 5;

    __shared__ uint4   sm_x[6u * 192u];
    __shared__ float   sm_xd[6u * 96u];
    __shared__ float   sm_v[6u * 4u];
    __shared__ int32_t sm_i[6u * 4u];

    for (uint32_t idx = threadIdx.x; idx < T * n16; idx += 256u) {
        const uint32_t t = idx / n16;
        const uint32_t i = idx - t * n16;
        sm_x[t * 192u + i] = reinterpret_cast<const uint4*>(x_q8 + static_cast<uint64_t>(m0 + t) * cols)[i];
    }
    for (uint32_t idx = threadIdx.x; idx < T * nb; idx += 256u) {
        const uint32_t t = idx / nb;
        const uint32_t b = idx - t * nb;
        sm_xd[t * 96u + b] = x_ds[static_cast<uint64_t>(m0 + t) * nb + b].x;
    }
    if (threadIdx.x < 24u) {
        sm_v[threadIdx.x] = -1e30f;
        sm_i[threadIdx.x] = -1;
    }
    __syncthreads();

    for (uint32_t row = blockIdx.x * 4u + wave; row < local_rows; row += gridDim.x * 4u) {
        const uint4*    w4 = reinterpret_cast<const uint4*>(w_qs + static_cast<uint64_t>(row) * row_stride_bytes);
        const uint16_t* ds = w_ds + static_cast<uint64_t>(row) * nb;
        float acc[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        #pragma nounroll
        for (uint32_t i = lane; i < n16; i += 64u) {
            const uint4 wv  = vegajet::ornith_dev::ldg_u32x4_stream(w4 + i);
            const float wdi = vegajet::ornith_dev::fp16_to_f32(ds[i >> 1]);
            const uint32_t bi = i >> 1;
            #pragma unroll
            for (uint32_t t = 0u; t < 6u; ++t) {
                if (t < T) {
                    const uint4 xv = sm_x[t * 192u + i];
                    int32_t dp = vegajet::ornith_dev::sdot4(wv.w, xv.w,
                                 vegajet::ornith_dev::sdot4(wv.z, xv.z,
                                 vegajet::ornith_dev::sdot4(wv.y, xv.y,
                                 vegajet::ornith_dev::sdot4(wv.x, xv.x, 0))));
                    acc[t] = fmaf(wdi * sm_xd[t * 96u + bi], static_cast<float>(dp), acc[t]);
                }
            }
        }
        #pragma unroll
        for (uint32_t t = 0u; t < 6u; ++t) {
            if (t < T) {
                const float s = vegajet::ornith_dev::wave64_sum(acc[t]);
                if (lane == 0u && s > sm_v[t * 4u + wave]) {
                    sm_v[t * 4u + wave] = s;
                    sm_i[t * 4u + wave] = static_cast<int32_t>(row_off + row);
                }
            }
        }
    }

    __syncthreads();
    if (threadIdx.x < T) {
        const uint32_t t = threadIdx.x;
        float   bv = sm_v[t * 4u];
        int32_t bi = sm_i[t * 4u];
        for (uint32_t k = 1u; k < 4u; ++k) {
            if (sm_v[t * 4u + k] > bv) { bv = sm_v[t * 4u + k]; bi = sm_i[t * 4u + k]; }
        }
        blk_val[static_cast<uint64_t>(m0 + t) * gridDim.x + blockIdx.x] = bv;
        blk_id[static_cast<uint64_t>(m0 + t) * gridDim.x + blockIdx.x]  = bi;
    }
}

// 10c. L2-cached M<=12 batched Q8_0 LM-head argmax: reads 413 MB output.weight ONCE at M=12 (vs 2x in batch6!),
//      reading 45 KiB x_q8/x_ds directly from 4 MiB L2 (384 B LDS, <= 48 VGPRs, 5 waves/SIMD!). Bit-identical.
extern "C" __global__ __launch_bounds__(256, 5)
void ornith_vocab_argmax_q8_batch12_l2_kernel(
    const int8_t*   __restrict__ w_qs,
    const uint16_t* __restrict__ w_ds,
    const int8_t*   __restrict__ x_q8,
    const float2*   __restrict__ x_ds,
    float*          __restrict__ blk_val,
    int32_t*        __restrict__ blk_id,
    uint32_t local_rows,
    uint32_t cols,
    uint32_t row_off,
    uint32_t row_stride_bytes,
    uint32_t M) {
    const uint32_t m0   = blockIdx.y * 12u;
    const uint32_t wave = threadIdx.x >> 6;
    const uint32_t lane = threadIdx.x & 63u;
    if (m0 >= M) return;
    const uint32_t T   = min(12u, M - m0);
    const uint32_t n16 = cols >> 4;
    const uint32_t nb  = cols >> 5;

    __shared__ float   sm_v[12u * 4u];
    __shared__ int32_t sm_i[12u * 4u];
    if (threadIdx.x < 48u) {
        sm_v[threadIdx.x] = -1e30f;
        sm_i[threadIdx.x] = -1;
    }
    __syncthreads();

    const uint4*  xq4_base = reinterpret_cast<const uint4*>(x_q8 + static_cast<uint64_t>(m0) * cols);
    const float2* xds_base = x_ds + static_cast<uint64_t>(m0) * nb;

    for (uint32_t row = blockIdx.x * 4u + wave; row < local_rows; row += gridDim.x * 4u) {
        const uint4*    w4 = reinterpret_cast<const uint4*>(w_qs + static_cast<uint64_t>(row) * row_stride_bytes);
        const uint16_t* ds = w_ds + static_cast<uint64_t>(row) * nb;
        float acc[12] = {0,0,0,0,0,0,0,0,0,0,0,0};
        #pragma nounroll
        for (uint32_t i = lane; i < n16; i += 64u) {
            const uint4 wv    = vegajet::ornith_dev::ldg_u32x4_stream(w4 + i);
            const uint32_t bi = i >> 1;
            const float wdi   = vegajet::ornith_dev::fp16_to_f32(ds[bi]);
            #pragma nounroll
            for (uint32_t t = 0u; t < T; ++t) {
                const uint4 xv = xq4_base[static_cast<uint64_t>(t) * n16 + i];
                int32_t dp = vegajet::ornith_dev::sdot4(wv.w, xv.w,
                             vegajet::ornith_dev::sdot4(wv.z, xv.z,
                             vegajet::ornith_dev::sdot4(wv.y, xv.y,
                             vegajet::ornith_dev::sdot4(wv.x, xv.x, 0))));
                acc[t] = fmaf(wdi * xds_base[static_cast<uint64_t>(t) * nb + bi].x, static_cast<float>(dp), acc[t]);
            }
        }
        #pragma nounroll
        for (uint32_t t = 0u; t < T; ++t) {
            const float s = vegajet::ornith_dev::wave64_sum(acc[t]);
            if (lane == 0u && s > sm_v[t * 4u + wave]) {
                sm_v[t * 4u + wave] = s;
                sm_i[t * 4u + wave] = static_cast<int32_t>(row_off + row);
            }
        }
    }

    __syncthreads();
    if (threadIdx.x < T) {
        const uint32_t t = threadIdx.x;
        float   bv = sm_v[t * 4u];
        int32_t bi = sm_i[t * 4u];
        for (uint32_t k = 1u; k < 4u; ++k) {
            if (sm_v[t * 4u + k] > bv) { bv = sm_v[t * 4u + k]; bi = sm_i[t * 4u + k]; }
        }
        blk_val[static_cast<uint64_t>(m0 + t) * gridDim.x + blockIdx.x] = bv;
        blk_id[static_cast<uint64_t>(m0 + t) * gridDim.x + blockIdx.x]  = bi;
    }
}

// 10d. L2-cached M<=16 batched Q8_0 LM-head argmax: reads 413 MB output.weight ONCE at M=16 (<=64 VGPRs, 0 B spill!).
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_vocab_argmax_q8_batch16_l2_kernel(
    const int8_t*   __restrict__ w_qs,
    const uint16_t* __restrict__ w_ds,
    const int8_t*   __restrict__ x_q8,
    const float2*   __restrict__ x_ds,
    float*          __restrict__ blk_val,
    int32_t*        __restrict__ blk_id,
    uint32_t local_rows,
    uint32_t cols,
    uint32_t row_off,
    uint32_t row_stride_bytes,
    uint32_t M) {
    const uint32_t m0   = blockIdx.y * 16u;
    const uint32_t wave = threadIdx.x >> 6;
    const uint32_t lane = threadIdx.x & 63u;
    if (m0 >= M) return;
    const uint32_t T   = min(16u, M - m0);
    const uint32_t n16 = cols >> 4;
    const uint32_t nb  = cols >> 5;

    __shared__ float   sm_v[16u * 4u];
    __shared__ int32_t sm_i[16u * 4u];
    if (threadIdx.x < 64u) {
        sm_v[threadIdx.x] = -1e30f;
        sm_i[threadIdx.x] = -1;
    }
    __syncthreads();

    const uint4*  xq4_base = reinterpret_cast<const uint4*>(x_q8 + static_cast<uint64_t>(m0) * cols);
    const float2* xds_base = x_ds + static_cast<uint64_t>(m0) * nb;

    for (uint32_t row = blockIdx.x * 4u + wave; row < local_rows; row += gridDim.x * 4u) {
        const uint4*    w4 = reinterpret_cast<const uint4*>(w_qs + static_cast<uint64_t>(row) * row_stride_bytes);
        const uint16_t* ds = w_ds + static_cast<uint64_t>(row) * nb;
        float acc[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        #pragma nounroll
        for (uint32_t i = lane; i < n16; i += 64u) {
            const uint4 wv    = vegajet::ornith_dev::ldg_u32x4_stream(w4 + i);
            const uint32_t bi = i >> 1;
            const float wdi   = vegajet::ornith_dev::fp16_to_f32(ds[bi]);
            #pragma nounroll
            for (uint32_t t = 0u; t < T; ++t) {
                const uint4 xv = xq4_base[static_cast<uint64_t>(t) * n16 + i];
                int32_t dp = vegajet::ornith_dev::sdot4(wv.w, xv.w,
                             vegajet::ornith_dev::sdot4(wv.z, xv.z,
                             vegajet::ornith_dev::sdot4(wv.y, xv.y,
                             vegajet::ornith_dev::sdot4(wv.x, xv.x, 0))));
                acc[t] = fmaf(wdi * xds_base[static_cast<uint64_t>(t) * nb + bi].x, static_cast<float>(dp), acc[t]);
            }
        }
        #pragma nounroll
        for (uint32_t t = 0u; t < T; ++t) {
            const float s = vegajet::ornith_dev::wave64_sum(acc[t]);
            if (lane == 0u && s > sm_v[t * 4u + wave]) {
                sm_v[t * 4u + wave] = s;
                sm_i[t * 4u + wave] = static_cast<int32_t>(row_off + row);
            }
        }
    }

    __syncthreads();
    if (threadIdx.x < T) {
        const uint32_t t = threadIdx.x;
        float   bv = sm_v[t * 4u];
        int32_t bi = sm_i[t * 4u];
        for (uint32_t k = 1u; k < 4u; ++k) {
            if (sm_v[t * 4u + k] > bv) { bv = sm_v[t * 4u + k]; bi = sm_i[t * 4u + k]; }
        }
        blk_val[static_cast<uint64_t>(m0 + t) * gridDim.x + blockIdx.x] = bv;
        blk_id[static_cast<uint64_t>(m0 + t) * gridDim.x + blockIdx.x]  = bi;
    }
}

// ============================================================================
// 11. Built-in blk.48 nextn MTP Kernels (0 B spill!):
//     (a) ornith_mtp_eh_norm_cat_q8_kernel: [RMSNorm(emb(tok), enorm), RMSNorm(h, hnorm)] -> q8/ds [M, 2*D]
//     (b) ornith_moe_gu_q4a8_kernel:        top-8 routed VJ_Q4_A8 Gate+Up + SiLU*Mul + b=32 INT8 quant (+ ds)
//     (c) ornith_moe_down_q4a8_kernel:      top-8 routed VJ_Q4_A8 Down + weighted accum + shared down
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 4)
void ornith_mtp_eh_norm_cat_q8_kernel(
    const int8_t*   __restrict__ emb_qs,
    const uint16_t* __restrict__ emb_ds,
    const int32_t*  __restrict__ tokens,
    const float*    __restrict__ h_in,     // [M, D]
    const float*    __restrict__ w_enorm,  // [D]
    const float*    __restrict__ w_hnorm,  // [D]
    int8_t*         __restrict__ cat_q8,   // [M, 2*D]
    float2*         __restrict__ cat_ds,   // [M, (2*D)/32]
    uint32_t M,
    uint32_t D,
    float    eps) {
    const uint32_t m    = blockIdx.x;
    const uint32_t tid  = threadIdx.x;
    const uint32_t wave = tid >> 6;
    const uint32_t lane = tid & 63u;
    const uint32_t sub  = tid & 31u;
    if (m >= M) return;

    __shared__ float sm_red[4];
    const uint64_t tok = static_cast<uint64_t>(tokens[m]);
    const int8_t*   eqs = emb_qs + tok * D;
    const uint16_t* eds = emb_ds + tok * (D >> 5);
    const float*    hv  = h_in + static_cast<uint64_t>(m) * D;

    int8_t* oq = cat_q8 + static_cast<uint64_t>(m) * (2u * D);
    float2* od = cat_ds + static_cast<uint64_t>(m) * ((2u * D) >> 5);

    // Part 0: enorm(embed(tok)) -> [0..D-1]
    float sse = 0.0f;
    #pragma nounroll
    for (uint32_t c = tid; c < D; c += 256u) {
        const float e = static_cast<float>(eqs[c]) * vegajet::ornith_dev::fp16_to_f32(eds[c >> 5]);
        sse = fmaf(e, e, sse);
    }
    sse = vegajet::ornith_dev::wave64_sum(sse);
    if (lane == 0u) sm_red[wave] = sse;
    __syncthreads();
    const float inv_e = rsqrtf((sm_red[0] + sm_red[1] + sm_red[2] + sm_red[3]) / static_cast<float>(D) + eps);

    #pragma nounroll
    for (uint32_t blk = tid >> 5; blk < (D >> 5); blk += 8u) {
        const uint32_t c = (blk << 5) + sub;
        const float e = static_cast<float>(eqs[c]) * vegajet::ornith_dev::fp16_to_f32(eds[blk]);
        const float y = e * inv_e * w_enorm[c];
        const float amax  = vegajet::ornith_dev::block32_max(fabsf(y));
        const float bsum  = vegajet::ornith_dev::block32_sum(y);
        const float d     = amax * (1.0f / 127.0f);
        const float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        oq[c] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(y * inv_d))));
        if (sub == 0u) od[blk] = make_float2(d, bsum);
    }
    __syncthreads();

    // Part 1: hnorm(h_in) -> [D..2*D-1]
    float ssh = 0.0f;
    #pragma nounroll
    for (uint32_t c = tid; c < D; c += 256u) {
        const float v = hv[c];
        ssh = fmaf(v, v, ssh);
    }
    ssh = vegajet::ornith_dev::wave64_sum(ssh);
    if (lane == 0u) sm_red[wave] = ssh;
    __syncthreads();
    const float inv_h = rsqrtf((sm_red[0] + sm_red[1] + sm_red[2] + sm_red[3]) / static_cast<float>(D) + eps);

    const uint32_t nbD = D >> 5;
    #pragma nounroll
    for (uint32_t blk = tid >> 5; blk < nbD; blk += 8u) {
        const uint32_t c = (blk << 5) + sub;
        const float y = hv[c] * inv_h * w_hnorm[c];
        const float amax  = vegajet::ornith_dev::block32_max(fabsf(y));
        const float bsum  = vegajet::ornith_dev::block32_sum(y);
        const float d     = amax * (1.0f / 127.0f);
        const float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        oq[D + c] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(y * inv_d))));
        if (sub == 0u) od[nbD + blk] = make_float2(d, bsum);
    }
}

