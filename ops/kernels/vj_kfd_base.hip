// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Minimal bare-metal /dev/kfd DMA helper kernels (gfx906, 0 B private spill).
#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__ __launch_bounds__(256, 8)
void kfd_memcpy_u32(uint32_t* __restrict__ dst,
                    const uint32_t* __restrict__ src,
                    uint64_t n_u32) {
    const uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_u32) {
        dst[idx] = src[idx];
    }
}

extern "C" __global__ __launch_bounds__(256, 8)
void kfd_memset_u32(uint32_t* __restrict__ dst,
                    uint32_t val,
                    uint64_t n_u32) {
    const uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n_u32) {
        dst[idx] = val;
    }
}

// ---- HBM2 arena soak (vegajet soak): deterministic pattern fill + streaming verify with on-GPU timing ----
static __device__ __forceinline__ uint32_t kfd_soak_pat(uint32_t i, uint32_t seed) {
    uint32_t x = (i * 2654435761u) ^ seed;
    x ^= x >> 15; x *= 0x2c1b3c6du; x ^= x >> 12;
    return x;
}

extern "C" __global__ __launch_bounds__(256, 4)
void kfd_soak_fill_u32x4(uint4* __restrict__ dst, uint64_t n_vec, uint32_t seed) {
    const uint64_t per_wg = (n_vec + gridDim.x - 1u) / gridDim.x;
    const uint64_t base   = static_cast<uint64_t>(blockIdx.x) * per_wg;
    const uint64_t end    = (base + per_wg < n_vec) ? (base + per_wg) : n_vec;
    #pragma nounroll
    for (uint64_t i = base + threadIdx.x; i < end; i += 256u) {
        const uint32_t j = static_cast<uint32_t>(i) * 4u;
        dst[i] = make_uint4(kfd_soak_pat(j, seed), kfd_soak_pat(j + 1u, seed), kfd_soak_pat(j + 2u, seed), kfd_soak_pat(j + 3u, seed));
    }
}

// ticks[2*wg] / ticks[2*wg+1] = s_memrealtime at workgroup start / after all its loads completed.
extern "C" __global__ __launch_bounds__(256, 4)
void kfd_soak_verify_u32x4(const uint4* __restrict__ src, uint64_t n_vec, uint32_t seed,
                           uint32_t* __restrict__ err_count, uint64_t* __restrict__ ticks) {
    if (threadIdx.x == 0u) ticks[2u * blockIdx.x] = __builtin_amdgcn_s_memrealtime();
    const uint64_t per_wg = (n_vec + gridDim.x - 1u) / gridDim.x;
    const uint64_t base   = static_cast<uint64_t>(blockIdx.x) * per_wg;
    const uint64_t end    = (base + per_wg < n_vec) ? (base + per_wg) : n_vec;
    uint32_t errs = 0u;
    #pragma nounroll
    for (uint64_t i = base + threadIdx.x; i < end; i += 256u) {
        const uint32_t j = static_cast<uint32_t>(i) * 4u;
        const uint4 v = src[i];
        errs += (v.x != kfd_soak_pat(j, seed)) + (v.y != kfd_soak_pat(j + 1u, seed)) +
                (v.z != kfd_soak_pat(j + 2u, seed)) + (v.w != kfd_soak_pat(j + 3u, seed));
    }
    if (errs != 0u) atomicAdd(err_count, errs);
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0u) ticks[2u * blockIdx.x + 1u] = __builtin_amdgcn_s_memrealtime();
}

// Single-instruction on-GPU s_memrealtime() timestamp for exact kernel-level A/B benchmarking.
extern "C" __global__ __launch_bounds__(64, 8)
void kfd_stamp_clock_u64(uint64_t* __restrict__ dst, uint32_t idx) {
    if (blockIdx.x == 0u && threadIdx.x == 0u) {
        __threadfence_system();
        dst[idx] = __builtin_amdgcn_s_memrealtime();
    }
}
