// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// VegaJet opt-in W4A4 prefill: block-64 orthonormal FWHT tile for the dense v_dot8_i32_i4 GEMM.
//
// Enabled with --prefill-dot8 --dot8-fwht. The default W4A8 prefill and the plain (unrotated) W4A4 path are
// untouched: this kernel only replaces vj_prefill_q8_to_q4_kernel's activation packing when the matching
// FWHT-rotated VJ_Q4_A8 weight copies were built at load (upload_fused_q4a8_tcc(..., fwht_out)).
//
// Layout contract (identical kernargs to vj_prefill_q8_to_q4_kernel, so the launcher can swap the kernel):
//   in : x_q8  [M, K]  INT8, block-32 symmetric, x_ds float2{d, sum} per 32 (d = amax/127)
//   out: x_q4  [M, K/8] uint32, 8 signed nibbles per word, element 8w+k in nibble k, q in [-7, +7]
//        x_ds4 [M, K/32] float2{xd_4, xd_4 * sum(q4)}  (FP32, unlike the FP16 weight d/m)
//        x_sum4[M, K/32] int32 sum(q4)                (kept for parity with the unrotated packer)
//
// Math: both operands are multiplied by the orthonormal 64-point Walsh-Hadamard H (H^T H = I): a weighted
// column block of 64 is rotated once at load (CPU, quant::fwht64_inplace) and each activation 64-block is
// rotated here, so sum_k (H w)_k (H a)_k = sum_k w_k a_k and no rescale factor is needed. Rotation spreads
// per-channel outliers over the whole 64-block, which is what lowers the INT4 error; it does NOT make the
// 4-bit activation/4-bit weight path lossless, and this kernel is never a substitute for the W4A8 default.
//
// Quantizer: per-32-block symmetric amax after the rotation, exactly like the production VJ_Q4_DOT8
// activation quantizer (vj_quant_act_q4_dot8_kernel): d = amax * clip_alpha / 7, q = rn(v / d) in [-7, +7].
// clip_alpha == 1 (default) means the rotated block maximum is representable; < 1 clips it on purpose.
//
// One 32-lane group owns one 64-block: lane holds element `lane` (low half, block 2b) and `lane + 32`
// (high half, block 2b+1). The first five FWHT stages are 32-lane shuffles on both registers, stage 32 is
// the in-register pair; the operation order and the final *1/8 match quant::fwht64_inplace bit-for-bit
// (verified by tests/test_prefill_dot8_fwht_kld.cpp).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

namespace vegajet {
namespace pfwht {

// Orthonormal 64-point FWHT for one 32-lane group holding (lane, lane+32) of the same 64-block.
// Stage order 1,2,4,8,16 (shuffles) then 32 (register pair), then *1/8: same order/rounding as
// quant::fwht64_inplace, including the "lower index adds, upper index subtracts" sign convention.
__device__ __forceinline__ void fwht64_pair(float& x0, float& x1, uint32_t lane) {
    #pragma unroll
    for (uint32_t s = 1u; s < 32u; s <<= 1u) {
        const float p0 = __shfl_xor(x0, static_cast<int>(s), 32);
        const float p1 = __shfl_xor(x1, static_cast<int>(s), 32);
        const bool up = (lane & s) != 0u;
        x0 = up ? (p0 - x0) : (x0 + p0);
        x1 = up ? (p1 - x1) : (x1 + p1);
    }
    const float t0 = x0 + x1;   // element lane     : partner lane+32 is the upper index
    const float t1 = x0 - x1;   // element lane+32
    x0 = t0 * 0.125f;
    x1 = t1 * 0.125f;
}

__device__ __forceinline__ float block32_max(float v) {
    v = fmaxf(v, __shfl_xor(v, 16, 32));
    v = fmaxf(v, __shfl_xor(v, 8, 32));
    v = fmaxf(v, __shfl_xor(v, 4, 32));
    v = fmaxf(v, __shfl_xor(v, 2, 32));
    v = fmaxf(v, __shfl_xor(v, 1, 32));
    return v;
}

__device__ __forceinline__ int32_t block32_sum_i32(int32_t v) {
    v += __shfl_xor(v, 16, 32);
    v += __shfl_xor(v, 8, 32);
    v += __shfl_xor(v, 4, 32);
    v += __shfl_xor(v, 2, 32);
    v += __shfl_xor(v, 1, 32);
    return v;
}

// Pack 8 consecutive signed nibbles of one 32-lane group into one dword per (lane & 7) == 0 lane.
__device__ __forceinline__ uint32_t pack_s4x8(uint32_t nib, uint32_t lane) {
    uint32_t p = nib << ((lane & 7u) * 4u);
    p |= __shfl_xor(p, 1, 32);
    p |= __shfl_xor(p, 2, 32);
    p |= __shfl_xor(p, 4, 32);
    return p;
}

} // namespace pfwht
} // namespace vegajet

// ============================================================================
// Block-64 orthonormal FWHT + block-32 symmetric INT4 [-7..+7] activation packer (stream, <= 48 VGPR).
// Grid: (ceil((K/64)/8) * 256, M, 1), Block: 256 (8 groups of 32 lanes = 8 64-blocks per WG).
// ============================================================================
extern "C" __global__ __launch_bounds__(256, 5)
void vj_prefill_q8_to_q4_fwht64_kernel(
    const int8_t*  __restrict__ x_q8,
    const float2*  __restrict__ x_ds,
    uint32_t*      __restrict__ x_q4_out,
    float2*        __restrict__ x_ds4_out,
    int32_t*       __restrict__ x_sum4_out,
    uint32_t M,
    uint32_t K,
    float    clip_alpha) {
    const uint32_t m = blockIdx.y;
    if (m >= M) return;
    const uint32_t tid   = threadIdx.x;
    const uint32_t lane  = tid & 31u;
    const uint32_t nb32  = K >> 5;
    const uint32_t nb64  = K >> 6;
    const uint32_t blk64 = blockIdx.x * 8u + (tid >> 5);
    if (blk64 >= nb64) return;

    const uint64_t row_q8 = static_cast<uint64_t>(m) * K;
    const float2 ds0 = x_ds[static_cast<uint64_t>(m) * nb32 + blk64 * 2u];
    const float2 ds1 = x_ds[static_cast<uint64_t>(m) * nb32 + blk64 * 2u + 1u];

    // INT8 -> FP32 on the original basis (each 32-block has its own scale), then the 64-point rotation.
    float x0 = static_cast<float>(x_q8[row_q8 + static_cast<uint64_t>(blk64) * 64u + lane]) * ds0.x;
    float x1 = static_cast<float>(x_q8[row_q8 + static_cast<uint64_t>(blk64) * 64u + 32u + lane]) * ds1.x;
    vegajet::pfwht::fwht64_pair(x0, x1, lane);

    // Per-32-block symmetric INT4 (block 2*blk64 = lanes' x0, block 2*blk64+1 = lanes' x1).
    const float a0 = vegajet::pfwht::block32_max(fabsf(x0));
    const float a1 = vegajet::pfwht::block32_max(fabsf(x1));
    const float d0 = a0 * clip_alpha * (1.0f / 7.0f);
    const float d1 = a1 * clip_alpha * (1.0f / 7.0f);
    const float i0 = (d0 > 0.0f) ? (1.0f / d0) : 0.0f;
    const float i1 = (d1 > 0.0f) ? (1.0f / d1) : 0.0f;
    const int32_t q0 = max(-7, min(7, __float2int_rn(x0 * i0)));
    const int32_t q1 = max(-7, min(7, __float2int_rn(x1 * i1)));
    const int32_t s0 = vegajet::pfwht::block32_sum_i32(q0);
    const int32_t s1 = vegajet::pfwht::block32_sum_i32(q1);

    const uint32_t p0 = vegajet::pfwht::pack_s4x8(static_cast<uint32_t>(q0) & 0xFu, lane);
    const uint32_t p1 = vegajet::pfwht::pack_s4x8(static_cast<uint32_t>(q1) & 0xFu, lane);
    if ((lane & 7u) == 0u) {
        const uint64_t wbase = static_cast<uint64_t>(m) * (K >> 3) + static_cast<uint64_t>(blk64) * 8u;
        x_q4_out[wbase + (lane >> 3)]      = p0;   // block 2*blk64    : nibble k = element 8w+k
        x_q4_out[wbase + 4u + (lane >> 3)] = p1;   // block 2*blk64+1
    }
    if (lane < 2u) {
        const uint64_t b = static_cast<uint64_t>(m) * nb32 + static_cast<uint64_t>(blk64) * 2u + lane;
        const float d  = (lane == 0u) ? d0 : d1;
        const int32_t s = (lane == 0u) ? s0 : s1;
        x_ds4_out[b]  = make_float2(d, d * static_cast<float>(s));
        x_sum4_out[b] = s;
    }
}
