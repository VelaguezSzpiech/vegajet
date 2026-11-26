// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Normal-engine Q and KV wire producer + attention epilogue for the sealed gfx906 Q8_0 / Q4_0
// Split-KV pair. Host contract and byte layout: ops/wire_normal_launch.hpp.
//
// This module is deliberately separate from the sealed pair (vj_flash_decode_splitkv.hsaco) and
// from the active attention module (vj_ornith_attn.hip). It only converts the engine's native
// FP32 Q and 272 B active Q8_0 KV rows into the sealed wire layout (and applies the active
// sigmoid-gate + block-32 INT8 epilogue after merge). It never reads or writes the sealed
// scratch/output regions besides the arguments it is given.
//
// Launch contracts (fixed in ops/wire_normal_launch.hpp):
//   vj_wire_q8_q_kernel / vj_wire_q4_q_kernel : grid (rows*16,1,1), block 256
//   vj_wire_q8_kv_kernel / vj_wire_q4_kv_kernel: grid (end-begin,1,1), block 256
//   vj_wire_attn_epilogue_kernel              : grid (rows*16,1,1), block 256
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

namespace {

constexpr uint32_t kHeads = 16u;
constexpr uint32_t kHeadDim = 256u;
constexpr uint32_t kBlocks = 8u;
constexpr uint32_t kValues = 32u;
constexpr uint32_t kNativeRowBytes = 272u;
constexpr uint32_t kQ8BlockBytes = 48u;
constexpr uint32_t kQ4BlockBytes = 32u;

__device__ __forceinline__ float block32_max(float value) {
    value = fmaxf(value, __shfl_xor(value, 16, 32));
    value = fmaxf(value, __shfl_xor(value, 8, 32));
    value = fmaxf(value, __shfl_xor(value, 4, 32));
    value = fmaxf(value, __shfl_xor(value, 2, 32));
    value = fmaxf(value, __shfl_xor(value, 1, 32));
    return value;
}

__device__ __forceinline__ float block32_sum(float value) {
    value += __shfl_xor(value, 16, 32);
    value += __shfl_xor(value, 8, 32);
    value += __shfl_xor(value, 4, 32);
    value += __shfl_xor(value, 2, 32);
    value += __shfl_xor(value, 1, 32);
    return value;
}

// Signed value with the largest magnitude inside one 32-lane block (first lane wins a tie).
__device__ __forceinline__ float block32_signed_max(float value, float magnitude) {
    #pragma unroll
    for (int delta = 16; delta; delta >>= 1) {
        const float other_magnitude = __shfl_xor(magnitude, delta, 32);
        const float other_value = __shfl_xor(value, delta, 32);
        // Lexicographic (magnitude, signed value) max: order-independent, and equal to the host
        // rule "among equal magnitudes the numerically larger signed value wins".
        if (other_magnitude > magnitude ||
            (other_magnitude == magnitude && other_value > value)) {
            magnitude = other_magnitude;
            value = other_value;
        }
    }
    return value;
}

__device__ __forceinline__ void write_header(uint8_t* block, uint16_t scale_bits) {
    uint4 header = make_uint4(static_cast<uint32_t>(scale_bits), 0u, 0u, 0u);
    *reinterpret_cast<uint4*>(block) = header;
}

__device__ __forceinline__ uint16_t half_bits(float value) {
    return __half_as_ushort(__float2half(value));
}

__device__ __forceinline__ float half_value(uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}

__device__ __forceinline__ int round_to_int(float value) {
    return __float2int_rn(value);
}

} // namespace

extern "C" __global__ __launch_bounds__(256)
void vj_wire_q8_q_kernel(const float* __restrict__ q, uint8_t* __restrict__ wire,
                         unsigned int rows, unsigned int heads) {
    const uint32_t index = blockIdx.x;
    if (heads != kHeads || index >= rows * kHeads) return;
    const uint32_t row = index / kHeads;
    const uint32_t head = index % kHeads;
    const uint32_t d = threadIdx.x;
    if (d >= kHeadDim) return;
    const uint32_t block = d >> 5;
    const uint32_t sub = d & 31u;
    const float x = q[(static_cast<size_t>(row) * kHeads + head) * kHeadDim + d];
    const float amax = block32_max(fabsf(x));
    const float scale = amax * (1.0f / 127.0f);
    const float inverse = amax > 0.0f ? (127.0f / amax) : 0.0f;
    const int qv = max(-127, min(127, round_to_int(x * inverse)));
    uint8_t* block_out = wire + (static_cast<size_t>(row) * kHeads + head) * kBlocks * kQ8BlockBytes +
                         block * kQ8BlockBytes;
    block_out[16u + sub] = static_cast<uint8_t>(static_cast<int8_t>(qv));
    if (sub == 0u) write_header(block_out, half_bits(scale));
}

extern "C" __global__ __launch_bounds__(256)
void vj_wire_q4_q_kernel(const float* __restrict__ q, uint8_t* __restrict__ wire,
                         unsigned int rows, unsigned int heads) {
    const uint32_t index = blockIdx.x;
    if (heads != kHeads || index >= rows * kHeads) return;
    const uint32_t row = index / kHeads;
    const uint32_t head = index % kHeads;
    const uint32_t d = threadIdx.x;
    if (d >= kHeadDim) return;
    const uint32_t block = d >> 5;
    const uint32_t sub = d & 31u;
    const float x = q[(static_cast<size_t>(row) * kHeads + head) * kHeadDim + d];
    const float signed_max = block32_signed_max(x, fabsf(x));
    const float scale = signed_max == 0.0f ? 0.0f : (signed_max / -8.0f);
    const uint16_t scale_bits = half_bits(scale);
    const float scale_value = half_value(scale_bits);
    const float inverse = scale_value != 0.0f ? (1.0f / scale_value) : 0.0f;
    const int qv = max(-8, min(7, round_to_int(x * inverse)));
    const uint32_t nibble = static_cast<uint32_t>(qv + 8);
    const uint32_t high = __shfl_xor(nibble, 16, 32);
    uint8_t* block_out = wire + (static_cast<size_t>(row) * kHeads + head) * kBlocks * kQ4BlockBytes +
                         block * kQ4BlockBytes;
    if (sub < 16u) block_out[16u + sub] = static_cast<uint8_t>(nibble | (high << 4));
    if (sub == 0u) write_header(block_out, scale_bits);
}

extern "C" __global__ __launch_bounds__(256)
void vj_wire_q8_kv_kernel(const uint8_t* __restrict__ src_k, const uint8_t* __restrict__ src_v,
                          uint8_t* __restrict__ dst_k, uint8_t* __restrict__ dst_v,
                          unsigned int begin, unsigned int end, unsigned int kv_heads) {
    const uint32_t row = begin + blockIdx.x;
    if (row >= end || kv_heads != 1u) return;
    const uint32_t d = threadIdx.x;
    if (d >= kHeadDim) return;
    const uint32_t block = d >> 5;
    const uint32_t sub = d & 31u;
    const uint8_t* source_k = src_k + static_cast<size_t>(row) * kNativeRowBytes;
    const uint8_t* source_v = src_v + static_cast<size_t>(row) * kNativeRowBytes;
    uint8_t* target_k = dst_k + static_cast<size_t>(row) * kBlocks * kQ8BlockBytes + block * kQ8BlockBytes;
    uint8_t* target_v = dst_v + static_cast<size_t>(row) * kBlocks * kQ8BlockBytes + block * kQ8BlockBytes;
    target_k[16u + sub] = source_k[16u + d];
    target_v[16u + sub] = source_v[16u + d];
    if (sub == 0u) {
        const uint16_t scale_k = static_cast<uint16_t>(
            source_k[2u * block] | (source_k[2u * block + 1u] << 8));
        const uint16_t scale_v = static_cast<uint16_t>(
            source_v[2u * block] | (source_v[2u * block + 1u] << 8));
        write_header(target_k, scale_k);
        write_header(target_v, scale_v);
    }
}

extern "C" __global__ __launch_bounds__(256)
void vj_wire_q4_kv_kernel(const uint8_t* __restrict__ src_k, const uint8_t* __restrict__ src_v,
                          uint8_t* __restrict__ dst_k, uint8_t* __restrict__ dst_v,
                          unsigned int begin, unsigned int end, unsigned int kv_heads) {
    const uint32_t row = begin + blockIdx.x;
    if (row >= end || kv_heads != 1u) return;
    const uint32_t d = threadIdx.x;
    if (d >= kHeadDim) return;
    const uint32_t block = d >> 5;
    const uint32_t sub = d & 31u;
    const uint8_t* source_k = src_k + static_cast<size_t>(row) * kNativeRowBytes;
    const uint8_t* source_v = src_v + static_cast<size_t>(row) * kNativeRowBytes;
    const float scale_k_in = half_value(static_cast<uint16_t>(
        source_k[2u * block] | (source_k[2u * block + 1u] << 8)));
    const float scale_v_in = half_value(static_cast<uint16_t>(
        source_v[2u * block] | (source_v[2u * block + 1u] << 8)));
    const float value_k = scale_k_in *
        static_cast<float>(static_cast<int8_t>(source_k[16u + d]));
    const float value_v = scale_v_in *
        static_cast<float>(static_cast<int8_t>(source_v[16u + d]));
    const float signed_k = block32_signed_max(value_k, fabsf(value_k));
    const float signed_v = block32_signed_max(value_v, fabsf(value_v));
    const float scale_k = signed_k == 0.0f ? 0.0f : (signed_k / -8.0f);
    const float scale_v = signed_v == 0.0f ? 0.0f : (signed_v / -8.0f);
    const uint16_t bits_k = half_bits(scale_k);
    const uint16_t bits_v = half_bits(scale_v);
    const float half_k = half_value(bits_k);
    const float half_v = half_value(bits_v);
    const float inverse_k = half_k != 0.0f ? (1.0f / half_k) : 0.0f;
    const float inverse_v = half_v != 0.0f ? (1.0f / half_v) : 0.0f;
    const uint32_t nibble_k = static_cast<uint32_t>(
        max(-8, min(7, round_to_int(value_k * inverse_k))) + 8);
    const uint32_t nibble_v = static_cast<uint32_t>(
        max(-8, min(7, round_to_int(value_v * inverse_v))) + 8);
    const uint32_t high_k = __shfl_xor(nibble_k, 16, 32);
    const uint32_t high_v = __shfl_xor(nibble_v, 16, 32);
    uint8_t* target_k = dst_k + static_cast<size_t>(row) * kBlocks * kQ4BlockBytes + block * kQ4BlockBytes;
    uint8_t* target_v = dst_v + static_cast<size_t>(row) * kBlocks * kQ4BlockBytes + block * kQ4BlockBytes;
    if (sub < 16u) {
        target_k[16u + sub] = static_cast<uint8_t>(nibble_k | (high_k << 4));
        target_v[16u + sub] = static_cast<uint8_t>(nibble_v | (high_v << 4));
    }
    if (sub == 0u) {
        write_header(target_k, bits_k);
        write_header(target_v, bits_v);
    }
}

// Active-engine epilogue: y = attention * sigmoid(gate), block-32 INT8 quant, float2{scale,sum}.
extern "C" __global__ __launch_bounds__(256)
void vj_wire_attn_epilogue_kernel(const float* __restrict__ attention,
                                  const float* __restrict__ gate,
                                  int8_t* __restrict__ out_q8, float2* __restrict__ out_ds,
                                  unsigned int rows, unsigned int heads) {
    const uint32_t index = blockIdx.x;
    if (heads != kHeads || index >= rows * kHeads) return;
    const uint32_t row = index / kHeads;
    const uint32_t head = index % kHeads;
    const uint32_t d = threadIdx.x;
    if (d >= kHeadDim) return;
    const uint32_t block = d >> 5;
    const uint32_t sub = d & 31u;
    const size_t base = (static_cast<size_t>(row) * kHeads + head) * kHeadDim;
    const size_t head_index = static_cast<size_t>(row) * kHeads + head;
    const float gv = gate[base + d];
    const float y = attention[base + d] * (1.0f / (1.0f + __expf(-gv)));
    const float amax = block32_max(fabsf(y));
    const float sum = block32_sum(y);
    const float scale = amax * (1.0f / 127.0f);
    const float inverse = amax > 0.0f ? (127.0f / amax) : 0.0f;
    out_q8[base + d] = static_cast<int8_t>(max(-127, min(127, round_to_int(y * inverse))));
    if (sub == 0u) out_ds[head_index * kBlocks + block] = make_float2(scale, sum);
}
