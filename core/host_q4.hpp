// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// 4-bit host-RAM packing for embedding-style tables (PLE per-layer embeddings, Engram n-gram memories) that stay
// in host RAM and are gathered per step.
//
// Block of 32 values: an FP16 scale s, then 16 bytes; byte j carries value j in the low nibble and value j + 16 in
// the high nibble (nibble n is the signed digit n - 8, so digits cover [-8, 7]). s = amax / 7 with amax the largest
// magnitude in the block, so every stored value is s * digit and the reconstruction error of one value is at most
// s / 2 = amax / 14 (plus the FP16 rounding of s). Row size is 18 * dim / 32 bytes, dim must be a multiple of 32.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vegajet::hostq4 {

constexpr uint32_t kBlock = 32;
constexpr uint32_t kBlockBytes = 18;

uint32_t row_bytes(uint32_t dim);          // 0 unless dim > 0 and dim % 32 == 0
bool pack_row(const float* x, uint32_t dim, uint8_t* out);
bool unpack_row(const uint8_t* in, uint32_t dim, float* out);
// Maximum |x - unpack(pack(x))| this row can produce: max over blocks of 0.5 * fp16(amax / 7), with the FP16
// rounding of the scale included. 0 for an empty / invalid row.
double max_abs_error_bound(const float* x, uint32_t dim);

// [rows][dim] table round trip; `dim` must be a multiple of 32.
bool pack_table(const float* x, uint32_t rows, uint32_t dim, std::vector<uint8_t>* out);
bool unpack_table(const uint8_t* in, uint32_t rows, uint32_t dim, float* out);

}  // namespace vegajet::hostq4
