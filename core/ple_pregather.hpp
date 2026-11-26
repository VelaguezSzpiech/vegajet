// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// One-shot host gather of per-layer embeddings (PLE): for a step of M tokens, every layer's embedding row of every
// token is dequantized into ONE contiguous FP32 buffer, so the device receives a single upload per step instead of
// n_layers x M small copies. OpenMP parallel over (layer, token) when compiled with -fopenmp; serial otherwise.
// Row formats: F32, F16, Q8_0 (blocks of 32: FP16 scale + 32 int8). Tokens are validated before anything is written.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vegajet::ple {

enum class RowFmt : uint8_t { F32 = 0, F16 = 1, Q8_0 = 2, Q4 = 3 };
enum class Layout : uint8_t {
    LayerMajor = 0,  // out[((l * M) + m) * dim + d]   (one slab per layer: the layer kernel reads one block)
    TokenMajor = 1,  // out[((m * L) + l) * dim + d]
};

size_t row_bytes(RowFmt f, uint32_t dim);   // 0 if dim is invalid for the format (Q8_0/Q4 need dim % 32 == 0)

struct PleTables {
    uint32_t n_layers = 0, vocab = 0, dim = 0;
    RowFmt fmt = RowFmt::F32;
    std::vector<const uint8_t*> layer;   // n_layers tables, each vocab x row_bytes(fmt, dim)
};

// Returns false (out untouched) on a bad table description or an out-of-range token.
bool pregather(const PleTables& t, const int32_t* tokens, uint32_t m, float* out, Layout lay, std::string* err = nullptr);
// Serial element-by-element reference (tests).
bool pregather_reference(const PleTables& t, const int32_t* tokens, uint32_t m, float* out, Layout lay);
int max_threads();   // OpenMP thread count, 1 without OpenMP

float half_to_float(uint16_t h);

} // namespace vegajet::ple
