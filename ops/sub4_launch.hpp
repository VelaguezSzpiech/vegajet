// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
#include <cstdint>
#include <cstddef>
namespace vegajet::sub4 {
// Identical ABI for plain and _xor entry points. M=1 for gemv, 1..8 for gemm.
// _xor requires cols%512==0, both planes permuted by block^(row&15),
// and expanded dm metadata [rows,cols/32]. Plain dm is [rows,cols/group].
struct Kernarg {
    uint64_t weights, dm, activations, activation_ds, output;
    uint32_t rows, cols, group, M;
};
static_assert(sizeof(Kernarg)==56 && offsetof(Kernarg, rows)==40);
}
