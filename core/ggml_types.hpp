// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// GGML tensor type ids and block geometry (ids/sizes follow the public GGUF/ggml format, MIT).
#pragma once

#include <cstdint>

namespace vegajet {

enum class GgmlType : uint32_t {
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
    Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15,
    IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18, IQ1_S = 19, IQ4_NL = 20, IQ3_S = 21,
    IQ2_S = 22, IQ4_XS = 23, I8 = 24, I16 = 25, I32 = 26, I64 = 27, F64 = 28,
    IQ1_M = 29, BF16 = 30, TQ1_0 = 34, TQ2_0 = 35, MXFP4 = 39, NVFP4 = 40,
    Q1_0 = 41, Q2_0 = 42, F8_E4M3 = 43,
    COUNT = 44,
};

struct GgmlTypeTraits {
    const char* name;   // nullptr for removed/unknown ids
    uint32_t block_elems;
    uint32_t block_bytes;
};

// Returns traits for any id; unknown ids return {nullptr, 0, 0}.
inline GgmlTypeTraits ggml_type_traits(uint32_t t) {
    switch (t) {
        case 0:  return {"F32", 1, 4};
        case 1:  return {"F16", 1, 2};
        case 2:  return {"Q4_0", 32, 18};
        case 3:  return {"Q4_1", 32, 20};
        case 6:  return {"Q5_0", 32, 22};
        case 7:  return {"Q5_1", 32, 24};
        case 8:  return {"Q8_0", 32, 34};
        case 9:  return {"Q8_1", 32, 36};
        case 10: return {"Q2_K", 256, 84};
        case 11: return {"Q3_K", 256, 110};
        case 12: return {"Q4_K", 256, 144};
        case 13: return {"Q5_K", 256, 176};
        case 14: return {"Q6_K", 256, 210};
        case 15: return {"Q8_K", 256, 292};
        case 16: return {"IQ2_XXS", 256, 66};
        case 17: return {"IQ2_XS", 256, 74};
        case 18: return {"IQ3_XXS", 256, 98};
        case 19: return {"IQ1_S", 256, 50};
        case 20: return {"IQ4_NL", 32, 18};
        case 21: return {"IQ3_S", 256, 110};
        case 22: return {"IQ2_S", 256, 82};
        case 23: return {"IQ4_XS", 256, 136};
        case 24: return {"I8", 1, 1};
        case 25: return {"I16", 1, 2};
        case 26: return {"I32", 1, 4};
        case 27: return {"I64", 1, 8};
        case 28: return {"F64", 1, 8};
        case 29: return {"IQ1_M", 256, 56};
        case 30: return {"BF16", 1, 2};
        case 34: return {"TQ1_0", 256, 54};
        case 35: return {"TQ2_0", 256, 66};
        case 39: return {"MXFP4", 32, 17};
        case 40: return {"NVFP4", 64, 36};
        case 41: return {"Q1_0", 128, 18};
        case 42: return {"Q2_0", 64, 18};
        case 43: return {"F8_E4M3", 1, 1};
        default: return {nullptr, 0, 0};
    }
}

inline const char* ggml_type_name(uint32_t t) {
    const char* n = ggml_type_traits(t).name;
    return n ? n : "UNKNOWN";
}

// Bytes for `n_elems` elements of type t (0 if unknown or not block-aligned).
inline uint64_t ggml_row_bytes(uint32_t t, uint64_t n_elems) {
    const GgmlTypeTraits tr = ggml_type_traits(t);
    if (tr.block_elems == 0 || n_elems % tr.block_elems != 0) return 0;
    return n_elems / tr.block_elems * tr.block_bytes;
}

} // namespace vegajet
