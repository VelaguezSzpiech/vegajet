// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
#include "core/quant_repacker.hpp"
#include <string_view>

namespace vegajet::quant {
// File presets describe mixed tensor encodings, not a new block layout.
// Always use each tensor's actual GGML id when unpacking.
const FormatInfo* tensor_unpacker(std::string_view name, DiskQuantType& type);
bool is_quantization_preset(std::string_view name);
// Reject unsupported, truncated, unaligned element counts and overflowing sizes
// before invoking the existing bit-exact decoder. Byte source may be unaligned.
bool unpack_quantized(DiskQuantType type, const void* source, size_t source_bytes,
                      float* destination, size_t destination_elements, size_t elements);
} // namespace vegajet::quant
