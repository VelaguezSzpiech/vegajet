// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/quant_unpackers.hpp"
#include <limits>

namespace vegajet::quant {
bool is_quantization_preset(std::string_view name) {
    return name == "Q5_K_M" || name == "Q4_K_M" || name == "Q4_K_S" || name == "IQ3_XS";
}

const FormatInfo* tensor_unpacker(std::string_view name, DiskQuantType& type) {
    if (is_quantization_preset(name)) return nullptr;
    if (name == "FP8_E4M3") name = "F8_E4M3";
    const auto& registry = DecoderRegistry::instance();
    for (auto candidate : registry.types()) {
        const auto* info = registry.find(candidate);
        if (name == info->name) { type = candidate; return info; }
    }
    return nullptr;
}

bool unpack_quantized(DiskQuantType type, const void* source, size_t source_bytes,
                      float* destination, size_t destination_elements, size_t elements) {
    const auto* info = DecoderRegistry::instance().find(type);
    if (!info || !info->decode || !info->block_elems || !info->block_bytes ||
        elements % info->block_elems || elements > destination_elements ||
        elements > std::numeric_limits<size_t>::max() / sizeof(float)) return false;
    const size_t blocks = elements / info->block_elems;
    if (blocks > std::numeric_limits<size_t>::max() / info->block_bytes ||
        blocks * info->block_bytes > source_bytes) return false;
    if (!elements) return true;
    if (!source || !destination) return false;
    // Decode is not in-place. Reject overlap without adding pointers beyond buffers.
    const auto src = reinterpret_cast<uintptr_t>(source);
    const auto dst = reinterpret_cast<uintptr_t>(destination);
    const size_t bytes = blocks * info->block_bytes;
    if ((dst >= src && dst - src < bytes) ||
        (src > dst && src - dst < elements * sizeof(float))) return false;
    info->decode(static_cast<const uint8_t*>(source), destination, elements);
    return true;
}
} // namespace vegajet::quant
