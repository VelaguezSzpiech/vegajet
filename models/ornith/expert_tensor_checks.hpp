// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
#include "core/gguf_reader.hpp"
#include "core/quant_repacker.hpp"
#include <string>

namespace vegajet::ornith {
// Native packers and kernels use 12 IQ2_S blocks (GU) / 4 IQ3_XXS blocks (Down).
// MTP uses the registered decoder, but must have the same logical expert rows.
inline bool validate_expert_tensor(const GgufTensorInfo* ti, const std::string& name,
                                   uint32_t n_embd, uint32_t n_ff_exp, uint32_t n_expert,
                                   bool down, bool native, std::string* error) {
    auto fail = [&](const std::string& why) {
        if (error) *error = "Invalid expert tensor " + name + ": " + why;
        return false;
    };
    if (!ti) return fail("missing tensor");
    if (n_embd != 3072u || n_ff_exp != 1024u || n_expert != 256u)
        return fail("GPU row layout requires n_embd=3072, n_ff_exp=1024, n_expert=256");
    const uint64_t cols = down ? n_ff_exp : n_embd;
    const uint64_t rows = down ? n_embd : n_ff_exp;
    // Accept the canonical GGUF expert axis, or its equivalent flattened row axis.
    const bool shape = ti->ne[0] == cols && ti->ne[3] == 1u &&
        ((ti->n_dims == 3u && ti->ne[1] == rows && ti->ne[2] == n_expert) ||
         (ti->n_dims == 2u && ti->ne[1] == rows * n_expert && ti->ne[2] == 1u));
    if (!shape) return fail("expected [" + std::to_string(cols) + "," + std::to_string(rows) +
                            ",256] or equivalent flattened rows (n_ff_exp=1024)");
    const auto expected = down ? GgmlType::IQ3_XXS : GgmlType::IQ2_S;
    if (native && ti->type != static_cast<uint32_t>(expected))
        return fail(down ? "native Down requires IQ3_XXS (392-byte rows)" :
                           "native Gate/Up requires IQ2_S (984-byte rows)");
    const auto* fmt = quant::DecoderRegistry::instance().find(static_cast<quant::DiskQuantType>(ti->type));
    if (!fmt || !fmt->decode || !fmt->block_elems || !fmt->block_bytes || cols % fmt->block_elems)
        return fail("unsupported or unaligned expert tensor type " + std::to_string(ti->type));
    const uint64_t row_bytes = (cols / fmt->block_elems) * fmt->block_bytes;
    if (native && row_bytes != (down ? 392u : 984u)) return fail("unexpected native row byte layout");
    if (ti->n_bytes() != row_bytes * rows * n_expert)
        return fail("payload byte size does not match expected expert row layout");
    return true;
}
} // namespace vegajet::ornith
