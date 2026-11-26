// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
#include <cstdint>
#include <cstddef>
namespace vegajet::ornith::kvq_super {
struct Args {
    uint64_t desc;
    uint32_t prev_epoch, out_epoch, pos, sense;
    uint64_t signs;
    uint32_t format, chronological;
};
static_assert(sizeof(Args)==40 && offsetof(Args, signs)==24 && offsetof(Args, format)==32);
// Same chunk boundaries as baseline GQA; enough chunks for head-parallel work at short context.
constexpr uint32_t chunk_length(uint32_t span, bool chronological) {
    return chronological ? span : span<=1920 ? 32 : span<=3840 ? 64 : span<=7680 ? 128 : 256;
}
}
