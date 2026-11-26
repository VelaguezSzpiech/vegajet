// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
#include <cstdint>
#include <vector>
#include <stdexcept>
namespace vegajet::batch {
inline uint32_t pack_slots(const std::vector<uint32_t>& slots) {
    if (slots.size()>8) throw std::invalid_argument("batch slots");
    uint32_t packed=0, seen=0;
    for (uint32_t i=0;i<slots.size();++i) {
        if (slots[i]>=8 || (seen&(1u<<slots[i]))) throw std::invalid_argument("duplicate batch slot");
        packed|=slots[i]<<(i*3u); seen|=1u<<slots[i];
    }
    return packed;
}
struct AttentionRows {
    uint64_t qkv, qgate, quant, scales, kv_segment;
    AttentionRows(uint32_t nq,uint32_t nkv,uint32_t max_ctx)
        : qkv((uint64_t(nq)*512u+uint64_t(nkv)*512u)*4u),
          qgate(uint64_t(nq)*256u*4u),quant(uint64_t(nq)*256u),
          scales(uint64_t(nq)*8u*8u),kv_segment(uint64_t(max_ctx/8u)*nkv*272u) {}
};
constexpr bool last_row_logits(bool logits,bool ring,bool independent,uint32_t rows) {
    return logits && !ring && !independent && rows>1u;
}
} // namespace vegajet::batch
