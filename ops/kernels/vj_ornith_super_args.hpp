#pragma once
#include <cstddef>
#include <cstdint>

namespace vegajet::ornith_super {
// Explicit arguments only; the runtime appends hidden launch fields after these.
struct GdnArgs {
    uint64_t desc, override_ring;
    uint32_t prev_epoch, out_epoch, cin, cout, shead, uring, sense, gdn_sense;
    uint64_t fault;
};
struct GqaArgs {
    uint64_t desc;
    uint32_t prev_epoch, out_epoch, pos0, sense;
    uint64_t fault;
};
struct MoeArgs {
    uint64_t desc;
    uint32_t attn_epoch, out_epoch, sense;
    uint64_t fault;
};
static_assert(sizeof(GdnArgs) == 56 && offsetof(GdnArgs, fault) == 48, "GDN ABI drift");
static_assert(sizeof(GqaArgs) == 32 && offsetof(GqaArgs, fault) == 24, "GQA ABI drift");
static_assert(sizeof(MoeArgs) == 32 && offsetof(MoeArgs, fault) == 24, "MoE ABI drift");
} // namespace vegajet::ornith_super
