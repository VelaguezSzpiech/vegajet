// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once

#include "ops/kernels/vj_ornith_super_desc.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace vegajet::ornith_mega {
constexpr uint32_t kBarrierBytes = 4096u;
// Each dispatch uses fewer than 64 monotonically increasing barrier generations.
// Reserve the whole interval before enqueue, including when switching back to super.
constexpr uint32_t kSenseStride = 64u;
constexpr uint32_t kWorkgroups = 240u;
constexpr uint32_t kSkipFirstAr = 1u;
constexpr uint32_t kFinalAr = 1u;
constexpr uint32_t kFinalNorm = 2u;
constexpr uint32_t kProfile = 4u;
constexpr uint32_t kProfileOffset = 4096u;
constexpr uint32_t kScratchBytes = 20480u;
using ProfileRecords = std::array<std::array<uint64_t, 17>, 97>;
static_assert(kProfileOffset >= kBarrierBytes);
static_assert(kProfileOffset + sizeof(ProfileRecords) <= kScratchBytes);

// Same endpoint order/slots as SUPER_PROFILE=2; record 96 is final AR.
// Reject incomplete/nonmonotonic records instead of unsigned underflow.
inline bool profile_deltas(const std::array<uint64_t, 17>& record,
                           std::array<uint64_t, 16>& deltas, uint32_t required_slots = 0u) {
    if (!record[16]) return false;
    uint64_t prev = record[16];
    bool have_endpoint = false;
    constexpr unsigned order[] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,14};
    for (unsigned slot : order) {
        if ((required_slots & (1u << slot)) && !record[slot]) return false;
        if (record[slot]) {
            if (record[slot] < prev) return false;
            have_endpoint = true;
            prev = record[slot];
        }
    }
    if (!have_endpoint) return false;
    prev = record[16];
    for (unsigned slot : order) if (record[slot]) {
        deltas[slot] += record[slot] - prev;
        prev = record[slot];
    }
    return true;
}

struct GdnArgs {
    uint64_t desc, spec_gdn_base;
    uint32_t il, gi, n_layers, ar_epoch, conv_in, conv_out;
    uint32_t ring_head, use_ring, super_sense, gdn_sense, flags;
};
struct GqaArgs {
    uint64_t desc;
    uint32_t il, ai, ar_epoch, pos, super_sense, flags;
};

static_assert(std::is_standard_layout_v<ornith_super::Mega48Desc>);
static_assert(std::is_trivially_copyable_v<ornith_super::Mega48Desc>);
static_assert(sizeof(ornith_super::Mega48Desc) == 256u);
static_assert(alignof(ornith_super::Mega48Desc) == 8u);
static_assert(offsetof(ornith_super::Mega48Desc, final_ar_rx) == 104u);
static_assert(offsetof(ornith_super::Mega48Desc, final_ar_flag) == 136u);
static_assert(offsetof(ornith_super::Mega48Desc, super_bar) == 200u);
static_assert(offsetof(ornith_super::Mega48Desc, voc_rows) == 224u);
static_assert(offsetof(ornith_super::Mega48Desc, eps) == 248u);
static_assert(sizeof(GdnArgs) == 64u && offsetof(GdnArgs, flags) == 56u);
static_assert(sizeof(GqaArgs) == 32u && offsetof(GqaArgs, flags) == 28u);
} // namespace vegajet::ornith_mega
