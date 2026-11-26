// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Shared device-fault record + token-id guard for the v1.2 correctness levers C1 (fail-closed barriers)
// and C3 (argmax bounds check). The device side is ops/kernels/vj_ornith_super.hip
// (fault_record()) -- the two layouts must stay in sync.
//
// Slot layout, one 16 B slot per rank in host-visible (userptr-pinned) memory:
//   [0] identity: (kFaultTag << 24) | (kind << 16) | (barrier_id << 8) | stage,  0 = clean
//   [1] epoch: the sense/epoch value the workgroup was waiting for
//   [2] rank: written once by the host at allocation time
//   [3] padding
// The host zeroes word 0/1 before every step and reads the slot back once the queues have drained.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

namespace vegajet {
namespace guard {

constexpr uint32_t kFaultWords     = 4u;
constexpr uint64_t kFaultSlotBytes = 16u;
constexpr uint32_t kFaultTag       = 0x56u;

enum FaultKind : uint32_t {
    kFaultNone    = 0u,
    kFaultBarrier = 1u,   // grid barrier spin budget exhausted
    kFaultMailbox = 2u,   // peer AllReduce mailbox epoch never arrived
};

enum : uint32_t {
    kFaultWIdentity = 0u,
    kFaultWEpoch    = 1u,
    kFaultWRank     = 2u,
};

inline const char* fault_kind_name(uint32_t kind) {
    switch (kind) {
        case kFaultBarrier: return "grid-barrier timeout";
        case kFaultMailbox: return "peer-mailbox timeout";
        default:            return "unknown fault";
    }
}

// A recorded fault carries the tag in the top byte; a clean slot is all zero.
inline bool fault_seen(const uint32_t* slot) { return (slot[kFaultWIdentity] >> 24) == kFaultTag; }
inline uint32_t fault_kind(const uint32_t* slot) { return (slot[kFaultWIdentity] >> 16) & 0xFFu; }
inline uint32_t fault_barrier_id(const uint32_t* slot) { return (slot[kFaultWIdentity] >> 8) & 0xFFu; }
inline uint32_t fault_stage(const uint32_t* slot) { return slot[kFaultWIdentity] & 0xFFu; }
inline uint32_t fault_epoch(const uint32_t* slot) { return slot[kFaultWEpoch]; }
inline uint32_t fault_rank(const uint32_t* slot) { return slot[kFaultWRank]; }

// "rank 1: grid-barrier timeout at barrier 2 stage 4 (awaited epoch 918)"
inline std::string fault_describe(const uint32_t* slot) {
    if (!fault_seen(slot)) {
        if (slot[kFaultWIdentity] != 0u) return "unreadable fault record (no tag)";
        return std::string();
    }
    char buf[160];
    std::snprintf(buf, sizeof(buf), "rank %u: %s at barrier %u stage %u (awaited epoch %u)",
                  fault_rank(slot), fault_kind_name(fault_kind(slot)), fault_barrier_id(slot),
                  fault_stage(slot), fault_epoch(slot));
    return std::string(buf);
}

// Clear a slot for the next step. The rank word is preserved, it is written once at allocation.
inline void fault_clear(uint32_t* slot) {
    slot[kFaultWIdentity] = 0u;
    slot[kFaultWEpoch]    = 0u;
}

} // namespace guard
} // namespace vegajet
