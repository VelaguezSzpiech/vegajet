// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Host side of the v1.2 C1 device-fault record: the exact word packing the super-kernels write
// (ops/kernels/vj_ornith_super.hip, fault_record()) must decode back to rank / kind / barrier id /
// stage / epoch, and re-arming a slot must keep the rank while clearing the fault.
#include "core/vj_device_guard.hpp"
#include "tests/test_common.hpp"

#include <cstring>
#include <string>

using namespace vegajet::guard;

namespace {

// What the device writes: rec[1] = awaited epoch, rec[0] = identity word. The host stamps rank at init.
void device_record(uint32_t* slot, uint32_t kind, uint32_t bar_id, uint32_t stage, uint32_t epoch) {
    slot[kFaultWEpoch]    = epoch;
    slot[kFaultWIdentity] = (kFaultTag << 24) | (kind << 16) | (bar_id << 8) | stage;
}

} // namespace

int main() {
    uint32_t slot[kFaultWords];
    std::memset(slot, 0, sizeof(slot));
    slot[kFaultWRank] = 1u;

    // A freshly armed slot is clean and describes nothing.
    VJ_CHECK(!fault_seen(slot));
    VJ_CHECK(fault_describe(slot).empty());

    // Barrier timeout: barrier 2, stage 4, epoch 918.
    device_record(slot, kFaultBarrier, 2u, 4u, 918u);
    VJ_CHECK(fault_seen(slot));
    VJ_CHECK(fault_kind(slot) == kFaultBarrier);
    VJ_CHECK(fault_barrier_id(slot) == 2u);
    VJ_CHECK(fault_stage(slot) == 4u);
    VJ_CHECK(fault_epoch(slot) == 918u);
    VJ_CHECK(fault_rank(slot) == 1u);
    {
        const std::string why = fault_describe(slot);
        VJ_CHECK(why.find("rank 1") != std::string::npos);
        VJ_CHECK(why.find("grid-barrier timeout") != std::string::npos);
        VJ_CHECK(why.find("barrier 2") != std::string::npos);
        VJ_CHECK(why.find("stage 4") != std::string::npos);
        VJ_CHECK(why.find("epoch 918") != std::string::npos);
    }

    // Re-arming keeps the rank word (written once at allocation) and clears the fault.
    fault_clear(slot);
    VJ_CHECK(!fault_seen(slot));
    VJ_CHECK(fault_epoch(slot) == 0u);
    VJ_CHECK(fault_rank(slot) == 1u);
    VJ_CHECK(fault_describe(slot).empty());

    // Peer-mailbox timeout on rank 0, full-width epoch and stage.
    std::memset(slot, 0, sizeof(slot));
    device_record(slot, kFaultMailbox, 0u, 0u, 0xFFFFFFFEu);
    VJ_CHECK(fault_kind(slot) == kFaultMailbox);
    VJ_CHECK(fault_epoch(slot) == 0xFFFFFFFEu);
    VJ_CHECK(fault_describe(slot).find("peer-mailbox timeout") != std::string::npos);

    // A word without the tag byte is not mistaken for a clean slot: the host must say so, not print a token.
    slot[kFaultWIdentity] = 0x12345678u;
    VJ_CHECK(!fault_seen(slot));
    VJ_CHECK(!fault_describe(slot).empty());

    // Two slips stay independent (one per rank).
    uint32_t slots[2 * kFaultWords];
    std::memset(slots, 0, sizeof(slots));
    slots[kFaultWRank]                   = 0u;
    slots[kFaultWords + kFaultWRank]     = 1u;
    device_record(slots + kFaultWords, kFaultBarrier, 3u, 3u, 77u);
    VJ_CHECK(!fault_seen(slots));
    VJ_CHECK(fault_seen(slots + kFaultWords));
    VJ_CHECK(fault_describe(slots + kFaultWords).find("rank 1") != std::string::npos);

    std::printf("test_device_guard: OK\n");
    return 0;
}
