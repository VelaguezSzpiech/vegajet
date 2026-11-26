// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Launch interfaces for the per-layer hot-expert VJ_Q4_DOT8 cache on gfx906 (host side, no device types).
//
// These builders turn the host LRU decision (core/moe_lru_cache.hpp) into the exact descriptors the sealed
// production kernels need, without duplicating either the cache policy or the kernel bodies:
//   * dispatch  : vj_moe_gu_dot8_lru_kernel, one launch per card. The kernel resolves every routed expert
//                 through the per-layer exp_to_slot table and handles a hit (DOT8 cache) and a miss (base
//                 IQ2_S lazy fallback) inside the same dispatch, so a miss never needs a second launch.
//                 Grid = (top_k 8 * local_ff/32, M), block 256; valid for the production verify range M=2..16.
//   * refill    : vj_repack_iq2s_to_dot8_gu_kernel, one launch per card for an admitted (layer, expert, slot).
//                 Grid = (local_ff), block 128; it repacks the expert-major base IQ2_S plane into the slot's
//                 DOT8 rows. The cache is replicated on every card, so the same expert/slot is launched on each.
//   * eviction  : no device kernel exists; the engine evicts by writing -1 into that layer's host
//                 expert->slot table and uploading the 1 KiB table to every card (SlotTableUpload). The
//                 upload must happen only after all refill launches of the step completed; installing the
//                 admitted expert's slot index in the same upload keeps the table self-consistent.
//
// Every builder validates the addresses, shapes and strides against the real dot8_gu_iq2s_layout() so a
// launch cannot be built against a cache plane the kernel would misread. Nothing here opens /dev/kfd or
// talks to a GPU; the caller enqueues the returned descriptors.
#pragma once

#include "core/moe_lru_cache.hpp"
#include "core/moe_slot_budget.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vegajet::moeslot {

inline constexpr uint32_t kMaxCards = 2;
inline constexpr uint32_t kRouterTopK = 8;       // routed experts per token (top_ids[m * 8 + k])
inline constexpr uint32_t kLruMinTokens = 2;     // production gating: M >= 2 uses the LRU verify kernel
inline constexpr uint32_t kLruMaxTokens = 16;    // production gating: M <= 16
inline constexpr uint32_t kLruBlockThreads = 256;
inline constexpr uint32_t kRepackBlockThreads = 128;

// Exact .kernarg_segment_size of the sealed kernels in the gfx906 code object (verified offline against the
// compiled amdhsa metadata; the C++ structs below carry the 4-byte tail padding the ABI does not count).
inline constexpr uint32_t kDispatchKernargBytes = 88;
inline constexpr uint32_t kRefillKernargBytes = 44;

// Production symbol names (ops/kernels/vj_moe_dot8.hip; both are listed in ops/kernels/hsaco_gates.txt).
constexpr const char* kDispatchKernel = "vj_moe_gu_dot8_lru_kernel";
constexpr const char* kRefillKernel = "vj_repack_iq2s_to_dot8_gu_kernel";

// ABI image of vj_moe_gu_dot8_lru_kernel. Field order and widths follow the kernel parameter list:
// (wgu_iq2s, cache_w, cache_sc, exp_to_slot, x_q8, x_ds, top_ids, mid_q8, mid_d, M, local_ff, w_stride,
// s_stride). Nine pointers, four 32-bit scalars -> 88 bytes, no tail padding.
struct DispatchKernarg {
    uint64_t wgu_iq2s = 0;    // base expert-major merged IQ2_S Gate+Up plane (miss path)
    uint64_t cache_w = 0;     // this layer's DOT8 weight plane on the card
    uint64_t cache_sc = 0;    // scale region (embedded: cache_w + s_offset_bytes, as the runner passes)
    uint64_t exp_to_slot = 0; // [n_experts] int32 slot table on the card
    uint64_t x_q8 = 0;
    uint64_t x_ds = 0;
    uint64_t top_ids = 0;
    uint64_t mid_q8 = 0;
    uint64_t mid_d = 0;
    uint32_t tokens = 0;   // M
    uint32_t local_ff = 0; // 512 per rank for Ornith-122B
    uint32_t w_stride = 0;
    uint32_t s_stride = 0;
};
static_assert(sizeof(DispatchKernarg) == kDispatchKernargBytes, "dispatch kernarg must be 88 bytes");
static_assert(offsetof(DispatchKernarg, wgu_iq2s) == 0 && offsetof(DispatchKernarg, exp_to_slot) == 24 &&
                  offsetof(DispatchKernarg, mid_d) == 64 && offsetof(DispatchKernarg, tokens) == 72 &&
                  offsetof(DispatchKernarg, s_stride) == 84,
              "dispatch kernarg offsets must match the sealed kernel parameter ABI");

// ABI image of vj_repack_iq2s_to_dot8_gu_kernel: (wgu_iq2s, cache_w, cache_sc, exp_id, slot_id, local_ff,
// w_stride, s_stride) -> 44 ABI bytes; C++ adds 4 bytes of tail padding, which the runner's 512-byte
// kernarg slot absorbs (implicit args start at align_up(sizeof, 8)).
struct RefillKernarg {
    uint64_t wgu_iq2s = 0;
    uint64_t cache_w = 0;
    uint64_t cache_sc = 0;
    uint32_t expert = 0;
    uint32_t slot = 0;
    uint32_t local_ff = 0;
    uint32_t w_stride = 0;
    uint32_t s_stride = 0;
    uint32_t _tail_pad = 0;
};
static_assert(sizeof(RefillKernarg) == 48, "refill struct is 44 ABI bytes + 4 bytes tail padding");
static_assert(offsetof(RefillKernarg, expert) == 24 && offsetof(RefillKernarg, s_stride) == 40,
              "refill kernarg offsets must match the sealed kernel parameter ABI");

struct LaunchDesc {
    bool ok = false;
    std::string why;
    const char* symbol = nullptr;
    uint32_t grid_x = 0, grid_y = 0, grid_z = 1;
    uint32_t block_x = 0, block_y = 1, block_z = 1;
    uint32_t kernarg_bytes = 0; // ABI bytes to hand to Runtime::enqueue_kernel
    bool barrier = true;
};

// Per-layer addresses on one card. `iq2s_plane` is the expert-major base IQ2_S merged Gate+Up plane the
// refill reads and the dispatch falls back to on a miss; `w_plane`/`s_plane`/`e2s_table` are this layer's
// cache. `s_plane` is the scale region address exactly as the runner passes it (for the current embedded
// layout: w_plane + s_offset_bytes).
struct CardCacheAddr {
    uint64_t iq2s_plane = 0;
    uint64_t w_plane = 0;
    uint64_t s_plane = 0;
    uint64_t e2s_table = 0;
};

struct SlotCacheView {
    uint32_t n_layers = 0;
    uint32_t n_experts = 0;
    uint32_t local_ff = 0;
    uint32_t hidden = 0;
    uint32_t slots_per_layer = 0;
    uint32_t cards = kMaxCards;
    std::vector<std::array<CardCacheAddr, kMaxCards>> layer;
};

// Shared per-card decode buffers the LRU kernel reads/writes (see the runner's per-card workspace).
struct DispatchWorkspace {
    uint64_t x_q8 = 0;
    uint64_t x_ds = 0;
    uint64_t top_ids = 0;
    uint64_t mid_q8 = 0;
    uint64_t mid_d = 0;
};

bool validate_slot_cache_view(const SlotCacheView& view, std::string* why);

struct DispatchPlan {
    bool ok = false;
    std::string why;
    LaunchDesc launch;
    std::array<DispatchKernarg, kMaxCards> args{};
    uint32_t cards = 0;
};

struct RefillPlan {
    bool ok = false;
    std::string why;
    LaunchDesc launch;
    std::array<RefillKernarg, kMaxCards> args{};
    uint32_t cards = 0;
};

// One launch per card; every card addresses the same expert and slot (replicated cache).
DispatchPlan build_dispatch_plan(const SlotCacheView& view, uint32_t layer, uint32_t tokens,
                                 const std::array<DispatchWorkspace, kMaxCards>& ws);
RefillPlan build_refill_plan(const SlotCacheView& view, uint32_t layer, uint32_t expert, uint32_t slot);

// Host expert->slot table patching (evict: new_slot = -1, admit: expert -> slot).
struct SlotTablePatch {
    uint32_t layer = 0;
    uint32_t expert = 0;
    int32_t new_slot = -1;
    uint64_t table_offset = 0; // layer * n_experts * 4
};

struct SlotTableUpload {
    uint32_t layer = 0;
    uint64_t table_offset = 0; // byte offset inside the layer's [n_experts] table
    uint64_t bytes = 0;        // full layer table (n_experts * 4); one upload per touched layer per step
    bool after_refills = true; // ORDER: upload only after every refill launch of the step has completed
};

SlotTablePatch make_evict_patch(uint32_t layer, uint32_t expert, uint32_t n_experts);

// Eviction has no device kernel in production: it is a host table patch plus a post-refill table upload.
// The explicit plan makes that host-only path (and its ordering flag) part of the launch interface.
struct EvictionPlan {
    bool ok = false;
    std::string why;
    bool host_only = true;
    SlotTablePatch patch;
    SlotTableUpload upload;
};
EvictionPlan build_eviction_plan(const SlotCacheView& view, uint32_t layer, uint32_t expert);
SlotTablePatch make_admit_patch(uint32_t layer, uint32_t expert, uint32_t slot, uint32_t n_experts);
SlotTableUpload make_table_upload(uint32_t layer, uint32_t n_experts);

// Applies one patch to a flattened [n_layers][n_experts] int32 host table. False on out-of-range ids,
// on evicting an expert that owns a different slot, or on admitting an expert twice.
bool apply_slot_patch(std::vector<int32_t>& table, uint32_t n_layers, uint32_t n_experts,
                      const SlotTablePatch& patch, std::string* why = nullptr);

// Assembles one decode step's cache maintenance from the host LRU admissions, in the only order the
// engine may enqueue: (1) host table evict/install patches, (2) refill launches, (3) wait, (4) table
// uploads. `uploads` is deduplicated per layer.
struct AdmissionOp {
    moecache::Admission admission;
    bool has_evict = false;  // false when admission.evicted < 0
    SlotTablePatch evict;    // only applied when has_evict
    SlotTablePatch install;
    RefillPlan refill;
};

struct AdmissionPlan {
    bool ok = false;
    std::string why;
    std::vector<AdmissionOp> ops;
    std::vector<SlotTableUpload> uploads;
    bool uploads_after_refills = true;
};

AdmissionPlan build_admission_plan(const SlotCacheView& view, const std::vector<moecache::Admission>& admissions);

} // namespace vegajet::moeslot
