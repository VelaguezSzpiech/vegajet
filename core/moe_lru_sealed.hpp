// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Launch interfaces for the SEALED cache kernels in ops/kernels/vj_moe_lru_refill_dot8.hiph (host side).
//
// This is a different cache payload from the ACTIVE production path:
//   * active production (`ops/kernels/vj_moe_dot8.hip`, `vj_moe_gu_dot8_lru_kernel`) uses
//     Dot8RowLayout from core/q4_dot8_repack.hpp: merged Gate+Up rows, embedded scales, 3,328 B row
//     stride at D=3072, and the expert->slot table is read inside the dispatch;
//   * the sealed header instead writes a linear code plane (row stride = stride(cols/2)) plus a
//     separate Scale plane (row stride = stride((cols/32)*4)), and `vj_moe_lru_hit_dot{4,8}` take a
//     per-job `slots[]` array. The two layouts are not interchangeable.
//
// Symbols covered (exact names, all in ops/kernels/vj_moe_lru_refill_dot8.hiph):
//   vj_moe_lru_refill_dot8 : (src, qs, ds, rows, cols, type, tables)         grid=ceil(rows*cols/32/256), block=256
//   vj_moe_lru_hit_dot4    : (qs, ds, x, xd, slots, tokens, y, rows, cols, type, stagger)
//   vj_moe_lru_hit_dot8    : same arguments; packed-int4 activations instead of int8
//                                                                           grid=(ceil(rows/4), jobs), block=256
//
// The refill kernel has no slot parameter: the caller must pass the destination slot base
// (cache_base + slot * rows * row_stride), which this builder computes. The hit kernels receive the
// cache base and index `slots[job]` themselves, so their base must stay slot 0.
//
// No device access: builders only produce descriptors and validate host-visible layout invariants.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vegajet::moeslot::sealed {

inline constexpr uint32_t kMaxCards = 2;
inline constexpr uint32_t kRefillBlockThreads = 256;
inline constexpr uint32_t kHitBlockThreads = 256;
inline constexpr uint32_t kHitRowsPerBlock = 4; // 4 rows x 64 lanes
inline constexpr uint32_t kRefillKernargBytes = 64; // verified amdhsa .kernarg_segment_size (TheRock 7.12)
inline constexpr uint32_t kHitKernargBytes = 72;    // verified amdhsa .kernarg_segment_size (TheRock 7.12)

constexpr const char* kRefillSymbol = "vj_moe_lru_refill_dot8";
constexpr const char* kHitDot4Symbol = "vj_moe_lru_hit_dot4";
constexpr const char* kHitDot8Symbol = "vj_moe_lru_hit_dot8";

// Must match `enum class Source : uint32_t { iq2_s, iq3_xxs }` in the sealed header.
enum class Source : uint32_t { iq2_s = 0, iq3_xxs = 1 };

// Device addresses of the three GGML tables. Tables is 24 ABI bytes inside the refill kernarg.
struct Tables {
    uint64_t iq2 = 0;
    uint64_t iq3 = 0;
    uint64_t signs = 0;
};
static_assert(sizeof(Tables) == 24, "Tables is 24 bytes");

// ABI image of vj_moe_lru_refill_dot8. Offset 36..39 is the ABI's implicit 8-byte alignment padding
// before Tables; the C++ struct reproduces it automatically.
struct RefillKernarg {
    uint64_t src = 0; // raw expert-major source bytes for this one expert
    uint64_t qs = 0;  // destination code plane, already offset to this slot
    uint64_t ds = 0;  // destination Scale plane, already offset to this slot
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t type = 0;
    uint32_t _pad = 0;
    Tables tables{};
};
static_assert(sizeof(RefillKernarg) == kRefillKernargBytes, "refill kernarg must be 64 bytes");
static_assert(offsetof(RefillKernarg, src) == 0 && offsetof(RefillKernarg, qs) == 8 &&
                  offsetof(RefillKernarg, ds) == 16 && offsetof(RefillKernarg, rows) == 24 &&
                  offsetof(RefillKernarg, cols) == 28 && offsetof(RefillKernarg, type) == 32 &&
                  offsetof(RefillKernarg, tables) == 40,
              "refill kernarg offsets must match the sealed kernel ABI");

// ABI image of both hit kernels (the argument list is identical).
struct HitKernarg {
    uint64_t qs = 0;
    uint64_t ds = 0;
    uint64_t x = 0;
    uint64_t xd = 0;
    uint64_t slots = 0;
    uint64_t tokens = 0;
    uint64_t y = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t type = 0;
    uint32_t stagger = 0; // 0 = canonical K order; nonzero rotates tiles and may change FP summation order
};
static_assert(sizeof(HitKernarg) == kHitKernargBytes, "hit kernarg must be 72 bytes");
static_assert(offsetof(HitKernarg, qs) == 0 && offsetof(HitKernarg, y) == 48 &&
                  offsetof(HitKernarg, rows) == 56 && offsetof(HitKernarg, cols) == 60 &&
                  offsetof(HitKernarg, type) == 64 && offsetof(HitKernarg, stagger) == 68,
              "hit kernarg offsets must match the sealed kernel ABI");

struct LaunchDesc {
    bool ok = false;
    std::string why;
    const char* symbol = nullptr;
    uint32_t grid_x = 0, grid_y = 0, grid_z = 1;
    uint32_t block_x = 0, block_y = 1, block_z = 1;
    uint32_t kernarg_bytes = 0;
    bool barrier = true;
};

// Exact sealed cache layout. Keep these names distinct from Dot8RowLayout on purpose.
struct CacheLayout {
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t blocks32 = 0;        // cols / 32
    uint32_t qs_row_bytes = 0;    // cols / 2 used bytes per row
    uint32_t qs_row_stride = 0;   // sealed `stride(cols/2)`: (((bytes+255)/256)|1)*256
    uint32_t scale_row_bytes = 0; // blocks32 * 4 used bytes per row
    uint32_t scale_row_stride = 0;
    uint64_t qs_slot_bytes() const { return static_cast<uint64_t>(rows) * qs_row_stride; }
    uint64_t scale_slot_bytes() const { return static_cast<uint64_t>(rows) * scale_row_stride; }
    uint64_t slot_bytes() const { return qs_slot_bytes() + scale_slot_bytes(); }
};

// The device-side stride helper, reproduced exactly.
uint32_t sealed_stride(uint32_t bytes);

// Validates rows>0, cols>0 and cols%256==0; returns a zeroed layout with `why` set on failure.
CacheLayout cache_layout(uint32_t rows, uint32_t cols, std::string* why = nullptr);

uint64_t qs_slot_offset(const CacheLayout& lay, uint32_t slot);
uint64_t scale_slot_offset(const CacheLayout& lay, uint32_t slot);
uint64_t qs_row_offset(const CacheLayout& lay, uint32_t slot, uint32_t row);
uint64_t scale_row_offset(const CacheLayout& lay, uint32_t slot, uint32_t row);

// Raw bytes per expert for the refill source: rows * (cols/256) * (82 IQ2_S | 98 IQ3_XXS).
uint64_t raw_expert_bytes(uint32_t rows, uint32_t cols, Source type, std::string* why = nullptr);

struct RefillRequest {
    uint32_t rows = 0;
    uint32_t cols = 0;
    Source source = Source::iq2_s;
    Tables tables{};
    uint32_t cards = 2;
    std::array<uint64_t, kMaxCards> raw_plane{}; // expert-major raw tensor base on each card
    std::array<uint64_t, kMaxCards> qs_cache{};  // cache code plane base (slot 0) on each card
    std::array<uint64_t, kMaxCards> ds_cache{};  // cache Scale plane base (slot 0) on each card
};

struct RefillPlan {
    bool ok = false;
    std::string why;
    LaunchDesc launch;
    CacheLayout layout;
    uint64_t qs_slot_offset = 0;
    uint64_t scale_slot_offset = 0;
    std::array<RefillKernarg, kMaxCards> args{};
    uint32_t cards = 0;
};

// One launch per card for the same expert/slot. `slot` must be < `slots`.
RefillPlan build_refill_plan(const RefillRequest& req, uint32_t expert, uint32_t slot, uint32_t slots);

enum class HitVariant : uint8_t { dot4, dot8 };

struct HitRequest {
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t jobs = 0;       // elements in slots[], tokens[] and rows of y
    uint32_t slot_count = 0; // number of cache slots (bounds host_slots when supplied)
    Source source = Source::iq2_s;
    uint32_t cards = 2;
    std::array<uint64_t, kMaxCards> qs_cache{};
    std::array<uint64_t, kMaxCards> ds_cache{};
    std::array<uint64_t, kMaxCards> x{};      // packed activations
    std::array<uint64_t, kMaxCards> xd{};     // per-block float scales
    std::array<uint64_t, kMaxCards> slots{};  // device uint32[jobs]
    std::array<uint64_t, kMaxCards> tokens{}; // device uint32[jobs]
    std::array<uint64_t, kMaxCards> y{};      // device float[jobs][rows]
    // Optional host copies used only for validation, never for launch arguments.
    const std::vector<uint32_t>* host_slots = nullptr;
    const std::vector<uint32_t>* host_tokens = nullptr;
    uint32_t token_count = 0;
};

struct HitPlan {
    bool ok = false;
    std::string why;
    LaunchDesc launch;
    CacheLayout layout;
    std::array<HitKernarg, kMaxCards> args{};
    uint32_t cards = 0;
};

// `stagger` must stay 0 for parity; a nonzero value is rejected unless allow_noncanonical is true.
HitPlan build_hit_plan(const HitRequest& req, HitVariant variant, uint32_t stagger = 0,
                       bool allow_noncanonical = false);

// Host-side slot readiness/generation ledger for the sealed cache. The sealed hit kernels read a
// per-job slot array with no device-side table and no "loading" check, so the caller must never build a
// hit dispatch from a slot that is being refilled, and must not use a snapshot taken before an eviction.
// Generation increments on every eviction/refill start; completing a refill does NOT increment it.
class SlotLedger {
public:
    SlotLedger(uint32_t layers, uint32_t slots);

    uint32_t layers() const { return static_cast<uint32_t>(layers_.size()); }
    uint32_t slots() const { return slots_; }
    uint64_t generation(uint32_t layer) const;
    bool pending(uint32_t layer, uint32_t slot) const;
    bool ready(uint32_t layer, uint32_t slot) const;
    int32_t expert(uint32_t layer, uint32_t slot) const;
    uint32_t pending_count() const;

    // Starts (or restarts) a refill. Returns a ticket, or 0 on invalid arguments. Starting a refill on a
    // slot that is ready or pending evicts/invalidates its previous binding and bumps the generation.
    uint64_t begin_refill(uint32_t layer, uint32_t expert, uint32_t slot, std::string* why = nullptr);
    // Marks the slot ready. Fails if the ticket is stale (slot re-bound or evicted in between).
    bool complete_refill(uint64_t ticket, std::string* why = nullptr);
    // Clears a binding and bumps the generation. Fails if the slot is already empty and not pending.
    bool evict(uint32_t layer, uint32_t slot, std::string* why = nullptr);

    std::vector<uint32_t> ready_slots(uint32_t layer) const;

    struct Snapshot {
        uint32_t layer = 0;
        uint64_t generation = 0;
        std::vector<uint32_t> slots;
    };

    // Captures ready slots at the current layer generation; refuses pending/unready slots.
    bool snapshot(uint32_t layer, const std::vector<uint32_t>& slots, Snapshot* out,
                  std::string* why = nullptr) const;
    // True only when the snapshot's generation is current and every slot is still ready.
    bool validate_snapshot(const Snapshot& snap, std::string* why = nullptr) const;

private:
    struct Slot {
        int32_t expert = -1;
        uint64_t ticket = 0;
        bool pending = false;
        bool ready = false;
    };
    struct Layer {
        uint64_t generation = 0;
        std::vector<Slot> slot;
    };
    std::vector<Layer> layers_;
    uint32_t slots_ = 0;
    uint64_t next_ticket_ = 1;
};

// Builds a hit plan only if `ledger` still matches `snapshot` (current generation, every slot ready).
// A stale snapshot or a slot under refill returns ok=false; this is the dispatch-side enforcement of
// the generation/pending contract.
HitPlan build_hit_plan_checked(const HitRequest& req, HitVariant variant, const SlotLedger& ledger,
                               const SlotLedger::Snapshot& snapshot, uint32_t stagger = 0,
                               bool allow_noncanonical = false);

} // namespace vegajet::moeslot::sealed
