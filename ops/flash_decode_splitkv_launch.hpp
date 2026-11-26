// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Host launch contract for the sealed gfx906 Q8_0 / Q4_0 Split-KV FlashDecoding pair in
// ops/kernels/vj_flash_decode_splitkv.hiph:
//   vj_flash_decode_q8_partition, vj_flash_decode_q4_partition, vj_flash_decode_merge.
//
// The device kernels are sealed: this header does not redefine or wrap their math. It gives one
// host-only definition (pure C++17, no HIP headers) of
//   * the kernel argument images and the declared kernarg byte counts,
//   * the launch geometry,
//   * the scratch byte plan with overflow-checked sizes and 256 B alignment,
//   * a safe selection policy that only engages Split-KV when the sealed contract and the
//     caller's storage layout allow it, and otherwise returns a typed fallback reason.
//
// WIRE LAYOUT (must stay byte-identical to the device header; all allocations 16 B aligned):
//   q8 block 48 B: FP16 d at byte 0, ignored bytes 2..15, 32 signed int8 at byte 16.
//   q4 block 32 B: FP16 d at byte 0, ignored bytes 2..15, 16 native nibble pairs at byte 16
//                  (low nibble = element j, high nibble = element j+16, unsigned biased by 8).
//   Q[M][16][8][B], K[T][8][B], V[T][8][B]; B is 48 (Q8) or 32 (Q4).
//   The 48/32 B blocks are aligned envelopes of native blocks, NOT byte-compatible GGUF strides.
//
// ACTIVE ATTENTION MAPPING (ornith_attn_flash_quant_kernel: 16 local Q heads, 1 local KV head,
// head_dim 256, causal):
//   with pos0 tokens cached before this step and M query rows,
//     T     = pos0 + M cache rows,
//     past  = pos0,
//     stop  = min(split_end, past + m + 1)  ==  [0, pos0 + m] for row m,
//   which is exactly the active kernel's visible range (seqlen = pos0 + m + 1). Rows 0..M-1 are
//   therefore decoded/verified with one shared split partition, and only the causal tail differs.
//
// STORAGE PRECONDITION (why selection can refuse): the active Q8 KV cache row is 272 B
// (8 FP16 scales in bytes 0..15, then 256 int8) and the active 1D flash kernel only reads that
// layout, while the sealed partition reads 384 B (Q8) / 256 B (Q4) rows with the block scale
// inline in every block. The two row layouts cannot be converted in place, so Split-KV is only
// selected when Q, K and V are already declared to live in the same wire format. Otherwise the
// caller keeps the existing single-pass path. S == 1 remains the sealed pair's correctness case
// (the device header says so) and is what the CPU contract test exercises for tails; the policy
// below never selects it for a production launch because two dispatches buy no parallelism.
//
// KERNEL CONTRACT (quoted, enforced below): block (256,1,1); partition grid (S,M,1); merge grid
// (16,M,1); M is 1 or 5..8; T >= 0, past >= 0, S >= 1; empty splits write zeros and -inf;
// attention_scale finite with finite scores; all pointers nonaliasing and valid for their whole
// declared extent; dimensions fit signed int. Scratch products are checked against uint64.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace vegajet::flash_decode_splitkv {

// ---- sealed wire constants -------------------------------------------------
constexpr uint32_t kHeads = 16;           // Q heads covered by one partition block
constexpr uint32_t kHeadDim = 256;        // elements per head
constexpr uint32_t kBlocksPerHead = 8;    // 32-element blocks per 256-dim head
constexpr uint32_t kQ8BlockBytes = 48;
constexpr uint32_t kQ4BlockBytes = 32;
constexpr uint32_t kPartitionThreads = 256;
constexpr uint32_t kMergeThreads = 256;
constexpr uint32_t kMaxSplits = 64;       // host policy cap (the device accepts any S >= 1)
constexpr uint32_t kScratchAlign = 256;   // one arena/TCC stride unit
// Declared kernarg segment sizes of the compiled gfx906 code object (verified against the
// amdhsa metadata by evidence/M1/verify.sh).
constexpr uint32_t kPartitionKernargBytes = 56;
constexpr uint32_t kMergeKernargBytes = 36;
constexpr float kDefaultAttentionScale = 1.0f / 16.0f;

static_assert(kHeads * 16u == kHeadDim, "16 lanes x 16 accumulators cover one 256-dim head");
static_assert(kBlocksPerHead * 32u == kHeadDim, "8 blocks x 32 elements == one head");

enum class WireFormat : uint8_t { None = 0, Q8, Q4 };

constexpr uint32_t block_bytes(WireFormat format) {
    return format == WireFormat::Q8 ? kQ8BlockBytes
         : format == WireFormat::Q4 ? kQ4BlockBytes : 0u;
}
constexpr uint32_t head_bytes(WireFormat format) {
    return kBlocksPerHead * block_bytes(format);
}

// ---- kernel argument images -----------------------------------------------
// Field order and types are the device parameter lists. Offsets are pinned to the HIP kernarg
// ABI (natural alignment, declaration order) so a C++ struct cannot silently drift.
struct PartitionArgs {
    uint64_t q;       // device VA of Q[M][16][8][B]
    uint64_t k;       // device VA of K[T][8][B]
    uint64_t v;       // device VA of V[T][8][B]
    uint64_t partial; // device VA of [M][16][S][256] FP32
    uint64_t lse;     // device VA of [M][16][S] FP32
    int T;
    int past;
    int splits;
    float attention_scale;
};
static_assert(sizeof(PartitionArgs) == kPartitionKernargBytes,
              "partition kernarg image must be 56 B");
static_assert(offsetof(PartitionArgs, q) == 0);
static_assert(offsetof(PartitionArgs, k) == 8);
static_assert(offsetof(PartitionArgs, v) == 16);
static_assert(offsetof(PartitionArgs, partial) == 24);
static_assert(offsetof(PartitionArgs, lse) == 32);
static_assert(offsetof(PartitionArgs, T) == 40);
static_assert(offsetof(PartitionArgs, past) == 44);
static_assert(offsetof(PartitionArgs, splits) == 48);
static_assert(offsetof(PartitionArgs, attention_scale) == 52);
static_assert(sizeof(int) == 4, "kernel scalars are 32-bit int");

struct MergeArgs {
    uint64_t partial;
    uint64_t lse;
    uint64_t output;     // device VA of [M][16][256] FP32
    uint64_t output_lse; // device VA of [M][16] FP32
    int splits;
};
// The declared kernarg segment carries the five fields without trailing padding (36 B);
// the C++ struct rounds to 40 B. Passing sizeof(MergeArgs) is harmless (the runtime kernarg
// slot is 512 B) but the declared size is what the metadata gate checks.
static_assert(offsetof(MergeArgs, partial) == 0);
static_assert(offsetof(MergeArgs, lse) == 8);
static_assert(offsetof(MergeArgs, output) == 16);
static_assert(offsetof(MergeArgs, output_lse) == 24);
static_assert(offsetof(MergeArgs, splits) == 32);
static_assert(sizeof(MergeArgs) == 40, "natural tail padding of the 36 B kernarg image");

struct LaunchGeometry {
    uint32_t grid_x, grid_y, grid_z;
    uint32_t block_x, block_y, block_z;
};

constexpr LaunchGeometry partition_geometry(uint32_t splits, uint32_t rows) {
    return LaunchGeometry{splits, rows, 1u, kPartitionThreads, 1u, 1u};
}
constexpr LaunchGeometry merge_geometry(uint32_t rows) {
    return LaunchGeometry{kHeads, rows, 1u, kMergeThreads, 1u, 1u};
}

// ---- split boundaries (device formula, uint64 product) --------------------
constexpr uint32_t split_begin(uint64_t cache_rows, uint32_t split, uint32_t splits) {
    return splits == 0u ? 0u : static_cast<uint32_t>(cache_rows * split / splits);
}
constexpr uint32_t split_end(uint64_t cache_rows, uint32_t split, uint32_t splits) {
    return splits == 0u ? 0u : static_cast<uint32_t>(cache_rows * (split + 1u) / splits);
}
// Exclusive device stop: min(split_end, past + row + 1).
constexpr uint32_t split_stop(uint64_t cache_rows, uint32_t split, uint32_t splits,
                              uint64_t causal_end) {
    const uint32_t end = split_end(cache_rows, split, splits);
    return causal_end < end ? static_cast<uint32_t>(causal_end) : end;
}

// ---- shape and wire byte plan ---------------------------------------------
struct Shape {
    uint32_t rows = 0;        // M (query rows): 1 or 5..8
    uint32_t heads = 0;       // n_q_local: 16
    uint32_t kv_heads = 0;    // n_kv_local: 1 (one K/V wire row per position)
    uint32_t head_dim = 0;    // 256
    uint64_t pos0 = 0;        // tokens cached before this step (== past)
    uint64_t cache_rows = 0;  // T cache rows, must equal pos0 + rows
};

constexpr Shape active_shape(uint32_t rows, uint64_t pos0, uint32_t heads = kHeads,
                             uint32_t kv_heads = 1, uint32_t head_dim = kHeadDim) {
    return Shape{rows, heads, kv_heads, head_dim, pos0, pos0 + rows};
}

struct WirePlan {
    uint64_t q_bytes = 0;
    uint64_t k_bytes = 0;
    uint64_t v_bytes = 0;
    uint64_t total_bytes = 0;
    bool overflow = false;
};

namespace detail {

constexpr bool mul_checked(uint64_t a, uint64_t b, uint64_t& out) {
    if (a == 0u || b == 0u) { out = 0u; return true; }
    if (a > std::numeric_limits<uint64_t>::max() / b) return false;
    out = a * b;
    return true;
}

constexpr bool add_checked(uint64_t a, uint64_t b, uint64_t& out) {
    if (a > std::numeric_limits<uint64_t>::max() - b) return false;
    out = a + b;
    return true;
}

constexpr bool align_up(uint64_t value, uint64_t alignment, uint64_t& out) {
    if (alignment == 0u) return false;
    const uint64_t remainder = value % alignment;
    if (remainder == 0u) { out = value; return true; }
    return add_checked(value, alignment - remainder, out);
}

} // namespace detail

inline WirePlan plan_wire(const Shape& shape, WireFormat format) {
    WirePlan plan;
    const uint64_t head = head_bytes(format);
    if (head == 0u) return plan; // no wire launch requested
    uint64_t rows_heads = 0, q = 0, kv = 0, sum = 0;
    const bool ok = detail::mul_checked(shape.rows, shape.heads, rows_heads) &&
                    detail::mul_checked(rows_heads, head, q) &&
                    detail::mul_checked(shape.cache_rows, head, kv) &&
                    detail::add_checked(q, kv, sum) &&
                    detail::add_checked(sum, kv, plan.total_bytes);
    if (!ok) {
        plan.overflow = true;
        plan.total_bytes = 0u;
        return plan;
    }
    plan.q_bytes = q;
    plan.k_bytes = kv;
    plan.v_bytes = kv; // V has exactly the K extent
    return plan;
}

// Transient Scratch for the sealed pair, laid out in one arena slice:
//   partial[M][16][S][256] FP32, lse[M][16][S] FP32,
//   output[M][16][256] FP32, output_lse[M][16] FP32,
// each region starting on a 256 B boundary. total_bytes is the aligned end, so a caller can
// hand one arena allocation to partition (first two regions) and merge (last two regions).
struct ScratchPlan {
    uint64_t partial_offset = 0, partial_bytes = 0;
    uint64_t lse_offset = 0, lse_bytes = 0;
    uint64_t output_offset = 0, output_bytes = 0;
    uint64_t output_lse_offset = 0, output_lse_bytes = 0;
    uint64_t total_bytes = 0;
    bool overflow = false;
};

inline ScratchPlan plan_scratch(uint64_t rows, uint64_t splits) {
    ScratchPlan plan;
    if (rows == 0u || splits == 0u) { // fail closed: the device contract needs M >= 1, S >= 1
        plan.overflow = true;
        return plan;
    }
    uint64_t row_heads = 0, row_heads_splits = 0;
    uint64_t elems = 0, lse_elems = 0, output_elems = 0, output_lse_elems = 0;
    uint64_t cursor = 0;
    const bool ok = detail::mul_checked(rows, kHeads, row_heads) &&
                    detail::mul_checked(row_heads, splits, row_heads_splits) &&
                    detail::mul_checked(row_heads_splits, kHeadDim, elems) &&
                    detail::mul_checked(row_heads, kHeadDim, output_elems);
    if (!ok) { plan.overflow = true; return plan; }
    lse_elems = row_heads_splits;       // [M][16][S]
    output_lse_elems = row_heads;       // [M][16]

    const auto place = [&](uint64_t count, uint64_t& offset, uint64_t& bytes) -> bool {
        if (!detail::mul_checked(count, 4u, bytes)) return false;
        if (!detail::align_up(cursor, kScratchAlign, offset)) return false;
        if (!detail::add_checked(offset, bytes, cursor)) return false;
        return true;
    };
    if (!place(elems, plan.partial_offset, plan.partial_bytes) ||
        !place(lse_elems, plan.lse_offset, plan.lse_bytes) ||
        !place(output_elems, plan.output_offset, plan.output_bytes) ||
        !place(output_lse_elems, plan.output_lse_offset, plan.output_lse_bytes) ||
        !detail::align_up(cursor, kScratchAlign, plan.total_bytes)) {
        plan = ScratchPlan{};
        plan.overflow = true;
    }
    return plan;
}

// ---- selection -------------------------------------------------------------
struct Policy {
    uint32_t min_tokens_per_split = 32;  // no split shorter than this (floor(T/S))
    uint32_t min_total_tokens = 256;     // do not split contexts this short at all
    // The sequential active path launches one WG per Q head (kHeads * M); Split-KV launches
    // S * M WGs, so S must be strictly larger than kHeads to buy parallelism at all.
    uint32_t min_splits = kHeads + 1;
    uint32_t max_splits = kMaxSplits;
    uint64_t scratch_budget_bytes = 0;   // 0 = unbounded
    bool allow_q4 = false;               // the active pipeline has no Q4 wire writer yet
};

constexpr Policy sanitize_policy(Policy policy) {
    if (policy.min_tokens_per_split == 0u) policy.min_tokens_per_split = 1u;
    if (policy.min_tokens_per_split > (1u << 20)) policy.min_tokens_per_split = 1u << 20;
    if (policy.min_splits < 2u) policy.min_splits = 2u;
    if (policy.min_splits > kMaxSplits) policy.min_splits = kMaxSplits;
    if (policy.max_splits > kMaxSplits) policy.max_splits = kMaxSplits;
    return policy;
}

enum class Fallback : uint8_t {
    None = 0,
    Dimensions,
    BatchRows,
    CacheRows,
    Format,
    AttentionScale,
    Policy,
    SmallContext,
    ScratchBudget,
};

struct Selection {
    bool use_split_kv = false;
    WireFormat format = WireFormat::None;
    uint32_t splits = 1;              // 1 on every fallback; not a launch request
    uint32_t min_split_tokens = 0;    // floor(T/S), 0 unless engaged
    uint32_t max_split_tokens = 0;    // ceil(T/S)
    uint64_t scratch_bytes = 0;
    Fallback fallback = Fallback::None;
    const char* reason = "";
};

inline Selection select(const Shape& shape, WireFormat q_format, WireFormat k_format,
                        WireFormat v_format, float attention_scale,
                        Policy policy = Policy{}) {
    policy = sanitize_policy(policy);

    const auto reject = [](Fallback why, const char* text) {
        Selection out;
        out.fallback = why;
        out.reason = text;
        return out;
    };

    if (shape.heads != kHeads || shape.kv_heads != 1u || shape.head_dim != kHeadDim)
        return reject(Fallback::Dimensions, "sealed pair is 16 local Q heads / 1 KV head / 256 dims");
    if (shape.rows != 1u && (shape.rows < 5u || shape.rows > 8u))
        return reject(Fallback::BatchRows, "sealed pair accepts M = 1 or 5..8");
    if (shape.cache_rows != shape.pos0 + static_cast<uint64_t>(shape.rows))
        return reject(Fallback::CacheRows, "active mapping requires T == pos0 + M");
    if (shape.cache_rows > 0x7fffffffull || shape.pos0 > 0x7fffffffull)
        return reject(Fallback::CacheRows, "T/past must fit the signed int kernel arguments");
    if (q_format == WireFormat::None || q_format != k_format || q_format != v_format)
        return reject(Fallback::Format, "Q/K/V must be present in one identical wire format");
    if (q_format == WireFormat::Q4 && !policy.allow_q4)
        return reject(Fallback::Format, "Q4 wire rejected by policy (no active Q4 cache writer)");
    if (!(attention_scale > 0.0f) || !std::isfinite(attention_scale))
        return reject(Fallback::AttentionScale, "attention_scale must be finite and positive");
    if (policy.max_splits < 2u || policy.max_splits < policy.min_splits)
        return reject(Fallback::Policy, "policy cap leaves fewer splits than the floor");

    const uint64_t total = shape.cache_rows;
    if (total < policy.min_total_tokens ||
        total < 2ull * static_cast<uint64_t>(policy.min_tokens_per_split))
        return reject(Fallback::SmallContext, "context below policy minimum for a useful split");

    uint64_t splits = total / policy.min_tokens_per_split;
    if (splits > policy.max_splits) splits = policy.max_splits;
    if (splits < policy.min_splits)
        return reject(Fallback::SmallContext,
                      "fewer splits than the kHeads+1 parallelism floor");

    ScratchPlan scratch = plan_scratch(shape.rows, splits);
    if (scratch.overflow)
        return reject(Fallback::ScratchBudget, "scratch size does not fit uint64");
    if (policy.scratch_budget_bytes != 0u) {
        while (scratch.total_bytes > policy.scratch_budget_bytes) {
            uint64_t next = splits / 2u;
            if (next < policy.min_splits) next = policy.min_splits;
            if (next >= splits) break; // already at the floor
            splits = next;
            scratch = plan_scratch(shape.rows, splits);
        }
        if (scratch.total_bytes > policy.scratch_budget_bytes)
            return reject(Fallback::ScratchBudget,
                          "scratch budget does not fit the parallelism floor");
    }

    Selection out;
    out.use_split_kv = true;
    out.format = q_format;
    out.splits = static_cast<uint32_t>(splits);
    out.min_split_tokens = static_cast<uint32_t>(total / splits);
    out.max_split_tokens = static_cast<uint32_t>((total + splits - 1u) / splits);
    out.scratch_bytes = scratch.total_bytes;
    out.reason = "split-kv";
    return out;
}

struct Launch {
    Selection selection;
    ScratchPlan scratch;
    WirePlan wire;
    LaunchGeometry partition{0u, 0u, 0u, 0u, 0u, 0u};
    LaunchGeometry merge{0u, 0u, 0u, 0u, 0u, 0u};
    uint32_t partition_kernarg_bytes = kPartitionKernargBytes;
    uint32_t merge_kernarg_bytes = kMergeKernargBytes;
    // Code-object symbols to enqueue; both null on a fallback.
    const char* partition_entry = nullptr;
    const char* merge_entry = nullptr;
};

inline Launch plan_launch(const Shape& shape, WireFormat q_format, WireFormat k_format,
                          WireFormat v_format, float attention_scale,
                          Policy policy = Policy{}) {
    Launch launch;
    launch.selection = select(shape, q_format, k_format, v_format, attention_scale, policy);
    launch.wire = plan_wire(shape, launch.selection.format);
    if (launch.selection.use_split_kv) {
        launch.scratch = plan_scratch(shape.rows, launch.selection.splits);
        launch.partition = partition_geometry(launch.selection.splits, shape.rows);
        launch.merge = merge_geometry(shape.rows);
        launch.partition_entry = launch.selection.format == WireFormat::Q4
                                     ? "vj_flash_decode_q4_partition"
                                     : "vj_flash_decode_q8_partition";
        launch.merge_entry = "vj_flash_decode_merge";
    }
    return launch;
}

// ---- argument builders -----------------------------------------------------
// Addresses are device virtual addresses (kfd::Runtime arenas), matching the rest of the
// engine's kernarg packing. All extents must be nonaliasing and valid for the whole launch;
// output/output_lse must not alias q/k/v/partial/lse.
inline PartitionArgs partition_args(uint64_t q, uint64_t k, uint64_t v, uint64_t partial,
                                    uint64_t lse, const Shape& shape, uint32_t splits,
                                    float attention_scale) {
    PartitionArgs args{};
    args.q = q;
    args.k = k;
    args.v = v;
    args.partial = partial;
    args.lse = lse;
    args.T = static_cast<int>(shape.cache_rows);
    args.past = static_cast<int>(shape.pos0);
    args.splits = static_cast<int>(splits);
    args.attention_scale = attention_scale;
    return args;
}

inline MergeArgs merge_args(uint64_t partial, uint64_t lse, uint64_t output,
                            uint64_t output_lse, uint32_t splits) {
    MergeArgs args{};
    args.partial = partial;
    args.lse = lse;
    args.output = output;
    args.output_lse = output_lse;
    args.splits = static_cast<int>(splits);
    return args;
}

} // namespace vegajet::flash_decode_splitkv
