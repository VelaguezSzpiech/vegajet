// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Shared host/device contract for the opt-in 240-WG router + shared Gate/Up kernel
// (ops/kernels/vj_ornith_router_s240.hip).
//
// The kernel and every host packer/reader include this header, so the 256-B tile addressing is literally
// the same expression on both sides:
//   * plain mode: physical tile == logical tile inside a caller-chosen row stride (the engine's existing
//     odd-256B padded planes pass their stride here);
//   * XOR mode: every row owns next_pow2(ceil(raw/256)) whole 256-B tiles and a logical tile kt lives in
//     physical tile kt ^ (row & mask). The power-of-two capacity guarantees the XOR never leaves the row.
//     Partial final tiles and non-power-of-two tile counts are supported; padding tiles are never read.
//
// Split-K is a separate dimension: split_begin()/split_end() partition the 32-weight blocks of a row into
// contiguous, coverage-exact ranges. The phase-1 kernel writes partials[split][row]; the reduce kernel adds
// them in fixed part order (0..split-1), so results are deterministic across runs (no FP atomics).
#pragma once

#include <stdint.h>

#if defined(__HIP__) || defined(__HIPCC__) || defined(__CUDACC__)
#define VJ_RTR_S240_HD __host__ __device__
#else
#define VJ_RTR_S240_HD
#endif

namespace vegajet {
namespace router_s240 {

constexpr uint32_t kTileBytes = 256u;    // TCC tile granularity
constexpr uint32_t kTileU32x4 = 16u;     // 16 x 16 B per tile
constexpr uint32_t kRouterRows = 256u;   // Ornith router experts
constexpr uint32_t kPhase1Wgs = 240u;    // 4 WGs/CU x 60 CUs (max resident on the 2x MI50 box)
constexpr uint32_t kThreads = 256u;
constexpr uint32_t kLanes = 32u;         // one wave32 per row-part
constexpr uint32_t kMaxSplitK = 8u;
constexpr uint32_t kScratchAlign = 256u;

VJ_RTR_S240_HD inline uint32_t ceil_div_u32(uint32_t a, uint32_t b) { return b ? (a + b - 1u) / b : 0u; }

VJ_RTR_S240_HD inline uint32_t next_pow2_u32(uint32_t v) {
    if (v <= 1u) return 1u;
    --v;
    v |= v >> 1u;
    v |= v >> 2u;
    v |= v >> 4u;
    v |= v >> 8u;
    v |= v >> 16u;
    return v + 1u;
}

struct PlaneLayout {
    uint32_t raw_row_bytes = 0;   // data bytes per row
    uint32_t stride_bytes = 0;    // physical row pitch (multiple of kTileBytes in XOR mode)
    uint32_t data_tiles = 0;      // ceil(raw_row_bytes / kTileBytes)
    uint32_t phys_tiles = 0;      // stride_bytes / kTileBytes
    uint32_t xor_mask = 0;        // 0: plain; else phys_tiles - 1 (power of two)
    uint32_t xor_enabled = 0;
};

// plain_stride_bytes == 0 selects raw_row_bytes as the pitch (dense rows). A non-zero value must be
// >= raw_row_bytes and a multiple of 16 (uint4 alignment); the engine passes its odd-256B TCC pitch.
VJ_RTR_S240_HD inline PlaneLayout make_plane_layout(uint32_t raw_row_bytes, uint32_t plain_stride_bytes,
                                                    uint32_t xor_enabled) {
    PlaneLayout l;
    l.raw_row_bytes = raw_row_bytes;
    l.data_tiles = ceil_div_u32(raw_row_bytes, kTileBytes);
    if (xor_enabled != 0u && raw_row_bytes != 0u) {
        l.phys_tiles = next_pow2_u32(l.data_tiles ? l.data_tiles : 1u);
        l.stride_bytes = l.phys_tiles * kTileBytes;
        l.xor_mask = l.phys_tiles - 1u;
        l.xor_enabled = 1u;
    } else {
        l.stride_bytes = plain_stride_bytes ? plain_stride_bytes : raw_row_bytes;
        l.phys_tiles = ceil_div_u32(l.stride_bytes, kTileBytes);
        l.xor_mask = 0u;
        l.xor_enabled = 0u;
    }
    return l;
}

// Physical 256-B tile of a logical tile in `row`. Only called with kt < data_tiles.
VJ_RTR_S240_HD inline uint32_t phys_tile(const PlaneLayout& l, uint32_t row, uint32_t kt) {
    return kt ^ (row & l.xor_mask);
}

// Byte offset of logical byte `off` (< raw_row_bytes) of `row`.
VJ_RTR_S240_HD inline uint64_t plane_byte_offset(const PlaneLayout& l, uint32_t row, uint32_t off) {
    const uint32_t kt = off / kTileBytes;
    const uint32_t in = off % kTileBytes;
    return static_cast<uint64_t>(row) * l.stride_bytes +
           static_cast<uint64_t>(phys_tile(l, row, kt)) * kTileBytes + in;
}

// Index in 16-B (uint4) units. `idx` is the logical index; returns the physical one.
VJ_RTR_S240_HD inline uint32_t block_index(const PlaneLayout& l, uint32_t row, uint32_t idx) {
    const uint32_t tile = idx / kTileU32x4;
    const uint32_t in = idx % kTileU32x4;
    return (phys_tile(l, row, tile) * kTileU32x4) + in;
}

// Deterministic Split-K: contiguous 32-weight blocks [begin, end) of `nb` total blocks.
VJ_RTR_S240_HD inline uint32_t split_begin(uint32_t nb, uint32_t split_k, uint32_t part) {
    return split_k ? (part * nb) / split_k : 0u;
}
VJ_RTR_S240_HD inline uint32_t split_end(uint32_t nb, uint32_t split_k, uint32_t part) {
    return split_k ? ((part + 1u) * nb) / split_k : 0u;
}

// Phase-1 work distribution over the 239 weight WGs (WG 239 owns the shared gate scalar).
// Every 32-lane group takes two round-robin streams:
//   shared Gate/Up: t = grp*239 + wg, stride 239*8 -> one row-part each (1x cost)
//   router FP16    : t = grp*239 + wg, stride 239*8 -> one row-part each (4x cost: 12 uint4 vs 3)
// Column-major group ids interleave the two streams across WGs, so every WG carries near-identical
// weighted work for any sh_rows and split_k, including non-power-of-two counts and a short tail.
constexpr uint32_t kWorkerWgs = kPhase1Wgs - 1u;   // 239
constexpr uint32_t kWorkerGroups = kWorkerWgs * (kThreads / kLanes);  // 1912
static_assert(kThreads == 256u && kLanes == 32u && kWorkerGroups == 1912u,
              "phase-1 kernel signature assumes 256 threads = 8 groups of 32 lanes");

struct TaskPlan {
    uint32_t sh_tasks = 0;      // sh_rows * split_k
    uint32_t rtr_tasks = 0;     // 256 * split_k
    uint32_t stride = 0;        // kWorkerGroups
    uint32_t reduce_rows = 0;
    uint32_t reduce_wgs = 0;
};

VJ_RTR_S240_HD inline TaskPlan make_task_plan(uint32_t sh_rows, uint32_t split_k) {
    TaskPlan p;
    p.sh_tasks = sh_rows * split_k;
    p.rtr_tasks = kRouterRows * split_k;
    p.stride = kWorkerGroups;
    p.reduce_rows = sh_rows + kRouterRows + 1u;  // + shared gate scalar
    p.reduce_wgs = ceil_div_u32(p.reduce_rows, kThreads);
    return p;
}

// Task index -> (split part, row) for each stream. Part-major, contiguous.
VJ_RTR_S240_HD inline void sh_task_split(uint32_t t, uint32_t sh_rows, uint32_t split_k,
                                         uint32_t* part, uint32_t* row) {
    *part = (split_k && sh_rows) ? t / sh_rows : 0u;
    *row = sh_rows ? t % sh_rows : 0u;
}
VJ_RTR_S240_HD inline void rtr_task_split(uint32_t t, uint32_t split_k, uint32_t* part, uint32_t* row) {
    *part = split_k ? t / kRouterRows : 0u;
    *row = t % kRouterRows;
}

// Scratch is three FP32 partial arrays carved out of one base (byte offsets, 256-B aligned).
struct ScratchPlan {
    uint64_t gate_off = 0;
    uint64_t rtr_off = 0;
    uint64_t sh_off = 0;
    uint64_t bytes = 0;
};

VJ_RTR_S240_HD inline uint64_t align_up_u64(uint64_t v, uint64_t a) { return (v + a - 1u) / a * a; }

VJ_RTR_S240_HD inline ScratchPlan make_scratch_plan(uint32_t sh_rows, uint32_t split_k) {
    ScratchPlan s;
    s.gate_off = 0u;  // split_k * 4 B, rounded up below
    s.rtr_off = align_up_u64(static_cast<uint64_t>(split_k) * 4u, kScratchAlign);
    s.sh_off = align_up_u64(s.rtr_off + static_cast<uint64_t>(split_k) * kRouterRows * 4u, kScratchAlign);
    s.bytes = s.sh_off + static_cast<uint64_t>(split_k) * sh_rows * 4u;
    return s;
}

// Kernel argument layouts (by value; the host and the device compile this same struct).
struct Phase1Args {
    uint64_t w_router_f16;   // [256, wr_stride_elems] FP16
    uint64_t w_sh_gate;      // [D] FP32
    uint64_t x_norm;         // [D] FP32
    uint64_t sh_w_qs;        // VJ_Q4_A8 shared Gate/Up codes
    uint64_t sh_w_dm;        // VJ_Q4_A8 shared Gate/Up (FP16 d | FP16 m << 16) per 32
    uint64_t x_q8;           // [D] int8, 32-weight blocks
    uint64_t x_ds;           // [D/32] float2
    uint64_t rtr_part;       // [split_k, 256] FP32
    uint64_t sh_part;        // [split_k, sh_rows] FP32
    uint64_t gate_part;      // [split_k] FP32
    uint32_t D;
    uint32_t sh_rows;
    uint32_t wr_stride_elems;
    uint32_t sh_qs_stride_bytes;
    uint32_t sh_dm_stride_u32;
    uint32_t split_k;
    uint32_t wr_xor_mask;
    uint32_t sh_xor_mask;
};

struct ReduceArgs {
    uint64_t rtr_part;
    uint64_t sh_part;
    uint64_t gate_part;
    uint64_t logits;         // [256]
    uint64_t sh_gu;          // [sh_rows]
    uint64_t sh_gate_sc;     // [1]
    uint32_t sh_rows;
    uint32_t split_k;
};

static_assert(sizeof(Phase1Args) % 8u == 0u, "Phase1Args must stay 8-byte aligned for the AQL kernarg copy");
static_assert(sizeof(ReduceArgs) % 8u == 0u, "ReduceArgs must stay 8-byte aligned for the AQL kernarg copy");

} // namespace router_s240
} // namespace vegajet
