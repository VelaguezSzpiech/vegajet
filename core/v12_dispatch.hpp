#pragma once
#include <cstdint>
#include <algorithm>
#include "core/grid_planner.hpp"

namespace vegajet::dispatch {
struct CollectiveEpoch {
    uint32_t epoch;
    uint32_t parity;
};
inline CollectiveEpoch next_collective(uint32_t& next) {
    const uint32_t epoch = next++;
    return {epoch, epoch & 1u};
}
// The batched MTP Down ABI still requires symmetric 512-column experts.
inline bool mtp_batch_layout_supported(bool asymmetric, bool batch_catchup) {
    return !(asymmetric && batch_catchup);
}
// Fused verify parks 128 WGs. Unknown metadata must use two dispatches.
inline bool fused_verify_resident(bool metadata, uint32_t n_cu, uint32_t vgpr, uint32_t sgpr,
                                  uint32_t lds, uint32_t wave_cap, uint32_t wgs = 128u) {
    if (!metadata || !n_cu || !vgpr || !sgpr || !wave_cap || !wgs) return false;
    const auto occ = planner::compute_gfx906_occupancy(vgpr, sgpr, lds, 256u, n_cu);
    // Four wave64s per WG, four SIMDs/CU; also respect descriptor wave limits.
    const uint32_t per_cu = std::min(16u, std::min(occ.resident_wgs_per_cu, wave_cap));
    return static_cast<uint64_t>(per_cu) * n_cu >= wgs;
}
inline bool collective_slice_needs_drain(uint32_t offset, uint32_t rows, uint32_t total) {
    return offset < total && rows != 0u && rows < total - offset;
}
inline uint32_t eh_grid_x(uint32_t rows) { return rows * 256u; }
inline bool kv_span_fits(uint32_t pos, uint32_t rows, uint32_t max_ctx) {
    return rows != 0u && pos < max_ctx && rows <= max_ctx - pos;
}
inline bool split_scratch_fits(uint32_t rows, uint32_t heads, uint32_t chunks,
                               uint32_t row_chunk_capacity, uint32_t head_capacity) {
    return rows != 0u && heads != 0u && heads <= head_capacity && chunks != 0u &&
           static_cast<uint64_t>(rows) * chunks <= row_chunk_capacity;
}
} // namespace vegajet::dispatch
