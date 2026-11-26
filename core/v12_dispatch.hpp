#pragma once
#include <cstdint>

namespace vegajet::dispatch {
struct CollectiveEpoch {
    uint32_t epoch;
    uint32_t parity;
};
inline CollectiveEpoch next_collective(uint32_t& next) {
    const uint32_t epoch = next++;
    return {epoch, epoch & 1u};
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
