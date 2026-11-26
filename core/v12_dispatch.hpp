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
} // namespace vegajet::dispatch
