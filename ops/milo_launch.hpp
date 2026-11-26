// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
#include <cstdint>
namespace vegajet::milo {
// INT3 factor storage. x is the original block32 Q8 packet, not the lossy int4 packet.
struct Kernarg {
    uint64_t u, v, us, vs, x, xds, tmp, y;
    uint32_t rows, cols, rank, M;
};
static_assert(sizeof(Kernarg)==80);
}
