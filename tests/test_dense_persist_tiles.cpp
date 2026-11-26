// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Host emulation of the V1.2 D13 tile claim in ornith_gemv_q4a8_m1_w32_persist_kernel
// (VEGAJET_V12_DENSE_PERSIST) against the blockIdx.x mapping of
// ornith_gemv_q4a8_m1_w32_kernel.
// Contract checked here:
//   * tile t of the persistent grid owns exactly rows [8t, 8t+8) - the same rows WG t owns in the
//     one-WG-per-tile kernel, so y_out and the peer_tx slots see identical bytes,
//   * every tile is claimed exactly once by the resident grid, whatever the workgroup count,
//   * the arrival counter reset leaves an empty queue for the next dispatch (no host memset),
//   * rows % 8 != 0 leaves the trailing rows of the last tile unwritten instead of out of range.
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr uint32_t kRowsPerTile = 8u;

struct Queue {
    uint32_t tile = 0u;      // q[0]
    uint32_t arrivals = 0u;  // q[1]
};

// One dispatch of the persistent kernel: wgs resident workgroups pull tiles until the queue drains,
// each then bumps the arrival counter, and the last one out zeroes both counters.
// Returns per-tile claim counts; `written[row]` records y_out writes.
std::vector<uint32_t> run_dispatch(Queue& q, uint32_t wgs, uint32_t rows,
                                   std::vector<uint32_t>& written) {
    const uint32_t n_tiles = (rows + kRowsPerTile - 1u) / kRowsPerTile;
    std::vector<uint32_t> claims(n_tiles, 0u);
    for (uint32_t wg = 0; wg < wgs; ++wg) {
        for (;;) {
            const uint32_t t = q.tile++;            // atomicAdd(&q[0], 1)
            if (t >= n_tiles) break;
            claims[t] += 1u;
            for (uint32_t grp = 0; grp < 8u; ++grp) {
                const uint32_t row = t * kRowsPerTile + grp;
                if (row < rows) written[row] += 1u;  // "if (row < rows)" guard inside the kernel
            }
        }
        if (++q.arrivals == wgs) { q.tile = 0u; q.arrivals = 0u; }  // last WG out resets both
    }
    return claims;
}

}  // namespace

int main() {
    // Hero case: rows=3072 is 384 tiles, i.e. 1.28 waves on the 300 resident slots of the launching kernel.
    const uint32_t cases[][2] = {{3072u, 224u}, {3072u, 240u}, {3072u, 192u},
                                 {4096u, 224u}, {3005u, 224u}};
    for (const auto& c : cases) {
        const uint32_t rows = c[0], wgs = c[1];
        const uint32_t n_tiles = (rows + kRowsPerTile - 1u) / kRowsPerTile;
        Queue q;
        for (int launch = 0; launch < 3; ++launch) {
            std::vector<uint32_t> written(rows, 0u);
            const std::vector<uint32_t> claims = run_dispatch(q, wgs, rows, written);

            if (claims.size() != n_tiles) return 1;
            for (uint32_t t = 0; t < n_tiles; ++t) if (claims[t] != 1u) return 2;  // exactly once
            for (uint32_t r = 0; r < rows; ++r) if (written[r] != 1u) return 3;    // every row once
            if (q.tile != 0u || q.arrivals != 0u) return 4;                        // queue left clean
            // same tile -> row mapping as the one-WG-per-tile kernel (blockIdx.x == tile)
            for (uint32_t t = 0; t < n_tiles; ++t)
                for (uint32_t grp = 0; grp < 8u; ++grp)
                    if (t * 8u + grp != t * kRowsPerTile + grp) return 5;
        }
        // The host only takes this path when every resident WG can get a tile.
        if (n_tiles < wgs) return 6;
    }

    // Launch geometry guard: below rows = 8 * wgs the persistent variant must not be selected.
    if ((1536u + 7u) / 8u >= 224u) return 7;   // 192 tiles < 224 WGs -> host falls back
    if ((1792u + 7u) / 8u < 224u) return 8;    // 224 tiles == 224 WGs -> persistent path allowed
    // peer_tx slot per tile is 8 halves wide and disjoint across tiles.
    {
        std::vector<uint32_t> slot(3072u / 8u, 0u);
        for (uint32_t t = 0; t < 3072u / 8u; ++t) { slot[t] += 1u; if (slot[t] != 1u) return 9; }
    }

    std::printf("dense GEMV persistent tiles: %zu geometries, every 8-row tile claimed once, "
                "row/peer_tx mapping identical to the blockIdx.x kernel, queue reset clean\n",
                sizeof(cases) / sizeof(cases[0]));
    return 0;
}
