// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// CPU oracle: row scheduling, cross-WG packet ownership, fixed-order expert reduction,
// and critical endpoints with a deliberately slow nonzero WG. No device access.
#include "tests/test_common.hpp"
#include <algorithm>
#include <array>
#include <cmath>

int main() {
    std::array<float, 3072> reference{}, scheduled{};
    std::array<unsigned, 3072> visits{}, packets{};
    auto row_value = [](unsigned row) {
        std::array<float, 16> lanes{};
        for (unsigned lane = 0; lane < 16; ++lane) {
            for (unsigned slot = 0; slot < 8; ++slot) {
                const float ew = float(slot + 1) / 37.0f;
                const float gdb = float((row % 31) + 1) / 19.0f;
                const float xd = float(lane + 1) / 127.0f;
                const int dp = int((row * 13 + lane * 7 + slot * 11) % 201) - 100;
                lanes[lane] = std::fma(ew * gdb * xd, float(dp), lanes[lane]);
            }
        }
        // Same fixed butterfly reduction for each row irrespective of WG ownership.
        for (unsigned mask = 8; mask; mask >>= 1) {
            auto prev = lanes;
            for (unsigned lane = 0; lane < 16; ++lane) lanes[lane] += prev[lane ^ mask];
        }
        return lanes[0] + float(row % 17) / 23.0f;
    };
    for (unsigned row = 0; row < 3072; ++row) reference[row] = row_value(row);
    // Execute WGs in reverse order: no packet consumer may depend on producer order.
    for (unsigned wg = 240; wg-- > 0;) {
        unsigned n = wg < 192 ? 13 : 12;
        unsigned base = wg < 192 ? wg * 13 : 2496 + (wg - 192) * 12;
        for (unsigned group = 0; group < 16; ++group) if (group < n) {
            VJ_CHECK(base + group < 3072);
            ++visits[base + group];
            scheduled[base + group] = row_value(base + group);
        }
    }
    // Full producer barrier, followed by the unchanged 16/8-row packet owners.
    for (unsigned wg = 0; wg < 240; ++wg) {
        unsigned n = wg < 144 ? 16 : 8;
        unsigned base = wg < 144 ? wg * 16 : 2304 + (wg - 144) * 8;
        VJ_CHECK(base % 8 == 0);
        for (unsigned packet = 0; packet < n / 8; ++packet)
            for (unsigned j = 0; j < 8; ++j) {
                unsigned row = base + packet * 8 + j;
                ++packets[row];
                VJ_CHECK(std::memcmp(&reference[row], &scheduled[row], sizeof(float)) == 0);
            }
    }
    for (unsigned row = 0; row < 3072; ++row) VJ_CHECK(visits[row] == 1 && packets[row] == 1);
    uint64_t ends[2] = {};
    for (unsigned wg = 0; wg < 240; ++wg) {
        ends[0] = std::max(ends[0], uint64_t(wg == 239 ? 500 : 200));
        ends[1] = std::max(ends[1], uint64_t(wg == 238 ? 900 : 700));
    }
    VJ_CHECK(ends[0] - 100 == 400 && ends[1] - ends[0] == 400);
    // 96 records fit only in the opt-in 16 KiB allocation, above all barrier slots.
    VJ_CHECK(2048 + 96 * 17 * sizeof(uint64_t) <= 16384);
    return 0;
}
