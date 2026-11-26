// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// v1.2 (D8) host oracle for the IQ2_S grid decode used by the MoE super-kernel.
//
// The kernel decodes a 4-value group of IQ2_S magnitudes either
//   (a) 64-bit table: iq2s_grid[i] byte j = magnitude {8, 25, 43}, sign applied by apply_4signs_i8x4
//   (b) 32-bit table: iq2s_grid32[i] byte j = mag_idx[j] | mag_idx[j + 4] << 2, sign applied by the
//       v_perm_b32 trick in iq2s_perm4 (one constant holds -{8,25,43}, the other +{8,25,43})
// Path (b) is what the super-kernels use, and the D8 variant changes only *where the grid word is
// read from* (LDS staging vs the L1-resident global table). This test proves the two decodes are
// bit-exact - and therefore that the 32-bit packing table itself is right - for all 1024 entries and
// all 4-bit sign nibbles, on the host. The device code cannot run here, so both sides are re-typed
// instruction-for-instruction from ops/kernels/vj_ornith_common.hiph.
#include "core/ggml_iq_tables.hpp"
#include "tests/test_common.hpp"

#include <cstdio>
#include <cstdint>

namespace {

// V_PERM_B32 on gfx9: selector byte bits[2:0] pick a byte, bit 2 = 0 selects the *second* builtin
// operand, bit 2 = 1 selects the first (clang's __builtin_amdgcn_perm(src0, src1, sel) maps to
// v_perm_b32 dst, src0, src1, sel; selectors 0..3/4..7 are src1/src0 bytes). Bits[3] (zero / 0xFF
// fill) are never produced by iq2s_perm4's selector, so they are not modelled.
uint32_t v_perm_b32(uint32_t src0, uint32_t src1, uint32_t sel) {
    uint32_t out = 0;
    for (uint32_t i = 0; i < 4u; ++i) {
        const uint32_t s = (sel >> (8u * i)) & 0xFFu;
        VJ_CHECK(s < 8u); // only magnitude (0..2) + sign (bit 2) selectors are used
        const uint32_t src = (s & 0x4u) ? src0 : src1;
        out |= ((src >> (8u * (s & 0x3u))) & 0xFFu) << (8u * i);
    }
    return out;
}

// ops/kernels/vj_ornith_common.hiph:113
uint32_t apply_4signs_i8x4(uint32_t mag_u32, uint32_t sign4) {
    const uint32_t add  = (sign4 * 0x00204081u) & 0x01010101u;
    const uint32_t xorm = (add << 8) - add;
    return (mag_u32 ^ xorm) + add;
}

// ops/kernels/vj_ornith_common.hiph:121
uint32_t iq2s_perm4(uint32_t mag4_sel, uint32_t sign4) {
    const uint32_t sel = (mag4_sel & 0x03030303u) | ((sign4 * 0x00810204u) & 0x04040404u);
    return v_perm_b32(0x00D5E7F8u, 0x002B1908u, sel);
}

// iq2s grid magnitudes are {8, 25, 43} -> 2-bit codes {0, 1, 2} (code 3 never occurs).
uint32_t mag_code(uint32_t byte) {
    if (byte == 8u)  return 0u;
    if (byte == 25u) return 1u;
    if (byte == 43u) return 2u;
    VJ_CHECK(byte == 8u || byte == 25u || byte == 43u);
    return 3u;
}

uint32_t grid_byte(uint64_t g, uint32_t j) {
    return static_cast<uint32_t>((g >> (8u * j)) & 0xFFu);
}

} // namespace

int main() {
    using namespace vegajet::ggml_tables;
    uint32_t pairs = 0;
    for (uint32_t i = 0; i < 1024u; ++i) {
        const uint64_t g   = iq2s_grid[i];
        const uint32_t g32 = iq2s_grid32[i];
        // 1. the 32-bit packing is the documented one: byte j = code(j) | code(j + 4) << 2.
        for (uint32_t j = 0; j < 4u; ++j) {
            const uint32_t b = (g32 >> (8u * j)) & 0xFFu;
            VJ_CHECK((b & 0x3u) == mag_code(grid_byte(g, j)));
            VJ_CHECK(((b >> 2) & 0x3u) == mag_code(grid_byte(g, j + 4u)));
        }
        // 2. every sign nibble decodes identically on both halves of the entry.
        for (uint32_t sign4 = 0; sign4 < 16u; ++sign4) {
            VJ_CHECK(iq2s_perm4(g32, sign4) ==
                     apply_4signs_i8x4(static_cast<uint32_t>(g), sign4));
            VJ_CHECK(iq2s_perm4(g32 >> 2, sign4) ==
                     apply_4signs_i8x4(static_cast<uint32_t>(g >> 32), sign4));
            pairs += 2u;
        }
    }
    // 1024 entries x 16 sign nibbles x 2 halves.
    VJ_CHECK(pairs == 32768u);
    // Spot check against the literal table bytes: entry 1 = {43, 8, 8, 8, 8, 8, 8, 8} -> first
    // group decodes to {43, 8, 8, 8} and, with sign bit 0 set, {-43, 8, 8, 8}.
    VJ_CHECK(iq2s_perm4(iq2s_grid32[1], 0x0u) == 0x0808082Bu);
    VJ_CHECK(iq2s_perm4(iq2s_grid32[1], 0x1u) == 0x080808D5u);
    std::printf("test_iq2s_perm4: 32-bit grid table + v_perm_b32 == 64-bit grid + SWAR signs "
                "(%u decodes) OK\nPASS\n", pairs);
    return 0;
}
