// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Host-side matching packer/reader for core/router_s240_layout.hpp.
//
//   * pack_plane_rows() writes exactly the bytes the 240-WG kernel reads: logical 256-B tile kt of row r
//     goes to physical tile kt ^ (r & xor_mask). A partial final tile copies its data bytes and zero-fills
//     the remainder; padding tiles created by the power-of-two XOR capacity stay zero and are never read.
//   * plane_byte() / plane_block() are the CPU mirror of the kernel's uint4 indexing, built from the same
//     inline block_index()/plane_byte_offset() helpers the device code uses.
//   * unpack_plane_rows() is the inverse, used to prove byte-exact round trips.
// Host-only (uses std::vector); includes no HIP headers.
#pragma once

#include "core/router_s240_layout.hpp"

#include <cstring>
#include <vector>

namespace vegajet {
namespace router_s240 {

inline std::vector<uint8_t> pack_plane_rows(const uint8_t* src, uint32_t rows, const PlaneLayout& l,
                                            uint32_t src_row_stride_bytes = 0) {
    std::vector<uint8_t> out(static_cast<size_t>(rows) * l.stride_bytes, 0u);
    if (src == nullptr || l.raw_row_bytes == 0u) return out;
    const uint32_t src_stride = src_row_stride_bytes ? src_row_stride_bytes : l.raw_row_bytes;
    for (uint32_t r = 0; r < rows; ++r) {
        const uint8_t* s = src + static_cast<size_t>(r) * src_stride;
        uint8_t* d = out.data() + static_cast<size_t>(r) * l.stride_bytes;
        for (uint32_t kt = 0; kt < l.data_tiles; ++kt) {
            const uint32_t off = kt * kTileBytes;
            const uint32_t n = (l.raw_row_bytes - off < kTileBytes) ? (l.raw_row_bytes - off) : kTileBytes;
            std::memcpy(d + static_cast<size_t>(phys_tile(l, r, kt)) * kTileBytes, s + off, n);
        }
    }
    return out;
}

inline std::vector<uint8_t> unpack_plane_rows(const uint8_t* packed, uint32_t rows, const PlaneLayout& l) {
    std::vector<uint8_t> out(static_cast<size_t>(rows) * l.raw_row_bytes);
    if (packed == nullptr || l.raw_row_bytes == 0u) return out;
    for (uint32_t r = 0; r < rows; ++r) {
        const uint8_t* s = packed + static_cast<size_t>(r) * l.stride_bytes;
        uint8_t* d = out.data() + static_cast<size_t>(r) * l.raw_row_bytes;
        for (uint32_t kt = 0; kt < l.data_tiles; ++kt) {
            const uint32_t off = kt * kTileBytes;
            const uint32_t n = (l.raw_row_bytes - off < kTileBytes) ? (l.raw_row_bytes - off) : kTileBytes;
            std::memcpy(d + off, s + static_cast<size_t>(phys_tile(l, r, kt)) * kTileBytes, n);
        }
    }
    return out;
}

// Logical byte `off` of `row` inside a packed plane (CPU mirror of the kernel's address arithmetic).
inline const uint8_t* plane_byte(const uint8_t* packed, const PlaneLayout& l, uint32_t row, uint32_t off) {
    return packed + plane_byte_offset(l, row, off);
}

// Logical 16-B block `idx` of `row` (the kernel reads exactly this uint4).
inline const uint8_t* plane_block(const uint8_t* packed, const PlaneLayout& l, uint32_t row, uint32_t idx) {
    const uint32_t tile = idx / kTileU32x4;
    const uint32_t in = idx % kTileU32x4;
    return packed + static_cast<uint64_t>(row) * l.stride_bytes +
           static_cast<uint64_t>(phys_tile(l, row, tile)) * kTileBytes + static_cast<uint64_t>(in) * 16u;
}

} // namespace router_s240
} // namespace vegajet
