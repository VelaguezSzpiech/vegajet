// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// TCC helpers that sit on top of quant::select_tcc_layout (core/quant_repacker.hpp):
//  * per-plane layouts for split weight formats: the scale plane needs its own layout (a 4096-column VJ_Q4_A8 dm
//    plane is 512 B = 2 tiles, which camps on 8 of 16 channels when stored at its natural stride);
//  * channel of a (row, logical tile) access under a layout;
//  * per-workgroup k_start stagger: WG w walks the K tiles of its rows starting at tile (w mod tiles), so sibling WGs
//    that read the same rows at the same time start on different channels while each still reads every tile once.
// Header-only, CPU-compilable; kernels compute the same expressions inline.
#pragma once

#include "core/quant_repacker.hpp"

#include <cstdint>

namespace vegajet::quant {

struct SplitPlaneLayouts {
    TccRowLayout qs;  // quantized codes
    TccRowLayout sc;  // per-32 scale (and min) plane
};

// VJ_Q4_A8: u4 codes (cols/2 B) + FP16 (d, m) per 32 weights (cols/8 B).
inline SplitPlaneLayouts select_q4a8_layouts(uint32_t cols) {
    return {select_tcc_layout(cols / 2u), select_tcc_layout(cols / 8u)};
}
// VJ_Q4_DOT8: s4 codes (cols/2 B) + FP16 d per 32 weights (cols/16 B).
inline SplitPlaneLayouts select_q4dot8_layouts(uint32_t cols) {
    return {select_tcc_layout(cols / 2u), select_tcc_layout(cols / 16u)};
}
// VJ_Q8_DP4A: s8 codes (cols B) + FP16 d per 32 weights (cols/16 B).
inline SplitPlaneLayouts select_q8_layouts(uint32_t cols) {
    return {select_tcc_layout(cols), select_tcc_layout(cols / 16u)};
}

inline uint32_t tcc_physical_tile(uint32_t row, uint32_t logical_tile, const TccRowLayout& lay) {
    return lay.mode == TccLayoutMode::XorTileSwizzle ? (logical_tile ^ (row & (kTccChannels - 1u))) : logical_tile;
}

inline uint32_t tcc_channel(uint32_t row, uint32_t logical_tile, const TccRowLayout& lay) {
    const uint64_t addr = static_cast<uint64_t>(row) * lay.stride_bytes +
                          static_cast<uint64_t>(tcc_physical_tile(row, logical_tile, lay)) * kTccTileBytes;
    return static_cast<uint32_t>((addr / kTccTileBytes) & (kTccChannels - 1u));
}

// Logical tiles that carry data (padding tiles are never read).
inline uint32_t tcc_data_tiles(const TccRowLayout& lay) {
    return (lay.raw_row_bytes + kTccTileBytes - 1u) / kTccTileBytes;
}

inline uint32_t stagger_start_tile(uint32_t wg, uint32_t data_tiles) { return data_tiles ? wg % data_tiles : 0u; }
// Logical tile WG `wg` reads at step `iter` (0 <= iter < data_tiles): a rotation, so every tile is read exactly once.
inline uint32_t stagger_tile(uint32_t wg, uint32_t iter, uint32_t data_tiles) {
    return data_tiles ? (stagger_start_tile(wg, data_tiles) + iter) % data_tiles : 0u;
}

} // namespace vegajet::quant
