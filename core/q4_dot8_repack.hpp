// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// VJ_Q4_DOT8: lossless 4-bit re-code of IQ3_XXS / IQ2_S GGUF blocks for M=1 decode on gfx906.
//
// NOTE: this is not vegajet::quant::NativeFormat::VJ_Q4_DOT8 (core/quant_repacker.hpp), which is a lossy
// FP32 requantization. Everything here is an exact re-encoding of the IQ integers; no value is approximated.
//
// ---------------------------------------------------------------------------------------------------
// Exactness (checked against core/ggml_iq_tables.hpp by tests/test_q4_dot8.cpp)
//   IQ3_XXS: every weight is  d(fp16, per 256) * (0.5 + s4) * 0.5 * g,  s4 = 4-bit scale per 32 weights,
//            g = +/- one of the 8 grid magnitudes {4,12,20,28,36,44,52,62}. 7 of them are 8k+4, but 62 is
//            not 60, so an affine (odd-integer) nibble is NOT exact. The 16 signed values are still exactly
//            16 levels, so a 4-bit code is exact as long as decode is a 16-entry lookup.
//   IQ2_S:   every weight is  d(fp16, per 256) * (0.5 + s4) * 0.25 * g,  s4 = 4-bit scale per 16 weights,
//            g = +/- one of {8,25,43}: 6 levels, a 3-bit code is exact.
//
// Decode without LDS or memory LUTs: the lookup is done in registers with v_perm_b32, which picks bytes of a
// 64-bit {src0:src1} pair by selector byte (0..7 = byte, 8..11 = sign of byte 1/3/5/7 replicated, 12 = 0x00,
// >=13 = 0xFF). Four signed int8 weights are produced per v_perm, then fed to v_dot4_i32_i8:
//   IQ2_S   (codes 0..7):  w4 = perm(T_hi, T_lo, c)                                     [1 perm]
//   IQ3_XXS (codes 0..15): w4 = perm(A_hi, A_lo, c) ^ perm(B_hi, B_lo, c & 0x07070707)  [2 perm + xor]
//   For IQ3_XXS the A perm returns a fixed "junk" byte for codes 8..15; B is solved so that the XOR cancels it.
//   The integer dot product is therefore identical to the one computed by the existing IQ kernels (9b/9c),
//   and the float scale is formed with the same expression, so the per-row sums are bit-identical.
//
// Block (32 weights, 16 B = one uint4): word j (0..3), byte b (0..3): low nibble = code of weight 4j+b,
// high nibble = code of weight 16+4j+b (same element order as VJ_Q4_A8 / ggml Q4_0, so the low nibbles pair
// with activation bytes 0..15 and the high nibbles with bytes 16..31).
//   IQ3_XXS code = k | (neg << 3), k = magnitude index into {4,12,20,28,36,44,52,62}.
//   IQ2_S   code = k + 3 * neg,    k = magnitude index into {8,25,43}  (codes 6,7 unused).
//
// TCC anti-camping (choice: odd multiple of 256 B row stride, no swizzle). Weight rows are padded to an odd
// number of 256 B tiles (gcd(16, tiles) == 1 -> the same column tile of 16 consecutive rows hits 16 distinct
// channels). If the scale region fits in that padding for free it is embedded at the end of the row
// (scale_embedded = true), otherwise it is a separate, densely packed plane. Kernels take
// (w_base, w_row_stride_bytes, s_base, s_row_stride_bytes) and do not care which case applies.
#pragma once

#include "core/ggml_types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vegajet {

inline constexpr uint32_t kDot8BlockElems = 32u;   // weights per code block
inline constexpr uint32_t kDot8BlockBytes = 16u;   // bytes per code block
inline constexpr uint32_t kDot8SuperElems = 256u;  // weights per IQ superblock
inline constexpr uint32_t kDot8TileBytes  = 256u;  // TCC channel interleave
inline constexpr uint32_t kIq3xxsBlockBytes = 98u;
inline constexpr uint32_t kIq2sBlockBytes   = 82u;

// ---------------------------------------------------------------------------------------------------
// v_perm_b32 reference model and the register lookup tables
// ---------------------------------------------------------------------------------------------------
// D = perm(src0, src1, sel): bytes 0..3 of {src0:src1} come from src1, bytes 4..7 from src0.
constexpr uint32_t dot8_perm_b32(uint32_t src0, uint32_t src1, uint32_t sel) {
    const uint64_t in = (static_cast<uint64_t>(src0) << 32) | src1;
    uint32_t out = 0;
    for (uint32_t i = 0; i < 4u; ++i) {
        const uint32_t s = (sel >> (8u * i)) & 0xFFu;
        uint32_t b = 0u;
        if (s >= 13u)      b = 0xFFu;
        else if (s == 12u) b = 0x00u;
        else if (s >= 8u)  b = ((in >> (8u * ((s - 8u) * 2u + 1u) + 7u)) & 1u) ? 0xFFu : 0x00u;
        else               b = static_cast<uint32_t>((in >> (8u * s)) & 0xFFu);
        out |= b << (8u * i);
    }
    return out;
}

inline constexpr int8_t kDot8Iq3Mag[8] = {4, 12, 20, 28, 36, 44, 52, 62};
inline constexpr int8_t kDot8Iq2Mag[3] = {8, 25, 43};

// Signed value of each 4-bit IQ3_XXS code / 3-bit IQ2_S code.
constexpr int8_t dot8_iq3_level(uint32_t c) {
    return (c & 8u) ? static_cast<int8_t>(-kDot8Iq3Mag[c & 7u]) : kDot8Iq3Mag[c & 7u];
}
constexpr int8_t dot8_iq2_level(uint32_t c) {
    return c < 3u ? kDot8Iq2Mag[c] : (c < 6u ? static_cast<int8_t>(-kDot8Iq2Mag[c - 3u]) : int8_t{0});
}

struct Dot8Iq3PermTables { uint32_t a_lo = 0, a_hi = 0, b_lo = 0, b_hi = 0; };

// Solves A, B so that perm(A_hi,A_lo,c) ^ perm(B_hi,B_lo,c&7) == dot8_iq3_level(c) for all 16 codes.
// For c < 8: A[c] ^ B[c] = L[c].  For c >= 8: J_A(c) ^ B[c-8] = L[c], where J_A(8..11) is the sign of
// A byte 1/3/5/7, J_A(12) = 0 and J_A(13..15) = 0xFF. The A bytes whose sign feeds J_A are solved first.
constexpr Dot8Iq3PermTables dot8_solve_iq3_tables() {
    uint8_t A[8] = {}, B[8] = {};
    auto solve = [&](uint32_t k, uint8_t junk) {
        B[k] = static_cast<uint8_t>(static_cast<uint8_t>(dot8_iq3_level(8u + k)) ^ junk);
        A[k] = static_cast<uint8_t>(static_cast<uint8_t>(dot8_iq3_level(k)) ^ B[k]);
    };
    auto sgn = [&](uint32_t k) -> uint8_t { return (A[k] & 0x80u) ? 0xFFu : 0x00u; };
    solve(7u, 0xFFu); solve(6u, 0xFFu); solve(5u, 0xFFu); solve(4u, 0x00u);
    solve(3u, sgn(7u)); solve(2u, sgn(5u)); solve(1u, sgn(3u)); solve(0u, sgn(1u));
    Dot8Iq3PermTables t;
    for (uint32_t i = 0; i < 4u; ++i) {
        t.a_lo |= static_cast<uint32_t>(A[i]) << (8u * i);
        t.a_hi |= static_cast<uint32_t>(A[i + 4u]) << (8u * i);
        t.b_lo |= static_cast<uint32_t>(B[i]) << (8u * i);
        t.b_hi |= static_cast<uint32_t>(B[i + 4u]) << (8u * i);
    }
    return t;
}
inline constexpr Dot8Iq3PermTables kDot8Iq3Perm = dot8_solve_iq3_tables();

constexpr uint32_t dot8_iq2_table_word(uint32_t base) {
    uint32_t w = 0;
    for (uint32_t i = 0; i < 4u; ++i) w |= static_cast<uint32_t>(static_cast<uint8_t>(dot8_iq2_level(base + i))) << (8u * i);
    return w;
}
inline constexpr uint32_t kDot8Iq2PermLo = dot8_iq2_table_word(0u);
inline constexpr uint32_t kDot8Iq2PermHi = dot8_iq2_table_word(4u);

// Device-identical decode of one packed word group (4 codes in bytes) -> 4 signed int8 in a uint32.
constexpr uint32_t dot8_decode4_iq3(uint32_t nib_bytes /* (w >> 0|4) & 0x0F0F0F0F */) {
    return dot8_perm_b32(kDot8Iq3Perm.a_hi, kDot8Iq3Perm.a_lo, nib_bytes) ^
           dot8_perm_b32(kDot8Iq3Perm.b_hi, kDot8Iq3Perm.b_lo, nib_bytes & 0x07070707u);
}
constexpr uint32_t dot8_decode4_iq2(uint32_t nib_bytes) {
    return dot8_perm_b32(kDot8Iq2PermHi, kDot8Iq2PermLo, nib_bytes & 0x07070707u);
}

// Exact bits per weight actually streamed per token (codes + scales, padding excluded).
inline constexpr double kDot8Iq3ReadBpw = 4.25;    // 16 B / 32 w + (4-bit s per 32 + f32 d per 256)
inline constexpr double kDot8Iq2ReadBpw = 4.3125;  // 16 B / 32 w + (4-bit s per 16 + fp16 d per 256)

// ---------------------------------------------------------------------------------------------------
// Row layouts
// ---------------------------------------------------------------------------------------------------
// Smallest odd multiple of 256 B that holds raw_bytes.
uint32_t dot8_odd256_stride(uint32_t raw_bytes);

// Common physical row layout for one packed tensor.
struct Dot8RowLayout {
    uint64_t rows = 0;
    uint32_t cols = 0;                 // multiple of 256
    uint32_t w_raw_row_bytes = 0;      // code bytes per row
    uint32_t s_raw_row_bytes = 0;      // scale bytes per row
    uint32_t w_row_stride_bytes = 0;   // odd multiple of 256
    bool     scale_embedded = false;   // scale region lives in the weight row's TCC padding
    uint32_t s_offset_bytes = 0;       // embedded: offset of the scale region inside a weight row
    uint32_t s_row_stride_bytes = 0;   // embedded: == w_row_stride_bytes; separate: == s_raw_row_bytes
    uint64_t w_plane_bytes() const { return rows * w_row_stride_bytes; }
    uint64_t s_plane_bytes() const { return scale_embedded ? 0ull : rows * s_row_stride_bytes; }
    uint64_t vram_bytes() const { return w_plane_bytes() + s_plane_bytes(); }
    uint64_t read_bytes_per_row() const { return static_cast<uint64_t>(w_raw_row_bytes) + s_raw_row_bytes; }
};

// IQ3_XXS-sourced tensor, one code block per 32 weights.
//   scale region per row: per superblock 8 B {u32 sub-scale nibbles (block j of the superblock at bits
//   4j..4j+3), f32 d (exact fp16 -> f32)}. Device: lane for block b loads the uint2 at superblock b >> 3.
Dot8RowLayout dot8_iq3_layout(uint64_t rows, uint32_t cols);
// Toggle-only Down layout: {u32 sub-scale, f32 superblock d} per 32 weights.
Dot8RowLayout dot8_iq3_asym_layout(uint64_t rows, uint32_t cols);
bool dot8_repack_iq3xxs_asym_rows(const uint8_t* src, uint64_t src_stride, uint32_t col_begin,
                                const Dot8RowLayout& lay, uint64_t row_begin, uint64_t row_end,
                                uint8_t* w, uint8_t* s, std::string* err);

// Merged Gate+Up pair of IQ2_S-sourced tensors with the same shape (row r of gate and row r of up together).
//   codes per row: per block ib {gate uint4, up uint4} = 32 B.
//   scale region per row: [nblk x u16 (gate scales byte | up scales byte << 8)] [nsb x u32 (gate d fp16 |
//   up d fp16 << 16)]. The scales byte keeps the IQ2_S layout: low nibble -> weights 0..15, high -> 16..31.
Dot8RowLayout dot8_gu_iq2s_layout(uint64_t rows, uint32_t cols);

// Lossless 3-bit bit-sliced merged Gate+Up IQ2_S layout (VJ_Q3_IQ2S):
//   w_plane per row: [nblk x uint4 2-bit mags (gate lo, gate hi, up lo, up hi)] [nblk x uint2 1-bit signs (gate, up)]
//   For D=3072 (nblk=96): 1536 + 768 = 2304 B = 9 x 256 B (exact odd-256B, 0 padding bytes!).
//   s_plane per row: [nblk x u16 (gs | us<<8)] [nsb x u32 (gd_f16 | ud_f16<<16)] padded to odd-256B (256 B for D=3072).
Dot8RowLayout q3_gu_iq2s_layout(uint64_t rows, uint32_t cols);
bool q3_repack_gu_iq2s_rows(const uint8_t* src_gate, const uint8_t* src_up, uint64_t src_row_stride_bytes,
                            uint32_t sb_begin, const Dot8RowLayout& lay, uint64_t row_begin, uint64_t row_end,
                            uint8_t* w_plane, uint8_t* s_plane, std::string* err = nullptr);
void q3_decode_gu_row_int(const uint8_t* w_row, uint32_t cols, int8_t* out_gate, int8_t* out_up);

// ---------------------------------------------------------------------------------------------------
// Repackers (model agnostic, thread safe on disjoint row ranges)
// ---------------------------------------------------------------------------------------------------
// Row r (lay row index) reads the IQ superblocks starting at  src + (r * src_row_stride_bytes) +
// sb_begin * block_bytes  (sb_begin selects a TP column slice). Writes rows [row_begin, row_end) of the
// weight plane (w_plane, w_row_stride_bytes) and the scale region (s_plane, or w_plane when embedded).
// Padding bytes are written as zero. Returns false (and sets *err) on bad shapes or out-of-table indices.
bool dot8_repack_iq3xxs_rows(const uint8_t* src, uint64_t src_row_stride_bytes, uint32_t sb_begin,
                             const Dot8RowLayout& lay, uint64_t row_begin, uint64_t row_end,
                             uint8_t* w_plane, uint8_t* s_plane, std::string* err = nullptr);

bool dot8_repack_gu_iq2s_rows(const uint8_t* src_gate, const uint8_t* src_up, uint64_t src_row_stride_bytes,
                              uint32_t sb_begin, const Dot8RowLayout& lay, uint64_t row_begin, uint64_t row_end,
                              uint8_t* w_plane, uint8_t* s_plane, std::string* err = nullptr);

// Single 32-weight block encoders (exposed for tests): codes for weights 0..31 -> packed uint4 bytes.
void dot8_pack_codes(const uint8_t codes[32], uint8_t out16[16]);
void dot8_unpack_codes(const uint8_t in16[16], uint8_t codes[32]);
// IQ3_XXS sub-block ib32 (0..7) of one 98 B superblock -> 32 codes + its 4-bit scale. False on bad grid byte.
bool dot8_encode_iq3xxs_sub(const uint8_t* blk98, uint32_t ib32, uint8_t codes[32], uint8_t* scale4);
// IQ2_S sub-block ib32 of one 82 B superblock -> 32 codes + its scales byte (lo: w0..15, hi: w16..31).
bool dot8_encode_iq2s_sub(const uint8_t* blk82, uint32_t ib32, uint8_t codes[32], uint8_t* scales8);

// CPU reference of the device decode: row -> int8 weights (cols) through the packed words, perm model included.
void dot8_decode_row_iq3_int(const uint8_t* w_row, uint32_t cols, int8_t* out);
void dot8_decode_row_iq2_int(const uint8_t* w_codes_row /* one tensor's blocks, 16 B stride */, uint32_t cols,
                             uint32_t block_stride_bytes, int8_t* out);

// ---------------------------------------------------------------------------------------------------
// Promotion planner
// ---------------------------------------------------------------------------------------------------
enum class Dot8Role : uint32_t { MoeDown = 0, MoeGateUp = 1, Other = 2 };
enum class Dot8Target : uint32_t { VJ_Q4_DOT8 = 0, VJ_Q4_A8 = 1 };

struct Dot8ClassSpec {
    std::string name;
    GgmlType src_type = GgmlType::IQ3_XXS;   // IQ3_XXS or IQ2_S
    Dot8Role role = Dot8Role::MoeDown;
    Dot8Target target = Dot8Target::VJ_Q4_DOT8;
    std::vector<uint32_t> layers;            // candidate layers, in promotion order
    uint64_t old_vram_bytes_per_layer = 0;   // per GPU
    uint64_t new_vram_bytes_per_layer = 0;   // per GPU (incl. padding)
    uint64_t old_read_bytes_per_layer = 0;   // bytes streamed per token per GPU (same element count)
    uint64_t new_read_bytes_per_layer = 0;
    double   bw_old_gbs = 0.0;               // measured per GPU, same shape, M = batch_m
    double   bw_new_gbs = 0.0;
    bool     frees_old = true;               // old copy released after promotion (else the full new size is charged)
    uint32_t batch_m = 1;
};

struct Dot8Promotion {
    size_t   class_index = 0;
    std::string name;
    uint32_t layer = 0;
    int64_t  delta_bytes = 0;                // charged to the pool (can be negative)
    uint64_t cum_bytes = 0;                  // pool use after this promotion
    double   us_saved = 0.0;                 // per token per GPU, from the measured bandwidths
};

struct Dot8ClassVerdict {
    std::string name;
    uint32_t priority = 0;                   // 0 = first
    bool     eligible = false;
    std::string reason;
    double   bw_ratio = 0.0;                 // bw_new / bw_old
    double   bpw_ratio = 0.0;                // new_read / old_read
    uint32_t layers_promoted = 0;
};

struct Dot8PromotionPlan {
    std::vector<Dot8Promotion> promotions;   // in the order they must be applied
    std::vector<Dot8ClassVerdict> verdicts;  // one per input class, input order
    uint64_t pool_bytes = 0;
    uint64_t used_bytes = 0;
    double   us_saved_total = 0.0;
};

// Priority: IQ3_XXS MoeDown (0) > IQ3_XXS MoeGateUp (1) > other IQ3_XXS (2) > IQ2_S (3); ties keep input order.
// A class is eligible only if  bw_new/bw_old > bpw_new/bpw_old  (strict), both bandwidths are measured, and it
// is not an IQ2_S -> VJ_Q4_A8 promotion at M=1 (banned regression). Layers are charged
// new - (frees_old ? old : 0) bytes against pool_bytes_per_gpu. With strict_priority (default) a class is only
// considered once every eligible higher-priority class is fully promoted.
uint32_t dot8_class_priority(const Dot8ClassSpec& c);
Dot8PromotionPlan plan_dot8_promotion(const std::vector<Dot8ClassSpec>& classes, uint64_t pool_bytes_per_gpu,
                                      bool strict_priority = true);

} // namespace vegajet
