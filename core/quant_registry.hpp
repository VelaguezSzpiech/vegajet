// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Universal quantization unpacker registry: one small hook per GGUF disk format that converts its blocks into one
// of the three native gfx906 targets.
//
//   VJ_Q8_DP4A    int8 codes + FP16 d per 32 (v_dot4_i32_i8). Float formats and >= 5-bit / affine 4-bit K-quants
//                 are requantized here (cosine >= 0.999 vs the FP32 decode); Q8_0 is copied bit-exactly.
//   VJ_Q4_DOT8    signed nibbles + FP16 d per 32 (v_dot8_i32_i4). Q4_0 transcodes LOSSLESSLY: its codes q - 8 lie in
//                 [-8, 7], exactly a two's-complement nibble, and d is copied. The device image is laid out with the
//                 odd-256B TCC skew (every row stride an odd number of 256 B tiles, so a column of rows spreads over
//                 all 16 TCC channels).
//   VJ_IQ_PACKED  lossless split-plane re-layout (scale / sign / index planes) of codebook formats: IQ4_NL, IQ4_XS,
//                 IQ3_XXS, IQ2_S.
//
// A hook is {disk type, target, lossless flag, convert function}; the default set covers F16, BF16, FP8_E4M3, Q8_0,
// Q6_K, Q5_K, Q4_0, Q4_1, Q4_K, IQ4_NL, IQ4_XS, IQ3_XXS and IQ2_S. Converters validate sizes and alignment before
// calling the bit-exact decoders of core/quant_repacker, so malformed input returns false instead of aborting.
// Which target a tensor actually uses on a given rig is decided by core/rig_planner (crossover law).
#pragma once

#include "core/quant_repacker.hpp"

#include <cstddef>
#include <array>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace vegajet::quant {

struct NativeTensor {
    NativeFormat format = NativeFormat::VJ_Q8_DP4A;
    DiskQuantType source = DiskQuantType::F16;
    uint32_t rows = 0, cols = 0;
    bool lossless = false;
    PackedVjQ8 q8;          // VJ_Q8_DP4A
    PackedVjQ4Dot8 q4;      // VJ_Q4_DOT8 (natural layout, host side)
    PackedVjIq iq;          // VJ_IQ_PACKED
    // VJ_Q4_DOT8 device image: rows x qs_layout.stride_bytes (codes), then rows x ds_layout.stride_bytes (FP16 d).
    TccRowLayout qs_layout, ds_layout;
    std::vector<uint8_t> device_image;
    uint64_t device_bytes() const;   // bytes uploaded to VRAM (padding included)
    uint64_t payload_bytes() const;  // bytes actually read per full pass (padding excluded)
};

using ConvertFn = bool (*)(DiskQuantType disk, const uint8_t* src, uint32_t rows, uint32_t cols, NativeTensor& out);

struct UnpackerHook {
    DiskQuantType disk = DiskQuantType::F16;
    NativeFormat target = NativeFormat::VJ_Q8_DP4A;
    bool lossless = false;
    ConvertFn convert = nullptr;
};

class UnpackerRegistry {
public:
    static UnpackerRegistry& instance();
    // Adds a hook. False if the disk type already has one, has no CPU decoder, or the hook is incomplete.
    bool add(const UnpackerHook& h);
    const UnpackerHook* find(DiskQuantType t) const;
    std::vector<UnpackerHook> hooks() const;  // ascending ggml id
    // Validates (cols block-aligned, src_bytes == rows * row_bytes, rows/cols non-zero) and converts.
    bool convert(DiskQuantType t, const void* src, size_t src_bytes, uint32_t rows, uint32_t cols,
                 NativeTensor& out, std::string* err = nullptr) const;

private:
    UnpackerRegistry();
    mutable std::mutex mu_;
    // Indexed by ggml id: entries never move, so find() pointers stay valid while hooks are added.
    std::array<UnpackerHook, static_cast<size_t>(GgmlType::COUNT)> hooks_{};
    std::array<bool, static_cast<size_t>(GgmlType::COUNT)> present_{};
};

// Generic converters a hook can point at (also usable to build a non-default mapping in tests).
bool convert_requant_q8(DiskQuantType disk, const uint8_t* src, uint32_t rows, uint32_t cols, NativeTensor& out);
bool convert_requant_q4_dot8(DiskQuantType disk, const uint8_t* src, uint32_t rows, uint32_t cols, NativeTensor& out);
bool convert_q4_0_exact_dot8(DiskQuantType disk, const uint8_t* src, uint32_t rows, uint32_t cols, NativeTensor& out);
bool convert_iq_packed(DiskQuantType disk, const uint8_t* src, uint32_t rows, uint32_t cols, NativeTensor& out);

// Lays out q4 (natural) into the odd-256B skewed device image.
void build_q4_dot8_device_image(NativeTensor& t);
// FP32 [rows * cols] of a native tensor; Q4_DOT8 is read back through the skewed device image.
bool dequantize_native(const NativeTensor& t, std::vector<float>& out);

double disk_bits_per_weight(DiskQuantType t);                // block bytes * 8 / block elems (0 if unknown)
double native_bits_per_weight(NativeFormat f, DiskQuantType disk);  // payload bpw (IQ_PACKED = disk bpw)
double cosine_similarity(const float* a, const float* b, size_t n);

} // namespace vegajet::quant
