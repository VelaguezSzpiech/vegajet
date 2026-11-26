// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Universal GGUF -> gfx906 repacker: "many formats on disk, three native formats in VRAM".
//
//  * Pluggable CPU block decoders (ggml type id -> decode to FP32), bit-exact with the ggml reference.
//  * Packers into VJ_Q8_DP4A, VJ_Q4_DOT8 (optional FWHT-64 rotation) and VJ_IQ_PACKED (lossless split planes).
//  * Roofline 0/1 VRAM knapsack that scores upgrades by ms saved per extra GiB and rejects negative ROI.
//  * TCC L2 anti-camping row layout (static XOR tile swizzle or odd-256B row padding).
#pragma once

#include "core/ggml_types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace vegajet::quant {

// =================================================================================================
// Formats
// =================================================================================================
enum class NativeFormat : uint32_t {
    VJ_Q8_DP4A = 0,    // ~8.5 bpw: int8 qs + FP16 scale per 32 (v_dot4_i32_i8), rows 16-byte aligned
    VJ_Q4_DOT8 = 1,    // ~4.5 bpw: 8 signed nibbles per uint32 + FP16 scale per 32 (v_dot8_i32_i4)
    VJ_IQ_PACKED = 2,  // disk bpw: lossless split-plane (scale / sign / index) re-layout of sub-4-bit blocks
    VJ_Q2_DOT8 = 4,
    VJ_Q3_DOT8 = 5,
    VJ_Q4_A8 = 3,      // ~5.0 bpw: unsigned 4-bit nibbles (low=0..15, high=16..31) + packed FP16 (d,m) per 32, TCC anti-camping on qs & dm
};
const char* native_format_name(NativeFormat f);

// Disk types are ggml type ids. Note: the llama.cpp "IQ3_XS" file type stores IQ3_S (id 21) tensors.
using DiskQuantType = GgmlType;

enum class TensorRole : uint32_t {
    DenseProj = 0,     // attention / GDN projections, dense FFN
    SharedExpert = 1,  // MoE shared expert
    VocabHead = 2,     // output projection / token embedding
    MtpDraft = 3,      // speculative draft / MTP layer
    RoutedMoE = 4,     // routed expert trunk (read rate = hit_rate)
    Other = 5,         // norms, gates, small vectors (kept F32/F16 on device)
};

// =================================================================================================
// Scalar helpers
// =================================================================================================
float fp16_to_f32(uint16_t h);
uint16_t f32_to_fp16(float f);
float bf16_to_f32(uint16_t b);
float fp8_e4m3_to_f32(uint8_t b);
void fwht64_inplace(float* x64);  // orthonormal 64-point Walsh-Hadamard (self-inverse)

// =================================================================================================
// Pluggable decoder registry
// =================================================================================================
// Decodes `n_elems` (multiple of block_elems) consecutive elements of `type` from `src` into `dst`.
using BlockDecodeFn = void (*)(const uint8_t* src, float* dst, uint64_t n_elems);

struct FormatInfo {
    uint32_t block_elems = 1;
    uint32_t block_bytes = 4;
    BlockDecodeFn decode = nullptr;
    const char* name = "UNKNOWN";
};

class DecoderRegistry {
public:
    static DecoderRegistry& instance();
    void register_format(DiskQuantType type, const FormatInfo& info);
    const FormatInfo* find(DiskQuantType type) const;
    bool supports(DiskQuantType type) const { return find(type) != nullptr; }
    std::vector<DiskQuantType> types() const;
    // Bytes of one row of `cols` elements (0 if unsupported or not block aligned).
    uint64_t row_bytes(DiskQuantType type, uint64_t cols) const;
    bool decode_row_f32(DiskQuantType type, const uint8_t* src, float* dst, uint64_t cols) const;

private:
    DecoderRegistry();
    std::unordered_map<uint32_t, FormatInfo> formats_;
};

// =================================================================================================
// Native packed tensors
// =================================================================================================
struct PackedVjQ8 {
    uint32_t rows = 0, cols = 0;   // cols multiple of 32
    std::vector<int8_t> qs;        // [rows * cols]
    std::vector<uint16_t> ds;      // [rows * cols/32] FP16
    size_t total_bytes() const { return qs.size() + ds.size() * 2; }
};

struct PackedVjQ4Dot8 {
    uint32_t rows = 0, cols = 0;   // cols multiple of 32 (64 when fwht64)
    bool fwht64 = false;
    std::vector<uint32_t> qs;      // [rows * cols/8], nibble k of word w = element 8w+k, range [-7, 7]
    std::vector<uint16_t> ds;      // [rows * cols/32] FP16
    size_t total_bytes() const { return qs.size() * 4 + ds.size() * 2; }
};

struct PackedVjQ4A8 {
    uint32_t rows = 0, cols = 0;          // cols multiple of 32
    uint32_t qs_row_stride_bytes = 0;     // odd-256B TCC stride for qs plane (bytes)
    uint32_t dm_row_stride_u32 = 0;       // odd-256B TCC stride for dm plane (uint32_t elems, bytes = *4)
    std::vector<uint8_t>  qs;             // [rows * qs_row_stride_bytes], low=0..15, high=16..31
    std::vector<uint32_t> dm;             // [rows * dm_row_stride_u32], fp16(d) | (fp16(m) << 16)
    size_t total_bytes() const { return qs.size() + dm.size() * 4; }
    size_t real_read_bytes() const { return static_cast<size_t>(rows) * (cols / 2u + (cols / 32u) * 4u); }
};

// Zero-LUT affine formats. Row-major blocks of 32; Q2: 2 words, Q3: 3 words.
// i -> low word i/16, bits 4*((i%16)/2)+2*(i%2).
// Q3 word 2: bit 4*((i%16)/2)+2*(i/16)+(i%2). Each stripe
// occupies one bit position across the eight nibbles (not a contiguous byte).
// Four zero guard words at end permit 16-byte vector loads of the final block.
// dm = fp16(scale) | fp16(zero_point)<<16, per 64/128 weights;
// dequant = scale*(unsigned_code-zero_point). Zero point is fractional.
struct PackedVjSub4 {
    NativeFormat format = NativeFormat::VJ_Q2_DOT8;
    uint32_t rows = 0, cols = 0, group = 64;
    bool tcc_xor = false; // XOR layout expands metadata to one word per block32
    std::vector<uint32_t> qs, dm;
    size_t total_bytes() const { return (qs.size()+dm.size())*4; }
};
PackedVjSub4 pack_f32_to_vj_sub4(const float*, uint32_t rows, uint32_t cols,
                               NativeFormat, uint32_t group = 64);
PackedVjSub4 pack_f16_to_vj_sub4(const uint16_t*, uint32_t rows, uint32_t cols,
                               NativeFormat, uint32_t group = 64);
uint32_t sub4_code(const PackedVjSub4&, size_t index);
void dequantize_vj_sub4(const PackedVjSub4&, float*);
// x: consecutive signed int4 codes represented as int8, scale per block32.
// Integer dot and sum, then affine epilogue. Output [M,rows].
void gemv_vj_sub4(const PackedVjSub4&, const int8_t* x, const float* scales,
                  float* y, uint32_t M = 1);
// Physical block mapping is involutive; requires complete 16-block tiles.
PackedVjQ4A8 repack_tcc_xor(const PackedVjQ4A8&);
PackedVjSub4 repack_tcc_xor(const PackedVjSub4&);
uint32_t sub4_dot8_bits(); // VEGAJET_SUB4_DOT8=q2|q3, otherwise off
bool tcc_xor_enabled(); // VEGAJET_TCC_XOR=1
enum class MiloStorage { Fp16, Int3 };
struct MiloResidual {
    uint32_t rows = 0, cols = 0, rank = 0;
    MiloStorage storage = MiloStorage::Fp16;
    std::vector<uint16_t> u, v; // row-major [rows,rank], [rank,cols], fp16
    std::vector<uint32_t> u3, v3; // contiguous signed INT3, per-row scales
    std::vector<float> us, vs;
    size_t total_bytes() const { return (u.size()+v.size())*2 + (u3.size()+v3.size()+us.size()+vs.size())*4; }
};
bool milo_enabled(); // VEGAJET_MILO=1, default off; explicit offline opt-in
MiloResidual fit_milo(const float* original, const PackedVjSub4&, uint32_t rank = 16,
                      uint32_t iterations = 24, MiloStorage storage = MiloStorage::Fp16);
float milo_factor(const MiloResidual&, bool u, size_t index);
void apply_milo(const MiloResidual&, const float* x, float* y); // adds U@(V@x)

enum class IqPlane : uint8_t { Scale = 0, Sign = 1, Index = 2 };

struct IqSegment { uint16_t offset; uint16_t bytes; IqPlane plane; };

struct PackedVjIq {
    DiskQuantType orig_type = DiskQuantType::IQ3_XXS;
    uint32_t rows = 0, cols = 0;
    std::vector<uint8_t> scale_plane, sign_plane, index_plane;
    size_t total_bytes() const { return scale_plane.size() + sign_plane.size() + index_plane.size(); }
};

// Byte segments of one disk block assigned to planes (covers the block exactly once).
std::vector<IqSegment> iq_block_segments(DiskQuantType t);

PackedVjQ8 pack_f32_to_vj_q8(const float* src, uint32_t rows, uint32_t cols);
PackedVjQ4Dot8 pack_f32_to_vj_q4_dot8(const float* src, uint32_t rows, uint32_t cols, bool fwht64 = false);
PackedVjQ4A8 pack_f32_to_vj_q4_a8(const float* src, uint32_t rows, uint32_t cols, bool tcc_pad_dm = true);
// Disk rows -> native. Q8_0 -> Q8 and Q2_0 -> Q8/Q4 are lossless fast paths (no requantization).
PackedVjQ8 repack_to_vj_q8(DiskQuantType t, const uint8_t* src, uint32_t rows, uint32_t cols);
PackedVjQ4Dot8 repack_to_vj_q4_dot8(DiskQuantType t, const uint8_t* src, uint32_t rows, uint32_t cols,
                                    bool fwht64 = false);
PackedVjQ4A8 repack_to_vj_q4_a8(DiskQuantType t, const uint8_t* src, uint32_t rows, uint32_t cols,
                                bool tcc_pad_dm = true);
PackedVjIq repack_to_vj_iq(DiskQuantType t, const uint8_t* src, uint32_t rows, uint32_t cols);

void dequantize_vj_q8(const PackedVjQ8& p, float* out);
void dequantize_vj_q4_dot8(const PackedVjQ4Dot8& p, float* out);
void dequantize_vj_q4_a8(const PackedVjQ4A8& p, float* out);
void dequantize_vj_iq(const PackedVjIq& p, float* out);
std::vector<uint8_t> unpack_vj_iq_to_disk(const PackedVjIq& p);  // exact inverse of repack_to_vj_iq

// =================================================================================================
// Hardware roofline + 0/1 VRAM knapsack
// =================================================================================================
struct GpuRooflineInfo {
    std::string id;              // PCI BDF or free-form device label
    std::string unique_id;       // 16-char lowercase hex hardware unique_id (optional)
    uint32_t n_cu = 60;
    double sclk_ghz = 1.725;
    double mclk_mhz = 1000.0;
    double bw_hbm_gbs = 800.0;   // sustained streaming bandwidth
    double power_cap_w = 225.0;
    double eta = 0.55;           // sustained VALU issue efficiency
    // I* = (n_cu * 64 * f_sclk * eta) / BW_hbm  [VALU instructions per byte at the roofline ridge],
    // divided by the batch reuse factor (tokens verified per weight read).
    double i_star(double reuse_factor = 1.0) const;
};

struct HardwareRooflineProfile {
    std::vector<GpuRooflineInfo> gpus;
    double min_i_star(double reuse_factor = 1.0) const;
    static std::string default_cache_path();  // $HOME/.config/vegajet/hw_profile.json
    static bool load_json(const std::string& path, HardwareRooflineProfile& out);
    static bool save_json(const std::string& path, const HardwareRooflineProfile& p);
};

struct TensorPlanSpec {
    std::string name;
    DiskQuantType disk_type = DiskQuantType::F16;
    TensorRole role = TensorRole::DenseProj;
    uint64_t rows = 0, cols = 0, experts = 1;
    double hit_rate = 1.0;   // fraction of the tensor read per token (routed MoE: k/n_expert)
    bool prefer_w4a4 = false;
};

struct TensorPlanDecision {
    std::string name;
    NativeFormat target_format = NativeFormat::VJ_IQ_PACKED;
    bool use_fwht64 = false;
    uint64_t estimated_vram_bytes = 0;
    double expected_ms_saved = 0.0;
    double roi_ms_per_gib = 0.0;
    bool rejected_negative_roi = false;
};

class VramBudgetPlanner {
public:
    static uint64_t estimate_bytes(NativeFormat fmt, DiskQuantType disk, uint64_t rows, uint64_t cols,
                                   uint64_t experts = 1);
    // Decode cost model: VALU instructions per weight byte for each disk / native format.
    static double format_inst_per_byte(DiskQuantType disk);
    static double native_inst_per_byte(NativeFormat fmt, DiskQuantType disk, bool w4a4 = false);
    // ms to stream `bytes` at bw, inflated by max(1, ipb / I*) when decode is VALU-bound, times hit rate.
    static double estimate_read_ms(uint64_t bytes, double inst_per_byte, double bw_gbs, double i_star,
                                   double hit_rate);
    // Pass 1: minimal-footprint baseline (8-bit/float disk -> Q8, Q4_0/Q4_1/Q4_K -> Q4, everything else -> IQ_PACKED).
    // Pass 2: one best positive-ROI upgrade candidate per tensor; negative ROI candidates are flagged.
    // Pass 3: 0/1 knapsack by ROI density (ms saved / extra GiB) within vram_budget - reserve.
    static std::vector<TensorPlanDecision> plan_roofline_knapsack(const std::vector<TensorPlanSpec>& specs,
                                                                  uint64_t vram_budget_bytes,
                                                                  uint64_t reserve_bytes,
                                                                  const HardwareRooflineProfile& hw,
                                                                  double batch_reuse_factor = 1.0);
};

// =================================================================================================
// TCC L2 anti-camping layout (16 channels x 256 B interleave)
// =================================================================================================
enum class TccLayoutMode : uint32_t {
    NaturalOdd256 = 0,   // row stride already an odd number of 256 B tiles
    XorTileSwizzle = 1,  // >= 16 tiles and a multiple of 16: phys_tile = k_tile ^ (row & 15), zero extra VRAM
    PadOdd256 = 2,       // otherwise pad the stride to the next odd tile count (gcd(16, tiles) == 1)
};

struct TccRowLayout {
    TccLayoutMode mode = TccLayoutMode::NaturalOdd256;
    uint32_t raw_row_bytes = 0;
    uint32_t stride_bytes = 0;
    uint32_t tiles_per_row = 0;
};

constexpr uint32_t kTccChannels = 16u;
constexpr uint32_t kTccTileBytes = 256u;

TccRowLayout select_tcc_layout(uint32_t raw_row_bytes);
// Odd-256B padded TCC layout (never selects XorTileSwizzle): guarantees gcd(16, tiles) == 1 (16/16 channel spread)
// for linear-addressed weight and scale planes (qs, dm, ds).
TccRowLayout select_tcc_odd256_layout(uint32_t raw_row_bytes);
std::vector<uint8_t> pack_tcc_deswizzled(const uint8_t* src, uint32_t rows, const TccRowLayout& layout);
std::vector<uint8_t> unpack_tcc_deswizzled(const uint8_t* swz, uint32_t rows, const TccRowLayout& layout);
// Distinct TCC channels hit by logical tile `logical_tile` across the first `rows` rows (16 is ideal).
uint32_t tcc_channel_spread_count(uint32_t rows, const TccRowLayout& layout, uint32_t logical_tile = 0);

} // namespace vegajet::quant
