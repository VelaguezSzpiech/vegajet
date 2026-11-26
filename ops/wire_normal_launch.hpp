// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Normal-engine Q and KV wire producer/adaptor for the sealed gfx906 Q8_0 / Q4_0 Split-KV pair.
//
// The sealed pair (ops/kernels/vj_flash_decode_splitkv.hiph, host contract
// ops/flash_decode_splitkv_launch.hpp) reads Q[M][16][8][B], K[T][8][B], V[T][8][B] wire bytes and
// refuses anything else. The normal engine stores
//   * Q as FP32 [M][16][256] in q_buf (ornith_attn_qkv_prep_kernel), and
//   * K/V as 272 B active Q8_0 rows: 8 FP16 block scales in bytes 0..15, 256 int8 in bytes 16..271.
// Neither is the sealed wire layout, so this header defines the missing producer/adaptor plus the
// host-side plan the engine enqueues. It is pure C++17 (no HIP) so the CPU tests can compile it with
// GCC/Clang and compare every produced byte against an independent native-layout reference.
//
// LAYOUT CONTRACT (must stay byte-identical to ops/kernels/vj_wire_normal.hip):
//   Q8 wire block, 48 B: FP16 d at byte 0, bytes 2..15 zero, 32 signed int8 at byte 16.
//   Q4 wire block, 32 B: FP16 d at byte 0, bytes 2..15 zero, 16 native nibble pairs at byte 16
//                        (low nibble = element j, high nibble = element j+16, unsigned biased by 8).
//   Q wire row = 16 heads * 8 blocks * B; K/V wire row = 1 KV head * 8 blocks * B.
// Q8 values are the active engine's quantizer (amax/127, round-to-nearest-even, clamp +-127).
// Q4 follows the GGML q4_0 convention: d = signed-max / -8 rounded to FP16, id = 1/d, q = rn(x*id),
// clamp [-8,7], nibble = q+8. The sealed kernel's value is (nibble-8)*d, so a negative d is legal and
// is what lets the signed extreme quantize to -8 exactly.
//
// The active 272 B KV row can be adapted to Q8 wire losslessly: block b's FP16 scale is copied and
// its 32 int8 payload bytes are copied unchanged. Q4 necessarily requantizes (4-bit payload).
#pragma once

#include "ops/flash_decode_splitkv_launch.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace vegajet::wire_normal {

using WireFormat = flash_decode_splitkv::WireFormat;
using FallbackBase = flash_decode_splitkv::Fallback;

// ---- layouts ---------------------------------------------------------------
constexpr uint32_t kHeads = flash_decode_splitkv::kHeads;        // 16 local Q heads
constexpr uint32_t kKvHeads = 1u;                                // one local KV head
constexpr uint32_t kHeadDim = flash_decode_splitkv::kHeadDim;    // 256
constexpr uint32_t kBlocksPerHead = flash_decode_splitkv::kBlocksPerHead; // 8
constexpr uint32_t kValuesPerBlock = 32u;
constexpr uint32_t kQ8BlockBytes = flash_decode_splitkv::kQ8BlockBytes;
constexpr uint32_t kQ4BlockBytes = flash_decode_splitkv::kQ4BlockBytes;
constexpr uint32_t kNativeKvRowBytes = 272u;  // active engine Q8_0 cache row
constexpr uint32_t kNativeScaleBytes = 16u;   // 8 FP16 scales
constexpr uint32_t kNativePayloadBytes = 256u;
constexpr uint32_t kProducerThreads = 256u;
constexpr uint32_t kMaxRows = 8u;             // sealed M = 1 or 5..8

constexpr uint32_t wire_head_bytes(WireFormat format) {
    return flash_decode_splitkv::head_bytes(format);
}
constexpr uint32_t wire_block_bytes(WireFormat format) {
    return flash_decode_splitkv::block_bytes(format);
}
constexpr uint32_t wire_row_bytes(WireFormat format, uint32_t kv_heads = kKvHeads) {
    return kv_heads * wire_head_bytes(format);
}
constexpr uint32_t q_wire_row_bytes(WireFormat format) {
    return kHeads * wire_head_bytes(format);
}

// ---- IEEE binary16 (round-to-nearest-even), deterministic on host and device -------------
inline uint16_t f32_to_f16_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
    const uint32_t exponent = (bits >> 23) & 0xFFu;
    const uint32_t mantissa = bits & 0x7FFFFFu;
    if (exponent == 0xFFu) { // infinity or NaN
        if (mantissa == 0u) return static_cast<uint16_t>(sign | 0x7C00u);
        uint32_t payload = mantissa >> 13;
        if (payload == 0u) payload = 1u;
        return static_cast<uint16_t>(sign | 0x7C00u | payload);
    }
    const int32_t exponent32 = static_cast<int32_t>(exponent) - 127;
    if (exponent32 > 15) return static_cast<uint16_t>(sign | 0x7C00u); // overflow -> infinity
    if (exponent32 >= -14) { // normal half
        uint32_t half_exp = static_cast<uint32_t>(exponent32 + 15);
        uint32_t kept = mantissa >> 13;
        const uint32_t dropped = mantissa & 0x1FFFu;
        if (dropped > 0x1000u || (dropped == 0x1000u && (kept & 1u) != 0u)) ++kept;
        if (kept == 0x400u) {
            kept = 0u;
            ++half_exp;
            if (half_exp >= 31u) return static_cast<uint16_t>(sign | 0x7C00u);
        }
        return static_cast<uint16_t>(sign | (half_exp << 10) | kept);
    }
    if (exponent32 < -25) return sign; // below half the smallest subnormal
    const uint32_t implicit = mantissa | 0x800000u;
    const uint32_t shift = static_cast<uint32_t>(-exponent32 - 1);
    uint32_t out = implicit >> shift;
    const uint32_t remainder = implicit & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1u);
    if (remainder > halfway || (remainder == halfway && (out & 1u) != 0u)) ++out;
    return static_cast<uint16_t>(sign | out);
}

inline float f16_bits_to_f32(uint16_t bits) {
    const uint32_t sign = (bits >> 15) & 1u;
    const uint32_t exponent = (bits >> 10) & 0x1Fu;
    const uint32_t mantissa = bits & 0x3FFu;
    float magnitude;
    if (exponent == 0u) {
        magnitude = std::ldexp(static_cast<float>(mantissa), -24);
    } else if (exponent == 31u) {
        magnitude = mantissa == 0u ? std::numeric_limits<float>::infinity()
                                   : std::numeric_limits<float>::quiet_NaN();
    } else {
        magnitude = std::ldexp(static_cast<float>(1024u + mantissa),
                               static_cast<int>(exponent) - 25);
    }
    return sign != 0u ? -magnitude : magnitude;
}

// ---- host producers --------------------------------------------------------
// Quantize one 32-float block into the sealed Q8 block envelope. Exact active-engine convention.
inline void quantize_q8_block(const float* x, uint8_t* block) {
    float amax = 0.0f;
    for (uint32_t i = 0; i < kValuesPerBlock; ++i) {
        const float a = std::fabs(x[i]);
        if (a > amax) amax = a;
    }
    const float scale = amax * (1.0f / 127.0f);
    const float inverse = amax > 0.0f ? (127.0f / amax) : 0.0f;
    std::memset(block, 0, kNativeScaleBytes);
    const uint16_t half = f32_to_f16_bits(scale);
    block[0] = static_cast<uint8_t>(half & 0xFFu);
    block[1] = static_cast<uint8_t>(half >> 8);
    for (uint32_t i = 0; i < kValuesPerBlock; ++i) {
        int q = static_cast<int>(std::nearbyint(x[i] * inverse));
        if (q < -127) q = -127;
        if (q > 127) q = 127;
        block[16u + i] = static_cast<uint8_t>(static_cast<int8_t>(q));
    }
}

// Quantize one 32-float block into the sealed Q4 block envelope (GGML q4_0 convention).
inline void quantize_q4_block(const float* x, uint8_t* block) {
    float amax = 0.0f;
    float signed_max = 0.0f;
    for (uint32_t i = 0; i < kValuesPerBlock; ++i) {
        const float a = std::fabs(x[i]);
        // Deterministic tie rule shared with the device reduction: among equal magnitudes the
        // numerically larger signed value wins, so lane order cannot change the sign of d.
        if (a > amax || (a == amax && x[i] > signed_max)) {
            amax = a;
            signed_max = x[i];
        }
    }
    const float d = signed_max == 0.0f ? 0.0f : (signed_max / -8.0f);
    const uint16_t half = f32_to_f16_bits(d);
    const float half_d = f16_bits_to_f32(half);
    const float inverse = half_d != 0.0f ? (1.0f / half_d) : 0.0f;
    uint8_t nibbles[kValuesPerBlock];
    for (uint32_t i = 0; i < kValuesPerBlock; ++i) {
        int q = static_cast<int>(std::nearbyint(x[i] * inverse));
        if (q < -8) q = -8;
        if (q > 7) q = 7;
        nibbles[i] = static_cast<uint8_t>(q + 8);
    }
    std::memset(block, 0, kNativeScaleBytes);
    block[0] = static_cast<uint8_t>(half & 0xFFu);
    block[1] = static_cast<uint8_t>(half >> 8);
    for (uint32_t j = 0; j < 16u; ++j) {
        block[16u + j] = static_cast<uint8_t>(nibbles[j] | (nibbles[j + 16u] << 4));
    }
}

inline float native_value(const uint8_t* native_row, uint32_t element) {
    const uint32_t block = element >> 5;
    const uint32_t sub = element & 31u;
    const uint16_t scale_bits =
        static_cast<uint16_t>(native_row[2u * block] | (native_row[2u * block + 1u] << 8));
    const int8_t q = static_cast<int8_t>(native_row[16u + (block << 5) + sub]);
    return f16_bits_to_f32(scale_bits) * static_cast<float>(q);
}

// Lossless adaptor: the active 272 B Q8_0 row -> the sealed Q8 wire row.
inline void q8_wire_from_native_row(const uint8_t* native_row, uint8_t* wire_row) {
    for (uint32_t block = 0; block < kBlocksPerHead; ++block) {
        uint8_t* dst = wire_row + block * flash_decode_splitkv::kQ8BlockBytes;
        std::memset(dst, 0, kNativeScaleBytes);
        dst[0] = native_row[2u * block];
        dst[1] = native_row[2u * block + 1u];
        std::memcpy(dst + 16u, native_row + 16u + block * kValuesPerBlock, kValuesPerBlock);
    }
}

// Lossy adaptor: dequantize the active Q8_0 row and requantize each 32-value block to Q4 wire.
inline void q4_wire_from_native_row(const uint8_t* native_row, uint8_t* wire_row) {
    for (uint32_t block = 0; block < kBlocksPerHead; ++block) {
        float values[kValuesPerBlock];
        for (uint32_t i = 0; i < kValuesPerBlock; ++i)
            values[i] = native_value(native_row, block * kValuesPerBlock + i);
        quantize_q4_block(values, wire_row + block * flash_decode_splitkv::kQ4BlockBytes);
    }
}

// FP32 q_buf [rows][heads][256] -> sealed Q wire. `out` must hold q_wire_row_bytes(format)*rows.
inline void produce_q_wire(const float* q, uint32_t rows, WireFormat format, uint8_t* out) {
    const uint32_t block_bytes = wire_block_bytes(format);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t head = 0; head < kHeads; ++head) {
            const float* source = q + (static_cast<size_t>(row) * kHeads + head) * kHeadDim;
            uint8_t* destination =
                out + (static_cast<size_t>(row) * kHeads + head) * wire_head_bytes(format);
            for (uint32_t block = 0; block < kBlocksPerHead; ++block) {
                const float* values = source + block * kValuesPerBlock;
                uint8_t* target = destination + block * block_bytes;
                if (format == WireFormat::Q4) quantize_q4_block(values, target);
                else quantize_q8_block(values, target);
            }
        }
    }
}

// Active 272 B K/V rows [rows][kv_heads] -> sealed wire rows. `wire_k`/`wire_v` are indexed from
// row 0, so the caller passes the full-cache bases and the row range to refresh.
inline void produce_kv_wire(const uint8_t* native_k, const uint8_t* native_v, uint32_t begin,
                            uint32_t end, WireFormat format, uint8_t* wire_k, uint8_t* wire_v) {
    const uint32_t stride = wire_row_bytes(format);
    for (uint32_t row = begin; row < end; ++row) {
        for (uint32_t head = 0; head < kKvHeads; ++head) {
            const size_t source = static_cast<size_t>(row) * kKvHeads + head;
            const size_t target = source * stride;
            if (format == WireFormat::Q4) {
                q4_wire_from_native_row(native_k + source * kNativeKvRowBytes, wire_k + target);
                q4_wire_from_native_row(native_v + source * kNativeKvRowBytes, wire_v + target);
            } else {
                q8_wire_from_native_row(native_k + source * kNativeKvRowBytes, wire_k + target);
                q8_wire_from_native_row(native_v + source * kNativeKvRowBytes, wire_v + target);
            }
        }
    }
}

// Active-engine attention epilogue reference: sigmoid gate, block-32 INT8 quant, float2 {scale,sum}.
inline void epilogue_reference(const float* attention, const float* gate, uint32_t rows,
                               int8_t* out_q8, float* out_ds) {
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t head = 0; head < kHeads; ++head) {
            const size_t base = (static_cast<size_t>(row) * kHeads + head) * kHeadDim;
            for (uint32_t block = 0; block < kBlocksPerHead; ++block) {
                float values[kValuesPerBlock];
                float amax = 0.0f;
                float sum = 0.0f;
                for (uint32_t i = 0; i < kValuesPerBlock; ++i) {
                    const float g = gate[base + block * kValuesPerBlock + i];
                    const float y = attention[base + block * kValuesPerBlock + i] /
                                    (1.0f + std::exp(-g));
                    values[i] = y;
                    const float a = std::fabs(y);
                    if (a > amax) amax = a;
                    sum += y;
                }
                const float scale = amax * (1.0f / 127.0f);
                const float inverse = amax > 0.0f ? (127.0f / amax) : 0.0f;
                for (uint32_t i = 0; i < kValuesPerBlock; ++i) {
                    int q = static_cast<int>(std::nearbyint(values[i] * inverse));
                    if (q < -127) q = -127;
                    if (q > 127) q = 127;
                    out_q8[base + block * kValuesPerBlock + i] = static_cast<int8_t>(q);
                }
                float* pair = out_ds +
                    ((static_cast<size_t>(row) * kHeads + head) * kBlocksPerHead + block) * 2u;
                pair[0] = scale;
                pair[1] = sum;
            }
        }
    }
}

// ---- kernel argument images (must match ops/kernels/vj_wire_normal.hip) -------------------
constexpr const char* kQ8QEntry = "vj_wire_q8_q_kernel";
constexpr const char* kQ4QEntry = "vj_wire_q4_q_kernel";
constexpr const char* kQ8KvEntry = "vj_wire_q8_kv_kernel";
constexpr const char* kQ4KvEntry = "vj_wire_q4_kv_kernel";
constexpr const char* kEpilogueEntry = "vj_wire_attn_epilogue_kernel";

constexpr uint32_t kQProducerKernargBytes = 24u;
constexpr uint32_t kKvProducerKernargBytes = 44u;
constexpr uint32_t kEpilogueKernargBytes = 40u;

struct QProducerArgs {
    uint64_t q_f32 = 0;  // device VA of [rows][16][256] FP32
    uint64_t q_wire = 0; // device VA of [rows][16][8][B]
    uint32_t rows = 0;
    uint32_t heads = 0;
};
static_assert(sizeof(QProducerArgs) == kQProducerKernargBytes,
              "q producer kernarg image must be 24 B");
static_assert(offsetof(QProducerArgs, q_f32) == 0);
static_assert(offsetof(QProducerArgs, q_wire) == 8);
static_assert(offsetof(QProducerArgs, rows) == 16);
static_assert(offsetof(QProducerArgs, heads) == 20);

struct KvProducerArgs {
    uint64_t src_k = 0; // device VA of active 272 B rows
    uint64_t src_v = 0;
    uint64_t dst_k = 0; // device VA of wire rows (full cache base; grid covers [begin,end))
    uint64_t dst_v = 0;
    uint32_t begin = 0;
    uint32_t end = 0;
    uint32_t kv_heads = 0;
};
// The declared kernarg segment carries the seven fields without trailing padding (44 B); the C++
// struct rounds to 48 B. Passing sizeof(KvProducerArgs) is harmless (the runtime kernarg slot is
// 512 B), but the metadata gate checks 44.
static_assert(sizeof(KvProducerArgs) == 48, "natural tail padding of the 44 B kernarg image");
static_assert(offsetof(KvProducerArgs, src_k) == 0);
static_assert(offsetof(KvProducerArgs, src_v) == 8);
static_assert(offsetof(KvProducerArgs, dst_k) == 16);
static_assert(offsetof(KvProducerArgs, dst_v) == 24);
static_assert(offsetof(KvProducerArgs, begin) == 32);
static_assert(offsetof(KvProducerArgs, end) == 36);
static_assert(offsetof(KvProducerArgs, kv_heads) == 40);

struct EpilogueArgs {
    uint64_t attention = 0; // [rows][16][256] FP32
    uint64_t gate = 0;      // [rows][16][256] FP32
    uint64_t out_q8 = 0;    // [rows][16][256] int8
    uint64_t out_ds = 0;    // [rows][16][8] float2
    uint32_t rows = 0;
    uint32_t heads = 0;
};
static_assert(sizeof(EpilogueArgs) == kEpilogueKernargBytes,
              "epilogue kernarg image must be 40 B");
static_assert(offsetof(EpilogueArgs, attention) == 0);
static_assert(offsetof(EpilogueArgs, gate) == 8);
static_assert(offsetof(EpilogueArgs, out_q8) == 16);
static_assert(offsetof(EpilogueArgs, out_ds) == 24);
static_assert(offsetof(EpilogueArgs, rows) == 32);
static_assert(offsetof(EpilogueArgs, heads) == 36);

// ---- cache updates, ranges, plans -----------------------------------------
struct CacheRange {
    bool valid = false;
    uint32_t begin = 0;
    uint32_t end = 0;
    const char* reason = "";
};

// Refresh [min(produced_rows, pos0), pos0 + rows). Clamping to pos0 is the speculative-verify
// rollback rule: rows written for rejected drafts are never reused, they are overwritten by the
// next prep dispatch and must be re-adapted. produced_rows == end means the rows the sealed pair
// is about to read are already in the wire cache and no KV dispatch is needed.
inline CacheRange plan_cache_update(uint32_t produced_rows, uint32_t pos0, uint32_t rows,
                                    uint32_t capacity_rows) {
    CacheRange range;
    if (rows == 0u || rows > kMaxRows || (rows != 1u && rows < 5u)) {
        range.reason = "rows must be 1 or 5..8";
        return range;
    }
    if (pos0 > capacity_rows || rows > capacity_rows - pos0) {
        range.reason = "pos0 + rows exceeds the wire cache capacity";
        return range;
    }
    const uint32_t end = pos0 + rows;
    const uint32_t begin = produced_rows > pos0 ? pos0 : produced_rows;
    if (begin > end) {
        range.reason = "produced row high-water mark is past the request";
        return range;
    }
    range.valid = true;
    range.begin = begin;
    range.end = end;
    range.reason = "ok";
    return range;
}

struct ProducerPlan {
    bool valid = false;
    WireFormat format = WireFormat::None;
    uint32_t rows = 0;
    uint32_t cache_rows = 0;
    uint32_t kv_begin = 0;
    uint32_t kv_end = 0;
    uint32_t q_grid_x = 0;
    uint32_t kv_grid_x = 0;
    uint64_t q_bytes = 0;
    uint64_t k_bytes = 0;
    uint64_t v_bytes = 0;
    const char* q_entry = nullptr;
    const char* kv_entry = nullptr;
    const char* reason = "";
};

inline ProducerPlan plan_producers(uint32_t rows, uint32_t cache_rows, uint32_t produced_rows,
                                   uint32_t pos0, uint32_t capacity_rows, WireFormat format) {
    ProducerPlan plan;
    if (format != WireFormat::Q8 && format != WireFormat::Q4) {
        plan.reason = "no wire format requested";
        return plan;
    }
    if (cache_rows > capacity_rows) {
        plan.reason = "cache_rows exceeds the wire cache capacity";
        return plan;
    }
    const CacheRange range = plan_cache_update(produced_rows, pos0, rows, capacity_rows);
    if (!range.valid) {
        plan.reason = range.reason;
        return plan;
    }
    plan.format = format;
    plan.rows = rows;
    plan.cache_rows = cache_rows;
    plan.kv_begin = range.begin;
    plan.kv_end = range.end;
    plan.q_grid_x = rows * kHeads;
    plan.kv_grid_x = range.end - range.begin;
    plan.q_bytes = static_cast<uint64_t>(rows) * q_wire_row_bytes(format);
    plan.k_bytes = static_cast<uint64_t>(cache_rows) * wire_row_bytes(format);
    plan.v_bytes = plan.k_bytes;
    plan.q_entry = format == WireFormat::Q4 ? kQ4QEntry : kQ8QEntry;
    plan.kv_entry = format == WireFormat::Q4 ? kQ4KvEntry : kQ8KvEntry;
    plan.valid = plan.q_grid_x != 0u && plan.q_bytes != 0u && plan.k_bytes != 0u;
    plan.reason = plan.valid ? "ok" : "producer plan has an empty extent";
    return plan;
}

// ---- selection policy and typed fallback ----------------------------------
struct Policy {
    bool enabled = false;             // opt-in: false keeps the active single-pass path untouched
    bool allow_q4 = false;            // Q4 wire is lossy and opt-in
    bool exercise_short_context = false; // tests/A-B: relax the production split floor
    uint64_t scratch_budget_bytes = 0;   // 0 = unbounded
    uint32_t max_splits = flash_decode_splitkv::kMaxSplits;
    float attention_scale = flash_decode_splitkv::kDefaultAttentionScale;
};

enum class Fallback : uint8_t {
    None = 0,
    NotRequested,
    Dimensions,
    BatchRows,
    CacheRows,
    Format,
    AttentionScale,
    Policy,
    SmallContext,
    ScratchBudget,
    ProducerRange,
    BufferCapacity,
    ProducerMissing,
    LaunchFailed,
};

inline Fallback map_sealed_fallback(FallbackBase base) {
    switch (base) {
    case FallbackBase::None: return Fallback::None;
    case FallbackBase::Dimensions: return Fallback::Dimensions;
    case FallbackBase::BatchRows: return Fallback::BatchRows;
    case FallbackBase::CacheRows: return Fallback::CacheRows;
    case FallbackBase::Format: return Fallback::Format;
    case FallbackBase::AttentionScale: return Fallback::AttentionScale;
    case FallbackBase::Policy: return Fallback::Policy;
    case FallbackBase::SmallContext: return Fallback::SmallContext;
    case FallbackBase::ScratchBudget: return Fallback::ScratchBudget;
    }
    return Fallback::Policy;
}

// One explicit priority decision. A requested wire path disables every common fast path that could
// hide it (fused M=1 prep+flash, automatic Split-KV, spec flash), so an A/B cannot silently measure
// the baseline twice.
struct PathFlags {
    bool wire_requested = false;
    bool common_fused_m1 = false;
    bool common_auto_splitkv = false;
    bool common_spec_flash = false;
};

struct PathChoice {
    bool wire = false;
    bool common_fused_m1 = false;
    bool common_auto_splitkv = false;
    bool common_spec_flash = false;
};

inline PathChoice choose_path(const PathFlags& flags) {
    PathChoice choice;
    if (flags.wire_requested) {
        choice.wire = true; // common fast paths intentionally stay false
        return choice;
    }
    choice.common_fused_m1 = flags.common_fused_m1;
    choice.common_auto_splitkv = flags.common_auto_splitkv;
    choice.common_spec_flash = flags.common_spec_flash;
    return choice;
}

inline Policy production_policy(bool enabled, bool allow_q4) {
    Policy policy;
    policy.enabled = enabled;
    policy.allow_q4 = allow_q4;
    return policy;
}

struct Attempt {
    bool requested = false;
    bool engaged = false;
    bool common_fast_path_blocked = false;
    WireFormat format = WireFormat::None;
    Fallback fallback = Fallback::NotRequested;
    const char* reason = "";
    uint32_t rows = 0;
    uint32_t cache_rows = 0;
    uint32_t splits = 0;
    uint32_t kv_begin = 0;
    uint32_t kv_end = 0;
    uint32_t q_dispatches = 0;
    uint32_t kv_dispatches = 0;
    uint32_t partition_dispatches = 0;
    uint32_t merge_dispatches = 0;
    uint32_t epilogue_dispatches = 0;
    uint32_t dispatch_total = 0;
    uint32_t partition_grid_x = 0;
    uint32_t partition_grid_y = 0;
    uint32_t merge_grid_x = 0;
    uint32_t merge_grid_y = 0;
    uint32_t epilogue_grid_x = 0;
    uint32_t q_grid_x = 0;
    uint32_t kv_grid_x = 0;
    uint64_t q_bytes = 0;
    uint64_t kv_bytes = 0;
    uint64_t scratch_bytes = 0;
    flash_decode_splitkv::Selection sealed{};
    flash_decode_splitkv::ScratchPlan scratch{};
    ProducerPlan producers{};
};

struct Capacities {
    uint32_t cache_rows = 0;
    uint64_t q_bytes = 0;
    uint64_t scratch_bytes = 0;
};

// Decide whether the sealed pair can run for this step. It never counts a fallback as exercise:
// engaged is set only when every planned dispatch has a nonzero grid and every byte extent fits.
inline Attempt decide(const flash_decode_splitkv::Shape& shape, WireFormat requested,
                      uint32_t produced_rows, const Capacities& capacities, bool producers_loaded,
                      Policy policy) {
    Attempt attempt;
    attempt.requested = policy.enabled && requested != WireFormat::None;
    attempt.common_fast_path_blocked = attempt.requested;
    attempt.rows = shape.rows;
    attempt.cache_rows = static_cast<uint32_t>(shape.cache_rows);
    if (!attempt.requested) {
        attempt.fallback = Fallback::NotRequested;
        attempt.reason = "wire attention not requested";
        return attempt;
    }
    flash_decode_splitkv::Policy sealed_policy;
    sealed_policy.max_splits = policy.max_splits;
    sealed_policy.scratch_budget_bytes = policy.scratch_budget_bytes;
    sealed_policy.allow_q4 = policy.allow_q4;
    if (policy.exercise_short_context) {
        sealed_policy.min_total_tokens = 2u;
        sealed_policy.min_tokens_per_split = 1u;
        sealed_policy.min_splits = 2u;
    }
    const flash_decode_splitkv::Launch launch = flash_decode_splitkv::plan_launch(
        shape, requested, requested, requested, policy.attention_scale, sealed_policy);
    attempt.sealed = launch.selection;
    if (!launch.selection.use_split_kv) {
        attempt.fallback = map_sealed_fallback(launch.selection.fallback);
        attempt.reason = launch.selection.reason;
        return attempt;
    }
    attempt.format = launch.selection.format;
    attempt.splits = launch.selection.splits;
    attempt.scratch = launch.scratch;
    attempt.scratch_bytes = launch.scratch.total_bytes;
    if (launch.scratch.overflow || launch.scratch.total_bytes > capacities.scratch_bytes) {
        attempt.fallback = Fallback::BufferCapacity;
        attempt.reason = "scratch plan exceeds the engine scratch buffer";
        return attempt;
    }
    const ProducerPlan producers = plan_producers(shape.rows, static_cast<uint32_t>(shape.cache_rows),
                                                  produced_rows, static_cast<uint32_t>(shape.pos0),
                                                  capacities.cache_rows, attempt.format);
    attempt.producers = producers;
    attempt.q_bytes = producers.q_bytes;
    attempt.kv_bytes = producers.k_bytes;
    attempt.kv_begin = producers.kv_begin;
    attempt.kv_end = producers.kv_end;
    attempt.q_grid_x = producers.q_grid_x;
    attempt.kv_grid_x = producers.kv_grid_x;
    if (!producers.valid) {
        attempt.fallback = Fallback::ProducerRange;
        attempt.reason = producers.reason;
        return attempt;
    }
    if (producers.q_bytes > capacities.q_bytes) {
        attempt.fallback = Fallback::BufferCapacity;
        attempt.reason = "Q wire bytes exceed the engine buffer";
        return attempt;
    }
    if (!producers_loaded) {
        attempt.fallback = Fallback::ProducerMissing;
        attempt.reason = "wire producer or sealed kernel module is not loaded";
        return attempt;
    }
    attempt.partition_grid_x = launch.partition.grid_x;
    attempt.partition_grid_y = launch.partition.grid_y;
    attempt.merge_grid_x = launch.merge.grid_x;
    attempt.merge_grid_y = launch.merge.grid_y;
    attempt.epilogue_grid_x = shape.rows * kHeads;
    attempt.q_dispatches = 1u; // always: the current batch's Q rows
    attempt.kv_dispatches = producers.kv_end > producers.kv_begin ? 1u : 0u;
    attempt.partition_dispatches = 1u;
    attempt.merge_dispatches = 1u;
    attempt.epilogue_dispatches = 1u;
    attempt.dispatch_total = attempt.q_dispatches + attempt.kv_dispatches +
                             attempt.partition_dispatches + attempt.merge_dispatches +
                             attempt.epilogue_dispatches;
    attempt.fallback = Fallback::None;
    attempt.reason = "sealed wire attention";
    attempt.engaged = attempt.dispatch_total != 0u && attempt.partition_grid_x != 0u &&
                      attempt.partition_grid_y != 0u && attempt.merge_grid_x != 0u &&
                      attempt.merge_grid_y != 0u && attempt.epilogue_grid_x != 0u &&
                      attempt.q_grid_x != 0u && attempt.q_bytes != 0u && attempt.kv_bytes != 0u;
    if (!attempt.engaged) {
        attempt.fallback = Fallback::ProducerRange;
        attempt.reason = "planned dispatch or byte extent is empty";
    }
    return attempt;
}

} // namespace vegajet::wire_normal
