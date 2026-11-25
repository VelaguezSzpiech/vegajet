// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// CPU reference (pack / unpack / score) and error oracles for two quantized KV-cache formats on gfx906.
//
// q4_0 in the v_dot8 layout (18 B per 32 values, same size and rounding as ggml Q4_0):
//   Per head vector of D values (D % 32 == 0): a code plane of D/8 dwords and a scale plane of D/32 fp16.
//   Dword w, nibble n (bits 4n..4n+3) holds the signed 4-bit code (two's complement, -8..7) of element 8w+n, so one
//   v_dot8_i32_i4 multiplies 8 consecutive K elements with 8 consecutive int4 query elements; a block of 32 values
//   is 4 dwords = one 16-byte load. Value = d * code, d = fp16(max / -8) where max is the signed value of largest
//   magnitude (ggml rule: code = min(15, int(x / d + 8.5)) - 8).
//   ggml block_q4_0 (fp16 d, qs[16]: low nibble = element j, high nibble = element j+16, unsigned with offset 8)
//   converts to and from this layout bit-exactly (same d, same codes).
//
// turbo3 (14 B per 32 values = 3.5 bits, the size HEAD's kv::KvTier::TQ3 budgets):
//   1. randomized Walsh-Hadamard rotation of the whole head vector: y = H (s .* x) / sqrt(D), s = +-1 signs from a
//      fixed seed (D must be a power of two: 128 / 256); H / sqrt(D) is orthonormal, so q.k = (Rq).(Rk).
//   2. per 32 rotated values: 3-bit index into the 8-level Lloyd-Max codebook of N(0, 1) and one fp16 scale
//      (least-squares refit of the block RMS). Codes are stored as two bit planes per block: 2 dwords holding the
//      low 2 bits (element 16w+j at bits 2j..2j+1 of dword w) and 1 dword holding the high bit (element j at bit j).
//   Scores are computed in the rotated domain: rotate the query once per head, then dot it with dequantized codes;
//   no inverse transform is needed on the K side. V needs the inverse rotation once per output vector.
//
// The oracles compare attention scores q.k / sqrt(D), softmax distributions (KL divergence, top-1 agreement) and
// attention outputs against float32 on synthetic heads with persistent outlier channels (the case where per-block
// absmax formats suffer and a rotation helps).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vegajet::kvq {

constexpr uint32_t kBlock = 32;
constexpr uint32_t kQ4BytesPerBlock = 18;      // 16 B codes + 2 B fp16 scale
constexpr uint32_t kTurbo3BytesPerBlock = 14;  // 12 B codes + 2 B fp16 scale

uint16_t f32_to_f16(float f);  // round to nearest even, overflow -> inf
float f16_to_f32(uint16_t h);

// v_dot8_i32_i4 reference: c + sum over the 8 nibbles of sext4(a) * sext4(b).
constexpr int32_t dot8_i4(uint32_t a, uint32_t b, int32_t c) {
    for (int n = 0; n < 8; ++n) {
        const int32_t x = static_cast<int32_t>((a >> (4 * n)) & 0xFu), y = static_cast<int32_t>((b >> (4 * n)) & 0xFu);
        c += (x >= 8 ? x - 16 : x) * (y >= 8 ? y - 16 : y);
    }
    return c;
}

// ---- q4_0 / v_dot8 layout. codes: D/8 dwords, scales: D/32 fp16.
void q4_pack(const float* x, uint32_t dim, uint32_t* codes, uint16_t* scales);
void q4_unpack(const uint32_t* codes, const uint16_t* scales, uint32_t dim, float* out);
void ggml_q4_0_to_dot8(const uint8_t* block18, uint32_t codes[4], uint16_t* scale);
void dot8_to_ggml_q4_0(const uint32_t codes[4], uint16_t scale, uint8_t* block18);

// Query side of the integer score paths, per 32-block symmetric scales (kept in float32 registers).
void query_pack_i4(const float* q, uint32_t dim, uint32_t* codes, float* scales);  // int4, scale = amax / 7
void query_pack_i8(const float* q, uint32_t dim, int8_t* codes, float* scales);    // int8, scale = amax / 127
// Raw dot products (no 1/sqrt(D)): int4 x int4 with v_dot8 lanes, int8 x int4 (the dp4a path after nibble unpack).
float q4_dot_i4(const uint32_t* qc, const float* qs, const uint32_t* kc, const uint16_t* ks, uint32_t dim);
float q4_dot_i8(const int8_t* qc, const float* qs, const uint32_t* kc, const uint16_t* ks, uint32_t dim);

// ---- turbo3. codes: 3 * D/32 dwords, scales: D/32 fp16.
extern const float kTurbo3Codebook[8];
std::vector<int8_t> turbo3_signs(uint32_t dim, uint64_t seed);
void fwht_normalized(float* x, uint32_t n);  // in place, orthonormal (self-inverse); n must be a power of two
void turbo3_rotate(const float* x, uint32_t dim, const int8_t* signs, float* out);
void turbo3_unrotate(const float* y, uint32_t dim, const int8_t* signs, float* out);
void turbo3_pack(const float* x, uint32_t dim, const int8_t* signs, uint32_t* codes, uint16_t* scales);
void turbo3_pack_rotated(const float* y, uint32_t dim, uint32_t* codes, uint16_t* scales);
void turbo3_unpack_rotated(const uint32_t* codes, const uint16_t* scales, uint32_t dim, float* y);
void turbo3_unpack(const uint32_t* codes, const uint16_t* scales, uint32_t dim, const int8_t* signs, float* out);
float turbo3_dot_rotated(const float* q_rot, const uint32_t* codes, const uint16_t* scales, uint32_t dim);

// ---- oracles
enum class KvFormat : uint8_t { F32 = 0, Q4Dot8 = 1, Turbo3 = 2 };
enum class QueryPath : uint8_t { F32 = 0, I8 = 1, I4 = 2 };  // I8 / I4 only apply to Q4Dot8 keys
const char* format_name(KvFormat f);
const char* query_path_name(QueryPath p);
double bits_per_value(KvFormat f);

struct OracleConfig {
    uint32_t head_dim = 128;
    uint32_t n_keys = 512;
    uint32_t n_queries = 64;
    uint32_t outlier_channels = 4;  // channels whose magnitude is multiplied by outlier_scale in every key
    float outlier_scale = 12.0f;
    float focus = 4.0f;             // one key per query is boosted by `focus` score std devs -> a peaked softmax
    uint64_t seed = 1;
    uint64_t turbo_seed = 0x5455524230u;
};

struct OracleStats {
    double score_max_abs = 0;   // max |s - s_ref| over all (query, key), s = q.k / sqrt(D)
    double score_rel_rms = 0;   // rms(s - s_ref) / rms(s_ref)
    double kld_mean = 0, kld_max = 0;  // KL(p_ref || p) of the softmax over keys, nats
    double top1_agree = 0;      // fraction of queries with the same argmax key
    double out_rel_err = 0;     // mean ||o - o_ref|| / ||o_ref||, o = softmax . V (V in v_fmt)
    double k_rel_mse = 0;       // ||k - k_ref||^2 / ||k_ref||^2 over all keys
};

OracleStats run_oracle(const OracleConfig& cfg, KvFormat k_fmt, KvFormat v_fmt, QueryPath qp, std::string* err = nullptr);

} // namespace vegajet::kvq
