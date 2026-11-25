// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/prefill_planner.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstring>

// The INT8 wire must reproduce the device's un-contracted multiplies: every product / sum that the device keeps
// separate goes through a volatile so no compiler can fuse it into an FMA.

namespace vegajet::prefill {

ModelDims ornith_122b_tp2() {
    ModelDims d;
    d.D = 3072; d.n_layers = 48; d.n_gdn_layers = 36; d.n_attn_layers = 12;
    d.n_q_local = 16; d.n_kv_local = 1; d.head_dim = 256; d.kv_row_bytes = 272;
    d.gdn_nk = 8; d.gdn_nv = 32; d.gdn_dk = 128; d.gdn_dv = 128;
    d.gdn_conv_dim = 6144; d.gdn_in_stride = 10304;
    d.n_experts = 256; d.top_k = 8; d.expert_ff_local = 512; d.shared_ff_local = 512;
    d.attn_qkv_rows = 8704; d.attn_o_cols = 4096; d.gdn_out_cols = 4096;
    return d;
}

namespace {

uint64_t ceil_div(uint64_t a, uint64_t b) { return (a + b - 1u) / b; }

} // namespace

Plan plan(const ModelDims& dm, const Request& rq) {
    Plan p;
    if (dm.D == 0 || dm.D % 64u != 0u) { p.error = "D must be a non-zero multiple of 64"; return p; }
    if (rq.prompt_tokens == 0) { p.error = "empty prompt"; return p; }
    if (rq.gdn_chunk != 16u && rq.gdn_chunk != 32u) { p.error = "gdn_chunk must be 16 or 32"; return p; }
    if (dm.n_kv_local == 0 || dm.n_q_local % dm.n_kv_local != 0u) { p.error = "n_q_local % n_kv_local != 0"; return p; }

    uint32_t M = std::clamp<uint32_t>(rq.macro_m, 16u, 1024u);
    M = (M / 16u) * 16u;
    M = std::min<uint32_t>(M, static_cast<uint32_t>(ceil_div(rq.prompt_tokens, 16u) * 16u));
    p.macro_m = M;
    p.n_macro = static_cast<uint32_t>(ceil_div(rq.prompt_tokens, M));

    const bool f16 = rq.precision == Precision::kFp16Intermediate;
    const uint64_t fe = f16 ? 2u : 4u;   // bytes per float intermediate
    const char* ft = f16 ? "f16" : "f32";
    const uint64_t m = M;
    const uint64_t D = dm.D;
    auto add = [&](const char* n, uint64_t b, const char* t, uint32_t cls) { p.buffers.push_back(Buffer{n, b, t, cls}); };

    // Always live (class 0).
    add("h (residual, fp32 state)", m * D * 4u, "f32", 0);
    add("x_q8", m * D, "i8", 0);
    add("x_ds", m * (D / 32u) * 8u, "f32x2", 0);
    add("x_norm (router input)", m * D * fe, ft, 0);
    add("partial (pre-AllReduce)", m * D * fe, ft, 0);
    add("mid_q8", m * std::max(dm.attn_o_cols, dm.gdn_out_cols), "i8", 0);
    add("mid_ds", m * (std::max(dm.attn_o_cols, dm.gdn_out_cols) / 32u) * 8u, "f32x2", 0);
    // MoE
    add("router_logits", m * dm.n_experts * 4u, "f32", 0);
    add("top_ids + top_w", m * dm.top_k * 8u, "i32+f32", 0);
    add("moe sort (off, cnt, sorted_ms)", (dm.n_experts * 2u + 1u) * 4u + m * dm.top_k * 4u, "u32", 0);
    add("moe work queue", 128u, "u32", 0);
    add("sh_gate_sc", m * 4u, "f32", 0);
    add("sh_gu", m * 2u * dm.shared_ff_local * fe, ft, 0);
    add("sh_q8 + sh_ds", m * dm.shared_ff_local + m * (dm.shared_ff_local / 32u) * 8u, "i8+f32x2", 0);
    add("sh_down", m * D * fe, ft, 0);
    add("moe_q8 (routed mid)", m * dm.top_k * dm.expert_ff_local, "i8", 0);
    add("moe_d", m * dm.top_k * (dm.expert_ff_local / 32u) * 4u, "f32", 0);
    // AllReduce INT8 staging (local pack, used by the SDMA path; the FP16 path reads `partial` directly).
    add("ar int8 stage (tx)", m * (D + D / 8u), "i8+f32", 0);

    // GDN mixer (class 1).
    if (dm.n_gdn_layers) {
        add("gdn_qkv_gab", m * dm.gdn_in_stride * fe, ft, 1);
        add("gdn_conv", m * dm.gdn_conv_dim * fe, ft, 1);
        add("gdn_decay + gdn_beta", m * dm.gdn_nv * 8u, "f32", 1);
        // Deferred head-wide RMS of vj_prefill_gdn.hip: per token the unnormalised outputs (FP32, needed bit-exact
        // for parity with kernel 5b) and the dv/2 pair-squares that feed the 5b wave64 tree.
        add("gdn_fo_scratch", m * dm.gdn_nv * dm.gdn_dv * 4u, "f32", 1);
        add("gdn_sq_scratch", m * dm.gdn_nv * (dm.gdn_dv / 2u) * 4u, "f32", 1);
        add("gdn barrier", 256u, "u32", 1);
    }
    // Attention mixer (class 2).
    if (dm.n_attn_layers) {
        const uint64_t qh = static_cast<uint64_t>(dm.n_q_local) * dm.head_dim;
        add("attn_qkv", m * dm.attn_qkv_rows * fe, ft, 2);
        add("q_buf", m * qh * fe, ft, 2);
        add("gate_buf", m * qh * fe, ft, 2);
    }

    uint64_t c0 = 0, c1 = 0, c2 = 0;
    for (const auto& b : p.buffers) {
        p.bytes_no_alias += b.bytes;
        (b.alias_class == 0 ? c0 : (b.alias_class == 1 ? c1 : c2)) += b.bytes;
    }
    p.bytes_aliased = c0 + std::max(c1, c2);

    // GDN schedule.
    p.gdn.chunk = rq.gdn_chunk;
    p.gdn.chunks_per_macro = static_cast<uint32_t>(ceil_div(M, rq.gdn_chunk));
    p.gdn.wgs_per_head = rq.gdn_wgs_per_head;
    p.gdn.wgs = dm.gdn_nv * rq.gdn_wgs_per_head;

    // Attention schedule (16 head-packed rows x 32-key tiles, causal by absolute position).
    if (dm.n_attn_layers) {
        const uint32_t G = dm.n_q_local / dm.n_kv_local;
        p.attn.rows_per_wg = rq.attn_br;
        p.attn.keys_per_tile = rq.attn_bc;
        for (uint32_t mb = 0; mb < p.n_macro; ++mb) {
            const uint32_t t0 = mb * M;
            const uint32_t mm = std::min<uint32_t>(M, rq.prompt_tokens - t0);
            const uint64_t rows = static_cast<uint64_t>(mm) * G;
            const uint64_t wgs = ceil_div(rows, rq.attn_br);
            if (mb == 0) p.attn.wgs_per_macro = static_cast<uint32_t>(wgs * dm.n_kv_local);
            for (uint64_t w = 0; w < wgs; ++w) {
                const uint64_t last_row = std::min<uint64_t>(rows - 1u, (w + 1u) * rq.attn_br - 1u);
                const uint64_t keys = rq.pos0 + t0 + last_row / G + 1u;
                const uint64_t tiles = ceil_div(keys, rq.attn_bc);
                p.attn.kv_tiles_total += tiles * dm.n_kv_local;
                p.attn.score_flops += 2ull * rq.attn_br * tiles * rq.attn_bc * dm.head_dim * dm.n_kv_local;
            }
        }
        p.attn.kv_tiles_total *= dm.n_attn_layers;
        p.attn.score_flops *= dm.n_attn_layers;
        p.attn.kv_bytes_streamed = p.attn.kv_tiles_total * 2u * rq.attn_bc * dm.kv_row_bytes;
    }

    // AllReduce.
    p.ar.fp16_bytes = m * D * 2u;
    p.ar.int8_bytes = m * (D + D / 8u);
    p.ar.fp16_chunks = static_cast<uint32_t>(ceil_div(p.ar.fp16_bytes, rq.mailbox_bytes));
    p.ar.int8_chunks = static_cast<uint32_t>(ceil_div(p.ar.int8_bytes, rq.mailbox_bytes));
    p.ar.use_int8 = rq.allow_int8_wire;
    p.ar.use_sdma = (p.ar.use_int8 ? p.ar.int8_bytes : p.ar.fp16_bytes) >= rq.sdma_min_bytes;

    p.ok = true;
    return p;
}

std::string describe(const Plan& p) {
    if (!p.ok) return "plan error: " + p.error;
    std::string s;
    char line[256];
    std::snprintf(line, sizeof(line), "macro M=%u x %u | GDN C=%u (%u chunks, %u WGs) | attn %u WGs/macro, %llu KV tiles\n",
                  p.macro_m, p.n_macro, p.gdn.chunk, p.gdn.chunks_per_macro, p.gdn.wgs, p.attn.wgs_per_macro,
                  static_cast<unsigned long long>(p.attn.kv_tiles_total));
    s += line;
    for (const auto& b : p.buffers) {
        std::snprintf(line, sizeof(line), "  %-34s %10.3f MiB  %-9s class %u\n", b.name.c_str(),
                      static_cast<double>(b.bytes) / 1048576.0, b.dtype, b.alias_class);
        s += line;
    }
    std::snprintf(line, sizeof(line), "  total %.2f MiB/GPU (aliased GDN|attn: %.2f MiB/GPU)\n",
                  static_cast<double>(p.bytes_no_alias) / 1048576.0, static_cast<double>(p.bytes_aliased) / 1048576.0);
    s += line;
    std::snprintf(line, sizeof(line), "  AllReduce/macro: fp16 %.2f MiB (%u chunks) int8 %.2f MiB (%u chunks) wire=%s sdma=%s\n",
                  static_cast<double>(p.ar.fp16_bytes) / 1048576.0, p.ar.fp16_chunks,
                  static_cast<double>(p.ar.int8_bytes) / 1048576.0, p.ar.int8_chunks,
                  p.ar.use_int8 ? "int8" : "fp16", p.ar.use_sdma ? "yes" : "no");
    s += line;
    return s;
}

// ------------------------------------------------------------------------------------------------------------
// INT8 / FP16 AllReduce wire references
// ------------------------------------------------------------------------------------------------------------
namespace {

// __float2int_rn: round half to even (independent of the current FP environment).
inline int32_t rn_even(float x) {
    const float r = std::nearbyint(x);   // FE_TONEAREST is the default and is not changed by this library
    return static_cast<int32_t>(r);
}

inline float mul_nc(float a, float b) {
    volatile float r = a * b;            // keep the product rounded on its own (no FMA contraction)
    return r;
}

} // namespace

void ar_q8_pack_row(const float* a, uint32_t D, int8_t* q_out, float* scale_out) {
    for (uint32_t b = 0; b < D / 32u; ++b) {
        float amax = 0.0f;
        for (uint32_t i = 0; i < 32u; ++i) amax = std::fmax(amax, std::fabs(a[b * 32u + i]));
        const float ld = mul_nc(amax, 1.0f / 127.0f);
        const float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        scale_out[b] = ld;
        for (uint32_t i = 0; i < 32u; ++i) {
            const int32_t q = std::max(-127, std::min(127, rn_even(mul_nc(a[b * 32u + i], inv))));
            q_out[b * 32u + i] = static_cast<int8_t>(q);
        }
    }
}

void ar_q8_unpack_row(const int8_t* q, const float* scale, uint32_t D, float* out) {
    for (uint32_t i = 0; i < D; ++i) out[i] = mul_nc(static_cast<float>(q[i]), scale[i / 32u]);
}

void ar_q8_accum_row(float* h, const int8_t* ql, const float* sl, const int8_t* qp, const float* sp, uint32_t D) {
    for (uint32_t i = 0; i < D; ++i) {
        const float l = mul_nc(static_cast<float>(ql[i]), sl[i / 32u]);
        const float p = mul_nc(static_cast<float>(qp[i]), sp[i / 32u]);
        volatile float lp = l + p;
        h[i] += lp;
    }
}

float ar_q8_err_bound(float block_amax) {
    // |a*inv - rn(a*inv)| <= 0.5 plus the rounding of a*inv (|a*inv| <= 127, 1 ulp <= 2^-17), then the
    // dequant product rounding (<= 2^-24 relative of |q*ld| <= amax) and ld's own rounding (2^-24 relative).
    const float ld = block_amax / 127.0f;
    return ld * (0.5f + 0x1p-16f) + block_amax * 0x1p-22f;
}

uint16_t f32_to_f16_rne(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFFu) == 0xFFu) return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t hm = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1u);
        const uint32_t half = 1u << (shift - 1u);
        if (rem > half || (rem == half && (hm & 1u))) ++hm;
        return static_cast<uint16_t>(sign | hm);
    }
    uint32_t hm = mant >> 13;
    const uint32_t rem = mant & 0x1FFFu;
    uint32_t he = static_cast<uint32_t>(exp);
    if (rem > 0x1000u || (rem == 0x1000u && (hm & 1u))) {
        ++hm;
        if (hm == 0x400u) { hm = 0; ++he; if (he >= 31u) return static_cast<uint16_t>(sign | 0x7C00u); }
    }
    return static_cast<uint16_t>(sign | (he << 10) | hm);
}

float f16_to_f32(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t x;
    if (exp == 0) {
        if (mant == 0) x = sign;
        else {
            int32_t e = -1;
            do { mant <<= 1; ++e; } while ((mant & 0x400u) == 0);
            x = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) | ((mant & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        x = sign | 0x7F800000u | (mant << 13);
    } else {
        x = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

void ar_f16_accum_row(float* h, const float* l, const float* p, uint32_t D) {
    for (uint32_t i = 0; i < D; ++i) {
        const float lh = f16_to_f32(f32_to_f16_rne(l[i]));
        const float ph = f16_to_f32(f32_to_f16_rne(p[i]));
        volatile float s = lh + ph;
        h[i] += s;
    }
}

} // namespace vegajet::prefill
