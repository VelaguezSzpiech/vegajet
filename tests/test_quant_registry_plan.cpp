// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// core/quant_registry (unpacker hooks -> 3 native targets, round-trip cosine vs the FP32 decode) and
// core/rig_planner (crossover law, per-GPU VRAM breakdown, context headroom, MoE offload, rig matrix).
#include "core/quant_registry.hpp"
#include "core/rig_planner.hpp"
#include "tests/test_common.hpp"

#include <cmath>
#include <cstring>
#include <map>
#include <random>

using namespace vegajet;
using namespace vegajet::quant;

namespace {

std::mt19937 rng(20261125u);

float gauss(float sigma) {
    std::normal_distribution<float> n(0.0f, sigma);
    return n(rng);
}
float uni(float a, float b) {
    std::uniform_real_distribution<float> u(a, b);
    return u(rng);
}
void put_f16(uint8_t* p, float v) {
    const uint16_t h = f32_to_fp16(v);
    std::memcpy(p, &h, 2);
}

uint8_t f32_to_e4m3_nearest(float x) {
    uint8_t best = 0;
    float err = INFINITY;
    for (int c = 0; c < 256; ++c) {
        const float v = fp8_e4m3_to_f32(static_cast<uint8_t>(c));
        if (!std::isfinite(v)) continue;
        if (std::fabs(v - x) < err) { err = std::fabs(v - x); best = static_cast<uint8_t>(c); }
    }
    return best;
}

// Valid random disk data: random codes, finite positive scales in the FP16 fields (offsets from the ggml layouts).
std::vector<uint8_t> make_disk(GgmlType t, uint32_t rows, uint32_t cols) {
    const auto tr = ggml_type_traits(static_cast<uint32_t>(t));
    const size_t n_blocks = static_cast<size_t>(rows) * cols / tr.block_elems;
    std::vector<uint8_t> d(n_blocks * tr.block_bytes);
    if (t == GgmlType::F16 || t == GgmlType::BF16 || t == GgmlType::F8_E4M3) {
        for (size_t i = 0; i < n_blocks; ++i) {
            const float v = gauss(t == GgmlType::F8_E4M3 ? 1.0f : 0.02f);
            if (t == GgmlType::F16) put_f16(&d[i * 2], v);
            else if (t == GgmlType::F8_E4M3) d[i] = f32_to_e4m3_nearest(v);
            else {
                uint32_t bits;
                std::memcpy(&bits, &v, 4);
                const uint16_t b = static_cast<uint16_t>((bits + 0x8000u) >> 16);
                std::memcpy(&d[i * 2], &b, 2);
            }
        }
        return d;
    }
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& x : d) x = static_cast<uint8_t>(byte(rng));
    for (size_t b = 0; b < n_blocks; ++b) {
        uint8_t* blk = &d[b * tr.block_bytes];
        switch (t) {
            case GgmlType::Q4_1: put_f16(blk, uni(0.001f, 0.01f)); put_f16(blk + 2, -uni(0.0f, 0.08f)); break;
            case GgmlType::Q4_K: case GgmlType::Q5_K:
                put_f16(blk, uni(0.001f, 0.01f)); put_f16(blk + 2, uni(0.0005f, 0.005f)); break;
            case GgmlType::Q6_K: put_f16(blk + 208, uni(0.0001f, 0.001f)); break;
            default: put_f16(blk, uni(0.001f, 0.01f)); break;  // Q8_0, Q4_0, Q5_0, IQ4_NL, IQ4_XS, IQ3_XXS, IQ2_S
        }
    }
    return d;
}

std::vector<float> decode_ref(GgmlType t, const std::vector<uint8_t>& disk, uint32_t rows, uint32_t cols) {
    const auto& reg = DecoderRegistry::instance();
    const uint64_t rb = reg.row_bytes(t, cols);
    std::vector<float> out(static_cast<size_t>(rows) * cols);
    for (uint32_t r = 0; r < rows; ++r)
        VJ_CHECK(reg.decode_row_f32(t, disk.data() + r * rb, out.data() + static_cast<size_t>(r) * cols, cols));
    for (float v : out) VJ_CHECK(std::isfinite(v));
    return out;
}

struct RoundTrip {
    double cos = 0, max_abs = 0;
    uint64_t device_bytes = 0;
    bool exact = false;
};

RoundTrip round_trip(GgmlType t, uint32_t rows, uint32_t cols, ConvertFn alt = nullptr) {
    const auto disk = make_disk(t, rows, cols);
    const auto ref = decode_ref(t, disk, rows, cols);
    NativeTensor nt;
    std::string err;
    if (alt) VJ_CHECK(alt(t, disk.data(), rows, cols, nt));
    else VJ_CHECK(UnpackerRegistry::instance().convert(t, disk.data(), disk.size(), rows, cols, nt, &err));
    std::vector<float> back;
    VJ_CHECK(dequantize_native(nt, back));
    RoundTrip r;
    r.cos = cosine_similarity(ref.data(), back.data(), ref.size());
    r.exact = true;
    for (size_t i = 0; i < ref.size(); ++i) {
        r.max_abs = std::max(r.max_abs, static_cast<double>(std::fabs(ref[i] - back[i])));
        if (ref[i] != back[i]) r.exact = false;
    }
    r.device_bytes = nt.device_bytes();
    return r;
}

void test_round_trip_table() {
    const auto hooks = UnpackerRegistry::instance().hooks();
    VJ_CHECK(hooks.size() == 13);
    const uint32_t rows = 48, cols = 1024;
    std::printf("Unpacker round trip (%u x %u, random valid blocks, cosine vs FP32 decode of the disk data)\n", rows, cols);
    std::printf("  %-8s %-13s %-8s %13s %10s %12s\n", "disk", "target", "lossless", "bpw disk->dev", "cosine", "max |err|");
    for (const auto& h : hooks) {
        const auto t = static_cast<GgmlType>(h.disk);
        const RoundTrip r = round_trip(t, rows, cols);
        std::printf("  %-8s %-13s %-8s %5.2f -> %-5.2f %10.6f %12.3g\n", ggml_type_name(static_cast<uint32_t>(t)),
                    native_format_name(h.target), h.lossless ? "yes" : "no", disk_bits_per_weight(h.disk),
                    native_bits_per_weight(h.target, h.disk), r.cos, r.max_abs);
        VJ_CHECK(r.cos >= 0.999);
        if (h.lossless) VJ_CHECK(r.exact);  // bit-exact: same floats as the disk decode
    }
    // Every format the milestone names has a hook.
    for (GgmlType t : {GgmlType::F16, GgmlType::BF16, GgmlType::F8_E4M3, GgmlType::Q8_0, GgmlType::Q6_K, GgmlType::Q5_K,
                       GgmlType::Q4_0, GgmlType::Q4_1, GgmlType::Q4_K, GgmlType::IQ4_NL, GgmlType::IQ4_XS,
                       GgmlType::IQ3_XXS, GgmlType::IQ2_S})
        VJ_CHECK(UnpackerRegistry::instance().find(t) != nullptr);
    // Why Q4_1 / Q4_K are not mapped to the symmetric nibble format: requantization falls below the bar.
    const RoundTrip k4 = round_trip(GgmlType::Q4_K, rows, cols, convert_requant_q4_dot8);
    const RoundTrip q4 = round_trip(GgmlType::Q4_0, rows, cols, convert_requant_q4_dot8);
    std::printf("  non-default: Q4_K -> VJ_Q4_DOT8 requant cosine %.6f; Q4_0 requant %.6f (exact transcode: 1.000000)\n",
                k4.cos, q4.cos);
    VJ_CHECK(k4.cos < 0.999 && k4.cos > 0.98);
    VJ_CHECK(q4.cos < 1.0);
}

void test_q4_dot8_tcc_skew() {
    for (uint32_t cols : {256u, 1024u, 4096u, 5120u}) {
        const uint32_t rows = 32;
        const auto disk = make_disk(GgmlType::Q4_0, rows, cols);
        NativeTensor nt;
        VJ_CHECK(UnpackerRegistry::instance().convert(GgmlType::Q4_0, disk.data(), disk.size(), rows, cols, nt));
        VJ_CHECK(nt.format == NativeFormat::VJ_Q4_DOT8 && nt.lossless);
        VJ_CHECK(nt.qs_layout.tiles_per_row % 2 == 1 && nt.ds_layout.tiles_per_row % 2 == 1);
        VJ_CHECK(nt.qs_layout.stride_bytes % kTccTileBytes == 0 && nt.ds_layout.stride_bytes % kTccTileBytes == 0);
        VJ_CHECK(tcc_channel_spread_count(rows, nt.qs_layout) == 16);
        VJ_CHECK(tcc_channel_spread_count(rows, nt.ds_layout) == 16);
        VJ_CHECK(nt.payload_bytes() == static_cast<uint64_t>(rows) * cols * 9 / 16);  // 4.5 bpw
        VJ_CHECK(nt.device_bytes() >= nt.payload_bytes());
        std::vector<float> back;
        VJ_CHECK(dequantize_native(nt, back));
        VJ_CHECK(back == decode_ref(GgmlType::Q4_0, disk, rows, cols));
    }
    // A corrupt device image is rejected, not read out of bounds.
    const auto disk = make_disk(GgmlType::Q4_0, 4, 256);
    NativeTensor nt;
    VJ_CHECK(UnpackerRegistry::instance().convert(GgmlType::Q4_0, disk.data(), disk.size(), 4, 256, nt));
    nt.device_image.pop_back();
    std::vector<float> back;
    VJ_CHECK(!dequantize_native(nt, back));
}

void test_plugin_and_validation() {
    auto& reg = UnpackerRegistry::instance();
    // Duplicate, decoder-less and incomplete hooks are refused.
    VJ_CHECK(!reg.add({GgmlType::Q4_K, NativeFormat::VJ_Q8_DP4A, false, convert_requant_q8}));
    VJ_CHECK(!reg.add({GgmlType::I32, NativeFormat::VJ_Q8_DP4A, false, convert_requant_q8}));
    VJ_CHECK(!reg.add({GgmlType::Q5_0, NativeFormat::VJ_Q8_DP4A, false, nullptr}));
    // A new format is one hook.
    VJ_CHECK(reg.find(GgmlType::Q5_0) == nullptr);
    VJ_CHECK(reg.add({GgmlType::Q5_0, NativeFormat::VJ_Q8_DP4A, false, convert_requant_q8}));
    const UnpackerHook* h = reg.find(GgmlType::Q5_0);
    VJ_CHECK(h && h->target == NativeFormat::VJ_Q8_DP4A);
    VJ_CHECK(round_trip(GgmlType::Q5_0, 16, 512).cos >= 0.999);
    // Validation: wrong byte count, unaligned columns, empty tensor, null source, unknown format.
    const auto disk = make_disk(GgmlType::Q4_K, 2, 512);
    NativeTensor nt;
    std::string err;
    VJ_CHECK(!reg.convert(GgmlType::Q4_K, disk.data(), disk.size() - 1, 2, 512, nt, &err));
    VJ_CHECK(err.find("source size") != std::string::npos);
    VJ_CHECK(!reg.convert(GgmlType::Q4_K, disk.data(), disk.size(), 2, 500, nt, &err));
    VJ_CHECK(err.find("aligned") != std::string::npos);
    VJ_CHECK(!reg.convert(GgmlType::Q4_K, disk.data(), 0, 0, 512, nt, &err));
    VJ_CHECK(!reg.convert(GgmlType::Q4_K, nullptr, disk.size(), 2, 512, nt, &err));
    VJ_CHECK(!reg.convert(GgmlType::TQ1_0, disk.data(), disk.size(), 2, 512, nt, &err));
    VJ_CHECK(!convert_q4_0_exact_dot8(GgmlType::Q4_1, disk.data(), 2, 512, nt));
    VJ_CHECK(!convert_iq_packed(GgmlType::Q6_K, disk.data(), 2, 512, nt));
}

// ------------------------------------------------------------------------------------------------- planner
void add(rig::ModelSummary& m, const std::string& name, GgmlType t, uint64_t elems, uint32_t dims = 2) {
    m.tensors.push_back({name, static_cast<uint32_t>(t), dims, elems});
}

// Dense GQA transformer: n_layer x (attn q/k/v/o + ffn gate/up/down), embedding, head, norms.
rig::ModelSummary dense_model(const char* arch, uint64_t layers, uint64_t embd, uint64_t ffn, uint64_t heads,
                              uint64_t kv_heads, uint64_t vocab, GgmlType body, GgmlType head) {
    rig::ModelSummary m;
    m.arch = arch;
    m.n_layer = layers; m.n_embd = embd; m.n_head = heads; m.n_head_kv = kv_heads; m.head_dim = 128;
    m.n_ctx_train = 131072; m.n_kv_layers = layers;
    add(m, "token_embd.weight", head, vocab * embd);
    add(m, "output.weight", head, vocab * embd);
    add(m, "output_norm.weight", GgmlType::F32, embd, 1);
    const uint64_t q = heads * 128, kv = kv_heads * 128;
    for (uint64_t l = 0; l < layers; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        add(m, b + "attn_norm.weight", GgmlType::F32, embd, 1);
        add(m, b + "attn_q.weight", body, embd * q);
        add(m, b + "attn_k.weight", body, embd * kv);
        add(m, b + "attn_v.weight", body, embd * kv);
        add(m, b + "attn_output.weight", body, q * embd);
        add(m, b + "ffn_gate.weight", body, embd * ffn);
        add(m, b + "ffn_up.weight", body, embd * ffn);
        add(m, b + "ffn_down.weight", body, ffn * embd);
    }
    return m;
}

rig::ModelSummary moe_model(const char* arch, uint64_t layers, uint64_t embd, uint64_t expert_ffn, uint64_t experts,
                            uint64_t used, GgmlType attn, GgmlType exps) {
    rig::ModelSummary m;
    m.arch = arch;
    m.n_layer = layers; m.n_embd = embd; m.n_head = 64; m.n_head_kv = 4; m.head_dim = 128;
    m.n_ctx_train = 262144; m.n_kv_layers = layers; m.n_expert = experts; m.n_expert_used = used;
    add(m, "token_embd.weight", GgmlType::Q6_K, 151936 * embd);
    add(m, "output.weight", GgmlType::Q6_K, 151936 * embd);
    for (uint64_t l = 0; l < layers; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        add(m, b + "attn_norm.weight", GgmlType::F32, embd, 1);
        add(m, b + "attn_q.weight", attn, embd * 64 * 128);
        add(m, b + "attn_k.weight", attn, embd * 4 * 128);
        add(m, b + "attn_v.weight", attn, embd * 4 * 128);
        add(m, b + "attn_output.weight", attn, 64 * 128 * embd);
        add(m, b + "ffn_gate_inp.weight", GgmlType::F32, embd * experts);
        add(m, b + "ffn_gate_exps.weight", exps, embd * expert_ffn * experts);
        add(m, b + "ffn_up_exps.weight", exps, embd * expert_ffn * experts);
        add(m, b + "ffn_down_exps.weight", exps, expert_ffn * embd * experts);
    }
    return m;
}

const rig::TypeDecision* find_dec(const rig::RigPlan& p, GgmlType t) {
    for (const auto& d : p.decisions)
        if (d.disk_type == static_cast<uint32_t>(t)) return &d;
    return nullptr;
}

void check_law(const rig::RigPlan& p) {
    for (const auto& d : p.decisions) {
        if (d.decision == rig::Decision::Repack) VJ_CHECK(d.bw_new / d.bw_old > d.bpw_new / d.bpw_old);
        if (d.decision == rig::Decision::KeepCrossover) VJ_CHECK(!(d.bw_new / d.bw_old > d.bpw_new / d.bpw_old) || d.lossless);
        if (d.decision == rig::Decision::Relayout) VJ_CHECK(d.lossless && d.bpw_new <= d.bpw_old * 1.0001);
    }
}

void test_crossover_and_breakdown() {
    // 27B-class dense, Q4_K body, Q6_K head, plus an F16 projection to exercise a shrinking repack.
    auto m = dense_model("qwen3", 64, 5120, 17408, 40, 8, 151936, GgmlType::Q4_K, GgmlType::Q6_K);
    add(m, "blk.0.ssm_extra.weight", GgmlType::F16, 5120 * 1024);
    add(m, "blk.0.ffn_extra.weight", GgmlType::Q4_0, 5120 * 1024);
    add(m, "blk.0.ffn_iq3.weight", GgmlType::IQ3_XXS, 5120 * 1024);
    rig::RigSpec rig;
    rig.gpus = 2; rig.vram_gib = 32; rig.ctx = 8192;
    const auto p = rig::plan_rig(m, rig);
    check_law(p);
    VJ_CHECK(find_dec(p, GgmlType::Q4_K)->decision == rig::Decision::KeepCrossover);  // 1.0 < 8.5 / 4.5
    VJ_CHECK(find_dec(p, GgmlType::F16)->decision == rig::Decision::Repack);          // fewer bytes, faster
    VJ_CHECK(find_dec(p, GgmlType::Q4_0)->decision == rig::Decision::Relayout);       // lossless transcode
    VJ_CHECK(find_dec(p, GgmlType::IQ3_XXS)->decision == rig::Decision::Relayout);
    VJ_CHECK(find_dec(p, GgmlType::F32) == nullptr);  // norms are not repack candidates (1-D)
    VJ_CHECK(p.fits && !p.moe_offload_required);
    // KV: 2 (K,V) x 64 layers x (8 heads / 2 GPUs) x 128 x 2 B.
    VJ_CHECK(p.kv_bytes_per_token == 2ull * 64 * 4 * 128 * 2);
    VJ_CHECK(p.gpus.size() == 2 && p.gpus[0].kv == p.kv_bytes_per_token * 8192);
    for (uint32_t n : {1u, 2u, 3u, 4u, 8u, 16u, 200u}) {
        auto ranks = rig; ranks.gpus = n;
        VJ_CHECK(rig::plan_rig(m, ranks).gpus.size() == n);
    }
    VJ_CHECK(p.gpus[0].embed_head > p.gpus[1].embed_head);  // embedding lives on GPU 0
    VJ_CHECK(p.max_ctx >= 8192 && p.max_ctx <= m.n_ctx_train);
    // The Q6_K head decision depends on the rig's ridge point: same law, different verdicts for 60 / 64 CU.
    rig::RigSpec r60 = rig, r64 = rig;
    r64.n_cu = 64;
    const auto p60 = rig::plan_rig(m, r60), p64 = rig::plan_rig(m, r64);
    check_law(p60);
    check_law(p64);
    VJ_CHECK(p64.i_star > p60.i_star);
    std::printf("  Q6_K head: 60 CU %s (bw ratio %.4f vs bpw ratio %.4f), 64 CU %s (bw ratio %.4f)\n",
                rig::decision_name(find_dec(p60, GgmlType::Q6_K)->decision),
                find_dec(p60, GgmlType::Q6_K)->bw_new / find_dec(p60, GgmlType::Q6_K)->bw_old, 8.5 / 6.5625,
                rig::decision_name(find_dec(p64, GgmlType::Q6_K)->decision),
                find_dec(p64, GgmlType::Q6_K)->bw_new / find_dec(p64, GgmlType::Q6_K)->bw_old);
    VJ_CHECK(find_dec(p64, GgmlType::Q6_K)->decision == rig::Decision::KeepCrossover);
    std::printf("%s", rig::format_rig_plan(m, rig, p).c_str());
}

void test_budget_revert() {
    // All Q6_K: fits at disk size (6.56 bpw) on 1 x 16 GiB but not after a Q6_K -> Q8 repack (8.5 bpw).
    rig::ModelSummary m = dense_model("llama", 40, 5120, 20480, 40, 8, 32000, GgmlType::Q6_K, GgmlType::Q6_K);
    rig::RigSpec rig;
    rig.gpus = 1; rig.vram_gib = 16; rig.ctx = 4096; rig.n_cu = 60; rig.bw_gbs = 800;
    // Force the law to accept the repack: a slow-decoding disk format (VALU bound) on a narrow-ridge rig.
    rig.bw_gbs = 1600;  // halves I*, so Q6_K decode is VALU bound and the int8 format streams much faster
    const auto p = rig::plan_rig(m, rig);
    const auto* d = find_dec(p, GgmlType::Q6_K);
    VJ_CHECK(d && d->bw_new / d->bw_old > d->bpw_new / d->bpw_old);  // the law alone would repack
    VJ_CHECK(d->decision == rig::Decision::KeepBudget);               // the budget says no
    VJ_CHECK(p.fits);
    VJ_CHECK(d->final_bytes == d->disk_bytes);
}

void test_rig_matrix() {
    struct Case { const char* label; rig::ModelSummary m; };
    std::vector<Case> cases;
    cases.push_back({"23B dense Q4_K", dense_model("qwen3", 64, 5120, 17408, 40, 8, 151936, GgmlType::Q4_K, GgmlType::Q6_K)});
    cases.push_back({"31B-A3B MoE Q4_K", moe_model("qwen3moe", 48, 2048, 768, 128, 8, GgmlType::Q6_K, GgmlType::Q4_K)});
    cases.push_back({"160B MoE IQ3_XXS", moe_model("qwen3moe", 64, 4096, 1536, 128, 8, GgmlType::Q6_K, GgmlType::IQ3_XXS)});
    cases.push_back({"70B dense Q4_K", dense_model("llama", 80, 8192, 28672, 64, 8, 128256, GgmlType::Q4_K, GgmlType::Q6_K)});
    rig::RigSpec base;
    base.ctx = 8192;
    const auto rigs = rig::standard_rigs(base);
    VJ_CHECK(rigs.size() == 6);
    std::map<std::string, std::vector<rig::RigPlan>> plans;
    for (const auto& c : cases) {
        std::printf("%s", rig::format_rig_matrix(c.m, rigs).c_str());
        for (const auto& r : rigs) {
            plans[c.label].push_back(rig::plan_rig(c.m, r));
            const auto& p = plans[c.label].back();
            check_law(p);
            VJ_CHECK(p.gpus.size() == r.gpus);
            if (p.fits)
                for (const auto& g : p.gpus) VJ_CHECK(g.free_bytes() >= 0 && g.reserve >= (2ull << 30));
        }
    }
    // rigs: 0 = 1x16, 1 = 2x16, 2 = 2x32, 3 = 3x32, 4 = 4x32, 5 = 8x32
    const auto& d27 = plans["23B dense Q4_K"];
    VJ_CHECK(!d27[0].fits && d27[0].shortfall_bytes > 0);      // ~16 GiB of weights do not fit one 16 GiB card
    VJ_CHECK(d27[1].fits && d27[2].fits && d27[5].fits);
    VJ_CHECK(d27[5].max_ctx >= d27[2].max_ctx);
    const auto& m30 = plans["31B-A3B MoE Q4_K"];
    VJ_CHECK(m30[0].moe_offload_required && m30[0].fits);       // 1 x 16 GiB: experts spill to host RAM
    VJ_CHECK(m30[0].host_expert_bytes > 0 && m30[0].expert_resident_fraction < 1.0);
    VJ_CHECK(!m30[1].moe_offload_required && m30[1].fits);
    const auto& m120 = plans["160B MoE IQ3_XXS"];
    VJ_CHECK(m120[0].fits && m120[0].moe_offload_required);     // a 100B+ MoE runs on one 16 GiB card via offload
    VJ_CHECK(m120[1].moe_offload_required);
    VJ_CHECK(m120[5].fits && !m120[5].moe_offload_required);
    VJ_CHECK(m120[0].expert_resident_fraction < m120[1].expert_resident_fraction);
    const auto& d70 = plans["70B dense Q4_K"];
    VJ_CHECK(!d70[0].fits && !d70[1].fits && !d70[0].moe_offload_required);  // dense: no offload path
    VJ_CHECK(d70[3].fits && d70[4].fits && d70[5].fits);
}

void test_summarize_gguf() {
    vjtest::GgufWriter w;
    w.kv_str("general.architecture", "llama");
    w.kv_u32("llama.block_count", 2);
    w.kv_u32("llama.embedding_length", 256);
    w.kv_u32("llama.attention.head_count", 4);
    w.kv_arr_i32("llama.attention.head_count_kv", {2, 1});  // per-layer array: the planner takes the max
    w.kv_u32("llama.context_length", 4096);
    const auto t256 = [](GgmlType t, uint64_t n) { return std::vector<uint8_t>(ggml_row_bytes(static_cast<uint32_t>(t), n)); };
    w.tensor("token_embd.weight", {256, 64}, static_cast<uint32_t>(GgmlType::Q4_K), t256(GgmlType::Q4_K, 256 * 64));
    w.tensor("blk.0.attn_norm.weight", {256}, 0, t256(GgmlType::F32, 256));
    w.tensor("blk.0.attn_k.weight", {256, 128}, static_cast<uint32_t>(GgmlType::Q8_0), t256(GgmlType::Q8_0, 256 * 128));
    w.tensor("blk.1.ffn_up.weight", {256, 512}, static_cast<uint32_t>(GgmlType::IQ4_XS), t256(GgmlType::IQ4_XS, 256 * 512));
    const std::string path = vjtest::write_temp(w.build(), "bin");
    GgufFile f;
    VJ_CHECK(f.open(path));
    rig::ModelSummary m;
    std::string err;
    VJ_CHECK(rig::summarize_gguf(f, m, &err));
    VJ_CHECK(m.arch == "llama" && m.n_layer == 2 && m.n_embd == 256 && m.n_head == 4 && m.n_head_kv == 2);
    VJ_CHECK(m.head_dim == 64 && m.n_ctx_train == 4096);
    VJ_CHECK(m.n_kv_layers == 1);  // only blk.0 has attn_k
    VJ_CHECK(m.tensors.size() == 4 && m.total_params() == 256u * 64 + 256 + 256 * 128 + 256 * 512);
    rig::RigSpec rig;
    rig.gpus = 1; rig.vram_gib = 16;
    const auto p = rig::plan_rig(m, rig);
    VJ_CHECK(p.fits && p.max_ctx == 4096);  // capped at the training context
    std::remove(path.c_str());
    // No architecture: refused.
    vjtest::GgufWriter w2;
    w2.kv_u32("x.block_count", 1);
    const std::string p2 = vjtest::write_temp(w2.build(), "bin");
    GgufFile f2;
    VJ_CHECK(f2.open(p2));
    VJ_CHECK(!rig::summarize_gguf(f2, m, &err) && err.find("architecture") != std::string::npos);
    std::remove(p2.c_str());
}

} // namespace

int main() {
    test_round_trip_table();
    test_q4_dot8_tcc_skew();
    test_plugin_and_validation();
    test_crossover_and_breakdown();
    test_budget_revert();
    test_rig_matrix();
    test_summarize_gguf();
    std::printf("test_quant_registry_plan: OK\n");
    return 0;
}
