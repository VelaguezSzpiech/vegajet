// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Ornith plugin: dry-run metadata load, knapsack repack plan, TP sharding, KV/state planning and op-graph
// construction. Always runs on a synthetic miniature qwen35moe GGUF; additionally runs against real files
// when VEGAJET_ORNITH_GGUF (target) and VEGAJET_ORNITH_DRAFT_GGUF (DFlash draft) point to them.
#include "core/quant_repacker.hpp"
#include "models/ornith/ornith_model.hpp"
#include "models/registry.hpp"
#include "tests/test_common.hpp"

#include <cmath>
#include <map>
#include <set>

using namespace vegajet;
using namespace vegajet::ornith;

namespace {

constexpr uint64_t GiB = 1ull << 30;

double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double ab = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < a.size(); ++i) { ab += double(a[i]) * b[i]; aa += double(a[i]) * a[i]; bb += double(b[i]) * b[i]; }
    return (aa == 0 || bb == 0) ? (aa == bb ? 1.0 : 0.0) : ab / std::sqrt(aa * bb);
}

std::vector<uint8_t> f16_payload(uint64_t n, uint32_t seed) {
    std::vector<uint8_t> b(n * 2);
    uint32_t s = seed * 2654435761u + 1;
    for (uint64_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        const float v = (static_cast<int32_t>(s >> 8) % 2001 - 1000) / 8000.0f;
        const uint16_t h = quant::f32_to_fp16(v);
        std::memcpy(&b[i * 2], &h, 2);
    }
    return b;
}

std::string build_mini_ornith() {
    vjtest::GgufWriter w;
    const std::string a = "qwen35moe";
    w.kv_str("general.architecture", a);
    w.kv_u32(a + ".block_count", 5);
    w.kv_u32(a + ".context_length", 4096);
    w.kv_u32(a + ".embedding_length", 256);
    w.kv_u32(a + ".attention.head_count", 4);
    w.kv_u32(a + ".attention.head_count_kv", 2);
    w.kv_u32(a + ".attention.key_length", 64);
    w.kv_u32(a + ".attention.value_length", 64);
    w.kv_f32(a + ".rope.freq_base", 1e7f);
    w.kv_u32(a + ".rope.dimension_count", 32);
    w.kv_u32(a + ".expert_count", 8);
    w.kv_u32(a + ".expert_used_count", 2);
    w.kv_u32(a + ".expert_feed_forward_length", 64);
    w.kv_u32(a + ".expert_shared_feed_forward_length", 64);
    w.kv_u32(a + ".nextn_predict_layers", 1);
    w.kv_u32(a + ".ssm.conv_kernel", 4);
    w.kv_u32(a + ".ssm.state_size", 32);
    w.kv_u32(a + ".ssm.group_count", 2);
    w.kv_u32(a + ".ssm.time_step_rank", 4);
    w.kv_u32(a + ".ssm.inner_size", 128);
    w.kv_arr_i32(a + ".attention.recurrent_layers", {1, 1, 1, 0, 0});
    w.kv_u32(a + ".full_attention_interval", 4);
    uint32_t seed = 1;
    auto mat = [&](const std::string& n, uint64_t in, uint64_t out, uint64_t ex = 1) {
        w.tensor(n, ex > 1 ? std::vector<uint64_t>{in, out, ex} : std::vector<uint64_t>{in, out}, 1, f16_payload(in * out * ex, seed++));
    };
    auto vec = [&](const std::string& n, std::vector<uint64_t> ne) {
        uint64_t e = 1;
        for (auto x : ne) e *= x;
        w.tensor(n, ne, 0, std::vector<uint8_t>(e * 4, 0));
    };
    std::vector<uint8_t> emb(512 * 256 / 32 * 34, 0x01);
    w.tensor("token_embd.weight", {256, 512}, 8, emb);
    mat("output.weight", 256, 512);
    vec("output_norm.weight", {256});
    const bool rec[5] = {true, true, true, false, false};
    for (int il = 0; il < 5; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        vec(p + "attn_norm.weight", {256});
        vec(p + "post_attention_norm.weight", {256});
        vec(p + "ffn_gate_inp.weight", {256, 8});
        vec(p + "ffn_gate_inp_shexp.weight", {256});
        mat(p + "ffn_gate_exps.weight", 256, 64, 8);
        mat(p + "ffn_up_exps.weight", 256, 64, 8);
        mat(p + "ffn_down_exps.weight", 64, 256, 8);
        mat(p + "ffn_gate_shexp.weight", 256, 64);
        mat(p + "ffn_up_shexp.weight", 256, 64);
        mat(p + "ffn_down_shexp.weight", 64, 256);
        if (rec[il]) {
            mat(p + "attn_qkv.weight", 256, 256);
            mat(p + "attn_gate.weight", 256, 128);
            mat(p + "ssm_alpha.weight", 256, 32);
            mat(p + "ssm_beta.weight", 256, 32);
            mat(p + "ssm_out.weight", 128, 256);
            vec(p + "ssm_conv1d.weight", {4, 256});
            vec(p + "ssm_a", {4});
            vec(p + "ssm_dt.bias", {4});
            vec(p + "ssm_norm.weight", {32});
        } else {
            mat(p + "attn_q.weight", 256, 512);
            mat(p + "attn_k.weight", 256, 128);
            mat(p + "attn_v.weight", 256, 128);
            mat(p + "attn_output.weight", 256, 256);
            vec(p + "attn_q_norm.weight", {64});
            vec(p + "attn_k_norm.weight", {64});
        }
        if (il == 4) {
            mat(p + "nextn.eh_proj.weight", 512, 256);
            vec(p + "nextn.enorm.weight", {256});
            vec(p + "nextn.hnorm.weight", {256});
        }
    }
    return vjtest::write_temp(w.build(), "mini_ornith");
}

std::map<ops::OpKind, uint32_t> histogram(const ops::OpGraph& g) {
    std::map<ops::OpKind, uint32_t> h;
    for (const auto& d : g) h[d.kind]++;
    return h;
}

void check_graph(const OrnithModel& m, const ops::OpGraph& g) {
    for (const auto& d : g) {
        VJ_CHECK(static_cast<uint32_t>(d.kind) < static_cast<uint32_t>(ops::OpKind::Count));
        VJ_CHECK(ops::kernel_symbol(d.kind)[0] != '\0');
        if (!d.tensor.empty() && d.kind != ops::OpKind::RmsNorm && d.kind != ops::OpKind::GdnConv1d &&
            d.kind != ops::OpKind::GdnChunkWy && d.kind != ops::OpKind::GdnRecurrentStep && d.kind != ops::OpKind::MoeRouterTopK)
            VJ_CHECK(m.find_plan(d.tensor) != nullptr);
        if (d.kind == ops::OpKind::GemvQ8Dp4a || d.kind == ops::OpKind::GemvQ4Dot8 || d.kind == ops::OpKind::GemvIqPacked)
            VJ_CHECK(d.n > 0 && d.k > 0 && d.split_k >= 1);
    }
}

LoadOptions mi50s(size_t n, uint64_t ctx, std::vector<double> bw = {}) {
    LoadOptions o;
    o.gpus.assign(n, GpuTarget{60, 800.0, 1.725, 32 * GiB});
    for (size_t i = 0; i < bw.size() && i < n; ++i) o.gpus[i].bw_hbm_gbs = bw[i];
    o.context_length = static_cast<uint32_t>(ctx);
    o.kv_scratch_reserve_bytes = 2 * GiB;
    return o;
}

void print_report(const char* label, const OrnithModel& m, const LoadReport& rep) {
    std::map<std::string, double> fmt;
    for (const auto& p : m.tensor_plan())
        fmt[p.keep_float ? std::string("F32 (kept)") : std::string(quant::native_format_name(p.format))] += p.bytes_total / double(GiB);
    std::printf("  [%s] ok=%d %s\n", label, rep.ok, rep.ok ? "" : rep.error.c_str());
    for (const auto& [k, v] : fmt) std::printf("      %-14s %8.3f GiB\n", k.c_str(), v);
    for (size_t i = 0; i < rep.weight_bytes_per_gpu.size(); ++i)
        std::printf("      gpu%zu weights %6.2f GiB | KV %5.2f GiB | GDN state %5.3f GiB | total+reserve %6.2f GiB\n", i,
                    rep.weight_bytes_per_gpu[i] / double(GiB), rep.kv_bytes_per_gpu[i] / double(GiB),
                    rep.state_bytes_per_gpu[i] / double(GiB),
                    (rep.weight_bytes_per_gpu[i] + rep.kv_bytes_per_gpu[i] + rep.state_bytes_per_gpu[i] + 2 * GiB) / double(GiB));
    for (const auto& n : rep.notes) std::printf("      note: %s\n", n.c_str());
}

void test_synthetic() {
    const std::string path = build_mini_ornith();
    GgufFile f;
    VJ_CHECK(f.open(path));
    std::string err;
    auto model = ModelRegistry::instance().create_for(f, &err);
    VJ_CHECK(model && model->name() == "ornith");
    auto& m = static_cast<OrnithModel&>(*model);

    for (size_t tp : {1, 2, 3}) {
        LoadOptions o = mi50s(tp, 2048);
        o.gpus.assign(tp, GpuTarget{60, 800.0, 1.725, 1 * GiB});
        o.kv_scratch_reserve_bytes = 64ull << 20;
        const LoadReport rep = m.load(f, o);
        VJ_CHECK(rep.ok);
        const auto& h = m.hparams();
        VJ_CHECK(h.n_block == 5 && h.n_trunk == 4 && h.n_nextn == 1 && h.n_gdn == 3 && h.n_attn == 1 && h.vocab == 512);
        VJ_CHECK(rep.weight_bytes_per_gpu.size() == tp && rep.kv_bytes_per_gpu[0] > 0 && rep.state_bytes_per_gpu[0] > 0);
        // Expert parallel split is exact.
        const TensorPlan* ex = m.find_plan("blk.0.ffn_gate_exps.weight");
        uint64_t sum = 0;
        for (auto e : ex->local_out) sum += e;
        VJ_CHECK(ex && sum == 8);
        // KV: Q8_0 pages for 2 attention blocks (trunk + nextn).
        const auto& kvp = m.kv_plans()[0];
        VJ_CHECK(kvp.fits && kvp.spans[0].tier == kv::KvTier::Q8_0);
        const uint64_t kvh = m.attn_shards().gpus[0].kv_heads.size();
        VJ_CHECK(kvp.total_bytes == kv::kv_bytes(kv::KvGeometry{2, static_cast<uint32_t>(kvh), 64, 64}, 2048, kv::KvTier::Q8_0));

        auto dec = m.decode_step(7, 1);
        check_graph(m, dec.graph);
        auto hgram = histogram(dec.graph);
        VJ_CHECK(hgram[ops::OpKind::GdnRecurrentStep] == 3 * tp);
        VJ_CHECK(hgram[ops::OpKind::FlashDecodeWave64] == 1 * tp);
        VJ_CHECK(hgram[ops::OpKind::MoeFusedExperts] >= 4);
        VJ_CHECK(hgram[ops::OpKind::GemmPrefill] == 0 && hgram[ops::OpKind::GdnChunkWy] == 0);
        VJ_CHECK(hgram[ops::OpKind::AllReduce] == (tp > 1 ? 1 + 2 * 4 + 1 : 0));
        VJ_CHECK(hgram[ops::OpKind::LmHead] == tp);

        std::vector<int32_t> prompt(600);
        for (size_t i = 0; i < prompt.size(); ++i) prompt[i] = static_cast<int32_t>(i % 37);
        auto pf = m.prefill(prompt);
        check_graph(m, pf.graph);
        auto hp = histogram(pf.graph);
        VJ_CHECK(hp[ops::OpKind::Embed] == 2);  // 512 + 88 token chunks
        VJ_CHECK(hp[ops::OpKind::GdnChunkWy] == 2 * 3 * tp && hp[ops::OpKind::FlashPrefill] == 2 * tp);
        VJ_CHECK(hp[ops::OpKind::SsmCheckpoint] == 3 * tp);

        auto ver = m.decode_step(3, 4);
        VJ_CHECK(histogram(ver.graph)[ops::OpKind::DraftVerify] == tp);
        const auto rb = m.accept_draft({5, 6}, 3);  // 1 of 3 accepted + bonus token
        VJ_CHECK(rb.ok && rb.restore_checkpoint && rb.replay_tokens == 2);

        if (tp == 3) {
            const auto& at = m.attn_shards();
            VJ_CHECK(at.total_kv_heads_stored() == 3);  // KV head 1 straddles ranks 1/2 -> replicated
            VJ_CHECK(at.gpus[1].replicated_kv == 1 && at.gpus[2].replicated_kv == 1);
        }
        // Real repack through the loader path (F16 disk -> planned native format) for rank 0 rows.
        std::vector<float> nat, ref;
        VJ_CHECK(m.repack_shard_for_test(f, "blk.0.attn_qkv.weight", 0, 64, nat, ref, &err));
        VJ_CHECK(cosine(ref, nat) >= 0.995);
    }
    // Suffix drafting with the governor warmed to high acceptance.
    for (int i = 0; i < 40; ++i) m.governor().observe(4, 4);
    VJ_CHECK(m.governor().k_star() > 0);
    std::vector<int32_t> ctx = {1, 2, 3, 4, 5, 6, 1, 2, 3};
    m.suffix_tree().extend(ctx);
    auto dr = m.draft_step(ctx, 4);
    VJ_CHECK(m.draft_source() == DraftSource::Suffix && dr.draft_tokens > 0 && dr.graph.empty());
    std::printf("[PASS] synthetic qwen35moe: TP=1/2/3 dry-run load, Q8_0 paged KV, ReplaySSM, decode/prefill/verify graphs, "
                "GQA boundary KV replication, suffix drafting\n");
}

void test_real(const char* path, const char* draft) {
    GgufFile f;
    VJ_CHECK(f.open(path));
    std::string err;
    auto model = ModelRegistry::instance().create_for(f, &err);
    VJ_CHECK(model != nullptr);
    auto& m = static_cast<OrnithModel&>(*model);
    LoadReport rep = m.load(f, mi50s(3, 32768));
    VJ_CHECK(rep.ok);
    const auto& h = m.hparams();
    std::printf("  arch=%s blocks=%u (trunk %u + nextn %u) | GDN %u + GQA %u | n_embd=%u heads=%u kv=%u head_dim=%u\n",
                h.arch.c_str(), h.n_block, h.n_trunk, h.n_nextn, h.n_gdn, h.n_attn, h.n_embd, h.n_head, h.n_head_kv, h.head_dim_k);
    std::printf("  experts=%u used=%u ff_exp=%u ff_shexp=%u | GDN conv=%u state=%u k_groups=%u v_heads=%u inner=%u\n",
                h.n_expert, h.n_expert_used, h.n_ff_exp, h.n_ff_shexp, h.ssm_conv, h.ssm_state, h.ssm_groups, h.ssm_v_heads, h.ssm_inner);
    std::printf("  rope dims=%u base=%.0f sections=[%u,%u,%u,%u] ctx=%llu vocab=%llu (no YaRN keys present)\n", h.rope_dims,
                h.rope_base, h.rope_sections.size() > 0 ? h.rope_sections[0] : 0, h.rope_sections.size() > 1 ? h.rope_sections[1] : 0,
                h.rope_sections.size() > 2 ? h.rope_sections[2] : 0, h.rope_sections.size() > 3 ? h.rope_sections[3] : 0,
                (unsigned long long)h.context_length, (unsigned long long)h.vocab);
    VJ_CHECK(h.n_gdn + h.n_attn == h.n_trunk && h.n_gdn == 3 * h.n_attn);
    for (double reuse : {1.0, 2.0}) {
        for (size_t tp : {2, 3}) {
            LoadOptions o = mi50s(tp, 32768);
            o.batch_reuse_factor = reuse;
            rep = m.load(f, o);
            char label[64];
            std::snprintf(label, sizeof(label), "TP=%zu x 32 GiB, ctx 32K, reuse %.0fx", tp, reuse);
            print_report(label, m, rep);
            if (tp == 3) VJ_CHECK(rep.ok);
            if (tp == 3 && reuse > 1.0)  // upgraded expert trunk: requantized sub-4-bit -> VJ_Q4_DOT8 (+FWHT)
                for (const char* t : {"blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight"}) {
                    std::vector<float> nat, ref;
                    VJ_CHECK(m.repack_shard_for_test(f, t, 0, 64, nat, ref, &err));
                    const TensorPlan* p = m.find_plan(t);
                    const double c = cosine(ref, nat);
                    std::printf("      upgraded %-27s %-7s -> %s%s cos=%.6f\n", t, ggml_type_name(static_cast<uint32_t>(p->disk)),
                                quant::native_format_name(p->format), p->fwht64 ? "+FWHT" : "", c);
                    VJ_CHECK(p->format == quant::NativeFormat::VJ_Q4_DOT8 && c >= 0.995);
                }
        }
    }
    LoadOptions asym = mi50s(3, 32768, {1000.0, 800.0, 800.0});
    rep = m.load(f, asym);
    print_report("TP=3 asymmetric bandwidth 1000/800/800 GB/s", m, rep);
    VJ_CHECK(rep.weight_bytes_per_gpu[0] > rep.weight_bytes_per_gpu[1]);

    rep = m.load(f, mi50s(3, 32768));
    auto dec = m.decode_step(1, 1);
    check_graph(m, dec.graph);
    auto hg = histogram(dec.graph);
    std::printf("  decode graph TP=3: %zu ops (GEMV q8=%u q4=%u iq=%u, GDN steps=%u, FlashDecode=%u, fused MoE=%u, AllReduce=%u)\n",
                dec.graph.size(), hg[ops::OpKind::GemvQ8Dp4a], hg[ops::OpKind::GemvQ4Dot8], hg[ops::OpKind::GemvIqPacked],
                hg[ops::OpKind::GdnRecurrentStep], hg[ops::OpKind::FlashDecodeWave64], hg[ops::OpKind::MoeFusedExperts],
                hg[ops::OpKind::AllReduce]);
    VJ_CHECK(hg[ops::OpKind::GdnRecurrentStep] == h.n_gdn * 3 && hg[ops::OpKind::FlashDecodeWave64] == h.n_attn * 3);

    for (const char* t : {"blk.0.ssm_alpha.weight", "blk.3.attn_k.weight", "blk.7.attn_output.weight", "blk.5.ssm_out.weight",
                          "output.weight", "token_embd.weight", "blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight"}) {
        std::vector<float> nat, ref;
        VJ_CHECK(m.repack_shard_for_test(f, t, 0, 64, nat, ref, &err));
        const TensorPlan* p = m.find_plan(t);
        const double c = cosine(ref, nat);
        std::printf("  real repack %-27s %-7s -> %-12s%s rank0 rows=64 cos=%.6f\n", t, ggml_type_name(static_cast<uint32_t>(p->disk)),
                    quant::native_format_name(p->format), p->fwht64 ? "+FWHT" : "     ", c);
        VJ_CHECK(c >= (p->format == quant::NativeFormat::VJ_Q8_DP4A ? 0.9999 : 0.995));
    }

    if (draft) {
        LoadOptions o = mi50s(3, 32768);
        o.draft_gguf = draft;
        rep = m.load(f, o);
        VJ_CHECK(rep.ok && m.draft_source() == DraftSource::ExternalDFlash);
        const auto& d = m.draft_hparams();
        std::printf("  draft: arch=%s layers=%u block=%u heads=%u/%u head_dim=%u ff=%u taps=%zu mask=%llu\n", d.arch.c_str(), d.n_layer,
                    d.block_size, d.n_head, d.n_head_kv, d.head_dim, d.n_ff, d.target_layers.size(), (unsigned long long)d.mask_token);
        print_report("TP=3 + DFlash draft", m, rep);
        for (int i = 0; i < 40; ++i) m.governor().observe(8, 7);
        auto dr = m.draft_step({1, 2, 3}, 8);
        check_graph(m, dr.graph);
        VJ_CHECK(dr.draft_tokens > 0 && !dr.graph.empty());
        std::printf("  draft_step: K*=%u -> %u draft tokens, %zu ops (alpha=%.2f)\n", m.governor().k_star(), dr.draft_tokens,
                    dr.graph.size(), m.governor().alpha());
    }
    std::printf("[PASS] real Ornith GGUF dry-run plan, sharding, graphs and repack parity\n");
}

} // namespace

int main() {
    test_synthetic();
    const char* real = std::getenv("VEGAJET_ORNITH_GGUF");
    if (real && *real && access(real, R_OK) == 0) test_real(real, std::getenv("VEGAJET_ORNITH_DRAFT_GGUF"));
    else std::printf("[SKIP] VEGAJET_ORNITH_GGUF not set or unreadable; real-model checks skipped\n");
    return 0;
}
