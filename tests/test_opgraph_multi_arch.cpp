// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Declarative OpGraph plans (ops/layer_plan) for three model families, N = 1..4 GPUs, decode / verify / prefill:
//   1. hybrid GDN + GQA MoE with a shared expert,
//   2. pure MoE (shared-less split-FF, and whole-expert parallel with 8 experts / top-2),
//   3. dense GQA.
// An independent validator checks shape chains, shard sums, kind selection, barrier placement (the only barrier-free
// edge is shared-expert gate/up after the router) and AllReduce placement (every row-parallel output is reduced before
// the next norm; N=1 has none). "No hardcoded dimensions" is checked twice: 300 random descriptors must validate, and
// ops/layer_plan.cpp may not contain an integer literal above 2.
#include "ops/layer_plan.hpp"
#include "tests/test_common.hpp"

#include <cctype>
#include <cstdio>
#include <fstream>
#include <map>
#include <random>
#include <sstream>

#ifndef VJ_LAYER_PLAN_SRC
#define VJ_LAYER_PLAN_SRC "ops/layer_plan.cpp"
#endif

using namespace vegajet;
using namespace vegajet::ops;

namespace {

uint32_t R(std::mt19937& g) { return static_cast<uint32_t>(g()); }

bool ends_with(const std::string& s, const char* suf) {
    const std::string t(suf);
    return s.size() >= t.size() && s.compare(s.size() - t.size(), t.size(), t) == 0;
}

struct Sums {
    uint64_t vocab = 0, dense_ff = 0, shared_ff = 0, expert_ff = 0, experts = 0, gdn_v = 0, q_heads = 0;
};

// Validates the plan and returns per-quantity sums over GPUs for layer `probe` (and the LM head).
Sums validate(const ModelDesc& m, const TpDesc& tp, const StepDesc& s, const StepPlan& plan, uint32_t probe) {
    const uint32_t N = tp.n_gpus();
    const uint32_t D = m.d_model;
    Sums sums;
    VJ_CHECK(plan.n_gpus == N);
    if (N == 1) VJ_CHECK(plan.count(OpKind::AllReduce) == 0);
    for (uint32_t g = 0; g < N; ++g) {
        const auto ops = plan.ops_for_gpu(g);
        VJ_CHECK(!ops.empty() && ops.front().op.kind == OpKind::Embed && ops.back().op.kind == OpKind::LmHead);
        bool partial = false;
        std::map<std::string, uint32_t> last_n;  // per layer: kind/tensor -> n / k of the latest op
        int32_t cur_layer = -2;
        for (size_t i = 0; i < ops.size(); ++i) {
            const OpDesc& d = ops[i].op;
            VJ_CHECK(d.m == s.m && d.n > 0 && d.split_k >= 1);
            if (is_projection(d.kind) || d.kind == OpKind::MoeFusedExperts) VJ_CHECK(d.k > 0);
            if (d.layer != cur_layer && d.kind != OpKind::AllReduce) { cur_layer = d.layer; last_n.clear(); }
            // Kind selection per step type.
            if (s.prefill) {
                VJ_CHECK(d.kind != OpKind::GemvQ8Dp4a && d.kind != OpKind::GemvQ4Dot8 && d.kind != OpKind::GemvIqPacked);
                VJ_CHECK(d.kind != OpKind::FlashDecodeWave64 && d.kind != OpKind::GdnRecurrentStep);
                if (is_projection(d.kind)) VJ_CHECK(d.split_k == 1);
            } else {
                VJ_CHECK(d.kind != OpKind::GemmPrefill && d.kind != OpKind::FlashPrefill && d.kind != OpKind::GdnChunkWy);
                if (is_projection(d.kind)) VJ_CHECK(d.kind == projection_kind(m.fmt_dense, false));
            }
            // Barrier placement.
            if (!ops[i].barrier) {
                VJ_CHECK(i > 0 && ops[i - 1].op.kind == OpKind::MoeRouterTopK);
                VJ_CHECK(is_projection(d.kind) && ends_with(d.tensor, "ffn_gate_up_shexp"));
            }
            if (d.kind == OpKind::MoeFusedExperts) VJ_CHECK(ops[i].barrier);
            // AllReduce placement.
            switch (d.kind) {
                case OpKind::RmsNorm:
                case OpKind::LmHead:
                    VJ_CHECK(!partial);
                    break;
                case OpKind::AllReduce:
                    VJ_CHECK(N > 1 && d.gpu == (1u << N) - 1u);
                    // A rank that owns no heads / rows for the stage contributes zeros: partial may be false.
                    if (d.aux == kArSumHidden) { VJ_CHECK(d.n == D); partial = false; }
                    else { VJ_CHECK(d.aux == kArFlashMerge && d.n == m.attn.n_q_heads * m.attn.head_dim); }
                    break;
                case OpKind::MoeFusedExperts:
                    VJ_CHECK(d.n == D);
                    if (N > 1) partial = true;
                    break;
                default:
                    if (is_projection(d.kind) &&
                        (ends_with(d.tensor, "attn_output") || ends_with(d.tensor, "ssm_out") ||
                         ends_with(d.tensor, "ffn_down") || ends_with(d.tensor, "ffn_down_shexp"))) {
                        VJ_CHECK(d.n == D);
                        if (N > 1) partial = true;
                    }
            }
            // Shape chains inside a layer.
            if (is_projection(d.kind)) {
                if (ends_with(d.tensor, "attn_output") && last_n.count("flash") &&
                    d.k != 0 && !(m.attn.n_kv_heads == 1 && N > 1))
                    VJ_CHECK(d.k == last_n["flash"]);
                if (ends_with(d.tensor, "ffn_down")) VJ_CHECK(d.k == last_n["swiglu"]);
                if (ends_with(d.tensor, "ffn_down_shexp")) VJ_CHECK(d.k == last_n["swiglu"]);
                if (ends_with(d.tensor, "ssm_out")) VJ_CHECK(d.k == last_n["recur"]);
                if (ends_with(d.tensor, "attn_qkv") || ends_with(d.tensor, "ssm_in") ||
                    ends_with(d.tensor, "ffn_gate_up") || ends_with(d.tensor, "ffn_gate_up_shexp"))
                    VJ_CHECK(d.k == D);
                last_n["proj_n"] = d.n;
            }
            if (d.kind == OpKind::SwiGlu) { VJ_CHECK(d.k == 2 * d.n && last_n["proj_n"] == d.k); last_n["swiglu"] = d.n; }
            if (d.kind == OpKind::FlashDecodeWave64 || d.kind == OpKind::FlashPrefill) {
                VJ_CHECK(d.n % m.attn.head_dim == 0);
                last_n["flash"] = d.n;
                if (d.layer == static_cast<int32_t>(probe) && !(m.attn.n_kv_heads == 1 && N > 1))
                    sums.q_heads += d.n / m.attn.head_dim;
            }
            if (d.kind == OpKind::GdnRecurrentStep || d.kind == OpKind::GdnChunkWy) {
                last_n["recur"] = d.n;
                if (d.layer == static_cast<int32_t>(probe)) sums.gdn_v += d.n;
            }
            if (d.layer == static_cast<int32_t>(probe)) {
                if (is_projection(d.kind) && ends_with(d.tensor, "ffn_down")) sums.dense_ff += d.k;
                if (is_projection(d.kind) && ends_with(d.tensor, "ffn_down_shexp")) sums.shared_ff += d.k;
                if (d.kind == OpKind::MoeFusedExperts) {
                    if (m.moe.shard == ExpertShard::SplitFf) sums.expert_ff += d.k;
                    else {
                        VJ_CHECK(d.k == m.moe.expert_ff);
                        const auto lb = d.tensor.find('['), colon = d.tensor.find(':'), rb = d.tensor.find(')');
                        VJ_CHECK(lb != std::string::npos && colon != std::string::npos && rb != std::string::npos);
                        sums.experts += std::stoull(d.tensor.substr(colon + 1, rb - colon - 1)) -
                                        std::stoull(d.tensor.substr(lb + 1, colon - lb - 1));
                    }
                }
            }
            if (d.kind == OpKind::LmHead) { VJ_CHECK(d.k == D); sums.vocab += d.n; }
        }
        VJ_CHECK(!partial);
    }
    return sums;
}

void check_sums(const ModelDesc& m, const Sums& s, uint32_t probe, uint32_t N) {
    VJ_CHECK(s.vocab == m.vocab);
    if (m.mixer[probe] == MixerKind::Gqa && !(m.attn.n_kv_heads == 1 && N > 1)) VJ_CHECK(s.q_heads == m.attn.n_q_heads);
    if (m.mixer[probe] == MixerKind::Gdn) VJ_CHECK(s.gdn_v == static_cast<uint64_t>(m.gdn.n_v_heads) * m.gdn.head_v_dim);
    if (m.ffn[probe] == FfnKind::Dense) VJ_CHECK(s.dense_ff == m.dense_ff);
    if (m.ffn[probe] == FfnKind::Moe) {
        VJ_CHECK(s.shared_ff == m.moe.shared_ff);
        if (m.moe.shard == ExpertShard::SplitFf) VJ_CHECK(s.expert_ff == m.moe.expert_ff);
        else VJ_CHECK(s.experts == m.moe.n_experts);
    }
}

ModelDesc hybrid_gdn_gqa_moe() {
    ModelDesc m;
    m.arch = "hybrid-gdn-gqa-moe";
    m.d_model = 2048; m.vocab = 151936;
    m.attn = {16, 2, 256};
    m.gdn = {16, 32, 128, 128, 4};
    m.moe = {128, 8, 512, 512, ExpertShard::SplitFf};
    for (uint32_t L = 0; L < 24; ++L) { m.mixer.push_back(L % 4 == 3 ? MixerKind::Gqa : MixerKind::Gdn); m.ffn.push_back(FfnKind::Moe); }
    return m;
}
ModelDesc pure_moe_split_ff() {
    ModelDesc m;
    m.arch = "pure-moe";
    m.d_model = 2048; m.vocab = 151936;
    m.attn = {32, 4, 128};
    m.moe = {128, 8, 768, 0, ExpertShard::SplitFf};
    m.mixer.assign(48, MixerKind::Gqa); m.ffn.assign(48, FfnKind::Moe);
    return m;
}
ModelDesc pure_moe_whole_experts() {
    ModelDesc m;
    m.arch = "pure-moe-ep";
    m.d_model = 4096; m.vocab = 32000;
    m.attn = {32, 8, 128};
    m.moe = {8, 2, 14336, 0, ExpertShard::WholeExperts};
    m.fmt_expert = quant::NativeFormat::VJ_Q4_DOT8;
    m.mixer.assign(32, MixerKind::Gqa); m.ffn.assign(32, FfnKind::Moe);
    return m;
}
ModelDesc dense_gqa() {
    ModelDesc m;
    m.arch = "dense-gqa";
    m.d_model = 4096; m.vocab = 128256; m.dense_ff = 14336;
    m.attn = {32, 8, 128};
    m.fmt_dense = quant::NativeFormat::VJ_Q8_DP4A;
    m.mixer.assign(32, MixerKind::Gqa); m.ffn.assign(32, FfnKind::Dense);
    return m;
}

ModelDesc random_desc(std::mt19937& rng) {
    auto pick = [&](std::initializer_list<uint32_t> v) { return *(v.begin() + R(rng) % v.size()); };
    ModelDesc m;
    const uint32_t family = R(rng) % 3;
    m.arch = "fuzz";
    m.d_model = pick({512, 768, 1024, 1536, 2048, 2560, 3072, 4096, 5120, 7168});
    m.vocab = 1000 + R(rng) % 300000;
    const uint32_t nkv = pick({1, 2, 4, 8});
    m.attn = {nkv * pick({1, 2, 4, 6, 8, 16}), nkv, pick({64, 96, 128, 256})};
    const uint32_t kg = pick({4, 8, 16, 32});
    m.gdn = {kg, kg * pick({1, 2, 4}), pick({64, 128}), pick({64, 128}), pick({2, 3, 4})};
    const uint32_t ne = pick({8, 16, 64, 128, 256, 512});
    m.moe = {ne, std::min<uint32_t>(ne, pick({1, 2, 4, 6, 8, 10})), 32 * (1 + R(rng) % 64),
             (R(rng) % 2) ? 32u * (1 + R(rng) % 64) : 0u, (R(rng) % 2) ? ExpertShard::SplitFf : ExpertShard::WholeExperts};
    m.dense_ff = 32 * (1 + R(rng) % 600) + (R(rng) % 2) * 7;  // ragged widths too
    const uint32_t layers = 1 + R(rng) % 12;
    for (uint32_t L = 0; L < layers; ++L) {
        m.mixer.push_back(family == 0 && (R(rng) % 4) ? MixerKind::Gdn : MixerKind::Gqa);
        m.ffn.push_back(family == 2 ? FfnKind::Dense : FfnKind::Moe);
    }
    const quant::NativeFormat fmts[] = {quant::NativeFormat::VJ_Q8_DP4A, quant::NativeFormat::VJ_Q4_DOT8,
                                        quant::NativeFormat::VJ_IQ_PACKED};
    m.fmt_dense = fmts[R(rng) % 3];
    return m;
}

TpDesc tp_for(uint32_t n, bool asym, bool mi60 = false) {
    TpDesc tp;
    for (uint32_t g = 0; g < n; ++g) {
        tp.bw_gbs.push_back(asym ? (g == 0 ? 840.0 : 972.0) : 900.0);
        tp.n_cu.push_back(mi60 ? 64 : 60);
    }
    return tp;
}

// Integer literals > 2 outside comments / strings / char literals.
std::vector<std::string> scan_literals(const std::string& src) {
    std::string code;
    for (size_t i = 0; i < src.size();) {
        if (src.compare(i, 2, "//") == 0) { while (i < src.size() && src[i] != '\n') ++i; continue; }
        if (src.compare(i, 2, "/*") == 0) { const size_t e = src.find("*/", i + 2); i = e == std::string::npos ? src.size() : e + 2; continue; }
        if (src[i] == '"' || src[i] == '\'') {
            const char q = src[i++];
            while (i < src.size() && src[i] != q) i += src[i] == '\\' ? 2 : 1;
            ++i;
            code += ' ';
            continue;
        }
        code += src[i++];
    }
    std::vector<std::string> bad;
    for (size_t i = 0; i < code.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(code[i]))) continue;
        if (i > 0 && (std::isalnum(static_cast<unsigned char>(code[i - 1])) || code[i - 1] == '_' || code[i - 1] == '.')) {
            while (i < code.size() && std::isalnum(static_cast<unsigned char>(code[i]))) ++i;
            continue;
        }
        size_t j = i;
        while (j < code.size() && (std::isalnum(static_cast<unsigned char>(code[j])) || code[j] == '.')) ++j;
        const std::string tok = code.substr(i, j - i);
        if (std::strtod(tok.c_str(), nullptr) > 2.0) bad.push_back(tok);
        i = j;
    }
    return bad;
}

} // namespace

int main() {
    struct Fam { const char* name; ModelDesc m; };
    const Fam fams[] = {
        {"hybrid GDN+GQA MoE", hybrid_gdn_gqa_moe()},
        {"pure MoE (split-FF)", pure_moe_split_ff()},
        {"pure MoE (whole experts)", pure_moe_whole_experts()},
        {"dense GQA", dense_gqa()},
    };
    uint32_t plans = 0;
    for (const auto& f : fams) {
        const uint32_t probe = f.m.n_layers() - 1;  // a GQA layer in every family
        for (uint32_t n = 1; n <= 4; ++n) {
            for (bool asym : {false, true}) {
                for (bool mi60 : {false, true}) {
                    const TpDesc tp = tp_for(n, asym, mi60);
                    for (const StepDesc s : {StepDesc{1, false, 4096}, StepDesc{8, false, 16384}, StepDesc{512, true, 512}}) {
                        StepPlan plan;
                        std::string err;
                        VJ_CHECK(build_step_plan(f.m, tp, s, plan, &err));
                        const Sums sums = validate(f.m, tp, s, plan, probe);
                        check_sums(f.m, sums, probe, n);
                        if (f.m.mixer[0] == MixerKind::Gdn) check_sums(f.m, validate(f.m, tp, s, plan, 0), 0, n);
                        ++plans;
                    }
                }
            }
        }
    }

    // Exact per-layer histograms for decode (N=2).
    {
        StepPlan p;
        std::string err;
        const auto m = dense_gqa();
        VJ_CHECK(build_step_plan(m, tp_for(2, true), {1, false, 4096}, p, &err));
        const uint32_t L = m.n_layers();
        VJ_CHECK(p.count(OpKind::AllReduce) == 2 * L);
        VJ_CHECK(p.count(OpKind::RmsNorm) == 2 * (2 * L + 1));
        VJ_CHECK(p.count(OpKind::GemvQ8Dp4a) == 2 * 4 * L);
        VJ_CHECK(p.count(OpKind::FlashDecodeWave64) == 2 * L && p.count(OpKind::SwiGlu) == 2 * L);
        VJ_CHECK(p.count(OpKind::LmHead) == 2 && p.count(OpKind::Embed) == 2);
        std::printf("  dense GQA decode N=2: %zu ops/step (%u per GPU per layer + head)\n", p.ops.size(),
                    static_cast<uint32_t>((p.ops.size() - 6 - 2 * L) / (2 * L)));
    }
    {
        StepPlan p;
        std::string err;
        const auto m = hybrid_gdn_gqa_moe();
        VJ_CHECK(build_step_plan(m, tp_for(2, true), {1, false, 4096}, p, &err));
        uint32_t gdn = 0;
        for (auto k : m.mixer) gdn += k == MixerKind::Gdn;
        VJ_CHECK(p.count(OpKind::GdnRecurrentStep) == 2 * gdn && p.count(OpKind::GdnConv1d) == 2 * gdn);
        VJ_CHECK(p.count(OpKind::MoeRouterTopK) == 2 * m.n_layers() && p.count(OpKind::MoeFusedExperts) == 2 * m.n_layers());
        uint32_t free_edges = 0;
        for (const auto& o : p.ops) free_edges += !o.barrier;
        VJ_CHECK(free_edges == 2 * m.n_layers());  // router || shared gate/up, once per layer per GPU
        // N=1 removes every AllReduce and nothing else changes kind.
        StepPlan p1;
        VJ_CHECK(build_step_plan(m, tp_for(1, false), {1, false, 4096}, p1, &err));
        VJ_CHECK(p1.count(OpKind::AllReduce) == 0 && p1.count(OpKind::MoeFusedExperts) == m.n_layers());
    }
    // Sequence-sharded single-KV attention emits the flash merge AllReduce.
    {
        auto m = pure_moe_split_ff();
        m.attn = {16, 1, 128};
        StepPlan p;
        std::string err;
        VJ_CHECK(build_step_plan(m, tp_for(2, true), {1, false, 8192}, p, &err));
        uint32_t merges = 0;
        for (const auto& o : p.ops) merges += o.op.kind == OpKind::AllReduce && o.op.aux == kArFlashMerge;
        VJ_CHECK(merges == m.n_layers());
        validate(m, tp_for(2, true), {1, false, 8192}, p, 0);
    }

    // Invalid descriptors are rejected, not asserted.
    {
        std::string err;
        StepPlan p;
        auto bad = dense_gqa(); bad.attn.n_kv_heads = 5;
        VJ_CHECK(!build_step_plan(bad, tp_for(2, false), {}, p, &err) && !err.empty());
        bad = pure_moe_split_ff(); bad.moe.top_k = bad.moe.n_experts + 1;
        VJ_CHECK(!build_step_plan(bad, tp_for(2, false), {}, p, &err));
        bad = dense_gqa(); bad.ffn.pop_back();
        VJ_CHECK(!build_step_plan(bad, tp_for(2, false), {}, p, &err));
        VJ_CHECK(!build_step_plan(dense_gqa(), TpDesc{}, {}, p, &err));
        TpDesc t = tp_for(2, false); t.n_cu.pop_back();
        VJ_CHECK(!build_step_plan(dense_gqa(), t, {}, p, &err));
    }

    // Randomized descriptors: no dimension is baked into the planner.
    std::mt19937 rng(20261124);
    uint32_t fuzz = 0;
    for (int it = 0; it < 300; ++it) {
        const ModelDesc m = random_desc(rng);
        const uint32_t n = 1 + R(rng) % 4;
        TpDesc tp;
        for (uint32_t g = 0; g < n; ++g) { tp.bw_gbs.push_back(700.0 + R(rng) % 400); tp.n_cu.push_back((R(rng) % 2) ? 60 : 64); }
        const StepDesc s = (R(rng) % 3 == 0) ? StepDesc{256, true, 256} : StepDesc{1 + R(rng) % 16, false, 64 + R(rng) % 65536};
        StepPlan plan;
        std::string err;
        VJ_CHECK(build_step_plan(m, tp, s, plan, &err));
        for (uint32_t probe = 0; probe < m.n_layers(); ++probe) check_sums(m, validate(m, tp, s, plan, probe), probe, n);
        ++fuzz;
    }

    // Source scan: ops/layer_plan.cpp carries no model dimensions.
    std::ifstream src(VJ_LAYER_PLAN_SRC);
    VJ_CHECK(src.good());
    std::stringstream ss;
    ss << src.rdbuf();
    const auto lits = scan_literals(ss.str());
    for (const auto& l : lits) std::fprintf(stderr, "  hardcoded literal in layer_plan.cpp: %s\n", l.c_str());
    VJ_CHECK(lits.empty());
    // The scanner itself must catch a planted dimension.
    VJ_CHECK(scan_literals("int d = 3072; // 4096\nconst char* s = \"248320\";").size() == 1);

    std::printf("test_opgraph_multi_arch: %u family plans (4 families x N=1..4 x sym/asym x 60/64 CU x decode/verify/prefill), "
                "%u random descriptors, 0 hardcoded literals\n", plans, fuzz);
    std::printf("PASS\n");
    return 0;
}
