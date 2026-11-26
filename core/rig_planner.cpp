// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/rig_planner.hpp"
#include <stdexcept>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <map>
#include <set>

namespace vegajet::rig {

namespace {

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

bool contains(const std::string& s, const char* needle) { return s.find(needle) != std::string::npos; }

uint64_t ceil_div(uint64_t a, uint64_t b) { return b ? (a + b - 1) / b : 0; }

uint64_t bytes_at_bpw(uint64_t elems, double bpw) { return static_cast<uint64_t>(static_cast<double>(elems) * bpw / 8.0 + 0.5); }

uint64_t disk_bytes_of(const TensorEntry& t) {
    const uint64_t b = ggml_row_bytes(t.type, t.elems);
    return b ? b : bytes_at_bpw(t.elems, quant::disk_bits_per_weight(static_cast<quant::DiskQuantType>(t.type)));
}

std::string gib(double bytes) {
    char b[32];
    std::snprintf(b, sizeof b, "%.2f", bytes / kGiB);
    return b;
}

// Max over a scalar or a numeric array (per-layer head_count_kv in some files).
bool arch_u64_or_max(const GgufFile& f, const std::string& suffix, uint64_t& out) {
    if (f.get_arch_u64(suffix, out)) return true;
    const GgufValue* v = f.find(f.architecture() + "." + suffix);
    if (!v || !v->is_array() || v->arr_num.empty()) return false;
    out = static_cast<uint64_t>(*std::max_element(v->arr_num.begin(), v->arr_num.end()));
    return true;
}

} // namespace

const char* kind_name(Kind k) {
    switch (k) {
        case Kind::Embed: return "embed";
        case Kind::Output: return "output";
        case Kind::Dense: return "dense";
        case Kind::RoutedExpert: return "routed-expert";
        case Kind::SharedExpert: return "shared-expert";
        case Kind::Router: return "router";
        case Kind::Norm: return "norm";
    }
    return "?";
}

Kind classify_tensor(const std::string& name, uint32_t n_dims) {
    if (name == "token_embd.weight") return Kind::Embed;
    if (name == "output.weight") return Kind::Output;
    if (n_dims <= 1 || contains(name, "norm") || contains(name, ".bias")) return Kind::Norm;
    if (contains(name, "_exps")) return Kind::RoutedExpert;
    if (contains(name, "_shexp")) return Kind::SharedExpert;
    if (contains(name, "ffn_gate_inp")) return Kind::Router;
    return Kind::Dense;
}

const char* decision_name(Decision d) {
    switch (d) {
        case Decision::Relayout: return "RELAYOUT (lossless)";
        case Decision::Repack: return "REPACK";
        case Decision::KeepCrossover: return "KEEP (crossover law)";
        case Decision::KeepBudget: return "KEEP (VRAM budget)";
        case Decision::KeepNoHook: return "KEEP (no hook)";
        case Decision::KeepFloat: return "KEEP (float vectors)";
    }
    return "?";
}

uint64_t ModelSummary::total_params() const {
    uint64_t n = 0;
    for (const auto& t : tensors) n += t.elems;
    return n;
}

uint64_t ModelSummary::disk_bytes() const {
    uint64_t n = 0;
    for (const auto& t : tensors) n += disk_bytes_of(t);
    return n;
}

bool summarize_gguf(const GgufFile& f, ModelSummary& m, std::string* err) {
    m = ModelSummary{};
    m.arch = f.architecture();
    if (m.arch.empty()) {
        if (err) *err = "general.architecture missing";
        return false;
    }
    f.get_arch_u64("block_count", m.n_layer);
    f.get_arch_u64("embedding_length", m.n_embd);
    arch_u64_or_max(f, "attention.head_count", m.n_head);
    arch_u64_or_max(f, "attention.head_count_kv", m.n_head_kv);
    if (!f.get_arch_u64("attention.key_length", m.head_dim) && m.n_head) m.head_dim = m.n_embd / m.n_head;
    f.get_arch_u64("context_length", m.n_ctx_train);
    f.get_arch_u64("expert_count", m.n_expert);
    f.get_arch_u64("expert_used_count", m.n_expert_used);
    if (m.n_head_kv == 0) m.n_head_kv = m.n_head;

    std::set<std::string> kv_layers;
    for (const auto& t : f.tensors()) {
        m.tensors.push_back({t.name, t.type, t.n_dims, t.n_elems()});
        if (t.name.rfind("blk.", 0) == 0 &&
            (contains(t.name, ".attn_k.") || contains(t.name, ".attn_qkv.") || contains(t.name, ".attn_kv_a_mqa.")))
            kv_layers.insert(t.name.substr(0, t.name.find('.', 4)));
    }
    uint64_t interval = 0;
    if (f.get_arch_u64("full_attention_interval", interval) && interval > 0) m.n_kv_layers = m.n_layer / interval;
    else m.n_kv_layers = kv_layers.empty() ? m.n_layer : kv_layers.size();

    uint64_t state = 0, inner = 0, conv = 0, groups = 1;
    if (f.get_arch_u64("ssm.state_size", state) && f.get_arch_u64("ssm.inner_size", inner)) {
        f.get_arch_u64("ssm.conv_kernel", conv);
        f.get_arch_u64("ssm.group_count", groups);
        const uint64_t ssm_layers = m.n_layer > m.n_kv_layers ? m.n_layer - m.n_kv_layers : 0;
        const uint64_t conv_dim = inner + 2 * groups * state;
        m.ssm_state_bytes = ssm_layers * (inner * state * 4 + (conv ? conv - 1 : 0) * conv_dim * 4);
    }
    // Multi-head latent attention: one compressed latent + rope key per token and layer, shared by all heads.
    uint64_t lora = 0, rope = 0;
    if (f.get_arch_u64("attention.kv_lora_rank", lora) && lora > 0) {
        f.get_arch_u64("rope.dimension_count", rope);
        m.n_head_kv = 1;
        m.head_dim = (lora + rope + 1) / 2;  // 2 * head_dim elements per layer and token (K slot + V slot)
    }
    return true;
}

double effective_bw_gbs(double inst_per_byte, const RigSpec& rig) {
    quant::GpuRooflineInfo g;
    g.n_cu = rig.n_cu;
    g.bw_hbm_gbs = rig.bw_gbs;
    const double istar = g.i_star(1.0);
    return rig.bw_gbs / std::max(1.0, inst_per_byte / istar);
}

namespace {

struct Totals {
    uint64_t dense = 0, experts = 0, embed = 0, output = 0, norms = 0;
};

Totals totals_for(const ModelSummary& m, const std::map<uint32_t, TypeDecision>& dec) {
    Totals t;
    for (const auto& e : m.tensors) {
        const Kind k = classify_tensor(e.name, e.n_dims);
        uint64_t b = disk_bytes_of(e);
        if (k != Kind::Norm) {
            const auto it = dec.find(e.type);
            if (it != dec.end() && (it->second.decision == Decision::Repack || it->second.decision == Decision::Relayout))
                b = bytes_at_bpw(e.elems, it->second.bpw_new);
        }
        switch (k) {
            case Kind::Embed: t.embed += b; break;
            case Kind::Output: t.output += b; break;
            case Kind::RoutedExpert: t.experts += b; break;
            case Kind::Norm: t.norms += b; break;
            default: t.dense += b; break;
        }
    }
    return t;
}

std::vector<GpuBreakdown> breakdown(const ModelSummary& m, const RigSpec& rig, const Totals& t, uint64_t kv_per_tok,
                                    uint64_t ctx, double expert_fraction) {
    const uint32_t n = std::max<uint32_t>(1, rig.gpus);
    const bool tied = t.output == 0;
    std::vector<GpuBreakdown> g(n);
    for (uint32_t i = 0; i < n; ++i) {
        auto& b = g[i];
        b.vram = static_cast<uint64_t>(rig.vram_gib) << 30;
        b.dense = ceil_div(t.dense, n);
        b.experts = static_cast<uint64_t>(static_cast<double>(ceil_div(t.experts, n)) * expert_fraction + 0.5);
        b.embed_head = tied ? ceil_div(t.embed, n) : ceil_div(t.output, n) + (i == 0 ? t.embed : 0);
        b.norms = t.norms;
        b.kv = kv_per_tok * ctx;
        b.state = ceil_div(m.ssm_state_bytes, n);
        b.reserve = std::max<uint64_t>(rig.reserve_bytes, 2ull << 30);
    }
    return g;
}

int64_t min_free(const std::vector<GpuBreakdown>& g) {
    int64_t f = INT64_MAX;
    for (const auto& b : g) f = std::min(f, b.free_bytes());
    return f;
}

} // namespace

RigPlan plan_rig(const ModelSummary& m, const RigSpec& rig_in) {
    RigSpec rig = rig_in;
    if (rig.gpus > 256) throw std::invalid_argument("rig planner supports at most 256 ranks");
    rig.gpus = std::max<uint32_t>(1, rig.gpus);
    RigPlan p;
    quant::GpuRooflineInfo g;
    g.n_cu = rig.n_cu;
    g.bw_hbm_gbs = rig.bw_gbs;
    p.i_star = g.i_star(1.0);

    // ---- per disk type: target, crossover law
    std::map<uint32_t, TypeDecision> dec;
    for (const auto& e : m.tensors) {
        if (classify_tensor(e.name, e.n_dims) == Kind::Norm) continue;
        auto& d = dec[e.type];
        d.disk_type = e.type;
        d.tensors += 1;
        d.elems += e.elems;
        d.disk_bytes += disk_bytes_of(e);
    }
    const auto& reg = quant::UnpackerRegistry::instance();
    for (auto& kv : dec) {
        auto& d = kv.second;
        const auto disk = static_cast<quant::DiskQuantType>(d.disk_type);
        d.bpw_old = d.bpw_new = quant::disk_bits_per_weight(disk);
        d.bw_old = d.bw_new = effective_bw_gbs(quant::VramBudgetPlanner::format_inst_per_byte(disk), rig);
        if (d.disk_type == static_cast<uint32_t>(GgmlType::F32)) { d.decision = Decision::KeepFloat; continue; }
        const quant::UnpackerHook* h = reg.find(disk);
        if (!h) { d.decision = Decision::KeepNoHook; continue; }
        d.target = h->target;
        d.lossless = h->lossless;
        d.bpw_new = quant::native_bits_per_weight(h->target, disk);
        d.bw_new = effective_bw_gbs(quant::VramBudgetPlanner::native_inst_per_byte(h->target, disk), rig);
        if (h->target == quant::NativeFormat::VJ_IQ_PACKED) d.decision = Decision::Relayout;
        else if (h->lossless && d.bpw_new <= d.bpw_old * 1.0001)
            d.decision = d.bw_new >= d.bw_old ? Decision::Relayout : Decision::KeepCrossover;
        else d.decision = (d.bw_new / d.bw_old > d.bpw_new / d.bpw_old) ? Decision::Repack : Decision::KeepCrossover;
    }

    // ---- KV per token per GPU
    const uint64_t kv_bytes = rig.kv_bits == 8 ? 1 : 2;
    const uint64_t heads_per_gpu = m.n_head_kv ? ceil_div(m.n_head_kv, rig.gpus) : 0;
    p.kv_bytes_per_token = 2 * m.n_kv_layers * heads_per_gpu * m.head_dim * kv_bytes;
    const uint64_t ctx = rig.ctx;

    // ---- budget: revert byte-growing repacks with the lowest ms-saved per extra byte until it fits
    Totals t = totals_for(m, dec);
    auto gpus = breakdown(m, rig, t, p.kv_bytes_per_token, ctx, 1.0);
    if (min_free(gpus) < 0) {
        std::vector<std::pair<double, uint32_t>> roi;
        const double read_frac = m.n_expert ? static_cast<double>(std::max<uint64_t>(1, m.n_expert_used)) / m.n_expert : 1.0;
        for (const auto& kv : dec) {
            const auto& d = kv.second;
            if (d.decision != Decision::Repack || d.bpw_new <= d.bpw_old) continue;
            const double old_b = static_cast<double>(d.elems) * d.bpw_old / 8, new_b = static_cast<double>(d.elems) * d.bpw_new / 8;
            const double saved_ms = (old_b / d.bw_old - new_b / d.bw_new) * 1e-6 * read_frac;
            roi.push_back({saved_ms / (new_b - old_b), kv.first});
        }
        std::sort(roi.begin(), roi.end());
        for (const auto& r : roi) {
            dec[r.second].decision = Decision::KeepBudget;
            t = totals_for(m, dec);
            gpus = breakdown(m, rig, t, p.kv_bytes_per_token, ctx, 1.0);
            if (min_free(gpus) >= 0) break;
        }
    }
    for (auto& kv : dec) {
        auto& d = kv.second;
        const bool applied = d.decision == Decision::Repack || d.decision == Decision::Relayout;
        d.final_bytes = applied ? bytes_at_bpw(d.elems, d.bpw_new) : d.disk_bytes;
        p.decisions.push_back(d);
    }

    // ---- MoE host offload
    if (min_free(gpus) < 0 && t.experts > 0) {
        auto none = breakdown(m, rig, t, p.kv_bytes_per_token, ctx, 0.0);
        const int64_t room = min_free(none);
        if (room > 0) {
            const double per_gpu = static_cast<double>(ceil_div(t.experts, rig.gpus));
            const double frac = std::min(1.0, static_cast<double>(room) / per_gpu);
            p.moe_offload_required = true;
            p.expert_resident_fraction = frac;
            p.host_expert_bytes = static_cast<uint64_t>(static_cast<double>(t.experts) * (1.0 - frac) + 0.5);
            gpus = breakdown(m, rig, t, p.kv_bytes_per_token, ctx, frac);
            // Rounding can leave a few bytes over: shave them off the resident experts.
            for (auto& b : gpus)
                if (b.free_bytes() < 0) b.experts -= static_cast<uint64_t>(-b.free_bytes());
        }
    }
    p.gpus = gpus;
    const int64_t mf = min_free(gpus);
    p.fits = mf >= 0;
    p.shortfall_bytes = p.fits ? 0 : -mf;

    // ---- context headroom (at the chosen expert residency)
    if (p.fits) {
        uint64_t room = UINT64_MAX;
        for (const auto& b : gpus) room = std::min(room, static_cast<uint64_t>(b.free_bytes()) + b.kv);
        uint64_t cap = m.n_ctx_train ? m.n_ctx_train : (1ull << 20);
        p.max_ctx = p.kv_bytes_per_token ? std::min<uint64_t>(cap, room / p.kv_bytes_per_token) : cap;
    }
    char buf[160];
    if (!p.fits)
        std::snprintf(buf, sizeof buf, "DOES NOT FIT (short %s GiB per GPU)", gib(static_cast<double>(p.shortfall_bytes)).c_str());
    else if (p.moe_offload_required)
        std::snprintf(buf, sizeof buf, "FITS WITH HOST-RAM MOE OFFLOAD (%s GiB experts in pinned host RAM, %.0f%% resident)",
                      gib(static_cast<double>(p.host_expert_bytes)).c_str(), p.expert_resident_fraction * 100.0);
    else
        std::snprintf(buf, sizeof buf, "FITS");
    p.verdict = buf;
    return p;
}

std::string format_rig_plan(const ModelSummary& m, const RigSpec& rig, const RigPlan& p) {
    std::string o;
    char b[512];
    std::snprintf(b, sizeof b,
                  "VegaJet rig plan: %s, %.2f B params, %s GiB on disk\n"
                  "Rig: %u x %u GiB (%u CU, %.0f GB/s), TP=%u, ctx %" PRIu64 ", KV %s, reserve %s GiB/GPU, I* = %.2f inst/B\n\n",
                  m.arch.c_str(), static_cast<double>(m.total_params()) / 1e9, gib(static_cast<double>(m.disk_bytes())).c_str(),
                  rig.gpus, rig.vram_gib, rig.n_cu, rig.bw_gbs, rig.gpus, rig.ctx, rig.kv_bits == 8 ? "q8" : "f16",
                  gib(static_cast<double>(std::max<uint64_t>(rig.reserve_bytes, 2ull << 30))).c_str(), p.i_star);
    o += b;
    o += "Repacking (crossover law: repack iff bw_new / bw_old > bpw_new / bpw_old)\n";
    std::snprintf(b, sizeof b, "  %-8s %7s %9s  %-13s %13s %17s %15s  %s\n", "type", "tensors", "disk GiB", "target",
                  "bpw old->new", "bw old->new GB/s", "bw | bpw ratio", "decision");
    o += b;
    for (const auto& d : p.decisions) {
        std::snprintf(b, sizeof b, "  %-8s %7" PRIu64 " %9s  %-13s %6.2f->%-6.2f %8.0f->%-8.0f %6.3f | %-6.3f  %s\n",
                      ggml_type_name(d.disk_type), d.tensors, gib(static_cast<double>(d.disk_bytes)).c_str(),
                      d.decision == Decision::KeepFloat || d.decision == Decision::KeepNoHook ? "-" : quant::native_format_name(d.target),
                      d.bpw_old, d.bpw_new, d.bw_old, d.bw_new, d.bw_old > 0 ? d.bw_new / d.bw_old : 0.0,
                      d.bpw_old > 0 ? d.bpw_new / d.bpw_old : 0.0, decision_name(d.decision));
        o += b;
    }
    o += "\nPer-GPU VRAM (GiB)\n";
    std::snprintf(b, sizeof b, "  %-4s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s\n", "gpu", "vram", "dense", "experts",
                  "emb/head", "norms", "kv", "state", "reserve", "used", "free");
    o += b;
    for (size_t i = 0; i < p.gpus.size(); ++i) {
        const auto& g = p.gpus[i];
        std::snprintf(b, sizeof b, "  %-4zu %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s\n", i, gib(static_cast<double>(g.vram)).c_str(),
                      gib(static_cast<double>(g.dense)).c_str(), gib(static_cast<double>(g.experts)).c_str(),
                      gib(static_cast<double>(g.embed_head)).c_str(), gib(static_cast<double>(g.norms)).c_str(),
                      gib(static_cast<double>(g.kv)).c_str(), gib(static_cast<double>(g.state)).c_str(),
                      gib(static_cast<double>(g.reserve)).c_str(), gib(static_cast<double>(g.used())).c_str(),
                      gib(static_cast<double>(g.free_bytes())).c_str());
        o += b;
    }
    std::snprintf(b, sizeof b, "\nKV cache: %" PRIu64 " B/token per GPU (%" PRIu64 " KV layers); ", p.kv_bytes_per_token, m.n_kv_layers);
    o += b;
    if (p.fits) std::snprintf(b, sizeof b, "max context on this rig: %" PRIu64 " tokens\n", p.max_ctx);
    else std::snprintf(b, sizeof b, "no context fits\n");
    o += b;
    if (m.n_ctx_train && rig.ctx > m.n_ctx_train) {
        std::snprintf(b, sizeof b, "Note: --ctx %" PRIu64 " exceeds the training context (%" PRIu64 "); KV sized for %" PRIu64 " anyway\n",
                      rig.ctx, m.n_ctx_train, rig.ctx);
        o += b;
    }
    if (m.n_expert) {
        if (p.moe_offload_required)
            std::snprintf(b, sizeof b, "MoE: %" PRIu64 " experts (%" PRIu64 " used); host-RAM offload REQUIRED: %s GiB, %.1f%% of routed experts resident\n",
                          m.n_expert, m.n_expert_used, gib(static_cast<double>(p.host_expert_bytes)).c_str(), p.expert_resident_fraction * 100.0);
        else
            std::snprintf(b, sizeof b, "MoE: %" PRIu64 " experts (%" PRIu64 " used); all routed experts resident, no host offload\n",
                          m.n_expert, m.n_expert_used);
        o += b;
    }
    o += "Verdict: " + p.verdict + "\n";
    return o;
}

std::vector<RigSpec> standard_rigs(const RigSpec& base) {
    const uint32_t rigs[][2] = {{1, 16}, {2, 16}, {2, 32}, {3, 32}, {4, 32}, {8, 32}};
    std::vector<RigSpec> out;
    for (const auto& r : rigs) {
        RigSpec s = base;
        s.gpus = r[0];
        s.vram_gib = r[1];
        out.push_back(s);
    }
    return out;
}

std::string format_rig_matrix(const ModelSummary& m, const std::vector<RigSpec>& rigs) {
    std::string o;
    char b[320];
    std::snprintf(b, sizeof b, "Rig matrix: %s, %.2f B params, %s GiB on disk, ctx %" PRIu64 "\n", m.arch.c_str(),
                  static_cast<double>(m.total_params()) / 1e9, gib(static_cast<double>(m.disk_bytes())).c_str(),
                  rigs.empty() ? 0 : rigs[0].ctx);
    o += b;
    std::snprintf(b, sizeof b, "  %-10s %10s %10s %10s  %s\n", "rig", "GPU0 used", "GPU0 free", "max ctx", "verdict");
    o += b;
    for (const auto& r : rigs) {
        const RigPlan p = plan_rig(m, r);
        char rig[24];
        std::snprintf(rig, sizeof rig, "%ux%uGiB", r.gpus, r.vram_gib);
        std::snprintf(b, sizeof b, "  %-10s %10s %10s %10" PRIu64 "  %s\n", rig, gib(static_cast<double>(p.gpus[0].used())).c_str(),
                      gib(static_cast<double>(p.gpus[0].free_bytes())).c_str(), p.max_ctx, p.verdict.c_str());
        o += b;
    }
    return o;
}

} // namespace vegajet::rig
