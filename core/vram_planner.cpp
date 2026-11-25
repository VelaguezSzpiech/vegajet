// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/vram_planner.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <sstream>

namespace vegajet::vram {

const char* fmt_name(Fmt f) {
    static const char* const k[] = {"IQ2_S", "IQ3_XXS", "Q4_K", "Q6_K", "VJ_Q4_A8", "VJ_Q4_DOT8", "VJ_DOT8_EXACT",
                                    "VJ_Q8_DP4A", "F16"};
    return f < Fmt::Count ? k[static_cast<size_t>(f)] : "?";
}

double fmt_bpw(Fmt f) {
    switch (f) {
        case Fmt::IQ2_S: return 82.0 * 8 / 256;     // 2.5625
        case Fmt::IQ3_XXS: return 98.0 * 8 / 256;   // 3.0625
        case Fmt::Q4_K: return 4.5;
        case Fmt::Q6_K: return 6.5625;
        case Fmt::VJ_Q4_A8: return 5.0;
        case Fmt::VJ_Q4_DOT8: return 4.5;
        case Fmt::VJ_DOT8_EXACT: return 4.25;
        case Fmt::VJ_Q8_DP4A: return 8.5;
        case Fmt::F16: return 16.0;
        default: return 0.0;
    }
}

const char* role_name(Role r) {
    static const char* const k[] = {"dense", "router", "shared_expert", "routed_gate_up", "routed_down", "lm_head", "embed"};
    return k[static_cast<size_t>(r)];
}

const char* verdict_name(Verdict v) {
    static const char* const k[] = {"promoted", "kept", "rejected:ratio", "rejected:ban", "rejected:budget"};
    return k[static_cast<size_t>(v)];
}

// Planning defaults for one MI50 (60 CU, ~840 GB/s measured streaming ceiling). The M=1 values encode the
// structure seen on hardware: plain 4/8-bit GEMVs stream near the ceiling, while sub-4-bit codebook decode is
// VALU/LDS-bound at M=1. They are placeholders: replace them with per-format `vegajet bench` measurements.
BandwidthTable BandwidthTable::gfx906_defaults() {
    return BandwidthTable({
        {Fmt::IQ2_S, 1, 560}, {Fmt::IQ2_S, 16, 600}, {Fmt::IQ2_S, 1u << 30, 420},
        {Fmt::IQ3_XXS, 1, 380}, {Fmt::IQ3_XXS, 16, 420}, {Fmt::IQ3_XXS, 1u << 30, 300},
        {Fmt::VJ_Q4_A8, 1, 740}, {Fmt::VJ_Q4_A8, 16, 760}, {Fmt::VJ_Q4_A8, 1u << 30, 700},
        {Fmt::VJ_Q4_DOT8, 1, 760}, {Fmt::VJ_Q4_DOT8, 16, 770}, {Fmt::VJ_Q4_DOT8, 1u << 30, 760},
        {Fmt::VJ_DOT8_EXACT, 1, 760}, {Fmt::VJ_DOT8_EXACT, 16, 770}, {Fmt::VJ_DOT8_EXACT, 1u << 30, 760},
        {Fmt::VJ_Q8_DP4A, 1, 800}, {Fmt::VJ_Q8_DP4A, 16, 800}, {Fmt::VJ_Q8_DP4A, 1u << 30, 780},
        {Fmt::Q6_K, 1, 600}, {Fmt::Q4_K, 1, 650}, {Fmt::F16, 1, 820},
    });
}

double BandwidthTable::gbs(Fmt f, uint32_t m) const {
    double best = 0.0;
    uint32_t best_max = UINT32_MAX;
    for (const auto& e : e_)
        if (e.fmt == f && m <= e.m_max && e.m_max <= best_max) { best = e.gbs; best_max = e.m_max; }
    return best;
}

void BandwidthTable::set(Fmt f, uint32_t m_max, double v) {
    for (auto& e : e_) if (e.fmt == f && e.m_max == m_max) { e.gbs = v; return; }
    e_.push_back({f, m_max, v});
}

std::vector<Ban> PlannerConfig::default_bans() {
    return {{Fmt::IQ2_S, Fmt::VJ_Q4_A8, 1, "IQ2_S -> VJ_Q4_A8 at M=1 is a measured decode regression (banned)"}};
}

uint64_t tensor_bytes(Fmt f, uint64_t weights) {
    return static_cast<uint64_t>(std::ceil(static_cast<double>(weights) * fmt_bpw(f) / 8.0));
}

double stream_ms(Fmt f, uint64_t weights, double share, double read_fraction, double gbs) {
    if (gbs <= 0.0) return 1e30;
    return static_cast<double>(tensor_bytes(f, weights)) * share * read_fraction / (gbs * 1e9) * 1e3;
}

namespace {

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

struct Cand {
    uint32_t t = 0;
    Fmt to = Fmt::VJ_Q4_A8;
    int cls = 1;
    double ms_saved = 0.0, extra_gib = 0.0, roi = 0.0;
    std::vector<uint64_t> extra;  // per GPU
};

} // namespace

PlanResult plan(const std::vector<TensorSpec>& ts, const std::vector<GpuBudget>& gpus, const BandwidthTable& bw,
                const PlannerConfig& cfg) {
    PlanResult r;
    const size_t G = gpus.size();
    if (G == 0) { r.error = "no GPUs"; return r; }
    const uint64_t reserve = std::max<uint64_t>(cfg.reserve_bytes, 2ull << 30);
    r.gpus.resize(G);
    for (size_t g = 0; g < G; ++g) {
        auto& gp = r.gpus[g];
        gp.vram = gpus[g].vram_bytes; gp.reserve = reserve; gp.kv = gpus[g].kv_bytes;
        gp.state = gpus[g].state_bytes; gp.cache = gpus[g].cache_bytes; gp.other = gpus[g].other_bytes;
    }
    auto share = [&](const TensorSpec& t, size_t g) { return g < t.gpu_share.size() ? t.gpu_share[g] : 1.0 / G; };
    auto gpu_ms = [&](Fmt f, const TensorSpec& t, size_t g) {
        return stream_ms(f, t.weights, share(t, g), t.read_fraction, bw.gbs(f, cfg.m) * gpus[g].bw_scale);
    };
    std::vector<Fmt> cur(ts.size());
    r.decisions.resize(ts.size());
    for (size_t i = 0; i < ts.size(); ++i) {
        const auto& t = ts[i];
        cur[i] = t.base;
        auto& d = r.decisions[i];
        d.name = t.name; d.role = t.role; d.layer = t.layer; d.from = d.to = t.base;
        for (size_t g = 0; g < G; ++g)
            r.gpus[g].weights_base += static_cast<uint64_t>(std::ceil(tensor_bytes(t.base, t.weights) * share(t, g)));
    }
    for (auto& gp : r.gpus) gp.weights_final = gp.weights_base;
    for (size_t g = 0; g < G; ++g) {
        if (r.gpus[g].headroom() < 0) {
            r.error = "baseline does not fit on GPU " + std::to_string(g) + " (short by " +
                      std::to_string(-r.gpus[g].headroom() >> 20) + " MiB above the reserve)";
            return r;
        }
    }

    // Candidate screening: ratio rule and bans. Keep the best surviving candidate per tensor.
    std::vector<Cand> cands;
    for (uint32_t i = 0; i < ts.size(); ++i) {
        const auto& t = ts[i];
        auto& d = r.decisions[i];
        const double bw_old = bw.gbs(t.base, cfg.m);
        bool any_ratio_fail = false, any_ban = false;
        std::string ban_why, ratio_why;
        Cand best;
        bool have = false;
        for (Fmt to : t.candidates) {
            if (to == t.base) continue;
            const double bw_new = bw.gbs(to, cfg.m);
            const double bw_ratio = bw_old > 0 ? bw_new / bw_old : 0.0;
            const double bpw_ratio = fmt_bpw(to) / fmt_bpw(t.base);
            d.bw_ratio = bw_ratio;
            d.bpw_ratio = bpw_ratio;
            bool banned = false;
            for (const auto& b : cfg.bans)
                if (b.from == t.base && b.to == to && cfg.m <= b.m_max) { banned = true; ban_why = b.why; }
            if (banned) { any_ban = true; continue; }
            if (!(bw_ratio > bpw_ratio)) {
                any_ratio_fail = true;
                char buf[160];
                std::snprintf(buf, sizeof(buf), "%s->%s: bw_new/bw_old=%.3f <= bpw_new/bpw_old=%.3f", fmt_name(t.base),
                              fmt_name(to), bw_ratio, bpw_ratio);
                ratio_why = buf;
                continue;
            }
            Cand c;
            c.t = i; c.to = to;
            c.cls = (cfg.iq3_down_first && t.role == Role::RoutedDown && t.base == Fmt::IQ3_XXS) ? 0 : 1;
            c.extra.resize(G);
            for (size_t g = 0; g < G; ++g) {
                const double saved = gpu_ms(t.base, t, g) - gpu_ms(to, t, g);
                c.ms_saved = std::max(c.ms_saved, saved);
                const double eb = (static_cast<double>(tensor_bytes(to, t.weights)) - tensor_bytes(t.base, t.weights)) * share(t, g);
                c.extra[g] = eb > 0 ? static_cast<uint64_t>(std::ceil(eb)) : 0;
                c.extra_gib = std::max(c.extra_gib, eb / kGiB);
            }
            c.roi = c.extra_gib > 0 ? c.ms_saved / c.extra_gib : 1e30;
            if (!have || c.cls < best.cls || (c.cls == best.cls && c.roi > best.roi)) { best = c; have = true; }
        }
        if (have) {
            d.bw_ratio = bw.gbs(best.to, cfg.m) / bw_old;
            d.bpw_ratio = fmt_bpw(best.to) / fmt_bpw(t.base);
            cands.push_back(best);
        } else if (any_ban) {
            d.verdict = Verdict::RejectedBan; d.why = ban_why;
        } else if (any_ratio_fail) {
            d.verdict = Verdict::RejectedRatio; d.why = ratio_why;
        }
    }
    // Acceptance: policy class, then ROI (desc), then name for determinism.
    std::stable_sort(cands.begin(), cands.end(), [&](const Cand& a, const Cand& b) {
        if (a.cls != b.cls) return a.cls < b.cls;
        if (a.roi != b.roi) return a.roi > b.roi;
        return ts[a.t].name < ts[b.t].name;
    });
    for (const auto& c : cands) {
        bool fits = true;
        for (size_t g = 0; g < G; ++g)
            if (r.gpus[g].headroom() < static_cast<int64_t>(c.extra[g])) fits = false;
        auto& d = r.decisions[c.t];
        if (!fits) {
            d.verdict = Verdict::RejectedBudget;
            d.why = "needs " + std::to_string(static_cast<uint64_t>(c.extra_gib * 1024)) + " MiB more than the headroom above the reserve";
            continue;
        }
        for (size_t g = 0; g < G; ++g) r.gpus[g].weights_final += c.extra[g];
        cur[c.t] = c.to;
        d.to = c.to;
        d.verdict = Verdict::Promoted;
        d.ms_saved = c.ms_saved;
        d.extra_gib = c.extra_gib;
        r.promotion_order.push_back(c.t);
    }
    for (size_t g = 0; g < G; ++g) {
        double base = 0, fin = 0;
        for (size_t i = 0; i < ts.size(); ++i) { base += gpu_ms(ts[i].base, ts[i], g); fin += gpu_ms(cur[i], ts[i], g); }
        r.ms_per_token_base = std::max(r.ms_per_token_base, base);
        r.ms_per_token_final = std::max(r.ms_per_token_final, fin);
    }
    r.ok = true;
    return r;
}

std::string format_plan(const PlanResult& r, const PlannerConfig& cfg) {
    std::ostringstream o;
    char buf[256];
    o << "vegajet plan (dry run, M=" << cfg.m << ")\n";
    if (!r.ok) { o << "  ERROR: " << r.error << "\n"; return o.str(); }
    auto gib = [](uint64_t b) { return static_cast<double>(b) / kGiB; };
    o << "  GPU   VRAM GiB  reserve  KV+state  cache  weights(base->final)  headroom\n";
    for (size_t g = 0; g < r.gpus.size(); ++g) {
        const auto& p = r.gpus[g];
        std::snprintf(buf, sizeof(buf), "  %-4zu %9.2f %8.2f %9.2f %6.2f %9.2f -> %-9.2f %8.2f\n", g, gib(p.vram), gib(p.reserve),
                      gib(p.kv + p.state), gib(p.cache + p.other), gib(p.weights_base), gib(p.weights_final),
                      static_cast<double>(p.headroom()) / kGiB);
        o << buf;
    }
    // Group decisions by (role, from, to, verdict) with layer counts.
    std::map<std::string, std::pair<uint32_t, std::string>> groups;
    std::vector<std::string> order;
    for (const auto& d : r.decisions) {
        if (d.verdict == Verdict::KeptBase) continue;
        std::snprintf(buf, sizeof(buf), "%-16s %-14s %-8s -> %-13s", verdict_name(d.verdict), role_name(d.role),
                      fmt_name(d.from), fmt_name(d.verdict == Verdict::Promoted ? d.to : d.from));
        const std::string key = buf;
        if (!groups.count(key)) order.push_back(key);
        auto& gr = groups[key];
        gr.first++;
        if (gr.second.empty() && !d.why.empty()) gr.second = d.why;
    }
    o << "  decisions:\n";
    for (const auto& k : order) {
        const auto& gr = groups[k];
        std::snprintf(buf, sizeof(buf), "    %s x%-3u", k.c_str(), gr.first);
        o << buf;
        if (!gr.second.empty()) o << "  (" << gr.second << ")";
        o << "\n";
    }
    if (!r.promotion_order.empty()) {
        const auto& first = r.decisions[r.promotion_order.front()];
        o << "  first promotion: " << first.name << " (" << fmt_name(first.from) << " -> " << fmt_name(first.to) << ")\n";
    }
    std::snprintf(buf, sizeof(buf), "  weight streaming: %.3f -> %.3f ms/token (slowest GPU, decode M=%u)\n",
                  r.ms_per_token_base, r.ms_per_token_final, cfg.m);
    o << buf;
    return o.str();
}

} // namespace vegajet::vram
