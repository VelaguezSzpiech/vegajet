// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/tp3_plan.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace vegajet::tp3 {

namespace {

std::string fmt(const char* f, double a, double b = 0.0, double c = 0.0) {
    char buf[160];
    std::snprintf(buf, sizeof buf, f, a, b, c);
    return buf;
}

std::vector<double> weights(const Tp3Config& cfg) { return {cfg.share[0], cfg.share[1], cfg.share[2]}; }

// shard::split_rows with a configurable unit (contiguous ranges, multiples of `unit` except the last).
std::vector<shard::Range> split_rows_unit(uint64_t rows, const std::vector<double>& bw, uint32_t unit) {
    unit = std::max<uint32_t>(1, unit);
    const auto u = shard::apportion((rows + unit - 1) / unit, bw);
    std::vector<shard::Range> r(u.size());
    uint64_t pos = 0;
    for (size_t i = 0; i < u.size(); ++i) {
        r[i].begin = std::min(pos, rows);
        pos += u[i] * unit;
        r[i].end = std::min(pos, rows);
    }
    return r;
}

} // namespace

std::vector<Check> check_power(const Tp3Config& cfg) {
    std::vector<Check> out;
    double sum = 0.0;
    for (uint32_t g = 0; g < kGpus; ++g) {
        const double w = cfg.power_cap_w[g];
        sum += w;
        Check c{"power.cap.gpu" + std::to_string(g), true, fmt("%.0f W", w)};
        if (!(w <= cfg.cap_ceiling_w)) { c.ok = false; c.detail = fmt("%.0f W exceeds the %.0f W per-GPU ceiling for 3 GPUs", w, cfg.cap_ceiling_w); }
        else if (!(w >= cfg.cap_floor_w)) { c.ok = false; c.detail = fmt("%.0f W is below the %.0f W floor", w, cfg.cap_floor_w); }
        out.push_back(c);
    }
    if (cfg.psu_budget_w > 0.0) {
        Check c{"power.psu_budget", sum + cfg.host_w <= cfg.psu_budget_w,
                fmt("caps %.0f W + host %.0f W vs budget %.0f W", sum, cfg.host_w, cfg.psu_budget_w)};
        out.push_back(c);
    }
    return out;
}

std::string Plan::summary() const {
    std::string s;
    for (uint32_t g = 0; g < kGpus; ++g) {
        const GpuPlan& p = gpu[g];
        char buf[256];
        std::snprintf(buf, sizeof buf, "gpu%u: q heads %llu, kv heads %llu, ffn rows %llu, expert rows %llu, vocab rows %llu, "
                                       "%.2f GB (%.1f%%)\n",
                      g, (unsigned long long)p.q_heads.size(), (unsigned long long)p.kv_heads.size(),
                      (unsigned long long)p.ffn_rows.size(), (unsigned long long)p.expert_rows.size(),
                      (unsigned long long)p.vocab_rows.size(), p.weight_bytes / 1e9, 100.0 * p.realized_share);
        s += buf;
    }
    for (const auto& c : checks) s += (c.ok ? "  PASS " : "  FAIL ") + c.name + ": " + c.detail + "\n";
    return s;
}

Plan plan_tp3(const ops::ModelDesc& m, const Tp3Config& cfg) {
    Plan p;
    double tot = 0.0;
    bool shares_ok = true;
    for (double s : cfg.share) { shares_ok = shares_ok && s > 0.0; tot += s; }
    p.checks.push_back({"shares", shares_ok, fmt("%.3f / %.3f / %.3f", cfg.share[0], cfg.share[1], cfg.share[2])});
    const uint32_t L = m.n_layers();
    const bool model_ok = m.d_model > 0 && L > 0 && m.ffn.size() == L && m.attn.n_q_heads > 0 && m.attn.n_kv_heads > 0 &&
                          m.attn.head_dim > 0 && m.attn.n_q_heads % m.attn.n_kv_heads == 0;
    p.checks.push_back({"model", model_ok, model_ok ? m.arch : "incomplete ModelDesc (dims, heads or per-layer kinds)"});
    for (const auto& c : check_power(cfg)) p.checks.push_back(c);
    if (!shares_ok || !model_ok) return p;

    const std::vector<double> bw = weights(cfg);
    const double d = m.d_model;
    const auto attn = shard::plan_attention(m.attn.n_q_heads, m.attn.n_kv_heads, bw);
    uint32_t n_gqa = 0, n_gdn = 0, n_dense = 0, n_moe = 0;
    for (uint32_t l = 0; l < L; ++l) {
        (m.mixer[l] == ops::MixerKind::Gdn ? n_gdn : n_gqa)++;
        (m.ffn[l] == ops::FfnKind::Moe ? n_moe : n_dense)++;
    }
    std::vector<shard::GdnShard> gdn;
    if (n_gdn) gdn = shard::plan_gdn(m.gdn.n_k_groups, m.gdn.n_v_heads, bw);
    const auto ffn = split_rows_unit(m.dense_ff, bw, cfg.row_unit);
    const auto erows = split_rows_unit(m.moe.expert_ff, bw, cfg.row_unit);
    const auto srows = split_rows_unit(m.moe.shared_ff, bw, cfg.row_unit);
    const auto whole = shard::split_experts(m.moe.n_experts, bw);
    const auto vocab = split_rows_unit(m.vocab, bw, cfg.row_unit);
    const double bd = cfg.bits_dense / 8, be = cfg.bits_expert / 8, bh = cfg.bits_head / 8;

    for (uint32_t g = 0; g < kGpus; ++g) {
        GpuPlan& gp = p.gpu[g];
        gp.q_heads = attn.gpus[g].q_heads;
        gp.kv_heads = attn.gpus[g].kv_heads;
        gp.seq_fraction = attn.gpus[g].seq_fraction;
        const double hd = m.attn.head_dim, qd = double(gp.q_heads.size()) * hd, kvd = double(gp.kv_heads.size()) * hd;
        double bytes = n_gqa * (2.0 * d * qd + 2.0 * d * kvd) * bd;  // q, o + k, v
        if (n_gdn) {
            gp.gdn_groups = gdn[g].k_groups;
            const double nk = double(gdn[g].k_groups.size()), nv = double(gdn[g].v_heads.size());
            const double qkv = d * (2.0 * nk * m.gdn.head_k_dim + nv * m.gdn.head_v_dim);
            bytes += n_gdn * (qkv + 2.0 * d * nv * m.gdn.head_v_dim + 2.0 * d * nv) * bd;  // qkv, z + out, a/b
        }
        gp.ffn_rows = ffn[g];
        bytes += n_dense * 3.0 * d * double(ffn[g].size()) * bd;
        if (n_moe) {
            if (m.moe.shard == ops::ExpertShard::WholeExperts) {
                gp.experts = whole[g];
                bytes += n_moe * double(whole[g].size()) * 3.0 * d * m.moe.expert_ff * be;
            } else {
                gp.expert_rows = erows[g];
                gp.experts = {0, m.moe.n_experts};
                bytes += n_moe * double(m.moe.n_experts) * 3.0 * d * double(erows[g].size()) * be;
            }
            bytes += n_moe * (3.0 * d * double(srows[g].size()) + d * m.moe.n_experts) * bd;  // shared expert, router
        }
        gp.vocab_rows = vocab[g];
        bytes += double(vocab[g].size()) * d * bh;
        gp.weight_bytes = bytes;
        p.total_weight_bytes += bytes;
    }
    for (uint32_t g = 0; g < kGpus; ++g) {
        GpuPlan& gp = p.gpu[g];
        gp.realized_share = p.total_weight_bytes > 0 ? gp.weight_bytes / p.total_weight_bytes : 0.0;
        p.max_share_error = std::max(p.max_share_error, std::fabs(gp.realized_share - cfg.share[g] / tot));
        p.checks.push_back({"vram.gpu" + std::to_string(g), gp.weight_bytes <= cfg.vram_bytes[g],
                            fmt("weights %.2f GB of %.2f GB (token embedding gathered on the host, not counted)",
                                gp.weight_bytes / 1e9, cfg.vram_bytes[g] / 1e9)});
    }
    p.checks.push_back({"balance", p.max_share_error <= cfg.balance_tolerance,
                        fmt("max |realized - target| share %.4f (tolerance %.3f, row unit %.0f)", p.max_share_error,
                            cfg.balance_tolerance, cfg.row_unit)});
    p.ok = std::all_of(p.checks.begin(), p.checks.end(), [](const Check& c) { return c.ok; });
    return p;
}

// ------------------------------------------------------------------------------------------------ all-reduce
const char* algo_name(ArAlgo a) { return a == ArAlgo::Direct ? "direct" : "ring"; }

namespace {

double phase_time(const std::vector<ArStep>& steps, uint32_t phase, uint32_t eb, const Tp3Config& cfg) {
    double t_max = 0.0;
    double tx_bytes_time[kGpus] = {0, 0, 0}, rx_bytes_time[kGpus] = {0, 0, 0}, tx_lat[kGpus] = {0, 0, 0}, rx_lat[kGpus] = {0, 0, 0};
    for (const auto& s : steps) {
        if (s.phase != phase || s.end == s.begin) continue;
        const Link& lk = cfg.link[s.src][s.dst];
        const double bytes = double(s.end - s.begin) * eb;
        const double lat = lk.p2p ? lk.latency_us : 2.0 * lk.latency_us;
        const double gbs = lk.p2p ? lk.gbs : 0.5 * cfg.host_gbs;
        const double xfer = bytes / (gbs * 1e3);  // GB/s -> bytes per us = gbs * 1e3
        t_max = std::max(t_max, lat + xfer);
        tx_bytes_time[s.src] += xfer;
        rx_bytes_time[s.dst] += xfer;
        tx_lat[s.src] = std::max(tx_lat[s.src], lat);
        rx_lat[s.dst] = std::max(rx_lat[s.dst], lat);
    }
    if (cfg.shared_egress)
        for (uint32_t g = 0; g < kGpus; ++g) t_max = std::max({t_max, tx_lat[g] + tx_bytes_time[g], rx_lat[g] + rx_bytes_time[g]});
    return t_max;
}

} // namespace

ArSchedule plan_allreduce(uint64_t elems, uint32_t elem_bytes, ArAlgo algo, const Tp3Config& cfg) {
    ArSchedule s;
    s.algo = algo;
    s.elems = elems;
    s.elem_bytes = elem_bytes;
    constexpr uint64_t kUnit = 64;  // 128 B at fp16: keeps every chunk boundary DMA-friendly
    const auto units = shard::apportion((elems + kUnit - 1) / kUnit, weights(cfg));
    uint64_t pos = 0;
    for (uint32_t g = 0; g < kGpus; ++g) {
        s.chunk[g].begin = std::min(pos, elems);
        pos += units[g] * kUnit;
        s.chunk[g].end = std::min(pos, elems);
    }
    if (algo == ArAlgo::Direct) {
        s.phases = 2;
        s.canonical_order = true;
        for (uint32_t j = 0; j < kGpus; ++j)
            for (uint32_t g = 0; g < kGpus; ++g)
                if (g != j) s.steps.push_back({0, g, j, s.chunk[j].begin, s.chunk[j].end, true});
        for (uint32_t j = 0; j < kGpus; ++j)
            for (uint32_t g = 0; g < kGpus; ++g)
                if (g != j) s.steps.push_back({1, j, g, s.chunk[j].begin, s.chunk[j].end, false});
    } else {
        s.phases = 2 * (kGpus - 1);
        for (uint32_t st = 0; st < kGpus - 1; ++st)  // reduce-scatter: rank r sends chunk (r - st) to r + 1
            for (uint32_t r = 0; r < kGpus; ++r) {
                const uint32_t c = (r + kGpus - st) % kGpus;
                s.steps.push_back({st, r, (r + 1) % kGpus, s.chunk[c].begin, s.chunk[c].end, true});
            }
        for (uint32_t st = 0; st < kGpus - 1; ++st)  // all-gather: rank r sends chunk (r + 1 - st) to r + 1
            for (uint32_t r = 0; r < kGpus; ++r) {
                const uint32_t c = (r + 1 + kGpus - st) % kGpus;
                s.steps.push_back({kGpus - 1 + st, r, (r + 1) % kGpus, s.chunk[c].begin, s.chunk[c].end, false});
            }
    }
    for (uint32_t ph = 0; ph < s.phases; ++ph) s.modeled_us += phase_time(s.steps, ph, elem_bytes, cfg);
    return s;
}

ArSchedule choose_allreduce(uint64_t elems, uint32_t elem_bytes, const Tp3Config& cfg) {
    ArSchedule d = plan_allreduce(elems, elem_bytes, ArAlgo::Direct, cfg);
    ArSchedule r = plan_allreduce(elems, elem_bytes, ArAlgo::Ring, cfg);
    return r.modeled_us < d.modeled_us ? r : d;
}

bool simulate_allreduce(const ArSchedule& s, std::array<std::vector<float>, kGpus>& bufs, std::string* err) {
    auto fail = [&](const char* m) { if (err) *err = m; return false; };
    for (const auto& b : bufs)
        if (b.size() != s.elems) return fail("buffer size != schedule elems");
    for (const auto& st : s.steps)
        if (st.src >= kGpus || st.dst >= kGpus || st.src == st.dst || st.begin > st.end || st.end > s.elems || st.phase >= s.phases)
            return fail("malformed step");
    for (uint32_t ph = 0; ph < s.phases; ++ph) {
        const auto snap = bufs;  // all transfers of a phase run concurrently and read phase-start data
        std::array<std::array<const std::vector<float>*, kGpus>, kGpus> staged{};
        std::array<shard::Range, kGpus> staged_range{};
        std::vector<std::vector<uint8_t>> touched(kGpus, std::vector<uint8_t>(s.elems, 0));
        for (const auto& st : s.steps) {
            if (st.phase != ph) continue;
            for (uint64_t i = st.begin; i < st.end; ++i) {
                if (touched[st.dst][i] && !(st.reduce && s.algo == ArAlgo::Direct)) return fail("overlapping writes in one phase");
                touched[st.dst][i] = 1;
            }
            if (st.reduce && s.algo == ArAlgo::Direct) {
                if (staged[st.dst][st.src]) return fail("duplicate staged partial");
                staged[st.dst][st.src] = &snap[st.src];
                staged_range[st.dst] = {st.begin, st.end};
            } else if (st.reduce) {
                for (uint64_t i = st.begin; i < st.end; ++i) bufs[st.dst][i] = snap[st.dst][i] + snap[st.src][i];
            } else {
                for (uint64_t i = st.begin; i < st.end; ++i) bufs[st.dst][i] = snap[st.src][i];
            }
        }
        for (uint32_t dst = 0; dst < kGpus; ++dst) {  // owner sums the three partials in rank order
            bool any = false;
            for (uint32_t r = 0; r < kGpus; ++r) any = any || staged[dst][r];
            if (!any) continue;
            for (uint32_t r = 0; r < kGpus; ++r)
                if (r != dst && !staged[dst][r]) return fail("owner is missing a partial");
            for (uint64_t i = staged_range[dst].begin; i < staged_range[dst].end; ++i) {
                float acc = (dst == 0 ? snap[0] : *staged[dst][0])[i];
                for (uint32_t r = 1; r < kGpus; ++r) acc += (dst == r ? snap[r] : *staged[dst][r])[i];
                bufs[dst][i] = acc;
            }
        }
    }
    if (err) err->clear();
    return true;
}

} // namespace vegajet::tp3
