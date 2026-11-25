// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/topology_planner.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>

namespace vegajet::topo {

const char* mode_name(Mode m) {
    switch (m) {
        case Mode::SingleGpu: return "single_gpu";
        case Mode::TensorParallel: return "tensor_parallel";
        case Mode::Swarm: return "swarm";
        case Mode::Insufficient: break;
    }
    return "insufficient";
}

namespace {

uint64_t usable_of(const hw::Gpu& g, const PlanConfig& cfg) {
    return g.vram_bytes > cfg.reserve_bytes ? g.vram_bytes - cfg.reserve_bytes : 0;
}

std::string mib_text(uint64_t bytes) {
    return std::to_string((bytes + (1ull << 20) - 1ull) >> 20);
}

struct Alloc {
    bool ok = false;
    std::string why;
    std::vector<uint64_t> bytes;
    std::vector<double> share;
};

// Splits `resident_bytes + kv_bytes` over `members` proportionally to calibrated bandwidth, pinning a member
// that cannot hold its proportional share at its usable capacity and redistributing the rest. Integers are
// assigned by floor plus largest-remainder so the byte totals are exact and no member exceeds its capacity.
Alloc split_capacity(const hw::Profile& profile, const ModelFit& m, const PlanConfig& cfg,
                     const std::vector<uint32_t>& members) {
    Alloc a;
    const size_t k = members.size();
    a.bytes.assign(k, 0);
    a.share.assign(k, 0.0);
    const uint64_t need = m.resident_bytes + m.kv_bytes;
    if (need == 0) {
        a.ok = true;
        return a;
    }
    std::vector<double> usable(k, 0.0), bw(k, 0.0), alloc(k, 0.0);
    std::vector<bool> pinned(k, false);
    double usable_sum = 0.0;
    for (size_t i = 0; i < k; ++i) {
        const hw::Gpu& g = profile.gpus[members[i]];
        usable[i] = static_cast<double>(usable_of(g, cfg));
        bw[i] = g.bw_gbs;
        usable_sum += usable[i];
        if (!(bw[i] > 0.0)) {
            a.why = "gpu " + std::to_string(members[i]) + " has no streaming bandwidth";
            return a;
        }
    }
    if (usable_sum + 1.0 < static_cast<double>(need)) {
        a.why = "capacity: need " + mib_text(need) + " MiB, " + mib_text(static_cast<uint64_t>(usable_sum)) +
                " MiB usable after the " + mib_text(cfg.reserve_bytes) + " MiB reserve";
        return a;
    }

    double remaining = static_cast<double>(need);
    while (true) {
        double bwsum = 0.0;
        for (size_t i = 0; i < k; ++i) {
            if (!pinned[i]) bwsum += bw[i];
        }
        if (bwsum <= 0.0) {
            a.why = "capacity: no bandwidth left to place " + mib_text(static_cast<uint64_t>(std::max(0.0, remaining))) +
                    " MiB";
            return a;
        }
        std::vector<double> target(k, 0.0);
        for (size_t i = 0; i < k; ++i) {
            if (!pinned[i]) target[i] = remaining * bw[i] / bwsum;
        }
        bool changed = false;
        for (size_t i = 0; i < k; ++i) {
            if (pinned[i] || !(target[i] > usable[i] + 1e-9)) continue;
            pinned[i] = true;
            alloc[i] = usable[i];
            remaining -= usable[i];
            changed = true;
        }
        if (!changed) {
            for (size_t i = 0; i < k; ++i) {
                if (!pinned[i]) alloc[i] = target[i];
            }
            break;
        }
        if (remaining <= 0.0) break;
    }

    double sum = 0.0;
    for (size_t i = 0; i < k; ++i) sum += alloc[i];
    if (sum + 1e-6 < static_cast<double>(need)) {
        a.why = "capacity: placement left " + mib_text(static_cast<uint64_t>(std::max(0.0, static_cast<double>(need) - sum))) +
                " MiB unplaced";
        return a;
    }

    const uint64_t need_i = need;
    std::vector<double> frac(k, 0.0);
    uint64_t assigned = 0;
    for (size_t i = 0; i < k; ++i) {
        const double v = std::max(0.0, std::min(alloc[i], usable[i]));
        const double base = std::floor(v);
        a.bytes[i] = static_cast<uint64_t>(base);
        frac[i] = v - base;
        assigned += a.bytes[i];
        if (static_cast<double>(a.bytes[i]) > usable[i]) {
            a.why = "capacity: gpu " + std::to_string(members[i]) + " allocation exceeds usable VRAM";
            return a;
        }
    }
    uint64_t rest = need_i > assigned ? need_i - assigned : 0;
    while (rest > 0) {
        size_t pick = k;
        double pick_frac = -1.0;
        for (size_t i = 0; i < k; ++i) {
            if (static_cast<double>(a.bytes[i]) >= usable[i]) continue;
            const bool better = pick == k || frac[i] > pick_frac + 1e-15 ||
                                (std::fabs(frac[i] - pick_frac) <= 1e-15 && i < pick);
            if (better) {
                pick = i;
                pick_frac = frac[i];
            }
        }
        if (pick == k) {
            a.why = "capacity: rounding could not place the final bytes";
            return a;
        }
        ++a.bytes[pick];
        frac[pick] = 0.0;
        --rest;
    }
    uint64_t total = 0;
    for (size_t i = 0; i < k; ++i) {
        total += a.bytes[i];
        if (static_cast<double>(a.bytes[i]) > usable[i]) {
            a.why = "capacity: gpu " + std::to_string(members[i]) + " allocation exceeds usable VRAM";
            return a;
        }
        a.share[i] = static_cast<double>(a.bytes[i]) / static_cast<double>(need_i);
    }
    if (total != need_i) {
        a.why = "capacity: byte split is not exact";
        return a;
    }
    a.ok = true;
    return a;
}

bool build_instance(const hw::Profile& profile, const ModelFit& m, const PlanConfig& cfg,
                    const std::vector<uint32_t>& members, InstancePlan& out, std::string* why) {
    if (members.empty()) {
        if (why) *why = "empty instance";
        return false;
    }
    std::vector<uint32_t> ms = members;
    std::sort(ms.begin(), ms.end());
    ms.erase(std::unique(ms.begin(), ms.end()), ms.end());
    const uint32_t tp = static_cast<uint32_t>(ms.size());

    Alloc a = split_capacity(profile, m, cfg, ms);
    if (!a.ok) {
        if (why) *why = a.why;
        return false;
    }

    out = InstancePlan{};
    out.gpus = ms;
    out.tp = tp;
    out.share = a.share;
    out.resident_bytes.assign(tp, 0);
    out.kv_bytes.assign(tp, 0);
    for (uint32_t i = 0; i < tp; ++i) {
        const double r = std::floor(a.share[i] * static_cast<double>(m.resident_bytes));
        out.resident_bytes[i] = static_cast<uint64_t>(r);
        out.kv_bytes[i] = a.bytes[i] >= out.resident_bytes[i] ? a.bytes[i] - out.resident_bytes[i] : 0;
    }

    const uint64_t token_bytes = m.token_bytes != 0 ? m.token_bytes : m.resident_bytes;
    const double token = static_cast<double>(token_bytes);
    out.compute_ms = 0.0;
    for (uint32_t i = 0; i < tp; ++i) {
        const hw::Gpu& g = profile.gpus[ms[i]];
        const double ms_i = a.share[i] * token / (g.bw_gbs * 1e9) * 1e3;
        out.compute_ms = std::max(out.compute_ms, ms_i);
    }

    if (tp > 1) {
        uint64_t ar_bytes = m.ar_bytes_per_token;
        if (ar_bytes == 0 && m.layers > 0 && m.hidden > 0) {
            ar_bytes = static_cast<uint64_t>(m.layers) * static_cast<uint64_t>(m.hidden) * 2ull * 2ull;
        }
        out.ar_bytes = ar_bytes;
        const double volume = static_cast<double>(ar_bytes) *
                              (static_cast<double>(tp - 1) / static_cast<double>(tp));
        double link = std::numeric_limits<double>::infinity();
        double latency = 0.0;
        for (uint32_t i = 0; i < tp; ++i) {
            for (uint32_t j = i + 1; j < tp; ++j) {
                const hw::P2pLink l = profile.link(static_cast<size_t>(ms[i]), static_cast<size_t>(ms[j]));
                double gbs = 0.0;
                double lat = 0.0;
                if (l.direct() && l.gbs > 0.0) {
                    gbs = l.gbs;
                    lat = l.latency_us > 0.0 ? l.latency_us : cfg.ar_latency_us;
                } else {
                    if (cfg.links.require_direct_p2p) {
                        if (why)
                            *why = "no direct P2P between gpu " + std::to_string(ms[i]) + " and gpu " +
                                   std::to_string(ms[j]);
                        return false;
                    }
                    out.host_bounced = true;
                    out.p2p_complete = false;
                    gbs = cfg.links.host_gbs;
                    lat = cfg.links.host_latency_us > 0.0 ? cfg.links.host_latency_us : cfg.ar_latency_us;
                }
                if (!(gbs > 0.0)) {
                    if (why) *why = "no usable all-reduce link between gpu " + std::to_string(ms[i]) + " and gpu " +
                                    std::to_string(ms[j]);
                    return false;
                }
                link = std::min(link, gbs);
                latency = std::max(latency, lat);
            }
        }
        if (!(link >= cfg.links.min_link_gbs)) {
            if (why)
                *why = "all-reduce link " + std::to_string(static_cast<uint64_t>(link)) + " GB/s is below the " +
                       std::to_string(static_cast<uint64_t>(cfg.links.min_link_gbs)) + " GB/s floor";
            return false;
        }
        out.effective_link_gbs = link;
        out.ar_ms = volume / (link * 1e9) * 1e3 + 2.0 * latency / 1000.0;
    }

    out.per_token_ms = out.compute_ms + out.ar_ms;
    out.tps = out.per_token_ms > 0.0 ? 1000.0 / out.per_token_ms : 0.0;
    return true;
}

bool lex_less(const std::vector<InstancePlan>& a, const std::vector<InstancePlan>& b) {
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        if (a[i].gpus != b[i].gpus) return a[i].gpus < b[i].gpus;
    }
    return a.size() < b.size();
}

struct Candidate {
    uint32_t tp = 0;
    std::vector<InstancePlan> groups;
    double aggregate = 0.0;
    uint32_t host_bounced = 0;
    bool ok() const { return !groups.empty(); }
};

Candidate make_candidate(uint32_t tp, std::vector<InstancePlan> groups) {
    Candidate c;
    c.tp = tp;
    c.groups = std::move(groups);
    for (const auto& g : c.groups) {
        c.aggregate += g.tps;
        if (g.host_bounced) ++c.host_bounced;
    }
    return c;
}

bool candidate_better(const Candidate& a, const Candidate& b) {
    const double scale = std::max(1.0, std::max(std::fabs(a.aggregate), std::fabs(b.aggregate)));
    if (a.aggregate > b.aggregate + 1e-9 * scale) return true;
    if (b.aggregate > a.aggregate + 1e-9 * scale) return false;
    if (a.host_bounced != b.host_bounced) return a.host_bounced < b.host_bounced;
    if (a.tp != b.tp) return a.tp < b.tp;
    return lex_less(a.groups, b.groups);
}

struct SearchCtx {
    const hw::Profile* hw = nullptr;
    const ModelFit* model = nullptr;
    const PlanConfig* cfg = nullptr;
    uint32_t d = 1;
    Candidate best;
    bool have = false;
    uint32_t failed = 0;
    std::string first_why;
};

void evaluate(SearchCtx& c, const std::vector<InstancePlan>& cur) {
    if (cur.empty()) return;
    Candidate cand = make_candidate(c.d, cur);
    if (!c.have || candidate_better(cand, c.best)) {
        c.best = std::move(cand);
        c.have = true;
    }
}

void enumerate(std::vector<uint32_t> remaining, std::vector<InstancePlan> cur, SearchCtx& c);

void enumerate_groups(uint32_t first, const std::vector<uint32_t>& pool, size_t start, uint32_t need,
                      std::vector<uint32_t>& chosen, const std::vector<uint32_t>& remaining,
                      std::vector<InstancePlan>& cur, SearchCtx& c) {
    if (need == 0) {
        std::vector<uint32_t> members;
        members.reserve(chosen.size() + 1);
        members.push_back(first);
        members.insert(members.end(), chosen.begin(), chosen.end());
        std::sort(members.begin(), members.end());
        InstancePlan p;
        std::string why;
        if (build_instance(*c.hw, *c.model, *c.cfg, members, p, &why)) {
            std::vector<uint32_t> next;
            next.reserve(remaining.size());
            for (uint32_t g : remaining) {
                if (std::find(members.begin(), members.end(), g) == members.end()) next.push_back(g);
            }
            cur.push_back(std::move(p));
            enumerate(std::move(next), cur, c);
            cur.pop_back();
        } else {
            if (c.failed++ == 0) c.first_why = why;
        }
        return;
    }
    for (size_t k = start; k + need <= pool.size(); ++k) {
        chosen.push_back(pool[k]);
        enumerate_groups(first, pool, k + 1, need - 1, chosen, remaining, cur, c);
        chosen.pop_back();
    }
}

void enumerate(std::vector<uint32_t> remaining, std::vector<InstancePlan> cur, SearchCtx& c) {
    if (remaining.size() < c.d) {
        evaluate(c, cur);
        return;
    }
    // Leave the first GPU idle so partitions that do not contain it are also explored.
    if (remaining.size() > c.d) {
        std::vector<uint32_t> rest(remaining.begin() + 1, remaining.end());
        enumerate(std::move(rest), cur, c);
    }
    const uint32_t first = remaining.front();
    const std::vector<uint32_t> pool(remaining.begin() + 1, remaining.end());
    std::vector<uint32_t> chosen;
    enumerate_groups(first, pool, 0, c.d - 1, chosen, remaining, cur, c);
    // The last group could not be formed: the groups already placed are still a valid (smaller) plan.
    if (remaining.size() == c.d) evaluate(c, cur);
}

void keep_best(Candidate& c, uint32_t limit) {
    if (limit == 0 || c.groups.size() <= limit) return;
    std::stable_sort(c.groups.begin(), c.groups.end(), [](const InstancePlan& a, const InstancePlan& b) {
        if (a.tps != b.tps) return a.tps > b.tps;
        return a.gpus < b.gpus;
    });
    c.groups.resize(limit);
    const uint32_t tp = c.tp;
    c = make_candidate(tp, std::move(c.groups));
}

bool build_candidates(const hw::Profile& hw, const ModelFit& m, const PlanConfig& cfg, uint32_t d,
                      Candidate& out, std::string* why) {
    const size_t n = hw.gpus.size();
    if (d == 0 || static_cast<size_t>(d) > n) {
        if (why) *why = "TP=" + std::to_string(d) + " needs " + std::to_string(d) + " GPUs, only " + std::to_string(n) +
                        " visible";
        return false;
    }
    if (d == 1) {
        std::vector<InstancePlan> all;
        std::string first_why;
        for (uint32_t i = 0; i < static_cast<uint32_t>(n); ++i) {
            InstancePlan p;
            std::string w;
            if (build_instance(hw, m, cfg, {i}, p, &w)) all.push_back(std::move(p));
            else if (first_why.empty()) first_why = w;
        }
        if (all.empty()) {
            if (why) *why = first_why.empty() ? "no GPU can host the model" : first_why;
            return false;
        }
        out = make_candidate(1, std::move(all));
    } else {
        if (!cfg.allow_tp) {
            if (why) *why = "tensor parallelism is disabled by configuration";
            return false;
        }
        SearchCtx ctx;
        ctx.hw = &hw;
        ctx.model = &m;
        ctx.cfg = &cfg;
        ctx.d = d;
        std::vector<uint32_t> pool(n);
        for (size_t i = 0; i < n; ++i) pool[i] = static_cast<uint32_t>(i);
        std::vector<InstancePlan> cur;
        enumerate(std::move(pool), std::move(cur), ctx);
        if (!ctx.have) {
            if (why) *why = ctx.first_why.empty() ? "no feasible TP=" + std::to_string(d) + " group" : ctx.first_why;
            return false;
        }
        out = std::move(ctx.best);
    }

    uint32_t limit = cfg.max_instances;
    if (!cfg.allow_swarm && (limit == 0 || limit > 1)) limit = 1;
    if (limit != 0) keep_best(out, limit);
    if (out.groups.empty()) {
        if (why) *why = "no usable instance";
        return false;
    }
    return true;
}

} // namespace

Plan plan(const hw::Profile& hw, const ModelFit& model, const PlanConfig& cfg) {
    Plan r;
    std::string why;
    if (!hw.valid(&why)) {
        r.checks.push_back({"profile", false, why});
        r.detail = "invalid hardware profile: " + why;
        return r;
    }
    r.checks.push_back({"profile", true, hw.arch + " gpus=" + std::to_string(hw.size())});
    if (model.resident_bytes == 0) {
        r.checks.push_back({"model", false, "no resident weight bytes"});
        r.detail = "model footprint is empty";
        return r;
    }
    if (model.resident_bytes > std::numeric_limits<uint64_t>::max() - model.kv_bytes) {
        r.checks.push_back({"model", false, "resident + KV footprint overflows"});
        r.detail = "model footprint overflows 64-bit bytes";
        return r;
    }

    const uint32_t n = static_cast<uint32_t>(hw.size());
    if (cfg.force_tp > kTpCap) {
        r.checks.push_back({"selection", false, "forced TP=" + std::to_string(cfg.force_tp) + " exceeds TP=" +
                                                     std::to_string(kTpCap)});
        r.detail = "forced TP is outside the supported range";
        return r;
    }
    if (cfg.force_tp > n) {
        r.checks.push_back({"selection", false, "forced TP=" + std::to_string(cfg.force_tp) + " needs more GPUs than " +
                                                     std::to_string(n)});
        r.detail = "forced TP needs more GPUs than are visible";
        return r;
    }

    uint32_t limit = std::min(n, kTpCap);
    if (cfg.max_tp != 0) limit = std::min(limit, cfg.max_tp);
    if (limit == 0) limit = 1;

    Candidate best;
    bool have = false;
    std::string best_why;
    if (cfg.force_tp != 0) {
        if (build_candidates(hw, model, cfg, cfg.force_tp, best, &why)) have = true;
        else best_why = why;
    } else {
        for (uint32_t d = 1; d <= limit; ++d) {
            Candidate c;
            std::string w;
            if (build_candidates(hw, model, cfg, d, c, &w)) {
                if (!have || candidate_better(c, best)) {
                    best = std::move(c);
                    have = true;
                }
            } else if (best_why.empty()) {
                best_why = w;
            }
        }
    }

    const uint64_t footprint = model.resident_bytes + model.kv_bytes;
    if (!have) {
        uint64_t usable_sum = 0;
        uint64_t best_usable = 0;
        for (const auto& g : hw.gpus) {
            const uint64_t u = usable_of(g, cfg);
            if (u > std::numeric_limits<uint64_t>::max() - usable_sum) usable_sum = std::numeric_limits<uint64_t>::max();
            else usable_sum += u;
            best_usable = std::max(best_usable, u);
        }
        r.mode = Mode::Insufficient;
        r.checks.push_back({"capacity", usable_sum >= footprint,
                            "need " + mib_text(footprint) + " MiB, " + mib_text(usable_sum) + " MiB usable across " +
                                std::to_string(n) + " GPUs, best single GPU " + mib_text(best_usable) +
                                " MiB after the " + mib_text(cfg.reserve_bytes) + " MiB reserve"});
        r.detail = cfg.force_tp != 0 ? "forced TP=" + std::to_string(cfg.force_tp) + ": " + best_why
                                     : "no feasible topology: " + best_why;
        return r;
    }

    r.ok = true;
    r.groups = std::move(best.groups);
    r.tp = best.tp;
    r.instances = static_cast<uint32_t>(r.groups.size());
    r.allreduce = r.tp > 1;
    r.compute_ms = 0.0;
    r.ar_ms = 0.0;
    r.per_token_ms = 0.0;
    r.aggregate_tps = 0.0;
    r.p2p_complete = true;
    uint64_t ar_bytes = 0;
    uint64_t min_headroom = std::numeric_limits<uint64_t>::max();
    double min_effective_link = std::numeric_limits<double>::infinity();
    for (const auto& g : r.groups) {
        r.compute_ms = std::max(r.compute_ms, g.compute_ms);
        r.ar_ms = std::max(r.ar_ms, g.ar_ms);
        r.per_token_ms = std::max(r.per_token_ms, g.per_token_ms);
        r.aggregate_tps += g.tps;
        r.p2p_complete = r.p2p_complete && g.p2p_complete;
        if (g.ar_bytes > ar_bytes) ar_bytes = g.ar_bytes;
        if (g.tp > 1 && g.effective_link_gbs > 0.0) min_effective_link = std::min(min_effective_link, g.effective_link_gbs);
        for (size_t i = 0; i < g.gpus.size(); ++i) {
            const uint64_t usable = usable_of(hw.gpus[g.gpus[i]], cfg);
            const uint64_t used = g.resident_bytes[i] + g.kv_bytes[i];
            if (used <= usable) min_headroom = std::min(min_headroom, usable - used);
            else min_headroom = 0;
        }
    }
    if (!std::isfinite(min_effective_link)) min_effective_link = 0.0;
    r.ar_bytes_per_token = ar_bytes;
    if (r.instances > 1) r.mode = Mode::Swarm;
    else if (r.tp > 1) r.mode = Mode::TensorParallel;
    else r.mode = Mode::SingleGpu;
    if (n == 1) r.mode = Mode::SingleGpu;

    double best_single_bw = 0.0;
    for (const auto& g : hw.gpus) {
        if (usable_of(g, cfg) >= footprint && g.bw_gbs > best_single_bw) best_single_bw = g.bw_gbs;
    }
    if (best_single_bw > 0.0) {
        const double token = static_cast<double>(model.token_bytes != 0 ? model.token_bytes : model.resident_bytes);
        r.single_gpu_ms = token / (best_single_bw * 1e9) * 1e3;
        r.speedup = r.per_token_ms > 0.0 ? r.single_gpu_ms / r.per_token_ms : 0.0;
        r.checks.push_back({"single_gpu", true,
                            "reference " + std::to_string(static_cast<uint64_t>(r.single_gpu_ms)) +
                                " ms on a single card"});
    } else {
        r.checks.push_back({"single_gpu", false,
                            "needs TP: " + mib_text(footprint) + " MiB footprint exceeds every single usable capacity"});
    }

    r.checks.push_back({"capacity", true,
                        "every member fits; min headroom " +
                            mib_text(min_headroom == std::numeric_limits<uint64_t>::max() ? 0 : min_headroom) +
                            " MiB above the reserve"});

    if (r.tp == 1) {
        r.checks.push_back({"links", true, "not used: TP=1 all-reduce is elided"});
    } else if (r.p2p_complete) {
        r.checks.push_back({"links", true,
                            "direct P2P on every TP pair, slowest " +
                                std::to_string(static_cast<uint64_t>(min_effective_link)) + " GB/s"});
    } else {
        r.checks.push_back({"links", true, "host-bounced pair(s) present on the TP group"});
    }

    if (!r.allreduce) {
        r.checks.push_back({"allreduce", true, "elided (TP=1): 0 B/token, 0 ms"});
    } else {
        r.checks.push_back({"allreduce", true,
                            std::to_string(r.ar_bytes_per_token) + " B/token over " +
                                std::to_string(static_cast<uint64_t>(min_effective_link)) + " GB/s, " +
                                std::to_string(static_cast<uint64_t>(r.ar_ms * 1000.0)) + " us"});
    }

    double max_dev = 0.0;
    for (const auto& g : r.groups) {
        double sumbw = 0.0;
        for (uint32_t idx : g.gpus) sumbw += hw.gpus[idx].bw_gbs;
        if (!(sumbw > 0.0)) continue;
        for (size_t i = 0; i < g.gpus.size(); ++i) {
            const double target = hw.gpus[g.gpus[i]].bw_gbs / sumbw;
            max_dev = std::max(max_dev, std::fabs(g.share[i] - target));
        }
    }
    {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "max share deviation from bandwidth-proportional: %.4f", max_dev);
        r.checks.push_back({"balance", true, buf});
    }

    {
        char buf[192];
        std::snprintf(buf, sizeof(buf), "mode=%s tp=%u instances=%u aggregate=%.1f tok/s", mode_name(r.mode), r.tp,
                      r.instances, r.aggregate_tps);
        r.checks.push_back({"selection", true, buf});
    }

    r.detail = std::string("selected ") + mode_name(r.mode) + " tp=" + std::to_string(r.tp) + " instances=" +
               std::to_string(r.instances);
    return r;
}

Plan plan_tp(const hw::Profile& hw, const ModelFit& model, uint32_t tp, const PlanConfig& cfg) {
    PlanConfig c = cfg;
    c.force_tp = tp;
    return plan(hw, model, c);
}

std::string Plan::summary() const {
    std::ostringstream o;
    char buf[256];
    o << "topology: mode=" << mode_name(mode) << " tp=" << tp << " instances=" << instances
      << " allreduce=" << (allreduce ? "on" : "elided") << "\n";
    std::snprintf(buf, sizeof(buf),
                  "  per_token=%.3f ms compute=%.3f ms ar=%.3f ms aggregate=%.1f tok/s speedup=%.3fx\n", per_token_ms,
                  compute_ms, ar_ms, aggregate_tps, speedup);
    o << buf;
    for (size_t i = 0; i < groups.size(); ++i) {
        const InstancePlan& g = groups[i];
        o << "  instance " << i << " gpus=[";
        for (size_t j = 0; j < g.gpus.size(); ++j) o << (j == 0 ? "" : ",") << g.gpus[j];
        o << "] share=[";
        for (size_t j = 0; j < g.share.size(); ++j) {
            std::snprintf(buf, sizeof(buf), "%s%.4f", j == 0 ? "" : ",", g.share[j]);
            o << buf;
        }
        std::snprintf(buf, sizeof(buf), "] per_token=%.3f ms tps=%.1f link=%.1f GB/s%s\n", g.per_token_ms, g.tps,
                      g.effective_link_gbs, g.host_bounced ? " host-bounced" : "");
        o << buf;
    }
    for (const auto& c : checks) {
        o << "  check " << c.name << ": " << (c.ok ? "ok" : "fail") << " (" << c.detail << ")\n";
    }
    return o.str();
}

} // namespace vegajet::topo
