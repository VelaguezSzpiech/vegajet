// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// core/rank_timing (v1.3 M7): synthetic s_memrealtime stamp streams for N = 1, 2, 3, 8 ranks, built from a known
// schedule with a different counter origin per rank (each GPU has its own clock), go through the same booking the
// N-rank engine does (stage brackets + collective entry / flags / peers / exit stamps). The test checks the exact
// integer split active / AllReduce wait / host-AQL gap / unstamped = wall per rank and per stage, the table,
// the vegajet.rank_timing.v1 JSON round trip, its rejection rules, the [RANK-TIMING] log line inside a
// vegajet.bench.v1 scorecard, and that --profile-stages is admitted on the generic N-rank path.
#include "core/rank_timing.hpp"

#include "core/bench_json.hpp"
#include "models/ornith/ornith_tp_exec.hpp"
#include "tests/test_common.hpp"

#include <string>
#include <vector>

using namespace vegajet;
using namespace vegajet::ranktiming;

namespace {

struct Expect { uint64_t active = 0, wait = 0, gap = 0, unstamped = 0, wall = 0, spans = 0, mixer = 0, moe = 0, ar = 0, vocab = 0; };

// One rank's device clock walking through `tokens` decode tokens of `layers` layers in queue order, with known
// durations. ar_wait != 0 only when n > 1 (N=1 uses the bypass copy: no collective kernel, no stamps).
struct Clock {
    uint64_t t;
    Expect e;
    void span(Aggregator& a, uint32_t rank, const char* stage, uint64_t len, uint64_t wait, uint64_t& slot) {
        VJ_CHECK(a.add(rank, stage, t, t + len, wait, true, nullptr));
        e.active += len - wait; e.wait += wait; e.spans++; slot += len - wait;
        t += len;
    }
    void gap(uint64_t g) { t += g; e.gap += g; }
    void unstamped(uint64_t g) { t += g; e.unstamped += g; }
};

Expect run(Aggregator& a, uint32_t n, uint32_t rank, uint32_t tokens, uint32_t layers) {
    Clock c{1000003ull * (rank + 1) + 77ull * rank * rank, {}};  // unrelated origins: never compared across ranks
    const uint64_t first = c.t;
    bool started = false;
    for (uint32_t tok = 0; tok < tokens; ++tok) {
        a.break_chain(rank);  // token boundary as in the engine
        if (started) c.unstamped(500 + 11 * tok);  // argmax read-back / embedding upload
        started = true;
        for (uint32_t il = 0; il < layers; ++il) {
            c.span(a, rank, "mixer", 300 + 7 * rank + il, 0, c.e.mixer);
            if (n > 1) {
                c.gap(20 + rank);                                            // host submit + AQL launch of the collective
                c.span(a, rank, "allreduce", 90 + 3 * rank, 40 + (rank % 3) * 5, c.e.ar);  // includes peer-flag wait
                c.gap(15 + il % 2);                                          // host wait_idle return + next submit
            } else {
                a.break_chain(rank);
                c.unstamped(12);  // bypass copy kernel: unstamped device work
            }
            c.span(a, rank, "moe", 800 + 13 * il, 0, c.e.moe);
            if (n > 1) {
                c.gap(21);
                c.span(a, rank, "allreduce", 95, 30 + rank, c.e.ar);
                c.gap(17);
            } else {
                a.break_chain(rank);
                c.unstamped(12);
            }
        }
        c.span(a, rank, "final_norm+vocab", 400 + rank, 0, c.e.vocab);
    }
    c.e.wall = c.t - first;  // last span end - first span begin
    VJ_CHECK(c.e.wall == c.e.active + c.e.wait + c.e.gap + c.e.unstamped);  // the schedule itself adds up
    return c.e;
}

const StageTotals* stage(const RankTotals& r, const char* name) {
    for (const auto& s : r.stages) if (s.stage == name) return &s;
    return nullptr;
}

void check_n(uint32_t n) {
    const uint32_t tokens = 3, layers = 4;
    Aggregator a(n);
    std::vector<Expect> want;
    for (uint32_t r = 0; r < n; ++r) want.push_back(run(a, n, r, tokens, layers));
    const Report rep = a.report("synthetic", tokens);
    VJ_CHECK(rep.ranks.size() == n);
    for (uint32_t r = 0; r < n; ++r) {
        const auto& k = rep.ranks[r];
        const auto& e = want[r];
        VJ_CHECK(k.rank == r);
        VJ_CHECK(k.active == e.active && k.wait == e.wait && k.gap == e.gap && k.unstamped == e.unstamped);
        VJ_CHECK(k.wall == e.wall && k.active + k.wait + k.gap + k.unstamped == k.wall);
        VJ_CHECK(k.spans == e.spans);
        VJ_CHECK(stage(k, "mixer")->active == e.mixer && stage(k, "mixer")->count == uint64_t(tokens) * layers);
        VJ_CHECK(stage(k, "moe")->active == e.moe);
        VJ_CHECK(stage(k, "final_norm+vocab")->active == e.vocab && stage(k, "final_norm+vocab")->count == tokens);
        if (n == 1) {
            VJ_CHECK(!stage(k, "allreduce") && k.wait == 0 && k.gap == 0);  // bypass: no wait and no host gap invented
        } else {
            VJ_CHECK(stage(k, "allreduce")->active == e.ar && stage(k, "allreduce")->wait == k.wait && k.wait > 0);
            VJ_CHECK(stage(k, "allreduce")->count == uint64_t(tokens) * layers * 2);
        }
        VJ_CHECK(k.stages.size() == (n == 1 ? 3u : 4u) && k.stages[0].stage == "mixer");  // first-seen order
    }
    // Table and JSON round trip (integer ticks are exact).
    const std::string table = to_table(rep);
    VJ_CHECK(table.find("source=synthetic, 3 token(s)") != std::string::npos);
    VJ_CHECK(table.find("AR wait") != std::string::npos && table.find("host/AQL gap") != std::string::npos);
    const std::string js = to_json(rep);
    Report back;
    std::string err;
    VJ_CHECK(from_json(js, back, &err));
    VJ_CHECK(to_json(back) == js);
    VJ_CHECK(back.ranks.size() == n && back.ranks[n - 1].wall == rep.ranks[n - 1].wall);
    std::printf("  N=%u: rank0 active=%llu wait=%llu gap=%llu unstamped=%llu wall=%llu ticks\n", n,
                (unsigned long long)rep.ranks[0].active, (unsigned long long)rep.ranks[0].wait, (unsigned long long)rep.ranks[0].gap,
                (unsigned long long)rep.ranks[0].unstamped, (unsigned long long)rep.ranks[0].wall);
}

void check_rejects() {
    Aggregator a(2);
    std::string err;
    VJ_CHECK(!a.add(2, "mixer", 0, 1, 0, true, &err) && err.find("out of range") != std::string::npos);
    VJ_CHECK(!a.add(0, "mixer", 10, 9, 0, true, &err) && err.find("end stamp before begin") != std::string::npos);
    VJ_CHECK(!a.add(0, "allreduce", 10, 20, 11, true, &err) && err.find("wait longer") != std::string::npos);
    VJ_CHECK(!a.add(0, "", 10, 20, 0, true, &err));
    VJ_CHECK(a.add(0, "mixer", 100, 200, 0, true, &err));
    VJ_CHECK(!a.add(0, "moe", 150, 300, 0, true, &err) && err.find("before the previous end") != std::string::npos);
    // A non-contiguous span books its interval as unstamped, a contiguous one as host gap.
    VJ_CHECK(a.add(0, "moe", 250, 300, 0, false, &err));
    VJ_CHECK(a.add(0, "allreduce", 310, 400, 60, true, &err));
    const auto r = a.report("x", 0);
    VJ_CHECK(r.ranks.size() == 1);  // rank 1 has no stamps: no entry, nothing estimated
    VJ_CHECK(r.ranks[0].unstamped == 50 && r.ranks[0].gap == 10 && r.ranks[0].wait == 60 && r.ranks[0].active == 100 + 50 + 30);
    VJ_CHECK(r.ranks[0].wall == 300);

    // JSON rejection: wrong schema, fractional ticks, totals that do not add up, duplicate rank.
    const std::string good = to_json(r);
    Report out;
    VJ_CHECK(from_json(good, out, &err));
    auto swap = [&](const std::string& from, const std::string& to) {
        std::string s = good;
        const auto p = s.find(from);
        VJ_CHECK(p != std::string::npos);
        return s.replace(p, from.size(), to);
    };
    VJ_CHECK(!from_json(swap("vegajet.rank_timing.v1", "vegajet.rank_timing.v0"), out, &err));
    VJ_CHECK(!from_json(swap("\"host_gap_ticks\": 10", "\"host_gap_ticks\": 10.5"), out, &err));
    VJ_CHECK(!from_json(swap("\"host_gap_ticks\": 10", "\"host_gap_ticks\": 11"), out, &err) && err.find("do not add up") != std::string::npos);
    VJ_CHECK(!from_json(swap("\"active_ticks\": 100,", "\"active_ticks\": 101,"), out, &err));  // stage sum != rank total
    {
        Report two = r;
        two.ranks.push_back(two.ranks[0]);
        VJ_CHECK(!from_json(to_json(two), out, &err) && err.find("duplicate rank") != std::string::npos);
    }
}

void check_bench_json() {
    Aggregator a(3);
    for (uint32_t k = 0; k < 3; ++k) {
        VJ_CHECK(a.add(k, "mixer", 1000 * k, 1000 * k + 100, 0, true, nullptr));
        VJ_CHECK(a.add(k, "allreduce", 1000 * k + 120, 1000 * k + 200, 25 + k, true, nullptr));
    }
    const Report rep = a.report("nrank-profile-stages", 1);
    benchjson::Record rec;
    rec.command = "bench";
    rec.model = "model-file";
    VJ_CHECK(benchjson::parse_line(std::string(kLogTag) + to_json(rep), rec));
    VJ_CHECK(rec.rank_timing && rec.rank_timing->ranks.size() == 3 && rec.rank_timing->ranks[2].wait == 27);
    VJ_CHECK(!benchjson::parse_line(std::string(kLogTag) + "{\"schema\": \"other\"}", rec));  // malformed: not recognized
    VJ_CHECK(rec.rank_timing->ranks.size() == 3);                                             // and the record keeps the last good one
    const std::string doc = benchjson::to_json(rec);
    VJ_CHECK(doc.find("\"rank_timing\": {\"schema\": \"vegajet.rank_timing.v1\"") != std::string::npos);
    benchjson::Record back;
    std::string err;
    VJ_CHECK(benchjson::from_json(doc, back, &err));
    VJ_CHECK(back.rank_timing && to_json(*back.rank_timing) == to_json(rep));
    VJ_CHECK(benchjson::to_json(back) == doc);
    // A run without stamps keeps rank_timing null and does not erase a stored one on merge.
    benchjson::Record plain;
    plain.command = "bench";
    VJ_CHECK(benchjson::to_json(plain).find("\"rank_timing\": null") != std::string::npos);
    benchjson::merge(back, plain);
    VJ_CHECK(back.rank_timing && back.rank_timing->ranks.size() == 3);
    benchjson::Record from_plain;
    VJ_CHECK(benchjson::from_json(benchjson::to_json(plain), from_plain, &err) && !from_plain.rank_timing);
}

void check_admit() {
    for (uint32_t n : {1u, 2u, 3u, 8u}) {
        ornith::OrnithGpuOptions o;
        o.tp = n;
        o.profile_stages = true;
        VJ_CHECK(ornith::ornith_raw_admit(o).empty());
        o.profile_stages = false;
        o.ar_timeline = true;
        VJ_CHECK(ornith::ornith_raw_admit(o).empty() == (n == 2));
    }
}

}  // namespace

int main() {
    for (uint32_t n : {1u, 2u, 3u, 8u}) check_n(n);
    check_rejects();
    check_bench_json();
    check_admit();
    std::printf("test_rank_timing: all passed (N=1,2,3,8 exact split, table, JSON round trip and rejects, bench.v1 "
                "rank_timing, --profile-stages admit)\n");
    return 0;
}
