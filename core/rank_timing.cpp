// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/rank_timing.hpp"

#include "core/json_lite.hpp"

#include <cinttypes>
#include <cmath>
#include <cstdio>

namespace vegajet::ranktiming {

namespace {
bool fail(std::string* err, const std::string& m) { if (err) *err = m; return false; }
std::string u64s(uint64_t v) { char b[24]; std::snprintf(b, sizeof(b), "%" PRIu64, v); return b; }
std::string us(uint64_t ticks, double hz) { char b[40]; std::snprintf(b, sizeof(b), "%.3f", ticks_to_us(ticks, hz)); return b; }
bool tick(const json::Value& o, const char* k, uint64_t& out) {
    const auto* v = o.get(k);
    if (!v || !v->is_number() || !(v->num >= 0.0) || v->num > 9007199254740992.0 || std::floor(v->num) != v->num) return false;
    out = static_cast<uint64_t>(v->num);
    return true;
}
}  // namespace

double ticks_to_us(uint64_t ticks, double tick_hz) { return static_cast<double>(ticks) * 1e6 / tick_hz; }

Aggregator::Aggregator(uint32_t ranks, double tick_hz) : r_(ranks), hz_(tick_hz) {
    for (uint32_t i = 0; i < ranks; ++i) r_[i].t.rank = i;
}

void Aggregator::break_chain(uint32_t rank) { if (rank < r_.size()) r_[rank].broken = true; }

bool Aggregator::add(uint32_t rank, const std::string& stage, uint64_t begin, uint64_t end, uint64_t wait, bool contiguous,
                     std::string* err) {
    if (rank >= r_.size()) return fail(err, "rank " + std::to_string(rank) + " out of range");
    if (stage.empty()) return fail(err, "empty stage name");
    if (end < begin) return fail(err, "rank " + std::to_string(rank) + " stage " + stage + ": end stamp before begin stamp");
    if (wait > end - begin) return fail(err, "rank " + std::to_string(rank) + " stage " + stage + ": wait longer than its span");
    auto& p = r_[rank];
    if (p.last_end && begin < *p.last_end)
        return fail(err, "rank " + std::to_string(rank) + " stage " + stage + ": begin stamp before the previous end stamp");
    if (p.last_end) (contiguous && !p.broken ? p.t.gap : p.t.unstamped) += begin - *p.last_end;
    if (!p.first) p.first = begin;
    p.last_end = end;
    p.broken = false;
    p.t.spans++;
    p.t.active += end - begin - wait;
    p.t.wait += wait;
    p.t.wall = end - *p.first;
    StageTotals* s = nullptr;
    for (auto& x : p.t.stages) if (x.stage == stage) s = &x;
    if (!s) { p.t.stages.push_back(StageTotals{stage, 0, 0, 0}); s = &p.t.stages.back(); }
    s->count++;
    s->active += end - begin - wait;
    s->wait += wait;
    return true;
}

Report Aggregator::report(const std::string& source, uint32_t tokens) const {
    Report r;
    r.source = source;
    r.tick_hz = hz_;
    r.tokens = tokens;
    for (const auto& p : r_) if (p.t.spans) r.ranks.push_back(p.t);
    return r;
}

std::string to_table(const Report& r) {
    const double div = r.tokens ? static_cast<double>(r.tokens) : 1.0;
    auto v = [&](uint64_t t) { return ticks_to_us(t, r.tick_hz) / div; };
    std::string o;
    char b[256];
    std::snprintf(b, sizeof(b), "[RANK-TIMING table] source=%s, %u token(s), s_memrealtime %.3f MHz, us %s:\n", r.source.c_str(), r.tokens,
                  r.tick_hz / 1e6, r.tokens ? "per token" : "total");
    o += b;
    std::snprintf(b, sizeof(b), "  %-6s %-22s %7s %12s %12s %12s %12s %12s\n", "rank", "stage", "spans", "active", "AR wait", "host/AQL gap",
                  "unstamped", "wall");
    o += b;
    for (const auto& k : r.ranks) {
        for (const auto& s : k.stages) {
            std::snprintf(b, sizeof(b), "  %-6u %-22s %7" PRIu64 " %12.1f %12.1f %12s %12s %12s\n", k.rank, s.stage.c_str(), s.count, v(s.active),
                          v(s.wait), "-", "-", "-");
            o += b;
        }
        std::snprintf(b, sizeof(b), "  %-6u %-22s %7" PRIu64 " %12.1f %12.1f %12.1f %12.1f %12.1f\n", k.rank, "total", k.spans, v(k.active),
                      v(k.wait), v(k.gap), v(k.unstamped), v(k.wall));
        o += b;
    }
    return o;
}

std::string to_json(const Report& r) {
    char hz[40];
    std::snprintf(hz, sizeof(hz), "%.10g", r.tick_hz);
    std::string o = "{\"schema\": " + json::quote(kSchema) + ", \"source\": " + json::quote(r.source) + ", \"tick_hz\": " + hz +
                    ", \"tokens\": " + std::to_string(r.tokens) + ", \"ranks\": [";
    for (size_t i = 0; i < r.ranks.size(); ++i) {
        const auto& k = r.ranks[i];
        o += (i ? ", " : "") + std::string("{\"rank\": ") + std::to_string(k.rank) + ", \"spans\": " + u64s(k.spans) +
             ", \"active_ticks\": " + u64s(k.active) + ", \"allreduce_wait_ticks\": " + u64s(k.wait) + ", \"host_gap_ticks\": " + u64s(k.gap) +
             ", \"unstamped_ticks\": " + u64s(k.unstamped) + ", \"wall_ticks\": " + u64s(k.wall) + ", \"active_us\": " + us(k.active, r.tick_hz) +
             ", \"allreduce_wait_us\": " + us(k.wait, r.tick_hz) + ", \"host_gap_us\": " + us(k.gap, r.tick_hz) +
             ", \"unstamped_us\": " + us(k.unstamped, r.tick_hz) + ", \"wall_us\": " + us(k.wall, r.tick_hz) + ", \"stages\": [";
        for (size_t j = 0; j < k.stages.size(); ++j) {
            const auto& s = k.stages[j];
            o += (j ? ", " : "") + std::string("{\"stage\": ") + json::quote(s.stage) + ", \"count\": " + u64s(s.count) +
                 ", \"active_ticks\": " + u64s(s.active) + ", \"wait_ticks\": " + u64s(s.wait) + "}";
        }
        o += "]}";
    }
    return o + "]}";
}

bool from_value(const json::Value& v, Report& r, std::string* err) {
    if (!v.is_object()) return fail(err, "rank_timing: not an object");
    const auto* schema = v.get("schema");
    if (!schema || !schema->is_string() || schema->str != kSchema) return fail(err, std::string("rank_timing: schema is not ") + kSchema);
    Report out;
    const auto* src = v.get("source");
    const auto* hz = v.get("tick_hz");
    uint64_t tokens = 0;
    if (!src || !src->is_string() || !hz || !hz->is_number() || !(hz->num > 0.0) || !tick(v, "tokens", tokens) || tokens > UINT32_MAX)
        return fail(err, "rank_timing: source / tick_hz / tokens missing or invalid");
    out.source = src->str;
    out.tick_hz = hz->num;
    out.tokens = static_cast<uint32_t>(tokens);
    const auto* ranks = v.get("ranks");
    if (!ranks || !ranks->is_array()) return fail(err, "rank_timing: ranks is not an array");
    for (const auto& e : ranks->arr) {
        RankTotals k;
        uint64_t rank = 0;
        if (!e.is_object() || !tick(e, "rank", rank) || rank > UINT32_MAX || !tick(e, "spans", k.spans) || !tick(e, "active_ticks", k.active) ||
            !tick(e, "allreduce_wait_ticks", k.wait) || !tick(e, "host_gap_ticks", k.gap) || !tick(e, "unstamped_ticks", k.unstamped) ||
            !tick(e, "wall_ticks", k.wall))
            return fail(err, "rank_timing: rank entry has missing or non-integer tick fields");
        k.rank = static_cast<uint32_t>(rank);
        const auto* st = e.get("stages");
        if (!st || !st->is_array()) return fail(err, "rank_timing: stages is not an array");
        uint64_t n = 0, a = 0, w = 0;
        for (const auto& s : st->arr) {
            StageTotals t;
            const auto* name = s.is_object() ? s.get("stage") : nullptr;
            if (!name || !name->is_string() || name->str.empty() || !tick(s, "count", t.count) || !tick(s, "active_ticks", t.active) ||
                !tick(s, "wait_ticks", t.wait))
                return fail(err, "rank_timing: stage entry invalid");
            t.stage = name->str;
            n += t.count; a += t.active; w += t.wait;
            k.stages.push_back(t);
        }
        // The stored split must be exact: stages add up to the rank totals, and the totals add up to wall.
        if (n != k.spans || a != k.active || w != k.wait || k.active + k.wait + k.gap + k.unstamped != k.wall)
            return fail(err, "rank_timing: rank " + std::to_string(k.rank) + " totals do not add up");
        for (const auto& prev : out.ranks)
            if (prev.rank == k.rank) return fail(err, "rank_timing: duplicate rank " + std::to_string(k.rank));
        out.ranks.push_back(std::move(k));
    }
    r = std::move(out);
    return true;
}

bool from_json(const std::string& text, Report& r, std::string* err) {
    json::Value v;
    if (!json::parse(text, v)) return fail(err, "rank_timing: not JSON");
    return from_value(v, r, err);
}

}  // namespace vegajet::ranktiming
