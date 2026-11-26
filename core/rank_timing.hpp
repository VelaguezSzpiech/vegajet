// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// v1.3 M7: per-rank / per-stage device timing from s_memrealtime stamps, schema "vegajet.rank_timing.v1".
//
// Input: for every rank, the stamped spans of one run in queue order. A span is [begin, end] in device ticks of
// that rank's own s_memrealtime counter, with `wait` ticks inside it that the rank spent polling for its peers
// (AllReduce flag wait; 0 for compute stages). Ticks of different ranks are never compared: each GPU has its own
// counter, so every quantity below is a difference of stamps taken on one GPU.
//
// Per rank, the device time from the first begin to the last end is split exactly (integer ticks, no rounding):
//   active   = sum(end - begin - wait)  stamped work (kernels between the two stamps, including the dispatch
//                                       boundaries inside one stamped span)
//   wait     = sum(wait)                AllReduce peer wait measured inside the collective kernel
//   gap      = begin(i) - end(i-1)      when span i is `contiguous`: no unstamped dispatch ran between the two
//                                       stamps, so the interval is host submit/wait plus AQL packet processing
//   unstamped= begin(i) - end(i-1)      when span i is not `contiguous` (other, unstamped kernels ran there)
//   wall     = active + wait + gap + unstamped
// Nothing is estimated: a rank without stamps has no entry, and malformed stamps are rejected, not clamped.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace vegajet {
namespace json { struct Value; }
namespace ranktiming {

constexpr const char* kSchema = "vegajet.rank_timing.v1";
constexpr double kGfx906TickHz = 25.0e6;  // s_memrealtime on gfx906: 25 MHz reference clock (0.04 us per tick)

struct StageTotals {
    std::string stage;
    uint64_t count = 0, active = 0, wait = 0;  // ticks
};

struct RankTotals {
    uint32_t rank = 0;
    uint64_t spans = 0, active = 0, wait = 0, gap = 0, unstamped = 0, wall = 0;  // ticks
    std::vector<StageTotals> stages;  // first-seen order
};

struct Report {
    std::string source;      // "nrank-profile-stages", "tp2-ar-timeline", ...
    double tick_hz = kGfx906TickHz;
    uint32_t tokens = 0;     // decode tokens covered (per-token averages in the table; 0 = totals only)
    std::vector<RankTotals> ranks;
};

class Aggregator {
public:
    Aggregator(uint32_t ranks, double tick_hz = kGfx906TickHz);
    // Appends one span of `rank` (queue order). false + err for rank out of range, end < begin, wait > end - begin,
    // or begin before the rank's previous end (stamps of one queue cannot go backwards).
    bool add(uint32_t rank, const std::string& stage, uint64_t begin, uint64_t end, uint64_t wait, bool contiguous,
             std::string* err);
    // Next span of `rank` follows unstamped device work (e.g. a collective bypass copy): book the interval as unstamped.
    void break_chain(uint32_t rank);
    Report report(const std::string& source, uint32_t tokens) const;
    uint32_t ranks() const { return static_cast<uint32_t>(r_.size()); }

private:
    struct PerRank {
        RankTotals t;
        std::optional<uint64_t> first, last_end;
        bool broken = false;
    };
    std::vector<PerRank> r_;
    double hz_;
};

double ticks_to_us(uint64_t ticks, double tick_hz);
std::string to_table(const Report& r);
std::string to_json(const Report& r);  // one line; tick counts as exact integers, *_us derived from tick_hz
bool from_value(const json::Value& v, Report& r, std::string* err);
bool from_json(const std::string& text, Report& r, std::string* err);
// The engine prints `[RANK-TIMING] <to_json>` once per run; bench_json parses it back from the log.
constexpr const char* kLogTag = "[RANK-TIMING] ";

}  // namespace ranktiming
}  // namespace vegajet
