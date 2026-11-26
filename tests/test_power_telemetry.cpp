// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// CPU only: exact timestamps and observations, no sysfs, clocks, sleeps or GPU.
#include "core/power_telemetry.hpp"
#include "core/bench_json.hpp"
#include "core/json_lite.hpp"
#include "tests/test_common.hpp"

#include <cmath>
#include <cstdio>
#include <limits>

using namespace vegajet;
using telemetry::Card;
using telemetry::PowerWindow;
using telemetry::Verdict;

namespace {
void near(double actual, double expected) { VJ_CHECK(std::abs(actual - expected) < 1e-8); }
Card constant(const char* id, double watts, double seconds = 0.25) {
    Card c;
    c.id = id;
    c.observe(100.0, watts, 50.0, 150.0);
    c.observe(100.0 + seconds, watts, 51.0, 150.0);
    return c;
}
}

int main() {
    const double missing = telemetry::unavailable();
    // No measurements and one observation have no time-average, not a zero/pass.
    VJ_CHECK(telemetry::summarize({}).verdict == Verdict::incomplete);
    Card single;
    single.id = "card1";
    VJ_CHECK(single.verdict(2) == Verdict::incomplete);
    single.observe(0.0, 220.0, 50.0, 150.0);
    VJ_CHECK(single.verdict(2) == Verdict::incomplete && std::isnan(single.power.avg_w()));
    near(single.power.window_s(), 0.0);
    near(single.power.peak_w(), 220.0);
    Card single_peak;
    single_peak.observe(0.0, 270.0, missing, missing);
    VJ_CHECK(single_peak.verdict(2) == Verdict::fail);  // Known violations beat missing fields.

    // A sub-cadence run has a nonzero observed window; zero watts is a valid mean.
    const Card zero = constant("card0", 0.0, 0.125);
    near(zero.power.avg_w(), 0.0);
    near(zero.power.window_s(), 0.125);
    VJ_CHECK(zero.verdict(2) == Verdict::pass);
    VJ_CHECK(std::string(zero.power.window_kind()) == "short_observed");
    Card hot = constant("card1", 220.0, 0.125);
    VJ_CHECK(hot.verdict(2) == Verdict::fail);
    near(hot.power.avg_w(), 220.0);
    near(hot.power.avg1s_w(), 220.0);
    VJ_CHECK(constant("card0", 205.0).verdict(2) == Verdict::pass);
    VJ_CHECK(constant("card0", 205.01).verdict(2) == Verdict::fail);
    VJ_CHECK(constant("card0", 155.0).verdict(3) == Verdict::pass);
    VJ_CHECK(constant("card0", 155.01).verdict(3) == Verdict::fail);

    // Irregular intervals use elapsed time, not sample counts or a nominal 250 ms.
    PowerWindow uneven;
    VJ_CHECK(uneven.add(1000.0, 100.0));
    VJ_CHECK(uneven.add(1000.125, 200.0));
    VJ_CHECK(uneven.add(1002.125, 0.0));
    near(uneven.mean_w(), (100.0 * 0.125 + 200.0 * 2.0) / 2.125);
    near(uneven.avg_w(), uneven.mean_w());
    near(uneven.avg1s_w(), 200.0);
    near(uneven.window1s_s(), 1.0);
    VJ_CHECK(!uneven.add(1002.125, 1.0) && !uneven.add(1002.0, 1.0));
    VJ_CHECK(!uneven.add(1003.0, missing) && !uneven.add(1003.0, -1.0));
    VJ_CHECK(!uneven.add(std::numeric_limits<double>::infinity(), 1.0));
    VJ_CHECK(uneven.samples() == 3);

    // Exact cutoff: the [2,12] window includes one second of the sample at t=0.
    PowerWindow cutoff;
    cutoff.add(0.0, 100.0);
    cutoff.add(3.0, 200.0);
    cutoff.add(12.0, 50.0);
    near(cutoff.avg_w(), 190.0);
    near(cutoff.mean_w(), 175.0);
    near(cutoff.window_s(), 10.0);
    VJ_CHECK(std::string(cutoff.window_kind()) == "full_10s");
    cutoff.add(40.0, 0.0);
    near(cutoff.avg_w(), 190.0);  // Retain the worst, not just the final window.

    // Worst window can end BETWEEN reads: [2,12] peaks at 140 W, while t=25 is 100 W.
    PowerWindow interior;
    interior.add(0.0, 0.0);
    interior.add(2.0, 200.0);
    interior.add(6.0, 100.0);
    interior.add(25.0, 0.0);
    near(interior.avg_w(), 140.0);
    near(interior.avg1s_w(), 200.0);
    PowerWindow cadence;
    for (int i = 0; i < 40; ++i) cadence.add(i * 0.25, 150.0);
    near(cadence.window_s(), 9.75);  // Forty samples do not span ten seconds.
    VJ_CHECK(std::string(cadence.window_kind()) == "short_observed");
    cadence.add(10.0, 150.0);
    near(cadence.window_s(), 10.0);
    VJ_CHECK(std::string(cadence.window_kind()) == "full_10s");

    // Missing power, temperature or cap cannot pass, even after recovery.
    for (int field = 0; field < 3; ++field) {
        Card c = constant("card0", 150.0);
        c.observe(101.0, field == 0 ? missing : 150.0, field == 1 ? missing : 50.0, field == 2 ? missing : 150.0);
        c.observe(102.0, 150.0, 50.0, 150.0);
        VJ_CHECK(c.missing == 1 && c.verdict(2) == Verdict::incomplete);
    }
    Card cap_drift = constant("card0", 150.0);
    cap_drift.observe(101.0, 150.0, 50.0, 149.0);
    near(cap_drift.cap_w, 149.0);
    near(cap_drift.cap_min_w, 149.0);
    near(cap_drift.cap_max_w, 150.0);
    VJ_CHECK(cap_drift.verdict(2) == Verdict::fail);
    Card over_cap;
    over_cap.observe(0.0, missing, missing, 225.0);
    VJ_CHECK(over_cap.verdict(2) == Verdict::fail);
    Card warm = constant("card0", 150.0);
    warm.observe(101.0, 150.0, 65.0, 150.0);
    VJ_CHECK(warm.verdict(2) == Verdict::fail);
    hot.observe(101.0, 220.0, missing, 150.0);
    VJ_CHECK(hot.verdict(2) == Verdict::fail);

    // Each card owns its clock origin and span. A short failing card cannot hide
    // behind the other card's completed ten-second window or sample count.
    const Card full = constant("card0", 150.0, 20.0);
    Card short_card;
    short_card.id = "card2";
    short_card.observe(9000.0, 220.0, 50.0, 150.0);
    short_card.observe(9000.125, 220.0, 51.0, 150.0);
    auto summary = telemetry::summarize({full, short_card});
    VJ_CHECK(summary.verdict == Verdict::fail && summary.samples == 4);
    near(summary.avg_w, 220.0);
    near(summary.window_s, 0.125);
    VJ_CHECK(std::string(summary.window_kind) == "short_observed");
    VJ_CHECK(telemetry::summarize({full, Card{}}).verdict == Verdict::incomplete);
    VJ_CHECK(telemetry::summarize({full, short_card, Card{}}).verdict == Verdict::fail);

    // Exercise the actual engine formatter through both JSON directions. Preserve
    // the selected average's duration and all per-card evidence when merging runs.
    benchjson::Record run;
    run.timestamp_utc = "2026-11-25T00:00:00Z";
    const std::string log = telemetry::report({full, short_card});
    VJ_CHECK(log.find("short observed rolling avg=220.") != std::string::npos);
    benchjson::parse_log(log, run);
    VJ_CHECK(run.telem_verdict == "FAIL" && run.telemetry_cards.size() == 2);
    near(*run.power_avg_w, 220.0);
    near(*run.power_avg_limit_w, 205.0);  // Also parse the mean=... variant.
    near(*run.power_window_s, 0.125);
    VJ_CHECK(run.power_window_kind == "short_observed" && run.power_hold == "previous");
    near(*run.telemetry_cards[0].cap_w, 150.0);
    near(*run.telemetry_cards[0].mean_w, 150.0);
    near(*run.telemetry_cards[0].window_s, 10.0);
    near(*run.telemetry_cards[0].observed_s, 20.0);
    near(*run.telemetry_cards[1].peak_w, 220.0);
    near(*run.telemetry_cards[1].window_s, 0.125);
    VJ_CHECK(run.telemetry_cards[1].timestamp_utc == run.timestamp_utc);
    std::string err;
    benchjson::Record restored;
    const std::string encoded = benchjson::to_json(run);
    VJ_CHECK(benchjson::from_json(encoded, restored, &err) && benchjson::to_json(restored) == encoded);

    benchjson::Record no_data;
    Card unreadable;
    unreadable.id = "card1";
    benchjson::parse_log(telemetry::report({unreadable}), no_data);
    VJ_CHECK(no_data.telem_verdict == "INCOMPLETE" && !no_data.power_avg_w && !no_data.power_peak_w);
    VJ_CHECK(!no_data.telemetry_cards[0].mean_w && !no_data.telemetry_cards[0].cap_w);
    json::Value absent;
    VJ_CHECK(json::parse(benchjson::to_json(no_data), absent));
    VJ_CHECK(absent.get("telemetry")->get("power_rolling_avg_w")->is_null());
    benchjson::Record healthy;
    benchjson::parse_log(telemetry::report({full}), healthy);
    benchjson::merge(healthy, no_data);
    VJ_CHECK(healthy.telem_verdict == "INCOMPLETE");
    benchjson::merge(healthy, run);
    VJ_CHECK(healthy.telem_verdict == "FAIL" && healthy.telemetry_cards.size() == 4);
    near(*healthy.power_avg_w, 220.0);
    near(*healthy.power_window_s, 0.125);
    benchjson::Record later;
    benchjson::parse_log(telemetry::report({full}), later);
    benchjson::merge(healthy, later);
    VJ_CHECK(healthy.telem_verdict == "FAIL" && healthy.power_window_kind == "short_observed");
    VJ_CHECK(benchjson::from_json(benchjson::to_json(healthy), restored, &err));
    VJ_CHECK(benchjson::to_json(restored) == benchjson::to_json(healthy));

    std::puts("test_power_telemetry: timestamp windows, exact cutoff, short/missing gates, cap drift, JSON OK\nPASS");
    return 0;
}
