// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// `vegajet bench --json-out <path>`: machine-readable bench scorecard, schema "vegajet.bench.v1".
//
// Sources, in order of trust:
//   * the engine's own log lines, captured while the bench runs (StdoutTee) and parsed with the same formats
//     tools/scorecard.py reads: [BENCH tg128 raw M=1], [BENCH pp512|pp2048 M=..], [BENCH tg128|tg512 <mode>] for
//     Suffix / MTP / Hybrid / DFlash2, [PARITY code vs Tier-1 llama.cpp], [TELEM] power / TMem, and the proposed
//     [KLD dot8 prefill] <kld> line (the engine computes the held-out dot8 KLD offline today; the line is how a
//     bench run reports it);
//   * read-only sysfs snapshots before / after the run (core/doctor_rules): per-GPU TMem, power cap, and the 6
//     RAS blocks (umc, gfx, hdp, mmhub, pcie_bif, sdma) as ue / de / ce deltas over the run;
//   * the CLI itself: model file name (never its directory), architecture, TP, flags (path arguments reduced to
//     their file names), GPU selection, prefill path (w4a8, or dot8 with --prefill-dot8), code-object spill.
// Nothing is estimated: a metric the run did not produce is null.
//
// JSON layout (key names are what tools/regression_gate.py classifies: *_tok_s throughput, *kld*, parity objects
// {match,total}, first_diff, spill*, tmem*, ue / de):
//   { "schema", "meta": {...}, "throughput": {raw_m1_tok_s, pp512_w4a8_tok_s, pp2048_w4a8_tok_s, pp512_dot8_tok_s,
//     pp2048_dot8_tok_s, "spec": {<mode>_tok_s}}, "decode": {...}, "prefill_batch_m": {...}, "speculative":
//     {<mode>: {label, tokens, speedup_vs_raw, acceptance_pct}}, "kld": {dot8_prefill_kld}, "parity":
//     {code_vs_tier1: {match,total,verdict}, <mode>_lossless: {match,total}}, "first_diff": {<mode>: n},
//     "spill": {private_segment_bytes_max}, "telemetry": {...}, "gpus": [{id, tmem_c, power_cap_w}],
//     "ras": {gpuN: {block: {ue, de, ce}}} }
// Writing to an existing file of the same schema, model and commit merges: measured values of this run replace
// the stored ones, the rest is kept (so a W4A8 run and a --prefill-dot8 run fill one scorecard), telemetry keeps
// the worst case and RAS deltas add up. telemetry.cards retains each run's timestamped per-card observations.
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace vegajet::benchjson {

constexpr const char* kSchema = "vegajet.bench.v1";

struct Spec {
    std::string key;    // "mtp_k4", "dflash2_k7_code", "tg512_suffix", ...
    std::string label;  // as printed: "MTP K=4"
    uint32_t tokens = 128;
    double tok_s = 0.0;
    std::optional<double> speedup, acceptance_pct;
    std::optional<uint32_t> match, total;
    std::optional<int32_t> first_diff;
};

struct RasBlock {
    std::string block;
    uint64_t ue = 0, de = 0, ce = 0;
    bool available = false;
};

struct GpuState {
    std::string id;  // "gpu0", "gpu1", ... (sysfs PCI order; never a bus address)
    double tmem_c = -1, power_cap_w = -1;
    std::vector<RasBlock> ras;
};

struct TelemetryCard {
    // One row per DRM card per run; merging retains rows, not fictitious combined means.
    std::string id, timestamp_utc, window_kind, hold, verdict;
    std::optional<double> cap_w, cap_min_w, cap_max_w, mean_w, avg_w, window_s;
    std::optional<double> avg1s_w, window1s_s, peak_w, tmem_max_c, observed_s;
    std::optional<uint64_t> samples, missing;
};

struct Record {
    // meta
    std::string vegajet_version = "0.1.2", commit = "unknown", command = "bench", model, architecture, flags;
    std::string gpu_selection = "all", prefill_path = "w4a8", plan_status, error, timestamp_utc;
    uint32_t tp = 1;
    bool executed = false;
    // decode
    std::optional<double> raw_tok_s, raw_ms_per_tok, eff_hbm2_gbs, hbm2_pct, hbm2_ceiling_gbs, real_read_gb_per_tok;
    std::optional<uint32_t> dispatches_per_tok;
    // prefill: "pp512_w4a8" -> tok/s, and the batch M printed with it ("pp512" -> M)
    std::map<std::string, double> prefill;
    std::map<std::string, uint32_t> prefill_m;
    std::optional<double> dot8_kld;
    std::vector<Spec> spec;
    std::optional<uint32_t> parity_match, parity_total;
    std::string parity_verdict;
    std::optional<double> power_avg_w, power_avg_limit_w, power_peak_w, tmem_max_c;
    std::optional<double> power_mean_w, power_window_s, power_avg1s_w;
    std::string power_window_kind, power_hold;
    std::vector<TelemetryCard> telemetry_cards;
    std::string telem_verdict;
    std::optional<uint64_t> spill_bytes_max;
    std::vector<GpuState> gpus_before, gpus_after;
};

bool parse_line(const std::string& line, Record& r);  // true when the line was recognized
void parse_log(const std::string& text, Record& r);
std::string to_json(const Record& r);
bool from_json(const std::string& text, Record& r, std::string* err);
void merge(Record& into, const Record& run);  // `run`'s measured values win; telemetry worst case; RAS deltas add
// Writes `r` to path (merging with an existing compatible file). false + err on I/O failure.
bool write_file(const std::string& path, const Record& r, std::string* err, bool* merged = nullptr);

// Read-only snapshot of every gfx906 under <root>/sys/bus/pci/devices (TMem, power cap, 6-block RAS counters).
std::vector<GpuState> snapshot(const std::string& root);
// "a/b/model-file" -> "model-file"; arguments containing '/' are reduced the same way in meta.flags.
std::string sanitize_flags(const std::vector<std::string>& args);
std::string utc_now();

// Tees stdout (fd 1) into a buffer while still printing it, for parsing the engine's log of the run.
class StdoutTee {
public:
    StdoutTee() = default;
    ~StdoutTee();
    StdoutTee(const StdoutTee&) = delete;
    StdoutTee& operator=(const StdoutTee&) = delete;
    bool begin();
    std::string end();

private:
    int saved_ = -1, rd_ = -1;
    std::thread th_;
    std::string buf_;
};

}  // namespace vegajet::benchjson
