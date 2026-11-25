// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Serving observability: a lock-free Prometheus registry, a JSONL request log, and read-only GPU sysfs telemetry.
//
//   Registry        counters / gauges / histograms on relaxed atomics (no lock on any hot path; a scrape reads
//                   whatever is current). Request metrics: requests, failures, cancellations, prompt / generated
//                   tokens, TTFT / TPOT / end-to-end histograms, active batch size, queue depth. Engine metrics fed
//                   by whoever has them: prefix-cache hit rate (hit / looked-up prompt tokens), MoE expert LRU hit
//                   rate, speculative decoding mean_accept (accepted draft tokens per verify step) and acceptance
//                   rate. exposition() renders the Prometheus text format 0.0.4.
//   instrument()    wraps a serve::GenerateFn: measures TTFT (first piece), TPOT ((last - first) / (n - 1)), counts
//                   pieces, tracks the active-request gauge, and writes one JSONL record per request.
//   JsonlLogger     one JSON object per line ({"ts","id","status","finish","prompt_tokens","completion_tokens",
//                   "ttft_ms","tpot_ms","e2e_ms","cancelled"}), each line a single append write: lines from
//                   concurrent requests never interleave. No prompt or completion text is ever logged.
//   read_gpu_sysfs  amdgpu cards under <root>/class/drm/cardN (vendor 0x1002 only): hwmon temperatures (edge,
//                   junction, mem = TMem), power (power1_average, else power1_input), and the RAS ECC counters of six
//                   blocks (umc, sdma, gfx, mmhub, pcie_bif, hdp: "ue: N" / "ce: N"). Reads only, never writes, and
//                   reports cards by index (no PCI address).
//   parse_exposition / validate_exposition   the text format back into samples (vegajet top, tests).
#pragma once

#include "core/server.hpp"

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace vegajet::metrics {

class Histogram {
public:
    static constexpr size_t kMaxBuckets = 24;
    explicit Histogram(std::vector<double> upper_bounds);  // seconds, ascending; +Inf is implicit
    void observe(double seconds);
    struct Snapshot {
        std::vector<double> bounds;
        std::vector<uint64_t> counts;  // per bucket (not cumulative); counts.size() == bounds.size() + 1
        double sum = 0.0;
        uint64_t count = 0;
        double quantile(double q) const;  // linear inside the bucket; NaN when empty
    };
    Snapshot snapshot() const;

private:
    std::vector<double> bounds_;
    std::array<std::atomic<uint64_t>, kMaxBuckets + 1> counts_{};
    std::atomic<uint64_t> sum_ns_{0};
    std::atomic<uint64_t> count_{0};
};

struct RequestRecord {
    uint64_t id = 0;
    int status = 200;                 // 200 ok, 499 client gone, 500 generator error
    std::string finish;               // stop / length / cancelled / error / tool_calls
    uint32_t prompt_tokens = 0, completion_tokens = 0;
    double ttft_ms = 0.0, tpot_ms = 0.0, e2e_ms = 0.0;
    bool cancelled = false;
};

std::string to_jsonl(const RequestRecord& r, double unix_ts);

class JsonlLogger {
public:
    JsonlLogger() = default;
    ~JsonlLogger();
    JsonlLogger(const JsonlLogger&) = delete;
    JsonlLogger& operator=(const JsonlLogger&) = delete;
    bool open(const std::string& path, std::string* err = nullptr);  // append, created 0600
    bool is_open() const { return fd_ >= 0; }
    void log(const RequestRecord& r);
    uint64_t lines() const { return lines_.load(std::memory_order_relaxed); }

private:
    int fd_ = -1;
    std::atomic<uint64_t> lines_{0};
};

struct GpuHw {
    int card = -1;                     // N of /sys/class/drm/cardN
    double temp_edge_c = NAN, temp_junction_c = NAN, temp_mem_c = NAN;
    double power_w = NAN;
    struct Ras {
        std::string block;
        bool present = false;
        uint64_t ue = 0, ce = 0;
    };
    std::vector<Ras> ras;              // the six blocks, in kRasBlocks order
};
extern const char* const kRasBlocks[6];
std::vector<GpuHw> read_gpu_sysfs(const std::string& root = "/sys");

class Registry {
public:
    Registry();
    // request lifecycle (instrument() calls these)
    void request_started();
    void request_finished(const RequestRecord& r);
    // engine-fed metrics
    void add_prefix_cache(uint64_t hit_tokens, uint64_t lookup_tokens);
    void add_moe_lru(uint64_t hits, uint64_t misses);
    void add_spec(uint64_t drafted, uint64_t accepted, uint64_t verify_steps);
    void set_batch_size(uint32_t n);   // overrides the active-request count when an engine batches internally
    void set_queue_depth(uint32_t n);
    // Prometheus text; hardware samples appended when hw is given.
    std::string exposition(const std::vector<GpuHw>* hw = nullptr) const;

    uint64_t requests() const { return requests_.load(std::memory_order_relaxed); }
    uint32_t active() const { return static_cast<uint32_t>(active_.load(std::memory_order_relaxed)); }
    const Histogram& ttft() const { return ttft_; }
    const Histogram& tpot() const { return tpot_; }

private:
    std::atomic<uint64_t> requests_{0}, failed_{0}, cancelled_{0}, prompt_tokens_{0}, gen_tokens_{0};
    std::atomic<int64_t> active_{0}, batch_override_{-1}, queue_depth_{0};
    std::atomic<uint64_t> pc_hit_{0}, pc_lookup_{0}, moe_hit_{0}, moe_miss_{0};
    std::atomic<uint64_t> spec_drafted_{0}, spec_accepted_{0}, spec_steps_{0};
    Histogram ttft_, tpot_, e2e_;
};

// The HTTP front end's own counters (OpenAiServer::stats()) as vegajet_http_* families.
std::string server_exposition(const serve::OpenAiServer::Stats& s);

// Wraps a generator with request metrics and (optionally) the JSONL log. reg / log must outlive the result.
serve::GenerateFn instrument(serve::GenerateFn inner, Registry& reg, JsonlLogger* log = nullptr);

// --------------------------------------------------------------------------------------------- exposition parsing
struct Sample {
    std::string name;
    std::map<std::string, std::string> labels;
    double value = 0.0;
};
bool parse_exposition(std::string_view text, std::vector<Sample>& out, std::string* err = nullptr);
// Format checks beyond parsing: metric / label names, one HELP and TYPE per family before its samples, counter
// values >= 0, histogram buckets cumulative and ending in le="+Inf" equal to _count.
bool validate_exposition(std::string_view text, std::string* err = nullptr);
// Value of the first sample with this name whose labels include `match` (NaN if absent).
double find_sample(const std::vector<Sample>& s, std::string_view name, const std::map<std::string, std::string>& match = {});

} // namespace vegajet::metrics
