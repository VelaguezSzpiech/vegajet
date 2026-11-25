// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// `vegajet top`: a zero-dependency ANSI dashboard for a running `vegajet serve` (its GET /metrics) and / or the local
// GPUs' sysfs telemetry. Read-only by construction: it issues HTTP GETs and reads sysfs files, never writes either.
//
//   vegajet top [--url http://127.0.0.1:8080] [--sysfs] [--sysfs-root DIR] [--interval S] [--frames N] [--once]
//
// Shows requests and rates (req/s, generated tok/s from counter deltas), TTFT p50 / p95 and TPOT p50 (from the
// histogram buckets), active batch / queue, prefix-cache and MoE LRU hit rates, speculative mean_accept, HTTP
// counters, and per GPU: edge / junction / memory temperature, power and the six RAS blocks' UE / CE counts (from
// the server's vegajet_gpu_* samples, or read directly with --sysfs).
#pragma once

#include "core/metrics.hpp"

#include <string>
#include <vector>

namespace vegajet::cli {

struct TopView {
    bool server = false;               // metrics were read from a server
    std::string source;                // URL or "sysfs"
    std::vector<metrics::Sample> samples;
    std::vector<metrics::GpuHw> gpus;  // from samples, or sysfs
};

// GPU telemetry carried in vegajet_gpu_* samples.
std::vector<metrics::GpuHw> gpus_from_samples(const std::vector<metrics::Sample>& s);

// One frame. prev (may be null) and dt_s give rates; width is the terminal width (>= 60); color = ANSI styles.
std::string render_top(const TopView& cur, const TopView* prev, double dt_s, int width, bool color);

int run_top(const std::vector<std::string>& args);

}  // namespace vegajet::cli
