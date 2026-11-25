// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// The frozen 5-method model-plugin contract. A plugin turns GGUF metadata into a load plan and,
// per step, into an ops::OpGraph that the runtime executes. Plugins never touch devices directly.
#pragma once

#include "core/gguf_reader.hpp"
#include "ops/op_desc.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace vegajet {

struct GpuTarget {
    uint32_t n_cu = 60;               // compute units on this device
    double bw_hbm_gbs = 800.0;        // sustained HBM bandwidth (GB/s) used for proportional sharding
    double sclk_ghz = 1.725;          // sustained shader clock for the roofline ridge (I*)
    uint64_t vram_bytes = 32ull << 30;
};

struct LoadOptions {
    std::vector<GpuTarget> gpus = {GpuTarget{}};
    uint32_t context_length = 32768;  // planned KV capacity in tokens
    uint64_t kv_scratch_reserve_bytes = 2ull << 30;  // per GPU, activations + scratch
    bool dry_run = true;              // plan only: no weight payloads are read or repacked
    std::string draft_gguf;           // optional external draft model (e.g. a DFlash/MTP GGUF)
    double batch_reuse_factor = 1.0;  // >1 when steps verify several tokens per weight read
};

struct LoadReport {
    bool ok = false;
    std::string error;
    std::string architecture;
    uint64_t weight_bytes_disk = 0;
    std::vector<uint64_t> weight_bytes_per_gpu;  // after repack + sharding
    std::vector<uint64_t> kv_bytes_per_gpu;
    std::vector<uint64_t> state_bytes_per_gpu;
    std::vector<std::string> notes;
};

struct StepResult {
    ops::OpGraph graph;               // ops to execute for this step
    uint32_t tokens_in = 0;
    uint32_t draft_tokens = 0;        // only for draft_step
};

class VegaJetModel {
public:
    virtual ~VegaJetModel() = default;
    // True when this plugin understands the file's general.architecture and tensor layout.
    virtual bool can_load(const GgufFile& f) const = 0;
    // Builds the full load plan (repack formats, shards, KV/state pages); dry_run skips payload reads.
    virtual LoadReport load(GgufFile& f, const LoadOptions& opt) = 0;
    // Emits the op graph for a prompt chunk of `tokens`.
    virtual StepResult prefill(const std::vector<int32_t>& tokens) = 0;
    // Emits the op graph for one autoregressive token (or a verified draft tree of `verify_len`).
    virtual StepResult decode_step(int32_t token, uint32_t verify_len = 1) = 0;
    // Proposes up to `max_draft` tokens (external draft model or suffix tree) and emits its graph.
    virtual StepResult draft_step(const std::vector<int32_t>& context, uint32_t max_draft) = 0;
    virtual std::string name() const = 0;
};

} // namespace vegajet
