// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Engine providers for `vegajet serve <model-file>`.
//
// A provider turns a model file + GPU selection into an IEngine (single-sequence streaming) and, optionally, a
// sched::BatchEngine (continuous batching through core/request_scheduler). Providers register per GGUF
// general.architecture. The registry itself never opens a device; only a provider's create function may.
//
// State of HEAD (b74f820): no streaming GPU engine exists yet. models/ornith/ornith_gpu.* exposes bench / parity /
// microbench entry points (run_ornith_gpu, run_ornith_ar_bench, run_ornith_gemv_ab_bench), none of which streams
// tokens through a callback, so nothing is registered and create() fails with a precise message (exit code 3 in the
// CLI). The contract a GPU provider has to meet is spelled out in INTEGRATION.md ("Registering a GPU engine").
#pragma once

#include "core/engine_interface.hpp"
#include "core/request_scheduler.hpp"

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace vegajet::serve {

struct EngineSpec {
    std::string model_path;             // the model file (GGUF)
    std::string architecture;           // GGUF general.architecture
    std::vector<uint32_t> gpu_indices;  // selected GPUs (positions in the gfx906 enumeration)
    uint32_t tp = 1;
    uint32_t context = 8192;
    std::string hsaco_dir;
    std::string tp_shards, tp_plan; // validated JSON text in EngineSpec; file paths in ServeConfig
};

struct EngineHandle {
    std::unique_ptr<IEngine> engine;            // always set on success
    std::unique_ptr<sched::BatchEngine> batch;  // optional: enables --slots > 1
    std::string description;
};

using EngineProvider = std::function<bool(const EngineSpec&, EngineHandle&, std::string* err)>;

class EngineRegistry {
public:
    static EngineRegistry& instance();
    void add(const std::string& architecture, EngineProvider p);
    bool has(const std::string& architecture) const;
    std::vector<std::string> architectures() const;
    // false + err when no provider handles spec.architecture (or the provider fails).
    bool create(const EngineSpec& spec, EngineHandle& out, std::string* err) const;

private:
    mutable std::mutex mu_;
    std::map<std::string, EngineProvider> providers_;
};

// Batch engine behind `vegajet serve --mock --slots N`: replays MockEngine's canned reply per request, so its
// output is token-for-token identical to the single-sequence mock while exercising the scheduler, paged KV and
// ServeLoop on the CPU. Called from the ServeLoop thread only.
class MockBatchEngine final : public sched::BatchEngine {
public:
    explicit MockBatchEngine(MockEngine& m);  // records the canned reply by running m.generate() once
    std::vector<sched::StepOutput> step(const sched::StepPlan& plan) override;
    int32_t eos() const { return eos_; }

private:
    int32_t canned(uint32_t g) const { return g < reply_.size() ? reply_[g] : eos_; }
    std::vector<int32_t> reply_;
    int32_t eos_ = -1;
    std::map<uint64_t, uint32_t> prompt_len_;  // request -> position of its first generated token
};

}  // namespace vegajet::serve
