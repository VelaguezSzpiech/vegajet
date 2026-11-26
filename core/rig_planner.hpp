// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// `vegajet plan <model-file> --gpus N --vram 16|32`: CPU-only rig estimator. Reads the GGUF header only (no tensor
// payload, no device) and answers "does this model run on this rig, and how":
//
//  * repacking decisions per disk type, using core/quant_registry targets and the crossover law
//        repack old -> new   iff   bw_new / bw_old > bpw_new / bpw_old
//    with bw the effective streaming GB/s of each format's decode kernel. bw comes from the VALU roofline model in
//    core/quant_repacker (instructions per byte vs the ridge I* = n_cu * 64 * f_sclk * eta / BW_hbm), so a format
//    whose decode is VALU-bound streams below the HBM ceiling. Lossless re-layouts (IQ_PACKED, Q4_0 -> DOT8, Q8_0)
//    are applied whenever they are not slower. A repack the law accepts is still reverted if it breaks the budget;
//  * an exact per-GPU VRAM breakdown for tensor parallelism over N GPUs (dense, routed experts, embedding / head,
//    norms, KV cache, recurrent state, reserve);
//  * KV / context headroom: the largest context that fits after the weights;
//  * whether routed MoE experts must be offloaded to host RAM, and how much.
#pragma once

#include "core/gguf_reader.hpp"
#include "core/quant_registry.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace vegajet::rig {

enum class Kind : uint8_t { Embed, Output, Dense, RoutedExpert, SharedExpert, Router, Norm };
const char* kind_name(Kind k);
Kind classify_tensor(const std::string& name, uint32_t n_dims);

struct TensorEntry {
    std::string name;
    uint32_t type = 0;  // ggml id
    uint32_t n_dims = 2;
    uint64_t elems = 0;
};

struct ModelSummary {
    std::string arch;
    uint64_t n_layer = 0, n_embd = 0, n_head = 0, n_head_kv = 0, head_dim = 0;
    uint64_t n_ctx_train = 0, n_expert = 0, n_expert_used = 0;
    uint64_t n_kv_layers = 0;       // layers with a KV cache (hybrid models: full-attention layers only)
    uint64_t ssm_state_bytes = 0;   // recurrent state per sequence, all layers (0 if none)
    std::vector<TensorEntry> tensors;
    uint64_t total_params() const;
    uint64_t disk_bytes() const;
};

// Header-only summary of an open GGUF file.
bool summarize_gguf(const GgufFile& f, ModelSummary& out, std::string* err = nullptr);

struct RigSpec {
    uint32_t gpus = 1;            // 1..256, tensor parallel
    uint32_t vram_gib = 32;       // 16 or 32 per card
    uint32_t n_cu = 60;           // 60 (MI50) or 64 (MI60)
    double bw_gbs = 800.0;        // sustained HBM2 streaming per card
    uint64_t ctx = 8192;          // requested context (tokens, one sequence)
    uint32_t kv_bits = 16;        // 16 (F16) or 8
    uint64_t reserve_bytes = 2ull << 30;  // per GPU: activations, scratch, allocator slack
};

enum class Decision : uint8_t { Relayout, Repack, KeepCrossover, KeepBudget, KeepNoHook, KeepFloat };
const char* decision_name(Decision d);

struct TypeDecision {
    uint32_t disk_type = 0;
    quant::NativeFormat target = quant::NativeFormat::VJ_Q8_DP4A;
    Decision decision = Decision::KeepNoHook;
    uint64_t tensors = 0, elems = 0;
    uint64_t disk_bytes = 0, final_bytes = 0;
    double bpw_old = 0, bpw_new = 0, bw_old = 0, bw_new = 0;
    bool lossless = false;
};

struct GpuBreakdown {
    uint64_t vram = 0, dense = 0, experts = 0, embed_head = 0, norms = 0, kv = 0, state = 0, reserve = 0;
    uint64_t used() const { return dense + experts + embed_head + norms + kv + state + reserve; }
    int64_t free_bytes() const { return static_cast<int64_t>(vram) - static_cast<int64_t>(used()); }
};

struct RigPlan {
    bool fits = false;                 // fits on the GPUs, possibly with host-RAM expert offload
    bool moe_offload_required = false;
    double expert_resident_fraction = 1.0;
    uint64_t host_expert_bytes = 0;    // routed expert bytes kept in pinned host RAM
    uint64_t kv_bytes_per_token = 0;   // per GPU
    uint64_t max_ctx = 0;              // largest context that fits (capped at the training context, if known)
    int64_t shortfall_bytes = 0;       // per GPU, when it does not fit
    double i_star = 0;
    std::vector<TypeDecision> decisions;
    std::vector<GpuBreakdown> gpus;
    std::string verdict;
};

double effective_bw_gbs(double inst_per_byte, const RigSpec& rig);  // roofline: BW / max(1, ipb / I*)
RigPlan plan_rig(const ModelSummary& m, const RigSpec& rig);
std::string format_rig_plan(const ModelSummary& m, const RigSpec& rig, const RigPlan& p);
// The standard matrix: 1x16, 2x16, 2x32, 3x32, 4x32, 8x32 GiB.
std::vector<RigSpec> standard_rigs(const RigSpec& base);
std::string format_rig_matrix(const ModelSummary& m, const std::vector<RigSpec>& rigs);

} // namespace vegajet::rig
