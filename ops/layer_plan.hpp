// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Descriptor-driven step planner: ModelDesc + TpDesc + StepDesc -> per-GPU OpDesc sequence with barrier bits.
//
// Every dimension comes from ModelDesc (filled from GGUF metadata by a plugin). Sharding uses core/sharder
// (bandwidth-proportional, b=32 rows, GQA KV replication, sequence-sharded single-KV attention); GEMV split-K and
// decode split-KV come from core/launch_planner. N == 1 emits no AllReduce at all.
//
// Covered families: hybrid GDN+GQA MoE, pure MoE (shared expert optional, split-FF or whole-expert sharding),
// dense GQA. Pure data, CPU-compilable; the GPU backend binds OpKind -> kernel symbol.
#pragma once

#include "core/quant_repacker.hpp"
#include "ops/op_desc.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace vegajet::ops {

enum class MixerKind : uint8_t { Gqa = 0, Gdn = 1 };
enum class FfnKind : uint8_t { Dense = 0, Moe = 1 };
enum class ExpertShard : uint8_t {
    SplitFf = 0,      // every expert's FF width is split across GPUs (each token touches all GPUs)
    WholeExperts = 1, // experts are distributed whole (expert parallel)
};

struct AttnDims { uint32_t n_q_heads = 0, n_kv_heads = 0, head_dim = 0; };
struct GdnDims { uint32_t n_k_groups = 0, n_v_heads = 0, head_k_dim = 0, head_v_dim = 0, conv_kernel = 0; };
struct MoeDims {
    uint32_t n_experts = 0, top_k = 0, expert_ff = 0, shared_ff = 0;  // shared_ff == 0: no shared expert
    ExpertShard shard = ExpertShard::SplitFf;
};

struct ModelDesc {
    std::string arch;
    uint32_t d_model = 0, vocab = 0, dense_ff = 0;
    AttnDims attn;
    GdnDims gdn;
    MoeDims moe;
    std::vector<MixerKind> mixer;  // one entry per layer
    std::vector<FfnKind> ffn;      // one entry per layer
    quant::NativeFormat fmt_dense = quant::NativeFormat::VJ_Q4_DOT8;
    quant::NativeFormat fmt_expert = quant::NativeFormat::VJ_IQ_PACKED;
    quant::NativeFormat fmt_head = quant::NativeFormat::VJ_Q8_DP4A;
    uint32_t n_layers() const { return static_cast<uint32_t>(mixer.size()); }
};

struct TpDesc {
    std::vector<double> bw_gbs;   // one per GPU (sustained HBM GB/s): drives proportional sharding
    std::vector<uint32_t> n_cu;   // one per GPU: drives split-K / split-KV
    uint32_t n_gpus() const { return static_cast<uint32_t>(bw_gbs.size()); }
};

struct StepDesc {
    uint32_t m = 1;               // tokens in this step
    bool prefill = false;         // prefill uses GEMM / FlashPrefill / chunked GDN kinds
    uint32_t ctx_tokens = 4096;   // KV length seen by attention in this step
};

// AllReduce aux values.
constexpr uint32_t kArSumHidden = 0;   // sum partial hidden states (row-parallel output)
constexpr uint32_t kArFlashMerge = 1;  // merge split-sequence attention partials (max / sum-exp / weighted out)

struct PlannedOp {
    OpDesc op;
    bool barrier = true;  // wait for the previous op on the same GPU queue before starting
};

struct StepPlan {
    uint32_t n_gpus = 0;
    std::vector<PlannedOp> ops;  // global order; AllReduce has gpu = participating-rank bitmask
    OpGraph graph() const;
    uint32_t count(OpKind k) const;
    std::vector<PlannedOp> ops_for_gpu(uint32_t g) const;  // ops executed by rank g, AllReduce included
};

bool validate_model_desc(const ModelDesc& m, std::string* err);
bool validate_tp_desc(const TpDesc& tp, std::string* err);
// Returns false (and sets err) on an invalid descriptor; never asserts.
bool build_step_plan(const ModelDesc& m, const TpDesc& tp, const StepDesc& s, StepPlan& out, std::string* err);

// GEMV/GEMM kind for a native format (prefill steps always use GemmPrefill).
OpKind projection_kind(quant::NativeFormat f, bool prefill);
bool is_projection(OpKind k);

} // namespace vegajet::ops
