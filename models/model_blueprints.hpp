// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Declarative model blueprints: GGUF metadata -> per-layer structure (mixer / FFN kind), the full expected tensor
// map with shapes, offloadable embedding tables, and a lowering to HEAD's ops::ModelDesc for the step planner.
//
//   qwen3.8-flash-next   hybrid gated-delta-net + gated GQA (pattern from attention.recurrent_layers or
//                        full_attention_interval), routed MoE + shared expert, optional per-layer embeddings (PLE)
//   hy4-flash            MLA + sliding-window GQA (pattern from attention.sliding_window_pattern: 1 = SWA; the
//                        reference configuration is 12 MLA + 30 SWA), LatentMoE (experts run in a latent width:
//                        ffn_latent_down d->l, experts l->ff->l, ffn_latent_up l->d; reference 512 experts top-16),
//                        optional leading dense blocks and an Engram n-gram table
//   qwen3.8-27b          dense GQA with QK norm
//
// Rules (same as agent/MODEL_TEMPLATE): no model dimension is hard-coded; every number comes from GGUF keys, and a
// structural key the blueprint needs is an error when missing (no guessing). Tensor names follow llama.cpp / HEAD
// conventions (qwen3next / ornith loader, deepseek2 MLA, qwen3); LatentMoE and Engram tensor / key names are
// PROPOSED (no public GGUF exists to confirm them) and are listed in INTEGRATION.md. The general.architecture
// values a blueprint accepts are likewise expectations to confirm once real files exist.
#pragma once

#include "core/async_table_prefetch.hpp"
#include "core/gguf_reader.hpp"
#include "ops/layer_plan.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace vegajet::blueprint {

enum class Mixer : uint8_t { Gdn = 0, Gqa = 1, Swa = 2, Mla = 3 };
enum class Ffn : uint8_t { Dense = 0, Moe = 1, LatentMoe = 2 };
const char* mixer_name(Mixer m);
const char* ffn_name(Ffn f);

struct Dims {
    uint32_t d_model = 0, vocab = 0, n_layers = 0, nextn_layers = 0;
    uint32_t n_q_heads = 0, n_kv_heads = 0, head_dim = 0;         // full / gated GQA
    uint32_t swa_window = 0, swa_kv_heads = 0, swa_head_dim = 0;  // sliding-window layers
    uint32_t q_lora = 0, kv_lora = 0, qk_rope = 0, qk_nope = 0, v_head_dim = 0;  // MLA
    uint32_t gdn_k_groups = 0, gdn_v_heads = 0, gdn_k_dim = 0, gdn_v_dim = 0, conv_kernel = 0;
    uint32_t dense_ff = 0, n_experts = 0, top_k = 0, expert_ff = 0, shared_ff = 0, latent = 0, leading_dense = 0;
    uint32_t ple_dim = 0;
    uint64_t engram_rows = 0;
    uint32_t engram_dim = 0, engram_ngram = 0;
};

struct TensorSpec {
    std::string name;
    std::vector<std::vector<uint64_t>> shapes;  // accepted shapes (ne order); 0 = any extent
    bool required = true;
};

struct TableSpec {
    std::string tensor;
    uint64_t rows = 0;
    uint32_t row_elems = 0;
    std::string purpose;  // "ple" (row = token id) or "engram" (row = n-gram bucket)
};

struct Description {
    std::string blueprint;
    std::string arch;
    Dims dims;
    std::vector<Mixer> mixer;  // one per main layer (MTP / nextn layers excluded)
    std::vector<Ffn> ffn;
    std::vector<TensorSpec> tensors;
    std::vector<TableSpec> tables;
    std::vector<std::string> notes;
    uint32_t count(Mixer m) const;
    uint32_t count(Ffn f) const;
};

struct TensorCheck {
    uint32_t checked = 0, present = 0, missing_required = 0, missing_optional = 0, bad_shape = 0, unexpected = 0;
    std::vector<std::string> problems;  // first 32
    bool ok() const { return missing_required == 0 && bad_shape == 0; }
};

struct Blueprint {
    const char* name;
    std::vector<std::string> archs;
    bool (*describe)(const GgufFile& f, Description& out, std::string* err);
};

const std::vector<Blueprint>& blueprints();
const Blueprint* find_by_name(const std::string& name);
const Blueprint* find_for(const GgufFile& f);  // by general.architecture

// bp == nullptr: pick by general.architecture.
bool describe(const GgufFile& f, Description& out, std::string* err, const Blueprint* bp = nullptr);
TensorCheck check_tensors(const GgufFile& f, const Description& d);
// HEAD's ModelDesc has one attention geometry and no SWA / MLA / latent-MoE kinds; the lowering keeps what the
// planner can use and lists every approximation in `notes` (lossless = notes empty).
bool lower(const Description& d, ops::ModelDesc& out, std::vector<std::string>& notes, std::string* err);
// Row-prefetch descriptor for a 2D GGUF table tensor (F16 or Q4_0) of an open file descriptor.
bool table_desc(const GgufFile& f, const TableSpec& t, int fd, prefetch::TableDesc& out, std::string* err);

} // namespace vegajet::blueprint
