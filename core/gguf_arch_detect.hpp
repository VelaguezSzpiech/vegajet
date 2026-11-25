// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Structural GGUF family classification for the four supported stacks, using only metadata keys and
// tensor-name patterns (never the architecture string as a whitelist):
//
//   DenseGqa       GQA/SWA attention + dense FFN on every main layer
//   PureMoe        GQA/SWA attention + routed experts on every main layer
//   HybridGdnMoe   gated-delta-net and GQA/SWA layers mixed by attention.recurrent_layers /
//                  full_attention_interval, routed experts (+ optional shared expert)
//   MlaMoe         MLA latent-KV layers (attention.kv_lora_rank + rope.dimension_count) mixed with
//                  GQA/SWA layers via attention.sliding_window_pattern, routed experts, optional
//                  leading dense blocks and LatentMoE (moe_latent_size + ffn_latent_down/up)
//
// Conflicting or incomplete metadata is rejected (unknown family + problems) instead of guessed:
// expert_count without expert tensors, ssm.* keys without GDN tensors, kv_lora_rank without MLA tensors,
// recurrent/pattern arrays that disagree with the tensor layout, dense and expert FFN in one block,
// GDN and MLA mixers in one block.
#pragma once

#include "core/gguf_reader.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace vegajet::arch {

enum class Family : uint8_t { Unknown = 0, DenseGqa, PureMoe, HybridGdnMoe, MlaMoe };

const char* family_name(Family family);

struct Detection {
    Family family = Family::Unknown;
    std::string arch;
    uint32_t n_layers = 0;  // main layers (nextn_predict_layers excluded)
    uint32_t gqa_layers = 0, swa_layers = 0, gdn_layers = 0, mla_layers = 0;
    uint32_t dense_ffn_layers = 0, moe_ffn_layers = 0;
    uint32_t n_experts = 0, top_k = 0;
    uint32_t n_q_heads = 0, n_kv_heads = 0;
    std::vector<std::string> evidence;
    std::vector<std::string> problems;
    bool ok() const { return family != Family::Unknown && problems.empty(); }
};

// Classifies an already-opened GGUF. Returns out.ok(); on failure `err` carries the first problem.
bool detect(const GgufFile& file, Detection& out, std::string* err);
bool detect_file(const std::string& path, Detection& out, std::string* err);

} // namespace vegajet::arch
