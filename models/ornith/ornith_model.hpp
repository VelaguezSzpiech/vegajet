// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Ornith / Qwen3.5-MoE hybrid plugin (GGUF architecture "qwen35moe").
//
// Layout read from GGUF metadata (never hardcoded): N trunk blocks with a recurrent pattern
// (`attention.recurrent_layers`, typically 3 gated-delta-net blocks per full-attention block), plus
// `nextn_predict_layers` MTP blocks at the end. Every block has a routed MoE FFN with a shared expert.
//
// load() wires: gguf_reader -> quant_repacker knapsack (native formats within VRAM headroom) ->
// sharder (bandwidth-proportional TP, GQA boundary KV replication, GDN group split) -> kv_cache (Q8_0
// paged KV for the attention blocks) -> ssm_state (ReplaySSM for the recurrent blocks).
// Steps emit ops::OpGraph only: no hyper-connections, no host-side per-layer embeddings; GDN uses the
// chunked WY prefill + recurrent decode ops, attention uses Wave64 FlashDecoding, MoE is one fused op.
// Drafting: an external DFlash GGUF (LoadOptions::draft_gguf) or model-free SuffixDecoding, with the
// MoE K* governor choosing the draft length.
#pragma once

#include "core/kv_cache.hpp"
#include "core/model_interface.hpp"
#include "core/quant_repacker.hpp"
#include "core/sharder.hpp"
#include "core/spec_engine.hpp"
#include "core/ssm_state.hpp"

#include <memory>
#include <string>
#include <vector>

namespace vegajet::ornith {

struct Hparams {
    std::string arch;
    uint32_t n_block = 0, n_nextn = 0, n_trunk = 0;
    uint32_t n_embd = 0, n_head = 0, n_head_kv = 0, head_dim_k = 0, head_dim_v = 0;
    uint32_t rope_dims = 0;
    std::vector<uint32_t> rope_sections;
    double rope_base = 0, rms_eps = 0;
    uint32_t n_expert = 0, n_expert_used = 0, n_ff_exp = 0, n_ff_shexp = 0;
    uint32_t ssm_conv = 0, ssm_state = 0, ssm_groups = 0, ssm_v_heads = 0, ssm_inner = 0;
    uint32_t full_attn_interval = 0;
    uint64_t context_length = 0;
    uint64_t vocab = 0;
    std::vector<bool> recurrent;  // per block (trunk + nextn)
    uint32_t n_gdn = 0, n_attn = 0;  // trunk counts
    bool is_recurrent(uint32_t il) const { return il < recurrent.size() && recurrent[il]; }
};

struct DraftHparams {
    std::string arch;  // "dflash"
    uint32_t n_layer = 0, n_embd = 0, n_head = 0, n_head_kv = 0, head_dim = 0, n_ff = 0, block_size = 0;
    uint32_t conv_kernel_size = 0, conv_group_size = 0, selector_rank = 0, selector_top_k = 0, sliding_window = 0;
    bool     attention_causal = false;
    double   rope_base = 1e7, rms_eps = 1e-6;
    uint64_t mask_token = 0;
    std::vector<uint32_t> target_layers;
    bool is_dflash2() const { return selector_top_k > 0u; }
};

enum class ShardKind : uint32_t { Replicated, Rank0, RowSplit, ColSplit, QHeads, KvHeads, GdnGroups, Experts };

struct TensorPlan {
    std::string name;
    int32_t layer = -1;
    quant::TensorRole role = quant::TensorRole::Other;
    quant::DiskQuantType disk = quant::DiskQuantType::F32;
    uint64_t in = 0, out = 0, experts = 1;  // ne0 (K), ne1 (N), ne2
    quant::NativeFormat format = quant::NativeFormat::VJ_IQ_PACKED;
    bool keep_float = false;   // vectors / router gates stay F32 on device
    bool fwht64 = false;
    ShardKind shard = ShardKind::Replicated;
    uint64_t bytes_total = 0;  // after repack, before sharding
    std::vector<uint64_t> bytes_per_gpu;
    std::vector<uint64_t> local_out;  // output rows (or experts) per GPU
};

enum class DraftSource : uint32_t { None, ExternalDFlash, Suffix };

class OrnithModel final : public VegaJetModel {
public:
    bool can_load(const GgufFile& f) const override;
    LoadReport load(GgufFile& f, const LoadOptions& opt) override;
    StepResult prefill(const std::vector<int32_t>& tokens) override;
    StepResult decode_step(int32_t token, uint32_t verify_len) override;
    StepResult draft_step(const std::vector<int32_t>& context, uint32_t max_draft) override;
    std::string name() const override { return "ornith"; }

    const Hparams& hparams() const { return hp_; }
    const DraftHparams& draft_hparams() const { return dhp_; }
    DraftSource draft_source() const { return draft_src_; }
    const std::vector<TensorPlan>& tensor_plan() const { return plan_; }
    const TensorPlan* find_plan(const std::string& name) const;
    const shard::AttnShardPlan& attn_shards() const { return attn_; }
    const std::vector<shard::GdnShard>& gdn_shards() const { return gdn_; }
    const std::vector<kv::KvPlan>& kv_plans() const { return kv_; }
    spec::MoeKGovernor& governor() { return gov_; }
    spec::SuffixTree& suffix_tree() { return suffix_; }
    ssm::ReplayRing& replay_ring() { return ring_; }

    // Repacks one tensor's shard for `rank` from the mmapped GGUF into its planned native format and
    // returns the dequantized FP32 shard (rows x in) plus the disk-decoded reference for parity checks.
    bool repack_shard_for_test(GgufFile& f, const std::string& tensor, uint32_t rank, uint32_t max_rows,
                               std::vector<float>& native_f32, std::vector<float>& disk_f32, std::string* err) const;

    // After a verify step of `proposed` drafts: `verified` = accepted drafts + the bonus token. Updates the
    // K* governor and suffix tree, and returns how the device must roll back the recurrent state.
    ssm::ReplayRing::Rollback accept_draft(const std::vector<int32_t>& verified, uint32_t proposed);

    static constexpr uint32_t kPrefillChunk = 512;

private:
    bool parse_hparams(const GgufFile& f, std::string* err);
    bool parse_draft(const std::string& path, LoadReport& rep);
    void emit_layer(ops::OpGraph& g, uint32_t il, uint32_t m, bool prefill) const;
    void emit_gemv(ops::OpGraph& g, const std::string& tensor, int32_t layer, uint32_t m, bool prefill) const;
    void emit_allreduce(ops::OpGraph& g, int32_t layer, uint32_t m) const;

    Hparams hp_;
    DraftHparams dhp_;
    DraftSource draft_src_ = DraftSource::None;
    LoadOptions opt_;
    std::vector<double> bw_;
    std::vector<TensorPlan> plan_;
    shard::AttnShardPlan attn_;
    std::vector<shard::GdnShard> gdn_;
    std::vector<kv::KvPlan> kv_;
    spec::MoeKGovernor gov_;
    spec::SuffixTree suffix_;
    ssm::ReplayRing ring_;
    uint64_t pos_ = 0;
    bool loaded_ = false;
};

} // namespace vegajet::ornith
