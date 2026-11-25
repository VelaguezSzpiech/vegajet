// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "declarative_model.hpp"

#include <algorithm>

namespace vegajet::declarative_template {

namespace {

bool need(const GgufFile& f, const char* key, uint32_t& out, std::string* err) {
    uint64_t v = 0;
    if (!f.get_arch_u64(key, v) || v == 0 || v > UINT32_MAX) {
        if (err) *err = f.architecture() + "." + key + " missing or invalid";
        return false;
    }
    out = static_cast<uint32_t>(v);
    return true;
}

uint32_t opt_u32(const GgufFile& f, const char* key, uint32_t fallback) {
    uint64_t v = 0;
    return f.get_arch_u64(key, v) && v <= UINT32_MAX ? static_cast<uint32_t>(v) : fallback;
}

// Pass-1 baseline of the roofline knapsack: 8-bit / float -> Q8, 4-bit K/legacy -> Q4, everything else stays packed.
// TODO: replace with the per-tensor decisions of quant::VramBudgetPlanner::plan_roofline_knapsack once wired.
quant::NativeFormat native_for(const GgufFile& f, const std::string& tensor, quant::NativeFormat fallback) {
    const GgufTensorInfo* t = f.tensor(tensor);
    if (!t) return fallback;
    switch (static_cast<GgmlType>(t->type)) {
        case GgmlType::F32: case GgmlType::F16: case GgmlType::BF16: case GgmlType::Q8_0: case GgmlType::Q6_K:
            return quant::NativeFormat::VJ_Q8_DP4A;
        case GgmlType::Q4_0: case GgmlType::Q4_1: case GgmlType::Q4_K: case GgmlType::Q5_K:
            return quant::NativeFormat::VJ_Q4_DOT8;
        default:
            return quant::NativeFormat::VJ_IQ_PACKED;
    }
}

} // namespace

bool DeclarativeModel::describe(const GgufFile& f, ops::ModelDesc& m, std::string* err) {
    m = ops::ModelDesc{};
    m.arch = f.architecture();
    uint32_t blocks = 0;
    if (!need(f, "embedding_length", m.d_model, err) || !need(f, "block_count", blocks, err)) return false;
    const uint32_t layers = blocks - std::min(blocks - 1u, opt_u32(f, "nextn_predict_layers", 0));

    // Attention (every architecture has at least one attention layer in this template).
    if (!need(f, "attention.head_count", m.attn.n_q_heads, err) ||
        !need(f, "attention.head_count_kv", m.attn.n_kv_heads, err))
        return false;
    m.attn.head_dim = opt_u32(f, "attention.key_length", m.d_model / m.attn.n_q_heads);

    // Recurrent (gated delta net) layers: present when the ssm.* keys are.
    const bool hybrid = opt_u32(f, "ssm.group_count", 0) != 0;
    if (hybrid) {
        uint32_t inner = 0;
        if (!need(f, "ssm.group_count", m.gdn.n_k_groups, err) || !need(f, "ssm.time_step_rank", m.gdn.n_v_heads, err) ||
            !need(f, "ssm.state_size", m.gdn.head_k_dim, err) || !need(f, "ssm.inner_size", inner, err) ||
            !need(f, "ssm.conv_kernel", m.gdn.conv_kernel, err))
            return false;
        m.gdn.head_v_dim = inner / m.gdn.n_v_heads;
    }
    const uint32_t interval = opt_u32(f, "full_attention_interval", 0);
    const GgufValue* rec = f.find(m.arch + ".attention.recurrent_layers");
    for (uint32_t L = 0; L < layers; ++L) {
        bool recurrent = false;
        if (hybrid && rec && rec->arr_num.size() >= layers) recurrent = rec->arr_num[L] != 0.0;
        else if (hybrid && interval) recurrent = (L + 1) % interval != 0;
        m.mixer.push_back(recurrent ? ops::MixerKind::Gdn : ops::MixerKind::Gqa);
    }

    // FFN: MoE when expert_count is set, dense otherwise.
    const uint32_t n_expert = opt_u32(f, "expert_count", 0);
    if (n_expert) {
        m.moe.n_experts = n_expert;
        if (!need(f, "expert_used_count", m.moe.top_k, err)) return false;
        m.moe.expert_ff = opt_u32(f, "expert_feed_forward_length", opt_u32(f, "feed_forward_length", 0));
        m.moe.shared_ff = opt_u32(f, "expert_shared_feed_forward_length", 0);
        m.moe.shard = ops::ExpertShard::SplitFf;  // TODO: WholeExperts when n_expert >= 4 x n_gpus and top_k is small
        m.ffn.assign(layers, ops::FfnKind::Moe);
    } else {
        if (!need(f, "feed_forward_length", m.dense_ff, err)) return false;
        m.ffn.assign(layers, ops::FfnKind::Dense);
    }

    if (const GgufTensorInfo* o = f.tensor("output.weight")) m.vocab = static_cast<uint32_t>(o->ne[1]);
    else if (const GgufTensorInfo* e = f.tensor("token_embd.weight")) m.vocab = static_cast<uint32_t>(e->ne[1]);

    // TODO: map your architecture's tensor names; these are the common llama.cpp names.
    const std::string probe = m.mixer.front() == ops::MixerKind::Gdn ? "blk.0.ssm_out.weight" : "blk.0.attn_output.weight";
    m.fmt_dense = native_for(f, probe, quant::NativeFormat::VJ_Q4_DOT8);
    m.fmt_expert = native_for(f, "blk.0.ffn_down_exps.weight", quant::NativeFormat::VJ_IQ_PACKED);
    m.fmt_head = native_for(f, "output.weight", quant::NativeFormat::VJ_Q8_DP4A);
    return ops::validate_model_desc(m, err);
}

bool DeclarativeModel::can_load(const GgufFile& f) const {
    return f.architecture() == kArchitecture && f.tensor("token_embd.weight") != nullptr;
}

LoadReport DeclarativeModel::load(GgufFile& f, const LoadOptions& opt) {
    LoadReport rep;
    rep.architecture = f.architecture();
    if (!describe(f, desc_, &rep.error)) return rep;
    tp_ = ops::TpDesc{};
    for (const auto& g : opt.gpus) {
        tp_.bw_gbs.push_back(g.bw_hbm_gbs);
        tp_.n_cu.push_back(g.n_cu);
    }
    if (!ops::validate_tp_desc(tp_, &rep.error)) return rep;
    ctx_capacity_ = opt.context_length;
    ctx_tokens_ = 0;
    for (const auto& t : f.tensors()) rep.weight_bytes_disk += t.n_bytes();
    // TODO (ONBOARD_MODEL step 3): per-tensor repack + shard plan, KV (kv::plan_kv) and recurrent state
    // (ssm::describe) bytes per GPU, then the VRAM check. Until then the report carries the disk total only.
    ops::StepPlan probe;
    if (!ops::build_step_plan(desc_, tp_, {1, false, std::max<uint32_t>(1, ctx_capacity_)}, probe, &rep.error)) return rep;
    rep.notes.push_back("decode plan: " + std::to_string(probe.ops.size()) + " ops/step over " +
                        std::to_string(tp_.n_gpus()) + " GPU(s), " + std::to_string(probe.count(ops::OpKind::AllReduce)) +
                        " AllReduce");
    loaded_ = true;
    rep.ok = true;
    return rep;
}

StepResult DeclarativeModel::plan(uint32_t m, bool prefill) const {
    StepResult r;
    if (!loaded_ || m == 0) return r;
    ops::StepPlan p;
    std::string err;
    if (ops::build_step_plan(desc_, tp_, {m, prefill, std::max<uint32_t>(1, ctx_tokens_ + m)}, p, &err)) r.graph = p.graph();
    r.tokens_in = m;
    return r;
}

StepResult DeclarativeModel::prefill(const std::vector<int32_t>& tokens) {
    StepResult r = plan(static_cast<uint32_t>(tokens.size()), true);
    ctx_tokens_ += static_cast<uint32_t>(tokens.size());
    return r;
}

StepResult DeclarativeModel::decode_step(int32_t /*token*/, uint32_t verify_len) {
    const uint32_t m = std::max<uint32_t>(1, verify_len);
    StepResult r = plan(m, false);
    ctx_tokens_ += 1;  // TODO: advance by the accepted length once the verifier reports it
    return r;
}

StepResult DeclarativeModel::draft_step(const std::vector<int32_t>& /*context*/, uint32_t /*max_draft*/) {
    return {};  // TODO: MTP head / external draft model / suffix tree; an empty graph means "no draft".
}

} // namespace vegajet::declarative_template
