// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Minimal dense-transformer plugin skeleton (GQA attention + SwiGLU FFN). Replace the TODOs for a real model.
#include "agent/MODEL_TEMPLATE/template_model.hpp"

namespace vegajet::model_template {

bool TemplateModel::can_load(const GgufFile& f) const {
    // TODO: also check a tensor-layout signature unique to this architecture.
    return f.architecture() == kArchitecture && f.tensor("blk.0.attn_q.weight") != nullptr;
}

bool TemplateModel::parse_hparams(const GgufFile& f, std::string* err) {
    uint64_t u = 0;
    auto need = [&](const char* key, uint32_t& out) {
        if (!f.get_arch_u64(key, u)) { if (err) *err = std::string("missing ") + kArchitecture + "." + key; return false; }
        out = static_cast<uint32_t>(u);
        return true;
    };
    if (!need("block_count", hp_.n_layer) || !need("embedding_length", hp_.n_embd) || !need("attention.head_count", hp_.n_head) ||
        !need("attention.head_count_kv", hp_.n_head_kv) || !need("feed_forward_length", hp_.n_ff))
        return false;
    hp_.head_dim = f.get_arch_u64("attention.key_length", u) ? static_cast<uint32_t>(u) : hp_.n_embd / hp_.n_head;
    if (const GgufTensorInfo* o = f.tensor("output.weight")) hp_.vocab = o->ne[1];
    return true;
}

LoadReport TemplateModel::load(GgufFile& f, const LoadOptions& opt) {
    LoadReport rep;
    rep.architecture = f.architecture();
    if (!can_load(f) || !parse_hparams(f, &rep.error)) { if (rep.error.empty()) rep.error = "not a template GGUF"; return rep; }
    bw_.clear();
    for (const auto& g : opt.gpus) bw_.push_back(g.bw_hbm_gbs);
    rep.weight_bytes_per_gpu.assign(bw_.size(), 0);
    for (const auto& t : f.tensors()) {
        rep.weight_bytes_disk += t.n_bytes();
        if (t.n_dims < 2 || t.ne[0] % 32 != 0) continue;  // vectors stay F32 (TODO: count them per GPU)
        PlannedTensor p{t.name, quant::NativeFormat::VJ_Q8_DP4A, t.ne[0], t.ne[1], shard::split_rows(t.ne[1], bw_)};
        // TODO: pick formats with quant::VramBudgetPlanner::plan_roofline_knapsack instead of all-Q8.
        const uint64_t bytes = quant::VramBudgetPlanner::estimate_bytes(p.format, static_cast<quant::DiskQuantType>(t.type), p.out, p.in);
        for (size_t i = 0; i < bw_.size(); ++i) rep.weight_bytes_per_gpu[i] += bytes * p.rows_per_gpu[i].size() / std::max<uint64_t>(1, p.out);
        plan_.push_back(std::move(p));
    }
    // TODO: KV via kv::plan_kv, attention heads via shard::plan_attention.
    rep.ok = true;
    loaded_ = true;
    return rep;
}

void TemplateModel::emit_layer(ops::OpGraph& g, uint32_t il, uint32_t m, bool prefill) const {
    const int32_t L = static_cast<int32_t>(il);
    const std::string p = "blk." + std::to_string(il) + ".";
    for (uint32_t r = 0; r < bw_.size(); ++r) {
        auto gemv = [&](const char* t, uint32_t n, uint32_t k) {
            g.push_back(ops::OpDesc{prefill ? ops::OpKind::GemmPrefill : ops::OpKind::GemvQ8Dp4a, L, r, m, n, k, 1, 0, p + t});
        };
        g.push_back(ops::OpDesc{ops::OpKind::RmsNorm, L, r, m, hp_.n_embd, 0, 1, 0, p + "attn_norm.weight"});
        gemv("attn_q.weight", hp_.n_head * hp_.head_dim, hp_.n_embd);
        gemv("attn_k.weight", hp_.n_head_kv * hp_.head_dim, hp_.n_embd);
        gemv("attn_v.weight", hp_.n_head_kv * hp_.head_dim, hp_.n_embd);
        g.push_back(ops::OpDesc{ops::OpKind::RopeYarn, L, r, m, hp_.head_dim, hp_.head_dim, 1, 0, ""});
        g.push_back(ops::OpDesc{ops::OpKind::KvAppend, L, r, m, hp_.n_head_kv, hp_.head_dim, 1, 0, ""});
        g.push_back(ops::OpDesc{prefill ? ops::OpKind::FlashPrefill : ops::OpKind::FlashDecodeWave64, L, r, m, hp_.n_head, hp_.head_dim, 1, 0, ""});
        gemv("attn_output.weight", hp_.n_embd, hp_.n_head * hp_.head_dim);
        g.push_back(ops::OpDesc{ops::OpKind::RmsNorm, L, r, m, hp_.n_embd, 0, 1, 0, p + "ffn_norm.weight"});
        gemv("ffn_gate.weight", hp_.n_ff, hp_.n_embd);
        gemv("ffn_up.weight", hp_.n_ff, hp_.n_embd);
        g.push_back(ops::OpDesc{ops::OpKind::SwiGlu, L, r, m, hp_.n_ff, 0, 1, 0, ""});
        gemv("ffn_down.weight", hp_.n_embd, hp_.n_ff);
    }
    if (bw_.size() > 1) g.push_back(ops::OpDesc{ops::OpKind::AllReduce, L, (1u << bw_.size()) - 1, m, hp_.n_embd, 0, 1, 0, ""});
}

StepResult TemplateModel::prefill(const std::vector<int32_t>& tokens) {
    StepResult r;
    r.tokens_in = static_cast<uint32_t>(tokens.size());
    if (!loaded_) return r;
    for (uint32_t il = 0; il < hp_.n_layer; ++il) emit_layer(r.graph, il, r.tokens_in, true);
    return r;
}

StepResult TemplateModel::decode_step(int32_t, uint32_t verify_len) {
    StepResult r;
    r.tokens_in = verify_len;
    if (!loaded_) return r;
    for (uint32_t il = 0; il < hp_.n_layer; ++il) emit_layer(r.graph, il, verify_len, false);
    return r;
}

StepResult TemplateModel::draft_step(const std::vector<int32_t>&, uint32_t) {
    return {};  // TODO: spec::SuffixTree or an external draft model, gated by spec::MoeKGovernor for MoE targets
}

} // namespace vegajet::model_template
