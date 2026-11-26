// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Op-graph emission for Ornith prefill / decode / draft steps.
#include "core/grid_planner.hpp"
#include "models/ornith/ornith_model.hpp"
#include "models/registry.hpp"

#include <algorithm>
#include <cmath>

namespace vegajet::ornith {

using ops::OpDesc;
using ops::OpKind;
using quant::NativeFormat;

namespace {
std::string blk(uint32_t il, const char* suffix) { return "blk." + std::to_string(il) + "." + suffix; }
} // namespace

void OrnithModel::emit_gemv(ops::OpGraph& g, const std::string& tensor, int32_t layer, uint32_t m, bool prefill) const {
    const TensorPlan* p = find_plan(tensor);
    if (!p) return;
    for (uint32_t r = 0; r < bw_.size(); ++r) {
        if (p->bytes_per_gpu.empty() || p->bytes_per_gpu[r] == 0) continue;
        OpDesc d;
        d.layer = layer;
        d.gpu = r;
        d.m = m;
        d.tensor = tensor;
        const bool col = p->shard == ShardKind::ColSplit ||
                         (p->shard == ShardKind::GdnGroups && p->name.find("ssm_out") != std::string::npos) ||
                         (p->shard == ShardKind::QHeads && p->name.find("attn_output") != std::string::npos);
        const double frac = double(p->bytes_per_gpu[r]) / std::max<uint64_t>(1, p->bytes_total);
        d.n = col ? static_cast<uint32_t>(p->out) : static_cast<uint32_t>(p->local_out[r]);
        d.k = col ? static_cast<uint32_t>(std::max<double>(32.0, p->in * frac)) : static_cast<uint32_t>(p->in);
        if (prefill) {
            d.kind = OpKind::GemmPrefill;
            d.aux = static_cast<uint32_t>(p->format);
        } else {
            d.kind = p->format == NativeFormat::VJ_Q8_DP4A ? OpKind::GemvQ8Dp4a
                   : p->format == NativeFormat::VJ_Q4_DOT8 ? OpKind::GemvQ4Dot8 : OpKind::GemvIqPacked;
            d.split_k = planner::plan_gemv_splitk(d.n, d.k, opt_.gpus[r].n_cu).split_k;
        }
        g.push_back(d);
    }
}

void OrnithModel::emit_allreduce(ops::OpGraph& g, int32_t layer, uint32_t m) const {
    if (bw_.size() < 2) return;
    OpDesc d;
    d.kind = OpKind::AllReduce;
    d.layer = layer;
    d.gpu = (1u << bw_.size()) - 1;  // participating ranks mask
    d.m = m;
    d.n = hp_.n_embd;
    g.push_back(d);
}

void OrnithModel::emit_layer(ops::OpGraph& g, uint32_t il, uint32_t m, bool prefill) const {
    const int32_t L = static_cast<int32_t>(il);
    const uint32_t n = static_cast<uint32_t>(bw_.size());
    auto all = [&](OpKind k, uint32_t nn, uint32_t kk, uint32_t aux, const std::string& t) {
        for (uint32_t r = 0; r < n; ++r) g.push_back(OpDesc{k, L, r, m, nn, kk, 1, aux, t});
    };
    all(OpKind::RmsNorm, hp_.n_embd, 0, 0, blk(il, "attn_norm.weight"));
    if (hp_.is_recurrent(il)) {
        for (const char* t : {"attn_qkv.weight", "attn_gate.weight", "ssm_alpha.weight", "ssm_beta.weight"})
            emit_gemv(g, blk(il, t), L, m, prefill);
        const uint32_t head_v = hp_.ssm_inner / std::max(1u, hp_.ssm_v_heads);
        for (uint32_t r = 0; r < n; ++r) {
            const uint32_t vh = static_cast<uint32_t>(gdn_[r].v_heads.size());
            const uint32_t kg = static_cast<uint32_t>(gdn_[r].k_groups.size());
            g.push_back(OpDesc{OpKind::GdnConv1d, L, r, m, 2 * kg * hp_.ssm_state + vh * head_v, hp_.ssm_conv, 1, 0, blk(il, "ssm_conv1d.weight")});
            if (prefill) g.push_back(OpDesc{OpKind::GdnChunkWy, L, r, m, vh, hp_.ssm_state, 1, 64, blk(il, "ssm_a")});
            else g.push_back(OpDesc{OpKind::GdnRecurrentStep, L, r, m, vh, hp_.ssm_state, 1, m, blk(il, "ssm_a")});
        }
        emit_gemv(g, blk(il, "ssm_out.weight"), L, m, prefill);
    } else {
        for (const char* t : {"attn_q.weight", "attn_k.weight", "attn_v.weight"}) emit_gemv(g, blk(il, t), L, m, prefill);
        for (uint32_t r = 0; r < n; ++r) {
            const uint32_t qh = static_cast<uint32_t>(attn_.gpus[r].q_heads.size());
            const uint32_t kvh = static_cast<uint32_t>(attn_.gpus[r].kv_heads.size());
            g.push_back(OpDesc{OpKind::RopeYarn, L, r, m, hp_.rope_dims, hp_.head_dim_k, 1, qh + kvh, ""});
            g.push_back(OpDesc{OpKind::KvAppend, L, r, m, kvh, hp_.head_dim_k, 1, kvh, ""});
            g.push_back(OpDesc{prefill ? OpKind::FlashPrefill : OpKind::FlashDecodeWave64, L, r, m, qh, hp_.head_dim_k, 1, kvh, ""});
        }
        emit_gemv(g, blk(il, "attn_output.weight"), L, m, prefill);
    }
    emit_allreduce(g, L, m);
    all(OpKind::RmsNorm, hp_.n_embd, 0, 0, blk(il, "post_attention_norm.weight"));
    all(OpKind::MoeRouterTopK, hp_.n_expert, hp_.n_embd, hp_.n_expert_used, blk(il, "ffn_gate_inp.weight"));
    if (const TensorPlan* ex = find_plan(blk(il, "ffn_gate_exps.weight")))
        for (uint32_t r = 0; r < n; ++r)
            if (ex->local_out[r]) g.push_back(OpDesc{OpKind::MoeFusedExperts, L, r, m, hp_.n_ff_exp, hp_.n_embd, 1,
                                                     static_cast<uint32_t>(ex->local_out[r]), ex->name});
    emit_gemv(g, blk(il, "ffn_gate_shexp.weight"), L, m, prefill);
    emit_gemv(g, blk(il, "ffn_up_shexp.weight"), L, m, prefill);
    all(OpKind::SwiGlu, hp_.n_ff_shexp, 0, 0, "");
    emit_gemv(g, blk(il, "ffn_down_shexp.weight"), L, m, prefill);
    emit_allreduce(g, L, m);  // routed + shared expert partial sums reduced once
}

StepResult OrnithModel::prefill(const std::vector<int32_t>& tokens) {
    StepResult res;
    res.tokens_in = static_cast<uint32_t>(tokens.size());
    if (!loaded_ || tokens.empty()) return res;
    auto& g = res.graph;
    for (size_t c = 0; c < tokens.size(); c += kPrefillChunk) {
        const uint32_t m = static_cast<uint32_t>(std::min<size_t>(kPrefillChunk, tokens.size() - c));
        g.push_back(OpDesc{OpKind::Embed, -1, 0, m, hp_.n_embd, 0, 1, 0, "token_embd.weight"});
        emit_allreduce(g, -1, m);
        for (uint32_t il = 0; il < hp_.n_trunk; ++il) emit_layer(g, il, m, true);
    }
    for (uint32_t r = 0; r < bw_.size(); ++r) g.push_back(OpDesc{OpKind::RmsNorm, -1, r, 1, hp_.n_embd, 0, 1, 0, "output_norm.weight"});
    emit_gemv(g, "output.weight", -1, 1, false);
    for (auto& d : g) if (d.tensor == "output.weight") d.kind = OpKind::LmHead;
    emit_allreduce(g, -1, 1);
    suffix_.extend(tokens);
    for (size_t i = 0; i < tokens.size(); ++i) ring_.push();
    ring_.take_checkpoint();
    for (uint32_t il = 0; il < hp_.n_trunk; ++il)
        if (hp_.is_recurrent(il))
            for (uint32_t r = 0; r < bw_.size(); ++r) g.push_back(OpDesc{OpKind::SsmCheckpoint, static_cast<int32_t>(il), r, 1, 0, 0, 1, 0, ""});
    pos_ += tokens.size();
    return res;
}

StepResult OrnithModel::decode_step(int32_t token, uint32_t verify_len) {
    StepResult res;
    verify_len = std::max(1u, verify_len);
    res.tokens_in = verify_len;
    if (!loaded_) return res;
    auto& g = res.graph;
    g.push_back(OpDesc{OpKind::Embed, -1, 0, verify_len, hp_.n_embd, 0, 1, 0, "token_embd.weight"});
    emit_allreduce(g, -1, verify_len);
    for (uint32_t il = 0; il < hp_.n_trunk; ++il) emit_layer(g, il, verify_len, false);
    for (uint32_t r = 0; r < bw_.size(); ++r) g.push_back(OpDesc{OpKind::RmsNorm, -1, r, verify_len, hp_.n_embd, 0, 1, 0, "output_norm.weight"});
    emit_gemv(g, "output.weight", -1, verify_len, false);
    for (auto& d : g) if (d.tensor == "output.weight") d.kind = OpKind::LmHead;
    emit_allreduce(g, -1, verify_len);
    if (verify_len > 1) {
        for (uint32_t r = 0; r < bw_.size(); ++r) g.push_back(OpDesc{OpKind::DraftVerify, -1, r, verify_len, 0, 0, 1, verify_len, ""});
        return res;  // commit happens after acceptance is known (spec governor + replay ring)
    }
    suffix_.push(token);
    ++pos_;
    if (ring_.push()) {
        ring_.take_checkpoint();
        for (uint32_t il = 0; il < hp_.n_trunk; ++il)
            if (hp_.is_recurrent(il))
                for (uint32_t r = 0; r < bw_.size(); ++r) g.push_back(OpDesc{OpKind::SsmCheckpoint, static_cast<int32_t>(il), r, 1, 0, 0, 1, 0, ""});
    }
    return res;
}

StepResult OrnithModel::draft_step(const std::vector<int32_t>& context, uint32_t max_draft) {
    StepResult res;
    if (!loaded_) return res;
    // The controller only limits the prefix.  A zero proposal is a cost decision: the caller must run one
    // ordinary target decode step and emit that token; nothing here commits or rewrites a token.
    const uint32_t available = std::min(max_draft, kEvictMaxDepth);
    // One-shot measured union of the last completed target pass.  draft_step() consumes it, so a deferred or
    // missing readback becomes 0 (unknown) instead of reusing a stale count.
    uint32_t k = evict_.choose(available, route_feedback_.take());
    if (k == 0) return res;
    if (draft_src_ == DraftSource::ExternalDFlash) {
        k = std::min(k, dhp_.block_size - 1);
        const uint32_t m = dhp_.block_size;
        auto& g = res.graph;
        g.push_back(OpDesc{OpKind::Embed, -1, 0, m, hp_.n_embd, 0, 1, static_cast<uint32_t>(dhp_.mask_token), "token_embd.weight"});
        emit_gemv(g, "draft/fc.weight", -1, m, false);
        for (uint32_t il = 0; il < dhp_.n_layer; ++il) {
            const int32_t L = static_cast<int32_t>(il);
            const std::string p = "draft/blk." + std::to_string(il) + ".";
            g.push_back(OpDesc{OpKind::RmsNorm, L, 0, m, hp_.n_embd, 0, 1, 0, p + "attn_norm.weight"});
            for (const char* t : {"attn_q.weight", "attn_k.weight", "attn_v.weight"}) emit_gemv(g, p + t, L, m, false);
            g.push_back(OpDesc{OpKind::RopeYarn, L, 0, m, dhp_.head_dim, dhp_.head_dim, 1, dhp_.n_head + dhp_.n_head_kv, ""});
            g.push_back(OpDesc{OpKind::FlashPrefill, L, 0, m, dhp_.n_head, dhp_.head_dim, 1, dhp_.n_head_kv, ""});
            emit_gemv(g, p + "attn_output.weight", L, m, false);
            g.push_back(OpDesc{OpKind::RmsNorm, L, 0, m, hp_.n_embd, 0, 1, 0, p + "ffn_norm.weight"});
            emit_gemv(g, p + "ffn_gate.weight", L, m, false);
            emit_gemv(g, p + "ffn_up.weight", L, m, false);
            g.push_back(OpDesc{OpKind::SwiGlu, L, 0, m, dhp_.n_ff, 0, 1, 0, ""});
            emit_gemv(g, p + "ffn_down.weight", L, m, false);
        }
        g.push_back(OpDesc{OpKind::RmsNorm, -1, 0, m, hp_.n_embd, 0, 1, 0, "draft/output_norm.weight"});
        emit_allreduce(g, -1, k);  // broadcast the drafted hidden block to the vocab shards
        emit_gemv(g, "output.weight", -1, k, false);  // draft shares the target LM head
        for (auto& d : g) if (d.tensor == "output.weight") d.kind = OpKind::LmHead;
        res.draft_tokens = k;
        return res;
    }
    const auto nodes = suffix_.draft(context, k);
    res.draft_tokens = static_cast<uint32_t>(nodes.size());  // CPU-only proposal: no device ops
    return res;
}

bool OrnithModel::note_route_ids(const int32_t* ids, uint32_t layers, uint32_t tokens, uint32_t top_k,
                                 uint32_t layer_stride_ids) {
    RouteUnionConfig cfg;
    cfg.max_layers = std::min<uint32_t>(kRouteUnionMaxLayers, hp_.n_trunk ? hp_.n_trunk : 48u);
    cfg.max_experts = std::min<uint32_t>(kRouteUnionMaxExperts, hp_.n_expert ? hp_.n_expert : 256u);
    cfg.max_tokens = kRouteUnionMaxTokens;
    cfg.top_k = top_k;
    return record_completed_routes(&route_feedback_, ids, layers, tokens, top_k, layer_stride_ids, cfg) ==
           RouteUnionStatus::kOk;
}

bool OrnithModel::note_route_union(uint64_t unique_experts, uint32_t tokens) {
    // The count must come from an exact RouteUnion dedup over completed route IDs; 0 is unknown.
    return route_feedback_.observe_count(unique_experts, tokens);
}

ssm::ReplayRing::Rollback OrnithModel::accept_draft(const std::vector<int32_t>& verified, uint32_t proposed) {
    const uint32_t accepted = verified.empty() ? 0 : static_cast<uint32_t>(verified.size() - 1);
    evict_.observe(proposed, accepted);
    gov_.observe(proposed, accepted);
    for (uint32_t i = 0; i < proposed + 1; ++i) ring_.push();  // the device advanced over every verified slot
    const auto rb = ring_.rollback(proposed - std::min(accepted, proposed));
    for (int32_t t : verified) suffix_.push(t);
    pos_ += verified.size();
    return rb;
}

} // namespace vegajet::ornith

namespace vegajet {
// Ornith registration (called from register_builtin_models).
void register_ornith(ModelRegistry& r) {
    r.add({"ornith", {"qwen35moe", "qwen3_5_moe", "ornith"}, [] { return std::make_unique<ornith::OrnithModel>(); }});
}
} // namespace vegajet
