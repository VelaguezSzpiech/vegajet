// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Model-agnostic: this file must not contain model dimensions. tests/test_opgraph_multi_arch.cpp scans it and fails
// on any integer literal above 2.
#include "ops/layer_plan.hpp"

#include "core/launch_planner.hpp"
#include "core/sharder.hpp"

#include <algorithm>
#include <climits>
#include <cmath>

namespace vegajet::ops {

namespace {

std::string tname(int32_t layer, const char* what) {
    return layer < 0 ? std::string(what) : "blk." + std::to_string(layer) + "." + what;
}

bool fail(std::string* err, const std::string& msg) {
    if (err) *err = msg;
    return false;
}

class Emitter {
public:
    Emitter(const ModelDesc& m, const TpDesc& tp, const StepDesc& s, StepPlan& p) : m_(m), tp_(tp), s_(s), p_(p) {}

    void op(OpKind k, int32_t layer, uint32_t gpu, uint32_t n, uint32_t kdim, uint32_t split, uint32_t aux,
            std::string tensor, bool barrier = true) {
        if (n == 0) return;  // this rank owns no rows / heads / experts for this op
        OpDesc d;
        d.kind = k;
        d.layer = layer;
        d.gpu = gpu;
        d.m = s_.m;
        d.n = n;
        d.k = kdim;
        d.split_k = std::max<uint32_t>(1, split);
        d.aux = aux;
        d.tensor = std::move(tensor);
        p_.ops.push_back({d, barrier});
    }

    void proj(int32_t layer, uint32_t gpu, uint32_t n, uint32_t kdim, quant::NativeFormat f, const char* what,
              bool barrier = true) {
        if (n == 0 || kdim == 0) return;
        const uint32_t split = s_.prefill ? 1u : planner::plan_gemv_launch(n, kdim, tp_.n_cu[gpu]).split_k;
        op(projection_kind(f, s_.prefill), layer, gpu, n, kdim, split, static_cast<uint32_t>(f), tname(layer, what),
           barrier);
    }

    void all_reduce(int32_t layer, uint32_t n, uint32_t aux) {
        const uint32_t g = tp_.n_gpus();
        if (g < 2) return;  // N == 1: nothing to reduce
        op(OpKind::AllReduce, layer, (1u << g) - 1u, n, 0, 1, aux, "");
    }

    void norm(int32_t layer, uint32_t gpu, const char* what) {
        op(OpKind::RmsNorm, layer, gpu, m_.d_model, m_.d_model, 1, 0, tname(layer, what));
    }

    void gqa(int32_t L) {
        const auto& a = m_.attn;
        const uint32_t D = m_.d_model;
        const auto plan = shard::plan_attention(a.n_q_heads, a.n_kv_heads, tp_.bw_gbs);
        if (plan.mode == shard::AttnShardMode::SequenceSharded) {
            const auto q_split = shard::apportion(a.n_q_heads, tp_.bw_gbs);
            for (uint32_t g = 0; g < tp_.n_gpus(); ++g) {
                norm(L, g, "attn_norm");
                const auto& sh = plan.gpus[g];
                const uint32_t ctx_g =
                    static_cast<uint32_t>(std::ceil(static_cast<double>(s_.ctx_tokens) * sh.seq_fraction));
                proj(L, g, (a.n_q_heads + 2u * a.n_kv_heads) * a.head_dim, D, m_.fmt_dense, "attn_qkv");
                op(OpKind::RopeYarn, L, g, (a.n_q_heads + a.n_kv_heads) * a.head_dim, a.head_dim, 1, a.n_kv_heads, "");
                op(OpKind::KvAppend, L, g, 2u * a.n_kv_heads * a.head_dim, a.head_dim, 1, a.n_kv_heads, "");
                flash(L, g, a.n_q_heads, a.n_kv_heads, ctx_g);
            }
            all_reduce(L, a.n_q_heads * a.head_dim, kArFlashMerge);
            for (uint32_t g = 0; g < tp_.n_gpus(); ++g)
                proj(L, g, D, static_cast<uint32_t>(q_split[g]) * a.head_dim, m_.fmt_dense, "attn_output");
        } else {
            for (uint32_t g = 0; g < tp_.n_gpus(); ++g) {
                norm(L, g, "attn_norm");
                const auto& sh = plan.gpus[g];
                const uint32_t q = static_cast<uint32_t>(sh.q_heads.size());
                const uint32_t kv = static_cast<uint32_t>(sh.kv_heads.size());
                if (q == 0) continue;
                proj(L, g, (q + 2u * kv) * a.head_dim, D, m_.fmt_dense, "attn_qkv");
                op(OpKind::RopeYarn, L, g, (q + kv) * a.head_dim, a.head_dim, 1, kv, "");
                op(OpKind::KvAppend, L, g, 2u * kv * a.head_dim, a.head_dim, 1, kv, "");
                flash(L, g, q, kv, s_.ctx_tokens);
                proj(L, g, D, q * a.head_dim, m_.fmt_dense, "attn_output");
            }
        }
        all_reduce(L, D, kArSumHidden);
    }

    void flash(int32_t L, uint32_t g, uint32_t q, uint32_t kv, uint32_t ctx) {
        const uint32_t n = q * m_.attn.head_dim;
        if (s_.prefill) {
            op(OpKind::FlashPrefill, L, g, n, ctx, 1, kv, "");
        } else {
            const auto kvp = planner::plan_split_kv(q, ctx, tp_.n_cu[g]);
            op(OpKind::FlashDecodeWave64, L, g, n, ctx, kvp.chunks, kv, "");
        }
    }

    void gdn(int32_t L) {
        const auto& d = m_.gdn;
        const uint32_t D = m_.d_model;
        const auto sh = shard::plan_gdn(d.n_k_groups, d.n_v_heads, tp_.bw_gbs);
        for (uint32_t g = 0; g < tp_.n_gpus(); ++g) {
            norm(L, g, "attn_norm");
            const uint32_t kg = static_cast<uint32_t>(sh[g].k_groups.size());
            const uint32_t vh = static_cast<uint32_t>(sh[g].v_heads.size());
            if (vh == 0) continue;
            // q, k per key group; v, z per value head; a, b scalars per value head.
            proj(L, g, 2u * kg * d.head_k_dim + 2u * vh * d.head_v_dim + 2u * vh, D, m_.fmt_dense, "ssm_in");
            op(OpKind::GdnConv1d, L, g, 2u * kg * d.head_k_dim + vh * d.head_v_dim, d.conv_kernel, 1, d.conv_kernel,
               tname(L, "ssm_conv1d"));
            op(s_.prefill ? OpKind::GdnChunkWy : OpKind::GdnRecurrentStep, L, g, vh * d.head_v_dim, d.head_k_dim, 1, vh,
               "");
            proj(L, g, D, vh * d.head_v_dim, m_.fmt_dense, "ssm_out");
        }
        all_reduce(L, D, kArSumHidden);
    }

    void dense_ffn(int32_t L) {
        const uint32_t D = m_.d_model;
        const auto r = shard::split_rows(m_.dense_ff, tp_.bw_gbs);
        for (uint32_t g = 0; g < tp_.n_gpus(); ++g) {
            norm(L, g, "ffn_norm");
            const uint32_t ff = static_cast<uint32_t>(r[g].size());
            proj(L, g, 2u * ff, D, m_.fmt_dense, "ffn_gate_up");
            op(OpKind::SwiGlu, L, g, ff, 2u * ff, 1, 0, "");
            proj(L, g, D, ff, m_.fmt_dense, "ffn_down");
        }
        all_reduce(L, D, kArSumHidden);
    }

    void moe_ffn(int32_t L) {
        const auto& e = m_.moe;
        const uint32_t D = m_.d_model;
        const auto shared = shard::split_rows(e.shared_ff, tp_.bw_gbs);
        const auto ff_split = shard::split_rows(e.expert_ff, tp_.bw_gbs);
        const auto ex_split = shard::split_experts(e.n_experts, tp_.bw_gbs);
        for (uint32_t g = 0; g < tp_.n_gpus(); ++g) {
            norm(L, g, "ffn_norm");
            op(OpKind::MoeRouterTopK, L, g, e.n_experts, D, 1, e.top_k, tname(L, "ffn_gate_inp"));
            const uint32_t sff = e.shared_ff ? static_cast<uint32_t>(shared[g].size()) : 0u;
            if (sff) {
                // Independent of the router: may overlap it (the only barrier-free edge in a layer).
                proj(L, g, 2u * sff, D, m_.fmt_dense, "ffn_gate_up_shexp", /*barrier=*/false);
                op(OpKind::SwiGlu, L, g, sff, 2u * sff, 1, 0, "");
                proj(L, g, D, sff, m_.fmt_dense, "ffn_down_shexp");
            }
            if (e.shard == ExpertShard::SplitFf) {
                if (ff_split[g].size() == 0) continue;
                op(OpKind::MoeFusedExperts, L, g, D, static_cast<uint32_t>(ff_split[g].size()), 1, e.top_k,
                   tname(L, "ffn_experts"));
            } else if (ex_split[g].size()) {
                op(OpKind::MoeFusedExperts, L, g, D, e.expert_ff, 1, e.top_k,
                   tname(L, "ffn_experts") + "[" + std::to_string(ex_split[g].begin) + ":" +
                       std::to_string(ex_split[g].end) + ")");
            }
        }
        all_reduce(L, D, kArSumHidden);
    }

    void model_head() {
        const auto r = shard::split_rows(m_.vocab, tp_.bw_gbs);
        for (uint32_t g = 0; g < tp_.n_gpus(); ++g) {
            norm(-1, g, "output_norm");
            // Each rank scores its vocab slice; the host merges the per-rank argmax / top-k candidates.
            op(OpKind::LmHead, -1, g, static_cast<uint32_t>(r[g].size()), m_.d_model, 1,
               static_cast<uint32_t>(m_.fmt_head), "output");
        }
    }

    void embed() {
        for (uint32_t g = 0; g < tp_.n_gpus(); ++g)
            op(OpKind::Embed, -1, g, m_.d_model, m_.vocab, 1, 0, "token_embd");
    }

private:
    const ModelDesc& m_;
    const TpDesc& tp_;
    const StepDesc& s_;
    StepPlan& p_;
};

} // namespace

OpKind projection_kind(quant::NativeFormat f, bool prefill) {
    if (prefill) return OpKind::GemmPrefill;
    switch (f) {
        case quant::NativeFormat::VJ_Q8_DP4A: return OpKind::GemvQ8Dp4a;
        case quant::NativeFormat::VJ_Q4_DOT8: return OpKind::GemvQ4Dot8;
        case quant::NativeFormat::VJ_Q4_A8:   return OpKind::GemvQ4Dot8;
        case quant::NativeFormat::VJ_IQ_PACKED: return OpKind::GemvIqPacked;
    }
    return OpKind::GemvQ8Dp4a;
}

bool is_projection(OpKind k) {
    return k == OpKind::GemvQ8Dp4a || k == OpKind::GemvQ4Dot8 || k == OpKind::GemvIqPacked || k == OpKind::GemmPrefill;
}

OpGraph StepPlan::graph() const {
    OpGraph g;
    g.reserve(ops.size());
    for (const auto& p : ops) g.push_back(p.op);
    return g;
}

uint32_t StepPlan::count(OpKind k) const {
    return static_cast<uint32_t>(std::count_if(ops.begin(), ops.end(), [k](const PlannedOp& p) { return p.op.kind == k; }));
}

std::vector<PlannedOp> StepPlan::ops_for_gpu(uint32_t g) const {
    std::vector<PlannedOp> out;
    for (const auto& p : ops) {
        const bool mine = p.op.kind == OpKind::AllReduce ? ((p.op.gpu >> g) & 1u) != 0 : p.op.gpu == g;
        if (mine) out.push_back(p);
    }
    return out;
}

bool validate_model_desc(const ModelDesc& m, std::string* err) {
    if (!m.d_model || !m.vocab) return fail(err, "d_model and vocab must be non-zero");
    if (m.mixer.empty()) return fail(err, "model has no layers");
    if (m.mixer.size() != m.ffn.size()) return fail(err, "mixer and ffn layer lists differ in length");
    const bool any_gqa = std::count(m.mixer.begin(), m.mixer.end(), MixerKind::Gqa) > 0;
    const bool any_gdn = std::count(m.mixer.begin(), m.mixer.end(), MixerKind::Gdn) > 0;
    const bool any_dense = std::count(m.ffn.begin(), m.ffn.end(), FfnKind::Dense) > 0;
    const bool any_moe = std::count(m.ffn.begin(), m.ffn.end(), FfnKind::Moe) > 0;
    if (any_gqa) {
        const auto& a = m.attn;
        if (!a.n_q_heads || !a.n_kv_heads || !a.head_dim) return fail(err, "attention dims must be non-zero");
        if (a.n_q_heads % a.n_kv_heads) return fail(err, "n_q_heads must be a multiple of n_kv_heads");
    }
    if (any_gdn) {
        const auto& d = m.gdn;
        if (!d.n_k_groups || !d.n_v_heads || !d.head_k_dim || !d.head_v_dim || !d.conv_kernel)
            return fail(err, "GDN dims must be non-zero");
        if (d.n_v_heads % d.n_k_groups) return fail(err, "n_v_heads must be a multiple of n_k_groups");
    }
    if (any_dense && !m.dense_ff) return fail(err, "dense layers need dense_ff");
    if (any_moe) {
        const auto& e = m.moe;
        if (!e.n_experts || !e.expert_ff || !e.top_k || e.top_k > e.n_experts)
            return fail(err, "MoE needs n_experts, expert_ff and 0 < top_k <= n_experts");
    }
    return true;
}

bool validate_tp_desc(const TpDesc& tp, std::string* err) {
    if (tp.bw_gbs.empty()) return fail(err, "TpDesc needs at least one GPU");
    if (tp.n_cu.size() != tp.bw_gbs.size()) return fail(err, "bw_gbs and n_cu differ in length");
    if (tp.bw_gbs.size() >= sizeof(uint32_t) * CHAR_BIT) return fail(err, "too many GPUs for the AllReduce rank mask");
    for (size_t i = 0; i < tp.bw_gbs.size(); ++i)
        if (!(tp.bw_gbs[i] > 0.0) || !tp.n_cu[i]) return fail(err, "every GPU needs bw_gbs > 0 and n_cu > 0");
    return true;
}

bool build_step_plan(const ModelDesc& m, const TpDesc& tp, const StepDesc& s, StepPlan& out, std::string* err) {
    out = StepPlan{};
    if (!validate_model_desc(m, err) || !validate_tp_desc(tp, err)) return false;
    if (!s.m || !s.ctx_tokens) return fail(err, "StepDesc needs m > 0 and ctx_tokens > 0");
    out.n_gpus = tp.n_gpus();
    Emitter e(m, tp, s, out);
    e.embed();
    for (uint32_t L = 0; L < m.n_layers(); ++L) {
        const auto li = static_cast<int32_t>(L);
        if (m.mixer[L] == MixerKind::Gqa) e.gqa(li); else e.gdn(li);
        if (m.ffn[L] == FfnKind::Dense) e.dense_ffn(li); else e.moe_ffn(li);
    }
    e.model_head();
    return true;
}

} // namespace vegajet::ops
