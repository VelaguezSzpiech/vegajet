// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "models/ornith/ornith_tp_exec.hpp"
#include "core/q4_dot8_repack.hpp"
#include "core/placement.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <map>
#include <stdexcept>

namespace vegajet::ornith {
namespace {

bool env_is(const char* v, const char* value) {
    return v && std::strcmp(v, value) == 0;
}
bool env_enabled(const char* v) {
    return v && v[0] && std::strcmp(v, "0") != 0;
}
} // namespace

ExecutorMode ornith_executor_mode(uint32_t tp) {
    return tp == 2 ? ExecutorMode::LegacyFast : ExecutorMode::Generic;
}

tp::FeatureFlags ornith_feature_flags(const OrnithGpuOptions& options) {
    tp::FeatureFlags f;
    // The fused fast path is the TP=2 default, but it is only *requested* for admission when the
    // owner selects it explicitly: TP != 2 runs the generic per-op path without it.
    f.super_kernels = env_is(std::getenv("VEGAJET_ENABLE_SUPER"), "1");
    f.asym_experts = env_is(std::getenv("VEGAJET_V12_ASYM_EXPERTS"), "1");
    f.mtp_draft = !options.draft_gguf.empty();
    f.speculative = options.use_suffix_spec || !options.draft_gguf.empty();
    f.int8_wire = options.ar_wire_int8;
    f.fused_verify = env_enabled(std::getenv("VEGAJET_V12_VERIFY_FUSED_NQ"));
    f.prefill_dot8 = options.prefill_dot8;
    f.ring_state = options.spec_gdn;
    return f;
}

std::string ornith_admit(uint32_t tp, const tp::FeatureFlags& features) {
    if (tp == 0 || tp > 256) return "TP rank count must be in 1..256";
    // TP=2 is the fused fast path: nothing to admit beyond the runner's own checks.
    if (ornith_executor_mode(tp) == ExecutorMode::LegacyFast) return {};
    return tp::admit_features(tp, features);
}

std::string ornith_raw_admit(const OrnithGpuOptions& o) {
    if (!o.placement_json.empty()) {
        try {
            const auto config = placement::from_json(o.placement_json);
            const auto why = placement::executor_error(config);
            if (!why.empty()) return why;
            if (config.tp_groups[0].devices.size() != o.tp) return "placement TP-group size conflicts with --tp";

        } catch (const std::exception& e) { return e.what(); }
    }
    const auto base = ornith_admit(o.tp, ornith_feature_flags(o));
    if (!base.empty()) return base;
    if (o.tp == 2 && o.placement_json.empty()) return (!o.tp_shard_json.empty() || !o.tp_collective_json.empty())
        ? "TP=2 legacy fast path: TP JSON overrides unsupported" : "";
    auto no = [](const char* name) { return std::string("generic Ornith: unsupported feature ") + name; };
    if (!o.max_ctx || o.max_ctx > 8192) return no("context outside 1..8192");
    if (o.fr_spec_rows || o.zero_tax_catchup) return no("MTP/FR-Spec/catch-up");
    if (o.spec_flash || o.spec_force_reject || o.gu_grp_verify || o.down_grp_verify) return no("speculative verify");
    if (o.gu_dot8_layers || o.dot8_ab) return no("Gate/Up promotion or DOT8 A/B");
    if (o.pp_stages || o.ar_timeline) return no("pp-stage/AR timeline diagnostics (TP=2 only; use --profile-stages)");
    if (!o.dump_hidden_path.empty() || !o.dump_logits_path.empty() || !o.gdn_state_dump.empty() || !o.dump_golden_layers_dir.empty())
        return no("hidden/logits/state/golden dump");
    if (o.interactive || !o.interactive_fifo.empty() || !o.sweep_spec.empty()) return no("interactive/sweep");
    if (!o.tp_bandwidths.empty() && o.tp_bandwidths.size() != o.tp) return no("bandwidth count must equal TP rank count");
    // Only explicit opt-ins are rejected: legacy default fusion settings never alter this path.
    for (const auto& flag : {
            std::pair<const char*, const char*>{"VEGAJET_V13_SUPER_NRANK", std::getenv("VEGAJET_V13_SUPER_NRANK")} ,
            std::pair<const char*, const char*>{"VEGAJET_SUPER_WGS", std::getenv("VEGAJET_SUPER_WGS")} ,
            std::pair<const char*, const char*>{"VEGAJET_GU_Q3", std::getenv("VEGAJET_GU_Q3")} ,
            std::pair<const char*, const char*>{"VEGAJET_GU_DOT8_LAYERS", std::getenv("VEGAJET_GU_DOT8_LAYERS")} ,
            std::pair<const char*, const char*>{"VEGAJET_V12_BARRIER_VOL", std::getenv("VEGAJET_V12_BARRIER_VOL")} ,
            std::pair<const char*, const char*>{"VEGAJET_V12_IQ2S_REG", std::getenv("VEGAJET_V12_IQ2S_REG")} ,
            std::pair<const char*, const char*>{"VEGAJET_V12_SERVE_SPEC", std::getenv("VEGAJET_V12_SERVE_SPEC")} ,
            std::pair<const char*, const char*>{"VEGAJET_V12_TREE_VERIFY", std::getenv("VEGAJET_V12_TREE_VERIFY")} ,
            std::pair<const char*, const char*>{"VEGAJET_FR_SPEC_ROWS", std::getenv("VEGAJET_FR_SPEC_ROWS")} ,
            std::pair<const char*, const char*>{"VEGAJET_V12_GDN_SPLITV", std::getenv("VEGAJET_V12_GDN_SPLITV")} ,
            std::pair<const char*, const char*>{"VEGAJET_V12_DENSE_PERSIST", std::getenv("VEGAJET_V12_DENSE_PERSIST")} ,
            std::pair<const char*, const char*>{"VEGAJET_EVICT_ADAPT", std::getenv("VEGAJET_EVICT_ADAPT")} })
        if (env_enabled(flag.second)) return no(flag.first);
    return {};
}

std::vector<uint32_t> expand_heads(const std::vector<uint32_t>& heads, uint32_t width) {
    std::vector<uint32_t> rows;
    for (auto h : heads) for (uint32_t c = 0; c < width; ++c) rows.push_back(h * width + c);
    return rows;
}

RawLayout raw_layout(const Hparams& h, uint32_t context,
                     const std::vector<tp::DeviceProfile>& devices, const tp::ShardOptions& options) {
    if (h.n_embd != 3072 || h.n_head != 32 || h.n_head_kv != 2 || h.head_dim_k != 256 ||
        h.head_dim_v != 256 || h.ssm_groups != 16 || h.ssm_v_heads != 64 || h.ssm_state != 128 ||
        h.ssm_conv != 4 || h.n_ff_exp != 1024 || h.n_ff_shexp != 1024 ||
        h.n_expert != 256 || h.n_expert_used != 8 || !h.vocab || h.vocab > INT32_MAX ||
        h.rope_dims > 256 || h.rope_dims % 2 || h.n_gdn + h.n_attn != h.n_trunk ||
        !context || context > 8192 || !h.n_trunk || h.n_trunk > 48)
        throw std::invalid_argument("generic Ornith: unsupported model geometry/context");
    if (h.recurrent.size() < h.n_trunk ||
        std::count(h.recurrent.begin(), h.recurrent.begin() + h.n_trunk, true) != h.n_gdn ||
        !std::isfinite(h.rms_eps) || h.rms_eps <= 0 || !std::isfinite(h.rope_base) || h.rope_base <= 0)
        throw std::invalid_argument("generic Ornith: invalid layer map/norm/RoPE metadata");
    if (options.row_tile != 32) throw std::invalid_argument("generic Ornith: row_tile=16 is unsupported (MoE needs 32)");
    // Upper bounds include padding, all trunk layers, and fixed scratch. Actual arena
    // allocations are checked too. No knapsack/format selection changes these costs.
    tp::ModelSpec spec;
    spec.replicated_bytes = (128ULL << 20) + uint64_t(h.n_trunk) * (3 * 3072 + 257 * 3072) * 4;
    // GU is 1968 B/intermediate; asymmetric Down has 0.5 B codes + 0.25 B scales
    // per output element. Nonlinear padding is charged after the owner layout is known.
    spec.rows = {{"expert_ff", tp::RowKind::MoeIntermediate, h.n_ff_exp, uint64_t(h.n_trunk) * 256 * (1968 + 2304)},
                 {"shared_ff", tp::RowKind::Dense, h.n_ff_shexp, uint64_t(h.n_trunk) * 3 * 3072},
                 {"vocab", tp::RowKind::Vocabulary, h.vocab, 4096}};
    const uint64_t gdn_cost = uint64_t(h.n_gdn) * (uint64_t(6 * 128 + 4 * 128 + 8) * 3072 +
                                               4ULL * 128 * 128 * 4 + 6ULL * 128 * 3 * 4);
    spec.attention = {{"attn", h.n_head, h.n_head_kv, uint64_t(h.n_attn) * 3 * 256 * 3072,
                       uint64_t(h.n_attn) * (2 * 256 * 3072 + uint64_t(context) * 2 * 256 / 32 * 34)},
                      {"gdn", h.ssm_groups, h.ssm_groups, gdn_cost, 1}};
    RawLayout out;
    out.shards = tp::plan_shards(spec, devices, options);
    for (auto& s : out.shards.ranks) {
        const auto ff = s.rows[0].rows.size();
        if (ff) {
            const auto down = dot8_iq3_asym_layout(256ULL * 3072, static_cast<uint32_t>(ff));
            const uint64_t padding = uint64_t(h.n_trunk) *
                (down.w_plane_bytes() + down.s_plane_bytes() - ff * 2304 * 256);
            s.rows[0].bytes += padding; s.used_bytes += padding;
        }
        if (s.used_bytes > devices[out.ranks.size()].usable_vram_bytes)
            throw std::runtime_error("generic Ornith: padded resident storage exceeds rank " + std::to_string(out.ranks.size()) + " capacity");
        RawRank r;
        r.ff = s.rows[0].rows; r.shared_ff = s.rows[1].rows; r.vocab = s.rows[2].rows;
        r.attn = s.attention[0]; r.groups = s.attention[1].q_heads;
        // GDN maps hv % global_nk: stripe value heads by multiplier, NOT by group.
        for (uint32_t m = 0; m < 4; ++m) for (auto g : r.groups) r.v_heads.push_back(m * 16 + g);
        r.conv_rows = expand_heads(r.groups, 128);
        for (auto i : expand_heads(r.groups, 128)) r.conv_rows.push_back(2048 + i);
        for (auto i : expand_heads(r.v_heads, 128)) r.conv_rows.push_back(4096 + i);
        r.qkv_rows = r.conv_rows;
        r.q_rows = expand_heads(r.attn.q_heads, 512); // interleaved Q and gate
        r.kv_rows = expand_heads(r.attn.kv_heads, 256);
        r.state_bytes = uint64_t(r.v_heads.size()) * 128 * 128 * 4;
        r.conv_bytes = uint64_t(r.conv_rows.size()) * 3 * 4;
        r.kv_bytes = uint64_t(context) * r.attn.kv_heads.size() * 2 * 256 / 32 * 34;
        r.scratch.logits = r.vocab.size() * 4;
        r.scratch.shared_gu = r.shared_ff.size() * 8;
        r.scratch.shared_q = r.shared_ff.size(); r.scratch.shared_ds = r.shared_ff.size() / 32 * 8;
        r.scratch.moe_q = 8 * r.ff.size(); r.scratch.moe_ds = 8 * r.ff.size() / 32 * 4;
        r.scratch.head_map = r.attn.q_to_local_kv.size() * 4;
        out.rank_order.push_back(out.ranks.size());
        out.ranks.push_back(std::move(r));
    }
    return out;
}

} // namespace vegajet::ornith
