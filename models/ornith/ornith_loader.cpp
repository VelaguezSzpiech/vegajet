// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "models/ornith/ornith_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace vegajet::ornith {

using quant::DiskQuantType;
using quant::NativeFormat;
using quant::TensorRole;

namespace {

bool ends_with(const std::string& s, const char* suf) {
    const size_t n = std::strlen(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

int32_t layer_of(const std::string& name) {
    if (name.rfind("blk.", 0) != 0) return -1;
    return static_cast<int32_t>(std::strtol(name.c_str() + 4, nullptr, 10));
}

bool arch_ok(const std::string& a) { return a == "qwen35moe" || a == "qwen3_5_moe" || a == "ornith"; }

} // namespace

bool OrnithModel::can_load(const GgufFile& f) const {
    if (!arch_ok(f.architecture())) return false;
    // Hybrid MoE signature: routed experts plus gated-delta-net state tensors.
    bool exps = false, ssm = false;
    for (const auto& t : f.tensors()) {
        exps |= ends_with(t.name, "ffn_gate_exps.weight");
        ssm |= ends_with(t.name, ".ssm_a");
        if (exps && ssm) return true;
    }
    return false;
}

bool OrnithModel::parse_hparams(const GgufFile& f, std::string* err) {
    Hparams h;
    h.arch = f.architecture();
    uint64_t u = 0;
    double d = 0;
    auto need = [&](const char* key, uint32_t& out) {
        if (!f.get_arch_u64(key, u)) { if (err) *err = "missing " + h.arch + "." + key; return false; }
        out = static_cast<uint32_t>(u);
        return true;
    };
    if (!need("block_count", h.n_block) || !need("embedding_length", h.n_embd) ||
        !need("attention.head_count", h.n_head) || !need("attention.head_count_kv", h.n_head_kv) ||
        !need("expert_count", h.n_expert) || !need("expert_used_count", h.n_expert_used) ||
        !need("ssm.conv_kernel", h.ssm_conv) || !need("ssm.state_size", h.ssm_state) ||
        !need("ssm.group_count", h.ssm_groups) || !need("ssm.time_step_rank", h.ssm_v_heads) ||
        !need("ssm.inner_size", h.ssm_inner))
        return false;
    h.n_nextn = f.get_arch_u64("nextn_predict_layers", u) ? static_cast<uint32_t>(u) : 0;
    h.n_trunk = h.n_block - h.n_nextn;
    h.head_dim_k = f.get_arch_u64("attention.key_length", u) ? static_cast<uint32_t>(u) : h.n_embd / h.n_head;
    h.head_dim_v = f.get_arch_u64("attention.value_length", u) ? static_cast<uint32_t>(u) : h.head_dim_k;
    h.rope_dims = f.get_arch_u64("rope.dimension_count", u) ? static_cast<uint32_t>(u) : h.head_dim_k;
    h.rope_base = f.get_arch_f64("rope.freq_base", d) ? d : 10000.0;
    h.rms_eps = f.get_arch_f64("attention.layer_norm_rms_epsilon", d) ? d : 1e-6;
    h.n_ff_exp = f.get_arch_u64("expert_feed_forward_length", u) ? static_cast<uint32_t>(u) : 0;
    h.n_ff_shexp = f.get_arch_u64("expert_shared_feed_forward_length", u) ? static_cast<uint32_t>(u) : 0;
    h.full_attn_interval = f.get_arch_u64("full_attention_interval", u) ? static_cast<uint32_t>(u) : 4;
    h.context_length = f.get_arch_u64("context_length", u) ? u : 0;
    if (const GgufValue* v = f.find(h.arch + ".rope.dimension_sections"))
        for (double x : v->arr_num) h.rope_sections.push_back(static_cast<uint32_t>(x));
    h.recurrent.assign(h.n_block, false);
    if (const GgufValue* v = f.find(h.arch + ".attention.recurrent_layers"); v && v->arr_num.size() == h.n_block) {
        for (uint32_t i = 0; i < h.n_block; ++i) h.recurrent[i] = v->arr_num[i] != 0.0;
    } else {
        for (uint32_t i = 0; i < h.n_trunk; ++i) h.recurrent[i] = (i + 1) % h.full_attn_interval != 0;
    }
    for (uint32_t i = 0; i < h.n_trunk; ++i) (h.recurrent[i] ? h.n_gdn : h.n_attn)++;
    if (const GgufTensorInfo* o = f.tensor("output.weight")) h.vocab = o->ne[1];
    else if (const GgufTensorInfo* e = f.tensor("token_embd.weight")) h.vocab = e->ne[1];

    // Structural validation: every block has the tensors its type needs.
    for (uint32_t il = 0; il < h.n_block; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        std::vector<std::string> req = {"attn_norm.weight", "post_attention_norm.weight", "ffn_gate_inp.weight",
                                        "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
        if (h.recurrent[il]) for (const char* s : {"attn_qkv.weight", "attn_gate.weight", "ssm_out.weight", "ssm_a"}) req.push_back(s);
        else for (const char* s : {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"}) req.push_back(s);
        for (const auto& r : req)
            if (!f.tensor(p + r)) { if (err) *err = "missing tensor " + p + r; return false; }
    }
    if (h.vocab == 0) { if (err) *err = "missing output.weight / token_embd.weight"; return false; }
    hp_ = std::move(h);
    return true;
}

bool OrnithModel::parse_draft(const std::string& path, LoadReport& rep) {
    GgufFile d;
    if (!d.open(path)) { rep.error = "draft: " + d.error(); return false; }
    if (d.architecture() != "dflash") { rep.error = "draft: unsupported architecture " + d.architecture(); return false; }
    DraftHparams h;
    h.arch = d.architecture();
    uint64_t u = 0;
    d.get_arch_u64("block_count", u); h.n_layer = static_cast<uint32_t>(u);
    d.get_arch_u64("embedding_length", u); h.n_embd = static_cast<uint32_t>(u);
    d.get_arch_u64("feed_forward_length", u); h.n_ff = static_cast<uint32_t>(u);
    d.get_arch_u64("attention.head_count", u); h.n_head = static_cast<uint32_t>(u);
    d.get_arch_u64("attention.head_count_kv", u); h.n_head_kv = static_cast<uint32_t>(u);
    d.get_arch_u64("attention.key_length", u); h.head_dim = static_cast<uint32_t>(u);
    d.get_arch_u64("block_size", u); h.block_size = static_cast<uint32_t>(u);
    if (d.get_arch_u64("conv_kernel_size", u)) h.conv_kernel_size = static_cast<uint32_t>(u);
    if (d.get_arch_u64("conv_group_size", u))  h.conv_group_size  = static_cast<uint32_t>(u);
    if (d.get_arch_u64("selector_rank", u))    h.selector_rank    = static_cast<uint32_t>(u);
    if (d.get_arch_u64("selector_top_k", u))   h.selector_top_k   = static_cast<uint32_t>(u);
    if (d.get_arch_u64("attention.sliding_window", u)) h.sliding_window = static_cast<uint32_t>(u);
    if (d.get_arch_u64("attention.causal", u)) h.attention_causal = (u != 0u);
    double dv = 0.0;
    if (d.get_arch_f64("rope.freq_base", dv)) h.rope_base = dv;
    if (d.get_arch_f64("attention.layer_norm_rms_epsilon", dv)) h.rms_eps = dv;
    if (d.get_u64("tokenizer.ggml.mask_token_id", u)) h.mask_token = u;
    if (const GgufValue* v = d.find("dflash.target_layers"))
        for (double x : v->arr_num) h.target_layers.push_back(static_cast<uint32_t>(x));
    if (h.n_embd != hp_.n_embd) { rep.error = "draft: embedding_length mismatch"; return false; }
    for (uint32_t t : h.target_layers)
        if (t > hp_.n_trunk) { rep.error = "draft: target layer out of range"; return false; }
    if (h.block_size < 2 || h.n_layer == 0) { rep.error = "draft: invalid block_size / block_count"; return false; }
    for (const auto& t : d.tensors()) {
        TensorPlan p;
        p.name = "draft/" + t.name;
        p.layer = layer_of(t.name);
        p.role = TensorRole::MtpDraft;
        p.disk = static_cast<DiskQuantType>(t.type);
        p.in = t.ne[0]; p.out = t.ne[1]; p.experts = t.ne[2];
        p.shard = ShardKind::Rank0;
        p.keep_float = t.n_dims != 2 || t.ne[0] % 32 != 0 || t.type == 0;
        p.format = (h.is_dflash2() && !p.keep_float) ? NativeFormat::VJ_Q4_DOT8 : NativeFormat::VJ_Q8_DP4A;
        const bool host_sel = (h.is_dflash2() && (t.name == "selector_predecessor.weight" || t.name == "selector_successor.weight"));
        p.bytes_total = host_sel ? 0ULL : (p.keep_float ? t.n_elems() * 4 : quant::VramBudgetPlanner::estimate_bytes(p.format, p.disk, p.out, p.in));
        plan_.push_back(p);
    }
    dhp_ = h;
    rep.notes.push_back("draft: dflash " + std::to_string(h.n_layer) + " layers, block " + std::to_string(h.block_size) +
                        ", " + std::to_string(h.target_layers.size()) + " target taps");
    return true;
}

const TensorPlan* OrnithModel::find_plan(const std::string& name) const {
    for (const auto& p : plan_) if (p.name == name) return &p;
    return nullptr;
}

LoadReport OrnithModel::load(GgufFile& f, const LoadOptions& opt) {
    LoadReport rep;
    rep.architecture = f.architecture();
    plan_.clear();
    loaded_ = false;
    if (!can_load(f)) { rep.error = "not a qwen35moe hybrid MoE GGUF"; return rep; }
    if (!parse_hparams(f, &rep.error)) return rep;
    if (opt.gpus.empty()) { rep.error = "no GPUs in LoadOptions"; return rep; }
    opt_ = opt;
    const size_t n = opt.gpus.size();
    bw_.clear();
    for (const auto& g : opt.gpus) bw_.push_back(g.bw_hbm_gbs);
    double bw_sum = 0;
    for (double b : bw_) bw_sum += b;

    attn_ = shard::plan_attention(hp_.n_head, hp_.n_head_kv, bw_);
    gdn_ = shard::plan_gdn(hp_.ssm_groups, hp_.ssm_v_heads, bw_);

    // ---- tensor classification ----------------------------------------------------------------
    for (const auto& t : f.tensors()) {
        TensorPlan p;
        p.name = t.name;
        p.layer = layer_of(t.name);
        p.disk = static_cast<DiskQuantType>(t.type);
        p.in = t.ne[0]; p.out = t.ne[1]; p.experts = t.ne[2];
        rep.weight_bytes_disk += t.n_bytes();
        const bool mtp = p.layer >= static_cast<int32_t>(hp_.n_trunk);
        const std::string& s = t.name;
        p.keep_float = t.n_dims < 2 || t.ne[0] % 32 != 0 || t.type == 0 /*F32 router gates, conv*/;
        if (s == "token_embd.weight") { p.role = TensorRole::Other; p.shard = ShardKind::Rank0; }
        else if (s == "output.weight") { p.role = TensorRole::VocabHead; p.shard = ShardKind::RowSplit; }
        else if (p.keep_float) { p.role = TensorRole::Other; p.shard = ShardKind::Replicated; }
        else if (ends_with(s, "_exps.weight")) { p.role = TensorRole::RoutedMoE; p.shard = ShardKind::Experts; }
        else if (ends_with(s, "ffn_down_shexp.weight")) { p.role = TensorRole::SharedExpert; p.shard = ShardKind::ColSplit; }
        else if (ends_with(s, "_shexp.weight")) { p.role = TensorRole::SharedExpert; p.shard = ShardKind::RowSplit; }
        else if (ends_with(s, "attn_q.weight") || ends_with(s, "attn_output.weight")) { p.role = TensorRole::DenseProj; p.shard = ShardKind::QHeads; }
        else if (ends_with(s, "attn_k.weight") || ends_with(s, "attn_v.weight")) { p.role = TensorRole::DenseProj; p.shard = ShardKind::KvHeads; }
        else if (ends_with(s, "attn_qkv.weight") || ends_with(s, "attn_gate.weight") || ends_with(s, "ssm_alpha.weight") ||
                 ends_with(s, "ssm_beta.weight") || ends_with(s, "ssm_out.weight")) { p.role = TensorRole::DenseProj; p.shard = ShardKind::GdnGroups; }
        else if (ends_with(s, "nextn.eh_proj.weight")) { p.role = TensorRole::MtpDraft; p.shard = ShardKind::RowSplit; }
        else { p.role = TensorRole::DenseProj; p.shard = ShardKind::RowSplit; }
        if (mtp && p.role == TensorRole::DenseProj) p.role = TensorRole::MtpDraft;
        if (!p.keep_float && !quant::DecoderRegistry::instance().supports(p.disk)) {
            rep.error = "tensor " + s + " uses unsupported type " + ggml_type_name(t.type);
            return rep;
        }
        plan_.push_back(p);
    }

    // ---- per-GPU fractions -----------------------------------------------------------------------
    const auto q_heads = shard::apportion(hp_.n_head, bw_);
    const auto experts = shard::apportion(hp_.n_expert, bw_);
    auto fraction = [&](const TensorPlan& p, size_t i) -> double {
        switch (p.shard) {
            case ShardKind::Replicated: return 1.0;
            case ShardKind::Rank0: return i == 0 ? 1.0 : 0.0;
            case ShardKind::RowSplit: { const auto r = shard::split_rows(p.out, bw_); return double(r[i].size()) / p.out; }
            case ShardKind::ColSplit: { const auto r = shard::split_rows(p.in, bw_); return double(r[i].size()) / p.in; }
            case ShardKind::QHeads: return double(q_heads[i]) / hp_.n_head;
            case ShardKind::KvHeads: return double(attn_.gpus[i].kv_heads.size()) / hp_.n_head_kv;
            case ShardKind::GdnGroups: return double(gdn_[i].k_groups.size()) / hp_.ssm_groups;
            case ShardKind::Experts: return double(experts[i]) / std::max<uint64_t>(1, p.experts);
        }
        return 1.0;
    };
    auto local_out = [&](const TensorPlan& p, size_t i) -> uint64_t {
        switch (p.shard) {
            case ShardKind::Experts: return experts[i];
            case ShardKind::ColSplit: case ShardKind::Replicated: return p.out;
            case ShardKind::Rank0: return i == 0 ? p.out : 0;
            default: return static_cast<uint64_t>(std::llround(fraction(p, i) * p.out));
        }
    };

    // ---- KV + recurrent state -------------------------------------------------------------------
    rep.kv_bytes_per_gpu.assign(n, 0);
    rep.state_bytes_per_gpu.assign(n, 0);
    kv_.assign(n, {});
    const uint32_t attn_blocks = hp_.n_attn + [&] { uint32_t c = 0; for (uint32_t i = hp_.n_trunk; i < hp_.n_block; ++i) c += !hp_.recurrent[i]; return c; }();
    for (size_t i = 0; i < n; ++i) {
        kv::KvGeometry g{attn_blocks, static_cast<uint32_t>(attn_.gpus[i].kv_heads.size()), hp_.head_dim_k, hp_.head_dim_v};
        kv_[i] = kv::plan_kv(g, opt.context_length, opt.gpus[i].vram_bytes, opt.context_length);
        rep.kv_bytes_per_gpu[i] = kv_[i].total_bytes;
        const uint32_t vh = static_cast<uint32_t>(gdn_[i].v_heads.size());
        const uint32_t kg = static_cast<uint32_t>(gdn_[i].k_groups.size());
        const uint32_t head_v = hp_.ssm_inner / std::max(1u, hp_.ssm_v_heads);
        ssm::GdnGeometry sg{hp_.n_gdn, vh, hp_.ssm_state, head_v, hp_.ssm_conv, 2 * kg * hp_.ssm_state + vh * head_v, 0};
        sg.replay_input_elems = sg.conv_channels + 2 * vh;
        rep.state_bytes_per_gpu[i] = ssm::describe(sg).total_bytes();
    }

    // ---- fixed (non-knapsack) bytes and the knapsack -----------------------------------------------
    if (!opt.draft_gguf.empty() && !parse_draft(opt.draft_gguf, rep)) return rep;
    std::vector<double> fixed(n, 0.0);
    std::vector<quant::TensorPlanSpec> specs;
    std::vector<size_t> spec_idx;
    for (size_t k = 0; k < plan_.size(); ++k) {
        auto& p = plan_[k];
        if (p.keep_float) p.bytes_total = p.in * p.out * p.experts * 4;
        else if (p.name.rfind("draft/", 0) == 0) { /* sized in parse_draft */ }
        else if (p.shard == ShardKind::Rank0) {
            p.format = (p.disk == DiskQuantType::Q8_0 || p.disk == DiskQuantType::F16 || p.disk == DiskQuantType::BF16 ||
                        p.disk == DiskQuantType::F32) ? NativeFormat::VJ_Q8_DP4A : NativeFormat::VJ_IQ_PACKED;
            p.bytes_total = quant::VramBudgetPlanner::estimate_bytes(p.format, p.disk, p.out, p.in, p.experts);
        } else {
            quant::TensorPlanSpec s{p.name, p.disk, p.role, p.out, p.in, p.experts,
                                    p.role == TensorRole::RoutedMoE ? double(hp_.n_expert_used) / hp_.n_expert : 1.0, false};
            specs.push_back(s);
            spec_idx.push_back(k);
            continue;
        }
        for (size_t i = 0; i < n; ++i) fixed[i] += p.bytes_total * fraction(p, i);
    }
    double agg_budget = 1e30;
    for (size_t i = 0; i < n; ++i) {
        const double usable = double(opt.gpus[i].vram_bytes) - double(opt.kv_scratch_reserve_bytes) - fixed[i] -
                              double(rep.kv_bytes_per_gpu[i]) - double(rep.state_bytes_per_gpu[i]);
        const double share = bw_[i] / bw_sum;
        agg_budget = std::min(agg_budget, std::max(0.0, usable) / share);
    }
    quant::HardwareRooflineProfile hw;
    for (size_t i = 0; i < n; ++i) {
        quant::GpuRooflineInfo g;
        g.id = "gpu" + std::to_string(i);
        g.n_cu = opt.gpus[i].n_cu;
        g.sclk_ghz = opt.gpus[i].sclk_ghz;
        g.bw_hbm_gbs = opt.gpus[i].bw_hbm_gbs;
        hw.gpus.push_back(g);
    }
    const auto dec = quant::VramBudgetPlanner::plan_roofline_knapsack(specs, static_cast<uint64_t>(agg_budget), 0, hw,
                                                                      opt.batch_reuse_factor);
    uint32_t upgraded = 0, rejected = 0;
    for (size_t j = 0; j < dec.size(); ++j) {
        auto& p = plan_[spec_idx[j]];
        p.format = dec[j].target_format;
        p.fwht64 = dec[j].use_fwht64;
        p.bytes_total = dec[j].estimated_vram_bytes;
        upgraded += dec[j].expected_ms_saved > 0;
        rejected += dec[j].rejected_negative_roi;
    }

    // ---- per-GPU accounting -----------------------------------------------------------------------
    rep.weight_bytes_per_gpu.assign(n, 0);
    for (auto& p : plan_) {
        p.bytes_per_gpu.assign(n, 0);
        p.local_out.assign(n, 0);
        for (size_t i = 0; i < n; ++i) {
            p.bytes_per_gpu[i] = static_cast<uint64_t>(std::llround(p.bytes_total * fraction(p, i)));
            p.local_out[i] = local_out(p, i);
            rep.weight_bytes_per_gpu[i] += p.bytes_per_gpu[i];
        }
    }
    rep.ok = true;
    for (size_t i = 0; i < n; ++i) {
        const uint64_t need = rep.weight_bytes_per_gpu[i] + rep.kv_bytes_per_gpu[i] + rep.state_bytes_per_gpu[i] +
                              opt.kv_scratch_reserve_bytes;
        if (need > opt.gpus[i].vram_bytes) {
            rep.ok = false;
            char buf[160];
            std::snprintf(buf, sizeof(buf), "gpu%zu needs %.2f GiB > %.2f GiB VRAM", i, need / 1073741824.0,
                          opt.gpus[i].vram_bytes / 1073741824.0);
            rep.error = buf;
        }
    }
    char buf[200];
    std::snprintf(buf, sizeof(buf), "knapsack: %u upgrades accepted, %u negative-ROI rejected (I*=%.2f, reuse %.1f)",
                  upgraded, rejected, hw.min_i_star(opt.batch_reuse_factor), opt.batch_reuse_factor);
    rep.notes.push_back(buf);
    for (size_t i = 0; i < n; ++i) rep.notes.push_back("gpu" + std::to_string(i) + " KV: " + kv_[i].summary());

    draft_src_ = opt.draft_gguf.empty() ? DraftSource::Suffix : DraftSource::ExternalDFlash;
    spec::MoeCostModel cm;
    cm.n_experts = hp_.n_expert;
    cm.experts_per_token = hp_.n_expert_used;
    {   // routed-expert share of per-token decode bytes, from the plan itself
        double moe = 0, total = 0;
        for (const auto& p : plan_) {
            if (p.name.rfind("draft/", 0) == 0 || p.name == "token_embd.weight") continue;
            const double per_tok = p.role == TensorRole::RoutedMoE ? p.bytes_total * double(hp_.n_expert_used) / hp_.n_expert
                                                                   : double(p.bytes_total);
            total += per_tok;
            if (p.role == TensorRole::RoutedMoE) moe += per_tok;
        }
        cm.moe_byte_fraction = total > 0 ? moe / total : 0.5;
    }
    gov_ = spec::MoeKGovernor(cm);
    ring_ = ssm::ReplayRing();
    suffix_ = spec::SuffixTree();
    pos_ = 0;
    loaded_ = rep.ok || opt.dry_run;
    return rep;
}

bool OrnithModel::repack_shard_for_test(GgufFile& f, const std::string& tensor, uint32_t rank, uint32_t max_rows,
                                        std::vector<float>& native, std::vector<float>& disk, std::string* err) const {
    const TensorPlan* p = find_plan(tensor);
    const GgufTensorInfo* t = f.tensor(tensor);
    if (!p || !t || p->keep_float || rank >= p->local_out.size() || (p->experts > 1 && rank != 0)) {
        if (err) *err = "tensor not repackable: " + tensor;
        return false;
    }
    // Locate the rank's row range (row-split style shards).
    // Expert tensors: rank 0 owns expert 0 first, so its leading rows are expert 0's rows.
    uint64_t begin = 0;
    for (uint32_t i = 0; i < rank; ++i) begin += p->local_out[i];
    const uint64_t avail = p->experts > 1 ? p->out : p->local_out[rank];
    const uint64_t rows = std::min<uint64_t>(avail, max_rows);
    const auto& reg = quant::DecoderRegistry::instance();
    const uint64_t rb = reg.row_bytes(p->disk, p->in);
    const uint8_t* base = f.tensor_data(*t);
    if (!base || rows == 0) { if (err) *err = "no data for " + tensor; return false; }
    const uint8_t* src = base + begin * rb;
    disk.assign(rows * p->in, 0.0f);
    for (uint64_t r = 0; r < rows; ++r) reg.decode_row_f32(p->disk, src + r * rb, disk.data() + r * p->in, p->in);
    native.assign(rows * p->in, 0.0f);
    const uint32_t R = static_cast<uint32_t>(rows), C = static_cast<uint32_t>(p->in);
    switch (p->format) {
        case NativeFormat::VJ_Q8_DP4A: quant::dequantize_vj_q8(quant::repack_to_vj_q8(p->disk, src, R, C), native.data()); break;
        case NativeFormat::VJ_Q4_DOT8: quant::dequantize_vj_q4_dot8(quant::repack_to_vj_q4_dot8(p->disk, src, R, C, p->fwht64), native.data()); break;
        case NativeFormat::VJ_Q4_A8:   quant::dequantize_vj_q4_a8(quant::repack_to_vj_q4_a8(p->disk, src, R, C), native.data()); break;
        case NativeFormat::VJ_IQ_PACKED: quant::dequantize_vj_iq(quant::repack_to_vj_iq(p->disk, src, R, C), native.data()); break;
    }
    f.release(src, rows * rb);
    return true;
}

} // namespace vegajet::ornith
