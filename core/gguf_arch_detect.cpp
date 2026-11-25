// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/gguf_arch_detect.hpp"

#include <algorithm>
#include <unordered_set>
#include <utility>

namespace vegajet::arch {

const char* family_name(Family family) {
    switch (family) {
        case Family::DenseGqa: return "dense-gqa";
        case Family::PureMoe: return "pure-moe";
        case Family::HybridGdnMoe: return "hybrid-gdn-moe";
        case Family::MlaMoe: return "mla-moe";
        case Family::Unknown: break;
    }
    return "unknown";
}

namespace {

struct LayerFacts {
    bool gdn = false, mla = false, attn = false, moe = false, dense = false, latent = false;
};

const char* const kSsmKeys[] = {"ssm.group_count", "ssm.time_step_rank", "ssm.state_size", "ssm.inner_size",
                                "ssm.conv_kernel"};

} // namespace

bool detect(const GgufFile& file, Detection& out, std::string* err) {
    out = Detection{};
    out.arch = file.architecture();

    auto fail = [&](const std::string& msg) {
        out.family = Family::Unknown;
        out.problems.push_back(msg);
        if (err) *err = "gguf family detection failed: " + msg;
        return false;
    };
    auto fail_accumulated = [&]() {
        out.family = Family::Unknown;
        if (err) {
            std::string msg = "gguf family detection failed:";
            const size_t shown = std::min<size_t>(out.problems.size(), 3);
            for (size_t i = 0; i < shown; ++i) msg += (i == 0 ? " " : "; ") + out.problems[i];
            if (out.problems.size() > shown) msg += " (+" + std::to_string(out.problems.size() - shown) + " more)";
            *err = msg;
        }
        return false;
    };
    auto conflict = [&](const std::string& msg) { out.problems.push_back(msg); };
    auto arch_key = [&](const char* suffix) { return out.arch + "." + suffix; };
    auto present = [&](const char* suffix) { return file.find(arch_key(suffix)) != nullptr; };

    if (out.arch.empty()) return fail("general.architecture is missing");

    uint64_t blocks64 = 0;
    if (!file.get_arch_u64("block_count", blocks64) || blocks64 == 0 || blocks64 > 4096)
        return fail(arch_key("block_count") + " is missing, zero or above 4096");
    const uint32_t blocks = static_cast<uint32_t>(blocks64);
    uint64_t nextn = 0;
    if (present("nextn_predict_layers") && !file.get_arch_u64("nextn_predict_layers", nextn))
        return fail(arch_key("nextn_predict_layers") + " is not an integer");
    if (nextn >= blocks) return fail(arch_key("nextn_predict_layers") + " covers every block");
    const uint32_t n_layers = blocks - static_cast<uint32_t>(nextn);
    out.n_layers = n_layers;

    uint64_t q_heads = 0, kv_heads = 0;
    if (!file.get_arch_u64("attention.head_count", q_heads) || q_heads == 0 || q_heads > 65536)
        return fail(arch_key("attention.head_count") + " is missing, zero or above 65536");
    if (present("attention.head_count_kv") &&
        (!file.get_arch_u64("attention.head_count_kv", kv_heads) || kv_heads == 0 || kv_heads > q_heads))
        return fail(arch_key("attention.head_count_kv") + " is not an integer in 1..attention.head_count");
    out.n_q_heads = static_cast<uint32_t>(q_heads);
    out.n_kv_heads = static_cast<uint32_t>(kv_heads);

    const GgufValue* recurrent = file.find(arch_key("attention.recurrent_layers"));
    if (recurrent) {
        if (!recurrent->is_array() || recurrent->arr_truncated || recurrent->arr_num.size() < n_layers)
            return fail(arch_key("attention.recurrent_layers") +
                        " must retain one numeric entry per main layer");
    }
    const GgufValue* swa_pattern = file.find(arch_key("attention.sliding_window_pattern"));
    if (swa_pattern) {
        if (!swa_pattern->is_array() || swa_pattern->arr_truncated || swa_pattern->arr_num.size() < n_layers)
            return fail(arch_key("attention.sliding_window_pattern") +
                        " must retain one numeric entry per main layer");
    }
    uint64_t interval = 0;
    if (present("full_attention_interval") && !file.get_arch_u64("full_attention_interval", interval))
        return fail(arch_key("full_attention_interval") + " is not an integer");
    if (present("full_attention_interval") && (interval == 0 || interval > blocks))
        return fail(arch_key("full_attention_interval") + " is zero or larger than block_count");

    std::unordered_set<std::string> names;
    names.reserve(file.tensors().size() * 2 + 1);
    for (const auto& t : file.tensors()) names.insert(t.name);
    auto has = [&](const std::string& name) { return names.find(name) != names.end(); };

    std::vector<LayerFacts> facts(n_layers);
    int first_gdn = -1, first_mla = -1, first_gqa = -1, first_moe = -1, first_dense = -1;
    for (uint32_t L = 0; L < n_layers; ++L) {
        const std::string p = "blk." + std::to_string(L) + ".";
        LayerFacts& lf = facts[L];
        lf.gdn = has(p + "ssm_a") || has(p + "ssm_conv1d.weight") || has(p + "ssm_out.weight") ||
                 has(p + "attn_qkv.weight");
        lf.mla = has(p + "attn_kv_a_mqa.weight") || has(p + "attn_kv_a_norm.weight") || has(p + "attn_kv_b.weight");
        lf.attn = has(p + "attn_q.weight") || has(p + "attn_q_a.weight") || has(p + "attn_q_b.weight") ||
                  has(p + "attn_k.weight") || has(p + "attn_v.weight") || has(p + "attn_output.weight");
        lf.moe = has(p + "ffn_gate_exps.weight") || has(p + "ffn_up_exps.weight") ||
                 has(p + "ffn_down_exps.weight") || has(p + "ffn_gate_inp.weight");
        lf.latent = has(p + "ffn_latent_down.weight") || has(p + "ffn_latent_up.weight");
        lf.dense = has(p + "ffn_gate.weight") || has(p + "ffn_up.weight") || has(p + "ffn_down.weight");

        if (lf.gdn && lf.mla) conflict("block " + std::to_string(L) + " mixes GDN and MLA attention tensors");
        if (!lf.gdn && !lf.mla && !lf.attn)
            conflict("block " + std::to_string(L) + " has no recognized attention tensor");
        if (lf.moe && lf.dense) conflict("block " + std::to_string(L) + " has both dense and expert FFN tensors");
        if (!lf.moe && !lf.dense && !lf.latent)
            conflict("block " + std::to_string(L) + " has no dense or expert FFN tensor");
        if (lf.latent && !lf.moe)
            conflict("block " + std::to_string(L) + " has LatentMoE tensors without expert tensors");
        if (lf.gdn && first_gdn < 0) first_gdn = static_cast<int>(L);
        if (lf.mla && first_mla < 0) first_mla = static_cast<int>(L);
        if (lf.moe && first_moe < 0) first_moe = static_cast<int>(L);
        if (lf.dense && first_dense < 0) first_dense = static_cast<int>(L);
        if (!lf.gdn && !lf.mla && lf.attn && first_gqa < 0) first_gqa = static_cast<int>(L);
    }

    for (uint32_t L = 0; L < n_layers; ++L) {
        const LayerFacts& lf = facts[L];
        if (recurrent) {
            const bool expected = recurrent->arr_num[L] != 0.0;
            if (expected != lf.gdn)
                conflict("attention.recurrent_layers entry " + std::to_string(L) +
                         " disagrees with the block's attention tensors");
        } else if (present("full_attention_interval")) {
            const bool expected = ((L + 1u) % static_cast<uint32_t>(interval)) != 0u;
            if (expected != lf.gdn)
                conflict("full_attention_interval implies a different mixer for block " + std::to_string(L));
        }
        if (lf.gdn) {
            ++out.gdn_layers;
        } else if (lf.mla) {
            ++out.mla_layers;
        } else {
            ++out.gqa_layers;
            if (swa_pattern && swa_pattern->arr_num[L] != 0.0) ++out.swa_layers;
        }
        if (lf.moe) ++out.moe_ffn_layers;
        if (lf.dense) ++out.dense_ffn_layers;
    }

    if (out.gdn_layers > 0 && out.mla_layers > 0)
        conflict("the model mixes GDN and MLA layers, which no supported family allows");

    if (swa_pattern && out.mla_layers > 0) {
        for (uint32_t L = 0; L < n_layers; ++L) {
            const bool expected_mla = swa_pattern->arr_num[L] == 0.0;
            if (expected_mla != facts[L].mla)
                conflict("attention.sliding_window_pattern entry " + std::to_string(L) +
                         " disagrees with the block's MLA tensors");
        }
    }

    bool ssm_meta = false;
    for (const char* key : kSsmKeys) ssm_meta = ssm_meta || present(key);
    if (ssm_meta && out.gdn_layers == 0) conflict("ssm.* metadata is present but no block has GDN tensors");
    if (!ssm_meta && out.gdn_layers > 0) conflict("GDN tensors are present without ssm.* metadata");

    uint64_t kv_lora = 0;
    const bool has_kv_lora = file.get_arch_u64("attention.kv_lora_rank", kv_lora);
    if (present("attention.kv_lora_rank") && !has_kv_lora)
        conflict(arch_key("attention.kv_lora_rank") + " is not an integer");
    if (has_kv_lora && kv_lora == 0) conflict(arch_key("attention.kv_lora_rank") + " is zero");
    uint64_t rope_dim = 0;
    const bool has_rope = file.get_arch_u64("rope.dimension_count", rope_dim);
    if (present("rope.dimension_count") && !has_rope)
        conflict(arch_key("rope.dimension_count") + " is not an integer");
    if ((out.mla_layers > 0) != (has_kv_lora && kv_lora > 0))
        conflict("MLA attention tensors and attention.kv_lora_rank disagree");
    if (out.mla_layers > 0 && (!has_rope || rope_dim == 0))
        conflict("MLA layers require a non-zero rope.dimension_count");

    uint64_t n_exp = 0, top_k = 0;
    const bool has_n_exp = file.get_arch_u64("expert_count", n_exp);
    const bool has_top_k = file.get_arch_u64("expert_used_count", top_k);
    if (present("expert_count") && !has_n_exp) conflict(arch_key("expert_count") + " is not an integer");
    if (present("expert_used_count") && !has_top_k) conflict(arch_key("expert_used_count") + " is not an integer");
    if (present("expert_count") != present("expert_used_count"))
        conflict("expert_count and expert_used_count must appear together");
    if (has_n_exp && (n_exp == 0 || n_exp > 4096)) conflict(arch_key("expert_count") + " is zero or above 4096");
    if (has_n_exp && has_top_k && (top_k == 0 || top_k > n_exp))
        conflict(arch_key("expert_used_count") + " must be in 1..expert_count");
    if ((out.moe_ffn_layers > 0) != (has_n_exp && n_exp > 0))
        conflict("expert FFN tensors and expert_count disagree");
    out.n_experts = has_n_exp && n_exp <= UINT32_MAX ? static_cast<uint32_t>(n_exp) : 0;
    out.top_k = has_top_k && top_k <= UINT32_MAX ? static_cast<uint32_t>(top_k) : 0;

    uint64_t latent = 0;
    const bool has_latent = file.get_arch_u64("moe_latent_size", latent);
    if (present("moe_latent_size") && !has_latent) conflict(arch_key("moe_latent_size") + " is not an integer");
    if (has_latent && latent == 0) conflict(arch_key("moe_latent_size") + " is zero");
    bool latent_tensors = false;
    for (const auto& lf : facts) latent_tensors = latent_tensors || lf.latent;
    if (latent_tensors != (has_latent && latent > 0))
        conflict("ffn_latent_down/up tensors and moe_latent_size disagree");

    if (out.dense_ffn_layers > 0 && out.moe_ffn_layers > 0) {
        uint64_t leading = 0;
        if (!file.get_arch_u64("leading_dense_block_count", leading) || leading != out.dense_ffn_layers) {
            conflict("mixed dense/expert FFN layers require leading_dense_block_count == the dense layer count");
        } else {
            for (uint32_t L = 0; L < out.dense_ffn_layers; ++L) {
                if (!facts[L].dense) {
                    conflict("leading dense FFN layers must be the first blocks");
                    break;
                }
            }
        }
    } else if (out.dense_ffn_layers == 0 && present("leading_dense_block_count")) {
        uint64_t leading = 0;
        if (file.get_arch_u64("leading_dense_block_count", leading) && leading > 0)
            conflict("leading_dense_block_count is set but no block has dense FFN tensors");
    }

    if (!out.problems.empty()) return fail_accumulated();

    if (out.mla_layers > 0) {
        out.family = Family::MlaMoe;
    } else if (out.gdn_layers > 0) {
        out.family = Family::HybridGdnMoe;
    } else if (out.moe_ffn_layers > 0) {
        out.family = Family::PureMoe;
    } else {
        out.family = Family::DenseGqa;
    }

    if (out.family != Family::DenseGqa && !(has_n_exp && n_exp > 0))
        conflict(std::string(family_name(out.family)) + " requires routed experts (expert_count)");
    if (out.gqa_layers > 0 && out.n_kv_heads == 0)
        conflict("GQA/SWA layers require attention.head_count_kv");
    if (!out.problems.empty()) return fail_accumulated();

    out.evidence.push_back("general.architecture=" + out.arch);
    out.evidence.push_back("block_count=" + std::to_string(blocks) + ", main layers=" + std::to_string(n_layers));
    out.evidence.push_back("mixers: gqa=" + std::to_string(out.gqa_layers) + " swa=" +
                           std::to_string(out.swa_layers) + " gdn=" + std::to_string(out.gdn_layers) +
                           " mla=" + std::to_string(out.mla_layers));
    out.evidence.push_back("ffn: moe=" + std::to_string(out.moe_ffn_layers) + " dense=" +
                           std::to_string(out.dense_ffn_layers));
    if (first_gqa >= 0)
        out.evidence.push_back("blk." + std::to_string(first_gqa) + ".attn_q.weight (GQA/SWA)");
    if (first_gdn >= 0)
        out.evidence.push_back("blk." + std::to_string(first_gdn) + ".attn_qkv.weight + ssm_a (GDN)");
    if (first_mla >= 0)
        out.evidence.push_back("blk." + std::to_string(first_mla) + ".attn_kv_a_mqa.weight (MLA)");
    if (first_moe >= 0)
        out.evidence.push_back("blk." + std::to_string(first_moe) + ".ffn_gate_exps.weight (routed experts)");
    if (first_dense >= 0)
        out.evidence.push_back("blk." + std::to_string(first_dense) + ".ffn_gate.weight (dense FFN)");
    if (has_n_exp)
        out.evidence.push_back("expert_count=" + std::to_string(n_exp) + ", expert_used_count=" +
                               std::to_string(top_k));
    if (has_kv_lora)
        out.evidence.push_back("attention.kv_lora_rank=" + std::to_string(kv_lora) + ", rope.dimension_count=" +
                               std::to_string(rope_dim));
    if (has_latent) out.evidence.push_back("moe_latent_size=" + std::to_string(latent));
    if (recurrent) out.evidence.push_back("attention.recurrent_layers matched the GDN/GQA layout");
    if (swa_pattern && out.mla_layers > 0)
        out.evidence.push_back("attention.sliding_window_pattern matched the MLA/SWA layout");
    if (interval) out.evidence.push_back("full_attention_interval=" + std::to_string(interval));
    return true;
}

bool detect_file(const std::string& path, Detection& out, std::string* err) {
    GgufFile file;
    if (!file.open(path)) {
        out = Detection{};
        out.problems.push_back(file.error());
        if (err) *err = "gguf family detection failed: " + file.error();
        return false;
    }
    return detect(file, out, err);
}

} // namespace vegajet::arch
