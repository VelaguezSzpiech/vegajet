// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "models/model_blueprints.hpp"

#include <algorithm>
#include <set>

namespace vegajet::blueprint {

const char* mixer_name(Mixer m) {
    switch (m) {
        case Mixer::Gdn: return "gdn";
        case Mixer::Gqa: return "gqa";
        case Mixer::Swa: return "swa";
        case Mixer::Mla: return "mla";
    }
    return "?";
}

const char* ffn_name(Ffn f) {
    switch (f) {
        case Ffn::Dense: return "dense";
        case Ffn::Moe: return "moe";
        case Ffn::LatentMoe: return "latent-moe";
    }
    return "?";
}

uint32_t Description::count(Mixer m) const { return static_cast<uint32_t>(std::count(mixer.begin(), mixer.end(), m)); }
uint32_t Description::count(Ffn f) const { return static_cast<uint32_t>(std::count(ffn.begin(), ffn.end(), f)); }

namespace {

using Shape = std::vector<uint64_t>;

bool need(const GgufFile& f, const char* key, uint32_t& out, std::string* err) {
    uint64_t v = 0;
    if (!f.get_arch_u64(key, v) || v == 0 || v > UINT32_MAX) {
        if (err) *err = f.architecture() + "." + key + " missing or invalid";
        return false;
    }
    out = static_cast<uint32_t>(v);
    return true;
}

uint32_t opt(const GgufFile& f, const char* key, uint32_t fallback) {
    uint64_t v = 0;
    return f.get_arch_u64(key, v) && v <= UINT32_MAX ? static_cast<uint32_t>(v) : fallback;
}

const GgufValue* arch_array(const GgufFile& f, const char* key, uint32_t min_len) {
    const GgufValue* v = f.find(f.architecture() + "." + key);
    return v && v->is_array() && v->arr_num.size() >= min_len ? v : nullptr;
}

bool common(const GgufFile& f, Description& d, std::string* err) {
    d.arch = f.architecture();
    uint32_t blocks = 0;
    if (!need(f, "embedding_length", d.dims.d_model, err) || !need(f, "block_count", blocks, err)) return false;
    d.dims.nextn_layers = std::min(blocks - 1, opt(f, "nextn_predict_layers", 0));
    d.dims.n_layers = blocks - d.dims.nextn_layers;
    if (const GgufTensorInfo* o = f.tensor("output.weight")) d.dims.vocab = static_cast<uint32_t>(o->ne[1]);
    else if (const GgufTensorInfo* e = f.tensor("token_embd.weight")) d.dims.vocab = static_cast<uint32_t>(e->ne[1]);
    if (!d.dims.vocab) { if (err) *err = "neither output.weight nor token_embd.weight present"; return false; }
    if (d.dims.nextn_layers) d.notes.push_back(std::to_string(d.dims.nextn_layers) + " MTP / nextn layer(s) excluded from the main stack");
    return true;
}

// ---- tensor-map builders (ne order: [in, out(, experts)])
struct Map {
    std::vector<TensorSpec>& out;
    void add(const std::string& n, std::vector<Shape> s, bool req = true) { out.push_back({n, std::move(s), req}); }
};

void globals(Map m, const Dims& D, bool tied_ok) {
    m.add("token_embd.weight", {{D.d_model, D.vocab}});
    m.add("output_norm.weight", {{D.d_model}});
    m.add("output.weight", {{D.d_model, D.vocab}}, !tied_ok);
}

void gqa(Map m, const std::string& p, const Dims& D, bool gated_q, bool qk_norm_required) {
    const uint64_t q = uint64_t(D.n_q_heads) * D.head_dim, kv = uint64_t(D.n_kv_heads) * D.head_dim;
    std::vector<Shape> qs = {{D.d_model, q}};
    if (gated_q) qs.push_back({D.d_model, 2 * q});  // query + output gate packed
    m.add(p + "attn_q.weight", qs);
    m.add(p + "attn_k.weight", {{D.d_model, kv}});
    m.add(p + "attn_v.weight", {{D.d_model, kv}});
    m.add(p + "attn_output.weight", {{q, D.d_model}});
    m.add(p + "attn_q_norm.weight", {{D.head_dim}}, qk_norm_required);
    m.add(p + "attn_k_norm.weight", {{D.head_dim}}, qk_norm_required);
}

void swa(Map m, const std::string& p, const Dims& D) {
    const uint64_t q = uint64_t(D.n_q_heads) * D.swa_head_dim, kv = uint64_t(D.swa_kv_heads) * D.swa_head_dim;
    m.add(p + "attn_q.weight", {{D.d_model, q}});
    m.add(p + "attn_k.weight", {{D.d_model, kv}});
    m.add(p + "attn_v.weight", {{D.d_model, kv}});
    m.add(p + "attn_output.weight", {{q, D.d_model}});
}

void mla(Map m, const std::string& p, const Dims& D) {
    const uint64_t qk = uint64_t(D.n_q_heads) * (D.qk_nope + D.qk_rope);
    if (D.q_lora) {
        m.add(p + "attn_q_a.weight", {{D.d_model, D.q_lora}});
        m.add(p + "attn_q_a_norm.weight", {{D.q_lora}});
        m.add(p + "attn_q_b.weight", {{D.q_lora, qk}});
    } else {
        m.add(p + "attn_q.weight", {{D.d_model, qk}});
    }
    m.add(p + "attn_kv_a_mqa.weight", {{D.d_model, uint64_t(D.kv_lora) + D.qk_rope}});
    m.add(p + "attn_kv_a_norm.weight", {{D.kv_lora}});
    m.add(p + "attn_kv_b.weight", {{D.kv_lora, uint64_t(D.n_q_heads) * (D.qk_nope + D.v_head_dim)}});
    m.add(p + "attn_output.weight", {{uint64_t(D.n_q_heads) * D.v_head_dim, D.d_model}});
}

void gdn(Map m, const std::string& p, const Dims& D) {
    const uint64_t qkv = 2ull * D.gdn_k_groups * D.gdn_k_dim + uint64_t(D.gdn_v_heads) * D.gdn_v_dim;
    const uint64_t v = uint64_t(D.gdn_v_heads) * D.gdn_v_dim;
    m.add(p + "attn_qkv.weight", {{D.d_model, qkv}});      // required by HEAD's ornith loader as well
    m.add(p + "attn_gate.weight", {{D.d_model, v}});
    m.add(p + "ssm_out.weight", {{v, D.d_model}});
    m.add(p + "ssm_a", {{D.gdn_v_heads}});
    m.add(p + "ssm_alpha.weight", {{D.d_model, D.gdn_v_heads}}, false);
    m.add(p + "ssm_beta.weight", {{D.d_model, D.gdn_v_heads}}, false);
    m.add(p + "ssm_conv1d.weight", {{D.conv_kernel, qkv}}, false);
    m.add(p + "ssm_dt.bias", {{D.gdn_v_heads}}, false);
    m.add(p + "ssm_norm.weight", {{D.gdn_v_dim}}, false);
}

void dense_ffn(Map m, const std::string& p, uint32_t d, uint32_t ff) {
    m.add(p + "ffn_gate.weight", {{d, ff}});
    m.add(p + "ffn_up.weight", {{d, ff}});
    m.add(p + "ffn_down.weight", {{ff, d}});
}

void shared_expert(Map m, const std::string& p, const Dims& D) {
    if (!D.shared_ff) return;
    m.add(p + "ffn_gate_shexp.weight", {{D.d_model, D.shared_ff}});
    m.add(p + "ffn_up_shexp.weight", {{D.d_model, D.shared_ff}});
    m.add(p + "ffn_down_shexp.weight", {{D.shared_ff, D.d_model}});
    m.add(p + "ffn_gate_inp_shexp.weight", {{D.d_model}, {D.d_model, 1}}, false);
}

void moe(Map m, const std::string& p, const Dims& D, uint32_t width) {
    m.add(p + "ffn_gate_inp.weight", {{D.d_model, D.n_experts}});
    m.add(p + "ffn_gate_exps.weight", {{width, D.expert_ff, D.n_experts}});
    m.add(p + "ffn_up_exps.weight", {{width, D.expert_ff, D.n_experts}});
    m.add(p + "ffn_down_exps.weight", {{D.expert_ff, width, D.n_experts}});
    shared_expert(m, p, D);
}

void tables(Description& d) {
    const Dims& D = d.dims;
    if (D.ple_dim) {
        d.tables.push_back({"per_layer_token_embd.weight", D.vocab, D.n_layers * D.ple_dim, "ple"});
        d.tensors.push_back({"per_layer_token_embd.weight", {{uint64_t(D.n_layers) * D.ple_dim, D.vocab}}, true});
    }
    if (D.engram_rows) {
        d.tables.push_back({"engram_embd.weight", D.engram_rows, D.engram_dim, "engram"});
        d.tensors.push_back({"engram_embd.weight", {{D.engram_dim, D.engram_rows}}, true});
    }
}

// ---- blueprints
bool describe_flash_next(const GgufFile& f, Description& d, std::string* err) {
    if (!common(f, d, err)) return false;
    Dims& D = d.dims;
    uint32_t inner = 0;
    if (!need(f, "attention.head_count", D.n_q_heads, err) || !need(f, "attention.head_count_kv", D.n_kv_heads, err) ||
        !need(f, "ssm.group_count", D.gdn_k_groups, err) || !need(f, "ssm.time_step_rank", D.gdn_v_heads, err) ||
        !need(f, "ssm.state_size", D.gdn_k_dim, err) || !need(f, "ssm.inner_size", inner, err) ||
        !need(f, "ssm.conv_kernel", D.conv_kernel, err) || !need(f, "expert_count", D.n_experts, err) ||
        !need(f, "expert_used_count", D.top_k, err) || !need(f, "expert_feed_forward_length", D.expert_ff, err))
        return false;
    D.head_dim = opt(f, "attention.key_length", D.d_model / D.n_q_heads);
    D.gdn_v_dim = inner / D.gdn_v_heads;
    D.shared_ff = opt(f, "expert_shared_feed_forward_length", 0);
    D.ple_dim = opt(f, "embedding_length_per_layer_input", 0);
    const GgufValue* rec = arch_array(f, "attention.recurrent_layers", D.n_layers);
    const uint32_t interval = opt(f, "full_attention_interval", 0);
    if (!rec && !interval) {
        if (err) *err = d.arch + ": needs attention.recurrent_layers or full_attention_interval for the GDN/GQA pattern";
        return false;
    }
    for (uint32_t L = 0; L < D.n_layers; ++L) {
        const bool recurrent = rec ? rec->arr_num[L] != 0.0 : (L + 1) % interval != 0;
        d.mixer.push_back(recurrent ? Mixer::Gdn : Mixer::Gqa);
        d.ffn.push_back(Ffn::Moe);
    }
    Map m{d.tensors};
    globals(m, D, true);
    for (uint32_t L = 0; L < D.n_layers; ++L) {
        const std::string p = "blk." + std::to_string(L) + ".";
        m.add(p + "attn_norm.weight", {{D.d_model}});
        m.add(p + "post_attention_norm.weight", {{D.d_model}});
        if (d.mixer[L] == Mixer::Gdn) gdn(m, p, D);
        else gqa(m, p, D, true, false);
        moe(m, p, D, D.d_model);
    }
    tables(d);
    return true;
}

bool describe_hy4(const GgufFile& f, Description& d, std::string* err) {
    if (!common(f, d, err)) return false;
    Dims& D = d.dims;
    uint32_t key_len = 0;
    if (!need(f, "attention.head_count", D.n_q_heads, err) || !need(f, "attention.head_count_kv", D.n_kv_heads, err) ||
        !need(f, "attention.kv_lora_rank", D.kv_lora, err) || !need(f, "rope.dimension_count", D.qk_rope, err) ||
        !need(f, "attention.key_length", key_len, err) || !need(f, "attention.value_length", D.v_head_dim, err) ||
        !need(f, "attention.sliding_window", D.swa_window, err) || !need(f, "expert_count", D.n_experts, err) ||
        !need(f, "expert_used_count", D.top_k, err) || !need(f, "expert_feed_forward_length", D.expert_ff, err) ||
        !need(f, "moe_latent_size", D.latent, err))
        return false;
    if (key_len <= D.qk_rope) { if (err) *err = d.arch + ": attention.key_length must exceed rope.dimension_count"; return false; }
    D.qk_nope = key_len - D.qk_rope;
    D.q_lora = opt(f, "attention.q_lora_rank", 0);
    D.swa_kv_heads = opt(f, "attention.head_count_kv_swa", D.n_kv_heads);
    D.swa_head_dim = opt(f, "attention.key_length_swa", 0);
    if (!D.swa_head_dim) {
        D.swa_head_dim = D.v_head_dim;
        d.notes.push_back("attention.key_length_swa missing: SWA head dim taken from attention.value_length");
    }
    D.shared_ff = opt(f, "expert_shared_feed_forward_length", 0);
    D.leading_dense = opt(f, "leading_dense_block_count", 0);
    if (D.leading_dense && !need(f, "feed_forward_length", D.dense_ff, err)) return false;
    D.engram_rows = opt(f, "engram.rows", 0);
    D.engram_dim = opt(f, "engram.dim", 0);
    D.engram_ngram = opt(f, "engram.ngram", 0);
    if ((D.engram_rows == 0) != (D.engram_dim == 0)) { if (err) *err = d.arch + ": engram.rows and engram.dim go together"; return false; }
    const GgufValue* pat = arch_array(f, "attention.sliding_window_pattern", D.n_layers);
    if (!pat) {
        if (err) *err = d.arch + ": needs attention.sliding_window_pattern (one entry per layer, 1 = SWA, 0 = MLA)";
        return false;
    }
    for (uint32_t L = 0; L < D.n_layers; ++L) {
        d.mixer.push_back(pat->arr_num[L] != 0.0 ? Mixer::Swa : Mixer::Mla);
        d.ffn.push_back(L < D.leading_dense ? Ffn::Dense : Ffn::LatentMoe);
    }
    Map m{d.tensors};
    globals(m, D, false);
    for (uint32_t L = 0; L < D.n_layers; ++L) {
        const std::string p = "blk." + std::to_string(L) + ".";
        m.add(p + "attn_norm.weight", {{D.d_model}});
        m.add(p + "ffn_norm.weight", {{D.d_model}});
        if (d.mixer[L] == Mixer::Mla) mla(m, p, D);
        else swa(m, p, D);
        if (d.ffn[L] == Ffn::Dense) {
            dense_ffn(m, p, D.d_model, D.dense_ff);
        } else {
            m.add(p + "ffn_latent_down.weight", {{D.d_model, D.latent}});  // proposed name
            m.add(p + "ffn_latent_up.weight", {{D.latent, D.d_model}});    // proposed name
            moe(m, p, D, D.latent);
        }
    }
    tables(d);
    return true;
}

bool describe_dense27(const GgufFile& f, Description& d, std::string* err) {
    if (!common(f, d, err)) return false;
    Dims& D = d.dims;
    if (!need(f, "attention.head_count", D.n_q_heads, err) || !need(f, "attention.head_count_kv", D.n_kv_heads, err) ||
        !need(f, "feed_forward_length", D.dense_ff, err))
        return false;
    D.head_dim = opt(f, "attention.key_length", D.d_model / D.n_q_heads);
    d.mixer.assign(D.n_layers, Mixer::Gqa);
    d.ffn.assign(D.n_layers, Ffn::Dense);
    Map m{d.tensors};
    globals(m, D, true);
    for (uint32_t L = 0; L < D.n_layers; ++L) {
        const std::string p = "blk." + std::to_string(L) + ".";
        m.add(p + "attn_norm.weight", {{D.d_model}});
        m.add(p + "ffn_norm.weight", {{D.d_model}});
        gqa(m, p, D, false, true);
        dense_ffn(m, p, D.d_model, D.dense_ff);
    }
    return true;
}

} // namespace

const std::vector<Blueprint>& blueprints() {
    // Accepted general.architecture values: qwen3next / qwen3 are the existing llama.cpp names; the others are
    // expectations until real GGUF files exist (select a blueprint by name to override).
    static const std::vector<Blueprint> all = {
        {"qwen3.8-flash-next", {"qwen3next", "qwen38next"}, describe_flash_next},
        {"hy4-flash", {"hy4", "hy4flash"}, describe_hy4},
        {"qwen3.8-27b", {"qwen3", "qwen38"}, describe_dense27},
    };
    return all;
}

const Blueprint* find_by_name(const std::string& name) {
    for (const auto& b : blueprints())
        if (name == b.name) return &b;
    return nullptr;
}

const Blueprint* find_for(const GgufFile& f) {
    const std::string a = f.architecture();
    for (const auto& b : blueprints())
        if (std::find(b.archs.begin(), b.archs.end(), a) != b.archs.end()) return &b;
    return nullptr;
}

bool describe(const GgufFile& f, Description& out, std::string* err, const Blueprint* bp) {
    out = Description{};
    if (!bp) bp = find_for(f);
    if (!bp) { if (err) *err = "no blueprint for architecture '" + f.architecture() + "'"; return false; }
    out.blueprint = bp->name;
    if (!bp->describe(f, out, err)) return false;
    if (err) err->clear();
    return true;
}

TensorCheck check_tensors(const GgufFile& f, const Description& d) {
    TensorCheck c;
    std::set<std::string> known;
    auto problem = [&](const std::string& s) { if (c.problems.size() < 32) c.problems.push_back(s); };
    for (const auto& s : d.tensors) {
        ++c.checked;
        known.insert(s.name);
        const GgufTensorInfo* t = f.tensor(s.name);
        if (!t) {
            if (s.required) { ++c.missing_required; problem("missing " + s.name); }
            else ++c.missing_optional;
            continue;
        }
        ++c.present;
        bool match = false;
        for (const auto& shape : s.shapes) {
            if (shape.size() != t->n_dims) continue;
            bool all = true;
            for (size_t i = 0; i < shape.size(); ++i) all = all && (shape[i] == 0 || shape[i] == t->ne[i]);
            match = match || all;
        }
        if (!match) {
            ++c.bad_shape;
            std::string got = "[";
            for (uint32_t i = 0; i < t->n_dims; ++i) got += (i ? "," : "") + std::to_string(t->ne[i]);
            problem("shape " + s.name + " " + got + "]");
        }
    }
    for (const auto& t : f.tensors())
        if (!known.count(t.name)) { ++c.unexpected; problem("unexpected " + t.name); }
    return c;
}

bool lower(const Description& d, ops::ModelDesc& m, std::vector<std::string>& notes, std::string* err) {
    m = ops::ModelDesc{};
    notes.clear();
    const Dims& D = d.dims;
    m.arch = d.arch;
    m.d_model = D.d_model;
    m.vocab = D.vocab;
    m.dense_ff = D.dense_ff;
    const bool has_gqa = d.count(Mixer::Gqa) > 0, has_swa = d.count(Mixer::Swa) > 0, has_mla = d.count(Mixer::Mla) > 0;
    if (has_gqa) m.attn = {D.n_q_heads, D.n_kv_heads, D.head_dim};
    else if (has_swa) m.attn = {D.n_q_heads, D.swa_kv_heads, D.swa_head_dim};
    else if (has_mla) m.attn = {D.n_q_heads, 1, D.kv_lora + D.qk_rope};  // latent KV: sequence-sharded by HEAD's sharder
    if (has_swa) notes.push_back("SWA layers lowered to GQA: plan their steps with ctx_tokens <= " + std::to_string(D.swa_window));
    if (has_mla && (has_swa || has_gqa))
        notes.push_back("MLA layers share the planner's single attention geometry (" + std::to_string(m.attn.n_kv_heads) +
                        " KV heads x " + std::to_string(m.attn.head_dim) + "); their latent KV (" +
                        std::to_string(D.kv_lora + D.qk_rope) + " per token) is not modelled");
    if (has_mla && !has_swa && !has_gqa) notes.push_back("MLA lowered to single-KV attention (absorbed form)");
    m.gdn = {D.gdn_k_groups, D.gdn_v_heads, D.gdn_k_dim, D.gdn_v_dim, D.conv_kernel};
    m.moe = {D.n_experts, D.top_k, D.expert_ff, D.shared_ff, ops::ExpertShard::SplitFf};
    for (size_t L = 0; L < d.mixer.size(); ++L) {
        m.mixer.push_back(d.mixer[L] == Mixer::Gdn ? ops::MixerKind::Gdn : ops::MixerKind::Gqa);
        m.ffn.push_back(d.ffn[L] == Ffn::Dense ? ops::FfnKind::Dense : ops::FfnKind::Moe);
    }
    if (d.count(Ffn::LatentMoe))
        notes.push_back("LatentMoE lowered to MoE: experts are " + std::to_string(D.latent) + "-wide (not d_model " +
                        std::to_string(D.d_model) + "), plus ffn_latent_down/up per layer; expert byte counts are overstated");
    if (!d.tables.empty()) notes.push_back("embedding tables (PLE / Engram) are host-resident and fetched by row, not planned");
    return ops::validate_model_desc(m, err);
}

bool table_desc(const GgufFile& f, const TableSpec& t, int fd, prefetch::TableDesc& out, std::string* err) {
    const GgufTensorInfo* ti = f.tensor(t.tensor);
    if (!ti || ti->n_dims != 2) { if (err) *err = "table tensor " + t.tensor + " missing or not 2D"; return false; }
    out = prefetch::TableDesc{};
    if (ti->type == 1) out.fmt = prefetch::RowFormat::F16;        // GGML_TYPE_F16
    else if (ti->type == 2) out.fmt = prefetch::RowFormat::Q4;    // GGML_TYPE_Q4_0
    else { if (err) *err = "table " + t.tensor + ": only F16 and Q4_0 rows can be prefetched as stored"; return false; }
    out.fd = fd;
    out.row_elems = static_cast<uint32_t>(ti->ne[0]);
    out.n_rows = ti->ne[1];
    out.file_offset = f.data_offset() + ti->offset;
    if (out.row_bytes() == 0) { if (err) *err = "table " + t.tensor + ": row length invalid for its type"; return false; }
    return true;
}

} // namespace vegajet::blueprint
