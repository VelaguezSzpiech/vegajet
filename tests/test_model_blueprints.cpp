// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// models/model_blueprints on synthetic GGUF files (tiny dims; no real model files): metadata -> layer pattern for
// qwen3.8-flash-next (GDN/GQA 3:1, MoE + shared, PLE), hy4-flash (12 MLA + 30 SWA, LatentMoE 512 / top-16, leading
// dense block, Engram) and qwen3.8-27b (dense GQA); full tensor maps checked against files generated from them,
// mutations caught (missing / reshaped / unexpected tensors), missing structural keys rejected, lowering to HEAD's
// ModelDesc + build_step_plan on 1 and 3 GPUs, and PLE / Engram rows streamed out of the GGUF by the prefetcher.
#include "models/model_blueprints.hpp"
#include "tests/test_common.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <functional>

using namespace vegajet;
using namespace vegajet::blueprint;

namespace {

uint8_t pat(const std::string& name, size_t i) { return static_cast<uint8_t>((name.size() * 131 + i * 7) & 0xFF); }

using Meta = std::function<void(vjtest::GgufWriter&)>;

// Writes metadata + the given tensors (zeros, except table tensors which get a recognizable pattern).
std::string write(const Meta& meta, const std::vector<TensorSpec>& tensors, const std::string& tag,
                  const std::function<bool(const std::string&)>& skip = nullptr,
                  const std::function<void(std::string&, std::vector<uint64_t>&)>& mutate = nullptr) {
    vjtest::GgufWriter w;
    meta(w);
    for (const auto& t : tensors) {
        if (skip && skip(t.name)) continue;
        std::string name = t.name;
        std::vector<uint64_t> ne = t.shapes.front();
        if (mutate) mutate(name, ne);
        uint64_t n = 1;
        for (auto e : ne) n *= e;
        uint32_t type = 0;
        std::vector<uint8_t> data;
        if (name == "engram_embd.weight") {  // Q4_0 rows
            type = 2;
            data.resize(n / 32 * 18);
        } else if (name == "per_layer_token_embd.weight") {  // F16 rows
            type = 1;
            data.resize(n * 2);
        } else {
            data.resize(n * 4);
        }
        if (type) for (size_t i = 0; i < data.size(); ++i) data[i] = pat(name, i);
        w.tensor(name, ne, type, data);
    }
    return vjtest::write_temp(w.build(), tag);
}

// Pass 1: metadata + embeddings only -> describe; pass 2: the full tensor map from that description.
Description roundtrip(const Meta& meta, const std::string& tag, std::string& path) {
    const std::vector<TensorSpec> embed = {{"token_embd.weight", {{64, 128}}, true}, {"output.weight", {{64, 128}}, true}};
    const std::string p1 = write(meta, embed, tag + "_meta");
    GgufFile f1;
    VJ_CHECK(f1.open(p1));
    Description d;
    std::string err;
    VJ_CHECK(describe(f1, d, &err));
    std::remove(p1.c_str());
    path = write(meta, d.tensors, tag);
    GgufFile f2;
    VJ_CHECK(f2.open(path));
    Description d2;
    VJ_CHECK(describe(f2, d2, &err));
    const TensorCheck c = check_tensors(f2, d2);
    if (!c.ok() || c.unexpected) for (const auto& p : c.problems) std::printf("    %s\n", p.c_str());
    VJ_CHECK(c.ok() && c.unexpected == 0 && c.missing_optional == 0 && c.present == c.checked);
    return d2;
}

void common_meta(vjtest::GgufWriter& w, const char* arch, uint32_t blocks) {
    w.kv_str("general.architecture", arch);
    w.kv_u32(std::string(arch) + ".embedding_length", 64);
    w.kv_u32(std::string(arch) + ".block_count", blocks);
    w.kv_u32(std::string(arch) + ".attention.head_count", 4);
}

void flash_next_meta(vjtest::GgufWriter& w, bool pattern) {
    const std::string a = "qwen3next.";
    common_meta(w, "qwen3next", 9);  // 8 main layers + 1 MTP layer
    w.kv_u32(a + "nextn_predict_layers", 1);
    w.kv_u32(a + "attention.head_count_kv", 2);
    w.kv_u32(a + "attention.key_length", 16);
    w.kv_u32(a + "ssm.group_count", 2);
    w.kv_u32(a + "ssm.time_step_rank", 4);
    w.kv_u32(a + "ssm.state_size", 16);
    w.kv_u32(a + "ssm.inner_size", 64);
    w.kv_u32(a + "ssm.conv_kernel", 4);
    if (pattern) w.kv_u32(a + "full_attention_interval", 4);
    w.kv_u32(a + "expert_count", 16);
    w.kv_u32(a + "expert_used_count", 4);
    w.kv_u32(a + "expert_feed_forward_length", 32);
    w.kv_u32(a + "expert_shared_feed_forward_length", 32);
    w.kv_u32(a + "embedding_length_per_layer_input", 8);
}

void hy4_meta(vjtest::GgufWriter& w, bool pattern, bool engram_dim = true) {
    const std::string a = "hy4.";
    common_meta(w, "hy4", 42);
    w.kv_u32(a + "attention.head_count_kv", 2);
    w.kv_u32(a + "attention.q_lora_rank", 32);
    w.kv_u32(a + "attention.kv_lora_rank", 16);
    w.kv_u32(a + "rope.dimension_count", 8);
    w.kv_u32(a + "attention.key_length", 24);   // nope 16 + rope 8
    w.kv_u32(a + "attention.value_length", 16);
    w.kv_u32(a + "attention.key_length_swa", 16);
    w.kv_u32(a + "attention.sliding_window", 4096);
    w.kv_u32(a + "expert_count", 512);
    w.kv_u32(a + "expert_used_count", 16);
    w.kv_u32(a + "expert_feed_forward_length", 4);
    w.kv_u32(a + "moe_latent_size", 8);
    w.kv_u32(a + "expert_shared_feed_forward_length", 32);
    w.kv_u32(a + "leading_dense_block_count", 1);
    w.kv_u32(a + "feed_forward_length", 96);
    w.kv_u32(a + "engram.rows", 1000);
    if (engram_dim) w.kv_u32(a + "engram.dim", 32);
    w.kv_u32(a + "engram.ngram", 3);
    if (pattern) {
        std::vector<int32_t> swa(42);
        for (int L = 0; L < 42; ++L) swa[L] = (L % 7 == 0 || L % 7 == 3) ? 0 : 1;  // 12 MLA, 30 SWA
        w.kv_arr_i32(a + "attention.sliding_window_pattern", swa);
    }
}

void dense_meta(vjtest::GgufWriter& w) {
    const std::string a = "qwen3.";
    common_meta(w, "qwen3", 6);
    w.kv_u32(a + "attention.head_count_kv", 2);
    w.kv_u32(a + "attention.key_length", 16);
    w.kv_u32(a + "feed_forward_length", 160);
}

void plan_ok(const ops::ModelDesc& m) {
    for (uint32_t gpus : {1u, 3u}) {
        ops::TpDesc tp;
        for (uint32_t g = 0; g < gpus; ++g) { tp.bw_gbs.push_back(g == 0 && gpus == 3 ? 820.0 : 880.0); tp.n_cu.push_back(60); }
        ops::StepPlan sp;
        std::string err;
        VJ_CHECK(ops::build_step_plan(m, tp, {1, false, 4096}, sp, &err));
        VJ_CHECK(!sp.ops.empty() && (gpus == 1) == (sp.count(ops::OpKind::AllReduce) == 0));
    }
}

void stream_table(const std::string& path, const GgufFile& f, const Description& d, const char* purpose) {
    const TableSpec* t = nullptr;
    for (const auto& x : d.tables) if (x.purpose == purpose) t = &x;
    VJ_CHECK(t != nullptr);
    const int fd = ::open(path.c_str(), O_RDONLY);
    VJ_CHECK(fd >= 0);
    prefetch::TableDesc td;
    std::string err;
    VJ_CHECK(table_desc(f, *t, fd, td, &err));
    VJ_CHECK(td.n_rows == t->rows && td.row_elems == t->row_elems);
    prefetch::RowPrefetcher p(td, prefetch::PrefetchConfig{64, 16}, std::make_unique<prefetch::PreadvBackend>(2));
    VJ_CHECK(p.ok());
    const uint32_t rb = td.row_bytes();
    const uint32_t rows[] = {0, 1, 2, 3, static_cast<uint32_t>(td.n_rows - 1)};
    p.prefetch(rows, 5);
    std::vector<uint8_t> buf(rb);
    for (uint32_t r : rows) {
        VJ_CHECK(p.fetch(r, buf.data(), &err));
        for (uint32_t i = 0; i < rb; ++i) VJ_CHECK(buf[i] == pat(t->tensor, size_t(r) * rb + i));
    }
    std::vector<float> fl(td.row_elems);
    VJ_CHECK(p.fetch_f32(1, fl.data(), &err));
    ::close(fd);
}

} // namespace

int main() {
    VJ_CHECK(blueprints().size() == 3 && find_by_name("hy4-flash") && !find_by_name("nope"));
    std::string err;
    // ---- qwen3.8-flash-next
    {
        std::string path;
        const Description d = roundtrip([](vjtest::GgufWriter& w) { flash_next_meta(w, true); }, "bp_flashnext", path);
        VJ_CHECK(d.blueprint == "qwen3.8-flash-next" && d.dims.n_layers == 8 && d.dims.nextn_layers == 1);
        VJ_CHECK(d.count(Mixer::Gdn) == 6 && d.count(Mixer::Gqa) == 2 && d.mixer[3] == Mixer::Gqa && d.count(Ffn::Moe) == 8);
        VJ_CHECK(d.dims.gdn_v_dim == 16 && d.tables.size() == 1 && d.tables[0].row_elems == 8 * 8);
        GgufFile f;
        VJ_CHECK(f.open(path));
        ops::ModelDesc m;
        std::vector<std::string> notes;
        VJ_CHECK(lower(d, m, notes, &err) && m.n_layers() == 8 && m.moe.shared_ff == 32);
        VJ_CHECK(notes.size() == 1);  // only the host-resident PLE table
        plan_ok(m);
        stream_table(path, f, d, "ple");
        // Mutations: a required GDN tensor missing, a reshaped expert tensor, an unknown extra tensor.
        const std::string p2 = write([](vjtest::GgufWriter& w) { flash_next_meta(w, true); }, d.tensors, "bp_fn_bad",
                                     [](const std::string& n) { return n == "blk.0.attn_qkv.weight"; },
                                     [](std::string& n, std::vector<uint64_t>& ne) {
                                         if (n == "blk.5.ffn_up_exps.weight") ne[1] = 33;
                                         if (n == "blk.7.ffn_down_shexp.weight") n = "blk.7.ffn_down_shared.weight";
                                     });
        GgufFile g;
        VJ_CHECK(g.open(p2));
        const TensorCheck c = check_tensors(g, d);
        VJ_CHECK(!c.ok() && c.missing_required == 2 && c.bad_shape == 1 && c.unexpected == 1);
        std::printf("  flash-next mutations caught: %s; %s; %s\n", c.problems[0].c_str(), c.problems[1].c_str(), c.problems[2].c_str());
        std::remove(p2.c_str());
        std::remove(path.c_str());
        // No pattern key -> error, never a guess.
        const std::string p3 = write([](vjtest::GgufWriter& w) { flash_next_meta(w, false); },
                                     {{"token_embd.weight", {{64, 128}}, true}}, "bp_fn_nopat");
        GgufFile h;
        VJ_CHECK(h.open(p3));
        Description dx;
        VJ_CHECK(!describe(h, dx, &err) && err.find("full_attention_interval") != std::string::npos);
        std::remove(p3.c_str());
    }
    // ---- hy4-flash
    {
        std::string path;
        const Description d = roundtrip([](vjtest::GgufWriter& w) { hy4_meta(w, true); }, "bp_hy4", path);
        VJ_CHECK(d.blueprint == "hy4-flash" && d.dims.n_layers == 42);
        VJ_CHECK(d.count(Mixer::Mla) == 12 && d.count(Mixer::Swa) == 30);
        VJ_CHECK(d.dims.n_experts == 512 && d.dims.top_k == 16 && d.count(Ffn::LatentMoe) == 41 && d.ffn[0] == Ffn::Dense);
        VJ_CHECK(d.dims.qk_nope == 16 && d.dims.engram_rows == 1000 && d.tables.size() == 1);
        GgufFile f;
        VJ_CHECK(f.open(path));
        ops::ModelDesc m;
        std::vector<std::string> notes;
        VJ_CHECK(lower(d, m, notes, &err));
        VJ_CHECK(notes.size() == 4);  // SWA window, MLA geometry, LatentMoE width, Engram table
        for (const auto& n : notes) std::printf("  hy4-flash lowering note: %s\n", n.c_str());
        plan_ok(m);
        stream_table(path, f, d, "engram");
        std::remove(path.c_str());
        for (int variant = 0; variant < 2; ++variant) {
            const std::string p = write([variant](vjtest::GgufWriter& w) { hy4_meta(w, variant == 1, variant == 0); },
                                        {{"token_embd.weight", {{64, 128}}, true}, {"output.weight", {{64, 128}}, true}}, "bp_hy4_bad");
            GgufFile g;
            VJ_CHECK(g.open(p));
            Description dx;
            VJ_CHECK(!describe(g, dx, &err));
            VJ_CHECK(err.find(variant == 0 ? "sliding_window_pattern" : "engram") != std::string::npos);  // 0: no pattern, 1: no engram.dim
            std::remove(p.c_str());
        }
    }
    // ---- qwen3.8-27b dense
    {
        std::string path;
        const Description d = roundtrip(dense_meta, "bp_dense", path);
        VJ_CHECK(d.blueprint == "qwen3.8-27b" && d.count(Mixer::Gqa) == 6 && d.count(Ffn::Dense) == 6 && d.tables.empty());
        ops::ModelDesc m;
        std::vector<std::string> notes;
        VJ_CHECK(lower(d, m, notes, &err) && notes.empty() && m.dense_ff == 160);  // lossless
        plan_ok(m);
        GgufFile f;
        VJ_CHECK(f.open(path));
        VJ_CHECK(find_for(f) == find_by_name("qwen3.8-27b"));
        Description forced;  // a blueprint can be forced by name regardless of general.architecture
        VJ_CHECK(!describe(f, forced, &err, find_by_name("hy4-flash")));
        std::remove(path.c_str());
    }
    {
        vjtest::GgufWriter w;
        w.kv_str("general.architecture", "unknownarch");
        w.tensor("token_embd.weight", {64, 128}, 0, std::vector<uint8_t>(64 * 128 * 4));
        const std::string p = vjtest::write_temp(w.build(), "bp_unknown");
        GgufFile f;
        VJ_CHECK(f.open(p));
        Description d;
        VJ_CHECK(!describe(f, d, &err) && err.find("no blueprint") != std::string::npos);
        std::remove(p.c_str());
    }
    std::printf("test_model_blueprints: flash-next / hy4-flash / 27b descriptors, tensor maps, mutations, lowering, "
                "planner, PLE + Engram row streaming OK\nPASS\n");
    return 0;
}
