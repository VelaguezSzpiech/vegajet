// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// The declarative MODEL_TEMPLATE on synthetic GGUF headers: a dense GQA file and a hybrid GDN+GQA MoE file must
// describe(), dry-run load() on 1 and 2 GPUs, and emit decode / verify / prefill graphs with the expected AllReduce
// counts. No real model files are involved.
#include "declarative_model.hpp"
#include "tests/test_common.hpp"

#include <cstdio>

using namespace vegajet;
using namespace vegajet::declarative_template;

namespace {

std::vector<uint8_t> f32_bytes(size_t n) { return std::vector<uint8_t>(n * 4, 0); }

std::string write_gguf(bool hybrid_moe) {
    vjtest::GgufWriter w;
    w.kv_str("general.architecture", "template");
    w.kv_u32("template.embedding_length", 256);
    w.kv_u32("template.block_count", 8);
    w.kv_u32("template.attention.head_count", 8);
    w.kv_u32("template.attention.head_count_kv", 2);
    w.kv_u32("template.attention.key_length", 32);
    if (hybrid_moe) {
        w.kv_u32("template.expert_count", 16);
        w.kv_u32("template.expert_used_count", 4);
        w.kv_u32("template.expert_feed_forward_length", 128);
        w.kv_u32("template.expert_shared_feed_forward_length", 128);
        w.kv_u32("template.ssm.group_count", 4);
        w.kv_u32("template.ssm.time_step_rank", 8);
        w.kv_u32("template.ssm.state_size", 32);
        w.kv_u32("template.ssm.inner_size", 256);
        w.kv_u32("template.ssm.conv_kernel", 4);
        w.kv_u32("template.full_attention_interval", 4);
    } else {
        w.kv_u32("template.feed_forward_length", 704);
    }
    w.tensor("token_embd.weight", {256, 100}, 0, f32_bytes(256 * 100));
    w.tensor("output.weight", {256, 100}, 0, f32_bytes(256 * 100));
    return vjtest::write_temp(w.build(), hybrid_moe ? "tmpl_moe" : "tmpl_dense");
}

} // namespace

int main() {
    for (bool hybrid : {false, true}) {
        const std::string path = write_gguf(hybrid);
        GgufFile f;
        VJ_CHECK(f.open(path));
        DeclarativeModel model;
        VJ_CHECK(model.can_load(f));
        ops::ModelDesc d;
        std::string err;
        VJ_CHECK(DeclarativeModel::describe(f, d, &err));
        VJ_CHECK(d.d_model == 256 && d.vocab == 100 && d.n_layers() == 8 && d.attn.head_dim == 32);
        uint32_t gdn = 0;
        for (auto k : d.mixer) gdn += k == ops::MixerKind::Gdn;
        VJ_CHECK(gdn == (hybrid ? 6u : 0u));
        VJ_CHECK(d.fmt_head == quant::NativeFormat::VJ_Q8_DP4A);  // F32 output -> Q8
        for (uint32_t n_gpu : {1u, 2u}) {
            LoadOptions lo;
            lo.gpus.assign(n_gpu, GpuTarget{});
            lo.gpus[0].bw_hbm_gbs = 840.0;
            lo.context_length = 4096;
            const LoadReport rep = model.load(f, lo);
            VJ_CHECK(rep.ok && rep.error.empty() && !rep.notes.empty());
            const auto pre = model.prefill(std::vector<int32_t>(64, 1));
            const auto dec = model.decode_step(1, 1);
            const auto ver = model.decode_step(1, 8);
            VJ_CHECK(!pre.graph.empty() && !dec.graph.empty() && !ver.graph.empty() && model.draft_step({}, 4).graph.empty());
            uint32_t ar = 0;
            for (const auto& op : dec.graph) ar += op.kind == ops::OpKind::AllReduce;
            VJ_CHECK(ar == (n_gpu == 1 ? 0u : 2u * d.n_layers()));
            for (const auto& op : pre.graph) VJ_CHECK(op.kind != ops::OpKind::FlashDecodeWave64 && op.m == 64);
            for (const auto& op : ver.graph) VJ_CHECK(op.m == 8);
            std::printf("  %s, %u GPU: %s\n", hybrid ? "hybrid GDN+GQA MoE" : "dense GQA", n_gpu, rep.notes.front().c_str());
        }
        f.close();
        std::remove(path.c_str());
    }
    std::printf("test_declarative_template: describe + dry-run load + graphs OK\nPASS\n");
    return 0;
}
