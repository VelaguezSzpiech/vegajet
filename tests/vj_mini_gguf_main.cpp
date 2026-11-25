// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Test fixture: writes a miniature qwen35moe ("Ornith") GGUF that the Ornith plugin plans in dry-run mode, with a
// byte-level GPT-2 tokenizer (256 byte symbols, ChatML specials, filler up to the 512-row embedding) and a
// Qwen3-style chat template. Used by the CLI end-to-end tests: `vegajet plan`, `vegajet bench --json-out`,
// `vegajet serve <model>` (which must fail cleanly with exit 3 on a CPU-only build) and `vegajet inspect`.
//   vj_mini_gguf <out-file>
// Tensor shapes follow tests/test_ornith_loader.cpp (build_mini_ornith); payloads are zero (dry-run never reads them).
#include "tests/test_common.hpp"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

// GPT-2 byte -> unicode symbol (UTF-8), as in the byte-level BPE vocabularies.
std::vector<std::string> byte_symbols() {
    std::vector<std::string> sym(256);
    uint32_t extra = 0;
    for (uint32_t b = 0; b < 256; ++b) {
        const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174);
        const uint32_t cp = printable ? b : 256 + extra++;
        std::string s;
        if (cp < 0x80) s += static_cast<char>(cp);
        else { s += static_cast<char>(0xC0 | (cp >> 6)); s += static_cast<char>(0x80 | (cp & 0x3F)); }
        sym[b] = s;
    }
    return sym;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: %s <out-file>\n", argv[0]); return 2; }
    vjtest::GgufWriter w;
    const std::string a = "qwen35moe";
    w.kv_str("general.architecture", a);
    w.kv_str("general.name", "mini-ornith-fixture");
    w.kv_u32(a + ".block_count", 5);
    w.kv_u32(a + ".context_length", 4096);
    w.kv_u32(a + ".embedding_length", 256);
    w.kv_u32(a + ".attention.head_count", 4);
    w.kv_u32(a + ".attention.head_count_kv", 2);
    w.kv_u32(a + ".attention.key_length", 64);
    w.kv_u32(a + ".attention.value_length", 64);
    w.kv_f32(a + ".rope.freq_base", 1e7f);
    w.kv_u32(a + ".rope.dimension_count", 32);
    w.kv_u32(a + ".expert_count", 8);
    w.kv_u32(a + ".expert_used_count", 2);
    w.kv_u32(a + ".expert_feed_forward_length", 64);
    w.kv_u32(a + ".expert_shared_feed_forward_length", 64);
    w.kv_u32(a + ".nextn_predict_layers", 1);
    w.kv_u32(a + ".ssm.conv_kernel", 4);
    w.kv_u32(a + ".ssm.state_size", 32);
    w.kv_u32(a + ".ssm.group_count", 2);
    w.kv_u32(a + ".ssm.time_step_rank", 4);
    w.kv_u32(a + ".ssm.inner_size", 128);
    w.kv_arr_i32(a + ".attention.recurrent_layers", {1, 1, 1, 0, 0});
    w.kv_u32(a + ".full_attention_interval", 4);
    // Tokenizer: 256 byte symbols, 3 ChatML specials, filler to 512 rows.
    std::vector<std::string> toks = byte_symbols();
    std::vector<int32_t> types(256, 6);
    for (const char* sp : {"<|im_start|>", "<|im_end|>", "<|endoftext|>"}) { toks.push_back(sp); types.push_back(3); }
    while (toks.size() < 512) { toks.push_back("<filler_" + std::to_string(toks.size()) + ">"); types.push_back(5); }
    w.kv_str("tokenizer.ggml.model", "gpt2");
    w.kv_str("tokenizer.ggml.pre", "qwen2");
    w.kv_arr_str("tokenizer.ggml.tokens", toks);
    w.kv_arr_i32("tokenizer.ggml.token_type", types);
    w.kv_arr_str("tokenizer.ggml.merges", {});
    w.kv_u32("tokenizer.ggml.eos_token_id", 257);  // <|im_end|>
    w.kv_str("tokenizer.chat_template", "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
                                         "{% endfor %}<|im_start|>assistant\n<think>\n\n</think>\n\n");
    auto mat = [&](const std::string& n, uint64_t in, uint64_t out, uint64_t ex = 1) {
        w.tensor(n, ex > 1 ? std::vector<uint64_t>{in, out, ex} : std::vector<uint64_t>{in, out}, 1,
                 std::vector<uint8_t>(in * out * ex * 2, 0));
    };
    auto vec = [&](const std::string& n, std::vector<uint64_t> ne) {
        uint64_t e = 1;
        for (auto x : ne) e *= x;
        w.tensor(n, ne, 0, std::vector<uint8_t>(e * 4, 0));
    };
    w.tensor("token_embd.weight", {256, 512}, 8, std::vector<uint8_t>(512 * 256 / 32 * 34, 0x01));
    mat("output.weight", 256, 512);
    vec("output_norm.weight", {256});
    const bool rec[5] = {true, true, true, false, false};
    for (int il = 0; il < 5; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        vec(p + "attn_norm.weight", {256});
        vec(p + "post_attention_norm.weight", {256});
        vec(p + "ffn_gate_inp.weight", {256, 8});
        vec(p + "ffn_gate_inp_shexp.weight", {256});
        mat(p + "ffn_gate_exps.weight", 256, 64, 8);
        mat(p + "ffn_up_exps.weight", 256, 64, 8);
        mat(p + "ffn_down_exps.weight", 64, 256, 8);
        mat(p + "ffn_gate_shexp.weight", 256, 64);
        mat(p + "ffn_up_shexp.weight", 256, 64);
        mat(p + "ffn_down_shexp.weight", 64, 256);
        if (rec[il]) {
            mat(p + "attn_qkv.weight", 256, 256);
            mat(p + "attn_gate.weight", 256, 128);
            mat(p + "ssm_alpha.weight", 256, 32);
            mat(p + "ssm_beta.weight", 256, 32);
            mat(p + "ssm_out.weight", 128, 256);
            vec(p + "ssm_conv1d.weight", {4, 256});
            vec(p + "ssm_a", {4});
            vec(p + "ssm_dt.bias", {4});
            vec(p + "ssm_norm.weight", {32});
        } else {
            mat(p + "attn_q.weight", 256, 512);
            mat(p + "attn_k.weight", 256, 128);
            mat(p + "attn_v.weight", 256, 128);
            mat(p + "attn_output.weight", 256, 256);
            vec(p + "attn_q_norm.weight", {64});
            vec(p + "attn_k_norm.weight", {64});
        }
        if (il == 4) {
            mat(p + "nextn.eh_proj.weight", 512, 256);
            vec(p + "nextn.enorm.weight", {256});
            vec(p + "nextn.hnorm.weight", {256});
        }
    }
    const std::vector<uint8_t> bytes = w.build();
    std::ofstream f(argv[1], std::ios::binary);
    f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!f) { std::fprintf(stderr, "cannot write %s\n", argv[1]); return 1; }
    std::printf("%s\n", argv[1]);
    return 0;
}
