// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/gguf_reader.hpp"
#include "models/registry.hpp"
#include "tests/test_common.hpp"

#include <cmath>

using namespace vegajet;

int main() {
    vjtest::GgufWriter w;
    w.kv_str("general.architecture", "testarch");
    w.kv_u32("testarch.block_count", 3);
    w.kv_f32("testarch.rope.freq_base", 1e6f);
    w.kv_bool("testarch.flag", true);
    w.kv_u64("testarch.big", 1ull << 40);
    w.kv_i32("testarch.neg", -7);
    w.kv_arr_i32("testarch.pattern", {1, 0, 0, 1});
    std::vector<std::string> vocab;
    for (int i = 0; i < 1000; ++i) vocab.push_back("tok" + std::to_string(i));
    w.kv_arr_str("tokenizer.ggml.tokens", vocab);  // long string array: skipped, length kept

    std::vector<uint8_t> f32(4 * 8 * 2);
    for (size_t i = 0; i < f32.size() / 4; ++i) { float v = float(i) * 0.5f; std::memcpy(&f32[i * 4], &v, 4); }
    std::vector<uint8_t> q8(34 * 2, 0x11);  // two Q8_0 blocks
    w.tensor("a.weight", {8, 2}, 0, f32);
    w.tensor("b.weight", {64}, 8, q8);
    const std::string path = vjtest::write_temp(w.build(), "gguf");

    GgufFile g;
    VJ_CHECK(g.open(path));
    VJ_CHECK(g.version() == 3);
    VJ_CHECK(g.architecture() == "testarch");
    uint64_t u = 0; double d = 0;
    VJ_CHECK(g.get_arch_u64("block_count", u) && u == 3);
    VJ_CHECK(g.get_arch_f64("rope.freq_base", d) && std::fabs(d - 1e6) < 1);
    VJ_CHECK(g.get_arch_u64("flag", u) && u == 1);
    VJ_CHECK(g.get_arch_u64("big", u) && u == (1ull << 40));
    VJ_CHECK(g.get_arch_f64("neg", d) && d == -7.0);
    const GgufValue* pat = g.find("testarch.pattern");
    VJ_CHECK(pat && pat->is_array() && pat->arr_len == 4 && pat->arr_num[3] == 1.0);
    const GgufValue* tok = g.find("tokenizer.ggml.tokens");
    VJ_CHECK(tok && tok->arr_len == 1000 && tok->arr_str.empty() && tok->arr_truncated);

    VJ_CHECK(g.tensors().size() == 2);
    const GgufTensorInfo* a = g.tensor("a.weight");
    const GgufTensorInfo* b = g.tensor("b.weight");
    VJ_CHECK(a && b && a->n_elems() == 16 && a->n_bytes() == 64 && b->n_bytes() == 68);
    VJ_CHECK(g.data_offset() % 32 == 0);
    const uint8_t* pa = g.tensor_data(*a);
    VJ_CHECK(pa != nullptr);
    float v5; std::memcpy(&v5, pa + 5 * 4, 4);
    VJ_CHECK(v5 == 2.5f);
    VJ_CHECK(g.tensor_data(*b)[0] == 0x11);
    g.release(pa, a->n_bytes());
    std::memcpy(&v5, g.tensor_data(*a) + 5 * 4, 4);  // pages re-fault from the file after release
    VJ_CHECK(v5 == 2.5f);

    // Truncation and bad magic are rejected without crashing.
    auto bytes = w.build();
    bytes.resize(bytes.size() - 40);
    GgufFile t;
    VJ_CHECK(!t.open(vjtest::write_temp(bytes, "gguf_trunc")));
    VJ_CHECK(t.error().find("past end") != std::string::npos);
    bytes[0] = 'X';
    VJ_CHECK(!t.open(vjtest::write_temp(bytes, "gguf_magic")));

    // Registry refuses unknown architectures with a clear error.
    std::string err;
    VJ_CHECK(ModelRegistry::instance().create_for(g, &err) == nullptr);
    VJ_CHECK(err.find("testarch") != std::string::npos);

    std::printf("[PASS] test_gguf_reader: KV scalars/arrays, tensor offsets, lazy mmap + MADV_DONTNEED, "
                "truncation/magic rejection, registry dispatch\n");
    return 0;
}
