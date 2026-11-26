// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// P-B4: the tokenizer loader reserved file-controlled array counts, so a 40-byte GGUF threw
// std::length_error out of load_from_gguf() and aborted the process.
#include "core/tokenizer.hpp"
#include "tests/test_common.hpp"

#include <unistd.h>

using namespace vegajet;

namespace {

void expect_reject(uint32_t elem_type, const char* tag, const char* needle) {
    vjtest::RawGguf g(0, 0);
    g.kv_arr_scalars("tokenizer.ggml.tokens", elem_type, 1ull << 60);  // no payload follows
    const std::string path = vjtest::write_temp(g.build(0), tag);
    BpeTokenizer tok;
    std::string err;
    const bool ok = tok.load_from_gguf(path, &err);
    unlink(path.c_str());
    std::printf("  %s -> %s\n", tag, err.c_str());
    VJ_CHECK(!ok);
    VJ_CHECK(err.find(needle) != std::string::npos);
}

} // namespace

int main() {
    expect_reject(8, "tok_reserve", "implausible tokenizer.ggml.tokens count");
    expect_reject(5, "tok_reserve_i32", "implausible tokenizer.ggml.tokens count");

    {   // A vocabulary that the file can actually back up still loads.
        vjtest::RawGguf g(0, 0);
        g.kv_str("tokenizer.ggml.model", "gpt2");
        g.kv_arr_str("tokenizer.ggml.tokens", {"a", "b", "c"});
        const std::string path = vjtest::write_temp(g.build(0), "tok_reserve_ok");
        BpeTokenizer tok;
        std::string err;
        const bool ok = tok.load_from_gguf(path, &err);
        unlink(path.c_str());
        VJ_CHECK(ok);
        VJ_CHECK(tok.vocab_size() == 3);
    }

    std::printf("[PASS] test_parser_tok_count: oversized tokenizer counts rejected with an error\n");
    return 0;
}
