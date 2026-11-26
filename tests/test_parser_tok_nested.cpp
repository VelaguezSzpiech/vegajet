// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// P-B5: an unknown metadata key whose value is an ARRAY of ARRAY of ... recursed once per nesting
// level in skip_val(), so ~400 000 levels (4.8 MB) overflowed the stack. Depth is now capped.
#include "core/tokenizer.hpp"
#include "tests/test_common.hpp"

#include <unistd.h>

using namespace vegajet;

namespace {

// One KV of type ARRAY<ARRAY<...<UINT32>>> with `levels` nested array headers and no data.
std::vector<uint8_t> nested_array_file(uint64_t levels) {
    vjtest::RawGguf g(0, 0);
    g.kv_raw_value("general.unknown_nested", 9);  // value type: ARRAY
    for (uint64_t i = 0; i < levels; ++i) {
        g.u32(9);    // element type of this array: ARRAY again
        g.u64(1);    // exactly one element
    }
    g.u32(4);        // innermost element type: UINT32
    g.u64(0);        // empty
    return g.build(0);
}

} // namespace

int main() {
    {   // 400 000 levels: one stack frame per level before the fix.
        const std::string path = vjtest::write_temp(nested_array_file(400000), "tok_nested");
        BpeTokenizer tok;
        std::string err;
        const bool ok = tok.load_from_gguf(path, &err);
        unlink(path.c_str());
        std::printf("  tok_nested -> %s\n", err.c_str());
        VJ_CHECK(!ok);
        VJ_CHECK(err.find("nested array") != std::string::npos);
    }
    {   // Two levels of nesting are still skipped without complaint (the parser just ignores the key).
        vjtest::RawGguf g(0, 0);
        g.kv_str("tokenizer.ggml.model", "gpt2");
        g.kv_raw_value("general.two_levels", 9);  // ARRAY
        g.u32(9);        // of ARRAY
        g.u64(1);        // 1 element
        g.u32(4);        // that array holds UINT32
        g.u64(2);        // 2 elements
        g.u32(7);
        g.u32(8);
        g.kv_arr_str("tokenizer.ggml.tokens", {"a", "b"});
        const std::string path = vjtest::write_temp(g.build(0), "tok_nested_ok");
        BpeTokenizer tok;
        std::string err;
        const bool ok = tok.load_from_gguf(path, &err);
        unlink(path.c_str());
        VJ_CHECK(ok);
        VJ_CHECK(tok.vocab_size() == 2);
    }

    std::printf("[PASS] test_parser_tok_nested: nested GGUF array values depth-limited, no stack overflow\n");
    return 0;
}
