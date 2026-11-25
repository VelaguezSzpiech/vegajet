// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/tokenizer.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

#define VJ_CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "CHECK FAILED at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        std::abort(); \
    } \
} while (0)

namespace {

template <typename T>
void write_pod(std::vector<uint8_t>& buf, T v) {
    const auto* p = reinterpret_cast<const uint8_t*>(&v);
    buf.insert(buf.end(), p, p + sizeof(T));
}

void write_str(std::vector<uint8_t>& buf, const std::string& s) {
    write_pod<uint64_t>(buf, static_cast<uint64_t>(s.size()));
    buf.insert(buf.end(), s.begin(), s.end());
}

void write_kv_str(std::vector<uint8_t>& buf, const std::string& k, const std::string& v) {
    write_str(buf, k);
    write_pod<uint32_t>(buf, 8u); // GGUF_TYPE_STRING
    write_str(buf, v);
}

void write_kv_u32(std::vector<uint8_t>& buf, const std::string& k, uint32_t v) {
    write_str(buf, k);
    write_pod<uint32_t>(buf, 4u); // GGUF_TYPE_UINT32
    write_pod<uint32_t>(buf, v);
}

void write_kv_str_array(std::vector<uint8_t>& buf, const std::string& k, const std::vector<std::string>& arr) {
    write_str(buf, k);
    write_pod<uint32_t>(buf, 9u); // GGUF_TYPE_ARRAY
    write_pod<uint32_t>(buf, 8u); // STRING
    write_pod<uint64_t>(buf, static_cast<uint64_t>(arr.size()));
    for (const auto& s : arr) write_str(buf, s);
}

void write_kv_i32_array(std::vector<uint8_t>& buf, const std::string& k, const std::vector<int32_t>& arr) {
    write_str(buf, k);
    write_pod<uint32_t>(buf, 9u); // GGUF_TYPE_ARRAY
    write_pod<uint32_t>(buf, 5u); // INT32
    write_pod<uint64_t>(buf, static_cast<uint64_t>(arr.size()));
    for (int32_t v : arr) write_pod<int32_t>(buf, v);
}

void test_synthetic_gguf_bpe() {
    // Build a real GGUF v3 file on disk in /tmp and parse via read-only mmap
    const std::string g_sp = "\xC4\xA0"; // U+0120 'Ġ' (space)
    const std::string g_nl = "\xC4\x8A"; // U+010A 'Ċ' (newline)

    std::vector<std::string> tokens = {
        "<|bos|>", "<|eos|>", "<|im_start|>", "<|im_end|>",
        "h", "e", "l", "o", "w", "r", "d", "!", g_sp, g_nl,
        "he", "ll", "lo", "hello",
        g_sp + "w", g_sp + "wo", g_sp + "wor", g_sp + "worl", g_sp + "world"
    };
    std::vector<int32_t> ttypes(tokens.size(), 1);
    ttypes[0] = 3; ttypes[1] = 3; ttypes[2] = 3; ttypes[3] = 3;

    std::vector<std::string> merges = {
        "h e",
        "l o",
        "he l", // unused intermediate check
        "l l",
        "he ll",
        "he llo",
        "he l",
        "l lo",
        g_sp + " w",
        g_sp + "w o",
        g_sp + "wo r",
        g_sp + "wor l",
        g_sp + "worl d"
    };
    // Make clean deterministic merge chain for "hello" & " world"
    merges = {
        "h e",
        "l l",
        "he ll", // no token -> test valid merges directly:
    };
    tokens.push_back("hell");
    ttypes.push_back(1);
    merges = {
        "h e",
        "l l",
        "he ll",
        "hell o",
        g_sp + " w",
        g_sp + "w o",
        g_sp + "wo r",
        g_sp + "wor l",
        g_sp + "worl d"
    };

    std::vector<uint8_t> gguf;
    gguf.insert(gguf.end(), {'G', 'G', 'U', 'F'});
    write_pod<uint32_t>(gguf, 3u); // version 3
    write_pod<uint64_t>(gguf, 0ULL); // n_tensors
    write_pod<uint64_t>(gguf, 6ULL); // n_kv
    write_kv_str(gguf, "tokenizer.ggml.model", "gpt2");
    write_kv_str_array(gguf, "tokenizer.ggml.tokens", tokens);
    write_kv_str_array(gguf, "tokenizer.ggml.merges", merges);
    write_kv_i32_array(gguf, "tokenizer.ggml.token_type", ttypes);
    write_kv_u32(gguf, "tokenizer.ggml.bos_token_id", 0u);
    write_kv_u32(gguf, "tokenizer.ggml.eos_token_id", 1u);

    const std::string path = "/tmp/vegajet_test_tok_" + std::to_string(getpid()) + ".bin";
    {
        std::ofstream ofs(path, std::ios::binary);
        VJ_CHECK(ofs.is_open());
        ofs.write(reinterpret_cast<const char*>(gguf.data()), static_cast<std::streamsize>(gguf.size()));
    }

    vegajet::BpeTokenizer tok;
    std::string err;
    bool ok = tok.load_from_gguf(path, &err);
    unlink(path.c_str());
    if (!ok) {
        std::fprintf(stderr, "FAIL load_from_gguf: %s\n", err.c_str());
        std::abort();
    }

    VJ_CHECK(tok.vocab_size() == tokens.size());
    VJ_CHECK(tok.bos_id() == 0 && tok.eos_id() == 1);

    // 1. BPE merge + byte-level round-trip
    std::string s1 = "hello world!\n";
    auto ids1 = tok.encode(s1, false);
    // Expected: "hello"(17), " world"(22), "!"(11), "\n"(13)
    VJ_CHECK(ids1.size() == 4);
    VJ_CHECK(ids1[0] == 17 && ids1[1] == 22 && ids1[2] == 11 && ids1[3] == 13);
    VJ_CHECK(tok.decode(ids1) == s1);

    // 2. Special token preservation + skip_special_tokens
    std::string s2 = "<|im_start|>hello world<|im_end|>";
    auto ids2 = tok.encode(s2, true);
    VJ_CHECK(ids2.size() == 5);
    VJ_CHECK(ids2[0] == 0 && ids2[1] == 2 && ids2[2] == 17 && ids2[3] == 22 && ids2[4] == 3);
    VJ_CHECK(tok.decode(ids2, false) == "<|bos|><|im_start|>hello world<|im_end|>");
    VJ_CHECK(tok.decode(ids2, true) == "hello world");

    std::printf("[PASS] Synthetic GGUF v3 mmap BPE encode/decode/special-token test PASSED!\n");
}

void test_real_qwen_gguf_header_if_present() {
    const char* env = std::getenv("VEGAJET_QWEN_TOKENIZER_GGUF");
    const std::string qwen_gguf = env ? env : "";
    if (qwen_gguf.empty() || access(qwen_gguf.c_str(), R_OK) != 0) {
        std::printf("[SKIP] VEGAJET_QWEN_TOKENIZER_GGUF unset or unreadable; skipping optional real-vocab parity test.\n");
        return;
    }
    vegajet::BpeTokenizer tok;
    std::string err;
    bool ok = tok.load_from_gguf(qwen_gguf, &err);
    if (!ok) {
        std::fprintf(stderr, "FAIL real Qwen GGUF load: %s\n", err.c_str());
        std::abort();
    }
    VJ_CHECK(tok.vocab_size() > 200000);
    const std::string prompt = "The capital of France is";
    auto ids = tok.encode(prompt, false);
    std::string rt = tok.decode(ids, false);
    VJ_CHECK(rt == prompt);
    // Golden Qwen3.8 token IDs for "The capital of France is": {760, 6511, 314, 9338, 369}
    VJ_CHECK(ids.size() == 5);
    VJ_CHECK(ids[0] == 760 && ids[1] == 6511 && ids[2] == 314 && ids[3] == 9338 && ids[4] == 369);
    std::printf("[PASS] Real Qwen3.8 GGUF header (vocab=%zu) exact token-ID parity {760,6511,314,9338,369} PASSED!\n",
                tok.vocab_size());
}

} // namespace

int main() {
    test_synthetic_gguf_bpe();
    test_real_qwen_gguf_header_if_present();
    return 0;
}
