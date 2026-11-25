// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// core/bpe_tokenizer: pre-tokenizer cases derived by hand from the published gpt2 / llama3 / qwen2 patterns, BPE
// merges vs a naive O(n^2) reference on a vocabulary trained here, byte-exact round trips on random UTF-8 and raw
// bytes, atomic special tokens, GGUF metadata loading (synthetic file, truncation, garbage), agreement with HEAD's
// BpeTokenizer on ASCII text, chat templates, and the IEngine adapter wired to the template + tokenizer.
#include "core/bpe_tokenizer.hpp"
#include "core/server_engine_adapter.hpp"
#include "tests/test_common.hpp"

#include <algorithm>
#include <cstdio>
#include <map>
#include <random>

using namespace vegajet;
using namespace vegajet::bpe;

namespace {

using V = std::vector<std::string>;

std::string u8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) s += char(cp);
    else if (cp < 0x800) { s += char(0xC0 | (cp >> 6)); s += char(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { s += char(0xE0 | (cp >> 12)); s += char(0x80 | ((cp >> 6) & 0x3F)); s += char(0x80 | (cp & 0x3F)); }
    else { s += char(0xF0 | (cp >> 18)); s += char(0x80 | ((cp >> 12) & 0x3F)); s += char(0x80 | ((cp >> 6) & 0x3F)); s += char(0x80 | (cp & 0x3F)); }
    return s;
}

// Independent bytes_to_unicode.
std::vector<std::string> byte_table() {
    std::vector<std::string> t(256);
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        const bool direct = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        t[b] = u8(direct ? b : 256 + n++);
    }
    return t;
}

// Tiny BPE trainer: most frequent adjacent pair (ties -> lexicographically smallest) over pre-tokenized words.
Vocab train(const std::string& corpus, int n_merges) {
    const auto bt = byte_table();
    Vocab v;
    v.pre = "qwen2";
    for (int b = 0; b < 256; ++b) v.tokens.push_back(bt[b]);
    std::map<std::vector<std::string>, int> words;
    for (const auto& w : Tokenizer::pretokenize(corpus, PreTok::Qwen2)) {
        std::vector<std::string> sym;
        for (unsigned char c : w) sym.push_back(bt[c]);
        words[sym]++;
    }
    for (int m = 0; m < n_merges; ++m) {
        std::map<std::pair<std::string, std::string>, int> cnt;
        for (const auto& [w, f] : words) for (size_t i = 0; i + 1 < w.size(); ++i) cnt[{w[i], w[i + 1]}] += f;
        if (cnt.empty()) break;
        auto best = cnt.begin();
        for (auto it = cnt.begin(); it != cnt.end(); ++it) if (it->second > best->second) best = it;
        const auto [a, b] = best->first;
        v.merges.push_back(a + " " + b);
        v.tokens.push_back(a + b);
        std::map<std::vector<std::string>, int> next;
        for (const auto& [w, f] : words) {
            std::vector<std::string> o;
            for (size_t i = 0; i < w.size(); ++i) {
                if (i + 1 < w.size() && w[i] == a && w[i + 1] == b) { o.push_back(a + b); ++i; }
                else o.push_back(w[i]);
            }
            next[o] += f;
        }
        words.swap(next);
    }
    v.types.assign(v.tokens.size(), 1);
    for (const char* sp : {"<|endoftext|>", "<|im_start|>", "<|im_end|>"}) { v.tokens.push_back(sp); v.types.push_back(3); }
    v.tokens.push_back("<think>"); v.types.push_back(4);
    v.tokens.push_back("</think>"); v.types.push_back(4);
    v.eos = static_cast<int32_t>(v.tokens.size()) - 3;  // <|im_end|>
    v.bos = static_cast<int32_t>(v.tokens.size()) - 5;  // <|endoftext|>
    return v;
}

// Naive reference BPE for one pre-token.
std::vector<std::string> naive_bpe(const std::string& w, const Vocab& v) {
    const auto bt = byte_table();
    std::map<std::string, int> rank;
    for (size_t i = 0; i < v.merges.size(); ++i) rank.emplace(v.merges[i], static_cast<int>(i));
    std::vector<std::string> s;
    for (unsigned char c : w) s.push_back(bt[c]);
    while (true) {
        int best = -1;
        size_t at = 0;
        for (size_t i = 0; i + 1 < s.size(); ++i) {
            const auto it = rank.find(s[i] + " " + s[i + 1]);
            if (it != rank.end() && (best < 0 || it->second < best)) { best = it->second; at = i; }
        }
        if (best < 0) break;
        s[at] += s[at + 1];
        s.erase(s.begin() + at + 1);
    }
    return s;
}

const char* kCorpus =
    "The quick brown fox jumps over the lazy dog. The dog sleeps; the fox runs away!\n"
    "def main(argc, argv):\n    for i in range(10):\n        print(i * 2)  # double\n    return 0\n\n"
    "I'm sure they're here, we've seen it. You'll see, he'd say. It's 2026 and 12345 + 678 = 13023.\n"
    "Grüße aus München! Привет, мир. 你好，世界。こんにちは。 🙂👍 café naïve résumé\n";

} // namespace

int main() {
    // ---- pre-tokenizer cases
    VJ_CHECK((Tokenizer::pretokenize("Hello world", PreTok::Qwen2) == V{"Hello", " world"}));
    VJ_CHECK((Tokenizer::pretokenize("I'm 12345", PreTok::Qwen2) == V{"I", "'m", " ", "1", "2", "3", "4", "5"}));
    VJ_CHECK((Tokenizer::pretokenize("I'M 12345", PreTok::Llama3) == V{"I", "'M", " ", "123", "45"}));
    VJ_CHECK((Tokenizer::pretokenize("I'M 12345", PreTok::Gpt2) == V{"I", "'", "M", " 12345"}));  // case-sensitive
    VJ_CHECK((Tokenizer::pretokenize("  \n\n  x", PreTok::Qwen2) == V{"  \n\n", " ", " x"}));
    VJ_CHECK((Tokenizer::pretokenize("Hello!!!\n\nWorld", PreTok::Qwen2) == V{"Hello", "!!!\n\n", "World"}));
    VJ_CHECK((Tokenizer::pretokenize("a  b", PreTok::Gpt2) == V{"a", " ", " b"}));
    VJ_CHECK((Tokenizer::pretokenize("abc123def", PreTok::Qwen2) == V{"abc", "1", "2", "3", "def"}));
    VJ_CHECK((Tokenizer::pretokenize("x = foo(bar);", PreTok::Qwen2) == V{"x", " =", " foo", "(bar", ");"}));
    VJ_CHECK((Tokenizer::pretokenize("\xE4\xBD\xA0\xE5\xA5\xBD\xEF\xBC\x8C\xE4\xB8\x96", PreTok::Qwen2) ==
              V{"\xE4\xBD\xA0\xE5\xA5\xBD", "\xEF\xBC\x8C\xE4\xB8\x96"}));  // 你好 + ，世 (punct + letter)
    VJ_CHECK((Tokenizer::pretokenize("ok \xF0\x9F\x99\x82\xF0\x9F\x91\x8D!", PreTok::Qwen2) == V{"ok", " \xF0\x9F\x99\x82\xF0\x9F\x91\x8D!"}));
    VJ_CHECK((Tokenizer::pretokenize("caf\xC3\xA9 na\xC3\xAFve", PreTok::Qwen2) == V{"caf\xC3\xA9", " na\xC3\xAFve"}));
    VJ_CHECK((Tokenizer::pretokenize("tab\there   ", PreTok::Qwen2) == V{"tab", "\there", "   "}));
    VJ_CHECK((Tokenizer::pretokenize("a\xFF\xFE b", PreTok::Qwen2) == V{"a", "\xFF\xFE", " b"}));  // invalid bytes kept

    // ---- trained vocabulary: merges equal the naive reference
    const Vocab v = train(kCorpus, 300);
    Tokenizer tok;
    std::string err;
    VJ_CHECK(tok.load(v, &err));
    std::mt19937 rng(9);
    for (const auto& w : Tokenizer::pretokenize(kCorpus, PreTok::Qwen2)) {
        const auto ref = naive_bpe(w, v);
        const auto ids = tok.encode(w, false, false);
        VJ_CHECK(ids.size() == ref.size());
        for (size_t i = 0; i < ids.size(); ++i) VJ_CHECK(v.tokens[ids[i]] == ref[i]);
    }
    for (int i = 0; i < 300; ++i) {  // random words built from corpus letters (unseen combinations)
        std::string w = " ";
        for (int k = 0; k < 1 + int(rng() % 12); ++k) w += "thequickbrownfox"[rng() % 16];
        const auto ref = naive_bpe(w, v);
        const auto ids = tok.encode(w, false, false);
        VJ_CHECK(ids.size() == ref.size());
        for (size_t k = 0; k < ids.size(); ++k) VJ_CHECK(v.tokens[ids[k]] == ref[k]);
    }
    // ---- round trips: random UTF-8 (all planes) and raw bytes
    for (int i = 0; i < 400; ++i) {
        std::string s;
        for (int k = 0; k < 40; ++k) {
            const uint32_t pick = rng() % 6;
            const uint32_t cp = pick == 0 ? 32 + rng() % 95 : pick == 1 ? 0x80 + rng() % 0x780 : pick == 2 ? 0x800 + rng() % 0xD000
                              : pick == 3 ? 0x1F300 + rng() % 0x600 : pick == 4 ? "\n\t "[rng() % 3] : 0x4E00 + rng() % 0x5000;
            if (cp >= 0xD800 && cp <= 0xDFFF) continue;
            s += u8(cp);
        }
        VJ_CHECK(tok.decode(tok.encode(s)) == s);
        std::string raw(1 + rng() % 64, '\0');
        for (auto& c : raw) c = static_cast<char>(rng());
        VJ_CHECK(tok.decode(tok.encode(raw, false, false)) == raw);
    }
    VJ_CHECK(tok.decode(tok.encode(kCorpus)) == kCorpus);
    // ---- special tokens: atomic when parsed, plain text otherwise; control tokens skippable
    {
        const std::string s = "<|im_start|>user\nhi there<|im_end|>\n<think>x</think>";
        const auto ids = tok.encode(s);
        VJ_CHECK(ids.front() == tok.token_id("<|im_start|>") && std::count(ids.begin(), ids.end(), tok.token_id("<|im_end|>")) == 1);
        VJ_CHECK(std::count(ids.begin(), ids.end(), tok.token_id("<think>")) == 1);
        VJ_CHECK(tok.decode(ids) == s);
        VJ_CHECK(tok.decode(ids, true) == "user\nhi there\n<think>x</think>");  // control skipped, user-defined kept
        const auto plain = tok.encode(s, false, false);
        VJ_CHECK(std::find(plain.begin(), plain.end(), tok.token_id("<|im_start|>")) == plain.end());
        VJ_CHECK(tok.decode(plain) == s);
        VJ_CHECK(tok.encode("x", true).front() == v.bos);
        VJ_CHECK(tok.unknown_symbols() == 0);
    }
    // ---- GGUF metadata loading
    {
        vjtest::GgufWriter w;
        w.kv_str("general.architecture", "qwen35moe");
        w.kv_u32("qwen35moe.block_count", 4);
        w.kv_str("tokenizer.ggml.model", "gpt2");
        w.kv_str("tokenizer.ggml.pre", "qwen2");
        w.kv_arr_str("tokenizer.ggml.tokens", v.tokens);
        w.kv_arr_str("tokenizer.ggml.merges", v.merges);
        w.kv_arr_i32("tokenizer.ggml.token_type", v.types);
        w.kv_u32("tokenizer.ggml.eos_token_id", static_cast<uint32_t>(v.eos));
        w.kv_u32("tokenizer.ggml.bos_token_id", static_cast<uint32_t>(v.bos));
        w.kv_bool("tokenizer.ggml.add_bos_token", false);
        w.kv_str("tokenizer.chat_template", "{% for m in messages %}<|im_start|>{{ m.role }}...<think>{% endfor %}");
        w.tensor("token_embd.weight", {8, 4}, 0, std::vector<uint8_t>(8 * 4 * 4, 0));
        const auto bytes = w.build();
        const std::string path = vjtest::write_temp(bytes, "bpe_vocab");
        Vocab g;
        VJ_CHECK(read_gguf_vocab(path, g, &err));
        VJ_CHECK(g.tokens == v.tokens && g.merges == v.merges && g.types == v.types && g.eos == v.eos && g.bos == v.bos);
        VJ_CHECK(g.architecture == "qwen35moe" && g.pre == "qwen2" && !g.chat_template.empty());
        VJ_CHECK(detect_chat_format(g.chat_template, g.architecture) == ChatFormat::Ornith);
        Tokenizer t2;
        VJ_CHECK(t2.load(g, &err) && t2.encode(kCorpus) == tok.encode(kCorpus));
        Vocab m;
        VJ_CHECK(parse_gguf_vocab(bytes.data(), bytes.size(), m, &err) && m.tokens == v.tokens);
        for (size_t cut : {size_t(3), size_t(30), bytes.size() / 3, bytes.size() / 2}) {
            Vocab x;
            VJ_CHECK(!parse_gguf_vocab(bytes.data(), cut, x, &err));  // truncated: error, no crash
        }
        std::vector<uint8_t> junk(bytes);
        for (size_t i = 24; i < junk.size(); i += 97) junk[i] ^= 0xFF;  // corrupt lengths/types
        Vocab x;
        (void)parse_gguf_vocab(junk.data(), junk.size(), x, &err);  // must not crash (ASan)
        VJ_CHECK(!read_gguf_vocab(path + ".missing", x, &err));
        Vocab spm = v;
        spm.model = "llama";
        VJ_CHECK(!Tokenizer().load(spm, &err) && err.find("byte-level") != std::string::npos);
        std::remove(path.c_str());
    }
    // ---- agreement with HEAD's BpeTokenizer pre-tokenization (exact chunk check, ASCII).
    // Every qwen2 pre-token of the corpus becomes one whole vocabulary token (plus all 256 byte symbols, no merges), so
    // HEAD's encode (whole-chunk fast path, byte fallback otherwise) returns the expected ids iff HEAD chunks the text
    // exactly like the qwen2 pattern. run_tests.sh builds it against HEAD as committed (report only), with
    // tokenizer_qwen2_newline.patch applied (-DVJ_HEAD_TOKENIZER_PATCHED: the bug class must show zero diffs), or,
    // when HEAD already delegates to bpe::Tokenizer (ec982d2 and later), with -DVJ_HEAD_TOKENIZER_QWEN2 (no diffs).
    {
        const auto bt = byte_table();
        auto mapped = [&](const std::string& c) { std::string m; for (unsigned char b : c) m += bt[b]; return m; };
        // Bug class: a punctuation run directly followed by newlines (" ?[^\s\p{L}\p{N}]+[\r\n]*").
        const std::vector<std::string> bug = {
            "Hello!!!\n\nWorld", "if x == 1:\n    pass\n", "f(1, 2):\n    return 0;\n}\n", "x = [1, 2];\r\n",
            "end.\n\n\nNext", "wait ...\nok", "a: 1,\n b: 2\n", "</div>\n", "  return {};\n};\n\n",
            "x <= 4;\n\n", "s = \"a\";\n"};
        // Already identical before the patch (no punctuation+newline adjacency, no residual class below).
        const std::vector<std::string> same = {"The quick brown fox jumps over the lazy dog", "It's 2026, they're here; we'll see",
                                               "x = 12345 * 6 + 7", "World  again   ", "a\n\nb\r\nc"};
        // Residual qwen2 differences that the patch does NOT address (report only, see INTEGRATION.md):
        // a single non-letter prefix before letters ("(bar", ".y"), whitespace before newlines ("  \n").
        const std::vector<std::string> residual = {"foo(bar)", "x.y", "a  \nb", "#include"};
        std::map<std::string, int32_t> ids;
        Vocab hv;
        for (int b = 0; b < 256; ++b) { ids.emplace(bt[b], 0); }
        auto all = bug; all.insert(all.end(), same.begin(), same.end()); all.insert(all.end(), residual.begin(), residual.end());
        for (const auto& t : all)
            for (const auto& c : Tokenizer::pretokenize(t, PreTok::Qwen2)) ids.emplace(mapped(c), 0);
        for (auto& [str, id] : ids) { id = static_cast<int32_t>(hv.tokens.size()); hv.tokens.push_back(str); }
        hv.types.assign(hv.tokens.size(), 1);
        BpeTokenizer head;
        VJ_CHECK(head.load_from_vectors(hv.tokens, {}, hv.types, -1, -1, -1, true));
        auto diff = [&](const std::string& t) {
            std::vector<int32_t> want;
            for (const auto& c : Tokenizer::pretokenize(t, PreTok::Qwen2)) want.push_back(ids.at(mapped(c)));
            return head.encode(t, false) != want;
        };
        uint32_t d_bug = 0, d_same = 0, d_res = 0;
        for (const auto& t : bug) d_bug += diff(t);
        for (const auto& t : same) d_same += diff(t);
        for (const auto& t : residual) d_res += diff(t);
        VJ_CHECK((Tokenizer::pretokenize("argv):\n", PreTok::Qwen2) == V{"argv", "):\n"}));
        VJ_CHECK(d_same == 0);
#if defined(VJ_HEAD_TOKENIZER_PATCHED)
        VJ_CHECK(d_bug == 0);
        const char* which = "HEAD + tokenizer_qwen2_newline.patch";
#elif defined(VJ_HEAD_TOKENIZER_QWEN2)
        VJ_CHECK(d_bug == 0 && d_res == 0);  // HEAD delegates to bpe::Tokenizer::pretokenize(Qwen2): everything agrees
        const char* which = "HEAD (BpeTokenizer delegates to bpe::Tokenizer)";
#else
        const char* which = "HEAD as committed";
#endif
        std::printf("  %s vs qwen2 pre-tokens: punctuation+newline %u/%zu differ, plain %u/%zu, residual (not in the "
                    "patch's scope) %u/%zu\n", which, d_bug, bug.size(), d_same, same.size(), d_res, residual.size());
    }
    // ---- chat templates
    {
        const std::vector<Message> msgs = {{"user", "Hi"}, {"assistant", "<think>\nplan\n</think>\n\nHello!"}, {"user", "More"}};
        VJ_CHECK(apply_chat_template(ChatFormat::ChatML, msgs) ==
                 "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\nplan\n</think>\n\nHello!<|im_end|>\n"
                 "<|im_start|>user\nMore<|im_end|>\n<|im_start|>assistant\n");
        VJ_CHECK(apply_chat_template(ChatFormat::Qwen3, msgs) ==
                 "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello!<|im_end|>\n"
                 "<|im_start|>user\nMore<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        ChatOptions think;
        think.enable_thinking = true;
        const std::string ot = apply_chat_template(ChatFormat::Ornith, {{"user", "Q"}}, think);
        VJ_CHECK(ot == "<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n");
        VJ_CHECK(apply_chat_template(ChatFormat::Qwen25, {{"user", "Q"}}).rfind("<|im_start|>system\nYou are Qwen", 0) == 0);
        VJ_CHECK(apply_chat_template(ChatFormat::Qwen25, {{"system", "S"}, {"user", "Q"}}).rfind("<|im_start|>system\nS<|im_end|>", 0) == 0);
        VJ_CHECK(detect_chat_format("", "qwen3moe") == ChatFormat::Qwen3 && detect_chat_format("", "llama") == ChatFormat::ChatML);
        VJ_CHECK(detect_chat_format("<|im_start|>... You are Qwen ...", "qwen2") == ChatFormat::Qwen25);
    }
    // ---- adapter: template + sidecar tokenizer feeding the HEAD MockEngine
    {
        MockEngine engine("mock");  // ignores prompt contents; reports prompt length and canned text
        const auto fn = serve::make_engine_generate_fn(engine, tok, ChatFormat::Qwen3, ChatOptions{});
        serve::GenerateRequest rq;
        rq.messages = {{"user", "Hello"}};
        rq.max_tokens = 4;
        std::string got;
        const auto res = fn(rq, [&](std::string_view p) { got.append(p.data(), p.size()); return true; });
        VJ_CHECK(got == "Hello from VegaJet mock" && res.finish == serve::FinishReason::Length);
        const std::string prompt = apply_chat_template(ChatFormat::Qwen3, {{"user", "Hello"}});
        VJ_CHECK(res.prompt_tokens == tok.encode(prompt).size());
    }
    std::printf("test_bpe_tokenizer: pre-tokenizers, merges vs reference, round trips, specials, GGUF, HEAD agreement, "
                "chat templates, adapter OK\nPASS\n");
    return 0;
}
