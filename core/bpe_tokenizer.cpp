// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/bpe_tokenizer.hpp"

#include "core/pretok_families.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <queue>

namespace vegajet::bpe {

// =================================================================================================
// Unicode helpers
// =================================================================================================
namespace {

struct Cp {
    uint32_t cp;
    uint32_t off, len;  // byte span in the source
};

// Decodes UTF-8; every invalid byte becomes its own code point 0xFFFD with a 1-byte span (bytes are never lost).
std::vector<Cp> decode_utf8(std::string_view s) {
    std::vector<Cp> out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        const auto c = static_cast<unsigned char>(s[i]);
        uint32_t cp = 0xFFFD, n = 1;
        if (c < 0x80) { cp = c; }
        else if ((c & 0xE0) == 0xC0) n = 2, cp = c & 0x1F;
        else if ((c & 0xF0) == 0xE0) n = 3, cp = c & 0x0F;
        else if ((c & 0xF8) == 0xF0) n = 4, cp = c & 0x07;
        else n = 0;
        bool ok = n > 0 && i + n <= s.size();
        for (uint32_t k = 1; ok && k < n; ++k) {
            const auto cc = static_cast<unsigned char>(s[i + k]);
            if ((cc & 0xC0) != 0x80) ok = false;
            else cp = (cp << 6) | (cc & 0x3F);
        }
        if (ok && n > 1) {  // reject overlongs / surrogates / out of range
            static const uint32_t kMin[5] = {0, 0, 0x80, 0x800, 0x10000};
            if (cp < kMin[n] || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) ok = false;
        }
        if (!ok || n == 0) { out.push_back({0xFFFD, static_cast<uint32_t>(i), 1}); ++i; continue; }
        out.push_back({cp, static_cast<uint32_t>(i), n});
        i += n;
    }
    return out;
}

void append_utf8(std::string& s, uint32_t cp) {
    if (cp < 0x80) s += static_cast<char>(cp);
    else if (cp < 0x800) { s += static_cast<char>(0xC0 | (cp >> 6)); s += static_cast<char>(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) {
        s += static_cast<char>(0xE0 | (cp >> 12)); s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        s += static_cast<char>(0xF0 | (cp >> 18)); s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); s += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

struct Range { uint32_t lo, hi; };
bool in(const Range* r, size_t n, uint32_t cp) {
    for (size_t i = 0; i < n; ++i) if (cp >= r[i].lo && cp <= r[i].hi) return true;
    return false;
}

const Range kSpace[] = {{0x09, 0x0D}, {0x20, 0x20}, {0x85, 0x85}, {0xA0, 0xA0}, {0x1680, 0x1680}, {0x2000, 0x200A},
                        {0x2028, 0x2029}, {0x202F, 0x202F}, {0x205F, 0x205F}, {0x3000, 0x3000}};
const Range kNumber[] = {{0x30, 0x39}, {0xB2, 0xB3}, {0xB9, 0xB9}, {0xBC, 0xBE}, {0x660, 0x669}, {0x6F0, 0x6F9},
                         {0x7C0, 0x7C9}, {0x966, 0x96F}, {0x9E6, 0x9EF}, {0xA66, 0xA6F}, {0xAE6, 0xAEF}, {0xBE6, 0xBF2},
                         {0xC66, 0xC6F}, {0xCE6, 0xCEF}, {0xD66, 0xD6F}, {0xE50, 0xE59}, {0xED0, 0xED9}, {0xF20, 0xF33},
                         {0x1040, 0x1049}, {0x17E0, 0x17E9}, {0x1810, 0x1819}, {0x2070, 0x2070}, {0x2074, 0x2079},
                         {0x2080, 0x2089}, {0x2150, 0x2189}, {0x2460, 0x249B}, {0x24EA, 0x24FF}, {0x2776, 0x2793},
                         {0x3007, 0x3007}, {0x3021, 0x3029}, {0x3038, 0x303A}, {0x3192, 0x3195}, {0x3220, 0x3229},
                         {0x3248, 0x324F}, {0x3251, 0x325F}, {0x3280, 0x3289}, {0x32B1, 0x32BF}, {0xFF10, 0xFF19},
                         {0x1D7CE, 0x1D7FF}};
// Non-letter blocks above ASCII: Latin-1 punctuation/symbols, combining marks, general punctuation, symbols,
// arrows, math, box drawing, dingbats, CJK punctuation, fullwidth punctuation, private use, emoji, format chars.
const Range kNonLetter[] = {{0x80, 0xA9}, {0xAB, 0xB4}, {0xB6, 0xB9}, {0xBB, 0xBF}, {0xD7, 0xD7}, {0xF7, 0xF7},
                            {0x2C2, 0x2C5}, {0x2D2, 0x2DF}, {0x2E5, 0x2EB}, {0x2ED, 0x2ED}, {0x2EF, 0x36F},
                            {0x375, 0x375}, {0x37E, 0x37E}, {0x384, 0x385}, {0x387, 0x387}, {0x482, 0x489},
                            {0x55A, 0x55F}, {0x589, 0x58A}, {0x591, 0x5C7}, {0x5F3, 0x5F4}, {0x600, 0x61F},
                            {0x64B, 0x65F}, {0x66A, 0x66D}, {0x6D4, 0x6D4}, {0x6D6, 0x6ED}, {0x964, 0x965},
                            {0x1AB0, 0x1AFF}, {0x1DC0, 0x1DFF}, {0x2000, 0x2BFF}, {0x2E00, 0x2E7F}, {0x2FF0, 0x303F},
                            {0x3099, 0x309C}, {0x30A0, 0x30A0}, {0x30FB, 0x30FB}, {0x3200, 0x33FF}, {0xD800, 0xF8FF},
                            {0xFE00, 0xFE6F}, {0xFEFF, 0xFEFF}, {0xFF00, 0xFF0F}, {0xFF1A, 0xFF20}, {0xFF3B, 0xFF40},
                            {0xFF5B, 0xFF65}, {0xFFF0, 0xFFFF}, {0x1F000, 0x1FAFF}, {0xE0000, 0xE007F}};

bool is_space(uint32_t cp) { return in(kSpace, sizeof(kSpace) / sizeof(Range), cp); }
bool is_number(uint32_t cp) { return in(kNumber, sizeof(kNumber) / sizeof(Range), cp); }
bool is_letter(uint32_t cp) {
    if (cp < 0x80) return (cp | 0x20) >= 'a' && (cp | 0x20) <= 'z';
    if (cp == 0xFFFD) return false;
    if (is_space(cp) || is_number(cp)) return false;
    return !in(kNonLetter, sizeof(kNonLetter) / sizeof(Range), cp);
}
bool is_crlf(uint32_t cp) { return cp == '\r' || cp == '\n'; }
bool is_other(uint32_t cp) { return !is_space(cp) && !is_letter(cp) && !is_number(cp); }

uint32_t lower_ascii(uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

// Contractions 's 't 're 've 'm 'll 'd at i; returns length in code points (0 = none).
size_t contraction(const std::vector<Cp>& c, size_t i, bool ci) {
    if (c[i].cp != '\'' || i + 1 >= c.size()) return 0;
    auto at = [&](size_t k) { return i + k < c.size() ? (ci ? lower_ascii(c[i + k].cp) : c[i + k].cp) : 0u; };
    const uint32_t a = at(1), b = at(2);
    if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
    if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
    return 0;
}

} // namespace

const char* pretok_name(PreTok p) {
    static const char* const k[] = {"gpt2", "llama3", "qwen2", "deepseek-llm", "deepseek-coder", "deepseek-v3", "tekken", "spm"};
    const auto i = static_cast<size_t>(p);
    return i < sizeof(k) / sizeof(k[0]) ? k[i] : "?";
}

PreTok pretok_from_gguf(const std::string& pre, const std::string& model) {
    if (model == "llama") return PreTok::Spm;
    if (pre == "deepseek-llm") return PreTok::DeepSeekLlm;
    if (pre == "deepseek-coder") return PreTok::DeepSeekCoder;
    if (pre == "deepseek-v3") return PreTok::DeepSeekV3;
    if (pre == "tekken") return PreTok::Tekken;
    if (pre == "falcon3") return PreTok::Llama3;
    return pretok_from_gguf(pre);
}

PreTok pretok_from_gguf(const std::string& pre) {
    if (pre == "deepseek-llm") return PreTok::DeepSeekLlm;
    if (pre == "deepseek-coder") return PreTok::DeepSeekCoder;
    if (pre == "deepseek-v3") return PreTok::DeepSeekV3;
    if (pre == "tekken") return PreTok::Tekken;
    if (pre.rfind("qwen", 0) == 0) return PreTok::Qwen2;
    if (pre == "llama-bpe" || pre == "llama3" || pre == "llama-v3") return PreTok::Llama3;
    return PreTok::Gpt2;
}

std::vector<std::string> Tokenizer::pretokenize(std::string_view text, PreTok mode) {
    if (mode != PreTok::Gpt2 && mode != PreTok::Llama3 && mode != PreTok::Qwen2) return pretok::split(text, mode);
    const std::vector<Cp> c = decode_utf8(text);
    const size_t n = c.size();
    std::vector<std::string> out;
    auto emit = [&](size_t a, size_t b) {
        out.emplace_back(text.substr(c[a].off, c[b - 1].off + c[b - 1].len - c[a].off));
    };
    auto run = [&](size_t j, bool (*pred)(uint32_t)) { while (j < n && pred(c[j].cp)) ++j; return j; };
    size_t i = 0;
    while (i < n) {
        const uint32_t ch = c[i].cp;
        size_t e = 0;
        if (mode == PreTok::Gpt2) {
            if (size_t k = contraction(c, i, false)) e = i + k;
            else {
                const size_t j = (ch == ' ' && i + 1 < n) ? i + 1 : i;   // " ?" prefix
                if (is_letter(c[j].cp)) e = run(j, is_letter);
                else if (is_number(c[j].cp)) e = run(j, is_number);
                else if (is_other(c[j].cp)) e = run(j, is_other);
                else if (j != i && is_letter(ch)) e = run(i, is_letter);  // unreachable (space is not a letter)
            }
        } else {
            if (size_t k = contraction(c, i, true)) e = i + k;
            else if (!is_crlf(ch) && !is_letter(ch) && !is_number(ch) && i + 1 < n && is_letter(c[i + 1].cp))
                e = run(i + 1, is_letter);                                  // [^\r\n\p{L}\p{N}]?\p{L}+
            else if (is_letter(ch)) e = run(i, is_letter);
            else if (is_number(ch)) {                                       // \p{N} or \p{N}{1,3}
                e = i + 1;
                if (mode == PreTok::Llama3) while (e < n && e < i + 3 && is_number(c[e].cp)) ++e;
            } else {
                const size_t j = (ch == ' ' && i + 1 < n && is_other(c[i + 1].cp)) ? i + 1 : i;
                if (is_other(c[j].cp)) e = run(run(j, is_other), is_crlf);  // " ?[^\s\p{L}\p{N}]+[\r\n]*"
            }
        }
        if (e == 0 && is_space(ch)) {
            const size_t ws = run(i, is_space);
            if (mode != PreTok::Gpt2) {                                     // \s*[\r\n]+
                size_t last_nl = n;
                for (size_t k = i; k < ws; ++k) if (is_crlf(c[k].cp)) last_nl = k;
                if (last_nl != n) e = last_nl + 1;
            }
            if (e == 0) e = (ws == n || ws - i < 2) ? ws : ws - 1;          // \s+(?!\S) | \s+
        }
        if (e <= i) e = i + 1;  // defensive: always progress
        emit(i, e);
        i = e;
    }
    return out;
}

// =================================================================================================
// GGUF metadata reader (KV section only)
// =================================================================================================
namespace {

class Src {
public:
    Src(const uint8_t* d, size_t n) : mem_(d), size_(n) {}
    Src(std::ifstream* f, size_t n) : file_(f), size_(n) {}
    bool read(void* dst, size_t n) {
        if (n > size_ - pos_) return false;
        if (mem_) std::memcpy(dst, mem_ + pos_, n);
        else if (!file_->read(static_cast<char*>(dst), static_cast<std::streamsize>(n))) return false;
        pos_ += n;
        return true;
    }
    bool skip(uint64_t n) {
        if (n > size_ - pos_) return false;
        if (file_) file_->seekg(static_cast<std::streamoff>(n), std::ios::cur);
        pos_ += n;
        return true;
    }
    size_t remaining() const { return size_ - pos_; }
    template <class T> bool pod(T& v) { return read(&v, sizeof(T)); }
    bool str(std::string& s, uint64_t cap = 64u << 20) {
        uint64_t n = 0;
        if (!pod(n) || n > cap || n > remaining()) return false;
        s.resize(n);
        return n == 0 || read(&s[0], n);
    }
private:
    const uint8_t* mem_ = nullptr;
    std::ifstream* file_ = nullptr;
    size_t size_ = 0, pos_ = 0;
};

constexpr uint32_t kTypeSize[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};

bool read_int(Src& s, uint32_t t, int64_t& v) {
    switch (t) {
        case 0: { uint8_t x; if (!s.pod(x)) return false; v = x; return true; }
        case 1: { int8_t x; if (!s.pod(x)) return false; v = x; return true; }
        case 2: { uint16_t x; if (!s.pod(x)) return false; v = x; return true; }
        case 3: { int16_t x; if (!s.pod(x)) return false; v = x; return true; }
        case 4: { uint32_t x; if (!s.pod(x)) return false; v = x; return true; }
        case 5: { int32_t x; if (!s.pod(x)) return false; v = x; return true; }
        case 7: { uint8_t x; if (!s.pod(x)) return false; v = x; return true; }
        case 10: { uint64_t x; if (!s.pod(x)) return false; v = static_cast<int64_t>(x); return true; }
        case 11: { int64_t x; if (!s.pod(x)) return false; v = x; return true; }
        default: return false;
    }
}

bool skip_value(Src& s, uint32_t t, int depth = 0) {
    if (t < 13 && kTypeSize[t]) return s.skip(kTypeSize[t]);
    if (t == 8) { std::string x; return s.str(x); }
    if (t == 9 && depth < 4) {
        uint32_t et; uint64_t n;
        if (!s.pod(et) || !s.pod(n)) return false;
        if (et < 13 && kTypeSize[et]) return n <= s.remaining() / kTypeSize[et] && s.skip(n * kTypeSize[et]);
        if (n > s.remaining() / 8) return false;
        for (uint64_t i = 0; i < n; ++i) if (!skip_value(s, et, depth + 1)) return false;
        return true;
    }
    return false;
}

bool parse(Src& s, Vocab& v, std::string* err) {
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };
    char magic[4];
    uint32_t ver;
    uint64_t n_tensors, n_kv;
    if (!s.read(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) return fail("not a GGUF file");
    if (!s.pod(ver) || ver < 2 || ver > 3) return fail("unsupported GGUF version");
    if (!s.pod(n_tensors) || !s.pod(n_kv) || n_kv > (1u << 20)) return fail("bad GGUF header");
    v = Vocab{};
    v.model.clear();
    v.pre.clear();
    bool have_tokens = false;
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key;
        uint32_t t;
        if (!s.str(key, 1u << 16) || !s.pod(t)) return fail("truncated GGUF KV section");
        auto str_array = [&](std::vector<std::string>& out) {
            uint32_t et; uint64_t n;
            if (!s.pod(et) || !s.pod(n) || et != 8 || n > s.remaining() / 8) return false;
            out.resize(n);
            for (uint64_t k = 0; k < n; ++k) if (!s.str(out[k], 1u << 16)) return false;
            return true;
        };
        bool ok = true;
        if (key == "tokenizer.ggml.tokens" && t == 9) { ok = str_array(v.tokens); have_tokens = ok; }
        else if (key == "tokenizer.ggml.merges" && t == 9) ok = str_array(v.merges);
        else if (key == "tokenizer.ggml.scores" && t == 9) {
            uint32_t et; uint64_t n;
            ok = s.pod(et) && s.pod(n) && et == 6 && n <= s.remaining() / 4;
            if (ok) { v.scores.resize(n); ok = n == 0 || s.read(v.scores.data(), n * 4); }
        } else if (key == "tokenizer.ggml.add_space_prefix" && t == 7) {
            uint8_t b = 0;
            ok = s.pod(b);
            v.add_space_prefix = b != 0;
        }
        else if (key == "tokenizer.ggml.token_type" && t == 9) {
            uint32_t et; uint64_t n;
            ok = s.pod(et) && s.pod(n) && (et == 4 || et == 5) && n <= s.remaining() / 4;
            if (ok) { v.types.resize(n); ok = n == 0 || s.read(v.types.data(), n * 4); }
        } else if ((key == "tokenizer.ggml.model" || key == "tokenizer.ggml.pre" || key == "tokenizer.chat_template" ||
                    key == "general.architecture") && t == 8) {
            std::string val;
            ok = s.str(val);
            if (key == "tokenizer.ggml.model") v.model = val;
            else if (key == "tokenizer.ggml.pre") v.pre = val;
            else if (key == "tokenizer.chat_template") v.chat_template = val;
            else v.architecture = val;
        } else if (key == "tokenizer.ggml.bos_token_id" || key == "tokenizer.ggml.eos_token_id" ||
                   key == "tokenizer.ggml.padding_token_id" || key == "tokenizer.ggml.unknown_token_id" ||
                   key == "tokenizer.ggml.add_bos_token") {
            int64_t x = 0;
            ok = read_int(s, t, x);
            if (key == "tokenizer.ggml.bos_token_id") v.bos = static_cast<int32_t>(x);
            else if (key == "tokenizer.ggml.eos_token_id") v.eos = static_cast<int32_t>(x);
            else if (key == "tokenizer.ggml.padding_token_id") v.pad = static_cast<int32_t>(x);
            else if (key == "tokenizer.ggml.unknown_token_id") v.unk = static_cast<int32_t>(x);
            else v.add_bos = x != 0;
        } else {
            ok = skip_value(s, t);
        }
        if (!ok) return fail("malformed value for key " + key);
    }
    if (!have_tokens) return fail("no tokenizer.ggml.tokens");
    if (v.model.empty()) v.model = "gpt2";
    return true;
}

} // namespace

bool parse_gguf_vocab(const uint8_t* data, size_t size, Vocab& out, std::string* err) {
    if (!data) { if (err) *err = "null buffer"; return false; }
    Src s(data, size);
    return parse(s, out, err);
}

bool read_gguf_vocab(const std::string& path, Vocab& out, std::string* err) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { if (err) *err = "cannot open file"; return false; }
    const size_t size = static_cast<size_t>(f.tellg());
    f.seekg(0);
    Src s(&f, size);
    return parse(s, out, err);
}

// =================================================================================================
// Tokenizer
// =================================================================================================
bool Tokenizer::load(Vocab v, std::string* err) {
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };
    if (v.model != "gpt2" && v.model != "llama")
        return fail("only byte-level BPE (gpt2) and SentencePiece (llama) vocabularies are supported, got '" + v.model + "'");
    if (v.tokens.empty()) return fail("empty vocabulary");
    if (!v.types.empty() && v.types.size() != v.tokens.size()) return fail("token_type length differs from tokens");
    if (v.model == "llama" && v.scores.size() != v.tokens.size())
        return fail("SentencePiece vocabulary without tokenizer.ggml.scores (supported: byte-level BPE, or SentencePiece with scores)");
    v_ = std::move(v);
    pre_ = pretok_from_gguf(v_.pre, v_.model);
    tokens_ = v_.tokens;
    tok_id_.clear();
    tok_id_.reserve(tokens_.size());
    for (size_t i = 0; i < tokens_.size(); ++i) tok_id_.emplace(tokens_[i], static_cast<int32_t>(i));  // first wins
    merge_rank_.clear();
    merge_rank_.reserve(v_.merges.size());
    for (size_t i = 0; i < v_.merges.size(); ++i) merge_rank_.emplace(v_.merges[i], static_cast<int32_t>(i));
    // GPT-2 byte <-> unicode table.
    sym_byte_.clear();
    uint32_t extra = 0;
    for (uint32_t b = 0; b < 256; ++b) {
        const bool direct = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        const uint32_t cp = direct ? b : 256 + extra++;
        byte_sym_[b].clear();
        append_utf8(byte_sym_[b], cp);
        sym_byte_[cp] = static_cast<uint8_t>(b);
    }
    specials_.clear();
    for (size_t i = 0; i < tokens_.size(); ++i) {
        const int32_t t = v_.types.empty() ? 1 : v_.types[i];
        if ((t == 3 || t == 4) && !tokens_[i].empty()) specials_.push_back({tokens_[i], static_cast<int32_t>(i)});
    }
    std::stable_sort(specials_.begin(), specials_.end(), [](auto& a, auto& b) { return a.first.size() > b.first.size(); });
    unknown_ = 0;
    return true;
}

bool Tokenizer::is_special(int32_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= tokens_.size() || v_.types.empty()) return false;
    return v_.types[id] == 3 || v_.types[id] == 4;
}

int32_t Tokenizer::token_id(const std::string& t) const {
    const auto it = tok_id_.find(t);
    return it == tok_id_.end() ? -1 : it->second;
}

void Tokenizer::bpe_word(const std::string& word, std::vector<int32_t>& out) const {
    // Symbols: one per byte, mapped to the byte-level alphabet.
    struct Sym { std::string s; int prev, next; bool alive; };
    std::vector<Sym> sy;
    sy.reserve(word.size());
    for (size_t i = 0; i < word.size(); ++i)
        sy.push_back({byte_sym_[static_cast<unsigned char>(word[i])], static_cast<int>(i) - 1,
                      i + 1 < word.size() ? static_cast<int>(i) + 1 : -1, true});
    struct Cand { int32_t rank; int left, right; };
    auto cmp = [](const Cand& a, const Cand& b) { return a.rank != b.rank ? a.rank > b.rank : a.left > b.left; };
    std::priority_queue<Cand, std::vector<Cand>, decltype(cmp)> heap(cmp);
    auto rank_of = [&](int l, int r) -> int32_t {
        const auto it = merge_rank_.find(sy[l].s + " " + sy[r].s);
        return it == merge_rank_.end() ? -1 : it->second;
    };
    auto push = [&](int l) {
        if (l < 0 || sy[l].next < 0) return;
        const int32_t rk = rank_of(l, sy[l].next);
        if (rk >= 0) heap.push({rk, l, sy[l].next});
    };
    for (int i = 0; i + 1 < static_cast<int>(sy.size()); ++i) push(i);
    while (!heap.empty()) {
        const Cand c = heap.top();
        heap.pop();
        if (!sy[c.left].alive || !sy[c.right].alive || sy[c.left].next != c.right || rank_of(c.left, c.right) != c.rank) continue;
        sy[c.left].s += sy[c.right].s;
        sy[c.right].alive = false;
        sy[c.left].next = sy[c.right].next;
        if (sy[c.right].next >= 0) sy[sy[c.right].next].prev = c.left;
        push(sy[c.left].prev);
        push(c.left);
    }
    for (int i = sy.empty() ? -1 : 0; i >= 0; i = sy[i].next) {
        const auto it = tok_id_.find(sy[i].s);
        if (it != tok_id_.end()) { out.push_back(it->second); continue; }
        ++unknown_;
        // Fall back to single byte symbols (always in a complete byte-level vocabulary), else unk.
        const std::vector<Cp> cps = decode_utf8(sy[i].s);
        for (const Cp& cp : cps) {
            std::string one;
            append_utf8(one, cp.cp);
            const auto jt = tok_id_.find(one);
            if (jt != tok_id_.end()) out.push_back(jt->second);
            else if (v_.unk >= 0) out.push_back(v_.unk);
        }
    }
}

std::vector<int32_t> Tokenizer::encode(std::string_view text, bool add_bos, bool parse_special) const {
    std::vector<int32_t> out;
    if (add_bos && v_.bos >= 0) out.push_back(v_.bos);
    auto encode_plain = [&](std::string_view seg) {
        if (pre_ == PreTok::Spm) {
            const bool at_start = seg.data() == text.data();
            pretok::spm_encode(pretok::spm_escape(seg, v_.add_space_prefix && at_start), tok_id_, v_.scores, v_.unk, out);
            return;
        }
        for (const auto& w : pretokenize(seg, pre_)) bpe_word(w, out);
    };
    if (!parse_special || specials_.empty()) { encode_plain(text); return out; }
    size_t start = 0, i = 0;
    while (i < text.size()) {
        int32_t hit = -1;
        size_t hit_len = 0;
        for (const auto& sp : specials_) {
            if (sp.first.size() <= text.size() - i && text.compare(i, sp.first.size(), sp.first) == 0) {
                hit = sp.second;
                hit_len = sp.first.size();
                break;  // longest first
            }
        }
        if (hit < 0) { ++i; continue; }
        if (i > start) encode_plain(text.substr(start, i - start));
        out.push_back(hit);
        i += hit_len;
        start = i;
    }
    if (start < text.size()) encode_plain(text.substr(start));
    return out;
}

std::string Tokenizer::token_bytes(int32_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) return {};
    const int32_t t = v_.types.empty() ? 1 : v_.types[id];
    const std::string& s = tokens_[id];
    if (t == 3) return {};
    if (t == 4) return s;
    if (t == 6 && s.size() == 6 && s.compare(0, 3, "<0x") == 0 && s[5] == '>')
        return std::string(1, static_cast<char>(std::stoul(s.substr(3, 2), nullptr, 16)));
    if (pre_ == PreTok::Spm) {  // U+2581 -> ' '
        std::string out;
        for (size_t k = 0; k < s.size(); ++k) {
            if (s.compare(k, 3, "\xE2\x96\x81") == 0) { out += ' '; k += 2; }
            else out += s[k];
        }
        return out;
    }
    std::string out;
    for (const Cp& c : decode_utf8(s)) {
        const auto it = sym_byte_.find(c.cp);
        if (it != sym_byte_.end()) out += static_cast<char>(it->second);
        else out.append(s, c.off, c.len);
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids, bool skip_special) const {
    std::string out;
    for (int32_t id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) continue;
        const int32_t t = v_.types.empty() ? 1 : v_.types[id];
        if (t == 3) { if (!skip_special) out += tokens_[id]; continue; }
        if (t == 4) { out += tokens_[id]; continue; }  // user-defined tokens are text (e.g. <think>): always rendered
        out += token_bytes(id);
    }
    return out;
}

// =================================================================================================
// Chat templates
// =================================================================================================
const char* chat_format_name(ChatFormat f) {
    static const char* const k[] = {"chatml", "qwen2.5", "qwen3", "ornith"};
    return k[static_cast<size_t>(f)];
}

ChatFormat detect_chat_format(const std::string& tmpl, const std::string& arch) {
    if (!tmpl.empty() && tmpl.find("<|im_start|>") != std::string::npos) {
        if (tmpl.find("<think>") != std::string::npos) return arch == "qwen35moe" ? ChatFormat::Ornith : ChatFormat::Qwen3;
        if (tmpl.find("You are Qwen") != std::string::npos) return ChatFormat::Qwen25;
        return ChatFormat::ChatML;
    }
    if (arch == "qwen35moe") return ChatFormat::Ornith;
    if (arch.rfind("qwen3", 0) == 0) return ChatFormat::Qwen3;
    if (arch.rfind("qwen2", 0) == 0) return ChatFormat::Qwen25;
    return ChatFormat::ChatML;
}

namespace {

std::string strip_reasoning(const std::string& s) {  // drop "...</think>" prefix and the whitespace after it
    const size_t e = s.rfind("</think>");
    if (e == std::string::npos) return s;
    size_t k = e + 8;
    while (k < s.size() && (s[k] == '\n' || s[k] == ' ' || s[k] == '\r' || s[k] == '\t')) ++k;
    return s.substr(k);
}

} // namespace

std::string apply_chat_template(ChatFormat f, const std::vector<Message>& msgs, const ChatOptions& opt) {
    std::string out;
    auto turn = [&](const std::string& role, const std::string& content) {
        out += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
    };
    const bool has_system = !msgs.empty() && msgs.front().role == "system";
    if (!has_system) {
        if (!opt.default_system.empty()) turn("system", opt.default_system);
        else if (f == ChatFormat::Qwen25) turn("system", "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.");
    }
    const bool qwen3 = f == ChatFormat::Qwen3 || f == ChatFormat::Ornith;
    size_t last_user = msgs.size();
    for (size_t i = 0; i < msgs.size(); ++i) if (msgs[i].role == "user") last_user = i;
    for (size_t i = 0; i < msgs.size(); ++i) {
        const auto& m = msgs[i];
        if (qwen3 && m.role == "assistant" && (last_user == msgs.size() || i < last_user)) turn(m.role, strip_reasoning(m.content));
        else turn(m.role, m.content);
    }
    if (opt.add_generation_prompt) {
        out += "<|im_start|>assistant\n";
        if (qwen3 && !opt.enable_thinking) out += "<think>\n\n</think>\n\n";
    }
    return out;
}

} // namespace vegajet::bpe
