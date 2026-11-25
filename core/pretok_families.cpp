// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/pretok_families.hpp"

#include <algorithm>
#include <cstdio>
#include <queue>

namespace vegajet::pretok {

namespace {

struct UcRange {
    uint32_t first, last;
    uint8_t cls;
};
#include "core/unicode_class_table.inc"

bool in_table(const UcRange* t, size_t n, uint32_t cp, uint8_t* cls) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (cp < t[mid].first) hi = mid;
        else if (cp > t[mid].last) lo = mid + 1;
        else {
            if (cls) *cls = t[mid].cls;
            return true;
        }
    }
    return false;
}

struct Cps {
    std::vector<uint32_t> cp;
    std::vector<size_t> off;  // byte offset of each code point; off[n] = text size
};

// Invalid UTF-8 bytes become one code point each, mapped above U+10FFFF (class "other").
Cps decode(std::string_view s) {
    Cps c;
    size_t i = 0;
    while (i < s.size()) {
        const auto b0 = static_cast<uint8_t>(s[i]);
        uint32_t cp = 0x110000u + b0;
        size_t len = 1;
        auto cont = [&](size_t k) { return i + k < s.size() && (static_cast<uint8_t>(s[i + k]) & 0xC0) == 0x80; };
        if (b0 < 0x80) cp = b0;
        else if ((b0 & 0xE0) == 0xC0 && cont(1)) { cp = ((b0 & 0x1Fu) << 6) | (s[i + 1] & 0x3F); len = 2; if (cp < 0x80) { cp = 0x110000u + b0; len = 1; } }
        else if ((b0 & 0xF0) == 0xE0 && cont(1) && cont(2)) {
            cp = ((b0 & 0x0Fu) << 12) | ((s[i + 1] & 0x3Fu) << 6) | (s[i + 2] & 0x3F);
            len = 3;
            if (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF)) { cp = 0x110000u + b0; len = 1; }
        } else if ((b0 & 0xF8) == 0xF0 && cont(1) && cont(2) && cont(3)) {
            cp = ((b0 & 0x07u) << 18) | ((s[i + 1] & 0x3Fu) << 12) | ((s[i + 2] & 0x3Fu) << 6) | (s[i + 3] & 0x3F);
            len = 4;
            if (cp < 0x10000 || cp > 0x10FFFF) { cp = 0x110000u + b0; len = 1; }
        }
        c.cp.push_back(cp);
        c.off.push_back(i);
        i += len;
    }
    c.off.push_back(s.size());
    return c;
}

bool L(uint32_t cp) { return uc_class(cp) == 1; }
bool M(uint32_t cp) { return uc_class(cp) == 2; }
bool N(uint32_t cp) { return uc_class(cp) == 3; }
bool P(uint32_t cp) { return uc_class(cp) == 4; }
bool S(uint32_t cp) { return uc_class(cp) == 5; }
bool WS(uint32_t cp) { return uc_white_space(cp); }
bool CRLF(uint32_t cp) { return cp == '\r' || cp == '\n'; }
bool in_range(uint32_t cp, uint32_t a, uint32_t b) { return cp >= a && cp <= b; }
bool ds_letter(uint32_t cp) {
    return in_table(kDeepSeekLetters, sizeof(kDeepSeekLetters) / sizeof(UcRange), cp, nullptr);
}
bool ds_punct(uint32_t cp) {
    return in_range(cp, 0x21, 0x2F) || in_range(cp, 0x3A, 0x7E) || in_range(cp, 0xFF01, 0xFF0F) ||
           in_range(cp, 0xFF1A, 0xFF5E) || in_range(cp, 0x2018, 0x201F) || in_range(cp, 0x3000, 0x3002);
}
bool cjk_hangul(uint32_t cp) { return in_range(cp, 0x800, 0x9FA5) || in_range(cp, 0xAC00, 0xD7FF); }
bool cjk_kana(uint32_t cp) { return in_range(cp, 0x4E00, 0x9FA5) || in_range(cp, 0x3040, 0x30FF); }
bool ascii_alpha(uint32_t cp) { return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z'); }
bool ascii_punct(uint32_t cp) {
    return (cp >= 0x21 && cp <= 0x2F) || (cp >= 0x3A && cp <= 0x40) || (cp >= 0x5B && cp <= 0x60) || (cp >= 0x7B && cp <= 0x7E);
}

using Match = size_t (*)(const std::vector<uint32_t>& c, size_t i, size_t end);  // end of match, or i (no match)

template <class Pred> size_t run(const std::vector<uint32_t>& c, size_t j, size_t end, Pred p) {
    while (j < end && p(c[j])) ++j;
    return j;
}

// \s*[\r\n]+ | \s+(?!\S) | \s+   (the shared tail of the llama3 / qwen2 / tekken / deepseek-v3 alternations)
size_t ws_tail(const std::vector<uint32_t>& c, size_t i, size_t end) {
    if (!WS(c[i])) return i;
    const size_t w = run(c, i, end, WS);
    size_t last = end;
    for (size_t k = i; k < w; ++k)
        if (CRLF(c[k])) last = k;
    if (last != end) return last + 1;
    if (w == end) return w;
    return w - i >= 2 ? w - 1 : w;
}

// \s?X+  (greedy optional prefix, then backtrack to no prefix)
template <class X> size_t opt_ws_run(const std::vector<uint32_t>& c, size_t i, size_t end, X x) {
    if (WS(c[i]) && i + 1 < end && x(c[i + 1])) return run(c, i + 1, end, x);
    if (x(c[i])) return run(c, i, end, x);
    return i;
}

// ---- deepseek-llm passes
size_t ds_crlf(const std::vector<uint32_t>& c, size_t i, size_t) { return CRLF(c[i]) ? i + 1 : i; }
size_t ds_letters(const std::vector<uint32_t>& c, size_t i, size_t end) { return opt_ws_run(c, i, end, ds_letter); }
size_t ds_puncts(const std::vector<uint32_t>& c, size_t i, size_t end) { return opt_ws_run(c, i, end, ds_punct); }
size_t ds_trailing_ws(const std::vector<uint32_t>& c, size_t i, size_t end) {
    return WS(c[i]) && run(c, i, end, WS) == end ? end : i;
}
size_t ds_cjk(const std::vector<uint32_t>& c, size_t i, size_t end) { return run(c, i, end, cjk_hangul); }
size_t ds_digits(const std::vector<uint32_t>& c, size_t i, size_t end) { return run(c, i, end, N); }
// ---- deepseek-coder passes
size_t dc_letters(const std::vector<uint32_t>& c, size_t i, size_t end) { return opt_ws_run(c, i, end, L); }
size_t dc_puncts(const std::vector<uint32_t>& c, size_t i, size_t end) { return opt_ws_run(c, i, end, P); }
size_t dc_digit(const std::vector<uint32_t>& c, size_t i, size_t) { return N(c[i]) ? i + 1 : i; }
// ---- deepseek-v3 passes
size_t d3_digits(const std::vector<uint32_t>& c, size_t i, size_t end) {
    size_t j = i;
    while (j < end && j < i + 3 && N(c[j])) ++j;
    return j;
}
size_t d3_cjk(const std::vector<uint32_t>& c, size_t i, size_t end) { return run(c, i, end, cjk_kana); }
size_t d3_main(const std::vector<uint32_t>& c, size_t i, size_t end) {
    const uint32_t ch = c[i];
    // [!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+
    if (ascii_punct(ch) && i + 1 < end && ascii_alpha(c[i + 1])) return run(c, i + 1, end, ascii_alpha);
    // [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
    auto lm = [](uint32_t x) { return L(x) || M(x); };
    if (!CRLF(ch) && !L(ch) && !P(ch) && !S(ch) && i + 1 < end && lm(c[i + 1])) return run(c, i + 1, end, lm);
    if (lm(ch)) return run(c, i, end, lm);
    //  ?[\p{P}\p{S}]+[\r\n]*
    auto ps = [](uint32_t x) { return P(x) || S(x); };
    if (ch == ' ' && i + 1 < end && ps(c[i + 1])) return run(c, run(c, i + 1, end, ps), end, CRLF);
    if (ps(ch)) return run(c, run(c, i, end, ps), end, CRLF);
    return ws_tail(c, i, end);
}
// ---- tekken
bool tk_upper(uint32_t x) { return L(x) && !(x >= 'a' && x <= 'z'); }  // (?=[\p{L}])[^a-z]
bool tk_lower(uint32_t x) { return L(x) && !(x >= 'A' && x <= 'Z'); }  // (?=[\p{L}])[^A-Z]
bool tk_prefix(uint32_t x) { return !CRLF(x) && !L(x) && !N(x); }
bool tk_other(uint32_t x) { return !WS(x) && !L(x) && !N(x); }
// U* W+ starting at s (backtracking U* until W+ can start)
size_t tk_upper_star_lower_plus(const std::vector<uint32_t>& c, size_t s, size_t end) {
    const size_t u = run(c, s, end, tk_upper);
    for (size_t k = u + 1; k-- > s;)
        if (k < end && tk_lower(c[k])) return run(c, k, end, tk_lower);
    return s;
}
size_t tk_main(const std::vector<uint32_t>& c, size_t i, size_t end) {
    const uint32_t ch = c[i];
    const bool pre = tk_prefix(ch) && i + 1 < end;
    // [prefix]? U* W+
    if (pre) {
        const size_t m = tk_upper_star_lower_plus(c, i + 1, end);
        if (m > i + 1) return m;
    }
    {
        const size_t m = tk_upper_star_lower_plus(c, i, end);
        if (m > i) return m;
    }
    // [prefix]? U+ W*
    if (pre && tk_upper(c[i + 1])) return run(c, run(c, i + 1, end, tk_upper), end, tk_lower);
    if (tk_upper(ch)) return run(c, run(c, i, end, tk_upper), end, tk_lower);
    if (N(ch)) return i + 1;  // \p{N}
    //  ?[^\s\p{L}\p{N}]+[\r\n/]*
    auto tail = [](uint32_t x) { return CRLF(x) || x == '/'; };
    if (ch == ' ' && i + 1 < end && tk_other(c[i + 1])) return run(c, run(c, i + 1, end, tk_other), end, tail);
    if (tk_other(ch)) return run(c, run(c, i, end, tk_other), end, tail);
    return ws_tail(c, i, end);
}

struct Chunk { size_t a, b; };  // code point range

std::vector<Chunk> refine(const std::vector<uint32_t>& c, const std::vector<Chunk>& in, Match m) {
    std::vector<Chunk> out;
    for (const Chunk& ch : in) {
        size_t pos = ch.a, i = ch.a;
        while (i < ch.b) {
            const size_t e = m(c, i, ch.b);
            if (e > i) {
                if (i > pos) out.push_back({pos, i});
                out.push_back({i, e});
                pos = i = e;
            } else {
                ++i;
            }
        }
        if (pos < ch.b) out.push_back({pos, ch.b});
    }
    return out;
}

} // namespace

uint8_t uc_class(uint32_t cp) {
    if (cp < 0x80) {  // ASCII fast path
        if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return 1;
        if (cp >= '0' && cp <= '9') return 3;
        if (cp == ' ') return 6;
        if (cp < 0x20 || cp == 0x7F) return 0;
        return (cp == '$' || cp == '+' || cp == '<' || cp == '=' || cp == '>' || cp == '^' || cp == '`' || cp == '|' || cp == '~') ? 5 : 4;
    }
    uint8_t cls = 0;
    return in_table(kUcClass, sizeof(kUcClass) / sizeof(UcRange), cp, &cls) ? cls : 0;
}

bool uc_white_space(uint32_t cp) {
    return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

std::vector<std::string> split(std::string_view text, bpe::PreTok family) {
    std::vector<std::string> out;
    if (text.empty()) return out;
    if (family == bpe::PreTok::Spm) {
        out.emplace_back(text);
        return out;
    }
    const Cps c = decode(text);
    std::vector<Match> passes;
    switch (family) {
        case bpe::PreTok::DeepSeekLlm: passes = {ds_crlf, ds_letters, ds_puncts, ds_trailing_ws, ds_cjk, ds_digits}; break;
        case bpe::PreTok::DeepSeekCoder: passes = {ds_crlf, dc_letters, dc_puncts, ds_cjk, dc_digit}; break;
        case bpe::PreTok::DeepSeekV3: passes = {d3_digits, d3_cjk, d3_main}; break;
        case bpe::PreTok::Tekken: passes = {tk_main}; break;
        default: return bpe::Tokenizer::pretokenize(text, family);
    }
    std::vector<Chunk> chunks{{0, c.cp.size()}};
    for (Match m : passes) chunks = refine(c.cp, chunks, m);
    for (const Chunk& ch : chunks) out.emplace_back(text.substr(c.off[ch.a], c.off[ch.b] - c.off[ch.a]));
    return out;
}

std::string spm_escape(std::string_view text, bool add_space_prefix) {
    static const char kSpace[] = "\xE2\x96\x81";
    std::string out;
    out.reserve(text.size() + 8);
    if (add_space_prefix) out += kSpace;
    for (char ch : text) {
        if (ch == ' ') out += kSpace;
        else out += ch;
    }
    return out;
}

void spm_encode(std::string_view escaped, const std::unordered_map<std::string, int32_t>& ids,
                const std::vector<float>& scores, int32_t unk, std::vector<int32_t>& out) {
    struct Sym { size_t off, len; int prev, next; };
    std::vector<Sym> sym;
    const Cps c = decode(escaped);
    for (size_t k = 0; k < c.cp.size(); ++k)
        sym.push_back({c.off[k], c.off[k + 1] - c.off[k], static_cast<int>(k) - 1,
                       k + 1 < c.cp.size() ? static_cast<int>(k + 1) : -1});
    struct Bigram { float score; int left; size_t len; };
    auto worse = [](const Bigram& a, const Bigram& b) { return a.score < b.score || (a.score == b.score && a.left > b.left); };
    std::priority_queue<Bigram, std::vector<Bigram>, decltype(worse)> q(worse);
    auto try_add = [&](int l, int r) {
        if (l < 0 || r < 0) return;
        const std::string s(escaped.substr(sym[l].off, sym[l].len + sym[r].len));
        const auto it = ids.find(s);
        if (it == ids.end() || static_cast<size_t>(it->second) >= scores.size()) return;
        q.push({scores[it->second], l, s.size()});
    };
    for (size_t k = 1; k < sym.size(); ++k) try_add(static_cast<int>(k) - 1, static_cast<int>(k));
    while (!q.empty()) {
        const Bigram b = q.top();
        q.pop();
        Sym& l = sym[b.left];
        if (l.len == 0 || l.next < 0) continue;
        Sym& r = sym[l.next];
        if (l.len + r.len != b.len) continue;  // stale: one side already merged
        l.len += r.len;
        r.len = 0;
        l.next = r.next;
        if (r.next >= 0) sym[r.next].prev = b.left;
        try_add(l.prev, b.left);
        try_add(b.left, l.next);
    }
    for (int k = sym.empty() ? -1 : 0; k >= 0; k = sym[k].next) {
        const std::string s(escaped.substr(sym[k].off, sym[k].len));
        const auto it = ids.find(s);
        if (it != ids.end()) { out.push_back(it->second); continue; }
        for (unsigned char byte : s) {  // byte fallback
            char name[8];
            std::snprintf(name, sizeof name, "<0x%02X>", byte);
            const auto bt = ids.find(name);
            out.push_back(bt != ids.end() ? bt->second : unk);
        }
    }
}

} // namespace vegajet::pretok
