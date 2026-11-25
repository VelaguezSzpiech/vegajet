// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/grammar.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <set>

namespace vegajet::grammar {
namespace {

using Elem = Grammar::Elem;

inline bool has(const ByteSet& s, unsigned b) { return (s[b >> 6] >> (b & 63)) & 1u; }
inline void put(ByteSet& s, unsigned b) { s[b >> 6] |= uint64_t{1} << (b & 63); }
inline ByteSet one(unsigned b) {
    ByteSet s{};
    put(s, b);
    return s;
}
inline ByteSet range(unsigned lo, unsigned hi) {
    ByteSet s{};
    for (unsigned b = lo; b <= hi; ++b) put(s, b);
    return s;
}

void utf8_append(std::string& o, uint32_t cp) {
    if (cp < 0x80) {
        o += static_cast<char>(cp);
    } else if (cp < 0x800) {
        o += static_cast<char>(0xC0 | (cp >> 6));
        o += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        o += static_cast<char>(0xE0 | (cp >> 12));
        o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        o += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        o += static_cast<char>(0xF0 | (cp >> 18));
        o += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        o += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

// ---------------------------------------------------------------------------------------------------------------
// GBNF parser
// ---------------------------------------------------------------------------------------------------------------

constexpr uint32_t kMaxRepeat = 4096;

struct Builder {
    std::string_view s;
    size_t i = 0;
    std::string err;
    std::vector<std::vector<std::vector<Elem>>> rules;  // rule -> alternatives -> elements (no end marker)
    std::vector<std::string> names;
    std::vector<uint8_t> defined;
    std::unordered_map<std::string, uint32_t> ids;
    std::vector<ByteSet> classes;
    std::map<ByteSet, uint32_t> class_ids;
    int64_t multi_rule = -1, any_rule = -1;

    bool fail(const std::string& m) {
        if (err.empty()) {
            size_t line = 1;
            for (size_t k = 0; k < i && k < s.size(); ++k) line += s[k] == '\n';
            err = m + " (line " + std::to_string(line) + ")";
        }
        return false;
    }
    uint32_t rule_id(const std::string& n) {
        auto it = ids.find(n);
        if (it != ids.end()) return it->second;
        const uint32_t id = static_cast<uint32_t>(rules.size());
        rules.emplace_back();
        names.push_back(n);
        defined.push_back(0);
        ids.emplace(n, id);
        return id;
    }
    uint32_t helper(uint32_t parent, std::vector<std::vector<Elem>> alts) {
        std::string n = names[parent] + "_" + std::to_string(rules.size());
        const uint32_t id = rule_id(n);
        rules[id] = std::move(alts);
        defined[id] = 1;
        return id;
    }
    Elem cls(const ByteSet& b) {
        auto it = class_ids.find(b);
        if (it == class_ids.end()) {
            it = class_ids.emplace(b, static_cast<uint32_t>(classes.size())).first;
            classes.push_back(b);
        }
        return Elem{1, it->second};
    }
    static Elem ref(uint32_t r) { return Elem{2, r}; }
    uint32_t multibyte_rule() {
        if (multi_rule < 0) {
            const ByteSet c = range(0x80, 0xBF);
            std::vector<std::vector<Elem>> a;
            a.push_back({cls(range(0xC2, 0xDF)), cls(c)});
            a.push_back({cls(range(0xE0, 0xEF)), cls(c), cls(c)});
            a.push_back({cls(range(0xF0, 0xF4)), cls(c), cls(c), cls(c)});
            const uint32_t id = rule_id("__utf8_multibyte");
            rules[id] = std::move(a);
            defined[id] = 1;
            multi_rule = id;
        }
        return static_cast<uint32_t>(multi_rule);
    }
    uint32_t any_char_rule() {
        if (any_rule < 0) {
            const uint32_t m = multibyte_rule();
            const uint32_t id = rule_id("__any_char");
            rules[id] = {{cls(range(0x00, 0x7F))}, {ref(m)}};
            defined[id] = 1;
            any_rule = id;
        }
        return static_cast<uint32_t>(any_rule);
    }

    bool eof() const { return i >= s.size(); }
    void skip_ws() {
        while (!eof()) {
            const char c = s[i];
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
                ++i;
            } else if (c == '#') {
                while (!eof() && s[i] != '\n') ++i;
            } else {
                break;
            }
        }
    }
    static bool word_char(char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_';
    }
    size_t word_end(size_t k) const {
        while (k < s.size() && word_char(s[k])) ++k;
        return k;
    }
    bool at_rule_start() const {
        size_t k = word_end(i);
        if (k == i) return false;
        while (k < s.size() && (s[k] == ' ' || s[k] == '\t')) ++k;
        return s.substr(k, 3) == "::=";
    }

    // One code point from a literal or class; escapes per llama.cpp (\xHH etc. are code points).
    bool code_point(uint32_t& cp) {
        if (eof()) return fail("unexpected end of grammar");
        const auto c = static_cast<unsigned char>(s[i]);
        if (c == '\\') {
            ++i;
            if (eof()) return fail("unexpected end after backslash");
            const char e = s[i++];
            auto hex = [&](int n) {
                cp = 0;
                for (int k = 0; k < n; ++k) {
                    if (eof()) return fail("short hex escape");
                    const char h = s[i++];
                    cp <<= 4;
                    if (h >= '0' && h <= '9') cp |= static_cast<uint32_t>(h - '0');
                    else if (h >= 'a' && h <= 'f') cp |= static_cast<uint32_t>(h - 'a' + 10);
                    else if (h >= 'A' && h <= 'F') cp |= static_cast<uint32_t>(h - 'A' + 10);
                    else return fail("bad hex escape");
                }
                if (cp > 0x10FFFF) return fail("code point out of range");
                return true;
            };
            switch (e) {
                case 'n': cp = '\n'; return true;
                case 'r': cp = '\r'; return true;
                case 't': cp = '\t'; return true;
                case '0': cp = 0; return true;
                case 'x': return hex(2);
                case 'u': return hex(4);
                case 'U': return hex(8);
                case '\\': case '"': case '\'': case '[': case ']': case '-': case '^': case '/':
                    cp = static_cast<uint32_t>(e);
                    return true;
                default: return fail(std::string("unknown escape \\") + e);
            }
        }
        // raw UTF-8
        int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        if (n == 0 || i + static_cast<size_t>(n) > s.size()) return fail("invalid UTF-8 in grammar");
        cp = n == 1 ? c : n == 2 ? (c & 0x1Fu) : n == 3 ? (c & 0x0Fu) : (c & 0x07u);
        for (int k = 1; k < n; ++k) {
            const auto d = static_cast<unsigned char>(s[i + static_cast<size_t>(k)]);
            if ((d >> 6) != 2) return fail("invalid UTF-8 in grammar");
            cp = (cp << 6) | (d & 0x3Fu);
        }
        i += static_cast<size_t>(n);
        return true;
    }

    bool literal(std::vector<Elem>& out) {
        ++i;  // opening quote
        std::string bytes;
        while (true) {
            if (eof()) return fail("unterminated literal");
            if (s[i] == '"') {
                ++i;
                break;
            }
            uint32_t cp = 0;
            if (!code_point(cp)) return false;
            utf8_append(bytes, cp);
        }
        for (unsigned char b : bytes) out.push_back(cls(one(b)));
        return true;
    }

    bool char_class(uint32_t rule, std::vector<Elem>& out) {
        ++i;  // [
        bool neg = false;
        if (!eof() && s[i] == '^') {
            neg = true;
            ++i;
        }
        ByteSet ascii{};
        std::vector<std::pair<uint32_t, uint32_t>> wide;
        while (true) {
            if (eof()) return fail("unterminated character class");
            if (s[i] == ']') {
                ++i;
                break;
            }
            uint32_t lo = 0, hi = 0;
            if (!code_point(lo)) return false;
            hi = lo;
            if (!eof() && s[i] == '-' && i + 1 < s.size() && s[i + 1] != ']') {
                ++i;
                if (!code_point(hi)) return false;
                if (hi < lo) return fail("reversed class range");
            }
            for (uint32_t c = lo; c <= std::min<uint32_t>(hi, 0x7F); ++c) put(ascii, c);
            if (hi >= 0x80) wide.emplace_back(std::max<uint32_t>(lo, 0x80), hi);
        }
        if (neg) {
            if (!wide.empty()) return fail("negated class with non-ASCII members is not supported");
            ByteSet allowed{};
            for (unsigned c = 0; c < 0x80; ++c)
                if (!has(ascii, c)) put(allowed, c);
            std::vector<std::vector<Elem>> alts;
            alts.push_back({cls(allowed)});
            alts.push_back({ref(multibyte_rule())});
            out.push_back(ref(helper(rule, std::move(alts))));
            return true;
        }
        if (wide.empty()) {
            if (ascii == ByteSet{}) return fail("empty character class");
            out.push_back(cls(ascii));
            return true;
        }
        std::vector<std::vector<Elem>> alts;
        if (ascii != ByteSet{}) alts.push_back({cls(ascii)});
        for (auto [lo, hi] : wide) {
            if (hi - lo >= 256) return fail("non-ASCII class range wider than 256 code points");
            for (uint32_t c = lo; c <= hi; ++c) {
                std::string b;
                utf8_append(b, c);
                std::vector<Elem> seq;
                for (unsigned char x : b) seq.push_back(cls(one(x)));
                alts.push_back(std::move(seq));
            }
        }
        out.push_back(ref(helper(rule, std::move(alts))));
        return true;
    }

    bool number(uint32_t& v) {
        if (eof() || s[i] < '0' || s[i] > '9') return fail("expected a number in repetition");
        uint64_t x = 0;
        while (!eof() && s[i] >= '0' && s[i] <= '9') {
            x = x * 10 + static_cast<uint64_t>(s[i++] - '0');
            if (x > kMaxRepeat) return fail("repetition count too large");
        }
        v = static_cast<uint32_t>(x);
        return true;
    }

    void star(uint32_t rule, const std::vector<Elem>& item, std::vector<Elem>& out) {
        const uint32_t h = rule_id(names[rule] + "_" + std::to_string(rules.size()));
        std::vector<Elem> rec = item;
        rec.push_back(ref(h));
        rules[h] = {rec, {}};
        defined[h] = 1;
        out.push_back(ref(h));
    }

    bool quantify(uint32_t rule, std::vector<Elem>& item, std::vector<Elem>& out) {
        if (eof()) {
            out.insert(out.end(), item.begin(), item.end());
            return true;
        }
        const char q = s[i];
        if (q == '*') {
            ++i;
            star(rule, item, out);
        } else if (q == '+') {
            ++i;
            out.insert(out.end(), item.begin(), item.end());
            star(rule, item, out);
        } else if (q == '?') {
            ++i;
            out.push_back(ref(helper(rule, {item, {}})));
        } else if (q == '{') {
            ++i;
            uint32_t m = 0, n = 0;
            bool unbounded = false;
            skip_ws();
            if (!number(m)) return false;
            skip_ws();
            if (!eof() && s[i] == ',') {
                ++i;
                skip_ws();
                if (!eof() && s[i] == '}') {
                    unbounded = true;
                } else if (!number(n)) {
                    return false;
                }
            } else {
                n = m;
            }
            skip_ws();
            if (eof() || s[i] != '}') return fail("expected } in repetition");
            ++i;
            if (!unbounded && n < m) return fail("repetition {m,n} with n < m");
            for (uint32_t k = 0; k < m; ++k) out.insert(out.end(), item.begin(), item.end());
            if (unbounded) {
                star(rule, item, out);
            } else if (n > m) {
                uint32_t prev = helper(rule, {item, {}});
                for (uint32_t k = 1; k < n - m; ++k) {
                    std::vector<Elem> seq = item;
                    seq.push_back(ref(prev));
                    prev = helper(rule, {seq, {}});
                }
                out.push_back(ref(prev));
            }
        } else {
            out.insert(out.end(), item.begin(), item.end());
        }
        return true;
    }

    bool alternatives(uint32_t rule, bool nested, std::vector<std::vector<Elem>>& alts) {
        while (true) {
            std::vector<Elem> seq;
            if (!sequence(rule, nested, seq)) return false;
            alts.push_back(std::move(seq));
            skip_ws();
            if (!eof() && s[i] == '|') {
                ++i;
                continue;
            }
            return true;
        }
    }

    bool sequence(uint32_t rule, bool nested, std::vector<Elem>& seq) {
        while (true) {
            skip_ws();
            if (eof()) return true;
            const char c = s[i];
            if (c == '|' || c == ')') return true;
            if (!nested && at_rule_start()) return true;
            std::vector<Elem> item;
            if (c == '"') {
                if (!literal(item)) return false;
            } else if (c == '[') {
                if (!char_class(rule, item)) return false;
            } else if (c == '.') {
                ++i;
                item.push_back(ref(any_char_rule()));
            } else if (c == '(') {
                ++i;
                std::vector<std::vector<Elem>> alts;
                if (!alternatives(rule, true, alts)) return false;
                skip_ws();
                if (eof() || s[i] != ')') return fail("expected )");
                ++i;
                if (alts.size() == 1) item = std::move(alts[0]);
                else item.push_back(ref(helper(rule, std::move(alts))));
            } else if (word_char(c)) {
                const size_t e = word_end(i);
                item.push_back(ref(rule_id(std::string(s.substr(i, e - i)))));
                i = e;
            } else {
                return fail(std::string("unexpected character '") + c + "'");
            }
            if (!quantify(rule, item, seq)) return false;
        }
    }

    bool parse_all() {
        skip_ws();
        while (!eof()) {
            const size_t e = word_end(i);
            if (e == i) return fail("expected a rule name");
            const std::string name(s.substr(i, e - i));
            i = e;
            while (!eof() && (s[i] == ' ' || s[i] == '\t')) ++i;
            if (s.substr(i, 3) != "::=") return fail("expected ::= after " + name);
            i += 3;
            const uint32_t id = rule_id(name);
            if (defined[id]) return fail("rule " + name + " defined twice");
            defined[id] = 1;
            std::vector<std::vector<Elem>> alts;
            if (!alternatives(id, false, alts)) return false;
            rules[id] = std::move(alts);
            skip_ws();
        }
        return true;
    }
};

bool check_left_recursion(const Grammar& g, std::string& err) {
    const size_t n = g.alts.size();
    std::vector<uint8_t> nullable(n, 0);
    for (bool changed = true; changed;) {
        changed = false;
        for (size_t r = 0; r < n; ++r) {
            if (nullable[r]) continue;
            for (uint32_t a : g.alts[r]) {
                bool ok = true;
                for (uint32_t p = a; g.elems[p].kind != 0; ++p)
                    if (g.elems[p].kind == 1 || !nullable[g.elems[p].arg]) {
                        ok = false;
                        break;
                    }
                if (ok) {
                    nullable[r] = 1;
                    changed = true;
                    break;
                }
            }
        }
    }
    std::vector<std::vector<uint32_t>> left(n);
    for (size_t r = 0; r < n; ++r)
        for (uint32_t a : g.alts[r])
            for (uint32_t p = a; g.elems[p].kind == 2; ++p) {
                left[r].push_back(g.elems[p].arg);
                if (!nullable[g.elems[p].arg]) break;
            }
    std::vector<uint8_t> color(n, 0);
    // iterative DFS
    for (size_t start = 0; start < n; ++start) {
        if (color[start]) continue;
        std::vector<std::pair<uint32_t, size_t>> st{{static_cast<uint32_t>(start), 0}};
        color[start] = 1;
        while (!st.empty()) {
            auto& [r, k] = st.back();
            if (k < left[r].size()) {
                const uint32_t nx = left[r][k++];
                if (color[nx] == 1) {
                    err = "left recursion in rule " + g.names[nx];
                    return false;
                }
                if (color[nx] == 0) {
                    color[nx] = 1;
                    st.emplace_back(nx, 0);
                }
            } else {
                color[r] = 2;
                st.pop_back();
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// JSON Schema -> GBNF
// ---------------------------------------------------------------------------------------------------------------

std::string gbnf_literal(std::string_view bytes) {
    std::string o = "\"";
    for (unsigned char c : bytes) {
        if (c == '"') o += "\\\"";
        else if (c == '\\') o += "\\\\";
        else if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t";
        else if (c < 0x20 || c == 0x7F) {
            char b[8];
            std::snprintf(b, sizeof b, "\\x%02X", c);
            o += b;
        } else {
            o += static_cast<char>(c);
        }
    }
    return o + "\"";
}

void dump_compact(const json::Value& v, std::string& o) {
    switch (v.type) {
        case json::Value::Type::Null: o += "null"; break;
        case json::Value::Type::Bool: o += v.b ? "true" : "false"; break;
        case json::Value::Type::Number: {
            char b[40];
            if (std::isfinite(v.num) && v.num == std::floor(v.num) && std::fabs(v.num) < 1e15)
                std::snprintf(b, sizeof b, "%lld", static_cast<long long>(v.num));
            else
                std::snprintf(b, sizeof b, "%.17g", v.num);
            o += b;
            break;
        }
        case json::Value::Type::String: o += json::quote(v.str); break;
        case json::Value::Type::Array:
            o += '[';
            for (size_t k = 0; k < v.arr.size(); ++k) {
                if (k) o += ',';
                dump_compact(v.arr[k], o);
            }
            o += ']';
            break;
        case json::Value::Type::Object:
            o += '{';
            for (size_t k = 0; k < v.obj.size(); ++k) {
                if (k) o += ',';
                o += json::quote(v.obj[k].first);
                o += ':';
                dump_compact(v.obj[k].second, o);
            }
            o += '}';
            break;
    }
}

const char* const kPrims[][2] = {
    {"ws", "| \" \" | \"\\n\" [ \\t]{0,20}"},
    {"char", "[^\"\\\\\\x7F\\x00-\\x1F] | \"\\\\\" ([\"\\\\/bfnrt] | \"u\" [0-9a-fA-F]{4})"},
    {"string", "\"\\\"\" char* \"\\\"\""},
    {"integer", "\"-\"? (\"0\" | [1-9] [0-9]{0,15})"},
    {"number", "\"-\"? (\"0\" | [1-9] [0-9]{0,15}) (\".\" [0-9]{1,16})? ([eE] [-+]? [0-9]{1,3})?"},
    {"boolean", "\"true\" | \"false\""},
    {"null", "\"null\""},
    {"value", "object | array | string | number | boolean | null"},
    {"object", "\"{\" ws ( string \":\" ws value ws ( \",\" ws string \":\" ws value ws )* )? \"}\""},
    {"array", "\"[\" ws ( value ws ( \",\" ws value ws )* )? \"]\""},
};

class SchemaConverter {
public:
    explicit SchemaConverter(const json::Value& root) : root_(root) {
        for (const auto& p : kPrims) taken_.insert(p[0]);
        taken_.insert("root");
    }

    std::string run(std::string* err) {
        const std::string body = visit(root_, "schema", 0);
        if (!err_.empty()) {
            if (err) *err = err_;
            return {};
        }
        std::string o = "root ::= " + body + " ws\n";
        for (const auto& [n, b] : rules_) o += n + " ::= " + b + "\n";
        // primitives (with their dependencies)
        std::set<std::string> need = prims_;
        need.insert("ws");
        if (need.count("value")) need.insert({"object", "array", "string", "number", "boolean", "null"});
        if (need.count("object")) need.insert({"string", "value"});
        if (need.count("array")) need.insert("value");
        if (need.count("string")) need.insert("char");
        for (const auto& p : kPrims)
            if (need.count(p[0])) o += std::string(p[0]) + " ::= " + p[1] + "\n";
        return o;
    }

private:
    std::string fresh(const std::string& hint) {
        std::string n;
        for (char c : hint) n += (std::isalnum(static_cast<unsigned char>(c)) || c == '-') ? c : '-';
        if (n.empty()) n = "r";
        std::string cand = n;
        for (int k = 1; taken_.count(cand); ++k) cand = n + "-" + std::to_string(k);
        taken_.insert(cand);
        return cand;
    }
    std::string rule(const std::string& hint, const std::string& body) {
        auto it = by_body_.find(body);
        if (it != by_body_.end()) return it->second;
        const std::string n = fresh(hint);
        rules_.emplace_back(n, body);
        by_body_.emplace(body, n);
        return n;
    }
    std::string prim(const std::string& p) {
        prims_.insert(p);
        return p;
    }
    std::string fail(const std::string& m) {
        if (err_.empty()) err_ = m;
        return "null";
    }
    static std::string alt(const std::vector<std::string>& xs) {
        if (xs.size() == 1) return xs[0];
        std::string o = "(";
        for (size_t k = 0; k < xs.size(); ++k) o += (k ? " | " : " ") + xs[k];
        return o + " )";
    }
    static std::string rep(const std::string& item, uint64_t lo, uint64_t hi, bool unbounded) {
        if (unbounded) return lo == 0 ? item + "*" : lo == 1 ? item + "+" : item + "{" + std::to_string(lo) + ",}";
        if (lo == hi) return item + "{" + std::to_string(lo) + "}";
        return item + "{" + std::to_string(lo) + "," + std::to_string(hi) + "}";
    }
    static bool count(const json::Value& s, const char* key, uint64_t& out) {
        const json::Value* v = s.get(key);
        if (!v || !v->is_number() || v->num < 0) return false;
        out = static_cast<uint64_t>(std::min(v->num, double{kMaxRepeat}));
        return true;
    }

    std::string literal_of(const json::Value& v) {
        std::string d;
        dump_compact(v, d);
        return gbnf_literal(d);
    }

    std::string resolve_ref(const std::string& ref, int depth) {
        auto it = refs_.find(ref);
        if (it != refs_.end()) return it->second;
        const json::Value* target = nullptr;
        std::string label;
        if (ref == "#") {
            target = &root_;
            label = "root-ref";
        } else {
            for (const char* pfx : {"#/$defs/", "#/definitions/"}) {
                if (ref.rfind(pfx, 0) == 0) {
                    const std::string name = ref.substr(std::strlen(pfx));
                    const json::Value* defs = root_.get(std::string(pfx).substr(2, std::strlen(pfx) - 3));
                    if (defs) target = defs->get(name);
                    label = "def-" + name;
                }
            }
        }
        if (!target) return fail("unresolvable $ref " + ref);
        const std::string n = fresh(label);
        refs_.emplace(ref, n);
        const size_t slot = rules_.size();
        rules_.emplace_back(n, "");
        const std::string body = visit(*target, n + "-v", depth + 1);
        rules_[slot].second = body;
        return n;
    }

    std::string visit_object(const json::Value& s, const std::string& hint, int depth) {
        const json::Value* props = s.get("properties");
        const json::Value* addl = s.get("additionalProperties");
        if (!props || !props->is_object() || props->obj.empty()) {
            if (addl && addl->is_object()) {
                const std::string v = visit(*addl, hint + "-value", depth + 1);
                const std::string kv = prim("string") + " \":\" ws " + v + " ws";
                return rule(hint, "\"{\" ws ( " + kv + " ( \",\" ws " + kv + " )* )? \"}\"");
            }
            return prim("object");
        }
        std::set<std::string> required;
        if (const json::Value* r = s.get("required"); r && r->is_array())
            for (const auto& x : r->arr)
                if (x.is_string()) required.insert(x.str);
        std::vector<std::string> req, opt;
        for (const auto& [name, sub] : props->obj) {
            const std::string v = visit(sub, hint + "-" + name, depth + 1);
            const std::string kv = rule(hint + "-" + name + "-kv", gbnf_literal(json::quote(name)) + " \":\" ws " + v + " ws");
            (required.count(name) ? req : opt).push_back(kv);
        }
        std::string extra;
        if (addl && addl->is_object()) {
            const std::string v = visit(*addl, hint + "-extra", depth + 1);
            extra = " ( \",\" ws " + prim("string") + " \":\" ws " + v + " ws )*";
        }
        std::string body = "\"{\" ws";
        if (!req.empty()) {
            for (size_t k = 0; k < req.size(); ++k) body += (k ? " \",\" ws " : " ") + req[k];
            for (const auto& o : opt) body += " ( \",\" ws " + o + " )?";
            body += extra;
        } else {
            std::vector<std::string> starts;
            for (size_t k = 0; k < opt.size(); ++k) {
                std::string a = opt[k];
                for (size_t j = k + 1; j < opt.size(); ++j) a += " ( \",\" ws " + opt[j] + " )?";
                starts.push_back(a + extra);
            }
            body += " ( " + [&] {
                std::string o;
                for (size_t k = 0; k < starts.size(); ++k) o += (k ? " | " : "") + starts[k];
                return o;
            }() + " )?";
        }
        return rule(hint, body + " \"}\"");
    }

    std::string visit_array(const json::Value& s, const std::string& hint, int depth) {
        const json::Value* items = s.get("items");
        const std::string item = (items && (items->is_object() || items->is_bool()))
                                     ? visit(*items, hint + "-item", depth + 1)
                                     : prim("value");
        uint64_t lo = 0, hi = 0;
        count(s, "minItems", lo);
        const bool unbounded = !count(s, "maxItems", hi);
        if (!unbounded && hi < lo) return fail("maxItems < minItems");
        if (!items && lo == 0 && unbounded) return prim("array");
        if (!unbounded && hi == 0) return rule(hint, "\"[\" ws \"]\"");
        const std::string first = item + " ws";
        const std::string more = "( \",\" ws " + item + " ws )";
        std::string inner;
        if (lo >= 1) inner = first + " " + rep(more, lo - 1, unbounded ? 0 : hi - 1, unbounded);
        else inner = "( " + first + " " + rep(more, 0, unbounded ? 0 : hi - 1, unbounded) + " )?";
        return rule(hint, "\"[\" ws " + inner + " \"]\"");
    }

    std::string visit_type(const json::Value& s, const std::string& t, const std::string& hint, int depth) {
        if (t == "string") {
            uint64_t lo = 0, hi = 0;
            const bool has_lo = count(s, "minLength", lo);
            const bool has_hi = count(s, "maxLength", hi);
            if (!has_lo && !has_hi) return prim("string");
            if (has_hi && hi < lo) return fail("maxLength < minLength");
            prim("char");
            return rule(hint, "\"\\\"\" " + rep("char", lo, hi, !has_hi) + " \"\\\"\"");
        }
        if (t == "number") return prim("number");
        if (t == "integer") return prim("integer");
        if (t == "boolean") return prim("boolean");
        if (t == "null") return prim("null");
        if (t == "object") return visit_object(s, hint, depth);
        if (t == "array") return visit_array(s, hint, depth);
        return fail("unsupported type " + t);
    }

    std::string visit(const json::Value& s, const std::string& hint, int depth) {
        if (depth > 64) return fail("schema nesting too deep");
        if (s.is_bool()) return s.b ? prim("value") : fail("schema false matches nothing");
        if (!s.is_object()) return fail("schema must be an object");
        if (const json::Value* r = s.get("$ref"); r && r->is_string()) return resolve_ref(r->str, depth);
        if (const json::Value* c = s.get("const")) return literal_of(*c);
        if (const json::Value* e = s.get("enum"); e && e->is_array()) {
            if (e->arr.empty()) return fail("empty enum");
            std::vector<std::string> xs;
            for (const auto& v : e->arr) xs.push_back(literal_of(v));
            return alt(xs);
        }
        for (const char* k : {"anyOf", "oneOf"}) {
            if (const json::Value* a = s.get(k); a && a->is_array() && !a->arr.empty()) {
                std::vector<std::string> xs;
                for (size_t j = 0; j < a->arr.size(); ++j)
                    xs.push_back(visit(a->arr[j], hint + "-" + std::to_string(j), depth + 1));
                return alt(xs);
            }
        }
        if (const json::Value* a = s.get("allOf"); a && a->is_array() && a->arr.size() == 1)
            return visit(a->arr[0], hint, depth + 1);
        const json::Value* t = s.get("type");
        if (t && t->is_string()) return visit_type(s, t->str, hint, depth);
        if (t && t->is_array()) {
            std::vector<std::string> xs;
            for (const auto& x : t->arr)
                if (x.is_string()) xs.push_back(visit_type(s, x.str, hint + "-" + x.str, depth));
            if (xs.empty()) return fail("empty type list");
            return alt(xs);
        }
        if (s.get("properties")) return visit_object(s, hint, depth);
        if (s.get("items")) return visit_array(s, hint, depth);
        return prim("value");
    }

    const json::Value& root_;
    std::vector<std::pair<std::string, std::string>> rules_;
    std::map<std::string, std::string> by_body_;
    std::set<std::string> taken_, prims_;
    std::map<std::string, std::string> refs_;
    std::string err_;
};

inline int cont_bytes(unsigned lead) { return lead < 0xE0 ? 1 : lead < 0xF0 ? 2 : 3; }

} // namespace

// ---------------------------------------------------------------------------------------------------------------

std::shared_ptr<const Grammar> Grammar::parse(std::string_view gbnf, std::string* err, std::string_view root_name) {
    Builder b;
    b.s = gbnf;
    auto bad = [&](const std::string& m) -> std::shared_ptr<const Grammar> {
        if (err) *err = m;
        return nullptr;
    };
    if (!b.parse_all()) return bad(b.err);
    for (size_t r = 0; r < b.rules.size(); ++r)
        if (!b.defined[r]) return bad("undefined rule " + b.names[r]);
    auto it = b.ids.find(std::string(root_name));
    if (it == b.ids.end()) return bad("missing root rule " + std::string(root_name));
    auto g = std::make_shared<Grammar>();
    g->names = b.names;
    g->classes = b.classes;
    g->root = it->second;
    g->alts.resize(b.rules.size());
    for (size_t r = 0; r < b.rules.size(); ++r)
        for (const auto& a : b.rules[r]) {
            g->alts[r].push_back(static_cast<uint32_t>(g->elems.size()));
            g->elems.insert(g->elems.end(), a.begin(), a.end());
            g->elems.push_back(Elem{});
        }
    std::string lr;
    if (!check_left_recursion(*g, lr)) return bad(lr);
    return g;
}

std::string json_schema_to_gbnf(const json::Value& schema, std::string* err) {
    return SchemaConverter(schema).run(err);
}

std::string json_schema_to_gbnf(std::string_view schema_json, std::string* err) {
    json::Value v;
    if (!json::parse(schema_json, v)) {
        if (err) *err = "schema is not valid JSON";
        return {};
    }
    return json_schema_to_gbnf(v, err);
}

// ---------------------------------------------------------------------------------------------------------------
// TokenTable
// ---------------------------------------------------------------------------------------------------------------

TokenTable::TokenTable(std::vector<std::string> tokens, int32_t eos) : tok_(std::move(tokens)), eos_(eos) {
    std::vector<int32_t> order;
    order.reserve(tok_.size());
    for (size_t id = 0; id < tok_.size(); ++id)
        if (!tok_[id].empty() && static_cast<int32_t>(id) != eos_) order.push_back(static_cast<int32_t>(id));
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        const int c = tok_[static_cast<size_t>(a)].compare(tok_[static_cast<size_t>(b)]);
        return c != 0 ? c < 0 : a < b;
    });
    nodes_.emplace_back();
    // Build in DFS order; each node's children are contiguous in edges_.
    struct Frame {
        uint32_t node;
        size_t lo, hi, depth;
    };
    std::vector<Frame> work{{0, 0, order.size(), 0}};
    while (!work.empty()) {
        const Frame f = work.back();
        work.pop_back();
        size_t k = f.lo;
        nodes_[f.node].first_token = static_cast<uint32_t>(node_tokens_.size());
        while (k < f.hi && tok_[static_cast<size_t>(order[k])].size() == f.depth) node_tokens_.push_back(order[k++]);
        nodes_[f.node].n_tokens = static_cast<uint32_t>(node_tokens_.size()) - nodes_[f.node].first_token;
        std::vector<std::pair<size_t, size_t>> groups;
        while (k < f.hi) {
            const char c = tok_[static_cast<size_t>(order[k])][f.depth];
            size_t e = k + 1;
            while (e < f.hi && tok_[static_cast<size_t>(order[e])][f.depth] == c) ++e;
            groups.emplace_back(k, e);
            k = e;
        }
        nodes_[f.node].first_child = static_cast<uint32_t>(edges_.size());
        nodes_[f.node].n_children = static_cast<uint32_t>(groups.size());
        const size_t base = edges_.size();
        edges_.resize(base + groups.size());
        for (size_t gi = 0; gi < groups.size(); ++gi) {
            const uint32_t child = static_cast<uint32_t>(nodes_.size());
            nodes_.emplace_back();
            edges_[base + gi] = Edge{static_cast<uint8_t>(tok_[static_cast<size_t>(order[groups[gi].first])][f.depth]),
                                     child};
            work.push_back(Frame{child, groups[gi].first, groups[gi].second, f.depth + 1});
        }
    }
}

std::shared_ptr<const TokenTable::Region> TokenTable::region(const ByteSet& loop, const ByteSet& leads) const {
    std::lock_guard<std::mutex> lk(region_mu_);
    auto key = std::make_pair(loop, leads);
    auto it = regions_.find(key);
    if (it != regions_.end()) return it->second;
    auto r = std::make_shared<Region>();
    r->bits.assign(words(), 0);
    std::unordered_map<std::string, std::vector<int32_t>> tails;
    struct Item {
        uint32_t node;
        uint32_t depth;
        uint8_t need;  // continuation bytes still needed
    };
    std::vector<Item> st{{0u, 0u, uint8_t{0}}};
    std::vector<uint32_t> sub;
    while (!st.empty()) {
        const Item it2 = st.back();
        st.pop_back();
        const TrieNode& n = nodes_[it2.node];
        for (uint32_t t = 0; t < n.n_tokens; ++t) {
            const auto id = static_cast<uint32_t>(node_tokens_[n.first_token + t]);
            r->bits[id >> 6] |= uint64_t{1} << (id & 63);
        }
        for (uint32_t e = 0; e < n.n_children; ++e) {
            const Edge& ed = edges_[n.first_child + e];
            if (it2.need > 0) {
                if (ed.byte >= 0x80 && ed.byte <= 0xBF)
                    st.push_back(Item{ed.node, it2.depth + 1, static_cast<uint8_t>(it2.need - 1)});
            } else if (has(loop, ed.byte)) {
                st.push_back(Item{ed.node, it2.depth + 1, uint8_t{0}});
            } else if (has(leads, ed.byte)) {
                st.push_back(Item{ed.node, it2.depth + 1, static_cast<uint8_t>(cont_bytes(ed.byte))});
            } else {
                // every token below leaves the loop here: group by the remaining bytes
                sub.assign(1, ed.node);
                while (!sub.empty()) {
                    const TrieNode& c = nodes_[sub.back()];
                    sub.pop_back();
                    for (uint32_t t = 0; t < c.n_tokens; ++t) {
                        const int32_t id = node_tokens_[c.first_token + t];
                        tails[tok_[static_cast<size_t>(id)].substr(it2.depth)].push_back(id);
                    }
                    for (uint32_t k = 0; k < c.n_children; ++k) sub.push_back(edges_[c.first_child + k].node);
                }
            }
        }
    }
    r->exits.reserve(tails.size());
    for (auto& [tail, ids] : tails) r->exits.push_back(Region::Exit{tail, std::move(ids)});
    std::sort(r->exits.begin(), r->exits.end(),
              [](const Region::Exit& a, const Region::Exit& b) { return a.tail < b.tail; });
    if (regions_.size() >= 16) regions_.clear();
    regions_.emplace(key, r);
    return r;
}

// ---------------------------------------------------------------------------------------------------------------
// GrammarMatcher
// ---------------------------------------------------------------------------------------------------------------

GrammarMatcher::GrammarMatcher(std::shared_ptr<const Grammar> g, size_t mask_cache_entries, size_t max_dfa_states)
    : g_(std::move(g)), cache_cap_(std::max<size_t>(1, mask_cache_entries)),
      max_states_(std::max<size_t>(64, max_dfa_states)) {
    for (uint32_t a : g_->alts[g_->root]) {
        Stack work;
        if (g_->elems[a].kind != 0) work.push_back(a);
        expand(work, init_);
    }
    reset();
}

void GrammarMatcher::expand(Stack& work, std::vector<Stack>& out) const {
    if (out.size() > 65536) return;  // pathological ambiguity guard
    if (work.empty() || g_->elems[work.back()].kind == 1) {
        out.push_back(work);
        return;
    }
    const uint32_t p = work.back();
    const uint32_t r = g_->elems[p].arg;
    work.pop_back();
    const bool cont = g_->elems[p + 1].kind != 0;
    for (uint32_t a : g_->alts[r]) {
        const size_t n = work.size();
        if (cont) work.push_back(p + 1);
        if (g_->elems[a].kind != 0) work.push_back(a);
        expand(work, out);
        work.resize(n);
    }
    work.push_back(p);
}

int32_t GrammarMatcher::intern(std::vector<Stack>&& stacks) {
    std::sort(stacks.begin(), stacks.end());
    stacks.erase(std::unique(stacks.begin(), stacks.end()), stacks.end());
    std::string key;
    for (const Stack& s : stacks) {
        const auto n = static_cast<uint32_t>(s.size());
        key.append(reinterpret_cast<const char*>(&n), sizeof n);
        key.append(reinterpret_cast<const char*>(s.data()), s.size() * sizeof(uint32_t));
    }
    auto it = index_.find(key);
    if (it != index_.end()) return it->second;
    State st;
    for (const Stack& s : stacks) {
        if (s.empty()) st.accepting = true;
        else
            for (int w = 0; w < 4; ++w) st.first[static_cast<size_t>(w)] |= g_->classes[g_->elems[s.back()].arg][static_cast<size_t>(w)];
    }
    st.stacks = std::move(stacks);
    const auto id = static_cast<int32_t>(states_.size());
    states_.push_back(std::move(st));
    trans_.resize(trans_.size() + 256, -2);
    index_.emplace(std::move(key), id);
    ++st_.dfa_states;
    return id;
}

int32_t GrammarMatcher::step(int32_t s, uint8_t c) {
    const size_t at = static_cast<size_t>(s) * 256 + c;
    if (trans_[at] != -2) return trans_[at];
    if (!has(states_[static_cast<size_t>(s)].first, c)) return trans_[at] = -1;
    // Bytes with the same membership in every stack top class share the transition.
    std::vector<uint32_t> tops;
    for (const Stack& st : states_[static_cast<size_t>(s)].stacks)
        if (!st.empty()) tops.push_back(g_->elems[st.back()].arg);
    std::sort(tops.begin(), tops.end());
    tops.erase(std::unique(tops.begin(), tops.end()), tops.end());
    const bool share = tops.size() <= 64;
    auto sig = [&](unsigned b) {
        uint64_t v = 0;
        for (size_t k = 0; k < tops.size(); ++k) v |= uint64_t{has(g_->classes[tops[k]], b)} << k;
        return v;
    };
    std::vector<Stack> out;
    Stack work;
    for (const Stack& st : states_[static_cast<size_t>(s)].stacks) {
        if (st.empty()) continue;
        const uint32_t p = st.back();
        if (!has(g_->classes[g_->elems[p].arg], c)) continue;
        work.assign(st.begin(), st.end() - 1);
        if (g_->elems[p + 1].kind != 0) work.push_back(p + 1);
        expand(work, out);
    }
    const int32_t r = out.empty() ? -1 : intern(std::move(out));
    if (share) {
        const uint64_t mine = sig(c);
        for (unsigned b = 0; b < 256; ++b) {
            int32_t& t = trans_[static_cast<size_t>(s) * 256 + b];
            if (t == -2 && sig(b) == mine) t = r;
        }
    }
    trans_[at] = r;
    return r;
}

void GrammarMatcher::maybe_reset_dfa() {
    if (states_.size() <= max_states_) return;
    std::vector<Stack> cur;
    if (cur_ >= 0) cur = states_[static_cast<size_t>(cur_)].stacks;
    const bool alive = cur_ >= 0;
    states_.clear();
    trans_.clear();
    index_.clear();
    masks_.clear();
    cur_ = alive ? intern(std::move(cur)) : -1;
    ++st_.dfa_resets;
}

void GrammarMatcher::reset() {
    finished_ = false;
    maybe_reset_dfa();
    std::vector<Stack> init = init_;
    cur_ = init.empty() ? -1 : intern(std::move(init));
}

bool GrammarMatcher::can_end() const { return cur_ >= 0 && states_[static_cast<size_t>(cur_)].accepting; }

size_t GrammarMatcher::parse_stacks() const { return cur_ >= 0 ? states_[static_cast<size_t>(cur_)].stacks.size() : 0; }

bool GrammarMatcher::accept_bytes(std::string_view bytes) {
    if (finished_ || cur_ < 0) return false;
    maybe_reset_dfa();
    int32_t s = cur_;
    for (unsigned char c : bytes) {
        s = step(s, c);
        if (s < 0) return false;
    }
    cur_ = s;
    return true;
}

bool GrammarMatcher::accept_token(const TokenTable& vocab, int32_t id) {
    if (finished_ || cur_ < 0 || id < 0 || static_cast<size_t>(id) >= vocab.size()) return false;
    if (id == vocab.eos()) {
        if (!can_end()) return false;
        finished_ = true;
        return true;
    }
    const std::string& b = vocab.bytes(id);
    if (b.empty()) return false;
    return accept_bytes(b);
}

void GrammarMatcher::mask_candidates(const TokenTable& vocab, const int32_t* ids, size_t n, uint8_t* ok) {
    if (finished_ || cur_ < 0) {
        std::memset(ok, 0, n);
        return;
    }
    maybe_reset_dfa();
    const State& cs = states_[static_cast<size_t>(cur_)];
    if (cs.mask_slot >= 0) {
        const auto& bits = masks_[static_cast<size_t>(cs.mask_slot)].bits;
        for (size_t k = 0; k < n; ++k) {
            const auto id = static_cast<uint32_t>(ids[k]);
            ok[k] = ids[k] >= 0 && id < vocab.size() && ((bits[id >> 6] >> (id & 63)) & 1u);
        }
        return;
    }
    const bool accepting = cs.accepting;
    for (size_t k = 0; k < n; ++k) {
        ok[k] = 0;
        const int32_t id = ids[k];
        if (id < 0 || static_cast<size_t>(id) >= vocab.size()) continue;
        if (id == vocab.eos()) {
            ok[k] = accepting;
            continue;
        }
        const std::string& b = vocab.bytes(id);
        if (b.empty()) continue;
        int32_t s = cur_;
        for (unsigned char c : b) {
            s = step(s, c);
            if (s < 0) break;
        }
        ok[k] = s >= 0;
    }
}

void GrammarMatcher::dfs(const TokenTable& vocab, uint32_t node, int32_t s, std::vector<uint64_t>& bits) {
    ++st_.trie_nodes;
    const TokenTable::TrieNode& n = vocab.nodes()[node];
    for (uint32_t t = 0; t < n.n_tokens; ++t) {
        const auto id = static_cast<uint32_t>(vocab.node_tokens()[n.first_token + t]);
        bits[id >> 6] |= uint64_t{1} << (id & 63);
    }
    for (uint32_t e = 0; e < n.n_children; ++e) {
        const TokenTable::Edge& ed = vocab.edges()[n.first_child + e];
        const int32_t ns = step(s, ed.byte);
        if (ns >= 0) dfs(vocab, ed.node, ns, bits);
    }
}

bool GrammarMatcher::loop_region(int32_t s, ByteSet& loop, ByteSet& leads) {
    loop = ByteSet{};
    leads = ByteSet{};
    int n = 0;
    for (unsigned b = 0; b < 256; ++b)
        if (step(s, static_cast<uint8_t>(b)) == s) {
            put(loop, b);
            ++n;
        }
    if (n < 16) return false;
    const ByteSet cont = range(0x80, 0xBF);
    for (unsigned lead = 0xC2; lead <= 0xF4; ++lead) {
        if (has(loop, lead)) continue;
        int32_t t = step(s, static_cast<uint8_t>(lead));
        bool ok = t >= 0;
        for (int j = 0; ok && j < cont_bytes(lead); ++j) {
            const ByteSet& f = states_[static_cast<size_t>(t)].first;
            for (int w = 0; w < 4 && ok; ++w)
                if (f[static_cast<size_t>(w)] & ~cont[static_cast<size_t>(w)]) ok = false;
            if (!ok) break;
            const int32_t u = step(t, 0x80);
            for (unsigned c = 0x81; c <= 0xBF && ok; ++c)
                if (step(t, static_cast<uint8_t>(c)) != u) ok = false;
            t = u;
            if (t < 0) ok = false;
        }
        if (ok && t == s) put(leads, lead);
    }
    return true;
}

void GrammarMatcher::fill_mask(const TokenTable& vocab, std::vector<uint64_t>& bits) {
    bits.assign(vocab.words(), 0);
    if (finished_ || cur_ < 0) return;
    maybe_reset_dfa();
    const int32_t s = cur_;
    if (const int32_t slot = states_[static_cast<size_t>(s)].mask_slot; slot >= 0) {
        MaskEntry& m = masks_[static_cast<size_t>(slot)];
        m.last = ++tick_;
        bits = m.bits;
        ++st_.mask_hits;
        return;
    }
    ++st_.mask_misses;
    ByteSet loop, leads;
    if (loop_region(s, loop, leads)) {
        auto r = vocab.region(loop, leads);
        bits = r->bits;
        // Walk each exit tail; consecutive tails share prefixes, so resume from the deepest common state.
        std::vector<int32_t> at{s};  // at[k] = state after k bytes of the previous tail (-1 dead)
        const std::string* prev = nullptr;
        for (const auto& ex : r->exits) {
            size_t k = 0;
            if (prev) {
                const size_t lim = std::min({prev->size(), ex.tail.size(), at.size() - 1});
                while (k < lim && (*prev)[k] == ex.tail[k]) ++k;
            }
            at.resize(k + 1);
            int32_t cur = at[k];
            for (; k < ex.tail.size() && cur >= 0; ++k) {
                cur = step(cur, static_cast<uint8_t>(ex.tail[k]));
                at.push_back(cur);
            }
            prev = &ex.tail;
            st_.trie_nodes += ex.tail.size();
            if (cur < 0) continue;
            for (int32_t id : ex.ids) bits[static_cast<uint32_t>(id) >> 6] |= uint64_t{1} << (static_cast<uint32_t>(id) & 63);
        }
        ++st_.region_masks;
    } else {
        dfs(vocab, 0, s, bits);
    }
    if (states_[static_cast<size_t>(s)].accepting && vocab.eos() >= 0) {
        const auto e = static_cast<uint32_t>(vocab.eos());
        bits[e >> 6] |= uint64_t{1} << (e & 63);
    }
    size_t slot;
    if (masks_.size() < cache_cap_) {
        slot = masks_.size();
        masks_.emplace_back();
    } else {
        slot = 0;
        for (size_t k = 1; k < masks_.size(); ++k)
            if (masks_[k].last < masks_[slot].last) slot = k;
        if (masks_[slot].state >= 0) states_[static_cast<size_t>(masks_[slot].state)].mask_slot = -1;
    }
    masks_[slot].state = s;
    masks_[slot].last = ++tick_;
    masks_[slot].bits = bits;
    states_[static_cast<size_t>(s)].mask_slot = static_cast<int32_t>(slot);
}

void GrammarMatcher::apply(const TokenTable& vocab, float* logits) {
    std::vector<uint64_t> bits;
    fill_mask(vocab, bits);
    for (size_t id = 0; id < vocab.size(); ++id)
        if (!((bits[id >> 6] >> (id & 63)) & 1u)) logits[id] = -std::numeric_limits<float>::infinity();
}

} // namespace vegajet::grammar
