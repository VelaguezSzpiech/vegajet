// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/chat_template.hpp"

#include "core/json_lite.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <unordered_map>

namespace vegajet::chat {

namespace {

struct Error : std::runtime_error {
    using std::runtime_error::runtime_error;
};
[[noreturn]] void fail(const std::string& m) { throw Error(m); }

// ================================================================================================= values
struct Macro;
struct V;
using List = std::vector<V>;
using Dict = std::vector<std::pair<std::string, V>>;

struct V {
    enum T { Undef, None, Bool, Int, Float, Str, Lst, Dct, Fn } t = Undef;
    bool b = false;
    int64_t i = 0;
    double f = 0;
    std::string s;
    std::shared_ptr<List> l;   // shared: Python reference semantics (namespace(), list.append)
    std::shared_ptr<Dict> d;
    std::shared_ptr<Macro> fn;
    static V none() { V v; v.t = None; return v; }
    static V boolean(bool x) { V v; v.t = Bool; v.b = x; return v; }
    static V integer(int64_t x) { V v; v.t = Int; v.i = x; return v; }
    static V real(double x) { V v; v.t = Float; v.f = x; return v; }
    static V str(std::string x) { V v; v.t = Str; v.s = std::move(x); return v; }
    static V list(List x = {}) { V v; v.t = Lst; v.l = std::make_shared<List>(std::move(x)); return v; }
    static V dict(Dict x = {}) { V v; v.t = Dct; v.d = std::make_shared<Dict>(std::move(x)); return v; }
    bool num() const { return t == Int || t == Float || t == Bool; }
    double as_f() const { return t == Float ? f : t == Int ? static_cast<double>(i) : t == Bool ? (b ? 1.0 : 0.0) : 0.0; }
    const V* get(const std::string& k) const {
        if (t != Dct) return nullptr;
        for (const auto& kv : *d)
            if (kv.first == k) return &kv.second;
        return nullptr;
    }
    void set(const std::string& k, V v) {
        for (auto& kv : *d)
            if (kv.first == k) { kv.second = std::move(v); return; }
        d->emplace_back(k, std::move(v));
    }
};

bool truthy(const V& v) {
    switch (v.t) {
        case V::Undef: case V::None: return false;
        case V::Bool: return v.b;
        case V::Int: return v.i != 0;
        case V::Float: return v.f != 0.0;
        case V::Str: return !v.s.empty();
        case V::Lst: return !v.l->empty();
        case V::Dct: return !v.d->empty();
        case V::Fn: return true;
    }
    return false;
}

std::string py_float(double x) {
    if (std::isnan(x)) return "nan";
    if (std::isinf(x)) return x > 0 ? "inf" : "-inf";
    char b[40];
    for (int p = 1; p <= 17; ++p) {
        std::snprintf(b, sizeof b, "%.*g", p, x);
        if (std::strtod(b, nullptr) == x) break;
    }
    std::string s = b;
    if (s.find_first_of(".einn") == std::string::npos) s += ".0";
    return s;
}

std::string repr(const V& v);
std::string to_str(const V& v) {
    switch (v.t) {
        case V::Undef: return "";
        case V::None: return "None";
        case V::Bool: return v.b ? "True" : "False";
        case V::Int: return std::to_string(v.i);
        case V::Float: return py_float(v.f);
        case V::Str: return v.s;
        default: return repr(v);
    }
}
std::string repr(const V& v) {
    if (v.t == V::Str) {
        std::string o = "'";
        for (char c : v.s) {
            if (c == '\'' || c == '\\') o += '\\';
            if (c == '\n') { o += "\\n"; continue; }
            o += c;
        }
        return o + "'";
    }
    if (v.t == V::Lst) {
        std::string o = "[";
        for (size_t k = 0; k < v.l->size(); ++k) o += (k ? ", " : "") + repr((*v.l)[k]);
        return o + "]";
    }
    if (v.t == V::Dct) {
        std::string o = "{";
        for (size_t k = 0; k < v.d->size(); ++k) o += (k ? ", " : "") + repr(V::str((*v.d)[k].first)) + ": " + repr((*v.d)[k].second);
        return o + "}";
    }
    return to_str(v);
}

void json_str(std::string& o, const std::string& s) {
    o += '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            case '\b': o += "\\b"; break;
            case '\f': o += "\\f"; break;
            default:
                if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                else o += static_cast<char>(c);
        }
    }
    o += '"';
}

// json.dumps(ensure_ascii=False): separators (", ", ": ") without indent, ("," , ": ") with indent.
void to_json(std::string& o, const V& v, int indent, int depth) {
    auto nl = [&](int d) { if (indent >= 0) { o += '\n'; o.append(static_cast<size_t>(indent * d), ' '); } };
    switch (v.t) {
        case V::Undef: case V::None: o += "null"; return;
        case V::Bool: o += v.b ? "true" : "false"; return;
        case V::Int: o += std::to_string(v.i); return;
        case V::Float: o += py_float(v.f); return;
        case V::Str: json_str(o, v.s); return;
        case V::Lst:
            if (v.l->empty()) { o += "[]"; return; }
            o += '[';
            for (size_t k = 0; k < v.l->size(); ++k) {
                if (k) o += indent >= 0 ? "," : ", ";
                nl(depth + 1);
                to_json(o, (*v.l)[k], indent, depth + 1);
            }
            nl(depth);
            o += ']';
            return;
        case V::Dct:
            if (v.d->empty()) { o += "{}"; return; }
            o += '{';
            for (size_t k = 0; k < v.d->size(); ++k) {
                if (k) o += indent >= 0 ? "," : ", ";
                nl(depth + 1);
                json_str(o, (*v.d)[k].first);
                o += ": ";
                to_json(o, (*v.d)[k].second, indent, depth + 1);
            }
            nl(depth);
            o += '}';
            return;
        case V::Fn: o += "null"; return;
    }
}

V from_json(const json::Value& j) {
    switch (j.type) {
        case json::Value::Type::Null: return V::none();
        case json::Value::Type::Bool: return V::boolean(j.b);
        case json::Value::Type::Number:
            if (std::fabs(j.num) < 9.0e15 && j.num == std::floor(j.num)) return V::integer(static_cast<int64_t>(j.num));
            return V::real(j.num);
        case json::Value::Type::String: return V::str(j.str);
        case json::Value::Type::Array: {
            List l;
            for (const auto& x : j.arr) l.push_back(from_json(x));
            return V::list(std::move(l));
        }
        case json::Value::Type::Object: {
            Dict d;
            for (const auto& kv : j.obj) d.emplace_back(kv.first, from_json(kv.second));
            return V::dict(std::move(d));
        }
    }
    return V::none();
}

bool equal(const V& a, const V& b) {
    if (a.num() && b.num()) return a.as_f() == b.as_f();
    if (a.t != b.t) return (a.t == V::Undef || a.t == V::None) && (b.t == V::Undef || b.t == V::None);
    switch (a.t) {
        case V::Str: return a.s == b.s;
        case V::Lst:
            if (a.l->size() != b.l->size()) return false;
            for (size_t k = 0; k < a.l->size(); ++k)
                if (!equal((*a.l)[k], (*b.l)[k])) return false;
            return true;
        case V::Dct:
            if (a.d->size() != b.d->size()) return false;
            for (const auto& kv : *a.d) {
                const V* o = b.get(kv.first);
                if (!o || !equal(kv.second, *o)) return false;
            }
            return true;
        default: return true;
    }
}

bool less(const V& a, const V& b) {
    if (a.num() && b.num()) return a.as_f() < b.as_f();
    if (a.t == V::Str && b.t == V::Str) return a.s < b.s;
    fail("cannot order " + repr(a) + " and " + repr(b));
}

// ================================================================================================= expressions
struct Expr;
using EP = std::shared_ptr<Expr>;
struct Expr {
    enum K { Lit, Name, Attr, Item, Slice, Call, Filter, Test, Not, Neg, Bin, And, Or, Cond, ListE, DictE } k = Lit;
    V lit;
    std::string name;  // Name / Attr / Call-free filter or test name / Bin op
    std::vector<EP> a;  // operands / args (Slice: obj, start, stop, step; null = omitted)
    std::vector<std::pair<std::string, EP>> kw;
    bool negate = false;  // `is not` / `not in`
};

struct Tok {
    enum K { Name, Num, Str, Op, End } k = End;
    std::string s;
};

std::vector<Tok> lex_expr(std::string_view src) {
    std::vector<Tok> out;
    size_t i = 0;
    while (i < src.size()) {
        const char c = src[i];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { ++i; continue; }
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            size_t j = i;
            while (j < src.size() && (std::isalnum(static_cast<unsigned char>(src[j])) || src[j] == '_')) ++j;
            out.push_back({Tok::Name, std::string(src.substr(i, j - i))});
            i = j;
            continue;
        }
        if (std::isdigit(static_cast<unsigned char>(c))) {
            size_t j = i;
            while (j < src.size() && (std::isdigit(static_cast<unsigned char>(src[j])) || src[j] == '.' || src[j] == '_')) {
                if (src[j] == '.' && (j + 1 >= src.size() || !std::isdigit(static_cast<unsigned char>(src[j + 1])))) break;
                ++j;
            }
            out.push_back({Tok::Num, std::string(src.substr(i, j - i))});
            i = j;
            continue;
        }
        if (c == '\'' || c == '"') {
            std::string s;
            size_t j = i + 1;
            while (j < src.size() && src[j] != c) {
                if (src[j] == '\\' && j + 1 < src.size()) {
                    const char e = src[++j];
                    switch (e) {
                        case 'n': s += '\n'; break;
                        case 't': s += '\t'; break;
                        case 'r': s += '\r'; break;
                        case '0': s += '\0'; break;
                        case 'u': case 'x': {
                            const size_t n = e == 'u' ? 4 : 2;
                            if (j + n >= src.size()) fail("bad escape");
                            const uint32_t cp = static_cast<uint32_t>(std::strtoul(std::string(src.substr(j + 1, n)).c_str(), nullptr, 16));
                            if (cp < 0x80) s += static_cast<char>(cp);
                            else if (cp < 0x800) { s += static_cast<char>(0xC0 | (cp >> 6)); s += static_cast<char>(0x80 | (cp & 0x3F)); }
                            else { s += static_cast<char>(0xE0 | (cp >> 12)); s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); s += static_cast<char>(0x80 | (cp & 0x3F)); }
                            j += n;
                            break;
                        }
                        default: s += e;
                    }
                    ++j;
                    continue;
                }
                s += src[j++];
            }
            if (j >= src.size()) fail("unterminated string literal");
            out.push_back({Tok::Str, s});
            i = j + 1;
            continue;
        }
        static const char* const ops[] = {"==", "!=", "<=", ">=", "//", "**", "<", ">", "+", "-", "*", "/", "%", "~",
                                          "(", ")", "[", "]", "{", "}", ",", ":", ".", "|", "="};
        bool hit = false;
        for (const char* op : ops) {
            const size_t n = std::strlen(op);
            if (src.substr(i, n) == op) {
                out.push_back({Tok::Op, op});
                i += n;
                hit = true;
                break;
            }
        }
        if (!hit) fail(std::string("unexpected character '") + c + "' in expression");
    }
    out.push_back({Tok::End, ""});
    return out;
}

class ExprParser {
public:
    explicit ExprParser(std::vector<Tok> t) : t_(std::move(t)) {}
    EP parse_full() {
        EP e = expr();
        if (t_[p_].k != Tok::End) fail("unexpected '" + t_[p_].s + "' in expression");
        return e;
    }
    EP expr() { return cond(); }
    bool at_end() const { return t_[p_].k == Tok::End; }
    const Tok& peek(size_t o = 0) const { return t_[std::min(p_ + o, t_.size() - 1)]; }
    bool name_is(const char* n, size_t o = 0) const { return peek(o).k == Tok::Name && peek(o).s == n; }
    bool op_is(const char* n, size_t o = 0) const { return peek(o).k == Tok::Op && peek(o).s == n; }
    void expect_op(const char* n) { if (!op_is(n)) fail(std::string("expected '") + n + "'"); ++p_; }
    std::string take_name() { if (peek().k != Tok::Name) fail("expected a name"); return t_[p_++].s; }
    size_t pos() const { return p_; }

private:
    static EP mk(Expr::K k) { auto e = std::make_shared<Expr>(); e->k = k; return e; }
    EP cond() {
        EP v = or_();
        if (name_is("if")) {
            ++p_;
            EP c = or_();
            EP alt;
            if (name_is("else")) { ++p_; alt = cond(); }
            EP e = mk(Expr::Cond);
            e->a = {c, v, alt};
            return e;
        }
        return v;
    }
    EP or_() {
        EP l = and_();
        while (name_is("or")) { ++p_; EP e = mk(Expr::Or); e->a = {l, and_()}; l = e; }
        return l;
    }
    EP and_() {
        EP l = not_();
        while (name_is("and")) { ++p_; EP e = mk(Expr::And); e->a = {l, not_()}; l = e; }
        return l;
    }
    EP not_() {
        if (name_is("not")) { ++p_; EP e = mk(Expr::Not); e->a = {not_()}; return e; }
        return compare();
    }
    EP compare() {
        EP l = concat();
        for (;;) {
            std::string op;
            bool neg = false;
            if (peek().k == Tok::Op && (peek().s == "==" || peek().s == "!=" || peek().s == "<" || peek().s == ">" ||
                                        peek().s == "<=" || peek().s == ">=")) op = t_[p_++].s;
            else if (name_is("in")) { op = "in"; ++p_; }
            else if (name_is("not") && name_is("in", 1)) { op = "in"; neg = true; p_ += 2; }
            else if (name_is("is")) {
                ++p_;
                EP e = mk(Expr::Test);
                if (name_is("not")) { e->negate = true; ++p_; }
                e->name = take_name();
                e->a = {l};
                if (op_is("(")) { args(e); }
                else if (peek().k == Tok::Num || peek().k == Tok::Str) e->a.push_back(primary());
                l = e;
                continue;
            } else break;
            EP e = mk(Expr::Bin);
            e->name = op;
            e->negate = neg;
            e->a = {l, concat()};
            l = e;
        }
        return l;
    }
    EP concat() {
        EP l = additive();
        while (op_is("~")) { ++p_; EP e = mk(Expr::Bin); e->name = "~"; e->a = {l, additive()}; l = e; }
        return l;
    }
    EP additive() {
        EP l = mult();
        while (op_is("+") || op_is("-")) { EP e = mk(Expr::Bin); e->name = t_[p_++].s; e->a = {l, mult()}; l = e; }
        return l;
    }
    EP mult() {
        EP l = unary();
        while (op_is("*") || op_is("/") || op_is("//") || op_is("%")) { EP e = mk(Expr::Bin); e->name = t_[p_++].s; e->a = {l, unary()}; l = e; }
        return l;
    }
    EP unary() {
        if (op_is("-")) { ++p_; EP e = mk(Expr::Neg); e->a = {unary()}; return e; }
        if (op_is("+")) { ++p_; return unary(); }
        return power();
    }
    EP power() {
        EP l = postfix();
        if (op_is("**")) { ++p_; EP e = mk(Expr::Bin); e->name = "**"; e->a = {l, unary()}; return e; }
        return l;
    }
    void args(const EP& e) {  // after the callee / filter name: ( positional..., name=value... )
        expect_op("(");
        while (!op_is(")")) {
            if (peek().k == Tok::Name && op_is("=", 1)) {
                std::string k = take_name();
                ++p_;
                e->kw.emplace_back(k, expr());
            } else {
                e->a.push_back(expr());
            }
            if (!op_is(",")) break;
            ++p_;
        }
        expect_op(")");
    }
    EP postfix() {
        EP v = primary();
        for (;;) {
            if (op_is(".")) {
                ++p_;
                EP e = mk(Expr::Attr);
                e->name = take_name();
                e->a = {v};
                v = e;
            } else if (op_is("[")) {
                ++p_;
                EP parts[3];
                bool slice = false;
                for (int k = 0; k < 3; ++k) {
                    if (!op_is(":") && !op_is("]")) parts[k] = expr();
                    if (op_is(":")) { slice = true; ++p_; continue; }
                    break;
                }
                expect_op("]");
                EP e = mk(slice ? Expr::Slice : Expr::Item);
                e->a = slice ? std::vector<EP>{v, parts[0], parts[1], parts[2]} : std::vector<EP>{v, parts[0]};
                if (!slice && !parts[0]) fail("empty subscript");
                v = e;
            } else if (op_is("(")) {
                EP e = mk(Expr::Call);
                e->a = {v};
                args(e);
                v = e;
            } else if (op_is("|")) {
                ++p_;
                EP e = mk(Expr::Filter);
                e->name = take_name();
                e->a = {v};
                if (op_is("(")) args(e);
                v = e;
            } else {
                return v;
            }
        }
    }
    EP primary() {
        const Tok t = peek();
        if (t.k == Tok::Num) {
            ++p_;
            std::string s = t.s;
            s.erase(std::remove(s.begin(), s.end(), '_'), s.end());
            EP e = mk(Expr::Lit);
            e->lit = s.find('.') != std::string::npos ? V::real(std::strtod(s.c_str(), nullptr)) : V::integer(std::strtoll(s.c_str(), nullptr, 10));
            return e;
        }
        if (t.k == Tok::Str) {
            ++p_;
            std::string s = t.s;
            while (peek().k == Tok::Str) s += t_[p_++].s;  // implicit concatenation
            EP e = mk(Expr::Lit);
            e->lit = V::str(s);
            return e;
        }
        if (t.k == Tok::Name) {
            ++p_;
            EP e = mk(Expr::Lit);
            if (t.s == "true" || t.s == "True") e->lit = V::boolean(true);
            else if (t.s == "false" || t.s == "False") e->lit = V::boolean(false);
            else if (t.s == "none" || t.s == "None") e->lit = V::none();
            else { e->k = Expr::Name; e->name = t.s; }
            return e;
        }
        if (op_is("(")) {
            ++p_;
            if (op_is(")")) { ++p_; EP e = mk(Expr::ListE); return e; }
            EP first = expr();
            if (op_is(",")) {  // tuple -> list
                EP e = mk(Expr::ListE);
                e->a.push_back(first);
                while (op_is(",")) { ++p_; if (op_is(")")) break; e->a.push_back(expr()); }
                expect_op(")");
                return e;
            }
            expect_op(")");
            return first;
        }
        if (op_is("[")) {
            ++p_;
            EP e = mk(Expr::ListE);
            while (!op_is("]")) { e->a.push_back(expr()); if (!op_is(",")) break; ++p_; }
            expect_op("]");
            return e;
        }
        if (op_is("{")) {
            ++p_;
            EP e = mk(Expr::DictE);
            while (!op_is("}")) {
                e->a.push_back(expr());
                expect_op(":");
                e->a.push_back(expr());
                if (!op_is(",")) break;
                ++p_;
            }
            expect_op("}");
            return e;
        }
        fail("unexpected '" + t.s + "' in expression");
    }
    std::vector<Tok> t_;
    size_t p_ = 0;
};

} // namespace

// ================================================================================================= statements
struct Node {
    enum K { Seq, Text, Out, If, For, Set, SetBlock, Macro } k = Seq;
    std::string text;                       // Text; Set / SetBlock target name; Macro name
    std::string attr;                       // Set: `ns.attr` target attribute
    EP e;                                   // Out / Set value / For iterable
    EP filter;                              // For: `if` clause
    std::vector<std::string> vars;          // For loop variables; Macro parameter names
    std::vector<EP> defaults;               // Macro parameter defaults (null = required)
    std::vector<std::pair<EP, std::shared_ptr<Node>>> branches;  // If: (cond, body); For: (null, body)
    std::shared_ptr<Node> else_body;
    std::vector<std::shared_ptr<Node>> kids;  // Seq
};

namespace {

using NP = std::shared_ptr<Node>;

struct Macro {
    NP node;
};

struct RawTag {
    enum K { Text, Var, Block } k = Text;
    std::string s;
};

// Splits the source into text / {{ }} / {% %} tags, applying `-` trimming, trim_blocks and lstrip_blocks.
std::vector<RawTag> lex_template(std::string_view src) {
    std::vector<RawTag> out;
    size_t i = 0;
    bool trim_next = false, trim_nl = false;
    while (i <= src.size()) {
        const size_t open = src.find('{', i);
        size_t tag = std::string_view::npos;
        for (size_t k = open; k != std::string_view::npos && k + 1 < src.size(); k = src.find('{', k + 1)) {
            const char n = src[k + 1];
            if (n == '{' || n == '%' || n == '#') { tag = k; break; }
        }
        std::string text(src.substr(i, (tag == std::string_view::npos ? src.size() : tag) - i));
        if (trim_next) {
            size_t k = 0;
            while (k < text.size() && std::isspace(static_cast<unsigned char>(text[k]))) ++k;
            text.erase(0, k);
        } else if (trim_nl && !text.empty() && text[0] == '\n') {
            text.erase(0, 1);
        }
        trim_next = trim_nl = false;
        if (tag == std::string_view::npos) {
            if (!text.empty()) out.push_back({RawTag::Text, text});
            break;
        }
        const char kind = src[tag + 1];
        const bool minus_l = tag + 2 < src.size() && src[tag + 2] == '-';
        const bool plus_l = tag + 2 < src.size() && src[tag + 2] == '+';
        const char* close = kind == '{' ? "}}" : kind == '%' ? "%}" : "#}";
        const size_t end = src.find(close, tag + 2);
        if (end == std::string_view::npos) fail("unterminated tag");
        const bool minus_r = end > tag + 2 && src[end - 1] == '-';
        if (minus_l) {
            while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) text.pop_back();
        } else if (kind != '{' && !plus_l) {
            // lstrip_blocks: spaces / tabs between the start of the source line and a block tag are removed.
            size_t p = tag;
            while (p > 0 && (src[p - 1] == ' ' || src[p - 1] == '\t')) --p;
            if (p == 0 || src[p - 1] == '\n') {
                size_t k = text.size();
                while (k > 0 && (text[k - 1] == ' ' || text[k - 1] == '\t')) --k;
                text.erase(k);
            }
        }
        if (!text.empty()) out.push_back({RawTag::Text, text});
        size_t body_a = tag + 2 + ((minus_l || plus_l) ? 1 : 0);
        size_t body_b = end - (minus_r ? 1 : 0);
        if (kind != '#') out.push_back({kind == '{' ? RawTag::Var : RawTag::Block, std::string(src.substr(body_a, body_b - body_a))});
        trim_next = minus_r;
        trim_nl = !minus_r && kind != '{';  // trim_blocks
        i = end + 2;
    }
    return out;
}

class StmtParser {
public:
    explicit StmtParser(std::vector<RawTag> t) : t_(std::move(t)) {}
    NP parse() {
        std::string term;
        NP n = body({}, term);
        if (!term.empty()) fail("unexpected {% " + term + " %}");
        return n;
    }

private:
    static std::string first_word(const std::string& s) {
        size_t a = 0;
        while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
        size_t b = a;
        while (b < s.size() && (std::isalnum(static_cast<unsigned char>(s[b])) || s[b] == '_')) ++b;
        return s.substr(a, b - a);
    }
    static std::string rest(const std::string& s) {
        const std::string w = first_word(s);
        return s.substr(s.find(w) + w.size());
    }
    // Parses nodes until a block tag whose first word is in `ends`; returns it via `term` (its full text in term_text_).
    NP body(const std::vector<std::string>& ends, std::string& term) {
        auto seq = std::make_shared<Node>();
        while (p_ < t_.size()) {
            const RawTag& t = t_[p_];
            if (t.k == RawTag::Text) {
                auto n = std::make_shared<Node>();
                n->k = Node::Text;
                n->text = t.s;
                seq->kids.push_back(n);
                ++p_;
                continue;
            }
            if (t.k == RawTag::Var) {
                auto n = std::make_shared<Node>();
                n->k = Node::Out;
                n->e = ExprParser(lex_expr(t.s)).parse_full();
                seq->kids.push_back(n);
                ++p_;
                continue;
            }
            const std::string w = first_word(t.s);
            if (std::find(ends.begin(), ends.end(), w) != ends.end()) {
                term = w;
                term_text_ = t.s;
                ++p_;
                return seq;
            }
            ++p_;
            if (w == "if") seq->kids.push_back(parse_if(rest(t.s)));
            else if (w == "for") seq->kids.push_back(parse_for(rest(t.s)));
            else if (w == "set") seq->kids.push_back(parse_set(rest(t.s)));
            else if (w == "macro") seq->kids.push_back(parse_macro(rest(t.s)));
            else if (w == "generation" || w == "endgeneration") continue;  // HF assistant-mask markers: no output
            else fail("unsupported tag {% " + w + " %}");
        }
        term.clear();
        return seq;
    }
    NP parse_if(const std::string& cond) {
        auto n = std::make_shared<Node>();
        n->k = Node::If;
        EP c = ExprParser(lex_expr(cond)).parse_full();
        for (;;) {
            std::string term;
            NP b = body({"elif", "else", "endif"}, term);
            n->branches.emplace_back(c, b);
            if (term == "elif") { c = ExprParser(lex_expr(rest(term_text_))).parse_full(); continue; }
            if (term == "else") {
                n->else_body = body({"endif"}, term);
                if (term != "endif") fail("missing {% endif %}");
                return n;
            }
            if (term != "endif") fail("missing {% endif %}");
            return n;
        }
    }
    NP parse_for(const std::string& head) {
        auto n = std::make_shared<Node>();
        n->k = Node::For;
        ExprParser ep(lex_expr(head));
        n->vars.push_back(ep.take_name());
        while (ep.op_is(",")) { ep.expect_op(","); n->vars.push_back(ep.take_name()); }
        if (!ep.name_is("in")) fail("expected 'in' in for loop");
        ep.take_name();
        // The iterable stops before a top-level `if` (loop filter), which cond() would otherwise take as a ternary.
        std::vector<Tok> toks = lex_expr(head);
        size_t start = ep.pos(), depth = 0, cut = toks.size() - 1;
        for (size_t k = start; k < toks.size(); ++k) {
            const Tok& t = toks[k];
            if (t.k == Tok::Op && (t.s == "(" || t.s == "[" || t.s == "{")) ++depth;
            if (t.k == Tok::Op && (t.s == ")" || t.s == "]" || t.s == "}")) --depth;
            if (depth == 0 && t.k == Tok::Name && (t.s == "if" || t.s == "recursive")) { cut = k; break; }
        }
        std::vector<Tok> it(toks.begin() + static_cast<long>(start), toks.begin() + static_cast<long>(cut));
        it.push_back({Tok::End, ""});
        n->e = ExprParser(it).parse_full();
        if (cut < toks.size() - 1 && toks[cut].s == "if") {
            std::vector<Tok> f(toks.begin() + static_cast<long>(cut + 1), toks.end());
            n->filter = ExprParser(f).parse_full();
        }
        std::string term;
        NP b = body({"else", "endfor"}, term);
        n->branches.emplace_back(nullptr, b);
        if (term == "else") n->else_body = body({"endfor"}, term);
        if (term != "endfor") fail("missing {% endfor %}");
        return n;
    }
    NP parse_set(const std::string& s) {
        auto n = std::make_shared<Node>();
        ExprParser ep(lex_expr(s));
        n->text = ep.take_name();
        if (ep.op_is(".")) { ep.expect_op("."); n->attr = ep.take_name(); }
        if (ep.at_end()) {  // {% set x %}...{% endset %}
            n->k = Node::SetBlock;
            std::string term;
            n->else_body = body({"endset"}, term);
            if (term != "endset") fail("missing {% endset %}");
            return n;
        }
        n->k = Node::Set;
        ep.expect_op("=");
        n->e = ep.expr();
        if (!ep.at_end()) fail("unexpected tokens after {% set %}");
        return n;
    }
    NP parse_macro(const std::string& s) {
        auto n = std::make_shared<Node>();
        n->k = Node::Macro;
        ExprParser ep(lex_expr(s));
        n->text = ep.take_name();
        ep.expect_op("(");
        while (!ep.op_is(")")) {
            n->vars.push_back(ep.take_name());
            EP d;
            if (ep.op_is("=")) { ep.expect_op("="); d = ep.expr(); }
            n->defaults.push_back(d);
            if (!ep.op_is(",")) break;
            ep.expect_op(",");
        }
        ep.expect_op(")");
        std::string term;
        n->else_body = body({"endmacro"}, term);
        if (term != "endmacro") fail("missing {% endmacro %}");
        return n;
    }
    std::vector<RawTag> t_;
    size_t p_ = 0;
    std::string term_text_;
};

// ================================================================================================= renderer
constexpr size_t kMaxOutput = 32u << 20;
constexpr int kMaxDepth = 64;

class Renderer {
public:
    Renderer(std::string& out, std::string date) : out_(out), date_(std::move(date)) { scopes_.emplace_back(); }
    void set_global(const std::string& k, V v) { scopes_.front()[k] = std::move(v); }
    void render(const NP& n) {
        switch (n->k) {
            case Node::Seq:
                for (const auto& k : n->kids) render(k);
                return;
            case Node::Text: write(n->text); return;
            case Node::Out: write(to_str(eval(n->e))); return;
            case Node::If:
                for (const auto& br : n->branches)
                    if (truthy(eval(br.first))) { render(br.second); return; }
                if (n->else_body) render(n->else_body);
                return;
            case Node::For: render_for(n); return;
            case Node::Set: {
                V v = eval(n->e);
                if (n->attr.empty()) { scopes_.back()[n->text] = std::move(v); return; }
                V* obj = lookup(n->text);
                if (!obj || obj->t != V::Dct) fail("cannot set attribute on " + n->text);
                obj->set(n->attr, std::move(v));
                return;
            }
            case Node::SetBlock: {
                std::string saved;
                saved.swap(out_);
                render(n->else_body);
                saved.swap(out_);
                scopes_.back()[n->text] = V::str(saved);
                return;
            }
            case Node::Macro: {
                V f;
                f.t = V::Fn;
                f.fn = std::make_shared<Macro>(Macro{n});
                scopes_.back()[n->text] = f;
                return;
            }
        }
    }

private:
    void write(const std::string& s) {
        out_ += s;
        if (out_.size() > kMaxOutput) fail("template output exceeds 32 MiB");
    }
    V* lookup(const std::string& k) {
        for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
            auto f = it->find(k);
            if (f != it->end()) return &f->second;
        }
        return nullptr;
    }
    static List iterate(const V& v) {
        if (v.t == V::Lst) return *v.l;
        if (v.t == V::Dct) { List l; for (const auto& kv : *v.d) l.push_back(V::str(kv.first)); return l; }
        if (v.t == V::Str) { List l; for (char c : v.s) l.push_back(V::str(std::string(1, c))); return l; }
        if (v.t == V::Undef || v.t == V::None) return {};
        fail("cannot iterate over " + repr(v));
    }
    void render_for(const NP& n) {
        List items = iterate(eval(n->e));
        if (n->filter) {
            List kept;
            for (const auto& it : items) {
                scopes_.emplace_back();
                bind(n->vars, it);
                if (truthy(eval(n->filter))) kept.push_back(it);
                scopes_.pop_back();
            }
            items.swap(kept);
        }
        if (items.empty()) { if (n->else_body) render(n->else_body); return; }
        const int64_t len = static_cast<int64_t>(items.size());
        for (int64_t k = 0; k < len; ++k) {
            scopes_.emplace_back();
            Dict loop{{"index", V::integer(k + 1)}, {"index0", V::integer(k)}, {"revindex", V::integer(len - k)},
                      {"revindex0", V::integer(len - k - 1)}, {"first", V::boolean(k == 0)}, {"last", V::boolean(k == len - 1)},
                      {"length", V::integer(len)}, {"previtem", k > 0 ? items[static_cast<size_t>(k - 1)] : V{}},
                      {"nextitem", k + 1 < len ? items[static_cast<size_t>(k + 1)] : V{}}};
            scopes_.back()["loop"] = V::dict(std::move(loop));
            bind(n->vars, items[static_cast<size_t>(k)]);
            render(n->branches[0].second);
            scopes_.pop_back();
        }
    }
    void bind(const std::vector<std::string>& vars, const V& it) {
        if (vars.size() == 1) { scopes_.back()[vars[0]] = it; return; }
        if (it.t != V::Lst || it.l->size() != vars.size()) fail("cannot unpack loop item " + repr(it));
        for (size_t k = 0; k < vars.size(); ++k) scopes_.back()[vars[k]] = (*it.l)[k];
    }

    static int64_t norm_index(int64_t k, size_t n) { return k < 0 ? k + static_cast<int64_t>(n) : k; }

    V eval(const EP& e) {
        switch (e->k) {
            case Expr::Lit: return e->lit;
            case Expr::Name: {
                V* v = lookup(e->name);
                return v ? *v : V{};
            }
            case Expr::Attr: {
                V o = eval(e->a[0]);
                const V* v = o.get(e->name);
                return v ? *v : V{};
            }
            case Expr::Item: {
                V o = eval(e->a[0]), k = eval(e->a[1]);
                if (o.t == V::Dct) { const V* v = o.get(to_str(k)); return v ? *v : V{}; }
                if ((o.t == V::Lst || o.t == V::Str) && k.t == V::Int) {
                    const size_t n = o.t == V::Lst ? o.l->size() : o.s.size();
                    const int64_t i = norm_index(k.i, n);
                    if (i < 0 || i >= static_cast<int64_t>(n)) return V{};
                    return o.t == V::Lst ? (*o.l)[static_cast<size_t>(i)] : V::str(std::string(1, o.s[static_cast<size_t>(i)]));
                }
                return V{};
            }
            case Expr::Slice: return slice(eval(e->a[0]), e->a[1], e->a[2], e->a[3]);
            case Expr::Call: return call(e);
            case Expr::Filter: return filter(e);
            case Expr::Test: {
                const bool r = test(e->name, eval(e->a[0]), e->a.size() > 1 ? eval(e->a[1]) : V{});
                return V::boolean(r != e->negate);
            }
            case Expr::Not: return V::boolean(!truthy(eval(e->a[0])));
            case Expr::Neg: {
                V v = eval(e->a[0]);
                if (v.t == V::Int) return V::integer(-v.i);
                if (v.num()) return V::real(-v.as_f());
                fail("cannot negate " + repr(v));
            }
            case Expr::And: { V l = eval(e->a[0]); return truthy(l) ? eval(e->a[1]) : l; }
            case Expr::Or: { V l = eval(e->a[0]); return truthy(l) ? l : eval(e->a[1]); }
            case Expr::Cond: return truthy(eval(e->a[0])) ? eval(e->a[1]) : (e->a[2] ? eval(e->a[2]) : V{});
            case Expr::ListE: {
                List l;
                for (const auto& x : e->a) l.push_back(eval(x));
                return V::list(std::move(l));
            }
            case Expr::DictE: {
                Dict d;
                for (size_t k = 0; k + 1 < e->a.size(); k += 2) d.emplace_back(to_str(eval(e->a[k])), eval(e->a[k + 1]));
                return V::dict(std::move(d));
            }
            case Expr::Bin: return binary(e->name, eval(e->a[0]), eval(e->a[1]), e->negate);
        }
        return V{};
    }

    V slice(const V& o, const EP& a, const EP& b, const EP& c) {
        const int64_t step = c ? eval(c).i : 1;
        if (step == 0) fail("slice step cannot be zero");
        const int64_t n = o.t == V::Lst ? static_cast<int64_t>(o.l->size()) : o.t == V::Str ? static_cast<int64_t>(o.s.size()) : -1;
        if (n < 0) return V{};
        auto clampi = [&](const EP& x, int64_t dflt) {
            if (!x) return dflt;
            int64_t k = eval(x).i;
            if (k < 0) k += n;
            if (step > 0) return std::max<int64_t>(0, std::min(k, n));
            return std::max<int64_t>(-1, std::min(k, n - 1));
        };
        const int64_t start = clampi(a, step > 0 ? 0 : n - 1), stop = clampi(b, step > 0 ? n : -1);
        List l;
        std::string s;
        for (int64_t k = start; step > 0 ? k < stop : k > stop; k += step) {
            if (o.t == V::Lst) l.push_back((*o.l)[static_cast<size_t>(k)]);
            else s += o.s[static_cast<size_t>(k)];
        }
        return o.t == V::Lst ? V::list(std::move(l)) : V::str(s);
    }

    V binary(const std::string& op, const V& l, const V& r, bool neg) {
        if (op == "==") return V::boolean(equal(l, r));
        if (op == "!=") return V::boolean(!equal(l, r));
        if (op == "<") return V::boolean(less(l, r));
        if (op == ">") return V::boolean(less(r, l));
        if (op == "<=") return V::boolean(!less(r, l));
        if (op == ">=") return V::boolean(!less(l, r));
        if (op == "in") {
            bool hit = false;
            if (r.t == V::Str) hit = r.s.find(to_str(l)) != std::string::npos;
            else if (r.t == V::Lst) hit = std::any_of(r.l->begin(), r.l->end(), [&](const V& x) { return equal(x, l); });
            else if (r.t == V::Dct) hit = r.get(to_str(l)) != nullptr;
            return V::boolean(hit != neg);
        }
        if (op == "~") return V::str(to_str(l) + to_str(r));
        if (op == "+") {
            if (l.t == V::Str || r.t == V::Str) {
                if ((l.t == V::Str || l.t == V::Undef) && (r.t == V::Str || r.t == V::Undef)) return V::str(to_str(l) + to_str(r));
                fail("cannot add " + repr(l) + " and " + repr(r));
            }
            if (l.t == V::Lst && r.t == V::Lst) { List x = *l.l; x.insert(x.end(), r.l->begin(), r.l->end()); return V::list(std::move(x)); }
        }
        if (op == "*" && l.t == V::Str && r.t == V::Int) {
            std::string s;
            for (int64_t k = 0; k < std::min<int64_t>(r.i, 1 << 20); ++k) s += l.s;
            return V::str(s);
        }
        if (!l.num() || !r.num()) fail("unsupported operands for " + op + ": " + repr(l) + ", " + repr(r));
        const bool ints = l.t != V::Float && r.t != V::Float;
        const double a = l.as_f(), b = r.as_f();
        if (op == "+") return ints ? V::integer(static_cast<int64_t>(a) + static_cast<int64_t>(b)) : V::real(a + b);
        if (op == "-") return ints ? V::integer(static_cast<int64_t>(a) - static_cast<int64_t>(b)) : V::real(a - b);
        if (op == "*") return ints ? V::integer(static_cast<int64_t>(a) * static_cast<int64_t>(b)) : V::real(a * b);
        if (op == "**") return ints && b >= 0 ? V::integer(static_cast<int64_t>(std::pow(a, b))) : V::real(std::pow(a, b));
        if (b == 0) fail("division by zero");
        if (op == "/") return V::real(a / b);
        if (op == "//") return ints ? V::integer(static_cast<int64_t>(std::floor(a / b))) : V::real(std::floor(a / b));
        if (op == "%") {
            if (ints) { int64_t m = static_cast<int64_t>(a) % static_cast<int64_t>(b); if (m != 0 && ((m < 0) != (b < 0))) m += static_cast<int64_t>(b); return V::integer(m); }
            return V::real(a - b * std::floor(a / b));
        }
        fail("unknown operator " + op);
    }

    bool test(const std::string& t, const V& v, const V& arg) {
        if (t == "defined") return v.t != V::Undef;
        if (t == "undefined") return v.t == V::Undef;
        if (t == "none") return v.t == V::None;
        if (t == "string") return v.t == V::Str;
        if (t == "number") return v.t == V::Int || v.t == V::Float;
        if (t == "integer") return v.t == V::Int;
        if (t == "float") return v.t == V::Float;
        if (t == "boolean") return v.t == V::Bool;
        if (t == "true") return v.t == V::Bool && v.b;
        if (t == "false") return v.t == V::Bool && !v.b;
        if (t == "mapping") return v.t == V::Dct;
        if (t == "iterable") return v.t == V::Lst || v.t == V::Dct || v.t == V::Str;
        if (t == "sequence") return v.t == V::Lst || v.t == V::Str;
        if (t == "callable") return v.t == V::Fn;
        if (t == "even") return v.t == V::Int && v.i % 2 == 0;
        if (t == "odd") return v.t == V::Int && v.i % 2 != 0;
        if (t == "equalto" || t == "eq" || t == "sameas") return equal(v, arg);
        fail("unknown test '" + t + "'");
    }

    std::vector<V> eval_args(const EP& e, size_t from) {
        std::vector<V> out;
        for (size_t k = from; k < e->a.size(); ++k) out.push_back(eval(e->a[k]));
        return out;
    }
    V kwarg(const EP& e, const char* name) {
        for (const auto& kv : e->kw)
            if (kv.first == name) return eval(kv.second);
        return V{};
    }

    V filter(const EP& e) {
        const std::string& f = e->name;
        V v = eval(e->a[0]);
        std::vector<V> args = eval_args(e, 1);
        auto arg = [&](size_t k) { return k < args.size() ? args[k] : V{}; };
        if (f == "length" || f == "count") {
            if (v.t == V::Str) return V::integer(static_cast<int64_t>(v.s.size()));
            if (v.t == V::Lst) return V::integer(static_cast<int64_t>(v.l->size()));
            if (v.t == V::Dct) return V::integer(static_cast<int64_t>(v.d->size()));
            return V::integer(0);
        }
        if (f == "trim") return V::str(strip(to_str(v), " \t\n\r\f\v", true, true));
        if (f == "upper" || f == "lower") {
            std::string s = to_str(v);
            for (auto& c : s) c = static_cast<char>(f == "upper" ? std::toupper(static_cast<unsigned char>(c)) : std::tolower(static_cast<unsigned char>(c)));
            return V::str(s);
        }
        if (f == "capitalize" || f == "title") return method(V::str(to_str(v)), f, args);
        if (f == "tojson") {
            V ind = kwarg(e, "indent");
            if (ind.t == V::Undef && !args.empty()) ind = args[0];
            std::string o;
            to_json(o, v, ind.t == V::Int ? static_cast<int>(ind.i) : -1, 0);
            return V::str(o);
        }
        if (f == "string") return V::str(to_str(v));
        if (f == "int") return V::integer(v.t == V::Str ? std::strtoll(v.s.c_str(), nullptr, 10) : static_cast<int64_t>(v.as_f()));
        if (f == "float") return V::real(v.t == V::Str ? std::strtod(v.s.c_str(), nullptr) : v.as_f());
        if (f == "list") return V::list(iterate(v));
        if (f == "first" || f == "last") {
            List l = iterate(v);
            if (l.empty()) return V{};
            return f == "first" ? l.front() : l.back();
        }
        if (f == "join") {
            std::string o, sep = to_str(arg(0));
            V attr = kwarg(e, "attribute");
            List l = iterate(v);
            for (size_t k = 0; k < l.size(); ++k) {
                const V& x = attr.t == V::Str ? (l[k].get(attr.s) ? *l[k].get(attr.s) : V{}) : l[k];
                o += (k ? sep : "") + to_str(x);
            }
            return V::str(o);
        }
        if (f == "replace") return method(V::str(to_str(v)), "replace", args);
        if (f == "default" || f == "d") {
            const bool boolean = truthy(arg(1));
            return (v.t == V::Undef || (boolean && !truthy(v))) ? arg(0) : v;
        }
        if (f == "items" || f == "keys" || f == "values") return method(v, f, {});
        if (f == "reverse") {
            if (v.t == V::Str) return V::str(std::string(v.s.rbegin(), v.s.rend()));
            List l = iterate(v);
            std::reverse(l.begin(), l.end());
            return V::list(std::move(l));
        }
        if (f == "unique") {
            List out;
            for (const V& x : iterate(v))
                if (std::none_of(out.begin(), out.end(), [&](const V& y) { return equal(x, y); })) out.push_back(x);
            return V::list(std::move(out));
        }
        if (f == "sort") {
            List l = iterate(v);
            V attr = kwarg(e, "attribute");
            const bool rev = truthy(kwarg(e, "reverse"));
            auto key = [&](const V& x) { return attr.t == V::Str && x.get(attr.s) ? *x.get(attr.s) : x; };
            std::stable_sort(l.begin(), l.end(), [&](const V& a, const V& b) { return rev ? less(key(b), key(a)) : less(key(a), key(b)); });
            return V::list(std::move(l));
        }
        if (f == "abs") return v.t == V::Int ? V::integer(v.i < 0 ? -v.i : v.i) : V::real(std::fabs(v.as_f()));
        if (f == "round") {
            const double p = std::pow(10.0, static_cast<double>(arg(0).t == V::Int ? arg(0).i : 0));
            return V::real(std::round(v.as_f() * p) / p);
        }
        if (f == "safe") return v;
        if (f == "e" || f == "escape") {
            std::string o;
            for (char c : to_str(v)) {
                switch (c) {
                    case '&': o += "&amp;"; break;
                    case '<': o += "&lt;"; break;
                    case '>': o += "&gt;"; break;
                    case '"': o += "&#34;"; break;
                    case '\'': o += "&#39;"; break;
                    default: o += c;
                }
            }
            return V::str(o);
        }
        if (f == "indent") {
            const int64_t w = arg(0).t == V::Int ? arg(0).i : 4;
            const bool first = truthy(kwarg(e, "first")) || truthy(arg(1));
            const std::string pad(static_cast<size_t>(std::max<int64_t>(0, std::min<int64_t>(w, 256))), ' ');
            std::string s = to_str(v), o = first ? pad : "";
            for (size_t k = 0; k < s.size(); ++k) {
                o += s[k];
                if (s[k] == '\n' && k + 1 < s.size() && s[k + 1] != '\n') o += pad;
            }
            return V::str(o);
        }
        if (f == "map") {
            V attr = kwarg(e, "attribute");
            List out;
            for (const V& x : iterate(v)) {
                if (attr.t == V::Str) out.push_back(x.get(attr.s) ? *x.get(attr.s) : V{});
                else if (!args.empty()) {
                    auto fe = std::make_shared<Expr>();
                    fe->k = Expr::Filter;
                    fe->name = to_str(args[0]);
                    auto lit = std::make_shared<Expr>();
                    lit->lit = x;
                    fe->a = {lit};
                    out.push_back(filter(fe));
                } else out.push_back(x);
            }
            return V::list(std::move(out));
        }
        if (f == "selectattr" || f == "rejectattr") {
            List out;
            const std::string attr = to_str(arg(0));
            for (const V& x : iterate(v)) {
                const V a = x.get(attr) ? *x.get(attr) : V{};
                bool keep = args.size() >= 2 ? test(to_str(args[1]) == "==" ? "eq" : to_str(args[1]), a, arg(2)) : truthy(a);
                if (f == "rejectattr") keep = !keep;
                if (keep) out.push_back(x);
            }
            return V::list(std::move(out));
        }
        if (f == "sum") {
            double s = 0;
            bool all_int = true;
            for (const V& x : iterate(v)) { s += x.as_f(); all_int = all_int && x.t == V::Int; }
            return all_int ? V::integer(static_cast<int64_t>(s)) : V::real(s);
        }
        if (f == "min" || f == "max") {
            List l = iterate(v);
            if (l.empty()) return V{};
            V best = l[0];
            for (const V& x : l) if (f == "min" ? less(x, best) : less(best, x)) best = x;
            return best;
        }
        if (f == "wordcount") {
            int64_t n = 0;
            bool in = false;
            for (char c : to_str(v)) { const bool w = !std::isspace(static_cast<unsigned char>(c)); if (w && !in) ++n; in = w; }
            return V::integer(n);
        }
        fail("unknown filter '" + f + "'");
    }

    static std::string strip(const std::string& s, const std::string& chars, bool left, bool right) {
        size_t a = 0, b = s.size();
        if (left) while (a < b && chars.find(s[a]) != std::string::npos) ++a;
        if (right) while (b > a && chars.find(s[b - 1]) != std::string::npos) --b;
        return s.substr(a, b - a);
    }

    V method(const V& o, const std::string& m, const std::vector<V>& args) {
        auto arg = [&](size_t k) { return k < args.size() ? args[k] : V{}; };
        if (o.t == V::Str) {
            const std::string& s = o.s;
            if (m == "startswith" || m == "endswith") {
                List cands = arg(0).t == V::Lst ? *arg(0).l : List{arg(0)};
                for (const V& c : cands) {
                    const std::string p = to_str(c);
                    if (m == "startswith" ? s.compare(0, p.size(), p) == 0 && s.size() >= p.size()
                                          : s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0) return V::boolean(true);
                }
                return V::boolean(false);
            }
            if (m == "strip" || m == "lstrip" || m == "rstrip") {
                const std::string chars = arg(0).t == V::Str ? arg(0).s : std::string(" \t\n\r\f\v");
                return V::str(strip(s, chars, m != "rstrip", m != "lstrip"));
            }
            if (m == "split") {
                List out;
                if (arg(0).t != V::Str) {  // whitespace split
                    std::string cur;
                    for (char c : s) {
                        if (std::isspace(static_cast<unsigned char>(c))) { if (!cur.empty()) out.push_back(V::str(cur)); cur.clear(); }
                        else cur += c;
                    }
                    if (!cur.empty()) out.push_back(V::str(cur));
                    return V::list(std::move(out));
                }
                const std::string sep = arg(0).s;
                if (sep.empty()) fail("empty separator");
                int64_t maxsplit = arg(1).t == V::Int ? arg(1).i : -1;
                size_t a = 0;
                for (size_t k; maxsplit != 0 && (k = s.find(sep, a)) != std::string::npos; a = k + sep.size(), --maxsplit)
                    out.push_back(V::str(s.substr(a, k - a)));
                out.push_back(V::str(s.substr(a)));
                return V::list(std::move(out));
            }
            if (m == "splitlines") {
                List out;
                size_t a = 0;
                for (size_t k = 0; k < s.size(); ++k)
                    if (s[k] == '\n') { out.push_back(V::str(s.substr(a, k - a))); a = k + 1; }
                if (a < s.size()) out.push_back(V::str(s.substr(a)));
                return V::list(std::move(out));
            }
            if (m == "upper" || m == "lower") {
                std::string x = s;
                for (auto& c : x) c = static_cast<char>(m == "upper" ? std::toupper(static_cast<unsigned char>(c)) : std::tolower(static_cast<unsigned char>(c)));
                return V::str(x);
            }
            if (m == "capitalize" || m == "title") {
                std::string x = s;
                bool start = true;
                for (size_t k = 0; k < x.size(); ++k) {
                    const auto c = static_cast<unsigned char>(x[k]);
                    if (m == "capitalize") x[k] = static_cast<char>(k == 0 ? std::toupper(c) : std::tolower(c));
                    else { x[k] = static_cast<char>(start ? std::toupper(c) : std::tolower(c)); start = !std::isalpha(c); }
                }
                return V::str(x);
            }
            if (m == "replace") {
                const std::string a = to_str(arg(0)), b = to_str(arg(1));
                if (a.empty()) return o;
                std::string x;
                size_t p = 0;
                for (size_t k; (k = s.find(a, p)) != std::string::npos; p = k + a.size()) x += s.substr(p, k - p) + b;
                return V::str(x + s.substr(p));
            }
            if (m == "find") {
                const size_t k = s.find(to_str(arg(0)));
                return V::integer(k == std::string::npos ? -1 : static_cast<int64_t>(k));
            }
            if (m == "count") {
                const std::string a = to_str(arg(0));
                int64_t n = 0;
                if (!a.empty()) for (size_t k = s.find(a); k != std::string::npos; k = s.find(a, k + a.size())) ++n;
                return V::integer(n);
            }
        }
        if (o.t == V::Dct) {
            if (m == "get") { const V* v = o.get(to_str(arg(0))); return v ? *v : (args.size() > 1 ? args[1] : V::none()); }
            if (m == "items") { List l; for (const auto& kv : *o.d) l.push_back(V::list({V::str(kv.first), kv.second})); return V::list(std::move(l)); }
            if (m == "keys") { List l; for (const auto& kv : *o.d) l.push_back(V::str(kv.first)); return V::list(std::move(l)); }
            if (m == "values") { List l; for (const auto& kv : *o.d) l.push_back(kv.second); return V::list(std::move(l)); }
        }
        if (o.t == V::Lst && m == "append") {
            o.l->push_back(arg(0));
            return V::none();
        }
        fail("unknown method '" + m + "' on " + repr(o));
    }

    V call(const EP& e) {
        const EP& callee = e->a[0];
        std::vector<V> args = eval_args(e, 1);
        if (callee->k == Expr::Attr) return method(eval(callee->a[0]), callee->name, args);
        if (callee->k != Expr::Name) fail("unsupported call");
        const std::string& f = callee->name;
        if (V* m = lookup(f); m && m->t == V::Fn) return call_macro(*m->fn, args, e);
        if (f == "raise_exception") fail("raise_exception: " + (args.empty() ? std::string() : to_str(args[0])));
        if (f == "range") {
            int64_t a = 0, b = 0, s = 1;
            if (args.size() == 1) b = args[0].i;
            else if (args.size() >= 2) { a = args[0].i; b = args[1].i; if (args.size() > 2) s = args[2].i; }
            if (s == 0) fail("range step cannot be zero");
            List l;
            for (int64_t k = a; s > 0 ? k < b : k > b; k += s) {
                if (l.size() >= 100000) fail("range() longer than 100000");
                l.push_back(V::integer(k));
            }
            return V::list(std::move(l));
        }
        if (f == "namespace" || f == "dict") {
            Dict d;
            for (const auto& kv : e->kw) d.emplace_back(kv.first, eval(kv.second));
            return V::dict(std::move(d));
        }
        if (f == "strftime_now") return V::str(date_);
        fail("unknown function '" + f + "'");
    }

    V call_macro(const Macro& m, const std::vector<V>& args, const EP& e) {
        if (++depth_ > kMaxDepth) fail("macro recursion too deep");
        scopes_.emplace_back();
        const NP& n = m.node;
        for (size_t k = 0; k < n->vars.size(); ++k) {
            V v = k < args.size() ? args[k] : V{};
            for (const auto& kv : e->kw) if (kv.first == n->vars[k]) v = eval(kv.second);
            if (v.t == V::Undef && n->defaults[k]) v = eval(n->defaults[k]);
            scopes_.back()[n->vars[k]] = v;
        }
        std::string saved;
        saved.swap(out_);
        render(n->else_body);
        saved.swap(out_);
        scopes_.pop_back();
        --depth_;
        return V::str(saved);
    }

    std::string& out_;
    std::string date_;
    std::vector<std::unordered_map<std::string, V>> scopes_;
    int depth_ = 0;
};

V message_value(const ChatMessage& m) {
    Dict d{{"role", V::str(m.role)}, {"content", V::str(m.content)}};
    if (!m.reasoning_content.empty()) d.emplace_back("reasoning_content", V::str(m.reasoning_content));
    if (!m.tool_calls.empty()) {
        List calls;
        for (const auto& tc : m.tool_calls) {
            json::Value j;
            V args = json::Parser(tc.arguments).parse(j) ? from_json(j) : V::str(tc.arguments);
            Dict fn{{"name", V::str(tc.name)}, {"arguments", args}};
            Dict c{{"id", V::str(tc.id)}, {"type", V::str("function")}, {"function", V::dict(std::move(fn))}};
            calls.push_back(V::dict(std::move(c)));
        }
        d.emplace_back("tool_calls", V::list(std::move(calls)));
    }
    if (!m.tool_call_id.empty()) d.emplace_back("tool_call_id", V::str(m.tool_call_id));
    if (!m.name.empty()) d.emplace_back("name", V::str(m.name));
    return V::dict(std::move(d));
}

} // namespace

// ================================================================================================= Template
Template::Template() = default;
Template::~Template() = default;
Template::Template(Template&&) noexcept = default;
Template& Template::operator=(Template&&) noexcept = default;

bool Template::parse(std::string_view source, std::string* err) {
    try {
        root_ = StmtParser(lex_template(source)).parse();
        return true;
    } catch (const Error& e) {
        if (err) *err = e.what();
        root_.reset();
        return false;
    }
}

bool Template::render(const std::vector<ChatMessage>& messages, const RenderOptions& opt, std::string& out,
                      std::string* err) const {
    out.clear();
    if (!root_) { if (err) *err = "template not parsed"; return false; }
    try {
        Renderer r(out, opt.date_string);
        List msgs;
        for (const auto& m : messages) msgs.push_back(message_value(m));
        r.set_global("messages", V::list(std::move(msgs)));
        r.set_global("add_generation_prompt", V::boolean(opt.add_generation_prompt));
        r.set_global("bos_token", V::str(opt.bos_token));
        r.set_global("eos_token", V::str(opt.eos_token));
        r.set_global("date_string", V::str(opt.date_string));
        if (opt.enable_thinking >= 0) r.set_global("enable_thinking", V::boolean(opt.enable_thinking != 0));
        if (!opt.tools_json.empty()) {
            json::Value j;
            if (!json::Parser(opt.tools_json).parse(j) || !j.is_array()) fail("tools_json is not a JSON array");
            r.set_global("tools", from_json(j));
        }
        r.render(root_);
        return true;
    } catch (const Error& e) {
        if (err) *err = e.what();
        return false;
    }
}

bool Template::render_json(std::string_view vars_json, std::string& out, std::string* err) const {
    out.clear();
    if (!root_) { if (err) *err = "template not parsed"; return false; }
    try {
        json::Value j;
        if (!json::Parser(vars_json).parse(j) || !j.is_object()) fail("variables are not a JSON object");
        Renderer r(out, "26 Jul 2024");
        for (const auto& kv : j.obj) r.set_global(kv.first, from_json(kv.second));
        r.render(root_);
        return true;
    } catch (const Error& e) {
        if (err) *err = e.what();
        return false;
    }
}

// ================================================================================================= built-ins
namespace {

const char* const kChatml =
    "{%- for message in messages %}{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}";

const char* const kQwen3 = R"JINJA({%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %})JINJA";

const char* const kLlama3 = R"JINJA({{- bos_token }}
{%- if tools %}
{{- '<|start_header_id|>system<|end_header_id|>\n\nEnvironment: ipython\nCutting Knowledge Date: December 2023\nToday Date: ' + date_string + '\n\n' }}
{%- if messages[0].role == 'system' %}{{- messages[0].content | trim + '\n\n' }}{%- endif %}
{{- 'You have access to the following functions. To call a function, respond with JSON of the form {"name": function name, "parameters": dictionary of argument name and its value}.\n\n' }}
{%- for t in tools %}{{- t | tojson(indent=4) }}{{- '\n\n' }}{%- endfor %}
{{- '<|eot_id|>' }}
{%- endif %}
{%- for message in messages %}
{%- if not (tools and loop.first and message.role == 'system') %}
{%- if message.role == 'tool' or message.role == 'ipython' %}
{{- '<|start_header_id|>ipython<|end_header_id|>\n\n' + message.content | trim + '<|eot_id|>' }}
{%- elif message.tool_calls %}
{%- for tc in message.tool_calls %}
{{- '<|start_header_id|>assistant<|end_header_id|>\n\n{"name": "' + tc.function.name + '", "parameters": ' + tc.function.arguments | tojson + '}<|eot_id|>' }}
{%- endfor %}
{%- else %}
{{- '<|start_header_id|>' + message.role + '<|end_header_id|>\n\n' + message.content | trim + '<|eot_id|>' }}
{%- endif %}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{%- endif %})JINJA";

const char* const kDeepSeekV3 = R"JINJA({{- bos_token }}
{%- for message in messages %}{%- if message.role == 'system' %}{{- message.content }}{%- endif %}{%- endfor %}
{%- for message in messages %}
{%- if message.role == 'user' %}{{- '<｜User｜>' + message.content }}
{%- elif message.role == 'assistant' %}
{%- set content = message.content %}
{%- if '</think>' in content %}{%- set content = content.split('</think>')[-1] %}{%- endif %}
{{- '<｜Assistant｜>' + content }}
{%- for tc in message.tool_calls or [] %}
{{- '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>' + tc.function.name + '\n```json\n' + tc.function.arguments | tojson + '\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>' }}
{%- endfor %}
{{- '<｜end▁of▁sentence｜>' }}
{%- elif message.role == 'tool' %}{{- '<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message.content + '<｜tool▁output▁end｜><｜tool▁outputs▁end｜>' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<｜Assistant｜>' }}{%- if enable_thinking is defined and enable_thinking %}{{- '<think>\n' }}{%- endif %}{%- endif %})JINJA";

const char* const kMistralTekken = R"JINJA({{- bos_token }}
{%- set sys = namespace(text='') %}
{%- if messages and messages[0].role == 'system' %}{%- set sys.text = messages[0].content %}{%- endif %}
{%- if tools %}{{- '[AVAILABLE_TOOLS]' + tools | tojson + '[/AVAILABLE_TOOLS]' }}{%- endif %}
{%- for message in messages %}
{%- if message.role == 'user' %}
{{- '[INST]' + (sys.text + '\n\n' if sys.text and loop.index0 <= 1 else '') + message.content + '[/INST]' }}
{%- elif message.role == 'assistant' %}
{%- if message.tool_calls %}{{- '[TOOL_CALLS][' }}{%- for tc in message.tool_calls %}{{- '{"name": "' + tc.function.name + '", "arguments": ' + tc.function.arguments | tojson + ', "id": "' + tc.id + '"}' }}{%- if not loop.last %}{{- ', ' }}{%- endif %}{%- endfor %}{{- ']' + eos_token }}
{%- else %}{{- message.content + eos_token }}{%- endif %}
{%- elif message.role == 'tool' %}
{{- '[TOOL_RESULTS]{"content": ' + message.content | tojson + ', "call_id": "' + message.tool_call_id + '"}[/TOOL_RESULTS]' }}
{%- endif %}
{%- endfor %})JINJA";

const char* const kGemma = R"JINJA({{- bos_token }}
{%- set first_user_prefix = '' %}
{%- if messages and messages[0].role == 'system' %}{%- set first_user_prefix = messages[0].content + '\n\n' %}{%- set loop_messages = messages[1:] %}{%- else %}{%- set loop_messages = messages %}{%- endif %}
{%- for message in loop_messages %}
{%- set role = 'model' if message.role == 'assistant' else message.role %}
{{- '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else '') + message.content | trim + '<end_of_turn>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<start_of_turn>model\n' }}{%- endif %})JINJA";

} // namespace

const char* builtin_template(std::string_view family) {
    if (family == "chatml") return kChatml;
    if (family == "qwen3") return kQwen3;
    if (family == "llama3") return kLlama3;
    if (family == "deepseek-v3") return kDeepSeekV3;
    if (family == "mistral-tekken") return kMistralTekken;
    if (family == "gemma") return kGemma;
    return nullptr;
}

std::string detect_template_family(std::string_view tmpl, std::string_view arch) {
    auto has = [&](const char* s) { return tmpl.find(s) != std::string_view::npos; };
    if (has("<|start_header_id|>")) return "llama3";
    if (has("<start_of_turn>")) return "gemma";
    if (has("[INST]") || has("[AVAILABLE_TOOLS]")) return "mistral-tekken";
    if (has("<｜User｜>") || has("<｜Assistant｜>")) return "deepseek-v3";
    if (has("<|im_start|>")) return (has("<think>") || has("tool_call")) ? "qwen3" : "chatml";
    if (arch.rfind("llama", 0) == 0) return "llama3";
    if (arch.rfind("gemma", 0) == 0) return "gemma";
    if (arch.rfind("deepseek", 0) == 0) return "deepseek-v3";
    if (arch.rfind("mistral", 0) == 0) return "mistral-tekken";
    if (arch.rfind("qwen3", 0) == 0 || arch.rfind("qwen35", 0) == 0) return "qwen3";
    return "chatml";
}

// ================================================================================================= reasoning
Reasoning split_reasoning(std::string_view text, std::string_view open, std::string_view close) {
    Reasoning r;
    const size_t o = text.find(open);
    const size_t c = text.find(close);
    auto lstrip_nl = [](std::string_view s) { while (!s.empty() && (s.front() == '\n' || s.front() == '\r')) s.remove_prefix(1); return s; };
    auto strip_nl = [&](std::string_view s) { s = lstrip_nl(s); while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.remove_suffix(1); return s; };
    if (c != std::string_view::npos && (o == std::string_view::npos || o < c)) {
        const size_t start = o == std::string_view::npos ? 0 : o + open.size();
        r.reasoning = std::string(strip_nl(text.substr(start, c - start)));
        r.content = std::string(text.substr(0, o == std::string_view::npos ? 0 : o)) +
                    std::string(lstrip_nl(text.substr(c + close.size())));
        return r;
    }
    if (o != std::string_view::npos) {
        r.content = std::string(text.substr(0, o));
        r.reasoning = std::string(lstrip_nl(text.substr(o + open.size())));
        r.closed = false;
        return r;
    }
    r.content = std::string(text);
    return r;
}

ReasoningSplitter::ReasoningSplitter(bool started_inside, std::string open, std::string close)
    : open_(std::move(open)), close_(std::move(close)), inside_(started_inside) {}

void ReasoningSplitter::emit(std::string_view s, std::string& r, std::string& c) {
    if (inside_) {
        r.append(s);
        return;
    }
    if (strip_nl_) {
        while (!s.empty() && (s.front() == '\n' || s.front() == '\r')) s.remove_prefix(1);
        if (!s.empty()) strip_nl_ = false;
    }
    c.append(s);
}

void ReasoningSplitter::feed(std::string_view chunk, std::string& r, std::string& c) {
    std::string buf = hold_ + std::string(chunk);
    hold_.clear();
    size_t i = 0;
    while (i < buf.size()) {
        const std::string& tag = inside_ ? close_ : (seen_close_ ? std::string() : open_);
        if (tag.empty()) { emit(std::string_view(buf).substr(i), r, c); return; }
        const size_t k = buf.find(tag, i);
        if (k != std::string::npos) {
            emit(std::string_view(buf).substr(i, k - i), r, c);
            i = k + tag.size();
            if (inside_) { inside_ = false; seen_close_ = true; strip_nl_ = true; }
            else inside_ = true;
            continue;
        }
        // Hold back the longest suffix that is a prefix of the tag (it may complete in the next chunk).
        size_t keep = 0;
        for (size_t n = std::min(tag.size() - 1, buf.size() - i); n > 0; --n)
            if (buf.compare(buf.size() - n, n, tag, 0, n) == 0) { keep = n; break; }
        emit(std::string_view(buf).substr(i, buf.size() - i - keep), r, c);
        hold_ = buf.substr(buf.size() - keep);
        return;
    }
}

void ReasoningSplitter::finish(std::string& r, std::string& c) {
    const std::string h = std::move(hold_);
    hold_.clear();
    emit(h, r, c);
}

} // namespace vegajet::chat
