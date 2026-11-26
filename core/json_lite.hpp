// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Minimal JSON value, parser and string escaper (RFC 8259 subset: no duplicate-key semantics, depth-limited).
// Header-only, zero dependencies. Used by the OpenAI server and its tests.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vegajet::json {

struct Value {
    enum class Type { Null, Bool, Number, String, Array, Object } type = Type::Null;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::vector<Value> arr;
    std::vector<std::pair<std::string, Value>> obj;

    bool is_null() const { return type == Type::Null; }
    bool is_bool() const { return type == Type::Bool; }
    bool is_number() const { return type == Type::Number; }
    bool is_string() const { return type == Type::String; }
    bool is_array() const { return type == Type::Array; }
    bool is_object() const { return type == Type::Object; }
    const Value* get(std::string_view key) const {
        if (type != Type::Object) return nullptr;
        for (const auto& kv : obj) if (kv.first == key) return &kv.second;
        return nullptr;
    }
};

class Parser {
public:
    // max_nodes bounds the Value tree (one node per value), so a body made of delimiters ("[[[[…]]]]", "[{},{},…]")
    // cannot cost hundreds of bytes of heap per input byte.
    explicit Parser(std::string_view s, int max_depth = 64, size_t max_nodes = 1u << 20)
        : s_(s), max_depth_(max_depth), max_nodes_(max_nodes) {}
    bool parse(Value& out) {
        ws();
        if (!val(out, 0)) return false;
        ws();
        return i_ == s_.size();
    }
    size_t error_offset() const { return i_; }
    bool node_limit_hit() const { return limit_ == Limit::Nodes; }
    bool depth_limit_hit() const { return limit_ == Limit::Depth; }
    size_t nodes() const { return nodes_; }

private:
    enum class Limit : uint8_t { None, Depth, Nodes };
    void ws() { while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r')) ++i_; }
    bool lit(std::string_view w) {
        if (s_.substr(i_, w.size()) != w) return false;
        i_ += w.size();
        return true;
    }
    bool val(Value& v, int depth) {
        if (i_ >= s_.size()) return false;
        if (depth > max_depth_) { limit_ = Limit::Depth; return false; }
        if (++nodes_ > max_nodes_) { limit_ = Limit::Nodes; return false; }
        const char c = s_[i_];
        if (c == '{') return object(v, depth);
        if (c == '[') return array(v, depth);
        if (c == '"') { v.type = Value::Type::String; return string(v.str); }
        if (c == 't') { v.type = Value::Type::Bool; v.b = true; return lit("true"); }
        if (c == 'f') { v.type = Value::Type::Bool; v.b = false; return lit("false"); }
        if (c == 'n') { v.type = Value::Type::Null; return lit("null"); }
        return number(v);
    }
    bool number(Value& v) {
        const size_t start = i_;
        if (i_ < s_.size() && s_[i_] == '-') ++i_;
        auto digits = [&] { size_t n = 0; while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') { ++i_; ++n; } return n; };
        if (!digits()) return false;
        if (i_ < s_.size() && s_[i_] == '.') { ++i_; if (!digits()) return false; }
        if (i_ < s_.size() && (s_[i_] == 'e' || s_[i_] == 'E')) {
            ++i_;
            if (i_ < s_.size() && (s_[i_] == '+' || s_[i_] == '-')) ++i_;
            if (!digits()) return false;
        }
        const std::string tok(s_.substr(start, i_ - start));
        v.type = Value::Type::Number;
        v.num = std::strtod(tok.c_str(), nullptr);
        return true;
    }
    static void utf8(std::string& o, uint32_t cp) {
        if (cp < 0x80) o += static_cast<char>(cp);
        else if (cp < 0x800) { o += static_cast<char>(0xC0 | (cp >> 6)); o += static_cast<char>(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) {
            o += static_cast<char>(0xE0 | (cp >> 12)); o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            o += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            o += static_cast<char>(0xF0 | (cp >> 18)); o += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); o += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }
    bool hex4(uint32_t& cp) {
        if (i_ + 4 > s_.size()) return false;
        cp = 0;
        for (int k = 0; k < 4; ++k) {
            const char h = s_[i_++];
            cp <<= 4;
            if (h >= '0' && h <= '9') cp |= static_cast<uint32_t>(h - '0');
            else if (h >= 'a' && h <= 'f') cp |= static_cast<uint32_t>(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') cp |= static_cast<uint32_t>(h - 'A' + 10);
            else return false;
        }
        return true;
    }
    bool string(std::string& o) {
        ++i_;  // opening quote
        o.clear();
        while (i_ < s_.size()) {
            const char c = s_[i_++];
            if (c == '"') return true;
            if (static_cast<unsigned char>(c) < 0x20) return false;
            if (c != '\\') { o += c; continue; }
            if (i_ >= s_.size()) return false;
            const char e = s_[i_++];
            switch (e) {
                case '"': o += '"'; break;
                case '\\': o += '\\'; break;
                case '/': o += '/'; break;
                case 'b': o += '\b'; break;
                case 'f': o += '\f'; break;
                case 'n': o += '\n'; break;
                case 'r': o += '\r'; break;
                case 't': o += '\t'; break;
                case 'u': {
                    uint32_t cp;
                    if (!hex4(cp)) return false;
                    if (cp >= 0xD800 && cp <= 0xDBFF) {  // surrogate pair
                        uint32_t lo;
                        if (i_ + 2 > s_.size() || s_[i_] != '\\' || s_[i_ + 1] != 'u') return false;
                        i_ += 2;
                        if (!hex4(lo) || lo < 0xDC00 || lo > 0xDFFF) return false;
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        return false;
                    }
                    utf8(o, cp);
                    break;
                }
                default: return false;
            }
        }
        return false;
    }
    bool array(Value& v, int depth) {
        ++i_;
        v.type = Value::Type::Array;
        ws();
        if (i_ < s_.size() && s_[i_] == ']') { ++i_; return true; }
        while (true) {
            Value e;
            ws();
            if (!val(e, depth + 1)) return false;
            v.arr.push_back(std::move(e));
            ws();
            if (i_ >= s_.size()) return false;
            if (s_[i_] == ',') { ++i_; continue; }
            if (s_[i_] == ']') { ++i_; return true; }
            return false;
        }
    }
    bool object(Value& v, int depth) {
        ++i_;
        v.type = Value::Type::Object;
        ws();
        if (i_ < s_.size() && s_[i_] == '}') { ++i_; return true; }
        while (true) {
            ws();
            if (i_ >= s_.size() || s_[i_] != '"') return false;
            std::string k;
            if (!string(k)) return false;
            ws();
            if (i_ >= s_.size() || s_[i_] != ':') return false;
            ++i_;
            ws();
            Value e;
            if (!val(e, depth + 1)) return false;
            v.obj.emplace_back(std::move(k), std::move(e));
            ws();
            if (i_ >= s_.size()) return false;
            if (s_[i_] == ',') { ++i_; continue; }
            if (s_[i_] == '}') { ++i_; return true; }
            return false;
        }
    }

    std::string_view s_;
    size_t i_ = 0;
    int max_depth_;
    size_t max_nodes_;
    size_t nodes_ = 0;
    Limit limit_ = Limit::None;
};

// Convenience parse with explicit bounds. A body above them fails cleanly (Parser::node_limit_hit() /
// depth_limit_hit()); the caller answers 413/400 instead of letting the allocation abort the process.
inline bool parse(std::string_view s, Value& out, int max_depth = 64, size_t max_nodes = 1u << 20) {
    return Parser(s, max_depth, max_nodes).parse(out);
}

// Escapes for a JSON string body (no surrounding quotes). Bytes >= 0x80 pass through (input must be valid UTF-8).
inline std::string escape(std::string_view s) {
    std::string o;
    o.reserve(s.size() + 8);
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
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    o += buf;
                } else {
                    o += static_cast<char>(c);
                }
        }
    }
    return o;
}

inline std::string quote(std::string_view s) { return "\"" + escape(s) + "\""; }

} // namespace vegajet::json
