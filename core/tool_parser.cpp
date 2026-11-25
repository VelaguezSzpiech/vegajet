// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/tool_parser.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <initializer_list>

namespace vegajet::toolparse {
namespace {

using toolcall::Event;

constexpr std::string_view kThinkOpen = "<think>";
constexpr std::string_view kThinkClose = "</think>";
constexpr std::string_view kQwenOpen = "<tool_call>";
constexpr std::string_view kQwenClose = "</tool_call>";
constexpr std::string_view kDsCallsBegin = "<\xEF\xBD\x9Ctool\xE2\x96\x81" "calls\xE2\x96\x81" "begin\xEF\xBD\x9C>";
constexpr std::string_view kDsCallsEnd = "<\xEF\xBD\x9Ctool\xE2\x96\x81" "calls\xE2\x96\x81" "end\xEF\xBD\x9C>";
constexpr std::string_view kDsCallBegin = "<\xEF\xBD\x9Ctool\xE2\x96\x81" "call\xE2\x96\x81" "begin\xEF\xBD\x9C>";
constexpr std::string_view kDsCallEnd = "<\xEF\xBD\x9Ctool\xE2\x96\x81" "call\xE2\x96\x81" "end\xEF\xBD\x9C>";
constexpr std::string_view kDsSep = "<\xEF\xBD\x9Ctool\xE2\x96\x81" "sep\xEF\xBD\x9C>";
constexpr std::string_view kPythonTag = "<|python_tag|>";
constexpr std::string_view kEom = "<|eom_id|>";
constexpr std::string_view kEot = "<|eot_id|>";
constexpr std::string_view kFuncOpen = "<function=";
constexpr std::string_view kFuncClose = "</function>";

// Call kinds (which opening marker started the call)
enum Kind : int { kQwen = 1, kDsDirect = 2, kDsInBlock = 3, kPython = 4, kFunc = 5 };

inline bool is_ws(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

std::string_view trim(std::string_view s) {
    while (!s.empty() && is_ws(s.front())) s.remove_prefix(1);
    while (!s.empty() && is_ws(s.back())) s.remove_suffix(1);
    return s;
}

// Longest suffix of buf that is a proper prefix of one of the markers.
size_t hold(std::string_view buf, std::initializer_list<std::string_view> markers) {
    size_t best = 0;
    for (std::string_view m : markers) {
        const size_t lim = std::min(buf.size(), m.size() - 1);
        for (size_t k = lim; k > best; --k)
            if (buf.substr(buf.size() - k) == m.substr(0, k)) {
                best = k;
                break;
            }
    }
    return best;
}

// Earliest full occurrence of any marker; which = its index.
size_t find_first(std::string_view buf, std::initializer_list<std::string_view> markers, size_t& which) {
    size_t best = std::string_view::npos, k = 0;
    for (std::string_view m : markers) {
        const size_t p = buf.find(m);
        if (p < best) {
            best = p;
            which = k;
        }
        ++k;
    }
    return best;
}

bool starts_with(std::string_view s, std::string_view p) { return s.substr(0, p.size()) == p; }

// json::parse appends into a non-empty value after a failed attempt: always start from a fresh one.
bool parse_fresh(std::string_view text, json::Value& v) {
    v = json::Value{};
    return json::parse(text, v);
}

std::string strip_fences(std::string_view s) {
    s = trim(s);
    if (starts_with(s, "```")) {
        const size_t nl = s.find('\n');
        s = nl == std::string_view::npos ? std::string_view{} : s.substr(nl + 1);
        s = trim(s);
        if (s.size() >= 3 && s.substr(s.size() - 3) == "```") s.remove_suffix(3);
    }
    return std::string(trim(s));
}

bool parse_maybe_repair(std::string_view text, bool repair, json::Value& v, bool& repaired) {
    repaired = false;
    const std::string t = strip_fences(text);
    if (parse_fresh(t, v)) {
        repaired = t.size() != trim(text).size();
        return true;
    }
    if (!repair) return false;
    bool changed = false;
    const std::string fixed = repair_json(t, &changed);
    if (fixed.empty() || !parse_fresh(fixed, v)) return false;
    repaired = true;
    return true;
}

bool args_text(const json::Value* a, bool repair, std::string& out, bool& repaired) {
    if (!a) {
        out = "{}";
        return true;
    }
    if (a->is_object()) {
        out = toolcall::dump_json(*a, false);
        return true;
    }
    if (a->is_string()) {
        json::Value inner;
        bool r = false;
        if (!parse_maybe_repair(a->str, repair, inner, r) || !inner.is_object()) return false;
        repaired = repaired || r;
        out = toolcall::dump_json(inner, false);
        return true;
    }
    return false;
}

// {"name":..,"arguments"|"parameters":..} or {"type":"function","function":{...}}
bool call_from_object(const json::Value& v, bool repair, std::string& name, std::string& args, bool& repaired) {
    if (!v.is_object()) return false;
    if (const json::Value* f = v.get("function"); f && f->is_object()) return call_from_object(*f, repair, name, args, repaired);
    const json::Value* n = v.get("name");
    if (!n || !n->is_string() || n->str.empty()) return false;
    name = n->str;
    const json::Value* a = v.get("arguments");
    if (!a) a = v.get("parameters");
    return args_text(a, repair, args, repaired);
}

// Splits on top-level ';' (outside strings / brackets).
std::vector<std::string_view> split_calls(std::string_view s) {
    std::vector<std::string_view> out;
    int depth = 0;
    bool in_str = false, esc = false;
    size_t start = 0;
    for (size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (in_str) {
            if (esc) esc = false;
            else if (c == '\\') esc = true;
            else if (c == '"') in_str = false;
            continue;
        }
        if (c == '"') in_str = true;
        else if (c == '{' || c == '[') ++depth;
        else if (c == '}' || c == ']') --depth;
        else if (c == ';' && depth <= 0) {
            out.push_back(s.substr(start, i - start));
            start = i + 1;
        }
    }
    out.push_back(s.substr(start));
    std::vector<std::string_view> kept;
    for (auto p : out)
        if (!trim(p).empty()) kept.push_back(trim(p));
    return kept;
}

// End of the first complete top-level JSON object in s (index past '}'), or npos.
size_t object_end(std::string_view s) {
    int depth = 0;
    bool in_str = false, esc = false;
    for (size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (in_str) {
            if (esc) esc = false;
            else if (c == '\\') esc = true;
            else if (c == '"') in_str = false;
            continue;
        }
        if (c == '"') in_str = true;
        else if (c == '{' || c == '[') ++depth;
        else if (c == '}' || c == ']') {
            if (--depth == 0) return i + 1;
        }
    }
    return std::string_view::npos;
}

} // namespace

const char* format_name(Format f) {
    switch (f) {
        case Format::Qwen: return "qwen";
        case Format::DeepSeek: return "deepseek";
        case Format::Llama: return "llama";
    }
    return "?";
}

bool format_from_name(std::string_view n, Format& out) {
    std::string s(n);
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (s == "qwen" || s == "qwen2" || s == "qwen3" || s == "hermes" || s == "chatml" || s == "ornith") out = Format::Qwen;
    else if (s.rfind("deepseek", 0) == 0) out = Format::DeepSeek;
    else if (s.rfind("llama", 0) == 0) out = Format::Llama;
    else return false;
    return true;
}

// ------------------------------------------------------------------------------------------------- repair_json

std::string repair_json(std::string_view text, bool* changed) {
    if (changed) *changed = false;
    const std::string t = strip_fences(text);
    {
        json::Value v;
        if (parse_fresh(t, v)) {
            if (changed) *changed = t.size() != text.size();
            return toolcall::dump_json(v, false);
        }
    }
    if (changed) *changed = true;
    size_t start = t.find_first_of("{[");
    if (start == std::string::npos) return {};
    std::string out;
    std::string stack;  // expected closers
    struct Cut {
        size_t out_len;
        std::string stack;
    };
    std::vector<Cut> cuts;  // after each top-level-of-its-container comma: a safe truncation point
    bool done = false;
    size_t i = start;
    auto rstrip_comma = [&] {
        while (!out.empty() && is_ws(out.back())) out.pop_back();
        if (!out.empty() && out.back() == ',') out.pop_back();
    };
    while (i < t.size() && !done) {
        const char c = t[i];
        if (c == '"' || c == '\'') {
            const char q = c;
            out += '"';
            ++i;
            bool closed = false;
            while (i < t.size()) {
                const char d = t[i];
                if (d == '\\' && i + 1 < t.size()) {
                    const char e = t[i + 1];
                    if (e == '\'') out += '\'';
                    else {
                        out += '\\';
                        out += e;
                    }
                    i += 2;
                    continue;
                }
                if (d == q) {
                    ++i;
                    closed = true;
                    break;
                }
                if (d == '"') out += "\\\"";
                else if (d == '\n') out += "\\n";
                else if (d == '\r') out += "\\r";
                else if (d == '\t') out += "\\t";
                else out += d;
                ++i;
            }
            out += '"';
            if (!closed) break;
            continue;
        }
        if (c == '{' || c == '[') {
            stack += c == '{' ? '}' : ']';
            out += c;
            ++i;
            continue;
        }
        if (c == '}' || c == ']') {
            ++i;
            if (stack.empty()) {
                done = true;
                break;
            }
            const size_t m = stack.rfind(c);
            if (m == std::string::npos) continue;  // stray closer
            while (stack.size() > m) {
                rstrip_comma();
                if (!out.empty() && out.back() == ':') out += "null";
                out += stack.back();
                stack.pop_back();
            }
            if (stack.empty()) done = true;
            continue;
        }
        if (c == ',') {
            rstrip_comma();
            out += ',';
            cuts.push_back(Cut{out.size() - 1, stack});
            ++i;
            continue;
        }
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_' || c == '$') {
            size_t e = i;
            while (e < t.size() && (std::isalnum(static_cast<unsigned char>(t[e])) || t[e] == '_' || t[e] == '$')) ++e;
            const std::string w = t.substr(i, e - i);
            size_t k = e;
            while (k < t.size() && is_ws(t[k])) ++k;
            const bool key = k < t.size() && t[k] == ':';
            if (!key && (w == "true" || w == "True")) out += "true";
            else if (!key && (w == "false" || w == "False")) out += "false";
            else if (!key && (w == "null" || w == "None" || w == "nil" || w == "NaN" || w == "Infinity" || w == "undefined")) out += "null";
            else if (key) out += json::quote(w);
            else return {};  // a bare word as a value is not recoverable without guessing
            i = e;
            continue;
        }
        out += c;
        ++i;
    }
    auto close_all = [](std::string o, std::string st) {
        while (!o.empty() && is_ws(o.back())) o.pop_back();
        if (!o.empty() && o.back() == ',') o.pop_back();
        if (!o.empty() && o.back() == ':') o += "null";
        while (!st.empty()) {
            while (!o.empty() && is_ws(o.back())) o.pop_back();
            if (!o.empty() && o.back() == ',') o.pop_back();
            o += st.back();
            st.pop_back();
        }
        return o;
    };
    json::Value v;
    std::string cand = close_all(out, stack);
    if (parse_fresh(cand, v)) return toolcall::dump_json(v, false);
    // A dangling key or half value: cut back to the last comma and close from there.
    for (size_t k = cuts.size(); k-- > 0;) {
        cand = close_all(out.substr(0, cuts[k].out_len), cuts[k].stack);
        if (parse_fresh(cand, v)) return toolcall::dump_json(v, false);
    }
    return {};
}

// ------------------------------------------------------------------------------------------------- streaming

ToolStreamParser::ToolStreamParser(Options opt) : opt_(std::move(opt)) {}

void ToolStreamParser::emit(Event::Kind k, std::string_view s, std::vector<Event>& out) {
    if (s.empty()) return;
    if (!out.empty() && out.back().kind == k && k != Event::Kind::ToolCall) {
        out.back().text.append(s.data(), s.size());
        return;
    }
    Event e;
    e.kind = k;
    e.text.assign(s.data(), s.size());
    out.push_back(std::move(e));
}

bool ToolStreamParser::emit_call(std::string_view body, std::string_view name_hint, std::vector<Event>& out) {
    std::vector<std::pair<std::string, std::string>> calls;
    bool repaired = false;
    if (!name_hint.empty()) {
        // name known from the markup: body is the arguments object
        json::Value v;
        bool r = false;
        if (!parse_maybe_repair(body, opt_.repair, v, r) || !v.is_object()) return false;
        repaired = r;
        calls.emplace_back(std::string(trim(name_hint)), toolcall::dump_json(v, false));
    } else {
        for (std::string_view part : split_calls(body)) {
            json::Value v;
            bool r = false;
            if (!parse_maybe_repair(part, opt_.repair, v, r)) return false;
            std::string name, args;
            if (!call_from_object(v, opt_.repair, name, args, r)) return false;
            repaired = repaired || r;
            calls.emplace_back(std::move(name), std::move(args));
        }
        if (calls.empty()) return false;
    }
    if (!opt_.known_tools.empty())
        for (const auto& c : calls)
            if (std::find(opt_.known_tools.begin(), opt_.known_tools.end(), c.first) == opt_.known_tools.end())
                return false;
    for (auto& c : calls) {
        if (done_) break;
        Event e;
        e.kind = Event::Kind::ToolCall;
        e.index = n_calls_;
        e.call.id = opt_.id_prefix + std::to_string(n_calls_);
        e.call.name = std::move(c.first);
        e.call.arguments = std::move(c.second);
        out.push_back(std::move(e));
        ++n_calls_;
        if (!opt_.parallel) done_ = true;
    }
    if (repaired) ++n_repaired_;
    return true;
}

// The open call's text is complete in raw_ (markers included); body / name are extracted per kind.
bool ToolStreamParser::finish_call(std::vector<Event>& out, bool final) {
    (void)final;
    std::string_view r = raw_;
    // strip leading whitespace kept for the fallback
    size_t lead = 0;
    while (lead < r.size() && is_ws(r[lead])) ++lead;
    r.remove_prefix(lead);
    bool ok = false;
    auto strip_open = [&](std::string_view open) {
        if (starts_with(r, open)) r.remove_prefix(open.size());
    };
    auto strip_close = [&](std::initializer_list<std::string_view> closes) {
        for (std::string_view c : closes)
            if (r.size() >= c.size() && r.substr(r.size() - c.size()) == c) {
                r.remove_suffix(c.size());
                break;
            }
    };
    switch (call_kind_) {
        case kQwen:
            strip_open(kQwenOpen);
            strip_close({kQwenClose});
            ok = emit_call(r, {}, out);
            break;
        case kPython:
            strip_open(kPythonTag);
            strip_close({kEom, kEot});
            ok = emit_call(r, {}, out);
            break;
        case kFunc: {
            strip_open(kFuncOpen);
            strip_close({kFuncClose});
            const size_t gt = r.find('>');
            if (gt != std::string_view::npos && !trim(r.substr(0, gt)).empty())
                ok = emit_call(r.substr(gt + 1), r.substr(0, gt), out);
            break;
        }
        case kDsDirect:
        case kDsInBlock: {
            strip_open(kDsCallBegin);
            strip_close({kDsCallEnd});
            const size_t sep = r.find(kDsSep);
            if (sep == std::string_view::npos) break;
            const std::string_view head = trim(r.substr(0, sep));
            std::string_view rest = r.substr(sep + kDsSep.size());
            if (head.empty() || head == "function") {
                const size_t nl = rest.find('\n');
                const std::string_view name = trim(rest.substr(0, nl));
                if (name.empty() || name.find_first_of("{\"") != std::string_view::npos) break;
                rest = nl == std::string_view::npos ? std::string_view{} : rest.substr(nl + 1);
                ok = emit_call(rest, name, out);
            } else {
                ok = emit_call(rest, head, out);
            }
            break;
        }
        default: break;
    }
    if (!ok) {
        ++n_malformed_;
        emit(Event::Kind::Content, raw_, out);
    }
    raw_.clear();
    return ok;
}

void ToolStreamParser::push(std::string_view piece, std::vector<Event>& out) {
    if (st_ == State::Done) return;
    buf_.append(piece.data(), piece.size());
    run(out, false);
}

void ToolStreamParser::finish(std::vector<Event>& out) {
    if (st_ == State::Done) return;
    run(out, true);
    if (st_ != State::Done) {
        emit(Event::Kind::Content, buf_, out);
        buf_.clear();
    }
}

void ToolStreamParser::run(std::vector<Event>& out, bool final) {
    while (true) {
        if (done_) {
            st_ = State::Done;
            buf_.clear();
            return;
        }
        switch (st_) {
            case State::Done: return;
            case State::Start: {
                size_t i = 0;
                while (i < buf_.size() && is_ws(buf_[i])) ++i;
                if (i == buf_.size()) {
                    if (!final) return;
                    st_ = State::Content;
                    continue;
                }
                const std::string_view rest = std::string_view(buf_).substr(i);
                if (opt_.parse_reasoning) {
                    if (starts_with(rest, kThinkOpen)) {
                        buf_.erase(0, i + kThinkOpen.size());
                        st_ = State::Reasoning;
                        trim_ws_ = true;
                        continue;
                    }
                    if (!final && starts_with(kThinkOpen, rest)) return;
                }
                if (opt_.format == Format::Llama && rest.front() == '{') {
                    buf_.erase(0, i);
                    st_ = State::BareJson;
                    continue;
                }
                st_ = State::Content;
                continue;
            }
            case State::Reasoning: {
                if (trim_ws_) {
                    size_t i = 0;
                    while (i < buf_.size() && is_ws(buf_[i])) ++i;
                    buf_.erase(0, i);
                    if (buf_.empty()) {
                        if (!final) return;
                    } else {
                        trim_ws_ = false;
                    }
                }
                const size_t p = buf_.find(kThinkClose);
                if (p != std::string::npos) {
                    std::string_view body = std::string_view(buf_).substr(0, p);
                    while (!body.empty() && is_ws(body.back())) body.remove_suffix(1);
                    emit(Event::Kind::Reasoning, body, out);
                    buf_.erase(0, p + kThinkClose.size());
                    st_ = State::Content;
                    trim_ws_ = true;
                    continue;
                }
                if (final) {
                    emit(Event::Kind::Reasoning, buf_, out);
                    buf_.clear();
                    return;
                }
                size_t keep = buf_.size() - hold(buf_, {kThinkClose});
                while (keep > 0 && is_ws(buf_[keep - 1])) --keep;  // trailing whitespace before </think>
                emit(Event::Kind::Reasoning, std::string_view(buf_).substr(0, keep), out);
                buf_.erase(0, keep);
                return;
            }
            case State::Content: {
                if (trim_ws_) {
                    size_t i = 0;
                    while (i < buf_.size() && is_ws(buf_[i])) ++i;
                    buf_.erase(0, i);
                    if (buf_.empty()) return;
                    trim_ws_ = false;
                }
                size_t which = 0, p = std::string::npos;
                int kind = 0;
                std::array<std::string_view, 2> opens{kQwenOpen, kQwenOpen};
                if (opt_.format == Format::Qwen) {
                    p = find_first(buf_, {kQwenOpen}, which);
                    kind = kQwen;
                } else if (opt_.format == Format::DeepSeek) {
                    opens = {kDsCallsBegin, kDsCallBegin};
                    p = find_first(buf_, {kDsCallsBegin, kDsCallBegin}, which);
                    kind = which == 0 ? 0 : kDsDirect;
                } else {
                    opens = {kPythonTag, kFuncOpen};
                    p = find_first(buf_, {kPythonTag, kFuncOpen}, which);
                    kind = which == 0 ? kPython : kFunc;
                }
                if (p != std::string::npos) {
                    size_t c = p;
                    while (c > 0 && is_ws(buf_[c - 1])) --c;
                    emit(Event::Kind::Content, std::string_view(buf_).substr(0, c), out);
                    if (kind == 0) {  // DeepSeek calls block
                        buf_.erase(0, p + kDsCallsBegin.size());
                        st_ = State::DsBlock;
                        continue;
                    }
                    raw_ = buf_.substr(c, p - c);
                    buf_.erase(0, p);
                    call_kind_ = kind;
                    st_ = State::Call;
                    continue;
                }
                if (final) {
                    emit(Event::Kind::Content, buf_, out);
                    buf_.clear();
                    return;
                }
                size_t h = 0;
                for (std::string_view m : opens) h = std::max(h, hold(buf_, {m}));
                size_t keep = buf_.size() - h;
                while (keep > 0 && is_ws(buf_[keep - 1])) --keep;  // whitespace before a possible call
                emit(Event::Kind::Content, std::string_view(buf_).substr(0, keep), out);
                buf_.erase(0, keep);
                return;
            }
            case State::Call: {
                size_t which = 0, p = std::string::npos, mlen = 0;
                // skip the opening marker itself when searching for the close
                size_t from = 0;
                switch (call_kind_) {
                    case kQwen:
                        from = kQwenOpen.size();
                        p = buf_.find(kQwenClose, from);
                        mlen = kQwenClose.size();
                        break;
                    case kDsDirect:
                    case kDsInBlock:
                        from = kDsCallBegin.size();
                        p = buf_.find(kDsCallEnd, from);
                        mlen = kDsCallEnd.size();
                        break;
                    case kPython: {
                        from = kPythonTag.size();
                        const std::string_view tail = std::string_view(buf_).substr(std::min(from, buf_.size()));
                        const size_t q = find_first(tail, {kEom, kEot}, which);
                        if (q != std::string_view::npos) {
                            p = q + from;
                            mlen = which == 0 ? kEom.size() : kEot.size();
                        }
                        break;
                    }
                    case kFunc:
                        from = kFuncOpen.size();
                        p = buf_.find(kFuncClose, from);
                        mlen = kFuncClose.size();
                        break;
                    default: break;
                }
                if (p == std::string::npos) {
                    if (!final) return;
                    raw_ += buf_;
                    buf_.clear();
                    finish_call(out, true);
                    st_ = State::Content;
                    return;
                }
                raw_ += buf_.substr(0, p + mlen);
                buf_.erase(0, p + mlen);
                const int kind = call_kind_;
                const bool ok = finish_call(out, false);
                st_ = kind == kDsInBlock ? State::DsBlock : State::Content;
                trim_ws_ = ok;
                continue;
            }
            case State::DsBlock: {
                size_t i = 0;
                while (i < buf_.size() && is_ws(buf_[i])) ++i;
                buf_.erase(0, i);
                const std::string_view b = buf_;
                if (starts_with(b, kDsCallBegin)) {
                    raw_.clear();
                    call_kind_ = kDsInBlock;
                    st_ = State::Call;
                    continue;
                }
                if (starts_with(b, kDsCallsEnd)) {
                    buf_.erase(0, kDsCallsEnd.size());
                    st_ = State::Content;
                    trim_ws_ = true;
                    continue;
                }
                if (b.empty() || (!final && (starts_with(kDsCallBegin, b) || starts_with(kDsCallsEnd, b)))) {
                    if (final) {
                        st_ = State::Content;
                        continue;
                    }
                    return;
                }
                st_ = State::Content;  // unexpected text inside the block: treat as content
                continue;
            }
            case State::BareJson: {
                {
                    size_t i = 0;
                    while (i < buf_.size() && (is_ws(buf_[i]) || buf_[i] == ';')) ++i;
                    if (i == buf_.size()) {
                        if (!final) return;
                        buf_.clear();
                        st_ = State::Content;
                        continue;
                    }
                    if (buf_[i] != '{') {
                        st_ = State::Content;
                        trim_ws_ = true;
                        continue;
                    }
                    buf_.erase(0, i);
                }
                const size_t e = object_end(buf_);
                if (e == std::string::npos) {
                    if (!final) return;
                    raw_ = buf_;
                    buf_.clear();
                    if (!emit_call(raw_, {}, out)) emit(Event::Kind::Content, raw_, out);
                    raw_.clear();
                    st_ = State::Content;
                    return;
                }
                const std::string body = buf_.substr(0, e);
                if (!emit_call(body, {}, out)) {
                    st_ = State::Content;  // plain JSON reply: content as written
                    continue;
                }
                buf_.erase(0, e);
                continue;  // more calls may follow, separated by ';' / whitespace
            }
        }
    }
}

toolcall::Message parse_all(std::string_view text, const Options& opt) {
    ToolStreamParser p(opt);
    std::vector<Event> ev;
    p.push(text, ev);
    p.finish(ev);
    toolcall::Message m;
    toolcall::accumulate(ev, m);
    return m;
}

} // namespace vegajet::toolparse
