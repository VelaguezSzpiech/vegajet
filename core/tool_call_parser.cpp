// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/tool_call_parser.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>

namespace vegajet::toolcall {

namespace {

constexpr std::string_view kThinkOpen = "<think>";
constexpr std::string_view kThinkClose = "</think>";
constexpr std::string_view kCallOpen = "<tool_call>";
constexpr std::string_view kCallClose = "</tool_call>";
constexpr size_t kMaxBody = 1u << 20;  // a <tool_call> body larger than this is not a tool call

bool is_ws(char c) { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; }

std::string trim(std::string_view s) {
    size_t b = 0, e = s.size();
    while (b < e && is_ws(s[b])) ++b;
    while (e > b && is_ws(s[e - 1])) --e;
    return std::string(s.substr(b, e - b));
}

std::string strip_char(const std::string& s, char c, bool left, bool right) {
    size_t b = 0, e = s.size();
    if (left) while (b < e && s[b] == c) ++b;
    if (right) while (e > b && s[e - 1] == c) --e;
    return s.substr(b, e - b);
}

// Length of the longest suffix of `s` that is a proper prefix of `tag`.
size_t partial_suffix(std::string_view s, std::string_view tag) {
    for (size_t len = std::min(s.size(), tag.size() - 1); len > 0; --len)
        if (s.compare(s.size() - len, len, tag.substr(0, len)) == 0) return len;
    return 0;
}

size_t trailing_ws(std::string_view s, size_t end) {
    size_t n = 0;
    while (n < end && is_ws(s[end - 1 - n])) ++n;
    return n;
}

void dump_number(double x, std::string& o) {
    if (std::isfinite(x) && x == std::floor(x) && std::fabs(x) < 9007199254740992.0) {
        char b[32];
        std::snprintf(b, sizeof(b), "%lld", static_cast<long long>(x));
        o += b;
        return;
    }
    if (!std::isfinite(x)) { o += "null"; return; }
    char b[40];
    for (int p = 1; p <= 17; ++p) {
        std::snprintf(b, sizeof(b), "%.*g", p, x);
        if (std::strtod(b, nullptr) == x) break;
    }
    o += b;
}

void dump(const json::Value& v, bool pretty, std::string& o) {
    using T = json::Value::Type;
    switch (v.type) {
        case T::Null: o += "null"; return;
        case T::Bool: o += v.b ? "true" : "false"; return;
        case T::Number: dump_number(v.num, o); return;
        case T::String: o += json::quote(v.str); return;
        case T::Array:
            o += '[';
            for (size_t i = 0; i < v.arr.size(); ++i) {
                if (i) o += pretty ? ", " : ",";
                dump(v.arr[i], pretty, o);
            }
            o += ']';
            return;
        case T::Object:
            o += '{';
            for (size_t i = 0; i < v.obj.size(); ++i) {
                if (i) o += pretty ? ", " : ",";
                o += json::quote(v.obj[i].first);
                o += pretty ? ": " : ":";
                dump(v.obj[i].second, pretty, o);
            }
            o += '}';
            return;
    }
}

bool valid_name(const std::string& n) {
    if (n.empty() || n.size() > 64) return false;
    for (char c : n)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.'))
            return false;
    return true;
}

// "arguments" as given by a client or a model -> compact JSON object text; false if it is not an object.
bool normalize_arguments(const json::Value* a, std::string& out) {
    if (!a || a->is_null()) { out = "{}"; return true; }
    if (a->is_object()) { out = dump_json(*a, false); return true; }
    if (a->is_string()) {
        json::Value inner;
        if (!json::parse(a->str, inner) || !inner.is_object()) return false;
        out = dump_json(inner, false);
        return true;
    }
    return false;
}

const char* kQwenSystem = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

}  // namespace

std::string dump_json(const json::Value& v, bool pretty) {
    std::string o;
    dump(v, pretty, o);
    return o;
}

std::vector<std::string> ToolRequest::names() const {
    std::vector<std::string> n;
    for (const auto& t : tools) n.push_back(t.name);
    return n;
}

// ------------------------------------------------------------------------------------------------ request side
bool parse_tool_request(const json::Value& root, ToolRequest& out, std::string& err) {
    out = ToolRequest{};
    if (const auto* tools = root.get("tools"); tools && !tools->is_null()) {
        if (!tools->is_array()) { err = "'tools' must be an array"; return false; }
        if (tools->arr.size() > 128) { err = "at most 128 tools"; return false; }
        std::set<std::string> seen;
        for (const auto& t : tools->arr) {
            if (!t.is_object()) { err = "every tool must be an object"; return false; }
            if (const auto* ty = t.get("type"); ty && !(ty->is_string() && ty->str == "function")) {
                err = "only tools of type 'function' are supported";
                return false;
            }
            const auto* fn = t.get("function");
            const auto* name = fn && fn->is_object() ? fn->get("name") : nullptr;
            if (!name || !name->is_string() || !valid_name(name->str)) {
                err = "every tool needs function.name matching [A-Za-z0-9_.-]{1,64}";
                return false;
            }
            if (const auto* p = fn->get("parameters"); p && !p->is_null() && !p->is_object()) {
                err = "function.parameters must be a JSON Schema object";
                return false;
            }
            if (!seen.insert(name->str).second) { err = "duplicate tool name '" + name->str + "'"; return false; }
            out.tools.push_back({name->str, dump_json(t, true)});
        }
    }
    if (const auto* tc = root.get("tool_choice"); tc && !tc->is_null()) {
        if (tc->is_string()) {
            if (tc->str == "none") out.choice.mode = ChoiceMode::None;
            else if (tc->str == "auto") out.choice.mode = ChoiceMode::Auto;
            else if (tc->str == "required") out.choice.mode = ChoiceMode::Required;
            else { err = "'tool_choice' must be \"none\", \"auto\", \"required\" or a function object"; return false; }
        } else if (tc->is_object()) {
            const auto* fn = tc->get("function");
            const auto* name = fn && fn->is_object() ? fn->get("name") : nullptr;
            if (!name || !name->is_string()) { err = "'tool_choice' object needs function.name"; return false; }
            out.choice.mode = ChoiceMode::Named;
            out.choice.name = name->str;
            bool found = false;
            for (const auto& t : out.tools) found |= t.name == name->str;
            if (!found) { err = "tool_choice names '" + name->str + "', which is not in 'tools'"; return false; }
        } else {
            err = "'tool_choice' must be a string or an object";
            return false;
        }
        if (out.tools.empty() && (out.choice.mode == ChoiceMode::Required || out.choice.mode == ChoiceMode::Named)) {
            err = "'tool_choice' requires 'tools'";
            return false;
        }
    }
    if (const auto* p = root.get("parallel_tool_calls"); p && !p->is_null()) {
        if (!p->is_bool()) { err = "'parallel_tool_calls' must be a boolean"; return false; }
        out.parallel = p->b;
    }
    return true;
}

bool parse_message_tools(const json::Value& msg, Turn& t, std::string& err) {
    if (const auto* rc = msg.get("reasoning_content"); rc && rc->is_string()) t.reasoning_content = rc->str;
    if (const auto* calls = msg.get("tool_calls"); calls && !calls->is_null()) {
        if (t.role != "assistant") { err = "only assistant messages may carry 'tool_calls'"; return false; }
        if (!calls->is_array()) { err = "'tool_calls' must be an array"; return false; }
        for (const auto& c : calls->arr) {
            const auto* fn = c.is_object() ? c.get("function") : nullptr;
            const auto* name = fn && fn->is_object() ? fn->get("name") : nullptr;
            if (!name || !name->is_string() || name->str.empty()) { err = "every tool call needs function.name"; return false; }
            ToolCall tc;
            tc.name = name->str;
            if (const auto* id = c.get("id"); id && id->is_string()) tc.id = id->str;
            const auto* a = fn->get("arguments");
            if (a && a->is_string()) tc.arguments = a->str;  // passed through as sent (the Qwen3 template does too)
            else if (!normalize_arguments(a, tc.arguments)) { err = "function.arguments must be a string or an object"; return false; }
            t.tool_calls.push_back(std::move(tc));
        }
    }
    if (t.role == "tool") {
        if (const auto* id = msg.get("tool_call_id"); id && id->is_string()) t.tool_call_id = id->str;
        if (const auto* n = msg.get("name"); n && n->is_string()) t.name = n->str;
    }
    return true;
}

std::string generation_prefill(const ToolRequest& tools) {
    if (!tools.active()) return {};
    if (tools.choice.mode == ChoiceMode::Required) return "<tool_call>\n";
    if (tools.choice.mode == ChoiceMode::Named) return "<tool_call>\n{\"name\": \"" + tools.choice.name + "\", \"arguments\": ";
    return {};
}

std::string format_chat(bpe::ChatFormat f, const std::vector<Turn>& msgs, const ToolRequest& tools,
                        const bpe::ChatOptions& opt) {
    std::string out;
    auto turn = [&](const std::string& role, const std::string& content) {
        out += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
    };
    const bool qwen3 = f == bpe::ChatFormat::Qwen3 || f == bpe::ChatFormat::Ornith;
    const bool qwen25 = f == bpe::ChatFormat::Qwen25;
    const bool has_system = !msgs.empty() && msgs.front().role == "system";
    size_t first = 0;
    if (tools.active()) {
        out += "<|im_start|>system\n";
        if (has_system) { out += msgs.front().content + "\n\n"; first = 1; }
        else if (!opt.default_system.empty()) out += opt.default_system + "\n\n";
        else if (qwen25) out += std::string(kQwenSystem) + "\n\n";
        out += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
               "You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
        for (const auto& t : tools.tools) out += "\n" + t.json;
        out += "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
               "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": "
               "<args-json-object>}\n</tool_call><|im_end|>\n";
    } else if (!has_system) {
        if (!opt.default_system.empty()) turn("system", opt.default_system);
        else if (qwen25) turn("system", kQwenSystem);
    }
    size_t last_user = msgs.size();
    for (size_t i = 0; i < msgs.size(); ++i) if (msgs[i].role == "user") last_user = i;
    for (size_t i = first; i < msgs.size(); ++i) {
        const Turn& m = msgs[i];
        if (m.role == "tool") {
            if (i == first || msgs[i - 1].role != "tool") out += "<|im_start|>user";
            out += "\n<tool_response>\n" + m.content + "\n</tool_response>";
            if (i + 1 == msgs.size() || msgs[i + 1].role != "tool") out += "<|im_end|>\n";
            continue;
        }
        if (m.role != "assistant") { turn(m.role, m.content); continue; }
        std::string content = m.content;
        std::string nl_content;  // the template's `content` for the newline-before-<tool_call> rule
        const bool before_last_user = last_user == msgs.size() || i < last_user;
        if (qwen3 && before_last_user) {
            const size_t e = content.rfind("</think>");  // same rule as bpe::apply_chat_template
            if (e != std::string::npos) {
                size_t k = e + 8;
                while (k < content.size() && is_ws(content[k])) ++k;
                content = content.substr(k);
            }
        } else if (qwen3 && !m.reasoning_content.empty()) {
            nl_content = content;
            content = "<think>\n" + strip_char(m.reasoning_content, '\n', true, true) + "\n</think>\n\n" +
                      strip_char(content, '\n', true, false);
        }
        if (nl_content.empty() && m.reasoning_content.empty()) nl_content = content;
        else if (nl_content.empty() && !(qwen3 && !before_last_user)) nl_content = content;
        if (m.tool_calls.empty()) { turn(m.role, content); continue; }
        if (qwen25) {
            out += "<|im_start|>assistant";
            if (!content.empty()) out += "\n" + content;
        } else {
            out += "<|im_start|>assistant\n" + content;
        }
        for (size_t j = 0; j < m.tool_calls.size(); ++j) {
            const ToolCall& c = m.tool_calls[j];
            if (qwen25 || j > 0 || !nl_content.empty()) out += "\n";
            std::string args = c.arguments;
            if (qwen25) {  // the Qwen2.5 template applies tojson to the arguments object
                json::Value v;
                if (json::parse(args, v)) args = dump_json(v, true);
            }
            out += "<tool_call>\n{\"name\": \"" + c.name + "\", \"arguments\": " + args + "}\n</tool_call>";
        }
        out += "<|im_end|>\n";
    }
    if (opt.add_generation_prompt) {
        out += "<|im_start|>assistant\n";
        if (qwen3 && !opt.enable_thinking) out += "<think>\n\n</think>\n\n";
        out += generation_prefill(tools);
    }
    return out;
}

// ------------------------------------------------------------------------------------------------ streaming parser
StreamParser::StreamParser(ParserOptions opt) : opt_(std::move(opt)) {
    if (opt_.start_in_reasoning) { st_ = State::Reasoning; trim_ws_ = true; }
    if (!opt_.prefill.empty()) {
        std::vector<Event> sink;
        silent_ = true;
        buf_ = opt_.prefill;
        run(sink, false);
        silent_ = false;
        body_prefill_ = st_ == State::ToolCall ? body_.size() + buf_.size() : 0;
    }
}

void StreamParser::emit(Event::Kind k, std::string_view s, std::vector<Event>& out) {
    if (s.empty() || silent_) return;
    if (!out.empty() && out.back().kind == k && k != Event::Kind::ToolCall) { out.back().text.append(s.data(), s.size()); return; }
    Event e;
    e.kind = k;
    e.text.assign(s.data(), s.size());
    out.push_back(std::move(e));
}

void StreamParser::push(std::string_view piece, std::vector<Event>& out) {
    if (done_) return;
    buf_.append(piece.data(), piece.size());
    run(out, false);
}

void StreamParser::finish(std::vector<Event>& out) {
    if (done_) { buf_.clear(); return; }
    run(out, true);
    if (st_ == State::ToolCall) {  // no closing tag: accept a complete call, else fall back to the raw text
        body_ += buf_;
        buf_.clear();
        close_call(false, out);
        st_ = State::Content;
    }
    buf_.clear();
}

bool StreamParser::close_call(bool closed, std::vector<Event>& out) {
    const bool open_from_prefill = body_prefill_ > 0;
    const std::string model_part = body_.substr(std::min(body_prefill_, body_.size()));
    bool ok = false;
    ToolCall call;
    json::Value v;
    if (body_.size() <= kMaxBody && json::parse(trim(body_), v) && v.is_object()) {
        const auto* name = v.get("name");
        const json::Value* args = v.get("arguments");
        if (!args) args = v.get("parameters");
        if (name && name->is_string() && !name->str.empty() && normalize_arguments(args, call.arguments)) {
            ok = opt_.known_tools.empty() ||
                 std::find(opt_.known_tools.begin(), opt_.known_tools.end(), name->str) != opt_.known_tools.end();
            call.name = name->str;
        }
    }
    if (ok) {
        Event e;
        e.kind = Event::Kind::ToolCall;
        e.index = n_calls_;
        call.id = opt_.id_prefix + std::to_string(n_calls_);
        e.call = std::move(call);
        out.push_back(std::move(e));
        ++n_calls_;
        if (!opt_.parallel) done_ = true;
    } else {
        ++n_malformed_;
        std::string raw = ws_before_;  // the whitespace held back before the block belongs to the content too
        if (!open_from_prefill) raw += kCallOpen;
        raw += model_part;
        if (closed) raw += kCallClose;
        emit(Event::Kind::Content, raw, out);
    }
    body_.clear();
    body_prefill_ = 0;
    ws_before_.clear();
    return ok;
}

void StreamParser::run(std::vector<Event>& out, bool final) {
    for (;;) {
        switch (st_) {
            case State::Start: {
                if (opt_.parse_reasoning) {
                    const size_t n = std::min(buf_.size(), kThinkOpen.size());
                    if (buf_.compare(0, n, kThinkOpen.substr(0, n)) == 0) {  // buf_ is a prefix of "<think>" or starts with it
                        if (buf_.size() < kThinkOpen.size()) {
                            if (!final) return;
                        } else {
                            buf_.erase(0, kThinkOpen.size());
                            st_ = State::Reasoning;
                            trim_ws_ = true;
                            continue;
                        }
                    }
                }
                if (buf_.empty() && !final) return;
                st_ = State::Content;
                continue;
            }
            case State::Content: {
                if (trim_ws_) {
                    size_t k = 0;
                    while (k < buf_.size() && is_ws(buf_[k])) ++k;
                    buf_.erase(0, k);
                    if (buf_.empty()) return;
                    trim_ws_ = false;
                }
                if (!opt_.parse_tool_calls) {
                    emit(Event::Kind::Content, buf_, out);
                    buf_.clear();
                    return;
                }
                const size_t p = buf_.find(kCallOpen);
                if (p != std::string::npos) {
                    const size_t tw = trailing_ws(buf_, p);
                    emit(Event::Kind::Content, std::string_view(buf_).substr(0, p - tw), out);
                    ws_before_ = buf_.substr(p - tw, tw);
                    buf_.erase(0, p + kCallOpen.size());
                    body_.clear();
                    st_ = State::ToolCall;
                    continue;
                }
                size_t hold = 0;
                if (!final) {
                    hold = partial_suffix(buf_, kCallOpen);
                    hold += trailing_ws(buf_, buf_.size() - hold);  // dropped if a tool call follows
                }
                emit(Event::Kind::Content, std::string_view(buf_).substr(0, buf_.size() - hold), out);
                buf_.erase(0, buf_.size() - hold);
                return;
            }
            case State::Reasoning: {
                if (trim_ws_) {
                    size_t k = 0;
                    while (k < buf_.size() && is_ws(buf_[k])) ++k;
                    buf_.erase(0, k);
                    if (buf_.empty()) return;
                    trim_ws_ = false;
                }
                const size_t p = buf_.find(kThinkClose);
                if (p != std::string::npos) {
                    emit(Event::Kind::Reasoning, std::string_view(buf_).substr(0, p - trailing_ws(buf_, p)), out);
                    buf_.erase(0, p + kThinkClose.size());
                    st_ = State::Content;
                    trim_ws_ = true;
                    continue;
                }
                size_t hold = 0;
                if (!final) {
                    hold = partial_suffix(buf_, kThinkClose);
                    hold += trailing_ws(buf_, buf_.size() - hold);
                } else {
                    hold = trailing_ws(buf_, buf_.size());
                }
                emit(Event::Kind::Reasoning, std::string_view(buf_).substr(0, buf_.size() - hold), out);
                buf_.erase(0, buf_.size() - hold);
                if (final) buf_.clear();
                return;
            }
            case State::ToolCall: {
                const size_t p = buf_.find(kCallClose);
                if (p != std::string::npos) {
                    body_.append(buf_, 0, p);
                    buf_.erase(0, p + kCallClose.size());
                    trim_ws_ = close_call(true, out);  // whitespace after a fallback block is content
                    st_ = State::Content;
                    if (done_) { buf_.clear(); return; }
                    continue;
                }
                const size_t hold = final ? 0 : partial_suffix(buf_, kCallClose);
                body_.append(buf_, 0, buf_.size() - hold);
                buf_.erase(0, buf_.size() - hold);
                if (body_.size() > kMaxBody) {  // runaway block: not a tool call
                    close_call(false, out);
                    st_ = State::Content;
                    continue;
                }
                return;
            }
        }
    }
}

void accumulate(const std::vector<Event>& ev, Message& m) {
    for (const auto& e : ev) {
        if (e.kind == Event::Kind::Content) m.content += e.text;
        else if (e.kind == Event::Kind::Reasoning) m.reasoning += e.text;
        else m.tool_calls.push_back(e.call);
    }
}

}  // namespace vegajet::toolcall
