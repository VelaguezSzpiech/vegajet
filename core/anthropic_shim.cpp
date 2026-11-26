// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/anthropic_shim.hpp"

#include <algorithm>
#include <map>

namespace vegajet::anthropic {
namespace {

using json::Value;

Value jstr(std::string s) {
    Value v;
    v.type = Value::Type::String;
    v.str = std::move(s);
    return v;
}
Value jobj() {
    Value v;
    v.type = Value::Type::Object;
    return v;
}
Value jarr() {
    Value v;
    v.type = Value::Type::Array;
    return v;
}
Value jbool(bool b) {
    Value v;
    v.type = Value::Type::Bool;
    v.b = b;
    return v;
}

bool number(const Value& root, const char* key, double& out, std::string& err) {
    const Value* v = root.get(key);
    if (!v || v->is_null()) return true;
    if (!v->is_number()) {
        err = std::string("'") + key + "' must be a number";
        return false;
    }
    out = v->num;
    return true;
}

// Text of a string or an array of text blocks.
bool text_of(const Value& c, std::string& out, std::string& err, const char* what) {
    if (c.is_string()) {
        out += c.str;
        return true;
    }
    if (c.is_array()) {
        for (const Value& b : c.arr) {
            const Value* ty = b.get("type");
            const Value* tx = b.get("text");
            if (ty && ty->is_string() && ty->str == "text" && tx && tx->is_string()) {
                out += tx->str;
            } else {
                err = std::string(what) + ": only text blocks are supported";
                return false;
            }
        }
        return true;
    }
    if (c.is_null()) return true;
    err = std::string(what) + " must be a string or an array of text blocks";
    return false;
}

std::vector<std::string> utf8_pieces(const std::string& s, size_t max) {
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
        size_t n = std::min(max, s.size() - i);
        while (n > 0 && i + n < s.size() && (static_cast<unsigned char>(s[i + n]) & 0xC0) == 0x80) --n;
        if (n == 0) n = std::min(max, s.size() - i);
        out.push_back(s.substr(i, n));
        i += n;
    }
    return out;
}

} // namespace

std::string sse(std::string_view event, std::string_view data) {
    std::string o = "event: ";
    o.append(event.data(), event.size());
    o += "\ndata: ";
    o.append(data.data(), data.size());
    o += "\n\n";
    return o;
}

bool parse_request(const Value& root, const Limits& lim, serve::GenerateRequest& g, std::string& err) {
    if (!root.is_object()) {
        err = "request body must be a JSON object";
        return false;
    }
    if (const Value* m = root.get("model"); m && m->is_string()) {
        if (m->str.size() > lim.max_model_len) {
            err = "'model' must be at most " + std::to_string(lim.max_model_len) + " bytes";
            return false;
        }
        g.model = m->str;
    }
    if (g.model.empty()) g.model = lim.default_model;

    // system
    if (const Value* sys = root.get("system"); sys && !sys->is_null()) {
        std::string s;
        if (!text_of(*sys, s, err, "'system'")) return false;
        if (!s.empty()) g.messages.push_back(serve::ChatTurn{"system", s});
    }
    // messages
    const Value* msgs = root.get("messages");
    if (!msgs || !msgs->is_array() || msgs->arr.empty()) {
        err = "'messages' must be a non-empty array";
        return false;
    }
    std::map<std::string, std::string> tool_names;  // tool_use id -> name
    for (const Value& m : msgs->arr) {
        const Value* role = m.get("role");
        if (!role || !role->is_string() || (role->str != "user" && role->str != "assistant")) {
            err = "every message needs 'role' \"user\" or \"assistant\"";
            return false;
        }
        const bool assistant = role->str == "assistant";
        serve::ChatTurn t;
        t.role = role->str;
        const Value* c = m.get("content");
        if (!c || c->is_null()) {
            err = "every message needs 'content'";
            return false;
        }
        if (c->is_string()) {
            t.content = c->str;
            g.messages.push_back(std::move(t));
            continue;
        }
        if (!c->is_array()) {
            err = "message 'content' must be a string or an array of content blocks";
            return false;
        }
        for (const Value& b : c->arr) {
            const Value* ty = b.get("type");
            if (!ty || !ty->is_string()) {
                err = "every content block needs a 'type'";
                return false;
            }
            const std::string& type = ty->str;
            if (type == "text") {
                const Value* tx = b.get("text");
                if (!tx || !tx->is_string()) {
                    err = "text block needs 'text'";
                    return false;
                }
                t.content += tx->str;
            } else if (type == "tool_use" && assistant) {
                const Value* id = b.get("id");
                const Value* name = b.get("name");
                const Value* input = b.get("input");
                if (!id || !id->is_string() || !name || !name->is_string()) {
                    err = "tool_use block needs 'id' and 'name'";
                    return false;
                }
                if (input && !input->is_null() && !input->is_object()) {
                    err = "tool_use 'input' must be an object";
                    return false;
                }
                toolcall::ToolCall call;
                call.id = id->str;
                call.name = name->str;
                call.arguments = input && input->is_object() ? toolcall::dump_json(*input, false) : "{}";
                tool_names[id->str] = name->str;
                t.tool_calls.push_back(std::move(call));
            } else if (type == "tool_result" && !assistant) {
                const Value* id = b.get("tool_use_id");
                if (!id || !id->is_string()) {
                    err = "tool_result block needs 'tool_use_id'";
                    return false;
                }
                serve::ChatTurn tr;
                tr.role = "tool";
                tr.tool_call_id = id->str;
                if (auto it = tool_names.find(id->str); it != tool_names.end()) tr.name = it->second;
                if (const Value* rc = b.get("content"))
                    if (!text_of(*rc, tr.content, err, "tool_result 'content'")) return false;
                if (const Value* ie = b.get("is_error"); ie && ie->is_bool() && ie->b) tr.content = "Error: " + tr.content;
                g.messages.push_back(std::move(tr));
            } else if (type == "thinking" && assistant) {
                if (const Value* th = b.get("thinking"); th && th->is_string()) t.reasoning_content += th->str;
            } else if (type == "redacted_thinking" && assistant) {
                // opaque: nothing to render
            } else {
                err = "unsupported content block '" + type + "' for role " + role->str +
                      " (supported: text, tool_use, tool_result, thinking)";
                return false;
            }
        }
        if (!t.content.empty() || !t.tool_calls.empty() || !t.reasoning_content.empty() || assistant)
            g.messages.push_back(std::move(t));
    }

    // tools -> OpenAI function tools, then the shared validator
    Value oa = jobj();
    if (const Value* tools = root.get("tools"); tools && !tools->is_null()) {
        if (!tools->is_array()) {
            err = "'tools' must be an array";
            return false;
        }
        Value arr = jarr();
        for (const Value& t : tools->arr) {
            if (!t.is_object()) {
                err = "every tool must be an object";
                return false;
            }
            if (const Value* ty = t.get("type"); ty && !(ty->is_string() && ty->str == "custom")) {
                err = "only client tools (name, description, input_schema) are supported";
                return false;
            }
            const Value* name = t.get("name");
            if (!name || !name->is_string()) {
                err = "every tool needs a 'name'";
                return false;
            }
            Value fn = jobj();
            fn.obj.emplace_back("name", *name);
            if (const Value* d = t.get("description"); d && d->is_string()) fn.obj.emplace_back("description", *d);
            const Value* schema = t.get("input_schema");
            if (schema && !schema->is_object()) {
                err = "tool 'input_schema' must be a JSON Schema object";
                return false;
            }
            if (schema) {
                fn.obj.emplace_back("parameters", *schema);
            } else {
                Value s = jobj();
                s.obj.emplace_back("type", jstr("object"));
                fn.obj.emplace_back("parameters", std::move(s));
            }
            Value tool = jobj();
            tool.obj.emplace_back("type", jstr("function"));
            tool.obj.emplace_back("function", std::move(fn));
            arr.arr.push_back(std::move(tool));
        }
        oa.obj.emplace_back("tools", std::move(arr));
    }
    if (const Value* tc = root.get("tool_choice"); tc && !tc->is_null()) {
        const Value* ty = tc->is_object() ? tc->get("type") : nullptr;
        if (!ty || !ty->is_string()) {
            err = "'tool_choice' must be an object with a 'type'";
            return false;
        }
        if (ty->str == "auto") {
            oa.obj.emplace_back("tool_choice", jstr("auto"));
        } else if (ty->str == "any") {
            oa.obj.emplace_back("tool_choice", jstr("required"));
        } else if (ty->str == "none") {
            oa.obj.emplace_back("tool_choice", jstr("none"));
        } else if (ty->str == "tool") {
            const Value* n = tc->get("name");
            if (!n || !n->is_string()) {
                err = "tool_choice {\"type\":\"tool\"} needs 'name'";
                return false;
            }
            Value fn = jobj();
            fn.obj.emplace_back("name", *n);
            Value choice = jobj();
            choice.obj.emplace_back("type", jstr("function"));
            choice.obj.emplace_back("function", std::move(fn));
            oa.obj.emplace_back("tool_choice", std::move(choice));
        } else {
            err = "'tool_choice' type must be auto, any, tool or none";
            return false;
        }
        if (const Value* d = tc->get("disable_parallel_tool_use"); d && d->is_bool())
            oa.obj.emplace_back("parallel_tool_calls", jbool(!d->b));
    }
    if (!toolcall::parse_tool_request(oa, g.tools, err)) return false;

    double max_tokens = lim.default_max_tokens;
    if (!number(root, "max_tokens", max_tokens, err)) return false;
    if (max_tokens < 1 || max_tokens > 1e9) {
        err = "'max_tokens' must be >= 1";
        return false;
    }
    g.max_tokens = static_cast<int32_t>(max_tokens);
    if (lim.max_tokens_cap > 0) g.max_tokens = std::min(g.max_tokens, lim.max_tokens_cap);

    double temperature = 1.0, top_p = 1.0, top_k = 0;
    if (!number(root, "temperature", temperature, err) || !number(root, "top_p", top_p, err) ||
        !number(root, "top_k", top_k, err))
        return false;
    if (temperature < 0 || temperature > 1) {
        err = "'temperature' must be in [0, 1]";
        return false;
    }
    if (!(top_p > 0 && top_p <= 1)) {
        err = "'top_p' must be in (0, 1]";
        return false;
    }
    if (top_k < 0) {
        err = "'top_k' must be >= 0";
        return false;
    }
    g.sampling.temperature = static_cast<float>(temperature);
    g.sampling.top_p = static_cast<float>(top_p);
    g.sampling.top_k = static_cast<int32_t>(top_k);

    if (const Value* st = root.get("stop_sequences"); st && !st->is_null()) {
        if (!st->is_array()) {
            err = "'stop_sequences' must be an array of strings";
            return false;
        }
        for (const Value& s : st->arr) {
            if (!s.is_string() || s.str.empty()) {
                err = "'stop_sequences' entries must be non-empty strings";
                return false;
            }
            g.stop.push_back(s.str);
        }
        if (g.stop.size() > 16) {
            err = "at most 16 stop sequences";
            return false;
        }
    }
    if (const Value* s = root.get("stream"); s && !s->is_null()) {
        if (!s->is_bool()) {
            err = "'stream' must be a boolean";
            return false;
        }
        g.stream = s->b;
    }
    return true;
}

const char* stop_reason(serve::FinishReason f, bool stop_sequence_hit) {
    if (f == serve::FinishReason::ToolCalls) return "tool_use";
    if (stop_sequence_hit) return "stop_sequence";
    switch (f) {
        case serve::FinishReason::Length: return "max_tokens";
        case serve::FinishReason::ToolCalls: return "tool_use";
        default: return "end_turn";
    }
}

const char* error_type(int status) {
    if (status == 404) return "not_found_error";
    if (status == 503) return "overloaded_error";
    if (status >= 500) return "api_error";
    return "invalid_request_error";
}

std::string error_json(std::string_view type, std::string_view message) {
    return "{\"type\":\"error\",\"error\":{\"type\":" + json::quote(type) + ",\"message\":" + json::quote(message) + "}}";
}

static std::string input_of(const std::string& args) {
    Value v;
    if (json::parse(args, v) && v.is_object()) return toolcall::dump_json(v, false);
    return "{}";
}

static std::string stop_seq_json(std::string_view s) { return s.empty() ? std::string("null") : json::quote(s); }

std::string response_json(std::string_view id, std::string_view model, const toolcall::Message& m,
                          std::string_view reason, std::string_view stop_sequence, const Usage& u) {
    std::string content;
    auto add = [&](const std::string& block) {
        if (!content.empty()) content += ',';
        content += block;
    };
    if (!m.reasoning.empty())
        add("{\"type\":\"thinking\",\"thinking\":" + json::quote(m.reasoning) + ",\"signature\":\"\"}");
    if (!m.content.empty()) add("{\"type\":\"text\",\"text\":" + json::quote(m.content) + "}");
    for (const auto& c : m.tool_calls)
        add("{\"type\":\"tool_use\",\"id\":" + json::quote(c.id) + ",\"name\":" + json::quote(c.name) +
            ",\"input\":" + input_of(c.arguments) + "}");
    return "{\"id\":" + json::quote(id) + ",\"type\":\"message\",\"role\":\"assistant\",\"model\":" + json::quote(model) +
           ",\"content\":[" + content + "],\"stop_reason\":" + json::quote(reason) +
           ",\"stop_sequence\":" + stop_seq_json(stop_sequence) + ",\"usage\":{\"input_tokens\":" +
           std::to_string(u.input_tokens) + ",\"output_tokens\":" + std::to_string(u.output_tokens) + "}}";
}

// ------------------------------------------------------------------------------------------------- SSE

SseEncoder::SseEncoder(std::string id, std::string model, uint32_t input_tokens)
    : id_(std::move(id)), model_(std::move(model)), input_tokens_(input_tokens) {}

std::string SseEncoder::start() {
    const std::string msg = "{\"id\":" + json::quote(id_) +
                            ",\"type\":\"message\",\"role\":\"assistant\",\"model\":" + json::quote(model_) +
                            ",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":" +
                            std::to_string(input_tokens_) + ",\"output_tokens\":0}}";
    return sse("message_start", "{\"type\":\"message_start\",\"message\":" + msg + "}") +
           sse("ping", "{\"type\":\"ping\"}");
}

std::string SseEncoder::close_block() {
    if (open_ < 0) return {};
    open_ = -1;
    return sse("content_block_stop",
               "{\"type\":\"content_block_stop\",\"index\":" + std::to_string(next_ - 1) + "}");
}

std::string SseEncoder::open_block(int kind, const toolcall::ToolCall* call) {
    std::string o = close_block();
    std::string block;
    if (kind == 0) block = "{\"type\":\"text\",\"text\":\"\"}";
    else if (kind == 1) block = "{\"type\":\"thinking\",\"thinking\":\"\"}";
    else block = "{\"type\":\"tool_use\",\"id\":" + json::quote(call->id) + ",\"name\":" + json::quote(call->name) + ",\"input\":{}}";
    o += sse("content_block_start", "{\"type\":\"content_block_start\",\"index\":" + std::to_string(next_) +
                                        ",\"content_block\":" + block + "}");
    open_ = kind;
    ++next_;
    return o;
}

std::string SseEncoder::events(const std::vector<toolcall::Event>& ev) {
    std::string o;
    for (const auto& e : ev) {
        if (e.kind == toolcall::Event::Kind::Content) {
            if (e.text.empty()) continue;
            if (open_ != 0) o += open_block(0, nullptr);
            o += sse("content_block_delta", "{\"type\":\"content_block_delta\",\"index\":" + std::to_string(next_ - 1) +
                                                ",\"delta\":{\"type\":\"text_delta\",\"text\":" + json::quote(e.text) + "}}");
        } else if (e.kind == toolcall::Event::Kind::Reasoning) {
            if (e.text.empty()) continue;
            if (open_ != 1) o += open_block(1, nullptr);
            o += sse("content_block_delta", "{\"type\":\"content_block_delta\",\"index\":" + std::to_string(next_ - 1) +
                                                ",\"delta\":{\"type\":\"thinking_delta\",\"thinking\":" +
                                                json::quote(e.text) + "}}");
        } else {
            o += open_block(2, &e.call);
            const std::string input = input_of(e.call.arguments);
            for (const auto& piece : utf8_pieces(input, 256))
                o += sse("content_block_delta", "{\"type\":\"content_block_delta\",\"index\":" + std::to_string(next_ - 1) +
                                                    ",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":" +
                                                    json::quote(piece) + "}}");
            o += close_block();
        }
    }
    return o;
}

std::string SseEncoder::finish(std::string_view reason, std::string_view stop_sequence, const Usage& u) {
    std::string o = close_block();
    o += sse("message_delta", "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":" + json::quote(reason) +
                                  ",\"stop_sequence\":" + stop_seq_json(stop_sequence) +
                                  "},\"usage\":{\"input_tokens\":" + std::to_string(u.input_tokens) +
                                  ",\"output_tokens\":" + std::to_string(u.output_tokens) + "}}");
    o += sse("message_stop", "{\"type\":\"message_stop\"}");
    return o;
}

std::string SseEncoder::error(std::string_view type, std::string_view message) {
    return sse("error", error_json(type, message));
}

} // namespace vegajet::anthropic
