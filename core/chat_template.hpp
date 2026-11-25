// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Zero-dependency Jinja-subset chat template engine (the `tokenizer.chat_template` of a GGUF file), with the
// Hugging Face rendering environment: trim_blocks + lstrip_blocks, `{%-` / `-%}` / `{{-` / `-}}` whitespace control.
//
// Supported: {{ expr }}, {% if / elif / else %}, {% for x in xs [if cond] %} (tuple unpacking, `else`, loop.index,
// index0, revindex, revindex0, first, last, length, previtem, nextitem), {% set x = e %}, {% set ns.attr = e %},
// namespace(...), {% macro %} with defaults + calls, {# comments #}; literals (string, int, float, bool, none, list,
// dict, tuple), attribute / item access, slices ([::-1], [1:], [:-1]), `in` / `not in`, `is [not] test` (defined,
// undefined, none, string, number, integer, float, mapping, iterable, sequence, boolean, true, false, even, odd),
// arithmetic, `~`, `a if c else b`; filters: length, count, trim, upper, lower, capitalize, title, tojson (indent=),
// string, int, float, list, first, last, join, replace, default (d), items, keys, values, reverse, unique, sort,
// abs, round, safe, e / escape, indent, map(attribute=), selectattr / rejectattr (defined, undefined, none, equalto,
// eq, ==), sum, min, max, wordcount; string methods startswith, endswith, strip, lstrip, rstrip, split, splitlines,
// upper, lower, title, capitalize, replace, find, count, format-free `%`-less; dict methods get, items, keys,
// values; list append; globals range, namespace, raise_exception, strftime_now (deterministic: renders
// RenderOptions::date_string).
//
// Chat layer: messages with system / user / assistant / tool roles, assistant tool calls (arguments passed as parsed
// JSON objects, like Hugging Face), tool results, a `tools` schema list, add_generation_prompt, enable_thinking,
// bos_token / eos_token. Reasoning: split_reasoning() separates a <think>...</think> block from the answer, and
// ReasoningSplitter does the same incrementally for streamed text (tags may be cut across chunks).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vegajet::chat {

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments;  // JSON text; passed to the template as an object when it parses
};

struct ChatMessage {
    ChatMessage() = default;
    ChatMessage(std::string r, std::string c) : role(std::move(r)), content(std::move(c)) {}
    std::string role;               // system / user / assistant / tool
    std::string content;
    std::string reasoning_content;  // assistant: earlier <think> text ("" = absent)
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;       // tool
    std::string name;               // tool (function name) or participant name
};

struct RenderOptions {
    bool add_generation_prompt = true;
    int enable_thinking = -1;       // -1: leave undefined (template default), 0: false, 1: true
    std::string bos_token, eos_token;
    std::string tools_json;         // JSON array of tool schemas ("" = no tools)
    std::string date_string = "26 Jul 2024";  // strftime_now() / date_string variable (deterministic output)
};

struct Node;  // parsed template tree (chat_template.cpp)

class Template {
public:
    Template();
    ~Template();
    Template(Template&&) noexcept;
    Template& operator=(Template&&) noexcept;
    bool parse(std::string_view source, std::string* err = nullptr);
    bool render(const std::vector<ChatMessage>& messages, const RenderOptions& opt, std::string& out,
                std::string* err = nullptr) const;
    // Renders with arbitrary variables given as one JSON object (testing / non-chat use).
    bool render_json(std::string_view vars_json, std::string& out, std::string* err = nullptr) const;

private:
    std::shared_ptr<Node> root_;
};

// Built-in templates for when a file carries none (format of each family): chatml, qwen3, llama3, deepseek-v3,
// mistral-tekken, gemma. nullptr for unknown names.
const char* builtin_template(std::string_view family);
// Built-in family for an architecture / template string pair ("chatml" when unknown).
std::string detect_template_family(std::string_view gguf_template, std::string_view architecture);

struct Reasoning {
    std::string reasoning;  // inside <think>...</think>
    std::string content;    // the answer, leading newlines removed
    bool closed = true;     // false: <think> opened but never closed (generation cut off)
};
// Handles "<think>r</think>c", "r</think>c" (template already opened the block) and plain "c".
Reasoning split_reasoning(std::string_view text, std::string_view open = "<think>", std::string_view close = "</think>");

class ReasoningSplitter {
public:
    // started_inside: the prompt already ends inside <think> (e.g. a thinking template's generation prompt).
    explicit ReasoningSplitter(bool started_inside = false, std::string open = "<think>", std::string close = "</think>");
    // Feeds the next generated chunk; appends to the reasoning / content deltas.
    void feed(std::string_view chunk, std::string& reasoning_delta, std::string& content_delta);
    void finish(std::string& reasoning_delta, std::string& content_delta);  // flushes a held partial tag

private:
    void emit(std::string_view s, std::string& r, std::string& c);
    std::string open_, close_, hold_;
    bool inside_ = false, seen_close_ = false, strip_nl_ = false;
};

} // namespace vegajet::chat
