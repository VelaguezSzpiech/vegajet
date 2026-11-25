// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// OpenAI function calling for the Qwen / Ornith (Hermes-style) tool format, CPU-only and model-agnostic.
//
// Request side
//   parse_tool_request()   "tools", "tool_choice" ("none" | "auto" | "required" | {"type":"function","function":
//                          {"name":..}}) and "parallel_tool_calls" from a chat request.
//   parse_message_tools()  per-message "tool_calls" (assistant), "tool_call_id" / "name" (tool) and
//                          "reasoning_content" (assistant).
//   format_chat()          the Qwen2.5 / Qwen3 / Ornith / ChatML prompt with the "# Tools" system block (tool
//                          signatures serialized like Jinja's tojson: ", " / ": " separators, key order kept, UTF-8
//                          kept), assistant <tool_call> turns and consecutive tool results grouped into one user turn
//                          of <tool_response> blocks. With no tools and no tool turns the output is byte-identical to
//                          bpe::apply_chat_template(). tool_choice "none" leaves the tools out of the prompt;
//                          "required" / a named function prime the generation prompt with generation_prefill().
// Response side
//   StreamParser           incremental: <think>...</think> -> reasoning deltas, <tool_call>{json}</tool_call> ->
//                          tool-call events, everything else -> content deltas. Tags may be split across pieces at any
//                          byte (a possible tag prefix is held back, never emitted early). A tool call is buffered
//                          until its closing tag and validated (JSON object with a string "name" from the offered tools
//                          and an object / JSON-string "arguments") before anything about it is emitted, so a client
//                          never receives arguments that do not parse; a call that fails validation falls back to
//                          content, byte-for-byte as the model wrote it. Buffering costs no end-to-end latency: a tool
//                          call cannot be executed before its arguments are complete.
#pragma once

#include "core/bpe_tokenizer.hpp"
#include "core/json_lite.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace vegajet::toolcall {

struct ToolSpec {
    std::string name;
    std::string json;  // the request's tool object, re-serialized tojson-style for the prompt
};

enum class ChoiceMode : uint8_t { None = 0, Auto = 1, Required = 2, Named = 3 };

struct ToolChoice {
    ChoiceMode mode = ChoiceMode::Auto;
    std::string name;  // Named
};

struct ToolRequest {
    std::vector<ToolSpec> tools;
    ToolChoice choice;
    bool parallel = true;  // parallel_tool_calls
    // Tools are rendered and parsed only when offered and not switched off by tool_choice "none".
    bool active() const { return !tools.empty() && choice.mode != ChoiceMode::None; }
    std::vector<std::string> names() const;
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments;  // JSON text
};

// A chat message with the tool-related fields. serve::ChatTurn is this type; the default member initializers keep
// existing two-field brace initialization ({"user", "Hi"}) free of -Wmissing-field-initializers.
struct Turn {
    std::string role;
    std::string content;
    std::string reasoning_content{};     // assistant
    std::vector<ToolCall> tool_calls{};  // assistant
    std::string tool_call_id{};          // tool
    std::string name{};                  // tool
};

bool parse_tool_request(const json::Value& root, ToolRequest& out, std::string& err);
bool parse_message_tools(const json::Value& msg, Turn& t, std::string& err);

// JSON serializer. pretty = Python json.dumps defaults (", " / ": "), else compact (OpenAI "arguments" style).
// Numbers: integral values below 2^53 print as integers, others as the shortest round-trip %g form.
std::string dump_json(const json::Value& v, bool pretty);

std::string format_chat(bpe::ChatFormat f, const std::vector<Turn>& msgs, const ToolRequest& tools,
                        const bpe::ChatOptions& opt = {});
// Text appended after the generation prompt for tool_choice "required" / named ("" otherwise). A GenerateFn that
// renders its own prompt must append the same text; StreamParser is primed with it (ParserOptions::prefill).
std::string generation_prefill(const ToolRequest& tools);

// -------------------------------------------------------------------------------------------- streaming parser
struct ParserOptions {
    bool parse_reasoning = true;      // a leading <think> opens a reasoning block
    bool start_in_reasoning = false;  // the prompt already opened <think> (output starts inside the block)
    bool parse_tool_calls = false;    // only when tools are offered (ToolRequest::active())
    bool parallel = true;             // false: done() after the first tool call
    std::vector<std::string> known_tools;  // non-empty: calls to other names fall back to content
    std::string id_prefix = "call_";       // tool-call ids are id_prefix + index
    std::string prefill;                   // generation_prefill(): consumed silently before the first piece
};

struct Event {
    enum class Kind : uint8_t { Content = 0, Reasoning = 1, ToolCall = 2 };
    Kind kind = Kind::Content;
    std::string text;  // Content / Reasoning
    uint32_t index = 0;  // ToolCall: OpenAI tool_calls[].index
    ToolCall call;       // ToolCall
};

class StreamParser {
public:
    explicit StreamParser(ParserOptions opt = {});
    void push(std::string_view piece, std::vector<Event>& out);
    void finish(std::vector<Event>& out);  // end of generation: flushes held text, resolves an open block
    bool done() const { return done_; }    // parallel == false and one call completed: stop generating
    uint32_t tool_calls() const { return n_calls_; }
    uint32_t malformed() const { return n_malformed_; }

private:
    enum class State : uint8_t { Start, Content, Reasoning, ToolCall };
    void run(std::vector<Event>& out, bool final);
    void emit(Event::Kind k, std::string_view s, std::vector<Event>& out);
    bool close_call(bool closed, std::vector<Event>& out);  // true: valid call emitted

    ParserOptions opt_;
    State st_ = State::Start;
    std::string buf_;        // unclassified text (held back: possible tag prefix / trailing whitespace)
    std::string body_;       // current <tool_call> body
    size_t body_prefill_ = 0;  // leading bytes of body_ that came from the prefill (not model output)
    std::string ws_before_;    // whitespace dropped before the open <tool_call> (restored if it falls back)
    bool trim_ws_ = false;   // drop whitespace right after </think> / </tool_call> / <think>
    bool silent_ = false;    // consuming the prefill
    bool done_ = false;
    uint32_t n_calls_ = 0, n_malformed_ = 0;
};

// Collapses an event list into the non-streaming message parts.
struct Message {
    std::string content, reasoning;
    std::vector<ToolCall> tool_calls;
};
void accumulate(const std::vector<Event>& ev, Message& m);

}  // namespace vegajet::toolcall
