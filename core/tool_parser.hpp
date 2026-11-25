// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Streaming tool-call parser for the three common open-weight tool formats, with malformed-JSON recovery.
//
//   Qwen / Hermes   <tool_call>{"name": .., "arguments": {..}}</tool_call>               (one or more)
//   DeepSeek        <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME
//                   ```json\n{..}\n```<｜tool▁call▁end｜> ... <｜tool▁calls▁end｜>           (V3 / R1)
//                   <｜tool▁call▁begin｜>NAME<｜tool▁sep｜>{..}<｜tool▁call▁end｜>             (V3.1 style)
//   Llama 3.x       <|python_tag|>{"name": .., "parameters": {..}}[; {..}] up to <|eom_id|> / <|eot_id|> / end,
//                   <function=NAME>{..}</function>, or a bare {"name": .., "parameters": ..} object that opens the
//                   reply (several may follow, separated by ';' or whitespace).
//
// push() takes generated text in arbitrary pieces: a marker split across pieces at any byte (including inside the
// multi-byte DeepSeek markers) is held back, never emitted as content. A call is emitted once its text is complete
// (closing marker, or end of generation) and its JSON parses, after repair_json() if needed: code fences, single
// quotes, Python True / False / None, unquoted keys, trailing commas, missing closing brackets / quotes, and trailing
// garbage after the object are fixed. The arguments come out as compact JSON. A call that cannot be recovered, or
// names a tool that was not offered, falls back to content byte-for-byte. An optional leading <think> block is
// routed to reasoning events. Events are toolcall::Event, so the OpenAI and Anthropic encoders consume either parser.
#pragma once

#include "core/tool_call_parser.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace vegajet::toolparse {

enum class Format : uint8_t { Qwen = 0, DeepSeek = 1, Llama = 2 };
const char* format_name(Format f);
// "qwen", "hermes", "chatml" -> Qwen; "deepseek*" -> DeepSeek; "llama*" -> Llama. False if unknown.
bool format_from_name(std::string_view name, Format& out);

// Repairs almost-JSON. Returns compact JSON text, or "" if the input is not recoverable. *changed: a fix was needed.
std::string repair_json(std::string_view text, bool* changed = nullptr);

struct Options {
    Format format = Format::Qwen;
    bool parse_reasoning = true;           // a leading <think> opens a reasoning block
    bool parallel = true;                  // false: done() after the first call
    bool repair = true;                    // try repair_json() on bodies that do not parse
    std::vector<std::string> known_tools;  // non-empty: calls to other names fall back to content
    std::string id_prefix = "call_";
};

class ToolStreamParser {
public:
    explicit ToolStreamParser(Options opt = {});
    void push(std::string_view piece, std::vector<toolcall::Event>& out);
    void finish(std::vector<toolcall::Event>& out);
    bool done() const { return done_; }
    uint32_t calls() const { return n_calls_; }
    uint32_t malformed() const { return n_malformed_; }
    uint32_t repaired() const { return n_repaired_; }

private:
    enum class State : uint8_t { Start, Reasoning, Content, Call, DsBlock, BareJson, Done };
    void run(std::vector<toolcall::Event>& out, bool final);
    void emit(toolcall::Event::Kind k, std::string_view s, std::vector<toolcall::Event>& out);
    // Parses one call body; emits the call (true) or nothing (false).
    bool emit_call(std::string_view body, std::string_view name_hint, std::vector<toolcall::Event>& out);
    bool finish_call(std::vector<toolcall::Event>& out, bool final);

    Options opt_;
    State st_ = State::Start;
    std::string buf_;
    std::string raw_;        // text of the open call, including its markers (fallback content)
    int call_kind_ = 0;      // which opening marker started the call
    bool trim_ws_ = false;
    bool done_ = false;
    uint32_t n_calls_ = 0, n_malformed_ = 0, n_repaired_ = 0;
};

// Convenience: the whole text at once.
toolcall::Message parse_all(std::string_view text, const Options& opt = {});

} // namespace vegajet::toolparse
