// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Anthropic Messages API (POST /v1/messages) on top of the OpenAI-shaped serving core (core/server).
//
//   parse_request()  Messages request -> serve::GenerateRequest: "system" (string or text blocks), "messages" with
//                    string content or blocks (text, tool_use, tool_result, thinking; redacted_thinking is dropped;
//                    image / document blocks are refused), "tools" (name, description, input_schema -> function
//                    tools), "tool_choice" (auto / any / tool / none, disable_parallel_tool_use), "max_tokens",
//                    "temperature" (0..1), "top_p", "top_k", "stop_sequences", "stream". tool_result blocks become
//                    tool turns (in order, before any text of the same user message), tool_use blocks become the
//                    assistant's tool calls, thinking blocks its reasoning.
//   response_json()  the non-streaming Message object: content blocks thinking / text / tool_use (input = the parsed
//                    arguments object), stop_reason end_turn / max_tokens / stop_sequence / tool_use, usage.
//   SseEncoder       the streaming event sequence: message_start, ping, then per block content_block_start,
//                    content_block_delta (thinking_delta / text_delta / input_json_delta) and content_block_stop,
//                    finally message_delta (stop_reason, usage) and message_stop. Consumes toolcall::Event, the
//                    output of either tool-call parser, so OpenAI and Anthropic share one generation path.
//   error_json()     {"type":"error","error":{"type":..,"message":..}}.
#pragma once

#include "core/server.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace vegajet::anthropic {

struct Limits {
    std::string default_model = "vegajet";
    int32_t default_max_tokens = 256;  // "max_tokens" is required by the API; this is used when it is missing
    int32_t max_tokens_cap = 0;        // 0: no cap
    size_t max_model_len = 128;        // longer "model" names answer 400 (S-M3)
};

bool parse_request(const json::Value& root, const Limits& lim, serve::GenerateRequest& g, std::string& err);

// stop_reason for a finish reason; stop_sequence_hit distinguishes a stop string from EOS.
const char* stop_reason(serve::FinishReason f, bool stop_sequence_hit);

struct Usage {
    uint32_t input_tokens = 0, output_tokens = 0;
};

// stop_sequence: the matched stop string, or "" for null.
std::string response_json(std::string_view id, std::string_view model, const toolcall::Message& m,
                          std::string_view stop_reason, std::string_view stop_sequence, const Usage& u);

// "invalid_request_error" (4xx), "not_found_error" (404), "overloaded_error" (503), "api_error" (other 5xx).
const char* error_type(int http_status);
std::string error_json(std::string_view type, std::string_view message);

class SseEncoder {
public:
    SseEncoder(std::string id, std::string model, uint32_t input_tokens = 0);
    std::string start();                                            // message_start + ping
    std::string events(const std::vector<toolcall::Event>& ev);     // content blocks
    std::string finish(std::string_view stop_reason, std::string_view stop_sequence, const Usage& u);
    std::string error(std::string_view type, std::string_view message);
    uint32_t blocks() const { return next_; }

private:
    std::string close_block();
    std::string open_block(int kind, const toolcall::ToolCall* call);
    std::string id_, model_;
    uint32_t input_tokens_;
    int open_ = -1;      // kind of the open block: 0 text, 1 thinking, -1 none
    uint32_t next_ = 0;  // index of the next block
};

// One SSE event in the Anthropic framing.
std::string sse(std::string_view event, std::string_view data);

} // namespace vegajet::anthropic
