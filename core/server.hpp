// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Zero-dependency OpenAI-compatible HTTP/1.1 server (Linux epoll), decoupled from any model through GenerateFn.
//
//   GET  /health                 {"status":"ok", ...}
//   GET  /v1/models              one model entry (ServerOptions::model_id)
//   POST /v1/chat/completions    JSON, or Server-Sent Events when "stream": true
//   POST /v1/messages            Anthropic Messages API (core/anthropic_shim): same queue, generator and output
//                                parser; Anthropic request / response / SSE event shapes and error bodies
//
// Threads: one epoll I/O thread (accept, parse, small responses, keep-alive) and one generation worker. A GPU engine
// runs one sequence at a time, so requests queue FIFO (up to max_queue; beyond that 503 + Retry-After).
// The worker owns a connection while it generates; it detects a client that went away from a send failure and
// returns false from the TokenSink so the engine stops early. A peer that only half-closed (SHUT_WR) is a normal
// client: it still receives the full response. Stop sequences, max_tokens and UTF-8 boundaries are
// enforced here, so every GenerateFn gets them for free.
// Function calling and reasoning (core/tool_call_parser): requests may carry "tools" / "tool_choice" /
// "parallel_tool_calls" and tool-call / tool-result messages; output passes through a StreamParser, so a leading
// <think>...</think> becomes "reasoning_content" and <tool_call> blocks become OpenAI "tool_calls" (streamed as
// tool_calls deltas, finish_reason "tool_calls"). "reasoning_format": "none" in a request (or
// ServerOptions::parse_reasoning = false) keeps the raw text. ServerOptions::workers > 1 runs that many generations
// concurrently (for a GenerateFn that batches, e.g. core/request_scheduler); the default 1 keeps the FIFO behaviour.
// GET /metrics serves ServerOptions::metrics_text() (Prometheus text, core/metrics) when it is set, 404 otherwise.
#pragma once

#include "core/sampler.hpp"
#include "core/tool_call_parser.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace vegajet::serve {

// role, content, plus reasoning_content / tool_calls (assistant) and tool_call_id / name (tool).
using ChatTurn = toolcall::Turn;

enum class FinishReason : uint8_t { Stop = 0, Length = 1, Cancelled = 2, Error = 3, ToolCalls = 4 };
const char* finish_reason_name(FinishReason f);

struct GenerateRequest {
    std::string model;
    std::vector<ChatTurn> messages;
    int32_t max_tokens = 256;
    sampling::SamplerParams sampling;
    std::vector<std::string> stop;  // informational: the server already enforces these
    bool stream = false;
    toolcall::ToolRequest tools;    // render with toolcall::format_chat(); it appends generation_prefill(tools)
    bool parse_reasoning = true;    // false: "reasoning_format": "none"
};

struct GenerateResult {
    FinishReason finish = FinishReason::Stop;  // Stop (EOS) or Length; the server overrides on stop sequences
    uint32_t prompt_tokens = 0;
    uint32_t completion_tokens = 0;  // 0: the server counts TokenSink calls
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    std::string error;               // non-empty: request failed (HTTP 500 unless streaming already started)
};

// What a TokenSink returns. `stop` ends generation; `client_gone` says it stopped because the client went away, so
// the request is recorded as cancelled (status 499) instead of as a completed 200 (S-M2). A plain bool is still
// accepted (true = continue), so existing sinks keep working.
struct SinkResult {
    bool stop = false;
    bool client_gone = false;
    SinkResult() = default;
    SinkResult(bool continue_generation) : stop(!continue_generation) {}  // NOLINT: implicit on purpose
    static SinkResult Stop() { SinkResult r; r.stop = true; return r; }
    static SinkResult ClientGone() { SinkResult r; r.stop = true; r.client_gone = true; return r; }
    explicit operator bool() const { return !stop; }
};

// Receives one decoded token piece (any bytes; may split UTF-8 sequences). A falsy result means generation must stop
// (client gone, stop sequence hit, max_tokens reached, or server shutting down).
using TokenSink = std::function<SinkResult(std::string_view piece)>;
using GenerateFn = std::function<GenerateResult(const GenerateRequest&, const TokenSink&)>;

struct ServerOptions {
    std::string host = "127.0.0.1";
    uint16_t port = 8080;               // 0: ephemeral (read it back with port())
    std::string model_id = "vegajet";
    // A request's "model" must name this server (S-M3): longer names answer 400, others 404 model_not_found, and an
    // absent field means model_id. The name is also no longer repeated in every SSE chunk.
    size_t max_model_name_len = 128;
    uint32_t context_length = 0;        // reported in /v1/models when non-zero
    int32_t default_max_tokens = 256;
    int32_t max_tokens_cap = 0;         // 0: no cap
    size_t max_header_bytes = 16 * 1024;
    size_t max_body_bytes = 8u << 20;
    // /v1/chat/completions and /v1/messages bodies above this answer 413: the JSON Value tree costs ~50x the body, so
    // the parse is bounded by bytes and by node count before any allocation happens (S-B1, S-M4).
    size_t max_chat_body_bytes = 1u << 20;
    size_t max_json_nodes = 65536;      // JSON values accepted per request body; beyond this 413
    size_t max_queue = 16;              // queued + running generations
    size_t max_connections = 256;
    // Sockets kept open above max_connections for /health and /metrics only (so monitoring stays reachable when the
    // connection pool is full), plus a few that only receive the 503 for their request instead of a bare RST.
    size_t reserve_connections = 2;
    int idle_timeout_ms = 30000;        // idle keep-alive / slow-header connections are closed after this
    int send_timeout_ms = 10000;        // a client that stops reading for this long is dropped
    bool parse_reasoning = true;        // split a leading <think>...</think> into reasoning_content
    // The generator decodes greedily (no temperature/top-k/top-p/min-p/repeat-penalty): reject such parameters with
    // HTTP 400 unless they hold the greedy defaults (temperature 0, top_p 1, top_k 0, min_p 0, repeat_penalty 1).
    bool greedy_only = false;
    size_t workers = 1;                 // concurrent generations (> 1 needs a GenerateFn that is safe to call concurrently)
    std::function<std::string()> metrics_text;  // set: GET /metrics serves it (Prometheus text, core/metrics)
};

// Stop-sequence matcher with UTF-8 holdback (exposed for tests and for non-HTTP front ends).
class StopMatcher {
public:
    explicit StopMatcher(std::vector<std::string> stops = {});
    // Appends a piece; returns text that is safe to emit now (never part of a stop string, never a partial UTF-8
    // sequence). After a stop string is seen, stopped() is true and the stop string and anything after it are dropped.
    std::string push(std::string_view piece);
    // End of generation: returns held-back text (a dangling partial UTF-8 sequence becomes U+FFFD).
    std::string flush();
    bool stopped() const { return stopped_; }
    const std::string& matched() const { return matched_; }  // the stop string that ended the text ("" if none)

private:
    std::vector<std::string> stops_;
    std::string pending_;
    std::string matched_;
    bool stopped_ = false;
};

class OpenAiServer {
public:
    OpenAiServer(ServerOptions opt, GenerateFn gen);
    ~OpenAiServer();
    OpenAiServer(const OpenAiServer&) = delete;
    OpenAiServer& operator=(const OpenAiServer&) = delete;

    bool start(std::string* err = nullptr);  // bind + listen + spawn threads; non-blocking
    uint16_t port() const;
    void stop();   // idempotent: cancels the running generation, closes connections, joins threads
    void wait();   // blocks until stop() is called (from another thread or a signal handler via request_stop())
    void request_stop();  // async-signal-safe: only writes to an eventfd

    struct Stats {
        uint64_t requests = 0, completions = 0, streamed = 0, cancelled = 0, rejected = 0, errors = 0;
    };
    Stats stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vegajet::serve
