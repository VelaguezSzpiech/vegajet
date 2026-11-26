// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Binds an IEngine (core/engine_interface.hpp: the GPU engine or the CPU MockEngine) to the OpenAI server's
// GenerateFn. Header-only so `vegajet serve` can use it without new link dependencies.
//   make_engine_generate_fn(engine)                       prompt = engine.format_chat() + engine.tokenizer()
//   make_engine_generate_fn(engine, tok, format, options) prompt = bpe chat template + bpe::Tokenizer (GGUF vocab)
// The second form must use a tokenizer loaded from the same model file as the engine.
// Requests with tools (or tool-call / tool-result turns) are rendered with toolcall::format_chat(): the first form
// then uses the ChatML Hermes layout (IEngine::format_chat has no tool fields), the second form the GGUF's own family.
// Without tools both forms produce exactly the prompts they produced before.
#pragma once

#include "core/bpe_tokenizer.hpp"
#include "core/engine_interface.hpp"
#include "core/server.hpp"

#include <memory>
#include <mutex>

namespace vegajet::serve {

namespace detail {

inline GenerateResult run_engine(IEngine& engine, const std::vector<int32_t>& prompt, int32_t eos_id,
                                 const GenerateRequest& r, const TokenSink& sink) {
    GenerationParams gp;
    gp.max_tokens = r.max_tokens;
    gp.temperature = r.sampling.temperature;
    gp.top_p = r.sampling.top_p;
    gp.seed = r.sampling.seed;
    gp.stream = r.stream;
    // Stop sequences stay in the server's StopMatcher (it also handles pieces that straddle tokens).
    GenerateResult res;
    res.prompt_tokens = static_cast<uint32_t>(prompt.size());
    res.finish = FinishReason::Length;
    engine.generate(prompt, gp, [&](const TokenEvent& ev) {
        if (ev.prompt_tokens) res.prompt_tokens = ev.prompt_tokens;
        res.completion_tokens = ev.completion_tokens;
        res.prefill_ms = ev.prefill_ms;
        res.decode_ms = ev.decode_ms;
        const bool eos_token = ev.token_id >= 0 && ev.token_id == eos_id;
        if (!ev.text.empty() && !eos_token) {
            const SinkResult sr = sink(ev.text);
            if (!sr) {
                // S-M2: the metrics layer records Cancelled as status 499 + cancelled_total.
                if (sr.client_gone) res.finish = FinishReason::Cancelled;
                return false;
            }
        }
        if (ev.is_eos) {
            res.finish = (!eos_token && static_cast<int32_t>(ev.completion_tokens) >= r.max_tokens) ? FinishReason::Length
                                                                                                   : FinishReason::Stop;
            return false;
        }
        return true;
    });
    return res;
}

inline bool uses_tools(const GenerateRequest& r) {
    if (r.tools.active()) return true;
    for (const auto& t : r.messages)
        if (!t.tool_calls.empty() || t.role == "tool") return true;
    return false;
}

} // namespace detail

// The prompt text the second form encodes for a request (exposed for tests and front ends).
inline std::string format_request(const GenerateRequest& r, bpe::ChatFormat fmt, const bpe::ChatOptions& opt = {}) {
    return toolcall::format_chat(fmt, r.messages, r.tools, opt);
}

// The engine must outlive the server. Calls are serialized (the server already runs one generation at a time).
inline GenerateFn make_engine_generate_fn(IEngine& engine) {
    auto mu = std::make_shared<std::mutex>();
    return [&engine, mu](const GenerateRequest& r, const TokenSink& sink) {
        std::lock_guard<std::mutex> lk(*mu);
        const BpeTokenizer& tok = engine.tokenizer();
        std::string text;
        if (detail::uses_tools(r)) {
            text = toolcall::format_chat(bpe::ChatFormat::ChatML, r.messages, r.tools);
        } else {
            std::vector<ChatMessage> msgs;
            msgs.reserve(r.messages.size());
            for (const auto& t : r.messages) msgs.push_back({t.role, t.content});
            text = engine.format_chat(msgs);
        }
        const std::vector<int32_t> prompt = tok.encode(text, tok.add_bos_default());
        return detail::run_engine(engine, prompt, tok.eos_id(), r, sink);
    };
}

// Chat template + GGUF tokenizer chosen by the caller (both must outlive the server).
inline GenerateFn make_engine_generate_fn(IEngine& engine, const bpe::Tokenizer& tok, bpe::ChatFormat fmt,
                                          bpe::ChatOptions opt = {}) {
    auto mu = std::make_shared<std::mutex>();
    return [&engine, &tok, fmt, opt, mu](const GenerateRequest& r, const TokenSink& sink) {
        std::lock_guard<std::mutex> lk(*mu);
        const std::vector<int32_t> prompt = tok.encode(format_request(r, fmt, opt), tok.add_bos());
        return detail::run_engine(engine, prompt, tok.eos(), r, sink);
    };
}

} // namespace vegajet::serve
