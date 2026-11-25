// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once

#include "core/tokenizer.hpp"

#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace vegajet {

struct ChatMessage {
    std::string role;
    std::string content;
};

struct GenerationParams {
    int32_t max_tokens = 64;
    float temperature = 0.0f;
    float top_p = 1.0f;
    std::vector<std::string> stop;
    bool stream = false;
    uint64_t seed = 42;
};

struct TokenEvent {
    int32_t token_id = -1;
    std::string text;
    bool is_eos = false;
    uint32_t prompt_tokens = 0;
    uint32_t completion_tokens = 0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
};

struct ModelInfo {
    std::string id;
    std::string arch;
    uint32_t context_length = 16384;
    uint32_t vocab_size = 0;
    std::string quant_summary;
};

using TokenCallback = std::function<bool(const TokenEvent&)>; // return false to stop early

class IEngine {
public:
    virtual ~IEngine() = default;
    virtual ModelInfo model_info() const = 0;
    virtual const BpeTokenizer& tokenizer() const = 0;
    virtual std::string format_chat(const std::vector<ChatMessage>& messages) const = 0;
    virtual void generate(const std::vector<int32_t>& prompt_tokens,
                          const GenerationParams& params,
                          const TokenCallback& on_token) = 0;
};

// Strictly CPU-only deterministic mock engine (zero /dev/kfd or GPU calls) for HTTP/CLI testing.
class MockEngine final : public IEngine {
public:
    explicit MockEngine(std::string model_id = "vegajet-mock-gfx906")
        : model_id_(std::move(model_id)) {
        std::vector<std::string> vocab = {
            "<|im_start|>", "<|im_end|>", "<|endoftext|>",
            "Hello", "\xC4\xA0" "from", "\xC4\xA0" "VegaJet", "\xC4\xA0" "mock", "\xC4\xA0" "engine",
            "\xC4\xA0" "on", "\xC4\xA0" "CPU", ".", "\xC4\x8A",
            "The", "\xC4\xA0" "capital", "\xC4\xA0" "of", "\xC4\xA0" "France", "\xC4\xA0" "is", "\xC4\xA0" "Paris"
        };
        // Add ASCII single-byte fallback tokens so any prompt encodes cleanly
        for (int c = 33; c <= 126; ++c) {
            std::string s(1, static_cast<char>(c));
            vocab.push_back(s);
        }
        vocab.push_back("\xC4\xA0"); // space ' ' in GPT-2 BPE
        std::vector<int32_t> ttypes(vocab.size(), 1);
        ttypes[0] = 3; ttypes[1] = 3; ttypes[2] = 3;
        tok_.load_from_vectors(std::move(vocab), {}, std::move(ttypes), 0, 2, -1, true);
    }

    ModelInfo model_info() const override {
        return ModelInfo{
            model_id_,
            "mock-cpu",
            16384,
            static_cast<uint32_t>(tok_.vocab_size()),
            "CPU_MOCK"
        };
    }

    const BpeTokenizer& tokenizer() const override { return tok_; }

    std::string format_chat(const std::vector<ChatMessage>& messages) const override {
        std::string out;
        for (const auto& m : messages) {
            out += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
        }
        out += "<|im_start|>assistant\n";
        return out;
    }

    void generate(const std::vector<int32_t>& prompt_tokens,
                  const GenerationParams& params,
                  const TokenCallback& on_token) override {
        auto t0 = std::chrono::steady_clock::now();
        // Deterministic canned response token sequence: "Hello from VegaJet mock engine on CPU."
        const std::vector<int32_t> reply = {3, 4, 5, 6, 7, 8, 9, 10};
        const int32_t limit = (params.max_tokens > 0) ? params.max_tokens : 16;
        const uint32_t n_prompt = static_cast<uint32_t>(prompt_tokens.size());
        const double prefill_ms = 0.25;

        std::string accum;
        uint32_t produced = 0;
        for (int32_t i = 0; i < limit; ++i) {
            const bool is_last_canned = (static_cast<size_t>(i) >= reply.size());
            if (is_last_canned) {
                TokenEvent ev{};
                ev.token_id = tok_.eos_id();
                ev.text = "";
                ev.is_eos = true;
                ev.prompt_tokens = n_prompt;
                ev.completion_tokens = produced;
                ev.prefill_ms = prefill_ms;
                ev.decode_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - t0).count();
                on_token(ev);
                return;
            }
            int32_t tid = reply[static_cast<size_t>(i)];
            std::string piece = tok_.decode_token(tid, true);
            accum += piece;
            ++produced;

            bool hit_stop = false;
            for (const auto& s : params.stop) {
                if (!s.empty() && accum.find(s) != std::string::npos) {
                    hit_stop = true;
                    break;
                }
            }

            TokenEvent ev{};
            ev.token_id = tid;
            ev.text = piece;
            ev.is_eos = hit_stop || (i + 1 == limit);
            ev.prompt_tokens = n_prompt;
            ev.completion_tokens = produced;
            ev.prefill_ms = prefill_ms;
            ev.decode_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - t0).count();
            if (!on_token(ev) || ev.is_eos) return;
        }
    }

private:
    std::string model_id_;
    BpeTokenizer tok_;
};

} // namespace vegajet
