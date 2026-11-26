// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// `vegajet serve` and the CLI-wide `--gpu` option.
//
//   vegajet serve <model-file> [--gpu SPEC] [--tp N] [--ctx T] [--slots N] [--hsaco-dir DIR] [common options]
//   vegajet serve --mock [--gguf TOKENIZER-FILE] [--slots N] [common options]
//   common: [--host H] [--port P] [--model-id NAME] [--max-queue N] [--chat-format auto|chatml|qwen25|qwen3|ornith]
//           [--enable-thinking] [--reasoning-format auto|none]
//
// Wiring: GGUF tokenizer (core/bpe_tokenizer) + chat template + function calling (core/tool_call_parser, through
// the patched OpenAiServer and server_engine_adapter) for every engine; the model engine comes from
// serve::EngineRegistry (core/engine_factory). When the engine also offers a sched::BatchEngine and --slots > 1,
// requests run through core/request_scheduler (ServeLoop: continuous batching over paged KV) with that many
// concurrent server workers; otherwise they run one at a time through IEngine::generate. `--mock --slots N` uses
// the CPU MockBatchEngine, so the batched path is testable without a GPU.
//
// Exit codes: 0 clean shutdown or help, 1 model / tokenizer file unreadable, 2 usage error (bad flag, bad GPU
// selection, --tp mismatch), 3 engine unavailable (CPU-only build, no engine for the architecture, engine start or
// server bind failure).
#pragma once

#include "core/gpu_select.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace vegajet::cli {

struct ServeConfig {
    std::string model_path;       // positional (or --model)
    bool mock = false;
    std::string tokenizer_gguf;   // --gguf: tokenizer + chat template for --mock
    std::string host = "127.0.0.1";
    uint16_t port = 8080;
    std::string model_id;         // default: "vegajet-mock-gfx906" (mock) or the model file name
    uint32_t slots = 1;
    uint32_t max_queue = 16;
    uint32_t tp = 0;              // 0: number of selected GPUs
    uint32_t ctx = 8192;
    std::string chat_format = "auto";
    bool enable_thinking = false;
    bool parse_reasoning = true;
    std::string hsaco_dir;
    std::string tp_shards, tp_plan; // validated JSON text in EngineSpec; file paths in ServeConfig
    std::string request_log;      // --request-log: JSONL record per request (no prompt / completion text)
    bool metrics_hw = false;      // --metrics-hw: GPU sysfs telemetry (read-only) in GET /metrics
    int32_t max_tokens_cap = 0;   // --max-tokens-cap: cap on "max_tokens" (0: the context length, --ctx)
    std::string api_key;          // --api-key / VEGAJET_API_KEY: bearer key required for /v1/* when set
    std::string cors_origin;      // --cors-origin: the one origin allowed by CORS (empty: no CORS headers)
    bool help = false;
};

// Parses the arguments after "serve" (the --gpu option is extracted earlier). 0 on success, 2 on a usage error.
int parse_serve_args(const std::vector<std::string>& args, ServeConfig& c, std::string* err);
void print_serve_usage(std::FILE* f);

// Removes every "--gpu X" / "--gpu=X" from args. 0 ok, 2 when the value is missing or given twice.
int extract_gpu_option(std::vector<std::string>& args, std::string& spec, bool& given, std::string* err);
// The CLI-wide --gpu handling: resolves --gpu over VJ_GPU_DEVICES, exports the canonical selection as
// VJ_GPU_DEVICES for the KFD runtime, and handles "--gpu list" (prints the enumeration, sets *listed).
// 0 ok, 2 bad selection.
int apply_gpu_option(const std::string* spec, gpusel::Selection& out, bool* listed, std::string* err);

int run_serve(const std::vector<std::string>& args, const gpusel::Selection& gpus);

}  // namespace vegajet::cli
