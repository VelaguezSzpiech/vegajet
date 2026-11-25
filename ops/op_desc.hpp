// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// gfx906 execution descriptors. Model plugins emit a flat OpDesc sequence per step; the GPU backend
// binds each OpKind to a code-object kernel symbol (see kernel_symbol()). Pure data: CPU-compilable.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vegajet::ops {

enum class OpKind : uint32_t {
    Embed = 0,            // token embedding gather (dequant row -> fp32 hidden)
    RmsNorm,              // fused RMSNorm (+ optional residual add)
    GemvQ8Dp4a,           // y = W x, W in VJ_Q8_DP4A (v_dot4_i32_i8)
    GemvQ4Dot8,           // y = W x, W in VJ_Q4_DOT8 (v_dot8_i32_i4)
    GemvIqPacked,         // y = W x, W in VJ_IQ_PACKED (LDS codebook)
    GemmPrefill,          // batched prefill GEMM over the same native formats
    RopeYarn,             // rotary embedding (partial/YaRN aware)
    KvAppend,             // quantize + append K/V into the paged cache tier
    FlashDecodeWave64,    // split-KV FlashDecoding, Wave64, GQA-grouped
    FlashPrefill,         // causal blocked attention for prefill
    GdnConv1d,            // short causal conv over q/k/v streams (gated-delta-net front end)
    GdnChunkWy,           // chunked WY / Neumann-series gated delta rule (prefill)
    GdnRecurrentStep,     // single-token gated delta update (decode)
    SsmCheckpoint,        // ReplaySSM ring checkpoint of recurrent state
    MoeRouterTopK,        // router logits -> softmax/top-k (+ shared-expert gate)
    MoeFusedExperts,      // fused gate/up/act/down over the selected experts
    SwiGlu,               // dense FFN activation
    AllReduce,            // tensor-parallel reduction across GPUs (P2P/XGMI or host-staged)
    LmHead,               // vocab projection + argmax/sampling prep
    DraftVerify,          // batched verification of a draft token tree
    Count,
};

const char* op_name(OpKind k);
// Code-object kernel symbol the GPU backend resolves for this op (stable ABI name).
const char* kernel_symbol(OpKind k);

struct OpDesc {
    OpKind kind = OpKind::Embed;
    int32_t layer = -1;        // -1 for model-level ops
    uint32_t gpu = 0;          // TP rank that executes the op (AllReduce: participating ranks mask)
    uint32_t m = 1;            // tokens in this step (decode=1, prefill=chunk, verify=tree size)
    uint32_t n = 0;            // output features on this rank
    uint32_t k = 0;            // reduction features
    uint32_t split_k = 1;      // from the grid planner
    uint32_t aux = 0;          // op-specific (experts used, kv heads, chunk size, ...)
    std::string tensor;        // primary weight tensor name ("" for weight-free ops)
};

using OpGraph = std::vector<OpDesc>;

} // namespace vegajet::ops
