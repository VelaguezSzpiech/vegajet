// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "ops/op_desc.hpp"

namespace vegajet::ops {

namespace {
struct OpNames { const char* name; const char* symbol; };
constexpr OpNames kOps[static_cast<size_t>(OpKind::Count)] = {
    {"embed", "vj_embed_gather"},
    {"rms_norm", "vj_rms_norm_residual"},
    {"gemv_q8_dp4a", "vj_gemv_q8_dp4a"},
    {"gemv_q4_dot8", "vj_gemv_q4_dot8"},
    {"gemv_iq_packed", "vj_gemv_iq_packed"},
    {"gemm_prefill", "vj_gemm_prefill"},
    {"rope_yarn", "vj_rope_yarn"},
    {"kv_append", "vj_kv_append_paged"},
    {"flash_decode_w64", "vj_flash_decode_wave64"},
    {"flash_prefill", "vj_flash_prefill"},
    {"gdn_conv1d", "vj_gdn_conv1d"},
    {"gdn_chunk_wy", "vj_gdn_chunk_wy"},
    {"gdn_recurrent_step", "vj_gdn_recurrent_step"},
    {"ssm_checkpoint", "vj_ssm_checkpoint"},
    {"moe_router_topk", "vj_moe_router_topk"},
    {"moe_fused_experts", "vj_moe_fused_experts"},
    {"swiglu", "vj_swiglu"},
    {"all_reduce", "vj_all_reduce"},
    {"lm_head", "vj_lm_head"},
    {"draft_verify", "vj_draft_verify"},
};
} // namespace

const char* op_name(OpKind k) {
    const auto i = static_cast<size_t>(k);
    return i < static_cast<size_t>(OpKind::Count) ? kOps[i].name : "invalid";
}

const char* kernel_symbol(OpKind k) {
    const auto i = static_cast<size_t>(k);
    return i < static_cast<size_t>(OpKind::Count) ? kOps[i].symbol : "";
}

} // namespace vegajet::ops
