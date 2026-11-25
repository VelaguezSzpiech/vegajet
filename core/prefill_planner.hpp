// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Big Lever 3 batched-prefill planner (model-agnostic, CPU only).
//  * Decouples the dense + MoE macro-batch M (256..512 tokens) from the GDN sub-chunk C (16 / 32) and from the
//    FlashAttention-2 tile schedule (Br = 16 head-packed rows, Bc = 32 keys).
//  * Sizes every per-GPU prefill scratch buffer for a chosen intermediate precision, reports bytes per GPU with
//    and without aliasing of the (mutually exclusive) GDN-mixer and attention-mixer working sets.
//  * Plans the prefill AllReduce: payload bytes for the FP16 wire and the block-32 INT8 wire, number of
//    mailbox-sized chunks, and whether a given payload should take an SDMA copy path.
//  * Exact CPU reference of the block-32 INT8 AllReduce wire (bit-compatible with ar_q8_pack4 / ar_q8_accum4 in
//    ops/kernels/vj_ornith_ops.hip) with a per-element error bound, plus the FP16 wire reference.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vegajet::prefill {

// Per-GPU (tensor-parallel shard) model dimensions. All "local" values are per GPU.
struct ModelDims {
    uint32_t D = 0;                 // hidden size (replicated)
    uint32_t n_layers = 0;
    uint32_t n_gdn_layers = 0;
    uint32_t n_attn_layers = 0;
    // GQA attention (local)
    uint32_t n_q_local = 0;
    uint32_t n_kv_local = 0;
    uint32_t head_dim = 0;
    uint32_t kv_row_bytes = 0;      // bytes per (position, kv head) row in the KV cache (272 for Q8 block-32)
    // GDN (local)
    uint32_t gdn_nk = 0, gdn_nv = 0, gdn_dk = 0, gdn_dv = 0;
    uint32_t gdn_conv_dim = 0;      // conv output width per token
    uint32_t gdn_in_stride = 0;     // width of the fused qkv|z|a|b GEMM output per token
    // MoE (local)
    uint32_t n_experts = 0, top_k = 0;
    uint32_t expert_ff_local = 0;   // routed expert FF per GPU
    uint32_t shared_ff_local = 0;   // shared expert FF per GPU
    // Dense projection output widths (local)
    uint32_t attn_qkv_rows = 0;     // fused Q|gate|K|V rows
    uint32_t attn_o_cols = 0;       // = n_q_local * head_dim
    uint32_t gdn_out_cols = 0;      // = gdn_nv * gdn_dv
};

// Ornith-122B-AD-D4 at TP = 2 (values used by models/ornith/ornith_gpu.cpp).
ModelDims ornith_122b_tp2();

enum class Precision : uint8_t {
    kFp32Legacy = 0,     // what today's kernels consume (FP32 GEMM outputs, FP32 Q / gate / partials)
    kFp16Intermediate,   // Lever-3 target: FP16 for every float intermediate except the FP32 residual stream
};

struct Request {
    uint32_t prompt_tokens = 0;
    uint32_t pos0 = 0;               // tokens already in the KV cache / SSM state
    uint32_t macro_m = 512;          // dense + MoE macro batch (clamped to [16, 1024], multiple of 16)
    uint32_t gdn_chunk = 32;         // GDN sub-chunk C: 16 or 32
    uint32_t gdn_wgs_per_head = 4;   // split of each GDN head over WGs (>= 4 -> >= 128 WGs for nv = 32)
    uint32_t attn_br = 16, attn_bc = 32;
    Precision precision = Precision::kFp16Intermediate;
    uint32_t n_cu = 60;
    uint64_t mailbox_bytes = 1048576ULL;   // P2P payload slot size per direction
    uint64_t sdma_min_bytes = 262144ULL;   // payloads at least this large are worth an SDMA copy
    bool allow_int8_wire = false;          // INT8 AllReduce wire is opt-in (FP16 is the project default)
};

struct Buffer {
    std::string name;
    uint64_t bytes = 0;
    const char* dtype = "";
    // Aliasing class: 0 = always live, 1 = GDN-mixer working set, 2 = attention-mixer working set.
    // Classes 1 and 2 are never live at the same time (a layer is either GDN or attention).
    uint32_t alias_class = 0;
};

struct GdnPlan {
    uint32_t chunk = 0;            // C
    uint32_t chunks_per_macro = 0; // ceil(M / C)
    uint32_t wgs = 0;              // nv * wgs_per_head
    uint32_t wgs_per_head = 0;
};

struct AttnPlan {
    uint32_t rows_per_wg = 16;
    uint32_t keys_per_tile = 32;
    uint32_t wgs_per_macro = 0;          // first macro batch
    uint64_t kv_tiles_total = 0;         // sum over all macro batches and WGs of 32-key tiles
    uint64_t kv_bytes_streamed = 0;      // K + V tile bytes pulled into LDS over the whole prompt
    uint64_t score_flops = 0;            // 2 * rows * keys * head_dim for QK^T (causal, incl. masked tail)
};

struct AllReducePlan {
    uint64_t fp16_bytes = 0;      // per direction, per AllReduce, per macro batch
    uint64_t int8_bytes = 0;      // D INT8 + D/32 FP32 scales per row
    uint32_t fp16_chunks = 0;     // mailbox-sized pieces
    uint32_t int8_chunks = 0;
    bool use_int8 = false;        // only with Request::allow_int8_wire
    bool use_sdma = false;        // payload >= sdma_min_bytes
    uint32_t per_layer = 2;       // after the mixer and after the MoE
};

struct Plan {
    bool ok = false;
    std::string error;
    uint32_t macro_m = 0;
    uint32_t n_macro = 0;
    std::vector<Buffer> buffers;
    uint64_t bytes_no_alias = 0;   // sum of all buffers
    uint64_t bytes_aliased = 0;    // class 0 + max(class 1, class 2)
    GdnPlan gdn;
    AttnPlan attn;
    AllReducePlan ar;
};

Plan plan(const ModelDims& dims, const Request& req);
std::string describe(const Plan& p);

// ---------------------------------------------------------------------------------------------------------------
// Block-32 INT8 AllReduce wire reference (one row of D floats, D % 32 == 0).
// Wire record per row: D INT8 values followed by D/32 FP32 scales (D + D/8 bytes), as written by
// ornith_p2p_ar_i8_res_rms_quant_kernel.
//   amax = max |a| over the block (FP32), ld = amax * (1/127), inv = amax > 0 ? 127/amax : 0,
//   q = clamp(round_half_even(a * inv), -127, 127)
// Accumulate (both ranks, commutative bit-for-bit): h += float(q_l)*ld_l + float(q_p)*ld_p (no FMA contraction).
// ---------------------------------------------------------------------------------------------------------------
void ar_q8_pack_row(const float* a, uint32_t D, int8_t* q_out, float* scale_out);
void ar_q8_unpack_row(const int8_t* q, const float* scale, uint32_t D, float* out);
// h[i] += dequant(local)[i] + dequant(peer)[i], exactly the device expression.
void ar_q8_accum_row(float* h, const int8_t* ql, const float* sl, const int8_t* qp, const float* sp, uint32_t D);
// Per-element bound |a - float(q) * ld| <= ar_q8_err_bound(amax of its block).
float ar_q8_err_bound(float block_amax);
// FP16 wire reference: h += half(l) + half(p) (as ornith_p2p_ar_res_rms_quant_kernel).
void ar_f16_accum_row(float* h, const float* l, const float* p, uint32_t D);
uint16_t f32_to_f16_rne(float f);
float f16_to_f32(uint16_t h);

} // namespace vegajet::prefill
