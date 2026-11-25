// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Shared Host/Device Layer Descriptor Structs for VegaJet M=1 Persistent Super-Kernels
// (2 super-kernels/layer -> <=100 dispatches/token on 48L Ornith-122B, <=48 B kernargs!).
#pragma once

#include <stdint.h>

namespace vegajet {
namespace ornith_super {

struct GdnSuperDesc {
    // Stage 0: Optional previous Post-MoE P2P AllReduce + RMSNorm + Q8 Quant
    uint64_t x_partial;
    uint64_t ar_local_rx_base;   // pay_base + g * kArGpuBytes
    uint64_t ar_peer_flag;       // flg_base + peer * 4096 + (prev_slot & 63) * 64
    uint64_t ar_local_flag;      // flg_base + g    * 4096 + (prev_slot & 63) * 64
    uint64_t h_io;
    uint64_t w_attn_norm;
    uint64_t x_q8;
    uint64_t x_ds;

    // Stage 1: w_gdn_qkv_gab GEMV [10304, 3072] VJ_Q4_A8
    uint64_t w_qkv_qs;
    uint64_t w_qkv_dm;
    uint64_t gdn_qkv_gab;
    uint32_t qkv_rows;
    uint32_t qkv_cols;
    uint32_t qkv_qs_stride;
    uint32_t qkv_dm_stride;

    // Stage 2: GDN Conv1D + L2 + Gate + Split-V Recurrence + Gated RMSNorm + Quant
    uint64_t conv_w;
    uint64_t conv_ring;
    uint64_t dt_bias;
    uint64_t ssm_a;
    uint64_t ssm_ring_base;      // ssm_ring[gi] (or spec_gdn slot 0/1 passed dynamically)
    uint64_t ssm_norm_w;
    uint64_t mid_q8;
    uint64_t mid_ds;
    uint64_t gdn_bar;
    uint64_t gdn_sq;

    // Stage 3: w_ssm_out GEMV [3072, 4096] VJ_Q4_A8 + FP16 BAR0 prepush
    uint64_t w_out_qs;
    uint64_t w_out_dm;
    uint64_t out_peer_tx_base;   // pay_base + peer * kArGpuBytes
    uint32_t out_rows;
    uint32_t out_cols;
    uint32_t out_qs_stride;
    uint32_t out_dm_stride;

    // Shared barrier & constants
    uint64_t super_bar;
    uint32_t D;
    uint32_t nk;
    uint32_t nv;
    uint32_t conv_dim;
    uint32_t z_off;
    uint32_t alpha_off;
    uint32_t beta_off;
    float    eps;
    uint64_t out_peer_flag;
};

struct GqaSuperDesc {
    // Stage 0: Optional previous Post-MoE P2P AllReduce + RMSNorm + Q8 Quant
    uint64_t x_partial;
    uint64_t ar_local_rx_base;
    uint64_t ar_peer_flag;
    uint64_t ar_local_flag;
    uint64_t h_io;
    uint64_t w_attn_norm;
    uint64_t x_q8;
    uint64_t x_ds;

    // Stage 1: w_attn_qkv GEMV [8704, 3072] VJ_Q4_A8
    uint64_t w_qkv_qs;
    uint64_t w_qkv_dm;
    uint64_t attn_qkv;
    uint32_t qkv_rows;
    uint32_t qkv_cols;
    uint32_t qkv_qs_stride;
    uint32_t qkv_dm_stride;

    // Stage 2: GQA Prep + In-Kernel 240-WG Split-KV FlashAttention + Gate + Quant
    uint64_t q_norm_w;
    uint64_t k_norm_w;
    uint64_t k_cache_q8;
    uint64_t v_cache_q8;
    uint64_t q_buf;
    uint64_t part_o;
    uint64_t part_ml;
    uint64_t mid_q8;
    uint64_t mid_ds;
    uint32_t n_q_local;
    uint32_t n_kv_local;
    uint32_t rope_dims;
    float    rope_base;

    // Stage 3: w_o GEMV [3072, 4096] VJ_Q4_A8 + FP16 BAR0 prepush
    uint64_t w_o_qs;
    uint64_t w_o_dm;
    uint64_t out_peer_tx_base;
    uint32_t o_rows;
    uint32_t o_cols;
    uint32_t o_qs_stride;
    uint32_t o_dm_stride;

    uint64_t super_bar;
    uint32_t D;
    float    eps;
    uint64_t out_peer_flag;
};

struct MoeSuperDesc {
    // Stage 0: Post-Mixer Prepush FP16 P2P AllReduce + Residual + RMSNorm + Q8 Quant
    uint64_t x_partial;
    uint64_t ar_local_rx_base;
    uint64_t ar_peer_flag;
    uint64_t ar_local_flag;
    uint64_t h_io;
    uint64_t w_post_attn_norm;
    uint64_t x_norm;
    uint64_t x_q8;
    uint64_t x_ds;

    // Stage 1-2: Router FP16 + Shared Gate/Up + Shared Gate + Top-8 + Shared SwiGLU
    uint64_t w_router_f16;
    uint64_t w_sh_gate;
    uint64_t sh_gu_qs;
    uint64_t sh_gu_dm;
    uint64_t router_logits;
    uint64_t sh_gu_out;
    uint64_t sh_gate_sc;
    uint64_t top_ids;
    uint64_t top_w;
    uint64_t sh_q8;
    uint64_t sh_ds;
    uint32_t sh_gu_rows;
    uint32_t sh_gu_stride;
    uint32_t local_ff;
    uint32_t D;

    // Stage 3-4: 4-KiB sm_grid32 IQ2_S Gate+Up (in-place Q8 quant) + Shared Down + Routed DOT8 Down + FP16 BAR0 Prepush
    uint64_t w_gu_iq2s;
    uint64_t mid_q8;
    uint64_t mid_d;
    uint64_t sh_dn_qs;
    uint64_t sh_dn_dm;
    uint64_t sh_down_out;
    uint64_t w_dn_d8_w;
    uint64_t w_dn_d8_s;
    uint64_t out_peer_tx_base;
    uint32_t sh_dn_stride;
    uint32_t dn_w_stride;
    uint32_t dn_s_stride;
    float    eps;

    uint64_t super_bar;
    uint64_t out_peer_flag;
    uint64_t w_gu_d8_w;
    uint64_t w_gu_d8_s;
    uint32_t gu_d8_w_stride;
    uint32_t gu_d8_s_stride;
    uint64_t mid_f32;
};

} // namespace ornith_super
} // namespace vegajet
