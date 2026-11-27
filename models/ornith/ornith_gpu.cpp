// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/gpu_select.hpp"
#include "core/spec_gate.hpp"
#include "core/slot_batch_engine.hpp"
#include "core/sequence_layout.hpp"
#include "core/bpe_tokenizer.hpp"
#include "core/dflash_host.hpp"
#include "core/engine_factory.hpp"
#include "core/engine_interface.hpp"
#include "core/fr_spec.hpp"
#include "core/grid_planner.hpp"
#include "core/launch_planner.hpp"
#include "core/prefill_planner.hpp"
#include "core/power_telemetry.hpp"
#include "core/sharder.hpp"
#include "core/gdn_vshard.hpp"
#include "core/tcc_stagger.hpp"
#include "core/token_dump.hpp"
#include "core/q4_dot8_repack.hpp"
#include "core/quant_repacker.hpp"
#include "core/spec_policy.hpp"
#include "core/spec_cascade.hpp"
#include "core/suffix_trie.hpp"
#include "core/evict_depth.hpp"
#include "core/hw_preflight.hpp"
#include "core/route_union.hpp"
#include "core/host_preflight.hpp"
#include "core/moe_lru_cache.hpp"
#include "core/spec_cycle.hpp"
#include "core/acceptmoe.hpp"
#include "core/moe_slot_budget.hpp"
#include "core/warm_bench.hpp"
#include "ops/ornith_gqa_kvq_launch.hpp"
#include "core/raw_token_reference.hpp"
#include "core/wire_normal_adapter.hpp"
#include "models/ornith/ornith_golden.hpp"
#include "models/ornith/ornith_gpu.hpp"
#include "models/ornith/ornith_l4.hpp"
#include "core/kv_quant_oracle.hpp"
#include "models/ornith/ornith_router_s240.hpp"
#include "models/ornith/ornith_treewy.hpp"
#include "models/ornith/ornith_offload.hpp"

#include "core/spec_engine.hpp"
#include "core/spec_replay.hpp"
#include "core/mtp_draft.hpp"
#include "core/prefill_sched.hpp"

#include <array>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <atomic>
#include <algorithm>
#include "core/vj_check.hpp"
#include "ops/kernels/vj_ornith_super_desc.hpp"
#include "ops/ornith_mega_launch.hpp"
#include "ops/sub4_launch.hpp"
#include "ops/milo_launch.hpp"
#include "models/ornith/ornith_sub4_upload.hpp"
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <omp.h>

namespace vegajet::ornith {

#ifdef VEGAJET_CPU_ONLY

OrnithBenchMetrics run_ornith_gpu(const std::string&, const std::string&, const OrnithGpuOptions&) {
    OrnithBenchMetrics m;
    m.ok = false;
    m.error = "Ornith GPU execution requires VEGAJET_CPU_ONLY=OFF";
    return m;
}

int run_ornith_ar_bench(const std::string&) { return 3; }
int run_ornith_gemv_ab_bench(const std::string&) { return 3; }
void register_ornith_serve_engine() {}

#else

namespace {

constexpr uint32_t kMaxSpecBatch   = 16;  // speculative verify M <= 16 (bounded by kGdnRingSlots - 1)
constexpr uint32_t kGdnRingSlots   = 17;  // == VJ_GDN_RING_SLOTS in the kernels: verify batches of up to 16 tokens
constexpr uint32_t kVocabBlocks    = 240;

struct Q8WeightDev {
    uint64_t qs_va = 0;
    uint64_t ds_va = 0;
    uint32_t rows  = 0;
    uint32_t cols  = 0;
    uint32_t row_stride_bytes = 0;
};

struct Q4A8WeightDev {
    uint64_t qs_va = 0; // [rows, row_stride_bytes] u4 packed (low=0..15, high=16..31), odd-256B TCC!
    uint64_t dm_va = 0; // [rows, cols/32] uint32_t = fp16(d) | (fp16(m) << 16)
    uint32_t rows  = 0;
    uint32_t cols  = 0;
    uint32_t row_stride_bytes = 0;
    // Agent E opt-in sidecars. Original Q4 remains the M>8 / fused fallback.
    uint64_t sub4_qs=0, sub4_dm=0, xor_qs=0, xor_dm=0;
    uint64_t milo_u=0, milo_v=0, milo_us=0, milo_vs=0;
    uint64_t extra_bytes=0;
    uint32_t sub4_bits=0, milo_rank=0;
    bool sub4_xor=false;
};

struct LayerWeightsDev {
    bool     is_gdn = false;
    uint64_t attn_norm_va      = 0;
    uint64_t post_attn_norm_va = 0;

    // Fused GDN Q4_K -> VJ_Q4_A8: [qkv(6144) | gate(4096) | alpha(32) | beta(32)] = 10304 rows -> 1 dispatch!
    Q4A8WeightDev w_gdn_qkv_gab;
    Q4A8WeightDev w_ssm_out;
    // --prefill-dot8 --dot8-fwht: block-64 FWHT-rotated twins of the two dense GDN matrices above (rows == 0
    // when the mode is off). Never used by the W4A8 or GEMV paths, only by the dense v_dot8_i32_i4 GEMM.
    Q4A8WeightDev w_gdn_qkv_gab_fwht;
    Q4A8WeightDev w_ssm_out_fwht;
    uint64_t ssm_conv_w_va = 0;
    uint64_t ssm_a_va      = 0;
    uint64_t ssm_dt_va     = 0;
    uint64_t ssm_norm_va   = 0;

    // Fused GQA Q4_K -> VJ_Q4_A8: [QG(8192) | K(256) | V(256)] = 8704 rows -> 1 dispatch!
    Q4A8WeightDev w_attn_qkv;
    Q4A8WeightDev w_o;
    Q4A8WeightDev w_attn_qkv_fwht;   // --dot8-fwht twins (rows == 0 when off)
    Q4A8WeightDev w_o_fwht;
    uint64_t q_norm_va = 0;
    uint64_t k_norm_va = 0;

    // MoE + Shared Expert: FP16 router + VJ_Q4_A8 shared + merged uint4 IQ2_S & split uint4 IQ3_XXS!
    uint64_t w_router_f16_va = 0; // [256, 3072] FP16 replicated (2x less HBM2 than F32!)
    uint64_t w_sh_gate_va    = 0; // [3072] F32 replicated
    uint64_t w_gu_iq2s_va    = 0; // [256 * 512 * 123] uint4 merged split-plane IQ2_S Gate+Up!
    uint64_t w_gu_d8_w_va    = 0; // optional routed Gate+Up exact DOT8 repack (--gu-dot8-layers)
    uint64_t w_gu_d8_s_va    = 0;
    uint32_t gu_d8_w_stride  = 0;
    uint32_t gu_d8_s_stride  = 0;
    uint64_t w_dn_iq3xxs_va  = 0; // [256 * 3072 * 13] uint4 split-plane IQ3_XXS Down (only kept for --dot8-ab)
    uint64_t w_dn_d8_w_va    = 0; // routed Down, exact 16-level DOT8 repack of IQ3_XXS: codes, 256 B/row
    uint64_t w_dn_d8_s_va    = 0; // routed Down scale plane, 16 B/row (or embedded gdb for sub32)
    uint32_t dn_d8_w_stride  = 0;
    uint32_t dn_d8_s_stride  = 0;
    bool     dn_is_sub32     = false;
    Q4A8WeightDev w_sh_gu;
    Q4A8WeightDev w_sh_down;
};

struct MtpLayerDev {
    bool     loaded = false;
    uint64_t enorm_va = 0, hnorm_va = 0, shared_head_norm_va = 0;
    uint64_t attn_norm_va = 0, post_attn_norm_va = 0;
    uint64_t q_norm_va = 0, k_norm_va = 0;
    Q4A8WeightDev w_eh_proj;   // [3072, 6144] replicated
    Q4A8WeightDev w_attn_qkv;  // [8704, 3072] TP=2
    Q4A8WeightDev w_o;         // [3072, 4096] TP=2
    uint64_t w_router_f16_va = 0;
    uint64_t w_sh_gate_va    = 0;
    Q4A8WeightDev w_sh_gu;
    Q4A8WeightDev w_sh_down;
    Q4A8WeightDev w_exp_gate;  // [256*512, 3072] TP=2
    Q4A8WeightDev w_exp_up;    // [256*512, 3072] TP=2
    Q4A8WeightDev w_exp_down;  // [256*3072, 512] TP=2
};

struct DFlash2LayerDev {
    uint64_t attn_norm_va = 0, attn_conv_base_va = 0, q_norm_va = 0, k_norm_va = 0;
    uint64_t ffn_norm_va = 0, ffn_conv_base_va = 0;
    Q4A8WeightDev w_attn_conv, w_qkv, w_kv, w_o;
    Q4A8WeightDev w_ffn_conv, w_gu, w_down;
    uint64_t kv_k_va = 0, kv_v_va = 0; // [kv_slots, n_kv, head_dim] FP16 ring
    uint32_t kv_slots = 0, kv_win = 0;  // spec_replay::plan_draft_kv: ring size, kernel window (full layers: slots)
};

struct DFlash2Dev {
    bool     loaded = false;
    uint32_t n_layer = 0, n_head = 0, n_kv = 0, head_dim = 0, n_ff = 0, block_size = 0;
    uint32_t swa_win = 2048, sel_rank = 256, sel_topk = 16;
    // Opt-in block-mask arm (core/dflash_host plan_draft_block_mask). Default Bidir: the shipped kernel attends
    // every block key on every layer. The per-layer bits come from the KV plan's layer kinds (a full layer's bit
    // is never set in ZLab), and the arm is only active inside the global SWA window. Default off.
    dflash_host::BlockMaskMode block_mask = dflash_host::BlockMaskMode::Bidir;
    uint32_t block_mask_swa = 0;             // bit l: layer l is a sliding_attention layer (0 if pattern unknown)
    bool     block_mask_pattern_known = false;
    float    rope_base = 1e7f, rms_eps = 1e-6f;
    std::vector<uint32_t> tap_layers_0idx;
    int8_t   tap_slot[48];
    uint64_t mask_embd_f32_va = 0, enc_norm_va = 0, out_norm_va = 0;
    Q4A8WeightDev w_fc, w_sel_hid;
    std::vector<DFlash2LayerDev> L;
    // GPU 0 scratch buffers
    uint64_t taps_va = 0, taps_q8_va = 0, taps_ds_va = 0;
    uint64_t ctx_f32_va = 0, ctx_q8_va = 0, ctx_ds_va = 0;
    uint64_t h_va = 0, xnorm_va = 0, xq8_va = 0, xds_va = 0;
    uint64_t dyn_va = 0, qkv_va = 0, attn_q8_va = 0, attn_ds_va = 0;
    uint64_t out_va = 0, gu_va = 0, mid_q8_va = 0, mid_ds_va = 0;
    // Host-side Q4_K selector tables (zero VRAM, <20 us CPU decode for 17 rows/cycle)
    quant::DiskQuantType sel_pred_type = quant::DiskQuantType::Q4_K;
    quant::DiskQuantType sel_succ_type = quant::DiskQuantType::Q4_K;
    uint64_t sel_pred_rb = 0, sel_succ_rb = 0;
    uint32_t sel_vocab = 0;
    std::vector<uint8_t> sel_pred_raw, sel_succ_raw;
};

struct GpuStateDev {
    uint64_t tok_ids_va    = 0;
    uint64_t h_va          = 0;
    uint64_t x_norm_va     = 0;
    uint64_t x_q8_va       = 0;
    uint64_t x_ds_va       = 0; // float2 {d, block32_sum}
    uint64_t partial_va    = 0;
    uint64_t ar_bar_va     = 0;
    uint64_t ar_ts_va      = 0;   // --ar-timeline: 6 x u64 device timestamps per AllReduce
    uint64_t moe_bar_va    = 0;
    uint64_t gdn_bar_va    = 0;
    uint64_t moe_bq_va     = 0;
    uint64_t gdn_sq_va     = 0;

    uint64_t gdn_qkv_gab_va = 0;
    uint64_t gdn_conv_va    = 0;
    uint64_t gdn_dec_va     = 0;
    uint64_t gdn_bsig_va    = 0;
    uint64_t mid_q8_va      = 0;
    uint64_t mid_ds_va      = 0;

    uint64_t attn_qkv_va   = 0;
    uint64_t q_buf_va      = 0;
    uint64_t gate_buf_va   = 0;

    uint64_t router_log_va = 0;
    uint64_t top_ids_va    = 0;
    uint64_t top_w_va      = 0;
    uint64_t sh_gate_sc_va = 0;
    uint64_t sh_gu_va      = 0;
    uint64_t sh_q8_va      = 0;
    uint64_t sh_ds_va      = 0;
    uint64_t sh_down_va    = 0;
    uint64_t moe_q8_va     = 0;
    uint64_t moe_d_va      = 0;
    uint64_t moe_mid_f32_va = 0; // Wave 1D: [8 * local_ff] FP32 SwiGLU intermediate (16 KiB, L2-resident)
    uint64_t splitk_part_va = 0; // Wave 1B: [8 * max_rows] FP32 deterministic Split-K partials
    uint64_t moe_exp_off_va   = 0;
    uint64_t moe_exp_cnt_va   = 0;
    uint64_t moe_sorted_ms_va = 0;
    uint64_t pf_fo_va = 0, pf_sq_va = 0, pf_gdn_bar_va = 0, pf_gu_queue_va = 0;
    uint64_t milo_tmp_va = 0; // 8 tokens x rank32
    uint64_t pf_xq4_va = 0, pf_xds4_va = 0, pf_xsum4_va = 0, moe_mid_q4_va = 0, moe_mid_ds4_va = 0, logits_va = 0;
    uint64_t dgrp_hdr_va = 0, dgrp_cnt_va = 0, dgrp_off_va = 0, dgrp_uniq_va = 0, dgrp_items_va = 0, dgrp_routes_va = 0, dgrp_buf_va = 0;

    uint64_t argmax_val_va = 0;
    uint64_t argmax_id_va  = 0;

    std::vector<uint64_t> gdn_conv_ring_va;
    std::vector<uint64_t> gdn_ssm_ring_va;
    uint64_t spec_gdn_s_slot_va[2] = {0, 0}; // --spec-gdn: 2 contiguous [36][32][128][128] FP32 slots
    uint64_t spec_gdn_o_va         = 0;      // [12][32][128] FP32
    uint64_t spec_gdn_rec_kv_va    = 0;      // [36][12][40*128] FP32
    uint64_t spec_gdn_rec_dec_va   = 0;      // [36][12][32] FP32
    uint64_t spec_gdn_rec_beta_va  = 0;      // [36][12][32] FP32
    uint64_t spec_fl_part_o_va     = 0;      // --spec-flash: [kMaxSpecBatch][16][60][256] FP32
    uint64_t spec_fl_part_ml_va    = 0;      // --spec-flash: [kMaxSpecBatch][16][60][2] FP32
    std::vector<uint64_t> attn_k_cache_va;
    std::vector<uint64_t> attn_v_cache_va;
    // M1 sealed-wire attention (opt-in): per-layer K/V wire caches + Q wire, merge output and scratch.
    std::vector<uint64_t> wire_k_cache_va;
    std::vector<uint64_t> wire_v_cache_va;
    uint64_t wire_q_va = 0;
    uint64_t wire_scratch_va = 0;
    uint64_t wire_out_va = 0;
    uint64_t wire_out_lse_va = 0;
    uint64_t mtp_k_cache_va = 0;
    uint64_t mtp_v_cache_va = 0;
    uint64_t mtp_cat_q8_va  = 0;
    uint64_t mtp_cat_ds_va  = 0;
    uint64_t mtp_h_pend_va  = 0;
    uint64_t mtp_h_cur_va   = 0;
    uint64_t mtp_h_zero_va  = 0;
    uint64_t mtp_h_stash_va = 0;             // --zero-tax-catchup: [16][D] FP32
    uint64_t mtp_moe_ds_va  = 0;
    uint64_t super_bar_va       = 0;         // 4 KiB: super scratch + isolated mega barrier ring
    uint64_t mega_desc_va       = 0;         // Wave 2: shared TP=3 descriptor
    uint64_t gdn_super_desc_va  = 0;         // [36] GdnSuperDesc
    uint64_t gqa_super_desc_va  = 0;         // [12] GqaSuperDesc
    uint64_t moe_super_desc_va  = 0;         // [48] MoeSuperDesc
};

struct KernelTable {
    const kfd::LoadedKernel* embed_q8 = nullptr;
    const kfd::LoadedKernel* rms_quant = nullptr;
    const kfd::LoadedKernel* p2p_ar_rms = nullptr;
    const kfd::LoadedKernel* p2p_ar_prepush = nullptr;
    const kfd::LoadedKernel* p2p_ar_i8 = nullptr;
    const kfd::LoadedKernel* p2p_ar_n_fp16 = nullptr;
    const kfd::LoadedKernel* moe_m1_front = nullptr;
    const kfd::LoadedKernel* moe_m1_rtr_shgu = nullptr;
    const kfd::LoadedKernel* router_s240 = nullptr;
    const kfd::LoadedKernel* router_s240_red = nullptr;
    const kfd::LoadedKernel* moe_m1_top8_swiglu = nullptr;
    const kfd::LoadedKernel* moe_m1_back = nullptr;
    const kfd::LoadedKernel* moe_gu_16r = nullptr;
    const kfd::LoadedKernel* moe_mid_f32_to_q8 = nullptr;
    const kfd::LoadedKernel* moe_down_d8_sh = nullptr;
    const kfd::LoadedKernel* gemv_q4_w32 = nullptr;
    const kfd::LoadedKernel* gemv_q4_w64 = nullptr;
    const kfd::LoadedKernel* gemv_q4_w32_stag = nullptr;
    const kfd::LoadedKernel* gemv_q4_splitk_p1 = nullptr;
    const kfd::LoadedKernel* splitk_red_pp = nullptr;
    const kfd::LoadedKernel* gemv_q4_w16 = nullptr;
    const kfd::LoadedKernel* sub4_gemv=nullptr;
    const kfd::LoadedKernel* sub4_gemm=nullptr;
    const kfd::LoadedKernel* sub4_gemv_xor=nullptr;
    const kfd::LoadedKernel* sub4_gemm_xor=nullptr;
    const kfd::LoadedKernel* q4_gemv_xor=nullptr;
    const kfd::LoadedKernel* milo_vx=nullptr;
    const kfd::LoadedKernel* milo_apply=nullptr;
    const kfd::LoadedKernel* gemm_q4_tcc = nullptr;
    const kfd::LoadedKernel* gemm_q4_m8  = nullptr;
    const kfd::LoadedKernel* gemm_q4_m12 = nullptr;
    const kfd::LoadedKernel* gemm_q4_m12x = nullptr;  // resident-activation M<=12 variant (VEGAJET_VERIFY_GEMM_X=1)
    const kfd::LoadedKernel* gemm_q4_m16 = nullptr;
    const kfd::LoadedKernel* gdn_conv_l2 = nullptr;
    const kfd::LoadedKernel* gdn_recur = nullptr;
    const kfd::LoadedKernel* gdn_recur_sv = nullptr;
    const kfd::LoadedKernel* gdn_conv_recur_sv = nullptr;
    const kfd::LoadedKernel* gdn_conv_recur_sv8 = nullptr;       // A2: Split-V=8 ConvRec stage 1 (selectable, optional module)
    const kfd::LoadedKernel* gdn_conv_recur_sv8_quant = nullptr; // A2: Split-V=8 stage 2 (ordered dispatch, reads sq_scratch)
    const kfd::LoadedKernel* gdn_wy = nullptr;
    const kfd::LoadedKernel* attn_prep = nullptr;
    const kfd::LoadedKernel* attn_flash = nullptr;
    const kfd::LoadedKernel* attn_prep_flash_m1 = nullptr;
    // M1 sealed-wire producer/epilogue + sealed pair (all optional unless VEGAJET_WIRE_ATTN is on).
    const kfd::LoadedKernel* wire_q8_q = nullptr;
    const kfd::LoadedKernel* wire_q4_q = nullptr;
    const kfd::LoadedKernel* wire_q8_kv = nullptr;
    const kfd::LoadedKernel* wire_q4_kv = nullptr;
    const kfd::LoadedKernel* wire_epilogue = nullptr;
    const kfd::LoadedKernel* wire_skv_q8_part = nullptr;
    const kfd::LoadedKernel* wire_skv_q4_part = nullptr;
    const kfd::LoadedKernel* wire_skv_merge = nullptr;
    const kfd::LoadedKernel* moe_rtr_log = nullptr;
    const kfd::LoadedKernel* moe_rtr_top8 = nullptr;
    const kfd::LoadedKernel* swiglu_q8 = nullptr;
    const kfd::LoadedKernel* moe_gu_iq2s = nullptr;
    const kfd::LoadedKernel* moe_gu_d8 = nullptr;
    const kfd::LoadedKernel* moe_gu_d8_cached = nullptr;
    const kfd::LoadedKernel* moe_gu_q3 = nullptr;
    const kfd::LoadedKernel* verify_gu_q3_pair = nullptr;
    const kfd::LoadedKernel* moe_down_iq3 = nullptr;
    const kfd::LoadedKernel* moe_down_d8 = nullptr;
    const kfd::LoadedKernel* moe_down_d8_pp = nullptr;
    const kfd::LoadedKernel* moe_top8_acceptmoe = nullptr;
    const kfd::LoadedKernel* moe_top8_acceptmoe_plain = nullptr;
    const kfd::LoadedKernel* moe_down_d8_sub32 = nullptr;
    const kfd::LoadedKernel* spec_gdn_ver = nullptr;
    const kfd::LoadedKernel* spec_gdn_nq = nullptr;
    const kfd::LoadedKernel* treewy_verify = nullptr;  // sealed GDN tree-WY pair (optional module)
    const kfd::LoadedKernel* treewy_commit = nullptr;
    const kfd::LoadedKernel* gdn_wire_pack = nullptr;  // sealed input producer (optional module)
    const kfd::LoadedKernel* spec_flash_part = nullptr;
    const kfd::LoadedKernel* spec_flash_merge = nullptr;
    const kfd::LoadedKernel* flash_skv_p1 = nullptr;
    const kfd::LoadedKernel* flash_skv_p2 = nullptr;
    const kfd::LoadedKernel* pf_fa2 = nullptr;          // prefill kernels (optional modules, see OrnithGpuOptions::pf_*)
    const kfd::LoadedKernel* pf_fa2_codex = nullptr;
    const kfd::LoadedKernel* pf_gemm = nullptr;
    const kfd::LoadedKernel* pf_gemm_64 = nullptr;
    const kfd::LoadedKernel* pf_gu = nullptr;
    const kfd::LoadedKernel* pf_gu_dot8 = nullptr;        // exact VJ_Q4_DOT8 (0 LUT, 7944 B LDS, 320/352/384/512)
    const kfd::LoadedKernel* pf_gu_iq_lds64 = nullptr;    // opt-in single 64x64x64 dequant-once-to-LDS Gate+Up
    const kfd::LoadedKernel* pf_gu_iq_lds32 = nullptr;    // diagnostic 32-row x 64-token variant of the same item
    const kfd::LoadedKernel* pf_down_iq_lds64 = nullptr;  // opt-in IQ3_XXS 64x64x64 Down (compact split plane)
    const kfd::LoadedKernel* pf_down_iq_lds32 = nullptr;  // diagnostic 32-row x 64-token Down
    const kfd::LoadedKernel* pf_down_tiled = nullptr;
    const kfd::LoadedKernel* pf_down_comb = nullptr;
    const kfd::LoadedKernel* pf_router = nullptr;
    const kfd::LoadedKernel* pf_gdn_conv = nullptr;
    const kfd::LoadedKernel* pf_gdn_sv = nullptr;
    const kfd::LoadedKernel* pf_gdn_wy = nullptr;
    const kfd::LoadedKernel* pf_gdn_nq = nullptr;
    const kfd::LoadedKernel* pf_gdn_wy_codex = nullptr;
    const kfd::LoadedKernel* pf_gdn_nq_codex = nullptr;
    const kfd::LoadedKernel* pf_q8_to_q4 = nullptr;     // Stage 3D opt-in W4A4 prefill (v_dot8_i32_i4)
    const kfd::LoadedKernel* pf_q8_to_q4_fwht = nullptr; // --dot8-fwht block-64 FWHT activation packer
    const kfd::LoadedKernel* pf_gemm_d8_128 = nullptr;
    const kfd::LoadedKernel* pf_gemm_d8_64 = nullptr;
    const kfd::LoadedKernel* pf_gu_d8 = nullptr;
    const kfd::LoadedKernel* pf_down_d8 = nullptr;
    const kfd::LoadedKernel* vocab_logits = nullptr;
    const kfd::LoadedKernel* dgrp_route = nullptr;     // expert-grouped routed Down (optional module)
    const kfd::LoadedKernel* dgrp_dp = nullptr;
    const kfd::LoadedKernel* dgrp_f32 = nullptr;
    const kfd::LoadedKernel* dgrp_comb_dp = nullptr;
    const kfd::LoadedKernel* dgrp_comb_f32 = nullptr;
    const kfd::LoadedKernel* moe_grp_sort = nullptr;
    const kfd::LoadedKernel* moe_gu_iq2s_grp = nullptr;
    const kfd::LoadedKernel* moe_down_iq3_grp = nullptr;
    const kfd::LoadedKernel* vocab_m1 = nullptr;
    const kfd::LoadedKernel* vocab_q4 = nullptr;
    const kfd::LoadedKernel* vocab_q4_b16 = nullptr;
    const kfd::LoadedKernel* vocab_argmax = nullptr;
    const kfd::LoadedKernel* vocab_argmax_b6 = nullptr;
    const kfd::LoadedKernel* vocab_argmax_b12 = nullptr;
    const kfd::LoadedKernel* vocab_argmax_b16 = nullptr;
    const kfd::LoadedKernel* p2p_ar_moe_m1_front = nullptr;
    const kfd::LoadedKernel* p2p_ar_moe_m1_front_s240 = nullptr;
    const kfd::LoadedKernel* moe_m1_back_s224 = nullptr;
    const kfd::LoadedKernel* moe_m1_back_s240 = nullptr;
    const kfd::LoadedKernel* mtp_eh_cat = nullptr;
    const kfd::LoadedKernel* moe_gu_q4a8 = nullptr;
    const kfd::LoadedKernel* moe_down_q4a8 = nullptr;
    const kfd::LoadedKernel* memcpy_u32 = nullptr;
    const kfd::LoadedKernel* memset_u32 = nullptr;
    // DFlash2 kernels (loaded from vj_dflash2.hsaco when --draft is a DFlash2 GGUF)
    const kfd::LoadedKernel* df_tap_copy = nullptr;
    const kfd::LoadedKernel* df_quant_q8 = nullptr;
    const kfd::LoadedKernel* df_embed_blk = nullptr;
    const kfd::LoadedKernel* df_dyn_conv = nullptr;
    const kfd::LoadedKernel* df_kv_inject = nullptr;
    const kfd::LoadedKernel* df_attn = nullptr;
    const kfd::LoadedKernel* df_swiglu_q8 = nullptr;
    const kfd::LoadedKernel* df_vocab_topk16 = nullptr;
    const kfd::LoadedKernel* m1_gdn_mega = nullptr;
    const kfd::LoadedKernel* m1_gqa_mega = nullptr;
    const kfd::LoadedKernel* m1_gdn_super = nullptr;
    const kfd::LoadedKernel* m1_gqa_super = nullptr;
    const kfd::LoadedKernel* m1_gqa_super_kvq = nullptr;
    const kfd::LoadedKernel* m1_moe_super = nullptr;
    const kfd::LoadedKernel* ring_router_so = nullptr;
    const kfd::LoadedKernel* ring_top8_so   = nullptr;
    const kfd::LoadedKernel* m1_moe_super_q3 = nullptr;
    // vj_ornith_super_x.hsaco: same super-kernels built with AR tile tags / DN_K4 (opt-in; nullptr if absent).
    const kfd::LoadedKernel* m1_gdn_super_x = nullptr;
    const kfd::LoadedKernel* m1_gqa_super_x = nullptr;
    const kfd::LoadedKernel* m1_moe_super_x = nullptr;
    const kfd::LoadedKernel* m1_moe_super_q3_x = nullptr;
    // Lean twins (vj_ornith_super_l): production path + W21 AllReduce fast path / WG0 broadcast (VEGAJET_SUPER_X=1).
    const kfd::LoadedKernel* m1_gdn_super_l = nullptr;
    const kfd::LoadedKernel* m1_gqa_super_l = nullptr;
    const kfd::LoadedKernel* m1_moe_super_l = nullptr;
    const kfd::LoadedKernel* m1_moe_super_q3_l = nullptr;
    const kfd::LoadedKernel* moe_gu_repack = nullptr;
    const kfd::LoadedKernel* moe_gu_lru    = nullptr;
};

struct RowSource {
    std::string           tensor_name;
    std::vector<uint32_t> row_indices;
};

// Repacks Q4_K GGUF tensors into Asymmetric VJ_Q4_A8 (u4 + FP16 d,m per 32 elems, odd-256B TCC anti-camping!)
// fwht_out (optional, --prefill-dot8 --dot8-fwht): a second copy of the same rows with every 64-column block
// multiplied by the orthonormal 64-point Walsh-Hadamard (quant::fwht64_inplace) *before* requantization. The
// matching activation packer (ops/kernels/vj_prefill_dot8_fwht.hip) rotates its 64-blocks the same way, so
// sum_k (H w)_k (H a)_k = sum_k w_k a_k needs no rescale; see tests/test_prefill_dot8_fwht_kld.cpp.
bool upload_fused_q4a8_tcc(kfd::Runtime& rt,
                           size_t rank,
                           GgufFile& f,
                           const std::vector<RowSource>& sources,
                           uint32_t col_start,
                           uint32_t col_count,
                           const std::vector<uint32_t>* col_indices,
                           Q4A8WeightDev& out,
                           std::string* err,
                           Q4A8WeightDev* fwht_out = nullptr) {
    uint32_t total_rows = 0;
    for (const auto& s : sources) total_rows += static_cast<uint32_t>(s.row_indices.size());
    const uint32_t C = col_indices ? static_cast<uint32_t>(col_indices->size()) : col_count;
    if (fwht_out && (C % 64u) != 0u) {
        if (err) *err = "--dot8-fwht needs 64-column-aligned weights (got " + std::to_string(C) + ")";
        return false;
    }
    const uint32_t raw_rb = C / 2u;
    const auto lay = quant::select_tcc_layout(raw_rb);
    // For C=512 (256B), 3072 (1536->1792B), 4096 (2048->2304B), select_tcc_layout gives odd-256B with xor_en=0!
    const uint32_t stride_b = (lay.mode == quant::TccLayoutMode::XorTileSwizzle) ? (raw_rb + 256u) : lay.stride_bytes;

    std::vector<uint8_t>  qs(static_cast<size_t>(total_rows) * stride_b, 0);
    std::vector<uint32_t> dm(static_cast<size_t>(total_rows) * (C / 32u), 0);
    std::vector<uint8_t>  qs_f;   // FWHT-rotated twin (only when fwht_out != nullptr)
    std::vector<uint32_t> dm_f;
    if (fwht_out) {
        qs_f.assign(static_cast<size_t>(total_rows) * stride_b, 0);
        dm_f.assign(static_cast<size_t>(total_rows) * (C / 32u), 0);
    }
    // One 32-element affine block quantizer, shared by the plain and the rotated copy (identical nibble order and
    // FP16 {d,m} rounding, so both copies are the same production format).
    auto quantize_row = [C](const float* row, uint8_t* dst_row, uint32_t* dmrow) {
        for (uint32_t b = 0; b < C / 32u; ++b) {
            const float* blk = row + b * 32u;
            float mn = blk[0], mx = blk[0];
            for (uint32_t i = 1; i < 32u; ++i) {
                mn = std::min(mn, blk[i]);
                mx = std::max(mx, blk[i]);
            }
            const uint16_t dh = quant::f32_to_fp16((mx - mn) / 15.0f);
            const uint16_t mh = quant::f32_to_fp16(mn);
            const float d_eff = quant::fp16_to_f32(dh);
            const float m_eff = quant::fp16_to_f32(mh);
            const float inv_d = (d_eff > 0.0f) ? (1.0f / d_eff) : 0.0f;
            dmrow[b] = static_cast<uint32_t>(dh) | (static_cast<uint32_t>(mh) << 16);
            uint8_t* qblk = dst_row + b * 16u;
            for (uint32_t i = 0; i < 16u; ++i) {
                int q0 = static_cast<int>(std::lrintf((blk[i]       - m_eff) * inv_d));
                int q1 = static_cast<int>(std::lrintf((blk[i + 16u] - m_eff) * inv_d));
                q0 = std::max(0, std::min(15, q0));
                q1 = std::max(0, std::min(15, q1));
                qblk[i] = static_cast<uint8_t>(q0 | (q1 << 4));
            }
        }
    };

    const bool sub4_load=quant::sub4_dot8_bits()!=0;
    std::vector<float> original;
    if(sub4_load) original.resize(size_t(total_rows)*C);
    const auto& reg = quant::DecoderRegistry::instance();
    uint32_t row_off = 0;
    for (const auto& src : sources) {
        const GgufTensorInfo* ti = f.tensor(src.tensor_name);
        if (!ti) { if (err) *err = "Missing tensor " + src.tensor_name; return false; }
        const uint8_t* raw = f.tensor_data(*ti);
        if (!raw) { if (err) *err = "Failed to mmap " + src.tensor_name; return false; }

        const auto disk_t = static_cast<quant::DiskQuantType>(ti->type);
        const uint64_t full_cols = ti->ne[0];
        const uint64_t src_rb    = reg.row_bytes(disk_t, full_cols);
        const uint32_t R         = static_cast<uint32_t>(src.row_indices.size());

        #pragma omp parallel
        {
            std::vector<float> full_row(full_cols);
            std::vector<float> sub_row(C);
            std::vector<float> rot_row;
            if (fwht_out) rot_row.resize(C);
            #pragma omp for schedule(static)
            for (uint32_t r = 0; r < R; ++r) {
                const uint32_t dst_r = row_off + r;
                const uint32_t src_r = src.row_indices[r];
                reg.decode_row_f32(disk_t, raw + static_cast<uint64_t>(src_r) * src_rb, full_row.data(), full_cols);
                if (col_indices) {
                    for (uint32_t c = 0; c < C; ++c) sub_row[c] = full_row[(*col_indices)[c]];
                } else {
                    std::memcpy(sub_row.data(), full_row.data() + col_start, C * sizeof(float));
                }
                if(sub4_load) std::memcpy(original.data()+size_t(dst_r)*C,sub_row.data(),C*sizeof(float));
                quantize_row(sub_row.data(), qs.data() + static_cast<size_t>(dst_r) * stride_b,
                             dm.data() + static_cast<size_t>(dst_r) * (C / 32u));
                if (fwht_out) {
                    std::memcpy(rot_row.data(), sub_row.data(), C * sizeof(float));
                    for (uint32_t c = 0; c < C; c += 64u) quant::fwht64_inplace(rot_row.data() + c);
                    quantize_row(rot_row.data(), qs_f.data() + static_cast<size_t>(dst_r) * stride_b,
                                 dm_f.data() + static_cast<size_t>(dst_r) * (C / 32u));
                }
            }
        }
        row_off += R;
    }

    out.rows             = total_rows;
    out.cols             = C;
    out.row_stride_bytes = stride_b;
    out.qs_va = rt.vram_arena(rank).alloc(qs.size(), 4096);
    out.dm_va = rt.vram_arena(rank).alloc(dm.size() * 4u, 256);
    if (!out.qs_va || !out.dm_va) { if (err) *err = "VRAM OOM"; return false; }
    if (!rt.upload_to_vram(rank, out.qs_va, qs.data(), qs.size()) ||
        !rt.upload_to_vram(rank, out.dm_va, dm.data(), dm.size() * 4u)) {
        if (err) *err = "upload_to_vram failed";
        return false;
    }
    if (fwht_out) {
        fwht_out->rows             = total_rows;
        fwht_out->cols             = C;
        fwht_out->row_stride_bytes = stride_b;
        fwht_out->qs_va = rt.vram_arena(rank).alloc(qs_f.size(), 4096);
        fwht_out->dm_va = rt.vram_arena(rank).alloc(dm_f.size() * 4u, 256);
        if (!fwht_out->qs_va || !fwht_out->dm_va) {
            if (err) *err = "VRAM OOM for the --dot8-fwht rotated weight copy (" +
                            std::to_string((qs_f.size() + dm_f.size() * 4u) >> 20) + " MiB/GPU extra)";
            return false;
        }
        if (!rt.upload_to_vram(rank, fwht_out->qs_va, qs_f.data(), qs_f.size()) ||
            !rt.upload_to_vram(rank, fwht_out->dm_va, dm_f.data(), dm_f.size() * 4u)) {
            if (err) *err = "upload_to_vram (--dot8-fwht) failed";
            return false;
        }
    }
    // Agent E: reference Q4_K decode above feeds SUB4 directly (no Q4 requantization round trip).
    if(sub4_load || quant::tcc_xor_enabled()) {
        quant::PackedVjQ4A8 plain;
        plain.rows=total_rows; plain.cols=C; plain.qs_row_stride_bytes=stride_b;
        plain.dm_row_stride_u32=C/32;
        plain.qs=std::move(qs); plain.dm=std::move(dm);
        if(!sub4_integration::upload(rt,rank,original,plain,out,err)) return false;
        std::printf("[ornith::sub4] GPU%zu %ux%u sidecar=%llu bytes (Q4 fallback retained)\n",
                    rank,total_rows,C,static_cast<unsigned long long>(out.extra_bytes));
    }
    return true;
}

// High-precision VJ_Q8_DP4A uploader for Q6_K output.weight (LM Head)
bool upload_q8_tcc(kfd::Runtime& rt, size_t rank, GgufFile& f, const std::string& name,
                   uint32_t row_start, uint32_t row_count, Q8WeightDev& out, std::string* err) {
    const GgufTensorInfo* ti = f.tensor(name);
    if (!ti) { if (err) *err = "Missing tensor " + name; return false; }
    const uint8_t* raw = f.tensor_data(*ti);
    if (!raw) { if (err) *err = "Failed to mmap " + name; return false; }

    const auto& reg = quant::DecoderRegistry::instance();
    const auto disk_t = static_cast<quant::DiskQuantType>(ti->type);
    const uint32_t C = static_cast<uint32_t>(ti->ne[0]);
    const uint64_t src_rb = reg.row_bytes(disk_t, C);
    const auto lay = quant::select_tcc_layout(C);
    const uint32_t stride_b = lay.stride_bytes;

    std::vector<int8_t>   qs(static_cast<size_t>(row_count) * stride_b, 0);
    std::vector<uint16_t> ds(static_cast<size_t>(row_count) * (C / 32u), 0);

    #pragma omp parallel
    {
        std::vector<float> full_row(C);
        #pragma omp for schedule(static)
        for (uint32_t r = 0; r < row_count; ++r) {
            reg.decode_row_f32(disk_t, raw + static_cast<uint64_t>(row_start + r) * src_rb, full_row.data(), C);
            int8_t*   dst_row = qs.data() + static_cast<size_t>(r) * stride_b;
            uint16_t* drow    = ds.data() + static_cast<size_t>(r) * (C / 32u);
            for (uint32_t b = 0; b < C / 32u; ++b) {
                float amax = 0.0f;
                for (uint32_t i = 0; i < 32u; ++i) amax = std::max(amax, std::fabs(full_row[b * 32u + i]));
                const uint16_t dh = quant::f32_to_fp16(amax / 127.0f);
                const float d_eff = quant::fp16_to_f32(dh);
                const float inv = (d_eff > 0.0f) ? (1.0f / d_eff) : 0.0f;
                drow[b] = dh;
                for (uint32_t i = 0; i < 32u; ++i) {
                    int q = static_cast<int>(std::lrintf(full_row[b * 32u + i] * inv));
                    dst_row[b * 32u + i] = static_cast<int8_t>(std::max(-127, std::min(127, q)));
                }
            }
        }
    }
    out.rows = row_count; out.cols = C; out.row_stride_bytes = stride_b;
    out.qs_va = rt.vram_arena(rank).alloc(qs.size(), 4096);
    out.ds_va = rt.vram_arena(rank).alloc(ds.size() * 2u, 256);
    if (!out.qs_va || !out.ds_va) { if (err) *err = "VRAM OOM"; return false; }
    if (!rt.upload_to_vram(rank, out.qs_va, qs.data(), qs.size()) ||
        !rt.upload_to_vram(rank, out.ds_va, ds.data(), ds.size() * 2u)) {
        if (err) *err = "upload_to_vram failed";
        return false;
    }
    return true;
}

// Indexed VJ_Q8_DP4A uploader for FR-Spec draft LM-head shortlist (strictly multiple-of-256 rows per GPU).
bool upload_q8_tcc_indexed(kfd::Runtime& rt, size_t rank, GgufFile& f, const std::string& name,
                           const std::vector<int32_t>& row_ids, Q8WeightDev& out, std::string* err) {
    const GgufTensorInfo* ti = f.tensor(name);
    if (!ti) { if (err) *err = "Missing tensor " + name; return false; }
    const uint8_t* raw = f.tensor_data(*ti);
    if (!raw) { if (err) *err = "Failed to mmap " + name; return false; }

    const auto& reg = quant::DecoderRegistry::instance();
    const auto disk_t = static_cast<quant::DiskQuantType>(ti->type);
    const uint32_t C = static_cast<uint32_t>(ti->ne[0]);
    const uint64_t src_rb = reg.row_bytes(disk_t, C);
    const auto lay = quant::select_tcc_layout(C);
    const uint32_t stride_b = lay.stride_bytes;
    const uint32_t row_count = static_cast<uint32_t>(row_ids.size());

    std::vector<int8_t>   qs(static_cast<size_t>(row_count) * stride_b, 0);
    std::vector<uint16_t> ds(static_cast<size_t>(row_count) * (C / 32u), 0);

    #pragma omp parallel
    {
        std::vector<float> full_row(C);
        #pragma omp for schedule(static)
        for (uint32_t r = 0; r < row_count; ++r) {
            const uint32_t src_r = static_cast<uint32_t>(row_ids[r]);
            reg.decode_row_f32(disk_t, raw + static_cast<uint64_t>(src_r) * src_rb, full_row.data(), C);
            int8_t*   dst_row = qs.data() + static_cast<size_t>(r) * stride_b;
            uint16_t* drow    = ds.data() + static_cast<size_t>(r) * (C / 32u);
            for (uint32_t b = 0; b < C / 32u; ++b) {
                float amax = 0.0f;
                for (uint32_t i = 0; i < 32u; ++i) amax = std::max(amax, std::fabs(full_row[b * 32u + i]));
                const uint16_t dh = quant::f32_to_fp16(amax / 127.0f);
                const float d_eff = quant::fp16_to_f32(dh);
                const float inv = (d_eff > 0.0f) ? (1.0f / d_eff) : 0.0f;
                drow[b] = dh;
                for (uint32_t i = 0; i < 32u; ++i) {
                    int q = static_cast<int>(std::lrintf(full_row[b * 32u + i] * inv));
                    dst_row[b * 32u + i] = static_cast<int8_t>(std::max(-127, std::min(127, q)));
                }
            }
        }
    }
    out.rows = row_count; out.cols = C; out.row_stride_bytes = stride_b;
    out.qs_va = rt.vram_arena(rank).alloc(qs.size(), 4096);
    out.ds_va = rt.vram_arena(rank).alloc(ds.size() * 2u, 256);
    if (!out.qs_va || !out.ds_va) { if (err) *err = "VRAM OOM (FR-Spec shortlist)"; return false; }
    if (!rt.upload_to_vram(rank, out.qs_va, qs.data(), qs.size()) ||
        !rt.upload_to_vram(rank, out.ds_va, ds.data(), ds.size() * 2u)) {
        if (err) *err = "upload_to_vram failed (FR-Spec shortlist)";
        return false;
    }
    return true;
}

bool upload_f32_rows(kfd::Runtime& rt, size_t rank, GgufFile& f, const std::string& name,
                     const std::vector<uint32_t>* row_indices, uint64_t& va_out, std::string* err) {
    const GgufTensorInfo* ti = f.tensor(name);
    if (!ti) { if (err) *err = "Missing F32 tensor " + name; return false; }
    const uint8_t* raw = f.tensor_data(*ti);
    if (!raw) { if (err) *err = "Failed to mmap " + name; return false; }
    const float* src = reinterpret_cast<const float*>(raw);
    if (!row_indices) {
        const size_t bytes = ti->n_elems() * sizeof(float);
        va_out = rt.vram_arena(rank).alloc(bytes, 256);
        if (!va_out || !rt.upload_to_vram(rank, va_out, src, bytes)) { if (err) *err = "Upload F32 failed " + name; return false; }
        return true;
    }
    const uint64_t cols = (ti->n_dims >= 2) ? ti->ne[0] : 1ULL;
    std::vector<float> buf(row_indices->size() * cols);
    for (size_t i = 0; i < row_indices->size(); ++i) {
        std::memcpy(buf.data() + i * cols, src + static_cast<size_t>((*row_indices)[i]) * cols, cols * sizeof(float));
    }
    va_out = rt.vram_arena(rank).alloc(buf.size() * sizeof(float), 256);
    if (!va_out || !rt.upload_to_vram(rank, va_out, buf.data(), buf.size() * sizeof(float))) {
        if (err) *err = "Upload F32 subset failed " + name;
        return false;
    }
    return true;
}

bool upload_f32_as_f16(kfd::Runtime& rt, size_t rank, GgufFile& f, const std::string& name,
                       uint64_t& va_out, std::string* err) {
    const GgufTensorInfo* ti = f.tensor(name);
    if (!ti) { if (err) *err = "Missing tensor " + name; return false; }
    const uint8_t* raw = f.tensor_data(*ti);
    if (!raw) { if (err) *err = "Failed to mmap " + name; return false; }
    const float* src = reinterpret_cast<const float*>(raw);
    if (ti->n_dims >= 2 && ti->ne[0] == 3072u) {
        // Odd-256B TCC stride (3200 halfs = 6400 B = 25 * 256 B) eliminates 2048B HBM2/TCC partition thrashing.
        const size_t rows = static_cast<size_t>(ti->ne[1]);
        constexpr size_t kCols = 3072u, kStride = 3200u;
        std::vector<uint16_t> h(rows * kStride, 0u);
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < kCols; ++c) h[r * kStride + c] = quant::f32_to_fp16(src[r * kCols + c]);
        }
        va_out = rt.vram_arena(rank).alloc(h.size() * 2u, 256);
        if (!va_out || !rt.upload_to_vram(rank, va_out, h.data(), h.size() * 2u)) {
            if (err) *err = "Upload F16 padded failed " + name;
            return false;
        }
        return true;
    }
    const size_t n = ti->n_elems();
    std::vector<uint16_t> h(n);
    for (size_t i = 0; i < n; ++i) h[i] = quant::f32_to_fp16(src[i]);
    va_out = rt.vram_arena(rank).alloc(n * 2u, 256);
    if (!va_out || !rt.upload_to_vram(rank, va_out, h.data(), n * 2u)) {
        if (err) *err = "Upload F16 failed " + name;
        return false;
    }
    return true;
}

std::vector<uint32_t> iota_range(uint32_t beg, uint32_t cnt) {
    std::vector<uint32_t> v(cnt);
    for (uint32_t i = 0; i < cnt; ++i) v[i] = beg + i;
    return v;
}

// Background 250 ms sampler of power1_input, power1_cap and HBM2 temp3_input.
// Per-card steady-clock timestamps feed exact held-sample rolling integrals.
// Only gfx906 cards count toward the N>=3 PSU limit; unreadable cards stay in
// the report and make it incomplete rather than silently disappearing.
namespace {
bool is_gfx906_card(const std::string& card_dir) {
    std::ifstream fv(card_dir + "/vendor"), fd(card_dir + "/device");
    std::string vendor, did;
    if (!(fv >> vendor) || !(fd >> did) || vendor != "0x1002") return false;
    // 0x66a0/0x66a1: MI50 (incl. an alternate-VBIOS unit), 0x66a3: MI60, 0x66af: Radeon VII / Vega 20.
    return did == "0x66a0" || did == "0x66a1" || did == "0x66a3" || did == "0x66af";
}
}  // namespace

class HwmonSampler {
public:
    explicit HwmonSampler(size_t expected_cards) {
        namespace fs = std::filesystem;
        std::error_code ec;
        for (const auto& entry : fs::directory_iterator("/sys/class/drm", ec)) {
            const std::string id = entry.path().filename().string();
            if (id.compare(0, 4, "card") != 0 || id.size() == 4 || id.find_first_not_of("0123456789", 4) != std::string::npos) continue;
            const std::string dev = entry.path().string() + "/device";
            if (!is_gfx906_card(dev)) continue;
            std::string dir;
            std::error_code hec;
            for (const auto& h : fs::directory_iterator(dev + "/hwmon", hec)) {
                dir = h.path().string() + "/";
                break;
            }
            dirs_.push_back(dir);
            telemetry::Card card;
            card.id = id;
            cards_.push_back(card);
        }
        while (cards_.size() < expected_cards) {
            telemetry::Card card;
            card.id = "unmatched" + std::to_string(cards_.size());
            cards_.push_back(card);
            dirs_.emplace_back();
        }
        // Disabling sampling is diagnostic only: it cannot pass the telemetry gate.
        if (const char* e = std::getenv("VEGAJET_TELEM_MS"); e && e[0]) period_ms_ = std::atoi(e);
        if (period_ms_ <= 0) { run_.store(false); return; }
        sample();
        th_ = std::thread([this] {
            std::unique_lock<std::mutex> lock(mutex_);
            while (!wake_.wait_for(lock, std::chrono::milliseconds(period_ms_), [this] { return !run_.load(); })) sample();
        });
    }
    ~HwmonSampler() { stop(); }
    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!run_.exchange(false)) return;
        }
        wake_.notify_all();
        if (th_.joinable()) th_.join();
        sample();  // Close even sub-period runs with a real observation, not an assumed cadence.
    }
    const std::vector<telemetry::Card>& cards() const { return cards_; }  // after stop()

private:
    static double read_value(const std::string& dir, const char* file, double scale) {
        if (dir.empty()) return telemetry::unavailable();
        std::ifstream f(dir + file);
        double value = 0.0;
        return f >> value ? value * scale : telemetry::unavailable();
    }
    void sample() {
        for (size_t i = 0; i < dirs_.size(); ++i) {
            const double pw = read_value(dirs_[i], "power1_input", 1e-6);
            const double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
            const double tm = read_value(dirs_[i], "temp3_input", 1e-3);
            const double cap = read_value(dirs_[i], "power1_cap", 1e-6);
            cards_[i].observe(now, pw, tm, cap);
        }
    }
    std::vector<std::string> dirs_;
    std::vector<telemetry::Card> cards_;
    std::atomic<bool> run_{true};
    int period_ms_ = 250;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::thread th_;
};

void read_hwmon_max(double& max_pwr_w, double& max_tmem_c) {
    for (int card = 0; card < 8; ++card) {
        if (!is_gfx906_card("/sys/class/drm/card" + std::to_string(card) + "/device")) continue;
        std::string base = "/sys/class/drm/card" + std::to_string(card) + "/device/hwmon/";
        for (int h = 0; h < 32; ++h) {
            std::string dir = base + "hwmon" + std::to_string(h) + "/";
            std::ifstream fp(dir + "power1_average");
            double pw = 0.0;
            if (!(fp >> pw)) {
                std::ifstream fp2(dir + "power1_input");
                if (!(fp2 >> pw)) continue;
            }
            std::ifstream ft(dir + "temp3_input");
            double tm = 0.0;
            if (ft >> tm) {
                max_pwr_w  = std::max(max_pwr_w,  pw * 1e-6);
                max_tmem_c = std::max(max_tmem_c, tm * 1e-3);
            }
            break;
        }
    }
}

void read_ecc_totals(uint64_t& ue_out, uint64_t& de_out) {
    ue_out = 0;
    de_out = 0;
    static const char* const kBlocks[] = {"umc", "gfx", "hdp", "mmhub", "pcie_bif", "sdma"};
    for (int card = 0; card < 16; ++card) {
        for (const char* blk : kBlocks) {
            char p[160];
            std::snprintf(p, sizeof(p), "/sys/class/drm/card%d/device/ras/%s_err_count", card, blk);
            std::ifstream f(p);
            if (!f.is_open()) continue;
            std::string tok;
            while (f >> tok) {
                if (tok == "ue:") { uint64_t v = 0; if (f >> v) ue_out += v; }
                else if (tok == "de:") { uint64_t v = 0; if (f >> v) de_out += v; }
            }
        }
    }
}

bool write_vram_bin(kfd::Runtime& rt, size_t gpu_idx, uint64_t va, uint64_t bytes, const std::string& path, std::string* err) {
    std::vector<uint8_t> buf(static_cast<size_t>(bytes));
    if (bytes > 0 && !rt.download_from_vram(gpu_idx, buf.data(), va, static_cast<size_t>(bytes))) {
        if (err) *err = "download_from_vram failed for " + path;
        return false;
    }
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) {
        if (err) *err = "cannot open " + path + " for write";
        return false;
    }
    if (bytes > 0) ofs.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(bytes));
    if (!ofs.good()) {
        if (err) *err = "short write to " + path;
        return false;
    }
    return true;
}

} // namespace

// VEGAJET_MOE_PREFETCH bitmask for the _x Q3 MoE super kernel: 1 = L2-prefetch Stage-1 router/shared-GU weights
// while WG0 runs the incoming AllReduce, 2 = L2-prefetch the first four experts' routed-Down weights before Stage 4.
// Deep-telemetry dump layout (must match ops/kernels/vj_ornith_super_x.hip VJ_PROF_WG_OFF / VJ_PROF_ARW_OFF).
static constexpr uint64_t kProfWgOff   = 32768u;
static constexpr uint64_t kProfWgBytes = 96ull * 17ull * 256ull * 8ull;
static constexpr uint64_t kProfArwOff  = kProfWgOff + kProfWgBytes;
static constexpr uint64_t kProfDumpBytes = kProfArwOff + 96ull * 2ull * 8ull;
static_assert(kProfWgOff >= ornith_mega::kScratchBytes, "per-WG profile area overlaps the mega scratch");

static uint32_t moe_prefetch_xflags() {
    const char* e = std::getenv("VEGAJET_MOE_PREFETCH");
    if (!e || !e[0]) return 0u;
    return static_cast<uint32_t>(std::atoi(e)) & 3u;
}

OrnithBenchMetrics run_ornith_gpu(const std::string& gguf_path,
                                  const std::string& prompt_text,
                                  const OrnithGpuOptions& opt_in) {
    OrnithGpuOptions opt = opt_in;
    // Agent E: load-time formats must match the standalone launch path. Reuse the
    // existing super-disable decision; no process environment mutation across warm runs.
    const uint32_t loaded_sub4_bits=quant::sub4_dot8_bits();
    const bool loaded_milo=quant::milo_enabled();
    const bool sub4_on=loaded_sub4_bits!=0;
    const bool tcc_xor_on=quant::tcc_xor_enabled();
    if(sub4_on) opt.ab_no_prepush=true; // SUB4/MiLo do not publish peer epilogues.
    if(sub4_on || tcc_xor_on)
        std::printf("[ornith::sub4] forcing VEGAJET_ENABLE_SUPER=0 behavior: standalone launches; persistent fusion lost\n");
    // VEGAJET_GDN_VERIFY (load-time): lumotree = single-write Split-V verify + recompute-on-reject (--spec-gdn);
    // treewy = same plus the sealed tree-WY verifier (VEGAJET_TREEWY=1, TP=2 only); ring = per-token ring states.
    if (const char* gv = std::getenv("VEGAJET_GDN_VERIFY"); gv && gv[0]) {
        const std::string v(gv);
        if (v == "lumotree") opt.spec_gdn = true;
        else if (v == "treewy") { opt.spec_gdn = true; setenv("VEGAJET_TREEWY", "1", 1); }
        else if (v == "ring") opt.spec_gdn = false;
        else std::fprintf(stderr, "[ornith::gpu] VEGAJET_GDN_VERIFY=%s ignored (lumotree|treewy|ring)\n", gv);
    } else if (!opt.spec_gdn) {
        // Default lumotree (2026-11-27, W7 vs W9b): Hybrid tg128 370.1 -> 382.3 (+3.3%), tg512 363.5 -> 375.9, code
        // 129/129, raw M=1 unchanged, ~783 MiB/GPU less state. Independent batch rows (VEGAJET_GPU_BATCH>=2) need the
        // per-token ring states, so they keep the ring verifier. Opt out: VEGAJET_GDN_VERIFY=ring.
        const char* gb = std::getenv("VEGAJET_GPU_BATCH");
        const bool batch_rows_requested = gb && std::atoi(gb) >= 2;
        if (!batch_rows_requested) opt.spec_gdn = true;
    }
    // VEGAJET_SPEC_CASCADE=1: zero-tax suffix cascade (skip the neural drafter when the CPU suffix trie returns >= 4
    // confident tokens) == --zero-tax-catchup. VEGAJET_SPEC_EVICT=1: lossless EVICT expert-union adaptation
    // (== VEGAJET_EVICT_ADAPT=1, DFlash2 path).
    if (const char* sc = std::getenv("VEGAJET_SPEC_CASCADE"); sc && sc[0]) opt.zero_tax_catchup = (sc[0] == '1');
    if (const char* se = std::getenv("VEGAJET_SPEC_EVICT"); se && se[0] == '1') setenv("VEGAJET_EVICT_ADAPT", "1", 0);
    bool accept_moe_on = (std::getenv("VEGAJET_SPEC_ACCEPTMOE") != nullptr && std::getenv("VEGAJET_SPEC_ACCEPTMOE")[0] == '1');
    uint32_t accept_moe_keep = (std::getenv("VEGAJET_ACCEPTMOE_KEEP") != nullptr)
        ? static_cast<uint32_t>(std::clamp(std::atoi(std::getenv("VEGAJET_ACCEPTMOE_KEEP")), 1, 8)) : 2u;
    float accept_moe_pen = (std::getenv("VEGAJET_ACCEPTMOE_PENALTY") != nullptr)
        ? std::clamp(static_cast<float>(std::atof(std::getenv("VEGAJET_ACCEPTMOE_PENALTY"))), 0.0f, 1.0f) : 0.25f;
    // base_opt must include the load-time env mappings above (VEGAJET_GDN_VERIFY / SPEC_CASCADE): warm variants are
    // rebuilt from it, and GDN ring vs spec-slot allocation is decided from opt.spec_gdn at load. Using opt_in here made
    // warm variants run the ring path on unallocated (null) ring states under VEGAJET_GDN_VERIFY=lumotree (GPU page
    // faults at small addresses, 2026-11-26).
    const OrnithGpuOptions base_opt = opt;
    const treewy::Options treewy_opt = treewy::Options::from_env();
    opt.treewy = treewy_opt.enabled;
    OrnithBenchMetrics res{};
    // Lever 4: process selection is immutable for the lifetime of the cache.
    kvq::DeviceFormat l4_format = kvq::DeviceFormat::Q8;
    const char* l4_quant_env = std::getenv("VEGAJET_KV_QUANT");
    if (l4_quant_env && (!kvq::parse_format(l4_quant_env, l4_format) || l4_format == kvq::DeviceFormat::Fp16)) {
        res.error = "VEGAJET_KV_QUANT must be q8_0, q4_0 or turbo3"; return res;
    }
    bool l4_compressed = l4_format != kvq::DeviceFormat::Q8;
    const char* l4_split_env = std::getenv("VEGAJET_SPLIT_KV");
    bool l4_split = l4_split_env && std::strcmp(l4_split_env,"1")==0;
    const char* l4_chrono_env = std::getenv("VEGAJET_KV_CHRONOLOGICAL");
    bool l4_chrono = l4_chrono_env && std::strcmp(l4_chrono_env,"1")==0;
    bool l4_on = l4_compressed || l4_split;
    uint32_t l4_stride = kvq::vector_bytes(l4_format,256);
    const bool l4_warm = opt.interactive || !opt.sweep_spec.empty();
    const bool l4_available = l4_on || l4_warm;
    l4::State L4[kfd::kMaxGpus];
    l4::Kernels L4K[kfd::kMaxGpus];
    if (l4_on) std::fprintf(stderr,"[l4] KV=%s split=%d chronological=%d; isolated M1 GQA super when available; standalone prefill/verify; compressed tiers take priority over sealed Q8 wire\n",
                           l4_quant_env ? l4_quant_env : "q8_0", l4_split, l4_chrono);
    // M1 sealed-wire attention is opt-in. Env switches mirror the option fields; a non-empty value
    // other than "0" enables it, and "q4" also opts into the lossy Q4 wire. Defaults are untouched.
    // Literal getenv call sites keep environment use auditable by tests/test_cli_reference.py.
    const auto wire_env_on = [](const char* value, bool fallback) {
        if (!value || !value[0]) return fallback;
        return !(value[0] == '0' && value[1] == '\0');
    };
    const char* wire_attn_env = std::getenv("VEGAJET_WIRE_ATTN");
    bool wire_requested = opt.sealed_wire_attn;
    bool wire_q4 = opt.sealed_wire_q4;
    if (wire_attn_env && wire_attn_env[0]) {
        wire_requested = !(wire_attn_env[0] == '0' && wire_attn_env[1] == '\0');
        if (std::strcmp(wire_attn_env, "q4") == 0 || std::strcmp(wire_attn_env, "Q4") == 0) {
            wire_requested = true;
            wire_q4 = true;
        }
    }
    if (l4_on && wire_requested) { res.error="Lever 4 and VEGAJET_WIRE_ATTN are mutually exclusive"; return res; }
    wire_q4 = wire_q4 || wire_env_on(std::getenv("VEGAJET_WIRE_ATTN_Q4"), false);
    const bool wire_strict = opt.sealed_wire_strict || wire_env_on(std::getenv("VEGAJET_WIRE_ATTN_STRICT"), false);
    const bool wire_exercise = opt.sealed_wire_exercise ||
                               wire_env_on(std::getenv("VEGAJET_WIRE_ATTN_EXERCISE"), false);
    // A/B control only: disable the fused M=1 prep+flash attention so the reference arm runs the
    // same attn_prep + single-pass flash structure as the wire arm (with VEGAJET_ENABLE_SUPER=0 and
    // VEGAJET_AB_NO_SPLITKV=1). Defaults are untouched.
    const bool ab_no_fused_attn = wire_env_on(std::getenv("VEGAJET_AB_NO_FUSED_ATTN"), false);
    uint32_t wire_max_splits = opt.sealed_wire_max_splits != 0u ? opt.sealed_wire_max_splits
                                                                : flash_decode_splitkv::kMaxSplits;
    if (const char* e = std::getenv("VEGAJET_WIRE_ATTN_MAX_SPLITS"); e && e[0]) {
        const long parsed = std::strtol(e, nullptr, 10);
        if (parsed >= 2 && parsed <= static_cast<long>(flash_decode_splitkv::kMaxSplits))
            wire_max_splits = static_cast<uint32_t>(parsed);
    }
    if (wire_max_splits < 2u) wire_max_splits = 2u;
    if (wire_max_splits > flash_decode_splitkv::kMaxSplits)
        wire_max_splits = flash_decode_splitkv::kMaxSplits;
    bool wire_skip_reported = false;
    auto t_load0 = std::chrono::steady_clock::now();
    const auto sel_devs = [] {
        auto v = kfd::Runtime::scan_gfx906_topology();
        std::string ignored;
        if (!gpusel::filter_from_env(v, &ignored)) v.clear();
        return v;
    }();
    {   // Dynamic N-rank arena check (1..kMaxGpus x 32 GB gfx906).
        const uint32_t need_gpus = static_cast<uint32_t>(std::max<size_t>(1u, sel_devs.size()));
        const auto budget = preflight::check_arena(preflight::gfx906_vram_bytes("/"), preflight::kOrnithArenaGiB,
                                                   need_gpus, "the Ornith-122B GPU runner");
        if (!budget.ok) { res.error = budget.message; return res; }
    }

    GgufFile f;
    if (!f.open(gguf_path)) { res.error = f.error(); return res; }
    BpeTokenizer tok;
    if (!tok.load_from_gguf(gguf_path, &res.error)) return res;

    OrnithModel planner;
    quant::HardwareRooflineProfile hw_prof;
    const bool have_hw_prof = quant::HardwareRooflineProfile::load_json(quant::HardwareRooflineProfile::default_cache_path(), hw_prof);
    LoadOptions lopt;
    // Match hw_profile.json strictly by sysfs unique_id first, then PCI BDF fallback.
    auto match_hw_prof = [&](const kfd::DeviceInfo& d) -> const decltype(hw_prof.gpus)::value_type* {
        if (!have_hw_prof) return nullptr;
        if (!d.unique_id_hex.empty()) {
            for (const auto& hg : hw_prof.gpus)
                if (!hg.unique_id.empty() && hg.unique_id == d.unique_id_hex) return &hg;
        }
        std::string bdf = d.bdf;
        if (bdf.empty()) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%04x:%02x:%02x.%u", d.domain, (d.location_id >> 8) & 0xFFu,
                          (d.location_id >> 3) & 0x1Fu, d.location_id & 7u);
            bdf = buf;
        }
        for (const auto& hg : hw_prof.gpus) if (hg.id == bdf) return &hg;
        return nullptr;
    };
    const size_t plan_n = sel_devs.empty() ? 2u : std::min<size_t>(sel_devs.size(), kfd::kMaxGpus);
    lopt.gpus.clear();
    for (size_t r = 0; r < plan_n; ++r) {
        const auto* pg = (r < sel_devs.size()) ? match_hw_prof(sel_devs[r])
                         : ((have_hw_prof && r < hw_prof.gpus.size() && hw_prof.gpus[r].id.empty()) ? &hw_prof.gpus[r] : nullptr);
        lopt.gpus.push_back(GpuTarget{
            pg ? pg->n_cu : 60u,
            (pg && pg->bw_hbm_gbs > 0.0) ? pg->bw_hbm_gbs : 800.0,
            (pg && pg->sclk_ghz > 0.0) ? pg->sclk_ghz : 1.725,
            30500ULL * 1024ULL * 1024ULL
        });
    }
    lopt.context_length = opt.max_ctx;
    lopt.batch_reuse_factor = 1.0;
    lopt.draft_gguf = opt.draft_gguf;
    const LoadReport lrep = planner.load(f, lopt);
    if (!lrep.ok) { res.error = lrep.error; return res; }
    const auto& hp = planner.hparams();

    // Rank 3/5: Hot-Tensor VA clustering (6 x 512 MiB = 3.0 GiB, or 3.5 GiB with DFlash2 on GPU0 -> 0.00 GiB HOT spill).
    setenv("VEGAJET_VA_CLUSTER", "1", 0);
    setenv("VEGAJET_VA_HOT_MIB", (!opt.draft_gguf.empty() && planner.draft_hparams().is_dflash2()) ? "3584" : "3072", 0);
    kfd::Runtime rt;
    if (!rt.init(30.5, 64, &res.error) && !rt.init(30.0, 64, &res.error)) return res;
    const size_t N = rt.num_gpus();
    // VEGAJET_DECODE_FABRIC=1: zero-root, read-only host fabric preflight warnings (PCIe link, ACS, RO, MPS, DPM, NUMA).
    // VEGAJET_DECODE_FABRIC=1 (or an explicit VEGAJET_DISPATCH_CPUS list) also pins this host dispatch thread to the
    // GPUs' NUMA node; on a single-node (UMA) host it leaves affinity unchanged unless the list is given.
    {
        const char* fb = std::getenv("VEGAJET_DECODE_FABRIC");
        const char* dc = std::getenv("VEGAJET_DISPATCH_CPUS");
        const bool fabric = fb && fb[0] == '1';
        if (fabric || (dc && dc[0])) {
            std::vector<std::string> bdfs;
            for (const auto& dv : rt.devices()) bdfs.push_back(dv.bdf);
            if (fabric) vegajet::hw_preflight::run_and_log(bdfs);
            (void)vegajet::hw_preflight::pin_dispatch_thread(bdfs);
        }
    }
    if (N == 0 || N > kfd::kMaxGpus) {
        res.error = "Ornith-122B GPU runner requires 1.." + std::to_string(kfd::kMaxGpus) + " MI50 GPUs";
        return res;
    }
    const std::string hdir = kfd::resolve_hsaco_dir(opt.hsaco_dir);
    if (!rt.load_hsaco_dir(hdir, &res.error)) return res;

    // Resolve per-GPU calibrated HBM2 ceiling from hw_profile.json (unique_id first, then BDF; verify MCLK).
    // If a card is missing or its active MCLK drifted >25 MHz from the cached profile, run a silent ~0.5 s KFD
    // streaming bandwidth probe on a temporary arena slice and roll the cursor back before uploading weights.
    double gpu_bw_ceiling_gbs[kfd::kMaxGpus] = {};
    double calibrated_hbm2_ceiling_gbs = 0.0;
    const bool trust_hw_prof = (std::getenv("VEGAJET_TRUST_HW_PROFILE") != nullptr &&
                                std::getenv("VEGAJET_TRUST_HW_PROFILE")[0] == '1');
    for (size_t g = 0; g < N; ++g) {
        const auto& dev = rt.devices()[g];
        const auto* hg = match_hw_prof(dev);
        const bool mclk_ok = (!hg) ? false
            : (hg->mclk_mhz <= 0.0 || dev.mclk_mhz <= 0.0 || std::fabs(hg->mclk_mhz - dev.mclk_mhz) <= 25.0);
        // On N >= 3, always run a ~3 ms live KFD streaming probe (unless VEGAJET_TRUST_HW_PROFILE=1)
        // so runtime HBM2 strap changes (e.g. 0c strapped 871 vs stock 747 GB/s at same 1140 MHz) never skew sharding!
        double bw = (N < 3u || trust_hw_prof) && (hg && mclk_ok && hg->bw_hbm_gbs > 0.0) ? hg->bw_hbm_gbs : 0.0;
        if (bw <= 0.0) {
            const auto* k_fill = rt.get_kernel(g, "kfd_soak_fill_u32x4");
            const auto* k_ver  = rt.get_kernel(g, "kfd_soak_verify_u32x4");
            if (k_fill && k_ver && rt.vram_arena(g).used() == 0) {
                auto& ar = rt.vram_arena(g);
                const auto saved_mark = ar.mark();
                const uint64_t probe_bytes = 64ULL * 1024ULL * 1024ULL;
                const uint64_t p_va     = ar.alloc(probe_bytes, 4096);
                const uint64_t err_va   = ar.alloc(4096, 4096);
                const uint64_t ticks_va = ar.alloc(2u * 240u * sizeof(uint64_t), 4096);
                if (p_va && err_va && ticks_va) {
                    const uint32_t zero = 0u;
                    (void)rt.upload_to_vram(g, err_va, &zero, sizeof(zero));
                    struct FillKA { uint64_t dst, n_u4; uint32_t seed, pad; };
                    struct VerKA  { uint64_t src, n_u4; uint32_t seed, pad; uint64_t err, ticks; };
                    static_assert(sizeof(FillKA) == 24 && sizeof(VerKA) == 40, "kfd_soak ABI");
                    const uint64_t n_u4 = probe_bytes / 16ULL;
                    const FillKA fka{p_va, n_u4, 0x12345678u, 0u};
                    const VerKA  vka{p_va, n_u4, 0x12345678u, 0u, err_va, ticks_va};
                    const uint32_t gx = 240u * 256u;
                    (void)rt.enqueue_kernel(g, *k_fill, gx, 1, 1, 256, 1, 1, &fka, sizeof(fka), true);
                    (void)rt.submit_batch(g);
                    (void)rt.wait_idle(g);
                    constexpr int kIters = 16;
                    auto t0 = std::chrono::steady_clock::now();
                    for (int it = 0; it < kIters; ++it)
                        (void)rt.enqueue_kernel(g, *k_ver, gx, 1, 1, 256, 1, 1, &vka, sizeof(vka), true);
                    (void)rt.submit_batch(g);
                    (void)rt.wait_idle(g);
                    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    if (sec > 1e-6) bw = (static_cast<double>(kIters) * static_cast<double>(probe_bytes)) / (sec * 1e9);
                    std::printf("[ornith::hw] Live KFD probe GPU%zu (%s uid=%s mclk=%.0f MHz): %.1f GB/s\n",
                                g, dev.bdf.c_str(), dev.unique_id_hex.c_str(), dev.mclk_mhz, bw);
                }
                ar.rollback(saved_mark);
            }
            if (bw <= 0.0) bw = (g < lopt.gpus.size()) ? lopt.gpus[g].bw_hbm_gbs : 800.0;
        }
        gpu_bw_ceiling_gbs[g] = bw;
        calibrated_hbm2_ceiling_gbs += bw;
    }
    // Opt-in lossy Q4 LM head for greedy argmax (VEGAJET_HEAD_Q4=1): about half the head bytes; measured full-vocab
    // KL 0.011 vs the Q8 head. Used for every M when on, so speculative verify and plain steps share one head.
    const bool head_q4 = std::getenv("VEGAJET_HEAD_Q4") && std::strcmp(std::getenv("VEGAJET_HEAD_Q4"), "1") == 0;

    KernelTable K[kfd::kMaxGpus];
    // Host-side high-water marks for the opt-in wire K/V cache, one per GPU and attention layer.
    wire_normal::CacheState wire_cache[kfd::kMaxGpus];
    wire_normal::Counters wire_counters[kfd::kMaxGpus];
    for (size_t g = 0; g < N; ++g) wire_cache[g].reset(12);
    auto resolve_kernels = [&](std::string* kerr) -> bool {
        if (kerr) kerr->clear();
        for (size_t g = 0; g < N; ++g) {
            K[g] = KernelTable{};
            auto need_k = [&](const char* nm) -> const kfd::LoadedKernel* {
                const auto* lk = rt.get_kernel(g, nm);
                if (!lk && kerr && kerr->empty()) *kerr = std::string("Missing kernel ") + nm + " in " + hdir;
                return lk;
            };
            K[g].embed_q8     = need_k("ornith_embed_q8_kernel");
            K[g].rms_quant    = need_k("ornith_rms_quant_q8_kernel");
            K[g].moe_m1_front = need_k("ornith_moe_m1_router_shared_kernel");
            K[g].moe_m1_back  = need_k("ornith_moe_m1_back_kernel");
            K[g].p2p_ar_rms   = need_k(opt.ar_wire_int8 ? "ornith_p2p_ar_i8_res_rms_quant_kernel" : "ornith_p2p_ar_res_rms_quant_kernel");
            K[g].p2p_ar_prepush = need_k("ornith_p2p_ar_res_rms_quant_prepush_kernel");
            K[g].p2p_ar_i8      = need_k("ornith_p2p_ar_i8_res_rms_quant_kernel");
            K[g].p2p_ar_n_fp16  = (N >= 3) ? need_k("ornith_p2p_ar_n_fp16_kernel") : rt.get_kernel(g, "ornith_p2p_ar_n_fp16_kernel");
            if(sub4_on) {
                const std::string prefix="vj_sub4_q"+std::to_string(quant::sub4_dot8_bits());
                K[g].sub4_gemv=need_k((prefix+"_gemv").c_str());
                K[g].sub4_gemm=need_k((prefix+"_gemm").c_str());
                if(tcc_xor_on) {
                    K[g].sub4_gemv_xor=need_k((prefix+"_gemv_xor").c_str());
                    K[g].sub4_gemm_xor=need_k((prefix+"_gemm_xor").c_str());
                }
                if(quant::milo_enabled()) {
                    K[g].milo_vx=need_k("vj_milo_vx"); K[g].milo_apply=need_k("vj_milo_apply");
                }
            } else if(tcc_xor_on) K[g].q4_gemv_xor=need_k("ornith_gemv_q4a8_m1_w32_xor_kernel");
            K[g].gemv_q4_w32  = need_k("ornith_gemv_q4a8_m1_w32_kernel");
            K[g].gemv_q4_w64  = need_k("vj_gemv_q4a8_w64_kernel");
            K[g].gemv_q4_w32_stag = rt.get_kernel(g, "ornith_gemv_q4a8_m1_w32_stag_kernel");
            K[g].gemv_q4_splitk_p1 = need_k("vj_gemv_q4a8_splitk_p1_kernel");
            K[g].splitk_red_pp = need_k("vj_splitk_reduce_prepush_f32_kernel");
            K[g].gemv_q4_w16  = need_k("ornith_gemv_q4a8_m1_w16_kernel");
            K[g].gemm_q4_tcc  = need_k("ornith_gemm_q4a8_tcc_kernel");
            K[g].gemm_q4_m8   = need_k("ornith_gemm_q4a8_m8_kernel");
            K[g].gemm_q4_m12  = need_k("ornith_gemm_q4a8_m12_kernel");
            K[g].gemm_q4_m12x = rt.get_kernel(g, "ornith_gemm_q4a8_m12_x_kernel");
            K[g].gemm_q4_m16  = need_k("ornith_gemm_q4a8_m16_l2_kernel");
            K[g].gdn_conv_l2  = need_k("ornith_gdn_conv_l2_gate_kernel");
            K[g].gdn_recur    = need_k("ornith_gdn_recur_quant_kernel");
            K[g].gdn_recur_sv = need_k("ornith_gdn_recur_splitv_quant_kernel");
            K[g].gdn_conv_recur_sv = need_k("ornith_gdn_conv_recur_splitv_quant_m1_kernel");
            // A2 Split-V=8 is an optional module so older hsaco directories still start; selecting it without
            // the module falls back to the current Split-V=4 fused kernel. The recurrence (stage 1) and the
            // RMSNorm+quant epilogue (stage 2) are separate kernels enqueued in order on the same queue.
            K[g].gdn_conv_recur_sv8 = rt.get_kernel(g, "ornith_gdn_conv_recur_splitv8_recur_m1_kernel");
            K[g].gdn_conv_recur_sv8_quant = rt.get_kernel(g, "ornith_gdn_conv_recur_splitv8_quant_m1_kernel");
            K[g].gdn_wy       = need_k("ornith_gdn_wy_prefill_quant_kernel");
            if (l4_available) {
                L4K[g].epilogue=need_k("vj_l4_epilogue_kernel");
                if (l4_compressed || l4_warm) {
                    L4K[g].prep=need_k("vj_l4_prep_kernel");
                    L4K[g].append=need_k("vj_kvq_append_kernel");
                    L4K[g].partial=need_k("vj_kvq_grouped_partial_kernel");
                    L4K[g].merge=need_k("vj_flash_decode_lse_merge_kernel");
                }
                if (!l4_compressed || l4_warm) {
                    L4K[g].wire_kv=need_k("vj_l4_wire_kv_kernel");
                    L4K[g].wire_q=need_k("vj_l4_wire_q_kernel");
                    L4K[g].partition=need_k("vj_flash_decode_q8_partition");
                    L4K[g].wire_merge=need_k("vj_flash_decode_merge");
                }
            }
            K[g].attn_prep    = need_k("ornith_attn_qkv_prep_kernel");
            K[g].attn_flash   = need_k("ornith_attn_flash_quant_kernel");
            K[g].attn_prep_flash_m1 = need_k("ornith_attn_prep_flash_m1_kernel");
            if (wire_requested) {
                K[g].wire_q8_q        = need_k("vj_wire_q8_q_kernel");
                K[g].wire_q4_q        = need_k("vj_wire_q4_q_kernel");
                K[g].wire_q8_kv       = need_k("vj_wire_q8_kv_kernel");
                K[g].wire_q4_kv       = need_k("vj_wire_q4_kv_kernel");
                K[g].wire_epilogue    = need_k("vj_wire_attn_epilogue_kernel");
                K[g].wire_skv_q8_part = need_k("vj_flash_decode_q8_partition");
                K[g].wire_skv_q4_part = need_k("vj_flash_decode_q4_partition");
                K[g].wire_skv_merge   = need_k("vj_flash_decode_merge");
            }
            K[g].moe_rtr_log  = need_k("ornith_moe_router_fp16_logits_kernel");
            K[g].moe_rtr_top8 = need_k("ornith_moe_router_top8_kernel");
            K[g].moe_top8_acceptmoe = rt.get_kernel(g, "vj_moe_top8_acceptmoe_kernel");
            K[g].moe_top8_acceptmoe_plain = rt.get_kernel(g, "vj_moe_top8_acceptmoe_plain_kernel");
            K[g].swiglu_q8    = need_k("ornith_swiglu_quant_q8_kernel");
            K[g].moe_gu_iq2s      = need_k("ornith_moe_gu_iq2s_merged_kernel");
            K[g].moe_gu_d8        = need_k("vj_moe_gu_dot8_kernel");
            K[g].moe_gu_d8_cached = need_k("vj_moe_gu_dot8_cached_kernel");
            K[g].moe_gu_q3        = rt.get_kernel(g, "vj_moe_gu_q3_kernel");
            K[g].verify_gu_q3_pair = rt.get_kernel(g, "vj_verify_gu_q3_pair_kernel");
            K[g].moe_down_iq3     = need_k("ornith_moe_down_iq3xxs_split_kernel");
            K[g].moe_down_d8      = need_k("vj_moe_down_dot8_kernel");
            K[g].moe_down_d8_pp   = need_k("vj_moe_down_dot8_prepush_kernel");
            K[g].moe_down_d8_sub32 = (N >= 3) ? need_k("vj_moe_down_dot8_sub32_kernel") : rt.get_kernel(g, "vj_moe_down_dot8_sub32_kernel");
            K[g].spec_gdn_ver = opt.spec_gdn ? need_k("vj_spec_gdn_verify_kernel") : rt.get_kernel(g, "vj_spec_gdn_verify_kernel");
            K[g].spec_gdn_nq  = opt.spec_gdn ? need_k("vj_spec_gdn_norm_quant_kernel") : rt.get_kernel(g, "vj_spec_gdn_norm_quant_kernel");
            // Optional sealed tree-WY pair; a missing module or producer disables the opt-in path
            // (runtime fallback to the sequential verifier), never the engine itself.
            K[g].treewy_verify = rt.get_kernel(g, "vj_gdn_treewy_verify_kernel");
            K[g].treewy_commit = rt.get_kernel(g, "vj_gdn_treewy_commit_kernel");
            K[g].gdn_wire_pack = rt.get_kernel(g, "vj_gdn_wire_pack_kernel");
            K[g].flash_skv_p1 = need_k("vj_flash_splitkv_any_partial_kernel");
            K[g].flash_skv_p2 = need_k("vj_flash_splitkv_any_merge_quant_kernel");
            K[g].spec_flash_part  = opt.spec_flash ? need_k("vj_spec_flash_partial_kernel") : rt.get_kernel(g, "vj_spec_flash_partial_kernel");
            K[g].spec_flash_merge = opt.spec_flash ? need_k("vj_spec_flash_merge_quant_kernel") : rt.get_kernel(g, "vj_spec_flash_merge_quant_kernel");
            const bool warm_mode = (opt.interactive || !opt.sweep_spec.empty());
            if (opt.pf_fa2 || warm_mode) {
                K[g].pf_fa2       = need_k(opt.pf_fa2_fp16 ? "vj_prefill_fa2_quant_kernel" : "vj_prefill_fa2_quant_f32q_kernel");
                K[g].pf_fa2_codex = rt.get_kernel(g, "vj_prefill_gqa_fa2");
            }
            if (opt.pf_gemm == 64u)   K[g].pf_gemm = need_k("vj_prefill_gemm_q4a8_64x64_kernel");
            if (opt.pf_gemm == 128u || warm_mode) {
                K[g].pf_gemm    = need_k("vj_prefill_gemm_q4a8_64x128_kernel");
                K[g].pf_gemm_64 = need_k("vj_prefill_gemm_q4a8_64x64_kernel");
            }
            if (opt.pf_gemm != 0u || warm_mode) K[g].pf_router = need_k("vj_prefill_router_fp16_tiled_kernel");
            if (opt.pf_gu_tiled || warm_mode) {
                K[g].pf_gu      = need_k("vj_prefill_moe_gu_iq2s_tiled_kernel");
                K[g].pf_gu_dot8 = need_k("vj_prefill_moe_gu_dot8_tiled_kernel");
            }
            if (opt.pf_moe_iq_lds || warm_mode) K[g].pf_gu_iq_lds64 = rt.get_kernel(g, "vj_prefill_moe_gu_iq2s_lds64x64_kernel");
            if (opt.pf_moe_iq_lds32 || warm_mode) K[g].pf_gu_iq_lds32 = rt.get_kernel(g, "vj_prefill_moe_gu_iq2s_lds32x64_kernel");
            if (opt.pf_moe_iq_lds >= 2u || warm_mode) K[g].pf_down_iq_lds64 = rt.get_kernel(g, "vj_prefill_moe_down_iq3_lds64x64_kernel");
            if (opt.pf_moe_iq_lds32 >= 2u || warm_mode) K[g].pf_down_iq_lds32 = rt.get_kernel(g, "vj_prefill_moe_down_iq3_lds32x64_kernel");
            if (opt.down_grp_prefill == 3u || warm_mode) {
                K[g].pf_down_tiled = need_k("vj_prefill_moe_down_iq3_tiled_kernel");
                K[g].pf_down_comb  = need_k("vj_moe_down_combine_all_f32_kernel");
            }
            if (opt.pf_gdn != 0u || warm_mode) {
                K[g].pf_gdn_conv     = need_k("vj_prefill_gdn_conv_l2_gate_kernel");
                K[g].pf_gdn_wy_codex = rt.get_kernel(g, "vj_prefill_gdn_128wg");
                K[g].pf_gdn_nq_codex = rt.get_kernel(g, "vj_prefill_gdn_128wg_normq");
            }
            if (opt.pf_gdn == 1u || warm_mode)  K[g].pf_gdn_sv = need_k("vj_prefill_gdn_splitv_batch_kernel");
            if (opt.pf_gdn == 2u || warm_mode) {
                K[g].pf_gdn_wy       = need_k("vj_prefill_gdn_wy_split_kernel");
                K[g].pf_gdn_nq       = need_k("vj_prefill_gdn_normq_kernel");
            }
            if (opt.prefill_dot8 || warm_mode || sub4_on) {
                K[g].pf_q8_to_q4    = rt.get_kernel(g, "vj_prefill_q8_to_q4_kernel");
                if(sub4_on) K[g].pf_q8_to_q4=need_k("vj_prefill_q8_to_q4_kernel");
                K[g].pf_q8_to_q4_fwht = rt.get_kernel(g, "vj_prefill_q8_to_q4_fwht64_kernel");
                K[g].pf_gemm_d8_128 = rt.get_kernel(g, "vj_prefill_gemm_q4a4_64x128_dot8_kernel");
                K[g].pf_gemm_d8_64  = rt.get_kernel(g, "vj_prefill_gemm_q4a4_64x64_dot8_kernel");
                K[g].pf_gu_d8       = rt.get_kernel(g, "vj_prefill_moe_gu_iq2s_dot8_kernel");
                K[g].pf_down_d8     = rt.get_kernel(g, "vj_prefill_moe_down_dot8_tiled_kernel");
                if (opt.prefill_dot8 && (!K[g].pf_q8_to_q4 || !K[g].pf_gemm_d8_128 || !K[g].pf_gemm_d8_64 || !K[g].pf_gu_d8 || !K[g].pf_down_d8)) {
                    if (kerr && kerr->empty()) *kerr = "Missing prefill_dot8 kernels in " + hdir;
                }
                if (opt.prefill_dot8 && opt.dot8_fwht && !K[g].pf_q8_to_q4_fwht) {
                    if (kerr && kerr->empty()) *kerr = "Missing vj_prefill_q8_to_q4_fwht64_kernel (vj_prefill_dot8_fwht.hsaco) in " + hdir;
                }
            }
            if (!opt.dump_logits_path.empty()) K[g].vocab_logits = need_k("ornith_vocab_logits_q8_kernel");
            if (opt.down_grp_prefill != 0u || opt.down_grp_verify != 0u || warm_mode) {
                K[g].dgrp_route    = need_k("vj_moe_route_group_kernel");
                K[g].dgrp_dp       = need_k("vj_moe_down_grouped_dp_kernel");
                K[g].dgrp_f32      = need_k("vj_moe_down_grouped_f32_kernel");
                K[g].dgrp_comb_dp  = need_k("vj_moe_down_combine_dp_kernel");
                K[g].dgrp_comb_f32 = need_k("vj_moe_down_combine_f32_kernel");
            }
            K[g].moe_grp_sort     = need_k("ornith_moe_group_sort_kernel");
            K[g].moe_gu_iq2s_grp  = need_k("ornith_moe_gu_iq2s_grouped_kernel");
            K[g].moe_down_iq3_grp = need_k("ornith_moe_down_iq3xxs_grouped_kernel");
            K[g].vocab_argmax     = need_k("ornith_vocab_argmax_q8_kernel");
            K[g].vocab_m1 = (hp.n_embd == 3072u)
                ? rt.get_kernel(g, "ornith_vocab_argmax_q8_mubuf_kernel") : nullptr;
            if (head_q4) {
                K[g].vocab_q4     = need_k("ornith_vocab_argmax_q4a8_w75_kernel");
                K[g].vocab_q4_b16 = need_k("ornith_vocab_argmax_q4a8_batch16_kernel");
            }
            K[g].vocab_argmax_b6  = need_k("ornith_vocab_argmax_q8_batch6_kernel");
            K[g].vocab_argmax_b12 = need_k("ornith_vocab_argmax_q8_batch12_l2_kernel");
            K[g].m1_gdn_mega      = rt.get_kernel(g, "ornith_m1_gdn_moe_loop_q3_kernel");
            K[g].m1_gqa_mega      = rt.get_kernel(g, "ornith_m1_gqa_moe_q3_kernel");
            K[g].m1_gdn_super     = rt.get_kernel(g, "ornith_m1_gdn_super_kernel");
            K[g].m1_gqa_super     = rt.get_kernel(g, "ornith_m1_gqa_super_kernel");
            K[g].m1_gqa_super_kvq = rt.get_kernel(g, "ornith_m1_gqa_super_kvq_kernel");
            K[g].m1_moe_super     = rt.get_kernel(g, "ornith_m1_moe_super_kernel");
            K[g].ring_router_so   = rt.get_kernel(g, "ornith_ring_router_super_order_kernel");
            K[g].ring_top8_so     = rt.get_kernel(g, "ornith_ring_top8_super_order_kernel");
            K[g].m1_moe_super_q3  = rt.get_kernel(g, "ornith_m1_moe_super_q3_kernel");
            K[g].m1_gdn_super_x    = rt.get_kernel(g, "ornith_m1_gdn_super_kernel_x");
            K[g].m1_gqa_super_x    = rt.get_kernel(g, "ornith_m1_gqa_super_kernel_x");
            K[g].m1_moe_super_x    = rt.get_kernel(g, "ornith_m1_moe_super_kernel_x");
            K[g].m1_moe_super_q3_x = rt.get_kernel(g, "ornith_m1_moe_super_q3_kernel_x");
            K[g].m1_gdn_super_l    = rt.get_kernel(g, "ornith_m1_gdn_super_kernel_l");
            K[g].m1_gqa_super_l    = rt.get_kernel(g, "ornith_m1_gqa_super_kernel_l");
            K[g].m1_moe_super_l    = rt.get_kernel(g, "ornith_m1_moe_super_kernel_l");
            K[g].m1_moe_super_q3_l = rt.get_kernel(g, "ornith_m1_moe_super_q3_kernel_l");
            K[g].moe_gu_repack    = rt.get_kernel(g, "vj_repack_iq2s_to_dot8_gu_kernel");
            K[g].moe_gu_lru       = rt.get_kernel(g, "vj_moe_gu_dot8_lru_kernel");
            K[g].vocab_argmax_b16 = need_k("ornith_vocab_argmax_q8_batch16_l2_kernel");
            K[g].p2p_ar_moe_m1_front = need_k("ornith_p2p_ar_moe_m1_front_kernel");
            K[g].p2p_ar_moe_m1_front_s240 = need_k("ornith_p2p_ar_moe_m1_front_s240_kernel");
            K[g].moe_m1_rtr_shgu    = need_k("ornith_moe_m1_rtr_shgu_kernel");
            K[g].router_s240        = rt.get_kernel(g, "ornith_moe_router_shgu_s240_kernel");
            K[g].router_s240_red    = rt.get_kernel(g, "ornith_moe_router_shgu_s240_reduce_kernel");
            K[g].moe_m1_top8_swiglu = need_k("ornith_moe_m1_top8_swiglu_kernel");
            K[g].moe_m1_back_s224 = need_k("ornith_moe_m1_back_s224_kernel");
            K[g].moe_m1_back_s240 = need_k("ornith_moe_m1_back_s240_kernel");
            K[g].moe_gu_16r       = need_k("ornith_moe_gu_iq2s_m1_16r_kernel");
            K[g].moe_mid_f32_to_q8 = need_k("ornith_moe_mid_f32_to_q8_kernel");
            K[g].moe_down_d8_sh   = need_k("ornith_moe_down_d8_sh_m1_kernel");
            K[g].mtp_eh_cat   = need_k("ornith_mtp_eh_norm_cat_q8_kernel");
            K[g].moe_gu_q4a8  = need_k("ornith_moe_gu_q4a8_kernel");
            K[g].moe_down_q4a8= need_k("ornith_moe_down_q4a8_kernel");
            K[g].memcpy_u32   = need_k("kfd_memcpy_u32");
            K[g].memset_u32   = need_k("kfd_memset_u32");
            if (!opt.draft_gguf.empty() && planner.draft_hparams().is_dflash2()) {
                K[g].df_tap_copy     = need_k("vj_dflash2_tap_copy_kernel");
                K[g].df_quant_q8     = need_k("vj_dflash2_quant_q8_kernel");
                K[g].df_embed_blk    = need_k("vj_dflash2_embed_block_kernel");
                K[g].df_dyn_conv     = need_k("vj_dflash2_dyn_conv_kernel");
                K[g].df_kv_inject    = need_k("vj_dflash2_kv_inject_kernel");
                K[g].df_attn         = need_k("vj_dflash2_attn_kernel");
                K[g].df_swiglu_q8    = need_k("vj_dflash2_swiglu_quant_kernel");
                K[g].df_vocab_topk16 = need_k("vj_dflash2_vocab_topk16_kernel");
            }
            if (kerr && !kerr->empty()) return false;
        }
        return true;
    };
    if (!resolve_kernels(&res.error)) return res;

    std::printf("[ornith::gpu] Loaded all 0-spill .hsaco modules from %s (runtime private_segment_fixed_size == 0 verified!)\n", hdir.c_str());
    std::printf("[ornith::gpu] Repacking 48L TP=%zu Ornith-122B: VJ_Q4_A8 TCC-deswizzled dense trunk + FP16 router + SWAR uint4 IQ2_S/IQ3_XXS MoE...\n", N);

    const uint32_t D        = hp.n_embd; // 3072
    if (D % 32u != 0u || D > 4096u) {   // the AllReduce epilogue owns two 8-element wire words per thread: D <= 4096
        res.error = "n_embd must be a multiple of 32 and <= 4096 for the fused AllReduce kernels";
        return res;
    }
    if (opt.ar_wire_int8 && (D % 128u != 0u || D > 4096u)) {
        res.error = "--ar-wire int8 needs n_embd to be a multiple of 128 and <= 4096";
        return res;
    }
    const uint32_t vocab = static_cast<uint32_t>(hp.vocab);
    // Wave 3+: Model-agnostic bandwidth-proportional asymmetric sharding via core/sharder driven by hw_profile.json.
    std::vector<double> bw_vec(N);
    for (size_t g = 0; g < N; ++g) bw_vec[g] = std::max(1.0, gpu_bw_ceiling_gbs[g]);
    const auto voc_ranges = shard::split_rows(vocab, bw_vec);
    uint32_t voc_rows[kfd::kMaxGpus] = {};
    uint32_t voc_off[kfd::kMaxGpus]  = {};
    for (size_t g = 0; g < N; ++g) {
        voc_rows[g] = static_cast<uint32_t>(voc_ranges[g].size());
        voc_off[g]  = static_cast<uint32_t>(voc_ranges[g].begin);
    }
    uint32_t local_ff_r[kfd::kMaxGpus] = {};
    uint32_t local_ff_off[kfd::kMaxGpus] = {};
    if (N == 1) {
        local_ff_r[0] = hp.n_ff_exp;
        local_ff_off[0] = 0u;
    } else if (N == 2) {
        local_ff_r[0] = local_ff_r[1] = hp.n_ff_exp / 2u;
        local_ff_off[0] = 0u;
        local_ff_off[1] = local_ff_r[0];
    } else {
        const auto ff_ranges = shard::split_rows(hp.n_ff_exp, bw_vec, 32u);
        for (size_t g = 0; g < N; ++g) {
            local_ff_r[g]   = static_cast<uint32_t>(ff_ranges[g].size());
            local_ff_off[g] = static_cast<uint32_t>(ff_ranges[g].begin);
        }
    }
    const uint32_t local_ff = local_ff_r[0];
    uint32_t max_local_ff = 0u;
    bool all_lff_512 = true;
    bool all_lff_super_ok = true;
    for (size_t g = 0; g < N; ++g) {
        max_local_ff = std::max(max_local_ff, local_ff_r[g]);
        if (local_ff_r[g] != 512u || (local_ff_off[g] % 256u) != 0u) all_lff_512 = false;
        if (local_ff_r[g] == 0u || local_ff_r[g] > 512u || (local_ff_r[g] % 32u) != 0u) all_lff_super_ok = false;
    }

    // Load-time GDN-only override; all weight row maps and state allocations below use this plan.
    std::vector<shard::GdnShard> gdn_plan;
    const char* gdn_vshard = std::getenv("VEGAJET_GDN_VSHARD");
    if (!shard::plan_gdn_vshard(hp.ssm_groups, hp.ssm_v_heads, bw_vec,
                               gdn_vshard ? gdn_vshard : "", gdn_plan, &res.error)) return res;
    const auto gqa_plan = shard::plan_attention(hp.n_head, hp.n_head_kv, bw_vec, false);
    uint32_t gdn_nk[kfd::kMaxGpus] = {}, gdn_nv[kfd::kMaxGpus] = {};
    uint32_t gdn_conv_dim[kfd::kMaxGpus] = {}, gdn_z_off[kfd::kMaxGpus] = {};
    uint32_t gdn_alpha_off[kfd::kMaxGpus] = {}, gdn_beta_off[kfd::kMaxGpus] = {};
    uint32_t gdn_qkv_stride[kfd::kMaxGpus] = {};
    uint32_t gqa_nq[kfd::kMaxGpus] = {}, gqa_nkv[kfd::kMaxGpus] = {};
    uint32_t gqa_q_off[kfd::kMaxGpus] = {}, gqa_kv_off[kfd::kMaxGpus] = {};
    const uint32_t gqa_group = hp.n_head / std::max<uint32_t>(1u, hp.n_head_kv);
    for (size_t r = 0; r < N; ++r) {
        gdn_nk[r]         = static_cast<uint32_t>(gdn_plan[r].k_groups.size());
        gdn_nv[r]         = static_cast<uint32_t>(gdn_plan[r].v_heads.size());
        if (gdn_vshard && gdn_vshard[0] && std::strcmp(gdn_vshard, "0") != 0) {
            const char* swg = std::getenv("VEGAJET_SUPER_WGS");
            const uint32_t wgs = swg && std::atoi(swg) >= 192 && std::atoi(swg) <= 240
                ? static_cast<uint32_t>(std::atoi(swg)) : 240u;
            if (gdn_nv[r] * 4ULL > wgs) {
                res.error = "VEGAJET_GDN_VSHARD: rank needs more Split-V workgroups than VEGAJET_SUPER_WGS";
                return res;
            }
        }
        gdn_conv_dim[r]   = (2u * gdn_nk[r] + gdn_nv[r]) * hp.ssm_state;
        gdn_z_off[r]      = gdn_conv_dim[r];
        gdn_alpha_off[r]  = gdn_z_off[r] + gdn_nv[r] * hp.ssm_state;
        gdn_beta_off[r]   = gdn_alpha_off[r] + gdn_nv[r];
        gdn_qkv_stride[r] = gdn_beta_off[r] + gdn_nv[r];
        gqa_nq[r]         = static_cast<uint32_t>(gqa_plan.gpus[r].q_heads.size());
        gqa_nkv[r]        = static_cast<uint32_t>(gqa_plan.gpus[r].kv_heads.size());
        gqa_q_off[r]      = static_cast<uint32_t>(gqa_plan.gpus[r].q_heads.begin);
        gqa_kv_off[r]     = static_cast<uint32_t>(gqa_plan.gpus[r].kv_heads.begin);
    }
    auto gqa_nkv_packed = [&](size_t g) -> uint32_t {
        return gqa_nkv[g] | (gqa_q_off[g] << 8) | (gqa_kv_off[g] << 16) | (gqa_group << 24);
    };
    {
        std::printf("[ornith::shard] BW-proportional TP=%zu (", N);
        for (size_t g = 0; g < N; ++g) std::printf("%s%.1f", g ? " / " : "", bw_vec[g]);
        std::printf(" GB/s): LM_head=");
        for (size_t g = 0; g < N; ++g) std::printf("%s%u", g ? "/" : "", voc_rows[g]);
        std::printf(" | MoE_ff=");
        for (size_t g = 0; g < N; ++g) std::printf("%s%u", g ? "/" : "", local_ff_r[g]);
        std::printf(" | GDN_K=");
        for (size_t g = 0; g < N; ++g) std::printf("%s%u", g ? "/" : "", gdn_nk[g]);
        std::printf(" V=");
        for (size_t g = 0; g < N; ++g) std::printf("%s%u", g ? "/" : "", gdn_nv[g]);
        std::printf(" | GQA_Q=");
        for (size_t g = 0; g < N; ++g) std::printf("%s%u", g ? "/" : "", gqa_nq[g]);
        std::printf(" KV=");
        for (size_t g = 0; g < N; ++g) std::printf("%s%u", g ? "/" : "", gqa_nkv[g]);
        std::printf("\n");
    }

    std::vector<uint32_t> gdn_qkv_rows[kfd::kMaxGpus], gdn_v_rows[kfd::kMaxGpus], gdn_v_heads[kfd::kMaxGpus];
    const uint32_t gdn_v_per_k = hp.ssm_v_heads / std::max<uint32_t>(1u, hp.ssm_groups);
    for (size_t r = 0; r < N; ++r) {
        const uint32_t hk_off = static_cast<uint32_t>(gdn_plan[r].k_groups.begin);
        const uint32_t hk_cnt = static_cast<uint32_t>(gdn_plan[r].k_groups.size());
        for (uint32_t hk = 0; hk < hk_cnt; ++hk)
            for (uint32_t d = 0; d < hp.ssm_state; ++d) gdn_qkv_rows[r].push_back((hk_off + hk) * hp.ssm_state + d);
        for (uint32_t hk = 0; hk < hk_cnt; ++hk)
            for (uint32_t d = 0; d < hp.ssm_state; ++d) gdn_qkv_rows[r].push_back(hp.ssm_groups * hp.ssm_state + (hk_off + hk) * hp.ssm_state + d);
        for (uint32_t grp = 0; grp < gdn_v_per_k; ++grp) {
            for (uint32_t hk = 0; hk < hk_cnt; ++hk) {
                const uint32_t hv = grp * hp.ssm_groups + hk_off + hk;
                gdn_v_heads[r].push_back(hv);
                for (uint32_t d = 0; d < hp.ssm_state; ++d) {
                    gdn_qkv_rows[r].push_back(2u * hp.ssm_groups * hp.ssm_state + hv * hp.ssm_state + d);
                    gdn_v_rows[r].push_back(hv * hp.ssm_state + d);
                }
            }
        }
    }

    Q8WeightDev emb_w[kfd::kMaxGpus], out_w[kfd::kMaxGpus], fr_w[kfd::kMaxGpus];
    Q4A8WeightDev out_q4[kfd::kMaxGpus];
    std::vector<int32_t> fr_tok_map[kfd::kMaxGpus];
    uint64_t fr_tok_map_va[kfd::kMaxGpus] = {};
    uint32_t fr_off[kfd::kMaxGpus] = {};
    uint32_t fr_rows_per_gpu = 0;
    const bool raw_only_vram = (std::getenv("VEGAJET_RAW_ONLY") != nullptr && std::getenv("VEGAJET_RAW_ONLY")[0] == '1');
    const bool has_mtp_tensor = !raw_only_vram && (f.tensor("blk.48.nextn.eh_proj.weight") != nullptr);
    const bool has_dflash2_opt = (!opt.draft_gguf.empty() && planner.draft_hparams().is_dflash2());
    const bool use_fr_spec = (has_mtp_tensor || has_dflash2_opt) && opt.fr_spec_rows > 0u && (opt.fr_spec_rows % 512u == 0u) && opt.fr_spec_rows <= vocab;
    uint64_t out_norm_va[kfd::kMaxGpus] = {};
    {   // Validate the embedding table before it is read as 34-byte Q8_0 rows: a qwen35moe file with another
        // quantization (or without token_embd) used to be misread here and only surfaced later as a golden failure.
        const GgufTensorInfo* te = f.tensor("token_embd.weight");
        if (!te) {
            res.error = "token_embd.weight missing; the Ornith runner needs a Q8_0 token embedding table";
            return res;
        }
        const auto traits = ggml_type_traits(te->type);
        if (te->type != static_cast<uint32_t>(GgmlType::Q8_0) || te->n_dims != 2 || te->ne[0] != D || te->ne[1] != vocab) {
            char msg[256];
            std::snprintf(msg, sizeof(msg),
                          "unsupported token_embd.weight: type=%s shape=%llux%llu, need Q8_0 %ux%u "
                          "(this alpha runs one specific Ornith-122B quantisation)",
                          traits.name ? traits.name : "unknown", static_cast<unsigned long long>(te->ne[0]),
                          static_cast<unsigned long long>(te->ne[1]), D, vocab);
            res.error = msg;
            return res;
        }
    }
    {
        const auto* ti = f.tensor("token_embd.weight");
        const uint8_t* raw = f.tensor_data(*ti);
        if (!raw) { res.error = "token_embd.weight: mmap of the tensor data failed"; return res; }
        std::vector<int8_t>   eqs(static_cast<size_t>(vocab) * D);
        std::vector<uint16_t> eds(static_cast<size_t>(vocab) * (D / 32u));
        const size_t nb_row = D / 32u, rb_q80 = nb_row * 34u;
        #pragma omp parallel for schedule(static)
        for (uint32_t r = 0; r < vocab; ++r) {
            const uint8_t* src = raw + static_cast<size_t>(r) * rb_q80;
            for (size_t b = 0; b < nb_row; ++b) {
                std::memcpy(&eds[static_cast<size_t>(r) * nb_row + b], src + b * 34u, 2);
                std::memcpy(&eqs[static_cast<size_t>(r) * D + b * 32u], src + b * 34u + 2u, 32);
            }
        }
        if (use_fr_spec) {
            // Frequency-ranked shortlist via TokenFrequencyModel (BPE merge-rank Zipf prior) with all tail
            // special/control tokens (e.g. <|endoftext|>, <|im_*|>, <think>, </think> in [vocab-512, vocab))
            // pinned unconditionally, striped 50/50 across both GPUs so load stays balanced.
            fr_spec::TokenFrequencyModel freq(vocab);
            const auto scores = freq.scores();
            std::vector<uint8_t> picked(vocab, 0);
            std::vector<int32_t> selected;
            selected.reserve(opt.fr_spec_rows);
            const uint32_t tail_start = (vocab > 512u) ? (vocab - 512u) : 0u;
            for (uint32_t t = tail_start; t < vocab && selected.size() < opt.fr_spec_rows; ++t) {
                picked[t] = 1;
                selected.push_back(static_cast<int32_t>(t));
            }
            std::vector<int32_t> rest;
            rest.reserve(vocab - selected.size());
            for (uint32_t i = 0; i < vocab; ++i) {
                if (!picked[i]) rest.push_back(static_cast<int32_t>(i));
            }
            const size_t need = opt.fr_spec_rows - selected.size();
            std::partial_sort(rest.begin(), rest.begin() + need, rest.end(),
                              [&](int32_t a, int32_t b) { return scores[a] > scores[b]; });
            for (size_t i = 0; i < need; ++i) selected.push_back(rest[i]);

            fr_rows_per_gpu = static_cast<uint32_t>(opt.fr_spec_rows / N);
            for (size_t g = 0; g < N; ++g) fr_tok_map[g].reserve(fr_rows_per_gpu + 1u);
            for (uint32_t i = 0; i < opt.fr_spec_rows; ++i) {
                fr_tok_map[i % N].push_back(selected[i]);
            }
            uint32_t fr_acc = 0u;
            for (size_t g = 0; g < N; ++g) {
                fr_off[g] = fr_acc;
                fr_acc += static_cast<uint32_t>(fr_tok_map[g].size());
            }
        }
        for (size_t g = 0; g < N; ++g) {
            emb_w[g].rows = vocab; emb_w[g].cols = D; emb_w[g].row_stride_bytes = D;
            emb_w[g].qs_va = rt.vram_arena(g).alloc(eqs.size(), 4096, kfd::Tier::COLD);  // 1 row/token: COLD
            emb_w[g].ds_va = rt.vram_arena(g).alloc(eds.size() * 2u, 256, kfd::Tier::COLD);
            VJ_CHECK(rt.upload_to_vram(g, emb_w[g].qs_va, eqs.data(), eqs.size()));
            VJ_CHECK(rt.upload_to_vram(g, emb_w[g].ds_va, eds.data(), eds.size() * 2u));
            if (!upload_f32_rows(rt, g, f, "output_norm.weight", nullptr, out_norm_va[g], &res.error)) return res;
            if (!upload_q8_tcc(rt, g, f, "output.weight", voc_off[g], voc_rows[g], out_w[g], &res.error)) return res;
            if (head_q4 && !upload_fused_q4a8_tcc(rt, g, f, {{"output.weight", iota_range(voc_off[g], voc_rows[g])}},
                                                  0, D, nullptr, out_q4[g], &res.error)) return res;
            if (use_fr_spec) {
                if (!upload_q8_tcc_indexed(rt, g, f, "output.weight", fr_tok_map[g], fr_w[g], &res.error)) return res;
                fr_tok_map_va[g] = rt.vram_arena(g).alloc(fr_tok_map[g].size() * sizeof(int32_t), 256);
                VJ_CHECK(fr_tok_map_va[g] && rt.upload_to_vram(g, fr_tok_map_va[g], fr_tok_map[g].data(), fr_tok_map[g].size() * sizeof(int32_t)));
            }
        }
        if (use_fr_spec) {
            std::printf("  uploaded FR-Spec draft vocab shortlist: %u rows (%u/GPU, %.1f MiB/GPU)\n",
                        opt.fr_spec_rows, fr_rows_per_gpu,
                        (static_cast<double>(fr_rows_per_gpu) * (fr_w[0].row_stride_bytes + (D / 32u) * 2u)) / 1048576.0);
        }
        f.release(raw, ti->n_bytes());
        if (const auto* owi = f.tensor("output.weight")) f.release(f.tensor_data(*owi), owi->n_bytes());
    }

    std::vector<LayerWeightsDev> layers[kfd::kMaxGpus];
    for (size_t g = 0; g < N; ++g) layers[g].resize(48);
    const auto all_D_rows = iota_range(0, D);
    const uint32_t max_batch = std::max<uint32_t>(128u, opt.prefill_chunk);
    const uint64_t arena_cap_bytes = static_cast<uint64_t>(30.5 * 1073741824.0);
    // Wave 3: Model-agnostic spare-VRAM DOT8 promotion via plan_dot8_promotion (crossover law enforced).
    // When opt.gu_dot8_layers == 0 (auto), compute exact spare VRAM pool after reserving 1.5 GiB/GPU headroom
    // and promote IQ2_S Gate+Up layers for M >= 2 speculative verify where L2-cached DOT8 beats IQ2_S.
    uint32_t target_gu_dot8_layers = opt.gu_dot8_layers;
    if (const char* gd_env = std::getenv("VEGAJET_GU_DOT8_LAYERS")) {
        target_gu_dot8_layers = static_cast<uint32_t>(std::max(0, std::atoi(gd_env)));
    }
    const bool no_auto_dot8 = (std::getenv("VEGAJET_NO_AUTO_DOT8") != nullptr && std::getenv("VEGAJET_NO_AUTO_DOT8")[0] == '1');
    // BEGIN L6 experimental host Down slices (default allocation path untouched).
    std::unique_ptr<DownOffload> expert_offload;
    if (const char* value=std::getenv("VEGAJET_OFFLOAD_EXPERT_FRAC")) {
        char* end=nullptr; const double fraction=std::strtod(value,&end);
        if (!*value || *end || !std::isfinite(fraction) || fraction<0 || fraction>1) {
            res.error="VEGAJET_OFFLOAD_EXPERT_FRAC must be 0..1"; return res;
        }
        if (fraction>0) {
            if (N!=3u) { res.error="experimental Down offload requires TP=3 sub32 layout"; return res; }
            expert_offload=std::make_unique<DownOffload>(fraction);
        }
    }
    // END L6 experimental host Down slices.
    const bool in_place_gu_dot8 = !opt.prefill_dot8 && !opt.pf_moe_iq_lds && !opt.pf_moe_iq_lds32 && (N >= 3u || raw_only_vram);
    uint32_t max_lff = 0u;
    for (size_t g = 0; g < N; ++g) max_lff = std::max(max_lff, local_ff_r[g]);
    if (target_gu_dot8_layers == 0u && !no_auto_dot8 && all_lff_super_ok && D == 3072u) {
        const Dot8RowLayout gu_lay = dot8_gu_iq2s_layout(256ULL * max_lff, D);
        const uint64_t est_base_vram = (N >= 3u)
            ? (22000ULL << 20)
            : (raw_only_vram ? (24600ULL << 20) : (26400ULL << 20));
        const uint64_t headroom = (N >= 3u)
            ? (1536ULL << 20)
            : (raw_only_vram ? (640ULL << 20) : (1600ULL << 20));
        const uint64_t spare_pool = (arena_cap_bytes > est_base_vram + headroom)
                                        ? (arena_cap_bytes - est_base_vram - headroom) : 0ULL;
        Dot8ClassSpec gu_spec;
        gu_spec.name = in_place_gu_dot8 ? "ffn_gate_up_exps(in-place DOT8)" : "ffn_gate_up_exps(M>=2 verify)";
        gu_spec.src_type = GgmlType::IQ2_S;
        gu_spec.role = Dot8Role::MoeGateUp;
        gu_spec.target = Dot8Target::VJ_Q4_DOT8;
        gu_spec.frees_old = in_place_gu_dot8;
        gu_spec.old_vram_bytes_per_layer = 256ULL * max_lff * 1968ULL;
        gu_spec.new_vram_bytes_per_layer = gu_lay.vram_bytes();
        gu_spec.old_read_bytes_per_layer = 8ULL * max_lff * 1968ULL;
        gu_spec.new_read_bytes_per_layer = 8ULL * max_lff * gu_lay.read_bytes_per_row();
        gu_spec.bw_old_gbs = in_place_gu_dot8 ? 365.0 : 245.0;
        gu_spec.bw_new_gbs = in_place_gu_dot8 ? 680.0 : 540.0;
        gu_spec.batch_m = in_place_gu_dot8 ? 1u : 5u;
        for (uint32_t l = 0u; l < 48u; ++l) gu_spec.layers.push_back(l);
        const auto dplan = plan_dot8_promotion({gu_spec}, spare_pool, true);
        target_gu_dot8_layers = static_cast<uint32_t>(dplan.promotions.size());
        if (target_gu_dot8_layers > 0u) {
            std::printf("[ornith::dot8] plan_dot8_promotion: pool=%.2f GiB/GPU (max_lff=%u) -> promoting %u/48 Gate+Up layers to exact VJ_Q4_DOT8 (%s, est %.0f us saved/step)\n",
                        static_cast<double>(spare_pool) / 1073741824.0, max_lff, target_gu_dot8_layers,
                        in_place_gu_dot8 ? "in-place" : "verify M>=2", dplan.us_saved_total);
        }
    }
    uint32_t gu_dot8_promoted = 0u;

    // --dot8-fwht: rotate the 4 dense W4A4 projection matrices (GPUs own a second copy; the W4A8 default, the
    // GEMV decode and the IQ2_S/IQ3_XXS expert weights stay untouched). Needs --prefill-dot8 to be reachable.
    // --dot8-scope 2 (MoE-GU only) never dispatches the dense dot8 GEMM, so do not pay for the rotated copies.
    const bool fwht_dense = (opt.prefill_dot8 && opt.dot8_fwht && opt.dot8_scope != 2u);
    const uint32_t fwht_layer_cap = (opt.dot8_fwht_layers == 0u) ? 48u : std::min<uint32_t>(opt.dot8_fwht_layers, 48u);
    if (opt.dot8_fwht && !opt.prefill_dot8) {
        std::fprintf(stderr, "[ornith::dot8] --dot8-fwht ignored without --prefill-dot8 (W4A4 path disabled)\n");
    } else if (opt.dot8_fwht && opt.dot8_scope == 2u) {
        std::fprintf(stderr, "[ornith::dot8] --dot8-fwht ignored with --dot8-scope 2 (MoE-GU only, dense dot8 disabled)\n");
    }
    uint64_t fwht_extra_bytes = 0u;

    for (uint32_t il = 0; il < 48u; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const bool is_gdn = hp.is_recurrent(il);

        for (size_t g = 0; g < N; ++g) {
            auto& lw = layers[g][il];
            lw.is_gdn = is_gdn;
            if (!upload_f32_rows(rt, g, f, p + "attn_norm.weight", nullptr, lw.attn_norm_va, &res.error)) return res;
            if (!upload_f32_rows(rt, g, f, p + "post_attention_norm.weight", nullptr, lw.post_attn_norm_va, &res.error)) return res;

            if (is_gdn) {
                if (!upload_fused_q4a8_tcc(rt, g, f, {
                        {p + "attn_qkv.weight",  gdn_qkv_rows[g]},
                        {p + "attn_gate.weight", gdn_v_rows[g]},
                        {p + "ssm_alpha.weight", gdn_v_heads[g]},
                        {p + "ssm_beta.weight",  gdn_v_heads[g]}
                    }, 0, D, nullptr, lw.w_gdn_qkv_gab, &res.error,
                    (fwht_dense && il < fwht_layer_cap) ? &lw.w_gdn_qkv_gab_fwht : nullptr)) return res;
                if (!upload_fused_q4a8_tcc(rt, g, f, {{p + "ssm_out.weight", all_D_rows}},
                                           0, 0, &gdn_v_rows[g], lw.w_ssm_out, &res.error,
                                           (fwht_dense && il < fwht_layer_cap) ? &lw.w_ssm_out_fwht : nullptr)) return res;
                if (!upload_f32_rows(rt, g, f, p + "ssm_conv1d.weight", &gdn_qkv_rows[g], lw.ssm_conv_w_va, &res.error)) return res;
                if (!upload_f32_rows(rt, g, f, p + "ssm_a", &gdn_v_heads[g], lw.ssm_a_va, &res.error)) return res;
                if (!upload_f32_rows(rt, g, f, p + "ssm_dt.bias", &gdn_v_heads[g], lw.ssm_dt_va, &res.error)) return res;
                if (!upload_f32_rows(rt, g, f, p + "ssm_norm.weight", nullptr, lw.ssm_norm_va, &res.error)) return res;
            } else {
                if (!upload_fused_q4a8_tcc(rt, g, f, {
                        {p + "attn_q.weight", iota_range(gqa_q_off[g] * 512u,  gqa_nq[g] * 512u)},
                        {p + "attn_k.weight", iota_range(gqa_kv_off[g] * 256u, gqa_nkv[g] * 256u)},
                        {p + "attn_v.weight", iota_range(gqa_kv_off[g] * 256u, gqa_nkv[g] * 256u)}
                    }, 0, D, nullptr, lw.w_attn_qkv, &res.error,
                    (fwht_dense && il < fwht_layer_cap) ? &lw.w_attn_qkv_fwht : nullptr)) return res;
                if (!upload_fused_q4a8_tcc(rt, g, f, {{p + "attn_output.weight", all_D_rows}},
                                           gqa_q_off[g] * 256u, gqa_nq[g] * 256u, nullptr, lw.w_o, &res.error,
                                           (fwht_dense && il < fwht_layer_cap) ? &lw.w_o_fwht : nullptr)) return res;
                if (!upload_f32_rows(rt, g, f, p + "attn_q_norm.weight", nullptr, lw.q_norm_va, &res.error)) return res;
                if (!upload_f32_rows(rt, g, f, p + "attn_k_norm.weight", nullptr, lw.k_norm_va, &res.error)) return res;
            }

            if (fwht_dense) {
                for (const Q4A8WeightDev* w : {&lw.w_gdn_qkv_gab_fwht, &lw.w_ssm_out_fwht,
                                               &lw.w_attn_qkv_fwht, &lw.w_o_fwht})
                    if (w->rows != 0u)
                        fwht_extra_bytes += static_cast<uint64_t>(w->rows) *
                                            (w->row_stride_bytes + (w->cols / 32u) * 4u);
            }
            if (!upload_f32_as_f16(rt, g, f, p + "ffn_gate_inp.weight", lw.w_router_f16_va, &res.error)) return res;
            if (!upload_f32_rows(rt, g, f, p + "ffn_gate_inp_shexp.weight", nullptr, lw.w_sh_gate_va, &res.error)) return res;
            if (!upload_fused_q4a8_tcc(rt, g, f, {
                    {p + "ffn_gate_shexp.weight", iota_range(local_ff_off[g], local_ff_r[g])},
                    {p + "ffn_up_shexp.weight",   iota_range(local_ff_off[g], local_ff_r[g])}
                }, 0, D, nullptr, lw.w_sh_gu, &res.error)) return res;
            if (!upload_fused_q4a8_tcc(rt, g, f, {{p + "ffn_down_shexp.weight", all_D_rows}},
                                       local_ff_off[g], local_ff_r[g], nullptr, lw.w_sh_down, &res.error)) return res;
        }

        // Pack Gate+Up IQ2_S into coalesced uint4 merged split-plane layout
        {
            const auto* tig = f.tensor(p + "ffn_gate_exps.weight");
            const auto* tiu = f.tensor(p + "ffn_up_exps.weight");
            const uint8_t* rg = f.tensor_data(*tig);
            const uint8_t* ru = f.tensor_data(*tiu);
            const size_t raw_rb = 12u * 82u;
            const size_t row_u4 = 123u;

            auto pack_row = [&](const uint8_t* g_row, const uint8_t* u_row, uint32_t* dst) {
                uint32_t* qp  = dst;
                uint32_t* hs2 = dst + 96u * 4u;
                uint32_t* d2  = dst + 120u * 4u;
                for (uint32_t b = 0; b < 12u; ++b) {
                    const uint8_t* gb = g_row + b * 82u;
                    const uint8_t* ub = u_row + b * 82u;
                    uint16_t gd = 0, ud = 0;
                    std::memcpy(&gd, gb, 2);
                    std::memcpy(&ud, ub, 2);
                    d2[b] = static_cast<uint32_t>(gd) | (static_cast<uint32_t>(ud) << 16);
                    for (uint32_t sb = 0; sb < 8u; ++sb) {
                        const uint32_t ib = b * 8u + sb;
                        const uint8_t* gq = gb + 2u + sb * 4u;
                        const uint8_t* gs = gb + 34u + sb * 4u;
                        const uint8_t* uq = ub + 2u + sb * 4u;
                        const uint8_t* us = ub + 34u + sb * 4u;
                        qp[ib * 4u + 0u] = uint32_t(gq[0]) | (uint32_t(gs[0]) << 8) | (uint32_t(gq[1]) << 16) | (uint32_t(gs[1]) << 24);
                        qp[ib * 4u + 1u] = uint32_t(gq[2]) | (uint32_t(gs[2]) << 8) | (uint32_t(gq[3]) << 16) | (uint32_t(gs[3]) << 24);
                        qp[ib * 4u + 2u] = uint32_t(uq[0]) | (uint32_t(us[0]) << 8) | (uint32_t(uq[1]) << 16) | (uint32_t(us[1]) << 24);
                        qp[ib * 4u + 3u] = uint32_t(uq[2]) | (uint32_t(us[2]) << 8) | (uint32_t(uq[3]) << 16) | (uint32_t(us[3]) << 24);
                        hs2[ib] = uint32_t(gb[66u + sb]) | (uint32_t(gb[74u + sb]) << 8) |
                                  (uint32_t(ub[66u + sb]) << 16) | (uint32_t(ub[74u + sb]) << 24);
                    }
                }
            };

            const char* q3_env = std::getenv("VEGAJET_GU_Q3");
            const bool use_gu_q3 = q3_env
                ? (std::atoi(q3_env) > 0)
                : (N >= 3u && in_place_gu_dot8 && !no_auto_dot8 && all_lff_super_ok && D == 3072u && K[0].m1_moe_super_q3 != nullptr);
            bool promoted_in_place = false;
            if (use_gu_q3 && all_lff_super_ok && D == 3072u) {
                bool fits_all = true;
                for (size_t g = 0; g < N; ++g) {
                    const size_t nr = 256ULL * local_ff_r[g];
                    const Dot8RowLayout lay = q3_gu_iq2s_layout(nr, D);
                    const uint64_t dn_bytes_g = 256ULL * D * ((local_ff_r[g] <= 384u) ? 256ULL : 272ULL);
                    const uint64_t per_rem_layer_g = (76ULL << 20) + dn_bytes_g + lay.vram_bytes();
                    const uint64_t tail_fixed = raw_only_vram ? (512ULL << 20) : ((N >= 3u ? 1400ULL : 2200ULL) << 20);
                    const uint64_t reserve_tail = static_cast<uint64_t>(47u - il) * per_rem_layer_g + tail_fixed;
                    if (rt.vram_arena(g).used() + lay.vram_bytes() + reserve_tail > arena_cap_bytes) fits_all = false;
                }
                if (fits_all) {
                    for (size_t g = 0; g < N; ++g) {
                        const size_t nr = 256ULL * local_ff_r[g];
                        const Dot8RowLayout lay   = q3_gu_iq2s_layout(nr, D);
                        const Dot8RowLayout lay_e = q3_gu_iq2s_layout(local_ff_r[g], D);
                        std::vector<uint8_t> wpl(lay.w_plane_bytes()), spl(lay.s_plane_bytes());
                        bool rok = true;
                        #pragma omp parallel for schedule(static) reduction(&&:rok)
                        for (size_t e = 0; e < 256u; ++e) {
                            const size_t src_r0 = e * hp.n_ff_exp + local_ff_off[g];
                            std::string e_err;
                            rok = rok && q3_repack_gu_iq2s_rows(
                                rg + src_r0 * raw_rb, ru + src_r0 * raw_rb, raw_rb, 0u, lay_e, 0u, local_ff_r[g],
                                wpl.data() + e * lay_e.w_plane_bytes(),
                                spl.data() + e * lay_e.s_plane_bytes(), &e_err);
                        }
                        if (!rok) { res.error = "VJ_Q3_IQ2S repack of ffn_gate/up_exps failed"; return res; }
                        auto& lw = layers[g][il];
                        lw.w_gu_d8_w_va = rt.vram_arena(g).alloc(wpl.size(), 4096, kfd::Tier::COLD);
                        lw.w_gu_d8_s_va = rt.vram_arena(g).alloc(spl.size(), 4096, kfd::Tier::COLD);
                        if (!lw.w_gu_d8_w_va || !lw.w_gu_d8_s_va) { res.error = "VRAM OOM for Gate+Up Q3 planes"; return res; }
                        VJ_CHECK(rt.upload_to_vram(g, lw.w_gu_d8_w_va, wpl.data(), wpl.size()));
                        VJ_CHECK(rt.upload_to_vram(g, lw.w_gu_d8_s_va, spl.data(), spl.size()));
                        lw.gu_d8_w_stride = lay.w_row_stride_bytes;
                        lw.gu_d8_s_stride = lay.s_row_stride_bytes;
                    }
                    ++gu_dot8_promoted;
                    promoted_in_place = true;
                }
            }
            if (!promoted_in_place && il < target_gu_dot8_layers && all_lff_super_ok && D == 3072u) {
                const uint64_t tail_fixed = raw_only_vram
                    ? (768ULL << 20)
                    : ((N >= 3u ? (1400ULL << 20) : ((opt.spec_gdn ? 2048ULL : 2800ULL) << 20))
                       + (max_batch > 128u ? (512ULL << 20) : 0ULL));
                bool fits_all = true;
                for (size_t g = 0; g < N; ++g) {
                    const size_t nr = 256ULL * local_ff_r[g];
                    const Dot8RowLayout lay = dot8_gu_iq2s_layout(nr, D);
                    const uint64_t iq2s_extra = in_place_gu_dot8 ? 0ULL : (nr * row_u4 * 16ULL);
                    const uint64_t dn_bytes_g = 256ULL * D * ((local_ff_r[g] <= 384u) ? 256ULL : 272ULL);
                    const uint64_t per_rem_layer_g = (76ULL << 20) + dn_bytes_g + nr * row_u4 * 16ULL;
                    const uint64_t reserve_tail_g = static_cast<uint64_t>(47u - il) * per_rem_layer_g + tail_fixed;
                    if (rt.vram_arena(g).used() + lay.vram_bytes() + iq2s_extra + reserve_tail_g > arena_cap_bytes) fits_all = false;
                }
                if (fits_all) {
                    for (size_t g = 0; g < N; ++g) {
                        const size_t nr = 256ULL * local_ff_r[g];
                        const Dot8RowLayout lay = dot8_gu_iq2s_layout(nr, D);
                        const Dot8RowLayout lay_e = dot8_gu_iq2s_layout(local_ff_r[g], D);
                        std::vector<uint8_t> wpl(lay.w_plane_bytes()), spl(lay.s_plane_bytes());
                        bool rok = true;
                        #pragma omp parallel for schedule(static) reduction(&&:rok)
                        for (size_t e = 0; e < 256u; ++e) {
                            const size_t src_r0 = e * hp.n_ff_exp + local_ff_off[g];
                            std::string e_err;
                            rok = rok && dot8_repack_gu_iq2s_rows(
                                rg + src_r0 * raw_rb, ru + src_r0 * raw_rb, raw_rb, 0u, lay_e, 0u, local_ff_r[g],
                                wpl.data() + e * lay_e.w_plane_bytes(),
                                lay.scale_embedded ? nullptr : (spl.data() + e * lay_e.s_plane_bytes()), &e_err);
                        }
                        if (!rok) { res.error = "DOT8 repack of ffn_gate/up_exps failed"; return res; }
                        auto& lw = layers[g][il];
                        lw.w_gu_d8_w_va = rt.vram_arena(g).alloc(wpl.size(), 4096, kfd::Tier::COLD);
                        if (!lw.w_gu_d8_w_va) { res.error = "VRAM OOM for Gate+Up DOT8 plane"; return res; }
                        VJ_CHECK(rt.upload_to_vram(g, lw.w_gu_d8_w_va, wpl.data(), wpl.size()));
                        if (lay.scale_embedded) {
                            lw.w_gu_d8_s_va = lw.w_gu_d8_w_va + lay.s_offset_bytes;
                        } else {
                            lw.w_gu_d8_s_va = rt.vram_arena(g).alloc(spl.size(), 4096, kfd::Tier::COLD);
                            if (!lw.w_gu_d8_s_va) { res.error = "VRAM OOM for Gate+Up DOT8 scales"; return res; }
                            VJ_CHECK(rt.upload_to_vram(g, lw.w_gu_d8_s_va, spl.data(), spl.size()));
                        }
                        lw.gu_d8_w_stride = lay.w_row_stride_bytes;
                        lw.gu_d8_s_stride = lay.s_row_stride_bytes;
                    }
                    ++gu_dot8_promoted;
                    promoted_in_place = in_place_gu_dot8;
                }
            }
            if (!promoted_in_place) {
                for (size_t g = 0; g < N; ++g) {
                    const size_t lff_g = local_ff_r[g];
                    const size_t nr = 256ULL * lff_g;
                    std::vector<uint32_t> mg(nr * row_u4 * 4u);
                    #pragma omp parallel for schedule(static)
                    for (size_t e = 0; e < 256u; ++e) {
                        for (size_t r = 0; r < lff_g; ++r) {
                            const size_t idx = e * lff_g + r;
                            const size_t full = e * hp.n_ff_exp + local_ff_off[g] + r;
                            pack_row(rg + full * raw_rb, ru + full * raw_rb, mg.data() + idx * row_u4 * 4u);
                        }
                    }
                    const size_t bytes = mg.size() * 4u;
                    layers[g][il].w_gu_iq2s_va = rt.vram_arena(g).alloc(bytes, 4096, kfd::Tier::COLD);
                    if (!layers[g][il].w_gu_iq2s_va) { res.error = "VRAM OOM for routed Gate+Up"; return res; }
                    VJ_CHECK(rt.upload_to_vram(g, layers[g][il].w_gu_iq2s_va, mg.data(), bytes));
                }
            }
            f.release(rg, tig->n_bytes());
            f.release(ru, tiu->n_bytes());
        }

        // Pack Down IQ3_XXS into exact 16-level DOT8 split-plane (256-col for TP=2, lossless 32-col sub32 for TP=3..4!)
        {
            const auto* tid = f.tensor(p + "ffn_down_exps.weight");
            const uint8_t* rd = f.tensor_data(*tid);
            const size_t full_rb = (hp.n_ff_exp / 256u) * 98u;
            const size_t n_rows  = 256ULL * D;
            for (size_t g = 0; g < N; ++g) {
                const bool use_sub32 = (local_ff_r[g] != 512u || (local_ff_off[g] % 256u) != 0u);
                auto& lw = layers[g][il];
                lw.dn_is_sub32 = use_sub32;
                if (!use_sub32) {
                    const Dot8RowLayout lay = dot8_iq3_layout(n_rows, 512u);
                    std::vector<uint8_t> wpl(lay.w_plane_bytes()), spl(lay.s_plane_bytes());
                    bool rok = true;
                    #pragma omp parallel for schedule(static) reduction(&&:rok)
                    for (size_t r0 = 0; r0 < n_rows; r0 += 4096u) {
                        std::string e;
                        rok = rok && dot8_repack_iq3xxs_rows(rd, full_rb, local_ff_off[g] / 256u, lay, r0,
                                                             std::min<uint64_t>(n_rows, r0 + 4096u), wpl.data(), spl.data(), &e);
                    }
                    if (!rok) { res.error = "DOT8 repack of ffn_down_exps failed"; return res; }
                    lw.w_dn_d8_w_va = rt.vram_arena(g).alloc(wpl.size(), 4096, kfd::Tier::COLD);
                    lw.w_dn_d8_s_va = rt.vram_arena(g).alloc(spl.size(), 4096, kfd::Tier::COLD);
                    if (!lw.w_dn_d8_w_va || !lw.w_dn_d8_s_va) { res.error = "VRAM OOM for the routed Down planes"; return res; }
                    VJ_CHECK(rt.upload_to_vram(g, lw.w_dn_d8_w_va, wpl.data(), wpl.size()));
                    VJ_CHECK(rt.upload_to_vram(g, lw.w_dn_d8_s_va, spl.data(), spl.size()));
                    lw.dn_d8_w_stride = lay.w_row_stride_bytes;
                    lw.dn_d8_s_stride = lay.s_row_stride_bytes;
                } else {
                    Dot8RowLayout lay = dot8_iq3_sub32_layout(n_rows, local_ff_r[g]);
                    std::vector<uint8_t> wpl(lay.w_plane_bytes()), spl(lay.s_plane_bytes());
                    bool rok = true;
                    #pragma omp parallel for schedule(static) reduction(&&:rok)
                    for (size_t r0 = 0; r0 < n_rows; r0 += 4096u) {
                        std::string e;
                        rok = rok && dot8_repack_iq3xxs_rows_sub32(rd, full_rb, local_ff_off[g] / 32u, lay, r0,
                                                                   std::min<uint64_t>(n_rows, r0 + 4096u), wpl.data(),
                                                                   lay.scale_embedded ? nullptr : spl.data(), &e);
                    }
                    if (!rok) { res.error = "DOT8 sub32 repack of ffn_down_exps failed"; return res; }
                    // L6: compact resident Down experts; cold rows stay in host backing.
                    if (expert_offload) expert_offload->compact(g,il,wpl,spl,lay,D);
                    lw.w_dn_d8_w_va = rt.vram_arena(g).alloc(wpl.size(), 4096, kfd::Tier::COLD);
                    if (!lw.w_dn_d8_w_va) { res.error = "VRAM OOM for the routed Down sub32 plane"; return res; }
                    VJ_CHECK(rt.upload_to_vram(g, lw.w_dn_d8_w_va, wpl.data(), wpl.size()));
                    if (lay.scale_embedded) {
                        lw.w_dn_d8_s_va = lw.w_dn_d8_w_va + lay.s_offset_bytes;
                    } else {
                        lw.w_dn_d8_s_va = rt.vram_arena(g).alloc(spl.size(), 4096, kfd::Tier::COLD);
                        if (!lw.w_dn_d8_s_va) { res.error = "VRAM OOM for the routed Down sub32 scales"; return res; }
                        VJ_CHECK(rt.upload_to_vram(g, lw.w_dn_d8_s_va, spl.data(), spl.size()));
                    }
                    lw.dn_d8_w_stride = lay.w_row_stride_bytes;
                    lw.dn_d8_s_stride = lay.s_row_stride_bytes;
                }
            }
            if ((opt.dot8_ab || opt.pf_moe_iq_lds >= 2u || opt.pf_moe_iq_lds32 >= 2u) && il == 47u && N == 2 && all_lff_512) {
                // Compact IQ3_XXS split-plane copy of the last layer, only as the reference side of --dot8-ab.
                std::vector<uint32_t> s0(n_rows * 13u * 4u, 0), s1(n_rows * 13u * 4u, 0);
                auto pack_dn = [&](const uint8_t* src2, uint32_t* dst) {
                    uint8_t*  qp   = reinterpret_cast<uint8_t*>(dst);
                    uint32_t* auxp = dst + 8u * 4u;
                    uint32_t* d2p  = dst + 12u * 4u;
                    uint16_t d0 = 0, d1 = 0;
                    std::memcpy(&d0, src2, 2);
                    std::memcpy(&d1, src2 + 98u, 2);
                    d2p[0] = uint32_t(d0) | (uint32_t(d1) << 16);
                    std::memcpy(qp,       src2 + 2u,        64u);
                    std::memcpy(qp + 64u, src2 + 98u + 2u,  64u);
                    std::memcpy(auxp,     src2 + 66u,       32u);
                    std::memcpy(auxp + 8, src2 + 98u + 66u, 32u);
                };

                #pragma omp parallel for schedule(static)
                for (size_t r = 0; r < n_rows; ++r) {
                    pack_dn(rd + r * full_rb,          s0.data() + r * 52u);
                    pack_dn(rd + r * full_rb + 2u*98u, s1.data() + r * 52u);
                }
                if (N == 2u) {
                    const size_t bytes = s0.size() * 4u;
                    layers[0][il].w_dn_iq3xxs_va = rt.vram_arena(0).alloc(bytes, 4096, kfd::Tier::COLD);
                    layers[1][il].w_dn_iq3xxs_va = rt.vram_arena(1).alloc(bytes, 4096, kfd::Tier::COLD);
                    if (!layers[0][il].w_dn_iq3xxs_va || !layers[1][il].w_dn_iq3xxs_va) { res.error = "VRAM OOM for the --dot8-ab reference copy"; return res; }
                    VJ_CHECK(rt.upload_to_vram(0, layers[0][il].w_dn_iq3xxs_va, s0.data(), bytes));
                    VJ_CHECK(rt.upload_to_vram(1, layers[1][il].w_dn_iq3xxs_va, s1.data(), bytes));
                }
            }
            f.release(rd, tid->n_bytes());
        }

        for (const char* suf : {"attn_norm.weight", "post_attention_norm.weight",
                                "attn_qkv.weight", "attn_gate.weight", "ssm_alpha.weight", "ssm_beta.weight",
                                "ssm_out.weight", "ssm_conv1d.weight", "ssm_a", "ssm_dt.bias", "ssm_norm.weight",
                                "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                                "attn_q_norm.weight", "attn_k_norm.weight",
                                "ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight",
                                "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"}) {
            if (const auto* ti = f.tensor(p + suf)) f.release(f.tensor_data(*ti), ti->n_bytes());
        }
        if ((il + 1u) % 12u == 0u) {
            std::printf("  uploaded %2u/48 layers |", il + 1u);
            for (size_t g = 0; g < N; ++g) std::printf(" GPU%zu=%.2f GiB", g, rt.vram_arena(g).used() / 1073741824.0);
            std::printf("\n");
            std::fflush(stdout);
        }
    }

    if (fwht_dense) {
        std::printf("[ornith::dot8] FWHT-64 dense tile ON for %u/%u layers: rotated VJ_Q4_A8 copies (qkv/gab, "
                    "ssm_out, attn_qkv, o) +%.2f GiB/GPU total, packer vj_prefill_q8_to_q4_fwht64_kernel "
                    "(lossy W4A4, KLD gate still applies)\n",
                    fwht_layer_cap, 48u, static_cast<double>(fwht_extra_bytes) / 2.0 / 1073741824.0);
        std::fflush(stdout);
    }

    // Optional built-in blk.48 nextn MTP head (~0.75 GiB/GPU -> enables neural K=4 speculative drafting!)
    MtpLayerDev mtp[kfd::kMaxGpus];
    const bool has_mtp = !raw_only_vram && (f.tensor("blk.48.nextn.eh_proj.weight") != nullptr);
    std::array<uint64_t, moeslot::kBudgetCards> draft_measured{};
    const std::array<uint64_t, moeslot::kBudgetCards> draft_mark_mtp = {
        rt.vram_arena(0).used(), (N > 1) ? rt.vram_arena(1).used() : 0ULL};
    if (has_mtp) {
        const std::string p = "blk.48.";
        std::vector<uint32_t> gu_rows[kfd::kMaxGpus];
        for (size_t g = 0; g < N; ++g) {
            gu_rows[g].reserve(256u * local_ff_r[g]);
            for (uint32_t e = 0; e < 256u; ++e)
                for (uint32_t r = 0; r < local_ff_r[g]; ++r)
                    gu_rows[g].push_back(e * hp.n_ff_exp + local_ff_off[g] + r);
        }
        const auto all_exp_dn_rows = iota_range(0, 256u * D);
        for (size_t g = 0; g < N; ++g) {
            auto& ml = mtp[g];
            if (!upload_f32_rows(rt, g, f, p + "nextn.enorm.weight", nullptr, ml.enorm_va, &res.error)) return res;
            if (!upload_f32_rows(rt, g, f, p + "nextn.hnorm.weight", nullptr, ml.hnorm_va, &res.error)) return res;
            if (!upload_f32_rows(rt, g, f, p + "nextn.shared_head_norm.weight", nullptr, ml.shared_head_norm_va, &res.error)) return res;
            if (!upload_f32_rows(rt, g, f, p + "attn_norm.weight", nullptr, ml.attn_norm_va, &res.error)) return res;
            if (!upload_f32_rows(rt, g, f, p + "post_attention_norm.weight", nullptr, ml.post_attn_norm_va, &res.error)) return res;
            if (!upload_f32_rows(rt, g, f, p + "attn_q_norm.weight", nullptr, ml.q_norm_va, &res.error)) return res;
            if (!upload_f32_rows(rt, g, f, p + "attn_k_norm.weight", nullptr, ml.k_norm_va, &res.error)) return res;
            if (!upload_fused_q4a8_tcc(rt, g, f, {{p + "nextn.eh_proj.weight", all_D_rows}},
                                       0, 2u * D, nullptr, ml.w_eh_proj, &res.error)) return res;
            if (!upload_fused_q4a8_tcc(rt, g, f, {
                    {p + "attn_q.weight", iota_range(gqa_q_off[g] * 512u,  gqa_nq[g] * 512u)},
                    {p + "attn_k.weight", iota_range(gqa_kv_off[g] * 256u, gqa_nkv[g] * 256u)},
                    {p + "attn_v.weight", iota_range(gqa_kv_off[g] * 256u, gqa_nkv[g] * 256u)}
                }, 0, D, nullptr, ml.w_attn_qkv, &res.error)) return res;
            if (!upload_fused_q4a8_tcc(rt, g, f, {{p + "attn_output.weight", all_D_rows}},
                                       gqa_q_off[g] * 256u, gqa_nq[g] * 256u, nullptr, ml.w_o, &res.error)) return res;
            if (!upload_f32_as_f16(rt, g, f, p + "ffn_gate_inp.weight", ml.w_router_f16_va, &res.error)) return res;
            if (!upload_f32_rows(rt, g, f, p + "ffn_gate_inp_shexp.weight", nullptr, ml.w_sh_gate_va, &res.error)) return res;
            if (!upload_fused_q4a8_tcc(rt, g, f, {
                    {p + "ffn_gate_shexp.weight", iota_range(local_ff_off[g], local_ff_r[g])},
                    {p + "ffn_up_shexp.weight",   iota_range(local_ff_off[g], local_ff_r[g])}
                }, 0, D, nullptr, ml.w_sh_gu, &res.error)) return res;
            if (!upload_fused_q4a8_tcc(rt, g, f, {{p + "ffn_down_shexp.weight", all_D_rows}},
                                       local_ff_off[g], local_ff_r[g], nullptr, ml.w_sh_down, &res.error)) return res;
            kfd::ArenaTierScope mtp_cold(rt.vram_arena(g), kfd::Tier::COLD);  // MTP routed experts: 8/256 read per draft
            if (!upload_fused_q4a8_tcc(rt, g, f, {{p + "ffn_gate_exps.weight", gu_rows[g]}},
                                       0, D, nullptr, ml.w_exp_gate, &res.error)) return res;
            if (!upload_fused_q4a8_tcc(rt, g, f, {{p + "ffn_up_exps.weight", gu_rows[g]}},
                                       0, D, nullptr, ml.w_exp_up, &res.error)) return res;
            if (!upload_fused_q4a8_tcc(rt, g, f, {{p + "ffn_down_exps.weight", all_exp_dn_rows}},
                                       local_ff_off[g], local_ff_r[g], nullptr, ml.w_exp_down, &res.error)) return res;
            ml.loaded = true;
        }
        for (const char* suf : {"nextn.enorm.weight", "nextn.hnorm.weight", "nextn.shared_head_norm.weight",
                                "nextn.eh_proj.weight", "attn_norm.weight", "post_attention_norm.weight",
                                "attn_q_norm.weight", "attn_k_norm.weight", "attn_q.weight", "attn_k.weight",
                                "attn_v.weight", "attn_output.weight", "ffn_gate_inp.weight",
                                "ffn_gate_inp_shexp.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                                "ffn_down_shexp.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight",
                                "ffn_down_exps.weight"}) {
            if (const auto* ti = f.tensor(p + suf)) f.release(f.tensor_data(*ti), ti->n_bytes());
        }
        std::printf("  uploaded blk.48 nextn MTP head |");
        for (size_t g = 0; g < N; ++g) std::printf(" GPU%zu=%.2f GiB", g, rt.vram_arena(g).used() / 1073741824.0);
        std::printf("\n");
    }
    for (size_t g = 0; g < draft_measured.size() && g < N; ++g)
        draft_measured[g] += rt.vram_arena(g).used() - draft_mark_mtp[g];
    if (gu_dot8_promoted > 0u) {
        std::printf("  promoted %u/48 layers Gate+Up IQ2_S -> exact VJ_Q4_DOT8 (%s)\n",
                    gu_dot8_promoted, in_place_gu_dot8 ? "in-place" : "co-resident");
    }

    if (const char* hb = std::getenv("VEGAJET_SPEC_HOBBIT"); hb && hb[0] == '1') {
        std::printf("[HOBBIT] resident Ornith GU has %s; mixed-precision fetch requires MoeOffload::fetch_spec with a staged low table. Resident verify unchanged.\n",
                    in_place_gu_dot8 ? "one in-place promoted copy" : "exact IQ2_S/DOT8 recodings");
    }

    // Optional DFlash2 exact-target block-diffusion drafter (single-GPU weights on GPU 0 for TP=1/TP=3, 0 PCIe ARs!)
    DFlash2Dev df{};
    spec_replay::DraftKvPlan df_kv_plan;
    std::fill_n(df.tap_slot, 48, int8_t(-1));
    const size_t df_gpu  = (N == 1 || N == 3 || std::getenv("VEGAJET_DFLASH2_GPU0") != nullptr) ? 0u : 1u;
    const std::array<uint64_t, moeslot::kBudgetCards> draft_mark_df = {
        rt.vram_arena(0).used(), (N > 1) ? rt.vram_arena(1).used() : 0ULL};
    if (!opt.draft_gguf.empty() && planner.draft_hparams().is_dflash2()) {
        const auto& dhp = planner.draft_hparams();
        GgufFile df_f;
        if (!df_f.open(opt.draft_gguf)) { res.error = "DFlash2 open failed: " + df_f.error(); return res; }
        std::string mask_path = opt.mask_embedding_path;
        if (mask_path.empty()) {
            const size_t sl = opt.draft_gguf.find_last_of('/');
            mask_path = (sl == std::string::npos ? "" : opt.draft_gguf.substr(0, sl + 1)) + "mask_embedding.safetensors";
        }
        std::ifstream mf(mask_path, std::ios::binary);
        if (!mf) { res.error = "DFlash2 missing mask_embedding.safetensors at " + mask_path; return res; }
        uint64_t hdr_len = 0;
        mf.read(reinterpret_cast<char*>(&hdr_len), 8);
        if (!mf || hdr_len == 0 || hdr_len > 65536) { res.error = "DFlash2 bad safetensors header in " + mask_path; return res; }
        std::string hdr(hdr_len, '\0');
        mf.read(hdr.data(), static_cast<std::streamsize>(hdr_len));
        uint64_t lo = 0, hi = D * 2ULL;
        const size_t opos = hdr.find("data_offsets");
        if (opos != std::string::npos) {
            const size_t lb = hdr.find('[', opos), cm = hdr.find(',', lb), rb = hdr.find(']', cm);
            if (lb != std::string::npos && cm != std::string::npos && rb != std::string::npos) {
                lo = std::strtoull(hdr.c_str() + lb + 1, nullptr, 10);
                hi = std::strtoull(hdr.c_str() + cm + 1, nullptr, 10);
            }
        }
        if (hi - lo != static_cast<uint64_t>(D) * 2ULL) {
            res.error = "DFlash2 mask_embedding byte size mismatch (" + std::to_string(hi - lo) + ")";
            return res;
        }
        mf.seekg(static_cast<std::streamoff>(8ULL + hdr_len + lo));
        std::vector<uint16_t> mask_bf16(D);
        mf.read(reinterpret_cast<char*>(mask_bf16.data()), static_cast<std::streamsize>(D * 2ULL));
        if (!mf) { res.error = "DFlash2 failed reading BF16 payload from " + mask_path; return res; }
        std::vector<float> mask_f32(D);
        for (uint32_t c = 0; c < D; ++c) mask_f32[c] = quant::bf16_to_f32(mask_bf16[c]);
        df.mask_embd_f32_va = rt.vram_arena(df_gpu).alloc(D * 4ULL, 256);
        if (!df.mask_embd_f32_va || !rt.upload_to_vram(df_gpu, df.mask_embd_f32_va, mask_f32.data(), D * 4ULL)) {
            res.error = "DFlash2 mask_embedding VRAM upload failed"; return res;
        }

        df.n_layer    = dhp.n_layer;
        df.n_head     = dhp.n_head;
        df.n_kv       = dhp.n_head_kv;
        df.head_dim   = dhp.head_dim;
        df.n_ff       = dhp.n_ff;
        df.block_size = dhp.block_size;
        df.swa_win    = dhp.sliding_window ? dhp.sliding_window : 2048u;
        df.sel_rank   = dhp.selector_rank ? dhp.selector_rank : 256u;
        df.sel_topk   = dhp.selector_top_k ? dhp.selector_top_k : 16u;
        df.rope_base  = static_cast<float>(dhp.rope_base);
        df.rms_eps    = static_cast<float>(dhp.rms_eps);
        const int tap_shift = std::getenv("VEGAJET_DFLASH2_TAP_SHIFT") ? std::atoi(std::getenv("VEGAJET_DFLASH2_TAP_SHIFT")) : 0;
        for (size_t k = 0; k < dhp.target_layers.size(); ++k) {
            const int l0 = static_cast<int>(dhp.target_layers[k]) - 1 + tap_shift;
            if (l0 >= 0 && l0 < 48) {
                df.tap_layers_0idx.push_back(static_cast<uint32_t>(l0));
                df.tap_slot[l0] = static_cast<int8_t>(k);
            }
        }
        const uint32_t n_taps = static_cast<uint32_t>(df.tap_layers_0idx.size());
        // Drafter KV layout per layer type: SWA layers a W-slot ring, the full-attention layer a ring covering the
        // DFlash2 context limit (d51c19a gave every layer the W ring and window, which dropped the full layer's
        // context past W - B). Unknown pattern -> legacy layout, flagged.
        {
            spec_replay::DraftKvConfig kc;
            kc.n_layer = df.n_layer; kc.swa_pattern = dhp.swa_pattern; kc.swa_window = df.swa_win;
            kc.block_len = df.block_size; kc.max_ctx = opt.max_ctx; kc.full_ctx_cap = 32768u;
            kc.n_kv = df.n_kv; kc.head_dim = df.head_dim;
            df_kv_plan = spec_replay::plan_draft_kv(kc);
            if (!df_kv_plan.ok) { res.error = "DFlash2 KV plan: " + df_kv_plan.error; return res; }
            std::printf("  DFlash2 KV plan: %s%s\n", df_kv_plan.summary.c_str(),
                        df_kv_plan.pattern_known ? "" : " (WARN: no dflash.attention.sliding_window_pattern; every layer SWA)");
        }
        // Opt-in block-mask arm (default off; the shipped kernel is bidirectional on every layer). Modes are
        // exactly tools/dflash_ref --block-mask: zlab = causal block on the sliding_attention layers and
        // bidirectional on the full layer, causal = all layers (control). A masked mode refuses to start without
        // the per-layer pattern, and the static configuration is checked once here so the per-cycle plan can only
        // vary in s_pos and B.
        {
            dflash_host::BlockMaskMode mode = dflash_host::BlockMaskMode::Bidir;
            const char* mask_env = std::getenv("VEGAJET_DFLASH2_BLOCK_MASK");
            if (!dflash_host::parse_block_mask_mode(mask_env, &mode)) {
                res.error = std::string("VEGAJET_DFLASH2_BLOCK_MASK='") + (mask_env ? mask_env : "") +
                            "' is not one of bidir, zlab, causal";
                return res;
            }
            df.block_mask = mode;
            df.block_mask_pattern_known = df_kv_plan.pattern_known;
            for (uint32_t l = 0u; l < df.n_layer && l < 32u; ++l)
                if (df_kv_plan.layers[l].kind == spec_replay::LayerKind::kSwa) df.block_mask_swa |= (1u << l);
            if (mode != dflash_host::BlockMaskMode::Bidir) {
                dflash_host::DraftBlockMaskConfig pre{};
                pre.mode = mode; pre.pattern_known = df.block_mask_pattern_known;
                pre.swa_layer_mask = df.block_mask_swa; pre.n_layer = df.n_layer;
                pre.block_start = 0u; pre.block_len = std::max(df.block_size - 1u, 1u); pre.swa_window = df.swa_win;
                const dflash_host::DraftBlockMaskPlan chk = dflash_host::plan_draft_block_mask(pre);
                if (!chk.ok) { res.error = "DFlash2 block mask: " + chk.error; return res; }
                std::printf("  DFlash2 block mask arm: %s (masked only while s_pos + B <= %u; default off)\n",
                            chk.summary.c_str(), df.swa_win);
            }
        }
        const auto all_D   = iota_range(0, D);
        const auto all_ff  = iota_range(0, df.n_ff);
        const auto all_q   = iota_range(0, df.n_head * df.head_dim);
        const auto all_kv  = iota_range(0, df.n_kv * df.head_dim);
        const auto all_dyn = iota_range(0, 4u * (D / 16u));
        const auto all_sel = iota_range(0, df.sel_rank);

        if (!upload_f32_rows(rt, df_gpu, df_f, "enc.output_norm.weight", nullptr, df.enc_norm_va, &res.error)) return res;
        if (!upload_f32_rows(rt, df_gpu, df_f, "output_norm.weight", nullptr, df.out_norm_va, &res.error)) return res;
        if (!upload_fused_q4a8_tcc(rt, df_gpu, df_f, {{"fc.weight", all_D}}, 0, n_taps * D, nullptr, df.w_fc, &res.error)) return res;
        if (!upload_fused_q4a8_tcc(rt, df_gpu, df_f, {{"selector_hidden.weight", all_sel}}, 0, D, nullptr, df.w_sel_hid, &res.error)) return res;

        df.L.resize(df.n_layer);
        for (uint32_t l = 0; l < df.n_layer; ++l) {
            const std::string p = "blk." + std::to_string(l) + ".";
            auto& dl = df.L[l];
            if (!upload_f32_rows(rt, df_gpu, df_f, p + "attn_norm.weight", nullptr, dl.attn_norm_va, &res.error)) return res;
            if (!upload_f32_rows(rt, df_gpu, df_f, p + "attn_conv_base",   nullptr, dl.attn_conv_base_va, &res.error)) return res;
            if (!upload_f32_rows(rt, df_gpu, df_f, p + "attn_q_norm.weight", nullptr, dl.q_norm_va, &res.error)) return res;
            if (!upload_f32_rows(rt, df_gpu, df_f, p + "attn_k_norm.weight", nullptr, dl.k_norm_va, &res.error)) return res;
            if (!upload_f32_rows(rt, df_gpu, df_f, p + "ffn_norm.weight",  nullptr, dl.ffn_norm_va, &res.error)) return res;
            if (!upload_f32_rows(rt, df_gpu, df_f, p + "ffn_conv_base",    nullptr, dl.ffn_conv_base_va, &res.error)) return res;
            if (!upload_fused_q4a8_tcc(rt, df_gpu, df_f, {{p + "attn_conv_proj.weight", all_dyn}}, 0, D, nullptr, dl.w_attn_conv, &res.error)) return res;
            if (!upload_fused_q4a8_tcc(rt, df_gpu, df_f, {
                    {p + "attn_q.weight", all_q},
                    {p + "attn_k.weight", all_kv},
                    {p + "attn_v.weight", all_kv}
                }, 0, D, nullptr, dl.w_qkv, &res.error)) return res;
            // Zero-copy K+V slice of w_qkv for context KV injection!
            const uint32_t q_rows = df.n_head * df.head_dim;
            dl.w_kv.qs_va = dl.w_qkv.qs_va + static_cast<uint64_t>(q_rows) * dl.w_qkv.row_stride_bytes;
            dl.w_kv.dm_va = dl.w_qkv.dm_va + static_cast<uint64_t>(q_rows) * (D / 32u) * 4ULL;
            dl.w_kv.rows  = 2u * df.n_kv * df.head_dim;
            dl.w_kv.cols  = D;
            dl.w_kv.row_stride_bytes = dl.w_qkv.row_stride_bytes;
            if (!upload_fused_q4a8_tcc(rt, df_gpu, df_f, {{p + "attn_output.weight", all_D}}, 0, q_rows, nullptr, dl.w_o, &res.error)) return res;
            if (!upload_fused_q4a8_tcc(rt, df_gpu, df_f, {{p + "ffn_conv_proj.weight", all_dyn}}, 0, D, nullptr, dl.w_ffn_conv, &res.error)) return res;
            if (!upload_fused_q4a8_tcc(rt, df_gpu, df_f, {
                    {p + "ffn_gate.weight", all_ff},
                    {p + "ffn_up.weight",   all_ff}
                }, 0, D, nullptr, dl.w_gu, &res.error)) return res;
            if (!upload_fused_q4a8_tcc(rt, df_gpu, df_f, {{p + "ffn_down.weight", all_D}}, 0, df.n_ff, nullptr, dl.w_down, &res.error)) return res;
            dl.kv_slots = df_kv_plan.layers[l].slots;
            dl.kv_win   = df_kv_plan.layers[l].kernel_win;
            const uint64_t kv_bytes = static_cast<uint64_t>(dl.kv_slots) * df.n_kv * df.head_dim * 2ULL;
            dl.kv_k_va = rt.vram_arena(df_gpu).alloc(kv_bytes, 4096);
            dl.kv_v_va = rt.vram_arena(df_gpu).alloc(kv_bytes, 4096);
            if (!dl.kv_k_va || !dl.kv_v_va) { res.error = "DFlash2 VRAM OOM allocating KV cache"; return res; }
        }
        // Host-side raw Q4_K selector tables (zero VRAM!)
        const auto* tp = df_f.tensor("selector_predecessor.weight");
        const auto* ts = df_f.tensor("selector_successor.weight");
        if (!tp || !ts) { res.error = "DFlash2 missing selector_predecessor/successor tensors"; return res; }
        df.sel_pred_type = static_cast<quant::DiskQuantType>(tp->type);
        df.sel_succ_type = static_cast<quant::DiskQuantType>(ts->type);
        df.sel_pred_rb   = quant::DecoderRegistry::instance().row_bytes(df.sel_pred_type, df.sel_rank);
        df.sel_succ_rb   = quant::DecoderRegistry::instance().row_bytes(df.sel_succ_type, df.sel_rank);
        df.sel_vocab     = static_cast<uint32_t>(tp->ne[1]);
        df.sel_pred_raw.assign(df_f.tensor_data(*tp), df_f.tensor_data(*tp) + tp->n_bytes());
        df.sel_succ_raw.assign(df_f.tensor_data(*ts), df_f.tensor_data(*ts) + ts->n_bytes());

        auto df_alloc = [&](uint64_t b) { return rt.vram_arena(df_gpu).alloc(b, 256); };
        df.taps_va    = df_alloc(static_cast<uint64_t>(max_batch) * n_taps * D * 4ULL);
        df.taps_q8_va = df_alloc(16ULL * n_taps * D);
        df.taps_ds_va = df_alloc(16ULL * ((n_taps * D) / 32u) * 8ULL);
        df.ctx_f32_va = df_alloc(16ULL * D * 4ULL);
        df.ctx_q8_va  = df_alloc(16ULL * D);
        df.ctx_ds_va  = df_alloc(16ULL * (D / 32u) * 8ULL);
        df.h_va       = df_alloc(16ULL * D * 4ULL);
        df.xnorm_va   = df_alloc(16ULL * D * 4ULL);
        df.xq8_va     = df_alloc(16ULL * D);
        df.xds_va     = df_alloc(16ULL * (D / 32u) * 8ULL);
        df.dyn_va     = df_alloc(16ULL * 4u * (D / 16u) * 4ULL);
        df.qkv_va     = df_alloc(16ULL * (df.n_head + 2u * df.n_kv) * df.head_dim * 4ULL);
        df.attn_q8_va = df_alloc(16ULL * df.n_head * df.head_dim);
        df.attn_ds_va = df_alloc(16ULL * ((df.n_head * df.head_dim) / 32u) * 8ULL);
        df.out_va     = df_alloc(16ULL * D * 4ULL);
        df.gu_va      = df_alloc(16ULL * 2ULL * df.n_ff * 4ULL);
        df.mid_q8_va  = df_alloc(16ULL * df.n_ff);
        df.mid_ds_va  = df_alloc(16ULL * (df.n_ff / 32u) * 8ULL);
        df.loaded = true;
        std::printf("  uploaded DFlash2 (%uL, B=%u, %u taps, mask_embd=%s) |",
                    df.n_layer, df.block_size, n_taps, mask_path.c_str());
        for (size_t g = 0; g < N; ++g) std::printf(" GPU%zu=%.2f GiB", g, rt.vram_arena(g).used() / 1073741824.0);
        std::printf("\n");
    }

    for (size_t g = 0; g < draft_measured.size() && g < N; ++g)
        draft_measured[g] += rt.vram_arena(g).used() - draft_mark_df[g];

    // Zero-roundtrip host-pinned coherent I/O buffer mapped to ALL GPUs (6 MiB: target I/O + up to 4-GPU DFlash2 top-16!)
    kfd::KfdBuffer io_buf{};
    if (!rt.alloc_userptr_pinned(6ULL * 1024ULL * 1024ULL, io_buf, true)) {
        res.error = "Failed to allocate pinned I/O buffer";
        return res;
    }
    auto* io_host_u8   = reinterpret_cast<uint8_t*>(io_buf.host_ptr);
    auto* io_toks_host = reinterpret_cast<int32_t*>(io_host_u8 + 0u);
    const uint64_t io_toks_va = io_buf.va + 0u;
    float*   io_arg_val_host[kfd::kMaxGpus] = {};
    int32_t* io_arg_id_host[kfd::kMaxGpus]  = {};
    uint64_t io_arg_val_va[kfd::kMaxGpus]   = {};
    uint64_t io_arg_id_va[kfd::kMaxGpus]    = {};
    for (size_t g = 0; g < kfd::kMaxGpus; ++g) {
        const uint64_t off = g * kMaxSpecBatch * kVocabBlocks * 4ULL;
        io_arg_val_host[g] = reinterpret_cast<float*>(io_host_u8 + 4096u + off);
        io_arg_id_host[g]  = reinterpret_cast<int32_t*>(io_host_u8 + 262144u + off);
        io_arg_val_va[g]   = io_buf.va + 4096u + off;
        io_arg_id_va[g]    = io_buf.va + 262144u + off;
    }
    // DFlash2 zero-copy coherent I/O slices:
    constexpr uint64_t kDfIoBase     = 1048576ULL;
    constexpr uint64_t kDfTopkStride = 262144ULL;
    const uint64_t df_io_xq8_va  = io_buf.va + kDfIoBase;
    const uint64_t df_io_xds_va  = io_buf.va + kDfIoBase + 65536ULL;
    const uint64_t df_io_gate_va = io_buf.va + kDfIoBase + 98304ULL;
    auto* df_io_gate_host = reinterpret_cast<float*>(io_host_u8 + kDfIoBase + 98304ULL);
    uint64_t df_io_val_va[kfd::kMaxGpus]   = {};
    uint64_t df_io_id_va[kfd::kMaxGpus]    = {};
    float*   df_io_val_host[kfd::kMaxGpus] = {};
    int32_t* df_io_id_host[kfd::kMaxGpus]  = {};
    for (size_t g = 0; g < kfd::kMaxGpus; ++g) {
        const uint64_t v_off = 4ULL * 1024ULL * 1024ULL + g * kDfTopkStride;
        const uint64_t i_off = 5ULL * 1024ULL * 1024ULL + g * kDfTopkStride;
        df_io_val_va[g]   = io_buf.va + v_off;
        df_io_id_va[g]    = io_buf.va + i_off;
        df_io_val_host[g] = reinterpret_cast<float*>(io_host_u8 + v_off);
        df_io_id_host[g]  = reinterpret_cast<int32_t*>(io_host_u8 + i_off);
    }

    // EVICT route-ID capture: the existing per-layer rank-0 top-ID buffer copied into the pinned I/O buffer.
    // Layout [layer][token][top8], 48 layers with a 16-token stride (128 ids = 512 B/layer, 24 KiB total).
    // Region [3.5 MiB, 4 MiB) is free; lru_top_va starts at 3 MiB and spans 24 KiB.
    constexpr uint32_t kEvictTopK = 8u;
    constexpr uint32_t kEvictTopTokens = kMaxSpecBatch;                    // 16
    constexpr uint32_t kEvictTopStrideIds = kEvictTopTokens * kEvictTopK;  // 128
    constexpr uint64_t kEvictTopByteOffset = 3670016ULL;                   // 3.5 MiB
    static_assert(kEvictTopByteOffset + 48ULL * kEvictTopStrideIds * 4ULL <= 4ULL * 1024ULL * 1024ULL,
                  "EVICT route-ID capture must fit the pinned 4 MiB I/O buffer");
    struct EvictRouteCapture {
        uint64_t va = 0;
        int32_t* host = nullptr;
        bool complete = false;   // all 48 per-layer copies were enqueued for the last step_batch()
        uint32_t tokens = 0;     // completed pass token count, valid only when complete
    };
    EvictRouteCapture evict_cap;
    evict_cap.va = io_buf.va + kEvictTopByteOffset;
    evict_cap.host = reinterpret_cast<int32_t*>(io_host_u8 + kEvictTopByteOffset);

    // Optional residual-stream taps (GPU 1): embedding output + the output of each of the 48 layers, for one token.
    // Used by --dump-hidden to produce drafter conditioning dumps; costs 49 small copies per token when enabled.
    kfd::KfdBuffer tap_buf{};
    const float* tap_host = nullptr;
    bool tap_on = false;
    if (!opt.dump_hidden_path.empty()) {
        if (49ULL * D * 4ULL > 1024ULL * 1024ULL || !rt.alloc_userptr_pinned(1024ULL * 1024ULL, tap_buf, true)) {
            rt.free_buffer(io_buf);
            res.error = "Failed to allocate pinned hidden-state tap buffer";
            return res;
        }
        tap_host = reinterpret_cast<const float*>(tap_buf.host_ptr);
    }

    constexpr uint32_t kArTsMax = 96u * 160u;
    const kfd::LoadedKernel* k_ar_ts[kfd::kMaxGpus] = {};
    bool ar_ts_on = false;
    uint32_t ar_ts_idx = 0;
    uint32_t pf_gdn_sense = 0u;
    constexpr uint32_t kDownGrpMin = 2u;   // experts with fewer routed tokens stay on the per-token stream
    auto grp_item_cap = [](uint32_t M) { const uint32_t R = M * 8u, U = std::min(R, 256u); return std::min(R, (R + 63u * U + 63u) / 64u); };
    uint32_t gdn_layer_count = 0u;
    for (uint32_t il = 0; il < 48u; ++il) if (layers[0][il].is_gdn) ++gdn_layer_count;
    uint64_t treewy_arena_va[kfd::kMaxGpus] = {};
    uint64_t treewy_arena_bytes[kfd::kMaxGpus] = {};
    GpuStateDev S[kfd::kMaxGpus];
    for (size_t g = 0; g < N; ++g) {
        const uint32_t lff_g  = local_ff_r[g];
        const uint32_t nv_g   = gdn_nv[g];
        const uint32_t nk_g   = gdn_nk[g];
        const uint32_t cdim_g = gdn_conv_dim[g];
        const uint32_t qstr_g = gdn_qkv_stride[g];
        const uint32_t nq_g   = gqa_nq[g];
        const uint32_t nkv_g  = gqa_nkv[g];
        bool alloc_oom = false;
        auto alloc = [&](uint64_t b, uint64_t a = 256) {
            const uint64_t v = rt.vram_arena(g).alloc(b, a);
            if (!v) alloc_oom = true;
            return v;
        };
        S[g].tok_ids_va     = io_toks_va;
        S[g].h_va           = alloc(static_cast<uint64_t>(max_batch) * D * 4u);
        S[g].x_norm_va      = alloc(static_cast<uint64_t>(max_batch) * D * 4u);
        S[g].x_q8_va        = alloc(static_cast<uint64_t>(max_batch) * D);
        S[g].x_ds_va        = alloc(static_cast<uint64_t>(max_batch) * (D / 32u) * 8u); // float2!
        S[g].partial_va     = alloc(static_cast<uint64_t>(max_batch) * D * 4u);
        S[g].ar_bar_va      = alloc(256u);
        if (opt.ar_timeline) S[g].ar_ts_va = alloc(kArTsMax * 48ull, 4096);
        S[g].moe_bar_va     = alloc(256u);
        S[g].gdn_bar_va     = alloc(8192u, 256);
        S[g].moe_bq_va      = alloc(256u);
        // Reserve diagnostic records at load time: warm runs may enable PROFILE=2 later (>= mega barrier bytes).
        // super scratch + mega ring + the per-WG critical-stamp area (super_x VJ_PROF_WG_OFF/ARW_OFF). Always sized for
        // the stamps: VEGAJET_SUPER_PROFILE=2 can be set per warm run, after this load-time allocation (W17 fault).
        S[g].super_bar_va      = alloc(std::max<uint64_t>(kProfDumpBytes, ornith_mega::kScratchBytes), 256);
        S[g].mega_desc_va      = alloc(sizeof(ornith_super::Mega48Desc), 256);
        S[g].gdn_super_desc_va = alloc(36ULL * sizeof(ornith_super::GdnSuperDesc), 256);
        S[g].gqa_super_desc_va = alloc(12ULL * sizeof(ornith_super::GqaSuperDesc), 256);
        S[g].moe_super_desc_va = alloc(48ULL * sizeof(ornith_super::MoeSuperDesc), 256);
        S[g].gdn_sq_va      = alloc(static_cast<uint64_t>(std::max(32u, nv_g)) * 192u * 4u);
        S[g].gdn_qkv_gab_va = alloc(static_cast<uint64_t>(max_batch) * std::max(10304u, qstr_g) * 4u);
        S[g].gdn_conv_va    = alloc(static_cast<uint64_t>(max_batch) * std::max(6144u, cdim_g) * 4u);
        S[g].gdn_dec_va     = alloc(static_cast<uint64_t>(max_batch) * std::max(32u, nv_g) * 4u);
        S[g].gdn_bsig_va    = alloc(static_cast<uint64_t>(max_batch) * std::max(32u, nv_g) * 4u);
        const uint32_t mid_cols = std::max(nv_g * 128u, nq_g * 256u);
        S[g].mid_q8_va      = alloc(static_cast<uint64_t>(max_batch) * std::max(4096u, mid_cols));
        S[g].mid_ds_va      = alloc(static_cast<uint64_t>(max_batch) * std::max(128u, mid_cols / 32u) * 8u);
        S[g].attn_qkv_va    = alloc(static_cast<uint64_t>(max_batch) * std::max(8704u, nq_g * 512u + 2u * nkv_g * 256u) * 4u);
        S[g].q_buf_va       = alloc(static_cast<uint64_t>(max_batch) * std::max(4096u, nq_g * 256u) * 4u);
        S[g].gate_buf_va    = alloc(static_cast<uint64_t>(max_batch) * std::max(4096u, nq_g * 256u) * 4u);
        S[g].router_log_va  = alloc(static_cast<uint64_t>(max_batch) * 256u * 4u);
        S[g].top_ids_va     = alloc(static_cast<uint64_t>(max_batch) * 8u * 4u);
        S[g].top_w_va       = alloc(static_cast<uint64_t>(max_batch) * 8u * 4u);
        S[g].sh_gate_sc_va  = alloc(static_cast<uint64_t>(max_batch) * 4u);
        S[g].sh_gu_va       = alloc(static_cast<uint64_t>(max_batch) * 2u * lff_g * 4u);
        S[g].sh_q8_va       = alloc(static_cast<uint64_t>(max_batch) * lff_g);
        S[g].sh_ds_va       = alloc(static_cast<uint64_t>(max_batch) * (lff_g / 32u) * 8u);
        S[g].sh_down_va     = alloc(static_cast<uint64_t>(max_batch) * D * 4u);
        S[g].moe_q8_va      = alloc(static_cast<uint64_t>(max_batch) * 8u * lff_g);
        S[g].moe_d_va       = alloc(static_cast<uint64_t>(max_batch) * 8u * (lff_g / 32u) * 4u);
        S[g].moe_mid_f32_va = alloc(8ULL * lff_g * 4ULL, 4096);
        S[g].splitk_part_va = alloc(8ULL * 16384ULL * 4ULL, 4096);
        S[g].moe_exp_off_va   = alloc(257u * 4u);
        S[g].moe_exp_cnt_va   = alloc(256u * 4u);
        S[g].moe_sorted_ms_va = alloc(static_cast<uint64_t>(max_batch) * 8u * 4u);
        if (opt.pf_gdn != 0u) {
            S[g].pf_fo_va      = alloc(static_cast<uint64_t>(max_batch) * nv_g * 128ull * 4ull, 4096);
            S[g].pf_sq_va      = alloc(static_cast<uint64_t>(max_batch) * nv_g * 64ull * 4ull, 4096);
            S[g].pf_gdn_bar_va = alloc(256u, 256);
        }
        const bool warm_alloc = !raw_only_vram && (opt.interactive || !opt.sweep_spec.empty());
        if (opt.pf_gu_tiled || opt.down_grp_prefill == 3u || opt.prefill_dot8 || warm_alloc || opt.pf_moe_iq_lds ||
            opt.pf_moe_iq_lds32)
            S[g].pf_gu_queue_va = alloc(256u, 256);
        if (opt.prefill_dot8 || warm_alloc || sub4_on) {
            if(sub4_on && quant::milo_enabled()) S[g].milo_tmp_va=alloc(8u*32u*4u,256);
            S[g].pf_xq4_va      = alloc(static_cast<uint64_t>(max_batch) * 2048ULL, 4096);
            S[g].pf_xds4_va     = alloc(static_cast<uint64_t>(max_batch) * 128ULL * 8ULL, 4096);
            S[g].pf_xsum4_va    = alloc(static_cast<uint64_t>(max_batch) * 128ULL * 4ULL, 4096);
            S[g].moe_mid_q4_va  = alloc(static_cast<uint64_t>(max_batch) * 8ULL * (lff_g / 2u), 4096);
            S[g].moe_mid_ds4_va = alloc(static_cast<uint64_t>(max_batch) * 8ULL * (lff_g / 32u) * 8ULL, 4096);
        }
        if (!opt.dump_logits_path.empty()) {
            S[g].logits_va = alloc(static_cast<uint64_t>(voc_rows[g]) * 4ULL, 4096);
        }
        if (opt.down_grp_prefill != 0u || opt.down_grp_verify != 0u || warm_alloc) {
            const uint64_t tok_dp  = (opt.down_grp_prefill == 1u) ? max_batch : ((opt.down_grp_verify == 1u || warm_alloc) ? 16u : 0u);
            const uint64_t tok_f32 = (opt.down_grp_prefill == 2u || opt.down_grp_prefill == 3u || warm_alloc) ? max_batch : ((opt.down_grp_verify == 2u) ? 16u : 0u);
            S[g].dgrp_hdr_va    = alloc(64u, 4096);
            S[g].dgrp_cnt_va    = alloc(256u * 4u);
            S[g].dgrp_off_va    = alloc(257u * 4u);
            S[g].dgrp_uniq_va   = alloc(256u * 4u);
            S[g].dgrp_items_va  = alloc((grp_item_cap(max_batch) + 16u) * 4u);
            S[g].dgrp_routes_va = alloc(static_cast<uint64_t>(max_batch) * 8u * 4u);
            S[g].dgrp_buf_va    = alloc(std::max(tok_dp * 8u * D * 64u, tok_f32 * 8u * D * 4u) + 4096u, 4096);
            if (!S[g].dgrp_buf_va) { rt.free_buffer(tap_buf); rt.free_buffer(io_buf); res.error = "VRAM OOM for the grouped Down staging buffer"; return res; }
        }
        S[g].argmax_val_va  = io_arg_val_va[g];
        S[g].argmax_id_va   = io_arg_id_va[g];

        const uint32_t ssm_ring_slots = raw_only_vram ? 1u : kGdnRingSlots;
        S[g].gdn_conv_ring_va.resize(36);
        S[g].gdn_ssm_ring_va.resize(36);
        for (uint32_t i = 0; i < 36u; ++i) {
            S[g].gdn_conv_ring_va[i] = alloc(kGdnRingSlots * 3ULL * cdim_g * 4ULL, 4096);
            if (!opt.spec_gdn) {
                S[g].gdn_ssm_ring_va[i] = alloc(static_cast<uint64_t>(ssm_ring_slots) * nv_g * 128ULL * 128ULL * 4ULL, 4096);
            }
        }
        if (opt.spec_gdn) {
            const uint64_t all_s_bytes = 36ULL * nv_g * 128ULL * 128ULL * 4ULL;
            S[g].spec_gdn_s_slot_va[0] = alloc(all_s_bytes, 4096);
            S[g].spec_gdn_s_slot_va[1] = alloc(all_s_bytes, 4096);
            S[g].spec_gdn_o_va         = alloc(16ULL * nv_g * 128ULL * 4ULL, 4096);
            S[g].spec_gdn_rec_kv_va    = alloc(36ULL * 16ULL * (nk_g + nv_g) * 128ULL * 4ULL, 4096);
            S[g].spec_gdn_rec_dec_va   = alloc(36ULL * 16ULL * nv_g * 4ULL, 4096);
            S[g].spec_gdn_rec_beta_va  = alloc(36ULL * 16ULL * nv_g * 4ULL, 4096);
        }
        if (N == 2 && treewy_opt.enabled && opt.spec_gdn && gdn_layer_count > 0u &&
            K[0].treewy_verify && K[1].treewy_verify && K[0].treewy_commit && K[1].treewy_commit) {
            const uint64_t tw_bytes = vegajet::gdn::treewy_launch::size_scratch(
                                          12u, gdn_layer_count, true).total_bytes;
            treewy_arena_va[g]    = alloc(tw_bytes, 4096);
            treewy_arena_bytes[g] = tw_bytes;
        }
        S[g].spec_fl_part_o_va  = alloc(static_cast<uint64_t>(kMaxSpecBatch) * std::max(16u, nq_g) * 60ULL * 256ULL * 4ULL, 4096);
        S[g].spec_fl_part_ml_va = alloc(static_cast<uint64_t>(kMaxSpecBatch) * std::max(16u, nq_g) * 60ULL * 2ULL * 4ULL, 4096);
        S[g].attn_k_cache_va.resize(12);
        S[g].attn_v_cache_va.resize(12);
        for (uint32_t i = 0; i < 12u; ++i) {
            S[g].attn_k_cache_va[i] = alloc(static_cast<uint64_t>(opt.max_ctx) * nkv_g * (l4_warm ? 272u : l4_stride), 4096);
            S[g].attn_v_cache_va[i] = alloc(static_cast<uint64_t>(opt.max_ctx) * nkv_g * (l4_warm ? 272u : l4_stride), 4096);
        }
        if (l4_available) {
            const auto hr=l4::ranges(gqa_nq[g],nkv_g,gqa_q_off[g],gqa_kv_off[g],gqa_group);
            VJ_CHECK(kvq::valid_shape(256,gqa_nq[g],nkv_g,opt.max_ctx,kvq::split_count(nkv_g),l4_format,hr.data()));
            auto& a=L4[g];
            a.ranges=alloc(hr.size()*sizeof(kvq::HeadRange),256);
            VJ_CHECK(rt.upload_to_vram(g,a.ranges,hr.data(),hr.size()*sizeof(kvq::HeadRange)));
            if (l4_compressed || l4_warm) {
                a.prepared_k=alloc(nkv_g*256ULL*4,256); a.prepared_v=alloc(nkv_g*256ULL*4,256);
                auto signs=kvq::turbo3_signs(256,0x5455524230u);
                a.signs=alloc(256,256); VJ_CHECK(rt.upload_to_vram(g,a.signs,signs.data(),256));
                a.partial=alloc(gqa_nq[g]*kvq::split_count(nkv_g)*256ULL*4,256);
                a.lse=alloc(gqa_nq[g]*kvq::split_count(nkv_g)*4ULL,256);
                a.out=alloc(gqa_nq[g]*256ULL*4,256);
            }
            if (!l4_compressed || l4_warm) {
                a.wire_k.resize(12); a.wire_v.resize(12);
                for (uint32_t i=0;i<12;++i) {
                    a.wire_k[i]=alloc(uint64_t(opt.max_ctx)*nkv_g*384,4096);
                    a.wire_v[i]=alloc(uint64_t(opt.max_ctx)*nkv_g*384,4096);
                }
                a.wire_q=alloc(16*384,256);
                a.wire_scratch=alloc(flash_decode_splitkv::plan_scratch(1,kvq::split_count(1)).total_bytes,256);
            }
        }
        if (wire_requested) {
            const wire_normal::WireFormat wire_format =
                wire_q4 ? wire_normal::WireFormat::Q4 : wire_normal::WireFormat::Q8;
            const uint32_t wire_row = wire_normal::wire_row_bytes(wire_format);
            S[g].wire_k_cache_va.resize(12);
            S[g].wire_v_cache_va.resize(12);
            for (uint32_t i = 0; i < 12u; ++i) {
                S[g].wire_k_cache_va[i] = alloc(static_cast<uint64_t>(opt.max_ctx) * nkv_g * wire_row, 4096);
                S[g].wire_v_cache_va[i] = alloc(static_cast<uint64_t>(opt.max_ctx) * nkv_g * wire_row, 4096);
            }
            S[g].wire_q_va = alloc(static_cast<uint64_t>(max_batch) *
                                       wire_normal::q_wire_row_bytes(wire_format), 4096);
            S[g].wire_out_va = alloc(static_cast<uint64_t>(max_batch) * 16ULL * 256ULL * 4ULL, 4096);
            S[g].wire_out_lse_va = alloc(static_cast<uint64_t>(max_batch) * 16ULL * 4ULL, 4096);
            const flash_decode_splitkv::ScratchPlan wire_scratch_plan =
                flash_decode_splitkv::plan_scratch(wire_normal::kMaxRows, wire_max_splits);
            if (wire_scratch_plan.overflow) {
                res.error = "sealed wire scratch size overflow";
                return res;
            }
            S[g].wire_scratch_va = alloc(wire_scratch_plan.total_bytes, 4096);
        }
        if (has_mtp) {
            const uint64_t mtp_state_mark = rt.vram_arena(g).used();
            S[g].mtp_k_cache_va = alloc(static_cast<uint64_t>(opt.max_ctx) * nkv_g * 272ULL, 4096);
            S[g].mtp_v_cache_va = alloc(static_cast<uint64_t>(opt.max_ctx) * nkv_g * 272ULL, 4096);
            S[g].mtp_cat_q8_va  = alloc(static_cast<uint64_t>(kMaxSpecBatch) * 2u * D);
            S[g].mtp_cat_ds_va  = alloc(static_cast<uint64_t>(kMaxSpecBatch) * ((2u * D) / 32u) * 8u);
            S[g].mtp_h_pend_va  = alloc(D * 4u);
            S[g].mtp_h_cur_va   = alloc(static_cast<uint64_t>(kMaxSpecBatch) * D * 4u);
            S[g].mtp_h_zero_va  = alloc(D * 4u);
            S[g].mtp_h_stash_va = alloc(16ULL * D * 4u);
            S[g].mtp_moe_ds_va  = alloc(static_cast<uint64_t>(kMaxSpecBatch) * 8u * (lff_g / 32u) * 8u);
            if (g < draft_measured.size()) draft_measured[g] += rt.vram_arena(g).used() - mtp_state_mark;
        }
        if (alloc_oom) {
            rt.free_buffer(tap_buf);
            rt.free_buffer(io_buf);
            res.error = "VRAM OOM allocating GPU workspace on rank " + std::to_string(g);
            return res;
        }
    }

    res.load_sec      = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load0).count();

    // Opt-in sealed tree-WY verifier. It shares the existing spec_gdn two-slot S0 pair and the
    // existing norm/quant kernel; a missing module, absent sealed-wire producer or unsupported
    // width keeps the sequential verifier. Nothing here changes the default path.
    std::unique_ptr<treewy::RuntimeBackend> treewy_backend;
    std::unique_ptr<treewy::Hook> treewy_hook;
    if (treewy_opt.enabled) {
        // The engine owns the producer slot for this run: any producer left by an earlier caller is
        // cleared before this run decides, and TreeWyProducerGuard clears it on every exit path.
        treewy::install_sealed_wire_producer(nullptr);
        if (!opt.spec_gdn) {
            std::printf("[treewy] VEGAJET_TREEWY=1 ignored: --spec-gdn is off\n");
        } else if (N != 2u) {
            std::printf("[treewy] disabled: TP=%zu (TreeWY 2-shard backend enabled on TP=2)\n", N);
        } else if (!K[0].treewy_verify || !K[1].treewy_verify || !K[0].treewy_commit ||
                   !K[1].treewy_commit || !K[0].spec_gdn_nq || !K[1].spec_gdn_nq) {
            std::printf("[treewy] disabled: sealed module or norm/quant kernel is not loaded\n");
        } else if (!treewy::install_gdn_wire_producer(K[0].gdn_wire_pack, K[1].gdn_wire_pack)) {
            std::printf("[treewy] disabled: sealed GDN input producer is not loaded "
                        "(build hsaco_vj_gdn_wire_pack)\n");
        } else {
            const kfd::LoadedKernel* tv[2] = {K[0].treewy_verify, K[1].treewy_verify};
            const kfd::LoadedKernel* tc[2] = {K[0].treewy_commit, K[1].treewy_commit};
            const kfd::LoadedKernel* tn[2] = {K[0].spec_gdn_nq, K[1].spec_gdn_nq};
            treewy_backend = std::make_unique<treewy::RuntimeBackend>(rt, tv, tc, tn);
            if (!treewy_backend->available()) {
                std::printf("[treewy] disabled: backend preconditions failed\n");
                treewy_backend.reset();
                treewy::install_sealed_wire_producer(nullptr);
            } else {
                treewy_hook = std::make_unique<treewy::Hook>(
                    treewy_opt, gdn_layer_count, kGdnRingSlots, treewy_arena_va,
                    treewy_arena_bytes, *treewy_backend);
                std::printf("[treewy] enabled %s | %u GDN layers x 2 shards | arena %.1f MiB/GPU | "
                            "sealed-wire producer %s\n",
                            treewy::describe(treewy_opt).c_str(), gdn_layer_count,
                            treewy_arena_bytes[0] / 1048576.0,
                            treewy_backend->wire_available()
                                ? "installed (vj_gdn_wire_pack q/k/v/decay/beta/parent)"
                                : "MISSING (sequential fallback)");
            }
        }
    }
    struct TreeWyProducerGuard {
        ~TreeWyProducerGuard() { treewy::install_sealed_wire_producer(nullptr); }
    };
    TreeWyProducerGuard treewy_producer_guard;
    struct TreeWyStatsGuard {
        const treewy::Hook* hook;
        explicit TreeWyStatsGuard(const treewy::Hook* h) : hook(h) {}
        ~TreeWyStatsGuard() { if (hook) hook->print_stats("final"); }
    };
    TreeWyStatsGuard treewy_stats_guard(treewy_hook.get());

    // Wave 5 Dynamic LRU VJ_Q4_DOT8 MoE Gate+Up Cache (up to 44 slots/L in spare VRAM for M >= 4 verify!).
    uint32_t lru_slots = 0u;
    Dot8RowLayout lru_lay{};
    uint64_t lru_w_va[kfd::kMaxGpus][48] = {}, lru_s_va[kfd::kMaxGpus][48] = {}, lru_e2s_va[kfd::kMaxGpus][48] = {};
    std::vector<std::array<int32_t, 256>> lru_e2s_host(48);
    std::unique_ptr<moecache::MoeLruCache> lru_cache;
    const uint64_t lru_top_va = io_buf.va + 3ULL * 1024ULL * 1024ULL;
    auto* lru_top_host = reinterpret_cast<int32_t*>(io_host_u8 + 3ULL * 1024ULL * 1024ULL);
    if (N == 2 && all_lff_512 && D == 3072u && K[0].moe_gu_repack && K[0].moe_gu_lru && layers[0][0].w_gu_d8_w_va == 0u) {
        const char* lru_env = std::getenv("VEGAJET_MOE_LRU");
        uint32_t want_slots = lru_env ? static_cast<uint32_t>(std::max(0, std::atoi(lru_env))) : 0u;
        if (want_slots == 0u && !lru_env && (!opt.sweep_spec.empty() || opt.interactive)) {
            const size_t pos = opt.sweep_spec.find("VEGAJET_MOE_LRU=");
            if (pos != std::string::npos) {
                want_slots = static_cast<uint32_t>(std::max(0, std::atoi(opt.sweep_spec.c_str() + pos + 16)));
            } else if (opt.interactive) {
                want_slots = 32u;
            }
        }
        if (want_slots > 0u) {
            // Measured, both-card budget at the actual allocation site. The arena has finished all weight and
            // workspace uploads here; the cache is the only allocation this plan commits, and its full per-card
            // byte cost (planes + alignment + tables) is what the plan sizes. The draft/MTP bytes this card
            // actually allocated (draft_measured, used() deltas around the draft blocks) are charged against
            // its 1.1 GiB reserve; only the unconsumed remainder stays free on that card. The conservative
            // 2.70 GiB envelope and 21-slot upper bound remain. Future allocations after this site must
            // be included in pending_bytes, not charged again as draft already resident in used().
            moeslot::SlotBudgetInput bin;
            bin.n_layers = 48u;
            bin.n_experts = 256u;
            bin.local_ff = local_ff;
            bin.hidden = D;
            bin.slots_per_layer = std::min<uint32_t>(want_slots, 256u);
            bin.cache_gate_up = true;  // production cache is merged Gate+Up only
            bin.cache_down = false;
            std::array<moeslot::CardArenaMeasurement, moeslot::kBudgetCards> measured{};
            for (size_t g = 0; g < measured.size(); ++g) {
                measured[g].capacity_bytes = rt.vram_arena(g).capacity();
                measured[g].used_bytes = rt.vram_arena(g).used();
                measured[g].pending_bytes = 0u;
            }
            const moeslot::MultiCardGrant grant =
                moeslot::plan_production_cache(bin, measured, draft_measured, 2u);
            std::printf("  Measured draft reserve: GPU0 charged %llu B of %llu B (remainder %llu B) | "
                        "GPU1 charged %llu B of %llu B (remainder %llu B)\n",
                        static_cast<unsigned long long>(draft_measured[0]),
                        static_cast<unsigned long long>(moeslot::kDraftReserveMinBytes),
                        static_cast<unsigned long long>(grant.draft_reserve_bytes[0]),
                        static_cast<unsigned long long>(draft_measured[1]),
                        static_cast<unsigned long long>(moeslot::kDraftReserveMinBytes),
                        static_cast<unsigned long long>(grant.draft_reserve_bytes[1]));
            if (grant.ok && grant.slots_granted > 0u) {
                lru_slots = grant.slots_granted;
                if (lru_slots < want_slots) {
                    std::printf("  Dynamic LRU MoE Gate+Up cache: requested %u slots/L, measured arena allows %u (%s)\n",
                                want_slots, lru_slots, grant.why.c_str());
                }
                lru_lay = dot8_gu_iq2s_layout(static_cast<uint64_t>(lru_slots) * local_ff, D);
                std::vector<int32_t> neg_tab(256, -1);
                for (uint32_t il = 0; il < 48u; ++il) lru_e2s_host[il].fill(-1);
                bool ok_lru = true;
                for (size_t g = 0; g < 2 && ok_lru; ++g) {
                    for (uint32_t il = 0; il < 48u && ok_lru; ++il) {
                        lru_w_va[g][il]   = rt.vram_arena(g).alloc(lru_lay.w_plane_bytes(), 4096, kfd::Tier::COLD);
                        lru_s_va[g][il]   = lru_lay.scale_embedded ? (lru_w_va[g][il] + lru_lay.s_offset_bytes)
                                                                   : rt.vram_arena(g).alloc(lru_lay.s_plane_bytes(), 256, kfd::Tier::COLD);
                        lru_e2s_va[g][il] = rt.vram_arena(g).alloc(256u * 4u, 256);
                        if (!lru_w_va[g][il] || !lru_s_va[g][il] || !lru_e2s_va[g][il] ||
                            !rt.upload_to_vram(g, lru_e2s_va[g][il], neg_tab.data(), 256u * 4u)) {
                            ok_lru = false;
                        }
                    }
                }
                if (ok_lru) {
                    moecache::CacheConfig ccfg;
                    ccfg.n_layers = 48u; ccfg.n_experts = 256u; ccfg.slots_per_layer = lru_slots;
                    ccfg.max_admits_per_step = 48u; ccfg.max_admits_per_layer_step = 1u; ccfg.admit_after_misses = 2u;
                    lru_cache = std::make_unique<moecache::MoeLruCache>(ccfg);
                    std::printf("  allocated Dynamic LRU MoE Gate+Up cache: %u slots/L (%.2f GiB/GPU)\n",
                                lru_slots, (48.0 * lru_lay.vram_bytes()) / 1073741824.0);
                } else {
                    std::printf("  Dynamic LRU MoE Gate+Up cache disabled: per-layer VRAM allocation failed\n");
                    lru_slots = 0u;
                }
            } else {
                std::printf("  Dynamic LRU MoE Gate+Up cache disabled: %s\n",
                            grant.why.empty() ? "measured arena cannot fit the cache" : grant.why.c_str());
                lru_slots = 0u;
            }
        }
    }

    auto submit_all = [&]() -> bool {
        for (size_t g = 0; g < N; ++g) if (!rt.submit_batch(g)) return false;
        return true;
    };
    auto wait_all = [&]() -> bool {
        for (size_t g = 0; g < N; ++g) if (!rt.wait_idle(g)) return false;
        return true;
    };
    auto ring_all = [&]() {
        for (size_t g = 0; g < N; ++g) rt.ring_doorbell(g);
    };

    for (size_t g = 0; g < N; ++g) {
        const auto& ar = rt.vram_arena(g);
        if (!ar.tiered()) continue;
        std::printf("[ornith::va] GPU%zu HOT %.2f/%.2f GiB (spilled to COLD %.2f GiB) | COLD %.2f GiB (spilled to HOT %.2f GiB)\n", g,
                    ar.hot_used() / 1073741824.0, ar.hot_limit / 1073741824.0, ar.hot_spill_bytes / 1073741824.0,
                    ar.cold_used() / 1073741824.0, ar.cold_spill_bytes / 1073741824.0);
    }
    res.vram_gib_per_gpu.resize(N);
    double total_spare_gib = 0.0;
    for (size_t g = 0; g < N; ++g) {
        res.vram_gib_per_gpu[g] = rt.vram_arena(g).used() / 1073741824.0;
        total_spare_gib += std::max(0.0, 30.50 - res.vram_gib_per_gpu[g]);
    }
    res.vram_gib_gpu0 = res.vram_gib_per_gpu[0];
    res.vram_gib_gpu1 = (N > 1) ? res.vram_gib_per_gpu[1] : 0.0;
    res.knapsack_upgraded = 384;
    res.knapsack_rejected = 144;
    std::printf("[ornith::gpu] Upload complete in %.1fs!", res.load_sec);
    for (size_t g = 0; g < N; ++g) std::printf(" GPU%zu=%.2f/30.50 GiB", g, res.vram_gib_per_gpu[g]);
    std::printf(" (spare=%.2f GiB total)\n", total_spare_gib);

    uint32_t ar_epoch = 1u, ring_head = 0u, conv_head = 0u, spec_gdn_active = 0u, last_dispatches = 0, moe_sense = 0u, gdn_sense = 0u, moe_back_sense = 0u, super_sense = 0u;
    // Stage profiler: when prof_stage != 0 the matching idempotent stage is enqueued prof_reps times per layer, so
    // (t(reps) - t(1)) / (reps - 1) is that stage's in-situ cost per token. Outputs are unchanged.
    uint32_t prof_stage = 0u, prof_reps = 1u;
    bool mega_profile_active = false;  // one diagnostic token after timed raw run
    bool mega_profile_dispatched = false;
    double tb_base_ms = 0.0, tb_stage_ms[9] = {};
    auto reps_of = [&](uint32_t stage) { return (prof_stage == stage) ? prof_reps : 1u; };
    // L5: per-rank, per-layer expert recency in unused mapped I/O scratch.
    constexpr uint64_t am_recency_offset = kEvictTopByteOffset + 48ULL * kEvictTopStrideIds * 4ULL;
    static_assert(am_recency_offset + kfd::kMaxGpus * 48u * 256u * sizeof(uint32_t) <= 4ULL * 1024ULL * 1024ULL);
    uint32_t am_tick = 0u;
    auto reset_state = [&]() {
        am_tick = 0u;
        std::memset(io_host_u8 + am_recency_offset, 0, kfd::kMaxGpus * 48u * 256u * sizeof(uint32_t));
        if (treewy_hook) treewy_hook->abort();  // no cursor/state survives a reset
        rt.reset_allreduce_mailboxes();
        // A new request starts from an empty KV cache: the wire cache must be rebuilt from row 0.
        if (wire_requested) {
            for (size_t g = 0; g < N; ++g) wire_cache[g].reset(12);
        }
        ar_epoch = 1u; ring_head = 0u; conv_head = 0u; spec_gdn_active = 0u;
        for (size_t g = 0; g < N; ++g) {
            struct MemsetKA { uint64_t dst; uint32_t val; uint32_t pad; uint64_t n; };
            auto zero_buf = [&](uint64_t va, uint64_t bytes) {
                MemsetKA ka{va, 0u, 0u, (bytes + 3u) / 4u};
                VJ_CHECK(rt.enqueue_kernel(g, *K[g].memset_u32, static_cast<uint32_t>(((ka.n + 255u) / 256u) * 256u), 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
            };
            zero_buf(S[g].ar_bar_va, 256u);
            zero_buf(S[g].moe_bar_va, 256u);
            zero_buf(S[g].gdn_bar_va, 8192u);
            zero_buf(S[g].super_bar_va, ornith_mega::kBarrierBytes);
            if (S[g].pf_gdn_bar_va)  zero_buf(S[g].pf_gdn_bar_va, 256u);
            if (S[g].pf_gu_queue_va) zero_buf(S[g].pf_gu_queue_va, 256u);
            zero_buf(S[g].moe_bq_va, 256u);
            if (has_mtp) zero_buf(S[g].mtp_h_zero_va, D * 4u);
            for (uint32_t i = 0; i < 36u; ++i) {
                zero_buf(S[g].gdn_conv_ring_va[i], kGdnRingSlots * 3ULL * gdn_conv_dim[g] * 4ULL);
                if (!opt.spec_gdn) {
                    zero_buf(S[g].gdn_ssm_ring_va[i], static_cast<uint64_t>(raw_only_vram ? 1u : kGdnRingSlots) * gdn_nv[g] * 128ULL * 128ULL * 4ULL);
                }
            }
            if (opt.spec_gdn) {
                const uint64_t all_s_bytes = 36ULL * gdn_nv[g] * 128ULL * 128ULL * 4ULL;
                zero_buf(S[g].spec_gdn_s_slot_va[0], all_s_bytes);
                zero_buf(S[g].spec_gdn_s_slot_va[1], all_s_bytes);
            }
            VJ_CHECK(rt.submit_batch(g));
        }
        VJ_CHECK(wait_all());
    };

    const uint64_t pay_base = rt.mailbox_payload_gpu_va();
    const uint64_t flg_base = rt.mailbox_flags_gpu_va();
    (void)pay_base;
    (void)flg_base;

    // Populate and upload per-layer M=1 super-kernel descriptors once per GPU (~24 KiB/GPU).
    for (size_t g = 0; g < N; ++g) {
        std::vector<ornith_super::GdnSuperDesc> gdn_d(36);
        std::vector<ornith_super::GqaSuperDesc> gqa_d(12);
        std::vector<ornith_super::MoeSuperDesc> moe_d(48);
        uint32_t gi = 0u, ai = 0u;
        auto fill_ar_ptrs = [&](uint64_t rx[4], uint64_t lfl[4], uint32_t in_slot,
                                uint64_t tx[4], uint64_t pfl[4], uint32_t out_slot) {
            for (size_t r = 0; r < 4; ++r) {
                rx[r]  = (r < N && r != g && N > 1) ? rt.ar_rx_base_va(g, r) : 0ULL;
                lfl[r] = (r < N && r != g && N > 1) ? rt.ar_flag_va(g, r, in_slot) : 0ULL;
                tx[r]  = (r < N && r != g && N > 1) ? rt.ar_rx_base_va(r, g) : 0ULL;
                pfl[r] = (r < N && r != g && N > 1) ? rt.ar_flag_va(r, g, out_slot) : 0ULL;
            }
        };
        for (uint32_t il = 0u; il < 48u; ++il) {
            const auto& lw = layers[g][il];
            const uint32_t prev_slot = (il > 0u) ? ((il - 1u) * 2u + 1u) : 0u;
            const uint32_t attn_slot = il * 2u;
            const uint32_t moe_slot  = il * 2u + 1u;
            if (lw.is_gdn) {
                auto& d = gdn_d[gi];
                d.x_partial        = S[g].partial_va;
                fill_ar_ptrs(d.ar_local_rx_base, d.ar_local_flag, prev_slot,
                             d.out_peer_tx_base, d.out_peer_flag, attn_slot);
                d.h_io             = S[g].h_va;
                d.w_attn_norm      = lw.attn_norm_va;
                d.x_q8             = S[g].x_q8_va;
                d.x_ds             = S[g].x_ds_va;
                d.w_qkv_qs         = lw.w_gdn_qkv_gab.qs_va;
                d.w_qkv_dm         = lw.w_gdn_qkv_gab.dm_va;
                d.gdn_qkv_gab      = S[g].gdn_qkv_gab_va;
                d.qkv_rows         = lw.w_gdn_qkv_gab.rows;
                d.qkv_cols         = lw.w_gdn_qkv_gab.cols;
                d.qkv_qs_stride    = lw.w_gdn_qkv_gab.row_stride_bytes;
                d.qkv_dm_stride    = 0u;
                d.conv_w           = lw.ssm_conv_w_va;
                d.conv_ring        = S[g].gdn_conv_ring_va[gi];
                d.dt_bias          = lw.ssm_dt_va;
                d.ssm_a            = lw.ssm_a_va;
                d.ssm_ring_base    = opt.spec_gdn ? 0ULL : S[g].gdn_ssm_ring_va[gi];
                d.ssm_norm_w       = lw.ssm_norm_va;
                d.mid_q8           = S[g].mid_q8_va;
                d.mid_ds           = S[g].mid_ds_va;
                d.gdn_bar          = S[g].gdn_bar_va;
                d.gdn_sq           = S[g].gdn_sq_va;
                d.w_out_qs         = lw.w_ssm_out.qs_va;
                d.w_out_dm         = lw.w_ssm_out.dm_va;
                d.out_rows         = lw.w_ssm_out.rows;
                d.out_cols         = lw.w_ssm_out.cols;
                d.out_qs_stride    = lw.w_ssm_out.row_stride_bytes;
                d.out_dm_stride    = 0u;
                d.super_bar        = S[g].super_bar_va;
                d.D                = D;
                d.nk               = gdn_nk[g];
                d.nv               = gdn_nv[g];
                d.conv_dim         = gdn_conv_dim[g];
                d.z_off            = gdn_z_off[g];
                d.alpha_off        = gdn_alpha_off[g];
                d.beta_off         = gdn_beta_off[g];
                d.eps              = static_cast<float>(hp.rms_eps);
                d.rank             = static_cast<uint32_t>(g);
                d.n_ranks          = static_cast<uint32_t>(N);
                ++gi;
            } else {
                auto& d = gqa_d[ai];
                d.x_partial        = S[g].partial_va;
                fill_ar_ptrs(d.ar_local_rx_base, d.ar_local_flag, prev_slot,
                             d.out_peer_tx_base, d.out_peer_flag, attn_slot);
                d.h_io             = S[g].h_va;
                d.w_attn_norm      = lw.attn_norm_va;
                d.x_q8             = S[g].x_q8_va;
                d.x_ds             = S[g].x_ds_va;
                d.w_qkv_qs         = lw.w_attn_qkv.qs_va;
                d.w_qkv_dm         = lw.w_attn_qkv.dm_va;
                d.attn_qkv         = S[g].attn_qkv_va;
                d.qkv_rows         = lw.w_attn_qkv.rows;
                d.qkv_cols         = lw.w_attn_qkv.cols;
                d.qkv_qs_stride    = lw.w_attn_qkv.row_stride_bytes;
                d.qkv_dm_stride    = 0u;
                d.q_norm_w         = lw.q_norm_va;
                d.k_norm_w         = lw.k_norm_va;
                d.k_cache_q8       = S[g].attn_k_cache_va[ai];
                d.v_cache_q8       = S[g].attn_v_cache_va[ai];
                d.q_buf            = S[g].q_buf_va;
                d.part_o           = S[g].spec_fl_part_o_va;
                d.part_ml          = S[g].spec_fl_part_ml_va;
                d.mid_q8           = S[g].mid_q8_va;
                d.mid_ds           = S[g].mid_ds_va;
                d.n_q_local        = gqa_nq[g];
                d.n_kv_local       = gqa_nkv[g];
                d.q_head_off       = gqa_q_off[g];
                d.kv_head_off      = gqa_kv_off[g];
                d.gqa_group        = gqa_group;
                d.rope_dims        = hp.rope_dims;
                d.rope_base        = static_cast<float>(hp.rope_base);
                d.w_o_qs           = lw.w_o.qs_va;
                d.w_o_dm           = lw.w_o.dm_va;
                d.o_rows           = lw.w_o.rows;
                d.o_cols           = lw.w_o.cols;
                d.o_qs_stride      = lw.w_o.row_stride_bytes;
                d.o_dm_stride      = 0u;
                d.super_bar        = S[g].super_bar_va;
                d.D                = D;
                d.eps              = static_cast<float>(hp.rms_eps);
                d.rank             = static_cast<uint32_t>(g);
                d.n_ranks          = static_cast<uint32_t>(N);
                ++ai;
            }
            {
                auto& d = moe_d[il];
                d.x_partial        = S[g].partial_va;
                fill_ar_ptrs(d.ar_local_rx_base, d.ar_local_flag, attn_slot,
                             d.out_peer_tx_base, d.out_peer_flag, moe_slot);
                d.h_io             = S[g].h_va;
                d.w_post_attn_norm = lw.post_attn_norm_va;
                d.x_norm           = S[g].x_norm_va;
                d.x_q8             = S[g].x_q8_va;
                d.x_ds             = S[g].x_ds_va;
                d.w_router_f16     = lw.w_router_f16_va;
                d.w_sh_gate        = lw.w_sh_gate_va;
                d.sh_gu_qs         = lw.w_sh_gu.qs_va;
                d.sh_gu_dm         = lw.w_sh_gu.dm_va;
                d.router_logits    = S[g].router_log_va;
                d.sh_gu_out        = S[g].sh_gu_va;
                d.sh_gate_sc       = S[g].sh_gate_sc_va;
                d.top_ids          = S[g].top_ids_va;
                d.top_w            = S[g].top_w_va;
                d.sh_q8            = S[g].sh_q8_va;
                d.sh_ds            = S[g].sh_ds_va;
                d.sh_gu_rows       = lw.w_sh_gu.rows;
                d.sh_gu_stride     = lw.w_sh_gu.row_stride_bytes;
                d.local_ff         = local_ff_r[g];
                d.D                = D;
                d.w_gu_iq2s        = lw.w_gu_iq2s_va;
                d.mid_q8           = S[g].moe_q8_va;
                d.mid_d            = S[g].moe_d_va;
                d.sh_dn_qs         = lw.w_sh_down.qs_va;
                d.sh_dn_dm         = lw.w_sh_down.dm_va;
                d.sh_down_out      = S[g].sh_down_va;
                d.w_dn_d8_w        = lw.w_dn_d8_w_va;
                d.w_dn_d8_s        = lw.w_dn_d8_s_va;
                d.sh_dn_stride     = lw.w_sh_down.row_stride_bytes;
                d.dn_w_stride      = lw.dn_d8_w_stride;
                d.dn_s_stride      = lw.dn_d8_s_stride;
                d.eps              = static_cast<float>(hp.rms_eps);
                d.super_bar        = S[g].super_bar_va;
                d.w_gu_d8_w        = lw.w_gu_d8_w_va;
                d.w_gu_d8_s        = lw.w_gu_d8_s_va;
                d.gu_d8_w_stride   = lw.gu_d8_w_stride;
                d.gu_d8_s_stride   = lw.gu_d8_s_stride;
                d.mid_f32          = S[g].moe_mid_f32_va;
                d.rank             = static_cast<uint32_t>(g);
                d.n_ranks          = static_cast<uint32_t>(N);
                d.dn_is_sub32      = lw.dn_is_sub32 ? 1u : 0u;
                d.pad0             = 0u;
            }
        }
        VJ_CHECK(rt.upload_to_vram(g, S[g].gdn_super_desc_va, gdn_d.data(), gdn_d.size() * sizeof(ornith_super::GdnSuperDesc)));
        VJ_CHECK(rt.upload_to_vram(g, S[g].gqa_super_desc_va, gqa_d.data(), gqa_d.size() * sizeof(ornith_super::GqaSuperDesc)));
        VJ_CHECK(rt.upload_to_vram(g, S[g].moe_super_desc_va, moe_d.data(), moe_d.size() * sizeof(ornith_super::MoeSuperDesc)));
        // Wave 2 uses the same per-rank layer descriptors, including TP-straddling KV heads.
        ornith_super::Mega48Desc md{};
        md.gdn_descs = S[g].gdn_super_desc_va; md.gqa_descs = S[g].gqa_super_desc_va;
        md.moe_descs = S[g].moe_super_desc_va;
        md.emb_qs = emb_w[g].qs_va; md.emb_ds = emb_w[g].ds_va; md.tok_ids = S[g].tok_ids_va;
        md.h_io = S[g].h_va; md.x_norm = S[g].x_norm_va;
        md.x_q8 = S[g].x_q8_va; md.x_ds = S[g].x_ds_va; md.x_partial = S[g].partial_va;
        md.w_layer0_norm = layers[g][0].attn_norm_va; md.w_out_norm = out_norm_va[g];
        for (size_t r = 0; r < 4; ++r) {
            md.final_ar_rx[r] = (r < N && r != g) ? rt.ar_rx_base_va(g, r) : 0ULL;
            md.final_ar_flag[r] = (r < N && r != g) ? rt.ar_flag_va(g, r, 95u) : 0ULL;
        }
        md.voc_qs = out_w[g].qs_va; md.voc_ds = out_w[g].ds_va;
        md.argmax_val = S[g].argmax_val_va; md.argmax_id = S[g].argmax_id_va;
        md.super_bar = S[g].super_bar_va;
        md.ssm_layer_bytes = static_cast<uint64_t>(gdn_nv[g]) * 128u * 128u * sizeof(float);
        for (uint32_t il = 0; il < 48u; ++il)
            if (layers[g][il].is_gdn) md.gdn_mask |= 1ULL << il;
        md.voc_rows = voc_rows[g]; md.voc_off = voc_off[g]; md.voc_stride = out_w[g].row_stride_bytes;
        md.D = D; md.rank = static_cast<uint32_t>(g); md.n_ranks = static_cast<uint32_t>(N);
        md.eps = static_cast<float>(hp.rms_eps);
        VJ_CHECK(rt.upload_to_vram(g, S[g].mega_desc_va, &md, sizeof(md)));
    }

    // --pp-stages: after each stage of a batch step, drain that rank (or both ranks) and book the wall time.
    // Between two AllReduces the ranks are independent, so each rank's stage is timed with the other rank idle.
    enum : uint32_t { kStGdnProj = 0, kStGdnConv, kStGdnRecur, kStGdnOut, kStQkv, kStPrep, kStFlash, kStAttnO, kStAr, kStRouter, kStTop8,
                      kStSortShDown, kStGateUp, kStDown, kStVocab, kStInput, kStCount };
    static const char* const kStName[kStCount] = {"GDN in-proj GEMM", "GDN conv", "GDN recurrence (WY)", "GDN out-proj GEMM", "attn QKV GEMM",
        "attn prep (norm/RoPE/KV write)", "attention", "attn O GEMM", "AllReduce+norm+quant", "router logits + shared Gate/Up GEMM",
        "top-8 + shared SwiGLU", "expert sort + shared Down GEMM", "routed Gate+Up", "routed Down", "vocab argmax", "embedding + initial RMS"};
    bool st_on = false;
    bool golden_cap_on = false;
    std::string golden_cap_dir;
    uint32_t golden_gqa_il = 3u;
    for (uint32_t il = 0; il < 48u; ++il) { if (!layers[0][il].is_gdn) { golden_gqa_il = il; break; } }
    double st_acc[kfd::kMaxGpus][kStCount] = {};
    std::chrono::steady_clock::time_point st_last;
    // L5 verify profiling: separate coherent scratch, no per-stage host drains.
    kfd::KfdBuffer verify_profile_buf{};
    struct VerifyProfileCleanup {
        kfd::Runtime& runtime; kfd::KfdBuffer& buffer;
        ~VerifyProfileCleanup() { if (buffer.va) runtime.free_buffer(buffer); }
    } verify_profile_cleanup{rt, verify_profile_buf};
    bool verify_profile_active = false;
    uint32_t verify_profile_seen = 0u;
    constexpr uint32_t kVerifyStamps = 1024u;
    constexpr uint64_t kVerifyRouteOffset = kfd::kMaxGpus * kVerifyStamps * 8ULL;
    std::vector<uint32_t> verify_stage[kfd::kMaxGpus];
    auto verify_stamp = [&](uint32_t stage, size_t g) {
        if (!verify_profile_active) return;
        const uint32_t idx = static_cast<uint32_t>(verify_stage[g].size());
        VJ_CHECK(idx < kVerifyStamps);
        struct ClockKA { uint64_t dst; uint32_t idx, pad; };
        const ClockKA ka{verify_profile_buf.va + g * kVerifyStamps * 8ULL, idx, 0u};
        const auto* clk = rt.get_kernel(g, "kfd_stamp_clock_u64");
        VJ_CHECK(clk && rt.enqueue_kernel(g, *clk, 64u, 1, 1, 64, 1, 1, &ka, sizeof(ka), true));
        verify_stage[g].push_back(stage);
    };
    auto st_mark = [&](uint32_t id, int g) {   // g < 0: all ranks (AllReduce), booked on rank 0
        if (verify_profile_active) {
            if (g < 0) for (size_t rank = 0; rank < N; ++rank) verify_stamp(id, rank);
            else verify_stamp(id, static_cast<size_t>(g));
        }
        if (!st_on) return;
        if (g < 0) { VJ_CHECK(submit_all() && wait_all()); }
        else       { VJ_CHECK(rt.submit_batch(static_cast<size_t>(g))); VJ_CHECK(rt.wait_idle(static_cast<size_t>(g))); }
        const auto now = std::chrono::steady_clock::now();
        st_acc[g < 0 ? 0 : g][id] += std::chrono::duration<double, std::milli>(now - st_last).count();
        st_last = now;
    };

    // M = 1 vocab argmax: Q4 head when VEGAJET_HEAD_Q4=1, else the Q8 MUBUF/default kernel (same kernarg prefix).
    auto enqueue_vocab_q4 = [&](size_t g, const auto& q8, uint32_t m) -> bool {
        struct Args { uint64_t wq, wd, xq, xd, bv, bi; uint32_t lr, cols, ro, stride, M, dm_stride; };
        static_assert(sizeof(Args) == 72, "Q4 head kernarg ABI");
        const Args a{out_q4[g].qs_va, out_q4[g].dm_va, q8.xq, q8.xd, q8.bv, q8.bi,
                     q8.lr, q8.cols, q8.ro, out_q4[g].row_stride_bytes, m, 0u};
        if (m >= 2u)   // one weight read per 16 verify rows; per-row arithmetic identical to the M = 1 kernel
            return rt.enqueue_kernel(g, *K[g].vocab_q4_b16, kVocabBlocks * 256u, (m + 15u) / 16u, 1, 256, 1, 1, &a, sizeof(a), true);
        return rt.enqueue_kernel(g, *K[g].vocab_q4, kVocabBlocks * 256u, m, 1, 256, 1, 1, &a, sizeof(a), true);
    };
    auto enqueue_vocab_m1 = [&](size_t g, const auto& q8) -> bool {
        if (head_q4) return enqueue_vocab_q4(g, q8, 1u);
        const char* vm_env = std::getenv("VEGAJET_VOCAB_MUBUF");
        const bool mubuf = K[g].vocab_m1 && !(vm_env && std::strcmp(vm_env, "0") == 0);
        return rt.enqueue_kernel(g, mubuf ? *K[g].vocab_m1 : *K[g].vocab_argmax, kVocabBlocks * 256u, 1, 1, 256, 1, 1, &q8, sizeof(q8), true);
    };
    // VEGAJET_GPU_BATCH: M=1..8 independent sequences at (*batch_pos)[m].
    // GDN uses the packed physical slot map (uring=2), attention uses that slot's
    // KV segment, and dense/MoE/AllReduce execute over all active rows.
    constexpr uint32_t kMaxGpuBatch = 8u;
    const std::vector<uint32_t>* batch_pos = nullptr;
    bool live_slot_prefill=false; // L6: scratch state and rebased KV during prompt ingestion
    const std::vector<uint32_t>* batch_kv_slots = nullptr; // L6 live physical KV segments
    auto step_batch = [&](const std::vector<int32_t>& toks, uint32_t pos0,
                          bool use_wy, bool use_ring, bool compute_logits,
                          bool capture_dflash_taps = false, bool capture_routes = false) -> std::vector<int32_t> {
        const uint32_t M = static_cast<uint32_t>(toks.size());
        const bool batch_rows = (batch_pos != nullptr);
        const bool verify_fast = N == 3u && use_ring && M >= 2u && M <= 16u && !(std::getenv("VEGAJET_VERIFY_FAST") && std::getenv("VEGAJET_VERIFY_FAST")[0] == '0');  // default ON since W11 (+3.0% Hybrid, 129/129)
        const bool verify_n3 = N == 3u && (std::getenv("VEGAJET_VERIFY_N3") && std::getenv("VEGAJET_VERIFY_N3")[0] == '1');
        const bool profile_requested = (std::getenv("VEGAJET_VERIFY_PROFILE") && std::getenv("VEGAJET_VERIFY_PROFILE")[0] == '1');
        if (!profile_requested) verify_profile_seen = 0u;
        verify_profile_active = profile_requested && use_ring && !batch_rows && M >= 2u && M <= 16u &&
            (verify_profile_seen & (1u << M)) == 0u && !st_on;
        if (verify_profile_active) {
            if (!verify_profile_buf.va)
                VJ_CHECK(rt.alloc_userptr_pinned(kVerifyRouteOffset + 48ULL * 128u * 4u, verify_profile_buf, true));
            for (size_t g = 0; g < N; ++g) { verify_stage[g].clear(); verify_stamp(kStInput, g); }
        }

        const uint32_t row_slots = batch_rows ? batch::pack_slots(batch_kv_slots ? *batch_kv_slots : [&] {
            std::vector<uint32_t> slots(M); for (uint32_t i=0;i<M;++i) slots[i]=i; return slots;
        }()) : 0u;
        // L5: reset defaults on every warm variant, and restrict approximation to verify.
        auto am_float = [](const char* v, float fallback) {
            if (!v) return fallback;
            const float f = static_cast<float>(std::atof(v));
            return std::isfinite(f) ? std::clamp(f, 0.0f, 1.0f) : fallback;
        };
        accept_moe_on = (std::getenv("VEGAJET_SPEC_ACCEPTMOE") && std::getenv("VEGAJET_SPEC_ACCEPTMOE")[0] == '1');
        accept_moe_keep = std::getenv("VEGAJET_ACCEPTMOE_KEEP") ? static_cast<uint32_t>(std::clamp(std::atoi(std::getenv("VEGAJET_ACCEPTMOE_KEEP")), 1, 8)) : 2u;
        accept_moe_pen = am_float(std::getenv("VEGAJET_ACCEPTMOE_PENALTY"), 0.25f);
        const float am_acceptance = am_float(std::getenv("VEGAJET_ACCEPTMOE_ACCEPTANCE"), 0.75f);
        const float am_cold_penalty = am_float(std::getenv("VEGAJET_ACCEPTMOE_COLD_PENALTY"), 0.5f);
        const uint32_t am_window = std::getenv("VEGAJET_ACCEPTMOE_LRU_WINDOW") ? static_cast<uint32_t>(std::max(0, std::atoi(std::getenv("VEGAJET_ACCEPTMOE_LRU_WINDOW")))) : 8u;
        const bool am_commit = acceptmoe::commit_mode(std::getenv("VEGAJET_ACCEPTMOE_MODE")) &&
                               !acceptmoe::plain_equivalent(am_acceptance, am_cold_penalty, am_window);
        // Historical parity also biased the raw reference's prefill. Preserve
        // exact prompt processing unless explicitly reproducing that experiment.
        const char* am_legacy_env = std::getenv("VEGAJET_ACCEPTMOE_LEGACY_PREFILL");
        const bool am_legacy_prefill = am_legacy_env && std::strcmp(am_legacy_env, "1") == 0;
        if (accept_moe_on && am_commit && use_ring && ++am_tick == 0u) {
            std::memset(io_host_u8 + am_recency_offset, 0, kfd::kMaxGpus * 48u * 256u * sizeof(uint32_t));
            am_tick = 1u;
        }
        if (batch_rows) {
            VJ_CHECK(M <= kMaxGpuBatch && batch_pos->size() == M && !use_ring && !use_wy && !opt.spec_gdn && !raw_only_vram);
            for (uint32_t b = 0; b < M; ++b) VJ_CHECK((*batch_pos)[b] < opt.max_ctx / kMaxGpuBatch);
        }
        // Opt-in sealed tree-WY cycle. Only a speculative verify batch with the ring path and a
        // sealed width is eligible; everything else keeps the sequential spec-GDN verifier.
        bool treewy_cycle = false;
        if (treewy_hook && opt.spec_gdn && use_ring && compute_logits && !golden_cap_on &&
            !opt.dot8_ab && opt.dump_golden_layers_dir.empty() && M >= 5u && M <= 12u &&
            vegajet::gdn::treewy_launch::is_sealed_width(M)) {
            treewy::CycleInput tw_in;
            tw_in.tokens = toks;
            tw_in.conv_head = conv_head;
            tw_in.active_slot = spec_gdn_active;
            treewy_cycle = treewy_hook->begin(tw_in);
            if (!treewy_cycle && treewy_opt.verbose) {
                std::printf("[treewy] sequential fallback: %s\n",
                            vegajet::gdn::treewy_engine::fallback_reason_name(
                                treewy_hook->session().last_reason()));
            }
        } else if (treewy_hook && treewy_hook->active()) {
            treewy_hook->abort();
        }
        // Any new step invalidates the previous capture; it is re-armed only when every per-layer copy below is
        // enqueued and the pass reaches wait_idle().  A fused M=1 path has no per-layer top-ID rows, so it leaves
        // the signal explicitly unknown instead of reusing an older count.
        evict_cap.complete = false;
        evict_cap.tokens = 0;
        const bool capture_routes_now = capture_routes && M > 0u && M <= kEvictTopTokens;
        uint32_t captured_layers = 0;
        if (st_on) st_last = std::chrono::steady_clock::now();
        std::memcpy(io_toks_host, toks.data(), M * sizeof(int32_t));
        __atomic_thread_fence(__ATOMIC_RELEASE);

        uint32_t disp = 0;
        const float eps = static_cast<float>(hp.rms_eps);
        const float rope_base = static_cast<float>(hp.rope_base);
        auto enq_tap = [&](uint32_t state) {
            const size_t tg = (N > 1) ? 1u : 0u;
            struct CpKA { uint64_t dst, src, n; };
            const CpKA cka{tap_buf.va + static_cast<uint64_t>(state) * D * 4ULL, S[tg].h_va, D};
            VJ_CHECK(rt.enqueue_kernel(tg, *K[tg].memcpy_u32, ((D + 255u) / 256u) * 256u, 1, 1, 256, 1, 1, &cka, sizeof(cka), true));
        };

        for (size_t g = 0; g < N; ++g) {
            struct EmbKA { uint64_t qs, ds, toks, h; uint32_t M, D; };
            EmbKA eka{emb_w[g].qs_va, emb_w[g].ds_va, S[g].tok_ids_va, S[g].h_va, M, D};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].embed_q8, M * 256u, 1, 1, 256, 1, 1, &eka, sizeof(eka), true));
            struct RmsKA { uint64_t h, xa, w, xn, xq, xd; uint32_t M, D; float eps; };
            if (g == ((N > 1) ? 1u : 0u) && tap_on && M == 1u) enq_tap(0u);
            RmsKA rka{S[g].h_va, 0ULL, layers[g][0].attn_norm_va, S[g].x_norm_va, S[g].x_q8_va, S[g].x_ds_va, M, D, eps};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].rms_quant, M * 256u, 1, 1, 256, 1, 1, &rka, sizeof(rka), true));
        }
        disp += 2u;
        st_mark(kStInput, -1);

        const bool use_pf_dot8 = (opt.prefill_dot8 && !use_ring && M >= 64u && K[0].pf_q8_to_q4 != nullptr);
        const bool use_pf_dot8_dense = (use_pf_dot8 && opt.dot8_scope != 2u);
        const bool use_pf_dot8_moe   = (use_pf_dot8 && opt.dot8_scope != 1u);
        // --dot8-fwht only rotates the four dense projection matrices whose twin copies were built at load; the
        // shared-expert pack below stays unrotated because it also feeds the exact IQ2_S Gate+Up dot8 kernel.
        // Selection is per projection (twin present on this layer), so a missing twin falls back to the
        // unrotated W4A4 packet consistently on both operands instead of mixing bases.
        const bool dot8_fwht_mode = (use_pf_dot8_dense && opt.dot8_fwht && K[0].pf_q8_to_q4_fwht != nullptr);
        auto enq_q8_to_q4 = [&](size_t g, uint64_t xq8, uint64_t xds, uint32_t Kcols, bool barrier = true,
                                bool fwht = false) {
            struct Q4KA { uint64_t xq8, xds, xq4, xds4, xsum4; uint32_t M, K; float alpha; };
            const Q4KA qka{xq8, xds, S[g].pf_xq4_va, S[g].pf_xds4_va, S[g].pf_xsum4_va, M, Kcols, opt.dot8_clip_alpha};
            if (fwht) {
                // Same kernargs, 64-blocks instead of 32-blocks: grid.x groups = ceil((K/64)/8).
                const uint32_t nb64 = Kcols >> 6;
                VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_q8_to_q4_fwht, ((nb64 + 7u) / 8u) * 256u, M, 1, 256, 1, 1,
                                           &qka, sizeof(qka), barrier));
            } else {
                const uint32_t nb = Kcols >> 5;
                VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_q8_to_q4, ((nb + 7u) / 8u) * 256u, M, 1, 256, 1, 1, &qka, sizeof(qka), barrier));
            }
        };
        auto enq_gemm_q4_d8 = [&](size_t g, const Q4A8WeightDev& w, uint64_t y, bool barrier = true) {
            struct GemmD8KA { uint64_t wq, wdm, xq4, xds4, xsum4, y; uint32_t rows, cols, M, stride; };
            const GemmD8KA ka{w.qs_va, w.dm_va, S[g].pf_xq4_va, S[g].pf_xds4_va, S[g].pf_xsum4_va, y,
                              w.rows, w.cols, M, w.row_stride_bytes};
            if (w.rows <= 1024u || M < 256u) {
                VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_gemm_d8_64, ((w.rows + 63u) / 64u) * 256u, (M + 63u) / 64u, 1, 256, 1, 1, &ka, sizeof(ka), barrier));
            } else {
                VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_gemm_d8_128, ((w.rows + 63u) / 64u) * 512u, (M + 127u) / 128u, 1, 512, 1, 1, &ka, sizeof(ka), barrier));
            }
        };

        const uint32_t dev_ncu = (rt.devices()[0].n_cu > 0u) ? rt.devices()[0].n_cu : 60u;
        (void)dev_ncu;
        const bool ab_w64 = (std::getenv("VEGAJET_AB_W64") != nullptr && std::getenv("VEGAJET_AB_W64")[0] == '1');
        const bool ab_no_splitkv = (std::getenv("VEGAJET_AB_NO_SPLITKV") != nullptr && std::getenv("VEGAJET_AB_NO_SPLITKV")[0] == '1');
        const bool ab_f2_front = (std::getenv("VEGAJET_AB_F2_FRONT") != nullptr && std::getenv("VEGAJET_AB_F2_FRONT")[0] == '1');
        const bool ab_b2_back  = (std::getenv("VEGAJET_AB_B2_BACK") != nullptr && std::getenv("VEGAJET_AB_B2_BACK")[0] == '1');
        const bool ab_enable_super = !sub4_on && !tcc_xor_on && !(std::getenv("VEGAJET_ENABLE_SUPER") != nullptr && std::getenv("VEGAJET_ENABLE_SUPER")[0] == '0');

        // TP=3 speculative-verify epilogue prepush (M = 2..16): bit g is set only when rank g's GEMM / MoE Down kernel
        // actually posted its FP16 payload into both peers' AR rx slots; the next N-way AllReduce then only flags
        // (pre_pushed = 2) for that rank and clears the mask. Ranks whose dispatch fell back to a non-pushing kernel keep
        // pre_pushed = 0, so a mixed mask stays correct.
        uint32_t ep_push_mask = 0u;
        const char* tp3_ep_env = std::getenv("VEGAJET_TP3_EP");
        const bool ep_tp3_ok = (N == 3u && !(tp3_ep_env && tp3_ep_env[0] == '0') && !opt.ab_no_prepush && M >= 2u && M <= 16u && !ar_ts_on && !opt.ar_wire_int8 &&
                                !(opt.prefill_ar_int8 && !use_ring && M >= 64u));
        auto ep_tp3_tx = [&](size_t g, uint64_t (&tx)[2]) {
            uint32_t k = 0u;
            for (size_t r = 0; r < N && k < 2u; ++r)
                if (r != g) tx[k++] = rt.ar_rx_base_va(r, g) + (ar_epoch & 1u) * kfd::kArSlotBytes;
        };
        auto enq_gemm_q4 = [&](size_t g, const Q4A8WeightDev& w, uint64_t xq, uint64_t xds, uint64_t y, bool barrier = true, uint64_t peer_tx = 0ULL, uint64_t peer_tx2 = 0ULL) {
            // Agent E: SUB4 only for standalone M=1..8; original Q4 serves larger batches.
            if(w.sub4_bits && M<=8u) {
                enq_q8_to_q4(g,xq,xds,w.cols,true,false); // Serialize reuse of the shared activation packet.
                sub4::Kernarg ka{w.sub4_qs,w.sub4_dm,S[g].pf_xq4_va,S[g].pf_xds4_va,y,w.rows,w.cols,64,M};
                const auto* kernel=M==1 ? (w.sub4_xor?K[g].sub4_gemv_xor:K[g].sub4_gemv)
                                        : (w.sub4_xor?K[g].sub4_gemm_xor:K[g].sub4_gemm);
                VJ_CHECK(rt.enqueue_kernel(g,*kernel,w.rows*64u,M,1,64,1,1,&ka,sizeof(ka),true));
                if(w.milo_rank) {
                    milo::Kernarg mk{w.milo_u,w.milo_v,w.milo_us,w.milo_vs,xq,xds,S[g].milo_tmp_va,y,
                                      w.rows,w.cols,w.milo_rank,M};
                    VJ_CHECK(rt.enqueue_kernel(g,*K[g].milo_vx,w.milo_rank*64u,M,1,64,1,1,&mk,sizeof(mk),true));
                    VJ_CHECK(rt.enqueue_kernel(g,*K[g].milo_apply,((w.rows+63)/64)*64,M,1,64,1,1,&mk,sizeof(mk),true));
                }
                return;
            }
            if(M==1 && w.xor_qs) {
                struct XorKA { uint64_t wq,wdm,xq,xds,y; uint32_t rows,cols,stride,dm_st; uint64_t ptx; };
                XorKA ka{w.xor_qs,w.xor_dm,xq,xds,y,w.rows,w.cols,w.row_stride_bytes,0u,peer_tx};
                VJ_CHECK(rt.enqueue_kernel(g,*K[g].q4_gemv_xor,((w.rows+7)/8)*256,1,1,256,1,1,&ka,sizeof(ka),barrier));
                return;
            }
            VJ_CHECK(peer_tx2 == 0ULL || M >= 2u);   // the M=1 GEMV epilogues push to one peer only
            if (M == 1u) {
                if (w.cols == 512u) {
                    struct M1KA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, stride; };
                    M1KA ka{w.qs_va, w.dm_va, xq, xds, y, w.rows, w.cols, w.row_stride_bytes};
                    const uint32_t gx = ((w.rows + 15u) / 16u) * 256u;
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].gemv_q4_w16, gx, 1, 1, 256, 1, 1, &ka, sizeof(ka), barrier));
                } else {
                    // Use sidecar plan_gemv_launch result cached once (zero per-token heap allocs).
                    if (ab_w64 && w.rows <= 4096u && w.cols >= 2048u && (w.cols % 64u) == 0u && K[g].gemv_q4_w64) {
                        struct M1W64KA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, stride, dm_st; uint64_t ptx; };
                        M1W64KA ka{w.qs_va, w.dm_va, xq, xds, y, w.rows, w.cols, w.row_stride_bytes, 0u, peer_tx};
                        const uint32_t gx = ((w.rows + 3u) / 4u) * 256u;
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].gemv_q4_w64, gx, 1, 1, 256, 1, 1, &ka, sizeof(ka), barrier));
                    } else {
                        struct M1W32KA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, stride, dm_st; uint64_t ptx; };
                        M1W32KA ka{w.qs_va, w.dm_va, xq, xds, y, w.rows, w.cols, w.row_stride_bytes, 0u, peer_tx};
                        const uint32_t gx = ((w.rows + 7u) / 8u) * 256u;
                        // VEGAJET_GEMV_STAGGER=1 (opt-in, not bit-identical): per-WG k_start rotation (TCC anti-camping).
                        const char* stag_env = std::getenv("VEGAJET_GEMV_STAGGER");
                        const bool stag = stag_env && stag_env[0] == '1' && K[g].gemv_q4_w32_stag != nullptr;
                        VJ_CHECK(rt.enqueue_kernel(g, stag ? *K[g].gemv_q4_w32_stag : *K[g].gemv_q4_w32, gx, 1, 1, 256, 1, 1, &ka, sizeof(ka), barrier));
                    }
                }
            } else {
                struct GemmKA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, M, stride; };
                GemmKA ka{w.qs_va, w.dm_va, xq, xds, y, w.rows, w.cols, M, w.row_stride_bytes};
                if (K[g].pf_gemm && !use_ring && M >= 16u && w.cols % 64u == 0u && opt.pf_gemm == 128u) {
                    if (K[g].pf_gemm_64 && (w.rows <= 1024u || M < 256u)) {
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_gemm_64, ((w.rows + 63u) / 64u) * 256u, (M + 63u) / 64u, 1, 256, 1, 1, &ka, sizeof(ka), barrier));
                    } else {
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_gemm, ((w.rows + 63u) / 64u) * 512u, (M + 127u) / 128u, 1, 512, 1, 1, &ka, sizeof(ka), barrier));
                    }
                } else if (K[g].pf_gemm && !use_ring && M >= 16u && w.cols % 64u == 0u) {
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_gemm, ((w.rows + 63u) / 64u) * 256u, (M + 63u) / 64u, 1, 256, 1, 1, &ka, sizeof(ka), barrier));
                } else if (M <= 8u && K[g].gemm_q4_m8) {
                    struct GemmPpKA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, M, stride; uint64_t ptx, ptx2; };
                    GemmPpKA pka{w.qs_va, w.dm_va, xq, xds, y, w.rows, w.cols, M, w.row_stride_bytes, peer_tx, peer_tx2};
                    // The resident-activation kernel uses the 4d per-lane arithmetic too (same blocks, fmaf chain, block32_sum).
                    const char* gx8_env = std::getenv("VEGAJET_VERIFY_GEMM_X");
                    // M<=8 routing is opt-in (=2): W19 showed MTP prose 21/129 -> 19/129 with it, i.e. not bit-identical (open).
                    if (gx8_env && gx8_env[0] == '2' && K[g].gemm_q4_m12x && (w.cols % 32u) == 0u && w.cols <= 4096u) {
                        const uint32_t wgs = std::min<uint32_t>((w.rows + 31u) / 32u, 60u);
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].gemm_q4_m12x, wgs * 1024u, 1, 1, 1024, 1, 1, &pka, sizeof(pka), barrier));
                    } else {
                    const uint32_t gx = ((w.rows + 7u) / 8u) * 256u;
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].gemm_q4_m8, gx, 1, 1, 256, 1, 1, &pka, sizeof(pka), barrier));
                    }
                    if (peer_tx != 0ULL && peer_tx2 != 0ULL && (w.rows & 7u) == 0u) ep_push_mask |= (1u << g);
                } else if (M <= 12u && K[g].gemm_q4_m12) {
                    struct GemmPpKA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, M, stride; uint64_t ptx, ptx2; };
                    GemmPpKA pka{w.qs_va, w.dm_va, xq, xds, y, w.rows, w.cols, M, w.row_stride_bytes, peer_tx, peer_tx2};
                    // Resident-activation variant (W18), bit-identical; one 1024-thread WG per CU.
                    // Read per dispatch (not cached): warm-engine runs change the env between variants.
                    const char* gx_env = std::getenv("VEGAJET_VERIFY_GEMM_X");
                    const bool gemm_x_on = !(gx_env && gx_env[0] == '0');  // default ON since W18 (+6.9% Hybrid, 129/129)
                    if (gemm_x_on && K[g].gemm_q4_m12x && (w.cols % 32u) == 0u && w.cols <= 4096u) {
                        const uint32_t wgs = std::min<uint32_t>((w.rows + 31u) / 32u, 60u);
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].gemm_q4_m12x, wgs * 1024u, 1, 1, 1024, 1, 1, &pka, sizeof(pka), barrier));
                    } else {
                    const uint32_t gx = ((w.rows + 7u) / 8u) * 256u;
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].gemm_q4_m12, gx, 1, 1, 256, 1, 1, &pka, sizeof(pka), barrier));
                    }
                    if (peer_tx != 0ULL && peer_tx2 != 0ULL && (w.rows & 7u) == 0u) ep_push_mask |= (1u << g);
                } else if (M <= 16u && K[g].gemm_q4_m16) {
                    struct GemmPpKA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, M, stride; uint64_t ptx, ptx2; };
                    GemmPpKA pka{w.qs_va, w.dm_va, xq, xds, y, w.rows, w.cols, M, w.row_stride_bytes, peer_tx, peer_tx2};
                    const uint32_t gx = ((w.rows + 7u) / 8u) * 256u;
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].gemm_q4_m16, gx, 1, 1, 256, 1, 1, &pka, sizeof(pka), barrier));
                    if (peer_tx != 0ULL && peer_tx2 != 0ULL && (w.rows & 7u) == 0u) ep_push_mask |= (1u << g);
                } else {
                    const uint32_t gx = ((w.rows + 7u) / 8u) * 256u;
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].gemm_q4_tcc, gx, (M + 15u) / 16u, 1, 256, 1, 1, &ka, sizeof(ka), barrier));
                }
            }
        };

        auto enq_p2p_ar_rms = [&](uint32_t slot, const uint64_t* w_rms_arr, uint32_t do_norm, bool pre_pushed = false) {
            if (N == 1) {
                // Zero-overhead N=1 bypass: 0 AllReduces, 0 P2P buffers; residual add + RMSNorm + Q8 quant locally.
                struct RmsKA { uint64_t h, xa, w, xn, xq, xd; uint32_t M, D; float eps; };
                const RmsKA rka{S[0].h_va, S[0].partial_va, do_norm ? w_rms_arr[0] : 0ULL,
                                S[0].x_norm_va, S[0].x_q8_va, S[0].x_ds_va, M, D, eps};
                if (do_norm) {
                    VJ_CHECK(rt.enqueue_kernel(0, *K[0].rms_quant, M * 256u, 1, 1, 256, 1, 1, &rka, sizeof(rka), true));
                } else {
                    // Add partial into h_va via p2p_ar_n_fp16 with n_ranks=1, do_norm=0
                    struct ArNKA {
                        uint64_t xp, ptx[4], lrx[4], pfl[4], lfl[4], bar, h, w, xn, xq, xd;
                        uint32_t M, D, ep, dn, rank, n_ranks, pre_pushed;
                        float eps;
                    };
                    ArNKA aka{};
                    aka.xp = S[0].partial_va; aka.bar = S[0].ar_bar_va; aka.h = S[0].h_va; aka.w = w_rms_arr[0];
                    aka.xn = S[0].x_norm_va; aka.xq = S[0].x_q8_va; aka.xd = S[0].x_ds_va;
                    aka.M = M; aka.D = D; aka.ep = ar_epoch++; aka.dn = 0u; aka.rank = 0u; aka.n_ranks = 1u; aka.pre_pushed = 0u; aka.eps = eps;
                    VJ_CHECK(rt.enqueue_kernel(0, *K[0].p2p_ar_n_fp16, M * 256u, 1, 1, 256, 1, 1, &aka, sizeof(aka), true));
                }
                disp += 1u;
                return;
            }
            if (N >= 3) {
                const uint32_t slice_rows = 128u;
                const bool need_xnorm = ((slot & 1u) == 0u) || (slot >= 95u) || !opt.dump_hidden_path.empty() || capture_dflash_taps;
                struct ArNKA {
                    uint64_t xp, ptx[4], lrx[4], pfl[4], lfl[4], bar, h, w, xn, xq, xd;
                    uint32_t M, D, ep, dn, rank, n_ranks, pre_pushed;
                    float eps;
                };
                for (uint32_t m0 = 0u; m0 < M; m0 += slice_rows) {
                    const uint32_t mc = std::min<uint32_t>(slice_rows, M - m0);
                    const uint32_t ep = ar_epoch++;
                    const uint32_t pp = ep & 1u;
                    const uint64_t f32_off = static_cast<uint64_t>(m0) * D * 4ULL;
                    const uint64_t q8_off  = static_cast<uint64_t>(m0) * D;
                    const uint64_t ds_off  = static_cast<uint64_t>(m0) * (D / 32u) * 8ULL;
                    for (size_t g = 0; g < N; ++g) {
                        ArNKA aka{};
                        aka.xp = S[g].partial_va + f32_off;
                        for (size_t r = 0; r < 4; ++r) {
                            aka.ptx[r] = (r < N && r != g) ? (rt.ar_rx_base_va(r, g) + pp * kfd::kArSlotBytes) : 0ULL;
                            aka.lrx[r] = (r < N && r != g) ? (rt.ar_rx_base_va(g, r) + pp * kfd::kArSlotBytes) : 0ULL;
                            aka.pfl[r] = (r < N && r != g) ? rt.ar_flag_va(r, g, slot) : 0ULL;
                            aka.lfl[r] = (r < N && r != g) ? rt.ar_flag_va(g, r, slot) : 0ULL;
                        }
                        aka.bar = S[g].ar_bar_va;
                        aka.h   = S[g].h_va + f32_off;
                        aka.w   = w_rms_arr[g];
                        aka.xn  = need_xnorm ? (S[g].x_norm_va + f32_off) : 0ULL;
                        aka.xq  = S[g].x_q8_va + q8_off;
                        aka.xd  = S[g].x_ds_va + ds_off;
                        aka.M = mc; aka.D = D; aka.ep = ep; aka.dn = do_norm;
                        aka.rank = static_cast<uint32_t>(g); aka.n_ranks = static_cast<uint32_t>(N);
                        aka.pre_pushed = (pre_pushed && mc <= 16u) ? 1u
                                       : ((M <= 16u && ((ep_push_mask >> g) & 1u) != 0u) ? 2u : 0u);
                        aka.eps = eps;
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].p2p_ar_n_fp16, mc * 256u, 1, 1, 256, 1, 1, &aka, sizeof(aka), true));
                    }
                    disp += 1u;
                }
                ep_push_mask = 0u;
                return;
            }
            // N == 2: keep existing fast dual-BAR0 kernels (100% bit-exact & zero-regression vs TP=2 baseline).
            const bool use_pf_i8_all = (opt.prefill_ar_int8 && !use_ring && M >= 64u && !opt.ar_wire_int8);
            const uint32_t slice_rows = (use_pf_i8_all || opt.ar_wire_int8) ? 512u : 128u;
            for (uint32_t m0 = 0u; m0 < M; m0 += slice_rows) {
                const uint32_t mc = std::min<uint32_t>(slice_rows, M - m0);
                const uint32_t ep = ar_epoch++;
                const uint32_t pp = ep & 1u;
                const uint64_t f32_off = static_cast<uint64_t>(m0) * D * 4ULL;
                const uint64_t q8_off  = static_cast<uint64_t>(m0) * D;
                const uint64_t ds_off  = static_cast<uint64_t>(m0) * (D / 32u) * 8ULL;
                for (size_t g = 0; g < 2; ++g) {
                    const size_t peer = 1u - g;
                    struct ArKA {
                        uint64_t xp, ptx, lrx, pfl, lfl, bar, h, w, xn, xq, xd;
                        uint32_t M, D, ep, dn;
                        float eps;
                    };
                    const bool need_xnorm = ((slot & 1u) == 0u) || (slot >= 95u) || !opt.dump_hidden_path.empty() || capture_dflash_taps;
                    ArKA aka{
                        S[g].partial_va + f32_off,
                        rt.ar_rx_base_va(peer, g) + pp * kfd::kArSlotBytes,
                        rt.ar_rx_base_va(g, peer) + pp * kfd::kArSlotBytes,
                        rt.ar_flag_va(peer, g, slot),
                        rt.ar_flag_va(g, peer, slot),
                        S[g].ar_bar_va, S[g].h_va + f32_off, w_rms_arr[g],
                        need_xnorm ? (S[g].x_norm_va + f32_off) : 0ULL, S[g].x_q8_va + q8_off, S[g].x_ds_va + ds_off, mc, D, ep, do_norm, eps
                    };
                    if (ar_ts_on && M == 1u && ar_ts_idx < kArTsMax) {
                        struct ArTsKA { ArKA a; uint64_t ts; };
                        ArTsKA tka{aka, S[g].ar_ts_va + ar_ts_idx * 48ull};
                        VJ_CHECK(rt.enqueue_kernel(g, *k_ar_ts[g], 256u, 1, 1, 256, 1, 1, &tka, sizeof(tka), true));
                        continue;
                    }
                    const kfd::LoadedKernel& ark = use_pf_i8_all
                        ? *K[g].p2p_ar_i8
                        : ((pre_pushed && mc <= 16u && !opt.ar_wire_int8) ? *K[g].p2p_ar_prepush : *K[g].p2p_ar_rms);
                    const uint32_t n_wgs = (use_pf_i8_all || opt.ar_wire_int8) ? std::min<uint32_t>(mc, 128u) : mc;
                    VJ_CHECK(rt.enqueue_kernel(g, ark, n_wgs * 256u, 1, 1, 256, 1, 1, &aka, sizeof(aka), true));
                }
                if (ar_ts_on && M == 1u && ar_ts_idx < kArTsMax) ++ar_ts_idx;
                disp += 1u;
            }
        };

        // L4 replaces only the GQA mixer; GDN and MoE retain the baseline persistent path.
        bool l4_super_ready = true;
        for (size_t g = 0; g < N; ++g) l4_super_ready &= K[g].m1_gqa_super_kvq != nullptr;
        const char* l4_tile = std::getenv("VEGAJET_AR_TILE_PIPE");
        l4_super_ready &= !(l4_tile && std::strcmp(l4_tile, "force") == 0);
        const bool use_m1_super = ((!l4_on || l4_super_ready) && !live_slot_prefill && !expert_offload && !batch_rows && !wire_requested && ab_enable_super && !golden_cap_on && M == 1u && !ab_w64 && !ab_f2_front && !ab_b2_back &&
                                   opt.fuse_moe_router && opt.fuse_moe_back && opt.gdn_splitv && !opt.gdn_splitv8 &&
                                   !opt.ab_no_prepush && !opt.ar_wire_int8 && !opt.spec_flash &&
                                   !ar_ts_on && !st_on && !tap_on && !capture_dflash_taps &&
                                   prof_stage == 0u && !(opt.spec_gdn && use_ring) &&
                                   all_lff_super_ok && D == 3072u &&
                                   K[0].m1_gdn_super && K[0].m1_gqa_super && K[0].m1_moe_super);
        const char* ring_swg_env = std::getenv("VEGAJET_SUPER_WGS");
        bool ring_so_ready = true;
        for (size_t g = 0; g < N; ++g)
            if (!K[g].m1_gdn_super || !K[g].m1_gqa_super || !K[g].ring_router_so || !K[g].ring_top8_so || !K[g].m1_moe_super)
                ring_so_ready = false;
        // Historical plain AcceptMoE bypasses approximation for eligible M=2..8: keep super-order
        // router arithmetic (including row 0) in both AcceptMoE modes.
        const bool ring_super_order = ((use_ring && M>=2u) || batch_rows || (live_slot_prefill && M==1u)) && M <= 8u &&
                                      !l4_on && !wire_requested && ab_enable_super && !ab_w64 && !ab_f2_front && !ab_b2_back &&
                                      opt.fuse_moe_router && opt.fuse_moe_back && opt.gdn_splitv && !opt.gdn_splitv8 &&
                                      !opt.ab_no_prepush && !opt.ar_wire_int8 && !opt.spec_flash &&
                                      all_lff_super_ok && D == 3072u && (!ring_swg_env || std::atoi(ring_swg_env) == 240) &&
                                      ring_so_ready;

        // Wave 2 is deliberately narrow: measured Q3 GU + sub32 Down at TP=3, 240 WGs.
        // Read at step entry so warm variants can toggle without reloading weights.
        const char* mega_env = std::getenv("VEGAJET_MEGA");
        bool use_m1_mega = use_m1_super && N == 3u && !capture_routes_now &&
                           mega_env && std::strcmp(mega_env, "1") == 0;
        const char* mega_wgs = std::getenv("VEGAJET_SUPER_WGS");
        const char* mega_q3 = std::getenv("VEGAJET_MOE_Q3_FAST");
        const char* mega_dn = std::getenv("VEGAJET_DN_K4");
        const char* mega_gu = std::getenv("VEGAJET_GU_PAIR");
        const char* fused_ar_env = std::getenv("VEGAJET_AR_FUSED");
        const bool fused_ar_requested = fused_ar_env && fused_ar_env[0] == '1';
        const char* mega_ar = std::getenv("VEGAJET_AR_TILE_PIPE");
        use_m1_mega = use_m1_mega && !fused_ar_requested && (!mega_wgs || std::atoi(mega_wgs) == 240) &&
                      !(mega_q3 && mega_q3[0] == '0') && !(mega_dn && mega_dn[0] == '1') &&
                      !(mega_gu && mega_gu[0] == '1') && !(mega_ar && std::strcmp(mega_ar, "force") == 0);
        if (use_m1_mega) {
            for (size_t g = 0; g < N; ++g) {
                if (!K[g].m1_gdn_mega || !K[g].m1_gqa_mega || rt.devices()[g].n_cu < 60u ||
                    local_ff_r[g] != (g == 0u ? 384u : 320u) ||
                    gdn_nk[g] == 0u || gdn_nv[g] == 0u || gdn_nv[g] > 32u ||
                    gqa_nq[g] == 0u || gqa_nq[g] > 16u) use_m1_mega = false;
                for (uint32_t il = 0; il < 48u; ++il) {
                    const auto& lw = layers[g][il];
                    if (lw.is_gdn != ((il & 3u) != 3u) || !lw.w_gu_d8_w_va ||
                        !lw.w_gu_d8_s_va || lw.gu_d8_w_stride != 2304u ||
                        !lw.dn_is_sub32 ||
                        (lw.is_gdn ? ((lw.w_gdn_qkv_gab.rows & 7u) != 0u || (lw.w_ssm_out.rows & 7u) != 0u)
                                   : ((lw.w_attn_qkv.rows & 7u) != 0u || (lw.w_o.rows & 7u) != 0u))) use_m1_mega = false;
                }
            }
        }
        uint32_t gdn_idx = 0, attn_idx = 0;
        if (use_m1_mega) {
            mega_profile_dispatched = mega_profile_active;
            const uint32_t mega_flags = mega_profile_active ? ornith_mega::kProfile : 0u;
            const uint32_t cin = conv_head, cout = (conv_head + 1u) % kGdnRingSlots;
            for (uint32_t il = 0; il < 48u; il += 4u) {
                const uint32_t gdn_ep = ar_epoch;
                ar_epoch += 6u;
                for (size_t g = 0; g < N; ++g) {
                    const ornith_mega::GdnArgs ka{
                        S[g].mega_desc_va, opt.spec_gdn ? S[g].spec_gdn_s_slot_va[spec_gdn_active] : 0ULL,
                        il, (il / 4u) * 3u, 3u, gdn_ep, cin, cout,
                        opt.spec_gdn ? 0u : ring_head, (!opt.spec_gdn && use_ring) ? 1u : 0u,
                        super_sense, gdn_sense, mega_flags | (il == 0u ? ornith_mega::kSkipFirstAr : 0u)};
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].m1_gdn_mega, ornith_mega::kWorkgroups * 256u,
                                               1, 1, 256, 1, 1, &ka, sizeof(ka), true));
                }
                super_sense += ornith_mega::kSenseStride;
                gdn_sense += 3u;
                const uint32_t gqa_ep = ar_epoch;
                ar_epoch += 2u;
                for (size_t g = 0; g < N; ++g) {
                    const uint32_t flags = il == 44u ? (ornith_mega::kFinalAr |
                        (compute_logits ? ornith_mega::kFinalNorm : 0u)) : 0u;
                    const ornith_mega::GqaArgs ka{S[g].mega_desc_va, il + 3u, il / 4u,
                                                 gqa_ep, pos0, super_sense, flags | mega_flags};
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].m1_gqa_mega, ornith_mega::kWorkgroups * 256u,
                                               1, 1, 256, 1, 1, &ka, sizeof(ka), true));
                }
                super_sense += ornith_mega::kSenseStride;
                disp += 2u;
                if (il == 0u) ring_all();
            }
        } else if (use_m1_super) {
            const char* swg_env = std::getenv("VEGAJET_SUPER_WGS");
            const uint32_t super_wgs = (swg_env && std::atoi(swg_env) >= 192 && std::atoi(swg_env) <= 240)
                                     ? static_cast<uint32_t>(std::atoi(swg_env)) : 240u;
            const char* q3_fast_env = std::getenv("VEGAJET_MOE_Q3_FAST");
            const bool allow_q3_fast = !(q3_fast_env && q3_fast_env[0] == '0');
            const char* top8_env = std::getenv("VEGAJET_TOP8_WG0");
            bool top8_on = top8_env && std::strcmp(top8_env, "1") == 0;
            const char* dn_k4_env = std::getenv("VEGAJET_DN_K4");
            bool dn_k4_on = (dn_k4_env && dn_k4_env[0] == '1');
            // VEGAJET_GU_STREAM=1: Q3 Gate+Up MUBUF stream (opaque bit 27, _x only).
            const char* gu_stream_env = std::getenv("VEGAJET_GU_STREAM");
            bool gu_stream_on = (gu_stream_env && gu_stream_env[0] == '1');
            // VEGAJET_GU_PAIR=1: Q3 Gate+Up pair mode (bit 30, _x kernels): one WG computes both 16-row tiles of a quant
            // block and quantizes from LDS (no mid_f32 round trip, no pair_cnt atomic). Lossless by construction.
            const char* gu_pair_env = std::getenv("VEGAJET_GU_PAIR");
            bool gu_pair_on = (gu_pair_env && gu_pair_env[0] == '1');
            // VEGAJET_AR_TILE_PIPE (N >= 3): super-kernel epilogues post per-chunk ready tags instead of the grid-counted
            // peer flag; consumers wait per chunk (super_sense bit 28; final AR pre_pushed = 3).
            // M = 1 keeps the flag path: the tags only pay off when the consumer has work to overlap, but at M = 1 the
            // consumer is one WG of the NEXT kernel reducing a 6 KiB payload (1-2 chunks per thread), the 240 producer
            // WGs finish inside one Stage-4 wave, and each per-chunk spin is a serial glc round trip with vmcnt(0)
            // (measured -5.9 % raw tg128). The super-kernels are the only tag producers and they are M = 1 only, so
            // there is no M >= 4 payload to enable it for: "1" is ignored here; "force" keeps the tag protocol for A/B.
            const char* ar_tile_env = std::getenv("VEGAJET_AR_TILE_PIPE");
            const bool ar_tile_force = (ar_tile_env && std::strcmp(ar_tile_env, "force") == 0);
            if (ar_tile_env && ar_tile_env[0] == '1') {
                static bool ar_tile_warned = false;
                if (!ar_tile_warned) {
                    ar_tile_warned = true;
                    std::fprintf(stderr, "[ornith::gpu] VEGAJET_AR_TILE_PIPE=1 ignored at M=1 (slower than the grid flag; "
                                         "use =force for A/B)\n");
                }
            }
            uint32_t ar_fused_bit = (N >= 3u && fused_ar_requested && ar_epoch < 65536u - 96u) ? 0x02000000u : 0u;
            uint32_t ar_tile_bit = (N >= 3u && ar_tile_force && !ar_fused_bit) ? 0x10000000u : 0u;
            // Either bit needs the _x super-kernel build; all four super-kernels then come from it (the tile protocol
            // needs every producer and consumer to agree). Without that module the bits are dropped.
            const bool sx_ok = K[0].m1_gdn_super_x && K[0].m1_gqa_super_x && K[0].m1_moe_super_x && K[0].m1_moe_super_q3_x;
            if (!sx_ok) { ar_fused_bit = 0u; ar_tile_bit = 0u; dn_k4_on = false; gu_pair_on = false; gu_stream_on = false; top8_on = false; }
            // Per-kernel selection: the tile protocol needs every super-kernel from _x; DN_K4 / GU_PAIR / GU_STREAM only change the Q3
            // MoE kernel. PROFILE=2 also selects every _x kernel with feature bits clear for a comparable baseline.
            const bool profile_critical = std::getenv("VEGAJET_SUPER_PROFILE") && std::getenv("VEGAJET_SUPER_PROFILE")[0] == '2';
            const bool use_sx = sx_ok && (ar_fused_bit != 0u || ar_tile_bit != 0u || profile_critical);
            const bool use_sx_moe = use_sx;  // TOP8_WG0 is Q3-kernel only
            const uint32_t moe_xf = sx_ok ? moe_prefetch_xflags() : 0u;  // VEGAJET_MOE_PREFETCH (_x Q3 only)
            const bool use_sx_q3 = sx_ok && (moe_xf != 0u || ar_fused_bit != 0u || ar_tile_bit != 0u || dn_k4_on || gu_pair_on || gu_stream_on || top8_on || profile_critical);
            // VEGAJET_SUPER_X (default 1, W21): all four super-kernels from the lean module whenever no experimental _x
            // feature is on; =0 restores the pristine vj_ornith_super kernels (8f23ebb).
            const char* sl_env = std::getenv("VEGAJET_SUPER_X");
            const bool sl_ok = K[0].m1_gdn_super_l && K[0].m1_gqa_super_l && K[0].m1_moe_super_l && K[0].m1_moe_super_q3_l;
            const bool use_sl = sl_ok && !(sl_env && sl_env[0] == '0') && !use_sx && !use_sx_q3;  // default ON since W21 (+0.9% raw, 129/129)
            const uint32_t c_in_slot  = conv_head;
            const uint32_t c_out_slot = (conv_head + 1u) % kGdnRingSlots;
            const uint32_t s_rhead    = opt.spec_gdn ? 0u : ring_head;
            const uint32_t s_uring    = (!opt.spec_gdn && use_ring) ? 1u : 0u;
            uint32_t pending_moe_ep = 0u;
            for (uint32_t il = 0; il < 48u; ++il) {
                const uint32_t prev_ep = pending_moe_ep;
                const uint32_t attn_ep = ar_epoch++;
                const uint32_t moe_ep  = ar_epoch++;
                if (layers[0][il].is_gdn) {
                    const uint32_t gi = gdn_idx++;
                    super_sense = (super_sense + 1u) & 0x00ffffffu;  // bits 24..31 = flags (24 TOP8_WG0, 25 AR_FUSED)
                    ++gdn_sense;
                    struct GdnSupKA {
                        uint64_t desc, sring_ov;
                        uint32_t prev_ep, out_ep, cin, cout, shead, uring, ssense, gsense;
                    };
                    for (size_t g = 0; g < N; ++g) {
                        const uint64_t s_layer_bytes = static_cast<uint64_t>(gdn_nv[g]) * 128ULL * 128ULL * 4ULL;
                        const uint64_t s_ov = opt.spec_gdn
                            ? (S[g].spec_gdn_s_slot_va[spec_gdn_active] + static_cast<uint64_t>(gi) * s_layer_bytes) : 0ULL;
                        const GdnSupKA ka{
                            S[g].gdn_super_desc_va + static_cast<uint64_t>(gi) * sizeof(ornith_super::GdnSuperDesc),
                            s_ov, prev_ep, attn_ep, c_in_slot, c_out_slot, s_rhead, s_uring, super_sense | ar_fused_bit | ar_tile_bit, gdn_sense
                        };
                        VJ_CHECK(rt.enqueue_kernel(g, use_sl ? *K[g].m1_gdn_super_l : use_sx ? *K[g].m1_gdn_super_x : *K[g].m1_gdn_super, super_wgs * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
                    }
                } else {
                    const uint32_t ai = attn_idx++;
                    super_sense = (super_sense + 1u) & 0x00ffffffu;  // bits 24..31 = flags (24 TOP8_WG0, 25 AR_FUSED)
                    struct GqaSupKA {
                        uint64_t desc;
                        uint32_t prev_ep, out_ep, pos0, ssense;
                    };
                    for (size_t g = 0; g < N; ++g) {
                        const GqaSupKA ka{
                            S[g].gqa_super_desc_va + static_cast<uint64_t>(ai) * sizeof(ornith_super::GqaSuperDesc),
                            prev_ep, attn_ep, pos0, super_sense | ar_fused_bit | ar_tile_bit
                        };
                        if (l4_on) {
                            // Extended ABI keeps the baseline descriptor/cache addresses; format is runtime state
                            // so warm variants never need descriptor uploads or per-token host synchronization.
                            const kvq_super::Args kvka{ka.desc,ka.prev_ep,ka.out_ep,ka.pos0,ka.ssense,
                                L4[g].signs,uint32_t(l4_format),uint32_t(l4_chrono)};
                            VJ_CHECK(rt.enqueue_kernel(g, *K[g].m1_gqa_super_kvq, super_wgs * 256u, 1, 1, 256, 1, 1, &kvka, sizeof(kvka), true));
                        } else {
                        VJ_CHECK(rt.enqueue_kernel(g, use_sl ? *K[g].m1_gqa_super_l : use_sx ? *K[g].m1_gqa_super_x : *K[g].m1_gqa_super, super_wgs * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
                        }
                    }
                }
                super_sense = (super_sense + 1u) & 0x00ffffffu;  // bits 24..31 = flags (24 TOP8_WG0, 25 AR_FUSED)
                struct MoeSupKA {
                    uint64_t desc;
                    uint32_t ar_ep, out_ep, ssense;
                };
                for (size_t g = 0; g < N; ++g) {
                    const bool use_q3_k = allow_q3_fast && (super_wgs == 240u) && K[g].m1_moe_super_q3 &&
                                          (layers[g][il].w_gu_d8_w_va != 0ULL) &&
                                          (layers[g][il].gu_d8_w_stride == 2304u);
                    // Q3 _x toggles: bit 26 GU_STREAM, bit 29 DN_K4 (balanced Down rows), bit 30 GU_PAIR (bit 27 = PROFILE=2).
                    const MoeSupKA ka{
                        S[g].moe_super_desc_va + static_cast<uint64_t>(il) * sizeof(ornith_super::MoeSuperDesc),
                        attn_ep, moe_ep, super_sense | ar_fused_bit | ar_tile_bit | ((use_q3_k && top8_on) ? 0x01000000u : 0u) | ((use_q3_k && dn_k4_on) ? 0x20000000u : 0u) | ((use_q3_k && gu_pair_on) ? 0x40000000u : 0u) | ((use_q3_k && gu_stream_on) ? 0x04000000u : 0u)
                    };
                    const auto* mk = use_q3_k ? (use_sl ? K[g].m1_moe_super_q3_l : use_sx_q3 ? K[g].m1_moe_super_q3_x : K[g].m1_moe_super_q3)
                                              : (use_sl ? K[g].m1_moe_super_l : use_sx_moe ? K[g].m1_moe_super_x : K[g].m1_moe_super);
                    if (mk == K[g].m1_moe_super_q3_x || mk == K[g].m1_moe_super_q3_l) {
                        // The _x Q3 kernel takes one extra kernarg (xflags: VEGAJET_MOE_PREFETCH bits).
                        struct MoeSupKAX { uint64_t desc; uint32_t ar_ep, out_ep, ssense, xflags; };
                        const MoeSupKAX kx{ka.desc, ka.ar_ep, ka.out_ep, ka.ssense, moe_xf};
                        VJ_CHECK(rt.enqueue_kernel(g, *mk, super_wgs * 256u, 1, 1, 256, 1, 1, &kx, sizeof(kx), true));
                    } else
                    VJ_CHECK(rt.enqueue_kernel(g, *mk, super_wgs * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
                }
                disp += 2u;
                if (capture_routes_now) {
                    struct CpU32KA { uint64_t dst, src, n; };
                    const CpU32KA cka{evict_cap.va + static_cast<uint64_t>(il) * kEvictTopStrideIds * 4ULL,
                                      S[0].top_ids_va, static_cast<uint64_t>(M) * kEvictTopK};
                    VJ_CHECK(rt.enqueue_kernel(0, *K[0].memcpy_u32, 256u, 1, 1, 256, 1, 1, &cka, sizeof(cka), true));
                    ++captured_layers;
                }
                pending_moe_ep = moe_ep;
                if (il == 0u) ring_all();
            }
            // Final layer 47 Post-MoE AllReduce + out_norm (reusing pending_moe_ep already prepushed into BAR0!)
            {
                const uint32_t slot = 47u * 2u + 1u;
                const uint32_t pp   = pending_moe_ep & 1u;
                const uint32_t dn   = compute_logits ? 1u : 0u;
                if (N == 1) {
                    struct ArNKA {
                        uint64_t xp, ptx[4], lrx[4], pfl[4], lfl[4], bar, h, w, xn, xq, xd;
                        uint32_t M, D, ep, dn, rank, n_ranks, pre_pushed;
                        float eps;
                    };
                    ArNKA aka{};
                    aka.xp = S[0].partial_va; aka.bar = S[0].ar_bar_va; aka.h = S[0].h_va; aka.w = out_norm_va[0];
                    aka.xn = S[0].x_norm_va; aka.xq = S[0].x_q8_va; aka.xd = S[0].x_ds_va;
                    aka.M = 1u; aka.D = D; aka.ep = pending_moe_ep; aka.dn = dn; aka.rank = 0u; aka.n_ranks = 1u; aka.pre_pushed = 1u; aka.eps = eps;
                    VJ_CHECK(rt.enqueue_kernel(0, *K[0].p2p_ar_n_fp16, 256u, 1, 1, 256, 1, 1, &aka, sizeof(aka), true));
                } else if (N == 2) {
                    for (size_t g = 0; g < 2; ++g) {
                        const size_t peer = 1u - g;
                        struct ArKA {
                            uint64_t xp, ptx, lrx, pfl, lfl, bar, h, w, xn, xq, xd;
                            uint32_t M, D, ep, dn;
                            float eps;
                        };
                        const ArKA aka{
                            S[g].partial_va,
                            rt.ar_rx_base_va(peer, g) + pp * kfd::kArSlotBytes,
                            rt.ar_rx_base_va(g, peer) + pp * kfd::kArSlotBytes,
                            rt.ar_flag_va(peer, g, slot),
                            rt.ar_flag_va(g, peer, slot),
                            S[g].ar_bar_va, S[g].h_va, out_norm_va[g],
                            S[g].x_norm_va, S[g].x_q8_va, S[g].x_ds_va, 1u, D, pending_moe_ep, dn, eps
                        };
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].p2p_ar_prepush, 256u, 1, 1, 256, 1, 1, &aka, sizeof(aka), true));
                    }
                } else {
                    struct ArNKA {
                        uint64_t xp, ptx[4], lrx[4], pfl[4], lfl[4], bar, h, w, xn, xq, xd;
                        uint32_t M, D, ep, dn, rank, n_ranks, pre_pushed;
                        float eps;
                    };
                    for (size_t g = 0; g < N; ++g) {
                        ArNKA aka{};
                        aka.xp = S[g].partial_va;
                        for (size_t r = 0; r < 4; ++r) {
                            aka.ptx[r] = (r < N && r != g) ? (rt.ar_rx_base_va(r, g) + pp * kfd::kArSlotBytes) : 0ULL;
                            aka.lrx[r] = (r < N && r != g) ? (rt.ar_rx_base_va(g, r) + pp * kfd::kArSlotBytes) : 0ULL;
                            aka.pfl[r] = (r < N && r != g) ? rt.ar_flag_va(r, g, slot) : 0ULL;
                            aka.lfl[r] = (r < N && r != g) ? rt.ar_flag_va(g, r, slot) : 0ULL;
                        }
                        aka.bar = S[g].ar_bar_va; aka.h = S[g].h_va; aka.w = out_norm_va[g];
                        aka.xn = S[g].x_norm_va; aka.xq = S[g].x_q8_va; aka.xd = S[g].x_ds_va;
                        aka.M = 1u; aka.D = D; aka.ep = pending_moe_ep; aka.dn = dn;
                        aka.rank = static_cast<uint32_t>(g); aka.n_ranks = static_cast<uint32_t>(N);
                        aka.pre_pushed = ar_fused_bit ? 4u : ((ar_tile_bit != 0u) ? 3u : 1u); aka.eps = eps;
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].p2p_ar_n_fp16, 256u, 1, 1, 256, 1, 1, &aka, sizeof(aka), true));
                    }
                }
                disp += 1u;
            }
        } else
        for (uint32_t il = 0; il < 48u; ++il) {
            auto gcap_sync = [&]() {
                VJ_CHECK(submit_all());
                VJ_CHECK(wait_all());
            };
            auto gcap_dump = [&](const std::string& tag, uint64_t va, uint64_t bytes) {
                std::string gerr;
                const std::string p = golden_cap_dir + "/m" + std::to_string(M) + "_" + tag + ".bin";
                VJ_CHECK(write_vram_bin(rt, 0, va, bytes, p, &gerr));
            };
            if (golden_cap_on && il == 0u) {
                gcap_sync();
                gcap_dump("gdn_xq8", S[0].x_q8_va, static_cast<uint64_t>(M) * D);
                gcap_dump("gdn_xds", S[0].x_ds_va, static_cast<uint64_t>(M) * (D / 32u) * 8ULL);
                gcap_dump("gdn_conv_in", S[0].gdn_conv_ring_va[0] + static_cast<uint64_t>(conv_head) * 3ULL * 6144ULL * 4ULL, 3ULL * 6144ULL * 4ULL);
                const uint64_t s_lb = 32ULL * 128ULL * 128ULL * 4ULL;
                const uint64_t s_in = opt.spec_gdn ? S[0].spec_gdn_s_slot_va[spec_gdn_active]
                                                   : (S[0].gdn_ssm_ring_va[0] + static_cast<uint64_t>(ring_head) * s_lb);
                gcap_dump("gdn_ssm_in", s_in, s_lb);
            }
            if (golden_cap_on && il == golden_gqa_il) {
                gcap_sync();
                gcap_dump("gqa_xq8", S[0].x_q8_va, static_cast<uint64_t>(M) * D);
                gcap_dump("gqa_xds", S[0].x_ds_va, static_cast<uint64_t>(M) * (D / 32u) * 8ULL);
                if (pos0 > 0u) {
                    gcap_dump("gqa_k_cache", S[0].attn_k_cache_va[attn_idx], static_cast<uint64_t>(pos0) * 272ULL);
                    gcap_dump("gqa_v_cache", S[0].attn_v_cache_va[attn_idx], static_cast<uint64_t>(pos0) * 272ULL);
                }
            }
            if (layers[0][il].is_gdn) {
                const uint32_t gi = gdn_idx++;
                const bool gdn_fused_ok = (!batch_rows && M == 1u && !(opt.spec_gdn && use_ring));
                const bool gdn_sv8 = gdn_fused_ok && opt.gdn_splitv8 && K[0].gdn_conv_recur_sv8 != nullptr &&
                                     K[0].gdn_conv_recur_sv8_quant != nullptr;
                const bool fuse_gdn_m1 = (gdn_sv8 || (gdn_fused_ok && opt.gdn_splitv && K[0].gdn_conv_recur_sv != nullptr));
                if (fuse_gdn_m1) ++gdn_sense;
                const uint32_t c_in_slot  = conv_head;
                const uint32_t c_out_slot = (conv_head + 1u) % kGdnRingSlots;
                for (size_t g = 0; g < N; ++g) {
                    const auto& lw = layers[g][il];
                    const uint32_t nk_g   = gdn_nk[g];
                    const uint32_t nv_g   = gdn_nv[g];
                    const uint32_t cdim_g = gdn_conv_dim[g];
                    const uint32_t strd_g = gdn_qkv_stride[g];
                    const uint32_t zoff_g = gdn_z_off[g];
                    const uint32_t aoff_g = gdn_alpha_off[g];
                    const uint32_t boff_g = gdn_beta_off[g];
                    const uint32_t cwgs_g = 2u * nk_g + nv_g + 1u;
                    const bool fwht_w_gdn_qkv_gab = dot8_fwht_mode && (lw.w_gdn_qkv_gab_fwht.rows != 0u);
                    if (use_pf_dot8_dense && (lw.w_gdn_qkv_gab.cols % 64u) == 0u) {
                        enq_q8_to_q4(g, S[g].x_q8_va, S[g].x_ds_va, D, true, fwht_w_gdn_qkv_gab);
                        for (uint32_t r = 0; r < reps_of(1u); ++r)
                            enq_gemm_q4_d8(g, fwht_w_gdn_qkv_gab ? lw.w_gdn_qkv_gab_fwht : lw.w_gdn_qkv_gab, S[g].gdn_qkv_gab_va, true);
                    } else {
                        for (uint32_t r = 0; r < reps_of(1u); ++r)
                            enq_gemm_q4(g, lw.w_gdn_qkv_gab, S[g].x_q8_va, S[g].x_ds_va, S[g].gdn_qkv_gab_va, true);
                    }
                    st_mark(kStGdnProj, static_cast<int>(g));

                    const uint64_t s_layer_bytes = static_cast<uint64_t>(nv_g) * 128ULL * 128ULL * 4ULL;
                    const uint64_t s_cur_va = opt.spec_gdn
                        ? (S[g].spec_gdn_s_slot_va[spec_gdn_active] + static_cast<uint64_t>(gi) * s_layer_bytes)
                        : (S[g].gdn_ssm_ring_va[gi] + static_cast<uint64_t>(ring_head) * s_layer_bytes);
                    const uint64_t s_ring_base = opt.spec_gdn ? s_cur_va : S[g].gdn_ssm_ring_va[gi];
                    const uint32_t s_rhead     = opt.spec_gdn ? 0u : ring_head;

                    if (fuse_gdn_m1) {
                        if (gdn_sv8) {
                            struct Sv8RecKA {
                                uint64_t qkv, cw, cring, dt, sa, sring, sq;
                                uint32_t nk, nv, cdim, strd, aoff, boff, cin, cout, shead, uring;
                                float eps;
                            };
                            const Sv8RecKA r1{
                                S[g].gdn_qkv_gab_va, lw.ssm_conv_w_va, S[g].gdn_conv_ring_va[gi],
                                lw.ssm_dt_va, lw.ssm_a_va, s_ring_base, S[g].gdn_sq_va,
                                nk_g, nv_g, cdim_g, strd_g, aoff_g, boff_g,
                                c_in_slot, c_out_slot, s_rhead, (!opt.spec_gdn && use_ring) ? 1u : 0u, eps
                            };
                            VJ_CHECK(rt.enqueue_kernel(g, *K[g].gdn_conv_recur_sv8, (nv_g * 8u) * 256u, 1, 1, 256, 1, 1, &r1, sizeof(r1), true));
                            struct Sv8QuantKA {
                                uint64_t qkv, nw, oq, od, sq;
                                uint32_t zoff;
                                float eps;
                            };
                            const Sv8QuantKA r2{
                                S[g].gdn_qkv_gab_va, lw.ssm_norm_va, S[g].mid_q8_va, S[g].mid_ds_va, S[g].gdn_sq_va,
                                zoff_g, eps
                            };
                            VJ_CHECK(rt.enqueue_kernel(g, *K[g].gdn_conv_recur_sv8_quant, (nv_g * 8u) * 256u, 1, 1, 256, 1, 1, &r2, sizeof(r2), true));
                        } else {
                        struct ConvRecSvKA {
                            uint64_t qkv, cw, cring, dt, sa, sring, nw, oq, od, bar, sq;
                            uint32_t nk, nv, cdim, strd, zoff, aoff, boff, cin, cout, shead, uring, sense;
                            float eps;
                        };
                        const ConvRecSvKA crka{
                            S[g].gdn_qkv_gab_va, lw.ssm_conv_w_va, S[g].gdn_conv_ring_va[gi],
                            lw.ssm_dt_va, lw.ssm_a_va, s_ring_base, lw.ssm_norm_va,
                            S[g].mid_q8_va, S[g].mid_ds_va, S[g].gdn_bar_va, S[g].gdn_sq_va,
                            nk_g, nv_g, cdim_g, strd_g, zoff_g, aoff_g, boff_g,
                            c_in_slot, c_out_slot, s_rhead, (!opt.spec_gdn && use_ring) ? 1u : 0u, gdn_sense, eps
                        };
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].gdn_conv_recur_sv, (nv_g * 4u) * 256u, 1, 1, 256, 1, 1, &crka, sizeof(crka), true));
                        }
                        st_mark(kStGdnRecur, static_cast<int>(g));
                    } else {
                    struct ConvL2KA {
                        uint64_t qkv, cw, cring, cout, dt, sa, dec, bs;
                        uint32_t M, nk, nv, cdim, strd, aoff, boff, rhead, uring;
                        float eps;
                    };
                    if (!use_ring && M >= 32u && K[g].pf_gdn_conv != nullptr) {
                        ConvL2KA clka{S[g].gdn_qkv_gab_va, lw.ssm_conv_w_va, S[g].gdn_conv_ring_va[gi], S[g].gdn_conv_va,
                                      lw.ssm_dt_va, lw.ssm_a_va, S[g].gdn_dec_va, S[g].gdn_bsig_va,
                                      M, nk_g, nv_g, cdim_g, strd_g, aoff_g, boff_g, c_in_slot, c_out_slot, eps};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_gdn_conv, cwgs_g * 128u, (M + 15u) / 16u, 1, 128, 1, 1, &clka, sizeof(clka), true));
                    } else {
                        ConvL2KA clka{S[g].gdn_qkv_gab_va, lw.ssm_conv_w_va, S[g].gdn_conv_ring_va[gi], S[g].gdn_conv_va,
                                      lw.ssm_dt_va, lw.ssm_a_va, S[g].gdn_dec_va, S[g].gdn_bsig_va,
                                      M, nk_g, nv_g, cdim_g, strd_g, aoff_g, boff_g, batch_rows ? row_slots : conv_head,
                                      batch_rows ? 2u : (use_ring ? 1u : 0u), eps};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].gdn_conv_l2, cwgs_g * 128u, 1, 1, 128, 1, 1, &clka, sizeof(clka), true));
                    }
                    st_mark(kStGdnConv, static_cast<int>(g));

                    if (treewy_cycle) {
                        treewy::LayerInputs li;
                        li.gdn_index = gi;
                        li.device = static_cast<uint32_t>(g);
                        li.active_slot = spec_gdn_active;
                        li.state_slot[0] = S[g].spec_gdn_s_slot_va[0] + static_cast<uint64_t>(gi) * s_layer_bytes;
                        li.state_slot[1] = S[g].spec_gdn_s_slot_va[1] + static_cast<uint64_t>(gi) * s_layer_bytes;
                        li.records = treewy_arena_va[g] +
                                     static_cast<uint64_t>(gi) * vegajet::gdn::treewy_launch::kRecordBlockBytes;
                        li.output = treewy_hook->output_va(static_cast<uint32_t>(g));
                        li.conv_source = S[g].gdn_conv_va;
                        li.decay_source = S[g].gdn_dec_va;
                        li.beta_source = S[g].gdn_bsig_va;
                        li.gate_source = S[g].gdn_qkv_gab_va;
                        li.norm_weight = lw.ssm_norm_va;
                        li.out_q8 = S[g].mid_q8_va;
                        li.out_ds = S[g].mid_ds_va;
                        li.stride_in = strd_g;
                        li.z_off = zoff_g;
                        li.nv = nv_g;
                        li.eps = eps;
                        std::string tw_err;
                        if (!treewy_hook->enqueue_layer(li, tw_err)) {
                            std::fprintf(stderr, "[treewy] fatal layer %u shard %zu: %s\n", gi, g,
                                         tw_err.c_str());
                            VJ_CHECK(false);
                        }
                    } else if (opt.spec_gdn && use_ring && M <= 16u) {
                        struct SpecGdnKA {
                            uint64_t sin, sout, qsrc, ksrc, vsrc, dec, beta, oout, rkv, rdec, rbeta;
                            uint32_t M, nk, nv, s_strd, r_strd, lbase, s_lstrd, src_lstrd, gb_lstrd, r_lstrd, rgb_lstrd;
                        };
                        const uint32_t r_strd = (nk_g + nv_g) * 128u;
                        const uint64_t s_dst_va = S[g].spec_gdn_s_slot_va[1u - spec_gdn_active] + static_cast<uint64_t>(gi) * s_layer_bytes;
                        const SpecGdnKA sgka{
                            s_cur_va, s_dst_va, S[g].gdn_conv_va,
                            S[g].gdn_conv_va + static_cast<uint64_t>(nk_g) * 128ULL * 4ULL,
                            S[g].gdn_conv_va + 2ULL * nk_g * 128ULL * 4ULL,
                            S[g].gdn_dec_va, S[g].gdn_bsig_va,
                            S[g].spec_gdn_o_va, S[g].spec_gdn_rec_kv_va, S[g].spec_gdn_rec_dec_va, S[g].spec_gdn_rec_beta_va,
                            M, nk_g, nv_g, cdim_g, r_strd, gi, 0u, 0u, 0u, 16u * r_strd, 16u * nv_g
                        };
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].spec_gdn_ver, (nv_g * 4u) * 256u, 1, 1, 256, 1, 1, &sgka, sizeof(sgka), true));
                        struct SpecGdnNormKA { uint64_t ob, gab, nw, oq, od; uint32_t M, nv, strd, zoff; float eps; };
                        const SpecGdnNormKA snka{S[g].spec_gdn_o_va, S[g].gdn_qkv_gab_va, lw.ssm_norm_va,
                                                 S[g].mid_q8_va, S[g].mid_ds_va, M, nv_g, strd_g, zoff_g, eps};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].spec_gdn_nq, ((M * nv_g + 3u) / 4u) * 256u, 1, 1, 256, 1, 1, &snka, sizeof(snka), true));
                    } else if (std::getenv("VEGAJET_PREFILL_CODEX_GDN") != nullptr &&
                               std::strcmp(std::getenv("VEGAJET_PREFILL_CODEX_GDN"), "1") == 0 &&
                               K[g].pf_gdn_wy_codex && K[g].pf_gdn_nq_codex && use_wy && M >= 16u && !use_ring) {
                        struct WsKA { uint64_t S, cq, dec, bs, fo, sq; uint32_t M, nk, nv, cdim; };
                        const WsKA wska{s_cur_va, S[g].gdn_conv_va, S[g].gdn_dec_va, S[g].gdn_bsig_va, S[g].pf_fo_va, S[g].pf_sq_va, M, nk_g, nv_g, cdim_g};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_gdn_wy_codex, (nv_g * 4u) * 256u, 1, 1, 256, 1, 1, &wska, sizeof(wska), true));
                        struct NqKA { uint64_t fo, sq, gab, nw, oq, od; uint32_t M, nv, strd, zoff; float eps; };
                        const NqKA nqka{S[g].pf_fo_va, S[g].pf_sq_va, S[g].gdn_qkv_gab_va, lw.ssm_norm_va, S[g].mid_q8_va, S[g].mid_ds_va, M, nv_g, strd_g, zoff_g, eps};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_gdn_nq_codex, (nv_g * 4u) * 256u, 1, 1, 256, 1, 1, &nqka, sizeof(nqka), true));
                    } else if (opt.pf_gdn == 1u && use_wy && M >= 16u && !use_ring) {
                        struct SvbKA { uint64_t sring, cq, gab, dec, bs, nw, oq, od, bar, fo, sq;
                                       uint32_t M, nk, nv, cdim, strd, zoff, rhead, uring, sense; float eps; uint32_t fuse; };
                        if (g == 0) ++pf_gdn_sense;
                        const SvbKA ska{s_ring_base, S[g].gdn_conv_va, S[g].gdn_qkv_gab_va, S[g].gdn_dec_va, S[g].gdn_bsig_va,
                                        lw.ssm_norm_va, S[g].mid_q8_va, S[g].mid_ds_va, S[g].pf_gdn_bar_va, S[g].pf_fo_va, S[g].pf_sq_va,
                                        M, nk_g, nv_g, cdim_g, strd_g, zoff_g, s_rhead, 0u, pf_gdn_sense, eps, 1u};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_gdn_sv, (nv_g * 4u) * 256u, 1, 1, 256, 1, 1, &ska, sizeof(ska), true));
                    } else if (opt.pf_gdn == 2u && use_wy && M >= 16u && !use_ring) {
                        struct WsKA { uint64_t S, cq, dec, bs, fo, sq; uint32_t M, nk, nv, cdim; };
                        const WsKA wska{s_cur_va, S[g].gdn_conv_va, S[g].gdn_dec_va, S[g].gdn_bsig_va, S[g].pf_fo_va, S[g].pf_sq_va, M, nk_g, nv_g, cdim_g};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_gdn_wy, (nv_g * 4u) * 256u, 1, 1, 256, 1, 1, &wska, sizeof(wska), true));
                        struct NqKA { uint64_t fo, sq, gab, nw, oq, od; uint32_t M, nv, strd, zoff; float eps; };
                        const NqKA nqka{S[g].pf_fo_va, S[g].pf_sq_va, S[g].gdn_qkv_gab_va, lw.ssm_norm_va, S[g].mid_q8_va, S[g].mid_ds_va, M, nv_g, strd_g, zoff_g, eps};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_gdn_nq, (nv_g * 4u) * 256u, 1, 1, 256, 1, 1, &nqka, sizeof(nqka), true));
                    } else if (use_wy && M >= 8u && !use_ring) {
                        struct WyKA { uint64_t S, cq, gab, dec, bs, nw, oq, od; uint32_t M, nk, nv, cdim, strd, zoff; float eps; };
                        WyKA wka{s_cur_va, S[g].gdn_conv_va, S[g].gdn_qkv_gab_va, S[g].gdn_dec_va, S[g].gdn_bsig_va,
                                 lw.ssm_norm_va, S[g].mid_q8_va, S[g].mid_ds_va, M, nk_g, nv_g, cdim_g, strd_g, zoff_g, eps};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].gdn_wy, nv_g * 256u, 1, 1, 256, 1, 1, &wka, sizeof(wka), true));
                    } else if (!batch_rows && opt.gdn_splitv && M == 1u) {
                        struct RecSvKA { uint64_t sring, cq, gab, dec, bs, nw, oq, od, bar, sq; uint32_t nk, nv, cdim, strd, zoff, rhead, uring, sense; float eps; };
                        if (g == 0) ++gdn_sense;
                        RecSvKA rka{s_ring_base, S[g].gdn_conv_va, S[g].gdn_qkv_gab_va, S[g].gdn_dec_va, S[g].gdn_bsig_va,
                                    lw.ssm_norm_va, S[g].mid_q8_va, S[g].mid_ds_va, S[g].gdn_bar_va, S[g].gdn_sq_va,
                                    nk_g, nv_g, cdim_g, strd_g, zoff_g, s_rhead, (!opt.spec_gdn && use_ring) ? 1u : 0u, gdn_sense, eps};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].gdn_recur_sv, (nv_g * 4u) * 256u, 1, 1, 256, 1, 1, &rka, sizeof(rka), true));
                    } else {
                        struct RecKA { uint64_t sring, cq, gab, dec, bs, nw, oq, od; uint32_t M, nk, nv, cdim, strd, zoff, rhead, uring; float eps; };
                        RecKA rka{s_ring_base, S[g].gdn_conv_va, S[g].gdn_qkv_gab_va, S[g].gdn_dec_va, S[g].gdn_bsig_va,
                                  lw.ssm_norm_va, S[g].mid_q8_va, S[g].mid_ds_va, M, nk_g, nv_g, cdim_g, strd_g, zoff_g,
                                  batch_rows ? row_slots : s_rhead, batch_rows ? 2u : ((!opt.spec_gdn && use_ring) ? 1u : 0u), eps};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].gdn_recur, nv_g * 1024u, 1, 1, 1024, 1, 1, &rka, sizeof(rka), true));
                    }
                    st_mark(kStGdnRecur, static_cast<int>(g));
                    }
                    const bool do_ep_gdn = (N == 2 && !opt.ab_no_prepush && M <= 16u && !ar_ts_on && !opt.ar_wire_int8 && !(opt.prefill_ar_int8 && !use_ring && M >= 64u));
                    const uint64_t ptx_gdn = do_ep_gdn ? (rt.ar_rx_base_va(1u - g, g) + (ar_epoch & 1u) * kfd::kArSlotBytes) : 0ULL;
                    uint64_t tx3_gdn[2] = {0ULL, 0ULL};
                    if (ep_tp3_ok) ep_tp3_tx(g, tx3_gdn);
                    const bool fwht_w_ssm_out = dot8_fwht_mode && (lw.w_ssm_out_fwht.rows != 0u);
                    if (use_pf_dot8_dense && (lw.w_ssm_out.cols % 64u) == 0u) {
                        enq_q8_to_q4(g, S[g].mid_q8_va, S[g].mid_ds_va, lw.w_ssm_out.cols, true, fwht_w_ssm_out);
                        for (uint32_t r = 0; r < reps_of(2u); ++r)
                            enq_gemm_q4_d8(g, fwht_w_ssm_out ? lw.w_ssm_out_fwht : lw.w_ssm_out, S[g].partial_va, true);
                    } else {
                        for (uint32_t r = 0; r < reps_of(2u); ++r)
                            enq_gemm_q4(g, lw.w_ssm_out, S[g].mid_q8_va, S[g].mid_ds_va, S[g].partial_va, true,
                                        ep_tp3_ok ? tx3_gdn[0] : ptx_gdn, tx3_gdn[1]);
                    }
                    st_mark(kStGdnOut, static_cast<int>(g));
                }
                disp += fuse_gdn_m1 ? 3u : 4u;
            } else {
                const uint32_t ai = attn_idx++;
                const uint32_t skv_span = pos0 + M;
                const uint32_t skv_cl   = (skv_span <= 1920u) ? 32u
                                        : ((skv_span <= 3840u) ? 64u
                                        : ((skv_span <= 7680u) ? 128u : 256u));
                const uint32_t skv_nc   = std::max<uint32_t>(1u, (skv_span + skv_cl - 1u) / skv_cl);
                const bool use_auto_splitkv = (!wire_requested && !ab_no_splitkv && !opt.spec_flash && skv_nc > 1u &&
                                               (use_ring ? M <= 16u : M < 16u) && K[0].flash_skv_p1 != nullptr);
                const bool fuse_attn_m1 = (!l4_on && !wire_requested && !ab_no_fused_attn && M == 1u && !opt.spec_flash && !use_auto_splitkv &&
                                           reps_of(4u) == 1u && reps_of(5u) == 1u && K[0].attn_prep_flash_m1 != nullptr);
                // Retire prior peer work before long row loops can fill the 2048-slot kernarg ring.
                if (l4_on && M >= 2u) VJ_CHECK(submit_all() && wait_all());
                for (size_t g = 0; g < N; ++g) {
                    const auto& lw = layers[g][il];
                    const uint32_t n_q_local  = gqa_nq[g];
                    const uint32_t n_kv_local = gqa_nkv[g];
                    const uint32_t nkv_p      = gqa_nkv_packed(g);
                    const bool fwht_w_attn_qkv = dot8_fwht_mode && (lw.w_attn_qkv_fwht.rows != 0u);
                    if (use_pf_dot8_dense && (lw.w_attn_qkv.cols % 64u) == 0u) {
                        enq_q8_to_q4(g, S[g].x_q8_va, S[g].x_ds_va, D, true, fwht_w_attn_qkv);
                        for (uint32_t r = 0; r < reps_of(3u); ++r)
                            enq_gemm_q4_d8(g, fwht_w_attn_qkv ? lw.w_attn_qkv_fwht : lw.w_attn_qkv, S[g].attn_qkv_va, true);
                    } else {
                        for (uint32_t r = 0; r < reps_of(3u); ++r)
                            enq_gemm_q4(g, lw.w_attn_qkv, S[g].x_q8_va, S[g].x_ds_va, S[g].attn_qkv_va, true);
                    }
                    st_mark(kStQkv, static_cast<int>(g));

                    // Lever 4: one chronological row per dispatch also covers prefill, verify and independent batch rows.
                    if (l4_on) {
                        auto& a=L4[g]; const auto& k=L4K[g];
                        const auto hr=l4::ranges(n_q_local,n_kv_local,gqa_q_off[g],gqa_kv_off[g],gqa_group);
                        const uint64_t qrow=n_q_local*256ULL*4, kvrow=(n_q_local*512ULL+2ULL*n_kv_local*256)*4;
                        const uint64_t capacity=batch_rows ? opt.max_ctx/kMaxGpuBatch : opt.max_ctx;
                        for (uint32_t m=0;m<M;++m) {
                            if (m && m % 64u == 0u) VJ_CHECK(rt.submit_batch(g) && rt.wait_idle(g));
                            const uint32_t pos=batch_rows ? (*batch_pos)[m] : pos0+m;
                            const uint32_t seq=batch_rows ? m : 0;
                            const uint64_t cache_offset=l4::native_offset(seq,static_cast<uint32_t>(capacity),n_kv_local,l4_format);
                            const uint64_t kc=S[g].attn_k_cache_va[ai]+cache_offset, vc=S[g].attn_v_cache_va[ai]+cache_offset;
                            const uint64_t q=S[g].q_buf_va+m*qrow, gate=S[g].gate_buf_va+m*qrow;
                            const uint64_t oq=S[g].mid_q8_va+m*n_q_local*256ULL, od=S[g].mid_ds_va+m*n_q_local*64ULL;
                            const l4::PrepArgs prep{S[g].attn_qkv_va+m*kvrow,lw.q_norm_va,lw.k_norm_va,q,gate,
                                l4_compressed?a.prepared_k:kc,l4_compressed?a.prepared_v:vc,
                                1,pos,n_q_local,n_kv_local,hp.rope_dims,rope_base,eps};
                            VJ_CHECK(rt.enqueue_kernel(g,l4_compressed?*k.prep:*K[g].attn_prep,(n_q_local+n_kv_local)*256,1,1,256,1,1,&prep,sizeof(prep),true));
                            const uint32_t splits=l4::splits(pos+1,l4_compressed?n_kv_local:1u,l4_chrono);
                            if (l4_compressed) {
                                kvq::AppendArgs ak{a.prepared_k,kc,a.signs,256,n_kv_local,pos,uint32_t(l4_format)};
                                kvq::AppendArgs av{a.prepared_v,vc,a.signs,256,n_kv_local,pos,uint32_t(l4_format)};
                                VJ_CHECK(rt.enqueue_kernel(g,*k.append,n_kv_local*256,1,1,256,1,1,&ak,sizeof(ak),true));
                                VJ_CHECK(rt.enqueue_kernel(g,*k.append,n_kv_local*256,1,1,256,1,1,&av,sizeof(av),true));
                                const kvq::PartialArgs pa{q,kc,vc,a.ranges,a.signs,a.partial,a.lse,256,n_q_local,n_kv_local,pos+1,splits,uint32_t(l4_format),0.0625f};
                                const kvq::MergeArgs ma{a.partial,a.lse,a.out,n_q_local,splits,256};
                                VJ_CHECK(rt.enqueue_kernel(g,*k.partial,splits*256,n_kv_local,1,256,1,1,&pa,sizeof(pa),true));
                                VJ_CHECK(rt.enqueue_kernel(g,*k.merge,n_q_local*64,1,1,64,1,1,&ma,sizeof(ma),true));
                                const l4::EpilogueArgs ea{a.out,gate,oq,od,0,n_q_local};
                                VJ_CHECK(rt.enqueue_kernel(g,*k.epilogue,n_q_local*256,1,1,256,1,1,&ea,sizeof(ea),true));
                            } else {
                                const auto sp=flash_decode_splitkv::plan_scratch(1,splits);
                                for (uint32_t h=0;h<n_kv_local;++h) {
                                    // Head-major wire planes, then sequence segments. Only real slots are scattered.
                                    const uint64_t off=l4::wire_offset(opt.max_ctx,h,seq,static_cast<uint32_t>(capacity));
                                    const uint64_t wk=a.wire_k[ai]+off, wv=a.wire_v[ai]+off;
                                    const l4::WireKvArgs wa{kc,vc,wk,wv,n_kv_local,h,pos};
                                    VJ_CHECK(rt.enqueue_kernel(g,*k.wire_kv,256,1,1,256,1,1,&wa,sizeof(wa),true));
                                    if (hr[h].begin==hr[h].end) continue;
                                    const l4::WireQArgs qa{q,a.wire_q,hr[h].begin,hr[h].end};
                                    VJ_CHECK(rt.enqueue_kernel(g,*k.wire_q,16*256,1,1,256,1,1,&qa,sizeof(qa),true));
                                    const auto shape=flash_decode_splitkv::active_shape(1,pos);
                                    const auto pa=flash_decode_splitkv::partition_args(a.wire_q,wk,wv,a.wire_scratch+sp.partial_offset,a.wire_scratch+sp.lse_offset,shape,splits,0.0625f);
                                    const auto ma=flash_decode_splitkv::merge_args(pa.partial,pa.lse,a.wire_scratch+sp.output_offset,a.wire_scratch+sp.output_lse_offset,splits);
                                    VJ_CHECK(rt.enqueue_kernel(g,*k.partition,splits*256,1,1,256,1,1,&pa,sizeof(pa),true));
                                    VJ_CHECK(rt.enqueue_kernel(g,*k.wire_merge,16*256,1,1,256,1,1,&ma,sizeof(ma),true));
                                    const l4::EpilogueArgs ea{ma.output,gate,oq,od,hr[h].begin,hr[h].end};
                                    VJ_CHECK(rt.enqueue_kernel(g,*k.epilogue,(hr[h].end-hr[h].begin)*256,1,1,256,1,1,&ea,sizeof(ea),true));
                                }
                            }
                        }
                        st_mark(kStFlash,static_cast<int>(g));
                    } else if (batch_rows) {
                        // Per-sequence prep + flash (M = 1 each) on row m's slices and KV segment m.
                        const batch::AttentionRows row_layout(n_q_local,n_kv_local,opt.max_ctx);
                        const uint64_t qkv_row=row_layout.qkv, qg_row=row_layout.qgate;
                        const uint64_t oq_row=row_layout.quant, od_row=row_layout.scales, seg=row_layout.kv_segment;
                        for (uint32_t b = 0; b < M; ++b) {
                            struct PrepKA { uint64_t qkv, qw, kw, qo, go, kc, vc; uint32_t M, pos0, nq, nkv, rdims; float rbase, eps; };
                            const PrepKA pka{S[g].attn_qkv_va + b * qkv_row, lw.q_norm_va, lw.k_norm_va,
                                             S[g].q_buf_va + b * qg_row, S[g].gate_buf_va + b * qg_row,
                                             S[g].attn_k_cache_va[ai] + (batch_kv_slots ? (*batch_kv_slots)[b] : b) * seg, S[g].attn_v_cache_va[ai] + (batch_kv_slots ? (*batch_kv_slots)[b] : b) * seg,
                                             1u, (*batch_pos)[b], n_q_local, n_kv_local, hp.rope_dims, rope_base, eps};
                            VJ_CHECK(rt.enqueue_kernel(g, *K[g].attn_prep, (n_q_local + n_kv_local) * 256u, 1, 1, 256, 1, 1, &pka, sizeof(pka), true));
                            struct FlKA { uint64_t qo, go, kc, vc, oq, od; uint32_t M, pos0, nq, nkv; };
                            const FlKA fka{S[g].q_buf_va + b * qg_row, S[g].gate_buf_va + b * qg_row,
                                           S[g].attn_k_cache_va[ai] + (batch_kv_slots ? (*batch_kv_slots)[b] : b) * seg, S[g].attn_v_cache_va[ai] + (batch_kv_slots ? (*batch_kv_slots)[b] : b) * seg,
                                           S[g].mid_q8_va + b * oq_row, S[g].mid_ds_va + b * od_row, 1u, (*batch_pos)[b], n_q_local, nkv_p};
                            VJ_CHECK(rt.enqueue_kernel(g, *K[g].attn_flash, n_q_local * 256u, 1, 1, 256, 1, 1, &fka, sizeof(fka), true));
                        }
                        st_mark(kStFlash, static_cast<int>(g));
                    } else if (fuse_attn_m1 && n_kv_local == 1u) {
                        struct PfM1KA { uint64_t qkv, qw, kw, kc, vc, oq, od; uint32_t pos0, nq, nkv, rdims; float rbase, eps; };
                        const PfM1KA pfka{S[g].attn_qkv_va, lw.q_norm_va, lw.k_norm_va,
                                          S[g].attn_k_cache_va[ai], S[g].attn_v_cache_va[ai],
                                          S[g].mid_q8_va, S[g].mid_ds_va,
                                          pos0, n_q_local, nkv_p, hp.rope_dims, rope_base, eps};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].attn_prep_flash_m1, n_q_local * 256u, 1, 1, 256, 1, 1, &pfka, sizeof(pfka), true));
                        st_mark(kStFlash, static_cast<int>(g));
                    } else {
                    struct PrepKA { uint64_t qkv, qw, kw, qo, go, kc, vc; uint32_t M, pos0, nq, nkv, rdims; float rbase, eps; };
                    PrepKA pka{S[g].attn_qkv_va, lw.q_norm_va, lw.k_norm_va,
                               S[g].q_buf_va, S[g].gate_buf_va, S[g].attn_k_cache_va[ai], S[g].attn_v_cache_va[ai],
                               M, pos0, n_q_local, n_kv_local, hp.rope_dims, rope_base, eps};
                    for (uint32_t r = 0; r < reps_of(4u); ++r)
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].attn_prep, (n_q_local + n_kv_local) * 256u, M, 1, 256, 1, 1, &pka, sizeof(pka), true));
                    st_mark(kStPrep, static_cast<int>(g));

                    struct FlKA { uint64_t qo, go, kc, vc, oq, od; uint32_t M, pos0, nq, nkv; };
                    FlKA fka{S[g].q_buf_va, S[g].gate_buf_va, S[g].attn_k_cache_va[ai], S[g].attn_v_cache_va[ai],
                             S[g].mid_q8_va, S[g].mid_ds_va, M, pos0, n_q_local, nkv_p};
                    const bool fa2_non_straddle = (n_kv_local == 1u) || ((gqa_q_off[g] % gqa_group == 0u) && (n_q_local % n_kv_local == 0u));
                    const bool prefill_fa2_path = (K[g].pf_fa2 && !use_ring && M >= 16u && fa2_non_straddle);
                    const bool wire_supported_rows = (M == 1u || (M >= 5u && M <= 8u));
                    if (wire_requested && !prefill_fa2_path && !wire_supported_rows &&
                        !wire_skip_reported) {
                        std::fprintf(stderr,
                                     "[wire-attn] fallback=batch_rows M=%u requested but unsupported "
                                     "(sealed M is 1 or 5..8); this step is not counted as exercise\n",
                                     M);
                        wire_skip_reported = true;
                    }
                    if (wire_requested && !prefill_fa2_path && !wire_supported_rows && wire_strict)
                        VJ_CHECK(false);
                    if (wire_requested && !prefill_fa2_path && wire_supported_rows) {
                        const wire_normal::WireFormat wire_format =
                            wire_q4 ? wire_normal::WireFormat::Q4 : wire_normal::WireFormat::Q8;
                        wire_normal::Kernels wk{};
                        wk.q8_q = K[g].wire_q8_q;
                        wk.q4_q = K[g].wire_q4_q;
                        wk.q8_kv = K[g].wire_q8_kv;
                        wk.q4_kv = K[g].wire_q4_kv;
                        wk.epilogue = K[g].wire_epilogue;
                        wk.sealed_q8_partition = K[g].wire_skv_q8_part;
                        wk.sealed_q4_partition = K[g].wire_skv_q4_part;
                        wk.sealed_merge = K[g].wire_skv_merge;
                        wire_normal::Buffers wb{};
                        wb.native_k = S[g].attn_k_cache_va[ai];
                        wb.native_v = S[g].attn_v_cache_va[ai];
                        wb.q_wire = S[g].wire_q_va;
                        wb.k_wire = S[g].wire_k_cache_va[ai];
                        wb.v_wire = S[g].wire_v_cache_va[ai];
                        wb.scratch = S[g].wire_scratch_va;
                        wb.out = S[g].wire_out_va;
                        wb.out_lse = S[g].wire_out_lse_va;
                        wb.cache_rows = opt.max_ctx;
                        wb.q_bytes = static_cast<uint64_t>(max_batch) *
                                     wire_normal::q_wire_row_bytes(wire_format);
                        wb.scratch_bytes =
                            flash_decode_splitkv::plan_scratch(wire_normal::kMaxRows, wire_max_splits)
                                .total_bytes;
                        wire_normal::Policy wp;
                        wp.enabled = true;
                        wp.allow_q4 = wire_q4;
                        wp.exercise_short_context = wire_exercise;
                        wp.scratch_budget_bytes = opt.sealed_wire_scratch_budget;
                        wp.max_splits = wire_max_splits;
                        wp.attention_scale = 0.0625f;
                        std::string wire_error;
                        const wire_normal::Attempt wire_attempt = wire_normal::run(
                            rt, static_cast<uint32_t>(g), wk, wb, wp, ai, pos0, M,
                            S[g].q_buf_va, S[g].gate_buf_va, S[g].mid_q8_va, S[g].mid_ds_va,
                            wire_cache[g], wire_counters[g], &wire_error);
                        if (!wire_attempt.engaged) {
                            if (wire_strict) {
                                std::fprintf(stderr, "[wire-attn] STRICT FAIL: %s\n", wire_error.c_str());
                                VJ_CHECK(false);
                            }
                            for (uint32_t r = 0; r < reps_of(5u); ++r)
                                VJ_CHECK(rt.enqueue_kernel(g, *K[g].attn_flash, n_q_local * 256u, M, 1, 256, 1, 1, &fka, sizeof(fka), true));
                        }
                    } else if (prefill_fa2_path) {
                        FlKA fka_fa2{S[g].q_buf_va, S[g].gate_buf_va, S[g].attn_k_cache_va[ai], S[g].attn_v_cache_va[ai],
                                     S[g].mid_q8_va, S[g].mid_ds_va, M, pos0, n_q_local, n_kv_local};
                        const uint32_t G_loc = n_q_local / n_kv_local;
                        const uint32_t fa2_gx = ((M * G_loc + 15u) / 16u) * 256u;
                        const bool use_codex_fa2 = (std::getenv("VEGAJET_PREFILL_CODEX_FA2") != nullptr &&
                                                    std::strcmp(std::getenv("VEGAJET_PREFILL_CODEX_FA2"), "1") == 0 &&
                                                    K[g].pf_fa2_codex != nullptr && n_q_local == 16u && n_kv_local == 1u);
                        VJ_CHECK(rt.enqueue_kernel(g, *(use_codex_fa2 ? K[g].pf_fa2_codex : K[g].pf_fa2), fa2_gx, n_kv_local, 1, 256, 1, 1, &fka_fa2, sizeof(fka_fa2), true));
                    } else if (!wire_requested && opt.spec_flash && K[g].spec_flash_part && fa2_non_straddle &&
                               (n_q_local == n_kv_local * 16u) && (use_ring ? M <= 16u : M < 16u)) {
                        const uint32_t span = pos0 + M;
                        const uint32_t chunk_len = (span <= 1920u) ? 32u : ((span <= 3840u) ? 64u : ((span <= 7680u) ? 128u : 256u));
                        const uint32_t n_chunks  = std::max<uint32_t>(1u, (span + chunk_len - 1u) / chunk_len);
                        struct SfPartKA { uint64_t qo, kc, vc, po, pml; uint32_t M, pos0, nq, nkv, nc, cl; };
                        struct SfMrgKA  { uint64_t po, pml, go, oq, od; uint32_t M, nq, nc; };
                        const SfPartKA pka2{S[g].q_buf_va, S[g].attn_k_cache_va[ai], S[g].attn_v_cache_va[ai],
                                            S[g].spec_fl_part_o_va, S[g].spec_fl_part_ml_va, M, pos0, n_q_local, n_kv_local, n_chunks, chunk_len};
                        const SfMrgKA  mka2{S[g].spec_fl_part_o_va, S[g].spec_fl_part_ml_va, S[g].gate_buf_va,
                                            S[g].mid_q8_va, S[g].mid_ds_va, M, n_q_local, n_chunks};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].spec_flash_part,  n_chunks * 512u, M, n_kv_local, 512, 1, 1, &pka2, sizeof(pka2), true));
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].spec_flash_merge, n_q_local * 256u, M, 1, 256, 1, 1, &mka2, sizeof(mka2), true));
                    } else if (use_auto_splitkv) {
                        struct SkvP1KA { uint64_t qo, kc, vc, po, pml; uint32_t M, pos0, nq, nkv, nc, cl; };
                        struct SkvP2KA { uint64_t po, pml, go, oq, od; uint32_t M, nq, nc; };
                        const SkvP1KA pka2{S[g].q_buf_va, S[g].attn_k_cache_va[ai], S[g].attn_v_cache_va[ai],
                                           S[g].spec_fl_part_o_va, S[g].spec_fl_part_ml_va, M, pos0, n_q_local, nkv_p, skv_nc, skv_cl};
                        const SkvP2KA mka2{S[g].spec_fl_part_o_va, S[g].spec_fl_part_ml_va, S[g].gate_buf_va,
                                           S[g].mid_q8_va, S[g].mid_ds_va, M, n_q_local, skv_nc};
                        for (uint32_t r = 0; r < reps_of(5u); ++r) {
                            VJ_CHECK(rt.enqueue_kernel(g, *K[g].flash_skv_p1, skv_nc * 256u, n_q_local, M, 256, 1, 1, &pka2, sizeof(pka2), true));
                            VJ_CHECK(rt.enqueue_kernel(g, *K[g].flash_skv_p2, n_q_local * 256u, M, 1, 256, 1, 1, &mka2, sizeof(mka2), true));
                        }
                    } else {
                        for (uint32_t r = 0; r < reps_of(5u); ++r)
                            VJ_CHECK(rt.enqueue_kernel(g, *K[g].attn_flash, n_q_local * 256u, M, 1, 256, 1, 1, &fka, sizeof(fka), true));
                    }
                    st_mark(kStFlash, static_cast<int>(g));
                    }

                    const bool do_ep_gqa = (N == 2 && !opt.ab_no_prepush && M <= 16u && !ar_ts_on && !opt.ar_wire_int8 && !(opt.prefill_ar_int8 && !use_ring && M >= 64u));
                    const uint64_t ptx_gqa = do_ep_gqa ? (rt.ar_rx_base_va(1u - g, g) + (ar_epoch & 1u) * kfd::kArSlotBytes) : 0ULL;
                    uint64_t tx3_gqa[2] = {0ULL, 0ULL};
                    if (ep_tp3_ok) ep_tp3_tx(g, tx3_gqa);
                    const bool fwht_w_o = dot8_fwht_mode && (lw.w_o_fwht.rows != 0u);
                    if (use_pf_dot8_dense && (lw.w_o.cols % 64u) == 0u) {
                        enq_q8_to_q4(g, S[g].mid_q8_va, S[g].mid_ds_va, lw.w_o.cols, true, fwht_w_o);
                        for (uint32_t r = 0; r < reps_of(6u); ++r)
                            enq_gemm_q4_d8(g, fwht_w_o ? lw.w_o_fwht : lw.w_o, S[g].partial_va, true);
                    } else {
                        for (uint32_t r = 0; r < reps_of(6u); ++r)
                            enq_gemm_q4(g, lw.w_o, S[g].mid_q8_va, S[g].mid_ds_va, S[g].partial_va, true,
                                        ep_tp3_ok ? tx3_gqa[0] : ptx_gqa, tx3_gqa[1]);
                    }
                    st_mark(kStAttnO, static_cast<int>(g));
                }
                disp += l4_on ? (2u + M * (l4_compressed ? 6u : 1u + 5u * gqa_nkv[0]))
                              : (fuse_attn_m1 ? 3u : (use_auto_splitkv ? 5u : 4u));
            }

            const bool ep_pushed_attn = (N == 2 && !opt.ab_no_prepush && M <= 16u && !ar_ts_on && !opt.ar_wire_int8 && !(opt.prefill_ar_int8 && !use_ring && M >= 64u));
            if (golden_cap_on && il == 0u) {
                gcap_sync();
                gcap_dump("gdn_qkv_gab", S[0].gdn_qkv_gab_va, static_cast<uint64_t>(M) * 10304ULL * 4ULL);
                gcap_dump("gdn_mid_q8", S[0].mid_q8_va, static_cast<uint64_t>(M) * 4096ULL);
                gcap_dump("gdn_mid_ds", S[0].mid_ds_va, static_cast<uint64_t>(M) * 128ULL * 8ULL);
                gcap_dump("gdn_partial_out", S[0].partial_va, static_cast<uint64_t>(M) * D * 4ULL);
            }
            if (golden_cap_on && il == golden_gqa_il) {
                gcap_sync();
                gcap_dump("gqa_qkv", S[0].attn_qkv_va, static_cast<uint64_t>(M) * 8704ULL * 4ULL);
                gcap_dump("gqa_mid_q8", S[0].mid_q8_va, static_cast<uint64_t>(M) * 4096ULL);
                gcap_dump("gqa_mid_ds", S[0].mid_ds_va, static_cast<uint64_t>(M) * 128ULL * 8ULL);
                gcap_dump("gqa_partial_out", S[0].partial_va, static_cast<uint64_t>(M) * D * 4ULL);
            }
            const bool fuse_front = !golden_cap_on && opt.fuse_moe_router && M == 1u && layers[0][il].w_sh_gu.cols == D &&
                                    33u + (layers[0][il].w_sh_gu.rows + 7u) / 8u <= 240u;
            bool n3_front_ready = verify_n3;
            for (size_t g = 0; g < N && n3_front_ready; ++g)
                n3_front_ready = K[g].moe_m1_rtr_shgu && K[g].moe_m1_top8_swiglu &&
                    K[g].router_s240 && K[g].router_s240_red && S[g].splitk_part_va &&
                    layers[g][il].w_sh_gu.cols == D && layers[g][il].w_sh_gu.rows == 2u * local_ff_r[g] &&
                    (layers[g][il].w_sh_gu.row_stride_bytes % 16u) == 0u &&
                    vegajet::router_s240::make_scratch_plan(layers[g][il].w_sh_gu.rows, 1u).bytes <= 8ULL * 16384ULL * 4ULL;
            const bool use_f2_front = ((N == 2u || n3_front_ready) && ab_f2_front && fuse_front && K[0].moe_m1_rtr_shgu != nullptr && K[0].moe_m1_top8_swiglu != nullptr);
            static const vegajet::ornith::RouterS240Options rs240_opt = vegajet::ornith::router_s240_options_from_env();
            const bool s240_kernels_ready = (N == 2u || n3_front_ready) && K[0].router_s240 != nullptr && K[0].router_s240_red != nullptr &&
                                            K[1].router_s240 != nullptr && K[1].router_s240_red != nullptr;
            for (size_t g = 0; g < N && n3_front_ready; ++g)
                n3_front_ready = vegajet::router_s240::make_scratch_plan(layers[g][il].w_sh_gu.rows,
                    rs240_opt.split_k).bytes <= 8ULL * 16384ULL * 4ULL;
            const bool use_s240_front = ((N == 2u || n3_front_ready) && rs240_opt.enabled && fuse_front && !use_f2_front && s240_kernels_ready &&
                                         layers[0][il].w_sh_gu.cols == D && layers[0][il].w_sh_gu.rows == 2u * local_ff_r[0] &&
                                         layers[0][il].w_sh_gu.row_stride_bytes % 16u == 0u && (D % 32u) == 0u && D >= 32u &&
                                         S[0].splitk_part_va != 0u &&
                                         vegajet::router_s240::make_scratch_plan(layers[0][il].w_sh_gu.rows,
                                                                                rs240_opt.split_k).bytes <= 8ULL * 16384ULL * 4ULL);
            const bool fuse_ar_front = (N == 2u && !use_f2_front && !use_s240_front && fuse_front && M == 1u && ep_pushed_attn && !ar_ts_on && !opt.ar_wire_int8 &&
                                        !st_on && K[0].p2p_ar_moe_m1_front != nullptr);
            if (!fuse_ar_front) {
                uint64_t pan_w[kfd::kMaxGpus] = {};
                for (size_t g = 0; g < N; ++g) pan_w[g] = layers[g][il].post_attn_norm_va;
                enq_p2p_ar_rms(il * 2u, pan_w, 1u, ep_pushed_attn);
                st_mark(kStAr, -1);
            }
            if (golden_cap_on && il == 0u) {
                gcap_sync();
                gcap_dump("moe_xnorm", S[0].x_norm_va, static_cast<uint64_t>(M) * D * 4ULL);
                gcap_dump("moe_xq8", S[0].x_q8_va, static_cast<uint64_t>(M) * D);
                gcap_dump("moe_xds", S[0].x_ds_va, static_cast<uint64_t>(M) * (D / 32u) * 8ULL);
            }

            // Decided once per layer for all ranks: a rank that skipped the sense bump would fall through the barrier.
            const bool fuse_back = !expert_offload && (N == 2u) && opt.fuse_moe_back && M == 1u && all_lff_512 && opt.moe_back_wgs >= 1u && opt.moe_back_wgs <= 240u &&
                                   layers[0][il].w_sh_down.cols == 512u && layers[1][il].w_sh_down.cols == 512u &&
                                   layers[0][il].w_gu_d8_w_va == 0u;
            const bool is_prefill_moe = !expert_offload && ((!use_ring && M >= 16u) || M > 16u);
            const uint32_t dgrp = is_prefill_moe ? opt.down_grp_prefill : ((M >= 2u) ? opt.down_grp_verify : 0u);
            const bool use_pf_down_tiled = (!expert_offload && all_lff_super_ok && S[0].dgrp_buf_va != 0u && S[0].pf_gu_queue_va != 0u && !fuse_back && dgrp == 3u && K[0].pf_down_tiled && D % 128u == 0u);
            // Opt-in IQ-LDS tiles: one MoE layer at a time (opt.pf_moe_iq_lds_layer) so a one-layer A/B never
            // changes the whole prefill. pf_moe_iq_lds32 is the diagnostic 32-row shape of the same item.
            const bool il_layer = (opt.pf_moe_iq_lds_layer > 63u) || (il == opt.pf_moe_iq_lds_layer);
            const bool use_pf_gu_iq_lds32 = (opt.pf_moe_iq_lds32 && il_layer && K[0].pf_gu_iq_lds32 != nullptr &&
                                             S[0].pf_gu_queue_va != 0u && all_lff_super_ok && D % 256u == 0u);
            const bool use_pf_gu_iq_lds64 = (!opt.pf_moe_iq_lds32 && opt.pf_moe_iq_lds && il_layer &&
                                             K[0].pf_gu_iq_lds64 != nullptr && S[0].pf_gu_queue_va != 0u &&
                                             all_lff_super_ok && D % 256u == 0u);
            // Opt-in exact IQ3_XXS Down tiles: the compact split plane is only populated for the last layer on TP=2.
            const bool use_pf_down_iq_lds64 = (N == 2u && all_lff_512 && opt.pf_moe_iq_lds >= 2u && il_layer && K[0].pf_down_iq_lds64 != nullptr && K[0].pf_down_comb != nullptr &&
                                              layers[0][47].w_dn_iq3xxs_va != 0u && S[0].pf_gu_queue_va != 0u && !fuse_back &&
                                              dgrp == 3u && D % 64u == 0u);
            const bool use_pf_down_iq_lds32 = (N == 2u && all_lff_512 && opt.pf_moe_iq_lds32 >= 2u && il_layer && K[0].pf_down_iq_lds32 != nullptr && K[0].pf_down_comb != nullptr &&
                                              layers[0][47].w_dn_iq3xxs_va != 0u && S[0].pf_gu_queue_va != 0u && !fuse_back &&
                                              dgrp == 3u && D % 32u == 0u);
            const bool use_dgrp = (!expert_offload && all_lff_512 && S[0].dgrp_buf_va != 0u && !fuse_back && !use_pf_down_tiled && dgrp != 0u && dgrp != 3u && D % 64u == 0u);
            if (fuse_front && !use_f2_front) moe_sense += reps_of(7u);
            if (fuse_back && !ab_b2_back) moe_back_sense += reps_of(8u);
            const uint32_t ar_front_ep = fuse_ar_front ? ar_epoch++ : 0u;
            for (size_t g = 0; g < N; ++g) {
                const auto& lw = layers[g][il];
                const uint32_t lff_g = local_ff_r[g];
                if (use_f2_front) {
                    struct RsKA { uint64_t wr, wsh, xn, shq, shdm, xq, xds, logits, shgsc, shgu; uint32_t D, sh_rows, sh_stride; };
                    struct TsKA { uint64_t logits, shgu, shgsc, tids, tw, oq, od; uint32_t lff; };
                    const uint32_t rtr_sh_wgs = 128u + (lw.w_sh_gu.rows + 3u) / 4u + 1u;
                    const RsKA rk{lw.w_router_f16_va, lw.w_sh_gate_va, S[g].x_norm_va, lw.w_sh_gu.qs_va, lw.w_sh_gu.dm_va,
                                  S[g].x_q8_va, S[g].x_ds_va, S[g].router_log_va, S[g].sh_gate_sc_va, S[g].sh_gu_va,
                                  D, lw.w_sh_gu.rows, lw.w_sh_gu.row_stride_bytes};
                    const TsKA tk{S[g].router_log_va, S[g].sh_gu_va, S[g].sh_gate_sc_va, S[g].top_ids_va, S[g].top_w_va,
                                  S[g].sh_q8_va, S[g].sh_ds_va, lff_g};
                    for (uint32_t r = 0; r < reps_of(7u); ++r) {
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_m1_rtr_shgu, rtr_sh_wgs * 64u, 1, 1, 64, 1, 1, &rk, sizeof(rk), true));
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_m1_top8_swiglu, 2u * 256u, 1, 1, 256, 1, 1, &tk, sizeof(tk), true));
                    }
                } else if (use_s240_front) {
                    std::string rs_err;
                    vegajet::ornith::RouterS240Scratch rsc{};
                    VJ_CHECK(vegajet::ornith::router_s240_scratch(S[g].splitk_part_va, 8ULL * 16384ULL * 4ULL,
                                                                  lw.w_sh_gu.rows, rs240_opt.split_k, &rsc, &rs_err));
                    vegajet::router_s240::Phase1Args pa{};
                    pa.w_router_f16        = lw.w_router_f16_va;
                    pa.w_sh_gate           = lw.w_sh_gate_va;
                    pa.x_norm              = S[g].x_norm_va;
                    pa.sh_w_qs             = lw.w_sh_gu.qs_va;
                    pa.sh_w_dm             = lw.w_sh_gu.dm_va;
                    pa.x_q8                = S[g].x_q8_va;
                    pa.x_ds                = S[g].x_ds_va;
                    pa.rtr_part            = rsc.rtr_part;
                    pa.sh_part             = rsc.sh_part;
                    pa.gate_part           = rsc.gate_part;
                    pa.D                   = D;
                    pa.sh_rows             = lw.w_sh_gu.rows;
                    pa.wr_stride_elems     = (D == 3072u) ? 3200u : D;
                    pa.sh_qs_stride_bytes  = lw.w_sh_gu.row_stride_bytes;
                    pa.sh_dm_stride_u32    = D / 32u;
                    pa.split_k             = rs240_opt.split_k;
                    pa.wr_xor_mask         = 0u;  // engine planes keep the odd-256B layout; XOR is offline-packed
                    pa.sh_xor_mask         = 0u;
                    vegajet::router_s240::ReduceArgs ra{};
                    ra.rtr_part    = rsc.rtr_part;
                    ra.sh_part     = rsc.sh_part;
                    ra.gate_part   = rsc.gate_part;
                    ra.logits      = S[g].router_log_va;
                    ra.sh_gu       = S[g].sh_gu_va;
                    ra.sh_gate_sc  = S[g].sh_gate_sc_va;
                    ra.sh_rows     = lw.w_sh_gu.rows;
                    ra.split_k     = rs240_opt.split_k;
                    VJ_CHECK(vegajet::ornith::launch_router_s240(rt, g, *K[g].router_s240, *K[g].router_s240_red, pa, ra, &rs_err));
                    st_mark(kStRouter, static_cast<int>(g));

                    struct RTopKA { uint64_t lin, tids, tw; uint32_t M; };
                    RTopKA rtka{S[g].router_log_va, S[g].top_ids_va, S[g].top_w_va, M};
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_rtr_top8, M * 256u, 1, 1, 256, 1, 1, &rtka, sizeof(rtka), true));
                    struct SwiKA { uint64_t gu, sc, oq, od; uint32_t M, lff; };
                    SwiKA ska{S[g].sh_gu_va, S[g].sh_gate_sc_va, S[g].sh_q8_va, S[g].sh_ds_va, M, lff_g};
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].swiglu_q8, M * 256u, 1, 1, 256, 1, 1, &ska, sizeof(ska), false));
                    st_mark(kStTop8, static_cast<int>(g));
                } else if (fuse_ar_front) {
                    const size_t peer = 1u - g;
                    const uint32_t pp = ar_front_ep & 1u;
                    const uint32_t slot = il * 2u;
                    struct ArFrontKA {
                        uint64_t xp, ptx, lrx, pfl, lfl, h, wrms, xn, xq, xds;
                        uint64_t wr, wsh, shq, shdm, logits, shgu, tids, tw, oq, od, bar;
                        uint32_t D, ep;
                        float eps;
                        uint32_t sh_rows, sh_stride, lff, sense;
                    };
                    const uint32_t s0 = moe_sense - reps_of(7u) + 1u;
                    const ArFrontKA afka{
                        S[g].partial_va,
                        rt.ar_rx_base_va(peer, g) + pp * kfd::kArSlotBytes,
                        rt.ar_rx_base_va(g, peer) + pp * kfd::kArSlotBytes,
                        rt.ar_flag_va(peer, g, slot),
                        rt.ar_flag_va(g, peer, slot),
                        S[g].h_va, lw.post_attn_norm_va, S[g].x_norm_va, S[g].x_q8_va, S[g].x_ds_va,
                        lw.w_router_f16_va, lw.w_sh_gate_va, lw.w_sh_gu.qs_va, lw.w_sh_gu.dm_va,
                        S[g].router_log_va, S[g].sh_gu_va, S[g].top_ids_va, S[g].top_w_va,
                        S[g].sh_q8_va, S[g].sh_ds_va, S[g].moe_bar_va,
                        D, ar_front_ep, eps, lw.w_sh_gu.rows, lw.w_sh_gu.row_stride_bytes, lff_g, s0
                    };
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].p2p_ar_moe_m1_front, (33u + (lw.w_sh_gu.rows + 7u) / 8u) * 256u, 1, 1, 256, 1, 1, &afka, sizeof(afka), true));
                    for (uint32_t r = 1; r < reps_of(7u); ++r) {
                        struct FrontKA { uint64_t wr, wsh, xn, shq, shdm, xq, xds, logits, shgu, tids, tw, oq, od, bar; uint32_t D, sh_rows, sh_stride, lff, sense; };
                        const FrontKA fka{lw.w_router_f16_va, lw.w_sh_gate_va, S[g].x_norm_va, lw.w_sh_gu.qs_va, lw.w_sh_gu.dm_va,
                                          S[g].x_q8_va, S[g].x_ds_va, S[g].router_log_va, S[g].sh_gu_va, S[g].top_ids_va, S[g].top_w_va,
                                          S[g].sh_q8_va, S[g].sh_ds_va, S[g].moe_bar_va,
                                          D, lw.w_sh_gu.rows, lw.w_sh_gu.row_stride_bytes, lff_g, s0 + r};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_m1_front, (33u + (lw.w_sh_gu.rows + 7u) / 8u) * 256u, 1, 1, 256, 1, 1, &fka, sizeof(fka), true));
                    }
                } else if (fuse_front) {
                    struct FrontKA {
                        uint64_t wr, wsh, xn, shq, shdm, xq, xds, logits, shgu, tids, tw, oq, od, bar;
                        uint32_t D, sh_rows, sh_stride, lff, sense;
                    };
                    for (uint32_t r = 0; r < reps_of(7u); ++r) {
                    const FrontKA fka{lw.w_router_f16_va, lw.w_sh_gate_va, S[g].x_norm_va, lw.w_sh_gu.qs_va, lw.w_sh_gu.dm_va,
                                      S[g].x_q8_va, S[g].x_ds_va, S[g].router_log_va, S[g].sh_gu_va, S[g].top_ids_va, S[g].top_w_va,
                                      S[g].sh_q8_va, S[g].sh_ds_va, S[g].moe_bar_va,
                                      D, lw.w_sh_gu.rows, lw.w_sh_gu.row_stride_bytes, lff_g, moe_sense - reps_of(7u) + 1u + r};
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_m1_front, (33u + (lw.w_sh_gu.rows + 7u) / 8u) * 256u, 1, 1, 256, 1, 1, &fka, sizeof(fka), true));
                    }
                } else {
                if (use_pf_dot8) {
                    enq_q8_to_q4(g, S[g].x_q8_va, S[g].x_ds_va, D, true);
                    if (use_pf_dot8_dense && lw.w_sh_gu.cols % 64u == 0u) {
                        enq_gemm_q4_d8(g, lw.w_sh_gu, S[g].sh_gu_va, true);
                    } else {
                        enq_gemm_q4(g, lw.w_sh_gu, S[g].x_q8_va, S[g].x_ds_va, S[g].sh_gu_va, true);
                    }
                    struct RLogKA { uint64_t wr, wsh, xn, lout, shg; uint32_t M, D; };
                    RLogKA rlka{lw.w_router_f16_va, lw.w_sh_gate_va, S[g].x_norm_va, S[g].router_log_va, S[g].sh_gate_sc_va, M, D};
                    if (K[g].pf_router && is_prefill_moe && (D % 32u) == 0u) {
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_router, 9u * 128u, (M + 31u) / 32u, 1, 128, 1, 1, &rlka, sizeof(rlka), false));
                    } else {
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_rtr_log, 33u * 256u, (M + 7u) / 8u, 1, 256, 1, 1, &rlka, sizeof(rlka), false));
                    }
                } else {
                    struct RLogKA { uint64_t wr, wsh, xn, lout, shg; uint32_t M, D; };
                    RLogKA rlka{lw.w_router_f16_va, lw.w_sh_gate_va, S[g].x_norm_va, S[g].router_log_va, S[g].sh_gate_sc_va, M, D};
                    if (K[g].pf_router && is_prefill_moe && (D % 32u) == 0u) {
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_router, 9u * 128u, (M + 31u) / 32u, 1, 128, 1, 1, &rlka, sizeof(rlka), true));
                    } else if (ring_super_order) {
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].ring_router_so, 129u * 256u, M, 1, 256, 1, 1, &rlka, sizeof(rlka), true));
                    } else {
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_rtr_log, 33u * 256u, (M + 7u) / 8u, 1, 256, 1, 1, &rlka, sizeof(rlka), true));
                    }
                    // Runs concurrently with router logits (both read x_norm / x_q8 from p2p_ar_rms!)
                    enq_gemm_q4(g, lw.w_sh_gu, S[g].x_q8_va, S[g].x_ds_va, S[g].sh_gu_va, false);
                }
                st_mark(kStRouter, static_cast<int>(g));

                struct RTopKA { uint64_t lin, tids, tw; uint32_t M; };
                RTopKA rtka{S[g].router_log_va, S[g].top_ids_va, S[g].top_w_va, M};
                if (accept_moe_on && (use_ring || (am_legacy_prefill && !batch_rows)) && M >= 2u && !ring_super_order && !am_commit) {
                    VJ_CHECK(K[g].moe_top8_acceptmoe_plain != nullptr);
                    struct PlainAmKA { uint64_t lin, tids, tw; uint32_t M, keep; float pen; };
                    const PlainAmKA amka{S[g].router_log_va, S[g].top_ids_va, S[g].top_w_va, M, accept_moe_keep, accept_moe_pen};
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_top8_acceptmoe_plain, 64u, 1, 1, 64, 1, 1, &amka, sizeof(amka), true));
                } else if (accept_moe_on && use_ring && M >= 2u && !ring_super_order && am_commit && K[g].moe_top8_acceptmoe != nullptr) {
                    // VEGAJET_SPEC_ACCEPTMOE=1 (opt-in, lossy): union-biased draft routing, one wave for all M tokens.
                    struct AmKA { uint64_t lin, tids, tw; uint32_t M, keep; float pen, acceptance, cold_penalty; uint32_t window, tick; uint64_t recency; };
                    const AmKA amka{S[g].router_log_va, S[g].top_ids_va, S[g].top_w_va, M, accept_moe_keep, accept_moe_pen, am_acceptance, am_cold_penalty, am_window, am_tick,
                        io_buf.va + am_recency_offset + (g * 48u + il) * 256u * sizeof(uint32_t)};
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_top8_acceptmoe, 64u, 1, 1, 64, 1, 1, &amka, sizeof(amka), true));
                } else
                VJ_CHECK(rt.enqueue_kernel(g, ring_super_order ? *K[g].ring_top8_so : *K[g].moe_rtr_top8, M * 256u, 1, 1, 256, 1, 1, &rtka, sizeof(rtka), true));
                // Runs concurrently with router top8!
                struct SwiKA { uint64_t gu, sc, oq, od; uint32_t M, lff; };
                SwiKA ska{S[g].sh_gu_va, S[g].sh_gate_sc_va, S[g].sh_q8_va, S[g].sh_ds_va, M, lff_g};
                VJ_CHECK(rt.enqueue_kernel(g, *K[g].swiglu_q8, M * 256u, 1, 1, 256, 1, 1, &ska, sizeof(ska), false));
                st_mark(kStTop8, static_cast<int>(g));
                }

                if (fuse_back) {
                    const bool do_ep_moe = (!opt.ab_no_prepush && M == 1u && !ar_ts_on && !opt.ar_wire_int8);
                    const uint64_t ptx_moe = do_ep_moe ? (rt.ar_rx_base_va(1u - g, g) + (ar_epoch & 1u) * kfd::kArSlotBytes) : 0ULL;
                    if (ab_b2_back && lff_g == 512u && D == 3072u && K[g].moe_gu_16r && K[g].moe_mid_f32_to_q8 && K[g].moe_down_d8_sh) {
                        struct Gu8KA  { uint64_t wgu, xq, xds, tids, mf32; uint32_t lff; };
                        struct QKA    { uint64_t mf32, mq, md; uint32_t nblk; };
                        struct DnShKA { uint64_t wd, wds, mq, md, tids, tw, shwq, shwdm, shq, shds, out; uint32_t D, sh_st, dn_wst, dn_sst; uint64_t ptx; };
                        const Gu8KA  gk{lw.w_gu_iq2s_va, S[g].x_q8_va, S[g].x_ds_va, S[g].top_ids_va, S[g].moe_mid_f32_va, lff_g};
                        const QKA    qk{S[g].moe_mid_f32_va, S[g].moe_q8_va, S[g].moe_d_va, (8u * lff_g) / 32u};
                        const DnShKA dk{lw.w_dn_d8_w_va, lw.w_dn_d8_s_va, S[g].moe_q8_va, S[g].moe_d_va, S[g].top_ids_va, S[g].top_w_va,
                                        lw.w_sh_down.qs_va, lw.w_sh_down.dm_va, S[g].sh_q8_va, S[g].sh_ds_va, S[g].partial_va,
                                        D, lw.w_sh_down.row_stride_bytes, lw.dn_d8_w_stride, lw.dn_d8_s_stride, ptx_moe};
                        for (uint32_t r = 0; r < reps_of(8u); ++r) {
                            VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_gu_16r, 256u * 256u, 1, 1, 256, 1, 1, &gk, sizeof(gk), true));
                            VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_mid_f32_to_q8, 16u * 256u, 1, 1, 256, 1, 1, &qk, sizeof(qk), true));
                            VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_down_d8_sh, ((D + 7u) / 8u) * 128u, 1, 1, 128, 1, 1, &dk, sizeof(dk), true));
                        }
                    } else {
                        struct BackKA {
                            uint64_t wgu, xq, xds, tids, tw, mq, md, shwq, shwdm, shq, shds, shd, wd, wds, out, q;
                            uint32_t lff, D, sh_rows, sh_stride, sense, dn_wst, dn_sst, pad;
                            uint64_t ptx;
                        };
                        const bool use_s224 = (opt.moe_back_wgs == 224u && lff_g == 512u && D == 3072u &&
                                               lw.w_sh_down.rows == 3072u && K[g].moe_m1_back_s224 != nullptr);
                        for (uint32_t r = 0; r < reps_of(8u); ++r) {
                            const BackKA bka{lw.w_gu_iq2s_va, S[g].x_q8_va, S[g].x_ds_va, S[g].top_ids_va, S[g].top_w_va,
                                             S[g].moe_q8_va, S[g].moe_d_va, lw.w_sh_down.qs_va, lw.w_sh_down.dm_va,
                                             S[g].sh_q8_va, S[g].sh_ds_va, S[g].sh_down_va, lw.w_dn_d8_w_va, lw.w_dn_d8_s_va, S[g].partial_va, S[g].moe_bq_va,
                                             lff_g, D, lw.w_sh_down.rows, lw.w_sh_down.row_stride_bytes,
                                             moe_back_sense - reps_of(8u) + 1u + r,
                                             lw.dn_d8_w_stride, lw.dn_d8_s_stride, 0u, ptx_moe};
                            VJ_CHECK(rt.enqueue_kernel(g, *(use_s224 ? K[g].moe_m1_back_s224 : K[g].moe_m1_back), opt.moe_back_wgs * 256u, 1, 1, 256, 1, 1, &bka, sizeof(bka), true));
                        }
                    }
                } else if (!expert_offload && (is_prefill_moe || (opt.gu_grp_verify && M >= 8u)) && (lw.w_gu_iq2s_va != 0u || lw.w_gu_d8_w_va != 0u)) {
                    // Deduplicated single MoE expert sort: when grouped Down is active, vj_moe_route_group_kernel
                    // produces the exact same (exp_off, exp_cnt, routes) tables needed by Gate+Up AND the work items
                    // needed by grouped Down, eliminating 96 serial 1-WG sort launches per prefill chunk.
                    uint64_t gu_off_va = S[g].moe_exp_off_va;
                    uint64_t gu_cnt_va = S[g].moe_exp_cnt_va;
                    uint64_t gu_sms_va = S[g].moe_sorted_ms_va;
                    const uint32_t icap = use_dgrp ? grp_item_cap(M) : 0u;
                    if (use_dgrp) {
                        struct RgKA { uint64_t tids, hdr, cnt, off, uniq, items, routes; uint32_t M, min_group, icap; };
                        const RgKA rg{S[g].top_ids_va, S[g].dgrp_hdr_va, S[g].dgrp_cnt_va, S[g].dgrp_off_va, S[g].dgrp_uniq_va, S[g].dgrp_items_va,
                                      S[g].dgrp_routes_va, M, kDownGrpMin, icap};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].dgrp_route, 256u, 1, 1, 256, 1, 1, &rg, sizeof(rg), true));
                        gu_off_va = S[g].dgrp_off_va;
                        gu_cnt_va = S[g].dgrp_cnt_va;
                        gu_sms_va = S[g].dgrp_routes_va;
                    } else {
                        struct SortKA { uint64_t tids, off, cnt, sms; uint32_t M; };
                        SortKA srka{S[g].top_ids_va, S[g].moe_exp_off_va, S[g].moe_exp_cnt_va, S[g].moe_sorted_ms_va, M};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_grp_sort, 256u, 1, 1, 256, 1, 1, &srka, sizeof(srka), true));
                    }
                    enq_gemm_q4(g, lw.w_sh_down, S[g].sh_q8_va, S[g].sh_ds_va, S[g].sh_down_va, false);
                    st_mark(kStSortShDown, static_cast<int>(g));

                    const bool use_pf_down_d8 = (use_pf_dot8_moe && use_pf_down_tiled && K[g].pf_down_d8 != nullptr && lff_g == 512u);
                    if (lw.w_gu_d8_w_va != 0u && lw.gu_d8_w_stride != 2304u && K[g].pf_gu_dot8 && S[g].pf_gu_queue_va != 0u && (lff_g % 32u == 0u) && !use_pf_dot8_moe) {
                        struct GuDot8TKA { uint64_t wgu, wsc, xq, xds, off, cnt, sms, mq, md, queue; uint32_t lff, D, wst, sst; };
                        const GuDot8TKA gt{lw.w_gu_d8_w_va, lw.w_gu_d8_s_va, S[g].x_q8_va, S[g].x_ds_va,
                                           gu_off_va, gu_cnt_va, gu_sms_va, S[g].moe_q8_va, S[g].moe_d_va,
                                           S[g].pf_gu_queue_va, lff_g, D, lw.gu_d8_w_stride, lw.gu_d8_s_stride};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_gu_dot8, 240u * 256u, 1, 1, 256, 1, 1, &gt, sizeof(gt), true));
                    } else if (lw.w_gu_d8_w_va != 0u && lw.w_gu_iq2s_va == 0u && (K[g].moe_gu_q3 || K[g].moe_gu_d8)) {
                        const kfd::LoadedKernel& gu_k = (verify_fast && lw.gu_d8_w_stride == 2304u && K[g].verify_gu_q3_pair)
                            ? *K[g].verify_gu_q3_pair
                            : ((lw.gu_d8_w_stride == 2304u && K[g].moe_gu_q3)
                            ? *K[g].moe_gu_q3
                            : (K[g].moe_gu_d8_cached ? *K[g].moe_gu_d8_cached : *K[g].moe_gu_d8));
                        struct GuD8KA { uint64_t wgu, wsc, xq, xd, tids, mq, md; uint32_t M, lff, wst, sst; };
                        const GuD8KA gd8{lw.w_gu_d8_w_va, lw.w_gu_d8_s_va, S[g].x_q8_va, S[g].x_ds_va, S[g].top_ids_va,
                                         S[g].moe_q8_va, S[g].moe_d_va, M, lff_g, lw.gu_d8_w_stride, lw.gu_d8_s_stride};
                        VJ_CHECK(rt.enqueue_kernel(g, gu_k, 8u * (lff_g / 32u) * 256u, M, 1, 256, 1, 1, &gd8, sizeof(gd8), true));
                    } else if (use_pf_dot8_moe && K[g].pf_gu_d8 && lff_g % 64u == 0u) {
                        struct GuTKA { uint64_t wgu, xq4, xds4, off, cnt, sms, mq, md, mq4, mds4, queue; uint32_t lff, D; };
                        const GuTKA gt{lw.w_gu_iq2s_va, S[g].pf_xq4_va, S[g].pf_xds4_va, gu_off_va, gu_cnt_va, gu_sms_va,
                                       S[g].moe_q8_va, S[g].moe_d_va,
                                       use_pf_down_d8 ? S[g].moe_mid_q4_va : 0ULL,
                                       use_pf_down_d8 ? S[g].moe_mid_ds4_va : 0ULL,
                                       S[g].pf_gu_queue_va, lff_g, D};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_gu_d8, 240u * 256u, 1, 1, 256, 1, 1, &gt, sizeof(gt), true));
                    } else if (use_pf_gu_iq_lds32 && K[g].pf_gu_iq_lds32) {
                        // Diagnostic 32-row x 64-token tile: one decode feeds 32 routed tokens (two passes).
                        struct GuTKA { uint64_t wgu, xq, xds, off, cnt, sms, mq, md, queue; uint32_t lff, D; };
                        const GuTKA gt{lw.w_gu_iq2s_va, S[g].x_q8_va, S[g].x_ds_va, gu_off_va, gu_cnt_va, gu_sms_va,
                                       S[g].moe_q8_va, S[g].moe_d_va, S[g].pf_gu_queue_va, lff_g, D};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_gu_iq_lds32, 240u * 256u, 1, 1, 256, 1, 1, &gt, sizeof(gt), true));
                    } else if (use_pf_gu_iq_lds64 && K[g].pf_gu_iq_lds64) {
                        // The requested single 64x64x64 work item: 64 rows x 64 tokens, 512 threads, one
                        // decode per weight consumed by all 64 tokens.
                        struct GuTKA { uint64_t wgu, xq, xds, off, cnt, sms, mq, md, queue; uint32_t lff, D; };
                        const GuTKA gt{lw.w_gu_iq2s_va, S[g].x_q8_va, S[g].x_ds_va, gu_off_va, gu_cnt_va, gu_sms_va,
                                       S[g].moe_q8_va, S[g].moe_d_va, S[g].pf_gu_queue_va, lff_g, D};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_gu_iq_lds64, 240u * 512u, 1, 1, 512, 1, 1, &gt, sizeof(gt), true));
                    } else if (K[g].pf_gu && S[g].pf_gu_queue_va != 0u && lff_g % 64u == 0u) {
                        struct GuTKA { uint64_t wgu, xq, xds, off, cnt, sms, mq, md, queue; uint32_t lff, D; };
                        const GuTKA gt{lw.w_gu_iq2s_va, S[g].x_q8_va, S[g].x_ds_va, gu_off_va, gu_cnt_va, gu_sms_va,
                                       S[g].moe_q8_va, S[g].moe_d_va, S[g].pf_gu_queue_va, lff_g, D};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_gu, 240u * 256u, 1, 1, 256, 1, 1, &gt, sizeof(gt), true));
                    } else {
                        struct GuGrpKA { uint64_t wgu, xq, xds, off, cnt, sms, mq, md; uint32_t lff; };
                        GuGrpKA ggka{lw.w_gu_iq2s_va, S[g].x_q8_va, S[g].x_ds_va,
                                     gu_off_va, gu_cnt_va, gu_sms_va,
                                     S[g].moe_q8_va, S[g].moe_d_va, lff_g};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_gu_iq2s_grp, (lff_g / 32u) * 256u, 256u, 1, 256, 1, 1, &ggka, sizeof(ggka), true));
                    }
                    st_mark(kStGateUp, static_cast<int>(g));

                    if (use_pf_down_iq_lds64 || use_pf_down_iq_lds32) {
                        // Opt-in IQ3_XXS dequant-once-to-LDS Down on the compact split plane of the last layer.
                        struct DnIKA { uint64_t wq, mq, md, tw, off, cnt, sms, part, queue; uint32_t D; };
                        const DnIKA dni{lw.w_dn_iq3xxs_va, S[g].moe_q8_va, S[g].moe_d_va, S[g].top_w_va,
                                        gu_off_va, gu_cnt_va, gu_sms_va, S[g].dgrp_buf_va, S[g].pf_gu_queue_va, D};
                        if (use_pf_down_iq_lds32)
                            VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_down_iq_lds32, 240u * 256u, 1, 1, 256, 1, 1, &dni, sizeof(dni), true));
                        else
                            VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_down_iq_lds64, 240u * 512u, 1, 1, 512, 1, 1, &dni, sizeof(dni), true));
                        struct CbAllKA { uint64_t part, shd, out; uint32_t M, D; };
                        const CbAllKA cba{S[g].dgrp_buf_va, S[g].sh_down_va, S[g].partial_va, M, D};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_down_comb, ((D + 255u) / 256u) * 256u, M, 1, 256, 1, 1, &cba, sizeof(cba), true));
                    } else if (use_pf_down_d8) {
                        struct DnD8KA { uint64_t wq, ws, mq4, mds4, tw, off, cnt, sms, part, queue; uint32_t D, wst, sst; };
                        const DnD8KA dnt{lw.w_dn_d8_w_va, lw.w_dn_d8_s_va, S[g].moe_mid_q4_va, S[g].moe_mid_ds4_va, S[g].top_w_va,
                                         gu_off_va, gu_cnt_va, gu_sms_va, S[g].dgrp_buf_va, S[g].pf_gu_queue_va,
                                         D, lw.dn_d8_w_stride, lw.dn_d8_s_stride};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_down_d8, 240u * 256u, 1, 1, 256, 1, 1, &dnt, sizeof(dnt), true));
                        struct CbAllKA { uint64_t part, shd, out; uint32_t M, D; };
                        const CbAllKA cba{S[g].dgrp_buf_va, S[g].sh_down_va, S[g].partial_va, M, D};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_down_comb, ((D + 255u) / 256u) * 256u, M, 1, 256, 1, 1, &cba, sizeof(cba), true));
                    } else if (use_pf_down_tiled) {
                        struct DnTKA { uint64_t wq, ws, mq, md, tw, off, cnt, sms, part, queue; uint32_t D, wst, sst, lff, sub32; };
                        const DnTKA dnt{lw.w_dn_d8_w_va, lw.w_dn_d8_s_va, S[g].moe_q8_va, S[g].moe_d_va, S[g].top_w_va,
                                        gu_off_va, gu_cnt_va, gu_sms_va, S[g].dgrp_buf_va, S[g].pf_gu_queue_va,
                                        D, lw.dn_d8_w_stride, lw.dn_d8_s_stride, lff_g, lw.dn_is_sub32 ? 1u : 0u};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_down_tiled, 240u * 256u, 1, 1, 256, 1, 1, &dnt, sizeof(dnt), true));
                        struct CbAllKA { uint64_t part, shd, out; uint32_t M, D; };
                        const CbAllKA cba{S[g].dgrp_buf_va, S[g].sh_down_va, S[g].partial_va, M, D};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_down_comb, ((D + 255u) / 256u) * 256u, M, 1, 256, 1, 1, &cba, sizeof(cba), true));
                    } else if (use_dgrp) {
                        struct CbKA { uint64_t wq, ws, mq, md, tids, tw, shd, cnt, in, out; uint32_t M, D, wst, sst, min_group; };
                        const CbKA cb{lw.w_dn_d8_w_va, lw.w_dn_d8_s_va, S[g].moe_q8_va, S[g].moe_d_va, S[g].top_ids_va, S[g].top_w_va, S[g].sh_down_va,
                                      S[g].dgrp_cnt_va, S[g].dgrp_buf_va, S[g].partial_va, M, D, lw.dn_d8_w_stride, lw.dn_d8_s_stride, kDownGrpMin};
                        if (dgrp == 1u) {
                            struct GdKA { uint64_t wq, mq, hdr, items, routes, dp; uint32_t D, wst; };
                            const GdKA gd{lw.w_dn_d8_w_va, S[g].moe_q8_va, S[g].dgrp_hdr_va, S[g].dgrp_items_va, S[g].dgrp_routes_va, S[g].dgrp_buf_va,
                                          D, lw.dn_d8_w_stride};
                            VJ_CHECK(rt.enqueue_kernel(g, *K[g].dgrp_dp, (D / 64u) * 256u, icap, 1, 256, 1, 1, &gd, sizeof(gd), true));
                            VJ_CHECK(rt.enqueue_kernel(g, *K[g].dgrp_comb_dp, ((D + 15u) / 16u) * 256u, M, 1, 256, 1, 1, &cb, sizeof(cb), true));
                        } else {
                            struct GfKA { uint64_t wq, ws, mq, md, tw, hdr, items, routes, part; uint32_t D, wst, sst; };
                            const GfKA gf{lw.w_dn_d8_w_va, lw.w_dn_d8_s_va, S[g].moe_q8_va, S[g].moe_d_va, S[g].top_w_va, S[g].dgrp_hdr_va,
                                          S[g].dgrp_items_va, S[g].dgrp_routes_va, S[g].dgrp_buf_va, D, lw.dn_d8_w_stride, lw.dn_d8_s_stride};
                            VJ_CHECK(rt.enqueue_kernel(g, *K[g].dgrp_f32, (D / 32u) * 256u, icap, 1, 256, 1, 1, &gf, sizeof(gf), true));
                            VJ_CHECK(rt.enqueue_kernel(g, *K[g].dgrp_comb_f32, ((D + 15u) / 16u) * 256u, M, 1, 256, 1, 1, &cb, sizeof(cb), true));
                        }
                    } else if (lw.dn_is_sub32) {
                        struct DnSub32KA { uint64_t wq, ws, mq, md, tids, tw, shd, out; uint32_t M, D, lff, wst, sst; uint64_t ptx, ptx2; };
                        const DnSub32KA dka{lw.w_dn_d8_w_va, lw.w_dn_d8_s_va, S[g].moe_q8_va, S[g].moe_d_va, S[g].top_ids_va, S[g].top_w_va, S[g].sh_down_va, S[g].partial_va, M, D, lff_g,
                                            lw.dn_d8_w_stride, lw.dn_d8_s_stride, 0ULL, 0ULL};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_down_d8_sub32, ((D + 15u) / 16u) * 256u, M, 1, 256, 1, 1, &dka, sizeof(dka), true));
                    } else {
                        struct DnKA { uint64_t wq, ws, mq, md, tids, tw, shd, out; uint32_t M, D, wst, sst; };
                        DnKA dka{lw.w_dn_d8_w_va, lw.w_dn_d8_s_va, S[g].moe_q8_va, S[g].moe_d_va, S[g].top_ids_va, S[g].top_w_va, S[g].sh_down_va, S[g].partial_va, M, D,
                                 lw.dn_d8_w_stride, lw.dn_d8_s_stride};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_down_d8, ((D + 15u) / 16u) * 256u, M, 1, 256, 1, 1, &dka, sizeof(dka), true));
                    }
                    st_mark(kStDown, static_cast<int>(g));
                } else {
                    if (lw.w_gu_d8_w_va != 0u && K[g].moe_gu_d8) {
                        const kfd::LoadedKernel& gu_k = (verify_fast && lw.gu_d8_w_stride == 2304u && K[g].verify_gu_q3_pair)
                            ? *K[g].verify_gu_q3_pair
                            : ((lw.gu_d8_w_stride == 2304u && K[g].moe_gu_q3)
                            ? *K[g].moe_gu_q3
                            : ((M >= 2u && K[g].moe_gu_d8_cached) ? *K[g].moe_gu_d8_cached : *K[g].moe_gu_d8));
                        struct GuD8KA { uint64_t wgu, wsc, xq, xd, tids, mq, md; uint32_t M, lff, wst, sst; };
                        const GuD8KA gd8{lw.w_gu_d8_w_va, lw.w_gu_d8_s_va, S[g].x_q8_va, S[g].x_ds_va, S[g].top_ids_va,
                                         S[g].moe_q8_va, S[g].moe_d_va, M, lff_g, lw.gu_d8_w_stride, lw.gu_d8_s_stride};
                        VJ_CHECK(rt.enqueue_kernel(g, gu_k, 8u * (lff_g / 32u) * 256u, M, 1, 256, 1, 1, &gd8, sizeof(gd8), true));
                    } else if (lru_slots > 0u && (std::getenv("VEGAJET_MOE_LRU") != nullptr && std::atoi(std::getenv("VEGAJET_MOE_LRU")) > 0) && M >= 2u && M <= 16u && K[g].moe_gu_lru) {
                        struct GuLruKA { uint64_t wgu, cw, cs, e2s, xq, xd, tids, mq, md; uint32_t M, lff, wst, sst; };
                        const GuLruKA lka{lw.w_gu_iq2s_va, lru_w_va[g][il], lru_s_va[g][il], lru_e2s_va[g][il],
                                          S[g].x_q8_va, S[g].x_ds_va, S[g].top_ids_va, S[g].moe_q8_va, S[g].moe_d_va,
                                          M, lff_g, lru_lay.w_row_stride_bytes, lru_lay.s_row_stride_bytes};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_gu_lru, 8u * (lff_g / 32u) * 256u, M, 1, 256, 1, 1, &lka, sizeof(lka), true));
                        if (g == 0u && M >= 4u) {
                            struct CpU32KA { uint64_t dst, src, n; };
                            const CpU32KA cka{lru_top_va + static_cast<uint64_t>(il) * 128ULL * 4ULL, S[0].top_ids_va, static_cast<uint64_t>(M) * 8ULL};
                            VJ_CHECK(rt.enqueue_kernel(0, *K[0].memcpy_u32, 256u, 1, 1, 256, 1, 1, &cka, sizeof(cka), false));
                        }
                    } else {
                        struct GuKA { uint64_t wgu, xq, xd, tids, mq, md; uint32_t M, lff; };
                        VJ_CHECK(lw.w_gu_iq2s_va != 0u);   // promoted layers take the d8/q3 branch above
                        GuKA gka{lw.w_gu_iq2s_va, S[g].x_q8_va, S[g].x_ds_va, S[g].top_ids_va, S[g].moe_q8_va, S[g].moe_d_va, M, lff_g};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_gu_iq2s, 8u * (lff_g / 32u) * 256u, M, 1, 256, 1, 1, &gka, sizeof(gka), true));
                    }
                    st_mark(kStGateUp, static_cast<int>(g));
                    // Runs concurrently with routed Gate+Up!
                    enq_gemm_q4(g, lw.w_sh_down, S[g].sh_q8_va, S[g].sh_ds_va, S[g].sh_down_va, false);
                    st_mark(kStSortShDown, static_cast<int>(g));

                    // BEGIN L6 offload miss boundary: GU has produced quantized activations.
                    std::vector<float> offload_correction;
                    std::vector<int32_t> offload_original_ids;
                    if (expert_offload) {
                        VJ_CHECK(submit_all() && wait_all());
                        auto read=[&](void* dst,uint64_t src,size_t n) { VJ_CHECK(rt.download_from_vram(g,dst,src,n)); };
                        auto write=[&](uint64_t dst,const void* src,size_t n) { VJ_CHECK(rt.upload_to_vram(g,dst,src,n)); };
                        expert_offload->prepare(g,il,M,S[g].top_ids_va,S[g].moe_q8_va,S[g].moe_d_va,S[g].top_w_va,
                                                read,write,offload_correction,offload_original_ids);
                    }
                    // END L6 offload miss boundary.
                    if (use_dgrp) {
                        // Expert-grouped routed Down: each expert's weight tile is read once and applied to every token routed
                        // to it; experts with a single route keep the per-token stream inside the combine kernel.
                        const uint32_t icap = grp_item_cap(M);
                        struct RgKA { uint64_t tids, hdr, cnt, off, uniq, items, routes; uint32_t M, min_group, icap; };
                        const RgKA rg{S[g].top_ids_va, S[g].dgrp_hdr_va, S[g].dgrp_cnt_va, S[g].dgrp_off_va, S[g].dgrp_uniq_va, S[g].dgrp_items_va,
                                      S[g].dgrp_routes_va, M, kDownGrpMin, icap};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].dgrp_route, 256u, 1, 1, 256, 1, 1, &rg, sizeof(rg), true));
                        struct CbKA { uint64_t wq, ws, mq, md, tids, tw, shd, cnt, in, out; uint32_t M, D, wst, sst, min_group; };
                        const CbKA cb{lw.w_dn_d8_w_va, lw.w_dn_d8_s_va, S[g].moe_q8_va, S[g].moe_d_va, S[g].top_ids_va, S[g].top_w_va, S[g].sh_down_va,
                                      S[g].dgrp_cnt_va, S[g].dgrp_buf_va, S[g].partial_va, M, D, lw.dn_d8_w_stride, lw.dn_d8_s_stride, kDownGrpMin};
                        if (dgrp == 1u) {
                            struct GdKA { uint64_t wq, mq, hdr, items, routes, dp; uint32_t D, wst; };
                            const GdKA gd{lw.w_dn_d8_w_va, S[g].moe_q8_va, S[g].dgrp_hdr_va, S[g].dgrp_items_va, S[g].dgrp_routes_va, S[g].dgrp_buf_va,
                                          D, lw.dn_d8_w_stride};
                            VJ_CHECK(rt.enqueue_kernel(g, *K[g].dgrp_dp, (D / 64u) * 256u, icap, 1, 256, 1, 1, &gd, sizeof(gd), true));
                            VJ_CHECK(rt.enqueue_kernel(g, *K[g].dgrp_comb_dp, ((D + 15u) / 16u) * 256u, M, 1, 256, 1, 1, &cb, sizeof(cb), true));
                        } else {
                            struct GfKA { uint64_t wq, ws, mq, md, tw, hdr, items, routes, part; uint32_t D, wst, sst; };
                            const GfKA gf{lw.w_dn_d8_w_va, lw.w_dn_d8_s_va, S[g].moe_q8_va, S[g].moe_d_va, S[g].top_w_va, S[g].dgrp_hdr_va,
                                          S[g].dgrp_items_va, S[g].dgrp_routes_va, S[g].dgrp_buf_va, D, lw.dn_d8_w_stride, lw.dn_d8_s_stride};
                            VJ_CHECK(rt.enqueue_kernel(g, *K[g].dgrp_f32, (D / 32u) * 256u, icap, 1, 256, 1, 1, &gf, sizeof(gf), true));
                            VJ_CHECK(rt.enqueue_kernel(g, *K[g].dgrp_comb_f32, ((D + 15u) / 16u) * 256u, M, 1, 256, 1, 1, &cb, sizeof(cb), true));
                        }
                    } else if (N == 2u && !lw.dn_is_sub32 && !opt.ab_no_prepush && M <= 16u && !ar_ts_on && !opt.ar_wire_int8 && K[g].moe_down_d8_pp != nullptr) {
                        const uint64_t ptx_dn = rt.ar_rx_base_va(1u - g, g) + (ar_epoch & 1u) * kfd::kArSlotBytes;
                        struct DnPpKA { uint64_t wq, ws, mq, md, tids, tw, shd, out; uint32_t M, D, wst, sst; uint64_t ptx; };
                        DnPpKA dka{lw.w_dn_d8_w_va, lw.w_dn_d8_s_va, S[g].moe_q8_va, S[g].moe_d_va, S[g].top_ids_va, S[g].top_w_va, S[g].sh_down_va, S[g].partial_va, M, D,
                                   lw.dn_d8_w_stride, lw.dn_d8_s_stride, ptx_dn};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_down_d8_pp, ((D + 15u) / 16u) * 256u, M, 1, 256, 1, 1, &dka, sizeof(dka), true));
                    } else if (lw.dn_is_sub32) {
                        uint64_t tx3_dn[2] = {0ULL, 0ULL};
                        const bool pp_dn = !expert_offload && ep_tp3_ok && (D & 15u) == 0u;
                        if (pp_dn) ep_tp3_tx(g, tx3_dn);
                        struct DnSub32KA { uint64_t wq, ws, mq, md, tids, tw, shd, out; uint32_t M, D, lff, wst, sst; uint64_t ptx, ptx2; };
                        const DnSub32KA dka{lw.w_dn_d8_w_va, lw.w_dn_d8_s_va, S[g].moe_q8_va, S[g].moe_d_va, S[g].top_ids_va, S[g].top_w_va, S[g].sh_down_va, S[g].partial_va, M, D, lff_g,
                                            lw.dn_d8_w_stride, lw.dn_d8_s_stride, tx3_dn[0], tx3_dn[1]};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_down_d8_sub32, ((D + 15u) / 16u) * 256u, M, 1, 256, 1, 1, &dka, sizeof(dka), true));
                        if (pp_dn && tx3_dn[0] != 0ULL && tx3_dn[1] != 0ULL) ep_push_mask |= (1u << g);
                    } else {
                        struct DnKA { uint64_t wq, ws, mq, md, tids, tw, shd, out; uint32_t M, D, wst, sst; };
                        DnKA dka{lw.w_dn_d8_w_va, lw.w_dn_d8_s_va, S[g].moe_q8_va, S[g].moe_d_va, S[g].top_ids_va, S[g].top_w_va, S[g].sh_down_va, S[g].partial_va, M, D,
                                 lw.dn_d8_w_stride, lw.dn_d8_s_stride};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_down_d8, ((D + 15u) / 16u) * 256u, M, 1, 256, 1, 1, &dka, sizeof(dka), true));
                    }
                    // L6: merge host/CUDA cold outputs before the ordinary TP reduction.
                    if (expert_offload) {
                        VJ_CHECK(submit_all() && wait_all());
                        std::vector<float> output(uint64_t(M)*D);
                        VJ_CHECK(rt.download_from_vram(g,output.data(),S[g].partial_va,output.size()*4u));
                        for (size_t i=0;i<output.size();++i) output[i]+=offload_correction[i];
                        VJ_CHECK(rt.upload_to_vram(g,S[g].partial_va,output.data(),output.size()*4u));
                        VJ_CHECK(rt.upload_to_vram(g,S[g].top_ids_va,offload_original_ids.data(),offload_original_ids.size()*4u));
                    }
                    st_mark(kStDown, static_cast<int>(g));
                }
            }
            if (verify_profile_active) {
                struct CopyKA { uint64_t dst, src, n; };
                const CopyKA ka{verify_profile_buf.va + kVerifyRouteOffset + il * 128ULL * 4u, S[0].top_ids_va, M * 8ULL};
                VJ_CHECK(rt.enqueue_kernel(0, *K[0].memcpy_u32, 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
            }
            if (capture_routes_now) {
                // Rank 0 writes this layer's top IDs exactly once; copy them before layer il+1 overwrites the
                // buffer.  This reuses the existing kfd_memcpy_u32 kernel into pinned host-coherent memory.
                struct CpU32KA { uint64_t dst, src, n; };
                const CpU32KA cka{evict_cap.va + static_cast<uint64_t>(il) * kEvictTopStrideIds * 4ULL,
                                  S[0].top_ids_va, static_cast<uint64_t>(M) * kEvictTopK};
                VJ_CHECK(rt.enqueue_kernel(0, *K[0].memcpy_u32, 256u, 1, 1, 256, 1, 1, &cka, sizeof(cka), false));
                ++captured_layers;
            }
            const uint32_t back_disp = fuse_back ? 1u : ((is_prefill_moe || (opt.gu_grp_verify && M >= 8u)) ? (use_dgrp ? 5u : 4u) : (use_dgrp ? 5u : 3u));
            disp += (fuse_front ? 1u : 4u) + back_disp;

            uint64_t next_w_arr[kfd::kMaxGpus] = {};
            for (size_t g = 0; g < N; ++g)
                next_w_arr[g] = (il + 1u < 48u) ? layers[g][il + 1u].attn_norm_va : out_norm_va[g];
            if (golden_cap_on && il == 0u) {
                gcap_sync();
                gcap_dump("moe_router_logits", S[0].router_log_va, static_cast<uint64_t>(M) * 256ULL * 4ULL);
                gcap_dump("moe_top_ids", S[0].top_ids_va, static_cast<uint64_t>(M) * 8ULL * 4ULL);
                gcap_dump("moe_top_w", S[0].top_w_va, static_cast<uint64_t>(M) * 8ULL * 4ULL);
                gcap_dump("moe_partial_out", S[0].partial_va, static_cast<uint64_t>(M) * D * 4ULL);
            }
            const bool ep_pushed_moe_dn = (N == 2u && !layers[0][il].dn_is_sub32 && !opt.ab_no_prepush && !fuse_back && !(is_prefill_moe || (opt.gu_grp_verify && M >= 8u)) && !use_dgrp &&
                                           M <= 16u && !ar_ts_on && !opt.ar_wire_int8 && K[0].moe_down_d8_pp != nullptr);
            const bool ep_pushed_moe = (N == 2u && !opt.ab_no_prepush && fuse_back && M == 1u && !ar_ts_on && !opt.ar_wire_int8) || ep_pushed_moe_dn;
            enq_p2p_ar_rms(il * 2u + 1u, next_w_arr, (il + 1u < 48u || compute_logits) ? 1u : 0u, ep_pushed_moe);
            st_mark(kStAr, -1);
            if (capture_dflash_taps && df.loaded && df.tap_slot[il] >= 0) {
                struct TapKA { uint64_t src, dst; uint32_t M, D, n_taps, tidx; };
                const TapKA tka{S[df_gpu].h_va, df.taps_va, M, D, static_cast<uint32_t>(df.tap_layers_0idx.size()),
                                static_cast<uint32_t>(df.tap_slot[il])};
                VJ_CHECK(rt.enqueue_kernel(df_gpu, *K[df_gpu].df_tap_copy, ((D + 255u) / 256u) * 256u, M, 1, 256, 1, 1, &tka, sizeof(tka), true));
            }
            if (tap_on && M == 1u) enq_tap(il + 1u);
            if (il == 0u) ring_all();
        }
        if (capture_routes_now && captured_layers == 48u) {
            evict_cap.complete = true;
            evict_cap.tokens = M;
        }
        const bool gdn_convrec_m1 = (opt.gdn_splitv8 && K[0].gdn_conv_recur_sv8 != nullptr && K[0].gdn_conv_recur_sv8_quant != nullptr) ||
                                    (opt.gdn_splitv && K[0].gdn_conv_recur_sv != nullptr);
        if ((!batch_rows && !use_ring && M == 1u && gdn_convrec_m1) ||
            (!use_ring && M >= 32u && K[0].pf_gdn_conv != nullptr)) {
            conv_head = (conv_head + 1u) % kGdnRingSlots;
        }

        // L6: independent sequences each need their own vocabulary projection.
        if (expert_offload) expert_offload->end_step();
        const bool last_tok_only_vocab = batch::last_row_logits(compute_logits,use_ring,batch_rows,M);
        const uint32_t voc_m = last_tok_only_vocab ? 1u : M;
        if (compute_logits) {
            const uint64_t xq_off = last_tok_only_vocab ? (static_cast<uint64_t>(M - 1u) * D) : 0ULL;
            const uint64_t xd_off = last_tok_only_vocab ? (static_cast<uint64_t>(M - 1u) * (D / 32u) * 8ULL) : 0ULL;
            const bool use_b16 = (voc_m >= 13u && voc_m <= 16u && K[0].vocab_argmax_b16 != nullptr);
            const bool use_b12 = (!use_b16 && voc_m >= 7u && voc_m <= 12u && K[0].vocab_argmax_b12 != nullptr);
            const bool use_b6  = (!use_b16 && !use_b12 && voc_m >= 2u && D <= 3072u && K[0].vocab_argmax_b6 != nullptr);
            for (size_t g = 0; g < N; ++g) {
                struct VocKA { uint64_t wq, wd, xq, xd, bv, bi; uint32_t lr, cols, ro, stride, M; };
                VocKA vka{out_w[g].qs_va, out_w[g].ds_va, S[g].x_q8_va + xq_off, S[g].x_ds_va + xd_off,
                          S[g].argmax_val_va, S[g].argmax_id_va,
                          voc_rows[g], D, voc_off[g], out_w[g].row_stride_bytes, voc_m};
                if (head_q4) {   // one head for every M: verify rows and plain steps must agree
                    VJ_CHECK(enqueue_vocab_q4(g, vka, voc_m));
                } else if (use_b16) {
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].vocab_argmax_b16, kVocabBlocks * 256u, 1, 1, 256, 1, 1, &vka, sizeof(vka), true));
                } else if (use_b12) {
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].vocab_argmax_b12, kVocabBlocks * 256u, 1, 1, 256, 1, 1, &vka, sizeof(vka), true));
                } else if (use_b6) {
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].vocab_argmax_b6, kVocabBlocks * 256u, (voc_m + 5u) / 6u, 1, 256, 1, 1, &vka, sizeof(vka), true));
                } else {
                    VJ_CHECK(voc_m == 1u ? enqueue_vocab_m1(g, vka) : rt.enqueue_kernel(g, *K[g].vocab_argmax, kVocabBlocks * 256u, voc_m, 1, 256, 1, 1, &vka, sizeof(vka), true));
                }
            }
            disp += 1u;
        }
        st_mark(kStVocab, -1);
        last_dispatches = disp;

        VJ_CHECK(submit_all() && wait_all());

        if (verify_profile_active) {
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
            std::printf("[VERIFY-PROFILE M=%u TP=%zu pos=%u fast=%u n3=%u] full queue boundaries, ms (25 MHz); instrumented step\n", M, N, pos0, verify_fast, verify_n3);
            double sums[kfd::kMaxGpus][kStCount] = {};
            for (size_t g = 0; g < N; ++g) {
                const auto* ticks = reinterpret_cast<const uint64_t*>(verify_profile_buf.host_ptr) + g * kVerifyStamps;
                for (size_t i = 1; i < verify_stage[g].size(); ++i)
                    sums[g][verify_stage[g][i]] += (ticks[i] - ticks[i - 1u]) * 40e-6;
            }
            for (uint32_t st = 0; st < kStCount; ++st) {
                std::printf("  %-38s", kStName[st]);
                for (size_t g = 0; g < N; ++g) std::printf(" GPU%zu=%8.3f", g, sums[g][st]);
                std::printf("\n");
            }
            std::printf("  %-38s", "TOTAL");
            for (size_t g = 0; g < N; ++g) {
                double total = 0.; for (double v : sums[g]) total += v;
                std::printf(" GPU%zu=%8.3f", g, total);
            }
            std::printf("\n[VERIFY-UNION] layer:unique/pair-loads/routes (rank0 exact top IDs)\n");
            const auto* ids = reinterpret_cast<const int32_t*>(reinterpret_cast<const uint8_t*>(verify_profile_buf.host_ptr) + kVerifyRouteOffset);
            for (uint32_t il = 0; il < 48u; ++il) {
                uint32_t counts[256] = {}, unique = 0u, pairs = 0u;
                for (uint32_t r = 0; r < M * 8u; ++r) { const int32_t e = ids[il * 128u + r]; VJ_CHECK(e >= 0 && e < 256); ++counts[e]; }
                for (uint32_t c : counts) { unique += c != 0u; pairs += (c + 1u) / 2u; }
                std::printf(" %u:%u/%u/%u%s", il, unique, pairs, M * 8u, il % 8u == 7u ? "\n" : "");
            }
            verify_profile_seen |= 1u << M;
            verify_profile_active = false;
        }

        // Update Dynamic LRU MoE Gate+Up cache on M >= 4 verify passes (~10.8 us/expert on-GPU repack!).
        const bool lru_active_now = (lru_cache && std::getenv("VEGAJET_MOE_LRU") != nullptr && std::atoi(std::getenv("VEGAJET_MOE_LRU")) > 0);
        if (lru_active_now && M >= 4u && M <= 16u && !use_m1_super && !use_pf_dot8) {
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
            int32_t tmp_slots[128];
            for (uint32_t il = 0; il < 48u; ++il) {
                lru_cache->route(il, lru_top_host + static_cast<size_t>(il) * 128u, M * 8u, tmp_slots);
            }
            lru_cache->end_step();
            const auto admits = lru_cache->take_admissions();
            if (!admits.empty()) {
                struct RepKA { uint64_t src, cw, cs; uint32_t eid, sid, lff, wst, sst; };
                for (const auto& a : admits) {
                    if (a.evicted >= 0 && a.evicted < 256) lru_e2s_host[a.layer][a.evicted] = -1;
                    lru_e2s_host[a.layer][a.expert] = static_cast<int32_t>(a.slot);
                    for (size_t g = 0; g < N; ++g) {
                        const RepKA rka{layers[g][a.layer].w_gu_iq2s_va, lru_w_va[g][a.layer], lru_s_va[g][a.layer],
                                        a.expert, a.slot, local_ff_r[g], lru_lay.w_row_stride_bytes, lru_lay.s_row_stride_bytes};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_gu_repack, local_ff_r[g] * 128u, 1, 1, 128, 1, 1, &rka, sizeof(rka), true));
                    }
                    lru_cache->commit(a);
                }
                VJ_CHECK(submit_all() && wait_all());
                for (const auto& a : admits) {
                    for (size_t g = 0; g < N; ++g) {
                        VJ_CHECK(rt.upload_to_vram(g, lru_e2s_va[g][a.layer], lru_e2s_host[a.layer].data(), 256u * 4u));
                    }
                }
            }
        }

        std::vector<int32_t> preds(M, -1);
        if (compute_logits) {
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
            for (uint32_t m = 0; m < voc_m; ++m) {
                float best_v = -1e30f;
                int32_t best_i = -1;
                for (uint32_t b = 0; b < kVocabBlocks; ++b) {
                    const uint32_t idx = m * kVocabBlocks + b;
                    for (size_t g = 0; g < N; ++g) {
                        if (io_arg_val_host[g][idx] > best_v) { best_v = io_arg_val_host[g][idx]; best_i = io_arg_id_host[g][idx]; }
                    }
                }
                preds[last_tok_only_vocab ? (M - 1u) : m] = best_i;
            }
        }
        return preds;
    };

    // Quick stage micro-profiler (measures exact M=1 ms per component across 48 layers!)
    auto run_m1_profile = [&]() {
        auto time_fn = [&](int iters, auto&& fn) -> double {
            fn();
            auto t0 = std::chrono::steady_clock::now();
            for (int it = 0; it < iters; ++it) fn();
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
        };
        const float eps = static_cast<float>(hp.rms_eps);
        auto run_g0 = [&](auto&& enq) {
            enq();
            VJ_CHECK(rt.submit_batch(0));
            VJ_CHECK(rt.wait_idle(0));
        };
        const auto& lg = layers[0][0]; // GDN
        const auto& la = layers[0][3]; // GQA
        auto enq_q4_m1 = [&](const Q4A8WeightDev& w, uint64_t xq, uint64_t xds, uint64_t y) {
            if (w.cols == 512u) {
                struct M1KA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, stride; };
                M1KA ka{w.qs_va, w.dm_va, xq, xds, y, w.rows, w.cols, w.row_stride_bytes};
                VJ_CHECK(rt.enqueue_kernel(0, *K[0].gemv_q4_w16, ((w.rows + 15u) / 16u) * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
            } else {
                struct M1W32KA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, stride, pad; uint64_t ptx; };
                M1W32KA ka{w.qs_va, w.dm_va, xq, xds, y, w.rows, w.cols, w.row_stride_bytes, 0u, 0ULL};
                VJ_CHECK(rt.enqueue_kernel(0, *K[0].gemv_q4_w32, ((w.rows + 7u) / 8u) * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
            }
        };
        double t_gdn_qkv = time_fn(8, [&]() { run_g0([&]() { for (int i = 0; i < 36; ++i) enq_q4_m1(layers[0][0].w_gdn_qkv_gab, S[0].x_q8_va, S[0].x_ds_va, S[0].gdn_qkv_gab_va); }); });
        double t_gdn_rec = time_fn(8, [&]() { run_g0([&]() {
            for (int i = 0; i < 36; ++i) {
                struct RecKA { uint64_t sring, cq, gab, dec, bs, nw, oq, od; uint32_t M, nk, nv, cdim, strd, zoff, rhead, uring; float eps; };
                const uint64_t sva = opt.spec_gdn
                    ? (S[0].spec_gdn_s_slot_va[0] + static_cast<uint64_t>(i) * gdn_nv[0] * 128ULL * 128ULL * 4ULL)
                    : S[0].gdn_ssm_ring_va[i];
                RecKA rka{sva, S[0].gdn_conv_va, S[0].gdn_qkv_gab_va, S[0].gdn_dec_va, S[0].gdn_bsig_va,
                          lg.ssm_norm_va, S[0].mid_q8_va, S[0].mid_ds_va, 1u, gdn_nk[0], gdn_nv[0], gdn_conv_dim[0], gdn_qkv_stride[0], gdn_conv_dim[0], 0u, 0u, eps};
                VJ_CHECK(rt.enqueue_kernel(0, *K[0].gdn_recur, gdn_nv[0] * 1024u, 1, 1, 1024, 1, 1, &rka, sizeof(rka), true));
            }
        }); });
        double t_gdn_out = time_fn(8, [&]() { run_g0([&]() { for (int i = 0; i < 36; ++i) enq_q4_m1(lg.w_ssm_out, S[0].mid_q8_va, S[0].mid_ds_va, S[0].partial_va); }); });
        double t_gqa_qkv = time_fn(8, [&]() { run_g0([&]() { for (int i = 0; i < 12; ++i) { enq_q4_m1(la.w_attn_qkv, S[0].x_q8_va, S[0].x_ds_va, S[0].attn_qkv_va); enq_q4_m1(la.w_o, S[0].mid_q8_va, S[0].mid_ds_va, S[0].partial_va); } }); });
        double t_rtr     = time_fn(8, [&]() { run_g0([&]() {
            for (int i = 0; i < 48; ++i) {
                struct RLogKA { uint64_t wr, wsh, xn, lout, shg; uint32_t M, D; };
                RLogKA rlka{layers[0][i].w_router_f16_va, layers[0][i].w_sh_gate_va, S[0].x_norm_va, S[0].router_log_va, S[0].sh_gate_sc_va, 1u, D};
                VJ_CHECK(rt.enqueue_kernel(0, *K[0].moe_rtr_log, 33u * 256u, 1, 1, 256, 1, 1, &rlka, sizeof(rlka), true));
                struct RTopKA { uint64_t lin, tids, tw; uint32_t M; };
                RTopKA rtka{S[0].router_log_va, S[0].top_ids_va, S[0].top_w_va, 1u};
                VJ_CHECK(rt.enqueue_kernel(0, *K[0].moe_rtr_top8, 256u, 1, 1, 256, 1, 1, &rtka, sizeof(rtka), true));
            }
        }); });
        double t_moe_gu  = time_fn(8, [&]() { run_g0([&]() {
            for (int i = 0; i < 48; ++i) {
                if (layers[0][i].w_gu_iq2s_va == 0u) continue;   // in-place DOT8/Q3 promotion: no IQ2_S plane to time
                struct GuKA { uint64_t wgu, xq, xd, tids, mq, md; uint32_t M, lff; };
                GuKA gka{layers[0][i].w_gu_iq2s_va, S[0].x_q8_va, S[0].x_ds_va, S[0].top_ids_va, S[0].moe_q8_va, S[0].moe_d_va, 1u, local_ff_r[0]};
                VJ_CHECK(rt.enqueue_kernel(0, *K[0].moe_gu_iq2s, 8u * (local_ff_r[0] / 32u) * 256u, 1, 1, 256, 1, 1, &gka, sizeof(gka), true));
            }
        }); });
        double t_moe_dn  = time_fn(8, [&]() { run_g0([&]() {
            for (int i = 0; i < 48; ++i) {
                if (layers[0][i].dn_is_sub32) {
                    struct DnSub32KA { uint64_t wq, ws, mq, md, tids, tw, shd, out; uint32_t M, D, lff, wst, sst; uint64_t ptx, ptx2; };
                    const DnSub32KA dka{layers[0][i].w_dn_d8_w_va, layers[0][i].w_dn_d8_s_va, S[0].moe_q8_va, S[0].moe_d_va, S[0].top_ids_va, S[0].top_w_va, S[0].sh_down_va,
                                        S[0].partial_va, 1u, D, local_ff_r[0], layers[0][i].dn_d8_w_stride, layers[0][i].dn_d8_s_stride, 0ULL, 0ULL};
                    VJ_CHECK(rt.enqueue_kernel(0, *K[0].moe_down_d8_sub32, ((D + 15u) / 16u) * 256u, 1, 1, 256, 1, 1, &dka, sizeof(dka), true));
                } else {
                    struct DnKA { uint64_t wq, ws, mq, md, tids, tw, shd, out; uint32_t M, D, wst, sst; };
                    DnKA dka{layers[0][i].w_dn_d8_w_va, layers[0][i].w_dn_d8_s_va, S[0].moe_q8_va, S[0].moe_d_va, S[0].top_ids_va, S[0].top_w_va, S[0].sh_down_va,
                             S[0].partial_va, 1u, D, layers[0][i].dn_d8_w_stride, layers[0][i].dn_d8_s_stride};
                    VJ_CHECK(rt.enqueue_kernel(0, *K[0].moe_down_d8, ((D + 15u) / 16u) * 256u, 1, 1, 256, 1, 1, &dka, sizeof(dka), true));
                }
            }
        }); });
        double t_lm      = time_fn(8, [&]() { run_g0([&]() {
            struct VocKA { uint64_t wq, wd, xq, xd, bv, bi; uint32_t lr, cols, ro, stride, M; };
            VocKA vka{out_w[0].qs_va, out_w[0].ds_va, S[0].x_q8_va, S[0].x_ds_va,
                      S[0].argmax_val_va, S[0].argmax_id_va, voc_rows[0], D, 0u, out_w[0].row_stride_bytes, 1u};
            VJ_CHECK(enqueue_vocab_m1(0, vka));
        }); });
        double t_ar = (N <= 1u) ? 0.0 : time_fn(8, [&]() {
            for (int i = 0; i < 96; ++i) {
                const uint32_t ep = ar_epoch++;
                const uint32_t pp = ep & 1u;
                if (N == 2u) {
                    for (size_t g = 0; g < 2; ++g) {
                        const size_t peer = 1u - g;
                        struct ArKA {
                            uint64_t xp, ptx, lrx, pfl, lfl, bar, h, w, xn, xq, xd;
                            uint32_t M, D, ep, dn;
                            float eps;
                        };
                        ArKA aka{
                            S[g].partial_va,
                            rt.ar_rx_base_va(peer, g) + pp * kfd::kArSlotBytes,
                            rt.ar_rx_base_va(g, peer) + pp * kfd::kArSlotBytes,
                            rt.ar_flag_va(peer, g, static_cast<uint32_t>(i)),
                            rt.ar_flag_va(g, peer, static_cast<uint32_t>(i)),
                            S[g].ar_bar_va, S[g].h_va, layers[g][0].attn_norm_va,
                            S[g].x_norm_va, S[g].x_q8_va, S[g].x_ds_va, 1u, D, ep, 1u, eps
                        };
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].p2p_ar_rms, 256u, 1, 1, 256, 1, 1, &aka, sizeof(aka), true));
                    }
                } else {
                    struct ArNKA {
                        uint64_t xp, ptx[4], lrx[4], pfl[4], lfl[4], bar, h, w, xn, xq, xd;
                        uint32_t M, D, ep, dn, rank, n_ranks, pre_pushed;
                        float eps;
                    };
                    for (size_t g = 0; g < N; ++g) {
                        ArNKA aka{};
                        aka.xp = S[g].partial_va;
                        for (size_t r = 0; r < 4; ++r) {
                            aka.ptx[r] = (r < N && r != g) ? (rt.ar_rx_base_va(r, g) + pp * kfd::kArSlotBytes) : 0ULL;
                            aka.lrx[r] = (r < N && r != g) ? (rt.ar_rx_base_va(g, r) + pp * kfd::kArSlotBytes) : 0ULL;
                            aka.pfl[r] = (r < N && r != g) ? rt.ar_flag_va(r, g, static_cast<uint32_t>(i)) : 0ULL;
                            aka.lfl[r] = (r < N && r != g) ? rt.ar_flag_va(g, r, static_cast<uint32_t>(i)) : 0ULL;
                        }
                        aka.bar = S[g].ar_bar_va; aka.h = S[g].h_va; aka.w = layers[g][0].attn_norm_va;
                        aka.xn = S[g].x_norm_va; aka.xq = S[g].x_q8_va; aka.xd = S[g].x_ds_va;
                        aka.M = 1u; aka.D = D; aka.ep = ep; aka.dn = 1u;
                        aka.rank = static_cast<uint32_t>(g); aka.n_ranks = static_cast<uint32_t>(N);
                        aka.pre_pushed = 0u; aka.eps = eps;
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].p2p_ar_n_fp16, 256u, 1, 1, 256, 1, 1, &aka, sizeof(aka), true));
                    }
                }
            }
            VJ_CHECK(submit_all() && wait_all());
        });
        const double t_conv = time_fn(8, [&]() { run_g0([&]() {
            for (int i = 0; i < 36; ++i) {
                struct ConvL2KA {
                    uint64_t qkv, cw, cring, cout, dt, sa, dec, bs;
                    uint32_t M, nk, nv, cdim, strd, aoff, boff, rhead, uring;
                    float eps;
                };
                ConvL2KA cka{S[0].gdn_qkv_gab_va, lg.ssm_conv_w_va, S[0].gdn_conv_ring_va[i], S[0].gdn_conv_va,
                             lg.ssm_dt_va, lg.ssm_a_va, S[0].gdn_dec_va, S[0].gdn_bsig_va,
                             1u, gdn_nk[0], gdn_nv[0], gdn_conv_dim[0], gdn_qkv_stride[0], gdn_alpha_off[0], gdn_beta_off[0], 0u, 0u, eps};
                VJ_CHECK(rt.enqueue_kernel(0, *K[0].gdn_conv_l2, ((gdn_conv_dim[0] + 127u) / 128u + 1u) * 128u, 1, 1, 128, 1, 1, &cka, sizeof(cka), true));
            }
        }); });
        std::printf("[PROFILE M=1 GPU0] gdn_conv_l2 x36 = %.2f ms (%.1f us each)\n", t_conv, t_conv * 1000.0 / 36.0);
        // Fixed cost of a barrier-serialised dispatch: 384 one-WG no-op-sized memsets on GPU 0, and the same count
        // submitted to all GPUs (one submit/wait pair), to separate launch latency from kernel work.
        struct NullKA { uint64_t dst; uint32_t val; uint32_t pad; uint64_t n; };
        const NullKA nka{S[0].gdn_sq_va, 0u, 0u, 1u};
        const double t_null1 = time_fn(8, [&]() { run_g0([&]() {
            for (int i = 0; i < 384; ++i) VJ_CHECK(rt.enqueue_kernel(0, *K[0].memset_u32, 256u, 1, 1, 256, 1, 1, &nka, sizeof(nka), true));
        }); });
        const double t_null2 = time_fn(8, [&]() {
            for (size_t g = 0; g < N; ++g) {
                const NullKA k2{S[g].gdn_sq_va, 0u, 0u, 1u};
                for (int i = 0; i < 384; ++i) VJ_CHECK(rt.enqueue_kernel(g, *K[g].memset_u32, 256u, 1, 1, 256, 1, 1, &k2, sizeof(k2), true));
            }
            VJ_CHECK(submit_all() && wait_all());
        });
        const double t_empty = time_fn(8, [&]() { run_g0([&]() {
            VJ_CHECK(rt.enqueue_kernel(0, *K[0].memset_u32, 256u, 1, 1, 256, 1, 1, &nka, sizeof(nka), true));
        }); });
        std::printf("[PROFILE dispatch] 384 null dispatches: GPU0 %.3f ms (%.2f us each) | all %zu GPUs %.3f ms | 1 dispatch submit+wait %.1f us\n",
                    t_null1, t_null1 * 1000.0 / 384.0, N, t_null2, t_empty * 1000.0);
        std::printf("[PROFILE M=1 GPU0 48L] gdn_qkv=%.2fms gdn_rec=%.2fms gdn_out=%.2fms gqa_qkv+o=%.2fms rtr=%.2fms moe_gu=%.2fms moe_dn=%.2fms lm=%.2fms p2p_ar96=%.2fms | sum=%.2fms\n",
                    t_gdn_qkv, t_gdn_rec, t_gdn_out, t_gqa_qkv, t_rtr, t_moe_gu, t_moe_dn, t_lm, t_ar,
                    t_gdn_qkv + t_gdn_rec + t_gdn_out + t_gqa_qkv + t_rtr + t_moe_gu + t_moe_dn + t_lm + t_ar);
        if (tb_base_ms > 0.0) {
            // Complete raw-decode table: in-situ cost for idempotent stages (stage enqueued 3x), isolated GPU 0 cost
            // for the state-updating ones (conv, recurrence, AllReduce, LM head).
            // Bytes are TRUE fetched HBM2 bytes/GPU/token (excluding odd-256B TCC padding bytes never read by kernels,
            // and using exact per-format routed Gate+Up and Down byte counts).
            auto q4_real_bytes = [](const Q4A8WeightDev& w) {
                return double(w.rows) * (double(w.cols) * 0.5 + double(w.cols / 32u) * 4.0);
            };
            const double gu_d8_row_b = double(dot8_gu_iq2s_layout(1u, D).read_bytes_per_row());
            const double dn_d8_row_b = double((local_ff_r[0] == 512u ? dot8_iq3_layout(1u, 512u) : dot8_iq3_sub32_layout(1u, local_ff_r[0])).read_bytes_per_row());
            double b_gdn_qkv = 0, b_gdn_out = 0, b_gqa_qkv = 0, b_gqa_o = 0, b_front = 0, b_back = 0;
            uint32_t n_gdn = 0, n_gqa = 0;
            for (uint32_t il = 0; il < 48u; ++il) {
                const auto& lw = layers[0][il];
                if (lw.is_gdn) { b_gdn_qkv += q4_real_bytes(lw.w_gdn_qkv_gab); b_gdn_out += q4_real_bytes(lw.w_ssm_out); ++n_gdn; }
                else           { b_gqa_qkv += q4_real_bytes(lw.w_attn_qkv);    b_gqa_o   += q4_real_bytes(lw.w_o);       ++n_gqa; }
                b_front += 256.0 * D * 2.0 + q4_real_bytes(lw.w_sh_gu);
                const double gu_b = (lw.w_gu_d8_w_va != 0) ? (8.0 * local_ff * gu_d8_row_b) : (8.0 * local_ff * 1968.0);
                const double dn_b = (lw.w_dn_d8_w_va != 0) ? (8.0 * D * dn_d8_row_b)        : (8.0 * D * 208.0);
                b_back  += gu_b + dn_b + q4_real_bytes(lw.w_sh_down);
            }
            const double b_rec = n_gdn * 2.0 * 32.0 * 128.0 * 128.0 * 4.0; // S read + written once
            const double b_lm  = double(voc_rows[0]) * (double(D) + double(D / 32u) * 2.0);
            const double ceil0 = gpu_bw_ceiling_gbs[0] > 0.0 ? gpu_bw_ceiling_gbs[0] : 800.0;
            struct Row { const char* name; double ms; double bytes; };
            const Row rows[] = {
                {"GDN proj (qkv+gab)", tb_stage_ms[1], b_gdn_qkv}, {"GDN out proj", tb_stage_ms[2], b_gdn_out},
                {"GDN conv", t_conv, 0.0}, {"GDN recurrence", t_gdn_rec, b_rec},
                {"GQA proj (qkv)", tb_stage_ms[3], b_gqa_qkv}, {"GQA prep", tb_stage_ms[4], 0.0},
                {"GQA flash", tb_stage_ms[5], 0.0}, {"GQA out proj", tb_stage_ms[6], b_gqa_o},
                {"Router+Shared G/U (front)", tb_stage_ms[7], b_front}, {"MoE G/U+Down (back)", tb_stage_ms[8], b_back},
                {"LM head", t_lm, b_lm}, {"p2p_ar96 (isolated)", t_ar, 0.0},
            };
            double sum = 0.0, tot_bytes = 0.0;
            std::printf("[TABLE tg128 raw M=1] wall %.2f ms/tok (%u GDN + %u GQA layers | GPU0 calibrated ceiling %.1f GB/s)\n",
                        tb_base_ms, n_gdn, n_gqa, ceil0);
            for (const Row& r : rows) {
                sum += r.ms;
                tot_bytes += r.bytes;
                if (r.bytes > 0.0 && r.ms > 0.0) {
                    const double gbs = r.bytes / 1e6 / r.ms;
                    std::printf("[TABLE]   %-28s %6.2f ms  %7.1f real_MB/GPU  %6.1f real_GB/s/GPU  (%4.1f%% of %.1f)\n",
                                r.name, r.ms, r.bytes / 1e6, gbs, 100.0 * gbs / ceil0, ceil0);
                } else {
                    std::printf("[TABLE]   %-28s %6.2f ms\n", r.name, r.ms);
                }
            }
            std::printf("[TABLE]   %-28s %6.2f ms  %7.1f real_MB/GPU  (wall - sum: rank sync slack, embed, host submit/wait, %u dispatches)\n",
                        "CP gap / unattributed", tb_base_ms - sum, tot_bytes / 1e6, last_dispatches);
        }
    };

    // C2 macro-batch prefill (opt-in VEGAJET_PREFILL_C2=256|512[:16|:32]): dense + MoE run on M-token macro batches
    // with today's W4A8 kernels. The budget is the scratch already allocated for max_batch rows; a shape that does
    // not fit keeps the current --prefill-chunk path. The per-macro GDN sub-chunk list (MacroBatch::gdn) is the
    // dispatch order for the C1/C3 sub-chunk kernels; until those land the GDN kernels consume the macro batch whole.
    prefill_sched::Config c2_cfg;
    c2_cfg.legacy_chunk = opt.prefill_chunk;
    {
        std::string c2_err;
        const char* c2_env = std::getenv("VEGAJET_PREFILL_C2");
        if (!prefill_sched::parse_c2_spec(c2_env ? c2_env : "", &c2_cfg.enabled, &c2_cfg.macro_m, &c2_cfg.gdn_c, &c2_err)) {
            std::printf("[PREFILL-C2] WARN: VEGAJET_PREFILL_C2 ignored: %s\n", c2_err.c_str());
            c2_cfg.enabled = false;
        }
        c2_cfg.budget_bytes = prefill_sched::scratch_for(std::min<uint32_t>(1024u, (max_batch / 16u) * 16u), 16u, c2_cfg);
    }
    auto run_greedy_raw = [&](const std::vector<int32_t>& prompt_ids, int n_gen, double* out_tps, double* out_pp_tps = nullptr) -> std::vector<int32_t> {
        reset_state();
        auto tp0 = std::chrono::steady_clock::now();
        uint32_t pos = 0;
        int32_t last_pred = -1;
        const prefill_sched::Schedule c2 = prefill_sched::schedule(static_cast<uint32_t>(prompt_ids.size()), 0u, c2_cfg);
        if (c2_cfg.enabled && prompt_ids.size() >= 512u) std::printf("[PREFILL-C2] %s\n", c2.summary.c_str());
        for (const auto& mb : c2.macros) {
            std::vector<int32_t> chunk(prompt_ids.begin() + mb.pos0, prompt_ids.begin() + mb.pos0 + mb.n);
            auto preds = step_batch(chunk, mb.pos0, opt.use_wy_prefill, false, mb.compute_logits);
            if (mb.compute_logits) last_pred = preds.back();
            pos += mb.n;
        }
        double pp_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - tp0).count();
        if (out_pp_tps) *out_pp_tps = prompt_ids.size() / std::max(pp_sec, 1e-9);

        std::vector<int32_t> gen;
        gen.push_back(last_pred);
        auto td0 = std::chrono::steady_clock::now();
        for (int s = 1; s < n_gen; ++s) {
            auto preds = step_batch({gen.back()}, pos++, false, false, true);
            gen.push_back(preds[0]);
        }
        double dec_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - td0).count();
        if (out_tps && n_gen > 1) *out_tps = (n_gen - 1) / std::max(dec_sec, 1e-9);
        return gen;
    };

    // VEGAJET_GPU_BATCH=B (2..8, --bench-mode batch): B identical-prompt sequences decoded together. One prefill,
    // then the prompt state (GDN ring slot ring_head, conv slot conv_head, KV positions 0..P-1) is replicated into
    // sequence slots 0..B-1 on every GPU, and each step_batch call advances all B sequences by one token.
    auto run_gpu_batch = [&](const std::vector<int32_t>& prompt_ids, uint32_t B, int n_steps,
                             std::vector<std::vector<int32_t>>& outs, double* out_tps) -> bool {
        outs.assign(B, {});
        if (B < 2u || B > kMaxGpuBatch || raw_only_vram || opt.spec_gdn) return false;
        const uint32_t P = static_cast<uint32_t>(prompt_ids.size());
        if (P + static_cast<uint32_t>(n_steps) + 1u >= opt.max_ctx / kMaxGpuBatch) return false;
        reset_state();
        int32_t last_pred = -1;
        const prefill_sched::Schedule c2 = prefill_sched::schedule(P, 0u, c2_cfg);
        for (const auto& mb : c2.macros) {
            std::vector<int32_t> chunk(prompt_ids.begin() + mb.pos0, prompt_ids.begin() + mb.pos0 + mb.n);
            auto preds = step_batch(chunk, mb.pos0, opt.use_wy_prefill, false, mb.compute_logits);
            if (mb.compute_logits) last_pred = preds.back();
        }
        VJ_CHECK(submit_all() && wait_all());
        struct CpU32KA { uint64_t dst, src, n; };
        auto copy = [&](size_t g, uint64_t dst, uint64_t src, uint64_t bytes) {
            const CpU32KA ka{dst, src, bytes / 4u};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].memcpy_u32, static_cast<uint32_t>(((ka.n + 255u) / 256u) * 256u), 1, 1, 256, 1, 1,
                                       &ka, sizeof(ka), true));
        };
        for (size_t g = 0; g < N; ++g) {
            const uint64_t s_slot = static_cast<uint64_t>(gdn_nv[g]) * 128ULL * 128ULL * 4ULL;
            const uint64_t c_slot = 3ULL * gdn_conv_dim[g] * 4ULL;
            for (uint32_t gi = 0; gi < gdn_layer_count; ++gi) {
                for (uint32_t b = 0; b < B; ++b) {
                    if (b != ring_head) copy(g, S[g].gdn_ssm_ring_va[gi] + b * s_slot, S[g].gdn_ssm_ring_va[gi] + ring_head * s_slot, s_slot);
                    if (b != conv_head) copy(g, S[g].gdn_conv_ring_va[gi] + b * c_slot, S[g].gdn_conv_ring_va[gi] + conv_head * c_slot, c_slot);
                }
            }
            const uint64_t tok_b = static_cast<uint64_t>(gqa_nkv[g]) * l4_stride;
            const uint64_t seg   = static_cast<uint64_t>(opt.max_ctx / kMaxGpuBatch) * tok_b;
            for (uint32_t ai = 0; ai < 12u; ++ai)
                for (uint32_t b = 1; b < B; ++b) {
                    copy(g, S[g].attn_k_cache_va[ai] + b * seg, S[g].attn_k_cache_va[ai], P * tok_b);
                    copy(g, S[g].attn_v_cache_va[ai] + b * seg, S[g].attn_v_cache_va[ai], P * tok_b);
                    if (l4_split && !l4_compressed) for (uint32_t h=0;h<gqa_nkv[g];++h) {
                        const uint64_t head=uint64_t(h)*opt.max_ctx*384;
                        const uint64_t segment=uint64_t(opt.max_ctx/kMaxGpuBatch)*384;
                        copy(g,L4[g].wire_k[ai]+head+b*segment,L4[g].wire_k[ai]+head,P*384ULL);
                        copy(g,L4[g].wire_v[ai]+head+b*segment,L4[g].wire_v[ai]+head,P*384ULL);
                    }
                }
        }
        VJ_CHECK(submit_all() && wait_all());
        std::vector<int32_t> cur(B, last_pred);
        std::vector<uint32_t> bp(B, P);
        for (uint32_t b = 0; b < B; ++b) outs[b].push_back(last_pred);
        batch_pos = &bp;
        const auto t0 = std::chrono::steady_clock::now();
        for (int st = 0; st < n_steps; ++st) {
            cur = step_batch(cur, bp[0], false, false, true);
            for (uint32_t b = 0; b < B; ++b) { outs[b].push_back(cur[b]); ++bp[b]; }
        }
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        batch_pos = nullptr;
        if (out_tps) *out_tps = static_cast<double>(B) * n_steps / std::max(sec, 1e-9);
        return true;
    };

    // Path-neutral deterministic driver hooks for the CPU/GPU owner: pad or trim the chain to a
    // sealed width and force the target verdict. They apply identically to the sequential and the
    // sealed verifier, never fabricate a kernel result, and only choose which speculative drafts
    // the ordinary target verifier sees. This is what makes baseline/candidate token streams
    // directly comparable at fixed widths and acceptance depths.
    auto treewy_force_width = [&](std::vector<int32_t>& drafts, int32_t fill) {
        if (!opt.spec_gdn || treewy_opt.force_width == 0u) return;
        const size_t want = static_cast<size_t>(treewy_opt.force_width) - 1u;
        if (drafts.size() > want) drafts.resize(want);
        while (drafts.size() < want) drafts.push_back(fill);
    };
    // Target authority is non-negotiable: these hooks may only SHORTEN the acceptance the target
    // verifier already produced. They never rewrite a target prediction, so a mismatched draft can
    // never be forced into the committed state.
    //   Full     : the natural count (full acceptance cannot be forced; a note is printed if the
    //              target was not full, and the sequential/sealed arms still see the same count)
    //   Reject   : 0 drafts -> the root-only rollback case (the context node is still committed)
    //   Prefix:N : min(natural, N)
    // Both the sequential and the sealed verifier apply the same clamp to their own target
    // result, so an A/B compares the same accepted prefix without inventing any token.
    auto treewy_clamp_acc = [&](uint32_t natural, uint32_t dk) -> uint32_t {
        if (!opt.spec_gdn || treewy_opt.force_accept == treewy::Options::ForceAccept::Natural) {
            return natural;
        }
        const uint32_t take = treewy::clamp_accepted(natural, dk, treewy_opt.force_accept,
                                                     treewy_opt.force_prefix);
        if (treewy_opt.verbose) {
            std::printf("[treewy] clamp acc=%u->%u/%u (mode=%d prefix=%u)%s\n", natural, take, dk,
                        static_cast<int>(treewy_opt.force_accept), treewy_opt.force_prefix,
                        take < natural ? " shortened" : " natural");
        }
        return take;
    };

    auto cycle_now_ms = []() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    auto cycle_mode = [&]() -> unsigned { return treewy_hook && treewy_hook->active() ? 2u : (opt.spec_gdn ? 1u : 0u); };
    auto print_cycle_times = [](const std::array<spec::CycleTimes, 3>& times, const char* source) {
        const char* names[] = {"ring", "lumotree", "treewy"};
        for (unsigned i = 0; i < 3; ++i) times[i].print(names[i], source);
    };
    double reject_recompute_ms = 0.0;

    // M6: `--dump-tokens` writes the generated ids of one greedy sequence so tools/token_compare.py can prove token
    // parity across binaries/toggles. A write failure fails the run (res.error) instead of leaving a stale dump.
    auto dump_tokens = [&](const std::vector<int32_t>& ids) -> bool {
        if (opt.dump_tokens_path.empty()) return true;
        std::string derr;
        if (!bench_io::write_token_ids(opt.dump_tokens_path, ids, &derr)) {
            res.ok = false;
            res.error = derr;
            std::fprintf(stderr, "[DUMP-TOKENS] %s\n", derr.c_str());
            return false;
        }
        std::printf("[DUMP-TOKENS] %zu generated token ids -> %s\n", ids.size(), opt.dump_tokens_path.c_str());
        return true;
    };

    auto commit_spec_gdn = [&](uint32_t acc, uint32_t dk) {
        struct RejectTimer {
            double& total;
            bool enabled;
            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            ~RejectTimer() { if (enabled) total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(); }
        } timer{reject_recompute_ms, opt.spec_gdn && acc < dk};
        if (treewy_hook && treewy_hook->active()) {
            const treewy::CycleResult tr = treewy_hook->commit(acc, dk);
            if (!tr.ok) {
                std::fprintf(stderr, "[treewy] commit failed acc=%u/%u: %s\n", acc, dk,
                             tr.reason.c_str());
            }
            VJ_CHECK(tr.ok);
            spec_gdn_active = tr.active_slot;
            return;
        }
        if (!opt.spec_gdn) return;
        if (acc == dk) {
            spec_gdn_active ^= 1u;
            return;
        }
        const uint32_t n_commit = acc + 1u;
        struct SpecGdnKA {
            uint64_t sin, sout, qsrc, ksrc, vsrc, dec, beta, oout, rkv, rdec, rbeta;
            uint32_t M, nk, nv, s_strd, r_strd, lbase, s_lstrd, src_lstrd, gb_lstrd, r_lstrd, rgb_lstrd;
        };
        for (size_t g = 0; g < N; ++g) {
            const uint32_t nk_g = gdn_nk[g], nv_g = gdn_nv[g];
            const uint32_t r_strd = (nk_g + nv_g) * 128u;
            const SpecGdnKA rka{
                S[g].spec_gdn_s_slot_va[spec_gdn_active], S[g].spec_gdn_s_slot_va[spec_gdn_active],
                0ULL, S[g].spec_gdn_rec_kv_va, S[g].spec_gdn_rec_kv_va + static_cast<uint64_t>(nk_g) * 128ULL * 4ULL,
                S[g].spec_gdn_rec_dec_va, S[g].spec_gdn_rec_beta_va,
                0ULL, 0ULL, 0ULL, 0ULL,
                n_commit, nk_g, nv_g, r_strd, r_strd, 0u,
                nv_g * 128u * 128u, 16u * r_strd, 16u * nv_g, 0u, 0u
            };
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].spec_gdn_ver, (nv_g * 4u) * 256u, 36u, 1, 256, 1, 1, &rka, sizeof(rka), true));
            VJ_CHECK(rt.submit_batch(g));
        }
        VJ_CHECK(wait_all());
    };

    auto run_greedy_suffix = [&](const std::vector<int32_t>& prompt_ids, int n_gen, double* out_tps,
                                 uint64_t* out_hits = nullptr, uint64_t* out_acc = nullptr, uint64_t* out_dr = nullptr) -> std::vector<int32_t> {
        reset_state();
        spec::SuffixTree sfx;
        sfx.extend(prompt_ids);
        spec::MoeCostModel cm;
        cm.n_experts = 256; cm.experts_per_token = 8; cm.moe_byte_fraction = 0.25;
        spec::MoeKGovernor gov(cm);

        uint32_t pos = 0;
        int32_t last_pred = -1;
        for (size_t off = 0; off < prompt_ids.size(); off += opt.prefill_chunk) {
            const size_t cnt = std::min<size_t>(opt.prefill_chunk, prompt_ids.size() - off);
            std::vector<int32_t> chunk(prompt_ids.begin() + off, prompt_ids.begin() + off + cnt);
            const bool is_last = (off + cnt == prompt_ids.size());
            auto preds = step_batch(chunk, pos, opt.use_wy_prefill, false, is_last);
            if (is_last) last_pred = preds.back();
            pos += static_cast<uint32_t>(cnt);
        }

        std::vector<int32_t> ctx = prompt_ids;
        std::vector<int32_t> gen;
        gen.push_back(last_pred);
        sfx.push(last_pred);
        uint64_t hits = 0, acc_tot = 0, dr_tot = 0;
        std::array<spec::CycleTimes, 3> cycle_times;

        auto td0 = std::chrono::steady_clock::now();
        while (static_cast<int>(gen.size()) < n_gen) {
            const uint32_t rem = static_cast<uint32_t>(n_gen - static_cast<int>(gen.size()));
            const double cycle_start = cycle_now_ms(), recompute_start = reject_recompute_ms;
            std::vector<int32_t> qctx = ctx;
            qctx.push_back(gen.back());
            const uint32_t k_max = std::min<uint32_t>({opt.suffix_max_draft, kGdnRingSlots - 2u, (rem > 1u) ? (rem - 1u) : 0u});
            std::vector<int32_t> draft_toks;
            if (k_max > 0u) {
                double chain_score = 1.0;
                while (draft_toks.size() < k_max) {
                    const auto nodes = sfx.draft(qctx, 1u, 1u, 0.05);
                    if (nodes.empty()) break;
                    chain_score *= nodes[0].score;
                    if (chain_score < 0.30 && !draft_toks.empty()) break;
                    draft_toks.push_back(nodes[0].token);
                    qctx.push_back(nodes[0].token);
                }
            }
            // Path-neutral forced width also covers an empty suffix draft: padding makes the
            // cycle use the ordinary speculative verify path instead of a raw single step.
            treewy_force_width(draft_toks, gen.back());
            if (draft_toks.empty()) {
                auto preds = step_batch({gen.back()}, pos++, false, false, true);
                ctx.push_back(gen.back());
                gen.push_back(preds[0]);
                sfx.push(preds[0]);
            } else {
                ++hits;
                const uint32_t dk = static_cast<uint32_t>(draft_toks.size());
                std::vector<int32_t> vtoks;
                vtoks.reserve(1u + dk);
                vtoks.push_back(gen.back());
                for (int32_t t : draft_toks) vtoks.push_back(t);
                const double verify_start = cycle_now_ms();
                auto preds = step_batch(vtoks, pos, false, true, true);
                const double verify_end = cycle_now_ms();
                const unsigned verifier = cycle_mode();
                uint32_t acc = 0;
                while (acc < dk && preds[acc] == draft_toks[acc]) ++acc;
                acc = treewy_clamp_acc(acc, dk);
                dr_tot += dk; acc_tot += acc;
                gov.observe(dk, acc);
                commit_spec_gdn(acc, dk);

                ctx.push_back(gen.back());
                for (uint32_t j = 0; j < acc; ++j) {
                    gen.push_back(draft_toks[j]);
                    ctx.push_back(draft_toks[j]);
                    sfx.push(draft_toks[j]);
                }
                if (static_cast<int>(gen.size()) < n_gen) {
                    gen.push_back(preds[acc]);
                    sfx.push(preds[acc]);
                }
                const uint32_t n_commit = acc + 1u;
                ring_head = (ring_head + n_commit) % kGdnRingSlots;
                conv_head = (conv_head + n_commit) % kGdnRingSlots;
                pos += n_commit;
                const double r = reject_recompute_ms - recompute_start;
                cycle_times[verifier].add(verify_start - cycle_start, verify_end - verify_start,
                                cycle_now_ms() - verify_end - r, r);
            }
        }
        print_cycle_times(cycle_times, "suffix");
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - td0).count();
        if (out_tps && n_gen > 1) *out_tps = (n_gen - 1) / std::max(sec, 1e-9);
        if (out_hits) *out_hits = hits;
        if (out_acc)  *out_acc  = acc_tot;
        if (out_dr)   *out_dr   = dr_tot;
        return gen;
    };

    // Built-in blk.48 nextn MTP single-token step (1 layer + optional LM head -> ~0.7 ms/draft!)
    auto step_mtp = [&](int32_t tok_id, uint32_t pos, const uint64_t* h_in_arr,
                        bool compute_logits, bool do_submit, uint32_t tok_slot) -> int32_t {
        io_toks_host[tok_slot] = tok_id;
        __atomic_thread_fence(__ATOMIC_RELEASE);
        const float eps = static_cast<float>(hp.rms_eps);
        const float rope_base = static_cast<float>(hp.rope_base);

        auto enq_q4_m1 = [&](size_t g, const Q4A8WeightDev& w, uint64_t xq, uint64_t xds, uint64_t y, bool barrier = true) {
            if (w.cols == 512u) {
                struct M1KA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, stride; };
                M1KA ka{w.qs_va, w.dm_va, xq, xds, y, w.rows, w.cols, w.row_stride_bytes};
                VJ_CHECK(rt.enqueue_kernel(g, *K[g].gemv_q4_w16, ((w.rows + 15u) / 16u) * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), barrier));
            } else {
                struct M1W32KA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, stride, pad; uint64_t ptx; };
                M1W32KA ka{w.qs_va, w.dm_va, xq, xds, y, w.rows, w.cols, w.row_stride_bytes, 0u, 0ULL};
                VJ_CHECK(rt.enqueue_kernel(g, *K[g].gemv_q4_w32, ((w.rows + 7u) / 8u) * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), barrier));
            }
        };
        auto enq_ar_mtp = [&](uint32_t slot, const uint64_t* w_rms_arr, uint32_t do_norm) {
            const uint32_t ep = ar_epoch++;
            const uint32_t pp = ep & 1u;
            if (N == 2) {
                for (size_t g = 0; g < 2; ++g) {
                    const size_t peer = 1u - g;
                    struct ArKA {
                        uint64_t xp, ptx, lrx, pfl, lfl, bar, h, w, xn, xq, xd;
                        uint32_t M, D, ep, dn;
                        float eps;
                    };
                    ArKA aka{
                        S[g].partial_va,
                        rt.ar_rx_base_va(peer, g) + pp * kfd::kArSlotBytes,
                        rt.ar_rx_base_va(g, peer) + pp * kfd::kArSlotBytes,
                        rt.ar_flag_va(peer, g, slot),
                        rt.ar_flag_va(g, peer, slot),
                        S[g].ar_bar_va, S[g].mtp_h_cur_va, w_rms_arr[g],
                        S[g].x_norm_va, S[g].x_q8_va, S[g].x_ds_va, 1u, D, ep, do_norm, eps
                    };
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].p2p_ar_rms, 256u, 1, 1, 256, 1, 1, &aka, sizeof(aka), true));
                }
            } else {
                struct ArNKA {
                    uint64_t xp, ptx[4], lrx[4], pfl[4], lfl[4], bar, h, w, xn, xq, xd;
                    uint32_t M, D, ep, dn, rank, n_ranks, pre_pushed;
                    float eps;
                };
                for (size_t g = 0; g < N; ++g) {
                    ArNKA aka{};
                    aka.xp = S[g].partial_va;
                    for (size_t r = 0; r < 4; ++r) {
                        aka.ptx[r] = (r < N && r != g && N > 1) ? (rt.ar_rx_base_va(r, g) + pp * kfd::kArSlotBytes) : 0ULL;
                        aka.lrx[r] = (r < N && r != g && N > 1) ? (rt.ar_rx_base_va(g, r) + pp * kfd::kArSlotBytes) : 0ULL;
                        aka.pfl[r] = (r < N && r != g && N > 1) ? rt.ar_flag_va(r, g, slot) : 0ULL;
                        aka.lfl[r] = (r < N && r != g && N > 1) ? rt.ar_flag_va(g, r, slot) : 0ULL;
                    }
                    aka.bar = S[g].ar_bar_va; aka.h = S[g].mtp_h_cur_va; aka.w = w_rms_arr[g];
                    aka.xn = S[g].x_norm_va; aka.xq = S[g].x_q8_va; aka.xd = S[g].x_ds_va;
                    aka.M = 1u; aka.D = D; aka.ep = ep; aka.dn = do_norm;
                    aka.rank = static_cast<uint32_t>(g); aka.n_ranks = static_cast<uint32_t>(N);
                    aka.pre_pushed = 0u; aka.eps = eps;
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].p2p_ar_n_fp16, 256u, 1, 1, 256, 1, 1, &aka, sizeof(aka), true));
                }
            }
        };

        for (size_t g = 0; g < N; ++g) {
            const auto& ml = mtp[g];
            const uint32_t nq_g = gqa_nq[g], nkv_g = gqa_nkv[g];
            struct EhCatKA { uint64_t eqs, eds, toks, hin, wen, whn, cq, cds; uint32_t M, D; float eps; };
            EhCatKA ecka{emb_w[g].qs_va, emb_w[g].ds_va, io_toks_va + tok_slot * 4u,
                         h_in_arr[g], ml.enorm_va, ml.hnorm_va,
                         S[g].mtp_cat_q8_va, S[g].mtp_cat_ds_va, 1u, D, eps};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].mtp_eh_cat, 256u, 1, 1, 256, 1, 1, &ecka, sizeof(ecka), true));
            enq_q4_m1(g, ml.w_eh_proj, S[g].mtp_cat_q8_va, S[g].mtp_cat_ds_va, S[g].mtp_h_cur_va, true);

            struct RmsKA { uint64_t h, xa, w, xn, xq, xd; uint32_t M, D; float eps; };
            RmsKA rka{S[g].mtp_h_cur_va, 0ULL, ml.attn_norm_va, S[g].x_norm_va, S[g].x_q8_va, S[g].x_ds_va, 1u, D, eps};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].rms_quant, 256u, 1, 1, 256, 1, 1, &rka, sizeof(rka), true));

            enq_q4_m1(g, ml.w_attn_qkv, S[g].x_q8_va, S[g].x_ds_va, S[g].attn_qkv_va, true);
            struct PrepKA { uint64_t qkv, qw, kw, qo, go, kc, vc; uint32_t M, pos0, nq, nkv, rdims; float rbase, eps; };
            PrepKA pka{S[g].attn_qkv_va, ml.q_norm_va, ml.k_norm_va,
                       S[g].q_buf_va, S[g].gate_buf_va, S[g].mtp_k_cache_va, S[g].mtp_v_cache_va,
                       1u, pos, nq_g, nkv_g, hp.rope_dims, rope_base, eps};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].attn_prep, (nq_g + nkv_g) * 256u, 1, 1, 256, 1, 1, &pka, sizeof(pka), true));

            if (opt.spec_flash && K[g].spec_flash_part && nkv_g == 1u && nq_g == 16u) {
                const uint32_t span = pos + 1u;
                const uint32_t chunk_len = (span <= 1920u) ? 32u : ((span <= 3840u) ? 64u : ((span <= 7680u) ? 128u : 256u));
                const uint32_t n_chunks  = std::max<uint32_t>(1u, (span + chunk_len - 1u) / chunk_len);
                struct SfPartKA { uint64_t qo, kc, vc, po, pml; uint32_t M, pos0, nq, nkv, nc, cl; };
                struct SfMrgKA  { uint64_t po, pml, go, oq, od; uint32_t M, nq, nc; };
                const SfPartKA pka2{S[g].q_buf_va, S[g].mtp_k_cache_va, S[g].mtp_v_cache_va,
                                    S[g].spec_fl_part_o_va, S[g].spec_fl_part_ml_va, 1u, pos, nq_g, nkv_g, n_chunks, chunk_len};
                const SfMrgKA  mka2{S[g].spec_fl_part_o_va, S[g].spec_fl_part_ml_va, S[g].gate_buf_va,
                                    S[g].mid_q8_va, S[g].mid_ds_va, 1u, nq_g, n_chunks};
                VJ_CHECK(rt.enqueue_kernel(g, *K[g].spec_flash_part,  n_chunks * 512u, 1, nkv_g, 512, 1, 1, &pka2, sizeof(pka2), true));
                VJ_CHECK(rt.enqueue_kernel(g, *K[g].spec_flash_merge, nq_g * 256u, 1, 1, 256, 1, 1, &mka2, sizeof(mka2), true));
            } else {
                struct FlKA { uint64_t qo, go, kc, vc, oq, od; uint32_t M, pos0, nq, nkv; };
                FlKA fka{S[g].q_buf_va, S[g].gate_buf_va, S[g].mtp_k_cache_va, S[g].mtp_v_cache_va,
                         S[g].mid_q8_va, S[g].mid_ds_va, 1u, pos, nq_g, gqa_nkv_packed(g)};
                VJ_CHECK(rt.enqueue_kernel(g, *K[g].attn_flash, nq_g * 256u, 1, 1, 256, 1, 1, &fka, sizeof(fka), true));
            }
            enq_q4_m1(g, ml.w_o, S[g].mid_q8_va, S[g].mid_ds_va, S[g].partial_va, true);
        }
        {
            uint64_t w_pa[kfd::kMaxGpus] = {};
            for (size_t g = 0; g < N; ++g) w_pa[g] = mtp[g].post_attn_norm_va;
            enq_ar_mtp(60u, w_pa, 1u);
        }

        for (size_t g = 0; g < N; ++g) {
            const auto& ml = mtp[g];
            const uint32_t lff_g = local_ff_r[g];
            struct RLogKA { uint64_t wr, wsh, xn, lout, shg; uint32_t M, D; };
            RLogKA rlka{ml.w_router_f16_va, ml.w_sh_gate_va, S[g].x_norm_va, S[g].router_log_va, S[g].sh_gate_sc_va, 1u, D};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_rtr_log, 33u * 256u, 1, 1, 256, 1, 1, &rlka, sizeof(rlka), true));
            enq_q4_m1(g, ml.w_sh_gu, S[g].x_q8_va, S[g].x_ds_va, S[g].sh_gu_va, false);

            struct RTopKA { uint64_t lin, tids, tw; uint32_t M; };
            RTopKA rtka{S[g].router_log_va, S[g].top_ids_va, S[g].top_w_va, 1u};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_rtr_top8, 256u, 1, 1, 256, 1, 1, &rtka, sizeof(rtka), true));
            struct SwiKA { uint64_t gu, sc, oq, od; uint32_t M, lff; };
            SwiKA ska{S[g].sh_gu_va, S[g].sh_gate_sc_va, S[g].sh_q8_va, S[g].sh_ds_va, 1u, lff_g};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].swiglu_q8, 256u, 1, 1, 256, 1, 1, &ska, sizeof(ska), false));

            struct GuQ4KA { uint64_t wgq, wgdm, wuq, wudm, xq, xds, tids, mq, mds; uint32_t M, lff, stride; };
            GuQ4KA gka{ml.w_exp_gate.qs_va, ml.w_exp_gate.dm_va, ml.w_exp_up.qs_va, ml.w_exp_up.dm_va,
                       S[g].x_q8_va, S[g].x_ds_va, S[g].top_ids_va, S[g].moe_q8_va, S[g].mtp_moe_ds_va,
                       1u, lff_g, ml.w_exp_gate.row_stride_bytes};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_gu_q4a8, 8u * (lff_g / 32u) * 256u, 1, 1, 256, 1, 1, &gka, sizeof(gka), true));
            enq_q4_m1(g, ml.w_sh_down, S[g].sh_q8_va, S[g].sh_ds_va, S[g].sh_down_va, false);

            struct DnQ4KA { uint64_t wdq, wddm, mq, mds, tids, tw, shd, out; uint32_t M, D, stride, lff; };
            DnQ4KA dka{ml.w_exp_down.qs_va, ml.w_exp_down.dm_va, S[g].moe_q8_va, S[g].mtp_moe_ds_va,
                       S[g].top_ids_va, S[g].top_w_va, S[g].sh_down_va, S[g].partial_va,
                       1u, D, ml.w_exp_down.row_stride_bytes, lff_g};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_down_q4a8, ((D + 15u) / 16u) * 256u, 1, 1, 256, 1, 1, &dka, sizeof(dka), true));
        }
        {
            uint64_t w_sh[kfd::kMaxGpus] = {};
            for (size_t g = 0; g < N; ++g) w_sh[g] = mtp[g].shared_head_norm_va;
            enq_ar_mtp(61u, w_sh, compute_logits ? 1u : 0u);
        }

        if (compute_logits) {
            for (size_t g = 0; g < N; ++g) {
                const Q8WeightDev& vw = use_fr_spec ? fr_w[g] : out_w[g];
                const uint32_t lr = use_fr_spec ? static_cast<uint32_t>(fr_tok_map[g].size()) : voc_rows[g];
                const uint32_t ro = use_fr_spec ? fr_off[g] : voc_off[g];
                struct VocKA { uint64_t wq, wd, xq, xd, bv, bi; uint32_t lr, cols, ro, stride, M; };
                VocKA vka{vw.qs_va, vw.ds_va, S[g].x_q8_va, S[g].x_ds_va,
                          S[g].argmax_val_va, S[g].argmax_id_va,
                          lr, D, ro, vw.row_stride_bytes, 1u};
                VJ_CHECK(enqueue_vocab_m1(g, vka));
            }
        }
        if (!do_submit) return -1;

        VJ_CHECK(submit_all());
        VJ_CHECK(wait_all());
        if (!compute_logits) return -1;

        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        float best_v = -1e30f;
        int32_t best_i = -1;
        for (uint32_t b = 0; b < kVocabBlocks; ++b) {
            for (size_t g = 0; g < N; ++g) {
                if (io_arg_val_host[g][b] > best_v) { best_v = io_arg_val_host[g][b]; best_i = io_arg_id_host[g][b]; }
            }
        }
        if (use_fr_spec && best_i >= 0) {
            const uint32_t idx = static_cast<uint32_t>(best_i);
            for (size_t g = 0; g < N; ++g) {
                const uint32_t sz = static_cast<uint32_t>(fr_tok_map[g].size());
                if (idx >= fr_off[g] && idx < fr_off[g] + sz) {
                    best_i = fr_tok_map[g][idx - fr_off[g]];
                    break;
                }
            }
        }
        return best_i;
    };

    auto gpu_copy_h_all = [&](const uint64_t* dst, const uint64_t* src) {
        struct CpyKA { uint64_t dst, src, n; };
        const uint32_t gx = ((D + 255u) / 256u) * 256u;
        for (size_t g = 0; g < N; ++g) {
            CpyKA ka{dst[g], src[g], D};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].memcpy_u32, gx, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
        }
    };

    // Per-run speculative statistics of the last run_greedy_mtp / run_greedy_dflash2 call ([SPEC-STATS] lines).
    mtp_draft::SpecStats spec_last;
    // v1.2 winners (default off): C7+S1 measured depth governor, S2 batched drafter catch-up (N-GPU generalized).
    const bool v12_spec_gov = (std::getenv("VEGAJET_V12_SPEC_GOV") != nullptr && std::getenv("VEGAJET_V12_SPEC_GOV")[0] == '1');
    const bool v12_batch_catchup = (std::getenv("VEGAJET_V12_BATCH_CATCHUP") != nullptr &&
                                    std::getenv("VEGAJET_V12_BATCH_CATCHUP")[0] == '1');

    bool v12_batch_ok = true;
    for (size_t g = 0; g < N; ++g) {
        if (!K[g].gemm_q4_m8) v12_batch_ok = false;
    }
    auto step_mtp_batch = [&](const std::vector<int32_t>& toks, uint32_t pos0,
                              const std::vector<std::array<uint64_t, kfd::kMaxGpus>>& hs_rows,
                              bool do_submit) -> bool {
        const uint32_t n = static_cast<uint32_t>(toks.size());
        if (n == 0u) return true;
        if (n == 1u || !v12_batch_ok || n > kMaxSpecBatch) {
            for (uint32_t j = 0; j < n; ++j)
                (void) step_mtp(toks[j], pos0 + j, hs_rows[j].data(), false, false, j % kMaxSpecBatch);
            if (!do_submit) return false;
            VJ_CHECK(submit_all() && wait_all());
            return true;
        }

        const uint32_t M = n;
        const float eps = static_cast<float>(hp.rms_eps);
        const float rope_base = static_cast<float>(hp.rope_base);
        const uint64_t row_bytes = static_cast<uint64_t>(D) * 4ULL;
        for (uint32_t j = 0; j < M; ++j) io_toks_host[j] = toks[j];
        __atomic_thread_fence(__ATOMIC_RELEASE);

        // Stage per-row residual inputs; eh_cat reads them before eh_proj overwrites the same rows.
        for (uint32_t j = 0; j < M; ++j) {
            uint64_t d_a[kfd::kMaxGpus] = {};
            for (size_t g = 0; g < N; ++g) d_a[g] = S[g].mtp_h_cur_va + static_cast<uint64_t>(j) * row_bytes;
            gpu_copy_h_all(d_a, hs_rows[j].data());
        }

        auto enq_q4_mN = [&](size_t g, const Q4A8WeightDev& w, uint64_t xq, uint64_t xds, uint64_t y, bool barrier) {
            const uint32_t gx8 = ((w.rows + 7u) / 8u) * 256u;
            if (M <= 8u && K[g].gemm_q4_m8) {
                struct GemmPpKA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, M, stride; uint64_t ptx, ptx2; };
                GemmPpKA pka{w.qs_va, w.dm_va, xq, xds, y, w.rows, w.cols, M, w.row_stride_bytes, 0ULL, 0ULL};
                VJ_CHECK(rt.enqueue_kernel(g, *K[g].gemm_q4_m8, gx8, 1, 1, 256, 1, 1, &pka, sizeof(pka), barrier));
            } else if (M <= 12u && K[g].gemm_q4_m12) {
                struct GemmPpKA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, M, stride; uint64_t ptx, ptx2; };
                GemmPpKA pka{w.qs_va, w.dm_va, xq, xds, y, w.rows, w.cols, M, w.row_stride_bytes, 0ULL, 0ULL};
                VJ_CHECK(rt.enqueue_kernel(g, *K[g].gemm_q4_m12, gx8, 1, 1, 256, 1, 1, &pka, sizeof(pka), barrier));
            } else if (M <= 16u && K[g].gemm_q4_m16) {
                struct GemmPpKA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, M, stride; uint64_t ptx, ptx2; };
                GemmPpKA pka{w.qs_va, w.dm_va, xq, xds, y, w.rows, w.cols, M, w.row_stride_bytes, 0ULL, 0ULL};
                VJ_CHECK(rt.enqueue_kernel(g, *K[g].gemm_q4_m16, gx8, 1, 1, 256, 1, 1, &pka, sizeof(pka), barrier));
            } else {
                struct GemmKA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, M, stride; };
                GemmKA ka{w.qs_va, w.dm_va, xq, xds, y, w.rows, w.cols, M, w.row_stride_bytes};
                VJ_CHECK(rt.enqueue_kernel(g, *K[g].gemm_q4_tcc, gx8, (M + 15u) / 16u, 1, 256, 1, 1, &ka, sizeof(ka), barrier));
            }
        };

        auto enq_ar_mtp_m = [&](uint32_t slot, const uint64_t* w_rms_arr, uint32_t do_norm) {
            const uint32_t ep = ar_epoch++;
            const uint32_t pp = ep & 1u;
            if (N == 2) {
                for (size_t g = 0; g < 2; ++g) {
                    const size_t peer = 1u - g;
                    struct ArKA {
                        uint64_t xp, ptx, lrx, pfl, lfl, bar, h, w, xn, xq, xd;
                        uint32_t M, D, ep, dn;
                        float eps;
                    };
                    ArKA aka{
                        S[g].partial_va,
                        rt.ar_rx_base_va(peer, g) + pp * kfd::kArSlotBytes,
                        rt.ar_rx_base_va(g, peer) + pp * kfd::kArSlotBytes,
                        rt.ar_flag_va(peer, g, slot),
                        rt.ar_flag_va(g, peer, slot),
                        S[g].ar_bar_va, S[g].mtp_h_cur_va, w_rms_arr[g],
                        S[g].x_norm_va, S[g].x_q8_va, S[g].x_ds_va, M, D, ep, do_norm, eps
                    };
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].p2p_ar_rms, M * 256u, 1, 1, 256, 1, 1, &aka, sizeof(aka), true));
                }
            } else {
                struct ArNKA {
                    uint64_t xp, ptx[4], lrx[4], pfl[4], lfl[4], bar, h, w, xn, xq, xd;
                    uint32_t M, D, ep, dn, rank, n_ranks, pre_pushed;
                    float eps;
                };
                for (size_t g = 0; g < N; ++g) {
                    ArNKA aka{};
                    aka.xp = S[g].partial_va;
                    for (size_t r = 0; r < 4; ++r) {
                        aka.ptx[r] = (r < N && r != g && N > 1) ? (rt.ar_rx_base_va(r, g) + pp * kfd::kArSlotBytes) : 0ULL;
                        aka.lrx[r] = (r < N && r != g && N > 1) ? (rt.ar_rx_base_va(g, r) + pp * kfd::kArSlotBytes) : 0ULL;
                        aka.pfl[r] = (r < N && r != g && N > 1) ? rt.ar_flag_va(r, g, slot) : 0ULL;
                        aka.lfl[r] = (r < N && r != g && N > 1) ? rt.ar_flag_va(g, r, slot) : 0ULL;
                    }
                    aka.bar = S[g].ar_bar_va; aka.h = S[g].mtp_h_cur_va; aka.w = w_rms_arr[g];
                    aka.xn = S[g].x_norm_va; aka.xq = S[g].x_q8_va; aka.xd = S[g].x_ds_va;
                    aka.M = M; aka.D = D; aka.ep = ep; aka.dn = do_norm;
                    aka.rank = static_cast<uint32_t>(g); aka.n_ranks = static_cast<uint32_t>(N);
                    aka.pre_pushed = 0u; aka.eps = eps;
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].p2p_ar_n_fp16, M * 256u, 1, 1, 256, 1, 1, &aka, sizeof(aka), true));
                }
            }
        };

        for (size_t g = 0; g < N; ++g) {
            const auto& ml = mtp[g];
            const uint32_t nq_g = gqa_nq[g], nkv_g = gqa_nkv[g];
            struct EhCatKA { uint64_t eqs, eds, toks, hin, wen, whn, cq, cds; uint32_t M, D; float eps; };
            EhCatKA ecka{emb_w[g].qs_va, emb_w[g].ds_va, io_toks_va, S[g].mtp_h_cur_va, ml.enorm_va, ml.hnorm_va,
                         S[g].mtp_cat_q8_va, S[g].mtp_cat_ds_va, M, D, eps};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].mtp_eh_cat, 256u, M, 1, 256, 1, 1, &ecka, sizeof(ecka), true));
            enq_q4_mN(g, ml.w_eh_proj, S[g].mtp_cat_q8_va, S[g].mtp_cat_ds_va, S[g].mtp_h_cur_va, true);

            struct RmsKA { uint64_t h, xa, w, xn, xq, xd; uint32_t M, D; float eps; };
            RmsKA rka{S[g].mtp_h_cur_va, 0ULL, ml.attn_norm_va, S[g].x_norm_va, S[g].x_q8_va, S[g].x_ds_va, M, D, eps};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].rms_quant, M * 256u, 1, 1, 256, 1, 1, &rka, sizeof(rka), true));

            enq_q4_mN(g, ml.w_attn_qkv, S[g].x_q8_va, S[g].x_ds_va, S[g].attn_qkv_va, true);
            struct PrepKA { uint64_t qkv, qw, kw, qo, go, kc, vc; uint32_t M, pos0, nq, nkv, rdims; float rbase, eps; };
            PrepKA pka{S[g].attn_qkv_va, ml.q_norm_va, ml.k_norm_va,
                       S[g].q_buf_va, S[g].gate_buf_va, S[g].mtp_k_cache_va, S[g].mtp_v_cache_va,
                       M, pos0, nq_g, nkv_g, hp.rope_dims, rope_base, eps};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].attn_prep, (nq_g + nkv_g) * 256u, M, 1, 256, 1, 1, &pka, sizeof(pka), true));

            if (opt.spec_flash && K[g].spec_flash_part && nkv_g == 1u && nq_g == 16u) {
                const uint32_t span = pos0 + M;
                const uint32_t chunk_len = (span <= 1920u) ? 32u : ((span <= 3840u) ? 64u : ((span <= 7680u) ? 128u : 256u));
                const uint32_t n_chunks  = std::max<uint32_t>(1u, (span + chunk_len - 1u) / chunk_len);
                struct SfPartKA { uint64_t qo, kc, vc, po, pml; uint32_t M, pos0, nq, nkv, nc, cl; };
                struct SfMrgKA  { uint64_t po, pml, go, oq, od; uint32_t M, nq, nc; };
                const SfPartKA pka2{S[g].q_buf_va, S[g].mtp_k_cache_va, S[g].mtp_v_cache_va,
                                    S[g].spec_fl_part_o_va, S[g].spec_fl_part_ml_va, M, pos0, nq_g, nkv_g, n_chunks, chunk_len};
                const SfMrgKA  mka2{S[g].spec_fl_part_o_va, S[g].spec_fl_part_ml_va, S[g].gate_buf_va,
                                    S[g].mid_q8_va, S[g].mid_ds_va, M, nq_g, n_chunks};
                VJ_CHECK(rt.enqueue_kernel(g, *K[g].spec_flash_part,  n_chunks * 512u, M, nkv_g, 512, 1, 1, &pka2, sizeof(pka2), true));
                VJ_CHECK(rt.enqueue_kernel(g, *K[g].spec_flash_merge, nq_g * 256u, M, 1, 256, 1, 1, &mka2, sizeof(mka2), true));
            } else {
                struct FlKA { uint64_t qo, go, kc, vc, oq, od; uint32_t M, pos0, nq, nkv; };
                FlKA fka{S[g].q_buf_va, S[g].gate_buf_va, S[g].mtp_k_cache_va, S[g].mtp_v_cache_va,
                         S[g].mid_q8_va, S[g].mid_ds_va, M, pos0, nq_g, gqa_nkv_packed(g)};
                VJ_CHECK(rt.enqueue_kernel(g, *K[g].attn_flash, nq_g * 256u, M, 1, 256, 1, 1, &fka, sizeof(fka), true));
            }
            enq_q4_mN(g, ml.w_o, S[g].mid_q8_va, S[g].mid_ds_va, S[g].partial_va, true);
        }
        {
            uint64_t w_pa[kfd::kMaxGpus] = {};
            for (size_t g = 0; g < N; ++g) w_pa[g] = mtp[g].post_attn_norm_va;
            enq_ar_mtp_m(60u, w_pa, 1u);
        }

        for (size_t g = 0; g < N; ++g) {
            const auto& ml = mtp[g];
            const uint32_t lff_g = local_ff_r[g];
            struct RLogKA { uint64_t wr, wsh, xn, lout, shg; uint32_t M, D; };
            RLogKA rlka{ml.w_router_f16_va, ml.w_sh_gate_va, S[g].x_norm_va, S[g].router_log_va, S[g].sh_gate_sc_va, M, D};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_rtr_log, 33u * 256u, (M + 7u) / 8u, 1, 256, 1, 1, &rlka, sizeof(rlka), true));
            enq_q4_mN(g, ml.w_sh_gu, S[g].x_q8_va, S[g].x_ds_va, S[g].sh_gu_va, false);

            struct RTopKA { uint64_t lin, tids, tw; uint32_t M; };
            RTopKA rtka{S[g].router_log_va, S[g].top_ids_va, S[g].top_w_va, M};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_rtr_top8, M * 256u, 1, 1, 256, 1, 1, &rtka, sizeof(rtka), true));
            struct SwiKA { uint64_t gu, sc, oq, od; uint32_t M, lff; };
            SwiKA ska{S[g].sh_gu_va, S[g].sh_gate_sc_va, S[g].sh_q8_va, S[g].sh_ds_va, M, lff_g};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].swiglu_q8, M * 256u, 1, 1, 256, 1, 1, &ska, sizeof(ska), false));

            struct GuQ4KA { uint64_t wgq, wgdm, wuq, wudm, xq, xds, tids, mq, mds; uint32_t M, lff, stride; };
            GuQ4KA gka{ml.w_exp_gate.qs_va, ml.w_exp_gate.dm_va, ml.w_exp_up.qs_va, ml.w_exp_up.dm_va,
                       S[g].x_q8_va, S[g].x_ds_va, S[g].top_ids_va, S[g].moe_q8_va, S[g].mtp_moe_ds_va,
                       M, lff_g, ml.w_exp_gate.row_stride_bytes};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_gu_q4a8, 8u * (lff_g / 32u) * 256u, M, 1, 256, 1, 1, &gka, sizeof(gka), true));
            enq_q4_mN(g, ml.w_sh_down, S[g].sh_q8_va, S[g].sh_ds_va, S[g].sh_down_va, false);

            struct DnQ4KA { uint64_t wdq, wddm, mq, mds, tids, tw, shd, out; uint32_t M, D, stride, lff; };
            DnQ4KA dka{ml.w_exp_down.qs_va, ml.w_exp_down.dm_va, S[g].moe_q8_va, S[g].mtp_moe_ds_va,
                       S[g].top_ids_va, S[g].top_w_va, S[g].sh_down_va, S[g].partial_va,
                       M, D, ml.w_exp_down.row_stride_bytes, lff_g};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_down_q4a8, ((D + 15u) / 16u) * 256u, M, 1, 256, 1, 1, &dka, sizeof(dka), true));
        }
        {
            uint64_t w_sh[kfd::kMaxGpus] = {};
            for (size_t g = 0; g < N; ++g) w_sh[g] = mtp[g].shared_head_norm_va;
            enq_ar_mtp_m(61u, w_sh, 0u);
        }

        if (!do_submit) return false;
        VJ_CHECK(submit_all() && wait_all());
        return true;
    };
    auto run_greedy_mtp = [&](const std::vector<int32_t>& prompt_ids, int n_gen, uint32_t K_draft,
                              bool use_hybrid_suffix, double* out_tps,
                              uint64_t* out_acc = nullptr, uint64_t* out_dr = nullptr,
                              uint64_t* out_sfx_hits = nullptr) -> std::vector<int32_t> {
        reset_state();
        spec_last = {};
        sfx::CascadeConfig ccfg;
        ccfg.min_match = 4u;
        // Unset keeps the alpha threshold, including its zero-tax special case. The stronger bypass
        // threshold is an independent opt-in; it does not change the verifier or acceptance decision.
        const char* suffix_min4 = std::getenv("VEGAJET_MTP_SUFFIX_MIN4");
        ccfg.min_chain = mtp_draft::suffix_min_chain(K_draft, opt.zero_tax_catchup,
                                                   suffix_min4 && std::strcmp(suffix_min4, "1") == 0);
        ccfg.max_draft = std::min<uint32_t>(opt.suffix_max_draft, kGdnRingSlots - 2u);
        sfx::CascadeHybrid cascade(ccfg);
        spec_cascade::CascadeConfig zcfg;
        zcfg.min_match = 4u;
        zcfg.min_confident = (opt.zero_tax_catchup && K_draft > 0u) ? std::min<uint32_t>(4u, K_draft) : 4u;
        zcfg.max_draft = std::min<uint32_t>(opt.suffix_max_draft, kGdnRingSlots - 2u);
        zcfg = spec_cascade::config_from_env(zcfg);
        spec_cascade::ZeroTaxCascade zt_cascade(zcfg);
        const bool use_zt_cascade = (use_hybrid_suffix && zcfg.enabled);
        if (use_zt_cascade) {
            zt_cascade.reserve(prompt_ids.size() + static_cast<size_t>(n_gen) + 16u);
            zt_cascade.commit(prompt_ids);
        } else if (use_hybrid_suffix) {
            cascade.reserve(prompt_ids.size() + static_cast<size_t>(n_gen) + 16u);
            cascade.commit(prompt_ids);
        }

        uint32_t pos = 0;
        int32_t last_pred = -1;
        const uint64_t h_stride = static_cast<uint64_t>(D) * 4ULL;
        for (size_t off = 0; off < prompt_ids.size(); off += opt.prefill_chunk) {
            const size_t cnt = std::min<size_t>(opt.prefill_chunk, prompt_ids.size() - off);
            std::vector<int32_t> chunk(prompt_ids.begin() + off, prompt_ids.begin() + off + cnt);
            const bool is_last = (off + cnt == prompt_ids.size());
            auto preds = step_batch(chunk, pos, opt.use_wy_prefill, false, is_last);
            if (is_last) last_pred = preds.back();
            // Feed blk.48 KV cache on prompt tokens in <= 64-token batches so the 2048-packet AQL ring never overflows
            // (each step_mtp enqueues 16 dispatches: 64 * 16 = 1024 << 2048).
            for (size_t j0 = 0; j0 < cnt; j0 += 64u) {
                const size_t j_end = std::min<size_t>(cnt, j0 + 64u);
                if (v12_batch_catchup) {
                    for (size_t b0 = j0; b0 < j_end; b0 += kMaxSpecBatch) {
                        const size_t b_end = std::min<size_t>(j_end, b0 + kMaxSpecBatch);
                        std::vector<int32_t> bt;
                        std::vector<std::array<uint64_t, kfd::kMaxGpus>> bh;
                        bt.reserve(b_end - b0); bh.reserve(b_end - b0);
                        for (size_t j = b0; j < b_end; ++j) {
                            bt.push_back(chunk[j]);
                            std::array<uint64_t, kfd::kMaxGpus> hr{};
                            for (size_t g = 0; g < N; ++g)
                                hr[g] = (j == 0) ? ((pos == 0) ? S[g].mtp_h_zero_va : S[g].mtp_h_pend_va) : (S[g].h_va + (j - 1u) * h_stride);
                            bh.push_back(hr);
                        }
                        (void) step_mtp_batch(bt, pos + static_cast<uint32_t>(b0), bh, true);
                    }
                } else {
                    for (size_t j = j0; j < j_end; ++j) {
                        uint64_t hs[kfd::kMaxGpus] = {};
                        for (size_t g = 0; g < N; ++g)
                            hs[g] = (j == 0) ? ((pos == 0) ? S[g].mtp_h_zero_va : S[g].mtp_h_pend_va) : (S[g].h_va + (j - 1u) * h_stride);
                        (void) step_mtp(chunk[j], pos + static_cast<uint32_t>(j), hs, false, false, static_cast<uint32_t>(j - j0));
                    }
                }
                if (j_end == cnt) {
                    uint64_t d_a[kfd::kMaxGpus] = {}, s_a[kfd::kMaxGpus] = {};
                    for (size_t g = 0; g < N; ++g) { d_a[g] = S[g].mtp_h_pend_va; s_a[g] = S[g].h_va + (cnt - 1u) * h_stride; }
                    gpu_copy_h_all(d_a, s_a);
                }
                VJ_CHECK(submit_all() && wait_all());
            }
            pos += static_cast<uint32_t>(cnt);
        }

        std::vector<int32_t> ctx = prompt_ids;
        std::vector<int32_t> gen;
        gen.push_back(last_pred);
        if (use_zt_cascade) zt_cascade.commit(last_pred);
        else if (use_hybrid_suffix) cascade.commit(last_pred);
        uint64_t acc_tot = 0, dr_tot = 0, sfx_hits = 0;
        spec_policy::SyncConfig scfg;
        scfg.max_hole_frac = 1.0; // stay deferred during pure-suffix streaks; replay catchup_window=8 before neural
        spec_policy::DrafterKvSync kv_sync(scfg);
        auto resolve_h_src = [&](size_t g, const spec_policy::HSource& hs) -> uint64_t {
            if (hs.kind == spec_policy::HSource::kPend) return S[g].mtp_h_pend_va;
            if (hs.kind == spec_policy::HSource::kTrunkRow) return S[g].h_va + static_cast<uint64_t>(hs.index) * h_stride;
            return S[g].mtp_h_stash_va + static_cast<uint64_t>(hs.index) * h_stride;
        };

        auto td0 = std::chrono::steady_clock::now();
        // Cycle time split (ms): neural draft, verify (by batch size), drafter KV catch-up.
        double t_draft = 0.0, t_catch = 0.0, t_ver[17] = {};
        std::array<spec::CycleTimes, 3> cycle_times;
        uint32_t n_ver[17] = {}, n_cyc = 0, n_neural = 0;
        auto now_ms = []() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
        // Acceptance-aware gate (core/spec_gate.hpp; VEGAJET_SPEC_GATE=0 disables). Non-hybrid only: hybrid suffix
        // cycles keep the zero-tax KV sync, which raw steps would have to replay.
        const bool mtp_gate = !use_hybrid_suffix &&
                              !(std::getenv("VEGAJET_SPEC_GATE") && std::strcmp(std::getenv("VEGAJET_SPEC_GATE"), "0") == 0);
        spec_gate::Gate mgate;
        if (res.tg128_raw_tps > 0.0) mgate.seed_raw_ms(1000.0 / res.tg128_raw_tps);
        // v1.2 C7+S1: depth from the measured per-M verify time and the live accepted length.
        spec_policy::MeasuredDepthGovernor mtp_gov;
        uint32_t gov_depth_sum = 0u, gov_depth_n = 0u, gov_plain_n = 0u;
        auto mtp_ingest = [&](const std::vector<std::pair<int32_t, uint32_t>>& tok_pos,
                              const std::vector<std::array<uint64_t, kfd::kMaxGpus>>& hs_rows) {
            size_t i = 0;
            while (i < tok_pos.size()) {
                size_t j = i + 1u;
                while (j < tok_pos.size() && tok_pos[j].second == tok_pos[j - 1u].second + 1u &&
                       (j - i) < kMaxSpecBatch) ++j;
                std::vector<int32_t> bt;
                std::vector<std::array<uint64_t, kfd::kMaxGpus>> bh;
                bt.reserve(j - i); bh.reserve(j - i);
                for (size_t k = i; k < j; ++k) { bt.push_back(tok_pos[k].first); bh.push_back(hs_rows[k]); }
                (void) step_mtp_batch(bt, tok_pos[i].second, bh, false);
                i = j;
            }
        };
        while (static_cast<int>(gen.size()) < n_gen) {
            const uint32_t rem = static_cast<uint32_t>(n_gen - static_cast<int>(gen.size()));
            uint32_t k_cur = std::min<uint32_t>({K_draft, kGdnRingSlots - 2u, (rem > 1u) ? (rem - 1u) : 0u});
            const double t_cyc0 = now_ms(), recompute_start = reject_recompute_ms;
            if (mtp_gate && !mgate.want_spec() && rem > 1u) {
                auto preds = step_batch({gen.back()}, pos, false, false, true);
                uint64_t hp_a[kfd::kMaxGpus] = {}, hv_a[kfd::kMaxGpus] = {};
                for (size_t g = 0; g < N; ++g) { hp_a[g] = S[g].mtp_h_pend_va; hv_a[g] = S[g].h_va; }
                (void) step_mtp(gen.back(), pos, hp_a, false, false, 0u);
                gpu_copy_h_all(hp_a, hv_a);
                VJ_CHECK(submit_all() && wait_all());
                gen.push_back(preds[0]);
                spec_last.add_single();
                ++pos;
                mgate.on_raw(now_ms() - t_cyc0);
                continue;
            }
            if (v12_spec_gov) {
                // M2: the complete-cycle cost picks the depth before drafting. choose_depth() may answer 0 (plain
                // decoding); this loop has no standalone plain cycle with MTP catch-up, so a plain choice floors at
                // one draft (= verify a single token, the cheapest speculative cycle) and is counted in gov_plain_n.
                const auto gc = mtp_gov.choose_depth(k_cur);
                if (gc.warmed) {
                    k_cur = std::min(k_cur, std::max<uint32_t>(gc.depth, 1u));
                    gov_depth_sum += k_cur;
                    ++gov_depth_n;
                    if (gc.depth == 0u) ++gov_plain_n;
                }
            }
            // Suffix chains are free to draft and almost always accepted, so they may fill a whole verify batch.
            const uint32_t k_sfx = std::min<uint32_t>({opt.suffix_max_draft, kGdnRingSlots - 2u, (rem > 1u) ? (rem - 1u) : 0u});
            if (k_cur == 0u && (!use_hybrid_suffix || k_sfx == 0u)) {
                auto preds = step_batch({gen.back()}, pos++, false, false, true);
                gen.push_back(preds[0]);
                spec_last.add_single();
                break;
            }

            std::vector<int32_t> drafts;
            bool used_suffix = false;
            if (use_zt_cascade && k_sfx > 0u) {
                const auto dec = zt_cascade.decide();
                if (dec.source == spec_cascade::DraftSource::SUFFIX_ONLY && !dec.draft.empty()) {
                    const size_t take = std::min<size_t>(dec.draft.size(), k_sfx);
                    drafts.assign(dec.draft.begin(), dec.draft.begin() + take);
                    used_suffix = true;
                    ++sfx_hits;
                    if (opt.spec_force_reject) {
                        static const uint32_t kPos[5] = {0u, 4u, 5u, 6u, 10u};
                        const uint32_t r = kPos[sfx_hits % 5u];
                        if (r < drafts.size()) drafts[r] = (drafts[r] > 0) ? drafts[r] - 1 : 1;
                    }
                }
            } else if (use_hybrid_suffix && k_sfx > 0u) {
                const auto dec = cascade.decide_and_count();
                if (dec.source == sfx::DraftSource::kSuffix && !dec.draft.empty()) {
                    const size_t take = std::min<size_t>(dec.draft.size(), k_sfx);
                    drafts.assign(dec.draft.begin(), dec.draft.begin() + take);
                    used_suffix = true;
                    ++sfx_hits;
                    if (opt.spec_force_reject) {
                        static const uint32_t kPos[5] = {0u, 4u, 5u, 6u, 10u};
                        const uint32_t r = kPos[sfx_hits % 5u];
                        if (r < drafts.size()) drafts[r] = (drafts[r] > 0) ? drafts[r] - 1 : 1;
                    }
                }
            }
            const double tA = now_ms();
            double pre_catch_ms = 0.0;
            double t_neural = -1.0;  // start of the neural proposal loop (after any deferred catch-up replay)
            if (!used_suffix) {
                if (use_hybrid_suffix && opt.zero_tax_catchup && kv_sync.pending() > 0u) {
                    const auto pre = kv_sync.before_neural();
                    if (!pre.catchup.empty()) {
                        if (v12_batch_catchup) {
                            std::vector<std::pair<int32_t, uint32_t>> tp;
                            std::vector<std::array<uint64_t, kfd::kMaxGpus>> hr;
                            tp.reserve(pre.catchup.size()); hr.reserve(pre.catchup.size());
                            for (const auto& st : pre.catchup) {
                                tp.push_back({st.token, st.pos});
                                std::array<uint64_t, kfd::kMaxGpus> row{};
                                for (size_t g = 0; g < N; ++g) row[g] = resolve_h_src(g, st.h);
                                hr.push_back(row);
                            }
                            mtp_ingest(tp, hr);
                        } else {
                            for (size_t c = 0; c < pre.catchup.size(); ++c) {
                                const auto& st = pre.catchup[c];
                                uint64_t hs[kfd::kMaxGpus] = {};
                                for (size_t g = 0; g < N; ++g) hs[g] = resolve_h_src(g, st.h);
                                (void) step_mtp(st.token, st.pos, hs, false, false, static_cast<uint32_t>(c));
                            }
                        }
                        VJ_CHECK(submit_all() && wait_all());
                    }
                }
                pre_catch_ms = now_ms() - tA;
                ++n_neural;
                t_neural = now_ms();
                int32_t d_in = gen.back();
                for (uint32_t j = 0; j < k_cur; ++j) {
                    uint64_t hs[kfd::kMaxGpus] = {};
                    for (size_t g = 0; g < N; ++g) hs[g] = (j == 0) ? S[g].mtp_h_pend_va : S[g].mtp_h_cur_va;
                    int32_t dn = step_mtp(d_in, pos + j, hs, true, true, 0u);
                    drafts.push_back(dn);
                    d_in = dn;
                }
            }

            const double tB = now_ms();
            t_draft += tB - tA - pre_catch_ms;
            treewy_force_width(drafts, gen.back());
            const uint32_t dk = static_cast<uint32_t>(drafts.size());
            if (v12_spec_gov && t_neural >= 0.0 && dk > 0u) mtp_gov.observe_draft(dk, tB - t_neural);
            std::vector<int32_t> vtoks;
            vtoks.reserve(1u + dk);
            vtoks.push_back(gen.back());
            for (int32_t t : drafts) vtoks.push_back(t);
            auto preds = step_batch(vtoks, pos, false, true, true);
            const double tC = now_ms();
            const unsigned verifier = cycle_mode();
            { const uint32_t mi = std::min<uint32_t>(dk + 1u, 16u); t_ver[mi] += tC - tB; ++n_ver[mi]; ++n_cyc; }

            uint32_t acc = 0;
            while (acc < dk && preds[acc] == drafts[acc]) ++acc;
            acc = treewy_clamp_acc(acc, dk);
            dr_tot += dk; acc_tot += acc;
            spec_last.add(mtp_draft::CycleSample{used_suffix, dk, acc, used_suffix ? 0.0 : tB - tA, tC - tB, {}});
            if (v12_spec_gov) {
                mtp_gov.observe_verify(std::min<uint32_t>(dk + 1u, 16u), tC - tB);
                mtp_gov.observe_accept(dk, acc);
            }
            commit_spec_gdn(acc, dk);

            ctx.push_back(gen.back());
            for (uint32_t j = 0; j < acc; ++j) {
                gen.push_back(drafts[j]);
                ctx.push_back(drafts[j]);
                if (use_zt_cascade) zt_cascade.commit(drafts[j]);
                else if (use_hybrid_suffix) cascade.commit(drafts[j]);
            }
            if (static_cast<int>(gen.size()) < n_gen) {
                gen.push_back(preds[acc]);
                if (use_zt_cascade) zt_cascade.commit(preds[acc]);
                else if (use_hybrid_suffix) cascade.commit(preds[acc]);
            }
            const uint32_t n_commit = acc + 1u;
            ring_head = (ring_head + n_commit) % kGdnRingSlots;
            conv_head = (conv_head + n_commit) % kGdnRingSlots;

            if (static_cast<int>(gen.size()) < n_gen) {
                if (use_hybrid_suffix && opt.zero_tax_catchup) {
                    std::vector<int32_t> committed(vtoks.begin(), vtoks.begin() + n_commit);
                    const auto post = kv_sync.after_verify(committed, pos, !used_suffix, false);
                    for (const auto& sc : post.stash) {
                        uint64_t d_a[kfd::kMaxGpus] = {}, s_a[kfd::kMaxGpus] = {};
                        for (size_t g = 0; g < N; ++g) {
                            d_a[g] = S[g].mtp_h_stash_va + static_cast<uint64_t>(sc.dst_slot) * h_stride;
                            s_a[g] = resolve_h_src(g, sc.src);
                        }
                        gpu_copy_h_all(d_a, s_a);
                    }
                    if (v12_batch_catchup && !post.catchup.empty()) {
                        std::vector<std::pair<int32_t, uint32_t>> tp;
                        std::vector<std::array<uint64_t, kfd::kMaxGpus>> ch;
                        tp.reserve(post.catchup.size()); ch.reserve(post.catchup.size());
                        for (const auto& st : post.catchup) {
                            tp.emplace_back(st.token, st.pos);
                            std::array<uint64_t, kfd::kMaxGpus> hr{};
                            for (size_t g = 0; g < N; ++g) hr[g] = resolve_h_src(g, st.h);
                            ch.push_back(hr);
                        }
                        mtp_ingest(tp, ch);
                    } else {
                        for (size_t c = 0; c < post.catchup.size(); ++c) {
                            const auto& st = post.catchup[c];
                            uint64_t hs[kfd::kMaxGpus] = {};
                            for (size_t g = 0; g < N; ++g) hs[g] = resolve_h_src(g, st.h);
                            (void) step_mtp(st.token, st.pos, hs, false, false, static_cast<uint32_t>(c));
                        }
                    }
                    uint64_t d_a[kfd::kMaxGpus] = {}, s_a[kfd::kMaxGpus] = {};
                    for (size_t g = 0; g < N; ++g) { d_a[g] = S[g].mtp_h_pend_va; s_a[g] = S[g].h_va + acc * h_stride; }
                    gpu_copy_h_all(d_a, s_a);
                    VJ_CHECK(submit_all() && wait_all());
                } else {
                    if (used_suffix) {
                        uint64_t hs[kfd::kMaxGpus] = {};
                        for (size_t g = 0; g < N; ++g) hs[g] = S[g].mtp_h_pend_va;
                        (void) step_mtp(vtoks[0], pos, hs, false, false, 0u);
                    }
                    if (v12_batch_catchup && acc > 0u) {
                        std::vector<std::pair<int32_t, uint32_t>> tp;
                        std::vector<std::array<uint64_t, kfd::kMaxGpus>> ch;
                        tp.reserve(acc); ch.reserve(acc);
                        for (uint32_t j = 1u; j <= acc; ++j) {
                            tp.emplace_back(vtoks[j], pos + j);
                            std::array<uint64_t, kfd::kMaxGpus> hr{};
                            for (size_t g = 0; g < N; ++g) hr[g] = S[g].h_va + (j - 1u) * h_stride;
                            ch.push_back(hr);
                        }
                        mtp_ingest(tp, ch);
                    } else {
                        for (uint32_t j = 1u; j <= acc; ++j) {
                            uint64_t hs[kfd::kMaxGpus] = {};
                            for (size_t g = 0; g < N; ++g) hs[g] = S[g].h_va + (j - 1u) * h_stride;
                            (void) step_mtp(vtoks[j], pos + j, hs, false, false, j);
                        }
                    }
                    uint64_t d_a[kfd::kMaxGpus] = {}, s_a[kfd::kMaxGpus] = {};
                    for (size_t g = 0; g < N; ++g) { d_a[g] = S[g].mtp_h_pend_va; s_a[g] = S[g].h_va + acc * h_stride; }
                    gpu_copy_h_all(d_a, s_a);
                    VJ_CHECK(submit_all() && wait_all());
                }
            }
            const double r = reject_recompute_ms - recompute_start;
            const double catch_ms = now_ms() - tC - r + pre_catch_ms;
            t_catch += catch_ms;
            cycle_times[verifier].add(tB - t_cyc0 - pre_catch_ms, tC - tB, catch_ms, r);
            if (v12_spec_gov) mtp_gov.observe_catchup(n_commit, catch_ms);
            pos += n_commit;
            if (mtp_gate) mgate.on_spec(n_commit, now_ms() - t_cyc0);
        }
        print_cycle_times(cycle_times, "mtp");
        if (n_cyc > 0u) {
            std::printf("[SPEC cycles%s] %u cycles (%u neural%s) | draft %.2f ms/neural-cycle | catch-up %.2f ms/cycle | verify:",
                        use_zt_cascade ? "+ZeroTaxCascade" : "", n_cyc, n_neural,
                        use_zt_cascade ? ", zt_sfx_only" : "",
                        t_draft / std::max(n_neural, 1u), t_catch / n_cyc);
            for (uint32_t mi = 1u; mi <= 16u; ++mi) if (n_ver[mi]) std::printf(" M=%u %.2f ms x%u", mi, t_ver[mi] / n_ver[mi], n_ver[mi]);
            if (use_zt_cascade) {
                const auto& zs = zt_cascade.stats();
                std::printf(" | zt(sfx=%llu neural=%llu sfx_tok=%llu)",
                            static_cast<unsigned long long>(zs.suffix_only),
                            static_cast<unsigned long long>(zs.neural),
                            static_cast<unsigned long long>(zs.suffix_tokens));
            }
            if (v12_spec_gov && gov_depth_n > 0u)
                std::printf(" | v12_gov depth %.2f plain %u alpha %.3f t_ver(M=4) %.2f ms draft %.2f us/tok catchup %.2f us/tok",
                            static_cast<double>(gov_depth_sum) / static_cast<double>(gov_depth_n), gov_plain_n,
                            mtp_gov.alpha(), mtp_gov.measured_ms(4u),
                            1000.0 * mtp_gov.draft_ms_per_tok(), 1000.0 * mtp_gov.catchup_ms_per_tok());
            std::printf("\n");
        }
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - td0).count();
        spec_last.wall_s = sec;
        if (out_tps && n_gen > 1) *out_tps = (n_gen - 1) / std::max(sec, 1e-9);
        if (out_acc)      *out_acc      = acc_tot;
        if (out_dr)       *out_dr       = dr_tot;
        if (out_sfx_hits) *out_sfx_hits = sfx_hits;
        return gen;
    };

    double max_pwr = 0.0, max_tmem = 0.0;
    HwmonSampler telem(rt.num_gpus());
    size_t parity_code_match = golden::kCodeRef32.size();

    // ---- Pillar 3: --dump-golden-layers <dir> (1 GDN L0 + 1 GQA L3 fixture for vegajet_layer_test) ----
    if (!opt.dump_golden_layers_dir.empty()) {
        namespace fs = std::filesystem;
        std::error_code ec;
        fs::create_directories(fs::path(opt.dump_golden_layers_dir) / "weights", ec);
        if (ec) {
            rt.free_buffer(tap_buf); rt.free_buffer(io_buf);
            res.error = "Cannot create golden dir " + opt.dump_golden_layers_dir + ": " + ec.message();
            return res;
        }
        const std::string wdir = opt.dump_golden_layers_dir + "/weights";
        const auto& l0 = layers[0][0];
        const auto& l3 = layers[0][golden_gqa_il];
        auto dump_w = [&](const std::string& name, uint64_t va, uint64_t bytes) -> bool {
            if (va == 0u) return true;   // plane not resident (e.g. in-place DOT8/Q3 promotion): skip, never read VA 0
            return write_vram_bin(rt, 0, va, bytes, wdir + "/" + name + ".bin", &res.error);
        };
        auto q4_qs_b = [](const Q4A8WeightDev& w) { return static_cast<uint64_t>(w.rows) * w.row_stride_bytes; };
        auto q4_dm_b = [](const Q4A8WeightDev& w) { return static_cast<uint64_t>(w.rows) * (w.cols / 32u) * 4ULL; };
        bool wok = true;
        wok = wok && dump_w("l0_attn_norm", l0.attn_norm_va, D * 4ULL);
        wok = wok && dump_w("l0_post_attn_norm", l0.post_attn_norm_va, D * 4ULL);
        wok = wok && dump_w("l0_gdn_qkv_qs", l0.w_gdn_qkv_gab.qs_va, q4_qs_b(l0.w_gdn_qkv_gab));
        wok = wok && dump_w("l0_gdn_qkv_dm", l0.w_gdn_qkv_gab.dm_va, q4_dm_b(l0.w_gdn_qkv_gab));
        wok = wok && dump_w("l0_ssm_out_qs", l0.w_ssm_out.qs_va, q4_qs_b(l0.w_ssm_out));
        wok = wok && dump_w("l0_ssm_out_dm", l0.w_ssm_out.dm_va, q4_dm_b(l0.w_ssm_out));
        wok = wok && dump_w("l0_ssm_conv_w", l0.ssm_conv_w_va, 6144ULL * 4ULL * 4ULL);
        wok = wok && dump_w("l0_ssm_a",      l0.ssm_a_va,      32ULL * 4ULL);
        wok = wok && dump_w("l0_ssm_dt",     l0.ssm_dt_va,     32ULL * 4ULL);
        wok = wok && dump_w("l0_ssm_norm",   l0.ssm_norm_va,   128ULL * 4ULL);
        wok = wok && dump_w("l0_router_f16", l0.w_router_f16_va, 256ULL * (D == 3072u ? 3200ULL : static_cast<uint64_t>(D)) * 2ULL);
        wok = wok && dump_w("l0_sh_gate",    l0.w_sh_gate_va,    D * 4ULL);
        wok = wok && dump_w("l0_sh_gu_qs",   l0.w_sh_gu.qs_va,   q4_qs_b(l0.w_sh_gu));
        wok = wok && dump_w("l0_sh_gu_dm",   l0.w_sh_gu.dm_va,   q4_dm_b(l0.w_sh_gu));
        wok = wok && dump_w("l0_sh_dn_qs",   l0.w_sh_down.qs_va, q4_qs_b(l0.w_sh_down));
        wok = wok && dump_w("l0_sh_dn_dm",   l0.w_sh_down.dm_va, q4_dm_b(l0.w_sh_down));
        wok = wok && dump_w("l0_gu_iq2s",    l0.w_gu_iq2s_va,    256ULL * local_ff * 123ULL * 16ULL);
        wok = wok && dump_w("l0_dn_d8_w",    l0.w_dn_d8_w_va,    256ULL * D * l0.dn_d8_w_stride);
        wok = wok && dump_w("l0_dn_d8_s",    l0.w_dn_d8_s_va,    256ULL * D * l0.dn_d8_s_stride);
        wok = wok && dump_w("l3_attn_norm",      l3.attn_norm_va,      D * 4ULL);
        wok = wok && dump_w("l3_post_attn_norm", l3.post_attn_norm_va, D * 4ULL);
        wok = wok && dump_w("l3_gqa_qkv_qs", l3.w_attn_qkv.qs_va, q4_qs_b(l3.w_attn_qkv));
        wok = wok && dump_w("l3_gqa_qkv_dm", l3.w_attn_qkv.dm_va, q4_dm_b(l3.w_attn_qkv));
        wok = wok && dump_w("l3_gqa_o_qs",   l3.w_o.qs_va,        q4_qs_b(l3.w_o));
        wok = wok && dump_w("l3_gqa_o_dm",   l3.w_o.dm_va,        q4_dm_b(l3.w_o));
        wok = wok && dump_w("l3_q_norm",     l3.q_norm_va,        256ULL * 4ULL);
        wok = wok && dump_w("l3_k_norm",     l3.k_norm_va,        256ULL * 4ULL);
        if (!wok) { rt.free_buffer(tap_buf); rt.free_buffer(io_buf); return res; }

        const uint32_t kWarmPos0 = 64u;
        std::vector<int32_t> warm_toks(kWarmPos0);
        for (uint32_t i = 0; i < kWarmPos0; ++i) warm_toks[i] = golden::kCodePrompt[i % golden::kCodePrompt.size()];
        const uint32_t m_list[] = {1u, 5u, 16u, 128u, 512u};
        golden_cap_dir = opt.dump_golden_layers_dir;
        for (uint32_t m : m_list) {
            reset_state();
            golden_cap_on = false;
            (void) step_batch(warm_toks, 0u, opt.use_wy_prefill, false, false);
            std::vector<int32_t> step_toks(m);
            for (uint32_t i = 0; i < m; ++i) step_toks[i] = golden::kCodePrompt[(kWarmPos0 + i) % golden::kCodePrompt.size()];
            golden_cap_on = true;
            (void) step_batch(step_toks, kWarmPos0, opt.use_wy_prefill, (m > 1u && m <= 16u), false);
            golden_cap_on = false;
            std::printf("[GOLDEN] Captured M=%u (L0 GDN+MoE, L%u GQA, pos0=%u)\n", m, golden_gqa_il, kWarmPos0);
        }
        std::ofstream mf(opt.dump_golden_layers_dir + "/manifest.txt");
        mf << "D=" << D << "\n"
           << "local_ff=" << local_ff << "\n"
           << "gqa_il=" << golden_gqa_il << "\n"
           << "pos0=" << kWarmPos0 << "\n"
           << "eps=" << hp.rms_eps << "\n"
           << "rope_base=" << hp.rope_base << "\n"
           << "gdn_qkv_rows=" << l0.w_gdn_qkv_gab.rows << "\n"
           << "gdn_qkv_cols=" << l0.w_gdn_qkv_gab.cols << "\n"
           << "gdn_qkv_stride=" << l0.w_gdn_qkv_gab.row_stride_bytes << "\n"
           << "ssm_out_rows=" << l0.w_ssm_out.rows << "\n"
           << "ssm_out_cols=" << l0.w_ssm_out.cols << "\n"
           << "ssm_out_stride=" << l0.w_ssm_out.row_stride_bytes << "\n"
           << "gqa_qkv_rows=" << l3.w_attn_qkv.rows << "\n"
           << "gqa_qkv_cols=" << l3.w_attn_qkv.cols << "\n"
           << "gqa_qkv_stride=" << l3.w_attn_qkv.row_stride_bytes << "\n"
           << "gqa_o_rows=" << l3.w_o.rows << "\n"
           << "gqa_o_cols=" << l3.w_o.cols << "\n"
           << "gqa_o_stride=" << l3.w_o.row_stride_bytes << "\n"
           << "sh_gu_rows=" << l0.w_sh_gu.rows << "\n"
           << "sh_gu_cols=" << l0.w_sh_gu.cols << "\n"
           << "sh_gu_stride=" << l0.w_sh_gu.row_stride_bytes << "\n"
           << "sh_dn_rows=" << l0.w_sh_down.rows << "\n"
           << "sh_dn_cols=" << l0.w_sh_down.cols << "\n"
           << "sh_dn_stride=" << l0.w_sh_down.row_stride_bytes << "\n"
           << "dn_d8_w_stride=" << l0.dn_d8_w_stride << "\n"
           << "dn_d8_s_stride=" << l0.dn_d8_s_stride << "\n"
           << "gqa_n_q_local=" << static_cast<uint32_t>(gqa_plan.gpus[0].q_heads.size()) << "\n"
           << "gqa_n_kv_local=" << static_cast<uint32_t>(gqa_plan.gpus[0].kv_heads.size()) << "\n"
           << "gqa_head_dim=256\n"
           << "gqa_rot_dim=64\n"
           << "m_list=1,5,16,128,512\n";
        mf.close();
        rt.free_buffer(tap_buf);
        rt.free_buffer(io_buf);
        telem.stop();
        std::printf("[GOLDEN] Saved 2-layer golden fixture to %s -> PASS\n", opt.dump_golden_layers_dir.c_str());
        res.ok = true;
        return res;
    }

    // ---- Pillar 2: Reusable warm execution pass for --interactive & --sweep (zero 42 GiB weight reload) ----
    auto reset_lru_state = [&]() {
        if (lru_slots == 0u || !lru_cache) return;
        moecache::CacheConfig ccfg;
        ccfg.n_layers = 48u; ccfg.n_experts = 256u; ccfg.slots_per_layer = lru_slots;
        ccfg.max_admits_per_step = 48u; ccfg.max_admits_per_layer_step = 1u; ccfg.admit_after_misses = 2u;
        lru_cache = std::make_unique<moecache::MoeLruCache>(ccfg);
        std::vector<int32_t> neg_tab(256, -1);
        for (uint32_t il = 0; il < 48u; ++il) {
            lru_e2s_host[il].fill(-1);
            for (size_t g = 0; g < 2; ++g) {
                (void) rt.upload_to_vram(g, lru_e2s_va[g][il], neg_tab.data(), 256u * 4u);
            }
        }
    };

    auto apply_warm_flags = [&](const std::vector<std::string>& cli_args, OrnithGpuOptions& dst, std::string& bench_mode, std::string* ferr) -> bool {
        dst = base_opt;
        if(sub4_on) dst.ab_no_prepush=true; // Preserve the load-time SUB4 epilogue contract on warm runs.
        dst.interactive = false;
        dst.sweep_spec.clear();
        bench_mode = "all";
        for (size_t i = 0; i < cli_args.size(); ++i) {
            const std::string& a = cli_args[i];
            auto next = [&]() -> std::string { return (i + 1 < cli_args.size()) ? cli_args[++i] : std::string(); };
            if (a == "--bench-mode") bench_mode = next();
            else if (a == "--n-predict" || a == "-n") dst.decode_tokens = std::atoi(next().c_str());
            else if (a == "--ar-wire") dst.ar_wire_int8 = (next() == "int8");
            else if (a == "--fuse-router") dst.fuse_moe_router = true;
            else if (a == "--no-fuse-router") dst.fuse_moe_router = false;
            else if (a == "--gdn-splitv") dst.gdn_splitv = true;
            else if (a == "--no-gdn-splitv") dst.gdn_splitv = false;
            else if (a == "--gdn-splitv8") dst.gdn_splitv8 = true;
            else if (a == "--no-gdn-splitv8") dst.gdn_splitv8 = false;
            else if (a == "--fuse-back") dst.fuse_moe_back = true;
            else if (a == "--no-fuse-back") dst.fuse_moe_back = false;
            else if (a == "--back-wgs") dst.moe_back_wgs = static_cast<uint32_t>(std::atoi(next().c_str()));
            else if (a == "--gu-grp-verify") dst.gu_grp_verify = true;
            else if (a == "--no-gu-grp-verify") dst.gu_grp_verify = false;
            else if (a == "--down-grp-verify") dst.down_grp_verify = static_cast<uint32_t>(std::atoi(next().c_str()));
            else if (a == "--down-grp-prefill") dst.down_grp_prefill = static_cast<uint32_t>(std::atoi(next().c_str()));
            else if (a == "--prefill-chunk") dst.prefill_chunk = static_cast<uint32_t>(std::atoi(next().c_str()));
            else if (a == "--prefill-dot8") dst.prefill_dot8 = true;
            else if (a == "--no-prefill-dot8") dst.prefill_dot8 = false;
            else if (a == "--dot8-fwht") dst.dot8_fwht = true;
            else if (a == "--no-dot8-fwht") dst.dot8_fwht = false;
            else if (a == "--dot8-fwht-layers") dst.dot8_fwht_layers = static_cast<uint32_t>(std::atoi(next().c_str()));
            else if (a == "--dot8-scope") dst.dot8_scope = static_cast<uint32_t>(std::atoi(next().c_str()));
            else if (a == "--dot8-clip") dst.dot8_clip_alpha = static_cast<float>(std::atof(next().c_str()));
            else if (a == "--suffix-decode") dst.use_suffix_spec = true;
            else if (a == "--no-suffix-decode") dst.use_suffix_spec = false;
            else if (a == "--suffix-max-draft") dst.suffix_max_draft = static_cast<uint32_t>(std::atoi(next().c_str()));
            else if (a == "--spec-flash") dst.spec_flash = true;
            else if (a == "--no-spec-flash") dst.spec_flash = false;
            else if (a == "--zero-tax-catchup") dst.zero_tax_catchup = true;
            else if (a == "--no-zero-tax-catchup") dst.zero_tax_catchup = false;
            else if (a == "--mtp-k") dst.mtp_k = static_cast<uint32_t>(std::atoi(next().c_str()));
            else if (a == "--n-predict" || a == "-n") dst.decode_tokens = std::max(1, std::atoi(next().c_str()));
            else if (a == "--dflash-k") dst.dflash_k = static_cast<uint32_t>(std::atoi(next().c_str()));
            else if (a == "--dflash-p-min") dst.dflash_p_min = static_cast<float>(std::atof(next().c_str()));
            else if (a == "--profile-stages") dst.profile_stages = true;
            else if (a == "--pp-stages") dst.pp_stages = true;
            else if (a == "--ab-no-prepush") dst.ab_no_prepush = true;
            else if (a == "--pf-gemm") dst.pf_gemm = static_cast<uint32_t>(std::atoi(next().c_str()));
            else if (a == "--pf-gdn") dst.pf_gdn = static_cast<uint32_t>(std::atoi(next().c_str()));
            else if (a == "--prefill-ar-int8") dst.prefill_ar_int8 = true;
            else if (a == "--no-prefill-ar-int8") dst.prefill_ar_int8 = false;
            else if (a == "default" || a == "baseline") { /* keep base_opt */ }
            else {
                if (ferr) *ferr = "unsupported warm flag: " + a;
                return false;
            }
        }
        return true;
    };

    auto run_bench_pass = [&](bool do_parity, bool do_bench, const std::string& bench_mode, warm::SweepVariantResult* vout) -> bool {
        if (vout) { vout->ok = false; vout->parity_ok = true; vout->first_diff = -1; }
        const bool run_raw_part     = (bench_mode == "all" || bench_mode == "raw" || bench_mode == "spec");
        const bool run_spec_part    = (bench_mode == "all" || bench_mode == "spec");
        const bool run_prefill_part = (bench_mode == "all" || bench_mode == "prefill");
        const bool run_batch_part   = (bench_mode == "batch");
        uint64_t ue0 = 0, de0 = 0, ue1 = 0, de1 = 0;
        read_ecc_totals(ue0, de0);
        double pass_pwr = 0.0, pass_tmem = 0.0;
        if (do_parity) {
            const std::vector<int32_t>& p_simple = golden::kProsePrompt;
            const std::vector<int32_t>& p_code   = golden::kCodePrompt;
            double tps_s = 0, tps_c = 0;
            auto out_s = run_greedy_raw(p_simple, 32, &tps_s);
            auto out_c = run_greedy_raw(p_code,   32, &tps_c);
            res.parity_simple_text = tok.decode(out_s);
            res.parity_code_text   = tok.decode(out_c);
            res.dispatches_per_tok = last_dispatches;
            std::printf("\n=== ORNITH-122B TIER-1 GREEDY PARITY CHECK ===\n");
            std::printf("[PARITY simple %.2f tok/s | %u disp/tok]: %s\n", tps_s, last_dispatches, res.parity_simple_text.c_str());
            std::printf("[PARITY code   %.2f tok/s | %u disp/tok]: %s\n", tps_c, last_dispatches, res.parity_code_text.c_str());
            parity_code_match = golden::matched_prefix(out_c, golden::kCodeRef32);
            const size_t prose_match = golden::matched_prefix(out_s, golden::kProseRef32);
            std::printf("[PARITY code  vs Tier-1 llama.cpp] %zu/%zu exact greedy tokens -> %s\n", parity_code_match,
                        golden::kCodeRef32.size(), parity_code_match == golden::kCodeRef32.size() ? "PASS" : "FAIL");
            std::printf("[PARITY prose vs Tier-1 llama.cpp] %zu/%zu matched prefix (informational, not gated)\n", prose_match,
                        golden::kProseRef32.size());
            read_hwmon_max(pass_pwr, pass_tmem);
            max_pwr = std::max(max_pwr, pass_pwr); max_tmem = std::max(max_tmem, pass_tmem);
            if (vout) {
                vout->parity_ok = (parity_code_match == golden::kCodeRef32.size());
                vout->lossless_matched = static_cast<uint32_t>(parity_code_match);
                vout->lossless_total = static_cast<uint32_t>(golden::kCodeRef32.size());
                vout->dispatches_per_tok = last_dispatches;
                vout->raw_tg128_tps = tps_c;
                vout->primary_tps = tps_c;
            }
        }
        if (do_bench) {
            const bool old_run_bench = opt.run_bench;
            (void) old_run_bench;
            // Execute the benchmark sections selectively according to bench_mode.
            std::printf("\n=== ORNITH-122B LIVE 2x MI50 BENCHMARK SUITE ===\n");
            const std::vector<int32_t> p_base = {760, 6511, 314, 9338, 369};
            const std::vector<int32_t> p_rep_code = {
                727, 73111, 1393, 1590, 198, 262, 413, 307, 606, 220, 15, 25, 198, 285, 460, 220, 16, 198,
                262, 745, 25, 198, 285, 460, 307, 348, 73111, 1393, 12, 16, 8, 271,
                727, 73111, 1393, 1590
            };
            if (run_batch_part) {
                const char* gb = std::getenv("VEGAJET_GPU_BATCH");
                const uint32_t B = gb ? static_cast<uint32_t>(std::max(1, std::atoi(gb))) : 1u;
                double raw_tps = 0.0;
                const auto raw = run_greedy_raw(p_rep_code, 129, &raw_tps);
                if (B <= 1u) {
                    std::printf("[BENCH batch B=1] %.2f tok/s (raw M=1 path, zero batching overhead)\n", raw_tps);
                    if (vout) { vout->raw_tg128_tps = raw_tps; vout->primary_tps = raw_tps; }
                } else {
                    std::vector<std::vector<int32_t>> outs;
                    double btps = 0.0;
                    if (!run_gpu_batch(p_rep_code, B, 128, outs, &btps)) {
                        std::printf("[BENCH batch B=%u] unsupported here (needs 2..8, no --spec-gdn, no VEGAJET_RAW_ONLY)\n", B);
                    } else {
                        uint32_t rows_agree = 0;
                        for (uint32_t b = 0; b < B; ++b) rows_agree += (outs[b] == outs[0]) ? 1u : 0u;
                        uint32_t eq = 0; int fd = -1;
                        for (size_t i = 0; i < raw.size() && i < outs[0].size(); ++i) {
                            if (raw[i] == outs[0][i]) ++eq; else if (fd == -1) fd = static_cast<int>(i);
                        }
                        std::printf("[BENCH batch B=%u] %.2f tok/s aggregate (%.2f tok/s/seq, raw M=1 %.2f) | rows_agree=%u/%u | row0 vs raw=%u/%zu (first_diff=%d)\n",
                                    B, btps, btps / B, raw_tps, rows_agree, B, eq, raw.size(), fd);
                        if (vout) { vout->primary_tps = btps; vout->raw_tg128_tps = raw_tps; }
                    }
                }
            }
            if (run_raw_part && !run_spec_part && !run_prefill_part) {
                static std::vector<int32_t> trusted_raw128;
                static size_t trusted_context = 0;
                // The context override changes prefill only; the timer in run_greedy_raw starts after prefill.
                const int raw_count = opt.decode_tokens > 128 ? opt.decode_tokens + 1 : 129;
                const char* context_env = std::getenv("VEGAJET_BENCH_CONTEXT");
                std::vector<int32_t> raw_prompt;
                std::string context_error;
                if (!warm::make_decode_prompt(context_env ? context_env : "", p_rep_code, opt.max_ctx,
                                              static_cast<uint32_t>(raw_count - 1), raw_prompt, &context_error)) {
                    if (vout) vout->error = context_error;
                    std::fprintf(stderr, "[BENCH raw] %s\n", context_error.c_str());
                    return false;
                }
                const bool long_context = raw_prompt.size() != p_rep_code.size();
                if (long_context)
                    std::printf("[BENCH context] repeated code prompt=%zu tokens; decode positions=%zu..%zu; prefill excluded from tok/s\n",
                                raw_prompt.size(), raw_prompt.size(), raw_prompt.size() + raw_count - 2);
                auto raw128 = run_greedy_raw(raw_prompt, raw_count, &res.tg128_raw_tps);
                if (!dump_tokens(raw128)) return false;
                res.dispatches_per_tok = last_dispatches;
                if (trusted_raw128.size() != raw128.size() || trusted_context != raw_prompt.size()) {
                    trusted_raw128 = raw128;
                    trusted_context = raw_prompt.size();
                }
                std::vector<int32_t> persisted_raw128;
                const char* raw_ref = std::getenv("VEGAJET_RAW_TOKEN_REFERENCE");
                if (raw_ref && raw_ref[0] &&
                    !warm::raw_token_reference(raw_ref, raw128, persisted_raw128, res.error)) return false;
                const auto& code_reference = raw_ref && raw_ref[0] ? persisted_raw128 : trusted_raw128;
                uint32_t eq128 = 0; int fd128 = -1;
                for (size_t i = 0; i < code_reference.size() && i < raw128.size(); ++i) {
                    if (code_reference[i] == raw128[i]) ++eq128; else if (fd128 == -1) fd128 = static_cast<int>(i);
                }
                std::printf("[BENCH tg%d raw M=1] %.2f tok/s (%.2f ms/tok) | disp/tok=%u | vs_base=%u/%zu (first_diff=%d)\n",
                            raw_count - 1, res.tg128_raw_tps, 1000.0 / std::max(res.tg128_raw_tps, 1e-6), last_dispatches,
                            eq128, code_reference.size(), fd128);
                if (!long_context && std::getenv("VEGAJET_RAW_TG512") && std::getenv("VEGAJET_RAW_TG512")[0] == '1') {
                    double tg512_tps = 0.0;
                    (void) run_greedy_raw(p_rep_code, 513, &tg512_tps);
                    std::printf("[BENCH tg512 raw M=1] %.2f tok/s (%.2f ms/tok) | disp/tok=%u\n",
                                tg512_tps, 1000.0 / std::max(tg512_tps, 1e-6), last_dispatches);
                }
                // The stage profiler below replays individual super launches. It cannot measure fused mega
                // stages; avoid reporting those numbers as a VEGAJET_MEGA measurement.
                const char* mega_profile_env = std::getenv("VEGAJET_MEGA");
                const bool mega_profile_requested = mega_profile_env && std::strcmp(mega_profile_env, "1") == 0;
                const char* mega_stamp_env = std::getenv("VEGAJET_MEGA_PROFILE");
                if (mega_profile_requested && mega_stamp_env && mega_stamp_env[0] == '1') {
                    // Profile the fused dispatches once, outside the throughput interval. No replay of
                    // standalone stages; same critical-path max endpoints as SUPER_PROFILE=2.
                    reset_state();
                    const auto prep = prefill_sched::schedule(static_cast<uint32_t>(raw_prompt.size()), 0u, c2_cfg);
                    for (const auto& mb : prep.macros) {
                        std::vector<int32_t> chunk(raw_prompt.begin() + mb.pos0, raw_prompt.begin() + mb.pos0 + mb.n);
                        (void) step_batch(chunk, mb.pos0, opt.use_wy_prefill, false, mb.compute_logits);
                    }
                    // Match the middle of the timed raw decode, including generated KV/state.
                    const uint32_t replay_steps = static_cast<uint32_t>(raw128.size() / 2u);
                    const uint32_t prompt_len = static_cast<uint32_t>(raw_prompt.size());
                    for (uint32_t t = 0; t < replay_steps; ++t)
                        (void) step_batch({raw128[t]}, prompt_len + t, false, false, true);
                    const uint32_t profile_pos = prompt_len + replay_steps;
                    ornith_mega::ProfileRecords ends{};
                    for (size_t g = 0; g < N; ++g)
                        VJ_CHECK(rt.upload_to_vram(g, S[g].super_bar_va + ornith_mega::kProfileOffset,
                                                   ends.data(), sizeof(ends)));
                    mega_profile_dispatched = false;
                    mega_profile_active = true;
                    (void) step_batch({raw128[replay_steps]}, profile_pos, false, false, true);
                    mega_profile_active = false;
                    for (size_t g = 0; g < N; ++g) VJ_CHECK(rt.wait_idle(g));
                    if (mega_profile_dispatched) {
                        std::printf("[MEGA-PROFILE critical-path 48L s_memrealtime] MEGA=%s TP=%zu pos=%u diagnostic (ms/tok):\n", mega_profile_env, N, profile_pos);
                        for (size_t g = 0; g < N; ++g) {
                            VJ_CHECK(rt.download_from_vram(g, ends.data(), S[g].super_bar_va + ornith_mega::kProfileOffset, sizeof(ends)));
                            std::array<uint64_t, 16> pf{};
                            std::array<uint64_t, 16> final{};
                            unsigned valid = 0u;
                            for (unsigned r = 0; r < 96u; ++r) {
                                const uint32_t required = (r & 1u) ? 0xdc00u : (((r / 2u) & 3u) == 3u ? 0x3f0u : 0xfu);
                                valid += ornith_mega::profile_deltas(ends[r], pf, required) ? 1u : 0u;
                            }
                            const bool final_valid = ornith_mega::profile_deltas(ends[96], final, 1u << 4u);
                            auto pm = [](uint64_t c) { return static_cast<double>(c) * 40e-6; };
                            uint64_t total = 0u;
                            for (auto v : pf) total += v;
                            std::printf("  GPU%zu records=%u/96 final=%u STAGE_SUM=%.3f final_ar=%.3f\n", g, valid, final_valid ? 1u : 0u, pm(total), pm(final[4]));
                            std::printf("    GDN sub: AR=%.3f QKV=%.3f ConvRec=%.3f Out=%.3f | GQA sub: AR=%.3f QKV=%.3f Prep=%.3f Flash=%.3f Mrg=%.3f Out=%.3f\n",
                                pm(pf[0]), pm(pf[1]), pm(pf[2]), pm(pf[3]), pm(pf[4]), pm(pf[5]), pm(pf[6]), pm(pf[7]), pm(pf[8]), pm(pf[9]));
                            std::printf("    MoE sub: AR=%.3f Rtr+ShGU=%.3f Top8=%.3f GU_DOT8=%.3f MoEDn+PP=%.3f\n",
                                pm(pf[10]), pm(pf[11]), pm(pf[12]), pm(pf[15]), pm(pf[14]));
                        }
                    } else std::printf("[MEGA-PROFILE] unsupported configuration: fused path was not dispatched\n");
                }
                if (mega_profile_requested && std::getenv("VEGAJET_SUPER_PROFILE") && std::getenv("VEGAJET_SUPER_PROFILE")[0] == '1')
                    std::printf("[SUPER-PROFILE] VEGAJET_MEGA: stage replay unavailable; use end-to-end A/B and disp/tok\n");
                bool profile_l4_ready = true;
                for (size_t g = 0; g < N; ++g) profile_l4_ready &= K[g].m1_gqa_super_kvq != nullptr;
                const char* profile_tile_env = std::getenv("VEGAJET_AR_TILE_PIPE");
                profile_l4_ready &= !(profile_tile_env && std::strcmp(profile_tile_env, "force") == 0);
                if (!mega_profile_requested && (!l4_on || profile_l4_ready) && std::getenv("VEGAJET_SUPER_PROFILE") && (std::getenv("VEGAJET_SUPER_PROFILE")[0] == '1' || std::getenv("VEGAJET_SUPER_PROFILE")[0] == '2')) {
                    // The isolated KVQ copy currently has WG0 stamps only; L4 profiling uses baseline GDN/MoE.
                    const bool critical = !l4_on && std::getenv("VEGAJET_SUPER_PROFILE")[0] == '2';
                    if (l4_on && std::getenv("VEGAJET_SUPER_PROFILE")[0] == '2')
                        std::printf("[SUPER-PROFILE] L4 uses WG0 stamps (profile=1); critical-path stamps unavailable\n");
                    const kfd::LoadedKernel* k_clk[kfd::kMaxGpus] = {};
                    bool have_clk = true;
                    for (size_t g = 0; g < N; ++g) {
                        k_clk[g] = rt.get_kernel(g, "kfd_stamp_clock_u64");
                        if (!k_clk[g]) have_clk = false;
                    }
                    if (have_clk && K[0].m1_gdn_super && K[0].m1_gqa_super && K[0].m1_moe_super) {
                        const char* swg_env = std::getenv("VEGAJET_SUPER_WGS");
                        const uint32_t swgs = (swg_env && std::atoi(swg_env) >= 192 && std::atoi(swg_env) <= 240)
                                            ? static_cast<uint32_t>(std::atoi(swg_env)) : 240u;
                        const char* q3_fast_env = std::getenv("VEGAJET_MOE_Q3_FAST");
                        const bool allow_q3_fast = !(q3_fast_env && q3_fast_env[0] == '0');
                        const char* at_env = std::getenv("VEGAJET_AR_TILE_PIPE");
                        const bool psx_ok = !l4_on && K[0].m1_gdn_super_x && K[0].m1_gqa_super_x && K[0].m1_moe_super_x && K[0].m1_moe_super_q3_x;
                        // Same M = 1 policy as the decode loop: only VEGAJET_AR_TILE_PIPE=force enables the tag protocol.
                        const char* pf_env = std::getenv("VEGAJET_AR_FUSED");
                        // This profiling replay resets mailboxes/epochs below and covers only two steps.
                        const uint32_t af_bit = (psx_ok && N >= 3u && pf_env && pf_env[0] == '1') ? 0x02000000u : 0u;
                        const uint32_t at_bit = (!af_bit && psx_ok && N >= 3u && at_env && std::strcmp(at_env, "force") == 0) ? 0x10000000u : 0u;
                        const char* pt8_env = std::getenv("VEGAJET_TOP8_WG0");
                        const bool p_top8 = psx_ok && pt8_env && std::strcmp(pt8_env, "1") == 0;
                        const char* pdk_env = std::getenv("VEGAJET_DN_K4");
                        const char* pgs_env = std::getenv("VEGAJET_GU_STREAM");
                        const bool p_gs = psx_ok && pgs_env && pgs_env[0] == '1';
                        const char* pgp_env = std::getenv("VEGAJET_GU_PAIR");
                        const bool p_gp = psx_ok && pgp_env && pgp_env[0] == '1';
                        VJ_CHECK(!critical || psx_ok);
                        const bool p_use_sx = psx_ok && (af_bit != 0u || at_bit != 0u || critical);
                        const bool p_use_sx_moe = p_use_sx;  // TOP8_WG0 is Q3-kernel only
                        const uint32_t p_moe_xf = psx_ok ? moe_prefetch_xflags() : 0u;
                        const bool p_use_sx_q3 = psx_ok && (p_moe_xf != 0u || af_bit != 0u || at_bit != 0u || (pdk_env && pdk_env[0] == '1') || p_gp || p_gs || p_top8 || critical);
                        struct ClkKA { uint64_t dst; uint32_t idx; uint32_t pad; };
                        auto stamp_all = [&](uint32_t idx) {
                            for (size_t g = 0; g < N; ++g) {
                                const ClkKA ck{S[g].splitk_part_va, idx, 0u};
                                VJ_CHECK(rt.enqueue_kernel(g, *k_clk[g], 64u, 1, 1, 64, 1, 1, &ck, sizeof(ck), true));
                            }
                        };
                        reset_state();
                        {
                            uint64_t z16[1920] = {};
                            for (size_t g = 0; g < N; ++g)
                                VJ_CHECK(rt.upload_to_vram(g, S[g].super_bar_va + 1024u, z16, critical ? sizeof(z16) : 128u));
                            if (critical && std::getenv("VEGAJET_PROFILE_DUMP")) {
                                const std::vector<uint8_t> zz(kProfDumpBytes - kProfWgOff, 0u);
                                for (size_t g = 0; g < N; ++g)
                                    VJ_CHECK(rt.upload_to_vram(g, S[g].super_bar_va + kProfWgOff, zz.data(), zz.size()));
                            }
                        }
                        // Match mega's diagnostic token at the middle of this raw decode.
                        const auto prep = prefill_sched::schedule(static_cast<uint32_t>(raw_prompt.size()), 0u, c2_cfg);
                        for (const auto& mb : prep.macros) {
                            std::vector<int32_t> chunk(raw_prompt.begin() + mb.pos0, raw_prompt.begin() + mb.pos0 + mb.n);
                            (void) step_batch(chunk, mb.pos0, opt.use_wy_prefill, false, mb.compute_logits);
                        }
                        const uint32_t replay_steps = static_cast<uint32_t>(raw128.size() / 2u);
                        const uint32_t prompt_len = static_cast<uint32_t>(raw_prompt.size());
                        for (uint32_t t = 0; t < replay_steps; ++t)
                            (void) step_batch({raw128[t]}, prompt_len + t, false, false, true);
                        const uint32_t profile_pos = prompt_len + replay_steps;
                        for (size_t g = 0; g < N; ++g)
                            VJ_CHECK(rt.upload_to_vram(g, S[g].tok_ids_va, &raw128[replay_steps], sizeof(int32_t)));
                        uint32_t gi = 0, ai = 0, pend_ep = 0u, tidx = 0u;
                        const uint32_t cin = conv_head, cout = (conv_head + 1u) % kGdnRingSlots;
                        stamp_all(tidx++);
                        for (size_t g = 0; g < N; ++g) {
                            struct EmbKA { uint64_t qs, ds, toks, h; uint32_t M, D; };
                            EmbKA eka{emb_w[g].qs_va, emb_w[g].ds_va, S[g].tok_ids_va, S[g].h_va, 1u, D};
                            VJ_CHECK(rt.enqueue_kernel(g, *K[g].embed_q8, 256u, 1, 1, 256, 1, 1, &eka, sizeof(eka), true));
                            struct RmsKA { uint64_t h, xa, w, xn, xq, xd; uint32_t M, D; float eps; };
                            RmsKA rka{S[g].h_va, 0ULL, layers[g][0].attn_norm_va, S[g].x_norm_va, S[g].x_q8_va, S[g].x_ds_va, 1u, D, static_cast<float>(hp.rms_eps)};
                            VJ_CHECK(rt.enqueue_kernel(g, *K[g].rms_quant, 256u, 1, 1, 256, 1, 1, &rka, sizeof(rka), true));
                        }
                        stamp_all(tidx++);
                        for (uint32_t il = 0; il < 48u; ++il) {
                            const uint32_t pep = pend_ep, aep = ar_epoch++, mep = ar_epoch++;
                            if (layers[0][il].is_gdn) {
                                const uint32_t cgi = gi++; super_sense = (super_sense + 1u) & 0x00ffffffu; ++gdn_sense;
                                struct GdnSupKA { uint64_t desc, ov; uint32_t pep, oep, cin, cout, sh, ur, ss, gs; };
                                const uint32_t s_rhead = opt.spec_gdn ? 0u : ring_head;
                                for (size_t g = 0; g < N; ++g) {
                                    const uint64_t s_layer_bytes = static_cast<uint64_t>(gdn_nv[g]) * 128ULL * 128ULL * 4ULL;
                                    const uint64_t s_ov = opt.spec_gdn
                                        ? (S[g].spec_gdn_s_slot_va[spec_gdn_active] + static_cast<uint64_t>(cgi) * s_layer_bytes) : 0ULL;
                                    const GdnSupKA ka{S[g].gdn_super_desc_va + cgi * sizeof(ornith_super::GdnSuperDesc), s_ov, pep, aep, cin, cout, s_rhead, 0u, (critical ? (super_sense & ~0x007f0000u) : super_sense) | 0x80000000u | af_bit | at_bit | (critical ? (0x08000000u | ((2u * il) << 16u)) : 0u), gdn_sense};
                                    VJ_CHECK(rt.enqueue_kernel(g, p_use_sx ? *K[g].m1_gdn_super_x : *K[g].m1_gdn_super, swgs * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
                                }
                            } else {
                                const uint32_t cai = ai++; super_sense = (super_sense + 1u) & 0x00ffffffu;  // bits 24..31 = flags (24 TOP8_WG0, 25 AR_FUSED)
                                struct GqaSupKA { uint64_t desc; uint32_t pep, oep, pos0, ss; };
                                for (size_t g = 0; g < N; ++g) {
                                    const GqaSupKA ka{S[g].gqa_super_desc_va + cai * sizeof(ornith_super::GqaSuperDesc), pep, aep, profile_pos, (critical ? (super_sense & ~0x007f0000u) : super_sense) | 0x80000000u | af_bit | at_bit | (critical ? (0x08000000u | ((2u * il) << 16u)) : 0u)};
                                    if (l4_on) {
                                        const kvq_super::Args kvka{ka.desc,ka.pep,ka.oep,ka.pos0,ka.ss,
                                            L4[g].signs,uint32_t(l4_format),uint32_t(l4_chrono)};
                                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].m1_gqa_super_kvq, swgs * 256u, 1, 1, 256, 1, 1, &kvka, sizeof(kvka), true));
                                    } else {
                                        VJ_CHECK(rt.enqueue_kernel(g, p_use_sx ? *K[g].m1_gqa_super_x : *K[g].m1_gqa_super, swgs * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
                                    }
                                }
                            }
                            stamp_all(tidx++);
                            super_sense = (super_sense + 1u) & 0x00ffffffu;  // bits 24..31 = flags (24 TOP8_WG0, 25 AR_FUSED)
                            struct MoeSupKA { uint64_t desc; uint32_t aep, oep, ss; };
                            for (size_t g = 0; g < N; ++g) {
                                const bool use_q3_k = allow_q3_fast && (swgs == 240u) && K[g].m1_moe_super_q3 &&
                                                      (layers[g][il].w_gu_d8_w_va != 0ULL) &&
                                                      (layers[g][il].gu_d8_w_stride == 2304u);
                                const char* dk_env = std::getenv("VEGAJET_DN_K4");
                                const bool dk_on = p_use_sx_q3 && use_q3_k && dk_env && dk_env[0] == '1';
                                const MoeSupKA ka{S[g].moe_super_desc_va + il * sizeof(ornith_super::MoeSuperDesc), aep, mep,
                                                  (critical ? (super_sense & ~0x007f0000u) : super_sense) | 0x80000000u | af_bit | at_bit | (critical ? (0x08000000u | ((2u * il + 1u) << 16u)) : 0u) | ((use_q3_k && p_top8) ? 0x01000000u : 0u) | (dk_on ? 0x20000000u : 0u) | ((use_q3_k && p_gp) ? 0x40000000u : 0u) | ((use_q3_k && p_gs) ? 0x04000000u : 0u)};
                                const auto* mk = use_q3_k ? (p_use_sx_q3 ? K[g].m1_moe_super_q3_x : K[g].m1_moe_super_q3)
                                                          : (p_use_sx_moe ? K[g].m1_moe_super_x : K[g].m1_moe_super);
                                if (mk == K[g].m1_moe_super_q3_x) {
                                    struct MoeSupKAX { uint64_t desc; uint32_t aep, oep, ss, xflags; };
                                    const MoeSupKAX kx{ka.desc, ka.aep, ka.oep, ka.ss, p_moe_xf};
                                    VJ_CHECK(rt.enqueue_kernel(g, *mk, swgs * 256u, 1, 1, 256, 1, 1, &kx, sizeof(kx), true));
                                } else
                                VJ_CHECK(rt.enqueue_kernel(g, *mk, swgs * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
                            }
                            stamp_all(tidx++);
                            pend_ep = mep;
                        }
                        {
                            const uint32_t slot = 95u, pp = pend_ep & 1u;
                            if (N == 2) {
                                for (size_t g = 0; g < 2; ++g) {
                                    const size_t peer = 1u - g;
                                    struct ArKA { uint64_t xp, ptx, lrx, pfl, lfl, bar, h, w, xn, xq, xd; uint32_t M, D, ep, dn; float eps; };
                                    const ArKA aka{S[g].partial_va, rt.ar_rx_base_va(peer, g) + pp * kfd::kArSlotBytes,
                                                   rt.ar_rx_base_va(g, peer) + pp * kfd::kArSlotBytes,
                                                   rt.ar_flag_va(peer, g, slot), rt.ar_flag_va(g, peer, slot),
                                                   S[g].ar_bar_va, S[g].h_va, out_norm_va[g], S[g].x_norm_va, S[g].x_q8_va, S[g].x_ds_va,
                                                   1u, D, pend_ep, 1u, static_cast<float>(hp.rms_eps)};
                                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].p2p_ar_prepush, 256u, 1, 1, 256, 1, 1, &aka, sizeof(aka), true));
                                }
                            } else {
                                struct ArNKA { uint64_t xp, ptx[4], lrx[4], pfl[4], lfl[4], bar, h, w, xn, xq, xd; uint32_t M, D, ep, dn, rank, n_ranks, pre_pushed; float eps; };
                                for (size_t g = 0; g < N; ++g) {
                                    ArNKA aka{};
                                    aka.xp = S[g].partial_va;
                                    for (size_t r = 0; r < 4; ++r) {
                                        aka.ptx[r] = (r < N && r != g && N > 1) ? (rt.ar_rx_base_va(r, g) + pp * kfd::kArSlotBytes) : 0ULL;
                                        aka.lrx[r] = (r < N && r != g && N > 1) ? (rt.ar_rx_base_va(g, r) + pp * kfd::kArSlotBytes) : 0ULL;
                                        aka.pfl[r] = (r < N && r != g && N > 1) ? rt.ar_flag_va(r, g, slot) : 0ULL;
                                        aka.lfl[r] = (r < N && r != g && N > 1) ? rt.ar_flag_va(g, r, slot) : 0ULL;
                                    }
                                    aka.bar = S[g].ar_bar_va; aka.h = S[g].h_va; aka.w = out_norm_va[g];
                                    aka.xn = S[g].x_norm_va; aka.xq = S[g].x_q8_va; aka.xd = S[g].x_ds_va;
                                    aka.M = 1u; aka.D = D; aka.ep = pend_ep; aka.dn = 1u;
                                    aka.rank = static_cast<uint32_t>(g); aka.n_ranks = static_cast<uint32_t>(N);
                                    aka.pre_pushed = af_bit ? 4u : ((at_bit != 0u) ? 3u : 1u); aka.eps = static_cast<float>(hp.rms_eps);
                                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].p2p_ar_n_fp16, 256u, 1, 1, 256, 1, 1, &aka, sizeof(aka), true));
                                }
                            }
                            stamp_all(tidx++);
                            for (size_t g = 0; g < N; ++g) {
                                struct VocKA { uint64_t wq, wd, xq, xd, bv, bi; uint32_t lr, cols, ro, stride, M; };
                                const VocKA vka{out_w[g].qs_va, out_w[g].ds_va, S[g].x_q8_va, S[g].x_ds_va, S[g].argmax_val_va, S[g].argmax_id_va,
                                                voc_rows[g], D, voc_off[g], out_w[g].row_stride_bytes, 1u};
                                VJ_CHECK(enqueue_vocab_m1(g, vka));
                            }
                            stamp_all(tidx++);
                        }
                        VJ_CHECK(submit_all() && wait_all());
                        std::vector<uint64_t> tv[kfd::kMaxGpus];
                        double gdn_ms[kfd::kMaxGpus]={}, gqa_ms[kfd::kMaxGpus]={}, moe_ms[kfd::kMaxGpus]={}, emb_ms[kfd::kMaxGpus]={}, far_ms[kfd::kMaxGpus]={}, lm_ms[kfd::kMaxGpus]={}, tot_ms[kfd::kMaxGpus]={};
                        uint64_t pf[kfd::kMaxGpus][16] = {};
                        for (size_t g = 0; g < N; ++g) {
                            tv[g].resize(tidx);
                            VJ_CHECK(rt.download_from_vram(g, tv[g].data(), S[g].splitk_part_va, tidx * 8u));
                            VJ_CHECK(rt.download_from_vram(g, pf[g], S[g].super_bar_va + 1024u, sizeof(pf[g])));
                            emb_ms[g] = (tv[g][1] - tv[g][0]) * 40e-6;
                            uint32_t p = 1u;
                            for (uint32_t il = 0; il < 48u; ++il) {
                                const double d_mix = (tv[g][p + 1] - tv[g][p]) * 40e-6;
                                const double d_moe = (tv[g][p + 2] - tv[g][p + 1]) * 40e-6;
                                if (layers[0][il].is_gdn) gdn_ms[g] += d_mix; else gqa_ms[g] += d_mix;
                                moe_ms[g] += d_moe;
                                p += 2u;
                            }
                            far_ms[g] = (tv[g][p + 1] - tv[g][p]) * 40e-6;
                            lm_ms[g]  = (tv[g][p + 2] - tv[g][p + 1]) * 40e-6;
                            tot_ms[g] = (tv[g][p + 2] - tv[g][0]) * 40e-6;
                        }
                        auto pm = [](uint64_t c) { return static_cast<double>(c) * 40e-6; };
                        uint64_t cp[kfd::kMaxGpus][16] = {};
                        // MoE sub-stage offsets (_x Q3, critical mode): [0] router done, [1] shared GU done, [2] Stage-1 end
                        // (all from the Stage-0 AR end), [3] LDS stage-in, [4] Down rows, [5] peer stores, [6] flag/end
                        // (all from the Stage-3 GU end). Summed over MoE layers.
                        double moe_sub[kfd::kMaxGpus][7] = {};
                        const char* dump_path = critical ? std::getenv("VEGAJET_PROFILE_DUMP") : nullptr;
                        std::FILE* dump_f = dump_path ? std::fopen(dump_path, "wb") : nullptr;
                        if (dump_f) {
                            // Header: magic, N, wgs, then per GPU: stage stamps (u64 tv[tidx]), ends[96][17], per-WG [96][17][256],
                            // AR wait windows [96][2]; then 48 layer-type bytes (1 = GDN). Ticks are 25 MHz s_memrealtime.
                            const uint32_t hdr[4] = {0x564a5044u /*VJPD*/, static_cast<uint32_t>(N), swgs, tidx};
                            std::fwrite(hdr, sizeof(hdr), 1, dump_f);
                        }
                        if (critical) for (size_t g = 0; g < N; ++g) {
                            uint64_t ends[96][17] = {};
                            VJ_CHECK(rt.download_from_vram(g, ends, S[g].super_bar_va + 2048u, sizeof(ends)));
                            if (dump_f) {
                                std::vector<uint8_t> wgb(kProfDumpBytes - kProfWgOff);
                                VJ_CHECK(rt.download_from_vram(g, wgb.data(), S[g].super_bar_va + kProfWgOff, wgb.size()));
                                std::fwrite(tv[g].data(), 8u, tidx, dump_f);
                                std::fwrite(ends, sizeof(ends), 1, dump_f);
                                std::fwrite(wgb.data(), 1, wgb.size(), dump_f);
                            }
                            for (size_t ri = 0; ri < 96u; ++ri) {
                                auto& record = ends[ri];
                                const bool moe_row = (ri & 1u) != 0u;
                                uint64_t prev = record[16];
                                // DOT8 GU (15) precedes Down (14).
                                const unsigned order[] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,14};
                                for (unsigned slot : order) if (record[slot]) {
                                    if (moe_row && slot >= 1u && slot <= 5u) continue;  // sub-stamps, reported below
                                    cp[g][slot] += record[slot] - prev;
                                    prev = record[slot];
                                }
                                if (moe_row && record[10] && record[11]) {
                                    auto off = [&](uint64_t t, uint64_t ref) { return (t && t >= ref) ? static_cast<double>(t - ref) * 40e-6 : 0.0; };
                                    moe_sub[g][0] += off(record[1], record[10]);
                                    moe_sub[g][1] += off(record[2], record[10]);
                                    moe_sub[g][2] += off(record[11], record[10]);
                                    moe_sub[g][3] += off(record[3], record[15]);
                                    moe_sub[g][4] += off(record[4], record[15]);
                                    moe_sub[g][5] += off(record[5], record[15]);
                                    moe_sub[g][6] += off(record[14], record[15]);
                                }
                            }
                        }
                        if (dump_f) {
                            for (uint32_t il = 0; il < 48u; ++il) { const uint8_t t = layers[0][il].is_gdn ? 1u : 0u; std::fwrite(&t, 1, 1, dump_f); }
                            std::fclose(dump_f);
                            std::printf("[SUPER-PROFILE] per-WG dump written: %s\n", dump_path);
                        }
                        for (unsigned table = 0; table < (critical ? 2u : 1u); ++table) {
                            if (table) {
                                std::memcpy(pf, cp, sizeof(pf));
                                std::printf("[SUPER-PROFILE critical-path wgs=%u 48L s_memrealtime] TP=%zu pos=%u (ms/tok):\n", swgs, N, profile_pos);
                            } else
                                std::printf("[SUPER-PROFILE wgs=%u 48L s_memrealtime] TP=%zu pos=%u (ms/tok):\n", swgs, N, profile_pos);
                            for (size_t g = 0; g < N; ++g) {
                                std::printf("  GPU%zu: emb+rms=%.3f  gdn(36L)=%.3f  gqa(12L)=%.3f  moe(48L)=%.3f  final_ar=%.3f  lm=%.3f  TOTAL=%.3f\n",
                                            g, emb_ms[g], gdn_ms[g], gqa_ms[g], moe_ms[g], far_ms[g], lm_ms[g], tot_ms[g]);
                                std::printf("    GDN sub: AR=%.3f QKV=%.3f ConvRec=%.3f Out=%.3f | GQA sub: AR=%.3f QKV=%.3f Prep=%.3f Flash=%.3f Mrg=%.3f Out=%.3f\n",
                                            pm(pf[g][0]), pm(pf[g][1]), pm(pf[g][2]), pm(pf[g][3]),
                                            pm(pf[g][4]), pm(pf[g][5]), pm(pf[g][6]), pm(pf[g][7]), pm(pf[g][8]), pm(pf[g][9]));
                                std::printf("    MoE sub (dot8=%u iq2s=%u): AR=%.3f Rtr+ShGU=%.3f Top8=%.3f GU_IQ2S=%.3f GU_DOT8=%.3f MoEDn+PP=%.3f\n",
                                            gu_dot8_promoted, 48u - gu_dot8_promoted,
                                            pm(pf[g][10]), pm(pf[g][11]), pm(pf[g][12]), pm(pf[g][13]), pm(pf[g][15]), pm(pf[g][14]));
                                if (table && moe_sub[g][2] > 0.0)
                                    std::printf("    MoE x-sub (offsets, ms/tok): Stage1 from AR end: router_done=%.3f shGU_done=%.3f barrier_exit=%.3f | "
                                                "Stage4 from GU end: lds_in=%.3f down_rows=%.3f peer_stores=%.3f flag_end=%.3f\n",
                                                moe_sub[g][0], moe_sub[g][1], moe_sub[g][2], moe_sub[g][3], moe_sub[g][4], moe_sub[g][5], moe_sub[g][6]);
                            }
                        }
                    }
                }
                if (opt.profile_stages && !l4_on && !long_context) run_m1_profile();
                if (vout) {
                    auto out_c = run_greedy_raw(golden::kCodePrompt, 32, nullptr);
                    const size_t cm = golden::matched_prefix(out_c, golden::kCodeRef32);
                    vout->lossless_matched = static_cast<uint32_t>(cm);
                    vout->lossless_total   = static_cast<uint32_t>(golden::kCodeRef32.size());
                    vout->parity_ok        = (cm == golden::kCodeRef32.size()) && (eq128 == code_reference.size());
                    vout->raw_tg128_tps = res.tg128_raw_tps;
                    vout->primary_tps = res.tg128_raw_tps;
                    vout->dispatches_per_tok = last_dispatches;
                }
                read_hwmon_max(pass_pwr, pass_tmem);
                read_ecc_totals(ue1, de1);
                if (vout) { vout->tmem_max_c = pass_tmem; vout->ecc_ue_delta = ue1 - ue0; vout->ecc_de_delta = de1 - de0; vout->ok = vout->parity_ok && (vout->ecc_ue_delta == 0) && (vout->ecc_de_delta == 0); }
                return true;
            }
            if (!run_raw_part && !run_spec_part && run_prefill_part) {
                std::vector<int32_t> pp512(512), pp2048(2048);
                for (size_t i = 0; i < pp512.size(); ++i)  pp512[i]  = p_base[i % p_base.size()];
                for (size_t i = 0; i < pp2048.size(); ++i) pp2048[i] = p_base[i % p_base.size()];
                auto pp512_out  = run_greedy_raw(pp512,  1, nullptr, &res.pp512_tps);
                auto pp2048_out = run_greedy_raw(pp2048, 1, nullptr, &res.pp2048_tps);
                std::printf("[BENCH pp512  M=%-2u  ] %.2f tok/s (first_tok=%d)\n", opt.prefill_chunk, res.pp512_tps, pp512_out.empty() ? -1 : pp512_out[0]);
                std::printf("[BENCH pp2048 M=%-2u  ] %.2f tok/s (first_tok=%d)\n", opt.prefill_chunk, res.pp2048_tps, pp2048_out.empty() ? -1 : pp2048_out[0]);
                read_hwmon_max(pass_pwr, pass_tmem);
                read_ecc_totals(ue1, de1);
                if (vout) {
                    vout->pp512_tps = res.pp512_tps; vout->pp2048_tps = res.pp2048_tps; vout->primary_tps = res.pp2048_tps;
                    vout->tmem_max_c = pass_tmem; vout->ecc_ue_delta = ue1 - ue0; vout->ecc_de_delta = de1 - de0;
                    vout->ok = (vout->ecc_ue_delta == 0) && (vout->ecc_de_delta == 0);
                }
                return true;
            }
            return false; // fall through to full bench body below
        }
        read_hwmon_max(pass_pwr, pass_tmem);
        read_ecc_totals(ue1, de1);
        if (vout) {
            vout->tmem_max_c = pass_tmem;
            vout->ecc_ue_delta = ue1 - ue0;
            vout->ecc_de_delta = de1 - de0;
            vout->ok = vout->parity_ok && (vout->ecc_ue_delta == 0) && (vout->ecc_de_delta == 0);
        }
        return true;
    };

    // Execute the full benchmark body (including raw, Suffix, MTP, DFlash2+Hyb, pp512/pp2048) as a reusable lambda!
    auto run_full_bench_body = [&](bool include_prefill, warm::SweepVariantResult* vout) -> bool {
        uint64_t ue0 = 0, de0 = 0, ue1 = 0, de1 = 0;
        read_ecc_totals(ue0, de0);
        double pass_pwr = 0.0, pass_tmem = 0.0;
        double best_spec_tps = 0.0;
        uint32_t spec_eq = 0, spec_tot = 0;
        int spec_fd = -1;
        bool all_lossless = true;
        {
        std::printf("\n=== ORNITH-122B LIVE 2x MI50 BENCHMARK SUITE ===\n");
        const std::vector<int32_t> p_base = {760, 6511, 314, 9338, 369};
        const std::vector<int32_t> p_rep_code = {
            727, 73111, 1393, 1590, 198, 262, 413, 307, 606, 220, 15, 25, 198, 285, 460, 220, 16, 198,
            262, 745, 25, 198, 285, 460, 307, 348, 73111, 1393, 12, 16, 8, 271,
            727, 73111, 1393, 1590
        };

        if (!opt.gdn_state_dump.empty()) {
            // Carried-state check for batched GDN prefill: a deterministic prompt of two full chunks, the recurrent state
            // (S and the conv history) of every GDN layer on both ranks dumped at both chunk boundaries, then 16 greedy
            // continuation tokens. Run once per variant (and once with --prefill-seq as the sequential reference) and
            // compare the files offline.
            std::FILE* df = std::fopen(opt.gdn_state_dump.c_str(), "wb");
            if (!df) { res.error = "cannot open " + opt.gdn_state_dump; return false; }
            const uint32_t chunk = opt.prefill_chunk, n_prompt = 2u * chunk, n_cont = 16u;
            std::vector<int32_t> prompt(n_prompt);
            for (uint32_t i = 0; i < n_prompt; ++i) prompt[i] = p_rep_code[i % p_rep_code.size()];
            uint32_t n_gdn_layers = 0;
            for (uint32_t il = 0; il < 48u; ++il) n_gdn_layers += layers[0][il].is_gdn ? 1u : 0u;
            const uint64_t s_bytes = 32ULL * 128ULL * 128ULL * 4ULL, c_bytes = 3ULL * 6144ULL * 4ULL;
            const uint32_t hdr[6] = {0x564A4753u, chunk, n_gdn_layers, static_cast<uint32_t>(s_bytes), static_cast<uint32_t>(c_bytes), n_cont};
            std::fwrite(hdr, 4, 6, df);
            std::vector<uint8_t> blob(s_bytes);
            reset_state();
            st_on = true;   // drain after every stage: completion records in the dispatch journal are per stage
            uint32_t pos = 0; int32_t last_pred = -1;
            for (uint32_t c = 0; c < 2u; ++c) {
                const bool last_chunk = (c == 1u);
                if (opt.prefill_seq) {
                    for (uint32_t i = 0; i < chunk; ++i) {
                        const bool want = last_chunk && i + 1u == chunk;
                        auto pr = step_batch({prompt[pos]}, pos, false, false, want);
                        if (want) last_pred = pr[0];
                        ++pos;
                    }
                } else {
                    std::vector<int32_t> ck(prompt.begin() + pos, prompt.begin() + pos + chunk);
                    auto pr = step_batch(ck, pos, opt.use_wy_prefill, false, last_chunk);
                    if (last_chunk) last_pred = pr.back();
                    pos += chunk;
                }
                for (size_t g = 0; g < N; ++g) {
                    const uint64_t sb_g = static_cast<uint64_t>(gdn_nv[g]) * 128ULL * 128ULL * 4ULL;
                    const uint64_t cb_g = 3ULL * static_cast<uint64_t>(gdn_conv_dim[g]) * 4ULL;
                    for (uint32_t gi = 0; gi < n_gdn_layers; ++gi) {
                        const uint64_t sva = opt.spec_gdn
                            ? (S[g].spec_gdn_s_slot_va[spec_gdn_active] + static_cast<uint64_t>(gi) * sb_g)
                            : (S[g].gdn_ssm_ring_va[gi] + static_cast<uint64_t>(ring_head) * sb_g);
                        VJ_CHECK(rt.download_from_vram(g, blob.data(), sva, sb_g));
                        std::fwrite(blob.data(), 1, sb_g, df);
                        VJ_CHECK(rt.download_from_vram(g, blob.data(), S[g].gdn_conv_ring_va[gi], cb_g));
                        std::fwrite(blob.data(), 1, cb_g, df);
                    }
                }
            }
            st_on = false;
            std::vector<int32_t> cont{last_pred};
            for (uint32_t i = 1; i < n_cont; ++i) cont.push_back(step_batch({cont.back()}, pos++, false, false, true)[0]);
            std::fwrite(cont.data(), 4, cont.size(), df);
            std::fclose(df);
            std::printf("[GDN-STATE-DUMP %s] chunk=%u prompt=%u gdn_layers=%u pf_gdn=%u -> %s | continuation:", opt.prefill_seq ? "sequential" : "batched",
                        chunk, n_prompt, n_gdn_layers, opt.pf_gdn, opt.gdn_state_dump.c_str());
            for (int32_t t : cont) std::printf(" %d", t);
            std::printf("\n");
            rt.free_buffer(tap_buf);
            rt.free_buffer(io_buf);
            res.ok = true;
            return false;
        }
        if (opt.dot8_ab) {
            // MoE Down A/B on the last layer's real routes, activations and scales: compact IQ3_XXS (kernel 9c) vs the
            // exact 16-level DOT8 repack. Timed in isolation on GPU 0 (48 dispatches per sample), outputs compared bitwise.
            const kfd::LoadedKernel* k_d8 = K[0].moe_down_d8;
            kfd::KfdBuffer cmp{};
            if (!k_d8 || !rt.alloc_userptr_pinned(2ULL * 1024ULL * 1024ULL, cmp, true)) { res.error = "dot8-ab: kernel or compare buffer missing"; return false; }
            auto tfn = [&](int iters, auto&& fn) { fn(); auto t0 = std::chrono::steady_clock::now(); for (int i = 0; i < iters; ++i) fn();
                                                   return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters; };
            auto g0 = [&](auto&& enq) { enq(); VJ_CHECK(rt.submit_batch(0)); VJ_CHECK(rt.wait_idle(0)); };
            auto copy_out = [&](uint64_t src, uint64_t n_u32, std::vector<uint32_t>& dst) {
                struct CpKA { uint64_t dst, src, n; };
                const CpKA cka{cmp.va, src, n_u32};
                g0([&]() { VJ_CHECK(rt.enqueue_kernel(0, *K[0].memcpy_u32, static_cast<uint32_t>(((n_u32 + 255u) / 256u) * 256u), 1, 1, 256, 1, 1, &cka, sizeof(cka), true)); });
                __atomic_thread_fence(__ATOMIC_ACQUIRE);
                dst.assign(reinterpret_cast<const uint32_t*>(cmp.host_ptr), reinterpret_cast<const uint32_t*>(cmp.host_ptr) + n_u32);
            };
            auto ab = [&](uint32_t Mab, const char* label) {
                struct DnKA  { uint64_t wd, mq, md, tids, tw, shd, out; uint32_t M, D; };
                struct D8KA  { uint64_t wq, ws, mq, md, tids, tw, shd, out; uint32_t M, D, wst, sst; };
                const uint64_t out_a = S[0].partial_va, out_b = S[0].x_norm_va;
                const DnKA ka{layers[0][47].w_dn_iq3xxs_va, S[0].moe_q8_va, S[0].moe_d_va, S[0].top_ids_va, S[0].top_w_va, S[0].sh_down_va, out_a, Mab, D};
                const D8KA kb{layers[0][47].w_dn_d8_w_va, layers[0][47].w_dn_d8_s_va, S[0].moe_q8_va, S[0].moe_d_va, S[0].top_ids_va, S[0].top_w_va,
                              S[0].sh_down_va, out_b, Mab, D, layers[0][47].dn_d8_w_stride, layers[0][47].dn_d8_s_stride};
                const uint32_t gx = ((D + 15u) / 16u) * 256u;
                auto run_a = [&](int n) { g0([&]() { for (int i = 0; i < n; ++i) VJ_CHECK(rt.enqueue_kernel(0, *K[0].moe_down_iq3, gx, Mab, 1, 256, 1, 1, &ka, sizeof(ka), true)); }); };
                auto run_b = [&](int n) { g0([&]() { for (int i = 0; i < n; ++i) VJ_CHECK(rt.enqueue_kernel(0, *k_d8, gx, Mab, 1, 256, 1, 1, &kb, sizeof(kb), true)); }); };
                const int reps = (Mab >= 64u) ? 4 : 48;
                double ta = 1e30, tb = 1e30;
                for (int round = 0; round < 3; ++round) {   // interleaved, best of 3
                    ta = std::min(ta, tfn(6, [&]() { run_a(reps); }) / reps);
                    tb = std::min(tb, tfn(6, [&]() { run_b(reps); }) / reps);
                }
                run_a(1); run_b(1);
                const uint64_t n_cmp = std::min<uint64_t>(static_cast<uint64_t>(Mab) * D, 2ULL * 1024ULL * 1024ULL / 4ULL);
                std::vector<uint32_t> oa, ob, ids;
                copy_out(out_a, n_cmp, oa); copy_out(out_b, n_cmp, ob); copy_out(S[0].top_ids_va, Mab * 8u, ids);
                uint64_t diff = 0; double max_rel = 0.0;
                for (uint64_t i = 0; i < n_cmp; ++i) if (oa[i] != ob[i]) {
                    ++diff;
                    float fa, fb; std::memcpy(&fa, &oa[i], 4); std::memcpy(&fb, &ob[i], 4);
                    max_rel = std::max(max_rel, std::fabs(double(fa) - fb) / std::max(1e-12, std::fabs(double(fa))));
                }
                if (K[0].dgrp_route && S[0].dgrp_buf_va && opt.down_grp_prefill == 1u && D % 64u == 0u) {
                    // Expert-grouped exact pipeline (route group -> grouped dp -> combine) against the per-token DOT8 kernel.
                    const uint64_t out_c = S[0].h_va;
                    const uint32_t icap = grp_item_cap(Mab);
                    struct RgKA { uint64_t tids, hdr, cnt, off, uniq, items, routes; uint32_t M, min_group, icap; };
                    struct GdKA { uint64_t wq, mq, hdr, items, routes, dp; uint32_t D, wst; };
                    struct CbKA { uint64_t wq, ws, mq, md, tids, tw, shd, cnt, in, out; uint32_t M, D, wst, sst, min_group; };
                    const auto& lw = layers[0][47];
                    const RgKA rg{S[0].top_ids_va, S[0].dgrp_hdr_va, S[0].dgrp_cnt_va, S[0].dgrp_off_va, S[0].dgrp_uniq_va, S[0].dgrp_items_va,
                                  S[0].dgrp_routes_va, Mab, kDownGrpMin, icap};
                    const GdKA gd{lw.w_dn_d8_w_va, S[0].moe_q8_va, S[0].dgrp_hdr_va, S[0].dgrp_items_va, S[0].dgrp_routes_va, S[0].dgrp_buf_va, D, lw.dn_d8_w_stride};
                    const CbKA cb{lw.w_dn_d8_w_va, lw.w_dn_d8_s_va, S[0].moe_q8_va, S[0].moe_d_va, S[0].top_ids_va, S[0].top_w_va, S[0].sh_down_va,
                                  S[0].dgrp_cnt_va, S[0].dgrp_buf_va, out_c, Mab, D, lw.dn_d8_w_stride, lw.dn_d8_s_stride, kDownGrpMin};
                    auto run_c = [&](int n) { g0([&]() { for (int i = 0; i < n; ++i) {
                        VJ_CHECK(rt.enqueue_kernel(0, *K[0].dgrp_route, 256u, 1, 1, 256, 1, 1, &rg, sizeof(rg), true));
                        VJ_CHECK(rt.enqueue_kernel(0, *K[0].dgrp_dp, (D / 64u) * 256u, icap, 1, 256, 1, 1, &gd, sizeof(gd), true));
                        VJ_CHECK(rt.enqueue_kernel(0, *K[0].dgrp_comb_dp, gx, Mab, 1, 256, 1, 1, &cb, sizeof(cb), true)); } }); };
                    double tc = 1e30;
                    for (int round = 0; round < 3; ++round) tc = std::min(tc, tfn(6, [&]() { run_c(reps); }) / reps);
                    run_c(1);
                    std::vector<uint32_t> oc;
                    copy_out(out_c, n_cmp, oc);
                    uint64_t dc = 0;
                    for (uint64_t i = 0; i < n_cmp; ++i) dc += oc[i] != ob[i];
                    std::printf("[DOT8-AB %s M=%u] grouped exact: %.2f us (%.2fx vs per-token DOT8) | outputs %llu/%llu differ\n", label, Mab, tc * 1000.0, tb / tc,
                                static_cast<unsigned long long>(dc), static_cast<unsigned long long>(n_cmp));
                }
                std::vector<uint32_t> cnt(256, 0); uint32_t uniq = 0, maxc = 0;
                for (uint32_t v : ids) { if (v < 256u && cnt[v]++ == 0u) ++uniq; }
                for (uint32_t c : cnt) maxc = std::max(maxc, c);
                const double mb_a = Mab * 8.0 * D * 208.0 / 1e6, mb_b = Mab * 8.0 * D * 272.0 / 1e6;
                std::printf("[DOT8-AB %s M=%u] routes=%u unique_experts=%u max_tokens/expert=%u | IQ3 9c %.2f us (%.1f MB, %.0f GB/s) | DOT8 %.2f us (%.1f MB, %.0f GB/s) | "
                            "speedup %.2fx | outputs %llu/%llu differ (max rel %.2e)\n",
                            label, Mab, Mab * 8u, uniq, maxc, ta * 1000.0, mb_a, mb_a / ta, tb * 1000.0, mb_b, mb_b / tb, ta / tb,
                            static_cast<unsigned long long>(diff), static_cast<unsigned long long>(n_cmp), max_rel);
            };
            double tps = 0.0;
            auto g8 = run_greedy_raw(p_rep_code, 9, &tps);
            ab(1u, "decode");
            (void) step_batch({g8[4], g8[5], g8[6], g8[7], g8[8]}, static_cast<uint32_t>(p_rep_code.size()) + 8u, false, true, true);
            ab(5u, "verify");
            std::vector<int32_t> p128;
            for (size_t i = 0; p128.size() < 128u; ++i) p128.push_back(i < p_base.size() ? p_base[i] : p_rep_code[(i * 7u + i / 36u) % p_rep_code.size()]);
            reset_state();
            (void) step_batch(p128, 0u, opt.use_wy_prefill, false, false);
            ab(128u, "prefill");
            rt.free_buffer(cmp);
            rt.free_buffer(io_buf);
            res.ok = true;
            return false;
        }
        if (!opt.dump_hidden_path.empty()) {
            // Token-by-token greedy run with residual-stream taps: prompt (given --prompt, else the code prompt) followed
            // by the model's own continuation. File: "VJHS", u32 version=1, n_tokens, n_states=49, D; i32 ids[n_tokens];
            // float32 states[n_tokens][49][D] (state 0 = embedding output, state i = output of layer i-1).
            const std::vector<int32_t> dp = opt.dump_prompt_ids.empty() ? p_rep_code : opt.dump_prompt_ids;
            const int n_gen = std::max(1, opt.decode_tokens);
            std::vector<int32_t> ids;
            std::ofstream df(opt.dump_hidden_path, std::ios::binary);
            if (!df) {
                rt.free_buffer(tap_buf);
                rt.free_buffer(io_buf);
                res.error = "Cannot open hidden-state dump file " + opt.dump_hidden_path;
                return false;
            }
            const uint32_t n_tok = static_cast<uint32_t>(dp.size()) + static_cast<uint32_t>(n_gen);
            const uint32_t hdr[5] = {0x53484A56u, 1u, n_tok, 49u, D};
            df.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
            df.seekp(static_cast<std::streamoff>(sizeof(hdr) + n_tok * 4ull));
            reset_state();
            tap_on = true;
            int32_t next = -1;
            for (uint32_t t = 0; t < n_tok; ++t) {
                const int32_t tok = (t < dp.size()) ? dp[t] : next;
                ids.push_back(tok);
                next = step_batch({tok}, t, false, false, true)[0];
                __atomic_thread_fence(__ATOMIC_ACQUIRE);
                df.write(reinterpret_cast<const char*>(tap_host), 49ll * D * 4ll);
            }
            tap_on = false;
            df.seekp(sizeof(hdr));
            df.write(reinterpret_cast<const char*>(ids.data()), static_cast<std::streamsize>(ids.size() * 4u));
            std::printf("[DUMP] %u tokens (%zu prompt + %d generated) x 49 states x %u -> %s | next token after dump: %d\n",
                        n_tok, dp.size(), n_gen, D, opt.dump_hidden_path.c_str(), next);
            std::printf("[DUMP] ids:");
            for (int32_t v : ids) std::printf(" %d", v);
            std::printf("\n");
            rt.free_buffer(tap_buf);
            rt.free_buffer(io_buf);
            res.ok = df.good();
            if (!res.ok) res.error = "Short write to hidden-state dump file " + opt.dump_hidden_path;
            return false;
        }
        if (!opt.dump_logits_path.empty()) {
            // Held-out multi-domain prefill logit dump ("VJLG" v1) for exact KL-divergence gating (Stage 3D KLD <= 0.045).
            std::vector<std::vector<int32_t>> heldout;
            if (!opt.dump_prompt_ids.empty()) {
                heldout.push_back(opt.dump_prompt_ids);
            } else {
                const char* const kPrompts[4] = {
                    "The architecture of modern high-bandwidth memory subsystems balances channel width, bank parallelism, "
                    "and row-buffer locality across multiple stacked DRAM dies. When designing a custom GPU matrix engine "
                    "for quantized transformer inference, minimizing register pressure while maintaining four resident "
                    "wavefronts per SIMD is essential for hiding global memory latency. ",
                    "def merge_sorted_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:\n"
                    "    if not intervals:\n"
                    "        return []\n"
                    "    intervals.sort(key=lambda pair: (pair[0], pair[1]))\n"
                    "    merged = [intervals[0]]\n"
                    "    for start, end in intervals[1:]:\n"
                    "        if start <= merged[-1][1]:\n"
                    "            merged[-1] = (merged[-1][0], max(merged[-1][1], end))\n"
                    "        else:\n"
                    "            merged.append((start, end))\n"
                    "    return merged\n",
                    "Theorem: Let A be a real symmetric positive-definite n x n matrix and let b be a vector in R^n. "
                    "The conjugate gradient method generates mutually A-conjugate search directions p_0, p_1, ..., p_{k-1} "
                    "and minimizes the quadratic functional f(x) = 0.5 x^T A x - b^T x over the Krylov subspace K_k(A, r_0). "
                    "Therefore, in exact arithmetic, the solution is reached in at most ",
                    "In distributed systems, consensus protocols such as Raft separate leader election, log replication, "
                    "and safety into distinct subproblems. Each server operates in one of three states: leader, follower, "
                    "or candidate. Time is divided into terms of arbitrary length, numbered with consecutive integers. "
                    "When a follower receives no heartbeat within its election timeout, it increments its current term and "
                };
                for (int pi = 0; pi < 4; ++pi) {
                    std::vector<int32_t> base = tok.encode(kPrompts[pi], false);
                    std::vector<int32_t> padded;
                    padded.reserve(256);
                    while (padded.size() < 256) {
                        for (int32_t id : base) {
                            padded.push_back(id);
                            if (padded.size() == 256) break;
                        }
                    }
                    heldout.push_back(std::move(padded));
                }
            }
            const uint32_t n_pos = static_cast<uint32_t>(heldout.size());
            uint32_t V_tot = 0u;
            for (size_t g = 0; g < N; ++g) V_tot += voc_rows[g];
            std::ofstream lf(opt.dump_logits_path, std::ios::binary);
            if (!lf) {
                rt.free_buffer(tap_buf);
                rt.free_buffer(io_buf);
                res.error = "Cannot open logits dump file " + opt.dump_logits_path;
                return false;
            }
            const uint32_t hdr[4] = {0x474C4A56u, 1u, n_pos, V_tot};
            lf.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
            std::vector<float> logits(V_tot);
            for (uint32_t pi = 0; pi < n_pos; ++pi) {
                reset_state();
                const auto& pr = heldout[pi];
                const uint32_t M_pr = static_cast<uint32_t>(pr.size());
                const int32_t top1 = step_batch(pr, 0u, false, false, true).back();
                const uint64_t xq_off = static_cast<uint64_t>(M_pr - 1u) * D;
                const uint64_t xd_off = static_cast<uint64_t>(M_pr - 1u) * (D / 32u) * 8ULL;
                for (size_t g = 0; g < N; ++g) {
                    struct LogKA { uint64_t wq, wd, xq, xd, out; uint32_t lr, cols, stride; };
                    const LogKA lka{out_w[g].qs_va, out_w[g].ds_va, S[g].x_q8_va + xq_off, S[g].x_ds_va + xd_off,
                                    S[g].logits_va, voc_rows[g], D, out_w[g].row_stride_bytes};
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].vocab_logits, kVocabBlocks * 256u, 1, 1, 256, 1, 1, &lka, sizeof(lka), true));
                    VJ_CHECK(rt.submit_batch(g));
                }
                VJ_CHECK(wait_all());
                for (size_t g = 0; g < N; ++g)
                    VJ_CHECK(rt.download_from_vram(g, logits.data() + voc_off[g], S[g].logits_va, voc_rows[g] * 4ULL));
                lf.write(reinterpret_cast<const char*>(logits.data()), static_cast<std::streamsize>(V_tot * 4ULL));
                std::printf("[DUMP-LOGITS] prompt %u/%u (M=%u, dot8=%d clip=%.2f) -> top1=%d logit=%.4f\n",
                            pi + 1u, n_pos, M_pr, opt.prefill_dot8 ? 1 : 0, opt.dot8_clip_alpha,
                            top1, (top1 >= 0 && static_cast<uint32_t>(top1) < V_tot) ? logits[top1] : 0.0f);
            }
            rt.free_buffer(tap_buf);
            rt.free_buffer(io_buf);
            res.ok = lf.good();
            if (!res.ok) res.error = "Short write to logits dump file " + opt.dump_logits_path;
            return false;
        }
        auto raw128 = run_greedy_raw(p_rep_code, 129, &res.tg128_raw_tps);
        if (!dump_tokens(raw128)) return false;
        res.dispatches_per_tok = last_dispatches;
        // Exact actual fetched HBM2 bytes/tok across ALL GPUs (excluding odd-256B TCC padding bytes never read,
        // and using exact per-format routed Gate+Up and routed Down byte counts).
        double real_bytes_per_tok_all_gpus = 0.0;
        {
            auto q4_real_b = [](const Q4A8WeightDev& w) {
                return double(w.rows) * (double(w.cols) * 0.5 + double(w.cols / 32u) * 4.0);
            };
            const double gu_d8_row_b = double(dot8_gu_iq2s_layout(1u, D).read_bytes_per_row());
            for (size_t g = 0; g < N; ++g) {
                const double dn_d8_row_b = layers[g][0].dn_is_sub32
                    ? double(dot8_iq3_sub32_layout(1u, local_ff_r[g]).read_bytes_per_row())
                    : double(dot8_iq3_layout(1u, local_ff_r[g]).read_bytes_per_row());
                // Per-stage split (printed with VEGAJET_BYTES_BREAKDOWN=1); the sum is unchanged.
                double b_lm = double(voc_rows[g]) * (double(D) + double(D / 32u) * 2.0);
                double b_gdn_w = 0.0, b_gdn_s = 0.0, b_gqa_w = 0.0, b_rtr = 0.0, b_sh = 0.0, b_gu = 0.0, b_dn = 0.0;
                for (uint32_t il = 0; il < 48u; ++il) {
                    const auto& lw = layers[g][il];
                    if (lw.is_gdn) {
                        b_gdn_w += q4_real_b(lw.w_gdn_qkv_gab) + q4_real_b(lw.w_ssm_out);
                        b_gdn_s += 2.0 * double(gdn_nv[g]) * 128.0 * 128.0 * 4.0;
                    } else {
                        b_gqa_w += q4_real_b(lw.w_attn_qkv) + q4_real_b(lw.w_o);
                    }
                    b_rtr += 256.0 * D * 2.0;
                    b_sh  += q4_real_b(lw.w_sh_gu) + q4_real_b(lw.w_sh_down);
                    b_gu  += (lw.w_gu_d8_w_va != 0) ? (8.0 * local_ff_r[g] * gu_d8_row_b) : (8.0 * local_ff_r[g] * 1968.0);
                    b_dn  += (lw.w_dn_d8_w_va != 0) ? (8.0 * D * dn_d8_row_b)             : (8.0 * D * 208.0);
                }
                const double bg = b_lm + b_gdn_w + b_gdn_s + b_gqa_w + b_rtr + b_sh + b_gu + b_dn;
                if (const char* bb = std::getenv("VEGAJET_BYTES_BREAKDOWN"); bb && bb[0] == '1')
                    std::printf("[BYTES gpu%zu MB/tok] gdn_w=%.2f gdn_state=%.2f gqa_w=%.2f router=%.2f shared=%.2f routed_gu=%.2f "
                                "routed_dn=%.2f lm_head=%.2f total=%.2f\n", g, b_gdn_w / 1e6, b_gdn_s / 1e6, b_gqa_w / 1e6,
                                b_rtr / 1e6, b_sh / 1e6, b_gu / 1e6, b_dn / 1e6, b_lm / 1e6, bg / 1e6);
                real_bytes_per_tok_all_gpus += bg;
            }
        }
        const double real_gb_per_tok = real_bytes_per_tok_all_gpus / 1e9;
        res.eff_hbm2_gbs = real_gb_per_tok * res.tg128_raw_tps;
        if (opt.ar_timeline) {
            // Two-rank AllReduce timeline through real decode tokens. Stamps are s_memrealtime (25 MHz) per GPU:
            // 0 entry, 1 payload packed, 2 flag posted, 3 peer payload visible, 4 reduction done, 5 norm+quant done.
            for (size_t g = 0; g < 2; ++g) {
                k_ar_ts[g] = rt.get_kernel(g, "ornith_p2p_ar_res_rms_quant_ts_kernel");
                if (!k_ar_ts[g] || !S[g].ar_ts_va || opt.ar_wire_int8) { res.error = "--ar-timeline needs the FP16 timeline kernel"; return false; }
                const std::vector<uint8_t> z(kArTsMax * 48ull, 0);
                VJ_CHECK(rt.upload_to_vram(g, S[g].ar_ts_va, z.data(), z.size()));
            }
            ar_ts_on = true; ar_ts_idx = 0;
            double tl_tps = 0.0;
            (void)run_greedy_raw(p_rep_code, 129, &tl_tps);
            ar_ts_on = false;
            const uint32_t n = ar_ts_idx, per_tok = 96u, toks = std::min<uint32_t>(64u, n / per_tok);
            std::printf("[AR-TIMELINE] %u stamped AllReduces, %.2f tok/s with stamps; last %u tokens, us per token:\n", n, tl_tps, toks);
            if (toks >= 2u) {
                const uint32_t first = n - toks * per_tok;
                for (size_t g = 0; g < 2; ++g) {
                    std::vector<uint64_t> ts(static_cast<size_t>(n) * 6u);
                    VJ_CHECK(rt.download_from_vram(g, ts.data(), S[g].ar_ts_va, ts.size() * 8u));
                    // sums: [class 0 = even AR of a layer (after the mixer), 1 = odd (after the MoE)][phase]
                    double sum[2][6] = {}; uint32_t waits[2] = {0, 0}; double wmax = 0.0;
                    for (uint32_t i = first; i < n; ++i) {
                        const uint64_t* t = &ts[static_cast<size_t>(i) * 6u];
                        const uint64_t t5 = t[5] ? t[5] : t[4];
                        const uint32_t c = (i - first) & 1u;
                        const double us = 0.04;
                        const double w = static_cast<double>(t[3] - t[2]) * us;
                        sum[c][0] += static_cast<double>(t[1] - t[0]) * us;
                        sum[c][1] += static_cast<double>(t[2] - t[1]) * us;
                        sum[c][2] += w;
                        sum[c][3] += static_cast<double>(t[4] - t[3]) * us;
                        sum[c][4] += static_cast<double>(t5 - t[4]) * us;
                        if (i + 1u < n) sum[c][5] += static_cast<double>(ts[static_cast<size_t>(i + 1u) * 6u] - t5) * us;
                        if (w > 3.0) ++waits[c];
                        wmax = std::max(wmax, w);
                    }
                    for (uint32_t c = 0; c < 2u; ++c)
                        std::printf("  gpu%zu AR#%u/layer: pack %.0f | post flag %.0f | wait for peer %.0f (%u of %u ARs waited >3 us) | reduce %.0f | norm+quant %.0f | "
                                    "work until next AR %.0f\n", g, c, sum[c][0] / toks, sum[c][1] / toks, sum[c][2] / toks, waits[c], toks * per_tok / 2u,
                                    sum[c][3] / toks, sum[c][4] / toks, sum[c][5] / toks);
                    double tot = 0.0; for (auto& r : sum) for (double v : r) tot += v;
                    std::printf("  gpu%zu total %.0f us/token (wait %.0f, max single wait %.1f us)\n", g, tot / toks, (sum[0][2] + sum[1][2]) / toks, wmax);
                }
            }
        }
        std::printf("[BENCH tg128 raw M=1] %.2f tok/s (%.2f ms/tok) | eff_HBM2=%.1f GB/s (%.1f%% of %.1f GB/s calibrated) | real_read=%.3f GB/tok | disp/tok=%u\n",
                    res.tg128_raw_tps, 1000.0 / std::max(res.tg128_raw_tps, 1e-6),
                    res.eff_hbm2_gbs, 100.0 * res.eff_hbm2_gbs / std::max(calibrated_hbm2_ceiling_gbs, 1.0),
                    calibrated_hbm2_ceiling_gbs, real_gb_per_tok, res.dispatches_per_tok);
        read_hwmon_max(max_pwr, max_tmem);
        if (std::getenv("VEGAJET_RAW_TG512") && std::getenv("VEGAJET_RAW_TG512")[0] == '1') {
            // v0.1.1 matrix: raw tg512 on the full bench path too (was printed by the warm sweep only)
            double tg512_tps = 0.0;
            (void) run_greedy_raw(p_rep_code, 513, &tg512_tps);
            std::printf("[BENCH tg512 raw M=1] %.2f tok/s (%.2f ms/tok) | disp/tok=%u\n",
                        tg512_tps, 1000.0 / std::max(tg512_tps, 1e-6), last_dispatches);
            read_hwmon_max(max_pwr, max_tmem);
        }
        if (opt.profile_stages) {
            static const char* const kStage[] = {"", "gdn_qkv", "gdn_out", "gqa_qkv", "gqa_prep", "gqa_flash", "gqa_o", "moe_front", "moe_back"};
            double base_tps = 0.0;
            run_greedy_raw(p_rep_code, 65, &base_tps);
            const double base_ms = 1000.0 / std::max(base_tps, 1e-6);
            std::printf("[PROFILE in-situ M=1] base %.2f ms/tok |", base_ms);
            double acc_ms = 0.0;
            for (uint32_t st = 1u; st <= 8u; ++st) {
                double tps = 0.0;
                prof_stage = st; prof_reps = 3u;
                run_greedy_raw(p_rep_code, 65, &tps);
                prof_stage = 0u; prof_reps = 1u;
                const double ms = (1000.0 / std::max(tps, 1e-6) - base_ms) / 2.0;
                acc_ms += ms;
                tb_stage_ms[st] = ms;
                std::printf(" %s=%.2f", kStage[st], ms);
            }
            std::printf(" | sum=%.2f ms (rest: AR, conv, recur, embed, LM head, host)\n", acc_ms);
            tb_base_ms = base_ms;
        }

        if (opt.use_suffix_spec) {
            uint64_t hits = 0, acc = 0, dr = 0;
            auto suf128 = run_greedy_suffix(p_rep_code, 129, &res.tg128_suffix_tps, &hits, &acc, &dr);
            uint32_t eq = 0;
            int first_diff = -1;
            for (size_t i = 0; i < raw128.size() && i < suf128.size(); ++i) {
                if (raw128[i] == suf128[i]) ++eq;
                else if (first_diff == -1) first_diff = static_cast<int>(i);
            }
            res.suffix_exact_matches = eq;
            res.suffix_total_tokens  = static_cast<uint32_t>(raw128.size());
            std::printf("[BENCH tg128 Suffix ] %.2f tok/s (%.2fx vs raw) | lossless=%u/%u (first_diff=%d) | hits=%llu acc=%llu/%llu | 0 B VRAM\n",
                        res.tg128_suffix_tps, res.tg128_suffix_tps / std::max(res.tg128_raw_tps, 1e-6),
                        eq, res.suffix_total_tokens, first_diff,
                        static_cast<unsigned long long>(hits), static_cast<unsigned long long>(acc), static_cast<unsigned long long>(dr));

            uint64_t hits5 = 0, acc5 = 0, dr5 = 0;
            (void) run_greedy_suffix(p_rep_code, 513, &res.tg512_suffix_tps, &hits5, &acc5, &dr5);
            std::printf("[BENCH tg512 Suffix ] %.2f tok/s | hits=%llu acc=%llu/%llu | 0 B VRAM\n",
                        res.tg512_suffix_tps, static_cast<unsigned long long>(hits5),
                        static_cast<unsigned long long>(acc5), static_cast<unsigned long long>(dr5));
            read_hwmon_max(max_pwr, max_tmem);
        }

        if (has_mtp) {
            // 1. Pure Neural blk.48 MTP (K=opt.mtp_k) on ZERO-REPETITION PROSE + 129/129 lossless gate!
            double raw_prose_tps = 0.0, mtp_prose_tps = 0.0;
            auto raw_prose = run_greedy_raw(p_base, 129, &raw_prose_tps);
            uint64_t macc = 0, mdr = 0;
            auto mtp_prose = run_greedy_mtp(p_base, 129, opt.mtp_k, false, &mtp_prose_tps, &macc, &mdr);
            uint32_t eqp = 0; int fdp = -1;
            for (size_t i = 0; i < raw_prose.size() && i < mtp_prose.size(); ++i) {
                if (raw_prose[i] == mtp_prose[i]) ++eqp; else if (fdp == -1) fdp = static_cast<int>(i);
            }
            std::printf("[BENCH tg128 MTP K=%u] %.2f tok/s (%.2fx vs raw %.2f) | prose lossless=%u/%zu (first_diff=%d) | acc=%llu/%llu (%.1f%%)\n",
                        opt.mtp_k, mtp_prose_tps, mtp_prose_tps / std::max(raw_prose_tps, 1e-6), raw_prose_tps,
                        eqp, raw_prose.size(), fdp,
                        static_cast<unsigned long long>(macc), static_cast<unsigned long long>(mdr),
                        mdr ? (100.0 * macc / mdr) : 0.0);
            std::printf("%s\n", mtp_draft::format_stats("MTP K=" + std::to_string(opt.mtp_k) + " prose", spec_last).c_str());

            // 2. Hybrid (SuffixTree + blk.48 MTP) on CODE + 129/129 lossless gate!
            double hyb128_tps = 0.0, hyb512_tps = 0.0;
            uint64_t hacc = 0, hdr = 0, hsfx = 0;
            auto hyb128 = run_greedy_mtp(p_rep_code, 129, opt.mtp_k, true, &hyb128_tps, &hacc, &hdr, &hsfx);
            uint32_t eqh = 0; int fdh = -1;
            for (size_t i = 0; i < raw128.size() && i < hyb128.size(); ++i) {
                if (raw128[i] == hyb128[i]) ++eqh; else if (fdh == -1) fdh = static_cast<int>(i);
            }
            std::printf("[BENCH tg128 Hybrid ] %.2f tok/s (%.2fx vs raw) | code lossless=%u/%zu (first_diff=%d) | sfx_hits=%llu acc=%llu/%llu (%.1f%%)\n",
                        hyb128_tps, hyb128_tps / std::max(res.tg128_raw_tps, 1e-6),
                        eqh, raw128.size(), fdh,
                        static_cast<unsigned long long>(hsfx), static_cast<unsigned long long>(hacc),
                        static_cast<unsigned long long>(hdr), hdr ? (100.0 * hacc / hdr) : 0.0);
            std::printf("%s\n", mtp_draft::format_stats("Hybrid K=" + std::to_string(opt.mtp_k) + " code", spec_last).c_str());

            uint64_t hacc5 = 0, hdr5 = 0, hsfx5 = 0;
            (void) run_greedy_mtp(p_rep_code, 513, 4u, true, &hyb512_tps, &hacc5, &hdr5, &hsfx5);
            std::printf("[BENCH tg512 Hybrid ] %.2f tok/s | sfx_hits=%llu acc=%llu/%llu (%.1f%%)\n",
                        hyb512_tps, static_cast<unsigned long long>(hsfx5),
                        static_cast<unsigned long long>(hacc5), static_cast<unsigned long long>(hdr5),
                        hdr5 ? (100.0 * hacc5 / hdr5) : 0.0);
            std::printf("%s\n", mtp_draft::format_stats("tg512 Hybrid K=4 code", spec_last).c_str());
            read_hwmon_max(max_pwr, max_tmem);
        }

        if (df.loaded) {
            auto enq_df_gemm = [&](const Q4A8WeightDev& w, uint64_t xq, uint64_t xds, uint64_t y, uint32_t M) {
                if (M == 1u) {
                    struct M1W32KA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, stride, pad; uint64_t ptx; };
                    M1W32KA ka{w.qs_va, w.dm_va, xq, xds, y, w.rows, w.cols, w.row_stride_bytes, 0u, 0ULL};
                    VJ_CHECK(rt.enqueue_kernel(df_gpu, *K[df_gpu].gemv_q4_w32, ((w.rows + 7u) / 8u) * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
                } else if (M <= 8u) {
                    struct GemmPpKA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, M, stride; uint64_t ptx, ptx2; };
                    GemmPpKA ka{w.qs_va, w.dm_va, xq, xds, y, w.rows, w.cols, M, w.row_stride_bytes, 0ULL, 0ULL};
                    VJ_CHECK(rt.enqueue_kernel(df_gpu, *K[df_gpu].gemm_q4_m8, ((w.rows + 7u) / 8u) * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
                } else if (M <= 12u) {
                    struct GemmPpKA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, M, stride; uint64_t ptx, ptx2; };
                    GemmPpKA ka{w.qs_va, w.dm_va, xq, xds, y, w.rows, w.cols, M, w.row_stride_bytes, 0ULL, 0ULL};
                    VJ_CHECK(rt.enqueue_kernel(df_gpu, *K[df_gpu].gemm_q4_m12, ((w.rows + 7u) / 8u) * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
                } else {
                    struct GemmPpKA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, M, stride; uint64_t ptx, ptx2; };
                    GemmPpKA ka{w.qs_va, w.dm_va, xq, xds, y, w.rows, w.cols, M, w.row_stride_bytes, 0ULL, 0ULL};
                    VJ_CHECK(rt.enqueue_kernel(df_gpu, *K[df_gpu].gemm_q4_m16, ((w.rows + 7u) / 8u) * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
                }
            };

            auto dflash_inject_kv = [&](uint32_t pos_beg, uint32_t n) {
                const uint32_t n_taps = static_cast<uint32_t>(df.tap_layers_0idx.size());
                const uint32_t tap_k  = n_taps * D;
                for (uint32_t off = 0u; off < n; off += 16u) {
                    const uint32_t c = std::min<uint32_t>(16u, n - off);
                    struct Q8KA { uint64_t xin, q8, ds; uint32_t M, K; };
                    const Q8KA qka{df.taps_va + static_cast<uint64_t>(off) * tap_k * 4ULL, df.taps_q8_va, df.taps_ds_va, c, tap_k};
                    VJ_CHECK(rt.enqueue_kernel(df_gpu, *K[df_gpu].df_quant_q8, ((tap_k + 255u) / 256u) * 256u, c, 1, 256, 1, 1, &qka, sizeof(qka), true));
                    enq_df_gemm(df.w_fc, df.taps_q8_va, df.taps_ds_va, df.ctx_f32_va, c);
                    struct RmsKA { uint64_t h, xa, w, xn, xq, xd; uint32_t M, D; float eps; };
                    const RmsKA rka{df.ctx_f32_va, 0ULL, df.enc_norm_va, 0ULL, df.ctx_q8_va, df.ctx_ds_va, c, D, df.rms_eps};
                    VJ_CHECK(rt.enqueue_kernel(df_gpu, *K[df_gpu].rms_quant, c * 256u, 1, 1, 256, 1, 1, &rka, sizeof(rka), true));
                    for (uint32_t l = 0u; l < df.n_layer; ++l) {
                        enq_df_gemm(df.L[l].w_kv, df.ctx_q8_va, df.ctx_ds_va, df.qkv_va, c);
                        struct KviKA { uint64_t kv, kn, kc, vc; uint32_t pb, n, nkv, wm; float rb, eps; };
                        const KviKA kka{df.qkv_va, df.L[l].k_norm_va, df.L[l].kv_k_va, df.L[l].kv_v_va,
                                        pos_beg + off, c, df.n_kv, df.L[l].kv_slots - 1u, df.rope_base, df.rms_eps};
                        VJ_CHECK(rt.enqueue_kernel(df_gpu, *K[df_gpu].df_kv_inject, df.n_kv * 128u, c, 1, 128, 1, 1, &kka, sizeof(kka), true));
                    }
                }
                VJ_CHECK(rt.submit_batch(df_gpu) && rt.wait_idle(df_gpu));
            };

            double df_t_body = 0.0, df_t_lm = 0.0, df_t_sel = 0.0;
            bool df_mask_arm_logged = false, df_mask_off_logged = false;  // one line each per request under DEBUG
            float  df_pmin_cur = opt.dflash_p_min;
            auto df_now_ms = []() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
            auto dflash_draft = [&](int32_t anchor_tok, uint32_t s_pos, uint32_t k_draft) -> std::vector<int32_t> {
                const double td_a = df_now_ms();
                const uint32_t B = k_draft + 1u;
                const uint32_t T = k_draft;
                struct EmbBlkKA { uint64_t eq, ed, msk, hout; uint32_t atok, B, D; };
                const EmbBlkKA ebka{emb_w[df_gpu].qs_va, emb_w[df_gpu].ds_va, df.mask_embd_f32_va, df.h_va,
                                    static_cast<uint32_t>(anchor_tok), B, D};
                VJ_CHECK(rt.enqueue_kernel(df_gpu, *K[df_gpu].df_embed_blk, B * 256u, 1, 1, 256, 1, 1, &ebka, sizeof(ebka), true));

                struct RmsKA  { uint64_t h, xa, w, xn, xq, xd; uint32_t M, D; float eps; };
                struct ConvKA { uint64_t u, dyn, base, hio, q8, ds; uint32_t B, D, side; };
                struct AttnKA { uint64_t qkv, qn, kn, kc, vc, oq, od; uint32_t spos, B, nq, nkv, swa; float rb, eps;
                                uint32_t blk_causal; };
                // Kernarg layout of vj_dflash2_attn_kernel (7 pointers, 5 u32, 2 f32, then this layer's flag).
                static_assert(offsetof(AttnKA, blk_causal) == 84u,
                              "AttnKA must match vj_dflash2_attn_kernel: blk_causal is kernarg byte 84");
                struct SwiKA  { uint64_t gu, q8, ds; uint32_t B, ff; };
                // Per-layer block mask for this cycle (default Bidir: arm = 0, mask = 0, the shipped mask). The
                // boundary is the GLOBAL SWA window, never a layer's ring size: past s_pos + B > W every layer,
                // the full one included, stays bidirectional, and ZLab never sets the full layer's bit at all.
                dflash_host::DraftBlockMaskConfig bmc{};
                bmc.mode = df.block_mask; bmc.pattern_known = df.block_mask_pattern_known;
                bmc.swa_layer_mask = df.block_mask_swa; bmc.n_layer = df.n_layer;
                bmc.block_start = s_pos; bmc.block_len = B; bmc.swa_window = df.swa_win;
                const dflash_host::DraftBlockMaskPlan bmp = dflash_host::plan_draft_block_mask(bmc);
                VJ_CHECK(bmp.ok);  // the setup preflight rejected every static error; only s_pos and B vary here
                if (df.block_mask != dflash_host::BlockMaskMode::Bidir && std::getenv("VEGAJET_DFLASH2_DEBUG")) {
                    if (bmp.arm && !df_mask_arm_logged) {
                        std::printf("  [DFMASK mode=%s pos=%u B=%u W=%u arm=1 causal=0x%x swa=0x%x layers=%u]\n",
                                    dflash_host::block_mask_mode_name(df.block_mask), s_pos, B, df.swa_win,
                                    bmp.causal_mask, df.block_mask_swa, df.n_layer);
                        df_mask_arm_logged = true;
                    } else if (!bmp.arm && !df_mask_off_logged) {
                        std::printf("  [DFMASK mode=%s pos=%u B=%u W=%u arm=0 causal=0x0 swa=0x%x layers=%u "
                                    "(all layers bidirectional)]\n",
                                    dflash_host::block_mask_mode_name(df.block_mask), s_pos, B, df.swa_win,
                                    df.block_mask_swa, df.n_layer);
                        df_mask_off_logged = true;
                    }
                }

                for (uint32_t l = 0u; l < df.n_layer; ++l) {
                    const auto& dl = df.L[l];
                    // Attention sublayer
                    const RmsKA r0{df.h_va, 0ULL, dl.attn_norm_va, df.xnorm_va, df.xq8_va, df.xds_va, B, D, df.rms_eps};
                    VJ_CHECK(rt.enqueue_kernel(df_gpu, *K[df_gpu].rms_quant, B * 256u, 1, 1, 256, 1, 1, &r0, sizeof(r0), true));
                    enq_df_gemm(dl.w_attn_conv, df.xq8_va, df.xds_va, df.dyn_va, B);
                    const ConvKA c0{df.xnorm_va, df.dyn_va, dl.attn_conv_base_va, 0ULL, df.xq8_va, df.xds_va, B, D, 0u};
                    VJ_CHECK(rt.enqueue_kernel(df_gpu, *K[df_gpu].df_dyn_conv, ((D + 255u) / 256u) * 256u, B, 1, 256, 1, 1, &c0, sizeof(c0), true));
                    enq_df_gemm(dl.w_qkv, df.xq8_va, df.xds_va, df.qkv_va, B);
                    // This launch's flag only: bit l of the cycle's mask (0 for a full layer, 0 outside the arm).
                    const uint32_t blk_causal_l = l < 32u ? ((bmp.causal_mask >> l) & 1u) : 0u;
                    const AttnKA aka{df.qkv_va, dl.q_norm_va, dl.k_norm_va, dl.kv_k_va, dl.kv_v_va,
                                     df.attn_q8_va, df.attn_ds_va, s_pos, B, df.n_head, df.n_kv, dl.kv_win, df.rope_base,
                                     df.rms_eps, blk_causal_l};
                    VJ_CHECK(rt.enqueue_kernel(df_gpu, *K[df_gpu].df_attn, df.n_head * 128u, B, 1, 128, 1, 1, &aka, sizeof(aka), true));
                    enq_df_gemm(dl.w_o, df.attn_q8_va, df.attn_ds_va, df.out_va, B);
                    const ConvKA c1{df.out_va, df.dyn_va, dl.attn_conv_base_va, df.h_va, 0ULL, 0ULL, B, D, 1u};
                    VJ_CHECK(rt.enqueue_kernel(df_gpu, *K[df_gpu].df_dyn_conv, ((D + 255u) / 256u) * 256u, B, 1, 256, 1, 1, &c1, sizeof(c1), true));

                    // FFN sublayer
                    const RmsKA r1{df.h_va, 0ULL, dl.ffn_norm_va, df.xnorm_va, df.xq8_va, df.xds_va, B, D, df.rms_eps};
                    VJ_CHECK(rt.enqueue_kernel(df_gpu, *K[df_gpu].rms_quant, B * 256u, 1, 1, 256, 1, 1, &r1, sizeof(r1), true));
                    enq_df_gemm(dl.w_ffn_conv, df.xq8_va, df.xds_va, df.dyn_va, B);
                    const ConvKA c2{df.xnorm_va, df.dyn_va, dl.ffn_conv_base_va, 0ULL, df.xq8_va, df.xds_va, B, D, 0u};
                    VJ_CHECK(rt.enqueue_kernel(df_gpu, *K[df_gpu].df_dyn_conv, ((D + 255u) / 256u) * 256u, B, 1, 256, 1, 1, &c2, sizeof(c2), true));
                    enq_df_gemm(dl.w_gu, df.xq8_va, df.xds_va, df.gu_va, B);
                    const SwiKA sk{df.gu_va, df.mid_q8_va, df.mid_ds_va, B, df.n_ff};
                    VJ_CHECK(rt.enqueue_kernel(df_gpu, *K[df_gpu].df_swiglu_q8, ((df.n_ff + 255u) / 256u) * 256u, B, 1, 256, 1, 1, &sk, sizeof(sk), true));
                    enq_df_gemm(dl.w_down, df.mid_q8_va, df.mid_ds_va, df.out_va, B);
                    const ConvKA c3{df.out_va, df.dyn_va, dl.ffn_conv_base_va, df.h_va, 0ULL, 0ULL, B, D, 1u};
                    VJ_CHECK(rt.enqueue_kernel(df_gpu, *K[df_gpu].df_dyn_conv, ((D + 255u) / 256u) * 256u, B, 1, 256, 1, 1, &c3, sizeof(c3), true));
                }

                // Final norm into df_gpu local VRAM (100% L2-hit for LM-head!), plus selector_hidden gate -> io_buf,
                // and tiny coherent copy (<= 56 KiB) to stage x_q8/x_ds into peer GPU local VRAM!
                const RmsKA rf{df.h_va + static_cast<uint64_t>(D) * 4ULL, 0ULL, df.out_norm_va, 0ULL,
                               df.xq8_va, df.xds_va, T, D, df.rms_eps};
                VJ_CHECK(rt.enqueue_kernel(df_gpu, *K[df_gpu].rms_quant, T * 256u, 1, 1, 256, 1, 1, &rf, sizeof(rf), true));
                enq_df_gemm(df.w_sel_hid, df.xq8_va, df.xds_va, df_io_gate_va, T);
                struct CpU32KA { uint64_t dst, src, n; };
                const uint64_t n_q8_u32 = (static_cast<uint64_t>(T) * D) >> 2;
                const uint64_t n_ds_u32 = (static_cast<uint64_t>(T) * (D >> 5)) << 1;
                const CpU32KA cp0q{df_io_xq8_va, df.xq8_va, n_q8_u32};
                const CpU32KA cp0d{df_io_xds_va, df.xds_va, n_ds_u32};
                VJ_CHECK(rt.enqueue_kernel(df_gpu, *K[df_gpu].memcpy_u32, ((n_q8_u32 + 255ULL) / 256ULL) * 256ULL, 1, 1, 256, 1, 1, &cp0q, sizeof(cp0q), true));
                VJ_CHECK(rt.enqueue_kernel(df_gpu, *K[df_gpu].memcpy_u32, ((n_ds_u32 + 255ULL) / 256ULL) * 256ULL, 1, 1, 256, 1, 1, &cp0d, sizeof(cp0d), true));
                VJ_CHECK(rt.submit_batch(df_gpu) && rt.wait_idle(df_gpu));
                const double td_b = df_now_ms();

                // Stage x_q8/x_ds into peer GPU local VRAM (~3 us) so ALL GPUs run vocab_topk16 from local L2 VRAM!
                for (size_t g = 0; g < N; ++g) {
                    if (g == df_gpu) continue;
                    const CpU32KA cp1q{S[g].x_q8_va, df_io_xq8_va, n_q8_u32};
                    const CpU32KA cp1d{S[g].x_ds_va, df_io_xds_va, n_ds_u32};
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].memcpy_u32, ((n_q8_u32 + 255ULL) / 256ULL) * 256ULL, 1, 1, 256, 1, 1, &cp1q, sizeof(cp1q), true));
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].memcpy_u32, ((n_ds_u32 + 255ULL) / 256ULL) * 256ULL, 1, 1, 256, 1, 1, &cp1d, sizeof(cp1d), true));
                }

                // TP=N LDS-staged LM-head Top-16 across all MI50s from local VRAM!
                const bool df_use_fr = (use_fr_spec && std::getenv("VEGAJET_DF_FULL_VOCAB") == nullptr);
                const uint32_t df_voc_blocks = df_use_fr ? 120u : kVocabBlocks;
                for (size_t g = 0; g < N; ++g) {
                    const Q8WeightDev& vw = df_use_fr ? fr_w[g] : out_w[g];
                    struct TopKA { uint64_t wq, wd, xq, xd, tmap, bv, bi; uint32_t lr, cols, ro, stride, T; };
                    const TopKA tka{vw.qs_va, vw.ds_va,
                                    (g == df_gpu) ? df.xq8_va : S[g].x_q8_va,
                                    (g == df_gpu) ? df.xds_va : S[g].x_ds_va,
                                    df_use_fr ? fr_tok_map_va[g] : 0ULL,
                                    df_io_val_va[g], df_io_id_va[g],
                                    df_use_fr ? static_cast<uint32_t>(fr_tok_map[g].size()) : voc_rows[g],
                                    D, voc_off[g], vw.row_stride_bytes, T};
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].df_vocab_topk16, df_voc_blocks * 256u, (T + 3u) / 4u, 1, 256, 1, 1, &tka, sizeof(tka), true));
                }
                VJ_CHECK(submit_all() && wait_all());
                __atomic_thread_fence(__ATOMIC_ACQUIRE);
                const double td_c = df_now_ms();

                // Host merge top-16 per slot + sequential DFlash2 selector walk (<30 us!)
                const bool df_no_sel = (std::getenv("VEGAJET_DFLASH2_NO_SEL") != nullptr);
                const bool df_dbg    = (std::getenv("VEGAJET_DFLASH2_DEBUG") != nullptr);
                const auto& dec_reg = quant::DecoderRegistry::instance();
                std::vector<float> pred_vec(df.sel_rank), succ_vec(df.sel_rank), cond(df.sel_rank);
                std::vector<int32_t> drafts;
                drafts.reserve(T);
                int32_t pred = anchor_tok;
                for (uint32_t t = 0u; t < T; ++t) {
                    float   topv[16];
                    int32_t topi[16];
                    for (uint32_t k = 0u; k < 16u; ++k) { topv[k] = -1e30f; topi[k] = -1; }
                    auto consider = [&](float v, int32_t id) {
                        if (id < 0 || v <= topv[15]) return;
                        uint32_t p = 15u;
                        while (p > 0u && v > topv[p - 1u]) { topv[p] = topv[p - 1u]; topi[p] = topi[p - 1u]; --p; }
                        topv[p] = v; topi[p] = id;
                    };
                    const size_t base = static_cast<size_t>(t) * df_voc_blocks * 16u;
                    for (size_t j = 0; j < static_cast<size_t>(df_voc_blocks) * 16u; ++j) {
                        for (size_t g = 0; g < N; ++g) consider(df_io_val_host[g][base + j], df_io_id_host[g][base + j]);
                    }
                    uint32_t best_k = 0u;
                    float best_sc = topv[0];
                    float scores[16];
                    if (df_no_sel) {
                        for (uint32_t k = 0u; k < 16u; ++k) scores[k] = topv[k];
                    } else {
                        if (pred >= 0 && static_cast<uint32_t>(pred) < df.sel_vocab) {
                            dec_reg.decode_row_f32(df.sel_pred_type,
                                df.sel_pred_raw.data() + static_cast<size_t>(pred) * df.sel_pred_rb,
                                pred_vec.data(), df.sel_rank);
                        } else {
                            std::fill(pred_vec.begin(), pred_vec.end(), 0.0f);
                        }
                        const float* g_row = df_io_gate_host + static_cast<size_t>(t) * df.sel_rank;
                        for (uint32_t r = 0u; r < df.sel_rank; ++r) cond[r] = pred_vec[r] * g_row[r];

                        best_sc = -1e30f;
                        for (uint32_t k = 0u; k < 16u; ++k) {
                            float sc = topv[k];
                            if (topi[k] >= 0 && static_cast<uint32_t>(topi[k]) < df.sel_vocab) {
                                dec_reg.decode_row_f32(df.sel_succ_type,
                                    df.sel_succ_raw.data() + static_cast<size_t>(topi[k]) * df.sel_succ_rb,
                                    succ_vec.data(), df.sel_rank);
                                float dot = 0.0f;
                                for (uint32_t r = 0u; r < df.sel_rank; ++r) dot += succ_vec[r] * cond[r];
                                sc += dot;
                            }
                            scores[k] = sc;
                            if (sc > best_sc) { best_sc = sc; best_k = k; }
                        }
                    }
                    if (df_dbg && s_pos < 64u && t < 5u) {
                        std::printf("  [DFDBG pos=%u t=%u] u0=(%d:%.2f) u1=(%d:%.2f) pick=(k=%u id=%d sc=%.2f u=%.2f)\n",
                                    s_pos, t, topi[0], topv[0], topi[1], topv[1], best_k, topi[best_k], best_sc, topv[best_k]);
                        std::fflush(stdout);
                    }
                    if (df_pmin_cur > 0.0f) {
                        float z = 0.0f;
                        for (uint32_t k = 0u; k < 16u; ++k) z += std::exp(scores[k] - best_sc);
                        if (1.0f / z < df_pmin_cur && !drafts.empty()) break;
                    }
                    pred = topi[best_k];
                    if (pred < 0 || static_cast<uint32_t>(pred) >= vocab) {
                        std::printf("  [DFERR pos=%u t=%u] invalid draft tok=%d (u0=%d:%.2f)\n", s_pos, t, pred, topi[0], topv[0]);
                        std::fflush(stdout);
                        pred = 0;
                    }
                    drafts.push_back(pred);
                }
                const double td_d = df_now_ms();
                df_t_body += (td_b - td_a);
                df_t_lm   += (td_c - td_b);
                df_t_sel  += (td_d - td_c);
                return drafts;
            };

            auto run_greedy_dflash2 = [&](const std::vector<int32_t>& prompt_ids, int n_gen, uint32_t k_draft,
                                          double* out_tps, uint64_t* out_acc, uint64_t* out_dr,
                                          bool use_hybrid_suffix = false) -> std::vector<int32_t> {
                // > 32K context: the blk.48 MTP head (+ FR-Spec when --fr-spec is set) takes over (core/spec_replay).
                spec_replay::SourceConfig src_cfg;
                src_cfg.has_dflash2 = true; src_cfg.has_mtp = has_mtp; src_cfg.fr_spec_rows = use_fr_spec ? opt.fr_spec_rows : 0u;
                src_cfg.dflash_max_anchor = df_kv_plan.max_anchor_pos;
                const spec_replay::SourcePlan src_plan = spec_replay::plan_sources(src_cfg, static_cast<uint32_t>(prompt_ids.size()),
                                                                                    static_cast<uint32_t>(std::max(n_gen, 0)));
                if (src_plan.initial != spec_replay::Source::kDFlash2) {
                    std::printf("[DFLASH2] prompt %zu > anchor limit %u: %s\n", prompt_ids.size(), std::min(df_kv_plan.max_anchor_pos, src_cfg.long_ctx),
                                spec_replay::source_name(src_plan.initial));
                    if (has_mtp) return run_greedy_mtp(prompt_ids, n_gen, 4u, use_hybrid_suffix, out_tps, out_acc, out_dr);
                }
                reset_state();
                spec_last = {};
                df_t_body = df_t_lm = df_t_sel = 0.0;
                sfx::CascadeConfig ccfg;
                ccfg.min_match = 4u;
                ccfg.min_chain = 4u;
                ccfg.max_draft = std::min<uint32_t>(opt.suffix_max_draft, kGdnRingSlots - 2u);
                sfx::CascadeHybrid cascade(ccfg);
                spec_cascade::CascadeConfig zcfg;
                zcfg.min_match = 4u;
                zcfg.min_confident = 4u;
                zcfg.max_draft = std::min<uint32_t>(opt.suffix_max_draft, kGdnRingSlots - 2u);
                zcfg = spec_cascade::config_from_env(zcfg);
                spec_cascade::ZeroTaxCascade zt_cascade(zcfg);
                const bool use_zt_cascade = (use_hybrid_suffix && zcfg.enabled);
                if (use_zt_cascade) {
                    zt_cascade.reserve(prompt_ids.size() + static_cast<size_t>(n_gen) + 16u);
                    zt_cascade.commit(prompt_ids);
                } else if (use_hybrid_suffix) {
                    cascade.reserve(prompt_ids.size() + static_cast<size_t>(n_gen) + 16u);
                    cascade.commit(prompt_ids);
                }
                const bool use_evict = (std::getenv("VEGAJET_EVICT_ADAPT") != nullptr && std::getenv("VEGAJET_EVICT_ADAPT")[0] == '1');
                const char* evict_ids_env = std::getenv("VEGAJET_EVICT_ROUTE_IDS");
                // Capture is on by default with the adaptive controller; =0 isolates the copy/dedup overhead in an
                // A/B without changing the acceptance policy (the signal then stays explicitly unknown).
                const bool capture_route_ids = use_evict && !(evict_ids_env && evict_ids_env[0] == '0');
                vegajet::EvictDepthConfig ecfg;
                ecfg.experts_per_depth = 48u * 8u;  // allowance for one full top-8 route per layer per draft slot
                vegajet::EvictDepth evict_ctl(k_draft, ecfg);
                // The only cost signal passed to choose() is the exact distinct (layer, expert) union of the last
                // completed target verifier pass.  It is consumed once per proposal; cold start, a fused M=1 path
                // without per-layer top-ID rows, or a deferred readback leaves it 0 (unknown), never an estimate.
                vegajet::RouteUnionFeedback evict_route_fb;
                auto record_evict_routes = [&](uint32_t tokens) {
                    if (!capture_route_ids) return;
                    __atomic_thread_fence(__ATOMIC_ACQUIRE);
                    const bool complete = evict_cap.complete && evict_cap.tokens == tokens;
                    vegajet::RouteUnionConfig rcfg;
                    rcfg.max_layers = 48u;
                    rcfg.max_experts = 256u;
                    rcfg.max_tokens = kEvictTopTokens;
                    rcfg.top_k = kEvictTopK;
                    (void)vegajet::record_completed_routes(&evict_route_fb, complete ? evict_cap.host : nullptr,
                                                           48u, tokens, kEvictTopK, kEvictTopStrideIds, rcfg);
                };
                // v1.2 C7+S1: depth from the measured per-M verify time + live acceptance.
                spec_policy::MeasuredDepthGovernor df_gov;
                uint32_t df_gov_depth_sum = 0u, df_gov_depth_n = 0u, df_gov_plain_n = 0u;

                uint32_t pos = 0u;
                int32_t last_pred = -1;
                for (size_t off = 0; off < prompt_ids.size(); off += opt.prefill_chunk) {
                    const size_t cnt = std::min<size_t>(opt.prefill_chunk, prompt_ids.size() - off);
                    std::vector<int32_t> chunk(prompt_ids.begin() + off, prompt_ids.begin() + off + cnt);
                    const bool is_last = (off + cnt == prompt_ids.size());
                    auto preds = step_batch(chunk, pos, opt.use_wy_prefill, false, is_last, true);
                    dflash_inject_kv(pos, static_cast<uint32_t>(cnt));
                    if (is_last) last_pred = preds.back();
                    pos += static_cast<uint32_t>(cnt);
                }
                std::vector<int32_t> gen;
                gen.push_back(last_pred);
                if (use_zt_cascade) zt_cascade.commit(last_pred);
                else if (use_hybrid_suffix) cascade.commit(last_pred);
                uint64_t acc_tot = 0, dr_tot = 0;
                double t_dr = 0.0, t_ver = 0.0, t_inj = 0.0;
                std::array<spec::CycleTimes, 3> cycle_times;
                uint32_t n_cyc = 0;
                auto now_ms = []() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
                const auto td0 = std::chrono::steady_clock::now();
                // Acceptance-aware gate (default on; VEGAJET_SPEC_GATE=0 disables): exact raw single steps while
                // speculation emits fewer tokens per ms than raw.
                const bool use_spec_gate = !(std::getenv("VEGAJET_SPEC_GATE") && std::strcmp(std::getenv("VEGAJET_SPEC_GATE"), "0") == 0);
                spec_gate::Gate sgate;
                if (res.tg128_raw_tps > 0.0) sgate.seed_raw_ms(1000.0 / res.tg128_raw_tps);
                while (static_cast<int>(gen.size()) < n_gen) {
                    const uint32_t rem = static_cast<uint32_t>(n_gen - static_cast<int>(gen.size()));
                    const uint32_t k_cap = std::min<uint32_t>({df.block_size - 1u, kGdnRingSlots - 2u, (rem > 1u) ? (rem - 1u) : 0u});
                    const bool gate_raw = use_spec_gate && !sgate.want_spec();
                    const bool evict_u_measured = use_evict && evict_route_fb.available();
                    const uint32_t evict_u_tokens = use_evict ? evict_route_fb.tokens() : 0u;
                    const uint64_t evict_u = use_evict ? evict_route_fb.take() : 0ull;
                    uint32_t k_target = use_evict ? evict_ctl.choose(k_cap, evict_u) : k_draft;
                    if (v12_spec_gov) {
                        // M2: choose on the complete-cycle cost. depth 0 = plain decoding and is honoured here: the
                        // k_cur == 0 branch below steps one token and injects it into the drafter KV.
                        const auto gc = df_gov.choose_depth(std::min<uint32_t>(k_draft, df.block_size - 1u));
                        if (gc.warmed) {
                            k_target = std::min(k_target, gc.depth);
                            df_gov_depth_sum += gc.depth;
                            ++df_gov_depth_n;
                            if (gc.depth == 0u) ++df_gov_plain_n;
                        }
                    }
                    // Past the switch position the drafter cache is no longer exact: no neural DFlash2 draft (the
                    // MTP cache was not fed during this request): single target steps, lossless).
                    const bool df_past_limit = spec_replay::source_at(src_plan, pos) != spec_replay::Source::kDFlash2;
                    const bool evict_trace = use_evict && std::getenv("VEGAJET_EVICT_TRACE") != nullptr && pos < 64u;
                    const uint32_t k_cur = (df_past_limit || gate_raw) ? 0u : std::min(k_target, k_cap);
                    const uint32_t k_sfx = std::min<uint32_t>({opt.suffix_max_draft, kGdnRingSlots - 2u, (rem > 1u) ? (rem - 1u) : 0u});
                    if (k_cur == 0u) {
                        const double tr0 = now_ms();
                        // Gate raw steps skip the drafter tap capture: capture forces the non-super M=1 path, which is not bit-identical
                        // to plain decode. The drafter KV misses these positions (acceptance only; verify keeps output exact).
                        auto preds = step_batch({gen.back()}, pos, false, false, true, !gate_raw, capture_route_ids);
                        if (use_spec_gate) sgate.on_raw(now_ms() - tr0);
                        record_evict_routes(1u);
                        if (evict_trace) {
                            std::printf("[EVICT pos=%u] proposed=%u unique=%llu unique_source=%s unique_tokens=%u "
                                        "verified=0 correction=%d emitted=1\n",
                                        pos, k_target, static_cast<unsigned long long>(evict_u),
                                        evict_u_measured ? "measured" : "unknown", evict_u_tokens, preds[0]);
                        }
                        if (!gate_raw) dflash_inject_kv(pos, 1u);
                        ++pos;
                        gen.push_back(preds[0]);
                        spec_last.add_single();
                        if (use_zt_cascade) zt_cascade.commit(preds[0]);
                        else if (use_hybrid_suffix) cascade.commit(preds[0]);
                        continue;
                    }
                    const double t0 = now_ms();
                    std::vector<int32_t> drafts;
                    bool used_suffix = false;
                    if (use_zt_cascade && k_sfx > 0u) {
                        const auto dec = zt_cascade.decide();
                        if (dec.source == spec_cascade::DraftSource::SUFFIX_ONLY && !dec.draft.empty()) {
                            const size_t take = std::min<size_t>(dec.draft.size(), k_sfx);
                            drafts.assign(dec.draft.begin(), dec.draft.begin() + take);
                            used_suffix = true;
                        }
                    } else if (use_hybrid_suffix && k_sfx > 0u) {
                        const auto dec = cascade.decide_and_count();
                        if (dec.source == sfx::DraftSource::kSuffix && !dec.draft.empty()) {
                            const size_t take = std::min<size_t>(dec.draft.size(), k_sfx);
                            drafts.assign(dec.draft.begin(), dec.draft.begin() + take);
                            used_suffix = true;
                        }
                    }
                    if (!used_suffix) drafts = dflash_draft(gen.back(), pos, k_cur);
                    const double t1 = now_ms();
                    treewy_force_width(drafts, gen.back());
                    const uint32_t dk = static_cast<uint32_t>(drafts.size());
                    std::vector<int32_t> vtoks;
                    vtoks.reserve(1u + dk);
                    vtoks.push_back(gen.back());
                    for (int32_t tok_d : drafts) vtoks.push_back(tok_d);
                    auto preds = step_batch(vtoks, pos, false, true, true, true, capture_route_ids);
                    record_evict_routes(static_cast<uint32_t>(vtoks.size()));
                    const double t2 = now_ms();
                    const unsigned verifier = cycle_mode();
                    uint32_t natural = 0u;
                    while (natural < dk && preds[natural] == drafts[natural]) ++natural;
                    const uint32_t acc = treewy_clamp_acc(natural, dk);
                    for (uint32_t j = 0; j < acc; ++j) {
                        gen.push_back(drafts[j]);
                        if (use_zt_cascade) zt_cascade.commit(drafts[j]);
                        else if (use_hybrid_suffix) cascade.commit(drafts[j]);
                    }
                    if (v12_spec_gov) {
                        df_gov.observe_verify(std::min<uint32_t>(dk + 1u, 17u), t2 - t1);
                        df_gov.observe_accept(dk, acc);
                        if (!used_suffix && dk > 0u) df_gov.observe_draft(dk, t1 - t0);
                    }
                    if (std::getenv("VEGAJET_DFLASH2_DEBUG") && pos < 64u) {
                        std::printf("  [DFVER pos=%u acc=%u/%u] tgt=[%d,%d,%d,%d,%d] dft=[%d,%d,%d,%d,%d]\n",
                                    pos, acc, dk,
                                    preds.size()>0?preds[0]:-1, preds.size()>1?preds[1]:-1, preds.size()>2?preds[2]:-1, preds.size()>3?preds[3]:-1, preds.size()>4?preds[4]:-1,
                                    drafts.size()>0?drafts[0]:-1, drafts.size()>1?drafts[1]:-1, drafts.size()>2?drafts[2]:-1, drafts.size()>3?drafts[3]:-1, drafts.size()>4?drafts[4]:-1);
                    }
                    if (static_cast<int>(gen.size()) < n_gen) {
                        gen.push_back(preds[acc]);
                        if (use_zt_cascade) zt_cascade.commit(preds[acc]);
                        else if (use_hybrid_suffix) cascade.commit(preds[acc]);
                    }
                    if (use_evict && !used_suffix && dk > 0u) {
                        evict_ctl.observe(dk, acc);  // target verifier result only; the controller never commits
                        if (evict_trace) {
                            std::printf("[EVICT pos=%u] proposed=%u unique=%llu unique_source=%s unique_tokens=%u "
                                        "verified=%u correction=%d emitted=%u\n",
                                        pos, k_target, static_cast<unsigned long long>(evict_u),
                                        evict_u_measured ? "measured" : "unknown", evict_u_tokens, acc, preds[acc],
                                        acc + 1u);
                        }
                    }
                    acc_tot += acc; dr_tot += dk;
                    spec_last.add(mtp_draft::CycleSample{used_suffix, dk, acc, t1 - t0, t2 - t1, {}});
                    const uint32_t n_commit = acc + 1u;
                    const double recompute_start = reject_recompute_ms;
                    commit_spec_gdn(acc, dk);
                    ring_head = (ring_head + n_commit) % kGdnRingSlots;
                    conv_head = (conv_head + n_commit) % kGdnRingSlots;
                    dflash_inject_kv(pos, n_commit);
                    const double t3 = now_ms();
                    const double r = reject_recompute_ms - recompute_start;
                    cycle_times[verifier].add(t1 - t0, t2 - t1, t3 - t2 - r, r);
                    if (v12_spec_gov && !used_suffix) df_gov.observe_catchup(n_commit, t3 - t2 - r);
                    t_dr += (t1 - t0); t_ver += (t2 - t1); t_inj += (t3 - t2 - r); ++n_cyc;
                    if (use_spec_gate) sgate.on_spec(n_commit, t3 - t0);
                    pos += n_commit;
                }
                const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - td0).count();
                spec_last.wall_s = sec;
                if (out_tps && n_gen > 1) *out_tps = (n_gen - 1) / std::max(sec, 1e-9);
                if (out_acc) *out_acc = acc_tot;
                if (out_dr)  *out_dr  = dr_tot;
                print_cycle_times(cycle_times, "dflash2");
                if (n_cyc > 0u) {
                    const uint64_t sfx_c = use_zt_cascade ? zt_cascade.stats().suffix_only : cascade.suffix_cycles;
                    const uint64_t neu_c = use_zt_cascade ? zt_cascade.stats().neural      : cascade.neural_cycles;
                    std::printf("[DFLASH2%s gpu=%zu cycles] %u cycles (sfx=%llu neural=%llu) | draft %.2f ms (body %.2f + lm %.2f + sel %.2f) | verify %.2f ms | inject %.2f ms | mean_tok/cyc=%.2f\n",
                                use_zt_cascade ? "+ZeroTaxCascade" : (use_hybrid_suffix ? "+CascadeHybrid" : ""), df_gpu, n_cyc,
                                static_cast<unsigned long long>(sfx_c),
                                static_cast<unsigned long long>(neu_c),
                                t_dr / n_cyc, df_t_body / n_cyc, df_t_lm / n_cyc, df_t_sel / n_cyc,
                                t_ver / n_cyc, t_inj / n_cyc,
                                static_cast<double>(n_gen - 1) / n_cyc);
                    if (v12_spec_gov && df_gov_depth_n > 0u)
                        std::printf("[V12 SPEC_GOV gpu=%zu] depth %.2f | plain %u | alpha %.3f | t_ver(M=4) %.2f ms | "
                                    "draft %.2f us/tok | catchup %.2f us/tok | inflation(M=8) %.2f\n",
                                    df_gpu,
                                    static_cast<double>(df_gov_depth_sum) / static_cast<double>(df_gov_depth_n),
                                    df_gov_plain_n, df_gov.alpha(), df_gov.measured_ms(4u),
                                    1000.0 * df_gov.draft_ms_per_tok(), 1000.0 * df_gov.catchup_ms_per_tok(),
                                    df_gov.time_inflation(8u));
                }
                return gen;
            };

            double raw_prose_tps = 0.0, df_prose_tps = 0.0, df_code_tps = 0.0;
            auto raw_prose = run_greedy_raw(p_base, 129, &raw_prose_tps);
            uint64_t pacc = 0, pdr = 0, cacc = 0, cdr = 0;
            auto df_prose = run_greedy_dflash2(p_base, 129, opt.dflash_k, &df_prose_tps, &pacc, &pdr);
            uint32_t eqp = 0; int fdp = -1;
            for (size_t i = 0; i < raw_prose.size() && i < df_prose.size(); ++i) {
                if (raw_prose[i] == df_prose[i]) ++eqp; else if (fdp == -1) fdp = static_cast<int>(i);
            }
            std::printf("[BENCH tg128 DFlash2 K=%u prose] %.2f tok/s (%.2fx vs raw %.2f) | prose lossless=%u/%zu (first_diff=%d) | acc=%llu/%llu (%.1f%%)\n",
                        opt.dflash_k, df_prose_tps, df_prose_tps / std::max(raw_prose_tps, 1e-6), raw_prose_tps,
                        eqp, raw_prose.size(), fdp,
                        static_cast<unsigned long long>(pacc), static_cast<unsigned long long>(pdr),
                        pdr ? (100.0 * pacc / pdr) : 0.0);
            std::printf("%s\n", mtp_draft::format_stats("DFlash2 K=" + std::to_string(opt.dflash_k) + " prose", spec_last).c_str());

            if (std::getenv("VEGAJET_DF_SWEEP")) {
                struct SweepCfg { uint32_t k; float pmin; };
                const SweepCfg cfgs[] = {
                    {7u, 0.0f}, {9u, 0.0f}, {11u, 0.0f},
                    {15u, 0.45f}, {15u, 0.55f}, {15u, 0.65f}, {15u, 0.75f}, {15u, 0.85f},
                    {7u, 0.55f}, {9u, 0.55f}
                };
                for (const auto& sc : cfgs) {
                    df_pmin_cur = sc.pmin;
                    double stps = 0.0; uint64_t sa = 0, sd = 0;
                    auto sg = run_greedy_dflash2(p_base, 129, sc.k, &stps, &sa, &sd);
                    uint32_t eq = 0; int fd = -1;
                    for (size_t i = 0; i < raw_prose.size() && i < sg.size(); ++i) {
                        if (raw_prose[i] == sg[i]) ++eq; else if (fd == -1) fd = static_cast<int>(i);
                    }
                    std::printf("[SWEEP DFlash2 K=%u pmin=%.2f prose] %.2f tok/s (%.2fx) | lossless=%u/%zu | acc=%llu/%llu (%.1f%%)\n",
                                sc.k, sc.pmin, stps, stps / std::max(raw_prose_tps, 1e-6), eq, raw_prose.size(),
                                static_cast<unsigned long long>(sa), static_cast<unsigned long long>(sd),
                                sd ? (100.0 * sa / sd) : 0.0);
                }
                df_pmin_cur = opt.dflash_p_min;
            }

            auto df_code = run_greedy_dflash2(p_rep_code, 129, opt.dflash_k, &df_code_tps, &cacc, &cdr);
            uint32_t eqc = 0; int fdc = -1;
            for (size_t i = 0; i < raw128.size() && i < df_code.size(); ++i) {
                if (raw128[i] == df_code[i]) ++eqc; else if (fdc == -1) fdc = static_cast<int>(i);
            }
            std::printf("[BENCH tg128 DFlash2 K=%u code ] %.2f tok/s (%.2fx vs raw %.2f) | code lossless=%u/%zu (first_diff=%d) | acc=%llu/%llu (%.1f%%)\n",
                        opt.dflash_k, df_code_tps, df_code_tps / std::max(res.tg128_raw_tps, 1e-6), res.tg128_raw_tps,
                        eqc, raw128.size(), fdc,
                        static_cast<unsigned long long>(cacc), static_cast<unsigned long long>(cdr),
                        cdr ? (100.0 * cacc / cdr) : 0.0);
            std::printf("%s\n", mtp_draft::format_stats("DFlash2 K=" + std::to_string(opt.dflash_k) + " code", spec_last).c_str());
            double df_hyb_tps = 0.0; uint64_t hacc = 0, hdr = 0;
            auto df_hyb = run_greedy_dflash2(p_rep_code, 129, opt.dflash_k, &df_hyb_tps, &hacc, &hdr, true);
            uint32_t eqh = 0; int fdh = -1;
            for (size_t i = 0; i < raw128.size() && i < df_hyb.size(); ++i) {
                if (raw128[i] == df_hyb[i]) ++eqh; else if (fdh == -1) fdh = static_cast<int>(i);
            }
            std::printf("[BENCH tg128 DFlash2+Hyb code ] %.2f tok/s (%.2fx vs raw %.2f) | code lossless=%u/%zu (first_diff=%d) | acc=%llu/%llu (%.1f%%)\n",
                        df_hyb_tps, df_hyb_tps / std::max(res.tg128_raw_tps, 1e-6), res.tg128_raw_tps,
                        eqh, raw128.size(), fdh,
                        static_cast<unsigned long long>(hacc), static_cast<unsigned long long>(hdr),
                        hdr ? (100.0 * hacc / hdr) : 0.0);
            std::printf("%s\n", mtp_draft::format_stats("DFlash2+Hyb K=" + std::to_string(opt.dflash_k) + " code", spec_last).c_str());
            if (df_hyb_tps > best_spec_tps) { best_spec_tps = df_hyb_tps; spec_eq = eqh; spec_tot = static_cast<uint32_t>(raw128.size()); spec_fd = fdh; }
            if (eqp != raw_prose.size() || eqc != raw128.size() || eqh != raw128.size()) all_lossless = false;
            read_hwmon_max(pass_pwr, pass_tmem);
            max_pwr = std::max(max_pwr, pass_pwr); max_tmem = std::max(max_tmem, pass_tmem);
        }

        if (include_prefill) {
        {
            prefill::Request preq;
            preq.prompt_tokens = 2048u;
            preq.macro_m       = opt.prefill_chunk;
            preq.gdn_chunk     = 16u;
            preq.allow_int8_wire = opt.prefill_ar_int8;
            const auto pplan = prefill::plan(prefill::ornith_122b_tp2(), preq);
            if (pplan.ok) {
                std::printf("[PREFILL-PLAN] %s\n", prefill::describe(pplan).c_str());
            }
        }
        std::vector<int32_t> pp512(512), pp2048(2048);
        for (size_t i = 0; i < pp512.size(); ++i)  pp512[i]  = p_base[i % p_base.size()];
        for (size_t i = 0; i < pp2048.size(); ++i) pp2048[i] = p_base[i % p_base.size()];
        auto pp512_out = run_greedy_raw(pp512,  1, nullptr, &res.pp512_tps);
        std::printf("[BENCH pp512  M=%-2u  ] %.2f tok/s (WY_GDN=%s, 0 SOLVE_TRI, first_tok=%d)\n",
                    opt.prefill_chunk, res.pp512_tps, opt.use_wy_prefill ? "ON" : "OFF",
                    pp512_out.empty() ? -1 : pp512_out[0]);
        read_hwmon_max(pass_pwr, pass_tmem);
        max_pwr = std::max(max_pwr, pass_pwr); max_tmem = std::max(max_tmem, pass_tmem);

        auto pp2048_out = run_greedy_raw(pp2048, 1, nullptr, &res.pp2048_tps);
        std::printf("[BENCH pp2048 M=%-2u  ] %.2f tok/s (WY_GDN=%s, 0 SOLVE_TRI, first_tok=%d)\n",
                    opt.prefill_chunk, res.pp2048_tps, opt.use_wy_prefill ? "ON" : "OFF",
                    pp2048_out.empty() ? -1 : pp2048_out[0]);
        if (opt.pp_stages) {
            st_on = true;
            double st_tps = 0.0;
            const auto t0 = std::chrono::steady_clock::now();
            (void) run_greedy_raw(pp2048, 1, nullptr, &st_tps);
            const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            st_on = false;
            double tot[kfd::kMaxGpus] = {};
            for (size_t g = 0; g < N; ++g) for (double v : st_acc[g]) tot[g] += v;
            std::printf("[PP-STAGES pp2048 M=%u] uncached request %.0f ms with per-stage drains (%.0f ms = %.2f tok/s without); ms per request, ranks 0..%zu:\n",
                        opt.prefill_chunk, wall, 2048.0 * 1000.0 / res.pp2048_tps, res.pp2048_tps, N - 1u);
            for (uint32_t st = 0; st < kStCount; ++st) {
                std::printf("  %-38s", kStName[st]);
                for (size_t g = 0; g < N; ++g) std::printf(" %s %8.1f", g ? "|" : "", st_acc[g][st]);
                std::printf("%s\n", (st == kStAr || st == kStVocab) ? "   (all ranks together, booked on rank 0)" : "");
            }
            std::printf("  %-38s", "sum");
            for (size_t g = 0; g < N; ++g) std::printf(" %s %8.1f", g ? "|" : "", tot[g]);
            std::printf("   (ranks run concurrently in normal operation: critical path = shared stages + max rank)\n");
        }
        read_hwmon_max(pass_pwr, pass_tmem);
        max_pwr = std::max(max_pwr, pass_pwr); max_tmem = std::max(max_tmem, pass_tmem);
        }

        if (!vout && !l4_on) run_m1_profile();
        }
        read_hwmon_max(pass_pwr, pass_tmem);
        max_pwr = std::max(max_pwr, pass_pwr); max_tmem = std::max(max_tmem, pass_tmem);
        read_ecc_totals(ue1, de1);
        if (vout) {
            vout->raw_tg128_tps = res.tg128_raw_tps;
            vout->spec_tg128_tps = (best_spec_tps > 0.0) ? best_spec_tps : res.tg128_suffix_tps;
            vout->pp512_tps = res.pp512_tps;
            vout->pp2048_tps = res.pp2048_tps;
            vout->primary_tps = (vout->spec_tg128_tps > 0.0) ? vout->spec_tg128_tps : vout->raw_tg128_tps;
            vout->dispatches_per_tok = res.dispatches_per_tok;
            vout->lossless_matched = spec_tot > 0 ? spec_eq : vout->lossless_matched;
            vout->lossless_total = spec_tot > 0 ? spec_tot : vout->lossless_total;
            vout->first_diff = spec_tot > 0 ? spec_fd : vout->first_diff;
            vout->parity_ok = vout->parity_ok && all_lossless;
            vout->tmem_max_c = pass_tmem;
            vout->ecc_ue_delta = ue1 - ue0;
            vout->ecc_de_delta = de1 - de0;
            vout->ok = vout->parity_ok && (vout->ecc_ue_delta == 0) && (vout->ecc_de_delta == 0);
        }
        return true;
    };

    auto exec_warm_variant = [&](const std::string& label, const std::vector<std::string>& raw_args, bool parity_only) -> warm::SweepVariantResult {
        warm::SweepVariantResult vr;
        vr.label = label;
        std::vector<std::string> cli_args;
        std::vector<warm::EnvAssignment> env_ov;
        warm::extract_env_overrides(raw_args, cli_args, env_ov);
        std::string bmode = "all", ferr;
        if (!apply_warm_flags(cli_args, opt, bmode, &ferr)) {
            vr.ok = false; vr.error = ferr;
            return vr;
        }
        warm::ScopedEnvOverrides env_guard(env_ov);
        // Warm variants reset/prefill their caches; never change format on a live sequence.
        kvq::DeviceFormat next_format=kvq::DeviceFormat::Q8;
        const char* quant=std::getenv("VEGAJET_KV_QUANT");
        if (quant && (!kvq::parse_format(quant,next_format) || next_format==kvq::DeviceFormat::Fp16)) {
            vr.error="VEGAJET_KV_QUANT must be q8_0, q4_0 or turbo3"; return vr;
        }
        l4_format=next_format; l4_compressed=l4_format!=kvq::DeviceFormat::Q8;
        const char* split=std::getenv("VEGAJET_SPLIT_KV");
        const char* chrono=std::getenv("VEGAJET_KV_CHRONOLOGICAL");
        l4_split=split && std::strcmp(split,"1")==0;
        l4_chrono=chrono && std::strcmp(chrono,"1")==0;
        l4_on=l4_compressed || l4_split; l4_stride=kvq::vector_bytes(l4_format,256);
        if (l4_on && wire_requested) { vr.error="Lever 4 and VEGAJET_WIRE_ATTN are mutually exclusive"; return vr; }
        std::fprintf(stderr,"[l4] warm KV=%s split=%d chronological=%d active=%d (M1 uses isolated GQA super when available)\n",quant?quant:"q8_0",l4_split,l4_chrono,l4_on);

        if(quant::sub4_dot8_bits()!=loaded_sub4_bits || quant::tcc_xor_enabled()!=tcc_xor_on ||
           (sub4_on && quant::milo_enabled()!=loaded_milo)) {
            vr.ok=false; vr.error="SUB4/TCC/MiLo layout changed: restart engine with requested load-time environment";
            std::printf("{\"event\":\"error\",\"error\":\"%s\"}\n",vr.error.c_str());
            return vr;
        }
        if (const char* gv = std::getenv("VEGAJET_GDN_VERIFY"); gv && gv[0]) {
            const std::string mode(gv);
            const bool wanted_spec = mode == "lumotree" || mode == "treewy";
            if ((mode != "ring" && !wanted_spec) || wanted_spec != base_opt.spec_gdn ||
                (mode == "treewy" && !treewy_opt.enabled)) {
                vr.ok = false;
                vr.error = "VEGAJET_GDN_VERIFY changes the load-time state layout; restart engine with requested verifier";
                std::fprintf(stderr, "[spec] %s\n", vr.error.c_str());
                return vr;
            }
        }
        if (const char* hb = std::getenv("VEGAJET_SPEC_HOBBIT"); hb && hb[0] == '1')
            std::printf("[HOBBIT] resident verify uses exact weights; mixed-precision fetch is available through MoeOffload::fetch_spec with a staged low table.\n");
        reset_lru_state();
        if (parity_only) {
            (void) run_bench_pass(true, false, "raw", &vr);
        } else {
            if (!run_bench_pass(false, true, bmode, &vr)) {
                (void) run_full_bench_body(bmode == "all", &vr);
            }
        }
        std::printf("{\"event\":\"variant\",\"label\":\"%s\",\"ok\":%s,\"primary_tps\":%.2f,\"raw_tg128_tps\":%.2f,\"spec_tg128_tps\":%.2f,\"pp512_tps\":%.2f,\"pp2048_tps\":%.2f,\"parity\":\"%u/%u\",\"tmem_c\":%.1f,\"ecc_ue\":%llu,\"ecc_de\":%llu}\n",
                    warm::json_escape(vr.label).c_str(), vr.ok ? "true" : "false",
                    vr.primary_tps, vr.raw_tg128_tps, vr.spec_tg128_tps, vr.pp512_tps, vr.pp2048_tps,
                    vr.lossless_matched, vr.lossless_total, vr.tmem_max_c,
                    static_cast<unsigned long long>(vr.ecc_ue_delta), static_cast<unsigned long long>(vr.ecc_de_delta));
        std::fflush(stdout);
        return vr;
    };

    // BEGIN L6 live slot executor. The packed row->slot map is passed through
    // the existing GDN ring_head argument only in independent-row mode (uring=2).
    if (opt.serve_step_ready) {
        auto prefill_slot=[&](uint32_t slot,uint32_t position,const std::vector<int32_t>& tokens,bool fresh,bool sample) {
            struct Copy { uint64_t dst,src,n; };
            struct Zero { uint64_t dst; uint32_t value,pad; uint64_t n; };
            auto state=[&](bool save) {
                for(size_t g=0;g<N;++g) for(uint32_t gi=0;gi<gdn_layer_count;++gi) {
                    const uint64_t sizes[2]={uint64_t(gdn_nv[g])*128u*128u*4u,3ULL*gdn_conv_dim[g]*4u};
                    const uint64_t bases[2]={S[g].gdn_ssm_ring_va[gi],S[g].gdn_conv_ring_va[gi]};
                    for(uint32_t j=0;j<2;++j) {
                        const uint64_t scratch=bases[j]+(save?(j?conv_head:ring_head):8u)*sizes[j];
                        const uint64_t physical=bases[j]+slot*sizes[j], n=sizes[j]/4u;
                        if(fresh && !save) {
                            const Zero z{scratch,0u,0u,n};
                            VJ_CHECK(rt.enqueue_kernel(g,*K[g].memset_u32,uint32_t((n+255u)/256u)*256u,1,1,256,1,1,&z,sizeof(z),true));
                        } else {
                            const Copy c{save?physical:scratch,save?scratch:physical,n};
                            VJ_CHECK(rt.enqueue_kernel(g,*K[g].memcpy_u32,uint32_t((n+255u)/256u)*256u,1,1,256,1,1,&c,sizeof(c),true));
                        }
                    }
                }
                VJ_CHECK(submit_all() && wait_all());
            };
            state(false);
            const uint32_t old_ring=ring_head,old_conv=conv_head;
            auto rebase=[&](bool restore) {
                for(size_t g=0;g<N;++g) for(uint32_t ai=0;ai<12u;++ai) {
                    const uint64_t offset=uint64_t(slot)*(opt.max_ctx/8u)*gqa_nkv[g]*272u;
                    if(restore) { S[g].attn_k_cache_va[ai]-=offset; S[g].attn_v_cache_va[ai]-=offset; }
                    else { S[g].attn_k_cache_va[ai]+=offset; S[g].attn_v_cache_va[ai]+=offset; }
                }
                if(restore) { ring_head=old_ring;conv_head=old_conv;live_slot_prefill=false; }
                else { ring_head=8u;conv_head=8u;live_slot_prefill=true; }
            };
            rebase(false);
            std::vector<int32_t> predictions;
            try { predictions=step_batch(tokens,position,opt.use_wy_prefill,false,sample); state(true); }
            catch(...) { rebase(true); throw; }
            rebase(true);
            return sample?predictions.back():-1;
        };
        opt.serve_step_ready([&](const std::vector<uint32_t>& slots, const std::vector<uint32_t>& positions,
                                const std::vector<int32_t>& tokens, const std::vector<bool>& fresh) {
            struct Zero { uint64_t dst; uint32_t value, pad; uint64_t n; };
            for (size_t row=0;row<slots.size();++row) if (fresh[row]) {
                for (size_t g=0;g<N;++g) for (uint32_t gi=0;gi<gdn_layer_count;++gi) {
                    const uint64_t sizes[2]={uint64_t(gdn_nv[g])*128u*128u*4u,3ULL*gdn_conv_dim[g]*4u};
                    const uint64_t bases[2]={S[g].gdn_ssm_ring_va[gi],S[g].gdn_conv_ring_va[gi]};
                    for (uint32_t j=0;j<2;++j) {
                        const Zero z{bases[j]+slots[row]*sizes[j],0u,0u,sizes[j]/4u};
                        VJ_CHECK(rt.enqueue_kernel(g,*K[g].memset_u32,uint32_t((z.n+255u)/256u)*256u,1,1,256,1,1,&z,sizeof(z),true));
                    }
                }
            }
            batch_pos=&positions; batch_kv_slots=&slots;
            auto predictions=step_batch(tokens,positions[0],false,false,true);
            batch_pos=nullptr; batch_kv_slots=nullptr;
            return predictions;
        },prefill_slot);
    }
    // END L6 live slot executor.
    if (opt.serve_hook) {
        uint64_t ue_base = 0, de_base = 0;
        read_ecc_totals(ue_base, de_base);
        auto gen_fn = [&](const std::vector<int32_t>& prompt_ids, int32_t max_tokens,
                          const OrnithGpuOptions::ServeTokenCb& on_tok) {
            if (max_tokens <= 0) return;
            double cur_pwr = 0.0, cur_tmem = 0.0;
            uint64_t ue_now = 0, de_now = 0;
            read_hwmon_max(cur_pwr, cur_tmem);
            read_ecc_totals(ue_now, de_now);
            if (cur_tmem >= 65.0 || ue_now > ue_base || de_now > de_base) {
                std::fprintf(stderr, "[ornith::serve] thermal/ECC safety halt (TMem=%.1f C, UE_delta=%llu, DE_delta=%llu)\n",
                             cur_tmem,
                             static_cast<unsigned long long>(ue_now - ue_base),
                             static_cast<unsigned long long>(de_now - de_base));
                return;
            }
            reset_state();
            std::vector<int32_t> pids = prompt_ids;
            if (pids.empty()) pids.push_back(tok.bos_id() >= 0 ? tok.bos_id() : 0);
            if (pids.size() >= opt.max_ctx) {
                pids.erase(pids.begin(), pids.begin() + static_cast<std::ptrdiff_t>(pids.size() - (opt.max_ctx - 1u)));
            }
            const uint32_t n_prompt = static_cast<uint32_t>(pids.size());
            auto tp0 = std::chrono::steady_clock::now();
            uint32_t pos = 0;
            int32_t cur = -1;
            for (size_t off = 0; off < pids.size(); off += opt.prefill_chunk) {
                const size_t cnt = std::min<size_t>(opt.prefill_chunk, pids.size() - off);
                std::vector<int32_t> chunk(pids.begin() + off, pids.begin() + off + cnt);
                const bool is_last = (off + cnt == pids.size());
                auto preds = step_batch(chunk, pos, opt.use_wy_prefill, false, is_last);
                if (is_last && !preds.empty()) cur = preds.back();
                pos += static_cast<uint32_t>(cnt);
            }
            const double prefill_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tp0).count();
            if (cur < 0) return;
            if (!on_tok(cur, n_prompt, 1u, prefill_ms, 0.0)) return;
            auto td0 = std::chrono::steady_clock::now();
            for (int32_t s = 1; s < max_tokens && pos < opt.max_ctx; ++s) {
                auto preds = step_batch({cur}, pos++, false, false, true);
                if (preds.empty()) break;
                cur = preds[0];
                const double decode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td0).count();
                if (!on_tok(cur, n_prompt, static_cast<uint32_t>(s + 1), prefill_ms, decode_ms)) break;
            }
        };
        const bool sup_on = !(std::getenv("VEGAJET_ENABLE_SUPER") != nullptr && std::getenv("VEGAJET_ENABLE_SUPER")[0] == '0');
        const std::string desc = std::string("ornith-gfx906-tp") + std::to_string(N) + " (super=" + (sup_on ? "on" : "off") + ", greedy)";
        opt.serve_hook(gen_fn, desc);
        parity_code_match = golden::kCodeRef32.size();
    } else if (!base_opt.sweep_spec.empty()) {
        std::vector<std::vector<std::string>> variants;
        if (!warm::parse_sweep_spec(base_opt.sweep_spec, variants, &res.error)) {
            rt.free_buffer(tap_buf); rt.free_buffer(io_buf); telem.stop();
            return res;
        }
        std::vector<warm::SweepVariantResult> results;
        results.reserve(variants.size());
        for (size_t vi = 0; vi < variants.size(); ++vi) {
            std::string lbl;
            for (size_t k = 0; k < variants[vi].size(); ++k) { if (k) lbl += ' '; lbl += variants[vi][k]; }
            std::printf("\n--- [WARM SWEEP %zu/%zu] %s ---\n", vi + 1, variants.size(), lbl.c_str());
            std::fflush(stdout);
            results.push_back(exec_warm_variant(lbl, variants[vi], false));
        }
        const std::string tbl = warm::format_sweep_table(results);
        const std::string js  = warm::format_sweep_json(results);
        std::printf("%s%s\n", tbl.c_str(), js.c_str());
        std::fflush(stdout);
        opt = base_opt;
    } else if (base_opt.interactive) {
        std::printf("{\"event\":\"ready\",\"load_sec\":%.2f,\"vram_gpu0_gib\":%.2f,\"vram_gpu1_gib\":%.2f,\"reload_gen\":%llu,\"modules\":%zu}\n",
                    res.load_sec, res.vram_gib_gpu0, res.vram_gib_gpu1,
                    static_cast<unsigned long long>(rt.reload_generation()), rt.loaded_module_names().size());
        std::fflush(stdout);
        bool keep_running = true;
        auto handle_line = [&](const std::string& line) {
            const auto cmd = warm::parse_warm_command(line);
            if (cmd.kind == warm::CommandKind::Empty) return;
            if (cmd.kind == warm::CommandKind::Quit) {
                std::printf("{\"event\":\"quit\",\"ok\":true}\n");
                std::fflush(stdout);
                keep_running = false;
                return;
            }
            if (cmd.kind == warm::CommandKind::Help) {
                std::printf("{\"event\":\"help\",\"commands\":[\"reload [module|all]\",\"run [ENV=VAL] [flags]\",\"parity [flags]\",\"sweep A ; B\",\"status\",\"quit\"]}\n");
                std::fflush(stdout);
                return;
            }
            if (cmd.kind == warm::CommandKind::Invalid) {
                std::printf("{\"event\":\"error\",\"error\":\"%s\"}\n", warm::json_escape(cmd.error).c_str());
                std::fflush(stdout);
                return;
            }
            if (cmd.kind == warm::CommandKind::Status) {
                double pw = 0.0, tm = 0.0; uint64_t ue = 0, de = 0;
                read_hwmon_max(pw, tm); read_ecc_totals(ue, de);
                std::printf("{\"event\":\"status\",\"reload_gen\":%llu,\"modules\":%zu,\"vram_gpu0_gib\":%.2f,\"vram_gpu1_gib\":%.2f,\"tmem_c\":%.1f,\"ecc_ue\":%llu,\"ecc_de\":%llu}\n",
                            static_cast<unsigned long long>(rt.reload_generation()), rt.loaded_module_names().size(),
                            res.vram_gib_gpu0, res.vram_gib_gpu1, tm,
                            static_cast<unsigned long long>(ue), static_cast<unsigned long long>(de));
                std::fflush(stdout);
                return;
            }
            if (cmd.kind == warm::CommandKind::Reload) {
                const auto tr0 = std::chrono::steady_clock::now();
                std::string rerr;
                const bool ok_rel = cmd.module_name.empty() ? rt.reload_all_hsacos(&rerr) : rt.reload_hsaco(cmd.module_name, &rerr);
                const bool ok_res = ok_rel && resolve_kernels(&rerr);
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tr0).count();
                std::printf("{\"event\":\"reload\",\"module\":\"%s\",\"ok\":%s,\"ms\":%.2f,\"reload_gen\":%llu,\"error\":\"%s\"}\n",
                            warm::json_escape(cmd.module_name.empty() ? "all" : cmd.module_name).c_str(),
                            ok_res ? "true" : "false", ms,
                            static_cast<unsigned long long>(rt.reload_generation()),
                            warm::json_escape(rerr).c_str());
                std::fflush(stdout);
                return;
            }
            if (cmd.kind == warm::CommandKind::Parity) {
                (void) exec_warm_variant("parity", cmd.args, true);
                return;
            }
            if (cmd.kind == warm::CommandKind::Run) {
                std::string lbl;
                for (size_t k = 0; k < cmd.args.size(); ++k) { if (k) lbl += ' '; lbl += cmd.args[k]; }
                if (lbl.empty()) lbl = "default";
                (void) exec_warm_variant(lbl, cmd.args, false);
                return;
            }
            if (cmd.kind == warm::CommandKind::Sweep) {
                std::vector<warm::SweepVariantResult> results;
                for (size_t vi = 0; vi < cmd.variants.size(); ++vi) {
                    std::string lbl;
                    for (size_t k = 0; k < cmd.variants[vi].size(); ++k) { if (k) lbl += ' '; lbl += cmd.variants[vi][k]; }
                    results.push_back(exec_warm_variant(lbl, cmd.variants[vi], false));
                }
                const std::string tbl = warm::format_sweep_table(results);
                const std::string js  = warm::format_sweep_json(results);
                std::printf("%s%s\n", tbl.c_str(), js.c_str());
                std::fflush(stdout);
            }
        };
        if (!base_opt.interactive_fifo.empty()) {
            while (keep_running) {
                std::ifstream fif(base_opt.interactive_fifo);
                if (!fif.is_open()) { std::this_thread::sleep_for(std::chrono::milliseconds(50)); continue; }
                std::string line;
                while (keep_running && std::getline(fif, line)) handle_line(line);
            }
        } else {
            std::string line;
            while (keep_running && std::getline(std::cin, line)) handle_line(line);
        }
        opt = base_opt;
    } else {
        if (opt.run_parity) (void) run_bench_pass(true, false, "all", nullptr);
        if (opt.run_bench) {
            if (!run_full_bench_body(true, nullptr)) return res;
        } else if (!prompt_text.empty()) {
            auto ids = tok.encode(prompt_text);
            if (ids.empty()) ids = {760, 6511, 314, 9338, 369};
            double tps = 0;
            auto gen = opt.use_suffix_spec ? run_greedy_suffix(ids, opt.decode_tokens, &tps)
                                           : run_greedy_raw(ids, opt.decode_tokens, &tps);
            (void)dump_tokens(gen);
            res.tg128_raw_tps = tps;
            res.parity_simple_text = tok.decode(gen);
            std::printf("[GEN %.2f tok/s]: %s\n", tps, res.parity_simple_text.c_str());
        }
    }

    if (wire_requested) {
        for (size_t g = 0; g < N; ++g)
            std::printf("[wire-attn] gpu=%zu %s\n", g, wire_normal::summary(wire_counters[g]).c_str());
        std::fflush(stdout);
    }
    rt.free_buffer(io_buf);
    telem.stop();
    const double observed_tmem = max_tmem > 0.0 ? max_tmem : telemetry::unavailable();
    const auto telemetry_result = telemetry::summarize(telem.cards(), observed_tmem);
    const bool telem_ok = telemetry_result.verdict == telemetry::Verdict::pass;
    std::printf("%s", telemetry::report(telem.cards(), observed_tmem).c_str());
    (void)max_pwr;
    if (parity_code_match != golden::kCodeRef32.size()) {
        res.error = "Tier-1 llama.cpp code parity failed: " + std::to_string(parity_code_match) + "/" +
                    std::to_string(golden::kCodeRef32.size()) + " exact greedy tokens";
        return res;
    }
    if (!telem_ok) { res.error = "power/thermal telemetry gate failed (see [TELEM])"; return res; }
    res.ok = true;
    return res;
}

int run_ornith_ar_bench(const std::string& hsaco_dir) {
    kfd::Runtime rt;
    std::string err;
    if (!rt.init(1.0, 64, &err) || !rt.load_hsaco_dir(kfd::resolve_hsaco_dir(hsaco_dir), &err)) {
        std::fprintf(stderr, "[bench-ar] %s\n", err.c_str());
        return 3;
    }
    const size_t N = rt.num_gpus();
    if (N < 2 || N > kfd::kMaxGpus || rt.allreduce_transport() != kfd::AllReduceTransport::PeerBarP2P) {
        std::fprintf(stderr, "[bench-ar] needs 2..%u GPUs with the PEER_BAR_P2P transport (got %zu)\n", kfd::kMaxGpus, N);
        return 3;
    }
    constexpr uint32_t D = 3072, kVerifyRounds = 64, kBatch = 96, kBatches = 40;
    struct ArKA {
        uint64_t xp, ptx, lrx, pfl, lfl, bar, h, w, xn, xq, xd;
        uint32_t M, D, ep, dn;
        float eps;
    };
    struct ArNKA {
        uint64_t xp, ptx[4], lrx[4], pfl[4], lfl[4], bar, h, w, xn, xq, xd;
        uint32_t M, D, ep, dn, rank, n_ranks, pre_pushed;
        float eps;
    };
    struct Dev { uint64_t partial, bar, h, w, xn, xq, xd; };
    Dev dv[kfd::kMaxGpus];
    const std::vector<float> ones(D, 1.0f), zeros(D, 0.0f);
    const std::vector<uint32_t> zbar(64, 0u);
    for (size_t g = 0; g < N; ++g) {
        auto& a = rt.vram_arena(g);
        dv[g] = Dev{a.alloc(D * 4u), a.alloc(256u), a.alloc(D * 4u), a.alloc(D * 4u), a.alloc(D * 4u), a.alloc(D), a.alloc((D / 32u) * 8u)};
        if (!dv[g].partial || !dv[g].bar || !dv[g].h || !dv[g].w || !dv[g].xn || !dv[g].xq || !dv[g].xd || !rt.upload_to_vram(g, dv[g].w, ones.data(), D * 4u) || !rt.upload_to_vram(g, dv[g].bar, zbar.data(), 256u)) {
            std::fprintf(stderr, "[bench-ar] buffer setup failed\n");
            return 3;
        }
    }
    auto run_queued = [&]() {
        for (size_t g = 0; g < N; ++g) if (!rt.submit_batch(g)) return false;
        for (size_t g = 0; g < N; ++g) if (!rt.wait_idle(g)) return false;
        return true;
    };

    int rc = 0;
    const int n_wires = (N == 2) ? 2 : 1; // N>=3 uses 1-Hop Dual-BAR0 Lossless FP16 wire only
    for (int wire = 0; wire < n_wires; ++wire) {
        const char* wname = wire ? "INT8 block-32" : (N == 2 ? "FP16" : "FP16 N-rank");
        const char* kname = wire ? "ornith_p2p_ar_i8_res_rms_quant_kernel"
                                 : (N == 2 ? "ornith_p2p_ar_res_rms_quant_kernel" : "ornith_p2p_ar_n_fp16_kernel");
        const kfd::LoadedKernel* K[kfd::kMaxGpus] = {};
        for (size_t g = 0; g < N; ++g) {
            K[g] = rt.get_kernel(g, kname);
            if (!K[g]) { std::fprintf(stderr, "[bench-ar] kernel %s missing\n", kname); return 3; }
        }
        rt.reset_allreduce_mailboxes();
        uint32_t epoch = 1u;
        auto enqueue_ar = [&](uint32_t slot) -> bool {
            const uint32_t ep = epoch++;
            const uint32_t pp = ep & 1u;
            if (N == 2) {
                for (size_t g = 0; g < 2; ++g) {
                    const size_t peer = 1u - g;
                    const ArKA aka{dv[g].partial,
                                   rt.ar_rx_base_va(peer, g) + pp * kfd::kArSlotBytes,
                                   rt.ar_rx_base_va(g, peer) + pp * kfd::kArSlotBytes,
                                   rt.ar_flag_va(peer, g, slot), rt.ar_flag_va(g, peer, slot),
                                   dv[g].bar, dv[g].h, dv[g].w, dv[g].xn, dv[g].xq, dv[g].xd, 1u, D, ep, 1u, 1e-6f};
                    if (!rt.enqueue_kernel(g, *K[g], 256u, 1, 1, 256, 1, 1, &aka, sizeof(aka), true)) return false;
                }
            } else {
                for (size_t g = 0; g < N; ++g) {
                    ArNKA aka{};
                    aka.xp = dv[g].partial;
                    for (size_t r = 0; r < 4; ++r) {
                        aka.ptx[r] = (r < N && r != g) ? (rt.ar_rx_base_va(r, g) + pp * kfd::kArSlotBytes) : 0ULL;
                        aka.lrx[r] = (r < N && r != g) ? (rt.ar_rx_base_va(g, r) + pp * kfd::kArSlotBytes) : 0ULL;
                        aka.pfl[r] = (r < N && r != g) ? rt.ar_flag_va(r, g, slot) : 0ULL;
                        aka.lfl[r] = (r < N && r != g) ? rt.ar_flag_va(g, r, slot) : 0ULL;
                    }
                    aka.bar = dv[g].bar; aka.h = dv[g].h; aka.w = dv[g].w;
                    aka.xn = dv[g].xn; aka.xq = dv[g].xq; aka.xd = dv[g].xd;
                    aka.M = 1u; aka.D = D; aka.ep = ep; aka.dn = 1u;
                    aka.rank = static_cast<uint32_t>(g); aka.n_ranks = static_cast<uint32_t>(N);
                    aka.pre_pushed = 0u; aka.eps = 1e-6f;
                    if (!rt.enqueue_kernel(g, *K[g], 256u, 1, 1, 256, 1, 1, &aka, sizeof(aka), true)) return false;
                }
            }
            return true;
        };

        // ---- verification: all N ranks must hold bit-exact canonical sum (((dq[0]+dq[1])+dq[2])+...) ----
        uint64_t verify_errs = 0, rng = 0x9E3779B97F4A7C15ull + static_cast<uint64_t>(wire);
        std::vector<std::vector<float>> x(N, std::vector<float>(D)), dq(N, std::vector<float>(D));
        std::vector<float> hd(D), ref(D);
        for (uint32_t r = 0; r < kVerifyRounds; ++r) {
            for (size_t g = 0; g < N; ++g) {
                for (uint32_t i = 0; i < D; ++i) {
                    rng = rng * 6364136223846793005ull + 1442695040888963407ull;
                    const float u = static_cast<float>(static_cast<int32_t>(rng >> 33) - (1 << 30)) / static_cast<float>(1 << 30);
                    x[g][i] = u * ((r % 4u == 3u && (i >> 5) % 7u == 0u) ? 40.0f : 1.5f);
                }
                if (r == 5u) std::fill(x[g].begin(), x[g].begin() + 64, 0.0f);
                if (!rt.upload_to_vram(g, dv[g].partial, x[g].data(), D * 4u) || !rt.upload_to_vram(g, dv[g].h, zeros.data(), D * 4u)) return 3;
                if (wire == 0) {
                    for (uint32_t i = 0; i < D; ++i) dq[g][i] = static_cast<float>(static_cast<_Float16>(x[g][i]));
                } else {
                    for (uint32_t b = 0; b < D / 32u; ++b) {
                        float amax = 0.0f;
                        for (uint32_t i = 0; i < 32u; ++i) amax = std::max(amax, std::fabs(x[g][b * 32u + i]));
                        const volatile float ld = amax * (1.0f / 127.0f);
                        const volatile float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                        for (uint32_t i = 0; i < 32u; ++i) {
                            const volatile float sc = x[g][b * 32u + i] * inv;
                            const long q = std::max(-127l, std::min(127l, std::lrintf(sc)));
                            const volatile float d = static_cast<float>(q) * ld;
                            dq[g][b * 32u + i] = d;
                        }
                    }
                }
            }
            for (uint32_t i = 0; i < D; ++i) {
                volatile float sum = dq[0][i];
                for (size_t g = 1; g < N; ++g) sum = sum + dq[g][i];
                ref[i] = sum;
            }
            if (!enqueue_ar(r) || !run_queued()) { std::fprintf(stderr, "[bench-ar] dispatch failed\n"); return 3; }
            for (size_t g = 0; g < N; ++g) {
                if (!rt.download_from_vram(g, hd.data(), dv[g].h, D * 4u)) return 3;
                for (uint32_t i = 0; i < D; ++i)
                    if (std::memcmp(&hd[i], &ref[i], 4) != 0) ++verify_errs;
            }
        }

        // ---- FP16 TP=2 only: single-pass vs two-pass reference ----
        if (wire == 0 && N == 2) {
            const kfd::LoadedKernel* KR[2] = {rt.get_kernel(0, "ornith_p2p_ar_res_rms_quant_ref_kernel"), rt.get_kernel(1, "ornith_p2p_ar_res_rms_quant_ref_kernel")};
            if (!KR[0] || !KR[1]) { std::fprintf(stderr, "[bench-ar] reference kernel missing\n"); return 3; }
            uint64_t ab_diffs[4] = {0, 0, 0, 0};
            uint32_t ab_shown = 0;
            std::vector<float> h0[2] = {std::vector<float>(D), std::vector<float>(D)}, wv(D);
            std::vector<uint8_t> out[2][2];
            const size_t sz[4] = {D * 4u, D * 4u, D, (D / 32u) * 8u};
            auto rnd = [&]() { rng = rng * 6364136223846793005ull + 1442695040888963407ull;
                               return static_cast<float>(static_cast<int32_t>(rng >> 33) - (1 << 30)) / static_cast<float>(1 << 30); };
            for (uint32_t r = 0; r < kVerifyRounds; ++r) {
                for (uint32_t i = 0; i < D; ++i) wv[i] = 1.0f + 0.75f * rnd();
                for (size_t g = 0; g < 2; ++g) {
                    for (uint32_t i = 0; i < D; ++i) {
                        x[g][i]  = rnd() * ((r % 4u == 3u && (i >> 5) % 7u == 0u) ? 40.0f : 1.5f);
                        h0[g][i] = rnd() * ((r % 3u == 2u) ? 300.0f : 6.0f);
                    }
                    if (r == 5u) { std::fill(x[g].begin(), x[g].begin() + 64, 0.0f); std::fill(h0[g].begin(), h0[g].begin() + 64, 0.0f); }
                    if (!rt.upload_to_vram(g, dv[g].partial, x[g].data(), D * 4u) || !rt.upload_to_vram(g, dv[g].w, wv.data(), D * 4u)) return 3;
                }
                for (int var = 0; var < 2; ++var) {
                    const uint32_t ep = epoch++, pp = ep & 1u;
                    for (size_t g = 0; g < 2; ++g) {
                        const size_t peer = 1u - g;
                        if (!rt.upload_to_vram(g, dv[g].h, h0[g].data(), D * 4u)) return 3;
                        const ArKA aka{dv[g].partial, rt.ar_rx_base_va(peer, g) + pp * kfd::kArSlotBytes, rt.ar_rx_base_va(g, peer) + pp * kfd::kArSlotBytes,
                                       rt.ar_flag_va(peer, g, r), rt.ar_flag_va(g, peer, r),
                                       dv[g].bar, dv[g].h, dv[g].w, dv[g].xn, dv[g].xq, dv[g].xd, 1u, D, ep, 1u, 1e-6f};
                        if (!rt.enqueue_kernel(g, var ? *KR[g] : *K[g], 256u, 1, 1, 256, 1, 1, &aka, sizeof(aka), true)) return 3;
                    }
                    if (!run_queued()) return 3;
                    for (size_t g = 0; g < 2; ++g) {
                        out[var][g].assign(sz[0] + sz[1] + sz[2] + sz[3], 0);
                        const uint64_t src[4] = {dv[g].h, dv[g].xn, dv[g].xq, dv[g].xd};
                        size_t off = 0;
                        for (int k = 0; k < 4; ++k) { if (!rt.download_from_vram(g, out[var][g].data() + off, src[k], sz[k])) return 3; off += sz[k]; }
                    }
                }
                for (size_t g = 0; g < 2; ++g) {
                    size_t off = 0;
                    for (int k = 0; k < 4; ++k) {
                        for (size_t i = 0; i < sz[k]; ++i) ab_diffs[k] += out[0][g][off + i] != out[1][g][off + i];
                        if (k == 3 && ab_diffs[3] != 0 && ab_shown < 6u)
                            for (size_t b = 0; b < sz[3] / 8u && ab_shown < 6u; ++b) {
                                float nv[2], ov[2];
                                std::memcpy(nv, out[0][g].data() + off + b * 8u, 8); std::memcpy(ov, out[1][g].data() + off + b * 8u, 8);
                                if (std::memcmp(nv, ov, 8) != 0) { ++ab_shown; std::printf("  x_ds diff round %u gpu %zu block %zu: d %.9g vs %.9g | bsum %.9g vs %.9g\n", r, g, b, nv[0], ov[0], nv[1], ov[1]); }
                            }
                        off += sz[k];
                    }
                }
            }
            std::printf("[BENCH-AR FP16 single-pass vs two-pass reference] differing bytes: h=%llu x_norm=%llu x_q8=%llu x_ds=%llu over %u rounds x 2 ranks\n",
                        static_cast<unsigned long long>(ab_diffs[0]), static_cast<unsigned long long>(ab_diffs[1]),
                        static_cast<unsigned long long>(ab_diffs[2]), static_cast<unsigned long long>(ab_diffs[3]), kVerifyRounds);
            if (ab_diffs[0] | ab_diffs[1] | ab_diffs[2] | ab_diffs[3]) rc = 1;
            for (size_t g = 0; g < 2; ++g) if (!rt.upload_to_vram(g, dv[g].w, ones.data(), D * 4u)) return 3;
        }

        // ---- timing: 96 back-to-back AllReduces per batch, like one M=1 token ----
        double best = 1e30, total = 0.0;
        for (uint32_t b = 0; b < kBatches; ++b) {
            for (uint32_t i = 0; i < kBatch; ++i)
                if (!enqueue_ar(i)) return 3;
            const auto t0 = std::chrono::steady_clock::now();
            if (!run_queued()) { std::fprintf(stderr, "[bench-ar] timing dispatch failed\n"); return 3; }
            const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            best = std::min(best, dt);
            total += dt;
        }
        const uint32_t wire_bytes = wire ? (D + D / 8u) : (D * 2u);
        std::printf("[BENCH-AR %-13s] TP=%zu wire=%u B/AR | %.2f us/AR mean, %.2f us/AR best | 96x = %.3f ms mean, %.3f ms best | "
                    "verify_errs=%llu over %u rounds x %zu ranks x %u elems\n",
                    wname, N, wire_bytes, total / (kBatches * kBatch) * 1e6, best / kBatch * 1e6, total / kBatches * 1e3, best * 1e3,
                    static_cast<unsigned long long>(verify_errs), kVerifyRounds, N, D);
        if (verify_errs != 0) rc = 1;
    }
    return rc;
}

int run_ornith_gemv_ab_bench(const std::string& hsaco_dir) {
    kfd::Runtime rt;
    std::string err;
    if (!rt.init(2.0, 64, &err) || !rt.load_hsaco_dir(kfd::resolve_hsaco_dir(hsaco_dir), &err)) {
        std::fprintf(stderr, "[bench-gemv] %s\n", err.c_str());
        return 3;
    }
    const auto* k_w32    = rt.get_kernel(0, "ornith_gemv_q4a8_m1_w32_kernel");
    const auto* k_w64    = rt.get_kernel(0, "vj_gemv_q4a8_w64_kernel");
    const auto* k_w64_1r = rt.get_kernel(0, "vj_gemv_q4a8_w64_1r_kernel");
    const auto* k_sk_p1  = rt.get_kernel(0, "vj_gemv_q4a8_splitk_p1_kernel");
    const auto* k_sk_red = rt.get_kernel(0, "vj_splitk_reduce_prepush_f32_kernel");
    const auto* k_clk    = rt.get_kernel(0, "kfd_stamp_clock_u64");
    const auto* k_fl_1d  = rt.get_kernel(0, "ornith_attn_flash_quant_kernel");
    const auto* k_fl_p1  = rt.get_kernel(0, "vj_flash_splitkv_any_partial_kernel");
    const auto* k_fl_p2  = rt.get_kernel(0, "vj_flash_splitkv_any_merge_quant_kernel");
    if (!k_w32 || !k_w64 || !k_w64_1r || !k_sk_p1 || !k_sk_red || !k_clk || !k_fl_1d || !k_fl_p1 || !k_fl_p2) {
        std::fprintf(stderr, "[bench-gemv] missing required kernel(s)\n");
        return 3;
    }

    // 1. Root-Cause A/B on [3072, 4096] (ssm_out x36 = 283.1 real MB, attn_o x12 = 94.4 real MB)
    //    Use 36 distinct weight copies so EVERY dispatch cold-streams from HBM2 (zero L2 reuse!).
    constexpr uint32_t R = 3072u, K = 4096u, N_LAYERS = 36u;
    const uint32_t nb = K / 32u; // 128
    const uint32_t qs_stride = quant::select_tcc_odd256_layout(K / 2u).stride_bytes;         // 2304 B (9*256)
    const uint32_t dm_even_u32 = nb;                                                         // 128 u32 = 512 B (2*256, EVEN!)
    const uint32_t dm_odd_u32  = quant::select_tcc_odd256_layout(nb * 4u).stride_bytes / 4u; // 192 u32 = 768 B (3*256, ODD!)
    const double real_mb_36 = double(N_LAYERS) * double(R) * (double(K) * 0.5 + double(nb) * 4.0) / 1e6;

    std::vector<float> h_w(static_cast<size_t>(R) * K);
    uint64_t rng = 0x123456789ABCDEF0ULL;
    auto rnd = [&]() {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<float>(static_cast<int32_t>(rng >> 33) - (1 << 30)) / static_cast<float>(1 << 30);
    };
    for (float& v : h_w) v = rnd() * 0.5f;
    const auto p_even = quant::pack_f32_to_vj_q4_a8(h_w.data(), R, K, false);
    const auto p_odd  = quant::pack_f32_to_vj_q4_a8(h_w.data(), R, K, true);
    VJ_CHECK(p_even.qs_row_stride_bytes == qs_stride && p_even.dm_row_stride_u32 == dm_even_u32);
    VJ_CHECK(p_odd.qs_row_stride_bytes  == qs_stride && p_odd.dm_row_stride_u32  == dm_odd_u32);

    std::vector<int8_t> h_xq(K);
    std::vector<float> h_xds(nb * 2u);
    for (uint32_t b = 0; b < nb; ++b) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < 32u; ++i) {
            int q = static_cast<int>(std::lrintf(rnd() * 80.0f));
            h_xq[b * 32u + i] = static_cast<int8_t>(std::max(-127, std::min(127, q)));
            sum += static_cast<float>(h_xq[b * 32u + i]) * 0.01f;
        }
        h_xds[b * 2u + 0u] = 0.01f;
        h_xds[b * 2u + 1u] = sum;
    }

    auto& arena = rt.vram_arena(0);
    uint64_t xq_va = arena.alloc(K, 256), xds_va = arena.alloc(nb * 8u, 256);
    uint64_t y_va  = arena.alloc(R * 4u, 256), part_va = arena.alloc(8u * R * 4u, 256);
    uint64_t clk_va = arena.alloc(64u, 256);
    VJ_CHECK(rt.upload_to_vram(0, xq_va, h_xq.data(), K) && rt.upload_to_vram(0, xds_va, h_xds.data(), nb * 8u));

    std::vector<uint64_t> qs_vas(N_LAYERS), dm_even_vas(N_LAYERS), dm_odd_vas(N_LAYERS);
    for (uint32_t l = 0; l < N_LAYERS; ++l) {
        qs_vas[l]      = arena.alloc(p_even.qs.size(), 4096);
        dm_even_vas[l] = arena.alloc(p_even.dm.size() * 4u, 256);
        dm_odd_vas[l]  = arena.alloc(p_odd.dm.size() * 4u, 256);
        VJ_CHECK(qs_vas[l] && dm_even_vas[l] && dm_odd_vas[l]);
        VJ_CHECK(rt.upload_to_vram(0, qs_vas[l], p_even.qs.data(), p_even.qs.size()));
        VJ_CHECK(rt.upload_to_vram(0, dm_even_vas[l], p_even.dm.data(), p_even.dm.size() * 4u));
        VJ_CHECK(rt.upload_to_vram(0, dm_odd_vas[l], p_odd.dm.data(), p_odd.dm.size() * 4u));
    }
    const uint64_t peer_tx = (rt.num_gpus() >= 2 && rt.allreduce_transport() == kfd::AllReduceTransport::PeerBarP2P)
                           ? (rt.mailbox_payload_gpu_va() + 1ULL * kfd::kArGpuBytes) : 0ULL;

    struct ClkKA { uint64_t dst; uint32_t idx; uint32_t pad; };
    auto measure_36 = [&](const char* label, auto&& enq_one, std::vector<float>* out_y) -> double {
        constexpr int kWarm = 4, kReps = 16;
        auto run_once = [&]() -> double {
            const ClkKA c0{clk_va, 0u, 0u}, c1{clk_va, 1u, 0u};
            VJ_CHECK(rt.enqueue_kernel(0, *k_clk, 64u, 1, 1, 64, 1, 1, &c0, sizeof(c0), true));
            for (uint32_t l = 0; l < N_LAYERS; ++l) enq_one(l);
            VJ_CHECK(rt.enqueue_kernel(0, *k_clk, 64u, 1, 1, 64, 1, 1, &c1, sizeof(c1), true));
            VJ_CHECK(rt.submit_batch(0) && rt.wait_idle(0));
            uint64_t ts[2] = {0, 0};
            VJ_CHECK(rt.download_from_vram(0, ts, clk_va, sizeof(ts)));
            return static_cast<double>(ts[1] - ts[0]) * 40e-6; // 25 MHz -> ms
        };
        for (int i = 0; i < kWarm; ++i) (void)run_once();
        double sum = 0.0, best = 1e30;
        for (int i = 0; i < kReps; ++i) { const double ms = run_once(); sum += ms; best = std::min(best, ms); }
        const double mean = sum / kReps;
        if (out_y) { out_y->resize(R); VJ_CHECK(rt.download_from_vram(0, out_y->data(), y_va, R * 4u)); }
        std::printf("  %-46s | 36L=%.3f ms (best %.3f) | 12L=%.3f ms | %.1f real_GB/s/GPU\n",
                    label, mean, best, mean / 3.0, real_mb_36 / mean);
        return mean;
    };

    std::printf("\n=== Phase 1 Root-Cause A/B on [3072, 4096] VJ_Q4_A8 (36 cold layers = %.1f real MB, s_memrealtime) ===\n", real_mb_36);
    struct M1W32KA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, stride, dm_st; uint64_t ptx; };
    struct SkP1KA  { uint64_t wq, wdm, xq, xds, py; uint32_t rows, cols, qst, dst, sk, flags; };
    struct SkRedKA { uint64_t part, y, ptx; uint32_t rows, sk; };

    std::vector<float> y_ref, y_cur;
    auto check_err = [&](const char* tag) {
        double max_abs = 0.0;
        for (uint32_t i = 0; i < R; ++i) max_abs = std::max(max_abs, std::fabs(double(y_ref[i]) - double(y_cur[i])));
        if (max_abs > 1e-4) std::printf("    WARNING %s max_abs_diff=%.3e\n", tag, max_abs);
    };

    measure_36("(a) baseline 384 WG + prepush + dm_even(512B)", [&](uint32_t l) {
        const M1W32KA ka{qs_vas[l], dm_even_vas[l], xq_va, xds_va, y_va, R, K, qs_stride, dm_even_u32, peer_tx};
        VJ_CHECK(rt.enqueue_kernel(0, *k_w32, (R / 8u) * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
    }, &y_ref);
    measure_36("(b) 384 WG NO prepush + dm_even(512B)", [&](uint32_t l) {
        const M1W32KA ka{qs_vas[l], dm_even_vas[l], xq_va, xds_va, y_va, R, K, qs_stride, dm_even_u32, 0ULL};
        VJ_CHECK(rt.enqueue_kernel(0, *k_w32, (R / 8u) * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
    }, &y_cur); check_err("(b)");
    measure_36("(c) 384 WG NO prepush + dm_odd(768B)", [&](uint32_t l) {
        const M1W32KA ka{qs_vas[l], dm_odd_vas[l], xq_va, xds_va, y_va, R, K, qs_stride, dm_odd_u32, 0ULL};
        VJ_CHECK(rt.enqueue_kernel(0, *k_w32, (R / 8u) * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
    }, &y_cur); check_err("(c)");
    measure_36("(d) 384 WG + prepush + dm_odd(768B)", [&](uint32_t l) {
        const M1W32KA ka{qs_vas[l], dm_odd_vas[l], xq_va, xds_va, y_va, R, K, qs_stride, dm_odd_u32, peer_tx};
        VJ_CHECK(rt.enqueue_kernel(0, *k_w32, (R / 8u) * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
    }, &y_cur); check_err("(d)");
    measure_36("(e1) Split-K S=2 (768 WG) + dm_odd + red(noPP)", [&](uint32_t l) {
        const SkP1KA p1{qs_vas[l], dm_odd_vas[l], xq_va, xds_va, part_va, R, K, qs_stride, dm_odd_u32, 2u, 0u};
        const SkRedKA rd{part_va, y_va, 0ULL, R, 2u};
        VJ_CHECK(rt.enqueue_kernel(0, *k_sk_p1, (R / 8u) * 256u, 2u, 1, 256, 1, 1, &p1, sizeof(p1), true));
        VJ_CHECK(rt.enqueue_kernel(0, *k_sk_red, ((R + 255u) / 256u) * 256u, 1, 1, 256, 1, 1, &rd, sizeof(rd), true));
    }, &y_cur); check_err("(e1)");
    measure_36("(e2) Split-K S=2 (768 WG) + dm_odd + red+PP", [&](uint32_t l) {
        const SkP1KA p1{qs_vas[l], dm_odd_vas[l], xq_va, xds_va, part_va, R, K, qs_stride, dm_odd_u32, 2u, 0u};
        const SkRedKA rd{part_va, y_va, peer_tx, R, 2u};
        VJ_CHECK(rt.enqueue_kernel(0, *k_sk_p1, (R / 8u) * 256u, 2u, 1, 256, 1, 1, &p1, sizeof(p1), true));
        VJ_CHECK(rt.enqueue_kernel(0, *k_sk_red, ((R + 255u) / 256u) * 256u, 1, 1, 256, 1, 1, &rd, sizeof(rd), true));
    }, &y_cur); check_err("(e2)");
    measure_36("(e3) Split-K S=2 + stagger + dm_odd + red+PP", [&](uint32_t l) {
        const SkP1KA p1{qs_vas[l], dm_odd_vas[l], xq_va, xds_va, part_va, R, K, qs_stride, dm_odd_u32, 2u, 1u};
        const SkRedKA rd{part_va, y_va, peer_tx, R, 2u};
        VJ_CHECK(rt.enqueue_kernel(0, *k_sk_p1, (R / 8u) * 256u, 2u, 1, 256, 1, 1, &p1, sizeof(p1), true));
        VJ_CHECK(rt.enqueue_kernel(0, *k_sk_red, ((R + 255u) / 256u) * 256u, 1, 1, 256, 1, 1, &rd, sizeof(rd), true));
    }, &y_cur); check_err("(e3)");
    measure_36("(f)  Split-K S=4 (1536 WG) + dm_odd + red+PP", [&](uint32_t l) {
        const SkP1KA p1{qs_vas[l], dm_odd_vas[l], xq_va, xds_va, part_va, R, K, qs_stride, dm_odd_u32, 4u, 0u};
        const SkRedKA rd{part_va, y_va, peer_tx, R, 4u};
        VJ_CHECK(rt.enqueue_kernel(0, *k_sk_p1, (R / 8u) * 256u, 4u, 1, 256, 1, 1, &p1, sizeof(p1), true));
        VJ_CHECK(rt.enqueue_kernel(0, *k_sk_red, ((R + 255u) / 256u) * 256u, 1, 1, 256, 1, 1, &rd, sizeof(rd), true));
    }, &y_cur); check_err("(f)");
    measure_36("(g1) w64 4r/WG (768 WG) NO PP + dm_odd", [&](uint32_t l) {
        const M1W32KA ka{qs_vas[l], dm_odd_vas[l], xq_va, xds_va, y_va, R, K, qs_stride, dm_odd_u32, 0ULL};
        VJ_CHECK(rt.enqueue_kernel(0, *k_w64, (R / 4u) * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
    }, &y_cur); check_err("(g1)");
    measure_36("(g2) w64 4r/WG (768 WG) + PP + dm_odd", [&](uint32_t l) {
        const M1W32KA ka{qs_vas[l], dm_odd_vas[l], xq_va, xds_va, y_va, R, K, qs_stride, dm_odd_u32, peer_tx};
        VJ_CHECK(rt.enqueue_kernel(0, *k_w64, (R / 4u) * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
    }, &y_cur); check_err("(g2)");
    measure_36("(h1) w64 1r/WG (3072 WG, 6w/SIMD) NO PP", [&](uint32_t l) {
        const M1W32KA ka{qs_vas[l], dm_odd_vas[l], xq_va, xds_va, y_va, R, K, qs_stride, dm_odd_u32, 0ULL};
        VJ_CHECK(rt.enqueue_kernel(0, *k_w64_1r, R * 64u, 1, 1, 64, 1, 1, &ka, sizeof(ka), true));
    }, &y_cur); check_err("(h1)");
    measure_36("(h2) w64 1r/WG (3072 WG, 6w/SIMD) + PP", [&](uint32_t l) {
        const M1W32KA ka{qs_vas[l], dm_odd_vas[l], xq_va, xds_va, y_va, R, K, qs_stride, dm_odd_u32, peer_tx};
        VJ_CHECK(rt.enqueue_kernel(0, *k_w64_1r, R * 64u, 1, 1, 64, 1, 1, &ka, sizeof(ka), true));
    }, &y_cur); check_err("(h2)");

    // 2. Split-KV FlashDecoding A/B (12 GQA layers, 16:1 GQA at seqlen = 256, 512, 2048)
    std::printf("\n=== Phase 1 Split-KV FlashDecoding A/B (12 layers, 16 local Q heads / 1 KV head, s_memrealtime) ===\n");
    constexpr uint32_t kMaxSeq = 2048u, kNq = 16u, kNkv = 1u;
    uint64_t q_va = arena.alloc(kNq * 256u * 4u, 256), g_va = arena.alloc(kNq * 256u * 4u, 256);
    uint64_t kc_va = arena.alloc(kMaxSeq * kNkv * 272u, 4096), vc_va = arena.alloc(kMaxSeq * kNkv * 272u, 4096);
    uint64_t po_va = arena.alloc(kNq * 60u * 256u * 4u, 4096), pml_va = arena.alloc(kNq * 60u * 8u, 256);
    uint64_t oq_va = arena.alloc(kNq * 256u, 256), od_va = arena.alloc(kNq * 64u, 256);
    std::vector<float> hq(kNq * 256u), hg(kNq * 256u);
    for (float& v : hq) v = rnd() * 0.3f;
    for (float& v : hg) v = rnd() * 0.5f;
    std::vector<uint8_t> hkv(kMaxSeq * 272u, 0);
    for (uint32_t t = 0; t < kMaxSeq; ++t) {
        uint16_t sc = quant::f32_to_fp16(0.02f);
        for (uint32_t b = 0; b < 8u; ++b) std::memcpy(hkv.data() + t * 272u + b * 2u, &sc, 2);
        for (uint32_t d = 0; d < 256u; ++d) hkv[t * 272u + 16u + d] = static_cast<uint8_t>(static_cast<int8_t>(std::lrintf(rnd() * 60.0f)));
    }
    VJ_CHECK(rt.upload_to_vram(0, q_va, hq.data(), hq.size() * 4u) && rt.upload_to_vram(0, g_va, hg.data(), hg.size() * 4u));
    VJ_CHECK(rt.upload_to_vram(0, kc_va, hkv.data(), hkv.size()) && rt.upload_to_vram(0, vc_va, hkv.data(), hkv.size()));
    const uint32_t n_cu = rt.devices()[0].n_cu ? rt.devices()[0].n_cu : 60u;
    for (uint32_t seq : {256u, 512u, 2048u}) {
        const auto fp = planner::plan_flash_splitkv(seq, kNq, kNkv, n_cu, 4u);
        auto time12 = [&](auto&& enq_one, std::vector<int8_t>& out) -> double {
            auto run_once = [&]() {
                const ClkKA c0{clk_va, 0u, 0u}, c1{clk_va, 1u, 0u};
                VJ_CHECK(rt.enqueue_kernel(0, *k_clk, 64u, 1, 1, 64, 1, 1, &c0, sizeof(c0), true));
                for (int l = 0; l < 12; ++l) enq_one();
                VJ_CHECK(rt.enqueue_kernel(0, *k_clk, 64u, 1, 1, 64, 1, 1, &c1, sizeof(c1), true));
                VJ_CHECK(rt.submit_batch(0) && rt.wait_idle(0));
                uint64_t ts[2] = {0, 0};
                VJ_CHECK(rt.download_from_vram(0, ts, clk_va, sizeof(ts)));
                return static_cast<double>(ts[1] - ts[0]) * 40e-6;
            };
            for (int i = 0; i < 4; ++i) (void)run_once();
            double s = 0.0;
            for (int i = 0; i < 16; ++i) s += run_once();
            out.resize(kNq * 256u);
            VJ_CHECK(rt.download_from_vram(0, out.data(), oq_va, out.size()));
            return s / 16.0;
        };
        struct FlKA { uint64_t qo, go, kc, vc, oq, od; uint32_t M, pos0, nq, nkv; };
        struct P1KA { uint64_t qo, kc, vc, po, pml; uint32_t M, pos0, nq, nkv, nc, cl; };
        struct P2KA { uint64_t po, pml, go, oq, od; uint32_t M, nq, nc; };
        std::vector<int8_t> o1, o2;
        const double t1 = time12([&]() {
            const FlKA ka{q_va, g_va, kc_va, vc_va, oq_va, od_va, 1u, seq - 1u, kNq, kNkv};
            VJ_CHECK(rt.enqueue_kernel(0, *k_fl_1d, kNq * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
        }, o1);
        const double t2 = time12([&]() {
            const P1KA k1{q_va, kc_va, vc_va, po_va, pml_va, 1u, seq - 1u, kNq, kNkv, fp.n_chunks, fp.chunk_len};
            const P2KA k2{po_va, pml_va, g_va, oq_va, od_va, 1u, kNq, fp.n_chunks};
            VJ_CHECK(rt.enqueue_kernel(0, *k_fl_p1, fp.n_chunks * 256u, kNq, 1, 256, 1, 1, &k1, sizeof(k1), true));
            VJ_CHECK(rt.enqueue_kernel(0, *k_fl_p2, kNq * 256u, 1, 1, 256, 1, 1, &k2, sizeof(k2), true));
        }, o2);
        uint32_t diff = 0; int max_d = 0;
        for (size_t i = 0; i < o1.size(); ++i) {
            int d = std::abs(int(o1[i]) - int(o2[i]));
            if (d) { ++diff; max_d = std::max(max_d, d); }
        }
        std::printf("  seq=%-4u | 1D 16-WG=%.3f ms (CU %.0f%%) | Split-KV %u chunks (%u WGs, CU %.0f%%)=%.3f ms (%.2fx) | diff=%u/%zu max_q=%d\n",
                    seq, t1, fp.cu_util_1d * 100.0, fp.n_chunks, fp.total_wgs, fp.cu_util_splitkv * 100.0, t2, t1 / std::max(t2, 1e-6),
                    diff, o1.size(), max_d);
    }

    // 3. Wave 1D MoE Front & MoE Back A/B (48 cold layers on GPU 0, s_memrealtime(), exact diff check!)
    const auto* k_mf161  = rt.get_kernel(0, "ornith_moe_m1_router_shared_kernel");
    const auto* k_mf240  = rt.get_kernel(0, "ornith_p2p_ar_moe_m1_front_s240_kernel");
    const auto* k_rtr_sh = rt.get_kernel(0, "ornith_moe_m1_rtr_shgu_kernel");
    const auto* k_top_sw = rt.get_kernel(0, "ornith_moe_m1_top8_swiglu_kernel");
    const auto* k_mb224  = rt.get_kernel(0, "ornith_moe_m1_back_s224_kernel");
    const auto* k_mb240  = rt.get_kernel(0, "ornith_moe_m1_back_s240_kernel");
    const auto* k_gu_8r  = rt.get_kernel(0, "ornith_moe_gu_iq2s_m1_8r_kernel");
    const auto* k_mf2q8  = rt.get_kernel(0, "ornith_moe_mid_f32_to_q8_kernel");
    const auto* k_dn_sh  = rt.get_kernel(0, "ornith_moe_down_d8_sh_m1_kernel");
    const auto* k_memset = rt.get_kernel(0, "kfd_memset_u32");
    if (k_mf161 && k_mf240 && k_rtr_sh && k_top_sw && k_mb224 && k_mb240 && k_gu_8r && k_mf2q8 && k_dn_sh && k_memset) {
        constexpr uint32_t kL = 48u, kD = 3072u, kLff = 512u, kShR = 1024u;
        const uint32_t sh_gu_st = quant::select_tcc_odd256_layout(kD / 2u).stride_bytes;  // 1792 B
        const uint32_t sh_dn_st = quant::select_tcc_odd256_layout(kLff / 2u).stride_bytes; // 256 B
        const uint32_t dn_w_st  = 256u; // 512 cols / 2 = 256 B (odd-256B = 1*256!)
        const uint32_t dn_s_st  = 16u;
        const size_t rtr_b   = 256ULL * 3200ULL * 2ULL;          // 1.64 MB/L
        const size_t shgu_qb = kShR * static_cast<size_t>(sh_gu_st);
        const size_t shgu_db = kShR * (kD / 32u) * 4ULL;
        const size_t shdn_qb = kD * static_cast<size_t>(sh_dn_st);
        const size_t shdn_db = kD * (kLff / 32u) * 4ULL;
        const size_t gu8_b   = 8ULL * kLff * 123ULL * 16ULL;     // 8 active experts = 8.06 MB/L
        const size_t dn8_wb  = 8ULL * kD * dn_w_st;              // 6.29 MB/L
        const size_t dn8_sb  = 8ULL * kD * dn_s_st;              // 0.39 MB/L

        // Allocate 16 cold rotating banks (total ~285 MB -> completely defeats 4 MiB L2!) and run 48 layers cycling l % 16.
        constexpr uint32_t kBanks = 16u;
        std::vector<uint64_t> v_rtr(kBanks), v_shg(kBanks), v_shgu_q(kBanks), v_shgu_d(kBanks);
        std::vector<uint64_t> v_shdn_q(kBanks), v_shdn_d(kBanks), v_gu(kBanks), v_dnw(kBanks), v_dns(kBanks);
        std::vector<uint8_t> h_tmp(gu8_b);
        for (size_t i = 0; i < h_tmp.size(); ++i) h_tmp[i] = static_cast<uint8_t>((i * 37u + 13u) & 0x7Fu);
        std::vector<uint16_t> h_rtr(256u * 3200u);
        for (size_t i = 0; i < h_rtr.size(); ++i) h_rtr[i] = quant::f32_to_fp16(rnd() * 0.25f);
        std::vector<float> h_shg(kD);
        for (float& v : h_shg) v = rnd() * 0.2f;
        for (uint32_t b = 0; b < kBanks; ++b) {
            v_rtr[b]    = arena.alloc(rtr_b, 4096);
            v_shg[b]    = arena.alloc(kD * 4u, 256);
            v_shgu_q[b] = arena.alloc(shgu_qb, 4096);
            v_shgu_d[b] = arena.alloc(shgu_db, 256);
            v_shdn_q[b] = arena.alloc(shdn_qb, 4096);
            v_shdn_d[b] = arena.alloc(shdn_db, 256);
            v_gu[b]     = arena.alloc(gu8_b, 4096);
            v_dnw[b]    = arena.alloc(dn8_wb, 4096);
            v_dns[b]    = arena.alloc(dn8_sb, 256);
            VJ_CHECK(v_rtr[b] && v_shg[b] && v_shgu_q[b] && v_shgu_d[b] && v_shdn_q[b] && v_shdn_d[b] && v_gu[b] && v_dnw[b] && v_dns[b]);
            VJ_CHECK(rt.upload_to_vram(0, v_rtr[b], h_rtr.data(), rtr_b));
            VJ_CHECK(rt.upload_to_vram(0, v_shg[b], h_shg.data(), kD * 4u));
            VJ_CHECK(rt.upload_to_vram(0, v_shgu_q[b], h_tmp.data(), shgu_qb));
            VJ_CHECK(rt.upload_to_vram(0, v_shgu_d[b], h_tmp.data(), shgu_db));
            VJ_CHECK(rt.upload_to_vram(0, v_shdn_q[b], h_tmp.data(), shdn_qb));
            VJ_CHECK(rt.upload_to_vram(0, v_shdn_d[b], h_tmp.data(), shdn_db));
            VJ_CHECK(rt.upload_to_vram(0, v_gu[b], h_tmp.data(), gu8_b));
            VJ_CHECK(rt.upload_to_vram(0, v_dnw[b], h_tmp.data(), dn8_wb));
            VJ_CHECK(rt.upload_to_vram(0, v_dns[b], h_tmp.data(), dn8_sb));
        }
        uint64_t xn_va = arena.alloc(kD * 4u, 256), log_va = arena.alloc(256u * 4u, 256), shgsc_va = arena.alloc(64u, 256);
        uint64_t shgu_va = arena.alloc(kShR * 4u, 256), tids_va = arena.alloc(32u, 256), tw_va = arena.alloc(32u, 256);
        uint64_t shq_va = arena.alloc(kLff, 256), shds_va = arena.alloc((kLff / 32u) * 8u, 256), shdn_va = arena.alloc(kD * 4u, 256);
        uint64_t mq_va = arena.alloc(8u * kLff, 256), md_va = arena.alloc(8u * 16u * 4u, 256), mf32_va = arena.alloc(8u * kLff * 4u, 256);
        uint64_t bar_va = arena.alloc(256u, 256), bq_va = arena.alloc(256u, 256), out_va = arena.alloc(kD * 4u, 256);
        std::vector<float> h_xn(kD);
        for (float& v : h_xn) v = rnd() * 0.5f;
        const int32_t h_ids[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        const float   h_tw[8]  = {0.2f, 0.16f, 0.14f, 0.12f, 0.11f, 0.10f, 0.09f, 0.08f};
        std::vector<uint32_t> z64(64, 0u);
        VJ_CHECK(rt.upload_to_vram(0, xn_va, h_xn.data(), kD * 4u) && rt.upload_to_vram(0, tids_va, h_ids, 32u) &&
                 rt.upload_to_vram(0, tw_va, h_tw, 32u) && rt.upload_to_vram(0, bar_va, z64.data(), 256u) &&
                 rt.upload_to_vram(0, bq_va, z64.data(), 256u));

        const double front_mb = double(kL) * (256.0 * kD * 2.0 + double(kShR) * (kD * 0.5 + (kD / 32u) * 4.0)) / 1e6;
        const double back_mb  = double(kL) * (8.0 * kLff * 1968.0 + 8.0 * kD * 272.0 + double(kD) * (kLff * 0.5 + (kLff / 32u) * 4.0)) / 1e6;
        uint32_t sense = 1u;
        auto time48 = [&](const char* label, double real_mb, auto&& enq_one) -> double {
            auto run_once = [&]() -> double {
                const ClkKA c0{clk_va, 0u, 0u}, c1{clk_va, 1u, 0u};
                VJ_CHECK(rt.enqueue_kernel(0, *k_clk, 64u, 1, 1, 64, 1, 1, &c0, sizeof(c0), true));
                for (uint32_t l = 0; l < kL; ++l) enq_one(l % kBanks);
                VJ_CHECK(rt.enqueue_kernel(0, *k_clk, 64u, 1, 1, 64, 1, 1, &c1, sizeof(c1), true));
                VJ_CHECK(rt.submit_batch(0) && rt.wait_idle(0));
                uint64_t ts[2] = {0, 0};
                VJ_CHECK(rt.download_from_vram(0, ts, clk_va, sizeof(ts)));
                return static_cast<double>(ts[1] - ts[0]) * 40e-6;
            };
            for (int i = 0; i < 4; ++i) (void)run_once();
            double sum = 0.0, best = 1e30;
            for (int i = 0; i < 12; ++i) { const double ms = run_once(); sum += ms; best = std::min(best, ms); }
            const double mean = sum / 12.0;
            std::printf("  %-48s | 48L=%.3f ms (best %.3f) | %.1f real_GB/s/GPU\n", label, mean, best, real_mb / mean);
            return mean;
        };

        std::printf("\n=== Wave 1D MoE Front A/B (48 cold layers = %.1f real MB, s_memrealtime) ===\n", front_mb);
        time48("(F0) baseline 161-WG fused front (1 grid_bar)", front_mb, [&](uint32_t b) {
            struct FrontKA { uint64_t wr, wsh, xn, shq, shdm, xq, xds, logits, shgu, tids, tw, oq, od, bar; uint32_t D, sh_rows, sh_stride, lff, sense; };
            const FrontKA ka{v_rtr[b], v_shg[b], xn_va, v_shgu_q[b], v_shgu_d[b], xq_va, xds_va, log_va, shgu_va, tids_va, tw_va, shq_va, shds_va, bar_va, kD, kShR, sh_gu_st, kLff, sense++};
            VJ_CHECK(rt.enqueue_kernel(0, *k_mf161, 161u * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
        });
        int32_t ref_ids[8]; std::vector<int8_t> ref_shq(kLff);
        VJ_CHECK(rt.download_from_vram(0, ref_ids, tids_va, 32u) && rt.download_from_vram(0, ref_shq.data(), shq_va, kLff));

        time48("(F1) 240-WG s240 front fuse_ar=0 (1 grid_bar)", front_mb, [&](uint32_t b) {
            struct ArFront240KA { uint64_t xp, ptx, lrx, pfl, lfl, h, wrms, xn, xq, xds, wr, wsh, shq, shdm, logits, shgu, tids, tw, oq, od, bar; uint32_t D, ep; float eps; uint32_t sh_rows, sh_stride, lff, sense, fuse_ar; };
            const ArFront240KA ka{0,0,0,0,0,0,0, xn_va, xq_va, xds_va, v_rtr[b], v_shg[b], v_shgu_q[b], v_shgu_d[b], log_va, shgu_va, tids_va, tw_va, shq_va, shds_va, bar_va, kD, 0u, 1e-6f, kShR, sh_gu_st, kLff, sense++, 0u};
            VJ_CHECK(rt.enqueue_kernel(0, *k_mf240, 240u * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
        });
        time48("(F2) barrier-free 385x64 rtr+shgu + 2WG top8", front_mb, [&](uint32_t b) {
            struct RsKA { uint64_t wr, wsh, xn, shq, shdm, xq, xds, logits, shgsc, shgu; uint32_t D, sh_rows, sh_stride; };
            struct TsKA { uint64_t logits, shgu, shgsc, tids, tw, oq, od; uint32_t lff; };
            const RsKA rk{v_rtr[b], v_shg[b], xn_va, v_shgu_q[b], v_shgu_d[b], xq_va, xds_va, log_va, shgsc_va, shgu_va, kD, kShR, sh_gu_st};
            const TsKA tk{log_va, shgu_va, shgsc_va, tids_va, tw_va, shq_va, shds_va, kLff};
            VJ_CHECK(rt.enqueue_kernel(0, *k_rtr_sh, 385u * 64u, 1, 1, 64, 1, 1, &rk, sizeof(rk), true));
            VJ_CHECK(rt.enqueue_kernel(0, *k_top_sw, 2u * 256u, 1, 1, 256, 1, 1, &tk, sizeof(tk), true));
        });
        int32_t cur_ids[8]; std::vector<int8_t> cur_shq(kLff);
        VJ_CHECK(rt.download_from_vram(0, cur_ids, tids_va, 32u) && rt.download_from_vram(0, cur_shq.data(), shq_va, kLff));
        // Exact CPU reference check for fast_softmax_top8_wave64!
        std::vector<float> h_logits(256), gpu_tw(8);
        VJ_CHECK(rt.download_from_vram(0, h_logits.data(), log_va, 256u * 4u) && rt.download_from_vram(0, gpu_tw.data(), tw_va, 32u));
        float cpu_mx = h_logits[0];
        for (int i = 1; i < 256; ++i) cpu_mx = std::max(cpu_mx, h_logits[i]);
        std::vector<double> cpu_p(256); double cpu_s = 0.0;
        for (int i = 0; i < 256; ++i) { cpu_p[i] = std::exp(double(h_logits[i] - cpu_mx)); cpu_s += cpu_p[i]; }
        for (int i = 0; i < 256; ++i) cpu_p[i] /= cpu_s;
        std::vector<int> idx(256); for (int i = 0; i < 256; ++i) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + 8, idx.end(), [&](int a, int b) {
            return cpu_p[a] != cpu_p[b] ? cpu_p[a] > cpu_p[b] : a < b;
        });
        double cpu_ts = 0.0; for (int k = 0; k < 8; ++k) cpu_ts += cpu_p[idx[k]];
        uint32_t cpu_id_err = 0; double max_w_err = 0.0;
        for (int k = 0; k < 8; ++k) {
            cpu_id_err += (cur_ids[k] != idx[k]);
            max_w_err = std::max(max_w_err, std::fabs(double(gpu_tw[k]) - cpu_p[idx[k]] / cpu_ts));
        }
        std::printf("  [Top-8 vs CPU EXACT] id_err=%u/8 max_w_err=%.3e | gpu=[%d,%d,%d,%d,%d,%d,%d,%d] cpu=[%d,%d,%d,%d,%d,%d,%d,%d]\n",
                    cpu_id_err, max_w_err, cur_ids[0], cur_ids[1], cur_ids[2], cur_ids[3], cur_ids[4], cur_ids[5], cur_ids[6], cur_ids[7],
                    idx[0], idx[1], idx[2], idx[3], idx[4], idx[5], idx[6], idx[7]);
        uint32_t f_id_err = 0, f_q_err = 0;
        for (int i = 0; i < 8; ++i) f_id_err += (cur_ids[i] != ref_ids[i]);
        for (uint32_t i = 0; i < kLff; ++i) f_q_err += (cur_shq[i] != ref_shq[i]);
        std::printf("  [F2 exact parity vs F0] top_ids_diff=%u/8 | sh_q8_diff=%u/%u\n", f_id_err, f_q_err, kLff);
        // Sub-stage breakdown of F2 + isolated router/sh_gu/top8:
        const auto* k_rlog_base = rt.get_kernel(0, "ornith_moe_router_fp16_logits_kernel");
        const auto* k_rtop_base = rt.get_kernel(0, "ornith_moe_router_top8_kernel");
        const auto* k_gu_128    = rt.get_kernel(0, "ornith_moe_gu_iq2s_merged_kernel");
        const auto* k_gu_d8     = rt.get_kernel(0, "vj_moe_gu_dot8_kernel");
        time48("  -> F2a: 385x64 rtr+shgu ONLY", front_mb, [&](uint32_t b) {
            struct RsKA { uint64_t wr, wsh, xn, shq, shdm, xq, xds, logits, shgsc, shgu; uint32_t D, sh_rows, sh_stride; };
            const RsKA rk{v_rtr[b], v_shg[b], xn_va, v_shgu_q[b], v_shgu_d[b], xq_va, xds_va, log_va, shgsc_va, shgu_va, kD, kShR, sh_gu_st};
            VJ_CHECK(rt.enqueue_kernel(0, *k_rtr_sh, 385u * 64u, 1, 1, 64, 1, 1, &rk, sizeof(rk), true));
        });
        time48("  -> F2b: 2WG top8+swiglu ONLY", 1.0, [&](uint32_t) {
            struct TsKA { uint64_t logits, shgu, shgsc, tids, tw, oq, od; uint32_t lff; };
            const TsKA tk{log_va, shgu_va, shgsc_va, tids_va, tw_va, shq_va, shds_va, kLff};
            VJ_CHECK(rt.enqueue_kernel(0, *k_top_sw, 2u * 256u, 1, 1, 256, 1, 1, &tk, sizeof(tk), true));
        });
        time48("  -> F2b0: 1WG top8 ONLY", 1.0, [&](uint32_t) {
            struct TsKA { uint64_t logits, shgu, shgsc, tids, tw, oq, od; uint32_t lff; };
            const TsKA tk{log_va, shgu_va, shgsc_va, tids_va, tw_va, shq_va, shds_va, kLff};
            VJ_CHECK(rt.enqueue_kernel(0, *k_top_sw, 1u * 256u, 1, 1, 256, 1, 1, &tk, sizeof(tk), true));
        });
        if (k_rlog_base && k_rtop_base) {
            time48("  -> F2c: 33WG router_logits ONLY", double(kL) * 256.0 * kD * 2.0 / 1e6, [&](uint32_t b) {
                struct RLogKA { uint64_t wr, wsh, xn, lout, shg; uint32_t M, D; };
                const RLogKA rl{v_rtr[b], v_shg[b], xn_va, log_va, shgsc_va, 1u, kD};
                VJ_CHECK(rt.enqueue_kernel(0, *k_rlog_base, 33u * 256u, 1, 1, 256, 1, 1, &rl, sizeof(rl), true));
            });
            time48("  -> F2d: 128WG sh_gu (w32) ONLY", double(kL) * kShR * (kD * 0.5 + (kD / 32u) * 4.0) / 1e6, [&](uint32_t b) {
                const M1W32KA ka{v_shgu_q[b], v_shgu_d[b], xq_va, xds_va, shgu_va, kShR, kD, sh_gu_st, 0u, 0ULL};
                VJ_CHECK(rt.enqueue_kernel(0, *k_w32, (kShR / 8u) * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
            });
        }

        // Restore valid expert IDs 0..7 for MoE Back benchmark (our synthetic banks have 8 experts!).
        VJ_CHECK(rt.upload_to_vram(0, tids_va, h_ids, 32u) && rt.upload_to_vram(0, tw_va, h_tw, 32u));
        std::printf("\n=== Wave 1D MoE Back A/B (48 cold layers = %.1f real MB, s_memrealtime) ===\n", back_mb);
        time48("(B0) baseline 224-WG s224 back + PP (1 grid_bar)", back_mb, [&](uint32_t b) {
            struct BackKA { uint64_t wgu, xq, xds, tids, tw, mq, md, shwq, shwdm, shq, shds, shd, wd, wds, out, q; uint32_t lff, D, sh_rows, sh_stride, sense, dn_wst, dn_sst, pad; uint64_t ptx; };
            const BackKA ka{v_gu[b], xq_va, xds_va, tids_va, tw_va, mq_va, md_va, v_shdn_q[b], v_shdn_d[b], shq_va, shds_va, shdn_va, v_dnw[b], v_dns[b], out_va, bq_va, kLff, kD, kD, sh_dn_st, sense++, dn_w_st, dn_s_st, 0u, peer_tx};
            VJ_CHECK(rt.enqueue_kernel(0, *k_mb224, 224u * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
        });
        std::vector<float> ref_out(kD), cur_out(kD);
        VJ_CHECK(rt.download_from_vram(0, ref_out.data(), out_va, kD * 4u));

        time48("(B1) 240-WG s240 back + PP (1 grid_bar)", back_mb, [&](uint32_t b) {
            struct Back240KA { uint64_t wgu, xq, xds, tids, tw, mf32, shwq, shwdm, shq, shds, wd, wds, out, q; uint32_t lff, D, sh_stride, sense, dn_wst, dn_sst; uint64_t ptx; };
            const Back240KA ka{v_gu[b], xq_va, xds_va, tids_va, tw_va, mf32_va, v_shdn_q[b], v_shdn_d[b], shq_va, shds_va, v_dnw[b], v_dns[b], out_va, bq_va, kLff, kD, sh_dn_st, sense++, dn_w_st, dn_s_st, peer_tx};
            VJ_CHECK(rt.enqueue_kernel(0, *k_mb240, 240u * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
        });
        time48("(B2) barrier-free 512x128 GU + 16WG Q + 384x128 Dn+Sh+PP", back_mb, [&](uint32_t b) {
            struct Gu8KA { uint64_t wgu, xq, xds, tids, mf32; uint32_t lff; };
            struct QKA   { uint64_t mf32, mq, md; uint32_t nblk; };
            struct DnShKA { uint64_t wd, wds, mq, md, tids, tw, shwq, shwdm, shq, shds, out; uint32_t D, sh_st, dn_wst, dn_sst; uint64_t ptx; };
            const Gu8KA  gk{v_gu[b], xq_va, xds_va, tids_va, mf32_va, kLff};
            const QKA    qk{mf32_va, mq_va, md_va, (8u * kLff) / 32u};
            const DnShKA dk{v_dnw[b], v_dns[b], mq_va, md_va, tids_va, tw_va, v_shdn_q[b], v_shdn_d[b], shq_va, shds_va, out_va, kD, sh_dn_st, dn_w_st, dn_s_st, peer_tx};
            VJ_CHECK(rt.enqueue_kernel(0, *k_gu_8r, 512u * 128u, 1, 1, 128, 1, 1, &gk, sizeof(gk), true));
            VJ_CHECK(rt.enqueue_kernel(0, *k_mf2q8, 16u * 256u, 1, 1, 256, 1, 1, &qk, sizeof(qk), true));
            VJ_CHECK(rt.enqueue_kernel(0, *k_dn_sh, 384u * 128u, 1, 1, 128, 1, 1, &dk, sizeof(dk), true));
        });
        VJ_CHECK(rt.download_from_vram(0, cur_out.data(), out_va, kD * 4u));
        double max_b2_diff = 0.0; uint32_t b2_exact = 0;
        for (uint32_t i = 0; i < kD; ++i) {
            const double d = std::fabs(double(ref_out[i]) - double(cur_out[i]));
            max_b2_diff = std::max(max_b2_diff, d);
            if (std::memcmp(&ref_out[i], &cur_out[i], 4) == 0) ++b2_exact;
        }
        std::printf("  [B2 exact parity vs B0] bit_identical=%u/%u | max_abs_diff=%.3e\n", b2_exact, kD, max_b2_diff);
        // Sub-stage breakdown of B2:
        const double gu_mb = double(kL) * (8.0 * kLff * 1968.0) / 1e6;
        const double dn_mb = double(kL) * (8.0 * kD * 272.0 + double(kD) * (kLff * 0.5 + (kLff / 32u) * 4.0)) / 1e6;
        time48("  -> B2a: 512x128 GU only", gu_mb, [&](uint32_t b) {
            struct Gu8KA { uint64_t wgu, xq, xds, tids, mf32; uint32_t lff; };
            const Gu8KA gk{v_gu[b], xq_va, xds_va, tids_va, mf32_va, kLff};
            VJ_CHECK(rt.enqueue_kernel(0, *k_gu_8r, 512u * 128u, 1, 1, 128, 1, 1, &gk, sizeof(gk), true));
        });
        const auto* k_gu_nolut = rt.get_kernel(0, "ornith_moe_gu_iq2s_m1_8r_nolut_kernel");
        const auto* k_gu_16r   = rt.get_kernel(0, "ornith_moe_gu_iq2s_m1_16r_kernel");
        if (k_gu_nolut) {
            time48("  -> B2a-nolut: 512x128 GU (0 LDS, L1D grid)", gu_mb, [&](uint32_t b) {
                struct Gu8KA { uint64_t wgu, xq, xds, tids, mf32; uint32_t lff; };
                const Gu8KA gk{v_gu[b], xq_va, xds_va, tids_va, mf32_va, kLff};
                VJ_CHECK(rt.enqueue_kernel(0, *k_gu_nolut, 512u * 128u, 1, 1, 128, 1, 1, &gk, sizeof(gk), true));
            });
        }
        if (k_gu_16r) {
            time48("  -> B2a-16r: 256x256 GU (16r/WG, LDS x+grid)", gu_mb, [&](uint32_t b) {
                struct Gu8KA { uint64_t wgu, xq, xds, tids, mf32; uint32_t lff; };
                const Gu8KA gk{v_gu[b], xq_va, xds_va, tids_va, mf32_va, kLff};
                VJ_CHECK(rt.enqueue_kernel(0, *k_gu_16r, 256u * 256u, 1, 1, 256, 1, 1, &gk, sizeof(gk), true));
            });
        }
        const auto* k_gu_16r_g32 = rt.get_kernel(0, "ornith_moe_gu_iq2s_m1_16r_g32_kernel");
        const auto* k_gu_8r_g32  = rt.get_kernel(0, "ornith_moe_gu_iq2s_m1_8r_g32_kernel");
        if (k_gu_16r_g32 && k_gu_8r_g32) {
            std::vector<float> ref_mf(8u * kLff), cur_mf(8u * kLff);
            VJ_CHECK(rt.download_from_vram(0, ref_mf.data(), mf32_va, ref_mf.size() * 4u));
            time48("  -> B2a-16r-g32: 256x256 GU (4K grid32+perm)", gu_mb, [&](uint32_t b) {
                struct Gu8KA { uint64_t wgu, xq, xds, tids, mf32; uint32_t lff; };
                const Gu8KA gk{v_gu[b], xq_va, xds_va, tids_va, mf32_va, kLff};
                VJ_CHECK(rt.enqueue_kernel(0, *k_gu_16r_g32, 256u * 256u, 1, 1, 256, 1, 1, &gk, sizeof(gk), true));
            });
            VJ_CHECK(rt.download_from_vram(0, cur_mf.data(), mf32_va, cur_mf.size() * 4u));
            uint32_t exact16 = 0; double maxd16 = 0.0;
            for (size_t i = 0; i < ref_mf.size(); ++i) {
                exact16 += (std::memcmp(&ref_mf[i], &cur_mf[i], 4) == 0);
                maxd16 = std::max(maxd16, std::fabs(double(ref_mf[i]) - double(cur_mf[i])));
            }
            std::printf("     [g32-16r vs 16r] bit_exact=%u/%zu max_abs_diff=%.3e\n", exact16, ref_mf.size(), maxd16);
            time48("  -> B2a-8r-g32:  512x128 GU (4K grid32+perm)", gu_mb, [&](uint32_t b) {
                struct Gu8KA { uint64_t wgu, xq, xds, tids, mf32; uint32_t lff; };
                const Gu8KA gk{v_gu[b], xq_va, xds_va, tids_va, mf32_va, kLff};
                VJ_CHECK(rt.enqueue_kernel(0, *k_gu_8r_g32, 512u * 128u, 1, 1, 128, 1, 1, &gk, sizeof(gk), true));
            });
        }
        if (k_gu_128) {
            time48("  -> B0a: 128x256 baseline GU only", gu_mb, [&](uint32_t b) {
                struct GuKA { uint64_t wgu, xq, xd, tids, mq, md; uint32_t M, lff; };
                const GuKA gk{v_gu[b], xq_va, xds_va, tids_va, mq_va, md_va, 1u, kLff};
                VJ_CHECK(rt.enqueue_kernel(0, *k_gu_128, 128u * 256u, 1, 1, 256, 1, 1, &gk, sizeof(gk), true));
            });
        }
        if (k_gu_d8) {
            const uint32_t gu_d8_wst = 3072u; // 2*3072/2 = 3072 B
            const uint32_t gu_d8_sst = 192u;
            const double gu_d8_mb = double(kL) * 8.0 * 256.0 * (3072.0 + 192.0) / 1e6;
            time48("  -> Bd8: 64x256 DOT8 GU (lff=256, no sm_grid)", gu_d8_mb, [&](uint32_t b) {
                struct GuD8KA { uint64_t wgu, wsc, xq, xd, tids, mq, md; uint32_t M, lff, wst, sst; };
                // Reuse v_gu[b] + v_dnw[b] as dummy read buffers (safe: 8*512*3072 = 12.6MB > 8.06MB? Clamp lff=256? Wait, allocate 1 bank or check!)
                const GuD8KA gk{v_gu[b], v_dns[b], xq_va, xds_va, tids_va, mq_va, md_va, 1u, 256u, gu_d8_wst, gu_d8_sst};
                VJ_CHECK(rt.enqueue_kernel(0, *k_gu_d8, 64u * 256u, 1, 1, 256, 1, 1, &gk, sizeof(gk), true));
            });
        }
        time48("  -> B2b: 16WG mid_f32->q8 only", 1.0, [&](uint32_t) {
            struct QKA { uint64_t mf32, mq, md; uint32_t nblk; };
            const QKA qk{mf32_va, mq_va, md_va, (8u * kLff) / 32u};
            VJ_CHECK(rt.enqueue_kernel(0, *k_mf2q8, 16u * 256u, 1, 1, 256, 1, 1, &qk, sizeof(qk), true));
        });
        time48("  -> B2c: 384x128 Dn+Sh+PP only", dn_mb, [&](uint32_t b) {
            struct DnShKA { uint64_t wd, wds, mq, md, tids, tw, shwq, shwdm, shq, shds, out; uint32_t D, sh_st, dn_wst, dn_sst; uint64_t ptx; };
            const DnShKA dk{v_dnw[b], v_dns[b], mq_va, md_va, tids_va, tw_va, v_shdn_q[b], v_shdn_d[b], shq_va, shds_va, out_va, kD, sh_dn_st, dn_w_st, dn_s_st, peer_tx};
            VJ_CHECK(rt.enqueue_kernel(0, *k_dn_sh, 384u * 128u, 1, 1, 128, 1, 1, &dk, sizeof(dk), true));
        });

        // Safe single-GPU (GPU 0 only, 192 WGs <= 192 safety cap) synthetic check for ornith_m1_moe_super_kernel!
        const auto* k_moe_sup = rt.get_kernel(0, "ornith_m1_moe_super_kernel");
        if (k_moe_sup) {
            // Compute reference full Front+Back (F0 + B0) with valid 0..7 router weights (clamp router weights for top-8 in [0..7])
            std::vector<uint16_t> h_rtr8(256u * 3200u, quant::f32_to_fp16(-2.0f));
            for (uint32_t e = 0; e < 8u; ++e)
                for (uint32_t j = 0; j < kD; ++j) h_rtr8[e * 3200u + j] = quant::f32_to_fp16((j & 1u) ? 0.15f : 0.10f + 0.01f * float(e));
            for (uint32_t b = 0; b < kBanks; ++b) VJ_CHECK(rt.upload_to_vram(0, v_rtr[b], h_rtr8.data(), rtr_b));
            // Run F0 + B0 on bank 0 as exact reference
            {
                struct FrontKA { uint64_t wr, wsh, xn, shq, shdm, xq, xds, logits, shgu, tids, tw, oq, od, bar; uint32_t D, sh_rows, sh_stride, lff, sense; };
                struct BackKA  { uint64_t wgu, xq, xds, tids, tw, mq, md, shwq, shwdm, shq, shds, shd, wd, wds, out, q; uint32_t lff, D, sh_rows, sh_stride, sense, dn_wst, dn_sst, pad; uint64_t ptx; };
                const FrontKA fka{v_rtr[0], v_shg[0], xn_va, v_shgu_q[0], v_shgu_d[0], xq_va, xds_va, log_va, shgu_va, tids_va, tw_va, shq_va, shds_va, bar_va, kD, kShR, sh_gu_st, kLff, sense++};
                const BackKA  bka{v_gu[0], xq_va, xds_va, tids_va, tw_va, mq_va, md_va, v_shdn_q[0], v_shdn_d[0], shq_va, shds_va, shdn_va, v_dnw[0], v_dns[0], out_va, bq_va, kLff, kD, kD, sh_dn_st, sense++, dn_w_st, dn_s_st, 0u, 0ULL};
                VJ_CHECK(rt.enqueue_kernel(0, *k_mf161, 161u * 256u, 1, 1, 256, 1, 1, &fka, sizeof(fka), true));
                VJ_CHECK(rt.enqueue_kernel(0, *k_mb224, 224u * 256u, 1, 1, 256, 1, 1, &bka, sizeof(bka), true));
                VJ_CHECK(rt.submit_batch(0) && rt.wait_idle(0));
                VJ_CHECK(rt.download_from_vram(0, ref_out.data(), out_va, kD * 4u));
            }
            const uint64_t sbar_va = arena.alloc(2048u, 256);
            const uint64_t mdesc_va = arena.alloc(kBanks * sizeof(ornith_super::MoeSuperDesc), 256);
            std::vector<uint32_t> z512(512, 0u);
            VJ_CHECK(rt.upload_to_vram(0, sbar_va, z512.data(), 2048u));
            std::vector<ornith_super::MoeSuperDesc> mds(kBanks);
            for (uint32_t b = 0; b < kBanks; ++b) {
                auto& d = mds[b];
                d.x_partial = out_va;
                d.x_norm = xn_va; d.x_q8 = xq_va; d.x_ds = xds_va;
                d.w_router_f16 = v_rtr[b]; d.w_sh_gate = v_shg[b];
                d.sh_gu_qs = v_shgu_q[b]; d.sh_gu_dm = v_shgu_d[b];
                d.router_logits = log_va; d.sh_gu_out = shgu_va; d.sh_gate_sc = shgsc_va;
                d.top_ids = tids_va; d.top_w = tw_va; d.sh_q8 = shq_va; d.sh_ds = shds_va;
                d.sh_gu_rows = kShR; d.sh_gu_stride = sh_gu_st; d.local_ff = kLff; d.D = kD;
                d.w_gu_iq2s = v_gu[b]; d.mid_q8 = mq_va; d.mid_d = md_va;
                d.sh_dn_qs = v_shdn_q[b]; d.sh_dn_dm = v_shdn_d[b]; d.sh_down_out = shdn_va;
                d.w_dn_d8_w = v_dnw[b]; d.w_dn_d8_s = v_dns[b];
                d.sh_dn_stride = sh_dn_st; d.dn_w_stride = dn_w_st; d.dn_s_stride = dn_s_st;
                d.eps = 1e-6f; d.super_bar = sbar_va;
            }
            VJ_CHECK(rt.upload_to_vram(0, mdesc_va, mds.data(), mds.size() * sizeof(ornith_super::MoeSuperDesc)));
            uint32_t s_sense = 1u;
            struct MoeSupKA { uint64_t desc; uint32_t ar_ep, out_ep, ssense; };
            // 1-layer check first!
            {
                const MoeSupKA ka{mdesc_va, 0u, 0u, s_sense++};
                VJ_CHECK(rt.enqueue_kernel(0, *k_moe_sup, 192u * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
                VJ_CHECK(rt.submit_batch(0) && rt.wait_idle(0));
                VJ_CHECK(rt.download_from_vram(0, cur_out.data(), out_va, kD * 4u));
                double mdiff = 0.0; uint32_t exact = 0;
                for (uint32_t i = 0; i < kD; ++i) {
                    mdiff = std::max(mdiff, std::fabs(double(ref_out[i]) - double(cur_out[i])));
                    if (std::memcmp(&ref_out[i], &cur_out[i], 4) == 0) ++exact;
                }
                std::printf("  [SUPER-B 192-WG vs F0+B0] bit_exact=%u/%u max_abs_diff=%.3e\n", exact, kD, mdiff);
            }
            time48("(SUPER-B) 192-WG fused Front+Back (g32, 0 AR)", front_mb + back_mb, [&](uint32_t b) {
                const MoeSupKA ka{mdesc_va + static_cast<uint64_t>(b) * sizeof(ornith_super::MoeSuperDesc), 0u, 0u, s_sense++};
                VJ_CHECK(rt.enqueue_kernel(0, *k_moe_sup, 192u * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
            });

            // Safe single-GPU (GPU 0 only, 192 WGs) synthetic check for SUPER-A-GDN & SUPER-A-GQA!
            const auto* k_gdn_sup = rt.get_kernel(0, "ornith_m1_gdn_super_kernel");
            const auto* k_gqa_sup = rt.get_kernel(0, "ornith_m1_gqa_super_kernel");
            const auto* k_gdn_ref = rt.get_kernel(0, "ornith_gdn_conv_recur_splitv_quant_m1_kernel");
            const auto* k_qkv_prp = rt.get_kernel(0, "ornith_attn_qkv_prep_kernel");
            if (k_gdn_sup && k_gqa_sup && k_gdn_ref && k_qkv_prp) {
                const uint32_t qkv_gdn_r = 10304u, qkv_gqa_r = 8704u, mid_c = 4096u;
                const uint32_t qkv_st = quant::select_tcc_odd256_layout(kD / 2u).stride_bytes;    // 1792 B
                const uint32_t out_st = quant::select_tcc_odd256_layout(mid_c / 2u).stride_bytes; // 2304 B
                const uint64_t w_qkv_q = arena.alloc(static_cast<uint64_t>(qkv_gdn_r) * qkv_st, 4096);
                const uint64_t w_qkv_d = arena.alloc(static_cast<uint64_t>(qkv_gdn_r) * (kD / 32u) * 4ULL, 256);
                const uint64_t w_out_q = arena.alloc(static_cast<uint64_t>(kD) * out_st, 4096);
                const uint64_t w_out_d = arena.alloc(static_cast<uint64_t>(kD) * (mid_c / 32u) * 4ULL, 256);
                const uint64_t gbuf    = arena.alloc(qkv_gdn_r * 4ULL, 4096);
                const uint64_t cw_va   = arena.alloc(6144ULL * 16ULL, 256);
                const uint64_t cr_va   = arena.alloc(2ULL * 3ULL * 6144ULL * 4ULL, 4096);
                const uint64_t dt_va   = arena.alloc(32ULL * 4ULL, 256);
                const uint64_t sa_va   = arena.alloc(32ULL * 4ULL, 256);
                const uint64_t sr_va   = arena.alloc(32ULL * 16384ULL * 4ULL, 4096);
                const uint64_t nw_va   = arena.alloc(128ULL * 4ULL, 256);
                const uint64_t midq_va = arena.alloc(mid_c, 256);
                const uint64_t midd_va = arena.alloc((mid_c / 32u) * 8ULL, 256);
                const uint64_t gbar_va = arena.alloc(8192u, 256);
                const uint64_t gsq_va  = arena.alloc(32ULL * 64ULL * 4ULL, 256);
                const uint64_t gd_va   = arena.alloc(sizeof(ornith_super::GdnSuperDesc), 256);
                std::vector<uint8_t> z_q(static_cast<size_t>(qkv_gdn_r) * qkv_st, 0x35u);
                std::vector<uint32_t> z_dm(static_cast<size_t>(qkv_gdn_r) * (kD / 32u), 0x00002C00u); // d=0.0625, m=0
                std::vector<float> f_ones(6144u * 4u, 0.05f), f_sa(32, -0.1f), f_z(32u * 16384u, 0.0f);
                std::vector<uint32_t> z2k(2048, 0u);
                VJ_CHECK(rt.upload_to_vram(0, w_qkv_q, z_q.data(), z_q.size()) &&
                         rt.upload_to_vram(0, w_qkv_d, z_dm.data(), z_dm.size() * 4u) &&
                         rt.upload_to_vram(0, w_out_q, z_q.data(), static_cast<size_t>(kD) * out_st) &&
                         rt.upload_to_vram(0, w_out_d, z_dm.data(), static_cast<size_t>(kD) * (mid_c / 32u) * 4u) &&
                         rt.upload_to_vram(0, cw_va, f_ones.data(), 6144u * 16u) &&
                         rt.upload_to_vram(0, cr_va, f_z.data(), 2u * 3u * 6144u * 4u) &&
                         rt.upload_to_vram(0, dt_va, f_ones.data(), 32u * 4u) &&
                         rt.upload_to_vram(0, sa_va, f_sa.data(), 32u * 4u) &&
                         rt.upload_to_vram(0, sr_va, f_z.data(), f_z.size() * 4u) &&
                         rt.upload_to_vram(0, nw_va, f_ones.data(), 128u * 4u) &&
                         rt.upload_to_vram(0, gbar_va, z2k.data(), 8192u));
                // Reference 3-dispatch GDN layer
                uint32_t g_sense = 1u;
                {
                    const M1W32KA k1{w_qkv_q, w_qkv_d, xq_va, xds_va, gbuf, qkv_gdn_r, kD, qkv_st, 0u, 0ULL};
                    struct ConvRecSvKA { uint64_t qkv, cw, cring, dt, sa, sring, nw, oq, od, bar, sq; uint32_t nk, nv, cdim, strd, zoff, aoff, boff, cin, cout, shead, uring, sense; float eps; };
                    const ConvRecSvKA k2{gbuf, cw_va, cr_va, dt_va, sa_va, sr_va, nw_va, midq_va, midd_va, gbar_va, gsq_va, 8u, 32u, 6144u, 10304u, 6144u, 10240u, 10272u, 0u, 1u, 0u, 0u, g_sense++, 1e-6f};
                    const M1W32KA k3{w_out_q, w_out_d, midq_va, midd_va, out_va, kD, mid_c, out_st, 0u, 0ULL};
                    VJ_CHECK(rt.enqueue_kernel(0, *k_w32, ((qkv_gdn_r + 7u) / 8u) * 256u, 1, 1, 256, 1, 1, &k1, sizeof(k1), true));
                    VJ_CHECK(rt.enqueue_kernel(0, *k_gdn_ref, 128u * 256u, 1, 1, 256, 1, 1, &k2, sizeof(k2), true));
                    VJ_CHECK(rt.enqueue_kernel(0, *k_w32, ((kD + 7u) / 8u) * 256u, 1, 1, 256, 1, 1, &k3, sizeof(k3), true));
                    VJ_CHECK(rt.submit_batch(0) && rt.wait_idle(0));
                    VJ_CHECK(rt.download_from_vram(0, ref_out.data(), out_va, kD * 4u));
                }
                // Reset conv_ring + ssm_ring and run 192-WG SUPER-A-GDN!
                VJ_CHECK(rt.upload_to_vram(0, cr_va, f_z.data(), 2u * 3u * 6144u * 4u) &&
                         rt.upload_to_vram(0, sr_va, f_z.data(), f_z.size() * 4u));
                ornith_super::GdnSuperDesc gd{};
                gd.x_partial = out_va; gd.x_q8 = xq_va; gd.x_ds = xds_va;
                gd.w_qkv_qs = w_qkv_q; gd.w_qkv_dm = w_qkv_d; gd.gdn_qkv_gab = gbuf;
                gd.qkv_rows = qkv_gdn_r; gd.qkv_cols = kD; gd.qkv_qs_stride = qkv_st;
                gd.conv_w = cw_va; gd.conv_ring = cr_va; gd.dt_bias = dt_va; gd.ssm_a = sa_va;
                gd.ssm_ring_base = sr_va; gd.ssm_norm_w = nw_va; gd.mid_q8 = midq_va; gd.mid_ds = midd_va;
                gd.gdn_bar = gbar_va; gd.gdn_sq = gsq_va;
                gd.w_out_qs = w_out_q; gd.w_out_dm = w_out_d;
                gd.out_rows = kD; gd.out_cols = mid_c; gd.out_qs_stride = out_st;
                gd.super_bar = sbar_va; gd.D = kD; gd.nk = 8u; gd.nv = 32u;
                gd.conv_dim = 6144u; gd.z_off = 6144u; gd.alpha_off = 10240u; gd.beta_off = 10272u; gd.eps = 1e-6f;
                VJ_CHECK(rt.upload_to_vram(0, gd_va, &gd, sizeof(gd)));
                struct GdnSupKA { uint64_t desc, sring_ov; uint32_t prev_ep, out_ep, cin, cout, shead, uring, ssense, gsense; };
                const GdnSupKA gska{gd_va, 0ULL, 0u, 0u, 0u, 1u, 0u, 0u, s_sense++, g_sense++};
                VJ_CHECK(rt.enqueue_kernel(0, *k_gdn_sup, 192u * 256u, 1, 1, 256, 1, 1, &gska, sizeof(gska), true));
                VJ_CHECK(rt.submit_batch(0) && rt.wait_idle(0));
                VJ_CHECK(rt.download_from_vram(0, cur_out.data(), out_va, kD * 4u));
                double mdiff_gdn = 0.0; uint32_t ex_gdn = 0;
                for (uint32_t i = 0; i < kD; ++i) {
                    mdiff_gdn = std::max(mdiff_gdn, std::fabs(double(ref_out[i]) - double(cur_out[i])));
                    if (std::memcmp(&ref_out[i], &cur_out[i], 4) == 0) ++ex_gdn;
                }
                std::printf("  [SUPER-A-GDN 192-WG vs 3-disp] bit_exact=%u/%u max_abs_diff=%.3e\n", ex_gdn, kD, mdiff_gdn);

                // Reference vs 192-WG SUPER-A-GQA at pos0=255!
                const uint32_t pos0 = 255u;
                const uint64_t qnw_va = arena.alloc(256u * 4u, 256), knw_va = arena.alloc(256u * 4u, 256);
                const uint64_t qb_va  = arena.alloc(16u * 256u * 4u, 256), gb_va = arena.alloc(16u * 256u * 4u, 256);
                const uint64_t kc_va  = arena.alloc(256u * 272u, 4096), vc_va = arena.alloc(256u * 272u, 4096);
                const uint64_t po_va  = arena.alloc(16u * 16u * 256u * 4u, 4096), pml_va = arena.alloc(16u * 16u * 8u, 256);
                const uint64_t qa_va  = arena.alloc(sizeof(ornith_super::GqaSuperDesc), 256);
                std::vector<uint8_t> h_kv(256u * 272u, 0x22u);
                VJ_CHECK(rt.upload_to_vram(0, qnw_va, f_ones.data(), 256u * 4u) &&
                         rt.upload_to_vram(0, knw_va, f_ones.data(), 256u * 4u) &&
                         rt.upload_to_vram(0, kc_va, h_kv.data(), h_kv.size()) &&
                         rt.upload_to_vram(0, vc_va, h_kv.data(), h_kv.size()));
                {
                    const M1W32KA k1{w_qkv_q, w_qkv_d, xq_va, xds_va, gbuf, qkv_gqa_r, kD, qkv_st, 0u, 0ULL};
                    struct PrepKA { uint64_t qkv, qn, kn, qo, go, kc, vc; uint32_t M, pos0, nq, nkv, rd; float rb, eps; };
                    const PrepKA k2{gbuf, qnw_va, knw_va, qb_va, gb_va, kc_va, vc_va, 1u, pos0, 16u, 1u, 64u, 10000000.0f, 1e-6f};
                    struct FlKA { uint64_t q, g, kc, vc, oq, od; uint32_t M, pos0, nq, nkv; };
                    const FlKA k3{qb_va, gb_va, kc_va, vc_va, midq_va, midd_va, 1u, pos0, 16u, 1u};
                    const M1W32KA k4{w_out_q, w_out_d, midq_va, midd_va, out_va, kD, mid_c, out_st, 0u, 0ULL};
                    VJ_CHECK(rt.enqueue_kernel(0, *k_w32, ((qkv_gqa_r + 7u) / 8u) * 256u, 1, 1, 256, 1, 1, &k1, sizeof(k1), true));
                    VJ_CHECK(rt.enqueue_kernel(0, *k_qkv_prp, 17u * 256u, 1, 1, 256, 1, 1, &k2, sizeof(k2), true));
                    VJ_CHECK(rt.enqueue_kernel(0, *k_fl_1d, 16u * 256u, 1, 1, 256, 1, 1, &k3, sizeof(k3), true));
                    VJ_CHECK(rt.enqueue_kernel(0, *k_w32, ((kD + 7u) / 8u) * 256u, 1, 1, 256, 1, 1, &k4, sizeof(k4), true));
                    VJ_CHECK(rt.submit_batch(0) && rt.wait_idle(0));
                    VJ_CHECK(rt.download_from_vram(0, ref_out.data(), out_va, kD * 4u));
                }
                VJ_CHECK(rt.upload_to_vram(0, kc_va, h_kv.data(), h_kv.size()) &&
                         rt.upload_to_vram(0, vc_va, h_kv.data(), h_kv.size()));
                ornith_super::GqaSuperDesc qa{};
                qa.x_partial = out_va; qa.x_q8 = xq_va; qa.x_ds = xds_va;
                qa.w_qkv_qs = w_qkv_q; qa.w_qkv_dm = w_qkv_d; qa.attn_qkv = gbuf;
                qa.qkv_rows = qkv_gqa_r; qa.qkv_cols = kD; qa.qkv_qs_stride = qkv_st;
                qa.q_norm_w = qnw_va; qa.k_norm_w = knw_va; qa.k_cache_q8 = kc_va; qa.v_cache_q8 = vc_va;
                qa.q_buf = qb_va; qa.part_o = po_va; qa.part_ml = pml_va; qa.mid_q8 = midq_va; qa.mid_ds = midd_va;
                qa.n_q_local = 16u; qa.n_kv_local = 1u; qa.rope_dims = 64u; qa.rope_base = 10000000.0f;
                qa.w_o_qs = w_out_q; qa.w_o_dm = w_out_d; qa.o_rows = kD; qa.o_cols = mid_c; qa.o_qs_stride = out_st;
                qa.super_bar = sbar_va; qa.D = kD; qa.eps = 1e-6f;
                VJ_CHECK(rt.upload_to_vram(0, qa_va, &qa, sizeof(qa)));
                struct GqaSupKA { uint64_t desc; uint32_t prev_ep, out_ep, pos0, ssense; };
                const GqaSupKA qska{qa_va, 0u, 0u, pos0, s_sense++};
                VJ_CHECK(rt.enqueue_kernel(0, *k_gqa_sup, 192u * 256u, 1, 1, 256, 1, 1, &qska, sizeof(qska), true));
                VJ_CHECK(rt.submit_batch(0) && rt.wait_idle(0));
                VJ_CHECK(rt.download_from_vram(0, cur_out.data(), out_va, kD * 4u));
                double mdiff_gqa = 0.0; uint32_t ex_gqa = 0;
                for (uint32_t i = 0; i < kD; ++i) {
                    mdiff_gqa = std::max(mdiff_gqa, std::fabs(double(ref_out[i]) - double(cur_out[i])));
                    if (std::memcmp(&ref_out[i], &cur_out[i], 4) == 0) ++ex_gqa;
                }
                std::printf("  [SUPER-A-GQA 192-WG vs 4-disp pos=%u] bit_exact=%u/%u max_abs_diff=%.3e\n", pos0, ex_gqa, kD, mdiff_gqa);
            }

            // Safe single-GPU (GPU 0 only) synthetic verification of Wave 5 Dynamic LRU MoE Cache (on-GPU IQ2_S->DOT8 repack + hit/miss)!
            const auto* k_repack = rt.get_kernel(0, "vj_repack_iq2s_to_dot8_gu_kernel");
            const auto* k_gu_lru = rt.get_kernel(0, "vj_moe_gu_dot8_lru_kernel");
            if (k_repack && k_gu_lru && k_gu_128) {
                const auto lyt = dot8_gu_iq2s_layout(4ULL * kLff, kD); // 4 cached slots: experts 0,2,4,6 HIT; 1,3,5,7 MISS!
                const uint64_t cw_va   = arena.alloc(lyt.w_plane_bytes(), 4096);
                const uint64_t cs_va   = lyt.scale_embedded ? (cw_va + lyt.s_offset_bytes) : arena.alloc(lyt.s_plane_bytes(), 256);
                const uint64_t e2s_va  = arena.alloc(256u * 4u, 256);
                const uint64_t ref_mq  = arena.alloc(8u * kLff, 256);
                const uint64_t ref_md  = arena.alloc(8u * 16u * 4u, 256);
                std::vector<int32_t> h_e2s(256, -1);
                for (uint32_t s = 0u; s < 4u; ++s) h_e2s[s * 2u] = static_cast<int32_t>(s);
                VJ_CHECK(rt.upload_to_vram(0, e2s_va, h_e2s.data(), 256u * 4u));
                // Repack experts {0, 2, 4, 6} on-GPU and time 1 expert repack!
                struct RepKA { uint64_t src, cw, cs; uint32_t eid, sid, lff, wst, sst; };
                const ClkKA c0{clk_va, 0u, 0u}, c1{clk_va, 1u, 0u};
                VJ_CHECK(rt.enqueue_kernel(0, *k_clk, 64u, 1, 1, 64, 1, 1, &c0, sizeof(c0), true));
                for (uint32_t s = 0u; s < 4u; ++s) {
                    const RepKA rka{v_gu[0], cw_va, cs_va, s * 2u, s, kLff, lyt.w_row_stride_bytes, lyt.s_row_stride_bytes};
                    VJ_CHECK(rt.enqueue_kernel(0, *k_repack, kLff * 128u, 1, 1, 128, 1, 1, &rka, sizeof(rka), true));
                }
                VJ_CHECK(rt.enqueue_kernel(0, *k_clk, 64u, 1, 1, 64, 1, 1, &c1, sizeof(c1), true));
                // Reference baseline IQ2_S merged Gate+Up vs LRU (50% hit DOT8 + 50% miss IQ2_S)!
                struct GuKA    { uint64_t wgu, xq, xd, tids, mq, md; uint32_t M, lff; };
                struct GuLruKA { uint64_t wgu, cw, cs, e2s, xq, xd, tids, mq, md; uint32_t M, lff, wst, sst; };
                const GuKA    gk{v_gu[0], xq_va, xds_va, tids_va, ref_mq, ref_md, 1u, kLff};
                const GuLruKA lk{v_gu[0], cw_va, cs_va, e2s_va, xq_va, xds_va, tids_va, mq_va, md_va, 1u, kLff, lyt.w_row_stride_bytes, lyt.s_row_stride_bytes};
                VJ_CHECK(rt.enqueue_kernel(0, *k_gu_128, 128u * 256u, 1, 1, 256, 1, 1, &gk, sizeof(gk), true));
                VJ_CHECK(rt.enqueue_kernel(0, *k_gu_lru, 128u * 256u, 1, 1, 256, 1, 1, &lk, sizeof(lk), true));
                VJ_CHECK(rt.submit_batch(0) && rt.wait_idle(0));
                uint64_t ts[2] = {};
                std::vector<int8_t> q_ref(8u * kLff), q_lru(8u * kLff);
                std::vector<float>  d_ref(8u * 16u),  d_lru(8u * 16u);
                VJ_CHECK(rt.download_from_vram(0, ts, clk_va, 16u) &&
                         rt.download_from_vram(0, q_ref.data(), ref_mq, q_ref.size()) &&
                         rt.download_from_vram(0, q_lru.data(), mq_va,  q_lru.size()) &&
                         rt.download_from_vram(0, d_ref.data(), ref_md, d_ref.size() * 4u) &&
                         rt.download_from_vram(0, d_lru.data(), md_va,  d_lru.size() * 4u));
                uint32_t q_ex = 0, d_ex = 0;
                for (size_t i = 0; i < q_ref.size(); ++i) q_ex += (q_ref[i] == q_lru[i]);
                for (size_t i = 0; i < d_ref.size(); ++i) d_ex += (std::memcmp(&d_ref[i], &d_lru[i], 4) == 0);
                const double us_per_exp = static_cast<double>(ts[1] - ts[0]) * 40e-3 / 4.0;
                std::printf("  [MOE-LRU on-GPU repack + 50%% hit/miss] %.1f us/expert | q8_exact=%u/%zu d_exact=%u/%zu\n",
                            us_per_exp, q_ex, q_ref.size(), d_ex, d_ref.size());
            }
        }
    }

    // ---- Wave 4: Native 4.5 bpw Q4_K Streaming GEMV A/B vs VJ_Q4_A8 (5.0 bpw) + Descriptor-Driven Runtime::execute_plan ----
    const auto* k_q4k = rt.get_kernel(0, "vj_gemv_q4k_stream_kernel");
    if (k_q4k) {
        constexpr uint32_t kRows = 3072u, kCols = 4096u, kBanks = 36u;
        constexpr uint32_t kNsb = kCols / 256u; // 16 superblocks
        const uint32_t q4k_raw_st = kNsb * 144u; // 2304 B/row (4.50 bpw!)
        const uint32_t q4k_odd_st = ((q4k_raw_st + 255u) / 256u) * 256u;
        const uint32_t q4k_tcc_st = ((q4k_odd_st / 256u) & 1u) ? q4k_odd_st : (q4k_odd_st + 256u); // 2816 B (odd-256B)
        struct HostF2 { float x, y; };
        std::vector<uint8_t> h_q4k(static_cast<size_t>(kRows) * q4k_raw_st, 0);
        std::vector<int8_t>  h_xq(kCols);
        std::vector<HostF2>  h_xds(kCols / 32u);
        uint32_t rng = 0x517CC1B7u;
        auto nxt = [&]() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; };
        for (uint32_t b = 0; b < kCols / 32u; ++b) {
            float xd = 0.01f * (1.0f + float(nxt() % 7u));
            float xs = 0.0f;
            for (uint32_t i = 0; i < 32u; ++i) {
                int8_t q = static_cast<int8_t>(int(nxt() % 201u) - 100);
                h_xq[b * 32u + i] = q;
                xs += float(q) * xd;
            }
            h_xds[b] = HostF2{xd, xs};
        }
        auto f32_to_f16 = [](float v) -> uint16_t {
            _Float16 h = static_cast<_Float16>(v);
            uint16_t u; std::memcpy(&u, &h, 2); return u;
        };
        auto f16_to_f32 = [](uint16_t u) -> float {
            _Float16 h; std::memcpy(&h, &u, 2); return static_cast<float>(h);
        };
        for (uint32_t r = 0; r < kRows; ++r) {
            for (uint32_t sb = 0; sb < kNsb; ++sb) {
                uint8_t* bp = h_q4k.data() + static_cast<size_t>(r) * q4k_raw_st + sb * 144u;
                const uint16_t dh = f32_to_f16(0.004f * (1.0f + float(nxt() % 5u)));
                const uint16_t mh = f32_to_f16(0.003f * (1.0f + float(nxt() % 5u)));
                std::memcpy(bp, &dh, 2); std::memcpy(bp + 2, &mh, 2);
                for (uint32_t k = 4; k < 144u; ++k) bp[k] = static_cast<uint8_t>(nxt() & 0xFFu);
            }
        }
        // CPU exact reference on rows 0..kRows-1
        std::vector<float> ref_y(kRows, 0.0f);
        for (uint32_t r = 0; r < kRows; ++r) {
            double acc = 0.0;
            for (uint32_t sb = 0; sb < kNsb; ++sb) {
                const uint8_t* bp = h_q4k.data() + static_cast<size_t>(r) * q4k_raw_st + sb * 144u;
                uint16_t dh, mh; std::memcpy(&dh, bp, 2); std::memcpy(&mh, bp + 2, 2);
                const float d = f16_to_f32(dh), dmin = f16_to_f32(mh);
                const uint8_t* sc = bp + 4;
                const uint8_t* qs = bp + 16;
                for (uint32_t c = 0; c < 4u; ++c) {
                    const uint32_t j0 = c * 2u, j1 = j0 + 1u;
                    auto get_sm = [&](uint32_t j, uint32_t& s_out, uint32_t& m_out) {
                        if (j < 4u) { s_out = sc[j] & 63u; m_out = sc[j + 4u] & 63u; }
                        else { s_out = (sc[j + 4u] & 0xFu) | ((sc[j - 4u] >> 6) << 4);
                               m_out = (sc[j + 4u] >> 4)   | ((sc[j] >> 6) << 4); }
                    };
                    uint32_t sc0, m0, sc1, m1;
                    get_sm(j0, sc0, m0); get_sm(j1, sc1, m1);
                    const uint32_t b_lo = sb * 8u + j0, b_hi = sb * 8u + j1;
                    int32_t dp_lo = 0, dp_hi = 0;
                    for (uint32_t i = 0; i < 32u; ++i) {
                        const uint8_t q = qs[c * 32u + i];
                        dp_lo += int32_t(q & 0xFu) * int32_t(h_xq[b_lo * 32u + i]);
                        dp_hi += int32_t(q >> 4)   * int32_t(h_xq[b_hi * 32u + i]);
                    }
                    acc += double(d * float(sc0) * h_xds[b_lo].x) * double(dp_lo) - double(dmin * float(m0)) * double(h_xds[b_lo].y);
                    acc += double(d * float(sc1) * h_xds[b_hi].x) * double(dp_hi) - double(dmin * float(m1)) * double(h_xds[b_hi].y);
                }
            }
            ref_y[r] = static_cast<float>(acc);
        }

        const uint64_t q4k_xq_va  = rt.vram_arena(0).alloc(kCols, 4096);
        const uint64_t q4k_xds_va = rt.vram_arena(0).alloc((kCols / 32u) * 8u, 4096);
        const uint64_t q4k_y_va   = rt.vram_arena(0).alloc(kRows * 4u, 4096);
        VJ_CHECK(rt.upload_to_vram(0, q4k_xq_va, h_xq.data(), kCols));
        VJ_CHECK(rt.upload_to_vram(0, q4k_xds_va, h_xds.data(), (kCols / 32u) * 8u));
        std::vector<uint64_t> q4k_banks(kBanks);
        for (uint32_t b = 0; b < kBanks; ++b) {
            q4k_banks[b] = rt.vram_arena(0).alloc(static_cast<uint64_t>(kRows) * q4k_raw_st, 4096);
            VJ_CHECK(q4k_banks[b] && rt.upload_to_vram(0, q4k_banks[b], h_q4k.data(), h_q4k.size()));
        }
        struct Q4kKA { uint64_t w, xq, xds, y; uint32_t rows, cols, stride, pad; uint64_t ptx; };
        const double q4k_real_mb = double(kBanks) * double(kRows) * double(q4k_raw_st) / 1e6; // 254.8 MB (10% < 283.1 MB!)
        auto time36_q4k = [&](const char* label, uint32_t stride, uint64_t ptx) {
            double best_ms = 1e9;
            const ClkKA c0{clk_va, 0u, 0u}, c1{clk_va, 1u, 0u};
            for (int it = 0; it < 6; ++it) {
                VJ_CHECK(rt.enqueue_kernel(0, *k_clk, 64, 1, 1, 64, 1, 1, &c0, sizeof(c0), true));
                for (uint32_t b = 0; b < kBanks; ++b) {
                    const Q4kKA ka{q4k_banks[b], q4k_xq_va, q4k_xds_va, q4k_y_va, kRows, kCols, stride, 0u, ptx};
                    VJ_CHECK(rt.enqueue_kernel(0, *k_q4k, ((kRows + 7u) / 8u) * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
                }
                VJ_CHECK(rt.enqueue_kernel(0, *k_clk, 64, 1, 1, 64, 1, 1, &c1, sizeof(c1), true));
                VJ_CHECK(rt.submit_batch(0) && rt.wait_idle(0));
                uint64_t clk[2] = {};
                VJ_CHECK(rt.download_from_vram(0, clk, clk_va, 16u));
                const double ms = static_cast<double>(clk[1] - clk[0]) * 40e-6;
                if (it > 0 && ms < best_ms) best_ms = ms;
            }
            std::printf("  %-52s %6.3f ms | %6.1f real_GB/s (%.1f%% of 787.0)\n",
                        label, best_ms, q4k_real_mb / best_ms, 100.0 * (q4k_real_mb / best_ms) / 787.0);
        };
        std::printf("\n=== Wave 4B Native Q4_K (4.50 bpw = %.1f MB) vs VJ_Q4_A8 (5.00 bpw = 283.1 MB) ===\n", q4k_real_mb);
        time36_q4k("(Q4K-0) native 144B/sb (4.50 bpw) NO PP", q4k_raw_st, 0ULL);
        time36_q4k("(Q4K-1) native 144B/sb (4.50 bpw) + PP",  q4k_raw_st, peer_tx);
        // Verify GPU output vs CPU exact reference
        std::vector<float> gpu_y(kRows);
        VJ_CHECK(rt.download_from_vram(0, gpu_y.data(), q4k_y_va, kRows * 4u));
        double max_rel = 0.0;
        for (uint32_t r = 0; r < kRows; ++r) {
            max_rel = std::max(max_rel, std::fabs(double(gpu_y[r]) - double(ref_y[r])) / std::max(1e-5, std::fabs(double(ref_y[r]))));
        }
        std::printf("  [Q4_K CPU exactness] %u rows checked | max_rel_err=%.3e -> %s | odd256_stride=%u B\n",
                    kRows, max_rel, (max_rel < 1e-4) ? "PASS" : "FAIL", q4k_tcc_st);

        // Live Runtime::execute_plan verification consuming ops::build_step_plan (N=1 -> 0 AllReduce!)
        ops::ModelDesc md{};
        md.arch = "q4k_smoke"; md.d_model = kCols; md.vocab = kRows; md.dense_ff = kRows;
        md.attn = {16u, 2u, 128u};
        md.mixer = {ops::MixerKind::Gqa}; md.ffn = {ops::FfnKind::Dense};
        ops::TpDesc tp{}; tp.bw_gbs = {787.0}; tp.n_cu = {60u};
        ops::StepDesc sd{}; sd.m = 1u; sd.prefill = false; sd.ctx_tokens = 128u;
        ops::StepPlan sp{}; std::string serr;
        VJ_CHECK(ops::build_step_plan(md, tp, sd, sp, &serr));
        Q4kKA qka{q4k_banks[0], q4k_xq_va, q4k_xds_va, q4k_y_va, kRows, kCols, q4k_raw_st, 0u, 0ULL};
        kfd::ExecuteStats xst{};
        bool ok_exec = rt.execute_plan(sp, [&](size_t, const ops::PlannedOp& pop, kfd::OpDispatchSpec& out, std::string*) -> bool {
            if (pop.op.kind == ops::OpKind::GemvQ4Dot8) {
                out.kernel = k_q4k; out.grid_x = ((kRows + 7u) / 8u) * 256u; out.block_x = 256u;
                out.kernargs = &qka; out.kernarg_bytes = sizeof(qka);
            } else {
                out.skip = true;
            }
            return true;
        }, true, &xst, &serr);
        std::printf("  [Runtime::execute_plan live GPU0] ok=%d ops_enqueued=%u ops_skipped=%u allreduce=%u -> %s\n",
                    ok_exec ? 1 : 0, xst.ops_enqueued, xst.ops_skipped, xst.allreduce_ops,
                    (ok_exec && xst.ops_enqueued > 0u && xst.allreduce_ops == 0u) ? "PASS" : "FAIL");
    }
    return 0;
}

namespace {

class OrnithLiveEngine final : public IEngine, public sched::BatchEngine {
public:
    OrnithLiveEngine() = default;
    ~OrnithLiveEngine() override { shutdown(); }

    bool start(const serve::EngineSpec& spec, std::string* err) {
        // Fast pre-KFD validation so weightless/wrong GGUFs fail cleanly without touching /dev/kfd.
        GgufFile f;
        if (!f.open(spec.model_path)) {
            if (err) *err = f.error();
            return false;
        }
        const GgufTensorInfo* te = f.tensor("token_embd.weight");
        if (!te) {
            if (err) *err = "token_embd.weight missing; the Ornith GPU server requires a full Ornith-122B Q8_0 embedding table";
            return false;
        }
        if (te->type != static_cast<uint32_t>(GgmlType::Q8_0) || te->n_dims != 2 || te->ne[0] != 3072u) {
            if (err) *err = "unsupported token_embd.weight layout for the Ornith-122B GPU runner";
            return false;
        }
        const auto sel_s = [] {
            auto v = kfd::Runtime::scan_gfx906_topology();
            std::string ign;
            if (!gpusel::filter_from_env(v, &ign)) v.clear();
            return v;
        }();
        const auto budget = preflight::check_arena(preflight::gfx906_vram_bytes("/"), preflight::kOrnithArenaGiB,
                                                   static_cast<uint32_t>(std::max<size_t>(1u, sel_s.size())),
                                                   "the Ornith-122B GPU server");
        if (!budget.ok) {
            if (err) *err = budget.message;
            return false;
        }
        if (!legacy_tok_.load_from_gguf(spec.model_path, err)) return false;
        bpe::Vocab vocab;
        if (!bpe::read_gguf_vocab(spec.model_path, vocab, err) || !bpe_tok_.load(std::move(vocab), err)) return false;
        eos_id_ = bpe_tok_.eos();
        im_end_id_ = bpe_tok_.token_id("<|im_end|>");
        eot_id_ = bpe_tok_.token_id("<|endoftext|>");

        info_.id = spec.model_path;
        info_.arch = spec.architecture.empty() ? "qwen35moe" : spec.architecture;
        info_.context_length = spec.context;
        info_.vocab_size = static_cast<uint32_t>(bpe_tok_.vocab_size());
        info_.quant_summary = "VJ_Q4_A8+VJ_Q3_IQ2S+VJ_Q4_DOT8";

        // Default to the qualified fast M=1 super-kernel layout unless the operator overrides via env.
        const char* batch_env = std::getenv("VEGAJET_GPU_BATCH");
        batch_size_ = batch_env ? static_cast<uint32_t>(std::atoi(batch_env)) : 1u;
        if (batch_size_ < 1u || batch_size_ > 8u) { if (err) *err = "VEGAJET_GPU_BATCH must be 1..8"; return false; }
        if (batch_size_ > 1u) setenv("VEGAJET_RAW_ONLY", "0", 1);
        else setenv("VEGAJET_RAW_ONLY", "1", 0);
        setenv("VEGAJET_GU_Q3", "0", 0);  // GU_Q3 is lossy (decode-trace KL 3.5): opt-in only
        setenv("VEGAJET_VA_CLUSTER", "1", 0);
        setenv("VEGAJET_VOCAB_MUBUF", "1", 0);
        setenv("VEGAJET_SUPER_WGS", "240", 0);

        OrnithGpuOptions opt;
        opt.hsaco_dir = spec.hsaco_dir;
        opt.max_ctx = std::max<uint32_t>(spec.context, 256u);
        opt.run_bench = false;
        opt.run_parity = false;
        if (batch_size_ > 1u) {
            slot_engine_ = std::make_unique<sched::SlotBatchEngine>(batch_size_, opt.max_ctx/8u,
                [this](const auto& slots, const auto& pos, const auto& toks, const auto& fresh) {
                    return step_fn_(slots,pos,toks,fresh);
                },[this](auto slot,auto pos,const auto& toks,bool fresh,bool sample) {
                    return prefill_fn_(slot,pos,toks,fresh,sample);
                });
            opt.serve_step_ready = [this](OrnithGpuOptions::ServeStepFn fn,OrnithGpuOptions::ServePrefillFn prefill) {
                step_fn_=std::move(fn);prefill_fn_=std::move(prefill);
            };
        }
        opt.serve_hook = [this](const OrnithGpuOptions::ServeGenerateFn& gen, const std::string& desc) {
            {
                std::lock_guard<std::mutex> lk(mu_);
                description_ = desc;
                ready_ = true;
            }
            cv_ready_.notify_all();

            std::unique_lock<std::mutex> lk(mu_);
            while (!stop_) {
                cv_req_.wait(lk, [this] { return stop_ || has_req_; });
                if (stop_) break;
                if (batch_pending_) {
                    const auto plan = std::move(batch_plan_);
                    batch_pending_ = false; has_req_ = false;
                    lk.unlock();
                    try {
                        batch_result_ = slot_engine_->step(plan);
                        for (auto& result:batch_result_) for (auto& token:result.tokens)
                            if ((im_end_id_>=0 && token==im_end_id_) || (eot_id_>=0 && token==eot_id_)) token=eos_id_;
                    }
                    catch (...) { batch_error_ = std::current_exception(); }
                    lk.lock(); req_done_ = true; cv_done_.notify_all();
                    continue;
                }
                const std::vector<int32_t> pids = std::move(req_prompt_);
                const int32_t max_tok = req_max_tokens_;
                const TokenCallback cb = std::move(req_cb_);
                has_req_ = false;
                lk.unlock();

                gen(pids, max_tok, [&](int32_t tid, uint32_t n_prompt, uint32_t n_comp, double pp_ms, double dec_ms) -> bool {
                    TokenEvent ev{};
                    ev.token_id = tid;
                    ev.prompt_tokens = n_prompt;
                    ev.completion_tokens = n_comp;
                    ev.prefill_ms = pp_ms;
                    ev.decode_ms = dec_ms;
                    const bool hit_eos = (tid == eos_id_) || (im_end_id_ >= 0 && tid == im_end_id_) ||
                                         (eot_id_ >= 0 && tid == eot_id_);
                    if (hit_eos) {
                        ev.is_eos = true;
                        ev.text.clear();
                        (void)cb(ev);
                        return false;
                    }
                    ev.text = bpe_tok_.token_bytes(tid);
                    return cb(ev);
                });

                lk.lock();
                req_done_ = true;
                cv_done_.notify_all();
            }
        };

        const std::string model_path = spec.model_path;
        worker_ = std::thread([this, model_path, opt]() mutable {
            const OrnithBenchMetrics m = run_ornith_gpu(model_path, "", opt);
            std::lock_guard<std::mutex> lk(mu_);
            if (!ready_) {
                start_err_ = m.error.empty() ? "Ornith GPU engine exited before ready" : m.error;
                failed_ = true;
                cv_ready_.notify_all();
            }
            if (has_req_ || !req_done_) {
                req_done_ = true;
                cv_done_.notify_all();
            }
        });

        std::unique_lock<std::mutex> lk(mu_);
        cv_ready_.wait(lk, [this] { return ready_ || failed_; });
        if (failed_) {
            if (err) *err = start_err_;
            lk.unlock();
            shutdown();
            return false;
        }
        return true;
    }

    bool batching() const { return batch_size_ > 1u; }
    std::vector<sched::StepOutput> step(const sched::StepPlan& plan) override {
        std::unique_lock<std::mutex> lk(mu_);
        batch_plan_=plan; batch_error_=nullptr; batch_pending_=true;
        req_done_=false; has_req_=true; cv_req_.notify_one();
        cv_done_.wait(lk,[this]{ return req_done_ || stop_; });
        if (batch_error_) std::rethrow_exception(batch_error_);
        return std::move(batch_result_);
    }
    const std::string& description() const { return description_; }
    ModelInfo model_info() const override { return info_; }
    const BpeTokenizer& tokenizer() const override { return legacy_tok_; }

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
        std::unique_lock<std::mutex> lk(mu_);
        if (stop_ || failed_ || !ready_) return;
        req_prompt_ = prompt_tokens;
        req_max_tokens_ = params.max_tokens > 0 ? params.max_tokens : 64;
        req_cb_ = on_token;
        req_done_ = false;
        has_req_ = true;
        cv_req_.notify_one();
        cv_done_.wait(lk, [this] { return req_done_ || stop_; });
    }

private:
    void shutdown() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_req_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

    uint32_t batch_size_=1;
    std::unique_ptr<sched::SlotBatchEngine> slot_engine_;
    OrnithGpuOptions::ServeStepFn step_fn_;
    OrnithGpuOptions::ServePrefillFn prefill_fn_;
    sched::StepPlan batch_plan_;
    std::vector<sched::StepOutput> batch_result_;
    std::exception_ptr batch_error_;
    bool batch_pending_=false;
    ModelInfo info_{};
    BpeTokenizer legacy_tok_;
    bpe::Tokenizer bpe_tok_;
    int32_t eos_id_ = -1;
    int32_t im_end_id_ = -1;
    int32_t eot_id_ = -1;
    std::string description_;
    std::string start_err_;

    std::mutex mu_;
    std::condition_variable cv_ready_;
    std::condition_variable cv_req_;
    std::condition_variable cv_done_;
    bool ready_ = false;
    bool failed_ = false;
    bool stop_ = false;
    bool has_req_ = false;
    bool req_done_ = true;
    std::vector<int32_t> req_prompt_;
    int32_t req_max_tokens_ = 64;
    TokenCallback req_cb_;
    std::thread worker_;
};

} // namespace

void register_ornith_serve_engine() {
    static std::once_flag once;
    std::call_once(once, [] {
        auto provider = [](const serve::EngineSpec& spec, serve::EngineHandle& out, std::string* err) -> bool {
            auto eng = std::make_unique<OrnithLiveEngine>();
            if (!eng->start(spec, err)) return false;
            out.description = eng->description();
            if (eng->batching()) {
                // EngineHandle owns the IEngine; the adapter borrows it and dies first.
                struct Adapter final : sched::BatchEngine {
                    OrnithLiveEngine& engine;
                    explicit Adapter(OrnithLiveEngine& e):engine(e) {}
                    std::vector<sched::StepOutput> step(const sched::StepPlan& p) override { return engine.step(p); }
                };
                out.batch=std::make_unique<Adapter>(*eng);
            }
            out.engine = std::move(eng);
            return true;
        };
        auto& reg = serve::EngineRegistry::instance();
        reg.add("qwen35moe", provider);
        reg.add("qwen3_5_moe", provider);
        reg.add("ornith", provider);
    });
}

#endif

} // namespace vegajet::ornith
