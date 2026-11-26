// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "models/ornith/expert_tensor_checks.hpp"
#include "core/bpe_tokenizer.hpp"
#include "core/engine_factory.hpp"
#include "core/engine_interface.hpp"
#include "core/fr_spec.hpp"
#include "core/grid_planner.hpp"
#include "core/launch_planner.hpp"
#include "core/prefill_planner.hpp"
#include "core/sharder.hpp"
#include "core/tcc_stagger.hpp"
#include "core/token_dump.hpp"
#include "core/q4_dot8_repack.hpp"
#include "core/spec_policy.hpp"
#include "core/spec_cascade.hpp"
#include "core/suffix_trie.hpp"
#include "core/v12_dispatch.hpp"
#include "core/v12_modules.hpp"
#include "core/evict_controller.hpp"
#include "core/host_preflight.hpp"
#include "core/moe_lru_cache.hpp"
#include "core/warm_bench.hpp"
#include "models/ornith/ornith_golden.hpp"
#include "models/ornith/ornith_gpu.hpp"
#include "models/ornith/ornith_tp_exec.hpp"

#include "core/spec_engine.hpp"
#include "core/spec_tree.hpp"

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
#include "core/vj_device_guard.hpp"
#include "ops/kernels/vj_ornith_super_desc.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <omp.h>

namespace vegajet::ornith {

// Host planning is ready; never silently run the chain as a tree verifier.
static constexpr const char* kTreeVerifyUnavailable =
    "VEGAJET_V12_TREE_VERIFY=1 unavailable: S3 host planning only; "
    "GPU tree attention, GDN branching and KV commit are not integrated";

#ifdef VEGAJET_CPU_ONLY

OrnithBenchMetrics run_ornith_gpu(const std::string&, const std::string&, const OrnithGpuOptions&) {
    OrnithBenchMetrics m;
    m.ok = false;
    if (spec::tree_verify_requested(std::getenv("VEGAJET_V12_TREE_VERIFY"))) {
        m.error = kTreeVerifyUnavailable;
        return m;
    }
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

// v1.2 decode-kernel levers. Read once per process; every one of them defaults OFF, so the
// default path is byte-identical to v1.1.
bool v12_vocab_ilp() {
    static const bool on = [] { const char* e = std::getenv("VEGAJET_V12_VOCAB_ILP"); return e != nullptr && e[0] == '1'; }();
    return on;
}
bool v12_dense_persist() {
    static const bool on = [] { const char* e = std::getenv("VEGAJET_V12_DENSE_PERSIST"); return e != nullptr && e[0] == '1'; }();
    return on;
}
uint32_t v12_dense_wgs() {
    static const uint32_t wgs = [] {
        const char* e = std::getenv("VEGAJET_V12_DENSE_WGS");
        const long v = (e != nullptr) ? std::atol(e) : 224L;
        return static_cast<uint32_t>((v >= 96L && v <= 240L) ? v : 224L);
    }();
    return wgs;
}
// v1.2 D6: column-thread M=1 GDN ConvRec (no LDS partials between pred/o reductions). Default off.
bool v12_gdn_splitv() {
    static const bool on = [] { const char* e = std::getenv("VEGAJET_V12_GDN_SPLITV"); return e != nullptr && e[0] == '1'; }();
    return on;
}

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
};

struct LayerWeightsDev {
    bool     is_gdn = false;
    uint64_t attn_norm_va      = 0;
    uint64_t post_attn_norm_va = 0;

    // Fused GDN Q4_K -> VJ_Q4_A8: [qkv(6144) | gate(4096) | alpha(32) | beta(32)] = 10304 rows -> 1 dispatch!
    Q4A8WeightDev w_gdn_qkv_gab;
    Q4A8WeightDev w_ssm_out;
    uint64_t ssm_conv_w_va = 0;
    uint64_t ssm_a_va      = 0;
    uint64_t ssm_dt_va     = 0;
    uint64_t ssm_norm_va   = 0;

    // Fused GQA Q4_K -> VJ_Q4_A8: [QG(8192) | K(256) | V(256)] = 8704 rows -> 1 dispatch!
    Q4A8WeightDev w_attn_qkv;
    Q4A8WeightDev w_o;
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
    uint64_t w_dn_d8_s_va    = 0; // routed Down scale plane, 16 B/row
    uint32_t dn_d8_w_stride  = 0;
    uint32_t dn_d8_s_stride  = 0;
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
    uint64_t kv_k_va = 0, kv_v_va = 0; // [swa_win, n_kv, head_dim] FP16 ring on GPU 0
};

struct DFlash2Dev {
    bool     loaded = false;
    uint32_t n_layer = 0, n_head = 0, n_kv = 0, head_dim = 0, n_ff = 0, block_size = 0;
    uint32_t swa_win = 2048, sel_rank = 256, sel_topk = 16;
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
    uint64_t dense_q_va    = 0;   // v1.2 D13: dense GEMV tile counter / WG arrival pair
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
    uint64_t pf_xq4_va = 0, pf_xds4_va = 0, pf_xsum4_va = 0, moe_mid_q4_va = 0, moe_mid_ds4_va = 0, logits_va = 0;
    uint64_t dgrp_hdr_va = 0, dgrp_cnt_va = 0, dgrp_off_va = 0, dgrp_uniq_va = 0, dgrp_items_va = 0, dgrp_routes_va = 0, dgrp_buf_va = 0;

    uint64_t argmax_val_va = 0;
    uint64_t argmax_id_va  = 0;

    std::vector<uint64_t> gdn_conv_ring_va;
    std::vector<uint64_t> gdn_ssm_ring_va;
    uint64_t spec_gdn_s_slot_va[2] = {0, 0}; // --spec-gdn: 2 contiguous [36][32][128][128] FP32 slots
    uint64_t spec_gdn_o_va         = 0;      // [12][32][128] FP32
    uint64_t spec_nq_bar_va        = 0;      // v1.2 S5: 64 u32 grid barrier for the fused verify+nq launch
    uint64_t spec_gdn_rec_kv_va    = 0;      // [36][12][40*128] FP32
    uint64_t spec_gdn_rec_dec_va   = 0;      // [36][12][32] FP32
    uint64_t spec_gdn_rec_beta_va  = 0;      // [36][12][32] FP32
    uint64_t spec_fl_part_o_va     = 0;      // --spec-flash: [kMaxSpecBatch][16][60][256] FP32
    uint64_t spec_fl_part_ml_va    = 0;      // --spec-flash: [kMaxSpecBatch][16][60][2] FP32
    std::vector<uint64_t> attn_k_cache_va;
    std::vector<uint64_t> attn_v_cache_va;
    uint64_t mtp_k_cache_va = 0;
    uint64_t mtp_v_cache_va = 0;
    uint64_t mtp_cat_q8_va  = 0;
    uint64_t mtp_cat_ds_va  = 0;
    uint64_t mtp_h_pend_va  = 0;
    uint64_t mtp_h_cur_va   = 0;
    uint64_t mtp_h_zero_va  = 0;
    uint64_t mtp_h_stash_va = 0;             // --zero-tax-catchup: [16][D] FP32
    uint64_t mtp_moe_ds_va  = 0;
    uint64_t super_bar_va       = 0;         // Wave 5: 16 x 128 B grid-barrier slots for M=1 super-kernels
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
    const kfd::LoadedKernel* moe_m1_front = nullptr;
    const kfd::LoadedKernel* moe_m1_rtr_shgu = nullptr;
    const kfd::LoadedKernel* moe_m1_top8_swiglu = nullptr;
    const kfd::LoadedKernel* moe_m1_back = nullptr;
    const kfd::LoadedKernel* moe_gu_16r = nullptr;
    const kfd::LoadedKernel* moe_mid_f32_to_q8 = nullptr;
    const kfd::LoadedKernel* moe_down_d8_sh = nullptr;
    const kfd::LoadedKernel* gemv_q4_w32 = nullptr;
    const kfd::LoadedKernel* gemv_q4_w32p = nullptr; // v1.2 D13: persistent work-stealing grid variant
    const kfd::LoadedKernel* gemv_q4_w64 = nullptr;
    const kfd::LoadedKernel* gemv_q4_splitk_p1 = nullptr;
    const kfd::LoadedKernel* splitk_red_pp = nullptr;
    const kfd::LoadedKernel* gemv_q4_w16 = nullptr;
    const kfd::LoadedKernel* gemm_q4_tcc = nullptr;
    const kfd::LoadedKernel* gemm_q4_m8  = nullptr;
    const kfd::LoadedKernel* gemm_q4_m12 = nullptr;
    const kfd::LoadedKernel* gemm_q4_m16 = nullptr;
    const kfd::LoadedKernel* gemv_q4_mn  = nullptr;   // v1.2 S4: narrow M=2..16 verify GEMV
    const kfd::LoadedKernel* gdn_conv_l2 = nullptr;
    const kfd::LoadedKernel* gdn_recur = nullptr;
    const kfd::LoadedKernel* gdn_recur_sv = nullptr;
    const kfd::LoadedKernel* gdn_conv_recur_sv = nullptr;
    const kfd::LoadedKernel* gdn_conv_recur_ct = nullptr; // v1.2 D6 column-thread variant (VEGAJET_V12_GDN_SPLITV)
    const kfd::LoadedKernel* gdn_wy = nullptr;
    const kfd::LoadedKernel* attn_prep = nullptr;
    const kfd::LoadedKernel* attn_flash = nullptr;
    const kfd::LoadedKernel* attn_prep_flash_m1 = nullptr;
    const kfd::LoadedKernel* moe_rtr_log = nullptr;
    const kfd::LoadedKernel* moe_rtr_top8 = nullptr;
    const kfd::LoadedKernel* swiglu_q8 = nullptr;
    const kfd::LoadedKernel* moe_gu_iq2s = nullptr;
    const kfd::LoadedKernel* moe_gu_d8 = nullptr;
    const kfd::LoadedKernel* moe_gu_d8_cached = nullptr;
    const kfd::LoadedKernel* moe_gu_q3 = nullptr;
    const kfd::LoadedKernel* moe_down_iq3 = nullptr;
    const kfd::LoadedKernel* moe_down_d8 = nullptr;
    const kfd::LoadedKernel* moe_down_d8_pp = nullptr;
    const kfd::LoadedKernel* spec_gdn_ver = nullptr;
    const kfd::LoadedKernel* spec_gdn_nq = nullptr;
    const kfd::LoadedKernel* spec_gdn_ver_nq = nullptr;  // v1.2 S5: verify + norm/quant in one launch
    const kfd::LoadedKernel* spec_flash_part = nullptr;
    const kfd::LoadedKernel* spec_flash_merge = nullptr;
    const kfd::LoadedKernel* flash_skv_p1 = nullptr;
    const kfd::LoadedKernel* flash_skv_p2 = nullptr;
    const kfd::LoadedKernel* pf_fa2 = nullptr;          // prefill kernels (optional modules, see OrnithGpuOptions::pf_*)
    const kfd::LoadedKernel* pf_fa2_codex = nullptr;
    const kfd::LoadedKernel* pf_gemm = nullptr;
    const kfd::LoadedKernel* pf_gemm_64 = nullptr;
    const kfd::LoadedKernel* pf_gu = nullptr;
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
    const kfd::LoadedKernel* vocab_ilp = nullptr;  // v1.2 D3: 2-way ILP accumulator variant
    const kfd::LoadedKernel* vocab_argmax = nullptr;
    const kfd::LoadedKernel* vocab_argmax_b6 = nullptr;
    const kfd::LoadedKernel* vocab_argmax_b12 = nullptr;
    const kfd::LoadedKernel* vocab_argmax_b16 = nullptr;
    const kfd::LoadedKernel* p2p_ar_moe_m1_front = nullptr;
    const kfd::LoadedKernel* p2p_ar_moe_m1_front_s240 = nullptr;
    const kfd::LoadedKernel* moe_m1_back_s224 = nullptr;
    const kfd::LoadedKernel* moe_m1_back_asym = nullptr;
    const kfd::LoadedKernel* moe_down_asym = nullptr;
    const kfd::LoadedKernel* moe_down_asym_q4 = nullptr;
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
    const kfd::LoadedKernel* m1_gdn_super = nullptr;
    const kfd::LoadedKernel* m1_gqa_super = nullptr;
    const kfd::LoadedKernel* m1_moe_super = nullptr;
    const kfd::LoadedKernel* m1_moe_super_q3 = nullptr;
    const kfd::LoadedKernel* moe_gu_repack = nullptr;
    const kfd::LoadedKernel* moe_gu_lru    = nullptr;
};

struct RowSource {
    std::string           tensor_name;
    std::vector<uint32_t> row_indices;
};

// Repacks Q4_K GGUF tensors into Asymmetric VJ_Q4_A8 (u4 + FP16 d,m per 32 elems, odd-256B TCC anti-camping!)
bool upload_fused_q4a8_tcc(kfd::Runtime& rt,
                           size_t rank,
                           GgufFile& f,
                           const std::vector<RowSource>& sources,
                           uint32_t col_start,
                           uint32_t col_count,
                           const std::vector<uint32_t>* col_indices,
                           Q4A8WeightDev& out,
                           std::string* err) {
    uint32_t total_rows = 0;
    for (const auto& s : sources) total_rows += static_cast<uint32_t>(s.row_indices.size());
    const uint32_t C = col_indices ? static_cast<uint32_t>(col_indices->size()) : col_count;
    const uint32_t raw_rb = C / 2u;
    const auto lay = quant::select_tcc_layout(raw_rb);
    // For C=512 (256B), 3072 (1536->1792B), 4096 (2048->2304B), select_tcc_layout gives odd-256B with xor_en=0!
    const uint32_t stride_b = (lay.mode == quant::TccLayoutMode::XorTileSwizzle) ? (raw_rb + 256u) : lay.stride_bytes;

    std::vector<uint8_t>  qs(static_cast<size_t>(total_rows) * stride_b, 0);
    std::vector<uint32_t> dm(static_cast<size_t>(total_rows) * (C / 32u), 0);

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
                uint8_t*  dst_row = qs.data() + static_cast<size_t>(dst_r) * stride_b;
                uint32_t* dmrow   = dm.data() + static_cast<size_t>(dst_r) * (C / 32u);
                for (uint32_t b = 0; b < C / 32u; ++b) {
                    const float* blk = sub_row.data() + b * 32u;
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

// Background 250 ms sampler of power1_input (W) and temp3_input (HBM2 TMem) on every gfx906 (Vega 20) card that
// has both (MI50 has no power1_average). Tracks per card the instantaneous peak, the worst 10 s rolling average
// (the gated value) and the worst 1 s rolling average (informational). Each read is an SMU query that briefly
// stalls the GPU, so the cadence is kept low: 10 Hz sampling cost ~5 % decode throughput.
//
// Only gfx906 cards are sampled: an unrelated AMD display card that also exposes power1_input/temp3_input must not
// switch a two-GPU host to the N>=3 PSU limit or contaminate the peak / TMem numbers.
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
    HwmonSampler() {
        for (int card = 0; card < 8; ++card) {
            const std::string dev = "/sys/class/drm/card" + std::to_string(card) + "/device";
            if (!is_gfx906_card(dev)) continue;
            const std::string base = dev + "/hwmon/";
            for (int h = 0; h < 32; ++h) {
                const std::string dir = base + "hwmon" + std::to_string(h) + "/";
                double v = 0.0;
                std::ifstream fp(dir + "power1_input"), ft(dir + "temp3_input");
                if ((fp >> v) && (ft >> v)) { dirs_.push_back(dir); break; }
            }
        }
        win_.resize(dirs_.size());
        // VEGAJET_TELEM_MS=0 disables sampling (for A/B measurements of the sampling cost itself).
        if (const char* e = std::getenv("VEGAJET_TELEM_MS"); e && e[0]) period_ms_ = std::atoi(e);
        if (period_ms_ <= 0) { run_.store(false); return; }
        th_ = std::thread([this] { loop(); });
    }
    ~HwmonSampler() { stop(); }
    void stop() {
        run_.store(false);
        if (th_.joinable()) th_.join();
    }
    size_t cards() const { return dirs_.size(); }
    double max_mean_w() const {
        double m = 0.0;
        for (size_t i = 0; i < sum_w_.size(); ++i)
            if (cnt_w_[i] > 0) m = std::max(m, sum_w_[i] / static_cast<double>(cnt_w_[i]));
        return m;
    }
    double peak_w = 0.0, max_avg_w = 0.0, max_avg1s_w = 0.0, max_tmem_c = 0.0;
    uint64_t samples = 0;

private:
    void loop() {
        sum_w_.assign(dirs_.size(), 0.0);
        cnt_w_.assign(dirs_.size(), 0);
        while (run_.load()) {
            for (size_t i = 0; i < dirs_.size(); ++i) {
                double pw = 0.0, tm = 0.0;
                std::ifstream fp(dirs_[i] + "power1_input"), ft(dirs_[i] + "temp3_input");
                if (!(fp >> pw) || !(ft >> tm)) continue;
                pw *= 1e-6; tm *= 1e-3;
                peak_w = std::max(peak_w, pw);
                max_tmem_c = std::max(max_tmem_c, tm);
                sum_w_[i] += pw;
                ++cnt_w_[i];
                auto& w = win_[i];
                w.push_back(pw);
                const size_t n10 = static_cast<size_t>(std::max(1, 10000 / period_ms_));   // samples in 10 s
                const size_t n1  = static_cast<size_t>(std::max(1, 1000 / period_ms_));    // samples in 1 s
                if (w.size() > n10) w.pop_front();
                if (w.size() >= n1) {
                    double s1 = 0.0;
                    for (size_t k = w.size() - n1; k < w.size(); ++k) s1 += w[k];
                    max_avg1s_w = std::max(max_avg1s_w, s1 / static_cast<double>(n1));
                }
                if (w.size() == n10) {
                    double sum = 0.0;
                    for (double x : w) sum += x;
                    max_avg_w = std::max(max_avg_w, sum / static_cast<double>(n10));
                }
                ++samples;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(period_ms_));
        }
    }
    std::vector<std::string> dirs_;
    std::vector<std::deque<double>> win_;
    std::vector<double> sum_w_;
    std::vector<uint64_t> cnt_w_;
    std::atomic<bool> run_{true};
    int period_ms_ = 250;
    std::thread th_;
};

void read_hwmon_max(double& max_pwr_w, double& max_tmem_c) {
    for (int card = 0; card < 8; ++card) {
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

// v1.2 serving levers (read once per run): serve-side speculative decode.
static bool v12_serve_spec() {
    const char* v = std::getenv("VEGAJET_V12_SERVE_SPEC");
    return v != nullptr && v[0] == '1';
}

} // namespace

#include "models/ornith/ornith_nrank.inc"

OrnithBenchMetrics run_ornith_gpu(const std::string& gguf_path,
                                  const std::string& prompt_text,
                                  const OrnithGpuOptions& opt_in) {
    OrnithGpuOptions opt = opt_in;
    const OrnithGpuOptions base_opt = opt_in;
    OrnithBenchMetrics res{};
    res.error = ornith_raw_admit(opt);
    if (!res.error.empty()) return res;
    if (spec::tree_verify_requested(std::getenv("VEGAJET_V12_TREE_VERIFY"))) {
        res.error = kTreeVerifyUnavailable;
        return res;
    }
    auto t_load0 = std::chrono::steady_clock::now();
    if (opt.tp == 2) { // Legacy fast-path preflight; generic ranks use their own capacity plan.
        const auto budget = preflight::check_arena(preflight::gfx906_vram_bytes("/"), preflight::kOrnithArenaGiB,
                                                   preflight::kOrnithGpus, "the Ornith-122B GPU runner");
        if (!budget.ok) { res.error = budget.message; return res; }
    }

    GgufFile f;
    if (!f.open(gguf_path)) { res.error = f.error(); return res; }
    BpeTokenizer tok;
    if (!tok.load_from_gguf(gguf_path, &res.error)) return res;

    if (opt.tp != 2) {
        const auto ids = opt.dump_prompt_ids.empty() ? tok.encode(prompt_text.empty() ? "Hello" : prompt_text) : opt.dump_prompt_ids;
        return run_ornith_nrank(f, tok, ids, opt);
    }

    OrnithModel planner;
    quant::HardwareRooflineProfile hw_prof;
    const bool have_hw_prof = quant::HardwareRooflineProfile::load_json(quant::HardwareRooflineProfile::default_cache_path(), hw_prof);
    LoadOptions lopt;
    lopt.gpus = {
        GpuTarget{have_hw_prof && hw_prof.gpus.size() > 0 ? hw_prof.gpus[0].n_cu : 60u,
                  have_hw_prof && hw_prof.gpus.size() > 0 ? hw_prof.gpus[0].bw_hbm_gbs : 808.0,
                  have_hw_prof && hw_prof.gpus.size() > 0 ? hw_prof.gpus[0].sclk_ghz : 1.725,
                  30500ULL * 1024ULL * 1024ULL},
        GpuTarget{have_hw_prof && hw_prof.gpus.size() > 1 ? hw_prof.gpus[1].n_cu : 60u,
                  have_hw_prof && hw_prof.gpus.size() > 1 ? hw_prof.gpus[1].bw_hbm_gbs : 856.0,
                  have_hw_prof && hw_prof.gpus.size() > 1 ? hw_prof.gpus[1].sclk_ghz : 1.725,
                  30500ULL * 1024ULL * 1024ULL}
    };
    lopt.context_length = opt.max_ctx;
    lopt.batch_reuse_factor = 1.0;
    lopt.draft_gguf = opt.draft_gguf;
    const LoadReport lrep = planner.load(f, lopt);
    if (!lrep.ok) { res.error = lrep.error; return res; }
    const auto& hp = planner.hparams();
    const char* asym_env = std::getenv("VEGAJET_V12_ASYM_EXPERTS");
    const bool asym_experts = asym_env && std::strcmp(asym_env, "1") == 0;
    const bool v12_batch_catchup = (std::getenv("VEGAJET_V12_BATCH_CATCHUP") != nullptr &&
                                    std::getenv("VEGAJET_V12_BATCH_CATCHUP")[0] == '1');
    if (!dispatch::mtp_batch_layout_supported(asym_experts, v12_batch_catchup)) {
        res.error = "VEGAJET_V12_ASYM_EXPERTS=1 cannot be combined with VEGAJET_V12_BATCH_CATCHUP=1: "
                    "batched MTP requires symmetric 512-column experts; disable either toggle";
        return res;
    }

    // Validate metadata before mapping expert payloads or initializing GPU arenas.
    const char* raw_env = std::getenv("VEGAJET_RAW_ONLY");
    const bool load_mtp_experts = !(raw_env && raw_env[0] == '1') &&
                                  f.tensor("blk.48.nextn.eh_proj.weight") != nullptr;
    for (uint32_t il = 0; il < (load_mtp_experts ? 49u : 48u); ++il) {
        const std::string prefix = "blk." + std::to_string(il) + ".";
        for (const char* suffix : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}) {
            const std::string name = prefix + suffix;
            if (!validate_expert_tensor(f.tensor(name), name, hp.n_embd, hp.n_ff_exp, hp.n_expert,
                                        std::strcmp(suffix, "ffn_down_exps.weight") == 0, il < 48u, &res.error))
                return res;
        }
    }

    kfd::Runtime rt;
    if (!rt.init(30.5, 64, &res.error, 2) && !rt.init(30.0, 64, &res.error, 2)) return res;
    if (rt.num_gpus() != 2) { res.error = "Ornith-122B GPU runner requires exactly 2 selected MI50 GPUs"; return res; }
    // Resolve per-GPU calibrated HBM2 ceiling from hw_profile.json (matched by PCI BDF when available).
    double gpu_bw_ceiling_gbs[kfd::kMaxGpus] = {};
    double calibrated_hbm2_ceiling_gbs = 0.0;
    for (size_t g = 0; g < rt.num_gpus() && g < kfd::kMaxGpus; ++g) {
        const uint32_t loc = rt.devices()[g].location_id;
        char bdf_buf[32];
        std::snprintf(bdf_buf, sizeof(bdf_buf), "%04x:%02x:%02x.%u", rt.devices()[g].pci_domain,
                      (loc >> 8) & 0xFFu, (loc >> 3) & 0x1Fu, loc & 7u);
        double bw = (g < lopt.gpus.size()) ? lopt.gpus[g].bw_hbm_gbs : 800.0;
        if (have_hw_prof) {
            for (const auto& hg : hw_prof.gpus) {
                if (hg.id == bdf_buf && hg.bw_hbm_gbs > 0.0) { bw = hg.bw_hbm_gbs; break; }
            }
        }
        gpu_bw_ceiling_gbs[g] = bw;
        calibrated_hbm2_ceiling_gbs += bw;
    }
    // Resolve the requested variant before loading any experimental artifact.
    const char* vol = std::getenv("VEGAJET_V12_BARRIER_VOL");
    const char* iq2s = std::getenv("VEGAJET_V12_IQ2S_REG");
    const auto variant = select_super_variant(vol && vol[0] == '1', iq2s && iq2s[0] == '1');
    if (!variant.valid) {
        res.error = "VEGAJET_V12_BARRIER_VOL and VEGAJET_V12_IQ2S_REG are mutually exclusive";
        return res;
    }
    const std::string super_sfx = variant.suffix;
    const std::string hdir = kfd::resolve_hsaco_dir(opt.hsaco_dir);
    if (!rt.load_hsaco_dir(hdir, &res.error)) return res;

    KernelTable K[2];
    auto resolve_kernels = [&](std::string* kerr) -> bool {
        if (kerr) kerr->clear();
        for (size_t g = 0; g < 2; ++g) {
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
            K[g].gemv_q4_w32  = need_k("ornith_gemv_q4a8_m1_w32_kernel");
            K[g].gemv_q4_w32p = v12_dense_persist() ? rt.get_kernel(g, "ornith_gemv_q4a8_m1_w32_persist_kernel") : nullptr;
            K[g].gemv_q4_w64  = need_k("vj_gemv_q4a8_w64_kernel");
            K[g].gemv_q4_splitk_p1 = need_k("vj_gemv_q4a8_splitk_p1_kernel");
            K[g].splitk_red_pp = need_k("vj_splitk_reduce_prepush_f32_kernel");
            K[g].gemv_q4_w16  = need_k("ornith_gemv_q4a8_m1_w16_kernel");
            K[g].gemm_q4_tcc  = need_k("ornith_gemm_q4a8_tcc_kernel");
            K[g].gemm_q4_m8   = need_k("ornith_gemm_q4a8_m8_kernel");
            K[g].gemm_q4_m12  = need_k("ornith_gemm_q4a8_m12_kernel");
            K[g].gemm_q4_m16  = need_k("ornith_gemm_q4a8_m16_l2_kernel");
            K[g].gemv_q4_mn   = rt.get_kernel(g, "ornith_gemv_q4a8_mn_kernel");  // v1.2 S4 (optional)
            K[g].gdn_conv_l2  = need_k("ornith_gdn_conv_l2_gate_kernel");
            K[g].gdn_recur    = need_k("ornith_gdn_recur_quant_kernel");
            K[g].gdn_recur_sv = need_k("ornith_gdn_recur_splitv_quant_kernel");
            K[g].gdn_conv_recur_sv = need_k("ornith_gdn_conv_recur_splitv_quant_m1_kernel");
            if (v12_gdn_splitv()) {
                K[g].gdn_conv_recur_ct = need_k("ornith_gdn_conv_recur_colthread_quant_m1_kernel");
            }
            K[g].gdn_wy       = need_k("ornith_gdn_wy_prefill_quant_kernel");
            K[g].attn_prep    = need_k("ornith_attn_qkv_prep_kernel");
            K[g].attn_flash   = need_k("ornith_attn_flash_quant_kernel");
            K[g].attn_prep_flash_m1 = need_k("ornith_attn_prep_flash_m1_kernel");
            K[g].moe_rtr_log  = need_k("ornith_moe_router_fp16_logits_kernel");
            K[g].moe_rtr_top8 = need_k("ornith_moe_router_top8_kernel");
            K[g].swiglu_q8    = need_k("ornith_swiglu_quant_q8_kernel");
            K[g].moe_gu_iq2s      = need_k("ornith_moe_gu_iq2s_merged_kernel");
            K[g].moe_gu_d8        = need_k("vj_moe_gu_dot8_kernel");
            K[g].moe_gu_d8_cached = need_k("vj_moe_gu_dot8_cached_kernel");
            K[g].moe_gu_q3        = rt.get_kernel(g, "vj_moe_gu_q3_kernel");
            K[g].moe_down_iq3     = need_k("ornith_moe_down_iq3xxs_split_kernel");
            K[g].moe_down_d8      = need_k("vj_moe_down_dot8_kernel");
            K[g].moe_down_d8_pp   = need_k("vj_moe_down_dot8_prepush_kernel");
            K[g].spec_gdn_ver = opt.spec_gdn ? need_k("vj_spec_gdn_verify_kernel") : rt.get_kernel(g, "vj_spec_gdn_verify_kernel");
            K[g].spec_gdn_nq  = opt.spec_gdn ? need_k("vj_spec_gdn_norm_quant_kernel") : rt.get_kernel(g, "vj_spec_gdn_norm_quant_kernel");
            K[g].spec_gdn_ver_nq = rt.get_kernel(g, "vj_spec_gdn_verify_fused_nq_kernel");  // v1.2 S5 (optional)
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
            if (opt.pf_gu_tiled || warm_mode)   K[g].pf_gu   = need_k("vj_prefill_moe_gu_iq2s_tiled_kernel");
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
            if (opt.prefill_dot8 || warm_mode) {
                K[g].pf_q8_to_q4    = rt.get_kernel(g, "vj_prefill_q8_to_q4_kernel");
                K[g].pf_gemm_d8_128 = rt.get_kernel(g, "vj_prefill_gemm_q4a4_64x128_dot8_kernel");
                K[g].pf_gemm_d8_64  = rt.get_kernel(g, "vj_prefill_gemm_q4a4_64x64_dot8_kernel");
                K[g].pf_gu_d8       = rt.get_kernel(g, "vj_prefill_moe_gu_iq2s_dot8_kernel");
                K[g].pf_down_d8     = rt.get_kernel(g, "vj_prefill_moe_down_dot8_tiled_kernel");
                if (opt.prefill_dot8 && (!K[g].pf_q8_to_q4 || !K[g].pf_gemm_d8_128 || !K[g].pf_gemm_d8_64 || !K[g].pf_gu_d8 || !K[g].pf_down_d8)) {
                    if (kerr && kerr->empty()) *kerr = "Missing prefill_dot8 kernels in " + hdir;
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
            K[g].vocab_ilp = v12_vocab_ilp() ? rt.get_kernel(g, "ornith_vocab_argmax_q8_ilp_kernel") : nullptr;
            K[g].vocab_argmax_b6  = need_k("ornith_vocab_argmax_q8_batch6_kernel");
            K[g].vocab_argmax_b12 = need_k("ornith_vocab_argmax_q8_batch12_l2_kernel");
            K[g].m1_gdn_super     = rt.get_kernel(g, "ornith_m1_gdn_super_kernel" + super_sfx);
            K[g].m1_gqa_super     = rt.get_kernel(g, "ornith_m1_gqa_super_kernel" + super_sfx);
            K[g].m1_moe_super     = rt.get_kernel(g, "ornith_m1_moe_super_kernel" + super_sfx);
            K[g].m1_moe_super_q3  = rt.get_kernel(g, "ornith_m1_moe_super_q3_kernel" + super_sfx);
            if (!super_sfx.empty() && (!K[g].m1_gdn_super || !K[g].m1_gqa_super ||
                                       !K[g].m1_moe_super || !K[g].m1_moe_super_q3)) {
                // Fail closed: a requested variant that is not on disk would otherwise fall through
                // to the baseline kernels of a stale build and silently measure nothing.
                if (kerr && kerr->empty()) *kerr = "super-kernel variant " + super_sfx +
                    " requested but vj_ornith_super" + super_sfx + ".hsaco is missing from " + hdir;
                return false;
            }
            K[g].moe_gu_repack    = rt.get_kernel(g, "vj_repack_iq2s_to_dot8_gu_kernel");
            K[g].moe_gu_lru       = rt.get_kernel(g, "vj_moe_gu_dot8_lru_kernel");
            K[g].vocab_argmax_b16 = need_k("ornith_vocab_argmax_q8_batch16_l2_kernel");
            K[g].p2p_ar_moe_m1_front = need_k("ornith_p2p_ar_moe_m1_front_kernel");
            K[g].p2p_ar_moe_m1_front_s240 = need_k("ornith_p2p_ar_moe_m1_front_s240_kernel");
            K[g].moe_m1_rtr_shgu    = need_k("ornith_moe_m1_rtr_shgu_kernel");
            K[g].moe_m1_top8_swiglu = need_k("ornith_moe_m1_top8_swiglu_kernel");
            K[g].moe_m1_back_s224 = need_k("ornith_moe_m1_back_s224_kernel");
            K[g].moe_m1_back_s240 = need_k("ornith_moe_m1_back_s240_kernel");
            K[g].moe_gu_16r       = need_k("ornith_moe_gu_iq2s_m1_16r_kernel");
            K[g].moe_mid_f32_to_q8 = need_k("ornith_moe_mid_f32_to_q8_kernel");
            K[g].moe_down_d8_sh   = need_k("ornith_moe_down_d8_sh_m1_kernel");
            K[g].mtp_eh_cat   = need_k("ornith_mtp_eh_norm_cat_q8_kernel");
            K[g].moe_gu_q4a8  = need_k("ornith_moe_gu_q4a8_kernel");
            K[g].moe_down_q4a8= need_k("ornith_moe_down_q4a8_kernel");
            if (asym_experts) {
                K[g].moe_m1_back_asym = need_k("vj_moe_m1_back_asym_s224_kernel");
                K[g].moe_down_asym = need_k("vj_moe_down_asym_kernel");
                K[g].moe_down_asym_q4 = need_k("vj_moe_down_asym_q4a8_kernel");
            }
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
    std::printf("[ornith::gpu] Repacking 48L TP=2 Ornith-122B: VJ_Q4_A8 TCC-deswizzled dense trunk + FP16 router + SWAR uint4 IQ2_S/IQ3_XXS MoE...\n");

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
    // Wave 3: Model-agnostic bandwidth-proportional asymmetric sharding via core/sharder driven by hw_profile.json.
    const std::vector<double> bw_vec = {
        std::max(1.0, gpu_bw_ceiling_gbs[0]),
        std::max(1.0, gpu_bw_ceiling_gbs[1])
    };
    const auto voc_ranges = shard::split_rows(vocab, bw_vec);
    const uint32_t voc_rows[2] = { static_cast<uint32_t>(voc_ranges[0].size()),
                                   static_cast<uint32_t>(voc_ranges[1].size()) };
    const uint32_t voc_off[2]  = { static_cast<uint32_t>(voc_ranges[0].begin),
                                   static_cast<uint32_t>(voc_ranges[1].begin) };
    const auto ff_ranges = shard::split_expert_intermediate(hp.n_ff_exp, bw_vec, asym_experts);
    if (ff_ranges.size() != 2u || (asym_experts && hp.n_ff_exp != 1024u)) {
        res.error = "asymmetric experts require two nonempty 32-row shards of n_ff_exp=1024"; return res;
    }
    if (asym_experts && (opt.dot8_ab || !opt.dump_golden_layers_dir.empty())) {
        res.error = "asymmetric experts do not support legacy 512-only --dot8-ab / --dump-golden-layers; use --dump-logits";
        return res;
    }
    const uint32_t ff_size[2] = {static_cast<uint32_t>(ff_ranges[0].size()), static_cast<uint32_t>(ff_ranges[1].size())};
    const uint32_t ff_begin[2] = {0u, static_cast<uint32_t>(ff_ranges[1].begin)};
    // Capacity planning uses the larger shard; launches use the rank's width.
    const uint32_t local_ff = std::max(ff_size[0], ff_size[1]);
    if (asym_experts) std::printf("[ornith::asym] expert FF=%u/%u (BW %.1f/%.1f GB/s, row unit=32)\n",
                                  ff_size[0], ff_size[1], bw_vec[0], bw_vec[1]);

    const auto gdn_plan = shard::plan_gdn(hp.ssm_groups, hp.ssm_v_heads, bw_vec);
    const auto gqa_plan = shard::plan_attention(hp.n_head, hp.n_head_kv, bw_vec, false);
    std::printf("[ornith::shard] BW-proportional TP=2 (%.1f / %.1f GB/s): LM_head=%u/%u rows | GDN_K=%llu/%llu V=%llu/%llu | GQA_Q=%llu/%llu KV=%llu/%llu\n",
                bw_vec[0], bw_vec[1], voc_rows[0], voc_rows[1],
                static_cast<unsigned long long>(gdn_plan[0].k_groups.size()),
                static_cast<unsigned long long>(gdn_plan[1].k_groups.size()),
                static_cast<unsigned long long>(gdn_plan[0].v_heads.size()),
                static_cast<unsigned long long>(gdn_plan[1].v_heads.size()),
                static_cast<unsigned long long>(gqa_plan.gpus[0].q_heads.size()),
                static_cast<unsigned long long>(gqa_plan.gpus[1].q_heads.size()),
                static_cast<unsigned long long>(gqa_plan.gpus[0].kv_heads.size()),
                static_cast<unsigned long long>(gqa_plan.gpus[1].kv_heads.size()));

    std::vector<uint32_t> gdn_qkv_rows[2], gdn_v_rows[2], gdn_v_heads[2];
    const uint32_t gdn_v_per_k = hp.ssm_v_heads / std::max<uint32_t>(1u, hp.ssm_groups);
    for (size_t r = 0; r < 2; ++r) {
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

    Q8WeightDev emb_w[2], out_w[2], fr_w[2];
    std::vector<int32_t> fr_tok_map[2];
    uint64_t fr_tok_map_va[2] = {0, 0};
    uint32_t fr_rows_per_gpu = 0;
    const bool raw_only_vram = (std::getenv("VEGAJET_RAW_ONLY") != nullptr && std::getenv("VEGAJET_RAW_ONLY")[0] == '1');
    const bool has_mtp_tensor = !raw_only_vram && (f.tensor("blk.48.nextn.eh_proj.weight") != nullptr);
    const bool has_dflash2_opt = (!opt.draft_gguf.empty() && planner.draft_hparams().is_dflash2());
    const bool use_fr_spec = (has_mtp_tensor || has_dflash2_opt) && opt.fr_spec_rows > 0u && (opt.fr_spec_rows % 512u == 0u) && opt.fr_spec_rows <= vocab;
    uint64_t out_norm_va[2] = {0, 0};
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

            fr_rows_per_gpu = opt.fr_spec_rows / 2u;
            fr_tok_map[0].reserve(fr_rows_per_gpu);
            fr_tok_map[1].reserve(fr_rows_per_gpu);
            for (uint32_t i = 0; i < opt.fr_spec_rows; ++i) {
                fr_tok_map[i & 1u].push_back(selected[i]);
            }
        }
        for (size_t g = 0; g < 2; ++g) {
            emb_w[g].rows = vocab; emb_w[g].cols = D; emb_w[g].row_stride_bytes = D;
            emb_w[g].qs_va = rt.vram_arena(g).alloc(eqs.size(), 4096, kfd::Tier::COLD);  // 1 row/token: COLD
            emb_w[g].ds_va = rt.vram_arena(g).alloc(eds.size() * 2u, 256, kfd::Tier::COLD);
            VJ_CHECK(rt.upload_to_vram(g, emb_w[g].qs_va, eqs.data(), eqs.size()));
            VJ_CHECK(rt.upload_to_vram(g, emb_w[g].ds_va, eds.data(), eds.size() * 2u));
            if (!upload_f32_rows(rt, g, f, "output_norm.weight", nullptr, out_norm_va[g], &res.error)) return res;
            if (!upload_q8_tcc(rt, g, f, "output.weight", voc_off[g], voc_rows[g], out_w[g], &res.error)) return res;
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

    std::vector<LayerWeightsDev> layers[2];
    layers[0].resize(48);
    layers[1].resize(48);
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
    if (target_gu_dot8_layers == 0u && !no_auto_dot8 && (asym_experts || local_ff == 512u) && D == 3072u) {
        const Dot8RowLayout gu_lay = dot8_gu_iq2s_layout(256ULL * local_ff, D);
        const uint64_t est_base_vram = raw_only_vram ? (24600ULL << 20) : (26400ULL << 20);
        const uint64_t headroom      = raw_only_vram ? (640ULL << 20)   : (1600ULL << 20);
        const uint64_t spare_pool = (arena_cap_bytes > est_base_vram + headroom)
                                        ? (arena_cap_bytes - est_base_vram - headroom) : 0ULL;
        Dot8ClassSpec gu_spec;
        gu_spec.name = raw_only_vram ? "ffn_gate_up_exps(raw M=1 in-place)" : "ffn_gate_up_exps(M>=2 verify)";
        gu_spec.src_type = GgmlType::IQ2_S;
        gu_spec.role = Dot8Role::MoeGateUp;
        gu_spec.target = Dot8Target::VJ_Q4_DOT8;
        gu_spec.frees_old = raw_only_vram;
        gu_spec.old_vram_bytes_per_layer = 256ULL * local_ff * 1968ULL;
        gu_spec.new_vram_bytes_per_layer = gu_lay.vram_bytes();
        gu_spec.old_read_bytes_per_layer = 8ULL * local_ff * 1968ULL;
        gu_spec.new_read_bytes_per_layer = 8ULL * local_ff * (gu_lay.w_row_stride_bytes + gu_lay.s_row_stride_bytes);
        gu_spec.bw_old_gbs = raw_only_vram ? 365.0 : 245.0;
        gu_spec.bw_new_gbs = raw_only_vram ? 680.0 : 540.0;
        gu_spec.batch_m = raw_only_vram ? 1u : 5u;
        for (uint32_t l = 0u; l < 48u; ++l) gu_spec.layers.push_back(l);
        const auto dplan = plan_dot8_promotion({gu_spec}, spare_pool, true);
        target_gu_dot8_layers = static_cast<uint32_t>(dplan.promotions.size());
        if (target_gu_dot8_layers > 0u) {
            std::printf("[ornith::dot8] plan_dot8_promotion: pool=%.2f GiB/GPU -> promoting %u/48 Gate+Up layers to exact VJ_Q4_DOT8 (%s, est %.0f us saved/step)\n",
                        static_cast<double>(spare_pool) / 1073741824.0, target_gu_dot8_layers,
                        raw_only_vram ? "in-place raw M=1" : "verify M>=2", dplan.us_saved_total);
        }
    }
    uint32_t gu_dot8_promoted = 0u;

    for (uint32_t il = 0; il < 48u; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const bool is_gdn = hp.is_recurrent(il);

        for (size_t g = 0; g < 2; ++g) {
            const uint32_t local_ff = ff_size[g];
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
                    }, 0, D, nullptr, lw.w_gdn_qkv_gab, &res.error)) return res;
                if (!upload_fused_q4a8_tcc(rt, g, f, {{p + "ssm_out.weight", all_D_rows}},
                                           0, 0, &gdn_v_rows[g], lw.w_ssm_out, &res.error)) return res;
                if (!upload_f32_rows(rt, g, f, p + "ssm_conv1d.weight", &gdn_qkv_rows[g], lw.ssm_conv_w_va, &res.error)) return res;
                if (!upload_f32_rows(rt, g, f, p + "ssm_a", &gdn_v_heads[g], lw.ssm_a_va, &res.error)) return res;
                if (!upload_f32_rows(rt, g, f, p + "ssm_dt.bias", &gdn_v_heads[g], lw.ssm_dt_va, &res.error)) return res;
                if (!upload_f32_rows(rt, g, f, p + "ssm_norm.weight", nullptr, lw.ssm_norm_va, &res.error)) return res;
            } else {
                if (!upload_fused_q4a8_tcc(rt, g, f, {
                        {p + "attn_q.weight", iota_range(static_cast<uint32_t>(g) * 8192u, 8192u)},
                        {p + "attn_k.weight", iota_range(static_cast<uint32_t>(g) * 256u,  256u)},
                        {p + "attn_v.weight", iota_range(static_cast<uint32_t>(g) * 256u,  256u)}
                    }, 0, D, nullptr, lw.w_attn_qkv, &res.error)) return res;
                if (!upload_fused_q4a8_tcc(rt, g, f, {{p + "attn_output.weight", all_D_rows}},
                                           static_cast<uint32_t>(g) * 4096u, 4096u, nullptr, lw.w_o, &res.error)) return res;
                if (!upload_f32_rows(rt, g, f, p + "attn_q_norm.weight", nullptr, lw.q_norm_va, &res.error)) return res;
                if (!upload_f32_rows(rt, g, f, p + "attn_k_norm.weight", nullptr, lw.k_norm_va, &res.error)) return res;
            }

            if (!upload_f32_as_f16(rt, g, f, p + "ffn_gate_inp.weight", lw.w_router_f16_va, &res.error)) return res;
            if (!upload_f32_rows(rt, g, f, p + "ffn_gate_inp_shexp.weight", nullptr, lw.w_sh_gate_va, &res.error)) return res;
            if (!upload_fused_q4a8_tcc(rt, g, f, {
                    {p + "ffn_gate_shexp.weight", iota_range(ff_begin[g], local_ff)},
                    {p + "ffn_up_shexp.weight",   iota_range(ff_begin[g], local_ff)}
                }, 0, D, nullptr, lw.w_sh_gu, &res.error)) return res;
            if (!upload_fused_q4a8_tcc(rt, g, f, {{p + "ffn_down_shexp.weight", all_D_rows}},
                                       ff_begin[g], local_ff, nullptr, lw.w_sh_down, &res.error)) return res;
        }

        // Pack Gate+Up IQ2_S into coalesced uint4 merged split-plane layout
        {
            const auto* tig = f.tensor(p + "ffn_gate_exps.weight");
            const auto* tiu = f.tensor(p + "ffn_up_exps.weight");
            const uint8_t* rg = f.tensor_data(*tig);
            const uint8_t* ru = f.tensor_data(*tiu);
            const size_t raw_rb = 12u * 82u;
            const size_t n_rows_rank = 256ULL * local_ff;
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

            const bool use_gu_q3 = (std::getenv("VEGAJET_GU_Q3") != nullptr && std::atoi(std::getenv("VEGAJET_GU_Q3")) > 0);
            bool promoted_in_place = false;
            if (use_gu_q3 && (asym_experts || local_ff == 512u) && D == 3072u) {
                const Dot8RowLayout lay = q3_gu_iq2s_layout(n_rows_rank, D);
                const uint64_t reserve_tail = raw_only_vram
                    ? (static_cast<uint64_t>(47u - il) * (540ULL << 20) + (512ULL << 20))
                    : (static_cast<uint64_t>(47u - il) * (580ULL << 20)
                       + (opt.spec_gdn ? 2048ULL : 2800ULL) * (1ULL << 20));
                if (rt.vram_arena(0).used() + lay.vram_bytes() + reserve_tail <= arena_cap_bytes &&
                    rt.vram_arena(1).used() + lay.vram_bytes() + reserve_tail <= arena_cap_bytes) {
                    for (size_t g = 0; g < 2; ++g) {
                        const uint32_t local_ff = ff_size[g];
                        const auto lay = q3_gu_iq2s_layout(256ULL * local_ff, D);
                        const auto lay_e = q3_gu_iq2s_layout(local_ff, D);
                        std::vector<uint8_t> wpl(lay.w_plane_bytes()), spl(lay.s_plane_bytes());
                        bool rok = true;
                        #pragma omp parallel for schedule(static) reduction(&&:rok)
                        for (size_t e = 0; e < 256u; ++e) {
                            const size_t src_r0 = shard::expert_row_index(e, hp.n_ff_exp, ff_begin[g], 0u);
                            std::string e_err;
                            rok = rok && q3_repack_gu_iq2s_rows(
                                rg + src_r0 * raw_rb, ru + src_r0 * raw_rb, raw_rb, 0u, lay_e, 0u, local_ff,
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
            if (!promoted_in_place && il < target_gu_dot8_layers && (asym_experts || local_ff == 512u) && D == 3072u) {
                const Dot8RowLayout lay = dot8_gu_iq2s_layout(n_rows_rank, D);
                const uint64_t reserve_tail = raw_only_vram
                    ? (static_cast<uint64_t>(47u - il) * (505ULL << 20) + (768ULL << 20))
                    : (static_cast<uint64_t>(47u - il) * (520ULL << 20)
                       + (opt.spec_gdn ? 2048ULL : 2800ULL) * (1ULL << 20)
                       + (max_batch > 128u ? (700ULL << 20) : 0ULL));
                const uint64_t iq2s_extra = raw_only_vram ? 0ULL : (n_rows_rank * row_u4 * 16ULL);
                if (rt.vram_arena(0).used() + lay.vram_bytes() + iq2s_extra + reserve_tail <= arena_cap_bytes &&
                    rt.vram_arena(1).used() + lay.vram_bytes() + iq2s_extra + reserve_tail <= arena_cap_bytes) {
                    for (size_t g = 0; g < 2; ++g) {
                        const uint32_t local_ff = ff_size[g];
                        const auto lay = dot8_gu_iq2s_layout(256ULL * local_ff, D);
                        const auto lay_e = dot8_gu_iq2s_layout(local_ff, D);
                        std::vector<uint8_t> wpl(lay.w_plane_bytes()), spl(lay.s_plane_bytes());
                        bool rok = true;
                        #pragma omp parallel for schedule(static) reduction(&&:rok)
                        for (size_t e = 0; e < 256u; ++e) {
                            const size_t src_r0 = shard::expert_row_index(e, hp.n_ff_exp, ff_begin[g], 0u);
                            std::string e_err;
                            rok = rok && dot8_repack_gu_iq2s_rows(
                                rg + src_r0 * raw_rb, ru + src_r0 * raw_rb, raw_rb, 0u, lay_e, 0u, local_ff,
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
                    promoted_in_place = raw_only_vram;
                }
            }
            if (!promoted_in_place && asym_experts) {
                for (size_t g = 0; g < 2; ++g) {
                    const uint32_t lff = ff_size[g];
                    std::vector<uint32_t> packed(256ULL * lff * row_u4 * 4u);
                    #pragma omp parallel for schedule(static)
                    for (size_t e = 0; e < 256u; ++e) for (size_t r = 0; r < lff; ++r) {
                        const size_t src_r = shard::expert_row_index(e, hp.n_ff_exp, ff_begin[g], r);
                        pack_row(rg + src_r * raw_rb, ru + src_r * raw_rb,
                                 packed.data() + (e * lff + r) * row_u4 * 4u);
                    }
                    const size_t bytes = packed.size() * 4u;
                    auto& va = layers[g][il].w_gu_iq2s_va;
                    va = rt.vram_arena(g).alloc(bytes, 4096, kfd::Tier::COLD);
                    if (!va) { res.error = "VRAM OOM for routed Gate+Up"; return res; }
                    VJ_CHECK(rt.upload_to_vram(g, va, packed.data(), bytes));
                }
            } else if (!promoted_in_place) {
                std::vector<uint32_t> m0(n_rows_rank * row_u4 * 4u), m1(n_rows_rank * row_u4 * 4u);
                #pragma omp parallel for schedule(static)
                for (size_t e = 0; e < 256u; ++e) {
                    for (size_t r = 0; r < local_ff; ++r) {
                        const size_t idx = e * local_ff + r;
                        const size_t full0 = e * (2u * local_ff) + r;
                        const size_t full1 = full0 + local_ff;
                        pack_row(rg + full0 * raw_rb, ru + full0 * raw_rb, m0.data() + idx * row_u4 * 4u);
                        pack_row(rg + full1 * raw_rb, ru + full1 * raw_rb, m1.data() + idx * row_u4 * 4u);
                    }
                }
                const size_t bytes = m0.size() * 4u;
                layers[0][il].w_gu_iq2s_va = rt.vram_arena(0).alloc(bytes, 4096, kfd::Tier::COLD);
                layers[1][il].w_gu_iq2s_va = rt.vram_arena(1).alloc(bytes, 4096, kfd::Tier::COLD);
                if (!layers[0][il].w_gu_iq2s_va || !layers[1][il].w_gu_iq2s_va) { res.error = "VRAM OOM for routed Gate+Up"; return res; }
                VJ_CHECK(rt.upload_to_vram(0, layers[0][il].w_gu_iq2s_va, m0.data(), bytes));
                VJ_CHECK(rt.upload_to_vram(1, layers[1][il].w_gu_iq2s_va, m1.data(), bytes));
            }
            f.release(rg, tig->n_bytes());
            f.release(ru, tiu->n_bytes());
        }

        // Pack Down IQ3_XXS into coalesced uint2/uint32 split-plane layout
        {
            const auto* tid = f.tensor(p + "ffn_down_exps.weight");
            const uint8_t* rd = f.tensor_data(*tid);
            const size_t full_rb = 4u * 98u;
            const size_t n_rows  = 256ULL * D;
            // Routed Down is stored as the exact 16-level DOT8 repack (4.25 bpw read, register v_perm decode): it is
            // bit-identical to the compact IQ3_XXS layout on hardware and 1.16-1.65x faster at M = 1 / 5 / 128.
            {
                for (size_t g = 0; g < 2; ++g) {
                    const auto lay = asym_experts ? dot8_iq3_asym_layout(n_rows, ff_size[g])
                                                  : dot8_iq3_layout(n_rows, 512u);
                    std::vector<uint8_t> wpl(lay.w_plane_bytes()), spl(lay.s_plane_bytes());
                    bool rok = true;
                    #pragma omp parallel for schedule(static) reduction(&&:rok)
                    for (size_t r0 = 0; r0 < n_rows; r0 += 4096u) {
                        std::string e;
                        rok = rok && (asym_experts
                            ? dot8_repack_iq3xxs_asym_rows(rd, full_rb, ff_begin[g], lay, r0,
                                  std::min<uint64_t>(n_rows, r0 + 4096u), wpl.data(), spl.data(), &e)
                            : dot8_repack_iq3xxs_rows(rd, full_rb, static_cast<uint32_t>(2u * g), lay, r0,
                                  std::min<uint64_t>(n_rows, r0 + 4096u), wpl.data(), spl.data(), &e));
                    }
                    if (!rok) { res.error = "DOT8 repack of ffn_down_exps failed"; return res; }
                    auto& lw = layers[g][il];
                    lw.w_dn_d8_w_va = rt.vram_arena(g).alloc(wpl.size(), 4096, kfd::Tier::COLD);
                    lw.w_dn_d8_s_va = rt.vram_arena(g).alloc(spl.size(), 4096, kfd::Tier::COLD);
                    if (!lw.w_dn_d8_w_va || !lw.w_dn_d8_s_va) { res.error = "VRAM OOM for the routed Down planes"; return res; }
                    VJ_CHECK(rt.upload_to_vram(g, lw.w_dn_d8_w_va, wpl.data(), wpl.size()));
                    VJ_CHECK(rt.upload_to_vram(g, lw.w_dn_d8_s_va, spl.data(), spl.size()));
                    lw.dn_d8_w_stride = lay.w_row_stride_bytes;
                    lw.dn_d8_s_stride = lay.s_row_stride_bytes;
                }
            }
            if (opt.dot8_ab && il == 47u) {
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
                const size_t bytes = s0.size() * 4u;
                layers[0][il].w_dn_iq3xxs_va = rt.vram_arena(0).alloc(bytes, 4096, kfd::Tier::COLD);
                layers[1][il].w_dn_iq3xxs_va = rt.vram_arena(1).alloc(bytes, 4096, kfd::Tier::COLD);
                if (!layers[0][il].w_dn_iq3xxs_va || !layers[1][il].w_dn_iq3xxs_va) { res.error = "VRAM OOM for the --dot8-ab reference copy"; return res; }
                VJ_CHECK(rt.upload_to_vram(0, layers[0][il].w_dn_iq3xxs_va, s0.data(), bytes));
                VJ_CHECK(rt.upload_to_vram(1, layers[1][il].w_dn_iq3xxs_va, s1.data(), bytes));
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
            std::printf("  uploaded %2u/48 layers | GPU0=%.2f GiB GPU1=%.2f GiB\n",
                        il + 1u, rt.vram_arena(0).used() / 1073741824.0, rt.vram_arena(1).used() / 1073741824.0);
            std::fflush(stdout);
        }
    }

    // Optional built-in blk.48 nextn MTP head (~0.75 GiB/GPU -> enables neural K=4 speculative drafting!)
    MtpLayerDev mtp[2];
    const bool has_mtp = !raw_only_vram && (f.tensor("blk.48.nextn.eh_proj.weight") != nullptr);
    if (has_mtp) {
        const std::string p = "blk.48.";
        std::vector<uint32_t> gu_rows[2];
        for (size_t g = 0; g < 2; ++g) {
            const uint32_t local_ff = ff_size[g];
            gu_rows[g].reserve(256u * local_ff);
            for (uint32_t e = 0; e < 256u; ++e)
                for (uint32_t r = 0; r < local_ff; ++r)
                    gu_rows[g].push_back(shard::expert_row_index(e, hp.n_ff_exp, ff_begin[g], r));
        }
        const auto all_exp_dn_rows = iota_range(0, 256u * D);
        for (size_t g = 0; g < 2; ++g) {
            const uint32_t local_ff = ff_size[g];
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
                    {p + "attn_q.weight", iota_range(static_cast<uint32_t>(g) * 8192u, 8192u)},
                    {p + "attn_k.weight", iota_range(static_cast<uint32_t>(g) * 256u,  256u)},
                    {p + "attn_v.weight", iota_range(static_cast<uint32_t>(g) * 256u,  256u)}
                }, 0, D, nullptr, ml.w_attn_qkv, &res.error)) return res;
            if (!upload_fused_q4a8_tcc(rt, g, f, {{p + "attn_output.weight", all_D_rows}},
                                       static_cast<uint32_t>(g) * 4096u, 4096u, nullptr, ml.w_o, &res.error)) return res;
            if (!upload_f32_as_f16(rt, g, f, p + "ffn_gate_inp.weight", ml.w_router_f16_va, &res.error)) return res;
            if (!upload_f32_rows(rt, g, f, p + "ffn_gate_inp_shexp.weight", nullptr, ml.w_sh_gate_va, &res.error)) return res;
            if (!upload_fused_q4a8_tcc(rt, g, f, {
                    {p + "ffn_gate_shexp.weight", iota_range(ff_begin[g], local_ff)},
                    {p + "ffn_up_shexp.weight",   iota_range(ff_begin[g], local_ff)}
                }, 0, D, nullptr, ml.w_sh_gu, &res.error)) return res;
            if (!upload_fused_q4a8_tcc(rt, g, f, {{p + "ffn_down_shexp.weight", all_D_rows}},
                                       ff_begin[g], local_ff, nullptr, ml.w_sh_down, &res.error)) return res;
            kfd::ArenaTierScope mtp_cold(rt.vram_arena(g), kfd::Tier::COLD);  // MTP routed experts: 8/256 read per draft
            if (!upload_fused_q4a8_tcc(rt, g, f, {{p + "ffn_gate_exps.weight", gu_rows[g]}},
                                       0, D, nullptr, ml.w_exp_gate, &res.error)) return res;
            if (!upload_fused_q4a8_tcc(rt, g, f, {{p + "ffn_up_exps.weight", gu_rows[g]}},
                                       0, D, nullptr, ml.w_exp_up, &res.error)) return res;
            if (!upload_fused_q4a8_tcc(rt, g, f, {{p + "ffn_down_exps.weight", all_exp_dn_rows}},
                                       ff_begin[g], local_ff, nullptr, ml.w_exp_down, &res.error)) return res;
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
        std::printf("  uploaded blk.48 nextn MTP head | GPU0=%.2f GiB GPU1=%.2f GiB\n",
                    rt.vram_arena(0).used() / 1073741824.0, rt.vram_arena(1).used() / 1073741824.0);
    }
    if (gu_dot8_promoted > 0u) {
        std::printf("  promoted %u/48 layers Gate+Up IQ2_S -> exact VJ_Q4_DOT8 (%s)\n",
                    gu_dot8_promoted, raw_only_vram ? "in-place raw M=1" : "co-resident");
    }

    // Optional DFlash2 exact-target block-diffusion drafter (GPU 0 weights + host Q4_K selector codebooks)
    DFlash2Dev df{};
    std::fill_n(df.tap_slot, 48, int8_t(-1));
    const size_t df_gpu  = (std::getenv("VEGAJET_DFLASH2_GPU0") != nullptr) ? 0u : 1u;
    const size_t df_peer = 1u - df_gpu;
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
            const uint64_t kv_bytes = static_cast<uint64_t>(df.swa_win) * df.n_kv * df.head_dim * 2ULL;
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
        std::printf("  uploaded DFlash2 (%uL, B=%u, %u taps, mask_embd=%s) | GPU0=%.2f GiB GPU1=%.2f GiB\n",
                    df.n_layer, df.block_size, n_taps, mask_path.c_str(),
                    rt.vram_arena(0).used() / 1073741824.0, rt.vram_arena(1).used() / 1073741824.0);
    }

    // Zero-roundtrip host-pinned coherent I/O buffer mapped to BOTH GPUs (4 MiB: target I/O + DFlash2 top-16!)
    kfd::KfdBuffer io_buf{};
    if (!rt.alloc_userptr_pinned(4ULL * 1024ULL * 1024ULL, io_buf, true)) {
        res.error = "Failed to allocate pinned I/O buffer";
        return res;
    }
    auto* io_host_u8       = reinterpret_cast<uint8_t*>(io_buf.host_ptr);
    auto* io_toks_host     = reinterpret_cast<int32_t*>(io_host_u8 + 0u);
    auto* io_arg_val_host0 = reinterpret_cast<float*>(io_host_u8 + 4096u);
    auto* io_arg_val_host1 = reinterpret_cast<float*>(io_host_u8 + 4096u + kMaxSpecBatch * kVocabBlocks * 4u);
    auto* io_arg_id_host0  = reinterpret_cast<int32_t*>(io_host_u8 + 262144u);
    auto* io_arg_id_host1  = reinterpret_cast<int32_t*>(io_host_u8 + 262144u + kMaxSpecBatch * kVocabBlocks * 4u);
    const uint64_t io_toks_va     = io_buf.va + 0u;
    const uint64_t io_arg_val_va0 = io_buf.va + 4096u;
    const uint64_t io_arg_val_va1 = io_buf.va + 4096u + kMaxSpecBatch * kVocabBlocks * 4u;
    const uint64_t io_arg_id_va0  = io_buf.va + 262144u;
    const uint64_t io_arg_id_va1  = io_buf.va + 262144u + kMaxSpecBatch * kVocabBlocks * 4u;
    // DFlash2 zero-copy coherent I/O slices in the upper 3 MiB of io_buf:
    // Each top-16 buffer holds up to 16 * kVocabBlocks(240) * 16 * 4 B = 245,760 B -> 256 KiB stride!
    constexpr uint64_t kDfIoBase     = 1048576ULL;
    constexpr uint64_t kDfTopkStride = 262144ULL;
    const uint64_t df_io_xq8_va  = io_buf.va + kDfIoBase;                  // 16 * 3072 = 49,152 B
    const uint64_t df_io_xds_va  = io_buf.va + kDfIoBase + 65536ULL;       // 16 * 96 * 8 = 12,288 B
    const uint64_t df_io_gate_va = io_buf.va + kDfIoBase + 98304ULL;       // 16 * 256 * 4 = 16,384 B
    const uint64_t df_io_val_va0 = io_buf.va + kDfIoBase + 262144ULL;      // +256 KiB
    const uint64_t df_io_val_va1 = df_io_val_va0 + kDfTopkStride;          // +512 KiB
    const uint64_t df_io_id_va0  = df_io_val_va1 + kDfTopkStride;          // +768 KiB
    const uint64_t df_io_id_va1  = df_io_id_va0  + kDfTopkStride;          // +1024 KiB (ends at 2.25 MiB < 4 MiB)
    // v1.2 C1: one 16 B device-fault slot per rank at 3.5 MiB (above the DFlash2 slices and lru_top at
    // 3 MiB+48 KiB, below the 4 MiB end). Host-visible, zeroed by the allocator, rank stamped once.
    constexpr uint64_t kFaultIoBase = 3670016ULL;
    auto* io_fault_host = reinterpret_cast<uint32_t*>(io_host_u8 + kFaultIoBase);
    const uint64_t io_fault_va[2] = {io_buf.va + kFaultIoBase,
                                     io_buf.va + kFaultIoBase + guard::kFaultSlotBytes};
    for (size_t g = 0; g < 2; ++g) {
        io_fault_host[g * guard::kFaultWords + guard::kFaultWRank] = static_cast<uint32_t>(g);
    }
    auto* df_io_gate_host = reinterpret_cast<float*>(io_host_u8 + kDfIoBase + 98304ULL);
    auto* df_io_val_host0 = reinterpret_cast<float*>(io_host_u8 + kDfIoBase + 262144ULL);
    auto* df_io_val_host1 = reinterpret_cast<float*>(io_host_u8 + kDfIoBase + 262144ULL + kDfTopkStride);
    auto* df_io_id_host0  = reinterpret_cast<int32_t*>(io_host_u8 + kDfIoBase + 262144ULL + 2ULL * kDfTopkStride);
    auto* df_io_id_host1  = reinterpret_cast<int32_t*>(io_host_u8 + kDfIoBase + 262144ULL + 3ULL * kDfTopkStride);

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
    const kfd::LoadedKernel* k_ar_ts[2] = {nullptr, nullptr};
    bool ar_ts_on = false;
    uint32_t ar_ts_idx = 0;
    uint32_t pf_gdn_sense = 0u;
    constexpr uint32_t kDownGrpMin = 2u;   // experts with fewer routed tokens stay on the per-token stream
    auto grp_item_cap = [](uint32_t M) { const uint32_t R = M * 8u, U = std::min(R, 256u); return std::min(R, (R + 63u * U + 63u) / 64u); };
    GpuStateDev S[2];
    for (size_t g = 0; g < 2; ++g) {
        const uint32_t local_ff = ff_size[g];
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
        S[g].dense_q_va     = alloc(256u);
        S[g].super_bar_va      = alloc(2048u, 256);
        S[g].gdn_super_desc_va = alloc(36ULL * sizeof(ornith_super::GdnSuperDesc), 256);
        S[g].gqa_super_desc_va = alloc(12ULL * sizeof(ornith_super::GqaSuperDesc), 256);
        S[g].moe_super_desc_va = alloc(48ULL * sizeof(ornith_super::MoeSuperDesc), 256);
        S[g].gdn_sq_va      = alloc(32u * 64u * 4u);
        S[g].gdn_qkv_gab_va = alloc(static_cast<uint64_t>(max_batch) * 10304u * 4u);
        S[g].gdn_conv_va    = alloc(static_cast<uint64_t>(max_batch) * 6144u * 4u);
        S[g].gdn_dec_va     = alloc(static_cast<uint64_t>(max_batch) * 32u * 4u);
        S[g].gdn_bsig_va    = alloc(static_cast<uint64_t>(max_batch) * 32u * 4u);
        S[g].mid_q8_va      = alloc(static_cast<uint64_t>(max_batch) * 4096u);
        S[g].mid_ds_va      = alloc(static_cast<uint64_t>(max_batch) * 128u * 8u);      // float2!
        S[g].attn_qkv_va    = alloc(static_cast<uint64_t>(max_batch) * 8704u * 4u);
        S[g].q_buf_va       = alloc(static_cast<uint64_t>(max_batch) * 4096u * 4u);
        S[g].gate_buf_va    = alloc(static_cast<uint64_t>(max_batch) * 4096u * 4u);
        S[g].router_log_va  = alloc(static_cast<uint64_t>(max_batch) * 256u * 4u);
        S[g].top_ids_va     = alloc(static_cast<uint64_t>(max_batch) * 8u * 4u);
        S[g].top_w_va       = alloc(static_cast<uint64_t>(max_batch) * 8u * 4u);
        S[g].sh_gate_sc_va  = alloc(static_cast<uint64_t>(max_batch) * 4u);
        S[g].sh_gu_va       = alloc(static_cast<uint64_t>(max_batch) * 2u * local_ff * 4u);
        S[g].sh_q8_va       = alloc(static_cast<uint64_t>(max_batch) * local_ff);
        S[g].sh_ds_va       = alloc(static_cast<uint64_t>(max_batch) * (local_ff / 32u) * 8u); // float2!
        S[g].sh_down_va     = alloc(static_cast<uint64_t>(max_batch) * D * 4u);
        S[g].moe_q8_va      = alloc(static_cast<uint64_t>(max_batch) * 8u * local_ff);
        S[g].moe_d_va       = alloc(static_cast<uint64_t>(max_batch) * 8u * (local_ff / 32u) * 4u);
        S[g].moe_mid_f32_va = alloc(8ULL * local_ff * 4ULL, 4096);
        S[g].splitk_part_va = alloc(8ULL * 16384ULL * 4ULL, 4096);
        S[g].moe_exp_off_va   = alloc(257u * 4u);
        S[g].moe_exp_cnt_va   = alloc(256u * 4u);
        S[g].moe_sorted_ms_va = alloc(static_cast<uint64_t>(max_batch) * 8u * 4u);
        if (opt.pf_gdn != 0u) {
            S[g].pf_fo_va      = alloc(static_cast<uint64_t>(max_batch) * 32ull * 128ull * 4ull, 4096);
            S[g].pf_sq_va      = alloc(static_cast<uint64_t>(max_batch) * 32ull * 64ull * 4ull, 4096);
            S[g].pf_gdn_bar_va = alloc(256u, 256);
        }
        const bool warm_alloc = !raw_only_vram && (opt.interactive || !opt.sweep_spec.empty());
        if (opt.pf_gu_tiled || opt.down_grp_prefill == 3u || opt.prefill_dot8 || warm_alloc) S[g].pf_gu_queue_va = alloc(256u, 256);
        if (opt.prefill_dot8 || warm_alloc) {
            S[g].pf_xq4_va      = alloc(static_cast<uint64_t>(max_batch) * 2048ULL, 4096);
            S[g].pf_xds4_va     = alloc(static_cast<uint64_t>(max_batch) * 128ULL * 8ULL, 4096);
            S[g].pf_xsum4_va    = alloc(static_cast<uint64_t>(max_batch) * 128ULL * 4ULL, 4096);
            S[g].moe_mid_q4_va  = alloc(static_cast<uint64_t>(max_batch) * 8ULL * (local_ff / 2u), 4096);
            S[g].moe_mid_ds4_va = alloc(static_cast<uint64_t>(max_batch) * 8ULL * (local_ff / 32u) * 8ULL, 4096);
        }
        if (!opt.dump_logits_path.empty()) {
            S[g].logits_va = alloc(static_cast<uint64_t>(voc_rows[g]) * 4ULL, 4096);
        }
        if (opt.down_grp_prefill != 0u || opt.down_grp_verify != 0u || warm_alloc) {
            // The exact scheme keeps one int32 per lane (R * D * 64 B): sized for the modes actually enabled.
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
        S[g].argmax_val_va  = (g == 0) ? io_arg_val_va0 : io_arg_val_va1;
        S[g].argmax_id_va   = (g == 0) ? io_arg_id_va0  : io_arg_id_va1;

        const uint32_t ssm_ring_slots = raw_only_vram ? 1u : kGdnRingSlots;
        S[g].gdn_conv_ring_va.resize(36);
        S[g].gdn_ssm_ring_va.resize(36);
        for (uint32_t i = 0; i < 36u; ++i) {
            S[g].gdn_conv_ring_va[i] = alloc(kGdnRingSlots * 3ULL * 6144ULL * 4ULL, 4096);
            if (!opt.spec_gdn) {
                S[g].gdn_ssm_ring_va[i] = alloc(static_cast<uint64_t>(ssm_ring_slots) * 32ULL * 128ULL * 128ULL * 4ULL, 4096);
            }
        }
        if (opt.spec_gdn) {
            const uint64_t all_s_bytes = 36ULL * 32ULL * 128ULL * 128ULL * 4ULL;
            S[g].spec_gdn_s_slot_va[0] = alloc(all_s_bytes, 4096);
            S[g].spec_gdn_s_slot_va[1] = alloc(all_s_bytes, 4096);
            S[g].spec_gdn_o_va         = alloc(16ULL * 32ULL * 128ULL * 4ULL, 4096);
            S[g].spec_nq_bar_va        = alloc(256u, 256);
            S[g].spec_gdn_rec_kv_va    = alloc(36ULL * 16ULL * 40ULL * 128ULL * 4ULL, 4096);
            S[g].spec_gdn_rec_dec_va   = alloc(36ULL * 16ULL * 32ULL * 4ULL, 4096);
            S[g].spec_gdn_rec_beta_va  = alloc(36ULL * 16ULL * 32ULL * 4ULL, 4096);
        }
        // Always allocate Split-KV FlashDecoding partial buffers (15.7 MiB/GPU) so core/grid_planner can
        // automatically engage Split-KV whenever n_q_local < n_cu without requiring --spec-flash.
        S[g].spec_fl_part_o_va  = alloc(static_cast<uint64_t>(kMaxSpecBatch) * 16ULL * 60ULL * 256ULL * 4ULL, 4096);
        S[g].spec_fl_part_ml_va = alloc(static_cast<uint64_t>(kMaxSpecBatch) * 16ULL * 60ULL * 2ULL * 4ULL, 4096);
        S[g].attn_k_cache_va.resize(12);
        S[g].attn_v_cache_va.resize(12);
        for (uint32_t i = 0; i < 12u; ++i) {
            S[g].attn_k_cache_va[i] = alloc(static_cast<uint64_t>(opt.max_ctx) * 272ULL, 4096);
            S[g].attn_v_cache_va[i] = alloc(static_cast<uint64_t>(opt.max_ctx) * 272ULL, 4096);
        }
        if (has_mtp) {
            S[g].mtp_k_cache_va = alloc(static_cast<uint64_t>(opt.max_ctx) * 272ULL, 4096);
            S[g].mtp_v_cache_va = alloc(static_cast<uint64_t>(opt.max_ctx) * 272ULL, 4096);
            S[g].mtp_cat_q8_va  = alloc(static_cast<uint64_t>(kMaxSpecBatch) * 2u * D);
            S[g].mtp_cat_ds_va  = alloc(static_cast<uint64_t>(kMaxSpecBatch) * ((2u * D) / 32u) * 8u);
            S[g].mtp_h_pend_va  = alloc(D * 4u);
            S[g].mtp_h_cur_va   = alloc(static_cast<uint64_t>(kMaxSpecBatch) * D * 4u);
            S[g].mtp_h_zero_va  = alloc(D * 4u);
            S[g].mtp_h_stash_va = alloc(16ULL * D * 4u);
            S[g].mtp_moe_ds_va  = alloc(static_cast<uint64_t>(kMaxSpecBatch) * 8u * (local_ff / 32u) * 8u);
        }
        if (alloc_oom) {
            rt.free_buffer(tap_buf);
            rt.free_buffer(io_buf);
            res.error = "VRAM OOM allocating GPU workspace on rank " + std::to_string(g);
            return res;
        }
    }

    res.load_sec      = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load0).count();

    // Wave 5 Dynamic LRU VJ_Q4_DOT8 MoE Gate+Up Cache (up to 44 slots/L in spare VRAM for M >= 4 verify!).
    uint32_t lru_slots = 0u;
    Dot8RowLayout lru_lay[2]{};
    uint64_t lru_w_va[2][48] = {}, lru_s_va[2][48] = {}, lru_e2s_va[2][48] = {};
    std::vector<std::array<int32_t, 256>> lru_e2s_host(48);
    std::unique_ptr<moecache::MoeLruCache> lru_cache;
    const uint64_t lru_top_va = io_buf.va + 3ULL * 1024ULL * 1024ULL;
    auto* lru_top_host = reinterpret_cast<int32_t*>(io_host_u8 + 3ULL * 1024ULL * 1024ULL);
    if ((asym_experts || local_ff == 512u) && D == 3072u && K[0].moe_gu_repack && K[0].moe_gu_lru && layers[0][0].w_gu_d8_w_va == 0u) {
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
            const Dot8RowLayout one_slot = dot8_gu_iq2s_layout(local_ff, D);
            const uint64_t per_slot_all_layers = 48ULL * one_slot.vram_bytes();
            const uint64_t f0 = (rt.vram_arena(0).capacity() > rt.vram_arena(0).used()) ? (rt.vram_arena(0).capacity() - rt.vram_arena(0).used()) : 0ULL;
            const uint64_t f1 = (rt.vram_arena(1).capacity() > rt.vram_arena(1).used()) ? (rt.vram_arena(1).capacity() - rt.vram_arena(1).used()) : 0ULL;
            const uint64_t min_free = std::min(f0, f1);
            const uint64_t reserve_b = 384ULL * 1024ULL * 1024ULL;
            if (min_free > reserve_b + 8ULL * per_slot_all_layers) {
                const uint32_t max_fit = static_cast<uint32_t>((min_free - reserve_b) / per_slot_all_layers);
                lru_slots = std::min<uint32_t>(want_slots, max_fit);
                for (size_t g = 0; g < 2; ++g) lru_lay[g] = dot8_gu_iq2s_layout(static_cast<uint64_t>(lru_slots) * ff_size[g], D);
                std::vector<int32_t> neg_tab(256, -1);
                for (uint32_t il = 0; il < 48u; ++il) lru_e2s_host[il].fill(-1);
                bool ok_lru = true;
                for (size_t g = 0; g < 2 && ok_lru; ++g) {
                    for (uint32_t il = 0; il < 48u && ok_lru; ++il) {
                        lru_w_va[g][il]   = rt.vram_arena(g).alloc(lru_lay[g].w_plane_bytes(), 4096, kfd::Tier::COLD);
                        lru_s_va[g][il]   = lru_lay[g].scale_embedded ? (lru_w_va[g][il] + lru_lay[g].s_offset_bytes)
                                                                   : rt.vram_arena(g).alloc(lru_lay[g].s_plane_bytes(), 256, kfd::Tier::COLD);
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
                                lru_slots, (48.0 * std::max(lru_lay[0].vram_bytes(), lru_lay[1].vram_bytes())) / 1073741824.0);
                } else {
                    lru_slots = 0u;
                }
            }
        }
    }

    for (size_t g = 0; g < 2; ++g) {
        const auto& ar = rt.vram_arena(g);
        if (!ar.tiered()) continue;
        std::printf("[ornith::va] GPU%zu HOT %.2f/%.2f GiB (spilled to COLD %.2f GiB) | COLD %.2f GiB (spilled to HOT %.2f GiB)\n", g,
                    ar.hot_used() / 1073741824.0, ar.hot_limit / 1073741824.0, ar.hot_spill_bytes / 1073741824.0,
                    ar.cold_used() / 1073741824.0, ar.cold_spill_bytes / 1073741824.0);
    }
    res.vram_gib_gpu0 = rt.vram_arena(0).used() / 1073741824.0;
    res.vram_gib_gpu1 = rt.vram_arena(1).used() / 1073741824.0;
    res.knapsack_upgraded = 384;
    res.knapsack_rejected = 144;
    std::printf("[ornith::gpu] Upload complete in %.1fs! GPU0=%.2f / 30.50 GiB | GPU1=%.2f / 30.50 GiB (spare=%.2f GiB total)\n",
                res.load_sec, res.vram_gib_gpu0, res.vram_gib_gpu1,
                (30.50 - res.vram_gib_gpu0) + (30.50 - res.vram_gib_gpu1));

    uint32_t ar_epoch = 1u, ring_head = 0u, conv_head = 0u, spec_gdn_active = 0u, last_dispatches = 0, moe_sense = 0u, gdn_sense = 0u, moe_back_sense = 0u, super_sense = 0u, spec_nq_sense = 0u;
    // Stage profiler: when prof_stage != 0 the matching idempotent stage is enqueued prof_reps times per layer, so
    // (t(reps) - t(1)) / (reps - 1) is that stage's in-situ cost per token. Outputs are unchanged.
    uint32_t prof_stage = 0u, prof_reps = 1u;
    double tb_base_ms = 0.0, tb_stage_ms[9] = {};
    auto reps_of = [&](uint32_t stage) { return (prof_stage == stage) ? prof_reps : 1u; };
    auto reset_state = [&]() {
        rt.reset_allreduce_mailboxes();
        ar_epoch = 1u; ring_head = 0u; conv_head = 0u; spec_gdn_active = 0u;
        for (size_t g = 0; g < 2; ++g) {
            struct MemsetKA { uint64_t dst; uint32_t val; uint32_t pad; uint64_t n; };
            auto zero_buf = [&](uint64_t va, uint64_t bytes) {
                MemsetKA ka{va, 0u, 0u, (bytes + 3u) / 4u};
                VJ_CHECK(rt.enqueue_kernel(g, *K[g].memset_u32, static_cast<uint32_t>(((ka.n + 255u) / 256u) * 256u), 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
            };
            zero_buf(S[g].ar_bar_va, 256u);
            zero_buf(S[g].moe_bar_va, 256u);
            zero_buf(S[g].gdn_bar_va, 8192u);
            if (S[g].spec_nq_bar_va) zero_buf(S[g].spec_nq_bar_va, 256u);
            zero_buf(S[g].super_bar_va, 2048u);
            if (S[g].pf_gdn_bar_va)  zero_buf(S[g].pf_gdn_bar_va, 256u);
            if (S[g].pf_gu_queue_va) zero_buf(S[g].pf_gu_queue_va, 256u);
            zero_buf(S[g].moe_bq_va, 256u);
            zero_buf(S[g].dense_q_va, 256u);
            if (has_mtp) zero_buf(S[g].mtp_h_zero_va, D * 4u);
            for (uint32_t i = 0; i < 36u; ++i) {
                zero_buf(S[g].gdn_conv_ring_va[i], kGdnRingSlots * 3ULL * 6144ULL * 4ULL);
                if (!opt.spec_gdn) {
                    zero_buf(S[g].gdn_ssm_ring_va[i], static_cast<uint64_t>(raw_only_vram ? 1u : kGdnRingSlots) * 32ULL * 128ULL * 128ULL * 4ULL);
                }
            }
            if (opt.spec_gdn) {
                const uint64_t all_s_bytes = 36ULL * 32ULL * 128ULL * 128ULL * 4ULL;
                zero_buf(S[g].spec_gdn_s_slot_va[0], all_s_bytes);
                zero_buf(S[g].spec_gdn_s_slot_va[1], all_s_bytes);
            }
            VJ_CHECK(rt.submit_batch(g));
        }
        VJ_CHECK(rt.wait_idle(0) && rt.wait_idle(1));
    };

    const uint64_t pay_base = rt.mailbox_payload_gpu_va();
    const uint64_t flg_base = rt.mailbox_flags_gpu_va();

    // Populate and upload per-layer M=1 super-kernel descriptors once per GPU (~24 KiB/GPU).
    for (size_t g = 0; g < 2; ++g) {
        const uint32_t local_ff = ff_size[g];
        const size_t peer = 1u - g;
        std::vector<ornith_super::GdnSuperDesc> gdn_d(36);
        std::vector<ornith_super::GqaSuperDesc> gqa_d(12);
        std::vector<ornith_super::MoeSuperDesc> moe_d(48);
        uint32_t gi = 0u, ai = 0u;
        for (uint32_t il = 0u; il < 48u; ++il) {
            const auto& lw = layers[g][il];
            const uint32_t prev_slot = (il > 0u) ? ((il - 1u) * 2u + 1u) : 0u;
            const uint32_t attn_slot = il * 2u;
            const uint32_t moe_slot  = il * 2u + 1u;
            if (lw.is_gdn) {
                auto& d = gdn_d[gi];
                d.x_partial        = S[g].partial_va;
                d.ar_local_rx_base = pay_base + g * kfd::kArGpuBytes;
                d.ar_peer_flag     = flg_base + peer * 4096ULL + (prev_slot & 63u) * 64ULL;
                d.ar_local_flag    = flg_base + g    * 4096ULL + (prev_slot & 63u) * 64ULL;
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
                d.out_peer_tx_base = pay_base + peer * kfd::kArGpuBytes;
                d.out_rows         = lw.w_ssm_out.rows;
                d.out_cols         = lw.w_ssm_out.cols;
                d.out_qs_stride    = lw.w_ssm_out.row_stride_bytes;
                d.out_dm_stride    = 0u;
                d.super_bar        = S[g].super_bar_va;
                d.D                = D;
                d.nk               = 8u;
                d.nv               = 32u;
                d.conv_dim         = 6144u;
                d.z_off            = 6144u;
                d.alpha_off        = 10240u;
                d.beta_off         = 10272u;
                d.eps              = static_cast<float>(hp.rms_eps);
                d.out_peer_flag    = flg_base + peer * 4096ULL + (attn_slot & 63u) * 64ULL;
                ++gi;
            } else {
                auto& d = gqa_d[ai];
                d.x_partial        = S[g].partial_va;
                d.ar_local_rx_base = pay_base + g * kfd::kArGpuBytes;
                d.ar_peer_flag     = flg_base + peer * 4096ULL + (prev_slot & 63u) * 64ULL;
                d.ar_local_flag    = flg_base + g    * 4096ULL + (prev_slot & 63u) * 64ULL;
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
                d.n_q_local        = 16u;
                d.n_kv_local       = 1u;
                d.rope_dims        = hp.rope_dims;
                d.rope_base        = static_cast<float>(hp.rope_base);
                d.w_o_qs           = lw.w_o.qs_va;
                d.w_o_dm           = lw.w_o.dm_va;
                d.out_peer_tx_base = pay_base + peer * kfd::kArGpuBytes;
                d.o_rows           = lw.w_o.rows;
                d.o_cols           = lw.w_o.cols;
                d.o_qs_stride      = lw.w_o.row_stride_bytes;
                d.o_dm_stride      = 0u;
                d.super_bar        = S[g].super_bar_va;
                d.D                = D;
                d.eps              = static_cast<float>(hp.rms_eps);
                d.out_peer_flag    = flg_base + peer * 4096ULL + (attn_slot & 63u) * 64ULL;
                ++ai;
            }
            {
                auto& d = moe_d[il];
                d.x_partial        = S[g].partial_va;
                d.ar_local_rx_base = pay_base + g * kfd::kArGpuBytes;
                d.ar_peer_flag     = flg_base + peer * 4096ULL + (attn_slot & 63u) * 64ULL;
                d.ar_local_flag    = flg_base + g    * 4096ULL + (attn_slot & 63u) * 64ULL;
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
                d.local_ff         = local_ff;
                d.D                = D;
                d.w_gu_iq2s        = lw.w_gu_iq2s_va;
                d.mid_q8           = S[g].moe_q8_va;
                d.mid_d            = S[g].moe_d_va;
                d.sh_dn_qs         = lw.w_sh_down.qs_va;
                d.sh_dn_dm         = lw.w_sh_down.dm_va;
                d.sh_down_out      = S[g].sh_down_va;
                d.w_dn_d8_w        = lw.w_dn_d8_w_va;
                d.w_dn_d8_s        = lw.w_dn_d8_s_va;
                d.out_peer_tx_base = pay_base + peer * kfd::kArGpuBytes;
                d.sh_dn_stride     = lw.w_sh_down.row_stride_bytes;
                d.dn_w_stride      = lw.dn_d8_w_stride;
                d.dn_s_stride      = lw.dn_d8_s_stride;
                d.eps              = static_cast<float>(hp.rms_eps);
                d.super_bar        = S[g].super_bar_va;
                d.out_peer_flag    = flg_base + peer * 4096ULL + (moe_slot & 63u) * 64ULL;
                d.w_gu_d8_w        = lw.w_gu_d8_w_va;
                d.w_gu_d8_s        = lw.w_gu_d8_s_va;
                d.gu_d8_w_stride   = lw.gu_d8_w_stride;
                d.gu_d8_s_stride   = lw.gu_d8_s_stride;
                d.mid_f32          = S[g].moe_mid_f32_va;
            }
        }
        VJ_CHECK(rt.upload_to_vram(g, S[g].gdn_super_desc_va, gdn_d.data(), gdn_d.size() * sizeof(ornith_super::GdnSuperDesc)));
        VJ_CHECK(rt.upload_to_vram(g, S[g].gqa_super_desc_va, gqa_d.data(), gqa_d.size() * sizeof(ornith_super::GqaSuperDesc)));
        VJ_CHECK(rt.upload_to_vram(g, S[g].moe_super_desc_va, moe_d.data(), moe_d.size() * sizeof(ornith_super::MoeSuperDesc)));
    }

    // --pp-stages: after each stage of a batch step, drain that rank (or both ranks) and book the wall time.
    // Between two AllReduces the ranks are independent, so each rank's stage is timed with the other rank idle.
    enum : uint32_t { kStGdnProj = 0, kStGdnConv, kStGdnRecur, kStGdnOut, kStQkv, kStPrep, kStFlash, kStAttnO, kStAr, kStRouter, kStTop8,
                      kStSortShDown, kStGateUp, kStDown, kStVocab, kStCount };
    static const char* const kStName[kStCount] = {"GDN in-proj GEMM", "GDN conv", "GDN recurrence (WY)", "GDN out-proj GEMM", "attn QKV GEMM",
        "attn prep (norm/RoPE/KV write)", "attention", "attn O GEMM", "AllReduce+norm+quant", "router logits + shared Gate/Up GEMM",
        "top-8 + shared SwiGLU", "expert sort + shared Down GEMM", "routed Gate+Up", "routed Down", "vocab argmax"};
    bool st_on = false;
    bool golden_cap_on = false;
    std::string golden_cap_dir;
    uint32_t golden_gqa_il = 3u;
    for (uint32_t il = 0; il < 48u; ++il) { if (!layers[0][il].is_gdn) { golden_gqa_il = il; break; } }
    double st_acc[2][kStCount] = {};
    std::chrono::steady_clock::time_point st_last;
    auto st_mark = [&](uint32_t id, int g) {   // g < 0: both ranks (AllReduce), booked on rank 0
        if (!st_on) return;
        if (g < 0) { VJ_CHECK(rt.submit_batch(0) && rt.submit_batch(1)); VJ_CHECK(rt.wait_idle(0) && rt.wait_idle(1)); }
        else       { VJ_CHECK(rt.submit_batch(static_cast<size_t>(g))); VJ_CHECK(rt.wait_idle(static_cast<size_t>(g))); }
        const auto now = std::chrono::steady_clock::now();
        st_acc[g < 0 ? 0 : g][id] += std::chrono::duration<double, std::milli>(now - st_last).count();
        st_last = now;
    };

    // v1.2 C1: fail-closed barrier / peer-mailbox faults. The super-kernels stop as soon as a spin budget runs
    // out and leave a record in io_fault_host. Every step re-arms both slots and, once the queues have drained,
    // any record becomes a sticky host error so no token can ever be produced from a wedged grid.
    bool device_faulted = false;
    uint64_t dense_persist_launches = 0;
    guard::RunState run_state;
    auto fail_run = [&](const std::string& message) -> void {
        run_state.latch(message);
        res.error = run_state.error();
        rt.abort_dispatches();
        throw guard::RunFailure(run_state.error());
    };
    auto require_token = [&](int32_t id, const char* where) {
        run_state.check();
        if (!guard::token_id_in_range(id, static_cast<uint32_t>(hp.vocab)))
            fail_run(std::string(where) + ": token id " + std::to_string(id) + " is outside the vocabulary");
    };
    auto require_kv_span = [&](uint32_t pos, uint32_t rows) {
        if (!dispatch::kv_span_fits(pos, rows, opt.max_ctx))
            fail_run("KV position or batch exceeds configured context capacity");
    };
    auto require_split_scratch = [&](uint32_t rows, uint32_t heads, uint32_t chunks) {
        if (!dispatch::split_scratch_fits(rows, heads, chunks, kMaxSpecBatch * 60u, 16u))
            fail_run("split attention scratch capacity exceeded");
    };
    auto device_fault_check = [&](const char* where) -> bool {
        for (size_t g = 0; g < 2; ++g) {
            const uint32_t* slot = io_fault_host + g * guard::kFaultWords;
            if (!guard::fault_seen(slot)) continue;
            device_faulted = true;
            if (res.error.empty()) {
                res.error = std::string("device fault during ") + where + ": " + guard::fault_describe(slot);
                std::fprintf(stderr, "[ornith::gpu] DEVICE FAULT: %s\n", res.error.c_str());
            }
            return true;
        }
        return false;
    };

    double max_pwr = 0.0, max_tmem = 0.0;
    HwmonSampler telem;
    size_t parity_code_match = golden::kCodeRef32.size();
    bool bench_spec_mismatch = false;
    std::string bench_spec_mismatch_msg;
    try {
    auto step_batch = [&](const std::vector<int32_t>& toks, uint32_t pos0,
                          bool use_wy, bool use_ring, bool compute_logits,
                          bool capture_dflash_taps = false) -> std::vector<int32_t> {
        const uint32_t M = static_cast<uint32_t>(toks.size());
        // v1.2 C1: a funneled fault stops the run instead of re-dispatching onto a poisoned barrier arena.
        // Unwind the driver before callbacks, recurrent commits or drafter catch-up.
        run_state.check();
        if (device_faulted) fail_run(res.error);
        require_kv_span(pos0, M);
        if (M > max_batch || M > 1024u) fail_run("token batch exceeds staging capacity");
        // v1.2 C3: never turn an out-of-range token id into an embedding address offset.
        const uint32_t n_vocab = static_cast<uint32_t>(hp.vocab);
        for (const int32_t t : toks) {
            if (guard::token_id_in_range(t, n_vocab)) continue;
            if (res.error.empty()) {
                res.error = "token id " + std::to_string(t) + " is outside the vocabulary [0, " +
                            std::to_string(n_vocab) + ") before the embedding lookup";
                std::fprintf(stderr, "[ornith::gpu] BAD TOKEN ID: %s\n", res.error.c_str());
            }
            fail_run(res.error);
        }
        // A fault raised by a dispatch queued outside step_batch (stage-profiler path) is caught here.
        if (device_fault_check("previous dispatch")) fail_run(res.error);
        for (size_t g = 0; g < 2; ++g) guard::fault_clear(io_fault_host + g * guard::kFaultWords);
        if (st_on) st_last = std::chrono::steady_clock::now();
        std::memcpy(io_toks_host, toks.data(), M * sizeof(int32_t));
        __atomic_thread_fence(__ATOMIC_RELEASE);

        uint32_t disp = 0;
        const float eps = static_cast<float>(hp.rms_eps);
        const float rope_base = static_cast<float>(hp.rope_base);
        auto enq_tap = [&](uint32_t state) {
            struct CpKA { uint64_t dst, src, n; };
            const CpKA cka{tap_buf.va + static_cast<uint64_t>(state) * D * 4ULL, S[1].h_va, D};
            VJ_CHECK(rt.enqueue_kernel(1, *K[1].memcpy_u32, ((D + 255u) / 256u) * 256u, 1, 1, 256, 1, 1, &cka, sizeof(cka), true));
        };

        for (size_t g = 0; g < 2; ++g) {
            struct EmbKA { uint64_t qs, ds, toks, h; uint32_t M, D; };
            EmbKA eka{emb_w[g].qs_va, emb_w[g].ds_va, S[g].tok_ids_va, S[g].h_va, M, D};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].embed_q8, M * 256u, 1, 1, 256, 1, 1, &eka, sizeof(eka), true));
            struct RmsKA { uint64_t h, xa, w, xn, xq, xd; uint32_t M, D; float eps; };
            if (g == 1 && tap_on && M == 1u) enq_tap(0u);
            RmsKA rka{S[g].h_va, 0ULL, layers[g][0].attn_norm_va, S[g].x_norm_va, S[g].x_q8_va, S[g].x_ds_va, M, D, eps};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].rms_quant, M * 256u, 1, 1, 256, 1, 1, &rka, sizeof(rka), true));
        }
        disp += 2u;

        const bool use_pf_dot8 = (opt.prefill_dot8 && !use_ring && M >= 64u && K[0].pf_q8_to_q4 != nullptr);
        const bool use_pf_dot8_dense = (use_pf_dot8 && opt.dot8_scope != 2u);
        const bool use_pf_dot8_moe   = (use_pf_dot8 && opt.dot8_scope != 1u);
        auto enq_q8_to_q4 = [&](size_t g, uint64_t xq8, uint64_t xds, uint32_t Kcols, bool barrier = true) {
            struct Q4KA { uint64_t xq8, xds, xq4, xds4, xsum4; uint32_t M, K; float alpha; };
            const Q4KA qka{xq8, xds, S[g].pf_xq4_va, S[g].pf_xds4_va, S[g].pf_xsum4_va, M, Kcols, opt.dot8_clip_alpha};
            const uint32_t nb = Kcols >> 5;
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_q8_to_q4, ((nb + 7u) / 8u) * 256u, M, 1, 256, 1, 1, &qka, sizeof(qka), barrier));
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
        const bool ab_enable_super = !(std::getenv("VEGAJET_ENABLE_SUPER") != nullptr && std::getenv("VEGAJET_ENABLE_SUPER")[0] == '0');
        // v1.2 levers, both default off (unset or any value other than "1" keeps the old path).
        const bool v12_narrow_verify = (std::getenv("VEGAJET_V12_NARROW_VERIFY") != nullptr && std::getenv("VEGAJET_V12_NARROW_VERIFY")[0] == '1');
        const bool v12_fused_nq      = (std::getenv("VEGAJET_V12_VERIFY_FUSED_NQ") != nullptr && std::getenv("VEGAJET_V12_VERIFY_FUSED_NQ")[0] == '1');

        auto enq_gemm_q4 = [&](size_t g, const Q4A8WeightDev& w, uint64_t xq, uint64_t xds, uint64_t y, bool barrier = true, uint64_t peer_tx = 0ULL) {
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
                        const uint32_t n_tiles = (w.rows + 7u) / 8u;
                        const uint32_t pwgs    = v12_dense_wgs();
                        // The tile counter is shared by consecutive dispatches, so the variant is only
                        // taken when the launch is a full AQL barrier (barrier=0 lets packets overlap).
                        if (K[g].gemv_q4_w32p != nullptr && n_tiles >= pwgs && barrier) {
                            // v1.2 D13: resident grid pulls 8-row tiles from an atomic counter instead of
                            // launching one WG per tile (rows=3072 -> 384 WGs on 250-300 slots).
                            struct M1W32PKA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, stride, dm_st;
                                              uint64_t ptx, q; uint32_t n_tiles; };
                            M1W32PKA kap{w.qs_va, w.dm_va, xq, xds, y, w.rows, w.cols, w.row_stride_bytes, 0u,
                                         peer_tx, S[g].dense_q_va, n_tiles};
                            VJ_CHECK(rt.enqueue_kernel(g, *K[g].gemv_q4_w32p, pwgs * 256u, 1, 1, 256, 1, 1, &kap, sizeof(kap), barrier));
                            ++dense_persist_launches;
                        } else {
                            const uint32_t gx = ((w.rows + 7u) / 8u) * 256u;
                            VJ_CHECK(rt.enqueue_kernel(g, *K[g].gemv_q4_w32, gx, 1, 1, 256, 1, 1, &ka, sizeof(ka), barrier));
                        }
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
                } else if (v12_narrow_verify && M >= 2u && M <= 16u && (w.cols % 32u) == 0u && w.cols != 512u &&
                           K[g].gemv_q4_mn != nullptr) {
                    struct GemmPpKA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, M, stride; uint64_t ptx; };
                    GemmPpKA pka{w.qs_va, w.dm_va, xq, xds, y, w.rows, w.cols, M, w.row_stride_bytes, peer_tx};
                    const uint32_t gx = ((w.rows + 7u) / 8u) * 256u;
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].gemv_q4_mn, gx, 1, 1, 256, 1, 1, &pka, sizeof(pka), barrier));
                } else if (M <= 8u && K[g].gemm_q4_m8) {
                    struct GemmPpKA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, M, stride; uint64_t ptx; };
                    GemmPpKA pka{w.qs_va, w.dm_va, xq, xds, y, w.rows, w.cols, M, w.row_stride_bytes, peer_tx};
                    const uint32_t gx = ((w.rows + 7u) / 8u) * 256u;
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].gemm_q4_m8, gx, 1, 1, 256, 1, 1, &pka, sizeof(pka), barrier));
                } else if (M <= 12u && K[g].gemm_q4_m12) {
                    struct GemmPpKA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, M, stride; uint64_t ptx; };
                    GemmPpKA pka{w.qs_va, w.dm_va, xq, xds, y, w.rows, w.cols, M, w.row_stride_bytes, peer_tx};
                    const uint32_t gx = ((w.rows + 7u) / 8u) * 256u;
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].gemm_q4_m12, gx, 1, 1, 256, 1, 1, &pka, sizeof(pka), barrier));
                } else if (M <= 16u && K[g].gemm_q4_m16) {
                    struct GemmPpKA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, M, stride; uint64_t ptx; };
                    GemmPpKA pka{w.qs_va, w.dm_va, xq, xds, y, w.rows, w.cols, M, w.row_stride_bytes, peer_tx};
                    const uint32_t gx = ((w.rows + 7u) / 8u) * 256u;
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].gemm_q4_m16, gx, 1, 1, 256, 1, 1, &pka, sizeof(pka), barrier));
                } else {
                    const uint32_t gx = ((w.rows + 7u) / 8u) * 256u;
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].gemm_q4_tcc, gx, (M + 15u) / 16u, 1, 256, 1, 1, &ka, sizeof(ka), barrier));
                }
            }
        };

        auto enq_p2p_ar_rms = [&](uint32_t slot, uint64_t w_rms0, uint64_t w_rms1, uint32_t do_norm, bool pre_pushed = false) {
            // Segment batches into mailbox-sized slices: 256 rows for FP16 wire (1.5 MiB <= 2 MiB) and
            // 512 rows for INT8 wire (512 * 3456 = 1,769,472 B <= 2 MiB, 128 resident WGs x 4 rows/WG -> 1 slice for M=512!).
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
                        pay_base + peer * kfd::kArGpuBytes + pp * kfd::kArSlotBytes,
                        pay_base + g    * kfd::kArGpuBytes + pp * kfd::kArSlotBytes,
                        flg_base + peer * 4096ULL + (slot & 63u) * 64ULL,
                        flg_base + g    * 4096ULL + (slot & 63u) * 64ULL,
                        S[g].ar_bar_va, S[g].h_va + f32_off, (g == 0) ? w_rms0 : w_rms1,
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
                // A single readiness word cannot hold two unconsumed slice epochs.
                // Keep the single-slice fast path unchanged; drain BOTH ranks before reuse.
                if (dispatch::collective_slice_needs_drain(m0, mc, M)) {
                    if (!rt.submit_batch(0) || !rt.submit_batch(1) ||
                        !rt.wait_idle(0) || !rt.wait_idle(1))
                        fail_run("all-reduce slice completion failed or timed out before mailbox reuse");
                    if (device_fault_check("all-reduce slice")) fail_run(res.error);
                }
            }
        };

        const bool use_m1_super = (!asym_experts && ab_enable_super && !golden_cap_on && M == 1u && !ab_w64 && !ab_f2_front && !ab_b2_back &&
                                   opt.fuse_moe_router && opt.fuse_moe_back && opt.gdn_splitv &&
                                   !opt.ab_no_prepush && !opt.ar_wire_int8 && !opt.spec_flash &&
                                   !ar_ts_on && !st_on && !tap_on && !capture_dflash_taps &&
                                   prof_stage == 0u && !(opt.spec_gdn && use_ring) &&
                                   local_ff == 512u && D == 3072u &&
                                   K[0].m1_gdn_super && K[0].m1_gqa_super && K[0].m1_moe_super);

        if (use_m1_super && v12_dense_persist())
            fail_run("D13 is standalone-only: use VEGAJET_ENABLE_SUPER=0 for BOTH baseline and D13 runs");
        uint32_t gdn_idx = 0, attn_idx = 0;
        if (use_m1_super) {
            const char* swg_env = std::getenv("VEGAJET_SUPER_WGS");
            const uint32_t super_wgs = (swg_env && std::atoi(swg_env) >= 192 && std::atoi(swg_env) <= 240)
                                     ? static_cast<uint32_t>(std::atoi(swg_env)) : 240u;
            const char* q3_fast_env = std::getenv("VEGAJET_MOE_Q3_FAST");
            const bool allow_q3_fast = !(q3_fast_env && q3_fast_env[0] == '0');
            const uint32_t c_in_slot  = conv_head;
            const uint32_t c_out_slot = (conv_head + 1u) % kGdnRingSlots;
            const uint32_t s_rhead    = opt.spec_gdn ? 0u : ring_head;
            const uint32_t s_uring    = (!opt.spec_gdn && use_ring) ? 1u : 0u;
            const uint64_t s_layer_bytes = 32ULL * 128ULL * 128ULL * 4ULL;
            uint32_t pending_moe_ep = 0u;
            for (uint32_t il = 0; il < 48u; ++il) {
                const uint32_t prev_ep = pending_moe_ep;
                const uint32_t attn_ep = ar_epoch++;
                const uint32_t moe_ep  = ar_epoch++;
                if (layers[0][il].is_gdn) {
                    const uint32_t gi = gdn_idx++;
                    ++super_sense;
                    ++gdn_sense;
                    struct GdnSupKA {  // field order == ornith_m1_gdn_super_kernel parameter order
                        uint64_t desc, sring_ov;
                        uint32_t prev_ep, out_ep, cin, cout, shead, uring, ssense, gsense;
                        uint64_t fault;
                    };
                    static_assert(sizeof(GdnSupKA) == 56u, "kernarg ABI drift");
                    for (size_t g = 0; g < 2; ++g) {
                        const uint64_t s_ov = opt.spec_gdn
                            ? (S[g].spec_gdn_s_slot_va[spec_gdn_active] + static_cast<uint64_t>(gi) * s_layer_bytes) : 0ULL;
                        const GdnSupKA ka{
                            S[g].gdn_super_desc_va + static_cast<uint64_t>(gi) * sizeof(ornith_super::GdnSuperDesc),
                            s_ov, prev_ep, attn_ep, c_in_slot, c_out_slot, s_rhead, s_uring, super_sense, gdn_sense, io_fault_va[g]
                        };
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].m1_gdn_super, super_wgs * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
                    }
                } else {
                    const uint32_t ai = attn_idx++;
                    ++super_sense;
                    struct GqaSupKA {  // field order == ornith_m1_gqa_super_kernel parameter order
                        uint64_t desc;
                        uint32_t prev_ep, out_ep, pos0, ssense;
                        uint64_t fault;
                    };
                    static_assert(sizeof(GqaSupKA) == 32u, "kernarg ABI drift");
                    for (size_t g = 0; g < 2; ++g) {
                        const GqaSupKA ka{
                            S[g].gqa_super_desc_va + static_cast<uint64_t>(ai) * sizeof(ornith_super::GqaSuperDesc),
                            prev_ep, attn_ep, pos0, super_sense, io_fault_va[g]
                        };
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].m1_gqa_super, super_wgs * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
                    }
                }
                ++super_sense;
                struct MoeSupKA {  // field order == ornith_m1_moe_super[_q3]_kernel parameter order
                    uint64_t desc;
                    uint32_t ar_ep, out_ep, ssense;
                    uint64_t fault;
                };
                static_assert(sizeof(MoeSupKA) == 32u, "kernarg ABI drift");
                for (size_t g = 0; g < 2; ++g) {
                    const MoeSupKA ka{
                        S[g].moe_super_desc_va + static_cast<uint64_t>(il) * sizeof(ornith_super::MoeSuperDesc),
                        attn_ep, moe_ep, super_sense, io_fault_va[g]
                    };
                    const bool use_q3_k = allow_q3_fast && (super_wgs == 240u) && K[g].m1_moe_super_q3 &&
                                          (layers[g][il].w_gu_d8_w_va != 0ULL) &&
                                          (layers[g][il].gu_d8_w_stride == 2304u);
                    const auto* mk = use_q3_k ? K[g].m1_moe_super_q3 : K[g].m1_moe_super;
                    VJ_CHECK(rt.enqueue_kernel(g, *mk, super_wgs * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
                }
                disp += 2u;
                pending_moe_ep = moe_ep;
                if (il == 0u) { rt.ring_doorbell(0); rt.ring_doorbell(1); }
            }
            // Final layer 47 Post-MoE AllReduce + out_norm (reusing pending_moe_ep already prepushed into BAR0!)
            {
                const uint32_t slot = 47u * 2u + 1u;
                const uint32_t pp   = pending_moe_ep & 1u;
                const uint32_t dn   = compute_logits ? 1u : 0u;
                for (size_t g = 0; g < 2; ++g) {
                    const size_t peer = 1u - g;
                    struct ArKA {
                        uint64_t xp, ptx, lrx, pfl, lfl, bar, h, w, xn, xq, xd;
                        uint32_t M, D, ep, dn;
                        float eps;
                    };
                    const ArKA aka{
                        S[g].partial_va,
                        pay_base + peer * kfd::kArGpuBytes + pp * kfd::kArSlotBytes,
                        pay_base + g    * kfd::kArGpuBytes + pp * kfd::kArSlotBytes,
                        flg_base + peer * 4096ULL + (slot & 63u) * 64ULL,
                        flg_base + g    * 4096ULL + (slot & 63u) * 64ULL,
                        S[g].ar_bar_va, S[g].h_va, out_norm_va[g],
                        S[g].x_norm_va, S[g].x_q8_va, S[g].x_ds_va, 1u, D, pending_moe_ep, dn, eps
                    };
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].p2p_ar_prepush, 256u, 1, 1, 256, 1, 1, &aka, sizeof(aka), true));
                }
                disp += 1u;
            }
        } else
        for (uint32_t il = 0; il < 48u; ++il) {
            auto gcap_sync = [&]() {
                VJ_CHECK(rt.submit_batch(0) && rt.submit_batch(1));
                VJ_CHECK(rt.wait_idle(0) && rt.wait_idle(1));
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
                const bool fuse_gdn_m1 = (M == 1u && opt.gdn_splitv && !(opt.spec_gdn && use_ring) && K[0].gdn_conv_recur_sv != nullptr);
                if (fuse_gdn_m1) ++gdn_sense;
                const uint32_t c_in_slot  = conv_head;
                const uint32_t c_out_slot = (conv_head + 1u) % kGdnRingSlots;
                for (size_t g = 0; g < 2; ++g) {
                    const auto& lw = layers[g][il];
                    if (use_pf_dot8_dense && (lw.w_gdn_qkv_gab.cols % 64u) == 0u) {
                        enq_q8_to_q4(g, S[g].x_q8_va, S[g].x_ds_va, D, true);
                        for (uint32_t r = 0; r < reps_of(1u); ++r)
                            enq_gemm_q4_d8(g, lw.w_gdn_qkv_gab, S[g].gdn_qkv_gab_va, true);
                    } else {
                        for (uint32_t r = 0; r < reps_of(1u); ++r)
                            enq_gemm_q4(g, lw.w_gdn_qkv_gab, S[g].x_q8_va, S[g].x_ds_va, S[g].gdn_qkv_gab_va, true);
                    }
                    st_mark(kStGdnProj, static_cast<int>(g));

                    const uint64_t s_layer_bytes = 32ULL * 128ULL * 128ULL * 4ULL;
                    const uint64_t s_cur_va = opt.spec_gdn
                        ? (S[g].spec_gdn_s_slot_va[spec_gdn_active] + static_cast<uint64_t>(gi) * s_layer_bytes)
                        : (S[g].gdn_ssm_ring_va[gi] + static_cast<uint64_t>(ring_head) * s_layer_bytes);
                    const uint64_t s_ring_base = opt.spec_gdn ? s_cur_va : S[g].gdn_ssm_ring_va[gi];
                    const uint32_t s_rhead     = opt.spec_gdn ? 0u : ring_head;

                    if (fuse_gdn_m1) {
                        struct ConvRecSvKA {
                            uint64_t qkv, cw, cring, dt, sa, sring, nw, oq, od, bar, sq;
                            uint32_t nk, nv, cdim, strd, zoff, aoff, boff, cin, cout, shead, uring, sense;
                            float eps;
                        };
                        const ConvRecSvKA crka{
                            S[g].gdn_qkv_gab_va, lw.ssm_conv_w_va, S[g].gdn_conv_ring_va[gi],
                            lw.ssm_dt_va, lw.ssm_a_va, s_ring_base, lw.ssm_norm_va,
                            S[g].mid_q8_va, S[g].mid_ds_va, S[g].gdn_bar_va, S[g].gdn_sq_va,
                            8u, 32u, 6144u, 10304u, 6144u, 10240u, 10272u,
                            c_in_slot, c_out_slot, s_rhead, (!opt.spec_gdn && use_ring) ? 1u : 0u, gdn_sense, eps
                        };
                        // v1.2 D6: VEGAJET_V12_GDN_SPLITV=1 swaps in the column-thread variant (same args, same math).
                        const kfd::LoadedKernel* const cr_k =
                            (K[g].gdn_conv_recur_ct != nullptr) ? K[g].gdn_conv_recur_ct : K[g].gdn_conv_recur_sv;
                        VJ_CHECK(rt.enqueue_kernel(g, *cr_k, 128u * 256u, 1, 1, 256, 1, 1, &crka, sizeof(crka), true));
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
                                      M, 8u, 32u, 6144u, 10304u, 10240u, 10272u, c_in_slot, c_out_slot, eps};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_gdn_conv, 49u * 128u, (M + 15u) / 16u, 1, 128, 1, 1, &clka, sizeof(clka), true));
                    } else {
                        ConvL2KA clka{S[g].gdn_qkv_gab_va, lw.ssm_conv_w_va, S[g].gdn_conv_ring_va[gi], S[g].gdn_conv_va,
                                      lw.ssm_dt_va, lw.ssm_a_va, S[g].gdn_dec_va, S[g].gdn_bsig_va,
                                      M, 8u, 32u, 6144u, 10304u, 10240u, 10272u, conv_head, use_ring ? 1u : 0u, eps};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].gdn_conv_l2, 49u * 128u, 1, 1, 128, 1, 1, &clka, sizeof(clka), true));
                    }
                    st_mark(kStGdnConv, static_cast<int>(g));

                    if (opt.spec_gdn && use_ring && M <= 16u) {
                        struct SpecGdnKA {
                            uint64_t sin, sout, qsrc, ksrc, vsrc, dec, beta, oout, rkv, rdec, rbeta;
                            uint32_t M, nk, nv, s_strd, r_strd, lbase, s_lstrd, src_lstrd, gb_lstrd, r_lstrd, rgb_lstrd;
                        };
                        const uint64_t s_dst_va = S[g].spec_gdn_s_slot_va[1u - spec_gdn_active] + static_cast<uint64_t>(gi) * s_layer_bytes;
                        const SpecGdnKA sgka{
                            s_cur_va, s_dst_va, S[g].gdn_conv_va, S[g].gdn_conv_va + 8ULL * 128ULL * 4ULL,
                            S[g].gdn_conv_va + 16ULL * 128ULL * 4ULL, S[g].gdn_dec_va, S[g].gdn_bsig_va,
                            S[g].spec_gdn_o_va, S[g].spec_gdn_rec_kv_va, S[g].spec_gdn_rec_dec_va, S[g].spec_gdn_rec_beta_va,
                            M, 8u, 32u, 6144u, 40u * 128u, gi, 0u, 0u, 0u, 16u * 40u * 128u, 16u * 32u
                        };
                        if (g == 0 && v12_fused_nq) ++spec_nq_sense;
                        const auto* fused = K[g].spec_gdn_ver_nq;
                        const bool fused_resident = v12_fused_nq && fused && fused->wavefront_size == 64u &&
                            fused->max_flat_workgroup_size >= 256u && fused->private_segment_fixed_size == 0u &&
                            dispatch::fused_verify_resident(fused->from_msgpack_note, rt.devices()[g].n_cu,
                                fused->vgpr_count, fused->sgpr_count, fused->group_segment_fixed_size,
                                fused->resident_waves_per_simd);
                        if (v12_fused_nq && fused_resident && S[g].spec_nq_bar_va != 0) {
                            // v1.2 S5: verify + norm/quant in one launch (grid barrier inside).
                            struct SpecGdnNqKA {
                                uint64_t sin, sout, qsrc, ksrc, vsrc, dec, beta, oout, rkv, rdec, rbeta;
                                uint64_t gab, nw, oq, od, bar;
                                uint32_t M, nk, nv, s_strd, r_strd, lbase, s_lstrd, src_lstrd, gb_lstrd, r_lstrd, rgb_lstrd;
                                uint32_t strd_in, zoff, sense; float eps;
                                uint64_t fault;
                            };
                            static_assert(offsetof(SpecGdnNqKA, fault) == 192u && sizeof(SpecGdnNqKA) == 200u);
                            const SpecGdnNqKA fnka{
                                sgka.sin, sgka.sout, sgka.qsrc, sgka.ksrc, sgka.vsrc, sgka.dec, sgka.beta, sgka.oout,
                                sgka.rkv, sgka.rdec, sgka.rbeta,
                                S[g].gdn_qkv_gab_va, lw.ssm_norm_va, S[g].mid_q8_va, S[g].mid_ds_va, S[g].spec_nq_bar_va,
                                M, 8u, 32u, 6144u, 40u * 128u, gi, 0u, 0u, 0u, 16u * 40u * 128u, 16u * 32u,
                                10304u, 6144u, spec_nq_sense, eps, io_fault_va[g]};
                            VJ_CHECK(rt.enqueue_kernel(g, *K[g].spec_gdn_ver_nq, 128u * 256u, 1, 1, 256, 1, 1, &fnka, sizeof(fnka), true));
                        } else {
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].spec_gdn_ver, 128u * 256u, 1, 1, 256, 1, 1, &sgka, sizeof(sgka), true));
                        struct SpecGdnNormKA { uint64_t ob, gab, nw, oq, od; uint32_t M, nv, strd, zoff; float eps; };
                        const SpecGdnNormKA snka{S[g].spec_gdn_o_va, S[g].gdn_qkv_gab_va, lw.ssm_norm_va,
                                                 S[g].mid_q8_va, S[g].mid_ds_va, M, 32u, 10304u, 6144u, eps};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].spec_gdn_nq, ((M * 32u + 3u) / 4u) * 256u, 1, 1, 256, 1, 1, &snka, sizeof(snka), true));
                        }
                    } else if (std::getenv("VEGAJET_PREFILL_CODEX_GDN") != nullptr &&
                               std::strcmp(std::getenv("VEGAJET_PREFILL_CODEX_GDN"), "1") == 0 &&
                               K[g].pf_gdn_wy_codex && K[g].pf_gdn_nq_codex && use_wy && M >= 16u && !use_ring) {
                        struct WsKA { uint64_t S, cq, dec, bs, fo, sq; uint32_t M, nk, nv, cdim; };
                        const WsKA wska{s_cur_va, S[g].gdn_conv_va, S[g].gdn_dec_va, S[g].gdn_bsig_va, S[g].pf_fo_va, S[g].pf_sq_va, M, 8u, 32u, 6144u};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_gdn_wy_codex, 128u * 256u, 1, 1, 256, 1, 1, &wska, sizeof(wska), true));
                        struct NqKA { uint64_t fo, sq, gab, nw, oq, od; uint32_t M, nv, strd, zoff; float eps; };
                        const NqKA nqka{S[g].pf_fo_va, S[g].pf_sq_va, S[g].gdn_qkv_gab_va, lw.ssm_norm_va, S[g].mid_q8_va, S[g].mid_ds_va, M, 32u, 10304u, 6144u, eps};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_gdn_nq_codex, 128u * 256u, 1, 1, 256, 1, 1, &nqka, sizeof(nqka), true));
                    } else if (opt.pf_gdn == 1u && use_wy && M >= 16u && !use_ring) {
                        // Split-V batch recurrence, 128 workgroups, one grid barrier per dispatch, norm fused.
                        struct SvbKA { uint64_t sring, cq, gab, dec, bs, nw, oq, od, bar, fo, sq;
                                       uint32_t M, nk, nv, cdim, strd, zoff, rhead, uring, sense; float eps; uint32_t fuse; };
                        if (g == 0) ++pf_gdn_sense;
                        const SvbKA ska{s_ring_base, S[g].gdn_conv_va, S[g].gdn_qkv_gab_va, S[g].gdn_dec_va, S[g].gdn_bsig_va,
                                        lw.ssm_norm_va, S[g].mid_q8_va, S[g].mid_ds_va, S[g].pf_gdn_bar_va, S[g].pf_fo_va, S[g].pf_sq_va,
                                        M, 8u, 32u, 6144u, 10304u, 6144u, s_rhead, 0u, pf_gdn_sense, eps, 1u};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_gdn_sv, 128u * 256u, 1, 1, 256, 1, 1, &ska, sizeof(ska), true));
                    } else if (opt.pf_gdn == 2u && use_wy && M >= 16u && !use_ring) {
                        struct WsKA { uint64_t S, cq, dec, bs, fo, sq; uint32_t M, nk, nv, cdim; };
                        const WsKA wska{s_cur_va, S[g].gdn_conv_va, S[g].gdn_dec_va, S[g].gdn_bsig_va, S[g].pf_fo_va, S[g].pf_sq_va, M, 8u, 32u, 6144u};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_gdn_wy, 128u * 256u, 1, 1, 256, 1, 1, &wska, sizeof(wska), true));
                        struct NqKA { uint64_t fo, sq, gab, nw, oq, od; uint32_t M, nv, strd, zoff; float eps; };
                        const NqKA nqka{S[g].pf_fo_va, S[g].pf_sq_va, S[g].gdn_qkv_gab_va, lw.ssm_norm_va, S[g].mid_q8_va, S[g].mid_ds_va, M, 32u, 10304u, 6144u, eps};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_gdn_nq, 128u * 256u, 1, 1, 256, 1, 1, &nqka, sizeof(nqka), true));
                    } else if (use_wy && M >= 8u && !use_ring) {
                        struct WyKA { uint64_t S, cq, gab, dec, bs, nw, oq, od; uint32_t M, nk, nv, cdim, strd, zoff; float eps; };
                        WyKA wka{s_cur_va, S[g].gdn_conv_va, S[g].gdn_qkv_gab_va, S[g].gdn_dec_va, S[g].gdn_bsig_va,
                                 lw.ssm_norm_va, S[g].mid_q8_va, S[g].mid_ds_va, M, 8u, 32u, 6144u, 10304u, 6144u, eps};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].gdn_wy, 32u * 256u, 1, 1, 256, 1, 1, &wka, sizeof(wka), true));
                    } else if (opt.gdn_splitv && M == 1u) {
                        struct RecSvKA { uint64_t sring, cq, gab, dec, bs, nw, oq, od, bar, sq; uint32_t nk, nv, cdim, strd, zoff, rhead, uring, sense; float eps; };
                        if (g == 0) ++gdn_sense; // one barrier per dispatch; both ranks run the same layer sequence
                        RecSvKA rka{s_ring_base, S[g].gdn_conv_va, S[g].gdn_qkv_gab_va, S[g].gdn_dec_va, S[g].gdn_bsig_va,
                                    lw.ssm_norm_va, S[g].mid_q8_va, S[g].mid_ds_va, S[g].gdn_bar_va, S[g].gdn_sq_va,
                                    8u, 32u, 6144u, 10304u, 6144u, s_rhead, (!opt.spec_gdn && use_ring) ? 1u : 0u, gdn_sense, eps};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].gdn_recur_sv, 128u * 256u, 1, 1, 256, 1, 1, &rka, sizeof(rka), true));
                    } else {
                        struct RecKA { uint64_t sring, cq, gab, dec, bs, nw, oq, od; uint32_t M, nk, nv, cdim, strd, zoff, rhead, uring; float eps; };
                        RecKA rka{s_ring_base, S[g].gdn_conv_va, S[g].gdn_qkv_gab_va, S[g].gdn_dec_va, S[g].gdn_bsig_va,
                                  lw.ssm_norm_va, S[g].mid_q8_va, S[g].mid_ds_va, M, 8u, 32u, 6144u, 10304u, 6144u, s_rhead, (!opt.spec_gdn && use_ring) ? 1u : 0u, eps};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].gdn_recur, 32u * 1024u, 1, 1, 1024, 1, 1, &rka, sizeof(rka), true));
                    }
                    st_mark(kStGdnRecur, static_cast<int>(g));
                    }
                    const bool do_ep_gdn = (!opt.ab_no_prepush && M <= 16u && !ar_ts_on && !opt.ar_wire_int8 && !(opt.prefill_ar_int8 && !use_ring && M >= 64u));
                    const uint64_t ptx_gdn = do_ep_gdn ? (pay_base + (1u - g) * kfd::kArGpuBytes + (ar_epoch & 1u) * kfd::kArSlotBytes) : 0ULL;
                    if (use_pf_dot8_dense && (lw.w_ssm_out.cols % 64u) == 0u) {
                        enq_q8_to_q4(g, S[g].mid_q8_va, S[g].mid_ds_va, lw.w_ssm_out.cols, true);
                        for (uint32_t r = 0; r < reps_of(2u); ++r)
                            enq_gemm_q4_d8(g, lw.w_ssm_out, S[g].partial_va, true);
                    } else {
                        for (uint32_t r = 0; r < reps_of(2u); ++r)
                            enq_gemm_q4(g, lw.w_ssm_out, S[g].mid_q8_va, S[g].mid_ds_va, S[g].partial_va, true, ptx_gdn);
                    }
                    st_mark(kStGdnOut, static_cast<int>(g));
                }
                disp += fuse_gdn_m1 ? 3u : 4u;
            } else {
                const uint32_t ai = attn_idx++;
                const uint32_t n_q_local = 16u, n_kv_local = 1u;
                // Unified Split-KV chunk_len (32 up to 1920, 64 up to 3840, 128 up to 7680, 256 above) matching
                // ornith_m1_gqa_super_kernel so M=1 and M=2..16 verify see 100% identical chunk boundaries!
                const uint32_t skv_span = pos0 + M;
                const uint32_t skv_cl   = (skv_span <= 1920u) ? 32u
                                        : ((skv_span <= 3840u) ? 64u
                                        : ((skv_span <= 7680u) ? 128u : 256u));
                const uint32_t skv_nc   = std::max<uint32_t>(1u, (skv_span + skv_cl - 1u) / skv_cl);
                const bool use_auto_splitkv = (!ab_no_splitkv && !opt.spec_flash && skv_nc > 1u &&
                                               (use_ring ? M <= 16u : M < 16u) && K[0].flash_skv_p1 != nullptr);
                const bool fuse_attn_m1 = (M == 1u && !opt.spec_flash && !use_auto_splitkv &&
                                           reps_of(4u) == 1u && reps_of(5u) == 1u && K[0].attn_prep_flash_m1 != nullptr);
                for (size_t g = 0; g < 2; ++g) {
                    const auto& lw = layers[g][il];
                    if (use_pf_dot8_dense && (lw.w_attn_qkv.cols % 64u) == 0u) {
                        enq_q8_to_q4(g, S[g].x_q8_va, S[g].x_ds_va, D, true);
                        for (uint32_t r = 0; r < reps_of(3u); ++r)
                            enq_gemm_q4_d8(g, lw.w_attn_qkv, S[g].attn_qkv_va, true);
                    } else {
                        for (uint32_t r = 0; r < reps_of(3u); ++r)
                            enq_gemm_q4(g, lw.w_attn_qkv, S[g].x_q8_va, S[g].x_ds_va, S[g].attn_qkv_va, true);
                    }
                    st_mark(kStQkv, static_cast<int>(g));

                    if (fuse_attn_m1) {
                        struct PfM1KA { uint64_t qkv, qw, kw, kc, vc, oq, od; uint32_t pos0, nq, nkv, rdims; float rbase, eps; };
                        const PfM1KA pfka{S[g].attn_qkv_va, lw.q_norm_va, lw.k_norm_va,
                                          S[g].attn_k_cache_va[ai], S[g].attn_v_cache_va[ai],
                                          S[g].mid_q8_va, S[g].mid_ds_va,
                                          pos0, n_q_local, n_kv_local, hp.rope_dims, rope_base, eps};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].attn_prep_flash_m1, n_q_local * 256u, 1, 1, 256, 1, 1, &pfka, sizeof(pfka), true));
                        st_mark(kStFlash, static_cast<int>(g));
                    } else {
                    struct PrepKA { uint64_t qkv, qw, kw, qo, go, kc, vc; uint32_t M, pos0, nq, nkv, rdims; float rbase, eps; };
                    PrepKA pka{S[g].attn_qkv_va, lw.q_norm_va, lw.k_norm_va,
                               S[g].q_buf_va, S[g].gate_buf_va, S[g].attn_k_cache_va[ai], S[g].attn_v_cache_va[ai],
                               M, pos0, 16u, 1u, hp.rope_dims, rope_base, eps};
                    for (uint32_t r = 0; r < reps_of(4u); ++r)
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].attn_prep, 17u * 256u, M, 1, 256, 1, 1, &pka, sizeof(pka), true));
                    st_mark(kStPrep, static_cast<int>(g));

                    struct FlKA { uint64_t qo, go, kc, vc, oq, od; uint32_t M, pos0, nq, nkv; };
                    FlKA fka{S[g].q_buf_va, S[g].gate_buf_va, S[g].attn_k_cache_va[ai], S[g].attn_v_cache_va[ai],
                             S[g].mid_q8_va, S[g].mid_ds_va, M, pos0, n_q_local, n_kv_local};
                    if (K[g].pf_fa2 && !use_ring && M >= 16u) {
                        const bool use_codex_fa2 = (std::getenv("VEGAJET_PREFILL_CODEX_FA2") != nullptr &&
                                                    std::strcmp(std::getenv("VEGAJET_PREFILL_CODEX_FA2"), "1") == 0 &&
                                                    K[g].pf_fa2_codex != nullptr && n_q_local == 16u && n_kv_local == 1u);
                        VJ_CHECK(rt.enqueue_kernel(g, *(use_codex_fa2 ? K[g].pf_fa2_codex : K[g].pf_fa2), M * 256u, n_kv_local, 1, 256, 1, 1, &fka, sizeof(fka), true));
                    } else if (opt.spec_flash && K[g].spec_flash_part && (use_ring ? M <= 16u : M < 16u)) {
                        const uint32_t span = pos0 + M;
                        const uint32_t chunk_len = (span <= 1920u) ? 32u : ((span <= 3840u) ? 64u : ((span <= 7680u) ? 128u : 256u));
                        const uint32_t n_chunks = 1u + (span - 1u) / chunk_len;
                        require_split_scratch(M, n_q_local, n_chunks);
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
                                           S[g].spec_fl_part_o_va, S[g].spec_fl_part_ml_va, M, pos0, n_q_local, n_kv_local, skv_nc, skv_cl};
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

                    const bool do_ep_gqa = (!opt.ab_no_prepush && M <= 16u && !ar_ts_on && !opt.ar_wire_int8 && !(opt.prefill_ar_int8 && !use_ring && M >= 64u));
                    const uint64_t ptx_gqa = do_ep_gqa ? (pay_base + (1u - g) * kfd::kArGpuBytes + (ar_epoch & 1u) * kfd::kArSlotBytes) : 0ULL;
                    if (use_pf_dot8_dense && (lw.w_o.cols % 64u) == 0u) {
                        enq_q8_to_q4(g, S[g].mid_q8_va, S[g].mid_ds_va, lw.w_o.cols, true);
                        for (uint32_t r = 0; r < reps_of(6u); ++r)
                            enq_gemm_q4_d8(g, lw.w_o, S[g].partial_va, true);
                    } else {
                        for (uint32_t r = 0; r < reps_of(6u); ++r)
                            enq_gemm_q4(g, lw.w_o, S[g].mid_q8_va, S[g].mid_ds_va, S[g].partial_va, true, ptx_gqa);
                    }
                    st_mark(kStAttnO, static_cast<int>(g));
                }
                disp += fuse_attn_m1 ? 3u : (use_auto_splitkv ? 5u : 4u);
            }

            const bool ep_pushed_attn = (!opt.ab_no_prepush && M <= 16u && !ar_ts_on && !opt.ar_wire_int8 && !(opt.prefill_ar_int8 && !use_ring && M >= 64u));
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
                                    33u + (layers[0][il].w_sh_gu.rows + 7u) / 8u <= 240u &&
                                    (!asym_experts || 33u + (layers[1][il].w_sh_gu.rows + 7u) / 8u <= 240u);
            const bool use_f2_front = (ab_f2_front && fuse_front && K[0].moe_m1_rtr_shgu != nullptr && K[0].moe_m1_top8_swiglu != nullptr);
            const bool fuse_ar_front = (!use_f2_front && fuse_front && M == 1u && ep_pushed_attn && !ar_ts_on && !opt.ar_wire_int8 &&
                                        !st_on && K[0].p2p_ar_moe_m1_front != nullptr);
            if (!fuse_ar_front) {
                enq_p2p_ar_rms(il * 2u, layers[0][il].post_attn_norm_va, layers[1][il].post_attn_norm_va, 1u, ep_pushed_attn);
                st_mark(kStAr, -1);
            }
            if (golden_cap_on && il == 0u) {
                gcap_sync();
                gcap_dump("moe_xnorm", S[0].x_norm_va, static_cast<uint64_t>(M) * D * 4ULL);
                gcap_dump("moe_xq8", S[0].x_q8_va, static_cast<uint64_t>(M) * D);
                gcap_dump("moe_xds", S[0].x_ds_va, static_cast<uint64_t>(M) * (D / 32u) * 8ULL);
            }

            // Decided once per layer for both ranks: a rank that skipped the sense bump would fall through the barrier.
            const bool fuse_back = opt.fuse_moe_back && M == 1u && (asym_experts || local_ff == 512u) && (!asym_experts || opt.moe_back_wgs == 224u) && opt.moe_back_wgs >= 1u && opt.moe_back_wgs <= 240u &&
                                   layers[0][il].w_sh_down.cols == ff_size[0] && layers[1][il].w_sh_down.cols == ff_size[1] &&
                                   layers[0][il].w_gu_d8_w_va == 0u;
            const bool is_prefill_moe = (!use_ring && M >= 16u) || M > 16u;
            const uint32_t dgrp = is_prefill_moe ? opt.down_grp_prefill : ((M >= 2u) ? opt.down_grp_verify : 0u);
            // Grouped/tiled Down and the monolithic superkernel retain their legacy 512-column layouts.
            const bool use_pf_down_tiled = (!asym_experts && S[0].dgrp_buf_va != 0u && S[0].pf_gu_queue_va != 0u && !fuse_back && dgrp == 3u && K[0].pf_down_tiled && D % 128u == 0u && local_ff == 512u);
            const bool use_dgrp = (!asym_experts && S[0].dgrp_buf_va != 0u && !fuse_back && !use_pf_down_tiled && dgrp != 0u && dgrp != 3u && D % 64u == 0u);
            if (fuse_front && !use_f2_front) moe_sense += reps_of(7u);
            if (fuse_back && (!ab_b2_back || asym_experts)) moe_back_sense += reps_of(8u);
            const uint32_t ar_front_ep = fuse_ar_front ? ar_epoch++ : 0u;
            for (size_t g = 0; g < 2; ++g) {
                const uint32_t local_ff = ff_size[g];
                const auto& lw = layers[g][il];
                if (use_f2_front) {
                    struct RsKA { uint64_t wr, wsh, xn, shq, shdm, xq, xds, logits, shgsc, shgu; uint32_t D, sh_rows, sh_stride; };
                    struct TsKA { uint64_t logits, shgu, shgsc, tids, tw, oq, od; uint32_t lff; };
                    const uint32_t rtr_sh_wgs = 128u + (lw.w_sh_gu.rows + 3u) / 4u + 1u;
                    const RsKA rk{lw.w_router_f16_va, lw.w_sh_gate_va, S[g].x_norm_va, lw.w_sh_gu.qs_va, lw.w_sh_gu.dm_va,
                                  S[g].x_q8_va, S[g].x_ds_va, S[g].router_log_va, S[g].sh_gate_sc_va, S[g].sh_gu_va,
                                  D, lw.w_sh_gu.rows, lw.w_sh_gu.row_stride_bytes};
                    const TsKA tk{S[g].router_log_va, S[g].sh_gu_va, S[g].sh_gate_sc_va, S[g].top_ids_va, S[g].top_w_va,
                                  S[g].sh_q8_va, S[g].sh_ds_va, local_ff};
                    for (uint32_t r = 0; r < reps_of(7u); ++r) {
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_m1_rtr_shgu, rtr_sh_wgs * 64u, 1, 1, 64, 1, 1, &rk, sizeof(rk), true));
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_m1_top8_swiglu, 2u * 256u, 1, 1, 256, 1, 1, &tk, sizeof(tk), true));
                    }
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
                        pay_base + peer * kfd::kArGpuBytes + pp * kfd::kArSlotBytes,
                        pay_base + g    * kfd::kArGpuBytes + pp * kfd::kArSlotBytes,
                        flg_base + peer * 4096ULL + (slot & 63u) * 64ULL,
                        flg_base + g    * 4096ULL + (slot & 63u) * 64ULL,
                        S[g].h_va, lw.post_attn_norm_va, S[g].x_norm_va, S[g].x_q8_va, S[g].x_ds_va,
                        lw.w_router_f16_va, lw.w_sh_gate_va, lw.w_sh_gu.qs_va, lw.w_sh_gu.dm_va,
                        S[g].router_log_va, S[g].sh_gu_va, S[g].top_ids_va, S[g].top_w_va,
                        S[g].sh_q8_va, S[g].sh_ds_va, S[g].moe_bar_va,
                        D, ar_front_ep, eps, lw.w_sh_gu.rows, lw.w_sh_gu.row_stride_bytes, local_ff, s0
                    };
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].p2p_ar_moe_m1_front, (33u + (lw.w_sh_gu.rows + 7u) / 8u) * 256u, 1, 1, 256, 1, 1, &afka, sizeof(afka), true));
                    for (uint32_t r = 1; r < reps_of(7u); ++r) {
                        struct FrontKA { uint64_t wr, wsh, xn, shq, shdm, xq, xds, logits, shgu, tids, tw, oq, od, bar; uint32_t D, sh_rows, sh_stride, lff, sense; };
                        const FrontKA fka{lw.w_router_f16_va, lw.w_sh_gate_va, S[g].x_norm_va, lw.w_sh_gu.qs_va, lw.w_sh_gu.dm_va,
                                          S[g].x_q8_va, S[g].x_ds_va, S[g].router_log_va, S[g].sh_gu_va, S[g].top_ids_va, S[g].top_w_va,
                                          S[g].sh_q8_va, S[g].sh_ds_va, S[g].moe_bar_va,
                                          D, lw.w_sh_gu.rows, lw.w_sh_gu.row_stride_bytes, local_ff, s0 + r};
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
                                      D, lw.w_sh_gu.rows, lw.w_sh_gu.row_stride_bytes, local_ff, moe_sense - reps_of(7u) + 1u + r};
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
                    } else {
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_rtr_log, 33u * 256u, (M + 7u) / 8u, 1, 256, 1, 1, &rlka, sizeof(rlka), true));
                    }
                    // Runs concurrently with router logits (both read x_norm / x_q8 from p2p_ar_rms!)
                    enq_gemm_q4(g, lw.w_sh_gu, S[g].x_q8_va, S[g].x_ds_va, S[g].sh_gu_va, false);
                }
                st_mark(kStRouter, static_cast<int>(g));

                struct RTopKA { uint64_t lin, tids, tw; uint32_t M; };
                RTopKA rtka{S[g].router_log_va, S[g].top_ids_va, S[g].top_w_va, M};
                VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_rtr_top8, M * 256u, 1, 1, 256, 1, 1, &rtka, sizeof(rtka), true));
                // Runs concurrently with router top8!
                struct SwiKA { uint64_t gu, sc, oq, od; uint32_t M, lff; };
                SwiKA ska{S[g].sh_gu_va, S[g].sh_gate_sc_va, S[g].sh_q8_va, S[g].sh_ds_va, M, local_ff};
                VJ_CHECK(rt.enqueue_kernel(g, *K[g].swiglu_q8, M * 256u, 1, 1, 256, 1, 1, &ska, sizeof(ska), false));
                st_mark(kStTop8, static_cast<int>(g));
                }

                if (fuse_back) {
                    const bool do_ep_moe = (!opt.ab_no_prepush && M == 1u && !ar_ts_on && !opt.ar_wire_int8);
                    const uint64_t ptx_moe = do_ep_moe ? (pay_base + (1u - g) * kfd::kArGpuBytes + (ar_epoch & 1u) * kfd::kArSlotBytes) : 0ULL;
                    if (!asym_experts && ab_b2_back && local_ff == 512u && D == 3072u && K[g].moe_gu_16r && K[g].moe_mid_f32_to_q8 && K[g].moe_down_d8_sh) {
                        struct Gu8KA  { uint64_t wgu, xq, xds, tids, mf32; uint32_t lff; };
                        struct QKA    { uint64_t mf32, mq, md; uint32_t nblk; };
                        struct DnShKA { uint64_t wd, wds, mq, md, tids, tw, shwq, shwdm, shq, shds, out; uint32_t D, sh_st, dn_wst, dn_sst; uint64_t ptx; };
                        const Gu8KA  gk{lw.w_gu_iq2s_va, S[g].x_q8_va, S[g].x_ds_va, S[g].top_ids_va, S[g].moe_mid_f32_va, local_ff};
                        const QKA    qk{S[g].moe_mid_f32_va, S[g].moe_q8_va, S[g].moe_d_va, (8u * local_ff) / 32u};
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
                        const bool use_s224 = (opt.moe_back_wgs == 224u && local_ff == 512u && D == 3072u &&
                                               lw.w_sh_down.rows == 3072u && K[g].moe_m1_back_s224 != nullptr);
                        for (uint32_t r = 0; r < reps_of(8u); ++r) {
                            const BackKA bka{lw.w_gu_iq2s_va, S[g].x_q8_va, S[g].x_ds_va, S[g].top_ids_va, S[g].top_w_va,
                                             S[g].moe_q8_va, S[g].moe_d_va, lw.w_sh_down.qs_va, lw.w_sh_down.dm_va,
                                             S[g].sh_q8_va, S[g].sh_ds_va, S[g].sh_down_va, lw.w_dn_d8_w_va, lw.w_dn_d8_s_va, S[g].partial_va, S[g].moe_bq_va,
                                             local_ff, D, lw.w_sh_down.rows, lw.w_sh_down.row_stride_bytes,
                                             moe_back_sense - reps_of(8u) + 1u + r,
                                             lw.dn_d8_w_stride, lw.dn_d8_s_stride, 0u, ptx_moe};
                            VJ_CHECK(rt.enqueue_kernel(g, *(asym_experts ? K[g].moe_m1_back_asym : (use_s224 ? K[g].moe_m1_back_s224 : K[g].moe_m1_back)), opt.moe_back_wgs * 256u, 1, 1, 256, 1, 1, &bka, sizeof(bka), true));
                        }
                    }
                } else if (!asym_experts && (is_prefill_moe || (opt.gu_grp_verify && M >= 8u)) && lw.w_gu_iq2s_va != 0u) {
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

                    const bool use_pf_down_d8 = (use_pf_dot8_moe && use_pf_down_tiled && K[g].pf_down_d8 != nullptr && local_ff == 512u);
                    struct GuGrpKA { uint64_t wgu, xq, xds, off, cnt, sms, mq, md; uint32_t lff; };
                    GuGrpKA ggka{lw.w_gu_iq2s_va, S[g].x_q8_va, S[g].x_ds_va,
                                 gu_off_va, gu_cnt_va, gu_sms_va,
                                 S[g].moe_q8_va, S[g].moe_d_va, local_ff};
                    if (use_pf_dot8_moe && K[g].pf_gu_d8 && local_ff % 64u == 0u) {
                        struct GuTKA { uint64_t wgu, xq4, xds4, off, cnt, sms, mq, md, mq4, mds4, queue; uint32_t lff, D; };
                        const GuTKA gt{lw.w_gu_iq2s_va, S[g].pf_xq4_va, S[g].pf_xds4_va, gu_off_va, gu_cnt_va, gu_sms_va,
                                       S[g].moe_q8_va, S[g].moe_d_va,
                                       use_pf_down_d8 ? S[g].moe_mid_q4_va : 0ULL,
                                       use_pf_down_d8 ? S[g].moe_mid_ds4_va : 0ULL,
                                       S[g].pf_gu_queue_va, local_ff, D};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_gu_d8, 240u * 256u, 1, 1, 256, 1, 1, &gt, sizeof(gt), true));
                    } else if (K[g].pf_gu && S[g].pf_gu_queue_va != 0u && local_ff % 64u == 0u) {
                        struct GuTKA { uint64_t wgu, xq, xds, off, cnt, sms, mq, md, queue; uint32_t lff, D; };
                        const GuTKA gt{lw.w_gu_iq2s_va, S[g].x_q8_va, S[g].x_ds_va, gu_off_va, gu_cnt_va, gu_sms_va,
                                       S[g].moe_q8_va, S[g].moe_d_va, S[g].pf_gu_queue_va, local_ff, D};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_gu, 240u * 256u, 1, 1, 256, 1, 1, &gt, sizeof(gt), true));
                    } else
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_gu_iq2s_grp, (local_ff / 32u) * 256u, 256u, 1, 256, 1, 1, &ggka, sizeof(ggka), true));
                    st_mark(kStGateUp, static_cast<int>(g));

                    if (use_pf_down_d8) {
                        struct DnD8KA { uint64_t wq, ws, mq4, mds4, tw, off, cnt, sms, part, queue; uint32_t D, wst, sst; };
                        const DnD8KA dnt{lw.w_dn_d8_w_va, lw.w_dn_d8_s_va, S[g].moe_mid_q4_va, S[g].moe_mid_ds4_va, S[g].top_w_va,
                                         gu_off_va, gu_cnt_va, gu_sms_va, S[g].dgrp_buf_va, S[g].pf_gu_queue_va,
                                         D, lw.dn_d8_w_stride, lw.dn_d8_s_stride};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_down_d8, 240u * 256u, 1, 1, 256, 1, 1, &dnt, sizeof(dnt), true));
                        struct CbAllKA { uint64_t part, shd, out; uint32_t M, D; };
                        const CbAllKA cba{S[g].dgrp_buf_va, S[g].sh_down_va, S[g].partial_va, M, D};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].pf_down_comb, ((D + 255u) / 256u) * 256u, M, 1, 256, 1, 1, &cba, sizeof(cba), true));
                    } else if (use_pf_down_tiled) {
                        struct DnTKA { uint64_t wq, ws, mq, md, tw, off, cnt, sms, part, queue; uint32_t D, wst, sst; };
                        const DnTKA dnt{lw.w_dn_d8_w_va, lw.w_dn_d8_s_va, S[g].moe_q8_va, S[g].moe_d_va, S[g].top_w_va,
                                        gu_off_va, gu_cnt_va, gu_sms_va, S[g].dgrp_buf_va, S[g].pf_gu_queue_va,
                                        D, lw.dn_d8_w_stride, lw.dn_d8_s_stride};
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
                    } else {
                        struct DnKA { uint64_t wq, ws, mq, md, tids, tw, shd, out; uint32_t M, D, wst, sst; };
                        DnKA dka{lw.w_dn_d8_w_va, lw.w_dn_d8_s_va, S[g].moe_q8_va, S[g].moe_d_va, S[g].top_ids_va, S[g].top_w_va, S[g].sh_down_va, S[g].partial_va, M, D,
                                 lw.dn_d8_w_stride, lw.dn_d8_s_stride};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_down_d8, ((D + 15u) / 16u) * 256u, M, 1, 256, 1, 1, &dka, sizeof(dka), true));
                    }
                    st_mark(kStDown, static_cast<int>(g));
                } else {
                    if (lw.w_gu_d8_w_va != 0u && K[g].moe_gu_d8) {
                        const kfd::LoadedKernel& gu_k = (lw.gu_d8_w_stride == 2304u && K[g].moe_gu_q3)
                            ? *K[g].moe_gu_q3
                            : ((M >= 2u && K[g].moe_gu_d8_cached) ? *K[g].moe_gu_d8_cached : *K[g].moe_gu_d8);
                        struct GuD8KA { uint64_t wgu, wsc, xq, xd, tids, mq, md; uint32_t M, lff, wst, sst; };
                        const GuD8KA gd8{lw.w_gu_d8_w_va, lw.w_gu_d8_s_va, S[g].x_q8_va, S[g].x_ds_va, S[g].top_ids_va,
                                         S[g].moe_q8_va, S[g].moe_d_va, M, local_ff, lw.gu_d8_w_stride, lw.gu_d8_s_stride};
                        VJ_CHECK(rt.enqueue_kernel(g, gu_k, 8u * (local_ff / 32u) * 256u, M, 1, 256, 1, 1, &gd8, sizeof(gd8), true));
                    } else if (lru_slots > 0u && (std::getenv("VEGAJET_MOE_LRU") != nullptr && std::atoi(std::getenv("VEGAJET_MOE_LRU")) > 0) && M >= 2u && M <= 16u && K[g].moe_gu_lru) {
                        struct GuLruKA { uint64_t wgu, cw, cs, e2s, xq, xd, tids, mq, md; uint32_t M, lff, wst, sst; };
                        const GuLruKA lka{lw.w_gu_iq2s_va, lru_w_va[g][il], lru_s_va[g][il], lru_e2s_va[g][il],
                                          S[g].x_q8_va, S[g].x_ds_va, S[g].top_ids_va, S[g].moe_q8_va, S[g].moe_d_va,
                                          M, local_ff, lru_lay[g].w_row_stride_bytes, lru_lay[g].s_row_stride_bytes};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_gu_lru, 8u * (local_ff / 32u) * 256u, M, 1, 256, 1, 1, &lka, sizeof(lka), true));
                        if (g == 0u && M >= 4u) {
                            struct CpU32KA { uint64_t dst, src, n; };
                            const CpU32KA cka{lru_top_va + static_cast<uint64_t>(il) * 128ULL * 4ULL, S[0].top_ids_va, static_cast<uint64_t>(M) * 8ULL};
                            VJ_CHECK(rt.enqueue_kernel(0, *K[0].memcpy_u32, 256u, 1, 1, 256, 1, 1, &cka, sizeof(cka), false));
                        }
                    } else {
                        struct GuKA { uint64_t wgu, xq, xd, tids, mq, md; uint32_t M, lff; };
                        GuKA gka{lw.w_gu_iq2s_va, S[g].x_q8_va, S[g].x_ds_va, S[g].top_ids_va, S[g].moe_q8_va, S[g].moe_d_va, M, local_ff};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_gu_iq2s, 8u * (local_ff / 32u) * 256u, M, 1, 256, 1, 1, &gka, sizeof(gka), true));
                    }
                    // Runs concurrently with routed Gate+Up!
                    enq_gemm_q4(g, lw.w_sh_down, S[g].sh_q8_va, S[g].sh_ds_va, S[g].sh_down_va, false);

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
                    } else if (asym_experts) {
                        struct AsymDnKA {
                            uint64_t wq, ws, mq, md, tids, tw, shd, out;
                            uint32_t M, D, wst, sst, lff, pad;
                            uint64_t ptx;
                        };
                        static_assert(sizeof(AsymDnKA) == 96u, "asymmetric Down kernarg ABI");
                        const bool prepush = !opt.ab_no_prepush && M <= 16u && !ar_ts_on && !opt.ar_wire_int8;
                        const uint64_t ptx = prepush ? pay_base + (1u - g) * kfd::kArGpuBytes + (ar_epoch & 1u) * kfd::kArSlotBytes : 0u;
                        const AsymDnKA dka{lw.w_dn_d8_w_va, lw.w_dn_d8_s_va, S[g].moe_q8_va, S[g].moe_d_va,
                            S[g].top_ids_va, S[g].top_w_va, S[g].sh_down_va, S[g].partial_va,
                            M, D, lw.dn_d8_w_stride, lw.dn_d8_s_stride, local_ff, 0u, ptx};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_down_asym, ((D + 15u) / 16u) * 256u, M, 1,
                                                  256, 1, 1, &dka, sizeof(dka), true));
                    } else if (!opt.ab_no_prepush && M <= 16u && !ar_ts_on && !opt.ar_wire_int8 && K[g].moe_down_d8_pp != nullptr) {
                        const uint64_t ptx_dn = pay_base + (1u - g) * kfd::kArGpuBytes + (ar_epoch & 1u) * kfd::kArSlotBytes;
                        struct DnPpKA { uint64_t wq, ws, mq, md, tids, tw, shd, out; uint32_t M, D, wst, sst; uint64_t ptx; };
                        DnPpKA dka{lw.w_dn_d8_w_va, lw.w_dn_d8_s_va, S[g].moe_q8_va, S[g].moe_d_va, S[g].top_ids_va, S[g].top_w_va, S[g].sh_down_va, S[g].partial_va, M, D,
                                   lw.dn_d8_w_stride, lw.dn_d8_s_stride, ptx_dn};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_down_d8_pp, ((D + 15u) / 16u) * 256u, M, 1, 256, 1, 1, &dka, sizeof(dka), true));
                    } else {
                        struct DnKA { uint64_t wq, ws, mq, md, tids, tw, shd, out; uint32_t M, D, wst, sst; };
                        DnKA dka{lw.w_dn_d8_w_va, lw.w_dn_d8_s_va, S[g].moe_q8_va, S[g].moe_d_va, S[g].top_ids_va, S[g].top_w_va, S[g].sh_down_va, S[g].partial_va, M, D,
                                 lw.dn_d8_w_stride, lw.dn_d8_s_stride};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_down_d8, ((D + 15u) / 16u) * 256u, M, 1, 256, 1, 1, &dka, sizeof(dka), true));
                    }
                    st_mark(kStDown, static_cast<int>(g));
                }
            }
            const uint32_t back_disp = fuse_back ? 1u : ((is_prefill_moe || (opt.gu_grp_verify && M >= 8u)) ? (use_dgrp ? 5u : 4u) : (use_dgrp ? 5u : 3u));
            disp += (fuse_front ? 1u : 4u) + back_disp;

            const uint64_t next_w0 = (il + 1u < 48u) ? layers[0][il + 1u].attn_norm_va : out_norm_va[0];
            const uint64_t next_w1 = (il + 1u < 48u) ? layers[1][il + 1u].attn_norm_va : out_norm_va[1];
            if (golden_cap_on && il == 0u) {
                gcap_sync();
                gcap_dump("moe_router_logits", S[0].router_log_va, static_cast<uint64_t>(M) * 256ULL * 4ULL);
                gcap_dump("moe_top_ids", S[0].top_ids_va, static_cast<uint64_t>(M) * 8ULL * 4ULL);
                gcap_dump("moe_top_w", S[0].top_w_va, static_cast<uint64_t>(M) * 8ULL * 4ULL);
                gcap_dump("moe_partial_out", S[0].partial_va, static_cast<uint64_t>(M) * D * 4ULL);
            }
            // Both Down paths publish the same [M,D] FP16 payload, independent of the local intermediate width.
            const bool ep_pushed_moe_dn = (!opt.ab_no_prepush && !fuse_back && (asym_experts || !(is_prefill_moe || (opt.gu_grp_verify && M >= 8u))) && !use_dgrp &&
                                           M <= 16u && !ar_ts_on && !opt.ar_wire_int8 && (asym_experts || K[0].moe_down_d8_pp != nullptr));
            const bool ep_pushed_moe = (!opt.ab_no_prepush && fuse_back && M == 1u && !ar_ts_on && !opt.ar_wire_int8) || ep_pushed_moe_dn;
            enq_p2p_ar_rms(il * 2u + 1u, next_w0, next_w1, (il + 1u < 48u || compute_logits) ? 1u : 0u, ep_pushed_moe);
            st_mark(kStAr, -1);
            if (capture_dflash_taps && df.loaded && df.tap_slot[il] >= 0) {
                struct TapKA { uint64_t src, dst; uint32_t M, D, n_taps, tidx; };
                const TapKA tka{S[df_gpu].h_va, df.taps_va, M, D, static_cast<uint32_t>(df.tap_layers_0idx.size()),
                                static_cast<uint32_t>(df.tap_slot[il])};
                VJ_CHECK(rt.enqueue_kernel(df_gpu, *K[df_gpu].df_tap_copy, ((D + 255u) / 256u) * 256u, M, 1, 256, 1, 1, &tka, sizeof(tka), true));
            }
            if (tap_on && M == 1u) enq_tap(il + 1u);
            if (il == 0u) { rt.ring_doorbell(0); rt.ring_doorbell(1); }
        }
        if ((!use_ring && M == 1u && opt.gdn_splitv && K[0].gdn_conv_recur_sv != nullptr) ||
            (!use_ring && M >= 32u && K[0].pf_gdn_conv != nullptr)) {
            conv_head = (conv_head + 1u) % kGdnRingSlots;
        }

        const bool last_tok_only_vocab = (compute_logits && !use_ring && M > 1u);
        const uint32_t voc_m = last_tok_only_vocab ? 1u : M;
        if (compute_logits) {
            const uint64_t xq_off = last_tok_only_vocab ? (static_cast<uint64_t>(M - 1u) * D) : 0ULL;
            const uint64_t xd_off = last_tok_only_vocab ? (static_cast<uint64_t>(M - 1u) * (D / 32u) * 8ULL) : 0ULL;
            const bool use_b16 = (voc_m >= 13u && voc_m <= 16u && K[0].vocab_argmax_b16 != nullptr);
            const bool use_b12 = (!use_b16 && voc_m >= 7u && voc_m <= 12u && K[0].vocab_argmax_b12 != nullptr);
            const bool use_b6  = (!use_b16 && !use_b12 && voc_m >= 2u && D <= 3072u && K[0].vocab_argmax_b6 != nullptr);
            for (size_t g = 0; g < 2; ++g) {
                struct VocKA { uint64_t wq, wd, xq, xd, bv, bi; uint32_t lr, cols, ro, stride, M; };
                VocKA vka{out_w[g].qs_va, out_w[g].ds_va, S[g].x_q8_va + xq_off, S[g].x_ds_va + xd_off,
                          S[g].argmax_val_va, S[g].argmax_id_va,
                          voc_rows[g], D, voc_off[g], out_w[g].row_stride_bytes, voc_m};
                if (use_b16) {
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].vocab_argmax_b16, kVocabBlocks * 256u, 1, 1, 256, 1, 1, &vka, sizeof(vka), true));
                } else if (use_b12) {
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].vocab_argmax_b12, kVocabBlocks * 256u, 1, 1, 256, 1, 1, &vka, sizeof(vka), true));
                } else if (use_b6) {
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].vocab_argmax_b6, kVocabBlocks * 256u, (voc_m + 5u) / 6u, 1, 256, 1, 1, &vka, sizeof(vka), true));
                } else {
                    VJ_CHECK(rt.enqueue_kernel(g, *((voc_m == 1u && K[g].vocab_m1 && std::getenv("VEGAJET_VOCAB_MUBUF") && std::strcmp(std::getenv("VEGAJET_VOCAB_MUBUF"), "1") == 0) ? K[g].vocab_m1 : ((voc_m == 1u && K[g].vocab_ilp != nullptr) ? K[g].vocab_ilp : K[g].vocab_argmax)), kVocabBlocks * 256u, voc_m, 1, 256, 1, 1, &vka, sizeof(vka), true));
                }
            }
            disp += 1u;
        }
        st_mark(kStVocab, -1);
        last_dispatches = disp;

        VJ_CHECK(rt.submit_batch(0) && rt.submit_batch(1));
        VJ_CHECK(rt.wait_idle(0)    && rt.wait_idle(1));
        // v1.2 C1: both queues drained -> a recorded barrier/mailbox fault invalidates this step's logits.
        if (device_fault_check("decode step")) fail_run(res.error);

        std::vector<int32_t> preds(M, -1);
        if (compute_logits) {
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
            for (uint32_t m = 0; m < voc_m; ++m) {
                float best_v = -1e30f;
                int32_t best_i = -1;
                for (uint32_t b = 0; b < kVocabBlocks; ++b) {
                    const uint32_t idx = m * kVocabBlocks + b;
                    if (io_arg_val_host0[idx] > best_v) { best_v = io_arg_val_host0[idx]; best_i = io_arg_id_host0[idx]; }
                    if (io_arg_val_host1[idx] > best_v) { best_v = io_arg_val_host1[idx]; best_i = io_arg_id_host1[idx]; }
                }
                // v1.2 C3: a block reduction that never replaced its -1 sentinel (all-NaN / empty
                // logits) or an id outside the vocabulary must not reach the embedding gather.
                if (!guard::token_id_in_range(best_i, n_vocab)) {
                    if (res.error.empty()) {
                        res.error = "argmax token id " + std::to_string(best_i) + " is outside the vocabulary [0, " +
                                    std::to_string(n_vocab) + "); refusing to feed it to the embedding lookup";
                        std::fprintf(stderr, "[ornith::gpu] BAD ARGMAX: %s\n", res.error.c_str());
                    }
                    fail_run(res.error);
                }
                preds[last_tok_only_vocab ? (M - 1u) : m] = best_i;
            }
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
                    for (size_t g = 0; g < 2; ++g) {
                        const uint32_t local_ff = ff_size[g];
                        const RepKA rka{layers[g][a.layer].w_gu_iq2s_va, lru_w_va[g][a.layer], lru_s_va[g][a.layer],
                                        a.expert, a.slot, local_ff, lru_lay[g].w_row_stride_bytes, lru_lay[g].s_row_stride_bytes};
                        VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_gu_repack, local_ff * 128u, 1, 1, 128, 1, 1, &rka, sizeof(rka), true));
                    }
                    lru_cache->commit(a);
                }
                VJ_CHECK(rt.submit_batch(0) && rt.submit_batch(1));
                VJ_CHECK(rt.wait_idle(0)    && rt.wait_idle(1));
                for (const auto& a : admits) {
                    for (size_t g = 0; g < 2; ++g) {
                        VJ_CHECK(rt.upload_to_vram(g, lru_e2s_va[g][a.layer], lru_e2s_host[a.layer].data(), 256u * 4u));
                    }
                }
            }
        }

        return preds;
    };

    // Quick stage micro-profiler (measures exact M=1 ms per component across 48 layers!)
    auto run_m1_profile = [&]() {
        const uint32_t local_ff = ff_size[0];
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
                    ? (S[0].spec_gdn_s_slot_va[0] + static_cast<uint64_t>(i) * 32ULL * 128ULL * 128ULL * 4ULL)
                    : S[0].gdn_ssm_ring_va[i];
                RecKA rka{sva, S[0].gdn_conv_va, S[0].gdn_qkv_gab_va, S[0].gdn_dec_va, S[0].gdn_bsig_va,
                          lg.ssm_norm_va, S[0].mid_q8_va, S[0].mid_ds_va, 1u, 8u, 32u, 6144u, 10304u, 6144u, 0u, 0u, eps};
                VJ_CHECK(rt.enqueue_kernel(0, *K[0].gdn_recur, 32u * 1024u, 1, 1, 1024, 1, 1, &rka, sizeof(rka), true));
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
                if (asym_experts && layers[0][i].w_gu_d8_w_va != 0u) {
                    const auto& lw = layers[0][i];
                    struct GuD8KA { uint64_t wq, ws, xq, xd, tids, mq, md; uint32_t M, lff, wst, sst; };
                    const GuD8KA ka{lw.w_gu_d8_w_va, lw.w_gu_d8_s_va, S[0].x_q8_va, S[0].x_ds_va,
                        S[0].top_ids_va, S[0].moe_q8_va, S[0].moe_d_va, 1u, local_ff, lw.gu_d8_w_stride, lw.gu_d8_s_stride};
                    const auto* gu = (lw.gu_d8_w_stride == 2304u && K[0].moe_gu_q3) ? K[0].moe_gu_q3 : K[0].moe_gu_d8;
                    VJ_CHECK(rt.enqueue_kernel(0, *gu, 8u * (local_ff / 32u) * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
                    continue;
                }
                struct GuKA { uint64_t wgu, xq, xd, tids, mq, md; uint32_t M, lff; };
                GuKA gka{layers[0][i].w_gu_iq2s_va, S[0].x_q8_va, S[0].x_ds_va, S[0].top_ids_va, S[0].moe_q8_va, S[0].moe_d_va, 1u, local_ff};
                VJ_CHECK(rt.enqueue_kernel(0, *K[0].moe_gu_iq2s, 8u * (local_ff / 32u) * 256u, 1, 1, 256, 1, 1, &gka, sizeof(gka), true));
            }
        }); });
        double t_moe_dn  = time_fn(8, [&]() { run_g0([&]() {
            for (int i = 0; i < 48; ++i) {
                if (asym_experts) {
                    struct AsymDnKA { uint64_t wq, ws, mq, md, tids, tw, shd, out; uint32_t M, D, wst, sst, lff, pad; uint64_t ptx; };
                    const auto& lw = layers[0][i];
                    const AsymDnKA dka{lw.w_dn_d8_w_va, lw.w_dn_d8_s_va, S[0].moe_q8_va, S[0].moe_d_va,
                        S[0].top_ids_va, S[0].top_w_va, S[0].sh_down_va, S[0].partial_va,
                        1u, D, lw.dn_d8_w_stride, lw.dn_d8_s_stride, local_ff, 0u, 0u};
                    VJ_CHECK(rt.enqueue_kernel(0, *K[0].moe_down_asym, ((D + 15u) / 16u) * 256u, 1, 1, 256, 1, 1, &dka, sizeof(dka), true));
                    continue;
                }
                struct DnKA { uint64_t wq, ws, mq, md, tids, tw, shd, out; uint32_t M, D, wst, sst; };
                DnKA dka{layers[0][i].w_dn_d8_w_va, layers[0][i].w_dn_d8_s_va, S[0].moe_q8_va, S[0].moe_d_va, S[0].top_ids_va, S[0].top_w_va, S[0].sh_down_va,
                         S[0].partial_va, 1u, D, layers[0][i].dn_d8_w_stride, layers[0][i].dn_d8_s_stride};
                VJ_CHECK(rt.enqueue_kernel(0, *K[0].moe_down_d8, ((D + 15u) / 16u) * 256u, 1, 1, 256, 1, 1, &dka, sizeof(dka), true));
            }
        }); });
        double t_lm      = time_fn(8, [&]() { run_g0([&]() {
            struct VocKA { uint64_t wq, wd, xq, xd, bv, bi; uint32_t lr, cols, ro, stride, M; };
            VocKA vka{out_w[0].qs_va, out_w[0].ds_va, S[0].x_q8_va, S[0].x_ds_va,
                      S[0].argmax_val_va, S[0].argmax_id_va, voc_rows[0], D, 0u, out_w[0].row_stride_bytes, 1u};
            VJ_CHECK(rt.enqueue_kernel(0, *((K[0].vocab_m1 && std::getenv("VEGAJET_VOCAB_MUBUF") && std::strcmp(std::getenv("VEGAJET_VOCAB_MUBUF"), "1") == 0) ? K[0].vocab_m1 : (K[0].vocab_ilp != nullptr ? K[0].vocab_ilp : K[0].vocab_argmax)), kVocabBlocks * 256u, 1, 1, 256, 1, 1, &vka, sizeof(vka), true));
        }); });
        double t_ar = time_fn(8, [&]() {
            for (int i = 0; i < 96; ++i) {
                const uint32_t ep = ar_epoch++;
                const uint32_t pp = ep & 1u;
                for (size_t g = 0; g < 2; ++g) {
                    const size_t peer = 1u - g;
                    struct ArKA {
                        uint64_t xp, ptx, lrx, pfl, lfl, bar, h, w, xn, xq, xd;
                        uint32_t M, D, ep, dn;
                        float eps;
                    };
                    ArKA aka{
                        S[g].partial_va,
                        pay_base + peer * kfd::kArGpuBytes + pp * kfd::kArSlotBytes,
                        pay_base + g    * kfd::kArGpuBytes + pp * kfd::kArSlotBytes,
                        flg_base + peer * 4096ULL + (i & 63u) * 64ULL,
                        flg_base + g    * 4096ULL + (i & 63u) * 64ULL,
                        S[g].ar_bar_va, S[g].h_va, layers[g][0].attn_norm_va,
                        S[g].x_norm_va, S[g].x_q8_va, S[g].x_ds_va, 1u, D, ep, 1u, eps
                    };
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].p2p_ar_rms, 256u, 1, 1, 256, 1, 1, &aka, sizeof(aka), true));
                }
            }
            VJ_CHECK(rt.submit_batch(0) && rt.submit_batch(1));
            VJ_CHECK(rt.wait_idle(0) && rt.wait_idle(1));
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
                             1u, 8u, 32u, 6144u, 10304u, 10240u, 10272u, 0u, 0u, eps};
                VJ_CHECK(rt.enqueue_kernel(0, *K[0].gdn_conv_l2, 49u * 128u, 1, 1, 128, 1, 1, &cka, sizeof(cka), true));
            }
        }); });
        std::printf("[PROFILE M=1 GPU0] gdn_conv_l2 x36 = %.2f ms (%.1f us each)\n", t_conv, t_conv * 1000.0 / 36.0);
        // Fixed cost of a barrier-serialised dispatch: 384 one-WG no-op-sized memsets on GPU 0, and the same count
        // submitted to both GPUs (one submit/wait pair), to separate launch latency from kernel work.
        struct NullKA { uint64_t dst; uint32_t val; uint32_t pad; uint64_t n; };
        const NullKA nka{S[0].gdn_sq_va, 0u, 0u, 1u};
        const double t_null1 = time_fn(8, [&]() { run_g0([&]() {
            for (int i = 0; i < 384; ++i) VJ_CHECK(rt.enqueue_kernel(0, *K[0].memset_u32, 256u, 1, 1, 256, 1, 1, &nka, sizeof(nka), true));
        }); });
        const double t_null2 = time_fn(8, [&]() {
            for (size_t g = 0; g < 2; ++g) {
                const NullKA k2{S[g].gdn_sq_va, 0u, 0u, 1u};
                for (int i = 0; i < 384; ++i) VJ_CHECK(rt.enqueue_kernel(g, *K[g].memset_u32, 256u, 1, 1, 256, 1, 1, &k2, sizeof(k2), true));
            }
            VJ_CHECK(rt.submit_batch(0) && rt.submit_batch(1));
            VJ_CHECK(rt.wait_idle(0) && rt.wait_idle(1));
        });
        const double t_empty = time_fn(8, [&]() { run_g0([&]() {
            VJ_CHECK(rt.enqueue_kernel(0, *K[0].memset_u32, 256u, 1, 1, 256, 1, 1, &nka, sizeof(nka), true));
        }); });
        std::printf("[PROFILE dispatch] 384 null dispatches: GPU0 %.3f ms (%.2f us each) | both GPUs %.3f ms | 1 dispatch submit+wait %.1f us\n",
                    t_null1, t_null1 * 1000.0 / 384.0, t_null2, t_empty * 1000.0);
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
            const double dn_d8_row_b = double((asym_experts ? dot8_iq3_asym_layout(1u, local_ff) : dot8_iq3_layout(1u, local_ff)).read_bytes_per_row());
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

    auto run_greedy_raw = [&](const std::vector<int32_t>& prompt_ids, int n_gen, double* out_tps, double* out_pp_tps = nullptr) -> std::vector<int32_t> {
        reset_state();
        auto tp0 = std::chrono::steady_clock::now();
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
        for (size_t g = 0; g < 2; ++g) {
            const SpecGdnKA rka{
                S[g].spec_gdn_s_slot_va[spec_gdn_active], S[g].spec_gdn_s_slot_va[spec_gdn_active],
                0ULL, S[g].spec_gdn_rec_kv_va, S[g].spec_gdn_rec_kv_va + 8ULL * 128ULL * 4ULL,
                S[g].spec_gdn_rec_dec_va, S[g].spec_gdn_rec_beta_va,
                0ULL, 0ULL, 0ULL, 0ULL,
                n_commit, 8u, 32u, 40u * 128u, 40u * 128u, 0u,
                32u * 128u * 128u, 16u * 40u * 128u, 16u * 32u, 0u, 0u
            };
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].spec_gdn_ver, 128u * 256u, 36u, 1, 256, 1, 1, &rka, sizeof(rka), true));
            VJ_CHECK(rt.submit_batch(g));
        }
        VJ_CHECK(rt.wait_idle(0) && rt.wait_idle(1));
    };

    // Speculative decode driver (the bench's `tg128 Suffix` path): draft a suffix-tree chain, verify the whole batch,
    // keep only the tokens that match the greedy argmax, so the emitted sequence is the raw greedy one by
    // construction. Optional `on_tok` sees every committed token (same order, same ids as the returned vector) and can
    // stop the run; `pos_cap` bounds the KV position (serving passes opt.max_ctx).
    using SuffixTokenCb = std::function<bool(int32_t tid, uint32_t n_comp, double prefill_ms, double decode_ms)>;
    auto run_greedy_suffix = [&](const std::vector<int32_t>& prompt_ids, int n_gen, double* out_tps,
                                 uint64_t* out_hits = nullptr, uint64_t* out_acc = nullptr, uint64_t* out_dr = nullptr,
                                 const SuffixTokenCb* on_tok = nullptr,
                                 uint32_t pos_cap = 0xFFFFFFFFu) -> std::vector<int32_t> {
        reset_state();
        spec::SuffixTree sfx;
        sfx.extend(prompt_ids);
        spec::MoeCostModel cm;
        cm.n_experts = 256; cm.experts_per_token = 8; cm.moe_byte_fraction = 0.25;
        spec::MoeKGovernor gov(cm);

        uint32_t pos = 0;
        int32_t last_pred = -1;
        const auto tp0 = std::chrono::steady_clock::now();
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
        const double prefill_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tp0).count();
        if (on_tok && !(*on_tok)(last_pred, 1u, prefill_ms, 0.0)) return gen;

        auto td0 = std::chrono::steady_clock::now();
        while (static_cast<int>(gen.size()) < n_gen && pos < pos_cap) {
            const uint32_t rem = static_cast<uint32_t>(n_gen - static_cast<int>(gen.size()));
            const uint32_t room = pos_cap - pos;
            std::vector<int32_t> qctx = ctx;
            qctx.push_back(gen.back());
            const uint32_t k_max = std::min<uint32_t>({opt.suffix_max_draft, kGdnRingSlots - 2u, (rem > 1u) ? (rem - 1u) : 0u,
                                                       (room > 1u) ? (room - 1u) : 0u});
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
            const auto notify = [&](int32_t tid) {
                if (!on_tok) return true;
                const double dec_ms =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td0).count();
                return (*on_tok)(tid, static_cast<uint32_t>(gen.size()), prefill_ms, dec_ms);
            };
            if (draft_toks.empty()) {
                auto preds = step_batch({gen.back()}, pos++, false, false, true);
                ctx.push_back(gen.back());
                gen.push_back(preds[0]);
                sfx.push(preds[0]);
                if (!notify(preds[0])) return gen;
            } else {
                ++hits;
                const uint32_t dk = static_cast<uint32_t>(draft_toks.size());
                std::vector<int32_t> vtoks;
                vtoks.reserve(1u + dk);
                vtoks.push_back(gen.back());
                for (int32_t t : draft_toks) vtoks.push_back(t);
                auto preds = step_batch(vtoks, pos, false, true, true);
                uint32_t acc = 0;
                while (acc < dk && preds[acc] == draft_toks[acc]) ++acc;
                dr_tot += dk; acc_tot += acc;
                gov.observe(dk, acc);
                commit_spec_gdn(acc, dk);

                ctx.push_back(gen.back());
                for (uint32_t j = 0; j < acc; ++j) {
                    gen.push_back(draft_toks[j]);
                    ctx.push_back(draft_toks[j]);
                    sfx.push(draft_toks[j]);
                    if (!notify(draft_toks[j])) return gen;
                }
                if (static_cast<int>(gen.size()) < n_gen) {
                    gen.push_back(preds[acc]);
                    sfx.push(preds[acc]);
                    if (!notify(preds[acc])) return gen;
                }
                const uint32_t n_commit = acc + 1u;
                ring_head = (ring_head + n_commit) % kGdnRingSlots;
                conv_head = (conv_head + n_commit) % kGdnRingSlots;
                pos += n_commit;
            }
        }
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - td0).count();
        if (out_tps && n_gen > 1) *out_tps = (n_gen - 1) / std::max(sec, 1e-9);
        if (out_hits) *out_hits = hits;
        if (out_acc)  *out_acc  = acc_tot;
        if (out_dr)   *out_dr   = dr_tot;
        return gen;
    };

    // Built-in blk.48 nextn MTP single-token step (1 layer + optional LM head -> ~0.7 ms/draft!)
    auto step_mtp = [&](int32_t tok_id, uint32_t pos, uint64_t h_in0, uint64_t h_in1,
                        bool compute_logits, bool do_submit, uint32_t tok_slot) -> int32_t {
        require_token(tok_id, "MTP embedding");
        require_kv_span(pos, 1u);
        if (tok_slot >= 1024u) fail_run("MTP token staging capacity exceeded");
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
        auto enq_ar_mtp = [&](uint32_t slot, uint64_t w_rms0, uint64_t w_rms1, uint32_t do_norm) {
            const uint32_t ep = ar_epoch++;
            const uint32_t pp = ep & 1u;
            for (size_t g = 0; g < 2; ++g) {
                const size_t peer = 1u - g;
                struct ArKA {
                    uint64_t xp, ptx, lrx, pfl, lfl, bar, h, w, xn, xq, xd;
                    uint32_t M, D, ep, dn;
                    float eps;
                };
                ArKA aka{
                    S[g].partial_va,
                    pay_base + peer * kfd::kArGpuBytes + pp * kfd::kArSlotBytes,
                    pay_base + g    * kfd::kArGpuBytes + pp * kfd::kArSlotBytes,
                    flg_base + peer * 4096ULL + (slot & 63u) * 64ULL,
                    flg_base + g    * 4096ULL + (slot & 63u) * 64ULL,
                    S[g].ar_bar_va, S[g].mtp_h_cur_va, (g == 0) ? w_rms0 : w_rms1,
                    S[g].x_norm_va, S[g].x_q8_va, S[g].x_ds_va, 1u, D, ep, do_norm, eps
                };
                VJ_CHECK(rt.enqueue_kernel(g, *K[g].p2p_ar_rms, 256u, 1, 1, 256, 1, 1, &aka, sizeof(aka), true));
            }
        };

        for (size_t g = 0; g < 2; ++g) {
            const auto& ml = mtp[g];
            struct EhCatKA { uint64_t eqs, eds, toks, hin, wen, whn, cq, cds; uint32_t M, D; float eps; };
            EhCatKA ecka{emb_w[g].qs_va, emb_w[g].ds_va, io_toks_va + tok_slot * 4u,
                         (g == 0) ? h_in0 : h_in1, ml.enorm_va, ml.hnorm_va,
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
                       1u, pos, 16u, 1u, hp.rope_dims, rope_base, eps};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].attn_prep, 17u * 256u, 1, 1, 256, 1, 1, &pka, sizeof(pka), true));

            if (opt.spec_flash && K[g].spec_flash_part) {
                const uint32_t span = pos + 1u;
                const uint32_t chunk_len = (span <= 1920u) ? 32u : ((span <= 3840u) ? 64u : ((span <= 7680u) ? 128u : 256u));
                const uint32_t n_chunks = 1u + (span - 1u) / chunk_len;
                require_split_scratch(1u, 16u, n_chunks);
                struct SfPartKA { uint64_t qo, kc, vc, po, pml; uint32_t M, pos0, nq, nkv, nc, cl; };
                struct SfMrgKA  { uint64_t po, pml, go, oq, od; uint32_t M, nq, nc; };
                const SfPartKA pka2{S[g].q_buf_va, S[g].mtp_k_cache_va, S[g].mtp_v_cache_va,
                                    S[g].spec_fl_part_o_va, S[g].spec_fl_part_ml_va, 1u, pos, 16u, 1u, n_chunks, chunk_len};
                const SfMrgKA  mka2{S[g].spec_fl_part_o_va, S[g].spec_fl_part_ml_va, S[g].gate_buf_va,
                                    S[g].mid_q8_va, S[g].mid_ds_va, 1u, 16u, n_chunks};
                VJ_CHECK(rt.enqueue_kernel(g, *K[g].spec_flash_part,  n_chunks * 512u, 1, 1, 512, 1, 1, &pka2, sizeof(pka2), true));
                VJ_CHECK(rt.enqueue_kernel(g, *K[g].spec_flash_merge, 16u * 256u, 1, 1, 256, 1, 1, &mka2, sizeof(mka2), true));
            } else {
                struct FlKA { uint64_t qo, go, kc, vc, oq, od; uint32_t M, pos0, nq, nkv; };
                FlKA fka{S[g].q_buf_va, S[g].gate_buf_va, S[g].mtp_k_cache_va, S[g].mtp_v_cache_va,
                         S[g].mid_q8_va, S[g].mid_ds_va, 1u, pos, 16u, 1u};
                VJ_CHECK(rt.enqueue_kernel(g, *K[g].attn_flash, 16u * 256u, 1, 1, 256, 1, 1, &fka, sizeof(fka), true));
            }
            enq_q4_m1(g, ml.w_o, S[g].mid_q8_va, S[g].mid_ds_va, S[g].partial_va, true);
        }
        enq_ar_mtp(60u, mtp[0].post_attn_norm_va, mtp[1].post_attn_norm_va, 1u);

        for (size_t g = 0; g < 2; ++g) {
            const uint32_t local_ff = ff_size[g];
            const auto& ml = mtp[g];
            struct RLogKA { uint64_t wr, wsh, xn, lout, shg; uint32_t M, D; };
            RLogKA rlka{ml.w_router_f16_va, ml.w_sh_gate_va, S[g].x_norm_va, S[g].router_log_va, S[g].sh_gate_sc_va, 1u, D};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_rtr_log, 33u * 256u, 1, 1, 256, 1, 1, &rlka, sizeof(rlka), true));
            enq_q4_m1(g, ml.w_sh_gu, S[g].x_q8_va, S[g].x_ds_va, S[g].sh_gu_va, false);

            struct RTopKA { uint64_t lin, tids, tw; uint32_t M; };
            RTopKA rtka{S[g].router_log_va, S[g].top_ids_va, S[g].top_w_va, 1u};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_rtr_top8, 256u, 1, 1, 256, 1, 1, &rtka, sizeof(rtka), true));
            struct SwiKA { uint64_t gu, sc, oq, od; uint32_t M, lff; };
            SwiKA ska{S[g].sh_gu_va, S[g].sh_gate_sc_va, S[g].sh_q8_va, S[g].sh_ds_va, 1u, local_ff};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].swiglu_q8, 256u, 1, 1, 256, 1, 1, &ska, sizeof(ska), false));

            struct GuQ4KA { uint64_t wgq, wgdm, wuq, wudm, xq, xds, tids, mq, mds; uint32_t M, lff, stride; };
            GuQ4KA gka{ml.w_exp_gate.qs_va, ml.w_exp_gate.dm_va, ml.w_exp_up.qs_va, ml.w_exp_up.dm_va,
                       S[g].x_q8_va, S[g].x_ds_va, S[g].top_ids_va, S[g].moe_q8_va, S[g].mtp_moe_ds_va,
                       1u, local_ff, ml.w_exp_gate.row_stride_bytes};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_gu_q4a8, 8u * (local_ff / 32u) * 256u, 1, 1, 256, 1, 1, &gka, sizeof(gka), true));
            enq_q4_m1(g, ml.w_sh_down, S[g].sh_q8_va, S[g].sh_ds_va, S[g].sh_down_va, false);

            if (asym_experts) {
                struct DnQ4KA { uint64_t wdq, wddm, mq, mds, tids, tw, shd, out; uint32_t M, D, stride, lff; };
                const DnQ4KA dka{ml.w_exp_down.qs_va, ml.w_exp_down.dm_va, S[g].moe_q8_va, S[g].mtp_moe_ds_va,
                    S[g].top_ids_va, S[g].top_w_va, S[g].sh_down_va, S[g].partial_va,
                    1u, D, ml.w_exp_down.row_stride_bytes, local_ff};
                VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_down_asym_q4, ((D + 15u) / 16u) * 256u, 1, 1, 256, 1, 1, &dka, sizeof(dka), true));
            } else {
                struct DnQ4KA { uint64_t wdq, wddm, mq, mds, tids, tw, shd, out; uint32_t M, D, stride; };
                DnQ4KA dka{ml.w_exp_down.qs_va, ml.w_exp_down.dm_va, S[g].moe_q8_va, S[g].mtp_moe_ds_va,
                           S[g].top_ids_va, S[g].top_w_va, S[g].sh_down_va, S[g].partial_va,
                           1u, D, ml.w_exp_down.row_stride_bytes};
                VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_down_q4a8, ((D + 15u) / 16u) * 256u, 1, 1, 256, 1, 1, &dka, sizeof(dka), true));
            }
        }
        enq_ar_mtp(61u, mtp[0].shared_head_norm_va, mtp[1].shared_head_norm_va, compute_logits ? 1u : 0u);

        if (compute_logits) {
            for (size_t g = 0; g < 2; ++g) {
                const Q8WeightDev& vw = use_fr_spec ? fr_w[g] : out_w[g];
                const uint32_t lr = use_fr_spec ? fr_rows_per_gpu : voc_rows[g];
                const uint32_t ro = use_fr_spec ? (static_cast<uint32_t>(g) * lr) : voc_off[g];
                struct VocKA { uint64_t wq, wd, xq, xd, bv, bi; uint32_t lr, cols, ro, stride, M; };
                VocKA vka{vw.qs_va, vw.ds_va, S[g].x_q8_va, S[g].x_ds_va,
                          S[g].argmax_val_va, S[g].argmax_id_va,
                          lr, D, ro, vw.row_stride_bytes, 1u};
                VJ_CHECK(rt.enqueue_kernel(g, *((K[g].vocab_m1 && std::getenv("VEGAJET_VOCAB_MUBUF") && std::strcmp(std::getenv("VEGAJET_VOCAB_MUBUF"), "1") == 0) ? K[g].vocab_m1 : (K[g].vocab_ilp != nullptr ? K[g].vocab_ilp : K[g].vocab_argmax)), kVocabBlocks * 256u, 1, 1, 256, 1, 1, &vka, sizeof(vka), true));
            }
        }
        if (!do_submit) return -1;

        VJ_CHECK(rt.submit_batch(0) && rt.submit_batch(1));
        VJ_CHECK(rt.wait_idle(0)    && rt.wait_idle(1));
        if (device_fault_check("MTP step")) fail_run(res.error);
        if (!compute_logits) return -1;

        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        float best_v = -1e30f;
        int32_t best_i = -1;
        for (uint32_t b = 0; b < kVocabBlocks; ++b) {
            if (io_arg_val_host0[b] > best_v) { best_v = io_arg_val_host0[b]; best_i = io_arg_id_host0[b]; }
            if (io_arg_val_host1[b] > best_v) { best_v = io_arg_val_host1[b]; best_i = io_arg_id_host1[b]; }
        }
        if (use_fr_spec && best_i >= 0) {
            const uint32_t idx = static_cast<uint32_t>(best_i);
            const size_t g = (idx >= fr_rows_per_gpu) ? 1u : 0u;
            const uint32_t local_r = idx - static_cast<uint32_t>(g) * fr_rows_per_gpu;
            if (local_r >= fr_tok_map[g].size()) fail_run("MTP frequency-rank argmax is outside the token map");
            best_i = fr_tok_map[g][local_r];
        }
        require_token(best_i, "MTP argmax");
        return best_i;
    };

    auto gpu_copy_h_both = [&](uint64_t dst0, uint64_t src0, uint64_t dst1, uint64_t src1) {
        struct CpyKA { uint64_t dst, src, n; };
        CpyKA ka0{dst0, src0, D}, ka1{dst1, src1, D};
        const uint32_t gx = ((D + 255u) / 256u) * 256u;
        VJ_CHECK(rt.enqueue_kernel(0, *K[0].memcpy_u32, gx, 1, 1, 256, 1, 1, &ka0, sizeof(ka0), true));
        VJ_CHECK(rt.enqueue_kernel(1, *K[1].memcpy_u32, gx, 1, 1, 256, 1, 1, &ka1, sizeof(ka1), true));
    };

    // v1.2 levers, both default off: C7+S1 measured depth governor, S2 batched drafter catch-up.
    const bool v12_spec_gov = (std::getenv("VEGAJET_V12_SPEC_GOV") != nullptr && std::getenv("VEGAJET_V12_SPEC_GOV")[0] == '1');
    // Batch flag was validated against the expert layout before runtime initialization.

    // v1.2 S2: ingest n <= kMaxSpecBatch blk.48 MTP tokens in ONE M-row forward instead of n M=1 forwards.
    // Same kernels, buffers and order as step_mtp; only the grid gains the row dimension (attn_prep/spec_flash
    // are already row-parallel) and the residual inputs are staged into the mtp_h_cur rows eh_proj overwrites.
    // Positions must be consecutive (pos0 + j). Falls back to the per-token path when the M-row GEMVs are absent.
    const bool v12_batch_ok = (K[0].gemm_q4_m8 != nullptr) && (K[1].gemm_q4_m8 != nullptr);
    auto step_mtp_batch = [&](const std::vector<int32_t>& toks, uint32_t pos0,
                              const std::vector<uint64_t>& hs0, const std::vector<uint64_t>& hs1) -> bool {
        run_state.check();
        for (int32_t id : toks) require_token(id, "MTP batch embedding");
        const uint32_t n = static_cast<uint32_t>(toks.size());
        if (n == 0u) return true;
        require_kv_span(pos0, n);
        if (hs0.size() != n || hs1.size() != n) fail_run("MTP hidden-row count mismatch");
        if (n == 1u || n > kMaxSpecBatch || !v12_batch_ok) {
            for (uint32_t j = 0; j < n; ++j)
                (void) step_mtp(toks[j], pos0 + j, hs0[j], hs1[j], false, true, 0u);
            return true;
        }

        const uint32_t M = n;
        const float eps = static_cast<float>(hp.rms_eps);
        const float rope_base = static_cast<float>(hp.rope_base);
        const uint64_t row_bytes = static_cast<uint64_t>(D) * 4ULL;
        for (uint32_t j = 0; j < M; ++j) io_toks_host[j] = toks[j];
        __atomic_thread_fence(__ATOMIC_RELEASE);

        // Stage the per-row residual inputs; eh_cat reads them before eh_proj overwrites the same rows.
        for (uint32_t j = 0; j < M; ++j)
            gpu_copy_h_both(S[0].mtp_h_cur_va + static_cast<uint64_t>(j) * row_bytes, hs0[j],
                            S[1].mtp_h_cur_va + static_cast<uint64_t>(j) * row_bytes, hs1[j]);

        // M-row VJ_Q4_A8 GEMV: same kernel choice as enq_gemm_q4's M>1 branch.
        auto enq_q4_mN = [&](size_t g, const Q4A8WeightDev& w, uint64_t xq, uint64_t xds, uint64_t y, bool barrier) {
            const uint32_t gx8 = ((w.rows + 7u) / 8u) * 256u;
            if (M <= 8u && K[g].gemm_q4_m8) {
                struct GemmPpKA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, M, stride; uint64_t ptx; };
                GemmPpKA pka{w.qs_va, w.dm_va, xq, xds, y, w.rows, w.cols, M, w.row_stride_bytes, 0ULL};
                VJ_CHECK(rt.enqueue_kernel(g, *K[g].gemm_q4_m8, gx8, 1, 1, 256, 1, 1, &pka, sizeof(pka), barrier));
            } else if (M <= 12u && K[g].gemm_q4_m12) {
                struct GemmPpKA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, M, stride; uint64_t ptx; };
                GemmPpKA pka{w.qs_va, w.dm_va, xq, xds, y, w.rows, w.cols, M, w.row_stride_bytes, 0ULL};
                VJ_CHECK(rt.enqueue_kernel(g, *K[g].gemm_q4_m12, gx8, 1, 1, 256, 1, 1, &pka, sizeof(pka), barrier));
            } else if (M <= 16u && K[g].gemm_q4_m16) {
                struct GemmPpKA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, M, stride; uint64_t ptx; };
                GemmPpKA pka{w.qs_va, w.dm_va, xq, xds, y, w.rows, w.cols, M, w.row_stride_bytes, 0ULL};
                VJ_CHECK(rt.enqueue_kernel(g, *K[g].gemm_q4_m16, gx8, 1, 1, 256, 1, 1, &pka, sizeof(pka), barrier));
            } else {
                struct GemmKA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, M, stride; };
                GemmKA ka{w.qs_va, w.dm_va, xq, xds, y, w.rows, w.cols, M, w.row_stride_bytes};
                VJ_CHECK(rt.enqueue_kernel(g, *K[g].gemm_q4_tcc, gx8, (M + 15u) / 16u, 1, 256, 1, 1, &ka, sizeof(ka), barrier));
            }
        };

        for (size_t g = 0; g < 2; ++g) {
            const auto& ml = mtp[g];
            struct EhCatKA { uint64_t eqs, eds, toks, hin, wen, whn, cq, cds; uint32_t M, D; float eps; };
            EhCatKA ecka{emb_w[g].qs_va, emb_w[g].ds_va, io_toks_va, S[g].mtp_h_cur_va, ml.enorm_va, ml.hnorm_va,
                         S[g].mtp_cat_q8_va, S[g].mtp_cat_ds_va, M, D, eps};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].mtp_eh_cat, dispatch::eh_grid_x(M), 1, 1, 256, 1, 1, &ecka, sizeof(ecka), true));
            enq_q4_mN(g, ml.w_eh_proj, S[g].mtp_cat_q8_va, S[g].mtp_cat_ds_va, S[g].mtp_h_cur_va, true);

            struct RmsKA { uint64_t h, xa, w, xn, xq, xd; uint32_t M, D; float eps; };
            RmsKA rka{S[g].mtp_h_cur_va, 0ULL, ml.attn_norm_va, S[g].x_norm_va, S[g].x_q8_va, S[g].x_ds_va, M, D, eps};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].rms_quant, M * 256u, 1, 1, 256, 1, 1, &rka, sizeof(rka), true));

            enq_q4_mN(g, ml.w_attn_qkv, S[g].x_q8_va, S[g].x_ds_va, S[g].attn_qkv_va, true);
            struct PrepKA { uint64_t qkv, qw, kw, qo, go, kc, vc; uint32_t M, pos0, nq, nkv, rdims; float rbase, eps; };
            PrepKA pka{S[g].attn_qkv_va, ml.q_norm_va, ml.k_norm_va,
                       S[g].q_buf_va, S[g].gate_buf_va, S[g].mtp_k_cache_va, S[g].mtp_v_cache_va,
                       M, pos0, 16u, 1u, hp.rope_dims, rope_base, eps};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].attn_prep, 17u * 256u, M, 1, 256, 1, 1, &pka, sizeof(pka), true));

            if (opt.spec_flash && K[g].spec_flash_part) {
                const uint32_t span = pos0 + M;
                const uint32_t chunk_len = (span <= 1920u) ? 32u : ((span <= 3840u) ? 64u : ((span <= 7680u) ? 128u : 256u));
                const uint32_t n_chunks = 1u + (span - 1u) / chunk_len;
                require_split_scratch(M, 16u, n_chunks);
                struct SfPartKA { uint64_t qo, kc, vc, po, pml; uint32_t M, pos0, nq, nkv, nc, cl; };
                struct SfMrgKA  { uint64_t po, pml, go, oq, od; uint32_t M, nq, nc; };
                const SfPartKA pka2{S[g].q_buf_va, S[g].mtp_k_cache_va, S[g].mtp_v_cache_va,
                                    S[g].spec_fl_part_o_va, S[g].spec_fl_part_ml_va, M, pos0, 16u, 1u, n_chunks, chunk_len};
                const SfMrgKA  mka2{S[g].spec_fl_part_o_va, S[g].spec_fl_part_ml_va, S[g].gate_buf_va,
                                    S[g].mid_q8_va, S[g].mid_ds_va, M, 16u, n_chunks};
                VJ_CHECK(rt.enqueue_kernel(g, *K[g].spec_flash_part,  n_chunks * 512u, M, 1, 512, 1, 1, &pka2, sizeof(pka2), true));
                VJ_CHECK(rt.enqueue_kernel(g, *K[g].spec_flash_merge, 16u * 256u, M, 1, 256, 1, 1, &mka2, sizeof(mka2), true));
            } else {
                struct FlKA { uint64_t qo, go, kc, vc, oq, od; uint32_t M, pos0, nq, nkv; };
                FlKA fka{S[g].q_buf_va, S[g].gate_buf_va, S[g].mtp_k_cache_va, S[g].mtp_v_cache_va,
                         S[g].mid_q8_va, S[g].mid_ds_va, M, pos0, 16u, 1u};
                VJ_CHECK(rt.enqueue_kernel(g, *K[g].attn_flash, 16u * 256u, M, 1, 256, 1, 1, &fka, sizeof(fka), true));
            }
            enq_q4_mN(g, ml.w_o, S[g].mid_q8_va, S[g].mid_ds_va, S[g].partial_va, true);
        }

        // Post-attention AR + RMS over M rows (one slice, M <= 16), same slot as the M=1 path.
        const auto attn_collective = dispatch::next_collective(ar_epoch);
        for (size_t g = 0; g < 2; ++g) {
            const size_t peer = 1u - g;
            const uint32_t ep = attn_collective.epoch;
            const uint32_t pp = attn_collective.parity;
            struct ArKA {
                uint64_t xp, ptx, lrx, pfl, lfl, bar, h, w, xn, xq, xd;
                uint32_t M, D, ep, dn;
                float eps;
            };
            ArKA aka{S[g].partial_va,
                     pay_base + peer * kfd::kArGpuBytes + pp * kfd::kArSlotBytes,
                     pay_base + g    * kfd::kArGpuBytes + pp * kfd::kArSlotBytes,
                     flg_base + peer * 4096ULL + (60u & 63u) * 64ULL,
                     flg_base + g    * 4096ULL + (60u & 63u) * 64ULL,
                     S[g].ar_bar_va, S[g].mtp_h_cur_va, (g == 0) ? mtp[0].post_attn_norm_va : mtp[1].post_attn_norm_va,
                     S[g].x_norm_va, S[g].x_q8_va, S[g].x_ds_va, M, D, ep, 1u, eps};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].p2p_ar_rms, M * 256u, 1, 1, 256, 1, 1, &aka, sizeof(aka), true));
        }

        for (size_t g = 0; g < 2; ++g) {
            const auto& ml = mtp[g];
            struct RLogKA { uint64_t wr, wsh, xn, lout, shg; uint32_t M, D; };
            RLogKA rlka{ml.w_router_f16_va, ml.w_sh_gate_va, S[g].x_norm_va, S[g].router_log_va, S[g].sh_gate_sc_va, M, D};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_rtr_log, 33u * 256u, (M + 7u) / 8u, 1, 256, 1, 1, &rlka, sizeof(rlka), true));
            enq_q4_mN(g, ml.w_sh_gu, S[g].x_q8_va, S[g].x_ds_va, S[g].sh_gu_va, false);

            struct RTopKA { uint64_t lin, tids, tw; uint32_t M; };
            RTopKA rtka{S[g].router_log_va, S[g].top_ids_va, S[g].top_w_va, M};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_rtr_top8, M * 256u, 1, 1, 256, 1, 1, &rtka, sizeof(rtka), true));
            struct SwiKA { uint64_t gu, sc, oq, od; uint32_t M, lff; };
            SwiKA ska{S[g].sh_gu_va, S[g].sh_gate_sc_va, S[g].sh_q8_va, S[g].sh_ds_va, M, local_ff};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].swiglu_q8, M * 256u, 1, 1, 256, 1, 1, &ska, sizeof(ska), false));

            struct GuQ4KA { uint64_t wgq, wgdm, wuq, wudm, xq, xds, tids, mq, mds; uint32_t M, lff, stride; };
            GuQ4KA gka{ml.w_exp_gate.qs_va, ml.w_exp_gate.dm_va, ml.w_exp_up.qs_va, ml.w_exp_up.dm_va,
                       S[g].x_q8_va, S[g].x_ds_va, S[g].top_ids_va, S[g].moe_q8_va, S[g].mtp_moe_ds_va,
                       M, local_ff, ml.w_exp_gate.row_stride_bytes};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_gu_q4a8, 8u * (local_ff / 32u) * 256u, M, 1, 256, 1, 1, &gka, sizeof(gka), true));
            enq_q4_mN(g, ml.w_sh_down, S[g].sh_q8_va, S[g].sh_ds_va, S[g].sh_down_va, false);

            struct DnQ4KA { uint64_t wdq, wddm, mq, mds, tids, tw, shd, out; uint32_t M, D, stride; };
            DnQ4KA dka{ml.w_exp_down.qs_va, ml.w_exp_down.dm_va, S[g].moe_q8_va, S[g].mtp_moe_ds_va,
                       S[g].top_ids_va, S[g].top_w_va, S[g].sh_down_va, S[g].partial_va,
                       M, D, ml.w_exp_down.row_stride_bytes};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_down_q4a8, ((D + 15u) / 16u) * 256u, M, 1, 256, 1, 1, &dka, sizeof(dka), true));
        }

        // Post-MoE AR + shared-head norm over M rows; no LM head here (catch-up only ingests).
        const auto moe_collective = dispatch::next_collective(ar_epoch);
        for (size_t g = 0; g < 2; ++g) {
            const size_t peer = 1u - g;
            const uint32_t ep = moe_collective.epoch;
            const uint32_t pp = moe_collective.parity;
            struct ArKA {
                uint64_t xp, ptx, lrx, pfl, lfl, bar, h, w, xn, xq, xd;
                uint32_t M, D, ep, dn;
                float eps;
            };
            ArKA aka{S[g].partial_va,
                     pay_base + peer * kfd::kArGpuBytes + pp * kfd::kArSlotBytes,
                     pay_base + g    * kfd::kArGpuBytes + pp * kfd::kArSlotBytes,
                     flg_base + peer * 4096ULL + (61u & 63u) * 64ULL,
                     flg_base + g    * 4096ULL + (61u & 63u) * 64ULL,
                     S[g].ar_bar_va, S[g].mtp_h_cur_va, (g == 0) ? mtp[0].shared_head_norm_va : mtp[1].shared_head_norm_va,
                     S[g].x_norm_va, S[g].x_q8_va, S[g].x_ds_va, M, D, ep, 0u, eps};
            VJ_CHECK(rt.enqueue_kernel(g, *K[g].p2p_ar_rms, M * 256u, 1, 1, 256, 1, 1, &aka, sizeof(aka), true));
        }

        // All token/hidden staging is shared: complete consumption before any caller can reuse it.
        VJ_CHECK(rt.submit_batch(0) && rt.submit_batch(1));
        VJ_CHECK(rt.wait_idle(0)    && rt.wait_idle(1));
        return true;
    };

    auto run_greedy_mtp = [&](const std::vector<int32_t>& prompt_ids, int n_gen, uint32_t K_draft,
                              bool use_hybrid_suffix, double* out_tps,
                              uint64_t* out_acc = nullptr, uint64_t* out_dr = nullptr,
                              uint64_t* out_sfx_hits = nullptr) -> std::vector<int32_t> {
        reset_state();
        sfx::CascadeConfig ccfg;
        ccfg.min_match = 4u;
        ccfg.min_chain = (opt.zero_tax_catchup && K_draft > 0u) ? std::min<uint32_t>(4u, K_draft) : 1u;
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
                    // One M-row forward per <= kMaxSpecBatch consecutive prompt tokens.
                    for (size_t b0 = j0; b0 < j_end; b0 += kMaxSpecBatch) {
                        const size_t b_end = std::min<size_t>(j_end, b0 + kMaxSpecBatch);
                        std::vector<int32_t> bt;
                        std::vector<uint64_t> bh0, bh1;
                        bt.reserve(b_end - b0); bh0.reserve(b_end - b0); bh1.reserve(b_end - b0);
                        for (size_t j = b0; j < b_end; ++j) {
                            bt.push_back(chunk[j]);
                            bh0.push_back((j == 0) ? ((pos == 0) ? S[0].mtp_h_zero_va : S[0].mtp_h_pend_va)
                                                   : (S[0].h_va + (j - 1u) * h_stride));
                            bh1.push_back((j == 0) ? ((pos == 0) ? S[1].mtp_h_zero_va : S[1].mtp_h_pend_va)
                                                   : (S[1].h_va + (j - 1u) * h_stride));
                        }
                        (void) step_mtp_batch(bt, pos + static_cast<uint32_t>(b0), bh0, bh1);
                    }
                } else {
                    for (size_t j = j0; j < j_end; ++j) {
                        uint64_t h0 = (j == 0) ? ((pos == 0) ? S[0].mtp_h_zero_va : S[0].mtp_h_pend_va) : (S[0].h_va + (j - 1u) * h_stride);
                        uint64_t h1 = (j == 0) ? ((pos == 0) ? S[1].mtp_h_zero_va : S[1].mtp_h_pend_va) : (S[1].h_va + (j - 1u) * h_stride);
                        (void) step_mtp(chunk[j], pos + static_cast<uint32_t>(j), h0, h1, false, false, static_cast<uint32_t>(j - j0));
                    }
                }
                if (j_end == cnt) {
                    gpu_copy_h_both(S[0].mtp_h_pend_va, S[0].h_va + (cnt - 1u) * h_stride,
                                    S[1].mtp_h_pend_va, S[1].h_va + (cnt - 1u) * h_stride);
                }
                VJ_CHECK(rt.submit_batch(0) && rt.submit_batch(1));
                VJ_CHECK(rt.wait_idle(0)    && rt.wait_idle(1));
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
        uint32_t n_ver[17] = {}, n_cyc = 0, n_neural = 0;
        auto now_ms = []() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
        // v1.2 C7+S1: depth from the measured per-M verify time and the live accepted length.
        spec_policy::MeasuredDepthGovernor mtp_gov;
        uint32_t gov_depth_sum = 0u, gov_depth_n = 0u, gov_plain_n = 0u;
        auto mtp_ingest = [&](const std::vector<std::pair<int32_t, uint32_t>>& tok_pos,
                              const std::vector<uint64_t>& hs0, const std::vector<uint64_t>& hs1) {
            // Split into maximal runs of consecutive positions: one M-row forward per run (<= kMaxSpecBatch rows).
            size_t i = 0;
            while (i < tok_pos.size()) {
                size_t j = i + 1u;
                while (j < tok_pos.size() && tok_pos[j].second == tok_pos[j - 1u].second + 1u &&
                       (j - i) < kMaxSpecBatch) ++j;
                std::vector<int32_t> bt;
                std::vector<uint64_t> b0, b1;
                bt.reserve(j - i); b0.reserve(j - i); b1.reserve(j - i);
                for (size_t k = i; k < j; ++k) { bt.push_back(tok_pos[k].first); b0.push_back(hs0[k]); b1.push_back(hs1[k]); }
                (void) step_mtp_batch(bt, tok_pos[i].second, b0, b1);
                i = j;
            }
        };
        while (static_cast<int>(gen.size()) < n_gen) {
            const uint32_t rem = static_cast<uint32_t>(n_gen - static_cast<int>(gen.size()));
            uint32_t k_cur = std::min<uint32_t>({K_draft, kGdnRingSlots - 2u, (rem > 1u) ? (rem - 1u) : 0u});
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
            double t_neural = -1.0;  // start of the neural proposal loop (after any deferred catch-up replay)
            if (!used_suffix) {
                if (use_hybrid_suffix && opt.zero_tax_catchup && kv_sync.pending() > 0u) {
                    const auto pre = kv_sync.before_neural();
                    if (!pre.catchup.empty()) {
                        for (size_t c = 0; c < pre.catchup.size(); ++c) {
                            const auto& st = pre.catchup[c];
                            (void) step_mtp(st.token, st.pos, resolve_h_src(0, st.h), resolve_h_src(1, st.h),
                                            false, false, static_cast<uint32_t>(c));
                        }
                        VJ_CHECK(rt.submit_batch(0) && rt.submit_batch(1));
                        VJ_CHECK(rt.wait_idle(0)    && rt.wait_idle(1));
                    }
                }
                ++n_neural;
                t_neural = now_ms();
                int32_t d_in = gen.back();
                for (uint32_t j = 0; j < k_cur; ++j) {
                    uint64_t h0 = (j == 0) ? S[0].mtp_h_pend_va : S[0].mtp_h_cur_va;
                    uint64_t h1 = (j == 0) ? S[1].mtp_h_pend_va : S[1].mtp_h_cur_va;
                    int32_t dn = step_mtp(d_in, pos + j, h0, h1, true, true, 0u);
                    drafts.push_back(dn);
                    d_in = dn;
                }
            }

            const double tB = now_ms();
            t_draft += tB - tA;
            const uint32_t dk = static_cast<uint32_t>(drafts.size());
            if (v12_spec_gov && t_neural >= 0.0 && dk > 0u) mtp_gov.observe_draft(dk, tB - t_neural);
            std::vector<int32_t> vtoks;
            vtoks.reserve(1u + dk);
            vtoks.push_back(gen.back());
            for (int32_t t : drafts) vtoks.push_back(t);
            auto preds = step_batch(vtoks, pos, false, true, true);
            const double tC = now_ms();
            { const uint32_t mi = std::min<uint32_t>(dk + 1u, 16u); t_ver[mi] += tC - tB; ++n_ver[mi]; ++n_cyc; }

            uint32_t acc = 0;
            while (acc < dk && preds[acc] == drafts[acc]) ++acc;
            dr_tot += dk; acc_tot += acc;
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
                        gpu_copy_h_both(S[0].mtp_h_stash_va + static_cast<uint64_t>(sc.dst_slot) * h_stride, resolve_h_src(0, sc.src),
                                        S[1].mtp_h_stash_va + static_cast<uint64_t>(sc.dst_slot) * h_stride, resolve_h_src(1, sc.src));
                    }
                    if (v12_batch_catchup) {
                        std::vector<std::pair<int32_t, uint32_t>> tp;
                        std::vector<uint64_t> c0, c1;
                        for (const auto& st : post.catchup) {
                            tp.emplace_back(st.token, st.pos);
                            c0.push_back(resolve_h_src(0, st.h));
                            c1.push_back(resolve_h_src(1, st.h));
                        }
                        mtp_ingest(tp, c0, c1);
                    } else {
                        for (size_t c = 0; c < post.catchup.size(); ++c) {
                            const auto& st = post.catchup[c];
                            (void) step_mtp(st.token, st.pos, resolve_h_src(0, st.h), resolve_h_src(1, st.h),
                                            false, false, static_cast<uint32_t>(c));
                        }
                    }
                    gpu_copy_h_both(S[0].mtp_h_pend_va, S[0].h_va + acc * h_stride,
                                    S[1].mtp_h_pend_va, S[1].h_va + acc * h_stride);
                    VJ_CHECK(rt.submit_batch(0) && rt.submit_batch(1));
                    VJ_CHECK(rt.wait_idle(0)    && rt.wait_idle(1));
                } else {
                    if (used_suffix) {
                        (void) step_mtp(vtoks[0], pos, S[0].mtp_h_pend_va, S[1].mtp_h_pend_va, false, false, 0u);
                    }
                    if (v12_batch_catchup && acc > 0u) {
                        // Rows vtoks[1..acc] at pos+1..pos+acc; the vtoks[0] row (suffix ingest) stays a single step.
                        std::vector<std::pair<int32_t, uint32_t>> tp;
                        std::vector<uint64_t> c0, c1;
                        for (uint32_t j = 1u; j <= acc; ++j) {
                            tp.emplace_back(vtoks[j], pos + j);
                            c0.push_back(S[0].h_va + (j - 1u) * h_stride);
                            c1.push_back(S[1].h_va + (j - 1u) * h_stride);
                        }
                        mtp_ingest(tp, c0, c1);
                    } else {
                        for (uint32_t j = 1u; j <= acc; ++j) {
                            (void) step_mtp(vtoks[j], pos + j,
                                            S[0].h_va + (j - 1u) * h_stride, S[1].h_va + (j - 1u) * h_stride,
                                            false, false, j);
                        }
                    }
                    gpu_copy_h_both(S[0].mtp_h_pend_va, S[0].h_va + acc * h_stride,
                                    S[1].mtp_h_pend_va, S[1].h_va + acc * h_stride);
                    VJ_CHECK(rt.submit_batch(0) && rt.submit_batch(1));
                    VJ_CHECK(rt.wait_idle(0)    && rt.wait_idle(1));
                }
            }
            {
                const double tD = now_ms();
                t_catch += tD - tC;
                if (v12_spec_gov) mtp_gov.observe_catchup(n_commit, tD - tC);
            }
            pos += n_commit;
        }
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
        if (out_tps && n_gen > 1) *out_tps = (n_gen - 1) / std::max(sec, 1e-9);
        if (out_acc)      *out_acc      = acc_tot;
        if (out_dr)       *out_dr       = dr_tot;
        if (out_sfx_hits) *out_sfx_hits = sfx_hits;
        return gen;
    };

    // ---- Pillar 3: --dump-golden-layers <dir> (1 GDN L0 + 1 GQA L3 fixture for vegajet_layer_test) ----
    if (!opt.dump_golden_layers_dir.empty()) {
        const uint32_t local_ff = ff_size[0];
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
        if (device_fault_check("golden dump")) fail_run(res.error);
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
            if (run_raw_part && !run_spec_part && !run_prefill_part) {
                static std::vector<int32_t> trusted_raw128;
                auto raw128 = run_greedy_raw(p_rep_code, 129, &res.tg128_raw_tps);
                if (!dump_tokens(raw128)) return false;
                res.dispatches_per_tok = last_dispatches;
                if (trusted_raw128.empty()) trusted_raw128 = raw128;
                uint32_t eq128 = 0; int fd128 = -1;
                for (size_t i = 0; i < trusted_raw128.size() && i < raw128.size(); ++i) {
                    if (trusted_raw128[i] == raw128[i]) ++eq128; else if (fd128 == -1) fd128 = static_cast<int>(i);
                }
                std::printf("[BENCH tg128 raw M=1] %.2f tok/s (%.2f ms/tok) | disp/tok=%u | vs_base=%u/%zu (first_diff=%d)\n",
                            res.tg128_raw_tps, 1000.0 / std::max(res.tg128_raw_tps, 1e-6), last_dispatches,
                            eq128, trusted_raw128.size(), fd128);
                if (std::getenv("VEGAJET_RAW_TG512") && std::getenv("VEGAJET_RAW_TG512")[0] == '1') {
                    double tg512_tps = 0.0;
                    (void) run_greedy_raw(p_rep_code, 513, &tg512_tps);
                    std::printf("[BENCH tg512 raw M=1] %.2f tok/s (%.2f ms/tok) | disp/tok=%u\n",
                                tg512_tps, 1000.0 / std::max(tg512_tps, 1e-6), last_dispatches);
                }
                if (std::getenv("VEGAJET_SUPER_PROFILE") && std::getenv("VEGAJET_SUPER_PROFILE")[0] == '1') {
                    const kfd::LoadedKernel* k_clk0 = rt.get_kernel(0, "kfd_stamp_clock_u64");
                    const kfd::LoadedKernel* k_clk1 = rt.get_kernel(1, "kfd_stamp_clock_u64");
                    if (k_clk0 && k_clk1 && K[0].m1_gdn_super && K[0].m1_gqa_super && K[0].m1_moe_super) {
                        const char* swg_env = std::getenv("VEGAJET_SUPER_WGS");
                        const uint32_t swgs = (swg_env && std::atoi(swg_env) >= 192 && std::atoi(swg_env) <= 240)
                                            ? static_cast<uint32_t>(std::atoi(swg_env)) : 240u;
                        const char* q3_fast_env = std::getenv("VEGAJET_MOE_Q3_FAST");
                        const bool allow_q3_fast = !(q3_fast_env && q3_fast_env[0] == '0');
                        const uint64_t ts_va[2] = {S[0].splitk_part_va, S[1].splitk_part_va}; // scratch > 4KB
                        struct ClkKA { uint64_t dst; uint32_t idx; uint32_t pad; };
                        auto stamp = [&](size_t g, uint32_t idx) {
                            const ClkKA ck{ts_va[g], idx, 0u};
                            VJ_CHECK(rt.enqueue_kernel(g, *(g == 0 ? k_clk0 : k_clk1), 64u, 1, 1, 64, 1, 1, &ck, sizeof(ck), true));
                        };
                        reset_state();
                        // Zero sub-stage profiler accumulators in super_bar + 1024 (16 x u64 = 128 B).
                        {
                            uint64_t z16[16] = {};
                            VJ_CHECK(rt.upload_to_vram(0, S[0].super_bar_va + 1024u, z16, sizeof(z16)));
                            VJ_CHECK(rt.upload_to_vram(1, S[1].super_bar_va + 1024u, z16, sizeof(z16)));
                        }
                        // Warmup 1 token then stamp 1 full 48L super-kernel token on BOTH GPUs!
                        (void) step_batch({p_rep_code[0]}, 0u, false, false, true);
                        uint32_t gi = 0, ai = 0, pend_ep = 0u, tidx = 0u;
                        const uint32_t cin = conv_head, cout = (conv_head + 1u) % kGdnRingSlots;
                        stamp(0, tidx); stamp(1, tidx); ++tidx;
                        for (size_t g = 0; g < 2; ++g) {
                            struct EmbKA { uint64_t qs, ds, toks, h; uint32_t M, D; };
                            EmbKA eka{emb_w[g].qs_va, emb_w[g].ds_va, S[g].tok_ids_va, S[g].h_va, 1u, D};
                            VJ_CHECK(rt.enqueue_kernel(g, *K[g].embed_q8, 256u, 1, 1, 256, 1, 1, &eka, sizeof(eka), true));
                            struct RmsKA { uint64_t h, xa, w, xn, xq, xd; uint32_t M, D; float eps; };
                            RmsKA rka{S[g].h_va, 0ULL, layers[g][0].attn_norm_va, S[g].x_norm_va, S[g].x_q8_va, S[g].x_ds_va, 1u, D, static_cast<float>(hp.rms_eps)};
                            VJ_CHECK(rt.enqueue_kernel(g, *K[g].rms_quant, 256u, 1, 1, 256, 1, 1, &rka, sizeof(rka), true));
                        }
                        stamp(0, tidx); stamp(1, tidx); ++tidx; // 1: emb+rms
                        for (uint32_t il = 0; il < 48u; ++il) {
                            const uint32_t pep = pend_ep, aep = ar_epoch++, mep = ar_epoch++;
                            if (layers[0][il].is_gdn) {
                                const uint32_t cgi = gi++; ++super_sense; ++gdn_sense;
                                struct GdnSupKA { uint64_t desc, ov; uint32_t pep, oep, cin, cout, sh, ur, ss, gs; uint64_t fault; };
                                static_assert(sizeof(GdnSupKA) == 56u, "kernarg ABI drift");
                                const uint64_t s_layer_bytes = 32ULL * 128ULL * 128ULL * 4ULL;
                                const uint32_t s_rhead = opt.spec_gdn ? 0u : ring_head;
                                for (size_t g = 0; g < 2; ++g) {
                                    const uint64_t s_ov = opt.spec_gdn
                                        ? (S[g].spec_gdn_s_slot_va[spec_gdn_active] + static_cast<uint64_t>(cgi) * s_layer_bytes) : 0ULL;
                                    const GdnSupKA ka{S[g].gdn_super_desc_va + cgi * sizeof(ornith_super::GdnSuperDesc), s_ov, pep, aep, cin, cout, s_rhead, 0u, super_sense | 0x80000000u, gdn_sense, io_fault_va[g]};
                                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].m1_gdn_super, swgs * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
                                }
                            } else {
                                const uint32_t cai = ai++; ++super_sense;
                                struct GqaSupKA { uint64_t desc; uint32_t pep, oep, pos0, ss; uint64_t fault; };
                                static_assert(sizeof(GqaSupKA) == 32u, "kernarg ABI drift");
                                for (size_t g = 0; g < 2; ++g) {
                                    const GqaSupKA ka{S[g].gqa_super_desc_va + cai * sizeof(ornith_super::GqaSuperDesc), pep, aep, 1u, super_sense | 0x80000000u, io_fault_va[g]};
                                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].m1_gqa_super, swgs * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
                                }
                            }
                            stamp(0, tidx); stamp(1, tidx); ++tidx;
                            ++super_sense;
                            struct MoeSupKA { uint64_t desc; uint32_t aep, oep, ss; uint64_t fault; };
                            static_assert(sizeof(MoeSupKA) == 32u, "kernarg ABI drift");
                            for (size_t g = 0; g < 2; ++g) {
                                const MoeSupKA ka{S[g].moe_super_desc_va + il * sizeof(ornith_super::MoeSuperDesc), aep, mep, super_sense | 0x80000000u, io_fault_va[g]};
                                const bool use_q3_k = allow_q3_fast && (swgs == 240u) && K[g].m1_moe_super_q3 &&
                                                      (layers[g][il].w_gu_d8_w_va != 0ULL) &&
                                                      (layers[g][il].gu_d8_w_stride == 2304u);
                                const auto* mk = use_q3_k ? K[g].m1_moe_super_q3 : K[g].m1_moe_super;
                                VJ_CHECK(rt.enqueue_kernel(g, *mk, swgs * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
                            }
                            stamp(0, tidx); stamp(1, tidx); ++tidx;
                            pend_ep = mep;
                        }
                        // Final AR + LM head
                        {
                            const uint32_t slot = 95u, pp = pend_ep & 1u;
                            for (size_t g = 0; g < 2; ++g) {
                                const size_t peer = 1u - g;
                                struct ArKA { uint64_t xp, ptx, lrx, pfl, lfl, bar, h, w, xn, xq, xd; uint32_t M, D, ep, dn; float eps; };
                                const ArKA aka{S[g].partial_va, pay_base + peer * kfd::kArGpuBytes + pp * kfd::kArSlotBytes,
                                               pay_base + g * kfd::kArGpuBytes + pp * kfd::kArSlotBytes,
                                               flg_base + peer * 4096ULL + (slot & 63u) * 64ULL,
                                               flg_base + g * 4096ULL + (slot & 63u) * 64ULL,
                                               S[g].ar_bar_va, S[g].h_va, out_norm_va[g], S[g].x_norm_va, S[g].x_q8_va, S[g].x_ds_va,
                                               1u, D, pend_ep, 1u, static_cast<float>(hp.rms_eps)};
                                VJ_CHECK(rt.enqueue_kernel(g, *K[g].p2p_ar_prepush, 256u, 1, 1, 256, 1, 1, &aka, sizeof(aka), true));
                            }
                            stamp(0, tidx); stamp(1, tidx); ++tidx;
                            for (size_t g = 0; g < 2; ++g) {
                                struct VocKA { uint64_t wq, wd, xq, xd, bv, bi; uint32_t lr, cols, ro, stride, M; };
                                const VocKA vka{out_w[g].qs_va, out_w[g].ds_va, S[g].x_q8_va, S[g].x_ds_va, S[g].argmax_val_va, S[g].argmax_id_va,
                                                voc_rows[g], D, voc_off[g], out_w[g].row_stride_bytes, 1u};
                                VJ_CHECK(rt.enqueue_kernel(g, *((K[g].vocab_m1 && std::getenv("VEGAJET_VOCAB_MUBUF") && std::strcmp(std::getenv("VEGAJET_VOCAB_MUBUF"), "1") == 0) ? K[g].vocab_m1 : (K[g].vocab_ilp != nullptr ? K[g].vocab_ilp : K[g].vocab_argmax)), kVocabBlocks * 256u, 1, 1, 256, 1, 1, &vka, sizeof(vka), true));
                            }
                            stamp(0, tidx); stamp(1, tidx); ++tidx;
                        }
                        VJ_CHECK(rt.submit_batch(0) && rt.submit_batch(1));
                        VJ_CHECK(rt.wait_idle(0) && rt.wait_idle(1));
                        if (device_fault_check("super profile")) fail_run(res.error);
                        std::vector<uint64_t> t0(tidx), t1(tidx);
                        VJ_CHECK(rt.download_from_vram(0, t0.data(), ts_va[0], tidx * 8u));
                        VJ_CHECK(rt.download_from_vram(1, t1.data(), ts_va[1], tidx * 8u));
                        double gdn_ms[2]={}, gqa_ms[2]={}, moe_ms[2]={}, emb_ms[2]={}, far_ms[2]={}, lm_ms[2]={}, tot_ms[2]={};
                        for (size_t g = 0; g < 2; ++g) {
                            const auto& tv = (g == 0) ? t0 : t1;
                            emb_ms[g] = (tv[1] - tv[0]) * 40e-6;
                            uint32_t p = 1u;
                            for (uint32_t il = 0; il < 48u; ++il) {
                                const double d_mix = (tv[p + 1] - tv[p]) * 40e-6;
                                const double d_moe = (tv[p + 2] - tv[p + 1]) * 40e-6;
                                if (layers[0][il].is_gdn) gdn_ms[g] += d_mix; else gqa_ms[g] += d_mix;
                                moe_ms[g] += d_moe;
                                p += 2u;
                            }
                            far_ms[g] = (tv[p + 1] - tv[p]) * 40e-6;
                            lm_ms[g]  = (tv[p + 2] - tv[p + 1]) * 40e-6;
                            tot_ms[g] = (tv[p + 2] - tv[0]) * 40e-6;
                        }
                        std::printf("[SUPER-PROFILE wgs=%u 48L s_memrealtime] GPU0 | GPU1 (ms/tok):\n"
                                    "  emb+rms=%.3f|%.3f  gdn_super(36L)=%.3f|%.3f  gqa_super(12L)=%.3f|%.3f  moe_super(48L)=%.3f|%.3f  final_ar=%.3f|%.3f  lm_head=%.3f|%.3f  TOTAL=%.3f|%.3f\n",
                                    swgs, emb_ms[0], emb_ms[1], gdn_ms[0], gdn_ms[1], gqa_ms[0], gqa_ms[1],
                                    moe_ms[0], moe_ms[1], far_ms[0], far_ms[1], lm_ms[0], lm_ms[1], tot_ms[0], tot_ms[1]);
                        uint64_t p0[16] = {}, p1[16] = {};
                        VJ_CHECK(rt.download_from_vram(0, p0, S[0].super_bar_va + 1024u, sizeof(p0)));
                        VJ_CHECK(rt.download_from_vram(1, p1, S[1].super_bar_va + 1024u, sizeof(p1)));
                        auto pm = [](uint64_t c) { return static_cast<double>(c) * 40e-6; };
                        std::printf("  [GDN sub 36L] AR=%.3f|%.3f  QKV=%.3f|%.3f  ConvRec=%.3f|%.3f  Out=%.3f|%.3f\n",
                                    pm(p0[0]), pm(p1[0]), pm(p0[1]), pm(p1[1]), pm(p0[2]), pm(p1[2]), pm(p0[3]), pm(p1[3]));
                        std::printf("  [GQA sub 12L] AR=%.3f|%.3f  QKV=%.3f|%.3f  Prep=%.3f|%.3f  Flash=%.3f|%.3f  Merge=%.3f|%.3f  Out=%.3f|%.3f\n",
                                    pm(p0[4]), pm(p1[4]), pm(p0[5]), pm(p1[5]), pm(p0[6]), pm(p1[6]), pm(p0[7]), pm(p1[7]), pm(p0[8]), pm(p1[8]), pm(p0[9]), pm(p1[9]));
                        std::printf("  [MoE sub 48L (dot8=%u iq2s=%u)] AR=%.3f|%.3f  Rtr+ShGU=%.3f|%.3f  Top8+Swi=%.3f|%.3f  GU_IQ2S=%.3f|%.3f  GU_DOT8=%.3f|%.3f  MoEDn+PP=%.3f|%.3f\n",
                                    gu_dot8_promoted, 48u - gu_dot8_promoted,
                                    pm(p0[10]), pm(p1[10]), pm(p0[11]), pm(p1[11]), pm(p0[12]), pm(p1[12]),
                                    pm(p0[13]), pm(p1[13]), pm(p0[15]), pm(p1[15]), pm(p0[14]), pm(p1[14]));
                    }
                }
                if (opt.profile_stages) run_m1_profile();
                if (vout) {
                    auto out_c = run_greedy_raw(golden::kCodePrompt, 32, nullptr);
                    const size_t cm = golden::matched_prefix(out_c, golden::kCodeRef32);
                    vout->lossless_matched = static_cast<uint32_t>(cm);
                    vout->lossless_total   = static_cast<uint32_t>(golden::kCodeRef32.size());
                    vout->parity_ok        = (cm == golden::kCodeRef32.size()) && (eq128 == trusted_raw128.size());
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
    // v1.2 C5: a speculative/MTP/DFlash driver that disagrees with the raw greedy reference must fail the
    // run (non-zero exit) instead of only printing a lossless= counter next to a tok/s number.

    auto run_full_bench_body = [&](bool include_prefill, warm::SweepVariantResult* vout) -> bool {
        uint64_t ue0 = 0, de0 = 0, ue1 = 0, de1 = 0;
        read_ecc_totals(ue0, de0);
        double pass_pwr = 0.0, pass_tmem = 0.0;
        double best_spec_tps = 0.0;
        uint32_t spec_eq = 0, spec_tot = 0;
        int spec_fd = -1;
        bool all_lossless = true;
        // eq = matching prefix tokens, n_ref = raw reference length, n_got = driver output length.
        auto validate_spec = [&](const char* what, size_t eq, size_t n_ref, size_t n_got, int first_diff) {
            if (eq == n_ref && n_got == n_ref) return;
            all_lossless = false;
            if (bench_spec_mismatch) return;
            bench_spec_mismatch = true;
            bench_spec_mismatch_msg = std::string("speculative validation failed: ") + what + " matched " +
                                      std::to_string(eq) + "/" + std::to_string(n_ref) +
                                      " greedy tokens vs the raw reference (got " + std::to_string(n_got) +
                                      " tokens, first_diff=" + std::to_string(first_diff) + ")";
            std::fprintf(stderr, "[BENCH-VALIDATE] FAIL %s\n", bench_spec_mismatch_msg.c_str());
        };
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
                for (size_t g = 0; g < 2; ++g)
                    for (uint32_t gi = 0; gi < n_gdn_layers; ++gi) {
                        const uint64_t sva = opt.spec_gdn
                            ? (S[g].spec_gdn_s_slot_va[spec_gdn_active] + static_cast<uint64_t>(gi) * s_bytes)
                            : (S[g].gdn_ssm_ring_va[gi] + static_cast<uint64_t>(ring_head) * s_bytes);
                        VJ_CHECK(rt.download_from_vram(g, blob.data(), sva, s_bytes));
                        std::fwrite(blob.data(), 1, s_bytes, df);
                        VJ_CHECK(rt.download_from_vram(g, blob.data(), S[g].gdn_conv_ring_va[gi], c_bytes));
                        std::fwrite(blob.data(), 1, c_bytes, df);
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
            const uint32_t V_tot = voc_rows[0] + voc_rows[1];
            std::ofstream lf(opt.dump_logits_path, std::ios::binary);
            if (!lf) {
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
                for (size_t g = 0; g < 2; ++g) {
                    struct LogKA { uint64_t wq, wd, xq, xd, out; uint32_t lr, cols, stride; };
                    const LogKA lka{out_w[g].qs_va, out_w[g].ds_va, S[g].x_q8_va + xq_off, S[g].x_ds_va + xd_off,
                                    S[g].logits_va, voc_rows[g], D, out_w[g].row_stride_bytes};
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].vocab_logits, kVocabBlocks * 256u, 1, 1, 256, 1, 1, &lka, sizeof(lka), true));
                    VJ_CHECK(rt.submit_batch(g));
                }
                VJ_CHECK(rt.wait_idle(0) && rt.wait_idle(1));
                VJ_CHECK(rt.download_from_vram(0, logits.data(),               S[0].logits_va, voc_rows[0] * 4ULL));
                VJ_CHECK(rt.download_from_vram(1, logits.data() + voc_rows[0], S[1].logits_va, voc_rows[1] * 4ULL));
                lf.write(reinterpret_cast<const char*>(logits.data()), static_cast<std::streamsize>(V_tot * 4ULL));
                std::printf("[DUMP-LOGITS] prompt %u/%u (M=%u, dot8=%d clip=%.2f) -> top1=%d logit=%.4f\n",
                            pi + 1u, n_pos, M_pr, opt.prefill_dot8 ? 1 : 0, opt.dot8_clip_alpha,
                            top1, (top1 >= 0 && static_cast<uint32_t>(top1) < V_tot) ? logits[top1] : 0.0f);
            }
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
            for (size_t g = 0; g < 2; ++g) {
                const uint32_t local_ff = ff_size[g];
                const double dn_d8_row_b = double((asym_experts ? dot8_iq3_asym_layout(1u, local_ff) : dot8_iq3_layout(1u, local_ff)).read_bytes_per_row());
                double bg = double(voc_rows[g]) * (double(D) + double(D / 32u) * 2.0);
                for (uint32_t il = 0; il < 48u; ++il) {
                    const auto& lw = layers[g][il];
                    if (lw.is_gdn) {
                        bg += q4_real_b(lw.w_gdn_qkv_gab) + q4_real_b(lw.w_ssm_out) + 2.0 * 32.0 * 128.0 * 128.0 * 4.0;
                    } else {
                        bg += q4_real_b(lw.w_attn_qkv) + q4_real_b(lw.w_o);
                    }
                    bg += 256.0 * D * 2.0 + q4_real_b(lw.w_sh_gu) + q4_real_b(lw.w_sh_down);
                    bg += (lw.w_gu_d8_w_va != 0) ? (8.0 * local_ff * gu_d8_row_b) : (8.0 * local_ff * 1968.0);
                    bg += (lw.w_dn_d8_w_va != 0) ? (8.0 * D * dn_d8_row_b)        : (8.0 * D * 208.0);
                }
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
            validate_spec("tg128 suffix-tree draft", eq, raw128.size(), suf128.size(), first_diff);
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
            validate_spec("tg128 MTP (pure nextn) prose", eqp, raw_prose.size(), mtp_prose.size(), fdp);
            std::printf("[BENCH tg128 MTP K=%u] %.2f tok/s (%.2fx vs raw %.2f) | prose lossless=%u/%zu (first_diff=%d) | acc=%llu/%llu (%.1f%%)\n",
                        opt.mtp_k, mtp_prose_tps, mtp_prose_tps / std::max(raw_prose_tps, 1e-6), raw_prose_tps,
                        eqp, raw_prose.size(), fdp,
                        static_cast<unsigned long long>(macc), static_cast<unsigned long long>(mdr),
                        mdr ? (100.0 * macc / mdr) : 0.0);

            // 2. Hybrid (SuffixTree + blk.48 MTP) on CODE + 129/129 lossless gate!
            double hyb128_tps = 0.0, hyb512_tps = 0.0;
            uint64_t hacc = 0, hdr = 0, hsfx = 0;
            auto hyb128 = run_greedy_mtp(p_rep_code, 129, opt.mtp_k, true, &hyb128_tps, &hacc, &hdr, &hsfx);
            uint32_t eqh = 0; int fdh = -1;
            for (size_t i = 0; i < raw128.size() && i < hyb128.size(); ++i) {
                if (raw128[i] == hyb128[i]) ++eqh; else if (fdh == -1) fdh = static_cast<int>(i);
            }
            validate_spec("tg128 MTP+suffix hybrid code", eqh, raw128.size(), hyb128.size(), fdh);
            std::printf("[BENCH tg128 Hybrid ] %.2f tok/s (%.2fx vs raw) | code lossless=%u/%zu (first_diff=%d) | sfx_hits=%llu acc=%llu/%llu (%.1f%%)\n",
                        hyb128_tps, hyb128_tps / std::max(res.tg128_raw_tps, 1e-6),
                        eqh, raw128.size(), fdh,
                        static_cast<unsigned long long>(hsfx), static_cast<unsigned long long>(hacc),
                        static_cast<unsigned long long>(hdr), hdr ? (100.0 * hacc / hdr) : 0.0);

            uint64_t hacc5 = 0, hdr5 = 0, hsfx5 = 0;
            (void) run_greedy_mtp(p_rep_code, 513, 4u, true, &hyb512_tps, &hacc5, &hdr5, &hsfx5);
            std::printf("[BENCH tg512 Hybrid ] %.2f tok/s | sfx_hits=%llu acc=%llu/%llu (%.1f%%)\n",
                        hyb512_tps, static_cast<unsigned long long>(hsfx5),
                        static_cast<unsigned long long>(hacc5), static_cast<unsigned long long>(hdr5),
                        hdr5 ? (100.0 * hacc5 / hdr5) : 0.0);
            read_hwmon_max(max_pwr, max_tmem);
        }

        if (df.loaded) {
            auto enq_df_gemm = [&](const Q4A8WeightDev& w, uint64_t xq, uint64_t xds, uint64_t y, uint32_t M) {
                if (M == 1u) {
                    struct M1W32KA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, stride, pad; uint64_t ptx; };
                    M1W32KA ka{w.qs_va, w.dm_va, xq, xds, y, w.rows, w.cols, w.row_stride_bytes, 0u, 0ULL};
                    VJ_CHECK(rt.enqueue_kernel(df_gpu, *K[df_gpu].gemv_q4_w32, ((w.rows + 7u) / 8u) * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
                } else if (M <= 8u) {
                    struct GemmPpKA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, M, stride; uint64_t ptx; };
                    GemmPpKA ka{w.qs_va, w.dm_va, xq, xds, y, w.rows, w.cols, M, w.row_stride_bytes, 0ULL};
                    VJ_CHECK(rt.enqueue_kernel(df_gpu, *K[df_gpu].gemm_q4_m8, ((w.rows + 7u) / 8u) * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
                } else if (M <= 12u) {
                    struct GemmPpKA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, M, stride; uint64_t ptx; };
                    GemmPpKA ka{w.qs_va, w.dm_va, xq, xds, y, w.rows, w.cols, M, w.row_stride_bytes, 0ULL};
                    VJ_CHECK(rt.enqueue_kernel(df_gpu, *K[df_gpu].gemm_q4_m12, ((w.rows + 7u) / 8u) * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
                } else {
                    struct GemmPpKA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, M, stride; uint64_t ptx; };
                    GemmPpKA ka{w.qs_va, w.dm_va, xq, xds, y, w.rows, w.cols, M, w.row_stride_bytes, 0ULL};
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
                                        pos_beg + off, c, df.n_kv, df.swa_win - 1u, df.rope_base, df.rms_eps};
                        VJ_CHECK(rt.enqueue_kernel(df_gpu, *K[df_gpu].df_kv_inject, df.n_kv * 128u, c, 1, 128, 1, 1, &kka, sizeof(kka), true));
                    }
                }
                VJ_CHECK(rt.submit_batch(df_gpu) && rt.wait_idle(df_gpu));
            };

            double df_t_body = 0.0, df_t_lm = 0.0, df_t_sel = 0.0;
            float  df_pmin_cur = opt.dflash_p_min;
            auto df_now_ms = []() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
            auto dflash_draft = [&](int32_t anchor_tok, uint32_t s_pos, uint32_t k_draft) -> std::vector<int32_t> {
                require_token(anchor_tok, "DFlash anchor embedding");
                const double td_a = df_now_ms();
                const uint32_t B = k_draft + 1u;
                const uint32_t T = k_draft;
                struct EmbBlkKA { uint64_t eq, ed, msk, hout; uint32_t atok, B, D; };
                const EmbBlkKA ebka{emb_w[df_gpu].qs_va, emb_w[df_gpu].ds_va, df.mask_embd_f32_va, df.h_va,
                                    static_cast<uint32_t>(anchor_tok), B, D};
                VJ_CHECK(rt.enqueue_kernel(df_gpu, *K[df_gpu].df_embed_blk, B * 256u, 1, 1, 256, 1, 1, &ebka, sizeof(ebka), true));

                struct RmsKA  { uint64_t h, xa, w, xn, xq, xd; uint32_t M, D; float eps; };
                struct ConvKA { uint64_t u, dyn, base, hio, q8, ds; uint32_t B, D, side; };
                struct AttnKA { uint64_t qkv, qn, kn, kc, vc, oq, od; uint32_t spos, B, nq, nkv, swa; float rb, eps; };
                struct SwiKA  { uint64_t gu, q8, ds; uint32_t B, ff; };

                for (uint32_t l = 0u; l < df.n_layer; ++l) {
                    const auto& dl = df.L[l];
                    // Attention sublayer
                    const RmsKA r0{df.h_va, 0ULL, dl.attn_norm_va, df.xnorm_va, df.xq8_va, df.xds_va, B, D, df.rms_eps};
                    VJ_CHECK(rt.enqueue_kernel(df_gpu, *K[df_gpu].rms_quant, B * 256u, 1, 1, 256, 1, 1, &r0, sizeof(r0), true));
                    enq_df_gemm(dl.w_attn_conv, df.xq8_va, df.xds_va, df.dyn_va, B);
                    const ConvKA c0{df.xnorm_va, df.dyn_va, dl.attn_conv_base_va, 0ULL, df.xq8_va, df.xds_va, B, D, 0u};
                    VJ_CHECK(rt.enqueue_kernel(df_gpu, *K[df_gpu].df_dyn_conv, ((D + 255u) / 256u) * 256u, B, 1, 256, 1, 1, &c0, sizeof(c0), true));
                    enq_df_gemm(dl.w_qkv, df.xq8_va, df.xds_va, df.qkv_va, B);
                    const AttnKA aka{df.qkv_va, dl.q_norm_va, dl.k_norm_va, dl.kv_k_va, dl.kv_v_va,
                                     df.attn_q8_va, df.attn_ds_va, s_pos, B, df.n_head, df.n_kv, df.swa_win, df.rope_base, df.rms_eps};
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

                // Stage x_q8/x_ds into peer GPU local VRAM (~3 us) so BOTH GPUs run vocab_topk16 from local L2 VRAM!
                const CpU32KA cp1q{S[df_peer].x_q8_va, df_io_xq8_va, n_q8_u32};
                const CpU32KA cp1d{S[df_peer].x_ds_va, df_io_xds_va, n_ds_u32};
                VJ_CHECK(rt.enqueue_kernel(df_peer, *K[df_peer].memcpy_u32, ((n_q8_u32 + 255ULL) / 256ULL) * 256ULL, 1, 1, 256, 1, 1, &cp1q, sizeof(cp1q), true));
                VJ_CHECK(rt.enqueue_kernel(df_peer, *K[df_peer].memcpy_u32, ((n_ds_u32 + 255ULL) / 256ULL) * 256ULL, 1, 1, 256, 1, 1, &cp1d, sizeof(cp1d), true));

                // TP=2 LDS-staged LM-head Top-16 across both MI50s from local VRAM!
                // Uses FR-Spec shortlist (16,384 rows/GPU, 120 WGs x ceil(T/4)) when --fr-spec is enabled.
                const bool df_use_fr = (use_fr_spec && std::getenv("VEGAJET_DF_FULL_VOCAB") == nullptr);
                const uint32_t df_voc_blocks = df_use_fr ? 120u : kVocabBlocks;
                for (size_t g = 0; g < 2; ++g) {
                    const Q8WeightDev& vw = df_use_fr ? fr_w[g] : out_w[g];
                    struct TopKA { uint64_t wq, wd, xq, xd, tmap, bv, bi; uint32_t lr, cols, ro, stride, T; };
                    const TopKA tka{vw.qs_va, vw.ds_va,
                                    (g == df_gpu) ? df.xq8_va : S[df_peer].x_q8_va,
                                    (g == df_gpu) ? df.xds_va : S[df_peer].x_ds_va,
                                    df_use_fr ? fr_tok_map_va[g] : 0ULL,
                                    (g == 0) ? df_io_val_va0 : df_io_val_va1,
                                    (g == 0) ? df_io_id_va0  : df_io_id_va1,
                                    df_use_fr ? fr_rows_per_gpu : voc_rows[g],
                                    D, voc_off[g], vw.row_stride_bytes, T};
                    VJ_CHECK(rt.enqueue_kernel(g, *K[g].df_vocab_topk16, df_voc_blocks * 256u, (T + 3u) / 4u, 1, 256, 1, 1, &tka, sizeof(tka), true));
                }
                VJ_CHECK(rt.submit_batch(0) && rt.submit_batch(1));
                VJ_CHECK(rt.wait_idle(0) && rt.wait_idle(1));
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
                        consider(df_io_val_host0[base + j], df_io_id_host0[base + j]);
                        consider(df_io_val_host1[base + j], df_io_id_host1[base + j]);
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
                        fail_run("DFlash draft argmax is outside the vocabulary");
                    }
                    require_token(pred, "DFlash draft argmax");
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
                reset_state();
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
                evict::ControllerConfig ecfg;
                ecfg.n_layers = 48u; ecfg.n_experts = 256u; ecfg.k_min = 3u;
                ecfg.k_max = std::max<uint32_t>(3u, std::min<uint32_t>(k_draft, df.block_size - 1u));
                ecfg.max_inflation = 3.0;
                evict::CostModel ecost;
                ecost.fixed_us = 2500.0; ecost.per_token_us = 350.0; ecost.per_expert_us = 14.0;
                evict::EvictController evict_ctl(ecfg, ecost);
                // v1.2 C7+S1: depth and inflation from the measured per-M verify time + live acceptance.
                const bool v12_spec_gov = (std::getenv("VEGAJET_V12_SPEC_GOV") != nullptr &&
                                           std::getenv("VEGAJET_V12_SPEC_GOV")[0] == '1');
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
                uint32_t n_cyc = 0;
                auto now_ms = []() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
                const auto td0 = std::chrono::steady_clock::now();
                while (static_cast<int>(gen.size()) < n_gen) {
                    const uint32_t rem = static_cast<uint32_t>(n_gen - static_cast<int>(gen.size()));
                    uint32_t k_target = use_evict ? evict_ctl.next_depth() : k_draft;
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
                    const uint32_t k_cur = std::min<uint32_t>({k_target, df.block_size - 1u, kGdnRingSlots - 2u, (rem > 1u) ? (rem - 1u) : 0u});
                    const uint32_t k_sfx = std::min<uint32_t>({opt.suffix_max_draft, kGdnRingSlots - 2u, (rem > 1u) ? (rem - 1u) : 0u});
                    if (k_cur == 0u) {
                        auto preds = step_batch({gen.back()}, pos, false, false, true, true);
                        dflash_inject_kv(pos, 1u);
                        ++pos;
                        gen.push_back(preds[0]);
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
                    const uint32_t dk = static_cast<uint32_t>(drafts.size());
                    std::vector<int32_t> vtoks;
                    vtoks.reserve(1u + dk);
                    vtoks.push_back(gen.back());
                    for (int32_t tok_d : drafts) vtoks.push_back(tok_d);
                    auto preds = step_batch(vtoks, pos, false, true, true, true);
                    const double t2 = now_ms();
                    uint32_t acc = 0u;
                    while (acc < dk && preds[acc] == drafts[acc]) {
                        gen.push_back(drafts[acc]);
                        if (use_zt_cascade) zt_cascade.commit(drafts[acc]);
                        else if (use_hybrid_suffix) cascade.commit(drafts[acc]);
                        ++acc;
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
                        // Measured time inflation replaces the fabricated 1 + 0.35*dk constant (C7); before the
                        // first measurement the old constant keeps the controller warm-started.
                        double infl = v12_spec_gov ? df_gov.time_inflation(dk + 1u) : 0.0;
                        if (!(infl > 0.0)) infl = 1.0 + 0.35 * static_cast<double>(dk);
                        evict_ctl.observe(dk, acc, infl);
                    }
                    acc_tot += acc; dr_tot += dk;
                    const uint32_t n_commit = acc + 1u;
                    commit_spec_gdn(acc, dk);
                    ring_head = (ring_head + n_commit) % kGdnRingSlots;
                    conv_head = (conv_head + n_commit) % kGdnRingSlots;
                    dflash_inject_kv(pos, n_commit);
                    const double t3 = now_ms();
                    if (v12_spec_gov && !used_suffix) df_gov.observe_catchup(n_commit, t3 - t2);
                    t_dr += (t1 - t0); t_ver += (t2 - t1); t_inj += (t3 - t2); ++n_cyc;
                    pos += n_commit;
                }
                const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - td0).count();
                if (out_tps && n_gen > 1) *out_tps = (n_gen - 1) / std::max(sec, 1e-9);
                if (out_acc) *out_acc = acc_tot;
                if (out_dr)  *out_dr  = dr_tot;
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
            if (df_hyb_tps > best_spec_tps) { best_spec_tps = df_hyb_tps; spec_eq = eqh; spec_tot = static_cast<uint32_t>(raw128.size()); spec_fd = fdh; }
            validate_spec("tg128 DFlash2 code", eqc, raw128.size(), df_code.size(), fdc);
            validate_spec("tg128 DFlash2+hybrid code", eqh, raw128.size(), df_hyb.size(), fdh);
            if (eqp != raw_prose.size()) all_lossless = false;

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
            double tot[2] = {0.0, 0.0};
            for (size_t g = 0; g < 2; ++g) for (double v : st_acc[g]) tot[g] += v;
            std::printf("[PP-STAGES pp2048 M=%u] uncached request %.0f ms with per-stage drains (%.0f ms = %.2f tok/s without); ms per request, rank 0 | rank 1:\n",
                        opt.prefill_chunk, wall, 2048.0 * 1000.0 / res.pp2048_tps, res.pp2048_tps);
            for (uint32_t st = 0; st < kStCount; ++st)
                std::printf("  %-38s %8.1f | %8.1f%s\n", kStName[st], st_acc[0][st], st_acc[1][st],
                            (st == kStAr || st == kStVocab) ? "   (both ranks together, booked on rank 0)" : "");
            std::printf("  %-38s %8.1f | %8.1f   (ranks run concurrently in normal operation: critical path = shared stages + max rank)\n",
                        "sum", tot[0], tot[1]);
        }
        read_hwmon_max(pass_pwr, pass_tmem);
        max_pwr = std::max(max_pwr, pass_pwr); max_tmem = std::max(max_tmem, pass_tmem);
        }

        if (!vout) run_m1_profile();
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

    if (opt.serve_hook) {
        uint64_t ue_base = 0, de_base = 0;
        read_ecc_totals(ue_base, de_base);
        auto gen_fn = [&](const std::vector<int32_t>& prompt_ids, int32_t max_tokens,
                          const OrnithGpuOptions::ServeTokenCb& on_tok) {
            run_state.check();
            if (device_faulted) fail_run(res.error);
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
            std::vector<int32_t> pids = prompt_ids;
            if (pids.empty()) pids.push_back(tok.bos_id() >= 0 ? tok.bos_id() : 0);
            if (pids.size() >= opt.max_ctx) {
                pids.erase(pids.begin(), pids.begin() + static_cast<std::ptrdiff_t>(pids.size() - (opt.max_ctx - 1u)));
            }
            const uint32_t n_prompt = static_cast<uint32_t>(pids.size());
            // v1.2 D1 (VEGAJET_V12_SERVE_SPEC=1): serve through the same suffix-tree speculative driver the bench
            // measures (`tg128 Suffix`), which commits only tokens that match the greedy argmax, so the served text is
            // unchanged while the verify batches raise throughput. Off: the plain M=1 greedy loop below.
            if (v12_serve_spec()) {
                const SuffixTokenCb report = [&](int32_t tid, uint32_t n_comp, double pp_ms, double dec_ms) {
                    return on_tok(tid, n_prompt, n_comp, pp_ms, dec_ms);
                };
                double spec_tps = 0.0;
                (void)run_greedy_suffix(pids, max_tokens, &spec_tps, nullptr, nullptr, nullptr, &report, opt.max_ctx);
                return;
            }
            reset_state();
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
            require_token(cur, "serve prefill argmax");
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
        const std::string desc = std::string("ornith-gfx906-tp2 (super=") + (sup_on ? "on" : "off") + ", greedy" +
                                 (v12_serve_spec() ? "+suffix-spec)" : ")");
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
                // A partial reload/rebind invalidates cached K[] pointers. Never accept another command.
                if (!ok_res) fail_run("Hot-reload failed; runner stopped: " + rerr);
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

    } catch (const guard::RunFailure& failure) {
        run_state.latch(failure.what());
        res.error = run_state.error();
        rt.abort_dispatches();
        res.ok = false;
    }
    if (v12_dense_persist()) {
        std::printf("[D13] kernel=ornith_gemv_q4a8_m1_w32_persist_kernel launches=%llu\n",
                    static_cast<unsigned long long>(dense_persist_launches));
        if (dense_persist_launches == 0u && res.error.empty())
            res.error = "D13 requested but no persistent GEMV launched";
    }
    device_fault_check("run end");   // latch while io_fault_host is still allocated
    res.ok = false;
    rt.free_buffer(tap_buf);
    rt.free_buffer(io_buf);
    telem.stop();
    max_tmem = std::max(max_tmem, telem.max_tmem_c);
    const double avg_limit = telem.cards() >= 3 ? 155.0 : 205.0;
    const bool telem_ran = telem.samples != 0;
    // Runs shorter than the 10 s window never fill it; gate those on the whole-run mean power instead.
    const double gated_avg = telem.max_avg_w > 0.0 ? telem.max_avg_w : telem.max_mean_w();
    const bool telem_ok = !telem_ran || (gated_avg <= avg_limit && telem.peak_w < 270.0 && max_tmem < 65.0);
    std::printf("[TELEM] power1_input: worst 10 s rolling avg=%.1f W (mean=%.1f W, <=%.0f W) | worst 1 s avg=%.1f W (info) | "
                "instantaneous peak=%.1f W (<270 W) | max_TMem=%.1f C (<65 C) | %llu samples -> %s\n",
                telem.max_avg_w, telem.max_mean_w(), avg_limit, telem.max_avg1s_w, telem.peak_w, max_tmem,
                static_cast<unsigned long long>(telem.samples), !telem_ran ? "SKIPPED (no samples)" : (telem_ok ? "PASS" : "FAIL"));
    (void)max_pwr;
    if (!run_state.error().empty() || !res.error.empty() || device_faulted) { res.ok = false; return res; }
    if (parity_code_match != golden::kCodeRef32.size()) {
        res.error = "Tier-1 llama.cpp code parity failed: " + std::to_string(parity_code_match) + "/" +
                    std::to_string(golden::kCodeRef32.size()) + " exact greedy tokens";
        return res;
    }
    if (!telem_ok) { res.error = "power/thermal telemetry gate failed (see [TELEM])"; return res; }
    // v1.2 C5: a speculative/MTP mismatch in --bench is a hard failure, not a metric.
    if (bench_spec_mismatch) { res.error = bench_spec_mismatch_msg; return res; }
    // v1.2 C1: a fail-closed device fault is never reported as a clean run, even if the caller kept going.
    if (device_faulted) return res;
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
    if (rt.num_gpus() != 2 || rt.allreduce_transport() != kfd::AllReduceTransport::PeerBarP2P) {
        std::fprintf(stderr, "[bench-ar] needs 2 GPUs with the PEER_BAR_P2P transport\n");
        return 3;
    }
    constexpr uint32_t D = 3072, kVerifyRounds = 64, kBatch = 96, kBatches = 40;
    struct ArKA {
        uint64_t xp, ptx, lrx, pfl, lfl, bar, h, w, xn, xq, xd;
        uint32_t M, D, ep, dn;
        float eps;
    };
    struct Dev { uint64_t partial, bar, h, w, xn, xq, xd; };
    Dev dv[2];
    const std::vector<float> ones(D, 1.0f), zeros(D, 0.0f);
    const std::vector<uint32_t> zbar(64, 0u);
    for (size_t g = 0; g < 2; ++g) {
        auto& a = rt.vram_arena(g);
        dv[g] = Dev{a.alloc(D * 4u), a.alloc(256u), a.alloc(D * 4u), a.alloc(D * 4u), a.alloc(D * 4u), a.alloc(D), a.alloc((D / 32u) * 8u)};
        if (!dv[g].partial || !dv[g].bar || !dv[g].h || !dv[g].w || !dv[g].xn || !dv[g].xq || !dv[g].xd || !rt.upload_to_vram(g, dv[g].w, ones.data(), D * 4u) || !rt.upload_to_vram(g, dv[g].bar, zbar.data(), 256u)) {
            std::fprintf(stderr, "[bench-ar] buffer setup failed\n");
            return 3;
        }
    }
    const uint64_t pay_base = rt.mailbox_payload_gpu_va();
    const uint64_t flg_base = rt.mailbox_flags_gpu_va();


    int rc = 0;
    for (int wire = 0; wire < 2; ++wire) {
        const char* wname = wire ? "INT8 block-32" : "FP16";
        const char* kname = wire ? "ornith_p2p_ar_i8_res_rms_quant_kernel" : "ornith_p2p_ar_res_rms_quant_kernel";
        const kfd::LoadedKernel* K[2] = {rt.get_kernel(0, kname), rt.get_kernel(1, kname)};
        if (!K[0] || !K[1]) { std::fprintf(stderr, "[bench-ar] kernel %s missing\n", kname); return 3; }
        rt.reset_allreduce_mailboxes();
        uint32_t epoch = 1u;
        auto enqueue_ar = [&](uint32_t slot) -> bool {
            const uint32_t ep = epoch++;
            const uint32_t pp = ep & 1u;
            for (size_t g = 0; g < 2; ++g) {
                const size_t peer = 1u - g;
                const ArKA aka{dv[g].partial,
                               pay_base + peer * kfd::kArGpuBytes + pp * kfd::kArSlotBytes,
                               pay_base + g    * kfd::kArGpuBytes + pp * kfd::kArSlotBytes,
                               flg_base + peer * 4096ULL + (slot & 63u) * 64ULL,
                               flg_base + g    * 4096ULL + (slot & 63u) * 64ULL,
                               dv[g].bar, dv[g].h, dv[g].w, dv[g].xn, dv[g].xq, dv[g].xd, 1u, D, ep, 1u, 1e-6f};
                if (!rt.enqueue_kernel(g, *K[g], 256u, 1, 1, 256, 1, 1, &aka, sizeof(aka), true)) return false;
            }
            return true;
        };
        auto run_queued = [&]() { return rt.submit_batch(0) && rt.submit_batch(1) && rt.wait_idle(0) && rt.wait_idle(1); };

        // ---- verification: both ranks must hold exactly h = deq(x0) + deq(x1) ----
        uint64_t verify_errs = 0, rng = 0x9E3779B97F4A7C15ull + static_cast<uint64_t>(wire);
        std::vector<float> x[2] = {std::vector<float>(D), std::vector<float>(D)}, hd(D), ref(D), dq[2] = {std::vector<float>(D), std::vector<float>(D)};
        for (uint32_t r = 0; r < kVerifyRounds; ++r) {
            for (size_t g = 0; g < 2; ++g) {
                for (uint32_t i = 0; i < D; ++i) {
                    rng = rng * 6364136223846793005ull + 1442695040888963407ull;
                    const float u = static_cast<float>(static_cast<int32_t>(rng >> 33) - (1 << 30)) / static_cast<float>(1 << 30);   // [-1, 1)
                    x[g][i] = u * ((r % 4u == 3u && (i >> 5) % 7u == 0u) ? 40.0f : 1.5f);   // some large-range blocks
                }
                if (r == 5u) std::fill(x[g].begin(), x[g].begin() + 64, 0.0f);   // all-zero blocks (amax == 0)
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
            for (uint32_t i = 0; i < D; ++i) { const volatile float sum = dq[0][i] + dq[1][i]; ref[i] = sum; }
            if (!enqueue_ar(r) || !run_queued()) { std::fprintf(stderr, "[bench-ar] dispatch failed\n"); return 3; }
            for (size_t g = 0; g < 2; ++g) {
                if (!rt.download_from_vram(g, hd.data(), dv[g].h, D * 4u)) return 3;
                for (uint32_t i = 0; i < D; ++i)
                    if (std::memcmp(&hd[i], &ref[i], 4) != 0) ++verify_errs;
            }
        }

        // ---- FP16 only: the single-pass kernel must match the two-pass reference bit for bit on every output ----
        if (wire == 0) {
            const kfd::LoadedKernel* KR[2] = {rt.get_kernel(0, "ornith_p2p_ar_res_rms_quant_ref_kernel"), rt.get_kernel(1, "ornith_p2p_ar_res_rms_quant_ref_kernel")};
            if (!KR[0] || !KR[1]) { std::fprintf(stderr, "[bench-ar] reference kernel missing\n"); return 3; }
            uint64_t ab_diffs[4] = {0, 0, 0, 0};
            uint32_t ab_shown = 0;
            std::vector<float> h0[2] = {std::vector<float>(D), std::vector<float>(D)}, wv(D);
            std::vector<uint8_t> out[2][2];   // [variant][gpu]: h | xn | xq | xd
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
                        const ArKA aka{dv[g].partial, pay_base + peer * kfd::kArGpuBytes + pp * kfd::kArSlotBytes, pay_base + g * kfd::kArGpuBytes + pp * kfd::kArSlotBytes,
                                       flg_base + peer * 4096ULL + (r & 63u) * 64ULL, flg_base + g * 4096ULL + (r & 63u) * 64ULL,
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
        std::printf("[BENCH-AR %-13s] wire=%u B/AR | %.2f us/AR mean, %.2f us/AR best | 96x = %.3f ms mean, %.3f ms best | "
                    "verify_errs=%llu over %u rounds x 2 ranks x %u elems\n",
                    wname, wire_bytes, total / (kBatches * kBatch) * 1e6, best / kBatch * 1e6, total / kBatches * 1e3, best * 1e3,
                    static_cast<unsigned long long>(verify_errs), kVerifyRounds, D);
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
            struct MoeSupKA { uint64_t desc; uint32_t ar_ep, out_ep, ssense; uint64_t fault; };
            static_assert(sizeof(MoeSupKA) == 32u, "kernarg ABI drift");
            // 1-layer check first!
            {
                const MoeSupKA ka{mdesc_va, 0u, 0u, s_sense++, 0ULL};
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
                const MoeSupKA ka{mdesc_va + static_cast<uint64_t>(b) * sizeof(ornith_super::MoeSuperDesc), 0u, 0u, s_sense++, 0ULL};
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
                struct GdnSupKA { uint64_t desc, sring_ov; uint32_t prev_ep, out_ep, cin, cout, shead, uring, ssense, gsense; uint64_t fault; };
                static_assert(sizeof(GdnSupKA) == 56u, "kernarg ABI drift");
                const GdnSupKA gska{gd_va, 0ULL, 0u, 0u, 0u, 1u, 0u, 0u, s_sense++, g_sense++, 0ULL};
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
                struct GqaSupKA { uint64_t desc; uint32_t prev_ep, out_ep, pos0, ssense; uint64_t fault; };
                static_assert(sizeof(GqaSupKA) == 32u, "kernarg ABI drift");
                const GqaSupKA qska{qa_va, 0u, 0u, pos0, s_sense++, 0ULL};
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

class OrnithLiveEngine final : public IEngine {
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
        const auto budget = preflight::check_arena(preflight::gfx906_vram_bytes("/"), preflight::kOrnithArenaGiB,
                                                   spec.tp, "the Ornith-122B GPU server");
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

        // Default to the qualified fast M=1 super-kernel layout unless the operator overrides via env. With the
        // speculative serving driver on, RAW_ONLY must be off: it sizes the GDN ring to one slot and promotes the expert
        // weights in place for M=1, neither of which the M>=2 verify step can use.
        if (spec.tp == 2) {
        if (v12_serve_spec())
            setenv("VEGAJET_RAW_ONLY", "0", 1);
        else
            setenv("VEGAJET_RAW_ONLY", "1", 0);
        setenv("VEGAJET_GU_Q3", "1", 0);
        setenv("VEGAJET_VA_CLUSTER", "1", 0);
        setenv("VEGAJET_VOCAB_MUBUF", "1", 0);
        setenv("VEGAJET_SUPER_WGS", "240", 0);
        } else if (v12_serve_spec()) {
            if (err) *err = "generic Ornith: speculative serving is unsupported";
            return false;
        }

        OrnithGpuOptions opt;
        opt.hsaco_dir = spec.hsaco_dir;
        opt.tp = spec.tp;
        opt.tp_shard_json = spec.tp_shards; opt.tp_collective_json = spec.tp_plan;
        opt.max_ctx = std::max<uint32_t>(spec.context, 256u);
        opt.run_bench = false;
        opt.run_parity = false;
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
            if (!stop_) {
                start_err_ = m.error.empty() ? "Ornith GPU engine stopped unexpectedly" : m.error;
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

    const std::string& description() const { return description_; }
    ModelInfo model_info() const override { return info_; }
    const BpeTokenizer& tokenizer() const override { return legacy_tok_; }
    // The live runner decodes greedily (temperature/top_p/top_k are not implemented): `vegajet serve` turns that into
    // HTTP 400 for sampling parameters that differ from the greedy defaults instead of ignoring them.
    bool supports_sampling() const override { return false; }

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
        if (failed_) throw guard::RunFailure(start_err_);
        if (stop_ || !ready_) throw guard::RunFailure("Ornith GPU engine is not running");
        req_prompt_ = prompt_tokens;
        req_max_tokens_ = params.max_tokens > 0 ? params.max_tokens : 64;
        req_cb_ = on_token;
        req_done_ = false;
        has_req_ = true;
        cv_req_.notify_one();
        cv_done_.wait(lk, [this] { return req_done_ || stop_ || failed_; });
        if (failed_) throw guard::RunFailure(start_err_);
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
