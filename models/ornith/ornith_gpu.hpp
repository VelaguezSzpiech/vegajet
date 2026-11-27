// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Bare-metal 2x MI50 (gfx906) GPU Executor for Ornith-122B (qwen35moe).
// Uses hardened vegajet::kfd::Runtime, Roofline Knapsack + TCC L2 deswizzling,
// exact lossless IQ2_S/IQ3_XXS routed MoE experts, C=16 WY/Neumann GDN prefill,
// 6-slot ReplaySSM GDN decode, and Push-Only Dual-BAR0 P2P AllReduce.
#pragma once

#include "core/gguf_reader.hpp"
#include "core/kfd_runtime.hpp"
#include "core/quant_repacker.hpp"
#include "core/tokenizer.hpp"
#include "models/ornith/ornith_model.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace vegajet::ornith {

struct OrnithGpuOptions {
    std::string hsaco_dir;
    std::string draft_gguf;
    std::string mask_embedding_path;       // DFlash2: BF16 [3072] mask_embedding.safetensors (auto-detected next to draft_gguf)
    uint32_t    dflash_k        = 7;       // DFlash2 draft tokens per cycle (1..15; verify batch M = dflash_k + 1 <= 16)
    float       dflash_p_min    = 0.55f;   // DFlash2 selector confidence cutoff (live sweep optimum; 0 = disabled)
    uint32_t    max_ctx         = 8192;
    uint32_t    prefill_chunk   = 1024;
    bool        use_suffix_spec = false;
    bool        use_wy_prefill  = true;
    bool        run_bench       = false;
    bool        run_parity      = true;
    int         decode_tokens   = 32;
    bool        fuse_moe_router = true;    // M=1: one fused router+top8+shared Gate/Up+SwiGLU dispatch per layer (4 -> 1)
    bool        gdn_splitv      = true;    // M=1: Split-V GDN recurrence (128 WGs/GPU, 44 VGPRs) instead of the 32-WG kernel
    bool        gdn_splitv8     = false;   // M=1: select Split-V=8 GDN ConvRec, two ordered dispatches (256 WGs each); off = current path
    bool        fuse_moe_back   = true;    // M=1: routed Gate/Up + shared Down + routed Down as one work-stealing dispatch (3 -> 1)
    uint32_t    moe_back_wgs    = 224;     // resident WGs for M=1 MoE back (224 = exact sum16 s224; 240 = experimental s240)
    uint32_t    gu_dot8_layers  = 0;       // promote layers [0..N) Gate+Up IQ2_S -> exact VJ_Q4_DOT8 (for M<16 decode/verify A/B)
    bool        profile_stages  = false;   // bench: in-situ per-stage decode cost (repeats one idempotent stage 3x)
    std::vector<int32_t> dump_prompt_ids;  // prompt for --dump-hidden (empty: built-in code prompt)
    std::string dump_hidden_path;          // bench: write per-token residual-stream taps (49 states) and exit
    uint32_t    suffix_max_draft = 11;     // longest suffix-chain draft per cycle (verify batch = this + 1, max 12)
    bool        spec_force_reject = false; // test hook: corrupt one suffix draft per cycle (exercises partial commits)
    bool        spec_gdn        = false;   // M <= 12 verify: single-write Split-V GDN + recompute-on-reject (2-slot ring, ~783 MiB/GPU saved)
    bool        treewy         = false;   // opt-in: route the spec-GDN verify/commit through the sealed tree-WY pair (VEGAJET_TREEWY=1); sequential fallback
    bool        spec_flash      = false;   // M < 16 decode/verify: 16:1 head-packed Split-KV FlashDecoding
    bool        zero_tax_catchup = true;   // default since v0.1.1: lossless, +8-12 % Hybrid at TP=3 (VEGAJET_SPEC_CASCADE=0 / --no-zero-tax-catchup to disable). Hybrid MTP: DrafterKvSync deferred catch-up + confidence-gated suffix-first policy
    // M1 sealed-wire attention (opt-in, default off): producer-adapt the normal engine's FP32 Q and
    // 272 B active Q8_0 KV rows into the sealed Q8_0/Q4_0 Split-KV layout and dispatch the pair.
    // VEGAJET_WIRE_ATTN=1|q4 and VEGAJET_WIRE_ATTN_Q4=1 are the equivalent env switches.
    bool        sealed_wire_attn = false;
    bool        sealed_wire_q4 = false;
    bool        sealed_wire_strict = false;   // A/B guard: abort if a requested step falls back
    bool        sealed_wire_exercise = false; // tests/short-context A/B: relax the split floor
    uint32_t    sealed_wire_max_splits = 0;   // 0 = launch-header cap (64)
    uint64_t    sealed_wire_scratch_budget = 0; // 0 = unbounded (still bounded by the arena allocation)
    uint32_t    fr_spec_rows    = 0;       // FR-Spec draft LM-head shortlist size (0=off/default, 32768 or 65536, multiple of 512; verify stays 100% full-vocab)
    uint32_t    mtp_k           = 4;       // blk.48 MTP draft steps per cycle (1..6)
    bool        gu_grp_verify   = false;   // M >= 8 verify: use expert-grouped Gate+Up instead of per-token Gate+Up
    uint32_t    down_grp_prefill = 3;      // routed Down for M >= 16: 0 per-token, 1 grouped exact, 2 grouped fast, 3 LDS-tiled 128x16 f32 (default lossless)
    uint32_t    down_grp_verify  = 0;      // same for 2 <= M < 16 (speculative verify batches)
    bool        pf_fa2          = true;    // M >= 16: head-packed FlashAttention-2 tile kernel (default lossless)
    bool        pf_fa2_fp16     = true;    // with --pf-fa2: use FP16-Q v_dot2_f32_f16 variant (default lossless)
    uint32_t    pf_gemm         = 128;     // M >= 16 dense W4A8 GEMM: 0 legacy, 64 = 64x64 tile, 128 = 64x128 tile (default lossless)
    uint32_t    pf_gdn          = 1;       // M >= 16 GDN: 0 legacy WY (32 WGs), 1 Split-V batch (128 WGs, default lossless), 2 split WY
    bool        pf_gu_tiled     = true;    // M >= 16 routed Gate+Up: LDS-tiled kernel with a work queue (default lossless)
    uint32_t    pf_moe_iq_lds   = 0;       // 0 = off; 1 = routed Gate+Up through the single 64x64x64
                                           //   dequant-once-to-LDS work item (64 rows x 64 tokens x K64);
                                           //   2 = also route the last MoE layer Down through its IQ3_XXS twin
    uint32_t    pf_moe_iq_lds32 = 0;       // diagnostic alternative to pf_moe_iq_lds: 1 = 32-row x 64-token
                                           //   tile (two values map to a 64-row block), 2 = also 32-row Down
    uint32_t    pf_moe_iq_lds_layer = 0xFFFFFFFFu;  // 0..63 = apply the IQ-LDS tiles to that MoE layer only;
                                                   //   0xFFFFFFFF = every MoE layer (one-layer A/B switch)
    bool        prefill_ar_int8 = true;    // M >= 16 prefill-only block-32 INT8 AllReduce (decode M < 16 ALWAYS stays FP16!)
    bool        ab_no_prepush   = false;   // A/B diagnostic: disable epilogue BAR0 prepush on M=1/verify projections
    bool        prefill_dot8    = false;   // M >= 64 opt-in W4A4 v_dot8_i32_i4 prefill (gated on held-out KLD <= 0.045)
    uint32_t    dot8_scope      = 0;       // 0=all (dense+MoE GU), 1=dense-only, 2=MoE-GU-only
    float       dot8_clip_alpha = 1.0f;    // symmetric INT4 activation clip ratio (0.5..1.0)
    bool        dot8_fwht       = false;   // with --prefill-dot8: block-64 orthonormal FWHT tile on the 4 dense
                                           // projections (rotated weight copies + rotated INT4 activations)
    uint32_t    dot8_fwht_layers = 0;      // with --dot8-fwht: 0 = all 48 layers, N = rotate layers 0..N-1 only
                                           // (N=1 isolates one layer's contribution in an A/B)
    std::string dump_logits_path;          // bench: dump full-vocab FP32 logits on held-out prompts (for KLD gate) and exit
    std::string dump_tokens_path;          // bench: write generated token ids (tools/token_compare.py diffs two dumps)
    std::string gdn_state_dump;            // bench: two-chunk prompt + continuation; dump the carried GDN state at both chunk boundaries
    bool        prefill_seq     = false;   // with gdn_state_dump: feed the prompt one token at a time (sequential reference)
    bool        pp_stages       = false;   // bench: per-stage, per-rank wall time of one uncached pp2048 request
    bool        ar_timeline     = false;   // bench: per-AllReduce device timestamps through real decode tokens (both ranks)
    bool        dot8_ab         = false;   // bench: MoE Down A/B (IQ3_XXS vs exact DOT8 repack) on the last layer, then exit
    bool        ar_wire_int8    = false;   // decode/prefill AllReduce wire: block-32 INT8 + FP32 scales instead of FP16
    bool        interactive     = false;   // bench: warm stdin/FIFO command loop (zero 42 GiB weight reload between runs/reloads)
    std::string interactive_fifo;          // optional FIFO path for --interactive (default: stdin)
    std::string sweep_spec;                // bench: ';'-delimited multi-variant A/B sweep against resident VRAM arena
    std::string dump_golden_layers_dir;    // capture 1 GDN (L0) + 1 GQA (L3) golden fixture for vegajet_layer_test
    // Optional live-serving hook (used by `vegajet serve <model-file>` on GPU builds):
    // when set, run_ornith_gpu loads weights once, invokes serve_hook(gen_fn, engine_desc), and cleans up on return.
    using ServeTokenCb = std::function<bool(int32_t tok_id, uint32_t prompt_tokens, uint32_t completion_tokens,
                                            double prefill_ms, double decode_ms)>;
    using ServeGenerateFn = std::function<void(const std::vector<int32_t>& prompt_ids, int32_t max_tokens,
                                               const ServeTokenCb& on_tok)>;
    std::function<void(const ServeGenerateFn& gen, const std::string& engine_desc)> serve_hook;
    // L6: published on the GPU owner thread before serve_hook starts its mailbox.
    using ServeStepFn = std::function<std::vector<int32_t>(const std::vector<uint32_t>&,
        const std::vector<uint32_t>&, const std::vector<int32_t>&, const std::vector<bool>&)>;
    using ServePrefillFn = std::function<int32_t(uint32_t,uint32_t,const std::vector<int32_t>&,bool,bool)>;
    std::function<void(ServeStepFn,ServePrefillFn)> serve_step_ready;
};

struct OrnithBenchMetrics {
    bool     ok = false;
    double   load_sec = 0.0;
    double   vram_gib_gpu0 = 0.0;
    double   vram_gib_gpu1 = 0.0;
    std::vector<double> vram_gib_per_gpu;
    double   tg128_raw_tps = 0.0;
    double   tg128_suffix_tps = 0.0;
    double   tg512_suffix_tps = 0.0;
    double   pp512_tps = 0.0;
    double   pp2048_tps = 0.0;
    double   eff_hbm2_gbs = 0.0;
    uint32_t dispatches_per_tok = 0;
    uint32_t suffix_exact_matches = 0;
    uint32_t suffix_total_tokens = 0;
    uint32_t knapsack_upgraded = 0;
    uint32_t knapsack_rejected = 0;
    std::string parity_simple_text;
    std::string parity_code_text;
    std::string error;
};

// Executes Ornith-122B live on 2x MI50 via bare-metal /dev/kfd.
// Under VEGAJET_CPU_ONLY=1, returns a clean descriptive error without opening /dev/kfd.
OrnithBenchMetrics run_ornith_gpu(const std::string& gguf_path,
                                  const std::string& prompt_text,
                                  const OrnithGpuOptions& opt);

// `vegajet bench --bench-ar`: Push-Only Dual-BAR0 AllReduce microbench (no model load). Runs both wire formats,
// verifies both ranks' results bit-exactly against a host reference (verify_errs) and times 96-AR batches.
// Returns 0 when verify_errs == 0 for both wires.
int run_ornith_ar_bench(const std::string& hsaco_dir);

// `vegajet bench --bench-gemv`: On-GPU s_memrealtime() root-cause A/B on [3072, 4096] Q4A8 GEMVs
// (baseline 384 WG + prepush vs no-prepush vs odd-256B dm vs Split-K S=2/4 + stagger) + Split-KV FlashDecoding check.
int run_ornith_gemv_ab_bench(const std::string& hsaco_dir);

// Registers the live 2x MI50 Ornith streaming engine for `vegajet serve <model-file>` (no-op on CPU-only builds).
void register_ornith_serve_engine();

} // namespace vegajet::ornith
