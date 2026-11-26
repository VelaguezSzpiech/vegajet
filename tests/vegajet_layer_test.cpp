// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Pillar 3: 1-Layer Golden Tensor Test Harness (`vegajet_layer_test`, <2s GPU kernel verification).
// Loads a 2-layer golden fixture (Layer 0 GDN+MoE + Layer 3 GQA, captured via `--dump-golden-layers`)
// from `--golden-dir <dir>` or `$VEGAJET_GOLDEN_DIR`, executes selected GPU kernels for requested M values,
// checks bit/cosine parity against golden reference outputs, times via s_memrealtime() (25 MHz / 40 ns),
// reports achieved HBM2 GB/s, and verifies TMem < 65 C + 6-block RAS ECC delta (0 UE / 0 DE).
// Exits with code 77 (CTest SKIP) on CPU-only builds or when VEGAJET_GOLDEN_DIR is unset/absent.
#include "core/kfd_runtime.hpp"
#include "core/vj_check.hpp"
#include "core/warm_bench.hpp"
#include "ops/kernels/vj_ornith_super_desc.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kSkipCode = 77;
constexpr uint32_t kGdnRingSlots = 17u;

void usage() {
    std::printf("Usage: vegajet_layer_test [--golden-dir DIR] [--hsaco-dir DIR]\n"
                "                          [--stage gdn|gqa|moe|ar|prefill|all]\n"
                "                          [--m 1,5,16,128,512] [--iters N] [--json]\n"
                "  Defaults: DIR=$VEGAJET_GOLDEN_DIR, --stage all, --m 1,5,16,128,512, --iters 5.\n"
                "  Exits 77 (CTest skip) if built CPU-only or if golden fixture directory is not set.\n");
}

#ifndef VEGAJET_CPU_ONLY

bool read_bin_file(const std::string& path, std::vector<uint8_t>& out, std::string* err) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) { if (err) *err = "cannot open " + path; return false; }
    const std::streamsize sz = ifs.tellg();
    if (sz < 0) { if (err) *err = "tellg failed on " + path; return false; }
    out.resize(static_cast<size_t>(sz));
    ifs.seekg(0);
    if (sz > 0) ifs.read(reinterpret_cast<char*>(out.data()), sz);
    if (!ifs.good()) { if (err) *err = "short read on " + path; return false; }
    return true;
}

bool upload_file(vegajet::kfd::Runtime& rt, size_t gpu, const std::string& path, uint64_t& out_va, uint64_t& out_bytes, uint64_t align, std::string* err) {
    std::vector<uint8_t> buf;
    if (!read_bin_file(path, buf, err)) return false;
    out_bytes = buf.size();
    out_va = rt.vram_arena(gpu).alloc(std::max<uint64_t>(out_bytes, 64ULL), align);
    if (!out_va) { if (err) *err = "VRAM OOM allocating " + path; return false; }
    if (out_bytes > 0 && !rt.upload_to_vram(gpu, out_va, buf.data(), out_bytes)) {
        if (err) *err = "upload_to_vram failed for " + path;
        return false;
    }
    return true;
}

void read_hw_telemetry(double& max_tmem_c, uint64_t& ue_tot, uint64_t& de_tot) {
    max_tmem_c = 0.0;
    ue_tot = 0;
    de_tot = 0;
    static const char* const kBlocks[] = {"umc", "gfx", "hdp", "mmhub", "pcie_bif", "sdma"};
    for (int card = 0; card < 16; ++card) {
        std::string cdir = "/sys/class/drm/card" + std::to_string(card) + "/device/";
        for (int h = 0; h < 16; ++h) {
            std::ifstream ft(cdir + "hwmon/hwmon" + std::to_string(h) + "/temp3_input");
            double tm = 0.0;
            if (ft >> tm) { max_tmem_c = std::max(max_tmem_c, tm * 1e-3); break; }
        }
        for (const char* blk : kBlocks) {
            std::ifstream fr(cdir + "ras/" + blk + "_err_count");
            if (!fr.is_open()) continue;
            std::string tok;
            while (fr >> tok) {
                if (tok == "ue:") { uint64_t v = 0; if (fr >> v) ue_tot += v; }
                else if (tok == "de:") { uint64_t v = 0; if (fr >> v) de_tot += v; }
            }
        }
    }
}

struct ParityStats {
    double max_abs_err = 0.0;
    double cosine_sim  = 1.0;
    uint64_t n_elems   = 0;
    bool   nan_found   = false;
};

ParityStats compare_f32(const float* a, const float* b, size_t n) {
    ParityStats s;
    s.n_elems = n;
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double va = a[i], vb = b[i];
        if (!std::isfinite(va) || !std::isfinite(vb)) { s.nan_found = true; continue; }
        s.max_abs_err = std::max(s.max_abs_err, std::fabs(va - vb));
        dot += va * vb; na += va * va; nb += vb * vb;
    }
    const double denom = std::sqrt(na) * std::sqrt(nb);
    s.cosine_sim = (denom > 1e-30) ? (dot / denom) : (s.max_abs_err == 0.0 ? 1.0 : 0.0);
    return s;
}

struct StageResult {
    std::string stage;
    uint32_t M = 0;
    bool pass = false;
    double us = 0.0;
    double gb_s = 0.0;
    double max_err = 0.0;
    double cosine = 1.0;
    std::string detail;
};

#endif // !VEGAJET_CPU_ONLY

} // namespace

int main(int argc, char** argv) {
    std::string golden_dir, hsaco_dir, stage_arg = "all", m_arg = "1,5,16,128,512";
    int iters = 5;
    bool json_only = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (a == "--help" || a == "-h") { usage(); return 0; }
        else if (a == "--golden-dir") golden_dir = next();
        else if (a == "--hsaco-dir") hsaco_dir = next();
        else if (a == "--stage") stage_arg = next();
        else if (a == "--m") m_arg = next();
        else if (a == "--iters") iters = std::max(1, std::atoi(next().c_str()));
        else if (a == "--json") json_only = true;
        else { std::fprintf(stderr, "vegajet_layer_test: unknown option %s\n", a.c_str()); return 2; }
    }
    if (golden_dir.empty()) {
        if (const char* env = std::getenv("VEGAJET_GOLDEN_DIR"); env && env[0]) golden_dir = env;
    }

#ifdef VEGAJET_CPU_ONLY
    (void)iters;
    (void)json_only;
    std::printf("vegajet_layer_test: SKIP (VEGAJET_CPU_ONLY=ON)\n");
    return kSkipCode;
#else
    namespace fs = std::filesystem;
    if (golden_dir.empty() || !fs::exists(fs::path(golden_dir) / "manifest.txt")) {
        std::printf("vegajet_layer_test: SKIP (no golden fixture; set VEGAJET_GOLDEN_DIR or --golden-dir)\n");
        return kSkipCode;
    }

    // Parse manifest.txt
    std::map<std::string, std::string> mf;
    {
        std::ifstream ifs(fs::path(golden_dir) / "manifest.txt");
        std::string line;
        while (std::getline(ifs, line)) {
            const size_t eq = line.find('=');
            if (eq != std::string::npos) mf[line.substr(0, eq)] = line.substr(eq + 1);
        }
    }
    auto mf_u32 = [&](const char* k, uint32_t def = 0u) -> uint32_t {
        auto it = mf.find(k);
        return it == mf.end() ? def : static_cast<uint32_t>(std::strtoul(it->second.c_str(), nullptr, 10));
    };
    auto mf_f32 = [&](const char* k, float def = 0.0f) -> float {
        auto it = mf.find(k);
        return it == mf.end() ? def : std::strtof(it->second.c_str(), nullptr);
    };

    const uint32_t D              = mf_u32("D", 3072u);
    const uint32_t local_ff       = mf_u32("local_ff", 512u);
    const uint32_t pos0           = mf_u32("pos0", 64u);
    const float    eps            = mf_f32("eps", 1e-6f);
    const float    rope_base      = mf_f32("rope_base", 1e7f);
    const uint32_t gdn_qkv_rows   = mf_u32("gdn_qkv_rows", 10304u);
    const uint32_t gdn_qkv_cols   = mf_u32("gdn_qkv_cols", 3072u);
    const uint32_t gdn_qkv_stride = mf_u32("gdn_qkv_stride");
    const uint32_t ssm_out_rows   = mf_u32("ssm_out_rows", 3072u);
    const uint32_t ssm_out_cols   = mf_u32("ssm_out_cols", 4096u);
    const uint32_t ssm_out_stride = mf_u32("ssm_out_stride");
    const uint32_t gqa_qkv_rows   = mf_u32("gqa_qkv_rows", 8704u);
    const uint32_t gqa_qkv_cols   = mf_u32("gqa_qkv_cols", 3072u);
    const uint32_t gqa_qkv_stride = mf_u32("gqa_qkv_stride");
    const uint32_t gqa_o_rows     = mf_u32("gqa_o_rows", 3072u);
    const uint32_t gqa_o_cols     = mf_u32("gqa_o_cols", 4096u);
    const uint32_t gqa_o_stride   = mf_u32("gqa_o_stride");
    const uint32_t sh_gu_rows     = mf_u32("sh_gu_rows", 1024u);
    const uint32_t sh_gu_cols     = mf_u32("sh_gu_cols", 3072u);
    const uint32_t sh_gu_stride   = mf_u32("sh_gu_stride");
    const uint32_t sh_dn_rows     = mf_u32("sh_dn_rows", 3072u);
    const uint32_t sh_dn_cols     = mf_u32("sh_dn_cols", 512u);
    const uint32_t sh_dn_stride   = mf_u32("sh_dn_stride");
    const uint32_t dn_d8_w_stride = mf_u32("dn_d8_w_stride");
    const uint32_t dn_d8_s_stride = mf_u32("dn_d8_s_stride");
    const uint32_t head_dim       = mf_u32("gqa_head_dim", 256u);
    const uint32_t n_q_local      = gqa_o_cols / std::max<uint32_t>(1u, head_dim);
    const uint32_t n_kv_local     = mf_u32("gqa_n_kv_local", 1u);
    const uint32_t rot_dim        = mf_u32("gqa_rot_dim", 64u);

    std::vector<uint32_t> m_list;
    {
        std::stringstream ss(m_arg);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            if (!tok.empty()) m_list.push_back(static_cast<uint32_t>(std::strtoul(tok.c_str(), nullptr, 10)));
        }
    }
    if (m_list.empty()) m_list = {1u, 5u, 16u, 128u, 512u};

    const bool do_gdn = (stage_arg == "all" || stage_arg == "gdn" || stage_arg == "prefill" || stage_arg == "super_m1");
    const bool do_gqa = (stage_arg == "all" || stage_arg == "gqa" || stage_arg == "prefill" || stage_arg == "super_m1");
    const bool do_moe = (stage_arg == "all" || stage_arg == "moe" || stage_arg == "prefill" || stage_arg == "super_m1");
    const bool do_ar  = (stage_arg == "all" || stage_arg == "ar"  || stage_arg == "prefill");
    const bool do_super = (stage_arg == "super_m1");
    const bool prefill_only = (stage_arg == "prefill");

    double tmem0 = 0.0, tmem1 = 0.0;
    uint64_t ue0 = 0, de0 = 0, ue1 = 0, de1 = 0;
    read_hw_telemetry(tmem0, ue0, de0);

    const auto t_load0 = std::chrono::steady_clock::now();
    vegajet::kfd::Runtime rt;
    std::string err;
    if (!rt.init(2.0, 32, &err) || !rt.load_hsaco_dir(vegajet::kfd::resolve_hsaco_dir(hsaco_dir), &err)) {
        std::fprintf(stderr, "vegajet_layer_test: runtime init failed: %s\n", err.c_str());
        return 1;
    }

    auto need_k = [&](size_t g, const char* nm) -> const vegajet::kfd::LoadedKernel* {
        const auto* lk = rt.get_kernel(g, nm);
        if (!lk) { std::fprintf(stderr, "vegajet_layer_test: missing kernel %s\n", nm); std::exit(1); }
        return lk;
    };
    const auto* k_clk      = need_k(0, "kfd_stamp_clock_u64");
    const auto* k_memset   = need_k(0, "kfd_memset_u32");
    const auto* k_w32      = need_k(0, "ornith_gemv_q4a8_m1_w32_kernel");
    const auto* k_m8       = need_k(0, "ornith_gemm_q4a8_m8_kernel");
    const auto* k_m16      = need_k(0, "ornith_gemm_q4a8_m16_l2_kernel");
    const auto* k_pf_g128  = need_k(0, "vj_prefill_gemm_q4a8_64x128_kernel");
    const auto* k_pf_g64   = need_k(0, "vj_prefill_gemm_q4a8_64x64_kernel");

    const std::string wdir = golden_dir + "/weights";
    uint64_t dum = 0;
    // Load only the weights needed for requested stages!
    uint64_t l0_gdn_qkv_qs = 0, l0_gdn_qkv_dm = 0, l0_ssm_out_qs = 0, l0_ssm_out_dm = 0;
    uint64_t l0_ssm_cw = 0, l0_ssm_a = 0, l0_ssm_dt = 0, l0_ssm_nw = 0;
    uint64_t l0_gdn_bytes = 0;
    if (do_gdn) {
        uint64_t b0 = 0, b1 = 0, b2 = 0, b3 = 0;
        if (!upload_file(rt, 0, wdir + "/l0_gdn_qkv_qs.bin", l0_gdn_qkv_qs, b0, 4096, &err) ||
            !upload_file(rt, 0, wdir + "/l0_gdn_qkv_dm.bin", l0_gdn_qkv_dm, b1, 256, &err) ||
            !upload_file(rt, 0, wdir + "/l0_ssm_out_qs.bin", l0_ssm_out_qs, b2, 4096, &err) ||
            !upload_file(rt, 0, wdir + "/l0_ssm_out_dm.bin", l0_ssm_out_dm, b3, 256, &err) ||
            !upload_file(rt, 0, wdir + "/l0_ssm_conv_w.bin", l0_ssm_cw, dum, 256, &err) ||
            !upload_file(rt, 0, wdir + "/l0_ssm_a.bin",      l0_ssm_a,  dum, 256, &err) ||
            !upload_file(rt, 0, wdir + "/l0_ssm_dt.bin",     l0_ssm_dt, dum, 256, &err) ||
            !upload_file(rt, 0, wdir + "/l0_ssm_norm.bin",   l0_ssm_nw, dum, 256, &err)) {
            std::fprintf(stderr, "vegajet_layer_test: GDN weight load failed: %s\n", err.c_str());
            return 1;
        }
        l0_gdn_bytes = b0 + b1 + b2 + b3 + 2ULL * 32ULL * 128ULL * 128ULL * 4ULL;
    }

    uint64_t l3_qkv_qs = 0, l3_qkv_dm = 0, l3_o_qs = 0, l3_o_dm = 0, l3_qn = 0, l3_kn = 0;
    uint64_t l3_gqa_bytes = 0;
    if (do_gqa) {
        uint64_t b0 = 0, b1 = 0, b2 = 0, b3 = 0;
        if (!upload_file(rt, 0, wdir + "/l3_gqa_qkv_qs.bin", l3_qkv_qs, b0, 4096, &err) ||
            !upload_file(rt, 0, wdir + "/l3_gqa_qkv_dm.bin", l3_qkv_dm, b1, 256, &err) ||
            !upload_file(rt, 0, wdir + "/l3_gqa_o_qs.bin",   l3_o_qs,   b2, 4096, &err) ||
            !upload_file(rt, 0, wdir + "/l3_gqa_o_dm.bin",   l3_o_dm,   b3, 256, &err) ||
            !upload_file(rt, 0, wdir + "/l3_q_norm.bin",     l3_qn,     dum, 256, &err) ||
            !upload_file(rt, 0, wdir + "/l3_k_norm.bin",     l3_kn,     dum, 256, &err)) {
            std::fprintf(stderr, "vegajet_layer_test: GQA weight load failed: %s\n", err.c_str());
            return 1;
        }
        l3_gqa_bytes = b0 + b1 + b2 + b3;
    }

    uint64_t l0_rtr = 0, l0_sh_g = 0, l0_sh_gu_qs = 0, l0_sh_gu_dm = 0, l0_sh_dn_qs = 0, l0_sh_dn_dm = 0;
    uint64_t l0_gu_iq2s = 0, l0_dn_d8_w = 0, l0_dn_d8_s = 0;
    if (do_moe) {
        if (!upload_file(rt, 0, wdir + "/l0_router_f16.bin", l0_rtr,      dum, 4096, &err) ||
            !upload_file(rt, 0, wdir + "/l0_sh_gate.bin",    l0_sh_g,     dum, 256, &err) ||
            !upload_file(rt, 0, wdir + "/l0_sh_gu_qs.bin",   l0_sh_gu_qs, dum, 4096, &err) ||
            !upload_file(rt, 0, wdir + "/l0_sh_gu_dm.bin",   l0_sh_gu_dm, dum, 256, &err) ||
            !upload_file(rt, 0, wdir + "/l0_sh_dn_qs.bin",   l0_sh_dn_qs, dum, 4096, &err) ||
            !upload_file(rt, 0, wdir + "/l0_sh_dn_dm.bin",   l0_sh_dn_dm, dum, 256, &err) ||
            !upload_file(rt, 0, wdir + "/l0_gu_iq2s.bin",    l0_gu_iq2s,  dum, 4096, &err) ||
            !upload_file(rt, 0, wdir + "/l0_dn_d8_w.bin",    l0_dn_d8_w,  dum, 4096, &err) ||
            !upload_file(rt, 0, wdir + "/l0_dn_d8_s.bin",    l0_dn_d8_s,  dum, 4096, &err)) {
            std::fprintf(stderr, "vegajet_layer_test: MoE weight load failed: %s\n", err.c_str());
            return 1;
        }
    }

    // Allocate reusable GPU 0 scratch buffers (sized for max M = 512)
    auto& a0 = rt.vram_arena(0);
    const uint32_t max_M = 512u;
    const uint64_t clk_va      = a0.alloc(64u, 256);
    const uint64_t xnorm_va    = a0.alloc(static_cast<uint64_t>(max_M) * D * 4ULL, 4096);
    const uint64_t xq8_va      = a0.alloc(static_cast<uint64_t>(max_M) * D, 4096);
    const uint64_t xds_va      = a0.alloc(static_cast<uint64_t>(max_M) * (D / 32u) * 8ULL, 4096);
    const uint64_t partial_va  = a0.alloc(static_cast<uint64_t>(max_M) * D * 4ULL, 4096);
    const uint64_t gdn_qkv_va  = a0.alloc(static_cast<uint64_t>(max_M) * 10304ULL * 4ULL, 4096);
    const uint64_t gdn_conv_va = a0.alloc(static_cast<uint64_t>(max_M) * 6144ULL * 4ULL, 4096);
    const uint64_t gdn_dec_va  = a0.alloc(static_cast<uint64_t>(max_M) * 32ULL * 4ULL, 256);
    const uint64_t gdn_bsig_va = a0.alloc(static_cast<uint64_t>(max_M) * 32ULL * 4ULL, 256);
    const uint64_t mid_q8_va   = a0.alloc(static_cast<uint64_t>(max_M) * 4096ULL, 4096);
    const uint64_t mid_ds_va   = a0.alloc(static_cast<uint64_t>(max_M) * 128ULL * 8ULL, 4096);
    const uint64_t conv_ring_va = a0.alloc(kGdnRingSlots * 3ULL * 6144ULL * 4ULL, 4096);
    const uint64_t ssm_ring_va  = a0.alloc(kGdnRingSlots * 32ULL * 128ULL * 128ULL * 4ULL, 4096);
    const uint64_t gdn_bar_va  = a0.alloc(8192u, 256);
    const uint64_t gdn_sq_va   = a0.alloc(32u * 64u * 4u, 256);
    const uint64_t pf_fo_va    = a0.alloc(static_cast<uint64_t>(max_M) * 32ULL * 128ULL * 4ULL, 4096);
    const uint64_t pf_sq_va    = a0.alloc(static_cast<uint64_t>(max_M) * 32ULL * 64ULL * 4ULL, 4096);
    const uint64_t pf_bar_va   = a0.alloc(256u, 256);
    const uint64_t attn_qkv_va = a0.alloc(static_cast<uint64_t>(max_M) * 8704ULL * 4ULL, 4096);
    const uint64_t q_buf_va    = a0.alloc(static_cast<uint64_t>(max_M) * 4096ULL * 4ULL, 4096);
    const uint64_t gate_buf_va = a0.alloc(static_cast<uint64_t>(max_M) * 4096ULL * 4ULL, 4096);
    const uint64_t k_cache_va  = a0.alloc(static_cast<uint64_t>(pos0 + max_M + 64u) * 272ULL, 4096);
    const uint64_t v_cache_va  = a0.alloc(static_cast<uint64_t>(pos0 + max_M + 64u) * 272ULL, 4096);
    const uint64_t rtr_log_va  = a0.alloc(static_cast<uint64_t>(max_M) * 256ULL * 4ULL, 4096);
    const uint64_t top_ids_va  = a0.alloc(static_cast<uint64_t>(max_M) * 8ULL * 4ULL, 256);
    const uint64_t top_w_va    = a0.alloc(static_cast<uint64_t>(max_M) * 8ULL * 4ULL, 256);
    const uint64_t sh_sc_va    = a0.alloc(static_cast<uint64_t>(max_M) * 4ULL, 256);
    const uint64_t sh_gu_va    = a0.alloc(static_cast<uint64_t>(max_M) * 2ULL * local_ff * 4ULL, 4096);
    const uint64_t sh_q8_va    = a0.alloc(static_cast<uint64_t>(max_M) * local_ff, 4096);
    const uint64_t sh_ds_va    = a0.alloc(static_cast<uint64_t>(max_M) * (local_ff / 32u) * 8ULL, 256);
    const uint64_t sh_dn_va    = a0.alloc(static_cast<uint64_t>(max_M) * D * 4ULL, 4096);
    const uint64_t moe_q8_va   = a0.alloc(static_cast<uint64_t>(max_M) * 8ULL * local_ff, 4096);
    const uint64_t moe_d_va    = a0.alloc(static_cast<uint64_t>(max_M) * 8ULL * (local_ff / 32u) * 4ULL, 4096);
    const uint64_t moe_bq_va   = a0.alloc(256u, 256);
    const uint64_t pf_queue_va = a0.alloc(256u, 256);
    const uint64_t dgrp_hdr_va = a0.alloc(64u, 256);
    const uint64_t dgrp_cnt_va = a0.alloc(256u * 4u, 256);
    const uint64_t dgrp_off_va = a0.alloc(257u * 4u, 256);
    const uint64_t dgrp_unq_va = a0.alloc(256u * 4u, 256);
    const uint64_t dgrp_itm_va = a0.alloc(4096u * 4u, 256);
    const uint64_t dgrp_rte_va = a0.alloc(static_cast<uint64_t>(max_M) * 8ULL * 4ULL, 256);
    const uint64_t dgrp_buf_va = a0.alloc(static_cast<uint64_t>(max_M) * 8ULL * D * 4ULL + 4096ULL, 4096);
    VJ_CHECK(dgrp_buf_va != 0);

    const double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_load0).count();

    auto zero_gpu0 = [&](uint64_t va, uint64_t bytes) {
        struct MsKA { uint64_t dst; uint32_t val; uint32_t pad; uint64_t n; };
        const MsKA ka{va, 0u, 0u, (bytes + 3u) / 4u};
        VJ_CHECK(rt.enqueue_kernel(0, *k_memset, static_cast<uint32_t>(((ka.n + 255u) / 256u) * 256u), 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
    };

    auto enq_q4 = [&](uint64_t wq, uint64_t wdm, uint32_t rows, uint32_t cols, uint32_t stride,
                      uint64_t xq, uint64_t xds, uint64_t y, uint32_t M) {
        if (M == 1u) {
            struct M1KA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, stride, dm_st; uint64_t ptx; };
            const M1KA ka{wq, wdm, xq, xds, y, rows, cols, stride, cols >> 5, 0ULL};
            VJ_CHECK(rt.enqueue_kernel(0, *k_w32, ((rows + 7u) / 8u) * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
        } else if (M <= 8u) {
            struct M8KA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, M, stride; uint64_t ptx; };
            const M8KA ka{wq, wdm, xq, xds, y, rows, cols, M, stride, 0ULL};
            VJ_CHECK(rt.enqueue_kernel(0, *k_m8, ((rows + 7u) / 8u) * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
        } else if (M <= 16u) {
            struct M16KA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, M, stride; uint64_t ptx; };
            const M16KA ka{wq, wdm, xq, xds, y, rows, cols, M, stride, 0ULL};
            VJ_CHECK(rt.enqueue_kernel(0, *k_m16, ((rows + 7u) / 8u) * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
        } else {
            struct PfKA { uint64_t wq, wdm, xq, xds, y; uint32_t rows, cols, M, stride; };
            const PfKA ka{wq, wdm, xq, xds, y, rows, cols, M, stride};
            if (rows <= 1024u || M < 256u) {
                VJ_CHECK(rt.enqueue_kernel(0, *k_pf_g64, ((rows + 63u) / 64u) * 256u, (M + 63u) / 64u, 1, 256, 1, 1, &ka, sizeof(ka), true));
            } else {
                VJ_CHECK(rt.enqueue_kernel(0, *k_pf_g128, ((rows + 63u) / 64u) * 512u, (M + 127u) / 128u, 1, 512, 1, 1, &ka, sizeof(ka), true));
            }
        }
    };

    struct ClkKA { uint64_t dst; uint32_t idx; uint32_t pad; };
    auto time_gpu0 = [&](auto&& prep_fn, auto&& body_fn) -> double {
        double best_us = 1e30;
        for (int it = 0; it < iters + 1; ++it) {
            prep_fn();
            const ClkKA c0{clk_va, 0u, 0u}, c1{clk_va, 1u, 0u};
            VJ_CHECK(rt.enqueue_kernel(0, *k_clk, 64u, 1, 1, 64, 1, 1, &c0, sizeof(c0), true));
            body_fn();
            VJ_CHECK(rt.enqueue_kernel(0, *k_clk, 64u, 1, 1, 64, 1, 1, &c1, sizeof(c1), true));
            VJ_CHECK(rt.submit_batch(0) && rt.wait_idle(0));
            uint64_t ts[2] = {0, 0};
            VJ_CHECK(rt.download_from_vram(0, ts, clk_va, sizeof(ts)));
            const double us = static_cast<double>(ts[1] - ts[0]) * 0.04; // 25 MHz -> us
            if (it > 0) best_us = std::min(best_us, us);
        }
        return best_us;
    };

    std::vector<StageResult> results;
    uint32_t gdn_sense = 0u, moe_sense = 0u, pf_gdn_sense = 0u;

    // ---- STAGE 1: GDN (Layer 0) ----
    if (do_gdn && !do_super) {
        const auto* k_conv_m1 = need_k(0, "ornith_gdn_conv_recur_splitv_quant_m1_kernel");
        // v1.2 D6: VEGAJET_V12_GDN_SPLITV=1 runs the column-thread variant in the same harness (same args,
        // same golden), so `--stage gdn --m 1` A/Bs the two layouts on the same fixture.
        const char* ct_env = std::getenv("VEGAJET_V12_GDN_SPLITV");
        const auto* k_conv_m1_ct = (ct_env != nullptr && ct_env[0] == '1')
            ? need_k(0, "ornith_gdn_conv_recur_colthread_quant_m1_kernel") : nullptr;
        const auto* k_conv_m1_sel = (k_conv_m1_ct != nullptr) ? k_conv_m1_ct : k_conv_m1;
        const auto* k_conv    = need_k(0, "ornith_gdn_conv_l2_gate_kernel");
        const auto* k_rec     = need_k(0, "ornith_gdn_recur_quant_kernel");
        const auto* k_pf_conv = need_k(0, "vj_prefill_gdn_conv_l2_gate_kernel");
        const auto* k_pf_sv   = need_k(0, "vj_prefill_gdn_splitv_batch_kernel");
        for (uint32_t M : m_list) {
            if (prefill_only && M < 16u) continue;
            const std::string pfx = golden_dir + "/m" + std::to_string(M) + "_gdn_";
            std::vector<uint8_t> b_xq, b_xd, b_cin, b_sin, b_ref;
            if (!read_bin_file(pfx + "xq8.bin", b_xq, &err) ||
                !read_bin_file(pfx + "xds.bin", b_xd, &err) ||
                !read_bin_file(pfx + "conv_in.bin", b_cin, &err) ||
                !read_bin_file(pfx + "ssm_in.bin", b_sin, &err) ||
                !read_bin_file(pfx + "partial_out.bin", b_ref, &err)) {
                results.push_back({"gdn", M, false, 0, 0, 0, 0, err});
                continue;
            }
            VJ_CHECK(rt.upload_to_vram(0, xq8_va, b_xq.data(), b_xq.size()));
            VJ_CHECK(rt.upload_to_vram(0, xds_va, b_xd.data(), b_xd.size()));
            zero_gpu0(gdn_bar_va, 8192u);
            zero_gpu0(pf_bar_va, 256u);
            VJ_CHECK(rt.submit_batch(0) && rt.wait_idle(0));
            const bool use_ring = (M > 1u && M <= 16u);
            const double us = time_gpu0(
                [&]() {
                    VJ_CHECK(rt.upload_to_vram(0, conv_ring_va, b_cin.data(), b_cin.size()));
                    VJ_CHECK(rt.upload_to_vram(0, ssm_ring_va, b_sin.data(), b_sin.size()));
                },
                [&]() {
                    enq_q4(l0_gdn_qkv_qs, l0_gdn_qkv_dm, gdn_qkv_rows, gdn_qkv_cols, gdn_qkv_stride, xq8_va, xds_va, gdn_qkv_va, M);
                    if (M == 1u) {
                        ++gdn_sense;
                        struct CRKA { uint64_t qkv, cw, cr, dt, sa, sr, nw, oq, od, bar, sq; uint32_t nk, nv, cdim, strd, zoff, aoff, boff, cin, cout, sh, ur, sense; float eps; };
                        const CRKA ka{gdn_qkv_va, l0_ssm_cw, conv_ring_va, l0_ssm_dt, l0_ssm_a, ssm_ring_va, l0_ssm_nw,
                                      mid_q8_va, mid_ds_va, gdn_bar_va, gdn_sq_va, 8u, 32u, 6144u, 10304u, 6144u, 10240u, 10272u, 0u, 1u, 0u, 0u, gdn_sense, eps};
                        VJ_CHECK(rt.enqueue_kernel(0, *k_conv_m1_sel, 128u * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
                    } else if (M <= 16u) {
                        struct CLKA { uint64_t qkv, cw, cr, cout, dt, sa, dec, bs; uint32_t M, nk, nv, cdim, strd, aoff, boff, rh, ur; float eps; };
                        const CLKA cl{gdn_qkv_va, l0_ssm_cw, conv_ring_va, gdn_conv_va, l0_ssm_dt, l0_ssm_a, gdn_dec_va, gdn_bsig_va,
                                      M, 8u, 32u, 6144u, 10304u, 10240u, 10272u, 0u, use_ring ? 1u : 0u, eps};
                        VJ_CHECK(rt.enqueue_kernel(0, *k_conv, 49u * 128u, 1, 1, 128, 1, 1, &cl, sizeof(cl), true));
                        struct RecKA { uint64_t sring, cq, gab, dec, bs, nw, oq, od; uint32_t M, nk, nv, cdim, strd, zoff, rh, ur; float eps; };
                        const RecKA rk{ssm_ring_va, gdn_conv_va, gdn_qkv_va, gdn_dec_va, gdn_bsig_va, l0_ssm_nw,
                                       mid_q8_va, mid_ds_va, M, 8u, 32u, 6144u, 10304u, 6144u, 0u, use_ring ? 1u : 0u, eps};
                        VJ_CHECK(rt.enqueue_kernel(0, *k_rec, 32u * 1024u, 1, 1, 1024, 1, 1, &rk, sizeof(rk), true));
                    } else {
                        struct CLKA { uint64_t qkv, cw, cr, cout, dt, sa, dec, bs; uint32_t M, nk, nv, cdim, strd, aoff, boff, cin, cout_s; float eps; };
                        const CLKA cl{gdn_qkv_va, l0_ssm_cw, conv_ring_va, gdn_conv_va, l0_ssm_dt, l0_ssm_a, gdn_dec_va, gdn_bsig_va,
                                      M, 8u, 32u, 6144u, 10304u, 10240u, 10272u, 0u, 1u, eps};
                        VJ_CHECK(rt.enqueue_kernel(0, *k_pf_conv, 49u * 128u, (M + 15u) / 16u, 1, 128, 1, 1, &cl, sizeof(cl), true));
                        ++pf_gdn_sense;
                        struct SvbKA { uint64_t sring, cq, gab, dec, bs, nw, oq, od, bar, fo, sq; uint32_t M, nk, nv, cdim, strd, zoff, rh, ur, sense; float eps; uint32_t fuse; };
                        const SvbKA sk{ssm_ring_va, gdn_conv_va, gdn_qkv_va, gdn_dec_va, gdn_bsig_va, l0_ssm_nw,
                                       mid_q8_va, mid_ds_va, pf_bar_va, pf_fo_va, pf_sq_va,
                                       M, 8u, 32u, 6144u, 10304u, 6144u, 0u, 0u, pf_gdn_sense, eps, 1u};
                        VJ_CHECK(rt.enqueue_kernel(0, *k_pf_sv, 128u * 256u, 1, 1, 256, 1, 1, &sk, sizeof(sk), true));
                    }
                    enq_q4(l0_ssm_out_qs, l0_ssm_out_dm, ssm_out_rows, ssm_out_cols, ssm_out_stride, mid_q8_va, mid_ds_va, partial_va, M);
                });
            std::vector<float> got(static_cast<size_t>(M) * D);
            VJ_CHECK(rt.download_from_vram(0, got.data(), partial_va, got.size() * 4u));
            const auto ps = compare_f32(got.data(), reinterpret_cast<const float*>(b_ref.data()), got.size());
            const bool ok = !ps.nan_found && ps.max_abs_err <= 2e-4 && ps.cosine_sim >= 0.999999;
            const double gbs = (us > 0.0) ? (static_cast<double>(l0_gdn_bytes) / (us * 1e3)) : 0.0;
            results.push_back({"gdn", M, ok, us, gbs, ps.max_abs_err, ps.cosine_sim, ""});
        }
    }

    // ---- STAGE 2: GQA (Layer 3) ----
    if (do_gqa && !do_super) {
        const auto* k_prep    = need_k(0, "ornith_attn_qkv_prep_kernel");
        const auto* k_flash   = need_k(0, "ornith_attn_flash_quant_kernel");
        const auto* k_pf_fa2  = need_k(0, "vj_prefill_fa2_quant_kernel");
        for (uint32_t M : m_list) {
            if (prefill_only && M < 16u) continue;
            const std::string pfx = golden_dir + "/m" + std::to_string(M) + "_gqa_";
            std::vector<uint8_t> b_xq, b_xd, b_kc, b_vc, b_ref;
            if (!read_bin_file(pfx + "xq8.bin", b_xq, &err) ||
                !read_bin_file(pfx + "xds.bin", b_xd, &err) ||
                !read_bin_file(pfx + "k_cache.bin", b_kc, &err) ||
                !read_bin_file(pfx + "v_cache.bin", b_vc, &err) ||
                !read_bin_file(pfx + "partial_out.bin", b_ref, &err)) {
                results.push_back({"gqa", M, false, 0, 0, 0, 0, err});
                continue;
            }
            VJ_CHECK(rt.upload_to_vram(0, xq8_va, b_xq.data(), b_xq.size()));
            VJ_CHECK(rt.upload_to_vram(0, xds_va, b_xd.data(), b_xd.size()));
            const double us = time_gpu0(
                [&]() {
                    VJ_CHECK(rt.upload_to_vram(0, k_cache_va, b_kc.data(), b_kc.size()));
                    VJ_CHECK(rt.upload_to_vram(0, v_cache_va, b_vc.data(), b_vc.size()));
                },
                [&]() {
                    enq_q4(l3_qkv_qs, l3_qkv_dm, gqa_qkv_rows, gqa_qkv_cols, gqa_qkv_stride, xq8_va, xds_va, attn_qkv_va, M);
                    struct PrepKA { uint64_t qkv, qw, kw, qo, go, kc, vc; uint32_t M, pos0, nq, nkv, rdims; float rbase, eps; };
                    const PrepKA pka{attn_qkv_va, l3_qn, l3_kn, q_buf_va, gate_buf_va, k_cache_va, v_cache_va,
                                     M, pos0, n_q_local, n_kv_local, rot_dim, rope_base, eps};
                    VJ_CHECK(rt.enqueue_kernel(0, *k_prep, 17u * 256u, M, 1, 256, 1, 1, &pka, sizeof(pka), true));
                    struct FlKA { uint64_t qo, go, kc, vc, oq, od; uint32_t M, pos0, nq, nkv; };
                    const FlKA fka{q_buf_va, gate_buf_va, k_cache_va, v_cache_va, mid_q8_va, mid_ds_va,
                                   M, pos0, n_q_local, n_kv_local};
                    if (M >= 128u) {
                        VJ_CHECK(rt.enqueue_kernel(0, *k_pf_fa2, M * 256u, n_kv_local, 1, 256, 1, 1, &fka, sizeof(fka), true));
                    } else {
                        VJ_CHECK(rt.enqueue_kernel(0, *k_flash, n_q_local * 256u, M, 1, 256, 1, 1, &fka, sizeof(fka), true));
                    }
                    enq_q4(l3_o_qs, l3_o_dm, gqa_o_rows, gqa_o_cols, gqa_o_stride, mid_q8_va, mid_ds_va, partial_va, M);
                });
            std::vector<float> got(static_cast<size_t>(M) * D);
            VJ_CHECK(rt.download_from_vram(0, got.data(), partial_va, got.size() * 4u));
            const auto ps = compare_f32(got.data(), reinterpret_cast<const float*>(b_ref.data()), got.size());
            const bool ok = !ps.nan_found && ps.max_abs_err <= 2e-3 && ps.cosine_sim >= 0.99999;
            const double gbs = (us > 0.0) ? (static_cast<double>(l3_gqa_bytes) / (us * 1e3)) : 0.0;
            results.push_back({"gqa", M, ok, us, gbs, ps.max_abs_err, ps.cosine_sim, ""});
        }
    }

    // ---- STAGE 3: MoE (Layer 0) ----
    if (do_moe && !do_super) {
        const auto* k_rtr      = need_k(0, "ornith_moe_router_fp16_logits_kernel");
        const auto* k_pf_rtr   = need_k(0, "vj_prefill_router_fp16_tiled_kernel");
        const auto* k_top8     = need_k(0, "ornith_moe_router_top8_kernel");
        const auto* k_swiglu   = need_k(0, "ornith_swiglu_quant_q8_kernel");
        const auto* k_back224  = need_k(0, "ornith_moe_m1_back_s224_kernel");
        const auto* k_gu_iq2s  = need_k(0, "ornith_moe_gu_iq2s_merged_kernel");
        const auto* k_dn_d8    = need_k(0, "vj_moe_down_dot8_kernel");
        const auto* k_rg       = need_k(0, "vj_moe_route_group_kernel");
        const auto* k_pf_gu    = need_k(0, "vj_prefill_moe_gu_iq2s_tiled_kernel");
        const auto* k_pf_dn    = need_k(0, "vj_prefill_moe_down_iq3_tiled_kernel");
        const auto* k_pf_comb  = need_k(0, "vj_moe_down_combine_all_f32_kernel");
        for (uint32_t M : m_list) {
            if (prefill_only && M < 16u) continue;
            const std::string pfx = golden_dir + "/m" + std::to_string(M) + "_moe_";
            std::vector<uint8_t> b_xn, b_xq, b_xd, b_ref;
            if (!read_bin_file(pfx + "xnorm.bin", b_xn, &err) ||
                !read_bin_file(pfx + "xq8.bin", b_xq, &err) ||
                !read_bin_file(pfx + "xds.bin", b_xd, &err) ||
                !read_bin_file(pfx + "partial_out.bin", b_ref, &err)) {
                results.push_back({"moe", M, false, 0, 0, 0, 0, err});
                continue;
            }
            VJ_CHECK(rt.upload_to_vram(0, xnorm_va, b_xn.data(), b_xn.size()));
            VJ_CHECK(rt.upload_to_vram(0, xq8_va,   b_xq.data(), b_xq.size()));
            VJ_CHECK(rt.upload_to_vram(0, xds_va,   b_xd.data(), b_xd.size()));
            zero_gpu0(moe_bq_va, 256u);
            zero_gpu0(pf_queue_va, 256u);
            VJ_CHECK(rt.submit_batch(0) && rt.wait_idle(0));
            const uint64_t moe_est_bytes =
                (M == 1u) ? (256ULL * D * 2ULL + 8ULL * (local_ff * 123ULL * 16ULL + D * (dn_d8_w_stride + dn_d8_s_stride)) + 3600000ULL)
                          : (256ULL * D * 2ULL + std::min<uint64_t>(256ULL, static_cast<uint64_t>(M) * 8ULL) * (local_ff * 123ULL * 16ULL + D * (dn_d8_w_stride + dn_d8_s_stride)));
            const double us = time_gpu0(
                [&]() {
                    if (M >= 16u) {
                        zero_gpu0(pf_queue_va, 256u);
                        VJ_CHECK(rt.submit_batch(0) && rt.wait_idle(0));
                    }
                },
                [&]() {
                    struct RtrKA { uint64_t wr, ws, xn, log, sc; uint32_t M, D; };
                    const RtrKA rka{l0_rtr, l0_sh_g, xnorm_va, rtr_log_va, sh_sc_va, M, D};
                    if (M >= 32u) VJ_CHECK(rt.enqueue_kernel(0, *k_pf_rtr, 9u * 128u, (M + 31u) / 32u, 1, 128, 1, 1, &rka, sizeof(rka), true));
                    else          VJ_CHECK(rt.enqueue_kernel(0, *k_rtr, 33u * 256u, (M + 7u) / 8u, 1, 256, 1, 1, &rka, sizeof(rka), true));
                    enq_q4(l0_sh_gu_qs, l0_sh_gu_dm, sh_gu_rows, sh_gu_cols, sh_gu_stride, xq8_va, xds_va, sh_gu_va, M);
                    struct T8KA { uint64_t log, tids, tw; uint32_t M; };
                    const T8KA t8{rtr_log_va, top_ids_va, top_w_va, M};
                    VJ_CHECK(rt.enqueue_kernel(0, *k_top8, M * 256u, 1, 1, 256, 1, 1, &t8, sizeof(t8), true));
                    struct SwKA { uint64_t gu, sc, oq, od; uint32_t M, lff; };
                    const SwKA sw{sh_gu_va, sh_sc_va, sh_q8_va, sh_ds_va, M, local_ff};
                    VJ_CHECK(rt.enqueue_kernel(0, *k_swiglu, M * 256u, 1, 1, 256, 1, 1, &sw, sizeof(sw), true));
                    if (M == 1u) {
                        ++moe_sense;
                        struct BackKA {
                            uint64_t wgu, xq, xds, tids, tw, mq, md, shwq, shwdm, shq, shds, shd, wd, wds, out, q;
                            uint32_t lff, D, sh_rows, sh_stride, sense, dn_wst, dn_sst, pad; uint64_t ptx;
                        };
                        const BackKA bka{l0_gu_iq2s, xq8_va, xds_va, top_ids_va, top_w_va, moe_q8_va, moe_d_va,
                                         l0_sh_dn_qs, l0_sh_dn_dm, sh_q8_va, sh_ds_va, sh_dn_va, l0_dn_d8_w, l0_dn_d8_s, partial_va, moe_bq_va,
                                         local_ff, D, sh_dn_rows, sh_dn_stride, moe_sense, dn_d8_w_stride, dn_d8_s_stride, 0u, 0ULL};
                        VJ_CHECK(rt.enqueue_kernel(0, *k_back224, 224u * 256u, 1, 1, 256, 1, 1, &bka, sizeof(bka), true));
                    } else if (M < 16u) {
                        struct GuKA { uint64_t wgu, xq, xd, tids, mq, md; uint32_t M, lff; };
                        const GuKA gka{l0_gu_iq2s, xq8_va, xds_va, top_ids_va, moe_q8_va, moe_d_va, M, local_ff};
                        VJ_CHECK(rt.enqueue_kernel(0, *k_gu_iq2s, 8u * (local_ff / 32u) * 256u, M, 1, 256, 1, 1, &gka, sizeof(gka), true));
                        enq_q4(l0_sh_dn_qs, l0_sh_dn_dm, sh_dn_rows, sh_dn_cols, sh_dn_stride, sh_q8_va, sh_ds_va, sh_dn_va, M);
                        struct DnKA { uint64_t wq, ws, mq, md, tids, tw, shd, out; uint32_t M, D, wst, sst; };
                        const DnKA dka{l0_dn_d8_w, l0_dn_d8_s, moe_q8_va, moe_d_va, top_ids_va, top_w_va, sh_dn_va, partial_va, M, D, dn_d8_w_stride, dn_d8_s_stride};
                        VJ_CHECK(rt.enqueue_kernel(0, *k_dn_d8, ((D + 15u) / 16u) * 256u, M, 1, 256, 1, 1, &dka, sizeof(dka), true));
                    } else {
                        const uint32_t R = M * 8u, U = std::min(R, 256u);
                        const uint32_t icap = std::min(R, (R + 63u * U + 63u) / 64u);
                        struct RgKA { uint64_t tids, hdr, cnt, off, uniq, items, routes; uint32_t M, min_group, icap; };
                        const RgKA rg{top_ids_va, dgrp_hdr_va, dgrp_cnt_va, dgrp_off_va, dgrp_unq_va, dgrp_itm_va, dgrp_rte_va, M, 2u, icap};
                        VJ_CHECK(rt.enqueue_kernel(0, *k_rg, 256u, 1, 1, 256, 1, 1, &rg, sizeof(rg), true));
                        enq_q4(l0_sh_dn_qs, l0_sh_dn_dm, sh_dn_rows, sh_dn_cols, sh_dn_stride, sh_q8_va, sh_ds_va, sh_dn_va, M);
                        struct GuTKA { uint64_t wgu, xq, xds, off, cnt, sms, mq, md, queue; uint32_t lff, D; };
                        const GuTKA gt{l0_gu_iq2s, xq8_va, xds_va, dgrp_off_va, dgrp_cnt_va, dgrp_rte_va, moe_q8_va, moe_d_va, pf_queue_va, local_ff, D};
                        VJ_CHECK(rt.enqueue_kernel(0, *k_pf_gu, 240u * 256u, 1, 1, 256, 1, 1, &gt, sizeof(gt), true));
                        struct DnTKA { uint64_t wq, ws, mq, md, tw, off, cnt, sms, part, queue; uint32_t D, wst, sst; };
                        const DnTKA dnt{l0_dn_d8_w, l0_dn_d8_s, moe_q8_va, moe_d_va, top_w_va, dgrp_off_va, dgrp_cnt_va, dgrp_rte_va, dgrp_buf_va, pf_queue_va, D, dn_d8_w_stride, dn_d8_s_stride};
                        VJ_CHECK(rt.enqueue_kernel(0, *k_pf_dn, 240u * 256u, 1, 1, 256, 1, 1, &dnt, sizeof(dnt), true));
                        struct CbAllKA { uint64_t part, shd, out; uint32_t M, D; };
                        const CbAllKA cba{dgrp_buf_va, sh_dn_va, partial_va, M, D};
                        VJ_CHECK(rt.enqueue_kernel(0, *k_pf_comb, ((D + 255u) / 256u) * 256u, M, 1, 256, 1, 1, &cba, sizeof(cba), true));
                    }
                });
            std::vector<float> got(static_cast<size_t>(M) * D);
            VJ_CHECK(rt.download_from_vram(0, got.data(), partial_va, got.size() * 4u));
            const auto ps = compare_f32(got.data(), reinterpret_cast<const float*>(b_ref.data()), got.size());
            const bool ok = !ps.nan_found && ps.max_abs_err <= 2e-3 && ps.cosine_sim >= 0.99999;
            const double gbs = (us > 0.0) ? (static_cast<double>(moe_est_bytes) / (us * 1e3)) : 0.0;
            results.push_back({"moe", M, ok, us, gbs, ps.max_abs_err, ps.cosine_sim, ""});
        }
    }

    // ---- STAGE 4: P2P AllReduce (2x MI50 Dual-BAR0 FP16 & INT8) ----
    if (do_ar && rt.num_gpus() >= 2) {
        const vegajet::kfd::LoadedKernel* k_ar_fp16[2] = {need_k(0, "ornith_p2p_ar_res_rms_quant_kernel"), need_k(1, "ornith_p2p_ar_res_rms_quant_kernel")};
        const vegajet::kfd::LoadedKernel* k_ar_i8[2]   = {need_k(0, "ornith_p2p_ar_i8_res_rms_quant_kernel"), need_k(1, "ornith_p2p_ar_i8_res_rms_quant_kernel")};
        const auto* k_ms1        = need_k(1, "kfd_memset_u32");
        uint64_t p_va[2], h_va[2], w_va[2], xn_va[2], xq_va[2], xd_va[2], bar_va[2];
        std::vector<float> h_p0(max_M * D), h_p1(max_M * D), h_h(max_M * D), h_w(D, 1.0f);
        for (size_t i = 0; i < h_p0.size(); ++i) {
            h_p0[i] = static_cast<float>((static_cast<int>(i % 19) - 9)) * 0.05f;
            h_p1[i] = static_cast<float>((static_cast<int>((i * 3) % 17) - 8)) * 0.05f;
            h_h[i]  = static_cast<float>((static_cast<int>((i * 5) % 13) - 6)) * 0.10f;
        }
        for (size_t g = 0; g < 2; ++g) {
            auto& ar = rt.vram_arena(g);
            p_va[g]   = ar.alloc(max_M * D * 4ULL, 4096);
            h_va[g]   = ar.alloc(max_M * D * 4ULL, 4096);
            w_va[g]   = ar.alloc(D * 4ULL, 256);
            xn_va[g]  = ar.alloc(max_M * D * 4ULL, 4096);
            xq_va[g]  = ar.alloc(max_M * D, 4096);
            xd_va[g]  = ar.alloc(max_M * (D / 32u) * 8ULL, 256);
            bar_va[g] = ar.alloc(256u, 256);
            VJ_CHECK(rt.upload_to_vram(g, p_va[g], g == 0 ? h_p0.data() : h_p1.data(), max_M * D * 4ULL));
            VJ_CHECK(rt.upload_to_vram(g, w_va[g], h_w.data(), D * 4ULL));
            struct MsKA { uint64_t dst; uint32_t val; uint32_t pad; uint64_t n; };
            const MsKA mka{bar_va[g], 0u, 0u, 64u};
            VJ_CHECK(rt.dispatch_sync(g, *(g == 0 ? k_memset : k_ms1), 256u, 1, 1, 256, 1, 1, &mka, sizeof(mka)));
        }
        rt.reset_allreduce_mailboxes();
        uint32_t ep = 1u;
        const uint64_t pay_base = rt.mailbox_payload_gpu_va();
        const uint64_t flg_base = rt.mailbox_flags_gpu_va();
        for (uint32_t M : m_list) {
            if (prefill_only && M < 64u) continue;
            const bool use_i8 = (M >= 64u);
            double best_us = 1e30;
            for (int it = 0; it < iters + 1; ++it) {
                for (size_t g = 0; g < 2; ++g) VJ_CHECK(rt.upload_to_vram(g, h_va[g], h_h.data(), static_cast<size_t>(M) * D * 4u));
                const uint32_t cur_ep = ep++;
                const uint32_t pp = cur_ep & 1u;
                const ClkKA c0{clk_va, 0u, 0u}, c1{clk_va, 1u, 0u};
                VJ_CHECK(rt.enqueue_kernel(0, *k_clk, 64u, 1, 1, 64, 1, 1, &c0, sizeof(c0), true));
                for (size_t g = 0; g < 2; ++g) {
                    const size_t peer = 1u - g;
                    struct ArKA { uint64_t xp, ptx, lrx, pfl, lfl, bar, h, w, xn, xq, xd; uint32_t M, D, ep, dn; float eps; };
                    const ArKA aka{p_va[g], pay_base + peer * vegajet::kfd::kArGpuBytes + pp * vegajet::kfd::kArSlotBytes,
                                   pay_base + g * vegajet::kfd::kArGpuBytes + pp * vegajet::kfd::kArSlotBytes,
                                   flg_base + peer * 4096ULL, flg_base + g * 4096ULL,
                                   bar_va[g], h_va[g], w_va[g], xn_va[g], xq_va[g], xd_va[g], M, D, cur_ep, 1u, eps};
                    const uint32_t nwg = use_i8 ? std::min<uint32_t>(M, 128u) : M;
                    VJ_CHECK(rt.enqueue_kernel(g, *(use_i8 ? k_ar_i8[g] : k_ar_fp16[g]), nwg * 256u, 1, 1, 256, 1, 1, &aka, sizeof(aka), true));
                }
                VJ_CHECK(rt.enqueue_kernel(0, *k_clk, 64u, 1, 1, 64, 1, 1, &c1, sizeof(c1), true));
                VJ_CHECK(rt.submit_batch(0) && rt.submit_batch(1));
                VJ_CHECK(rt.wait_idle(0) && rt.wait_idle(1));
                uint64_t ts[2] = {0, 0};
                VJ_CHECK(rt.download_from_vram(0, ts, clk_va, sizeof(ts)));
                const double us = static_cast<double>(ts[1] - ts[0]) * 0.04;
                if (it > 0) best_us = std::min(best_us, us);
            }
            std::vector<float> got0(static_cast<size_t>(M) * D), got1(static_cast<size_t>(M) * D), ref(static_cast<size_t>(M) * D);
            VJ_CHECK(rt.download_from_vram(0, got0.data(), h_va[0], got0.size() * 4u));
            VJ_CHECK(rt.download_from_vram(1, got1.data(), h_va[1], got1.size() * 4u));
            for (size_t i = 0; i < ref.size(); ++i) ref[i] = h_h[i] + h_p0[i] + h_p1[i];
            const auto p01 = compare_f32(got0.data(), got1.data(), ref.size());
            const auto p0r = compare_f32(got0.data(), ref.data(), ref.size());
            const double tol = use_i8 ? 1.5e-2 : 2e-3;
            const bool ok = !p01.nan_found && !p0r.nan_found && p01.max_abs_err == 0.0 && p0r.max_abs_err <= tol;
            const double wire_bytes = static_cast<double>(M) * D * (use_i8 ? 1.125 : 2.0);
            const double gbs = (best_us > 0.0) ? (wire_bytes / (best_us * 1e3)) : 0.0;
            results.push_back({use_i8 ? "ar_i8" : "ar_fp16", M, ok, best_us, gbs, p0r.max_abs_err, p0r.cosine_sim, ""});
        }
    }

    // ---- STAGE 5: M=1 Super-Kernels (ornith_m1_{gdn,gqa,moe}_super_kernel) ----
    if (do_super) {
        const auto* k_gdn_sup = need_k(0, "ornith_m1_gdn_super_kernel");
        const auto* k_gqa_sup = need_k(0, "ornith_m1_gqa_super_kernel");
        const auto* k_moe_sup = need_k(0, "ornith_m1_moe_super_kernel");
        const uint64_t sbar_va     = a0.alloc(2048u, 256);
        const uint64_t gdn_desc_va = a0.alloc(sizeof(vegajet::ornith_super::GdnSuperDesc), 256);
        const uint64_t gqa_desc_va = a0.alloc(sizeof(vegajet::ornith_super::GqaSuperDesc), 256);
        const uint64_t moe_desc_va = a0.alloc(sizeof(vegajet::ornith_super::MoeSuperDesc), 256);
        const uint64_t part_o_va   = a0.alloc(16ULL * 60ULL * 256ULL * 4ULL, 4096);
        const uint64_t part_ml_va  = a0.alloc(16ULL * 60ULL * 2ULL * 4ULL, 4096);
        const char* swg_env = std::getenv("VEGAJET_SUPER_WGS");
        const uint32_t swgs = (swg_env && std::atoi(swg_env) >= 192 && std::atoi(swg_env) <= 240)
                            ? static_cast<uint32_t>(std::atoi(swg_env)) : 192u;
        uint32_t ssense = 0u;
        zero_gpu0(sbar_va, 2048u);
        zero_gpu0(gdn_bar_va, 8192u);
        VJ_CHECK(rt.submit_batch(0) && rt.wait_idle(0));

        // 1) GDN super M=1
        {
            std::vector<uint8_t> b_xq, b_xd, b_cin, b_sin, b_ref;
            if (read_bin_file(golden_dir + "/m1_gdn_xq8.bin", b_xq, &err) &&
                read_bin_file(golden_dir + "/m1_gdn_xds.bin", b_xd, &err) &&
                read_bin_file(golden_dir + "/m1_gdn_conv_in.bin", b_cin, &err) &&
                read_bin_file(golden_dir + "/m1_gdn_ssm_in.bin", b_sin, &err) &&
                read_bin_file(golden_dir + "/m1_gdn_partial_out.bin", b_ref, &err)) {
                VJ_CHECK(rt.upload_to_vram(0, xq8_va, b_xq.data(), b_xq.size()));
                VJ_CHECK(rt.upload_to_vram(0, xds_va, b_xd.data(), b_xd.size()));
                vegajet::ornith_super::GdnSuperDesc d{};
                d.x_q8 = xq8_va; d.x_ds = xds_va; d.x_partial = partial_va;
                d.w_qkv_qs = l0_gdn_qkv_qs; d.w_qkv_dm = l0_gdn_qkv_dm; d.gdn_qkv_gab = gdn_qkv_va;
                d.qkv_rows = gdn_qkv_rows; d.qkv_cols = gdn_qkv_cols; d.qkv_qs_stride = gdn_qkv_stride; d.qkv_dm_stride = gdn_qkv_cols >> 5;
                d.conv_w = l0_ssm_cw; d.conv_ring = conv_ring_va; d.dt_bias = l0_ssm_dt; d.ssm_a = l0_ssm_a;
                d.ssm_ring_base = ssm_ring_va; d.ssm_norm_w = l0_ssm_nw; d.mid_q8 = mid_q8_va; d.mid_ds = mid_ds_va;
                d.gdn_bar = gdn_bar_va; d.gdn_sq = gdn_sq_va;
                d.nk = 8u; d.nv = 32u; d.conv_dim = 6144u; d.z_off = 6144u; d.alpha_off = 10240u; d.beta_off = 10272u;
                d.w_out_qs = l0_ssm_out_qs; d.w_out_dm = l0_ssm_out_dm;
                d.out_rows = ssm_out_rows; d.out_cols = ssm_out_cols; d.out_qs_stride = ssm_out_stride; d.out_dm_stride = ssm_out_cols >> 5;
                d.D = D; d.eps = eps; d.super_bar = sbar_va;
                VJ_CHECK(rt.upload_to_vram(0, gdn_desc_va, &d, sizeof(d)));
                const double us = time_gpu0(
                    [&]() {
                        VJ_CHECK(rt.upload_to_vram(0, conv_ring_va, b_cin.data(), b_cin.size()));
                        VJ_CHECK(rt.upload_to_vram(0, ssm_ring_va, b_sin.data(), b_sin.size()));
                    },
                    [&]() {
                        ++ssense; ++gdn_sense;
                        struct KA { uint64_t desc, ov; uint32_t pep, oep, cin, cout, sh, ur, ss, gs; };
                        const KA ka{gdn_desc_va, 0ULL, 0u, 0u, 0u, 1u, 0u, 0u, ssense, gdn_sense};
                        VJ_CHECK(rt.enqueue_kernel(0, *k_gdn_sup, swgs * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
                    });
                std::vector<float> got(D);
                VJ_CHECK(rt.download_from_vram(0, got.data(), partial_va, D * 4u));
                const auto ps = compare_f32(got.data(), reinterpret_cast<const float*>(b_ref.data()), D);
                results.push_back({"sup_gdn", 1u, !ps.nan_found && ps.max_abs_err <= 2e-4 && ps.cosine_sim >= 0.999999,
                                   us, (us > 0.0) ? (static_cast<double>(l0_gdn_bytes) / (us * 1e3)) : 0.0, ps.max_abs_err, ps.cosine_sim, ""});
            }
        }
        // 2) GQA super M=1
        {
            std::vector<uint8_t> b_xq, b_xd, b_kc, b_vc, b_ref;
            if (read_bin_file(golden_dir + "/m1_gqa_xq8.bin", b_xq, &err) &&
                read_bin_file(golden_dir + "/m1_gqa_xds.bin", b_xd, &err) &&
                read_bin_file(golden_dir + "/m1_gqa_k_cache.bin", b_kc, &err) &&
                read_bin_file(golden_dir + "/m1_gqa_v_cache.bin", b_vc, &err) &&
                read_bin_file(golden_dir + "/m1_gqa_partial_out.bin", b_ref, &err)) {
                VJ_CHECK(rt.upload_to_vram(0, xq8_va, b_xq.data(), b_xq.size()));
                VJ_CHECK(rt.upload_to_vram(0, xds_va, b_xd.data(), b_xd.size()));
                vegajet::ornith_super::GqaSuperDesc d{};
                d.x_q8 = xq8_va; d.x_ds = xds_va; d.x_partial = partial_va;
                d.w_qkv_qs = l3_qkv_qs; d.w_qkv_dm = l3_qkv_dm; d.attn_qkv = attn_qkv_va;
                d.qkv_rows = gqa_qkv_rows; d.qkv_cols = gqa_qkv_cols; d.qkv_qs_stride = gqa_qkv_stride; d.qkv_dm_stride = gqa_qkv_cols >> 5;
                d.q_norm_w = l3_qn; d.k_norm_w = l3_kn; d.q_buf = q_buf_va;
                d.k_cache_q8 = k_cache_va; d.v_cache_q8 = v_cache_va;
                d.part_o = part_o_va; d.part_ml = part_ml_va; d.mid_q8 = mid_q8_va; d.mid_ds = mid_ds_va;
                d.n_q_local = n_q_local; d.n_kv_local = n_kv_local; d.rope_dims = rot_dim; d.rope_base = rope_base;
                d.w_o_qs = l3_o_qs; d.w_o_dm = l3_o_dm;
                d.o_rows = gqa_o_rows; d.o_cols = gqa_o_cols; d.o_qs_stride = gqa_o_stride; d.o_dm_stride = gqa_o_cols >> 5;
                d.D = D; d.eps = eps; d.super_bar = sbar_va;
                VJ_CHECK(rt.upload_to_vram(0, gqa_desc_va, &d, sizeof(d)));
                const double us = time_gpu0(
                    [&]() {
                        VJ_CHECK(rt.upload_to_vram(0, k_cache_va, b_kc.data(), b_kc.size()));
                        VJ_CHECK(rt.upload_to_vram(0, v_cache_va, b_vc.data(), b_vc.size()));
                    },
                    [&]() {
                        ++ssense;
                        struct KA { uint64_t desc; uint32_t pep, oep, p0, ss; };
                        const KA ka{gqa_desc_va, 0u, 0u, pos0, ssense};
                        VJ_CHECK(rt.enqueue_kernel(0, *k_gqa_sup, swgs * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
                    });
                std::vector<float> got(D);
                VJ_CHECK(rt.download_from_vram(0, got.data(), partial_va, D * 4u));
                const auto ps = compare_f32(got.data(), reinterpret_cast<const float*>(b_ref.data()), D);
                results.push_back({"sup_gqa", 1u, !ps.nan_found && ps.max_abs_err <= 2e-3 && ps.cosine_sim >= 0.99999,
                                   us, (us > 0.0) ? (static_cast<double>(l3_gqa_bytes) / (us * 1e3)) : 0.0, ps.max_abs_err, ps.cosine_sim, ""});
            }
        }
        // 3) MoE super M=1
        {
            std::vector<uint8_t> b_xn, b_xq, b_xd, b_ref;
            if (read_bin_file(golden_dir + "/m1_moe_xnorm.bin", b_xn, &err) &&
                read_bin_file(golden_dir + "/m1_moe_xq8.bin", b_xq, &err) &&
                read_bin_file(golden_dir + "/m1_moe_xds.bin", b_xd, &err) &&
                read_bin_file(golden_dir + "/m1_moe_partial_out.bin", b_ref, &err)) {
                VJ_CHECK(rt.upload_to_vram(0, xnorm_va, b_xn.data(), b_xn.size()));
                VJ_CHECK(rt.upload_to_vram(0, xq8_va,   b_xq.data(), b_xq.size()));
                VJ_CHECK(rt.upload_to_vram(0, xds_va,   b_xd.data(), b_xd.size()));
                vegajet::ornith_super::MoeSuperDesc d{};
                d.x_partial = partial_va; d.x_norm = xnorm_va; d.x_q8 = xq8_va; d.x_ds = xds_va;
                d.w_router_f16 = l0_rtr; d.w_sh_gate = l0_sh_g; d.sh_gu_qs = l0_sh_gu_qs; d.sh_gu_dm = l0_sh_gu_dm;
                d.router_logits = rtr_log_va; d.sh_gate_sc = sh_sc_va; d.sh_gu_out = sh_gu_va;
                d.top_ids = top_ids_va; d.top_w = top_w_va; d.sh_q8 = sh_q8_va; d.sh_ds = sh_ds_va;
                d.sh_gu_rows = sh_gu_rows; d.sh_gu_stride = sh_gu_stride; d.local_ff = local_ff; d.D = D;
                d.w_gu_iq2s = l0_gu_iq2s; d.mid_q8 = moe_q8_va; d.mid_d = moe_d_va;
                d.sh_dn_qs = l0_sh_dn_qs; d.sh_dn_dm = l0_sh_dn_dm; d.sh_down_out = sh_dn_va;
                d.w_dn_d8_w = l0_dn_d8_w; d.w_dn_d8_s = l0_dn_d8_s;
                d.sh_dn_stride = sh_dn_stride; d.dn_w_stride = dn_d8_w_stride; d.dn_s_stride = dn_d8_s_stride;
                d.eps = eps; d.super_bar = sbar_va; d.mid_f32 = dgrp_buf_va;
                VJ_CHECK(rt.upload_to_vram(0, moe_desc_va, &d, sizeof(d)));
                const uint64_t moe_b = 256ULL * D * 2ULL + 8ULL * (local_ff * 123ULL * 16ULL + D * (dn_d8_w_stride + dn_d8_s_stride)) + 3600000ULL;
                const double us = time_gpu0([]() {}, [&]() {
                    ++ssense;
                    struct KA { uint64_t desc; uint32_t aep, oep, ss; };
                    const KA ka{moe_desc_va, 0u, 0u, ssense};
                    VJ_CHECK(rt.enqueue_kernel(0, *k_moe_sup, swgs * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka), true));
                });
                std::vector<float> got(D);
                VJ_CHECK(rt.download_from_vram(0, got.data(), partial_va, D * 4u));
                const auto ps = compare_f32(got.data(), reinterpret_cast<const float*>(b_ref.data()), D);
                results.push_back({"sup_moe", 1u, !ps.nan_found && ps.max_abs_err <= 2e-3 && ps.cosine_sim >= 0.99999,
                                   us, (us > 0.0) ? (static_cast<double>(moe_b) / (us * 1e3)) : 0.0, ps.max_abs_err, ps.cosine_sim, ""});
            }
        }
    }

    read_hw_telemetry(tmem1, ue1, de1);
    const double max_tmem = std::max(tmem0, tmem1);
    const uint64_t ue_delta = (ue1 >= ue0) ? (ue1 - ue0) : 0ULL;
    const uint64_t de_delta = (de1 >= de0) ? (de1 - de0) : 0ULL;
    bool all_pass = !results.empty() && (ue_delta == 0ULL) && (de_delta == 0ULL) && (max_tmem < 65.0);
    double worst_err = 0.0, sum_us = 0.0, peak_gbs = 0.0;
    for (const auto& r : results) {
        if (!r.pass) all_pass = false;
        worst_err = std::max(worst_err, r.max_err);
        sum_us += r.us;
        peak_gbs = std::max(peak_gbs, r.gb_s);
    }

    if (!json_only) {
        std::printf("=== vegajet_layer_test (load=%.0f ms, stage=%s) ===\n", load_ms, stage_arg.c_str());
        for (const auto& r : results) {
            std::printf("  %-8s M=%-3u | %8.1f us | %7.1f GB/s | max_err=%.3e cos=%.6f | %s%s\n",
                        r.stage.c_str(), r.M, r.us, r.gb_s, r.max_err, r.cosine,
                        r.pass ? "PASS" : "FAIL", r.detail.empty() ? "" : (" (" + r.detail + ")").c_str());
        }
    }
    std::printf("{\"status\":\"%s\",\"cases\":%zu,\"load_ms\":%.1f,\"sum_us\":%.1f,\"peak_gb_s\":%.1f,\"max_err\":%.3e,\"tmem_c\":%.1f,\"ecc_ue\":%llu,\"ecc_de\":%llu}\n",
                all_pass ? "PASS" : "FAIL", results.size(), load_ms, sum_us, peak_gbs, worst_err, max_tmem,
                static_cast<unsigned long long>(ue_delta), static_cast<unsigned long long>(de_delta));
    return all_pass ? 0 : 1;
#endif
}
