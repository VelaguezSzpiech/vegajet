// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Live gfx906 MI50 GPU sanity check for Codex's 3-kernel zero-spill universal VM:
//   - vj_vm_aux_block  (RmsF32)
//   - vj_vm_attn_block (ArResRmsQ8)
//   - vj_vm_ffn_block  (ActMulQ8)
// Verifies runtime private_segment_fixed_size == 0, status == 0, and numerical parity vs vegajet::vm::run_cpu.
#include "core/kfd_runtime.hpp"
#include "universal_draft/ops/vj_vm_cpu.hpp"
#include "universal_draft/ops/vj_vm_desc.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct VmKA {
    uint64_t program;
    uint32_t num_ops;
    uint32_t cache_policy;
    uint64_t status;
};

}  // namespace

int main(int argc, char** argv) {
    using namespace vegajet;
    using namespace vegajet::vm;

    if (!kfd::Runtime::compiled_with_gpu()) {
        std::printf("test_universal_vm_gpu: SKIP (CPU-only build)\n");
        return 77;
    }

    std::string hsaco_path = (argc > 1 && argv[1] && argv[1][0]) ? argv[1] : "hsaco/vj_universal_vm.hsaco";
    std::string base_path  = (argc > 2 && argv[2] && argv[2][0]) ? argv[2] : "hsaco/vj_kfd_base.hsaco";

    kfd::Runtime rt;
    std::string err;
    if (!rt.init(0.5, 16, &err)) {
        std::printf("test_universal_vm_gpu: SKIP (%s)\n", err.c_str());
        return 77;
    }
    if (!rt.load_hsaco(0, base_path, &err)) {
        std::fprintf(stderr, "FAIL: load_hsaco(%s): %s\n", base_path.c_str(), err.c_str());
        return 1;
    }
    if (!rt.load_hsaco(0, hsaco_path, &err)) {
        std::fprintf(stderr, "FAIL: load_hsaco(%s): %s\n", hsaco_path.c_str(), err.c_str());
        return 1;
    }

    const auto* k_attn = rt.get_kernel(0, "vj_vm_attn_block");
    const auto* k_ffn  = rt.get_kernel(0, "vj_vm_ffn_block");
    const auto* k_aux  = rt.get_kernel(0, "vj_vm_aux_block");
    if (!k_attn || !k_ffn || !k_aux) {
        std::fprintf(stderr, "FAIL: missing one of vj_vm_{attn,ffn,aux}_block\n");
        return 1;
    }

    constexpr uint32_t kCols = 256u;
    constexpr uint32_t kBlocks = kCols / 32u;
    constexpr float kEps = 1e-6f;

    std::vector<float> h_x(kCols), h_res(kCols), h_w(kCols), h_gate(kCols), h_up(kCols);
    for (uint32_t i = 0; i < kCols; ++i) {
        h_x[i]    = std::sin(static_cast<float>(i + 1) * 0.13f);
        h_res[i]  = std::cos(static_cast<float>(i + 1) * 0.07f) * 0.5f;
        h_w[i]    = 0.8f + 0.002f * static_cast<float>(i % 64u);
        h_gate[i] = std::sin(static_cast<float>(i + 3) * 0.11f) * 1.5f;
        h_up[i]   = std::cos(static_cast<float>(i + 5) * 0.09f) * 1.2f;
    }

    auto& arena = rt.vram_arena(0);
    const uint64_t va_x      = arena.alloc(kCols * 4u, 256);
    const uint64_t va_res    = arena.alloc(kCols * 4u, 256);
    const uint64_t va_w      = arena.alloc(kCols * 4u, 256);
    const uint64_t va_rms    = arena.alloc(kCols * 4u, 256);
    const uint64_t va_q8     = arena.alloc(kCols, 256);
    const uint64_t va_sc     = arena.alloc(kBlocks * 4u, 256);
    const uint64_t va_gate   = arena.alloc(kCols * 4u, 256);
    const uint64_t va_up     = arena.alloc(kCols * 4u, 256);
    const uint64_t va_aq8    = arena.alloc(kCols, 256);
    const uint64_t va_asc    = arena.alloc(kBlocks * 4u, 256);
    const uint64_t va_prog   = arena.alloc(4u * sizeof(VjVmOpDesc), 256);
    const uint64_t va_status = arena.alloc(64u, 256);
    if (!va_x || !va_res || !va_w || !va_rms || !va_q8 || !va_sc || !va_gate || !va_up ||
        !va_aq8 || !va_asc || !va_prog || !va_status) {
        std::fprintf(stderr, "FAIL: VRAM allocation failed\n");
        return 1;
    }

    const uint32_t zero_status = 0u;
    if (!rt.upload_to_vram(0, va_x, h_x.data(), kCols * 4u) ||
        !rt.upload_to_vram(0, va_res, h_res.data(), kCols * 4u) ||
        !rt.upload_to_vram(0, va_w, h_w.data(), kCols * 4u) ||
        !rt.upload_to_vram(0, va_gate, h_gate.data(), kCols * 4u) ||
        !rt.upload_to_vram(0, va_up, h_up.data(), kCols * 4u) ||
        !rt.upload_to_vram(0, va_status, &zero_status, sizeof(zero_status))) {
        std::fprintf(stderr, "FAIL: upload_to_vram failed\n");
        return 1;
    }

    auto run_gpu_op = [&](const kfd::LoadedKernel& k, const VjVmOpDesc& op, uint32_t n_wgs) -> bool {
        if (!rt.upload_to_vram(0, va_prog, &op, sizeof(op))) return false;
        VmKA ka{va_prog, 1u, CacheDefault, va_status};
        return rt.dispatch_sync(0, k, n_wgs * 256u, 1, 1, 256, 1, 1, &ka, sizeof(ka));
    };

    // 1) vj_vm_aux_block: RmsF32
    VjVmOpDesc d_rms{};
    d_rms.opcode = RmsF32; d_rms.rows = 1u; d_rms.cols = kCols; d_rms.eps = kEps;
    d_rms.ptr0 = va_x; d_rms.ptr1 = va_w; d_rms.ptr2 = va_rms;
    if (!run_gpu_op(*k_aux, d_rms, 1u)) { std::fprintf(stderr, "FAIL: vj_vm_aux_block dispatch\n"); return 1; }

    // 2) vj_vm_attn_block: ArResRmsQ8
    VjVmOpDesc d_ar{};
    d_ar.opcode = ArResRmsQ8; d_ar.rows = 1u; d_ar.cols = kCols; d_ar.eps = kEps;
    d_ar.ptr0 = va_x; d_ar.ptr1 = va_res; d_ar.ptr2 = va_w; d_ar.ptr3 = va_q8; d_ar.ptr4 = va_sc;
    if (!run_gpu_op(*k_attn, d_ar, 1u)) { std::fprintf(stderr, "FAIL: vj_vm_attn_block dispatch\n"); return 1; }

    // 3) vj_vm_ffn_block: ActMulQ8 (SiLU)
    VjVmOpDesc d_act{};
    d_act.opcode = ActMulQ8; d_act.rows = 1u; d_act.cols = kCols; d_act.eps = kEps;
    d_act.ptr0 = va_gate; d_act.ptr1 = va_up; d_act.ptr2 = va_aq8; d_act.ptr3 = va_asc;
    if (!run_gpu_op(*k_ffn, d_act, 1u)) { std::fprintf(stderr, "FAIL: vj_vm_ffn_block dispatch\n"); return 1; }

    uint32_t gpu_status = 99u;
    std::vector<float> g_rms(kCols), g_res(kCols), g_sc(kBlocks), g_asc(kBlocks);
    std::vector<int8_t> g_q8(kCols), g_aq8(kCols);
    if (!rt.download_from_vram(0, &gpu_status, va_status, sizeof(gpu_status)) ||
        !rt.download_from_vram(0, g_rms.data(), va_rms, kCols * 4u) ||
        !rt.download_from_vram(0, g_res.data(), va_res, kCols * 4u) ||
        !rt.download_from_vram(0, g_q8.data(), va_q8, kCols) ||
        !rt.download_from_vram(0, g_sc.data(), va_sc, kBlocks * 4u) ||
        !rt.download_from_vram(0, g_aq8.data(), va_aq8, kCols) ||
        !rt.download_from_vram(0, g_asc.data(), va_asc, kBlocks * 4u)) {
        std::fprintf(stderr, "FAIL: download_from_vram failed\n");
        return 1;
    }
    if (gpu_status != 0u) {
        std::fprintf(stderr, "FAIL: VM status=%u (expected 0)\n", gpu_status);
        return 1;
    }

    // CPU reference via vegajet::vm::run_cpu
    std::vector<float> c_rms(kCols), c_res = h_res, c_sc(kBlocks), c_asc(kBlocks);
    std::vector<int8_t> c_q8(kCols), c_aq8(kCols);
    std::vector<VjVmOpDesc> cprog(3);
    cprog[0] = d_rms;
    cprog[0].ptr0 = reinterpret_cast<uint64_t>(h_x.data());
    cprog[0].ptr1 = reinterpret_cast<uint64_t>(h_w.data());
    cprog[0].ptr2 = reinterpret_cast<uint64_t>(c_rms.data());
    cprog[1] = d_ar;
    cprog[1].ptr0 = reinterpret_cast<uint64_t>(h_x.data());
    cprog[1].ptr1 = reinterpret_cast<uint64_t>(c_res.data());
    cprog[1].ptr2 = reinterpret_cast<uint64_t>(h_w.data());
    cprog[1].ptr3 = reinterpret_cast<uint64_t>(c_q8.data());
    cprog[1].ptr4 = reinterpret_cast<uint64_t>(c_sc.data());
    cprog[2] = d_act;
    cprog[2].ptr0 = reinterpret_cast<uint64_t>(h_gate.data());
    cprog[2].ptr1 = reinterpret_cast<uint64_t>(h_up.data());
    cprog[2].ptr2 = reinterpret_cast<uint64_t>(c_aq8.data());
    cprog[2].ptr3 = reinterpret_cast<uint64_t>(c_asc.data());
    if (!run_cpu(cprog, &err)) {
        std::fprintf(stderr, "FAIL: run_cpu: %s\n", err.c_str());
        return 1;
    }

    float max_rms_err = 0.0f, max_res_err = 0.0f, max_sc_err = 0.0f;
    int max_q8_diff = 0;
    for (uint32_t i = 0; i < kCols; ++i) {
        max_rms_err = std::max(max_rms_err, std::fabs(g_rms[i] - c_rms[i]));
        max_res_err = std::max(max_res_err, std::fabs(g_res[i] - c_res[i]));
        max_q8_diff = std::max(max_q8_diff, std::abs(static_cast<int>(g_q8[i]) - static_cast<int>(c_q8[i])));
        max_q8_diff = std::max(max_q8_diff, std::abs(static_cast<int>(g_aq8[i]) - static_cast<int>(c_aq8[i])));
    }
    for (uint32_t b = 0; b < kBlocks; ++b) {
        max_sc_err = std::max(max_sc_err, std::fabs(g_sc[b] - c_sc[b]));
        max_sc_err = std::max(max_sc_err, std::fabs(g_asc[b] - c_asc[b]));
    }
    std::printf("[UNIVERSAL_VM_GPU] 3 kernels PASS (status=0) | max_rms_err=%.3e max_res_err=%.3e max_sc_err=%.3e max_q8_diff=%d\n",
                max_rms_err, max_res_err, max_sc_err, max_q8_diff);
    if (max_rms_err > 1e-5f || max_res_err > 1e-6f || max_sc_err > 1e-5f || max_q8_diff > 1) {
        std::fprintf(stderr, "FAIL: GPU vs CPU parity exceeded tolerance\n");
        return 1;
    }
    rt.shutdown();
    return 0;
}
