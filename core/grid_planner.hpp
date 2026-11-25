// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Code-object introspection + CU-aware launch planning for gfx906.
//  * parse_code_object(): ELF64 reader for AMDGPU code objects; decodes the NT_AMDGPU_METADATA note
//    (MessagePack, `amdhsa.kernels`) into per-kernel register / segment / workgroup metadata.
//  * compute_gfx906_occupancy(): waves per SIMD from VGPR/SGPR/LDS usage.
//  * plan_gemv_splitk(): 2D (row x K-split) GEMV grid that fills all CUs (60-CU MI50 vs 64-CU MI60).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vegajet::planner {

struct KernelMeta {
    std::string name;
    uint32_t vgpr_count = 0;
    uint32_t sgpr_count = 0;
    uint32_t group_segment_fixed_size = 0;    // LDS bytes
    uint32_t private_segment_fixed_size = 0;  // scratch bytes per work-item (must be 0)
    uint32_t kernarg_segment_size = 0;
    uint32_t wavefront_size = 64;
    uint32_t max_flat_workgroup_size = 256;
};

struct CodeObjectInfo {
    bool ok = false;
    std::string error;
    std::string target;               // amdhsa.target, e.g. "amdgcn-amd-amdhsa--gfx906"
    std::vector<KernelMeta> kernels;
    const KernelMeta* find(const std::string& name) const;
};

CodeObjectInfo parse_code_object(const uint8_t* data, size_t size);
CodeObjectInfo parse_code_object_file(const std::string& path);
// Fails if any kernel spills to scratch (private_segment_fixed_size != 0); lists offenders in `err`.
bool validate_no_scratch(const CodeObjectInfo& co, std::string* err);

struct OccupancyInfo {
    uint32_t waves_per_simd_vgpr = 10, waves_per_simd_sgpr = 10, waves_per_simd = 10;
    uint32_t wgs_per_cu_lds = 40, resident_wgs_per_cu = 10, resident_wgs_total = 600;
};
OccupancyInfo compute_gfx906_occupancy(uint32_t vgpr, uint32_t sgpr, uint32_t lds_bytes, uint32_t wg_threads,
                                       uint32_t n_cu = 60);

struct SplitKPlan {
    uint32_t N = 0, K = 0, n_cu = 60, rows_per_wg = 4;
    uint32_t wgs_1d = 0, split_k = 1, total_wgs = 0;
    uint32_t k_chunk_elems = 0;  // multiple of 32
    double cu_eff_1d = 0.0, cu_eff_splitk = 0.0, predicted_speedup = 1.0;
};
SplitKPlan plan_gemv_splitk(uint32_t N, uint32_t K, uint32_t n_cu = 60, uint32_t rows_per_wg = 4,
                            double reduce_overhead_per_split = 0.012);

// Resident-wave-aware deterministic 2D Split-K planner (uses parsed .amdhsa.kernels ELF residency and
// live topology n_cu). Engages S in {2, 4, 8} (with K/S >= 256, multiple of 32) whenever
// wgs_1d < 4 * resident_wgs or tail-wave waste exceeds 25%.
SplitKPlan plan_gemv_splitk_resident(uint32_t N, uint32_t K, uint32_t n_cu, uint32_t resident_wgs_per_cu,
                                     uint32_t rows_per_wg = 8, double reduce_overhead_per_split = 0.015);

struct FlashSplitKvPlan {
    bool     use_split_kv = false;
    uint32_t n_q_local = 0, n_kv_local = 0, n_cu = 60;
    uint32_t n_chunks = 1, chunk_len = 0, total_wgs = 0;
    double   cu_util_1d = 0.0, cu_util_splitkv = 0.0;
};
// Automatically engages Split-KV FlashDecoding whenever n_q_local < n_cu and seqlen >= min_split_seq,
// generalizing to any n_q_local / n_kv_local GQA ratio. chunk_len is fixed by span buckets (independent
// of verify batch M) so M=1 decode and M<=16 verify share identical chunk boundaries.
FlashSplitKvPlan plan_flash_splitkv(uint32_t seqlen, uint32_t n_q_local, uint32_t n_kv_local,
                                    uint32_t n_cu = 60, uint32_t resident_wgs_per_cu = 4,
                                    uint32_t min_split_seq = 64, uint32_t max_chunks = 60);

} // namespace vegajet::planner
