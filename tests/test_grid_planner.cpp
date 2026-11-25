// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// MessagePack .amdhsa.kernels ELF note parser, private_segment_fixed_size == 0 gate, gfx906 occupancy and
// 60-CU vs 64-CU 2D Split-K grid selection. The code object is synthesized in-memory (no GPU toolchain).
#include "core/grid_planner.hpp"
#include "tests/test_common.hpp"

using namespace vegajet::planner;

namespace {

struct MpW {
    std::vector<uint8_t> b;
    void fixmap(uint8_t n) { b.push_back(0x80 | n); }
    void fixarr(uint8_t n) { b.push_back(0x90 | n); }
    void str(const std::string& s) {
        if (s.size() < 32) b.push_back(0xA0 | static_cast<uint8_t>(s.size()));
        else { b.push_back(0xD9); b.push_back(static_cast<uint8_t>(s.size())); }
        b.insert(b.end(), s.begin(), s.end());
    }
    void u(uint64_t v) {
        if (v < 128) b.push_back(static_cast<uint8_t>(v));
        else if (v < 65536) { b.push_back(0xCD); b.push_back(uint8_t(v >> 8)); b.push_back(uint8_t(v)); }
        else { b.push_back(0xCE); for (int s = 24; s >= 0; s -= 8) b.push_back(uint8_t(v >> s)); }
    }
};

void kernel(MpW& m, const std::string& sym, uint32_t vgpr, uint32_t lds, uint32_t priv) {
    m.fixmap(9);
    m.str(".symbol"); m.str(sym + ".kd");
    m.str(".vgpr_count"); m.u(vgpr);
    m.str(".sgpr_count"); m.u(24);
    m.str(".group_segment_fixed_size"); m.u(lds);
    m.str(".private_segment_fixed_size"); m.u(priv);
    m.str(".kernarg_segment_size"); m.u(272);
    m.str(".wavefront_size"); m.u(64);
    m.str(".max_flat_workgroup_size"); m.u(256);
    m.str(".args"); m.fixarr(1); m.fixmap(2); m.str(".size"); m.u(8); m.str(".value_kind"); m.str("global_buffer");
}

std::vector<uint8_t> make_elf(const std::vector<uint8_t>& meta) {
    // note: namesz, descsz, type, "AMDGPU\0" padded to 8, desc padded to 4
    std::vector<uint8_t> note;
    auto u32 = [&](std::vector<uint8_t>& v, uint32_t x) { for (int i = 0; i < 4; ++i) v.push_back(uint8_t(x >> (8 * i))); };
    u32(note, 7); u32(note, static_cast<uint32_t>(meta.size())); u32(note, 32);
    const char name[8] = {'A', 'M', 'D', 'G', 'P', 'U', 0, 0};
    note.insert(note.end(), name, name + 8);
    note.insert(note.end(), meta.begin(), meta.end());
    while (note.size() % 4) note.push_back(0);

    std::vector<uint8_t> elf(64, 0);
    elf[0] = 0x7f; elf[1] = 'E'; elf[2] = 'L'; elf[3] = 'F'; elf[4] = 2; elf[5] = 1; elf[6] = 1;
    elf[16] = 3;                    // ET_DYN
    elf[18] = 224;                  // EM_AMDGPU
    const uint64_t note_off = 64;
    elf.insert(elf.end(), note.begin(), note.end());
    while (elf.size() % 8) elf.push_back(0);
    const uint64_t shoff = elf.size();
    std::vector<uint8_t> sh(128, 0);  // [0] null, [1] note
    uint32_t type = 7;
    std::memcpy(&sh[64 + 4], &type, 4);
    std::memcpy(&sh[64 + 24], &note_off, 8);
    const uint64_t nsz = note.size();
    std::memcpy(&sh[64 + 32], &nsz, 8);
    elf.insert(elf.end(), sh.begin(), sh.end());
    std::memcpy(&elf[40], &shoff, 8);
    const uint16_t shentsize = 64, shnum = 2;
    std::memcpy(&elf[58], &shentsize, 2);
    std::memcpy(&elf[60], &shnum, 2);
    return elf;
}

std::vector<uint8_t> metadata(bool with_spill) {
    MpW m;
    m.fixmap(3);
    m.str("amdhsa.version"); m.fixarr(2); m.u(1); m.u(2);
    m.str("amdhsa.target"); m.str("amdgcn-amd-amdhsa--gfx906");
    m.str("amdhsa.kernels");
    m.fixarr(with_spill ? 2 : 1);
    kernel(m, "vj_gemv_q8_dp4a", 32, 4096, 0);
    if (with_spill) kernel(m, "vj_bad_spill", 256, 70000, 16);
    return m.b;
}

} // namespace

int main() {
    auto elf = make_elf(metadata(false));
    auto co = parse_code_object(elf.data(), elf.size());
    VJ_CHECK(co.ok && co.target == "amdgcn-amd-amdhsa--gfx906" && co.kernels.size() == 1);
    const KernelMeta* k = co.find("vj_gemv_q8_dp4a");
    VJ_CHECK(k && k->vgpr_count == 32 && k->sgpr_count == 24 && k->group_segment_fixed_size == 4096 &&
             k->private_segment_fixed_size == 0 && k->kernarg_segment_size == 272 && k->wavefront_size == 64 &&
             k->max_flat_workgroup_size == 256);
    std::string err;
    VJ_CHECK(validate_no_scratch(co, &err));

    auto elf2 = make_elf(metadata(true));
    auto co2 = parse_code_object(elf2.data(), elf2.size());
    VJ_CHECK(co2.ok && co2.kernels.size() == 2 && co2.find("vj_bad_spill")->group_segment_fixed_size == 70000);
    VJ_CHECK(!validate_no_scratch(co2, &err) && err.find("vj_bad_spill=16B") != std::string::npos);

    // Robustness: truncated metadata, wrong machine, non-ELF.
    auto meta = metadata(false);
    meta.resize(meta.size() / 2);
    auto elf3 = make_elf(meta);
    VJ_CHECK(!parse_code_object(elf3.data(), elf3.size()).ok);
    auto elf4 = elf;
    elf4[18] = 62;  // x86-64
    VJ_CHECK(!parse_code_object(elf4.data(), elf4.size()).ok);
    const uint8_t junk[80] = {1, 2, 3};
    VJ_CHECK(!parse_code_object(junk, sizeof(junk)).ok);
    std::printf("[PASS] MessagePack .amdhsa.kernels note parser + private_segment_fixed_size==0 gate (%s)\n", err.c_str());

    // Occupancy.
    auto o = compute_gfx906_occupancy(32, 24, 4096, 256, 60);
    VJ_CHECK(o.waves_per_simd_vgpr == 8 && o.waves_per_simd == 8 && o.resident_wgs_per_cu == 8 && o.resident_wgs_total == 480);
    VJ_CHECK(compute_gfx906_occupancy(128, 24, 0, 64, 60).waves_per_simd == 2);
    VJ_CHECK(compute_gfx906_occupancy(16, 24, 32768, 64, 64).wgs_per_cu_lds == 2);

    // 2D Split-K: 128 row-workgroups leave a 60-CU part 71% busy (3 waves for 2.13), a 64-CU part 100%.
    auto p60 = plan_gemv_splitk(512, 3072, 60, 4);
    auto p64 = plan_gemv_splitk(512, 3072, 64, 4);
    VJ_CHECK(p60.wgs_1d == 128 && p60.split_k > 1 && p60.cu_eff_splitk > p60.cu_eff_1d && p60.k_chunk_elems % 32 == 0);
    VJ_CHECK(p60.k_chunk_elems * p60.split_k >= 3072);
    VJ_CHECK(p64.split_k == 1 && p64.cu_eff_1d == 1.0);
    // Large N already fills both parts; short K never splits.
    VJ_CHECK(plan_gemv_splitk(248320, 3072, 60).split_k == 1);
    VJ_CHECK(plan_gemv_splitk(64, 256, 60).split_k == 1);
    std::printf("[PASS] Split-K N=512 K=3072: 60 CU -> split %u (eff %.3f -> %.3f, x%.2f), 64 CU -> split %u (eff %.3f)\n",
                p60.split_k, p60.cu_eff_1d, p60.cu_eff_splitk, p60.predicted_speedup, p64.split_k, p64.cu_eff_1d);

    // Resident-wave-aware Split-K: N=3072 K=4096 (384 WGs on 300 resident WGs -> 36% tail waste) engages S>=2.
    auto pr = plan_gemv_splitk_resident(3072, 4096, 60, 5, 8);
    VJ_CHECK(pr.wgs_1d == 384 && pr.split_k >= 2 && pr.cu_eff_splitk > pr.cu_eff_1d);
    auto pr_short = plan_gemv_splitk_resident(3072, 512, 60, 5, 16);
    VJ_CHECK(pr_short.split_k == 2 || pr_short.split_k == 1);
    auto pr_tall = plan_gemv_splitk_resident(10304, 3072, 60, 5, 8);
    VJ_CHECK(pr_tall.wgs_1d == 1288 && pr_tall.split_k == 1);
    std::printf("[PASS] Resident Split-K: [3072,4096] -> S=%u (eff %.3f -> %.3f), [10304,3072] -> S=%u\n",
                pr.split_k, pr.cu_eff_1d, pr.cu_eff_splitk, pr_tall.split_k);

    // Model-agnostic Split-KV FlashDecoding planner (generalizes across any n_q/n_kv ratio).
    auto f16 = plan_flash_splitkv(256, 16, 1, 60, 4);
    VJ_CHECK(f16.use_split_kv && f16.n_chunks >= 4 && f16.cu_util_splitkv >= 0.80);
    auto f8 = plan_flash_splitkv(512, 8, 2, 60, 4);
    VJ_CHECK(f8.use_split_kv && f8.cu_util_splitkv >= 0.80);
    auto f64 = plan_flash_splitkv(512, 64, 8, 60, 4);
    VJ_CHECK(!f64.use_split_kv && f64.cu_util_1d == 1.0);
    std::printf("[PASS] Split-KV Flash planner: 16:1@256 -> %u chunks (CU util %.0f%% -> %.0f%%), 8:2@512 -> %u chunks (%.0f%%)\n",
                f16.n_chunks, f16.cu_util_1d * 100.0, f16.cu_util_splitkv * 100.0, f8.n_chunks, f8.cu_util_splitkv * 100.0);
    return 0;
}
