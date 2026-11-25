// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/grid_planner.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>

namespace vegajet::planner {

namespace {

// Bounds-checked MessagePack reader (subset used by LLVM AMDHSA metadata v3-v5).
struct Mp {
    const uint8_t* p;
    size_t len;
    size_t pos = 0;
    bool bad = false;
    bool ok() const { return !bad && pos < len; }
    uint8_t u8() { if (pos >= len) { bad = true; return 0; } return p[pos++]; }
    uint64_t be(int n) { uint64_t v = 0; for (int i = 0; i < n; ++i) v = (v << 8) | u8(); return v; }
    bool str(std::string& out) {
        const uint8_t b = u8();
        size_t n;
        if ((b & 0xE0u) == 0xA0u) n = b & 0x1Fu;
        else if (b == 0xD9u) n = be(1);
        else if (b == 0xDAu) n = be(2);
        else if (b == 0xDBu) n = be(4);
        else { bad = true; return false; }
        if (pos + n > len) { bad = true; return false; }
        out.assign(reinterpret_cast<const char*>(p + pos), n);
        pos += n;
        return true;
    }
    bool uint(uint64_t& v) {
        if (pos >= len) { bad = true; return false; }
        const uint8_t b = p[pos];
        if (b < 0x80u) { ++pos; v = b; return true; }
        switch (b) {
            case 0xCCu: ++pos; v = be(1); return !bad;
            case 0xCDu: ++pos; v = be(2); return !bad;
            case 0xCEu: ++pos; v = be(4); return !bad;
            case 0xCFu: ++pos; v = be(8); return !bad;
            case 0xD0u: ++pos; v = static_cast<uint64_t>(static_cast<int8_t>(be(1))); return !bad;
            case 0xD1u: ++pos; v = static_cast<uint64_t>(static_cast<int16_t>(be(2))); return !bad;
            case 0xD2u: ++pos; v = static_cast<uint64_t>(static_cast<int32_t>(be(4))); return !bad;
            case 0xD3u: ++pos; v = be(8); return !bad;
            default: skip(); return false;
        }
    }
    bool map_len(size_t& n) {
        const uint8_t b = u8();
        if ((b & 0xF0u) == 0x80u) { n = b & 0x0Fu; return true; }
        if (b == 0xDEu) { n = be(2); return !bad; }
        if (b == 0xDFu) { n = be(4); return !bad; }
        bad = true;
        return false;
    }
    bool arr_len(size_t& n) {
        const uint8_t b = u8();
        if ((b & 0xF0u) == 0x90u) { n = b & 0x0Fu; return true; }
        if (b == 0xDCu) { n = be(2); return !bad; }
        if (b == 0xDDu) { n = be(4); return !bad; }
        bad = true;
        return false;
    }
    void adv(size_t n) { if (pos + n > len) bad = true; else pos += n; }
    void skip(int depth = 0) {
        if (depth > 64) { bad = true; return; }
        const uint8_t b = u8();
        if (bad) return;
        if (b < 0x80u || b >= 0xE0u) return;
        if ((b & 0xE0u) == 0xA0u) { adv(b & 0x1Fu); return; }
        if ((b & 0xF0u) == 0x90u) { for (size_t i = 0, n = b & 0x0Fu; i < n && !bad; ++i) skip(depth + 1); return; }
        if ((b & 0xF0u) == 0x80u) { for (size_t i = 0, n = b & 0x0Fu; i < n && !bad; ++i) { skip(depth + 1); skip(depth + 1); } return; }
        switch (b) {
            case 0xC0u: case 0xC2u: case 0xC3u: return;
            case 0xC4u: case 0xD9u: adv(be(1)); return;
            case 0xC5u: case 0xDAu: adv(be(2)); return;
            case 0xC6u: case 0xDBu: adv(be(4)); return;
            case 0xCAu: case 0xCEu: case 0xD2u: adv(4); return;
            case 0xCBu: case 0xCFu: case 0xD3u: adv(8); return;
            case 0xCCu: case 0xD0u: adv(1); return;
            case 0xCDu: case 0xD1u: adv(2); return;
            case 0xDCu: { size_t n = be(2); for (size_t i = 0; i < n && !bad; ++i) skip(depth + 1); return; }
            case 0xDDu: { size_t n = be(4); for (size_t i = 0; i < n && !bad; ++i) skip(depth + 1); return; }
            case 0xDEu: { size_t n = be(2); for (size_t i = 0; i < n && !bad; ++i) { skip(depth + 1); skip(depth + 1); } return; }
            case 0xDFu: { size_t n = be(4); for (size_t i = 0; i < n && !bad; ++i) { skip(depth + 1); skip(depth + 1); } return; }
            default: bad = true; return;
        }
    }
};

void parse_metadata(const uint8_t* d, size_t n, CodeObjectInfo& co) {
    Mp r{d, n};
    size_t top = 0;
    if (!r.map_len(top)) { co.error = "metadata is not a MessagePack map"; return; }
    for (size_t i = 0; i < top && r.ok(); ++i) {
        std::string key;
        if (!r.str(key)) break;
        if (key == "amdhsa.target") {
            r.str(co.target);
        } else if (key == "amdhsa.kernels") {
            size_t nk = 0;
            if (!r.arr_len(nk)) break;
            for (size_t k = 0; k < nk && r.ok(); ++k) {
                size_t nf = 0;
                if (!r.map_len(nf)) break;
                KernelMeta km;
                std::string sym;
                for (size_t f = 0; f < nf && r.ok(); ++f) {
                    std::string fk;
                    if (!r.str(fk)) break;
                    uint64_t v = 0;
                    if (fk == ".name") r.str(km.name);
                    else if (fk == ".symbol") r.str(sym);
                    else if (fk == ".vgpr_count") { if (r.uint(v)) km.vgpr_count = static_cast<uint32_t>(v); }
                    else if (fk == ".sgpr_count") { if (r.uint(v)) km.sgpr_count = static_cast<uint32_t>(v); }
                    else if (fk == ".group_segment_fixed_size") { if (r.uint(v)) km.group_segment_fixed_size = static_cast<uint32_t>(v); }
                    else if (fk == ".private_segment_fixed_size") { if (r.uint(v)) km.private_segment_fixed_size = static_cast<uint32_t>(v); }
                    else if (fk == ".kernarg_segment_size") { if (r.uint(v)) km.kernarg_segment_size = static_cast<uint32_t>(v); }
                    else if (fk == ".wavefront_size") { if (r.uint(v)) km.wavefront_size = static_cast<uint32_t>(v); }
                    else if (fk == ".max_flat_workgroup_size") { if (r.uint(v)) km.max_flat_workgroup_size = static_cast<uint32_t>(v); }
                    else r.skip();
                }
                if (km.name.empty() && sym.size() > 3 && sym.compare(sym.size() - 3, 3, ".kd") == 0)
                    km.name = sym.substr(0, sym.size() - 3);
                if (!km.name.empty()) co.kernels.push_back(km);
            }
        } else {
            r.skip();
        }
    }
    if (r.bad) co.error = "truncated or malformed MessagePack metadata";
}

void parse_notes(const uint8_t* d, size_t n, CodeObjectInfo& co, bool& found) {
    size_t off = 0;
    while (off + 12 <= n) {
        uint32_t namesz, descsz, type;
        std::memcpy(&namesz, d + off, 4);
        std::memcpy(&descsz, d + off + 4, 4);
        std::memcpy(&type, d + off + 8, 4);
        const size_t name_off = off + 12;
        const size_t desc_off = name_off + ((namesz + 3u) & ~3u);
        const size_t next = desc_off + ((descsz + 3u) & ~3u);
        if (next > n || desc_off + descsz > n) break;
        if (type == 32 && namesz >= 6 && std::memcmp(d + name_off, "AMDGPU", 6) == 0) {  // NT_AMDGPU_METADATA
            found = true;
            parse_metadata(d + desc_off, descsz, co);
        }
        off = next;
    }
}

} // namespace

const KernelMeta* CodeObjectInfo::find(const std::string& name) const {
    for (const auto& k : kernels) if (k.name == name) return &k;
    return nullptr;
}

CodeObjectInfo parse_code_object(const uint8_t* d, size_t n) {
    CodeObjectInfo co;
    if (n < 64 || std::memcmp(d, "\x7f" "ELF", 4) != 0) { co.error = "not an ELF file"; return co; }
    if (d[4] != 2 || d[5] != 1) { co.error = "expected little-endian ELF64"; return co; }
    uint16_t machine;
    std::memcpy(&machine, d + 18, 2);
    if (machine != 224) { co.error = "ELF machine is not EM_AMDGPU (224)"; return co; }
    uint64_t shoff;
    uint16_t shentsize, shnum;
    std::memcpy(&shoff, d + 40, 8);
    std::memcpy(&shentsize, d + 58, 2);
    std::memcpy(&shnum, d + 60, 2);
    if (shentsize < 64 || shoff + uint64_t(shnum) * shentsize > n) { co.error = "bad section header table"; return co; }
    bool found = false;
    for (uint16_t i = 0; i < shnum; ++i) {
        const uint8_t* sh = d + shoff + uint64_t(i) * shentsize;
        uint32_t type;
        uint64_t off, size;
        std::memcpy(&type, sh + 4, 4);
        std::memcpy(&off, sh + 24, 8);
        std::memcpy(&size, sh + 32, 8);
        if (type != 7) continue;  // SHT_NOTE
        if (off + size > n) { co.error = "note section out of bounds"; return co; }
        parse_notes(d + off, size, co, found);
    }
    if (!found && co.error.empty()) co.error = "no NT_AMDGPU_METADATA note";
    co.ok = co.error.empty();
    return co;
}

CodeObjectInfo parse_code_object_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { CodeObjectInfo co; co.error = "cannot open " + path; return co; }
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return parse_code_object(bytes.data(), bytes.size());
}

bool validate_no_scratch(const CodeObjectInfo& co, std::string* err) {
    std::string bad;
    for (const auto& k : co.kernels)
        if (k.private_segment_fixed_size != 0)
            bad += (bad.empty() ? "" : ", ") + k.name + "=" + std::to_string(k.private_segment_fixed_size) + "B";
    if (!bad.empty() && err) *err = "kernels use scratch (private_segment_fixed_size != 0): " + bad;
    return co.ok && bad.empty();
}

static uint32_t align_up(uint32_t v, uint32_t a) { return (v + a - 1u) & ~(a - 1u); }

OccupancyInfo compute_gfx906_occupancy(uint32_t vgpr, uint32_t sgpr, uint32_t lds, uint32_t wg_threads, uint32_t n_cu) {
    OccupancyInfo o;
    o.waves_per_simd_vgpr = std::min<uint32_t>(10, 256 / align_up(std::max<uint32_t>(4, vgpr), 4));
    o.waves_per_simd_sgpr = std::min<uint32_t>(10, 800 / align_up(std::max<uint32_t>(16, sgpr + 6), 16));
    o.waves_per_simd = std::min(o.waves_per_simd_vgpr, o.waves_per_simd_sgpr);
    const uint32_t waves_per_wg = std::max<uint32_t>(1, (wg_threads + 63) / 64);
    const uint32_t by_waves = o.waves_per_simd * 4 / waves_per_wg;
    o.wgs_per_cu_lds = lds == 0 ? 40 : 65536 / align_up(lds, 256);
    o.resident_wgs_per_cu = std::min(by_waves, o.wgs_per_cu_lds);
    o.resident_wgs_total = o.resident_wgs_per_cu * n_cu;
    return o;
}

static double cu_eff(uint32_t wgs, uint32_t n_cu) {
    if (!wgs || !n_cu) return 0.0;
    const uint32_t waves = (wgs + n_cu - 1) / n_cu;
    return static_cast<double>(wgs) / (static_cast<double>(waves) * n_cu);
}

SplitKPlan plan_gemv_splitk(uint32_t N, uint32_t K, uint32_t n_cu, uint32_t rows_per_wg, double overhead) {
    SplitKPlan p;
    p.N = N; p.K = K;
    p.n_cu = std::max<uint32_t>(1, n_cu);
    p.rows_per_wg = std::max<uint32_t>(1, rows_per_wg);
    p.wgs_1d = (N + p.rows_per_wg - 1) / p.rows_per_wg;
    p.cu_eff_1d = p.cu_eff_splitk = cu_eff(p.wgs_1d, p.n_cu);
    p.total_wgs = p.wgs_1d;
    p.k_chunk_elems = K;
    if (K < 512 || p.wgs_1d == 0) return p;
    double best = p.cu_eff_1d;
    const uint32_t max_s = std::min<uint32_t>(16, K / 256);
    for (uint32_t s = 2; s <= max_s; ++s) {
        if ((K / s) / 32 * 32 < 256) continue;
        const uint32_t wgs = p.wgs_1d * s;
        const double eff = cu_eff(wgs, p.n_cu);
        const double score = eff / (1.0 + overhead * (s - 1));
        if (score > best + 1e-4) {
            best = score;
            p.split_k = s;
            p.total_wgs = wgs;
            p.k_chunk_elems = align_up((K + s - 1) / s, 32);
            p.cu_eff_splitk = eff;
            p.predicted_speedup = score / std::max(1e-6, p.cu_eff_1d);
        }
    }
    return p;
}

SplitKPlan plan_gemv_splitk_resident(uint32_t N, uint32_t K, uint32_t n_cu, uint32_t resident_wgs_per_cu,
                                     uint32_t rows_per_wg, double overhead) {
    SplitKPlan p;
    p.N = N; p.K = K;
    p.n_cu = std::max<uint32_t>(1, n_cu);
    p.rows_per_wg = std::max<uint32_t>(1, rows_per_wg);
    p.wgs_1d = (N + p.rows_per_wg - 1u) / p.rows_per_wg;
    const uint32_t res_wgs = p.n_cu * std::max<uint32_t>(1, resident_wgs_per_cu);
    p.cu_eff_1d = p.cu_eff_splitk = cu_eff(p.wgs_1d, res_wgs);
    p.total_wgs = p.wgs_1d;
    p.k_chunk_elems = K;
    if (K < 512u || p.wgs_1d == 0u) return p;

    const double tail_waste_1d = 1.0 - p.cu_eff_1d;
    const bool underfilled = (p.wgs_1d < 2u * res_wgs) || (tail_waste_1d > 0.25);
    if (!underfilled) return p;

    double best = p.cu_eff_1d;
    const uint32_t kCands[3] = {2u, 4u, 8u};
    for (uint32_t s : kCands) {
        if (K < s * 256u) continue;
        if ((K / s) < 256u || ((K / s) % 32u) != 0u) continue;
        const uint32_t wgs = p.wgs_1d * s;
        const double eff = cu_eff(wgs, res_wgs);
        // Bonus when moving from <1 resident wave to >=1 resident wave (latency hiding jump).
        const double sat_bonus = (p.wgs_1d < res_wgs && wgs >= res_wgs) ? 1.15 : 1.0;
        const double score = (eff * sat_bonus) / (1.0 + overhead * (s - 1u));
        if (score > best + 1e-4) {
            best = score;
            p.split_k = s;
            p.total_wgs = wgs;
            p.k_chunk_elems = align_up((K + s - 1u) / s, 32u);
            p.cu_eff_splitk = eff;
            p.predicted_speedup = score / std::max(1e-6, p.cu_eff_1d);
        }
    }
    return p;
}

FlashSplitKvPlan plan_flash_splitkv(uint32_t seqlen, uint32_t n_q_local, uint32_t n_kv_local,
                                    uint32_t n_cu, uint32_t resident_wgs_per_cu,
                                    uint32_t min_split_seq, uint32_t max_chunks) {
    FlashSplitKvPlan p;
    p.n_q_local = n_q_local;
    p.n_kv_local = n_kv_local;
    p.n_cu = std::max<uint32_t>(1, n_cu);
    p.cu_util_1d = std::min(1.0, static_cast<double>(n_q_local) / p.n_cu);
    p.cu_util_splitkv = p.cu_util_1d;
    p.total_wgs = n_q_local;
    p.chunk_len = seqlen;
    if (n_q_local == 0u || n_kv_local == 0u || seqlen < min_split_seq || n_q_local >= p.n_cu) {
        return p;
    }
    // Fixed chunk_len buckets (powers of 2, multiple of 32) so M=1 decode and M<=16 verify share identical
    // chunk boundaries while keeping total WGs = n_q_local * n_chunks >= n_cu (and up to resident_wgs).
    const uint32_t target_chunks = std::max<uint32_t>(
        2u, std::min<uint32_t>(max_chunks, (p.n_cu * std::max<uint32_t>(1u, resident_wgs_per_cu) + n_q_local - 1u) / n_q_local));
    uint32_t cl = 32u;
    while ((seqlen + cl - 1u) / cl > target_chunks && cl < 512u) cl <<= 1;
    const uint32_t nc = std::max<uint32_t>(1u, std::min<uint32_t>(max_chunks, (seqlen + cl - 1u) / cl));
    if (nc <= 1u) return p;
    p.use_split_kv = true;
    p.chunk_len = cl;
    p.n_chunks = nc;
    p.total_wgs = n_q_local * nc;
    p.cu_util_splitkv = std::min(1.0, static_cast<double>(p.total_wgs) / p.n_cu);
    return p;
}

} // namespace vegajet::planner
