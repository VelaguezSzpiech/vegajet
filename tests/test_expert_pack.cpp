// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// core/expert_pack: format round trip through a pack written to a temp file, 4 KiB alignment of the index and
// every record, O_DIRECT-eligible record sizes, the io_uring and pread paths returning identical bytes (options and
// VJ_IO_URING=0), the unaligned-destination bounce path, and rejection of corrupt / truncated / trailing-byte packs.
#include "core/expert_pack.hpp"
#include "tests/test_common.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

using namespace vegajet::expertpack;

namespace {

constexpr PackDesc kDesc{4, 6, 64, 48, 32};
constexpr size_t kStride = size_t(kDesc.gate_elems + kDesc.up_elems + kDesc.down_elems);

std::vector<float> synthetic(uint64_t seed) {
    std::vector<float> v(kStride * kDesc.n_layers * kDesc.n_experts);
    uint64_t s = seed;
    for (auto& x : v) {
        s ^= s >> 33;
        s *= 0xFF51AFD7ED558CCDull;
        s ^= s >> 33;
        x = static_cast<float>(double(s >> 11) / double(1ull << 53) * 2.0 - 1.0);
    }
    return v;
}

std::string temp_path() {
    char tmpl[] = "./vj-expert-pack-XXXXXX";
    const int fd = ::mkstemp(tmpl);
    VJ_CHECK(fd >= 0);
    ::close(fd);
    return tmpl;
}

std::vector<uint8_t> read_file(const std::string& p) {
    std::FILE* f = std::fopen(p.c_str(), "rb");
    VJ_CHECK(f != nullptr);
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> b(static_cast<size_t>(n));
    VJ_CHECK(std::fread(b.data(), 1, b.size(), f) == b.size());
    std::fclose(f);
    return b;
}

void write_file(const std::string& p, const std::vector<uint8_t>& b) {
    std::FILE* f = std::fopen(p.c_str(), "wb");
    VJ_CHECK(f != nullptr);
    VJ_CHECK(std::fwrite(b.data(), 1, b.size(), f) == b.size());
    std::fclose(f);
}

bool rejects(const std::string& path) {
    PackReader r;
    std::string err;
    return !r.open(path, &err) && !err.empty();
}

}  // namespace

int main() {
    const std::vector<float> src = synthetic(0xABCDEF);
    const std::string path = temp_path();
    std::string err;
    VJ_CHECK(write_pack(path, kDesc, src.data(), &err));

    // ---- geometry: every record and index entry is 4 KiB aligned and O_DIRECT-eligible.
    VJ_CHECK(record_bytes(kDesc) == round_up(kStride * 4, kAlign));
    VJ_CHECK(record_bytes(kDesc) % kAlign == 0);
    VJ_CHECK(index_bytes(kDesc) == uint64_t(kDesc.n_layers) * kDesc.n_experts * kIndexEntryBytes);
    {
        PackReader r;
        VJ_CHECK(r.open(path, &err));
        const auto& d = r.desc();
        VJ_CHECK(d.n_layers == kDesc.n_layers && d.n_experts == kDesc.n_experts);
        VJ_CHECK(d.gate_elems == kDesc.gate_elems && d.up_elems == kDesc.up_elems && d.down_elems == kDesc.down_elems);
        uint64_t prev = 0;
        for (uint32_t l = 0; l < d.n_layers; ++l)
            for (uint32_t e = 0; e < d.n_experts; ++e) {
                const uint64_t off = r.expert_offset(l, e);
                VJ_CHECK(off % kAlign == 0);
                VJ_CHECK(off >= kHeaderBytes + index_bytes(kDesc));
                if (l || e) VJ_CHECK(off == prev + record_bytes(kDesc));
                prev = off;
            }
        VJ_CHECK(read_file(path).size() == size_t(prev + record_bytes(kDesc)));

        // ---- every record reads back byte-exactly, whichever path the reader picked.
        std::vector<float> g(d.gate_elems), u(d.up_elems), dn(d.down_elems);
        for (uint32_t l = 0; l < d.n_layers; ++l)
            for (uint32_t e = 0; e < d.n_experts; ++e) {
                VJ_CHECK(r.read_expert(l, e, g.data(), u.data(), dn.data(), &err));
                const float* want = &src[(size_t(l) * d.n_experts + e) * kStride];
                VJ_CHECK(std::memcmp(g.data(), want, size_t(d.gate_elems) * 4) == 0);
                VJ_CHECK(std::memcmp(u.data(), want + d.gate_elems, size_t(d.up_elems) * 4) == 0);
                VJ_CHECK(std::memcmp(dn.data(), want + d.gate_elems + d.up_elems, size_t(d.down_elems) * 4) == 0);
            }
        VJ_CHECK(r.stats().reads == uint64_t(d.n_layers) * d.n_experts + 2);  // header + index + records
        std::printf("  %s: %u records, record %llu B, mode %s\n", "pack", d.n_layers * d.n_experts,
                    static_cast<unsigned long long>(record_bytes(d)), r.io_mode());
        VJ_CHECK(r.stats().reads > 0);
        // A misaligned destination is served through the aligned bounce window.
        std::vector<uint8_t> raw(size_t(record_bytes(d)) + 3);
        VJ_CHECK(r.read_raw(1, 2, raw.data() + 1, raw.size() - 1, &err));
        const float* want = &src[(size_t(1) * d.n_experts + 2) * kStride];
        VJ_CHECK(std::memcmp(raw.data() + 1, want, kStride * 4) == 0);
        // Out-of-range ids and a too-small destination are refused.
        VJ_CHECK(!r.read_raw(d.n_layers, 0, raw.data(), raw.size(), &err));
        VJ_CHECK(!r.read_raw(0, d.n_experts, raw.data(), raw.size(), &err));
        VJ_CHECK(!r.read_raw(0, 0, raw.data(), 8, &err));
    }

    // ---- forced pread path (no io_uring) returns identical bytes.
    {
        PackOptions opt;
        opt.use_uring = false;
        PackReader r;
        VJ_CHECK(r.open(path, opt, &err));
        std::vector<float> g(kDesc.gate_elems), u(kDesc.up_elems), dn(kDesc.down_elems);
        VJ_CHECK(r.read_expert(3, 5, g.data(), u.data(), dn.data(), &err));
        const float* want = &src[(size_t(3) * kDesc.n_experts + 5) * kStride];
        VJ_CHECK(std::memcmp(g.data(), want, size_t(kDesc.gate_elems) * 4) == 0);
        VJ_CHECK(std::strstr(r.io_mode(), "pread") != nullptr);
        VJ_CHECK(r.stats().uring_reads == 0);
    }
    // ---- VJ_IO_URING=0 disables io_uring at runtime as well.
    {
        ::setenv("VJ_IO_URING", "0", 1);
        PackReader r;
        VJ_CHECK(r.open(path, &err));
        std::vector<float> g(kDesc.gate_elems), u(kDesc.up_elems), dn(kDesc.down_elems);
        VJ_CHECK(r.read_expert(0, 0, g.data(), u.data(), dn.data(), &err));
        VJ_CHECK(std::strstr(r.io_mode(), "pread") != nullptr && r.stats().uring_reads == 0);
        ::unsetenv("VJ_IO_URING");
    }
    // ---- O_DIRECT disabled: page-cache path still reads correctly.
    {
        PackOptions opt;
        opt.use_direct = false;
        PackReader r;
        VJ_CHECK(r.open(path, opt, &err));
        std::vector<float> g(kDesc.gate_elems), u(kDesc.up_elems), dn(kDesc.down_elems);
        VJ_CHECK(r.read_expert(2, 4, g.data(), u.data(), dn.data(), &err));
        const float* want = &src[(size_t(2) * kDesc.n_experts + 4) * kStride];
        VJ_CHECK(std::memcmp(dn.data(), want + kDesc.gate_elems + kDesc.up_elems, size_t(kDesc.down_elems) * 4) == 0);
        VJ_CHECK(std::strstr(r.io_mode(), "O_DIRECT") == nullptr);
    }

    // ---- automatic fallback: an unusable ring degrades the very next read to pread.
    {
        PackOptions opt;
        opt.simulate_ring_failure = true;
        PackReader r;
        VJ_CHECK(r.open(path, opt, &err));
        std::vector<float> g(kDesc.gate_elems), u(kDesc.up_elems), dn(kDesc.down_elems);
        VJ_CHECK(r.read_expert(2, 1, g.data(), u.data(), dn.data(), &err));
        const float* want = &src[(size_t(2) * kDesc.n_experts + 1) * kStride];
        VJ_CHECK(std::memcmp(g.data(), want, size_t(kDesc.gate_elems) * 4) == 0);
        if (r.stats().uring_failures > 0) {  // only when io_uring was available to break
            VJ_CHECK(r.stats().uring_reads == 0 && r.stats().pread_reads >= 1);
            VJ_CHECK(std::strstr(r.io_mode(), "pread") != nullptr);
        }
        std::printf("  ring failure -> %s (failures=%llu, pread=%llu)\n", r.io_mode(),
                    static_cast<unsigned long long>(r.stats().uring_failures),
                    static_cast<unsigned long long>(r.stats().pread_reads));
    }

    // ---- rejection: bad magic, truncated file, trailing bytes, bad record size.
    {
        const std::vector<uint8_t> good = read_file(path);
        std::vector<uint8_t> bad = good;
        bad[0] = 'X';
        const std::string p1 = temp_path();
        write_file(p1, bad);
        VJ_CHECK(rejects(p1));
        std::remove(p1.c_str());

        std::vector<uint8_t> trunc(good.begin(), good.begin() + static_cast<long>(good.size()) - 1);
        const std::string p2 = temp_path();
        write_file(p2, trunc);
        VJ_CHECK(rejects(p2));
        std::remove(p2.c_str());

        std::vector<uint8_t> longer = good;
        longer.push_back(0);
        const std::string p3 = temp_path();
        write_file(p3, longer);
        VJ_CHECK(rejects(p3));
        std::remove(p3.c_str());
    }
    {
        PackDesc bad = kDesc;
        bad.gate_elems = 0;
        const std::string p = temp_path();
        VJ_CHECK(!write_pack(p, bad, src.data(), &err));
        std::remove(p.c_str());
    }
    std::remove(path.c_str());
    std::printf("test_expert_pack: format, alignment, io_uring/pread fallback, bounce, rejection OK\nPASS\n");
    return 0;
}
