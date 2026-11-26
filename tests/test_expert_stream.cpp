// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// core/expert_stream: O(1) recency LRU over the staging slots (hit promotion, miss, evict order, intrusive-list
// invariants), the explicit decode-only policy (prefill bypass leaves the cache untouched), and the streamed expert
// load on a real pack — cache hits skip the pack read, prefill reads everything, and both return the same weights.
#include "core/expert_stream.hpp"
#include "tests/test_common.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

using namespace vegajet::expertstream;

namespace {

constexpr vegajet::expertpack::PackDesc kDesc{3, 8, 32, 32, 16};
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
    char tmpl[] = "./vj-expert-stream-XXXXXX";
    const int fd = ::mkstemp(tmpl);
    VJ_CHECK(fd >= 0);
    ::close(fd);
    return tmpl;
}

std::vector<float> expected(const std::vector<float>& src, uint32_t layer, uint32_t expert) {
    const float* p = &src[(size_t(layer) * kDesc.n_experts + expert) * kStride];
    return std::vector<float>(p, p + kStride);
}

}  // namespace

int main() {
    // ---- LRU: capacity 3, MRU -> LRU order, hit promotion, eviction of the least recently used key.
    {
        LruCache c(3);
        VJ_CHECK(c.capacity() == 3 && c.size() == 0 && c.check());
        VJ_CHECK(c.get(10) == -1 && c.stats().misses == 1);
        const int32_t a = c.admit(10), b = c.admit(11), d = c.admit(12);
        VJ_CHECK(a == 0 && b == 1 && d == 2 && c.check());
        VJ_CHECK((c.order() == std::vector<uint64_t>{12, 11, 10}));
        VJ_CHECK(c.get(10) == a && c.check());                    // 10 promoted to MRU
        VJ_CHECK((c.order() == std::vector<uint64_t>{10, 12, 11}));
        VJ_CHECK(c.get(11) == b && (c.order() == std::vector<uint64_t>{11, 10, 12}));
        const int32_t e = c.admit(13);                            // evicts 12 (LRU)
        VJ_CHECK(e == d && c.check() && c.stats().evictions == 1);
        VJ_CHECK(!c.contains(12) && c.contains(13) && c.size() == 3);
        VJ_CHECK((c.order() == std::vector<uint64_t>{13, 11, 10}));
        VJ_CHECK(c.admit(11) == b && c.check());                  // admit of a resident key is a promotion
        VJ_CHECK((c.order() == std::vector<uint64_t>{11, 13, 10}));
        VJ_CHECK(c.stats().hits == 2 && c.stats().misses == 1);  // admit() of a resident key is a promotion, not a hit
        // Re-admitting an evicted key evicts the current LRU.
        VJ_CHECK(c.admit(12) == a && !c.contains(10) && c.check() && c.stats().evictions == 2);
        VJ_CHECK(c.get(999) == -1 && c.stats().misses == 2 && c.check());
        // Capacity 0 never stores anything.
        LruCache z(0);
        VJ_CHECK(z.admit(1) == -1 && z.get(1) == -1 && z.size() == 0 && z.check());
    }

    // ---- prefill bypass: no lookup, no insert, no eviction, order untouched.
    {
        LruCache c(2);
        c.admit(1);
        c.admit(2);
        const auto before = c.order();
        const auto ev = c.stats().evictions;
        c.set_phase(Phase::Prefill);
        VJ_CHECK(c.get(1) == -1 && c.admit(3) == -1 && !c.contains(3));
        VJ_CHECK(c.order() == before && c.size() == 2 && c.stats().evictions == ev && c.check());
        VJ_CHECK(c.stats().bypassed >= 2);
        c.set_phase(Phase::Decode);
        VJ_CHECK(c.get(1) >= 0);   // resident keys are still there after the prefill phase
    }

    // ---- ExpertStream over a real pack: decode hits skip the read, misses read once, prefill bypasses.
    const std::vector<float> src = synthetic(0x1234);
    const std::string path = temp_path();
    std::string err;
    VJ_CHECK(vegajet::expertpack::write_pack(path, kDesc, src.data(), &err));
    {
        ExpertStream s;
        VJ_CHECK(s.open(path, 2, &err));
        VJ_CHECK(s.is_open());
        std::vector<float> g(kDesc.gate_elems), u(kDesc.up_elems), dn(kDesc.down_elems);
        auto load = [&](uint32_t l, uint32_t e, std::vector<float>& out) {
            VJ_CHECK(s.expert(l, e, g.data(), u.data(), dn.data(), &err));
            out.clear();
            out.insert(out.end(), g.begin(), g.end());
            out.insert(out.end(), u.begin(), u.end());
            out.insert(out.end(), dn.begin(), dn.end());
        };
        std::vector<float> got;
        load(1, 3, got);
        VJ_CHECK(got == expected(src, 1, 3) && s.stats().pack_reads == 1 && s.stats().cache_misses == 1);
        load(1, 3, got);   // hit
        VJ_CHECK(got == expected(src, 1, 3) && s.stats().pack_reads == 1 && s.stats().cache_hits == 1);
        load(2, 5, got);
        load(0, 1, got);   // cache full: (1, 3) is the LRU key
        VJ_CHECK((s.cache().order() == std::vector<uint64_t>{1u, (uint64_t(2) << 32) | 5u}));
        VJ_CHECK(!s.cache().contains((uint64_t(1) << 32) | 3));
        load(1, 3, got);   // miss again: read and evict (2, 5)
        VJ_CHECK(got == expected(src, 1, 3) && s.stats().pack_reads == 4);
        VJ_CHECK(s.stats().cache_hits == 1);

        // Prefill: everything is read, nothing is promoted or evicted.
        s.set_phase(Phase::Prefill);
        const auto order = s.cache().order();
        const uint64_t reads = s.stats().pack_reads;
        for (uint32_t e = 0; e < kDesc.n_experts; ++e) {
            load(0, e, got);
            VJ_CHECK(got == expected(src, 0, e));
        }
        VJ_CHECK(s.stats().pack_reads == reads + kDesc.n_experts);
        VJ_CHECK(s.cache().order() == order && s.stats().cache_hits == 1);
        VJ_CHECK(s.cache().check());
        // Back to decode: the prefill working set did not flush the decode set.
        s.set_phase(Phase::Decode);
        VJ_CHECK(s.cache().contains((uint64_t(1) << 32) | 3));
        load(1, 3, got);
        VJ_CHECK(got == expected(src, 1, 3) && s.stats().cache_hits == 2);
        // Null buffers and out-of-range ids are errors, not crashes.
        VJ_CHECK(!s.expert(0, 0, nullptr, u.data(), dn.data(), &err));
        VJ_CHECK(!s.expert(kDesc.n_layers, 0, g.data(), u.data(), dn.data(), &err));
    }
    {
        // Capacity 0: decode still loads correctly, straight from the pack every time.
        ExpertStream s;
        VJ_CHECK(s.open(path, 0, &err));
        std::vector<float> g(kDesc.gate_elems), u(kDesc.up_elems), dn(kDesc.down_elems);
        VJ_CHECK(s.expert(2, 7, g.data(), u.data(), dn.data(), &err));
        const auto want = expected(src, 2, 7);
        VJ_CHECK(std::memcmp(g.data(), want.data(), g.size() * 4) == 0);
        VJ_CHECK(s.stats().cache_hits == 0 && s.stats().pack_reads == 1);
    }
    std::remove(path.c_str());
    std::printf("test_expert_stream: LRU order/evict, prefill bypass, streamed loads OK\nPASS\n");
    return 0;
}
