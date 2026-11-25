// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// core/paged_kv: block accounting, prefix-hash reuse across sequences (shared system prompt), multi-sequence
// sharing with refcounts, copy-on-write on fork / truncate, LRU eviction of cached blocks only, all-or-nothing
// append on OOM, hash-collision safety, and a randomized fuzz against a token-level reference model.
#include "core/paged_kv.hpp"
#include "tests/test_common.hpp"

#include <cstdio>
#include <map>
#include <numeric>
#include <random>

using namespace vegajet::paged;

namespace {
std::vector<int32_t> seq(int32_t start, uint32_t n) {
    std::vector<int32_t> v(n);
    std::iota(v.begin(), v.end(), start);
    return v;
}
} // namespace

int main() {
    // ---- prefix reuse: a released prompt is served from cache to the next request
    {
        PagedKv kv(64, 16);
        const auto prompt = seq(1000, 100);  // 6 full blocks + 4 tokens
        const SeqId a = kv.create();
        VJ_CHECK(kv.attach_prefix(a, prompt.data(), 100) == 0);
        VJ_CHECK(kv.append(a, prompt.data(), 100) && kv.block_table(a).size() == 7);
        const auto ta = kv.block_table(a);
        kv.release(a);
        auto st = kv.stats();
        VJ_CHECK(st.cached == 6 && st.free == 58 && st.used == 0 && kv.check_invariants());
        auto p2 = prompt;
        p2.push_back(7);
        const SeqId b = kv.create();
        VJ_CHECK(kv.attach_prefix(b, p2.data(), 101) == 96);
        for (int i = 0; i < 6; ++i) VJ_CHECK(kv.block_table(b)[i] == ta[i]);  // the same physical blocks
        VJ_CHECK(kv.append(b, p2.data() + 96, 5) && kv.length(b) == 101 && kv.check_invariants());
        // Concurrent sharing: a third sequence attaches while b is alive -> refcount 2 on the shared blocks.
        const SeqId c = kv.create();
        VJ_CHECK(kv.attach_prefix(c, prompt.data(), 100) == 96);
        st = kv.stats();
        VJ_CHECK(st.used == 6 + 1 && st.prefix_hit_tokens == 192 && kv.check_invariants());
        // Different prefix: no hit.
        const auto other = seq(5000, 64);
        const SeqId d = kv.create();
        VJ_CHECK(kv.attach_prefix(d, other.data(), 64) == 0);
        kv.release(b); kv.release(c); kv.release(d);
        VJ_CHECK(kv.check_invariants());
    }
    // ---- fork + truncate copy-on-write
    {
        PagedKv kv(32, 8);
        const SeqId a = kv.create();
        const auto t = seq(1, 20);  // 2 full + 4
        VJ_CHECK(kv.append(a, t.data(), 20));
        const SeqId f = kv.fork(a);
        auto cp = kv.take_copies();
        VJ_CHECK(cp.size() == 1 && cp[0].tokens == 4 && cp[0].src == kv.block_table(a)[2] && cp[0].dst == kv.block_table(f)[2]);
        VJ_CHECK(kv.block_table(f)[0] == kv.block_table(a)[0] && kv.block_table(f)[2] != kv.block_table(a)[2]);
        const int32_t x[] = {99, 98};
        VJ_CHECK(kv.append(f, x, 2) && kv.length(a) == 20 && kv.length(f) == 22 && kv.check_invariants());
        // Truncate into a shared full block -> COW copy of the kept part.
        kv.truncate(f, 11);
        cp = kv.take_copies();
        VJ_CHECK(cp.size() == 1 && cp[0].tokens == 3 && cp[0].src == kv.block_table(a)[1]);
        VJ_CHECK((kv.block_contents(kv.block_table(f)[1]) == std::vector<int32_t>{9, 10, 11}));
        VJ_CHECK(kv.append(f, x, 1) && kv.check_invariants());
        // Sole owner truncating its own sealed block: no copy, block unregistered in place.
        kv.release(f);
        kv.truncate(a, 12);
        VJ_CHECK(kv.take_copies().empty() && kv.length(a) == 12 && kv.check_invariants());
        kv.release(a);
        VJ_CHECK(kv.check_invariants());
    }
    // ---- eviction: cached (unreferenced) blocks go LRU-first; referenced blocks are never evicted; OOM is atomic
    {
        PagedKv kv(8, 4);
        const SeqId a = kv.create();
        const auto ta = seq(100, 16);  // 4 full blocks
        VJ_CHECK(kv.append(a, ta.data(), 16));
        kv.release(a);                 // 4 cached
        const SeqId b = kv.create();
        const auto tb = seq(200, 16);
        VJ_CHECK(kv.append(b, tb.data(), 16));  // uses the 4 free blocks
        const SeqId c = kv.create();
        const auto tc = seq(300, 8);
        VJ_CHECK(kv.append(c, tc.data(), 8));   // evicts 2 cached blocks (the LRU end = deepest blocks of a)
        auto st = kv.stats();
        VJ_CHECK(st.evictions == 2 && st.cached == 2 && kv.check_invariants());
        const SeqId r = kv.create();
        VJ_CHECK(kv.attach_prefix(r, ta.data(), 16) == 8);  // the root blocks of a survived (released tail first)
        kv.release(r);
        const auto big = seq(400, 40);
        const SeqId d = kv.create();
        const auto before = kv.stats();
        VJ_CHECK(!kv.append(d, big.data(), 40));            // needs 10 blocks: fails with no side effects
        const auto after = kv.stats();
        VJ_CHECK(after.free == before.free && after.cached == before.cached && kv.length(d) == 0 && kv.check_invariants());
        kv.release(b); kv.release(c); kv.release(d);
    }
    // ---- fuzz against a token-level reference
    {
        std::mt19937 rng(42);
        PagedKv kv(96, 8);
        std::map<SeqId, std::vector<int32_t>> ref;
        const auto sys_prompt = seq(1, 40);
        uint64_t ops = 0, attached = 0;
        for (int it = 0; it < 6000; ++it) {
            const uint32_t op = rng() % 10;
            if (op < 2 || ref.empty()) {
                const SeqId s = kv.create();
                ref[s] = {};
                if (rng() % 2) {  // shared system prompt, then a private suffix
                    const uint32_t got = kv.attach_prefix(s, sys_prompt.data(), 40);
                    attached += got;
                    ref[s].assign(sys_prompt.begin(), sys_prompt.begin() + got);
                    if (kv.append(s, sys_prompt.data() + got, 40 - got)) ref[s] = sys_prompt;
                }
            } else {
                auto itr = ref.begin();
                std::advance(itr, rng() % ref.size());
                const SeqId s = itr->first;
                if (op < 6) {
                    std::vector<int32_t> t(1 + rng() % 12);
                    for (auto& x : t) x = static_cast<int32_t>(rng() % 50);
                    if (kv.append(s, t.data(), static_cast<uint32_t>(t.size()))) itr->second.insert(itr->second.end(), t.begin(), t.end());
                } else if (op < 7) {
                    const uint32_t n = itr->second.empty() ? 0 : static_cast<uint32_t>(rng() % itr->second.size());
                    kv.truncate(s, n);
                    itr->second.resize(kv.length(s));  // COW may drop a tail only when memory is exhausted
                } else if (op < 8) {
                    const SeqId f = kv.fork(s);
                    ref[f] = std::vector<int32_t>(itr->second.begin(), itr->second.begin() + kv.length(f));
                } else {
                    kv.release(s);
                    ref.erase(itr);
                }
            }
            kv.take_copies();
            for (const auto& [s, toks] : ref) VJ_CHECK(kv.length(s) == toks.size());
            VJ_CHECK(kv.check_invariants());
            ++ops;
        }
        for (const auto& [s, toks] : ref) {  // contents of every block equal the reference tokens
            for (size_t i = 0; i < kv.block_table(s).size(); ++i) {
                const auto c = kv.block_contents(kv.block_table(s)[i]);
                for (size_t k = 0; k < c.size() && i * 8 + k < toks.size(); ++k) VJ_CHECK(c[k] == toks[i * 8 + k]);
            }
        }
        const auto st = kv.stats();
        std::printf("  fuzz: %llu ops, %llu prefix tokens reused, %llu evictions, %llu COW copies, live seqs %zu\n",
                    static_cast<unsigned long long>(ops), static_cast<unsigned long long>(attached),
                    static_cast<unsigned long long>(st.evictions), static_cast<unsigned long long>(st.cow_copies), ref.size());
        VJ_CHECK(attached > 0 && st.evictions > 0 && st.cow_copies > 0);
    }
    std::printf("test_paged_kv: prefix reuse, sharing, COW, LRU eviction, atomic OOM, fuzz OK\nPASS\n");
    return 0;
}
