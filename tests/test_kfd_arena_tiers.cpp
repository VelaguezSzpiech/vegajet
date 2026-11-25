// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Hot-tensor VA clustering in KfdArena (CPU only, no device access):
//  * arena_chunk_sizes: exact 512 MiB chunks plus one tail, sums to the arena;
//  * untiered arenas allocate exactly like the original bump allocator (randomized, with quarantine);
//  * a tiered arena keeps every HOT allocation contiguous inside the first N whole chunks and every COLD one above,
//    with zero overlap, the tier boundary on a chunk boundary and the chunk list untouched;
//  * spills (HOT -> COLD, COLD -> HOT), exhaustion, quarantine inside the HOT tier, mark/rollback, reset,
//    ArenaTierScope and the VEGAJET_VA_CLUSTER / VEGAJET_VA_HOT_MIB parser.
#include "core/kfd_runtime.hpp"
#include "tests/test_common.hpp"

#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

using namespace vegajet::kfd;

namespace {

constexpr uint64_t MiB = 1ULL << 20;
constexpr uint64_t GiB = 1ULL << 30;
constexpr uint64_t kBase = 0x0000100000000000ULL;  // fake 2 MiB-aligned GPU VA

KfdArena make_arena(uint64_t bytes) {
    KfdArena a;
    a.slab.va = kBase;
    a.slab.size = bytes;
    uint64_t off = 0;
    for (uint64_t c : arena_chunk_sizes(bytes)) {
        KfdBuffer b;
        b.va = kBase + off;
        b.size = c;
        a.chunks.push_back(b);
        off += c;
    }
    return a;
}

// The pre-tiering KfdArena::alloc, verbatim, as the reference for the untiered path.
struct LegacyArena {
    uint64_t va = 0, size = 0, offset = 0;
    std::vector<std::pair<uint64_t, uint64_t>> quarantine;
    static uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1ULL) & ~(a - 1ULL); }
    uint64_t alloc(uint64_t bytes, uint64_t align) {
        uint64_t start = align_up(offset, align);
        for (bool moved = true; moved;) {
            moved = false;
            for (const auto& q : quarantine)
                if (start < q.second && start + bytes > q.first) { start = align_up(q.second, align); moved = true; }
        }
        if (start + bytes > size) return 0;
        offset = start + bytes;
        return va + start;
    }
};

struct Block { uint64_t begin, end; Tier asked; };

void check_no_overlap(std::vector<Block> v) {
    std::sort(v.begin(), v.end(), [](const Block& x, const Block& y) { return x.begin < y.begin; });
    for (size_t i = 1; i < v.size(); ++i) VJ_CHECK(v[i - 1].end <= v[i].begin);
}

void test_chunk_sizes() {
    const auto c = arena_chunk_sizes(61ULL * 512ULL * MiB);  // the 30.5 GiB default arena
    VJ_CHECK(c.size() == 61);
    for (uint64_t x : c) VJ_CHECK(x == kArenaChunkBytes);
    const uint64_t odd = 31ULL * GiB - 6ULL * MiB;  // not a chunk multiple: one tail chunk
    const auto d = arena_chunk_sizes(odd);
    uint64_t sum = 0;
    for (size_t i = 0; i < d.size(); ++i) {
        sum += d[i];
        if (i + 1 < d.size()) VJ_CHECK(d[i] == kArenaChunkBytes);
    }
    VJ_CHECK(sum == odd && d.back() == odd % kArenaChunkBytes);
    VJ_CHECK(arena_chunk_sizes(0).empty() && arena_chunk_sizes(5, 0).empty());
}

void test_untiered_matches_legacy() {
    std::mt19937_64 rng(42);
    for (int round = 0; round < 50; ++round) {
        KfdArena a = make_arena(4ULL * GiB);
        LegacyArena l{kBase, 4ULL * GiB, 0, {}};
        if (round % 2) {
            a.quarantine = {{100 * MiB, 120 * MiB}, {1 * GiB + 3 * MiB, 1 * GiB + 9 * MiB}};
            l.quarantine = a.quarantine;
        }
        for (int i = 0; i < 400; ++i) {
            const uint64_t bytes = 1 + rng() % (64 * MiB);
            const uint64_t align = 1ULL << (rng() % 13);  // 1 B .. 4 KiB
            const Tier t = (rng() & 1) ? Tier::HOT : Tier::COLD;  // ignored while untiered
            VJ_CHECK(a.alloc(bytes, align, t) == l.alloc(bytes, align));
            VJ_CHECK(a.used() == l.offset);
        }
        VJ_CHECK(a.hot_spill_bytes == 0 && a.cold_spill_bytes == 0 && !a.tiered());
    }
}

void test_tiered_layout() {
    KfdArena a = make_arena(61ULL * 512ULL * MiB);
    const auto chunks_before = a.chunks;
    VJ_CHECK(a.enable_tiers(2150ULL * MiB));  // ~2.1 GiB requested -> 5 whole chunks
    VJ_CHECK(a.hot_limit == kDefaultHotChunks * kArenaChunkBytes && a.hot_limit % kArenaChunkBytes == 0);
    VJ_CHECK(a.tiered() && a.default_tier == Tier::HOT && a.used() == 0);
    VJ_CHECK(a.chunks[kDefaultHotChunks].va == kBase + a.hot_limit);  // boundary is a BO chunk boundary

    // Ornith-like load order: per layer the dense trunk (default tier = HOT) interleaved with routed experts (COLD),
    // then the LM head, KV cache and workspace (HOT), with the token embedding COLD.
    std::vector<Block> blocks;
    auto take = [&](uint64_t bytes, uint64_t align, Tier t, bool use_default) {
        const uint64_t va = use_default ? a.alloc(bytes, align) : a.alloc(bytes, align, t);
        VJ_CHECK(va != 0);
        blocks.push_back({va - kBase, va - kBase + bytes, t});
    };
    take(763 * MiB, 4096, Tier::COLD, false);  // token embedding
    for (int il = 0; il < 48; ++il) {
        take(il % 4 == 3 ? 22 * MiB : 25 * MiB, 4096, Tier::HOT, true);  // attention / GDN projections
        take(3 * MiB, 256, Tier::HOT, true);                              // shared expert + router + norms
        take(258 * MiB, 4096, Tier::COLD, false);                         // routed Gate+Up
        take(204 * MiB, 4096, Tier::COLD, false);                         // routed Down
    }
    take(405 * MiB, 4096, Tier::HOT, true);  // LM head rows
    take(160 * MiB, 4096, Tier::HOT, true);  // KV cache + GDN state + workspace
    VJ_CHECK(a.hot_spill_bytes == 0 && a.cold_spill_bytes == 0);

    uint64_t hot_end = 0;
    for (const auto& b : blocks) {
        if (b.asked == Tier::HOT) {
            VJ_CHECK(b.end <= a.hot_limit);
            VJ_CHECK(b.begin >= hot_end && b.begin - hot_end < 4096);  // contiguous up to alignment padding
            hot_end = b.end;
        } else {
            VJ_CHECK(b.begin >= a.hot_limit && b.end <= a.capacity());
        }
    }
    VJ_CHECK(hot_end == a.hot_used());
    check_no_overlap(blocks);
    VJ_CHECK(a.used() == a.hot_used() + a.cold_used());
    VJ_CHECK(a.chunks.size() == chunks_before.size());
    for (size_t i = 0; i < a.chunks.size(); ++i)
        VJ_CHECK(a.chunks[i].va == chunks_before[i].va && a.chunks[i].size == kArenaChunkBytes);
    std::printf("  tiered Ornith-like load: HOT %.2f / %.2f GiB in %llu chunks, COLD %.2f GiB, 0 spills, %zu blocks, 0 overlaps\n",
                a.hot_used() / double(GiB), a.hot_limit / double(GiB),
                static_cast<unsigned long long>(a.hot_limit / kArenaChunkBytes), a.cold_used() / double(GiB), blocks.size());
}

void test_spills_and_exhaustion() {
    KfdArena a = make_arena(4ULL * kArenaChunkBytes);  // 2 GiB, 4 chunks
    VJ_CHECK(a.enable_tiers(kArenaChunkBytes));          // HOT = chunk 0, COLD = chunks 1..3
    std::vector<Block> blocks;
    auto take = [&](uint64_t bytes, Tier t) {
        const uint64_t va = a.alloc(bytes, 4096, t);
        if (va) blocks.push_back({va - kBase, va - kBase + bytes, t});
        return va;
    };
    VJ_CHECK(take(400 * MiB, Tier::HOT) == kBase);
    const uint64_t spilled = take(200 * MiB, Tier::HOT);  // does not fit the remaining 112 MiB of HOT
    VJ_CHECK(spilled == kBase + kArenaChunkBytes && a.hot_spill_bytes == 200 * MiB);
    VJ_CHECK(take(100 * MiB, Tier::HOT) == kBase + 400 * MiB);  // still fits HOT: stays there
    VJ_CHECK(take(1336 * MiB, Tier::COLD) != 0);                // COLD: 1536 - 200 = 1336 MiB free -> now full
    VJ_CHECK(take(10 * MiB, Tier::COLD) == kBase + 500 * MiB);  // COLD full -> HOT tail, counted
    VJ_CHECK(a.cold_spill_bytes == 10 * MiB);
    VJ_CHECK(take(64 * MiB, Tier::COLD) == 0 && take(64 * MiB, Tier::HOT) == 0);  // both full: clean failure
    check_no_overlap(blocks);
    VJ_CHECK(a.used() == 400 * MiB + 200 * MiB + 100 * MiB + 1336 * MiB + 10 * MiB && a.cold_used() == 1536 * MiB);
    VJ_CHECK(a.alloc(1ULL << 62, 4096, Tier::HOT) == 0);  // absurd size: no wrap
}

void test_quarantine_mark_reset_scope() {
    KfdArena a = make_arena(8ULL * kArenaChunkBytes);
    a.quarantine = {{10 * MiB, 30 * MiB}, {kArenaChunkBytes * 2, kArenaChunkBytes * 2 + 2 * MiB}};
    VJ_CHECK(a.enable_tiers(kArenaChunkBytes * 2));
    const uint64_t h0 = a.alloc(8 * MiB, 4096, Tier::HOT);
    const uint64_t h1 = a.alloc(8 * MiB, 4096, Tier::HOT);  // would cross [10, 30) MiB -> moved past it
    VJ_CHECK(h0 == kBase && h1 == kBase + 30 * MiB);
    const uint64_t c0 = a.alloc(kArenaChunkBytes, 4096, Tier::COLD);
    VJ_CHECK(c0 == kBase + kArenaChunkBytes * 2 + 2 * MiB);  // COLD starts at chunk 2 but skips the quarantine

    const KfdArena::Mark m = a.mark();
    const uint64_t used = a.used();
    VJ_CHECK(a.alloc(100 * MiB, 4096, Tier::HOT) != 0 && a.alloc(100 * MiB, 4096, Tier::COLD) != 0);
    a.rollback(m);
    VJ_CHECK(a.used() == used && a.alloc(8 * MiB, 4096, Tier::HOT) == kBase + 38 * MiB);

    {
        ArenaTierScope scope(a, Tier::COLD);
        VJ_CHECK(a.default_tier == Tier::COLD);
        VJ_CHECK(a.alloc(4096) >= kBase + a.hot_limit);
    }
    VJ_CHECK(a.default_tier == Tier::HOT && a.alloc(4096) < kBase + a.hot_limit);

    VJ_CHECK(!a.enable_tiers(kArenaChunkBytes));  // only on an empty arena
    a.reset();
    VJ_CHECK(a.used() == 0 && a.offset == a.hot_limit && a.hot_offset == 0 && a.tiered());

    KfdArena b = make_arena(2ULL * kArenaChunkBytes);
    VJ_CHECK(!b.enable_tiers(2ULL * kArenaChunkBytes) && !b.tiered());  // HOT must leave COLD space
    VJ_CHECK(!b.enable_tiers(0) && !b.tiered() && b.default_tier == Tier::COLD);
    VJ_CHECK(b.alloc(MiB) == kBase);  // still a plain arena
    VJ_CHECK(!b.enable_tiers(MiB));   // not empty any more
}

void test_env_parser() {
    VJ_CHECK(va_cluster_hot_bytes(nullptr, nullptr) == 0);
    VJ_CHECK(va_cluster_hot_bytes("0", "4096") == 0);
    VJ_CHECK(va_cluster_hot_bytes("yes", nullptr) == 0);
    VJ_CHECK(va_cluster_hot_bytes("1", nullptr) == kDefaultHotChunks * kArenaChunkBytes);
    VJ_CHECK(va_cluster_hot_bytes("1", "") == kDefaultHotChunks * kArenaChunkBytes);
    VJ_CHECK(va_cluster_hot_bytes("1", "3072") == 3072 * MiB);
    VJ_CHECK(va_cluster_hot_bytes("1", "12abc") == kDefaultHotChunks * kArenaChunkBytes);  // malformed -> default
    VJ_CHECK(va_cluster_hot_bytes("1", "0") == kDefaultHotChunks * kArenaChunkBytes);
    VJ_CHECK(va_cluster_hot_bytes("1", "99999999999999999999") == kDefaultHotChunks * kArenaChunkBytes);
}

} // namespace

int main() {
    test_chunk_sizes();
    test_untiered_matches_legacy();
    test_tiered_layout();
    test_spills_and_exhaustion();
    test_quarantine_mark_reset_scope();
    test_env_parser();
    std::printf("test_kfd_arena_tiers: PASS\n");
    return 0;
}
