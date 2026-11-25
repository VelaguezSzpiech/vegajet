// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// core/moe_offload against a synthetic 48-layer x 256-expert table written to a temporary file under the current
// directory: byte-exact reads through the pinned staging pool, prefetch hits / dedupe / LRU eviction, run
// coalescing into vectored reads, the io_uring and preadv backends, ring/tag wrap over many short batches,
// partial runs at end of file, a bad descriptor, out-of-range ids, the deterministic hot set, one- and two-card VRAM
// watermarks with hysteresis, and teardown with reads still in flight.
#include "core/moe_offload.hpp"
#include "tests/test_common.hpp"

#include <unistd.h>
#include <sys/mman.h>
#include <atomic>
#include <new>
#include <cstdlib>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

static std::atomic<size_t> allocations{0};
__attribute__((noinline)) void* operator new(size_t n) {
    allocations.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](size_t n) { return ::operator new(n); }
__attribute__((noinline)) void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { ::operator delete(p); }
void operator delete[](void* p) noexcept { ::operator delete(p); }
void operator delete[](void* p, size_t) noexcept { ::operator delete(p); }

using namespace vegajet::moeoffload;

namespace {

constexpr uint32_t kLayers = 48;
constexpr uint32_t kExperts = 256;
constexpr uint32_t kExpertBytes = 96;
constexpr uint64_t kHeader = 4096; // experts start after an unrelated header, like a tensor inside a GGUF

uint8_t expert_byte(uint32_t layer, uint32_t expert, uint32_t i) {
    return static_cast<uint8_t>((layer * 1315423911u + expert * 2654435761u + i * 40503u) >> 11);
}

// Temporary expert table under the current directory. `truncate_experts` keeps only that many fully written
// experts after the header, so a run can cross end of file.
struct TempExpertFile {
    std::string path;
    int fd = -1;

    explicit TempExpertFile(uint32_t layers = kLayers, uint32_t truncate_experts = UINT32_MAX) {
        char tmpl[] = "./vj-moeoffload-XXXXXX";
        fd = ::mkstemp(tmpl);
        VJ_CHECK(fd >= 0);
        path = tmpl;
        std::vector<uint8_t> buf(kHeader, 0xEE);
        buf.reserve(buf.size() + static_cast<size_t>(layers) * kExperts * kExpertBytes);
        for (uint32_t l = 0; l < layers; ++l)
            for (uint32_t e = 0; e < kExperts; ++e)
                for (uint32_t i = 0; i < kExpertBytes; ++i) buf.push_back(expert_byte(l, e, i));
        size_t off = 0;
        while (off < buf.size()) {
            const ssize_t w = ::write(fd, buf.data() + off, buf.size() - off);
            VJ_CHECK(w > 0);
            off += static_cast<size_t>(w);
        }
        if (truncate_experts != UINT32_MAX) {
            const off_t end = static_cast<off_t>(kHeader + static_cast<uint64_t>(truncate_experts) * kExpertBytes);
            VJ_CHECK(::ftruncate(fd, end) == 0);
        }
    }
    ~TempExpertFile() {
        if (fd >= 0) ::close(fd);
        if (!path.empty()) ::unlink(path.c_str());
    }
    TempExpertFile(const TempExpertFile&) = delete;
    TempExpertFile& operator=(const TempExpertFile&) = delete;

    ExpertTable table(uint32_t layers = kLayers) const {
        ExpertTable t;
        t.fd = fd;
        t.file_offset = kHeader;
        t.n_layers = layers;
        t.n_experts = kExperts;
        t.expert_bytes = kExpertBytes;
        return t;
    }
};

OffloadConfig cfg_slots(uint32_t slots, uint32_t max_run_rows = 16) {
    OffloadConfig c;
    c.staging_slots = slots;
    c.max_run_rows = max_run_rows;
    c.hot_per_layer = 16;
    c.io_entries = 64;
    c.threads = 2;
    c.io = IoKind::Auto;
    return c;
}

void all_experts(std::vector<uint32_t>& out) {
    out.resize(kExperts);
    for (uint32_t i = 0; i < kExperts; ++i) out[i] = i;
}

void poll_idle(MoeOffload& m) {
    for (int i = 0; i < 20000 && m.in_flight() > 0; ++i) {
        m.poll();
        if (m.in_flight() > 0) std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    VJ_CHECK(m.in_flight() == 0);
}

bool fetch_matches(MoeOffload& m, uint32_t layer, uint32_t expert) {
    std::vector<uint8_t> got(kExpertBytes);
    std::string err;
    if (!m.fetch(layer, expert, got.data(), &err)) return false;
    for (uint32_t i = 0; i < kExpertBytes; ++i)
        if (got[i] != expert_byte(layer, expert, i)) return false;
    return true;
}

// ---- byte-exact reads of every expert, prefetch -> hit path
void test_byte_exact_all() {
    TempExpertFile f;
    OffloadConfig cfg = cfg_slots(kExperts, 32);
    std::string why;
    MoeOffload m(f.table(), cfg, &why);
    VJ_CHECK(m.ok());
    VJ_CHECK(m.backend_name() != nullptr && *m.backend_name() != '\0');
    VJ_CHECK(m.staging_bytes() >= static_cast<size_t>(kExperts) * kExpertBytes);

    std::vector<uint32_t> all;
    all_experts(all);
    for (uint32_t l = 0; l < kLayers; ++l) {
        VJ_CHECK(m.prefetch(l, all.data(), all.size()) == all.size());
        poll_idle(m);
        for (uint32_t e = 0; e < kExperts; ++e) VJ_CHECK(m.staged(l, e));
        for (uint32_t e = 0; e < kExperts; ++e) VJ_CHECK(fetch_matches(m, l, e));
    }
    const OffloadStats& st = m.stats();
    VJ_CHECK(st.requested == static_cast<uint64_t>(kLayers) * kExperts);
    VJ_CHECK(st.issued == static_cast<uint64_t>(kLayers) * kExperts);
    VJ_CHECK(st.hits == static_cast<uint64_t>(kLayers) * kExperts);
    VJ_CHECK(st.misses == 0);
    VJ_CHECK(st.errors == 0);
    VJ_CHECK(st.completions == static_cast<uint64_t>(kLayers) * (kExperts / 32));
    VJ_CHECK(st.read_ops == static_cast<uint64_t>(kLayers) * (kExperts / 32));
    VJ_CHECK(st.bytes == static_cast<uint64_t>(kLayers) * kExperts * kExpertBytes);
}

// ---- fetch misses then hits, release drops the copy, and a full shuffled sweep wraps the engine rings
void test_hits_misses_and_wrap() {
    TempExpertFile f;
    {
        OffloadConfig cfg = cfg_slots(4, 2);
        cfg.io_entries = 4;
        cfg.max_in_flight = 4;
        MoeOffload m(f.table(), cfg);
        VJ_CHECK(m.ok());
        VJ_CHECK(!m.staged(0, 0));
        VJ_CHECK(fetch_matches(m, 0, 0)); // miss
        VJ_CHECK(m.stats().misses == 1);
        VJ_CHECK(m.stats().hits == 0);
        VJ_CHECK(m.staged(0, 0));
        VJ_CHECK(fetch_matches(m, 0, 0)); // hit
        VJ_CHECK(m.stats().hits == 1);
        VJ_CHECK(m.release(0, 0));
        VJ_CHECK(!m.staged(0, 0));
        VJ_CHECK(fetch_matches(m, 0, 0)); // miss again after release
        VJ_CHECK(m.stats().misses == 2);
        VJ_CHECK(m.stats().releases >= 1);
        VJ_CHECK(m.stats().released_bytes > 0);
    }
    {
        OffloadConfig cfg = cfg_slots(4, 2);
        cfg.io_entries = 4;
        cfg.max_in_flight = 4;
        MoeOffload m(f.table(), cfg);
        VJ_CHECK(m.ok());
        uint32_t rng = 12345u;
        for (uint32_t n = 0; n < kLayers * kExperts; ++n) {
            rng = rng * 1664525u + 1013904223u;
            const uint32_t l = (rng >> 8) % kLayers;
            const uint32_t e = (rng >> 4) % kExperts;
            VJ_CHECK(fetch_matches(m, l, e));
        }
        const OffloadStats& st = m.stats();
        VJ_CHECK(st.errors == 0);
        VJ_CHECK(st.hits + st.misses == static_cast<uint64_t>(kLayers) * kExperts);
        VJ_CHECK(st.misses > 0);
        VJ_CHECK(st.read_ops > 4096);       // many short batches recycle the request ring / SQEs
        VJ_CHECK(st.completions == st.misses);
        VJ_CHECK(st.evictions > 0);
    }
}

// ---- adjacent experts become one vectored read; staged rows are deduped
void test_coalescing_and_dedupe() {
    TempExpertFile f;
    MoeOffload m(f.table(), cfg_slots(kExperts, 8));
    VJ_CHECK(m.ok());
    const uint32_t rows[] = {5, 6, 7, 20, 21, 40};
    VJ_CHECK(m.prefetch(0, rows, 6) == 6);
    VJ_CHECK(m.stats().read_ops == 3); // [5,6,7] [20,21] [40]
    poll_idle(m);
    for (uint32_t e : rows) {
        VJ_CHECK(m.staged(0, e));
        VJ_CHECK(fetch_matches(m, 0, e));
    }
    const uint64_t issued_before = m.stats().issued;
    VJ_CHECK(m.prefetch(0, rows, 6) == 0); // everything already resident
    VJ_CHECK(m.stats().deduped == 6);
    VJ_CHECK(m.stats().issued == issued_before);
}

// ---- duplicates inside one prefetch, bad ids, end of file, bad descriptor
void test_failures() {
    TempExpertFile f;
    {
        MoeOffload m(f.table(), cfg_slots(8, 16));
        VJ_CHECK(m.ok());
        std::vector<uint8_t> dst(kExpertBytes);
        std::string err;
        VJ_CHECK(!m.fetch(0, kExperts, dst.data(), &err)); // expert out of range
        VJ_CHECK(!err.empty());
        VJ_CHECK(!m.fetch(kLayers, 0, dst.data(), &err)); // layer out of range
        VJ_CHECK(!m.fetch(0, 0, nullptr, &err));          // null destination
        const uint32_t bad[] = {0, 0, 999, 3};
        VJ_CHECK(m.prefetch(0, bad, 4) == 2); // 0 is deduped, 3 and 999: 999 is out of range
        VJ_CHECK(m.stats().errors >= 2);
        VJ_CHECK(m.prefetch(kLayers, bad, 4) == 0);
        VJ_CHECK(m.stats().deferred >= 4);
    }
    { // a run that crosses end of file: the fully written experts are served, the rest fail once
        TempExpertFile cut(1, 5); // only 5 experts physically on disk
        MoeOffload m(cut.table(), cfg_slots(32, 16));
        VJ_CHECK(m.ok());
        uint32_t rows[16];
        for (uint32_t i = 0; i < 16; ++i) rows[i] = i;
        VJ_CHECK(m.prefetch(0, rows, 16) == 16);
        poll_idle(m);
        for (uint32_t i = 0; i < 5; ++i) {
            VJ_CHECK(m.staged(0, i));
            VJ_CHECK(fetch_matches(m, 0, i));
        }
        for (uint32_t i = 6; i < 16; ++i) VJ_CHECK(!m.staged(0, i));
        std::vector<uint8_t> dst(kExpertBytes);
        std::string err;
        VJ_CHECK(!m.fetch(0, 10, dst.data(), &err));
        VJ_CHECK(err.find("short") != std::string::npos);
        VJ_CHECK(m.stats().short_reads >= 1);
        VJ_CHECK(m.stats().errors >= 1);
    }
    { // a descriptor closed behind the manager's back: the read fails, the slot is recoverable
        TempExpertFile f2;
        const int dead = ::dup(f2.fd);
        VJ_CHECK(dead >= 0);
        ::close(dead);
        ExpertTable t = f2.table();
        t.fd = dead;
        OffloadConfig cfg = cfg_slots(4, 4);
        cfg.io = IoKind::Preadv; // no ring fd can silently take over the closed descriptor number
        MoeOffload m(t, cfg);
        VJ_CHECK(m.ok());
        std::vector<uint8_t> dst(kExpertBytes);
        std::string err;
        VJ_CHECK(!m.fetch(0, 0, dst.data(), &err));
        VJ_CHECK(!err.empty());
        VJ_CHECK(m.stats().errors >= 1);
        VJ_CHECK(m.in_flight() == 0);
    }
}

// ---- release / release_all return the pinned pages and the data is still correct afterwards
void test_release_and_pin() {
    TempExpertFile f;
    OffloadConfig cfg = cfg_slots(8, 8);
    MoeOffload m(f.table(), cfg);
    VJ_CHECK(m.ok());
    std::vector<uint32_t> all;
    all_experts(all);
    VJ_CHECK(m.prefetch(0, all.data(), 8) == 8); // 8 slots: the rest is deferred
    poll_idle(m);
    for (uint32_t e = 0; e < 8; ++e) VJ_CHECK(m.staged(0, e));
    VJ_CHECK(m.release(0, 3));
    VJ_CHECK(!m.staged(0, 3));
    VJ_CHECK(m.staging_pinned());
    VJ_CHECK(!m.release(0, 3)); // already gone
    VJ_CHECK(fetch_matches(m, 0, 3)); // refaults the released pages, bytes still exact
    const uint64_t released_before = m.stats().released_bytes;
    m.release_all();
    VJ_CHECK(m.in_flight() == 0);
    VJ_CHECK(m.stats().released_bytes > released_before);
    std::vector<unsigned char> residency(m.staging_bytes() / size_t(sysconf(_SC_PAGESIZE)));
    VJ_CHECK(mincore(const_cast<void*>(m.staging_base()), m.staging_bytes(), residency.data()) == 0);
    for (auto v : residency) VJ_CHECK(!(v & 1));
    for (uint32_t e = 0; e < kExperts; e += 61) VJ_CHECK(fetch_matches(m, 0, e));
}

// ---- deterministic hot/cold split and one/two card watermarks with hysteresis
void test_hot_and_watermarks() {
    TempExpertFile f;
    OffloadConfig cfg = cfg_slots(16, 8);
    cfg.hot_per_layer = 3;
    cfg.cards = {{16, 16, 8}};
    MoeOffload a(f.table(), cfg);
    MoeOffload b(f.table(), cfg);
    VJ_CHECK(a.ok() && b.ok());
    VJ_CHECK(a.cards() == 1);
    VJ_CHECK(a.is_hot(0, 0) && a.is_hot(0, 2) && !a.is_hot(0, 3));
    VJ_CHECK(a.hot_total() == 3 * kLayers);
    const std::vector<Placement> pa = a.plan();
    const std::vector<Placement> pb = b.plan();
    VJ_CHECK(pa.size() == 16);
    VJ_CHECK(pa.size() == pb.size());
    for (size_t i = 0; i < pa.size(); ++i) {
        VJ_CHECK(pa[i].layer == pb[i].layer && pa[i].expert == pb[i].expert && pa[i].card == pb[i].card);
        VJ_CHECK(a.is_hot(pa[i].layer, pa[i].expert));
        VJ_CHECK(pa[i].card == 0);
    }
    VJ_CHECK(pa[0].layer == 0 && pa[0].expert == 0);
    VJ_CHECK(pa[15].layer == 5 && pa[15].expert == 0); // 3 hot experts per layer, 16 slots
    VJ_CHECK(a.resident_on(0) == 16);
    VJ_CHECK(a.resident_card(0, 0) == 0);
    VJ_CHECK(a.resident_card(5, 0) == 0);
    VJ_CHECK(a.resident_card(5, 1) == -1);
    VJ_CHECK(a.resident_card(5, 2) == -1);
    VJ_CHECK(a.resident_on(1) == 0);

    // Shrink the card: hysteresis trims residency down to the low mark (4), most recent first.
    VJ_CHECK(a.resize_card(0, 4) == 12);
    VJ_CHECK(a.resident_on(0) == 4);
    VJ_CHECK(a.resident_card(0, 2) == 0); // the earliest admits survive
    VJ_CHECK(a.resident_card(1, 3) == -1); // later admits are the ones demoted

    // Lowering the high mark below residency trims to the low mark, then refills to the new high mark.
    a.set_watermarks({{4, 2, 1}});
    VJ_CHECK(a.resident_on(0) == 2);
    VJ_CHECK(a.resident_card(0, 1) == 0);

    // Two cards: first-fit placement up to each card's own high watermark.
    OffloadConfig two = cfg_slots(8, 8);
    two.hot_per_layer = 4;
    two.cards = {{8, 8, 4}, {4, 4, 2}};
    MoeOffload c(f.table(), two);
    VJ_CHECK(c.ok());
    VJ_CHECK(c.cards() == 2);
    const std::vector<Placement> pc = c.plan();
    VJ_CHECK(pc.size() == 12);
    VJ_CHECK(c.resident_on(0) == 8);
    VJ_CHECK(c.resident_on(1) == 4);
    VJ_CHECK(c.resident_card(1, 3) == 0); // layer 1 still fits on card 0
    VJ_CHECK(c.resident_card(2, 0) == 1); // card 0 is full, card 1 takes layer 2
    VJ_CHECK(c.resident_card(4, 3) == -1);

    // hot_mask overrides the prefix: exactly one hot expert in the whole table.
    OffloadConfig masked = cfg_slots(8, 8);
    masked.hot_per_layer = 0;
    masked.cards = {{4, 4, 2}};
    masked.hot_mask.assign(static_cast<size_t>(kLayers) * kExperts, 0);
    masked.hot_mask[static_cast<size_t>(kLayers - 1) * kExperts + (kExperts - 1)] = 1;
    MoeOffload d(f.table(), masked);
    VJ_CHECK(d.ok());
    VJ_CHECK(d.hot_total() == 1);
    const std::vector<Placement> pd = d.plan();
    VJ_CHECK(pd.size() == 1);
    VJ_CHECK(pd[0].layer == kLayers - 1 && pd[0].expert == kExperts - 1);
}

// ---- both backends agree, byte for byte
void test_backends() {
    TempExpertFile f;
    for (IoKind kind : {IoKind::Preadv, IoKind::IoUring, IoKind::Auto}) {
        OffloadConfig cfg = cfg_slots(kExperts, 16);
        cfg.io = kind;
        std::string why;
        MoeOffload m(f.table(), cfg, &why);
        VJ_CHECK(m.ok());
        VJ_CHECK(!why.empty());
        std::vector<uint32_t> all;
        all_experts(all);
        const auto before = allocations.load();
        for (uint32_t l = 0; l < 4; ++l) {
            VJ_CHECK(m.prefetch(l, all.data(), all.size()) == all.size());
            poll_idle(m);
            for (uint32_t e = 0; e < kExperts; ++e) {
                uint8_t dst[kExpertBytes];
                VJ_CHECK(m.fetch(l, e, dst));
                for (uint32_t i = 0; i < kExpertBytes; ++i) VJ_CHECK(dst[i] == expert_byte(l, e, i));
            }
        }
        VJ_CHECK(allocations.load() == before);
        VJ_CHECK(m.stats().errors == 0);
        if (kind == IoKind::Preadv) VJ_CHECK(std::string(m.backend_name()).find("preadv") != std::string::npos);
    }
}

// ---- destroying / releasing while reads are outstanding must not touch a freed arena
void test_async_lifetime() {
    TempExpertFile f;
    std::vector<uint32_t> all;
    all_experts(all);
    {
        OffloadConfig cfg = cfg_slots(8, 32);
        cfg.io = IoKind::Preadv;
        MoeOffload m(f.table(), cfg);
        VJ_CHECK(m.ok());
        VJ_CHECK(m.prefetch(0, all.data(), all.size()) > 0);
        VJ_CHECK(m.in_flight() > 0);
        // destructor drains before unmapping the arena
    }
    {
        OffloadConfig cfg = cfg_slots(8, 8);
        MoeOffload m(f.table(), cfg);
        VJ_CHECK(m.ok());
        VJ_CHECK(m.prefetch(0, all.data(), all.size()) > 0);
        m.release_all(); // drains outstanding reads, then drops every staged page
        VJ_CHECK(m.in_flight() == 0);
        for (uint32_t e = 0; e < 16; ++e) VJ_CHECK(fetch_matches(m, 1, e));
    }
}

} // namespace

int main() {
    test_byte_exact_all();
    test_hits_misses_and_wrap();
    test_coalescing_and_dedupe();
    test_failures();
    test_release_and_pin();
    test_hot_and_watermarks();
    test_backends();
    test_async_lifetime();
    std::printf("test_moe_offload: ok\n");
    return 0;
}
