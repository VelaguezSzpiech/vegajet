// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// core/prefix_cache (radix tree, refcounts, LRU eviction, snapshots, O(L) match) and core/cont_batch (B = 1..8
// continuous batching, chunked prefill, priority preemption, cancellation without leaks).
//
// The mock engine stores real per-block KV contents and per-slot recurrent state. For every work item it re-derives
// the state from the KV it holds (attention) or checks the restored snapshot against it (hybrid), and samples a token
// that is a pure function of the whole token history. Any wrong block sharing, snapshot, rollback or preemption shows
// up as an output that differs from a direct reference generation.
#include "core/cont_batch.hpp"
#include "core/prefix_cache.hpp"
#include "tests/test_common.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <map>
#include <mutex>
#include <random>
#include <thread>

using namespace vegajet;
using batch::ContinuousBatcher;
using batch::End;
using batch::StateOp;

namespace {

uint64_t mix(uint64_t h, int32_t t) {
    uint64_t x = h ^ (static_cast<uint64_t>(static_cast<uint32_t>(t)) * 0x9E3779B97F4A7C15ull);
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 27; x *= 0x94D049BB133111EBull; x ^= x >> 31;
    return x;
}
constexpr uint64_t kH0 = 0x1234567;
constexpr int32_t kEos = 2;
int32_t token_from(uint64_t h) { return static_cast<int32_t>(h % 1000) + 2; }  // 2 = EOS with p = 1/1000

std::vector<int32_t> reference(const std::vector<int32_t>& prompt, uint32_t max_new, int32_t eos) {
    uint64_t h = kH0;
    for (int32_t t : prompt) h = mix(h, t);
    std::vector<int32_t> out;
    for (;;) {
        const int32_t t = token_from(h);
        out.push_back(t);
        if (t == eos || out.size() >= max_new) return out;
        h = mix(h, t);
    }
}

struct MockEngine {
    uint32_t bt;
    bool hybrid;
    std::vector<std::vector<int32_t>> kv;
    std::vector<uint64_t> state, snap;
    std::vector<uint32_t> state_len, snap_len;
    uint64_t errors = 0, restores = 0, saves = 0, steps = 0;
    uint32_t max_work = 0;
    std::function<void()> mid_step;  // runs inside the engine call (cancellation races)

    MockEngine(uint32_t block_tokens, uint32_t blocks, uint32_t snaps, bool hyb)
        : bt(block_tokens), hybrid(hyb), kv(blocks, std::vector<int32_t>(block_tokens, -1)), state(8, kH0),
          snap(snaps, 0), state_len(8, 0), snap_len(snaps, 0) {}

    batch::StepOut operator()(const batch::StepPlan& p) {
        ++steps;
        max_work = std::max<uint32_t>(max_work, static_cast<uint32_t>(p.work.size()));
        batch::StepOut out;
        for (const StateOp& op : p.pre) {
            if (op.kind == StateOp::Kind::Reset) { state[op.slot] = kH0; state_len[op.slot] = 0; }
            else if (op.kind == StateOp::Kind::Restore) {
                ++restores;
                state[op.slot] = snap[static_cast<size_t>(op.snap)];
                state_len[op.slot] = snap_len[static_cast<size_t>(op.snap)];
                if (state_len[op.slot] != op.tokens) ++errors;
            } else ++errors;
        }
        for (const batch::SlotWork& w : p.work) {
            uint64_t h = kH0;  // what the cached KV says the prefix is
            for (uint32_t i = 0; i < w.pos0; ++i) {
                const int32_t t = kv[static_cast<size_t>(w.block_table[i / bt])][i % bt];
                if (t < 0) ++errors;
                h = mix(h, t);
            }
            if (hybrid) {
                if (state_len[w.slot] != w.pos0 || state[w.slot] != h) ++errors;  // recurrent state must match KV
            }
            for (size_t i = 0; i < w.tokens.size(); ++i) {
                const uint32_t pos = w.pos0 + static_cast<uint32_t>(i);
                if (pos / bt >= w.block_table.size()) { ++errors; continue; }
                kv[static_cast<size_t>(w.block_table[pos / bt])][pos % bt] = w.tokens[i];
                h = mix(h, w.tokens[i]);
            }
            state[w.slot] = h;
            state_len[w.slot] = w.pos0 + static_cast<uint32_t>(w.tokens.size());
            if (w.sample) out.sampled.push_back({w.slot, token_from(h)});
        }
        for (const StateOp& op : p.post) {
            if (op.kind != StateOp::Kind::Save || state_len[op.slot] != op.tokens) { ++errors; continue; }
            ++saves;
            snap[static_cast<size_t>(op.snap)] = state[op.slot];
            snap_len[static_cast<size_t>(op.snap)] = state_len[op.slot];
        }
        if (mid_step) mid_step();
        return out;
    }
};

std::vector<int32_t> rand_tokens(std::mt19937& g, size_t n) {
    std::uniform_int_distribution<int32_t> d(3, 50000);
    std::vector<int32_t> v(n);
    for (auto& x : v) x = d(g);
    return v;
}

// ------------------------------------------------------------------------------------------------- radix cache
void test_radix_basics() {
    prefix::BlockPool pool(64);
    prefix::RadixCache c(pool, 4, 4, false);
    std::vector<int32_t> a(32), b;
    for (int i = 0; i < 32; ++i) a[static_cast<size_t>(i)] = i + 100;
    std::vector<prefix::BlockId> blocks_a;
    for (int i = 0; i < 8; ++i) blocks_a.push_back(pool.alloc());
    VJ_CHECK(!c.insert(a.data(), 32, blocks_a.data()));  // no snapshot given -> false, blocks cached
    for (auto x : blocks_a) VJ_CHECK(pool.refs(x) == 2);  // sequence + cache
    auto m = c.match(a.data(), 32);
    VJ_CHECK(m.tokens == 32 && m.blocks == blocks_a);
    c.release(m);
    m = c.match(a.data(), 31);  // block aligned: 28
    VJ_CHECK(m.tokens == 28 && m.blocks.size() == 7);
    c.release(m);
    // Divergence in block 5 splits the edge; the shared 5-block prefix is stored once.
    b = a;
    b[21] = 7;
    std::vector<prefix::BlockId> blocks_b(blocks_a.begin(), blocks_a.begin() + 5);
    for (int i = 5; i < 8; ++i) blocks_b.push_back(pool.alloc());
    c.insert(b.data(), 32, blocks_b.data());
    VJ_CHECK(c.stats().nodes == 3 && c.stats().cached_blocks == 11);
    m = c.match(b.data(), 32);
    VJ_CHECK(m.tokens == 32 && m.blocks == blocks_b);
    c.release(m);
    VJ_CHECK(c.check_invariants());
    // Release the "sequences": only cache references remain.
    for (auto x : blocks_a) pool.unref(x);
    for (int i = 5; i < 8; ++i) pool.unref(blocks_b[static_cast<size_t>(i)]);
    for (auto x : blocks_a) VJ_CHECK(pool.refs(x) == 1);
    // LRU eviction: touch b, evict -> a's private tail (3 blocks) goes first; pinned paths are never evicted.
    m = c.match(b.data(), 32);
    const uint32_t free0 = pool.free_count();
    VJ_CHECK(c.evict_blocks(free0 + 3) == 3);
    auto ma = c.match(a.data(), 32);
    VJ_CHECK(ma.tokens == 20);
    c.release(ma);
    VJ_CHECK(c.evict_blocks(64) == 0);  // b is pinned by m: its blocks (and the shared prefix) stay
    c.release(m);
    VJ_CHECK(c.evict_blocks(64) == 8);
    VJ_CHECK(pool.free_count() == 64 && c.stats().nodes == 0 && c.check_invariants());
}

void test_radix_snapshots() {
    prefix::BlockPool pool(32);
    prefix::RadixCache c(pool, 4, 2, true);
    std::vector<int32_t> t(24);
    for (int i = 0; i < 24; ++i) t[static_cast<size_t>(i)] = i + 1;
    std::vector<prefix::BlockId> bl;
    for (int i = 0; i < 6; ++i) bl.push_back(pool.alloc());
    VJ_CHECK(c.match(t.data(), 24).tokens == 0);  // nothing cached
    const auto s1 = c.alloc_snapshot();
    VJ_CHECK(c.insert(t.data(), 24, bl.data(), s1, 12));  // state after 12 tokens
    auto m = c.match(t.data(), 24);
    VJ_CHECK(m.tokens == 12 && m.snapshot == s1 && m.blocks.size() == 3);  // hybrid: only up to the snapshot
    c.release(m);
    const auto s2 = c.alloc_snapshot();
    VJ_CHECK(c.insert(t.data(), 24, bl.data(), s2, 20));
    m = c.match(t.data(), 23);
    VJ_CHECK(m.tokens == 20 && m.snapshot == s2);
    // Snapshot pool full (2): every snapshot on the pinned path is protected, so nothing can be evicted ...
    VJ_CHECK(c.alloc_snapshot() == -1);
    c.release(m);
    // ... until the match is released: then the least recently used one goes.
    const auto s3 = c.alloc_snapshot();
    VJ_CHECK(s3 == s1);
    VJ_CHECK(!c.insert(t.data(), 24, bl.data(), s3, 6));  // not block aligned: refused, freed
    VJ_CHECK(!c.insert(t.data(), 24, bl.data(), c.alloc_snapshot(), 20));  // node already has one: refused, freed
    VJ_CHECK(c.snapshots_free() == 1 && c.check_invariants());
    for (auto x : bl) pool.unref(x);
    c.clear();
    VJ_CHECK(pool.free_count() == 32 && c.snapshots_free() == 2);
}

void test_radix_long_prompt() {
    // 40K-token agent prompt, 16-token blocks: match is one pass over the prompt (O(L)).
    std::mt19937 g(5);
    prefix::BlockPool pool(8192);
    prefix::RadixCache c(pool, 16, 8, false);
    const auto sys = rand_tokens(g, 40960);
    std::vector<prefix::BlockId> bl;
    for (size_t i = 0; i < sys.size() / 16; ++i) bl.push_back(pool.alloc());
    c.insert(sys.data(), static_cast<uint32_t>(sys.size()), bl.data());
    for (int k = 0; k < 64; ++k) {  // 64 other cached conversations branching off the system prompt
        auto t = sys;
        t.resize(10240 + static_cast<size_t>(k) * 256);
        auto tail = rand_tokens(g, 512);
        t.insert(t.end(), tail.begin(), tail.end());
        std::vector<prefix::BlockId> tb(bl.begin(), bl.begin() + static_cast<long>((10240 + k * 256) / 16));
        for (size_t i = tb.size(); i < t.size() / 16; ++i) tb.push_back(pool.alloc());
        c.insert(t.data(), static_cast<uint32_t>(t.size()), tb.data());
    }
    VJ_CHECK(c.check_invariants());
    auto q = sys;
    q.push_back(42);
    double us[2];
    uint32_t lens[2] = {10240, 40960};
    for (int r = 0; r < 2; ++r) {
        std::vector<int32_t> p(q.begin(), q.begin() + lens[r]);
        p.push_back(7);
        const auto t0 = std::chrono::steady_clock::now();
        prefix::Match m;
        for (int it = 0; it < 20; ++it) { m = c.match(p.data(), static_cast<uint32_t>(p.size())); c.release(m); }
        us[r] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 20;
        VJ_CHECK(m.tokens == lens[r]);
    }
    std::printf("Radix match: 10K-token prefix %.1f us, 40K-token prefix %.1f us (%u nodes, %u cached blocks)\n", us[0], us[1],
                c.stats().nodes, c.stats().cached_blocks);
}

// ------------------------------------------------------------------------------------------------- batcher
struct Run {
    std::map<uint64_t, batch::Result> results;
    batch::BatchStats st;
    uint64_t engine_errors = 0, restores = 0;
    uint32_t max_work = 0;
};

Run run(batch::BatchConfig cfg, const std::vector<batch::Request>& reqs, bool check_each_step = true) {
    ContinuousBatcher b(cfg);
    MockEngine e(cfg.block_tokens, cfg.kv_blocks, cfg.max_snapshots, cfg.hybrid);
    for (const auto& r : reqs) VJ_CHECK(b.submit(r));
    uint64_t guard = 0;
    while (b.step(std::ref(e))) {
        if (check_each_step) VJ_CHECK(b.check_invariants());
        VJ_CHECK(++guard < 200000);
    }
    VJ_CHECK(b.check_invariants() && b.active() == 0 && b.waiting() == 0);
    Run out;
    for (const auto& r : reqs) out.results[r.id] = b.result(r.id);
    out.st = b.stats();
    out.engine_errors = e.errors;
    out.restores = e.restores;
    out.max_work = e.max_work;
    b.clear_cache();  // zero leaks: an idle batcher holds nothing once the cache is dropped
    VJ_CHECK(b.free_blocks() == cfg.kv_blocks && b.free_snapshots() == cfg.max_snapshots);
    return out;
}

std::vector<batch::Request> agent_requests(std::mt19937& g, size_t n, size_t sys_len) {
    const auto sys = rand_tokens(g, sys_len);
    std::vector<batch::Request> v;
    std::uniform_int_distribution<int> len(40, 300), gen(8, 48);
    for (size_t i = 0; i < n; ++i) {
        batch::Request r;
        r.id = 100 + i;
        r.prompt = sys;
        auto tail = rand_tokens(g, static_cast<size_t>(len(g)));
        r.prompt.insert(r.prompt.end(), tail.begin(), tail.end());
        r.max_new = static_cast<uint32_t>(gen(g));
        r.eos = kEos;
        v.push_back(r);
    }
    return v;
}

void check_reference(const Run& r, const std::vector<batch::Request>& reqs) {
    VJ_CHECK(r.engine_errors == 0);
    for (const auto& q : reqs) {
        const auto& res = r.results.at(q.id);
        VJ_CHECK(res.end == End::Eos || res.end == End::Length);
        VJ_CHECK(res.tokens == reference(q.prompt, q.max_new, q.eos));
    }
}

void test_shared_system_prompt() {
    std::mt19937 g(1);
    const auto reqs = agent_requests(g, 24, 2048);
    batch::BatchConfig cfg;
    cfg.max_batch = 8;
    cfg.kv_blocks = 2048;
    const Run on = run(cfg, reqs);
    check_reference(on, reqs);
    cfg.prefix_cache = false;
    const Run off = run(cfg, reqs);
    check_reference(off, reqs);
    std::printf("Agent batch (24 requests, 2048-token shared prompt, B=8): prefill %llu tokens with the cache vs %llu without "
                "(%llu prefix-hit tokens), %llu vs %llu steps\n",
                static_cast<unsigned long long>(on.st.prefill_tokens), static_cast<unsigned long long>(off.st.prefill_tokens),
                static_cast<unsigned long long>(on.st.prefix_hit_tokens), static_cast<unsigned long long>(on.st.steps),
                static_cast<unsigned long long>(off.st.steps));
    VJ_CHECK(on.st.prefix_hit_tokens >= 16u * 2048u);  // most requests start after the first batch cached the prompt
    VJ_CHECK(on.st.prefill_tokens * 3 < off.st.prefill_tokens);
    VJ_CHECK(on.st.peak_batch == 8 && on.max_work <= 8);
}

void test_batch_sizes() {
    std::mt19937 g(2);
    const auto reqs = agent_requests(g, 10, 300);
    for (uint32_t bsz = 1; bsz <= 8; ++bsz) {
        batch::BatchConfig cfg;
        cfg.max_batch = bsz;
        cfg.kv_blocks = 512;
        cfg.prefill_chunk = 64;  // chunked prefill interleaves with decode
        cfg.step_tokens = 128;
        const Run r = run(cfg, reqs);
        check_reference(r, reqs);
        VJ_CHECK(r.st.peak_batch == std::min<uint32_t>(bsz, 10) && r.max_work <= bsz);
    }
}

void test_hybrid_snapshots() {
    std::mt19937 g(3);
    const auto reqs = agent_requests(g, 16, 1000);  // 1000 = not block aligned: snapshot at 992
    batch::BatchConfig cfg;
    cfg.hybrid = true;
    cfg.kv_blocks = 2048;
    cfg.max_snapshots = 8;
    const Run r = run(cfg, reqs);
    check_reference(r, reqs);  // the engine also checked every restored state against the cached KV
    VJ_CHECK(r.restores >= 8);
    for (const auto& q : reqs) VJ_CHECK(r.results.at(q.id).prefix_hit_tokens % 16 == 0);
    std::printf("Hybrid (GDN state snapshots): %llu restores, %llu prefix-hit tokens, outputs identical\n",
                static_cast<unsigned long long>(r.restores), static_cast<unsigned long long>(r.st.prefix_hit_tokens));
}

void test_priority_preemption() {
    std::mt19937 g(4);
    batch::BatchConfig cfg;
    cfg.max_batch = 2;
    cfg.kv_blocks = 256;
    ContinuousBatcher b(cfg);
    MockEngine e(cfg.block_tokens, cfg.kv_blocks, cfg.max_snapshots, false);
    std::vector<batch::Request> reqs;
    for (int i = 0; i < 2; ++i) {
        batch::Request r;
        r.id = static_cast<uint64_t>(i + 1);
        r.prompt = rand_tokens(g, 100);
        r.max_new = 200;
        r.priority = 0;
        reqs.push_back(r);
        VJ_CHECK(b.submit(r));
    }
    for (int s = 0; s < 5; ++s) VJ_CHECK(b.step(std::ref(e)));
    batch::Request hi;
    hi.id = 9;
    hi.prompt = rand_tokens(g, 50);
    hi.max_new = 10;
    hi.priority = 5;
    reqs.push_back(hi);
    VJ_CHECK(b.submit(hi));
    while (!b.done(9)) VJ_CHECK(b.step(std::ref(e)));
    VJ_CHECK(!b.done(1) && !b.done(2));  // the urgent request finished while the long ones were still running
    VJ_CHECK(b.stats().preemptions >= 1);
    b.run_until_idle(std::ref(e));
    VJ_CHECK(e.errors == 0 && b.check_invariants());
    for (const auto& q : reqs) VJ_CHECK(b.result(q.id).tokens == reference(q.prompt, q.max_new, q.eos));
    VJ_CHECK(b.result(1).preemptions + b.result(2).preemptions >= 1);
}

void test_cancellation() {
    std::mt19937 g(6);
    const auto reqs = agent_requests(g, 12, 512);
    for (bool hybrid : {false, true}) {
        batch::BatchConfig cfg;
        cfg.max_batch = 4;
        cfg.kv_blocks = 1024;
        cfg.hybrid = hybrid;
        ContinuousBatcher b(cfg);
        MockEngine e(cfg.block_tokens, cfg.kv_blocks, cfg.max_snapshots, hybrid);
        for (const auto& r : reqs) VJ_CHECK(b.submit(r));
        VJ_CHECK(b.cancel(111));  // still waiting: dropped at once
        VJ_CHECK(b.done(111) && b.result(111).end == End::Cancelled && !b.cancel(111));
        VJ_CHECK(b.step(std::ref(e)) && b.step(std::ref(e)));
        VJ_CHECK(b.cancel(100));  // running, between steps: KV released immediately
        VJ_CHECK(b.done(100) && b.check_invariants());
        // Client disconnect while the engine executes a step: released as soon as the step returns.
        uint32_t fired = 0;
        e.mid_step = [&] {
            if (fired++ == 0) {
                VJ_CHECK(b.cancel(101));
                VJ_CHECK(!b.done(101));  // the engine still owns its blocks
            }
        };
        VJ_CHECK(b.step(std::ref(e)));
        VJ_CHECK(b.done(101) && b.result(101).end == End::Cancelled && b.check_invariants());
        e.mid_step = nullptr;
        // Cancellation from another thread while steps run.
        std::atomic<bool> stop{false};
        std::thread killer([&] {
            for (uint64_t id : {104, 107, 110}) {
                std::this_thread::sleep_for(std::chrono::microseconds(200));
                b.cancel(id);
            }
            stop = true;
        });
        while (b.step(std::ref(e))) VJ_CHECK(b.check_invariants());
        killer.join();
        b.run_until_idle(std::ref(e));
        VJ_CHECK(e.errors == 0 && b.active() == 0 && b.waiting() == 0 && b.check_invariants());
        for (const auto& q : reqs) {
            const auto res = b.result(q.id);
            if (res.end == End::Cancelled) continue;
            VJ_CHECK(res.tokens == reference(q.prompt, q.max_new, q.eos));
        }
        VJ_CHECK(b.stats().cancelled >= 3);
        b.clear_cache();
        VJ_CHECK(b.free_blocks() == cfg.kv_blocks && b.free_snapshots() == cfg.max_snapshots);  // zero leaks
    }
}

void test_memory_pressure() {
    std::mt19937 g(7);
    std::vector<batch::Request> reqs;
    for (int i = 0; i < 6; ++i) {
        batch::Request r;
        r.id = static_cast<uint64_t>(i + 1);
        r.prompt = rand_tokens(g, 200);
        r.max_new = 64;
        r.priority = i % 2;
        reqs.push_back(r);
    }
    batch::Request big;
    big.id = 99;
    big.prompt = rand_tokens(g, 1000);  // larger than the whole pool
    big.max_new = 4;
    std::vector<batch::Request> all = reqs;
    all.push_back(big);
    batch::BatchConfig cfg;
    cfg.max_batch = 4;
    cfg.kv_blocks = 40;  // 640 tokens for 4 slots of ~264 tokens: forced eviction + preemption
    ContinuousBatcher b(cfg);
    MockEngine e(cfg.block_tokens, cfg.kv_blocks, cfg.max_snapshots, false);
    for (const auto& r : all) VJ_CHECK(b.submit(r));
    uint64_t guard = 0;
    while (b.step(std::ref(e))) { VJ_CHECK(b.check_invariants()); VJ_CHECK(++guard < 100000); }
    VJ_CHECK(e.errors == 0 && b.active() == 0 && b.waiting() == 0);
    VJ_CHECK(b.result(99).end == End::Rejected);
    for (const auto& q : reqs) VJ_CHECK(b.result(q.id).tokens == reference(q.prompt, q.max_new, q.eos));
    std::printf("Memory pressure (40 blocks, 4 slots): %llu preemptions, outputs identical, oversize prompt rejected\n",
                static_cast<unsigned long long>(b.stats().preemptions));
    VJ_CHECK(b.stats().preemptions > 0);
    b.clear_cache();
    VJ_CHECK(b.free_blocks() == cfg.kv_blocks);
}

void test_random_stress() {
    std::mt19937 g(8);
    for (int trial = 0; trial < 6; ++trial) {
        batch::BatchConfig cfg;
        cfg.max_batch = 1 + static_cast<uint32_t>(trial) % 8;
        cfg.kv_blocks = 96 + 40 * static_cast<uint32_t>(trial);
        cfg.hybrid = trial % 2 == 1;
        cfg.max_snapshots = 3;
        cfg.prefill_chunk = 48;
        cfg.step_tokens = 96;
        ContinuousBatcher b(cfg);
        MockEngine e(cfg.block_tokens, cfg.kv_blocks, cfg.max_snapshots, cfg.hybrid);
        const auto sys = rand_tokens(g, 160);
        std::map<uint64_t, batch::Request> sent;
        std::uniform_int_distribution<int> coin(0, 99), plen(1, 200), gen(1, 40), pri(0, 3);
        uint64_t next = 1;
        for (int st = 0; st < 400; ++st) {
            if (coin(g) < 35) {
                batch::Request r;
                r.id = next++;
                r.prompt = coin(g) < 60 ? sys : std::vector<int32_t>{};
                auto t = rand_tokens(g, static_cast<size_t>(plen(g)));
                r.prompt.insert(r.prompt.end(), t.begin(), t.end());
                r.max_new = static_cast<uint32_t>(gen(g));
                r.priority = pri(g);
                r.eos = kEos;
                sent[r.id] = r;
                VJ_CHECK(b.submit(r));
            }
            if (coin(g) < 10 && next > 1) b.cancel(1 + static_cast<uint64_t>(coin(g)) % (next - 1));
            b.step(std::ref(e));
            VJ_CHECK(b.check_invariants());
        }
        b.run_until_idle(std::ref(e));
        VJ_CHECK(e.errors == 0 && b.active() == 0 && b.waiting() == 0 && b.check_invariants());
        for (const auto& kv : sent) {
            const auto res = b.result(kv.first);
            VJ_CHECK(res.end != End::None);
            if (res.end == End::Eos || res.end == End::Length)
                VJ_CHECK(res.tokens == reference(kv.second.prompt, kv.second.max_new, kv.second.eos));
        }
        b.clear_cache();
        VJ_CHECK(b.free_blocks() == cfg.kv_blocks && b.free_snapshots() == cfg.max_snapshots);
    }
}

} // namespace

int main() {
    test_radix_basics();
    test_radix_snapshots();
    test_radix_long_prompt();
    test_shared_system_prompt();
    test_batch_sizes();
    test_hybrid_snapshots();
    test_priority_preemption();
    test_cancellation();
    test_memory_pressure();
    test_random_stress();
    std::printf("test_prefix_batching: OK\n");
    return 0;
}
