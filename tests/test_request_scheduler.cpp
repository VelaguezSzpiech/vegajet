// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// core/request_scheduler on the CPU with a deterministic mock model: the next token depends only on the previous
// token and its position, so every request has one reference continuation no matter how it is batched, drafted,
// chunked, preempted and recomputed. Checked after every planned step: scheduler + PagedKv invariants, the step
// token budget and slot limit, and that the KV of every planned sequence holds exactly the request's context (shared
// prefix blocks included). Scenarios: single request, shared-prefix reuse, slot-limited concurrency, chunked
// prefill interleaved with decode, OOM preemption with recompute, prompt larger than the pool, context full, cancel,
// EVICT-driven drafts; then 1,200 randomized arrival / disconnect / config sequences; then ServeLoop behind
// OpenAiServer (workers = slots) with concurrent SSE clients, disconnects and KV drain.
#include "core/request_scheduler.hpp"
#include "core/json_lite.hpp"
#include "tests/test_common.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <map>
#include <random>
#include <thread>

using namespace vegajet;
using namespace vegajet::sched;

namespace {

constexpr int32_t kEos = 2;

uint64_t mix(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33;
    return x;
}
// Token at ctx position p given the token at p - 1. Printable letters (+3) so the server test can show text.
int32_t next_token(int32_t prev, uint32_t p, uint32_t eos_mod) {
    const uint64_t h = mix(uint64_t(uint32_t(prev)) * 1000003u + p);
    if (eos_mod && h % eos_mod == 0) return kEos;
    return static_cast<int32_t>(3 + 'a' + h % 26);
}

std::vector<int32_t> reference(const std::vector<int32_t>& prompt, uint32_t n, uint32_t eos_mod) {
    std::vector<int32_t> out;
    int32_t prev = prompt.back();
    for (uint32_t p = static_cast<uint32_t>(prompt.size()); out.size() < n; ++p) {
        prev = next_token(prev, p, eos_mod);
        out.push_back(prev);
        if (prev == kEos) break;
    }
    return out;
}

struct MockModel : BatchEngine {
    uint32_t eos_mod = 0;
    int sleep_ms = 0;
    std::atomic<uint64_t> steps{0};
    std::vector<StepOutput> step(const StepPlan& plan) override {
        if (sleep_ms) std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        ++steps;
        std::vector<StepOutput> out;
        for (const auto& w : plan.items) {
            if (!w.sample) continue;
            StepOutput o;
            o.req = w.req;
            const size_t base = w.tokens.size() - 1 - w.n_draft;  // index of the last non-draft input
            int32_t prev = w.tokens[base];
            uint32_t p = w.pos0 + static_cast<uint32_t>(base) + 1;
            for (size_t i = 0;; ++i, ++p) {  // verify drafts left to right, then the bonus token
                const int32_t t = next_token(prev, p, eos_mod);
                o.tokens.push_back(t);
                if (i >= w.n_draft || t == kEos || w.tokens[base + 1 + i] != t) break;
                prev = t;
            }
            out.push_back(std::move(o));
        }
        return out;
    }
};

Drafter make_drafter(uint32_t eos_mod, double acc, uint32_t seed) {
    auto rng = std::make_shared<std::mt19937>(seed);
    return [=](uint64_t, const std::vector<int32_t>& ctx, uint32_t k) {
        std::vector<int32_t> d;
        int32_t prev = ctx.back();
        for (uint32_t i = 0; i < k; ++i) {
            const uint32_t p = static_cast<uint32_t>(ctx.size() + i);
            int32_t t = next_token(prev, p, eos_mod);
            if (std::uniform_real_distribution<double>(0, 1)(*rng) > acc) t = 3 + 'A' + static_cast<int32_t>((*rng)() % 26);
            d.push_back(t);
            prev = t;
        }
        return d;
    };
}

struct Tracked {
    std::vector<int32_t> prompt, out;
    uint32_t max_tokens = 0;
    EndReason end = EndReason::None;
    int ends = 0;
    std::string error;
};

// Drives a scheduler to idle; checks invariants + KV contents every step. Returns steps taken.
struct Driver {
    RequestScheduler& s;
    MockModel& m;
    Drafter drafter;
    std::map<uint64_t, Tracked> t;
    uint64_t mixed_steps = 0;
    void apply(const std::vector<Delivery>& d) {
        for (const auto& x : d) {
            Tracked& r = t.at(x.req);
            VJ_CHECK(r.end == EndReason::None);  // nothing after the end
            r.out.insert(r.out.end(), x.tokens.begin(), x.tokens.end());
            if (x.end != EndReason::None) { r.end = x.end; r.error = x.error; ++r.ends; }
        }
    }
    void step() {
        std::vector<Delivery> d;
        StepPlan plan = s.schedule(drafter, d);
        apply(d);
        std::string why;
        if (!s.check_invariants(&why)) { std::fprintf(stderr, "invariant: %s\n", why.c_str()); VJ_CHECK(false); }
        VJ_CHECK(plan.tokens <= s.config().step_token_budget && plan.items.size() <= s.config().max_slots);
        bool has_decode = false, has_prefill = false;
        for (const auto& w : plan.items) {
            has_decode |= w.decode;
            has_prefill |= !w.decode;
            const Tracked& r = t.at(w.req);
            // KV of the sequence == ctx[0 .. pos0 + n) (drafts: ctx then the drafts)
            std::vector<int32_t> kv;
            for (paged::BlockId b : s.kv().block_table(w.seq)) {
                const auto c = s.kv().block_contents(b);
                kv.insert(kv.end(), c.begin(), c.end());
            }
            VJ_CHECK(kv.size() == w.pos0 + w.tokens.size());
            for (uint32_t i = 0; i < w.pos0 + w.tokens.size() - w.n_draft; ++i) {
                const int32_t want = i < r.prompt.size() ? r.prompt[i] : r.out[i - r.prompt.size()];
                VJ_CHECK(kv[i] == want);
            }
        }
        mixed_steps += has_decode && has_prefill;
        d.clear();
        s.commit(plan, m.step(plan), d);
        apply(d);
        VJ_CHECK(s.check_invariants(&why));
    }
    uint64_t submit(std::vector<int32_t> prompt, uint32_t max_tokens) {
        const uint64_t id = s.submit(prompt, max_tokens, kEos);
        Tracked r;
        r.prompt = std::move(prompt);
        r.max_tokens = max_tokens;
        t[id] = r;
        return id;
    }
    uint64_t run_to_idle(uint64_t max_steps = 100000) {
        uint64_t n = 0;
        while (!s.idle()) { step(); VJ_CHECK(++n < max_steps); }
        return n;
    }
    // Every finished request matches the reference.
    void verify(uint32_t eos_mod, uint64_t capacity) {
        for (const auto& kvp : t) {
            const Tracked& r = kvp.second;
            VJ_CHECK(r.ends == 1);
            const std::vector<int32_t> ref = reference(r.prompt, r.max_tokens, eos_mod);
            VJ_CHECK(r.out.size() <= ref.size() && std::equal(r.out.begin(), r.out.end(), ref.begin()));
            switch (r.end) {
                case EndReason::Eos: VJ_CHECK(r.out.size() == ref.size() && r.out.back() == kEos); break;
                case EndReason::Length: VJ_CHECK(r.out.size() == r.max_tokens && ref.back() != kEos ? true : r.out.size() == ref.size()); break;
                case EndReason::ContextFull: VJ_CHECK(r.prompt.size() + r.out.size() + 1 >= capacity); break;
                case EndReason::Failed: VJ_CHECK(r.prompt.size() + r.out.size() >= capacity && !r.error.empty()); break;
                case EndReason::Cancelled: break;
                case EndReason::None: VJ_CHECK(false);
            }
        }
        VJ_CHECK(s.kv().stats().used == 0);
    }
};

std::vector<int32_t> rand_prompt(std::mt19937& rng, size_t n, const std::vector<int32_t>& prefix = {}) {
    std::vector<int32_t> p(prefix);
    while (p.size() < n) p.push_back(static_cast<int32_t>(3 + rng() % 200));
    return p;
}

// ------------------------------------------------------------------ minimal HTTP client for the server test
int connect_local(uint16_t port) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    timeval tv{20, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    VJ_CHECK(::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) == 0);
    return fd;
}

std::string stream_content(uint16_t port, const std::string& user, int max_tokens, bool disconnect_early) {
    const std::string body = "{\"messages\":[{\"role\":\"user\",\"content\":" + json::quote(user) +
                             "}],\"stream\":true,\"max_tokens\":" + std::to_string(max_tokens) + "}";
    const int fd = connect_local(port);
    const std::string req = "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: " +
                            std::to_string(body.size()) + "\r\n\r\n" + body;
    VJ_CHECK(::send(fd, req.data(), req.size(), MSG_NOSIGNAL) == static_cast<ssize_t>(req.size()));
    std::string all;
    char buf[4096];
    for (ssize_t n; (n = ::recv(fd, buf, sizeof(buf), 0)) > 0;) {
        all.append(buf, static_cast<size_t>(n));
        if (disconnect_early && all.find("\"content\":\"", all.find("\"content\":\"") + 1) != std::string::npos) break;
    }
    ::close(fd);
    if (disconnect_early) return {};
    std::string content;
    for (size_t pos = all.find("\r\n\r\n") + 4; pos < all.size();) {
        const size_t e = all.find("\n\n", pos);
        VJ_CHECK(e != std::string::npos);
        const std::string ev = all.substr(pos + 6, e - pos - 6);
        pos = e + 2;
        if (ev == "[DONE]") break;
        json::Value v;
        VJ_CHECK(json::parse(ev, v));
        const auto& ch = v.get("choices")->arr;
        if (ch.empty()) continue;
        if (const auto* c = ch[0].get("delta")->get("content"); c && c->is_string()) content += c->str;
    }
    return content;
}

}  // namespace

int main() {
    std::mt19937 rng(42);
    // ---- single request, chunked prefill, no drafts
    {
        RequestScheduler s(SchedulerConfig{2, 64, 16, 64, 8, false, 1});
        MockModel m;
        Driver d{s, m, nullptr, {}, 0};
        d.submit(rand_prompt(rng, 50), 20);
        const uint64_t steps = d.run_to_idle();
        d.verify(0, 64 * 8);
        VJ_CHECK(steps == 4 + 19 && s.stats().prefill_tokens == 50);  // ceil(50/16) prefill steps (last samples) + 19 decodes
    }
    // ---- shared system prompt: later requests attach the cached blocks
    {
        RequestScheduler s(SchedulerConfig{4, 256, 64, 128, 16, false, 1});
        MockModel m;
        Driver d{s, m, nullptr, {}, 0};
        const std::vector<int32_t> sys = rand_prompt(rng, 96);
        d.submit(rand_prompt(rng, 120, sys), 8);
        d.run_to_idle();
        for (int i = 0; i < 3; ++i) d.submit(rand_prompt(rng, 110 + i, sys), 8);
        d.run_to_idle();
        d.verify(0, 128 * 16);
        VJ_CHECK(s.stats().prefix_hit_tokens == 3 * 96);
        std::printf("  prefix cache: 3 requests reused %llu prompt tokens\n", (unsigned long long)s.stats().prefix_hit_tokens);
    }
    // ---- slot limit + chunked prefill interleaved with decode
    {
        RequestScheduler s(SchedulerConfig{4, 96, 32, 512, 16, false, 1});
        MockModel m;
        Driver d{s, m, nullptr, {}, 0};
        for (int i = 0; i < 6; ++i) d.submit(rand_prompt(rng, 20), 40);
        for (int i = 0; i < 8; ++i) d.step();
        d.submit(rand_prompt(rng, 900), 10);  // long prompt arrives while others decode
        d.run_to_idle();
        d.verify(0, 512 * 16);
        VJ_CHECK(s.stats().max_active == 4 && s.stats().max_decode_batch == 4 && d.mixed_steps >= 10);
        std::printf("  interleave: %llu steps mixed prefill chunks with decode, max decode batch %u\n",
                    (unsigned long long)d.mixed_steps, s.stats().max_decode_batch);
    }
    // ---- OOM: a small pool forces preemption + recompute; outputs unchanged
    {
        RequestScheduler s(SchedulerConfig{4, 128, 32, 24, 8, false, 0});  // 192 tokens of KV for 4 x ~110
        MockModel m;
        Driver d{s, m, nullptr, {}, 0};
        for (int i = 0; i < 6; ++i) d.submit(rand_prompt(rng, 40 + 5 * i), 70);
        d.run_to_idle();
        d.verify(0, 24 * 8);
        VJ_CHECK(s.stats().preemptions > 0 && s.stats().finished == 6);
        std::printf("  OOM: %llu preemptions, all 6 outputs identical to the reference\n", (unsigned long long)s.stats().preemptions);
    }
    // ---- prompt larger than the pool -> Failed; generation filling the pool alone -> ContextFull
    {
        RequestScheduler s(SchedulerConfig{2, 64, 32, 8, 8, false, 0});
        MockModel m;
        Driver d{s, m, nullptr, {}, 0};
        const uint64_t big = d.submit(rand_prompt(rng, 65), 4);
        const uint64_t full = d.submit(rand_prompt(rng, 50), 100);
        d.run_to_idle();
        d.verify(0, 64);
        VJ_CHECK(d.t[big].end == EndReason::Failed && d.t[big].error.find("exceeds the KV pool") != std::string::npos);
        std::printf("  context full after %zu tokens (prompt 50, pool 64)\n", d.t[full].out.size());
        VJ_CHECK(d.t[full].end == EndReason::ContextFull && d.t[full].out.size() == 15);  // KV = 50 + 15 - 1 = 64
    }
    // ---- cancel while waiting and while running
    {
        RequestScheduler s(SchedulerConfig{1, 64, 16, 64, 8, false, 1});
        MockModel m;
        Driver d{s, m, nullptr, {}, 0};
        const uint64_t a = d.submit(rand_prompt(rng, 30), 50), b = d.submit(rand_prompt(rng, 30), 50);
        for (int i = 0; i < 5; ++i) d.step();
        s.cancel(a);
        s.cancel(b);
        d.run_to_idle();
        d.verify(0, 512);
        VJ_CHECK(d.t[a].end == EndReason::Cancelled && !d.t[a].out.empty() && d.t[b].end == EndReason::Cancelled && d.t[b].out.empty());
    }
    // ---- speculative decoding with EVICT draft depth
    {
        evict::EvictController ctl(evict::ControllerConfig{4, 64, 2, 8, 3.0, 0.25}, evict::CostModel{100, 5, 2});
        RequestScheduler s(SchedulerConfig{3, 128, 32, 256, 16, true, 1}, &ctl);
        MockModel m;
        Driver d{s, m, make_drafter(0, 0.8, 7), {}, 0};
        for (int i = 0; i < 3; ++i) d.submit(rand_prompt(rng, 30), 200);
        const uint64_t steps = d.run_to_idle();
        d.verify(0, 256 * 16);
        const auto& st = s.stats();
        VJ_CHECK(st.drafts_accepted > 0 && st.decode_steps < 3 * 199 && st.generated == 600);
        std::printf("  EVICT drafts: %llu proposed, %llu accepted (%.2f), %llu decode passes for 600 tokens, %llu steps, depth now %u\n",
                    (unsigned long long)st.drafts_proposed, (unsigned long long)st.drafts_accepted,
                    double(st.drafts_accepted) / std::max<uint64_t>(1, st.drafts_proposed), (unsigned long long)st.decode_steps,
                    (unsigned long long)steps, ctl.next_depth());
    }
    // ---- fuzz: 1,200 randomized sequences
    {
        SchedulerStats total;
        uint64_t requests = 0, steps = 0;
        for (uint32_t seq = 0; seq < 1200; ++seq) {
            std::mt19937 r(1000 + seq);
            SchedulerConfig c;
            c.max_slots = 1 + r() % 6;
            c.block_tokens = 1u << (2 + r() % 3);                  // 4, 8, 16
            c.kv_blocks = 4 + r() % 60;
            c.step_token_budget = c.max_slots + r() % 128;
            c.prefill_chunk = 1 + r() % 64;
            c.speculative = r() % 2;
            c.admit_reserve_blocks = r() % 3;
            const uint32_t eos_mod = r() % 3 ? 0 : 20 + r() % 60;
            evict::EvictController ctl(evict::ControllerConfig{4, 64, static_cast<uint32_t>(1 + r() % 3), static_cast<uint32_t>(4 + r() % 8), 3.0, 0.25}, evict::CostModel{100, 5, 2});
            RequestScheduler s(c, c.speculative ? &ctl : nullptr);
            MockModel m;
            m.eos_mod = eos_mod;
            Driver d{s, m, c.speculative ? make_drafter(eos_mod, 0.3 + 0.6 * (r() % 100) / 100.0, seq) : Drafter{}, {}, 0};
            const uint64_t capacity = uint64_t(c.kv_blocks) * c.block_tokens;
            std::vector<std::vector<int32_t>> prefixes;
            for (int k = 0; k < 3; ++k) prefixes.push_back(rand_prompt(r, 4 + r() % 40));
            const uint32_t n_req = 2 + r() % 14;
            uint32_t submitted = 0;
            std::vector<uint64_t> ids;
            for (uint64_t step = 0; submitted < n_req || !s.idle(); ++step) {
                VJ_CHECK(step < 200000);
                while (submitted < n_req && r() % 3 == 0) {
                    const size_t len = 1 + r() % std::min<uint64_t>(capacity + 8, 120);
                    std::vector<int32_t> p = r() % 2 ? rand_prompt(r, std::max(len, prefixes[r() % 3].size() + 1), prefixes[r() % 3])
                                                     : rand_prompt(r, len);
                    ids.push_back(d.submit(std::move(p), 1 + r() % 60));
                    ++submitted;
                }
                if (!ids.empty() && r() % 23 == 0) s.cancel(ids[r() % ids.size()]);  // disconnects (also of finished ids)
                if (!s.idle()) d.step();
                ++steps;
            }
            d.verify(eos_mod, capacity);
            requests += n_req;
            const auto& st = s.stats();
            total.preemptions += st.preemptions;
            total.prefix_hit_tokens += st.prefix_hit_tokens;
            total.drafts_accepted += st.drafts_accepted;
            total.cancelled += st.cancelled;
            total.failed += st.failed;
            total.context_full += st.context_full;
            total.finished += st.finished;
            total.draft_drops += st.draft_drops;
        }
        std::printf("  fuzz: 1200 sequences, %llu requests, %llu steps: %llu finished, %llu cancelled, %llu failed, "
                    "%llu context-full, %llu preemptions, %llu draft drops, %llu prefix-hit tokens, %llu drafts accepted\n",
                    (unsigned long long)requests, (unsigned long long)steps, (unsigned long long)total.finished,
                    (unsigned long long)total.cancelled, (unsigned long long)total.failed, (unsigned long long)total.context_full,
                    (unsigned long long)total.preemptions, (unsigned long long)total.draft_drops,
                    (unsigned long long)total.prefix_hit_tokens, (unsigned long long)total.drafts_accepted);
        VJ_CHECK(total.preemptions > 100 && total.cancelled > 100 && total.prefix_hit_tokens > 1000 &&
                 total.drafts_accepted > 1000 && total.failed > 0 && total.context_full > 0);
    }
    // ---- ServeLoop behind OpenAiServer: concurrent SSE clients + disconnects
    {
        MockModel m;
        m.sleep_ms = 2;
        ServeHooks h;
        h.eos = kEos;
        h.encode = [](const serve::GenerateRequest& r) {
            std::vector<int32_t> t;
            for (unsigned char c : r.messages.back().content) t.push_back(3 + c);
            return t;
        };
        h.decode = [](int32_t t) { return std::string(1, static_cast<char>(t - 3)); };
        evict::EvictController ctl(evict::ControllerConfig{4, 64, 2, 6, 3.0, 0.25}, evict::CostModel{100, 5, 2});
        h.drafter = make_drafter(0, 0.7, 3);
        SchedulerConfig c{4, 64, 16, 64, 8, true, 1};
        ServeLoop loop(c, m, h, &ctl);
        serve::ServerOptions so;
        so.port = 0;
        so.workers = 4;
        so.max_queue = 16;
        serve::OpenAiServer srv(so, loop.generate_fn());
        std::string err;
        VJ_CHECK(srv.start(&err));
        std::atomic<int> good{0};
        std::vector<std::thread> cl;
        for (int i = 0; i < 10; ++i)
            cl.emplace_back([&, i] {
                const std::string user = "prompt number " + std::to_string(i) + " about gfx906";
                const bool drop = i % 4 == 3;
                const std::string got = stream_content(srv.port(), user, 60, drop);
                if (drop) { ++good; return; }
                std::vector<int32_t> pt;
                for (unsigned char ch : user) pt.push_back(3 + ch);
                std::string want;
                for (int32_t t : reference(pt, 60, 0)) want += static_cast<char>(t - 3);
                if (got == want) ++good;
                else std::fprintf(stderr, "client %d: got '%s' want '%s'\n", i, got.c_str(), want.c_str());
            });
        for (auto& t : cl) t.join();
        for (int i = 0; i < 200 && !loop.kv_drained(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        const SchedulerStats st = loop.stats();
        std::printf("  ServeLoop + OpenAiServer: 10 clients (3 disconnect early), max active %u, max decode batch %u, "
                    "%llu cancelled, %llu drafts accepted, %llu engine steps, KV drained %s\n",
                    st.max_active, st.max_decode_batch, (unsigned long long)st.cancelled,
                    (unsigned long long)st.drafts_accepted, (unsigned long long)m.steps.load(), loop.kv_drained() ? "yes" : "no");
        VJ_CHECK(good == 10 && st.max_active >= 3 && st.max_decode_batch >= 3 && st.cancelled >= 1 && loop.kv_drained());
        srv.stop();
    }
    std::printf("test_request_scheduler: single, prefix reuse, slots, interleave, OOM preemption, failed/context-full, "
                "cancel, EVICT drafts, 1200-sequence fuzz, ServeLoop+server OK\nPASS\n");
    return 0;
}
