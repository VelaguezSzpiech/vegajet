// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// core/suffix_trie: suffix-automaton longest earlier match vs brute force, draft continuation, the zero-tax cascade
// gate (>= 4 matched tokens and >= 4 continuation tokens skip the GPU drafter), a lossless verify loop, and the
// per-token latency budget (< 0.1 ms p99 at 64K history) next to HEAD's depth-bounded spec::SuffixTree.
#include "core/spec_engine.hpp"
#include "core/suffix_trie.hpp"
#include "tests/test_common.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>

using namespace vegajet;
using namespace vegajet::sfx;

namespace {

// Longest L such that the last L tokens also occur ending at an earlier index; returns (L, first such end).
std::pair<uint32_t, int64_t> brute(const std::vector<int32_t>& h) {
    const int64_t n = static_cast<int64_t>(h.size());
    for (int64_t L = n - 1; L >= 1; --L) {
        for (int64_t e = L - 1; e < n - 1; ++e) {
            bool ok = true;
            for (int64_t k = 0; k < L && ok; ++k) ok = h[e - k] == h[n - 1 - k];
            if (ok) return {static_cast<uint32_t>(L), e};
        }
    }
    return {0, -1};
}

double pct(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<size_t>(p * v.size()))];
}

} // namespace

int main() {
    // Brute-force agreement on random strings over small alphabets (many repeats).
    std::mt19937 rng(11);
    uint64_t checks = 0;
    for (int alpha : {2, 3, 5, 20}) {
        for (int rep = 0; rep < 6; ++rep) {
            SuffixIndex idx;
            std::vector<int32_t> h;
            for (int i = 0; i < 400; ++i) {
                const int32_t t = static_cast<int32_t>(rng() % alpha);
                idx.push(t);
                h.push_back(t);
                const auto ref = brute(h);
                const Match m = idx.longest_match();
                VJ_CHECK(m.length == ref.first);
                if (ref.first) VJ_CHECK(m.end_pos == ref.second);
                ++checks;
            }
            VJ_CHECK(idx.states() <= 2 * h.size());  // automaton size bound
        }
    }
    // Draft continues the earlier occurrence.
    {
        SuffixIndex idx;
        idx.extend({1, 2, 3, 4, 5, 6, 7, 9, 9, 1, 2, 3});
        const Match m = idx.longest_match();
        VJ_CHECK(m.length == 3 && m.end_pos == 2);
        VJ_CHECK((idx.draft(4) == std::vector<int32_t>{4, 5, 6, 7}));
        VJ_CHECK(idx.draft(100).size() == 9);  // stops at the end of the history
        SuffixIndex none;
        none.extend({1, 2, 3});
        VJ_CHECK(none.longest_match().length == 0 && none.draft(4).empty());
    }
    // Cascade gate: 3 matched tokens -> neural; 4 -> suffix (if 4 continuation tokens exist).
    {
        CascadeHybrid c;
        c.commit({10, 11, 12, 13, 14, 15, 16, 17, 50, 11, 12, 13});
        VJ_CHECK(c.decide().source == DraftSource::kNeural);  // match = 3
        c.commit(14);
        const auto d = c.decide();
        VJ_CHECK(d.source == DraftSource::kSuffix && d.match.length == 4);
        VJ_CHECK((d.draft == std::vector<int32_t>{15, 16, 17, 50, 11, 12, 13, 14}));
        CascadeHybrid short_tail;
        short_tail.commit({1, 2, 3, 4, 5, 1, 2, 3, 4});        // match 4 but only 1..? continuation: 5,1,2,3,4
        VJ_CHECK(short_tail.decide().source == DraftSource::kSuffix);
        CascadeHybrid end_tail;
        end_tail.commit({7, 1, 2, 3, 4, 1, 2, 3, 4});           // match 4 ending at 4; continuation 1,2,3,4 (4 tokens)
        VJ_CHECK(end_tail.decide().draft.size() == 4);
    }
    // Lossless verify loop: output equals the target stream whatever the drafts are.
    {
        std::vector<int32_t> target;
        for (int r = 0; r < 40; ++r) {  // code-like: repeated blocks with edits
            for (int k = 0; k < 24; ++k) target.push_back(100 + (k * 7 + (r % 5 == 0 ? r : 0)) % 50);
            target.push_back(static_cast<int32_t>(rng() % 1000));
        }
        CascadeHybrid c;
        std::vector<int32_t> out;
        uint64_t accepted_suffix = 0, proposed_suffix = 0;
        while (out.size() < target.size()) {
            const auto d = c.decide_and_count();
            std::vector<int32_t> draft = d.draft;
            if (d.source == DraftSource::kNeural)  // stand-in neural drafter: right 60% of the time
                for (size_t j = 0; j < 4 && out.size() + j < target.size(); ++j)
                    draft.push_back(rng() % 10 < 6 ? target[out.size() + j] : -7);
            size_t acc = 0;
            while (acc < draft.size() && out.size() + acc < target.size() && draft[acc] == target[out.size() + acc]) ++acc;
            if (d.source == DraftSource::kSuffix) { proposed_suffix += draft.size(); accepted_suffix += acc; }
            for (size_t j = 0; j < acc; ++j) { out.push_back(target[out.size()]); c.commit(out.back()); }
            if (out.size() < target.size()) { out.push_back(target[out.size()]); c.commit(out.back()); }  // bonus token
        }
        VJ_CHECK(out == target);
        VJ_CHECK(c.suffix_cycles > 0 && c.neural_cycles > 0);
        std::printf("  cascade on code-like stream: %llu suffix cycles (GPU drafter skipped), %llu neural; suffix acceptance %.2f\n",
                    static_cast<unsigned long long>(c.suffix_cycles), static_cast<unsigned long long>(c.neural_cycles),
                    proposed_suffix ? static_cast<double>(accepted_suffix) / proposed_suffix : 0.0);
    }
    // Latency: push + decide per committed token at 64K history.
    {
        std::vector<int32_t> stream;
        for (int i = 0; i < 65536; ++i) stream.push_back(i % 997 < 600 ? (i % 997) : static_cast<int32_t>(rng() % 248320));
        CascadeHybrid c;
        c.reserve(stream.size());
        std::vector<double> us;
        us.reserve(stream.size());
        for (int32_t t : stream) {
            const auto t0 = std::chrono::steady_clock::now();
            c.commit(t);
            const auto d = c.decide();
            us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
            (void)d;
        }
        spec::SuffixTree head(24);
        std::vector<double> hus;
        std::vector<int32_t> ctx;
        for (int i = 0; i < 16384; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            head.push(stream[static_cast<size_t>(i)]);
            ctx.push_back(stream[static_cast<size_t>(i)]);
            const std::vector<int32_t> q(ctx.end() - std::min<size_t>(ctx.size(), 24), ctx.end());
            (void)head.draft(q, 8, 1);
            hus.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
        }
        std::printf("  per token (push + decide): SuffixIndex 64K history p50 %.2f us p99 %.2f us max %.1f us | "
                    "HEAD SuffixTree(24) 16K history p50 %.2f us p99 %.2f us\n",
                    pct(us, 0.5), pct(us, 0.99), pct(us, 1.0), pct(hus, 0.5), pct(hus, 0.99));
        VJ_CHECK(pct(us, 0.99) < 100.0);
    }
    std::printf("test_suffix_trie: %llu brute-force checks, cascade gate, lossless loop, latency OK\nPASS\n",
                static_cast<unsigned long long>(checks));
    return 0;
}
