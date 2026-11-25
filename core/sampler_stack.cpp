// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/sampler_stack.hpp"

#include "core/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <unordered_set>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vegajet::sampling {

namespace {

constexpr uint64_t kStreamDraw = 0x0;
constexpr uint64_t kStreamXtc = 0x5854430000000000ull;  // "XTC"
constexpr size_t kBlock = 64;
constexpr size_t kMaxHistory = 1u << 16;

// GCC / Clang vector extensions: 4 floats per value (16 bytes: no ABI change), lowered to SSE2 / AVX / NEON. Explicit, because the
// auto-vectorizer leaves the clamped polynomial exp scalar.
// x86-64 GCC >= 12: an extra AVX2 + FMA clone picked at load time (ifunc); elsewhere the baseline build only.
#if defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 12
#define VJ_SIMD_CLONES __attribute__((target_clones("arch=x86-64-v3", "default")))
#else
#define VJ_SIMD_CLONES
#endif
typedef float v4f __attribute__((vector_size(16)));
typedef uint32_t v4u __attribute__((vector_size(16)));

// 2^f on [-0.5, 0.5] (degree-5 fit, relative error 7.7e-8) times 2^n assembled in the exponent bits; x <= 0.
inline v4f exp_v4(v4f x) {
    const v4f lo = v4f{} - 87.0f;
    x = x < lo ? lo : x;
    const v4f t = x * 1.44269504088896341f;
    const v4f magic = v4f{} + 12582912.0f;  // 1.5 * 2^23: round-to-nearest through the FP adder
    const v4f r = t + magic;
    const v4f f = t - (r - magic);
    // Estrin: three dependent steps instead of five (the loop is latency bound otherwise).
    const v4f f2 = f * f;
    const v4f p01 = f * 6.9314694916e-1f + 1.0000000710f;
    const v4f p23 = f * 5.5507426160e-2f + 2.4022121754e-1f;
    const v4f p45 = f * 1.3266970385e-3f + 9.6754597454e-3f;
    const v4f p = p01 + f2 * (p23 + f2 * p45);
    v4u bits;
    std::memcpy(&bits, &r, sizeof bits);
    const v4u e = (bits - 0x4B400000u + 127u) << 23;  // biased exponent n + 127 in bits 23..30
    v4f scale;
    std::memcpy(&scale, &e, sizeof scale);
    return p * scale;
}

inline float exp_fast(float x) {
    const v4f v = v4f{} + x;
    return exp_v4(v)[0];
}

bool cand_before(const SamplerStack::Cand& a, const SamplerStack::Cand& b) {
    return a.logit > b.logit || (a.logit == b.logit && a.id < b.id);
}

// p = softmax over the candidates (sorted by logit), plus `tail` mass (relative to exp(c[0].logit - m)... see below).
// tail is expressed relative to the same shift m; returns the normalizer.
double softmax(std::vector<SamplerStack::Cand>& c, double tail, double m, float inv_t = 1.0f) {
    double z = tail;
    for (auto& x : c) {
        const double e = std::exp((static_cast<double>(x.logit) - m) * inv_t);
        x.p = static_cast<float>(e);
        z += e;
    }
    for (auto& x : c) x.p = static_cast<float>(x.p / z);
    return z;
}

} // namespace

namespace {
VJ_SIMD_CLONES
float block_max64(const float* x) {
    v4f acc[4];
    std::memcpy(acc, x, sizeof acc);
    for (int k = 16; k < 64; k += 16) {
        v4f v[4];
        std::memcpy(v, x + k, sizeof v);
        for (int q = 0; q < 4; ++q) acc[q] = v[q] > acc[q] ? v[q] : acc[q];
    }
    acc[0] = acc[1] > acc[0] ? acc[1] : acc[0];
    acc[2] = acc[3] > acc[2] ? acc[3] : acc[2];
    acc[0] = acc[2] > acc[0] ? acc[2] : acc[0];
    float mx = acc[0][0];
    for (int k = 1; k < 4; ++k) mx = acc[0][k] > mx ? acc[0][k] : mx;
    return mx;
}
} // namespace

namespace {
constexpr size_t kExpChunk = 4096;  // float lanes within a chunk, double across chunks: ~1e-7 over 248K terms

VJ_SIMD_CLONES
double sum_exp_chunk(const float* x, size_t n, float m, float inv_t) {
    v4f acc[4] = {};
    size_t j = 0;
    for (; j + 16 <= n; j += 16) {
        v4f v[4];
        std::memcpy(v, x + j, sizeof v);
        for (int q = 0; q < 4; ++q) acc[q] += exp_v4((v[q] - m) * inv_t);
    }
    double s = 0;
    for (int q = 0; q < 4; ++q)
        for (int k = 0; k < 4; ++k) s += acc[q][k];
    for (; j < n; ++j) s += exp_fast((x[j] - m) * inv_t);
    return s;
}
// Chunk sum shifted by the chunk's own maximum: independent of the global maximum, so it can run while the
// candidate pass is still finding it. Returns the chunk max in *mc (sum = 0, mc = -inf for an all -inf chunk).
VJ_SIMD_CLONES
double sum_exp_chunk_local(const float* x, size_t n, float inv_t, float* mc) {
    float mx = -std::numeric_limits<float>::infinity();
    size_t j = 0;
    for (; j + 64 <= n; j += 64) {
        const float b = block_max64(x + j);
        mx = b > mx ? b : mx;
    }
    for (; j < n; ++j) mx = x[j] > mx ? x[j] : mx;
    *mc = mx;
    if (!std::isfinite(mx)) return 0.0;
    return sum_exp_chunk(x, n, mx, inv_t);
}
} // namespace

double sum_exp_shifted(const float* x, size_t n, float m, float inv_t, int threads) {
    const size_t chunks = (n + kExpChunk - 1) / kExpChunk;
    double stack_part[128];
    std::vector<double> heap_part;
    double* part = stack_part;
    if (chunks > 128) { heap_part.resize(chunks); part = heap_part.data(); }
    // Fixed chunk partition and a fixed-order final sum: the result is identical for any thread count.
#ifdef _OPENMP
    if (threads > 1 && n >= (1u << 16)) {
        const long nc = static_cast<long>(chunks);
#pragma omp parallel for num_threads(threads) schedule(static)
        for (long c = 0; c < nc; ++c) {
            const size_t a = static_cast<size_t>(c) * kExpChunk;
            part[c] = sum_exp_chunk(x + a, std::min(kExpChunk, n - a), m, inv_t);
        }
    } else
#endif
    {
        (void)threads;
        for (size_t c = 0; c < chunks; ++c) {
            const size_t a = c * kExpChunk;
            part[c] = sum_exp_chunk(x + a, std::min(kExpChunk, n - a), m, inv_t);
        }
    }
    double total = 0.0;
    for (size_t c = 0; c < chunks; ++c) total += part[c];
    return total;
}


SamplerStack::SamplerStack(StackParams p) : p_(std::move(p)) { reset(); }

void SamplerStack::reset() {
    hist_.clear();
    step_ = 0;
    mu_ = 2.0f * p_.mirostat_tau;
}

void SamplerStack::accept(int32_t id) {
    hist_.push_back(id);
    if (hist_.size() > kMaxHistory) hist_.erase(hist_.begin(), hist_.begin() + static_cast<long>(hist_.size() - kMaxHistory / 2));
    ++step_;
}

namespace {

// Streaming top-k over consecutive ranges (k-th largest threshold; whole 64-logit blocks below it are skipped);
// overridden ids are skipped (their adjusted logits are added by the caller).
struct Collector {
    size_t k = 1, cap = 64;
    bool full = false;
    const uint32_t* stamp = nullptr;
    uint32_t epoch = 0;
    float thr = -std::numeric_limits<float>::infinity();
    float raw_max = -std::numeric_limits<float>::infinity();
    std::vector<SamplerStack::Cand> out;

    void init(size_t k_, bool full_, size_t n, const uint32_t* st, uint32_t ep) {
        k = k_;
        full = full_;
        stamp = st;
        epoch = ep;
        // Compacting at 2k keeps the threshold close to the running k-th largest, so most blocks are skipped whole.
        cap = full ? n : std::max<size_t>(2 * k, 64);
        thr = raw_max = -std::numeric_limits<float>::infinity();
        out.clear();
        out.reserve(std::min(n, cap + kBlock));
    }
    void feed(const float* logits, size_t b0, size_t e0) {
        for (size_t b = b0; b < e0; b += kBlock) {
            const size_t e = std::min(e0, b + kBlock);
            float bmax;
            if (e - b == kBlock) bmax = block_max64(logits + b);
            else {
                bmax = logits[b];
                for (size_t i = b + 1; i < e; ++i) bmax = logits[i] > bmax ? logits[i] : bmax;
            }
            raw_max = std::max(raw_max, bmax);
            if (!full && bmax < thr) continue;
            for (size_t i = b; i < e; ++i) {
                const float l = logits[i];
                if ((full || l >= thr) && !(stamp && stamp[i] == epoch) && !std::isnan(l))
                    out.push_back({l, static_cast<int32_t>(i), 0.0f});
            }
            if (!full && out.size() >= cap) {
                std::nth_element(out.begin(), out.begin() + static_cast<long>(k - 1), out.end(), cand_before);
                out.resize(k);
                thr = out[0].logit;
                for (const auto& c : out) thr = std::min(thr, c.logit);
            }
        }
    }
};

} // namespace

// Top-k candidates (logit desc, id asc) with overrides applied; raw_max = max over the raw array.
void SamplerStack::collect(const float* logits, size_t n_vocab, size_t k, std::vector<Cand>& out, float& raw_max) {
    k = std::max<size_t>(1, std::min(k, n_vocab));
    const bool full = k >= n_vocab;
    Collector col;
    col.init(k, full, n_vocab, overrides_.empty() ? nullptr : stamp_.data(), epoch_);
    col.feed(logits, 0, n_vocab);
    out.swap(col.out);
    raw_max = col.raw_max;
    for (const auto& o : overrides_)
        if (!std::isnan(o.second)) out.push_back({o.second, o.first, 0.0f});
    if (out.size() > k) {
        std::nth_element(out.begin(), out.begin() + static_cast<long>(k - 1), out.end(), cand_before);
        out.resize(k);
    }
    std::sort(out.begin(), out.end(), cand_before);
}

bool SamplerStack::run_chain(std::vector<Cand>& c, double tail, double m, int32_t& chosen, bool& need_full) {
    const auto& P = p_;
    need_full = false;
    const double u = counter_uniform(P.seed ^ kStreamDraw, step_);
    auto draw = [&](const std::vector<Cand>& cs, double tail_mass) -> int {
        double acc = 0;
        for (size_t i = 0; i < cs.size(); ++i) {
            acc += cs[i].p;
            if (u < acc) return static_cast<int>(i);
        }
        if (tail_mass > 0 && u >= acc) return -1;  // the draw lands in the tail
        return static_cast<int>(cs.size()) - 1;
    };
    auto drop_inf = [&] {
        while (c.size() > 1 && std::isinf(c.back().logit) && c.back().logit < 0) c.pop_back();
    };
    drop_inf();
    if (P.mirostat == 2) {
        const float inv_t = P.temperature > 0 ? 1.0f / P.temperature : 1.0f;
        if (P.temperature <= 0) { c.resize(1); chosen = c[0].id; return true; }
        softmax(c, tail, m, inv_t);
        size_t keep = 0;
        while (keep < c.size() && -std::log2(static_cast<double>(c[keep].p)) <= mu_) ++keep;
        // Every candidate passes mu: tail tokens (all less likely) may pass too, so redo over the whole vocabulary.
        if (keep == c.size() && tail > 0) { need_full = true; return false; }
        c.resize(std::max<size_t>(1, keep));
        softmax(c, 0.0, m, inv_t);
        const int idx = std::max(0, draw(c, 0.0));
        chosen = c[static_cast<size_t>(idx)].id;
        const double surprise = -std::log2(static_cast<double>(c[static_cast<size_t>(idx)].p));
        mu_ = static_cast<float>(mu_ - P.mirostat_eta * (surprise - P.mirostat_tau));
        return true;
    }
    // top-k: c already holds at most top_k candidates (collect)
    if (P.top_k > 0) tail = 0.0;
    if (P.typical_p < 1.0f) {
        if (tail > 0) { need_full = true; return false; }
        softmax(c, 0.0, m);
        double ent = 0;
        for (const auto& x : c) if (x.p > 0) ent -= x.p * std::log(static_cast<double>(x.p));
        std::vector<std::pair<double, size_t>> sh;
        for (size_t i = 0; i < c.size(); ++i) sh.push_back({std::fabs(-std::log(static_cast<double>(c[i].p)) - ent), i});
        std::stable_sort(sh.begin(), sh.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        double cum = 0;
        size_t last = sh.size();
        for (size_t i = 0; i < sh.size(); ++i) {
            cum += c[sh[i].second].p;
            if (cum > P.typical_p) { last = i + 1; break; }
        }
        std::vector<Cand> kept;
        for (size_t i = 0; i < last; ++i) kept.push_back(c[sh[i].second]);
        std::sort(kept.begin(), kept.end(), cand_before);
        c.swap(kept);
        tail = 0.0;
    }
    if (P.top_p < 1.0f) {
        softmax(c, tail, m);
        double cum = 0;
        size_t last = c.size();
        bool hit = false;
        for (size_t i = 0; i < c.size(); ++i) {
            cum += c[i].p;
            if (cum >= P.top_p) { last = i + 1; hit = true; break; }
        }
        if (!hit && tail > 0) { need_full = true; return false; }
        c.resize(last);
        tail = 0.0;
    }
    if (P.min_p > 0.0f) {
        const double floor_logit = static_cast<double>(c[0].logit) + std::log(static_cast<double>(P.min_p));
        if (tail > 0 && static_cast<double>(c.back().logit) >= floor_logit) { need_full = true; return false; }
        size_t keep = 0;
        while (keep < c.size() && static_cast<double>(c[keep].logit) >= floor_logit) ++keep;
        c.resize(std::max<size_t>(1, keep));
        tail = 0.0;
    }
    if (P.xtc_probability > 0.0f && P.xtc_threshold <= 0.5f && c.size() >= 2 &&
        counter_uniform(P.seed ^ kStreamXtc, step_) <= P.xtc_probability) {
        softmax(c, tail, m);
        size_t pos_last = 0;
        for (size_t i = 0; i < c.size(); ++i) {
            if (c[i].p >= P.xtc_threshold) pos_last = i;
            else break;
        }
        if (pos_last > 0) { c.erase(c.begin(), c.begin() + static_cast<long>(pos_last)); tail = 0.0; }
    }
    if (P.temperature <= 0.0f) { chosen = c[0].id; c.resize(1); return true; }
    softmax(c, tail, m, 1.0f / P.temperature);
    const int idx = draw(c, tail);
    if (idx < 0) { need_full = true; return false; }
    chosen = c[static_cast<size_t>(idx)].id;
    return true;
}

SampleResult SamplerStack::sample(const float* logits, size_t n_vocab) {
    SampleResult res;
    if (!logits || n_vocab == 0) return res;
    const auto& P = p_;
    // ---- sparse logit edits: bias, penalties, DRY
    if (stamp_.size() != n_vocab) { stamp_.assign(n_vocab, 0u); epoch_ = 0; }
    if (++epoch_ == 0) { std::fill(stamp_.begin(), stamp_.end(), 0u); epoch_ = 1; }
    if (slot_.size() != n_vocab) slot_.assign(n_vocab, 0u);
    overrides_.clear();
    auto edit = [&](int32_t id) -> float& {
        const auto i = static_cast<size_t>(id);
        if (stamp_[i] != epoch_) {
            stamp_[i] = epoch_;
            slot_[i] = static_cast<uint32_t>(overrides_.size());
            overrides_.push_back({id, logits[id]});
        }
        return overrides_[slot_[i]].second;
    };
    for (const auto& b : P.logit_bias)
        if (b.first >= 0 && static_cast<size_t>(b.first) < n_vocab) edit(b.first) += b.second;
    const bool penalties = P.penalty_last_n != 0 &&
                           (P.repeat_penalty != 1.0f || P.presence_penalty != 0.0f || P.frequency_penalty != 0.0f);
    if (penalties && !hist_.empty()) {
        const size_t n = P.penalty_last_n < 0 ? hist_.size() : std::min(hist_.size(), static_cast<size_t>(P.penalty_last_n));
        std::vector<int32_t> win(hist_.end() - static_cast<long>(n), hist_.end());
        std::sort(win.begin(), win.end());
        for (size_t i = 0; i < win.size();) {
            size_t j = i;
            while (j < win.size() && win[j] == win[i]) ++j;
            const int32_t id = win[i], count = static_cast<int32_t>(j - i);
            i = j;
            if (id < 0 || static_cast<size_t>(id) >= n_vocab) continue;
            float& l = edit(id);
            if (P.repeat_penalty != 1.0f) l = l > 0 ? l / P.repeat_penalty : l * P.repeat_penalty;
            l -= static_cast<float>(count) * P.frequency_penalty + P.presence_penalty;
        }
    }
    if (P.dry_multiplier > 0.0f && P.dry_allowed_length >= 1 && hist_.size() >= 2) {
        const size_t window = P.dry_last_n < 0 ? std::min<size_t>(hist_.size(), 8192)
                                               : std::min(hist_.size(), static_cast<size_t>(P.dry_last_n));
        std::vector<int32_t> r(hist_.end() - static_cast<long>(window), hist_.end());
        std::reverse(r.begin(), r.end());  // r[0] = last token
        const std::unordered_set<int32_t> brk(P.dry_breakers.begin(), P.dry_breakers.end());
        const size_t n = r.size();
        std::vector<size_t> free_run(n + 1, 0);  // breaker-free tokens starting at i
        for (size_t i = n; i-- > 0;) free_run[i] = brk.count(r[i]) ? 0 : free_run[i + 1] + 1;
        std::vector<size_t> z(n, 0);  // Z-function of r
        for (size_t i = 1, lo = 0, hi = 0; i < n; ++i) {
            if (i < hi) z[i] = std::min(hi - i, z[i - lo]);
            while (i + z[i] < n && r[z[i]] == r[i + z[i]]) ++z[i];
            if (i + z[i] > hi) { lo = i; hi = i + z[i]; }
        }
        std::unordered_map<int32_t, size_t> best;
        for (size_t i = 1; i < n; ++i) {
            const size_t len = std::min({z[i], free_run[i], free_run[0]});
            const int32_t next = r[i - 1];  // the token that followed the earlier occurrence
            if (len >= static_cast<size_t>(P.dry_allowed_length) && !brk.count(next)) {
                size_t& b = best[next];
                b = std::max(b, len);
            }
        }
        for (const auto& kv : best) {
            if (kv.first < 0 || static_cast<size_t>(kv.first) >= n_vocab) continue;
            const double ex = std::min<double>(40.0, static_cast<double>(kv.second) - P.dry_allowed_length);
            edit(kv.first) -= static_cast<float>(P.dry_multiplier * std::pow(static_cast<double>(P.dry_base), ex));
        }
    }

    // ---- candidates + normalizers
    const size_t n_lp = static_cast<size_t>(std::max(0, std::min(20, P.n_logprobs)));
    size_t k = P.top_k > 0 && P.mirostat != 2 ? static_cast<size_t>(P.top_k) : std::max<size_t>(P.candidate_cap, 1);
    k = std::max(k, n_lp);
    std::vector<Cand> c;
    float raw_max = 0;
    const bool need_z1 = n_lp > 0 || (P.top_k <= 0 && P.mirostat != 2);
    const bool need_zt = P.mirostat == 2 && P.temperature > 0;
    // The first normalizer (temperature 1, or 1/T for Mirostat) is summed per fixed chunk with a per-chunk shift,
    // concurrently with the candidate pass on thread 0; chunks combine in index order (thread-count independent).
    const float inv_first = need_z1 ? 1.0f : (need_zt ? 1.0f / P.temperature : 1.0f);
    const size_t chunks = (n_vocab + kExpChunk - 1) / kExpChunk;
    std::vector<float> cmax(need_z1 || need_zt ? chunks : 0);
    std::vector<double> csum(cmax.size());
    auto chunk_job = [&](size_t ch) {
        const size_t a = ch * kExpChunk;
        csum[ch] = sum_exp_chunk_local(logits + a, std::min(kExpChunk, n_vocab - a), inv_first, &cmax[ch]);
    };
    int team = 1;
#ifdef _OPENMP
    if (P.threads > 1 && n_vocab >= (1u << 16)) team = std::min(P.threads, 16);
#endif
    if (team > 1) {
        // One read of the logits, split over the team: each thread owns a contiguous chunk range and, per 16 KB chunk,
        // runs the candidate scan and (when needed) the chunk's exp-sum while the chunk is in L1. Candidates merge by
        // the same total order, chunk sums combine in index order: identical results for any team size.
        std::vector<Collector> cols(static_cast<size_t>(team));
        const size_t kk = std::max<size_t>(1, std::min(k, n_vocab));
        const uint32_t* st = overrides_.empty() ? nullptr : stamp_.data();
#ifdef _OPENMP
#pragma omp parallel num_threads(team)
#endif
        {
#ifdef _OPENMP
            const int t = omp_get_thread_num(), nt = omp_get_num_threads();
#else
            const int t = 0, nt = 1;
#endif
            const size_t c0 = chunks * static_cast<size_t>(t) / static_cast<size_t>(nt);
            const size_t c1 = chunks * static_cast<size_t>(t + 1) / static_cast<size_t>(nt);
            Collector& col = cols[static_cast<size_t>(t)];
            col.init(kk, false, n_vocab, st, epoch_);
            for (size_t ch = c0; ch < c1; ++ch) {
                const size_t a = ch * kExpChunk, e = std::min(n_vocab, a + kExpChunk);
                col.feed(logits, a, e);
                if (!cmax.empty()) chunk_job(ch);
            }
        }
        c.clear();
        raw_max = -std::numeric_limits<float>::infinity();
        for (auto& col : cols) {
            c.insert(c.end(), col.out.begin(), col.out.end());
            raw_max = std::max(raw_max, col.raw_max);
        }
        for (const auto& o : overrides_)
            if (!std::isnan(o.second)) c.push_back({o.second, o.first, 0.0f});
        if (c.size() > kk) {
            std::nth_element(c.begin(), c.begin() + static_cast<long>(kk - 1), c.end(), cand_before);
            c.resize(kk);
        }
        std::sort(c.begin(), c.end(), cand_before);
    } else {
        collect(logits, n_vocab, k, c, raw_max);
        for (size_t ch = 0; ch < cmax.size(); ++ch) chunk_job(ch);
    }
    float m = raw_max;
    for (const auto& o : overrides_) if (o.second > m && std::isfinite(o.second)) m = o.second;
    if (!std::isfinite(m)) m = 0.0f;
    auto full_z = [&](float inv_t) {
        double z = 0.0;
        if (inv_t == inv_first && !cmax.empty()) {
            for (size_t ch = 0; ch < chunks; ++ch)
                if (csum[ch] > 0) z += csum[ch] * std::exp((static_cast<double>(cmax[ch]) - m) * inv_t);
        } else {
            z = sum_exp_shifted(logits, n_vocab, m, inv_t, P.threads);
        }
        for (const auto& o : overrides_) {
            z -= static_cast<double>(exp_fast((logits[o.first] - m) * inv_t));
            if (std::isfinite(o.second)) z += std::exp((static_cast<double>(o.second) - m) * inv_t);
        }
        return std::max(z, 0.0);
    };
    double z1 = need_z1 ? full_z(1.0f) : 0.0;
    double zt = need_zt ? full_z(1.0f / P.temperature) : 0.0;
    if (n_lp > 0) {
        const double logz = std::log(z1) + m;
        for (size_t i = 0; i < std::min(n_lp, c.size()); ++i)
            res.top.push_back({c[i].id, static_cast<float>(static_cast<double>(c[i].logit) - logz)});
    }
    auto head_mass = [&](const std::vector<Cand>& cs, float inv_t) {
        double s = 0;
        for (const auto& x : cs) s += std::exp((static_cast<double>(x.logit) - m) * inv_t);
        return s;
    };
    double tail = 0.0;
    if (P.mirostat == 2) tail = need_zt ? std::max(0.0, zt - head_mass(c, 1.0f / P.temperature)) : 0.0;
    else if (P.top_k <= 0) tail = std::max(0.0, z1 - head_mass(c, 1.0f));
    if (tail < 1e-12 * std::max(1.0, z1)) tail = 0.0;  // below double rounding of the head sum

    int32_t chosen = -1;
    bool need_full = false;
    std::vector<Cand> work = c;
    const float mu_before = mu_;
    // A stage needed tokens beyond the candidate set: grow it 8x (exactly, same total order) until it suffices.
    while (!run_chain(work, tail, m, chosen, need_full) && need_full) {
        res.full_vocab_fallback = true;
        mu_ = mu_before;
        k = std::min(n_vocab, k * 8);
        collect(logits, n_vocab, k, c, raw_max);
        const double z = P.mirostat == 2 ? zt : z1;
        tail = k >= n_vocab ? 0.0 : std::max(0.0, z - head_mass(c, P.mirostat == 2 ? 1.0f / P.temperature : 1.0f));
        if (tail < 1e-12 * std::max(1.0, z)) tail = 0.0;
        work = c;
    }
    res.id = chosen;
    res.n_candidates = static_cast<uint32_t>(work.size());
    if (n_lp > 0 || P.mirostat == 2) {
        float l = chosen >= 0 ? logits[chosen] : 0.0f;
        for (const auto& o : overrides_) if (o.first == chosen) l = o.second;
        if (z1 == 0.0) z1 = full_z(1.0f);
        res.logprob = static_cast<float>(static_cast<double>(l) - m - std::log(z1));
    }
    return res;
}

} // namespace vegajet::sampling
