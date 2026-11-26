// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <limits>
#include <string>
#include <vector>

namespace vegajet::telemetry {

inline double unavailable() { return std::numeric_limits<double>::quiet_NaN(); }
inline void max_observed(double& dst, double value) {
    if (std::isfinite(value) && (!std::isfinite(dst) || value > dst)) dst = value;
}

// Power is held at the PREVIOUS observation between timestamps. There is no
// extrapolation before the first or after the last observation. Clip intervals
// exactly at the cutoff; sample count / requested cadence is not elapsed time.
class PowerWindow {
public:
    bool add(double seconds, double watts) {
        if (!std::isfinite(seconds) || !std::isfinite(watts) || watts < 0 ||
            (!points_.empty() && seconds <= points_.back().seconds)) return false;
        const double previous = points_.empty() ? seconds : points_.back().seconds;
        if (points_.empty()) first_ = seconds;
        else joules_ += (seconds - previous) * points_.back().watts;
        points_.push_back({seconds, watts});
        ++samples_;
        max_observed(peak_, watts);
        update_worst(10.0, previous, worst10_);
        update_worst(1.0, previous, worst1_);
        // Keep the observation immediately at/before the oldest cutoff.
        while (points_.size() > 2 && points_[1].seconds <= seconds - 10.0) points_.pop_front();
        return true;
    }
    double observed_s() const { return points_.empty() ? 0.0 : points_.back().seconds - first_; }
    double mean_w() const { return observed_s() > 0 ? joules_ / observed_s() : unavailable(); }
    double avg_w() const { return std::isfinite(worst10_) ? worst10_ : mean_w(); }
    double avg1s_w() const { return std::isfinite(worst1_) ? worst1_ : mean_w(); }
    double window_s() const { return std::min(10.0, observed_s()); }
    double window1s_s() const { return std::min(1.0, observed_s()); }
    const char* window_kind() const {
        return observed_s() <= 0 ? "missing" : (observed_s() < 10.0 ? "short_observed" : "full_10s");
    }
    double peak_w() const { return peak_; }
    uint64_t samples() const { return samples_; }

private:
    struct Point { double seconds, watts; };
    double average(double end, double width) const {
        double energy = 0.0;
        for (size_t i = 1; i < points_.size(); ++i) {
            const double dt = std::min(end, points_[i].seconds) - std::max(end - width, points_[i - 1].seconds);
            if (dt > 0) energy += dt * points_[i - 1].watts;
        }
        return energy / width;
    }
    void update_worst(double width, double previous, double& worst) {
        const double now = points_.back().seconds;
        if (now - first_ < width) return;
        // A held-sample rolling integral changes slope only at observations or
        // observation+width. Include the latter even between irregular reads.
        // ponytail: scan the bounded 10 s history; no extra indexing at 4 Hz.
        for (const auto& p : points_) {
            const double end = p.seconds + width;
            if (end > previous && end <= now) max_observed(worst, average(end, width));
        }
        max_observed(worst, average(now, width));
    }
    std::deque<Point> points_;
    double first_ = 0.0, joules_ = 0.0;
    double peak_ = unavailable(), worst10_ = unavailable(), worst1_ = unavailable();
    uint64_t samples_ = 0;
};

inline double average_limit(size_t cards) { return cards >= 3 ? 155.0 : 205.0; }
inline double cap_limit(size_t cards) { return cards >= 3 ? 150.0 : 200.0; }

enum class Verdict { pass, incomplete, fail };
inline const char* name(Verdict v) { return v == Verdict::fail ? "FAIL" : (v == Verdict::incomplete ? "INCOMPLETE" : "PASS"); }
inline Verdict worst(Verdict a, Verdict b) { return std::max(a, b); }

struct Card {
    std::string id;  // DRM cardN, not the PCI-order gpuN used by bench JSON snapshots.
    PowerWindow power;
    double cap_w = unavailable(), cap_min_w = unavailable(), cap_max_w = unavailable();
    double tmem_max_c = unavailable();
    uint64_t missing = 0;

    void observe(double seconds, double watts, double tmem_c, double cap) {
        bool complete = power.add(seconds, watts);
        if (std::isfinite(tmem_c) && tmem_c >= 0) max_observed(tmem_max_c, tmem_c);
        else complete = false;
        if (std::isfinite(cap) && cap > 0) {
            cap_w = cap;
            if (!std::isfinite(cap_min_w) || cap < cap_min_w) cap_min_w = cap;
            max_observed(cap_max_w, cap);
        } else complete = false;
        if (!complete) ++missing;
    }
    Verdict verdict(size_t cards) const {
        // Known violations remain FAIL even if some other telemetry is missing.
        if (power.avg_w() > average_limit(cards) || power.peak_w() >= 270.0 || tmem_max_c >= 65.0 ||
            cap_max_w > cap_limit(cards) || (cap_min_w != cap_max_w && std::isfinite(cap_min_w))) return Verdict::fail;
        if (missing || power.observed_s() <= 0 || !std::isfinite(tmem_max_c) || !std::isfinite(cap_w))
            return Verdict::incomplete;
        return Verdict::pass;
    }
};

struct Summary {
    double avg_w = unavailable(), mean_w = unavailable(), avg1s_w = unavailable();
    double peak_w = unavailable(), tmem_max_c = unavailable(), window_s = 0.0;
    const char* window_kind = "missing";
    uint64_t samples = 0;
    Verdict verdict = Verdict::incomplete;
};

inline Summary summarize(const std::vector<Card>& cards, double extra_tmem_c = unavailable()) {
    Summary s;
    if (!cards.empty()) s.verdict = Verdict::pass;
    for (const auto& c : cards) {
        s.verdict = worst(s.verdict, c.verdict(cards.size()));
        if (std::isfinite(c.power.avg_w()) && (!std::isfinite(s.avg_w) || c.power.avg_w() > s.avg_w)) {
            s.avg_w = c.power.avg_w();
            s.window_s = c.power.window_s();
            s.window_kind = c.power.window_kind();
        }
        max_observed(s.mean_w, c.power.mean_w());
        max_observed(s.avg1s_w, c.power.avg1s_w());
        max_observed(s.peak_w, c.power.peak_w());
        max_observed(s.tmem_max_c, c.tmem_max_c);
        s.samples += c.power.samples();
    }
    max_observed(s.tmem_max_c, extra_tmem_c);
    if (s.tmem_max_c >= 65.0) s.verdict = Verdict::fail;
    return s;
}

// Shared by the engine and deterministic log -> JSON tests. nan means unavailable
// in text; scorecard readers turn it into JSON null, never a successful zero.
inline std::string report(const std::vector<Card>& cards, double extra_tmem_c = unavailable()) {
    std::string text;
    char line[1024];
    for (const auto& c : cards) {
        std::snprintf(line, sizeof(line),
            "[TELEM card=%s] cap=%.6f W cap_min=%.6f W cap_max=%.6f W mean=%.6f W "
            "rolling avg=%.6f W window=%.6f s window_kind=%s hold=previous "
            "avg1s=%.6f W window1s=%.6f s peak=%.6f W max_TMem=%.6f C "
            "observed=%.6f s samples=%llu missing=%llu -> %s\n",
            c.id.c_str(), c.cap_w, c.cap_min_w, c.cap_max_w, c.power.mean_w(),
            c.power.avg_w(), c.power.window_s(), c.power.window_kind(), c.power.avg1s_w(), c.power.window1s_s(),
            c.power.peak_w(), c.tmem_max_c, c.power.observed_s(),
            static_cast<unsigned long long>(c.power.samples()), static_cast<unsigned long long>(c.missing), name(c.verdict(cards.size())));
        text += line;
    }
    const Summary s = summarize(cards, extra_tmem_c);
    std::snprintf(line, sizeof(line),
        "[TELEM] power1_input: %s rolling avg=%.6f W (mean=%.6f W, <=%.0f W) | "
        "window=%.6f s window_kind=%s hold=previous | worst 1 s avg=%.6f W (info; short observed if <1 s) | "
        "instantaneous peak=%.6f W (<270 W) | max_TMem=%.6f C (<65 C) | %llu samples -> %s\n",
        s.window_s >= 10.0 ? "worst 10 s" : "short observed", s.avg_w, s.mean_w, average_limit(cards.size()),
        s.window_s, s.window_kind, s.avg1s_w, s.peak_w, s.tmem_max_c,
        static_cast<unsigned long long>(s.samples), name(s.verdict));
    return text + line;
}

}  // namespace vegajet::telemetry
