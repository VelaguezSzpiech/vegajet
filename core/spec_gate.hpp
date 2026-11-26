// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
// Acceptance-aware speculative/raw gate. Model- and drafter-agnostic: it only sees per-cycle timings and the
// number of tokens each cycle emitted. Speculation pays only while (tokens per speculative cycle) / (cycle time)
// beats (1 token) / (raw step time). When it does not, the gate runs raw single steps for a back-off window and
// then probes speculation again. Raw steps are exact target steps, so the output stays lossless.
#include <cstdint>

namespace vegajet::spec_gate {

struct Config {
    double   ema_alpha    = 0.25;  // weight of the newest sample
    double   margin       = 1.02;  // speculation must beat raw by this factor to stay on
    uint32_t warmup       = 2;     // speculative cycles before the first decision
    uint32_t backoff_min  = 32;    // raw steps after the first switch-off
    uint32_t backoff_max  = 1024;  // back-off doubles up to this on repeated failures
};

class Gate {
public:
    explicit Gate(Config c = {}) : c_(c), backoff_(c.backoff_min) {}

    // True: run a speculative cycle now. False: run one raw step.
    bool want_spec() const { return raw_left_ == 0u; }

    // One speculative cycle emitted `emitted` tokens (accepted + correction) in `ms`.
    void on_spec(uint32_t emitted, double ms) {
        if (ms <= 0.0) return;
        const double tpms = static_cast<double>(emitted) / ms;
        spec_tpms_ = n_spec_ ? (1.0 - c_.ema_alpha) * spec_tpms_ + c_.ema_alpha * tpms : tpms;
        ++n_spec_;
        if (n_spec_ < c_.warmup || raw_ms_ <= 0.0) return;
        if (spec_tpms_ * raw_ms_ < c_.margin) {           // tokens per raw-step-time below margin: speculation loses
            raw_left_ = backoff_;
            backoff_ = backoff_ * 2u > c_.backoff_max ? c_.backoff_max : backoff_ * 2u;
            n_spec_ = 0u;                                  // re-warm after the back-off
            ++switches_;
        } else {
            backoff_ = c_.backoff_min;
        }
    }

    // One raw step took `ms`.
    void on_raw(double ms) {
        if (ms > 0.0) raw_ms_ = raw_ms_ > 0.0 ? (1.0 - c_.ema_alpha) * raw_ms_ + c_.ema_alpha * ms : ms;
        if (raw_left_) --raw_left_;
    }

    // Seed the raw step cost (e.g. from a measured raw tg128 run) so the first decision needs no raw sample.
    void seed_raw_ms(double ms) { if (ms > 0.0 && raw_ms_ <= 0.0) raw_ms_ = ms; }

    uint32_t switches() const { return switches_; }
    double   raw_ms() const { return raw_ms_; }
    double   spec_tokens_per_ms() const { return spec_tpms_; }

private:
    Config   c_;
    double   spec_tpms_ = 0.0, raw_ms_ = 0.0;
    uint32_t n_spec_ = 0u, raw_left_ = 0u, backoff_, switches_ = 0u;
};

} // namespace vegajet::spec_gate
