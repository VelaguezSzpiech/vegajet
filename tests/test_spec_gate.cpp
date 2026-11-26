// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/spec_gate.hpp"

#include <cstdio>

using vegajet::spec_gate::Config;
using vegajet::spec_gate::Gate;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    // Raw step 6.4 ms (156 tok/s). Speculation emitting 1.6 tokens in 16 ms = 0.1 tok/ms < 1/6.4: must switch off.
    {
        Gate g; g.seed_raw_ms(6.4);
        for (int i = 0; i < 2; ++i) { CHECK(g.want_spec()); g.on_spec(2u, 20.0); }
        CHECK(!g.want_spec());
        CHECK(g.switches() == 1u);
        int raw = 0;
        while (!g.want_spec()) { g.on_raw(6.4); ++raw; }
        CHECK(raw == 32);                      // backoff_min
        for (int i = 0; i < 2; ++i) g.on_spec(2u, 20.0);
        raw = 0;
        while (!g.want_spec()) { g.on_raw(6.4); ++raw; }
        CHECK(raw == 64);                      // doubled on repeated failure
        CHECK(g.switches() == 2u);
    }
    // Code-like speculation: 8 tokens in 24 ms = 0.33 tok/ms > 1/6.4 * 1.02: stays on.
    {
        Gate g; g.seed_raw_ms(6.4);
        for (int i = 0; i < 50; ++i) { CHECK(g.want_spec()); g.on_spec(8u, 24.0); }
        CHECK(g.switches() == 0u);
    }
    // No raw cost known yet: never switches (cannot compare).
    {
        Gate g;
        for (int i = 0; i < 20; ++i) g.on_spec(1u, 50.0);
        CHECK(g.want_spec());
    }
    // Back-off caps at backoff_max.
    {
        Config c; c.backoff_min = 8; c.backoff_max = 32; c.warmup = 1;
        Gate g(c); g.seed_raw_ms(6.4);
        int last = 0;
        for (int round = 0; round < 6; ++round) {
            g.on_spec(1u, 30.0);
            int raw = 0;
            while (!g.want_spec()) { g.on_raw(6.4); ++raw; }
            last = raw;
        }
        CHECK(last == 32);
    }
    // Recovery: after a successful spec phase the back-off resets to the minimum.
    {
        Config c; c.warmup = 1;
        Gate g(c); g.seed_raw_ms(6.4);
        g.on_spec(1u, 30.0);
        while (!g.want_spec()) g.on_raw(6.4);
        g.on_spec(1u, 30.0);                   // second failure: backoff now 64
        while (!g.want_spec()) g.on_raw(6.4);
        g.on_spec(10u, 20.0);                  // fresh EMA after back-off: 0.5 tok/ms, passes -> back-off resets to 32
        CHECK(g.want_spec());
        while (g.want_spec()) g.on_spec(1u, 300.0);  // EMA decays below margin
        int raw = 0;
        while (!g.want_spec()) { g.on_raw(6.4); ++raw; }
        CHECK(raw == 32);
    }
    if (fails == 0) std::puts("test_spec_gate: PASS");
    return fails ? 1 : 0;
}
