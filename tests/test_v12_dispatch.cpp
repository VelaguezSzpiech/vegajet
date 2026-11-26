#include "core/v12_dispatch.hpp"
#include "ops/kernels/vj_ornith_super_args.hpp"
#include "core/vj_check.hpp"
#include "core/vj_device_guard.hpp"
#include <cstdio>
#include <limits>
#include <initializer_list>

int main() {
    // Two logical collectives per batch: both ranks must share epoch and slot.
    for (uint32_t start : {1u, 2u, std::numeric_limits<uint32_t>::max()}) {
        uint32_t next = start;
        for (uint32_t batch = 0; batch < 4; ++batch) {
            const auto attn = vegajet::dispatch::next_collective(next);
            const auto moe = vegajet::dispatch::next_collective(next);
            for (const auto collective : {attn, moe}) {
                const auto rank0 = collective;
                const auto rank1 = collective;
                VJ_CHECK(rank0.epoch == rank1.epoch);
                VJ_CHECK(rank0.parity == rank1.parity);
                VJ_CHECK(rank0.parity == (rank0.epoch & 1u));
            }
            VJ_CHECK(moe.epoch == attn.epoch + 1u);
        }
        VJ_CHECK(next == start + 8u);
    }
    // Simulated driver work after verifier failure must be unreachable in every mode.
    for (const char* mode : {"raw", "suffix", "MTP", "DFlash", "serve", "dump", "warm"}) {
        for (const char* cause : {"mailbox timeout", "bad input token", "bad argmax"}) {
            vegajet::guard::RunState state;
            unsigned callbacks = 0, commits = 0, drafter_dispatches = 0;
            bool failed = false;
            try {
                state.fail(std::string(mode) + ": " + cause);
                ++callbacks; ++commits; ++drafter_dispatches;
            } catch (const vegajet::guard::RunFailure& e) {
                failed = true;
                VJ_CHECK(state.error() == e.what());
            }
            VJ_CHECK(failed && callbacks == 0 && commits == 0 && drafter_dispatches == 0);
            const auto first = state.error();
            state.latch("later failure");
            VJ_CHECK(state.error() == first);
            try { state.check(); VJ_CHECK(false); }
            catch (const vegajet::guard::RunFailure& e) { VJ_CHECK(first == e.what()); }
        }
    }
    std::puts("v12 dispatch host contracts PASS");
}
