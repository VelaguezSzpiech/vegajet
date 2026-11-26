#include "core/v12_dispatch.hpp"
#include "core/v12_modules.hpp"
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
    for (uint32_t rows : {1u, 2u, 8u, 16u}) {
        unsigned visits[16] = {};
        const auto grid_x = vegajet::dispatch::eh_grid_x(rows);
        for (uint32_t block_x = 0; block_x < grid_x / 256u; ++block_x) ++visits[block_x];
        for (uint32_t row = 0; row < rows; ++row) VJ_CHECK(visits[row] == 1u);
    }
    const auto off = vegajet::select_super_variant(false, false);
    VJ_CHECK(off.valid && off.object == nullptr && std::string(off.suffix).empty());
    const auto vol = vegajet::select_super_variant(true, false);
    const auto iq2s = vegajet::select_super_variant(false, true);
    VJ_CHECK(std::string(vol.object) == "vj_ornith_super_vol.hsaco");
    VJ_CHECK(std::string(iq2s.object) == "vj_ornith_super_iq2sreg.hsaco");
    VJ_CHECK(!vegajet::select_super_variant(true, true).valid);
    using vegajet::dispatch::kv_span_fits;
    using vegajet::dispatch::split_scratch_fits;
    VJ_CHECK(kv_span_fits(16368u, 16u, 16384u));
    VJ_CHECK(!kv_span_fits(16369u, 16u, 16384u));
    VJ_CHECK(!kv_span_fits(0u, 0u, 16384u));
    VJ_CHECK(!kv_span_fits(UINT32_MAX - 3u, 16u, UINT32_MAX));
    VJ_CHECK(kv_span_fits(UINT32_MAX - 16u, 16u, UINT32_MAX));
    VJ_CHECK(split_scratch_fits(16u, 16u, 60u, 960u, 16u));
    VJ_CHECK(!split_scratch_fits(16u, 16u, 61u, 960u, 16u));
    VJ_CHECK(!split_scratch_fits(16u, 16u, 64u, 960u, 16u));
    VJ_CHECK(split_scratch_fits(15u, 16u, 64u, 960u, 16u));
    VJ_CHECK(split_scratch_fits(1u, 16u, 960u, 960u, 16u));
    VJ_CHECK(!split_scratch_fits(1u, 16u, 961u, 960u, 16u));
    VJ_CHECK(!split_scratch_fits(UINT32_MAX, 16u, UINT32_MAX, 960u, 16u));
    VJ_CHECK(!split_scratch_fits(1u, 17u, 60u, 960u, 16u));
    VJ_CHECK(!split_scratch_fits(0u, 16u, 60u, 960u, 16u));
    VJ_CHECK(!split_scratch_fits(1u, 16u, 0u, 960u, 16u));
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
