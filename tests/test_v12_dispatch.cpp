#include "core/v12_dispatch.hpp"
#include "ops/kernels/vj_ornith_super_args.hpp"
#include <cassert>
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
                assert(rank0.epoch == rank1.epoch);
                assert(rank0.parity == rank1.parity);
                assert(rank0.parity == (rank0.epoch & 1u));
            }
            assert(moe.epoch == attn.epoch + 1u);
        }
        assert(next == start + 8u);
    }
    std::puts("v12 dispatch host contracts PASS");
}
