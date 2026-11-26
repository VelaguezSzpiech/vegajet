// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "models/ornith/expert_tensor_checks.hpp"
#include "core/vj_check.hpp"
#include <cstdio>
#include <string>

using namespace vegajet;
int main() {
    for (bool down : {false, true}) {
        GgufTensorInfo good;
        good.name = down ? "blk.0.ffn_down_exps.weight" : "blk.0.ffn_gate_exps.weight";
        good.n_dims = 3;
        good.ne[0] = down ? 1024 : 3072;
        good.ne[1] = down ? 3072 : 1024;
        good.ne[2] = 256;
        good.type = static_cast<uint32_t>(down ? GgmlType::IQ3_XXS : GgmlType::IQ2_S);
        std::string error;
        auto valid = [&](const GgufTensorInfo* t, bool native = true, uint32_t D = 3072,
                         uint32_t ff = 1024, uint32_t experts = 256) {
            error.clear();
            const bool ok = ornith::validate_expert_tensor(t, good.name, D, ff, experts, down, native, &error);
            if (!ok) VJ_CHECK(error.find(good.name) != std::string::npos);
            return ok;
        };
        VJ_CHECK(valid(&good));
        VJ_CHECK(valid(&good, false));  // MTP decoder path.
        VJ_CHECK(!valid(nullptr));
        VJ_CHECK(!valid(&good, true, 3072, 512));
        VJ_CHECK(!valid(&good, true, 4096));
        VJ_CHECK(!valid(&good, true, 3072, 1024, 128));
        const auto rb = down ? 392u : 984u;
        VJ_CHECK(good.n_bytes() == uint64_t(rb) * good.ne[1] * 256u);

        auto bad = good;
        bad.type = static_cast<uint32_t>(GgmlType::F32);
        VJ_CHECK(!valid(&bad));
        VJ_CHECK(error.find("requires IQ") != std::string::npos);
        VJ_CHECK(valid(&bad, false));  // Generic MTP decoder can handle F32.
        bad.type = UINT32_MAX;
        VJ_CHECK(!valid(&bad, false));
        for (unsigned axis = 0; axis < 4; ++axis) {
            bad = good;
            bad.ne[axis] *= 2;
            VJ_CHECK(!valid(&bad));
            bad.ne[axis] = UINT64_MAX;
            VJ_CHECK(!valid(&bad));
        }
        bad = good;
        bad.ne[1] *= 2;
        bad.ne[2] /= 2;  // Same n_elems/bytes does not establish the expert row layout.
        VJ_CHECK(bad.n_bytes() == good.n_bytes());
        VJ_CHECK(!valid(&bad));
        for (uint32_t dims : {0u, 1u, 2u, 4u}) {
            bad = good;
            bad.n_dims = dims;
            VJ_CHECK(!valid(&bad));
        }
        bad = good;
        bad.n_dims = 2;
        bad.ne[1] *= 256;
        bad.ne[2] = 1;
        VJ_CHECK(valid(&bad));
        // The audit's tiny declared tensor with full-size metadata must reject,
        // without a mmap, decoder call or payload read.
        bad = good;
        bad.n_dims = 2;
        bad.ne[0] = 256;
        bad.ne[1] = 1;
        bad.ne[2] = 1;
        VJ_CHECK(!valid(&bad));
    }
    std::puts("ornith expert type/shape/row layout checks PASS");
}
