// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// P-B2: `nextn_predict_layers > block_count` underflowed n_trunk (and 64-bit counts were truncated
// to uint32), so recurrent[] was written far past its end. Both must be refused by parse_hparams().
#include "models/ornith/ornith_model.hpp"
#include "tests/ornith_min_fixture.hpp"

using namespace vegajet;

namespace {

LoadReport load_with(const vjtest::OrnithHparams& p, const std::string& tag) {
    GgufFile f;
    VJ_CHECK(f.open(vjtest::write_min_ornith(p, tag)));
    ornith::OrnithModel m;
    VJ_CHECK(m.can_load(f));
    LoadOptions opt;
    return m.load(f, opt);
}

} // namespace

int main() {
    {   // block_count=1, nextn=2: n_trunk used to wrap to 0xFFFFFFFF.
        vjtest::OrnithHparams p;
        p.block_count = 1;
        p.nextn = 2;
        const LoadReport rep = load_with(p, "ornith_nextn");
        std::printf("  nextn 2 / block_count 1 -> %s\n", rep.error.c_str());
        VJ_CHECK(!rep.ok);
        VJ_CHECK(rep.error.find("nextn_predict_layers") != std::string::npos);
    }
    {   // Equal counts leave zero trunk layers.
        vjtest::OrnithHparams p;
        p.block_count = 4;
        p.nextn = 4;
        const LoadReport rep = load_with(p, "ornith_nextn_all");
        VJ_CHECK(!rep.ok && rep.error.find("trunk") != std::string::npos);
    }
    {   // 2^32+2 must be rejected, not truncated to 2.
        vjtest::OrnithHparams p;
        p.block_count = 3;
        p.nextn = 4294967298ull;
        const LoadReport rep = load_with(p, "ornith_nextn64");
        VJ_CHECK(!rep.ok && rep.error.find("out of range") != std::string::npos);
    }
    {   // block_count=0 and absurd counts are refused.
        vjtest::OrnithHparams p;
        p.block_count = 0;
        p.nextn_present = false;
        const LoadReport rep = load_with(p, "ornith_block0");
        VJ_CHECK(!rep.ok && rep.error.find("block_count") != std::string::npos);
    }
    {   // A consistent count still gets through hparams and fails later on the missing block tensors.
        vjtest::OrnithHparams p;
        p.block_count = 1;
        p.nextn = 0;
        const LoadReport rep = load_with(p, "ornith_nextn_ok");
        std::printf("  block_count 1 / nextn 0 -> %s\n", rep.error.c_str());
        VJ_CHECK(!rep.ok && rep.error.find("missing tensor") != std::string::npos);
    }

    std::printf("[PASS] test_parser_ornith_nextn: nextn_predict_layers underflow and uint32 truncation refused\n");
    return 0;
}
