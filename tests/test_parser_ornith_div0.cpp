// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// P-B3: metadata-driven divisors (attention.head_count, full_attention_interval) were used as-is,
// so a crafted GGUF raised SIGFPE instead of returning a LoadReport error.
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

void expect(const vjtest::OrnithHparams& p, const char* tag, const char* needle) {
    const LoadReport rep = load_with(p, tag);
    std::printf("  %s -> %s\n", tag, rep.error.c_str());
    VJ_CHECK(!rep.ok);
    VJ_CHECK(rep.error.find(needle) != std::string::npos);
}

} // namespace

int main() {
    {   // head_count == 0: n_embd / 0 used to trap.
        vjtest::OrnithHparams p;
        p.head_count = 0;
        expect(p, "ornith_div0", "attention.head_count");
    }
    {   // full_attention_interval == 0: (i + 1) % 0 used to trap.
        vjtest::OrnithHparams p;
        p.full_attn_interval = 0;
        expect(p, "ornith_iv0", "full_attention_interval");
    }
    {   // head_count_kv == 0 and kv > head are refused before any KV sharding math.
        vjtest::OrnithHparams p;
        p.head_count_kv = 0;
        expect(p, "ornith_kv0", "attention.head_count_kv");
        p.head_count_kv = 8;
        expect(p, "ornith_kv_gt_head", "attention.head_count_kv");
    }
    {   // A head count that does not divide the embedding width would give head_dim 0.
        vjtest::OrnithHparams p;
        p.embedding_length = 250;
        p.head_count = 4;
        expect(p, "ornith_head_mismatch", "not a multiple");
    }
    {   // Consistent numbers still parse; the error is the expected missing layer tensor.
        vjtest::OrnithHparams p;
        const LoadReport rep = load_with(p, "ornith_div_ok");
        VJ_CHECK(!rep.ok && rep.error.find("missing tensor") != std::string::npos);
    }

    std::printf("[PASS] test_parser_ornith_div0: zero/ill-shaped hparam divisors refused with an error\n");
    return 0;
}
