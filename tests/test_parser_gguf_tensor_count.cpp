// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// P-M2: the tensor count from the header must not drive an allocation directly; it has to fit the
// remaining bytes of tensor-info metadata before the table is reserved.
#include "core/gguf_reader.hpp"
#include "tests/test_common.hpp"

using namespace vegajet;

int main() {
    {   // 16.7M tensor infos announced, 0 bytes of metadata behind them (~1.2 GiB reserve before).
        vjtest::RawGguf g(1ull << 24, 0);
        GgufFile f;
        const std::string path = vjtest::write_temp(g.build(0), "gguf_tensor_count");
        VJ_CHECK(!f.open(path));
        std::printf("  tensor count header -> %s\n", f.error().c_str());
        VJ_CHECK(f.error().find("implausible tensor count") != std::string::npos);
    }
    {   // A modest count that the file cannot back up is refused too.
        vjtest::RawGguf g(4096, 0);
        GgufFile f;
        VJ_CHECK(!f.open(vjtest::write_temp(g.build(0), "gguf_tensor_count_small")));
        VJ_CHECK(f.error().find("implausible tensor count") != std::string::npos);
    }

    // Files with a sane tensor table still load.
    vjtest::GgufWriter w;
    w.tensor("a", {4}, 0 /*F32*/, std::vector<uint8_t>(16, 0x00));
    GgufFile ok;
    VJ_CHECK(ok.open(vjtest::write_temp(w.build(), "gguf_tensor_count_ok")));
    VJ_CHECK(ok.tensors().size() == 1);

    std::printf("[PASS] test_parser_gguf_tensor_count: tensor table reserve bounded by the file size\n");
    return 0;
}
