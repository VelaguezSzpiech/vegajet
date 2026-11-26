// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// P-B1: file-controlled tensor offsets must not wrap the `data_offset + offset + n_bytes` sum,
// and tensor_data() must re-validate the range instead of handing out an out-of-mapping pointer.
#include "core/gguf_reader.hpp"
#include "tests/test_common.hpp"

using namespace vegajet;

int main() {
    // data_offset is 64; the tensor claims 2 F32 elements at 2^64-72, so the old sum wrapped to 0
    // and passed the "past end of file" check while the pointer landed before the mapping.
    vjtest::RawGguf bad_file(1, 0);
    bad_file.tensor("a", {2}, 0 /*F32*/, 0xFFFFFFFFFFFFFFB8ull);
    const std::string bad = vjtest::write_temp(bad_file.build(16), "gguf_off_overflow");
    GgufFile f;
    VJ_CHECK(!f.open(bad));
    VJ_CHECK(f.error().find("past end") != std::string::npos);

    // A well-formed file still opens and hands out an in-range pointer.
    vjtest::GgufWriter w;
    const std::vector<uint8_t> f32(8, 0x3f);
    w.tensor("a", {2}, 0, f32);
    const std::string good = vjtest::write_temp(w.build(), "gguf_off_ok");
    GgufFile ok;
    VJ_CHECK(ok.open(good));
    const GgufTensorInfo* t = ok.tensor("a");
    VJ_CHECK(t != nullptr && ok.tensor_data(*t) != nullptr);

    // Hand-built infos (never vetted by open) are re-checked by tensor_data().
    GgufTensorInfo bogus = *t;
    bogus.offset = 0xFFFFFFFFFFFFFFF8ull;
    VJ_CHECK(ok.tensor_data(bogus) == nullptr);

    std::printf("[PASS] test_parser_gguf_overflow: wrapping tensor offset rejected, tensor_data() re-validates\n");
    return 0;
}
