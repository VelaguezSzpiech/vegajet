// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// P-M1: GGUF tensor geometry must be validated (no wrapping element/byte counts, no zero-sized
// axes, no block-misaligned rows) so an impossible tensor can never reach the loaders.
#include "core/gguf_reader.hpp"
#include "tests/test_common.hpp"

using namespace vegajet;

namespace {

std::string reject(const vjtest::RawGguf& g, const char* tag, const char* needle) {
    GgufFile f;
    const std::string path = vjtest::write_temp(g.build(64), tag);
    VJ_CHECK(!f.open(path));
    std::printf("  %s -> %s\n", tag, f.error().c_str());
    VJ_CHECK(f.error().find(needle) != std::string::npos);
    return f.error();
}

} // namespace

int main() {
    {   // 2^32 x 2^32 elements: the 64-bit element count wraps to 0.
        vjtest::RawGguf g(1, 0);
        g.tensor("a", {4294967296ull, 4294967296ull}, 0 /*F32*/, 0);
        reject(g, "gguf_ne_overflow", "element count overflows");
    }
    {   // Zero-sized axis: 0 bytes of payload would otherwise validate as an empty tensor.
        vjtest::RawGguf g(1, 0);
        g.tensor("z", {8, 0}, 0 /*F32*/, 0);
        reject(g, "gguf_ne_zero", "zero-sized dimension");
    }
    {   // Q4_0 needs whole 32-element blocks; 33 elements used to be accepted with n_bytes()==0.
        vjtest::RawGguf g(1, 0);
        g.tensor("q", {33}, 2 /*Q4_0*/, 0);
        reject(g, "gguf_ne_unaligned", "no byte size");
    }
    {   // 2^31 x 2^31 elements of I64: element count fits, byte count does not.
        vjtest::RawGguf g(1, 0);
        g.tensor("i", {2147483648ull, 2147483648ull}, 27 /*I64*/, 0);
        reject(g, "gguf_nb_overflow", "no byte size");
    }

    // A normal file still opens and reports the same geometry.
    vjtest::GgufWriter w;
    const std::vector<uint8_t> q8(34 * 2, 0x11);
    w.tensor("a.weight", {64}, 8 /*Q8_0*/, q8);
    GgufFile ok;
    VJ_CHECK(ok.open(vjtest::write_temp(w.build(), "gguf_ne_ok")));
    const GgufTensorInfo* a = ok.tensor("a.weight");
    VJ_CHECK(a && a->n_elems() == 64 && a->n_bytes() == 68);

    std::printf("[PASS] test_parser_gguf_ne: element/byte overflows, zero axes and block-misaligned rows rejected\n");
    return 0;
}
