// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/quant_unpackers.hpp"
#include "tests/test_common.hpp"
#include <limits>
using namespace vegajet;
using namespace vegajet::quant;
int main() {
    const char* names[] = {"F16", "BF16", "FP8_E4M3", "Q8_0", "Q6_K", "Q5_K", "Q4_0", "Q4_1",
        "Q4_K", "IQ4_XS", "IQ4_NL", "IQ3_XXS", "IQ3_S", "IQ2_XS", "IQ2_S", "Q3_K", "Q2_K"};
    for (const auto* name : names) {
        DiskQuantType type = DiskQuantType::COUNT;
        const auto* info = tensor_unpacker(name, type);
        VJ_CHECK(info && info->decode);
        // Zero scales are exact golden values, including codebook formats.
        std::vector<uint8_t> source(info->block_bytes + 1, 0);
        std::vector<float> values(info->block_elems, 123.0f);
        VJ_CHECK(unpack_quantized(type, source.data() + 1, info->block_bytes,
                                 values.data(), values.size(), values.size()));
        for (float value : values) VJ_CHECK(value == 0.0f);
        VJ_CHECK(!unpack_quantized(type, source.data(), info->block_bytes - 1,
                                  values.data(), values.size(), values.size()));
        VJ_CHECK(!unpack_quantized(type, source.data(), info->block_bytes,
                                  values.data(), values.size() - 1, values.size()));
        VJ_CHECK(!unpack_quantized(type, nullptr, info->block_bytes,
                                  values.data(), values.size(), values.size()));
        if (info->block_elems > 1)
            VJ_CHECK(!unpack_quantized(type, source.data(), source.size(), values.data(), values.size(), 1));
        VJ_CHECK(unpack_quantized(type, nullptr, 0, nullptr, 0, 0));
        if (info->block_elems >= 32) {
            const auto packed = repack_to_vj_iq(type, source.data() + 1, 1, info->block_elems);
            const auto restored = unpack_vj_iq_to_disk(packed);
            VJ_CHECK(restored.size() == info->block_bytes);
            VJ_CHECK(std::memcmp(restored.data(), source.data() + 1, restored.size()) == 0);
        }
    }
    DiskQuantType untouched = DiskQuantType::F32;
    for (const auto* preset : {"Q5_K_M", "Q4_K_M", "Q4_K_S", "IQ3_XS"}) {
        VJ_CHECK(is_quantization_preset(preset));
        VJ_CHECK(!tensor_unpacker(preset, untouched));
        VJ_CHECK(untouched == DiskQuantType::F32);
    }
    VJ_CHECK(!tensor_unpacker("unknown", untouched));
    float storage[32] = {};
    VJ_CHECK(!unpack_quantized(DiskQuantType::F32, storage, sizeof(storage), storage, 32, 32));
    VJ_CHECK(!unpack_quantized(DiskQuantType::COUNT, nullptr, 0, nullptr, 0, 0));
    VJ_CHECK(!unpack_quantized(DiskQuantType::F32, storage, sizeof(storage), storage, 32,
                              std::numeric_limits<size_t>::max()));
    // Q8_0 exact plane alignment: two blocks with different scales and signed codes.
    std::vector<uint8_t> source(68);
    for (size_t block = 0; block != 2; ++block) {
        const uint16_t scale = f32_to_fp16(float(block + 1));
        std::memcpy(source.data() + block * 34, &scale, 2);
        for (size_t i = 0; i != 32; ++i) source[block * 34 + 2 + i] = uint8_t(int(i) - 16);
    }
    const auto packed = repack_to_vj_q8(DiskQuantType::Q8_0, source.data(), 1, 64);
    VJ_CHECK(packed.qs.size() == 64 && packed.ds.size() == 2);
    for (size_t i = 0; i != 64; ++i) VJ_CHECK(packed.qs[i] == int(i % 32) - 16);
    VJ_CHECK(packed.ds[0] == f32_to_fp16(1) && packed.ds[1] == f32_to_fp16(2));
    std::puts("quant_unpackers: PASS (reference golden/cosine coverage in test_quant_repacker)");
}
