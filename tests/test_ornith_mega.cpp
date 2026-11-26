// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "ops/ornith_mega_launch.hpp"
#include "tests/test_common.hpp"
#include <array>
#include <limits>

using namespace vegajet;

// Decode bytes at fixed device ABI offsets, independently of C++ field access.
template<class T, class U> T wire(const U& object, size_t offset) {
    T value{};
    std::memcpy(&value, reinterpret_cast<const unsigned char*>(&object) + offset, sizeof(T));
    return value;
}

int main() {
    using ornith_super::Mega48Desc;
    std::array<Mega48Desc, 3> desc{};
    constexpr uint32_t ff[3] = {384, 320, 320};
    constexpr uint32_t nv[3] = {24, 20, 20};
    constexpr uint32_t nq[3] = {12, 10, 10};
    constexpr uint32_t nkv[3] = {1, 2, 1};
    constexpr uint32_t qoff[3] = {0, 12, 22};
    constexpr uint32_t koff[3] = {0, 0, 1};
    uint32_t total_ff = 0, total_q = 0;
    for (uint32_t rank = 0; rank < 3; ++rank) {
        auto& d = desc[rank];
        const uint64_t base = 0x1234567800000000ULL + rank * 0x100000ULL;
        d.gdn_descs = base; d.gqa_descs = base + 0x10000; d.moe_descs = base + 0x20000;
        d.super_bar = base + 0x30000;
        d.ssm_layer_bytes = static_cast<uint64_t>(nv[rank]) * 128u * 128u * 4u;
        d.D = 3072; d.rank = rank; d.n_ranks = 3; d.eps = 1e-6f;
        d.voc_rows = 78000u + rank * 512u; d.voc_off = rank * 91000u; d.voc_stride = 3072;
        for (uint32_t r = 0; r < 4; ++r) {
            d.final_ar_rx[r] = r < 3 && r != rank ? base + r * 0x1000u : 0;
            d.final_ar_flag[r] = r < 3 && r != rank ? base + r * 0x1000u + (95u & 63u) * 64u : 0;
            VJ_CHECK(wire<uint64_t>(d, 104u + r * 8u) == d.final_ar_rx[r]);
            VJ_CHECK(wire<uint64_t>(d, 136u + r * 8u) == d.final_ar_flag[r]);
        }
        for (uint32_t il = 0; il < 48; ++il) if ((il & 3u) != 3u) d.gdn_mask |= 1ULL << il;
        VJ_CHECK(wire<uint64_t>(d, 200u) == d.super_bar);
        VJ_CHECK(wire<uint64_t>(d, 208u) == d.ssm_layer_bytes);
        VJ_CHECK(wire<uint64_t>(d, 216u) == 0x777777777777ULL);
        VJ_CHECK(wire<uint32_t>(d, 224u) == d.voc_rows);
        VJ_CHECK(wire<uint32_t>(d, 228u) == d.voc_off);
        VJ_CHECK(wire<uint32_t>(d, 236u) == 3072);
        VJ_CHECK(wire<uint32_t>(d, 240u) == rank);
        VJ_CHECK(wire<uint32_t>(d, 244u) == 3);
        VJ_CHECK(wire<float>(d, 248u) == 1e-6f);
        VJ_CHECK(wire<uint32_t>(d, 252u) == 0);
        // TP-straddling head ownership must stay descriptor-driven, never nq/nkv rounded locally.
        for (uint32_t h = 0; h < nq[rank]; ++h)
            VJ_CHECK((qoff[rank] + h) / 16u - koff[rank] < nkv[rank]);
        VJ_CHECK((8u * (ff[rank] / 32u)) * 4u + 1536u <= 2048u);
        total_ff += ff[rank]; total_q += nq[rank];
    }
    VJ_CHECK(total_ff == 1024 && total_q == 32);
    VJ_CHECK(reinterpret_cast<const char*>(&desc[2]) - reinterpret_cast<const char*>(&desc[0]) == 512);
    ornith_mega::GdnArgs ga{0x123456789abcdef0ULL, 0xfedcba9876543210ULL,
        44, 33, 3, 89, 16, 0, 16, 1, 100, 101, ornith_mega::kSkipFirstAr};
    VJ_CHECK(wire<uint64_t>(ga, 0) == ga.desc && wire<uint64_t>(ga, 8) == ga.spec_gdn_base);
    VJ_CHECK(wire<uint32_t>(ga, 16) == 44 && wire<uint32_t>(ga, 20) == 33);
    VJ_CHECK(wire<uint32_t>(ga, 24) == 3 && wire<uint32_t>(ga, 28) == 89);
    VJ_CHECK(wire<uint32_t>(ga, 32) == 16 && wire<uint32_t>(ga, 36) == 0);
    VJ_CHECK(wire<uint32_t>(ga, 40) == 16 && wire<uint32_t>(ga, 44) == 1);
    VJ_CHECK(wire<uint32_t>(ga, 48) == 100 && wire<uint32_t>(ga, 52) == 101);
    VJ_CHECK(wire<uint32_t>(ga, 56) == ornith_mega::kSkipFirstAr);
    ornith_mega::GqaArgs qa{ga.desc, 47, 11, 95, 32767, 164, ornith_mega::kFinalAr | ornith_mega::kFinalNorm};
    VJ_CHECK(wire<uint64_t>(qa, 0) == ga.desc);
    VJ_CHECK(wire<uint32_t>(qa, 8) == 47 && wire<uint32_t>(qa, 12) == 11);
    VJ_CHECK(wire<uint32_t>(qa, 16) == 95 && wire<uint32_t>(qa, 20) == 32767);
    VJ_CHECK(wire<uint32_t>(qa, 24) == 164 && wire<uint32_t>(qa, 28) == 3);
    // Sparse endpoints preserve the super slot order (GU=15 before Down=14).
    std::array<uint64_t, 17> record{};
    std::array<uint64_t, 16> deltas{};
    record[16] = 100; record[10] = 120; record[11] = 150;
    record[12] = 160; record[15] = 230; record[14] = 280;
    VJ_CHECK(ornith_mega::profile_deltas(record, deltas));
    VJ_CHECK(deltas[10] == 20 && deltas[11] == 30 && deltas[12] == 10);
    VJ_CHECK(deltas[15] == 70 && deltas[14] == 50);
    const auto before = deltas;
    VJ_CHECK(!ornith_mega::profile_deltas(record, deltas, 1u << 0u));
    VJ_CHECK(deltas == before);  // incomplete record must not partially add
    record[14] = 200;  // bad endpoint must not underflow or partially add
    VJ_CHECK(!ornith_mega::profile_deltas(record, deltas));
    VJ_CHECK(deltas == before);
    record.fill(0);
    VJ_CHECK(!ornith_mega::profile_deltas(record, deltas));
    record[16] = 100;  // start without an endpoint is incomplete too
    VJ_CHECK(!ornith_mega::profile_deltas(record, deltas));
    // Final AR is an independent 97th record, outside the barrier region.
    ornith_mega::ProfileRecords records{};
    records[96][16] = 1000; records[96][4] = 1025;
    deltas.fill(0);
    VJ_CHECK(ornith_mega::profile_deltas(records[96], deltas) && deltas[4] == 25);
    VJ_CHECK(ornith_mega::kProfileOffset + sizeof(records) <= ornith_mega::kScratchBytes);
    std::puts("mega descriptor and kernarg ABI: OK (TP=3 uneven shards, 17-slot wrap)");
}
