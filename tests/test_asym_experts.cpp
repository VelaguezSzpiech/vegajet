// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/sharder.hpp"
#include "core/q4_dot8_repack.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); return 1; } } while (0)
using namespace vegajet;
static uint32_t rd32(const uint8_t* p) { uint32_t x; std::memcpy(&x, p, 4); return x; }
int main() {
    for (double a : {1.0, 1080.0, 1140.0, 840.0, 972.0, 1e-9, 1e9}) {
        for (double b : {1.0, 1080.0, 1140.0, 840.0, 972.0, 1e-9, 1e9}) {
            const auto off = shard::split_expert_intermediate(1024, {a,b}, false);
            CHECK(off.size() == 2 && off[0].size() == 512 && off[1].begin == 512 && off[1].size() == 512);
            const auto on = shard::split_expert_intermediate(1024, {a,b}, true);
            CHECK(on.size() == 2 && on[0].begin == 0 && on[0].end == on[1].begin && on[1].end == 1024);
            CHECK(on[0].size() >= 32 && on[1].size() >= 32 && on[0].size() % 32 == 0 && on[1].size() % 32 == 0);
            CHECK(std::fabs(double(on[0].size()) - 1024 * a / (a+b)) <= 32.000001);
        }
    }
    const auto clock = shard::split_expert_intermediate(1024, {1080,1140}, true);
    CHECK(clock[0].size() == 512 && clock[1].size() == 512); // 32-row rounding can hide a small clock skew.
    const auto measured = shard::split_expert_intermediate(1024, {840,972}, true);
    CHECK(measured[0].size() == 480 && measured[1].size() == 544);
    CHECK(shard::split_expert_intermediate(1024, {0,1}, true).empty());
    CHECK(shard::split_expert_intermediate(1024, {NAN,1}, true).empty());
    CHECK(shard::split_expert_intermediate(1024, {1,INFINITY}, true).empty());
    CHECK(shard::split_expert_intermediate(1024, {1}, true).empty());
    CHECK(shard::split_expert_intermediate(1000, {1,1}, true).empty());

    // Every possible aligned split: source expert stride, row coverage and local promotion offsets.
    for (uint32_t cut = 32; cut < 1024; cut += 32) {
        for (uint32_t e = 0; e < 256; ++e) for (uint32_t r = 0; r < 1024; ++r) {
            const uint32_t begin = r < cut ? 0 : cut;
            CHECK(shard::expert_row_index(e, 1024, begin, r-begin) == uint64_t(e)*1024+r);
        }
    }
    // Static s224 stage A covers each routed tile and shared output row exactly once.
    for (uint32_t lff=32; lff<1024; lff+=32) {
        const uint32_t n_gu=8*(lff/32);
        std::vector<uint32_t> gu(n_gu), shared(3072);
        for (uint32_t wg=0; wg<224; ++wg) for (uint32_t task=wg; task<n_gu+96; task+=224) {
            if (task<n_gu) ++gu[task];
            else for (uint32_t i=0; i<32; ++i) ++shared[(task-n_gu)*32+i];
        }
        for (auto n : gu) CHECK(n==1);
        for (auto n : shared) CHECK(n==1);
        const auto per_expert=dot8_gu_iq2s_layout(lff,3072);
        const auto cache=dot8_gu_iq2s_layout(44ULL*lff,3072);
        CHECK(cache.w_plane_bytes()==44*per_expert.w_plane_bytes());
        CHECK(cache.s_plane_bytes()==44*per_expert.s_plane_bytes());
        for (uint32_t slot=0; slot<44; ++slot) {
            const uint64_t row=uint64_t(slot)*lff+(lff-1);
            CHECK((row+1)*cache.w_row_stride_bytes<=cache.w_plane_bytes());
            if (!cache.scale_embedded) CHECK((row+1)*cache.s_row_stride_bytes<=cache.s_plane_bytes());
        }
    }
    // Distinct source rows/superblocks expose offsets crossing the 256-weight boundary.
    constexpr uint32_t rows = 6, rb = 4 * kIq3xxsBlockBytes;
    std::vector<uint8_t> src(rows*rb);
    uint32_t rng = 42;
    for (auto& x : src) { rng = rng*1664525u+1013904223u; x = uint8_t(rng >> 24); }
    for (uint32_t r=0; r<rows; ++r) for (uint32_t sb=0; sb<4; ++sb) {
        const uint16_t h = uint16_t(0x3400u + r*31u + sb*127u);
        std::memcpy(src.data()+r*rb+sb*kIq3xxsBlockBytes, &h, 2);
    }
    const auto full = dot8_iq3_layout(rows,1024);
    std::vector<uint8_t> fw(full.w_plane_bytes()), fs(full.s_plane_bytes());
    std::string err;
    CHECK(dot8_repack_iq3xxs_rows(src.data(),rb,0,full,0,rows,fw.data(),fs.data(),&err));
    uint32_t cases=0;
    for (uint32_t cut=32; cut<1024; cut+=32) for (uint32_t rank=0; rank<2; ++rank) {
        const uint32_t begin=rank?cut:0, cols=rank?1024-cut:cut;
        const auto lay=dot8_iq3_asym_layout(rows,cols);
        std::vector<uint8_t> w(lay.w_plane_bytes()+16,0xA5), s(lay.s_plane_bytes()+16,0xA5);
        // Exercise nonzero row_begin and chunked repacks (OpenMP uses disjoint ranges).
        CHECK(dot8_repack_iq3xxs_asym_rows(src.data(),rb,begin,lay,2,rows,w.data(),s.data(),&err));
        CHECK(w[0]==0xA5);
        CHECK(dot8_repack_iq3xxs_asym_rows(src.data(),rb,begin,lay,0,2,w.data(),s.data(),&err));
        for (uint32_t r=0; r<rows; ++r) {
            const auto* scale=lay.scale_embedded?w.data()+r*lay.w_row_stride_bytes+lay.s_offset_bytes:s.data()+r*lay.s_row_stride_bytes;
            const auto* ref=full.scale_embedded?fw.data()+r*full.w_row_stride_bytes+full.s_offset_bytes:fs.data()+r*full.s_row_stride_bytes;
            for (uint32_t b=0; b<cols/32; ++b) {
                const uint32_t sb=begin/32+b;
                CHECK(std::memcmp(w.data()+r*lay.w_row_stride_bytes+b*16,fw.data()+r*full.w_row_stride_bytes+sb*16,16)==0);
                CHECK(rd32(scale+b*8)==((rd32(ref+(sb/8)*8) >> ((sb%8)*4))&15));
                CHECK(rd32(scale+b*8+4)==rd32(ref+(sb/8)*8+4));
            }
        }
        for (uint32_t i=0; i<16; ++i) { CHECK(w[lay.w_plane_bytes()+i]==0xA5); CHECK(s[lay.s_plane_bytes()+i]==0xA5); }
        CHECK(!dot8_repack_iq3xxs_asym_rows(src.data(),rb,begin+1,lay,0,rows,w.data(),s.data(),&err));
        CHECK(!dot8_repack_iq3xxs_asym_rows(src.data(),rb,1024,lay,0,rows,w.data(),s.data(),&err));
        CHECK(!dot8_repack_iq3xxs_asym_rows(src.data(),rb,begin,lay,0,rows+1,w.data(),s.data(),&err));
        ++cases;
    }
    std::printf("PASS asym experts: default 512/512, measured 480/544, %u Down slices exact; GU offsets cover all experts\n",cases);
    return 0;
}
