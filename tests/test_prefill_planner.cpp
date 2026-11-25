// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Big Lever 3 prefill planner: scratch sizing, GDN / attention / AllReduce schedules, and the exact CPU reference
// of the block-32 INT8 AllReduce wire (pack / unpack / accumulate, error bound, rank commutativity).
#include "core/prefill_planner.hpp"
#include "tests/test_common.hpp"

#include <cmath>
#include <random>

using namespace vegajet::prefill;

namespace {

void test_ornith_plan() {
    const ModelDims dm = ornith_122b_tp2();
    for (uint32_t M : {256u, 384u, 512u}) {
        Request rq;
        rq.prompt_tokens = 2048;
        rq.macro_m = M;
        rq.gdn_chunk = 32;
        rq.precision = Precision::kFp16Intermediate;
        const Plan p = plan(dm, rq);
        VJ_CHECK(p.ok);
        VJ_CHECK(p.macro_m == M);
        VJ_CHECK(p.n_macro == (2048u + M - 1u) / M);
        const double mib = static_cast<double>(p.bytes_no_alias) / 1048576.0;
        const double mib_a = static_cast<double>(p.bytes_aliased) / 1048576.0;
        std::printf("[planner] M=%u fp16: %.2f MiB/GPU (aliased %.2f)\n", M, mib, mib_a);
        VJ_CHECK(p.bytes_aliased < p.bytes_no_alias);
        // Expected envelope for the FP16-intermediate layout at M = 512: 41..59 MiB/GPU once the GDN and attention
        // working sets share one region (a layer is one or the other).
        if (M == 512u) { VJ_CHECK(mib_a > 41.0 && mib_a < 59.0); }
        VJ_CHECK(p.gdn.wgs == 128u);
        VJ_CHECK(p.gdn.chunks_per_macro == M / 32u);
        VJ_CHECK(p.attn.wgs_per_macro == M);            // G = 16 -> one token (16 heads) per WG
        VJ_CHECK(p.ar.fp16_bytes == static_cast<uint64_t>(M) * 3072u * 2u);
        VJ_CHECK(p.ar.int8_bytes == static_cast<uint64_t>(M) * 3456u);
        VJ_CHECK(!p.ar.use_int8);                       // opt-in only
        VJ_CHECK(p.ar.use_sdma);
        VJ_CHECK(p.ar.fp16_chunks == (static_cast<uint64_t>(M) * 6144u + 1048575u) / 1048576u);
        if (M == 512u) std::printf("%s", describe(p).c_str());
    }
    {
        Request rq;
        rq.prompt_tokens = 512;
        rq.macro_m = 512;
        rq.precision = Precision::kFp32Legacy;
        const Plan p = plan(dm, rq);
        VJ_CHECK(p.ok);
        std::printf("[planner] M=512 fp32 legacy: %.2f MiB/GPU (aliased %.2f)\n",
                    static_cast<double>(p.bytes_no_alias) / 1048576.0, static_cast<double>(p.bytes_aliased) / 1048576.0);
        Request r16 = rq;
        r16.precision = Precision::kFp16Intermediate;
        VJ_CHECK(plan(dm, r16).bytes_no_alias < p.bytes_no_alias);
    }
    {
        // Attention tile count, causal: prompt 64 tokens, pos0 0, 1 macro: WG m covers keys 0..m -> ceil((m+1)/32).
        Request rq;
        rq.prompt_tokens = 64;
        rq.macro_m = 64;
        const Plan p = plan(dm, rq);
        uint64_t want = 0;
        for (uint32_t m = 0; m < 64; ++m) want += (m + 1u + 31u) / 32u;
        VJ_CHECK(p.attn.kv_tiles_total == want * dm.n_attn_layers);
        VJ_CHECK(p.attn.kv_bytes_streamed == p.attn.kv_tiles_total * 2u * 32u * 272u);
    }
    {
        Request rq;
        rq.prompt_tokens = 100;
        rq.gdn_chunk = 24;
        VJ_CHECK(!plan(dm, rq).ok);
        rq.gdn_chunk = 16;
        rq.macro_m = 512;
        const Plan p = plan(dm, rq);
        VJ_CHECK(p.ok && p.macro_m == 112u && p.n_macro == 1u);   // clamped to the prompt (multiple of 16)
    }
}

// Straightforward re-statement of the device expressions for an independent bit check.
void ref_pack(const float* a, int8_t* q, float* s) {
    for (int b = 0; b < 96; ++b) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(a[b * 32 + i]));
        volatile float ld = amax * (1.0f / 127.0f);
        s[b] = ld;
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        for (int i = 0; i < 32; ++i) {
            volatile float t = a[b * 32 + i] * inv;
            long r = std::lrintf(t);
            if (r > 127) r = 127;
            if (r < -127) r = -127;
            q[b * 32 + i] = static_cast<int8_t>(r);
        }
    }
}

void test_ar_int8() {
    constexpr uint32_t D = 3072;
    std::mt19937 rng(12345);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> l(D), p(D), h0(D), h1(D), h2(D), dq(D), sl(D / 32), sp(D / 32), s_ref(D / 32);
    std::vector<int8_t> ql(D), qp(D), q_ref(D);
    double max_ratio = 0.0;
    uint64_t checked = 0;
    for (int trial = 0; trial < 400; ++trial) {
        const float scale = std::ldexp(1.0f, (trial % 24) - 12);
        for (uint32_t i = 0; i < D; ++i) {
            l[i] = nd(rng) * scale;
            p[i] = nd(rng) * scale * ((i % 97 == 0) ? 50.0f : 1.0f);   // outliers
            h0[i] = nd(rng);
        }
        if (trial == 0) for (uint32_t i = 0; i < 64; ++i) l[i] = 0.0f;   // all-zero blocks -> scale 0
        ar_q8_pack_row(l.data(), D, ql.data(), sl.data());
        ar_q8_pack_row(p.data(), D, qp.data(), sp.data());
        ref_pack(l.data(), q_ref.data(), s_ref.data());
        VJ_CHECK(std::memcmp(q_ref.data(), ql.data(), D) == 0);
        VJ_CHECK(std::memcmp(s_ref.data(), sl.data(), sizeof(float) * D / 32) == 0);
        // Error bound per element.
        ar_q8_unpack_row(ql.data(), sl.data(), D, dq.data());
        for (uint32_t b = 0; b < D / 32; ++b) {
            float amax = 0.0f;
            for (uint32_t i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(l[b * 32 + i]));
            const float bound = ar_q8_err_bound(amax);
            for (uint32_t i = 0; i < 32; ++i) {
                const float e = std::fabs(dq[b * 32 + i] - l[b * 32 + i]);
                VJ_CHECK(e <= bound);
                if (bound > 0.0f) max_ratio = std::max(max_ratio, static_cast<double>(e / bound));
                ++checked;
            }
            for (uint32_t i = 0; i < 32; ++i) VJ_CHECK(ql[b * 32 + i] != -128);
        }
        // Rank commutativity: rank 0 adds (local=l, peer=p), rank 1 adds (local=p, peer=l): bit-identical.
        h1 = h0; h2 = h0;
        ar_q8_accum_row(h1.data(), ql.data(), sl.data(), qp.data(), sp.data(), D);
        ar_q8_accum_row(h2.data(), qp.data(), sp.data(), ql.data(), sl.data(), D);
        VJ_CHECK(std::memcmp(h1.data(), h2.data(), sizeof(float) * D) == 0);
        // FP16 wire is also commutative.
        h1 = h0; h2 = h0;
        ar_f16_accum_row(h1.data(), l.data(), p.data(), D);
        ar_f16_accum_row(h2.data(), p.data(), l.data(), D);
        VJ_CHECK(std::memcmp(h1.data(), h2.data(), sizeof(float) * D) == 0);
    }
    std::printf("[ar int8] %llu elements within bound, max err/bound = %.4f\n",
                static_cast<unsigned long long>(checked), max_ratio);
    VJ_CHECK(max_ratio > 0.9);   // the bound is tight (half a quantisation step)
}

void test_f16() {
    // Every finite half round-trips; RNE ties.
    for (uint32_t h = 0; h < 65536u; ++h) {
        if (((h >> 10) & 0x1Fu) == 0x1Fu) continue;
        const float f = f16_to_f32(static_cast<uint16_t>(h));
        VJ_CHECK(f32_to_f16_rne(f) == h || (f == 0.0f && (f32_to_f16_rne(f) & 0x7FFFu) == 0u));
    }
    VJ_CHECK(f32_to_f16_rne(1.0f + 0x1p-11f) == 0x3C00u);            // tie -> even
    VJ_CHECK(f32_to_f16_rne(1.0f + 3.0f * 0x1p-11f) == 0x3C02u);     // tie -> even (up)
    VJ_CHECK(f32_to_f16_rne(65520.0f) == 0x7C00u);                   // overflow -> inf
    VJ_CHECK(f16_to_f32(0x0001u) == 0x1p-24f);
}

} // namespace

int main() {
    test_ornith_plan();
    test_ar_int8();
    test_f16();
    std::printf("test_prefill_planner: OK\n");
    return 0;
}
