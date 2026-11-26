// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// CPU arithmetic/layout model, not a GPU or end-to-end parity test.
// Reference: vj_ornith_super.hip's 32 four-row FMA partials, XOR 32/16/8,
// then (wave0+wave1)+(wave2+wave3). Candidate: the 1024-thread verifier.
// The old 16 eight-row and super-off 8 sixteen-row orders are negative controls.
#include "tests/test_common.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace {
constexpr unsigned H = 128, NK = 8, NV = 32, K = NV * H;
constexpr unsigned CONV = (2 * NK + NV) * H, STRIDE = 10304, ZOFF = 6144;
constexpr unsigned MAXM = 6, SLOTS = 17, OUT = 32;
constexpr size_t STATE = NV * H * H;
struct DS { float d, sum; };
struct Fixture {
    std::vector<float> state, conv, gab, decay, beta;
    std::array<float, H> norm{};
};
struct Run {
    std::vector<float> raw, states;
    std::vector<int8_t> q8;
    std::vector<DS> ds;
    std::vector<float> partial;
};

uint32_t rng = 0x6a09e667u;
uint32_t next() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
float sample(float scale) { return float(int(next() % 20001u) - 10000) * (scale / 10007.0f); }
Fixture fixture() {
    Fixture f;
    f.state.resize(STATE); f.conv.resize(MAXM * CONV); f.gab.resize(MAXM * STRIDE);
    f.decay.resize(MAXM * NV); f.beta.resize(MAXM * NV);
    for (float& v : f.state) v = sample(0.2f);
    for (float& v : f.conv) v = sample(0.15f);
    for (float& v : f.gab) v = sample(2.0f);
    for (float& v : f.decay) v = 0.8f + sample(0.15f);
    for (float& v : f.beta) v = 0.5f + sample(0.4f);
    for (float& v : f.norm) v = 1.0f + sample(0.3f);
    return f;
}

// Source-independent per-column spelling of the production superkernel tree.
float super_dot(const float* s, const float* x) {
    float p[32], wave[4];
    for (unsigned g = 0; g < 32; ++g) {
        const unsigned r = 4 * g;
        p[g] = std::fma(s[(r + 3) * H], x[r + 3],
               std::fma(s[(r + 2) * H], x[r + 2],
               std::fma(s[(r + 1) * H], x[r + 1], s[r * H] * x[r])));
    }
    for (unsigned w = 0; w < 4; ++w) {
        const float* a = p + w * 8;
        wave[w] = ((a[0] + a[4]) + (a[2] + a[6]))
                + ((a[1] + a[5]) + (a[3] + a[7]));
    }
    return (wave[0] + wave[1]) + (wave[2] + wave[3]);
}
float old_dot(const float* s, const float* x, unsigned rows) {
    float total = 0.0f;
    for (unsigned r0 = 0; r0 < H; r0 += rows) {
        float p = 0.0f;
        for (unsigned r = r0; r < r0 + rows; ++r) p = std::fma(s[r * H], x[r], p);
        total += p;
    }
    return total;
}
Run sequential(const Fixture& f, unsigned rows) {
    Run out;
    out.raw.resize(MAXM * K); out.states.resize(MAXM * STATE);
    auto state = f.state;
    for (unsigned m = 0; m < MAXM; ++m) {
        for (unsigned h = 0; h < NV; ++h) {
            const float* base = f.conv.data() + m * CONV;
            const float* q = base + (h % NK) * H;
            const float* k = base + (NK + h % NK) * H;
            const float* v = base + (2 * NK + h) * H;
            float* s = state.data() + h * H * H;
            const float g = f.decay[m * NV + h], b = f.beta[m * NV + h];
            for (unsigned j = 0; j < H; ++j) {
                const float pred = rows == 4 ? super_dot(s + j, k) : old_dot(s + j, k, rows);
                const float dv = b * std::fma(-g, pred, v[j]);
                for (unsigned i = 0; i < H; ++i) s[i * H + j] = std::fma(k[i], dv, g * s[i * H + j]);
                out.raw[m * K + h * H + j] = rows == 4 ? super_dot(s + j, q) : old_dot(s + j, q, rows);
            }
        }
        std::copy(state.begin(), state.end(), out.states.begin() + m * STATE);
    }
    return out;
}

template<size_t N> void xor_sum(std::array<float, N>& v, unsigned mask) {
    const auto old = v;
    for (unsigned i = 0; i < N; ++i) v[i] += old[i ^ mask];
}

// Mirror the new kernel's actual 1024-thread mapping and LDS indices separately
// from super_dot. State remains register-resident across rows; each ring slot is
// checked, including wraparound. Sentinel slots catch unintended stores.
Run batched(const Fixture& f, unsigned M, unsigned head, bool ring, bool independent=false) {
    Run out;
    out.raw.resize(M * K); out.states.resize(M * STATE);
    std::vector<float> slots(SLOTS * STATE, -9876.5f);
    std::copy(f.state.begin(), f.state.end(), slots.begin() + head * STATE);
    if (independent) for (unsigned m=0;m<M;++m)
        std::copy(f.state.begin(),f.state.end(),slots.begin()+(head+m)*STATE);
    const unsigned nr = ring ? 4 : 8, nc = ring ? 4 : 2;
    auto row0 = [ring](unsigned t) { return ring ? ((t & 255) >> 3) * 4 : (t >> 6) * 8; };
    auto col0 = [ring](unsigned t) { return ring ? ((t >> 8) * 8 + (t & 7)) * 4 : (t & 63) * 2; };
    for (unsigned h = 0; h < NV; ++h) {
        float reg[1024][8][4];
        for (unsigned t = 0; t < 1024; ++t) {
            const unsigned r0 = row0(t), c0 = col0(t);
            for (unsigned r = 0; r < nr; ++r)
                for (unsigned c = 0; c < nc; ++c)
                    reg[t][r][c] = slots[head * STATE + h * H * H + (r0 + r) * H + c0 + c];
        }
        for (unsigned m = 0; m < M; ++m) {
            if (independent) for (unsigned t=0;t<1024;++t)
                for (unsigned r=0;r<nr;++r) for (unsigned c=0;c<nc;++c)
                    reg[t][r][c]=slots[(head+m)*STATE+h*H*H+(row0(t)+r)*H+col0(t)+c];
            const float* base = f.conv.data() + m * CONV;
            const float* q = base + (h % NK) * H;
            const float* k = base + (NK + h % NK) * H;
            const float* v = base + (2 * NK + h) * H;
            const float g = f.decay[m * NV + h], b = f.beta[m * NV + h];
            float red[16 * H], dv[H];
            for (unsigned pass = 0; pass < 2; ++pass) {
                float p[1024][4]{};
                for (unsigned t = 0; t < 1024; ++t) {
                    const unsigned r0 = row0(t), c0 = col0(t);
                    for (unsigned r = 0; r < nr; ++r) {
                        for (unsigned c = 0; c < nc; ++c) {
                            if (pass) reg[t][r][c] = std::fma(k[r0 + r], dv[c0 + c], g * reg[t][r][c]);
                            const float x = (pass ? q : k)[r0 + r];
                            p[t][c] = ring && r == 0 ? reg[t][r][c] * x : std::fma(reg[t][r][c], x, p[t][c]);
                        }
                    }
                }
                if (ring) {
                    for (unsigned w = 0; w < 16; ++w) {
                        for (unsigned c = 0; c < 4; ++c) {
                            std::array<float, 64> lanes;
                            for (unsigned l = 0; l < 64; ++l) lanes[l] = p[w * 64 + l][c];
                            for (unsigned mask : {32u, 16u, 8u}) xor_sum(lanes, mask);
                            for (unsigned l = 0; l < 8; ++l) {
                                const unsigned t = w * 64 + l;
                                const unsigned col4 = (t >> 8) * 8 + (t & 7);
                                red[(w & 3) * H + col4 * 4 + c] = lanes[l];
                            }
                        }
                    }
                } else {
                    for (unsigned t = 0; t < 1024; ++t)
                        for (unsigned c = 0; c < 2; ++c)
                            red[(t >> 6) * H + (t & 63) * 2 + c] = p[t][c];
                }
                for (unsigned j = 0; j < H; ++j) {
                    float value = 0.0f;
                    if (ring) value = (red[j] + red[H + j]) + (red[2 * H + j] + red[3 * H + j]);
                    else for (unsigned w = 0; w < 16; ++w) value += red[w * H + j];
                    if (pass) out.raw[m * K + h * H + j] = value;
                    else dv[j] = b * std::fma(-g, value, v[j]);
                }
            }
            const unsigned dst = independent ? head+m : (ring ? (head + m + 1) % SLOTS : head);
            for (unsigned t = 0; t < 1024; ++t) {
                const unsigned r0 = row0(t), c0 = col0(t);
                for (unsigned r = 0; r < nr; ++r) {
                    for (unsigned c = 0; c < nc; ++c) {
                        const size_t i = h * H * H + (r0 + r) * H + c0 + c;
                        out.states[m * STATE + i] = reg[t][r][c];
                        if (ring || m + 1 == M) slots[dst * STATE + i] = reg[t][r][c];
                    }
                }
            }
        }
    }
    for (unsigned slot = 0; slot < SLOTS; ++slot) {
        const unsigned distance = (slot + SLOTS - head) % SLOTS;
        if (independent && slot>=head && slot<head+M) {
            VJ_CHECK(std::memcmp(slots.data()+slot*STATE,out.states.data()+(slot-head)*STATE,STATE*sizeof(float))==0);
        } else if (independent) {
            for (size_t i=0;i<STATE;++i) VJ_CHECK(slots[slot*STATE+i]==-9876.5f);
        } else if (ring && distance > 0 && distance <= M)
            VJ_CHECK(std::memcmp(slots.data() + slot * STATE, out.states.data() + (distance - 1) * STATE, STATE * sizeof(float)) == 0);
        else if (slot == head) {
            const float* expected = ring ? f.state.data() : out.states.data() + (M - 1) * STATE;
            VJ_CHECK(std::memcmp(slots.data() + slot * STATE, expected, STATE * sizeof(float)) == 0);
        } else {
            for (size_t i = 0; i < STATE; ++i) VJ_CHECK(slots[slot * STATE + i] == -9876.5f);
        }
    }
    return out;
}

void quantize(const Fixture& f, Run& out) {
    const unsigned M = out.raw.size() / K;
    out.q8.resize(M * K); out.ds.resize(M * K / 32);
    for (unsigned m = 0; m < M; ++m) {
        for (unsigned h = 0; h < NV; ++h) {
            const float* fo = out.raw.data() + m * K + h * H;
            std::array<float, 64> sq;
            for (unsigned l = 0; l < 64; ++l) sq[l] = std::fma(fo[2 * l], fo[2 * l], fo[2 * l + 1] * fo[2 * l + 1]);
            for (unsigned mask : {1u, 2u, 4u, 8u, 16u, 32u}) xor_sum(sq, mask);
            float y[H];
            for (unsigned j = 0; j < H; ++j) {
                const float inv = 1.0f / std::sqrt(std::fma(sq[j / 2], 1.0f / H, 1e-6f));
                const float z = f.gab[m * STRIDE + ZOFF + h * H + j];
                y[j] = (fo[j] * inv * f.norm[j]) * (z / (1.0f + std::exp(-z)));
            }
            for (unsigned block = 0; block < 4; ++block) {
                float amax = 0.0f;
                std::array<float, 16> sum;
                for (unsigned j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(y[block * 32 + j]));
                for (unsigned l = 0; l < 16; ++l) sum[l] = y[block * 32 + 2 * l] + y[block * 32 + 2 * l + 1];
                for (unsigned mask : {8u, 4u, 2u, 1u}) xor_sum(sum, mask);
                const size_t b = (m * NV + h) * 4 + block;
                out.ds[b] = {amax * (1.0f / 127.0f), sum[0]};
                const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
                for (unsigned j = 0; j < 32; ++j)
                    out.q8[b * 32 + j] = static_cast<int8_t>(std::clamp(int(std::nearbyint(y[block * 32 + j] * inv)), -127, 127));
            }
        }
    }
}

// Both real projection kernels use b=lane+32*r, the same nested FMA, and XOR
// 16,8,4,2,1. Synthetic u4 + exactly representable FP16 d/m weights isolate DS.
void project(Run& out, bool batch) {
    const unsigned M = out.q8.size() / K;
    out.partial.resize(M * OUT);
    for (unsigned row = 0; row < OUT; ++row) {
        std::array<std::array<float, 32>, MAXM> acc{};
        auto mac = [&](unsigned m, unsigned block) {
            int dp = 0;
            for (unsigned j = 0; j < 32; ++j) {
                const int w = (row * 7 + block * 3 + j * 11) & 15;
                dp += w * out.q8[m * K + block * 32 + j];
            }
            const float wd = float(1 + ((row + block) & 15)) / 256.0f;
            const float wm = float(int((row * 3 + block) & 15) - 8) / 128.0f;
            const DS ds = out.ds[m * (K / 32) + block];
            float& a = acc[m][block & 31];
            a = std::fma(wd * ds.d, float(dp), std::fma(wm, ds.sum, a));
        };
        if (batch) {
            for (unsigned tile = 0; tile < K / 32; tile += 32)
                for (unsigned l = 0; l < 32; ++l)
                    for (unsigned m = 0; m < M; ++m) mac(m, tile + l);
        } else {
            for (unsigned m = 0; m < M; ++m)
                for (unsigned l = 0; l < 32; ++l)
                    for (unsigned pair = 0; pair < 2; ++pair) {
                        mac(m, l + pair * 64); mac(m, l + pair * 64 + 32);
                    }
        }
        for (unsigned m = 0; m < M; ++m) {
            for (unsigned mask : {16u, 8u, 4u, 2u, 1u}) xor_sum(acc[m], mask);
            out.partial[m * OUT + row] = acc[m][0];
        }
    }
}
size_t different(const void* a, const void* b, size_t bytes) {
    size_t n = 0;
    const auto* x = static_cast<const uint8_t*>(a); const auto* y = static_cast<const uint8_t*>(b);
    for (size_t i = 0; i < bytes; ++i) n += x[i] != y[i];
    return n;
}
} // namespace

int main() {
    const auto f = fixture();
    auto ref = sequential(f, 4), old = sequential(f, 8), split = sequential(f, 16);
    quantize(f, ref); quantize(f, old); quantize(f, split);
    project(ref, false); project(old, true);
    size_t stable_q8_blocks = 0;
    for (unsigned b = K / 32; b < 2 * K / 32; ++b)
        stable_q8_blocks += !std::memcmp(ref.q8.data() + b * 32, old.q8.data() + b * 32, 32)
                        && std::memcmp(&ref.ds[b], &old.ds[b], sizeof(DS)) != 0;
    VJ_CHECK(stable_q8_blocks > 0);
    for (unsigned m = 0; m < MAXM; ++m) {
        const size_t ds = different(ref.ds.data() + m * K / 32, old.ds.data() + m * K / 32, K / 32 * sizeof(DS));
        const size_t p = different(ref.partial.data() + m * OUT, old.partial.data() + m * OUT, OUT * sizeof(float));
        VJ_CHECK(ds > 0 && p > 0);
        std::printf("before row=%u q8_bytes=%zu/%u ds_bytes=%zu/%zu partial_bytes=%zu/%zu\n", m,
                    different(ref.q8.data() + m * K, old.q8.data() + m * K, K), K, ds, K / 32 * sizeof(DS), p, OUT * sizeof(float));
    }
    VJ_CHECK(std::memcmp(ref.states.data(), split.states.data(), ref.states.size() * sizeof(float)) != 0);
    std::printf("row1 unchanged-Q8/different-DS blocks=%zu/128; super-off 16-row negative control differs\n", stable_q8_blocks);
    for (unsigned M = 2; M <= MAXM; ++M) {
        for (bool ring : {false, true}) {
            const unsigned head = ring ? 16 : 0;
            auto candidate = batched(f, M, head, ring);
            const Run& expected = ring ? ref : old;
            VJ_CHECK(std::memcmp(expected.raw.data(), candidate.raw.data(), candidate.raw.size() * sizeof(float)) == 0);
            VJ_CHECK(std::memcmp(expected.states.data(), candidate.states.data(), candidate.states.size() * sizeof(float)) == 0);
            quantize(f, candidate); project(candidate, true);
            VJ_CHECK(std::memcmp(expected.q8.data(), candidate.q8.data(), candidate.q8.size()) == 0);
            VJ_CHECK(std::memcmp(expected.ds.data(), candidate.ds.data(), candidate.ds.size() * sizeof(DS)) == 0);
            VJ_CHECK(std::memcmp(expected.partial.data(), candidate.partial.data(), candidate.partial.size() * sizeof(float)) == 0);
            std::printf("after M=%u rows=0..%u ring=%u head=%u vs=%s raw/state/Q8/DS/partial bit-exact PASS\n",
                        M, M - 1, unsigned(ring), head, ring ? "super-M1" : "unchanged-nonring");
        }
    }
    // Independent sequences: same initial state, distinct token rows. Compare
    // every row against a separate raw M=1 invocation, including committed state.
    auto independent=batched(f,MAXM,0,true,true);
    quantize(f,independent); project(independent,true);
    for (unsigned m=0;m<MAXM;++m) {
        auto single=f;
        std::copy_n(f.conv.begin()+m*CONV,CONV,single.conv.begin());
        std::copy_n(f.gab.begin()+m*STRIDE,STRIDE,single.gab.begin());
        std::copy_n(f.decay.begin()+m*NV,NV,single.decay.begin());
        std::copy_n(f.beta.begin()+m*NV,NV,single.beta.begin());
        auto expected=sequential(single,4);
        quantize(single,expected); project(expected,false);
        VJ_CHECK(std::memcmp(expected.q8.data(),independent.q8.data()+m*K,K)==0);
        VJ_CHECK(std::memcmp(expected.ds.data(),independent.ds.data()+m*K/32,K/32*sizeof(DS))==0);
        VJ_CHECK(std::memcmp(expected.partial.data(),independent.partial.data()+m*OUT,OUT*sizeof(float))==0);
        VJ_CHECK(std::memcmp(expected.raw.data(),independent.raw.data()+m*K,K*sizeof(float))==0);
        VJ_CHECK(std::memcmp(expected.states.data(),independent.states.data()+m*STATE,STATE*sizeof(float))==0);
    }
    std::puts("test_gdn_batch_order: PASS (CPU synthetic ordering/layout proof; no device parity claim)");
}
