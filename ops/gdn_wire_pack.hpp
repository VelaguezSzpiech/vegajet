// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Production input producer for the sealed GDN tree-WY verifier.
//
// The normal engine's GDN conv kernel writes one interleaved token row per verified token:
//   conv[t][0                .. nk*128)        query  (already L2-normalized, q scaled by 1/sqrt(128))
//   conv[t][nk*128           .. 2*nk*128)      key    (already L2-normalized)
//   conv[t][2*nk*128         .. 2*nk*128+nv*128) value (post-conv SiLU)
// plus decay[t][nv] (alpha*exp form) and beta[t][nv] (sigmoid). The sealed verifier instead reads
// five token-major planes (query [m][8][128], key [m][8][128], value [m][32][128], decay [m][32],
// beta [m][32]) and an int32 parent[m] chain. This header owns the only description of that
// transformation: descriptor validation, the byte-exact kernarg image and a host reference used
// by the CPU tests. It is independent of the attention wire producer lane.
//
// The device kernel (ops/kernels/vj_gdn_wire_pack.hip) performs a value-preserving strided copy.
// It must not reorder, rescale or re-normalize anything: the sequential `vj_spec_gdn` verifier and
// the sealed verifier consume exactly the same conv outputs.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

namespace vegajet::gdn::wire_pack {

constexpr uint32_t kQueryHeads = 8;
constexpr uint32_t kValueHeads = 32;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kThreads = 256;
// Interleaved conv row: query heads, key heads, value heads (all head_dim wide).
constexpr uint32_t kConvStride = (2u * kQueryHeads + kValueHeads) * kHeadDim;

constexpr bool is_sealed_width(uint32_t m) { return m == 5u || m == 8u || m == 12u; }
constexpr uint32_t grid_x(uint32_t m) { return m * kThreads; }  // KfdRuntime::enqueue_kernel counts work-items

constexpr uint64_t plane_bytes(uint32_t m, uint32_t heads) {
    return static_cast<uint64_t>(m) * heads * kHeadDim * sizeof(float);
}
constexpr uint64_t query_bytes(uint32_t m) { return plane_bytes(m, kQueryHeads); }
constexpr uint64_t key_bytes(uint32_t m) { return plane_bytes(m, kQueryHeads); }
constexpr uint64_t value_bytes(uint32_t m) { return plane_bytes(m, kValueHeads); }
constexpr uint64_t gate_bytes(uint32_t m) { return static_cast<uint64_t>(m) * kValueHeads * sizeof(float); }
constexpr uint64_t parent_bytes(uint32_t m) { return static_cast<uint64_t>(m) * sizeof(int32_t); }

// Byte-exact HIP kernarg image. The device kernel takes the fields in this order, so the host can
// pass one standard-layout struct (the same convention as NormQuantArgs in the tree-WY caller).
struct Args {
    uint64_t conv_source = 0;  // [m][conv_stride] interleaved query/key/value
    uint64_t decay_source = 0; // [m][nv]
    uint64_t beta_source = 0;  // [m][nv] (already sigmoid-applied by the conv kernel)
    uint64_t query = 0;        // [m][nk][128] sealed plane (destination)
    uint64_t key = 0;          // [m][nk][128]
    uint64_t value = 0;        // [m][nv][128]
    uint64_t decay = 0;        // [m][nv]
    uint64_t beta = 0;         // [m][nv]
    uint64_t parent_source = 0; // [m] int32 staging (host chain uploaded once per cycle)
    uint64_t parent = 0;        // [m] int32 sealed plane
    uint32_t m = 0;             // sealed width: 5, 8 or 12
    uint32_t nk = kQueryHeads;
    uint32_t nv = kValueHeads;
    uint32_t conv_stride = kConvStride;
};
static_assert(std::is_standard_layout<Args>::value, "wire pack kernarg must be standard layout");
static_assert(sizeof(Args) == 96, "wire pack kernarg image is 96 bytes");
static_assert(offsetof(Args, m) == 80 && offsetof(Args, conv_stride) == 92,
              "wire pack kernarg offsets");

namespace detail {

inline bool checked_end(uint64_t base, uint64_t bytes, uint64_t& end) {
    if (base == 0) return false;
    if (base > UINT64_MAX - bytes) return false;
    end = base + bytes;
    return true;
}

inline bool overlaps(uint64_t a0, uint64_t a1, uint64_t b0, uint64_t b1) {
    return a0 < b1 && b0 < a1;
}

struct Span {
    const char* name;
    uint64_t begin;
    uint64_t end;
};

}  // namespace detail

// Validates every pointer, size and the sealed geometry before any device work. All buffers the
// kernel reads and writes must be non-null, 4-byte aligned and pairwise disjoint; a stale or
// aliased staging area is a pre-dispatch failure, never a partial copy.
inline bool validate(const Args& a, std::string& err) {
    err.clear();
    if (!is_sealed_width(a.m)) {
        err = "wire pack width must be 5, 8 or 12";
        return false;
    }
    if (a.nk != kQueryHeads || a.nv != kValueHeads) {
        err = "wire pack expects nk=8 query heads and nv=32 value heads";
        return false;
    }
    if (a.conv_stride < kConvStride) {
        err = "wire pack conv stride is smaller than the 6144-float conv row";
        return false;
    }
    const uint64_t conv_bytes =
        static_cast<uint64_t>(a.m) * static_cast<uint64_t>(a.conv_stride) * sizeof(float);
    const detail::Span spans[] = {
        {"conv source", a.conv_source, 0},
        {"decay source", a.decay_source, 0},
        {"beta source", a.beta_source, 0},
        {"query plane", a.query, 0},
        {"key plane", a.key, 0},
        {"value plane", a.value, 0},
        {"decay plane", a.decay, 0},
        {"beta plane", a.beta, 0},
        {"parent source", a.parent_source, 0},
        {"parent plane", a.parent, 0},
    };
    const uint64_t sizes[] = {
        conv_bytes,
        gate_bytes(a.m),
        gate_bytes(a.m),
        query_bytes(a.m),
        key_bytes(a.m),
        value_bytes(a.m),
        gate_bytes(a.m),
        gate_bytes(a.m),
        parent_bytes(a.m),
        parent_bytes(a.m),
    };
    constexpr size_t kSpans = sizeof(spans) / sizeof(spans[0]);
    detail::Span live[kSpans];
    for (size_t i = 0; i < kSpans; ++i) {
        if (!detail::checked_end(spans[i].begin, sizes[i], live[i].end)) {
            err = std::string("wire pack ") + spans[i].name + " pointer is null or overflows";
            return false;
        }
        if ((spans[i].begin & 3u) != 0u) {
            err = std::string("wire pack ") + spans[i].name + " is not 4-byte aligned";
            return false;
        }
        live[i].name = spans[i].name;
        live[i].begin = spans[i].begin;
    }
    for (size_t i = 0; i < kSpans; ++i) {
        for (size_t j = i + 1; j < kSpans; ++j) {
            if (detail::overlaps(live[i].begin, live[i].end, live[j].begin, live[j].end)) {
                err = std::string("wire pack ") + live[i].name + " overlaps " + live[j].name;
                return false;
            }
        }
    }
    return true;
}

// Host mirror of the device copy. Used by the CPU equivalence tests; production only enqueues
// the kernel. The caller must have validated Args first.
struct HostBuffers {
    const float* conv = nullptr;
    const float* decay = nullptr;
    const float* beta = nullptr;
    float* query = nullptr;
    float* key = nullptr;
    float* value = nullptr;
    float* decay_out = nullptr;
    float* beta_out = nullptr;
    const int32_t* parent = nullptr;
    int32_t* parent_out = nullptr;
};

inline bool pack_host(const Args& a, const HostBuffers& b, std::string& err) {
    err.clear();
    if (!b.conv || !b.decay || !b.beta || !b.query || !b.key || !b.value || !b.decay_out ||
        !b.beta_out || !b.parent || !b.parent_out) {
        err = "wire pack host buffers are incomplete";
        return false;
    }
    const uint32_t qk = a.nk * kHeadDim;
    const uint32_t vdim = a.nv * kHeadDim;
    for (uint32_t t = 0; t < a.m; ++t) {
        const float* src = b.conv + static_cast<uint64_t>(t) * a.conv_stride;
        float* q = b.query + static_cast<uint64_t>(t) * qk;
        float* k = b.key + static_cast<uint64_t>(t) * qk;
        float* v = b.value + static_cast<uint64_t>(t) * vdim;
        for (uint32_t i = 0; i < qk; ++i) {
            q[i] = src[i];
            k[i] = src[qk + i];
        }
        for (uint32_t i = 0; i < vdim; ++i) v[i] = src[2u * qk + i];
        for (uint32_t i = 0; i < a.nv; ++i) {
            b.decay_out[static_cast<uint64_t>(t) * a.nv + i] =
                b.decay[static_cast<uint64_t>(t) * a.nv + i];
            b.beta_out[static_cast<uint64_t>(t) * a.nv + i] =
                b.beta[static_cast<uint64_t>(t) * a.nv + i];
        }
        b.parent_out[t] = b.parent[t];
    }
    return true;
}

}  // namespace vegajet::gdn::wire_pack
