// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/quant_registry.hpp"

#include <cmath>
#include <cstring>
#include <limits>

namespace vegajet::quant {

namespace {

bool iq_packable(DiskQuantType t) {
    switch (t) {
        case DiskQuantType::IQ2_XXS: case DiskQuantType::IQ2_XS: case DiskQuantType::IQ2_S:
        case DiskQuantType::IQ3_XXS: case DiskQuantType::IQ3_S: case DiskQuantType::IQ4_XS:
        case DiskQuantType::IQ4_NL: case DiskQuantType::Q4_0: case DiskQuantType::Q2_0:
            return true;
        default:
            return false;
    }
}

void start(NativeTensor& out, NativeFormat f, DiskQuantType disk, uint32_t rows, uint32_t cols, bool lossless) {
    out = NativeTensor{};
    out.format = f;
    out.source = disk;
    out.rows = rows;
    out.cols = cols;
    out.lossless = lossless;
}

} // namespace

// ------------------------------------------------------------------------------------------------- converters
bool convert_requant_q8(DiskQuantType disk, const uint8_t* src, uint32_t rows, uint32_t cols, NativeTensor& out) {
    if (cols % 32) return false;
    start(out, NativeFormat::VJ_Q8_DP4A, disk, rows, cols, disk == DiskQuantType::Q8_0);
    out.q8 = repack_to_vj_q8(disk, src, rows, cols);  // Q8_0: bit-exact copy of codes and d
    return true;
}

bool convert_requant_q4_dot8(DiskQuantType disk, const uint8_t* src, uint32_t rows, uint32_t cols, NativeTensor& out) {
    if (cols % 32) return false;
    start(out, NativeFormat::VJ_Q4_DOT8, disk, rows, cols, false);
    out.q4 = repack_to_vj_q4_dot8(disk, src, rows, cols, false);
    build_q4_dot8_device_image(out);
    return true;
}

bool convert_q4_0_exact_dot8(DiskQuantType disk, const uint8_t* src, uint32_t rows, uint32_t cols, NativeTensor& out) {
    if (disk != DiskQuantType::Q4_0 || cols % 32) return false;
    start(out, NativeFormat::VJ_Q4_DOT8, disk, rows, cols, true);
    auto& p = out.q4;
    p.rows = rows;
    p.cols = cols;
    p.fwht64 = false;
    const size_t blocks = static_cast<size_t>(rows) * cols / 32;
    p.qs.assign(blocks * 4, 0u);
    p.ds.resize(blocks);
    for (size_t b = 0; b < blocks; ++b) {
        const uint8_t* blk = src + b * 18;  // block_q4_0: FP16 d, 16 bytes of nibbles
        std::memcpy(&p.ds[b], blk, 2);
        for (int j = 0; j < 32; ++j) {
            const uint8_t byte = blk[2 + (j & 15)];
            const int q = (j < 16 ? (byte & 0xF) : (byte >> 4)) - 8;          // [-8, 7]
            const uint32_t nib = static_cast<uint32_t>(q) & 0xFu;              // two's complement nibble
            p.qs[b * 4 + static_cast<size_t>(j / 8)] |= nib << ((j % 8) * 4);
        }
    }
    build_q4_dot8_device_image(out);
    return true;
}

bool convert_iq_packed(DiskQuantType disk, const uint8_t* src, uint32_t rows, uint32_t cols, NativeTensor& out) {
    if (!iq_packable(disk)) return false;
    start(out, NativeFormat::VJ_IQ_PACKED, disk, rows, cols, true);
    out.iq = repack_to_vj_iq(disk, src, rows, cols);
    return true;
}

void build_q4_dot8_device_image(NativeTensor& t) {
    const uint32_t qs_raw = t.cols / 2u, ds_raw = t.cols / 16u;
    t.qs_layout = select_tcc_odd256_layout(qs_raw);
    t.ds_layout = select_tcc_odd256_layout(ds_raw);
    const size_t qs_bytes = static_cast<size_t>(t.rows) * t.qs_layout.stride_bytes;
    t.device_image.assign(qs_bytes + static_cast<size_t>(t.rows) * t.ds_layout.stride_bytes, 0u);
    const auto* qs = reinterpret_cast<const uint8_t*>(t.q4.qs.data());
    const auto* ds = reinterpret_cast<const uint8_t*>(t.q4.ds.data());
    for (uint32_t r = 0; r < t.rows; ++r) {
        std::memcpy(t.device_image.data() + static_cast<size_t>(r) * t.qs_layout.stride_bytes,
                    qs + static_cast<size_t>(r) * qs_raw, qs_raw);
        std::memcpy(t.device_image.data() + qs_bytes + static_cast<size_t>(r) * t.ds_layout.stride_bytes,
                    ds + static_cast<size_t>(r) * ds_raw, ds_raw);
    }
}

uint64_t NativeTensor::device_bytes() const {
    switch (format) {
        case NativeFormat::VJ_Q8_DP4A: return q8.total_bytes();
        case NativeFormat::VJ_Q4_DOT8: return device_image.size();
        case NativeFormat::VJ_IQ_PACKED: return iq.total_bytes();
        default: return 0;
    }
}

uint64_t NativeTensor::payload_bytes() const {
    return format == NativeFormat::VJ_Q4_DOT8 ? q4.total_bytes() : device_bytes();
}

// ------------------------------------------------------------------------------------------------- registry
UnpackerRegistry& UnpackerRegistry::instance() {
    static UnpackerRegistry reg;
    return reg;
}

UnpackerRegistry::UnpackerRegistry() {
    using D = DiskQuantType;
    using N = NativeFormat;
    const UnpackerHook defaults[] = {
        {D::F16, N::VJ_Q8_DP4A, false, convert_requant_q8},
        {D::BF16, N::VJ_Q8_DP4A, false, convert_requant_q8},
        {D::F8_E4M3, N::VJ_Q8_DP4A, false, convert_requant_q8},
        {D::Q8_0, N::VJ_Q8_DP4A, true, convert_requant_q8},
        {D::Q6_K, N::VJ_Q8_DP4A, false, convert_requant_q8},
        {D::Q5_K, N::VJ_Q8_DP4A, false, convert_requant_q8},
        {D::Q4_0, N::VJ_Q4_DOT8, true, convert_q4_0_exact_dot8},
        // Affine 4-bit (d * q + m): a symmetric nibble cannot hold the offset without requantization noise
        // (cosine ~0.997 measured, below the 0.999 bar), so these map to int8 by default; the planner may keep the
        // disk encoding instead when the crossover law rejects the larger format.
        {D::Q4_1, N::VJ_Q8_DP4A, false, convert_requant_q8},
        {D::Q4_K, N::VJ_Q8_DP4A, false, convert_requant_q8},
        {D::IQ4_NL, N::VJ_IQ_PACKED, true, convert_iq_packed},
        {D::IQ4_XS, N::VJ_IQ_PACKED, true, convert_iq_packed},
        {D::IQ3_XXS, N::VJ_IQ_PACKED, true, convert_iq_packed},
        {D::IQ2_S, N::VJ_IQ_PACKED, true, convert_iq_packed},
    };
    for (const auto& h : defaults) add(h);
}

bool UnpackerRegistry::add(const UnpackerHook& h) {
    const auto id = static_cast<size_t>(h.disk);
    if (!h.convert || id >= hooks_.size() || !DecoderRegistry::instance().supports(h.disk)) return false;
    std::lock_guard<std::mutex> lk(mu_);
    if (present_[id]) return false;
    hooks_[id] = h;
    present_[id] = true;
    return true;
}

const UnpackerHook* UnpackerRegistry::find(DiskQuantType t) const {
    const auto id = static_cast<size_t>(t);
    std::lock_guard<std::mutex> lk(mu_);
    return id < hooks_.size() && present_[id] ? &hooks_[id] : nullptr;
}

std::vector<UnpackerHook> UnpackerRegistry::hooks() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<UnpackerHook> out;
    for (size_t i = 0; i < hooks_.size(); ++i)
        if (present_[i]) out.push_back(hooks_[i]);
    return out;
}

bool UnpackerRegistry::convert(DiskQuantType t, const void* src, size_t src_bytes, uint32_t rows, uint32_t cols,
                               NativeTensor& out, std::string* err) const {
    auto fail = [&](const char* m) {
        if (err) *err = std::string(ggml_type_name(static_cast<uint32_t>(t))) + ": " + m;
        return false;
    };
    const UnpackerHook* h = find(t);
    if (!h) return fail("no unpacker hook");
    const FormatInfo* fi = DecoderRegistry::instance().find(t);
    if (!fi || !fi->block_elems) return fail("no CPU decoder");
    if (rows == 0 || cols == 0) return fail("empty tensor");
    if (cols % fi->block_elems || cols % 32) return fail("columns not block aligned");
    const uint64_t rb = DecoderRegistry::instance().row_bytes(t, cols);
    if (rb == 0 || rb > std::numeric_limits<size_t>::max() / rows) return fail("row size overflow");
    if (!src || src_bytes != rb * rows) return fail("source size does not match rows x row bytes");
    if (!h->convert(t, static_cast<const uint8_t*>(src), rows, cols, out)) return fail("converter rejected input");
    return true;
}

// ------------------------------------------------------------------------------------------------- dequant + metrics
bool dequantize_native(const NativeTensor& t, std::vector<float>& out) {
    out.assign(static_cast<size_t>(t.rows) * t.cols, 0.0f);
    switch (t.format) {
        case NativeFormat::VJ_Q8_DP4A:
            if (t.q8.qs.size() != out.size()) return false;
            dequantize_vj_q8(t.q8, out.data());
            return true;
        case NativeFormat::VJ_Q4_DOT8: {
            const uint32_t qs_raw = t.cols / 2u, ds_raw = t.cols / 16u;
            const size_t qs_bytes = static_cast<size_t>(t.rows) * t.qs_layout.stride_bytes;
            if (t.qs_layout.stride_bytes < qs_raw || t.ds_layout.stride_bytes < ds_raw ||
                t.device_image.size() != qs_bytes + static_cast<size_t>(t.rows) * t.ds_layout.stride_bytes)
                return false;
            PackedVjQ4Dot8 p;  // read back through the skewed device image, not the host copy
            p.rows = t.rows;
            p.cols = t.cols;
            p.qs.resize(static_cast<size_t>(t.rows) * t.cols / 8);
            p.ds.resize(static_cast<size_t>(t.rows) * t.cols / 32);
            auto* qs = reinterpret_cast<uint8_t*>(p.qs.data());
            auto* ds = reinterpret_cast<uint8_t*>(p.ds.data());
            for (uint32_t r = 0; r < t.rows; ++r) {
                std::memcpy(qs + static_cast<size_t>(r) * qs_raw,
                            t.device_image.data() + static_cast<size_t>(r) * t.qs_layout.stride_bytes, qs_raw);
                std::memcpy(ds + static_cast<size_t>(r) * ds_raw,
                            t.device_image.data() + qs_bytes + static_cast<size_t>(r) * t.ds_layout.stride_bytes,
                            ds_raw);
            }
            dequantize_vj_q4_dot8(p, out.data());
            return true;
        }
        case NativeFormat::VJ_IQ_PACKED:
            dequantize_vj_iq(t.iq, out.data());
            return true;
        default:
            return false;
    }
}

double disk_bits_per_weight(DiskQuantType t) {
    const GgmlTypeTraits tr = ggml_type_traits(static_cast<uint32_t>(t));
    return tr.block_elems ? 8.0 * tr.block_bytes / tr.block_elems : 0.0;
}

double native_bits_per_weight(NativeFormat f, DiskQuantType disk) {
    switch (f) {
        case NativeFormat::VJ_Q8_DP4A: return 8.5;
        case NativeFormat::VJ_Q4_DOT8: return 4.5;
        case NativeFormat::VJ_Q4_A8: return 5.0;
        case NativeFormat::VJ_IQ_PACKED: return disk_bits_per_weight(disk);
        case NativeFormat::VJ_Q3_DOT8: return 3.5;   // 3 bit + fp16 (d, z) per 64-element group
        case NativeFormat::VJ_Q2_DOT8: return 2.5;   // 2 bit + fp16 (d, z) per 64-element group
    }
    return 0.0;
}

double cosine_similarity(const float* a, const float* b, size_t n) {
    double ab = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < n; ++i) {
        ab += static_cast<double>(a[i]) * b[i];
        aa += static_cast<double>(a[i]) * a[i];
        bb += static_cast<double>(b[i]) * b[i];
    }
    if (aa == 0.0 && bb == 0.0) return 1.0;
    if (aa == 0.0 || bb == 0.0) return 0.0;
    return ab / std::sqrt(aa * bb);
}

} // namespace vegajet::quant
