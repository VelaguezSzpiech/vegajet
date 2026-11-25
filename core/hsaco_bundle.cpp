// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/hsaco_bundle.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <unordered_set>
#include <utility>

namespace vegajet::bundle {

// ---------------------------------------------------------------------------------------------------------
// SHA-256 (FIPS 180-4), standalone: no OpenSSL / libcrypto dependency in this gate.
// ---------------------------------------------------------------------------------------------------------
namespace {

constexpr uint32_t kSha256K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

inline uint32_t rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32u - n)); }

void sha256_compress(uint32_t h[8], const uint8_t block[64]) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = (uint32_t(block[4 * i]) << 24) | (uint32_t(block[4 * i + 1]) << 16) |
               (uint32_t(block[4 * i + 2]) << 8) | uint32_t(block[4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const uint32_t ch = (e & f) ^ (~e & g);
        const uint32_t t1 = hh + s1 + ch + kSha256K[i] + w[i];
        const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t t2 = s0 + maj;
        hh = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += hh;
}

uint32_t rd16(const uint8_t* p) { return uint32_t(p[0]) | (uint32_t(p[1]) << 8); }
uint32_t rd32(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
uint64_t rd64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}
uint64_t be(const uint8_t* p, size_t n) {
    uint64_t v = 0;
    for (size_t i = 0; i < n; ++i) v = (v << 8) | p[i];
    return v;
}

int hex_nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

bool parse_u64_dec(std::string_view s, uint64_t& out) {
    if (s.empty()) return false;
    uint64_t v = 0;
    for (char c : s) {
        if (c < '0' || c > '9') return false;
        const uint64_t d = uint64_t(c - '0');
        if (v > (UINT64_MAX - d) / 10u) return false;
        v = v * 10u + d;
    }
    out = v;
    return true;
}

bool read_file(const std::string& path, uint64_t cap, std::vector<uint8_t>& out, std::string* err) {
    out.clear();
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        if (err) *err = "cannot open " + path;
        return false;
    }
    f.seekg(0, std::ios::end);
    const std::streamoff len = f.tellg();
    if (len < 0) {
        if (err) *err = "cannot size " + path;
        return false;
    }
    if (uint64_t(len) > cap) {
        if (err) *err = path + ": " + std::to_string(len) + " B exceeds the " + std::to_string(cap) + " B cap";
        return false;
    }
    f.seekg(0, std::ios::beg);
    out.resize(size_t(len));
    if (len != 0 && !f.read(reinterpret_cast<char*>(out.data()), len)) {
        if (err) *err = "short read on " + path;
        out.clear();
        return false;
    }
    if (!f && len != 0) {
        if (err) *err = "read failure on " + path;
        out.clear();
        return false;
    }
    return true;
}

std::string join_path(const std::string& root, const std::string& rel) {
    if (root.empty()) return rel;
    if (root.back() == '/') return root + rel;
    return root + "/" + rel;
}

// ---------------------------------------------------------------------------------------------------------
// Strict MessagePack reader for the NT_AMDGPU_METADATA note.
// ---------------------------------------------------------------------------------------------------------
struct MsgPack {
    const uint8_t* d = nullptr;
    size_t n = 0;
    size_t pos = 0;
    std::string err{};

    bool fail(const std::string& msg) {
        if (err.empty()) err = msg + " (byte offset " + std::to_string(pos) + ")";
        return false;
    }
    bool adv(uint64_t k) {
        if (k > uint64_t(n - pos)) return fail("truncated MessagePack");
        pos += size_t(k);
        return true;
    }
    bool u8(uint8_t& v) {
        if (pos >= n) return fail("truncated MessagePack");
        v = d[pos++];
        return true;
    }
    bool be_val(size_t bytes, uint64_t& v) {
        if (bytes == 0 || bytes > 8) return fail("bad MessagePack integer width");
        if (uint64_t(n - pos) < uint64_t(bytes)) return fail("truncated MessagePack");
        v = be(d + pos, bytes);
        pos += bytes;
        return true;
    }
    bool str(std::string& out) {
        uint8_t b = 0;
        if (!u8(b)) return false;
        uint64_t len = 0;
        if ((b & 0xe0u) == 0xa0u) {
            len = b & 0x1fu;
        } else if (b == 0xd9u) {
            if (!be_val(1, len)) return false;
        } else if (b == 0xdau) {
            if (!be_val(2, len)) return false;
        } else if (b == 0xdbu) {
            if (!be_val(4, len)) return false;
        } else {
            return fail("expected a MessagePack string");
        }
        if (len > uint64_t(n - pos)) return fail("truncated MessagePack string");
        if (len > (1ull << 20)) return fail("implausible MessagePack string length");
        out.assign(reinterpret_cast<const char*>(d + pos), size_t(len));
        pos += size_t(len);
        return true;
    }
    bool map_len(size_t& len) {
        uint8_t b = 0;
        if (!u8(b)) return false;
        uint64_t v = 0;
        if ((b & 0xf0u) == 0x80u) {
            v = b & 0x0fu;
        } else if (b == 0xdeu) {
            if (!be_val(2, v)) return false;
        } else if (b == 0xdfu) {
            if (!be_val(4, v)) return false;
        } else {
            return fail("expected a MessagePack map");
        }
        if (v > (1ull << 16)) return fail("implausible MessagePack map length");
        len = size_t(v);
        return true;
    }
    bool arr_len(size_t& len) {
        uint8_t b = 0;
        if (!u8(b)) return false;
        uint64_t v = 0;
        if ((b & 0xf0u) == 0x90u) {
            v = b & 0x0fu;
        } else if (b == 0xdcu) {
            if (!be_val(2, v)) return false;
        } else if (b == 0xddu) {
            if (!be_val(4, v)) return false;
        } else {
            return fail("expected a MessagePack array");
        }
        if (v > (1ull << 20)) return fail("implausible MessagePack array length");
        len = size_t(v);
        return true;
    }
    // Unsigned integer; signed encodings are accepted only when non-negative.
    bool number(uint64_t& out) {
        uint8_t b = 0;
        if (!u8(b)) return false;
        if (b < 0x80u) {
            out = b;
            return true;
        }
        switch (b) {
            case 0xccu: {
                uint64_t v = 0;
                if (!be_val(1, v)) return false;
                out = v;
                return true;
            }
            case 0xcdu: {
                uint64_t v = 0;
                if (!be_val(2, v)) return false;
                out = v;
                return true;
            }
            case 0xceu: {
                uint64_t v = 0;
                if (!be_val(4, v)) return false;
                out = v;
                return true;
            }
            case 0xcfu: {
                uint64_t v = 0;
                if (!be_val(8, v)) return false;
                out = v;
                return true;
            }
            case 0xd0u:
            case 0xd1u:
            case 0xd2u:
            case 0xd3u: {
                const size_t width = size_t(b) - 0xd0u + 1u;
                uint64_t raw = 0;
                if (!be_val(width, raw)) return false;
                const uint64_t sign = 1ull << (width * 8u - 1u);
                if ((raw & sign) != 0) return fail("negative integer where a non-negative value is required");
                out = raw;
                return true;
            }
            default:
                return fail("expected a MessagePack integer");
        }
    }
    bool skip(unsigned depth) {
        if (depth > 64) return fail("MessagePack nesting too deep");
        uint8_t b = 0;
        if (!u8(b)) return false;
        if (b < 0x80u || b >= 0xe0u) return true;  // fixint
        if ((b & 0xe0u) == 0xa0u) return adv(b & 0x1fu);  // fixstr
        if ((b & 0xf0u) == 0x90u) {  // fixarray
            const size_t cnt = b & 0x0fu;
            for (size_t i = 0; i < cnt; ++i)
                if (!skip(depth + 1)) return false;
            return true;
        }
        if ((b & 0xf0u) == 0x80u) {  // fixmap
            const size_t cnt = b & 0x0fu;
            for (size_t i = 0; i < cnt; ++i) {
                if (!skip(depth + 1) || !skip(depth + 1)) return false;
            }
            return true;
        }
        switch (b) {
            case 0xc0u:
            case 0xc2u:
            case 0xc3u:
                return true;
            case 0xc4u: {
                uint64_t v = 0;
                return be_val(1, v) && adv(v);
            }
            case 0xc5u: {
                uint64_t v = 0;
                return be_val(2, v) && adv(v);
            }
            case 0xc6u: {
                uint64_t v = 0;
                return be_val(4, v) && adv(v);
            }
            case 0xc7u: {  // ext8
                uint64_t v = 0;
                return be_val(1, v) && adv(1) && adv(v);
            }
            case 0xc8u: {  // ext16
                uint64_t v = 0;
                return be_val(2, v) && adv(1) && adv(v);
            }
            case 0xc9u: {  // ext32
                uint64_t v = 0;
                return be_val(4, v) && adv(1) && adv(v);
            }
            case 0xcau:
            case 0xceu:
            case 0xd2u:
                return adv(4);
            case 0xcbu:
            case 0xcfu:
            case 0xd3u:
                return adv(8);
            case 0xccu:
            case 0xd0u:
                return adv(1);
            case 0xcdu:
            case 0xd1u:
                return adv(2);
            case 0xd4u:
                return adv(1 + 1);
            case 0xd5u:
                return adv(1 + 2);
            case 0xd6u:
                return adv(1 + 4);
            case 0xd7u:
                return adv(1 + 8);
            case 0xd8u:
                return adv(1 + 16);
            case 0xd9u: {
                uint64_t v = 0;
                return be_val(1, v) && adv(v);
            }
            case 0xdau: {
                uint64_t v = 0;
                return be_val(2, v) && adv(v);
            }
            case 0xdbu: {
                uint64_t v = 0;
                return be_val(4, v) && adv(v);
            }
            case 0xdcu: {
                uint64_t v = 0;
                if (!be_val(2, v)) return false;
                for (uint64_t i = 0; i < v; ++i)
                    if (!skip(depth + 1)) return false;
                return true;
            }
            case 0xddu: {
                uint64_t v = 0;
                if (!be_val(4, v)) return false;
                if (v > (1ull << 20)) return fail("implausible MessagePack array length");
                for (uint64_t i = 0; i < v; ++i)
                    if (!skip(depth + 1)) return false;
                return true;
            }
            case 0xdeu: {
                uint64_t v = 0;
                if (!be_val(2, v)) return false;
                for (uint64_t i = 0; i < v; ++i) {
                    if (!skip(depth + 1) || !skip(depth + 1)) return false;
                }
                return true;
            }
            case 0xdfu: {
                uint64_t v = 0;
                if (!be_val(4, v)) return false;
                if (v > (1ull << 20)) return fail("implausible MessagePack map length");
                for (uint64_t i = 0; i < v; ++i) {
                    if (!skip(depth + 1) || !skip(depth + 1)) return false;
                }
                return true;
            }
            default:
                return fail("reserved or unsupported MessagePack type");
        }
    }
};

struct KernelFields {
    std::string name, symbol;
    bool has_name = false, has_symbol = false;
    uint64_t vgpr = 0, sgpr = 0, group = 0, priv = 0, kernarg = 0, wavefront = 0, max_wg = 0;
    bool has_vgpr = false, has_sgpr = false, has_group = false, has_priv = false, has_kernarg = false,
         has_wavefront = false, has_max_wg = false;
};

uint32_t clamp_u32(uint64_t v, const std::string& kernel, const char* field,
                   std::vector<std::string>& problems, bool& ok) {
    if (v > UINT32_MAX) {
        problems.push_back("kernel '" + kernel + "': " + field + " does not fit in 32 bits");
        ok = false;
        return 0;
    }
    return static_cast<uint32_t>(v);
}

bool parse_kernel(MsgPack& r, size_t depth, KernelInfo& out, std::vector<std::string>& problems) {
    size_t fields = 0;
    if (!r.map_len(fields)) return false;
    KernelFields k;
    std::unordered_set<std::string> seen;
    for (size_t i = 0; i < fields; ++i) {
        std::string key;
        if (!r.str(key)) return false;
        if (!seen.insert(key).second) {
            problems.push_back("kernel metadata repeats field '" + key + "'");
            return false;
        }
        if (key == ".name") {
            if (!r.str(k.name)) return false;
            k.has_name = true;
        } else if (key == ".symbol") {
            if (!r.str(k.symbol)) return false;
            k.has_symbol = true;
        } else if (key == ".vgpr_count") {
            if (!r.number(k.vgpr)) return false;
            k.has_vgpr = true;
        } else if (key == ".sgpr_count") {
            if (!r.number(k.sgpr)) return false;
            k.has_sgpr = true;
        } else if (key == ".group_segment_fixed_size") {
            if (!r.number(k.group)) return false;
            k.has_group = true;
        } else if (key == ".private_segment_fixed_size") {
            if (!r.number(k.priv)) return false;
            k.has_priv = true;
        } else if (key == ".kernarg_segment_size") {
            if (!r.number(k.kernarg)) return false;
            k.has_kernarg = true;
        } else if (key == ".wavefront_size") {
            if (!r.number(k.wavefront)) return false;
            k.has_wavefront = true;
        } else if (key == ".max_flat_workgroup_size") {
            if (!r.number(k.max_wg)) return false;
            k.has_max_wg = true;
        } else if (!r.skip(depth + 1)) {
            return false;
        }
    }

    if (k.has_name && k.name.empty()) {
        problems.push_back("kernel metadata has an empty .name");
        return false;
    }
    if (k.has_name && k.has_symbol && k.symbol != k.name + ".kd") {
        problems.push_back("kernel '" + k.name + "': .symbol '" + k.symbol + "' does not match .name (expected '" +
                           k.name + ".kd')");
        return false;
    }
    if (!k.has_name) {
        const std::string suffix = ".kd";
        if (!k.has_symbol || k.symbol.size() <= suffix.size() ||
            k.symbol.compare(k.symbol.size() - suffix.size(), suffix.size(), suffix) != 0) {
            problems.push_back("kernel metadata has neither .name nor a .symbol ending in '.kd'");
            return false;
        }
        k.name = k.symbol.substr(0, k.symbol.size() - suffix.size());
    }
    const char* required[6] = {"vgpr_count", "sgpr_count", "group_segment_fixed_size",
                               "private_segment_fixed_size", "kernarg_segment_size", "wavefront_size"};
    const bool present[6] = {k.has_vgpr, k.has_sgpr, k.has_group, k.has_priv, k.has_kernarg, k.has_wavefront};
    for (int i = 0; i < 6; ++i) {
        if (!present[i]) {
            problems.push_back("kernel '" + k.name + "': missing ." + required[i]);
            return false;
        }
    }
    bool ok = true;
    out = KernelInfo{};
    out.name = k.name;
    out.vgpr_count = clamp_u32(k.vgpr, k.name, ".vgpr_count", problems, ok);
    out.sgpr_count = clamp_u32(k.sgpr, k.name, ".sgpr_count", problems, ok);
    out.group_segment_fixed_size = clamp_u32(k.group, k.name, ".group_segment_fixed_size", problems, ok);
    out.private_segment_fixed_size = clamp_u32(k.priv, k.name, ".private_segment_fixed_size", problems, ok);
    out.kernarg_segment_size = clamp_u32(k.kernarg, k.name, ".kernarg_segment_size", problems, ok);
    out.wavefront_size = clamp_u32(k.wavefront, k.name, ".wavefront_size", problems, ok);
    out.max_flat_workgroup_size = clamp_u32(k.max_wg, k.name, ".max_flat_workgroup_size", problems, ok);
    return ok;
}

bool metadata_target_is_gfx906(const std::string& target) {
    const std::string prefix = "amdgcn-amd-amdhsa--gfx906";
    if (target.compare(0, prefix.size(), prefix) != 0) return false;
    return target.size() == prefix.size() || target[prefix.size()] == ':';
}

bool parse_metadata(const uint8_t* d, size_t n, CodeObjectReport& co) {
    MsgPack r{d, n};
    size_t top = 0;
    if (!r.map_len(top)) {
        co.error = "metadata is not a MessagePack map: " + r.err;
        return false;
    }
    bool have_target = false, have_kernels = false;
    std::unordered_set<std::string> top_seen;
    std::vector<std::string> problems;
    for (size_t i = 0; i < top; ++i) {
        std::string key;
        if (!r.str(key)) {
            co.error = "malformed metadata key: " + r.err;
            return false;
        }
        if (!top_seen.insert(key).second) {
            co.error = "metadata repeats top-level key '" + key + "'";
            return false;
        }
        if (key == "amdhsa.target") {
            if (have_target) {
                co.error = "metadata has more than one amdhsa.target";
                return false;
            }
            if (!r.str(co.target)) {
                co.error = "amdhsa.target is not a string: " + r.err;
                return false;
            }
            have_target = true;
        } else if (key == "amdhsa.kernels") {
            if (have_kernels) {
                co.error = "metadata has more than one amdhsa.kernels";
                return false;
            }
            size_t count = 0;
            if (!r.arr_len(count)) {
                co.error = "amdhsa.kernels is not an array: " + r.err;
                return false;
            }
            if (count == 0) {
                co.error = "amdhsa.kernels is empty";
                return false;
            }
            if (count > 4096) {
                co.error = "amdhsa.kernels declares an implausible number of kernels";
                return false;
            }
            for (size_t k = 0; k < count; ++k) {
                KernelInfo info;
                if (!parse_kernel(r, 1, info, problems)) {
                    co.error = r.err.empty() ? "malformed amdhsa.kernels entry" : r.err;
                    co.problems = problems;
                    return false;
                }
                for (const auto& existing : co.kernels) {
                    if (existing.name == info.name) {
                        co.error = "amdhsa.kernels repeats kernel '" + info.name + "'";
                        co.problems = problems;
                        return false;
                    }
                }
                co.kernels.push_back(std::move(info));
            }
            have_kernels = true;
        } else if (!r.skip(1)) {
            co.error = "malformed metadata value for '" + key + "': " + r.err;
            return false;
        }
    }
    if (!r.err.empty()) {
        co.error = r.err;
        return false;
    }
    if (r.pos != n) {
        co.error = "trailing bytes after the metadata map (" + std::to_string(n - r.pos) + " B)";
        return false;
    }
    if (!have_target) {
        co.error = "metadata has no amdhsa.target";
        return false;
    }
    if (!have_kernels) {
        co.error = "metadata has no amdhsa.kernels";
        return false;
    }
    if (!metadata_target_is_gfx906(co.target)) {
        co.error = "amdhsa.target '" + co.target + "' is not a gfx906 target";
        return false;
    }
    co.problems = std::move(problems);
    return co.problems.empty();
}

// Parses one ELF note section body (SHT_NOTE contents).
bool parse_note_section(const uint8_t* d, size_t n, bool& found, CodeObjectReport& co) {
    size_t off = 0;
    while (off + 12 <= n) {
        const uint32_t namesz = rd32(d + off);
        const uint32_t descsz = rd32(d + off + 4);
        const uint32_t type = rd32(d + off + 8);
        const uint64_t name_off = uint64_t(off) + 12;
        const uint64_t name_pad = (uint64_t(namesz) + 3u) & ~uint64_t(3u);
        const uint64_t desc_off = name_off + name_pad;
        const uint64_t desc_pad = (uint64_t(descsz) + 3u) & ~uint64_t(3u);
        const uint64_t next = desc_off + desc_pad;
        if (desc_off > n || uint64_t(descsz) > uint64_t(n) - desc_off || next > n) {
            co.error = "truncated or overlapping ELF note";
            return false;
        }
        const bool amdgpu_name = namesz == 7 && std::memcmp(d + name_off, "AMDGPU", 6) == 0 &&
                                 d[name_off + 6] == 0;
        if (type == 32u) {  // NT_AMDGPU_METADATA
            if (!amdgpu_name) {
                co.error = "NT_AMDGPU_METADATA note has a malformed owner name";
                return false;
            }
            if (found) {
                co.error = "duplicate NT_AMDGPU_METADATA note";
                return false;
            }
            found = true;
            if (!parse_metadata(d + desc_off, descsz, co)) return false;
        }
        off = size_t(next);
    }
    // Note sections may be padded to their section alignment; only zero padding is acceptable.
    for (size_t i = off; i < n; ++i) {
        if (d[i] != 0) {
            co.error = "trailing non-zero bytes in an ELF note section";
            return false;
        }
    }
    return true;
}

} // namespace

Digest sha256(const uint8_t* data, size_t size) {
    uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                     0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    size_t off = 0;
    for (; size - off >= 64; off += 64) sha256_compress(h, data + off);
    uint8_t tail[128] = {0};
    const size_t rem = size - off;
    if (rem != 0) std::memcpy(tail, data + off, rem);
    tail[rem] = 0x80;
    const size_t tail_len = rem < 56 ? 64 : 128;
    const uint64_t bits = uint64_t(size) * 8u;
    for (int i = 0; i < 8; ++i) tail[tail_len - 1 - size_t(i)] = uint8_t(bits >> (8u * unsigned(i)));
    sha256_compress(h, tail);
    if (tail_len == 128) sha256_compress(h, tail + 64);
    Digest out{};
    for (int i = 0; i < 8; ++i) {
        out[size_t(4 * i)] = uint8_t(h[i] >> 24);
        out[size_t(4 * i + 1)] = uint8_t(h[i] >> 16);
        out[size_t(4 * i + 2)] = uint8_t(h[i] >> 8);
        out[size_t(4 * i + 3)] = uint8_t(h[i]);
    }
    return out;
}

std::string to_hex(const Digest& digest) {
    static const char* kHex = "0123456789abcdef";
    std::string out;
    out.reserve(kSha256Bytes * 2);
    for (uint8_t b : digest) {
        out.push_back(kHex[b >> 4]);
        out.push_back(kHex[b & 0x0fu]);
    }
    return out;
}

bool digest_from_hex(std::string_view hex, Digest& out) {
    if (hex.size() != kSha256Bytes * 2) return false;
    for (size_t i = 0; i < kSha256Bytes; ++i) {
        const int hi = hex_nibble(hex[2 * i]);
        const int lo = hex_nibble(hex[2 * i + 1]);
        if (hi < 0 || lo < 0) return false;
        out[i] = uint8_t((hi << 4) | lo);
    }
    return true;
}

bool sanitize_relative_path(std::string_view in, std::string& out, std::string* why) {
    out.clear();
    auto reject = [&](const std::string& msg) {
        if (why) *why = msg;
        return false;
    };
    if (in.empty()) return reject("path is empty");
    if (in.size() > 1024) return reject("path is longer than 1024 bytes");
    if (in.front() == '/') return reject("path is absolute");
    if (in.back() == '/') return reject("path ends with '/'");
    size_t comp_start = 0;
    for (size_t i = 0; i <= in.size(); ++i) {
        if (i == in.size() || in[i] == '/') {
            const std::string_view comp = in.substr(comp_start, i - comp_start);
            if (comp.empty()) return reject("path has an empty component");
            if (comp == "." || comp == "..") return reject("path has a '" + std::string(comp) + "' component");
            comp_start = i + 1;
            continue;
        }
        const unsigned char c = static_cast<unsigned char>(in[i]);
        if (c == '\\') return reject("path contains a backslash");
        if (c <= 0x20u || c >= 0x7fu) return reject("path contains a space, control byte or non-ASCII byte");
    }
    out.assign(in.begin(), in.end());
    return true;
}

bool parse_manifest(std::string_view text, Manifest& out, std::string* err) {
    out = Manifest{};
    std::unordered_set<std::string> seen;
    size_t start = 0;
    size_t line_no = 0;
    while (start <= text.size()) {
        size_t end = text.find('\n', start);
        if (end == std::string_view::npos) end = text.size();
        std::string_view line = text.substr(start, end - start);
        ++line_no;
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        size_t i = 0;
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
        if (i == line.size() || line[i] == '#') {
            if (end == text.size()) break;
            start = end + 1;
            continue;
        }
        std::string_view fields[3];
        size_t n_fields = 0;
        while (i < line.size()) {
            while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
            if (i == line.size()) break;
            const size_t begin = i;
            while (i < line.size() && line[i] != ' ' && line[i] != '\t') ++i;
            if (n_fields == 3) {
                if (err) *err = "manifest line " + std::to_string(line_no) + ": more than three fields";
                return false;
            }
            fields[n_fields++] = line.substr(begin, i - begin);
        }
        if (n_fields != 3) {
            if (err) *err = "manifest line " + std::to_string(line_no) + ": expected '<sha256> <bytes> <path>'";
            return false;
        }
        ManifestEntry entry;
        if (!digest_from_hex(fields[0], entry.digest)) {
            if (err) *err = "manifest line " + std::to_string(line_no) + ": invalid SHA-256 hex";
            return false;
        }
        entry.digest_hex = to_hex(entry.digest);
        if (!parse_u64_dec(fields[1], entry.size)) {
            if (err) *err = "manifest line " + std::to_string(line_no) + ": invalid byte count";
            return false;
        }
        if (entry.size > kMaxBundleFileBytes) {
            if (err) *err = "manifest line " + std::to_string(line_no) + ": size exceeds the bundle file cap";
            return false;
        }
        std::string why;
        if (!sanitize_relative_path(fields[2], entry.path, &why)) {
            if (err) *err = "manifest line " + std::to_string(line_no) + ": " + why;
            return false;
        }
        if (!seen.insert(entry.path).second) {
            if (err) *err = "manifest line " + std::to_string(line_no) + ": duplicate path '" + entry.path + "'";
            return false;
        }
        out.entries.push_back(std::move(entry));
        if (end == text.size()) break;
        start = end + 1;
    }
    if (out.entries.empty()) {
        if (err) *err = "manifest has no entries";
        return false;
    }
    return true;
}

bool load_manifest(const std::string& path, Manifest& out, std::string* err) {
    std::vector<uint8_t> bytes;
    if (!read_file(path, 1ull << 20, bytes, err)) return false;
    const std::string_view text(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    return parse_manifest(text, out, err);
}

CodeObjectReport inspect_code_object(const uint8_t* data, size_t size, const Gfx906Budget& budget) {
    CodeObjectReport co;
    if (!data || size < 64) {
        co.error = "not an ELF64 file (shorter than 64 bytes)";
        return co;
    }
    static const uint8_t kMagic[4] = {0x7f, 'E', 'L', 'F'};
    if (std::memcmp(data, kMagic, 4) != 0) {
        co.error = "not an ELF file";
        return co;
    }
    if (data[4] != 2 || data[5] != 1 || data[6] != 1) {
        co.error = "expected a little-endian ELF64 object with identification version 1";
        return co;
    }
    if (rd16(data + 16) != 3) {
        co.error = "ELF type is not ET_DYN";
        return co;
    }
    if (rd16(data + 18) != 224) {
        co.error = "ELF machine is not EM_AMDGPU (224)";
        return co;
    }
    if (rd32(data + 20) != 1) {
        co.error = "ELF version is not 1";
        return co;
    }
    const uint32_t e_flags = rd32(data + 48);
    if ((e_flags & 0xffu) != 0x2fu) {  // EF_AMDGPU_MACH_AMDGCN_GFX906
        co.error = "e_flags machine is not gfx906";
        return co;
    }
    if (rd16(data + 52) != 64) {
        co.error = "ELF header size is not 64";
        return co;
    }
    if (rd16(data + 58) != 64) {
        co.error = "section header entry size is not 64";
        return co;
    }
    const uint64_t shoff = rd64(data + 40);
    uint64_t shnum = rd16(data + 60);
    if (shoff == 0) {
        co.error = "ELF has no section headers";
        return co;
    }
    if (shoff > size || (size - size_t(shoff)) / 64 < 1) {
        co.error = "section header table is out of bounds";
        return co;
    }
    if (shnum == 0) {  // extended numbering: section 0 sh_size holds the real count
        shnum = rd64(data + size_t(shoff) + 32);
        if (shnum == 0 || shnum > 65535) {
            co.error = "ELF extended section count is invalid";
            return co;
        }
    }
    if (shnum > (size - size_t(shoff)) / 64) {
        co.error = "section header table is out of bounds";
        return co;
    }
    bool found = false;
    for (uint64_t i = 0; i < shnum; ++i) {
        const uint8_t* sh = data + size_t(shoff) + size_t(i) * 64;
        const uint32_t type = rd32(sh + 4);
        if (type != 7) continue;  // SHT_NOTE
        const uint64_t off = rd64(sh + 24);
        const uint64_t len = rd64(sh + 32);
        if (off > size || len > size - size_t(off)) {
            co.error = "note section is out of bounds";
            return co;
        }
        if (!parse_note_section(data + size_t(off), size_t(len), found, co)) return co;
    }
    if (!found) {
        co.error = "no NT_AMDGPU_METADATA note";
        return co;
    }

    for (auto& k : co.kernels) {
        auto fail = [&](const std::string& msg) { k.problems.push_back("kernel '" + k.name + "': " + msg); };
        if (k.private_segment_fixed_size > budget.max_private_bytes) {
            fail("private_segment_fixed_size=" + std::to_string(k.private_segment_fixed_size) +
                 " B (spill/scratch must be " + std::to_string(budget.max_private_bytes) + " B)");
        }
        if (k.vgpr_count > budget.max_vgpr) {
            fail("vgpr_count=" + std::to_string(k.vgpr_count) + " exceeds the gfx906 budget " +
                 std::to_string(budget.max_vgpr));
        }
        if (k.group_segment_fixed_size > budget.max_lds_bytes) {
            fail("group_segment_fixed_size=" + std::to_string(k.group_segment_fixed_size) +
                 " B exceeds the gfx906 LDS budget " + std::to_string(budget.max_lds_bytes) + " B");
        }
        if (k.wavefront_size != budget.wavefront_size) {
            fail("wavefront_size=" + std::to_string(k.wavefront_size) + " (gfx906 requires " +
                 std::to_string(budget.wavefront_size) + ")");
        }
        k.within_budget = k.problems.empty();
        if (!k.within_budget) co.problems.insert(co.problems.end(), k.problems.begin(), k.problems.end());
    }
    if (co.error.empty() && co.kernels.empty()) co.error = "amdhsa.kernels is empty";
    co.ok = co.error.empty() && co.problems.empty();
    return co;
}

CodeObjectReport inspect_code_object_file(const std::string& path, const Gfx906Budget& budget) {
    std::vector<uint8_t> bytes;
    std::string err{};
    if (!read_file(path, kMaxBundleFileBytes, bytes, &err)) {
        CodeObjectReport co;
        co.error = err;
        return co;
    }
    return inspect_code_object(bytes.data(), bytes.size(), budget);
}

bool verify_bundle(const Manifest& manifest, const std::string& root, const Gfx906Budget& budget,
                   BundleReport& out, std::string* err) {
    out = BundleReport{};
    if (manifest.entries.empty()) {
        if (err) *err = "manifest has no entries";
        return false;
    }
    for (const auto& entry : manifest.entries) {
        FileReport fr;
        fr.path = entry.path;
        std::string why;
        std::string rel;
        if (!sanitize_relative_path(entry.path, rel, &why)) {
            fr.problems.push_back("manifest path rejected: " + why);
            out.files.push_back(std::move(fr));
            continue;
        }
        std::vector<uint8_t> bytes;
        std::string read_err;
        if (!read_file(join_path(root, rel), kMaxBundleFileBytes, bytes, &read_err)) {
            fr.problems.push_back(read_err);
            out.files.push_back(std::move(fr));
            continue;
        }
        fr.present = true;
        fr.size = uint64_t(bytes.size());
        fr.size_ok = fr.size == entry.size;
        if (!fr.size_ok) fr.problems.push_back("size " + std::to_string(fr.size) + " B != manifest " +
                                               std::to_string(entry.size) + " B");
        const Digest actual = sha256(bytes.data(), bytes.size());
        fr.digest_hex = to_hex(actual);
        fr.digest_ok = actual == entry.digest;
        if (!fr.digest_ok) fr.problems.push_back("SHA-256 mismatch: actual " + fr.digest_hex + ", expected " +
                                                 entry.digest_hex);
        fr.code = inspect_code_object(bytes.data(), bytes.size(), budget);
        if (!fr.code.error.empty()) {
            fr.problems.push_back("code object: " + fr.code.error);
        } else {
            for (const auto& p : fr.code.problems) fr.problems.push_back("code object: " + p);
        }
        fr.ok = fr.present && fr.size_ok && fr.digest_ok && fr.code.ok;
        out.files.push_back(std::move(fr));
    }
    for (const auto& fr : out.files) {
        if (!fr.ok) out.problems.push_back(fr.path.empty() ? std::string("<empty path>") : fr.path);
    }
    out.ok = !out.files.empty();
    for (const auto& fr : out.files) {
        if (!fr.ok) {
            out.ok = false;
            break;
        }
    }
    if (!out.ok && err) *err = "bundle verification failed";
    return out.ok;
}

bool verify_bundle(const Manifest& manifest, const std::string& root, BundleReport& out, std::string* err) {
    return verify_bundle(manifest, root, Gfx906Budget{}, out, err);
}

} // namespace vegajet::bundle
