// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Minimal test helpers + a synthetic GGUF writer shared by the CPU unit tests.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

#define VJ_CHECK(cond)                                                                  \
    do {                                                                                \
        if (!(cond)) {                                                                  \
            std::fprintf(stderr, "CHECK FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            std::exit(1);                                                               \
        }                                                                               \
    } while (0)

namespace vjtest {

class GgufWriter {
public:
    void kv_u32(const std::string& k, uint32_t v) { key(k, 4); pod(v); ++n_kv_; }
    void kv_i32(const std::string& k, int32_t v) { key(k, 5); pod(v); ++n_kv_; }
    void kv_f32(const std::string& k, float v) { key(k, 6); pod(v); ++n_kv_; }
    void kv_bool(const std::string& k, bool v) { key(k, 7); uint8_t b = v; pod(b); ++n_kv_; }
    void kv_u64(const std::string& k, uint64_t v) { key(k, 10); pod(v); ++n_kv_; }
    void kv_str(const std::string& k, const std::string& v) { key(k, 8); str(v); ++n_kv_; }
    void kv_arr_i32(const std::string& k, const std::vector<int32_t>& v) {
        key(k, 9); pod(uint32_t{5}); pod(uint64_t{v.size()});
        for (int32_t x : v) pod(x);
        ++n_kv_;
    }
    void kv_arr_str(const std::string& k, const std::vector<std::string>& v) {
        key(k, 9); pod(uint32_t{8}); pod(uint64_t{v.size()});
        for (const auto& s : v) str(s);
        ++n_kv_;
    }
    // ne = dims (ne[0] fastest), type = ggml id, payload bytes appended to the data section.
    void tensor(const std::string& name, std::vector<uint64_t> ne, uint32_t type, const std::vector<uint8_t>& data) {
        Tensor t{name, std::move(ne), type, data};
        tensors_.push_back(std::move(t));
    }
    std::vector<uint8_t> build(uint64_t align = 32) const {
        std::vector<uint8_t> out;
        auto put = [&](const void* p, size_t n) { out.insert(out.end(), (const uint8_t*)p, (const uint8_t*)p + n); };
        put("GGUF", 4);
        uint32_t ver = 3; put(&ver, 4);
        uint64_t nt = tensors_.size(); put(&nt, 8);
        uint64_t nkv = n_kv_; put(&nkv, 8);
        out.insert(out.end(), kv_.begin(), kv_.end());
        uint64_t off = 0;
        std::vector<uint64_t> offs;
        for (const auto& t : tensors_) {
            uint64_t n = t.name.size(); put(&n, 8); put(t.name.data(), n);
            uint32_t nd = static_cast<uint32_t>(t.ne.size()); put(&nd, 4);
            for (uint64_t d : t.ne) put(&d, 8);
            put(&t.type, 4);
            put(&off, 8);
            offs.push_back(off);
            off += (t.data.size() + align - 1) / align * align;
        }
        while (out.size() % align) out.push_back(0);
        for (const auto& t : tensors_) {
            out.insert(out.end(), t.data.begin(), t.data.end());
            while (out.size() % align) out.push_back(0);
        }
        return out;
    }

private:
    struct Tensor { std::string name; std::vector<uint64_t> ne; uint32_t type; std::vector<uint8_t> data; };
    template <typename T> void pod(const T& v) { kv_.insert(kv_.end(), (const uint8_t*)&v, (const uint8_t*)&v + sizeof(T)); }
    void str(const std::string& s) { pod(uint64_t{s.size()}); kv_.insert(kv_.end(), s.begin(), s.end()); }
    void key(const std::string& k, uint32_t type) { str(k); pod(type); }
    std::vector<uint8_t> kv_;
    std::vector<Tensor> tensors_;
    uint64_t n_kv_ = 0;
};

inline std::string write_temp(const std::vector<uint8_t>& bytes, const std::string& tag) {
    std::string path = "/tmp/vegajet_test_" + tag + "_" + std::to_string(getpid()) + ".bin";
    FILE* f = std::fopen(path.c_str(), "wb");
    VJ_CHECK(f != nullptr);
    VJ_CHECK(std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size());
    std::fclose(f);
    return path;
}

// Builds a GGUF byte stream with full control over every field (malformed-input tests: the
// GgufWriter above always derives consistent offsets/sizes, this one lets a test lie).
class RawGguf {
public:
    explicit RawGguf(uint64_t n_tensors, uint64_t n_kv) {
        raw("GGUF", 4);
        u32(3);
        u64(n_tensors);
        u64(n_kv);
    }
    void u32(uint32_t v) { pod(v); }
    void u64(uint64_t v) { pod(v); }
    void f32(float v) { pod(v); }
    void str(const std::string& s) { u64(s.size()); raw(s.data(), s.size()); }
    void kv_str(const std::string& k, const std::string& v) { str(k); u32(8); str(v); ++n_kv_; }
    void kv_u32(const std::string& k, uint32_t v) { str(k); u32(4); u32(v); ++n_kv_; }
    void kv_u64(const std::string& k, uint64_t v) { str(k); u32(10); u64(v); ++n_kv_; }
    void kv_f32(const std::string& k, float v) { str(k); u32(6); f32(v); ++n_kv_; }
    void kv_arr_scalars(const std::string& k, uint32_t elem_type, uint64_t count) {
        str(k); u32(9); u32(elem_type); u64(count);  // header only; caller appends elements
        ++n_kv_;
    }
    void kv_arr_str(const std::string& k, const std::vector<std::string>& v) {
        kv_arr_scalars(k, 8, v.size());
        for (const auto& s : v) str(s);
    }
    // Tensor info with a caller-supplied offset (may point anywhere, including past EOF).
    void tensor(const std::string& name, std::vector<uint64_t> ne, uint32_t type, uint64_t offset) {
        str(name);
        u32(static_cast<uint32_t>(ne.size()));
        for (uint64_t d : ne) u64(d);
        u32(type);
        u64(offset);
    }
    // Pads the metadata section to `align`, then appends `payload` zero bytes.
    // The n_kv header field is rewritten with the number of kv_*() calls actually made (the ctor
    // value is used only when none were), so a test cannot forget to keep the two in sync.
    std::vector<uint8_t> build(size_t payload = 0, uint64_t align = 32) const {
        std::vector<uint8_t> out = b_;
        if (n_kv_ > 0) std::memcpy(&out[16], &n_kv_, 8);
        while (out.size() % align) out.push_back(0);
        out.resize(out.size() + payload, 0);
        return out;
    }

private:
    template <typename T> void pod(const T& v) { raw(&v, sizeof(T)); }
    void raw(const void* p, size_t n) { b_.insert(b_.end(), (const uint8_t*)p, (const uint8_t*)p + n); }
    std::vector<uint8_t> b_;
    uint64_t n_kv_ = 0;
};

} // namespace vjtest
