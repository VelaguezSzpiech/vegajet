// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/gguf_reader.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace vegajet {

namespace {

// Buffered sequential pread() reader with bounds checking (never reads past EOF).
class PReader {
public:
    PReader(int fd, uint64_t size) : fd_(fd), size_(size) {}
    uint64_t pos() const { return pos_; }
    bool ok() const { return ok_; }

    bool read(void* dst, uint64_t n) {
        if (!ok_ || n > size_ - std::min(pos_, size_)) { ok_ = false; return false; }
        auto* out = static_cast<uint8_t*>(dst);
        while (n > 0) {
            if (pos_ < buf_pos_ || pos_ >= buf_pos_ + buf_len_) {
                buf_pos_ = pos_;
                const uint64_t want = std::min<uint64_t>(sizeof(buf_), size_ - pos_);
                ssize_t got = ::pread(fd_, buf_, want, static_cast<off_t>(pos_));
                if (got <= 0) { ok_ = false; return false; }
                buf_len_ = static_cast<uint64_t>(got);
            }
            const uint64_t off = pos_ - buf_pos_;
            const uint64_t take = std::min(n, buf_len_ - off);
            std::memcpy(out, buf_ + off, take);
            out += take; pos_ += take; n -= take;
        }
        return true;
    }
    bool skip(uint64_t n) {
        if (!ok_ || n > size_ - std::min(pos_, size_)) { ok_ = false; return false; }
        pos_ += n;
        return true;
    }
    template <typename T> bool pod(T& v) { return read(&v, sizeof(T)); }
    bool str(std::string& s, uint64_t max_len = (1ull << 24)) {
        uint64_t n = 0;
        if (!pod(n) || n > max_len) { ok_ = false; return false; }
        s.resize(n);
        return n == 0 || read(&s[0], n);
    }
    bool skip_str() {
        uint64_t n = 0;
        return pod(n) && skip(n);
    }

private:
    int fd_;
    uint64_t size_;
    uint64_t pos_ = 0;
    uint8_t buf_[1 << 16];
    uint64_t buf_pos_ = 0, buf_len_ = 0;
    bool ok_ = true;
};

uint64_t scalar_size(GgufValueType t) {
    switch (t) {
        case GgufValueType::UINT8: case GgufValueType::INT8: case GgufValueType::BOOL: return 1;
        case GgufValueType::UINT16: case GgufValueType::INT16: return 2;
        case GgufValueType::UINT32: case GgufValueType::INT32: case GgufValueType::FLOAT32: return 4;
        case GgufValueType::UINT64: case GgufValueType::INT64: case GgufValueType::FLOAT64: return 8;
        default: return 0;
    }
}

bool read_scalar(PReader& r, GgufValueType t, GgufValue& v) {
    uint8_t b[8] = {};
    const uint64_t n = scalar_size(t);
    if (n == 0 || !r.read(b, n)) return false;
    switch (t) {
        case GgufValueType::UINT8:  v.u = b[0]; v.i = b[0]; break;
        case GgufValueType::BOOL:   v.u = b[0] ? 1 : 0; v.i = static_cast<int64_t>(v.u); break;
        case GgufValueType::INT8:   v.i = static_cast<int8_t>(b[0]); v.u = static_cast<uint64_t>(v.i); break;
        case GgufValueType::UINT16: { uint16_t x; std::memcpy(&x, b, 2); v.u = x; v.i = x; break; }
        case GgufValueType::INT16:  { int16_t x;  std::memcpy(&x, b, 2); v.i = x; v.u = static_cast<uint64_t>(x); break; }
        case GgufValueType::UINT32: { uint32_t x; std::memcpy(&x, b, 4); v.u = x; v.i = x; break; }
        case GgufValueType::INT32:  { int32_t x;  std::memcpy(&x, b, 4); v.i = x; v.u = static_cast<uint64_t>(x); break; }
        case GgufValueType::UINT64: { uint64_t x; std::memcpy(&x, b, 8); v.u = x; v.i = static_cast<int64_t>(x); break; }
        case GgufValueType::INT64:  { int64_t x;  std::memcpy(&x, b, 8); v.i = x; v.u = static_cast<uint64_t>(x); break; }
        case GgufValueType::FLOAT32:{ float x;    std::memcpy(&x, b, 4); v.f = x; return true; }
        case GgufValueType::FLOAT64:{ double x;   std::memcpy(&x, b, 8); v.f = x; return true; }
        default: return false;
    }
    v.f = (t == GgufValueType::INT8 || t == GgufValueType::INT16 || t == GgufValueType::INT32 ||
           t == GgufValueType::INT64) ? static_cast<double>(v.i) : static_cast<double>(v.u);
    return true;
}

bool is_integer(GgufValueType t) {
    return t != GgufValueType::FLOAT32 && t != GgufValueType::FLOAT64 && t != GgufValueType::STRING &&
           t != GgufValueType::ARRAY;
}

} // namespace

std::string GgufValue::to_string(size_t max_items) const {
    char buf[64];
    switch (type) {
        case GgufValueType::STRING: {
            std::string e;
            for (char c : s) e += c == '\n' ? std::string("\\n") : std::string(1, c);
            if (max_items < 64 && e.size() > 96) e = e.substr(0, 96) + "... (" + std::to_string(s.size()) + " bytes)";
            return "\"" + e + "\"";
        }
        case GgufValueType::FLOAT32: case GgufValueType::FLOAT64:
            std::snprintf(buf, sizeof(buf), "%g", f); return buf;
        case GgufValueType::BOOL: return u ? "true" : "false";
        case GgufValueType::INT8: case GgufValueType::INT16: case GgufValueType::INT32: case GgufValueType::INT64:
            return std::to_string(i);
        case GgufValueType::ARRAY: {
            std::string out = "[";
            const size_t shown = arr_type == GgufValueType::STRING ? arr_str.size() : arr_num.size();
            for (size_t k = 0; k < shown && k < max_items; ++k) {
                if (k) out += ", ";
                if (arr_type == GgufValueType::STRING) out += "\"" + arr_str[k] + "\"";
                else { std::snprintf(buf, sizeof(buf), "%g", arr_num[k]); out += buf; }
            }
            if (arr_len > max_items || shown < arr_len) out += ", ...";
            out += "] (len " + std::to_string(arr_len) + ")";
            return out;
        }
        default: return std::to_string(u);
    }
}

uint64_t GgufTensorInfo::n_bytes() const { return ggml_row_bytes(type, n_elems()); }

GgufFile::~GgufFile() { close(); }

bool GgufFile::fail(const std::string& msg) {
    err_ = path_ + ": " + msg;
    close();
    return false;
}

void GgufFile::close() {
    if (map_) { ::munmap(map_, file_size_); map_ = nullptr; }
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
}

bool GgufFile::open(const std::string& path) {
    close();
    path_ = path;
    err_.clear();
    kv_.clear(); key_order_.clear(); tensors_.clear(); tensor_index_.clear();
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) return fail("cannot open for reading");
    struct stat st{};
    if (::fstat(fd_, &st) != 0) return fail("fstat failed");
    file_size_ = static_cast<uint64_t>(st.st_size);

    PReader r(fd_, file_size_);
    char magic[4];
    if (!r.read(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) return fail("bad magic (not GGUF)");
    if (!r.pod(version_) || version_ < 2 || version_ > 3) return fail("unsupported GGUF version");
    uint64_t n_tensors = 0, n_kv = 0;
    if (!r.pod(n_tensors) || !r.pod(n_kv)) return fail("truncated header");
    if (n_tensors > (1ull << 24) || n_kv > (1ull << 20)) return fail("implausible tensor/kv count");

    for (uint64_t k = 0; k < n_kv; ++k) {
        std::string key;
        uint32_t vt = 0;
        if (!r.str(key, 1 << 16) || !r.pod(vt)) return fail("truncated KV header");
        GgufValue v;
        v.type = static_cast<GgufValueType>(vt);
        if (v.type == GgufValueType::STRING) {
            if (!r.str(v.s)) return fail("bad string value for " + key);
        } else if (v.type == GgufValueType::ARRAY) {
            uint32_t at = 0;
            if (!r.pod(at) || !r.pod(v.arr_len)) return fail("bad array header for " + key);
            v.arr_type = static_cast<GgufValueType>(at);
            if (v.arr_type == GgufValueType::STRING) {
                const bool keep = v.arr_len <= 256;
                for (uint64_t j = 0; j < v.arr_len; ++j) {
                    if (keep) {
                        std::string s;
                        if (!r.str(s)) return fail("bad string array " + key);
                        v.arr_str.push_back(std::move(s));
                    } else if (!r.skip_str()) {
                        return fail("bad string array " + key);
                    }
                }
                v.arr_truncated = !keep;
            } else {
                const uint64_t es = scalar_size(v.arr_type);
                if (es == 0) return fail("unsupported array element type for " + key);
                const uint64_t keep = std::min<uint64_t>(v.arr_len, kMaxRetainedArray);
                v.arr_num.reserve(keep);
                for (uint64_t j = 0; j < keep; ++j) {
                    GgufValue e;
                    if (!read_scalar(r, v.arr_type, e)) return fail("bad numeric array " + key);
                    v.arr_num.push_back(e.f);
                }
                if (!r.skip((v.arr_len - keep) * es)) return fail("truncated array " + key);
                v.arr_truncated = keep < v.arr_len;
            }
        } else if (!read_scalar(r, v.type, v)) {
            return fail("bad scalar value for " + key);
        }
        if (!kv_.count(key)) key_order_.push_back(key);
        kv_[key] = std::move(v);
    }

    uint64_t align = 32;
    if (get_u64("general.alignment", align) && (align == 0 || (align & (align - 1)) != 0))
        return fail("general.alignment must be a power of two");
    alignment_ = align;

    tensors_.reserve(n_tensors);
    for (uint64_t t = 0; t < n_tensors; ++t) {
        GgufTensorInfo ti;
        if (!r.str(ti.name, 1 << 16) || !r.pod(ti.n_dims) || ti.n_dims > 4)
            return fail("bad tensor info header");
        for (uint32_t d = 0; d < ti.n_dims; ++d)
            if (!r.pod(ti.ne[d])) return fail("truncated tensor dims");
        if (!r.pod(ti.type) || !r.pod(ti.offset)) return fail("truncated tensor info");
        tensor_index_[ti.name] = tensors_.size();
        tensors_.push_back(std::move(ti));
    }
    data_offset_ = (r.pos() + alignment_ - 1) / alignment_ * alignment_;
    for (const auto& ti : tensors_) {
        const uint64_t nb = ti.n_bytes();
        if (nb == 0 && ggml_type_traits(ti.type).name == nullptr)
            return fail("tensor " + ti.name + " has unknown ggml type " + std::to_string(ti.type));
        // Overflow-safe: `offset` and `nb` are file-controlled, so a plain sum can wrap.
        if (data_offset_ > file_size_ || ti.offset > file_size_ - data_offset_ ||
            nb > file_size_ - data_offset_ - ti.offset)
            return fail("tensor " + ti.name + " extends past end of file");
    }
    return true;
}

const GgufValue* GgufFile::find(const std::string& key) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? nullptr : &it->second;
}

bool GgufFile::get_u64(const std::string& key, uint64_t& out) const {
    const GgufValue* v = find(key);
    if (!v || !is_integer(v->type)) return false;
    out = v->u;
    return true;
}

bool GgufFile::get_f64(const std::string& key, double& out) const {
    const GgufValue* v = find(key);
    if (!v || v->type == GgufValueType::STRING || v->type == GgufValueType::ARRAY) return false;
    out = v->f;
    return true;
}

bool GgufFile::get_str(const std::string& key, std::string& out) const {
    const GgufValue* v = find(key);
    if (!v || v->type != GgufValueType::STRING) return false;
    out = v->s;
    return true;
}

std::string GgufFile::architecture() const {
    std::string a;
    get_str("general.architecture", a);
    return a;
}

bool GgufFile::get_arch_u64(const std::string& suffix, uint64_t& out) const {
    return get_u64(architecture() + "." + suffix, out);
}

bool GgufFile::get_arch_f64(const std::string& suffix, double& out) const {
    return get_f64(architecture() + "." + suffix, out);
}

const GgufTensorInfo* GgufFile::tensor(const std::string& name) const {
    auto it = tensor_index_.find(name);
    return it == tensor_index_.end() ? nullptr : &tensors_[it->second];
}

const uint8_t* GgufFile::tensor_data(const GgufTensorInfo& t) {
    if (fd_ < 0) return nullptr;
    // Re-validate the range here: callers may hand in tensor infos that were never vetted by open().
    const uint64_t nb = t.n_bytes();
    if (nb == 0 || data_offset_ > file_size_ || t.offset > file_size_ - data_offset_ ||
        nb > file_size_ - data_offset_ - t.offset) {
        err_ = path_ + ": tensor " + t.name + " extends past end of file";
        return nullptr;
    }
    if (!map_) {
        void* m = ::mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (m == MAP_FAILED) { err_ = path_ + ": mmap failed"; return nullptr; }
        map_ = m;
    }
    return static_cast<const uint8_t*>(map_) + data_offset_ + t.offset;
}

void GgufFile::release(const uint8_t* ptr, uint64_t bytes) {
    if (!map_ || !ptr || bytes == 0) return;
    const long pg = ::sysconf(_SC_PAGESIZE);
    const uintptr_t lo = reinterpret_cast<uintptr_t>(ptr) & ~static_cast<uintptr_t>(pg - 1);
    const uintptr_t hi = reinterpret_cast<uintptr_t>(ptr) + bytes;
    ::madvise(reinterpret_cast<void*>(lo), hi - lo, MADV_DONTNEED);
}

} // namespace vegajet
