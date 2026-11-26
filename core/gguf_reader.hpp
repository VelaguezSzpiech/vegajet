// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Zero-copy GGUF v2/v3 header + tensor-metadata reader.
//
// The header is parsed with buffered pread() (bounded heap: metadata only). Tensor payloads are never
// read into the heap; tensor_data() lazily maps the file PROT_READ and release() drops resident pages
// with MADV_DONTNEED, so a 40+ GiB model can be walked without growing RSS.
#pragma once

#include "core/ggml_types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace vegajet {

enum class GgufValueType : uint32_t {
    UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3, UINT32 = 4, INT32 = 5, FLOAT32 = 6,
    BOOL = 7, STRING = 8, ARRAY = 9, UINT64 = 10, INT64 = 11, FLOAT64 = 12,
};

struct GgufValue {
    GgufValueType type = GgufValueType::UINT32;
    // Scalars: exactly one of these is meaningful according to `type`.
    uint64_t u = 0;
    int64_t i = 0;
    double f = 0.0;
    std::string s;
    // Arrays: element type + length; numeric arrays are retained up to kMaxRetainedArray
    // elements, string arrays only when they are short (tokenizer vocabularies are skipped).
    GgufValueType arr_type = GgufValueType::UINT8;
    uint64_t arr_len = 0;
    std::vector<double> arr_num;
    std::vector<std::string> arr_str;
    bool arr_truncated = false;

    bool is_array() const { return type == GgufValueType::ARRAY; }
    std::string to_string(size_t max_items = 8) const;
};

struct GgufTensorInfo {
    std::string name;
    uint32_t n_dims = 0;
    uint64_t ne[4] = {1, 1, 1, 1};
    uint32_t type = 0;       // GgmlType id
    uint64_t offset = 0;     // relative to data_offset
    // Overflow-checked element count: 0 if the dims product does not fit in 64 bits.
    uint64_t n_elems() const;
    uint64_t n_bytes() const;  // 0 if the type is unknown, the geometry is unaligned, or it overflows
};

class GgufFile {
public:
    static constexpr uint64_t kMaxRetainedArray = 4096;

    GgufFile() = default;
    ~GgufFile();
    GgufFile(const GgufFile&) = delete;
    GgufFile& operator=(const GgufFile&) = delete;

    // Parses header + KV + tensor infos. Returns false and sets error() on any malformed input.
    bool open(const std::string& path);
    void close();

    const std::string& error() const { return err_; }
    const std::string& path() const { return path_; }
    uint32_t version() const { return version_; }
    uint64_t file_size() const { return file_size_; }
    uint64_t alignment() const { return alignment_; }
    uint64_t data_offset() const { return data_offset_; }

    const std::vector<std::string>& keys() const { return key_order_; }
    const GgufValue* find(const std::string& key) const;
    bool get_u64(const std::string& key, uint64_t& out) const;  // any integer/bool scalar
    bool get_f64(const std::string& key, double& out) const;    // any numeric scalar
    bool get_str(const std::string& key, std::string& out) const;
    std::string architecture() const;  // general.architecture or ""
    // Convenience: "<arch>.<suffix>" lookup.
    bool get_arch_u64(const std::string& suffix, uint64_t& out) const;
    bool get_arch_f64(const std::string& suffix, double& out) const;

    const std::vector<GgufTensorInfo>& tensors() const { return tensors_; }
    const GgufTensorInfo* tensor(const std::string& name) const;

    // Lazily mmaps the file read-only and returns a pointer to the tensor payload (nullptr on error).
    const uint8_t* tensor_data(const GgufTensorInfo& t);
    // Drops resident pages for a byte range previously returned by tensor_data().
    void release(const uint8_t* ptr, uint64_t bytes);

private:
    bool fail(const std::string& msg);

    std::string path_;
    std::string err_;
    int fd_ = -1;
    uint64_t file_size_ = 0;
    uint32_t version_ = 0;
    uint64_t alignment_ = 32;
    uint64_t data_offset_ = 0;
    void* map_ = nullptr;
    std::unordered_map<std::string, GgufValue> kv_;
    std::vector<std::string> key_order_;
    std::vector<GgufTensorInfo> tensors_;
    std::unordered_map<std::string, size_t> tensor_index_;
};

} // namespace vegajet
