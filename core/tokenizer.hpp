// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vegajet {

class BpeTokenizer {
public:
    BpeTokenizer();

    // Load tokenizer vocabulary, merges, and special token metadata from a GGUF v2/v3 file (read-only mmap).
    bool load_from_gguf(const std::string& gguf_path, std::string* err_out = nullptr);

    // Programmatic load (used by unit tests and MockEngine).
    bool load_from_vectors(std::vector<std::string> tokens,
                           std::vector<std::string> merges,
                           std::vector<int32_t> token_types = {},
                           int32_t bos_id = -1,
                           int32_t eos_id = -1,
                           int32_t pad_id = -1,
                           bool byte_level_bpe = true,
                           const std::string& pre_type = "qwen2");

    bool is_loaded() const { return !id_to_token_.empty(); }
    size_t vocab_size() const { return id_to_token_.size(); }
    int32_t bos_id() const { return bos_id_; }
    int32_t eos_id() const { return eos_id_; }
    int32_t pad_id() const { return pad_id_; }
    bool add_bos_default() const { return add_bos_default_; }
    const std::string& model_type() const { return model_type_; }
    const std::string& pre_type() const { return pre_type_; }
    void set_pre_type(const std::string& pre) { pre_type_ = pre.empty() ? "qwen2" : pre; }

    // Encode UTF-8 text into token IDs (preserves special/control tokens, applies BPE merges by rank).
    std::vector<int32_t> encode(const std::string& text, bool add_bos = false) const;

    // Decode token IDs back into UTF-8 text.
    std::string decode(const std::vector<int32_t>& ids, bool skip_special_tokens = false) const;

    // Decode a single token ID into UTF-8 bytes (suitable for streaming SSE deltas).
    std::string decode_token(int32_t id, bool skip_special_tokens = false) const;

    bool is_special_token(int32_t id) const;

private:
    struct PairHash {
        size_t operator()(const std::pair<std::string, std::string>& p) const noexcept {
            size_t h1 = std::hash<std::string>{}(p.first);
            size_t h2 = std::hash<std::string>{}(p.second);
            return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
        }
    };

    void init_byte_tables();
    void rebuild_indices();
    std::string bytes_to_bpe_str(const std::string& raw) const;
    std::string bpe_str_to_bytes(const std::string& bpe) const;
    std::vector<std::string> pretokenize(const std::string& text) const;
    void encode_chunk(const std::string& chunk, std::vector<int32_t>& out) const;

    std::string model_type_ = "gpt2";
    std::string pre_type_ = "qwen2";
    bool byte_level_bpe_ = true;
    bool add_bos_default_ = false;
    int32_t bos_id_ = -1;
    int32_t eos_id_ = -1;
    int32_t pad_id_ = -1;
    int32_t unk_id_ = -1;

    std::vector<std::string> id_to_token_;
    std::vector<int32_t> token_types_; // 1=normal, 2=unknown, 3=control, 4=user_defined, 6=byte
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<std::pair<std::string, std::string>, uint32_t, PairHash> merge_ranks_;
    std::vector<std::pair<std::string, int32_t>> special_tokens_sorted_; // longest-first for greedy match

    std::string byte_to_utf8_[256];
    std::unordered_map<uint32_t, uint8_t> cp_to_byte_;
};

} // namespace vegajet
