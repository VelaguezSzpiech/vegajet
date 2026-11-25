// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/tokenizer.hpp"
#include "core/bpe_tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace vegajet {

namespace {

std::string cp_to_utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80u) {
        s.push_back(static_cast<char>(cp));
    } else if (cp < 0x800u) {
        s.push_back(static_cast<char>(0xC0u | (cp >> 6)));
        s.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else if (cp < 0x10000u) {
        s.push_back(static_cast<char>(0xE0u | (cp >> 12)));
        s.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        s.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else {
        s.push_back(static_cast<char>(0xF0u | (cp >> 18)));
        s.push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
        s.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        s.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    }
    return s;
}

// Decode next UTF-8 codepoint starting at s[pos], advancing pos.
uint32_t next_utf8_cp(const std::string& s, size_t& pos) {
    const uint8_t c0 = static_cast<uint8_t>(s[pos]);
    if (c0 < 0x80u) {
        ++pos;
        return c0;
    }
    if ((c0 & 0xE0u) == 0xC0u && pos + 1 < s.size()) {
        uint32_t cp = ((c0 & 0x1Fu) << 6) | (static_cast<uint8_t>(s[pos + 1]) & 0x3Fu);
        pos += 2;
        return cp;
    }
    if ((c0 & 0xF0u) == 0xE0u && pos + 2 < s.size()) {
        uint32_t cp = ((c0 & 0x0Fu) << 12) |
                      ((static_cast<uint8_t>(s[pos + 1]) & 0x3Fu) << 6) |
                      (static_cast<uint8_t>(s[pos + 2]) & 0x3Fu);
        pos += 3;
        return cp;
    }
    if ((c0 & 0xF8u) == 0xF0u && pos + 3 < s.size()) {
        uint32_t cp = ((c0 & 0x07u) << 18) |
                      ((static_cast<uint8_t>(s[pos + 1]) & 0x3Fu) << 12) |
                      ((static_cast<uint8_t>(s[pos + 2]) & 0x3Fu) << 6) |
                      (static_cast<uint8_t>(s[pos + 3]) & 0x3Fu);
        pos += 4;
        return cp;
    }
    ++pos;
    return c0;
}

// Safe bounds-checked GGUF binary reader
struct GgufReader {
    const uint8_t* p = nullptr;
    const uint8_t* end = nullptr;
    bool ok = true;

    template <typename T>
    T read() {
        if (!ok || static_cast<size_t>(end - p) < sizeof(T)) {
            ok = false;
            return T{};
        }
        T v{};
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }

    std::string read_str() {
        uint64_t len = read<uint64_t>();
        if (!ok || len > static_cast<uint64_t>(end - p)) {
            ok = false;
            return {};
        }
        std::string s(reinterpret_cast<const char*>(p), static_cast<size_t>(len));
        p += len;
        return s;
    }

    int64_t read_int_scalar(uint32_t vt) {
        switch (vt) {
            case 0: return static_cast<int64_t>(read<uint8_t>());
            case 1: return static_cast<int64_t>(read<int8_t>());
            case 2: return static_cast<int64_t>(read<uint16_t>());
            case 3: return static_cast<int64_t>(read<int16_t>());
            case 4: return static_cast<int64_t>(read<uint32_t>());
            case 5: return static_cast<int64_t>(read<int32_t>());
            case 7: return read<uint8_t>() ? 1 : 0;
            case 10: return static_cast<int64_t>(read<uint64_t>());
            case 11: return read<int64_t>();
            default: skip_val(vt); return -1;
        }
    }

    void skip_bytes(uint64_t n) {
        if (!ok || n > static_cast<uint64_t>(end - p)) {
            ok = false;
            return;
        }
        p += n;
    }

    void skip_val(uint32_t vt) {
        if (!ok) return;
        switch (vt) {
            case 0: case 1: case 7: skip_bytes(1); break;
            case 2: case 3: skip_bytes(2); break;
            case 4: case 5: case 6: skip_bytes(4); break;
            case 10: case 11: case 12: skip_bytes(8); break;
            case 8: { uint64_t l = read<uint64_t>(); skip_bytes(l); break; }
            case 9: {
                uint32_t et = read<uint32_t>();
                uint64_t cnt = read<uint64_t>();
                if (!ok) return;
                // Fast-path fixed-width arrays
                uint64_t elem_sz = 0;
                if (et == 0 || et == 1 || et == 7) elem_sz = 1;
                else if (et == 2 || et == 3) elem_sz = 2;
                else if (et == 4 || et == 5 || et == 6) elem_sz = 4;
                else if (et == 10 || et == 11 || et == 12) elem_sz = 8;
                if (elem_sz > 0) {
                    if (cnt > static_cast<uint64_t>(end - p) / elem_sz) { ok = false; return; }
                    p += cnt * elem_sz;
                } else {
                    for (uint64_t i = 0; i < cnt && ok; ++i) skip_val(et);
                }
                break;
            }
            default: ok = false; break;
        }
    }
};

} // namespace

BpeTokenizer::BpeTokenizer() {
    init_byte_tables();
}

void BpeTokenizer::init_byte_tables() {
    // Standard GPT-2 / Qwen byte-to-unicode bijection:
    // Printable ASCII [33..126] and Latin-1 [161..172], [174..255] map to themselves;
    // remaining 68 bytes (0..32, 127..160, 173) map sequentially to U+0100..U+0143.
    bool direct[256] = {};
    for (int b = 33; b <= 126; ++b) direct[b] = true;
    for (int b = 161; b <= 172; ++b) direct[b] = true;
    for (int b = 174; b <= 255; ++b) direct[b] = true;

    uint32_t next_cp = 256u;
    cp_to_byte_.clear();
    for (int b = 0; b < 256; ++b) {
        uint32_t cp = direct[b] ? static_cast<uint32_t>(b) : (next_cp++);
        byte_to_utf8_[b] = cp_to_utf8(cp);
        cp_to_byte_[cp] = static_cast<uint8_t>(b);
    }
}

void BpeTokenizer::rebuild_indices() {
    token_to_id_.clear();
    token_to_id_.reserve(id_to_token_.size());
    special_tokens_sorted_.clear();

    for (size_t i = 0; i < id_to_token_.size(); ++i) {
        const auto& tok = id_to_token_[i];
        int32_t id = static_cast<int32_t>(i);
        token_to_id_[tok] = id;
        if (is_special_token(id) && !tok.empty()) {
            special_tokens_sorted_.emplace_back(tok, id);
        }
    }
    std::sort(special_tokens_sorted_.begin(), special_tokens_sorted_.end(),
              [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
}

bool BpeTokenizer::is_special_token(int32_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) return false;
    if (id == bos_id_ || id == eos_id_ || id == pad_id_) return true;
    if (static_cast<size_t>(id) < token_types_.size()) {
        int32_t tt = token_types_[static_cast<size_t>(id)];
        if (tt == 3 || tt == 4) return true; // CONTROL or USER_DEFINED
    }
    return false;
}

bool BpeTokenizer::load_from_vectors(std::vector<std::string> tokens,
                                     std::vector<std::string> merges,
                                     std::vector<int32_t> token_types,
                                     int32_t bos_id,
                                     int32_t eos_id,
                                     int32_t pad_id,
                                     bool byte_level_bpe,
                                     const std::string& pre_type) {
    id_to_token_ = std::move(tokens);
    token_types_ = std::move(token_types);
    bos_id_ = bos_id;
    eos_id_ = eos_id;
    pad_id_ = pad_id;
    byte_level_bpe_ = byte_level_bpe;
    pre_type_ = pre_type.empty() ? "qwen2" : pre_type;

    merge_ranks_.clear();
    merge_ranks_.reserve(merges.size());
    for (size_t rank = 0; rank < merges.size(); ++rank) {
        const std::string& line = merges[rank];
        size_t sp = line.find(' ');
        if (sp == std::string::npos || sp == 0 || sp + 1 >= line.size()) continue;
        std::string a = line.substr(0, sp);
        std::string b = line.substr(sp + 1);
        merge_ranks_.emplace(std::make_pair(std::move(a), std::move(b)), static_cast<uint32_t>(rank));
    }

    rebuild_indices();
    return !id_to_token_.empty();
}

bool BpeTokenizer::load_from_gguf(const std::string& gguf_path, std::string* err_out) {
    int fd = open(gguf_path.c_str(), O_RDONLY);
    if (fd < 0) {
        if (err_out) *err_out = "Failed to open GGUF file: " + gguf_path;
        return false;
    }
    struct stat st{};
    if (fstat(fd, &st) != 0 || st.st_size < 24) {
        close(fd);
        if (err_out) *err_out = "Invalid GGUF file size: " + gguf_path;
        return false;
    }

    // Read-only mmap; touch only header KV metadata (OS never faults weight pages).
    void* base = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) {
        close(fd);
        if (err_out) *err_out = "mmap failed on GGUF file: " + gguf_path;
        return false;
    }

    GgufReader rd{
        reinterpret_cast<const uint8_t*>(base),
        reinterpret_cast<const uint8_t*>(base) + static_cast<size_t>(st.st_size),
        true
    };

    char magic[4] = {};
    for (int i = 0; i < 4; ++i) magic[i] = static_cast<char>(rd.read<uint8_t>());
    if (!rd.ok || std::memcmp(magic, "GGUF", 4) != 0) {
        munmap(base, static_cast<size_t>(st.st_size));
        close(fd);
        if (err_out) *err_out = "Bad GGUF magic in " + gguf_path;
        return false;
    }

    uint32_t version = rd.read<uint32_t>();
    if (!rd.ok || version < 2u || version > 3u) {
        munmap(base, static_cast<size_t>(st.st_size));
        close(fd);
        if (err_out) *err_out = "Unsupported GGUF version: " + std::to_string(version);
        return false;
    }

    (void) rd.read<uint64_t>(); // n_tensors
    uint64_t n_kv = rd.read<uint64_t>();

    std::vector<std::string> tokens;
    std::vector<std::string> merges;
    std::vector<int32_t> token_types;
    std::string model_type = "gpt2";
    std::string pre_type = "qwen2";
    int32_t bos = -1, eos = -1, pad = -1, unk = -1;
    bool add_bos = false;

    for (uint64_t k = 0; k < n_kv && rd.ok; ++k) {
        std::string key = rd.read_str();
        uint32_t vt = rd.read<uint32_t>();
        if (!rd.ok) break;

        if (key == "tokenizer.ggml.model" && vt == 8u) {
            model_type = rd.read_str();
        } else if (key == "tokenizer.ggml.pre" && vt == 8u) {
            pre_type = rd.read_str();
        } else if (key == "tokenizer.ggml.tokens" && vt == 9u) {
            uint32_t et = rd.read<uint32_t>();
            uint64_t cnt = rd.read<uint64_t>();
            if (et == 8u) {
                tokens.reserve(static_cast<size_t>(cnt));
                for (uint64_t i = 0; i < cnt && rd.ok; ++i) tokens.push_back(rd.read_str());
            } else {
                for (uint64_t i = 0; i < cnt && rd.ok; ++i) rd.skip_val(et);
            }
        } else if (key == "tokenizer.ggml.merges" && vt == 9u) {
            uint32_t et = rd.read<uint32_t>();
            uint64_t cnt = rd.read<uint64_t>();
            if (et == 8u) {
                merges.reserve(static_cast<size_t>(cnt));
                for (uint64_t i = 0; i < cnt && rd.ok; ++i) merges.push_back(rd.read_str());
            } else {
                for (uint64_t i = 0; i < cnt && rd.ok; ++i) rd.skip_val(et);
            }
        } else if (key == "tokenizer.ggml.token_type" && vt == 9u) {
            uint32_t et = rd.read<uint32_t>();
            uint64_t cnt = rd.read<uint64_t>();
            token_types.reserve(static_cast<size_t>(cnt));
            for (uint64_t i = 0; i < cnt && rd.ok; ++i) {
                token_types.push_back(static_cast<int32_t>(rd.read_int_scalar(et)));
            }
        } else if (key == "tokenizer.ggml.bos_token_id") {
            bos = static_cast<int32_t>(rd.read_int_scalar(vt));
        } else if (key == "tokenizer.ggml.eos_token_id") {
            eos = static_cast<int32_t>(rd.read_int_scalar(vt));
        } else if (key == "tokenizer.ggml.padding_token_id") {
            pad = static_cast<int32_t>(rd.read_int_scalar(vt));
        } else if (key == "tokenizer.ggml.unknown_token_id") {
            unk = static_cast<int32_t>(rd.read_int_scalar(vt));
        } else if (key == "tokenizer.ggml.add_bos_token") {
            add_bos = (rd.read_int_scalar(vt) != 0);
        } else {
            rd.skip_val(vt);
        }
    }

    munmap(base, static_cast<size_t>(st.st_size));
    close(fd);

    if (!rd.ok) {
        if (err_out) *err_out = "Truncated or malformed GGUF KV section in " + gguf_path;
        return false;
    }
    if (tokens.empty()) {
        if (err_out) *err_out = "No tokenizer.ggml.tokens found in " + gguf_path;
        return false;
    }

    model_type_ = model_type;
    unk_id_ = unk;
    add_bos_default_ = add_bos;
    bool is_byte_bpe = (model_type == "gpt2" || model_type == "bpe");
    return load_from_vectors(std::move(tokens), std::move(merges), std::move(token_types),
                             bos, eos, pad, is_byte_bpe, pre_type);
}

std::string BpeTokenizer::bytes_to_bpe_str(const std::string& raw) const {
    if (!byte_level_bpe_) return raw;
    std::string out;
    out.reserve(raw.size() * 2);
    for (unsigned char c : raw) {
        out += byte_to_utf8_[c];
    }
    return out;
}

std::string BpeTokenizer::bpe_str_to_bytes(const std::string& bpe) const {
    if (!byte_level_bpe_) {
        // SentencePiece U+2581 ("▁") -> space
        std::string out;
        out.reserve(bpe.size());
        for (size_t i = 0; i < bpe.size(); ) {
            if (i + 2 < bpe.size() &&
                static_cast<uint8_t>(bpe[i]) == 0xE2u &&
                static_cast<uint8_t>(bpe[i + 1]) == 0x96u &&
                static_cast<uint8_t>(bpe[i + 2]) == 0x81u) {
                out.push_back(' ');
                i += 3;
            } else {
                out.push_back(bpe[i++]);
            }
        }
        return out;
    }
    std::string out;
    out.reserve(bpe.size());
    size_t pos = 0;
    while (pos < bpe.size()) {
        size_t prev = pos;
        uint32_t cp = next_utf8_cp(bpe, pos);
        auto it = cp_to_byte_.find(cp);
        if (it != cp_to_byte_.end()) {
            out.push_back(static_cast<char>(it->second));
        } else {
            out.append(bpe, prev, pos - prev);
        }
    }
    return out;
}

std::vector<std::string> BpeTokenizer::pretokenize(const std::string& text) const {
    return bpe::Tokenizer::pretokenize(text, bpe::pretok_from_gguf(pre_type_));
}

void BpeTokenizer::encode_chunk(const std::string& chunk, std::vector<int32_t>& out) const {
    if (chunk.empty()) return;
    std::string mapped = bytes_to_bpe_str(chunk);

    // Fast path: whole chunk is already in vocab
    auto direct = token_to_id_.find(mapped);
    if (direct != token_to_id_.end()) {
        out.push_back(direct->second);
        return;
    }

    // Split into UTF-8 symbols
    std::vector<std::string> syms;
    for (size_t pos = 0; pos < mapped.size(); ) {
        size_t prev = pos;
        (void) next_utf8_cp(mapped, pos);
        syms.push_back(mapped.substr(prev, pos - prev));
    }

    // Iteratively merge the lowest-rank adjacent pair
    while (syms.size() >= 2) {
        uint32_t best_rank = std::numeric_limits<uint32_t>::max();
        size_t best_idx = syms.size();
        for (size_t i = 0; i + 1 < syms.size(); ++i) {
            auto it = merge_ranks_.find({syms[i], syms[i + 1]});
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
            }
        }
        if (best_idx == syms.size()) break;
        syms[best_idx] += syms[best_idx + 1];
        syms.erase(syms.begin() + static_cast<std::ptrdiff_t>(best_idx + 1));
    }

    for (const auto& s : syms) {
        auto it = token_to_id_.find(s);
        if (it != token_to_id_.end()) {
            out.push_back(it->second);
        } else if (unk_id_ >= 0) {
            out.push_back(unk_id_);
        }
    }
}

std::vector<int32_t> BpeTokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<int32_t> ids;
    if (add_bos && bos_id_ >= 0) {
        ids.push_back(bos_id_);
    }
    if (text.empty()) return ids;

    // Split text around exact special tokens (longest match first)
    size_t pos = 0;
    size_t seg_start = 0;
    while (pos < text.size()) {
        int32_t matched_id = -1;
        size_t matched_len = 0;
        for (const auto& [stok, sid] : special_tokens_sorted_) {
            if (stok.size() <= text.size() - pos &&
                text.compare(pos, stok.size(), stok) == 0) {
                matched_id = sid;
                matched_len = stok.size();
                break;
            }
        }
        if (matched_id >= 0) {
            if (pos > seg_start) {
                for (const auto& ch : pretokenize(text.substr(seg_start, pos - seg_start))) {
                    encode_chunk(ch, ids);
                }
            }
            ids.push_back(matched_id);
            pos += matched_len;
            seg_start = pos;
        } else {
            ++pos;
        }
    }
    if (seg_start < text.size()) {
        for (const auto& ch : pretokenize(text.substr(seg_start))) {
            encode_chunk(ch, ids);
        }
    }
    return ids;
}

std::string BpeTokenizer::decode_token(int32_t id, bool skip_special_tokens) const {
    if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) return "";
    if (is_special_token(id)) {
        return skip_special_tokens ? "" : id_to_token_[static_cast<size_t>(id)];
    }
    return bpe_str_to_bytes(id_to_token_[static_cast<size_t>(id)]);
}

std::string BpeTokenizer::decode(const std::vector<int32_t>& ids, bool skip_special_tokens) const {
    std::string out;
    for (int32_t id : ids) {
        out += decode_token(id, skip_special_tokens);
    }
    return out;
}

} // namespace vegajet
