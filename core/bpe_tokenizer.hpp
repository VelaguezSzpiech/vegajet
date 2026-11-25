// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Byte-level BPE tokenizer read straight from GGUF metadata, plus OpenAI messages[] chat templates.
//
//   * read_gguf_vocab(): streams only the GGUF KV section (bounded reads, never the tensor data) and collects
//     tokenizer.ggml.{model, pre, tokens, merges, token_type, bos/eos/padding/unknown_token_id, add_bos_token}
//     and tokenizer.chat_template. HEAD's GgufFile drops long string arrays, so the vocabulary needs this reader.
//   * Tokenizer: the pre-tokenizer families gpt2 / llama3 / qwen2 are hand-written matchers of their published
//     regexes, with Unicode letter / number / space classes from range tables (exact for ASCII, Latin-1, Greek,
//     Cyrillic, CJK, Hangul, emoji and common punctuation blocks; combining marks in rare scripts may differ).
//     Merges run with a heap (O(n log n) per pre-token, leftmost first among equal ranks). Control (type 3) and
//     user-defined (type 4) tokens are matched atomically, longest first. Byte tokens <0xNN> decode to raw bytes.
//     decode(encode(s)) == s for any byte string.
//   * Chat templates: ChatML, Qwen2.5 (default system prompt), Qwen3 (reasoning stripped from earlier assistant
//     turns, empty <think> block when thinking is off) and Ornith (the qwen35moe family; treated as Qwen3 until a
//     file's own template says otherwise). Jinja is not executed; detect_chat_format() maps a GGUF template string
//     or architecture onto one of these.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace vegajet::bpe {

// Gpt2 / Llama3 / Qwen2 are matched here; the other families live in core/pretok_families.
enum class PreTok : uint8_t {
    Gpt2 = 0, Llama3 = 1, Qwen2 = 2, DeepSeekLlm = 3, DeepSeekCoder = 4, DeepSeekV3 = 5, Tekken = 6, Spm = 7,
};
const char* pretok_name(PreTok p);
PreTok pretok_from_gguf(const std::string& pre);  // "qwen2"/"qwen*" -> Qwen2, "llama-bpe"/"llama3" -> Llama3, else Gpt2
// Family for a vocabulary: tokenizer.ggml.model "llama" (SentencePiece: Gemma, Llama-2, pre-Tekken Mistral) -> Spm;
// otherwise by tokenizer.ggml.pre: qwen* -> Qwen2, llama-bpe / llama3 / llama-v3 / falcon3 -> Llama3,
// deepseek-llm, deepseek-coder, deepseek-v3, tekken -> their own; anything else -> Gpt2.
PreTok pretok_from_gguf(const std::string& pre, const std::string& model);

struct Vocab {
    std::string model = "gpt2";        // tokenizer.ggml.model
    std::string pre = "qwen2";         // tokenizer.ggml.pre
    std::string architecture;          // general.architecture
    std::vector<std::string> tokens;
    std::vector<std::string> merges;   // "left right"
    std::vector<int32_t> types;        // 1 normal, 2 unknown, 3 control, 4 user defined, 5 unused, 6 byte
    std::vector<float> scores;         // tokenizer.ggml.scores (SentencePiece merge priority)
    bool add_space_prefix = false;     // tokenizer.ggml.add_space_prefix (SentencePiece)
    int32_t bos = -1, eos = -1, pad = -1, unk = -1;
    bool add_bos = false;
    std::string chat_template;
};

bool read_gguf_vocab(const std::string& path, Vocab& out, std::string* err = nullptr);
bool parse_gguf_vocab(const uint8_t* data, size_t size, Vocab& out, std::string* err = nullptr);

class Tokenizer {
public:
    bool load(Vocab v, std::string* err = nullptr);
    bool loaded() const { return !tokens_.empty(); }

    std::vector<int32_t> encode(std::string_view text, bool add_bos = false, bool parse_special = true) const;
    std::string decode(const std::vector<int32_t>& ids, bool skip_special = false) const;
    std::string token_bytes(int32_t id) const;   // raw bytes a token stands for ("" for control tokens)
    int32_t token_id(const std::string& token_text) const;  // vocabulary string -> id, -1 if absent

    size_t vocab_size() const { return tokens_.size(); }
    int32_t bos() const { return v_.bos; }
    int32_t eos() const { return v_.eos; }
    bool add_bos() const { return v_.add_bos; }
    bool is_special(int32_t id) const;
    PreTok pretokenizer() const { return pre_; }
    const Vocab& vocab() const { return v_; }

    static std::vector<std::string> pretokenize(std::string_view text, PreTok mode);
    uint64_t unknown_symbols() const { return unknown_; }  // symbols that had to fall back to bytes / unk

private:
    void bpe_word(const std::string& word, std::vector<int32_t>& out) const;

    Vocab v_;
    PreTok pre_ = PreTok::Qwen2;
    std::vector<std::string> tokens_;
    std::unordered_map<std::string, int32_t> tok_id_;
    std::unordered_map<std::string, int32_t> merge_rank_;   // "left right" -> rank
    std::vector<std::pair<std::string, int32_t>> specials_;  // longest first
    std::string byte_sym_[256];                             // byte -> unicode symbol (UTF-8)
    std::unordered_map<uint32_t, uint8_t> sym_byte_;         // symbol code point -> byte
    mutable uint64_t unknown_ = 0;
};

// ---------------------------------------------------------------------------------------------------------
// Chat templates
// ---------------------------------------------------------------------------------------------------------
enum class ChatFormat : uint8_t { ChatML = 0, Qwen25 = 1, Qwen3 = 2, Ornith = 3 };
const char* chat_format_name(ChatFormat f);

struct Message {
    std::string role;
    std::string content;
};

struct ChatOptions {
    bool add_generation_prompt = true;
    bool enable_thinking = false;   // Qwen3 / Ornith: false appends an empty <think></think> block
    std::string default_system;     // overrides the template's default system prompt ("" keeps the template's)
};

ChatFormat detect_chat_format(const std::string& gguf_chat_template, const std::string& architecture);
std::string apply_chat_template(ChatFormat f, const std::vector<Message>& msgs, const ChatOptions& opt = {});

} // namespace vegajet::bpe
