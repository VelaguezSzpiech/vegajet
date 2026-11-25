// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Pre-tokenizer families beyond gpt2 / llama3 / qwen2 (core/bpe_tokenizer), dispatched from tokenizer.ggml.pre and
// tokenizer.ggml.model, plus the SentencePiece (SPM) path used by Gemma and pre-Tekken Mistral / Llama-2 vocabularies.
//
//   deepseek-llm    6 regex passes (newlines, an explicit cased-letter class, ASCII/CJK punctuation, trailing space,
//                   CJK + Hangul runs, digit runs), each refining the chunks of the previous pass;
//   deepseek-coder  5 passes (newlines, \s?\p{L}+, \s?\p{P}+, CJK + Hangul runs, single digits);
//   deepseek-v3     digits in groups of up to 3, CJK + kana runs, then the main alternation;
//   tekken          Mistral Tekken: camel-case aware letter runs, single digits;
//   spm             SentencePiece: spaces become U+2581 and the text is not split; merges follow vocabulary scores.
// Each regex is implemented as a hand-written matcher with the regex engine's leftmost-alternative and greedy /
// backtracking semantics; Unicode classes come from core/unicode_class_table.inc (generated from the UCD).
// tests/pretok_goldens.inc holds splits produced by the published regexes (tools/gen_pretok_tables.py).
#pragma once

#include "core/bpe_tokenizer.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace vegajet::pretok {

// Unicode general-category class: 0 other (C), 1 L, 2 M, 3 N, 4 P, 5 S, 6 Z.
uint8_t uc_class(uint32_t cp);
bool uc_white_space(uint32_t cp);  // the White_Space property (\s)

// Splits `text` for one of the extended families (DeepSeekLlm, DeepSeekCoder, DeepSeekV3, Tekken, Spm).
// Concatenating the result always reproduces `text` byte for byte (Spm: one chunk, unescaped).
std::vector<std::string> split(std::string_view text, bpe::PreTok family);

// SPM escape: ' ' -> U+2581 ("\xE2\x96\x81"), optionally with a leading U+2581.
std::string spm_escape(std::string_view text, bool add_space_prefix);
// SentencePiece BPE: symbols are UTF-8 characters; the adjacent pair whose concatenation is a vocabulary token with
// the highest score merges first (ties: leftmost). Final symbols not in the vocabulary fall back to <0xNN> byte
// tokens, then to `unk`.
void spm_encode(std::string_view escaped, const std::unordered_map<std::string, int32_t>& ids,
                const std::vector<float>& scores, int32_t unk, std::vector<int32_t>& out);

} // namespace vegajet::pretok
