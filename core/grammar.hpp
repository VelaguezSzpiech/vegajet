// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Constrained decoding: GBNF grammars (llama.cpp dialect) and JSON Schema, matched over the bytes of vocabulary
// tokens, producing logit masks on the CPU.
//
//   Grammar::parse()       GBNF text: `name ::= alternatives`, "literals" (\n \t \r \\ \" \xHH \uXXXX \UXXXXXXXX),
//                          [classes] with ranges / ^negation / escapes, `.`, ( groups ), `|`, `*` `+` `?` `{m}`
//                          `{m,}` `{m,n}`, # comments; a rule ends where the next `name ::=` starts. Repetitions
//                          desugar into right-recursive helper rules. Classes match UTF-8 characters: ASCII members
//                          become byte sets, non-ASCII members (up to 256 code points per range) become literal
//                          alternatives, and a negated class matches any other ASCII byte or any multi-byte
//                          character (a negated class with non-ASCII members is refused). Left recursion is refused.
//   json_schema_to_gbnf()  type (string, number, integer, boolean, null, object, array, or a list of types),
//                          properties + required (required first, then optional ones in schema order, each of which
//                          may be omitted), additionalProperties (false unless given when properties exist), items,
//                          minItems / maxItems, minLength / maxLength, enum, const, anyOf / oneOf, $ref to #/$defs or
//                          #/definitions (recursion allowed). Unknown keywords (pattern, format, minimum, ...) widen to
//                          the plain type. Whitespace runs and number lengths are bounded, so a model cannot stall.
//   GrammarMatcher         the parse state is a set of rule stacks (llama.cpp style), interned into a lazily built
//                          DFA (state x byte -> state, computed once per byte equivalence class). Masks:
//                            mask_candidates(): checks a candidate list (the sampler's top-k);
//                            fill_mask():        the whole vocabulary, cached per state. A miss walks the vocabulary
//                                                byte trie; when the state loops on a large byte set (string
//                                                interiors), the tokens that stay inside the loop come from a
//                                                per-vocabulary region bitmap and only the region's exit edges
//                                                are walked.
#pragma once

#include "core/json_lite.hpp"

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace vegajet::grammar {

using ByteSet = std::array<uint64_t, 4>;

struct Grammar {
    struct Elem {
        uint8_t kind = 0;  // 0 end of alternative, 1 byte class, 2 rule reference
        uint32_t arg = 0;  // class index / rule id
    };
    std::vector<Elem> elems;                  // all alternatives back to back, each closed by an end element
    std::vector<std::vector<uint32_t>> alts;  // rule id -> start index of each alternative
    std::vector<std::string> names;           // rule id -> name (helpers are named "<rule>_<n>")
    std::vector<ByteSet> classes;             // byte sets
    uint32_t root = 0;

    static std::shared_ptr<const Grammar> parse(std::string_view gbnf, std::string* err = nullptr,
                                                std::string_view root_name = "root");
};

// JSON Schema -> GBNF text (root rule "root"); "" and *err on unsupported input.
std::string json_schema_to_gbnf(const json::Value& schema, std::string* err = nullptr);
std::string json_schema_to_gbnf(std::string_view schema_json, std::string* err = nullptr);

// Vocabulary as raw token bytes plus a byte trie over them (built once per vocabulary, shared by matchers).
class TokenTable {
public:
    // tokens[id] = the bytes the token stands for ("" for control tokens, which grammars never accept).
    explicit TokenTable(std::vector<std::string> tokens, int32_t eos = -1);
    size_t size() const { return tok_.size(); }
    size_t words() const { return (tok_.size() + 63) / 64; }
    const std::string& bytes(int32_t id) const { return tok_[static_cast<size_t>(id)]; }
    int32_t eos() const { return eos_; }

    struct TrieNode {
        uint32_t first_child = 0, n_children = 0;  // into edges()
        uint32_t first_token = 0, n_tokens = 0;    // tokens ending exactly here, into node_tokens()
    };
    struct Edge {
        uint8_t byte = 0;
        uint32_t node = 0;
    };
    const std::vector<TrieNode>& nodes() const { return nodes_; }
    const std::vector<Edge>& edges() const { return edges_; }
    const std::vector<int32_t>& node_tokens() const { return node_tokens_; }

    // Tokens whose bytes stay inside a loop: every byte at a character boundary is in `loop`, or starts a
    // multi-byte UTF-8 character whose lead byte is in `leads` (its continuation bytes are 0x80-0xBF).
    struct Region {
        std::vector<uint64_t> bits;  // tokens fully inside
        struct Exit {
            std::string tail;          // token bytes from the first byte that leaves the loop (at a boundary)
            std::vector<int32_t> ids;  // tokens sharing this tail
        };
        std::vector<Exit> exits;     // sorted by tail, so neighbours share walk prefixes
    };
    std::shared_ptr<const Region> region(const ByteSet& loop, const ByteSet& leads) const;

private:
    std::vector<std::string> tok_;
    int32_t eos_;
    std::vector<TrieNode> nodes_;
    std::vector<Edge> edges_;
    std::vector<int32_t> node_tokens_;
    mutable std::mutex region_mu_;
    mutable std::map<std::pair<ByteSet, ByteSet>, std::shared_ptr<const Region>> regions_;
};

class GrammarMatcher {
public:
    explicit GrammarMatcher(std::shared_ptr<const Grammar> g, size_t mask_cache_entries = 64,
                            size_t max_dfa_states = 8192);
    void reset();
    // Advances over the bytes; false (state unchanged) if the grammar rejects them.
    bool accept_bytes(std::string_view bytes);
    // EOS is accepted exactly when the grammar can end here (and finishes the match).
    bool accept_token(const TokenTable& vocab, int32_t id);
    bool can_end() const;   // an accepting parse exists at this point
    bool dead() const { return cur_ < 0; }
    bool finished() const { return finished_; }
    // ok[i] = 1 if ids[i] may follow (EOS: can_end()).
    void mask_candidates(const TokenTable& vocab, const int32_t* ids, size_t n, uint8_t* ok);
    // bits[id / 64] bit (id % 64) = token allowed. Cached per parse state.
    void fill_mask(const TokenTable& vocab, std::vector<uint64_t>& bits);
    // Sets disallowed logits to -inf (full vocabulary).
    void apply(const TokenTable& vocab, float* logits);

    struct Stats {
        uint64_t mask_hits = 0, mask_misses = 0, region_masks = 0, trie_nodes = 0, dfa_states = 0, dfa_resets = 0;
    };
    const Stats& stats() const { return st_; }
    size_t parse_stacks() const;

    using Stack = std::vector<uint32_t>;  // element indices; back() is the next byte class to match

private:
    struct State {
        std::vector<Stack> stacks;  // sorted, unique; an empty stack = the parse can end
        bool accepting = false;
        ByteSet first{};            // bytes some stack can take
        int32_t mask_slot = -1;
    };
    int32_t intern(std::vector<Stack>&& stacks);
    int32_t step(int32_t s, uint8_t c);
    void expand(Stack& work, std::vector<Stack>& out) const;
    void maybe_reset_dfa();
    void dfs(const TokenTable& vocab, uint32_t node, int32_t s, std::vector<uint64_t>& bits);
    bool loop_region(int32_t s, ByteSet& loop, ByteSet& leads);

    std::shared_ptr<const Grammar> g_;
    size_t cache_cap_, max_states_;
    std::vector<State> states_;
    std::vector<int32_t> trans_;  // states x 256; -2 unknown, -1 dead
    std::unordered_map<std::string, int32_t> index_;
    std::vector<Stack> init_;
    int32_t cur_ = -1;
    bool finished_ = false;
    struct MaskEntry {
        int32_t state = -1;
        uint64_t last = 0;
        std::vector<uint64_t> bits;
    };
    std::vector<MaskEntry> masks_;
    uint64_t tick_ = 0;
    Stats st_;
};

} // namespace vegajet::grammar
