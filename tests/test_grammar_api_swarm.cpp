// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// CPU tests for serving_grammar_api_swarm.patch:
//   grammar      GBNF parsing / matching (literals, classes, UTF-8, repetition, recursion, refusals), JSON Schema ->
//                GBNF acceptance, full-vocabulary masks (trie + loop regions + cache) equal to per-token checks at
//                every step of constrained generations, random constrained generations always yielding valid JSON,
//                and the mask latency gate on a 248K-token synthetic vocabulary (-O3 builds only).
//   tool parser  Qwen / DeepSeek / Llama formats, identical results for every 2-way split and byte-by-byte
//                streaming, malformed-JSON recovery, unknown tools and unrecoverable bodies falling back to content,
//                parallel = false, held-back partial markers at end of stream.
//   anthropic    request translation, response / SSE schema, and a live server: /v1/messages streaming and
//                non-streaming against /v1/chat/completions on the same scripted output (content, thinking,
//                tool calls, stop sequences, errors).
//   swarm        3-worker routing: least-busy spread, prefix affinity, affinity wait, queueing, priorities,
//                per-request cancellation, health, a concurrent stress run with invariants, and an end-to-end run
//                over 3 loopback `serve` instances.
#include "core/anthropic_shim.hpp"
#include "core/grammar.hpp"
#include "core/json_lite.hpp"
#include "core/server.hpp"
#include "core/swarm_router.hpp"
#include "core/tool_parser.hpp"
#include "tests/test_common.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <thread>

using namespace vegajet;
using Clock = std::chrono::steady_clock;

namespace {

#if defined(__OPTIMIZE__) && !defined(__SANITIZE_ADDRESS__) && !defined(__SANITIZE_THREAD__)
constexpr bool kTimed = true;
#else
constexpr bool kTimed = false;
#endif

double us_since(Clock::time_point t) { return std::chrono::duration<double, std::micro>(Clock::now() - t).count(); }

json::Value parse(const std::string& s) {
    json::Value v;
    if (!json::parse(s, v)) {
        std::fprintf(stderr, "bad JSON: %s\n", s.c_str());
        VJ_CHECK(false);
    }
    return v;
}

// ======================================================================================================== grammar

bool accepts(const std::shared_ptr<const grammar::Grammar>& g, std::string_view s) {
    grammar::GrammarMatcher m(g);
    return m.accept_bytes(s) && m.can_end();
}

void test_gbnf() {
    std::string err;
    struct Case {
        const char* gbnf;
        std::vector<std::pair<std::string, bool>> in;
    };
    const std::vector<Case> cases = {
        {"root ::= \"a\"+ \"b\"?", {{"a", 1}, {"aaab", 1}, {"b", 0}, {"", 0}, {"aba", 0}}},
        {"root ::= [a-c]{2,3}", {{"ab", 1}, {"abc", 1}, {"a", 0}, {"abca", 0}, {"ad", 0}}},
        {"root ::= [0-9]{2,}", {{"12", 1}, {"12345", 1}, {"1", 0}, {"1a", 0}}},
        {"root ::= x\nx ::= \"(\" x \")\" | ", {{"", 1}, {"()", 1}, {"(())", 1}, {"(()", 0}, {")(", 0}}},
        {"root ::= [^\"]* \"\\\"\"", {{"h\xC3\xA9llo\"", 1}, {"\"", 1}, {"a\"b", 0}, {"\xE4\xB8\xAD\xF0\x9F\x98\x80\"", 1}}},
        {"root ::= [\xC3\xA4\xC3\xB6\xC3\xBC]+", {{"\xC3\xA4\xC3\xB6", 1}, {"a", 0}, {"\xC3\xA5", 0}}},
        {"root ::= .{3}", {{"abc", 1}, {"a\xE2\x82\xAC" "b", 1}, {"ab", 0}, {"abcd", 0}}},
        {"# comment\nroot ::= ( \"x\" | \"y\" )* end  # trailing\nend ::= \"\\n\"", {{"\n", 1}, {"xyx\n", 1}, {"xz\n", 0}}},
        {"root ::= \"\\u00e9\\x41\" [\\t ]", {{"\xC3\xA9" "A\t", 1}, {"\xC3\xA9" "A ", 1}, {"eA ", 0}}},
        {"root ::= a b\na ::= \"a\"?\nb ::= \"b\"", {{"ab", 1}, {"b", 1}, {"a", 0}}},
    };
    for (const auto& c : cases) {
        auto g = grammar::Grammar::parse(c.gbnf, &err);
        if (!g) std::fprintf(stderr, "parse failed: %s: %s\n", c.gbnf, err.c_str());
        VJ_CHECK(g);
        for (const auto& [s, want] : c.in) {
            if (accepts(g, s) != want) std::fprintf(stderr, "grammar %s on '%s': want %d\n", c.gbnf, s.c_str(), want);
            VJ_CHECK(accepts(g, s) == want);
        }
    }
    // refusals
    const std::vector<std::pair<const char*, const char*>> bad = {
        {"root ::= root \"a\" | \"b\"", "left recursion"},
        {"root ::= x\nx ::= y \"a\"\ny ::= \"\" x", "left recursion"},
        {"root ::= foo", "undefined rule foo"},
        {"root ::= \"abc", "unterminated literal"},
        {"root ::= [a-", "unterminated character class"},
        {"root ::= \"a\"{3,1}", "n < m"},
        {"x ::= \"a\"", "missing root"},
        {"root ::= \"a\"\nroot ::= \"b\"", "defined twice"},
        {"root ::= [^\xC3\xA4]", "non-ASCII"},
    };
    for (const auto& [text, want] : bad) {
        err.clear();
        VJ_CHECK(!grammar::Grammar::parse(text, &err));
        if (err.find(want) == std::string::npos) std::fprintf(stderr, "error for %s: %s\n", text, err.c_str());
        VJ_CHECK(err.find(want) != std::string::npos);
    }
    // EOS / dead states
    auto g = grammar::Grammar::parse("root ::= \"ab\"", &err);
    grammar::TokenTable vt({"a", "b", "ab", "", "c"}, 3);
    grammar::GrammarMatcher m(g);
    VJ_CHECK(!m.accept_token(vt, 3));  // EOS before the end
    VJ_CHECK(!m.accept_token(vt, 4) && !m.dead());
    VJ_CHECK(m.accept_token(vt, 0) && !m.can_end());
    VJ_CHECK(m.accept_token(vt, 1) && m.can_end());
    VJ_CHECK(m.accept_token(vt, 3) && m.finished());
    VJ_CHECK(!m.accept_token(vt, 0));
    m.reset();
    VJ_CHECK(!m.finished() && m.accept_token(vt, 2) && m.can_end());
    std::vector<float> logits(vt.size(), 1.0f);
    m.reset();
    m.apply(vt, logits.data());
    VJ_CHECK(logits[0] == 1.0f && logits[2] == 1.0f && std::isinf(logits[1]) && std::isinf(logits[3]) && std::isinf(logits[4]));
    std::printf("[grammar] GBNF: %zu grammars, %zu refusals OK\n", cases.size(), bad.size());
}

const char* kWeatherSchema = R"({"type":"object","properties":{
    "location":{"type":"string","minLength":1,"maxLength":64},
    "unit":{"enum":["celsius","fahrenheit"]},
    "days":{"type":"integer"},
    "notes":{"type":"array","items":{"type":"string"},"maxItems":3},
    "precise":{"type":["boolean","null"]},
    "coords":{"$ref":"#/$defs/point"}},
  "required":["location","unit","days"],
  "$defs":{"point":{"type":"object","properties":{"lat":{"type":"number"},"lon":{"type":"number"}},"required":["lat","lon"]}}})";

void test_schema() {
    std::string err;
    const std::string gb = grammar::json_schema_to_gbnf(std::string_view(kWeatherSchema), &err);
    VJ_CHECK(!gb.empty());
    auto g = grammar::Grammar::parse(gb, &err);
    if (!g) std::fprintf(stderr, "%s\n%s\n", err.c_str(), gb.c_str());
    VJ_CHECK(g);
    const std::vector<std::pair<std::string, bool>> docs = {
        {R"({"location":"Paris","unit":"celsius","days":3})", true},
        {R"({"location": "Paris", "unit": "fahrenheit", "days": -2, "notes": ["a", "b\"c"], "precise": null, "coords": {"lat": 48.85, "lon": 2.35e0}})", true},
        {"{\n  \"location\": \"K\xC3\xB6ln\",\n  \"unit\": \"celsius\",\n  \"days\": 0,\n  \"notes\": []\n}", true},
        {R"({"location":"Paris","unit":"celsius","days":3,"precise":true})", true},
        {R"({"unit":"celsius","location":"Paris","days":3})", false},       // order: required first, schema order
        {R"({"location":"Paris","unit":"kelvin","days":3})", false},        // enum
        {R"({"location":"","unit":"celsius","days":3})", false},            // minLength
        {R"({"location":"Paris","unit":"celsius"})", false},                // missing required
        {R"({"location":"Paris","unit":"celsius","days":3.5})", false},     // integer
        {R"({"location":"Paris","unit":"celsius","days":01})", false},      // leading zero
        {R"({"location":"Paris","unit":"celsius","days":3,"notes":["a","b","c","d"]})", false},  // maxItems
        {R"({"location":"Paris","unit":"celsius","days":3,"extra":1})", false},  // additionalProperties
        {R"({"location":"Paris","unit":"celsius","days":3,"coords":{"lat":1}})", false},  // $ref required
        {"{\"location\":\"a\nb\",\"unit\":\"celsius\",\"days\":3}", false},  // raw control char in string
        {R"({"location":"a\nb \u00e9","unit":"celsius","days":3})", true},  // escapes are fine
    };
    for (const auto& [d, want] : docs) {
        if (accepts(g, d) != want) std::fprintf(stderr, "schema doc %s: want %d\n", d.c_str(), want);
        VJ_CHECK(accepts(g, d) == want);
    }
    // other shapes: anyOf, recursion, additionalProperties map, const, bounded arrays, empty schema
    struct S {
        const char* schema;
        std::vector<std::pair<std::string, bool>> docs;
    };
    const std::vector<S> more = {
        {R"({"anyOf":[{"type":"integer"},{"type":"string","maxLength":2}]})", {{"12", 1}, {"\"ab\"", 1}, {"\"abc\"", 0}, {"true", 0}}},
        {R"({"$ref":"#/definitions/node","definitions":{"node":{"type":"object","properties":{"v":{"type":"integer"},"kids":{"type":"array","items":{"$ref":"#/definitions/node"}}},"required":["v"]}}})",
         {{R"({"v":1,"kids":[{"v":2},{"v":3,"kids":[{"v":4}]}]})", 1}, {R"({"v":1,"kids":[{"w":2}]})", 0}}},
        {R"({"type":"object","additionalProperties":{"type":"number"}})", {{R"({"a":1,"b":2.5})", 1}, {R"({})", 1}, {R"({"a":"x"})", 0}}},
        {R"({"const":{"k":[1,"x"]}})", {{R"({"k":[1,"x"]})", 1}, {R"({"k":[1]})", 0}}},
        {R"({"type":"array","items":{"type":"boolean"},"minItems":2,"maxItems":3})", {{"[true,false]", 1}, {"[true]", 0}, {"[true,true,true,true]", 0}}},
        {R"({})", {{R"({"any":[1,{"x":null}]})", 1}, {"\"s\"", 1}, {"nul", 0}}},
        {R"({"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}}})", {{"{}", 1}, {R"({"b":1})", 1}, {R"({"a":1,"b":2})", 1}, {R"({"b":1,"a":2})", 0}}},
    };
    for (const auto& s : more) {
        const std::string text = grammar::json_schema_to_gbnf(std::string_view(s.schema), &err);
        auto gg = grammar::Grammar::parse(text, &err);
        if (!gg) std::fprintf(stderr, "%s: %s\n%s\n", s.schema, err.c_str(), text.c_str());
        VJ_CHECK(gg);
        for (const auto& [d, want] : s.docs) {
            if (accepts(gg, d) != want) std::fprintf(stderr, "schema %s doc %s: want %d\n", s.schema, d.c_str(), want);
            VJ_CHECK(accepts(gg, d) == want);
        }
    }
    VJ_CHECK(grammar::json_schema_to_gbnf(std::string_view(R"({"$ref":"#/$defs/missing"})"), &err).empty() &&
             err.find("unresolvable") != std::string::npos);
    VJ_CHECK(grammar::json_schema_to_gbnf(std::string_view("false"), &err).empty());
    VJ_CHECK(grammar::json_schema_to_gbnf(std::string_view("{not json"), &err).empty());
    std::printf("[grammar] JSON Schema: %zu + %zu shapes OK\n", docs.size(), more.size());
}

// Synthetic byte-level vocabulary: structural pieces, all single bytes, words with and without a leading space,
// quote / comma suffixed words, numbers, CJK and partial UTF-8 tokens.
std::vector<std::string> synth_vocab(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<std::string> toks;
    for (const char* p : {"{", "}", "[", "]", ",", ":", "\"", ", \"", "\":", "\": ", "\": \"", "\",", "\"}", "\"]", "\n",
                          "  ", "    ", " ", "\n  ", "true", "false", "null", "0", "1", "12", "345", "-", ".", "e", "\\n",
                          "\\\"", "{\"", "[\"", "},", "]}", "celsius", "\"celsius\"", "fahrenheit"})
        toks.push_back(p);
    for (int b = 0; b < 256; ++b) toks.push_back(std::string(1, static_cast<char>(b)));
    const char* al = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::uniform_int_distribution<int> len(1, 10), ch(0, 61), kind(0, 99), cont(0, 63);
    while (toks.size() < n) {
        const int k = kind(rng);
        std::string t;
        if (k < 8) {
            t = "\xE4\xB8";
            t += static_cast<char>(0x80 + cont(rng));
            if (k < 4) {
                t += "\xE5\x9C";
                t += static_cast<char>(0x80 + cont(rng));
            }
            if (k == 0) t.pop_back();  // partial UTF-8 token
        } else if (k < 12) {
            const int d = len(rng);
            for (int i = 0; i < d; ++i) t += static_cast<char>('0' + ch(rng) % 10);
        } else {
            if (k < 40) t = " ";
            const int m = len(rng);
            for (int i = 0; i < m; ++i) t += al[ch(rng)];
            if (k >= 90) t += "\"";
            if (k >= 95) t += ",";
        }
        toks.push_back(t);
    }
    toks.push_back("");  // EOS (control token)
    return toks;
}

bool bit(const std::vector<uint64_t>& b, size_t id) { return (b[id >> 6] >> (id & 63)) & 1u; }

void test_masks() {
    std::string err;
    const auto toks = synth_vocab(20000, 11);
    const int32_t eos = static_cast<int32_t>(toks.size()) - 1;
    grammar::TokenTable vt(toks, eos);
    auto g = grammar::Grammar::parse(grammar::json_schema_to_gbnf(std::string_view(kWeatherSchema), &err), &err);
    VJ_CHECK(g);
    std::vector<int32_t> all(vt.size());
    for (size_t i = 0; i < all.size(); ++i) all[i] = static_cast<int32_t>(i);
    std::vector<uint8_t> ok(vt.size());
    std::mt19937 rng(5);
    int finished = 0, steps_total = 0, compared = 0;
    for (int run = 0; run < 24; ++run) {
        grammar::GrammarMatcher m(g, run % 2 ? 4 : 64);  // small cache: exercises eviction
        grammar::GrammarMatcher ref(g);
        std::string text;
        std::vector<uint64_t> bits;
        for (int step = 0; step < 400; ++step) {
            m.fill_mask(vt, bits);
            // reference: the direct per-token walk, from a matcher that never builds full masks
            ref.mask_candidates(vt, all.data(), all.size(), ok.data());
            std::vector<int32_t> allowed;
            for (size_t id = 0; id < vt.size(); ++id) {
                if (bit(bits, id) != (ok[id] != 0)) {
                    std::fprintf(stderr, "mask mismatch run %d step %d token %zu '%s' after '%s'\n", run, step, id,
                                 toks[id].c_str(), text.c_str());
                    VJ_CHECK(false);
                }
                if (ok[id]) allowed.push_back(static_cast<int32_t>(id));
            }
            ++compared;
            VJ_CHECK(!allowed.empty());
            // bias towards finishing: EOS when allowed, structure-bearing tokens often
            int32_t pick;
            if (bit(bits, static_cast<size_t>(eos)) && rng() % 2 == 0) {
                pick = eos;
            } else {
                std::vector<int32_t> structural;
                for (int32_t id : allowed)
                    if (toks[static_cast<size_t>(id)].find_first_of("\"]},") != std::string::npos) structural.push_back(id);
                const auto& pool = (!structural.empty() && rng() % 3 != 0) ? structural : allowed;
                pick = pool[rng() % pool.size()];
            }
            VJ_CHECK(m.accept_token(vt, pick) && ref.accept_token(vt, pick));
            ++steps_total;
            if (pick == eos) break;
            text += toks[static_cast<size_t>(pick)];
        }
        if (m.finished()) {
            ++finished;
            const json::Value v = parse(text);  // every finished constrained generation is valid JSON ...
            VJ_CHECK(v.is_object() && v.get("location") && v.get("unit") && v.get("days"));  // ... of the schema
            VJ_CHECK(v.get("location")->is_string() && !v.get("location")->str.empty());
            VJ_CHECK(v.get("unit")->str == "celsius" || v.get("unit")->str == "fahrenheit");
            VJ_CHECK(v.get("days")->is_number() && v.get("days")->num == std::floor(v.get("days")->num));
            VJ_CHECK(accepts(g, text));
        }
    }
    VJ_CHECK(finished >= 12);
    std::printf("[grammar] masks: %d steps over 24 constrained generations, full mask == per-token walk at all %d; "
                "%d finished, all valid schema JSON\n",
                steps_total, compared, finished);
}

void test_mask_latency() {
    std::string err;
    const auto toks = synth_vocab(248000, 7);
    const int32_t eos = static_cast<int32_t>(toks.size()) - 1;
    auto t0 = Clock::now();
    grammar::TokenTable vt(toks, eos);
    const double build_ms = us_since(t0) / 1000.0;
    auto g = grammar::Grammar::parse(
        grammar::json_schema_to_gbnf(std::string_view(R"({"type":"object","properties":{
            "location":{"type":"string"},"unit":{"enum":["celsius","fahrenheit"]},"days":{"type":"integer"},
            "notes":{"type":"array","items":{"type":"string"}}},"required":["location","unit","days","notes"]})"), &err),
        &err);
    VJ_CHECK(g);
    const std::string target =
        R"({"location": "San Francisco Bay Area near the Golden Gate", "unit": "celsius", "days": 14, "notes": ["bring a jacket", "fog in the morning", "windy afternoons"]})";
    // Greedy longest-match tokenization of the target under the mask (the "model" always wants the target).
    std::vector<double> full_us, cand_us;
    double first_pass_ms = 0;
    for (int rep = 0; rep < 3; ++rep) {
        grammar::GrammarMatcher m(g);
        std::vector<uint64_t> bits;
        size_t pos = 0;
        const auto tr = Clock::now();
        std::vector<int32_t> cands;
        std::vector<uint8_t> ok;
        while (pos < target.size()) {
            auto t = Clock::now();
            m.fill_mask(vt, bits);
            const double us = us_since(t);
            int32_t best = -1;
            size_t bl = 0;
            cands.clear();
            for (size_t id = 0; id < vt.size(); ++id) {
                const std::string& s = toks[id];
                if (s.empty() || target.compare(pos, s.size(), s) != 0) continue;
                if (cands.size() < 64) cands.push_back(static_cast<int32_t>(id));
                if (bit(bits, id) && s.size() > bl) {
                    best = static_cast<int32_t>(id);
                    bl = s.size();
                }
            }
            for (size_t k = 0; cands.size() < 64; ++k) cands.push_back(static_cast<int32_t>((k * 7919) % vt.size()));
            ok.assign(cands.size(), 0);
            t = Clock::now();
            m.mask_candidates(vt, cands.data(), cands.size(), ok.data());
            const double cus = us_since(t);
            VJ_CHECK(best >= 0);
            if (rep > 0) {
                full_us.push_back(us);
                cand_us.push_back(cus);
            }
            VJ_CHECK(m.accept_token(vt, best));
            pos += bl;
        }
        if (rep == 0) first_pass_ms = us_since(tr) / 1000.0;
        m.fill_mask(vt, bits);
        VJ_CHECK(bit(bits, static_cast<size_t>(eos)));
        if (rep == 2) {
            const auto& s = m.stats();
            std::printf("[grammar] per generation: %zu tokens, %llu mask misses (%llu via loop regions), %llu hits, "
                        "%llu DFA states\n",
                        full_us.size() / 2, static_cast<unsigned long long>(s.mask_misses),
                        static_cast<unsigned long long>(s.region_masks), static_cast<unsigned long long>(s.mask_hits),
                        static_cast<unsigned long long>(s.dfa_states));
        }
    }
    auto pct = [](std::vector<double> v, double p) {
        std::sort(v.begin(), v.end());
        return v[std::min(v.size() - 1, static_cast<size_t>(p * static_cast<double>(v.size())))];
    };
    double mean = 0, cmean = 0;
    for (double x : full_us) mean += x;
    for (double x : cand_us) cmean += x;
    mean /= static_cast<double>(full_us.size());
    cmean /= static_cast<double>(cand_us.size());
    std::printf("[grammar] 248K vocab (%zu trie nodes, table %.0f ms; first generation incl. one-time loop regions "
                "%.1f ms)\n",
                vt.nodes().size(), build_ms, first_pass_ms);
    std::printf("[grammar] full-vocab mask: mean %.1f us/tok, p50 %.1f, p99 %.1f, max %.1f | top-64 candidates: mean "
                "%.2f us, max %.1f\n",
                mean, pct(full_us, 0.5), pct(full_us, 0.99), pct(full_us, 1.0), cmean, pct(cand_us, 1.0));
    if (kTimed) {
        VJ_CHECK(mean < 150.0);   // gate: < 0.15 ms/tok mask build (full vocabulary, amortized incl. misses)
        VJ_CHECK(cmean < 150.0);  // candidate path
    }
}

// ======================================================================================================== tools

std::string show(const toolcall::Message& m) {
    std::string o = "C[" + m.content + "] R[" + m.reasoning + "]";
    for (const auto& c : m.tool_calls) o += " T(" + c.id + "," + c.name + "," + c.arguments + ")";
    return o;
}

toolcall::Message stream_split(const toolparse::Options& o, const std::string& text, size_t k) {
    toolparse::ToolStreamParser p(o);
    std::vector<toolcall::Event> ev;
    p.push(std::string_view(text).substr(0, k), ev);
    p.push(std::string_view(text).substr(k), ev);
    p.finish(ev);
    toolcall::Message m;
    toolcall::accumulate(ev, m);
    return m;
}

// Parses whole, at every 2-way split, and byte by byte; all must agree. Returns the whole-text result.
toolcall::Message parse_everyway(const toolparse::Options& o, const std::string& text) {
    const toolcall::Message whole = toolparse::parse_all(text, o);
    for (size_t k = 0; k <= text.size(); ++k) {
        const toolcall::Message m = stream_split(o, text, k);
        if (show(m) != show(whole))
            std::fprintf(stderr, "split %zu of %s:\n  %s\n  %s\n", k, text.c_str(), show(m).c_str(), show(whole).c_str());
        VJ_CHECK(show(m) == show(whole));
    }
    toolparse::ToolStreamParser p(o);
    std::vector<toolcall::Event> ev;
    for (char c : text) p.push(std::string_view(&c, 1), ev);
    p.finish(ev);
    toolcall::Message m;
    toolcall::accumulate(ev, m);
    VJ_CHECK(show(m) == show(whole));
    return whole;
}

void test_tool_parser() {
    using toolparse::Format;
    toolparse::Options q;
    q.format = Format::Qwen;
    toolparse::Options ds = q;
    ds.format = Format::DeepSeek;
    toolparse::Options ll = q;
    ll.format = Format::Llama;
    int n = 0;
    auto expect = [&](const toolparse::Options& o, const std::string& text, const std::string& want) {
        const std::string got = show(parse_everyway(o, text));
        if (got != want) std::fprintf(stderr, "tool parse %s:\n  got  %s\n  want %s\n", text.c_str(), got.c_str(), want.c_str());
        VJ_CHECK(got == want);
        ++n;
    };
    // Qwen / Hermes
    expect(q, "<think>plan\n</think>\n\nChecking.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>",
           "C[Checking.] R[plan] T(call_0,get_weather,{\"city\":\"Paris\"})");
    expect(q, "<tool_call>{'name': 'f', 'arguments': {'a': True, 'b': None, 'c': [1,2,],}}</tool_call><tool_call>{\"name\":\"g\",\"arguments\":\"{\\\"x\\\": 1}\"}</tool_call>",
           "C[] R[] T(call_0,f,{\"a\":true,\"b\":null,\"c\":[1,2]}) T(call_1,g,{\"x\":1})");
    expect(q, "a <tool_call>{\"name\": broken</tool_call> b", "C[a <tool_call>{\"name\": broken</tool_call> b] R[]");
    expect(q, "cut <tool_call>{\"name\": \"x\", \"arguments\": {\"q\": \"abc", "C[cut] R[] T(call_0,x,{\"q\":\"abc\"})");
    expect(q, "text then <tool_ca", "C[text then <tool_ca] R[]");  // partial marker at the very end is content
    expect(q, "<tool_call>```json\n{\"name\": \"f\", \"arguments\": {}}\n```</tool_call>", "C[] R[] T(call_0,f,{})");
    expect(q, "<tool_call>{\"function\": {\"name\": \"f\", \"arguments\": {\"k\": \"\xE2\x82\xAC\"}}}</tool_call>",
           "C[] R[] T(call_0,f,{\"k\":\"\xE2\x82\xAC\"})");
    // DeepSeek V3 / R1 and V3.1
    const std::string dsb = "<\xEF\xBD\x9Ctool\xE2\x96\x81" "calls\xE2\x96\x81" "begin\xEF\xBD\x9C>";
    const std::string dse = "<\xEF\xBD\x9Ctool\xE2\x96\x81" "calls\xE2\x96\x81" "end\xEF\xBD\x9C>";
    const std::string dcb = "<\xEF\xBD\x9Ctool\xE2\x96\x81" "call\xE2\x96\x81" "begin\xEF\xBD\x9C>";
    const std::string dce = "<\xEF\xBD\x9Ctool\xE2\x96\x81" "call\xE2\x96\x81" "end\xEF\xBD\x9C>";
    const std::string sep = "<\xEF\xBD\x9Ctool\xE2\x96\x81" "sep\xEF\xBD\x9C>";
    expect(ds, "<think>r</think>Sure." + dsb + dcb + "function" + sep + "get_weather\n```json\n{\"city\": \"Tokyo\"}\n```" + dce +
                   "\n" + dcb + "function" + sep + "get_time\n```json\n{\"tz\": \"JST\",}\n```" + dce + dse,
           "C[Sure.] R[r] T(call_0,get_weather,{\"city\":\"Tokyo\"}) T(call_1,get_time,{\"tz\":\"JST\"})");
    expect(ds, dsb + dcb + "search" + sep + "{\"q\": \"x\"}" + dce + dse + "done", "C[done] R[] T(call_0,search,{\"q\":\"x\"})");
    expect(ds, "x" + dsb + dcb + "function" + sep + "f\n```json\n{\"a\": ", "C[x] R[] T(call_0,f,{\"a\":null})");
    expect(ds, dsb + dcb + "function" + sep + "\n```json\n{}\n```" + dce + dse,
           "C[" + dcb + "function" + sep + "\n```json\n{}\n```" + dce + "] R[]");  // no name: falls back
    // Llama 3.x
    expect(ll, "<|python_tag|>{\"name\": \"calc\", \"parameters\": {\"x\": 1}}; {\"name\": \"calc\", \"parameters\": {\"x\": 2}}<|eom_id|>",
           "C[] R[] T(call_0,calc,{\"x\":1}) T(call_1,calc,{\"x\":2})");
    expect(ll, "{\"name\": \"lookup\", \"parameters\": {\"id\": 7}}", "C[] R[] T(call_0,lookup,{\"id\":7})");
    expect(ll, "{\"name\": \"a\", \"parameters\": {}}\n{\"name\": \"b\", \"parameters\": {}}then text",
           "C[then text] R[] T(call_0,a,{}) T(call_1,b,{})");
    expect(ll, "{\"answer\": 42} is the json", "C[{\"answer\": 42} is the json] R[]");
    expect(ll, "<function=spotify>{\"song\": \"x\"}</function>", "C[] R[] T(call_0,spotify,{\"song\":\"x\"})");
    expect(ll, "<|python_tag|>{\"name\": \"f\", \"parameters\": {\"q\": 'single'}}", "C[] R[] T(call_0,f,{\"q\":\"single\"})");
    // known tools / parallel = false
    toolparse::Options k = q;
    k.known_tools = {"get_weather"};
    expect(k, "<tool_call>{\"name\": \"rm_rf\", \"arguments\": {}}</tool_call>", "C[<tool_call>{\"name\": \"rm_rf\", \"arguments\": {}}</tool_call>] R[]");
    toolparse::Options one = q;
    one.parallel = false;
    {
        toolparse::ToolStreamParser p(one);
        std::vector<toolcall::Event> ev;
        p.push("<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call><tool_call>{\"name\":\"b\",\"arguments\":{}}</tool_call>", ev);
        p.finish(ev);
        VJ_CHECK(p.done() && p.calls() == 1 && ev.size() == 1 && ev[0].call.name == "a");
    }
    toolparse::Options norepair = q;
    norepair.repair = false;
    expect(norepair, "<tool_call>{'name': 'f'}</tool_call>", "C[<tool_call>{'name': 'f'}</tool_call>] R[]");
    // repair_json directly
    bool ch = false;
    VJ_CHECK(toolparse::repair_json("```json\n{name: 'x', \"list\": [1,2,], \"ok\": True, \"half\": {\"a\": \"b", &ch) ==
                 "{\"name\":\"x\",\"list\":[1,2],\"ok\":true,\"half\":{\"a\":\"b\"}}" && ch);
    VJ_CHECK(toolparse::repair_json("{\"a\": 1, \"b", &ch) == "{\"a\":1}");
    VJ_CHECK(toolparse::repair_json("{\"a\": 1}}} trailing", &ch) == "{\"a\":1}");
    VJ_CHECK(toolparse::repair_json("{\"a\": \"line\nbreak\"}", &ch) == "{\"a\":\"line\\nbreak\"}");
    VJ_CHECK(toolparse::repair_json("{\"a\": [1, {\"b\": 2]}", &ch) == "{\"a\":[1,{\"b\":2}]}");
    VJ_CHECK(toolparse::repair_json("{\"a\": 1}", &ch) == "{\"a\":1}" && !ch);
    VJ_CHECK(toolparse::repair_json("no json here", &ch).empty());
    VJ_CHECK(toolparse::repair_json("{\"a\": maybe}", &ch).empty());
    toolparse::Format f;
    VJ_CHECK(toolparse::format_from_name("DeepSeek-V3", f) && f == Format::DeepSeek);
    VJ_CHECK(toolparse::format_from_name("llama3", f) && f == Format::Llama && !toolparse::format_from_name("gpt", f));
    std::printf("[tools] %d transcripts x every split point + bytewise identical; repair / fallback / parallel OK\n", n);
}

// ======================================================================================================== anthropic

void test_anthropic_translate() {
    std::string err;
    anthropic::Limits lim;
    {
        serve::GenerateRequest g;
        const json::Value r = parse(R"({"model":"m","max_tokens":64,"system":[{"type":"text","text":"Be brief."}],
            "messages":[{"role":"user","content":"Weather?"},
              {"role":"assistant","content":[{"type":"thinking","thinking":"need tool","signature":"x"},{"type":"text","text":"Checking."},
                 {"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Paris"}}]},
              {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":[{"type":"text","text":"17 C"}]},{"type":"text","text":"Thanks"}]}],
            "tools":[{"name":"get_weather","description":"Weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}}}}],
            "tool_choice":{"type":"any","disable_parallel_tool_use":true},
            "temperature":0.2,"top_k":5,"stop_sequences":["END"],"stream":true})");
        VJ_CHECK(anthropic::parse_request(r, lim, g, err));
        VJ_CHECK(g.model == "m" && g.max_tokens == 64 && g.stream && g.stop.size() == 1 && g.sampling.top_k == 5);
        VJ_CHECK(g.messages.size() == 5 && g.messages[0].role == "system" && g.messages[0].content == "Be brief.");
        VJ_CHECK(g.messages[2].role == "assistant" && g.messages[2].reasoning_content == "need tool" &&
                 g.messages[2].content == "Checking." && g.messages[2].tool_calls.size() == 1 &&
                 g.messages[2].tool_calls[0].arguments == "{\"city\":\"Paris\"}");
        VJ_CHECK(g.messages[3].role == "tool" && g.messages[3].tool_call_id == "toolu_1" && g.messages[3].name == "get_weather" &&
                 g.messages[3].content == "17 C");
        VJ_CHECK(g.messages[4].role == "user" && g.messages[4].content == "Thanks");
        VJ_CHECK(g.tools.tools.size() == 1 && g.tools.choice.mode == toolcall::ChoiceMode::Required && !g.tools.parallel);
        VJ_CHECK(g.tools.tools[0].json.find("\"parameters\"") != std::string::npos);
    }
    {
        serve::GenerateRequest g;
        VJ_CHECK(anthropic::parse_request(parse(R"({"messages":[{"role":"user","content":"x"}],"tools":[{"name":"f"}],"tool_choice":{"type":"tool","name":"f"}})"), lim, g, err));
        VJ_CHECK(g.tools.choice.mode == toolcall::ChoiceMode::Named && g.tools.choice.name == "f" && g.max_tokens == 256 &&
                 g.model == "vegajet");
    }
    const std::vector<std::pair<const char*, const char*>> bad = {
        {R"({"messages":[]})", "non-empty"},
        {R"({"messages":[{"role":"system","content":"x"}]})", "role"},
        {R"({"messages":[{"role":"user","content":[{"type":"image","source":{}}]}]})", "unsupported content block 'image'"},
        {R"({"messages":[{"role":"user","content":"x"}],"temperature":1.5})", "temperature"},
        {R"({"messages":[{"role":"user","content":"x"}],"tools":[{"type":"web_search_20250305","name":"web_search"}]})", "client tools"},
        {R"({"messages":[{"role":"user","content":"x"}],"tool_choice":{"type":"tool","name":"nope"},"tools":[{"name":"f"}]})", "not in 'tools'"},
        {R"({"messages":[{"role":"user","content":"x"}],"max_tokens":0})", "max_tokens"},
        {R"({"messages":[{"role":"user","content":[{"type":"tool_result"}]}]})", "tool_use_id"},
    };
    for (const auto& [body, want] : bad) {
        serve::GenerateRequest g;
        err.clear();
        VJ_CHECK(!anthropic::parse_request(parse(body), lim, g, err));
        if (err.find(want) == std::string::npos) std::fprintf(stderr, "%s -> %s\n", body, err.c_str());
        VJ_CHECK(err.find(want) != std::string::npos);
    }
    // response object
    toolcall::Message m;
    m.reasoning = "r";
    m.content = "hi";
    m.tool_calls.push_back(toolcall::ToolCall{"toolu_x", "f", "{\"a\":1}"});
    const json::Value resp = parse(anthropic::response_json("msg_1", "m", m, "tool_use", "", {7, 9}));
    VJ_CHECK(resp.get("type")->str == "message" && resp.get("role")->str == "assistant" && resp.get("stop_reason")->str == "tool_use");
    VJ_CHECK(resp.get("stop_sequence")->is_null() && resp.get("usage")->get("input_tokens")->num == 7 &&
             resp.get("usage")->get("output_tokens")->num == 9);
    const auto& c = resp.get("content")->arr;
    VJ_CHECK(c.size() == 3 && c[0].get("type")->str == "thinking" && c[1].get("text")->str == "hi" &&
             c[2].get("type")->str == "tool_use" && c[2].get("input")->get("a")->num == 1);
    VJ_CHECK(std::string(anthropic::stop_reason(serve::FinishReason::Length, false)) == "max_tokens");
    VJ_CHECK(std::string(anthropic::stop_reason(serve::FinishReason::Stop, true)) == "stop_sequence");
    VJ_CHECK(std::string(anthropic::stop_reason(serve::FinishReason::ToolCalls, true)) == "tool_use");
    const json::Value e = parse(anthropic::error_json(anthropic::error_type(503), "busy"));
    VJ_CHECK(e.get("type")->str == "error" && e.get("error")->get("type")->str == "overloaded_error");
    std::printf("[anthropic] translation: %zu refusals, response schema OK\n", bad.size());
}

// ---- live server

struct Script {
    std::mutex mu;
    std::string text;
    serve::GenerateRequest last;
    serve::GenerateResult operator()(const serve::GenerateRequest& r, const serve::TokenSink& sink) {
        std::string t;
        {
            std::lock_guard<std::mutex> lk(mu);
            last = r;
            t = text;
        }
        for (size_t i = 0, k = 0; i < t.size(); i += k) {
            k = 1 + (i * 7 + 3) % 3;
            if (!sink(std::string_view(t).substr(i, k))) break;
        }
        serve::GenerateResult res;
        res.prompt_tokens = 11;
        return res;
    }
};

int connect_local(uint16_t port) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    VJ_CHECK(fd >= 0);
    timeval tv{10, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    VJ_CHECK(::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) == 0);
    return fd;
}

struct Resp {
    int status = 0;
    std::string body;
};

Resp post(uint16_t port, const std::string& path, const std::string& body) {
    const int fd = connect_local(port);
    const std::string req = "POST " + path + " HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n"
                            "Content-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
    for (size_t off = 0; off < req.size();) {
        const ssize_t n = ::send(fd, req.data() + off, req.size() - off, MSG_NOSIGNAL);
        VJ_CHECK(n > 0);
        off += static_cast<size_t>(n);
    }
    std::string all;
    char buf[8192];
    for (ssize_t n; (n = ::recv(fd, buf, sizeof(buf), 0)) > 0;) all.append(buf, static_cast<size_t>(n));
    ::close(fd);
    Resp r;
    const size_t he = all.find("\r\n\r\n");
    VJ_CHECK(he != std::string::npos);
    r.status = std::atoi(all.c_str() + all.find(' ') + 1);
    r.body = all.substr(he + 4);
    return r;
}

struct Parts {  // what a client reconstructs from either API
    std::string content, reasoning, finish;
    std::vector<std::pair<std::string, std::string>> calls;  // name, compact arguments
};

Parts parse_openai_sse(const std::string& body) {
    Parts p;
    std::map<int, std::pair<std::string, std::string>> calls;
    bool done = false;
    for (size_t pos = 0; pos < body.size();) {
        const size_t e = body.find("\n\n", pos);
        VJ_CHECK(e != std::string::npos);
        const std::string ev = body.substr(pos, e - pos);
        pos = e + 2;
        if (ev == "data: [DONE]") {
            done = true;
            continue;
        }
        const json::Value c = parse(ev.substr(6));
        const auto& ch = c.get("choices")->arr;
        if (ch.empty()) continue;
        if (const auto* f = ch[0].get("finish_reason"); f && f->is_string()) p.finish = f->str;
        const json::Value* d = ch[0].get("delta");
        if (const auto* x = d->get("content"); x && x->is_string()) p.content += x->str;
        if (const auto* x = d->get("reasoning_content"); x && x->is_string()) p.reasoning += x->str;
        if (const auto* tc = d->get("tool_calls")) {
            const json::Value& t = tc->arr[0];
            auto& cl = calls[static_cast<int>(t.get("index")->num)];
            if (const auto* n = t.get("function")->get("name")) cl.first = n->str;
            cl.second += t.get("function")->get("arguments")->str;
        }
    }
    VJ_CHECK(done);
    for (auto& [i, c] : calls) p.calls.push_back(c);
    return p;
}

// Parses an Anthropic event stream and checks its grammar: message_start, ping, then blocks each opened by
// content_block_start, filled by deltas of the matching type, closed by content_block_stop with consecutive
// indices, then one message_delta and message_stop last.
Parts parse_anthropic_sse(const std::string& body, std::string* stop_sequence = nullptr, uint32_t* out_tokens = nullptr) {
    Parts p;
    std::vector<std::pair<std::string, std::string>> events;
    for (size_t pos = 0; pos < body.size();) {
        const size_t e = body.find("\n\n", pos);
        VJ_CHECK(e != std::string::npos);
        const std::string ev = body.substr(pos, e - pos);
        pos = e + 2;
        VJ_CHECK(ev.rfind("event: ", 0) == 0);
        const size_t nl = ev.find('\n');
        VJ_CHECK(nl != std::string::npos && ev.compare(nl + 1, 6, "data: ") == 0);
        events.emplace_back(ev.substr(7, nl - 7), ev.substr(nl + 7));
    }
    VJ_CHECK(events.size() >= 4 && events[0].first == "message_start" && events[1].first == "ping");
    VJ_CHECK(events.back().first == "message_stop" && events[events.size() - 2].first == "message_delta");
    const json::Value start = parse(events[0].second);
    VJ_CHECK(start.get("type")->str == "message_start" && start.get("message")->get("id")->str.rfind("msg_", 0) == 0 &&
             start.get("message")->get("content")->arr.empty() && start.get("message")->get("stop_reason")->is_null());
    int open = -1;
    std::string open_type, json_acc, name;
    int next_index = 0;
    for (size_t i = 2; i + 2 < events.size(); ++i) {
        const json::Value d = parse(events[i].second);
        VJ_CHECK(d.get("type")->str == events[i].first);
        const int idx = static_cast<int>(d.get("index")->num);
        if (events[i].first == "content_block_start") {
            VJ_CHECK(open < 0 && idx == next_index);
            open = idx;
            ++next_index;
            const json::Value* cb = d.get("content_block");
            open_type = cb->get("type")->str;
            if (open_type == "tool_use") {
                VJ_CHECK(cb->get("id")->str.rfind("toolu_", 0) == 0 && cb->get("input")->is_object());
                name = cb->get("name")->str;
                json_acc.clear();
            }
        } else if (events[i].first == "content_block_delta") {
            VJ_CHECK(idx == open);
            const json::Value* dl = d.get("delta");
            const std::string& t = dl->get("type")->str;
            if (open_type == "text") {
                VJ_CHECK(t == "text_delta");
                p.content += dl->get("text")->str;
            } else if (open_type == "thinking") {
                VJ_CHECK(t == "thinking_delta");
                p.reasoning += dl->get("thinking")->str;
            } else {
                VJ_CHECK(t == "input_json_delta");
                json_acc += dl->get("partial_json")->str;
            }
        } else {
            VJ_CHECK(events[i].first == "content_block_stop" && idx == open);
            if (open_type == "tool_use") p.calls.emplace_back(name, toolcall::dump_json(parse(json_acc), false));
            open = -1;
        }
    }
    VJ_CHECK(open < 0);
    const json::Value md = parse(events[events.size() - 2].second);
    p.finish = md.get("delta")->get("stop_reason")->str;
    if (stop_sequence) *stop_sequence = md.get("delta")->get("stop_sequence")->is_null() ? "" : md.get("delta")->get("stop_sequence")->str;
    if (out_tokens) *out_tokens = static_cast<uint32_t>(md.get("usage")->get("output_tokens")->num);
    return p;
}

void test_anthropic_server() {
    Script gen;
    serve::ServerOptions opt;
    opt.port = 0;
    opt.model_id = "vj-test";
    serve::OpenAiServer srv(opt, [&](const serve::GenerateRequest& r, const serve::TokenSink& s) { return gen(r, s); });
    std::string err;
    VJ_CHECK(srv.start(&err));
    const uint16_t port = srv.port();
    const std::string tools_oa =
        R"("tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}},{"type":"function","function":{"name":"search","parameters":{"type":"object"}}}])";
    const std::string tools_an =
        R"("tools":[{"name":"get_weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}}}},{"name":"search"}])";
    const std::vector<std::string> outputs = {
        "<think>User wants Paris weather.</think>\n\nLet me check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": "
        "{\"city\": \"Paris \xE2\x82\xAC \\\"quoted\\\"\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"search\", \"arguments\": {}}\n</tool_call>",
        "Plain answer with unicode \xE4\xB8\xAD\xE6\x96\x87 and \"quotes\".",
        "<think>only thinking</think>",
    };
    int parity = 0;
    for (const auto& out : outputs) {
        {
            std::lock_guard<std::mutex> lk(gen.mu);
            gen.text = out;
        }
        const Parts a = parse_openai_sse(
            post(port, "/v1/chat/completions", R"({"stream":true,"messages":[{"role":"user","content":"q"}],)" + tools_oa + "}").body);
        uint32_t ntok = 0;
        const Resp rs = post(port, "/v1/messages", R"({"stream":true,"max_tokens":1000,"messages":[{"role":"user","content":"q"}],)" + tools_an + "}");
        VJ_CHECK(rs.status == 200);
        const Parts b = parse_anthropic_sse(rs.body, nullptr, &ntok);
        // schema parity: the same text, reasoning and tool calls through both APIs
        if (!(a.content == b.content && a.reasoning == b.reasoning && a.calls == b.calls)) {
            std::fprintf(stderr, "openai: C[%s] R[%s] %zu calls\nanthropic: C[%s] R[%s] %zu calls\n", a.content.c_str(),
                         a.reasoning.c_str(), a.calls.size(), b.content.c_str(), b.reasoning.c_str(), b.calls.size());
            for (size_t i = 0; i < std::min(a.calls.size(), b.calls.size()); ++i)
                std::fprintf(stderr, "  %s %s | %s %s\n", a.calls[i].first.c_str(), a.calls[i].second.c_str(),
                             b.calls[i].first.c_str(), b.calls[i].second.c_str());
        }
        VJ_CHECK(a.content == b.content && a.reasoning == b.reasoning && a.calls == b.calls);
        VJ_CHECK((a.finish == "tool_calls") == (b.finish == "tool_use"));
        VJ_CHECK(a.finish != "stop" || b.finish == "end_turn");
        VJ_CHECK(ntok > 0);
        // non-streaming Anthropic response carries the same blocks
        const Resp rn = post(port, "/v1/messages", R"({"max_tokens":1000,"messages":[{"role":"user","content":"q"}],)" + tools_an + "}");
        VJ_CHECK(rn.status == 200);
        const json::Value m = parse(rn.body);
        Parts c;
        for (const auto& blk : m.get("content")->arr) {
            const std::string& t = blk.get("type")->str;
            if (t == "text") c.content += blk.get("text")->str;
            else if (t == "thinking") c.reasoning += blk.get("thinking")->str;
            else c.calls.emplace_back(blk.get("name")->str, toolcall::dump_json(*blk.get("input"), false));
        }
        VJ_CHECK(c.content == b.content && c.reasoning == b.reasoning && c.calls == b.calls &&
                 m.get("stop_reason")->str == b.finish && m.get("usage")->get("input_tokens")->num == 11);
        ++parity;
    }
    {  // the tool call made it through with the escaped quotes and the euro sign intact
        std::lock_guard<std::mutex> lk(gen.mu);
        gen.text = outputs[0];
    }
    const Parts b0 = parse_anthropic_sse(post(port, "/v1/messages", R"({"stream":true,"messages":[{"role":"user","content":"q"}],)" + tools_an + "}").body);
    VJ_CHECK(b0.calls.size() == 2 && parse(b0.calls[0].second).get("city")->str == "Paris \xE2\x82\xAC \"quoted\"");
    VJ_CHECK(gen.last.tools.tools.size() == 2 && gen.last.max_tokens == 256);
    // stop sequences and max_tokens
    {
        std::lock_guard<std::mutex> lk(gen.mu);
        gen.text = "one two END three";
    }
    std::string seq;
    Parts s = parse_anthropic_sse(post(port, "/v1/messages", R"({"stream":true,"stop_sequences":["NOPE","END"],"messages":[{"role":"user","content":"q"}]})").body, &seq);
    VJ_CHECK(s.content == "one two " && s.finish == "stop_sequence" && seq == "END");
    const json::Value ns = parse(post(port, "/v1/messages", R"({"stop_sequences":["END"],"messages":[{"role":"user","content":"q"}]})").body);
    VJ_CHECK(ns.get("stop_reason")->str == "stop_sequence" && ns.get("stop_sequence")->str == "END");
    s = parse_anthropic_sse(post(port, "/v1/messages", R"({"stream":true,"max_tokens":3,"messages":[{"role":"user","content":"q"}]})").body);
    VJ_CHECK(s.finish == "max_tokens");
    // errors in the Anthropic shape (and still OpenAI-shaped on the OpenAI route)
    const Resp e1 = post(port, "/v1/messages", R"({"messages":[{"role":"user","content":[{"type":"image"}]}]})");
    VJ_CHECK(e1.status == 400 && parse(e1.body).get("type")->str == "error" &&
             parse(e1.body).get("error")->get("type")->str == "invalid_request_error");
    const Resp e2 = post(port, "/v1/messages", "{nope");
    VJ_CHECK(e2.status == 400 && parse(e2.body).get("error")->get("type")->str == "invalid_request_error");
    const Resp e3 = post(port, "/v1/chat/completions", R"({"messages":[]})");
    VJ_CHECK(e3.status == 400 && parse(e3.body).get("error")->get("param"));
    srv.stop();
    std::printf("[anthropic] live server: SSE schema + parity with /v1/chat/completions on %d outputs, stop_sequence, "
                "max_tokens, errors OK\n",
                parity);
}

// ======================================================================================================== swarm

void test_swarm_routing() {
    using swarm::SwarmRouter;
    using swarm::Status;
    const std::vector<swarm::WorkerSpec> three = {{"w0", 1}, {"w1", 1}, {"w2", 1}};
    auto conv = [](int c, int turns) {  // a conversation's rendered prompt: shared system + per-conversation turns
        std::string s(600, 'S');
        for (int t = 0; t < turns; ++t) s += "conv" + std::to_string(c) + "-turn" + std::to_string(t) + std::string(1000, 'x');
        return swarm::block_hashes(s, 256);
    };
    {  // least busy: three cold, different conversations spread over three workers; the fourth queues
        swarm::RouterConfig cfg;
        cfg.affinity_wait_ms = 0;
        SwarmRouter r(three, cfg);
        std::set<int> used;
        std::vector<uint64_t> t;
        for (int c = 0; c < 3; ++c) {
            t.push_back(r.submit(conv(c, 1)));
            const int w = r.wait(t.back(), std::chrono::milliseconds(0));
            VJ_CHECK(w >= 0);
            used.insert(w);
        }
        VJ_CHECK(used.size() == 3);
        const uint64_t t4 = r.submit(conv(3, 1));
        VJ_CHECK(r.status(t4) == Status::Queued && r.queued() == 1);
        r.release(t[1]);
        int w4 = -1;
        VJ_CHECK(r.status(t4, &w4) == Status::Assigned && w4 == r.wait(t4, std::chrono::milliseconds(0)));
        VJ_CHECK(r.check_invariants());
    }
    {  // load before affinity on multi-slot workers, unless the warm worker is far ahead
        swarm::RouterConfig cfg;
        cfg.affinity_wait_ms = 0;
        SwarmRouter r({{"x", 2}, {"y", 2}}, cfg);
        const uint64_t t1 = r.submit(conv(0, 1));
        const int wx = r.wait(t1, std::chrono::milliseconds(0));
        const uint64_t t2 = r.submit(conv(1, 1));  // shares only the system prompt with wx: the idle worker
        const int wy = r.wait(t2, std::chrono::milliseconds(0));
        VJ_CHECK(wx >= 0 && wy >= 0 && wy != wx);
        const uint64_t t3 = r.submit(conv(0, 2));  // equal load: the warm worker
        VJ_CHECK(r.wait(t3, std::chrono::milliseconds(0)) == wx);
        r.release(t2);
        const uint64_t t4 = r.submit(conv(1, 2));  // wx is full; its own warm worker wy is idle
        VJ_CHECK(r.wait(t4, std::chrono::milliseconds(0)) == wy);
        r.release(t1);
        const uint64_t t5 = r.submit(conv(0, 3));  // wx (1 active, 6+ blocks warm) beats an equally loaded cold match
        VJ_CHECK(r.wait(t5, std::chrono::milliseconds(0)) == wx);
        for (uint64_t t : {t3, t4, t5}) r.release(t);
        VJ_CHECK(r.check_invariants());
    }
    {  // prefix affinity: the next turn of a conversation returns to the worker that holds its prefix
        SwarmRouter r(three);
        std::map<int, int> home;
        for (int c = 0; c < 3; ++c) {
            const uint64_t t = r.submit(conv(c, 1));
            home[c] = r.wait(t, std::chrono::milliseconds(0));
            r.release(t);
        }
        for (int round = 2; round < 6; ++round)
            for (int c = 2; c >= 0; --c) {
                const uint64_t t = r.submit(conv(c, round));
                const int w = r.wait(t, std::chrono::milliseconds(1000));
                VJ_CHECK(w == home[c]);
                VJ_CHECK(r.matched_blocks(t) >= 3);
                r.release(t);
            }
        VJ_CHECK(r.stats().affinity_hits >= 12 && r.check_invariants());
    }
    {  // affinity wait: a busy home worker is waited for (bounded), others are not held up
        swarm::RouterConfig cfg;
        cfg.affinity_wait_ms = 150;
        SwarmRouter r(three, cfg);
        const uint64_t a = r.submit(conv(0, 1));
        const int home = r.wait(a, std::chrono::milliseconds(0));
        const uint64_t b = r.submit(conv(0, 2));  // same conversation while its home is busy
        VJ_CHECK(r.status(b) == Status::Queued);
        const uint64_t c = r.submit(conv(1, 1));  // unrelated: goes to an idle worker at once
        const int wc = r.wait(c, std::chrono::milliseconds(0));
        VJ_CHECK(wc >= 0 && wc != home);
        std::thread rel([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            r.release(a);
        });
        VJ_CHECK(r.wait(b, std::chrono::milliseconds(2000)) == home);  // got its warm worker back
        rel.join();
        r.release(b);
        r.release(c);
        // the wait is bounded: with the home worker held, the request falls back after affinity_wait_ms
        const uint64_t d = r.submit(conv(0, 3));
        VJ_CHECK(r.wait(d, std::chrono::milliseconds(0)) == home);
        const uint64_t e = r.submit(conv(0, 4));
        const auto t0 = Clock::now();
        const int we = r.wait(e, std::chrono::milliseconds(3000));
        const double waited_ms = us_since(t0) / 1000.0;
        VJ_CHECK(we >= 0 && we != home && waited_ms >= 100 && waited_ms < 2000);
        VJ_CHECK(r.stats().affinity_waits >= 2);
        r.release(d);
        r.release(e);
        VJ_CHECK(r.check_invariants());
    }
    {  // cancellation, priorities, health, bounded queue
        swarm::RouterConfig cfg;
        cfg.max_queue = 2;
        cfg.affinity_wait_ms = 0;
        SwarmRouter r(three, cfg);
        std::vector<uint64_t> run;
        for (int c = 0; c < 3; ++c) run.push_back(r.submit(conv(c, 1)));
        const uint64_t lo = r.submit(conv(10, 1), 0);
        const uint64_t hi = r.submit(conv(11, 1), 5);
        VJ_CHECK(r.submit(conv(12, 1)) == 0 && r.stats().rejected == 1);  // queue full
        VJ_CHECK(r.cancel(lo) && r.status(lo) == Status::Unknown && r.cancelled(lo) && r.queued() == 1);
        // cancelling a running request frees its slot at once; the queued high-priority request takes it
        int w0 = -1;
        r.status(run[0], &w0);
        VJ_CHECK(r.cancel(run[0]) && r.cancelled(run[0]));
        int wh = -1;
        VJ_CHECK(r.status(hi, &wh) == Status::Assigned && wh == w0);
        r.release(run[0]);  // releasing a cancelled ticket is a no-op
        VJ_CHECK(r.check_invariants() && !r.cancel(run[0]));
        // priority order in the queue
        const uint64_t p0 = r.submit(conv(20, 1), 0);
        const uint64_t p9 = r.submit(conv(21, 1), 9);
        r.release(run[1]);
        VJ_CHECK(r.status(p9) == Status::Assigned && r.status(p0) == Status::Queued);
        // an unhealthy worker gets nothing and forgets its prefixes
        int wp9 = -1;
        r.status(p9, &wp9);
        r.set_healthy(static_cast<size_t>(wp9), false);
        r.release(p9);
        VJ_CHECK(r.status(p0) == Status::Queued);  // its worker is idle now, but unhealthy
        int w2 = -1;
        r.status(run[2], &w2);
        r.release(run[2]);
        int wp0 = -1;
        VJ_CHECK(r.status(p0, &wp0) == Status::Assigned && wp0 == w2 && wp0 != wp9);
        VJ_CHECK(r.workers()[static_cast<size_t>(wp9)].prefixes == 0);
        r.release(p0);
        r.release(hi);
        for (size_t i = 0; i < 3; ++i) r.set_healthy(i, false);
        VJ_CHECK(r.submit(conv(30, 1)) == 0);  // nobody healthy
        VJ_CHECK(r.check_invariants());
    }
    {  // concurrent stress: 8 client threads, random cancels, invariants hold, every ticket resolves
        swarm::RouterConfig cfg;
        cfg.max_queue = 1000;
        cfg.affinity_wait_ms = 2;
        SwarmRouter r({{"a", 1}, {"b", 2}, {"c", 1}}, cfg);
        std::atomic<int> served{0}, cancelled{0}, bad{0};
        std::vector<std::thread> th;
        for (int t = 0; t < 8; ++t)
            th.emplace_back([&, t] {
                std::mt19937 rng(static_cast<uint32_t>(t));
                for (int i = 0; i < 150; ++i) {
                    const uint64_t k = r.submit(conv(static_cast<int>(rng() % 6), 1 + static_cast<int>(rng() % 3)));
                    if (!k) {
                        ++bad;
                        continue;
                    }
                    if (rng() % 7 == 0) {
                        r.cancel(k);
                        ++cancelled;
                        r.release(k);
                        continue;
                    }
                    const int w = r.wait(k, std::chrono::milliseconds(5000));
                    if (w < 0) ++bad;
                    std::this_thread::sleep_for(std::chrono::microseconds(50 + rng() % 200));  // "generating"
                    if (!r.check_invariants()) ++bad;
                    if (rng() % 11 == 0) {
                        r.cancel(k);
                        ++cancelled;
                    } else {
                        ++served;
                    }
                    r.release(k);
                }
            });
        for (auto& x : th) x.join();
        VJ_CHECK(bad == 0 && served + cancelled == 1200 && r.queued() == 0 && r.check_invariants());
        uint64_t total = 0;
        int used = 0;
        for (const auto& w : r.workers()) {
            VJ_CHECK(w.active == 0);
            total += w.served;
            used += w.served > 0;
        }
        if (!(total == r.stats().assigned && used >= 2)) std::fprintf(stderr, "total %llu assigned %llu used %d\n", (unsigned long long)total, (unsigned long long)r.stats().assigned, used);
        VJ_CHECK(total == r.stats().assigned && used >= 2);
    }
    std::printf("[swarm] 3-worker routing: spread, affinity, bounded affinity wait, cancel, priority, health, queue "
                "bound, 1200-request concurrent stress OK\n");
}

void test_swarm_e2e() {
    // three independent N=1 serve instances; the router picks, the client talks to the chosen worker directly
    struct Slow {
        std::atomic<int> active{0}, max_active{0};
    };
    std::vector<std::unique_ptr<serve::OpenAiServer>> servers;
    std::vector<uint16_t> ports;
    Slow slow[3];
    for (int i = 0; i < 3; ++i) {
        serve::ServerOptions o;
        o.port = 0;
        o.model_id = "worker-" + std::to_string(i);
        Slow* s = &slow[i];
        servers.push_back(std::make_unique<serve::OpenAiServer>(
            o, [s, i](const serve::GenerateRequest&, const serve::TokenSink& sink) {
                const int a = ++s->active;
                for (int m = s->max_active; a > m && !s->max_active.compare_exchange_weak(m, a);) {}
                for (int k = 0; k < 5; ++k) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(4));
                    if (!sink("w" + std::to_string(i) + " ")) break;
                }
                --s->active;
                return serve::GenerateResult{};
            }));
        std::string err;
        VJ_CHECK(servers.back()->start(&err));
        ports.push_back(servers.back()->port());
    }
    swarm::SwarmRouter router({{"worker-0", 1}, {"worker-1", 1}, {"worker-2", 1}});
    std::mutex mu;
    std::map<int, std::set<std::string>> conv_models;  // conversation -> models that served it (turns 2+)
    std::atomic<int> ok{0};
    std::vector<std::thread> clients;
    for (int c = 0; c < 6; ++c)
        clients.emplace_back([&, c] {
            std::string history = R"({"role":"system","content":")" + std::string(400, 'S') + R"("})";
            for (int turn = 0; turn < 4; ++turn) {
                history += R"(,{"role":"user","content":"conversation )" + std::to_string(c) + " turn " +
                           std::to_string(turn) + std::string(1500, '.') + R"("})";
                const uint64_t t = router.submit(swarm::block_hashes(history, 256));
                VJ_CHECK(t);
                const int w = router.wait(t, std::chrono::milliseconds(10000));
                VJ_CHECK(w >= 0);
                const Resp r = post(ports[static_cast<size_t>(w)], "/v1/chat/completions", R"({"messages":[)" + history + "]}");
                router.release(t);
                VJ_CHECK(r.status == 200);
                const json::Value v = parse(r.body);
                const std::string model = v.get("model")->str;
                VJ_CHECK(model == "worker-" + std::to_string(w));  // the request really ran on the routed worker
                const std::string reply = v.get("choices")->arr[0].get("message")->get("content")->str;
                history += R"(,{"role":"assistant","content":)" + json::quote(reply) + "}";
                if (turn > 0) {
                    std::lock_guard<std::mutex> lk(mu);
                    conv_models[c].insert(model);
                }
                ++ok;
            }
        });
    for (auto& t : clients) t.join();
    int sticky = 0;
    for (const auto& [c, models] : conv_models) sticky += models.size() == 1;
    for (int i = 0; i < 3; ++i) VJ_CHECK(slow[i].max_active <= 1);  // never two requests on an N=1 worker
    const auto st = router.stats();
    VJ_CHECK(ok == 24 && router.check_invariants() && st.completed == 24);
    VJ_CHECK(sticky >= 4);  // most conversations keep their worker for all later turns (6 clients, 3 workers)
    for (auto& s : servers) s->stop();
    std::printf("[swarm] e2e over 3 loopback serve workers: 24 requests from 6 conversations, never 2 in flight per "
                "worker, %d/6 conversations stayed on one worker, %llu affinity hits\n",
                sticky, static_cast<unsigned long long>(st.affinity_hits));
}

} // namespace

int main() {
    test_gbnf();
    test_schema();
    test_masks();
    test_mask_latency();
    test_tool_parser();
    test_anthropic_translate();
    test_anthropic_server();
    test_swarm_routing();
    test_swarm_e2e();
    std::printf("test_grammar_api_swarm: all passed\n");
    return 0;
}
