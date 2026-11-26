// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Pre-tokenizer families (parity with the published llama.cpp regexes), SentencePiece encode / decode, the Jinja
// chat template engine (parity with the hand-written ChatML / Qwen3 formatters, tools, reasoning), and the sampler
// stack (reference parity, penalties, DRY, XTC, Mirostat v2, logprobs, determinism, 248K-vocabulary latency).
#include "core/bpe_tokenizer.hpp"
#include "core/chat_template.hpp"
#include "core/pretok_families.hpp"
#include "core/sampler.hpp"
#include "core/sampler_stack.hpp"
#include "tests/test_common.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <time.h>

using namespace vegajet;
using bpe::PreTok;

namespace {

struct Golden {
    PreTok fam;
    const char* text;
    std::vector<std::string> chunks;
};
const Golden kGoldens[] = {
#include "tests/pretok_goldens.inc"
};

void test_pretok_parity() {
    int fails[8] = {0}, total[8] = {0};
    for (const auto& g : kGoldens) {
        const auto got = bpe::Tokenizer::pretokenize(g.text, g.fam);
        const auto f = static_cast<size_t>(g.fam);
        ++total[f];
        if (got != g.chunks) {
            ++fails[f];
            std::fprintf(stderr, "pretok mismatch [%s] \"%s\":\n  want:", bpe::pretok_name(g.fam), g.text);
            for (const auto& c : g.chunks) std::fprintf(stderr, " [%s]", c.c_str());
            std::fprintf(stderr, "\n  got: ");
            for (const auto& c : got) std::fprintf(stderr, " [%s]", c.c_str());
            std::fprintf(stderr, "\n");
        }
    }
    std::printf("Pre-tokenizer parity vs the llama.cpp regexes (tests/pretok_goldens.inc)\n");
    int all_fail = 0;
    for (size_t f = 0; f < 8; ++f) {
        if (!total[f]) continue;
        std::printf("  %-15s %2d / %2d strings identical\n", bpe::pretok_name(static_cast<PreTok>(f)), total[f] - fails[f], total[f]);
        all_fail += fails[f];
    }
    VJ_CHECK(all_fail == 0);
    // Every split concatenates back to the input, invalid UTF-8 included.
    for (PreTok f : {PreTok::DeepSeekLlm, PreTok::DeepSeekCoder, PreTok::DeepSeekV3, PreTok::Tekken}) {
        const std::string s = "a\xFF\xFE b\xC3 x\xE4\xBD";
        std::string cat;
        for (const auto& c : bpe::Tokenizer::pretokenize(s, f)) cat += c;
        VJ_CHECK(cat == s);
        VJ_CHECK(bpe::Tokenizer::pretokenize("", f).empty());
    }
}

void test_pretok_dispatch() {
    VJ_CHECK(bpe::pretok_from_gguf("qwen2", "gpt2") == PreTok::Qwen2);
    VJ_CHECK(bpe::pretok_from_gguf("qwen35", "gpt2") == PreTok::Qwen2);
    VJ_CHECK(bpe::pretok_from_gguf("llama-bpe", "gpt2") == PreTok::Llama3);
    VJ_CHECK(bpe::pretok_from_gguf("falcon3", "gpt2") == PreTok::Llama3);
    VJ_CHECK(bpe::pretok_from_gguf("deepseek-llm", "gpt2") == PreTok::DeepSeekLlm);
    VJ_CHECK(bpe::pretok_from_gguf("deepseek-coder", "gpt2") == PreTok::DeepSeekCoder);
    VJ_CHECK(bpe::pretok_from_gguf("deepseek-v3", "gpt2") == PreTok::DeepSeekV3);
    VJ_CHECK(bpe::pretok_from_gguf("tekken", "gpt2") == PreTok::Tekken);
    VJ_CHECK(bpe::pretok_from_gguf("", "llama") == PreTok::Spm);       // Gemma / Llama-2 / Mistral v1 SentencePiece
    VJ_CHECK(bpe::pretok_from_gguf("default", "gpt2") == PreTok::Gpt2);
    VJ_CHECK(bpe::pretok_from_gguf("deepseek-v3") == PreTok::DeepSeekV3);  // no longer folded into qwen2
    // A loaded byte-level vocabulary dispatches on its own `pre`.
    bpe::Tokenizer probe;
    bpe::Vocab bytes;
    // Build the byte-symbol strings the same way the tokenizer does (GPT-2 byte -> unicode table).
    uint32_t extra = 0;
    for (uint32_t b = 0; b < 256; ++b) {
        const bool direct = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        const uint32_t cp = direct ? b : 256 + extra++;
        std::string u;
        if (cp < 0x80) u += static_cast<char>(cp);
        else { u += static_cast<char>(0xC0 | (cp >> 6)); u += static_cast<char>(0x80 | (cp & 0x3F)); }
        bytes.tokens.push_back(u);
    }
    for (const char* pre : {"tekken", "deepseek-v3", "deepseek-llm", "deepseek-coder", "qwen2"}) {
        bpe::Vocab vv = bytes;
        vv.pre = pre;
        VJ_CHECK(probe.load(vv));
        VJ_CHECK(probe.pretokenizer() == bpe::pretok_from_gguf(pre, "gpt2"));
        const std::string s = "Hello WORLD 12345 \xE4\xBD\xA0\xE5\xA5\xBD!";
        VJ_CHECK(probe.decode(probe.encode(s)) == s);
    }
}

void test_spm() {
    // Tiny SentencePiece vocabulary: control tokens, byte fallback, scored pieces.
    bpe::Vocab v;
    v.model = "llama";
    v.add_space_prefix = false;  // Gemma
    auto add = [&](const std::string& t, float score, int type) {
        v.tokens.push_back(t);
        v.scores.push_back(score);
        v.types.push_back(type);
    };
    add("<unk>", 0, 2);
    add("<bos>", 0, 3);
    for (int b = 0; b < 256; ++b) {
        char n[8];
        std::snprintf(n, sizeof n, "<0x%02X>", b);
        add(n, 0, 6);
    }
    const std::string sp = "\xE2\x96\x81";
    for (const char* t : {"h", "e", "l", "o", "w", "r", "d"}) add(t, -10, 1);
    add(sp, -9, 1);
    add("he", -3, 1);
    add("ll", -2, 1);
    add("llo", -1.5f, 1);
    add("hello", -1, 1);
    add(sp + "w", -4, 1);
    add(sp + "wo", -3.8f, 1);
    add(sp + "wor", -3.5f, 1);
    add(sp + "worl", -0.8f, 1);
    add("or", -5, 1);
    add(sp + "world", -0.5f, 1);
    v.unk = 0;
    v.bos = 1;
    bpe::Tokenizer t;
    std::string err;
    VJ_CHECK(t.load(v, &err));
    VJ_CHECK(t.pretokenizer() == PreTok::Spm);
    const auto ids = t.encode("hello world");
    VJ_CHECK(ids.size() == 2);
    VJ_CHECK(t.token_id("hello") == ids[0] && t.token_id(sp + "world") == ids[1]);
    VJ_CHECK(t.decode(ids) == "hello world");
    // Byte fallback: U+00E9 is not in the vocabulary -> <0xC3><0xA9>.
    const auto e = t.encode("h\xC3\xA9");
    VJ_CHECK(e.size() == 3 && e[1] == t.token_id("<0xC3>") && e[2] == t.token_id("<0xA9>"));
    VJ_CHECK(t.decode(e) == "h\xC3\xA9");
    // Llama-2 style: a leading U+2581 is added to the text start.
    v.add_space_prefix = true;
    VJ_CHECK(t.load(v));
    const auto w = t.encode("world");
    VJ_CHECK(w.size() == 1 && w[0] == t.token_id(sp + "world"));
    // Scores are required for SentencePiece; other models are still rejected.
    bpe::Vocab bad = v;
    bad.scores.clear();
    VJ_CHECK(!t.load(bad, &err) && err.find("scores") != std::string::npos);
    bad = v;
    bad.model = "bert";
    VJ_CHECK(!t.load(bad, &err));
    VJ_CHECK(pretok::spm_escape("a b", true) == sp + "a" + sp + "b");
}

// ------------------------------------------------------------------------------------------------- templates
std::string render(const char* src, const std::vector<chat::ChatMessage>& msgs, const chat::RenderOptions& opt) {
    chat::Template t;
    std::string err, out;
    if (!t.parse(src, &err)) { std::fprintf(stderr, "parse: %s\n", err.c_str()); VJ_CHECK(false); }
    if (!t.render(msgs, opt, out, &err)) { std::fprintf(stderr, "render: %s\n", err.c_str()); VJ_CHECK(false); }
    return out;
}

std::string jinja(const char* src, const char* vars = "{}") {
    chat::Template t;
    std::string err, out;
    if (!t.parse(src, &err)) { std::fprintf(stderr, "parse: %s\n", err.c_str()); VJ_CHECK(false); }
    if (!t.render_json(vars, out, &err)) { std::fprintf(stderr, "render: %s\n", err.c_str()); VJ_CHECK(false); }
    return out;
}

bool jinja_fails(const char* src, const char* needle, const char* vars = "{}") {
    chat::Template t;
    std::string err, out;
    if (t.parse(src, &err) && t.render_json(vars, out, &err)) return false;
    return err.find(needle) != std::string::npos;
}

void test_template_parity() {
    using M = chat::ChatMessage;
    const std::vector<std::vector<M>> convos = {
        {{"user", "Hi"}},
        {{"system", "You are terse."}, {"user", "2+2?"}},
        {{"system", "S"}, {"user", "a"}, {"assistant", "<think>\nhmm\n</think>\n\nb"}, {"user", "c"}},
        {{"user", "one"}, {"assistant", "two"}, {"user", "three"}, {"assistant", "four"}, {"user", "five"}},
        {{"user", "multi\nline\n  content"}},
    };
    for (const auto& c : convos) {
        std::vector<bpe::Message> bm;
        for (const auto& m : c) bm.push_back({m.role, m.content});
        for (bool gen : {true, false}) {
            bpe::ChatOptions bo;
            bo.add_generation_prompt = gen;
            bo.enable_thinking = false;
            chat::RenderOptions ro;
            ro.add_generation_prompt = gen;
            ro.enable_thinking = 0;
            VJ_CHECK(render(chat::builtin_template("qwen3"), c, ro) == bpe::apply_chat_template(bpe::ChatFormat::Qwen3, bm, bo));
            VJ_CHECK(render(chat::builtin_template("chatml"), c, ro) == bpe::apply_chat_template(bpe::ChatFormat::ChatML, bm, bo));
        }
    }
    std::printf("Chat template parity: qwen3 + chatml Jinja == hand-written formatters on %zu conversations x 2\n", convos.size());
}

void test_template_tools() {
    using M = chat::ChatMessage;
    M call{"assistant", ""};
    call.tool_calls.push_back({"call_1", "get_weather", "{\"city\": \"Paris\", \"days\": 2}"});
    M result{"tool", "{\"temp\": 21}"};
    result.tool_call_id = "call_1";
    const std::vector<M> msgs = {{"system", "Be brief."}, {"user", "Weather in Paris?"}, call, result};
    chat::RenderOptions ro;
    ro.tools_json = R"([{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}])";
    const std::string out = render(chat::builtin_template("qwen3"), msgs, ro);
    const std::string want =
        "<|im_start|>system\nBe brief.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
        "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"parameters\": {\"type\": \"object\"}}}\n</tools>\n\n"
        "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n"
        "<|im_start|>user\nWeather in Paris?<|im_end|>\n"
        // not the last message and no reasoning: Qwen3 renders the call without a <think> block
        "<|im_start|>assistant\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"days\": 2}}\n</tool_call><|im_end|>\n"
        "<|im_start|>user\n<tool_response>\n{\"temp\": 21}\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n";
    if (out != want) std::fprintf(stderr, "qwen3 tools render:\n%s\n---\n%s\n", out.c_str(), want.c_str());
    VJ_CHECK(out == want);
    // Llama 3: tools as indented JSON, tool calls, ipython results, trim filter precedence.
    chat::RenderOptions lo = ro;
    lo.bos_token = "<|begin_of_text|>";
    const std::string l3 = render(chat::builtin_template("llama3"), msgs, lo);
    VJ_CHECK(l3.rfind("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nEnvironment: ipython\n", 0) == 0);
    VJ_CHECK(l3.find("Today Date: 26 Jul 2024") != std::string::npos);
    VJ_CHECK(l3.find("{\n    \"type\": \"function\",\n    \"function\": {\n        \"name\": \"get_weather\",") != std::string::npos);
    VJ_CHECK(l3.find("{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\", \"days\": 2}}<|eot_id|>") != std::string::npos);
    VJ_CHECK(l3.find("<|start_header_id|>ipython<|end_header_id|>\n\n{\"temp\": 21}<|eot_id|>") != std::string::npos);
    VJ_CHECK(l3.size() > 30 && l3.compare(l3.size() - 30, 30, "<|start_header_id|>assistant<|end_header_id|>\n\n") != 0);
    VJ_CHECK(l3.find("<|start_header_id|>assistant<|end_header_id|>\n\n", l3.size() - 50) != std::string::npos);
    // Other built-ins render a plain conversation.
    const std::vector<M> plain = {{"system", "S"}, {"user", "hi"}, {"assistant", "yo"}, {"user", "bye"}};
    chat::RenderOptions po;
    po.bos_token = "<bos>";
    po.eos_token = "</s>";
    VJ_CHECK(render(chat::builtin_template("gemma"), plain, po) ==
             "<bos><start_of_turn>user\nS\n\nhi<end_of_turn>\n<start_of_turn>model\nyo<end_of_turn>\n<start_of_turn>user\nbye<end_of_turn>\n<start_of_turn>model\n");
    VJ_CHECK(render(chat::builtin_template("mistral-tekken"), plain, po) == "<bos>[INST]S\n\nhi[/INST]yo</s>[INST]bye[/INST]");
    VJ_CHECK(render(chat::builtin_template("deepseek-v3"), plain, po) == "<bos>S<｜User｜>hi<｜Assistant｜>yo<｜end▁of▁sentence｜><｜User｜>bye<｜Assistant｜>");
    VJ_CHECK(chat::detect_template_family("{{ '<|start_header_id|>' }}", "") == "llama3");
    VJ_CHECK(chat::detect_template_family("<|im_start|>{% if tools %}<tool_call>{% endif %}", "") == "qwen3");
    VJ_CHECK(chat::detect_template_family("", "gemma3") == "gemma");
    VJ_CHECK(chat::detect_template_family("", "unknown") == "chatml");
    VJ_CHECK(chat::builtin_template("nope") == nullptr);
}

void test_jinja_engine() {
    VJ_CHECK(jinja("{% for x in xs %}{{ loop.index }}:{{ x }}{% if not loop.last %},{% endif %}{% endfor %}", R"({"xs": ["a","b","c"]})") == "1:a,2:b,3:c");
    VJ_CHECK(jinja("{{ xs[::-1] | join('') }}|{{ xs[1:] | length }}|{{ xs[-1] }}|{{ s[:3] }}", R"({"xs": [1,2,3], "s": "abcdef"})") == "321|2|3|abc");
    VJ_CHECK(jinja("{% set ns = namespace(n=0) %}{% for i in range(5) %}{% set ns.n = ns.n + i %}{% endfor %}{{ ns.n }}") == "10");
    VJ_CHECK(jinja("{% set x = 1 %}{% for i in [1] %}{% set x = 2 %}{% endfor %}{{ x }}") == "1");  // loop scope
    VJ_CHECK(jinja("{% macro tag(name, cls='c') %}<{{ name }} {{ cls }}>{% endmacro %}{{ tag('a') }}{{ tag('b', cls='d') }}") == "<a c><b d>");
    VJ_CHECK(jinja("{% for k, v in d.items() %}{{ k }}={{ v }};{% endfor %}", R"({"d": {"a": 1, "b": true}})") == "a=1;b=True;");
    VJ_CHECK(jinja("{{ x is defined }} {{ y is defined }} {{ n is none }} {{ s is string }} {{ 3 is odd }}", R"({"x": 1, "n": null, "s": "q"})") ==
             "True False True True True");
    VJ_CHECK(jinja("{{ 'yes' if a and not b else 'no' }} {{ 'x' in 'xyz' }} {{ 4 not in [1,2] }}", R"({"a": 1, "b": 0})") == "yes True True");
    VJ_CHECK(jinja("{{ ' Hi '|trim|upper }} {{ 'a,b'.split(',')|last }} {{ 'abc'.startswith('ab') }} {{ 'x\\n'.rstrip('\\n') ~ '!' }}") == "HI b True x!");
    VJ_CHECK(jinja("{{ d | tojson }} {{ [1.5, none, 'q\"'] | tojson }}", R"({"d": {"k": [1, 2], "u": "é"}})") ==
             "{\"k\": [1, 2], \"u\": \"é\"} [1.5, null, \"q\\\"\"]");
    VJ_CHECK(jinja("{{ x | default('dflt') }}|{{ '' | default('e', true) }}|{{ 7 // 2 }} {{ 7 % 3 }} {{ 2 ** 3 }} {{ 1 / 2 }}") == "dflt|e|3 1 8 0.5");
    VJ_CHECK(jinja("{{ users | selectattr('admin') | map(attribute='name') | join(',') }}",
                   R"({"users": [{"name": "a", "admin": true}, {"name": "b", "admin": false}, {"name": "c", "admin": true}]})") == "a,c");
    VJ_CHECK(jinja("{% for x in xs if x > 1 %}{{ x }}{% else %}none{% endfor %}", R"({"xs": [1, 2, 3]})") == "23");
    VJ_CHECK(jinja("{% for x in xs %}{{ x }}{% else %}none{% endfor %}", R"({"xs": []})") == "none");
    VJ_CHECK(jinja("{% set t %}in{{ 1 }}{% endset %}[{{ t }}]") == "[in1]");
    // Whitespace: trim_blocks drops the newline after a block tag, lstrip_blocks the indentation before it, `-` trims.
    VJ_CHECK(jinja("{% if true %}\n  yes\n{% endif %}\n") == "  yes\n");
    VJ_CHECK(jinja("a\n    {% if true %}\nb\n    {% endif %}\nc") == "a\nb\nc");
    VJ_CHECK(jinja("x   {{- ' y ' -}}   z") == "x y z");
    VJ_CHECK(jinja("{# comment #}\nv") == "v");
    VJ_CHECK(jinja("{{ 'a' ~ 1 ~ none }}") == "a1None");
    VJ_CHECK(jinja("{% set l = [] %}{% set _ = l.append(3) %}{{ l }}") == "[3]");
    VJ_CHECK(jinja("{{ messages | selectattr('role', 'equalto', 'user') | list | length }}",
                   R"({"messages": [{"role": "user"}, {"role": "assistant"}, {"role": "user"}]})") == "2");
    // Errors: raise_exception, unknown filter, bad syntax, unclosed blocks.
    VJ_CHECK(jinja_fails("{{ raise_exception('Only user/assistant roles') }}", "Only user/assistant roles"));
    VJ_CHECK(jinja_fails("{{ x | frobnicate }}", "unknown filter"));
    VJ_CHECK(jinja_fails("{% if x %}open", "endif"));
    VJ_CHECK(jinja_fails("{{ 1 + }}", "unexpected"));
    VJ_CHECK(jinja_fails("{{ 'abc", "unterminated"));
    VJ_CHECK(jinja_fails("{% while x %}{% endwhile %}", "unsupported tag"));
    VJ_CHECK(jinja_fails("{% macro f() %}{{ f() }}{% endmacro %}{{ f() }}", "recursion"));
    VJ_CHECK(jinja_fails("{{ range(1000000) | length }}", "range"));
}

void test_reasoning() {
    auto r = chat::split_reasoning("<think>\nplan\n</think>\n\nanswer");
    VJ_CHECK(r.reasoning == "plan" && r.content == "answer" && r.closed);
    r = chat::split_reasoning("plan only</think>answer");  // template opened <think> in the prompt
    VJ_CHECK(r.reasoning == "plan only" && r.content == "answer");
    r = chat::split_reasoning("<think>cut off");
    VJ_CHECK(r.reasoning == "cut off" && r.content.empty() && !r.closed);
    r = chat::split_reasoning("plain");
    VJ_CHECK(r.reasoning.empty() && r.content == "plain");
    // Streaming: any chunking gives the same split as the whole text.
    const std::string text = "<think>\nstep 1\nstep <2>\n</think>\n\nfinal <b>answer</b> </think>";
    for (size_t chunk : {1u, 2u, 3u, 5u, 7u, 100u}) {
        chat::ReasoningSplitter sp;
        std::string rs, cs;
        for (size_t i = 0; i < text.size(); i += chunk) sp.feed(std::string_view(text).substr(i, chunk), rs, cs);
        sp.finish(rs, cs);
        VJ_CHECK(rs == "\nstep 1\nstep <2>\n");
        VJ_CHECK(cs == "final <b>answer</b> </think>");
    }
    chat::ReasoningSplitter inside(true);
    std::string rs, cs;
    inside.feed("thinking</th", rs, cs);
    VJ_CHECK(rs == "thinking" && cs.empty());
    inside.feed("ink>done", rs, cs);
    inside.finish(rs, cs);
    VJ_CHECK(rs == "thinking" && cs == "done");
}

// ------------------------------------------------------------------------------------------------- sampler
// Straightforward reference over the full sorted vocabulary (llama.cpp chain semantics), no candidate tricks.
int32_t reference_sample(const std::vector<float>& logits, const sampling::StackParams& P, uint64_t step) {
    struct C { float l; int32_t id; double p; };
    std::vector<C> c;
    for (size_t i = 0; i < logits.size(); ++i) c.push_back({logits[i], static_cast<int32_t>(i), 0});
    std::sort(c.begin(), c.end(), [](const C& a, const C& b) { return a.l > b.l || (a.l == b.l && a.id < b.id); });
    const double m = c[0].l;
    auto softmax = [&](double inv_t) {
        double z = 0;
        for (auto& x : c) { x.p = std::exp((x.l - m) * inv_t); z += x.p; }
        for (auto& x : c) x.p /= z;
    };
    if (P.top_k > 0 && static_cast<size_t>(P.top_k) < c.size()) c.resize(static_cast<size_t>(P.top_k));
    if (P.top_p < 1.0f) {
        softmax(1.0);
        double cum = 0;
        for (size_t i = 0; i < c.size(); ++i) { cum += c[i].p; if (cum >= P.top_p) { c.resize(i + 1); break; } }
    }
    if (P.min_p > 0) {
        const double fl = c[0].l + std::log(static_cast<double>(P.min_p));
        size_t k = 0;
        while (k < c.size() && c[k].l >= fl) ++k;
        c.resize(std::max<size_t>(1, k));
    }
    if (P.temperature <= 0) return c[0].id;
    softmax(1.0 / P.temperature);
    const double u = sampling::counter_uniform(P.seed, step);
    double acc = 0;
    for (const auto& x : c) { acc += x.p; if (u < acc) return x.id; }
    return c.back().id;
}

std::vector<float> random_logits(std::mt19937& g, size_t n, float sigma, int peaks) {
    std::normal_distribution<float> nd(0.0f, sigma);
    std::vector<float> l(n);
    for (auto& x : l) x = nd(g);
    std::uniform_int_distribution<size_t> pick(0, n - 1);
    for (int k = 0; k < peaks; ++k) l[pick(g)] += 8.0f + static_cast<float>(k);
    return l;
}

void test_sampler_reference_parity() {
    std::mt19937 g(7);
    int agree_k = 0, agree_nok = 0, fallbacks = 0;
    const int n = 200;
    for (int t = 0; t < n; ++t) {
        const auto logits = random_logits(g, 4096, 2.0f, t % 5);
        sampling::StackParams p;
        p.seed = 1000 + static_cast<uint64_t>(t);
        p.temperature = 0.7f + 0.01f * static_cast<float>(t % 50);
        p.top_k = 40;
        p.top_p = 0.9f;
        p.min_p = 0.02f;
        sampling::SamplerStack s(p);
        agree_k += s.sample(logits.data(), logits.size()).id == reference_sample(logits, p, 0);
        p.top_k = 0;
        sampling::SamplerStack s2(p);
        const auto r = s2.sample(logits.data(), logits.size());
        fallbacks += r.full_vocab_fallback;
        agree_nok += r.id == reference_sample(logits, p, 0);
    }
    std::printf("Sampler vs full-sort reference: top_k=40 %d/%d identical, top_k=0 %d/%d identical (%d exact fallbacks)\n",
                agree_k, n, agree_nok, n, fallbacks);
    VJ_CHECK(agree_k == n);
    VJ_CHECK(agree_nok >= n - 2);  // top_k=0: the tail normalizer is a 2e-6-accurate polynomial sum
}

void test_sampler_edits() {
    // Penalties + bias, observed through exact logprobs on a flat 8-token vocabulary.
    std::vector<float> flat(8, 0.0f);
    sampling::StackParams p;
    p.temperature = 0;
    p.top_k = 0;
    p.top_p = 1;
    p.min_p = 0;
    p.n_logprobs = 8;
    p.frequency_penalty = 0.5f;
    p.presence_penalty = 1.0f;
    p.repeat_penalty = 2.0f;
    p.logit_bias = {{6, 1.5f}, {7, -INFINITY}};
    sampling::SamplerStack s(p);
    for (int t : {3, 3, 5}) s.accept(t);
    const auto r = s.sample(flat.data(), flat.size());
    VJ_CHECK(r.id == 6);  // bias wins under greedy
    std::vector<double> want = {0, 0, 0, -2.0, 0, -1.5, 1.5, -INFINITY};  // rep on 0 is a no-op; 3: -2*0.5-1; 5: -0.5-1
    double z = 0;
    for (double w : want) z += std::exp(w);
    for (const auto& lp : r.top) {
        if (std::isinf(want[static_cast<size_t>(lp.id)])) { VJ_CHECK(std::isinf(lp.logprob)); continue; }
        VJ_CHECK(std::fabs(lp.logprob - (want[static_cast<size_t>(lp.id)] - std::log(z))) < 2e-5);
    }
    VJ_CHECK(r.top.size() == 8 && r.top[0].id == 6 && std::fabs(r.logprob - r.top[0].logprob) < 1e-6);
    // Banned tokens are never drawn, even at high temperature with a flat distribution.
    sampling::StackParams q;
    q.temperature = 5;
    q.top_k = 0;
    q.top_p = 1;
    q.min_p = 0;
    q.logit_bias = {{2, -INFINITY}};
    for (uint64_t seed = 0; seed < 300; ++seed) {
        q.seed = seed;
        sampling::SamplerStack ss(q);
        VJ_CHECK(ss.sample(flat.data(), flat.size()).id != 2);
    }
}

void test_dry() {
    std::vector<float> flat(16, 0.0f);
    sampling::StackParams p;
    p.temperature = 0;
    p.top_k = 0; p.top_p = 1; p.min_p = 0;
    p.n_logprobs = 16;
    p.dry_multiplier = 0.8f;
    p.dry_base = 1.75f;
    p.dry_allowed_length = 2;
    sampling::SamplerStack s(p);
    for (int t : {1, 2, 3, 4, 9, 1, 2, 3}) s.accept(t);  // "1 2 3" repeats; continuing with 4 extends it
    const auto r = s.sample(flat.data(), flat.size());
    auto lp = [&](int32_t id) { for (const auto& x : r.top) if (x.id == id) return static_cast<double>(x.logprob); return 0.0; };
    // match length 3 -> penalty 0.8 * 1.75^(3-2) = 1.4 on token 4; no other token is penalized
    VJ_CHECK(std::fabs((lp(0) - lp(4)) - 1.4) < 1e-4);
    VJ_CHECK(std::fabs(lp(0) - lp(5)) < 1e-6 && std::fabs(lp(0) - lp(9)) < 1e-6);
    // A breaker inside the repeated run limits the match: "3" is a breaker -> no match reaches allowed length.
    p.dry_breakers = {3};
    sampling::SamplerStack b(p);
    for (int t : {1, 2, 3, 4, 9, 1, 2, 3}) b.accept(t);
    const auto rb = b.sample(flat.data(), flat.size());
    for (const auto& x : rb.top) VJ_CHECK(std::fabs(x.logprob - rb.top[0].logprob) < 1e-6);
}

void test_xtc_mirostat_determinism() {
    // XTC with probability 1: the two tokens above the threshold go except the least likely of them.
    std::vector<float> l = {5.0f, 4.0f, 0.0f, -1.0f, -2.0f};
    sampling::StackParams x;
    x.temperature = 0;  // greedy after XTC: shows which token survived on top
    x.top_k = 0; x.top_p = 1; x.min_p = 0;
    x.xtc_probability = 1.0f;
    x.xtc_threshold = 0.1f;
    sampling::SamplerStack xs(x);
    VJ_CHECK(xs.sample(l.data(), l.size()).id == 1);
    x.xtc_probability = 0;
    sampling::SamplerStack x0(x);
    VJ_CHECK(x0.sample(l.data(), l.size()).id == 0);
    // Mirostat v2: mu starts at 2 tau and moves toward the target surprise; same seed replays exactly.
    std::mt19937 g(3);
    sampling::StackParams m;
    m.mirostat = 2;
    m.mirostat_tau = 3.0f;
    m.mirostat_eta = 0.2f;
    m.temperature = 1.0f;
    m.seed = 99;
    sampling::SamplerStack a(m), b(m);
    VJ_CHECK(a.mirostat_mu() == 6.0f);
    std::vector<int32_t> sa, sb;
    double surprise_sum = 0;
    for (int t = 0; t < 400; ++t) {
        const auto lg = random_logits(g, 2048, 1.5f, 3);
        const auto ra = a.sample(lg.data(), lg.size());
        const auto rb = b.sample(lg.data(), lg.size());
        sa.push_back(ra.id);
        sb.push_back(rb.id);
        a.accept(ra.id);
        b.accept(rb.id);
        surprise_sum += -ra.logprob / std::log(2.0);
    }
    VJ_CHECK(sa == sb);
    std::printf("Mirostat v2: tau 3.0, mean observed surprise %.2f bits over 400 steps, final mu %.2f\n",
                surprise_sum / 400, static_cast<double>(a.mirostat_mu()));
    VJ_CHECK(std::fabs(surprise_sum / 400 - 3.0) < 1.0);
    // Different seeds diverge; the same seed replays a whole sampled sequence.
    sampling::StackParams d;
    d.temperature = 1.0f;
    d.top_k = 0; d.top_p = 1; d.min_p = 0;
    auto run = [&](uint64_t seed) {
        d.seed = seed;
        sampling::SamplerStack s(d);
        std::mt19937 gg(11);
        std::vector<int32_t> out;
        for (int t = 0; t < 64; ++t) {
            const auto lg = random_logits(gg, 1024, 1.0f, 0);
            out.push_back(s.sample(lg.data(), lg.size()).id);
            s.accept(out.back());
        }
        return out;
    };
    VJ_CHECK(run(5) == run(5));
    VJ_CHECK(run(5) != run(6));
}

void test_sum_exp_accuracy() {
    std::mt19937 g(1);
    std::normal_distribution<float> nd(0.0f, 6.0f);
    std::vector<float> x(100003);
    for (auto& v : x) v = nd(g);
    const float m = *std::max_element(x.begin(), x.end());
    double ref = 0;
    for (float v : x) ref += std::exp(static_cast<double>(v) - m);
    const double got = sampling::sum_exp_shifted(x.data(), x.size(), m);
    VJ_CHECK(std::fabs(got / ref - 1.0) < 2e-6);
    const double got_t = sampling::sum_exp_shifted(x.data(), x.size(), m, 0.5f);
    double ref_t = 0;
    for (float v : x) ref_t += std::exp((static_cast<double>(v) - m) * 0.5);
    VJ_CHECK(std::fabs(got_t / ref_t - 1.0) < 2e-6);
    // Thread count never changes the bits (fixed chunking, fixed-order reduction).
    std::vector<float> big(300001);
    for (auto& v : big) v = nd(g);
    const float mb = *std::max_element(big.begin(), big.end());
    const double s1 = sampling::sum_exp_shifted(big.data(), big.size(), mb, 1.0f, 1);
    for (int th : {2, 3, 4, 8}) VJ_CHECK(sampling::sum_exp_shifted(big.data(), big.size(), mb, 1.0f, th) == s1);
}

#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_UNDEFINED__)
constexpr bool kSanitized = true;
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(undefined_behavior_sanitizer)
constexpr bool kSanitized = true;
#else
constexpr bool kSanitized = false;
#endif
#else
constexpr bool kSanitized = false;
#endif

double thread_cpu_us() {
    timespec ts{};
    VJ_CHECK(clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) == 0);
    return static_cast<double>(ts.tv_sec) * 1e6 + static_cast<double>(ts.tv_nsec) / 1e3;
}

void test_latency() {
    // LLM-like logits: a N(0, 1.5) bulk over the vocabulary plus a few dominant tokens that move every step
    // (~1 % tail mass at temperature 1). Frames are prepared outside the timed region.
    const size_t V = 248320;  // Qwen3.5 / Ornith vocabulary
    std::mt19937 g(42);
    std::normal_distribution<float> nd(0.0f, 1.5f);
    std::vector<std::vector<float>> bulk(4, std::vector<float>(V));
    for (auto& f : bulk) for (auto& x : f) x = nd(g);
    std::vector<float> frame(V);
    std::uniform_int_distribution<int32_t> tok(0, static_cast<int32_t>(V) - 1);
    auto bench = [&](sampling::StackParams p, const char* label, int* fallbacks, double work_budget = 0.0) {
        sampling::SamplerStack s(p);
        auto serial_params = p;
        serial_params.threads = 1;
        sampling::SamplerStack serial(serial_params);
        for (int k = 0; k < 512; ++k) {
            const auto id = tok(g);
            s.accept(id);
            serial.accept(id);
        }
        std::vector<double> us, serial_us, reference_us, work_ratios;
        for (int t = 0; t < 300; ++t) {
            frame = bulk[static_cast<size_t>(t) % bulk.size()];
            for (int k = 0; k < 6; ++k) frame[static_cast<size_t>(tok(g))] = 18.0f - 1.5f * static_cast<float>(k);
            sampling::SampleResult serial_result;
            if (work_budget > 0.0) {
                double sample_cpu = 0.0, reference_cpu = 0.0;
                auto sample_serial = [&] {
                    const double start = thread_cpu_us();
                    serial_result = serial.sample(frame.data(), frame.size());
                    sample_cpu = thread_cpu_us() - start;
                };
                auto reference_work = [&] {
                    const double start = thread_cpu_us();
                    // Independent scalar O(V) work, not another call into the sampler:
                    // a max pass plus a full double-precision exp normalizer.
                    const double m = *std::max_element(frame.begin(), frame.end());
                    double z = 0.0;
                    for (float x : frame) z += std::exp(static_cast<double>(x) - m);
                    reference_cpu = thread_cpu_us() - start;
                    VJ_CHECK(std::isfinite(z) && z >= 1.0);
                };
                // Pair on the same frame and alternate order to balance cache/frequency effects.
                // Thread CPU time excludes descheduling and idle OpenMP worker spin time.
                if (t % 2) { reference_work(); sample_serial(); }
                else { sample_serial(); reference_work(); }
                VJ_CHECK(sample_cpu > 0.0 && reference_cpu > 0.0);
                if (t >= 10) {  // exclude allocation / cold-start warmup, not slow measured samples
                    serial_us.push_back(sample_cpu);
                    reference_us.push_back(reference_cpu);
                    work_ratios.push_back(sample_cpu / reference_cpu);
                }
            }
            const auto t0 = std::chrono::steady_clock::now();
            const auto r = s.sample(frame.data(), frame.size());
            const auto t1 = std::chrono::steady_clock::now();
            VJ_CHECK(r.id >= 0 && static_cast<size_t>(r.id) < V);
            if (work_budget > 0.0) {
                // Exercise the unchanged default team as well as the timed serial path.
                VJ_CHECK(r.id == serial_result.id && r.logprob == serial_result.logprob);
                VJ_CHECK(r.n_candidates == serial_result.n_candidates);
                VJ_CHECK(r.full_vocab_fallback == serial_result.full_vocab_fallback);
                VJ_CHECK(r.top.size() == serial_result.top.size());
                for (size_t i = 0; i < r.top.size(); ++i) {
                    VJ_CHECK(r.top[i].id == serial_result.top[i].id);
                    VJ_CHECK(r.top[i].logprob == serial_result.top[i].logprob);
                }
                serial.accept(serial_result.id);
            }
            if (fallbacks) *fallbacks += r.full_vocab_fallback;
            s.accept(r.id);
            us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        std::sort(us.begin(), us.end());
        const double med = us[us.size() / 2], p90 = us[us.size() * 9 / 10];
        std::printf("  %-62s median %7.1f us  p90 %7.1f us\n", label, med, p90);
        if (work_budget > 0.0) {
            std::sort(serial_us.begin(), serial_us.end());
            std::sort(reference_us.begin(), reference_us.end());
            std::sort(work_ratios.begin(), work_ratios.end());
            const size_t mid = work_ratios.size() / 2;
            std::printf("    serial CPU median %.1f us; scalar reference %.1f us; paired ratio %.3f (budget < %.2f)%s\n",
                        serial_us[mid], reference_us[mid], work_ratios[mid], work_budget,
                        kSanitized ? " [diagnostic under sanitizers]" : "");
            // Full chain must beat scalar normalization alone; without logprobs it must
            // cost less than half that work. Full-sort / scalar-exp regressions remain gated.
            if (!kSanitized) VJ_CHECK(work_ratios[mid] < work_budget);
        }
        return med;
    };
    std::printf("Sampler latency, %zu-token vocabulary, 300 steps (%s build), requested default threads=%d\n",
                V, kSanitized ? "sanitizer" : "optimized", sampling::StackParams{}.threads);
    sampling::StackParams full;
    full.temperature = 0.8f;
    full.top_k = 40; full.top_p = 0.95f; full.min_p = 0.05f; full.typical_p = 0.95f;
    full.xtc_probability = 0.5f; full.xtc_threshold = 0.1f;
    full.repeat_penalty = 1.1f; full.presence_penalty = 0.5f; full.frequency_penalty = 0.3f; full.penalty_last_n = 256;
    full.dry_multiplier = 0.8f; full.dry_last_n = 512; full.dry_breakers = {10, 13, 25};
    full.logit_bias = {{100, 2.0f}, {200, -INFINITY}, {300, -1.0f}};
    full.n_logprobs = 5;
    full.seed = 7;
    const double t_full = bench(full, "full chain (k40 p min-p typ XTC DRY pen bias) + 5 logprobs", nullptr, 1.0);
    sampling::StackParams nolp = full;
    nolp.n_logprobs = 0;
    const double t_nolp = bench(nolp, "full chain, no logprobs", nullptr, 0.5);
    sampling::StackParams one = full;
    one.threads = 1;
    bench(one, "full chain + 5 logprobs, 1 thread", nullptr);
    sampling::StackParams greedy;
    greedy.temperature = 0;
    greedy.top_k = 1;
    greedy.top_p = 1;
    greedy.min_p = 0;
    bench(greedy, "greedy (top_k = 1)", nullptr);
    sampling::StackParams mir = full;
    mir.mirostat = 2;
    mir.mirostat_tau = 1.5f;
    bench(mir, "Mirostat v2 (tau 1.5) + penalties + DRY + bias + 5 logprobs", nullptr);
    sampling::StackParams nok = full;
    nok.top_k = 0;
    nok.typical_p = 1.0f;
    int fb = 0;
    bench(nok, "top_k off (p .95, min-p .05) + 5 logprobs", &fb);
    std::printf("  top_k off: %d / 300 steps grew the candidate set\n", fb);
    // Default OpenMP wall latency remains visible, but is not a shared-host acceptance clock:
    // StackParams explicitly requests four threads even when OMP_NUM_THREADS=1. RUN_SERIAL
    // only excludes sibling CTests, not unrelated host work or OpenMP scheduling delays.
    // Preserve the original absolute gates for an explicitly controlled latency run.
    const char* strict_latency = std::getenv("VJ_SAMPLER_STRICT_LATENCY");
    if (!kSanitized && strict_latency && std::strcmp(strict_latency, "1") == 0) {
        const bool ci_vm = (std::getenv("CI") != nullptr);
        VJ_CHECK(t_nolp < (ci_vm ? 1000.0 : 220.0));
        VJ_CHECK(t_full < (ci_vm ? 1500.0 : 350.0));
    }
}

} // namespace

int main() {
    test_pretok_parity();
    test_pretok_dispatch();
    test_spm();
    test_template_parity();
    test_template_tools();
    test_jinja_engine();
    test_reasoning();
    test_sampler_reference_parity();
    test_sampler_edits();
    test_dry();
    test_xtc_mirostat_determinism();
    test_sum_exp_accuracy();
    test_latency();
    std::printf("test_tokenizer_samplers: OK\n");
    return 0;
}
