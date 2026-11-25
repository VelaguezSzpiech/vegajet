// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// core/tool_call_parser: tojson-style serialization, tools / tool_choice / message validation, the Hermes prompt
// (golden strings for Qwen3 and Qwen2.5, byte-identity with bpe::apply_chat_template when no tools are involved), and
// the streaming parser: every scripted model output is fed whole, split at every byte (two pieces) and in 300 random
// multi-piece splits; all must give the same content / reasoning / tool calls. Covers tags split mid-tag, partial
// tag look-alikes that must pass through, malformed JSON / unknown tools / unterminated blocks falling back to the
// exact raw text, prefill priming for tool_choice required / named, parallel_tool_calls = false, and early emission
// (plain text is never held back).
#include "core/tool_call_parser.hpp"
#include "tests/test_common.hpp"

#include <cstdio>
#include <random>

using namespace vegajet;
using namespace vegajet::toolcall;

namespace {

json::Value J(const std::string& s) {
    json::Value v;
    if (!json::parse(s, v)) { std::fprintf(stderr, "bad JSON in test: %s\n", s.c_str()); VJ_CHECK(false); }
    return v;
}

Message run(const ParserOptions& o, const std::vector<std::string>& pieces, StreamParser* keep = nullptr) {
    StreamParser p(o);
    std::vector<Event> ev;
    for (const auto& s : pieces) p.push(s, ev);
    p.finish(ev);
    Message m;
    accumulate(ev, m);
    for (size_t i = 0; i < ev.size(); ++i)  // tool-call indices are dense and in order
        if (ev[i].kind == Event::Kind::ToolCall) VJ_CHECK(ev[i].call.id == o.id_prefix + std::to_string(ev[i].index));
    if (keep) *keep = p;
    return m;
}

bool same(const Message& a, const Message& b) {
    if (a.content != b.content || a.reasoning != b.reasoning || a.tool_calls.size() != b.tool_calls.size()) return false;
    for (size_t i = 0; i < a.tool_calls.size(); ++i)
        if (a.tool_calls[i].id != b.tool_calls[i].id || a.tool_calls[i].name != b.tool_calls[i].name ||
            a.tool_calls[i].arguments != b.tool_calls[i].arguments)
            return false;
    return true;
}

// Whole, every 2-way split, and random multi-way splits must agree. Returns the whole-input result.
Message check_splits(const ParserOptions& o, const std::string& text, uint32_t seed) {
    const Message whole = run(o, {text});
    for (size_t cut = 0; cut <= text.size(); ++cut) {
        const Message m = run(o, {text.substr(0, cut), text.substr(cut)});
        if (!same(m, whole)) {
            std::fprintf(stderr, "split at %zu differs for: %s\n  content '%s' vs '%s'\n", cut, text.c_str(),
                         m.content.c_str(), whole.content.c_str());
            VJ_CHECK(false);
        }
    }
    std::mt19937 rng(seed);
    for (int it = 0; it < 300; ++it) {
        std::vector<std::string> pieces;
        for (size_t i = 0; i < text.size();) {
            const size_t k = std::min(text.size() - i, size_t(rng() % 5));  // 0-byte pieces included
            pieces.push_back(text.substr(i, k));
            i += k;
        }
        VJ_CHECK(same(run(o, pieces), whole));
    }
    return whole;
}

ParserOptions tools_on(std::vector<std::string> known = {"get_weather", "search"}) {
    ParserOptions o;
    o.parse_tool_calls = true;
    o.known_tools = std::move(known);
    return o;
}

}  // namespace

int main() {
    // ---------------------------------------------------------------- serialization
    {
        const auto v = J(R"({"b":[1,-3,0.1,1e-05,2.5e20,true,null],"a":{"x":"café \"q\" \n\t\\"},"e":[],"o":{}})");
        VJ_CHECK(dump_json(v, false) == "{\"b\":[1,-3,0.1,1e-05,2.5e+20,true,null],\"a\":{\"x\":\"caf\xC3\xA9 \\\"q\\\" \\n\\t\\\\\"},\"e\":[],\"o\":{}}");
        VJ_CHECK(dump_json(v, true) == "{\"b\": [1, -3, 0.1, 1e-05, 2.5e+20, true, null], \"a\": {\"x\": \"caf\xC3\xA9 \\\"q\\\" \\n\\t\\\\\"}, \"e\": [], \"o\": {}}");
        VJ_CHECK(dump_json(J("0.30000000000000004"), false) == "0.30000000000000004");
        VJ_CHECK(dump_json(J("123456789012"), false) == "123456789012");
    }
    // ---------------------------------------------------------------- request validation
    const std::string kTools =
        R"([{"type":"function","function":{"name":"get_weather","description":"Current weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"unit":{"type":"string","enum":["c","f"]}},"required":["city"]}}},)"
        R"({"type":"function","function":{"name":"search","parameters":{"type":"object","properties":{"q":{"type":"string"}}}}}])";
    {
        ToolRequest r;
        std::string err;
        VJ_CHECK(parse_tool_request(J("{\"tools\":" + kTools + "}"), r, err) && r.tools.size() == 2 && r.active());
        VJ_CHECK(r.choice.mode == ChoiceMode::Auto && r.parallel && r.names()[1] == "search");
        VJ_CHECK(r.tools[0].json.rfind("{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\"", 0) == 0);
        VJ_CHECK(parse_tool_request(J("{\"tools\":" + kTools + ",\"tool_choice\":\"none\"}"), r, err) && !r.active());
        VJ_CHECK(parse_tool_request(J("{\"tools\":" + kTools + ",\"tool_choice\":\"required\",\"parallel_tool_calls\":false}"), r, err) &&
                 r.choice.mode == ChoiceMode::Required && !r.parallel);
        VJ_CHECK(parse_tool_request(J("{\"tools\":" + kTools + R"(,"tool_choice":{"type":"function","function":{"name":"search"}}})"), r, err) &&
                 r.choice.mode == ChoiceMode::Named && r.choice.name == "search");
        VJ_CHECK(parse_tool_request(J("{}"), r, err) && !r.active());
        const char* bad[] = {
            R"({"tools":{}})",
            R"({"tools":[{"type":"code_interpreter"}]})",
            R"({"tools":[{"type":"function","function":{"name":"has space"}}]})",
            R"({"tools":[{"type":"function","function":{"name":"a"}},{"type":"function","function":{"name":"a"}}]})",
            R"({"tools":[{"type":"function","function":{"name":"a","parameters":"x"}}]})",
            R"({"tools":[{"type":"function","function":{"name":"a"}}],"tool_choice":"sometimes"})",
            R"({"tools":[{"type":"function","function":{"name":"a"}}],"tool_choice":{"type":"function","function":{"name":"b"}}})",
            R"({"tool_choice":"required"})",
            R"({"tools":[{"type":"function","function":{"name":"a"}}],"parallel_tool_calls":1})",
        };
        for (const char* b : bad) { err.clear(); VJ_CHECK(!parse_tool_request(J(b), r, err) && !err.empty()); }
        Turn t;
        t.role = "assistant";
        VJ_CHECK(parse_message_tools(J(R"({"role":"assistant","content":null,"reasoning_content":"r","tool_calls":[{"id":"c1","type":"function","function":{"name":"search","arguments":"{\"q\": \"x\"}"}},{"type":"function","function":{"name":"get_weather","arguments":{"city":"Oslo"}}}]})"), t, err));
        VJ_CHECK(t.tool_calls.size() == 2 && t.tool_calls[0].id == "c1" && t.tool_calls[0].arguments == "{\"q\": \"x\"}" &&
                 t.tool_calls[1].arguments == "{\"city\":\"Oslo\"}" && t.reasoning_content == "r");
        Turn u;
        u.role = "user";
        VJ_CHECK(!parse_message_tools(J(R"({"role":"user","tool_calls":[]})"), u, err));
        Turn tool;
        tool.role = "tool";
        VJ_CHECK(parse_message_tools(J(R"({"role":"tool","tool_call_id":"c1","content":"ok"})"), tool, err) && tool.tool_call_id == "c1");
    }
    // ---------------------------------------------------------------- prompt formatting
    {
        // Without tools: byte-identical to bpe::apply_chat_template for every format and message mix.
        const char* roles[] = {"system", "user", "assistant", "user", "assistant"};
        const char* contents[] = {"Be brief.", "Hi", "<think>\nhmm\n</think>\n\nHello!", "And 2+2?", "4"};
        for (int f = 0; f < 4; ++f)
            for (int start = 0; start < 2; ++start)
                for (int n = 1; n <= 5 - start; ++n)
                    for (int think = 0; think < 2; ++think)
                        for (int gen = 0; gen < 2; ++gen) {
                            std::vector<bpe::Message> a;
                            std::vector<Turn> b;
                            for (int i = start; i < start + n; ++i) {
                                a.push_back({roles[i], contents[i]});
                                Turn t;
                                t.role = roles[i];
                                t.content = contents[i];
                                b.push_back(t);
                            }
                            bpe::ChatOptions o;
                            o.enable_thinking = think;
                            o.add_generation_prompt = gen;
                            if (f == 0 && n == 2) o.default_system = "Custom.";
                            const auto fmt = static_cast<bpe::ChatFormat>(f);
                            VJ_CHECK(format_chat(fmt, b, ToolRequest{}, o) == bpe::apply_chat_template(fmt, a, o));
                        }
        // Qwen3 golden with tools, a tool-call round trip and two grouped tool results.
        ToolRequest tr;
        std::string err;
        VJ_CHECK(parse_tool_request(J(R"({"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]})"), tr, err));
        std::vector<Turn> msgs(5);
        msgs[0].role = "system"; msgs[0].content = "You are terse.";
        msgs[1].role = "user"; msgs[1].content = "Weather in Oslo and Paris?";
        msgs[2].role = "assistant"; msgs[2].content = "";
        msgs[2].tool_calls = {{"c0", "get_weather", "{\"city\": \"Oslo\"}"}, {"c1", "get_weather", "{\"city\": \"Paris\"}"}};
        msgs[3].role = "tool"; msgs[3].content = "{\"t\": 3}";
        msgs[4].role = "tool"; msgs[4].content = "{\"t\": 17}";
        const std::string want3 =
            "<|im_start|>system\nYou are terse.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
            "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}}}}\n"
            "</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n"
            "<|im_start|>user\nWeather in Oslo and Paris?<|im_end|>\n"
            "<|im_start|>assistant\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}\n</tool_call>\n"
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call><|im_end|>\n"
            "<|im_start|>user\n<tool_response>\n{\"t\": 3}\n</tool_response>\n<tool_response>\n{\"t\": 17}\n</tool_response><|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n";
        VJ_CHECK(format_chat(bpe::ChatFormat::Qwen3, msgs, tr) == want3);
        // Qwen2.5: default system prompt before the tools block, "assistant" + "\n<tool_call>", tojson'd arguments.
        std::vector<Turn> m25(msgs.begin() + 1, msgs.begin() + 4);
        m25[1].content = "Checking.";
        m25[1].tool_calls.resize(1);
        m25[1].tool_calls[0].arguments = "{\"city\":\"Oslo\"}";
        const std::string got25 = format_chat(bpe::ChatFormat::Qwen25, m25, tr);
        VJ_CHECK(got25.rfind("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n# Tools\n", 0) == 0);
        VJ_CHECK(got25.find("<|im_start|>assistant\nChecking.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}\n</tool_call><|im_end|>\n"
                            "<|im_start|>user\n<tool_response>\n{\"t\": 3}\n</tool_response><|im_end|>\n<|im_start|>assistant\n") != std::string::npos);
        // Qwen3 keeps reasoning_content for assistant turns after the last user turn (multi-step tool use).
        std::vector<Turn> rc(msgs.begin(), msgs.begin() + 4);
        rc[2].reasoning_content = "\nNeed both cities.\n";
        bpe::ChatOptions think;
        think.enable_thinking = true;
        const std::string gotrc = format_chat(bpe::ChatFormat::Qwen3, rc, tr, think);
        VJ_CHECK(gotrc.find("<|im_start|>assistant\n<think>\nNeed both cities.\n</think>\n\n<tool_call>") != std::string::npos);
        VJ_CHECK(gotrc.size() > 22 && gotrc.compare(gotrc.size() - 22, 22, "<|im_start|>assistant\n") == 0);
        // tool_choice none: no tools block; required / named: primed generation prompt.
        ToolRequest none = tr;
        none.choice.mode = ChoiceMode::None;
        VJ_CHECK(format_chat(bpe::ChatFormat::Qwen3, msgs, none).find("# Tools") == std::string::npos);
        ToolRequest req = tr;
        req.choice.mode = ChoiceMode::Required;
        VJ_CHECK(format_chat(bpe::ChatFormat::Qwen3, msgs, req).size() > 12 &&
                 format_chat(bpe::ChatFormat::Qwen3, msgs, req).substr(format_chat(bpe::ChatFormat::Qwen3, msgs, req).size() - 12) == "<tool_call>\n");
        ToolRequest named = tr;
        named.choice = {ChoiceMode::Named, "get_weather"};
        VJ_CHECK(generation_prefill(named) == "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": ");
    }
    // ---------------------------------------------------------------- streaming parser
    uint32_t seed = 1;
    {
        ParserOptions plain;  // reasoning on, tools off (the server default for requests without tools)
        Message m = check_splits(plain, "Hello world, caf\xC3\xA9 <b> <tool_call>{\"name\":\"x\"}</tool_call>", seed++);
        VJ_CHECK(m.content == "Hello world, caf\xC3\xA9 <b> <tool_call>{\"name\":\"x\"}</tool_call>" && m.tool_calls.empty());
        m = check_splits(plain, "<think>\nLet me think: 2+2.\n</think>\n\nThe answer is 4.", seed++);
        VJ_CHECK(m.reasoning == "Let me think: 2+2." && m.content == "The answer is 4.");
        m = check_splits(plain, "No think here <think>x</think>", seed++);  // only a leading <think> opens reasoning
        VJ_CHECK(m.content == "No think here <think>x</think>" && m.reasoning.empty());
        m = check_splits(plain, "<thinking is fun", seed++);
        VJ_CHECK(m.content == "<thinking is fun");
        m = check_splits(plain, "<think>\nran out of tokens mid-thought\n", seed++);
        VJ_CHECK(m.reasoning == "ran out of tokens mid-thought" && m.content.empty());
        ParserOptions raw;
        raw.parse_reasoning = false;
        VJ_CHECK(check_splits(raw, "<think>x</think>y", seed++).content == "<think>x</think>y");
        ParserOptions open = plain;
        open.start_in_reasoning = true;
        m = check_splits(open, "already thinking\n</think>\n\nfinal", seed++);
        VJ_CHECK(m.reasoning == "already thinking" && m.content == "final");
    }
    {
        const ParserOptions o = tools_on();
        Message m = check_splits(o, "I'll check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"unit\": \"c\"}}\n</tool_call>", seed++);
        VJ_CHECK(m.content == "I'll check." && m.tool_calls.size() == 1 && m.tool_calls[0].name == "get_weather" &&
                 m.tool_calls[0].arguments == "{\"city\":\"Paris\",\"unit\":\"c\"}" && m.tool_calls[0].id == "call_0");
        m = check_splits(o, "<think>\nTwo cities.\n</think>\n\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}\n</tool_call>\n"
                            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>", seed++);
        VJ_CHECK(m.reasoning == "Two cities." && m.content.empty() && m.tool_calls.size() == 2 &&
                 m.tool_calls[1].arguments == "{\"city\":\"Paris\"}" && m.tool_calls[1].id == "call_1");
        // Tag look-alikes in content pass through untouched; whitespace is only dropped right before a tool call.
        m = check_splits(o, "a < b, <tool and <tool_cal x </tool_call> \n end \n", seed++);
        VJ_CHECK(m.content == "a < b, <tool and <tool_cal x </tool_call> \n end \n" && m.tool_calls.empty());
        // Arguments as a JSON string, "parameters" alias, missing arguments.
        m = check_splits(o, "<tool_call>{\"name\":\"search\",\"arguments\":\"{\\\"q\\\": \\\"gfx906\\\"}\"}</tool_call>"
                            "<tool_call>{\"name\":\"search\",\"parameters\":{\"q\":\"x\"}}</tool_call><tool_call>{\"name\":\"search\"}</tool_call>", seed++);
        VJ_CHECK(m.tool_calls.size() == 3 && m.tool_calls[0].arguments == "{\"q\":\"gfx906\"}" &&
                 m.tool_calls[1].arguments == "{\"q\":\"x\"}" && m.tool_calls[2].arguments == "{}");
        // Malformed JSON, unknown tool, non-object arguments: the raw block comes back as content, byte for byte.
        const std::string bad1 = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": }\n</tool_call>";
        StreamParser keep(o);
        m = run(o, {bad1}, &keep);
        VJ_CHECK(m.content == bad1 && m.tool_calls.empty() && keep.malformed() == 1);
        VJ_CHECK(check_splits(o, bad1, seed++).content == bad1);
        const std::string bad2 = "Sure. <tool_call>{\"name\": \"rm_rf\", \"arguments\": {}}</tool_call> done";
        VJ_CHECK(check_splits(o, bad2, seed++).content == bad2);
        const std::string bad3 = "<tool_call>{\"name\": \"search\", \"arguments\": [1, 2]}</tool_call>";
        VJ_CHECK(check_splits(o, bad3, seed++).content == bad3);
        // Mixed: a good call, a broken one, a good one -> indices stay dense (0, 1).
        m = check_splits(o, "<tool_call>{\"name\":\"search\",\"arguments\":{}}</tool_call>\n<tool_call>{oops</tool_call>\n"
                            "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Rome\"}}</tool_call>", seed++);
        VJ_CHECK(m.tool_calls.size() == 2 && m.content == "<tool_call>{oops</tool_call>" && m.tool_calls[1].id == "call_1");
        // Unterminated at end of generation: a complete call is accepted, an incomplete one is raw content.
        m = check_splits(o, "<tool_call>\n{\"name\": \"search\", \"arguments\": {\"q\": \"a\"}}\n", seed++);
        VJ_CHECK(m.tool_calls.size() == 1 && m.content.empty());
        VJ_CHECK(check_splits(o, "<tool_call>\n{\"name\": \"sea", seed++).content == "<tool_call>\n{\"name\": \"sea");
        // Tools offered but the model answers in prose: unchanged.
        VJ_CHECK(check_splits(o, "It is sunny in Paris \xE2\x98\x80.", seed++).content == "It is sunny in Paris \xE2\x98\x80.");
    }
    {
        // parallel_tool_calls = false: the first call ends the turn.
        ParserOptions o = tools_on();
        o.parallel = false;
        StreamParser p(o);
        std::vector<Event> ev;
        p.push("<tool_call>{\"name\":\"search\",\"arguments\":{}}</tool_call>\n<tool_call>{\"name\":\"search\",", ev);
        VJ_CHECK(p.done() && p.tool_calls() == 1);
        p.push("\"arguments\":{}}</tool_call>", ev);
        p.finish(ev);
        Message m;
        accumulate(ev, m);
        VJ_CHECK(m.tool_calls.size() == 1 && m.content.empty());
        // tool_choice named: the parser is primed with the prefill; the model continues inside the call.
        ToolRequest named;
        named.tools = {{"get_weather", "{}"}};
        named.choice = {ChoiceMode::Named, "get_weather"};
        ParserOptions pn = tools_on({"get_weather"});
        pn.prefill = generation_prefill(named);
        m = check_splits(pn, "{\"city\": \"Oslo\"}}\n</tool_call>", seed++);
        VJ_CHECK(m.tool_calls.size() == 1 && m.tool_calls[0].name == "get_weather" && m.tool_calls[0].arguments == "{\"city\":\"Oslo\"}");
        // tool_choice required but the model refuses: only the model's own text is returned (no prefill leak).
        ToolRequest req = named;
        req.choice.mode = ChoiceMode::Required;
        ParserOptions pr = tools_on({"get_weather"});
        pr.prefill = generation_prefill(req);
        VJ_CHECK(check_splits(pr, "Sorry, I can't.", seed++).content == "Sorry, I can't.");
        // Custom id prefix (the server derives it from the completion id).
        ParserOptions pid = tools_on();
        pid.id_prefix = "call_abc_";
        VJ_CHECK(run(pid, {"<tool_call>{\"name\":\"search\"}</tool_call>"}).tool_calls[0].id == "call_abc_0");
    }
    {
        // Early emission: text that cannot start a tag is emitted immediately; possible tag prefixes are held.
        StreamParser p(tools_on());
        std::vector<Event> ev;
        p.push("Hel", ev);
        VJ_CHECK(ev.size() == 1 && ev[0].text == "Hel");
        ev.clear();
        p.push(" <tool", ev);
        VJ_CHECK(ev.empty());  // " " is held with the tag prefix (dropped before a call)
    }
    std::printf("test_tool_call_parser: serialization, validation, Qwen3/Qwen2.5 golden prompts, template identity, "
                "streaming splits (every byte + random), fallbacks, prefill, parallel=false OK\nPASS\n");
    return 0;
}
