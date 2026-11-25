// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Loopback SSE / JSON tests for function calling and reasoning in core/server (server_tool_calls.patch), CPU only.
// A scripted generator streams model text in 1-3 byte pieces (tags split mid-tag) and the test checks the OpenAI
// wire format: reasoning_content deltas, tool_calls deltas (header with id / type / name, then argument fragments,
// accumulated by index), finish_reason "tool_calls", non-streaming message.tool_calls with content null, multi-turn
// tool messages reaching the GenerateFn, tool_choice none / named / unknown, parallel_tool_calls = false stopping the
// generator, malformed blocks and stop sequences, reasoning_format none, ServerOptions::workers concurrency, and the
// IEngine adapter (HEAD MockEngine) rendering tools into the prompt.
#include "core/json_lite.hpp"
#include "core/server.hpp"
#include "core/server_engine_adapter.hpp"
#include "tests/test_common.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <thread>

using namespace vegajet;
using namespace vegajet::serve;

namespace {

struct Script {
    std::mutex mu;
    std::string text;
    int delay_ms = 0;
    GenerateRequest last;
    std::atomic<int> active{0}, max_active{0}, calls{0};
    std::atomic<bool> last_cancelled{false};
    GenerateResult operator()(const GenerateRequest& r, const TokenSink& sink) {
        std::string t;
        int delay;
        {
            std::lock_guard<std::mutex> lk(mu);
            last = r;
            t = text;
            delay = delay_ms;
        }
        const int a = ++active;
        for (int m = max_active; a > m && !max_active.compare_exchange_weak(m, a);) {}
        ++calls;
        bool cancelled = false;
        for (size_t i = 0, k = 0; i < t.size(); i += k) {
            k = 1 + (i * 7 + 3) % 3;  // 1..3-byte pieces: every tag gets split somewhere
            if (delay) std::this_thread::sleep_for(std::chrono::milliseconds(delay));
            if (!sink(std::string_view(t).substr(i, k))) { cancelled = true; break; }
        }
        last_cancelled = cancelled;
        --active;
        GenerateResult res;
        res.finish = FinishReason::Stop;
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

// One request on a fresh connection (Connection: close), reads until the server closes.
Resp post(uint16_t port, const std::string& body) {
    const int fd = connect_local(port);
    const std::string req = "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n"
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

json::Value parse(const std::string& s) {
    json::Value v;
    if (!json::parse(s, v)) { std::fprintf(stderr, "bad JSON: %s\n", s.c_str()); VJ_CHECK(false); }
    return v;
}

struct Call {
    std::string id, type, name, args;
    int headers = 0;
};
struct Stream {
    std::string content, reasoning, finish;
    std::map<int, Call> calls;
    int chunks = 0;
    bool done = false;
};

Stream parse_sse(const std::string& body) {
    Stream s;
    for (size_t pos = 0; pos < body.size();) {
        const size_t e = body.find("\n\n", pos);
        VJ_CHECK(e != std::string::npos);
        const std::string ev = body.substr(pos, e - pos);
        pos = e + 2;
        VJ_CHECK(ev.rfind("data: ", 0) == 0 && !s.done);
        if (ev == "data: [DONE]") { s.done = true; continue; }
        const json::Value c = parse(ev.substr(6));
        ++s.chunks;
        const auto& ch = c.get("choices")->arr;
        if (ch.empty()) continue;
        if (const auto* f = ch[0].get("finish_reason"); f && f->is_string()) s.finish = f->str;
        const json::Value* d = ch[0].get("delta");
        if (const auto* x = d->get("content"); x && x->is_string()) s.content += x->str;
        if (const auto* x = d->get("reasoning_content"); x && x->is_string()) s.reasoning += x->str;
        if (const auto* tc = d->get("tool_calls"); tc) {
            VJ_CHECK(tc->is_array() && tc->arr.size() == 1);
            const json::Value& t = tc->arr[0];
            Call& cl = s.calls[static_cast<int>(t.get("index")->num)];
            if (const auto* id = t.get("id")) { cl.id = id->str; cl.type = t.get("type")->str; ++cl.headers; }
            const json::Value* fn = t.get("function");
            if (const auto* n = fn->get("name")) cl.name = n->str;
            cl.args += fn->get("arguments")->str;
        }
    }
    VJ_CHECK(s.done);
    return s;
}

const std::string kTools =
    R"("tools":[{"type":"function","function":{"name":"get_weather","description":"Weather now","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}},)"
    R"({"type":"function","function":{"name":"search","parameters":{"type":"object","properties":{"q":{"type":"string"}}}}}])";
const std::string kUser = R"("messages":[{"role":"user","content":"Weather in Paris?"}])";
const std::string kCallParis = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";

}  // namespace

int main() {
    Script gen;
    ServerOptions opt;
    opt.port = 0;
    opt.workers = 1;
    OpenAiServer srv(opt, [&](const GenerateRequest& r, const TokenSink& s) { return gen(r, s); });
    std::string err;
    VJ_CHECK(srv.start(&err));
    const uint16_t port = srv.port();
    auto set = [&](const std::string& t) { std::lock_guard<std::mutex> lk(gen.mu); gen.text = t; };

    // ---- streaming: reasoning + one tool call, tags split across 1-3 byte pieces
    set("<think>\nUser wants Paris weather.\n</think>\n\n" + kCallParis);
    Stream s = parse_sse(post(port, "{" + kUser + "," + kTools + ",\"stream\":true}").body);
    VJ_CHECK(s.reasoning == "User wants Paris weather." && s.content.empty() && s.finish == "tool_calls");
    VJ_CHECK(s.calls.size() == 1 && s.calls[0].headers == 1 && s.calls[0].type == "function" && s.calls[0].name == "get_weather");
    VJ_CHECK(s.calls[0].args == "{\"city\":\"Paris\"}" && s.calls[0].id.rfind("call_", 0) == 0 && s.calls[0].id.size() > 7);
    {
        std::lock_guard<std::mutex> lk(gen.mu);
        VJ_CHECK(gen.last.tools.tools.size() == 2 && gen.last.tools.active() && gen.last.tools.parallel);
    }
    // ---- non-streaming: content null, reasoning_content, tool_calls[].function.arguments is a JSON string
    {
        const Resp r = post(port, "{" + kUser + "," + kTools + "}");
        VJ_CHECK(r.status == 200);
        const json::Value v = parse(r.body);
        const json::Value& ch = v.get("choices")->arr[0];
        const json::Value* m = ch.get("message");
        VJ_CHECK(ch.get("finish_reason")->str == "tool_calls" && m->get("content")->is_null());
        VJ_CHECK(m->get("reasoning_content")->str == "User wants Paris weather.");
        const auto& tc = m->get("tool_calls")->arr;
        VJ_CHECK(tc.size() == 1 && tc[0].get("type")->str == "function" && tc[0].get("function")->get("name")->str == "get_weather");
        VJ_CHECK(parse(tc[0].get("function")->get("arguments")->str).get("city")->str == "Paris");
    }
    // ---- parallel calls with content before them
    set("Checking both.\n" + kCallParis + "\n<tool_call>\n{\"name\": \"search\", \"arguments\": {\"q\": \"Oslo weather\"}}\n</tool_call>");
    s = parse_sse(post(port, "{" + kUser + "," + kTools + ",\"stream\":true}").body);
    VJ_CHECK(s.content == "Checking both." && s.calls.size() == 2 && s.calls[1].name == "search" &&
             s.calls[1].args == "{\"q\":\"Oslo weather\"}" && s.calls[0].id != s.calls[1].id && s.finish == "tool_calls");
    // ---- parallel_tool_calls = false: the generator is stopped after the first call
    set(kCallParis + "\n" + kCallParis + std::string(400, ' ') + "tail");
    s = parse_sse(post(port, "{" + kUser + "," + kTools + ",\"parallel_tool_calls\":false,\"stream\":true}").body);
    VJ_CHECK(s.calls.size() == 1 && s.content.empty() && s.finish == "tool_calls" && gen.last_cancelled);
    // ---- multi-turn: assistant tool_calls + tool results reach the GenerateFn; the adapter renders them
    set("It is 17 C in Paris.");
    {
        const std::string body = "{" + kTools + R"(,"messages":[{"role":"user","content":"Weather in Paris?"},)"
            R"({"role":"assistant","content":null,"tool_calls":[{"id":"call_x_0","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},)"
            R"({"role":"tool","tool_call_id":"call_x_0","content":"{\"temp_c\": 17}"}],"stream":true})";
        s = parse_sse(post(port, body).body);
        VJ_CHECK(s.content == "It is 17 C in Paris." && s.calls.empty() && s.finish == "stop");
        std::lock_guard<std::mutex> lk(gen.mu);
        const auto& ms = gen.last.messages;
        VJ_CHECK(ms.size() == 3 && ms[1].tool_calls.size() == 1 && ms[1].tool_calls[0].arguments == "{\"city\":\"Paris\"}" &&
                 ms[2].role == "tool" && ms[2].tool_call_id == "call_x_0");
        const std::string prompt = format_request(gen.last, bpe::ChatFormat::Qwen3);
        VJ_CHECK(prompt.find("<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\":\"Paris\"}}\n</tool_call><|im_end|>\n"
                             "<|im_start|>user\n<tool_response>\n{\"temp_c\": 17}\n</tool_response><|im_end|>") != std::string::npos);
        VJ_CHECK(prompt.find("# Tools") != std::string::npos);
    }
    // ---- tool_choice none: no parsing, tags stay content
    set(kCallParis);
    s = parse_sse(post(port, "{" + kUser + "," + kTools + ",\"tool_choice\":\"none\",\"stream\":true}").body);
    VJ_CHECK(s.content == kCallParis && s.calls.empty() && s.finish == "stop");
    // ---- tool_choice named: the parser is primed with the prefill the prompt ends with
    set("{\"q\": \"gfx906\"}}\n</tool_call>");
    s = parse_sse(post(port, "{" + kUser + "," + kTools + R"(,"tool_choice":{"type":"function","function":{"name":"search"}},"stream":true})").body);
    VJ_CHECK(s.calls.size() == 1 && s.calls[0].name == "search" && s.calls[0].args == "{\"q\":\"gfx906\"}" && s.content.empty());
    {
        std::lock_guard<std::mutex> lk(gen.mu);
        const std::string p = format_request(gen.last, bpe::ChatFormat::Qwen3);
        const std::string want = "<tool_call>\n{\"name\": \"search\", \"arguments\": ";
        VJ_CHECK(p.size() > want.size() && p.compare(p.size() - want.size(), want.size(), want) == 0);
    }
    // ---- malformed block: raw text as content; stop sequence cutting a complete call: still a tool call
    set("Let me try. <tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": }</tool_call>");
    s = parse_sse(post(port, "{" + kUser + "," + kTools + ",\"stream\":true}").body);
    VJ_CHECK(s.content == "Let me try. <tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": }</tool_call>" &&
             s.calls.empty() && s.finish == "stop");
    set(kCallParis + "\nmore text");
    s = parse_sse(post(port, "{" + kUser + "," + kTools + ",\"stop\":[\"</tool_call>\"],\"stream\":true}").body);
    VJ_CHECK(s.calls.size() == 1 && s.calls[0].args == "{\"city\":\"Paris\"}" && s.finish == "tool_calls" && s.content.empty());
    // ---- reasoning_format none keeps the raw text; reasoning is parsed without tools too
    set("<think>x</think>\n\ny");
    s = parse_sse(post(port, "{" + kUser + ",\"reasoning_format\":\"none\",\"stream\":true}").body);
    VJ_CHECK(s.content == "<think>x</think>\n\ny" && s.reasoning.empty());
    s = parse_sse(post(port, "{" + kUser + ",\"stream\":true}").body);
    VJ_CHECK(s.content == "y" && s.reasoning == "x" && s.finish == "stop");
    // ---- validation errors -> 400
    const char* bad[] = {
        R"("tool_choice":{"type":"function","function":{"name":"nope"}})",
        R"("tool_choice":"always")",
        R"("parallel_tool_calls":"yes")",
        R"("reasoning_format":7)",
    };
    for (const char* b : bad) {
        const Resp r = post(port, "{" + kUser + "," + kTools + "," + b + "}");
        VJ_CHECK(r.status == 400 && r.body.find("invalid_request_error") != std::string::npos);
    }
    VJ_CHECK(post(port, R"({"messages":[{"role":"user","content":"x","tool_calls":[]}]})").status == 400);
    srv.stop();

    // ---- workers: 3 concurrent generations vs the FIFO default
    for (size_t workers : {size_t(1), size_t(3)}) {
        Script slow;
        slow.text = "abcdefghij";  // 6 pieces x 25 ms = 150 ms per request
        slow.delay_ms = 25;
        ServerOptions o;
        o.port = 0;
        o.workers = workers;
        OpenAiServer w(o, [&](const GenerateRequest& r, const TokenSink& sk) { return slow(r, sk); });
        VJ_CHECK(w.start(&err));
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> cl;
        std::atomic<int> ok{0};
        for (int i = 0; i < 3; ++i)
            cl.emplace_back([&] { if (parse_sse(post(w.port(), "{" + kUser + ",\"stream\":true}").body).content == "abcdefghij") ++ok; });
        for (auto& t : cl) t.join();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("  workers=%zu: 3 x 150 ms requests in %.0f ms, max concurrent %d\n", workers, ms, slow.max_active.load());
        VJ_CHECK(ok == 3 && slow.max_active == static_cast<int>(workers));
        w.stop();
    }

    // ---- IEngine adapter on HEAD's MockEngine: tools enlarge the prompt, the canned reply is unchanged
    {
        MockEngine eng;
        ServerOptions o;
        o.port = 0;
        OpenAiServer m(o, make_engine_generate_fn(eng));
        VJ_CHECK(m.start(&err));
        const json::Value plain = parse(post(m.port(), "{" + kUser + "}").body);
        const json::Value tooled = parse(post(m.port(), "{" + kUser + "," + kTools + "}").body);
        const double p0 = plain.get("usage")->get("prompt_tokens")->num, p1 = tooled.get("usage")->get("prompt_tokens")->num;
        VJ_CHECK(p1 > p0 + 100);
        VJ_CHECK(tooled.get("choices")->arr[0].get("message")->get("content")->str ==
                 plain.get("choices")->arr[0].get("message")->get("content")->str);
        VJ_CHECK(tooled.get("choices")->arr[0].get("finish_reason")->str != "tool_calls");
        std::printf("  MockEngine adapter: prompt %g tokens -> %g with 2 tools\n", p0, p1);
        m.stop();
    }
    std::printf("test_server_tools: reasoning/tool_calls SSE deltas, non-streaming tool_calls, multi-turn, tool_choice "
                "none/named, parallel=false, malformed, stop, reasoning_format, 400s, workers, MockEngine OK\nPASS\n");
    return 0;
}
