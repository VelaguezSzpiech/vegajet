// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Loopback integration test for core/server (OpenAI-compatible HTTP/1.1 + SSE), CPU only.
// Boots the server on an ephemeral 127.0.0.1 port with a mock token generator and checks: /health, /v1/models,
// non-streaming JSON, SSE chunk parsing (UTF-8 split across tokens), stop sequences (incl. ones spanning tokens),
// max_tokens -> "length", usage/timings, keep-alive, pipelining, Expect: 100-continue, error statuses, queue-full 503,
// client disconnect cancelling generation, the IEngine adapter on the HEAD MockEngine, and clean shutdown.
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
#include <cmath>
#include <functional>
#include <cstdio>
#include <map>
#include <mutex>
#include <thread>

using namespace vegajet;
using namespace vegajet::serve;

namespace {

const std::vector<std::string> kPieces = {"Hel", "lo", " wor", "ld", ",", " caf", "\xC3", "\xA9", " ", "\xF0\x9F",
                                          "\x9A\x80", " END", " tail"};
const std::string kFull = "Hello world, caf\xC3\xA9 \xF0\x9F\x9A\x80 END tail";

struct MockGen {
    std::atomic<int> delay_ms{0};
    std::atomic<int> repeat{1};          // emit kPieces this many times
    std::atomic<int> last_emitted{0};    // pieces the sink accepted in the last call
    std::atomic<bool> last_cancelled{false};
    std::atomic<int> calls{0};
    std::atomic<bool> running{false};
    std::mutex mu;
    GenerateRequest last;

    GenerateResult operator()(const GenerateRequest& r, const TokenSink& sink) {
        running = true;
        ++calls;
        { std::lock_guard<std::mutex> lk(mu); last = r; }
        GenerateResult res;
        for (const auto& m : r.messages) res.prompt_tokens += static_cast<uint32_t>(m.content.size());
        int emitted = 0;
        bool cancelled = false;
        for (int rep = 0; rep < repeat && !cancelled; ++rep) {
            for (const auto& p : kPieces) {
                if (delay_ms) std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
                ++emitted;
                if (!sink(p)) { cancelled = true; break; }
            }
        }
        last_emitted = emitted;
        last_cancelled = cancelled;
        res.finish = FinishReason::Stop;
        running = false;
        return res;
    }
};

int connect_local(uint16_t port) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    VJ_CHECK(fd >= 0);
    timeval tv{5, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0) { ::close(fd); return -1; }
    return fd;
}

void send_str(int fd, const std::string& s) {
    size_t off = 0;
    while (off < s.size()) {
        const ssize_t n = ::send(fd, s.data() + off, s.size() - off, MSG_NOSIGNAL);
        VJ_CHECK(n > 0);
        off += static_cast<size_t>(n);
    }
}

struct Resp {
    int status = 0;
    std::map<std::string, std::string> headers;
    std::string body;
};

// Reads one response; `buf` carries bytes across pipelined responses on the same socket.
bool read_resp(int fd, Resp& r, std::string& buf) {
    char tmp[8192];
    size_t hdr_end;
    while ((hdr_end = buf.find("\r\n\r\n")) == std::string::npos) {
        const ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
        if (n <= 0) return false;
        buf.append(tmp, static_cast<size_t>(n));
    }
    r = Resp{};
    const std::string head = buf.substr(0, hdr_end);
    r.status = std::atoi(head.c_str() + head.find(' ') + 1);
    for (size_t pos = head.find("\r\n"); pos != std::string::npos && pos + 2 < head.size();) {
        const size_t e = head.find("\r\n", pos + 2);
        const std::string line = head.substr(pos + 2, (e == std::string::npos ? head.size() : e) - pos - 2);
        const size_t c = line.find(':');
        std::string k = line.substr(0, c);
        for (auto& ch : k) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        r.headers[k] = line.substr(line.find_first_not_of(' ', c + 1));
        pos = e;
    }
    buf.erase(0, hdr_end + 4);
    if (r.status == 100) return read_resp(fd, r, buf);
    if (r.headers.count("content-length")) {
        const size_t len = std::stoul(r.headers["content-length"]);
        while (buf.size() < len) {
            const ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
            if (n <= 0) return false;
            buf.append(tmp, static_cast<size_t>(n));
        }
        r.body = buf.substr(0, len);
        buf.erase(0, len);
    } else {  // SSE: until the server closes
        while (true) {
            const ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
            if (n <= 0) break;
            buf.append(tmp, static_cast<size_t>(n));
        }
        r.body.swap(buf);
    }
    return true;
}

std::string req_text(const std::string& method, const std::string& path, const std::string& body,
                     const std::string& extra = "") {
    std::string s = method + " " + path + " HTTP/1.1\r\nHost: localhost\r\n" + extra;
    if (!body.empty() || method == "POST") s += "Content-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) + "\r\n";
    return s + "\r\n" + body;
}

Resp request(uint16_t port, const std::string& method, const std::string& path, const std::string& body = "",
             const std::string& extra = "Connection: close\r\n") {
    const int fd = connect_local(port);
    VJ_CHECK(fd >= 0);
    send_str(fd, req_text(method, path, body, extra));
    Resp r;
    std::string buf;
    VJ_CHECK(read_resp(fd, r, buf));
    ::close(fd);
    return r;
}

json::Value parse_json(const std::string& s) {
    json::Value v;
    if (!json::parse(s, v)) { std::fprintf(stderr, "bad JSON: %s\n", s.c_str()); VJ_CHECK(false); }
    return v;
}

bool valid_utf8(const std::string& s) {
    for (size_t i = 0; i < s.size();) {
        const auto c = static_cast<unsigned char>(s[i]);
        const size_t n = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 0;
        if (!n || i + n > s.size()) return false;
        for (size_t k = 1; k < n; ++k) if ((static_cast<unsigned char>(s[i + k]) & 0xC0) != 0x80) return false;
        i += n;
    }
    return true;
}

struct Sse {
    std::vector<json::Value> chunks;
    std::string content;
    std::string finish;
    bool done = false;
    const json::Value* usage = nullptr;
};

Sse parse_sse(const std::string& body) {
    Sse s;
    size_t pos = 0;
    while (pos < body.size()) {
        const size_t e = body.find("\n\n", pos);
        VJ_CHECK(e != std::string::npos);  // every event is terminated
        const std::string ev = body.substr(pos, e - pos);
        pos = e + 2;
        VJ_CHECK(ev.rfind("data: ", 0) == 0);
        const std::string data = ev.substr(6);
        VJ_CHECK(!s.done);  // nothing after [DONE]
        if (data == "[DONE]") { s.done = true; continue; }
        s.chunks.push_back(parse_json(data));
    }
    for (const auto& c : s.chunks) {
        VJ_CHECK(c.get("object")->str == "chat.completion.chunk");
        const auto& ch = c.get("choices")->arr;
        if (ch.empty()) { s.usage = c.get("usage"); continue; }
        const auto* delta = ch[0].get("delta");
        if (const auto* t = delta->get("content"); t && t->is_string()) {
            VJ_CHECK(valid_utf8(t->str));  // no split code points on the wire
            s.content += t->str;
        }
        if (const auto* f = ch[0].get("finish_reason"); f && f->is_string()) s.finish = f->str;
    }
    return s;
}

std::string chat_body(const std::string& extra) {
    return "{\"model\":\"m\",\"messages\":[{\"role\":\"system\",\"content\":\"be brief\"},"
           "{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"hi\"}]}]" + extra + "}";
}

void wait_for(const std::function<bool()>& pred, int ms = 3000) {
    for (int i = 0; i < ms / 5 && !pred(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    VJ_CHECK(pred());
}

} // namespace

int main() {
    // ---- StopMatcher unit checks.
    {
        StopMatcher sm({"END", "\n\n"});
        std::string out = sm.push("ab");
        out += sm.push("cE");
        VJ_CHECK(out == "abc");  // "E" held back (prefix of END)
        out += sm.push("NX");    // "ENX" is not END
        VJ_CHECK(out == "abcENX");
        out += sm.push("y\xC3");  // partial UTF-8 held back
        VJ_CHECK(out == "abcENXy");
        out += sm.push("\xA9 EN");
        out += sm.push("D more");
        VJ_CHECK(sm.stopped() && out == "abcENXy\xC3\xA9 ");
        VJ_CHECK(sm.push("x").empty() && sm.flush().empty());
        StopMatcher tail({});
        VJ_CHECK(tail.push("a\xF0\x9F") == "a" && tail.flush() == "\xEF\xBF\xBD");
    }

    MockGen mock;
    ServerOptions opt;
    opt.port = 0;
    opt.model_id = "vegajet-test";
    opt.context_length = 32768;
    opt.max_body_bytes = 64 * 1024;
    opt.max_queue = 1;
    auto srv = std::make_unique<OpenAiServer>(opt, [&](const GenerateRequest& r, const TokenSink& s) { return mock(r, s); });
    std::string err;
    VJ_CHECK(srv->start(&err));
    const uint16_t port = srv->port();
    VJ_CHECK(port != 0);

    // ---- /health and /v1/models
    {
        Resp h = request(port, "GET", "/health");
        VJ_CHECK(h.status == 200 && h.headers["content-type"] == "application/json");
        const auto j = parse_json(h.body);
        VJ_CHECK(j.get("status")->str == "ok" && j.get("model")->str == "vegajet-test");
        Resp m = request(port, "GET", "/v1/models?x=1");
        const auto mj = parse_json(m.body);
        VJ_CHECK(m.status == 200 && mj.get("object")->str == "list");
        VJ_CHECK(mj.get("data")->arr.size() == 1 && mj.get("data")->arr[0].get("id")->str == "vegajet-test");
        VJ_CHECK(mj.get("data")->arr[0].get("context_length")->num == 32768);
    }

    // ---- non-streaming JSON + request field parsing
    {
        Resp r = request(port, "POST", "/v1/chat/completions",
                               chat_body(",\"temperature\":0.7,\"top_p\":0.9,\"top_k\":40,\"min_p\":0.05,"
                                         "\"repeat_penalty\":1.1,\"seed\":1234,\"max_tokens\":100"));
        VJ_CHECK(r.status == 200);
        const auto j = parse_json(r.body);
        VJ_CHECK(j.get("object")->str == "chat.completion" && j.get("model")->str == "m");
        const auto& ch = j.get("choices")->arr[0];
        VJ_CHECK(ch.get("message")->get("role")->str == "assistant");
        VJ_CHECK(ch.get("message")->get("content")->str == kFull);
        VJ_CHECK(ch.get("finish_reason")->str == "stop");
        const auto* u = j.get("usage");
        VJ_CHECK(u->get("prompt_tokens")->num == 10 && u->get("completion_tokens")->num == kPieces.size());
        VJ_CHECK(u->get("total_tokens")->num == 10 + kPieces.size());
        VJ_CHECK(j.get("timings")->get("decode_tok_per_s")->is_number());
        std::lock_guard<std::mutex> lk(mock.mu);
        VJ_CHECK(mock.last.messages.size() == 2 && mock.last.messages[1].content == "hi");
        VJ_CHECK(mock.last.max_tokens == 100 && mock.last.sampling.top_k == 40 && mock.last.sampling.seed == 1234);
        VJ_CHECK(std::fabs(mock.last.sampling.temperature - 0.7f) < 1e-6f && std::fabs(mock.last.sampling.min_p - 0.05f) < 1e-6f);
    }

    // ---- stop sequence + max_tokens
    {
        const auto j = parse_json(request(port, "POST", "/v1/chat/completions", chat_body(",\"stop\":[\" END\"]")).body);
        VJ_CHECK(j.get("choices")->arr[0].get("message")->get("content")->str == "Hello world, caf\xC3\xA9 \xF0\x9F\x9A\x80");
        VJ_CHECK(j.get("choices")->arr[0].get("finish_reason")->str == "stop");
        VJ_CHECK(mock.last_cancelled && mock.last_emitted == 12);  // generation stopped at the stop token
        const auto k = parse_json(request(port, "POST", "/v1/chat/completions", chat_body(",\"max_tokens\":3")).body);
        VJ_CHECK(k.get("choices")->arr[0].get("message")->get("content")->str == "Hello wor");
        VJ_CHECK(k.get("choices")->arr[0].get("finish_reason")->str == "length");
        VJ_CHECK(k.get("usage")->get("completion_tokens")->num == 3);
    }

    // ---- SSE streaming
    {
        Resp r = request(port, "POST", "/v1/chat/completions",
                               chat_body(",\"stream\":true,\"stream_options\":{\"include_usage\":true}"));
        VJ_CHECK(r.status == 200 && r.headers["content-type"] == "text/event-stream");
        const Sse s = parse_sse(r.body);
        VJ_CHECK(s.done && s.content == kFull && s.finish == "stop");
        VJ_CHECK(s.chunks.front().get("choices")->arr[0].get("delta")->get("role")->str == "assistant");
        VJ_CHECK(s.usage && s.usage->get("completion_tokens")->num == kPieces.size());
        VJ_CHECK(s.chunks.size() >= 10);  // token-by-token, not one blob
        // Stop string spanning three tokens ("ld" + "," + " caf"): nothing of it may leak.
        const Sse t = parse_sse(request(port, "POST", "/v1/chat/completions", chat_body(",\"stream\":true,\"stop\":\"ld, ca\"")).body);
        VJ_CHECK(t.done && t.content == "Hello wor" && t.finish == "stop" && t.usage == nullptr);
        const Sse l = parse_sse(request(port, "POST", "/v1/chat/completions", chat_body(",\"stream\":true,\"max_tokens\":2")).body);
        VJ_CHECK(l.content == "Hello" && l.finish == "length");
    }

    // ---- keep-alive (two requests, one socket) and pipelining
    {
        const int fd = connect_local(port);
        std::string buf;
        Resp a, b;
        send_str(fd, req_text("POST", "/v1/chat/completions", chat_body("")));
        VJ_CHECK(read_resp(fd, a, buf) && a.status == 200 && a.headers["connection"] == "keep-alive");
        send_str(fd, req_text("POST", "/v1/chat/completions", chat_body(",\"max_tokens\":1")));
        VJ_CHECK(read_resp(fd, b, buf) && b.status == 200);
        VJ_CHECK(parse_json(b.body).get("choices")->arr[0].get("message")->get("content")->str == "Hel");
        send_str(fd, req_text("GET", "/health", "") + req_text("GET", "/v1/models", ""));
        VJ_CHECK(read_resp(fd, a, buf) && a.status == 200 && parse_json(a.body).get("status"));
        VJ_CHECK(read_resp(fd, b, buf) && b.status == 200 && parse_json(b.body).get("data"));
        ::close(fd);
    }

    // ---- Expect: 100-continue
    {
        const int fd = connect_local(port);
        const std::string body = chat_body(",\"max_tokens\":2");
        send_str(fd, "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nConnection: close\r\nExpect: 100-continue\r\n"
                     "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n");
        char tmp[64] = {};
        const ssize_t n = ::recv(fd, tmp, sizeof(tmp) - 1, 0);
        VJ_CHECK(n > 0 && std::string(tmp).rfind("HTTP/1.1 100 Continue", 0) == 0);
        send_str(fd, body);
        Resp r;
        std::string buf;
        VJ_CHECK(read_resp(fd, r, buf) && r.status == 200);
        ::close(fd);
    }

    // ---- errors
    {
        VJ_CHECK(request(port, "POST", "/v1/chat/completions", "{not json").status == 400);
        Resp e = request(port, "POST", "/v1/chat/completions", "{\"messages\":[]}");
        VJ_CHECK(e.status == 400 && parse_json(e.body).get("error")->get("message")->is_string());
        VJ_CHECK(request(port, "POST", "/v1/chat/completions", chat_body(",\"n\":2")).status == 400);
        VJ_CHECK(request(port, "POST", "/v1/chat/completions", chat_body(",\"temperature\":\"hot\"")).status == 400);
        VJ_CHECK(request(port, "GET", "/v1/chat/completions").status == 405);
        VJ_CHECK(request(port, "POST", "/health", "{}").status == 405);
        VJ_CHECK(request(port, "GET", "/nope").status == 404);
        VJ_CHECK(request(port, "POST", "/v1/chat/completions", std::string(70 * 1024, ' ')).status == 413);
        VJ_CHECK(request(port, "POST", "/v1/chat/completions", "", "Transfer-Encoding: chunked\r\n").status == 501);
        VJ_CHECK(request(port, "OPTIONS", "/v1/chat/completions").status == 204);
    }

    // ---- client disconnect cancels generation; queue-full -> 503
    {
        mock.delay_ms = 5;
        mock.repeat = 200;  // 2600 pieces ~ 13 s if never cancelled
        const int fd = connect_local(port);
        send_str(fd, req_text("POST", "/v1/chat/completions", chat_body(",\"stream\":true,\"max_tokens\":100000"),
                              "Connection: close\r\n"));
        wait_for([&] { return mock.running.load(); });
        // A second request while the only queue slot is taken.
        Resp busy = request(port, "POST", "/v1/chat/completions", chat_body(""));
        VJ_CHECK(busy.status == 503 && busy.headers.count("retry-after"));
        char tmp[256];
        VJ_CHECK(::recv(fd, tmp, sizeof(tmp), 0) > 0);  // stream started
        const auto t_close = std::chrono::steady_clock::now();
        ::close(fd);
        wait_for([&] { return !mock.running.load(); });
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_close).count();
        VJ_CHECK(mock.last_cancelled && mock.last_emitted < 200);
        VJ_CHECK(ms < 1000.0);
        wait_for([&] { return srv->stats().cancelled >= 1; });
        std::printf("  disconnect: generator stopped %.1f ms after client close, %d pieces emitted of %zu\n", ms,
                    mock.last_emitted.load(), kPieces.size() * 200);
        mock.delay_ms = 0;
        mock.repeat = 1;
        VJ_CHECK(request(port, "GET", "/health").status == 200);  // server still healthy
    }

    const auto st = srv->stats();
    VJ_CHECK(st.completions >= 8 && st.streamed >= 3 && st.rejected == 1 && st.errors == 0);

    // ---- clean shutdown with a generation in flight
    {
        mock.delay_ms = 5;
        mock.repeat = 200;
        const int fd = connect_local(port);
        send_str(fd, req_text("POST", "/v1/chat/completions", chat_body(",\"max_tokens\":100000"), ""));
        wait_for([&] { return mock.running.load(); });
        const auto t0 = std::chrono::steady_clock::now();
        srv->stop();
        srv->stop();  // idempotent
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        VJ_CHECK(!mock.running.load() && mock.last_cancelled && ms < 2000.0);
        ::close(fd);
        VJ_CHECK(connect_local(port) < 0);  // listener closed
        srv.reset();
        std::printf("  shutdown with in-flight generation: %.1f ms\n", ms);
    }

    // ---- IEngine adapter on the HEAD MockEngine
    {
        MockEngine engine("mock-engine");
        ServerOptions o;
        o.port = 0;
        o.model_id = "mock-engine";
        OpenAiServer s2(o, make_engine_generate_fn(engine));
        VJ_CHECK(s2.start(&err));
        const auto j = parse_json(request(s2.port(), "POST", "/v1/chat/completions", chat_body("")).body);
        VJ_CHECK(j.get("choices")->arr[0].get("message")->get("content")->str == "Hello from VegaJet mock engine on CPU.");
        VJ_CHECK(j.get("choices")->arr[0].get("finish_reason")->str == "stop");
        VJ_CHECK(j.get("usage")->get("prompt_tokens")->num > 0);
        const Sse sse = parse_sse(request(s2.port(), "POST", "/v1/chat/completions", chat_body(",\"stream\":true,\"max_tokens\":3")).body);
        VJ_CHECK(sse.done && sse.content == "Hello from VegaJet" && sse.finish == "length");
        s2.stop();
    }

    std::printf("test_server_sse: health/models/JSON/SSE/stop/length/keep-alive/pipelining/100-continue/errors/503/"
                "disconnect/shutdown/IEngine adapter OK (%llu requests served)\n",
                static_cast<unsigned long long>(st.requests));
    std::printf("PASS\n");
    return 0;
}
