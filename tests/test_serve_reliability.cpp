// SPDX-License-Identifier: Noncommercial-1.0.0
// HTTP server reliability findings from bench/v12/AUDIT-server.md, CPU only, against a mock generator:
//   S-B1  a request body cannot abort the process: the JSON parse is bounded by body size and node count and its
//         allocation failures answer 413/400 instead of std::terminate.
//   S-B2  at the connection limit the server answers 503 with a body and keeps /health and /metrics reachable.
//   S-M1  a half-closed client (SHUT_WR) still receives the full response, streaming and not.
//   S-M2  a request cancelled by the client is logged and counted as cancelled (status 499), not as a 200.
//   S-M3  the model field is length-checked, an unknown model answers 404 and the model is not repeated in every
//         SSE chunk.
//   S-M4  max_tokens is capped (--max-tokens-cap) and an over-long prompt answers 400/413 with a clear message.
//   S-M5  an optional bearer / x-api-key is checked in constant time; CORS is only sent when an origin is configured.
#include "core/engine_interface.hpp"
#include "core/json_lite.hpp"
#include "core/metrics.hpp"
#include "core/server.hpp"
#include "core/server_engine_adapter.hpp"

#include "tests/test_common.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace vegajet;
using namespace vegajet::serve;

namespace {

const std::vector<std::string> kPieces = {"Hel", "lo", " wor", "ld"};
const std::string kFull = "Hello world";

struct MockGen {
    std::atomic<int> delay_ms{0};
    std::atomic<bool> last_cancelled{false};

    GenerateResult operator()(const GenerateRequest& r, const TokenSink& sink) {
        (void)r;
        GenerateResult res;
        res.prompt_tokens = 7;
        bool cancelled = false;
        for (const auto& p : kPieces) {
            if (delay_ms) std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
            if (!sink(p)) { cancelled = true; break; }
        }
        last_cancelled = cancelled;
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
    bool complete = false;
};

// Reads one response. SSE (no content-length) reads until the server closes.
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
        if (c == std::string::npos) { pos = e; continue; }
        std::string k = line.substr(0, c);
        for (auto& ch : k) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        r.headers[k] = line.substr(line.find_first_not_of(' ', c + 1));
        pos = e;
    }
    buf.erase(0, hdr_end + 4);
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
    r.complete = true;
    return true;
}

std::string chat_head(int body_len, bool keep_alive, const std::string& extra = {}) {
    return "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n" +
           std::string("Content-Length: ") + std::to_string(body_len) + "\r\n" +
           (keep_alive ? "" : "Connection: close\r\n") + extra + "\r\n";
}

std::string get_head(const std::string& path) {
    return "GET " + path + " HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
}

Resp one(int port, const std::string& raw) {
    const int fd = connect_local(port);
    VJ_CHECK(fd >= 0);
    send_str(fd, raw);
    Resp r;
    std::string buf;
    const bool ok = read_resp(fd, r, buf);
    ::close(fd);
    VJ_CHECK(ok);
    return r;
}

Resp chat(int port, const std::string& body, const std::string& extra = {}) {
    return one(port, chat_head(static_cast<int>(body.size()), false, extra) + body);
}

bool wait_until(const std::function<bool()>& pred, int ms = 5000) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    while (std::chrono::steady_clock::now() < end) {
        if (pred()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return pred();
}

std::string read_text_file(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    VJ_CHECK(f != nullptr);
    std::string out;
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
    std::fclose(f);
    return out;
}

// Sends on an already open connection (keep-alive), reusing its read buffer.
Resp chat_on(int fd, std::string& buf, const std::string& body) {
    send_str(fd, chat_head(static_cast<int>(body.size()), true) + body);
    Resp r;
    const bool ok = read_resp(fd, r, buf);
    VJ_CHECK(ok);
    return r;
}

std::string chat_body(const std::string& model, const std::string& content, const std::string& tail = {}) {
    return "{\"model\":" + json::quote(model) + ",\"messages\":[{\"role\":\"user\",\"content\":" +
           json::quote(content) + "}]" + tail + "}";
}

// ------------------------------------------------------------------------------------------------ S-B1
// A body made of delimiters costs ~50 bytes of heap per byte of input; the parse must refuse it, not abort.
void test_b1_json_limits() {
    json::Value v;
    {
        std::string many = "[";
        for (int i = 0; i < 200; ++i) many += i ? ",{}" : "{}";
        many += "]";
        json::Parser p(many, 64, 100);
        VJ_CHECK(!p.parse(v));
        VJ_CHECK(p.node_limit_hit());
        VJ_CHECK(!p.depth_limit_hit());
        VJ_CHECK(p.nodes() == 101);
    }
    {
        const std::string deep(200, '[');
        json::Parser p(deep, 64, 65536);
        VJ_CHECK(!p.parse(v));
        VJ_CHECK(p.depth_limit_hit());
    }
    {  // a normal body still parses
        json::Parser p("{\"a\":[1,2,3],\"b\":\"x\"}", 64, 100);
        VJ_CHECK(p.parse(v));
        VJ_CHECK(!p.node_limit_hit());
    }
}

void test_b1_server_survives() {
    MockGen gen;
    ServerOptions opt;
    opt.port = 0;
    opt.model_id = "m";
    opt.max_json_nodes = 64;
    opt.max_chat_body_bytes = 16384;
    OpenAiServer srv(opt, [&](const GenerateRequest& r, const TokenSink& s) { return gen(r, s); });
    std::string err;
    VJ_CHECK(srv.start(&err));

    // Node bomb: many JSON values in a body far below max_body_bytes.
    std::string bomb = "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":[";
    for (int i = 0; i < 100; ++i) bomb += i ? ",{\"type\":\"text\",\"text\":\"x\"}" : "{\"type\":\"text\",\"text\":\"x\"}";
    bomb += "]}]}";
    VJ_CHECK(bomb.size() < 16384);
    Resp r = chat(srv.port(), bomb);
    VJ_CHECK(r.status == 413);
    VJ_CHECK(r.body.find("JSON values") != std::string::npos);

    // Oversized body: refused before the parse.
    std::string big = "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"";
    big.append(20000, 'a');
    big += "\"}]}";
    r = chat(srv.port(), big);
    VJ_CHECK(r.status == 413);
    VJ_CHECK(r.body.find("larger than") != std::string::npos);

    srv.stop();

    // Depth and the normal path on a server whose node budget allows deep nesting to reach the depth check.
    MockGen gen2;
    ServerOptions opt2 = opt;
    opt2.max_json_nodes = 65536;
    OpenAiServer srv2(opt2, [&](const GenerateRequest& r, const TokenSink& s) { return gen2(r, s); });
    VJ_CHECK(srv2.start(&err));

    std::string deep = "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":";
    deep.append(300, '[');
    deep.append(300, ']');
    deep += "}]}";
    r = chat(srv2.port(), deep);
    VJ_CHECK(r.status == 400);
    VJ_CHECK(r.body.find("64 JSON levels") != std::string::npos);

    // The process is still serving: a normal request and /health work.
    r = chat(srv2.port(), chat_body("m", "hi"));
    VJ_CHECK(r.status == 200);
    r = one(srv2.port(), get_head("/health"));
    VJ_CHECK(r.status == 200);
    srv2.stop();
    VJ_CHECK(srv2.stats().errors == 0);
}

// ------------------------------------------------------------------------------------------------ S-B2
// 256 idle sockets used to make every endpoint, /health included, answer with a bare RST.
void test_b2_connection_limit() {
    MockGen gen;
    ServerOptions opt;
    opt.port = 0;
    opt.model_id = "m";
    opt.max_connections = 4;
    opt.reserve_connections = 2;
    opt.metrics_text = [] { return std::string("# metrics ok\n"); };
    OpenAiServer srv(opt, [&](const GenerateRequest& r, const TokenSink& s) { return gen(r, s); });
    std::string err;
    VJ_CHECK(srv.start(&err));

    const std::string keepalive_get_health = "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n";
    std::vector<int> conns;
    std::vector<std::string> bufs;
    std::vector<Resp> resps;
    for (int i = 0; i < 6; ++i) {  // 4 normal + 2 reserve, in accept order
        const int fd = connect_local(srv.port());
        VJ_CHECK(fd >= 0);
        send_str(fd, keepalive_get_health);
        Resp r;
        std::string b;
        VJ_CHECK(read_resp(fd, r, b));
        VJ_CHECK(r.status == 200);  // a reserve socket still reaches /health
        conns.push_back(fd);
        bufs.push_back(b);
        resps.push_back(r);
    }

    // /metrics is reachable on a reserve socket too.
    send_str(conns[4], "GET /metrics HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n");
    Resp r;
    VJ_CHECK(read_resp(conns[4], r, bufs[4]));
    VJ_CHECK(r.status == 200);
    VJ_CHECK(r.body.find("metrics ok") != std::string::npos);

    // A normal socket in the pool still generates.
    r = chat_on(conns[0], bufs[0], chat_body("m", "hi"));
    VJ_CHECK(r.status == 200);

    // The next socket is over both limits: it gets a 503 with a body, not a reset.
    const int reject = connect_local(srv.port());
    VJ_CHECK(reject >= 0);
    send_str(reject, keepalive_get_health);
    Resp rj;
    std::string rb;
    VJ_CHECK(read_resp(reject, rj, rb));
    VJ_CHECK(rj.status == 503);
    VJ_CHECK(rj.headers.count("retry-after") == 1);
    VJ_CHECK(rj.body.find("connection limit") != std::string::npos);
    ::close(reject);

    // A reserve socket may not generate (the 503 closes it).
    r = chat_on(conns[5], bufs[5], chat_body("m", "hi"));
    VJ_CHECK(r.status == 503);
    VJ_CHECK(r.headers.count("retry-after") == 1);
    VJ_CHECK(r.body.find("connection limit") != std::string::npos);

    for (int fd : conns) ::close(fd);
    srv.stop();
}

// ------------------------------------------------------------------------------------------------ S-M1
// A client that half-closes after its request (legal HTTP) used to get 0 bytes for a non-stream request and a
// truncated SSE stream: POLLRDHUP alone was taken as "gone".
void test_m1_half_close() {
    MockGen gen;
    ServerOptions opt;
    opt.port = 0;
    opt.model_id = "m";
    OpenAiServer srv(opt, [&](const GenerateRequest& r, const TokenSink& s) { return gen(r, s); });
    std::string err;
    VJ_CHECK(srv.start(&err));

    {  // non-stream: the full JSON body arrives
        const std::string body = chat_body("m", "hi");
        const int fd = connect_local(srv.port());
        VJ_CHECK(fd >= 0);
        send_str(fd, chat_head(static_cast<int>(body.size()), true) + body);
        VJ_CHECK(::shutdown(fd, SHUT_WR) == 0);
        Resp r;
        std::string buf;
        VJ_CHECK(read_resp(fd, r, buf));
        VJ_CHECK(r.status == 200);
        VJ_CHECK(r.body.find("Hello world") != std::string::npos);
        ::close(fd);
    }
    {  // stream: every chunk and the terminator arrive
        const std::string body = chat_body("m", "hi", ",\"stream\":true");
        const int fd = connect_local(srv.port());
        VJ_CHECK(fd >= 0);
        send_str(fd, chat_head(static_cast<int>(body.size()), true) + body);
        VJ_CHECK(::shutdown(fd, SHUT_WR) == 0);
        Resp r;
        std::string buf;
        VJ_CHECK(read_resp(fd, r, buf));
        VJ_CHECK(r.status == 200);
        VJ_CHECK(r.body.find("data: [DONE]") != std::string::npos);
        for (const auto& p : kPieces) VJ_CHECK(r.body.find(p) != std::string::npos);
        ::close(fd);
    }
    srv.stop();
    VJ_CHECK(srv.stats().cancelled == 0);  // a half-close is not a cancellation
}

// ------------------------------------------------------------------------------------------------ S-M2
// A client abort was logged as status 200 / cancelled:false and never reached vegajet_requests_cancelled_total.
// SlowEngine emits one piece per 20 ms through the real adapter (core/server_engine_adapter), which is where a
// client-gone sink result becomes FinishReason::Cancelled.
struct SlowEngine : IEngine {
    MockEngine inner;
    ModelInfo model_info() const override { return inner.model_info(); }
    const BpeTokenizer& tokenizer() const override { return inner.tokenizer(); }
    std::string format_chat(const std::vector<ChatMessage>& m) const override { return inner.format_chat(m); }
    void generate(const std::vector<int32_t>&, const GenerationParams& gp, const TokenCallback& on) override {
        for (int i = 0; i < 8; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            TokenEvent ev;
            ev.token_id = 3;
            ev.text = "tok";
            ev.completion_tokens = static_cast<uint32_t>(i + 1);
            if (!on(ev)) return;
        }
        TokenEvent eos;
        eos.is_eos = true;
        eos.completion_tokens = 8;
        if (gp.max_tokens > 8) on(eos);
    }
};

void test_m2_cancelled_accounting() {
    char dir[] = "/tmp/vj_rel_XXXXXX";
    VJ_CHECK(::mkdtemp(dir) != nullptr);
    const std::string path = std::string(dir) + "/requests.jsonl";

    metrics::Registry reg;
    metrics::JsonlLogger log;
    std::string err;
    VJ_CHECK(log.open(path, &err));

    SlowEngine engine;
    ServerOptions opt;
    opt.port = 0;
    opt.model_id = "m";
    OpenAiServer srv(opt, metrics::instrument(make_engine_generate_fn(engine), reg, &log));
    VJ_CHECK(srv.start(&err));

    const std::string body = chat_body("m", "hi", ",\"stream\":true,\"max_tokens\":8");
    const int fd = connect_local(srv.port());
    VJ_CHECK(fd >= 0);
    send_str(fd, chat_head(static_cast<int>(body.size()), true) + body);
    // Read the SSE head, then walk away mid-stream with a RST so the next write fails.
    char tmp[256];
    VJ_CHECK(::recv(fd, tmp, sizeof(tmp), 0) > 0);
    linger lg{1, 0};
    setsockopt(fd, SOL_SOCKET, SO_LINGER, &lg, sizeof(lg));
    ::close(fd);

    VJ_CHECK(wait_until([&] { return log.lines() == 1; }, 5000));
    srv.stop();
    VJ_CHECK(srv.stats().cancelled == 1);
    VJ_CHECK(srv.stats().completions == 0);

    const std::string rec = read_text_file(path);
    VJ_CHECK(rec.find("\"cancelled\":true") != std::string::npos);
    VJ_CHECK(rec.find("\"status\":499") != std::string::npos);
    const std::string expo = reg.exposition(nullptr);
    VJ_CHECK(expo.find("vegajet_requests_cancelled_total 1") != std::string::npos);
}

}  // namespace

int main() {
    test_b1_json_limits();
    test_b1_server_survives();
    test_b2_connection_limit();
    test_m1_half_close();
    test_m2_cancelled_accounting();
    std::printf("test_serve_reliability: S-B1 bounded JSON parse, S-B2 connection limit 503 + health reserve,\n"
                "                        S-M1 half-closed client gets the full response, S-M2 cancellation is logged as 499\n");
    std::printf("PASS\n");
    return 0;
}
