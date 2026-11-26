// SPDX-License-Identifier: Noncommercial-1.0.0
// v1.2 serving levers, CPU only:
//   D3  greedy-only engines answer HTTP 400 for sampling parameters they do not implement (ServerOptions::greedy_only),
//       and keep accepting requests that hold the greedy defaults on both /v1/chat/completions and /v1/messages.
//   D2  the bounded SSE output queue (VEGAJET_V12_ASYNC_OUT, default on) must produce the exact same byte stream as the
//       inline writer, must not lose or reorder chunks for a slow reader, and must still cancel generation when the
//       client disconnects.
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
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <mutex>
#include <regex>
#include <thread>

using namespace vegajet;
using namespace vegajet::serve;

namespace {

const std::vector<std::string> kPieces = {"Hel", "lo", " wor", "ld", ",", " caf", "\xC3", "\xA9", " ", "\xF0\x9F",
                                          "\x9A\x80", " END", " tail"};
const std::string kFull = "Hello world, caf\xC3\xA9 \xF0\x9F\x9A\x80 END tail";

struct MockGen {
    std::atomic<int> delay_ms{0};
    std::atomic<int> repeat{1};
    std::atomic<int> last_emitted{0};
    std::atomic<bool> last_cancelled{false};
    std::atomic<bool> running{false};

    GenerateResult operator()(const GenerateRequest& r, const TokenSink& sink) {
        (void)r;
        running = true;
        GenerateResult res;
        res.prompt_tokens = 7;
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
};

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

std::string req_text(const std::string& method, const std::string& path, const std::string& body) {
    std::string s = method + " " + path + " HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n";
    if (!body.empty() || method == "POST")
        s += "Content-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) + "\r\n";
    return s + "\r\n" + body;
}

Resp request(uint16_t port, const std::string& method, const std::string& path, const std::string& body = "") {
    const int fd = connect_local(port);
    VJ_CHECK(fd >= 0);
    send_str(fd, req_text(method, path, body));
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

std::string chat_body(const std::string& extra) {
    return "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]" + extra + "}";
}

std::string anthropic_body(const std::string& extra) {
    return "{\"max_tokens\":16,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]" + extra + "}";
}

int count_of(const std::string& hay, const std::string& needle) {
    int n = 0;
    for (size_t p = hay.find(needle); p != std::string::npos; p = hay.find(needle, p + needle.size())) ++n;
    return n;
}

// The completion id embeds a per-request counter, so it is the only byte that legitimately differs between two
// otherwise identical responses.
std::string normalize_ids(const std::string& s) {
    static const std::regex id_re("chatcmpl-[0-9]+-[0-9]+");
    return std::regex_replace(s, id_re, "chatcmpl-ID");
}

// Concatenated delta content of an SSE body, parsed (not string-matched) so JSON escapes are decoded.
std::string sse_content(const std::string& body) {
    std::string out;
    size_t pos = 0;
    while (pos < body.size()) {
        const size_t e = body.find("\n\n", pos);
        if (e == std::string::npos) break;
        const std::string line = body.substr(pos, e - pos);
        pos = e + 2;
        if (line.rfind("data: {", 0) != 0) continue;  // [DONE]
        json::Value v;
        VJ_CHECK(json::parse(line.substr(6), v));
        const auto* ch = v.get("choices");
        if (!ch || ch->arr.empty()) continue;
        const auto* d = ch->arr[0].get("delta");
        const auto* c = d ? d->get("content") : nullptr;
        if (c && c->is_string()) out += c->str;
    }
    return out;
}

void wait_for(const std::function<bool()>& pred, int ms = 3000) {
    for (int i = 0; i < ms / 5 && !pred(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    VJ_CHECK(pred());
}

} // namespace

int main() {
    MockGen mock;
    ServerOptions opt;
    opt.port = 0;
    opt.model_id = "vegajet-v12";
    opt.context_length = 32768;
    opt.max_queue = 4;
    auto srv = std::make_unique<OpenAiServer>(opt, [&](const GenerateRequest& r, const TokenSink& s) { return mock(r, s); });
    std::string err;
    VJ_CHECK(srv->start(&err));
    const uint16_t port = srv->port();

    // ---- D3: greedy-only engine rejects unimplemented sampling parameters (OpenAI endpoint)
    {
        ServerOptions g = opt;
        g.port = 0;
        g.greedy_only = true;
        OpenAiServer gsrv(g, [&](const GenerateRequest& r, const TokenSink& s) { return mock(r, s); });
        VJ_CHECK(gsrv.start(&err));
        const uint16_t gp = gsrv.port();

        struct Case { const char* extra; const char* field; };
        const Case rejected[] = {
            {",\"temperature\":0.7", "temperature"}, {",\"top_k\":40", "top_k"},     {",\"top_p\":0.9", "top_p"},
            {",\"min_p\":0.05", "min_p"},            {",\"repeat_penalty\":1.1", "repeat_penalty"},
            {",\"presence_penalty\":0.5", "presence_penalty"},
            {",\"frequency_penalty\":0.5", "frequency_penalty"},
            {",\"logprobs\":true", "logprobs"},      {",\"top_logprobs\":3", "top_logprobs"},
        };
        for (const auto& c : rejected) {
            const Resp r = request(gp, "POST", "/v1/chat/completions", chat_body(c.extra));
            const auto j = parse_json(r.body);
            VJ_CHECK(r.status == 400);
            VJ_CHECK(j.get("error")->get("type")->str == "invalid_request_error");
            const std::string msg = j.get("error")->get("message")->str;
            if (msg.find(c.field) == std::string::npos) {
                std::fprintf(stderr, "  missing '%s' in: %s\n", c.field, msg.c_str());
                VJ_CHECK(false);
            }
            if (msg.find("greedily") == std::string::npos) {
                std::fprintf(stderr, "  no reason in: %s\n", msg.c_str());
                VJ_CHECK(false);
            }
        }
        // A request that names no knob at all is accepted (the API defaults differ per endpoint; they are not requests).
        VJ_CHECK(request(gp, "POST", "/v1/chat/completions", chat_body("")).status == 200);
        VJ_CHECK(request(gp, "POST", "/v1/messages", anthropic_body("")).status == 200);
        // Explicit greedy defaults (and the non-sampling knobs) stay accepted; seed cannot change a greedy draw.
        const Resp ok = request(gp, "POST", "/v1/chat/completions",
                                chat_body(",\"temperature\":0,\"top_p\":1,\"top_k\":0,\"min_p\":0,"
                                          "\"repeat_penalty\":1,\"presence_penalty\":0,\"frequency_penalty\":0,"
                                          "\"seed\":1234,\"logprobs\":false,\"max_tokens\":32"));
        VJ_CHECK(ok.status == 200);
        VJ_CHECK(parse_json(ok.body).get("choices")->arr[0].get("message")->get("content")->str == kFull);
        // /v1/messages goes through the shared post-parse check.
        const Resp anth = request(gp, "POST", "/v1/messages", anthropic_body(",\"temperature\":0.7"));
        VJ_CHECK(anth.status == 400);
        VJ_CHECK(parse_json(anth.body).get("error")->get("message")->str.find("temperature") != std::string::npos);
        VJ_CHECK(request(gp, "POST", "/v1/messages", anthropic_body(",\"temperature\":0,\"top_k\":0")).status == 200);
        gsrv.stop();

        // A sampling-capable generator (the default) still accepts the same request unchanged.
        const Resp s = request(port, "POST", "/v1/chat/completions", chat_body(",\"temperature\":0.7,\"top_k\":40"));
        VJ_CHECK(s.status == 200 && parse_json(s.body).get("choices")->arr[0].get("message")->get("content")->str == kFull);
        std::printf("  greedy-only: %zu sampling knobs rejected with 400, greedy defaults accepted\n",
                    sizeof(rejected) / sizeof(rejected[0]));
    }

    // ---- D2: the async writer produces the same bytes as the inline writer
    {
        const std::string body = chat_body(",\"stream\":true,\"stop\":\"END\",\"max_tokens\":64");
        ::unsetenv("VEGAJET_V12_ASYNC_OUT");  // default: queue on
        const Resp queued = request(port, "POST", "/v1/chat/completions", body);
        ::setenv("VEGAJET_V12_ASYNC_OUT", "0", 1);  // inline writer
        const Resp inline_ = request(port, "POST", "/v1/chat/completions", body);
        ::unsetenv("VEGAJET_V12_ASYNC_OUT");
        VJ_CHECK(queued.status == 200 && inline_.status == 200);
        VJ_CHECK(normalize_ids(queued.body) == normalize_ids(inline_.body));
        VJ_CHECK(count_of(queued.body, "data: ") >= 10);
        VJ_CHECK(sse_content(queued.body) == "Hello world, caf\xC3\xA9 \xF0\x9F\x9A\x80 ");  // stop "END": trailing space stays
        VJ_CHECK(queued.body.find("\"finish_reason\":\"stop\"") != std::string::npos);
        VJ_CHECK(queued.body.rfind("data: [DONE]\n\n") == queued.body.size() - 14);

        // Same check without a stop sequence and with usage: identical apart from the timings (two runs, two clocks)
        // and the per-request id, so only the content sequence and the number of chunks are compared.
        const std::string body2 = chat_body(",\"stream\":true");
        const Resp q2 = request(port, "POST", "/v1/chat/completions", body2);
        ::setenv("VEGAJET_V12_ASYNC_OUT", "0", 1);
        const Resp i2 = request(port, "POST", "/v1/chat/completions", body2);
        ::unsetenv("VEGAJET_V12_ASYNC_OUT");
        VJ_CHECK(normalize_ids(q2.body) == normalize_ids(i2.body));
        std::printf("  async queue: byte-identical to inline writer (%d events)\n", count_of(q2.body, "data: "));
    }

    // ---- D2: a slow reader gets every chunk, in order (bounded queue, coalesced writes)
    {
        const std::string body = chat_body(",\"stream\":true");
        const int fd = connect_local(port);
        send_str(fd, req_text("POST", "/v1/chat/completions", body));
        std::string acc;
        char tmp[16];
        // Drain slowly (tiny reads) so the writer cannot always keep up: the stream must still be complete.
        while (acc.find("data: [DONE]\n\n") == std::string::npos) {
            const ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
            if (n <= 0) break;
            acc.append(tmp, static_cast<size_t>(n));
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ::close(fd);
        VJ_CHECK(acc.find("data: [DONE]\n\n") != std::string::npos);
        const size_t hdr = acc.find("\r\n\r\n");
        VJ_CHECK(hdr != std::string::npos);
        const std::string stream = acc.substr(hdr + 4);
        VJ_CHECK(sse_content(stream) == kFull);
        VJ_CHECK(count_of(stream, "data: ") >= 10);  // chunks are not merged into one event
        std::printf("  slow reader: full stream of %zu bytes delivered in 16-byte reads\n", stream.size());
    }

    // ---- D2: a client that disappears still stops the generation (writer detects the failed send)
    {
        mock.delay_ms = 5;
        mock.repeat = 200;
        const int fd = connect_local(port);
        send_str(fd, req_text("POST", "/v1/chat/completions", chat_body(",\"stream\":true,\"max_tokens\":100000")));
        wait_for([&] { return mock.running.load(); });
        char tmp[256];
        VJ_CHECK(::recv(fd, tmp, sizeof(tmp), 0) > 0);  // stream started
        const auto t_close = std::chrono::steady_clock::now();
        ::close(fd);
        wait_for([&] { return !mock.running.load(); });
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_close).count();
        VJ_CHECK(mock.last_cancelled && mock.last_emitted < 2600);
        VJ_CHECK(ms < 2000.0);
        wait_for([&] { return srv->stats().cancelled >= 1; });
        mock.delay_ms = 0;
        mock.repeat = 1;
        VJ_CHECK(request(port, "GET", "/health").status == 200);
        std::printf("  disconnect: generation stopped %.1f ms after client close (%d pieces)\n", ms, mock.last_emitted.load());
    }

    const auto st = srv->stats();
    VJ_CHECK(st.errors == 0);
    srv->stop();
    std::printf("test_serve_v12: greedy-only 400s, async SSE equality/order, slow reader, disconnect OK\n");
    std::printf("PASS\n");
    return 0;
}
