// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/server.hpp"

#include "core/anthropic_shim.hpp"

#include "core/json_lite.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace vegajet::serve {

const char* finish_reason_name(FinishReason f) {
    switch (f) {
        case FinishReason::Stop: return "stop";
        case FinishReason::Length: return "length";
        case FinishReason::Cancelled: return "cancelled";
        case FinishReason::Error: return "error";
        case FinishReason::ToolCalls: return "tool_calls";
    }
    return "stop";
}

// =================================================================================================
// StopMatcher
// =================================================================================================
namespace {

// Length of `s` without a trailing incomplete UTF-8 sequence.
size_t utf8_complete_len(std::string_view s) {
    const size_t n = s.size();
    for (size_t back = 1; back <= 4 && back <= n; ++back) {
        const auto c = static_cast<unsigned char>(s[n - back]);
        if ((c & 0xC0u) == 0x80u) continue;  // continuation byte: keep looking for the lead byte
        const size_t need = c < 0x80u ? 1 : (c & 0xE0u) == 0xC0u ? 2 : (c & 0xF0u) == 0xE0u ? 3 : (c & 0xF8u) == 0xF0u ? 4 : 1;
        return back < need ? n - back : n;
    }
    return n;  // only continuation bytes: invalid input, pass through
}

} // namespace

StopMatcher::StopMatcher(std::vector<std::string> stops) {
    for (auto& s : stops)
        if (!s.empty()) stops_.push_back(std::move(s));
}

std::string StopMatcher::push(std::string_view piece) {
    if (stopped_) return {};
    pending_.append(piece.data(), piece.size());
    size_t cut = std::string::npos;
    for (const auto& s : stops_) {
        const size_t p = pending_.find(s);
        if (p < cut) {
            cut = p;
            matched_ = s;
        }
    }
    if (cut != std::string::npos) {
        stopped_ = true;
        std::string out = pending_.substr(0, cut);
        pending_.clear();
        return out;
    }
    size_t hold = 0;  // longest suffix of pending_ that is a proper prefix of some stop string
    for (const auto& s : stops_) {
        for (size_t len = std::min(s.size() - 1, pending_.size()); len > hold; --len) {
            if (pending_.compare(pending_.size() - len, len, s, 0, len) == 0) { hold = len; break; }
        }
    }
    const size_t emit = utf8_complete_len(std::string_view(pending_).substr(0, pending_.size() - hold));
    std::string out = pending_.substr(0, emit);
    pending_.erase(0, emit);
    return out;
}

std::string StopMatcher::flush() {
    if (stopped_) return {};
    std::string out;
    out.swap(pending_);
    const size_t ok = utf8_complete_len(out);
    if (ok < out.size()) out.replace(ok, std::string::npos, "\xEF\xBF\xBD");
    return out;
}

// =================================================================================================
// HTTP helpers
// =================================================================================================
namespace {

using Clock = std::chrono::steady_clock;

int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count();
}

std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string trim(const std::string& s) {
    const size_t b = s.find_first_not_of(" \t");
    if (b == std::string::npos) return {};
    return s.substr(b, s.find_last_not_of(" \t") - b + 1);
}

struct HttpRequest {
    std::string method, target, version;
    std::vector<std::pair<std::string, std::string>> headers;  // names lower-cased
    std::string body;
    bool keep_alive = true;
};

enum class Parse { Incomplete, Done, Error };

struct ParseOut {
    Parse status = Parse::Incomplete;
    size_t consumed = 0;
    int err_status = 400;
    const char* err_msg = "bad request";
    bool expect_continue = false;  // headers complete, body pending, client sent Expect: 100-continue
};

ParseOut parse_http(const std::string& buf, size_t max_hdr, size_t max_body, HttpRequest& req) {
    ParseOut r;
    const size_t hdr_end = buf.find("\r\n\r\n");
    if (hdr_end == std::string::npos) {
        if (buf.size() > max_hdr) { r.status = Parse::Error; r.err_status = 431; r.err_msg = "request header too large"; }
        return r;
    }
    if (hdr_end > max_hdr) { r.status = Parse::Error; r.err_status = 431; r.err_msg = "request header too large"; return r; }
    req = HttpRequest{};
    size_t line_end = buf.find("\r\n");
    const std::string line = buf.substr(0, line_end);
    const size_t s1 = line.find(' '), s2 = line.rfind(' ');
    if (s1 == std::string::npos || s2 == s1) { r.status = Parse::Error; r.err_msg = "malformed request line"; return r; }
    req.method = line.substr(0, s1);
    req.target = line.substr(s1 + 1, s2 - s1 - 1);
    req.version = line.substr(s2 + 1);
    if (req.version.rfind("HTTP/1.", 0) != 0 || req.target.empty()) {
        r.status = Parse::Error; r.err_status = 505; r.err_msg = "only HTTP/1.x is supported"; return r;
    }
    size_t content_length = 0;
    std::string connection, expect;
    for (size_t pos = line_end + 2; pos < hdr_end;) {
        const size_t e = buf.find("\r\n", pos);
        const std::string h = buf.substr(pos, e - pos);
        pos = e + 2;
        const size_t colon = h.find(':');
        if (colon == std::string::npos || colon == 0) { r.status = Parse::Error; r.err_msg = "malformed header"; return r; }
        std::string name = lower(h.substr(0, colon));
        std::string value = trim(h.substr(colon + 1));
        if (name == "content-length") {
            if (value.empty() || value.size() > 12 || value.find_first_not_of("0123456789") != std::string::npos) {
                r.status = Parse::Error; r.err_msg = "invalid Content-Length"; return r;
            }
            content_length = std::stoull(value);
            if (content_length > max_body) { r.status = Parse::Error; r.err_status = 413; r.err_msg = "request body too large"; return r; }
        } else if (name == "transfer-encoding") {
            r.status = Parse::Error; r.err_status = 501; r.err_msg = "chunked request bodies are not supported"; return r;
        } else if (name == "connection") {
            connection = lower(value);
        } else if (name == "expect") {
            expect = lower(value);
        }
        req.headers.emplace_back(std::move(name), std::move(value));
    }
    req.keep_alive = req.version == "HTTP/1.1" ? connection.find("close") == std::string::npos
                                               : connection.find("keep-alive") != std::string::npos;
    const size_t body_at = hdr_end + 4;
    if (buf.size() - body_at < content_length) {
        r.expect_continue = expect == "100-continue";
        return r;
    }
    req.body = buf.substr(body_at, content_length);
    r.consumed = body_at + content_length;
    r.status = Parse::Done;
    return r;
}

const char* reason_phrase(int status) {
    switch (status) {
        case 200: return "OK";
        case 204: return "No Content";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 413: return "Payload Too Large";
        case 431: return "Request Header Fields Too Large";
        case 500: return "Internal Server Error";
        case 501: return "Not Implemented";
        case 503: return "Service Unavailable";
        case 505: return "HTTP Version Not Supported";
    }
    return "Error";
}

std::string http_response(int status, std::string_view ctype, std::string_view body, bool keep_alive,
                          const std::string& extra = {}) {
    std::string h = "HTTP/1.1 " + std::to_string(status) + " " + reason_phrase(status) + "\r\n";
    if (!ctype.empty()) h += "Content-Type: " + std::string(ctype) + "\r\n";
    h += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    h += keep_alive ? "Connection: keep-alive\r\n" : "Connection: close\r\n";
    h += "Access-Control-Allow-Origin: *\r\n";
    h += extra;
    h += "\r\n";
    h.append(body.data(), body.size());
    return h;
}

std::string error_json(std::string_view msg, std::string_view type, std::string_view code = {}) {
    return "{\"error\":{\"message\":" + json::quote(msg) + ",\"type\":" + json::quote(type) +
           ",\"param\":null,\"code\":" + (code.empty() ? std::string("null") : json::quote(code)) + "}}";
}

std::string error_response(int status, std::string_view msg, bool keep_alive, const std::string& extra = {},
                           std::string_view code = {}) {
    const char* type = status == 503 ? "server_busy" : status >= 500 ? "server_error" : "invalid_request_error";
    return http_response(status, "application/json", error_json(msg, type, code), keep_alive, extra);
}

// Bounded request-body parse. The JSON Value tree costs ~50x the body size, so a hostile body used to reach
// std::bad_alloc and abort the process (S-B1). Limits are checked while parsing; allocation failures are caught and
// turned into 413/400. Returns 0 on success, otherwise the HTTP status to answer with.
int parse_body_bounded(std::string_view body, size_t max_nodes, json::Value& out, std::string& err) {
    json::Parser p(body, 64, max_nodes);
    try {
        if (p.parse(out)) return 0;
    } catch (const std::bad_alloc&) {
        err = "request body is too expensive to parse (out of memory); send a smaller request";
        return 413;
    } catch (const std::exception& e) {
        err = std::string("request body could not be parsed: ") + e.what();
        return 400;
    }
    if (p.node_limit_hit()) {
        err = "request body holds more than " + std::to_string(max_nodes) + " JSON values";
        return 413;
    }
    if (p.depth_limit_hit()) {
        err = "request body nests more than 64 JSON levels deep";
        return 400;
    }
    err = "request body is not valid JSON";
    return 400;
}

// Rough prompt size in tokens from the message bytes (>= 4 bytes per token for English and code, so this is a lower
// bound). It rejects a prompt that cannot fit before any encoding work; the engine adapter repeats the check with the
// exact token count once the prompt is encoded (S-M4).
uint32_t estimate_prompt_tokens(const GenerateRequest& r) {
    size_t bytes = 0;
    for (const auto& m : r.messages) bytes += m.content.size() + m.role.size() + 8;
    return static_cast<uint32_t>(bytes / 4 + 8);
}

std::string fmt_num(double v) {
    char b[64];
    std::snprintf(b, sizeof(b), "%.3f", v);
    return b;
}

bool get_number(const json::Value& root, const char* key, double& out, std::string& err) {
    const auto* v = root.get(key);
    if (!v || v->is_null()) return true;
    if (!v->is_number()) { err = std::string("'") + key + "' must be a number"; return false; }
    out = v->num;
    return true;
}

// OpenAI chat request -> GenerateRequest. Unknown fields are ignored.
bool parse_chat(const json::Value& root, const ServerOptions& opt, GenerateRequest& g, bool& include_usage,
                std::string& err) {
    if (!root.is_object()) { err = "request body must be a JSON object"; return false; }
    if (const auto* m = root.get("model"); m && !m->is_null()) {
        if (!m->is_string()) { err = "'model' must be a string"; return false; }
        if (m->str.size() > opt.max_model_name_len) {
            err = "'model' must be at most " + std::to_string(opt.max_model_name_len) + " bytes";
            return false;
        }
        g.model = m->str;
    }
    if (g.model.empty()) g.model = opt.model_id;
    const auto* msgs = root.get("messages");
    if (!msgs || !msgs->is_array() || msgs->arr.empty()) { err = "'messages' must be a non-empty array"; return false; }
    for (const auto& m : msgs->arr) {
        const auto* role = m.get("role");
        if (!role || !role->is_string()) { err = "every message needs a string 'role'"; return false; }
        ChatTurn t;
        t.role = role->str;
        const auto* c = m.get("content");
        if (c && c->is_string()) {
            t.content = c->str;
        } else if (c && c->is_array()) {
            for (const auto& part : c->arr) {
                const auto* type = part.get("type");
                const auto* text = part.get("text");
                if (type && type->is_string() && type->str == "text" && text && text->is_string()) t.content += text->str;
                else { err = "only text content parts are supported"; return false; }
            }
        } else if (c && !c->is_null()) {
            err = "message 'content' must be a string, an array of text parts, or null";
            return false;
        }
        if (!toolcall::parse_message_tools(m, t, err)) return false;
        g.messages.push_back(std::move(t));
    }
    if (!toolcall::parse_tool_request(root, g.tools, err)) return false;
    if (const auto* rf = root.get("reasoning_format"); rf && !rf->is_null()) {
        if (!rf->is_string()) { err = "'reasoning_format' must be a string"; return false; }
        g.parse_reasoning = rf->str != "none";
    }
    if (const auto* n = root.get("n"); n && !n->is_null() && !(n->is_number() && n->num == 1.0)) {
        err = "only n = 1 is supported";
        return false;
    }
    double max_tokens = opt.default_max_tokens;
    if (!get_number(root, "max_tokens", max_tokens, err) || !get_number(root, "max_completion_tokens", max_tokens, err))
        return false;
    if (max_tokens < 1 || max_tokens > 1e9) { err = "'max_tokens' must be >= 1"; return false; }
    g.max_tokens = static_cast<int32_t>(max_tokens);
    if (opt.max_tokens_cap > 0) g.max_tokens = std::min(g.max_tokens, opt.max_tokens_cap);

    double temperature = 1.0, top_p = 1.0, top_k = 0, min_p = 0.0, rep = 1.0, seed = 0;
    if (!get_number(root, "temperature", temperature, err) || !get_number(root, "top_p", top_p, err) ||
        !get_number(root, "top_k", top_k, err) || !get_number(root, "min_p", min_p, err) ||
        !get_number(root, "repeat_penalty", rep, err) || !get_number(root, "seed", seed, err))
        return false;
    if (temperature < 0 || temperature > 100) { err = "'temperature' must be in [0, 100]"; return false; }
    if (!(top_p > 0 && top_p <= 1)) { err = "'top_p' must be in (0, 1]"; return false; }
    if (min_p < 0 || min_p > 1) { err = "'min_p' must be in [0, 1]"; return false; }
    if (rep <= 0) { err = "'repeat_penalty' must be > 0"; return false; }
    g.sampling.temperature = static_cast<float>(temperature);
    g.sampling.top_p = static_cast<float>(top_p);
    g.sampling.top_k = static_cast<int32_t>(std::max(0.0, top_k));
    g.sampling.min_p = static_cast<float>(min_p);
    g.sampling.repeat_penalty = static_cast<float>(rep);
    g.sampling.seed = static_cast<uint64_t>(std::max(0.0, seed));

    if (const auto* st = root.get("stop"); st && !st->is_null()) {
        if (st->is_string()) g.stop.push_back(st->str);
        else if (st->is_array()) {
            for (const auto& s : st->arr) {
                if (!s.is_string()) { err = "'stop' entries must be strings"; return false; }
                g.stop.push_back(s.str);
            }
        } else { err = "'stop' must be a string or an array of strings"; return false; }
        if (g.stop.size() > 16) { err = "at most 16 stop sequences"; return false; }
    }
    if (const auto* s = root.get("stream"); s && !s->is_null()) {
        if (!s->is_bool()) { err = "'stream' must be a boolean"; return false; }
        g.stream = s->b;
    }
    include_usage = false;
    if (const auto* so = root.get("stream_options"); so && so->is_object())
        if (const auto* iu = so->get("include_usage"); iu && iu->is_bool()) include_usage = iu->b;
    return true;
}

// Splits s into pieces of at most `max` bytes without cutting a UTF-8 sequence.
std::vector<std::string> utf8_pieces(const std::string& s, size_t max) {
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
        size_t cut = std::min(s.size(), i + max);
        while (cut < s.size() && cut > i + 1 && (static_cast<unsigned char>(s[cut]) & 0xC0u) == 0x80u) --cut;
        out.push_back(s.substr(i, cut - i));
        i = cut;
    }
    return out;
}

// A greedy generator has no sampling knobs, so a request that asks for one cannot be honored. Rejecting it (HTTP 400)
// is the honest answer: silently returning argmax text for a request that asked for temperature 0.7 makes the client
// believe it sampled. Only explicitly present fields are checked, so a request that omits them (whatever the API's own
// default is) and one that sends the greedy defaults are both accepted.
bool greedy_knobs_supported(const json::Value& root, std::string& err) {
    struct Knob { const char* name; double greedy; const char* shown; };
    const Knob numeric[] = {{"temperature", 0.0, "0"},  {"top_p", 1.0, "1"},          {"top_k", 0.0, "0"},
                            {"min_p", 0.0, "0"},        {"repeat_penalty", 1.0, "1"},   {"presence_penalty", 0.0, "0"},
                            {"frequency_penalty", 0.0, "0"}, {"top_logprobs", 0.0, "0"}};
    for (const auto& k : numeric) {
        const auto* v = root.get(k.name);
        if (!v || v->is_null()) continue;
        if (!v->is_number()) { err = std::string("'") + k.name + "' must be a number"; return false; }
        if (v->num != k.greedy) {
            err = std::string("this engine decodes greedily: '") + k.name + "' is not implemented (greedy default " +
                  k.shown + "); remove it or send the greedy default";
            return false;
        }
    }
    if (const auto* v = root.get("logprobs"); v && !v->is_null() && !(v->is_bool() && !v->b)) {
        err = "this engine decodes greedily: 'logprobs' is not implemented";
        return false;
    }
    return true;
}

// v1.2 lever: bounded SSE output queue (VEGAJET_V12_ASYNC_OUT, default on; "0" restores the inline writer).
bool async_out_enabled() {
    const char* v = std::getenv("VEGAJET_V12_ASYNC_OUT");
    return !(v && std::strcmp(v, "0") == 0);
}

void set_nonblocking(int fd, bool on) {
    const int fl = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, on ? (fl | O_NONBLOCK) : (fl & ~O_NONBLOCK));
}

} // namespace

// =================================================================================================
// Server
// =================================================================================================
struct OpenAiServer::Impl {
    // Over the connection limit a socket is still accepted far enough to answer 503 with a body (S-B2): Reserve
    // sockets only reach /health and /metrics, Reject sockets get the 503 for their first request.
    enum class Lane { Normal, Reserve, Reject };
    static constexpr size_t kMaxRejectSockets = 8;  // sockets kept only long enough to answer 503
    struct Conn {
        int fd = -1;
        std::string in, out;
        bool close_after = false;
        bool busy = false;           // owned by the worker
        bool sent_continue = false;
        bool peer_closed = false;
        Lane lane = Lane::Normal;
        int64_t last_ms = 0;
    };
    struct Job {
        int fd = -1;
        GenerateRequest req;
        bool keep_alive = true;
        bool include_usage = false;
        bool anthropic = false;  // POST /v1/messages
    };
    struct Done {
        int fd = -1;
        bool keep_alive = false;
    };

    ServerOptions opt;
    GenerateFn gen;
    int listen_fd = -1, ep = -1, wake_fd = -1;
    uint16_t bound_port = 0;
    int64_t created = 0;
    std::atomic<bool> stopping{false};
    bool started = false;
    std::thread io;
    std::vector<std::thread> workers;

    std::mutex mu;
    std::condition_variable cv;
    std::deque<Job> jobs;
    std::deque<Done> done;
    size_t in_flight = 0;

    std::mutex wait_mu;
    std::condition_variable wait_cv;
    bool io_exited = false;

    std::unordered_map<int, Conn> conns;  // I/O thread only (and stop() after joins)
    std::atomic<uint64_t> next_id{1};
    std::atomic<uint64_t> n_requests{0}, n_completions{0}, n_streamed{0}, n_cancelled{0}, n_rejected{0}, n_errors{0};

    void wake() {
        const uint64_t one = 1;
        if (wake_fd >= 0) { ssize_t r = ::write(wake_fd, &one, sizeof(one)); (void)r; }
    }

    // ---------------------------------------------------------------- I/O thread
    void watch(int fd, bool want_out, int op) {
        epoll_event ev{};
        ev.events = EPOLLIN | EPOLLRDHUP | (want_out ? EPOLLOUT : 0u);
        ev.data.fd = fd;
        epoll_ctl(ep, op, fd, &ev);
    }

    void close_conn(int fd) {
        epoll_ctl(ep, EPOLL_CTL_DEL, fd, nullptr);
        ::close(fd);
        conns.erase(fd);
    }

    // A half-closed peer (SHUT_WR) is a legal HTTP client: it has finished sending and still wants the response.
    // Closing while unread request bytes are queued would make the kernel answer with a RST and drop the response
    // already in the send buffer, so shut the write side first and drain what is left (S-M1).
    static void close_graceful(int fd) {
        ::shutdown(fd, SHUT_WR);
        char buf[4096];
        for (int i = 0; i < 32; ++i) {
            const ssize_t n = ::recv(fd, buf, sizeof(buf), MSG_DONTWAIT);
            if (n <= 0) break;
        }
        ::close(fd);
    }

    void close_conn_graceful(int fd) {
        epoll_ctl(ep, EPOLL_CTL_DEL, fd, nullptr);
        close_graceful(fd);
        conns.erase(fd);
    }

    // Returns false if the connection was closed.
    bool flush_out(int fd) {
        auto it = conns.find(fd);
        if (it == conns.end()) return false;
        Conn& c = it->second;
        while (!c.out.empty()) {
            const ssize_t n = ::send(fd, c.out.data(), c.out.size(), MSG_NOSIGNAL | MSG_DONTWAIT);
            if (n > 0) { c.out.erase(0, static_cast<size_t>(n)); continue; }
            if (n < 0 && errno == EINTR) continue;
            if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) { watch(fd, true, EPOLL_CTL_MOD); return true; }
            close_conn(fd);
            return false;
        }
        if (c.close_after || c.peer_closed) { close_conn_graceful(fd); return false; }
        watch(fd, false, EPOLL_CTL_MOD);
        return true;
    }

    void respond(int fd, std::string bytes, bool keep_alive) {
        auto it = conns.find(fd);
        if (it == conns.end()) return;
        it->second.out += bytes;
        if (!keep_alive) it->second.close_after = true;
        flush_out(fd);
    }

    void process_input(int fd) {
        while (true) {
            auto it = conns.find(fd);
            if (it == conns.end()) return;
            Conn& c = it->second;
            if (c.busy || !c.out.empty() || c.close_after || c.in.empty()) return;
            HttpRequest req;
            const ParseOut p = parse_http(c.in, opt.max_header_bytes, opt.max_body_bytes, req);
            if (p.status == Parse::Incomplete) {
                if (p.expect_continue && !c.sent_continue) {
                    c.sent_continue = true;
                    c.out += "HTTP/1.1 100 Continue\r\n\r\n";
                    flush_out(fd);
                }
                return;
            }
            if (p.status == Parse::Error) {
                c.in.clear();
                respond(fd, error_response(p.err_status, p.err_msg, false), false);
                return;
            }
            c.in.erase(0, p.consumed);
            c.sent_continue = false;
            ++n_requests;
            // Nothing escapes into the I/O thread: an exception here would std::terminate the process (S-B1).
            try {
                route(fd, req, c.lane);
            } catch (const std::bad_alloc&) {
                respond_if_idle(fd, error_response(503, "server out of memory while building the response", false), false);
            } catch (const std::exception& e) {
                respond_if_idle(fd, error_response(500, e.what(), false), false);
            } catch (...) {
                respond_if_idle(fd, error_response(500, "internal server error", false), false);
            }
        }
    }

    // A route() that already handed the socket to a worker must not also write to it.
    void respond_if_idle(int fd, std::string bytes, bool keep_alive) {
        auto it = conns.find(fd);
        if (it != conns.end() && !it->second.busy) respond(fd, std::move(bytes), keep_alive);
    }

    void route(int fd, const HttpRequest& req, Lane lane) {
        std::string path = req.target.substr(0, req.target.find('?'));
        const bool ka = req.keep_alive;
        if (lane == Lane::Reject) {
            // The pool is full: say so instead of dropping the socket (S-B2).
            ++n_rejected;
            respond(fd, error_response(503, "server busy: connection limit reached", false, "Retry-After: 1\r\n"), false);
            return;
        }
        if (lane == Lane::Reserve && path != "/health" && path != "/metrics") {
            ++n_rejected;
            respond(fd, error_response(503, "server busy: connection limit reached", false, "Retry-After: 1\r\n"), false);
            return;
        }
        if (req.method == "OPTIONS") {
            respond(fd, http_response(204, "", "", ka,
                                      "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                                      "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"),
                    ka);
            return;
        }
        if (path == "/metrics" && opt.metrics_text) {
            if (req.method != "GET" && req.method != "HEAD") {
                respond(fd, error_response(405, "method not allowed", ka, "Allow: GET\r\n"), ka);
                return;
            }
            const std::string body = opt.metrics_text();
            std::string resp = http_response(200, "text/plain; version=0.0.4; charset=utf-8", body, ka);
            if (req.method == "HEAD") resp.resize(resp.size() - body.size());
            respond(fd, std::move(resp), ka);
            return;
        }
        if (path == "/health" || path == "/v1/models") {
            if (req.method != "GET" && req.method != "HEAD") {
                respond(fd, error_response(405, "method not allowed", ka, "Allow: GET\r\n"), ka);
                return;
            }
            std::string body;
            if (path == "/health") {
                size_t q;
                { std::lock_guard<std::mutex> lk(mu); q = in_flight; }
                body = "{\"status\":\"ok\",\"model\":" + json::quote(opt.model_id) + ",\"in_flight\":" + std::to_string(q) +
                       ",\"max_queue\":" + std::to_string(opt.max_queue) + "}";
            } else {
                body = "{\"object\":\"list\",\"data\":[{\"id\":" + json::quote(opt.model_id) +
                       ",\"object\":\"model\",\"created\":" + std::to_string(created) + ",\"owned_by\":\"vegajet\"";
                if (opt.context_length) body += ",\"context_length\":" + std::to_string(opt.context_length);
                body += "}]}";
            }
            std::string resp = http_response(200, "application/json", body, ka);
            if (req.method == "HEAD") resp.resize(resp.size() - body.size());
            respond(fd, std::move(resp), ka);
            return;
        }
        if (path == "/v1/chat/completions" || path == "/v1/messages") {
            const bool anth = path == "/v1/messages";
            auto fail = [&](int status, const std::string& msg, const std::string& extra = {},
                            std::string_view code = {}) {
                respond(fd,
                        anth ? http_response(status, "application/json",
                                             anthropic::error_json(anthropic::error_type(status), msg), ka, extra)
                             : error_response(status, msg, ka, extra, code),
                        ka);
            };
            if (req.method != "POST") {
                fail(405, "method not allowed", "Allow: POST\r\n");
                return;
            }
            json::Value root;
            std::string err;
            if (req.body.size() > opt.max_chat_body_bytes) {
                fail(413, "request body is larger than this server accepts (" +
                              std::to_string(opt.max_chat_body_bytes) + " bytes)");
                return;
            }
            if (const int st = parse_body_bounded(req.body, opt.max_json_nodes, root, err); st) {
                fail(st, err);
                return;
            }
            // Greedy-only generators (ServerOptions::greedy_only) answer 400 for sampling knobs they cannot honor.
            if (opt.greedy_only && !greedy_knobs_supported(root, err)) {
                fail(400, err);
                return;
            }
            Job job;
            const bool ok = anth ? anthropic::parse_request(
                                       root,
                                       anthropic::Limits{opt.model_id, opt.default_max_tokens, opt.max_tokens_cap,
                                                         opt.max_model_name_len},
                                       job.req, err)
                                 : parse_chat(root, opt, job.req, job.include_usage, err);
            if (!ok) {
                fail(400, err);
                return;
            }
            // S-M3: an unknown model answers 404 like OpenAI, instead of a 200 with a foreign model name.
            if (job.req.model != opt.model_id) {
                fail(404, "model '" + job.req.model + "' does not exist; this server serves '" + opt.model_id + "'", {},
                     "model_not_found");
                return;
            }
            // S-M4: bound the request to the context before any encoding work. A prompt that cannot fit is a clear
            // 400; max_tokens is then capped to what is left, so one request cannot hold a queue slot indefinitely.
            if (opt.context_length) {
                const uint32_t est = estimate_prompt_tokens(job.req);
                if (est >= opt.context_length) {
                    fail(400, "prompt is about " + std::to_string(est) + " tokens, longer than the context length of " +
                                  std::to_string(opt.context_length) + " (code: context_length_exceeded)",
                         {}, "context_length_exceeded");
                    return;
                }
                const int32_t room = static_cast<int32_t>(opt.context_length - est);
                if (job.req.max_tokens > room) job.req.max_tokens = room;
            }
            {
                std::lock_guard<std::mutex> lk(mu);
                if (in_flight >= opt.max_queue || stopping) {
                    ++n_rejected;
                    fail(503, "server busy: generation queue is full", "Retry-After: 1\r\n");
                    return;
                }
                ++in_flight;
                job.fd = fd;
                job.anthropic = anth;
                job.keep_alive = ka;
                jobs.push_back(std::move(job));
            }
            auto it = conns.find(fd);
            it->second.busy = true;
            epoll_ctl(ep, EPOLL_CTL_DEL, fd, nullptr);  // the worker owns the socket until it posts Done
            cv.notify_one();
            return;
        }
        respond(fd, error_response(404, "not found", ka), ka);
    }

    void on_readable(int fd) {
        auto it = conns.find(fd);
        if (it == conns.end()) return;
        Conn& c = it->second;
        char buf[16384];
        while (true) {
            const ssize_t n = ::recv(fd, buf, sizeof(buf), MSG_DONTWAIT);
            if (n > 0) {
                c.in.append(buf, static_cast<size_t>(n));
                c.last_ms = now_ms();
                if (c.in.size() > opt.max_header_bytes + opt.max_body_bytes + 4096) { close_conn(fd); return; }
                continue;
            }
            if (n == 0) { c.peer_closed = true; break; }
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) break;
            close_conn(fd);
            return;
        }
        process_input(fd);
        auto again = conns.find(fd);
        if (again != conns.end() && again->second.peer_closed && !again->second.busy && again->second.out.empty())
            close_conn_graceful(fd);
    }

    void drain_done() {
        std::deque<Done> d;
        { std::lock_guard<std::mutex> lk(mu); d.swap(done); }
        for (const auto& x : d) {
            auto it = conns.find(x.fd);
            if (it == conns.end()) { ::close(x.fd); continue; }
            Conn& c = it->second;
            c.busy = false;
            if (!x.keep_alive || stopping || c.peer_closed) {
                close_graceful(x.fd);  // already removed from epoll; delivers the response to a half-closed client
                conns.erase(it);
                continue;
            }
            set_nonblocking(x.fd, true);
            c.last_ms = now_ms();
            watch(x.fd, false, EPOLL_CTL_ADD);
            process_input(x.fd);  // pipelined request bytes that arrived before the handoff
        }
    }

    void accept_all() {
        while (true) {
            sockaddr_storage ss{};
            socklen_t len = sizeof(ss);
            const int fd = ::accept4(listen_fd, reinterpret_cast<sockaddr*>(&ss), &len, SOCK_NONBLOCK | SOCK_CLOEXEC);
            if (fd < 0) {
                if (errno == EINTR) continue;
                return;  // EAGAIN or a transient error
            }
            Lane lane = Lane::Normal;
            if (conns.size() >= opt.max_connections) {
                if (conns.size() >= opt.max_connections + opt.reserve_connections) {
                    size_t reject = 0;
                    for (const auto& kv : conns)
                        if (kv.second.lane == Lane::Reject) ++reject;
                    if (reject >= kMaxRejectSockets) { ::close(fd); continue; }  // hard ceiling: no fd left to keep
                    lane = Lane::Reject;
                } else {
                    lane = Lane::Reserve;
                }
            }
            const int one = 1;
            setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
            Conn c;
            c.fd = fd;
            c.lane = lane;
            c.last_ms = now_ms();
            conns.emplace(fd, std::move(c));
            watch(fd, false, EPOLL_CTL_ADD);
        }
    }

    void io_loop() {
        epoll_event evs[64];
        int64_t last_sweep = now_ms();
        while (!stopping) {
            const int n = epoll_wait(ep, evs, 64, 250);
            if (n < 0 && errno != EINTR) break;
            for (int i = 0; i < n && !stopping; ++i) {
                const int fd = evs[i].data.fd;
                const uint32_t e = evs[i].events;
                if (fd == listen_fd) { accept_all(); continue; }
                if (fd == wake_fd) {
                    uint64_t v;
                    ssize_t r = ::read(wake_fd, &v, sizeof(v));
                    (void)r;
                    drain_done();
                    continue;
                }
                auto it = conns.find(fd);
                if (it == conns.end() || it->second.busy) continue;
                if (e & (EPOLLERR | EPOLLHUP)) { close_conn(fd); continue; }
                if (e & EPOLLIN) on_readable(fd);
                if (conns.count(fd) && (e & EPOLLOUT)) {
                    if (flush_out(fd)) process_input(fd);
                }
                if (conns.count(fd) && (e & EPOLLRDHUP)) {
                    // Half-close only: the peer stopped sending, not listening (S-M1).
                    auto& c = conns[fd];
                    c.peer_closed = true;
                    if (!c.busy && c.out.empty()) close_conn_graceful(fd);
                }
            }
            const int64_t t = now_ms();
            if (t - last_sweep >= 500) {
                last_sweep = t;
                std::vector<int> idle;
                for (const auto& kv : conns)
                    if (!kv.second.busy && t - kv.second.last_ms > opt.idle_timeout_ms) idle.push_back(kv.first);
                for (int fd : idle) close_conn(fd);
            }
        }
        {
            std::lock_guard<std::mutex> lk(wait_mu);
            io_exited = true;
        }
        wait_cv.notify_all();
    }

    // ---------------------------------------------------------------- worker thread
    static bool send_blocking(int fd, std::string_view s) {
        size_t off = 0;
        while (off < s.size()) {
            const ssize_t n = ::send(fd, s.data() + off, s.size() - off, MSG_NOSIGNAL);
            if (n > 0) { off += static_cast<size_t>(n); continue; }
            if (n < 0 && errno == EINTR) continue;
            return false;  // EPIPE, ECONNRESET, or SO_SNDTIMEO expired
        }
        return true;
    }

    // v1.2 lever: bounded queue of already-formatted SSE bytes between the generation thread and the socket. The
    // generation thread only enqueues (FIFO, so the byte stream is the same as writing inline); one writer thread owns
    // every blocking send and merges whatever is queued into one write, so a slow client costs generation a full queue
    // and nothing more. A write failure marks the pump aborted: the next push returns false, which cancels generation
    // exactly like the inline path's send failure did. Disabled: push() writes inline (the pre-v1.2 behaviour).
    class OutputPump {
    public:
        static constexpr size_t kMaxChunks = 32;          // producer blocks beyond this many queued chunks
        static constexpr size_t kMaxBytes = 256u << 10;   // and beyond this many queued bytes
        explicit OutputPump(int fd, bool on) : fd_(fd), on_(on) {
            if (on_) th_ = std::thread([this] { run(); });
        }
        ~OutputPump() { finish(); }
        OutputPump(const OutputPump&) = delete;
        OutputPump& operator=(const OutputPump&) = delete;

        // False means the client is gone (or the queue was closed): stop generating.
        bool push(std::string s) {
            if (!on_) return send_blocking(fd_, s);
            std::unique_lock<std::mutex> lk(m_);
            cv_.wait(lk, [&] {
                // `q_.empty()` lets a single chunk larger than the byte cap through instead of waiting forever.
                return aborted_ || closing_ ||
                       (q_.size() < kMaxChunks && (q_.empty() || bytes_ + s.size() <= kMaxBytes));
            });
            if (aborted_ || closing_) return false;
            bytes_ += s.size();
            q_.push_back(std::move(s));
            cv_.notify_all();
            return true;
        }

        // Flush what is queued and join the writer (idempotent; nothing to do when disabled).
        void finish() {
            if (!th_.joinable()) return;
            {
                std::lock_guard<std::mutex> lk(m_);
                closing_ = true;
            }
            cv_.notify_all();
            th_.join();
        }

    private:
        void run() {
            while (true) {
                std::string batch;
                {
                    std::unique_lock<std::mutex> lk(m_);
                    cv_.wait(lk, [&] { return aborted_ || !q_.empty() || closing_; });
                    if (aborted_) return;
                    if (q_.empty()) return;  // closing_ and nothing left to write
                    for (const auto& s : q_) batch += s;
                    q_.clear();
                    bytes_ = 0;
                    cv_.notify_all();  // room for the producer
                }
                if (!send_blocking(fd_, batch)) {
                    std::lock_guard<std::mutex> lk(m_);
                    aborted_ = true;
                    q_.clear();
                    bytes_ = 0;
                    cv_.notify_all();
                    return;
                }
            }
        }

        int fd_;
        bool on_;
        std::mutex m_;
        std::condition_variable cv_;
        std::deque<std::string> q_;
        size_t bytes_ = 0;
        bool closing_ = false;
        bool aborted_ = false;
        std::thread th_;
    };

    void run_job(Job& job) {
        bool released = false, posted = false;
        try {
            run_job_impl(job, released, posted);
        } catch (const std::bad_alloc&) {
            job_failed(job, released, posted, "server out of memory while building the response");
        } catch (const std::exception& e) {
            job_failed(job, released, posted, e.what());
        } catch (...) {
            job_failed(job, released, posted, "internal server error");
        }
    }

    // A generation job that threw must still release its queue slot and hand the socket back, or the server would
    // stall (S-B1). Answer 500 only if nothing was written yet.
    void job_failed(const Job& job, bool released, bool posted, const char* msg) {
        if (!posted) {
            if (!released) send_blocking(job.fd, error_response(500, msg, false));
            std::lock_guard<std::mutex> lk(mu);
            if (!released) --in_flight;
            done.push_back({job.fd, false});
        }
        ++n_errors;
        wake();
    }

    void run_job_impl(Job& job, bool& released, bool& posted) {
        const int fd = job.fd;
        set_nonblocking(fd, false);
        timeval tv{opt.send_timeout_ms / 1000, (opt.send_timeout_ms % 1000) * 1000};
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

        const GenerateRequest& rq = job.req;
        const bool stream = rq.stream;
        // Streaming responses: SSE bytes go through the bounded queue, so blocking socket writes never run on the
        // generation thread (VEGAJET_V12_ASYNC_OUT=0 restores the inline writer).
        OutputPump pump(fd, stream && async_out_enabled());
        const std::string id = "chatcmpl-" + std::to_string(created) + "-" + std::to_string(next_id++);
        const std::string ts = std::to_string(static_cast<long long>(std::time(nullptr)));
        const std::string model = json::quote(rq.model);
        // The model name is sent in the first and the final chunk only: repeating it per token multiplied a long
        // request model into a huge response (S-M3).
        auto chunk = [&](const std::string& delta, const char* finish, bool with_model) {
            return "data: {\"id\":\"" + id + "\",\"object\":\"chat.completion.chunk\",\"created\":" + ts +
                   (with_model ? ",\"model\":" + model : std::string()) + ",\"choices\":[{\"index\":0,\"delta\":" +
                   delta + ",\"finish_reason\":" + (finish ? json::quote(finish) : std::string("null")) + "}]}\n\n";
        };

        const bool anth = job.anthropic;
        anthropic::SseEncoder enc("msg_" + id.substr(9), rq.model);
        bool gone = false;
        if (stream) {
            const std::string head =
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                "Connection: close\r\nAccess-Control-Allow-Origin: *\r\nX-Accel-Buffering: no\r\n\r\n";
            gone = !pump.push(head + (anth ? enc.start()
                                           : chunk("{\"role\":\"assistant\",\"content\":\"\"}", nullptr, true)));
        }

        StopMatcher sm(rq.stop);
        // Output parser (core/tool_call_parser): leading <think> -> reasoning_content, <tool_call> -> tool_calls.
        toolcall::ParserOptions po;
        po.parse_reasoning = opt.parse_reasoning && rq.parse_reasoning;
        po.parse_tool_calls = rq.tools.active();
        po.parallel = rq.tools.parallel;
        if (po.parse_tool_calls) po.known_tools = rq.tools.names();
        po.id_prefix = (anth ? "toolu_" : "call_") + id.substr(9) + "_";  // deterministic per completion id
        po.prefill = toolcall::generation_prefill(rq.tools);
        toolcall::StreamParser parser(po);
        toolcall::Message msg;  // non-streaming accumulation
        uint32_t pieces = 0;
        bool hit_length = false;
        const auto t0 = Clock::now();
        Clock::time_point t_first{};
        auto deliver = [&](std::vector<toolcall::Event>& ev) {
            if (gone || ev.empty()) { ev.clear(); return; }
            if (!stream) { toolcall::accumulate(ev, msg); ev.clear(); return; }
            if (anth) {
                const std::string bytes = enc.events(ev);
                ev.clear();
                if (!bytes.empty()) gone = !pump.push(bytes);
                return;
            }
            std::string out;
            for (const auto& e : ev) {
                if (e.kind == toolcall::Event::Kind::Content) {
                    out += chunk("{\"content\":" + json::quote(e.text) + "}", nullptr, false);
                } else if (e.kind == toolcall::Event::Kind::Reasoning) {
                    out += chunk("{\"reasoning_content\":" + json::quote(e.text) + "}", nullptr, false);
                } else {  // OpenAI streaming shape: header delta (id, type, name), then argument fragments
                    const std::string idx = std::to_string(e.index);
                    out += chunk("{\"tool_calls\":[{\"index\":" + idx + ",\"id\":" + json::quote(e.call.id) +
                                 ",\"type\":\"function\",\"function\":{\"name\":" + json::quote(e.call.name) +
                                 ",\"arguments\":\"\"}}]}", nullptr, false);
                    for (const auto& a : utf8_pieces(e.call.arguments, 256))
                        out += chunk("{\"tool_calls\":[{\"index\":" + idx + ",\"function\":{\"arguments\":" +
                                     json::quote(a) + "}}]}", nullptr, false);
                }
            }
            ev.clear();
            if (!out.empty()) gone = !pump.push(std::move(out));
        };
        auto emit = [&](const std::string& text) {
            if (text.empty() || gone) return;
            std::vector<toolcall::Event> ev;
            parser.push(text, ev);
            deliver(ev);
        };
        const TokenSink sink = [&](std::string_view piece) -> SinkResult {
            if (stopping) return SinkResult::Stop();
            if (gone) return SinkResult::ClientGone();
            if (pieces++ == 0) t_first = Clock::now();
            emit(sm.push(piece));
            if (gone) return SinkResult::ClientGone();  // a write failed: the client is gone
            if (sm.stopped()) return SinkResult::Stop();
            if (parser.done()) return SinkResult::Stop();  // parallel_tool_calls = false: the first call ends the turn
            if (static_cast<int32_t>(pieces) >= rq.max_tokens) { hit_length = true; return SinkResult::Stop(); }
            return SinkResult();
        };

        GenerateResult res;
        if (!gone) {
            try {
                res = gen(rq, sink);
            } catch (const std::exception& e) {
                res.error = e.what();
            } catch (...) {
                res.error = "generator threw a non-standard exception";
            }
        }
        if (!gone && !stopping) {
            emit(sm.flush());
            std::vector<toolcall::Event> ev;
            parser.finish(ev);
            deliver(ev);
        }
        // Generation is over: free the queue slot before the final write. Releasing it after the write (as before)
        // let a client that sends its next request the moment it holds this response be refused with 503 by its
        // own previous request (seen at max_queue=1 under load: test_server_sse, 1-5 in 480 runs).
        {
            std::lock_guard<std::mutex> lk(mu);
            --in_flight;
        }
        released = true;

        bool keep_alive = false;
        if (gone || stopping) {
            ++n_cancelled;
        } else if (!res.error.empty()) {
            // A 400 (e.g. context_length_exceeded, S-M4) is a client error, not a server failure.
            const int status = res.http_status ? res.http_status : 500;
            if (status < 500) ++n_rejected; else ++n_errors;
            if (stream) {
                pump.push(anth ? enc.error(anthropic::error_type(status), res.error)
                               : "data: " + error_json(res.error, status < 500 ? "invalid_request_error" : "server_error") +
                                     "\n\ndata: [DONE]\n\n");
            } else {
                keep_alive = job.keep_alive;
                pump.push(anth ? http_response(status, "application/json",
                                               anthropic::error_json(anthropic::error_type(status), res.error), keep_alive)
                               : (status == 503 ? error_response(503, res.error, keep_alive, "Retry-After: 1\r\n")
                                                : error_response(status, res.error, keep_alive,
                                                                 {}, status == 400 ? "context_length_exceeded" : "")));
            }
        } else {
            FinishReason fr = sm.stopped() ? FinishReason::Stop : hit_length ? FinishReason::Length : res.finish;
            if (parser.tool_calls() > 0 && fr != FinishReason::Length) fr = FinishReason::ToolCalls;
            const uint32_t completion = res.completion_tokens ? res.completion_tokens : pieces;
            const double wall_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            const double decode_ms = res.decode_ms > 0 ? res.decode_ms
                                     : pieces ? std::chrono::duration<double, std::milli>(Clock::now() - t_first).count()
                                              : 0.0;
            const double prefill_ms = res.prefill_ms > 0 ? res.prefill_ms : std::max(0.0, wall_ms - decode_ms);
            const std::string usage = "{\"prompt_tokens\":" + std::to_string(res.prompt_tokens) +
                                      ",\"completion_tokens\":" + std::to_string(completion) +
                                      ",\"total_tokens\":" + std::to_string(res.prompt_tokens + completion) + "}";
            const std::string timings =
                "{\"prefill_ms\":" + fmt_num(prefill_ms) + ",\"decode_ms\":" + fmt_num(decode_ms) +
                ",\"prompt_tok_per_s\":" + fmt_num(prefill_ms > 0 ? res.prompt_tokens * 1000.0 / prefill_ms : 0.0) +
                ",\"decode_tok_per_s\":" + fmt_num(decode_ms > 0 ? completion * 1000.0 / decode_ms : 0.0) + "}";
            ++n_completions;
            const anthropic::Usage au{res.prompt_tokens, completion};
            const char* areason = anthropic::stop_reason(fr, sm.stopped());
            const std::string aseq = std::string_view(areason) == "stop_sequence" ? sm.matched() : std::string();
            if (anth && stream) {
                ++n_streamed;
                pump.push(enc.finish(areason, aseq, au));
            } else if (anth) {
                keep_alive = job.keep_alive;
                const std::string body = anthropic::response_json("msg_" + id.substr(9), rq.model, msg, areason, aseq, au);
                if (!pump.push(http_response(200, "application/json", body, keep_alive))) keep_alive = false;
            } else if (stream) {
                ++n_streamed;
                std::string tail = chunk("{}", finish_reason_name(fr), true);
                if (job.include_usage)
                    tail += "data: {\"id\":\"" + id + "\",\"object\":\"chat.completion.chunk\",\"created\":" + ts +
                            ",\"model\":" + model + ",\"choices\":[],\"usage\":" + usage + ",\"timings\":" + timings +
                            "}\n\n";
                tail += "data: [DONE]\n\n";
                pump.push(std::move(tail));
            } else {
                keep_alive = job.keep_alive;
                std::string message = "{\"role\":\"assistant\",\"content\":" +
                                      (msg.content.empty() && !msg.tool_calls.empty() ? std::string("null") : json::quote(msg.content));
                if (!msg.reasoning.empty()) message += ",\"reasoning_content\":" + json::quote(msg.reasoning);
                if (!msg.tool_calls.empty()) {
                    message += ",\"tool_calls\":[";
                    for (size_t i = 0; i < msg.tool_calls.size(); ++i) {
                        const auto& c = msg.tool_calls[i];
                        message += std::string(i ? "," : "") + "{\"id\":" + json::quote(c.id) +
                                   ",\"type\":\"function\",\"function\":{\"name\":" + json::quote(c.name) +
                                   ",\"arguments\":" + json::quote(c.arguments) + "}}";
                    }
                    message += "]";
                }
                message += "}";
                const std::string body =
                    "{\"id\":\"" + id + "\",\"object\":\"chat.completion\",\"created\":" + ts + ",\"model\":" + model +
                    ",\"choices\":[{\"index\":0,\"message\":" + message +
                    ",\"finish_reason\":" + json::quote(finish_reason_name(fr)) + "}],\"usage\":" + usage +
                    ",\"timings\":" + timings + "}";
                if (!pump.push(http_response(200, "application/json", body, keep_alive))) keep_alive = false;
            }
        }
        pump.finish();  // flush queued SSE bytes before the socket goes back to the I/O thread
        {
            std::lock_guard<std::mutex> lk(mu);
            done.push_back({fd, keep_alive});
        }
        posted = true;
        wake();
    }

    void worker_loop() {
        while (true) {
            Job job;
            {
                std::unique_lock<std::mutex> lk(mu);
                // Timed wait: request_stop() may set `stopping` from a signal handler without notifying.
                while (jobs.empty() && !stopping) cv.wait_for(lk, std::chrono::milliseconds(200));
                if (jobs.empty()) return;
                job = std::move(jobs.front());
                jobs.pop_front();
                if (stopping) {  // shutting down: hand the socket back unanswered
                    done.push_back({job.fd, false});
                    --in_flight;
                    ++n_cancelled;
                    continue;
                }
            }
            run_job(job);
        }
    }
};

OpenAiServer::OpenAiServer(ServerOptions opt, GenerateFn gen) : impl_(std::make_unique<Impl>()) {
    impl_->opt = std::move(opt);
    impl_->gen = std::move(gen);
}

OpenAiServer::~OpenAiServer() { stop(); }

bool OpenAiServer::start(std::string* err) {
    Impl& s = *impl_;
    auto fail = [&](const std::string& m) {
        if (err) *err = m + (errno ? std::string(": ") + std::strerror(errno) : std::string());
        if (s.listen_fd >= 0) { ::close(s.listen_fd); s.listen_fd = -1; }
        if (s.ep >= 0) { ::close(s.ep); s.ep = -1; }
        if (s.wake_fd >= 0) { ::close(s.wake_fd); s.wake_fd = -1; }
        return false;
    };
    if (s.started) return fail("already started");
    if (!s.gen) return fail("no GenerateFn");
    errno = 0;
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE | AI_NUMERICSERV;
    addrinfo* res = nullptr;
    const std::string port = std::to_string(s.opt.port);
    if (getaddrinfo(s.opt.host.empty() ? nullptr : s.opt.host.c_str(), port.c_str(), &hints, &res) != 0 || !res)
        return fail("cannot resolve listen address " + s.opt.host);
    for (addrinfo* a = res; a; a = a->ai_next) {
        const int fd = ::socket(a->ai_family, a->ai_socktype | SOCK_NONBLOCK | SOCK_CLOEXEC, a->ai_protocol);
        if (fd < 0) continue;
        const int one = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        if (::bind(fd, a->ai_addr, a->ai_addrlen) == 0 && ::listen(fd, 128) == 0) { s.listen_fd = fd; break; }
        ::close(fd);
    }
    freeaddrinfo(res);
    if (s.listen_fd < 0) return fail("cannot bind " + s.opt.host + ":" + port);
    sockaddr_storage ss{};
    socklen_t len = sizeof(ss);
    getsockname(s.listen_fd, reinterpret_cast<sockaddr*>(&ss), &len);
    s.bound_port = ntohs(ss.ss_family == AF_INET6 ? reinterpret_cast<sockaddr_in6*>(&ss)->sin6_port
                                                  : reinterpret_cast<sockaddr_in*>(&ss)->sin_port);
    s.ep = epoll_create1(EPOLL_CLOEXEC);
    s.wake_fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (s.ep < 0 || s.wake_fd < 0) return fail("epoll/eventfd setup failed");
    s.watch(s.listen_fd, false, EPOLL_CTL_ADD);
    epoll_event ev{};
    ev.events = EPOLLIN;
    ev.data.fd = s.wake_fd;
    epoll_ctl(s.ep, EPOLL_CTL_ADD, s.wake_fd, &ev);
    s.created = static_cast<int64_t>(std::time(nullptr));
    s.started = true;
    s.io = std::thread([&s] { s.io_loop(); });
    for (size_t i = 0; i < std::max<size_t>(1, s.opt.workers); ++i) s.workers.emplace_back([&s] { s.worker_loop(); });
    return true;
}

uint16_t OpenAiServer::port() const { return impl_->bound_port; }

void OpenAiServer::request_stop() {
    impl_->stopping.store(true);
    impl_->wake();
}

void OpenAiServer::stop() {
    Impl& s = *impl_;
    if (!s.started) return;
    s.stopping.store(true);
    s.wake();
    s.cv.notify_all();
    if (s.io.joinable()) s.io.join();
    for (auto& w : s.workers)
        if (w.joinable()) w.join();
    s.workers.clear();
    for (const auto& kv : s.conns) ::close(kv.first);
    s.conns.clear();
    {
        std::lock_guard<std::mutex> lk(s.mu);
        s.done.clear();  // every handed-back fd is still in `conns` (busy) and was closed above
        s.jobs.clear();
    }
    if (s.listen_fd >= 0) ::close(s.listen_fd);
    if (s.ep >= 0) ::close(s.ep);
    if (s.wake_fd >= 0) ::close(s.wake_fd);
    s.listen_fd = s.ep = s.wake_fd = -1;
    s.started = false;
    {
        std::lock_guard<std::mutex> lk(s.wait_mu);
        s.io_exited = true;
    }
    s.wait_cv.notify_all();
}

void OpenAiServer::wait() {
    {
        std::unique_lock<std::mutex> lk(impl_->wait_mu);
        impl_->wait_cv.wait(lk, [&] { return impl_->io_exited; });
    }
    stop();
}

OpenAiServer::Stats OpenAiServer::stats() const {
    const Impl& s = *impl_;
    Stats st;
    st.requests = s.n_requests;
    st.completions = s.n_completions;
    st.streamed = s.n_streamed;
    st.cancelled = s.n_cancelled;
    st.rejected = s.n_rejected;
    st.errors = s.n_errors;
    return st;
}

} // namespace vegajet::serve
