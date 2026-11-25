// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "cli/chat_repl.hpp"

#include "cli/http_mini.hpp"
#include "core/json_lite.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

namespace vegajet::cli {
namespace {

using Clock = std::chrono::steady_clock;

void usage(std::FILE* f) {
    std::fprintf(f,
                 "Usage: vegajet chat [--url http://127.0.0.1:8080] [--model NAME] [--system TEXT] [--max-tokens N]\n"
                 "                    [--temperature T] [--no-stream]\n"
                 "  --url          OpenAI-compatible server (vegajet serve, the swarm router, ...)\n"
                 "  --model        model name sent with each request (default: the server's)\n"
                 "  --system       system prompt\n"
                 "  --max-tokens   tokens per reply, 1..1000000 (default 512)\n"
                 "  --temperature  sampling temperature, 0..2 (default: the server's)\n"
                 "  --no-stream    wait for whole replies instead of streaming them\n"
                 "  Commands: /stats  /clear  /save FILE  /load FILE  /system TEXT  /help  /quit\n"
                 "  Exit codes: 0 ok, 1 server unreachable, 2 usage error.\n");
}

const char* kHelp =
    "/stats        turns, tokens, last time-to-first-token and tok/s\n"
    "/clear        forget the conversation (the system prompt stays)\n"
    "/save FILE    write the conversation as JSON\n"
    "/load FILE    replace the conversation with a saved one\n"
    "/system TEXT  set the system prompt (empty: none)\n"
    "/quit         leave (also /exit, Ctrl-D)\n";

std::string fmt(const char* f, double v) {
    char b[64];
    std::snprintf(b, sizeof b, f, v);
    return b;
}

}  // namespace

ChatSession::ChatSession(ChatConfig cfg) : cfg_(std::move(cfg)) {}

std::string ChatSession::request_json(const std::string& user) const {
    std::string msgs;
    auto add = [&](const std::string& role, const std::string& content) {
        if (!msgs.empty()) msgs += ',';
        msgs += "{\"role\":" + json::quote(role) + ",\"content\":" + json::quote(content) + "}";
    };
    if (!cfg_.system.empty()) add("system", cfg_.system);
    for (const auto& t : hist_) add(t.role, t.content);
    add("user", user);
    std::string body = "{";
    if (!cfg_.model.empty()) body += "\"model\":" + json::quote(cfg_.model) + ",";
    body += "\"messages\":[" + msgs + "],\"max_tokens\":" + std::to_string(cfg_.max_tokens);
    if (cfg_.temperature >= 0) body += ",\"temperature\":" + fmt("%.4g", cfg_.temperature);
    if (cfg_.stream) body += ",\"stream\":true,\"stream_options\":{\"include_usage\":true}";
    return body + "}";
}

bool ChatSession::send(const std::string& user, std::ostream& out) {
    http::Url u;
    std::string err;
    if (!http::parse_url(cfg_.url, u, &err)) {
        out << "error: " << err << "\n";
        ++st_.errors;
        return false;
    }
    const char* dim = cfg_.color ? "\x1b[2m" : "";
    const char* norm = cfg_.color ? "\x1b[0m" : "";
    const std::string body = request_json(user);
    std::string raw, sse, content, reasoning, error_msg;
    bool in_reasoning = false, any = false;
    uint64_t usage_prompt = 0, usage_completion = 0;
    uint32_t deltas = 0;
    const auto t0 = Clock::now();
    Clock::time_point first{}, last{};
    auto on_delta = [&](const std::string& text, bool is_reasoning) {
        if (text.empty()) return;
        last = Clock::now();
        if (!any) first = last;
        any = true;
        ++deltas;
        if (is_reasoning != in_reasoning) {
            out << (is_reasoning ? dim : norm);
            in_reasoning = is_reasoning;
        }
        out << text << std::flush;
        (is_reasoning ? reasoning : content) += text;
    };
    auto handle_event = [&](const std::string& data) {
        if (data == "[DONE]") return;
        json::Value v;
        if (!json::parse(data, v)) return;
        if (const auto* e = v.get("error"); e && e->is_object()) {
            if (const auto* m = e->get("message"); m && m->is_string()) error_msg = m->str;
            return;
        }
        if (const auto* us = v.get("usage"); us && us->is_object()) {
            if (const auto* p = us->get("prompt_tokens"); p && p->is_number()) usage_prompt = static_cast<uint64_t>(p->num);
            if (const auto* c = us->get("completion_tokens"); c && c->is_number()) usage_completion = static_cast<uint64_t>(c->num);
        }
        const auto* ch = v.get("choices");
        if (!ch || !ch->is_array() || ch->arr.empty()) return;
        const json::Value& c0 = ch->arr[0];
        const json::Value* d = c0.get("delta");
        if (!d) d = c0.get("message");  // non-streaming
        if (!d) return;
        if (const auto* r = d->get("reasoning_content"); r && r->is_string()) on_delta(r->str, true);
        if (const auto* c = d->get("content"); c && c->is_string()) on_delta(c->str, false);
    };
    const int status = http::request(
        u, "POST", "/v1/chat/completions", body,
        [&](std::string_view piece) {
            if (!cfg_.stream) {
                raw.append(piece.data(), piece.size());
                return true;
            }
            sse.append(piece.data(), piece.size());
            for (size_t e; (e = sse.find("\n\n")) != std::string::npos;) {
                const std::string ev = sse.substr(0, e);
                sse.erase(0, e + 2);
                std::istringstream ls(ev);
                for (std::string line; std::getline(ls, line);) {
                    if (!line.empty() && line.back() == '\r') line.pop_back();
                    if (line.rfind("data: ", 0) == 0) handle_event(line.substr(6));
                    else if (line.rfind("{", 0) == 0) raw += line;  // an error body on a streaming request
                }
            }
            return true;
        },
        &err, 600000);
    if (cfg_.stream && status != 200) raw += sse;  // an error answer is a plain JSON body, not an event stream
    if (!cfg_.stream || status != 200) {
        if (!raw.empty()) {
            if (status == 200) handle_event(raw);
            else {
                json::Value v;
                if (json::parse(raw, v))
                    if (const auto* e = v.get("error"); e && e->is_object())
                        if (const auto* m = e->get("message"); m && m->is_string()) error_msg = m->str;
            }
        }
    }
    if (in_reasoning) out << norm;
    if (status != 200 || !error_msg.empty()) {
        if (any) out << "\n";
        out << "error: " << (status < 0 ? err : error_msg.empty() ? "HTTP " + std::to_string(status) : error_msg) << "\n";
        ++st_.errors;
        return false;
    }
    out << "\n";
    hist_.push_back({"user", user});
    hist_.push_back({"assistant", content});
    ++st_.turns;
    st_.prompt_tokens += usage_prompt;
    const uint64_t completion = usage_completion ? usage_completion : deltas;
    st_.completion_tokens += completion;
    st_.last_ttft_ms = any ? std::chrono::duration<double, std::milli>(first - t0).count() : 0.0;
    const double gen_s = any ? std::chrono::duration<double>(last - first).count() : 0.0;
    st_.total_gen_s += std::chrono::duration<double>(Clock::now() - t0).count();
    st_.last_tok_s = gen_s > 0 && completion > 1 ? static_cast<double>(completion - 1) / gen_s : 0.0;
    return true;
}

bool ChatSession::save(const std::string& path, std::string* err) const {
    std::string o = "{\"version\":1,\"model\":" + json::quote(cfg_.model) + ",\"system\":" + json::quote(cfg_.system) +
                    ",\"messages\":[";
    for (size_t i = 0; i < hist_.size(); ++i)
        o += std::string(i ? "," : "") + "\n  {\"role\":" + json::quote(hist_[i].role) + ",\"content\":" +
             json::quote(hist_[i].content) + "}";
    o += "\n]}\n";
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0) {
        if (err) *err = "cannot write " + path;
        return false;
    }
    const ssize_t n = ::write(fd, o.data(), o.size());
    ::close(fd);
    if (n != static_cast<ssize_t>(o.size())) {
        if (err) *err = "short write to " + path;
        return false;
    }
    return true;
}

bool ChatSession::load(const std::string& path, std::string* err) {
    std::ifstream f(path);
    if (!f) {
        if (err) *err = "cannot read " + path;
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    json::Value v;
    if (!json::parse(ss.str(), v) || !v.is_object()) {
        if (err) *err = path + " is not a saved conversation (invalid JSON)";
        return false;
    }
    const auto* msgs = v.get("messages");
    if (!msgs || !msgs->is_array()) {
        if (err) *err = path + " has no \"messages\" array";
        return false;
    }
    std::vector<ChatTurnText> h;
    for (const auto& m : msgs->arr) {
        const auto* r = m.get("role");
        const auto* c = m.get("content");
        if (!r || !r->is_string() || (r->str != "user" && r->str != "assistant") || !c || !c->is_string()) {
            if (err) *err = path + ": every message needs role user / assistant and string content";
            return false;
        }
        h.push_back({r->str, c->str});
    }
    hist_ = std::move(h);
    if (const auto* s = v.get("system"); s && s->is_string()) cfg_.system = s->str;
    return true;
}

bool ChatSession::handle(const std::string& line_in, std::ostream& out) {
    std::string line = line_in;
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
    if (line.empty()) return true;
    if (line[0] != '/') {
        send(line, out);
        return true;
    }
    const size_t sp = line.find(' ');
    const std::string cmd = line.substr(0, sp);
    std::string arg = sp == std::string::npos ? "" : line.substr(sp + 1);
    while (!arg.empty() && arg.front() == ' ') arg.erase(0, 1);
    std::string err;
    if (cmd == "/quit" || cmd == "/exit") return false;
    if (cmd == "/help") {
        out << kHelp;
    } else if (cmd == "/stats") {
        out << "turns " << st_.turns << " | prompt tokens " << st_.prompt_tokens << " | completion tokens "
            << st_.completion_tokens << " | last TTFT " << fmt("%.1f", st_.last_ttft_ms) << " ms | last "
            << fmt("%.1f", st_.last_tok_s) << " tok/s | errors " << st_.errors << " | history " << hist_.size()
            << " messages\n";
    } else if (cmd == "/clear") {
        hist_.clear();
        out << "(conversation cleared)\n";
    } else if (cmd == "/save") {
        if (arg.empty()) out << "usage: /save FILE\n";
        else if (save(arg, &err)) out << "(saved " << hist_.size() << " messages)\n";
        else out << "error: " << err << "\n";
    } else if (cmd == "/load") {
        if (arg.empty()) out << "usage: /load FILE\n";
        else if (load(arg, &err)) out << "(loaded " << hist_.size() << " messages)\n";
        else out << "error: " << err << "\n";
    } else if (cmd == "/system") {
        cfg_.system = arg;
        out << (arg.empty() ? "(system prompt cleared)\n" : "(system prompt set)\n");
    } else {
        out << "unknown command " << cmd << " (/help lists them)\n";
    }
    return true;
}

int run_chat(const std::vector<std::string>& args) {
    ChatConfig cfg;
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        auto value = [&](std::string& v) {
            if (i + 1 >= args.size()) return false;
            v = args[++i];
            return true;
        };
        std::string v;
        if (a == "--help" || a == "-h") {
            usage(stdout);
            return 0;
        } else if (a == "--url") {
            if (!value(cfg.url)) { usage(stderr); return 2; }
        } else if (a == "--model") {
            if (!value(cfg.model)) { usage(stderr); return 2; }
        } else if (a == "--system") {
            if (!value(cfg.system)) { usage(stderr); return 2; }
        } else if (a == "--max-tokens") {
            if (!value(v)) { usage(stderr); return 2; }
            const long n = std::atol(v.c_str());
            if (n < 1 || n > 1000000) { std::fprintf(stderr, "--max-tokens takes 1..1000000\n"); return 2; }
            cfg.max_tokens = static_cast<int32_t>(n);
        } else if (a == "--temperature") {
            if (!value(v)) { usage(stderr); return 2; }
            cfg.temperature = std::atof(v.c_str());
            if (!(cfg.temperature >= 0 && cfg.temperature <= 2)) { std::fprintf(stderr, "--temperature takes 0..2\n"); return 2; }
        } else if (a == "--no-stream") {
            cfg.stream = false;
        } else {
            std::fprintf(stderr, "Unknown option for chat: %s\n", a.c_str());
            usage(stderr);
            return 2;
        }
    }
    http::Url u;
    std::string err, body;
    if (!http::parse_url(cfg.url, u, &err)) {
        std::fprintf(stderr, "[vegajet] %s\n", err.c_str());
        return 2;
    }
    if (http::get(u, "/health", body, &err, 5000) < 0) {
        std::fprintf(stderr, "[vegajet] server unreachable: %s\n", err.c_str());
        return 1;
    }
    const bool tty_in = ::isatty(STDIN_FILENO), tty_out = ::isatty(STDOUT_FILENO);
    cfg.color = tty_out;
    ChatSession s(cfg);
    if (tty_in) std::printf("vegajet chat: %s (/help for commands, /quit to leave)\n", cfg.url.c_str());
    std::string line;
    while (true) {
        if (tty_in) {
            std::fputs(tty_out ? "\x1b[1m> \x1b[0m" : "> ", stdout);
            std::fflush(stdout);
        }
        if (!std::getline(std::cin, line)) break;
        if (!s.handle(line, std::cout)) break;
        std::cout.flush();
    }
    return 0;
}

}  // namespace vegajet::cli
