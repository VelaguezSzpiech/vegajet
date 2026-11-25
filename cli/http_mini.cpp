// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "cli/http_mini.hpp"

#include <netdb.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>

namespace vegajet::cli::http {

bool parse_url(const std::string& url, Url& out, std::string* err) {
    auto fail = [&](const std::string& m) {
        if (err) *err = m;
        return false;
    };
    const std::string pfx = "http://";
    if (url.compare(0, pfx.size(), pfx) != 0) return fail("only http:// URLs are supported: " + url);
    std::string rest = url.substr(pfx.size());
    const size_t slash = rest.find('/');
    std::string hostport = rest.substr(0, slash);
    out.base = slash == std::string::npos ? "" : rest.substr(slash);
    while (!out.base.empty() && out.base.back() == '/') out.base.pop_back();
    if (hostport.empty()) return fail("URL without a host: " + url);
    out.port = 80;
    if (hostport.front() == '[') {  // [v6]:port
        const size_t rb = hostport.find(']');
        if (rb == std::string::npos) return fail("bad IPv6 host in " + url);
        out.host = hostport.substr(1, rb - 1);
        hostport = hostport.substr(rb + 1);
        if (!hostport.empty() && hostport[0] != ':') return fail("bad URL " + url);
        if (!hostport.empty()) hostport = hostport.substr(1);
        else hostport.clear();
        if (!hostport.empty()) {
            char* end = nullptr;
            const long p = std::strtol(hostport.c_str(), &end, 10);
            if (*end || p < 1 || p > 65535) return fail("bad port in " + url);
            out.port = static_cast<uint16_t>(p);
        }
        return true;
    }
    const size_t colon = hostport.rfind(':');
    out.host = hostport.substr(0, colon);
    if (colon != std::string::npos) {
        char* end = nullptr;
        const std::string ps = hostport.substr(colon + 1);
        const long p = std::strtol(ps.c_str(), &end, 10);
        if (ps.empty() || *end || p < 1 || p > 65535) return fail("bad port in " + url);
        out.port = static_cast<uint16_t>(p);
    }
    if (out.host.empty()) return fail("URL without a host: " + url);
    return true;
}

namespace {

int connect_to(const Url& u, int timeout_ms, std::string* err) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    const std::string port = std::to_string(u.port);
    if (const int rc = ::getaddrinfo(u.host.c_str(), port.c_str(), &hints, &res); rc != 0) {
        if (err) *err = "cannot resolve " + u.host + ": " + gai_strerror(rc);
        return -1;
    }
    int fd = -1;
    for (addrinfo* a = res; a; a = a->ai_next) {
        fd = ::socket(a->ai_family, a->ai_socktype | SOCK_CLOEXEC, a->ai_protocol);
        if (fd < 0) continue;
        timeval tv{timeout_ms / 1000, (timeout_ms % 1000) * 1000};
        ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
        ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
        if (::connect(fd, a->ai_addr, a->ai_addrlen) == 0) break;
        ::close(fd);
        fd = -1;
    }
    ::freeaddrinfo(res);
    if (fd < 0 && err) *err = "cannot connect to " + u.host + ":" + port + ": " + std::strerror(errno);
    return fd;
}

bool send_all(int fd, std::string_view s) {
    size_t off = 0;
    while (off < s.size()) {
        const ssize_t n = ::send(fd, s.data() + off, s.size() - off, MSG_NOSIGNAL);
        if (n > 0) off += static_cast<size_t>(n);
        else if (n < 0 && errno == EINTR) continue;
        else return false;
    }
    return true;
}

}  // namespace

int request(const Url& u, std::string_view method, std::string_view path, std::string_view body,
            const std::function<bool(std::string_view)>& on_body, std::string* err, int timeout_ms) {
    const int fd = connect_to(u, timeout_ms, err);
    if (fd < 0) return -1;
    std::string req;
    req.append(method.data(), method.size());
    req += ' ' + u.base;
    req.append(path.data(), path.size());
    req += " HTTP/1.1\r\nHost: " + u.host + ":" + std::to_string(u.port) + "\r\nConnection: close\r\n";
    if (!body.empty() || method == "POST")
        req += "Content-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) + "\r\n";
    req += "\r\n";
    req.append(body.data(), body.size());
    if (!send_all(fd, req)) {
        if (err) *err = "send failed";
        ::close(fd);
        return -1;
    }
    std::string buf;
    int status = -1;
    bool headers_done = false, chunked = false, aborted = false;
    size_t chunk_left = 0;  // chunked: bytes left in the current chunk
    bool chunk_trailer = false;
    char tmp[16384];
    auto deliver = [&](std::string_view s) {
        if (!aborted && !s.empty() && !on_body(s)) aborted = true;
    };
    // Processes whatever body bytes sit in buf.
    auto drain = [&] {
        if (!chunked) {
            deliver(buf);
            buf.clear();
            return;
        }
        while (!aborted) {
            if (chunk_trailer) {  // CRLF after a chunk
                if (buf.size() < 2) return;
                buf.erase(0, 2);
                chunk_trailer = false;
            }
            if (chunk_left == 0) {
                const size_t e = buf.find("\r\n");
                if (e == std::string::npos) return;
                chunk_left = std::strtoul(buf.c_str(), nullptr, 16);
                buf.erase(0, e + 2);
                if (chunk_left == 0) {
                    aborted = true;  // last chunk: done
                    return;
                }
            }
            const size_t n = std::min(chunk_left, buf.size());
            if (n == 0) return;
            deliver(std::string_view(buf).substr(0, n));
            buf.erase(0, n);
            chunk_left -= n;
            if (chunk_left == 0) chunk_trailer = true;
        }
    };
    while (!aborted) {
        const ssize_t n = ::recv(fd, tmp, sizeof tmp, 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        buf.append(tmp, static_cast<size_t>(n));
        if (!headers_done) {
            const size_t he = buf.find("\r\n\r\n");
            if (he == std::string::npos) {
                if (buf.size() > 65536) break;
                continue;
            }
            const std::string head = buf.substr(0, he);
            buf.erase(0, he + 4);
            headers_done = true;
            const size_t sp = head.find(' ');
            if (sp != std::string::npos) status = std::atoi(head.c_str() + sp + 1);
            std::string lower = head;
            for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            chunked = lower.find("transfer-encoding: chunked") != std::string::npos;
        }
        drain();
    }
    ::close(fd);
    if (!headers_done || status <= 0) {
        if (err) *err = "no HTTP response from " + u.host + ":" + std::to_string(u.port);
        return -1;
    }
    return status;
}

int get(const Url& u, std::string_view path, std::string& body, std::string* err, int timeout_ms) {
    body.clear();
    return request(
        u, "GET", path, {},
        [&](std::string_view s) {
            body.append(s.data(), s.size());
            return true;
        },
        err, timeout_ms);
}

}  // namespace vegajet::cli::http
