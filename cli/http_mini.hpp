// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Minimal blocking HTTP/1.1 client for the CLI front ends (vegajet top, vegajet chat): plain http:// only,
// Connection: close, Content-Length or chunked bodies, body delivered incrementally (SSE streams print as they
// arrive). Standard sockets, no dependencies.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>

namespace vegajet::cli::http {

struct Url {
    std::string host = "127.0.0.1";
    uint16_t port = 80;
    std::string base;  // path prefix without a trailing '/', e.g. "" or "/proxy"
};

bool parse_url(const std::string& url, Url& out, std::string* err = nullptr);

// Sends one request. on_body receives the (de-chunked) body in pieces; returning false aborts. Returns the HTTP
// status, or -1 with *err set when the server cannot be reached or the response is malformed.
int request(const Url& u, std::string_view method, std::string_view path, std::string_view body,
            const std::function<bool(std::string_view)>& on_body, std::string* err = nullptr, int timeout_ms = 30000);

// Convenience: whole body.
int get(const Url& u, std::string_view path, std::string& body, std::string* err = nullptr, int timeout_ms = 5000);

}  // namespace vegajet::cli::http
