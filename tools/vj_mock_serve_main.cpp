// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// HEAD's OpenAiServer bound to HEAD's CPU MockEngine (no model file, no GPU, no /dev/kfd). Test fixture for
// tools/api_stress_test.py and tools/swarm_router.py; same HTTP surface as `vegajet serve`.
//   vj_mock_serve [--host 127.0.0.1] [--port 0] [--max-queue 16] [--model-id vegajet-mock]
// Prints "listening on <host>:<port>" once bound (port 0 = ephemeral) and serves until SIGINT / SIGTERM.
#include "core/engine_interface.hpp"
#include "core/server.hpp"
#include "core/server_engine_adapter.hpp"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {
vegajet::serve::OpenAiServer* g_server = nullptr;
void on_signal(int) {
    if (g_server) g_server->request_stop();
}
} // namespace

int main(int argc, char** argv) {
    vegajet::serve::ServerOptions opt;
    opt.port = 0;
    opt.model_id = "vegajet-mock";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const char* v = i + 1 < argc ? argv[i + 1] : nullptr;
        if (a == "--host" && v) { opt.host = v; ++i; }
        else if (a == "--port" && v) { opt.port = static_cast<uint16_t>(std::atoi(v)); ++i; }
        else if (a == "--max-queue" && v) { opt.max_queue = static_cast<size_t>(std::atoi(v)); ++i; }
        else if (a == "--model-id" && v) { opt.model_id = v; ++i; }
        else { std::fprintf(stderr, "usage: %s [--host H] [--port P] [--max-queue N] [--model-id ID]\n", argv[0]); return 2; }
    }
    vegajet::MockEngine engine(opt.model_id);
    opt.context_length = engine.model_info().context_length;
    vegajet::serve::OpenAiServer server(opt, vegajet::serve::make_engine_generate_fn(engine));
    std::string err;
    if (!server.start(&err)) {
        std::fprintf(stderr, "vj_mock_serve: %s\n", err.c_str());
        return 1;
    }
    g_server = &server;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    std::printf("listening on %s:%u\n", opt.host.c_str(), static_cast<unsigned>(server.port()));
    std::fflush(stdout);
    server.wait();
    server.stop();
    const auto st = server.stats();
    std::fprintf(stderr, "vj_mock_serve: requests %llu completions %llu streamed %llu cancelled %llu rejected %llu errors %llu\n",
                 (unsigned long long)st.requests, (unsigned long long)st.completions, (unsigned long long)st.streamed,
                 (unsigned long long)st.cancelled, (unsigned long long)st.rejected, (unsigned long long)st.errors);
    g_server = nullptr;
    return 0;
}
