# VegaJet HTTP API layer — code-quality and reliability audit (release/v1.2)

Scope: `core/server.cpp` (epoll HTTP/1.1 + OpenAI/Anthropic surface), `core/server.hpp`, `core/http_*`-adjacent
helpers, `core/json_lite.hpp`, `core/anthropic_shim.{hpp,cpp}`, `core/server_engine_adapter.hpp`,
`core/metrics.{hpp,cpp}`, `cli/serve_cmd.cpp`. Target question: **does the server stay up, keep the accounting
correct, and answer bad requests with clear errors?**

Method: source read of the files above at `3bc0015` (branch `release/v1.2`, worktree untouched) plus live probes
against a CPU build. All measurements below are from real runs, not inference:

* `build-audit-http/vj_mock_serve` — the shipped fixture (canned 8-token reply, so it cannot hold a slot).
* `build-audit-http/audit_slow_serve` and `build-audit-http/audit_adapter_serve` — audit-only harnesses built in the
  untracked build dir (`audit_slow_serve_main.cpp`, `audit_adapter_serve_main.cpp`) so that queue saturation,
  cancellation and aborts are observable. The second one wires **exactly** `cli/serve_cmd.cpp`'s single-slot path:
  `make_engine_generate_fn(engine)` → `metrics::instrument(...)` → `OpenAiServer`, over a slow `IEngine`.
* Parser edge cases against a `-fsanitize=undefined,float-cast-overflow` build (`build-audit-http/ubsan2`).
* Probe scripts and raw JSON: `~/.hermes/cache/scratch/audit_http/` (`probe1..13`, `results*.json`).

Nothing tracked was modified; every server started for a probe was stopped (SIGTERM) and its exit status recorded.

## What is already sound (verified, so it is not re-reviewed below)

| Checked | Result |
|---|---|
| Header/body size limits (16 KiB / 8 MiB) | 431 and 413 with JSON bodies + `Connection: close` — correct |
| 404 / 405 (`Allow:`) / 400 empty body / 501 chunked / 505 | well-formed JSON, correct `Content-Length`, no body on `HEAD` |
| Pipelined requests in one segment (2 requests, 1 write) | 2 responses, correct framing |
| keep-alive, `Connection: close`, HTTP/1.0 semantics | correct (`Connection: close` for HTTP/1.0 without `keep-alive`) |
| `Expect: 100-continue` | interim `100 Continue`, then 200 |
| JSON depth limit (64) | deep nesting → 400, no stack growth |
| Malformed input (23 shapes: H2 preface, lone LF, NUL in header, `+5`/`-5`/21-digit Content-Length, CONNECT, garbage after a valid request) | server stayed up, 400 for real syntax errors |
| `in_flight` / `max_queue` / 503 under a real slow generator | `in_flight` tracked 2→1→0 exactly, 3rd request got 503 (`rejected 1`), no leak after abrupt client drops |
| Per-token SSE framing, `[DONE]`, `finish_reason`, `stream_options.include_usage` | correct |
| Anthropic `message_start … message_stop` event sequence | correct ordering, `event:` + `data:` per event |

---

## BLOCKER

### S-B1 — One request can abort the whole process: uncaught `std::bad_alloc` from the JSON tree

`core/server.cpp:569`, `core/server.cpp:197`, `core/json_lite.hpp:16-22`

```
core/server.cpp:569        if (!json::parse(req.body, root)) {
core/json_lite.hpp:16      struct Value {
core/json_lite.hpp:20          std::string str;
core/json_lite.hpp:21          std::vector<Value> arr;
core/json_lite.hpp:22          std::vector<std::pair<std::string, Value>> obj;
```

`Value` is ~96 bytes, is built per JSON node (`arr.push_back(std::move(e))`), and `json::parse` runs **on the epoll
thread** inside `route()`, which has no `try`/`catch` anywhere. Every allocation failure therefore escapes
`io_loop()` — and an exception escaping a `std::thread` body calls `std::terminate`. The worker has the same hole
outside its `try` (`server.cpp:828` pump thread construction, `831` `json::quote(rq.model)`, `832-837` every
`chunk()` concatenation is between the `try` at 912 and the response write).

Measured amplification (valid body, request is then rejected, so the cost is pure parser cost):

| body | process VmSize | VmHWM | outcome |
|---|---|---|---|
| 2 000 054 B `[0,0,…]` in `content` | 23 MB → **191 MB** | 4 MB → **108 MB** | 400 (request rejected, parser cost already paid) |

Reproduce (server dies, no 5xx, no log line):

```bash
ulimit -v 196608                     # 192 MiB address space; also reproduced at 160 MiB and 96 MiB
build-audit-http/vj_mock_serve --port 18331 &
python3 -c "import socket;b=b'{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":['+b'0,'*3999999+b'0]}]}';
s=socket.create_connection(('127.0.0.1',18331));s.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: %d\r\n\r\n'%len(b)+b);s.recv(100)"
# server: terminate called after throwing an instance of 'std::bad_alloc'   (exit -6 / SIGABRT)
```
Observed three times (`results11.json`): `alive:false, rc:-6`, and the earlier independent run in this tree recorded
the same (`build-audit-http/audit_http_results.json`: `json_under_192MiB_address_space … server_exit: -6`). On a
cgroup/`RLIMIT_AS`-limited deployment, or when several 8 MiB bodies are in flight (see S-M4), the same thing happens
without any attacker-supplied limit trick — malloc/NEW simply fails once.

Fix (all three, they are cheap):
1. Make `process_input()`/`route()` and the non-`gen` part of `run_job()` exception-safe: wrap each request in
   `try { … } catch (const std::bad_alloc&) { respond(503 or 500, "server out of memory"); close } catch (...){}`.
   An exception must never reach `io_loop()`/`worker_loop()`.
2. Bound the parser: count nodes in `json::Parser` (e.g. 100 000) and fail the parse with 413 when exceeded — that
   caps the tree at ~10 MB regardless of body size.
3. Shrink `Value` (a tagged union / `std::variant`, or store strings out-of-line in an arena) so a number costs
   16 B, not 96 B.

### S-B2 — 256 idle sockets make the entire API unreachable, and the refusal is a TCP RST, not a 503

`core/server.cpp:664`

```
core/server.cpp:664   if (conns.size() >= opt.max_connections) { ::close(fd); continue; }
```

Any connection over `max_connections` (default 256) is accepted and then `close()`d **with nothing written**, on a
socket that already has queued input — the kernel answers with RST. This is not a capacity signal, it is
indistinguishable from a crashed server, and it also hits `/health` and `/metrics`, so monitoring goes blind at
exactly the moment it matters. The holder cost is one `socket()` per connection and zero bytes.

Reproduced (`results3.json`): 260 sockets opened and left silent →
`health_while_exhausted: ConnectionResetError(104, 'Connection reset by peer')` immediately; every endpoint reset, and
`/health` answered 200 again only after the holders closed (`health_recovered`). The only thing that ever reclaims
those slots is the 30 s idle sweep, so a peer that re-opens 256 sockets every 30 s holds the API down indefinitely
(the earlier run in this tree measured the same refusal still in force at 32 s).

Fix:
1. Refuse with a response: build the 503 bytes once and `send(fd, 503, MSG_DONTWAIT|MSG_NOSIGNAL)`, then
   `shutdown(fd, SHUT_WR)`, then close — clients and probes then see `HTTP/1.1 503 … Retry-After: 1`.
2. Reserve headroom for the cheap endpoints: keep e.g. 8 slots that only `/health` + `/metrics` may use, or serve
   those from a separate accept path.
3. Rate-limit per peer address (a small token bucket keyed by the `accept()` address) so one host cannot hold the
   whole budget; and consider `SO_REUSEPORT`+`TCP_DEFER_ACCEPT` rather than an L1 accept cap.

---

## MAJOR

### S-M1 — A client that half-closes its write side gets an empty (non-stream) or truncated (stream) response

`core/server.cpp:733-737` (`peer_alive`), `core/server.cpp:906` (sink), `core/server.cpp:623` (EOF)

```
core/server.cpp:733   static bool peer_alive(int fd) {
core/server.cpp:734       pollfd p{fd, POLLRDHUP, 0};
core/server.cpp:735       if (::poll(&p, 1, 0) < 0) return true;
core/server.cpp:736       return (p.revents & (POLLRDHUP | POLLHUP | POLLERR | POLLNVAL)) == 0;
core/server.cpp:906           if (!peer_alive(fd)) { gone = true; return false; }
```

`POLLRDHUP` means *the peer finished sending*, not *the peer stopped reading*. HTTP/1.1 explicitly allows a client to
`shutdown(SHUT_WR)` after the request and then read the response (`nc -N`, several proxies, half-duplex fuzzers).
For such a client `on_readable` sets `peer_closed` (`:623`), the very first token trips `peer_alive()`, `gone = true`,
and `run_job` discards the response entirely (`:936 ++n_cancelled`).

Reproduced with the production adapter path (`results7.json`):

| case | bytes received by the client |
|---|---|
| `POST` (non-stream) + `SHUT_WR`, then read | **0 bytes — no HTTP response at all** |
| `POST stream:true` + `SHUT_WR`, then read | 519 B, 2 `data:` chunks, **no `finish_reason`, no `[DONE]`, connection closed** |

Nothing is logged, nothing is counted as an error, and the client just hangs until its own timeout. Same result
with the shipped `vj_mock_serve` (`results1.json`: `half_close_nonstream bytes:0`), so it is not a harness artefact.

Fix: delete the `peer_alive` check from the token sink and rely on the write path, which already detects death
(`send_blocking` → `EPIPE/ECONNRESET` → `pump` aborted → `push` false). A FIN on the read half is not evidence that
the response cannot be delivered. If a cheap early abort is still wanted, gate it on "the client has also not read
for `send_timeout_ms`" (a write timeout), never on RDHUP alone.

### S-M2 — Cancellations are invisible to the request log and to `vegajet_requests_cancelled_total`; aborted requests are logged as successful `length` completions

`core/server_engine_adapter.hpp:34`, `:41`; `core/metrics.cpp:414-415`

```
core/server_engine_adapter.hpp:34     res.finish = FinishReason::Length;
core/server_engine_adapter.hpp:41         if (!ev.text.empty() && !eos_token && !sink(ev.text)) return false;
core/metrics.cpp:414                  rec.cancelled = res.finish == serve::FinishReason::Cancelled;
core/metrics.cpp:415                  if (rec.cancelled) rec.status = 499;
```

When the sink returns `false` because the client went away, `run_engine` just stops iterating and returns the
initialised `FinishReason::Length`. Only the batch scheduler sets `Cancelled`
(`core/request_scheduler.cpp:365`), so the **default** single-slot serve path never reports a cancellation.

Reproduced (`results7.json`, real client abort mid-stream):

```
vegajet_requests_cancelled_total 0        # engine-side view
vegajet_http_cancelled_total 1            # server-side view: the server knew
JSONL: {"id":1,"status":200,"finish":"length","prompt_tokens":21,"completion_tokens":1,"cancelled":false}
```
The request log says a 1-token `length` completion happened. Every operator dashboard built on the JSONL (error
rate, tokens/s, truncation rate) is therefore wrong for exactly the requests people investigate.

Fix: make the sink report *why* it stopped. `using TokenSink = std::function<SinkResult(std::string_view)>` with
`{Continue, Stop, ClientGone}`; `run_engine` sets `Cancelled` on `ClientGone` and leaves `Length`/`Stop` to the
existing logic. Backport the same mapping in `request_scheduler.cpp` so both paths agree.

(Related but distinct from `bench/v12/REVIEW.md` N3, which covers a *final* write failure inside `OutputPump` never
reaching this accounting: together they mean a request can fail on the wire, or be abandoned mid-stream, and still be
recorded as a 200 completion.)

### S-M3 — `model` is never validated and is echoed into *every* SSE chunk (response amplification), unknown models get 200

`core/server.cpp:831`, `:832-837`, `:846`

```
core/server.cpp:831   const std::string model = json::quote(rq.model);
core/server.cpp:833       return "data: {\"id\":\"" + id + "\",\"object\":\"chat.completion.chunk\",\"created\":" + ts +
core/server.cpp:834              ",\"model\":" + model + ",\"choices\":[{…
```

`rq.model` comes straight from the client (any string, up to the 8 MiB body limit) and is embedded in the first
chunk, in every content delta, and in the final chunk. Measured (`results1.json`): a 4 191-byte request with a 4 KiB
model name produced a 42 894-byte stream (10×); with a 1 MiB model name and 1 000 tokens the same request emits
~1 GB. `model` is also not compared with `opt.model_id`, so `"model":"nonexistent-model"` returns 200 and the
response echoes the bogus name back.

Fix: `if (!rq.model.empty() && rq.model != opt.model_id) → 404 {"error":{"code":"model_not_found"}}` (or accept a
configured alias list); cap the accepted name length (e.g. 128 chars, else 400). Separately, do not repeat `model`
in every chunk beyond the first — 1 line of change, removes the amplification entirely.

### S-M4 — No prompt-size, context-size or `max_tokens` bound: one request grows the process to ~400 MB (≈49× the body), sixteen can pin the box

`core/server.cpp:296-301`, `core/server.hpp:73-74`, `cli/serve_cmd.cpp:236`

```
core/server.cpp:298   if (max_tokens < 1 || max_tokens > 1e9) { err = "'max_tokens' must be >= 1"; return false; }
core/server.cpp:301   if (opt.max_tokens_cap > 0) g.max_tokens = std::min(g.max_tokens, opt.max_tokens_cap);
core/server.hpp:74    int32_t max_tokens_cap = 0;         // 0: no cap
```

`max_tokens_cap` is never set by `vegajet serve` (no flag), `context_length` is only *reported* in `/v1/models`
(`server.cpp:547`), and the prompt is never measured before tokenization. Measured (`results9.json`, idle server,
2.9 GB free‑ish host, `--delay-ms 40` generator, real adapter):

| request body | VmHWM after | wall time (256-token reply at 40 ms/token ≈ 10.2 s fixed) |
|---|---|---|
| 1 MB prompt | 60 MB | 10.45 s |
| 2 MB prompt | 122 MB | 10.53 s |
| 4 MB prompt | 213 MB | 10.80 s |
| **8 MB prompt (the documented body limit)** | **391 MB** | **11.33 s** |

All four returned 200 — an 8 MB prompt is tokenized into ~2 M tokens for a 16 K-context model, which the engine then
has to reject, after the process has grown to 391 MB and spent ~0.9 s of extra CPU on the encode alone (the rest of
the wall time above is the fixed 256-token generation). `max_tokens: 1000000000` is accepted, so a
client can also occupy a slot until EOS or disconnection; with `max_queue` = 16 a single host can hold every slot.

Fix: (a) expose `--max-tokens-cap` and default it to `context_length`; (b) after tokenization (or via a byte→token
estimate before it) reject `prompt_tokens + max_tokens > context_length` with 400
`"context_length_exceeded"`; (c) cap the body accepted for `/v1/chat/completions` far below 8 MiB (a 16 K-token
prompt is ~64 KiB) while keeping 8 MiB for endpoints that need it.

### S-M5 — There is no authentication at all, the banner advertises an API key that is never checked, and `Access-Control-Allow-Origin: *` lets any web page drive the model

`cli/serve_cmd.cpp:302-303`, `core/server.cpp:226`

```
cli/serve_cmd.cpp:302   "[vegajet]   OpenAI API:    BASE_URL=http://%s:%u/v1  MODEL=%s  API_KEY=vegajet-local\n"
core/server.cpp:226     h += "Access-Control-Allow-Origin: *\r\n";
```

`Authorization`/`x-api-key` appear nowhere in the server except in the CORS `Access-Control-Allow-Headers` list
(`server.cpp:518`) — no header is ever validated. So the printed `API_KEY=vegajet-local` is decorative: clients that
send it work, clients that send nothing (or anything) work identically. Combined with `ACAO: *` on every response,
**any page the operator visits** can `fetch("http://127.0.0.1:8080/v1/chat/completions")` and both drive the box and
read the completions back (DNS-rebinding / CSRF into the local port; the same class of issue `OLLAMA_ORIGINS` exists
for). The default bind (`127.0.0.1`, `server.hpp:71`) is correct and should stay the default.

Fix: (a) either implement the key — `--api-key`, constant-time compare against a `Authorization: Bearer TOKEN`
prefix **and** an `x-api-key: TOKEN` header (Anthropic clients), answering 401
`{"error":{"type":"authentication_error"}}` otherwise — or stop printing one; (b) keep the loopback-only default
bind and refuse to bind a non-loopback address when no key is configured (warn, exit 3); (c) narrow CORS: echo
`Origin` only for a configured allowlist instead of sending `Access-Control-Allow-Origin: *` on every response.

---

## MINOR

**S-m1 Duplicate `Content-Length` accepted, last value wins** — `server.cpp:173-180`. No conflict check
(`if (content_length != <previous>) → 400`), so `Content-Length: 5` + `Content-Length: 66` parses as 66
(`results1.json`: 200). Harmless in front of nothing, a smuggling primitive in front of any proxy that uses the
first. Fix: record that CL was seen and reject a second, differing value (RFC 9112 §6.3).

**S-m2 `Transfer-Encoding` answered with 501 + `"type":"server_error"`** — `server.cpp:177-179`. Any value,
including `identity`, is refused; the body says "chunked request bodies are not supported", the error type says the
*server* failed. Chunked request bodies are a documented limitation, but the status/type should be
`411 Length Required` + `invalid_request_error` (or implement chunked decode — it is ~40 lines and some SDK proxies
always chunk POSTs).

**S-m3 `top_k`/`seed` out of range are undefined behaviour, `max_tokens: 1.5` is silently truncated** —
`server.cpp:314-315`:
```
core/server.cpp:314   g.sampling.top_k = static_cast<int32_t>(std::max(0.0, top_k));
core/server.cpp:315   g.sampling.seed = static_cast<uint64_t>(std::max(0.0, seed));
```
UBSan (`build-audit-http/ubsan2`, `-fsanitize=undefined,float-cast-overflow`):
`core/server.cpp:314:53: runtime error: 1e+100 is outside the range of representable values of type 'int'` for
`{"top_k": 1e100}` — accepted with 200 by the normal build. `temperature`, `top_p`, `min_p`, `repeat_penalty` are all
range-checked; these two are not. Fix: `if (top_k < 0 || top_k > 2^31-1) → 400`, same for `seed`, and require
`max_tokens`/`n` to be integral (`std::floor(v) == v`).

**S-m4 Malformed-but-parsable headers are silently accepted** — `server.cpp:171-186`. A header whose name contains a
space (`Content-Length : 5`, `X-Bad Header: yes`) or a NUL in the value is stored and ignored rather than rejected;
obs-fold is silently dropped. `results1.json`/earlier run: all 200. Fix: validate the name against the RFC token set
and reject anything else with 400; reject obs-fold.

**S-m5 Invalid UTF-8 is reflected verbatim, so responses are not valid JSON** — `json_lite.hpp:195-213`
(`escape()` passes bytes ≥ 0x80 through). A request whose content (or `model`) contains a lone `\xff` is echoed into
`choices[].message.content` and into every chunk; both endpoints (`results` from the earlier run: `invalid_utf8 …
response_valid_utf8_json: false`). Fix: decode-validate request strings at parse time (400) or sanitize on output
(replace invalid sequences with U+FFFD in `escape()`); the `StopMatcher` already does this for model output.

**S-m6 `vegajet_http_requests_total` counts health probes and every 4xx, but there is no 4xx counter** —
`metrics.cpp:374-381` + `server.cpp:507` (`++n_requests` before routing). Measured (`results5.json`): 3 chat POSTs
(one of them rejected with 503) + 9 `/health` GETs → `requests 12`. So `requests = completions + errors + cancelled +
rejected` never holds, and a 1 s Prometheus scrape interval alone fabricates a steady request rate. Fix: either
exclude `/health`, `/metrics` and pre-generation 4xx from `n_requests`, or add
`vegajet_http_client_errors_total{status}` so the identity closes.

**S-m7 The request log only ever contains completions** — `metrics.cpp:385-423`: `instrument()` wraps the
*generator*, so 400/404/405/413/431/501/503 responses (all produced in `route()`, before `gen`) leave no JSONL
record. Debugging "clients get 400 from my endpoint" is impossible from the logs. Fix: emit a `RequestRecord` for
rejected requests too (`finish:"rejected"`, `status:<code>`), from `route()` via a logger handle.

**S-m8 `/metrics` does blocking sysfs reads on the epoll thread** — `cli/serve_cmd.cpp:282-288`
(`read_gpu_sysfs("/sys")` inside `metrics_text`), called from `route()` (`server.cpp:527`) on the I/O thread. With
`--metrics-hw` each scrape opens ~10 files per card while all sockets are stalled. Fix: read the GPU telemetry on a
timer (e.g. every 5 s in a helper thread) and have `metrics_text` snapshot the cache.

**S-m9 OpenAI/Anthropic schema gaps that change client-visible behaviour**
* `response_format` is accepted and silently ignored (`{"type":"json_object"}` → 200, plain prose) — either 400 or
  document it; silently not honouring structured output is worse than refusing it.
* `logprobs: true` / `top_logprobs` are ignored on non-greedy engines (no `logprobs` in the response).
* `temperature: 1.0` — the documented default of *both* APIs — is rejected on a `greedy_only` engine
  (`server.cpp:355-372`) with "not implemented (greedy default 0)", while `top_p: 1.0` is accepted as the greedy
  value. A client that sends the API default now gets a 400; accept `temperature == 1.0` as "no scaling", or at
  minimum document it in the error text.
* `/v1/messages` treats `max_tokens` as optional (`anthropic_shim.cpp:274`, default 256) where the real API rejects
  its absence with 400 `invalid_request_error`. `message_delta.usage` also carries `input_tokens`, which Anthropic
  only reports in `message_start`.
* `anthropic-version` and `x-api-key` are ignored (consequence of S-M5).

**S-m10 Close after `Connection: close` does not drain unread input** — `server.cpp:471` (flush then `close_conn`)
and `server.cpp:644` (`::close(x.fd)` in `drain_done`). If the peer sent a pipelined request whose bytes the server has
not read, `close()` on a socket with unread data emits RST and the kernel may discard the tail of the response still
in the send buffer. Fix: `shutdown(fd, SHUT_WR)` first, then read-and-discard until EOF/`EAGAIN` with a short
deadline, then close (standard graceful-close).

**S-m11 Slow-loris on headers is only bounded by `max_connections`** — `server.cpp:619-621`: `c.last_ms = now_ms()`
is refreshed on *every* received byte, so a client sending one byte per 29 s keeps a connection forever. A
half-completed body has no 408 either (the client just sits until the 30 s idle sweep). Fix: separate
`last_progress_ms` from the request's start time and enforce a header/body deadline (e.g. 10 s to complete a
request head, 60 s for the body) → `408 Request Timeout`.

**S-m12 `/health`'s `in_flight` mixes queued and running** — `server.cpp:541-543` reports the `max_queue` counter,
while `vegajet_active_requests` (`metrics.cpp:332`) counts only generating requests. Two numbers, one name; a stuck
queue looks identical to a busy GPU. Fix: report `{"queued":N,"running":M}`, or rename to `queue_depth`.

**S-m13 503 is decided while holding the queue mutex and answered inside it** — `server.cpp:589-597`: `fail(503,…)`
is called under `std::lock_guard lk(mu)`, so the io thread does a socket write (and possibly a close) with the queue
lock held. Not a deadlock today (the io thread is the only writer to `conns`), but it is an unnecessary coupling.
Fix: decide under the lock, unlock, then `respond()`.

**S-m14 `JsonlLogger::log` ignores short writes** — `metrics.cpp:216-224`: `const ssize_t n = ::write(fd_, line.data(),
line.size());` and `lines_` is bumped only on a full write, so a partial write leaves a truncated JSONL line (invalid
JSON for every consumer) with no counter to notice it. Fix: loop until the line is written, or drop the record and
bump a `log_errors` counter.

---

## BLOCKERS

1. **S-B1** — a single request can abort the process: the JSON `Value` tree costs ~54× the body (2 MB → 191 MB
   VmSize / 108 MB HWM) and `json::parse` (plus the worker's string building) runs with no `catch`, so malloc
   failure → `std::bad_alloc` → `std::terminate` → SIGABRT. Reproduced (`rc:-6`) at 8 MB/192 MiB, 4 MB/160 MiB,
   2 MB/96 MiB.
2. **S-B2** — 256 idle sockets make every endpoint, including `/health`, refuse connections with a TCP RST instead of
   a 503 (no capacity signal, no error body, monitoring blind). Trivially remote, unauthenticated.

## MAJORS

1. **S-M1** — a client that half-closes (`SHUT_WR`, legal HTTP) gets **0 bytes** for a non-stream request and a
   truncated SSE stream with no `[DONE]` for a streaming one; `peer_alive()` treats `POLLRDHUP` as "gone".
2. **S-M2** — client aborts are logged as `status:200, finish:"length", cancelled:false` and never increment
   `vegajet_requests_cancelled_total`, while `vegajet_http_cancelled_total` does; `run_engine` never sets
   `FinishReason::Cancelled`.
3. **S-M3** — `model` is unvalidated and repeated in every SSE chunk: a 4 KB request yields 42 KB of output and a
   1 MB model name yields ~1 GB per 1 000 tokens; unknown models answer 200.
4. **S-M4** — no prompt/context/`max_tokens` bound: an 8 MB body is accepted, grows the process to **391 MB**
   (~49× the body) and costs ~0.9 s of extra CPU per request; `--max-tokens-cap` is not wired (default 0), so a
   single request can hold a queue slot indefinitely.
5. **S-M5** — no authentication whatsoever, a decorative `API_KEY=vegajet-local` in the banner, and
   `Access-Control-Allow-Origin: *` on every response: any web page can drive the model and read the output.
