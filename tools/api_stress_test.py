#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Stress and determinism harness for an OpenAI-compatible VegaJet server (stdlib only).

    api_stress_test.py --url http://127.0.0.1:8080 [--clients 24] [--rounds 3] [--json]
    api_stress_test.py --spawn build/bin/vj_mock_serve [--spawn-arg=--max-queue --spawn-arg=2 --expect-backpressure]
                       (starts the server on an ephemeral port)

Checks (every one must pass for exit code 0):
  health        GET /health and /v1/models answer and name the model.
  baseline      non-streaming greedy completion; its content SHA-256 is the reference.
  concurrent    clients x rounds streaming requests at once; every stream's content hashes to the reference.
                503 + Retry-After (queue full) is retried with backoff and counted, never treated as a failure.
  disconnects   clients that vanish mid-stream (RST and orderly close); afterwards in_flight returns to 0 and the
                next request still hashes to the reference.
  stops         stop sequences (inside one token, across tokens, several at once) give the same text and
                finish_reason streaming and non-streaming; max_tokens gives finish_reason "length".
  prefix        repeated requests that share a long prefix: identical prompt_tokens and content hashes; reports
                usage.prompt_tokens_details.cached_tokens when the server provides it.
  errors        malformed JSON -> 400, unknown path -> 404, and the server keeps serving.
Only content and finish_reason enter the hashes (ids and timestamps differ per request).
"""
import argparse
import hashlib
import http.client
import json
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.parse

MESSAGES = [{"role": "user", "content": "Say hello."}]


class Client:
    def __init__(self, host, port, timeout=30.0):
        self.host, self.port, self.timeout = host, port, timeout

    def conn(self):
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    def get(self, path):
        c = self.conn()
        try:
            c.request("GET", path)
            r = c.getresponse()
            return r.status, r.read()
        finally:
            c.close()

    def post_raw(self, path, body, headers=None):
        c = self.conn()
        try:
            h = {"Content-Type": "application/json"}
            h.update(headers or {})
            c.request("POST", path, body=body, headers=h)
            r = c.getresponse()
            return r.status, dict(r.getheaders()), r.read()
        finally:
            c.close()

    def chat(self, payload, retries=200, stats=None):
        """Returns (status, content, finish_reason, usage). Streams when payload['stream'] is true."""
        body = json.dumps(payload).encode()
        delay = 0.05
        for _ in range(retries):
            c = self.conn()
            try:
                c.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
                r = c.getresponse()
                if r.status == 503:
                    r.read()
                    if stats is not None:
                        with stats["lock"]:
                            stats["busy_503"] += 1
                    wait = float(r.getheader("Retry-After", "1") or 1)
                    time.sleep(min(delay, wait))
                    delay = min(delay * 1.5, 1.0)
                    continue
                if r.status != 200:
                    return r.status, r.read().decode(errors="replace"), None, None
                if not payload.get("stream"):
                    doc = json.loads(r.read())
                    ch = doc["choices"][0]
                    return 200, ch["message"]["content"], ch.get("finish_reason"), doc.get("usage")
                return (200,) + parse_sse(r)
            finally:
                c.close()
        return 503, "", None, None


def parse_sse(resp):
    text, finish, usage, done = [], None, None, False
    buf = b""
    while True:
        chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            event, buf = buf.split(b"\n\n", 1)
            for line in event.split(b"\n"):
                if not line.startswith(b"data: "):
                    continue
                data = line[6:]
                if data == b"[DONE]":
                    done = True
                    continue
                doc = json.loads(data)
                if doc.get("usage"):
                    usage = doc["usage"]
                for ch in doc.get("choices", []):
                    if ch.get("delta", {}).get("content"):
                        text.append(ch["delta"]["content"])
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
    if not done:
        raise RuntimeError("stream ended without data: [DONE]")
    return "".join(text), finish, usage


def sha(content, finish):
    return hashlib.sha256(json.dumps([content, finish]).encode()).hexdigest()


class Report:
    def __init__(self):
        self.checks, self.facts = [], {}

    def check(self, name, ok, detail=""):
        self.checks.append({"name": name, "ok": bool(ok), "detail": detail})
        print(("  PASS " if ok else "  FAIL ") + name + (": " + detail if detail else ""), flush=True)
        return ok

    @property
    def ok(self):
        return all(c["ok"] for c in self.checks)


def run_health(cl, rep):
    st, body = cl.get("/health")
    doc = json.loads(body) if st == 200 else {}
    rep.check("health", st == 200 and doc.get("status") == "ok", body.decode(errors="replace")[:120])
    st, body = cl.get("/v1/models")
    doc = json.loads(body) if st == 200 else {}
    ids = [m.get("id") for m in doc.get("data", [])]
    rep.check("models", st == 200 and len(ids) >= 1, "ids %s" % ids)
    return ids[0] if ids else None


def run_baseline(cl, rep, model):
    st, content, finish, usage = cl.chat({"model": model, "messages": MESSAGES, "max_tokens": 64, "temperature": 0})
    rep.check("baseline", st == 200 and content and finish in ("stop", "length"),
              "%r finish=%s usage=%s" % (content, finish, usage))
    st2, content2, finish2, _ = cl.chat({"model": model, "messages": MESSAGES, "max_tokens": 64, "temperature": 0,
                                         "stream": True})
    rep.check("baseline.stream_equals_nonstream", st2 == 200 and (content2, finish2) == (content, finish),
              "%r finish=%s" % (content2, finish2))
    rep.facts["reference_sha256"] = sha(content, finish)
    return sha(content, finish), usage


def run_concurrent(cl, rep, model, ref, clients, rounds):
    stats = {"lock": threading.Lock(), "busy_503": 0}
    results, errors = [], []
    lock = threading.Lock()

    def worker():
        for _ in range(rounds):
            try:
                st, content, finish, _ = cl.chat({"model": model, "messages": MESSAGES, "max_tokens": 64,
                                                  "temperature": 0, "stream": True}, stats=stats)
                with lock:
                    results.append((st, sha(content, finish)))
            except Exception as e:  # noqa: BLE001 - recorded as a failure
                with lock:
                    errors.append(repr(e))

    t0 = time.time()
    ts = [threading.Thread(target=worker) for _ in range(clients)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    dt = time.time() - t0
    good = sum(1 for st, h in results if st == 200 and h == ref)
    total = clients * rounds
    rep.facts["concurrent"] = {"requests": total, "ok": good, "busy_503_retries": stats["busy_503"], "seconds": round(dt, 3)}
    rep.check("concurrent.sse_deterministic", good == total and not errors,
              "%d/%d streams hash to the reference, %d x 503 retried, %.2f s%s" %
              (good, total, stats["busy_503"], dt, (", errors: %s" % errors[:3]) if errors else ""))


def raw_stream_then_drop(host, port, model, rst):
    s = socket.create_connection((host, port), timeout=10)
    body = json.dumps({"model": model, "messages": MESSAGES, "max_tokens": 64, "stream": True}).encode()
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
              b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
    got = b""
    while b"\"content\":\"" not in got.replace(b"\"content\":\"\"", b""):  # first non-empty delta
        chunk = s.recv(4096)
        if not chunk:
            break
        got += chunk
    if rst:  # SO_LINGER 0 -> RST instead of FIN
        s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    s.close()
    return b"200 OK" in got


def run_disconnects(cl, rep, model, ref, n):
    ok = [raw_stream_then_drop(cl.host, cl.port, model, rst=(i % 2 == 0)) for i in range(n)]
    deadline, in_flight = time.time() + 10, None
    while time.time() < deadline:
        st, body = cl.get("/health")
        in_flight = json.loads(body).get("in_flight") if st == 200 else None
        if in_flight == 0:
            break
        time.sleep(0.05)
    st, content, finish, _ = cl.chat({"model": model, "messages": MESSAGES, "max_tokens": 64, "stream": True})
    rep.check("disconnects", all(ok) and in_flight == 0 and st == 200 and sha(content, finish) == ref,
              "%d clients dropped mid-stream (%d RST), in_flight back to %s, next stream matches the reference" %
              (n, (n + 1) // 2, in_flight))


def run_stops(cl, rep, model, full):
    cases = []
    words = full.split(" ")
    if len(words) >= 3:
        cases.append(([" " + words[2]], " ".join(words[:2])))             # stop at a whole later token
    if len(full) > 12:
        mid = full[7:12]                                                  # spans token boundaries
        cases.append(([mid], full[:full.find(mid)]))
        cases.append((["zzz-not-present", mid, full[3:5]], full[:3]))     # several stops: earliest wins
    cases.append((["zzz-not-present"], full))
    all_ok, details = True, []
    for stops, want in cases:
        outs = []
        for stream in (False, True):
            st, content, finish, _ = cl.chat({"model": model, "messages": MESSAGES, "max_tokens": 64,
                                              "stop": stops, "stream": stream})
            outs.append((st, content, finish))
        expect_finish = "stop"
        good = all(o == (200, want, expect_finish) for o in outs) or (
            want == full and all(o[0] == 200 and o[1] == full for o in outs) and outs[0][2] == outs[1][2])
        all_ok &= good
        details.append("%r->%r%s" % (stops, outs[0][1], "" if good else " (stream %r)" % (outs[1],)))
    st, content, finish, usage = cl.chat({"model": model, "messages": MESSAGES, "max_tokens": 2})
    st2, content2, finish2, _ = cl.chat({"model": model, "messages": MESSAGES, "max_tokens": 2, "stream": True})
    length_ok = st == 200 and finish == "length" and (content2, finish2) == (content, finish) and \
        (usage or {}).get("completion_tokens") == 2
    rep.check("stops", all_ok, "; ".join(details))
    rep.check("max_tokens.length", length_ok, "%r finish=%s usage=%s" % (content, finish, usage))


def run_prefix(cl, rep, model, repeats=4):
    system = {"role": "system", "content": "You are a careful coding assistant. " * 40}
    hashes, prompt_tokens, cached = {}, {}, []
    for i in range(repeats):
        for tail in ("Write a unit test.", "Explain the bug."):
            msgs = [system, {"role": "user", "content": tail}]
            st, content, finish, usage = cl.chat({"model": model, "messages": msgs, "max_tokens": 32,
                                                  "stream": bool(i % 2), "stream_options": {"include_usage": True}})
            if st != 200:
                rep.check("prefix", False, "HTTP %s" % st)
                return
            hashes.setdefault(tail, set()).add(sha(content, finish))
            prompt_tokens.setdefault(tail, set()).add((usage or {}).get("prompt_tokens"))
            details = (usage or {}).get("prompt_tokens_details") or {}
            if "cached_tokens" in details:
                cached.append(details["cached_tokens"])
    same = all(len(v) == 1 for v in hashes.values()) and all(len(v) == 1 and None not in v for v in prompt_tokens.values())
    pt = {k: next(iter(v)) for k, v in prompt_tokens.items()}
    rep.facts["prefix"] = {"prompt_tokens": pt, "cached_tokens": cached or "not reported by this server"}
    rep.check("prefix.repeatable", same, "prompt_tokens %s, content identical across %d repeats (streaming and not); "
              "cached_tokens: %s" % (pt, repeats, cached if cached else "not reported by this server"))


def run_errors(cl, rep, model, ref):
    st1, _, _ = cl.post_raw("/v1/chat/completions", b"{not json")
    st2, _ = cl.get("/v1/nothing-here")
    st3, _, _ = cl.post_raw("/v1/chat/completions", json.dumps({"model": model, "messages": "x"}).encode())
    st, content, finish, _ = cl.chat({"model": model, "messages": MESSAGES, "max_tokens": 64})
    rep.check("errors", st1 == 400 and st2 == 404 and st3 == 400 and st == 200 and sha(content, finish) == ref,
              "malformed JSON %s, unknown path %s, bad messages %s, server still serving" % (st1, st2, st3))


def spawn(binary, extra):
    p = subprocess.Popen([binary, "--port", "0"] + list(extra), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    line = p.stdout.readline()
    if not line.startswith("listening on "):
        p.kill()
        raise SystemExit("server did not start: %r %s" % (line, p.stderr.read()))
    host, port = line.split()[-1].rsplit(":", 1)
    return p, host, int(port)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--url", help="base URL of a running server")
    g.add_argument("--spawn", metavar="BIN", help="start BIN --port 0 and test it (e.g. build/bin/vj_mock_serve)")
    ap.add_argument("--clients", type=int, default=24)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--disconnects", type=int, default=12)
    ap.add_argument("--spawn-arg", action="append", default=[], metavar="ARG",
                    help="extra argument for --spawn (repeatable), e.g. --spawn-arg=--max-queue --spawn-arg=2")
    ap.add_argument("--expect-backpressure", action="store_true",
                    help="require at least one 503 + Retry-After during the concurrent phase (small server queue)")
    ap.add_argument("--json", action="store_true", help="print a JSON summary at the end")
    a = ap.parse_args(argv)
    proc = None
    if a.spawn:
        proc, host, port = spawn(a.spawn, a.spawn_arg)
    else:
        u = urllib.parse.urlparse(a.url)
        host, port = u.hostname, u.port or 80
    rep = Report()
    try:
        cl = Client(host, port)
        model = run_health(cl, rep)
        ref, _ = run_baseline(cl, rep, model)
        _, full, _, _ = cl.chat({"model": model, "messages": MESSAGES, "max_tokens": 64})
        run_concurrent(cl, rep, model, ref, a.clients, a.rounds)
        if a.expect_backpressure:
            n = rep.facts["concurrent"]["busy_503_retries"]
            rep.check("concurrent.backpressure", n > 0, "%d requests saw 503 + Retry-After and succeeded on retry" % n)
        run_disconnects(cl, rep, model, ref, a.disconnects)
        run_stops(cl, rep, model, full)
        run_prefix(cl, rep, model)
        run_errors(cl, rep, model, ref)
    finally:
        if proc:
            proc.terminate()
            try:
                _, err = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                _, err = proc.communicate()
            rep.facts["server_exit"] = proc.returncode
            rep.facts["server_stderr"] = err.strip()[-400:]
            # A sanitizer report or a crash in the server is a failure even if every HTTP check passed.
            rep.check("server.clean_exit", proc.returncode == 0 and "runtime error" not in err and "Sanitizer" not in err,
                      "exit %s; %s" % (proc.returncode, err.strip().splitlines()[-1] if err.strip() else ""))
    if a.json:
        print(json.dumps({"ok": rep.ok, "checks": rep.checks, "facts": rep.facts}, indent=2))
    print("api_stress_test: %d/%d checks passed" % (sum(c["ok"] for c in rep.checks), len(rep.checks)))
    print("PASS" if rep.ok else "FAIL")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
