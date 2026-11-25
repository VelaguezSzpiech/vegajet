#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""HTTP/SSE load balancer for parallel coding sub-agents in front of independent N=1 `vegajet serve` workers.

    swarm_router.py --listen 127.0.0.1:8090 --worker http://127.0.0.1:8081 --worker http://127.0.0.1:8082 \\
                    --worker http://127.0.0.1:8083 [--max-queue 64] [--queue-timeout 600] [--affinity-wait 0.25]

Each worker runs one generation at a time (one GPU or one TP group each), so the router never sends a worker a
second request while one is in flight; extra requests wait in one FIFO queue (bounded: beyond --max-queue the
client gets 503 + Retry-After, like `vegajet serve` itself).

Routing: requests of the same conversation (same messages except the last one) prefer the same worker
(rendezvous hashing over healthy workers), so the worker's KV / prefix state for that sub-agent stays warm. If
the preferred worker is busy the request waits up to --affinity-wait seconds for it, then takes any idle worker.

Streaming responses (text/event-stream) are relayed chunk by chunk. When the client goes away the upstream
connection is closed immediately, so the worker sees the disconnect and stops generating (vegajet serve does).
A worker that refuses connections or fails its /health probe is taken out of rotation until a probe succeeds;
a request that could not be delivered to a worker is retried on another one (never after bytes reached the client).

Endpoints: POST /v1/chat/completions (routed), GET /v1/models (any healthy worker), GET /health (router state).
Responses carry X-Swarm-Worker: <index>. Standard library only.
"""
import argparse
import hashlib
import http.client
import http.server
import json
import select
import socket
import sys
import threading
import time
import urllib.parse


class Worker:
    def __init__(self, index, url):
        u = urllib.parse.urlparse(url)
        if u.scheme != "http" or not u.hostname:
            raise ValueError("worker URL must be http://host:port: %r" % url)
        self.index, self.url, self.host, self.port = index, url, u.hostname, u.port or 80
        self.healthy, self.busy, self.served, self.failures = True, False, 0, 0

    def state(self):
        return {"index": self.index, "url": self.url, "healthy": self.healthy, "busy": self.busy,
                "served": self.served, "failures": self.failures}


class Pool:
    """One FIFO queue over N single-slot workers."""

    def __init__(self, urls, max_queue, affinity_wait):
        self.workers = [Worker(i, u) for i, u in enumerate(urls)]
        self.max_queue, self.affinity_wait = max_queue, affinity_wait
        self.cv = threading.Condition()
        self.next_ticket = self.serving = 0
        self.waiting = 0
        self.skipped = set()  # tickets that left the line before their turn

    def ranked(self, key):
        def score(w):
            return hashlib.sha256(("%s|%d" % (key, w.index)).encode()).digest()
        return sorted((w for w in self.workers if w.healthy), key=score, reverse=True)

    def acquire(self, key, timeout, gone=lambda: False):
        """Returns a worker (marked busy), None on timeout / client gone, or 'full' when the queue is full."""
        with self.cv:
            # Requests that an idle worker will take (possibly after the short affinity wait) are not "queued".
            idle = sum(1 for w in self.workers if w.healthy and not w.busy)
            if self.waiting - idle >= self.max_queue:
                return "full"
            ticket = self.next_ticket
            self.next_ticket += 1
            self.waiting += 1
            deadline = time.monotonic() + timeout
            first_turn = None
            try:
                while True:
                    while self.serving in self.skipped:
                        self.skipped.discard(self.serving)
                        self.serving += 1
                    now = time.monotonic()
                    if ticket == self.serving:
                        if first_turn is None:
                            first_turn = now
                        order = self.ranked(key)
                        idle = [w for w in order if not w.busy]
                        if order and not order[0].busy:
                            pick = order[0]
                        elif idle and now - first_turn >= self.affinity_wait:
                            pick = idle[0]
                        else:
                            pick = None
                        if pick is not None:
                            pick.busy = True
                            self.serving += 1
                            self.cv.notify_all()
                            return pick
                    if now >= deadline or gone():
                        if ticket == self.serving:
                            self.serving += 1
                        else:  # leave the line: later tickets must not wait for this one
                            self.skipped.add(ticket)
                        self.cv.notify_all()
                        return None
                    self.cv.wait(min(0.05, max(0.0, deadline - now)))
            finally:
                self.waiting -= 1

    def release(self, w, ok=True, unhealthy=False):
        with self.cv:
            w.busy = False
            if ok:
                w.served += 1
            else:
                w.failures += 1
            if unhealthy:
                w.healthy = False
            self.cv.notify_all()

    def mark(self, w, healthy):
        with self.cv:
            if w.healthy != healthy:
                w.healthy = healthy
                self.cv.notify_all()

    def snapshot(self):
        with self.cv:
            return [w.state() for w in self.workers], self.waiting


def affinity_key(doc):
    msgs = doc.get("messages") if isinstance(doc, dict) else None
    if not isinstance(msgs, list) or not msgs:
        return ""
    head = msgs[:-1] if len(msgs) > 1 else msgs[:1]
    return hashlib.sha256(json.dumps(head, sort_keys=True).encode()).hexdigest()


def make_handler(pool, args):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "vegajet-swarm-router"

        def log_message(self, fmt, *a):  # quiet by default
            if args.verbose:
                sys.stderr.write("router: " + fmt % a + "\n")

        def send_json(self, code, obj, extra=None):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def client_gone(self):
            try:
                r, _, _ = select.select([self.connection], [], [], 0)
                return bool(r) and self.connection.recv(1, socket.MSG_PEEK) == b""
            except OSError:
                return True

        def do_GET(self):
            if self.path == "/health":
                workers, waiting = pool.snapshot()
                up = sum(w["healthy"] for w in workers)
                self.send_json(200 if up else 503, {"status": "ok" if up == len(workers) else ("degraded" if up else "down"),
                                                    "workers": workers, "queued": waiting})
                return
            if self.path == "/v1/models":
                for w in pool.ranked(""):
                    try:
                        c = http.client.HTTPConnection(w.host, w.port, timeout=5)
                        c.request("GET", "/v1/models")
                        r = c.getresponse()
                        body = r.read()
                        c.close()
                        self.send_response(r.status)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(body)))
                        self.send_header("X-Swarm-Worker", str(w.index))
                        self.end_headers()
                        self.wfile.write(body)
                        return
                    except OSError:
                        pool.mark(w, False)
                self.send_json(503, {"error": {"message": "no healthy worker", "type": "unavailable"}}, {"Retry-After": "1"})
                return
            self.send_json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})
                return
            try:
                n = int(self.headers.get("Content-Length", "-1"))
            except ValueError:
                n = -1
            if n < 0 or n > args.max_body:
                self.send_json(413 if n > args.max_body else 411, {"error": {"message": "bad Content-Length", "type": "invalid_request_error"}})
                self.close_connection = True
                return
            body = self.rfile.read(n)
            try:
                doc = json.loads(body)
            except ValueError:
                self.send_json(400, {"error": {"message": "invalid JSON", "type": "invalid_request_error"}})
                return
            key = affinity_key(doc)
            for _attempt in range(len(pool.workers) + 1):
                w = pool.acquire(key, args.queue_timeout, self.client_gone)
                if w == "full":
                    self.send_json(503, {"error": {"message": "router queue is full", "type": "server_busy"}}, {"Retry-After": "1"})
                    return
                if w is None:
                    if not self.client_gone():
                        self.send_json(503, {"error": {"message": "no worker became free in time", "type": "server_busy"}},
                                       {"Retry-After": "1"})
                    self.close_connection = True
                    return
                outcome = self.forward(w, body)
                if outcome != "retry":
                    return
            self.send_json(502, {"error": {"message": "no worker accepted the request", "type": "bad_gateway"}})

        def forward(self, w, body):
            up = http.client.HTTPConnection(w.host, w.port, timeout=args.upstream_timeout)
            try:
                up.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
                r = up.getresponse()
            except OSError:
                up.close()
                pool.release(w, ok=False, unhealthy=True)  # refused / reset before any response: try another
                return "retry"
            if r.status == 503:  # worker busy (someone else talks to it directly): give it back and retry
                r.read()
                up.close()
                pool.release(w, ok=False)
                time.sleep(0.05)
                return "retry"
            ctype = r.getheader("Content-Type", "application/json")
            ok = True
            try:
                if ctype.startswith("text/event-stream"):
                    self.send_response(r.status)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.send_header("X-Swarm-Worker", str(w.index))
                    self.end_headers()
                    self.close_connection = True
                    while True:
                        chunk = r.read1(65536)
                        if not chunk:
                            break
                        try:
                            self.wfile.write(chunk)
                            self.wfile.flush()
                        except OSError:  # client gone: drop upstream now so the worker cancels
                            ok = False
                            break
                else:
                    data = r.read()
                    self.send_response(r.status)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("X-Swarm-Worker", str(w.index))
                    self.end_headers()
                    self.wfile.write(data)
            except OSError:
                ok = False
            finally:
                try:
                    up.sock and up.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                up.close()
                pool.release(w, ok=ok)
            return "done"

    return Handler


def health_loop(pool, interval, stop):
    while not stop.wait(interval):
        for w in pool.workers:
            if w.busy and w.healthy:
                continue  # an N=1 worker answers /health while generating, but skip it to keep probes cheap
            try:
                c = http.client.HTTPConnection(w.host, w.port, timeout=2)
                c.request("GET", "/health")
                ok = c.getresponse().status == 200
                c.close()
            except OSError:
                ok = False
            pool.mark(w, ok)


class _Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError, TimeoutError)):
            return  # client went away between keep-alive requests: nothing to report
        super().handle_error(request, client_address)


class Router:
    """Embeddable router (used by the tests): Router(args).start() / .stop(); .port is the bound port."""

    def __init__(self, args):
        self.pool = Pool(args.worker, args.max_queue, args.affinity_wait)
        host, port = args.listen.rsplit(":", 1)
        self.httpd = _Server((host, int(port)), make_handler(self.pool, args))
        self.port = self.httpd.server_address[1]
        self.stop_ev = threading.Event()
        self.threads = [threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True),
                        threading.Thread(target=health_loop, args=(self.pool, args.health_interval, self.stop_ev), daemon=True)]

    def start(self):
        for t in self.threads:
            t.start()
        return self

    def stop(self):
        self.stop_ev.set()
        self.httpd.shutdown()
        self.httpd.server_close()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen", default="127.0.0.1:8090", help="host:port (port 0: ephemeral)")
    ap.add_argument("--worker", action="append", required=True, help="worker base URL (repeat once per worker)")
    ap.add_argument("--max-queue", type=int, default=64)
    ap.add_argument("--queue-timeout", type=float, default=600.0, help="seconds a request may wait for a worker")
    ap.add_argument("--affinity-wait", type=float, default=0.25, help="seconds to wait for the preferred worker")
    ap.add_argument("--health-interval", type=float, default=2.0)
    ap.add_argument("--upstream-timeout", type=float, default=900.0)
    ap.add_argument("--max-body", type=int, default=8 << 20)
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    r = Router(args).start()
    print("swarm_router listening on %s (port %d), %d workers" % (args.listen, r.port, len(args.worker)), flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        r.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
