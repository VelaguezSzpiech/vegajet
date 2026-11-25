# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""tools/swarm_router.py against in-process fake N=1 workers (and, when VJ_MOCK_SERVE points at the vj_mock_serve
binary, three real HEAD OpenAiServer + MockEngine processes). Everything binds 127.0.0.1 with ephemeral ports."""
import http.client
import http.server
import json
import os
import socket
import subprocess
import sys
import threading
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import swarm_router  # noqa: E402


class FakeWorker:
    """One-generation-at-a-time worker that records overlap, cancellations and what it served."""

    def __init__(self, idx, delay=0.03, chunks=5):
        self.idx, self.delay, self.chunks = idx, delay, chunks
        self.lock = threading.Lock()
        self.inflight = self.max_inflight = self.cancelled = self.served = 0
        fw = self

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def reply(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                if self.path == "/health":
                    self.reply(200, {"status": "ok"})
                elif self.path == "/v1/models":
                    self.reply(200, {"object": "list", "data": [{"id": "fake-%d" % fw.idx}]})
                else:
                    self.reply(404, {})

            def do_POST(self):
                doc = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with fw.lock:
                    fw.inflight += 1
                    fw.max_inflight = max(fw.max_inflight, fw.inflight)
                done = False
                try:
                    if not doc.get("stream"):
                        time.sleep(fw.delay * fw.chunks)
                        with fw.lock:
                            fw.inflight -= 1
                            fw.served += 1
                        done = True
                        self.reply(200, {"choices": [{"message": {"content": "w%d" % fw.idx}, "finish_reason": "stop"}]})
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    for i in range(fw.chunks):
                        ev = {"choices": [{"delta": {"content": "w%d-%d " % (fw.idx, i)}, "finish_reason": None}]}
                        self.wfile.write(b"data: " + json.dumps(ev).encode() + b"\n\n")
                        self.wfile.flush()
                        time.sleep(fw.delay)
                    with fw.lock:  # free before the final bytes: the router dispatches again only after EOF
                        fw.inflight -= 1
                        fw.served += 1
                    done = True
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except OSError:
                    with fw.lock:
                        fw.cancelled += 1
                finally:
                    if not done:
                        with fw.lock:
                            fw.inflight -= 1

        self.httpd = swarm_router._Server(("127.0.0.1", 0), H)  # quiet about clients that reset keep-alive sockets
        self.url = "http://127.0.0.1:%d" % self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def router_for(urls, *extra):
    args = swarm_router.parse_args(["--listen", "127.0.0.1:0", "--health-interval", "0.1", "--affinity-wait", "0.05"] +
                                   [x for u in urls for x in ("--worker", u)] + list(extra))
    return swarm_router.Router(args).start()


def chat(port, msgs, stream=True, timeout=30):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request("POST", "/v1/chat/completions", body=json.dumps({"messages": msgs, "stream": stream}),
              headers={"Content-Type": "application/json"})
    r = c.getresponse()
    body = r.read()
    c.close()
    text = ""
    if stream and r.status == 200:
        for line in body.split(b"\n"):
            if line.startswith(b"data: ") and line != b"data: [DONE]":
                for ch in json.loads(line[6:]).get("choices", []):
                    text += ch.get("delta", {}).get("content") or ""
    return r.status, r.getheader("X-Swarm-Worker"), text, body, r.getheader("Retry-After")


class SwarmRouterTest(unittest.TestCase):
    def setUp(self):
        self.workers = [FakeWorker(i) for i in range(3)]
        self.router = None

    def tearDown(self):
        if self.router:
            self.router.stop()
        for w in self.workers:
            try:
                w.stop()
            except OSError:
                pass

    def parallel(self, n, fn):
        out = [None] * n
        ts = [threading.Thread(target=lambda i=i: out.__setitem__(i, fn(i))) for i in range(n)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        return out

    def test_parallel_never_overlaps_a_worker(self):
        self.router = router_for([w.url for w in self.workers])
        t0 = time.time()
        res = self.parallel(9, lambda i: chat(self.router.port, [{"role": "user", "content": "task %d" % i}]))
        dt = time.time() - t0
        for st, wid, text, _, _ in res:
            self.assertEqual(st, 200)
            self.assertEqual(text, "".join("w%s-%d " % (wid, k) for k in range(5)))
        self.assertEqual([w.max_inflight for w in self.workers], [1, 1, 1])
        self.assertEqual(sum(w.served for w in self.workers), 9)
        self.assertTrue(all(w.served >= 1 for w in self.workers))
        self.assertGreater(dt, 3 * 5 * 0.03 * 0.9)  # 9 jobs over 3 single-slot workers: at least 3 rounds

    def test_conversation_affinity(self):
        self.router = router_for([w.url for w in self.workers])
        convo = [{"role": "system", "content": "agent A"}, {"role": "user", "content": "step 1"}]
        first = chat(self.router.port, convo)[1]
        for k in range(2, 6):
            convo = convo + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "step %d" % k}]
            # prefix for routing = everything but the last message; keep it fixed to model one turn's retries
            same = chat(self.router.port, convo[:2] + [{"role": "user", "content": "retry %d" % k}])[1]
            self.assertEqual(same, first)
        spread = {chat(self.router.port, [{"role": "system", "content": "agent %d" % i}, {"role": "user", "content": "x"}])[1]
                  for i in range(12)}
        self.assertGreaterEqual(len(spread), 2)

    def test_client_disconnect_cancels_upstream(self):
        self.workers[0].chunks = self.workers[1].chunks = self.workers[2].chunks = 40
        self.router = router_for([w.url for w in self.workers])
        s = socket.create_connection(("127.0.0.1", self.router.port))
        body = json.dumps({"messages": [{"role": "user", "content": "long"}], "stream": True}).encode()
        s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: " +
                  str(len(body)).encode() + b"\r\n\r\n" + body)
        got = b""
        while b"data: " not in got:
            got += s.recv(4096)
        s.close()
        deadline = time.time() + 3
        while time.time() < deadline and sum(w.cancelled for w in self.workers) == 0:
            time.sleep(0.02)
        self.assertEqual(sum(w.cancelled for w in self.workers), 1)
        deadline = time.time() + 3
        while time.time() < deadline and any(w["busy"] for w in self.router.pool.snapshot()[0]):
            time.sleep(0.02)
        self.assertFalse(any(w["busy"] for w in self.router.pool.snapshot()[0]))

    def test_failover_and_health(self):
        self.router = router_for([w.url for w in self.workers])
        self.workers[1].stop()
        res = self.parallel(6, lambda i: chat(self.router.port, [{"role": "user", "content": "f%d" % i}], stream=bool(i % 2)))
        self.assertTrue(all(r[0] == 200 for r in res), res)
        self.assertNotIn("1", {r[1] for r in res})
        c = http.client.HTTPConnection("127.0.0.1", self.router.port, timeout=5)
        c.request("GET", "/health")
        doc = json.loads(c.getresponse().read())
        self.assertEqual(doc["status"], "degraded")
        self.assertFalse(doc["workers"][1]["healthy"])
        c.request("GET", "/v1/models")
        r = c.getresponse()
        self.assertEqual(r.status, 200)
        self.assertIn(json.loads(r.read())["data"][0]["id"], ("fake-0", "fake-2"))
        c.close()

    def test_queue_full_returns_503(self):
        for w in self.workers:
            w.delay, w.chunks = 0.1, 8
        self.router = router_for([w.url for w in self.workers], "--max-queue", "2")
        out = [None] * 9

        def go(i):
            out[i] = chat(self.router.port, [{"role": "user", "content": "q%d" % i}])

        first = [threading.Thread(target=go, args=(i,)) for i in range(3)]
        for t in first:
            t.start()
        deadline = time.time() + 5  # all three workers busy before the burst
        while time.time() < deadline and sum(w.inflight for w in self.workers) < 3:
            time.sleep(0.01)
        burst = [threading.Thread(target=go, args=(i,)) for i in range(3, 9)]
        for t in burst:
            t.start()
        for t in first + burst:
            t.join()
        codes = sorted(r[0] for r in out)
        self.assertEqual(codes.count(200), 5, codes)   # 3 running + 2 queued
        self.assertEqual(codes.count(503), 4, codes)
        self.assertTrue(all(r[4] == "1" for r in out if r[0] == 503))
        self.assertEqual([w.max_inflight for w in self.workers], [1, 1, 1])

    def test_bad_requests(self):
        self.router = router_for([w.url for w in self.workers])
        c = http.client.HTTPConnection("127.0.0.1", self.router.port, timeout=5)
        c.request("POST", "/v1/chat/completions", body=b"{nope", headers={"Content-Type": "application/json"})
        self.assertEqual(c.getresponse().status, 400)
        c.close()
        c = http.client.HTTPConnection("127.0.0.1", self.router.port, timeout=5)
        c.request("GET", "/v1/elsewhere")
        r = c.getresponse()
        r.read()
        self.assertEqual(r.status, 404)
        c.close()

    @unittest.skipUnless(os.environ.get("VJ_MOCK_SERVE"), "set VJ_MOCK_SERVE to the vj_mock_serve binary")
    def test_three_real_mock_servers(self):
        procs, urls = [], []
        try:
            for _ in range(3):
                p = subprocess.Popen([os.environ["VJ_MOCK_SERVE"], "--port", "0"], stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True)
                procs.append(p)
                urls.append("http://" + p.stdout.readline().split()[-1])
            self.router = router_for(urls)
            res = self.parallel(12, lambda i: chat(self.router.port, [{"role": "user", "content": "agent %d" % i}]))
            self.assertTrue(all(r[0] == 200 and r[2] == "Hello from VegaJet mock engine on CPU." for r in res), res)
            self.assertGreaterEqual(len({r[1] for r in res}), 2)
        finally:
            for p in procs:
                p.terminate()
                _, err = p.communicate(timeout=10)
                self.assertEqual(p.returncode, 0, err)
                self.assertNotIn("runtime error", err)
                self.assertNotIn("Sanitizer", err)


if __name__ == "__main__":
    unittest.main()
