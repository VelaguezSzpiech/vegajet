# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""End-to-end tests of the `vegajet` binary (CPU-only build): help / version / exit codes, `plan` on a miniature
qwen35moe GGUF, `serve <model-file>` failing cleanly with exit 3 when no engine exists (never opening a device),
`serve --mock` single-sequence and with 4 scheduler slots (identical replies, tools accepted, SIGTERM -> 0, the
HEAD API stress harness against the batched server), the CLI-wide --gpu option and VJ_GPU_DEVICES (parsing errors,
selection vs a synthetic KFD topology, --tp mismatch, `--gpu list`), and `plan --json-out` scorecards
going through tools/regression_gate.py and tools/scorecard.py.

Environment: VEGAJET_BIN (the vegajet binary), VJ_MINI_GGUF (the vj_mini_gguf fixture writer)."""
import http.client
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = os.environ.get("VEGAJET_BIN")
MINI = os.environ.get("VJ_MINI_GGUF")


def run(*args, env=None, timeout=60):
    e = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    e.pop("VJ_GPU_DEVICES", None)
    e.update(env or {})
    p = subprocess.run([BIN, *map(str, args)], capture_output=True, text=True, env=e, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


class Server:
    """Starts `vegajet serve ... --port 0`, parses the port from the listening line."""

    def __init__(self, *args, env=None):
        e = dict(os.environ)
        e.pop("VJ_GPU_DEVICES", None)
        e.update(env or {})
        self.p = subprocess.Popen([BIN, "serve", *args, "--port", "0"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, env=e)
        line = self.p.stdout.readline()
        m = re.search(r"listening on http://([\d.]+):(\d+) \((.*)\)", line)
        if not m:
            self.p.kill()
            raise AssertionError("no listening line: %r / %s" % (line, self.p.stderr.read()))
        self.host, self.port, self.info = m.group(1), int(m.group(2)), m.group(3)

    def req(self, method, path, body=None):
        c = http.client.HTTPConnection(self.host, self.port, timeout=30)
        c.request(method, path, body=json.dumps(body) if body is not None else None,
                  headers={"Content-Type": "application/json"} if body is not None else {})
        r = c.getresponse()
        data = r.read().decode()
        c.close()
        return r.status, data

    def chat(self, content, stream=False, **extra):
        body = {"messages": [{"role": "user", "content": content}], "stream": stream, **extra}
        st, data = self.req("POST", "/v1/chat/completions", body)
        assert st == 200, (st, data)
        if not stream:
            return json.loads(data)["choices"][0]["message"]["content"]
        out = ""
        for ev in data.split("\n\n"):
            if ev.startswith("data: {"):
                ch = json.loads(ev[6:])["choices"]
                if ch and ch[0]["delta"].get("content"):
                    out += ch[0]["delta"]["content"]
        return out

    def stop(self):
        self.p.send_signal(signal.SIGTERM)
        try:
            return self.p.wait(timeout=20)
        finally:
            self.p.stdout.close()
            self.p.stderr.close()


def fake_topology(root, n):
    base = Path(root) / "sys/class/kfd/kfd/topology/nodes"
    (base / "0").mkdir(parents=True)
    (base / "0/properties").write_text("simd_count 0\ngfx_target_version 0\n")
    for i in range(n):
        d = base / str(i + 1)
        (d / "mem_banks/0").mkdir(parents=True)
        (d / "properties").write_text("simd_count 240\ngfx_target_version 90006\ndrm_render_minor %d\nlocation_id %d\n"
                                      % (128 + i, 256 * (i + 1)))
        (d / "gpu_id").write_text("%d\n" % (1000 + i))
        (d / "mem_banks/0/properties").write_text("size_in_bytes 34342961152\n")


MOCK_REPLY = "Hello from VegaJet mock engine on CPU."


@unittest.skipUnless(BIN and Path(BIN).exists(), "VEGAJET_BIN not set")
class Cli(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = Path(cls.tmp.name)
        cls.model = cls.d / "mini-ornith-model"
        if MINI:
            subprocess.run([MINI, str(cls.model)], check=True, capture_output=True)
        cls.topo = cls.d / "root"
        fake_topology(cls.topo, 2)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def need_gguf(self):
        if not self.model.exists():
            self.skipTest("VJ_MINI_GGUF not set")

    # ---- basics
    def test_help_version_exit_codes(self):
        rc, out, _ = run("--help")
        self.assertEqual(rc, 0)
        for sub in ("inspect", "plan", "bench", "run", "doctor", "soak", "serve", "--gpu", "VJ_GPU_DEVICES", "Exit codes"):
            self.assertIn(sub, out)
        self.assertEqual(run("--version")[1].strip(), "vegajet 0.1.1")
        self.assertEqual(run()[0], 1)
        self.assertEqual(run("frobnicate")[0], 2)
        for sub in ("plan", "serve", "doctor", "inspect"):
            self.assertEqual(run(sub, "--help")[0], 0, sub)

    def test_plan(self):
        self.need_gguf()
        rc, out, err = run("plan", self.model, "--tp", "2")
        self.assertEqual(rc, 0, err)
        self.assertIn("plugin=ornith arch=qwen35moe TP=2", out)
        self.assertIn("fits", out)
        self.assertEqual(run("plan")[0], 2)

    # ---- serve with a model file: no engine on a CPU-only build -> exit 3, before any device is touched
    def test_serve_model_without_engine(self):
        self.need_gguf()
        rc, _, err = run("serve", self.model, "--port", "0")
        self.assertEqual(rc, 3)
        self.assertTrue(
            "no engine for architecture 'qwen35moe'" in err
            or "no streaming engine is registered for architecture 'qwen35moe'" in err
            or "failed to start" in err,
            err,
        )
        self.assertEqual(run("serve", self.d / "missing-model")[0], 1)
        self.assertEqual(run("serve", self.model, "--mock")[0], 2)

    # ---- --gpu / VJ_GPU_DEVICES
    def test_gpu_selection(self):
        self.need_gguf()
        env = {"VJ_SYSFS_ROOT": str(self.topo)}
        rc, out, _ = run("serve", "--gpu", "list", env=env)
        self.assertEqual(rc, 0)
        self.assertIn("selection: all (default)", out)
        self.assertEqual(len(re.findall(r"^\s+\d+\s+\*", out, re.M)), 2)
        rc, out, _ = run("plan", "--gpu", "list", env=dict(env, VJ_GPU_DEVICES="1"))
        self.assertIn("selection: 1 (VJ_GPU_DEVICES)", out)
        for bad in (["--gpu", "9-1"], ["--gpu", "a"], ["--gpu", "0", "--gpu", "1"], ["--gpu"]):
            self.assertEqual(run("serve", "--mock", *bad, env=env)[0], 2, bad)
        self.assertEqual(run("serve", "--mock", env=dict(env, VJ_GPU_DEVICES="x"))[0], 2)
        rc, _, err = run("serve", self.model, "--gpu", "5", env=env)
        self.assertEqual(rc, 2)
        self.assertIn("GPU 5 selected (--gpu) but only 2 gfx906 GPU(s) found", err)
        rc, _, err = run("serve", self.model, "--gpu", "1", "--tp", "2", env=env)
        self.assertEqual(rc, 2)
        self.assertIn("does not match the 1 GPU(s) selected", err)
        rc, _, err = run("serve", self.model, "--gpu", "1", env=env)  # valid selection: then no engine -> 3
        self.assertEqual(rc, 3)
        self.assertEqual(run("doctor", "--gpu", "zz")[0], 2)

    # ---- serve --mock, single sequence and batched
    def test_serve_mock_single(self):
        s = Server("--mock", "--gpu", "0")
        try:
            self.assertIn("engine=mock,", s.info)
            self.assertIn("gpus=0", s.info)
            st, data = s.req("GET", "/health")
            self.assertEqual(st, 200)
            self.assertEqual(json.loads(data)["status"], "ok")
            self.assertEqual(s.chat("hi"), MOCK_REPLY)
            self.assertEqual(s.chat("hi", stream=True), MOCK_REPLY)
            tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
            self.assertEqual(s.chat("hi", tools=tools), MOCK_REPLY)  # tools accepted; the mock answers in prose
            st, _ = s.req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "tool_choice": "bogus"})
            self.assertEqual(st, 400)
        finally:
            self.assertEqual(s.stop(), 0)

    def test_serve_mock_scheduler(self):
        s = Server("--mock", "--slots", "4")
        try:
            self.assertIn("engine=mock+scheduler", s.info)
            self.assertIn("slots=4", s.info)
            results, errors = [], []

            def one(i):
                try:
                    results.append(s.chat("request %d" % i, stream=bool(i % 2)))
                except Exception as e:  # noqa: BLE001 - surfaced below
                    errors.append(repr(e))
            ts = [threading.Thread(target=one, args=(i,)) for i in range(12)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            self.assertEqual(errors, [])
            self.assertEqual(results, [MOCK_REPLY] * 12)  # identical to the single-sequence mock
            self.assertEqual(s.chat("x", max_tokens=3), "Hello from VegaJet")
            stress = subprocess.run([sys.executable, str(ROOT / "tools/api_stress_test.py"), "--url",
                                     "http://%s:%d" % (s.host, s.port), "--clients", "16", "--rounds", "2"],
                                    capture_output=True, text=True, timeout=300,
                                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
            self.assertEqual(stress.returncode, 0, stress.stdout[-3000:] + stress.stderr[-2000:])
        finally:
            self.assertEqual(s.stop(), 0)

    # ---- bench JSON from the CLI (bench_json_schema.patch)
    def test_plan_and_bench_json(self):
        self.need_gguf()
        out = self.d / "score.json"
        rc, stdout, err = run("plan", self.model, "--tp", "2", "--json-out", out)
        if rc == 2 and "unknown option --json-out" in err:
            self.skipTest("bench_json_schema.patch not applied")
        self.assertEqual(rc, 0, err)
        self.assertIn("bench json: ", stdout)
        doc = json.loads(out.read_text())
        self.assertEqual(doc["schema"], "vegajet.bench.v1")
        self.assertEqual(doc["meta"]["command"], "plan")
        self.assertFalse(doc["meta"]["executed"])
        self.assertEqual(doc["meta"]["model"], "mini-ornith-model")
        self.assertEqual(doc["meta"]["plan_status"], "fits")
        self.assertNotIn(str(self.d), out.read_text())  # file names only, never directories
        self.assertIsNone(doc["throughput"]["raw_m1_tok_s"])
        gate = subprocess.run([sys.executable, str(ROOT / "tools/regression_gate.py"), str(out), str(out)],
                              capture_output=True, text=True)
        self.assertEqual(gate.returncode, 2, gate.stdout + gate.stderr)  # a plan-only scorecard has nothing to gate
        sc = subprocess.run([sys.executable, str(ROOT / "tools/scorecard.py"), str(out), "--json"], capture_output=True, text=True)
        self.assertEqual(sc.returncode, 0, sc.stderr)
        self.assertIsNone(json.loads(sc.stdout)["raw"])
        # A second plan into the same file merges (same model + commit). `vegajet bench` itself is never run here:
        # the suite stays on plan (bench --dry-run), which returns before any runtime is constructed.
        rc, stdout, _ = run("plan", self.model, "--json-out", out, "--prefill-dot8")
        self.assertEqual(rc, 0)
        self.assertIn("(merged)", stdout)
        self.assertEqual(json.loads(out.read_text())["meta"]["prefill_path"], "w4a8+dot8")
        self.assertEqual(run("plan", self.model, "--json-out")[0], 2)  # no value: usage error, nothing written
        self.assertEqual(run("plan", self.model, "--json-out", "--tp", "2")[0], 2)
        self.assertEqual(sorted(p.name for p in Path.cwd().glob("-*")), [])


if __name__ == "__main__":
    unittest.main()
