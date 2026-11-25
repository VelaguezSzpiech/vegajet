# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""vegajet.bench.v1 round trip: test_bench_json (C++, core/bench_json) writes scorecards from a bench log printed with
the engine's own format strings; this test feeds them to tools/regression_gate.py (a healthy candidate passes, a
degraded one fails on exactly the injected regressions) and to tools/scorecard.py (the README cells and the filled
README from the JSON equal those from the log itself; a merged W4A8 + dot8 scorecard reports W4A8 prefill).

Environment: BENCH_JSON_EMIT = path of the test_bench_json binary."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EMIT = os.environ.get("BENCH_JSON_EMIT")


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gate(*args):
    p = subprocess.run([sys.executable, str(ROOT / "tools/regression_gate.py"), *map(str, args)], capture_output=True,
                       text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    return p.returncode, p.stdout + p.stderr


@unittest.skipUnless(EMIT and Path(EMIT).exists(), "BENCH_JSON_EMIT not set")
class RoundTrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = Path(cls.tmp.name)
        subprocess.run([EMIT, str(cls.d)], check=True, capture_output=True)
        sys.dont_write_bytecode = True
        cls.sc = load_tool("scorecard")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_gate_healthy_candidate(self):
        rc, out = gate(self.d / "base.json", self.d / "cand_ok.json")
        self.assertEqual(rc, 0, out)
        self.assertIn("GATE: PASS", out)

    def test_gate_degraded_candidate(self):
        rc, out = gate(self.d / "base.json", self.d / "cand_bad.json", "--json")
        self.assertEqual(rc, 1)
        fails = {(c["kind"], c["metric"]) for c in json.loads(out)["checks"] if c["status"] == "FAIL"}
        kinds = {k for k, _ in fails}
        self.assertEqual(kinds, {"throughput", "kld", "parity", "tmem", "ue", "spill"})
        self.assertIn(("throughput", "throughput.raw_m1_tok_s"), fails)
        self.assertIn(("parity", "parity.code_vs_tier1"), fails)
        self.assertIn(("ue", "ras.gpu1.umc.ue"), fails)
        self.assertIn(("kld", "kld.dot8_prefill_kld"), fails)
        self.assertNotIn(("de", "ras.gpu1.umc.de"), fails)

    def test_gate_self_compare(self):
        rc, out = gate(self.d / "base.json", self.d / "base.json")
        self.assertEqual(rc, 0, out)

    def test_scorecard_json_equals_log(self):
        log = (self.d / "bench.log").read_text().splitlines()
        from_log = self.sc.parse(log)
        from_json = self.sc.from_bench_json(json.loads((self.d / "base.json").read_text()))
        self.assertEqual(self.sc.cells(from_log), self.sc.cells(from_json))
        self.assertTrue(all(v is not None for v in self.sc.cells(from_json).values()))
        readme = "| Metric | 2x MI50 |\n| :--- | :--- |\n" + "".join(
            "| %s | %s |\n" % (k, self.sc.PLACEHOLDER) for k in self.sc.cells(from_log))
        a = self.sc.fill_readme(readme, from_log, "abc1234", "--tp 2")
        b = self.sc.fill_readme(readme, from_json, "abc1234", "--tp 2")
        self.assertEqual(a, b)
        self.assertEqual(a[2], [])  # nothing left unfilled
        # the DFlash2+Hyb line is now parsed from logs too (previously dropped by the spec regex)
        self.assertIn("DFlash2+Hyb code", [s["name"] for s in from_log["spec"]])

    def test_scorecard_cli_accepts_json(self):
        p = subprocess.run([sys.executable, str(ROOT / "tools/scorecard.py"), str(self.d / "base.json"), "--json"],
                           capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        self.assertEqual(p.returncode, 0, p.stderr)
        r = json.loads(p.stdout)
        self.assertAlmostEqual(r["raw"]["tps"], 33.1)
        self.assertEqual(r["parity"]["match"], 32)
        self.assertEqual(r["prefill"]["2048"]["m"], 512)

    def test_merged_scorecard_prefers_w4a8_prefill(self):
        doc = json.loads((self.d / "w4a8_then_dot8.json").read_text())
        self.assertIsNotNone(doc["throughput"]["pp2048_dot8_tok_s"])
        self.assertIsNotNone(doc["kld"]["dot8_prefill_kld"])
        r = self.sc.from_bench_json(doc)
        self.assertEqual(r["prefill"][2048]["tps"], doc["throughput"]["pp2048_w4a8_tok_s"])
        rc, out = gate(self.d / "base.json", self.d / "w4a8_then_dot8.json")
        self.assertIn("kld.dot8_prefill_kld", out)


if __name__ == "__main__":
    unittest.main()
