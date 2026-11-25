# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""tools/regression_gate.py through its CLI: exit codes, JSON and text scorecards, every gate rule at and around
its threshold, missing / new metrics, unusable input, --json output, and the release-readiness marker check
(including that release/docs/OPERATIONS.md itself carries no unfilled marker)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GATE = ROOT / "tools" / "regression_gate.py"
MARKER = "[FILL FROM FINAL SCORECARD]"

BASE = {
    "commit": "abc1234",
    "throughput": {"decode_m1_tok_s": [50.0, 51.0, 52.0], "prefill_2048_tok_s": 1430.0, "mtp_tok_s": 70.0},
    "kld": 0.0009,
    "parity": {"code": "32/32", "spec": {"match": 129, "total": 129}},
    "first_diff": -1,
    "spill_bytes": 0,
    "telemetry": {"tmem_c": [52.0, 54.5], "ue": 0, "de": 0},
}


def run(*args):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, str(GATE), *map(str, args)], capture_output=True, text=True, env=env, timeout=60)
    return p.returncode, p.stdout + p.stderr


class Gate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def card(self, name, obj):
        p = self.d / name
        p.write_text(json.dumps(obj) if not isinstance(obj, str) else obj)
        return p

    def variant(self, **changes):
        c = json.loads(json.dumps(BASE))
        for path, v in changes.items():
            node = c
            keys = path.split("__")
            for k in keys[:-1]:
                node = node[k]
            if v is None:
                del node[keys[-1]]
            else:
                node[keys[-1]] = v
        return c

    def gate(self, cand, *extra):
        return run(self.card("base.json", BASE), self.card("cand.json", cand), *extra)

    def test_identical_passes(self):
        rc, out = self.gate(BASE)
        self.assertEqual(rc, 0, out)
        self.assertIn("GATE: PASS", out)

    def test_throughput_budget(self):
        self.assertEqual(self.gate(self.variant(throughput__decode_m1_tok_s=50.0))[0], 0)    # -1.96 % vs median 51
        rc, out = self.gate(self.variant(throughput__decode_m1_tok_s=49.9))                  # -2.16 %
        self.assertEqual(rc, 1, out)
        self.assertIn("decode_m1_tok_s", out)
        self.assertEqual(self.gate(self.variant(throughput__prefill_2048_tok_s=2000.0))[0], 0)  # faster is fine
        self.assertEqual(self.gate(self.variant(throughput__decode_m1_tok_s=49.9), "--max-drop", "5")[0], 0)

    def test_missing_and_new_throughput(self):
        rc, out = self.gate(self.variant(throughput__mtp_tok_s=None))
        self.assertEqual(rc, 1)
        self.assertIn("missing from candidate", out)
        rc, out = self.gate(self.variant(throughput__new_tok_s=10.0))
        self.assertEqual(rc, 0)
        self.assertIn("INFO", out)

    def test_absolute_gates(self):
        cases = [("kld", 0.0451, 1), ("kld", 0.045, 0), ("spill_bytes", 4, 1), ("telemetry__tmem_c", [54.0, 65.0], 1),
                 ("telemetry__tmem_c", 64.9, 0), ("telemetry__ue", 1, 1), ("telemetry__de", 2, 1),
                 ("parity__code", "31/32", 1), ("parity__spec", {"match": 128, "total": 129}, 1), ("first_diff", 17, 1)]
        for key, value, want in cases:
            with self.subTest(key=key, value=value):
                rc, out = self.gate(self.variant(**{key: value}))
                self.assertEqual(rc, want, out)

    def test_json_output(self):
        rc, out = self.gate(self.variant(kld=0.1), "--json")
        self.assertEqual(rc, 1)
        doc = json.loads(out)
        self.assertFalse(doc["pass"])
        self.assertTrue(any(c["kind"] == "kld" and c["status"] == "FAIL" for c in doc["checks"]))

    def test_text_scorecards(self):
        base = self.card("base.txt", "[BENCH tg128 raw M=1] 50.0 tok/s (20.0 ms/tok)\n[BENCH pp2048 M=512] 1400 tok/s\n"
                                     "[PARITY code vs Tier-1 llama.cpp] 32/32\n")
        good = self.card("good.txt", "[BENCH tg128 raw M=1] 49.5 tok/s (20.2 ms/tok)\n[BENCH pp2048 M=512] 1410 tok/s\n"
                                     "[PARITY code vs Tier-1 llama.cpp] 32/32\n[TELEM] TMem=55.0C UE 0 DE 0\nKLD 0.0011\n")
        bad = self.card("bad.txt", "[BENCH tg128 raw M=1] 49.5 tok/s\n[BENCH pp2048 M=512] 1410 tok/s\n"
                                   "[PARITY code vs Tier-1 llama.cpp] 31/32\n[TELEM] TMem=66.0C UE 0 DE 3\nspill: 8\n")
        self.assertEqual(run(base, good)[0], 0)
        rc, out = run(base, bad)
        self.assertEqual(rc, 1)
        for word in ("parity", "tmem", "de", "spill"):
            self.assertRegex(out, rf"FAIL\s+{word}\b")

    def test_mixed_json_baseline_text_candidate(self):
        base = self.card("b.json", {"decode_tok_s": 50.0})
        cand = self.card("c.txt", "decode_tok_s: 48.0 tok/s\n")
        self.assertEqual(run(base, cand)[0], 1)

    def test_unusable_input(self):
        self.assertEqual(run(self.d / "nope.json", self.card("c.json", BASE))[0], 2)
        self.assertEqual(run(self.card("b.json", BASE), self.card("c.txt", "hello world\n"))[0], 2)
        self.assertEqual(run(self.card("b.json", BASE), self.card("c.json", "{not json"))[0], 2)

    def test_release_check(self):
        ready = self.card("ready.md", "# Release\n| Decode | 50.1 tok/s |\n")
        notready = self.card("notready.md", f"| Decode | {MARKER} |\n| Prefill | {MARKER} |\n")
        self.assertEqual(run("--release-check", ready)[0], 0)
        rc, out = run("--release-check", ready, notready)
        self.assertEqual(rc, 1)
        self.assertIn("notready.md:2", out)
        self.assertIn("2 unfilled marker(s)", out)
        self.assertEqual(run("--release-check", self.d / "missing.md")[0], 2)
        rc, out = run("--release-check", notready, "--json")
        self.assertFalse(json.loads(out)["ready"])

    def test_operations_doc_has_no_markers(self):
        ops = ROOT / "release" / "docs" / "OPERATIONS.md"
        self.assertTrue(ops.is_file())
        rc, out = run("--release-check", ops)
        self.assertEqual(rc, 0, out)

    def test_self_test(self):
        rc, out = run("--self-test")
        self.assertEqual(rc, 0, out)


if __name__ == "__main__":
    unittest.main()
