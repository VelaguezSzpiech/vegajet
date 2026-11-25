# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Unit tests for tools/dflash_parity_diag.py (standard library only, synthetic logs and dumps)."""
import contextlib
import io
import json
import math
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

import dflash_parity_diag as dp  # noqa: E402


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = dp.main(argv)
    return rc, out.getvalue(), err.getvalue()


class ParityDiag(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="vj_dfdiag_test_")
        self.d = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def path(self, name):
        return os.path.join(self.d, name)

    def write_logs(self, engine_bad_from=None, n=12, engine_slot=2):
        """Reference accepts 5 of 7 per cycle; the engine copies it until cycle engine_bad_from."""
        with open(self.path("e.log"), "w") as fe, open(self.path("r.log"), "w") as fr:
            for c in range(n):
                pos = 30 + 6 * c
                tgt = [pos * 7 + i for i in range(8)]
                rd = tgt[:7]
                rd[5] += 1
                ed = list(rd)
                if engine_bad_from is not None and c >= engine_bad_from:
                    ed[engine_slot] += 11
                fe.write("  [DFVER pos=%d acc=%d/7] tgt=[%s] dft=[%s]\n" % (
                    pos, dp.accepted_prefix(ed, tgt), ",".join(map(str, tgt[:5])), ",".join(map(str, ed[:5]))))
                fr.write("  p=%4d anchor=%6d acc=%2d draft: %s | target: %s\n" % (
                    pos, 1, dp.accepted_prefix(rd, tgt), " ".join("%d*" % d for d in rd[:5]) + " %d %d" % (rd[5], rd[6]),
                    " ".join(map(str, tgt[:7]))))

    def test_self_test(self):
        rc, out, _ = run(["--self-test"])
        self.assertEqual(rc, 0)
        self.assertIn("PASS", out)

    def test_parse_formats(self):
        self.write_logs()
        kind, e = dp.parse_cycles(self.path("e.log"))
        self.assertEqual(kind, "engine")
        self.assertEqual((e[0].pos, e[0].accepted, e[0].n_draft, len(e[0].draft)), (30, 5, 7, 5))
        kind, r = dp.parse_cycles(self.path("r.log"))
        self.assertEqual(kind, "reference")
        self.assertEqual((r[0].pos, r[0].accepted, len(r[0].draft), len(r[0].target)), (30, 5, 7, 7))
        with open(self.path("j.jsonl"), "w") as fh:
            fh.write(json.dumps({"pos": 3, "draft": [1, 2, 3], "target": [1, 2, 4]}) + "\n")
        kind, j = dp.parse_cycles(self.path("j.jsonl"))
        self.assertEqual((kind, j[0].accepted, j[0].n_draft), ("jsonl", 2, 3))

    def test_matching_engine_not_flagged(self):
        self.write_logs()
        rc, out, _ = run(["--engine", self.path("e.log"), "--reference", self.path("r.log")])
        self.assertEqual(rc, 0, out)
        self.assertIn("mean_accepted_drafts=5.000 over 12 full cycles -> PASS", out)
        self.assertIn("engine drafts equal the reference on all 12 aligned cycles", out)
        self.assertIn("not flagged", out)

    def test_divergent_engine_flagged_with_position_and_slot(self):
        self.write_logs(engine_bad_from=3, engine_slot=2)
        rc, out, _ = run(["--engine", self.path("e.log"), "--reference", self.path("r.log"), "--json"])
        self.assertEqual(rc, 1)
        rep = json.loads(out)
        self.assertTrue(rep["flagged"])
        self.assertAlmostEqual(rep["engine"]["mean_accepted"], (3 * 5 + 9 * 2) / 12.0)
        self.assertEqual(rep["engine"]["gate"], "ABORT")
        self.assertEqual(rep["engine"]["hist"][2], 9)
        self.assertEqual(rep["first_divergence"], {"pos": 48, "slot": 2, "engine_accepted": 2, "reference_accepted": 5})
        self.assertEqual(rep["first_divergence_index"], 3)

    def test_marginal_gate(self):
        e = [dp.Cycle(p, [1] * 7, [1] * 3 + [2] * 4) for p in range(4)]  # accepted 3 each
        e.append(dp.Cycle(9, [1] * 7, [1] * 7))                          # accepted 7 -> mean 3.8
        rep = dp.diagnose(e, e, 8)
        self.assertAlmostEqual(rep["engine"]["mean_accepted"], 3.8)
        self.assertEqual(rep["engine"]["gate"], "MARGINAL")
        self.assertFalse(rep["flagged"])

    def test_layer_localisation_3d_and_4d(self):
        self.write_logs(engine_bad_from=2)
        L, B, D = 3, 8, 5
        base = [math.cos(i * 0.11) for i in range(L * B * D)]
        bad = list(base)
        for d in range(D):
            bad[(1 * B + 2) * D + d] += 0.5
        dp.write_npy(self.path("e3.npy"), (L, B, D), bad)
        dp.write_npy(self.path("r3.npy"), (L, B, D), base)
        rc, out, _ = run(["--engine", self.path("e.log"), "--reference", self.path("r.log"),
                          "--engine-layers", self.path("e3.npy"), "--reference-layers", self.path("r3.npy")])
        self.assertEqual(rc, 1)
        self.assertIn("first diverging drafter layer: 1 (block slot 2", out)
        self.assertIn("at aligned cycle 2", out)
        # 4-D: cycles x layers x B x D; the divergent cycle (index 2) carries the error.
        dp.write_npy(self.path("e4.npy"), (3, L, B, D), base + base + bad)
        dp.write_npy(self.path("r4.npy"), (3, L, B, D), base + base + base)
        rc, out, _ = run(["--engine", self.path("e.log"), "--reference", self.path("r.log"), "--json",
                          "--engine-layers", self.path("e4.npy"), "--reference-layers", self.path("r4.npy")])
        rep = json.loads(out)
        self.assertEqual((rep["layer_divergence"]["layer"], rep["layer_divergence"]["slot"]), (1, 2))
        rc, out, _ = run(["--engine", self.path("e.log"), "--reference", self.path("r.log"), "--json", "--cycle-index", "0",
                          "--engine-layers", self.path("e4.npy"), "--reference-layers", self.path("r4.npy")])
        self.assertEqual(json.loads(out)["layer_divergence"]["layer"], -1)

    def test_errors(self):
        self.write_logs()
        self.assertEqual(run([])[0], 2)
        self.assertEqual(run(["--engine", self.path("e.log"), "--reference", self.path("missing.log")])[0], 2)
        self.assertEqual(run(["--engine", self.path("e.log"), "--reference", self.path("r.log"),
                              "--engine-layers", self.path("x.npy")])[0], 2)
        dp.write_npy(self.path("a.npy"), (2, 8, 4), [0.0] * 64)
        dp.write_npy(self.path("b.npy"), (2, 8, 5), [0.0] * 80)
        rc, _, err = run(["--engine", self.path("e.log"), "--reference", self.path("r.log"),
                          "--engine-layers", self.path("a.npy"), "--reference-layers", self.path("b.npy")])
        self.assertEqual(rc, 2)
        self.assertIn("differ in shape", err)
        rc, out, _ = run(["--engine", self.path("e.log"), "--reference", self.path("r.log"), "--block", "4"])
        self.assertIn("defined for B=8", out)


if __name__ == "__main__":
    unittest.main()
