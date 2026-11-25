# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Unit tests for tools/dflash_diff.py and tools/scorecard.py (standard library only, synthetic data only).

Run from the sidecar root:  python3 -m unittest discover -s tests -p 'test_tools*.py'
"""
import contextlib
import io
import math
import os
import random
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

import dflash_diff as dd  # noqa: E402
import scorecard as sc  # noqa: E402


def quiet(fn, *a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(*a)
    return rc, buf.getvalue()


class DflashDiffTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        rng = random.Random(3)
        self.T, self.S, self.D = 3, 49, 8
        self.states = [[[rng.gauss(0, 1) for _ in range(self.D)] for _ in range(self.S)] for _ in range(self.T)]
        self.ids = [11, 12, 13]

    def tearDown(self):
        self.tmp.cleanup()

    def p(self, name):
        return os.path.join(self.d, name)

    def taps_npy(self, name, offset, perturb_layer=None):
        flat = []
        for t in range(self.T):
            for L in dd.DEFAULT_TAPS:
                v = list(self.states[t][L + offset])
                if L == perturb_layer:
                    v = [x + 0.5 for x in v]
                flat.extend(v)
        dd.write_npy(self.p(name), (self.T, len(dd.DEFAULT_TAPS), self.D), flat)
        return self.p(name)

    def test_vjhs_and_npy_roundtrip(self):
        dd.write_vjhs(self.p("a.vjhs"), self.ids, self.states)
        v = dd.Vjhs(self.p("a.vjhs"))
        self.assertEqual((v.n_tok, v.n_states, v.dim, v.ids), (3, 49, 8, self.ids))
        for a, b in zip(v.state(2, 47), self.states[2][47]):
            self.assertAlmostEqual(a, b, places=5)
        dd.write_npy(self.p("m.npy"), (2, 3), [1, 0, 1, 0, 0, 1], "|b1")
        self.assertEqual(dd.read_npy(self.p("m.npy")), ((2, 3), [1, 0, 1, 0, 0, 1]))
        with open(self.p("bad.vjhs"), "wb") as fh:
            fh.write(b"XXXX" + bytes(16))
        with self.assertRaises(ValueError):
            dd.Vjhs(self.p("bad.vjhs"))
        with open(self.p("a.vjhs"), "rb") as fh:
            data = fh.read()
        with open(self.p("short.vjhs"), "wb") as fh:
            fh.write(data[:-4])
        with self.assertRaises(ValueError):
            dd.Vjhs(self.p("short.vjhs"))

    def test_taps_offset_detection(self):
        dd.write_vjhs(self.p("a.vjhs"), self.ids, self.states)
        rc, out = quiet(dd.main, ["taps", self.p("a.vjhs"), self.taps_npy("t1.npy", 1)])
        self.assertEqual(rc, 0, out)
        self.assertIn("TAPS: MATCH (offset 1", out)
        # Engine tapped the layer INPUT (offset 0): auto mode names the off-by-one; forcing offset 1 fails.
        rc, out = quiet(dd.main, ["taps", self.p("a.vjhs"), self.taps_npy("t0.npy", 0)])
        self.assertIn("TAPS: MATCH (offset 0", out)
        rc, out = quiet(dd.main, ["taps", self.p("a.vjhs"), self.p("t0.npy"), "--offset", "1"])
        self.assertEqual(rc, 1)
        rc, out = quiet(dd.main, ["taps", self.p("a.vjhs"), self.taps_npy("tp.npy", 1, perturb_layer=27)])
        self.assertEqual(rc, 1)
        self.assertIn("MISMATCH", out)
        # Full-state dump vs full-state dump.
        dd.write_vjhs(self.p("b.vjhs"), self.ids, self.states)
        rc, out = quiet(dd.main, ["taps", self.p("a.vjhs"), self.p("b.vjhs"), "--offset", "1"])
        self.assertEqual(rc, 0, out)

    def test_fc_order(self):
        rng = random.Random(5)
        n = len(dd.DEFAULT_TAPS)
        D = self.D
        W = [rng.gauss(0, 0.3) for _ in range(D * D * n)]
        nw = [1.0 + 0.1 * rng.gauss(0, 1) for _ in range(D)]
        dd.write_npy(self.p("fc.npy"), (D, D * n), W)
        dd.write_npy(self.p("nw.npy"), (D,), nw)
        taps = self.taps_npy("t.npy", 1)
        x = []
        for t in range(self.T):
            cat = [v for L in dd.DEFAULT_TAPS for v in dd.rmsnorm(self.states[t][L + 1], nw, 1e-6)]
            x.extend(sum(W[r * D * n + k] * cat[k] for k in range(D * n)) for r in range(D))
        dd.write_npy(self.p("x.npy"), (self.T, D), x)
        rc, out = quiet(dd.main, ["fc", taps, "--fc-weight", self.p("fc.npy"), "--norm-weight", self.p("nw.npy"),
                                  "--drafter-input", self.p("x.npy"), "--rows", "8", "--tokens", "3"])
        self.assertEqual(rc, 0, out)
        self.assertIn("FC: best order fc(cat(norm(tap)))", out)
        # Drafter input produced by norm(fc(cat)) is detected too (exact mode computes all rows).
        x2 = []
        for t in range(self.T):
            cat = [v for L in dd.DEFAULT_TAPS for v in self.states[t][L + 1]]
            full = [sum(W[r * D * n + k] * cat[k] for k in range(D * n)) for r in range(D)]
            x2.extend(dd.rmsnorm(full, nw, 1e-6))
        dd.write_npy(self.p("x2.npy"), (self.T, D), x2)
        for extra in ([], ["--exact"]):
            rc, out = quiet(dd.main, ["fc", taps, "--fc-weight", self.p("fc.npy"), "--norm-weight", self.p("nw.npy"),
                                      "--drafter-input", self.p("x2.npy"), "--rows", "8", "--tokens", "3"] + extra)
            self.assertEqual(rc, 0, out)
            self.assertIn("FC: best order norm(fc(cat))", out)

    def test_mask(self):
        n_ctx, block, W = 20, 4, 8
        good = dd.expected_mask(n_ctx, block, W, "swa", "lt")
        dd.write_npy(self.p("good.npy"), (block, n_ctx + block), [int(v) for row in good for v in row], "|b1")
        rc, out = quiet(dd.main, ["mask", self.p("good.npy"), "--window", str(W)])
        self.assertEqual(rc, 0, out)
        self.assertIn("MASK: MATCH", out)
        off = dd.expected_mask(n_ctx, block, W, "swa", "le")
        dd.write_npy(self.p("off.npy"), (block, n_ctx + block), [int(v) for row in off for v in row], "|b1")
        rc, out = quiet(dd.main, ["mask", self.p("off.npy"), "--window", str(W)])
        self.assertEqual(rc, 1)
        self.assertIn("window off by one", out)
        # Causal (not bidirectional) draft block: block mismatches are reported.
        causal = [[v if j < n_ctx else j - n_ctx <= i for j, v in enumerate(row)] for i, row in enumerate(good)]
        dd.write_npy(self.p("causal.npy"), (block, n_ctx + block), [int(v) for row in causal for v in row], "|b1")
        rc, out = quiet(dd.main, ["mask", self.p("causal.npy"), "--window", str(W)])
        self.assertEqual(rc, 1)
        self.assertIn("'block': 6", out)
        # Additive float mask (0 / -inf) for a full-attention layer.
        full = dd.expected_mask(n_ctx, block, W, "full", "lt")
        dd.write_npy(self.p("f.npy"), (block, n_ctx + block), [0.0 if v else -math.inf for row in full for v in row])
        rc, out = quiet(dd.main, ["mask", self.p("f.npy"), "--kind", "full", "--float-mask"])
        self.assertEqual(rc, 0, out)
        # Production geometry: W=4096 covers positions q-4095 .. q (llama.cpp standard SWA), so query 5000 sees 4095
        # context keys, query 5001 sees 4094, and every block query also sees the whole (bidirectional) block.
        m = dd.expected_mask(5000, 2, 4096, "swa", "lt")
        self.assertEqual((sum(m[0][:5000]), sum(m[1][:5000])), (4095, 4094))
        self.assertEqual((sum(m[0][5000:]), sum(m[1][5000:])), (2, 2))


LOG = """\
[BENCH tg128 raw M=1] 33.10 tok/s (30.21 ms/tok) | eff_HBM2=612.0 GB/s (71.2% of 860.0 GB/s calibrated) | real_read=18.490 GB/tok | disp/tok=410
[PROFILE in-situ M=1] base 30.21 ms/tok | gdn_qkv=1.20 gdn_out=0.80 gqa_qkv=0.50 gqa_prep=0.10 gqa_flash=0.40 gqa_o=0.30 moe_front=12.00 moe_back=8.70 | sum=24.00 ms (rest: AR, conv, recur, embed, LM head, host)
[BENCH tg128 Suffix ] 41.00 tok/s (1.24x vs raw) | lossless=128/128 (first_diff=-1) | hits=40 acc=60/90 | 0 B VRAM
[BENCH tg512 Suffix ] 44.00 tok/s | hits=150 acc=200/300 | 0 B VRAM
[BENCH tg128 MTP K=2] 52.00 tok/s (1.57x vs raw 33.10) | prose lossless=127/128 (first_diff=90) | acc=50/80 (62.5%)
[BENCH tg128 Hybrid ] 47.50 tok/s (1.44x vs raw) | code lossless=128/128 (first_diff=-1) | sfx_hits=30 acc=70/100 (70.0%)
[BENCH tg128 DFlash2 K=8 code ] 45.00 tok/s (1.36x vs raw 33.10) | code lossless=128/128 (first_diff=-1) | acc=10/20 (50.0%)
[BENCH pp512  M=8   ] 380.00 tok/s (WY_GDN=on, 0 SOLVE_TRI, first_tok=42)
[BENCH pp2048 M=8   ] 410.22 tok/s (WY_GDN=on, 0 SOLVE_TRI, first_tok=42)
[PARITY code  vs Tier-1 llama.cpp] 32/32 exact greedy tokens -> PASS
[PROFILE M=1 GPU0 48L] gdn_qkv=1.10ms gdn_rec=0.90ms gdn_out=0.70ms gqa_qkv+o=0.60ms rtr=0.20ms moe_gu=9.00ms moe_dn=7.00ms lm=1.10ms p2p_ar96=1.50ms | sum=22.10ms
[TELEM] power1_input: worst 10 s rolling avg=180.2 W (<=205 W) | worst 1 s avg=190.0 W (info) | instantaneous peak=240.0 W (<270 W) | max_TMem=61.0 C (<65 C) | 900 samples -> PASS
"""


class ScorecardTest(unittest.TestCase):
    def test_parse_head_lines(self):
        r = sc.parse(LOG.splitlines())
        self.assertAlmostEqual(r["raw"]["tps"], 33.10)
        self.assertAlmostEqual(r["raw"]["hbm2_ceiling_gbs"], 860.0)
        self.assertEqual(r["raw"]["dispatches_per_tok"], 410)
        self.assertEqual(sorted(r["prefill"]), [512, 2048])
        self.assertEqual(len(r["spec"]), 5)
        self.assertEqual(r["parity"], {"match": 32, "total": 32, "verdict": "PASS"})
        self.assertEqual(r["telem"]["verdict"], "PASS")
        self.assertAlmostEqual(r["insitu"]["unattributed_ms"], 6.21, places=3)
        self.assertEqual(len(r["insitu"]["stages_ms"]), 8)
        self.assertAlmostEqual(r["gpu0"]["kernels_ms"]["gqa_qkv+o"], 0.60)
        # MTP is fastest but not lossless (127/128): the scorecard must pick the fastest LOSSLESS mode.
        self.assertEqual(sc.best_spec(r)["name"].strip(), "Hybrid")

    def test_kstamp_accounting(self):
        # 100 MHz -> 0.01 us per tick. Token 0: wall 1000 ticks, kernels 300 + 400 (overlapping by 100).
        lines = [
            "[KWALL] gpu=0 tok=0 start=0 end=1000",
            "[KSTAMP] gpu=0 tok=0 name=moe_gu start=100 end=400",
            "[KSTAMP] gpu=0 tok=0 name=moe_dn start=300 end=700",
            "[KWALL] gpu=0 tok=1 start=2000 end=2500",
            "[KSTAMP] gpu=0 tok=1 name=moe_gu start=2000 end=2400",
            "[KSTAMP] gpu=1 tok=9 name=orphan start=1 end=2",
            "[KSTAMP] gpu=0 tok=1 name=wrapped start=9 end=3",
        ]
        r = sc.parse(lines, rtc_mhz=100.0)
        g = r["kstamp"]["gpus"]["0"]
        self.assertEqual(g["tokens"], 2)
        self.assertAlmostEqual(g["wall_us"]["mean"], 7.5)                 # (10 + 5) / 2 us
        self.assertAlmostEqual(g["kernel_sum_us"]["mean"], 5.5)           # (7 + 4) / 2
        self.assertAlmostEqual(g["host_gap_us"]["mean"], 2.0)             # (3 + 1) / 2
        self.assertAlmostEqual(g["idle_us"]["mean"], 2.5)                 # union 6 -> idle 4; token1 idle 1
        self.assertAlmostEqual(g["host_gap_pct"], 26.67, places=2)
        self.assertEqual(g["top_kernels_us_per_tok"][0][0], "moe_gu")
        joined = " ".join(r["warnings"])
        self.assertIn("without KWALL", joined)
        self.assertIn("end < start", joined)
        self.assertEqual(sc._union([(0, 5), (3, 8), (10, 12)]), 10)

    def test_fill_readme(self):
        readme = ("## Performance\n\n| Metric | 2x GPU |\n| :--- | :--- |\n"
                  "| Decode tokens/s, M=1 | [FILL FROM FINAL SCORECARD] |\n"
                  "| Decode tokens/s with speculative decoding | [FILL FROM FINAL SCORECARD] |\n"
                  "| Prefill tokens/s, 2048-token prompt | [FILL FROM FINAL SCORECARD] |\n"
                  "| HBM2 utilization, decode (per GPU, vs measured soak ceiling) | [FILL FROM FINAL SCORECARD] |\n"
                  "| Logit parity vs reference implementation | [FILL FROM FINAL SCORECARD] |\n\nNext paragraph.\n")
        r = sc.parse(LOG.splitlines())
        new, filled, missing = sc.fill_readme(readme, r, commit="abc1234", flags="--tp 2")
        self.assertEqual((len(filled), missing), (5, []))
        self.assertNotIn(sc.PLACEHOLDER, new)
        self.assertIn("| 33.10 tok/s (30.21 ms/tok) |", new)
        self.assertIn("47.50 tok/s (Hybrid, 1.44x vs raw, lossless 128/128)", new)
        self.assertIn("410.22 tok/s (M=8)", new)
        self.assertIn("71.2% (612.0 of 860.0 GB/s calibrated)", new)
        self.assertIn("32/32 exact greedy tokens (PASS)", new)
        self.assertIn("Recorded run: commit `abc1234`; flags `--tp 2`; calibrated HBM2 ceiling 860.0 GB/s.", new)
        self.assertTrue(new.endswith("Next paragraph.\n"))
        # Idempotent: a second fill adds nothing.
        again, f2, _ = sc.fill_readme(new, r, commit="abc1234", flags="--tp 2")
        self.assertEqual((again, f2), (new, []))
        # Partial log: missing metrics stay as placeholders and are reported.
        part = sc.parse([l for l in LOG.splitlines() if "pp2048" not in l and "PARITY" not in l])
        new, filled, missing = sc.fill_readme(readme, part)
        self.assertEqual(len(filled), 3)
        self.assertEqual(missing, ["Prefill tokens/s, 2048-token prompt", "Logit parity vs reference implementation"])
        self.assertIn("commit `UNKNOWN`", new)

    def test_cli_strict(self):
        with tempfile.TemporaryDirectory() as d:
            log, rd, out = (os.path.join(d, n) for n in ("b.log", "R.md", "out.md"))
            with open(log, "w") as fh:
                fh.write(LOG.split("[BENCH pp512")[0])
            with open(rd, "w") as fh:
                fh.write("| Metric | X |\n| :- | :- |\n| Prefill tokens/s, 2048-token prompt | [FILL FROM FINAL SCORECARD] |\n")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc, _ = quiet(sc.main, [log, "--readme", rd, "--out", out, "--strict"])
            self.assertEqual(rc, 1)
            self.assertIn("missing", err.getvalue())
            with contextlib.redirect_stderr(err):
                rc, js = quiet(sc.main, [log, "--json"])
            self.assertEqual(rc, 0)
            self.assertIn('"tps": 33.1', js)


class VerifyModelTest(unittest.TestCase):
    """release/agent/verify_model.py helpers (no binaries, synthetic sysfs only)."""

    @classmethod
    def setUpClass(cls):
        for d in (os.path.join(HERE, "..", "release", "agent"), os.path.join(HERE, "..", "agent")):  # sidecar / repo
            if os.path.isfile(os.path.join(d, "verify_model.py")):
                sys.path.insert(0, d)
                break
        import verify_model  # noqa: E402
        cls.vm = verify_model

    def test_ceilings_and_utilization(self):
        vm = self.vm
        self.assertEqual(vm.parse_ceilings("850, 830", None, 2), [850.0, 830.0])
        self.assertEqual(vm.parse_ceilings(None, 900.0, 3), [900.0] * 3)
        self.assertEqual(vm.parse_ceilings(None, None, 2), [])
        with self.assertRaises(SystemExit):
            vm.parse_ceilings("850,0", None, 2)
        ach, tot, pct = vm.hbm_utilization(40.0, 20e9, [850.0, 850.0])
        self.assertAlmostEqual(ach, 800.0)
        self.assertAlmostEqual(tot, 1700.0)
        self.assertAlmostEqual(pct, 100.0 * 800 / 1700)

    def test_logit_parity(self):
        import array as arr
        with tempfile.TemporaryDirectory() as d:
            ref, good, bad = (os.path.join(d, n) for n in ("r.f32", "g.f32", "b.f32"))
            a = [math.sin(i) for i in range(32)]
            arr.array("f", a).tofile(open(ref, "wb"))
            arr.array("f", [x + 1e-5 for x in a]).tofile(open(good, "wb"))
            b = list(a)
            b[0], b[1] = b[1] + 3.0, b[0]
            arr.array("f", b).tofile(open(bad, "wb"))
            ok, msg = self.vm.logit_parity(ref, good, 16)
            self.assertTrue(ok, msg)
            self.assertIn("argmax_agree=2/2", msg)
            ok, _ = self.vm.logit_parity(ref, bad, 16)
            self.assertFalse(ok)
            ok, msg = self.vm.logit_parity(ref, good, 5)
            self.assertFalse(ok)
            self.assertIn("length mismatch", msg)

    def test_ras_counts_fixture(self):
        with tempfile.TemporaryDirectory() as d:
            for name, vendor, umc in (("gpu0", "0x1002", "ue: 0\nce: 2\n"), ("gpu1", "0x1002", "ue: 1\nce: 0\n"),
                                      ("nic0", "0x8086", "ue: 9\nce: 9\n")):
                p = os.path.join(d, "sys/bus/pci/devices", name)
                os.makedirs(os.path.join(p, "ras"))
                open(os.path.join(p, "vendor"), "w").write(vendor + "\n")
                open(os.path.join(p, "ras", "umc_err_count"), "w").write(umc)
            old = os.environ.get("VJ_SYSFS_ROOT")
            os.environ["VJ_SYSFS_ROOT"] = d
            try:
                self.assertEqual(self.vm.ras_counts(), {"ue": 1, "ce": 2, "files": 2})
            finally:
                if old is None:
                    del os.environ["VJ_SYSFS_ROOT"]
                else:
                    os.environ["VJ_SYSFS_ROOT"] = old


if __name__ == "__main__":
    unittest.main()
