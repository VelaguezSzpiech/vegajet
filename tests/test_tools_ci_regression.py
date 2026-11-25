# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""tools/ci_regression.py (ci_regression_suite.patch): argument handling, the GPU-matrix gates on synthetic
scorecards (pass, and each gate failing on its own), --dry-run never executing anything, the stacked patch
apply-check stage on a scratch repository (in-order stacking, a conflicting patch stops the stack), the tool
self-test stage, and the workflow wiring (CPU job on push / pull request, GPU matrix only nightly / manual on a
self-hosted runner, repository variables passed through the environment, not spliced into the script)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CI = ROOT / "tools/ci_regression.py"
WORKFLOW = ROOT / ".github/workflows/cpu-ci.yml"

GOOD = {"commit": "abc1234", "throughput": {"decode_m1_tok_s": 50.0, "prefill_2048_tok_s": 1400.0}, "kld": 0.001,
        "parity": {"code": "32/32", "spec": {"match": 129, "total": 129}}, "spill_bytes": 0,
        "telemetry": {"tmem_c": 55.0, "ue": 0, "de": 0}}


def run(*args, cwd=None):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, str(CI), *map(str, args)], capture_output=True, text=True, env=env,
                          timeout=300, cwd=cwd)


class CiRegression(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def cards(self, name, docs):
        d = self.d / name
        d.mkdir()
        for n, doc in docs.items():
            (d / ("tp%d.json" % n)).write_text(json.dumps(doc))
        return d

    def test_self_test(self):
        p = run("--self-test")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("PASS", p.stdout)

    def test_usage_errors(self):
        self.assertEqual(run().returncode, 2)
        self.assertEqual(run("--gpu-matrix", "0", "--model", "m").returncode, 2)
        self.assertEqual(run("--gpu-matrix", "1,x", "--model", "m").returncode, 2)
        self.assertEqual(run("--gpu-matrix", "1").returncode, 2)  # no --model, no --from-json
        self.assertEqual(run("--cpu-only", "--stages", "opt,frobnicate", "--dry-run").returncode, 2)
        self.assertEqual(run("--cpu-only", "--gpu-matrix", "1").returncode, 2)  # one mode only

    def test_matrix_gates_pass(self):
        cand = self.cards("cand", {1: GOOD, 2: GOOD, 3: GOOD})
        base = self.cards("base", {1: GOOD, 2: GOOD, 3: GOOD})
        rep = self.d / "r.json"
        p = run("--gpu-matrix", "1,2,3", "--from-json", cand, "--baseline-dir", base, "--report", rep)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        doc = json.loads(rep.read_text())
        self.assertTrue(doc["ok"])
        names = {s["name"] for s in doc["stages"]}
        for n in (1, 2, 3):
            for g in ("spill", "ue", "de", "parity", "baseline"):
                self.assertIn("tp%d %s" % (n, g), names)

    def test_matrix_gates_fail_individually(self):
        def failing(doc, base=None):
            cand = self.cards("c%d" % len(list(self.d.iterdir())), {1: doc})
            args = ["--gpu-matrix", "1", "--from-json", cand, "--report", self.d / "r.json"]
            if base is not None:
                args += ["--baseline-dir", self.cards("b%d" % len(list(self.d.iterdir())), {1: base})]
            p = run(*args)
            self.assertEqual(p.returncode, 1, p.stdout)
            return {s["name"] for s in json.loads((self.d / "r.json").read_text())["stages"] if not s["ok"]}

        self.assertEqual(failing(dict(GOOD, spill_bytes=512)), {"tp1 spill"})
        self.assertEqual(failing(dict(GOOD, telemetry={"tmem_c": 50.0, "ue": 1, "de": 0})), {"tp1 ue"})
        self.assertEqual(failing(dict(GOOD, telemetry={"tmem_c": 50.0, "ue": 0, "de": 3})), {"tp1 de"})
        self.assertEqual(failing(dict(GOOD, parity={"code": "31/32", "spec": {"match": 129, "total": 129}})), {"tp1 parity"})
        self.assertEqual(failing(dict(GOOD, parity={"code": "32/32", "spec": {"match": 128, "total": 129}})), {"tp1 parity"})
        self.assertEqual(failing(dict(GOOD, parity={"code": "32/32"})), {"tp1 parity"})  # 129/129 missing
        no_spill = {k: v for k, v in GOOD.items() if k != "spill_bytes"}
        self.assertEqual(failing(no_spill), {"tp1 spill"})  # not reported counts as a failure
        slow = dict(GOOD, throughput={"decode_m1_tok_s": 40.0, "prefill_2048_tok_s": 1400.0})
        self.assertEqual(failing(slow, base=GOOD), {"tp1 baseline"})
        # a missing scorecard
        empty = self.d / "empty"
        empty.mkdir()
        p = run("--gpu-matrix", "2", "--from-json", empty, "--report", self.d / "r.json")
        self.assertEqual(p.returncode, 1)

    def test_dry_run_executes_nothing(self):
        marker = self.d / "ran"
        fake = self.d / "vegajet"
        fake.write_text("#!/bin/sh\ntouch %s\n" % marker)
        fake.chmod(0o755)
        p = run("--gpu-matrix", "1,2,3", "--model", "m.bin", "--vegajet", fake, "--preflight", "--runner", "wrap --",
                "--dry-run", "--out-dir", self.d / "out")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertFalse(marker.exists())
        self.assertFalse((self.d / "out").exists())
        for spec in ("--tp 1 --gpu 0 ", "--tp 2 --gpu 0-1 ", "--tp 3 --gpu 0-2 "):
            self.assertIn(spec, p.stdout)
        self.assertIn("wrap -- %s doctor --json" % fake, p.stdout)
        p = run("--cpu-only", "--dry-run", "--stages", "opt,san,privacy", "--build-root", self.d / "b")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("-DVEGAJET_CPU_ONLY=ON", p.stdout)
        self.assertIn("-fsanitize=address,undefined", p.stdout)
        self.assertIn("-E 'vegajet_doctor|test_prefill_dot8_kld'", p.stdout)
        self.assertFalse((self.d / "b").exists())

    def git(self, repo, *args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                       env=dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t",
                                GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t"))

    def test_patch_stack_stage(self):
        repo = self.d / "repo"
        repo.mkdir()
        (repo / "a.txt").write_text("one\ntwo\nthree\n")
        self.git(repo, "init", "-q")
        self.git(repo, "add", "-A")
        self.git(repo, "commit", "-qm", "base")
        p1 = self.d / "1.patch"
        p1.write_text("--- a/a.txt\n+++ b/a.txt\n@@ -1,3 +1,4 @@\n one\n+one-and-a-half\n two\n three\n")
        p2 = self.d / "2.patch"  # needs p1 underneath
        p2.write_text("--- a/a.txt\n+++ b/a.txt\n@@ -1,4 +1,4 @@\n one\n-one-and-a-half\n+1.5\n two\n three\n")
        bad = self.d / "3.patch"  # conflicts with the stack
        bad.write_text("--- a/a.txt\n+++ b/a.txt\n@@ -1,3 +1,3 @@\n one\n-two\n+zwei\n four\n")
        rep = self.d / "r.json"
        p = run("--cpu-only", "--stages", "patches", "--source", repo, "--patch", p1, "--patch", p2, "--report", rep)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        st = json.loads(rep.read_text())["stages"][0]
        self.assertEqual([x["ok"] for x in st["patches"]], [True, True])
        p = run("--cpu-only", "--stages", "patches", "--source", repo, "--patch", p2, "--patch", p1, "--report", rep)
        self.assertEqual(p.returncode, 1)  # wrong order: the second patch's base is missing
        p = run("--cpu-only", "--stages", "patches", "--source", repo, "--patch", p1, "--patch", bad, "--patch", p2,
                "--report", rep)
        self.assertEqual(p.returncode, 1)
        st = json.loads(rep.read_text())["stages"][0]
        self.assertEqual([x["ok"] for x in st["patches"]], [True, False])  # stops at the first failure
        self.assertEqual((repo / "a.txt").read_text(), "one\ntwo\nthree\n")  # the source is never written
        p = run("--cpu-only", "--stages", "patches", "--source", repo)
        self.assertEqual(p.returncode, 0)  # no patches: skipped

    def test_selftest_stage(self):
        p = run("--cpu-only", "--stages", "selftests")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)

    def test_workflow(self):
        w = WORKFLOW.read_text()
        self.assertIn("python3 tools/ci_regression.py --cpu-only", w)
        self.assertIn("--gpu-matrix 1,2,3", w)
        self.assertIn("permissions:\n  contents: read", w)
        self.assertIn("schedule:", w)
        self.assertIn("if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'", w)
        self.assertIn("runs-on: [self-hosted, linux, gfx906]", w)
        # repository variables reach the script through env, never spliced into a run: line
        in_run = False
        for line in w.splitlines():
            s = line.strip()
            if s.startswith("run:") or s.startswith("- run:"):
                in_run = True
            elif s.startswith("- name:") or s.startswith("env:") or s.startswith("uses:") or s.startswith("with:"):
                in_run = False
            if in_run:
                self.assertNotIn("${{", line, line)


if __name__ == "__main__":
    unittest.main()
