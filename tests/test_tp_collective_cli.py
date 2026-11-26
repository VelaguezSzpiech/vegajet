#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""CPU-only CLI admission checks; no execution command can reach a device node."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

BINARY = os.environ.get("VJ_TP_BINARY", "")


@unittest.skipUnless(BINARY, "requires CMake VJ_TP_BINARY")
class CollectiveCLI(unittest.TestCase):
    def run_cli(self, *args):
        env = dict(os.environ)
        env.pop("VJ_GPU_DEVICES", None)
        return subprocess.run([BINARY, "bench", "--bench-ar", *args], env=env,
                              text=True, capture_output=True, timeout=20)

    def test_rank_matrix(self):
        for n in [1, 2, 3, 4, 6, 8, 12, 16, 20, 23]:
            with self.subTest(n=n):
                p = self.run_cli("--tp", str(n), "--dry-run")
                self.assertEqual(p.returncode, 0, p.stderr)
                self.assertIn(f"TP={n} ", p.stdout)
                if n == 1:
                    self.assertIn("collective=bypass wire=none", p.stdout)
                    self.assertIn("payload=0 flags=0 banks=0", p.stdout)
                else:
                    self.assertIn("wire=FP16", p.stdout)
                    self.assertIn("host-bounce", p.stdout)
                    self.assertIn("flag-stride=64", p.stdout)
                if n != 2:
                    p = self.run_cli("--tp", str(n))
                    self.assertEqual(p.returncode, 3)
                    self.assertIn("this build has no GPU backend", p.stderr)

    def test_default_and_invalid(self):
        p = self.run_cli("--dry-run")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("TP=2 ", p.stdout)
        for value in ["0", "-1", "three", "3x", "1.0", "4294967296", "4294967295"]:
            p = self.run_cli("--tp", value, "--dry-run")
            self.assertEqual(p.returncode, 2, (value, p.stderr))
        self.assertEqual(self.run_cli("--tp").returncode, 2)
        self.assertEqual(self.run_cli("--tp-plan").returncode, 2)
        self.assertEqual(self.run_cli("--ar-wire", "int8", "--dry-run").returncode, 2)

    def test_profile_and_override(self):
        with tempfile.TemporaryDirectory(prefix="vj-tp-cli-") as tmp:
            profile = Path(tmp) / "profile.json"
            owner = Path(tmp) / "owner.json"
            gpus = []
            for s in range(3):
                gpus.append({"p2p": [{"kind": "pcie" if s != d else "unknown",
                                      "gbs": 8 if s != d else 0} for d in range(3)]})
            profile.write_text(json.dumps({"schema": 1, "gpus": gpus}))
            owner.write_text(json.dumps({"schema": 1, "stages": [[2, 0], [1]],
                                         "host_bounce": [[0, 1]]}))
            args = ("--tp", "3", "--tp-profile", str(profile), "--tp-plan", str(owner))
            p = self.run_cli(*args, "--dry-run")
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("pipeline TP stages: [2,0] [1]", p.stdout)
            self.assertIn("rank 0 -> 1: host-bounce", p.stdout)
            self.assertIn("rank 2 -> 0: peer-push", p.stdout)
            self.assertEqual(self.run_cli(*args).returncode, 3)
            self.assertEqual(self.run_cli("--tp", "2", "--tp-profile", str(profile), "--dry-run").returncode, 2)
            owner.write_text('{"schema":1,"stages":[[0],[1]]}')
            self.assertEqual(self.run_cli(*args, "--dry-run").returncode, 2)
            owner.write_text('{"schema":1,"typo":true}')
            self.assertEqual(self.run_cli(*args, "--dry-run").returncode, 2)
            profile.write_text("{broken")
            self.assertEqual(self.run_cli(*args, "--dry-run").returncode, 2)


if __name__ == "__main__":
    unittest.main()
