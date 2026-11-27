# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Unit tests for tools/token_compare.py (standard library only, synthetic dumps only).

Covers the four shapes the parity gate cares about: equal sequences, a strict prefix (different length), a first
difference in the middle, and empty dumps, plus the CLI exit codes that bench/v12/run.sh relies on.

Run from the repo root:  python3 -m unittest discover -s tests -p 'test_token_compare*.py'
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

import token_compare as tc  # noqa: E402


class TokenCompareTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def dump(self, name, ids, header=True):
        path = os.path.join(self.d, name)
        with open(path, "w", encoding="utf-8") as fh:
            if header:
                fh.write("# vegajet token ids: %d tokens, one id per line (tools/token_compare.py)\n" % len(ids))
            for i in ids:
                fh.write("%d\n" % i)
        return path

    def run_cli(self, *argv):
        buf, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            rc = tc.main(list(argv))
        return rc, buf.getvalue(), err.getvalue()

    # ---- the four required shapes -------------------------------------------------------------------------
    def test_equal_sequences_match(self):
        ref = self.dump("ref.tokens", [760, 6511, 314, 9338])
        cand = self.dump("cand.tokens", [760, 6511, 314, 9338])
        rec = tc.compare(tc.read_dump(ref), tc.read_dump(cand))
        self.assertEqual(rec["matched"], 4)
        self.assertEqual(rec["total"], 4)
        self.assertEqual(rec["first_diff"], -1)
        self.assertTrue(rec["equal"])

    def test_prefix_equal_different_length(self):
        ref = self.dump("ref.tokens", [1, 2, 3, 4, 5])
        cand = self.dump("cand.tokens", [1, 2, 3])
        rec = tc.compare(tc.read_dump(ref), tc.read_dump(cand))
        self.assertEqual(rec["first_diff"], 3)
        self.assertEqual(rec["matched"], 3)
        self.assertEqual(rec["total"], 5)
        self.assertEqual(rec["ref_token"], 4)
        self.assertIsNone(rec["cand_token"])
        self.assertFalse(rec["equal"])
        # ... and the other direction: the candidate is longer, still a mismatch.
        rec = tc.compare(tc.read_dump(cand), tc.read_dump(ref))
        self.assertEqual(rec["first_diff"], 3)
        self.assertIsNone(rec["ref_token"])
        self.assertEqual(rec["cand_token"], 4)
        self.assertFalse(rec["equal"])

    def test_first_diff_in_middle(self):
        ref = self.dump("ref.tokens", [10, 20, 30, 40, 50])
        cand = self.dump("cand.tokens", [10, 20, 99, 40, 50])
        rec = tc.compare(tc.read_dump(ref), tc.read_dump(cand))
        self.assertEqual(rec["first_diff"], 2)
        self.assertEqual(rec["matched"], 2)
        self.assertEqual(rec["total"], 5)
        self.assertEqual(rec["ref_token"], 30)
        self.assertEqual(rec["cand_token"], 99)
        self.assertFalse(rec["equal"])

    def test_empty_dumps_match(self):
        ref = self.dump("ref.tokens", [])
        cand = self.dump("cand.tokens", [])
        rec = tc.compare(tc.read_dump(ref), tc.read_dump(cand))
        self.assertEqual(rec["matched"], 0)
        self.assertEqual(rec["total"], 0)
        self.assertEqual(rec["first_diff"], -1)
        self.assertTrue(rec["equal"])
        rc, out, _ = self.run_cli(ref, cand)
        self.assertEqual(rc, 0, out)

    # ---- CLI contract used by bench/v12/run.sh -------------------------------------------------------------
    def test_cli_exit_codes(self):
        ref = self.dump("ref.tokens", [5, 6, 7])
        same = self.dump("same.tokens", [5, 6, 7])
        diff = self.dump("diff.tokens", [5, 6, 8])
        self.assertEqual(self.run_cli(ref, same)[0], 0)
        self.assertEqual(self.run_cli(ref, diff)[0], 1)
        self.assertEqual(self.run_cli(ref, os.path.join(self.d, "missing.tokens"))[0], 2)

    def test_cli_reports_counts(self):
        ref = self.dump("ref.tokens", [1, 2, 3, 4])
        cand = self.dump("cand.tokens", [1, 2, 3, 4])
        rc, out, _ = self.run_cli(ref, cand)
        self.assertEqual(rc, 0)
        self.assertIn("MATCH 4/4", out)
        cand2 = self.dump("cand2.tokens", [1, 2, 9, 4])
        rc, out, _ = self.run_cli(ref, cand2)
        self.assertEqual(rc, 1)
        self.assertIn("MISMATCH 2/4", out)
        self.assertIn("first_diff=2", out)
        self.assertIn("ref=3", out)
        self.assertIn("cand=9", out)

    def test_cli_json(self):
        ref = self.dump("ref.tokens", [1, 2])
        cand = self.dump("cand.tokens", [1, 2])
        rc, out, _ = self.run_cli(ref, cand, "--json")
        self.assertEqual(rc, 0)
        rec = json.loads(out)
        self.assertTrue(rec["equal"])
        self.assertEqual(rec["matched"], 2)
        self.assertEqual(rec["schema"], "vegajet.token_compare.v1")

    # ---- parser tolerance ----------------------------------------------------------------------------------
    def test_json_and_loose_text(self):
        path = os.path.join(self.d, "ids.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"tokens": [4, 5]}, fh)
        self.assertEqual(tc.read_dump(path), [4, 5])
        path = os.path.join(self.d, "loose.tokens")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("# header\n\n7 8\n9\n")
        self.assertEqual(tc.read_dump(path), [7, 8, 9])

    def test_unusable_input(self):
        bad = os.path.join(self.d, "bad.tokens")
        with open(bad, "w", encoding="utf-8") as fh:
            fh.write("1\nnope\n")
        with self.assertRaises(ValueError):
            tc.read_dump(bad)
        rc, _, err = self.run_cli(bad, bad)
        self.assertEqual(rc, 2)
        self.assertIn("token_compare:", err)


if __name__ == "__main__":
    unittest.main()
