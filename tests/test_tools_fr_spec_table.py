# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Unit tests for tools/build_fr_spec_table.py (standard library only, synthetic data only).

Run from the repo root:  python3 -m unittest discover -s tests -p 'test_tools*.py'
"""
import contextlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

import build_fr_spec_table as fr  # noqa: E402


def run(argv, stdin=b""):
    out, err = io.StringIO(), io.StringIO()

    class FakeStdin:
        buffer = io.BytesIO(stdin)

    old = sys.stdin
    sys.stdin = FakeStdin()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = fr.main(argv)
    finally:
        sys.stdin = old
    return rc, out.getvalue(), err.getvalue()


class BuildFrSpecTableTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def path(self, name, data):
        p = os.path.join(self.tmp.name, name)
        with open(p, "wb") as f:
            f.write(data if isinstance(data, bytes) else data.encode())
        return p

    def test_formats_and_order(self):
        txt = self.path("a.txt", "5 3, 3\n9 5 5\n")
        jl = self.path("b.jsonl", json.dumps({"ids": [3, 7]}) + "\n\n" + json.dumps([[5], [7, 7]]) + "\n")
        js = self.path("c.json", json.dumps({"input_ids": [[1, 2], [2]]}))
        rc, out, err = run(["--vocab", "10", txt, jl, js, "--report", "1"])
        self.assertEqual(rc, 0, err)
        lines = out.splitlines()
        self.assertEqual(lines[0], fr.HEADER)
        self.assertEqual(lines[1], "# vocab 10")
        self.assertEqual(lines[2], "# tokens 14 distinct 6")
        # 5 x4, 3 x3, 7 x3, 2 x2, 9 x1, 1 x1 -> count desc, id asc
        self.assertEqual(lines[3:], ["5 4", "3 3", "7 3", "2 2", "1 1", "9 1"])
        self.assertIn("top 256 rows", err)
        self.assertIn("100.00%", err)

    def test_u32_and_stdin(self):
        data = struct.pack("<6I", 4, 4, 1, 4, 0, 1)
        p = self.path("ids.bin", data)
        rc, out, _ = run(["--vocab", "8", "--u32", p, "--report", ""])
        self.assertEqual(rc, 0)
        self.assertEqual(out.splitlines()[3:], ["4 3", "1 2", "0 1"])
        rc, out, _ = run(["--vocab", "8", "--u32", "-", "--report", ""], stdin=data)
        self.assertEqual(out.splitlines()[3:], ["4 3", "1 2", "0 1"])
        bad = self.path("odd.bin", b"\x01\x02\x03")
        rc, _, err = run(["--vocab", "8", "--u32", bad])
        self.assertEqual(rc, 1)
        self.assertIn("multiple of 4", err)

    def test_rejects_bad_input(self):
        for name, data, msg in [
            ("x.txt", "1 2 -3", "non-negative"),
            ("y.txt", "1 99", "outside the vocabulary"),
            ("z.jsonl", json.dumps({"text": "hi"}) + "\n", "object without"),
            ("w.json", json.dumps([1, "a"]), "non-integer"),
            ("e.txt", "  \n", "no token ids"),
        ]:
            rc, _, err = run(["--vocab", "50", self.path(name, data)])
            self.assertEqual(rc, 1, name)
            self.assertIn(msg, err, name)
        rc, _, err = run(["--vocab", "50", os.path.join(self.tmp.name, "missing.txt")])
        self.assertEqual(rc, 1)

    def test_top_min_count_and_file_output(self):
        p = self.path("c.txt", " ".join(["1"] * 5 + ["2"] * 3 + ["3"] * 1))
        outp = os.path.join(self.tmp.name, "table.txt")
        rc, out, _ = run(["--vocab", "4", "--min-count", "2", "--top", "1", p, "-o", outp, "--report", ""])
        self.assertEqual((rc, out), (0, ""))
        with open(outp, encoding="utf-8") as f:
            text = f.read()
        self.assertEqual(text.splitlines()[3:], ["1 5"])
        self.assertEqual(fr.parse_table(text, 4), [(1, 5.0)])
        with self.assertRaises(fr.TableError):
            fr.parse_table(text, 5)  # vocab header mismatch, as the C++ loader

    def test_round_rows_and_coverage(self):
        self.assertEqual([fr.round_rows(k) for k in (1, 256, 257, 32768, 32769)], [256, 256, 512, 32768, 33024])
        rows = [(i, 1000 - i) for i in range(1000)]
        total = sum(c for _, c in rows)
        self.assertAlmostEqual(fr.coverage(rows, total, 1000), 1.0)
        self.assertLess(fr.coverage(rows, total, 1), 1.0)
        self.assertEqual(fr.coverage([], 0, 256), 0.0)


if __name__ == "__main__":
    unittest.main()
