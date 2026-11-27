#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Compare two generated token-id dumps (reference vs candidate) and report matched/total + first_diff.

  token_compare.py REF CAND [--json]

Inputs are token-id dumps written by `vegajet bench --dump-tokens FILE` (format: '#' comment lines, then one
decimal token id per line) or a JSON array / {"tokens": [...]} document. Blank lines and '#' comments are ignored
in the text form. The comparison is exact and positional: a dump is a full match only when both sequences have the
same length and every id agrees, so exit 0 always means the complete emitted sequence is identical.

  matched    number of leading ids that agree (= len of the common prefix)
  total      length of the reference sequence
  first_diff index of the first disagreement, -1 for a full match; when one sequence is a strict prefix of the
             other it is the shorter length, and the missing side's token is reported as null

Exit codes: 0 full match, 1 mismatch, 2 unusable input (missing file, unparsable id, bad argument).
"""
import argparse
import json
import sys


def read_dump(path):
    """Parses one token-id dump into a list of ints. Raises ValueError with a message on unusable input."""
    with open(path, "rb") as fh:
        raw = fh.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("%s: not UTF-8 text (%s); write dumps with --dump-tokens" % (path, exc))
    stripped = text.lstrip()
    if stripped[:1] in ("[", "{"):
        try:
            doc = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError("%s: invalid JSON dump (%s)" % (path, exc))
        if isinstance(doc, dict):
            for key in ("tokens", "ids", "token_ids"):
                if key in doc:
                    doc = doc[key]
                    break
            else:
                raise ValueError("%s: JSON object needs a \"tokens\" (or \"ids\") list" % path)
        if not isinstance(doc, list):
            raise ValueError("%s: JSON dump is not a list of token ids" % path)
        out = []
        for i, v in enumerate(doc):
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError("%s: element %d is not an integer token id" % (path, i))
            out.append(v)
        return out
    ids = []
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for field in line.split():
            try:
                ids.append(int(field, 10))
            except ValueError:
                raise ValueError("%s:%d: %r is not a decimal token id" % (path, lineno, field))
    return ids


def compare(ref, cand):
    """Returns the comparison record. first_diff is -1 only for a full match."""
    total = len(ref)
    n = min(len(ref), len(cand))
    first_diff = -1
    for i in range(n):
        if ref[i] != cand[i]:
            first_diff = i
            break
    if first_diff < 0 and len(ref) != len(cand):
        first_diff = n
    matched = total if first_diff < 0 else first_diff
    return {
        "matched": matched,
        "total": total,
        "ref_len": len(ref),
        "cand_len": len(cand),
        "first_diff": first_diff,
        "ref_token": ref[first_diff] if 0 <= first_diff < len(ref) else None,
        "cand_token": cand[first_diff] if 0 <= first_diff < len(cand) else None,
        "equal": first_diff < 0,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare two `vegajet bench --dump-tokens` dumps.")
    ap.add_argument("ref", help="reference dump (e.g. main / all-off run)")
    ap.add_argument("cand", help="candidate dump (e.g. v1.2 toggle run)")
    ap.add_argument("--json", action="store_true", help="print the comparison as one JSON object")
    args = ap.parse_args(argv)
    try:
        rec = compare(read_dump(args.ref), read_dump(args.cand))
    except (OSError, ValueError) as exc:
        sys.stderr.write("token_compare: %s\n" % exc)
        return 2
    if args.json:
        rec = dict(rec, schema="vegajet.token_compare.v1", ref=args.ref, cand=args.cand)
        print(json.dumps(rec, sort_keys=True))
    elif rec["equal"]:
        print("token_compare: MATCH %d/%d ids (ref %d, cand %d)%s"
              % (rec["matched"], rec["total"], rec["ref_len"], rec["cand_len"],
                 "" if rec["total"] else " [both dumps empty]"))
    else:
        print("token_compare: MISMATCH %d/%d ids first_diff=%d ref=%s cand=%s (ref %d, cand %d)"
              % (rec["matched"], rec["total"], rec["first_diff"], rec["ref_token"], rec["cand_token"],
                 rec["ref_len"], rec["cand_len"]))
    return 0 if rec["equal"] else 1


if __name__ == "__main__":
    sys.exit(main())
