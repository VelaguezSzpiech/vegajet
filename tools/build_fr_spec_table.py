#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Build an FR-Spec token frequency table from already-tokenized corpora (standard library only).

Reads token-id streams and writes the text table that core/fr_spec.cpp (load_frequency_table) parses:

    # vegajet fr-spec frequency table v1
    # vocab 248320
    # tokens <total> distinct <n>
    <token_id> <count>        (count desc, id asc)

Inputs (any mix, '-' = stdin):
  * *.jsonl  one JSON value per line: a list of ids, or an object with "ids", "input_ids" or "tokens"
  * *.json   a list of ids, a list of lists, or one such object
  * --u32    raw little-endian uint32 ids (e.g. a dumped token buffer)
  * anything else: integers separated by whitespace and/or commas

Examples:
  build_fr_spec_table.py --vocab 248320 corpus_ids.jsonl -o fr_table.txt
  build_fr_spec_table.py --vocab 248320 --u32 dump.bin --report 32768,65536
"""
import argparse
import array
import json
import os
import re
import sys

HEADER = "# vegajet fr-spec frequency table v1"
ID_KEYS = ("ids", "input_ids", "tokens")
GRANULE = 256


class TableError(Exception):
    pass


def _ids_from_json(value, where):
    if isinstance(value, dict):
        for k in ID_KEYS:
            if k in value:
                return _ids_from_json(value[k], where)
        raise TableError(f"{where}: object without one of {', '.join(ID_KEYS)}")
    if isinstance(value, list):
        out = []
        for v in value:
            if isinstance(v, list) or isinstance(v, dict):
                out.extend(_ids_from_json(v, where))
            elif isinstance(v, int) and not isinstance(v, bool):
                out.append(v)
            else:
                raise TableError(f"{where}: non-integer token id {v!r}")
        return out
    raise TableError(f"{where}: expected a list of token ids")


def read_ids(name, data, u32=False):
    """Token ids of one input. `data` is bytes."""
    if u32:
        if len(data) % 4:
            raise TableError(f"{name}: size is not a multiple of 4 bytes")
        a = array.array("I")
        a.frombytes(data)
        if sys.byteorder != "little":
            a.byteswap()
        return list(a)
    text = data.decode("utf-8")
    lower = name.lower()
    if lower.endswith(".jsonl"):
        out = []
        for i, line in enumerate(text.splitlines(), 1):
            if line.strip():
                out.extend(_ids_from_json(json.loads(line), f"{name}:{i}"))
        return out
    if lower.endswith(".json"):
        return _ids_from_json(json.loads(text), name)
    out = []
    for i, tok in enumerate(t for t in re.split(r"[\s,]+", text) if t):
        if not re.fullmatch(r"\d+", tok):
            raise TableError(f"{name}: token #{i} {tok!r} is not a non-negative integer")
        out.append(int(tok))
    return out


def count_ids(streams, vocab):
    counts = {}
    total = 0
    for name, ids in streams:
        for t in ids:
            if t < 0 or t >= vocab:
                raise TableError(f"{name}: token id {t} outside the vocabulary (vocab {vocab})")
            counts[t] = counts.get(t, 0) + 1
            total += 1
    return counts, total


def ranked(counts, min_count=1, top=0):
    rows = sorted(((t, c) for t, c in counts.items() if c >= min_count), key=lambda tc: (-tc[1], tc[0]))
    return rows[:top] if top else rows


def round_rows(rows):
    return -(-rows // GRANULE) * GRANULE


def coverage(rows, total, k):
    """Share of the corpus covered by the top round_rows(k) ids of the table."""
    if total <= 0:
        return 0.0
    return sum(c for _, c in rows[: round_rows(k)]) / total


def render(rows, vocab, total):
    lines = [HEADER, f"# vocab {vocab}", f"# tokens {total} distinct {len(rows)}"]
    lines.extend(f"{t} {c}" for t, c in rows)
    return "\n".join(lines) + "\n"


def parse_table(text, vocab):
    """Python mirror of fr_spec::parse_frequency_table for counted tables (used by the tests)."""
    out = {}
    for ln, line in enumerate(text.splitlines(), 1):
        f = line.split()
        if not f:
            continue
        if f[0].startswith("#"):
            body = line.split("#", 1)[1].split()
            if len(body) >= 2 and body[0] == "vocab" and int(body[1]) != vocab:
                raise TableError(f"line {ln}: table vocab {body[1]} != {vocab}")
            continue
        if len(f) != 2:
            raise TableError(f"line {ln}: expected '<id> <count>'")
        t, c = int(f[0]), float(f[1])
        if not 0 <= t < vocab or c < 0:
            raise TableError(f"line {ln}: bad entry")
        out[t] = out.get(t, 0.0) + c
    return sorted(out.items(), key=lambda tc: (-tc[1], tc[0]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("inputs", nargs="+", help="token-id files ('-' = stdin)")
    ap.add_argument("--vocab", type=int, required=True, help="target vocabulary size (e.g. 248320)")
    ap.add_argument("--u32", action="store_true", help="inputs are raw little-endian uint32 ids")
    ap.add_argument("--min-count", type=int, default=1, help="drop ids seen fewer times (default 1)")
    ap.add_argument("--top", type=int, default=0, help="keep only the N most frequent ids (0 = all)")
    ap.add_argument("--report", default="32768,65536",
                    help="comma-separated draft sizes whose corpus coverage is printed to stderr ('' = none)")
    ap.add_argument("-o", "--out", default="-", help="output file ('-' = stdout)")
    a = ap.parse_args(argv)
    if a.vocab <= 0 or a.min_count < 1 or a.top < 0:
        ap.error("--vocab must be > 0, --min-count >= 1, --top >= 0")
    try:
        streams = []
        for name in a.inputs:
            if name == "-":
                data = sys.stdin.buffer.read()
            else:
                with open(name, "rb") as f:
                    data = f.read()
            streams.append((os.path.basename(name) if name != "-" else "stdin", read_ids(name, data, a.u32)))
        counts, total = count_ids(streams, a.vocab)
    except (OSError, ValueError, TableError) as e:
        print(f"build_fr_spec_table: {e}", file=sys.stderr)
        return 1
    if total == 0:
        print("build_fr_spec_table: no token ids in the inputs", file=sys.stderr)
        return 1
    rows = ranked(counts, a.min_count, a.top)
    text = render(rows, a.vocab, total)
    if a.out == "-":
        sys.stdout.write(text)
    else:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
    for k in (int(x) for x in a.report.split(",") if x.strip()):
        print(f"[fr_spec] top {round_rows(k)} rows (requested {k}) cover {100.0 * coverage(rows, total, k):.2f}% "
              f"of {total} corpus tokens ({len(rows)} distinct ids)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
