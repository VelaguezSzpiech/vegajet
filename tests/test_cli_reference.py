# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""release/docs/CLI_REFERENCE.md must match the code, in both directions:

* per subcommand, the flags its parser compares against (`a == "--x"`) vs the flag rows of its doc section;
* the subcommands main() dispatches vs the doc's subcommand table;
* every getenv("NAME") / kEnvVar = "NAME" in core/, models/, cli/, ops/ vs the environment table (a getenv of a
  non-literal name fails the test, since it cannot be checked), and for every `getenv("NAME")[0] == 'c'` test the
  NAME row must mention `c` (catches a default flipping, e.g. VEGAJET_ENABLE_SUPER at 950c137);
* the exit codes each CLI source file returns vs the codes its doc section lists, and all of them vs the exit-code
  table;
* each subcommand's usage text (the printf holding "Usage:") lists exactly the flags its parser accepts;
* with VEGAJET_BIN set, the live `--help` output of the top level, inspect, plan, doctor and serve (bench, run and
  soak are never executed, not even for --help).
A self-test mutates the document and checks that every kind of drift is reported."""
import os
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(os.environ.get("VJ_CLI_SRC") or Path(__file__).resolve().parent.parent)
DOC = ROOT / "release/docs/CLI_REFERENCE.md"
BIN = os.environ.get("VEGAJET_BIN")
HELP = {"--help", "-h"}
FLAG_RE = re.compile(r"(?<![\w./-])(--?[a-z][a-z0-9]*(?:-[a-z0-9]+)*)")  # not inside a path


def src(rel):
    return (ROOT / rel).read_text()


def body(text, signature):
    """Text of the function whose definition starts with `signature`, up to its closing brace at column 0."""
    i = text.index(signature)
    j = text.index("\n}\n", i)
    return text[i:j]


def parsed_flags(text, var="a"):
    return set(re.findall(r'\b%s\s*==\s*"(--?[a-z][a-z0-9-]*)"' % var, text))


def usage_literals(text, marker):
    """Concatenated string literals of the printf / fprintf call whose format contains `marker`."""
    k = text.index(marker)
    start = text.rindex("printf(", 0, k) + len("printf(")
    out, i, depth = [], start, 1
    while i < len(text):
        c = text[i]
        if c == '"':
            j, lit = i + 1, []
            while text[j] != '"':
                if text[j] == "\\":
                    lit.append(text[j:j + 2])
                    j += 2
                    continue
                lit.append(text[j])
                j += 1
            out.append("".join(lit))
            i = j + 1
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1
    return "".join(out).replace("\\n", "\n")


def flags_in_text(text):
    text = re.sub(r"--\[no-\]([a-z0-9-]+)", r"--\1 --no-\1", text)
    return set(FLAG_RE.findall(text))


def exit_codes(text):
    """Literal exit codes: `return N;`, `return finish(N);`, `return cond ? N : M;` (either arm) and `rc = N;`."""
    codes = set()
    d = r"(?:finish\()?\s*(\d)\s*\)?"
    for m in re.finditer(r"\breturn\b([^;]*);", text):
        expr = m.group(1).strip()
        whole = re.fullmatch(d, expr)
        if whole:
            codes.add(int(whole.group(1)))
            continue
        tern = re.search(r"\?\s*(.+?)\s*:\s*(.+?)\s*$", expr)
        if tern:
            codes |= {int(x.group(1)) for x in (re.fullmatch(d, tern.group(1)), re.fullmatch(d, tern.group(2))) if x}
    codes |= {int(x) for x in re.findall(r"\brc\s*=\s*(\d)\s*;", text)}
    return codes


# ------------------------------------------------------------------------------------------------- document model
def doc_sections(text):
    secs, name = {}, None
    for line in text.splitlines():
        if line.startswith("## "):
            name = line[3:].strip()
            secs[name] = []
        elif name:
            secs[name].append(line)
    return secs


def table_first_cells(lines):
    rows = [ln for ln in lines if ln.startswith("| ") and not ln.startswith("| :")]
    return [r.split(" | ")[0][2:] for r in rows[1:]]  # skip the header row


def doc_flags(lines):
    return {f for cell in table_first_cells(lines) for f in re.findall(r"`(--?[a-z][a-z0-9-]*)", cell)}


def doc_exit_line(lines):
    text = " ".join(lines)
    m = re.search(r"Exit codes: (.*?)(?:\.\s*$|\.\s{2}|$)", text)
    assert m, "no 'Exit codes:' line"
    return {int(x) for x in re.findall(r"(?:^|, )(\d) ", m.group(1))}


SECTIONS = {  # doc section -> (source file, parser function signature, compared variable, usage marker)
    "`vegajet inspect`": ("cli/inspect.cpp", "int run_inspect(", "a", "Usage: vegajet inspect"),
    "`vegajet bench` and `vegajet run`": ("cli/bench.cpp", "int run_bench(", "a", "Usage: vegajet %s"),
    "`vegajet doctor`": ("cli/main.cpp", "int run_doctor(", "arg", "Usage: %s doctor"),
    "`vegajet soak`": ("cli/soak.cpp", "int run_soak(", "a", "Usage: vegajet soak"),
    "`vegajet serve`": ("cli/serve_cmd.cpp", "int parse_serve_args(", "a", "Usage: vegajet serve"),
    "`vegajet top`": ("cli/tui_top.cpp", "int run_top(", "a", "Usage: vegajet top"),
    "`vegajet chat`": ("cli/chat_repl.cpp", "int run_chat(", "a", "Usage: vegajet chat"),
}
EXIT_SECTIONS = {  # source file -> doc sections whose exit lines together cover it
    "cli/inspect.cpp": ["`vegajet inspect`"],
    "cli/bench.cpp": ["`vegajet plan`", "`vegajet bench` and `vegajet run`"],
    "cli/soak.cpp": ["`vegajet soak`"],
    "cli/serve_cmd.cpp": ["`vegajet serve`"],
    "cli/tui_top.cpp": ["`vegajet top`"],
    "cli/chat_repl.cpp": ["`vegajet chat`"],
}
ENV_DIRS = ("core", "models", "cli", "ops")


def code_model():
    m = {"flags": {}, "usage": {}, "exit": {}}
    for sec, (path, sig, var, marker) in SECTIONS.items():
        text = src(path)
        m["flags"][sec] = parsed_flags(body(text, sig), var) - HELP
        m["usage"][sec] = flags_in_text(usage_literals(text, marker)) - HELP
    main = src("cli/main.cpp")
    serve = src("cli/serve_cmd.cpp")
    m["global"] = {f for f in re.findall(r'\bcmd\s*==\s*"(--?[a-z]+)"', main)} | set(
        re.findall(r'args\[i\]\s*==\s*"(--[a-z]+)"', body(serve, "int extract_gpu_option(")))
    m["subcommands"] = {s for s in re.findall(r'\bcmd\s*==\s*"([a-z]+)"', main)}
    env, opaque, values = set(), [], {}
    for d in ENV_DIRS:
        for f in sorted((ROOT / d).rglob("*")):
            if f.suffix not in (".cpp", ".hpp", ".h", ".cc") or "tests" in f.parts:
                continue
            t = f.read_text(errors="replace")
            env |= set(re.findall(r'getenv\(\s*"([A-Za-z0-9_]+)"\s*\)', t))
            env |= set(re.findall(r'\bkEnvVar\s*=\s*"([A-Z0-9_]+)"', t))
            for name, ch in re.findall(r'getenv\(\s*"([A-Z0-9_]+)"\s*\)\s*\[0\]\s*==\s*\'(.)\'', t):
                values.setdefault(name, set()).add(ch)
            for name in re.findall(r'getenv\(\s*([A-Za-z_][A-Za-z0-9_:]*)\s*\)', t):
                if name.split("::")[-1] != "kEnvVar":
                    opaque.append("%s: getenv(%s)" % (f.relative_to(ROOT), name))
    m["env"], m["env_opaque"], m["env_values"] = env, opaque, values
    for path in list(EXIT_SECTIONS) + ["cli/main.cpp"]:
        m["exit"][path] = exit_codes(src(path))
    return m


def drift(doc_text, m):
    """Every mismatch between a CLI_REFERENCE.md text and the code model, as human-readable strings."""
    out, secs = [], doc_sections(doc_text)

    def cmp(what, code, doc):
        for x in sorted(code - doc):
            out.append("%s: %s is in the code but not documented" % (what, x))
        for x in sorted(doc - code):
            out.append("%s: %s is documented but not in the code" % (what, x))

    for sec in SECTIONS:
        if sec not in secs:
            out.append("missing doc section %s" % sec)
            continue
        cmp(sec, m["flags"][sec], doc_flags(secs[sec]) - HELP)
        cmp(sec + " usage text", m["flags"][sec], m["usage"][sec] - {"--gpu"})
    cmp("global options", m["global"], doc_flags(secs.get("Global options", [])))
    first = doc_text.split("\n## ", 1)[0]
    doc_subs = {re.sub(r"[`\s]", "", c) for c in table_first_cells(first.splitlines())}
    cmp("subcommands", m["subcommands"] | {"inspect"}, doc_subs)
    doc_env = {re.sub(r"`", "", c).strip() for c in table_first_cells(secs.get("Environment variables", []))}
    cmp("environment", m["env"], doc_env)
    out += ["environment: %s reads a non-literal name" % o for o in m["env_opaque"]]
    rows = {re.sub(r"`", "", r.split(" | ")[0][2:]).strip(): r
            for r in secs.get("Environment variables", []) if r.startswith("| `")}
    for name, chars in sorted(m["env_values"].items()):  # the code tests getenv(name)[0] == 'c': the row names `c`
        for ch in sorted(chars):
            if name in rows and "`%s`" % ch not in rows[name]:
                out.append("environment: %s is tested for '%s' but its row never mentions `%s`" % (name, ch, ch))
    table = {int(c) for c in table_first_cells(secs.get("Exit codes", [])) if c.strip().isdigit()}
    cmp("exit-code table", set().union(*m["exit"].values()), table)
    for path, names in EXIT_SECTIONS.items():
        documented = set().union(*(doc_exit_line(secs[n]) for n in names if n in secs))
        for c in sorted(m["exit"][path] - documented):
            out.append("%s returns %d, not listed in %s" % (path, c, " / ".join(names)))
    return out


class CliReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT / "cli/main.cpp").exists() or not (ROOT / "cli/serve_cmd.cpp").exists():
            raise unittest.SkipTest("needs the patched source tree (cli/main.cpp with serve_real_model.patch)")
        cls.doc = DOC.read_text()
        cls.m = code_model()

    def test_doc_matches_code(self):
        self.assertEqual(drift(self.doc, self.m), [])

    def test_model_is_not_vacuous(self):
        m = self.m
        self.assertGreaterEqual(len(m["flags"]["`vegajet bench` and `vegajet run`"]), 50)
        self.assertIn("--json-out", m["flags"]["`vegajet bench` and `vegajet run`"])
        self.assertIn("--mock", m["flags"]["`vegajet serve`"])
        self.assertEqual(m["global"], {"--help", "-h", "--version", "-v", "--gpu"})
        self.assertEqual(m["subcommands"], {"plan", "bench", "run", "soak", "doctor", "serve", "inspect", "top", "chat"})
        self.assertIn("--request-log", m["flags"]["`vegajet serve`"])
        self.assertEqual(m["flags"]["`vegajet chat`"], {"--url", "--model", "--system", "--max-tokens", "--temperature", "--no-stream"})
        for v in ("VEGAJET_ENABLE_SUPER", "VJ_IO_URING", "VJ_GPU_DEVICES"):
            self.assertIn(v, m["env"])
        self.assertEqual(set().union(*m["exit"].values()), {0, 1, 2, 3})

    def test_detects_drift(self):
        doc = self.doc.replace("| `--mock` | CPU mock engine (no model, no GPU). |\n", "")
        doc = doc.replace("| `--all-keys` |", "| `--all-keys` | x |\n| `--frobnicate` |")
        doc = doc.replace("| `VJ_IO_URING` |", "| `VJ_NOT_READ` | x | x |\n| `VJ_IO_URING_X` |")
        doc = doc.replace("Exit codes: 0 soak passed, 1 data-pattern errors found, 2 usage error, 3", "Exit codes: 0 soak passed, 1 data-pattern errors found, 2 usage error, 4")
        doc = doc.replace("| 3 | GPU backend", "| 4 | GPU backend")
        doc = doc.replace("| `VEGAJET_NO_AUTO_DOT8` | Ornith GPU bench | `1` disables", "| `VEGAJET_NO_AUTO_DOT8` | Ornith GPU bench | Set to disable")
        d = "\n".join(drift(doc, self.m))
        for needle in ("--mock is in the code but not documented", "--frobnicate is documented but not in the code",
                       "VJ_NOT_READ is documented", "VJ_IO_URING is in the code", "cli/soak.cpp returns 3",
                       "exit-code table: 3 is in the code", "VEGAJET_NO_AUTO_DOT8 is tested for '1'"):
            self.assertIn(needle, d)
        m = dict(self.m, usage=dict(self.m["usage"]))
        m["usage"]["`vegajet serve`"] = m["usage"]["`vegajet serve`"] - {"--slots"}
        self.assertIn("usage text: --slots is in the code", "\n".join(drift(self.doc, m)))

    @unittest.skipUnless(BIN and Path(BIN).exists(), "VEGAJET_BIN not set")
    def test_live_help(self):
        def help_of(*args):
            p = subprocess.run([BIN, *args, "--help"], capture_output=True, text=True, timeout=30)
            self.assertEqual(p.returncode, 0, args)
            return p.stdout + p.stderr
        top = help_of()
        for s in self.m["subcommands"] | {"--gpu", "VJ_GPU_DEVICES", "VJ_IO_URING", "VEGAJET_ENABLE_SUPER", "Exit codes"}:
            self.assertIn(s, top)
        for sub, sec in (("inspect", "`vegajet inspect`"), ("doctor", "`vegajet doctor`"), ("serve", "`vegajet serve`"),
                         ("top", "`vegajet top`"), ("chat", "`vegajet chat`")):
            self.assertEqual(flags_in_text(help_of(sub)) - HELP - {"--gpu"}, self.m["flags"][sec], sub)
        self.assertIn("bench --dry-run", help_of("plan"))


if __name__ == "__main__":
    unittest.main()
