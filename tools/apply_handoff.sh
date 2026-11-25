#!/usr/bin/env bash
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Pillar 5: Zero-friction multi-agent HANDOFF.json consumer.
# Validates status=="READY", runs git apply --check on declared patches, copies declared files,
# and prints a concise 5-line apply status report.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    echo "Usage: tools/apply_handoff.sh <sidecar_dir> [--dry-run] [--require-base-sha SHA]"
    echo "       tools/apply_handoff.sh --self-test"
}

if [[ $# -lt 1 || "$1" == "--help" || "$1" == "-h" ]]; then
    usage
    exit 0
fi

if [[ "$1" == "--self-test" ]]; then
    TMP_DIR="$(mktemp -d /tmp/vj_handoff_selftest.XXXXXX)"
    trap 'rm -rf "${TMP_DIR}"' EXIT
    cat > "${TMP_DIR}/HANDOFF.json" <<'JSON'
{
  "status": "READY",
  "base_sha": "HEAD",
  "patches": [],
  "files": [],
  "gates_passed": ["ctest", "privacy_audit"],
  "summary": "self-test handoff payload"
}
JSON
    "${BASH_SOURCE[0]}" "${TMP_DIR}" --dry-run >/dev/null
    echo '{"tool":"apply_handoff","status":"SELF_TEST_PASS"}'
    exit 0
fi

SIDECAR_DIR="$1"
shift
DRY_RUN=0
REQ_BASE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --require-base-sha) REQ_BASE="$2"; shift 2 ;;
        *) echo "apply_handoff: unknown arg $1" >&2; exit 2 ;;
    esac
done

HANDOFF="${SIDECAR_DIR}/HANDOFF.json"
if [[ ! -f "${HANDOFF}" ]]; then
    echo "HANDOFF: MISSING (${HANDOFF})" >&2
    exit 1
fi

python3 - "${ROOT_DIR}" "${SIDECAR_DIR}" "${HANDOFF}" "${DRY_RUN}" "${REQ_BASE}" <<'PY'
import json, os, shutil, subprocess, sys

root_dir, sidecar_dir, handoff_path, dry_s, req_base = sys.argv[1:]
dry_run = (dry_s == "1")

with open(handoff_path, "r", encoding="utf-8") as f:
    h = json.load(f)

status = str(h.get("status", "UNKNOWN"))
base_sha = str(h.get("base_sha", ""))
patches = list(h.get("patches", []))
files = list(h.get("files", []))
gates = list(h.get("gates_passed", []))
summary = str(h.get("summary", ""))

if status != "READY":
    print(f"HANDOFF_STATUS : {status} (expected READY)")
    print(f"SUMMARY        : {summary}")
    sys.exit(3)

if req_base and base_sha and not base_sha.startswith(req_base) and not req_base.startswith(base_sha):
    print(f"HANDOFF_STATUS : BASE_MISMATCH (handoff={base_sha} required={req_base})")
    sys.exit(4)

patch_ok = 0
for p in patches:
    pp = os.path.join(sidecar_dir, p)
    if not os.path.isfile(pp):
        print(f"HANDOFF_ERROR  : missing patch {pp}", file=sys.stderr)
        sys.exit(1)
    r = subprocess.run(["git", "-C", root_dir, "apply", "--check", pp], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"HANDOFF_ERROR  : git apply --check failed for {p}: {r.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    patch_ok += 1

file_ok = 0
for item in files:
    if isinstance(item, dict):
        src_rel = item["src"]
        dst_rel = item.get("dst", src_rel)
    else:
        src_rel = str(item)
        dst_rel = src_rel
    src_abs = os.path.join(sidecar_dir, src_rel)
    if not os.path.isfile(src_abs):
        print(f"HANDOFF_ERROR  : missing file {src_abs}", file=sys.stderr)
        sys.exit(1)
    file_ok += 1

if not dry_run:
    for p in patches:
        pp = os.path.join(sidecar_dir, p)
        subprocess.run(["git", "-C", root_dir, "apply", pp], check=True)
    for item in files:
        if isinstance(item, dict):
            src_rel, dst_rel = item["src"], item.get("dst", item["src"])
        else:
            src_rel = dst_rel = str(item)
        dst_abs = os.path.join(root_dir, dst_rel)
        os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
        shutil.copy2(os.path.join(sidecar_dir, src_rel), dst_abs)

mode = "DRY_RUN_OK" if dry_run else "APPLIED"
print(f"HANDOFF_STATUS : {mode} (base_sha={base_sha})")
print(f"PATCHES        : {patch_ok}/{len(patches)} verified")
print(f"FILES          : {file_ok}/{len(files)} verified")
print(f"GATES_PASSED   : {','.join(str(g) for g in gates)}")
print(f"SUMMARY        : {summary}")
PY
