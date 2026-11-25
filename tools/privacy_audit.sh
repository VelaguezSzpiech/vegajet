#!/usr/bin/env bash
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Release leak audit: fails if the working tree or the full git history (git log -p) contains personal
# identifiers, private mount paths, PCI addresses, firmware/model file references or copyleft headers.
# Patterns are assembled from fragments so this script does not match itself.
# The models-mount pattern only matches an absolute path (not preceded by a path character), so the repository's own
# models/ directory and the public OpenAI-compatible model-list route are not findings.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
home='/ho''me/'
mount='/mn''t/'
bdf='(^|[^[:xdigit:]])[[:xdigit:]]{4}:[[:xdigit:]]{2}:[[:xdigit:]]{2}\.[0-7]([^[:xdigit:]]|$)'
models='(^|[^[:alnum:]_./-])/mod''els([/"'"'"' ]|$)'
rom='\.r''om([^[:alnum:]]|$)'
gguf='\.gg''uf([^[:alnum:]]|$)'
copyleft_header='SPDX-License-Identifier:.*(A?G''PL|LG''PL)-'
re="${home}|${mount}|${models}|${bdf}|${rom}|${gguf}|${copyleft_header}"
# Materialize inputs so a Git failure or SIGPIPE cannot masquerade as a clean scan.
tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT
git ls-files -co --exclude-standard -z > "$tmp/files"
git log --no-color -p --all --format='commit %H%n%an <%ae>%n%cn <%ce>%n%B' > "$tmp/history"
# Optional newline-separated private literals supplied by the release owner, never committed.
scan() {
    local term rc
    while IFS= read -r term; do
        [[ -n $term ]] || continue
        if LC_ALL=C grep -Fq -- "$term" "$1"; then echo 'private literal matched'; fail=1; else
            rc=$?; [[ $rc == 1 ]] || exit "$rc"
        fi
    done <<< "${VJ_PRIVATE_TERMS:-}"
    if LC_ALL=C grep -nIE -- "$re" "$1"; then fail=1; else
        rc=$?; [[ $rc == 1 ]] || exit "$rc"
    fi
}
fail=0
echo '== working tree'
while IFS= read -r -d '' path; do
    [[ -f "$path" && ! -L "$path" ]] || { echo "unexpected non-file: $path"; exit 1; }
    scan "$path"
done < "$tmp/files"
echo '== git history, messages and authors'
scan "$tmp/history"
if [[ $fail == 0 ]]; then echo 'PRIVACY AUDIT: PASS'; else echo 'PRIVACY AUDIT: FAIL'; exit 1; fi
