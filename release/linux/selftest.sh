#!/usr/bin/env bash
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Acceptance checks for a staged VegaJet bundle. Read-only with respect to the bundle: every check runs on a
# throwaway copy, so this is safe to point at an installed tree. It does not build or download anything.
#
#   release/linux/selftest.sh --bundle DIR [--keep]
#
# Covers the packaged launcher, the integrity verifier (tamper and extra-file failures), the explicit GPU-asset
# refusal, the stream-test client harness against `serve --mock`, the no-sudo installer, and relocatability of
# the installed prefix. Exit 0 only when every check passes.

set -uo pipefail

PROG=vegajet-selftest
bundle=
keep=0
while [ $# -gt 0 ]; do
    case $1 in
        --bundle)
            [ $# -ge 2 ] || { printf '%s: E_USAGE: --bundle needs a directory\n' "$PROG" >&2; exit 2; }
            bundle=$2
            shift 2
            ;;
        --bundle=*) bundle=${1#--bundle=}; shift ;;
        --keep) keep=1; shift ;;
        --help|-h) printf 'usage: %s --bundle DIR [--keep]\n' "$PROG"; exit 0 ;;
        *) printf '%s: E_USAGE: unknown option: %s\n' "$PROG" "$1" >&2; exit 2 ;;
    esac
done

if [ -z "$bundle" ]; then
    self=$(readlink -f -- "$0")
    bundle=$(dirname -- "$self")
fi
[ -f "$bundle/SHA256SUMS" ] || { printf '%s: E_USAGE: not a bundle: %s\n' "$PROG" "$bundle" >&2; exit 2; }
bundle=$(CDPATH= cd -- "$bundle" && pwd -P)

tmp=$(mktemp -d "${TMPDIR:-/tmp}/vegajet-selftest.XXXXXX") || exit 1
if [ "$keep" = 1 ]; then
    printf '%s: workdir %s\n' "$PROG" "$tmp"
else
    trap 'rm -rf -- "$tmp"' EXIT
fi

failed=0
ok() { printf '%s: ok %s\n' "$PROG" "$1"; }
bad() { printf '%s: FAIL %s\n' "$PROG" "$1" >&2; failed=1; }

copy() {
    local dest=$tmp/$1
    cp -a "$bundle" "$dest" || { bad "copy $1"; return 1; }
    printf '%s' "$dest"
}

# 1. launcher pass-through --------------------------------------------------------------------------------
good=$(copy good) || exit 1
if out=$("$good/bin/vegajet" --version 2>&1) && printf '%s' "$out" | grep -q '^vegajet '; then
    ok "launcher --version ($out)"
else
    bad "launcher --version: $out"
fi

# 2. bundle self-check ------------------------------------------------------------------------------------
if out=$("$good/bin/vegajet" --bundle-check 2>&1) && printf '%s' "$out" | grep -q 'check: integrity ok'; then
    ok "bundle-check"
    printf '%s\n' "$out" | sed 's/^/    /'
else
    bad "bundle-check: $out"
fi

# 3. tampered file -> E_INTEGRITY -------------------------------------------------------------------------
tamper=$(copy tamper) || exit 1
printf '\ntampered\n' >> "$tamper/share/doc/vegajet/README.md"
if out=$("$tamper/verify-bundle.sh" --dir "$tamper" --quiet 2>&1); then
    bad "tamper not detected"
elif printf '%s' "$out" | grep -q 'E_INTEGRITY'; then
    ok "tamper -> E_INTEGRITY"
else
    bad "tamper error did not name E_INTEGRITY: $out"
fi

# 4. unlisted file -> E_EXTRA_FILE ------------------------------------------------------------------------
extra=$(copy extra) || exit 1
: > "$extra/UNLISTED"
if out=$("$extra/verify-bundle.sh" --dir "$extra" --quiet 2>&1); then
    bad "extra file not detected"
elif printf '%s' "$out" | grep -q 'E_EXTRA_FILE'; then
    ok "extra file -> E_EXTRA_FILE"
else
    bad "extra-file error did not name E_EXTRA_FILE: $out"
fi

# 5. missing executable -> E_BUNDLE_EXEC ------------------------------------------------------------------
broken=$(copy broken) || exit 1
rm -f "$broken/libexec/vegajet/vegajet"
if out=$("$broken/bin/vegajet" --bundle-check 2>&1); then
    bad "missing executable not detected"
elif printf '%s' "$out" | grep -q 'E_BUNDLE_EXEC'; then
    ok "missing executable -> E_BUNDLE_EXEC"
else
    bad "missing-executable error did not name E_BUNDLE_EXEC: $out"
fi

# 6. GPU policy -------------------------------------------------------------------------------------------
gpu_assets=$(sed -n 's/^gpu_assets: //p' "$good/BUNDLE_INFO.txt" | head -n 1)
if [ "$gpu_assets" = "unavailable" ]; then
    rc=0
    out=$("$good/bin/vegajet" run /nonexistent-model 2>&1) || rc=$?
    if [ "$rc" = 3 ] && printf '%s' "$out" | grep -q 'E_GPU_ASSETS_MISSING'; then
        ok "GPU command refused -> E_GPU_ASSETS_MISSING (exit 3)"
    else
        bad "GPU refusal expected exit 3 + E_GPU_ASSETS_MISSING, got exit $rc: $out"
    fi
    rc=0
    out=$("$good/bin/vegajet" --bundle-check --require-gpu 2>&1) || rc=$?
    if [ "$rc" = 3 ] && printf '%s' "$out" | grep -q 'E_GPU_ASSETS_MISSING'; then
        ok "--bundle-check --require-gpu -> exit 3"
    else
        bad "--require-gpu expected exit 3, got $rc"
    fi
else
    if out=$("$good/bin/vegajet" --bundle-check 2>&1) && printf '%s' "$out" | grep -q 'check: gpu-assets verified'; then
        ok "GPU assets verified by launcher"
    else
        bad "GPU assets not verified: $out"
    fi
fi

# 7. packaged stream-test harness -------------------------------------------------------------------------
if out=$("$good/bin/stream-test" --vegajet "$good/bin/vegajet" 2>&1) \
    && printf '%s' "$out" | grep -q 'PASS engine=mock' \
    && printf '%s' "$out" | grep -q 'no GPU inference'; then
    ok "stream-test mock SSE"
    printf '%s\n' "$out" | sed 's/^/    /'
else
    bad "stream-test: $out"
fi

# 8. no-sudo install and relocation -----------------------------------------------------------------------
prefix=$tmp/prefix
if out=$("$good/install.sh" --prefix "$prefix" 2>&1); then
    ok "install --prefix <temp> (no sudo)"
    installed=1
else
    bad "install: $out"
    installed=0
fi
if [ "$installed" = 1 ]; then
    if out=$("$prefix/bin/vegajet" --version 2>&1) && printf '%s' "$out" | grep -q '^vegajet '; then
        ok "installed launcher resolves through relative symlink"
    else
        bad "installed launcher: $out"
    fi
    moved=$tmp/moved-prefix
    if mv "$prefix" "$moved" && out=$("$moved/bin/vegajet" --version 2>&1) \
        && printf '%s' "$out" | grep -q '^vegajet '; then
        ok "prefix relocation"
    else
        bad "relocated prefix did not work: $out"
    fi
    if out=$("$moved/bin/stream-test" 2>&1) && printf '%s' "$out" | grep -q 'PASS engine=mock'; then
        ok "installed stream-test"
    else
        bad "installed stream-test: $out"
    fi
fi

if [ "$failed" = 0 ]; then
    printf '%s: PASS\n' "$PROG"
    exit 0
fi
printf '%s: FAIL\n' "$PROG" >&2
exit 1
