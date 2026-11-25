#!/bin/sh
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Verify a VegaJet bundle against its SHA256SUMS manifest.
#
#   verify-bundle.sh [--dir DIR] [--quiet]
#
# DIR defaults to the directory holding this script. Every manifest entry must exist and match; every regular
# file in the tree must be listed (extra files are an integrity failure). Paths are relative and cannot escape
# DIR. Stable diagnostic names: E_USAGE, E_MANIFEST_MISSING, E_INTEGRITY, E_DEPENDENCY, E_EXTRA_FILE.

set -u

PROG=vegajet-verify
die() {
    _name=$1
    _rc=$2
    shift 2
    printf '%s: %s: %s\n' "$PROG" "$_name" "$*" >&2
    exit "$_rc"
}

self=$0
case $self in
    */*) ;;
    *) self=$(command -v "$self" 2>/dev/null) || die E_USAGE 2 "cannot find '$0' on PATH" ;;
esac
resolved=$(readlink -f -- "$self" 2>/dev/null) || resolved=$self
default_dir=$(CDPATH= cd -- "$(dirname -- "$resolved")" 2>/dev/null && pwd -P) || die E_USAGE 2 "cannot resolve script directory"

dir=$default_dir
quiet=0
while [ $# -gt 0 ]; do
    case $1 in
        --dir)
            [ $# -ge 2 ] || die E_USAGE 2 "--dir needs a directory"
            dir=$2
            shift 2
            ;;
        --dir=*)
            dir=${1#--dir=}
            shift
            ;;
        --quiet|-q) quiet=1; shift ;;
        --help|-h)
            printf 'usage: %s [--dir DIR] [--quiet]\n' "$PROG"
            exit 0
            ;;
        *) die E_USAGE 2 "unknown option: $1" ;;
    esac
done

[ -d "$dir" ] || die E_USAGE 2 "not a directory: $dir"
dir=$(CDPATH= cd -- "$dir" && pwd -P) || die E_USAGE 2 "cannot resolve: $dir"
manifest=$dir/SHA256SUMS
[ -f "$manifest" ] || die E_MANIFEST_MISSING 1 "no SHA256SUMS in $dir"
command -v sha256sum >/dev/null 2>&1 || die E_DEPENDENCY 1 "sha256sum is required"

expected=0
while IFS= read -r line || [ -n "$line" ]; do
    [ -n "$line" ] || continue
    case $line in
        *"  "*) path=${line#*  } ;;
        *) die E_MANIFEST_MISSING 1 "malformed manifest line: $line" ;;
    esac
    case $path in
        /*|*..*) die E_MANIFEST_MISSING 1 "unsafe manifest path: $path" ;;
    esac
    expected=$((expected + 1))
done < "$manifest"

actual=$(find "$dir" -type f ! -path "$manifest" -print 2>/dev/null | wc -l | tr -d ' ')
if [ "$actual" != "$expected" ]; then
    if [ "$actual" -gt "$expected" ]; then
        die E_EXTRA_FILE 1 "$dir has $actual files but the manifest lists $expected"
    fi
    die E_INTEGRITY 1 "$dir has $actual files but the manifest lists $expected"
fi

report=$(cd "$dir" && sha256sum -c SHA256SUMS 2>&1) || {
    failed=$(printf '%s\n' "$report" | grep -E 'FAILED$|No such file|No such file or directory' | head -n 5)
    die E_INTEGRITY 1 "${failed:-checksum verification failed}"
}

[ "$quiet" = 1 ] || printf '%s: bundle integrity ok: %s files (%s)\n' "$PROG" "$expected" "$dir"
exit 0
