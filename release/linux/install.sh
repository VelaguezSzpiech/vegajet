#!/bin/sh
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Install a locally built VegaJet bundle for the current user. No sudo, no root-owned files, no network.
#
#   ./install.sh [--prefix DIR] [--force] [--dry-run] [--verify-only]
#
# Default prefix: $HOME/.local. The bundle is copied verbatim into <prefix>/opt/vegajet-<version>, then
# <prefix>/bin/{vegajet,stream-test,vegajet-verify} are pointed at it with *relative* symlinks, so the whole
# prefix can be moved. The copy is verified against its own SHA256SUMS before and after installation.
#
# Stable diagnostic names: E_USAGE, E_PREFIX, E_MANIFEST_MISSING, E_INTEGRITY, E_WRITE, E_CHECK, E_DEPENDENCY.

set -u

PROG=vegajet-install
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
root=$(CDPATH= cd -- "$(dirname -- "$resolved")" 2>/dev/null && pwd -P) || die E_USAGE 2 "cannot resolve bundle root"

prefix=${HOME:-}/.local
force=0
dry_run=0
verify_only=0
while [ $# -gt 0 ]; do
    case $1 in
        --prefix)
            [ $# -ge 2 ] || die E_USAGE 2 "--prefix needs a directory"
            prefix=$2
            shift 2
            ;;
        --prefix=*)
            prefix=${1#--prefix=}
            shift
            ;;
        --force) force=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        --verify-only) verify_only=1; shift ;;
        --help|-h)
            printf 'usage: %s [--prefix DIR] [--force] [--dry-run] [--verify-only]\n' "$PROG"
            exit 0
            ;;
        *) die E_USAGE 2 "unknown option: $1" ;;
    esac
done

[ -f "$root/SHA256SUMS" ] || die E_MANIFEST_MISSING 1 "not a VegaJet bundle (no SHA256SUMS): $root"
[ -x "$root/verify-bundle.sh" ] || die E_MANIFEST_MISSING 1 "bundle has no verify-bundle.sh"
[ -f "$root/BUNDLE_INFO.txt" ] || die E_MANIFEST_MISSING 1 "bundle has no BUNDLE_INFO.txt"

version=$(sed -n 's/^version: //p' "$root/BUNDLE_INFO.txt" | head -n 1)
[ -n "$version" ] || die E_MANIFEST_MISSING 1 "BUNDLE_INFO.txt has no version line"

"$root/verify-bundle.sh" --dir "$root" --quiet || die E_INTEGRITY 1 "bundle did not verify; refusing to install"

if [ "$verify_only" = 1 ]; then
    printf '%s: bundle verified: version=%s root=%s\n' "$PROG" "$version" "$root"
    exit 0
fi

"$root/bin/vegajet" --bundle-check >/dev/null 2>&1 \
    || die E_CHECK 1 "bundle launcher failed --bundle-check; refusing to install"

[ -n "${HOME:-}" ] || [ "$prefix" != "/.local" ] || die E_PREFIX 2 "HOME is not set; pass --prefix DIR"
case $prefix in
    /*) ;;
    *) die E_PREFIX 2 "prefix must be an absolute path: $prefix" ;;
esac
case $version in
    *[!A-Za-z0-9._+-]*) die E_MANIFEST_MISSING 1 "unsafe version string: $version" ;;
esac
requested_prefix=$prefix
if [ -d "$requested_prefix" ]; then
    prefix=$(CDPATH= cd -- "$requested_prefix" && pwd -P) || die E_PREFIX 2 "cannot resolve prefix: $requested_prefix"
elif [ "$dry_run" = 1 ]; then
    prefix=${requested_prefix%/}
else
    mkdir -p "$requested_prefix" 2>/dev/null || die E_WRITE 1 "cannot create prefix: $requested_prefix"
    prefix=$(CDPATH= cd -- "$requested_prefix" && pwd -P) || die E_WRITE 1 "cannot resolve prefix: $requested_prefix"
fi

target=$prefix/opt/vegajet-$version
bin=$prefix/bin
case "$target" in
    "$root"/*) die E_PREFIX 2 "prefix would place the bundle inside itself: $target" ;;
esac

if [ -e "$target" ] && [ "$force" != 1 ] && [ "$root" != "$target" ]; then
    die E_PREFIX 1 "already installed at $target (use --force to replace)"
fi

if [ "$dry_run" = 1 ]; then
    printf '%s: would install version=%s\n' "$PROG" "$version"
    printf '%s:   copy %s -> %s\n' "$PROG" "$root" "$target"
    printf '%s:   link %s -> %s\n' "$PROG" "$bin/vegajet" "../opt/vegajet-$version/bin/vegajet"
    printf '%s:   link %s -> %s\n' "$PROG" "$bin/stream-test" "../opt/vegajet-$version/bin/stream-test"
    printf '%s:   link %s -> %s\n' "$PROG" "$bin/vegajet-verify" "../opt/vegajet-$version/verify-bundle.sh"
    exit 0
fi

mkdir -p "$prefix/opt" "$bin" 2>/dev/null || die E_WRITE 1 "cannot create $prefix/opt and $bin"

if [ "$root" != "$target" ]; then
    staging=$target.tmp.$$
    rm -rf "$staging" 2>/dev/null || true
    cp -R "$root" "$staging" || { rm -rf "$staging"; die E_WRITE 1 "cannot copy bundle into $staging"; }
    chmod 755 "$staging" 2>/dev/null || true
    rm -rf "$target" 2>/dev/null || true
    mv "$staging" "$target" || { rm -rf "$staging"; die E_WRITE 1 "cannot move bundle into $target"; }
fi

ln -sfn "../opt/vegajet-$version/bin/vegajet" "$bin/vegajet" || die E_WRITE 1 "cannot link $bin/vegajet"
ln -sfn "../opt/vegajet-$version/bin/stream-test" "$bin/stream-test" || die E_WRITE 1 "cannot link $bin/stream-test"
ln -sfn "../opt/vegajet-$version/verify-bundle.sh" "$bin/vegajet-verify" || die E_WRITE 1 "cannot link $bin/vegajet-verify"

"$target/verify-bundle.sh" --dir "$target" --quiet || die E_INTEGRITY 1 "installed tree did not verify"

if ! "$bin/vegajet" --bundle-check >/dev/null 2>&1; then
    die E_CHECK 1 "installed launcher failed --bundle-check; run $bin/vegajet --bundle-check"
fi

printf '%s: installed version=%s prefix=%s\n' "$PROG" "$version" "$prefix"
printf '%s:   %s\n' "$PROG" "$bin/vegajet"
printf '%s:   %s\n' "$PROG" "$bin/stream-test"
case :$PATH: in
    *:$bin:*) ;;
    *) printf '%s: note: add %s to PATH to run vegajet and stream-test by name\n' "$PROG" "$bin" ;;
esac
printf '%s: GPU commands are available only when the bundle carries gfx906 code objects verified by the build;\n' "$PROG"
printf '%s: otherwise they stop with E_GPU_ASSETS_MISSING and the CPU tools keep working.\n' "$PROG"
