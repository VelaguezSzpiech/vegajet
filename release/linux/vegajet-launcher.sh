#!/bin/sh
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Relocatable launcher for an installed VegaJet bundle. The bundle creator installs this script as
# <bundle>/bin/vegajet, next to the packaged executable at <bundle>/libexec/vegajet/vegajet. A per-user install
# reaches it through <prefix>/bin/vegajet (a relative symlink), so every path is derived from the launcher's own
# resolved location. No environment variable, absolute prefix or sudo is required.
#
# What it does:
#   1. Resolves the bundle root from its own real path (follows symlinks).
#   2. Checks that the packaged executable is present and executable.
#   3. Checks shared-library dependencies by name (ldd) and names every missing library.
#   4. Verifies packaged gfx906 code objects (share/vegajet/hsaco) against their SHA-256 manifest before any
#      GPU command, and exports VEGAJET_HSACO_DIR only for a verified set. Without verified code objects GPU
#      commands are refused explicitly; the CPU tools and `serve --mock` keep working.
#   5. Executes the packaged executable with the caller's arguments.
#
# Stable diagnostic names: E_PREFIX, E_BUNDLE_EXEC, E_DEPENDENCY, E_GPU_ASSETS_MISSING,
# E_GPU_ASSETS_INTEGRITY, E_USAGE. Exit 3 means "GPU backend / GPU assets unavailable", matching the CLI.
#
# `vegajet --bundle-check [--require-gpu]` prints the same checks without executing the CLI.

set -u

PROG=vegajet-bundle

die() {
    _name=$1
    _rc=$2
    shift 2
    printf '%s: %s: %s\n' "$PROG" "$_name" "$*" >&2
    exit "$_rc"
}

# ---- locate the bundle root from this script's real path ------------------------------------------------
self=$0
case $self in
    */*) ;;
    *)
        self=$(command -v "$self" 2>/dev/null) || die E_PREFIX 1 "cannot find '$0' on PATH"
        ;;
esac
resolved=$(readlink -f -- "$self" 2>/dev/null) || resolved=$self
[ -f "$resolved" ] || die E_PREFIX 1 "launcher path does not exist: $resolved"
bin_dir=$(CDPATH= cd -- "$(dirname -- "$resolved")" 2>/dev/null && pwd -P) || die E_PREFIX 1 "cannot resolve launcher directory"
root=$(CDPATH= cd -- "$bin_dir/.." 2>/dev/null && pwd -P) || die E_PREFIX 1 "cannot resolve bundle root"
exe=$root/libexec/vegajet/vegajet
hsaco_bundle=$root/share/vegajet/hsaco

have_ldd=0
command -v ldd >/dev/null 2>&1 && have_ldd=1
have_sha256sum=0
command -v sha256sum >/dev/null 2>&1 && have_sha256sum=1

# Optional ROCm runtime directory for a GPU bundle. Kept opt-in: the bundle never guesses another install's libs.
if [ -n "${VEGAJET_ROCM_LIB:-}" ] && [ -d "${VEGAJET_ROCM_LIB:-}" ]; then
    LD_LIBRARY_PATH=$VEGAJET_ROCM_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    export LD_LIBRARY_PATH
fi

# Prints missing shared libraries (space separated) to stdout; returns 0 when the binary is loadable.
ldd_missing() {
    _out=$(ldd -- "$1" 2>&1) || true
    case $_out in
        *"not a dynamic executable"*|*"statically linked"*) return 0 ;;
        *"not a valid dynamic program"*|*"not an ELF"*) return 2 ;;
    esac
    printf '%s\n' "$_out" | sed -n 's/^[[:space:]]*\([^[:space:]]*\) => not found.*$/\1/p' | tr '\n' ' '
}

# ---- packaged gfx906 code objects ----------------------------------------------------------------------
gpu_state=unavailable
gpu_reason=""
gpu_dir=""
gpu_count=0

bundle_reason() {
    _reason_file=$root/share/vegajet/GPU_UNAVAILABLE.txt
    if [ -f "$_reason_file" ]; then
        _r=$(sed -n 's/^reason: //p' "$_reason_file" | head -n 1)
        [ -n "$_r" ] && printf '%s' "$_r" && return 0
    fi
    printf '%s' "no verified gfx906 code objects are packaged in this bundle"
}

# 0 = verified, 1 = no code objects, 2 = integrity failure. Sets gpu_err/gpu_count.
verify_hsaco_dir() {
    _d=$1
    gpu_err=""
    [ -d "$_d" ] || { gpu_err="not a directory: $_d"; return 1; }
    if [ "$have_sha256sum" != 1 ]; then
        gpu_err="sha256sum is required to verify packaged code objects"
        return 2
    fi
    _manifest=$_d/SHA256SUMS
    [ -f "$_manifest" ] || { gpu_err="missing code-object manifest: $_d/SHA256SUMS"; return 2; }
    _n=$(find "$_d" -maxdepth 1 -type f -name '*.hsaco' -print 2>/dev/null | wc -l | tr -d ' ')
    _m=$(grep -c '\.hsaco' "$_manifest" 2>/dev/null || true)
    [ "${_n:-0}" -gt 0 ] || { gpu_err="no code objects in $_d"; return 1; }
    [ "$_n" = "$_m" ] || {
        gpu_err="manifest lists $_m code objects but $_d holds $_n"
        return 2
    }
    if ! ( cd "$_d" && sha256sum --quiet -c SHA256SUMS >/dev/null 2>&1 ); then
        _bad=$( cd "$_d" && sha256sum --quiet -c SHA256SUMS 2>&1 | sed -n 's/: FAILED$//p' | head -n 3 | tr '\n' ' ')
        gpu_err="checksum mismatch: ${_bad:-unidentified code object}"
        return 2
    fi
    gpu_count=$_n
    return 0
}

resolve_gpu_assets() {
    if [ -d "$hsaco_bundle" ] || { [ -n "${VEGAJET_HSACO_DIR:-}" ] && [ -d "${VEGAJET_HSACO_DIR:-}" ]; }; then
        if [ "$have_sha256sum" != 1 ]; then
            die E_DEPENDENCY 1 "sha256sum is required to verify packaged code objects"
        fi
    fi
    gpu_reason=$(bundle_reason)
    if [ -d "$hsaco_bundle" ]; then
        verify_hsaco_dir "$hsaco_bundle"
        case $? in
            0) gpu_state=verified; gpu_dir=$hsaco_bundle; gpu_reason=""; return 0 ;;
            2) die E_GPU_ASSETS_INTEGRITY 3 "$gpu_err (bundle: $hsaco_bundle)" ;;
        esac
    fi
    # An operator-supplied directory is honored only when it carries its own code-object manifest.
    if [ -n "${VEGAJET_HSACO_DIR:-}" ] && [ "${VEGAJET_HSACO_DIR:-}" != "$hsaco_bundle" ]; then
        if [ -d "$VEGAJET_HSACO_DIR" ]; then
            verify_hsaco_dir "$VEGAJET_HSACO_DIR"
            case $? in
                0) gpu_state=verified-operator; gpu_dir=$VEGAJET_HSACO_DIR; gpu_reason=""; return 0 ;;
                2) die E_GPU_ASSETS_INTEGRITY 3 "$gpu_err (operator: $VEGAJET_HSACO_DIR)" ;;
            esac
        else
            gpu_reason="VEGAJET_HSACO_DIR is not a directory: $VEGAJET_HSACO_DIR"
        fi
    fi
    return 1
}

# ---- --bundle-check ------------------------------------------------------------------------------------
if [ $# -gt 0 ] && [ "$1" = "--bundle-check" ]; then
    shift
    require_gpu=0
    while [ $# -gt 0 ]; do
        case $1 in
            --require-gpu) require_gpu=1; shift ;;
            --help|-h)
                printf 'usage: vegajet --bundle-check [--require-gpu]\n'
                exit 0
                ;;
            *) die E_USAGE 2 "unknown option for --bundle-check: $1" ;;
        esac
    done
    rc=0
    printf 'check: bundle ok root=%s\n' "$root"
    _ver=$(sed -n 's/^version: //p' "$root/BUNDLE_INFO.txt" 2>/dev/null | head -n 1)
    [ -n "$_ver" ] && printf 'check: version=%s\n' "$_ver"
    if [ -x "$exe" ]; then
        printf 'check: binary ok\n'
    else
        printf 'check: fail E_BUNDLE_EXEC missing=%s\n' "$exe"
        rc=1
    fi
    if [ "$have_ldd" = 1 ] && [ -x "$exe" ]; then
        _missing=$(ldd_missing "$exe")
        _ldd_rc=$?
        if [ "$_ldd_rc" = 2 ]; then
            printf 'check: fail E_BUNDLE_EXEC not-a-loadable-executable\n'
            rc=1
        elif [ -n "$_missing" ]; then
            printf 'check: fail E_DEPENDENCY missing=%s\n' "$_missing"
            rc=1
        else
            printf 'check: dependencies ok\n'
        fi
    else
        printf 'check: dependencies skipped\n'
    fi
    if [ "$have_sha256sum" != 1 ]; then
        printf 'check: fail E_DEPENDENCY missing=sha256sum\n'
        rc=1
    elif [ -x "$root/verify-bundle.sh" ]; then
        if "$root/verify-bundle.sh" --dir "$root" --quiet >/dev/null 2>&1; then
            printf 'check: integrity ok\n'
        else
            printf 'check: fail E_INTEGRITY\n'
            rc=1
        fi
    else
        printf 'check: integrity skipped\n'
    fi
    resolve_gpu_assets || true
    case $gpu_state in
        verified|verified-operator)
            printf 'check: gpu-assets verified files=%s source=%s\n' "$gpu_count" "$gpu_dir"
            printf 'check: mode gpu\n'
            ;;
        *)
            printf 'check: gpu-assets unavailable reason=%s\n' "$gpu_reason"
            printf 'check: mode cpu\n'
            if [ "$require_gpu" = 1 ]; then
                printf 'check: fail E_GPU_ASSETS_MISSING reason=%s\n' "$gpu_reason"
                rc=3
            fi
            ;;
    esac
    exit "$rc"
fi

# ---- normal invocation ---------------------------------------------------------------------------------
[ -x "$exe" ] || die E_BUNDLE_EXEC 1 "packaged executable is missing or not executable: $exe"

if [ "$have_ldd" = 1 ]; then
    missing=$(ldd_missing "$exe")
    case $? in
        2) die E_BUNDLE_EXEC 1 "packaged executable is not a loadable Linux binary: $exe" ;;
    esac
    if [ -n "$missing" ]; then
        die E_DEPENDENCY 1 "missing shared libraries: $missing"
    fi
fi

gpu_mode=0
helpish=0
for arg in "$@"; do
    case $arg in --help|-h) helpish=1 ;; esac
done
if [ "$helpish" = 0 ] && [ $# -ge 2 ]; then
    case $1 in
        run|soak) gpu_mode=1 ;;
        bench)
            gpu_mode=1
            for arg in "$@"; do [ "$arg" = "--dry-run" ] && gpu_mode=0; done
            ;;
        serve)
            gpu_mode=1
            for arg in "$@"; do [ "$arg" = "--mock" ] && gpu_mode=0; done
            ;;
    esac
fi

if [ "$gpu_mode" = 1 ]; then
    resolve_gpu_assets || true
    case $gpu_state in
        verified|verified-operator)
            VEGAJET_HSACO_DIR=$gpu_dir
            export VEGAJET_HSACO_DIR
            ;;
        *)
            printf '%s: E_GPU_ASSETS_MISSING: %s\n' "$PROG" "$gpu_reason" >&2
            printf '%s: this bundle cannot run GPU commands; the CPU tools and "serve --mock" still work.\n' "$PROG" >&2
            exit 3
            ;;
    esac
fi

exec "$exe" "$@"
