#!/usr/bin/env bash
# Shared helpers for scripts/reference/*.sh (sourced, not executed).

BISCUIT_FC_SHA256=e70aea6290ca3312a290618a8092b8992444bc47a7d3b42680109dfd714d4329

die() { echo "error: $*" >&2; exit 1; }

# run CMD...: execute, or print "+ CMD" when DRY=1 (dry runs never touch the filesystem)
run() {
    if [ "${DRY:-0}" = 1 ]; then
        printf '+'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

now_s() {
    local t
    t=$(date +%s.%N)
    case $t in *N) t=$(date +%s) ;; esac
    printf '%s' "$t"
}

# find_apptainer: echo apptainer or singularity (honours $APPTAINER); fails if neither exists
find_apptainer() {
    if [ -n "${APPTAINER:-}" ]; then
        printf '%s' "$APPTAINER"
    elif command -v apptainer >/dev/null 2>&1; then
        printf apptainer
    elif command -v singularity >/dev/null 2>&1; then
        printf singularity
    else
        return 1
    fi
}

# verify_sif SIF EXPECTED_SHA: refuse a wrong image (VERIFY=0 skips, DRY=1 only prints)
verify_sif() {
    local sif=$1 want=${2:-$BISCUIT_FC_SHA256}
    if [ "${DRY:-0}" = 1 ]; then
        echo "+ verify sha256($sif) == $want"
        return 0
    fi
    [ -f "$sif" ] || die "image not found: $sif"
    [ "${VERIFY:-1}" = 1 ] || { echo "warning: image hash not verified" >&2; return 0; }
    local got
    got=$(sha256_of "$sif")
    [ "$got" = "$want" ] || die "image sha256 $got != expected $want"
}
