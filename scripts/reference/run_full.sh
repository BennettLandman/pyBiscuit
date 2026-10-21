#!/usr/bin/env bash
# B1.1: run biscuit_FC_v2.2.sif on a MaCRUISE v3 input directory with intermediates kept,
# optional PATH-shim command tracing (B1.2), and an archive plus SHA-256 manifest.
#
# Never run this on macOS: the image is x86-64 Linux. See docs/reference-artifacts.md.
#
#   run_full.sh --sif IMG --input DIR --out DIR [--license FILE] [--reg mb101|oasis45|...]
#               [--surf-recon macruise] [--thread N] [--no-trace] [--strace]
#               [--extra "run_biscuit options"] [--sif-sha256 HEX | --no-verify] [--dry-run]
#
# Outputs (WORK = <out>.harness):
#   <out>/                 biscuit output tree including .tmp (run_biscuit's `rm -rf $TMP` removed)
#   WORK/trace/            one JSON record + stdout/stderr per sub-tool call
#   WORK/trace.jsonl       the records merged and sorted by start time
#   WORK/run.json          image hash, arguments, host, timing, exit code
#   WORK/MANIFEST.sha256   SHA-256 of every file in <out> and WORK/trace*
#   WORK/<name>.tar.gz     archive of all of the above
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/reference/_lib.sh
. "$here/_lib.sh"

sif="" in="" out="" license="" reg=mb101 recon=macruise thread="" extra="" want=$BISCUIT_FC_SHA256
trace=1 strace=0
DRY=0 VERIFY=1
while [ $# -gt 0 ]; do
    case $1 in
        --sif) sif=$2; shift 2 ;;
        --input) in=$2; shift 2 ;;
        --out) out=$2; shift 2 ;;
        --license) license=$2; shift 2 ;;
        --reg) reg=$2; shift 2 ;;
        --surf-recon) recon=$2; shift 2 ;;
        --thread) thread=$2; shift 2 ;;
        --extra) extra=$2; shift 2 ;;
        --sif-sha256) want=$2; shift 2 ;;
        --no-verify) VERIFY=0; shift ;;
        --no-trace) trace=0; shift ;;
        --strace) strace=1; shift ;;
        --dry-run) DRY=1; shift ;;
        -h | --help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) die "unknown option $1" ;;
    esac
done
[ -n "$sif" ] && [ -n "$in" ] && [ -n "$out" ] || die "--sif, --input and --out are required"
export DRY VERIFY

if [ "$DRY" = 1 ]; then
    app=${APPTAINER:-apptainer}
else
    app=$(find_apptainer) || die "neither apptainer nor singularity on PATH (Linux x86-64 host required)"
    [ "$(uname -s)" = Linux ] || [ -n "${APPTAINER:-}" ] || die "run on a Linux host (set APPTAINER only for the mock test)"
    [ -d "$in" ] || die "input dir not found: $in"
    [ -z "$license" ] || [ -f "$license" ] || die "licence file not found: $license"
    in=$(cd "$in" && pwd)
    mkdir -p "$out"
    out=$(cd "$out" && pwd)
fi
work="${out%/}.harness"
name=$(basename "${out%/}")

verify_sif "$sif" "$want"
run mkdir -p "$work/shims" "$work/trace" "$work/home"
if [ "$trace" = 1 ]; then
    shim_args=()
    [ "$DRY" != 1 ] || shim_args+=(--dry-run)
    "$here/install_shims.sh" "${shim_args[@]+"${shim_args[@]}"}" "$work/shims"
fi

run cp "$here/container_inner.sh" "$work/run_inner.sh"
run chmod 755 "$work/run_inner.sh"

binds=(-B "$in:/INPUTS:ro" -B "$out:/OUTPUTS" -B "$work:/harness")
[ -z "$license" ] || binds+=(-B "$license:/opt/biscuit/license.txt:ro")
envs=(--env "REG=$reg" --env "SURF_RECON=$recon" --env "TRACE=$trace" --env "STRACE=$strace"
    --env "THREAD=$thread" --env "EXTRA=$extra" --env "BISCUIT_TRACE_DIR=/harness/trace")
cmd=("$app" exec --cleanenv --home "$work/home" "${binds[@]}" "${envs[@]}" "$sif" bash /harness/run_inner.sh)

t0=$(now_s)
rc=0
if [ "$DRY" = 1 ]; then
    run "${cmd[@]}"
else
    "${cmd[@]}" || rc=$?
fi
t1=$(now_s)

echo "+ merge trace records -> $work/trace.jsonl"
if [ "$DRY" != 1 ]; then
    : >"$work/trace.jsonl"
    # shellcheck disable=SC2012
    for f in $(ls "$work/trace"/*.json 2>/dev/null | sort); do
        cat "$f" >>"$work/trace.jsonl"
    done
    cat >"$work/run.json" <<JSON
{"sif": "$(basename "$sif")", "sif_sha256": "$(sha256_of "$sif")",
 "reg": "$reg", "surf_recon": "$recon", "thread": "$thread", "extra": "$extra",
 "trace": $trace, "start": $t0, "end": $t1, "exit": $rc,
 "host": "$(hostname)", "uname": "$(uname -srm)", "runtime": "$("$app" --version 2>&1 | head -1)"}
JSON
    (
        cd "$(dirname "$out")"
        find "$name" "$name.harness/trace" "$name.harness/trace.jsonl" "$name.harness/run.json" -type f 2>/dev/null |
            sort | while IFS= read -r f; do printf '%s  %s\n' "$(sha256_of "$f")" "$f"; done \
            >"$work/MANIFEST.sha256"
        tar -czf "$work/$name.tar.gz" "$name" "$name.harness/trace" "$name.harness/trace.jsonl" \
            "$name.harness/run.json" "$name.harness/MANIFEST.sha256" "$name.harness/run_inner.sh"
    )
    echo "archive: $work/$name.tar.gz ($(wc -c <"$work/$name.tar.gz") bytes)"
    echo "manifest: $work/MANIFEST.sha256 ($(wc -l <"$work/MANIFEST.sha256") files)"
else
    echo "+ write run.json, MANIFEST.sha256 (sha256 of every file in $out and trace), tar -czf $work/$name.tar.gz"
fi
exit "$rc"
