#!/usr/bin/env bash
# Driver for the pyBiscuit image-job bundle (see README.md next to it). Linux x86-64 + Apptainer only.
#
#   SIF=/path/biscuit_FC_v2.2.sif FS_LICENSE=/path/license.txt ./run_jobs.sh [--dry-run] [JOB ...]
#   ./run_jobs.sh --list
#
# Environment: SIF (required), FS_LICENSE (required for the macruise-* jobs), RUNS (default ./runs),
# THREADS (default: all cores, or $SLURM_CPUS_PER_TASK), PYTHON (default python3; goldens job only).
# A job whose run already finished with exit 0 is skipped, so rerunning after a failure is safe.
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ref=$here/pyBiscuit/scripts/reference
RUNS=${RUNS:-$here/runs}
THREADS=${THREADS:-${SLURM_CPUS_PER_TASK:-$(nproc 2>/dev/null || echo 1)}}
PYTHON=${PYTHON:-python3}
SIF_SHA256=${SIF_SHA256:-e70aea6290ca3312a290618a8092b8992444bc47a7d3b42680109dfd714d4329}  # override: tests only

FS_SUBJECTS=(0050952 0050961 0050971)
ALL=(check macruise-oasis45 macruise-mb101)
for s in "${FS_SUBJECTS[@]}"; do ALL+=("fs-$s"); done
ALL+=(goldens pack)
DEFAULT=(check macruise-oasis45 fs-0050952 macruise-mb101 fs-0050961 fs-0050971 pack)

DRY=()
jobs=()
for a in "$@"; do
    case $a in
        --dry-run) DRY=(--dry-run) ;;
        --list) printf '%s\n' "${ALL[@]}"; exit 0 ;;
        -h | --help) sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) jobs+=("$a") ;;
    esac
done
[ "${#jobs[@]}" -gt 0 ] || jobs=("${DEFAULT[@]}")
mkdir -p "$RUNS/logs"
status=$RUNS/status.tsv

log() { printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" | tee -a "$status"; }
die() { echo "error: $*" >&2; exit 1; }
sha() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -c1-64; }
need_sif() { [ -n "${SIF:-}" ] || die "set SIF=/path/to/biscuit_FC_v2.2.sif"; }

done_ok() {  # done_ok OUT: the run.json run_full.sh wrote says exit 0
    [ -f "$1.harness/run.json" ] && grep -q '"exit": 0' "$1.harness/run.json"
}

full_run() {  # full_run NAME INPUT RECON REG [LICENSE]
    local name=$1 in=$2 recon=$3 reg=$4 lic=${5:-} out=$RUNS/$1
    if [ "${#DRY[@]}" = 0 ] && done_ok "$out"; then log "$name" "skipped (already done)"; return 0; fi
    local args=(--sif "$SIF" --sif-sha256 "$SIF_SHA256" --input "$in" --out "$out"
        --surf-recon "$recon" --reg "$reg" --thread "$THREADS")
    [ -z "$lic" ] || args+=(--license "$lic")
    log "$name" "start (threads $THREADS)"
    "$ref/run_full.sh" "${args[@]}" "${DRY[@]+"${DRY[@]}"}" 2>&1 | tee "$RUNS/logs/$name.log"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" = 0 ] && [ "${#DRY[@]}" = 0 ] && [ -f "$out.harness/trace.jsonl" ]; then
        "$PYTHON" "$ref/trace_to_spec.py" "$out.harness/trace.jsonl" -o "$out.harness/orchestrator-trace.md" \
            >>"$RUNS/logs/$name.log" 2>&1 || true
    fi
    log "$name" "exit $rc"
    return 0
}

job_check() {
    need_sif
    {
        echo "date: $(date -u)"
        echo "host: $(hostname)  uname: $(uname -srm)"
        echo "cpus: $THREADS  mem: $(free -g 2>/dev/null | awk '/Mem:/{print $2 " GB"}')"
        echo "runtime: $( (apptainer --version || singularity --version) 2>&1 | head -1)"
        echo "disk free in $RUNS: $(df -h "$RUNS" | awk 'NR==2{print $4}')"
        echo "sif: $SIF"
        echo "bundle: $(cat "$here/BUNDLE_VERSION" 2>/dev/null)"
    } | tee "$RUNS/env.txt"
    [ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] || die "needs Linux x86-64"
    command -v apptainer >/dev/null || command -v singularity >/dev/null || die "apptainer/singularity not on PATH"
    [ -n "${FS_LICENSE:-}" ] && [ -f "$FS_LICENSE" ] || echo "warning: FS_LICENSE not set or missing; the macruise-* jobs need it"
    echo "checking image hash (takes a minute)..."
    local got
    got=$(sha "$SIF")
    echo "sif sha256: $got" | tee -a "$RUNS/env.txt"
    [ "$got" = "$SIF_SHA256" ] || die "image hash differs from $SIF_SHA256: stop and report the hash above"
    (cd "$here" && sha256sum -c --quiet BUNDLE.sha256) || die "bundle files damaged (BUNDLE.sha256 check failed)"
    log check ok
}

job_goldens() {  # optional: per-tool goldens on synthetic meshes (needs Python + pip)
    need_sif
    local venv=$RUNS/venv
    if [ ! -x "$venv/bin/python" ]; then
        "$PYTHON" -m venv "$venv" && "$venv/bin/python" -m pip install -q -U pip &&
            "$venv/bin/python" -m pip install -q -e "$here/pyBiscuit" || { log goldens "pip install failed"; return 0; }
    fi
    log goldens start
    (cd "$here/pyBiscuit" && "$venv/bin/python" scripts/reference/make_image_golden.py --sif "$SIF" \
        --sif-sha256 "$SIF_SHA256" --all --keep-going --out "$RUNS/golden-image" "${DRY[@]+"${DRY[@]}"}") \
        2>&1 | tee "$RUNS/logs/goldens.log"
    log goldens "exit ${PIPESTATUS[0]}"
}

job_pack() {  # everything to send back, in one tar (inner archives are already compressed)
    local name
    name=pybiscuit-image-results-$(hostname -s 2>/dev/null || hostname)-$(date +%Y%m%d)
    local items=(status.tsv logs)
    [ -f "$RUNS/env.txt" ] && items+=(env.txt)
    [ -d "$RUNS/golden-image" ] && items+=(golden-image)
    for d in "$RUNS"/*.harness; do
        [ -d "$d" ] || continue
        local b
        b=$(basename "$d")
        for f in run.json trace.jsonl MANIFEST.sha256 orchestrator-trace.md "${b%.harness}.tar.gz"; do
            [ -f "$d/$f" ] && items+=("$b/$f")
        done
    done
    (cd "$RUNS" && tar -cf "$name.tar" "${items[@]}") || die "tar failed"
    log pack "$RUNS/$name.tar"
    echo "send back: $RUNS/$name.tar ($(du -h "$RUNS/$name.tar" | cut -f1)), sha256 $(sha "$RUNS/$name.tar")"
}

for j in "${jobs[@]}"; do
    case $j in
        check) job_check ;;
        macruise-*)
            need_sif
            [ -n "${FS_LICENSE:-}" ] || die "$j needs FS_LICENSE (the MaCRUISE route runs FreeSurfer tools)"
            full_run "$j" "$here/inputs/macruise/sample" macruise "${j#macruise-}" "$FS_LICENSE" ;;
        fs-*)
            need_sif
            [ -d "$here/inputs/freesurfer/${j#fs-}" ] || die "no input for $j"
            full_run "$j" "$here/inputs/freesurfer/${j#fs-}" freesurfer mb101 "${FS_LICENSE:-}" ;;
        goldens) job_goldens ;;
        pack) job_pack ;;
        *) die "unknown job $j (see --list)" ;;
    esac
done
