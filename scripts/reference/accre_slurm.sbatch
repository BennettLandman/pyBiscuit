#!/bin/bash
# ACCRE (Vanderbilt) Slurm template for the image reference harness (plan B1.1, B1.2, B1.4).
# Copy, edit the CONFIG block, then:  sbatch scripts/reference/accre_slurm.sbatch
# Nothing here uploads the image or the licence; outputs stay under $OUTROOT.
#SBATCH --job-name=biscuit-ref
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.log

set -euo pipefail

# ---- CONFIG (owner to fill in; Q1 in docs/open-questions.md) ---------------------------------
REPO=${REPO:-$HOME/pyBiscuit}                      # checkout of this repository
SIF=${SIF:-/path/to/biscuit_FC_v2.2.sif}            # the production image (not in git)
SIF_SHA256=${SIF_SHA256:-e70aea6290ca3312a290618a8092b8992444bc47a7d3b42680109dfd714d4329}
FS_LICENSE=${FS_LICENSE:-$HOME/.freesurfer/license.txt}  # FreeSurfer licence (never copied anywhere)
CASE_INPUT=${CASE_INPUT:-}                          # MaCRUISE v3 input dir for the full run; empty = skip
REG=${REG:-oasis45}                                 # --reg for the full run (default image template: mb101)
OUTROOT=${OUTROOT:-/scratch/$USER/biscuit-ref/$SLURM_JOB_ID}
PYTHON=${PYTHON:-python3}                           # needs numpy + this repo's pybiscuit on PYTHONPATH
MODULES=${MODULES:-}                                # e.g. "Apptainer" if ACCRE provides it as a module
# ---------------------------------------------------------------------------------------------

# DRY_RUN=1 bash accre_slurm.sbatch  prints the commands without running them
run() { if [ "${DRY_RUN:-0}" = 1 ]; then printf '+'; printf ' %q' "$@"; printf '\n'; else "$@"; fi; }

if [ -n "$MODULES" ] && [ "${DRY_RUN:-0}" != 1 ]; then
    # shellcheck disable=SC2086 # intentional word splitting of the module list
    module load $MODULES
fi
export PYTHONPATH=$REPO/src${PYTHONPATH:+:$PYTHONPATH}
export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-$OUTROOT/tmp}
export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-$OUTROOT/cache}
run mkdir -p "$OUTROOT" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
[ "${DRY_RUN:-0}" = 1 ] || cd "$REPO"

echo "== per-tool goldens (B1.4) -> $OUTROOT/golden-image"
run "$PYTHON" scripts/reference/make_image_golden.py --sif "$SIF" --sif-sha256 "$SIF_SHA256" \
    --all --keep-going --out "$OUTROOT/golden-image" || echo "some golden cases failed (see log)"

if [ -n "$CASE_INPUT" ]; then
    echo "== full run with command trace (B1.1, B1.2) -> $OUTROOT/full"
    run scripts/reference/run_full.sh --sif "$SIF" --sif-sha256 "$SIF_SHA256" \
        --input "$CASE_INPUT" --out "$OUTROOT/full/run" --license "$FS_LICENSE" --reg "$REG" \
        --thread "${SLURM_CPUS_PER_TASK:-1}"
    run "$PYTHON" scripts/reference/trace_to_spec.py "$OUTROOT/full/run.harness/trace.jsonl" \
        -o "$OUTROOT/full/orchestrator-trace.md"
fi
echo "done: copy $OUTROOT/golden-image into tests/data/golden-image/ and review by hand"
