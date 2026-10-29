# Linux runbook: image reference jobs (owner-run)

The owner runs these on a Linux x86-64 host with Apptainer or Singularity
(ACCRE or similar) when asked, using `git pull`. Each job says what to send
back. Nothing here runs on macOS. Never commit or upload the image, the
FreeSurfer licence, or subject-derived outputs.

## Bundle for someone else to run (preferred, 2026-10-29)

`scripts/reference/make_bundle.sh` (on the Mac, from a clean commit) writes
`~/dev/biscuit/bundles/pybiscuit-image-jobs-<date>-<commit>.tar.gz`: a snapshot of this repository, the
driver `run_jobs.sh`, and staged inputs (the MaCRUISE v3 sample and the 18 FreeSurfer files of each
open-examples subject, `stage_freesurfer_input.sh`). The recipient needs only the image, a licence and
Apptainer; the instructions are the bundle's `README.md` (source: `scripts/reference/bundle/README.md`).
Jobs: `macruise-oasis45`, `macruise-mb101` (job 2 below), `fs-<subject>` (FreeSurfer-route reruns with
intermediates: the image's own run-to-run spread for Q15, `tmp-lgi-*`, HSD intermediates), optional
`goldens` (job 1), then `pack`, which writes one `.tar` to send back. Mock-tested in
`tests/test_reference_harness.py`; not yet run on a real host. Results go to `~/dev/biscuit/image-runs/`,
never into git.

## 0. One-time setup

```bash
git clone https://github.com/BennettLandman/pyBiscuit.git   # later: cd pyBiscuit && git pull
cd pyBiscuit
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip && python -m pip install -e '.[dev]'
```

You also need:

- `biscuit_FC_v2.2.sif` on the host. It must match
  `e70aea6290ca3312a290618a8092b8992444bc47a7d3b42680109dfd714d4329`; the
  scripts check this. If MASI's ACCRE copy has a different hash, report that
  hash rather than overriding it.
- A FreeSurfer `license.txt` (jobs 2 and 3 only).
- Job 2's input case: copy `~/dev/mipav/sample-macruise/output/` from the Mac
  (only the four files below are used).

Set:

```bash
export SIF=/path/to/biscuit_FC_v2.2.sif
export FS_LICENSE=/path/to/license.txt
export RUNS=/path/to/scratch/biscuit-runs      # outside the repo
```

Add `--dry-run` to any script call to print its commands without running them.

## Job 1 — per-tool image goldens (B1.4)  ← run first

Replays the oracle cases inside the image, plus the image-only tools
(MeshProperty, MeshDeform, ShapeComplexityIndex, klaplace, OuterHull). It
uses only synthetic fixtures, so the outputs may be committed. No licence
needed.

```bash
git pull
python scripts/reference/make_image_golden.py --sif "$SIF" --list
python scripts/reference/make_image_golden.py --sif "$SIF" --all --keep-going 2>&1 | tee "$RUNS/job1.log"
```

**Send back:** push a branch with the goldens:

```bash
git switch -c image-goldens-$(date +%Y%m%d)
git add tests/data/golden-image && git commit -m "B1.4: image goldens from biscuit_FC_v2.2.sif"
git push -u origin HEAD
```

Also send `$RUNS/job1.log`. Some image-only tool flags come from `strings`,
not from running the tools, so failed cases are expected on the first pass.
Their `help.txt` lets us fix the flags, and you rerun job 1.

## Job 2 — full BISCUIT run with intermediates and command trace (B1.1, B1.2)

Uses the legacy MaCRUISE v3 sample (subject-derived: **never commit it or its
outputs**).

```bash
git pull
scripts/reference/stage_macruise_input.sh /path/to/sample-macruise/output "$RUNS/sample-in"
scripts/reference/run_full.sh --sif "$SIF" --input "$RUNS/sample-in" --out "$RUNS/sample-oasis45" \
    --license "$FS_LICENSE" --reg oasis45 2>&1 | tee "$RUNS/job2.log"
python scripts/reference/trace_to_spec.py "$RUNS/sample-oasis45.harness/trace.jsonl" \
    -o "$RUNS/orchestrator-trace.md"
```

If time allows, repeat with `--reg mb101` (the image default) into
`$RUNS/sample-mb101`.

**Send back** (copy to the Mac at `~/dev/biscuit/image-runs/`, not to git):
`sample-oasis45.harness/` (trace, `run.json`, `MANIFEST.sha256`, and the
`.tar.gz` of the output tree including `.tmp`), `orchestrator-trace.md`,
`job2.log`, and the `PDF/biscuit.pdf` from the output tree.

## Job 3 — cohort parity set (B1.6, Q2), later

Requested once the pipeline runs end to end: 10–20 MASI subjects that have
both MaCRUISE v3 outputs and existing BISCUIT outputs
(`--reg oasis45 --surf_recon macruise`), with ≥ 2 scan–rescan pairs if
available. Send paths and hashes only; the data stays on ACCRE.

## Troubleshooting

- **Hash mismatch:** stop and report the host's SHA-256.
- **`apptainer: command not found`:** load the module (ACCRE:
  `module load apptainer`, or see `scripts/reference/accre_slurm.sbatch`).
- **Batch use:** `scripts/reference/accre_slurm.sbatch` wraps jobs 1–2; set
  `SIF`, `FS_LICENSE`, `CASE_INPUT` and `MODULES`, and check it first with
  `DRY_RUN=1 bash scripts/reference/accre_slurm.sbatch`.
