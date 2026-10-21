# Reference artifacts

## Images

| File | SHA-256 | Content |
|---|---|---|
| `biscuit_FC_v2.2.sif` | `e70aea6290ca3312a290618a8092b8992444bc47a7d3b42680109dfd714d4329` | BISCUIT 2.2 (CBT v1.3) built `From: biscuit_cbt_v1.3.sif`; adds `reg/`, `parc/`, `run_biscuit_FC`, `biscuit_FC.conf` |
| `biscuit_cbt_v1.3.sif` | `f22acb83abe22bd896958eb5d087dbe7b4fd0da18afde035521239b9541a06b7` | base: `biscuit`, cmorph v1.7 (`lgi`), python3 + numpy 1.17.3, scipy 1.3.2, torch 1.4.0+cpu |

**Use the FC image as the reference.** All `cmorph/bin/*`, `biscuit/bin/*`,
and `biscuit/sphere/*` files are byte-identical between the two images, but
the `biscuit` orchestrator differs (FC: 8,463,674 B, 2021-06-25; cbt:
7,631,950 B, 2021-06-24), as do `run_biscuit`, `biscuit.conf`, the parc
`test.sh`/`model.py`/`ops.py`, and the icosphere `.pkl` files (re-pickled). The
FC CTF manifest is [reference/ctf-manifest.tsv](reference/ctf-manifest.tsv);
the cbt one is kept as [reference/ctf-manifest-cbt_v1.3.tsv](reference/ctf-manifest-cbt_v1.3.tsv).

Both: SIF squashfs partition at byte offset 45056. Runtime environment:
`MCRROOT=/usr/local/MATLAB/MATLAB_Compiler_Runtime/v717`,
`FREESURFER_HOME=/opt/biscuit`, `OMP_NUM_THREADS=1`, Xvfb, TeX Live, Ghostscript.
Production runs bind a FreeSurfer `license.txt` to `/opt/biscuit/license.txt`.

The VUIIS doc (`vuiis-cci-info/available-pipelines/BISCUIT-curve_extract-surf.md`)
describes v2.0.0 (`biscuit_FC_v2.0.simg`); our images are v2.2 — note the
version gap when comparing to its example PDFs.

## Extract for reading (macOS or Linux; no Singularity needed)

```bash
python scripts/extract_reference.py --sif ../biscuit_FC_v2.2.sif --out ../reference
```

This writes `reference/sif_FC_v2.2/root/{opt,.singularity.d}`,
`reference/ctf_plain/` (plaintext CTF members), and refreshes
`docs/reference/ctf-manifest.tsv`. Requires `unsquashfs` (squashfs-tools).

Upstream sources (pin the commit you port from in each port note):

```bash
mkdir -p ../reference/repos && cd ../reference/repos
for r in CurveExtraction HSD klaplace LocalGyrificationIndex MeshLib SphericalRemesh SurfRemesh SphericalLabeling; do
  git clone https://github.com/ilwoolyu/$r.git; done
git clone https://github.com/NIRALUser/ShapeComplexityIndex.git
```

**Image binaries match upstream `master`** (checked 2026-10-20 from version strings in the extracted ELF files): HSD v1.3.9 (`f169595`), CurveExtraction/TRACE v1.1.2 (`ded932d`), Gyrification v1.2.2 (`b9fa760`), SphericalRemesh v1.2.1 (`b677c39`). klaplace has no version string.

Commits at planning time: CurveExtraction `ded932d`, HSD `f169595`,
LocalGyrificationIndex `b9fa760`, MeshLib `dcef807`, ShapeComplexityIndex
`aea9afd`, SphericalLabeling `37fa7a0`, SphericalRemesh `b677c39`,
SurfRemesh `6d90ee3`, klaplace `93b0d02`.

## Harvesting reference outputs (Linux x86-64 + Apptainer/Singularity)

The image cannot run on macOS/ARM. Reference generation needs a Linux host
(ACCRE or a VM). Three tiers, all scripted under `scripts/reference/` (plan B1):

1. **Full pipeline with intermediates.** `run_biscuit` deletes `$OUT/.tmp`.
   Run the orchestrator directly to keep everything:

   ```bash
   apptainer exec -e --home "$HOME_TMP" \
     -B "$IN":/INPUTS -B "$OUT":/OUTPUTS -B "$FS_LICENSE":/opt/biscuit/license.txt \
     biscuit_FC_v2.2.sif bash -c '
       export TMP=/OUTPUTS/.tmp; mkdir -p $TMP
       grep -vE "^(fs_|% freesurfer|cv_|% civet)" /opt/biscuit/biscuit.conf \
         | sed "s/^surf_recon.*/surf_recon = macruise/; s/^reg_temp.*/reg_temp = oasis45/" > $TMP/biscuit.conf
       export MCR_CACHE_ROOT=$(mktemp -d); check_input && cd $TMP &&
       xvfb-run -a --server-args="-screen 0 1024x768x24 -ac +extension GLX" \
         biscuit /INPUTS /OUTPUTS $TMP/biscuit.conf'
   ```

   Also capture: `strace -f -e trace=execve` (or a `PATH` shim that logs argv)
   to record every sub-tool invocation with arguments, timing, and the files
   each reads/writes. This command log is the de-facto spec for the encrypted
   orchestrator.
2. **Per-tool goldens.** Call each binary in the image (`HSD`,
   `CurveExtraction`, `Gyrification`, `klaplace`, `OuterHull`, `MeshProperty`,
   `ShapeComplexityIndex`, `SphericalRemesh`, `MeshDeform`, parc `test.sh`) on
   committed synthetic meshes and on intermediates from tier 1. Small goldens
   go in `tests/data/golden/` with the exact command and image hash.
3. **Cohort parity.** ≥10 subjects with existing MaCRUISE v3 outputs (MASI
   cohorts already processed with `--reg oasis45 --surf_recon macruise`), to
   estimate between-implementation variability against test–retest variability.

Inputs for the first tier-1 case: `~/dev/mipav/sample-macruise/output/`
(`Surfaces_FreeView/target_image_GMimg_innerSurf.asc`,
`Surfaces_MNI/target_image_GMimg_innerSurf.asc` → `*_mni.asc`,
`SegRefine/SEG_refine.nii.gz`, `Raw/T1.nii.gz`).

## How to run the image harness (Linux x86-64 only)

Everything below lives in `scripts/reference/` (overview: its `README.md`) and has been
mock-tested on macOS without the image (`tests/test_reference_harness.py`); it has **not** yet been
run against the real image. Never run it on macOS, never trigger the workflow before Q1 is answered,
and never upload the image or a licence file anywhere except where the owner decides.

Prerequisites on the host: Apptainer or Singularity, the image (SHA-256 above is checked; override with
`--sif-sha256`), Python with NumPy and `pip install -e .` of this repo (fixture generators only), and for
full runs a FreeSurfer `license.txt` plus a MaCRUISE v3 input directory.

```bash
# 1. Per-tool goldens (B1.4): replays scripts/oracles/make_golden.py cases and the image-only tools.
python scripts/reference/make_image_golden.py --sif biscuit_FC_v2.2.sif --list
python scripts/reference/make_image_golden.py --sif biscuit_FC_v2.2.sif --all --keep-going
# -> tests/data/golden-image/<tool>/<case>/{outputs, command.json ("oracle": "image-biscuit_FC_v2.2", image_sha256), help.txt}
```

```bash
# 2. Full pipeline with intermediates and command trace (B1.1 + B1.2)
scripts/reference/run_full.sh --sif biscuit_FC_v2.2.sif --input "$IN" --out "$OUT/subj" \
    --license "$HOME/.freesurfer/license.txt" --reg oasis45      # add --dry-run to print the commands
python scripts/reference/trace_to_spec.py "$OUT/subj.harness/trace.jsonl" -o orchestrator-trace.md
```

`run_full.sh` writes the output tree with `.tmp` kept (a copy of `run_biscuit` without its
`rm -rf $TMP` and `rm $CONF` lines is run), `<out>.harness/{trace.jsonl, trace/, run.json,
MANIFEST.sha256, <name>.tar.gz}`. The trace comes from `PATH` shims for every compiled sub-tool of
`/opt/biscuit/bin` and `/opt/cmorph/bin`, plus a copy of the `lgi` tcsh script placed beside them so its
`$0:h/<tool>` calls are traced as well. Known limits: calls made by absolute path from inside the compiled
MATLAB orchestrator bypass the shims (use `--strace` to add `strace -f -e trace=execve` if the image has
strace, or compare the call list against `trace_to_spec.py` output); "modified" files are detected by
size or whole-second mtime. Review the first trace by hand before treating it as the orchestrator spec (B1.3).

**Flags not yet confirmed.** MeshProperty, MeshDeform, ShapeComplexityIndex, klaplace and OuterHull flags in
`make_image_golden.py` come from `strings` on the binaries and the `lgi` script, not from running them.
Each such golden stores `help.txt` (`<tool> --help`); a case with wrong flags fails loudly and
`--keep-going` records the rest, so fix the flags from the first `help.txt` and rerun.

**ACCRE:** copy `scripts/reference/accre_slurm.sbatch`, set `SIF`, `FS_LICENSE`, optional `CASE_INPUT`,
`MODULES`, then `sbatch` it (`DRY_RUN=1 bash accre_slurm.sbatch` prints the commands).
**GitHub Actions:** `.github/workflows/reference.yml` (`workflow_dispatch` only, input `confirm` must be
`RUN`): downloads the image from a release asset given by `release_tag`, `sif_asset`, `sif_sha256`,
checks the hash, takes the licence from the secret `FS_LICENSE`, runs step 1 (and step 2 when
`full_case_asset`, a tar.gz containing an `input/` directory, and `full_case_sha256` are given), and
uploads the goldens and trace as workflow artifacts. It never commits; review the artifacts and commit
goldens by hand (each case stays under ~1 MB, `make_image_golden.py` warns otherwise).

Tests choose goldens with `tests/_golden.py::golden_dir(tool, case)`, which prefers
`tests/data/golden-image/` over `tests/data/golden/` when both exist.
