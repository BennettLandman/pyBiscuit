# pyBiscuit image jobs: instructions

This bundle runs the original BISCUIT 2.2 container (`biscuit_FC_v2.2.sif`) on a few test cases and **keeps every
intermediate file**, which the normal `run_biscuit` deletes. pyBiscuit (a Python port of BISCUIT) uses these
outputs as its reference. The container only runs on Linux x86-64, which is why we need your help.

You don't need to understand BISCUIT. The script runs everything, and you send back one `.tar` file.

## What you need

| | |
|---|---|
| Machine | Linux x86-64 with Apptainer or Singularity (e.g. ACCRE). Not macOS or ARM. |
| Image | `biscuit_FC_v2.2.sif`, SHA-256 `e70aea6290ca3312a290618a8092b8992444bc47a7d3b42680109dfd714d4329`. It is not in this bundle; ask the sender where to get it. |
| FreeSurfer licence | A `license.txt` (free from FreeSurfer). Needed for the two `macruise-*` jobs. |
| Resources | 8 cores and 32 GB RAM per job are plenty. Allow about 10 GB of disk per job; the run times are not measured yet (expect a few hours per job, so request 24 h). |
| Python | `python3` (standard library only), used to summarise the command traces. The optional `goldens` job also needs internet access for `pip`. |

## Steps

### 1. Unpack and check the bundle

```bash
tar -xzf pybiscuit-image-jobs-*.tar.gz
cd pybiscuit-image-jobs-*/
sha256sum -c --quiet BUNDLE.sha256 && echo bundle OK
```

### 2. Point the script at the image and the licence

```bash
export SIF=/path/to/biscuit_FC_v2.2.sif
export FS_LICENSE=/path/to/license.txt
export RUNS=/path/to/scratch/pybiscuit-runs   # optional; default is ./runs inside the bundle
```

On ACCRE, load Apptainer first if it is not on your `PATH` (for example `module load apptainer`).

### 3. Check the setup

```bash
./run_jobs.sh check
```

This prints the machine details, verifies the image hash and the bundle, and stops with a clear message if something is
wrong. **If it reports a different image hash, stop and send us that hash**; don't run the other jobs with that image.

Optional: `./run_jobs.sh --dry-run macruise-oasis45` prints the container commands without running anything.

### 4. Run the jobs

Either all of them, one after another, in one session (use `tmux` or `screen`, it takes many hours):

```bash
./run_jobs.sh
```

Or one Slurm job each, in parallel (ACCRE):

```bash
for j in macruise-oasis45 fs-0050952 macruise-mb101 fs-0050961 fs-0050971; do
  sbatch --job-name=$j --cpus-per-task=8 --mem=32G --time=24:00:00 --output=$RUNS/$j-%j.out \
         --export=ALL,THREADS=8 --wrap "cd $PWD && ./run_jobs.sh $j"
done
```

A job that already finished is skipped when you rerun it, so after a failure you can rerun the same command.
`$RUNS/status.tsv` lists every start and exit code.

| Job | What it runs | Licence | Priority |
|---|---|---|---|
| `macruise-oasis45` | Full BISCUIT on the MaCRUISE sample subject, template `oasis45` | yes | 1 |
| `fs-0050952` | Full BISCUIT on a FreeSurfer subject (ABIDE NYU 0050952), template `mb101` | no | 1 |
| `macruise-mb101` | As `macruise-oasis45`, with the default template `mb101` | yes | 2 |
| `fs-0050961`, `fs-0050971` | As `fs-0050952`, two more subjects | no | 3 |
| `goldens` | Optional. Small per-tool tests on synthetic meshes (needs `pip install`) | no | 4 |

If time is short, priority 1 alone already answers most of our questions.

### 5. Pack and send back

When the jobs are finished (or you have stopped):

```bash
./run_jobs.sh pack
```

It prints the path and SHA-256 of `pybiscuit-image-results-<host>-<date>.tar` (several GB). Send that file, and its
SHA-256, by the route the sender gave you (Box, ACCRE share, ...). Send it even if some jobs failed: the logs inside tell us
why.

## Rules

- Don't upload the image or your licence anywhere, and don't put either in the results.
- The MaCRUISE sample input and every output are derived from real subjects. Keep them on the machine you ran them on, and
  send the results only to the sender, by the agreed route.
- Don't edit the scripts. If something fails, send the logs (`$RUNS/logs/`, `$RUNS/status.tsv`) instead.

## What is inside

| Path | Content |
|---|---|
| `run_jobs.sh` | The driver above |
| `pyBiscuit/` | Snapshot of the pyBiscuit repository; the container harness is `pyBiscuit/scripts/reference/` (`run_full.sh`) |
| `inputs/macruise/sample/` | MaCRUISE v3 sample subject: inner surface (native and MNI), T1, refined segmentation |
| `inputs/freesurfer/<id>/` | The 18 FreeSurfer files BISCUIT reads (`surf/`, `label/?h.cortex.label`) for three ABIDE subjects |
| `BUNDLE.sha256`, `BUNDLE_VERSION` | Checksums of every file; the pyBiscuit commit the bundle was built from |

Each job writes `$RUNS/<job>/` (the BISCUIT output tree including `.tmp`) and `$RUNS/<job>.harness/` (a trace of every
tool call the container made, `run.json`, a checksum manifest, and a `.tar.gz` of all of it).

## Troubleshooting

| Problem | What to do |
|---|---|
| `apptainer/singularity not on PATH` | Load the module (`module load apptainer`) or ask your admin. |
| `image hash differs` | Stop. Send the hash `check` printed. |
| A job exits non-zero | Rerun it once (finished steps are not redone). If it fails again, run `pack` and send the results. |
| Out of disk | Set `RUNS` to a scratch directory with more space and rerun. |
| `goldens`: `pip install failed` | Skip it; it is optional. |

## Why (for the curious)

The pyBiscuit port is compared against these outputs. The MaCRUISE runs give the first image reference for the MaCRUISE
route (pyBiscuit plan rows B1.1, B1.2, B5.4–B5.7, B11.1). The FreeSurfer runs repeat runs whose final outputs we already
have, now with intermediates: they show how much the image differs from its own earlier run (Q15), and provide the LGI and
registration intermediates (`tmp-lgi-*`, HSD) that the final outputs alone cannot pin down.
