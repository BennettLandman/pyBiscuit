# Image reference harness (plan B1.1, B1.2, B1.4)

Scripts that harvest reference outputs from `biscuit_FC_v2.2.sif`. **Nothing here may run on
macOS, and the image and the FreeSurfer licence are never uploaded or committed.** Everything
was written and mock-tested without the image (`tests/test_reference_harness.py`); the first
real run needs a Linux x86-64 host with Apptainer or Singularity (open question Q1,
[docs/open-questions.md](../../docs/open-questions.md)).

| Script | Purpose |
|---|---|
| `run_full.sh` | B1.1: whole pipeline with intermediates kept, tracing on, archive + SHA-256 manifest |
| `trace_shim.sh`, `install_shims.sh` | B1.2: `PATH` shims that log every sub-tool call as JSON |
| `container_inner.sh` | runs inside the image (patched `run_biscuit`, shim PATH) |
| `trace_to_spec.py` | B1.2/B1.3: trace JSONL to a markdown call graph and file-flow table |
| `make_image_golden.py` | B1.4: replays the oracle cases and the image-only tools inside the image |
| `accre_slurm.sbatch` | the same commands as an ACCRE Slurm job |
| `../../.github/workflows/reference.yml` | the same on a GitHub Actions runner (manual dispatch only) |

Full instructions: "How to run the image harness" in
[docs/reference-artifacts.md](../../docs/reference-artifacts.md). Every script accepts
`--dry-run` (where it makes sense) and prints the exact commands.

## GitHub Actions workflow: disabled until the owner acts

`reference.yml` has only a `workflow_dispatch` trigger and refuses to start unless the
`confirm` input is `RUN`. It cannot run until the owner has (1) uploaded the image as a release
asset of this private repository (owner decision: the image contains MCR and FreeSurfer
binaries) and (2) created the repository secret `FS_LICENSE` (only for full runs). Nothing in the
workflow commits or pushes; goldens come back as a workflow artifact to review and commit by hand.
