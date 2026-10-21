# Agent brief 10 — Image reference harness, ready to run (B1.1, B1.2, B1.4 tooling)

**Written 2026-10-20.** Independent of briefs 04–09. Read
[AGENTS.md](../../AGENTS.md), [reference-artifacts.md](../reference-artifacts.md),
`docs/ports/oracles.md`, and the [parallel rules](README.md#parallel-sessions).

## Context

No Linux host or image upload is available yet (Q1 open). Build every script
and workflow needed to harvest **image** references, so that when Q1 is
answered a single command (on ACCRE or GitHub Actions) produces goldens that
drop in beside the existing oracle goldens. Nothing here may run the image
on this Mac. Never upload the image or any licence file anywhere.

## Scope

| Item | Deliverable |
|---|---|
| B1.1 | `scripts/reference/run_full.sh`: run `biscuit_FC_v2.2.sif` (Apptainer or Singularity) on a MaCRUISE v3 input dir with intermediates kept (bypass `run_biscuit`'s `rm -rf $TMP` as in reference-artifacts.md), conf overrides (`--reg`, `--surf_recon macruise`), an FS licence bind, and timing. Output: an archive plus a SHA-256 manifest. |
| B1.2 | Command tracing without strace: a `PATH` shim directory bind-mounted ahead of `/opt/biscuit/bin` and `/opt/cmorph/bin` that logs argv, cwd, env subset, start/end time, exit code, and the files touched (pre/post listing of cwd and argument paths) for every sub-tool, then `exec`s the real binary. Output: JSONL. `scripts/reference/trace_to_spec.py` turns the JSONL into a markdown call graph (the draft of B1.3's orchestrator spec). |
| B1.4 | `scripts/reference/make_image_golden.py`: replays **the same cases** as `scripts/oracles/make_golden.py`, but inside the image (`apptainer exec … <tool>`), writing to `tests/data/golden-image/<tool>/<case>/` with `"oracle": "image-biscuit_FC_v2.2"` and the image SHA-256. Include cases for MeshProperty, MeshDeform, ShapeComplexityIndex, and klaplace (image-only tools; flags from their binaries' strings/help). Tests prefer image goldens over oracle goldens when both exist. |
| Workflow | `.github/workflows/reference.yml`: `workflow_dispatch` only. Installs Apptainer on `ubuntu-latest`, downloads the SIF from a release asset whose name and hash are inputs, verifies SHA-256, takes the FS licence from a secret, runs B1.4 (and optionally B1.1 + B1.2 on a case named by input), and uploads goldens as an artifact (never committed by CI). Disabled by default with a clear README note; it needs owner action (Q1). |
| ACCRE | `scripts/reference/accre_slurm.sbatch` template for the same commands. |

## Acceptance

- `shellcheck` clean; `--dry-run` on every script prints the exact commands.
- Mock test: a fake "image" (a directory with stub executables, plus a
  `apptainer` stub on `PATH`) exercises the shims, JSONL logging,
  trace_to_spec, and golden writing end to end on macOS in pytest.
- Workflow YAML validates (`actionlint` if installable, else a YAML parse plus
  a documented manual review). Do not trigger it.
- docs: a "How to run" section added to reference-artifacts.md, plus an update
  to Q1 in open-questions.md listing exactly what the owner must provide
  (release asset or ACCRE path, licence secret name).
