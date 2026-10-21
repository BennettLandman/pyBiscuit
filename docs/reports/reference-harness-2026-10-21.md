# Image reference harness: build and mock test (brief 10)

Date 2026-10-21. Plan rows B1.1, B1.2, B1.4 (tooling only; nothing was run against the image).

## What exists

| Piece | File |
|---|---|
| Full run, intermediates kept, archive + manifest | `scripts/reference/run_full.sh`, `container_inner.sh`, `_lib.sh` |
| Command tracing (PATH shims, JSON per call) | `scripts/reference/trace_shim.sh`, `install_shims.sh` |
| Trace to markdown call graph / file flow / flag table | `scripts/reference/trace_to_spec.py` |
| Image goldens (oracle cases replayed + image-only tools) | `scripts/reference/make_image_golden.py`, `tests/_golden.py` |
| ACCRE job | `scripts/reference/accre_slurm.sbatch` |
| GitHub Actions (manual dispatch only, `confirm = RUN`) | `.github/workflows/reference.yml` |
| Mock end-to-end test | `tests/test_reference_harness.py` |

## Verification done (macOS arm64, no image, no Linux)

- `shellcheck` (shellcheck-py 0.11, dev tool installed outside the repo) clean on every `.sh` and the sbatch template.
- `actionlint` clean on `reference.yml` (and `ci.yml`).
- `--dry-run` on `run_full.sh`, `install_shims.sh`, `make_image_golden.py`; `DRY_RUN=1` on the sbatch: prints the exact commands, touches nothing.
- pytest mock: fake image (stub executables) + `apptainer` stub exercise the shims (argv with spaces, quotes, newlines; exit code 3 propagated; missing real binary gives 127), JSONL merge, `lgi` copy routing its tools through the shims, `.tmp` kept, archive and manifest, failure still archived, `trace_to_spec.py` (file-flow edge CurveExtraction to Gyrification), golden writing (`"oracle": "image-biscuit_FC_v2.2"`, image hash) and `golden_dir` preference.

## Not verified / risks (need the first real run)

1. **Flags** of MeshProperty, MeshDeform, ShapeComplexityIndex, klaplace, OuterHull come from `strings` and the `lgi` script. Each golden stores `help.txt`; failing cases are recorded with `--keep-going`.
2. The compiled MATLAB orchestrator may call tools by absolute path (bypasses the shims); `--strace` is provided if the image has strace.
3. Apptainer on `ubuntu-latest` (24.04) needs unprivileged user namespaces: the workflow sets the AppArmor sysctl; unproven.
4. The layout of `/INPUTS` for the full run is whatever the production runs used; staging the legacy sample (`sample-macruise/output`) into it is left to the first real run.
5. parc `test.sh` goldens (B1.4) are not covered.
