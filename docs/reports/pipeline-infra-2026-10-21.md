# Pipeline infrastructure report (brief 08) — 2026-10-21

Scope: `pybiscuit.config`, `pybiscuit.pipeline` (stage graph, signatures, checkpoints,
`--resume`, `run.json`), input adapters, `pybiscuit run [--resume] [--dry-run]`.
No stage science. Port note: [docs/ports/pipeline.md](../ports/pipeline.md).

## Acceptance
| Criterion | Evidence |
|---|---|
| Signature changes on any input, param, asset or code-version change; unrelated settings (threads) do not | `tests/test_pipeline.py::test_signature_sensitivity` (asset change tested both by template choice and by patching a registry hash) |
| Resume skips valid stages | `test_resume_skips_valid_stages` (body call counts, no new attempt dirs) |
| Corrupted/missing outputs, changed params/inputs/assets/code are refused with a message naming the stage and what changed | `test_resume_detects_corrupted_and_missing_outputs`, `test_resume_refuses_changed_param`, `test_resume_refuses_changed_input_and_asset_and_code` |
| Attempt numbering | `test_attempt_numbering_and_failure_recovery` (failed attempt kept, next is `attempt-02`, history recorded) |
| Adapter finds every expected product in the BASE run and in `sample-macruise/output` (and `100249`) | `@reference` `test_pymacruise_reference_run`, `test_legacy_reference_outputs` |
| `aff.txt` maps FreeView inner vertices onto MNI within 1e-4 mm | `test_legacy_aff_maps_native_onto_mni`: **5.0e-6 mm** (using the inverse of `aff.txt`, see below) |
| `pybiscuit run --dry-run` prints a full plan, blocked stages marked, exit 0 on both reference inputs | `test_dry_run_reference_inputs` (8 of 9 stages BLOCKED, `ingest` runs); real `run` on the sample ingests, then exits 3 with `stage 'surfprep' is blocked: B5: ...` |
| No new runtime dependency | `tomllib` on 3.11+. `pyproject.toml` gains `tomli>=2.0; python_version < '3.11'` (MIT, one line, needed only so TOML configs work on 3.10; `biscuit.conf` needs nothing). Owner can drop it and lose TOML on 3.10. |
| Full suite and lint | 480 passed, 18 skipped (after rebase onto e225437); `ruff check .` clean |

## Findings
1. `Surfaces_MNI/aff.txt` maps **MNI -> native**, not native -> MNI. Applying it forward to the FreeView vertices leaves 77 mm residual; its inverse leaves 5.0e-6 mm. `native_to_mni_matrix()` encodes this. docs/io-contract.md calls it "the rigid transform" without direction.
2. PyMaCRUISE's `run.json` records no PyMaCRUISE version (only `schema_version`, `run_signature`, `implementation_signature`); the adapter records those and warns. **Request to PyMaCRUISE:** add a `pymacruise_version` field.
3. The BASE run's T1 is the `orientation-only-inputs/T1.nii.gz` derived resource (RAS); `inputs.t1` is the original-orientation file. The adapter prefers the former because surfaces are RAS and grids must agree.
4. macOS is case-insensitive, so `Output/` vs `output/` probing cannot distinguish them there; tests compare with `samefile`.

## Decisions to review
- Stage names, output file names and stage-to-stage inputs in `pipeline/stages.py` are placeholders from docs/io-contract.md and the plan rows; they must be revised against the brief 11 orchestrator spec / B1.3. Each blocked reason cites its rows (B5, B5.6/B3.2, B6, B7.1/B7.4/B7.5, B8, B9, B10.1, B12.3). Rebased onto `e225437`: HSD (PR #5), TRACE (PR #4) and the parcellation net (PR #3) are on main but, like SphericalRemesh, not wired in because their inputs and call order are the orchestrator's.
- FreeSurfer-route `REQUIRED` products (white, pial, inflated, sphere per hemisphere) are a guess pending the orchestrator spec. No FS6 subject exists locally; that adapter is tested on synthetic NiBabel-written surfaces only.
- Resume never recomputes a stage whose signature changed; it refuses (PyMaCRUISE-like). The message says to use a new output directory or remove the stage's attempt and `run.json` entry. A `--restart-from` option can be added when real stages exist.
- Asset digests come from the committed registry rather than hashing installed files (plans work without assets); verifying installed assets against the registry is left to `fetch-assets`/`assets.verify`.

## Not done / out of scope
Stage science; wiring HSD/TRACE/parc/SphericalRemesh into stage bodies; `legacy/` output naming (B10.4); CIVET route (no adapter).

## Shared-file edits
`cli.py`: one block (`run` dispatch, marked with comments). `pyproject.toml`: the `tomli` marker line. `plan.md`: B10.2 and B10.3 Status cells only.

## Config cross-check against the MATLAB source (2026-10-21)
`BISCUIT-master/BISCUIT/GenericRecon/biscuit/biscuit.conf` has the same keys, order and values as the FC image's file except the four `mc_*` entries, which are tree-relative there (`Surfaces_FreeView/target_image_GMimg_innerSurf.asc`, `Surfaces_MNI/target_image_GMimg_innerSurf.asc`, `Raw/T1.nii.gz`, `SegRefine/SEG_refine.nii.gz`) instead of flat file names. That is the layout the `macruise-v3` adapter already reads. Defaults stay those of the FC image (the acceptance reference); the 2022 source's `mc_*` values are an owner decision.
