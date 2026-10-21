# Agent brief 08 — Pipeline infrastructure and input adapters (B10.2, B10.3, config)

**Written 2026-10-20.** Independent of briefs 04–07. Read
[AGENTS.md](../../AGENTS.md), [plan.md](../../plan.md),
[io-contract.md](../io-contract.md), and the [parallel rules](README.md#parallel-sessions).

## Context

The orchestrator's stage semantics are still blocked (encrypted MATLAB;
source requested from Lyu). The **machinery** around stages is not: config,
checkpointing, resume, manifests, and input adapters. Build that
machinery now, with stage bodies that are placeholders for blocked steps.
Model it on PyMaCRUISE: `~/dev/mipav/pymacruise/src/pymacruise/base_resume_workflow.py`,
`base_workflow.py`, and a real manifest at
`~/dev/mipav/pymacruise/build/invivo-base-v3-dura002-no-regrid-2026-10-14/run.json`
(`schema_version` 1). Note that `final_outputs` there is a **list of file
records** (`path`, `sha256`, `bytes`, `shape`, `axis_codes`) with no
semantic keys, so the adapter must identify products by filename and stage.

## Scope

| Item | Deliverable |
|---|---|
| Config | `pybiscuit.config`: read the original `biscuit.conf` format (`key = value`, `%` comments, the `fs_`/`mc_`/`cv_` prefix filtering done by `run_biscuit`) plus TOML; defaults from the **FC** image's `biscuit.conf` (`reference/sif_FC_v2.2/root/opt/biscuit/biscuit.conf`); CLI overrides named as in `run_biscuit` (`--reg --parc --trace_r --trace_delta --curv_smooth --lgi_kernel --lgi_aniso --lgi_ref --sci_kernel --sci_bins --thread --proj --subj --sess --surf_recon`). Round-trip test against the shipped file. |
| B10.2 | `pybiscuit.pipeline`: a `Stage` with declared inputs, params, and outputs; content-addressed signatures over inputs, params, code version, and asset hashes; `stages/<name>/attempt-NN/` with `provenance.json`; `run.json` (inputs+hashes, config, versions, assets, frames, stage records, status); `--resume` that verifies signatures and output hashes and refuses mismatches with a clear message. Use PyMaCRUISE's behavior and wording where sensible. |
| Stage graph | Register the MaCRUISE-route stages from plan.md (surfprep, features, register, parc, trace, quant, stats, report). Implemented components (SphericalRemesh, etc.) are wired as callable stage bodies only where their inputs are defined. Every blocked stage raises `StageBlocked("B5: needs BISCUIT MATLAB source")` or similar, citing the plan row. |
| B10.3 | Input adapters: (a) PyMaCRUISE run dir (read `run.json`, check `status` and stage completion, locate inner/central/outer surfaces, level sets, `SEG_refine`, and the T1 by filename and stage; record hashes and the PyMaCRUISE version); (b) legacy MaCRUISE v3 dir (`.asc` read with `frame="RAS"`, `Surfaces_MNI/aff.txt` parsed as a rigid transform, `SegRefine`, `Raw/T1`); (c) FreeSurfer 6 subject dir, reading only (`surf/`, `label/`), via NiBabel. |
| CLI | `pybiscuit run <input> <output> [--resume] [--dry-run]`: `--dry-run` prints the resolved config, adapter findings, and the stage plan with blocked stages marked. |

## Acceptance

- Unit tests on synthetic runs: signature changes on any input, param, or
  asset change; resume skips valid stages; corrupted outputs are detected;
  attempt numbering works.
- `@pytest.mark.reference` tests: the adapter finds every expected product in
  the PyMaCRUISE run above and in `~/dev/mipav/sample-macruise/output/`; the
  legacy `aff.txt` transform maps FreeView inner vertices onto the MNI ones
  with max residual ≤ 1e-4 mm (brief 01 measured 5e-6).
- `pybiscuit run --dry-run` on both reference inputs prints a full plan with
  blocked stages marked; exit code 0.
- No new runtime dependencies (TOML: `tomllib` on 3.11+, `tomli` on 3.10
  only if needed; justify it).

## Out of scope

Any stage science; output-tree naming for `legacy/` (B10.4, waits for B1.3).
