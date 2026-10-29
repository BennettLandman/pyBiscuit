# Pipeline infrastructure (config, stage graph, adapters) — port note

Rows: B10.2, B10.3 (machinery only). Modelled on PyMaCRUISE's restartable workflow
(`base_resume_workflow.py`, `run.json` schema_version 1), not a port of BISCUIT code:
the orchestrator is encrypted MATLAB; its stage semantics wait on the orchestrator spec
(brief 11, written from the MATLAB source Lyu supplied on 2026-10-21).

## Config (`pybiscuit.config`)
- Reads the original `biscuit.conf` (`key = value`, `%` comment lines); unknown keys are
  rejected. Defaults equal the FC image's file (`reference/.../biscuit.conf`); the writer
  reproduces the shipped layout (`tests/test_config.py::test_roundtrip_shipped_conf`).
- TOML: top-level keys or a `[biscuit]` table, same key names (`tomllib`; `tomli` marker
  dependency on Python 3.10 only, MIT).
- Precedence: defaults < `biscuit.conf` < TOML < CLI. CLI flags are the `run_biscuit`
  names (`CLI_FLAGS`): `--proj --subj --sess --thread --surf_recon --reg --parc --trace_r
  --trace_delta --curv_smooth --lgi_kernel --lgi_aniso --lgi_ref --sci_kernel --sci_bins`.
- `run_biscuit` quirks kept: `--surf_recon` is lower-cased; blanks are stripped from
  `parc_protocol`; the `fs_`/`mc_`/`cv_` blocks of the routes not selected are dropped
  (`Config.active_items`). Not kept: its `sed s/old/new/` substitution (a latent bug when
  the old value occurs elsewhere on the line); values are simply set.
- `run` takes `surf_recon` from the adapter (PyMaCRUISE / MaCRUISE v3 -> `macruise`,
  FreeSurfer subject -> `freesurfer`) and errors if an explicit setting contradicts it.

## Stage graph and signatures (`pybiscuit.pipeline`)
- `Stage(name, plan_rows, inputs, params, outputs, assets, routes, version, body, blocked)`.
  Inputs are adapter product names, `"*"` (all), or `"stage:output"`.
- Signature = SHA-256 of canonical JSON of {stage, stage version, pybiscuit version,
  input hashes, upstream stage signatures, declared config values, asset digests, output
  names}. Asset digests come from `data/asset_registry.json` (not from disk), so plans work
  without the 1.2 GB of assets installed. Thread count is deliberately not a parameter.
  The stored `components` dict lets a refused resume say what changed.
- `stages/<name>/attempt-NN/` (next number = max existing + 1, so failed attempts are kept)
  with `provenance.json`; `run.json` is written atomically after every stage change.
- `--resume`: refuses (before running anything) if input hashes differ from the recorded
  ones, a completed stage's signature differs, or a recorded output is missing/changed.
  It never recomputes silently. Without `--resume`, an existing `run.json` is an error.
- Stage registry (`stages.py`): `ingest` (records the adapter findings) then `surfprep`
  (macruise route only), `features`, `register`, `parc`, `trace`, `quant`, `stats`, `report`.
  `STAGES` keeps the blocked placeholders (`--blocked`, diagnostic); `get_executable_stages()`
  attaches the bodies of `bodies.py` (stage version 2 since 2026-10-28).
- Bodies (`bodies.py`, 2026-10-28) follow the orchestrator spec phase by phase and work in
  BISCUIT's own file names inside each attempt directory (`<h>.white.vtk`, `<h>.res.H.txt`,
  `<h>.coeff.txt`, `<h>.parc.<p>.txt`, `<h>.gray.scurve.bary`, `<h>.mid.H.txt`, …); a stage
  copies what it needs from the latest attempt of an upstream stage by name and raises
  `StageInputError` when a file is missing. They call the validated ports only:
  `prepare_fs_subject` (FS route, spec §12) or `run_macruise_surfprep` + MeshProperty
  features (spec §5.1), `SphericalRemesh` T18/T20/T22/T23, `run_chain` (T21), `parcellate`
  (§6), `curve_extract_pipeline` (§8), `quant_mid_features`, `sulcal_depth`, `lgi_pipeline`
  (klaplace correspondence), `shape_complexity_index` (§9), `run_stats_pipeline` (§10).
- **No placeholder results.** The earlier bodies caught every exception and wrote
  3-vertex meshes and zero maps, so a run reported `complete` whatever happened (several
  calls had wrong argument names and never ran). Any failure now propagates: the runner
  records it in `run.json` / `provenance.json` and `pybiscuit run` exits 1.
- MaCRUISE route, open point: `parc.m` reads `<h>.inflated.H` (FreeSurfer's curvature of the
  inflated surface); the body uses `iH` in its place until B5.6 defines that file.
- `--legacy` assembles `<out>/legacy/` from the latest attempt of every stage plus the run
  root (`pdf/`, `PDF/`) with `populate_legacy_tree`.
- End-to-end evidence: [e2e-open-examples-2026-10-28](../reports/e2e-open-examples-2026-10-28.md).

## Adapters (`pipeline/adapters/`)
| Adapter | Identifies products by | Checks |
|---|---|---|
| `pymacruise` | stage name + file name in `stages{}.outputs`; T1 from `derived_resources[role=t1]` (the RAS copy PyMaCRUISE worked on), else `inputs.t1` | `schema_version` 1, `status` complete, `converged`, required stages complete, SHA-256 of every consumed file vs `run.json`, VTK header `coordinate_frame=RAS units=mm`, volumes RAS and on one grid; moved run dirs re-rooted at `stages/`; records `run_signature`, `implementation_signature` (run.json has no PyMaCRUISE version) |
| `macruise-v3` | fixed MaCRUISE v3 tree (`Surfaces_FreeView/`, `Surfaces_MNI/`, `SegRefine/`, `Raw/`, `Surfaces/`) | `.asc` headers and native/MNI vertex+face counts; `aff.txt` is 4x4 and a proper rotation; T1/SEG_refine share a grid |
| `freesurfer` | `surf/{lh,rh}.*`, `label/*.annot`, `mri/*.mgz` via NiBabel, read-only | surfaces readable, white/pial vertex counts equal, tkRAS frame recorded with `c_ras` |

Finding: `Surfaces_MNI/aff.txt` maps **MNI -> native**; `native_to_mni_matrix` (its inverse)
maps the FreeView `.asc` vertices onto the MNI ones with max residual 5.0e-6 mm on the
sample.
